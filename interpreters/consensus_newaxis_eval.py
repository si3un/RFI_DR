"""
consensus_newaxis_eval.py — experiment adding **new axes** instead of further changing the model-aggregation function.

Swapping only the averaging function over the same 14 maps is considered converged, so three new axes are opened.

  1) ScaleMax-HybridMeanQ75   ← scale axis
     The current *_saliency_ms8 stores 1.0x and 1.5x **already averaged**
     (multiscale_sess: acc += sm; ms = acc/nsc), which can dilute small-lesion signals.
     Each per-scale map is rank-normalized, the **pixel-wise maximum** is taken and re-ranked →
     then HybridMeanQ75 across models.
     ※ 1.0x = --scale1_root and 1.5x = --scale2_root must be given **separately**.
       If no 1.5x-only root exists, this method is skipped automatically.

  2) RRAConsensus              ← statistical rank consensus axis
     Robust Rank Aggregation. At each pixel the K normalized ranks are flipped to p = 1 - r and
     sorted ascending; beta_k = I_{p_(k)}(k, K-k+1), rho = min_k beta_k,
     score = -log(rho). Emphasizes pixels that are "repeatedly near the top, more than by chance".
     No quantile needs to be chosen, so there is no tuning parameter.
     ※ The 14 models are not independent, so rho must not be read as statistical significance;
       it is used only as a rank aggregation score.

  3) Local-HybridMeanQ75       ← spatial tolerance axis
     The native saliency grid is 32x32 (14 px patches), so models may point to the same micro-lesion
     one patch apart. **Before** model fusion, a 3x3 native-grid local max (= 42 px at 448) is applied
     to each rank map, treating "within one patch" as the same location,
     followed by HybridMeanQ75.

Compared methods
  RankMean / HybridMeanQ75 (previous best) / the 3 above / [ref] Lineage RelBoost / 1 individual model

Fixed conditions: multi-scale saliency, AUROC>=auc_min pool, image-wise percentile rank,
          no lineage information (except the reference), no per-dataset or per-lesion tuning.

Usage:
  python interpreters/consensus_newaxis_eval.py --dataset idrid \
    --root $IDRID_SEG_ROOT \
    --sal_root outputs/idrid_saliency_ms8 --single_root outputs/idrid_saliency_attn \
    --scale1_root outputs/idrid_saliency_attn --scale2_root outputs/idrid_saliency_15only \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir outputs/idrid_newaxis --types MA HE EX SE
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from scipy.special import betainc
from scipy.ndimage import maximum_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
import fgadr_localization as F
import idrid_localization as ID
import consensus_compare as CC
from threshold_sweep_erx import DISCOVER, DEF_TYPES, load_set, best8_of, relboost
from final_localization_eval import sweep_dual, norm1


def hybrid_mean_q75(Rk, fov):
    """Cross-model fusion: 0.5*rank(RankMean) + 0.5*rank(Q75).  The most stable combination so far."""
    K = Rk.shape[0]
    srt = np.sort(Rk, axis=0)
    mean = Rk.mean(axis=0)
    pos = 0.75 * (K - 1); lo = int(np.floor(pos)); hi = min(lo + 1, K - 1); fr = pos - lo
    q75 = srt[lo] * (1.0 - fr) + srt[hi] * fr
    return norm1(0.5 * CC.rank_in_fov(mean, fov) + 0.5 * CC.rank_in_fov(q75, fov), fov)


def rra_score(Rk, fov):
    """Robust Rank Aggregation score = -log(min_k I_{p_(k)}(k, K-k+1)),  p = 1 - rank."""
    K = Rk.shape[0]
    P = np.sort(1.0 - Rk[:, fov], axis=0)
    ks = np.arange(1, K + 1, dtype=np.float64)[:, None]
    beta = betainc(ks, K - ks + 1.0, np.clip(P, 0.0, 1.0))
    rho = beta.min(axis=0)
    out = np.zeros(Rk.shape[1:], dtype=np.float64)
    out[fov] = -np.log(np.maximum(rho, 1e-300))
    return norm1(out, fov)


def local_max_ranks(Rk, fov, size):
    """Apply a local max equivalent to a 3x3 native grid to each rank map, then re-rank."""
    return np.stack([CC.rank_in_fov(maximum_filter(Rk[i], size=size) * fov, fov)
                     for i in range(Rk.shape[0])], axis=0)


def product_mean_q75(Rk, fov):
    """A. rank(RankMean) x rank(Q75) — AND gate.
    With an additive score, being high in one term is enough to survive; the product keeps only 'places supported by both'."""
    K = Rk.shape[0]
    srt = np.sort(Rk, axis=0)
    mean = Rk.mean(axis=0)
    pos = 0.75 * (K - 1); lo = int(np.floor(pos)); hi = min(lo + 1, K - 1); fr = pos - lo
    q75 = srt[lo] * (1.0 - fr) + srt[hi] * fr
    return norm1(CC.rank_in_fov(mean, fov) * CC.rank_in_fov(q75, fov), fov)


def max_mean_q75(Rk, fov):
    """MaxMeanQ75 (= UnionRank). Pixel-wise maximum after rank-normalizing each consensus **independently**.

    additive/product both act as 'mean or AND', so when M and Q are highly correlated the ordering
    barely changes (measured: Product vs Hybrid ordering correlation 0.999). max acts as OR, so
    evidence is preserved even if only one side is strong — broad lesions where RankMean is strong and
    focal lesions where Q75 is strong do not erode each other.
    ※ A max over raw values is dominated by the larger scale, so always rank-normalize before max.
    """
    K = Rk.shape[0]
    srt = np.sort(Rk, axis=0)
    mean = Rk.mean(axis=0)
    pos = 0.75 * (K - 1); lo = int(np.floor(pos)); hi = min(lo + 1, K - 1); fr = pos - lo
    q75 = srt[lo] * (1.0 - fr) + srt[hi] * fr
    M = CC.rank_in_fov(mean, fov)
    Q = CC.rank_in_fov(q75, fov)
    return norm1(np.maximum(M, Q), fov)


def otsu_vote_rank(stack_mm, Rk, fov):
    """C. Per-model Otsu foreground vote count + RankMean (tie-breaking).

    ※ Otsu is applied to the **value map (min-max)**. A rank map is uniformly distributed within the FOV,
      so Otsu always splits at ~50% and is not adaptive (verified empirically).
      On the value map it adapts to lesion size, at around 0.5-10%.
    V is an integer in 0..K and RankMean is in (0,1], so votes take priority and ties are resolved by mean saliency.
    """
    V = np.zeros(fov.shape, dtype=np.float64)
    for j in range(stack_mm.shape[0]):
        raw = stack_mm[j]
        v = raw[fov]
        mn, mx = float(v.min()), float(v.max())
        if mx <= mn:
            continue
        a01 = np.clip((raw - mn) / (mx - mn + 1e-12), 0, 1) * fov
        V += ((a01 > C.otsu(a01[fov])) & fov).astype(np.float64)
    return norm1(V + Rk.mean(axis=0), fov)


def mm_rank_stack(names, fn, cams, fidx, fov):
    """Return (stack of min-max value maps, stack of percentile ranks) together."""
    mm = np.stack([E.mm(cams[m][fidx[m][fn]].numpy()) * fov for m in names], axis=0)
    R = np.stack([CC.rank_in_fov(mm[j], fov) for j in range(len(names))], axis=0)
    return mm, R


def rank_stack(names, fn, cams, fidx, fov):
    return mm_rank_stack(names, fn, cams, fidx, fov)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DISCOVER))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sal_root", required=True, help="current multi-scale (=*_saliency_ms8)")
    ap.add_argument("--single_root", required=True, help="individual (1.0x = *_saliency_attn)")
    ap.add_argument("--scale1_root", default="", help="1.0x root for ScaleMax")
    ap.add_argument("--scale2_root", default="", help="1.5x-only root for ScaleMax")
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_min", type=float, default=0.92)
    ap.add_argument("--local_size", type=int, default=42,
                    help="local max neighbourhood size (px). 3x3 of the 448/32=14px patch = 42 (default)")
    ap.add_argument("--individuals", choices=["best", "none"], default="best")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    types = a.types or DEF_TYPES[a.dataset]
    items = DISCOVER[a.dataset](a.root, types)
    auc = SC.load_auc_map(a.model_roots)

    cn, ccams, cfidx = load_set(a.sal_root)
    cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
    clin = {m: C.lineage_of(m) for m in cn}; cft = {m: L.ft_of(m) for m in cn}
    POOL = [m for m in sorted(cn, key=lambda m: -auc.get(m, 0)) if auc.get(m, 0) >= a.auc_min]
    if len(POOL) < 2:
        sys.exit("[abort] AUROC>=%.2f pool has %d models." % (a.auc_min, len(POOL)))
    best8 = best8_of(cn, clin, cft, auc)
    lin_tag = "[ref] Lineage RelBoost @best8 (n=%d)" % len(best8)

    inames, icams, ifidx = load_set(a.single_root)
    inames = [m for m in inames if C.lineage_of(m) and L.ft_of(m)]
    inames = sorted(inames, key=lambda m: -(auc.get(m) or 0))
    inames = inames[:1] if a.individuals == "best" else []
    pretty = {m: "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in inames}

    # ── ScaleMax inputs ──
    s1 = s2 = None
    if a.scale1_root and a.scale2_root:
        if not (os.path.isdir(a.scale1_root) and os.path.isdir(a.scale2_root)):
            print("  [skip] scale root path missing → ScaleMax excluded")
        else:
            n1, c1, f1 = load_set(a.scale1_root)
            n2, c2, f2 = load_set(a.scale2_root)
            miss = [m for m in POOL if m not in f1 or m not in f2]
            if miss:
                print("  [skip] %d pool models missing from scale root → ScaleMax excluded: %s"
                      % (len(miss), miss[:3]))
            else:
                s1, s2 = (c1, f1), (c2, f2)
    else:
        print("  [skip] --scale1_root/--scale2_root not set → ScaleMax excluded")

    T_BASE = "RankMean @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_HYB = "HybridMeanQ75 @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_RRA = "RRAConsensus @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_LOC = "Local%dpx-HybridMeanQ75 (n=%d)" % (a.local_size, len(POOL))
    T_SCM = "ScaleMax-HybridMeanQ75 (n=%d)" % len(POOL)
    T_SCP = "MasterScaleProduct-HybridMeanQ75 (n=%d)" % len(POOL)
    T_PRD = "ProductMeanQ75 @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_OTV = "OtsuVoteRank @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_MAX = "MaxMeanQ75 @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_SMX2 = "ScaleMax-MaxMeanQ75 (n=%d)" % len(POOL)
    ORDER = [T_BASE, T_HYB, T_MAX, T_PRD, T_OTV, T_RRA, T_LOC] + \
            ([T_SCM, T_SCP, T_SMX2] if s1 else []) + ([lin_tag] if best8 else [])

    print("[%s] cases %d | pool n=%d | local=%dpx | ScaleMax=%s"
          % (a.dataset, len(items), len(POOL), a.local_size, "ON" if s1 else "OFF"))
    print("  evaluating: %s" % ", ".join(ORDER))

    fov_cache, rows, skipped = {}, [], 0
    for i, r in enumerate(items):
        fn, lt = r["image_id"], r["lesion_type"]
        if not all(fn in cfidx[m] for m in POOL) or not all(fn in ifidx[m] for m in inames):
            skipped += 1; continue
        if best8 and not all(fn in cfidx[m] for m in best8):
            skipped += 1; continue
        if s1 and (not all(fn in s1[1][m] for m in POOL) or not all(fn in s2[1][m] for m in POOL)):
            skipped += 1; continue
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        fov = fov_cache[fn]
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            skipped += 1; continue

        MM, Rk = mm_rank_stack(POOL, fn, ccams, cfidx, fov)
        maps = {T_BASE: norm1(Rk.mean(axis=0), fov),
                T_HYB: hybrid_mean_q75(Rk, fov),
                T_MAX: max_mean_q75(Rk, fov),
                T_PRD: product_mean_q75(Rk, fov),
                T_OTV: otsu_vote_rank(MM, Rk, fov),
                T_RRA: rra_score(Rk, fov),
                T_LOC: hybrid_mean_q75(local_max_ranks(Rk, fov, a.local_size), fov)}
        if s1:
            R1 = rank_stack(POOL, fn, s1[0], s1[1], fov)
            R2 = rank_stack(POOL, fn, s2[0], s2[1], fov)
            # OR: pixel-wise max of per-scale ranks → re-rank → Hybrid across models
            Rmx = np.stack([CC.rank_in_fov(np.maximum(R1[j], R2[j]), fov)
                            for j in range(len(POOL))], axis=0)
            maps[T_SCM] = hybrid_mean_q75(Rmx, fov)
            # scale OR + model OR: both stages preserve complementary evidence
            maps[T_SMX2] = max_mean_q75(Rmx, fov)
            # AND (Master-CAM style): 1.0x guides the location, 1.5x provides detail
            Rpr = np.stack([CC.rank_in_fov(R1[j] * R2[j], fov)
                            for j in range(len(POOL))], axis=0)
            maps[T_SCP] = hybrid_mean_q75(Rpr, fov)
        if best8:
            lc = relboost(fn, best8, clin, cft, ccams, cfidx, fov)
            if lc is None:
                skipped += 1; continue
            maps[lin_tag] = lc
        for m in inames:
            maps[pretty[m]] = E.mm(icams[m][ifidx[m][fn]].numpy()) * fov

        per = {k: sweep_dual(A, Lm, fov) for k, A in maps.items()}
        if any(x is None for x in per.values()):
            skipped += 1; continue
        for k, two in per.items():
            for ana, v in two.items():
                rows.append(dict(dataset=a.dataset, lesion_type=lt, image_id=fn,
                                 method=k, analysis=ana, **v))
        if (i + 1) % 200 == 0:
            print("  ... %d/%d" % (i + 1, len(items)))

    if not rows:
        sys.exit("[abort] no cases evaluated.")
    df = pd.DataFrame(rows)
    p0 = os.path.join(a.out_dir, "%s_newaxis_per_case.csv" % a.dataset)
    df.to_csv(p0, index=False)
    print("  evaluated case-items %d, excluded %d" % (df.groupby(["image_id", "lesion_type"]).ngroups, skipped))

    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)
    out = []
    for ana in ["primary", "sensitivity"]:
        d0 = df[df.analysis == ana]
        for lt in ["ALL"] + list(types):
            sub = d0 if lt == "ALL" else d0[d0.lesion_type == lt]
            if sub.empty:
                continue
            g = sub.groupby(["dataset", "method"], as_index=False).agg(
                N=("max_dice", "size"), Dice=("max_dice", "mean"), Dice_SD=("max_dice", "std"),
                IoU=("iou_at_best", "mean"), k_median=("best_topk_percent", "median"),
                k_Q1=("best_topk_percent", q1), k_Q3=("best_topk_percent", q3),
                pos_over_G=("pos_over_G", "median"), at_min=("at_grid_min", "mean"))
            g.insert(0, "analysis", ana); g.insert(1, "lesion_type", lt)
            out.append(g)
    S = pd.concat(out, ignore_index=True)
    se = S.Dice_SD / np.sqrt(S.N.clip(lower=1))
    S["Dice_lo"], S["Dice_hi"] = S.Dice - 1.96 * se, S.Dice + 1.96 * se
    for ana in ["primary", "sensitivity"]:
        S[S.analysis == ana].round(6).to_csv(
            os.path.join(a.out_dir, "%s_newaxis_%s.csv" % (a.dataset, ana)), index=False)
    S.round(6).to_csv(os.path.join(a.out_dir, "%s_newaxis_summary.csv" % a.dataset), index=False)

    cols = ["N", "Dice", "Dice_lo", "Dice_hi", "IoU", "k_median", "pos_over_G", "at_min"]
    for ana in ["primary", "sensitivity"]:
        for lt in ["ALL"] + list(types):
            s = S[(S.analysis == ana) & (S.lesion_type == lt)]
            if s.empty:
                continue
            keep = [m for m in ORDER if m in set(s.method)]
            t = s.set_index("method").reindex(keep)
            print("\n===== [%s] %s / %s (N=%d) =====" % (ana, a.dataset, lt, int(s.N.iloc[0])))
            print(t[cols].round(4).to_string())
            base = t.Dice.get(T_HYB, np.nan)
            if np.isfinite(base):
                d = (t.Dice - base) / base * 100
                print("  vs HybridMeanQ75 %%: " +
                      ", ".join("%s %+.1f%%" % (m.split(" @")[0].split(" (")[0], d[m])
                                for m in keep if m != T_HYB))
    print("\nSaved:", p0)


if __name__ == "__main__":
    main()
