"""
final_localization_eval.py — final saliency localization analysis (per-image optimal operating point).

Finalized settings
  1. Saliency map : 1.0x + 1.5x multi-scale fusion  (--sal_root = *_saliency_ms8)
  2. Consensus pool: classification AUROC >= threshold (default 0.92 -> 14 models, 0.95 -> 10 models)
  3. Consensus methods (per pool threshold):
       CentRelBoost      : centrality-weighted mean x (1 + majority-of-K agreement)  [existing]
       RankMean          : pixel-wise mean after converting to percentile ranks within the FOV
       TrimmedRankMean   : pixel-wise mean after dropping trim values at the top and bottom (default trim=2)
     Reference: Lineage RelBoost @best8 (existing method), individual models
     ※ The ClusterRankMean family is excluded (at corr>=0.8 the clusters split and it becomes identical to RankMean)
  4. Threshold sweep — both computed simultaneously **from the same sort**:
       primary     : top 1%, 2%, ..., 99%            (1-percentile increments)
       sensitivity : top 0.1%~0.9% (0.1 steps) + 1%~99%  (sub-1% added at uniform spacing)
     For each (image x lesion x method), choose the k with maximum Dice and record Dice/IoU at that point.
  5. Output: primary / sensitivity saved as **separate CSVs/tables** for direct comparison.
     Also reports the median/Q1/Q3 of the optimal k and pos_over_G — to check whether sub-1% is
     actually needed for tiny lesions (rate of hitting the grid lower bound, at_min).

Empty GT: if |L| = 0 after intersecting with the FOV, that (image x lesion) is excluded (existing rule kept).
N consistency: only cases evaluable by all methods are aggregated.

Usage:
  python interpreters/final_localization_eval.py --dataset idrid \
    --root $IDRID_SEG_ROOT \
    --sal_root outputs/idrid_saliency_ms8 --single_root outputs/idrid_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir outputs/idrid_final --types MA HE EX SE --auc_mins 0.92 0.95

  python interpreters/final_localization_eval.py --merge \
    outputs/idrid_final/idrid_final_per_case.csv \
    outputs/fgadr_final/fgadr_final_per_case.csv \
    outputs/eophtha_final/eophtha_final_per_case.csv \
    --out_dir outputs/final_tables_loc
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
import fgadr_localization as F
import idrid_localization as ID
import consensus_compare as CC
from threshold_sweep_erx import (DISCOVER, DEF_TYPES, load_set, best8_of,
                                 pool_variants, relboost)

PRIMARY = np.arange(1.0, 100.0, 1.0)                       # 1,2,...,99
SUB = np.round(np.arange(0.1, 1.0, 0.1), 1)                # 0.1,...,0.9
KS_ALL = np.unique(np.concatenate([SUB, PRIMARY]))
IS_PRIMARY = KS_ALL >= 1.0


def norm1(a, fov):
    x = a * fov
    s = x.sum()
    return x / s if s > 1e-12 else x


def sweep_dual(A, Lm, fov):
    """Compute the Dice maximum on both the primary and sensitivity grids with **a single sort**.

    Sort the FOV interior in descending order -> cumulative sum of GT membership -> intersection for each k with one lookup.
    k% -> rank-based top-N taking exactly N = round(k/100 * |FOV|) pixels (robust to ties).
    IoU = Dice/(2-Dice) is monotone, so the argmax differs only on ties.
    """
    if A is None:
        return None
    fi = np.flatnonzero(fov.ravel())
    n = fi.size
    Lf = Lm.ravel()[fi].astype(bool)
    G = int(Lf.sum())
    if G == 0 or n == 0:
        return None
    s = A.ravel()[fi]
    order = np.argsort(s, kind="stable")[::-1]
    csum = np.cumsum(Lf[order].astype(np.int64))

    Ns = np.maximum(1, np.rint(KS_ALL / 100.0 * n).astype(np.int64))
    Ns = np.minimum(Ns, n)
    inter = csum[Ns - 1].astype(np.float64)
    dice = 2.0 * inter / (Ns + G)
    union = Ns + G - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    out = {}
    for tag, mask in (("primary", IS_PRIMARY), ("sensitivity", np.ones_like(IS_PRIMARY))):
        idx = np.flatnonzero(mask)
        b = idx[int(np.argmax(dice[idx]))]
        out[tag] = dict(best_topk_percent=float(KS_ALL[b]),
                        max_dice=float(dice[b]),
                        iou_at_best=float(iou[b]),
                        n_pos=int(Ns[b]),
                        pos_over_G=float(Ns[b] / G),
                        # whether the optimal k sits at the grid's lower bound (if so, the grid is too coarse)
                        at_grid_min=int(b == idx[0]),
                        G=G, fov_px=int(n))
    return out


def build_methods(fn, fov, ctx):
    """Build all maps to evaluate for this image. {label: 2D map}"""
    (pools, tag_cent, tag_rank, tag_trim, best8, lin_tag, clin, cft,
     ccams, cfidx, inames, icams, ifidx, pretty, trim) = ctx
    maps = {}
    for t, pl in pools:
        v = pool_variants(fn, pl, ccams, cfidx, fov)
        if not v:
            return None
        maps[tag_cent[t]] = v["Cent+RelBoost"]
        # rank maps are built once and shared by RankMean / TrimmedRankMean
        R = np.stack([CC.rank_in_fov(E.mm(ccams[m][cfidx[m][fn]].numpy()) * fov, fov)
                      for m in pl], axis=0)
        K = R.shape[0]
        srt = np.sort(R, axis=0)
        tt = int(min(trim, (K - 1) // 2))
        maps[tag_rank[t]] = norm1(R.mean(axis=0), fov)
        maps[tag_trim[t]] = norm1(srt[tt:K - tt].mean(axis=0), fov)
    if best8:
        lc = relboost(fn, best8, clin, cft, ccams, cfidx, fov)
        if lc is None:
            return None
        maps[lin_tag] = lc
    for m in inames:
        maps[pretty[m]] = E.mm(icams[m][ifidx[m][fn]].numpy()) * fov
    return maps


def run_dataset(a):
    types = a.types or DEF_TYPES[a.dataset]
    items = DISCOVER[a.dataset](a.root, types)
    auc = SC.load_auc_map(a.model_roots)

    cn, ccams, cfidx = load_set(a.sal_root)
    cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
    clin = {m: C.lineage_of(m) for m in cn}; cft = {m: L.ft_of(m) for m in cn}
    order_c = sorted(cn, key=lambda m: -auc.get(m, 0))
    pools, seen = [], set()
    for t in sorted(a.auc_mins):
        pl = [m for m in order_c if auc.get(m, 0) >= t]
        if len(pl) < 2 or tuple(pl) in seen:
            if len(pl) >= 2:
                print("  [merged] AUROC>=%.2f gives the same pool as the previous one (n=%d)" % (t, len(pl)))
            continue
        seen.add(tuple(pl)); pools.append((t, pl))
    if not pools:
        sys.exit("[abort] consensus pool is empty. sal_root=%s" % a.sal_root)
    best8 = best8_of(cn, clin, cft, auc)
    lin_tag = "Lineage RelBoost @best8 (n=%d)" % len(best8)

    inames, icams, ifidx = load_set(a.single_root)
    inames = [m for m in inames if C.lineage_of(m) and L.ft_of(m)]
    inames = sorted(inames, key=lambda m: -(auc.get(m) or 0))
    if a.individuals == "best":
        inames = inames[:1]
    elif a.individuals == "none":
        inames = []
    pretty = {m: "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in inames}
    best_auc_single = pretty.get(inames[0]) if inames else None

    tag_cent = {t: "CentRelBoost @AUROC>=%.2f (n=%d)" % (t, len(pl)) for t, pl in pools}
    tag_rank = {t: "RankMean @AUROC>=%.2f (n=%d)" % (t, len(pl)) for t, pl in pools}
    tag_trim = {t: "TrimmedRankMean @AUROC>=%.2f (n=%d, trim=%d)" % (t, len(pl), a.trim)
                for t, pl in pools}
    ORDER = ([tag_cent[t] for t, _ in pools] + [tag_rank[t] for t, _ in pools]
             + [tag_trim[t] for t, _ in pools] + ([lin_tag] if best8 else []))

    print("[%s] cases %d | individual %d | multi-scale=%s" % (a.dataset, len(items), len(inames),
                                                          os.path.basename(a.sal_root)))
    for t, pl in pools:
        print("  AUROC>=%.2f pool n=%d" % (t, len(pl)))
    if best8:
        print("  %s : %s" % (lin_tag, [("%s-%s" % (L.backbone_pretty(m), cft[m])) for m in best8]))
    print("  sweep: primary %d (1~99%%) | sensitivity %d (adds 0.1~0.9%%)"
          % (IS_PRIMARY.sum(), len(KS_ALL)))

    need = sorted({m for _, pl in pools for m in pl} | set(best8))
    ctx = (pools, tag_cent, tag_rank, tag_trim, best8, lin_tag, clin, cft,
           ccams, cfidx, inames, icams, ifidx, pretty, a.trim)
    fov_cache, rows, skipped = {}, [], 0
    for i, r in enumerate(items):
        fn, lt = r["image_id"], r["lesion_type"]
        if not all(fn in cfidx[m] for m in need) or not all(fn in ifidx[m] for m in inames):
            skipped += 1; continue
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        fov = fov_cache[fn]
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            skipped += 1; continue
        maps = build_methods(fn, fov, ctx)
        if maps is None:
            skipped += 1; continue
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
    print("  evaluated case-items %d, excluded %d" % (df.groupby(["image_id", "lesion_type"]).ngroups, skipped))
    return df, types, ORDER, best_auc_single


def summarize(df, types):
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
                IoU=("iou_at_best", "mean"),
                k_median=("best_topk_percent", "median"),
                k_Q1=("best_topk_percent", q1), k_Q3=("best_topk_percent", q3),
                pos_over_G=("pos_over_G", "median"),
                at_min=("at_grid_min", "mean"))
            g.insert(0, "analysis", ana); g.insert(1, "lesion_type", lt)
            out.append(g)
    S = pd.concat(out, ignore_index=True)
    se = S.Dice_SD / np.sqrt(S.N.clip(lower=1))
    S["Dice_lo"], S["Dice_hi"] = S.Dice - 1.96 * se, S.Dice + 1.96 * se
    return S


def report(S, ORDER, best_single, types):
    show = ["N", "Dice", "Dice_lo", "Dice_hi", "IoU", "k_median", "k_Q1", "k_Q3",
            "pos_over_G", "at_min"]
    for ana in ["primary", "sensitivity"]:
        for lt in ["ALL"] + list(types):
            s = S[(S.analysis == ana) & (S.lesion_type == lt)]
            if s.empty:
                continue
            keep = [m for m in ORDER if m in set(s.method)]
            if best_single and best_single in set(s.method):
                keep = keep + [best_single]
            t = s.set_index("method").reindex(keep)
            print("\n===== [%s] %s / %s (N=%d) =====" % (ana, s.dataset.iloc[0], lt, int(s.N.iloc[0])))
            print(t[show].round(4).to_string())
            rk = s.sort_values("Dice", ascending=False).reset_index(drop=True)
            for m in keep:
                w = rk.index[rk.method == m]
                if len(w):
                    print("   %-46s rank among %d: %d" % (m, len(rk), int(w[0]) + 1))
    # Direct comparison: primary vs sensitivity
    p = S[S.analysis == "primary"].set_index(["lesion_type", "method"])
    q = S[S.analysis == "sensitivity"].set_index(["lesion_type", "method"])
    j = p[["Dice", "k_median", "at_min"]].join(
        q[["Dice", "k_median", "at_min"]], lsuffix="_pri", rsuffix="_sen").dropna()
    j["ΔDice"] = j.Dice_sen - j.Dice_pri
    j["gain%"] = 100 * j["ΔDice"] / j.Dice_pri.replace(0, np.nan)
    print("\n===== effect of adding sub-1% (per lesion, sorted by ΔDice) =====")
    print(j.sort_values("ΔDice", ascending=False).head(25).round(4).to_string())
    print("\n  Large at_min_pri and large gain%% mean the 1%% grid was too coarse for that lesion.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--dataset", choices=list(DISCOVER))
    ap.add_argument("--root")
    ap.add_argument("--sal_root", help="1.0x+1.5x multi-scale (=*_saliency_ms8)")
    ap.add_argument("--single_root", help="individual models (1.0x = *_saliency_attn)")
    ap.add_argument("--model_roots", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_mins", nargs="+", type=float, default=[0.92, 0.95])
    ap.add_argument("--trim", type=int, default=2)
    ap.add_argument("--individuals", choices=["all", "best", "none"], default="all",
                    help="all=all individuals (ranking context) | best=highest-AUROC model only | none")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.merge:
        df = pd.concat([pd.read_csv(p) for p in a.merge], ignore_index=True)
        df["dataset"] = "POOLED"
        types = sorted(df.lesion_type.unique())
        S = summarize(df, types)
        S.round(6).to_csv(os.path.join(a.out_dir, "pooled_summary.csv"), index=False)
        for ana in ["primary", "sensitivity"]:
            S[S.analysis == ana].round(6).to_csv(
                os.path.join(a.out_dir, "pooled_%s.csv" % ana), index=False)
        cons = [m for m in df.method.unique()
                if any(m.startswith(x) for x in ("CentRelBoost", "RankMean",
                                                 "TrimmedRankMean", "Lineage"))]
        report(S, sorted(cons), None, types)
        print("\nSaved:", a.out_dir)
        return

    for req in ("dataset", "root", "sal_root", "single_root", "model_roots"):
        if getattr(a, req) is None:
            sys.exit("[abort] --%s is required (or --merge)." % req)

    df, types, ORDER, best_single = run_dataset(a)
    p0 = os.path.join(a.out_dir, "%s_final_per_case.csv" % a.dataset)
    df.to_csv(p0, index=False)
    S = summarize(df, types)
    for ana in ["primary", "sensitivity"]:
        p = os.path.join(a.out_dir, "%s_final_%s.csv" % (a.dataset, ana))
        S[S.analysis == ana].round(6).to_csv(p, index=False)
    S.round(6).to_csv(os.path.join(a.out_dir, "%s_final_summary.csv" % a.dataset), index=False)
    report(S, ORDER, best_single, types)
    print("\nSaved:", p0)
    print("Saved:", os.path.join(a.out_dir, "%s_final_{primary,sensitivity}.csv" % a.dataset))


if __name__ == "__main__":
    main()
