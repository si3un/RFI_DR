"""
consensus_compare.py — comparison of 5 consensus fusion schemes (simple mean → agreement-aware).

5-row table (per lesion_type):
  1) Primary      = 14 models (Partial+LoRA, AUC≥0.9), lineage-balanced mean
  2) Lineage      = 8 models (best Partial + best LoRA per lineage), mean (sensitivity analysis)
  3) Agreement-Gated (CAG) = M(x)·A(x)   [main proposal]
  4) Geometric    = (R1·R2·R3·R4)^(1/4)  [soft intersection]
  5) Soft-Voting  = M(x)·V(x), V=¼Σ I(Rk>q)  [q=0.9, sensitivity analysis]

Methods 3-5 build 4 lineage-level maps (L1..L4) from the 8 models, convert each map to a **percentile rank within the FOV**
Rk(x)∈[0,1], then fuse. Ranks are used because saliency distributions (peaked/diffuse) differ across models,
so "what top percentage this pixel is in for this lineage" is fairer than raw intensity for computing agreement.

Metrics, preprocessing and bootstrap are reused from eophtha_localization (E) / snu_localization (L).

Usage:
  python interpreters/consensus_compare.py --dataset fgadr \
    --root $FGADR_SEG_ROOT \
    --sal_root outputs/fgadr_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir outputs/fgadr_consensus_compare --types MA HE EX SE
"""
import os, sys, glob, argparse
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.ndimage import binary_dilation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
import fgadr_localization as F
import idrid_localization as ID

DISCOVER = {"eophtha": E.discover, "fgadr": F.discover, "idrid": ID.discover}
DEFAULT_TYPES = {"eophtha": ["EX", "MA"], "fgadr": ["MA", "HE", "EX", "SE"],
                 "idrid": ["MA", "HE", "EX", "SE"]}
METHOD_ORDER = ["Primary", "Lineage", "Agreement-Gated", "Geometric",
                "Soft-Voting", "Soft-Voting-Tol", "Second-2of4", "Third-3of4",
                "Median", "Continuous-2of4", "AgreeBoost-2of4", "RelBoost-2of4"]
RANK_METHODS = {"Agreement-Gated", "Geometric", "Soft-Voting", "Soft-Voting-Tol",
                "Second-2of4", "Third-3of4", "Median", "Continuous-2of4",
                "AgreeBoost-2of4", "RelBoost-2of4"}


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r)


def norm1(a, fov):
    a = a * fov
    s = a.sum()
    return a / s if s > 1e-12 else a


def lineage_maps(fn, mods, lin, ft, cams, fidx):
    """model set → per-lineage **FT-balanced** map dict {lineage: map}.
    Within a lineage: mean per FT (Partial/LoRA) → equal weight per FT (same as the Primary scheme).
    The 8-set (1P+1L per lineage) is unaffected; for the 14-set this prevents FT imbalance."""
    by = {}
    for m in mods:
        if fn not in fidx[m]:
            return None
        by.setdefault(lin[m], {}).setdefault(ft[m], []).append(L.prep(cams[m][fidx[m][fn]].numpy()))
    out = {}
    for g, ftd in by.items():
        out[g] = np.mean([np.mean(v, axis=0) for v in ftd.values()], axis=0)
    return out


def rank_in_fov(m, fov):
    r = np.zeros(m.shape, dtype=np.float64)
    vals = m[fov]
    if vals.size:
        r[fov] = rankdata(vals, method="average") / vals.size   # (0,1]
    return r


def cag_from_ranks(R, M, fov):
    A = 1.0 - np.mean([np.abs(R[i] - R[j]) for i, j in combinations(range(len(R)), 2)], axis=0)
    return norm1(M * A, fov)


def geometric_from_ranks(R, fov, eps=1e-6):
    logsum = np.zeros(R[0].shape)
    for r in R:
        logsum += np.log(r + eps)
    return norm1(np.exp(logsum / len(R)), fov)


def softvote_from_ranks(R, M, fov, q=0.9):
    V = np.mean([(r > q).astype(np.float64) for r in R], axis=0)
    return norm1(M * V, fov)


def softvote_tol_from_ranks(R, M, fov, q=0.9, radius=11):
    """Spatially-tolerant voting: dilate each lineage's top-q mask with a disk (≈max-pool), then vote.
    Agreement is credited for 'agreement within a neighbourhood', not only the exact same pixel → protects small/scattered lesions (MA/HE)."""
    st = _disk(radius)
    V = np.mean([binary_dilation(r > q, structure=st).astype(np.float64) for r in R], axis=0)
    return norm1(M * V, fov)


def second_highest_from_ranks(R, fov):
    """C(x)=R_(2)(x): second-highest of the 4 lineage ranks = soft 'at least 2 lineages agree'.
    No threshold or radius needed. Places where only one model is high die out; overlaps of 2+ survive strongly."""
    Rs = np.sort(np.stack(R, axis=0), axis=0)     # ascending, [-1]=max
    c = Rs[-2] if len(R) >= 2 else Rs[-1]
    return norm1(c, fov)


def third_3of4_from_ranks(R, fov):
    """C(x)=R_(3): third-highest rank = 'supported by at least 3 lineages'. 3-of-4 version of Second-2of4 (r₂)."""
    Rs = np.sort(np.stack(R, axis=0), axis=0)
    c = Rs[-3] if len(R) >= 3 else Rs[0]
    return norm1(c, fov)


def median_from_ranks(R, fov):
    """C(x)=median(R1..R4)=(R_(2)+R_(3))/2: more conservative than Second-highest (requires 2-3 to agree)."""
    return norm1(np.median(np.stack(R, axis=0), axis=0), fov)


def continuous_2of4_from_ranks(R, M, fov):
    """C(x)=√(M·r₂): between Second-2of4 and Soft-Voting. No threshold, radius or λ.
    Only 1 high → r₂↓ suppressed / 2 agree → r₂↑ preserved (HE) / 3-4 → M↑ too, reinforced (EX/SE). Expected to recover intensity concentration."""
    second = np.sort(np.stack(R, axis=0), axis=0)[-2] if len(R) >= 2 else M
    return norm1(np.sqrt(np.clip(M * second, 0, None)), fov)


def agreement_boosted_from_ranks(raw_list, R, fov):
    """C(x)=M(x)·(1+A₂), M = arithmetic mean of raw lineage saliency (preserves intensity), A₂=R_(2) (agreement).
    Raw saliency is not 'replaced' by ranks but only boosted by agreement → preserves In/Out and Otsu while strengthening agreement.
    parameter-free (1+A, no λ)."""
    base = np.mean(np.stack(raw_list, axis=0), axis=0)
    A2 = np.sort(np.stack(R, axis=0), axis=0)[-2] if len(R) >= 2 else np.zeros_like(base)
    return norm1(base * (1.0 + A2), fov)


def relboost_from_ranks(raw_list, R, fov):
    """Reliability(cross-lineage centrality)-weighted AgreeBoost (GT-free).
    B(x)=Σ w_k L_k, w_k ∝ G_k = mean rank correlation with the other lineages (= Spearman centrality; clip≥0).
    C(x)=B(x)·(1+A₂).  → more trust in stable lineages supported by other FMs (instead of a flat 25%).
    ※ The stability term (S_k) needs flip-augmented generation → centrality-only here."""
    n = len(R)
    if n > 1:
        Cm = np.nan_to_num(np.corrcoef(np.stack([r[fov] for r in R])))   # rank → Spearman
        G = np.array([(Cm[k].sum() - 1.0) / (n - 1) for k in range(n)])
    else:
        G = np.array([1.0])
    w = np.clip(G, 0, None); sw = w.sum()
    w = w / sw if sw > 0 else np.ones(n) / n
    B = np.tensordot(w, np.stack(raw_list, axis=0), axes=1)
    A2 = np.sort(np.stack(R, axis=0), axis=0)[-2] if n >= 2 else np.zeros_like(B)
    return norm1(B * (1.0 + A2), fov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DISCOVER))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sal_root", required=True)
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_min", type=float, default=0.9)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--topq", type=float, default=0.9, help="Soft-Voting rank threshold (default 0.9=top10%)")
    ap.add_argument("--tol_radius", type=int, default=11, help="Soft-Voting-Tol dilation radius in px (~1 patch; at 448)")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="select which methods to compute (default all). e.g. --methods Second-2of4 Median")
    ap.add_argument("--set", dest="mset", choices=["lineage", "primary"], default="lineage",
                    help="source of lineage maps for rank-based methods: lineage=8-set (best P+L per lineage) | "
                         "primary=14-set (all P+L, FT-balanced). Formulas (RelBoost etc.) are unchanged.")
    ap.add_argument("--lock_best8", nargs="*", default=None,
                    help="fix Lineage(8) membership to this list (overrides AUC-based auto-selection). "
                         "use for mild evaluation with the same set as main — compares pure feature differences only.")
    ap.add_argument("--lock_prim", nargs="*", default=None,
                    help="fix Primary(14) membership to this list (overrides AUC-based auto-selection).")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    types = a.types or DEFAULT_TYPES[a.dataset]
    active = set(a.methods) if a.methods else set(METHOD_ORDER)
    bad = active - set(METHOD_ORDER)
    if bad:
        sys.exit(f"[abort] unknown method: {bad}\n  available: {METHOD_ORDER}")

    items = DISCOVER[a.dataset](a.root, types)
    if not items:
        sys.exit(f"[abort] 0 (image, mask) pairs — check path: {a.root}")
    print(f"[{a.dataset}] lesion cases {len(items)} | by type:",
          {t: sum(r['lesion_type'] == t for r in items) for t in types})

    # Load models + Primary(14)/Lineage(8) sets
    names = sorted([os.path.basename(d) for d in glob.glob(os.path.join(a.sal_root, "*"))
                    if os.path.exists(os.path.join(d, "saliency_map.pth"))])
    auc = SC.load_auc_map(a.model_roots)
    cams, fidx, lin, ft = {}, {}, {}, {}
    for m in names:
        cam, f2i = C.load_saliency(os.path.join(a.sal_root, m))
        cams[m] = cam; fidx[m] = f2i
        lin[m] = C.lineage_of(m); ft[m] = L.ft_of(m)
    names = [m for m in names if lin[m] and ft[m]]
    prim = [m for m in names if ft[m] in ("Partial", "LoRA") and auc.get(m, 0) >= a.auc_min]
    best8 = []
    for g in ["CLIP", "DINOv2", "DINOv3", "MAE"]:
        for f in ("Partial", "LoRA"):
            cand = [(auc.get(m, np.nan), m) for m in prim if lin[m] == g and ft[m] == f]
            cand = [(x, m) for x, m in cand if np.isfinite(x)]
            if cand:
                best8.append(max(cand)[1])
    def _lock(req, cur, tag):
        keep = [m for m in req if m in names]
        miss = [m for m in req if m not in names]
        if miss:
            print(f"[warning] --lock_{tag} entries without saliency (ignored): {miss}")
        print(f"[lock] {tag} membership fixed: {len(keep)} (auto-selection overridden)")
        return keep
    if a.lock_best8 is not None:
        best8 = _lock(a.lock_best8, best8, "best8")
    if a.lock_prim is not None:
        prim = _lock(a.lock_prim, prim, "prim")
    rank_src = best8 if a.mset == "lineage" else prim   # lineage map source for rank-based methods
    print(f"models {len(names)} | Primary {len(prim)} | Lineage(8) {len(best8)} | "
          f"rank_set={a.mset}({len(rank_src)} models) | topq={a.topq} | tol_radius={a.tol_radius}px")
    print(f"  Lineage(8) members: {best8}")
    print(f"  Primary({len(prim)}) members: {prim}")

    per, grid_acc = [], {}

    def record(A, tag, fn, ltype, Lm, fov):
        met = L.metrics(A, Lm, fov)
        if not met:
            return
        dg, ig = E.dice_iom_sweep(A, Lm, fov)
        rec = dict(image_id=fn, lesion_type=ltype, map_name=tag, **met)
        if dg is not None and np.isfinite(dg).any():
            bi = int(np.nanargmax(dg)); fb = float(E.FGRID[bi])
            rec["dice_best"] = float(dg[bi]); rec["f_best"] = fb
            rec["iom_best"] = np.nan if fb >= 0.999 else float(ig[bi])
            grid_acc.setdefault((tag, ltype), []).append((dg, ig))
            grid_acc.setdefault((tag, "ALL"), []).append((dg, ig))
        else:
            rec["dice_best"] = np.nan; rec["iom_best"] = np.nan; rec["f_best"] = np.nan
        per.append(rec)

    fov_cache = {}
    excl = 0
    for n_i, r in enumerate(items):
        fn = r["image_id"]; ltype = r["lesion_type"]
        if not all(fn in fidx[m] for m in prim):
            excl += 1; continue
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        fov = fov_cache[fn]
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            excl += 1; continue

        # 1) Primary(14 avg), 2) Lineage(8 avg)
        if "Primary" in active:
            cP = L.build_consensus(fn, prim, lin, ft, cams, fidx)
            if cP is not None:
                record(cP, "Primary", fn, ltype, Lm, fov)
        if "Lineage" in active:
            cL = L.build_consensus(fn, best8, lin, ft, cams, fidx)
            if cL is not None:
                record(cL, "Lineage", fn, ltype, Lm, fov)

        # 3~8) 8-set → 4 lineage maps → rank → fusion (only if at least one rank-based method is active)
        if active & RANK_METHODS:
            lmaps = lineage_maps(fn, rank_src, lin, ft, cams, fidx)
            if lmaps and len(lmaps) >= 2:
                gk = sorted(lmaps)
                raw_list = [lmaps[g] for g in gk]          # raw lineage saliency (to preserve intensity)
                R = [rank_in_fov(lmaps[g], fov) for g in gk]
                M = np.mean(R, axis=0)
                if "Agreement-Gated" in active:
                    record(cag_from_ranks(R, M, fov), "Agreement-Gated", fn, ltype, Lm, fov)
                if "Geometric" in active:
                    record(geometric_from_ranks(R, fov), "Geometric", fn, ltype, Lm, fov)
                if "Soft-Voting" in active:
                    record(softvote_from_ranks(R, M, fov, a.topq), "Soft-Voting", fn, ltype, Lm, fov)
                if "Soft-Voting-Tol" in active:
                    record(softvote_tol_from_ranks(R, M, fov, a.topq, a.tol_radius), "Soft-Voting-Tol", fn, ltype, Lm, fov)
                if "Second-2of4" in active:
                    record(second_highest_from_ranks(R, fov), "Second-2of4", fn, ltype, Lm, fov)
                if "Third-3of4" in active:
                    record(third_3of4_from_ranks(R, fov), "Third-3of4", fn, ltype, Lm, fov)
                if "Median" in active:
                    record(median_from_ranks(R, fov), "Median", fn, ltype, Lm, fov)
                if "Continuous-2of4" in active:
                    record(continuous_2of4_from_ranks(R, M, fov), "Continuous-2of4", fn, ltype, Lm, fov)
                if "AgreeBoost-2of4" in active:
                    record(agreement_boosted_from_ranks(raw_list, R, fov), "AgreeBoost-2of4", fn, ltype, Lm, fov)
                if "RelBoost-2of4" in active:
                    record(relboost_from_ranks(raw_list, R, fov), "RelBoost-2of4", fn, ltype, Lm, fov)

        if (n_i + 1) % 500 == 0:
            print(f"  ...{n_i+1}/{len(items)}", flush=True)

    per = pd.DataFrame(per)
    per.to_csv(os.path.join(a.out_dir, f"{a.dataset}_consensus_per_image.csv"), index=False)
    print(f"valid cases (Primary) {int((per['map_name']=='Primary').sum())} | excluded {excl}")

    # Summary: (method × lesion_type) + ALL
    rows = []
    for tag in METHOD_ORDER:
        if tag not in per["map_name"].values:
            continue
        sub_all = per[per.map_name == tag]
        for t in types + ["ALL"]:
            sub = sub_all if t == "ALL" else sub_all[sub_all.lesion_type == t]
            if len(sub):
                rows.append(dict(method=tag, lesion_type=t,
                                 **E.summarize(sub, grid_acc.get((tag, t), []), a.nboot)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, f"{a.dataset}_consensus_compare.csv"), index=False)

    cols = ["method", "N", "Dice_global", "IoM_global", "f_star", "Dice_oracle",
            "PG_pct", "AUPRC_mean", "nAUPRC_mean", "otsu_iou", "inout_geomean"]
    ord_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    for t in types + ["ALL"]:
        sub = out[out.lesion_type == t].copy()
        if not len(sub):
            continue
        sub = sub.sort_values("method", key=lambda s: s.map(ord_map))
        print(f"\n===== Consensus scheme comparison ({t}) =====")
        print(sub[cols].to_string(index=False))

    print("\nSaved:", a.out_dir, f"({a.dataset}_consensus_compare.csv, {a.dataset}_consensus_per_image.csv)")


if __name__ == "__main__":
    main()
