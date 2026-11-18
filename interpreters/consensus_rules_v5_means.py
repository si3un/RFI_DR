# -*- coding: utf-8 -*-
"""
consensus_rules_v5_means.py -- 2026-09-30 lab-meeting follow-up.

Questions raised at the meeting and what this script answers
  * "why the geometric mean and not the arithmetic (or harmonic) mean?"      -> 3 means
  * "min-max values carry a different meaning in each map; use ranks if no threshold may be used"
                                                                              -> 3 per-map normalisations
  * "dropping some values looks like a hack; use a fixed, generic rule such as Olympic scoring
     (discard the highest and the lowest)"                                    -> symmetric and one-sided trimming
  * "were there zeros that break the geometric mean?"                        -> zero diagnostics

Default design: 3 normalisations x 3 means x 3 symmetric trimming levels = 27 rules, plus the current manuscript rule.
  normalisation (per map, inside the FOV, no labels, no thresholds)              --norms
     MinMax : the stored saliency map, min-max scaled to [0, 1] (E.mm)
     UnitSum: the min-max map divided by its own sum over the FOV pixels (each map sums to 1; spatial, per model)
     Rank   : percentile rank of each pixel inside the FOV, (0, 1], ties averaged (rank_in_fov)
     SumOnly: optional (--norms SumOnly): the stored map (ReLU only, minimum NOT subtracted) divided by its FOV sum;
              differs from UnitSum only by the min-max step's subtraction of each map's FOV minimum
  mean (per pixel, across the pool maps that remain after trimming)              --means
     Arith  : sum / n
     Geo    : exp(mean(log x))
     Harm   : n / sum(1 / x)
  trimming (per pixel; values sorted across the pool maps)                       --trims
     T0     : none                                   (14 values)
     T1x1   : drop the highest 1 and the lowest 1   (12 values)   symmetric, Olympic scoring
     T2x2   : drop the highest 2 and the lowest 2   (10 values)
     L1 L2 L4 : drop only the lowest 1 / 2 / 4      (one-sided: guards against a member that misses a region)
     U1 U2 U4 : drop only the highest 1 / 2 / 4     (one-sided: guards against a member that spikes alone)
  reference (always added)
     Current: UnitSum-Geo-Low4 : manuscript rule (unit-sum maps, drop the 4 lowest per pixel, geometric mean)
                                 = UnitSum-Geo-L4, kept under its own name as the sanity check

For Geo and Harm a fixed floor (applied to every value, always, not as a fallback) keeps log(0) and 1/0 finite:
`--eps` (default 1e-6) for the [0, 1]-scaled MinMax and Rank values, and the manuscript's original 1e-12 for UnitSum
values (which are ~1e-6 per pixel), so that UnitSum-Geo-T0 and the reference rule reproduce the earlier values exactly.
No lesion mask is used to build any map; masks are used only to score them.

--shard i/n evaluates only the images whose position in the sorted image list is i modulo n (all lesion-type cases of
an image stay together) and writes {ds}_shard{i}of{n}_*.csv; run n shards in parallel and read the per-case files
together (consensus_rules_stats.py takes globs).

Outputs (out_dir):
  {ds}[_shard]_rules_per_case.csv     per case (image x lesion type) and rule: per-case optimal Dice
  {ds}[_shard]_rules_summary.csv      mean per rule
  {ds}[_shard]_zero_diag.csv          per image: share of FOV pixels at which >= j pool maps are exactly zero (MinMax)
  {ds}[_shard]_zero_diag_by_model.csv per model: mean share of FOV pixels that are exactly zero

Usage (SNU, run from RFI-DR-clean):
  python interpreters/consensus_rules_v5_means.py --dataset idrid \
    --root $IDRID_SEG_ROOT \
    --sal_root outputs/idrid_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux models/_stage \
    --out_dir outputs/consensus_rules_v5_means
  one-sided trimming only, unit-sum maps, FGADR in 8 parallel shards:
    ... --norms UnitSum --trims L1 L2 L4 U1 U2 U4 --shard 0/8   (and 1/8 ... 7/8)
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
from threshold_sweep_erx import DISCOVER, DEF_TYPES, load_set
from optimal_threshold_eval import sweep_best
from consensus_newaxis_eval import mm_rank_stack
from threshold_optimal_k import make_grid

NORMS = ("MinMax", "UnitSum", "Rank")
OPTIONAL_NORMS = ("SumOnly",)   # SumOnly: stored map (ReLU, no min subtraction) divided by its own FOV sum
EPS_UNITSUM = 1e-12
MEANS = ("Arith", "Geo", "Harm")
TRIMS = {"T0": (0, 0), "T1x1": (1, 1), "T2x2": (2, 2),
         "L1": (1, 0), "L2": (2, 0), "L4": (4, 0),
         "U1": (0, 1), "U2": (0, 2), "U4": (0, 4)}           # name -> (dropped lowest, dropped highest)
DEFAULT_TRIMS = ("T0", "T1x1", "T2x2")
REF_NAME = "Current: UnitSum-Geo-Low4"


def norm1(a, fov):
    x = a * fov
    s = x.sum()
    return x / s if s > 1e-12 else x


def combine(X, mean, lo, hi, eps):
    """X: K x N values (models on axis 0). Drop the `lo` lowest and the `hi` highest values at each pixel,
    then take the requested mean over the remaining K - lo - hi values."""
    K = X.shape[0]
    S = np.sort(X, axis=0)[lo:K - hi] if (lo or hi) else X
    if mean == "Arith":
        return S.mean(0)
    Sf = np.maximum(S, eps)
    if mean == "Geo":
        return np.exp(np.log(Sf).mean(0))
    if mean == "Harm":
        return S.shape[0] / (1.0 / Sf).sum(0)
    raise ValueError(mean)


def rule_names(norms=NORMS, means=MEANS, trims=DEFAULT_TRIMS):
    return ["%s-%s-%s" % (n, m, t) for n in norms for m in means for t in trims] + [REF_NAME]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DISCOVER))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sal_root", required=True, help="1.0x single-scale maps, e.g. outputs/idrid_saliency_attn")
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs/consensus_rules_v5_means")
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_min", type=float, default=0.92)
    ap.add_argument("--grid", default="ultrafine")
    ap.add_argument("--eps", type=float, default=1e-6, help="fixed floor for Geo / Harm on [0, 1]-scaled values")
    ap.add_argument("--norms", nargs="+", default=list(NORMS), choices=list(NORMS) + list(OPTIONAL_NORMS))
    ap.add_argument("--means", nargs="+", default=list(MEANS), choices=list(MEANS))
    ap.add_argument("--trims", nargs="+", default=list(DEFAULT_TRIMS), choices=list(TRIMS))
    ap.add_argument("--shard", default="", help="i/n: evaluate only every n-th image starting at i (0-based)")
    ap.add_argument("--max_cases", type=int, default=0, help="smoke test: evaluate only the first n cases")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    KS = make_grid(a.grid, 0.01, 99.0, 0.01)
    types = a.types or DEF_TYPES[a.dataset]
    items = sorted(DISCOVER[a.dataset](a.root, types), key=lambda r: r["image_id"])
    tag = a.dataset
    if a.shard:
        si, sn = (int(v) for v in a.shard.split("/"))
        ids = sorted({r["image_id"] for r in items})
        keep = {f for j, f in enumerate(ids) if j % sn == si}
        items = [r for r in items if r["image_id"] in keep]
        tag = "%s_shard%dof%d" % (a.dataset, si, sn)
    if a.max_cases:
        items = items[:a.max_cases]
    auc = SC.load_auc_map(a.model_roots)

    cn, ccams, cfidx = load_set(a.sal_root)
    cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
    POOL = [m for m in sorted(cn, key=lambda m: -auc.get(m, 0)) if auc.get(m, 0) >= a.auc_min]
    K = len(POOL)
    tags = ["%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in POOL]
    print("[%s] cases %d | pool %d (AUROC >= %.2f) | rules %d | eps %g" % (
        tag, len(items), K, a.auc_min, len(rule_names(a.norms, a.means, a.trims)), a.eps), flush=True)
    print("pool:", ", ".join(tags), flush=True)
    if K != 14:
        print("[note] pool does not have 14 models (%d). Check sal_root / model_roots." % K)

    ipath = {r["image_id"]: r["image_path"] for r in items}
    fov_cache = {}

    def fov_of(f):
        if f not in fov_cache:
            fov_cache[f] = C.FOV & E.fundus_mask_448(E.letterbox_img(ipath[f]))
        return fov_cache[f]

    rows, zrows, mzero, floor, skipped = [], [], [], [], 0
    cur_fn, cur_maps = None, None
    for i, r in enumerate(items):
        fn, lt = r["image_id"], r["lesion_type"]
        if not all(fn in cfidx[m] for m in POOL):
            skipped += 1; continue
        fov = fov_of(fn)
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            skipped += 1; continue
        if fn != cur_fn:
            fi = fov.astype(bool)
            MM, Rk = mm_rank_stack(POOL, fn, ccams, cfidx, fov)          # K x H x W each
            U = np.stack([norm1(x, fov) for x in MM], 0)[:, fi]          # unit-sum maps (manuscript)
            X = {"MinMax": MM[:, fi], "UnitSum": U, "Rank": Rk[:, fi]}   # K x Nfov
            EPS = {"MinMax": a.eps, "UnitSum": EPS_UNITSUM, "Rank": a.eps, "SumOnly": EPS_UNITSUM}
            if "SumOnly" in a.norms:   # stored map without the min-max step (no minimum subtracted), unit sum in the FOV
                RAW = np.stack([np.maximum(ccams[m][cfidx[m][fn]].numpy().astype(np.float64), 0.0)[fi] for m in POOL], 0)
                X["SumOnly"] = RAW / np.maximum(RAW.sum(1, keepdims=True), 1e-12)
                floor.append(RAW.min(1) / np.maximum(RAW.max(1), 1e-12))     # FOV minimum as a share of the maximum
            cur_maps = {}
            for n in a.norms:
                for m in a.means:
                    for t in a.trims:
                        lo, hi = TRIMS[t]
                        v = np.zeros(fov.shape); v[fi] = combine(X[n], m, lo, hi, EPS[n])
                        cur_maps["%s-%s-%s" % (n, m, t)] = v
            v = np.zeros(fov.shape); v[fi] = np.exp(np.log(np.maximum(np.sort(U, axis=0)[4:], EPS_UNITSUM)).mean(0))
            cur_maps[REF_NAME] = v
            # zero diagnostics on the min-max maps (Rank maps have no zeros by construction: min = 1/Nfov)
            nz = (X["MinMax"] <= 0).sum(0)                               # number of pool maps that are 0 at each pixel
            zrows.append(dict(dataset=a.dataset, image_id=fn, n_fov=int(fi.sum()),
                              **{"share_ge%d" % j: float((nz >= j).mean()) for j in (1, 2, 3, 5, 7, 14)}))
            mzero.append((X["MinMax"] <= 0).mean(1))
            cur_fn = fn
        for name, A in cur_maps.items():
            sb = sweep_best(A, Lm, fov, KS)
            if sb is None:
                continue
            rows.append(dict(dataset=a.dataset, image_id=fn, lesion_type=lt, method=name, kind="consensus",
                             max_dice=sb["max_dice"], best_k=sb["best_topk_percent"]))
        if (i + 1) % 300 == 0:
            print("  ... %d/%d" % (i + 1, len(items)), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "%s_rules_per_case.csv" % tag), index=False)
    S = df.groupby("method").agg(n=("max_dice", "size"), dice=("max_dice", "mean"),
                                 best_k_med=("best_k", "median")).reset_index().sort_values("dice", ascending=False)
    S.to_csv(os.path.join(a.out_dir, "%s_rules_summary.csv" % tag), index=False)
    Z = pd.DataFrame(zrows)
    Z.to_csv(os.path.join(a.out_dir, "%s_zero_diag.csv" % tag), index=False)
    ZM = pd.DataFrame(dict(model=tags, mean_zero_share=np.mean(mzero, axis=0)))
    if floor:
        ZM["mean_floor_over_max"] = np.mean(floor, axis=0)   # how much the min-max step removes from each map
    ZM.to_csv(os.path.join(a.out_dir, "%s_zero_diag_by_model.csv" % tag), index=False)

    pd.set_option("display.width", 200)
    print("cases evaluated %d | skipped %d" % (df.groupby(["image_id", "lesion_type"]).ngroups, skipped))
    print("\n===== [%s] per-case optimal Dice by rule =====" % tag)
    print(S.round(4).to_string(index=False))
    W = S[S.method != REF_NAME]
    W = W.assign(norm=W.method.str.split("-").str[0], mean=W.method.str.split("-").str[1],
                 trim=W.method.str.split("-").str[2])
    W = W.pivot_table(index=["norm", "mean"], columns="trim", values="dice")
    print("\n----- same values as a grid (rows: normalisation x mean, columns: trimming) -----")
    print(W[[t for t in a.trims if t in W.columns]].round(4).to_string())
    print("\n----- zero diagnostics (MinMax maps): mean share of FOV pixels where >= j of %d pool maps are exactly 0 -----" % K)
    print(Z[[c for c in Z.columns if c.startswith("share_")]].mean().round(4).to_string())
    print("  (T1x1 still sees a 0 where >= 2 maps are 0; T2x2 where >= 3; the current rule where >= 5)")
    print("\nper model, mean share of FOV pixels that are exactly 0:")
    print(ZM.round(4).to_string(index=False))
    print("\nsaved ->", a.out_dir)


if __name__ == "__main__":
    main()
