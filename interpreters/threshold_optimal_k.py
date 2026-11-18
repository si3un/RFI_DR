"""
threshold_optimal_k.py - optimal top-k threshold analysis (requested at the 8/26 AI meeting).

Purpose (this does not propose a new thresholding method):
  1. With threshold arbitrariness removed by a per-case oracle threshold,
     does the consensus still outperform individual models?
  2. What is the actual distribution of the optimal threshold?
  3. Is the consensus advantage retained with a single representative fixed threshold?

Method (applied identically to every method):
  · grid: --grid coarse = 1..99% (1% steps, original)
          --grid fine   = 0.01%(<2%) + 0.1%(2~10%) + 1%(>=10%)
          --grid ultrafine = 0.01% over the full range (--no_curves recommended)
  · Rank-based top-N pixel selection (robust to ties). N = round(k/100 * |FOV|)
  · case = (image_id x lesion_type). Lesion types are kept separate because GT differs per lesion even for the same image.
  · If the maximum Dice is tied across several k, the **smallest k** is chosen (argmax returns the first maximum).
  · GT is used only for scoring and for deriving optimal-k, never for building maps.

Compared methods:
  · individual : set via --individuals. Default all = every individual model in single_root
  · consensus  : HybridMeanQ75 @AUROC>=0.92 (n=14)
  · The reference individual in the tables is the highest-AUROC model (= DINOv2-LoRA); per dataset,
    the "individual with the highest oracle Dice" is also selected for comparison.

Output (outputs/threshold_analysis/):
  {ds}_optimal_k_per_case.csv    dataset,image_id,lesion_type,method,optimal_k,
                                 optimal_dice,optimal_iou,lesion_area,lesion_frac
  {ds}_distribution_summary.csv  N,mean,SD,median,Q1,Q3,min,max,mode,IQR,at_min
  {ds}_fixed_k_performance.csv   oracle / fixed median-k / mean-k / mode-k (+ supplementary)
  {ds}_curves.npz                Dice/IoU curves per k (for fixed-k recomputation, no re-sweep needed)

Usage:
  python interpreters/threshold_optimal_k.py --dataset idrid \
    --root $IDRID_SEG_ROOT \
    --sal_root outputs/idrid_saliency_ms8 --single_root outputs/idrid_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir outputs/threshold_analysis --individuals all
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
from threshold_sweep_erx import DISCOVER, DEF_TYPES, load_set
from consensus_newaxis_eval import hybrid_mean_q75, mm_rank_stack

KS = np.arange(1.0, 100.0, 1.0)    # primary grid (replaced via --grid in main())


def make_grid(name, kmin, kmax, step):
    """top-k grid (%). fine uses a dense 0.01% step only in the low range where optima cluster."""
    if name == "coarse":                        # original: 1..99 %
        g = np.arange(1.0, 100.0, 1.0)
    elif name == "ultrafine":                   # 0.01 % over the full range (9,900 points; saving curves not recommended)
        g = np.arange(0.01, 99.0 + 1e-9, 0.01)
    elif name == "fine":                        # 0.01% (<2%) / 0.1% (2~10%) / 1% (>=10%)
        g = np.concatenate([np.arange(0.01, 2.0, 0.01),
                            np.arange(2.0, 10.0, 0.1),
                            np.arange(10.0, 100.0, 1.0)])
    else:                                       # custom: --k_min/--k_max/--k_step
        g = np.arange(kmin, kmax + 1e-9, step)
    return np.round(np.unique(g), 4)


def snap(k):
    """Snap a representative k to the nearest grid point. Ties go to the smaller k (argmin returns the first minimum)."""
    j = int(np.argmin(np.abs(KS - float(k))))
    return j, float(KS[j])


def sweep_curves(A, Lm, fov):
    """Dice/IoU curves for all k, computed exactly with one sort + cumulative sum."""
    fi = np.flatnonzero(fov.ravel())
    s = np.asarray(A).ravel()[fi]
    Lf = np.asarray(Lm).ravel()[fi]
    n = s.size
    G = int(Lf.sum())
    if n == 0 or G == 0:
        return None
    order = np.argsort(s, kind="stable")[::-1]
    csum = np.cumsum(Lf[order].astype(np.int64))
    Ns = np.clip(np.round(KS / 100.0 * n).astype(np.int64), 1, n)
    inter = csum[Ns - 1].astype(np.float64)
    dice = 2.0 * inter / (Ns + G)
    iou = inter / (Ns + G - inter)
    return dice, iou, G, n


def mode_k(v):
    """Mode. Ties go to the smallest k."""
    c = pd.Series(np.asarray(v)).value_counts()
    return float(min(c[c == c.max()].index))


def summarize(v):
    v = np.asarray(v, dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(N=len(v), mean=v.mean(), SD=v.std(ddof=1) if len(v) > 1 else 0.0,
                median=float(np.median(v)), Q1=float(q1), Q3=float(q3), IQR=float(q3 - q1),
                min=float(v.min()), max=float(v.max()), mode=mode_k(v),
                at_min=float((v <= KS[0]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DISCOVER))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sal_root", required=True, help="multi-scale consensus saliency root")
    ap.add_argument("--single_root", required=True, help="saliency root of individual models")
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs/threshold_analysis")
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_min", type=float, default=0.92)
    ap.add_argument("--individuals", nargs="*", default=["all"],
                    help="all = all models | auc = AUROC>=auc_min only | model names = those models only")
    ap.add_argument("--grid", default="coarse",
                    choices=["coarse", "fine", "ultrafine", "custom"],
                    help="coarse=1..99%%(1%%) | fine=0.01%%(<2%%)+0.1%%+1%% | "
                         "ultrafine=0.01%% over full range | custom=--k_min/--k_max/--k_step")
    ap.add_argument("--k_min", type=float, default=0.01)
    ap.add_argument("--k_max", type=float, default=99.0)
    ap.add_argument("--k_step", type=float, default=0.01)
    ap.add_argument("--no_curves", action="store_true",
                    help="do not save curve npz (saves space with fine/ultrafine)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    global KS
    KS = make_grid(a.grid, a.k_min, a.k_max, a.k_step)
    types = a.types or DEF_TYPES[a.dataset]
    items = DISCOVER[a.dataset](a.root, types)
    auc = SC.load_auc_map(a.model_roots)

    cn, ccams, cfidx = load_set(a.sal_root)
    cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
    POOL = [m for m in sorted(cn, key=lambda m: -auc.get(m, 0)) if auc.get(m, 0) >= a.auc_min]
    if len(POOL) < 3:
        sys.exit("[abort] AUROC>=%.2f pool has %d models." % (a.auc_min, len(POOL)))

    inames, icams, ifidx = load_set(a.single_root)
    inames = [m for m in inames if C.lineage_of(m) and L.ft_of(m)]
    inames = sorted(inames, key=lambda m: -(auc.get(m) or 0))
    sel = a.individuals or ["all"]
    if len(sel) == 1 and sel[0] == "all":
        IND = list(inames)
    elif len(sel) == 1 and sel[0] == "auc":
        IND = [m for m in inames if (auc.get(m) or 0) >= a.auc_min]
    else:
        IND = [m for m in inames if any(s in m for s in sel)]
        miss = [s for s in sel if not any(s in m for m in inames)]
        if miss:
            sys.exit("[abort] individual not found: %s\n  available: %s" % (miss, inames))
    if not IND:
        sys.exit("[abort] 0 individuals")

    itag = {m: "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in IND}
    if len(set(itag.values())) != len(IND):                 # avoid name collisions
        for m in IND:
            itag[m] = "%s [%s]" % (itag[m], m[:18])
    T_HYB = "HybridMeanQ75 @AUROC>=%.2f (n=%d)" % (a.auc_min, len(POOL))
    T_REF = itag[IND[0]]                                    # highest-AUROC individual model
    print("[%s] candidate cases %d | consensus pool %d | individuals %d"
          % (a.dataset, len(items), len(POOL), len(IND)))
    print("  reference individual = %s (AUROC %.4f)" % (T_REF, auc.get(IND[0], float("nan"))))
    print("  grid[%s]: top-k = %g..%g %% (%d points), smallest k on ties"
          % (a.grid, KS[0], KS[-1], len(KS)))

    ipath = {r["image_id"]: r["image_path"] for r in items}
    fov_cache = {}

    def fov_of(f):
        if f not in fov_cache:
            fov_cache[f] = C.FOV & E.fundus_mask_448(E.letterbox_img(ipath[f]))
        return fov_cache[f]

    rows, curves, skipped = [], {}, 0
    cur_fn, cur_maps = None, None
    for i, r in enumerate(sorted(items, key=lambda r: r["image_id"])):
        fn, lt = r["image_id"], r["lesion_type"]
        if not all(fn in cfidx[m] for m in POOL) or not all(fn in ifidx.get(m, {}) for m in IND):
            skipped += 1
            continue
        fov = fov_of(fn)
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            skipped += 1
            continue
        if fn != cur_fn:
            _MM, Rk = mm_rank_stack(POOL, fn, ccams, cfidx, fov)
            cur_maps = {T_HYB: hybrid_mean_q75(Rk, fov)}
            for m in IND:
                cur_maps[itag[m]] = E.mm(icams[m][ifidx[m][fn]].numpy()) * fov
            cur_fn = fn
        got, ok = {}, True
        for tag, A in cur_maps.items():
            out = sweep_curves(A, Lm, fov)
            if out is None:
                ok = False
                break
            got[tag] = out
        if not ok:
            skipped += 1
            continue
        for tag, (dice, iou, G, n) in got.items():
            j = int(np.argmax(dice))          # first maximum = smallest k (tie rule)
            rows.append(dict(dataset=a.dataset, image_id=fn, lesion_type=lt, method=tag,
                             optimal_k=float(KS[j]), optimal_dice=float(dice[j]),
                             optimal_iou=float(iou[j]), lesion_area=int(G),
                             lesion_frac=float(G) / float(n), fov_px=int(n),
                             k_px=int(round(KS[j] / 100.0 * n))))
            curves[(tag, fn, lt)] = (dice.astype(np.float32), iou.astype(np.float32))
        if (i + 1) % 300 == 0:
            print("  ... %d/%d" % (i + 1, len(items)))

    if not rows:
        sys.exit("[abort] no cases evaluated.")
    df = pd.DataFrame(rows)
    df.round(6).to_csv(os.path.join(a.out_dir, "%s_optimal_k_per_case.csv" % a.dataset),
                       index=False)
    print("  evaluated case-items %d, excluded %d | methods %d"
          % (df.groupby(["image_id", "lesion_type"]).ngroups, skipped, df["method"].nunique()))

    nbytes = len(curves) * len(KS) * 2 * 4
    if a.no_curves:
        print("  [--no_curves] curve npz not saved (~%.1f GB saved)" % (nbytes / 1e9))
    else:
        if nbytes > 4e9:
            print("  [warn] curves take ~%.1f GB. --no_curves is recommended." % (nbytes / 1e9))
        keys = sorted(curves)
        np.savez_compressed(os.path.join(a.out_dir, "%s_curves.npz" % a.dataset),
                            ks=KS, keys=np.array(["|".join(k) for k in keys]),
                            dice=np.stack([curves[k][0] for k in keys]),
                            iou=np.stack([curves[k][1] for k in keys]))

    IND_T = [itag[m] for m in IND]
    METHODS = [T_HYB] + IND_T
    # Individual with the highest oracle Dice per dataset
    oall = df[df["lesion_type"].notna()].groupby("method")["optimal_dice"].mean()
    T_BEST = oall[IND_T].idxmax()
    FOCUS = [T_HYB, T_REF] + ([T_BEST] if T_BEST != T_REF else [])

    # ── 3. Distribution summary ──
    srows = []
    for m in METHODS:
        for lt in ["ALL"] + list(types):
            s = df[df["method"] == m] if lt == "ALL" else \
                df[(df["method"] == m) & (df["lesion_type"] == lt)]
            if s.empty:
                continue
            srows.append(dict(dataset=a.dataset, method=m, lesion_type=lt,
                              is_consensus=int(m == T_HYB),
                              **summarize(s["optimal_k"].to_numpy())))
    S = pd.DataFrame(srows)
    S.round(4).to_csv(os.path.join(a.out_dir, "%s_distribution_summary.csv" % a.dataset),
                      index=False)
    cols = ["method", "N", "mean", "SD", "median", "Q1", "Q3", "min", "max", "mode",
            "IQR", "at_min"]
    print("\n" + "=" * 100)
    print("[1] OPTIMAL TOP-K distribution - %s / ALL (all methods)" % a.dataset)
    print(S[S["lesion_type"] == "ALL"].sort_values("median")[cols].to_string(index=False))
    print("\n[1b] per lesion (consensus + reference/best individual only; see CSV for all)")
    print(S[(S["lesion_type"] != "ALL") & (S["method"].isin(FOCUS))]
          [["lesion_type"] + cols].to_string(index=False))

    # ── 4. Representative fixed-k performance ──
    def curve_stack(m, sub):
        idx = [(m, r.image_id, r.lesion_type) for r in sub.itertuples()]
        return (np.stack([curves[k][0] for k in idx]),
                np.stack([curves[k][1] for k in idx]))

    frows = []
    for m in METHODS:
        for lt in ["ALL"] + list(types):
            sub = df[df["method"] == m] if lt == "ALL" else \
                df[(df["method"] == m) & (df["lesion_type"] == lt)]
            if sub.empty:
                continue
            v = sub["optimal_k"].to_numpy()
            D, I = curve_stack(m, sub)
            od, oi = float(sub["optimal_dice"].mean()), float(sub["optimal_iou"].mean())
            rec = dict(dataset=a.dataset, method=m, lesion_type=lt, N=len(sub),
                       is_consensus=int(m == T_HYB), oracle_dice=od, oracle_iou=oi,
                       grid=a.grid,
                       best_fixed_k=float(KS[int(np.argmax(D.mean(axis=0)))]),
                       best_fixed_dice=float(D.mean(axis=0).max()))
            for tag, kk in (("median", float(np.median(v))), ("mean", float(v.mean())),
                            ("mode", float(mode_k(v))), ("max", float(v.max()))):
                j, kap = snap(kk)             # snap to nearest grid point
                rec["k_%s" % tag] = kap
                rec["k_%s_raw" % tag] = kk
                rec["dice_%s" % tag] = float(D[:, j].mean())
                rec["iou_%s" % tag] = float(I[:, j].mean())
                rec["retention_%s" % tag] = 100.0 * float(D[:, j].mean()) / od if od > 0 else np.nan
            frows.append(rec)
    F = pd.DataFrame(frows)
    F.round(6).to_csv(os.path.join(a.out_dir, "%s_fixed_k_performance.csv" % a.dataset),
                      index=False)

    print("\n" + "=" * 100)
    print("[2] representative FIXED-K performance - %s / ALL, sorted by oracle Dice desc (ret = fixed/oracle x100)"
          % a.dataset)
    s = F[F["lesion_type"] == "ALL"].sort_values("oracle_dice", ascending=False)
    print("   %-34s %8s | %6s %7s %6s | %6s %7s %6s | %6s %7s %6s"
          % ("Method", "Oracle", "medk", "Dice", "ret%", "meank", "Dice", "ret%",
             "modek", "Dice", "ret%"))
    for _, r in s.iterrows():
        mark = " <<<" if r["is_consensus"] else ""
        print("   %-34s %8.4f | %6.2f %7.4f %5.1f | %6.2f %7.4f %5.1f | %6.2f %7.4f %5.1f%s"
              % (r["method"][:34], r["oracle_dice"],
                 r["k_median"], r["dice_median"], r["retention_median"],
                 r["k_mean"], r["dice_mean"], r["retention_mean"],
                 r["k_mode"], r["dice_mode"], r["retention_mode"], mark))

    print("\n[2b] IoU - %s / ALL" % a.dataset)
    for _, r in s.iterrows():
        print("   %-34s oracle %.4f | med-k %.4f | mean-k %.4f | mode-k %.4f%s"
              % (r["method"][:34], r["oracle_iou"], r["iou_median"], r["iou_mean"],
                 r["iou_mode"], " <<<" if r["is_consensus"] else ""))

    print("\n[2c] per lesion (consensus + reference/best individual)")
    for lt in list(types):
        sl = F[(F["lesion_type"] == lt) & (F["method"].isin(FOCUS))]
        if sl.empty:
            continue
        print("  --- %s ---" % lt)
        for _, r in sl.iterrows():
            print("   %-34s oracle %.4f | med-k %6.2f -> %.4f (ret %.1f%%)"
                  % (r["method"][:34], r["oracle_dice"], r["k_median"],
                     r["dice_median"], r["retention_median"]))

    print("\n[2d] supplementary - max-k (excluded from primary: strongly affected by extremes)")
    for _, r in F[(F["lesion_type"] == "ALL") & (F["method"].isin(FOCUS))].iterrows():
        print("   %-34s max-k %6.2f -> %.4f (ret %.1f%%) | ref. best-fixed-k %6.2f -> %.4f"
              % (r["method"][:34], r["k_max"], r["dice_max"], r["retention_max"],
                 r["best_fixed_k"], r["best_fixed_dice"]))

    # ── 5. Key question: does the consensus beat individuals even at a fixed threshold ──
    print("\n" + "=" * 100)
    print("[3] Key question - does the consensus beat individuals? (%s)" % a.dataset)
    for lt in ["ALL"] + list(types):
        sl = F[F["lesion_type"] == lt].set_index("method")
        if T_HYB not in sl.index:
            continue
        ind = [m for m in IND_T if m in sl.index]
        print("\n  --- %s / %s (N=%d, individuals %d) ---"
              % (a.dataset, lt, int(sl.loc[T_HYB, "N"]), len(ind)))
        for tag, lab in (("oracle", "per-case oracle"), ("median", "fixed median-k"),
                         ("mean", "fixed mean-k"), ("mode", "fixed mode-k")):
            col = "oracle_dice" if tag == "oracle" else "dice_%s" % tag
            hv = float(sl.loc[T_HYB, col])
            iv = sl.loc[ind, col].astype(float)
            rank = int((iv > hv).sum()) + 1
            bi = iv.idxmax()
            kk = ("" if tag == "oracle" else
                  "  [k: Hyb %g / best-ind %g]" % (sl.loc[T_HYB, "k_%s" % tag],
                                                   sl.loc[bi, "k_%s" % tag]))
            print("    %-16s Hybrid %.4f | best-ind %.4f (%s) | %s %.4f%s"
                  % (lab, hv, iv.max(), bi[:22], T_REF[:16],
                     float(sl.loc[T_REF, col]), kk))
            print("      %14s vs best-ind %+6.1f%% | vs %s %+6.1f%% | rank among %d individuals: %d"
                  % ("", 100 * (hv - iv.max()) / iv.max() if iv.max() > 0 else float("nan"),
                     T_REF[:12],
                     100 * (hv - float(sl.loc[T_REF, col])) / float(sl.loc[T_REF, col])
                     if float(sl.loc[T_REF, col]) > 0 else float("nan"),
                     len(ind), rank))

    # ── 6. Operating-point stability ──
    print("\n" + "=" * 100)
    print("[4] optimal-k stability - is the consensus operating point narrower than individuals'?")
    for lt in ["ALL"] + list(types):
        sl = S[S["lesion_type"] == lt].set_index("method")
        if T_HYB not in sl.index:
            continue
        ind = [m for m in IND_T if m in sl.index]
        hS, hQ = float(sl.loc[T_HYB, "SD"]), float(sl.loc[T_HYB, "IQR"])
        iS, iQ = sl.loc[ind, "SD"].astype(float), sl.loc[ind, "IQR"].astype(float)
        rS, rQ = int((iS < hS).sum()) + 1, int((iQ < hQ).sum()) + 1
        print("\n  %-5s SD  Hyb %6.2f | ind median %6.2f (range %.2f~%.2f) -> rank %d/%d (narrowest first)"
              % (lt, hS, iS.median(), iS.min(), iS.max(), rS, len(ind) + 1))
        print("        IQR Hyb %6.2f | ind median %6.2f (range %.2f~%.2f) -> rank %d/%d"
              % (hQ, iQ.median(), iQ.min(), iQ.max(), rQ, len(ind) + 1))
        print("        at_min Hyb %.3f | ind median %.3f"
              % (sl.loc[T_HYB, "at_min"], sl.loc[ind, "at_min"].astype(float).median()))
        both = (hS < iS.median()) and (hQ <= iQ.median())
        wide = (hS > iS.median()) and (hQ >= iQ.median())
        print("        -> %s" % ("consensus is narrower (stable operating point)" if both else
                                 "consensus is wider" if wide else
                                 "mixed - SD/IQR disagree in direction"))

    print("\nSaved: %s" % a.out_dir)
    print("Next: generate figures with threshold_figures.py (no re-sweep needed)")


if __name__ == "__main__":
    main()
