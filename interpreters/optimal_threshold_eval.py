"""
optimal_threshold_eval.py — localization evaluation based on a per-image **optimal operating point**.

Motivation:
  Saliency values are defined only up to a per-image monotone normalization, i.e. they are **ordinal**,
  so applying the same fixed threshold to every image is inappropriate. Instead, each map is
  allowed its most favourable operating point on its own value distribution before comparison.

  "each map was thresholded at the 1-percentile level of its own value distribution
   that maximized Dice against the corresponding segmentation, and the resulting
   per-image maxima were averaged across the test set."

Procedure (per image × map):
  1. Evaluation region = same FOV as before (C.FOV & fundus_mask_448)
  2. Sweep all top k% (k = 1..99, in 1% steps)
  3. Dice / IoU against the GT lesion mask at each k
  4. Pick the k with maximum Dice and store best_topk_percent / max_dice / iou_at_best_threshold

Implementation notes:
  · Default is **rank-based top-N** (--threshold_mode rank). The FOV interior is sorted in descending order
    and N = round(k/100 · |FOV|) pixels are taken, so the mask is exactly k% even with many ties or zeros.
    (Same rationale and method as eophtha_localization.dice_iom_sweep.)
    With --threshold_mode quantile: thr = percentile(s, 100-k), then s >= thr.
    Without ties the two modes differ by at most 1 px in mask size, i.e. they are effectively identical;
    with many ties the quantile area can greatly exceed the nominal k%.
    Check the actual impact via tie_frac / max_tie_run / pos_frac in the per-case CSV.
  · The 99 masks are never built. One sort + one cumulative sum of GT membership gives the
    intersection for every k with a single lookup → ~15 ms per map.
  · For the same mask IoU = Dice/(2−Dice) **always** holds (monotone increasing), so
    argmax Dice and argmax IoU differ only on ties. The actual number of disagreements is reported.

Difference from Adaptive-k:
  Adaptive **forces** |S| = |L|, giving the GT size as the budget,
  whereas here each map picks the size most favourable to itself. GT is used only for that choice.

Empty GT: if |L| = 0 after intersecting with the FOV, that (image × lesion) is excluded from aggregation.

Targets: 21 individual models (single_root, 1.0x) + Consensus (sal_root=ms8, AUROC>=auc_min).
      Uses the same model set / preprocessing / FOV as gt_localization_all.py.

Output:
  <ds>_optthr_per_case.csv : dataset, lesion_type, image_id, method,
                             best_topk_percent, max_dice, iou_at_best_threshold, ...
  <ds>_optthr_summary.csv  : dataset × lesion × method summary
                             (Mean/SD/95%CI of Dice·IoU, mean/median/IQR of best-k)

Usage:
  python interpreters/optimal_threshold_eval.py --dataset idrid \
    --root $IDRID_SEG_ROOT \
    --sal_root outputs/idrid_saliency_ms8 --single_root outputs/idrid_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir outputs/idrid_optthr --types MA HE EX SE --auc_mins 0.90 0.92 0.95

  python interpreters/optimal_threshold_eval.py --merge \
    outputs/idrid_optthr/idrid_optthr_per_case.csv \
    outputs/fgadr_optthr/fgadr_optthr_per_case.csv \
    outputs/eophtha_optthr/eophtha_optthr_per_case.csv \
    --out_dir outputs/final_tables_optthr
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
from threshold_sweep_erx import (DISCOVER, DEF_TYPES, pool_variants, load_set,
                                 relboost, best8_of)


def sweep_best(A, Lm, fov, ks, mode="rank"):
    """Sweep all top-k% (k in ks) and return the point of maximum Dice.

    One sort + one cumulative sum covers every k:
        order = FOV interior values in descending order,  csum[i] = number of GT pixels among the top (i+1)
        Dice = 2*inter/(N+G),  IoU = inter/(N+G-inter)

    mode="rank"     : takes **exactly** N = round(k/100 * |FOV|) pixels (robust to ties).
    mode="quantile" : thr = percentile(s, 100-k) , then s >= thr.
                      With ties, |mask| can exceed the nominal k%.
    Both modes reuse the same sort / cumulative sum, so there is no cost difference
    (quantile just sets N to 'the number of values >= thr' — the same set as the top N in descending order).
    """
    if A is None:
        return None
    fi = np.flatnonzero(fov.ravel())
    n = fi.size
    Lf = Lm.ravel()[fi].astype(bool)
    G = int(Lf.sum())
    if G == 0 or n == 0:
        return None                      # empty GT -> excluded
    s = A.ravel()[fi]
    order = np.argsort(s, kind="stable")[::-1]     # ties broken by descending flatten index (deterministic)
    csum = np.cumsum(Lf[order].astype(np.int64))
    asc = s[order][::-1]                            # ascending values (for searchsorted)

    # Tie diagnostics: fraction of unique values and longest tie run
    chg = np.flatnonzero(np.diff(asc))
    n_uniq = chg.size + 1
    runs = np.diff(np.concatenate(([-1], chg, [n - 1])))
    max_run = int(runs.max()) if runs.size else n

    kk = np.asarray(ks, float)
    if mode == "quantile":
        thr = np.percentile(s, 100.0 - kk)
        Ns = n - np.searchsorted(asc, thr, side="left")     # |{s >= thr}|
        Ns = np.maximum(1, np.minimum(Ns, n)).astype(np.int64)
    else:
        Ns = np.maximum(1, np.rint(kk / 100.0 * n).astype(np.int64))
        Ns = np.minimum(Ns, n)

    inter = csum[Ns - 1].astype(np.float64)
    dice = 2.0 * inter / (Ns + G)
    union = Ns + G - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    bi = int(np.argmax(dice))
    bj = int(np.argmax(iou))
    return dict(best_topk_percent=float(ks[bi]),
                max_dice=float(dice[bi]),
                iou_at_best_threshold=float(iou[bi]),
                best_topk_by_iou=float(ks[bj]),
                max_iou=float(iou[bj]),
                n_pos=int(Ns[bi]),                 # number of selected mask pixels
                pos_frac=float(Ns[bi] / n),        # actual selected fraction (compare with nominal k%)
                pos_over_G=float(Ns[bi] / G),      # selected area as a multiple of GT
                G=G, fov_px=int(n),
                tie_frac=float(1.0 - n_uniq / n),  # fraction of tied pixels
                max_tie_run=max_run,               # size of the largest tie group
                # If the optimum sits at the grid edge, the true optimum may lie outside the grid.
                # For lesions like MA, where |L| is 0.03% of the FOV, even top-1% is 30x the GT,
                # so a 1% grid cannot find the optimum → add sub-1% values via --k_extra.
                at_grid_min=int(bi == 0),
                at_grid_max=int(bi == len(kk) - 1),
                argmax_mismatch=int(bi != bj))


def run_dataset(a, ks):
    types = a.types or DEF_TYPES[a.dataset]
    items = DISCOVER[a.dataset](a.root, types)
    auc = SC.load_auc_map(a.model_roots)

    # ── consensus inputs (ms8) ── same as gt_localization_all.py ──
    cn, ccams, cfidx = load_set(a.sal_root)
    cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
    clin = {m: C.lineage_of(m) for m in cn}
    cft = {m: L.ft_of(m) for m in cn}
    order_c = sorted(cn, key=lambda m: -auc.get(m, 0))
    # Lineage Consensus: best-AUROC model per 4 lineages x FT (Partial/LoRA) combination -> up to 8 models
    # (best8_of internally uses a fixed AUROC>=0.90 — the original definition of this method is kept)
    best8 = best8_of(cn, clin, cft, auc)
    LIN_TAG = "Lineage Consensus (best8, n=%d)" % len(best8)
    # Compare several thresholds in one run — identical pools are not recomputed; only their labels are merged
    pools, seen = [], {}
    for t in sorted(a.auc_mins):
        p = tuple(m for m in order_c if auc.get(m, 0) >= t)
        if not p:
            print("  [skip] AUROC>=%.2f pool is empty." % t); continue
        if p in seen:
            seen[p].append(t)
        else:
            seen[p] = [t]; pools.append(p)
    if not pools:
        sys.exit("[abort] all consensus pools are empty. sal_root=%s" % a.sal_root)
    pool_all = sorted({m for p in pools for m in p} | set(best8))

    # ── 21 individual models (1.0x attn) ──
    inames, icams, ifidx = load_set(a.single_root)
    inames = [m for m in inames if C.lineage_of(m) and L.ft_of(m)]
    inames = sorted(inames, key=lambda m: -(auc.get(m) or 0))
    if not inames:
        sys.exit("[abort] individual pool is empty. single_root=%s" % a.single_root)
    pretty = {m: "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in inames}
    tag_of = {}
    for p in pools:
        ts = seen[p]
        lab = "/".join("%.2f" % t for t in ts)
        tag_of[p] = "Consensus (AUROC>=%s, n=%d)" % (lab, len(p))

    print("[%s] cases %d | individual %d | top-k sweep %g~%g%% (%d step) | mode=%s"
          % (a.dataset, len(items), len(inames), ks[0], ks[-1], len(ks), a.threshold_mode))
    for p in pools:
        print("  %s" % tag_of[p])
    if best8:
        print("  %s : %s" % (LIN_TAG,
              [("%s-%s" % (L.backbone_pretty(m), cft[m])) for m in best8]))
    else:
        print("  [warn] best8 is empty; cannot build Lineage Consensus.")
    if len(pools) < len(a.auc_mins):
        print("  ※ thresholds yielding the same pool were merged into one "
              "(the ms8 root holds only Partial/LoRA, so 0.90 and 0.92 are usually identical).")

    fov_cache, rows = {}, []
    skipped = 0
    for i, r in enumerate(items):
        fn, lt = r["image_id"], r["lesion_type"]
        # N consistency: only cases where all 22 rows are available (same rule as gt_localization_all.py)
        if not all(fn in ifidx[m] for m in inames):
            skipped += 1; continue
        if not all(fn in cfidx[m] for m in pool_all):
            skipped += 1; continue
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        fov = fov_cache[fn]
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            skipped += 1; continue         # empty GT → excluded

        maps = {pretty[m]: E.mm(icams[m][ifidx[m][fn]].numpy()) * fov for m in inames}
        ok = True
        for p in pools:
            v = pool_variants(fn, list(p), ccams, cfidx, fov)
            if not v:
                ok = False; break
            maps[tag_of[p]] = v["Cent+RelBoost"]
        if best8:
            lc = relboost(fn, best8, clin, cft, ccams, cfidx, fov)
            if lc is None:
                ok = False
            else:
                maps[LIN_TAG] = lc
        if not ok:
            skipped += 1; continue

        per = {k: sweep_best(A, Lm, fov, ks, a.threshold_mode) for k, A in maps.items()}
        if any(x is None for x in per.values()):
            skipped += 1; continue
        for k, x in per.items():
            rows.append(dict(dataset=a.dataset, lesion_type=lt, image_id=fn, method=k, **x))
        if (i + 1) % 200 == 0:
            print("  ... %d/%d" % (i + 1, len(items)))

    if not rows:
        sys.exit("[abort] no cases evaluated. Check root/sal_root/single_root.")
    df = pd.DataFrame(rows)
    nmis = int(df.argmax_mismatch.sum())
    print("  evaluated case-items %d, excluded %d" % (df.groupby(["image_id", "lesion_type"]).ngroups, skipped))
    print("  [argmax check] Dice-max k vs IoU-max k mismatch: %d / %d rows (%.3f%%)"
          % (nmis, len(df), 100 * nmis / len(df)))
    if nmis:
        print("    ※ IoU = Dice/(2−Dice) is monotone increasing, so the argmax should in principle coincide.")
        print("       Mismatches arise only from **ties**, when adjacent k on the 1%% grid give the same Dice.")
        d = df[df.argmax_mismatch == 1]
        print("       mean |Δk| = %.2f%%, mean |ΔDice| = %.2e"
              % ((d.best_topk_by_iou - d.best_topk_percent).abs().mean(),
                 (d.max_dice - (2 * d.max_iou / (1 + d.max_iou))).abs().mean()))
    return df, types, [tag_of[p] for p in pools] + ([LIN_TAG] if best8 else [])


def summarize(df, cons_tag=None):
    """dataset × lesion(+ALL) × method summary. Mean/SD/95%CI + best-k distribution."""
    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)
    out = []
    for lt in ["ALL"] + sorted(df.lesion_type.unique()):
        sub = df if lt == "ALL" else df[df.lesion_type == lt]
        if sub.empty:
            continue
        g = sub.groupby(["dataset", "method"], as_index=False).agg(
            N=("max_dice", "size"),
            Dice=("max_dice", "mean"), Dice_SD=("max_dice", "std"),
            IoU=("iou_at_best_threshold", "mean"), IoU_SD=("iou_at_best_threshold", "std"),
            k_mean=("best_topk_percent", "mean"), k_median=("best_topk_percent", "median"),
            k_Q1=("best_topk_percent", q1), k_Q3=("best_topk_percent", q3),
            pos_over_G=("pos_over_G", "median"), pos_frac=("pos_frac", "median"),
            tie_frac=("tie_frac", "mean"), max_tie_run=("max_tie_run", "max"))
        g.insert(1, "lesion_type", lt)
        out.append(g)
    r = pd.concat(out, ignore_index=True)
    for m, sd in (("Dice", "Dice_SD"), ("IoU", "IoU_SD")):
        se = r[sd] / np.sqrt(r.N.clip(lower=1))
        r[m + "_lo"] = r[m] - 1.96 * se
        r[m + "_hi"] = r[m] + 1.96 * se
    return r


def report(r, df, cons_tag):
    for (ds, lt), g in r.groupby(["dataset", "lesion_type"], sort=False):
        g = g.sort_values("Dice", ascending=False).reset_index(drop=True)
        g.insert(0, "rank", g.index + 1)
        pos = g.index[g.method.astype(str).str.startswith("Consensus")]
        g["method"] = [("★ " if str(m).startswith("Consensus") else "   ") + str(m)
                       for m in g.method]
        print("\n===== %s / %s : per-image optimal operating point (N=%d) ====="
              % (ds, lt, int(g.N.iloc[0])))
        show = ["rank", "method", "N", "Dice", "Dice_lo", "Dice_hi", "IoU",
                "k_median", "k_Q1", "k_Q3", "pos_over_G"]
        print(g[show].round(4).to_string(index=False))
        if len(pos):
            print("  → Consensus rank: %d / %d" % (int(pos[0]) + 1, len(g)))

    if "at_grid_min" in df.columns:
        gm = df.groupby("method").agg(at_min=("at_grid_min", "mean"),
                                      at_max=("at_grid_max", "mean"),
                                      k_med=("best_topk_percent", "median"),
                                      pos_over_G=("pos_over_G", "median"))
        print("\n===== grid boundary diagnostics =====")
        print(gm.sort_values("at_min", ascending=False).round(4).to_string())
        bad = gm[gm.at_min > 0.2]
        if len(bad):
            print("  [warning] for the methods below, the optimal k sits at the grid **lower bound** in >=20% of cases "
                  "— the true optimum may lie lower:")
            for m in list(bad.index)[:6]:
                print("    %s (at lower bound %.0f%%, k_median=%.2f, pos/G=%.1fx)"
                      % (m, 100 * bad.at_min[m], bad.k_med[m], bad.pos_over_G[m]))
            print("  → rerun with --k_extra 0.05 0.1 0.2 0.3 0.5 0.75.")

    print("\n===== tie diagnostics =====")
    t = df.groupby("method").agg(tie_frac=("tie_frac", "mean"),
                                 max_tie_run=("max_tie_run", "max"),
                                 fov_px=("fov_px", "median"))
    t["max_tie_pct"] = 100 * t.max_tie_run / t.fov_px
    print(t.sort_values("tie_frac", ascending=False).round(4).to_string())
    print("  tie_frac = fraction of tied FOV pixels | max_tie_run = size of the largest tie group")
    print("  If tie_frac is close to 0, the rank and quantile modes are effectively identical.")

    # best-k distribution (overall)
    print("\n===== optimal top-k(%) distribution — by method =====")
    d = df.groupby("method").best_topk_percent.describe(
        percentiles=[.1, .25, .5, .75, .9])[["count", "mean", "10%", "25%", "50%", "75%", "90%"]]
    print(d.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge per-dataset *_optthr_per_case.csv files into a pooled summary")
    ap.add_argument("--dataset", choices=list(DISCOVER))
    ap.add_argument("--root")
    ap.add_argument("--sal_root", help="consensus inputs (ms8)")
    ap.add_argument("--single_root", help="21 individual models (1.0x attn)")
    ap.add_argument("--model_roots", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_mins", nargs="+", type=float, default=[0.95],
                    help="Consensus pool thresholds. If several are given, all are compared in one run. "
                         "e.g. --auc_mins 0.90 0.92 0.95 (identical pools are merged automatically)")
    ap.add_argument("--threshold_mode", choices=["rank", "quantile"], default="rank",
                    help="rank=select exactly top-k%% of pixels (robust to ties, recommended) | "
                         "quantile=threshold at the percentile value (area grows with ties)")
    ap.add_argument("--kmin", type=float, default=1.0)
    ap.add_argument("--kmax", type=float, default=99.0)
    ap.add_argument("--kstep", type=float, default=1.0,
                    help="top-k%% sweep step (default 1%%)")
    ap.add_argument("--k_extra", nargs="*", type=float, default=[],
                    help="extra sub-1%% values to add to the grid. For tiny lesions such as MA, even top-1%% is "
                         "tens of times the GT, so a 1%% grid cannot find the optimum. "
                         "recommended: --k_extra 0.05 0.1 0.2 0.3 0.5 0.75 "
                         "(the earlier dice_iom_sweep started at 0.05%%)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.merge:
        df = pd.concat([pd.read_csv(p) for p in a.merge], ignore_index=True)
        df["dataset"] = "POOLED"
        r = summarize(df)
        p = os.path.join(a.out_dir, "optthr_pooled_summary.csv")
        r.round(6).to_csv(p, index=False)
        cons = [m for m in df.method.unique() if str(m).startswith("Consensus")]
        report(r, df, cons[0] if cons else None)
        print("\nSaved:", p)
        return

    for req in ("dataset", "root", "sal_root", "single_root", "model_roots"):
        if getattr(a, req) is None:
            sys.exit("[abort] --%s is required (or use --merge)." % req)

    ks = np.arange(a.kmin, a.kmax + 1e-9, a.kstep)
    if a.k_extra:
        ks = np.unique(np.concatenate([np.asarray(a.k_extra, float), ks]))
    df, types, cons_tags = run_dataset(a, ks)
    p1 = os.path.join(a.out_dir, "%s_optthr_per_case.csv" % a.dataset)
    df.to_csv(p1, index=False)
    r = summarize(df)
    p2 = os.path.join(a.out_dir, "%s_optthr_summary.csv" % a.dataset)
    r.round(6).to_csv(p2, index=False)
    report(r, df, cons_tags)
    print("\n  Dice/IoU are taken at an optimal top-k **chosen separately for each image**,")
    print("  so they are structurally higher than fixed top-K%% or Otsu values. For comparing methods, not as absolute values.")
    print("  pos_over_G = selected mask pixels / GT pixels (median). >1 means the selection is wider than the GT.")
    print("Saved:", p1, "\nSaved:", p2)


if __name__ == "__main__":
    main()
