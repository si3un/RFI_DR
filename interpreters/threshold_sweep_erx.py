"""
threshold_sweep.py — top-K% threshold sweep curves for consensus/best-single maps.

Reviewer comments: "the top-10% cutoff is arbitrary" + "absolute values are low".
For each top-K% (K=1..50), binarize and compute Dice/IoU/IoM/Recall/Precision against GT → curves.
IoM and Recall are much higher than IoU for small lesions, so they are examined as alternative absolute metrics.

Maps evaluated (lineage-agnostic flat pools + lineage RelBoost + best single, optionally Enhanced ms8):
  BestSingle / FlatMean(AUC>=0.90, AUC>=0.95, Top-3/5/10) / Lineage-RelBoost [/ Enhanced-RelBoost]

Metrics (per K, averaged over images):
  Dice, IoU, IoM(=inter/min(|P|,|G|)), Recall(=inter/|G|), Precision(=inter/|P|)

Threshold-free continuous saliency metrics (computed per image, then summarized):
  ER   = saliency mass inside the lesion / total saliency mass in the FOV
  ERx  = ER / (lesion area / FOV area)
         → 1.0 = uniform saliency baseline, >1 = lesion enrichment
  ERx can spike for tiny lesions, so mean and median/IQR are saved together.

Usage:
  python interpreters/threshold_sweep.py --dataset idrid \
    --root "$IDRID_SEG_ROOT" \
    --sal_root outputs/idrid_saliency_attn --model_roots models/dr_clean models/ckpts_aux \
    --out_dir outputs/idrid_sweep --types MA HE EX SE \
    [--ms_root outputs/idrid_saliency_ms8]
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

DISCOVER = {"eophtha": E.discover, "fgadr": F.discover, "idrid": ID.discover}
DEF_TYPES = {"fgadr": ["MA", "HE", "EX", "SE"], "idrid": ["MA", "HE", "EX", "SE"], "eophtha": ["EX", "MA"]}
KGRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def topk_mask(cam, fov, k):
    v = cam[fov]
    if v.size == 0:
        return np.zeros_like(cam, bool)
    thr = np.quantile(v, 1 - k / 100.0)
    return (cam >= thr) & fov


def metrics_at(mask, Lm):
    inter = int(np.logical_and(mask, Lm).sum())
    P = int(mask.sum()); G = int(Lm.sum())
    if G == 0:
        return None
    union = int(np.logical_or(mask, Lm).sum())
    return dict(Dice=2 * inter / (P + G) if (P + G) else 0.0,
                IoU=inter / union if union else 0.0,
                IoM=inter / min(P, G) if min(P, G) else 0.0,
                Recall=inter / G,
                Precision=inter / P if P else 0.0)


def erx_metrics(cam, Lm, fov):
    """Threshold-free saliency enrichment metrics.

    ER  = saliency mass inside lesion / total saliency mass inside FOV
    ERx = ER / lesion-area fraction inside FOV

    Interpretation of ERx:
      1.0  -> uniform saliency baseline
      >1.0 -> saliency enriched inside lesion
      <1.0 -> saliency depleted inside lesion

    All consensus/best-single maps are already built as non-negative maps in builders(),
    but NaN/Inf are removed and values below 0 are clipped for numerical stability.
    """
    if cam is None:
        return None
    G = int(Lm.sum())
    F = int(fov.sum())
    if G == 0 or F == 0:
        return None

    x = np.asarray(cam, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)

    total_mass = float(x[fov].sum())
    if total_mass <= 0:
        return dict(ER=0.0, ERx=0.0, LesionAreaFrac=G / F)

    lesion_mass = float(x[Lm].sum())
    er = lesion_mass / total_mass
    area_frac = G / F
    erx = er / area_frac if area_frac > 0 else np.nan
    return dict(ER=er, ERx=erx, LesionAreaFrac=area_frac)


def flat_mean(fn, pool, cams, fidx, fov):
    ms = [E.mm(cams[m][fidx[m][fn]].numpy()) * fov for m in pool if fn in fidx[m]]
    return np.mean(ms, axis=0) if ms else None


def relboost(fn, best8, lin, ft, cams, fidx, fov):
    """(Reference only) previous lineage-based RelBoost — 8→4 lineage centrality + 2-of-4."""
    lm = CC.lineage_maps(fn, best8, lin, ft, cams, fidx)
    if not lm or len(lm) < 2:
        return None
    gk = sorted(lm); raw = [lm[g] for g in gk]; Rk = [CC.rank_in_fov(lm[g], fov) for g in gk]
    return CC.relboost_from_ranks(raw, Rk, fov)


def model_relboost(fn, pool, cams, fidx, fov):
    """Lineage-free Model-RelBoost: simple-mean base × majority-of-K agreement."""
    v = pool_variants(fn, pool, cams, fidx, fov)
    return v.get("RelBoost") if v else None


def pool_variants(fn, pool, cams, fidx, fov):
    """Compute all 4 combination methods for one pool (K models, no lineage grouping).
      base:  FlatMean = (1/K)ΣM_i           |  Centrality = Σ w_i M_i  (w_i ∝ mean rank correlation with the other models, clip≥0)
      boost: agreement A = ceil(K/2)-th highest rank (majority-of-K, consistent with the original 2-of-4)
      Returns: FlatMean, Centrality, RelBoost(=FlatMean×(1+A)), Cent+RelBoost(=Centrality×(1+A))
      (threshold sweep/ERx need no normalization — they are based on quantiles and mass ratios)"""
    maps = [E.mm(cams[m][fidx[m][fn]].numpy()) * fov for m in pool if fn in fidx[m]]
    if len(maps) < 2:
        return {}
    K = len(maps)
    stack = np.stack(maps, axis=0)
    R = [CC.rank_in_fov(mp, fov) for mp in maps]
    # model-level centrality (same formula as relboost_from_ranks, per model instead of per lineage)
    Cm = np.nan_to_num(np.corrcoef(np.stack([r[fov] for r in R])))
    G = np.array([(Cm[k].sum() - 1.0) / (K - 1) for k in range(K)]) if K > 1 else np.array([1.0])
    w = np.clip(G, 0, None); sw = w.sum(); w = w / sw if sw > 0 else np.ones(K) / K
    flat = np.mean(stack, axis=0)
    cent = np.tensordot(w, stack, axes=1)
    mth = int(np.ceil(K / 2.0))
    A = np.sort(np.stack(R, axis=0), axis=0)[-mth]               # majority-of-K agreement
    return {"FlatMean": flat, "Centrality": cent,
            "RelBoost": flat * (1.0 + A), "Cent+RelBoost": cent * (1.0 + A)}


def load_set(sal_root):
    names = sorted([os.path.basename(d) for d in glob.glob(os.path.join(sal_root, "*"))
                    if os.path.exists(os.path.join(d, "saliency_map.pth"))])
    cams, fidx = {}, {}
    for m in names:
        cam, f2i = C.load_saliency(os.path.join(sal_root, m)); cams[m] = cam; fidx[m] = f2i
    return names, cams, fidx


def best8_of(names, lin, ft, auc):
    prim = [m for m in names if ft.get(m) in ("Partial", "LoRA") and auc.get(m, 0) >= 0.9]
    b8 = []
    for g in ["CLIP", "DINOv2", "DINOv3", "MAE"]:
        for f in ("Partial", "LoRA"):
            cnd = [(auc.get(m, np.nan), m) for m in prim if lin[m] == g and ft[m] == f]
            cnd = [(x, m) for x, m in cnd if np.isfinite(x)]
            if cnd:
                b8.append(max(cnd)[1])
    return b8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DISCOVER))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sal_root", required=True)
    ap.add_argument("--ms_root", default="", help="if an ms8 root is given, add the Enhanced-RelBoost curve")
    ap.add_argument("--single_root", default="", help="root of BestSingle individual maps (default=sal_root). "
                    "set to saliency_attn to keep individual maps at 1.0×")
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    types = a.types or DEF_TYPES[a.dataset]
    items = DISCOVER[a.dataset](a.root, types)

    auc = SC.load_auc_map(a.model_roots)
    names, cams, fidx = load_set(a.sal_root)
    lin = {m: C.lineage_of(m) for m in names}; ft = {m: L.ft_of(m) for m in names}
    names = [m for m in names if lin[m] and ft[m]]
    order = sorted(names, key=lambda m: -auc.get(m, 0))
    # BestSingle individual baseline: from single_root (e.g. 1.0× attn) if given, otherwise from sal_root
    if a.single_root and os.path.isdir(a.single_root):
        s_names, single_cams, single_fidx = load_set(a.single_root)
        s_names = [m for m in s_names if C.lineage_of(m) and L.ft_of(m)]
        best_single = sorted(s_names, key=lambda m: -auc.get(m, 0))[0]
    else:
        single_cams, single_fidx = cams, fidx
        best_single = order[0]
    pools = {"Top-3": order[:3], "Top-5": order[:5],
             "AUC>=0.95": [m for m in order if auc.get(m, 0) >= 0.95],   # = Top-10 (rank1~10)
             "AUC>=0.90": [m for m in order if auc.get(m, 0) >= 0.90]}
    best8 = best8_of(names, lin, ft, auc)

    ms_names = ms_cams = ms_fidx = ms_best8 = None
    if a.ms_root and os.path.isdir(a.ms_root):
        ms_names, ms_cams, ms_fidx = load_set(a.ms_root)
        ms_lin = {m: C.lineage_of(m) for m in ms_names}; ms_ft = {m: L.ft_of(m) for m in ms_names}
        ms_names = [m for m in ms_names if ms_lin[m] and ms_ft[m]]
        ms_best8 = best8_of(ms_names, ms_lin, ms_ft, auc)

    print(f"[{a.dataset}] cases {len(items)} | best_single={best_single} | best8={len(best8)}"
          + (f" | ms_best8={len(ms_best8)}" if ms_best8 else ""))
    for k, v in pools.items():
        print(f"  pool {k:10} ({len(v):2})")

    def builders(fn, fov):
        bs = (E.mm(single_cams[best_single][single_fidx[best_single][fn]].numpy()) * fov
              if fn in single_fidx.get(best_single, {}) else None)
        d = {f"BestSingle:{L.backbone_pretty(best_single)}-{L.ft_of(best_single)}(1.0x)": bs}
        # each pool × 4 methods (FlatMean/Centrality/RelBoost/Cent+RelBoost) — all lineage-free
        for k, v in pools.items():
            for meth, mp in pool_variants(fn, v, cams, fidx, fov).items():
                d[f"{meth}/{k}"] = mp
        # previous lineage (best8) for reference: no weighting/boost (FlatMean) vs final (centrality+2of4)
        lm = CC.lineage_maps(fn, best8, lin, ft, cams, fidx)
        if lm and len(lm) >= 2:
            gk = sorted(lm); raw = [lm[g] for g in gk]; Rk = [CC.rank_in_fov(lm[g], fov) for g in gk]
            d["Lineage-FlatMean"] = np.mean(raw, axis=0)
            d["Lineage-RelBoost(orig)"] = CC.relboost_from_ranks(raw, Rk, fov)
        return d

    # accumulate: (map, lesion, K) -> [sums..., n]
    MET = ["Dice", "IoU", "IoM", "Recall", "Precision"]
    acc = {}

    # threshold-free ER/ERx do not depend on K, so per-image values are saved separately.
    # ERx can be long-tailed for tiny lesions, so raw per-case values are kept so that
    # the summary can report median/IQR as well as the mean.
    cont_rows = []
    def add(mp, lt, k, m):
        key = (mp, lt, k)
        if key not in acc:
            acc[key] = {x: 0.0 for x in MET}; acc[key]["n"] = 0
        for x in MET:
            acc[key][x] += m[x]
        acc[key]["n"] += 1

    fov_cache = {}
    for r in items:
        fn = r["image_id"]; lt = r["lesion_type"]
        if not all(fn in fidx[m] for m in best8):
            continue
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        fov = fov_cache[fn]
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            continue
        maps = builders(fn, fov)
        for mp, cam in maps.items():
            if cam is None:
                continue

            # --- NEW: threshold-free ER / ERx ---
            cm = erx_metrics(cam, Lm, fov)
            if cm:
                cont_rows.append(dict(image_id=fn, map_name=mp, lesion_type=lt, **cm))
                # also add to ALL per lesion item, same as the existing ALL aggregation
                cont_rows.append(dict(image_id=fn, map_name=mp, lesion_type="ALL", **cm))

            # existing top-K% threshold sweep
            for k in KGRID:
                m = metrics_at(topk_mask(cam, fov, k), Lm)
                if m:
                    add(mp, lt, k, m); add(mp, "ALL", k, m)

    rows = []
    for (mp, lt, k), s in acc.items():
        n = s["n"]
        rows.append(dict(map_name=mp, lesion_type=lt, K_pct=k, N=n,
                         **{x: round(s[x] / n, 4) for x in MET}))
    out = pd.DataFrame(rows).sort_values(["lesion_type", "map_name", "K_pct"])
    out.to_csv(os.path.join(a.out_dir, f"{a.dataset}_threshold_sweep.csv"), index=False)

    # --- NEW: save threshold-free ER / ERx results ---
    if cont_rows:
        cont = pd.DataFrame(cont_rows)
        cont = cont.sort_values(["lesion_type", "map_name", "image_id"])
        cont_path = os.path.join(a.out_dir, f"{a.dataset}_erx_per_case.csv")
        cont.to_csv(cont_path, index=False)

        def q25(x): return x.quantile(0.25)
        def q75(x): return x.quantile(0.75)
        summary = (
            cont.groupby(["lesion_type", "map_name"], as_index=False)
                .agg(N=("ERx", "size"),
                     ER_mean=("ER", "mean"),
                     ER_median=("ER", "median"),
                     ERx_mean=("ERx", "mean"),
                     ERx_median=("ERx", "median"),
                     ERx_Q1=("ERx", q25),
                     ERx_Q3=("ERx", q75),
                     LesionAreaFrac_mean=("LesionAreaFrac", "mean"))
        )
        for c in ["ER_mean", "ER_median", "ERx_mean", "ERx_median",
                  "ERx_Q1", "ERx_Q3", "LesionAreaFrac_mean"]:
            summary[c] = summary[c].round(4)
        summary_path = os.path.join(a.out_dir, f"{a.dataset}_erx_summary.csv")
        summary.to_csv(summary_path, index=False)

        all_erx = summary[summary.lesion_type == "ALL"].copy()
        if not all_erx.empty:
            print(f"\n===== {a.dataset} / ALL / ERx (threshold-free) =====")
            print(all_erx[["map_name", "N", "ER_mean", "ERx_mean",
                           "ERx_median", "ERx_Q1", "ERx_Q3"]]
                  .sort_values("ERx_mean", ascending=False).to_string(index=False))
            print("  ERx interpretation: 1.0=uniform baseline, >1=lesion enrichment")

    # console: ALL, mainly IoM/Recall per K (map x K)
    for met in ["IoM", "Recall", "Dice", "IoU"]:
        piv = out[out.lesion_type == "ALL"].pivot(index="map_name", columns="K_pct", values=met)
        print(f"\n===== {a.dataset} / ALL / {met} (rows=map, columns=top-K%) =====")
        print(piv.to_string())
    print("\nSaved:", os.path.join(a.out_dir, f"{a.dataset}_threshold_sweep.csv"))
    if cont_rows:
        print("Saved:", os.path.join(a.out_dir, f"{a.dataset}_erx_per_case.csv"))
        print("Saved:", os.path.join(a.out_dir, f"{a.dataset}_erx_summary.csv"))


if __name__ == "__main__":
    main()
