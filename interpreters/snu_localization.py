"""
snu_localization.py — SNUH lesion localization (the same evaluation code applied to 23 maps).
Maps: 21 individual + Primary consensus (14) + Sensitivity consensus (8).
Preprocessing (per image and map): relu(max(A,0)) → 0 outside the FOV → sum=1 inside the FOV.  (lesion masks are used for evaluation only)
Three metrics:
  ① nAUPRC   = (AUPRC − a)/(1 − a),  a=|L∩FOV|/|FOV|   (chance-corrected; can be negative)
  ② Pointing Game = argmax_FOV(A) ∈ L gives 1
  ③ Inside/Outside = mean_in / mean_out  (within the FOV, ratio of mean density inside/outside the lesion)
Common images: only those with all 23 maps and a valid lesion mask (high-conf ≥2/3).
No composite score is built (the metrics answer different questions).

Outputs:
  snu_metrics_per_image.csv  (one row per image×map + QC)
  snu_loc_consensus.csv      (Table 1: Primary/Sensitivity)
  snu_loc_permodel.csv       (Table 2: 21 individual models)

Usage:
  python interpreters/snu_localization.py --sal_root outputs/snu_saliency_attn_lb \
    --model_roots models/dr_clean models/ckpts_aux \
    --lesion_csv $SNUH_LESION_CSV --crop_csv .../coord.csv \
    --out_dir outputs/snu_localization --min_agree 0.67
"""
import os, sys, re, glob, argparse
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC

EPS = 1e-6
FT_LABEL = {"Linear": "Classifier-only", "Partial": "Partial", "LoRA": "LoRA"}


def ft_of(name):
    n = name.lower()
    if "lora" in n: return "LoRA"
    if "partial" in n: return "Partial"
    if "linear" in n: return "Linear"
    return None


def backbone_pretty(name):
    n = name.lower()
    if "clip" in n or "laion" in n: return "OpenCLIP"
    if "retfound_dinov2" in n: return "RETFound-DINOv2"
    if "retfound_mae" in n or (n.startswith("retfound") and "dinov2" not in n): return "RETFound"
    if "dinov3" in n:
        return "DINOv3-Huge" if ("huge" in n or "vith" in n) else "DINOv3-Large"
    if "dinov2" in n: return "DINOv2"
    if "mae" in n: return "MAE"
    return name


def prep(a2d):
    """relu → FOV → sum=1."""
    a = np.maximum(a2d, 0.0) * C.FOV
    s = a.sum()
    return a / s if s > 1e-12 else a


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else np.nan


def metrics(A, L, fov):
    y = L[fov].astype(int); s = A[fov]
    a = float(y.mean())
    if y.sum() == 0 or y.sum() == len(y):
        return None
    ap = float(average_precision_score(y, s))
    nap = (ap - a) / (1 - a) if a < 1 else np.nan
    # Pointing game: argmax within the FOV
    Am = np.where(fov, A, -np.inf)
    amax = np.unravel_index(np.argmax(Am), Am.shape)
    pg = int(L[amax])
    inside, outside = fov & L, fov & (~L)
    mi = float(A[inside].mean()) if inside.sum() else np.nan
    mo = float(A[outside].mean()) if outside.sum() else np.nan
    rio = (mi + EPS) / (mo + EPS)
    # IoU vs lesion: Otsu binarization (within the FOV) + top-10%
    Lf = L & fov
    mn, mx = float(s.min()), float(s.max())
    tk = {5: np.nan, 10: np.nan, 20: np.nan}
    if mx > mn:
        a01 = np.clip((A - mn) / (mx - mn + 1e-12), 0, 1) * fov
        iou_otsu = iou((a01 > C.otsu(a01[fov])) & fov, Lf)
        for kp in (5, 10, 20):
            tk[kp] = iou((A >= np.quantile(A[fov], 1 - kp / 100)) & fov, Lf)
    else:
        iou_otsu = np.nan
    return dict(nAUPRC=nap, raw_AUPRC=ap, pointing_game=pg, inside_outside=rio,
                otsu_iou=iou_otsu, top05_iou=tk[5], top10_iou=tk[10], top20_iou=tk[20],
                mean_inside=mi, mean_outside=mo, lesion_area=a, attention_sum=float(A[fov].sum()))


def boot_ci(x, fn=np.median, n=2000, seed=42):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [fn(rng.choice(x, x.size, replace=True)) for _ in range(n)]
    return (float(fn(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def boot_geomean(x, n=2000, seed=42):
    """geometric mean = exp(mean(log R)) + bootstrap 95% CI (corrects for the right skew of the ratio)."""
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    lg = np.log(x); rng = np.random.default_rng(seed)
    b = [np.exp(np.mean(rng.choice(lg, lg.size, replace=True))) for _ in range(n)]
    return (float(np.exp(np.mean(lg))), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def build_consensus(fn, mods, lin, ft, cams, fidx):
    by = {}
    for m in mods:
        if fn not in fidx[m]:
            return None
        by.setdefault(lin[m], {}).setdefault(ft[m], []).append(prep(cams[m][fidx[m][fn]].numpy()))
    lin_maps = []
    for g, ftd in by.items():
        ft_means = [np.mean(v, axis=0) for v in ftd.values()]
        lin_maps.append(np.mean(ft_means, axis=0))
    return prep(np.mean(lin_maps, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sal_root", required=True)
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--lesion_csv", required=True)
    ap.add_argument("--crop_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_agree", type=float, default=0.67)
    ap.add_argument("--auc_min", type=float, default=0.9)
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names = sorted([os.path.basename(d) for d in glob.glob(os.path.join(a.sal_root, "*"))
                    if os.path.exists(os.path.join(d, "saliency_map.pth"))])
    auc = SC.load_auc_map(a.model_roots)
    cams, fidx, lin, ft = {}, {}, {}, {}
    for m in names:
        cam, f2i = C.load_saliency(os.path.join(a.sal_root, m)); cams[m] = cam; fidx[m] = f2i
        lin[m] = C.lineage_of(m); ft[m] = ft_of(m)
    names = [m for m in names if lin[m] and ft[m]]
    prim = [m for m in names if ft[m] in ("Partial", "LoRA") and auc.get(m, 0) >= a.auc_min]
    best8 = []
    for g in ["CLIP", "DINOv2", "DINOv3", "MAE"]:
        for f in ("Partial", "LoRA"):
            cand = [(auc.get(m, np.nan), m) for m in prim if lin[m] == g and ft[m] == f]
            cand = [(x, m) for x, m in cand if np.isfinite(x)]
            if cand:
                best8.append(max(cand)[1])

    reg_hi, _ = C.load_lesion_regions(a.lesion_csv, a.min_agree)
    crop = C.load_crop_lut(a.crop_csv)

    def lesion_mask(fn):
        ms = []
        for reg in reg_hi[fn]:
            try:
                ms.append(C.region_mask_448(fn, reg, crop) & C.FOV)
            except Exception:
                pass
        return np.logical_or.reduce(ms) if ms else None

    # common valid images: valid lesion mask + maps from all 21 models
    cand = sorted([fn for fn in reg_hi if fn in crop.index])
    common, excl = [], []
    for fn in cand:
        if not all(fn in fidx[m] for m in names):
            excl.append((fn, "missing_saliency")); continue
        L = lesion_mask(fn)
        if L is None or L.sum() == 0:
            excl.append((fn, "empty_lesion_mask")); continue
        if (L & C.FOV).sum() >= C.FOV.sum():
            excl.append((fn, "lesion_covers_FOV")); continue
        common.append(fn)
    N = len(common)
    print(f"models {len(names)} | valid lesion images N={N} (excluded {len(excl)})")
    for fn, why in excl:
        print("  [excl]", fn[-34:], why)

    # map list: 21 individual + 2 consensus
    per_rows = []
    for fn in common:
        L = lesion_mask(fn)
        # individual
        for m in names:
            r = metrics(prep(cams[m][fidx[m][fn]].numpy()), L, C.FOV)
            if r is None:
                continue
            per_rows.append(dict(image_id=fn, map_name=f"{backbone_pretty(m)}-{FT_LABEL[ft[m]]}",
                                 map_type="Individual", lineage=lin[m], ft=FT_LABEL[ft[m]], valid=True,
                                 exclusion_reason="", **r))
        # consensus
        for tag, mods in [("Primary", prim), ("Sensitivity", best8)]:
            cons = build_consensus(fn, mods, lin, ft, cams, fidx)
            r = metrics(cons, L, C.FOV) if cons is not None else None
            if r is None:
                continue
            per_rows.append(dict(image_id=fn, map_name=tag, map_type="Consensus",
                                 lineage="—", ft="—", valid=True, exclusion_reason="", **r))
    per = pd.DataFrame(per_rows)
    per.to_csv(os.path.join(a.out_dir, "snu_metrics_per_image.csv"), index=False)

    def summarize(sub):
        napm = boot_ci(sub["nAUPRC"].values, np.mean, a.nboot)     # PRIMARY = mean
        napmd = boot_ci(sub["nAUPRC"].values, np.median, a.nboot)  # median for comparison
        riog = boot_geomean(sub["inside_outside"].values, a.nboot)  # PRIMARY = geometric mean
        riomd = boot_ci(sub["inside_outside"].values, np.median, a.nboot)  # median for comparison
        pg = boot_ci(sub["pointing_game"].values, np.mean, a.nboot)
        iou_o = boot_ci(sub["otsu_iou"].values, np.mean, a.nboot)      # mIoU (mean)
        i05 = boot_ci(sub["top05_iou"].values, np.mean, a.nboot)
        i10 = boot_ci(sub["top10_iou"].values, np.mean, a.nboot)
        i20 = boot_ci(sub["top20_iou"].values, np.mean, a.nboot)
        npg = int(sub["pointing_game"].sum()); nn = len(sub)
        return dict(N=nn,
                    nAUPRC_mean=round(napm[0], 3), nAUPRC_mean_CI=f"[{napm[1]:.3f}, {napm[2]:.3f}]",
                    nAUPRC_median=round(napmd[0], 3),
                    PG_n=npg, PG_pct=round(100 * pg[0], 1), PG_CI=f"[{100*pg[1]:.1f}, {100*pg[2]:.1f}]",
                    otsu_iou_mean=round(iou_o[0], 3), otsu_iou_CI=f"[{iou_o[1]:.3f}, {iou_o[2]:.3f}]",
                    top05_iou_mean=round(i05[0], 3), top05_iou_CI=f"[{i05[1]:.3f}, {i05[2]:.3f}]",
                    top10_iou_mean=round(i10[0], 3), top10_iou_CI=f"[{i10[1]:.3f}, {i10[2]:.3f}]",
                    top20_iou_mean=round(i20[0], 3), top20_iou_CI=f"[{i20[1]:.3f}, {i20[2]:.3f}]",
                    inout_geomean=round(riog[0], 3), inout_geomean_CI=f"[{riog[1]:.3f}, {riog[2]:.3f}]",
                    inout_median=round(riomd[0], 3))

    # Table 1 consensus
    t1 = []
    for tag, lbl, n in [("Primary", "Primary consensus (14)", len(prim)), ("Sensitivity", "Sensitivity consensus (8)", len(best8))]:
        s = summarize(per[per.map_name == tag]); s = {"consensus": lbl, **s}; t1.append(s)
    df1 = pd.DataFrame(t1); df1.to_csv(os.path.join(a.out_dir, "snu_loc_consensus.csv"), index=False)

    # Table 2: 21 individual
    t2 = []
    order = {g: i for i, g in enumerate(["CLIP", "DINOv2", "DINOv3", "MAE"])}
    ford = {"Classifier-only": 0, "Partial": 1, "LoRA": 2}
    for m in names:
        nm = f"{backbone_pretty(m)}-{FT_LABEL[ft[m]]}"
        s = summarize(per[per.map_name == nm])
        t2.append(dict(model=backbone_pretty(m), lineage=lin[m], fine_tuning=FT_LABEL[ft[m]],
                       _l=order.get(lin[m], 9), _f=ford[FT_LABEL[ft[m]]], **s))
    df2 = pd.DataFrame(t2).sort_values(["_l", "model", "_f"]).drop(columns=["_l", "_f"]).reset_index(drop=True)
    df2.to_csv(os.path.join(a.out_dir, "snu_loc_permodel.csv"), index=False)

    print("\n===== Table 1. Consensus lesion localization =====")
    print(df1[["consensus", "N", "nAUPRC_mean", "nAUPRC_mean_CI", "PG_n", "PG_pct", "PG_CI",
               "otsu_iou_mean", "otsu_iou_CI", "inout_geomean", "inout_geomean_CI"]].to_string(index=False))
    print("\n===== Table 2. Per-model lesion localization =====")
    print(df2[["model", "lineage", "fine_tuning", "N", "nAUPRC_mean", "nAUPRC_mean_CI",
               "PG_n", "PG_pct", "otsu_iou_mean", "otsu_iou_CI", "top10_iou_mean",
               "inout_geomean", "inout_geomean_CI"]].to_string(index=False))

    print("\n===== Rank comparison (mean-based statistic vs median) =====")
    for label, cnew, cold in [("nAUPRC: mean vs median", "nAUPRC_mean", "nAUPRC_median"),
                              ("Inside/Outside: geomean vs median", "inout_geomean", "inout_median")]:
        d = df2[["model", "fine_tuning", cnew, cold]].copy()
        d["rank_new"] = d[cnew].rank(ascending=False, method="min").astype(int)
        d["rank_old"] = d[cold].rank(ascending=False, method="min").astype(int)
        d["drank"] = d["rank_old"] - d["rank_new"]
        rho = d[cnew].corr(d[cold], method="spearman")
        chg = d[d["drank"] != 0].sort_values("rank_new")
        print(f"\n[{label}]  rank correlation Spearman ρ={rho:.3f}  |  rank changed for {len(chg)}/{len(d)} maps")
        if len(chg):
            print(chg[["model", "fine_tuning", cnew, cold, "rank_new", "rank_old", "drank"]].to_string(index=False))

    # ── IoU comparison table (Otsu·top5·top10·top20; 21 individual + 2 consensus) ──
    ic = ["otsu_iou_mean", "top05_iou_mean", "top10_iou_mean", "top20_iou_mean"]
    cc = df1[["consensus"] + ic].rename(columns={"consensus": "map"})
    pp = df2.copy(); pp["map"] = pp["model"] + " · " + pp["fine_tuning"]
    iou_tbl = pd.concat([cc[["map"] + ic], pp[["map"] + ic]], ignore_index=True)
    iou_tbl.columns = ["map", "Otsu", "top5%", "top10%", "top20%"]
    iou_tbl.to_csv(os.path.join(a.out_dir, "snu_loc_iou.csv"), index=False)
    print("\n===== IoU comparison (21 individual + 2 consensus) =====")
    print(iou_tbl.to_string(index=False))

    print("\nSaved:", a.out_dir, "(snu_metrics_per_image.csv, snu_loc_consensus.csv, snu_loc_permodel.csv, snu_loc_iou.csv)")


if __name__ == "__main__":
    main()
