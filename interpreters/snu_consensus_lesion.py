"""
snu_consensus_lesion.py — SNU consensus map vs actual lesion GT (localization validation)

Unlike the external consensus (leave-one-out agreement), SNU has 8-region lesion GT, so this directly tests
"whether the consensus map of multiple models concentrates on actual lesions" (inside/outside ratio).

Consensus construction: FOV sum-normalize each model in the set → within-lineage mean → equal-weight mean over lineages
  → prevents over-representation of a lineage due to unequal lineage sizes (same rationale as external).

Three sets:
  best4   : best internal-AUC model per lineage (CLIP/DINOv2/DINOv3/MAE) (4 models)
  auc09   : all models with internal AUC >= auc_min (default 0.9)
  lora    : names containing lora (single FT method)

Metrics (consensus map per image → aggregated per set):
  logratio_median(+boot 95%CI), contrast_mean, ratio_gt1_rate, topk_enrich_median, pg_rate

Usage:
  python interpreters/snu_consensus_lesion.py --sal_root .../snu_saliency \
    --model_roots .../dr_clean .../ckpts_aux \
    --lesion_csv $SNUH_LESION_CSV --crop_csv .../annotated_..._coord.csv \
    --out_dir .../snu_eval --auc_min 0.9 --variant lenient --min_agree 0 \
    --exclude retfound_dinov2_partial_ft_2
"""
import os, sys, re, glob, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C

EPS = 1e-6


def strip_ts(name):
    return re.sub(r"_\d{8}_\d{6}$", "", name)


def load_auc_map(model_roots):
    m = {}
    for root in model_roots:
        if not root or not os.path.isdir(root):
            continue
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(d):
                continue
            name = strip_ts(os.path.basename(d))
            auc = C.read_internal_auc(d)
            if np.isfinite(auc):
                m[name] = auc
    return m


def fov_sum_norm(cam):
    c = cam * C.FOV
    s = c.sum()
    return c / s if s > C.ZERO_EPS else c


def topk_enrichment(cam, gt, fov, k):
    vals = cam[fov]
    if vals.size == 0:
        return np.nan
    thr = np.quantile(vals, 1.0 - k)
    topk = (cam >= thr) & fov
    if topk.sum() == 0:
        return np.nan
    frac_in = (topk & gt).sum() / topk.sum()
    gt_frac = (gt & fov).sum() / fov.sum()
    return float(frac_in / gt_frac) if gt_frac > 0 else np.nan


def boot_median_ci(x, n=2000, seed=42):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [np.median(rng.choice(x, x.size, replace=True)) for _ in range(n)]
    return (float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def select_sets(models, auc_map, auc_min):
    """models: [name,...] → {set_name: [name,...]}"""
    lin = {m: C.lineage_of(m) for m in models}
    sets = {}
    # best4: best-AUC model per lineage
    best4 = []
    for g in ["CLIP", "DINOv2", "DINOv3", "MAE"]:
        cand = [(auc_map.get(m, np.nan), m) for m in models if lin[m] == g]
        cand = [(a, m) for a, m in cand if np.isfinite(a)]
        if cand:
            best4.append(max(cand)[1])
    sets["best4"] = best4
    # auc>=min
    sets[f"auc{auc_min:g}"] = [m for m in models if auc_map.get(m, -1) >= auc_min]
    # lora
    sets["lora"] = [m for m in models if "lora" in m.lower()]
    return sets, lin


def build_consensus(fname, mods, lin, cam_cache, fn2idx_cache):
    """Lineage-balanced consensus map. All mods must have fname (common images) → otherwise None."""
    by_lin = {}
    for m in mods:
        if fname not in fn2idx_cache[m]:
            return None
        c = fov_sum_norm(cam_cache[m][fn2idx_cache[m][fname]].numpy())
        by_lin.setdefault(lin[m], []).append(c)
    lin_maps = [np.mean(v, axis=0) for v in by_lin.values()]
    return np.mean(lin_maps, axis=0)   # equal weight per lineage


# ── FT-balanced consensus (exclude classifier-only; equal weight for Partial/LoRA) ──
def ft_of(name):
    n = name.lower()
    if "lora" in n:
        return "LoRA"
    if "partial" in n:
        return "Partial"
    if "linear" in n:
        return "Linear"
    return None


def select_sets_ftbal(models, auc_map):
    """Partial+LoRA only. ftbal14=all, ftbal8=best Partial+best LoRA per lineage."""
    lin = {m: C.lineage_of(m) for m in models}
    ft = {m: ft_of(m) for m in models}
    pl = [m for m in models if ft[m] in ("Partial", "LoRA")]
    sets = {"ftbal14": pl}
    best8 = []
    for g in ["CLIP", "DINOv2", "DINOv3", "MAE"]:
        for f in ("Partial", "LoRA"):
            cand = [(auc_map.get(m, np.nan), m) for m in pl if lin[m] == g and ft[m] == f]
            cand = [(a, m) for a, m in cand if np.isfinite(a)]
            if cand:
                best8.append(max(cand)[1])
    sets["ftbal8"] = best8
    return sets, lin, ft


def build_consensus_ftbal(fname, mods, lin, ft, cam_cache, fn2idx_cache):
    """mean within lineage × FT → equal weight for Partial/LoRA → equal weight per lineage (sum-norm, for quantitative use)."""
    by = {}
    for m in mods:
        if fname not in fn2idx_cache[m]:
            return None
        c = fov_sum_norm(cam_cache[m][fn2idx_cache[m][fname]].numpy())
        by.setdefault(lin[m], {}).setdefault(ft[m], []).append(c)
    lin_maps = []
    for g, ftd in by.items():
        ft_means = [np.mean(v, axis=0) for v in ftd.values()]  # mean within FT
        lin_maps.append(np.mean(ft_means, axis=0))             # equal weight for Partial/LoRA
    return np.mean(lin_maps, axis=0)                            # equal weight per lineage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sal_root", required=True)
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--lesion_csv", required=True)
    ap.add_argument("--crop_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--auc_min", type=float, default=0.9)
    ap.add_argument("--variant", default="lenient")
    ap.add_argument("--min_agree", type=float, default=0.0)
    ap.add_argument("--topk", type=float, default=0.10)
    ap.add_argument("--exclude", nargs="*", default=[])   # attention rollout has no zero-CAMs → include all
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"_hc{a.min_agree}" if a.min_agree > 0 else ""

    # prepare lesion GT union
    img_regions, _ = C.load_lesion_regions(a.lesion_csv, a.min_agree)
    crop_lut = C.load_crop_lut(a.crop_csv)
    union_cache = {}
    for fname, regs in img_regions.items():
        if fname not in crop_lut.index:
            continue
        masks = []
        for reg in regs:
            try:
                masks.append(C.region_mask_448(fname, reg, crop_lut) & C.FOV)
            except Exception:
                pass
        if masks:
            union_cache[fname] = np.logical_or.reduce(masks)

    # load models/saliency
    all_models = sorted([os.path.basename(d) for d in glob.glob(os.path.join(a.sal_root, "*"))
                         if os.path.exists(os.path.join(d, "saliency_map.pth"))])
    models = [m for m in all_models if not any(ex in m for ex in a.exclude)]
    auc_map = load_auc_map(a.model_roots)
    cam_cache, fn2idx_cache = {}, {}
    for m in models:
        cam, fn2idx = C.load_saliency(os.path.join(a.sal_root, m))
        cam_cache[m] = cam; fn2idx_cache[m] = fn2idx

    sets, lin, ft = select_sets_ftbal(models, auc_map)   # Primary(ftbal14)/Sensitivity(ftbal8)
    print(f"[{a.variant}{tag}] lesions in {len(union_cache)} images | candidate models {len(models)} | auc_min={a.auc_min}")
    for sn, mods in sets.items():
        cnt = {}
        for m in mods:
            cnt[lin[m]] = cnt.get(lin[m], 0) + 1
        print(f"  [{sn}] {len(mods)} models {dict(cnt)}")
        for m in mods:
            print(f"      {m}  (auc={auc_map.get(m, float('nan')):.4f}, {lin[m]})")

    # evaluate the consensus map per set
    detail_rows, summ_rows = [], []
    for sn, mods in sets.items():
        if len(mods) == 0:
            continue
        recs = []
        for fname in union_cache:
            cons = build_consensus_ftbal(fname, mods, lin, ft, cam_cache, fn2idx_cache)
            if cons is None:
                continue
            union = union_cache[fname]
            if cons.max() <= C.ZERO_EPS:
                continue
            din, dout, ratio = C.inout_ratio(cons, union, C.FOV)
            if not np.isfinite(ratio):
                continue
            logr = float(np.log(ratio))                      # log of the same ratio → consistent
            contrast = float((din - dout) / (din + dout + EPS))
            nap = C.nauprc(cons, union, C.FOV)               # PRIMARY
            tk = C.topk_enrich_fixed(cons, union, C.FOV, a.topk)
            amax = np.unravel_index(np.argmax(cons), cons.shape)
            rec = {"set": sn, "file_name": fname, "inside_mean": din, "outside_mean": dout,
                   "ratio": ratio, "log_ratio": logr, "contrast": contrast,
                   "nauprc": nap, "topk_enrich": tk, "pg": int(union[amax])}
            recs.append(rec); detail_rows.append(rec)
        if not recs:
            continue
        df = pd.DataFrame(recs)
        nmed, nlo, nhi = boot_median_ci(df["nauprc"].values)
        med, lo, hi = boot_median_ci(df["log_ratio"].values)
        rmed, rlo, rhi = boot_median_ci(df["ratio"].values)          # inside/outside ratio
        tmed, tlo, thi = boot_median_ci(df["topk_enrich"].values)    # top-k enrichment CI
        summ_rows.append({
            "set": sn, "n_models": len(mods), "n_images": len(df),
            # PRIMARY
            "nauprc_median": round(nmed, 4),
            "nauprc_95CI": f"[{nlo:.3f}, {nhi:.3f}]",
            # SECONDARY
            "pointing_game_rate": round(float(df["pg"].mean()), 3),
            "topk10_median": round(tmed, 3),
            "topk10_95CI": f"[{tlo:.3f}, {thi:.3f}]",
            # DESCRIPTIVE / AUX
            "inout_ratio_median": round(rmed, 3),
            "inout_ratio_95CI": f"[{rlo:.3f}, {rhi:.3f}]",
            "logratio_median": round(med, 3),
            "logratio_95CI": f"[{lo:.3f}, {hi:.3f}]",
            "hit_rate": round(float((df["log_ratio"] > 0).mean()), 3),
            "contrast_mean": round(float(np.nanmean(df["contrast"])), 3),
        })

    pd.DataFrame(detail_rows).to_csv(
        os.path.join(a.out_dir, f"snu_consensus_detail_{a.variant}{tag}.csv"), index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(os.path.join(a.out_dir, f"snu_consensus_summary_{a.variant}{tag}.csv"), index=False)
    print(f"\n===== SNU consensus vs lesion GT ({a.variant}{tag}) =====")
    print(summ.to_string(index=False))

    # ── individual model evaluation (each model's own saliency) ──
    pm_rows = []
    for m in models:
        recs = []
        for fname in union_cache:
            if fname not in fn2idx_cache[m]:
                continue
            c = fov_sum_norm(cam_cache[m][fn2idx_cache[m][fname]].numpy())
            union = union_cache[fname]
            if c.max() <= C.ZERO_EPS:
                continue
            din, dout, ratio = C.inout_ratio(c, union, C.FOV)
            if not np.isfinite(ratio):
                continue
            amax = np.unravel_index(np.argmax(c), c.shape)
            recs.append({"nauprc": C.nauprc(c, union, C.FOV),
                         "topk_enrich": C.topk_enrich_fixed(c, union, C.FOV, a.topk),
                         "ratio": ratio, "pg": int(union[amax])})
        if not recs:
            continue
        d = pd.DataFrame(recs)
        nmed, nlo, nhi = boot_median_ci(d["nauprc"].values)
        rmed, rlo, rhi = boot_median_ci(d["ratio"].values)
        pm_rows.append({
            "model": m, "lineage": C.lineage_of(m), "ft": ft_of(m),
            "auc": round(auc_map.get(m, float("nan")), 4), "n_images": len(d),
            "nauprc_median": round(nmed, 4), "nauprc_95CI": f"[{nlo:.3f}, {nhi:.3f}]",
            "pointing_game_rate": round(float(d["pg"].mean()), 3),
            "topk10_median": round(float(np.nanmedian(d["topk_enrich"])), 3),
            "inout_ratio_median": round(rmed, 3), "inout_ratio_95CI": f"[{rlo:.3f}, {rhi:.3f}]",
        })
    pm = pd.DataFrame(pm_rows)
    if len(pm):
        pm = pm.sort_values(["lineage", "ft", "model"]).reset_index(drop=True)
    pm.to_csv(os.path.join(a.out_dir, f"snu_permodel_summary_{a.variant}{tag}.csv"), index=False)
    print(f"\n===== SNU individual models vs lesion GT ({a.variant}{tag}) =====")
    print(pm.to_string(index=False))

    print(f"\nSaved: {a.out_dir}/snu_consensus_{{detail,summary}} and snu_permodel_summary_{a.variant}{tag}.csv")


if __name__ == "__main__":
    main()
