"""
fgadr_localization.py — external validation of saliency localization with FGADR pixel-level lesion GT.

Reuses the **same metrics, preprocessing, consensus, bootstrap and tables** as eophtha_localization.py,
replacing only the dataset adapter (discover/mask paths) for the FGADR folder structure.
  - Original_Images/<stem>.png  +  <Lesion>_Masks/<stem>.png (same stem, one folder per lesion type)
  - only non-empty masks are taken as cases (FGADR has mask files for all 1842 images, many of them all black)
Reported separately per lesion type (MA/HE/EX/SE, +IRMA/NV).

Metrics (per image, same as eophtha_localization):
  Dice_global(Ayhan, dataset-optimized f*) · IoM_global(Zoom-in-Net @f*) · Dice_oracle/IoM_oracle (upper bound)
  · Pointing Game(Saporta) · AUPRC(Arun)/nAUPRC · IoU(Otsu/top-N) · Inside/Outside
Preprocessing: saliency=relu→FOV→sum=1 (sum1 consensus) ; per-image fundus FOV(fmask) ; mask=letterbox 448 NEAREST.

Usage:
  python interpreters/fgadr_localization.py \
    --fgadr_root $FGADR_SEG_ROOT \
    --sal_root  outputs/fgadr_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir   outputs/fgadr_localization --types MA HE EX SE --per_model
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E   # letterbox_mask/letterbox_img/fundus_mask_448/mm/build_consensus_mm/FGRID/dice_iom_sweep/global_fstar/summarize reused

# FGADR lesion folders (spelled exactly as in the Readme)
LES = {"MA": "Microaneurysms_Masks", "HE": "Hemohedge_Masks", "EX": "HardExudate_Masks",
       "SE": "SoftExudate_Masks", "IRMA": "IRMA_Masks", "NV": "Neovascularization_Masks"}


def discover(root, types):
    """Original_Images/<stem> ↔ <Lesion>_Masks/<stem>.png. Only non-empty masks become cases."""
    rows = []
    for ip in sorted(glob.glob(os.path.join(root, "Original_Images", "*"))):
        if not ip.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue
        stem = os.path.splitext(os.path.basename(ip))[0]
        for t in types:
            mp = os.path.join(root, LES[t], stem + ".png")
            if os.path.exists(mp) and np.asarray(Image.open(mp).convert("L")).max() > 0:
                rows.append(dict(image_id=os.path.basename(ip), image_path=ip,
                                 mask_path=mp, lesion_type=t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgadr_root", required=True, help=".../FGADR-Seg-set/Seg-set")
    ap.add_argument("--sal_root", required=True)
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=["MA", "HE", "EX", "SE"])
    ap.add_argument("--auc_min", type=float, default=0.9)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--per_model", action="store_true", help="also compute the 21 individual models + Sensitivity")
    ap.add_argument("--consensus_norm", choices=["sum1", "minmax"], default="sum1",
                    help="per-model normalization for consensus: sum1=quantitative definition (default, recommended) | minmax=for sensitivity")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    build_cons = L.build_consensus if a.consensus_norm == "sum1" else E.build_consensus_mm
    print(f"[consensus_norm={a.consensus_norm}] {'sum-to-1 (quantitative definition)' if a.consensus_norm=='sum1' else 'min-max (sensitivity)'}")

    # 1) target images and masks (non-empty)
    items = discover(a.fgadr_root, a.types)
    if not items:
        sys.exit(f"[abort] 0 (image, mask) pairs — check paths/folder names: {a.fgadr_root}")
    by_t = {t: sum(r['lesion_type'] == t for r in items) for t in a.types}
    print(f"target lesion cases {len(items)} | per type: {by_t} | unique images {len({r['image_id'] for r in items})}")

    # 2) load saliency + consensus sets
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
    print(f"models {len(names)} | Primary consensus {len(prim)} models" + (f" | Sensitivity {len(best8)} models" if a.per_model else ""))

    # 3) per-image metrics
    per, excl, grid_acc = [], [], {}

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

    # FOV and letterbox computed once per image (shared across lesion types)
    fov_cache = {}
    for r in items:
        fn = r["image_id"]; ltype = r["lesion_type"]
        if not all(fn in fidx[m] for m in prim):
            excl.append((fn, ltype, "missing_saliency")); continue
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        fov = fov_cache[fn]
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            excl.append((fn, ltype, "empty_mask_in_FOV")); continue
        maps = [("Primary", prim)]
        if a.per_model:
            maps.append(("Sensitivity", best8))
        for tag, mods in maps:
            cons = build_cons(fn, mods, lin, ft, cams, fidx)
            if cons is not None:
                record(cons, tag, fn, ltype, Lm, fov)
        if a.per_model:
            for m in names:
                record(L.prep(cams[m][fidx[m][fn]].numpy()),
                       f"{L.backbone_pretty(m)}-{ft[m]}", fn, ltype, Lm, fov)
    per = pd.DataFrame(per)
    per.to_csv(os.path.join(a.out_dir, "fgadr_metrics_per_image.csv"), index=False)
    n_cases = int((per["map_name"] == "Primary").sum()) if len(per) else 0
    print(f"valid lesion cases = {n_cases} | unique fundus images = {per['image_id'].nunique() if len(per) else 0} (excluded {len(excl)})")
    for fn, t, why in excl[:15]:
        print("  [excl]", fn, t, why)

    # 4) summary (map × lesion_type + ALL)
    rows = []
    for tag in per["map_name"].unique():
        sub_all = per[per.map_name == tag]
        for t in a.types + ["ALL"]:
            sub = sub_all if t == "ALL" else sub_all[sub_all.lesion_type == t]
            if len(sub):
                rows.append(dict(map_name=tag, lesion_type=t,
                                 **E.summarize(sub, grid_acc.get((tag, t), []), a.nboot)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, "fgadr_localization_summary.csv"), index=False)

    # overall table of all maps (21 individual + consensus), per lesion_type
    BB_ORD = ["OpenCLIP", "DINOv2", "RETFound-DINOv2", "DINOv3-Large", "DINOv3-Huge", "MAE", "RETFound-MAE"]
    FT_ORD = {"Linear": 0, "Partial": 1, "LoRA": 2}

    def _key(nm):
        if nm == "Primary":     return (0, 0, 0)
        if nm == "Sensitivity": return (0, 1, 0)
        bb, _, ftn = nm.rpartition("-")
        return (1, BB_ORD.index(bb) if bb in BB_ORD else 99, FT_ORD.get(ftn, 9))

    cols = ["map_name", "N", "Dice_global", "IoM_global", "f_star", "Dice_oracle", "IoM_oracle",
            "PG_pct", "AUPRC_mean", "nAUPRC_mean", "otsu_iou", "top10_iou", "inout_geomean"]
    tbls = []
    for t in a.types + ["ALL"]:
        sub = out[out.lesion_type == t].copy()
        if not len(sub):
            continue
        sub["_k"] = sub["map_name"].map(_key)
        sub = sub.sort_values("_k").drop(columns="_k")
        print(f"\n===== FGADR all maps ({t}) : 21 individual + Primary/Sensitivity =====")
        print(sub[cols].to_string(index=False))
        tbls.append(sub.assign(lesion_type=t))
    pd.concat(tbls, ignore_index=True)[["lesion_type"] + cols].to_csv(
        os.path.join(a.out_dir, "fgadr_permap_table.csv"), index=False)

    print("\nSaved:", a.out_dir,
          "(fgadr_metrics_per_image.csv, fgadr_localization_summary.csv, fgadr_permap_table.csv)")


if __name__ == "__main__":
    main()
