"""
eophtha_localization.py — external validation of saliency localization against E-Ophtha pixel-level lesion GT.

Reuses the **same metrics, preprocessing, consensus and bootstrap** as snu_localization.py,
replacing only the evaluation mask: SNU 8-region coarse → E-Ophtha **pixel masks (converted to the 448 letterbox)**.
Reported separately per lesion type (EX = hard exudates / MA = microaneurysms).

Metrics (per image, same as snu_localization):
  ① nAUPRC = (AUPRC−a)/(1−a),  a=|L∩FOV|/|FOV|   (chance-corrected)
  ② Pointing Game = argmax_FOV(saliency) ∈ L
  ③ Inside/Outside = mean_in / mean_out
  ④ IoU(Otsu, top5/10/20)  ※ MA is extremely small, so IoU≈0 is expected — interpret mainly via PG and nAUPRC

Preprocessing: saliency = relu → FOV → sum=1 (reuses snu_localization.prep).
        mask = original letterbox (zero-pad short side → square) → 448 NEAREST → intersect with FOV
               (same space as snu_gen_saliency_attn._letterbox).

Usage:
  python interpreters/eophtha_localization.py \
    --eophtha_root $EOPHTHA_ROOT \
    --sal_root  outputs/eophtha_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux \
    --out_dir   outputs/eophtha_localization --types EX MA
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, binary_closing, binary_fill_holes, label as cc_label
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L          # reuse prep / metrics / build_consensus / boot_ci / boot_geomean / ft_of / backbone_pretty

IMG = 448
SPEC = {"EX": ("e_optha_EX/EX", "e_optha_EX/Annotation_EX", "_EX"),
        "MA": ("e_optha_MA/MA", "e_optha_MA/Annotation_MA", "")}   # MA masks have no suffix (<stem>.png)


def letterbox_mask(path):
    """original mask → (zero-pad short side to square) → 448 NEAREST → bool. Same space as the saliency image preprocessing.
    Adaptive threshold: robust to both 0/255 and 0/1 (IDRiD .tif). NEAREST, so no intermediate values."""
    m = Image.open(path).convert("L")
    W, H = m.size
    S = max(W, H)
    c = Image.new("L", (S, S), 0)
    c.paste(m, ((S - W) // 2, (S - H) // 2))
    c = c.resize((IMG, IMG), Image.NEAREST)
    arr = np.array(c)
    mx = int(arr.max())
    return arr > (mx * 0.5) if mx > 0 else np.zeros(arr.shape, bool)  # robust to 0/76, 0/1, 0/255; 0/255 behaves as before


def letterbox_img(path):
    """original image → (zero-pad short side to square) → 448 BILINEAR → float[0,1] RGB. Same geometry as the masks."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    S = max(W, H)
    c = Image.new("RGB", (S, S), (0, 0, 0))
    c.paste(im, ((S - W) // 2, (S - H) // 2))
    c = c.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(c).astype(np.float32) / 255.0


def fundus_mask_448(orig):
    """actual fundus region (excluding black margins) of the 448 letterbox original — **same** as eophtha_qual_page.
    Removes saliency noise from letterbox black bars / cropped borders from the evaluation FOV (#1 per-image FOV fix)."""
    m = orig.mean(2) > 0.06
    m = binary_closing(m, iterations=3)
    m = binary_fill_holes(m)
    lab, n = cc_label(m)
    if n > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        m = lab == cnt.argmax()
    return binary_erosion(m, iterations=4)


def discover(root, types):
    rows = []
    for t in types:
        idir, adir, suf = SPEC[t]
        for ip in glob.glob(os.path.join(root, idir, "*", "*")):
            if not ip.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(os.path.basename(ip))[0]
            case = os.path.basename(os.path.dirname(ip))
            mp = os.path.join(root, adir, case, stem + suf + ".png")
            if os.path.exists(mp):
                rows.append(dict(image_id=os.path.basename(ip), image_path=ip,
                                 mask_path=mp, lesion_type=t))
    return rows


# ── two consensus normalizations (use minmax to score the same map as the qualitative page) ──
def mm(cam):
    """min-max [0,1] over FOV — same as snu_qual_viz.mm (qualitative page method)."""
    v = np.maximum(cam, 0.0) * C.FOV
    vals = v[C.FOV]
    mn, mx = float(vals.min()), float(vals.max())
    return (v - mn) / (mx - mn + 1e-8) * C.FOV if mx > mn else v * 0.0


def build_consensus_mm(fn, mods, lin, ft, cams, fidx):
    """FT-balanced consensus, mean after per-model min-max (same as snu_qual_viz.build_cons_disp)."""
    by = {}
    for m in mods:
        if fn not in fidx[m]:
            return None
        by.setdefault(lin[m], {}).setdefault(ft[m], []).append(mm(cams[m][fidx[m][fn]].numpy()))
    lin_maps = []
    for g, ftd in by.items():
        lin_maps.append(np.mean([np.mean(v, axis=0) for v in ftd.values()], axis=0))
    return mm(np.mean(lin_maps, axis=0))


# ── Ayhan 2022: top f_ν fraction of |saliency| as salient region S(f), Dice/IoM sweep ──
FGRID = np.geomspace(5e-4, 1.0, 20)   # 20 log-scale points (0.0005~1)


def dice_iom_sweep(A, L_, fov):
    """A: saliency, L_: bool lesion, fov: bool. At each f_ν, S = **exactly the top N=round(f·|FOV|)** pixels within the FOV.
    Rank-based top-N instead of a quantile threshold (exact fraction f even with many ties/zeros; point 4 fix).
    return (dice[20], iom[20]). Always read IoM at the f chosen by Dice (at f=1, IoM≡1 is meaningless)."""
    idx = np.flatnonzero(fov.ravel())
    s = A.ravel()[idx]
    Lf = (L_ & fov).ravel()[idx].astype(bool)
    ltot = int(Lf.sum())
    if ltot == 0:
        return None, None
    n = s.size
    order = np.argsort(s, kind="stable")[::-1]     # descending; ties deterministic via stable sort
    csum = np.cumsum(Lf[order].astype(np.int64))   # cumulative GT intersection among the top N
    dice = np.full(len(FGRID), np.nan)
    iom = np.full(len(FGRID), np.nan)
    for i, f in enumerate(FGRID):
        N = max(1, int(round(f * n)))
        inter = int(csum[N - 1])
        dice[i] = 2.0 * inter / (N + ltot)
        iom[i] = inter / min(N, ltot)
    return dice, iom


def global_fstar(grid_list):
    """dataset-optimal f* (Ayhan style): single f maximizing image-mean Dice → (Dice*, IoM@f*, f*).
    If f*=1, IoM≡1 (selecting the whole FOV is meaningless), so IoM is set to NaN."""
    grid_list = [g for g in grid_list if g is not None]
    if not grid_list:
        return (np.nan, np.nan, np.nan)
    D = np.vstack([g[0] for g in grid_list])   # [n,20]
    I = np.vstack([g[1] for g in grid_list])
    dmean = np.nanmean(D, axis=0)
    if not np.isfinite(dmean).any():
        return (np.nan, np.nan, np.nan)
    fi = int(np.nanargmax(dmean))
    f = float(FGRID[fi])
    iom = np.nan if f >= 0.999 else float(np.nanmean(I[:, fi]))   # f*=1 → IoM not reported
    return float(dmean[fi]), iom, f


def summarize(sub, grid_list, nboot):
    rap = L.boot_ci(sub["raw_AUPRC"].values, np.mean, nboot)   # Arun 2021: pixel AUPRC
    nap = L.boot_ci(sub["nAUPRC"].values, np.mean, nboot)      # chance-corrected
    pg = L.boot_ci(sub["pointing_game"].values, np.mean, nboot)  # Saporta 2022: hit rate
    rio = L.boot_geomean(sub["inside_outside"].values, nboot)
    io = L.boot_ci(sub["otsu_iou"].values, np.mean, nboot)     # Saporta 2022: IoU
    i10 = L.boot_ci(sub["top10_iou"].values, np.mean, nboot)
    dgstar, igstar, fstar = global_fstar(grid_list)            # Ayhan main: dataset-optimal f* (dataset-optimized)
    db = L.boot_ci(sub["dice_best"].values, np.mean, nboot)    # oracle: per-image optimal f using GT (upper bound, supplementary)
    ib = L.boot_ci(sub["iom_best"].values, np.mean, nboot)     # IoM at the per-image Dice-optimal f
    return dict(N=len(sub),
                Dice_global=round(dgstar, 3), IoM_global=round(igstar, 3), f_star=round(fstar, 4),
                Dice_oracle=round(db[0], 3), Dice_oracle_CI=f"[{db[1]:.3f}, {db[2]:.3f}]",
                IoM_oracle=round(ib[0], 3),
                AUPRC_mean=round(rap[0], 3), AUPRC_CI=f"[{rap[1]:.3f}, {rap[2]:.3f}]",
                nAUPRC_mean=round(nap[0], 3), nAUPRC_CI=f"[{nap[1]:.3f}, {nap[2]:.3f}]",
                PG_n=int(sub["pointing_game"].sum()), PG_pct=round(100 * pg[0], 1),
                PG_CI=f"[{100*pg[1]:.1f}, {100*pg[2]:.1f}]",
                inout_geomean=round(rio[0], 3), inout_CI=f"[{rio[1]:.3f}, {rio[2]:.3f}]",
                otsu_iou=round(io[0], 3), top10_iou=round(i10[0], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eophtha_root", required=True)
    ap.add_argument("--sal_root", required=True)
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--types", nargs="+", default=["EX", "MA"])
    ap.add_argument("--auc_min", type=float, default=0.9)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--per_model", action="store_true", help="also compute the 21 individual models + Sensitivity")
    ap.add_argument("--consensus_norm", choices=["sum1", "minmax"], default="sum1",
                    help="per-model normalization for consensus: sum1=quantitative definition (default, recommended) | minmax=for sensitivity")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    build_cons = L.build_consensus if a.consensus_norm == "sum1" else build_consensus_mm
    print(f"[consensus_norm={a.consensus_norm}] {'sum-to-1 (quantitative definition)' if a.consensus_norm=='sum1' else 'min-max (sensitivity)'}")

    # 1) Target images and masks
    items = discover(a.eophtha_root, a.types)
    ids = [r["image_id"] for r in items]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dup = [k for k, v in Counter(ids).items() if v > 1]
        print(f"[warning] {len(dup)} duplicate image_ids (EX/MA basename collision) — e.g.: {dup[:3]}")
    print(f"targets {len(items)} images | by type:",
          {t: sum(r['lesion_type'] == t for r in items) for t in a.types})

    # 2) Load saliency + build consensus (same as snu_localization)
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

    # 3) Per-image metrics
    per = []
    excl = []
    grid_acc = {}   # (map_name, lesion_type or "ALL") -> [ (dice[20], iom[20]) ]  : for computing dataset f*

    def record(A, tag, fn, ltype, Lm, fov):
        met = L.metrics(A, Lm, fov)
        if not met:
            return
        dg, ig = dice_iom_sweep(A, Lm, fov)
        rec = dict(image_id=fn, lesion_type=ltype, map_name=tag, **met)
        if dg is not None and np.isfinite(dg).any():
            bi = int(np.nanargmax(dg))                     # per-image Dice-optimal f (oracle)
            fb = float(FGRID[bi])
            rec["dice_best"] = float(dg[bi]); rec["f_best"] = fb
            rec["iom_best"] = np.nan if fb >= 0.999 else float(ig[bi])   # f=1 → IoM not reported
            grid_acc.setdefault((tag, ltype), []).append((dg, ig))
            grid_acc.setdefault((tag, "ALL"), []).append((dg, ig))
        else:
            rec["dice_best"] = np.nan; rec["iom_best"] = np.nan; rec["f_best"] = np.nan
        per.append(rec)

    for r in items:
        fn = r["image_id"]; ltype = r["lesion_type"]
        if not all(fn in fidx[m] for m in prim):
            excl.append((fn, ltype, "missing_saliency")); continue
        fov = C.FOV & fundus_mask_448(letterbox_img(r["image_path"]))   # #1 per-image FOV (fundus ∩ excluding black margins)
        Lm = letterbox_mask(r["mask_path"]) & fov
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
    per.to_csv(os.path.join(a.out_dir, "eophtha_metrics_per_image.csv"), index=False)
    n_cases = int((per["map_name"] == "Primary").sum())
    print(f"valid lesion cases = {n_cases} | unique fundus images = {per['image_id'].nunique()} (excluded {len(excl)})")
    for fn, t, why in excl[:15]:
        print("  [excl]", fn, t, why)

    # 4) Summary: (map × lesion_type) + overall per map
    rows = []
    for tag in per["map_name"].unique():
        sub_all = per[per.map_name == tag]
        for t in a.types + ["ALL"]:
            sub = sub_all if t == "ALL" else sub_all[sub_all.lesion_type == t]
            if len(sub):
                rows.append(dict(map_name=tag, lesion_type=t,
                                 **summarize(sub, grid_acc.get((tag, t), []), a.nboot)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, "eophtha_localization_summary.csv"), index=False)

    # ── Combined table of all maps (21 individual + 2 consensus): per lesion_type, sorted consensus→lineage→FT ──
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
        print(f"\n===== all maps combined ({t}) : 21 individual + Primary/Sensitivity =====")
        print(sub[cols].to_string(index=False))
        tbls.append(sub.assign(lesion_type=t))
    pd.concat(tbls, ignore_index=True)[["lesion_type"] + cols].to_csv(
        os.path.join(a.out_dir, "eophtha_permap_table.csv"), index=False)

    print("\nSaved:", a.out_dir,
          "(eophtha_metrics_per_image.csv, eophtha_localization_summary.csv, eophtha_permap_table.csv)")


if __name__ == "__main__":
    main()
