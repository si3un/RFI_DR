"""
baseline_avgmask_patchceiling.py - computes the 2 recommendations from the literature review (compass) with the Table 3 protocol (per-case optimal top-k, grid step 0.01%).

  (1) Average-mask baseline (Arun et al. 2021): the mean GT mask of the other images of the same dataset and lesion type (leave-one-out) is
      treated as saliency → per-case optimal Dice/IoU. Consensus must exceed this floor to claim localization utility.
  (2) Patch-resolution ceiling: optimal Dice of the oracle patch mask that best covers the GT at patch resolution (14 px, 16 px).
      Because ViT saliency comes from the patch grid, no individual/consensus map can in principle easily exceed this value.
  (+) Random-rank reference: optimal Dice of uniform random saliency (seed 42) (lower-bound reference).

Reads: dataset root (same as threshold_sweep_erx.DISCOVER); no saliency needed.
Output: {out_dir}/{ds}_baseline_per_case.csv  (dataset,image_id,lesion_type,method,optimal_k,optimal_dice,optimal_iou,gt_frac)
        {out_dir}/baseline_summary.csv        (dataset,lesion_type,method,n,mean_dice,lo,hi,mean_iou)
Existing threshold_analysis_* folders are not touched.

Usage:
  python interpreters/baseline_avgmask_patchceiling.py --dataset eophtha --root $EOPHTHA_ROOT --out_dir outputs/baseline_v1
  python interpreters/baseline_avgmask_patchceiling.py --dataset idrid   --root $IDRID_SEG_ROOT --out_dir outputs/baseline_v1
  python interpreters/baseline_avgmask_patchceiling.py --dataset fgadr   --root $FGADR_SEG_ROOT --out_dir outputs/baseline_v1
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import eophtha_localization as E
from threshold_sweep_erx import DISCOVER, DEF_TYPES
import threshold_optimal_k as T

IMG = 448


def boot_ci(x, n=2000, seed=42):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def optimal(A, Lm, fov):
    r = T.sweep_curves(A, Lm, fov)
    if r is None: return None
    dice, iou, G, n = r
    j = int(np.argmax(dice))          # on ties, the smallest k (first argmax)
    return float(T.KS[j]), float(dice[j]), float(iou[j]), G / n


def patch_ceiling(Lm, fov, p):
    """oracle patch mask: sort patches by GT pixel count, add them one by one, and take the max Dice (GT within the FOV)."""
    L = (Lm & fov).astype(np.int64)
    H = IMG // p; L2 = L[:H * p, :H * p].reshape(H, p, H, p).sum(axis=(1, 3))     # GT pixel count per patch
    F2 = fov[:H * p, :H * p].reshape(H, p, H, p).sum(axis=(1, 3))                 # FOV pixel count per patch
    G = int(L.sum())
    if G == 0: return None
    order = np.argsort(L2.ravel(), kind="stable")[::-1]
    inter = np.cumsum(L2.ravel()[order]); area = np.cumsum(F2.ravel()[order])
    dice = 2.0 * inter / (area + G); iou = inter / (area + G - inter)
    j = int(np.argmax(dice))
    return float(area[j] / fov.sum() * 100), float(dice[j]), float(iou[j])


def patch_frac_map(Lm, fov, p, scale):
    """perfect saliency at one generation scale: per-patch GT fraction on the p-px grid of the scaled image,
    mapped back to 448 exactly as the saliency generator does (bilinear), then min-max in FOV."""
    from scipy.ndimage import zoom
    L = (Lm & fov).astype(np.float64)
    S = int(round(scale * IMG))
    if S != IMG:
        L = zoom(L, S / IMG, order=0)
    H = S // p
    F = L[:H * p, :H * p].reshape(H, p, H, p).mean(axis=(1, 3))
    up = np.zeros((S, S), np.float64); up[:H * p, :H * p] = np.repeat(np.repeat(F, p, 0), p, 1)
    if S != IMG:
        up = zoom(up, IMG / S, order=1)[:IMG, :IMG]
    up = up * fov
    mx = up.max()
    return up / mx if mx > 0 else up


def patch_ceiling_ms(Lm, fov, p, scales):
    """perfect score using the multi-scale saliency resolution: average of the per-scale perfect maps (as the
    generator averages 1.0x and 1.5x), evaluated with the identical per-case optimal top-k sweep."""
    A = np.mean([patch_frac_map(Lm, fov, p, sc) for sc in scales], axis=0)
    return optimal(A, Lm, fov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eophtha", "idrid", "fgadr"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default="outputs/baseline_v1")
    ap.add_argument("--patch_sizes", nargs="*", type=int, default=[14, 16])
    ap.add_argument("--limit", type=int, default=0, help="for debugging: first N cases only")
    ap.add_argument("--ms_scales", nargs="*", type=float, default=[1.0, 1.5], help="multi-scale perfect score: generation scales")
    ap.add_argument("--ms_patch", type=int, default=14)
    ap.add_argument("--ms_only", action="store_true", help="compute only the multi-scale perfect score")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    T.KS = T.make_grid("ultrafine", 0.01, 99.0, 0.01)          # same grid as Table 3, step 0.01% (9,900 points)
    ds = a.dataset
    items = DISCOVER[ds](a.root, DEF_TYPES[ds])
    if a.limit: items = items[:a.limit]
    print("%s: %d cases (image x lesion)" % (ds, len(items)))
    # 1) load masks and FOV
    fov_cache, masks = {}, []
    for r in items:
        fn = r["image_id"]
        if fn not in fov_cache:
            fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(r["image_path"]))
        masks.append((E.letterbox_mask(r["mask_path"]) & fov_cache[fn]).astype(np.float32))
    # 2) per-lesion mask sums (for the leave-one-out mean)
    lts = sorted(set(r["lesion_type"] for r in items))
    sums = {lt: np.zeros((IMG, IMG), np.float64) for lt in lts}; cnt = {lt: 0 for lt in lts}
    for r, m in zip(items, masks): sums[r["lesion_type"]] += m; cnt[r["lesion_type"]] += 1
    rng = np.random.default_rng(42)
    rows = []
    for idx, (r, m) in enumerate(zip(items, masks)):
        fn, lt = r["image_id"], r["lesion_type"]; fov = fov_cache[fn]; Lm = m.astype(bool)
        if Lm.sum() == 0: continue
        if a.ms_scales:
            o = patch_ceiling_ms(Lm, fov, a.ms_patch, a.ms_scales)
            if o: rows.append(dict(dataset=ds, image_id=fn, lesion_type=lt, method="PatchCeiling-MS%d(%s)" % (a.ms_patch, "+".join("%g" % x for x in a.ms_scales)),
                                   optimal_k=o[0], optimal_dice=o[1], optimal_iou=o[2], gt_frac=o[3]))
        if a.ms_only:
            if (idx + 1) % 200 == 0: print("  %d / %d" % (idx + 1, len(items)), flush=True)
            continue
        if cnt[lt] > 1:
            avg = (sums[lt] - m) / (cnt[lt] - 1)
            o = optimal(avg, Lm, fov)
            if o: rows.append(dict(dataset=ds, image_id=fn, lesion_type=lt, method="AverageMask-LOO", optimal_k=o[0], optimal_dice=o[1], optimal_iou=o[2], gt_frac=o[3]))
        o = optimal(rng.random((IMG, IMG)), Lm, fov)
        if o: rows.append(dict(dataset=ds, image_id=fn, lesion_type=lt, method="RandomRank", optimal_k=o[0], optimal_dice=o[1], optimal_iou=o[2], gt_frac=o[3]))
        for p in a.patch_sizes:
            c = patch_ceiling(Lm, fov, p)
            if c: rows.append(dict(dataset=ds, image_id=fn, lesion_type=lt, method="PatchCeiling-%d" % p, optimal_k=c[0], optimal_dice=c[1], optimal_iou=c[2], gt_frac=float(Lm.sum() / fov.sum())))
        if (idx + 1) % 200 == 0: print("  %d / %d" % (idx + 1, len(items)), flush=True)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(a.out_dir, "%s_baseline_per_case.csv" % ds), index=False)
    # 3) summary (ALL + per lesion)
    S = []
    for meth in sorted(set(D.method)):
        for lt in ["ALL"] + lts:
            sub = D[(D.method == meth) & ((D.lesion_type == lt) if lt != "ALL" else True)]
            if len(sub) == 0: continue
            lo, hi = boot_ci(sub.optimal_dice)
            S.append(dict(dataset=ds, lesion_type=lt, method=meth, n=len(sub), mean_dice=sub.optimal_dice.mean(), lo=lo, hi=hi,
                          mean_iou=sub.optimal_iou.mean(), median_k=sub.optimal_k.median()))
    S = pd.DataFrame(S)
    fs = os.path.join(a.out_dir, "baseline_summary.csv")
    if os.path.exists(fs):
        old = pd.read_csv(fs); S = pd.concat([old[old.dataset != ds], S], ignore_index=True)
    S.to_csv(fs, index=False)
    print(S[S.dataset == ds].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
