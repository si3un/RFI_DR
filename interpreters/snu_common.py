"""
snu_common.py — shared module for SNU lesion evaluation

Contents:
  - get_mask / SEG2CODE / region_mask_448 : pseudo lesion masks built from the 8-region lesion codes (from the original SNUH evaluation code)
  - FOV / otsu
  - inout_ratio : inside/outside density ratio (area-corrected main measure)
  - iom / iou / pg : for reference only (area-biased), supplementary
  - bootstrap_ci
  - lineage_of / read_internal_auc : consensus lineage/filter (same as external consensus_overlap.py)
  - load_saliency : {out}/saliency_map.pth + used.csv → cam[N,H,W], {file_name: idx}
  - load_crop_lut / load_lesion_regions
"""
import os, json, glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

IMG_SIZE = 448
ZERO_EPS = 1e-8

# ── get_mask (from the original SNUH evaluation code) ──
try:
    from skimage.draw import disk as _disk
    def _circle(r, c, radius, shape=None):
        return _disk((r, c), radius, shape=shape)
except ImportError:
    from skimage.draw import circle as _circle


def _get_dist(coord):
    return (coord[0] ** 2 + coord[1] ** 2) ** 0.5


def _line_eqn(x_range, x1, y1, x2, y2):
    return ((y2 - y1) / (x2 - x1)) * (x_range - x1) + y1


def get_mask(h_img, w_img, dx, dy, fx, fy, d_ratio=2 / 5., f_ratio=2 / 3.):
    h_img, w_img = int(h_img), int(w_img)
    segmap = np.zeros((h_img, w_img))
    d = np.array([dy, dx]); f = np.array([fy, fx])
    dist = _get_dist([dx - fx, dy - fy])
    dr = d_ratio * dist; fr = f_ratio * dist
    a = (dr ** 2 - fr ** 2 + dist ** 2) / (2 * dist)
    m = d + a * (f - d) / dist
    m_y, m_x = m
    hh = (dr ** 2 - a ** 2) ** 0.5
    m2_x = m_x + hh * (fy - dy) / dist
    m2_y = m_y - hh * (fx - dx) / dist
    m2 = [m2_y, m2_x]; d2 = m2 - m + d; d2_y, d2_x = d2
    x_range_0 = np.arange(0, segmap.shape[1], 1)
    y_range_0 = np.arange(0, segmap.shape[0], 1)
    dcr, dcc = _circle(dy, dx, dr, shape=segmap.shape)
    fcr, fcc = _circle(fy, fx, fr, shape=segmap.shape)
    segmap[dcr, dcc] = 1
    segmap[fcr, fcc] = 3
    if dx > fx:
        x_range = np.round(_line_eqn(y_range_0, m_y, m_x, m2_y, m2_x)).astype(int)
        for x, y in zip(x_range, y_range_0):
            x = max(0, min(x, w_img)); segmap[y, x:] -= (segmap[y, x:] == 3) * 2
        x_range = np.round(_line_eqn(y_range_0, dy, dx, d2_y, d2_x)).astype(int)
        for x, y in zip(x_range, y_range_0):
            x = max(0, min(x, w_img))
            segmap[y, x:] += (segmap[y, x:] == 0) * 5
            segmap[y, :x] += (segmap[y, :x] == 0) * 7
        y_range = np.round(_line_eqn(x_range_0, dx, dy, fx, fy)).astype(int)
        for x, y in zip(x_range_0, y_range):
            y = max(0, min(y, h_img))
            for v in (1, 3, 5, 7):
                segmap[:y, x] -= (segmap[:y, x] == v) * 1
    else:
        x_range = np.round(_line_eqn(y_range_0, m_y, m_x, m2_y, m2_x)).astype(int)
        for x, y in zip(x_range, y_range_0):
            x = max(0, min(x, w_img - 1)); segmap[y, :x] -= (segmap[y, :x] == 3) * 2
        x_range = np.round(_line_eqn(y_range_0, dy, dx, d2_y, d2_x)).astype(int)
        for x, y in zip(x_range, y_range_0):
            x = max(0, min(x, w_img - 1))
            segmap[y, :x] += (segmap[y, :x] == 0) * 5
            segmap[y, x:] += (segmap[y, x:] == 0) * 7
        y_range = np.round(_line_eqn(x_range_0, dx, dy, fx, fy)).astype(int)
        for x, y in zip(x_range_0, y_range):
            y = max(0, min(y, h_img - 1))
            for v in (1, 3, 5, 7):
                segmap[:y, x] -= (segmap[:y, x] == v) * 1
    return segmap


# seg integer → region code (verified mapping)
SEG2CODE = {0: 'sd', 1: 'id', 2: 'sm', 3: 'im', 4: 'sn', 5: 'in', 6: 'st', 7: 'it'}
CODE2SEG = {v: k for k, v in SEG2CODE.items()}


def make_fov(size=IMG_SIZE):
    y, x = np.ogrid[:size, :size]
    c, r = size / 2, size * 0.49
    return (y - c) ** 2 + (x - c) ** 2 <= r ** 2


FOV = make_fov()


def region_mask_448(fname, region_code, crop_lut):
    """Lesion region codes → 448 boolean mask (get_mask + nearest resize)."""
    o = crop_lut.loc[fname]
    w_img = int(o['image_width_crop']); h_img = int(o['image_height_crop'])
    # original disc/fovea code: disc_x_crop=(disk_x-L)/image_width_crop → x is normalized by WIDTH, y by HEIGHT
    dx = o['disc_x_crop'] * w_img; fx = o['fovea_x_crop'] * w_img
    dy = o['disc_y_crop'] * h_img; fy = o['fovea_y_crop'] * h_img
    seg = get_mask(h_img, w_img, dx, dy, fx, fy)
    # letterbox: zero-pad the short side to a square (max side) (same space as the saliency preprocessing _letterbox) → no loss or distortion
    S = max(h_img, w_img); yy0 = (S - h_img) // 2; xx0 = (S - w_img) // 2
    sq = np.zeros((S, S), seg.dtype); sq[yy0:yy0 + h_img, xx0:xx0 + w_img] = seg
    seg = sq
    seg_bin = (seg == CODE2SEG[region_code]).astype(np.float32)
    t = torch.from_numpy(seg_bin)[None, None]
    r = F.interpolate(t, (IMG_SIZE, IMG_SIZE), mode='nearest')[0, 0].numpy()
    return r > 0.5


def otsu(vals, bins=256):
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float); w0 = np.cumsum(hist); total = w0[-1]
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2
    csum = np.cumsum(hist * centers); w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = csum / w0; m1 = (csum[-1] - csum) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -1
    return float(centers[np.argmax(between)])


# ═══════════════ metrics ═══════════════
def inout_ratio(cam, inside_mask, fov=FOV, eps=1e-6):
    """inside/outside density ratio (area-corrected = mean per pixel).
    Returns: (inside_density, outside_density, ratio). ratio>1 = more concentrated in the lesion region.
    ratio = (din+eps)/(dout+eps) — guarantees the same eps and the same valid set as logratio=log(ratio)
    (nan only when the mask is empty). Scale-invariant (normalization cancels out)."""
    inside = inside_mask & fov
    outside = fov & (~inside_mask)
    if inside.sum() == 0 or outside.sum() == 0:
        return np.nan, np.nan, np.nan
    din = float(cam[inside].mean())
    dout = float(cam[outside].mean())
    ratio = (din + eps) / (dout + eps)
    return din, dout, ratio


try:
    from sklearn.metrics import average_precision_score as _aps
except Exception:
    _aps = None


def nauprc(cam, inside_mask, fov=FOV):
    """Prevalence-corrected per-image AUPRC (literature formula, primary).
    nAUPRC = (AP - prev)/(1 - prev). 0=random, 1=perfect, <0=worse than random.
    Insensitive to saliency scale and ratio blow-up (rank-based PR)."""
    if _aps is None:
        return np.nan
    y = (inside_mask & fov)[fov].astype(np.uint8)
    s = cam[fov].astype(np.float64)
    prev = float(y.mean())
    if prev <= 0 or prev >= 1 or s.std() < 1e-12:
        return np.nan
    ap = float(_aps(y, s))
    return (ap - prev) / (1.0 - prev)


def topk_enrich_fixed(cam, inside_mask, fov=FOV, q=0.10):
    """Fixed-area top-q% enrichment: P(lesion|top-q%)/prevalence.
    Selects exactly the top q% pixels by rank → same area for every model → controls for sparsity.
    E=1 random, E>1 lesion enrichment."""
    s = cam[fov].astype(np.float64)
    y = (inside_mask & fov)[fov].astype(np.uint8)
    prev = float(y.mean())
    if prev <= 0 or s.size == 0:
        return np.nan
    k = max(1, int(round(q * s.size)))
    idx = np.argpartition(-s, k - 1)[:k]
    return float(y[idx].mean() / prev)


def iom(a_bin, g_bin):
    inter = np.logical_and(a_bin, g_bin).sum()
    denom = min(a_bin.sum(), g_bin.sum())
    return float(inter / denom) if denom > 0 else np.nan


def iou(a_bin, g_bin):
    inter = np.logical_and(a_bin, g_bin).sum()
    union = np.logical_or(a_bin, g_bin).sum()
    return float(inter / union) if union > 0 else np.nan


def bootstrap_ci(x, n_boot=2000, ci=95, seed=42):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    a = (100 - ci) / 2
    return (float(x.mean()), float(np.percentile(boots, a)),
            float(np.percentile(boots, 100 - a)), int(x.size))


# ═══════════════ consensus lineage/filter (same as external) ═══════════════
def lineage_of(name):
    n = name.lower()
    if "clip" in n:
        return "CLIP"
    if "dinov3" in n:
        return "DINOv3"
    if "dinov2" in n:           # includes retfound_dinov2
        return "DINOv2"
    if "mae" in n or "retfound" in n:   # mae, retfound_mae
        return "MAE"
    return None


def read_internal_auc(model_dir):
    """Internal AUC (0_auc) from test_metrics.json. nan if missing."""
    p = os.path.join(model_dir, "test_metrics.json")
    if not os.path.exists(p):
        return np.nan
    try:
        with open(p) as f:
            d = json.load(f)
        for k in ("0_auc", "auc", "test_auc"):
            if k in d:
                v = d[k]
                if isinstance(v, str):
                    v = float(v.split()[0])
                return float(v)
    except Exception:
        pass
    return np.nan


# ═══════════════ loading ═══════════════
def load_saliency(sal_dir):
    """{sal_dir}/saliency_map.pth [N,H,W] + used.csv(file_name) → cam, {file_name: idx}."""
    cam = torch.load(os.path.join(sal_dir, "saliency_map.pth"), map_location="cpu").float()
    used = pd.read_csv(os.path.join(sal_dir, "used.csv"))
    if "file_name" in used.columns:
        fnames = used["file_name"].astype(str).tolist()
    else:  # if only image_path exists, use its basename
        fnames = [os.path.basename(p) for p in used["image_path"].astype(str)]
    return cam, {fn: i for i, fn in enumerate(fnames)}


def load_crop_lut(crop_csv):
    return pd.read_csv(crop_csv).drop_duplicates('file_name').set_index('file_name')


def load_lesion_regions(lesion_csv, min_agree=0.0):
    """Lesion CSV → {file_name: [region,...]} (agreement>=min_agree). Also returns the original df."""
    les = pd.read_csv(lesion_csv)
    if min_agree > 0:
        les = les[les['agreement'] >= min_agree]
    img_regions = les.groupby('file_name')['region'].apply(lambda s: sorted(set(s))).to_dict()
    return img_regions, les
