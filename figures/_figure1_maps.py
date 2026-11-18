"""Schematic saliency maps in the look of real ViT attention maps (broad, patchy, whole-field), for Figure 1.
Defines: PATCHY_SHOWN (5 shown models), PATCHY_CONS (consensus of 14), LES (lesion field),
render_patchy(map) -> RGBA disc, SPEC colormap. Exec'd by make_figure1.py after the setup of _figure1_layout.py."""
from matplotlib.colors import LinearSegmentedColormap

SPEC = LinearSegmentedColormap.from_list("spec", ["#CFC6E6", "#B5DCE3", "#C9E79A", "#F3E27C", "#F5A95B", "#DF5A3C", "#A9262C"])
PG = 28                                                     # ViT-like patch grid

def quantize(m, sig=3.0):
    """Average a map over the ViT patch grid and soften the patch edges (blocky attention look)."""
    small = np.asarray(Image.fromarray(np.asarray(m, np.float32)).resize((PG, PG), Image.BOX))
    big = np.asarray(Image.fromarray(small).resize((N, N), Image.NEAREST))
    return ndimage.gaussian_filter(big, sig)


def les_field(wc):
    """Broad, patch-quantised attention on the lesion clusters, cluster c weighted by wc[c]."""
    f = ndimage.gaussian_filter(sum(w_ * c for w_, c in zip(wc, clusters)), 16)
    f = quantize(f)
    return f / quantize(ndimage.gaussian_filter(sum(clusters), 16)).max()


LES = les_field(np.ones(KC))
print("lesion clusters:", KC)


def patchnoise(seed, sigma_p=1.1, power=1.8):
    r = np.random.default_rng(seed).standard_normal((PG, PG))
    r = ndimage.gaussian_filter(r, sigma_p)
    big = np.asarray(Image.fromarray(r.astype(np.float32)).resize((N, N), Image.NEAREST))
    big = ndimage.gaussian_filter(big, 3.0)                  # soft-edged patches
    big = (big - big.min()) / (big.max() - big.min())
    return big ** power


rr = np.hypot(xx - N / 2 + 0.5, yy - N / 2 + 0.5) / (N / 2)
ang = np.arctan2(yy - N / 2, xx - N / 2)


def edge(seed):
    rng_ = np.random.default_rng(seed)
    mod = 0.5 + 0.5 * np.cos(ang * rng_.integers(2, 5) + rng_.uniform(0, 6.3))
    return np.exp(-((rr - 0.93) / 0.06) ** 2) * mod


SHARED = patchnoise(999, sigma_p=1.6, power=1.2)            # attention shared by all models (anatomy)


def mk_map(w, seed, e=0.0, spot=None, wc=None):
    """w: weight of lesion attention; wc: per-cluster weights (a model may miss some lesions)."""
    L = LES if wc is None else les_field(wc)
    m = w * L + (1 - w) * patchnoise(seed) + 0.22 * SHARED + e * edge(seed + 50)
    if spot is not None:
        m = m + spot
    return l1(m + 0.02)


_r = np.random.default_rng(21)


def misses(k_miss, lo=0.55):
    wc = _r.uniform(lo, 1.0, KC)
    wc[_r.choice(KC, k_miss, replace=False)] *= 0.1
    return wc


PATCHY_SHOWN = [
    mk_map(0.60, 301, e=0.25, wc=misses(1)),                        # Model 1
    mk_map(0.74, 302, wc=misses(1, lo=0.70)),                        # Model 2 (closest to the consensus)
    mk_map(0.50, 303, e=0.15, spot=quantize(blob(1100, 1200, 34, 0.6)), wc=misses(2)),   # Model 3
    mk_map(0.58, 304, e=0.12, wc=misses(1, lo=0.7)),                # Model 4
    mk_map(0.15, 305, e=0.40, spot=quantize(blob(*DISC, 38, 1.0))),          # Model 5 (optic disc, periphery)
]
_rng = np.random.default_rng(7)
_extra = [mk_map(_rng.uniform(0.60, 0.90), 400 + i, e=_rng.uniform(0, 0.3), wc=misses(1)) for i in range(9)]
_pool = np.array(PATCHY_SHOWN + _extra)
PATCHY_CONS = np.sort(_pool, axis=0)[2:-2].mean(0)

_sp = np.array([spearmanr(m[fov], PATCHY_CONS[fov])[0] for m in PATCHY_SHOWN])
_ov = np.array([(topk(m) & topk(PATCHY_CONS)).sum() / topk(m).sum() for m in PATCHY_SHOWN])
_les = lesion & fov
_dice = []
for m in PATCHY_SHOWN + [PATCHY_CONS]:
    t = topk(m, _les.sum() / fov.sum() * 3)
    _dice.append(2 * (t & _les).sum() / (t.sum() + _les.sum()))
print("patchy Spearman", _sp.round(3), "top-k", _ov.round(3), "lesion overlap (M1-5, cons)", np.round(_dice, 3))

FUNDUS_BRIGHT = np.clip((fundus_hr - 0.02) * 1.40, 0, 1) ** 0.85     # lighter photo, as in the reference


def scale(m, gamma=0.9):
    m = ndimage.gaussian_filter(np.asarray(m, float), 2.0)
    m = m - m[fov].min()
    return np.clip(m / np.quantile(m[fov], 0.995), 0, 1) ** gamma


def render_patchy(m, a_lo=0.42, a_hi=0.72):
    s_ = np.clip(up(scale(m)), 0, 1)
    a = (a_lo + (a_hi - a_lo) * s_)[..., None]
    rgb = FUNDUS_BRIGHT * (1 - a) + SPEC(s_)[..., :3] * a
    return np.dstack([rgb, fov_hr])
