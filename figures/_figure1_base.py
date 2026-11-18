"""Figure 1 draft (direction C): three-step infographic of the CoSA idea.

1  Many models explain the same image   (fundus + 5 model maps, each with its own error)
2  Where they agree marks the lesions    (consensus map: trimmed mean of 14 maps)
3  Agreement ranks the models            (models ranked by agreement with the consensus)

Fundus photograph: e-Ophtha C0014759 (cite Decenciere et al., IRBM 2013).
Saliency maps are SCHEMATIC: shared hot spots are placed on the annotated exudate clusters,
plus one model-specific spurious spot per model. The consensus uses the actual rule
(L1 normalisation, per-pixel trim 2+2, mean of 10) and agreement is the Spearman correlation
with the consensus inside the field of view, both computed on these schematic maps.
Helper for make_figure1.py, which uses only its set-up part (fundus photograph, masks, schematic maps).
Run on its own it draws an earlier draft (Figure1_C_draft.png / .pdf).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EOPHTHA = os.environ.get("EOPHTHA_ROOT", os.path.join(ROOT, "E-Ophtha"))   # public e-Ophtha dataset folder
IMG = os.path.join(EOPHTHA, "e_optha_EX", "EX", "E0020333", "C0014759.jpg")
MSK = os.path.join(EOPHTHA, "e_optha_EX", "Annotation_EX", "E0020333", "C0014759_EX.png")

plt.rcParams["font.family"] = "Arial"
K, DG, LG, ACC = "#000000", "#555555", "#BDBDBD", "#E07B00"
N = 480

# ------------------------------------------------------------------ fundus + lesion clusters
im = np.asarray(Image.open(IMG).convert("RGB")).astype(float) / 255
mk = np.asarray(Image.open(MSK))
mk = (mk[..., 0] if mk.ndim == 3 else mk) > 0
H, W = mk.shape
cx, cy, half = 1272, 848, 765          # full circle inside the photograph


def crop(a):
    out = np.zeros((2 * half, 2 * half) + a.shape[2:])
    y0, x0 = cy - half, cx - half
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(H, cy + half), min(W, cx + half)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def down(a):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((N, N), Image.LANCZOS)) / 255


fundus = down(crop(im))
inside = down(crop(np.ones((H, W)))) > 0.5
yy, xx = np.mgrid[0:N, 0:N]
circle = (xx - N / 2 + 0.5) ** 2 + (yy - N / 2 + 0.5) ** 2 <= (N / 2 - 1) ** 2
fov = ndimage.binary_fill_holes(circle & inside & (ndimage.gaussian_filter(fundus.mean(-1), 2) > 0.05))
lesion = down(crop(mk.astype(float))) > 0.02
lab, nlab = ndimage.label(ndimage.binary_dilation(lesion, iterations=6))
clusters = []
for i in range(1, nlab + 1):
    c = (lab == i) & lesion
    if c.sum() >= 4:
        g = ndimage.gaussian_filter(c.astype(float), 13)
        clusters.append(g / g.max())
KC = len(clusters)


def to_xy(px, py):
    return (px - (cx - half)) * N / (2 * half), (py - (cy - half)) * N / (2 * half)


def blob(px, py, s, a=1.0):
    x, y = to_xy(px, py)
    return a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s ** 2))


def field(seed, amp):
    r = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal((N, N)), 14)
    return amp * (r - r.min()) / (r.max() - r.min())


def l1(s):
    s = np.clip(s, 0, None) * fov
    return s / s.sum()


# ------------------------------------------------------------------ 14 schematic model maps
DISC = (1600, 800)
rng = np.random.default_rng(11)
SPUR = [DISC, (1950, 1350), (700, 1300), (1250, 250), (2050, 650), (900, 1500), (1900, 300),
        (500, 800), (1500, 1550), (2150, 1000), (1100, 1200), (1700, 500), (650, 450), (1350, 1450)]
pool = []
for i in range(14):
    w = rng.uniform(0.5, 1.0, KC)
    if i % 4 == 0:
        w[rng.integers(0, KC)] *= 0.1                       # misses one lesion
    s = sum(wk * c for wk, c in zip(w, clusters))
    s = s + blob(*SPUR[i], rng.uniform(18, 26), rng.uniform(0.8, 1.2))   # model-specific error
    pool.append(l1(s + field(100 + i, 0.10)))
# one clearly poor model: weak lesion signal, strong errors (optic disc, periphery)
poor = 0.25 * sum(clusters) + blob(*DISC, 24, 1.3) + blob(2000, 1450, 26, 1.1) + field(199, 0.25)
pool[13] = l1(poor)
pool = np.array(pool)
consensus = np.sort(pool, axis=0)[2:-2].mean(0)              # per-pixel trim 2+2, mean of 10
show = lambda m: m / m.max()
agree = np.array([spearmanr(m[fov], consensus[fov])[0] for m in pool])

jet = plt.get_cmap("jet")


def overlay(sal, amax=0.8):
    sal = show(sal) if sal.max() > 0 else sal
    a = np.clip((sal - 0.15) / 0.5, 0, 1)[..., None] * amax
    rgb = fundus * (1 - a) + jet(sal)[..., :3] * a
    return np.dstack([rgb, fov.astype(float)])


# ------------------------------------------------------------------ canvas
FW, FH = 16.0, 6.0
fig = plt.figure(figsize=(FW, FH), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW * 10)
ax.set_ylim(0, FH * 10)
ax.axis("off")


def T(x, y, s, size=11, color=K, weight="normal", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def disc(img, x, y, d, z=3, ring=None, lw=1.6):
    ax.imshow(img, extent=[x - d / 2, x + d / 2, y - d / 2, y + d / 2], zorder=z, interpolation="bilinear")
    if ring:
        ax.add_patch(Circle((x, y), d / 2 * 0.98, fc="none", ec=ring, lw=lw, zorder=z + 0.1))


def arrow(x0, y0, x1, y1, ms=14, lw=1.4, color=K):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def step(x, n, title):
    ax.add_patch(Circle((x + 1.9, 54.5), 1.9, fc=ACC, ec="none", zorder=5))
    T(x + 1.9, 54.45, str(n), 12, "white", "bold", zorder=6)
    T(x + 4.9, 54.5, title, 13.5, ha="left")


# 1 ---------------------------------------------------------------------------------------
step(2.0, 1, "Many models explain the same image")
c1 = np.array([26.5, 29.0])
disc(np.dstack([fundus, fov.astype(float)]), *c1, 15.0)
srt = list(np.argsort(-agree))
show_idx = [srt[0], srt[3], srt[6], srt[10], 13]
R, dm = 15.6, 10.5
for j, i in enumerate(show_idx):
    ang = np.deg2rad(90 + j * 72)
    p = c1 + R * np.array([np.cos(ang), np.sin(ang)])
    u = (p - c1) / np.linalg.norm(p - c1)
    ax.plot([c1[0] + u[0] * 7.9, p[0] - u[0] * 5.6], [c1[1] + u[1] * 7.9, p[1] - u[1] * 5.6],
            color=LG, lw=1.0, zorder=2)
    disc(overlay(pool[i]), *p, dm)
T(26.5, 5.6, "Each highlights the lesions, plus errors of its own", 10.5, DG)

arrow(50.5, 29.0, 55.5, 29.0)

# 2 ---------------------------------------------------------------------------------------
step(56.5, 2, "Where they agree marks the lesions")
disc(overlay(consensus), 80.5, 29.0, 31.0, ring=ACC, lw=2.2)
T(80.5, 9.0, "Consensus map", 12, ACC)
T(80.5, 5.6, "Shared highlights remain; single-model errors fade", 10.5, DG)

arrow(104.5, 29.0, 109.5, 29.0)

# 3 ---------------------------------------------------------------------------------------
step(110.5, 3, "Agreement ranks the models")
order = np.argsort(-agree[show_idx])
T(127.0, 47.4, "Agreement with the consensus map", 10, DG, ha="left")
y = 42.0
for r, o in enumerate(order):
    i = show_idx[o]
    T(112.6, y, str(r + 1), 11, ACC if r == 0 else DG, "bold" if r == 0 else "normal")
    disc(overlay(pool[i]), 119.5, y, 7.2)
    lo = agree[show_idx].min() - 0.08
    frac = (agree[i] - lo) / (agree[show_idx].max() - lo)
    ax.add_patch(FancyBboxPatch((125.5, y - 0.9), 26.0 * frac, 1.8, boxstyle="square,pad=0",
                                fc=ACC if r == 0 else "#9A9A9A", ec="none", zorder=3))
    y -= 7.6
T(134.0, 5.6, "Explanations ranked without lesion annotations", 10.5, DG)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"Figure1_C_draft.{ext}"), dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
print("saved; agreement:", np.round(agree, 3))
