"""Figure 1 candidate (G): layout of the edited GPT version, rebuilt with two fixes.

Claims shown
  1  The shared signal of different high-performing models marks the lesions (panel a)
  2  Agreement with the consensus map reflects lesion localization (panel b, right)
  3  So a model can be selected without lesion annotations; annotations are used only for
     validation (panel b, title and labels)
Fixes
  (1) ring roles: the model ranked first (Model 2) marks only the shared lesion foci; the lone,
      model-specific rings belong to lower-ranked models
  (2) panel b shows the selected and the lowest-ranked model side by side,
      shown as individual saliency maps (highest vs lowest CoSA)
Maps are schematic. The consensus uses the actual rule (L1, per-pixel trim 2+2, mean of 10 of 14).
Bar lengths are schematic; the validation ranking matches the CoSA ranking except for a near-tied
swap at ranks 3 and 4 (concept illustration).
Fundus photograph: e-Ophtha C0014759 (cite Decenciere et al., IRBM 2013).
Helper for make_figure1.py, which uses only its set-up part. Run on its own it draws an earlier draft (Figure1_G).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "_figure1_base.py"), encoding="utf-8").read()
exec(_src.split("# ------------------------------------------------------------------ canvas")[0])

from matplotlib.patches import Rectangle

INK, MUTED, RULE = "#1A1A1A", "#5C5C5C", "#CFCFCF"
MCOL = ["#4C9BE8", "#7BD389", "#C58AF9", "#F27FB2", "#E3B505"]


# ---------------------------------------------------------------- the five models shown
def lesions(w):
    return sum(wk * c for wk, c in zip(w, clusters))


# Every model = lesion part + one model-specific spot + shared background (BGn) + own noise.
# The ranks 2 > 4 > 1 > 3 > 5 are built in step by step: weaker lesion weights, larger
# model-specific spot and more own noise; Model 3 also misses the small right lesion.
BG = sum(field(100 + i, 0.10) for i in range(14)) / 14
BGn = (BG - BG.min()) / (BG.max() - BG.min()) * 0.10          # background shared across models
shown = [
    l1(lesions([0.75, 0.75, 0.85, 0.75, 0.70]) + blob(1700, 500, 18, 0.85) + field(201, 0.07) + BGn),   # Model 1
    l1(lesions([1.00, 1.00, 1.00, 1.00, 0.80]) + blob(1500, 1100, 11, 0.68) + field(202, 0.02) + BGn),  # Model 2
    l1(lesions([0.80, 0.80, 0.85, 0.80, 0.10]) + blob(1100, 1200, 20, 0.90) + field(203, 0.09) + BGn),  # Model 3
    l1(lesions([0.95, 0.95, 1.00, 0.85, 0.75]) + blob(1350, 1450, 15, 0.75) + field(204, 0.045) + BGn), # Model 4
    l1(0.25 * lesions(np.ones(KC)) + blob(*DISC, 24, 1.3) + blob(1250, 250, 22, 1.0) + field(205, 0.14) + BGn),  # 5
]
# ring level per model: rings on a shared lesion nest instead of hiding each other. The same
# level cuts the heatmaps in panel b, so a model's heatmap there covers exactly its rings in a.
LEVEL = [0.35, 0.55, 0.22, 0.45, 0.5]
pool[:5] = np.array(shown)                                       # 5 shown + 9 others = 14
consensus = np.sort(pool, axis=0)[2:-2].mean(0)                  # per-pixel trim 2+2, mean of 10


def topk(m, frac=0.05):
    return (m >= np.quantile(m[fov], 1 - frac)) & fov


sp = np.array([spearmanr(m[fov], consensus[fov])[0] for m in shown])
ov = np.array([(topk(m) & topk(consensus)).sum() / topk(m).sum() for m in shown])
order = list(np.argsort((np.argsort(np.argsort(-sp)) + np.argsort(np.argsort(-ov))) / 2, kind="stable"))
print("Spearman", sp.round(3), "top-k", ov.round(3), "CoSA order (model idx)", order)
assert list(np.argsort(-sp)) == list(np.argsort(-ov)) == [1, 3, 0, 2, 4], "both measures: 2 > 4 > 1 > 3 > 5"


def overlay_at(sal, lev):
    """Heatmap shown only where the map reaches the model's ring level (same area as its rings)."""
    s_ = show(sal)
    a = np.where(s_ >= lev, np.clip(0.5 + 0.4 * (s_ - lev) / (1 - lev), 0, 0.9), 0.0)[..., None]
    rgb = fundus * (1 - a) + jet(s_)[..., :3] * a
    return np.dstack([rgb, fov.astype(float)])


# ---------------------------------------------------------------- canvas
FW, FH = 16.0, 8.9
fig = plt.figure(figsize=(FW, FH), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW * 10)
ax.set_ylim(0, FH * 10)
ax.axis("off")


def T(x, y, s, size=14, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def T_after(t, s, size, color, weight="normal"):
    """Text continuing on the same line right after text artist t."""
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x, y = ax.transData.inverted().transform((bb.x1, (bb.y0 + bb.y1) / 2))
    return T(x, y, s, size, color, weight)


def arrow(x0, y0, x1, y1, ms=16, lw=1.4, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def panel(y, letter, title):
    T(2.0, y, letter, 24, INK, "bold")
    return T(7.0, y, title, 19, INK, "bold")


def ext(x, y, d):
    return [x - d / 2, x + d / 2, y - d / 2, y + d / 2]


def grid_xy(x, y, d):
    return x - d / 2 + (np.arange(N) + 0.5) * d / N, y + d / 2 - (np.arange(N) + 0.5) * d / N


def disc(img, x, y, d, ring=None, lw=2.4):
    ax.imshow(img, extent=ext(x, y, d), zorder=3, interpolation="bilinear")
    if ring:
        ax.add_patch(Circle((x, y), d / 2 * 0.985, fc="none", ec=ring, lw=lw, zorder=4))


plain = np.dstack([fundus * 0.92, fov.astype(float)])

# ---- a ----------------------------------------------------------------------------------
panel(83.5, "a", "Consensus map generation from different foundation models")
YR, D = 63.0, 28.0
XM, XC, XT = 36.0, 83.0, 128.0
disc(plain, XM, YR, D)
xs, ys = grid_xy(XM, YR, D)
# Model 2 (ranked first) is drawn on top so all of its rings stay visible
for k in (0, 2, 3, 4, 1):
    ax.contour(xs, ys, show(shown[k]) * fov, levels=[LEVEL[k]], colors=[MCOL[k]],
               linewidths=2.2 if k == 1 else 1.8, zorder=6 if k == 1 else 5)


def on_disc(px, py, x=XM, y=YR, d=D):
    gx, gy = to_xy(px, py)
    return x - d / 2 + gx / N * d, y + d / 2 - gy / N * d


def callout(pt, txt_xy, text):
    ax.plot([pt[0], txt_xy[0]], [pt[1], txt_xy[1]], color=INK, lw=0.9, zorder=6)
    ax.plot(*pt, "o", ms=3.6, color=INK, zorder=7)
    T(txt_xy[0] - 0.6, txt_xy[1], text, 14, INK, ha="right", style="italic")


callout(on_disc(791, 792), (XM - D / 2 - 3.0, YR + 8.5), "Shared focus:\nmodels agree")
callout(on_disc(1350, 1450), (XM - D / 2 - 3.0, YR - 8.5), "Model-specific\nfocus")
T(XM, YR - D / 2 - 3.0, "Different high-performing foundation models mark what they see", 14, MUTED, ha="center")
for k, c in enumerate(MCOL):
    x = 10.0 + k * 10.6
    ax.plot(x, YR - D / 2 - 6.6, "o", ms=8, mfc="none", mec=c, mew=2.0)
    T(x + 1.6, YR - D / 2 - 6.6, f"Model {k + 1}", 13, MUTED)
arrow(XM + D / 2 + 3.0, YR, XC - D / 2 - 3.0, YR)
disc(overlay(consensus), XC, YR, D, ring=ACC, lw=2.6)
T(XC, YR - D / 2 - 3.0, "Consensus map", 16, ACC, "bold", ha="center")
T(XC, YR - D / 2 - 6.4, "Closer to the lesions than single models", 13, MUTED, ha="center")
T((XC + XT) / 2, YR, "≈", 38, INK, ha="center")
les = ndimage.binary_dilation(lesion, iterations=2)
ann = fundus.copy()
ann[les] = ann[les] * 0.15 + np.array([0.0, 0.9, 1.0]) * 0.85
disc(np.dstack([ann, fov.astype(float)]), XT, YR, D)
T(XT + 1.6, YR - D / 2 - 3.0, "Expert lesion annotation", 14, MUTED, ha="center")
ax.add_patch(Rectangle((XT - 11.4, YR - D / 2 - 3.9), 1.8, 1.8, fc="#00E5FF", ec="none"))   # cyan key
T(XT, YR - D / 2 - 6.4, "Used only for validation", 13, MUTED, ha="center", style="italic")
ax.plot([2.0, 158.0], [38.5, 38.5], color=RULE, lw=0.9)

# ---- b ----------------------------------------------------------------------------------
t = panel(33.0, "b", "Consensus map-based model selection ")
T_after(t, "without lesion annotations", 19, ACC, "bold")
names = [f"Model {k + 1}" for k in range(5)]


def ranking(x0, title, sub, sub_color, idx, lens):
    T(x0, 27.0, title, 16, INK, "bold")
    T(x0, 23.6, sub, 14, sub_color)
    for r, (k, ln) in enumerate(zip(idx, lens)):
        y = 19.4 - r * 3.6
        ax.plot(x0 + 1.0, y, "o", ms=8, mfc="none", mec=MCOL[k], mew=2.0)
        T(x0 + 2.8, y, names[k], 14, INK)
        ax.add_patch(Rectangle((x0 + 13.0, y - 1.0), 21.0 * ln, 2.0,
                               fc=ACC if r == 0 else "#B3B3B3", ec="none"))


ranking(7.0, "CoSA ranking", "Consensus-based Saliency Agreement", ACC, order, [1.0, 0.86, 0.66, 0.60, 0.30])

# middle: selected vs lowest-ranked model, consensus hot spots dashed
DM, YM = 17.0, 12.0
X1, X2 = 62.5, 84.5
T((X1 + X2) / 2, 27.0, "Individual saliency maps", 16, INK, "bold", ha="center")
T((X1 + X2) / 2, 23.6, "highest vs lowest CoSA", 13, MUTED, ha="center", style="italic")
for x, k, lab in ((X1, order[0], "Selected (highest CoSA)"), (X2, order[-1], "Lowest CoSA")):
    disc(overlay_at(shown[k], LEVEL[k]), x, YM, DM, ring=MCOL[k], lw=2.6)   # same areas as its rings in a
    sel = lab.startswith("Selected")
    T(x, YM - DM / 2 - 2.0, names[k], 14, INK, "bold" if sel else "normal", ha="center")
    T(x, YM - DM / 2 - 5.0, lab, 13, ACC if sel else MUTED, "bold" if sel else "normal", ha="center")
arrow(42.5, 19.4, X1 - DM / 2 - 1.5, YM + 2.0, ms=13, lw=1.1)

# right: validation ranking (similar order: only the near-tied ranks 3 and 4 swap)
ax.plot([96.0, 96.0], [2.0, 29.0], color=RULE, lw=0.9)
ranking(99.0, "Lesion-localization ranking", "Expert annotation · validation only", MUTED,
        order[:2] + [order[3], order[2]] + order[4:],     # ranks 3 and 4 swapped (near-tie)
        [0.93, 0.79, 0.61, 0.57, 0.24])
T(136.5, 19.4, "Agreement with", 14, INK, "bold")
T(136.5, 15.8, "the consensus map", 14, INK, "bold")
T(136.5, 12.2, "reflects lesion", 14, INK, "bold")
T(136.5, 8.6, "localization", 14, INK, "bold")

for e in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"Figure1_G.{e}"), dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
print("saved")
