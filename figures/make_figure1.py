"""Figure 1 (final): CoSA concept schematic. Panel a: overlaid saliency maps of five illustrative models with per-model dashed outlines and the shared focus outlined in orange; patchy ViT-style schematic maps from _figure1_maps.py.

Message
  a  Similarly accurate foundation models highlight different regions; the consensus saliency map
     retains their shared focus and lies closer to the lesions than any single model
  b  How CoSA is obtained: agreement of each model's saliency map with the consensus map
  c  CoSA ranking tracks the lesion-localization ranking, so the model with the highest CoSA can be
     selected without lesion annotations (annotations are used only for validation)
Maps, models and the five shown models are the same as in G (_figure1_layout.py); the CoSA order
2 > 4 > 1 > 3 > 5 is computed there. Bar lengths are schematic; the validation ranking matches the
CoSA ranking except for a near-tied swap at ranks 3 and 4 (concept illustration).
Layout grid: three columns (centres C1, C2, C3) shared by all panels; body text starts at the left
edge of column 1 and ends at the right edge of column 3; every panel has the same title-to-content
and rule-to-title spacing.
Fundus photograph: e-Ophtha C0014759 (cite Decenciere et al., IRBM 2013).
Output: Figure1.png / .pdf (same folder)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "_figure1_layout.py"), encoding="utf-8").read()
exec(_src.split("\n# ---------------------------------------------------------------- canvas\n")[0])

BAND, BAR = "#FDEBD9", "#BDBDBD"
plt.rcParams["pdf.fonttype"] = 42                 # editable text in the PDF (journal requirement)

# ---------------------------------------------------------------- high-resolution images
# The maps are computed on the N x N grid (above); for display they are upsampled smoothly and laid
# over the photograph at its native resolution (the cropped circle is 2 * half = 1530 px).
HR = 2 * half


def up(a):
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((HR, HR), Image.BICUBIC))


fundus_hr = np.clip(crop(im), 0, 1)
fov_hr = np.clip(up(ndimage.gaussian_filter(fov.astype(float), 0.6)), 0, 1)   # soft, smooth rim


def overlay(sal, amax=0.8):
    s_ = np.clip(up(show(sal)), 0, 1)
    a = np.clip((s_ - 0.15) / 0.5, 0, 1)[..., None] * amax
    return np.dstack([fundus_hr * (1 - a) + jet(s_)[..., :3] * a, fov_hr])


def overlay_at(sal, lev):
    """Heatmap shown only where the map reaches the model's ring level (same area as its rings)."""
    s_ = np.clip(up(show(sal)), 0, 1)
    a = np.where(s_ >= lev, np.clip(0.5 + 0.4 * (s_ - lev) / (1 - lev), 0, 0.9), 0.0)[..., None]
    return np.dstack([fundus_hr * (1 - a) + jet(s_)[..., :3] * a, fov_hr])


# ---------------------------------------------------------------- patchy schematic maps (replace I's)
exec(open(os.path.join(HERE, "_figure1_maps.py"), encoding="utf-8").read())
shown, consensus = PATCHY_SHOWN, PATCHY_CONS
# crisp, saturated model colours (the pastel set read as mould on the heatmap); used in all panels
MCOL = ["#1F5FBF", "#00897B", "#6A3FB5", "#C2185B", "#37474F"]
overlay = lambda m, *a, **k: render_patchy(m)
overlay_at = lambda m, *a, **k: render_patchy(m)
fundus_hr = FUNDUS_BRIGHT                                   # annotation disc uses the same lighter photo

# ---------------------------------------------------------------- canvas
FW, FH = 16.0, 17.4
fig = plt.figure(figsize=(FW, FH), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW * 10)
ax.set_ylim(0, FH * 10)
ax.axis("off")
R = fig.canvas.get_renderer()

# grid
D = 28.0                                          # fundus discs in panel a
C1, C2, C3 = 32.0, 80.0, 128.0                    # column centres, centred on the canvas
LEFT, RIGHT = C1 - D / 2, C3 + D / 2              # body text block: 18 .. 142 (equal side margins)
GAP_TITLE, GAP_RULE = 7.5, 6.0                    # title -> content, rule -> next title

# font sizes. npj Digital Medicine (Nature portfolio): figure text 5-7 pt and panel letters 8 pt bold
# at the final double-column width (180 mm). The saved figure is ~16.2 in wide, so it is printed at
# ~0.44x; sizes below give 7.0 pt for all text (bold for headings) and 8.1 pt panel letters.
F_LET, F_TITLE, F_HEAD, F_BODY = 18.5, 16.0, 16.0, 16.0
LINE = 3.3                                        # line step for two-line text at F_BODY


def T(x, y, s, size=14, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def width(t):
    """Width of a text artist in data units."""
    bb = t.get_window_extent(renderer=R)
    return (bb.x1 - bb.x0) / ax.transData.get_matrix()[0, 0]


def T_after(t, s, size, color, weight="normal", **kw):
    """Text continuing on the same line right after text artist t (t must be left-aligned)."""
    x, y = t.get_position()
    return T(x + width(t), y, s, size, color, weight, **kw)


def arrow(x0, y0, x1, y1, ms=16, lw=1.4, color=INK, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def panel(y, letter, title):
    T(2.0, y, letter, F_LET, INK, "bold")
    return T(6.5, y, title, F_TITLE, INK)        # regular weight: the bold panel letter stands out


def rule(y):
    ax.plot([2.0, 158.0], [y, y], color=RULE, lw=0.9)


def ext(x, y, d):
    return [x - d / 2, x + d / 2, y - d / 2, y + d / 2]


def grid_xy(x, y, d):
    return x - d / 2 + (np.arange(N) + 0.5) * d / N, y + d / 2 - (np.arange(N) + 0.5) * d / N


def disc(img, x, y, d, ring=None, lw=2.4):
    ax.imshow(img, extent=ext(x, y, d), zorder=3, interpolation="antialiased")
    if ring:
        ax.add_patch(Circle((x, y), d / 2 * 0.985, fc="none", ec=ring, lw=lw, zorder=4))


plain = np.dstack([fundus_hr * 0.92, fov_hr])
names = [f"Model {k + 1}" for k in range(5)]

# ---- a ----------------------------------------------------------------------------------
YA = FH * 10 - 4.0                                 # top of the canvas (content runs down from here)
panel(YA, "a", "Generating a consensus saliency map from diverse foundation models")
YR = YA - GAP_TITLE - D / 2
# One fundus, as in H: the saliency maps of the five models overlaid (pixel-wise maximum), so both the
# shared focus (lesions) and the model-specific foci are visible; the trimmed mean keeps only the former.
UNION = np.max([scale(m) for m in shown], axis=0)
FX, FY, DF = C1, YR, D                            # one overlaid disc (no stack)
ax.imshow(render_patchy(UNION), extent=ext(FX, FY, DF), zorder=3, interpolation="antialiased")
# rings (as before, crisp colours with a white halo), clipped to the fundus so none sticks out
def to_data(r_, c_):
    return FX - DF / 2 + (c_ + 0.5) / N * DF, FY + DF / 2 - (r_ + 0.5) / N * DF


BR, BF = 1.25, 12.0                               # legend marker size (kept for the legend below)
CLIP = Circle((FX, FY), DF / 2 * 0.985, transform=ax.transData)


def ring(xy, rad, k):
    for ec, lw, z, al in (("white", 3.4, 5.9, 0.85), (MCOL[k], 1.8, 6, 1.0)):
        p = Circle(xy, rad, fc="none", ec=ec, lw=lw, alpha=al, zorder=z)
        ax.add_patch(p)
        p.set_clip_path(CLIP)


# outlines (multi-rater style): each model's high-saliency region outlined in its colour. Where the
# models overlap (shared focus on the lesions) the outlines are solid and the area is lifted with a soft
# white glow; outside it (model-specific focus) the outlines are dashed.
from matplotlib.path import Path
from matplotlib.patches import PathPatch
xs_g = FX - DF / 2 + (np.arange(N) + 0.5) * DF / N
ys_g = FY + DF / 2 - (np.arange(N) + 0.5) * DF / N

masks = []
for k in range(5):
    m_ = ndimage.gaussian_filter(scale(shown[k]), 4.0) >= 0.62
    m_ &= fov
    lab_, n_ = ndimage.label(m_)
    keep = np.zeros_like(m_)
    for i_ in range(1, n_ + 1):
        if (lab_ == i_).sum() >= 260:                 # drop small fragments
            keep |= lab_ == i_
    masks.append(keep)
# shared zone = the lesion regions all accurate models attend to (same zones as the consensus map)
shared_zone = ndimage.binary_dilation(LES >= 0.5, iterations=6) & fov
# soft white glow over the shared zone
glow = ndimage.gaussian_filter(shared_zone.astype(float), 6)
gimg = np.zeros((N, N, 4)); gimg[..., :3] = 1.0; gimg[..., 3] = np.clip(glow * 1.3, 0, 1) * 0.30 * fov
ax.imshow(gimg, extent=ext(FX, FY, DF), zorder=4, interpolation="bilinear")

# clip paths: inside the shared zone (solid lines) and outside it (dashed lines)
def zone_segs(mask):
    cs_ = plt.contour(xs_g, ys_g, ndimage.gaussian_filter(mask.astype(float), 1.5), levels=[0.5])
    segs = [np.asarray(sg) for sg in cs_.allsegs[0] if len(sg) > 2]
    for coll in getattr(cs_, "collections", [cs_]):
        coll.remove()
    return segs


def segs_path(segs, reverse=False):
    verts, codes = [], []
    for sg in segs:
        sg = sg[::-1] if reverse else sg
        verts += list(sg) + [sg[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(sg) - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


SEGS = zone_segs(shared_zone)
P_IN = segs_path(SEGS)
circ = Path.circle((FX, FY), DF / 2 * 0.985)
P_OUT = Path.make_compound_path(circ, segs_path(SEGS, reverse=True))   # disc with the shared zone cut out
pp_in = PathPatch(P_IN, transform=ax.transData, fc="none", ec="none")
pp_out = PathPatch(P_OUT, transform=ax.transData, fc="none", ec="none")
ax.add_patch(pp_in); ax.add_patch(pp_out)

# shared zone: every model's outline drawn dashed inside it; the zone itself outlined solid in the
# consensus colour (orange), since this is the focus the consensus map keeps
for k in (4, 2, 0, 3, 1):                         # Model 2 drawn last (on top)
    sm = ndimage.gaussian_filter(masks[k].astype(float), 2.0)
    cs = ax.contour(xs_g, ys_g, sm, levels=[0.5], colors=[MCOL[k]], linewidths=1.6, zorder=6,
                    linestyles=[(0, (2.6, 1.8))])
    for coll in getattr(cs, "collections", [cs]):
        coll.set_clip_path(pp_in)
for ec, lw, z in (("white", 4.2, 6.4), (ACC, 2.6, 6.5)):
    zone_line = ax.contour(xs_g, ys_g, ndimage.gaussian_filter(shared_zone.astype(float), 2.5), levels=[0.5],
                           colors=[ec], linewidths=lw, zorder=z)
    for coll in getattr(zone_line, "collections", [zone_line]):
        coll.set_clip_path(CLIP)

# model-specific focus: only representative regions, dashed (Model 3: its largest off-lesion region;
# Model 5: optic disc and periphery)
REP = {2: 1, 4: 1}                                 # Model 3 lower region, Model 5 optic disc (the two callout targets)
for k, n_keep in REP.items():
    lab_, n_ = ndimage.label(masks[k] & ~ndimage.binary_dilation(shared_zone, iterations=4))
    comps = sorted([lab_ == i_ for i_ in range(1, n_ + 1)], key=lambda c: -c.sum())[:n_keep]
    if not comps:
        continue
    sm = ndimage.gaussian_filter(np.sum(comps, axis=0).astype(float), 2.0)
    cs = ax.contour(xs_g, ys_g, sm, levels=[0.5], colors=[MCOL[k]], linewidths=1.8, zorder=6,
                    linestyles=[(0, (2.6, 1.8))])
    for coll in getattr(cs, "collections", [cs]):
        coll.set_clip_path(CLIP)


def on_disc(px, py, x=None, y=None, d=None):
    x, y, d = (FX if x is None else x), (FY if y is None else y), (DF if d is None else d)
    gx, gy = to_xy(px, py)
    return x - d / 2 + gx / N * d, y + d / 2 - gy / N * d


def callout(pts, txt_xy, text):
    for pt in pts:
        ax.plot([pt[0], txt_xy[0]], [pt[1], txt_xy[1]], color=INK, lw=0.9, zorder=6)
        ax.plot(*pt, "o", ms=3.6, color=INK, zorder=7)
    T(txt_xy[0] - 0.6, txt_xy[1], text, F_BODY, INK, ha="right", linespacing=1.3)


XCALL = LEFT - 3.6
# one callout line to each shared lesion region (centre of each part of the orange-outlined zone)
_zl, _zn = ndimage.label(shared_zone)
_parts = sorted([_zl == i_ for i_ in range(1, _zn + 1)], key=lambda c: -c.sum())[:2]
_pts = [to_data(*ndimage.center_of_mass(c)) for c in _parts]
callout(_pts, (XCALL, YR + 8.5), "Shared focus\nacross models")
callout([on_disc(1100, 1200), on_disc(*DISC)], (XCALL, YR - 8.5), "Model-specific\nfocus")

Y_CAP1 = YR - D / 2 - 3.2                         # first caption line under every disc
Y_CAP2 = Y_CAP1 - 4.0                             # second caption line
t = T(0, Y_CAP1, "Similarly accurate foundation models highlight different regions", F_BODY, MUTED)
t.set_x(max(2.0, C1 - width(t) / 2))
probe = T(0, 0, names[-1], F_BODY)
STEP = 1.6 + width(probe) + 3.3                   # marker, name, gap
span = 4 * STEP + 1.6 + width(probe) + 0.7
probe.remove()
x_first = max(2.0, C1 - span / 2) + 0.7
for k, c in enumerate(MCOL):
    x = x_first + k * STEP
    ax.plot(x, Y_CAP2, "o", ms=8, mfc="none", mec=c, mew=2.0)
    T(x + 1.6, Y_CAP2, names[k], F_BODY, MUTED)
YK = Y_CAP2 - 4.0                                 # line-style key under the model legend
xk = x_first - 0.7
ax.plot([xk, xk + 4.0], [YK, YK], color=ACC, lw=2.6)
tk = T(xk + 5.0, YK, "Shared focus across models", F_BODY, MUTED)

CA, GX, DM = C1, 0.0, D                           # merge arrow starts at the right edge of the disc
XM0 = FX + DF / 2 + 2.5                      # merge arrow: from the block to the consensus map
arrow(XM0, YR, C2 - D / 2 - 3.0, YR)
T((XM0 + C2 - D / 2 - 3.0) / 2, YR + 4.6, "Pixel-wise\ntrimmed mean", F_BODY, MUTED, ha="center", linespacing=1.3)
disc(overlay(consensus), C2, YR, D, ring=ACC, lw=2.6)
T(C2, Y_CAP1, "Consensus map", F_HEAD, ACC, "bold", ha="center")
T(C2, Y_CAP2, "Shared focus retained", F_BODY, MUTED, ha="center")

XA = (C2 + C3) / 2
T(XA, YR + 3.0, "≈", 38, INK, ha="center")
T(XA, YR - 8.0, "Closer to lesions\nthan any single model", F_BODY, MUTED, ha="center", linespacing=1.5)

les = ndimage.binary_dilation(crop(mk.astype(float)) > 0.5, iterations=round(2 * HR / N))
ann = fundus_hr.copy()
LES_HEX = "#00E5FF"                               # lesion colour: original cyan
LES_RGB = np.array([0.0, 0.9, 1.0])
ann[les] = ann[les] * 0.15 + LES_RGB * 0.85
disc(np.dstack([ann, fov_hr]), C3, YR, D)
t = T(0, Y_CAP1, "Expert lesion annotation", F_BODY, MUTED)
KEY, KGAP = 1.8, 1.0                              # cyan key + gap, centred together with the text
x0 = C3 - (KEY + KGAP + width(t)) / 2
ax.add_patch(Rectangle((x0, Y_CAP1 - KEY / 2 - 0.15), KEY, KEY, fc=LES_HEX, ec="none"))
t.set_x(x0 + KEY + KGAP)
T(C3, Y_CAP2, "Used only for validation", F_BODY, MUTED, ha="center")
Y_RULE_A = Y_CAP2 - 9.5
rule(Y_RULE_A)

# ---- b ----------------------------------------------------------------------------------
YB_T = Y_RULE_A - GAP_RULE
t = panel(YB_T, "b", "Computing Consensus-based Saliency Agreement (CoSA) of each saliency map with the consensus map")
assert t.get_position()[0] + width(t) <= 158.0, "panel b title runs past the rule"   # CoSA spelled out at first use
DS, DB = 18.0, D                                  # individual maps; consensus map as wide as in a (edge at LEFT)
YH = YB_T - GAP_TITLE - DS / 2                    # highest-CoSA model (Model 2)
YL = YH - 25.0                                    # lowest-CoSA model (Model 5)
YB = (YH + YL) / 2
disc(overlay(consensus), C1, YB, DB, ring=ACC, lw=2.6)
T(C1, YB - DB / 2 - 3.2, "Consensus map", F_HEAD, ACC, "bold", ha="center")
for yk in (YH, YL):
    s = 1 if yk > YB else -1
    arrow(C1 + DB / 2 + 1.5, YB + s * 1.8, C2 - DS / 2 - 1.5, yk - s * 2.5,
          style="<|-|>", lw=1.2, ms=14, color="#444444")
T((C1 + DB / 2 + C2 - DS / 2) / 2, YB, "Agreement", F_HEAD, INK, "bold", ha="center")

NOTES = ("Saliency map resembles the consensus", "Saliency map differs from the consensus")
probe = [T(0, 0, n_, F_BODY) for n_ in NOTES]
XBAR, LBAR = RIGHT - max(width(t_) for t_ in probe), 26.0   # CoSA column: longest note ends at RIGHT
for t_ in probe:
    t_.remove()
T(XBAR, YH + 4.2, "CoSA", F_HEAD, INK, "bold")             # spelled out in the panel title
for k, yk, lab, frac, colr, note in (
        (order[0], YH, "High", 1.0, ACC, NOTES[0]),
        (order[-1], YL, "Low", 0.24, BAR, NOTES[1])):
    disc(overlay_at(shown[k], LEVEL[k]), C2, yk, DS, ring=MCOL[k], lw=2.6)
    T(C2, yk - DS / 2 - 2.6, f"{names[k]} saliency map", F_BODY, INK, ha="center")
    arrow(C2 + DS / 2 + 2.5, yk, XBAR - 3.0, yk, ms=14, lw=1.2)
    ax.add_patch(Rectangle((XBAR, yk - 1.3), LBAR * frac, 2.6, fc=colr, ec="none"))
    T(XBAR + LBAR * frac + 1.2, yk, lab, F_HEAD, ACC if lab == "High" else MUTED, "bold")
    T(XBAR, yk - 4.0, note, F_BODY, MUTED)
def text_lines(y, lines):
    """Lines of (text, colour, weight) parts at F_BODY starting at LEFT; each must end by RIGHT.
    Returns the y of the last line."""
    for i_, parts in enumerate(lines):
        x = LEFT
        for s_, c_, w_ in parts:
            x += width(T(x, y - i_ * LINE, s_, F_BODY, c_, w_))
        assert x <= RIGHT + 0.3, f"line {i_ + 1} ends at {x:.1f} > {RIGHT}"
    return y - (len(lines) - 1) * LINE


Y_NOTE = YL - DS / 2 - 8.0
y_last = text_lines(Y_NOTE, [
    [("CoSA = agreement (rank correlation and top-k overlap) between a model’s saliency map and the "
      "consensus map,", MUTED, "normal")],
    [("computed on images ", MUTED, "normal"), ("without lesion annotations.", INK, "bold")]])
Y_RULE_B = y_last - 5.0
rule(Y_RULE_B)

# ---- c ----------------------------------------------------------------------------------
# reading order: selected model = first row of the CoSA ranking (one highlighted row); the
# lesion-localization ranking on the right is only the validation of the CoSA ranking
YC_T = Y_RULE_B - GAP_RULE
t = panel(YC_T, "c", "Ranking and selecting models by CoSA ")
T_after(t, "without lesion annotations", F_TITLE, ACC, "bold")
lesion_order = order[:2] + [order[3], order[2]] + order[4:]      # ranks 3 and 4 swapped (near-tie)
NUM_W, NAME_W, BAR_W, DY = 2.6, 15.0, 19.5, 3.9   # marker offset, bar offset, bar length, row step
BLOCK = NAME_W + BAR_W                            # width of one ranking
SUB_R = "Overlap with expert annotations · validation only"
probe = T(0, 0, SUB_R, F_BODY)
W_SUB_R = width(probe)
probe.remove()
XL = LEFT + 24.5                                  # CoSA ranking (after the selected-model column)
XR = RIGHT - max(BLOCK, W_SUB_R)                  # lesion-localization ranking: its subtitle ends at RIGHT
Y_HEAD = YC_T - GAP_TITLE
Y_SUB = Y_HEAD - 3.6
Y0 = Y_SUB - 5.0                                  # first row


def ranking(x0, title, sub, idx, lens, top_color):
    T(x0, Y_HEAD, title, F_HEAD, INK, "bold")
    T(x0, Y_SUB, sub, F_BODY, MUTED)
    for r_, (k, ln) in enumerate(zip(idx, lens)):
        y = Y0 - r_ * DY
        T(x0 + 0.9, y, str(r_ + 1), F_BODY, MUTED, ha="center")
        ax.plot(x0 + NUM_W + 1.0, y, "o", ms=8, mfc="none", mec=MCOL[k], mew=2.0)
        t_ = T(x0 + NUM_W + 2.8, y, names[k], F_BODY, INK, "bold" if r_ == 0 else "normal")
        assert NUM_W + 2.8 + width(t_) <= NAME_W - 0.8, "model name runs into its bar"
        ax.add_patch(Rectangle((x0 + NAME_W, y - 1.0), BAR_W * ln, 2.0,
                               fc=top_color if r_ == 0 else BAR, ec="none", zorder=2))


# selected model, left of the CoSA ranking on the same highlighted row (thumbnail, "Selected",
# CoSA rank 1); the band stops at the end of the CoSA ranking
probe = [T(0, 0, "Selected", F_HEAD, weight="bold"), T(0, 0, "highest CoSA", F_BODY)]
W_SEL, W_HC = (width(p_) for p_ in probe)
for p_ in probe:
    p_.remove()
XSEL = LEFT + W_HC / 2                            # thumbnail column (with its caption) starts at LEFT
DSEL = 2 * (XL - 2.0 - W_SEL - 1.2 - XSEL)        # thumbnail fills the space up to "Selected"
assert DSEL <= W_HC, (DSEL, W_HC)
ax.add_patch(Rectangle((LEFT - 1.0, Y0 - DY / 2), XL + BLOCK + 1.0 - (LEFT - 1.0), DY,
                       fc=BAND, ec="none", zorder=0))
disc(overlay_at(shown[order[0]], LEVEL[order[0]]), XSEL, Y0, DSEL, ring=MCOL[order[0]], lw=2.4)
T(XL - 2.0 - W_SEL, Y0, "Selected", F_HEAD, ACC, "bold")
T(XSEL, Y0 - DSEL / 2 - 2.4, names[order[0]], F_BODY, INK, "bold", ha="center")
t_hc = T(XSEL, Y0 - DSEL / 2 - 2.4 - LINE, "highest CoSA", F_BODY, MUTED, ha="center")
print(f"thumbnail {DSEL:.1f}; 'highest CoSA' spans {XSEL - width(t_hc) / 2:.1f}..{XSEL + width(t_hc) / 2:.1f}"
      f" (LEFT {LEFT})")

ranking(XL, "CoSA ranking", "Agreement with the consensus map", order, [1.0, 0.86, 0.66, 0.60, 0.30], ACC)
ranking(XR, "Lesion-localization ranking", SUB_R, lesion_order, [1.0, 0.85, 0.66, 0.61, 0.26], "#8C8C8C")
LX0, LX1 = XL + BLOCK + 2.5, XR - 2.0
T((LX0 + LX1) / 2, Y0 - 4 * DY - 4.0, "Rankings closely agree", F_BODY, INK, "bold", ha="center")   # under the lines
print(f"connecting lines {LX0:.1f}..{LX1:.1f}; lesion ranking starts at {XR:.1f}")
for i, k in enumerate(order):
    j = lesion_order.index(k)
    ax.plot([LX0, LX1], [Y0 - i * DY, Y0 - j * DY], color=ACC if i == 0 else "#AFAFAF",
            lw=2.0 if i == 0 else 1.1, solid_capstyle="round")
text_lines(Y0 - 4 * DY - 10.5, [
    [("Among accurate models, CoSA ranking tracks lesion-localization ranking,", INK, "normal")],
    [("so the model with the highest CoSA can be selected without lesion annotations.", INK, "normal")]])

for e in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"Figure1.{e}"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
print("saved")
