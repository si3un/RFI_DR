"""Figure 2 at journal size: same data and design as fig2_arith_trim2.py (MODE best), drawn at the
final double-column width (180 mm) in Arial with Nature-portfolio text sizes: 6-7 pt text,
8 pt bold panel letters. Same text sizes as Figure 1.
Output: Figure2.png (600 dpi) / .pdf (same folder); the earlier outputs are kept.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})
F_TICK, F_TEXT, F_TITLE, F_LET = 6.5, 7.0, 7.0, 8.0
MM = 1 / 25.4

BB = ["OpenCLIP", "DINOv2", "RETFound-\nDINOv2", "DINOv3\n(Large)", "DINOv3\n(Huge)", "MAE", "RETFound"]
# single-scale (1.0x) per-case optimal Dice, order: classifier-only, partial, LoRA
D = {
 "e-Ophtha": dict(n=195, perfect=0.124, rand=0.004, cons=0.1039, v=[
   (0.0102, 0.0092, 0.0352), (0.0067, 0.0764, 0.0789), (0.0211, 0.0709, 0.0834), (0.0046, 0.0105, 0.0628),
   (0.0069, 0.0281, 0.0247), (0.0037, 0.0671, 0.0164), (0.0042, 0.0539, 0.0062)]),
 "IDRiD": dict(n=282, perfect=0.299, rand=0.020, cons=0.1506, v=[
   (0.0337, 0.0312, 0.0792), (0.0299, 0.1024, 0.1371), (0.0506, 0.0997, 0.1363), (0.0263, 0.0407, 0.1009),
   (0.0342, 0.0626, 0.0655), (0.0290, 0.1200, 0.0474), (0.0287, 0.1228, 0.0331)]),
 "FGADR": dict(n=4784, perfect=0.455, rand=0.023, cons=0.1901, v=[
   (0.0300, 0.0460, 0.0793), (0.0378, 0.1483, 0.1750), (0.0574, 0.1251, 0.1858), (0.0298, 0.0407, 0.1087),
   (0.0422, 0.0725, 0.0691), (0.0283, 0.1632, 0.0630), (0.0345, 0.1365, 0.0383)]),
}
YMAX = {"e-Ophtha": 0.135, "IDRiD": 0.32, "FGADR": 0.50}
YT = {"e-Ophtha": np.arange(0, 0.121, 0.02), "IDRiD": np.arange(0, 0.301, 0.05), "FGADR": np.arange(0, 0.501, 0.10)}
C_CLS, C_PFT, C_LORA, C_CONS = "#d6e3f5", "#5b93d8", "#1c3a70", "#111111"
C_PERF, C_RAND = "#e0474c", "#2a8c7a"
LS_PERF, LS_RAND = (0, (5, 3)), (0, (6, 2, 1, 2))

fig, axes = plt.subplots(3, 1, figsize=(180 * MM, 175 * MM))
w = 0.26
titles = []
for ax, (letter, (name, d)) in zip(axes, zip("abc", D.items())):
    x = np.arange(len(BB))
    V = np.array(d["v"])
    for j, (col, edge) in enumerate(((C_CLS, "#9fb4d6"), (C_PFT, C_PFT), (C_LORA, C_LORA))):
        ax.bar(x + (j - 1) * w, V[:, j], w * 0.94, color=col, edgecolor=edge, linewidth=0.4, zorder=3)
    xc = len(BB) + 0.6
    ax.bar(xc, d["cons"], w * 2.2, color=C_CONS, zorder=3)
    ax.text(xc, d["cons"] + YMAX[name] * 0.02, "%.3f" % d["cons"], ha="center", va="bottom",
            fontweight="bold", fontsize=F_TEXT)
    bi, bj = np.unravel_index(np.argmax(V), V.shape)                 # best single configuration
    ax.text(bi + (bj - 1) * w, V[bi, bj] + YMAX[name] * 0.02, "%.3f" % V[bi, bj], ha="center", va="bottom",
            fontsize=F_TICK, color="#333333")
    ax.axhline(d["perfect"], color=C_PERF, ls=LS_PERF, lw=0.9, zorder=2)
    ax.axhline(d["rand"], color=C_RAND, ls=LS_RAND, lw=0.8, zorder=2)
    ax.text(xc + 0.55, d["perfect"], "%.3f" % d["perfect"], color=C_PERF, va="center", ha="left", fontsize=F_TICK)
    ax.text(xc + 0.55, d["rand"], "%.3f" % d["rand"], color=C_RAND, va="center", ha="left", fontsize=F_TICK)
    ax.set_xticks(list(x) + [xc])
    ax.set_xticklabels(BB + ["Consensus\nmap"], fontsize=F_TICK, linespacing=1.1)
    ax.set_xlim(-0.6, xc + 0.5)
    ax.set_ylim(0, YMAX[name])
    ax.set_yticks(YT[name])
    ax.set_yticklabels(["%.2f" % t for t in YT[name]], fontsize=F_TICK)
    ax.set_ylabel("Dice coefficient", fontsize=F_TEXT)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#333333", length=2, width=0.6, pad=2)
    t = ax.set_title(name, loc="left", fontsize=F_TITLE, fontweight="bold", pad=5)
    titles.append((ax, t, letter, "  (n = %s cases)" % format(d["n"], ",")))
handles = [Patch(facecolor=C_CLS, edgecolor="#9fb4d6", linewidth=0.4, label="Classifier-only"),
           Patch(facecolor=C_PFT, label="Partial fine-tuning"),
           Patch(facecolor=C_LORA, label="LoRA"), Patch(facecolor=C_CONS, label="Consensus map"),
           Line2D([0], [0], color=C_PERF, ls=LS_PERF, lw=0.9, label="Perfect score using saliency resolution"),
           Line2D([0], [0], color=C_RAND, ls=LS_RAND, lw=0.8, label="Random reference")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=F_TEXT,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=2.0)
note = fig.text(0, 0.004, "Y-axis scales differ between panels.", fontsize=F_TICK, color="#555555")
fig.tight_layout(rect=(0.02, 0.02, 1, 0.94), h_pad=1.6)
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for ax, t, letter, extra in titles:
    bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
    ax.text(bb.x1, bb.y0, extra, transform=ax.transAxes, fontsize=F_TITLE, va="bottom", ha="left")
    fb = t.get_window_extent(renderer=r).transformed(inv)
    fig.text(0.004, fb.y0, letter, fontsize=F_LET, fontweight="bold", va="bottom", ha="left")
note.set_x(axes[-1].get_position().x0)                           # note starts at the axes' left edge
for e in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"Figure2.{e}"), dpi=600)
print("saved Figure2 (180 x 175 mm)")
