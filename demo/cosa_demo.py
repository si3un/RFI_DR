"""Small demo of the consensus map and CoSA on synthetic saliency maps (CPU, a few seconds).

No real images, models or annotations are needed. The script creates one synthetic fundus field of view with
lesion blobs and 14 synthetic "configuration" saliency maps whose lesion focus varies from weak to strong.
It then uses the same functions as the paper's analysis (interpreters/ext_consensus_similarity_loo.py) to

  1. build the consensus map (equation 1: L1-normalised maps, trimmed mean discarding 2 values at each end),
  2. compute CoSA for every map (Spearman rank correlation and top-k overlap AUC with the consensus),
  3. score lesion localization with the per-case optimal Dice (threshold chosen per map to maximise Dice),

and prints whether the CoSA ranking follows the Dice ranking, as in the paper.

Usage:
  python demo/cosa_demo.py            # prints a table and writes demo/cosa_demo.png
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "interpreters"))
from ext_consensus_similarity_loo import consensus_rule, norm_auc, spearman_fov, topk_ious  # noqa: E402

RNG = np.random.default_rng(0)
H = W = 128
KS = [1, 2, 5, 10, 20]


def blobs(n, sigma, inside):
    """Random smooth blobs inside the field of view."""
    m = np.zeros((H, W))
    ys, xs = np.nonzero(inside)
    for i in RNG.choice(len(ys), n, replace=False):
        m[ys[i], xs[i]] = 1.0
    m = gaussian_filter(m, sigma)
    return m / m.max()


def minmax(a, fov):
    v = a[fov]
    return np.where(fov, (a - v.min()) / (v.max() - v.min() + 1e-12), 0.0)


def optimal_dice(sal, mask, fov):
    """Per-case optimal Dice: best Dice over thresholds that keep the top p% of FOV pixels."""
    v = sal[fov]
    best = 0.0
    for p in np.arange(0.5, 50.5, 0.5):
        thr = np.percentile(v, 100 - p)
        pred = (sal >= thr) & fov
        inter = (pred & mask).sum()
        best = max(best, 2 * inter / (pred.sum() + mask.sum()))
    return best


def main():
    yy, xx = np.mgrid[:H, :W]
    fov = (yy - H / 2) ** 2 + (xx - W / 2) ** 2 <= (0.49 * H) ** 2

    lesion_signal = blobs(6, 2.5, fov)
    lesion = (lesion_signal > 0.35) & fov                      # synthetic "expert annotation"
    context = gaussian_filter(lesion_signal, 10); context /= context.max()   # broad lesion-related context

    names, maps = [], []
    for k, w in enumerate(np.linspace(0.2, 0.6, 14)):          # w = how strongly a configuration focuses on lesions
        own = blobs(6, 4, fov)                                 # configuration-specific off-target regions
        noise = gaussian_filter(RNG.normal(size=(H, W)), 3)
        m = w * (lesion_signal + context) + (1 - w) * own + 0.05 * noise
        names.append("config_%02d (w=%.2f)" % (k + 1, w))
        maps.append(minmax(m, fov))
    Mk = np.stack(maps)

    consensus = consensus_rule("arith_trim2", Mk, fov)         # equation 1

    rows = []
    for n, m in zip(names, maps):
        rho = spearman_fov(m, consensus, fov)
        auc = norm_auc(topk_ious(m, consensus, fov, KS), KS)
        rows.append((n, rho, auc, optimal_dice(m, lesion, fov)))
    cons_dice = optimal_dice(consensus, lesion, fov)

    print("%-20s %10s %12s %10s" % ("map", "CoSA rho", "CoSA top-k", "Dice"))
    for n, rho, auc, d in rows:
        print("%-20s %10.3f %12.3f %10.3f" % (n, rho, auc, d))
    print("%-20s %10s %12s %10.3f" % ("consensus map", "-", "-", cons_dice))

    r = np.array([x[1] for x in rows]); a = np.array([x[2] for x in rows]); d = np.array([x[3] for x in rows])
    print("\nRank correlation with Dice: CoSA rho %.2f, CoSA top-k %.2f"
          % (spearmanr(r, d).correlation, spearmanr(a, d).correlation))
    print("Consensus Dice %.3f vs best individual %.3f" % (cons_dice, d.max()))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(-r)
        show = [("Lesion annotation", lesion.astype(float)), ("Consensus map", consensus)]
        show += [("Highest CoSA\n" + names[order[0]], maps[order[0]]), ("Lowest CoSA\n" + names[order[-1]], maps[order[-1]])]
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.8))
        for ax, (t, im) in zip(axs, show):
            ax.imshow(np.where(fov, im, np.nan), cmap="jet" if t != "Lesion annotation" else "gray")
            ax.set_title(t, fontsize=9); ax.axis("off")
        fig.tight_layout()
        out = os.path.join(HERE, "cosa_demo.png")
        fig.savefig(out, dpi=120)
        print("Figure written to %s" % out)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
