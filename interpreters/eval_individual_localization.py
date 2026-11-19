# -*- coding: utf-8 -*-
"""
eval_individual_localization.py -- per-case optimal Dice of the individual saliency maps (and alternative consensus rules),
and score every map with metrics other than Dice.

Rules (same pool = AUROC>=0.92, n=14; no GT used)
  rank family : RankMean, RankMedian, Q75Rank, HybridMeanQ75, UpperHalfMean, RankMax, RRA
  literature  : ZscoreMean (z-score + mean), UCB (mean-0.5*sd), AggVar (mean/sd)  [arXiv 2207.01565; Rieger&Hansen 2019]
  weighted    : RRF tau={0.005,0.02,0.05} (reciprocal rank fusion, Cormack 2009), per-image weights without GT -> ConcWeightedMean/Geo (entropy concentration, EWC-style),
                AgreeWeightedMean (leave-one-out Spearman agreement, Arbitrator-lite), EWC-lite (conc x agree),
                AUROCWeightedMean (fixed internal-AUROC weights), TrimmedTop7-Max, GeoMean Top7, NoisyOR, VoteGatedMean
  other       : RankMin (AND), Top{2,4,7}Mean (per-pixel mean of the m highest ranks), UnitSumMean (AGG-Mean),
                GeoMean raw (ViTMix-style multiplicative fusion of unit-sum maps), GeoMean rank,
                FamilyMean-backbone (7 backbones) / FamilyMean-lineage (4 lineages), vote at 1% added
  vote family : each model map binarised at top-t% (t = --vote_topk), then per-pixel count of models
                that fired (0..K). OR (>=1) / majority / AND (=K) are thresholds of this vote map.
     (a) VoteCount used as a saliency map -> per-case optimal top-k Dice (same protocol as other rules)
     (b) fixed binary rules OR / quarter / majority / three-quarter / AND -> Dice, IoU, PPV, Sens, area
         (no threshold sweep at all)

Metrics applied to every map
  max_dice (per-case optimal) . nauprc (prevalence-corrected AUPRC) . topk_enrich10 (fixed top-10%)
  pointing (is the single highest pixel inside a lesion) . dice_fixedk (one common fixed top-k%, --fixed_k)

Individual 21 configurations are scored with the same metrics (single_root = individual maps).
Outputs: {out_dir}/{dataset}_rules_per_case.csv, {dataset}_rules_summary.csv,
         {dataset}_binary_rules_per_case.csv, {dataset}_binary_rules.csv

Usage:
  python interpreters/eval_individual_localization.py --dataset idrid \
    --root $IDRID_SEG_ROOT \
    --sal_root outputs/idrid_saliency_attn --single_root outputs/idrid_saliency_attn \
    --model_roots models/dr_clean models/ckpts_aux models/_stage \
    --out_dir outputs/individual_localization --grid ultrafine
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
from threshold_sweep_erx import DISCOVER, DEF_TYPES, load_set
from optimal_threshold_eval import sweep_best
from consensus_newaxis_eval import hybrid_mean_q75, mm_rank_stack, rra_score
from threshold_optimal_k import make_grid


def norm1(a, fov):
    x = a * fov
    s = x.sum()
    return x / s if s > 1e-12 else x


def _block_sum(x, f):
    """mass-preserving downsampling of the last two axes by an integer factor f"""
    h, w = x.shape[-2] // f, x.shape[-1] // f
    return x[..., :h * f, :w * f].reshape(*x.shape[:-2], h, f, w, f).sum(axis=(-3, -1))


def _upsample(x, shape):
    from scipy.ndimage import zoom
    return zoom(x, (shape[0] / x.shape[0], shape[1] / x.shape[1]), order=1)


def wasserstein_barycenter(U, fov, f=4, sigma_px=14.0, iters=150, debiased=True):
    """Entropic Wasserstein barycenter of the unit-sum maps (Agueh & Carlier 2011; Cuturi & Doucet 2014), computed with
    the convolutional Sinkhorn algorithm on the image grid (Solomon et al. 2015) and optionally debiased (Janati et al.
    2020, ICML), equal weights. Maps are block-summed by f (448 -> 112) to keep it cheap and the result is upsampled.
    The Gaussian kernel width is fixed a priori at one ViT patch (sigma = 14 px at 448), no tuning."""
    A = _block_sum(U, f).astype(np.float64)
    A = A / np.maximum(A.sum(axis=(1, 2), keepdims=True), 1e-300)
    n = A.shape[-1]
    t = np.arange(n, dtype=np.float64)
    s = sigma_px / f
    Kx = np.exp(-(t[:, None] - t[None, :]) ** 2 / (2.0 * s * s))
    conv = lambda X: Kx @ X @ Kx.T
    K = A.shape[0]; w = 1.0 / K; tiny = 1e-300
    # same iteration as POT ot.bregman.convolutional_barycenter2d(_debiased)
    bar = np.ones(A.shape[1:]) / A[0].size
    KU = np.stack([conv(np.ones(A.shape[1:]))] * K)
    c = np.ones(A.shape[1:])
    for _ in range(iters):
        V = bar[None] / np.maximum(KU, tiny)
        KV = np.stack([conv(V[k]) for k in range(K)])
        Um = A / np.maximum(KV, tiny)
        KU = np.stack([conv(Um[k]) for k in range(K)])
        bar = np.exp((w * np.log(np.maximum(KU, tiny))).sum(0))
        if debiased:
            bar = c * bar
            c = np.sqrt(c * bar / np.maximum(conv(c), tiny))
    out = _upsample(bar, fov.shape)
    return norm1(np.clip(out, 0, None) * fov, fov)


def downsampled_geo(U, fov, f=4, drop=0, eps=1e-12):
    """control for wasserstein_barycenter: the same block-sum / upsample path applied to the (trimmed) geometric mean"""
    A = np.sort(_block_sum(U, f), axis=0)[drop:]
    g = np.exp(np.log(np.maximum(A, eps)).mean(0))
    return norm1(np.clip(_upsample(g, fov.shape), 0, None) * fov, fov)


def robust_log_pool(U, fi, kind="huber", iters=30, eps=1e-12):
    """per-pixel robust M-estimate of location of the K log-maps (Huber 1964, c = 1.345; Tukey biweight, c = 4.685;
    textbook 95%-efficiency constants), scale = 1.4826 x MAD across models at that pixel. An adaptive alternative to
    trimming a fixed number of values: members far from the others at a pixel are down-weighted, the rest averaged."""
    L = np.log(np.maximum(U[:, fi], eps))
    mu = np.median(L, axis=0)
    sc = np.maximum(1.4826 * np.median(np.abs(L - mu), axis=0), 1e-6)
    c = 1.345 if kind == "huber" else 4.685
    for _ in range(iters):
        r = (L - mu) / sc
        if kind == "huber":
            wt = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
        else:
            wt = np.where(np.abs(r) < c, (1.0 - (r / c) ** 2) ** 2, 0.0)
        sw = wt.sum(0)
        mu = np.where(sw > 0, (wt * L).sum(0) / np.maximum(sw, 1e-12), mu)
    out = np.zeros(fi.shape); out[fi] = np.exp(mu - mu.max())
    return out


def log_z_pool(U, fi, drop=0, eps=1e-12):
    """score normalisation before fusion (Jain, Nandakumar & Ross 2005, Pattern Recognit.): each log-map is robustly
    z-scored inside the FOV (median / 1.4826 MAD) so that every member has the same spread, then averaged
    (optionally dropping the `drop` lowest per pixel)"""
    L = np.log(np.maximum(U[:, fi], eps))
    med = np.median(L, axis=1, keepdims=True)
    mad = np.maximum(1.4826 * np.median(np.abs(L - med), axis=1, keepdims=True), 1e-6)
    Z = np.sort((L - med) / mad, axis=0)[drop:].mean(0)
    out = np.zeros(fi.shape); out[fi] = np.exp(Z - Z.max())
    return out


def q_rank(Rk, q):
    K = Rk.shape[0]
    srt = np.sort(Rk, axis=0)
    pos = q * (K - 1)
    lo = int(np.floor(pos)); hi = min(lo + 1, K - 1); fr = pos - lo
    return srt[lo] * (1.0 - fr) + srt[hi] * fr


def vote_count(Rk, fov, topk):
    """binarise each model's rank map at top-topk% -> per-pixel number of models that fired (0..K)"""
    thr = 1.0 - topk / 100.0
    B = (Rk >= thr) & fov[None]
    return B.sum(axis=0).astype(np.float64)


class FusedCams:
    """per-model map built on the fly from the 1.0x and 1.5x single-scale files:
    mean (= how *_ms8 was built), geo (sqrt of product, scale multiplication), max, s10 / s15 (one scale only)."""

    def __init__(self, cam1, idx1, cam2, idx2, fusion):
        self.cam1, self.idx1, self.cam2, self.idx2, self.fusion = cam1, idx1, cam2, idx2, fusion

    def keys(self):
        return {fn: fn for fn in self.idx1 if fn in self.idx2}

    def __getitem__(self, fn):
        a = self.cam1[self.idx1[fn]].float(); b = self.cam2[self.idx2[fn]].float()
        if self.fusion == "mean": return (a + b) / 2.0
        if self.fusion == "geo": return (a.clamp(min=0) * b.clamp(min=0)).sqrt()
        if self.fusion == "max": return torch.maximum(a, b)
        if self.fusion == "s10": return a
        if self.fusion == "s15": return b
        raise ValueError(self.fusion)


def load_scale_fused(root1, root2, fusion):
    n1, c1, f1 = load_set(root1); n2, c2, f2 = load_set(root2)
    names = [m for m in n1 if m in c2]
    cams = {m: FusedCams(c1[m], f1[m], c2[m], f2[m], fusion) for m in names}
    fidx = {m: cams[m].keys() for m in names}
    return names, cams, fidx


# Spearman agreement with the consensus on the SNUH external images (manuscript Table 3); fixed, label-agnostic weights
SNUH_AGREE = {"OpenCLIP-Partial": 0.149, "OpenCLIP-LoRA": 0.196, "DINOv2-Partial": 0.534, "DINOv2-LoRA": 0.569,
              "RETFound-DINOv2-Partial": 0.578, "RETFound-DINOv2-LoRA": 0.567, "DINOv3-Large-Partial": 0.525,
              "DINOv3-Large-LoRA": 0.463, "DINOv3-Huge-Partial": 0.402, "DINOv3-Huge-LoRA": 0.482,
              "MAE-Partial": 0.599, "MAE-LoRA": 0.453, "RETFound-Partial": 0.482, "RETFound-LoRA": 0.132}


def staple(Rk, fov, q, n_iter=15):
    """STAPLE (Warfield 2004) on the top-q% binarisation of each rank map; returns the posterior W (H x W)."""
    fi = fov.astype(bool)
    D = (Rk[:, fi] >= 1.0 - q / 100.0).astype(np.float64)           # K x N
    K, N = D.shape
    gamma = float(D.mean())                                           # scalar prior
    sens = np.full(K, 0.9); spec = np.full(K, 0.9)
    for _ in range(n_iter):
        la = np.log(gamma + 1e-12) + (D * np.log(sens + 1e-12)[:, None] + (1 - D) * np.log(1 - sens + 1e-12)[:, None]).sum(0)
        lb = np.log(1 - gamma + 1e-12) + (D * np.log(1 - spec + 1e-12)[:, None] + (1 - D) * np.log(spec + 1e-12)[:, None]).sum(0)
        W = 1.0 / (1.0 + np.exp(lb - la))
        sw = W.sum() + 1e-12; snw = (1 - W).sum() + 1e-12
        sens = np.clip((D * W).sum(1) / sw, 1e-4, 1 - 1e-4)
        spec = np.clip(((1 - D) * (1 - W)).sum(1) / snw, 1e-4, 1 - 1e-4)
    out = np.zeros(fov.shape, dtype=np.float64); out[fi] = W
    return out


def sml_weights(Q, n_iter=20):
    """Spectral Meta-Learner (Parisi et al., PNAS 2014): the off-diagonal of the predictor covariance Q is rank one,
    v v^T, with v_i proportional to predictor i's balanced accuracy. Estimate v by alternating between an
    eigendecomposition and re-filling the (unknown) diagonal with v_i^2. Returns non-negative weights summing to one."""
    Q = np.array(Q, dtype=np.float64); K = Q.shape[0]
    off = Q - np.diag(np.diag(Q))
    d = np.abs(off).sum(1) / max(K - 1, 1)
    for _ in range(n_iter):
        Qh = off + np.diag(d)
        w, V = np.linalg.eigh(Qh)
        v = V[:, -1] * np.sqrt(max(w[-1], 1e-12))
        if v.sum() < 0: v = -v
        d = v ** 2
    v = np.clip(v, 0.0, None)
    return v / v.sum() if v.sum() > 0 else np.ones(K) / K


def geometric_median(X, n_iter=20, eps=1e-8):
    """smoothed Weiszfeld algorithm (Pillutla et al. 2022): the point minimising the sum of Euclidean distances to
    the K vectors in X (K x N). Outlying vectors get small weights automatically; no parameters, no labels."""
    z = X.mean(0)
    for _ in range(n_iter):
        d = np.sqrt(((X - z[None]) ** 2).sum(1)) + eps
        w = 1.0 / d
        z = (w[:, None] * X).sum(0) / w.sum()
    return z


def quantile_norm_geo(MM, fi):
    X = MM[:, fi]; order = np.argsort(X, axis=1); prof = np.sort(X, axis=1).mean(0)
    Q = np.empty_like(X)
    for k in range(X.shape[0]):
        Q[k, order[k]] = prof
    out = np.zeros(MM.shape[1:], dtype=np.float64); out[fi] = np.exp(np.log(np.maximum(Q, 1e-12)).mean(0))
    return out


def quantile_norm_mean(MM, fi):
    """quantile normalisation over FOV pixels: each map's sorted values are replaced by the mean sorted profile"""
    X = MM[:, fi]                                   # K x N
    order = np.argsort(X, axis=1)
    prof = np.sort(X, axis=1).mean(0)               # mean sorted profile (N,)
    Q = np.empty_like(X)
    for k in range(X.shape[0]):
        Q[k, order[k]] = prof
    out = np.zeros(MM.shape[1:], dtype=np.float64); out[fi] = Q.mean(0)
    return out


def local_geo(U, fov, r, eps):
    from scipy.ndimage import maximum_filter
    D = np.stack([maximum_filter(u, size=2 * r + 1) for u in U], 0) * fov[None]
    return np.exp(np.log(np.maximum(D, eps)).mean(0)) * fov


def comb_mnz(U, Rk, fov, q):
    M = (Rk >= 1.0 - q / 100.0) & fov[None]
    V = U * M
    return V.sum(0) * M.sum(0)


def binary_metrics(mask, Lm, fov):
    m = mask & fov; g = Lm & fov
    inter = float(np.logical_and(m, g).sum()); nm = float(m.sum()); ng = float(g.sum())
    dice = 2 * inter / (nm + ng) if nm + ng > 0 else np.nan
    iou = inter / (nm + ng - inter) if nm + ng - inter > 0 else np.nan
    return dict(dice=dice, iou=iou, ppv=(inter / nm if nm > 0 else np.nan),
                sens=(inter / ng if ng > 0 else np.nan), area_frac=nm / float(fov.sum()))


def pointing(A, Lm, fov):
    s = np.where(fov, A, -np.inf)
    return float(Lm.ravel()[int(np.argmax(s))])


def dice_at_fixed_k(A, Lm, fov, k):
    fi = np.flatnonzero(fov.ravel()); s = A.ravel()[fi]; Lf = Lm.ravel()[fi].astype(bool)
    n = fi.size; N = max(1, int(round(k / 100.0 * n)))
    idx = np.argpartition(-s, N - 1)[:N]
    inter = float(Lf[idx].sum()); G = float(Lf.sum())
    return 2 * inter / (N + G) if N + G > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DISCOVER))
    ap.add_argument("--root", required=True)
    ap.add_argument("--sal_root", default="")
    ap.add_argument("--single_root", required=True)
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs/individual_localization")
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--auc_min", type=float, default=0.92)
    ap.add_argument("--grid", default="ultrafine")
    ap.add_argument("--vote_topk", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--fixed_k", type=float, default=2.0)
    ap.add_argument("--no_individuals", action="store_true")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--scale1_root", default="", help="1.0x single-scale maps (*_saliency_attn)")
    ap.add_argument("--scale2_root", default="", help="1.5x single-scale maps (*_saliency_15only)")
    ap.add_argument("--scale_fusion", default="mean", choices=["mean", "geo", "max", "s10", "s15"])
    ap.add_argument("--stab_root1", default="", help="1.0x maps for the scale-stability rules")
    ap.add_argument("--stab_root2", default="", help="1.5x maps for the scale-stability rules")
    ap.add_argument("--null_root", default="", help="classifier-only (unadapted) maps: their geometric mean is a label-free null/background attention map; consensus is divided by it")
    ap.add_argument("--sml", action="store_true", help="add Spectral Meta-Learner weighted rules (per-image and dataset-global covariance)")
    ap.add_argument("--faith_csv", default="", help="faithfulness_per_image.csv (faithfulness_score.py): deletion-based, label-free per-image model reliability")
    ap.add_argument("--pred_csv", default="", help="per-image referable-DR predictions of every model (collect_pool_predictions.py); enables confidence-weighted rules")
    ap.add_argument("--rules", nargs="+", default=None, help="evaluate only rules whose name contains one of these substrings")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    KS = make_grid(a.grid, 0.01, 99.0, 0.01)
    types = a.types or DEF_TYPES[a.dataset]
    items = sorted(DISCOVER[a.dataset](a.root, types), key=lambda r: r["image_id"])
    if a.max_cases:
        items = items[:a.max_cases]
    auc = SC.load_auc_map(a.model_roots)

    if a.scale1_root and a.scale2_root:
        cn, ccams, cfidx = load_scale_fused(a.scale1_root, a.scale2_root, a.scale_fusion)
        print("scale fusion: %s  (1.0x = %s, 1.5x = %s)" % (a.scale_fusion, a.scale1_root, a.scale2_root))
    else:
        cn, ccams, cfidx = load_set(a.sal_root)
    cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
    POOL = [m for m in sorted(cn, key=lambda m: -auc.get(m, 0)) if auc.get(m, 0) >= a.auc_min]
    K = len(POOL)
    IND, icams, ifidx = [], {}, {}
    if not a.no_individuals:
        # individual multi-scale maps: LoRA / partial configs live in sal_root (same files as the pool),
        # classifier-only configs live in single_root (the *_cls folder). Merge both -> 21 configurations.
        inames = list(cn); icams = dict(ccams); ifidx = dict(cfidx)
        if a.single_root:
            n2, c2, f2 = load_set(a.single_root)
            for m in n2:
                if m not in icams:
                    inames.append(m); icams[m] = c2[m]; ifidx[m] = f2[m]
        IND = sorted([m for m in inames if C.lineage_of(m) and L.ft_of(m)], key=lambda m: -(auc.get(m) or 0))
    itag = {m: "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in IND}
    # families fixed a priori from pretraining pedigree (not from results): 7 backbones / 4 lineages
    fam_bb, fam_lin = {}, {}
    for j, m in enumerate(POOL):
        fam_bb.setdefault(L.backbone_pretty(m), []).append(j)
        fam_lin.setdefault(C.lineage_of(m), []).append(j)
    print("families (backbone):", {k: len(v) for k, v in fam_bb.items()})
    print("families (lineage) :", {k: len(v) for k, v in fam_lin.items()})
    print("[%s] cases %d | pool %d | individuals %d | vote top-k %s" % (a.dataset, len(items), K, len(IND), a.vote_topk))

    ipath = {r["image_id"]: r["image_path"] for r in items}
    fov_cache = {}
    PRED = {}
    if a.pred_csv:
        Pd = pd.read_csv(a.pred_csv)
        Pd = Pd[Pd.dataset == a.dataset] if "dataset" in Pd.columns else Pd
        PRED = {(str(m), str(st)): float(v) for m, st, v in zip(Pd.model, Pd.stem, Pd.pred)}
        hit = [m for m in POOL if any((m, os.path.splitext(r["image_id"])[0]) in PRED for r in items[:50])]
        print("confidence weights: %d predictions | pool models matched: %d / %d" % (len(PRED), len(hit), K))
    n_conf_missing = [0]
    NULL = None
    if a.null_root:
        nn_, nc_, nf_ = load_set(a.null_root)
        NULLM = [m for m in nn_ if L.ft_of(m) == "Linear" and C.lineage_of(m)]
        NULL = (NULLM, nc_, nf_)
        print("null-model correction: %d classifier-only maps from %s" % (len(NULLM), a.null_root))
    W_SML = None
    if a.sml:                                   # first pass: dataset-level covariance of the rank maps (label-free)
        Qsum = np.zeros((K, K)); nimg = 0; seen = set()
        for r in items:
            fn = r["image_id"]
            if fn in seen or not all(fn in cfidx[m] for m in POOL): continue
            seen.add(fn)
            if fn not in fov_cache:
                fov_cache[fn] = C.FOV & E.fundus_mask_448(E.letterbox_img(ipath[fn]))
            fov0 = fov_cache[fn]; fi0 = fov0.astype(bool)
            _, Rk0 = mm_rank_stack(POOL, fn, ccams, cfidx, fov0)
            Qsum += np.cov(Rk0[:, fi0]); nimg += 1
        W_SML = sml_weights(Qsum / max(nimg, 1))
        print("SML global weights (%d images):" % nimg, {("%s-%s" % (L.backbone_pretty(m), L.ft_of(m))): round(float(w), 3) for m, w in zip(POOL, W_SML)})
    FAITH, FAITH_G = {}, {}
    if a.faith_csv:
        Fd = pd.read_csv(a.faith_csv)
        FAITH = {(str(m), str(i)): float(v) for m, i, v in zip(Fd.model, Fd.image_id, Fd.faith)}
        FAITH_G = Fd.groupby("model").faith.mean().to_dict()
        print("faithfulness: %d per-image values | pool models covered: %d / %d" % (len(FAITH), sum(m in FAITH_G for m in POOL), K))
    STAB = None
    if a.stab_root1 and a.stab_root2:
        _, sc1, sf1 = load_set(a.stab_root1); _, sc2, sf2 = load_set(a.stab_root2); STAB = (sc1, sf1, sc2, sf2)
        print("scale-stability rules enabled (%s vs %s)" % (a.stab_root1, a.stab_root2))
    bb_pool = [L.backbone_pretty(m) for m in POOL]
    partner = [next((j for j, m2 in enumerate(POOL) if j != k and bb_pool[j] == bb_pool[k]), None) for k in range(K)]

    def fov_of(f):
        if f not in fov_cache:
            fov_cache[f] = C.FOV & E.fundus_mask_448(E.letterbox_img(ipath[f]))
        return fov_cache[f]

    BIN_RULES = (("OR (>=1)", 1), ("Quarter (>=K/4)", int(np.ceil(K / 4.0))), ("Majority (>=K/2)", int(np.ceil(K / 2.0))),
                 ("ThreeQuarter (>=3K/4)", int(np.ceil(3.0 * K / 4.0))), ("AND (=K)", K))
    rows, brows, skipped = [], [], 0
    cur_fn, cur_maps, cur_votes = None, None, None
    for i, r in enumerate(items):
        fn, lt = r["image_id"], r["lesion_type"]
        if not all(fn in cfidx[m] for m in POOL) or (IND and not all(fn in ifidx.get(m, {}) for m in IND)):
            skipped += 1; continue
        fov = fov_of(fn)
        Lm = E.letterbox_mask(r["mask_path"]) & fov
        if Lm.sum() == 0:
            skipped += 1; continue
        if fn != cur_fn:
            MM, Rk = mm_rank_stack(POOL, fn, ccams, cfidx, fov)
            srt = np.sort(Rk, axis=0)
            mean = Rk.mean(0); sd = Rk.std(0)
            # literature rules: z-score normalisation + mean (arXiv 2207.01565), UCB eps=-0.5 (same paper),
            # AGG-Var = mean / local std (Rieger & Hansen 2019)
            fi = fov.astype(bool)
            Z = np.stack([(m - m[fi].mean()) / (m[fi].std() + 1e-8) for m in MM], 0) * fov
            zmean = Z.mean(0)
            # unit-sum normalised raw maps (AGG-Mean, Rieger & Hansen) and their geometric mean (ViTMix-style fusion)
            U = np.stack([norm1(m, fov) for m in MM], 0)
            eps_u = 1e-12
            geo_raw = np.exp(np.log(np.maximum(U, eps_u)).mean(0)) * fov
            geo_rank = np.exp(np.log(np.maximum(Rk, 1e-6)).mean(0)) * fov
            # family-balanced means: average rank maps within a family, then average the family maps equally
            fam_maps = {}
            for key, groups in (("FamilyMean-backbone", fam_bb), ("FamilyMean-lineage", fam_lin)):
                fam_maps[key] = np.mean([Rk[idx].mean(0) for idx in groups.values()], axis=0)
            # ---- v4: per-image weights without GT ----------------------------------------------------
            Uf = U[:, fi]                                                    # K x Nfov (unit-sum maps)
            Nfov = float(fi.sum())
            Hk = -(Uf * np.log(Uf + 1e-12)).sum(1)                           # entropy per map
            conc = np.maximum(np.log(Uf.shape[1]) - Hk, 1e-6)                # concentration (EWC-style)
            w_conc = conc / conc.sum()
            Cm = np.corrcoef(Rk[:, fi])                                      # Spearman between rank maps
            agree = np.clip((Cm.sum(1) - 1.0) / (K - 1), 1e-6, None)         # mean agreement with the others
            w_agree = agree / agree.sum()
            w_ewc = w_conc * w_agree; w_ewc /= w_ewc.sum()
            w_auc = np.array([auc.get(m, 0.0) - a.auc_min + 0.01 for m in POOL]); w_auc /= w_auc.sum()
            srtU = np.sort(U, axis=0)
            _gm_raw = geometric_median(U[:, fi]); gm_raw_full = np.zeros(fov.shape); gm_raw_full[fi] = np.clip(_gm_raw, 0, None)
            _gm_log = geometric_median(np.log(np.maximum(U[:, fi], eps_u))); gm_log_full = np.zeros(fov.shape); gm_log_full[fi] = np.exp(_gm_log)
            bb_max = np.stack([U[idx].max(0) for idx in fam_bb.values()], 0)
            bb_mean = np.stack([U[idx].mean(0) for idx in fam_bb.values()], 0)
            w_snuh = np.array([SNUH_AGREE.get("%s-%s" % (L.backbone_pretty(m), L.ft_of(m)), 0.0) for m in POOL]); w_snuh /= w_snuh.sum()
            stab_rules = {}
            fi_b = fov.astype(bool)
            logU_s = np.log(np.maximum(U, eps_u))
            adapt = np.array([np.corrcoef(Rk[k][fi_b], Rk[partner[k]][fi_b])[0, 1] if partner[k] is not None else np.nan for k in range(K)])
            adapt = np.where(np.isnan(adapt), np.nanmean(adapt), adapt); wa = np.clip(adapt, 1e-3, None) ** 2; wa /= wa.sum()
            stab_rules["AdaptStabWeightedGeo p=2"] = norm1(np.exp(np.tensordot(wa, logU_s, 1)) * fov, fov)
            stab_rules["AdaptStabGateGeo (upper half)"] = norm1(np.exp(logU_s[np.argsort(-adapt, kind="stable")[:K // 2]].mean(0)) * fov, fov)
            if STAB is not None and all(fn in STAB[1][m] and fn in STAB[3][m] for m in POOL):
                from scipy.stats import rankdata as _rd
                stab = []
                for m in POOL:
                    m1 = E.mm(STAB[0][m][STAB[1][m][fn]].numpy())[fi_b]; m2 = E.mm(STAB[2][m][STAB[3][m][fn]].numpy())[fi_b]
                    stab.append(np.corrcoef(_rd(m1), _rd(m2))[0, 1])
                stab = np.array(stab); ws = np.clip(stab, 1e-3, None) ** 2; ws /= ws.sum()
                stab_rules["ScaleStabWeightedGeo p=2"] = norm1(np.exp(np.tensordot(ws, logU_s, 1)) * fov, fov)
                stab_rules["ScaleStabGateGeo (upper half)"] = norm1(np.exp(logU_s[np.argsort(-stab, kind="stable")[:K // 2]].mean(0)) * fov, fov)
            if FAITH and all((m, fn) in FAITH for m in POOL):
                fa = np.array([FAITH[(m, fn)] for m in POOL]); fg = np.array([FAITH_G[m] for m in POOL])
                wf = np.clip(fa, 0.0, None) + 1e-3; wf /= wf.sum()
                wfg = np.clip(fg, 0.0, None) + 1e-3; wfg /= wfg.sum()
                stab_rules["FaithWeightedGeo (per image)"] = norm1(np.exp(np.tensordot(wf, logU_s, 1)) * fov, fov)
                stab_rules["FaithWeightedGeo (global mean)"] = norm1(np.exp(np.tensordot(wfg, logU_s, 1)) * fov, fov)
                stab_rules["FaithGateGeo (upper half, per image)"] = norm1(np.exp(logU_s[np.argsort(-fa, kind="stable")[:K // 2]].mean(0)) * fov, fov)
                stab_rules["FaithGateGeo (upper half, global)"] = norm1(np.exp(logU_s[np.argsort(-fg, kind="stable")[:K // 2]].mean(0)) * fov, fov)
            if NULL is not None and all(fn in NULL[2][m] for m in NULL[0]):
                Un = np.stack([norm1(E.mm(NULL[1][m][NULL[2][m][fn]].numpy()) * fov, fov) for m in NULL[0]], 0)
                null_geo = np.exp(np.log(np.maximum(Un, eps_u)).mean(0)) * fov
                null_geo = null_geo / (null_geo[fi_b].mean() + 1e-12)                       # mean 1 over FOV
                g_raw = np.exp(logU_s.mean(0)) * fov
                g_d2 = np.exp(np.log(np.maximum(srtU[2:], eps_u)).mean(0)) * fov
                for pw in (0.5, 1.0):
                    stab_rules["GeoMean raw / Null^%g" % pw] = norm1(g_raw / (null_geo ** pw + 1e-12), fov)
                    stab_rules["GeoMean drop2 / Null^%g" % pw] = norm1(g_d2 / (null_geo ** pw + 1e-12), fov)
                    for m in IND:
                        if itag[m] in ("RETFound-DINOv2-LoRA", "DINOv2-LoRA"):
                            um = norm1(E.mm(icams[m][ifidx[m][fn]].numpy()) * fov, fov)
                            stab_rules["%s / Null^%g" % (itag[m], pw)] = norm1(um / (null_geo ** pw + 1e-12), fov)
            if a.sml:
                w_img = sml_weights(np.cov(Rk[:, fi_b]))
                stab_rules["SMLWeightedGeo (per image)"] = norm1(np.exp(np.tensordot(w_img, logU_s, 1)) * fov, fov)
                stab_rules["SMLWeightedMean (rank, per image)"] = norm1(np.tensordot(w_img, Rk, 1), fov)
                stab_rules["SMLWeightedGeo (global)"] = norm1(np.exp(np.tensordot(W_SML, logU_s, 1)) * fov, fov)
                stab_rules["SMLWeightedMean (rank, global)"] = norm1(np.tensordot(W_SML, Rk, 1), fov)
                stab_rules["SMLGateGeo (upper half, global)"] = norm1(np.exp(logU_s[np.argsort(-W_SML, kind="stable")[:K // 2]].mean(0)) * fov, fov)
                stab_rules["SMLGateGeo drop2 (global)"] = norm1(np.exp(np.log(np.maximum(np.sort(U[np.argsort(-W_SML, kind="stable")[:K // 2]], axis=0)[2:], eps_u)).mean(0)) * fov, fov)
            conf_rules = dict(stab_rules)
            if PRED:
                stem = os.path.splitext(fn)[0]
                pc = np.array([PRED.get((m, stem), np.nan) for m in POOL])
                if np.isnan(pc).any():
                    n_conf_missing[0] += 1; pc = np.where(np.isnan(pc), np.nanmean(pc) if np.isfinite(np.nanmean(pc)) else 0.5, pc)
                pc = np.clip(pc, 1e-3, 1.0)
                w1 = pc / pc.sum(); w2 = pc ** 2 / (pc ** 2).sum(); wca = pc * w_snuh; wca /= wca.sum()
                top5 = np.argsort(-pc)[:5]
                logU = np.log(np.maximum(U, eps_u))
                conf_rules.update({
                    "ConfWeightedGeo": norm1(np.exp(np.tensordot(w1, logU, 1)) * fov, fov),
                    "ConfWeightedGeo p=2": norm1(np.exp(np.tensordot(w2, logU, 1)) * fov, fov),
                    "ConfTop5Geo": norm1(np.exp(logU[top5].mean(0)) * fov, fov),
                    "ConfxAgreeGeo": norm1(np.exp(np.tensordot(wca, logU, 1)) * fov, fov),
                    "ConfWeightedMean (raw)": norm1(np.tensordot(w1, U, 1), fov),
                })
            w_snuh2 = w_snuh ** 2; w_snuh2 /= w_snuh2.sum()
            w_snuh4 = w_snuh ** 4; w_snuh4 /= w_snuh4.sum()
            w_snuh6 = w_snuh ** 6; w_snuh6 /= w_snuh6.sum()
            w_snuh8 = w_snuh ** 8; w_snuh8 /= w_snuh8.sum()
            gate_top = np.argsort(-w_snuh, kind="stable")[:K // 2]          # the K/2 pool members with the highest SNUH agreement
            owa_w_lin = np.arange(1, K + 1, dtype=np.float64); owa_w_lin /= owa_w_lin.sum()
            owa_w_quad = np.arange(1, K + 1, dtype=np.float64) ** 2; owa_w_quad /= owa_w_quad.sum()
            vote5 = vote_count(Rk, fov, 5.0) / float(K)
            cur_maps = {
                "ConcWeightedMean (raw)": norm1(np.tensordot(w_conc, U, 1), fov),
                "ConcWeightedGeo (raw)": norm1(np.exp(np.tensordot(w_conc, np.log(np.maximum(U, eps_u)), 1)) * fov, fov),
                "AgreeWeightedMean (rank)": norm1(np.tensordot(w_agree, Rk, 1), fov),
                "EWC-lite (conc x agree, raw)": norm1(np.tensordot(w_ewc, U, 1), fov),
                "AUROCWeightedMean (rank)": norm1(np.tensordot(w_auc, Rk, 1), fov),
                "TrimmedTop7-Max (rank)": norm1(srt[K // 2:K - 1].mean(0), fov),
                "GeoMean Top7 (raw)": norm1(np.exp(np.log(np.maximum(srtU[K // 2:], eps_u)).mean(0)) * fov, fov),
                "NoisyOR (minmax)": norm1(1.0 - np.prod(1.0 - np.clip(MM, 0, 1), axis=0), fov),
                "VoteGatedMean (mean x vote5%)": norm1(mean * (vote5 + 1e-3), fov),
                # trimmed geometric mean (drop the t lowest values per pixel), quantile-normalised geometric mean,
                #     STAPLE posterior (Warfield 2004) on top-q% binarisations, SNUH-agreement-weighted geometric mean
                "GeoMean drop1": norm1(np.exp(np.log(np.maximum(srtU[1:], eps_u)).mean(0)) * fov, fov),
                "GeoMean drop2": norm1(np.exp(np.log(np.maximum(srtU[2:], eps_u)).mean(0)) * fov, fov),
                "GeoMean drop2 a=0.5": norm1(np.exp(np.log(0.5 * srtU[2:] + 0.5 / Nfov).mean(0)) * fov, fov),
                "GeoMean drop2 a=0.2": norm1(np.exp(np.log(0.8 * srtU[2:] + 0.2 / Nfov).mean(0)) * fov, fov),
                # geometric median of the K maps (robust aggregation, Pillutla et al. 2022): raw space and log space
                "GeoMedian (raw, Weiszfeld)": norm1(np.where(fov, gm_raw_full, 0.0), fov),
                "GeoMedian (log, Weiszfeld)": norm1(np.where(fov, gm_log_full, 0.0), fov),
                # Winsorized geometric mean (Jose & Winkler 2008): raise the k lowest values per pixel to the (k+1)-th lowest
                "GeoMean winsor2": norm1(np.exp(np.log(np.maximum(np.maximum(U, srtU[2][None]), eps_u)).mean(0)) * fov, fov),
                "GeoMean winsor4": norm1(np.exp(np.log(np.maximum(np.maximum(U, srtU[4][None]), eps_u)).mean(0)) * fov, fov),
                **{"GeoMean drop%d" % t: norm1(np.exp(np.log(np.maximum(srtU[t:], eps_u)).mean(0)) * fov, fov) for t in (5, 6, 7, 8, 9, 10, 11, 12)},
                **{"GeoMean winsor%d" % t: norm1(np.exp(np.log(np.maximum(np.maximum(U, srtU[t][None]), eps_u)).mean(0)) * fov, fov) for t in (6, 8, 10)},
                "GeoMean drop4": norm1(np.exp(np.log(np.maximum(srtU[4:], eps_u)).mean(0)) * fov, fov),
                "GeoMean drop4 a=0.5": norm1(np.exp(np.log(0.5 * srtU[4:] + 0.5 / Nfov).mean(0)) * fov, fov),
                "GeoMean drop3 a=0.5": norm1(np.exp(np.log(0.5 * srtU[3:] + 0.5 / Nfov).mean(0)) * fov, fov),
                "RawMedian": norm1(q_rank(U, 0.5), fov),
                "GeoMean drop3": norm1(np.exp(np.log(np.maximum(srtU[3:], eps_u)).mean(0)) * fov, fov),
                "QuantileNormGeo": norm1(quantile_norm_geo(MM, fi), fov),
                "STAPLE q=1%": norm1(staple(Rk, fov, 1.0) + 1e-6 * mean, fov),
                "STAPLE q=2%": norm1(staple(Rk, fov, 2.0) + 1e-6 * mean, fov),
                "STAPLE q=5%": norm1(staple(Rk, fov, 5.0) + 1e-6 * mean, fov),
                "AgreeWeightedGeo (SNUH, global)": norm1(np.exp(np.tensordot(w_snuh, np.log(np.maximum(U, eps_u)), 1)) * fov, fov),
                **conf_rules,
                "AgreeWeightedGeo p=2": norm1(np.exp(np.tensordot(w_snuh2, np.log(np.maximum(U, eps_u)), 1)) * fov, fov),
                "AgreeWeightedGeo p=4": norm1(np.exp(np.tensordot(w_snuh4, np.log(np.maximum(U, eps_u)), 1)) * fov, fov),
                "AgreeWeightedGeo p=6": norm1(np.exp(np.tensordot(w_snuh6, np.log(np.maximum(U, eps_u)), 1)) * fov, fov),
                "AgreeWeightedGeo p=8": norm1(np.exp(np.tensordot(w_snuh8, np.log(np.maximum(U, eps_u)), 1)) * fov, fov),
                "AgreeGateGeo (upper half)": norm1(np.exp(np.log(np.maximum(U[gate_top], eps_u)).mean(0)) * fov, fov),
                "AgreeGateGeo (upper half) a=0.5": norm1(np.exp(np.log(0.5 * U[gate_top] + 0.5 / Nfov).mean(0)) * fov, fov),
                "AgreeWeightedGeo p=2 a=0.5": norm1(np.exp(np.tensordot(w_snuh2, np.log(0.5 * U + 0.5 / Nfov), 1)) * fov, fov),
                # OWA (Yager 1988): weights on the sorted values, increasing towards the highest rank
                "OWA linear (rank)": norm1(np.tensordot(owa_w_lin, srt, 1), fov),
                "OWA quadratic (rank)": norm1(np.tensordot(owa_w_quad, srt, 1), fov),
                "OWA linear (raw)": norm1(np.tensordot(owa_w_lin, srtU, 1), fov),
                # quantile normalisation (Bolstad 2003): give every map the same value distribution, then average
                "QuantileNormMean": norm1(quantile_norm_mean(MM, fi), fov),
                # backbone-level fusion: per pixel take the stronger of the two adaptations of each backbone (7 backbones),
                # then geometric mean across backbones (optionally trimming the lowest backbone value per pixel)
                "BackboneMaxGeo": norm1(np.exp(np.log(np.maximum(bb_max, eps_u)).mean(0)) * fov, fov),
                "BackboneMaxGeo drop1": norm1(np.exp(np.log(np.maximum(np.sort(bb_max, axis=0)[1:], eps_u)).mean(0)) * fov, fov),
                "BackboneMaxGeo drop2": norm1(np.exp(np.log(np.maximum(np.sort(bb_max, axis=0)[2:], eps_u)).mean(0)) * fov, fov),
                "BackboneMeanGeo": norm1(np.exp(np.log(np.maximum(bb_mean, eps_u)).mean(0)) * fov, fov),
                "BackboneMaxGeo a=0.5": norm1(np.exp(np.log(0.5 * bb_max + 0.5 / Nfov).mean(0)) * fov, fov),
                # spatially tolerant agreement: local max-pool each unit-sum map (radius r px) before the geometric
                # mean, so that peaks offset by patch-grid differences (14 vs 16 px backbones, 1.5x scale) still count
                "LocalGeo r=2": norm1(local_geo(U, fov, 2, eps_u), fov),
                "LocalGeo r=4": norm1(local_geo(U, fov, 4, eps_u), fov),
                "LocalGeo x Mean r=2": norm1(local_geo(U, fov, 2, eps_u) * U.mean(0), fov),
                "LocalGeo x Mean r=4": norm1(local_geo(U, fov, 4, eps_u) * U.mean(0), fov),
                # lineage-level agreement: max within each lineage (members share grids/representations), geo across lineages
                "FamilyMaxGeo (lineage)": norm1(np.exp(np.mean([np.log(np.maximum(U[idx].max(0), eps_u)) for idx in fam_lin.values()], 0)) * fov, fov),
                # CombMNZ (Fox & Shaw 1994; Lee 1997): keep each model's top-q% pixels of its unit-sum map,
                # then sum of retained scores x number of models that retained the pixel
                "CombMNZ q=2%": norm1(comb_mnz(U, Rk, fov, 2.0), fov),
                "CombMNZ q=5%": norm1(comb_mnz(U, Rk, fov, 5.0), fov),
                "CombMNZ q=10%": norm1(comb_mnz(U, Rk, fov, 10.0), fov),
                # floored geometric mean: mix alpha of a uniform map (1/N over FOV) into each unit-sum map first
                "GeoMean raw a=0.05": norm1(np.exp(np.log(0.95 * U + 0.05 / Nfov).mean(0)) * fov, fov),
                "GeoMean raw a=0.2": norm1(np.exp(np.log(0.8 * U + 0.2 / Nfov).mean(0)) * fov, fov),
                "GeoMean raw a=0.5": norm1(np.exp(np.log(0.5 * U + 0.5 / Nfov).mean(0)) * fov, fov),
                # Reciprocal rank fusion (Cormack 2009), continuous form: q = 1 - percentile rank (0 = top pixel)
                "RRF tau=0.005": norm1((0.005 / (0.005 + (1.0 - Rk))).mean(0), fov),
                "RRF tau=0.02": norm1((0.02 / (0.02 + (1.0 - Rk))).mean(0), fov),
                "RRF tau=0.05": norm1((0.05 / (0.05 + (1.0 - Rk))).mean(0), fov),
                "RankMin (AND)": norm1(srt[0], fov),
                "Top2Mean": norm1(srt[-2:].mean(0), fov),
                "Top4Mean": norm1(srt[-4:].mean(0), fov),
                "Top7Mean": norm1(srt[-7:].mean(0), fov),
                "UnitSumMean (AGG-Mean)": norm1(U.mean(0), fov),
                "MinMaxMean (raw)": norm1(MM.mean(0), fov),
                "PowerMean p=0.5 (raw)": norm1(np.sqrt(U).mean(0) ** 2, fov),
                "GeoMean raw (ViTMix)": norm1(geo_raw, fov),
                "GeoMean rank": norm1(geo_rank, fov),
                "FamilyMean-backbone": norm1(fam_maps["FamilyMean-backbone"], fov),
                "FamilyMean-lineage": norm1(fam_maps["FamilyMean-lineage"], fov),
                "ZscoreMean (raw)": norm1(zmean - zmean[fi].min(), fov),
                "UCB mean-0.5sd": norm1(np.clip(mean - 0.5 * sd, 0, None), fov),
                "AggVar mean/sd": norm1(mean / (sd + 0.05), fov),
                "RankMean": norm1(mean, fov),
                "RankMedian": norm1(q_rank(Rk, 0.5), fov),
                "Q75Rank": norm1(q_rank(Rk, 0.75), fov),
                "HybridMeanQ75": hybrid_mean_q75(Rk, fov),
                "UpperHalfMean": norm1(srt[K // 2:].mean(0), fov),
                "RankMax": norm1(srt[-1], fov),
                "RRA": norm1(rra_score(Rk, fov), fov),
            }
            cur_votes = {}
            for t in a.vote_topk:
                vc = vote_count(Rk, fov, t)
                cur_maps["VoteCount@top%g%%" % t] = norm1(vc + 1e-6 * mean, fov)   # ties broken by mean rank
                cur_votes[t] = vc
            # v11 (literature round 2026-09-25), computed only when requested because the barycenter is slower
            want = lambda name: (not a.rules) or any(sub.lower() in name.lower() for sub in a.rules)
            if want("WassBary conv (Solomon 2015)"):
                cur_maps["WassBary conv (Solomon 2015)"] = wasserstein_barycenter(U, fov, debiased=False)
            if want("WassBary debiased (Janati 2020)"):
                cur_maps["WassBary debiased (Janati 2020)"] = wasserstein_barycenter(U, fov, debiased=True)
            if want("GeoMean raw @112"):
                cur_maps["GeoMean raw @112"] = downsampled_geo(U, fov, drop=0)
            if want("GeoMean drop2 @112"):
                cur_maps["GeoMean drop2 @112"] = downsampled_geo(U, fov, drop=2)
            if want("HuberLogPool c=1.345"):
                cur_maps["HuberLogPool c=1.345"] = norm1(robust_log_pool(U, fi, "huber"), fov)
            if want("TukeyLogPool c=4.685"):
                cur_maps["TukeyLogPool c=4.685"] = norm1(robust_log_pool(U, fi, "tukey"), fov)
            if want("LogZ-MAD mean"):
                cur_maps["LogZ-MAD mean"] = norm1(log_z_pool(U, fi, 0), fov)
            if want("LogZ-MAD drop2"):
                cur_maps["LogZ-MAD drop2"] = norm1(log_z_pool(U, fi, 2), fov)
            if a.rules:
                cur_maps = {k: v for k, v in cur_maps.items() if any(sub.lower() in k.lower() for sub in a.rules)}
            for m in IND:
                cur_maps[itag[m]] = E.mm(icams[m][ifidx[m][fn]].numpy()) * fov
            cur_fn = fn
        for name, A in cur_maps.items():
            sb = sweep_best(A, Lm, fov, KS)
            if sb is None:
                continue
            rows.append(dict(dataset=a.dataset, image_id=fn, lesion_type=lt, method=name,
                             kind=("individual" if name in itag.values() else "consensus"),
                             max_dice=sb["max_dice"], best_k=sb["best_topk_percent"],
                             nauprc=C.nauprc(A, Lm, fov), topk_enrich10=C.topk_enrich_fixed(A, Lm, fov, 0.10),
                             pointing=pointing(A, Lm, fov), dice_fixedk=dice_at_fixed_k(A, Lm, fov, a.fixed_k)))
        for t, vc in cur_votes.items():
            for rule, thr in BIN_RULES:
                brows.append(dict(dataset=a.dataset, image_id=fn, lesion_type=lt, vote_topk=t, rule=rule,
                                  **binary_metrics(vc >= thr, Lm, fov)))
        if (i + 1) % 300 == 0:
            print("  ... %d/%d" % (i + 1, len(items)))

    if PRED:
        print("images with a missing confidence value (filled with the image mean): %d" % n_conf_missing[0])
    df = pd.DataFrame(rows); bd = pd.DataFrame(brows)
    df.to_csv(os.path.join(a.out_dir, "%s_rules_per_case.csv" % a.dataset), index=False)
    bd.to_csv(os.path.join(a.out_dir, "%s_binary_rules_per_case.csv" % a.dataset), index=False)
    print("cases evaluated %d | skipped %d" % (df.groupby(["image_id", "lesion_type"]).ngroups, skipped))

    S = df.groupby(["kind", "method"]).agg(n=("max_dice", "size"), dice=("max_dice", "mean"), nauprc=("nauprc", "mean"),
                                           enrich10=("topk_enrich10", "mean"), pointing=("pointing", "mean"),
                                           dice_fixedk=("dice_fixedk", "mean"), best_k_med=("best_k", "median")).reset_index()
    S = S.sort_values(["kind", "dice"], ascending=[True, False])
    S.to_csv(os.path.join(a.out_dir, "%s_rules_summary.csv" % a.dataset), index=False)
    pd.set_option("display.width", 220)
    print("\n===== [%s] consensus rules: per-case optimal Dice and other metrics (n cases = %d) =====" % (a.dataset, int(S.n.max())))
    print(S[S.kind == "consensus"].drop(columns=["kind"]).round(4).to_string(index=False))
    if IND:
        print("\n----- top-5 individuals (same metrics) -----")
        print(S[S.kind == "individual"].head(5).drop(columns=["kind"]).round(4).to_string(index=False))
    BS = bd.groupby(["vote_topk", "rule"]).agg(dice=("dice", "mean"), iou=("iou", "mean"), ppv=("ppv", "mean"),
                                               sens=("sens", "mean"), area=("area_frac", "mean")).reset_index()
    BS.to_csv(os.path.join(a.out_dir, "%s_binary_rules.csv" % a.dataset), index=False)
    print("\n===== fixed binary rules (no threshold sweep): mean Dice / IoU / PPV / Sens / selected FOV fraction =====")
    print(BS.round(4).to_string(index=False))
    print("\nsaved ->", a.out_dir)


if __name__ == "__main__":
    main()
