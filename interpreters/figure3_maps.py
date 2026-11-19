"""
figure3_maps.py - Figure 3 and Supplementary Figure S2: representative individual and consensus saliency maps.

Rows = dataset (e-Ophtha / IDRiD / FGADR), columns = [fundus + lesion contours | individual x3 | consensus].
  * Case selection: in each dataset, the image whose "per-image mean consensus Dice" is closest to the dataset median
    (avoids cherry-picking; --min_types sets the minimum number of lesion types).
  * 3 individual maps: if --sim_csv (model_similarity_summary.csv from ext_consensus_similarity_loo.py, scope=ALL) is given, the worst / median / best
    models by Full rho within the pool are selected automatically; otherwise they are given via --models.
  * All saliency panels show the percentile rank within the FOV (same normalization as the consensus formula), same colormap,
    values below the --alpha_cut quantile are transparent. GT is drawn as a 1px contour (MA blue / HE yellow / EX cyan / SE red).
  * Each panel corner shows the per-case optimal Dice of that image (mean over lesion types).

Inputs:
  --thr_dir      threshold_optimal_k output (per-case CSV)  -> case selection + Dice labels
  --sal_attn     {ds}_saliency_attn  parent of these roots (outputs)   individual maps, 1.0x
  --sal_ms       {ds}_saliency_ms8   parent of these roots (outputs)   consensus inputs
  --roots        3 dataset roots (order: eophtha idrid fgadr), same as threshold_optimal_k
Usage:
  python interpreters/figure3_maps.py --thr_dir outputs/threshold_analysis_002 \
      --roots $EOPHTHA_ROOT $IDRID_SEG_ROOT <FGADR_ROOT> \
      --model_roots models/dr_clean models/ckpts_aux --sim_csv outputs/model_similarity_summary.csv \
      --out_dir outputs/paper_figures
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snu_common as C
import snu_consensus_lesion as SC
import snu_localization as L
import eophtha_localization as E
from threshold_sweep_erx import DISCOVER, DEF_TYPES, load_set
from consensus_newaxis_eval import hybrid_mean_q75, mm_rank_stack
from threshold_table3_ppt import DS_ORDER, DS_LABEL, classify

CONS_KEY = "HybridMeanQ75"
LT_COLOR = {"MA": "#1f77ff", "HE": "#ffd400", "EX": "#00e5ff", "SE": "#ff2a2a"}
IMG = 448


def rank_in_fov(m, fov):
    r = np.zeros(m.shape, dtype=np.float64); v = m[fov]
    if v.size: r[fov] = rankdata(v, method="average") / v.size
    return r


def pick_case(D, ds, CONS, min_types):
    s = D[(D.dataset == ds) & (D.method == CONS)]
    per_img = s.groupby("image_id").agg(dice=("optimal_dice", "mean"), ntypes=("lesion_type", "nunique"))
    cand = per_img[per_img.ntypes >= min_types] if (per_img.ntypes >= min_types).any() else per_img
    med = float(per_img.dice.median())
    fn = (cand.dice - med).abs().idxmin()
    return fn, float(cand.loc[fn, "dice"]), med, int(cand.loc[fn, "ntypes"])


def pick_models(a, inames, itag):
    if a.models:
        sel = []
        for s in a.models:
            hit = [m for m in inames if s in m]
            if not hit: sys.exit("[abort] --models %s not found. Available: %s" % (s, inames))
            sel.append(hit[0])
        return sel, ["individual"] * len(sel)
    S = pd.read_csv(a.sim_csv)
    S = S[(S.scope == "ALL") & (S.in_pool == 1)].sort_values("rho_full_mean", ascending=False)
    # tag (e.g. 'DINOv2-LoRA') -> raw name match by pretty tag
    tag2raw = {}
    for m in inames:
        t = "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)); tag2raw[t] = m
    ranked = [tag2raw[t] for t in S.model if t in tag2raw]
    if len(ranked) < 3: sys.exit("[abort] could not match sim_csv model names to saliency model names: %s" % list(S.model)[:5])
    best, mid, worst = ranked[0], ranked[len(ranked) // 2], ranked[-1]
    return [worst, mid, best], ["least similar", "median", "most similar"]


try:   # dataset label: official spelling e-Ophtha
    DS_LABEL = dict(DS_LABEL); DS_LABEL["eophtha"] = "e-Ophtha"
except NameError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr_dir", default="", help="old format: threshold_optimal_k per-case CSVs (HybridMeanQ75)")
    ap.add_argument("--per_case", nargs="*", default=[], help="new format: consensus_rules_v2 *_rules_per_case.csv globs (individuals + consensus rule)")
    ap.add_argument("--cons_method", default="GeoMean drop4", help="consensus method name in --per_case CSVs")
    ap.add_argument("--cons_rule", choices=["hybrid", "geo_trim", "arith_trim"], default="hybrid", help="how the consensus panel is built")
    ap.add_argument("--cons_trim", type=int, default=4, help="geo_trim: number of lowest values dropped per pixel")
    ap.add_argument("--cons_sal", default="ms8", help="consensus material: {ds}_saliency_<this> (ms8 = old multi-scale, attn = 1.0x)")
    ap.add_argument("--roots", nargs=3, required=True, help="eophtha idrid fgadr roots (in this order)")
    ap.add_argument("--outputs", default="outputs", help="parent of {ds}_saliency_attn / {ds}_saliency_ms8")
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--auc_min", type=float, default=0.92)
    ap.add_argument("--sim_csv", default="", help="model_similarity_summary.csv (auto-selects 3 models by Full rho)")
    ap.add_argument("--models", nargs="*", default=[], help="explicitly specify 3 individual models (substring match)")
    ap.add_argument("--min_types", type=int, default=2)
    ap.add_argument("--cases", nargs="*", default=[], help="explicit image_id per dataset in order (eophtha idrid fgadr), with or without extension. If empty, use the median rule")
    ap.add_argument("--show_top", type=float, default=0.15, help="show only this top fraction (rank) of the FOV. 0.15 = top 15%%")
    ap.add_argument("--cmap", default="jet", help="saliency colormap (jet: blue=low, red=high)")
    ap.add_argument("--alpha_max", type=float, default=0.75)
    ap.add_argument("--style", choices=["lab", "topk", "slide"], default="slide",
                    help="slide: same as qual_page_all.overlay (gray fundus, jet(min-max), alpha=0.65*value, blur 0.5) | lab: uniform alpha 0.45 blend over the original | topk: show rank only for the top show_top")
    ap.add_argument("--blur", type=float, default=0.5, help="gaussian blur for slide style (qual_page_all default 0.5)")
    ap.add_argument("--color_gamma", type=float, default=1.0, help="slide style: colour = cmap(v ** this); <1 pushes blobs toward red")
    ap.add_argument("--alpha_floor", type=float, default=0.0, help="slide style: opacity = amax * clip((v - floor) / (1 - floor)); values below floor are not drawn")
    ap.add_argument("--alpha_cut", type=float, default=0.0, help="slide style: values below this are transparent; opacity rises to amax over the next 0.15")
    ap.add_argument("--alpha_gamma", type=float, default=1.0, help="slide style: alpha = amax * value**alpha_gamma (1 = linear)")
    ap.add_argument("--amax", type=float, default=0.65, help="maximum alpha for slide style (qual_page_all default 0.65)")
    ap.add_argument("--blend_alpha", type=float, default=0.45, help="heatmap weight for lab style (plain alpha blend)")
    ap.add_argument("--base", choices=["gray", "color", "none"], default="gray", help="saliency panel background: gray (grayscale fundus) | color (color fundus) | none (black)")
    ap.add_argument("--contour_topk", choices=["none", "optimal", "fixed"], default="optimal",
                    help="white contour of the top-k region on each panel: optimal = that map's per-case optimal k (mean over lesion types), fixed = --fixed_k %%")
    ap.add_argument("--fixed_k", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=2.0, help="color compression for topk style: color = cmap(Rs**gamma). Larger values confine red to the very top")
    ap.add_argument("--gt_overlay", choices=["first", "all"], default="first", help="draw lesion contours on the first column only (first) / on all panels (all)")
    ap.add_argument("--layout", choices=["landscape", "portrait"], default="portrait", help="landscape: rows=dataset | portrait: columns=dataset, rows=panel (vertical, for the paper)")
    ap.add_argument("--crop_fov", type=int, default=1, help="1: crop panels to the fundus (FOV) bounding box (removes letterbox margins)")
    ap.add_argument("--crop_pad", type=int, default=4)
    ap.add_argument("--consensus_row", type=int, default=2, choices=[2, 5], help="position of the consensus row in portrait: 2 (second row) or 5 (last)")
    ap.add_argument("--grid21", action="store_true", help="Fig S1: separately save a per-dataset grid of [fundus, consensus] + 21 configurations (7 backbone x 3 FT)")
    ap.add_argument("--out_dir", default="outputs/paper_figures")
    a = ap.parse_args()
    if not a.models and not a.sim_csv: sys.exit("[abort] either --sim_csv or --models is required")
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from skimage import measure

    if a.per_case:
        fs = [f for pat in a.per_case for f in sorted(glob.glob(pat))]
        D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        D = D.drop_duplicates(["dataset", "image_id", "lesion_type", "method"], keep="last")
        D = D.rename(columns={"max_dice": "optimal_dice", "best_k": "optimal_k"})
        CONS = a.cons_method
        if CONS not in set(D.method): sys.exit("[abort] --cons_method %s not found in the per-case CSV" % CONS)
    else:
        fs = sorted(glob.glob(os.path.join(a.thr_dir, "*_optimal_k_per_case.csv")))
        D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        CONS = [m for m in set(D.method) if CONS_KEY in m][0]
    auc = SC.load_auc_map(a.model_roots)
    roots = dict(zip(["eophtha", "idrid", "fgadr"], a.roots))
    dss = [d for d in DS_ORDER if d in set(D.dataset)]

    cmap = plt.get_cmap(a.cmap)
    panels = []   # (ds, fn, [(title, rgba_or_None, dice_text)], contours)
    sel_models, sel_roles = None, None
    for ds in dss:
        fn, dice_fn, med, nt = pick_case(D, ds, CONS, a.min_types if ds != "eophtha" else 1)
        if a.cases:
            want = a.cases[dss.index(ds)]
            ids = sorted(set(D[D.dataset == ds].image_id))
            hit = [x for x in ids if x == want or os.path.splitext(x)[0] == os.path.splitext(want)[0]]
            if not hit: sys.exit("[abort] %s: --cases %s not found in the per-case CSV (e.g. %s)" % (ds, want, ids[:3]))
            fn = hit[0]
            sub = D[(D.dataset == ds) & (D.method == CONS) & (D.image_id == fn)]
            dice_fn, nt = float(sub.optimal_dice.mean()), int(sub.lesion_type.nunique())
        items = [r for r in DISCOVER[ds](roots[ds], DEF_TYPES[ds]) if r["image_id"] == fn]
        if not items: sys.exit("[abort] %s: %s not found in the dataset root" % (ds, fn))
        cn, ccams, cfidx = load_set(os.path.join(a.outputs, "%s_saliency_%s" % (ds, a.cons_sal)))
        cn = [m for m in cn if C.lineage_of(m) and L.ft_of(m)]
        POOL = [m for m in sorted(cn, key=lambda m: -auc.get(m, 0)) if auc.get(m, 0) >= a.auc_min]
        inames, icams, ifidx = load_set(os.path.join(a.outputs, "%s_saliency_attn" % ds))
        inames = [m for m in inames if C.lineage_of(m) and L.ft_of(m)]
        itag = {m: "%s-%s" % (L.backbone_pretty(m), L.ft_of(m)) for m in inames}
        if sel_models is None:
            sel_models, sel_roles = pick_models(a, inames, itag)
            print("3 individual maps:", [(itag[m], r) for m, r in zip(sel_models, sel_roles)], "| pool n=%d" % len(POOL))
        img = E.letterbox_img(items[0]["image_path"])
        fov = C.FOV & E.fundus_mask_448(img)
        gray = img.mean(2)
        _MM, Rk = mm_rank_stack(POOL, fn, ccams, cfidx, fov)
        if a.cons_rule == "geo_trim":   # trimmed geometric mean of unit-sum maps (paper equation 1)
            fb = fov.astype(bool)
            U = np.stack([(m * fov) / max((m * fov).sum(), 1e-12) for m in _MM], 0)
            G = np.exp(np.log(np.maximum(np.sort(U, axis=0)[a.cons_trim:], 1e-12)).mean(0)) * fov
            cons_map = G / max(G[fb].sum(), 1e-12)
        elif a.cons_rule == "arith_trim":   # symmetric trimmed mean of unit-sum maps (paper equation 1)
            fb = fov.astype(bool)
            U = np.stack([(m * fov) / max((m * fov).sum(), 1e-12) for m in _MM], 0)
            S = np.sort(U, axis=0)
            G = S[a.cons_trim:S.shape[0] - a.cons_trim].mean(0) * fov
            cons_map = G / max(G[fb].sum(), 1e-12)
        else:
            cons_map = hybrid_mean_q75(Rk, fov)
        maps = [("Consensus", rank_in_fov(cons_map, fov), E.mm(cons_map) * fov)]
        for m in sel_models:
            vm = E.mm(icams[m][ifidx[m][fn]].numpy()) * fov
            maps.insert(-1, (itag[m], rank_in_fov(vm, fov), vm))
        if a.grid21:
            for m in inames:
                if m in sel_models: continue
                vm = E.mm(icams[m][ifidx[m][fn]].numpy()) * fov
                maps.insert(-1, (itag[m], rank_in_fov(vm, fov), vm))
        # Dice per method for this image (mean over its lesion cases)
        dice_of = lambda meth: D[(D.dataset == ds) & (D.image_id == fn) & (D.method == meth)].optimal_dice.mean()
        k_of = lambda meth: D[(D.dataset == ds) & (D.image_id == fn) & (D.method == meth)].optimal_k.mean()
        contours = []
        for r in items:
            Lm = E.letterbox_mask(r["mask_path"]) & fov
            for c in measure.find_contours(Lm.astype(float), 0.5):
                contours.append((r["lesion_type"], c))
        panels.append((ds, fn, gray, fov, maps, contours, itag, dice_of, dice_fn, med, nt, img, k_of))
        print("%-8s case=%s  consensus Dice(img mean)=%.3f  dataset median=%.3f  lesion types=%d" % (ds, fn, dice_fn, med, nt))

    # ================= plotting: GridSpec, 15 identical square panels =================
    from scipy.ndimage import gaussian_filter
    import matplotlib as mpl
    ind_tags = [itag[m] for m in sel_models]                       # given order: least, median, selected
    if a.consensus_row == 2:
        row_specs = [("fundus", "Expert annotations"), ("Consensus", "Consensus")] + \
                    [(t, "%s\n(%s)" % (t, r)) for t, r in zip(ind_tags[::-1], sel_roles[::-1])]
    else:
        row_specs = [("fundus", "Expert annotations")] + \
                    [(t, "%s\n(%s)" % (t, r)) for t, r in zip(ind_tags, sel_roles)] + [("Consensus", "Consensus")]
    nP, nD = len(row_specs), len(panels)
    LBL_FS = 11   # row labels and dataset titles: same size, bold
    portrait = a.layout == "portrait"
    if portrait:
        fig = plt.figure(figsize=(2.4 * nD + 1.8, 2.4 * nP + 1.2))
        gs = fig.add_gridspec(nP, nD, left=0.20, right=0.99, top=0.95, bottom=0.075, wspace=0.04, hspace=0.05)
    else:
        fig = plt.figure(figsize=(2.4 * nP + 1.0, 2.4 * nD + 1.0))
        gs = fig.add_gridspec(nD, nP, left=0.02, right=0.99, top=0.93, bottom=0.10, wspace=0.04, hspace=0.05)

    def square_box(fov):
        ys, xs = np.where(fov)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        side = max(y1 - y0, x1 - x0) + 2 * a.crop_pad
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        return cx - side / 2.0, cx + side / 2.0, cy + side / 2.0, cy - side / 2.0   # xlim0, xlim1, ylim0(bottom), ylim1(top)

    def render(ax, kind, ds, fn, gray, fov, maps, contours, dice_of, img):
        ax.set_box_aspect(1); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_facecolor("black")
        if kind == "fundus":
            ax.imshow(np.clip(img, 0, 1))
            for lt, c in contours: ax.plot(c[:, 1], c[:, 0], color=LT_COLOR.get(lt, "w"), lw=0.9)
        else:
            name, R, V = [m for m in maps if m[0] == kind][0]
            if a.base == "gray": ax.imshow(gray * 0.85, cmap="gray", vmin=0, vmax=1)
            elif a.base == "color": ax.imshow(np.clip(img, 0, 1))
            else: ax.imshow(np.zeros_like(gray), cmap="gray", vmin=0, vmax=1)
            if a.style == "slide":
                v = np.clip(gaussian_filter(V, a.blur) if a.blur > 0 else V, 0, 1)
                base = np.repeat(gray[..., None], 3, axis=2); jet = cmap(v ** a.color_gamma)[..., :3]; al = (a.amax * (np.clip((v - a.alpha_cut) / 0.15, 0, 1) if a.alpha_cut > 0 else np.clip((v - a.alpha_floor) / (1 - a.alpha_floor), 0, 1) ** a.alpha_gamma))[..., None]
                ax.images[-1].remove(); ax.imshow(np.clip(base * (1 - al) + jet * al, 0, 1))
            elif a.style == "lab":
                heat = cmap(V)[..., :3]; base = np.clip(img, 0, 1)
                ax.images[-1].remove(); ax.imshow(np.where(fov[..., None], base * (1 - a.blend_alpha) + heat * a.blend_alpha, base))
            else:
                cut = 1.0 - a.show_top
                Rs = np.clip((R - cut) / (1 - cut), 0, 1)
                rgba = cmap(Rs ** a.gamma); rgba[..., 3] = (Rs > 0) * (0.35 + 0.65 * Rs) * a.alpha_max * fov
                ax.imshow(rgba)
            meth = name if name != "Consensus" else CONS
            if a.contour_topk != "none":
                kk = float(k_of(meth)) if a.contour_topk == "optimal" else a.fixed_k
                if np.isfinite(kk):
                    TK = (R >= 1.0 - kk / 100.0) & fov
                    for c in measure.find_contours(TK.astype(float), 0.5): ax.plot(c[:, 1], c[:, 0], color="white", lw=0.6, alpha=0.9)
            if a.gt_overlay == "all":
                for lt, c in contours: ax.plot(c[:, 1], c[:, 0], color=LT_COLOR.get(lt, "w"), lw=0.7)
            dv = dice_of(meth)
            if np.isfinite(dv):
                ax.text(0.03, 0.03, "Dice %.3f" % dv, transform=ax.transAxes, color="white", fontsize=9, ha="left", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.45, pad=1.5, lw=0))
        x0, x1, y0, y1 = square_box(fov) if a.crop_fov else (0, IMG - 1, IMG - 1, 0)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    axes_grid = {}
    for di, (ds, fn, gray, fov, maps, contours, itag_, dice_of, dice_fn, med, nt, img, k_of) in enumerate(panels):
        for pi, (kind, label) in enumerate(row_specs):
            ax = fig.add_subplot(gs[pi, di] if portrait else gs[di, pi])
            render(ax, kind, ds, fn, gray, fov, maps, contours, dice_of, img)
            axes_grid[(pi, di)] = ax
    # titles: dataset (bold) + image id (small)
    for di, (ds, fn, *_rest) in enumerate(panels):
        ax = axes_grid[(0, di)] if portrait else axes_grid[(0, di)]
        if portrait:
            ax.text(0.5, 1.12, DS_LABEL.get(ds, ds), transform=ax.transAxes, ha="center", va="bottom", fontsize=LBL_FS, fontweight="bold")
            ax.text(0.5, 1.03, os.path.splitext(fn)[0], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#444")
    # row labels centred on each row (portrait) / column titles (landscape)
    if portrait:
        for pi, (kind, label) in enumerate(row_specs):
            bb = axes_grid[(pi, 0)].get_position()
            fig.text(bb.x0 - 0.012, (bb.y0 + bb.y1) / 2, label, ha="right", va="center", fontsize=LBL_FS, fontweight="bold", linespacing=1.3)
    else:
        for pi, (kind, label) in enumerate(row_specs):
            bb = axes_grid[(pi, 0)].get_position()
            fig.text((bb.x0 + bb.x1) / 2, 0.955, label.replace("\n", " "), ha="center", va="bottom", fontsize=9)
        for di, (ds, fn, *_rest) in enumerate(panels):
            bb = axes_grid[(0, di)].get_position()
            fig.text(0.012, (bb.y0 + bb.y1) / 2, DS_LABEL.get(ds, ds), rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    # lesion legend (bottom-left) + colour bar (bottom-centre/right), both in figure coords
    h = [plt.Line2D([0], [0], color=LT_COLOR[k], lw=1.6) for k in LT_COLOR]
    fig.legend(h, ["MA", "HE", "EX", "SE"], loc="lower left", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.19, 0.012), handlelength=1.6)
    if a.style == "slide" and (a.alpha_floor > 0 or a.color_gamma != 1):   # colour bar = colours actually drawn
        from matplotlib.colors import ListedColormap
        cb_cmap = ListedColormap(cmap(np.linspace(a.alpha_floor, 1, 256) ** a.color_gamma))
    else:
        cb_cmap = cmap
    cax = fig.add_axes([0.58, 0.028, 0.32, 0.011])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cb_cmap, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["Low", "High"], fontsize=8.5); cb.outline.set_visible(False)
    cb.set_label("Saliency (top %d%% of FOV shown)" % round(a.show_top * 100) if a.style == "topk" else "Saliency", fontsize=8.5, labelpad=2)
    if os.environ.get("PANEL_DIR"):   # also save each fundus / consensus panel as its own PNG
        pdir = os.environ["PANEL_DIR"]; os.makedirs(pdir, exist_ok=True)
        fig.canvas.draw(); rr = fig.canvas.get_renderer()
        for (pi, di), ax in axes_grid.items():
            kind = row_specs[pi][0]
            if kind not in ("fundus", "Consensus"): continue
            bb = ax.get_window_extent(rr).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(pdir, "%s__%s__%s.png" % (panels[di][0], os.path.splitext(panels[di][1])[0], kind)), dpi=150, bbox_inches=bb)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig2_maps." + ext), dpi=300)
    if a.grid21:
        BB_ORDER = ["OpenCLIP", "DINOv2", "RETFound-DINOv2", "DINOv3-Large", "DINOv3-Huge", "MAE", "RETFound"]
        FT_ORDER = [("Linear", "Classifier-only"), ("Partial", "Partial fine-tuning"), ("LoRA", "LoRA")]
        for ds, fn, gray, fov, maps, contours, itag_, dice_of, dice_fn, med, nt, img, k_of in panels:
            have = {t for t, _, _ in maps}
            nR = 1 + len(BB_ORDER)
            fg = plt.figure(figsize=(2.2 * 3 + 1.9, 2.2 * nR + 1.6))
            # rows: 0 = fundus/consensus, 1 = spacer (FT titles), 2.. = backbones
            g2 = fg.add_gridspec(nR + 1, 3, left=0.21, right=0.99, top=0.95, bottom=0.07, wspace=0.04, hspace=0.05,
                                 height_ratios=[1, 0.22] + [1] * len(BB_ORDER))
            row0 = [fg.add_subplot(g2[0, 0]), fg.add_subplot(g2[0, 1])]
            render(row0[0], "fundus", ds, fn, gray, fov, maps, contours, dice_of, img)
            render(row0[1], "Consensus", ds, fn, gray, fov, maps, contours, dice_of, img)
            row0[0].text(0.5, 1.04, "Expert annotations", transform=row0[0].transAxes, ha="center", va="bottom", fontsize=LBL_FS - 1, fontweight="bold")
            row0[1].text(0.5, 1.04, "Consensus", transform=row0[1].transAxes, ha="center", va="bottom", fontsize=LBL_FS - 1, fontweight="bold")
            first_col = {}
            for ri, bb in enumerate(BB_ORDER, start=1):
                for ci, (ftk, ftl) in enumerate(FT_ORDER):
                    tag = "%s-%s" % (bb, ftk)
                    ax = fg.add_subplot(g2[ri + 1, ci])
                    if tag in have: render(ax, tag, ds, fn, gray, fov, maps, contours, dice_of, img)
                    else: ax.set_axis_off()
                    if ri == 1: ax.text(0.5, 1.03, ftl, transform=ax.transAxes, ha="center", va="bottom", fontsize=LBL_FS, fontweight="bold")
                    if ci == 0: first_col[ri] = ax
            for ri, bb in enumerate(BB_ORDER, start=1):
                b = first_col[ri].get_position()
                fg.text(0.20, (b.y0 + b.y1) / 2, bb, ha="right", va="center", fontsize=LBL_FS, fontweight="bold")
            fg.suptitle("%s  (%s)" % (DS_LABEL.get(ds, ds), os.path.splitext(fn)[0]), fontsize=LBL_FS + 1, fontweight="bold", y=0.985)
            hh = [plt.Line2D([0], [0], color=LT_COLOR[k], lw=1.6) for k in LT_COLOR]
            fg.legend(hh, ["MA", "HE", "EX", "SE"], loc="lower left", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.20, 0.012), handlelength=1.6)
            cax2 = fg.add_axes([0.62, 0.028, 0.30, 0.009])
            cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cb_cmap, orientation="horizontal", ticks=[0, 1])
            cb2.ax.set_xticklabels(["Low", "High"], fontsize=8.5); cb2.outline.set_visible(False)
            cb2.set_label("Saliency (top %d%% of FOV shown)" % round(a.show_top * 100) if a.style == "topk" else "Saliency", fontsize=8.5, labelpad=2)
            for ext in ("png", "pdf"):
                fg.savefig(os.path.join(a.out_dir, "figS1_grid21_%s.%s" % (ds, ext)), dpi=250)
            plt.close(fg)
            print("Saved: figS1_grid21_%s.png (%d/21 configurations found)" % (ds, sum(1 for t in have if t != "Consensus")))
    print("Saved:", os.path.join(a.out_dir, "fig2_maps.png / .pdf"))


if __name__ == "__main__":
    main()
