"""
ext_consensus_similarity_loo.py - CoSA: agreement of each configuration's saliency map with the consensus map
on external images without lesion annotations (Table 3, Supplementary Table S6).

No lesion annotation is used here.
  consensus = --rule (paper: arith_trim2 = L1-normalised maps, symmetric trimmed mean discarding the 2 highest
              and the 2 lowest values at each pixel; equation 1)
  pool      = configurations with internal AUROC >= --auc_min (paper: 0.92), adaptation strategies given by --pool_ft
  primary   = image-wise Spearman rank correlation within the FOV
  secondary = top-k spatial IoU (k = --topk_list, paper: 1, 2, 5, 10, 20%) and the normalised AUC of that curve

Three comparison modes:
  0) Full        agreement with the consensus of the whole pool
  A) Self-LOO    the target itself is excluded from the pool (targets outside the pool equal Full)
  B) Lineage-LOO the target's whole pretraining lineage is excluded from the pool
  self_influence = rho_full - rho_self

Lineages are defined in consensus_overlap.lineage_of.

Resolution --sp_size:
  56   quick check (minutes)
  448  full resolution used in the paper (large memory; hours)

Usage (paper setting):
  python interpreters/ext_consensus_similarity_loo.py --rule arith_trim2 --targets pool \
    --datasets aptos deepdrid messidor2 --auc_min 0.92 --sp_size 448 \
    --sal_root $OUTPUT_ROOT/saliency_attn --cons_sal_root $OUTPUT_ROOT/saliency_attn \
    --model_roots $SAVE_ROOT/dr_clean $SAVE_ROOT/ckpts_aux \
    --out_dir $OUTPUT_ROOT/ext_cons_sim_1x14_arith_trim2
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
import torch
import torch.nn.functional as TF
from scipy.stats import rankdata

EPS = 1e-12


def ft_of(name):
    n = name.lower()
    return ("LoRA" if "lora" in n else "Partial" if "partial" in n
            else "Linear" if "linear" in n else None)


def bpretty(name):
    n = name.lower()
    if "clip" in n or "laion" in n: return "OpenCLIP"
    if "retfound_dinov2" in n: return "RETFound-DINOv2"
    if "retfound_mae" in n or (n.startswith("retfound") and "dinov2" not in n): return "RETFound"
    if "dinov3" in n:
        return "DINOv3-Huge" if ("huge" in n or "vith" in n) else "DINOv3-Large"
    if "dinov2" in n: return "DINOv2"
    if "mae" in n: return "MAE"
    return name


# ── consensus formula (same as RFI) ─────────────────────────────
def mm(cam, fov):
    v = np.maximum(cam, 0.0) * fov
    vals = v[fov]
    mn, mx = float(vals.min()), float(vals.max())
    return (v - mn) / (mx - mn + 1e-8) * fov if mx > mn else v * 0.0


def rank_in_fov(m, fov):
    r = np.zeros(m.shape, dtype=np.float64)
    v = m[fov]
    if v.size:
        r[fov] = rankdata(v, method="average") / v.size
    return r


def norm1(a, fov):
    x = a * fov
    s = x.sum()
    return x / s if s > 1e-12 else x


def hybrid_mean_q75(Rk, fov):
    """0.5*rank(RankMean) + 0.5*rank(Q75). Same as RFI consensus_newaxis_eval."""
    K = Rk.shape[0]
    srt = np.sort(Rk, axis=0)
    mean = Rk.mean(axis=0)
    pos = 0.75 * (K - 1)
    lo = int(np.floor(pos)); hi = min(lo + 1, K - 1); fr = pos - lo
    q75 = srt[lo] * (1.0 - fr) + srt[hi] * fr
    return norm1(0.5 * rank_in_fov(mean, fov) + 0.5 * rank_in_fov(q75, fov), fov)


RULES = ("hybrid", "geo", "geo_drop1", "geo_drop2", "geo_drop3", "geo_drop4", "arith_trim1", "arith_trim2")


def consensus_rule(rule, Mk, fov):
    """Mk: stack of min-max maps (K x H x W) of the pool members.
    hybrid      = HybridMeanQ75 on percentile ranks (the original rule)
    geo         = geometric mean of unit-sum maps (ViTMix-style multiplicative fusion)
    geo_dropT   = geometric mean after dropping the T lowest values at each pixel (trimmed)
    arith_trimT = arithmetic mean of unit-sum maps after dropping the T highest and the T lowest values at each pixel
                  (symmetric trimmed mean; same rule as UnitSum-Arith-T{T}x{T} in eval_consensus_localization.py)"""
    if rule == "hybrid":
        return hybrid_mean_q75(np.stack([rank_in_fov(m, fov) for m in Mk]), fov)
    U = np.stack([norm1(m, fov) for m in Mk]); srt = np.sort(U, axis=0)
    if rule.startswith("arith_trim"):
        t = int(rule[len("arith_trim"):]); K = srt.shape[0]
        if K - 2 * t < 1:
            raise ValueError("pool of %d maps is too small for %s" % (K, rule))
        return norm1(srt[t:K - t].mean(0) * fov, fov)
    t = {"geo": 0, "geo_drop1": 1, "geo_drop2": 2, "geo_drop3": 3, "geo_drop4": 4}[rule]
    return norm1(np.exp(np.log(np.maximum(srt[t:], 1e-12)).mean(0)) * fov, fov)


def hybrid_reference(Rk, fov):
    """Independent implementation of the 5-step spec, for sanity cross-check."""
    mean = Rk.mean(axis=0)
    q75 = np.quantile(Rk, 0.75, axis=0, method="linear")
    return norm1(0.5 * rank_in_fov(mean, fov) + 0.5 * rank_in_fov(q75, fov), fov)


# ── metric ───────────────────────────────────────────────
def spearman_fov(a, b, fov):
    x, y = np.asarray(a)[fov], np.asarray(b)[fov]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    rx = rankdata(x).astype(np.float64); ry = rankdata(y).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def topk_mask(A, fov, k):
    fi = np.flatnonzero(fov.ravel())
    v = np.asarray(A).ravel()[fi]
    N = int(np.clip(round(k / 100.0 * v.size), 1, v.size))
    idx = fi[np.argpartition(-v, N - 1)[:N]]
    m = np.zeros(A.size, bool); m[idx] = True
    return m.reshape(A.shape)


def topk_ious(A, B, fov, ks):
    out = []
    for k in ks:
        ma, mb = topk_mask(A, fov, k), topk_mask(B, fov, k)
        u = np.logical_or(ma, mb).sum()
        out.append(float(np.logical_and(ma, mb).sum()) / u if u else np.nan)
    return out


def norm_auc(ious, ks):
    """normalized AUC of the overlap curve = trapezoid(IoU, k) / (k_max - k_min).  [0,1]
    np.trapz was removed in numpy 2.x, so the trapezoidal integral is computed directly."""
    v = np.asarray(ious, float)
    kk = np.asarray(ks, float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.nan
    v, kk = v[ok], kk[ok]
    o = np.argsort(kk)
    v, kk = v[o], kk[o]
    area = float(np.sum(np.diff(kk) * (v[:-1] + v[1:]) / 2.0))
    span = float(kk.max() - kk.min())
    return area / span if span > 0 else np.nan


def boot_ci(v, n=2000, seed=42):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


class Tee(object):
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sal_root", required=True, help="individual-model target maps (baseline = 1.0x)")
    ap.add_argument("--cons_sal_root", default="", help="consensus inputs. Empty = --sal_root")
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--auc_min", type=float, default=0.92)
    ap.add_argument("--pool_ft", nargs="*", default=["Partial", "LoRA"],
                    help="FT types for the consensus pool. Default Partial LoRA (= RFI 14-model pool)")
    ap.add_argument("--sp_size", type=int, default=56,
                    help="evaluation resolution. 56=quick check | 448=full resolution (same space as RFI, heavy)")
    ap.add_argument("--targets", choices=["pool", "all"], default="all")
    ap.add_argument("--topk_list", nargs="+", type=float, default=[1, 2, 5, 10, 20])
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="use only these datasets; empty = all. Datasets with GT (idrid/fgadr) "
                         "are already covered by the GT-based evaluation and should be excluded here: "
                         "--datasets aptos messidor2 deepdrid")
    ap.add_argument("--exclude_datasets", nargs="*", default=[],
                    help="datasets to exclude; can be combined with --datasets")
    ap.add_argument("--max_img", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--sanity_images", type=int, default=5)
    ap.add_argument("--min_pool", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--rule", choices=RULES, default="hybrid", help="consensus aggregation rule (default = original HybridMeanQ75)")
    ap.add_argument("--modes", nargs="+", default=["full", "self", "lineage"],
                    choices=["full", "self", "lineage"],
                    help="comparison modes to compute. For main-text Table 4 only, use --modes full (~3x faster)")
    a = ap.parse_args()
    MODES = list(a.modes)
    if "full" not in MODES: MODES = ["full"] + MODES
    os.makedirs(a.out_dir, exist_ok=True)
    if not a.dry_run:
        sys.stdout = Tee(os.path.join(a.out_dir, "analysis.log"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import consensus_overlap as co

    SP = int(a.sp_size)
    FS = co.FOV if SP == co.IMG_SIZE else co.make_fov(SP)
    CROOT = a.cons_sal_root or a.sal_root
    same_src = (os.path.abspath(CROOT) == os.path.abspath(a.sal_root))

    def resize_to(cam_t, chunk=128):
        if SP == co.IMG_SIZE:
            return cam_t.numpy()
        out = []
        for s in range(0, len(cam_t), chunk):
            b = cam_t[s:s + chunk].float().unsqueeze(1)
            out.append(TF.interpolate(b, (SP, SP), mode="bilinear",
                                      align_corners=False).squeeze(1))
        return torch.cat(out).numpy()

    def scan(root):
        nm = sorted([os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
                     if os.path.exists(os.path.join(d, "saliency_map.pth"))])
        return {m: pd.read_csv(os.path.join(root, m, "used.csv"))["unique_name"]
                .astype(str).tolist() for m in nm}

    tkeys = scan(a.sal_root)
    ckeys = tkeys if same_src else scan(CROOT)
    meta = {m: dict(lin=co.lineage_of(m), ft=ft_of(m),
                    auc=co.read_auc(co._clean(m), a.model_roots))
            for m in sorted(set(list(tkeys) + list(ckeys)))}
    ok = [m for m in meta if meta[m]["lin"] and meta[m]["ft"]]
    want = set(s.lower() for s in a.pool_ft)
    POOL = sorted([m for m in ok if m in ckeys and str(meta[m]["ft"]).lower() in want
                   and np.isfinite(meta[m]["auc"]) and meta[m]["auc"] >= a.auc_min],
                  key=lambda m: -meta[m]["auc"])
    TARGETS = sorted([m for m in ok if m in tkeys], key=lambda m: -(meta[m]["auc"]
                                                                   if np.isfinite(meta[m]["auc"])
                                                                   else -1))
    if a.targets == "pool":
        TARGETS = [m for m in TARGETS if m in POOL]
    if len(POOL) < 3:
        sys.exit("[abort] pool has %d models." % len(POOL))

    tag = {m: "%s-%s" % (bpretty(m), meta[m]["ft"]) for m in set(POOL) | set(TARGETS)}
    lin = {m: meta[m]["lin"] for m in tag}
    sib = {m: sum(1 for x in POOL if lin[x] == lin[m] and x != m) for m in tag}
    inp = {m: (m in POOL) for m in tag}

    print("=" * 100)
    print("external consensus similarity  |  resolution %d  |  FOV pixels %d" % (SP, int(FS.sum())))
    print("  target root    : %s   (individual-model baseline)" % a.sal_root)
    print("  consensus root : %s%s" % (CROOT, "   (= target root)" if same_src else ""))
    if same_src:
        print("  consensus inputs are the same maps as the individual targets (1.0x)")
    print("  pool: AUROC >= %.2f, FT %s  ->  n = %d | evaluated targets %d"
          % (a.auc_min, a.pool_ft, len(POOL), len(TARGETS)))
    print("=" * 100)
    print("  %-3s %-32s %-10s %-8s %-12s %-4s" % ("#", "model", "lineage", "AUROC",
                                                  "in_pool", "sib"))
    for i, m in enumerate(TARGETS, 1):
        print("  %-3d %-32s %-10s %-8.4f %-12s %-4d"
              % (i, tag[m], lin[m], meta[m]["auc"],
                 "eligible" if inp[m] else "target only", sib[m]))
    byl = {}
    for m in POOL:
        byl.setdefault(lin[m], []).append(m)
    print("\n  pool size per lineage: %s"
          % ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(byl.items())))

    self_pool, lin_pool, warn = {}, {}, []
    print("\nLOO pool composition (the target itself is always excluded in both modes)")
    for m in TARGETS:
        sp_ = [x for x in POOL if x != m]
        lp = [x for x in POOL if lin[x] != lin[m]]
        self_pool[m], lin_pool[m] = sp_, lp
        if len(lp) < a.min_pool:
            warn.append((tag[m], len(lp)))
        print("  %-32s Full n=%-3d | Self n=%-3d | Lineage n=%-3d  (excluded: %s)"
              % (tag[m], len(POOL), len(sp_), len(lp),
                 ", ".join(tag[x] for x in POOL if lin[x] == lin[m]) or "-"))
        assert m not in sp_ and m not in lp
    if warn:
        print("  [warning] Lineage-LOO pool smaller than %d: %s" % (a.min_pool, warn))

    # ── common images ──
    common = set(tkeys[TARGETS[0]])
    for m in TARGETS:
        common &= set(tkeys[m])
    for m in POOL:
        common &= set(ckeys[m])
    # unique_name -> dataset. Uses the dataset column of used.csv (same as the existing ext scripts).
    ds_of = {}
    for m in TARGETS:
        u = pd.read_csv(os.path.join(a.sal_root, m, "used.csv"))
        if "dataset" in u.columns:
            ds_of.update(dict(zip(u["unique_name"].astype(str), u["dataset"].astype(str))))
    miss_ds = [k for k in common if k not in ds_of]
    if miss_ds:
        print("  [warn] used.csv has no dataset column; dataset inferred from name prefix for %d images"
              % len(miss_ds))
        for k in miss_ds:
            ds_of[k] = k.split("__")[0] if "__" in k else k.split("_")[0]
    avail = pd.Series([ds_of[k] for k in common]).value_counts()
    print("")
    print("common images (before filtering) %d | dataset values: %s"
          % (len(common), ", ".join("%s=%d" % (k, v) for k, v in avail.items()) or "(none)"))
    if a.datasets:
        keep = set(x.lower() for x in a.datasets)
        hit = [k for k in avail.index if k.lower() in keep]
        if not hit:
            print("  [abort] no values match --datasets %s." % a.datasets)
            print("     actual values: %s" % list(avail.index))
            sys.exit(1)
        common = {k for k in common if ds_of[k].lower() in keep}
    if a.exclude_datasets:
        drop = set(x.lower() for x in a.exclude_datasets)
        common = {k for k in common if ds_of[k].lower() not in drop}
    keys = sorted(common)
    if a.max_img and len(keys) > a.max_img:
        rng = np.random.default_rng(a.seed)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), a.max_img, replace=False))]
    cnt = pd.Series([ds_of[k] for k in keys]).value_counts()
    print("\ncommon images %d | datasets: %s"
          % (len(keys), ", ".join("%s=%d" % (k, v) for k, v in cnt.items())))
    gb = (len(set(POOL) | set(TARGETS)) * len(keys) * SP * SP * 4) / 1e9
    print("  estimated resident memory %.1f GB" % gb)
    if not keys:
        sys.exit("[abort] 0 common images")
    if a.dry_run:
        print("\n[--dry_run] stopping here. Maps were not loaded.")
        return

    # ── load maps ──
    def load(root, names, keyset):
        out = {}
        for m in names:
            kk = pd.read_csv(os.path.join(root, m, "used.csv"))["unique_name"].astype(str).tolist()
            pos = {k: i for i, k in enumerate(kk)}
            cam = torch.load(os.path.join(root, m, "saliency_map.pth"), map_location="cpu")
            cam = cam if torch.is_tensor(cam) else cam["saliency"]
            sel = [pos[k] for k in keyset]
            out[m] = resize_to(cam[sel])
            del cam
        return out

    print("\nloading maps...")
    TM = load(a.sal_root, TARGETS, keys)
    CM = TM if (same_src and set(POOL) <= set(TARGETS)) else load(CROOT, POOL, keys)
    print("  targets %d / consensus inputs %d" % (len(TM), len(CM)))

    # ── sanity ──
    print("\nconsensus rule: %s" % a.rule)
    if a.rule == "hybrid":
        print("\n[sanity] HybridMeanQ75: existing formula vs independent implementation")
        rr = []
        for i in range(min(a.sanity_images, len(keys))):
            Rk = np.stack([rank_in_fov(mm(CM[m][i], FS), FS) for m in POOL])
            r = spearman_fov(hybrid_mean_q75(Rk, FS), hybrid_reference(Rk, FS), FS)
            rr.append(r)
            print("  %-34s Spearman = %.9f" % (keys[i][:34], r))
        if not all(np.isfinite(x) and x > 0.999 for x in rr):
            sys.exit("[abort] sanity check failed")
        print("  -> passed")

    KS = list(a.topk_list)
    uniq_lin = sorted(set(lin[m] for m in TARGETS))
    key_self = {m: tuple(self_pool[m]) for m in TARGETS}
    uniq_self = sorted(set(key_self.values()))
    print("\ncomputing similarity: %d images x %d targets | top-k %s%% | consensus maps per image %d"
          % (len(keys), len(TARGETS), KS, 1 + len(uniq_self) + len(uniq_lin)))
    idx = {m: j for j, m in enumerate(POOL)}
    rows, n_nan = [], 0
    for i, k in enumerate(keys):
        Mk = np.stack([mm(CM[m][i], FS) for m in POOL])
        cons_full = consensus_rule(a.rule, Mk, FS)
        cons_self = ({kk: consensus_rule(a.rule, Mk[[idx[x] for x in kk]], FS) for kk in uniq_self}
                     if "self" in MODES else {})
        cons_lin = ({g: consensus_rule(a.rule, Mk[[idx[x] for x in POOL if lin[x] != g]], FS) for g in uniq_lin}
                    if "lineage" in MODES else {})
        for m in TARGETS:
            T = rank_in_fov(mm(TM[m][i], FS), FS)
            rec = dict(image_id=k, dataset=ds_of[k], model=tag[m], model_raw=m,
                       lineage=lin[m], siblings=sib[m], in_pool=int(inp[m]),
                       self_pool_n=len(self_pool[m]), lineage_pool_n=len(lin_pool[m]))
            pairs = [("full", cons_full)]
            if "self" in MODES: pairs.append(("self", cons_self[key_self[m]]))
            if "lineage" in MODES: pairs.append(("lineage", cons_lin[lin[m]]))
            for lab, Cx in pairs:
                r = spearman_fov(T, Cx, FS)
                if not np.isfinite(r):
                    n_nan += 1
                rec["rho_%s" % lab] = r
                ious = topk_ious(T, Cx, FS, KS)
                for kk, v in zip(KS, ious):
                    rec["iou%g_%s" % (kk, lab)] = v
                rec["aucov_%s" % lab] = norm_auc(ious, KS)
            rec["self_influence"] = rec["rho_full"] - rec["rho_self"] if "self" in MODES else np.nan
            rec["delta_rho"] = (rec["rho_self"] - rec["rho_lineage"]) if ("self" in MODES and "lineage" in MODES) else np.nan
            rows.append(rec)
        if (i + 1) % 200 == 0:
            print("  ... %d/%d" % (i + 1, len(keys)))
    P = pd.DataFrame(rows)
    P.round(6).to_csv(os.path.join(a.out_dir, "per_image_similarity.csv"), index=False)
    print("  per-image rows %d | Spearman NaN %d" % (len(P), n_nan))

    def summarize(sub, label):
        out = []
        for m in TARGETS:
            s = sub[sub["model"] == tag[m]]
            if s.empty:
                continue
            rec = dict(scope=label, model=tag[m], lineage=lin[m], in_pool=int(inp[m]),
                       siblings=sib[m], N=int(s["rho_full"].notna().sum()))
            for lab in MODES:
                v = s["rho_%s" % lab].dropna().to_numpy()
                lo, hi = boot_ci(v, a.bootstrap)
                rec["rho_%s_mean" % lab] = float(np.mean(v)) if len(v) else np.nan
                rec["rho_%s_sd" % lab] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
                rec["rho_%s_median" % lab] = float(np.median(v)) if len(v) else np.nan
                rec["rho_%s_ci_lo" % lab], rec["rho_%s_ci_hi" % lab] = lo, hi
                rec["aucov_%s_mean" % lab] = float(s["aucov_%s" % lab].mean())
                for kk in KS:
                    rec["iou%g_%s_mean" % (kk, lab)] = float(s["iou%g_%s" % (kk, lab)].mean())
            rec["self_influence"] = (rec["rho_full_mean"] - rec["rho_self_mean"]) if "self" in MODES else np.nan
            rec["delta_rho"] = (rec["rho_self_mean"] - rec["rho_lineage_mean"]) if ("self" in MODES and "lineage" in MODES) else np.nan
            out.append(rec)
        return pd.DataFrame(out)

    S = summarize(P, "ALL")
    per_ds = [summarize(P[P["dataset"] == d], d) for d in sorted(set(P["dataset"]))]
    ALL = pd.concat([S] + [x for x in per_ds if len(x)], ignore_index=True)
    ALL.round(6).to_csv(os.path.join(a.out_dir, "model_similarity_summary.csv"), index=False)

    W = 150
    TITLES = {"full": "Full (no exclusion)", "self": "Self-LOO", "lineage": "Lineage-LOO"}
    for lab, title in [(m_, TITLES[m_]) for m_ in MODES]:
        print("\n" + "=" * W)
        print("Table. %s  -  all %d images" % (title, len(keys)))
        print("=" * W)
        t = S.sort_values("rho_%s_mean" % lab, ascending=False)
        print("   %-32s %-9s %-7s %9s %9s %9s"
              % ("model", "lineage", "in_pool", "rho", "SD", "TopkAUC"))
        for _, r in t.iterrows():
            print("   %-32s %-9s %-7d %9.4f %9.4f %9.4f"
                  % (r["model"][:32], r["lineage"], r["in_pool"],
                     r["rho_%s_mean" % lab], r["rho_%s_sd" % lab], r["aucov_%s_mean" % lab]))

    print("\n" + "=" * W)
    print("mode comparison / self-inclusion diagnostics")
    print("=" * W)
    if len(MODES) < 3:
        print("  (--modes %s: self/lineage mode comparison skipped)" % MODES)
    else:
        z = (S["self_influence"].abs() < 1e-12).sum()
        print("  mean rho:  full %.4f | self %.4f | lineage %.4f"
              % (S["rho_full_mean"].mean(), S["rho_self_mean"].mean(), S["rho_lineage_mean"].mean()))
        print("  self_influence mean %.4f / max %.4f | models with exactly 0: %d (= outside pool: %d)"
              % (S["self_influence"].mean(), S["self_influence"].max(), z,
                 sum(1 for m in TARGETS if not inp[m])))
        for k1, k2 in (("full", "self"), ("full", "lineage"), ("self", "lineage")):
            d = S[["rho_%s_mean" % k1, "rho_%s_mean" % k2]].dropna()
            print("  rank agreement %s vs %s : Spearman %+.4f"
                  % (k1, k2, d.corr(method="spearman").iloc[0, 1] if len(d) >= 3 else np.nan))

    print("\n" + "=" * W)
    print("Full rho per dataset (top 5 models)")
    print("=" * W)
    piv = ALL.pivot_table(index="model", columns="scope", values="rho_full_mean")
    print(piv.round(4).to_string())

    print("\nSaved: %s" % a.out_dir)
    for n in ("per_image_similarity.csv", "model_similarity_summary.csv", "analysis.log"):
        print("  %s" % n)


if __name__ == "__main__":
    main()
