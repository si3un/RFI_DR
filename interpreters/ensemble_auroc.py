"""
ensemble_auroc.py — recomputes Table 3 (Ensemble vs Single-Model classification).

Reads only the saved per-model prediction CSVs, **averages probabilities** per pool, and recomputes AUROC.
No new inference, so no GPU is needed; it finishes in a few seconds.

Inputs per split (all already exist):
  internal : {model_dir}/test_predictions.csv       (original df + output_0/1/...)
  external : {model_dir}/external_predictions.csv   (image_path, label, pred)
  snuh     : {snu_dir}/{model}/snu_preds.csv        (idx, label, pred[, dataset])

Pool definitions (all based on internal test AUROC = 0_auc in test_metrics.json):
  Single         : all individual models (only the top few by AUROC are printed)
  Top-5          : top 5 by AUROC
  Lineage-8      : best-AUROC model for each of 4 lineages (CLIP/DINOv2/DINOv3/MAE) × FT (Partial/LoRA)
  AUROC>=0.90    : **all** models passing the threshold (including Linear — no filtering here)
  AUROC>=0.95    : 〃
  All            : all models

Note: this script does not filter by adaptation strategy, so each pool contains every configuration
   passing the threshold. To restrict a pool to partial fine-tuning and LoRA, pass --ft Partial LoRA.

Usage:
  # internal / external (CSVs are stored in the model directories)
  python interpreters/ensemble_auroc.py --split internal \
    --model_roots models/dr_clean models/ckpts_aux --out_dir outputs/ens_internal
  python interpreters/ensemble_auroc.py --split external \
    --model_roots models/dr_clean models/ckpts_aux --out_dir outputs/ens_external

  # SNUH (requires the snu_eval_dr.py output directory)
  python interpreters/ensemble_auroc.py --split snuh \
    --model_roots models/dr_clean models/ckpts_aux \
    --snu_dir outputs/snu_eval_main --out_dir outputs/ens_snuh

  # for the Any-DR (mild) model set, use --model_roots models/dr_clean_mild models/ckpts_aux_mild
"""
import os, sys, re, glob, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LINEAGES = ["CLIP", "DINOv2", "DINOv3", "MAE"]
LABEL_CANDS = ["label", "target", "y", "gt"]


def strip_ts(n):
    return re.sub(r"_\d{8}_\d{6}$", "", n)


def lineage_of(name):
    n = name.lower()
    if "clip" in n or "laion" in n: return "CLIP"
    if "dinov3" in n: return "DINOv3"
    if "dinov2" in n: return "DINOv2"
    if "mae" in n or "retfound" in n: return "MAE"
    return None


def ft_of(name):
    n = name.lower()
    return "LoRA" if "lora" in n else "Partial" if "partial" in n else "Linear" if "linear" in n else None


def pretty(name):
    n = name.lower()
    bb = ("OpenCLIP" if ("clip" in n or "laion" in n) else
          "RETFound-DINOv2" if "retfound_dinov2" in n else
          "RETFound" if ("retfound_mae" in n or n.startswith("retfound")) else
          ("DINOv3-Huge" if ("huge" in n or "vith" in n) else "DINOv3-Large") if "dinov3" in n else
          "DINOv2" if "dinov2" in n else "MAE" if "mae" in n else name)
    return "%s-%s" % (bb, ft_of(name))


def read_auc(mdir):
    p = os.path.join(mdir, "test_metrics.json")
    if not os.path.exists(p):
        return np.nan
    try:
        d = json.load(open(p))
        for k in ("0_auc", "auc", "test_auc"):
            if k in d:
                v = d[k]
                return float(v.split()[0]) if isinstance(v, str) else float(v)
    except Exception:
        pass
    return np.nan


def to_prob(x):
    """Apply sigmoid to raw logits; leave probabilities as is. The ensemble must average probabilities."""
    x = np.asarray(x, dtype=np.float64)
    if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
        return 1.0 / (1.0 + np.exp(-x))
    return x


def load_split(name, mdir, split, snu_dir):
    """Returns (key[], label[], prob[]). key is used to align samples across models."""
    if split == "internal":
        p = os.path.join(mdir, "test_predictions.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        lab = next((c for c in LABEL_CANDS if c in df.columns), None)
        if lab is None or "output_0" not in df.columns:
            print("  [skip] %s: missing column (label=%s, output_0=%s) → %s"
                  % (name, lab, "output_0" in df.columns, list(df.columns)[:8]))
            return None
        key = df["image_path"].astype(str) if "image_path" in df.columns else pd.Series(np.arange(len(df)))
        return key.to_numpy(), df[lab].to_numpy(), to_prob(df["output_0"].to_numpy())

    if split == "external":
        p = os.path.join(mdir, "external_predictions.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        if not {"label", "pred"} <= set(df.columns):
            return None
        key = df["image_path"].astype(str) if "image_path" in df.columns else pd.Series(np.arange(len(df)))
        return key.to_numpy(), df["label"].to_numpy(), to_prob(df["pred"].to_numpy())

    p = os.path.join(snu_dir, name, "snu_preds.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if not {"label", "pred"} <= set(df.columns):
        return None
    key = df["idx"].to_numpy() if "idx" in df.columns else np.arange(len(df))
    return key, df["label"].to_numpy(), to_prob(df["pred"].to_numpy())


def boot_ci(y, s, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], s[i]))
    if not out:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["internal", "external", "snuh"])
    ap.add_argument("--model_roots", nargs="+", required=True)
    ap.add_argument("--snu_dir", default="", help="snu_eval_dr.py output directory when split=snuh")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.90, 0.95])
    ap.add_argument("--ft", nargs="*", default=[],
                    help="restrict AUROC pools to these FT methods (e.g. Partial LoRA). Default: no restriction")
    ap.add_argument("--nboot", type=int, default=2000, help="0 skips the CI")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.split == "snuh" and not a.snu_dir:
        sys.exit("[abort] --split snuh requires --snu_dir.")

    # ── collect models ──
    dirs = {}
    for root in a.model_roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                dirs.setdefault(strip_ts(os.path.basename(d)), d)
    models, auc, P = [], {}, {}
    for name, d in dirs.items():
        if lineage_of(name) is None or ft_of(name) is None:
            continue
        r = load_split(name, d, a.split, a.snu_dir)
        if r is None:
            continue
        auc[name] = read_auc(d)
        P[name] = r
        models.append(name)
    if not models:
        sys.exit("[abort] no models with prediction CSVs. split=%s, roots=%s" % (a.split, a.model_roots))

    # ── align on common samples (rows may differ between models) ──
    common = None
    for m in models:
        s = set(P[m][0].tolist())
        common = s if common is None else (common & s)
    common = sorted(common)
    idx_of, y_ref = {}, None
    for m in models:
        k, y, p = P[m]
        pos = {kk: i for i, kk in enumerate(k)}
        sel = np.array([pos[c] for c in common])
        idx_of[m] = (y[sel], p[sel])
        if y_ref is None:
            y_ref = y[sel]
        elif not np.array_equal(y_ref, y[sel]):
            print("  [warning] %s: labels disagree with other models — check key alignment" % m)
    y = np.asarray(y_ref)
    print("[%s] %d models | %d common samples | %d positive (%.1f%%)"
          % (a.split, len(models), len(y), int(y.sum()), 100 * y.mean()))

    order = sorted(models, key=lambda m: -(auc[m] if np.isfinite(auc[m]) else -1))
    noauc = [m for m in models if not np.isfinite(auc[m])]
    if noauc:
        print("  [note] %d models without internal AUROC (excluded from threshold pools):" % len(noauc), noauc)

    def restrict(ms):
        if not a.ft:
            return ms
        want = {x.lower() for x in a.ft}
        return [m for m in ms if str(ft_of(m)).lower() in want]

    # ── build pools ──
    pools = {}
    for m in order[:3]:
        pools["Single: %s" % pretty(m)] = [m]
    pools["Top-5 consensus"] = order[:5]
    b8 = []
    for g in LINEAGES:
        for f in ("Partial", "LoRA"):
            c = [(auc[m], m) for m in models
                 if lineage_of(m) == g and ft_of(m) == f and np.isfinite(auc[m])]
            if c:
                b8.append(max(c)[1])
    pools["Lineage consensus"] = b8
    for t in a.thresholds:
        sel = restrict([m for m in order if np.isfinite(auc[m]) and auc[m] >= t])
        pools["AUROC>=%.2f consensus" % t] = sel
    pools["All models consensus"] = list(order)

    rows = []
    for tag, ms in pools.items():
        if not ms:
            print("  [skip] %s: pool is empty" % tag); continue
        s = np.mean([idx_of[m][1] for m in ms], axis=0)     # mean probability
        au = roc_auc_score(y, s)
        lo, hi = boot_ci(y, s, a.nboot) if a.nboot else (np.nan, np.nan)
        rows.append(dict(config=tag, n_models=len(ms), AUROC=au, lo=lo, hi=hi,
                         members="|".join(pretty(m) for m in ms)))
    out = pd.DataFrame(rows)
    p = os.path.join(a.out_dir, "ensemble_auroc_%s.csv" % a.split)
    out.round(6).to_csv(p, index=False)

    print("\n===== %s : Ensemble vs Single =====" % a.split)
    disp = out[["config", "n_models", "AUROC", "lo", "hi"]].copy()
    disp["AUROC"] = disp.AUROC.round(4)
    disp["95%CI"] = ["%.4f–%.4f" % (l, h) if np.isfinite(l) else "-"
                     for l, h in zip(disp.lo, disp.hi)]
    print(disp[["config", "n_models", "AUROC", "95%CI"]].to_string(index=False))
    print("\n  Pools are based on internal test AUROC (0_auc). Ensemble = simple mean of predicted probabilities.")
    if a.ft:
        print("  [note] pools restricted with --ft %s." % " ".join(a.ft))
    print("Saved:", p)


if __name__ == "__main__":
    main()
