"""
per_dataset_external_auroc.py - Supplementary Table S2(c): external AUROC per dataset for all 21 configurations.

Reads: {run_dir}/external_predictions.csv (columns: image_path,label,grade,dataset,pred)
Output: {out_dir}/per_dataset_external_auroc.csv  (model, dataset, n, n_pos, auroc, lo, hi)
CI    : image-level bootstrap (default 1,000 resamples, seed 42) - per-dataset external convention.
Existing results are not touched (saved only to the new out_dir).

Usage:
  python interpreters/per_dataset_external_auroc.py \
      --run_roots $OUTPUT_ROOT/dr_clean $OUTPUT_ROOT/ckpts_aux \
      --out_dir $OUTPUT_ROOT/per_dataset_external_auroc
"""
import os, glob, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def boot_auc(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed); N = len(y); out = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        if y[i].min() == y[i].max(): continue
        out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for root in a.run_roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            f = os.path.join(d, "external_predictions.csv")
            if not os.path.isfile(f) or "_old" in d: continue
            model = os.path.basename(d).rsplit("_2026", 1)[0]
            P = pd.read_csv(f)
            for ds, g in P.groupby("dataset"):
                y, p = g.label.to_numpy().astype(int), g.pred.to_numpy().astype(float)
                if y.min() == y.max(): continue
                lo, hi = boot_auc(y, p, a.nboot)
                rows.append(dict(model=model, dataset=ds, n=len(y), n_pos=int(y.sum()), auroc=roc_auc_score(y, p), lo=lo, hi=hi))
            y, p = P.label.to_numpy().astype(int), P.pred.to_numpy().astype(float)
            lo, hi = boot_auc(y, p, a.nboot)
            rows.append(dict(model=model, dataset="ALL", n=len(y), n_pos=int(y.sum()), auroc=roc_auc_score(y, p), lo=lo, hi=hi))
            print("%-45s done (%d datasets)" % (model, P.dataset.nunique()))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out_dir, "per_dataset_external_auroc.csv"), index=False)
    R["cell"] = R.apply(lambda r: "%.3f (%.3f-%.3f)" % (r.auroc, r.lo, r.hi), axis=1)
    W = R.pivot(index="model", columns="dataset", values="cell")
    W.to_csv(os.path.join(a.out_dir, "per_dataset_external_auroc_wide.csv"))
    print(W.to_string())
    print("n per dataset:", R[R.model == R.model.iloc[0]][["dataset", "n", "n_pos"]].to_string(index=False))


if __name__ == "__main__":
    main()
