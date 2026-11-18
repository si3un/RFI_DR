"""
bootstrap_classification_ci.py - bootstrap 95% CI for Table 2 / Table S2 (image-level, 2,000 resamples, seed 42).

Background: the CIs in test_metrics.json / external_metrics.json from trainers/utils.py use the normal-approximation (Wald) formula,
      which differs from the paper's Methods (bootstrap percentile). They are recomputed from the prediction CSVs. Existing files are not touched.

Reads: {run_dir}/test_predictions.csv, external_predictions.csv (columns include label, pred),
        0_best_thr from {run_dir}/test_metrics.json (= validation Youden threshold; same as checkpoint val_thresholds)
Output: {out_dir}/bootstrap_classification_ci.csv
        model, split(internal/external), n, n_pos, thr, auroc, auroc_lo, auroc_hi, sens, sens_lo, sens_hi, spec, spec_lo, spec_hi

Usage:
  python interpreters/bootstrap_classification_ci.py \
      --run_roots $OUTPUT_ROOT/dr_clean $OUTPUT_ROOT/ckpts_aux \
      --out_dir $OUTPUT_ROOT/bootstrap_ci_v1
"""
import os, glob, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def boot(y, p, thr, n=2000, seed=42):
    rng = np.random.default_rng(seed); N = len(y); A, S, P = [], [], []
    pred = (p >= thr).astype(int)
    for _ in range(n):
        i = rng.integers(0, N, N); yi, pi, di = y[i], p[i], pred[i]
        if yi.min() == yi.max(): continue
        A.append(roc_auc_score(yi, pi))
        S.append(((di == 1) & (yi == 1)).sum() / max((yi == 1).sum(), 1))
        P.append(((di == 0) & (yi == 0)).sum() / max((yi == 0).sum(), 1))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(A), q(S), q(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for root in a.run_roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if "_old" in d or not os.path.isfile(os.path.join(d, "test_metrics.json")): continue
            model = os.path.basename(d).rsplit("_2026", 1)[0]
            thr = float(json.load(open(os.path.join(d, "test_metrics.json")))["0_best_thr"])
            for split, fn in (("internal", "test_predictions.csv"), ("external", "external_predictions.csv")):
                f = os.path.join(d, fn)
                if not os.path.isfile(f): print("  [missing]", f); continue
                P = pd.read_csv(f)
                lab = "label" if "label" in P.columns else [c for c in P.columns if "label" in c.lower()][0]
                prd = "pred" if "pred" in P.columns else ("output_0" if "output_0" in P.columns else [c for c in P.columns if "pred" in c.lower() or "prob" in c.lower()][0])
                if "test" in P.columns: P = P[P["test"] == True]   # test_predictions.csv: test rows only
                y, p = P[lab].to_numpy().astype(int), P[prd].to_numpy().astype(float)
                if split == "internal": print("   internal rows %d (test==True), thr %.4f" % (len(y), thr))
                (alo, ahi), (slo, shi), (plo, phi) = boot(y, p, thr, a.nboot, a.seed)
                pred = (p >= thr).astype(int)
                rows.append(dict(model=model, split=split, n=len(y), n_pos=int(y.sum()), thr=thr,
                                 auroc=roc_auc_score(y, p), auroc_lo=alo, auroc_hi=ahi,
                                 sens=((pred == 1) & (y == 1)).sum() / (y == 1).sum(), sens_lo=slo, sens_hi=shi,
                                 spec=((pred == 0) & (y == 0)).sum() / (y == 0).sum(), spec_lo=plo, spec_hi=phi))
            print("%-45s done" % model)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out_dir, "bootstrap_classification_ci.csv"), index=False)
    R["auroc_cell"] = R.apply(lambda r: "%.3f (%.3f-%.3f)" % (r.auroc, r.auroc_lo, r.auroc_hi), axis=1)
    R["sens_cell"] = R.apply(lambda r: "%.3f (%.3f-%.3f)" % (r.sens, r.sens_lo, r.sens_hi), axis=1)
    R["spec_cell"] = R.apply(lambda r: "%.3f (%.3f-%.3f)" % (r.spec, r.spec_lo, r.spec_hi), axis=1)
    print(R.pivot(index="model", columns="split", values="auroc_cell").to_string())
    print(R[R.split == "internal"][["model", "thr", "sens_cell", "spec_cell"]].to_string(index=False))


if __name__ == "__main__":
    main()
