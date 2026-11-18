# -*- coding: utf-8 -*-
"""
table3_pairwise.py -- numbers for the Table 3 paragraph that model_similarity_summary.csv does not contain.

  [1] paired differences in agreement between two configurations (image-level paired bootstrap, 2,000 resamples,
      seed 42; two-sided bootstrap P = 2 x the smaller share of resampled mean differences at or beyond zero),
      for Spearman (rho_full) and top-k overlap AUC (aucov_full)
  [2] lineage-leave-one-out: mean and 95% CI per pool configuration, the drop from the full consensus, the rank
      correlation of the lineage ranking with the full ranking, and the top three by lineage agreement

Usage (A100):
  python interpreters/table3_pairwise.py --runs outputs/ext_cons_sim_1x14_geo_drop4 \
      --pairs "DINOv2-LoRA:RETFound-DINOv2-LoRA" "DINOv2-LoRA:MAE-Partial"
"""
import os, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def paired_boot(x, y, n=2000, seed=42):
    d = x - y
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = 2.0 * min((b <= 0).mean(), (b >= 0).mean())
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", default=["DINOv2-LoRA:RETFound-DINOv2-LoRA", "DINOv2-LoRA:MAE-Partial"])
    a = ap.parse_args()
    D = pd.concat([pd.read_csv(os.path.join(r, "per_image_similarity.csv")) for r in a.runs], ignore_index=True)
    print("rows %d | images %d | models %d" % (len(D), D.image_id.nunique(), D.model.nunique()))
    print("\n[1] paired differences in agreement (first minus second)")
    for pr in a.pairs:
        m1, m2 = pr.split(":")
        for col in ("rho_full", "aucov_full"):
            X = D[D.model == m1].set_index("image_id")[col]
            Y = D[D.model == m2].set_index("image_id")[col]
            j = pd.concat([X.rename("x"), Y.rename("y")], axis=1).dropna()
            dm, lo, hi, p = paired_boot(j.x.to_numpy(), j.y.to_numpy())
            print("  %-22s - %-22s %-10s n=%d  diff %.4f  95%% CI %.4f to %.4f  P = %.4f" % (m1, m2, col, len(j), dm, lo, hi, p))
    P = D[D.in_pool == 1]
    if "rho_lineage" in P.columns:
        print("\n[2] lineage-leave-one-out (pool configurations)")
        rows = []
        rng = np.random.default_rng(42)
        for m, g in P.groupby("model"):
            v = g.rho_lineage.dropna().to_numpy()
            b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            rows.append(dict(model=m, full=g.rho_full.mean(), self=g.rho_self.mean(), lineage=v.mean(),
                             lin_ci_lo=np.percentile(b, 2.5), lin_ci_hi=np.percentile(b, 97.5),
                             aucov_lineage=g.aucov_lineage.mean() if "aucov_lineage" in g else np.nan))
        R = pd.DataFrame(rows).sort_values("lineage", ascending=False)
        pd.set_option("display.width", 200)
        print(R.round(4).to_string(index=False))
        print("  mean drop full -> lineage %.4f | rank corr lineage vs full %.3f | lineage vs self %.3f"
              % ((R.full - R.lineage).mean(), spearmanr(R.lineage, R.full).correlation, spearmanr(R.lineage, R.self).correlation))
        print("  top three by lineage agreement:", list(R.model[:3]), "| by full:", list(R.sort_values("full", ascending=False).model[:3]))


if __name__ == "__main__":
    main()
