"""Supplementary Table S2(a): AUPRC in the SNUH cohort (with the random-baseline AUPRC = prevalence).

Reads the per-image predictions written by interpreters/snu_eval_dr.py (<pred_root>/<configuration>/snu_preds.csv,
columns: idx, label, pred). Run on the SNU server; no data leave it.

Usage:
  python analysis/snuh_auprc.py --pred_roots $OUTPUT_ROOT/snu_metrics_lenient \
      $OUTPUT_ROOT/snu_metrics_lenient_rfdv2 $OUTPUT_ROOT/snu_metrics_lenient_rfdv2_partial \
      --out $OUTPUT_ROOT/snuh_auprc.csv
"""
import argparse
import glob
import os
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import norm_name, strategy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_roots", nargs="+", required=True)
    ap.add_argument("--out", default="snuh_auprc.csv")
    a = ap.parse_args()

    rows = {}
    for root in a.pred_roots:                       # later roots override earlier ones (re-runs)
        for f in sorted(glob.glob(os.path.join(root, "*", "snu_preds.csv"))):
            p = pd.read_csv(f)
            name = norm_name(os.path.basename(os.path.dirname(f)))
            rows[name] = dict(model=name, strategy=strategy(name), n=len(p), n_pos=int(p["label"].sum()),
                              prevalence=p["label"].mean(), auroc=roc_auc_score(p["label"], p["pred"]),
                              auprc=average_precision_score(p["label"], p["pred"]), source=root)
    t = pd.DataFrame(rows.values()).sort_values(["model"])
    t.round(4).to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(t[["model", "n", "n_pos", "prevalence", "auroc", "auprc"]].round(4).to_string(index=False))
    print("\nsaved ->", a.out)


if __name__ == "__main__":
    main()
