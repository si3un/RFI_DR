"""Model selection with CoSA (Methods, 'Consensus-based Saliency Agreement and model selection') and its
sensitivity to the AUROC threshold (Results: 'The selection was unchanged for thresholds from 0.92 to 0.96').

Rule: among configurations whose internal-test AUROC is >= threshold, rank the two agreement measures
(mean Spearman correlation and mean top-k overlap AUC with the consensus map; higher is better), average the two
ranks, pick the best mean rank, and break ties by internal AUROC.

Inputs: model_similarity_summary.csv of the pool run (interpreters/ext_consensus_similarity_loo.py) and
bootstrap_classification_ci.csv (interpreters/bootstrap_classification_ci.py).

Usage:
  python analysis/selection_rule.py \
      --cosa $OUTPUT_ROOT/ext_cons_sim_1x14_arith_trim2/model_similarity_summary.csv \
      --auroc $OUTPUT_ROOT/bootstrap_ci/bootstrap_classification_ci.csv \
      --thresholds 0.92 0.93 0.94 0.95 0.96 --main 0.95
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import cosa_table, internal_auroc  # noqa: E402


def select(tab: pd.DataFrame, thr: float) -> pd.DataFrame:
    c = tab[tab["auroc"] >= thr].copy()
    c["rank_spearman"] = c["rho_full_mean"].rank(ascending=False, method="min")
    c["rank_topk"] = c["aucov_full_mean"].rank(ascending=False, method="min")
    c["mean_rank"] = (c["rank_spearman"] + c["rank_topk"]) / 2
    return c.sort_values(["mean_rank", "auroc"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosa", required=True)
    ap.add_argument("--auroc", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.92, 0.93, 0.94, 0.95, 0.96])
    ap.add_argument("--main", type=float, default=0.95)
    a = ap.parse_args()

    cosa = cosa_table(a.cosa)
    tab = cosa[["rho_full_mean", "aucov_full_mean"]].join(internal_auroc(a.auroc).rename("auroc"), how="inner")
    missing = sorted(set(cosa.index) - set(tab.index))
    if missing:
        print("[warn] no internal AUROC for:", missing)

    pd.set_option("display.width", 200)
    print("Main rule (internal AUROC >= %.2f):" % a.main)
    print(select(tab, a.main).round(4).to_string())
    print("\nSensitivity to the threshold:")
    for thr in a.thresholds:
        s = select(tab, thr)
        top = s.index[0] if len(s) else "-"
        tie = s[s["mean_rank"] == s["mean_rank"].iloc[0]].index.tolist() if len(s) else []
        print("  >= %.2f : n=%2d  selected=%s%s" % (thr, len(s), top,
                                                   ("  (tied on mean rank: %s)" % ", ".join(tie)) if len(tie) > 1 else ""))


if __name__ == "__main__":
    main()
