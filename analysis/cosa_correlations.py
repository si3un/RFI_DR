"""Correlations reported in Results (CoSA section) and Supplementary Table S6, computed from saved results:

  * rank correlation (Spearman) between agreement with the consensus (CoSA) and lesion-localization Dice,
    for the pool configurations, per annotated dataset (Spearman CoSA and top-k overlap AUC);
  * the same for internal AUROC vs Dice (discrimination does not track localization);
  * self- and lineage-leave-one-out: mean drop in Spearman agreement and rank correlation of the LOO ranking
    with the full-consensus ranking.

Inputs: model_similarity_summary.csv (interpreters/ext_consensus_similarity_loo.py, pool run),
the Dice table written by interpreters/supp_table_s5_dice.py (Supplementary Table S5), and
bootstrap_classification_ci.csv (interpreters/bootstrap_classification_ci.py).
Note: the manuscript values were first computed on rounded (3-decimal) table values; full precision may differ
in the last digit.

Usage:
  python analysis/cosa_correlations.py \
      --cosa  $OUTPUT_ROOT/ext_cons_sim_1x14_arith_trim2/model_similarity_summary.csv \
      --dice  $OUTPUT_ROOT/supp_table_s5_dice.csv \
      --auroc $OUTPUT_ROOT/bootstrap_ci_v1/bootstrap_classification_ci.csv
"""
import argparse
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import cosa_table, internal_auroc, norm_name  # noqa: E402

ORDER = ["eophtha", "idrid", "fgadr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosa", required=True)
    ap.add_argument("--dice", required=True)
    ap.add_argument("--auroc", required=True)
    a = ap.parse_args()

    cosa = cosa_table(a.cosa)
    pool = cosa[cosa["in_pool"] == 1]
    dice = pd.read_csv(a.dice)
    dice["model"] = dice["method"].map(norm_name)
    dice = dice.pivot_table(index="model", columns="dataset", values="dice")
    auroc = internal_auroc(a.auroc)

    print("Pool configurations: %d" % len(pool))
    print("\nSpearman rank correlation with per-case optimal Dice (pool configurations)")
    print("%-10s %12s %12s %12s" % ("dataset", "CoSA rho", "CoSA top-k", "int. AUROC"))
    for ds in [d for d in ORDER if d in dice.columns]:
        j = pool[["rho_full_mean", "aucov_full_mean"]].join(dice[ds].rename("dice"), how="inner").join(
            auroc.rename("auroc"), how="left")
        r1 = spearmanr(j["rho_full_mean"], j["dice"]).correlation
        r2 = spearmanr(j["aucov_full_mean"], j["dice"]).correlation
        r3 = spearmanr(j["auroc"], j["dice"]).correlation
        print("%-10s %12.2f %12.2f %12.2f   (n=%d)" % (ds, r1, r2, r3, len(j)))

    print("\nLeave-one-out sensitivity (pool configurations, all external images)")
    for lab in ("self", "lineage"):
        col = "rho_%s_mean" % lab
        if col not in pool.columns:
            continue
        drop = (pool["rho_full_mean"] - pool[col]).mean()
        rc = spearmanr(pool["rho_full_mean"], pool[col]).correlation
        print("  %-8s mean drop in Spearman agreement %.3f | rank correlation with full ranking %.2f" % (lab, drop, rc))


if __name__ == "__main__":
    main()
