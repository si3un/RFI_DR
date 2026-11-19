# -*- coding: utf-8 -*-
"""
table3_compare.py -- compare CoSA results (Table 3) between consensus rules (Supplementary Table S6, trimming one value
at each end instead of two).

Reads model_similarity_summary.csv (scope == ALL) from a base pair of run folders (pool-14 run + classifier-only run,
paper rule arith_trim2) and from one or more comparison pairs, prints per-model Spearman agreement (rho_full_mean, rho_self_mean) and top-k overlap
AUC side by side, rank changes, the top-agreement model, and -- if per-dataset Dice files are given -- the Spearman
rank correlation across the 14 pool configurations between agreement and mean per-case optimal Dice.

Dice input: *_rules_summary.csv files from eval_individual_localization.py; dataset is taken from the file-name prefix
(eophtha_/idrid_/fgadr_), rows with kind == individual, columns method (tag) and dice.

Usage:
  python interpreters/table3_compare.py \
    --base outputs/ext_cons_sim_1x14_arith_trim2 outputs/ext_cons_sim_1x_cls2617_arith_trim2 \
    --compare trim1=outputs/ext_cons_sim_1x14_arith_trim1,outputs/ext_cons_sim_1x_cls2617_arith_trim1 \
    --dice outputs/individual_localization/eophtha_rules_summary.csv outputs/individual_localization/idrid_rules_summary.csv \
           outputs/individual_localization/fgadr_rules_summary.csv \
    --out_csv table3_compare_arith_trim1.csv
"""
import os, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def load_pair(dirs):
    S = pd.concat([pd.read_csv(os.path.join(d, "model_similarity_summary.csv")) for d in dirs], ignore_index=True)
    if "scope" in S.columns:
        S = S[S.scope == "ALL"]
    S = S.drop_duplicates("model", keep="first").set_index("model")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="run folders of the paper rule (arith_trim2)")
    ap.add_argument("--compare", nargs="+", required=True, help="name=dir1,dir2 ...")
    ap.add_argument("--dice", nargs="*", default=[])
    ap.add_argument("--out_csv", default="table3_compare.csv")
    a = ap.parse_args()
    runs = {"arith_trim2 (paper)": load_pair(a.base)}
    for spec in a.compare:
        name, dirs = spec.split("=", 1)
        runs[name] = load_pair(dirs.split(","))
    models = list(runs["arith_trim2 (paper)"].index)
    pool = [m for m in models if int(runs["arith_trim2 (paper)"].loc[m, "in_pool"]) == 1]
    pd.set_option("display.width", 250)

    for col, label in (("rho_full_mean", "Spearman agreement with the full consensus (Table 3 column)"),
                       ("rho_self_mean", "Spearman agreement, self-LOO"),
                       ("aucov_full_mean", "top-k overlap AUC, full consensus")):
        T = pd.DataFrame({k: v[col].reindex(models) for k, v in runs.items()})
        for k in runs:
            T["rank " + k] = T[k].where(T.index.isin(pool)).rank(ascending=False)
        print("\n===== %s =====" % label)
        print(T.round(3).to_string())
        best = {k: T[k].where(T.index.isin(pool)).idxmax() for k in runs}
        print("top pool model:", best)

    if a.dice:
        print("\n===== agreement vs Dice rank correlation across the pool (Spearman rho, P) =====")
        rows = []
        for f in a.dice:
            ds = os.path.basename(f).split("_")[0]
            D = pd.read_csv(f)
            D = D[D.kind == "individual"].set_index("method")["dice"]
            for k, S in runs.items():
                x = S["rho_full_mean"].reindex(pool); y = D.reindex(pool)
                ok = x.notna() & y.notna()
                r, p = spearmanr(x[ok], y[ok])
                rows.append(dict(dataset=ds, rule=k, n=int(ok.sum()), rho=r, p=p))
        R = pd.DataFrame(rows).pivot(index="rule", columns="dataset", values="rho")
        print(R.round(3).to_string())
        pd.DataFrame(rows).to_csv(a.out_csv.replace(".csv", "_rankcorr.csv"), index=False)
    out = pd.concat({k: v[["in_pool", "rho_full_mean", "rho_self_mean", "aucov_full_mean"]].reindex(models) for k, v in runs.items()}, axis=1)
    out.to_csv(a.out_csv)
    print("\nsaved ->", a.out_csv)


if __name__ == "__main__":
    main()
