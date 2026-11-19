"""Supplementary Table S7: per-case optimal Dice of consensus maps built with alternative normalizations
(min-max, L1 = unit sum, percentile rank) and amounts of trimming (none, one or two values at each end;
arithmetic mean of the remaining values).

Reads the per-case files written by interpreters/consensus_rules_v5_means.py
({dataset}[_shardXofY]_rules_per_case.csv; method names such as 'UnitSum-Arith-T2x2'). FGADR shards are pooled.

Usage:
  python analysis/supp_s7_norm_trim.py \
      --per_case "$OUTPUT_ROOT/consensus_rules_v5b_means/*_rules_per_case.csv" \
                 "$OUTPUT_ROOT/consensus_rules_v5c_means/*_rules_per_case.csv" \
      --out $OUTPUT_ROOT/supp_s7_norm_trim.csv
"""
import argparse
import glob

import pandas as pd

ORDER = ["eophtha", "idrid", "fgadr"]
NORM_LABEL = {"MinMax": "Min-max", "UnitSum": "L1 (unit sum)", "Rank": "Percentile rank"}
TRIM_LABEL = {"T0": "None", "T1x1": "One", "T2x2": "Two"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_case", nargs="+", required=True)
    ap.add_argument("--mean", default="Arith")
    ap.add_argument("--out", default="supp_s7_norm_trim.csv")
    a = ap.parse_args()

    files = [f for pat in a.per_case for f in sorted(glob.glob(pat))]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d["kind"] == "consensus"]
    d = d.drop_duplicates(["dataset", "image_id", "lesion_type", "method"], keep="last")
    parts = d["method"].str.split("-", n=2, expand=True)
    d = d.assign(norm=parts[0], mean=parts[1], trim=parts[2])
    d = d[(d["mean"] == a.mean) & d["norm"].isin(NORM_LABEL) & d["trim"].isin(TRIM_LABEL)]

    t = d.groupby(["norm", "trim", "dataset"])["max_dice"].mean().unstack("dataset")
    t = t[[c for c in ORDER if c in t.columns]].reset_index()
    t["Normalization"] = t["norm"].map(NORM_LABEL)
    t["Values discarded at each end"] = t["trim"].map(TRIM_LABEL)
    t = t.sort_values(["norm", "trim"], key=lambda s: s.map({k: i for i, k in enumerate(list(NORM_LABEL) + list(TRIM_LABEL))}))
    out = t[["Normalization", "Values discarded at each end"] + [c for c in ORDER if c in t.columns]]
    out.round(4).to_csv(a.out, index=False)
    print(out.round(3).to_string(index=False))
    print("\nn cases per dataset:", d.groupby("dataset")[["image_id", "lesion_type"]].apply(
        lambda g: len(g.drop_duplicates())).to_dict())
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
