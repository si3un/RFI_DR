# -*- coding: utf-8 -*-
"""
supp_table_s5_dice.py -- Supplementary Table S5 (values shown in Figure 2): per-case optimal Dice of every configuration and of the consensus map, with
image-level bootstrap 95% CIs (2,000 resamples, seed 42; each image resampled with all of its lesion-type cases),
plus the reference rows (random reference, lesion-location reference, perfect score using saliency resolution)
taken from baseline_avgmask_patchceiling.py. No maps are recomputed.

Usage:
  python interpreters/supp_table_s5_dice.py \
      --per_case "outputs/individual_localization/*_rules_per_case.csv" \
                 "outputs/consensus_localization/*_rules_per_case.csv" --consensus "UnitSum-Arith-T2x2" \
      --baseline outputs/baseline/baseline_summary.csv --out outputs/supp_table_s5_dice.csv
"""
import glob, argparse
import numpy as np, pandas as pd

ORDER = ["eophtha", "idrid", "fgadr"]
REF = {"RandomRank": "Random reference", "AverageMask-LOO": "Lesion-location reference",
       "PatchCeiling-14": "Perfect score using saliency resolution"}


def boot_ci(per_img_sum, per_img_n, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    m = len(per_img_sum)
    out = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, m, m)
        out[b] = per_img_sum[idx].sum() / per_img_n[idx].sum()
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_case", nargs="+", required=True)
    ap.add_argument("--consensus", default="UnitSum-Arith-T2x2")
    ap.add_argument("--baseline", default="outputs/baseline/baseline_summary.csv")
    ap.add_argument("--out", default="outputs/supp_table_s5_dice.csv")
    a = ap.parse_args()
    files = [f for pat in a.per_case for f in sorted(glob.glob(pat))]
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    D = D.drop_duplicates(["dataset", "image_id", "lesion_type", "method"], keep="last")
    D = D[(D.kind == "individual") | (D.method == a.consensus)]
    rows = []
    for ds in [d for d in ORDER if d in set(D.dataset)]:
        for m, g in D[D.dataset == ds].groupby("method"):
            s = g.groupby("image_id").max_dice.agg(["sum", "count"])
            lo, hi = boot_ci(s["sum"].to_numpy(), s["count"].to_numpy())
            rows.append(dict(dataset=ds, method=("Consensus map" if m == a.consensus else m), n_cases=len(g),
                             n_images=len(s), dice=g.max_dice.mean(), ci_lo=lo, ci_hi=hi))
    B = pd.read_csv(a.baseline)
    B = B[(B.lesion_type == "ALL") & B.method.isin(REF)]
    for _, r in B.iterrows():
        rows.append(dict(dataset=r.dataset, method=REF[r.method], n_cases=int(r.n), n_images=np.nan,
                         dice=r.mean_dice, ci_lo=r.lo, ci_hi=r.hi))
    T = pd.DataFrame(rows)
    T.round(4).to_csv(a.out, index=False)
    W = T.assign(cell=T.apply(lambda r: "%.3f (%.3f-%.3f)" % (r.dice, r.ci_lo, r.ci_hi), axis=1)).pivot(index="method", columns="dataset", values="cell")
    W = W[[d for d in ORDER if d in W.columns]]
    key = T[T.dataset == "fgadr"].set_index("method").dice if "fgadr" in set(T.dataset) else None
    if key is not None:
        W = W.loc[key.reindex(W.index).sort_values(ascending=False).index]
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    print(W.to_string())
    print("\nsaved ->", a.out)


if __name__ == "__main__":
    main()
