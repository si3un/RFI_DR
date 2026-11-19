# -*- coding: utf-8 -*-
"""
compare_consensus_vs_individual.py -- image-level paired statistics, consensus map vs individual configurations
(as described in the Methods): per-case optimal Dice differences are averaged within each image,
tested with the two-sided Wilcoxon signed-rank test on image-level differences, bootstrap 95% CI resamples
images (2,000, seed 42), Holm correction over the comparisons within each dataset.

Inputs: one or more *_rules_per_case.csv files (from eval_individual_localization.py runs); rows are concatenated and
de-duplicated on (dataset, image_id, lesion_type, method), the last file winning.

Usage (the paper's consensus against every individual configuration is the default; --vs narrows the comparison):
  python interpreters/compare_consensus_vs_individual.py \
    --per_case "outputs/individual_localization/*_rules_per_case.csv" "outputs/consensus_localization/*_rules_per_case.csv" \
    --consensus "UnitSum-Arith-T2x2" \
    --vs "RETFound-DINOv2-LoRA" "DINOv2-LoRA" \
    --out_dir outputs/consensus_vs_individual
"""
import os, glob, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (n - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def image_level(D, ds, a_name, b_name):
    """image-level mean of (a - b) over the lesion-type cases of each image"""
    x = D[(D.dataset == ds) & (D.method == a_name)].set_index(["image_id", "lesion_type"]).max_dice.rename("a")
    y = D[(D.dataset == ds) & (D.method == b_name)].set_index(["image_id", "lesion_type"]).max_dice.rename("b")
    j = pd.concat([x, y], axis=1).dropna()
    if len(j) == 0:
        return None
    d = (j.a - j.b).groupby(level="image_id").mean()
    return d.to_numpy(), float(j.a.mean()), float(j.b.mean()), len(j)


def boot_ci(d, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_case", nargs="+", required=True)
    ap.add_argument("--consensus", nargs="+", required=True)
    ap.add_argument("--vs", nargs="+", default=["RETFound-DINOv2-LoRA", "DINOv2-LoRA"])
    ap.add_argument("--all_individuals", action="store_true", help="also compare against every individual configuration")
    ap.add_argument("--out_dir", default="outputs/consensus_vs_individual")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = [f for pat in a.per_case for f in sorted(glob.glob(pat))]
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    D = D.drop_duplicates(["dataset", "image_id", "lesion_type", "method"], keep="last")
    print("files:", len(files), "| rows:", len(D), "| methods:", D.method.nunique())
    rows = []
    for ds in sorted(D.dataset.unique()):
        vs = list(a.vs)
        if a.all_individuals:
            vs += [m for m in D[(D.dataset == ds) & (D.kind == "individual")].method.unique() if m not in vs]
        for c in a.consensus:
            for v in vs:
                if v == c:
                    continue
                r = image_level(D, ds, c, v)
                if r is None:
                    continue
                d, mc, mv, ncase = r
                try:
                    p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
                except ValueError:
                    p = np.nan
                lo, hi = boot_ci(d)
                rows.append(dict(dataset=ds, consensus=c, vs=v, n_cases=ncase, n_images=len(d), dice_consensus=mc, dice_vs=mv,
                                 diff=float(d.mean()), ci_lo=lo, ci_hi=hi, win_rate=float((d > 0).mean()), p_wilcoxon=p))
    S = pd.DataFrame(rows)
    S["p_holm"] = np.nan
    for (ds, c), g in S.groupby(["dataset", "consensus"]):
        S.loc[g.index, "p_holm"] = holm(g.p_wilcoxon.fillna(1.0))
    S.to_csv(os.path.join(a.out_dir, "paired_image_level.csv"), index=False)
    pd.set_option("display.width", 220)
    for ds in S.dataset.unique():
        print("\n===== %s : consensus - comparator, image-level (Dice) =====" % ds)
        t = S[S.dataset == ds][["consensus", "vs", "n_images", "dice_consensus", "dice_vs", "diff", "ci_lo", "ci_hi", "win_rate", "p_wilcoxon", "p_holm"]]
        print(t.round(4).to_string(index=False))
    print("\nsaved ->", os.path.join(a.out_dir, "paired_image_level.csv"))


if __name__ == "__main__":
    main()
