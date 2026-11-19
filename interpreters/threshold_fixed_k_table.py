"""
threshold_fixed_k_table.py - localization with a representative fixed threshold (supplementary analysis; also
provides the dataset order and labels used by figures/make_figure3.py).

Rebuilds the table by reading only the precomputed {ds}_fixed_k_performance.csv. No re-sweep.
The Mean-k rounding rule, the .5 handling of Median-k and the Max-k integer rule are those of threshold_optimal_k.py.

Contents:
  · Mean / Median / Max of the per-case optimal k for each dataset x method, used as fixed thresholds
  · Dice / IoU when that k is applied uniformly to all cases of the dataset
  · per-case oracle shown alongside as reference

Model order is fixed by backbone x fine-tuning:
  OpenCLIP / DINOv2 / RETFound-DINOv2 / DINOv3-Large / DINOv3-Huge / MAE / RETFound
  each as Classifier only -> Partial fine-tuning -> LoRA tuning,  with Consensus Map last.

Outputs:
  fixed_representative_k.csv        main table (ALL)
  fixed_representative_k.md         Markdown version of the table
  consensus_fixed_k_summary.csv            Consensus-only summary (+ oracle reference)
  fixed_representative_k_by_lesion.csv   appendix (saved only, not printed)

Usage:
  python interpreters/threshold_fixed_k_table.py --out_dir outputs/threshold_analysis
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd

DS_ORDER = ["eophtha", "idrid", "fgadr"]
DS_LABEL = {"eophtha": "E-Ophtha", "idrid": "IDRiD", "fgadr": "FGADR"}
LT_ORDER = ["MA", "HE", "EX", "SE"]

# (display name, predicate) - more specific entries are checked first
BACKBONES = [
    ("OpenCLIP",        lambda s: ("clip" in s) or ("laion" in s)),
    ("DINOv2",          lambda s: ("dinov2" in s) and ("retfound" not in s)),
    ("RETFound-DINOv2", lambda s: ("retfound" in s) and ("dinov2" in s)),
    ("DINOv3-Large",    lambda s: ("dinov3" in s) and (("vitl" in s) or ("l16" in s)
                                                       or ("large" in s))),
    ("DINOv3-Huge",     lambda s: ("dinov3" in s) and (("vith" in s) or ("h16" in s)
                                                       or ("huge" in s) or ("plus" in s))),
    ("MAE",             lambda s: ("mae" in s) and ("retfound" not in s)),
    ("RETFound",        lambda s: ("retfound" in s) and ("dinov2" not in s)),
]
FTS = [("Classifier only", lambda s: not (("lora" in s) or ("partial" in s))),
       ("Partial fine-tuning", lambda s: "partial" in s),
       ("LoRA tuning", lambda s: "lora" in s)]


def classify(m):
    s = m.lower()
    if "hybridmeanq75" in s:
        return ("ZZ_Consensus", "")
    bb = next((n for n, f in BACKBONES if f(s)), None)
    ft = next((n for n, f in FTS if f(s)), None)
    return (bb, ft)


def cons_name(m):
    n = m.split("n=")[-1].rstrip(")") if "n=" in m else "?"
    return "Consensus Map (HybridMeanQ75, AUROC>=0.92, n=%s)" % n


def md_table(df, title, note=""):
    out = ["", "### %s" % title, ""]
    if note:
        out += [note, ""]
    out += ["| " + " | ".join(map(str, df.columns)) + " |",
            "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/threshold_analysis")
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.out_dir, "*_fixed_k_performance.csv")))
    if not fs:
        sys.exit("[abort] no *_fixed_k_performance.csv in %s.\n"
                 "  Run threshold_optimal_k.py first." % a.out_dir)
    F = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    need = ["k_mean", "dice_mean", "iou_mean", "k_median", "dice_median", "iou_median",
            "k_max", "dice_max", "iou_max", "oracle_dice", "oracle_iou"]
    miss = [c for c in need if c not in F.columns]
    if miss:
        sys.exit("[abort] missing columns %s - rerun the latest threshold_optimal_k.py." % miss)
    dss = [d for d in DS_ORDER if d in set(F["dataset"])] + \
          [d for d in sorted(set(F["dataset"])) if d not in DS_ORDER]

    meth = sorted(set(F["method"]))
    info = {}
    for m in meth:
        bb, ft = classify(m)
        info[m] = (bb, ft, cons_name(m) if bb == "ZZ_Consensus" else m)
    unmatched = [m for m in meth if info[m][0] is None or
                 (info[m][0] != "ZZ_Consensus" and info[m][1] is None)]
    bb_idx = {n: i for i, (n, _) in enumerate(BACKBONES)}
    bb_idx["ZZ_Consensus"] = len(BACKBONES)
    ft_idx = {n: i for i, (n, _) in enumerate(FTS)}
    ORD = sorted([m for m in meth if m not in unmatched],
                 key=lambda m: (bb_idx[info[m][0]], ft_idx.get(info[m][1], 0))) + unmatched

    print("dataset %s | %d methods" % ([DS_LABEL.get(d, d) for d in dss], len(meth)))
    if unmatched:
        print("  [note] %d methods not classified by backbone/fine-tuning - appended at the end of the table:"
              % len(unmatched))
        for m in unmatched:
            print("     %s" % m)

    def cell(v):
        return "-" if not np.isfinite(v) else "%.4f" % v

    def build(lt):
        rows = []
        for m in ORD:
            bb, ft, disp = info[m]
            rec = {"lesion_type": lt, "Backbone": "" if bb == "ZZ_Consensus" else (bb or "?"),
                   "Fine-tuning": ft or "", "Model": disp}
            for ds in dss:
                s = F[(F["dataset"] == ds) & (F["lesion_type"] == lt) & (F["method"] == m)]
                lb = DS_LABEL.get(ds, ds)
                if s.empty:
                    for tag in ("Mean", "Median", "Max"):
                        rec["%s %s-k" % (lb, tag)] = "-"
                        rec["%s %s-k Dice/IoU" % (lb, tag)] = "-"
                    rec["%s Oracle Dice/IoU" % lb] = "-"
                    continue
                r = s.iloc[0]
                for tag, key in (("Mean", "mean"), ("Median", "median"), ("Max", "max")):
                    rec["%s %s-k" % (lb, tag)] = "%g" % float(r["k_%s" % key])
                    rec["%s %s-k Dice/IoU" % (lb, tag)] = "%s / %s" % (
                        cell(r["dice_%s" % key]), cell(r["iou_%s" % key]))
                rec["%s Oracle Dice/IoU" % lb] = "%s / %s" % (
                    cell(r["oracle_dice"]), cell(r["oracle_iou"]))
            rows.append(rec)
        return pd.DataFrame(rows)

    T = build("ALL")
    T.drop(columns=["lesion_type"]).to_csv(
        os.path.join(a.out_dir, "fixed_representative_k.csv"), index=False)
    APP = pd.concat([build(lt) for lt in
                     [t for t in LT_ORDER if t in set(F["lesion_type"])]], ignore_index=True)
    APP.to_csv(os.path.join(a.out_dir,
                            "fixed_representative_k_by_lesion.csv"), index=False)

    KCOL = ["Model"] + ["%s %s-k" % (DS_LABEL.get(d, d), t)
                        for d in dss for t in ("Mean", "Median", "Max")]
    PCOL = ["Model"] + ["%s %s-k Dice/IoU" % (DS_LABEL.get(d, d), t)
                        for d in dss for t in ("Mean", "Median", "Max")]
    W = 200
    print("\n" + "=" * W)
    print("Representative fixed threshold values (Mean / Median / Max of the per-case optimal k, %)  [ALL]")
    print("=" * W)
    print(T[KCOL].to_string(index=False))
    print("\n" + "=" * W)
    print("Performance with the representative fixed threshold - Dice / IoU  [ALL case-item]")
    print("=" * W)
    print(T[PCOL].to_string(index=False))
    print("\n" + "=" * W)
    print("Per-case oracle (reference)  [ALL]")
    print("=" * W)
    print(T[["Model"] + ["%s Oracle Dice/IoU" % DS_LABEL.get(d, d) for d in dss]]
          .to_string(index=False))

    # ── 4/5. Consensus summary + oracle reference ──
    cm = [m for m in meth if classify(m)[0] == "ZZ_Consensus"]
    crows = []
    if cm:
        for ds in dss:
            s = F[(F["dataset"] == ds) & (F["lesion_type"] == "ALL") & (F["method"] == cm[0])]
            if s.empty:
                continue
            r = s.iloc[0]
            crows.append({"Dataset": DS_LABEL.get(ds, ds), "N": int(r["N"]),
                          "Oracle Dice": round(float(r["oracle_dice"]), 4),
                          "Oracle IoU": round(float(r["oracle_iou"]), 4),
                          "Mean k": float(r["k_mean"]),
                          "Fixed Mean-k Dice": round(float(r["dice_mean"]), 4),
                          "Fixed Mean-k IoU": round(float(r["iou_mean"]), 4),
                          "Median k": float(r["k_median"]),
                          "Fixed Median-k Dice": round(float(r["dice_median"]), 4),
                          "Fixed Median-k IoU": round(float(r["iou_median"]), 4),
                          "Max k": float(r["k_max"]),
                          "Fixed Max-k Dice": round(float(r["dice_max"]), 4),
                          "Fixed Max-k IoU": round(float(r["iou_max"]), 4),
                          "Ret Mean%": round(100 * float(r["dice_mean"]) / float(r["oracle_dice"]), 1),
                          "Ret Median%": round(100 * float(r["dice_median"]) / float(r["oracle_dice"]), 1),
                          "Ret Max%": round(100 * float(r["dice_max"]) / float(r["oracle_dice"]), 1)})
        Cn = pd.DataFrame(crows)
        Cn.to_csv(os.path.join(a.out_dir, "consensus_fixed_k_summary.csv"), index=False)
        print("\n" + "=" * W)
        print("Consensus Map summary  [ALL]")
        print("=" * W)
        print(Cn[["Dataset", "N", "Mean k", "Fixed Mean-k Dice", "Fixed Mean-k IoU",
                  "Median k", "Fixed Median-k Dice", "Fixed Median-k IoU",
                  "Max k", "Fixed Max-k Dice", "Fixed Max-k IoU"]].to_string(index=False))
        print("\n" + "=" * W)
        print("Consensus Map - relative to oracle (retention = fixed / oracle x 100)")
        print("=" * W)
        print(Cn[["Dataset", "Oracle Dice", "Oracle IoU", "Fixed Mean-k Dice", "Ret Mean%",
                  "Fixed Median-k Dice", "Ret Median%", "Fixed Max-k Dice",
                  "Ret Max%"]].to_string(index=False))
    else:
        Cn = pd.DataFrame()
        print("\n[note] Consensus (HybridMeanQ75) row not found.")

    md = ["# Representative fixed top-k threshold evaluation", "",
          "case = (image x lesion type). Representative k = Mean / Median / Max of the "
          "per-case optimal top-k, applied as a single fixed threshold to every case of "
          "that dataset. ALL = pooled case-items."]
    md.append(md_table(T[KCOL], "Representative threshold values (%)"))
    md.append(md_table(T[PCOL], "Dice / IoU at representative fixed thresholds"))
    md.append(md_table(T[["Model"] + ["%s Oracle Dice/IoU" % DS_LABEL.get(d, d) for d in dss]],
                       "Per-case oracle (reference)"))
    if len(Cn):
        md.append(md_table(Cn, "Consensus Map summary"))
    with open(os.path.join(a.out_dir, "fixed_representative_k.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\nSaved:")
    for n in ("fixed_representative_k.csv", "fixed_representative_k.md",
              "consensus_fixed_k_summary.csv", "fixed_representative_k_by_lesion.csv"):
        print("  %s" % os.path.join(a.out_dir, n))
    print("  (per-lesion results are saved to the appendix only and omitted from the main output)")


if __name__ == "__main__":
    main()
