"""Small helpers shared by the analysis/ scripts (they only read result files; no images or models)."""
import re

import pandas as pd


# display names used in some result tables (e.g. 'DINOv2-LoRA') -> internal configuration names
BACKBONE = {"openclip": "vit_large_patch14_clip_224.laion2b", "dinov2": "dinov2_vitl14",
            "dinov3-large": "dinov3_vitl16", "dinov3-huge": "dinov3_vith16plus", "mae": "mae",
            "retfound": "retfound_mae", "retfound-dinov2": "retfound_dinov2"}
STRATEGY = {"lora": "lora", "partial": "partial_ft_2", "partial fine-tuning": "partial_ft_2",
            "linear": "linear", "classifier only": "linear", "classifier-only": "linear", "cls": "linear"}


def norm_name(name: str) -> str:
    """Internal configuration name, e.g.
    'dinov2_vitl14_lora_20260728_190535' -> 'dinov2_vitl14_lora';  'DINOv2-LoRA' -> 'dinov2_vitl14_lora';
    'DINOv3 (Huge) + Partial fine-tuning' -> 'dinov3_vith16plus_partial_ft_2'."""
    name = str(name).strip()
    name = re.sub(r"_20\d{6}_\d{6}$", "", name)
    name = name.rsplit("_2026", 1)[0]
    key = name.lower().replace(" (", "-").replace(")", "").replace(" with ", "-").replace(" + ", "-")
    for b in sorted(BACKBONE, key=len, reverse=True):          # longest backbone name first
        if key.startswith(b + "-"):
            s = key[len(b) + 1:].strip()
            if s in STRATEGY:
                return "%s_%s" % (BACKBONE[b], STRATEGY[s])
    return name


def strategy(name: str) -> str:
    n = norm_name(name)
    if n.endswith("_lora") or "_lora_" in n:
        return "LoRA"
    if "partial" in n:
        return "Partial fine-tuning"
    return "Classifier only"


def internal_auroc(bootstrap_csv: str) -> pd.Series:
    """Internal-test AUROC per configuration from interpreters/bootstrap_classification_ci.py output."""
    b = pd.read_csv(bootstrap_csv)
    b = b[b["split"] == "internal"].copy()
    b["model"] = b["model"].map(norm_name)
    return b.set_index("model")["auroc"]


def cosa_table(summary_csv: str, scope: str = "ALL") -> pd.DataFrame:
    """CoSA per configuration (mean Spearman and top-k overlap AUC against the full consensus) from
    interpreters/ext_consensus_similarity_loo.py output (model_similarity_summary.csv)."""
    s = pd.read_csv(summary_csv)
    s = s[s["scope"] == scope].copy()
    s["model"] = s["model"].map(norm_name)
    return s.set_index("model")
