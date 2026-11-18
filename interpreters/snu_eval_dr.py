"""
snu_eval_dr.py — evaluates DR-clean-format models on the SNU dataset (shared by main and mild).

Inherits the metric logic of 3_evaluate.py + DR-clean conventions (config.yml→build_encoder, ckpt_best.pth):
  - model build: config.yml → build_encoder (same as gen/saliency)
  - threshold: ckpt_best.pth["val_thresholds"][0] (fixed Youden threshold from training validation = avoids data leakage)
  - metrics: AUC + Sensitivity/Specificity/PPV/NPV/F1/Accuracy, bootstrap 95% CI
  - ⚠️ import only after adding code_root to sys.path (trainers/models modules)

SNU CSV: dataset, image_path, label[, grade]

Usage (main models, 40 positives = referable ≥1/3):
  python interpreters/snu_eval_dr.py \
    --dr_root models/dr_clean --aux_root models/ckpts_aux \
    --snu_csv $SNUH_EVAL_CSV \
    --code_root . --out_dir outputs/snu_eval_main

Usage (mild models, 97 positives = any-DR ≥1/3):
  python interpreters/snu_eval_dr.py \
    --dr_root models/dr_clean_mild --aux_root models/ckpts_aux_mild \
    --snu_csv $SNUH_EVAL_ANYDR_CSV \
    --code_root . --out_dir outputs/snu_eval_mild
"""
import sys, os, glob, argparse, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torchvision.transforms as tfms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

IMG_SIZE = 448
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════ metrics (same as 3_evaluate.py) ═══════════════
def _threshold_metrics(y_true, y_pred, thr):
    pb = (y_pred >= thr).astype(int)
    tp = ((pb == 1) & (y_true == 1)).sum(); tn = ((pb == 0) & (y_true == 0)).sum()
    fp = ((pb == 1) & (y_true == 0)).sum(); fn = ((pb == 0) & (y_true == 1)).sum()
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else np.nan
    sen = tp / (tp + fn) if (tp + fn) else np.nan
    spe = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = (2 * ppv * sen / (ppv + sen)) if (ppv and sen and (ppv + sen)) else np.nan
    return acc, sen, spe, ppv, npv, f1


def _ci(a):
    a = [x for x in a if not np.isnan(x)]
    if len(a) < 10:
        return np.nan, np.nan
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def compute_metrics(preds, labels, thr, n_boot=2000, seed=42):
    labels = np.asarray(labels).astype(int); preds = np.asarray(preds).reshape(-1)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = np.nan
    rng = np.random.default_rng(seed); n = len(labels)
    ba, bs, bp = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            ba.append(roc_auc_score(labels[idx], preds[idx]))
        except ValueError:
            pass
        _, se, sp, _, _, _ = _threshold_metrics(labels[idx], preds[idx], thr)
        bs.append(se); bp.append(sp)
    acc, sen, spe, ppv, npv, f1 = _threshold_metrics(labels, preds, thr)
    al, ah = _ci(ba); sl, sh = _ci(bs); pl, ph = _ci(bp)
    fmt = lambda v, lo, hi: f"{round(v,3)} ({lo:.3f}-{hi:.3f})" if not np.isnan(lo) else round(v, 3)
    return {
        "snu_auc": fmt(auc, al, ah),
        "sensitivity": fmt(sen, sl, sh),
        "specificity": fmt(spe, pl, ph),
        "youden_thr": round(float(thr), 6),
        "accuracy": round(float(acc), 4),
        "ppv": round(float(ppv), 4) if not np.isnan(ppv) else np.nan,
        "npv": round(float(npv), 4) if not np.isnan(npv) else np.nan,
        "f1": round(float(f1), 4),
        "auc_raw": round(float(auc), 4) if not np.isnan(auc) else np.nan,
        "n": int(n), "n_pos": int(labels.sum()),
    }


# ═══════════════ Dataset ═══════════════
class SNUEval(Dataset):
    def __init__(self, csv, img_sz=IMG_SIZE):
        df = pd.read_csv(csv, low_memory=False).reset_index(drop=True)
        self.df = df
        self.paths = df["image_path"].astype(str).values
        self.labels = df["label"].values.astype("float32")
        self.datasets = df["dataset"].values if "dataset" in df.columns else None
        self.resize = tfms.Resize((img_sz, img_sz))
        self.norm = tfms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        im = self.resize(Image.open(self.paths[i]).convert("RGB"))
        x = torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)
        return self.norm(x), torch.tensor(self.labels[i])


@torch.no_grad()
def infer(model, loader):
    ps, ls = [], []
    for x, y in tqdm(loader, desc="SNU eval", leave=False):
        out = model(x.to(dev))
        if isinstance(out, tuple):
            out = out[0]                       # DR head (aux grade head separated)
        ps.append(torch.sigmoid(out[:, 0]).cpu().numpy())
        ls.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ls)


def _strip_ts(name):
    return re.sub(r"_\d{8}_\d{6}$", "", name)


def discover(dr_root, aux_root, exclude):
    """Folders under dr_root + aux_root containing ckpt_best.pth + config.yml (names with the timestamp stripped)."""
    dirs = []
    for r in [dr_root, aux_root]:
        if r and os.path.isdir(r):
            dirs += glob.glob(os.path.join(r, "*"))
    out = []
    for d in sorted(set(dirs)):
        if not os.path.isdir(d):
            continue
        if not (os.path.exists(os.path.join(d, "ckpt_best.pth")) and
                os.path.exists(os.path.join(d, "config.yml"))):
            continue
        name = _strip_ts(os.path.basename(d))
        if any(ex in name for ex in exclude):
            print(f"  [excluded] {name}")
            continue
        out.append((name, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr_root", required=True)
    ap.add_argument("--aux_root", default="")
    ap.add_argument("--snu_csv", required=True)
    ap.add_argument("--code_root", required=True, help="DR-clean code root (for importing build_encoder)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--pretrained_root", default="",
                    help="override pretrained_root in the config (A100 absolute path → SNU weights path). "
                         "e.g. $PRETRAINED_ROOT")
    ap.add_argument("--dinov3_repo", default="",
                    help="override dinov3_repo (hub source) in the config (A100→SNU). "
                         "e.g. $DINOV3_REPO")
    a = ap.parse_args()

    # resolve data/model paths to absolute paths before chdir(code_root)
    a.dr_root = os.path.abspath(a.dr_root)
    a.aux_root = os.path.abspath(a.aux_root) if a.aux_root else ""
    a.snu_csv = os.path.abspath(a.snu_csv)
    a.out_dir = os.path.abspath(a.out_dir)
    a.pretrained_root = os.path.abspath(a.pretrained_root) if a.pretrained_root else ""
    a.dinov3_repo = os.path.abspath(a.dinov3_repo) if a.dinov3_repo else ""

    # ⚠️ import trainers/models only "after" adding code_root to sys.path
    code_root = os.path.abspath(a.code_root)
    if not os.path.isdir(os.path.join(code_root, "trainers")):
        sys.exit(f"[abort] no trainers/ in code_root: {code_root}\n"
                 f"      → set --code_root to the DR-clean code root (e.g. DR-clean)")
    sys.path.insert(0, code_root)
    os.chdir(code_root)
    from trainers.utils import load_yaml, dict2namespace
    from models import build_encoder
    from models.utils import unfreeze

    os.makedirs(a.out_dir, exist_ok=True)
    ds = SNUEval(a.snu_csv)
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers,
                        shuffle=False, pin_memory=True)
    print(f"SNU: {len(ds)} images (pos {int(ds.labels.sum())}) | CSV={a.snu_csv}")

    models = discover(a.dr_root, a.aux_root, a.exclude)
    print(f"Evaluating {len(models)} models")

    rows, pd_rows = [], []
    for name, mdir in models:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        try:
            cfg = dict2namespace(load_yaml(os.path.join(mdir, "config.yml")))
            if a.pretrained_root and hasattr(cfg, "pretrained_root"):
                cfg.pretrained_root = a.pretrained_root   # A100 → SNU weights path
            if a.dinov3_repo and hasattr(cfg, "dinov3_repo"):
                cfg.dinov3_repo = a.dinov3_repo           # A100 → SNU dinov3 hub source
            model = build_encoder(cfg); unfreeze(model); model = model.to(dev).eval()
            ck = torch.load(os.path.join(mdir, "ckpt_best.pth"), map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"], strict=False)
            thr = float(ck.get("val_thresholds", {0: 0.5}).get(0, 0.5))
        except Exception as e:
            import traceback; print(f"  [ERROR] load: {e}"); traceback.print_exc(); continue

        preds, labels = infer(model, loader)
        od = os.path.join(a.out_dir, name); os.makedirs(od, exist_ok=True)
        save = {"idx": np.arange(len(preds)), "label": labels, "pred": preds}
        if ds.datasets is not None:
            save["dataset"] = ds.datasets
        pd.DataFrame(save).to_csv(os.path.join(od, "snu_preds.csv"), index=False)

        m = compute_metrics(preds, labels, thr)
        rows.append({"model": name, **m})
        for k, v in m.items():
            print(f"    {k}: {v}")

        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "snu_results.csv"), index=False)
        del model; torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  SNU RESULTS\n{'='*60}")
    if rows:
        print(pd.DataFrame(rows)[["model", "snu_auc", "sensitivity", "specificity", "n", "n_pos"]].to_string(index=False))
    print("Saved:", a.out_dir, "(snu_results.csv, {model}/snu_preds.csv)")


if __name__ == "__main__":
    main()
