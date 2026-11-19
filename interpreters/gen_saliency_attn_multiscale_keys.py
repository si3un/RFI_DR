"""
gen_saliency_attn_multiscale.py — DR external TP **multi-scale(SESS) attention-rollout** saliency generation.

Multi-scale version of gen_saliency_attn.py (1.0x single scale).
Data loading (external TP + M0 mapping) is **exactly the same** as gen_saliency_attn.py,
and only the CAM computation is replaced by SESS multi-scale.

SESS procedure (same as the RFI original):
  ① at each scale s, upscale the input to s*448
  ② 448 sliding window (stride = stride_frac * 448)
  ③ per-window target-class score σ(logit) as the fusion weight
  ④ window saliency is min-max normalized per window before fusion
  ⑤ map back to original coordinates, then weighted average excluding zero-pad (Σ s_p·Â_p / Σ s_p)
  ⑥ mean over scales → FOV → final min-max ([0,1], same as the single-scale storage format)

Number of windows per scale (stride_frac=0.5):
  1.0x → 448,  1 window
  1.5x → 672,  starts {0,224}       → 2x2 =  4
  2.0x → 896,  starts {0,224,448}   → 3x3 =  9
  i.e. --scales 1.0 1.5 means 5 passes per image; 1.0 1.5 2.0 means 14 passes.
  (RFI's *_saliency_ms8 was made with 1.0+1.5 = 5 passes)

Output format is the same as gen_saliency_attn.py: {out_root}/{model}/saliency_map.pth [N,448,448] + used.csv
→ existing code such as consensus_overlap / ext_consensus_similarity reads it as is.

Usage (generating only the 14 AUROC>=0.90 models used as consensus inputs is sufficient):
  CUDA_VISIBLE_DEVICES=0 python interpreters/gen_saliency_attn_multiscale.py \
      --dr_root  $OUTPUT_ROOT/dr_clean \
      --aux_root $OUTPUT_ROOT/ckpts_aux \
      --m0_dir   $AUTOMORPH_M0_DIR \
      --map_csv  $AUTOMORPH_MAP_CSV \
      --out_root $OUTPUT_ROOT/saliency_attn_ms \
      --scales 1.0 1.5 --stride_frac 0.5 --auc_min 0.90
"""
import sys, os, math, glob, argparse, warnings, gc, json
warnings.filterwarnings("ignore")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch, numpy as np, pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

from trainers.utils import load_yaml, dict2namespace
from models import build_encoder
from models.utils import unfreeze

IMG_SIZE = 448
EPS = 1e-8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = []          # filled from --datasets (empty list = all)
EXCL_DATASETS = []     # --exclude_datasets

_pre = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
_nrm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def make_fov(size=IMG_SIZE):
    y, x = np.ogrid[:size, :size]
    c, r = size / 2, size * 0.49
    return ((y - c) ** 2 + (x - c) ** 2 <= r ** 2)
FOV_B = torch.from_numpy(make_fov()).bool().to(dev)
FOV_F = FOV_B.float()


# ── attention capture/rollout: reuses the same implementation as gen_saliency_attn.py ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_saliency_attn import (patch_attention_to_capture, _grid_from_N,
                               find_vit_backbone, discover_models, _strip_ts)


class RolloutMS:
    """Same rollout as gen_saliency_attn.ViTAttnRollout, but
    returns the raw relu cam without FOV/min-max together with the class score
    (needed for per-window min-max and score weighting in multi-scale fusion)."""

    def __init__(self, model, vit, mode="grad_rollout", class_idx=0):
        self.model = model.eval(); self.vit = vit
        self.mode = mode; self.class_idx = class_idx
        patch_attention_to_capture(vit)

    @staticmethod
    def _norm(M):
        M = M / (M.sum(-1, keepdim=True) + 1e-6)
        T = M.size(-1)
        M = M + torch.eye(T, device=M.device, dtype=M.dtype)
        return M / (M.sum(-1, keepdim=True) + 1e-6)

    def one(self, x):
        """x:[1,3,448,448] normalized input → (cam[448,448] relu, class_score=σ(logit))."""
        self.model.zero_grad(set_to_none=True)
        ug = (self.mode == "grad_rollout")
        if ug:
            x = x.detach().requires_grad_(True)
        with torch.set_grad_enabled(ug):
            out = self.model(x)
            if isinstance(out, tuple):
                out = out[0]
        logit = out.reshape(-1)[self.class_idx] if out.ndim == 1 else out[0, self.class_idx]
        score = float(torch.sigmoid(logit.detach()))
        if ug:
            logit.backward()
        rollout = None
        for blk in self.vit.blocks:
            A = getattr(blk.attn, "_last_attn", None)
            if A is None:
                continue
            a = A[0].mean(0)
            M = (a * A.grad[0].mean(0)).clamp(min=0) if (ug and A.grad is not None) else a.clamp(min=0)
            M = self._norm(M)
            rollout = M if rollout is None else torch.matmul(rollout, M)
        N = rollout.size(-1); prefix, s = _grid_from_N(N)
        cam = rollout[0, prefix:prefix + s * s].reshape(s, s)
        cam = F.interpolate(cam[None, None].float(), (IMG_SIZE, IMG_SIZE),
                            mode="bicubic", align_corners=False).squeeze()
        return F.relu(cam).detach(), score


def minmax_t(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + EPS) if mx > mn else a * 0.0


def multiscale_sess(engine, x448, scales, stride_frac, prefilter):
    """SESS-faithful multi-scale fusion. return 448 min-max saliency (cpu numpy [0,1]·FOV)."""
    x448 = x448.to(dev)
    acc = torch.zeros(IMG_SIZE, IMG_SIZE, device=dev); nsc = 0
    for s in scales:
        S = int(round(s * IMG_SIZE))
        up = x448 if S == IMG_SIZE else F.interpolate(
            x448.unsqueeze(0), (S, S), mode="bilinear", align_corners=False)[0]
        stride = max(1, int(IMG_SIZE * stride_frac))
        starts = sorted(set(list(range(0, S - IMG_SIZE + 1, stride)) + [max(0, S - IMG_SIZE)]))
        canvas = torch.zeros(S, S, device=dev); wsum = torch.zeros(S, S, device=dev)
        used = 0
        for y in starts:
            for x in starts:
                patch = up[:, y:y + IMG_SIZE, x:x + IMG_SIZE]
                cam, score = engine.one(_nrm(patch).unsqueeze(0))
                if score < prefilter:
                    continue
                a = minmax_t(cam)
                canvas[y:y + IMG_SIZE, x:x + IMG_SIZE] += score * a
                wsum[y:y + IMG_SIZE, x:x + IMG_SIZE] += score
                used += 1
        if used == 0:
            continue
        sm = canvas / wsum.clamp(min=EPS)
        if S != IMG_SIZE:
            sm = F.interpolate(sm[None, None], (IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False)[0, 0]
        acc += sm; nsc += 1
    ms = acc / max(nsc, 1)
    mf = ms * FOV_F
    mx = mf.max()
    mf = mf / mx if mx > EPS else mf
    return (mf * FOV_F).cpu().numpy().astype(np.float32)


def read_auc(mdir):
    mj = os.path.join(mdir, "test_metrics.json")
    if os.path.exists(mj):
        try:
            return float(json.load(open(mj)).get("0_auc", np.nan))
        except Exception:
            return np.nan
    return np.nan


def run_one(name, mdir, m0_dir, mp, out_root, mode, scales, stride_frac, prefilter):
    od = os.path.join(out_root, name)
    if os.path.exists(os.path.join(od, "saliency_map.pth")):
        print(f"  [skip] {name}: already done"); return
    model = engine = None
    try:
        cfg = dict2namespace(load_yaml(os.path.join(mdir, "config.yml")))
        model = build_encoder(cfg); unfreeze(model); model = model.to(dev).eval()
        ck = torch.load(os.path.join(mdir, "ckpt_best.pth"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        for p in model.parameters():
            p.requires_grad_(False)
        thr = float(ck.get("val_thresholds", {0: 0.5}).get(0, 0.5))

        vit = find_vit_backbone(model)
        engine = RolloutMS(model, vit, mode=mode, class_idx=0)

        # ── target images: external TP selection exactly as in gen_saliency_attn.py ──
        ext = pd.read_csv(os.path.join(mdir, "external_predictions.csv"))
        tp = ext[(ext["pred"] >= thr) & (ext["label"] == 1)].copy() if KEYS is None else ext.copy()
        tp = tp.merge(mp[["image_path", "unique_name", "group_id"]], on="image_path", how="inner")
        tp["key"] = tp["unique_name"].map(lambda u: os.path.splitext(str(u))[0])
        tp["m0_path"] = tp["key"].map(lambda k: os.path.join(m0_dir, k + ".png"))
        tp = tp[tp["m0_path"].map(os.path.exists)].reset_index(drop=True)
        if KEYS is not None:
            n0 = len(tp)
            tp = tp[tp["key"].isin(KEYS)].drop_duplicates("key").sort_values("key").reset_index(drop=True)
            print(f"  [keys_csv] {n0} -> {len(tp)} images (requested {len(KEYS)}, missing {len(KEYS) - len(tp)})")
        if DATASETS or EXCL_DATASETS:
            if "dataset" not in tp.columns:
                print(f"  [warning] {name}: no dataset column; dataset filter ignored")
            else:
                n0 = len(tp)
                if DATASETS:
                    tp = tp[tp["dataset"].astype(str).isin(DATASETS)]
                if EXCL_DATASETS:
                    tp = tp[~tp["dataset"].astype(str).isin(EXCL_DATASETS)]
                tp = tp.reset_index(drop=True)
                print(f"  [dataset filter] {n0} → {len(tp)} images")
        if len(tp) == 0:
            print(f"  [SKIP] {name}: 0 M0-mapped TP images"); return

        cams = torch.empty((len(tp), IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        os.makedirs(od, exist_ok=True)
        part = os.path.join(od, "partial.pth"); start = 0
        if os.path.exists(part):
            pk = torch.load(part, map_location="cpu")
            if pk.get("keys") == tp["key"].tolist():
                start = int(pk["n_done"]); cams[:start] = pk["cams"][:start]
                print(f"  [resume] {name}: {start}/{len(tp)}")
        for i, p in enumerate(tqdm(tp["m0_path"].tolist(), desc=f"{name}[SESS-MS]", leave=False)):
            if i < start:
                continue
            if i > start and i % SAVE_EVERY == 0:
                torch.save({"keys": tp["key"].tolist(), "n_done": i, "cams": cams[:i].clone()}, part)
            # multiscale_sess applies _nrm per window, so the tensor is passed **before** normalization.
            x = _pre(Image.open(p).convert("RGB"))
            cams[i] = torch.from_numpy(
                multiscale_sess(engine, x, scales, stride_frac, prefilter))

        os.makedirs(od, exist_ok=True)
        torch.save(cams, os.path.join(od, "saliency_map.pth"))
        if os.path.exists(part):
            os.remove(part)
        tp["unique_name"] = tp["key"]
        cols = ["unique_name", "image_path", "group_id"] + (["dataset"] if "dataset" in tp.columns else [])
        tp[cols].to_csv(os.path.join(od, "used.csv"), index=False)
        n_zero = int((cams.flatten(1).sum(1) <= EPS).sum())
        print(f"  [✓] {name}: N={len(tp)} (zero {n_zero})  → {od}")
    finally:
        del engine, model
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr_root", required=True)
    ap.add_argument("--aux_root", default="")
    ap.add_argument("--m0_dir", required=True)
    ap.add_argument("--map_csv", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--attn_mode", choices=["grad_rollout", "pure_rollout"], default="grad_rollout")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.5])
    ap.add_argument("--stride_frac", type=float, default=0.5)
    ap.add_argument("--prefilter", type=float, default=0.0)
    ap.add_argument("--exclude", nargs="*", default=["retfound_dinov2_partial_ft_2"])
    ap.add_argument("--only", nargs="*", default=[], help="only models whose name contains any of these strings")
    ap.add_argument("--auc_min", type=float, default=0.0,
                    help=">0 : only models whose 0_auc in test_metrics.json is >= this value "
                         "(0.90 to generate only the consensus inputs)")
    ap.add_argument("--ft", nargs="*", default=[],
                    help="fine-tuning type filter (Linear/Partial/LoRA). The RFI ms8 root "
                         "contains only Partial/LoRA, so use --ft Partial LoRA to match the Table 1 pool")
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="generate only these datasets (e.g. aptos messidor2 deepdrid). Empty = all")
    ap.add_argument("--exclude_datasets", nargs="*", default=[],
                    help="exclude these datasets (e.g. fgadr idrid — if already used in the GT analysis)")
    ap.add_argument("--only_exact", nargs="*", default=[], help="exact model folder names")
    ap.add_argument("--list_models", action="store_true", help="print the selected model names and exit")
    ap.add_argument("--keys_csv", default="", help="use exactly these images (column --keys_col) instead of the model's own TP set")
    ap.add_argument("--keys_col", default="image_id")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads per process (0 = default)")
    ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()
    KEYS = None
    if args.keys_csv:
        KEYS = set(os.path.splitext(str(k))[0] for k in pd.read_csv(args.keys_csv)[args.keys_col].astype(str).unique())
        print(f"[keys_csv] {len(KEYS)} keys from {args.keys_csv}")
    SAVE_EVERY = max(1, int(args.save_every))
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    DATASETS = [str(x) for x in args.datasets]
    EXCL_DATASETS = [str(x) for x in args.exclude_datasets]

    mp = pd.read_csv(args.map_csv)
    models = discover_models(args.dr_root, args.aux_root, args.exclude)
    if args.only:
        models = [(n, d) for n, d in models if any(o in n for o in args.only)]
    if args.only_exact:
        models = [(n, d) for n, d in models if n in set(args.only_exact)]
    if args.list_models:
        for n, d in models: print("MODEL", n)
        sys.exit(0)
    if args.ft:
        want = {str(x).lower() for x in args.ft}
        keep = []
        for n, d in models:
            f = ("lora" if "lora" in n.lower() else
                 "partial" if "partial" in n.lower() else
                 "linear" if "linear" in n.lower() else None)
            if f in want:
                keep.append((n, d))
            else:
                print(f"  [excluded] FT={f}: {n}")
        models = keep
    if args.auc_min > 0:
        keep = []
        for n, d in models:
            au = read_auc(d)
            if np.isfinite(au) and au >= args.auc_min:
                keep.append((n, d))
            else:
                print(f"  [excluded] AUC<{args.auc_min}: {n} (auc={au})")
        models = keep

    # Report the number of windows up front (for cost estimation)
    npass = 0
    for s in args.scales:
        S = int(round(s * IMG_SIZE)); st = max(1, int(IMG_SIZE * args.stride_frac))
        starts = sorted(set(list(range(0, S - IMG_SIZE + 1, st)) + [max(0, S - IMG_SIZE)]))
        npass += len(starts) ** 2
    print(f"target models {len(models)} | scales={args.scales} stride_frac={args.stride_frac} "
          f"→ {npass} passes per image | mode={args.attn_mode}")
    os.makedirs(args.out_root, exist_ok=True)
    for name, mdir in models:
        try:
            run_one(name, mdir, args.m0_dir, mp, args.out_root, args.attn_mode,
                    args.scales, args.stride_frac, args.prefilter)
        except Exception as e:
            import traceback; print(f"  [ERROR] {name}: {e}"); traceback.print_exc()
    print("\nDone. Saved to:", args.out_root)
