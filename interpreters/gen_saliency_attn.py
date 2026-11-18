"""
gen_saliency_attn.py — generates **attention-rollout** saliency for DR external TPs
(instead of the GradCAM in gen_saliency.py, uses the ViT attention rollout from the 2025/2026 notebooks)

Method (port of the notebook's ViTGradCamRollout):
  - capture each block's self-attention probabilities A[B,H,N,N] via a manual reimplementation (bypasses xformers → works for DINOv2/v3 too)
  - grad_rollout (default): backward from the positive logit → per-block M=relu(A·dScore/dA), mean over heads, matmul rollout across blocks
    pure_rollout: rollout with M=relu(A) only, no gradient (pure attention map)
  - CLS→patch row (rollout[0, prefix:]) → restore grid (register tokens auto-detected) → interpolate to 448
  - normalization/storage format same as gen_saliency.py: [0,1] min-max within the FOV (r=0.49), zero stays 0

→ the storage format matches the GradCAM version, so consensus_*/vessel_ratio code is reused as is.
Output (per model): {out_root}/{model}/saliency_map.pth [N,H,W] + used.csv

Usage:
  CUDA_VISIBLE_DEVICES=1 python interpreters/gen_saliency_attn.py \
      --dr_root  $OUTPUT_ROOT/dr_clean \
      --aux_root $OUTPUT_ROOT/ckpts_aux \
      --m0_dir   $AUTOMORPH_M0_DIR \
      --map_csv  $AUTOMORPH_MAP_CSV \
      --out_root $OUTPUT_ROOT/saliency_attn \
      --attn_mode grad_rollout \
      --exclude  retfound_dinov2_partial_ft_2
"""
import sys, os, math, glob, argparse, warnings, gc
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
BS = 4                    # attention is memory-heavy → keep small (I/O batch; computation is a per-image backward)
EPS = 1e-8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_pre = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
_nrm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def make_fov(size=IMG_SIZE):
    y, x = np.ogrid[:size, :size]
    c, r = size / 2, size * 0.49
    return ((y - c) ** 2 + (x - c) ** 2 <= r ** 2)
FOV = torch.from_numpy(make_fov()).bool().to(dev)


# ── replace each block.attn.forward with a manual MHSA → capture attn probabilities (+grad) ──
def patch_attention_to_capture(vit):
    for blk in vit.blocks:
        attn = blk.attn
        if getattr(attn, "_patched_capture", False):
            continue

        def wrapped_forward(x, _attn=attn, **kwargs):
            B, N, C = x.shape
            qkv = _attn.qkv(x)
            head_dim = C // _attn.num_heads
            qkv = qkv.reshape(B, N, 3, _attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if hasattr(_attn, "q_norm") and _attn.q_norm is not None:
                q = _attn.q_norm(q.transpose(2, 3)).transpose(2, 3)
            if hasattr(_attn, "k_norm") and _attn.k_norm is not None:
                k = _attn.k_norm(k.transpose(2, 3)).transpose(2, 3)
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))    # [B,H,N,N]
            attn_probs = attn_scores.softmax(dim=-1)
            _attn._last_attn = attn_probs
            try:
                _attn._last_attn.retain_grad()
            except Exception:
                pass
            p = attn_probs
            if callable(getattr(_attn, "attn_drop", None)):
                p = _attn.attn_drop(p)
            x_out = torch.matmul(p, v).transpose(1, 2).reshape(B, N, C)
            x_out = _attn.proj(x_out)
            if callable(getattr(_attn, "proj_drop", None)):
                x_out = _attn.proj_drop(x_out)
            return x_out

        attn.forward = wrapped_forward
        attn._patched_capture = True


def _grid_from_N(N):
    """auto-detect prefix (CLS+register) → (prefix, s) with s*s = N-prefix."""
    for e in (1, 2, 5, 6, 9, 0):
        s = int(round(math.sqrt(N - e)))
        if s * s == N - e:
            return e, s
    raise RuntimeError(f"failed to infer grid N={N}")


class ViTAttnRollout:
    def __init__(self, model, vit, mode="grad_rollout", class_idx=0):
        self.model = model.eval(); self.vit = vit
        self.mode = mode; self.class_idx = class_idx
        patch_attention_to_capture(vit)

    @staticmethod
    def _norm_attn(M):
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-6)
        T = M.size(-1)
        M = M + torch.eye(T, device=M.device, dtype=M.dtype)
        return M / (M.sum(dim=-1, keepdim=True) + 1e-6)

    def _one(self, x):  # x [1,3,H,W] → cam [H,W] cpu float32 ([0,1] FOV, zero→0)
        self.model.zero_grad(set_to_none=True)
        use_grad = (self.mode == "grad_rollout")
        if use_grad:
            x = x.detach().requires_grad_(True)   # only the input is a grad leaf → ensures grads for attention activations
        with torch.set_grad_enabled(use_grad):
            out = self.model(x)
            if isinstance(out, tuple): out = out[0]
        if use_grad:
            score = out.reshape(-1)[self.class_idx] if out.ndim == 1 else out[0, self.class_idx]
            score.backward()

        rollout = None
        for blk in self.vit.blocks:
            A = getattr(blk.attn, "_last_attn", None)
            if A is None: continue
            a = A[0].mean(0)                       # [N,N] mean over heads
            if use_grad and A.grad is not None:
                g = A.grad[0].mean(0)
                M = (a * g).clamp(min=0)
            else:
                M = a.clamp(min=0)
            M = self._norm_attn(M)
            rollout = M if rollout is None else torch.matmul(rollout, M)

        N = rollout.size(-1)
        prefix, s = _grid_from_N(N)
        cam = rollout[0, prefix:prefix + s * s].reshape(s, s)             # CLS→patch
        cam = F.interpolate(cam[None, None].float(), (IMG_SIZE, IMG_SIZE),
                            mode="bicubic", align_corners=False).squeeze()
        cam = F.relu(cam) * FOV.float()
        mx = cam.max()
        cam = cam / mx if mx > EPS else torch.zeros_like(cam)
        return (cam * FOV.float()).detach().cpu()

    def compute(self, x):  # x [B,3,H,W] → [B,H,W] (per-image backward)
        return torch.stack([self._one(x[i:i + 1]) for i in range(x.shape[0])], 0)


def find_vit_backbone(m):
    if hasattr(m, "blocks"): return m
    for _, c in m.named_children():
        r = find_vit_backbone(c)
        if r is not None: return r
    return None


def _strip_ts(name):
    import re
    return re.sub(r"_\d{8}_\d{6}$", "", name)


def discover_models(dr_root, aux_root, exclude):
    dirs = []
    for root in [dr_root, aux_root]:
        if root and os.path.isdir(root):
            dirs += glob.glob(os.path.join(root, "*"))
    out = []
    for d in sorted(set(dirs)):
        if not os.path.isdir(d): continue
        if not (os.path.exists(os.path.join(d, "ckpt_best.pth")) and
                os.path.exists(os.path.join(d, "external_predictions.csv"))): continue
        clean = _strip_ts(os.path.basename(d))
        if any(ex in clean for ex in exclude):
            print(f"  [excluded] {clean}"); continue
        out.append((clean, d))
    return out


def run_one(name, mdir, m0_dir, mp, out_root, mode):
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
            p.requires_grad_(False)   # do not store param grads (saves memory) — only attention activation grads are needed
        thr = float(ck.get("val_thresholds", {0: 0.5}).get(0, 0.5))

        vit = find_vit_backbone(model)
        engine = ViTAttnRollout(model, vit, mode=mode, class_idx=0)

        ext = pd.read_csv(os.path.join(mdir, "external_predictions.csv"))
        tp = ext[(ext["pred"] >= thr) & (ext["label"] == 1)].copy()
        tp = tp.merge(mp[["image_path", "unique_name", "group_id"]], on="image_path", how="inner")
        tp["key"] = tp["unique_name"].map(lambda u: os.path.splitext(str(u))[0])
        tp["m0_path"] = tp["key"].map(lambda k: os.path.join(m0_dir, k + ".png"))
        tp = tp[tp["m0_path"].map(os.path.exists)].reset_index(drop=True)
        if len(tp) == 0:
            print(f"  [SKIP] {name}: 0 TP images mapped by M0"); return

        cams = torch.empty((len(tp), IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        off = 0
        for s in tqdm(range(0, len(tp), BS), desc=f"{name}[{mode}]", leave=False):
            paths = tp["m0_path"].iloc[s:s + BS].tolist()
            x = torch.stack([_nrm(_pre(Image.open(p).convert("RGB"))) for p in paths]).to(dev)
            c = engine.compute(x)
            cams[off:off + c.shape[0]].copy_(c); off += c.shape[0]

        os.makedirs(od, exist_ok=True)
        torch.save(cams, os.path.join(od, "saliency_map.pth"))
        tp["unique_name"] = tp["key"]
        tp[["unique_name", "image_path", "group_id", "dataset"]].to_csv(
            os.path.join(od, "used.csv"), index=False)
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
    ap.add_argument("--exclude", nargs="*", default=["retfound_dinov2_partial_ft_2"])
    args = ap.parse_args()

    mp = pd.read_csv(args.map_csv)
    models = discover_models(args.dr_root, args.aux_root, args.exclude)
    print(f"Target models: {len(models)} | mode={args.attn_mode} | M0={args.m0_dir}")
    os.makedirs(args.out_root, exist_ok=True)
    for name, mdir in models:
        try:
            run_one(name, mdir, args.m0_dir, mp, args.out_root, args.attn_mode)
        except Exception as e:
            import traceback; print(f"  [ERROR] {name}: {e}"); traceback.print_exc()
    print("\nDone. Saved to:", args.out_root)
