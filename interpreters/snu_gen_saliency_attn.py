"""
snu_gen_saliency_attn.py — generates **attention-rollout** saliency for the annotated positives (21 trained configurations)

Same structure and storage format as snu_gen_saliency.py (GradCAM); only the saliency computation is
replaced with the ViT attention rollout from the 2025/2026 notebooks (same engine as gen_saliency_attn.py).
  - grad_rollout (default): attention rollout weighted by positive-logit gradients (class-discriminative)
  - pure_rollout    : pure attention rollout without gradients
Storage: [0,1] min-max within the FOV (r=0.49) (same as the GradCAM version) → used as is by snu_lesion_robust/snu_consensus_lesion.

Output (per model): {out_root}/{model}/saliency_map.pth [N,448,448] + used.csv(file_name,image_path,label[,dataset])

Usage:
  python interpreters/snu_gen_saliency_attn.py \
      --dr_root   models/dr_clean --aux_root models/ckpts_aux \
      --lesion_csv $SNUH_LESION_CSV \
      --img_dir   $SNUH_IMG_DIR \
      --code_root . \
      --out_root  outputs/snu_saliency_attn --attn_mode grad_rollout \
      --exclude   retfound_dinov2_partial_ft_2
"""
import sys, os, math, glob, re, argparse, warnings, gc
warnings.filterwarnings("ignore")
import torch, numpy as np, pandas as pd
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

IMG_SIZE = 448
EPS = 1e-8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVR_PRETRAINED = ""   # override pretrained_root in the mild config (another machine)
OVR_DINOV3 = ""       # override dinov3_repo in the mild config (another machine)

def _letterbox(im):
    """Preserve aspect ratio: pad the short side with black to a square (max side) → no distortion or loss (same space as GT and display)."""
    W, H = im.size; S = max(W, H)
    c = Image.new("RGB", (S, S), (0, 0, 0))
    c.paste(im, ((S - W) // 2, (S - H) // 2))
    return c


_pre = Compose([_letterbox, Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
_nrm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def make_fov(size=IMG_SIZE):
    y, x = np.ogrid[:size, :size]
    c, r = size / 2, size * 0.49
    return ((y - c) ** 2 + (x - c) ** 2 <= r ** 2)
FOV = torch.from_numpy(make_fov()).bool().to(dev)


# ── manual MHSA for capturing attention (bypasses xformers) ──
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
            attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))
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

    def _one(self, x):
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
            a = A[0].mean(0)
            if use_grad and A.grad is not None:
                M = (a * A.grad[0].mean(0)).clamp(min=0)
            else:
                M = a.clamp(min=0)
            M = self._norm_attn(M)
            rollout = M if rollout is None else torch.matmul(rollout, M)
        N = rollout.size(-1)
        prefix, s = _grid_from_N(N)
        cam = rollout[0, prefix:prefix + s * s].reshape(s, s)
        cam = F.interpolate(cam[None, None].float(), (IMG_SIZE, IMG_SIZE),
                            mode="bicubic", align_corners=False).squeeze()
        cam = F.relu(cam) * FOV.float()
        mx = cam.max()
        cam = cam / mx if mx > EPS else torch.zeros_like(cam)
        return (cam * FOV.float()).detach().cpu()

    def compute(self, x):
        return torch.stack([self._one(x[i:i + 1]) for i in range(x.shape[0])], 0)


def find_vit_backbone(m):
    if hasattr(m, "blocks"): return m
    for _, c in m.named_children():
        r = find_vit_backbone(c)
        if r is not None: return r
    return None


class SNUPosDS(Dataset):
    def __init__(self, df):
        self.paths = df["image_path"].astype(str).tolist()
        self.fnames = [os.path.basename(p) for p in self.paths]

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        return _nrm(_pre(Image.open(self.paths[i]).convert("RGB"))), i


def _strip_ts(name): return re.sub(r"_\d{8}_\d{6}$", "", name)


def discover(dr_root, aux_root, exclude):
    dirs = []
    for r in [dr_root, aux_root]:
        if r and os.path.isdir(r): dirs += glob.glob(os.path.join(r, "*"))
    out = []
    for d in sorted(set(dirs)):
        if not os.path.isdir(d): continue
        if not (os.path.exists(os.path.join(d, "ckpt_best.pth")) and
                os.path.exists(os.path.join(d, "config.yml"))): continue
        name = _strip_ts(os.path.basename(d))
        if any(ex in name for ex in exclude):
            print(f"  [excluded] {name}"); continue
        out.append((name, d))
    return out


def run_one(name, mdir, snu_pos, out_root, bs, nw, mode):
    od = os.path.join(out_root, name)
    if os.path.exists(os.path.join(od, "saliency_map.pth")):
        print(f"  [skip] {name}: already done")
        return
    from trainers.utils import load_yaml, dict2namespace
    from models import build_encoder
    from models.utils import unfreeze
    model = engine = None
    try:
        cfg = dict2namespace(load_yaml(os.path.join(mdir, "config.yml")))
        if OVR_PRETRAINED and hasattr(cfg, "pretrained_root"):
            cfg.pretrained_root = OVR_PRETRAINED     # weights path on this machine
        if OVR_DINOV3 and hasattr(cfg, "dinov3_repo"):
            cfg.dinov3_repo = OVR_DINOV3             # dinov3 hub source on this machine
        model = build_encoder(cfg); unfreeze(model); model = model.to(dev).eval()
        ck = torch.load(os.path.join(mdir, "ckpt_best.pth"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        for p in model.parameters():
            p.requires_grad_(False)   # do not store param grads (saves memory) — only attention activation grads are needed
        vit = find_vit_backbone(model)
        engine = ViTAttnRollout(model, vit, mode=mode, class_idx=0)

        ds = SNUPosDS(snu_pos)
        loader = DataLoader(ds, batch_size=bs, num_workers=nw, shuffle=False, pin_memory=True)
        cams = torch.empty((len(ds), IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        off = 0
        for x, _ in tqdm(loader, desc=f"{name}[{mode}]", leave=False):
            c = engine.compute(x.to(dev))
            cams[off:off + c.shape[0]].copy_(c); off += c.shape[0]

        os.makedirs(od, exist_ok=True)
        torch.save(cams, os.path.join(od, "saliency_map.pth"))
        used = snu_pos.copy()
        used["file_name"] = ds.fnames
        cols = ["file_name", "image_path", "label"] + (["dataset"] if "dataset" in used.columns else [])
        used[cols].to_csv(os.path.join(od, "used.csv"), index=False)
        n_zero = int((cams.flatten(1).sum(1) <= EPS).sum())
        print(f"  [✓] {name}: N={len(ds)} (zero {n_zero}) → {od}")
    finally:
        del engine, model
        gc.collect(); torch.cuda.empty_cache()


def build_targets(snu_csv, lesion_csvs, img_dir):
    rows = {}
    if snu_csv:
        df = pd.read_csv(snu_csv, low_memory=False)
        df = df[df["label"] == 1] if "label" in df.columns else df
        for _, r in df.iterrows():
            bn = os.path.basename(str(r["image_path"]))
            rows[bn] = {"file_name": bn, "image_path": str(r["image_path"]),
                        "label": int(r.get("label", 1)), "dataset": r.get("dataset", "snuh")}
    for lc in (lesion_csvs or []):
        les = pd.read_csv(lc)
        for fn in les["file_name"].astype(str).unique():
            if fn not in rows:
                rows[fn] = {"file_name": fn, "image_path": os.path.join(img_dir, fn),
                            "label": 1, "dataset": "snuh"}
    out = pd.DataFrame(list(rows.values()))
    out = out[out["image_path"].map(os.path.exists)].reset_index(drop=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr_root", required=True)
    ap.add_argument("--aux_root", default="")
    ap.add_argument("--snu_csv", default="")
    ap.add_argument("--lesion_csv", nargs="*", default=[])
    ap.add_argument("--img_dir", default=os.environ.get("SNUH_IMG_DIR", ""))
    ap.add_argument("--code_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--attn_mode", choices=["grad_rollout", "pure_rollout"], default="grad_rollout")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=["retfound_dinov2_partial_ft_2"])
    ap.add_argument("--pretrained_root", default="", help="override the weights path in the config (another machine)")
    ap.add_argument("--dinov3_repo", default="", help="override the dinov3 hub path in the config (another machine)")
    a = ap.parse_args()
    OVR_PRETRAINED = os.path.abspath(a.pretrained_root) if a.pretrained_root else ""
    OVR_DINOV3 = os.path.abspath(a.dinov3_repo) if a.dinov3_repo else ""

    a.dr_root = os.path.abspath(a.dr_root)
    a.aux_root = os.path.abspath(a.aux_root) if a.aux_root else ""
    a.snu_csv = os.path.abspath(a.snu_csv) if a.snu_csv else ""
    a.lesion_csv = [os.path.abspath(p) for p in a.lesion_csv]
    a.img_dir = os.path.abspath(a.img_dir)
    a.out_root = os.path.abspath(a.out_root)

    code_root = os.path.abspath(a.code_root)
    sys.path.insert(0, code_root); os.chdir(code_root)

    snu_pos = build_targets(a.snu_csv, a.lesion_csv, a.img_dir)
    if len(snu_pos) == 0:
        raise SystemExit("0 target images — check --snu_csv or --lesion_csv")
    print(f"Target images: {len(snu_pos)} | mode={a.attn_mode} | lesion={a.lesion_csv}")

    models = discover(a.dr_root, a.aux_root, a.exclude)
    print(f"Processing {len(models)} models")
    os.makedirs(a.out_root, exist_ok=True)
    for name, mdir in models:
        try:
            run_one(name, mdir, snu_pos, a.out_root, a.batch_size, a.num_workers, a.attn_mode)
        except Exception as e:
            import traceback; print(f"  [ERROR] {name}: {e}"); traceback.print_exc()
    print("\nDone:", a.out_root)
