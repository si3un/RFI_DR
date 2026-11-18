from pathlib import Path
import torch
import torch.nn as nn
import loralib as lora
from .utils import freeze, unfreeze
from .utils import sanity_check
from .classifier import ClassifierWithAux
from trainers.utils import cfg_get

dino_available_models = [
    # DINOv2 models
    'dinov2_vits14',
    'dinov2_vitb14',
    'dinov2_vitl14',
    'dinov2_vitg14',
    # DINOv3 models
    # need to download weights from https://github.com/facebookresearch/dinov3
    'dinov3_vits16',
    'dinov3_vitb16',
    'dinov3_vitl16',      # DR: dinov3_large (ViT-L/16)
    'dinov3_vith16plus',  # DR: dinov3 (ViT-H+/16, Huge) - our main model
    'dinov3_vit7b16',
]

def dino_default_gradcam_target_layer(model):
    # Wrapped as ClassifierWithAux(backbone=...), so paths are relative to backbone
    num_layer = len(model.backbone.blocks)
    return f'backbone.blocks.{num_layer-1}.norm1'

def build_dino_model(cfg):
    """DINOv2 / DINOv3 encoder + (GAP pooling) + (optional aux grade head).

    - Binary classification: cfg.num_classes = 1  -> head outputs 1 (BCEWithLogitsLoss)
    - linear/partial/lora all use GAP pooling (cfg.pool, default 'gap')
    - Weight file: cfg.pretrained_root/{cfg.enc_name}.pth
    """
    img_size = cfg.data.img_size

    if 'dinov2' in cfg.enc_name:
        local_repo = cfg_get(cfg, 'dinov2_local_repo', None)
        if local_repo:
            model = torch.hub.load(local_repo, cfg.enc_name, source='local', pretrained=False)
        else:
            cached_repo = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
            if cached_repo.is_dir():
                model = torch.hub.load(str(cached_repo), cfg.enc_name, source='local', pretrained=False)
            else:
                model = torch.hub.load('facebookresearch/dinov2:main', cfg.enc_name, pretrained=False)
    elif 'dinov3' in cfg.enc_name:
        # DR: pass img_size to the hub entrypoint for 448 training (pos embed interpolation).
        model = torch.hub.load(
            cfg.dinov3_repo,
            cfg.enc_name,
            source='local',
            pretrained=False,
            img_size=img_size,
        )

    feat_dim = model.embed_dim
    freeze(model)

    ft = cfg.ft.method
    if ft in ('linear', 'fast_linear'):
        pass  # freeze the whole backbone; only the wrapper's head below is trained

    elif ft == 'partial_ft':
        n_layers = cfg_get(cfg.ft, 'n_layers', 1)
        if n_layers == -1:
            unfreeze(model)
        else:
            unfreeze(model.norm)
            for i, blk in enumerate(reversed(list(model.blocks))):
                if i < n_layers:
                    unfreeze(blk)

    elif ft == 'lora':
        n_layers = cfg_get(cfg.ft, 'n_layers', -1)
        for i, blk in enumerate(reversed(list(model.blocks))):
            if n_layers == -1 or i < n_layers:
                inf = blk.attn.qkv.in_features
                ouf = blk.attn.qkv.out_features
                blk.attn.qkv = lora.MergedLinear(
                    inf, ouf,
                    r=cfg.ft.lora_rank,
                    lora_alpha=cfg.ft.lora_rank,
                    enable_lora=[True, False, True],
                    merge_weights=False
                )
        # Re-unfreeze the final norm "after" marking only lora parameters as trainable
        # (fixes a bug in the reference code: safe because the head is created later in the wrapper)
        lora.mark_only_lora_as_trainable(model)
        unfreeze(model.norm)

    # Load backbone weights (strict=False so pretrained qkv goes into MergedLinear.weight under lora)
    model.load_state_dict(
        torch.load(Path(cfg.pretrained_root) / f'{cfg.enc_name}.pth',
                   map_location='cpu', weights_only=True),
        strict=False
    )

    model = ClassifierWithAux(
        model,
        feat_dim=feat_dim,
        num_classes=cfg.num_classes,
        pool=cfg_get(cfg, 'pool', 'token'),   # DINO's CLS token is a good global representation -> default CLS (same as opportunistic)
        num_prefix_tokens=1,
        use_aux=cfg_get(cfg, 'use_aux', False),
        n_grades=cfg_get(cfg, 'n_grades', 5),
    )

    sanity_check(model)
    return model
