from pathlib import Path
import torch
import timm
import loralib as lora
from .utils import freeze, unfreeze, sanity_check
from .classifier import ClassifierWithAux
from trainers.utils import cfg_get

openclip_available_models = [
    'vit_large_patch14_clip_224.laion2b',
    'vit_huge_patch14_clip_224.laion2b',
]

def openclip_default_gradcam_target_layer(model):
    num_layer = len(model.backbone.blocks)
    return f'backbone.blocks.{num_layer-1}.norm1'

def build_openclip_model(cfg):
    """OpenCLIP (timm) encoder + GAP pooling + (optional) aux grade head."""
    model = timm.create_model(
        cfg.enc_name,
        num_classes=0,
        pretrained=False,
        img_size=cfg.data.img_size,
    )
    feat_dim = model.num_features
    num_prefix = model.num_prefix_tokens

    freeze(model)

    ft = cfg.ft.method
    if ft in ('linear', 'fast_linear'):
        pass
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
                    inf, ouf, r=cfg.ft.lora_rank, lora_alpha=cfg.ft.lora_rank,
                    enable_lora=[True, False, True], merge_weights=False)
        lora.mark_only_lora_as_trainable(model)
        unfreeze(model.norm)

    ckpt = torch.load(Path(cfg.pretrained_root) / f'{cfg.enc_name}.pth',
                      map_location='cpu', weights_only=True)
    for hk in ('head.weight', 'head.bias'):
        ckpt.pop(hk, None)
    model.load_state_dict(ckpt, strict=False)

    model = ClassifierWithAux(
        model, feat_dim=feat_dim, num_classes=cfg.num_classes,
        pool=cfg_get(cfg, 'pool', 'token'),   # CLIP also defaults to CLS (same as opportunistic)
        num_prefix_tokens=num_prefix,
        use_aux=cfg_get(cfg, 'use_aux', False), n_grades=cfg_get(cfg, 'n_grades', 5))

    sanity_check(model)
    return model
