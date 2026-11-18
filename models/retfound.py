from pathlib import Path
from functools import partial
import torch
import torch.nn as nn
import timm
import loralib as lora
from .utils import freeze, unfreeze, sanity_check, interpolate_pos_embed
from .classifier import ClassifierWithAux
from trainers.utils import cfg_get

retfound_available_models = [
    'retfound_mae',      # MAE pretraining (ViT-L/16)
    'retfound_dinov2',   # DINOv2 pretraining (ViT-L/14)
]

def retfound_default_gradcam_target_layer(model):
    num_layer = len(model.backbone.blocks)
    return f'backbone.blocks.{num_layer-1}.norm1'


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ViT for RETFound (MAE). forward_features returns the normed token sequence [B,N,D]."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x, attn_mask=None, is_causal=False):
        # Recent timm calls forward_features(x, attn_mask=, is_causal=) -> accept and ignore the arguments
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def RETFound_mae(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def build_retfound_model(cfg):
    """RETFound (MAE/DINOv2) encoder + GAP pooling + (optional) aux grade head.

    Note: retfound_mae is MAE-pretrained, so its CLS token is not a good global representation;
       using CLS in a linear probe makes external collapse to 0.57. Therefore
       cfg.pool='gap' (default) is used to average the patch tokens.
    """
    img_size = cfg.data.img_size

    if cfg.enc_name == 'retfound_mae':
        model = RETFound_mae(img_size=img_size, num_classes=0, global_pool='token')
        feat_dim = model.embed_dim
        num_prefix = 1
    elif cfg.enc_name == 'retfound_dinov2':
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                                  pretrained=False, img_size=img_size, num_classes=0)
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

    # Special handling of RETFound checkpoints (based on the original release format)
    ckpt = torch.load(Path(cfg.pretrained_root) / f'{cfg.enc_name}.pth',
                      map_location='cpu', weights_only=False)
    if cfg.enc_name == 'retfound_mae':
        ckpt = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    elif cfg.enc_name == 'retfound_dinov2':
        ckpt = ckpt['teacher'] if isinstance(ckpt, dict) and 'teacher' in ckpt else ckpt
    ckpt = {k.replace('backbone.', ''): v for k, v in ckpt.items()}
    ckpt = {k.replace('mlp.w12.', 'mlp.fc1.'): v for k, v in ckpt.items()}
    ckpt = {k.replace('mlp.w3.', 'mlp.fc2.'): v for k, v in ckpt.items()}
    for hk in ('head.weight', 'head.bias'):
        ckpt.pop(hk, None)
    interpolate_pos_embed(model, ckpt)
    model.load_state_dict(ckpt, strict=False)

    # GAP only for retfound_mae (MAE) (validated: external CLS 0.54 -> GAP 0.69); CLS for retfound_dinov2 (DINOv2)
    _default_pool = 'gap' if cfg.enc_name == 'retfound_mae' else 'token'
    model = ClassifierWithAux(
        model, feat_dim=feat_dim, num_classes=cfg.num_classes,
        pool=cfg_get(cfg, 'pool', _default_pool), num_prefix_tokens=num_prefix,
        use_aux=cfg_get(cfg, 'use_aux', False), n_grades=cfg_get(cfg, 'n_grades', 5))

    sanity_check(model)
    return model
