"""
models/classifier.py - shared classification wrapper.

A plain backbone builder would attach `model.head = nn.Linear(...)` directly and rely on the backbone's
own forward_head (pooling). This task, however, needs
  (1) an aux grade head (multi-task: referable BCE + grade CE)
  (2) GAP (global average pooling) instead of the CLS token for linear probing
      -> to fix the problem where frozen linear probes of the MAE family (retfound_mae, mae) using
        the CLS token collapse to 0.57 on external (= for MAE, CLS is not a good global representation),
        by using GAP instead.
To add both "consistently" to every backbone, it is cleaner to control pooling directly in a
shared wrapper than to rely on each backbone's differing forward_head.

Important - order of trainable (requires_grad) setup:
  Each builder wraps the backbone with this wrapper "after" all freeze/unfreeze/lora steps on the
  backbone are done. head/grade_head are created here, so they default to requires_grad=True, and
  no code freezes them again afterwards. (lora's mark_only_lora_as_trainable applies only to the
  backbone and is called before the wrapper is created, so it does not touch the head.)
"""

import torch
import torch.nn as nn


def pool_features(feats, pool: str, num_prefix_tokens: int = 1):
    """backbone.forward_features output -> pooled [B, D] vector.

    Supported forms:
      - dict (dinov2/v3 hub): uses the 'x_norm_clstoken' / 'x_norm_patchtokens' keys
      - [B, N, D] tensor (timm ViT): skips num_prefix_tokens (cls/register) tokens
      - [B, D] tensor: already pooled, returned as-is
    pool:
      - 'gap'   : mean of patch tokens (required for MAE-family linear probes; applied uniformly to all backbones)
      - 'token' : CLS token
    """
    if isinstance(feats, dict):
        if pool == 'token':
            return feats['x_norm_clstoken']
        # 'gap' - mean of patch tokens
        return feats['x_norm_patchtokens'].mean(dim=1)

    if feats.dim() == 3:                       # [B, N, D]
        if pool == 'token':
            return feats[:, 0, :]
        return feats[:, num_prefix_tokens:, :].mean(dim=1)  # 'gap'

    return feats                               # [B, D] - already pooled


class ClassifierWithAux(nn.Module):
    """Classifier that wraps a backbone with pooling + an (optional) aux grade head.

    Args:
        backbone : encoder with freeze/unfreeze/lora already applied
        feat_dim : embedding dimension (Large=1024, Huge=1280, etc.)
        num_classes : classification head outputs (binary is 1 -> BCEWithLogitsLoss)
        pool : 'gap' (default) or 'token'
        num_prefix_tokens : number of leading tokens to skip in a [B,N,D] tensor (cls+register)
        use_aux : if True, add grade_head (multi-task)
        n_grades : number of grade classes (default 5, DR 0-4)
    """

    def __init__(self, backbone, feat_dim, num_classes=1, pool='gap',
                 num_prefix_tokens=1, use_aux=False, n_grades=5):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.num_prefix_tokens = num_prefix_tokens
        self.use_aux = use_aux
        self.head = nn.Linear(feat_dim, num_classes)
        self.grade_head = nn.Linear(feat_dim, n_grades) if use_aux else None

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x, return_aux: bool = False):
        feats = self.forward_features(x)
        pooled = pool_features(feats, self.pool, self.num_prefix_tokens)
        out = self.head(pooled)
        if return_aux and self.use_aux:
            return out, self.grade_head(pooled)
        return out
