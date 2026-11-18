#!/usr/bin/env python3
"""
Script that downloads OpenCLIP pretrained weights via timm and saves them to pretrained_root
in the format that this repository's models/openclip.py reads directly (plain state_dict .pth).

[Background] Unlike dinov2/rad_dino/chexagent, OpenCLIP does not call torch.hub/HuggingFace Hub
  directly in code (offline-first design, readme.md section 7.1); it only reads local .pth files in
  pretrained_root. So the weights must be fetched once in advance with this script
  (after that, this file alone is reused without internet access).

[Usage]
  python scripts/download_openclip_weights.py --pretrained-root examples/data/fm_weights
  # Default is vit_large_patch14_clip_224.laion2b (used by example 3 in readme.md section 8.1).
  # If you need a larger model:
  python scripts/download_openclip_weights.py --enc-name vit_huge_patch14_clip_224.laion2b \
      --pretrained-root examples/data/fm_weights
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Save OpenCLIP (timm) pretrained weights as a local .pth")
    ap.add_argument('--enc-name', default='vit_large_patch14_clip_224.laion2b',
                     help="one of openclip_available_models in models/openclip.py")
    ap.add_argument('--pretrained-root', required=True,
                     help="same path as pretrained_root in the config (saved here as <enc-name>.pth)")
    args = ap.parse_args()

    import timm
    import torch

    out_dir = Path(args.pretrained_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.enc_name}.pth'

    print(f'[download] Downloading {args.enc_name} pretrained weights from timm...')
    model = timm.create_model(args.enc_name, pretrained=True)
    torch.save(model.state_dict(), out_path)

    print(f"\nDone! {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  pretrained_root = {out_dir}")
    print(f"  enc_name        = {args.enc_name}")
    print("  (you can also use configs/examples/xray_openclip_linear.yml as-is)")


if __name__ == '__main__':
    main()
