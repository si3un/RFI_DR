#!/usr/bin/env python3
"""
Script that downloads OpenCLIP pretrained weights via timm and saves them to pretrained_root
in the format that this repository's models/openclip.py reads directly (plain state_dict .pth).

models/openclip.py does not download weights itself; it reads a local .pth file from pretrained_root.
Fetch the weights once with this script (no internet access is needed afterwards).

Usage:
  python scripts/download_openclip_weights.py --pretrained-root $PRETRAINED_ROOT
  # default: vit_large_patch14_clip_224.laion2b (the OpenCLIP backbone of the paper)
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


if __name__ == '__main__':
    main()
