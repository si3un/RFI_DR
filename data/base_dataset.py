"""
data/base_dataset.py - CSV-driven fundus image dataset for DR classification.

CSV contract:
  - image path column : cfg.data.input_label   (default 'image_path', absolute or relative to data_root)
  - label column (binary) : cfg.data.target_label  (default 'label', 0/1 -> float, BCEWithLogitsLoss)
  - grade column (aux)  : cfg.data.grade_label   (default 'grade', 0-4 int, only when cfg.use_aux=True)
  - split            : train/valid/test bool columns (filtered by column == True)
  - external         : all rows of a separate CSV (cfg.data.external_meta_path) (external validation set)

Returns:
  - use_aux=False : (img, label)
  - use_aux=True  : (img, label, grade)   ← train/valid/test
  - split='external' : always (img, label) (grade not required)
"""
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from trainers.utils import cfg_get


class BaseDataset(Dataset):
    def __init__(self, cfg, split='train', force_eval_tfms=False, use_aux=False):
        assert split in ['train', 'valid', 'test', 'external']

        self.input_label  = cfg_get(cfg, 'input_label', 'image_path')
        self.target_label = cfg_get(cfg, 'target_label', 'label')
        self.grade_label  = cfg_get(cfg, 'grade_label', 'grade')
        self.data_root    = cfg_get(cfg, 'data_root', None)
        # The external split may have no grade, so aux is disabled
        self.use_aux = bool(use_aux) and split != 'external'

        if split == 'external':
            meta_path = cfg_get(cfg, 'external_meta_path', None)
            assert meta_path is not None, "split='external' but cfg.data.external_meta_path is not set"
            df = pd.read_csv(meta_path, low_memory=False)
            self.df = df.reset_index(drop=True)
        else:
            df = pd.read_csv(cfg.meta_path, low_memory=False)
            self.df = df[df[split] == True].copy().reset_index(drop=True)

        norm = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        if split == 'train' and not force_eval_tfms:
            self.tfms = transforms.Compose([
                transforms.RandomResizedCrop((cfg.img_size, cfg.img_size)),
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((cfg.img_size, cfg.img_size)),
                transforms.ToTensor(),
                norm,
            ])

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, rel_or_abs):
        if self.data_root:
            return Path(self.data_root) / rel_or_abs
        return Path(rel_or_abs)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self._resolve_path(row[self.input_label])
        with Image.open(img_path) as im:
            im.load()
            img = im.convert('RGB')
        img = self.tfms(img)

        label = torch.tensor(row[self.target_label], dtype=torch.float32)
        if self.use_aux:
            grade = torch.tensor(int(row[self.grade_label]), dtype=torch.long)
            return img, label, grade
        return img, label
