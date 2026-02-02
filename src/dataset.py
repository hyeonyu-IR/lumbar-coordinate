
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class LumbarDataset(Dataset):
    """
    Dataset that groups rows by filename and returns an image tensor and a 10-dim target
    with (relative_x, relative_y) for each of 5 levels in fixed order.
    CSV is expected to contain: filename, level, relative_x, relative_y (and optionally source).
    Images are discovered under images_root by basename or relative path match.
    """
    def __init__(self, df: pd.DataFrame, images_root: Path, levels: List[str], transform=None, allow_missing=True):
        self.df = df.copy()
        self.images_root = Path(images_root)
        self.levels = list(levels)
        self.level2idx = {lvl:i for i,lvl in enumerate(self.levels)}
        self.transform = transform
        self.allow_missing = allow_missing
        self.image_map = self._index_images(self.images_root)
        self.samples = self._build_samples(self.df)
        if len(self.samples) == 0:
            raise RuntimeError('No samples constructed; check paths and CSV content.')

    def _index_images(self, root: Path) -> Dict[str, Path]:
        exts = {'.jpg','.jpeg','.png','.bmp'}
        mp: Dict[str, Path] = {}
        if not root.exists():
            return mp
        for r,_,files in os.walk(root):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    abs_p = Path(r)/f
                    rel = abs_p.relative_to(root).as_posix().lower()
                    mp[rel] = abs_p
                    mp[f.lower()] = abs_p
        return mp

    def _build_samples(self, df: pd.DataFrame):
        samples = []
        for fn, g in df.groupby('filename'):
            key = str(fn).strip().lower()
            base = Path(fn).name.lower()
            img_path = self.image_map.get(key) or self.image_map.get(base)
            if img_path is None:
                if self.allow_missing:
                    continue
                else:
                    raise FileNotFoundError(f'Image not found for {fn}')
            y = np.full((len(self.levels), 2), np.nan, dtype=np.float32)
            for _, row in g.iterrows():
                idx = self.level2idx.get(row['level'])
                if idx is not None:
                    y[idx,0] = float(row['relative_x'])
                    y[idx,1] = float(row['relative_y'])
            if np.isnan(y).any():
                # ensure all 5 levels present
                continue
            samples.append((fn, img_path, y.reshape(-1)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, img_path, target = self.samples[idx]
        im = Image.open(img_path).convert('RGB')
        if self.transform:
            im = self.transform(im)
        target = torch.from_numpy(target.copy())
        return im, target, fn
