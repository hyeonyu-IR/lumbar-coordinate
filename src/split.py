
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def make_train_val_splits(pretrain_csv: Path, out_dir: Path, test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(pretrain_csv)
    # image-level: ensure 5 levels per image
    img = (df.groupby(['filename','source'])['level']
             .nunique().reset_index(name='n_levels'))
    img = img[img['n_levels']==5].reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(sss.split(img['filename'], img['source']))
    train_files = set(img.loc[train_idx,'filename'])
    val_files   = set(img.loc[val_idx,'filename'])

    df_train = df[df['filename'].isin(train_files)].copy()
    df_val   = df[df['filename'].isin(val_files)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir / 'train.csv', index=False)
    df_val.to_csv(out_dir / 'val.csv', index=False)
    return (out_dir / 'train.csv', out_dir / 'val.csv')
