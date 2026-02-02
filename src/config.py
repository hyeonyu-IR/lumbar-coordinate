
"""
Project configuration and defaults.
- Data lives OUTSIDE this repo. Point to it via environment variable LUMBAR_DATA_ROOT
  or by editing DEFAULT_DATA_ROOT below.
- All paths use pathlib and are OS-independent.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import json

# ----- External data root (edit to your local path or set env var) -----
DEFAULT_DATA_ROOT = Path(os.getenv(
    'LUMBAR_DATA_ROOT',
    r'C:/Users/hyeon/Documents/miniconda_medimg_env/data/lumbar-coordinate'
))

@dataclass
class Paths:
    project_root: Path
    runs_dir: Path
    reports_dir: Path
    # External (read-only) data
    data_root: Path
    data_images_dir: Path  # e.g., data_root / 'data'
    coords_pretrain_csv: Path  # e.g., data_root / 'coords_pretrain.csv'
    coords_rsna_csv: Path      # e.g., data_root / 'coords_rsna_improved.csv'

    @staticmethod
    def build(project_root: Path, data_root: Path | None = None) -> 'Paths':
        project_root = project_root.resolve()
        data_root = (data_root or DEFAULT_DATA_ROOT).resolve()
        return Paths(
            project_root=project_root,
            runs_dir=project_root / 'runs',
            reports_dir=project_root / 'reports',
            data_root=data_root,
            data_images_dir=data_root / 'data',
            coords_pretrain_csv=data_root / 'coords_pretrain.csv',
            coords_rsna_csv=data_root / 'coords_rsna_improved.csv',
        )

@dataclass
class TrainConfig:
    # data & task
    levels: tuple = ("L1/L2","L2/L3","L3/L4","L4/L5","L5/S1")
    image_size: int = 320
    batch_size: int = 32
    num_workers: int = 2
    # optimization
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber_beta: float = 0.01
    # model
    backbone: str = 'resnet18'
    pretrained: bool = True
    out_dim: int = 10  # 5 levels * (x,y)
    # randomness
    seed: int = 42
    # scheduling
    scheduler: str = 'plateau'  # 'plateau' or 'cosine'
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    min_lr: float = 1e-6
    # early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    # ordering regularization (anatomical prior)
    order_reg_lambda: float = 0.1
    order_reg_margin: float = 0.0  # allow strict monotonicity when 0.0

    def to_json(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))


def make_run_dirs(paths: Paths) -> Path:
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M')
    run_dir = paths.runs_dir / run_id
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'figures').mkdir(parents=True, exist_ok=True)
    return run_dir
