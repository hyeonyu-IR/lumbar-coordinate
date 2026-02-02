
from __future__ import annotations
from pathlib import Path
import json, random, os
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_config_snapshot(run_dir: Path, cfg_obj) -> Path:
    path = run_dir / 'config.json'
    if hasattr(cfg_obj, '__dict__'):
        data = cfg_obj.__dict__
    else:
        data = dict(cfg_obj)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
