
from __future__ import annotations
import torch
import numpy as np

def mae_per_level(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Return mean absolute error per level (normalized units). pred/target: (B,10)."""
    pred = pred.view(-1,5,2)
    target = target.view(-1,5,2)
    ae = (pred - target).abs()
    return ae.mean(dim=0).mean(dim=1).detach().cpu().numpy()  # (5,)

def overall_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean().item())
