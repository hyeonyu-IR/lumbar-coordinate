
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

from .config import TrainConfig
from .train_utils import train_one_epoch, evaluate


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    for pg in optimizer.param_groups:
        return float(pg.get('lr', 0.0))
    return 0.0


def ordering_penalty(pred: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Anatomical prior: enforce non-decreasing order of y-coordinates from L1/2 -> L5/S1.
    pred: (B,10) normalized outputs. Interpret columns as (x1,y1,x2,y2,...,x5,y5) per sample.
    penalty = ReLU( y_i - y_{i+1} + margin ) averaged over batch and adjacent pairs.
    """
    B = pred.shape[0]
    y = pred.view(B, 5, 2)[:, :, 1]  # (B,5)
    diffs = y[:, :-1] - y[:, 1:] + margin  # (B,4)
    return torch.relu(diffs).mean()


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, value: float):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def fit(model, train_dl: DataLoader, val_dl: DataLoader, cfg: TrainConfig, run_dir: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.SmoothL1Loss(beta=cfg.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.plateau_factor, patience=cfg.plateau_patience,
            min_lr=cfg.min_lr, verbose=True
        )
        use_plateau = True
    elif cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
        use_plateau = False
    else:
        scheduler = None
        use_plateau = False

    es = EarlyStopping(patience=cfg.early_stop_patience, min_delta=cfg.early_stop_min_delta)

    metrics_rows = []
    best_val = float('inf')
    best_path = run_dir / 'checkpoints' / 'best.pt'

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        model.train()
        train_loss_meter = []
        for x, y, _ in train_dl:
            x = x.to(device)
            y = y.to(device).float()
            optimizer.zero_grad()
            out = model(x)
            base_loss = criterion(out, y)
            ord_pen = ordering_penalty(out, cfg.order_reg_margin) * cfg.order_reg_lambda
            loss = base_loss + ord_pen
            loss.backward()
            optimizer.step()
            train_loss_meter.append(float(loss.item()))
        tr_loss = float(np.mean(train_loss_meter)) if train_loss_meter else float('nan')

        # --- Validate ---
        model.eval()
        val_losses = []
        val_ord_pens = []
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(device)
                y = y.to(device).float()
                out = model(x)
                base = criterion(out, y)
                ordp = ordering_penalty(out, cfg.order_reg_margin) * cfg.order_reg_lambda
                total = base + ordp
                val_losses.append(float(total.item()))
                val_ord_pens.append(float(ordp.item()))
        val_total = float(np.mean(val_losses)) if val_losses else float('nan')
        val_ord = float(np.mean(val_ord_pens)) if val_ord_pens else 0.0

        # Secondary metrics (per-level MAE) using existing evaluate()
        val_base_loss, val_mae_lv = evaluate(model, val_dl, nn.SmoothL1Loss(beta=cfg.huber_beta), device)

        # Scheduler step
        if scheduler is not None:
            if use_plateau:
                scheduler.step(val_total)
            else:
                scheduler.step()

        lr = _current_lr(optimizer)

        row = {
            'epoch': epoch,
            'train_total_loss': tr_loss,
            'val_total_loss': val_total,
            'val_base_loss': val_base_loss,
            'val_order_penalty': val_ord,
            'lr': lr,
            **{f'mae_{i}': float(x) for i, x in enumerate(val_mae_lv)}
        }
        metrics_rows.append(row)
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val_total {val_total:.4f} | val_base {val_base_loss:.4f} | ord {val_ord:.4f} | lr {lr:.2e} | per-level MAE {np.round(val_mae_lv,4)}")

        # Checkpointing
        if val_total < best_val - cfg.early_stop_min_delta:
            best_val = val_total
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_total': val_total}, best_path)

        # Early stopping
        es.step(val_total)
        if es.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # save metrics
    pd.DataFrame(metrics_rows).to_csv(run_dir / 'metrics.csv', index=False)
    return best_path
