
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .metrics import mae_per_level

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0; self.n = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.n += n
    @property
    def avg(self):
        return self.sum / max(1,self.n)

def train_one_epoch(model, dl, criterion, optimizer, device):
    model.train()
    meter = AverageMeter()
    for x, y, _ in dl:
        x = x.to(device)
        y = y.to(device).float()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), x.size(0))
    return meter.avg

@torch.no_grad()
def evaluate(model, dl, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    sum_lv = np.zeros(5, dtype=np.float64)
    count = 0
    for x, y, _ in dl:
        x = x.to(device)
        y = y.to(device).float()
        out = model(x)
        loss = criterion(out, y)
        loss_meter.update(loss.item(), x.size(0))
        lv = mae_per_level(out, y)
        sum_lv += lv
        count += 1
    return loss_meter.avg, (sum_lv / max(1,count))
