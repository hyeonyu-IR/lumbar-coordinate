
from __future__ import annotations
from pathlib import Path
from typing import Sequence
from PIL import Image, ImageDraw

def overlay_points(img_path: Path, pred10, gt10, levels: Sequence[str], save_path: Path):
    im = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    w,h = im.size
    pred = pred10.reshape(5,2)
    gt   = gt10.reshape(5,2)
    for i,lvl in enumerate(levels):
        px, py = float(pred[i,0])*w, float(pred[i,1])*h
        gx, gy = float(gt[i,0])*w,   float(gt[i,1])*h
        r=3
        draw.ellipse((gx-r,gy-r,gx+r,gy+r), fill=(0,255,0))  # GT green
        draw.ellipse((px-r,py-r,px+r,py+r), fill=(255,0,0))  # Pred red
        draw.text((gx+4, gy-10), lvl, fill=(0,255,0))
    im.save(save_path)
