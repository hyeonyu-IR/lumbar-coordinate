
# Lumbar Coordinate Prediction — Reusable Research Template

**Author:** Hyeon Yu, MD  **Domain:** Radiology / Interventional Radiology  **Task:** Predict lumbar disc center coordinates (L1/L2–L5/S1) from sagittal spine images  **Framework:** PyTorch — notebook orchestration with modular `src/`

---

## 1) What this repo provides

- **Modular code in `src/`** (dataset, transforms, model, engine, metrics, visualization).  
- **Notebook-driven workflow** for split → train → validate → report.  
- **External data** (kept outside the repo).  
- **Per-run artifacts** saved under `runs/run_YYYYMMDD_HHMM/`.  
- **Automated PDF report** with methods, dataset summary, **training curves with LR markers**, best-epoch metrics, and **representative overlays** (best/median/worst).

```
repo/
├─ src/
│  ├─ config.py          # Paths + TrainConfig (schedules, early stopping, ordering prior)
│  ├─ dataset.py         # LumbarDataset (5 levels × 2 coords)
│  ├─ transforms.py      # Torchvision transforms
│  ├─ model.py           # ResNet backbones → 10-dim regression head
│  ├─ engine.py          # Training (early stopping, Plateau/Cosine, ordering prior)
│  ├─ train_utils.py     # train_one_epoch / evaluate
│  ├─ metrics.py         # per-level MAE utilities
│  ├─ visualization.py   # green=GT / red=Pred overlays
│  ├─ split.py           # Stratified train/val by source
│  └─ report_pdf.py      # PDF report (curves + LR markers + reps)
├─ notebooks/
│  ├─ 01_split.ipynb
│  ├─ 02_train.ipynb
│  └─ 03_validate_report.ipynb
├─ data/processed/       # small CSVs (train/val)
├─ runs/                 # per-run outputs (ignored by git)
└─ reports/              # generated PDFs (ignored by git)
```

---

## 2) Data location (outside repo)

Set an environment variable or edit `src/config.py`:

```bash
# preferred (Windows PowerShell example)
$env:LUMBAR_DATA_ROOT = "C:/Users/hyeon/Documents/miniconda_medimg_env/data/lumbar-coordinate"
```

```python
# src/config.py (fallback default)
DEFAULT_DATA_ROOT = Path("C:/Users/hyeon/Documents/miniconda_medimg_env/data/lumbar-coordinate")
```

Expected external layout:
```
<LUMBAR_DATA_ROOT>/
├─ data/                  # image folders
├─ coords_pretrain.csv
└─ coords_rsna_improved.csv
```

---

## 3) Notebooks — step-by-step

> All notebooks start with a small **bootstrap** cell to add the repo root to `sys.path` so that `import src.*` works whether you open from `repo/` or `repo/notebooks/`.

### 3.1) 01_split.ipynb — Create Train/Val

- Stratified by `source` (spider/lsd/osf/tseg).  
- Ensures 5 levels per image.  
- Saves CSVs to `data/processed/`:

```
train.csv
val.csv
```

### 3.2) 02_train.ipynb — Train model

- Loads `train.csv`/`val.csv`.  
- Builds datasets & loaders.  
- Instantiates model (ResNet-18/34 → 10 outputs).  
- Trains with **Smooth L1** and **AdamW**.  
- **Early stopping**, **scheduler**, and **ordering prior** are controlled via `TrainConfig`:

```python
from src.config import TrainConfig
cfg = TrainConfig(
    epochs=40,
    scheduler='plateau',          # or 'cosine'
    plateau_factor=0.5,
    plateau_patience=3,
    min_lr=1e-6,
    early_stop_patience=7,
    early_stop_min_delta=1e-4,
    order_reg_lambda=0.1,
    order_reg_margin=0.0,
)
```

**Outputs per run** (auto):
```
runs/run_YYYYMMDD_HHMM/
├─ checkpoints/best.pt
├─ figures/
├─ metrics.csv          # loss, per-level MAE, lr
└─ config.json
```

### 3.3) 03_validate_report.ipynb — Validate & Report

- Loads **best checkpoint**.
- Computes predictions on validation set.
- Saves overlays to `runs/<run>/figures/`.
- Exports per-level MAE table as `val_per_level_mae.csv` (optional but recommended).
- **NEW:** export per-image predictions (for representative cases):

```python
# Add to your validation notebook after inference
import numpy as np, pandas as pd
levels = ["L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]
rows = []
for i, fn in enumerate(all_fn):
    row = {"filename": fn}
    pr = pred[i].reshape(5,2); gtv = tgt[i].reshape(5,2)
    dists = []
    for j,lvl in enumerate(levels):
        row[f"pred_x_{lvl}"], row[f"pred_y_{lvl}"] = float(pr[j,0]), float(pr[j,1])
        row[f"gt_x_{lvl}"],   row[f"gt_y_{lvl}"]   = float(gtv[j,0]), float(gtv[j,1])
        dists.append(np.sqrt(((pr[j,0]-gtv[j,0])**2)+((pr[j,1]-gtv[j,1])**2)))
    row["mean_error"] = float(np.mean(dists))
    rows.append(row)

pd.DataFrame(rows).to_csv(RUN_DIR / "predictions.csv", index=False)
print("Saved:", RUN_DIR/"predictions.csv")
```

- Generate **PDF**:

```python
from src.report_pdf import generate_pdf_report
from src.config import Paths
from pathlib import Path
import json

PROJECT_ROOT = Path.cwd().resolve().parents[0] if Path.cwd().name=="notebooks" else Path.cwd()
paths = Paths.build(PROJECT_ROOT)
RUN_DIR = max((paths.runs_dir).glob("run_*"))
processed_dir = PROJECT_ROOT / "data" / "processed"

# methods section from config
with open(RUN_DIR/"config.json") as f:
    cfg_dict = json.load(f)
methods = {
    "Backbone": cfg_dict.get("backbone"),
    "Pretrained": cfg_dict.get("pretrained"),
    "Image size": cfg_dict.get("image_size"),
    "Batch size": cfg_dict.get("batch_size"),
    "Optimizer": "AdamW",
    "Base loss": f"SmoothL1 (beta={cfg_dict.get('huber_beta',0.01)})",
    "Scheduler": cfg_dict.get("scheduler"),
    "LR": cfg_dict.get("lr"),
    "Weight decay": cfg_dict.get("weight_decay"),
    "Early stopping": f"patience={cfg_dict.get('early_stop_patience')}, min_delta={cfg_dict.get('early_stop_min_delta')}",
    "Ordering prior": f"lambda={cfg_dict.get('order_reg_lambda')}, margin={cfg_dict.get('order_reg_margin')}"
}

fig_dir = RUN_DIR / "figures"
overlay_paths = sorted(fig_dir.glob("*_overlay.jpg"))
out_pdf = paths.reports_dir / f"report_{RUN_DIR.name}.pdf"

generate_pdf_report(
    project_root=PROJECT_ROOT,
    run_dir=RUN_DIR,
    train_csv=processed_dir/"train.csv",
    val_csv=processed_dir/"val.csv",
    metrics_csv=RUN_DIR/"metrics.csv",
    methods_dict=methods,
    overlay_paths=overlay_paths,
    out_pdf=out_pdf,
)
print("PDF saved to:", out_pdf)
```

---

## 4) Key configuration features

- **Schedulers**: `plateau` (ReduceLROnPlateau) or `cosine` (CosineAnnealingLR).  
- **Early stopping**: patience/min_delta on `val_total_loss`.  
- **Ordering prior**: hinge penalty enforcing cranio–caudal order of predicted y: `order_reg_lambda`, `order_reg_margin`.  
- **Level list** and **image size** configured centrally in `TrainConfig`.

---

## 5) What to commit vs ignore

- Commit: `src/`, `notebooks/`, `data/processed/*.csv`, `README.md`.  
- Ignore: raw data, `runs/`, and `reports/` (already in `.gitignore`).

---

## 6) Reproducibility

- Seeds set in `src/utils.py`.
- Config snapshot saved per run (`runs/<run>/config.json`).
- Metrics and curves saved per run (`metrics.csv`, training curves PNG, overlays).

---

## 7) Extensions

- Heatmap-based keypoint heads for comparison.  
- Level‑weighted loss (L4/L5, L5/S1 emphasis).  
- RSNA‑based pretraining or priors.  
- Automated “best/median/worst” case selection (now included in PDF generation when `predictions.csv` is present).

---

**Contact / Maintainer:** hyeonyu-IR  
**License:** MIT (or add your choice in LICENSE)
