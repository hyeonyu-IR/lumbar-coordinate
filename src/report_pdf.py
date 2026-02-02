
from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Optional, Tuple
import json
import math
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.units import inch


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    return t


def _kv_table(d: Dict[str, str]):
    data = [("Key", "Value")] + [(str(k), str(v)) for k, v in d.items()]
    return _table(data, col_widths=[2.2*inch, 4.8*inch])


# ---------- LR change detection ----------

def _find_lr_change_epochs(dfm: pd.DataFrame) -> Sequence[Tuple[int, float]]:
    """Return list of (epoch, lr) where learning rate changes compared to previous row."""
    if 'lr' not in dfm.columns:
        return []
    changes = []
    for i in range(1, len(dfm)):
        if not math.isclose(float(dfm.loc[i, 'lr']), float(dfm.loc[i-1, 'lr']), rel_tol=1e-12, abs_tol=0):
            changes.append((int(dfm.loc[i, 'epoch']), float(dfm.loc[i, 'lr'])))
    return changes


# ---------- Training curves ----------

def _plot_training_curves(
    metrics_csv: Path,
    out_png: Path,
    level_names=("L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"),
    figsize=(10, 4)
) -> Optional[Path]:
    if not metrics_csv.exists():
        return None
    dfm = pd.read_csv(metrics_csv)
    if 'epoch' not in dfm.columns:
        return None

    lr_changes = _find_lr_change_epochs(dfm)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_l, ax_r = axes

    x = dfm['epoch'].values
    if 'train_total_loss' in dfm.columns:
        ax_l.plot(x, dfm['train_total_loss'].values, label='train_total_loss', lw=2)
    if 'val_total_loss' in dfm.columns:
        ax_l.plot(x, dfm['val_total_loss'].values, label='val_total_loss', lw=2)
    if 'val_base_loss' in dfm.columns:
        ax_l.plot(x, dfm['val_base_loss'].values, label='val_base_loss', lw=2, ls='--')

    # Annotate LR changes
    for ep, lr in lr_changes:
        ax_l.axvline(ep, color='grey', ls=':', alpha=0.8)
        ax_l.text(ep+0.1, ax_l.get_ylim()[0], f"LR→{lr:.1e}", rotation=90,
                  va='bottom', ha='left', fontsize=8, color='grey')

    ax_l.set_title('Loss vs. Epoch')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('Loss')
    ax_l.grid(True, alpha=0.4)
    ax_l.legend(loc='best')

    # Right: per-level MAE
    for i, lvl in enumerate(level_names):
        col = f'mae_{i}'
        if col in dfm.columns:
            ax_r.plot(x, dfm[col].values, label=lvl, lw=2)

    ax_r.set_title('Validation MAE (normalized) vs. Epoch')
    ax_r.set_xlabel('Epoch')
    ax_r.set_ylabel('MAE (norm)')
    ax_r.grid(True, alpha=0.4)
    ax_r.legend(loc='best', fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


# ---------- Representative cases ----------

def _compute_per_image_error(predictions_csv: Path) -> Optional[pd.DataFrame]:
    """
    Expect a predictions CSV with at least: filename, and columns:
    pred_x_L1/L2, pred_y_L1/L2, ..., pred_x_L5/S1, pred_y_L5/S1
    gt_x_L1/L2, gt_y_L1/L2, ..., gt_x_L5/S1, gt_y_L5/S1
    All in normalized units (0-1).
    Returns a DataFrame with columns [filename, mean_error] sorted ascending.
    """
    if not predictions_csv.exists():
        return None
    df = pd.read_csv(predictions_csv)
    # infer level names
    levels = ["L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]
    errs = []
    for _, r in df.iterrows():
        per_levels = []
        for lvl in levels:
            px = r.get(f'pred_x_{lvl}', None)
            py = r.get(f'pred_y_{lvl}', None)
            gx = r.get(f'gt_x_{lvl}', None)
            gy = r.get(f'gt_y_{lvl}', None)
            if px is None or py is None or gx is None or gy is None:
                continue
            per_levels.append(((px-gx)**2 + (py-gy)**2)**0.5)
        if per_levels:
            errs.append((r['filename'], float(sum(per_levels)/len(per_levels))))
    dfe = pd.DataFrame(errs, columns=['filename','mean_error']).sort_values('mean_error').reset_index(drop=True)
    return dfe


def _select_representatives(dfe: pd.DataFrame, k_each: int = 2) -> Dict[str, Sequence[str]]:
    """Return dict with keys 'best','median','worst' -> list of filenames (length up to k_each)."""
    n = len(dfe)
    reps = {"best": [], "median": [], "worst": []}
    if n == 0:
        return reps
    # best
    reps["best"] = list(dfe.head(k_each)['filename'])
    # worst
    reps["worst"] = list(dfe.tail(k_each)['filename'])
    # median around middle index
    mid = n // 2
    start = max(0, mid - (k_each//2))
    reps["median"] = list(dfe.iloc[start:start+k_each]['filename'])
    return reps


def _filenames_to_overlay_paths(filenames: Sequence[str], figures_dir: Path) -> Sequence[Path]:
    """Map filenames to overlay images assuming naming <stem>_overlay.jpg in figures_dir."""
    out = []
    for fn in filenames:
        stem = Path(fn).stem
        cand = figures_dir / f"{stem}_overlay.jpg"
        if cand.exists():
            out.append(cand)
    return out


# ---------- Main PDF generator ----------

def generate_pdf_report(
    project_root: Path,
    run_dir: Path,
    train_csv: Path,
    val_csv: Path,
    metrics_csv: Path,
    methods_dict: Dict[str, str],
    overlay_paths: Sequence[Path],
    out_pdf: Path
):
    project_root = Path(project_root)
    run_dir = Path(run_dir)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Lumbar Coordinate Prediction – Automated Report", styles['Title']))
    story.append(Spacer(1, 0.15*inch))

    # Methods summary
    story.append(Paragraph("Methods (Summary)", styles['Heading2']))
    story.append(_kv_table(methods_dict))
    story.append(Spacer(1, 0.15*inch))

    # Dataset summary
    story.append(Paragraph("Dataset Summary", styles['Heading2']))
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    n_train = df_train['filename'].nunique()
    n_val = df_val['filename'].nunique()
    # by source
    src_train = df_train.groupby('source')['filename'].nunique().sort_index()
    src_val = df_val.groupby('source')['filename'].nunique().sort_index()

    data_ds = [("Split", "#Images", "By Source (images)")]
    data_ds.append(("Train", n_train, ", ".join(f"{k}: {v}" for k, v in src_train.items())))
    data_ds.append(("Val", n_val, ", ".join(f"{k}: {v}" for k, v in src_val.items())))
    story.append(_table(data_ds, col_widths=[1.0*inch, 1.0*inch, 5.0*inch]))
    story.append(Spacer(1, 0.15*inch))

    # Validation Metrics (Best Epoch by val_total_loss)
    story.append(Paragraph("Validation Metrics (Best Epoch by val_total_loss)", styles['Heading2']))
    dfm = pd.read_csv(metrics_csv)
    best_idx = dfm['val_total_loss'].idxmin()
    best = dfm.loc[best_idx].to_dict()
    levels = ["L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]
    data_mt = [("Metric", "Value")]
    data_mt.append(("best_epoch", int(best.get('epoch', -1))))
    data_mt.append(("val_total_loss", f"{best.get('val_total_loss', float('nan')):.4f}"))
    data_mt.append(("val_base_loss", f"{best.get('val_base_loss', float('nan')):.4f}"))
    data_mt.append(("val_order_penalty", f"{best.get('val_order_penalty', 0.0):.4f}"))
    for i, lvl in enumerate(levels):
        key = f"mae_{i}"
        if key in best:
            data_mt.append((f"MAE {lvl}", f"{float(best[key]):.4f}"))
    story.append(_table(data_mt, col_widths=[2.0*inch, 5.0*inch]))
    story.append(Spacer(1, 0.15*inch))

    # Optional: include sample-aggregated MAE if available (val_per_level_mae.csv)
    val_mae_csv = run_dir / "val_per_level_mae.csv"
    if val_mae_csv.exists():
        story.append(Paragraph("Per-level MAE (Best Checkpoint; Sample-aggregated)", styles['Heading2']))
        dfv = pd.read_csv(val_mae_csv)
        data_v = [("Level","MAE (norm)")] + [(r["level"], f"{float(r['mae_norm']):.4f}") for _, r in dfv.iterrows()]
        story.append(_table(data_v, col_widths=[2.0*inch, 5.0*inch]))
        story.append(Spacer(1, 0.15*inch))

    # Training curves with LR markers
    story.append(Paragraph("Training Curves", styles['Heading2']))
    curves_png = run_dir / "figures" / "train_curves.png"
    plotted = _plot_training_curves(metrics_csv=metrics_csv, out_png=curves_png)
    if plotted and plotted.exists():
        story.append(RLImage(str(plotted), width=6.4*inch, height=2.8*inch))
        story.append(Spacer(1, 0.15*inch))
    else:
        story.append(Paragraph("Training curves unavailable (metrics.csv missing or malformed).", styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))

    # Representative overlays (generic grid for quick glance)
    story.append(Paragraph("Representative Overlays (Sample)", styles['Heading2']))
    max_imgs = min(6, len(overlay_paths))
    imgs = [RLImage(str(p), width=2.7*inch, height=2.7*inch) for p in overlay_paths[:max_imgs]]
    rows = []
    row = []
    for i, im in enumerate(imgs):
        row.append(im)
        if (i % 2 == 1) or (i == max_imgs - 1):
            rows.append(row)
            row = []
    if rows:
        t = Table(rows, colWidths=[3*inch, 3*inch])
        story.append(t)
        story.append(Spacer(1, 0.15*inch))

    # NEW: Best / Median / Worst overlays by per-image error if predictions.csv available
    pred_csv = run_dir / "predictions.csv"
    dfe = _compute_per_image_error(pred_csv)
    if dfe is not None and len(dfe) > 0:
        figs_dir = run_dir / 'figures'
        reps = _select_representatives(dfe, k_each=2)

        for section in ["best","median","worst"]:
            title = {
                "best": "Representative Cases – Best (lowest error)",
                "median": "Representative Cases – Median",
                "worst": "Representative Cases – Worst (highest error)",
            }[section]
            story.append(Paragraph(title, styles['Heading2']))
            files = reps[section]
            paths = _filenames_to_overlay_paths(files, figs_dir)
            if not paths:
                story.append(Paragraph("No overlays found for this category.", styles['BodyText']))
                story.append(Spacer(1, 0.1*inch))
                continue
            # two per row
            imgs_sec = [RLImage(str(p), width=2.7*inch, height=2.7*inch) for p in paths]
            rows = []
            row = []
            for i, im in enumerate(imgs_sec):
                row.append(im)
                if (i % 2 == 1) or (i == len(imgs_sec) - 1):
                    rows.append(row)
                    row = []
            t = Table(rows, colWidths=[3*inch, 3*inch])
            story.append(t)
            story.append(Spacer(1, 0.15*inch))

    # Build PDF
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    doc.build(story)
