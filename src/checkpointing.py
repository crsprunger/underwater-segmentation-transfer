"""Checkpointing, signal handling, history export, and training visualization.

Contains atomic checkpoint save/load, graceful shutdown signal handling,
CSV history export, and training progress plotting.
"""

import csv
import signal
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.training_config import TrainingConfig


# ── Checkpoint paths ──────────────────────────────────────────────────


def _checkpoint_path(cfg: TrainingConfig, kind: str = "latest") -> Path:
    """Return the checkpoint file path for a given kind ('latest' or 'best')."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    if kind == "best":
        return ckpt_dir / "best_model.pth"
    return ckpt_dir / "latest_checkpoint.pth"


def save_checkpoint(
    cfg: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    history: dict,
    kind: str = "latest",
):
    """Atomic checkpoint save: write to .tmp, then rename."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap torch.compile's OptimizedModule to get clean state_dict keys
    raw_model = getattr(model, "_orig_mod", model)
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history,
        "config": asdict(cfg),
    }

    path = _checkpoint_path(cfg, kind)
    tmp_path = path.with_suffix(".pth.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)  # atomic on same filesystem
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(cfg: TrainingConfig) -> Optional[dict]:
    """Load latest checkpoint if it exists."""
    path = _checkpoint_path(cfg, "latest")
    if not path.exists():
        print("No existing checkpoint found. Starting from scratch.")
        return None
    print(f"Resuming from checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Signal Handling ───────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, _frame):
    """Set the shutdown flag on SIGINT/SIGTERM so the training loop can exit cleanly."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    print(f"\n{'=' * 60}")
    print(f"Received {sig_name}. Stopping training...")
    print(f"{'=' * 60}")
    _shutdown_requested = True


def install_signal_handlers():
    """Register SIGINT/SIGTERM handlers for graceful training shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def is_shutdown_requested() -> bool:
    """Check whether a graceful shutdown has been requested via signal."""
    return _shutdown_requested


# ── History export ────────────────────────────────────────────────────


def save_history_csv(
    history: dict, cfg: TrainingConfig, ckpt_dir_override: Optional[str] = None
):
    """Save training history as CSV files in the checkpoint directory.

    Writes two files:
      training_history.csv — one row per epoch, scalar metrics only
      grad_norms.csv — one row per optimizer step: epoch, step, grad_norm
    """
    if ckpt_dir_override is not None:
        ckpt_dir = Path(ckpt_dir_override)
    else:
        ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Scalar metrics CSV ──
    # Skip nested lists (grad_norms) — those go in a separate file
    preferred_order = [
        "loss",
        "mask_mAP",
        "mask_mAP_50",
        "mask_mAP_75",
        "box_mAP",
        "box_mAP_50",
        "box_mAP_75",
        "grad_norm",  # legacy single-value field from older checkpoints
    ]

    def _is_scalar_list(v):
        return isinstance(v, list) and v and not isinstance(v[0], list)

    columns = [
        k for k in preferred_order if k in history and _is_scalar_list(history[k])
    ]
    for k in sorted(history.keys()):
        if k not in columns and k in history and _is_scalar_list(history[k]):
            columns.append(k)

    n_rows = max((len(history[k]) for k in columns), default=0)
    if n_rows > 0:
        out_path = ckpt_dir / "training_history.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + columns)
            for i in range(n_rows):
                row = [i]
                for col in columns:
                    vals = history.get(col, [])
                    row.append(vals[i] if i < len(vals) else "")
                writer.writerow(row)
        print(f"  History CSV saved: {out_path}")

    # ── Per-step grad norms CSV ──
    grad_norms_per_epoch = history.get("grad_norms", [])
    if grad_norms_per_epoch:
        out_path = ckpt_dir / "grad_norms.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "grad_norm"])
            for epoch_idx, norms in enumerate(grad_norms_per_epoch):
                if isinstance(norms, list):
                    for step_idx, norm in enumerate(norms):
                        writer.writerow([epoch_idx, step_idx, norm])
        print(f"  Grad norms CSV saved: {out_path}")


# ── Visualization ─────────────────────────────────────────────────────


def plot_training_progress(history: dict, cfg: TrainingConfig):
    """Save a training progress plot to the checkpoint directory."""
    epochs = list(range(len(history.get("loss", []))))
    if not epochs:
        return

    has_metrics = len(history.get("mask_mAP", [])) > 0
    n_plots = 3 if has_metrics else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(epochs, history["loss"], "b-o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Avg Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if has_metrics:
        metric_epochs = list(range(len(history["mask_mAP"])))

        # mAP curves
        axes[1].plot(
            metric_epochs, history["mask_mAP"], "r-o", markersize=3, label="mask mAP"
        )
        axes[1].plot(
            metric_epochs, history["box_mAP"], "b-o", markersize=3, label="box mAP"
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP @ 0.5:0.95")
        axes[1].set_title("Validation mAP")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AP50 curves
        axes[2].plot(
            metric_epochs,
            history["mask_mAP_50"],
            "r-o",
            markersize=3,
            label="mask AP50",
        )
        axes[2].plot(
            metric_epochs, history["box_mAP_50"], "b-o", markersize=3, label="box AP50"
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("mAP @ 0.5")
        axes[2].set_title("Validation AP50")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(cfg.checkpoint_dir) / "training_progress.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")
