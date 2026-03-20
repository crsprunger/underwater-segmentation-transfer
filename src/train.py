"""
Mask R-CNN training on COU (Common Objects Underwater) dataset.

Robustness features:
  - Atomic checkpointing (write-tmp-then-rename)
  - Auto-resume from latest checkpoint on startup
  - Graceful SIGINT/SIGTERM handling (saves checkpoint after current batch)

Usage:
  uv run python train.py          # start or auto-resume
  Ctrl+C                          # graceful stop with checkpoint
"""

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from src.training_config import TrainingConfig
from src.dataset import CocoDataset, BatchCollator, get_transforms
from src.model import build_model, get_param_groups
from src.evaluation import evaluate
from src.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    install_signal_handlers,
    is_shutdown_requested,
    save_history_csv,
    plot_training_progress,
)


# ── Configuration ─────────────────────────────────────────────────────


def load_config(config_path: Path) -> TrainingConfig:
    """Create TrainingConfig, optionally overriding defaults from a YAML file.

    The dataclass values in TrainingConfig remain the canonical defaults.
    A YAML file (if provided) may specify a flat mapping of field_name -> value.
    Only known TrainingConfig fields are allowed; unknown keys raise an error.
    """
    cfg_default = TrainingConfig()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(str(config_path), "r") as f:
        overrides = yaml.load(f, Loader=yaml.FullLoader) or {}

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping at the top level."
        )

    valid_keys = set(asdict(cfg_default).keys())
    if "checkpoint_dir" not in overrides:
        overrides["checkpoint_dir"] = str(config_path.parent)
    for key in overrides.keys():
        if key not in valid_keys:
            raise KeyError(
                f"Unknown config key {key!r} in {config_path}. "
                f"Valid keys: {sorted(valid_keys)}"
            )

    merged = asdict(cfg_default)
    merged.update(overrides)
    return TrainingConfig(**merged)


# ── Training Loop ─────────────────────────────────────────────────────


def _compute_grad_stats(grad_norms: list, clip_threshold: float) -> dict:
    """Compute summary statistics from per-step gradient norms."""
    import math

    # Filter out inf/nan from AMP overflow events
    finite = [g for g in grad_norms if math.isfinite(g)]
    n_total = len(grad_norms)
    n_finite = len(finite)
    n_inf = n_total - n_finite

    if not finite:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "clip_fraction": float("nan"),
            "n_inf": n_inf,
        }

    finite.sort()
    mean = sum(finite) / n_finite
    median = finite[n_finite // 2]
    p95 = finite[int(n_finite * 0.95)]
    max_val = finite[-1]
    n_clipped = (
        sum(1 for g in finite if g > clip_threshold) if clip_threshold > 0 else 0
    )
    clip_fraction = n_clipped / n_finite

    return {
        "mean": mean,
        "median": median,
        "p95": p95,
        "max": max_val,
        "clip_fraction": clip_fraction,
        "n_inf": n_inf,
    }


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    cfg,
    scaler,
    scheduler,
    best_metric,
    history,
):
    """Train one epoch. Returns (avg_loss, was_interrupted, grad_norms)."""
    model.train()

    # 1. Initialize as a GPU tensor instead of a float to prevent CPU/GPU syncs
    running_loss = torch.tensor(0.0, device=device)
    num_batches = 0
    accum = cfg.accumulation_steps
    grad_norms = []  # per-optimizer-step grad norms (before clipping)

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(data_loader):
        if is_shutdown_requested():
            print(f"  Shutdown requested at epoch {epoch}, batch {batch_idx}")
            # Safe sync on shutdown
            final_loss = running_loss.item() / max(num_batches, 1)
            final_norms = [
                gn.item() if isinstance(gn, torch.Tensor) else gn for gn in grad_norms
            ]
            return final_loss, True, final_norms

        # Async transfer to GPU
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if accum > 1:
                losses = losses / accum  # normalize for accumulation

        scaler.scale(losses).backward()

        # Optimizer step every `accum` batches (or at end of epoch)
        if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(data_loader):
            scaler.unscale_(optimizer)

            # clip_grad_norm_ returns the total norm BEFORE clipping
            if cfg.clip_grad_norm > 0.0:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.clip_grad_norm,
                )
            else:
                # Compute norm without clipping (same set of params)
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float("inf"),  # inf max_norm = no clipping, just compute
                )

            if cfg.log_grad_norm:
                # 2. Append the detached tensor to avoid sync!
                grad_norms.append(pre_clip_norm.detach())

                if cfg.verbose and (batch_idx + 1) % (accum * cfg.print_freq) == 0:
                    # Only sync the specific print value when necessary
                    print(f"      [Grad Norm: {pre_clip_norm.item():.4f}]")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()

        # 3. Accumulate using .detach() (stays on GPU, no sync)
        running_loss += losses.detach() * (accum if accum > 1 else 1)
        num_batches += 1

        # 4. Handle periodic console print (the only place we sync during the loop)
        if batch_idx % cfg.print_freq == 0:
            head_lr = optimizer.param_groups[0]["lr"]

            # If we have 4 param groups, group [2] contains the backbone weights
            if len(optimizer.param_groups) > 2:
                bb_lr = optimizer.param_groups[2]["lr"]
                lr_str = f"Head LR: {head_lr:.2e} | BB LR: {bb_lr:.2e}"
            else:
                lr_str = f"LR: {head_lr:.2e}"
            if cfg.verbose:
                # Calculate the printable loss just for this batch
                current_loss_val = (
                    losses.detach() * (accum if accum > 1 else 1)
                ).item()

                print(
                    f"  Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] "
                    f"Loss: {current_loss_val:.4f}  {lr_str}"
                )

    # 5. End of Epoch: Sync the accumulated values back to the CPU exactly once
    epoch_loss = running_loss.item() / max(num_batches, 1)

    # Convert the list of GPU tensors to a list of Python floats
    if cfg.log_grad_norm:
        grad_norms = [gn.item() for gn in grad_norms]

    return epoch_loss, False, grad_norms


# ── Main ──────────────────────────────────────────────────────────────


def main():
    """Entry point: parse args, build model/data, run the training loop.

    Loads or creates a TrainingConfig from YAML, sets up the model, optimizer,
    scheduler, and data loaders, then runs training with periodic evaluation.
    Automatically resumes from the latest checkpoint if one exists.
    """
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on COU.")
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model directory to load the config from. Must be a subdirectory of checkpoints/ with a training_config.yaml file.",
    )
    args = parser.parse_args()

    config_path = Path("checkpoints") / args.model_name / "training_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_config(config_path)

    install_signal_handlers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Validate required paths are set
    for field in ("train_ann_path", "val_ann_path", "train_img_dir", "val_img_dir"):
        if not getattr(cfg, field):
            raise ValueError(f"Config field '{field}' must be set before training.")

    # Datasets
    train_dataset = CocoDataset(
        coco_ann_path=cfg.train_ann_path,
        img_dir=Path(cfg.train_img_dir),
        transforms=get_transforms(train=True, cfg=cfg),
        cfg=cfg,
    )
    val_dataset = CocoDataset(
        coco_ann_path=cfg.val_ann_path,
        img_dir=Path(cfg.val_img_dir),
        transforms=get_transforms(train=False, cfg=cfg),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=BatchCollator(cfg),
        pin_memory=True,
        persistent_workers=True,
        drop_last=cfg.use_torch_compile,
        prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=BatchCollator(TrainingConfig(use_copy_paste=False)),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Model
    model = build_model(cfg)
    model.to(device)

    # Optimizer, scheduler, scaler
    # Exclude norm layer params and biases from weight decay (Detectron2 convention)
    # Identify norm params by module type, not name (catches bn1/bn2/bn3 and indexed GN layers)
    param_groups = get_param_groups(model, cfg)

    # Split backbone.body params into separate groups for differential lr/wd

    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=cfg.adam_betas,
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {cfg.optimizer!r}. Use 'sgd', or 'adamw'."
        )
    print(
        f"Optimizer: {cfg.optimizer.upper()}, LR: {cfg.lr}, "
        f"Effective batch: {cfg.batch_size * cfg.accumulation_steps}"
    )

    # Calculate steps for a continuous, per-batch schedule
    iters_per_epoch = len(train_loader) // cfg.accumulation_steps
    warmup_steps = iters_per_epoch * cfg.warmup_epochs
    total_steps = iters_per_epoch * cfg.num_epochs
    if cfg.lr_scheduler_type == "cosine":
        eta_min = cfg.cosine_eta_min
        # 1. Linear Warmup Phase
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=cfg.warmup_factor, total_iters=warmup_steps
        )
        # 2. Cosine Decay Phase
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(total_steps - warmup_steps), eta_min=eta_min
        )
        # 3. Chain them together seamlessly
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        print(
            f"Sequential schedule: {warmup_steps} warmup steps, then Cosine to eta_min={eta_min:.2e}"
        )
    else:
        # 1. Linear Warmup Phase
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=cfg.warmup_factor, total_iters=warmup_steps
        )
        # 2. MultiStep Decay Phase
        step_milestones = [m * iters_per_epoch for m in cfg.lr_milestones]
        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_milestones, gamma=cfg.lr_gamma
        )
        # 3. Chain them together seamlessly
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, multistep_scheduler],
            milestones=[warmup_steps],
        )
        print(
            f"Sequential schedule: {warmup_steps} warmup steps, then MultiStep to ({', '.join(f'{x:.1e}' for x in [cfg.lr * cfg.lr_gamma**(i + 1) for i in range(len(cfg.lr_milestones))])}) at {cfg.lr_milestones}"
        )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    start_epoch = 0
    best_metric = 0.0
    history = {
        "loss": [],
        "mask_mAP": [],
        "mask_mAP_50": [],
        "mask_mAP_75": [],
        "mask_mAP_small": [],
        "box_mAP": [],
        "box_mAP_50": [],
        "box_mAP_75": [],
        "box_mAP_small": [],
        "grad_norms": [],
    }

    # Resume from checkpoint
    checkpoint = load_checkpoint(cfg)
    if checkpoint is not None:
        # Strip _orig_mod. prefix from compiled-model checkpoints
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]

        # Support both old (loss_history) and new (history) checkpoint formats
        if "history" in checkpoint:
            loaded_hist = checkpoint["history"]
            # Merge loaded history into our template (handles missing keys from older checkpoints)
            for key in history:
                if key in loaded_hist:
                    history[key] = loaded_hist[key]
        elif "loss_history" in checkpoint:
            history["loss"] = checkpoint["loss_history"]
        print(f"Resumed from epoch {checkpoint['epoch']}, best mAP: {best_metric:.4f}")

    if cfg.use_torch_compile:
        model = torch.compile(model)
    else:
        print("Not using torch.compile")

    # Save config to checkpoint directory for reference during evaluation
    config_path = Path(cfg.checkpoint_dir) / "training_config_full.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_path}")

    print(f"\nStarting training: epochs {start_epoch} to {cfg.num_epochs - 1}")
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    print(f"Batch size: {cfg.batch_size}, Batches/epoch: {len(train_loader)}")

    session_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{cfg.num_epochs - 1}")
        print(f"{'=' * 60}")

        epoch_start = time.perf_counter()

        avg_loss, interrupted, epoch_grad_norms = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            cfg,
            scaler,
            scheduler,
            best_metric,
            history,
        )

        if interrupted or is_shutdown_requested():
            # Epoch incomplete — don't update history or overwrite the
            # completed-epoch checkpoint. latest_checkpoint.pth still
            # reflects the last full epoch.
            print(
                "Training interrupted mid-epoch. Resume will restart from last completed epoch."
            )
            sys.exit(0)

        history["loss"].append(avg_loss)
        if cfg.log_grad_norm:
            history["grad_norms"].append(epoch_grad_norms)
            gs = _compute_grad_stats(epoch_grad_norms, cfg.clip_grad_norm)
        epoch_elapsed = time.perf_counter() - epoch_start
        session_elapsed = time.perf_counter() - session_start
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")
        if cfg.log_grad_norm:
            print(
                f"  Grad norm — median: {gs['median']:.2f}  mean: {gs['mean']:.2f}  "
                f"p95: {gs['p95']:.2f}  max: {gs['max']:.2f}  "
                f"clipped: {gs['clip_fraction']:.0%}  AMP overflows: {gs['n_inf']}"
            )
        print(
            f"  Epoch time: {epoch_elapsed / 60:.1f}min  |  Session total: {session_elapsed / 60:.1f}min"
        )

        # Evaluate raw model (unwrap torch.compile to avoid slow eval-mode retracing)
        raw_eval_model = getattr(model, "_orig_mod", model)
        print("  Running evaluation (raw model)...")
        t_eval_start = time.perf_counter()
        metrics = evaluate(raw_eval_model, val_loader)
        eval_elapsed = time.perf_counter() - t_eval_start
        print(
            f"  Raw  — mask_mAP: {metrics['mask_mAP']:.4f}  mask_AP50: {metrics['mask_mAP_50']:.4f}"
        )
        print(
            f"         box_mAP:  {metrics['box_mAP']:.4f}   box_AP50:  {metrics['box_mAP_50']:.4f}"
        )
        print(f"         mask_mAP_small: {metrics['mask_mAP_small']:.4f}")
        print(f"         box_mAP_small:  {metrics['box_mAP_small']:.4f}")

        # Record raw model metrics in history
        history["mask_mAP"].append(metrics["mask_mAP"])
        history["mask_mAP_50"].append(metrics["mask_mAP_50"])
        history["mask_mAP_75"].append(metrics["mask_mAP_75"])
        history["box_mAP"].append(metrics["box_mAP"])
        history["box_mAP_50"].append(metrics["box_mAP_50"])
        history["box_mAP_75"].append(metrics["box_mAP_75"])
        history["mask_mAP_small"].append(metrics["mask_mAP_small"])
        history["box_mAP_small"].append(metrics["box_mAP_small"])

        eval_elapsed = time.perf_counter() - t_eval_start
        print(f"  Eval time: {eval_elapsed / 60:.1f}min")

        # Save latest
        save_checkpoint(
            cfg,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            history,
        )

        # Save best model
        if metrics["mask_mAP"] > best_metric:
            best_metric = metrics["mask_mAP"]
            save_checkpoint(
                cfg,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                history,
                kind="best",
            )
            print(f"  New best mask mAP: {best_metric:.4f}")

        # Plot progress
        plot_training_progress(history, cfg)
        save_history_csv(history, cfg)

        if is_shutdown_requested():
            print("Training interrupted between epochs. Exiting.")
            sys.exit(0)

    print(f"\nTraining complete. Best mask mAP: {best_metric:.4f}")


if __name__ == "__main__":
    main()
