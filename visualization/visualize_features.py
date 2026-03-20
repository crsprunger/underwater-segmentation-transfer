"""
Feature space visualization for cross-dataset analysis.

Three levels of analysis, all saved to --output-dir:

  1. Image-level (GAP of layer4), colored by source dataset
  2. Image-level (GAP of layer4), colored by dominant class per image
  3. Per-object (ROI-aligned layer4 crop), colored by class and/or dataset

Silhouette scores (cosine, raw 2048-dim features) are computed for every
grouping and annotated on each plot.

Usage:
    python visualize_features.py \
        --models \
            tc:checkpoints/trashcan_chunksplit/model6v2/best_model.pth:checkpoints/trashcan_chunksplit/model6v2/training_config_full.yaml:trashcan \
            sc:checkpoints/seaclear_chunksplit/model6v2/best_model.pth:checkpoints/seaclear_chunksplit/model6v2/training_config_full.yaml:seaclear \
            pooled:checkpoints/pooled_chunksplit_unfiltered/model6v2/best_model.pth:checkpoints/pooled_chunksplit_unfiltered/model6v2/training_config_full.yaml:pooled \
        --tc-ann trashcan_data/dataset/instance_version/trashcan_chunksplit_val_tc20.json trashcan_data/dataset/instance_version/trashcan_chunksplit_test_tc20.json \
        --tc-img trashcan_data/dataset/instance_version/images \
        --sc-ann seaclear_data/seaclear_480p_chunksplit_val_tc20.json seaclear_data/seaclear_480p_chunksplit_test_tc20.json \
        --sc-img seaclear_data/images_480p \
        --output-dir feature_plots

    # Re-plot from saved .npz without re-extracting:
    python visualize_features.py --plot-only --output-dir feature_plots \\
        --tc-ann ... --sc-ann ...  # still needed for category names
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.ops as tv_ops
from torchvision.io import decode_image
from pycocotools.coco import COCO
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model import build_model
from src.train import load_config


# ═══════════════════════════════════════════════════════════════════════
# COCO helpers
# ═══════════════════════════════════════════════════════════════════════


def merge_cocos(paths: list) -> COCO:
    """
    Load and merge multiple COCO annotation JSON files into one COCO object.
    Assumes non-overlapping image IDs across files (e.g. val + test from same dataset).
    """
    merged: dict = {"images": [], "annotations": [], "categories": None}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        merged["images"].extend(data["images"])
        merged["annotations"].extend(data["annotations"])
        if merged["categories"] is None:
            merged["categories"] = data["categories"]
    coco = COCO()
    coco.dataset = merged
    coco.createIndex()
    return coco


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction — image-level (layer4 global average pool)
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_image_features(
    model, coco: COCO, img_dir: Path, batch_size: int = 16
) -> tuple[np.ndarray, list]:
    """
    Extract 2048-dim per-image embeddings (layer4 GAP).
    Returns (features: (N, 2048), img_ids: [N]).
    """
    device = next(model.parameters()).device
    model.eval()
    captured = {}

    def hook_fn(_module, _input, output):
        captured["feat"] = output.mean(dim=[2, 3])  # (B, 2048)

    handle = model.backbone.body.layer4.register_forward_hook(hook_fn)
    img_ids = coco.getImgIds()
    all_features = []

    for i in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[i : i + batch_size]
        batch_imgs = []
        for img_id in batch_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_dir / Path(img_info["file_name"]).name
            img = decode_image(str(img_path), mode=torchvision.io.ImageReadMode.RGB)
            batch_imgs.append(img.float() / 255.0)

        images, _ = model.transform(batch_imgs, None)
        model.backbone(images.tensors.to(device))
        all_features.append(captured["feat"].cpu().numpy())

        if (i // batch_size + 1) % 20 == 0:
            print(f"    {min(i + batch_size, len(img_ids))}/{len(img_ids)} images ...")

    handle.remove()
    features = np.concatenate(all_features, axis=0)
    print(f"    Done: {features.shape[0]} image embeddings ({features.shape[1]}-dim)")
    return features, img_ids


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction — per-object (ROI-align layer4 to GT boxes)
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_roi_features(
    model, coco: COCO, img_dir: Path, batch_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-object 2048-dim features by ROI-aligning the layer4 feature
    map to each GT bounding box (7×7 pool, then mean → 2048-dim vector).

    Returns:
        features: (M, 2048) float32
        labels:   (M,) int32 category_id per object
    """
    device = next(model.parameters()).device
    model.eval()
    captured = {}

    def hook_fn(_module, _input, output):
        captured["feat"] = output  # (B, 2048, H_feat, W_feat)

    handle = model.backbone.body.layer4.register_forward_hook(hook_fn)
    img_ids = coco.getImgIds()
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []

    for i in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[i : i + batch_size]
        batch_imgs = []
        batch_orig_sizes = []
        batch_anns = []

        for img_id in batch_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_dir / Path(img_info["file_name"]).name
            img = decode_image(str(img_path), mode=torchvision.io.ImageReadMode.RGB)
            batch_imgs.append(img.float() / 255.0)
            batch_orig_sizes.append((img.shape[-2], img.shape[-1]))  # (H, W)
            batch_anns.append(coco.loadAnns(coco.getAnnIds(imgIds=img_id)))

        images, _ = model.transform(batch_imgs, None)
        model.backbone(images.tensors.to(device))
        feat_maps = captured["feat"]  # (B, 2048, H_feat, W_feat)

        for b_idx, (orig_h, orig_w) in enumerate(batch_orig_sizes):
            new_h, new_w = images.image_sizes[b_idx]  # after resize, before padding
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            # Process per-image to avoid padding-offset issues
            feat_map = feat_maps[b_idx : b_idx + 1]  # (1, 2048, H_feat, W_feat)

            for ann in batch_anns[b_idx]:
                x, y, w, h = ann["bbox"]
                if w < 2 or h < 2:
                    continue
                x1 = x * scale_x
                y1 = y * scale_y
                x2 = (x + w) * scale_x
                y2 = (y + h) * scale_y

                # roi_align expects [batch_idx, x1, y1, x2, y2] in input-image pixel coords;
                # spatial_scale=1/32 maps those to feature-map coordinates.
                boxes = torch.tensor(
                    [[0.0, x1, y1, x2, y2]], dtype=torch.float32, device=device
                )
                roi_feat = tv_ops.roi_align(
                    feat_map,
                    boxes,
                    output_size=(7, 7),
                    spatial_scale=1 / 32.0,
                    aligned=True,
                )  # (1, 2048, 7, 7)
                all_features.append(roi_feat.mean(dim=[2, 3]).cpu().numpy()[0])
                all_labels.append(ann["category_id"])

        if (i // batch_size + 1) % 20 == 0:
            n_done = min(i + batch_size, len(img_ids))
            print(
                f"    {n_done}/{len(img_ids)} images, {len(all_features)} objects ..."
            )

    handle.remove()
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    print(f"    Done: {len(features)} object ROI embeddings ({features.shape[1]}-dim)")
    return features, labels


# ═══════════════════════════════════════════════════════════════════════
# Annotation helpers
# ═══════════════════════════════════════════════════════════════════════


def get_dominant_class(coco: COCO, img_ids: list) -> dict:
    """
    Returns {img_id: cat_id} where cat_id has the largest total mask area
    across all annotations in that image. Images with no annotations omitted.
    """
    result = {}
    for img_id in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if not anns:
            continue
        area_by_cat: dict[int, float] = {}
        for ann in anns:
            c = ann["category_id"]
            area_by_cat[c] = area_by_cat.get(c, 0.0) + ann.get("area", 0.0)
        result[img_id] = max(area_by_cat, key=area_by_cat.get)
    return result


def get_cat_names(coco: COCO) -> dict:
    """Return {category_id: category_name} from a COCO object."""
    return {c["id"]: c["name"] for c in coco.cats.values()}


# ═══════════════════════════════════════════════════════════════════════
# Projection
# ═══════════════════════════════════════════════════════════════════════


def run_tsne(
    features: np.ndarray, perplexity: float = 40, seed: int = 42
) -> np.ndarray:
    """Project high-dimensional features to 2D using t-SNE (PCA-initialized)."""
    print(
        f"  t-SNE (perplexity={perplexity}) on {features.shape[0]} × {features.shape[1]} ..."
    )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        init="pca",
    )
    return tsne.fit_transform(features)


def run_umap(
    features: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = 42
) -> np.ndarray:
    """Project high-dimensional features to 2D using UMAP."""
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn required: uv add umap-learn")
    print(
        f"  UMAP (n_neighbors={n_neighbors}) on {features.shape[0]} × {features.shape[1]} ..."
    )
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed
    )
    return reducer.fit_transform(features)


def project(features: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Dispatch to t-SNE or UMAP based on method name."""
    return (
        run_tsne(features, **kwargs)
        if method == "tsne"
        else run_umap(features, **kwargs)
    )


# ═══════════════════════════════════════════════════════════════════════
# Silhouette scoring
# ═══════════════════════════════════════════════════════════════════════


def silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
    max_n: int = 2000,
    seed: int = 42,
) -> float | None:
    """
    Silhouette score on raw high-dim features (cosine distance).
    Subsamples to max_n for speed. Returns None if not computable.
    """
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts >= 2]
    if len(valid) < 2:
        return None
    mask = np.isin(labels, valid)
    feats, labs = features[mask], labels[mask]
    if len(feats) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(feats), max_n, replace=False)
        feats, labs = feats[idx], labs[idx]
        if len(np.unique(labs)) < 2:
            return None
    return float(silhouette_score(feats, labs, metric=metric))


# ═══════════════════════════════════════════════════════════════════════
# Color / style
# ═══════════════════════════════════════════════════════════════════════

DATASET_COLORS = {"tc": "#2196F3", "sc": "#FF5722"}
DATASET_LABELS = {"tc": "TrashCan", "sc": "SeaClear"}
MODEL_MARKERS = {"tc": "o", "sc": "s", "pooled": "^"}
MODEL_DISPLAY = {"tc": "TC model", "sc": "SC model", "pooled": "Pooled model"}
ROI_DS_MARKERS = {"tc": "o", "sc": "s"}


def _no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _sil_text(ax, score: float | None):
    if score is not None:
        ax.text(
            0.02,
            0.02,
            f"silhouette: {score:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
            zorder=5,
        )


# ═══════════════════════════════════════════════════════════════════════
# Plot 1 — image-level, colored by dataset
# ═══════════════════════════════════════════════════════════════════════


def plot_image_by_dataset(model_results: dict, method: str, sil: dict, out_path: Path):
    """One panel per model; points colored blue (TC) / orange (SC)."""
    n = len(model_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, model_results.items()):
        ax.scatter(
            d["emb_tc"][:, 0],
            d["emb_tc"][:, 1],
            c=DATASET_COLORS["tc"],
            s=12,
            alpha=0.5,
            linewidths=0,
        )
        ax.scatter(
            d["emb_sc"][:, 0],
            d["emb_sc"][:, 1],
            c=DATASET_COLORS["sc"],
            s=12,
            alpha=0.5,
            linewidths=0,
        )
        ax.set_title(
            f"{MODEL_DISPLAY.get(name, name)} — {method}",
            fontsize=11,
            fontweight="bold",
        )
        _no_ticks(ax)
        _sil_text(ax, sil.get(name))
        ax.legend(
            handles=[
                mpatches.Patch(color=DATASET_COLORS["tc"], label="TrashCan"),
                mpatches.Patch(color=DATASET_COLORS["sc"], label="SeaClear"),
            ],
            fontsize=8,
        )

    fig.suptitle(
        f"Image Features ({method}) — Colored by Dataset  |  Layer4 GAP",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_image_combined(model_results: dict, method: str, out_path: Path):
    """One panel per dataset; marker shape encodes model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ds_key, ax in zip(["tc", "sc"], axes):
        for name, d in model_results.items():
            ax.scatter(
                d[f"emb_{ds_key}"][:, 0],
                d[f"emb_{ds_key}"][:, 1],
                marker=MODEL_MARKERS.get(name, "o"),
                s=15,
                alpha=0.55,
                linewidths=0,
                label=MODEL_DISPLAY.get(name, name),
            )
        ax.set_title(
            f"{DATASET_LABELS[ds_key]} images — {method}",
            fontsize=11,
            fontweight="bold",
        )
        _no_ticks(ax)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Image Features ({method}) — Models Compared  |  Marker = model", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2 — image-level, one plot per class
# ═══════════════════════════════════════════════════════════════════════


def plot_image_per_class(
    model_results: dict,
    dominant_tc: dict,
    dominant_sc: dict,
    cat_id: int,
    cat_name: str,
    method: str,
    sil: dict,
    out_path: Path,
):
    """
    One panel per model for a single class. Foreground = images whose dominant
    class is cat_id, colored TC blue (○) / SC orange (□). All other images shown
    as faint gray context so the viewer can see where this class sits globally.
    sil[model_name] = per-class TC/SC silhouette score.
    """
    n = len(model_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, model_results.items()):
        # Context: all images in gray
        ax.scatter(
            d["emb_tc"][:, 0],
            d["emb_tc"][:, 1],
            c="#cccccc",
            s=6,
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )
        ax.scatter(
            d["emb_sc"][:, 0],
            d["emb_sc"][:, 1],
            c="#cccccc",
            s=6,
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )

        # Foreground: this class only
        tc_idx = [
            i
            for i, img_id in enumerate(d["img_ids_tc"])
            if dominant_tc.get(img_id) == cat_id
        ]
        sc_idx = [
            i
            for i, img_id in enumerate(d["img_ids_sc"])
            if dominant_sc.get(img_id) == cat_id
        ]

        if tc_idx:
            pts = d["emb_tc"][tc_idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=DATASET_COLORS["tc"],
                marker="o",
                s=30,
                alpha=0.85,
                linewidths=0,
                zorder=3,
                label=f"TrashCan (n={len(tc_idx)})",
            )
        if sc_idx:
            pts = d["emb_sc"][sc_idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=DATASET_COLORS["sc"],
                marker="s",
                s=30,
                alpha=0.85,
                linewidths=0,
                zorder=3,
                label=f"SeaClear (n={len(sc_idx)})",
            )

        ax.set_title(
            f"{MODEL_DISPLAY.get(name, name)} — {method}",
            fontsize=11,
            fontweight="bold",
        )
        _no_ticks(ax)
        _sil_text(ax, sil.get(name))
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Image Features ({method}) — Class: {cat_name}  |  Shape = dataset (○TC □SC)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3 — per-object ROI features, one plot per class
# ═══════════════════════════════════════════════════════════════════════


def plot_roi_per_class(
    roi_results: dict,
    cat_id: int,
    cat_name: str,
    method: str,
    sil: dict,
    out_path: Path,
):
    """
    One panel per model for a single class. Foreground = objects of cat_id,
    colored TC blue (○) / SC orange (□). All other objects shown as gray context.
    sil[model_name] = per-class TC/SC silhouette score.
    """
    n = len(roi_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, roi_results.items()):
        # Context: all objects in gray
        ax.scatter(
            d["emb_tc"][:, 0],
            d["emb_tc"][:, 1],
            c="#cccccc",
            s=5,
            alpha=0.15,
            linewidths=0,
            zorder=1,
        )
        ax.scatter(
            d["emb_sc"][:, 0],
            d["emb_sc"][:, 1],
            c="#cccccc",
            s=5,
            alpha=0.15,
            linewidths=0,
            zorder=1,
        )

        # Foreground: this class only
        tc_mask = d["labels_tc"] == cat_id
        sc_mask = d["labels_sc"] == cat_id

        if tc_mask.any():
            pts = d["emb_tc"][tc_mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=DATASET_COLORS["tc"],
                marker="o",
                s=20,
                alpha=0.75,
                linewidths=0,
                zorder=3,
                label=f"TrashCan (n={tc_mask.sum()})",
            )
        if sc_mask.any():
            pts = d["emb_sc"][sc_mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=DATASET_COLORS["sc"],
                marker="s",
                s=20,
                alpha=0.75,
                linewidths=0,
                zorder=3,
                label=f"SeaClear (n={sc_mask.sum()})",
            )

        ax.set_title(
            f"{MODEL_DISPLAY.get(name, name)} — {method}",
            fontsize=11,
            fontweight="bold",
        )
        _no_ticks(ax)
        _sil_text(ax, sil.get(name))
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Per-Object ROI Features ({method}) — Class: {cat_name}  |  (○TC □SC)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5 — silhouette summary bar chart
# ═══════════════════════════════════════════════════════════════════════


def plot_silhouette_summary(
    per_class_sil: dict,
    cat_names: dict,
    feature_type: str,
    method: str,
    out_path: Path,
):
    """
    Bar chart: x = class, y = TC/SC silhouette score, grouped by model.
    per_class_sil[model_name][cat_id] = score (or None).
    """
    model_names = list(per_class_sil.keys())
    all_cats = sorted({c for scores in per_class_sil.values() for c in scores})
    if not all_cats:
        return

    x = np.arange(len(all_cats))
    width = 0.8 / max(len(model_names), 1)
    cmap = plt.get_cmap("tab10", len(model_names))

    fig, ax = plt.subplots(figsize=(max(10, len(all_cats) * 0.7), 5))
    for i, name in enumerate(model_names):
        scores = [per_class_sil[name].get(c) for c in all_cats]
        vals = [s if s is not None else 0.0 for s in scores]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=MODEL_DISPLAY.get(name, name),
            color=cmap(i),
            alpha=0.8,
        )
        # Hatch bars where score was None (insufficient data)
        for bar, s in zip(bars, scores):
            if s is None:
                bar.set_hatch("///")
                bar.set_alpha(0.3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [cat_names.get(c, f"id_{c}") for c in all_cats],
        rotation=40,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Silhouette score (TC vs SC)")
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    ax.set_title(
        f"Per-Class TC/SC Dataset Separability — {feature_type} Features ({method})\n"
        "Higher = TC and SC more separated in feature space for that class",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4 — per-object ROI features, colored by dataset
# ═══════════════════════════════════════════════════════════════════════


def plot_roi_by_dataset(roi_results: dict, method: str, sil: dict, out_path: Path):
    """One panel per model; points colored blue (TC) / orange (SC)."""
    n = len(roi_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, roi_results.items()):
        ax.scatter(
            d["emb_tc"][:, 0],
            d["emb_tc"][:, 1],
            c=DATASET_COLORS["tc"],
            s=8,
            alpha=0.4,
            linewidths=0,
            label="TrashCan",
        )
        ax.scatter(
            d["emb_sc"][:, 0],
            d["emb_sc"][:, 1],
            c=DATASET_COLORS["sc"],
            s=8,
            alpha=0.4,
            linewidths=0,
            label="SeaClear",
        )
        ax.set_title(
            f"{MODEL_DISPLAY.get(name, name)} — {method}",
            fontsize=11,
            fontweight="bold",
        )
        _no_ticks(ax)
        _sil_text(ax, sil.get(name))
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Per-Object ROI Features ({method}) — Colored by Dataset  |  Layer4 ROI-Align",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def parse_model_spec(spec: str):
    """Parse 'name:checkpoint:config:source_dataset'."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"Model spec must be 'name:ckpt:config:source', got: {spec!r}")
    name, ckpt, cfg_path, source = parts
    return name, Path(ckpt), Path(cfg_path), source


def main():
    """CLI entry point: extract features, compute projections, and generate all plots."""
    parser = argparse.ArgumentParser(
        description="Visualize backbone feature spaces for cross-dataset analysis."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME:CKPT:CONFIG:SOURCE",
        help="Model specs: name:checkpoint:config:source_dataset",
    )
    parser.add_argument(
        "--tc-ann",
        nargs="+",
        help="TrashCan annotation JSON(s) in TC20 space (e.g. val + test)",
    )
    parser.add_argument("--tc-img", help="TrashCan image directory")
    parser.add_argument(
        "--sc-ann",
        nargs="+",
        help="SeaClear annotation JSON(s) in TC20 space (e.g. val + test)",
    )
    parser.add_argument("--sc-img", help="SeaClear image directory")
    parser.add_argument("--output-dir", default="feature_plots")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tsne", "umap"],
        choices=["tsne", "umap"],
    )
    parser.add_argument("--tsne-perplexity", type=float, default=40)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument(
        "--skip-roi",
        action="store_true",
        help="Skip per-object ROI feature extraction (faster, image-level only)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip extraction; load saved .npz embeddings and re-plot",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap images per dataset (for quick testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Annotation objects (always needed for category names / dominant class) ──
    if not all([args.tc_ann, args.sc_ann]):
        parser.error("--tc-ann and --sc-ann are required (for category names)")
    tc_coco = merge_cocos(args.tc_ann)
    sc_coco = merge_cocos(args.sc_ann)
    cat_names = {**get_cat_names(tc_coco), **get_cat_names(sc_coco)}

    # ── Raw embeddings storage ───────────────────────────────────────────────
    # image_data[name] = {"tc": (N,2048), "img_ids_tc": [...], "sc": ..., "img_ids_sc": ...}
    # roi_data[name]   = {"tc": (M,2048), "labels_tc": (M,), "sc": ..., "labels_sc": ...}
    image_data: dict = {}
    roi_data: dict = {}

    if not args.plot_only:
        if not args.models:
            parser.error("--models required unless --plot-only")
        if not all([args.tc_img, args.sc_img]):
            parser.error("--tc-img and --sc-img required unless --plot-only")

        # Optionally cap images
        tc_img_ids = tc_coco.getImgIds()
        sc_img_ids = sc_coco.getImgIds()
        if args.max_images:
            tc_img_ids = tc_img_ids[: args.max_images]
            sc_img_ids = sc_img_ids[: args.max_images]
            tc_coco.imgs = {i: tc_coco.imgs[i] for i in tc_img_ids}
            sc_coco.imgs = {i: sc_coco.imgs[i] for i in sc_img_ids}

        for spec in args.models:
            name, ckpt_path, cfg_path, source = parse_model_spec(spec)
            print(f"\n{'='*60}")
            print(f"Model: {name}  ({source})")
            print(f"{'='*60}")

            cfg = load_config(cfg_path)
            model = build_model(cfg)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = {
                k.replace("_orig_mod.", ""): v
                for k, v in ckpt["model_state_dict"].items()
            }
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            # ── Image-level ────────────────────────────────────────────────
            print(f"\n  [image-level] TC ({len(tc_coco.getImgIds())} images)...")
            feats_tc, ids_tc = extract_image_features(
                model, tc_coco, Path(args.tc_img), args.batch_size
            )
            print(f"  [image-level] SC ({len(sc_coco.getImgIds())} images)...")
            feats_sc, ids_sc = extract_image_features(
                model, sc_coco, Path(args.sc_img), args.batch_size
            )

            image_data[name] = {
                "tc": feats_tc,
                "img_ids_tc": ids_tc,
                "sc": feats_sc,
                "img_ids_sc": ids_sc,
            }
            np.savez(
                out_dir / f"img_emb_{name}.npz",
                tc=feats_tc,
                img_ids_tc=ids_tc,
                sc=feats_sc,
                img_ids_sc=ids_sc,
            )
            print(f"  Saved: img_emb_{name}.npz")

            # ── ROI-level ──────────────────────────────────────────────────
            if not args.skip_roi:
                print("\n  [roi-level] TC ...")
                roi_tc, labs_tc = extract_roi_features(
                    model, tc_coco, Path(args.tc_img), args.batch_size
                )
                print("  [roi-level] SC ...")
                roi_sc, labs_sc = extract_roi_features(
                    model, sc_coco, Path(args.sc_img), args.batch_size
                )

                roi_data[name] = {
                    "tc": roi_tc,
                    "labels_tc": labs_tc,
                    "sc": roi_sc,
                    "labels_sc": labs_sc,
                }
                np.savez(
                    out_dir / f"roi_emb_{name}.npz",
                    tc=roi_tc,
                    labels_tc=labs_tc,
                    sc=roi_sc,
                    labels_sc=labs_sc,
                )
                print(f"  Saved: roi_emb_{name}.npz")

            del model
            torch.cuda.empty_cache()

    else:
        # Load saved image embeddings
        for f in sorted(out_dir.glob("img_emb_*.npz")):
            name = f.stem.replace("img_emb_", "")
            d = np.load(f)
            image_data[name] = {
                "tc": d["tc"],
                "img_ids_tc": d["img_ids_tc"].tolist(),
                "sc": d["sc"],
                "img_ids_sc": d["img_ids_sc"].tolist(),
            }
            print(
                f"Loaded image embeddings '{name}': TC={d['tc'].shape}, SC={d['sc'].shape}"
            )
        # Load saved ROI embeddings if present
        for f in sorted(out_dir.glob("roi_emb_*.npz")):
            name = f.stem.replace("roi_emb_", "")
            d = np.load(f)
            roi_data[name] = {
                "tc": d["tc"],
                "labels_tc": d["labels_tc"],
                "sc": d["sc"],
                "labels_sc": d["labels_sc"],
            }
            print(
                f"Loaded ROI embeddings '{name}': TC={d['tc'].shape}, SC={d['sc'].shape}"
            )
        if not image_data:
            raise FileNotFoundError(f"No img_emb_*.npz found in {out_dir}")

    print(f"\nModels with image embeddings: {list(image_data.keys())}")
    print(f"Models with ROI embeddings:   {list(roi_data.keys())}")

    # ── Dominant class per image ─────────────────────────────────────────────
    # Use first model's img_ids (all models use the same images)
    first = next(iter(image_data.values()))
    dominant_tc = get_dominant_class(tc_coco, first["img_ids_tc"])
    dominant_sc = get_dominant_class(sc_coco, first["img_ids_sc"])

    # ── Silhouette on raw features ───────────────────────────────────────────
    print("\n=== Silhouette scores (cosine, raw 2048-dim) ===")
    all_scores = {}

    for name, d in image_data.items():
        combined = np.concatenate([d["tc"], d["sc"]], axis=0)
        ds_labels = np.array([0] * len(d["tc"]) + [1] * len(d["sc"]))
        sil_ds = silhouette(combined, ds_labels)

        # Class labels via dominant class
        img_ids_all = d["img_ids_tc"] + d["img_ids_sc"]
        dom_all = {**dominant_tc, **dominant_sc}
        class_labels = np.array([dom_all.get(i, -1) for i in img_ids_all])
        valid = class_labels >= 0
        sil_cls = (
            silhouette(combined[valid], class_labels[valid]) if valid.any() else None
        )

        print(
            f"  {name}: dataset_sep={sil_ds:.3f}"
            + (f", class_sep={sil_cls:.3f}" if sil_cls else "")
            + "  [image-level]"
        )
        all_scores[f"image_{name}"] = {"dataset": sil_ds, "class": sil_cls}

    for name, d in roi_data.items():
        combined = np.concatenate([d["tc"], d["sc"]], axis=0)
        ds_labels = np.array([0] * len(d["tc"]) + [1] * len(d["sc"]))
        cls_labels = np.concatenate([d["labels_tc"], d["labels_sc"]])
        sil_ds = silhouette(combined, ds_labels)
        sil_cls = silhouette(combined, cls_labels)
        print(
            f"  {name}: dataset_sep={sil_ds:.3f}"
            + (f", class_sep={sil_cls:.3f}" if sil_cls else "")
            + "  [ROI-level]"
        )
        all_scores[f"roi_{name}"] = {"dataset": sil_ds, "class": sil_cls}

    with open(out_dir / "silhouette_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    print("  Saved: silhouette_scores.json")

    # ── Projection & plotting ────────────────────────────────────────────────
    print("\n=== Projecting & plotting ===")

    proj_kwargs = {
        "tsne": {"perplexity": args.tsne_perplexity},
        "umap": {"n_neighbors": args.umap_neighbors},
    }

    for method in args.methods:
        ml = "t-SNE" if method == "tsne" else "UMAP"
        print(f"\n--- {ml} ---")

        # ── Image-level projections ──────────────────────────────────────
        img_proj: dict = {}  # name → {emb_tc, img_ids_tc, emb_sc, img_ids_sc}
        for name, d in image_data.items():
            combined = np.concatenate([d["tc"], d["sc"]], axis=0)
            n_tc = len(d["tc"])
            proj = project(combined, method, **proj_kwargs[method])
            img_proj[name] = {
                "emb_tc": proj[:n_tc],
                "img_ids_tc": d["img_ids_tc"],
                "emb_sc": proj[n_tc:],
                "img_ids_sc": d["img_ids_sc"],
            }
            np.savez(
                out_dir / f"proj_img_{method}_{name}.npz",
                emb_tc=proj[:n_tc],
                emb_sc=proj[n_tc:],
            )

        img_ds_sil = {name: all_scores[f"image_{name}"]["dataset"] for name in img_proj}

        # Plot 1a — image-level, dataset-colored, per-model
        plot_image_by_dataset(
            img_proj, ml, img_ds_sil, out_dir / f"{method}_image_by_dataset.png"
        )
        # Plot 1b — image-level, dataset-colored, combined
        plot_image_combined(img_proj, ml, out_dir / f"{method}_image_combined.png")

        # Plot 2 — image-level, one plot per class
        # Per-class TC/SC silhouette: {model_name: {cat_id: score}}
        img_per_class_sil: dict = {name: {} for name in img_proj}
        all_dom_cats = sorted(set(dominant_tc.values()) | set(dominant_sc.values()))
        print(f"  Generating per-class image plots ({len(all_dom_cats)} classes)...")
        img_class_dir = out_dir / f"{method}_image_per_class"
        img_class_dir.mkdir(exist_ok=True)
        for cat_id in all_dom_cats:
            cat_name = cat_names.get(cat_id, f"id_{cat_id}")
            per_model_sil = {}
            for name, d in img_proj.items():
                tc_idx = np.array(
                    [
                        i
                        for i, img_id in enumerate(d["img_ids_tc"])
                        if dominant_tc.get(img_id) == cat_id
                    ]
                )
                sc_idx = np.array(
                    [
                        i
                        for i, img_id in enumerate(d["img_ids_sc"])
                        if dominant_sc.get(img_id) == cat_id
                    ]
                )
                if len(tc_idx) >= 2 and len(sc_idx) >= 2:
                    feats = np.concatenate(
                        [
                            image_data[name]["tc"][tc_idx],
                            image_data[name]["sc"][sc_idx],
                        ]
                    )
                    labs = np.array([0] * len(tc_idx) + [1] * len(sc_idx))
                    per_model_sil[name] = silhouette(feats, labs)
                img_per_class_sil[name][cat_id] = per_model_sil.get(name)
            safe_name = cat_name.replace("/", "_").replace(" ", "_")
            plot_image_per_class(
                img_proj,
                dominant_tc,
                dominant_sc,
                cat_id,
                cat_name,
                ml,
                per_model_sil,
                img_class_dir / f"{safe_name}.png",
            )

        # Silhouette summary bar chart (image-level)
        plot_silhouette_summary(
            img_per_class_sil,
            cat_names,
            "Image",
            ml,
            out_dir / f"{method}_image_silhouette_summary.png",
        )

        # ── ROI-level projections ────────────────────────────────────────
        if roi_data:
            roi_proj: dict = {}
            for name, d in roi_data.items():
                combined = np.concatenate([d["tc"], d["sc"]], axis=0)
                n_tc = len(d["tc"])
                proj = project(combined, method, **proj_kwargs[method])
                roi_proj[name] = {
                    "emb_tc": proj[:n_tc],
                    "labels_tc": d["labels_tc"],
                    "emb_sc": proj[n_tc:],
                    "labels_sc": d["labels_sc"],
                }
                np.savez(
                    out_dir / f"proj_roi_{method}_{name}.npz",
                    emb_tc=proj[:n_tc],
                    labels_tc=d["labels_tc"],
                    emb_sc=proj[n_tc:],
                    labels_sc=d["labels_sc"],
                )

            roi_ds_sil = {
                name: all_scores[f"roi_{name}"]["dataset"] for name in roi_proj
            }

            # Plot 3 — ROI, one plot per class
            all_roi_cats = sorted(
                set(
                    int(c)
                    for d in roi_data.values()
                    for c in list(d["labels_tc"]) + list(d["labels_sc"])
                )
            )
            roi_per_class_sil: dict = {name: {} for name in roi_proj}
            print(f"  Generating per-class ROI plots ({len(all_roi_cats)} classes)...")
            roi_class_dir = out_dir / f"{method}_roi_per_class"
            roi_class_dir.mkdir(exist_ok=True)
            for cat_id in all_roi_cats:
                cat_name = cat_names.get(cat_id, f"id_{cat_id}")
                per_model_sil = {}
                for name, d in roi_proj.items():
                    tc_mask = roi_data[name]["labels_tc"] == cat_id
                    sc_mask = roi_data[name]["labels_sc"] == cat_id
                    if tc_mask.sum() >= 2 and sc_mask.sum() >= 2:
                        feats = np.concatenate(
                            [
                                roi_data[name]["tc"][tc_mask],
                                roi_data[name]["sc"][sc_mask],
                            ]
                        )
                        labs = np.array(
                            [0] * int(tc_mask.sum()) + [1] * int(sc_mask.sum())
                        )
                        per_model_sil[name] = silhouette(feats, labs)
                    roi_per_class_sil[name][cat_id] = per_model_sil.get(name)
                safe_name = cat_name.replace("/", "_").replace(" ", "_")
                plot_roi_per_class(
                    roi_proj,
                    cat_id,
                    cat_name,
                    ml,
                    per_model_sil,
                    roi_class_dir / f"{safe_name}.png",
                )

            # Silhouette summary bar chart (ROI-level)
            plot_silhouette_summary(
                roi_per_class_sil,
                cat_names,
                "ROI",
                ml,
                out_dir / f"{method}_roi_silhouette_summary.png",
            )

            # Plot 4 — ROI, dataset-colored (whole-dataset view)
            plot_roi_by_dataset(
                roi_proj, ml, roi_ds_sil, out_dir / f"{method}_roi_by_dataset.png"
            )

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
