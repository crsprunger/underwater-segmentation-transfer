"""
Visualize best and worst predictions from cross-dataset evaluation.

For each eval cell (TC->TC, TC->SC, SC->TC, SC->SC, Pooled->TC, Pooled->SC),
shows side-by-side GT vs prediction overlays for the highest- and lowest-scoring
images. Per-image score is match rate: fraction of GT objects matched by a
same-class prediction at IoU > 0.5.

Uses the same model loading and inference pipeline as cross_dataset_eval.py.
Category space remapping is handled generically via src.category_groups.

Usage:
    # Using --model-name (auto-resolves checkpoint/config, auto-detects source space)
    python visualize_cross_dataset.py \
        --model-name trashcan_chunksplit \
        --target-ann trashcan_data/dataset/instance_version/trashcan_chunksplit_test_tc20.json \
        --target-img trashcan_data/dataset/instance_version/images \
        --out-dir viz/cross_dataset/tc_on_tc \
        --top-k 10 --bottom-k 5 --min-gap 15

    # Cross-domain: TC model on SC test
    python visualize_cross_dataset.py \
        --model-name trashcan_chunksplit \
        --target-ann seaclear_data/seaclear_480p_chunksplit_test_tc20.json \
        --target-img seaclear_data/images_480p \
        --out-dir viz/cross_dataset/tc_on_sc \
        --top-k 10 --bottom-k 5 --min-gap 10

    # With explicit checkpoint/config (legacy style)
    python visualize_cross_dataset.py \
        --checkpoint checkpoints/seaclear_chunksplit/model6v2/best_model.pth \
        --config checkpoints/seaclear_chunksplit/model6v2/training_config_full.yaml \
        --source-dataset seaclear \
        --target-ann trashcan_data/dataset/instance_version/trashcan_chunksplit_test_tc20.json \
        --target-img trashcan_data/dataset/instance_version/images \
        --out-dir viz/cross_dataset/sc_on_tc \
        --top-k 10 --bottom-k 5 --min-gap 10

    # With CLAHE preprocessing
    python visualize_cross_dataset.py \
        --model-name trashcan_chunksplit \
        --target-ann seaclear_data/seaclear_480p_chunksplit_test_tc20.json \
        --target-img seaclear_data/images_480p \
        --preprocess clahe \
        --out-dir viz/cross_dataset/tc_on_sc_clahe \
        --top-k 10 --bottom-k 5 --min-gap 10
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torchvision
from torchvision.io import decode_image
from pycocotools.coco import COCO

from src.model import build_model
from src.train import load_config
from src.category_groups import (
    get_scheme,
    detect_source_space,
    coarsest_common_space,
)
from evaluation.cross_dataset_eval import PREPROCESS_FNS
from registry import MODEL_VERSION


# ═══════════════════════════════════════════════════════════════════════
# Per-image scoring
# ═══════════════════════════════════════════════════════════════════════


def compute_iou_matrix(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of binary masks.

    Args:
        masks_a: (N, H, W) uint8
        masks_b: (M, H, W) uint8

    Returns:
        (N, M) IoU matrix
    """
    N, H, W = masks_a.shape
    M = masks_b.shape[0]
    a = masks_a.reshape(N, -1).astype(bool)
    b = masks_b.reshape(M, -1).astype(bool)
    # (N, M) intersection and union
    inter = (a[:, None, :] & b[None, :, :]).sum(axis=2)
    union = (a[:, None, :] | b[None, :, :]).sum(axis=2)
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def score_image(
    gt_masks,
    gt_labels,
    pred_masks,
    pred_labels,
    pred_scores,
    iou_thresh=0.5,
    score_thresh=0.3,
):
    """Score a single image by GT match rate.

    Returns:
        match_rate: float in [0, 1] (fraction of GT objects matched)
        n_gt: number of GT objects
        n_pred: number of predictions above score_thresh
        n_matched: number of GT objects matched
        n_fp: number of unmatched predictions (false positives)
    """
    # Filter predictions by confidence
    keep = pred_scores >= score_thresh
    pred_masks = pred_masks[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    if n_gt == 0:
        return (1.0 if n_pred == 0 else 0.0), 0, n_pred, 0, n_pred

    if n_pred == 0:
        return 0.0, n_gt, 0, 0, 0

    iou = compute_iou_matrix(gt_masks, pred_masks)  # (n_gt, n_pred)

    matched_gt = set()
    matched_pred = set()

    # Greedy matching: iterate over GT objects, find best unmatched prediction
    for gi in range(n_gt):
        best_iou = 0.0
        best_pi = -1
        for pi in range(n_pred):
            if pi in matched_pred:
                continue
            if gt_labels[gi] != pred_labels[pi]:
                continue
            if iou[gi, pi] > best_iou:
                best_iou = iou[gi, pi]
                best_pi = pi
        if best_iou >= iou_thresh and best_pi >= 0:
            matched_gt.add(gi)
            matched_pred.add(best_pi)

    n_matched = len(matched_gt)
    n_fp = n_pred - len(matched_pred)
    match_rate = n_matched / n_gt

    return match_rate, n_gt, n_pred, n_matched, n_fp


# ═══════════════════════════════════════════════════════════════════════
# Inference (per-image, keeping masks for visualization)
# ═══════════════════════════════════════════════════════════════════════


def remap_per_image_labels(
    pred_labels: np.ndarray,
    pred_masks: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    mapping: dict[int, int],
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remap prediction labels using a category mapping, dropping unmapped ones.

    This is the per-image analogue of remap_results() from src.category_groups,
    but operates on raw numpy arrays instead of COCO-format result dicts.
    """
    if not len(pred_labels):
        return pred_labels, pred_masks, pred_boxes, pred_scores

    keep = []
    remapped = []
    for j, cat_id in enumerate(pred_labels):
        new_id = mapping.get(int(cat_id))
        if new_id is not None:
            keep.append(j)
            remapped.append(new_id)

    if keep:
        keep = np.array(keep)
        return (
            np.array(remapped, dtype=np.int64),
            pred_masks[keep],
            pred_boxes[keep],
            pred_scores[keep],
        )
    return (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, orig_h, orig_w), dtype=np.uint8),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


@torch.no_grad()
def run_inference_per_image(
    model,
    coco_gt: COCO,
    img_dir: Path,
    source_space: str,
    eval_space: str,
    score_thresh: float = 0.3,
    preprocess: str = "none",
):
    """Run inference and return per-image predictions + GT, remapped to eval_space.

    Args:
        model: loaded Mask R-CNN model
        coco_gt: COCO object for GT (already remapped to eval_space)
        img_dir: path to image directory
        source_space: category space of model predictions
        eval_space: target category space for evaluation
        score_thresh: minimum prediction score (not applied here, just stored)
        preprocess: test-time preprocessing key

    Returns:
        list of dicts, one per image:
            img_id, file_name, image_tensor (C,H,W float),
            gt_masks (N,H,W uint8), gt_labels (N,), gt_boxes (N,4),
            pred_masks (M,H,W uint8), pred_labels (M,), pred_boxes (M,4), pred_scores (M,)
    """
    device = next(model.parameters()).device
    model.eval()
    img_ids = coco_gt.getImgIds()
    results = []

    # Build remapping (if source != eval space)
    pred_mapping = None
    if source_space != eval_space:
        scheme = get_scheme(eval_space, source=source_space)
        pred_mapping = scheme["mapping"]

    preprocess_fn = PREPROCESS_FNS.get(preprocess)

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            img_path = img_dir / Path(img_info["file_name"]).name

        img = decode_image(str(img_path), mode=torchvision.io.ImageReadMode.RGB)
        img_float = img.float() / 255.0

        # Apply test-time preprocessing
        if preprocess_fn is not None:
            img_input = preprocess_fn(img_float)
        else:
            img_input = img_float

        # GT (already in eval_space via remapped coco_gt)
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_masks_list = []
        gt_labels_list = []
        gt_boxes_list = []
        for ann in anns:
            m = coco_gt.annToMask(ann)
            gt_masks_list.append(m)
            gt_labels_list.append(ann["category_id"])
            x, y, w, h = ann["bbox"]
            gt_boxes_list.append([x, y, x + w, y + h])

        gt_masks = (
            np.array(gt_masks_list, dtype=np.uint8)
            if gt_masks_list
            else np.zeros((0, img_info["height"], img_info["width"]), dtype=np.uint8)
        )
        gt_labels = np.array(gt_labels_list, dtype=np.int64)
        gt_boxes = (
            np.array(gt_boxes_list, dtype=np.float32).reshape(-1, 4)
            if gt_boxes_list
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Inference
        output = model([img_input.to(device)])[0]
        pred_boxes = output["boxes"].cpu().numpy()
        pred_scores_raw = output["scores"].cpu().numpy()
        pred_labels_raw = output["labels"].cpu().numpy().astype(np.int64)
        masks_tensor = output["masks"]

        orig_h, orig_w = img_info["height"], img_info["width"]
        if masks_tensor.shape[-2] != orig_h or masks_tensor.shape[-1] != orig_w:
            masks_tensor = torch.nn.functional.interpolate(
                masks_tensor,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        pred_masks_raw = (masks_tensor[:, 0] > 0.5).to(torch.uint8).cpu().numpy()

        # Remap predictions to eval space
        if pred_mapping is not None:
            pred_labels_raw, pred_masks_raw, pred_boxes, pred_scores_raw = (
                remap_per_image_labels(
                    pred_labels_raw,
                    pred_masks_raw,
                    pred_boxes,
                    pred_scores_raw,
                    pred_mapping,
                    orig_h,
                    orig_w,
                )
            )

        results.append(
            {
                "img_id": img_id,
                "file_name": img_info["file_name"],
                "image": img_float,  # (C, H, W) — original image for display
                "gt_masks": gt_masks,
                "gt_labels": gt_labels,
                "gt_boxes": gt_boxes,
                "pred_masks": pred_masks_raw,
                "pred_labels": pred_labels_raw,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores_raw,
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(img_ids)} images ...")

    print(f"  Inference done: {len(results)} images")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════


def draw_overlay(
    ax,
    image_tensor,
    masks,
    labels,
    boxes,
    cat_names,
    scores=None,
    alpha=0.4,
    score_thresh=0.3,
):
    """Draw image with mask + box overlays.

    Args:
        image_tensor: (C, H, W) float [0,1]
        masks: (N, H, W) uint8
        labels: (N,) int
        boxes: (N, 4) float [x1, y1, x2, y2]
        cat_names: dict {cat_id: name}
        scores: (N,) float or None (for GT, no scores)
        alpha: mask transparency
        score_thresh: only show predictions above this score
    """
    img = image_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)

    n = len(labels)
    if n == 0:
        return

    for i in range(n):
        if scores is not None and scores[i] < score_thresh:
            continue

        x1, y1, x2, y2 = boxes[i]
        cat_id = int(labels[i])
        name = cat_names.get(cat_id, f"id_{cat_id}")
        color = plt.cm.tab20(cat_id % 20)

        # Label text
        if scores is not None:
            label_text = f"{name} {scores[i]:.2f}"
        else:
            label_text = name

        # Box
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 2,
            label_text,
            fontsize=6,
            color=color,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5),
        )

        # Mask overlay
        if i < len(masks):
            m = masks[i]
            if m.sum() > 0:
                mask_rgba = np.zeros((*m.shape, 4))
                mask_rgba[..., :3] = color[:3]
                mask_rgba[..., 3] = m * alpha
                ax.imshow(mask_rgba)


def save_grid(image_infos, cat_names, out_path, title, score_thresh=0.3):
    """Save a grid of GT vs prediction visualizations.

    Args:
        image_infos: list of dicts from run_inference_per_image, with added 'match_rate' key
        cat_names: dict {cat_id: name}
        out_path: Path to save the figure
        title: figure title
        score_thresh: prediction confidence threshold for display
    """
    n = len(image_infos)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n), layout="compressed")
    if n == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for row, info in enumerate(image_infos):
        # GT (left)
        ax_gt = axes[row, 0]
        draw_overlay(
            ax_gt,
            info["image"],
            info["gt_masks"],
            info["gt_labels"],
            info["gt_boxes"],
            cat_names,
            scores=None,
            alpha=0.4,
        )
        ax_gt.set_title(
            f"GT — {info['file_name']}  (objects: {len(info['gt_labels'])})", fontsize=9
        )
        ax_gt.axis("off")

        # Predictions (right)
        ax_pred = axes[row, 1]
        draw_overlay(
            ax_pred,
            info["image"],
            info["pred_masks"],
            info["pred_labels"],
            info["pred_boxes"],
            cat_names,
            scores=info["pred_scores"],
            alpha=0.4,
            score_thresh=score_thresh,
        )
        n_pred_shown = (
            int((info["pred_scores"] >= score_thresh).sum())
            if len(info["pred_scores"]) > 0
            else 0
        )
        ax_pred.set_title(
            f"Pred — match: {info['match_rate']:.0%}, "
            f"matched: {info['n_matched']}/{info['n_gt']} GT, "
            f"preds: {n_pred_shown} (score>{score_thresh})",
            fontsize=9,
        )
        ax_pred.axis("off")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point: run inference, rank images by IoU, and save best/worst grids."""
    parser = argparse.ArgumentParser(
        description="Visualize best/worst cross-dataset predictions."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to training_config.yaml"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model directory name under checkpoints/ (alternative to --checkpoint/--config)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=MODEL_VERSION,
        help=f"Model version subdirectory (default: {MODEL_VERSION})",
    )
    parser.add_argument(
        "--source-dataset",
        default=None,
        choices=["trashcan", "seaclear", "pooled", "tc20"],
        help="Category space of model predictions. Auto-detected if not specified.",
    )
    parser.add_argument("--target-ann", required=True)
    parser.add_argument("--target-img", required=True)
    parser.add_argument("--out-dir", required=True, help="Output directory for figures")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of best images to visualize"
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=None,
        help="Number of worst images to visualize (defaults to --top-k)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.3,
        help="Prediction confidence threshold for display and scoring",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for matching GT to predictions",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=0,
        help="Minimum distance (in lexicographic filename order) between "
        "selected examples. E.g., 50 ensures at most one frame per chunk. "
        "0 disables (default).",
    )
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=["none", "clahe", "histogram_eq", "grayscale"],
        help="Test-time preprocessing: 'clahe' (CLAHE on L channel), "
        "'histogram_eq' (histogram equalization on L channel), "
        "'grayscale' (convert to grayscale), or 'none' (default)",
    )
    args = parser.parse_args()

    # Resolve checkpoint/config from --model-name if needed
    if args.config is None or args.checkpoint is None:
        if args.model_name is None:
            parser.error(
                "Either --config and --checkpoint or --model-name must be provided"
            )
        args.config = str(
            Path("checkpoints")
            / args.model_name
            / args.model_version
            / "training_config_full.yaml"
        )
        args.checkpoint = str(
            Path("checkpoints")
            / args.model_name
            / args.model_version
            / "best_model.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint} ...")
    cfg = load_config(Path(args.config))
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Auto-detect or validate source category space
    if args.source_dataset is None:
        train_ann = cfg.train_ann_path
        if not train_ann or not Path(train_ann).exists():
            parser.error(
                "Cannot auto-detect source category space from training config. "
                "Use --source-dataset to specify manually."
            )
        train_coco = COCO(train_ann)
        args.source_dataset = detect_source_space(train_coco)
        print(f"  Auto-detected source category space: {args.source_dataset}")
        del train_coco

    print(f"  Source dataset: {args.source_dataset}")

    # Load target annotations and detect eval space
    print(f"\nLoading target annotations: {args.target_ann}")
    coco_gt_raw = COCO(args.target_ann)
    gt_space = detect_source_space(coco_gt_raw)
    eval_space = coarsest_common_space(args.source_dataset, gt_space)
    print(f"  {len(coco_gt_raw.getImgIds())} images")
    print(f"  GT space: {gt_space} → eval space: {eval_space}")

    # Remap GT to eval space if needed
    if gt_space == eval_space:
        coco_gt = coco_gt_raw
    else:
        from src.category_groups import remap_coco_gt

        gt_scheme = get_scheme(eval_space, source=gt_space)
        coco_gt = remap_coco_gt(
            coco_gt_raw, gt_scheme["mapping"], gt_scheme["categories"]
        )
        print(f"  Remapped GT: {gt_space} → {eval_space}")

    # Category names for display (from eval-space GT)
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    # Run inference
    preprocess_label = (
        f" (preprocess: {args.preprocess})" if args.preprocess != "none" else ""
    )
    print(f"\nRunning inference{preprocess_label} ...")
    per_image = run_inference_per_image(
        model,
        coco_gt,
        Path(args.target_img),
        args.source_dataset,
        eval_space,
        args.score_thresh,
        preprocess=args.preprocess,
    )

    # Score each image
    print("\nScoring images ...")
    has_gt = []
    for info in per_image:
        match_rate, n_gt, n_pred, n_matched, n_fp = score_image(
            info["gt_masks"],
            info["gt_labels"],
            info["pred_masks"],
            info["pred_labels"],
            info["pred_scores"],
            iou_thresh=args.iou_thresh,
            score_thresh=args.score_thresh,
        )
        info["match_rate"] = match_rate
        info["n_gt"] = n_gt
        info["n_pred"] = n_pred
        info["n_matched"] = n_matched
        info["n_fp"] = n_fp
        if n_gt > 0:
            has_gt.append(info)

    print(f"  {len(has_gt)} images with GT objects (out of {len(per_image)} total)")

    # Build filename -> position index for minimum distance filtering
    all_filenames = sorted(info["file_name"] for info in per_image)
    fname_to_idx = {f: i for i, f in enumerate(all_filenames)}

    def select_with_gap(ranked, k, min_gap):
        """Greedily select up to k items, enforcing a minimum filename index gap."""
        if min_gap <= 0:
            return ranked[:k]
        selected = []
        used_indices = []
        for info in ranked:
            idx = fname_to_idx[info["file_name"]]
            if all(abs(idx - u) >= min_gap for u in used_indices):
                selected.append(info)
                used_indices.append(idx)
            if len(selected) >= k:
                break
        return selected

    # Sort by match rate
    has_gt.sort(key=lambda x: (-x["match_rate"], -x["n_matched"], x["n_fp"]))

    # For "best", only consider images with at least one match
    has_match = [info for info in has_gt if info["n_matched"] > 0]

    top_k = min(args.top_k, len(has_match))
    bottom_k = min(
        args.bottom_k if args.bottom_k is not None else args.top_k, len(has_gt)
    )
    best = select_with_gap(has_match, top_k, args.min_gap)
    worst = select_with_gap(has_gt[::-1], bottom_k, args.min_gap)

    # Determine label for output
    target_name = "seaclear" if "seaclear" in args.target_ann.lower() else "trashcan"
    label = f"{args.source_dataset} model → {target_name} data ({eval_space} space)"
    if args.preprocess != "none":
        label += f" [{args.preprocess}]"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print summary stats
    match_rates = [info["match_rate"] for info in has_gt]
    print(f"\n  {label}")
    print(f"  Mean match rate: {np.mean(match_rates):.3f}")
    print(f"  Median match rate: {np.median(match_rates):.3f}")
    print(
        f"  Images with >=1 match: {sum(1 for r in match_rates if r > 0)}/{len(match_rates)}"
    )

    # Save best
    print(f"\nSaving top-{top_k} best ...")
    save_grid(
        best,
        cat_names,
        out_dir / "best.png",
        f"BEST — {label} (top {top_k})",
        score_thresh=args.score_thresh,
    )

    # Save worst
    print(f"Saving bottom-{bottom_k} worst ...")
    save_grid(
        worst,
        cat_names,
        out_dir / "worst.png",
        f"WORST — {label} (bottom {bottom_k})",
        score_thresh=args.score_thresh,
    )

    # Also save a summary CSV
    csv_path = out_dir / "per_image_scores.csv"
    with open(csv_path, "w") as f:
        f.write("rank,img_id,file_name,match_rate,n_gt,n_matched,n_pred,n_fp\n")
        for rank, info in enumerate(has_gt):
            f.write(
                f"{rank+1},{info['img_id']},{info['file_name']},"
                f"{info['match_rate']:.4f},{info['n_gt']},{info['n_matched']},"
                f"{info['n_pred']},{info['n_fp']}\n"
            )
    print(f"\n  Per-image scores saved to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
