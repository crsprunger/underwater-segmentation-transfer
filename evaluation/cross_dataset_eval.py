"""
Cross-dataset evaluation for underwater instance segmentation.

All evaluation is done in TrashCan category space (20 overlapping classes).
Annotation files should be pre-converted using convert_seaclear_to_tc.py.

For TC-trained models, predictions are already in TC space.
For SC-trained models, predictions are remapped SC -> TC at eval time.

Usage examples:
    # TC model on TC test (in-domain)
    python cross_dataset_eval.py \
        --model-name trashcan_chunksplit \
        --target-ann trashcan_data/dataset/instance_version/trashcan_chunksplit_test.json \
        --target-img trashcan_data/dataset/instance_version/images \
        --category-group coarse ternary binary

    # TC model on SC test (cross-domain)
    python cross_dataset_eval.py \
        --model-name trashcan_chunksplit \
        --target-ann seaclear_data/seaclear_480p_chunksplit_test.json \
        --target-img seaclear_data/images_480p \
        --category-group coarse ternary binary

    # SC model on SC test (in-domain, evaluated in TC space)
    python cross_dataset_eval.py \
        --model-name seaclear_chunksplit \
        --target-ann seaclear_data/seaclear_480p_chunksplit_test.json \
        --target-img seaclear_data/images_480p \
        --category-group coarse ternary binary

    # SC model on TC test (cross-domain)
    python cross_dataset_eval.py \
        --model-name seaclear_chunksplit \
        --target-ann trashcan_data/dataset/instance_version/trashcan_chunksplit_test.json \
        --target-img trashcan_data/dataset/instance_version/images \
        --category-group coarse ternary binary

    # Pooled model on TC test (evaluated in TC space)
    python cross_dataset_eval.py \
        --model-name pooled_chunksplit \
        --target-ann trashcan_data/dataset/instance_version/trashcan_chunksplit_test.json \
        --target-img trashcan_data/dataset/instance_version/images \
        --category-group coarse ternary binary

    # Pooled model on SC test (evaluated in TC space)
    python cross_dataset_eval.py \
        --model-name pooled_chunksplit \
        --target-ann seaclear_data/seaclear_480p_chunksplit_test.json \
        --target-img seaclear_data/images_480p \
        --category-group coarse ternary binary
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.io import decode_image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

from src.model import build_model
from src.train import load_config
from src.evaluation import TTAWrapper
from src.category_groups import (
    get_scheme,
    remap_results,
    remap_coco_gt,
    detect_source_space,
    coarsest_common_space,
    spaces_coarser_than,
)


def remap_predictions_to_space(
    results: list[dict], source_space: str, target_space: str
) -> list[dict]:
    """Remap model predictions to an arbitrary target category space.

    Args:
        results: list of prediction dicts with 'category_id'
        source_space: detected source space ('trashcan', 'seaclear', 'tc20', etc.)
        target_space: target space to remap to

    Returns:
        Remapped predictions (unmapped categories dropped)
    """
    if source_space == target_space:
        return results
    scheme = get_scheme(target_space, source=source_space)
    return remap_results(results, scheme["mapping"])


# ═══════════════════════════════════════════════════════════════════════
# Test-time preprocessing
# ═══════════════════════════════════════════════════════════════════════


def apply_clahe(
    img_tensor: torch.Tensor, clip_limit: float = 2.0, tile_grid: int = 8
) -> torch.Tensor:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a float [0,1] RGB tensor.
    Operates on the L channel in LAB space to avoid color shifts.
    Returns float [0,1] RGB tensor of the same shape.
    """
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return (
        torch.from_numpy(result.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .to(img_tensor.device)
    )


def apply_histogram_eq(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply standard histogram equalization (L channel in LAB space).
    Returns float [0,1] RGB tensor of the same shape.
    """
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return (
        torch.from_numpy(result.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .to(img_tensor.device)
    )


def apply_grayscale(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to grayscale (3-channel, all channels identical).
    Input/output: float [0,1] RGB tensor of shape (3, H, W).
    """
    # ITU-R BT.601 luminance weights
    gray = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
    return gray.unsqueeze(0).expand(3, -1, -1).contiguous()


# ═══════════════════════════════════════════════════════════════════════
# Fourier Domain Adaptation (FDA)
# ═══════════════════════════════════════════════════════════════════════


def compute_fda_reference(img_dir: Path, max_images: int = 200) -> np.ndarray:
    """Compute the average low-frequency Fourier amplitude spectrum of a dataset.

    Args:
        img_dir: directory of images to compute the reference spectrum from
        max_images: max images to sample (for speed)

    Returns:
        mean_amplitude: (H, W, 3) mean amplitude spectrum in frequency domain,
                        where H and W are the median image dimensions.
                        Images are resized to this common size before averaging.
    """
    import glob

    paths = sorted(
        glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))
    )
    if len(paths) > max_images:
        rng = np.random.RandomState(42)
        paths = list(rng.choice(paths, max_images, replace=False))

    # First pass: find median dimensions
    heights, widths = [], []
    for p in paths[:50]:
        img = cv2.imread(p)
        if img is not None:
            heights.append(img.shape[0])
            widths.append(img.shape[1])
    target_h = int(np.median(heights))
    target_w = int(np.median(widths))

    # Second pass: accumulate amplitude spectra
    amp_sum = np.zeros((target_h, target_w, 3), dtype=np.float64)
    count = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h)).astype(np.float32) / 255.0
        for c in range(3):
            f = np.fft.fft2(img[:, :, c])
            f_shift = np.fft.fftshift(f)
            amp_sum[:, :, c] += np.abs(f_shift)
        count += 1

    if count == 0:
        raise ValueError(f"No valid images found in {img_dir}")

    return amp_sum / count


def apply_fda(
    img_tensor: torch.Tensor, ref_amplitude: np.ndarray, beta: float = 0.01
) -> torch.Tensor:
    """Apply Fourier Domain Adaptation to a single image.

    Replaces the low-frequency amplitude of the input image with that of the
    reference (training domain) spectrum. The phase is preserved, so structure
    and edges remain intact — only the global color/style shifts.

    Args:
        img_tensor: (3, H, W) float [0,1] RGB tensor
        ref_amplitude: (ref_H, ref_W, 3) mean amplitude spectrum from compute_fda_reference
        beta: fraction of the spectrum to replace (0.01 = 1% of each dimension,
              only the very lowest frequencies). Typical range: 0.005 to 0.05.

    Returns:
        (3, H, W) float [0,1] RGB tensor with adapted style
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    h, w, _ = img_np.shape
    ref_h, ref_w, _ = ref_amplitude.shape

    # Resize reference amplitude to match input image dimensions
    if ref_h != h or ref_w != w:
        ref_amp = cv2.resize(ref_amplitude, (w, h))
    else:
        ref_amp = ref_amplitude

    # Build low-frequency mask: 1 in the center (beta fraction), 0 elsewhere
    cy, cx = h // 2, w // 2
    bh = max(1, int(h * beta))
    bw = max(1, int(w * beta))

    result = np.zeros_like(img_np)
    for c in range(3):
        # FFT of input
        f = np.fft.fft2(img_np[:, :, c])
        f_shift = np.fft.fftshift(f)

        amp = np.abs(f_shift)
        phase = np.angle(f_shift)

        # Replace low-frequency amplitude with reference
        amp[cy - bh : cy + bh, cx - bw : cx + bw] = ref_amp[
            cy - bh : cy + bh, cx - bw : cx + bw, c
        ]

        # Reconstruct
        f_new = amp * np.exp(1j * phase)
        f_new = np.fft.ifftshift(f_new)
        result[:, :, c] = np.real(np.fft.ifft2(f_new))

    result = np.clip(result, 0, 1).astype(np.float32)
    return torch.from_numpy(result).permute(2, 0, 1).to(img_tensor.device)


PREPROCESS_FNS = {
    "none": None,
    "clahe": apply_clahe,
    "histogram_eq": apply_histogram_eq,
    "grayscale": apply_grayscale,
    # "fda" is handled specially — needs a reference spectrum, so it's set up in main()
}


# ═══════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_inference(
    model,
    coco_gt: COCO,
    img_dir: Path,
    batch_size: int = 8,
    preprocess: str = "none",
    img_ids: list[int] = None,
) -> list[dict]:
    """Run model inference on all images in coco_gt, return COCO-format results."""
    device = next(model.parameters()).device
    model.eval()

    if img_ids is None:
        img_ids = coco_gt.getImgIds()
    results = []

    for i in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[i : i + batch_size]
        batch_imgs = []
        batch_info = []

        preprocess_fn = PREPROCESS_FNS.get(preprocess)
        for img_id in batch_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_dir / Path(img_info["file_name"]).name
            img = decode_image(str(img_path), mode=torchvision.io.ImageReadMode.RGB)
            img = img.float() / 255.0
            if preprocess_fn is not None:
                img = preprocess_fn(img)
            batch_imgs.append(img.to(device))
            batch_info.append((img_id, img_info["width"], img_info["height"]))

        outputs = model(batch_imgs)

        for (img_id, orig_w, orig_h), output in zip(batch_info, outputs):
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"]

            # Resize masks to original image dimensions if needed
            if masks.shape[-2] != orig_h or masks.shape[-1] != orig_w:
                masks = torch.nn.functional.interpolate(
                    masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )
            masks = (masks[:, 0] > 0.5).to(torch.uint8).cpu().numpy()

            for j in range(len(scores)):
                x1, y1, x2, y2 = boxes[j]
                rle = mask_util.encode(np.asfortranarray(masks[j]))
                rle["counts"] = rle["counts"].decode("utf-8")

                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(labels[j]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(scores[j]),
                        "segmentation": rle,
                    }
                )

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {min(i + batch_size, len(img_ids))}/{len(img_ids)} images ...")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════


def compute_metrics(coco_gt: COCO, results: list[dict]) -> dict:
    """Run COCO evaluation and return metrics dict with per-class AP."""
    if not results:
        print("  WARNING: No predictions to evaluate.")
        return {"mask_mAP": 0.0, "mask_mAP_50": 0.0, "box_mAP": 0.0, "box_mAP_50": 0.0}

    coco_dt = coco_gt.loadRes(results)
    metrics = {}

    for iou_type in ["bbox", "segm"]:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        prefix = "box" if iou_type == "bbox" else "mask"
        metrics[f"{prefix}_mAP"] = coco_eval.stats[0]
        metrics[f"{prefix}_mAP_50"] = coco_eval.stats[1]
        metrics[f"{prefix}_mAP_75"] = coco_eval.stats[2]
        metrics[f"{prefix}_mAP_small"] = coco_eval.stats[3]
        metrics[f"{prefix}_mAP_medium"] = coco_eval.stats[4]
        metrics[f"{prefix}_mAP_large"] = coco_eval.stats[5]

    # Per-class AP (segm)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()

    cat_ids = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    per_class = {}

    # precision shape: [T=10, R=101, K=num_cats, A=4, M=3]
    precision = coco_eval.eval["precision"]
    for k_idx, cat_id in enumerate(coco_eval.params.catIds):
        ap = precision[:, :, k_idx, 0, 2]  # all IoU, all recall, area=all, maxDet=100
        mean_ap = float(ap[ap > -1].mean()) if (ap > -1).any() else 0.0
        name = cat_names.get(cat_id, f"id_{cat_id}")
        per_class[name] = round(mean_ap, 4)

    metrics["per_class_ap"] = per_class
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point: load model, run inference, compute COCO metrics, save JSON."""
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation in TrashCan category space."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training_config.yaml for the model",
    )
    parser.add_argument(
        "--source-dataset",
        default=None,
        choices=["trashcan", "seaclear", "pooled", "tc20"],
        help="Category space of the model's training annotations. "
        "Auto-detected from the training config if not specified.",
    )
    parser.add_argument(
        "--target-ann",
        required=True,
        help="Target annotation JSON (any supported category space; auto-detected)",
    )
    parser.add_argument(
        "--target-img", required=True, help="Target dataset image directory"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=["none", "clahe", "histogram_eq", "grayscale", "fda"],
        help="Test-time preprocessing: 'clahe' (CLAHE on L channel), "
        "'histogram_eq' (histogram equalization on L channel), "
        "'grayscale' (convert to grayscale), "
        "'fda' (Fourier Domain Adaptation — requires --fda-ref-dir), "
        "or 'none' (default)",
    )
    parser.add_argument(
        "--fda-ref-dir",
        type=str,
        default=None,
        help="Directory of training images to compute the FDA reference spectrum from. "
        "Used when --preprocess fda is set and --fda-ref is not provided.",
    )
    parser.add_argument(
        "--fda-ref",
        type=str,
        default=None,
        help="Path to precomputed FDA reference spectrum (.npz from compute_fda_spectrum.py). "
        "Faster than --fda-ref-dir since it skips recomputation.",
    )
    parser.add_argument(
        "--fda-beta",
        type=float,
        default=0.01,
        help="FDA beta parameter: fraction of the Fourier spectrum to replace. "
        "Smaller = gentler adaptation. Typical range: 0.005–0.05. Default: 0.01.",
    )
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable test-time augmentation (uses TTAWrapper from train.py)",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        default=False,
        help="TTA: include horizontal flip",
    )
    parser.add_argument(
        "--tta-grayscale",
        action="store_true",
        help="TTA: include grayscale view",
        default=False,
    )
    parser.add_argument(
        "--tta-clahe",
        action="store_true",
        help="TTA: include CLAHE-enhanced view",
        default=False,
    )
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="*",
        default=[1.0],
        help="TTA: scales to use (default: [1.0], i.e., original only)",
    )
    parser.add_argument(
        "--tta-nms-threshold",
        type=float,
        default=0.5,
        help="TTA: NMS IoU threshold for merging predictions",
    )
    parser.add_argument(
        "--category-group",
        type=str,
        nargs="*",
        default=[],
        help="Evaluate with coarser category groupings in addition to the primary eval space. "
        "Available: 'coarse' (5 categories: rov, organic, trash_easy, trash_entangled, trash_heavy), "
        "'ternary' (3 categories: rov, non_trash, trash), "
        "'binary' (2 categories: non_trash, trash). "
        "Can specify multiple: --category-group coarse ternary binary",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the model to use for evaluation",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="model6v2",
        help="Version of the model to use for evaluation",
    )
    args = parser.parse_args()

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
        if not train_ann:
            parser.error(
                "Cannot auto-detect source category space: training config has no "
                "train_ann_path. Use --source-dataset to specify manually."
            )
        if not Path(train_ann).exists():
            parser.error(
                f"Cannot auto-detect source category space: training annotation file "
                f"not found at {train_ann}. Use --source-dataset to specify manually."
            )
        train_coco = COCO(train_ann)
        args.source_dataset = detect_source_space(train_coco)
        print(f"  Auto-detected source category space: {args.source_dataset}")
        del train_coco  # free memory

    print(
        f"  Source dataset: {args.source_dataset} ({cfg.num_classes} classes incl. background)"
    )

    # Wrap with TTA if requested
    if (
        args.tta_grayscale
        or args.tta_clahe
        or args.tta_flip
        or args.tta_scales != [1.0]
        or args.tta_nms_threshold != 0.5
    ):
        # Build a TTA config from CLI args
        cfg.tta_enabled = True
        cfg.tta_flip = args.tta_flip
        cfg.tta_grayscale = args.tta_grayscale
        cfg.tta_clahe = args.tta_clahe
        cfg.tta_scales = tuple(args.tta_scales)
        cfg.tta_nms_threshold = args.tta_nms_threshold
        model = TTAWrapper(model, cfg)
        model.to(device)
        tta_parts = ["flip"] if args.tta_flip else []
        if args.tta_grayscale:
            tta_parts.append("grayscale")
        if args.tta_clahe:
            tta_parts.append("clahe")
        if any(abs(s - 1.0) > 1e-3 for s in args.tta_scales):
            tta_parts.append(f"scales={args.tta_scales}")
        print(f"  TTA enabled: {', '.join(tta_parts)}")

    # Load target annotations
    print(f"\nLoading target annotations: {args.target_ann}")
    coco_gt_raw = COCO(args.target_ann)
    n_images = len(coco_gt_raw.getImgIds())
    n_anns = len(coco_gt_raw.getAnnIds())
    gt_cat_names = [c["name"] for c in coco_gt_raw.loadCats(coco_gt_raw.getCatIds())]
    gt_space = detect_source_space(coco_gt_raw)
    print(f"  {n_images} images, {n_anns} annotations, {len(gt_cat_names)} categories")
    print(f"  Detected annotation space: {gt_space}")

    # Determine common evaluation space (coarsest of source and target)
    eval_space = coarsest_common_space(args.source_dataset, gt_space)
    print(f"\n  Model trained in: {args.source_dataset}")
    print(f"  Target annotations in: {gt_space}")
    print(f"  Evaluation space: {eval_space}")

    # Remap GT to eval space if needed
    if gt_space == eval_space:
        coco_gt = coco_gt_raw
    else:
        gt_scheme = get_scheme(eval_space, source=gt_space)
        coco_gt = remap_coco_gt(
            coco_gt_raw, gt_scheme["mapping"], gt_scheme["categories"]
        )
        n_anns_remapped = len(coco_gt.getAnnIds())
        print(
            f"  Remapped GT {gt_space}→{eval_space}: {n_anns_remapped}/{n_anns} annotations kept"
        )

    # Set up FDA if requested
    if args.preprocess == "fda":
        from functools import partial

        if args.fda_ref is not None:
            print(f"\nLoading precomputed FDA reference from {args.fda_ref} ...")
            data = np.load(args.fda_ref)
            ref_amp = data["mean_amplitude"]
            print(
                f"  Spectrum shape: {ref_amp.shape}, from {int(data['n_images'])} images"
            )
        elif args.fda_ref_dir is not None:
            print(f"\nComputing FDA reference spectrum from {args.fda_ref_dir} ...")
            ref_amp = compute_fda_reference(Path(args.fda_ref_dir))
        else:
            parser.error(
                "--fda-ref or --fda-ref-dir is required when --preprocess fda is used"
            )
        print(f"  Reference spectrum shape: {ref_amp.shape}, beta={args.fda_beta}")
        PREPROCESS_FNS["fda"] = partial(
            apply_fda, ref_amplitude=ref_amp, beta=args.fda_beta
        )

    # Run inference
    if args.preprocess != "none":
        print(f"\nPreprocessing: {args.preprocess}")
    print(f"\nRunning inference on {n_images} images ...")
    results = run_inference(
        model,
        coco_gt,
        Path(args.target_img),
        batch_size=args.batch_size,
        preprocess=args.preprocess,
    )

    # Keep raw results for --category-group (direct remapping from original spaces)
    raw_results = results

    # Remap predictions to eval space
    n_before = len(results)
    results = remap_predictions_to_space(results, args.source_dataset, eval_space)
    if len(results) != n_before:
        print(
            f"  Remapped predictions {args.source_dataset}→{eval_space}: "
            f"{len(results)}/{n_before} kept"
        )

    # Compute metrics
    print("\nComputing metrics ...")
    metrics = compute_metrics(coco_gt, results)

    # Determine target dataset name from annotation path
    target_name = "seaclear" if "seaclear" in args.target_ann.lower() else "trashcan"

    print(f"\n{'='*60}")
    print(
        f"Results: {args.source_dataset} model → {target_name} data ({eval_space} space)"
    )
    print(f"{'='*60}")
    print(f"  mask mAP:    {metrics['mask_mAP']:.4f}")
    print(f"  mask mAP50:  {metrics['mask_mAP_50']:.4f}")
    print(f"  mask mAP75:  {metrics['mask_mAP_75']:.4f}")
    print(f"  box  mAP:    {metrics['box_mAP']:.4f}")
    print(f"  box  mAP50:  {metrics['box_mAP_50']:.4f}")

    if "per_class_ap" in metrics:
        print("\n  Per-class mask AP:")
        for name, ap in sorted(metrics["per_class_ap"].items(), key=lambda x: -x[1]):
            print(f"    {name:<30s} {ap:.4f}")

    # Grouped category evaluations (only for spaces coarser than eval_space)
    # Remap directly from original spaces to avoid losing categories
    # that are excluded from intermediate spaces like TC20.
    valid_groups = set(spaces_coarser_than(eval_space))
    grouped_metrics = {}
    if args.category_group:
        for group_name in args.category_group:
            if group_name not in valid_groups:
                print(
                    f"\n  Skipping --category-group '{group_name}': not coarser than "
                    f"eval space '{eval_space}'"
                )
                continue

            # Remap GT directly from original gt_space (not from eval_space)
            gt_scheme = get_scheme(group_name, source=gt_space)
            # Remap predictions directly from model's source space
            pred_scheme = get_scheme(group_name, source=args.source_dataset)

            categories = gt_scheme["categories"]

            print(f"\n{'='*60}")
            print(
                f"Re-evaluating with '{group_name}' grouping "
                f"({len(categories)} categories) ..."
            )
            print(f"  GT remapped:   {gt_space} → {group_name}")
            print(f"  Preds remapped: {args.source_dataset} → {group_name}")
            print(f"{'='*60}")

            grouped_gt = remap_coco_gt(coco_gt_raw, gt_scheme["mapping"], categories)
            grouped_results = remap_results(raw_results, pred_scheme["mapping"])

            n_gt_anns = len(grouped_gt.getAnnIds())
            print(f"  GT annotations: {n_gt_anns}/{n_anns} kept")
            print(f"  Predictions: {len(grouped_results)}/{len(raw_results)} kept")

            group_metrics = compute_metrics(grouped_gt, grouped_results)
            grouped_metrics[group_name] = group_metrics

            print(f"\n  mask mAP:    {group_metrics['mask_mAP']:.4f}")
            print(f"  mask mAP50:  {group_metrics['mask_mAP_50']:.4f}")
            print(f"  mask mAP75:  {group_metrics['mask_mAP_75']:.4f}")
            print(f"  box  mAP:    {group_metrics['box_mAP']:.4f}")
            print(f"  box  mAP50:  {group_metrics['box_mAP_50']:.4f}")

            if "per_class_ap" in group_metrics:
                print(f"\n  Per-class mask AP ({group_name}):")
                for name, ap in sorted(
                    group_metrics["per_class_ap"].items(), key=lambda x: -x[1]
                ):
                    print(f"    {name:<30s} {ap:.4f}")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "source_dataset": args.source_dataset,
            "target_dataset": target_name,
            "target_ann": args.target_ann,
            "checkpoint": args.checkpoint,
            "eval_space": eval_space,
            "preprocess": args.preprocess,
            "metrics": metrics,
        }
        if grouped_metrics:
            save_data["grouped_metrics"] = grouped_metrics
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
