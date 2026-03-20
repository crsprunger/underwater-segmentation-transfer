"""
Resize a COCO-format dataset's images and annotations.

Scales each image so that its short side (--target-short) or long side
(--target-long) matches the given pixel count, preserving aspect ratio.
Annotation bounding boxes, segmentation polygons, and areas are scaled
to match.  The per-image scale factor is computed individually, so
mixed-resolution datasets (like TrashCan) are handled correctly.

Usage:
    # Downscale SeaClear to 480p (short side = 270)
    python resize_dataset.py \
        --ann seaclear_data/seaclear_train.json \
              seaclear_data/seaclear_val.json \
              seaclear_data/seaclear_test.json \
        --img-dir seaclear_data/images \
        --out-img-dir seaclear_data/images_480p \
        --target-short 270

    # Upscale TrashCan so long side = 1333
    python resize_dataset.py \
        --ann trashcan_data/.../instances_train.json \
        --img-dir trashcan_data/.../images \
        --out-img-dir trashcan_data/images_1333q \
        --target-long 1333
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ── Image resizing ────────────────────────────────────────────────────


def _compute_scale(orig_w: int, orig_h: int, target_short=None, target_long=None) -> float:
    """Compute the uniform scale factor to resize an image to the target dimension."""
    if target_short is not None:
        return target_short / min(orig_w, orig_h)
    else:
        return target_long / max(orig_w, orig_h)


def resize_images(img_dir: Path, out_dir: Path, ann_images: list[dict],
                  target_short=None, target_long=None) -> dict[int, float]:
    """Resize images and return {image_id: scale_factor}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_scale = {}

    for img_entry in tqdm(ann_images, desc="Resizing images"):
        orig_w, orig_h = img_entry["width"], img_entry["height"]
        scale = _compute_scale(orig_w, orig_h, target_short, target_long)
        new_w, new_h = round(orig_w * scale), round(orig_h * scale)
        id_to_scale[img_entry["id"]] = (scale, new_w, new_h)

        src = img_dir / img_entry["file_name"]
        dst = out_dir / img_entry["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists():
            with Image.open(src) as im:
                resampler = Image.LANCZOS if scale > 1 else Image.BILINEAR
                im.resize((new_w, new_h), resampler).save(dst, quality=95)

    return id_to_scale


# ── Annotation scaling ────────────────────────────────────────────────


def scale_annotations(data: dict, id_to_scale: dict[int, tuple]) -> dict:
    """Return a new COCO dict with images and annotations scaled in-place."""
    data = json.loads(json.dumps(data))  # deep copy

    # Update image dimensions
    for img in data["images"]:
        entry = id_to_scale.get(img["id"])
        if entry is None:
            continue
        _, new_w, new_h = entry
        img["width"] = new_w
        img["height"] = new_h

    # Scale annotations
    for ann in data["annotations"]:
        entry = id_to_scale.get(ann["image_id"])
        if entry is None:
            continue
        s, _, _ = entry

        # Bounding box [x, y, w, h]
        bx, by, bw, bh = ann["bbox"]
        ann["bbox"] = [bx * s, by * s, bw * s, bh * s]

        # Segmentation polygons: flat list [x1,y1,x2,y2,...]
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            ann["segmentation"] = [
                [coord * s for coord in poly]
                for poly in ann["segmentation"]
            ]

        # Area scales by s^2
        if "area" in ann:
            ann["area"] = ann["area"] * s * s

    return data


# ── Main ──────────────────────────────────────────────────────────────


def main():
    """CLI entry point: resize images and scale COCO annotations to match."""
    parser = argparse.ArgumentParser(
        description="Resize a COCO dataset's images and annotations."
    )
    parser.add_argument(
        "--ann", nargs="+", required=True,
        help="COCO annotation JSON file(s) to process",
    )
    parser.add_argument("--img-dir", required=True, help="Source image directory")
    parser.add_argument("--out-img-dir", required=True, help="Output image directory")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument(
        "--target-short", type=int, default=None,
        help="Target short-side pixel count",
    )
    size.add_argument(
        "--target-long", type=int, default=None,
        help="Target long-side pixel count",
    )
    parser.add_argument(
        "--output-suffix", type=str, default=None,
        help="Suffix inserted before .json in output filenames "
             "(e.g. '_480p' → train.json → train_480p.json). "
             "If omitted, annotation files are overwritten in-place.",
    )
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_img_dir = Path(args.out_img_dir)

    # Collect the union of all images across annotation files
    all_images = {}
    for ann_path in args.ann:
        with open(ann_path) as f:
            data = json.load(f)
        for img in data["images"]:
            all_images[img["id"]] = img

    print(f"Resizing {len(all_images)} unique images ...")
    id_to_scale = resize_images(
        img_dir, out_img_dir, list(all_images.values()),
        target_short=args.target_short, target_long=args.target_long,
    )

    # Report scale factors
    scales = sorted(set(round(s, 4) for s, _, _ in id_to_scale.values()))
    print(f"  Scale factors: {scales}")

    # Process each annotation file
    for ann_path in args.ann:
        with open(ann_path) as f:
            data = json.load(f)

        scaled = scale_annotations(data, id_to_scale)

        if args.output_suffix:
            p = Path(ann_path)
            out_path = p.with_name(p.stem + args.output_suffix + p.suffix)
        else:
            out_path = Path(ann_path)

        with open(out_path, "w") as f:
            json.dump(scaled, f)

        n_imgs = len(scaled["images"])
        n_anns = len(scaled["annotations"])
        print(f"  {out_path.name}: {n_imgs} images, {n_anns} annotations")

    print(f"\nImages saved to: {out_img_dir}")


if __name__ == "__main__":
    main()
