"""
Create a pooled dataset by merging TrashCan and SeaClear annotations into
a single COCO JSON, with a combined image directory using symlinks.

Input annotation files can be in any category space — each is auto-detected
and remapped to the target space using the direct mappings in category_groups.py.
This avoids losing categories that would be dropped through an intermediate
TC20 conversion.

Image IDs and annotation IDs from the second dataset are offset to avoid
collisions.

Usage:

# Pooled in TC20 space (inputs already converted to TC20)
splits=("train" "val" "test"); for split in "${splits[@]}"; do
    python create_pooled_dataset.py \
        --tc-ann "trashcan_data/.../trashcan_chunksplit_${split}_tc20.json" \
        --tc-img trashcan_data/dataset/instance_version/images \
        --sc-ann "seaclear_data/seaclear_480p_chunksplit_${split}_tc20.json" \
        --sc-img seaclear_data/images_480p \
        --output-dir pooled_data \
        --output-json-name "pooled_${split}.json" \
        --output-img-name images
done

# Pooled in coarse space (from original annotations, direct mapping)
splits=("train" "val" "test"); for split in "${splits[@]}"; do
    python create_pooled_dataset.py \
        --target coarse \
        --tc-ann "trashcan_data/.../instances_${split}.json" \
        --tc-img trashcan_data/dataset/instance_version/images \
        --sc-ann "seaclear_data/seaclear_480p_chunksplit_${split}.json" \
        --sc-img seaclear_data/images_480p \
        --output-dir pooled_data/coarse \
        --output-json-name "pooled_${split}_coarse.json" \
        --output-img-name images
done

"""

import argparse
import json
from pathlib import Path

from src.category_groups import (
    get_scheme,
    detect_source_space,
    available_targets,
    SPACE_HIERARCHY,
)
from pycocotools.coco import COCO


# Offset applied to SC image/annotation IDs to avoid collisions with TC
SC_ID_OFFSET = 1_000_000


def load_and_offset(
    ann_path: str,
    img_dir: str,
    symlink_dir: Path,
    id_offset: int = 0,
    cat_mapping: dict | None = None,
) -> tuple[list, list]:
    """Load a COCO JSON, create symlinks, optionally offset IDs and remap categories.

    Args:
        ann_path: path to COCO annotation JSON
        img_dir: path to image directory
        symlink_dir: directory to create image symlinks in
        id_offset: offset to add to image/annotation IDs
        cat_mapping: optional category ID mapping (source → target).
                     Annotations with unmapped IDs are dropped.

    Returns (images, annotations) with updated IDs and file_names pointing
    to symlinks in symlink_dir.
    """
    with open(ann_path) as f:
        data = json.load(f)

    img_dir = Path(img_dir).resolve()
    images = []
    annotations = []

    for img in data["images"]:
        src = img_dir / img["file_name"]
        dst = symlink_dir / img["file_name"]
        if not dst.exists():
            dst.symlink_to(src)

        new_img = img.copy()
        new_img["id"] = img["id"] + id_offset
        images.append(new_img)

    mapping_keys = set(cat_mapping.keys()) if cat_mapping else None
    dropped = 0
    for ann in data["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] = ann["id"] + id_offset
        new_ann["image_id"] = ann["image_id"] + id_offset
        if cat_mapping is not None:
            if ann["category_id"] not in mapping_keys:
                dropped += 1
                continue
            new_ann["category_id"] = cat_mapping[ann["category_id"]]
        annotations.append(new_ann)

    if dropped:
        print(f"    {dropped} annotations dropped (unmapped categories)")

    return images, annotations


def main():
    """CLI entry point: merge TC and SC annotations into a pooled COCO JSON with symlinked images."""
    parser = argparse.ArgumentParser(
        description="Create pooled TC+SC dataset in a unified category space."
    )
    parser.add_argument(
        "--target",
        default=None,
        choices=available_targets(),
        help="Target category space for the pooled output. If not specified, "
        "the input annotations must already be in the same category space "
        "(auto-detected). When specified, each input is auto-detected and "
        "remapped directly to the target space.",
    )
    parser.add_argument(
        "--tc-ann",
        required=True,
        help="TrashCan annotation JSON (any supported category space)",
    )
    parser.add_argument("--tc-img", required=True, help="TrashCan image directory")
    parser.add_argument(
        "--sc-ann",
        required=True,
        help="SeaClear annotation JSON (any supported category space)",
    )
    parser.add_argument("--sc-img", required=True, help="SeaClear image directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (will contain images/ and JSON)",
    )
    parser.add_argument(
        "--output-json-name",
        required=True,
        help="Output annotation JSON filename (e.g. pooled_train.json)",
    )
    parser.add_argument(
        "--output-img-name",
        required=True,
        help="Output image folder name (e.g. images_576p)",
        default="images",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_out_dir = out_dir / args.output_img_name
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect source spaces
    tc_coco = COCO(args.tc_ann)
    tc_space = detect_source_space(tc_coco)
    del tc_coco

    sc_coco = COCO(args.sc_ann)
    sc_space = detect_source_space(sc_coco)
    del sc_coco

    # Determine target space and mappings
    if args.target is not None:
        target_space = args.target
    else:
        # No --target: inputs must be in the same space already
        if tc_space != sc_space:
            parser.error(
                f"Input annotations are in different spaces ({tc_space}, {sc_space}). "
                f"Use --target to specify the output category space."
            )
        target_space = tc_space

    print(f"Target category space: {target_space}")
    print(
        f"  TC annotations: {tc_space}"
        + (f" → {target_space}" if tc_space != target_space else " (no remap)")
    )
    print(
        f"  SC annotations: {sc_space}"
        + (f" → {target_space}" if sc_space != target_space else " (no remap)")
    )

    # Get mappings (None if already in target space)
    if tc_space != target_space:
        tc_scheme = get_scheme(target_space, source=tc_space)
        tc_mapping = tc_scheme["mapping"]
    else:
        tc_scheme = get_scheme(target_space, source=tc_space)
        tc_mapping = None

    if sc_space != target_space:
        sc_scheme = get_scheme(target_space, source=sc_space)
        sc_mapping = sc_scheme["mapping"]
    else:
        sc_scheme = get_scheme(target_space, source=sc_space)
        sc_mapping = None

    # Get target categories from scheme
    target_categories = [
        {
            "id": c["id"],
            "name": c["name"],
            "supercategory": c.get("supercategory", c["name"]),
        }
        for c in tc_scheme["categories"].values()
    ]

    # Verify max TC IDs are below offset
    with open(args.tc_ann) as f:
        tc_data = json.load(f)
    max_tc_img_id = max(img["id"] for img in tc_data["images"])
    max_tc_ann_id = (
        max(ann["id"] for ann in tc_data["annotations"])
        if tc_data["annotations"]
        else 0
    )
    if max_tc_img_id >= SC_ID_OFFSET or max_tc_ann_id >= SC_ID_OFFSET:
        raise ValueError(
            f"TC IDs ({max_tc_img_id}, {max_tc_ann_id}) exceed offset {SC_ID_OFFSET}"
        )
    del tc_data

    # Load TC (no offset)
    print(f"\nLoading TrashCan: {args.tc_ann}")
    tc_images, tc_anns = load_and_offset(
        args.tc_ann,
        args.tc_img,
        img_out_dir,
        id_offset=0,
        cat_mapping=tc_mapping,
    )
    print(f"  {len(tc_images)} images, {len(tc_anns)} annotations")

    # Load SC (with offset)
    print(f"Loading SeaClear: {args.sc_ann}")
    sc_images, sc_anns = load_and_offset(
        args.sc_ann,
        args.sc_img,
        img_out_dir,
        id_offset=SC_ID_OFFSET,
        cat_mapping=sc_mapping,
    )
    print(f"  {len(sc_images)} images, {len(sc_anns)} annotations")

    # Merge
    merged = {
        "images": tc_images + sc_images,
        "annotations": tc_anns + sc_anns,
        "categories": target_categories,
    }

    out_path = out_dir / args.output_json_name
    with open(out_path, "w") as f:
        json.dump(merged, f)

    print(f"\nPooled dataset ({target_space} space):")
    print(f"  {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    print(f"  {len(target_categories)} categories")
    print(f"  Annotation JSON: {out_path}")
    print(f"  Image symlinks:  {img_out_dir}")

    # Per-category counts
    cat_name = {c["id"]: c["name"] for c in target_categories}
    tc_counts = {}
    sc_counts = {}
    for ann in tc_anns:
        name = cat_name.get(ann["category_id"], f"id_{ann['category_id']}")
        tc_counts[name] = tc_counts.get(name, 0) + 1
    for ann in sc_anns:
        name = cat_name.get(ann["category_id"], f"id_{ann['category_id']}")
        sc_counts[name] = sc_counts.get(name, 0) + 1

    all_cats = sorted(set(tc_counts.keys()) | set(sc_counts.keys()))
    print(f"\n  {'category':<30s} {'TC':>6s} {'SC':>6s} {'total':>6s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
    for cat in all_cats:
        tc = tc_counts.get(cat, 0)
        sc = sc_counts.get(cat, 0)
        print(f"  {cat:<30s} {tc:>6d} {sc:>6d} {tc+sc:>6d}")


if __name__ == "__main__":
    main()
