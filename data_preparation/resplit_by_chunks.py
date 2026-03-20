"""
Re-split COCO annotation files using chunk-based grouping to prevent
data leakage from temporally adjacent frames.

Images are sorted by filename (lexicographic), grouped into consecutive
chunks, and entire chunks are assigned to train/val/test splits. This
ensures that nearby frames (which are visually similar) stay in the same
split.

Works for both TrashCan and SeaClear datasets:
- TrashCan: filenames encode vid_idx and frame_idx, so lex sort groups
  by video then by frame order within each video.
- SeaClear: filenames vary, but consecutive images in lex order tend to
  be from the same capture session.

Usage:
    # TrashCan (re-split the original train + val into train/val/test)
    python resplit_by_chunks.py \
        --input trashcan_data/dataset/instance_version/instances_train_trashcan.json \
               trashcan_data/dataset/instance_version/instances_val_trashcan.json \
        --output-dir trashcan_data/dataset/instance_version \
        --output-prefix trashcan_chunksplit \
        --chunk-size 50

    # SeaClear 480p
    python resplit_by_chunks.py \
        --input seaclear_data/seaclear_480p_train.json \
               seaclear_data/seaclear_480p_val.json \
               seaclear_data/seaclear_480p_test.json \
        --output-dir seaclear_data \
        --output-prefix seaclear_480p_chunksplit \
        --chunk-size 50

    # SeaClear original
    python resplit_by_chunks.py \
        --input seaclear_data/seaclear_train.json \
               seaclear_data/seaclear_val.json \
               seaclear_data/seaclear_test.json \
        --output-dir seaclear_data \
        --output-prefix seaclear_chunksplit \
        --chunk-size 50
"""

import argparse
import json
import random
from pathlib import Path


def load_and_merge(input_paths: list[str]) -> dict:
    """Load one or more COCO JSON files and merge them into a single dataset.

    Handles duplicate image IDs by keeping the first occurrence. Annotations
    reference their original image IDs, so as long as images are consistent
    across files, merging is safe.
    """
    merged_images = {}
    merged_annotations = []
    categories = None

    for path in input_paths:
        print(f"  Loading {path} ...")
        with open(path) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]

        for img in data["images"]:
            if img["id"] not in merged_images:
                merged_images[img["id"]] = img

        merged_annotations.extend(data["annotations"])

    # Deduplicate annotations by id
    seen_ann_ids = set()
    unique_anns = []
    for ann in merged_annotations:
        if ann["id"] not in seen_ann_ids:
            seen_ann_ids.add(ann["id"])
            unique_anns.append(ann)

    images = list(merged_images.values())
    print(f"  Merged: {len(images)} images, {len(unique_anns)} annotations")
    return {
        "images": images,
        "annotations": unique_anns,
        "categories": categories,
    }


def chunk_split(
    images: list[dict], chunk_size: int, train_ratio: float, val_ratio: float, seed: int
) -> tuple[list, list, list]:
    """Sort images by filename, group into chunks, assign chunks to splits.

    Returns:
        (train_images, val_images, test_images)
    """
    # Sort by filename
    images_sorted = sorted(images, key=lambda img: img["file_name"])

    # Group into chunks
    chunks = []
    for i in range(0, len(images_sorted), chunk_size):
        chunks.append(images_sorted[i : i + chunk_size])

    # print(
    #     f"  {len(chunks)} chunks of size <={chunk_size} "
    #     f"({len(images_sorted)} images)"
    # )

    # Shuffle chunks (not individual images)
    rng = random.Random(seed)
    rng.shuffle(chunks)

    # Assign chunks to splits
    n_train = int(len(chunks) * train_ratio)
    n_val = int(len(chunks) * val_ratio)

    train_chunks = chunks[:n_train]
    val_chunks = chunks[n_train : n_train + n_val]
    test_chunks = chunks[n_train + n_val :]

    train_images = [img for chunk in train_chunks for img in chunk]
    val_images = [img for chunk in val_chunks for img in chunk]
    test_images = [img for chunk in test_chunks for img in chunk]

    return train_images, val_images, test_images


def save_split(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    output_path: str,
) -> tuple[int, int]:
    """Save a COCO-format JSON for a split."""
    img_ids = {img["id"] for img in images}
    split_anns = [ann for ann in annotations if ann["image_id"] in img_ids]
    data = {
        "images": images,
        "annotations": split_anns,
        "categories": categories,
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    return len(images), len(split_anns)


def compute_per_split_class_distributions(
    train_imgs: list[dict],
    val_imgs: list[dict],
    test_imgs: list[dict],
    data: dict,
) -> dict[str, dict[str, int]]:
    """Compute class distribution per split."""
    img_id_to_split = {}
    for img in train_imgs:
        img_id_to_split[img["id"]] = "train"
    for img in val_imgs:
        img_id_to_split[img["id"]] = "val"
    for img in test_imgs:
        img_id_to_split[img["id"]] = "test"

    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    split_cat_counts = {"train": {}, "val": {}, "test": {}}
    for ann in data["annotations"]:
        split = img_id_to_split.get(ann["image_id"])
        if split:
            name = cat_name.get(ann["category_id"], f"id_{ann['category_id']}")
            split_cat_counts[split][name] = split_cat_counts[split].get(name, 0) + 1

    return split_cat_counts


def compute_split_cat_counts_score(
    split_cat_counts: dict[str, dict[str, int]],
    all_cats: list[str],
    train_ratio: float,
    test_ratio: float,
) -> float:
    """Compute a score for the split class distributions."""
    score = 0
    zero_count = 0
    for cat in all_cats:
        train_count = split_cat_counts["train"].get(cat, 0)
        val_count = split_cat_counts["val"].get(cat, 0)
        test_count = split_cat_counts["test"].get(cat, 0)
        total_count = float(train_count + val_count + test_count)
        if train_count == 0 or test_count == 0:
            zero_count += 1
        score += abs(
            min(train_count / total_count - train_ratio, 0.0)
            + min(test_count / total_count - test_ratio, 0.0)
        )
    return score * (zero_count**2)


def main():
    parser = argparse.ArgumentParser(
        description="Re-split COCO annotations by consecutive chunks to avoid data leakage."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input COCO JSON file(s) to merge and re-split",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for split JSON files"
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output filename prefix (e.g. 'instances_chunksplit_trashcan')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of consecutive images per chunk (default: 50)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Fraction of chunks for training (default: 0.75)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of chunks for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk shuffling (default: 42)",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Try different random seeds to find the best split",
    )
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    print("Chunk-based dataset re-split")
    print(f"  Chunk size: {args.chunk_size}")
    print(
        f"  Ratios: train={args.train_ratio:.0%}, "
        f"val={args.val_ratio:.0%}, test={test_ratio:.0%}"
    )
    if args.search:
        print("  Searching for the best split...")
    else:
        print(f"  Seed: {args.seed}")

    # Load and merge inputs
    print(f"\nLoading {len(args.input)} input file(s) ...")
    data = load_and_merge(args.input)

    print_freq = 50
    iters = 0
    best_seed = None
    best_split_cat_counts = None
    best_split_cat_counts_score = None
    if args.search:
        seed = 0
    else:
        seed = args.seed
    try:
        while args.search or iters < 1:
            # Split
            train_imgs, val_imgs, test_imgs = chunk_split(
                data["images"],
                args.chunk_size,
                args.train_ratio,
                args.val_ratio,
                seed,
            )

            split_cat_counts = compute_per_split_class_distributions(
                train_imgs,
                val_imgs,
                test_imgs,
                data,
            )

            if iters == 0:
                all_cats = sorted(
                    set().union(*(d.keys() for d in split_cat_counts.values()))
                )

            split_cat_counts_score = compute_split_cat_counts_score(
                split_cat_counts, all_cats, args.train_ratio, test_ratio
            )

            if (
                best_split_cat_counts is None
                or split_cat_counts_score < best_split_cat_counts_score
            ):
                best_split_cat_counts = split_cat_counts
                best_split_cat_counts_score = split_cat_counts_score
                best_seed = seed

            if iters > 0 and iters % print_freq == 0:
                print(
                    f"  Best seed after {iters} iterations: {best_seed} (score={best_split_cat_counts_score:.2f})"
                )

            seed += 1
            iters += 1
    except KeyboardInterrupt:
        print("\nInterrupted! Printing best results before exiting the program.")

    finally:
        # Save splits
        if iters > 0:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            splits = [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]
            print("\nSaving splits:")
            for split_name, split_imgs in splits:
                out_path = out_dir / f"{args.output_prefix}_{split_name}.json"
                n_img, n_ann = save_split(
                    split_imgs, data["annotations"], data["categories"], str(out_path)
                )
                print(
                    f"  {split_name:5s}: {n_img:>5d} images, {n_ann:>5d} annotations -> {out_path}"
                )

            # Print class distribution summary
            print(
                f"\nBest split (score={best_split_cat_counts_score:.2f}, seed={best_seed}): per-category annotation counts:"
            )
            print(f"  {'category':<30s} {'train':>6s} {'val':>6s} {'test':>6s}")
            print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
            for cat in all_cats:
                tr = best_split_cat_counts["train"].get(cat, 0)
                va = best_split_cat_counts["val"].get(cat, 0)
                te = best_split_cat_counts["test"].get(cat, 0)
                total = tr + va + te
                tr_pct = tr / total * 100 if total > 0 else 0
                print(f"  {cat:<30s} {tr:>6d} {va:>6d} {te:>6d}  ({tr_pct:.0f}% train)")
        print("Done.")


if __name__ == "__main__":
    main()
