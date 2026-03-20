"""
Convert COCO annotation files between category spaces.

Unified replacement for convert_trashcan_to_tc20.py and convert_seaclear_to_tc20.py.
Uses the mappings defined in category_groups.py.

Usage:
    # TrashCan → TC20 (same as convert_trashcan_to_tc20.py)
    python convert_annotations.py \
        --source trashcan --target tc20 \
        --input trashcan_data/.../instances_train.json \
        --output trashcan_data/.../instances_train_tc20.json

    # SeaClear → TC20 (same as convert_seaclear_to_tc20.py)
    python convert_annotations.py \
        --source seaclear --target tc20 \
        --input seaclear_data/seaclear_train.json \
        --output seaclear_data/seaclear_train_tc20.json

    # TC20 → coarse (for training with coarse labels)
    python convert_annotations.py \
        --source tc20 --target coarse \
        --input trashcan_data/.../instances_train_tc20.json \
        --output trashcan_data/.../instances_train_coarse.json

    # SeaClear → ternary (direct, no categories lost)
    python convert_annotations.py \
        --source seaclear --target ternary \
        --input seaclear_data/seaclear_train.json \
        --output seaclear_data/seaclear_train_ternary.json

    # SeaClear → binary (direct, no categories lost)
    python convert_annotations.py \
        --source seaclear --target binary \
        --input seaclear_data/seaclear_train.json \
        --output seaclear_data/seaclear_train_binary.json

    # Batch convert (glob pattern)
    python convert_annotations.py \
        --source trashcan --target tc20 \
        --input "trashcan_data/.../instances_*.json" \
        --output-suffix _tc20
"""

import argparse
import glob
from pathlib import Path

from src.category_groups import get_scheme, remap_annotation_file, available_targets, available_sources


def main():
    """CLI entry point: remap a COCO annotation file from one category space to another."""
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations between category spaces."
    )
    parser.add_argument(
        "--source", required=True,
        choices=available_sources(),
        help="Source category space of the input annotations",
    )
    parser.add_argument(
        "--target", required=True,
        choices=available_targets(),
        help="Target category space for the output annotations",
    )
    parser.add_argument(
        "--input", required=True, type=str,
        help="Input annotation JSON path (or glob pattern for batch mode)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output annotation JSON path (single file mode)",
    )
    parser.add_argument(
        "--output-suffix", type=str, default=None,
        help="Suffix to append before .json for batch mode "
        "(e.g. '_tc20' turns 'train.json' into 'train_tc20.json')",
    )
    args = parser.parse_args()

    if args.output is None and args.output_suffix is None:
        parser.error("Either --output or --output-suffix must be provided")

    scheme = get_scheme(args.target, source=args.source)
    mapping = scheme["mapping"]
    categories = scheme["categories"]

    # Resolve input files (support glob patterns)
    input_paths = sorted(glob.glob(args.input))
    if not input_paths:
        # Try as literal path
        input_paths = [args.input]

    if len(input_paths) > 1 and args.output is not None:
        parser.error("--output cannot be used with multiple input files; use --output-suffix instead")

    print(f"Converting: {args.source} → {args.target}")
    print(f"  {len(mapping)} source categories mapped → {len(categories)} target categories")
    print(f"  {len(input_paths)} file(s) to convert")

    for input_path in input_paths:
        if not Path(input_path).exists():
            print(f"\n  Skipping {input_path} (not found)")
            continue

        if args.output is not None:
            output_path = args.output
        else:
            p = Path(input_path)
            output_path = str(p.with_name(p.stem + args.output_suffix + p.suffix))

        print(f"\n  {input_path}")
        print(f"  → {output_path}")

        stats = remap_annotation_file(input_path, output_path, mapping, categories)
        print(f"  Annotations: {stats['orig_anns']} → {stats['kept_anns']} "
              f"({stats['dropped']} dropped)")

        if stats["per_category_counts"]:
            print("  Per-category counts:")
            for name, count in sorted(stats["per_category_counts"].items()):
                print(f"    {name:<30s} {count:>5d}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
