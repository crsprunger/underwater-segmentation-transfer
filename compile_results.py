"""
Run cross-dataset evaluation for all model × dataset combinations and
compile results into a single summary JSON for the Streamlit app.

This script is meant to be run on the main machine where checkpoint .pth
files are available. It orchestrates three stages:

  1. Cross-dataset evaluation (cross_dataset_eval.py) for each
     (model, target_dataset) pair → results/eval/*.json
  2. Feature extraction and visualization (visualize_features.py)
     → results/feature_plots/*.npz + silhouette_scores.json
  3. Compile summary.json from all eval JSONs for the Streamlit app.

Usage:
    # Run everything (evals + features + compile)
    python compile_results.py

    # Evals only, then compile
    python compile_results.py --skip-features

    # Features only (no evals)
    python compile_results.py --skip-evals

    # Also generate best/worst prediction visualizations
    python compile_results.py --viz

    # Compile summary from already-generated eval JSONs (no GPU needed)
    python compile_results.py --compile-only

    # Run a specific subset of evals
    python compile_results.py --models tc sc --targets tc sc

Output structure:
    results/
      eval/
        tc_on_tc.json          # individual eval JSONs
        tc_on_sc.json
        ...
      feature_plots/
        proj_img_tsne_tc.npz   # t-SNE/UMAP projections for app.py
        proj_roi_tsne_tc.npz
        silhouette_scores.json
        ...
      viz/cross_dataset/       # best/worst prediction grids (--viz)
        tc_on_tc/best.png
        tc_on_sc/worst.png
        ...
      summary.json             # compiled summary for app.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from registry import (
    MODELS,
    TARGETS,
    CATEGORY_GROUPS,
    MODEL_VERSION,
    FEATURE_SPLITS,
    FEATURE_COMPARISONS,
    FEATURE_TC_IMG,
    FEATURE_SC_IMG,
)

# Ensure subprocesses can import `src` regardless of which script is invoked.
_ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)}


# ── Eval matrix configuration ────────────────────────────────────────

EVAL_DIR = Path("results/eval")
FEATURE_DIR = Path("results/feature_plots")
VIZ_DIR = Path("results/viz/cross_dataset")


# ── Helpers ───────────────────────────────────────────────────────────


def eval_output_path(model_key: str, target_key: str) -> Path:
    """Return the canonical path for one eval cell's JSON output."""
    return EVAL_DIR / f"{model_key}_on_{target_key}.json"


def run_eval(model_key: str, target_key: str) -> bool:
    """Run cross_dataset_eval.py for one (model, target) pair. Returns True on success."""
    model_name = MODELS[model_key][0]
    target_ann, target_img, _ = TARGETS[target_key]
    out_path = eval_output_path(model_key, target_key)

    # Skip if output already exists
    if out_path.exists():
        print(f"  ✓ {out_path.name} already exists, skipping")
        return True

    cmd = [
        sys.executable,
        "evaluation/cross_dataset_eval.py",
        "--model-name",
        model_name,
        "--model-version",
        MODEL_VERSION,
        "--target-ann",
        target_ann,
        "--target-img",
        target_img,
        "--category-group",
        *CATEGORY_GROUPS,
        "--output",
        str(out_path),
    ]

    print(f"  Running: {model_key} on {target_key} ...")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=_ENV)

    if result.returncode != 0:
        print(f"  ✗ Failed (exit code {result.returncode})")
        return False

    print(f"  ✓ Saved to {out_path}")
    return True


def run_viz(
    model_key: str,
    target_key: str,
    top_k: int = 10,
    bottom_k: int = 5,
    min_gap: int = 10,
) -> bool:
    """Run visualize_cross_dataset.py for one (model, target) pair. Returns True on success."""
    model_name = MODELS[model_key][0]
    out_dir = VIZ_DIR / f"{model_key}_on_{target_key}"
    target_ann, target_img, _ = TARGETS[target_key]

    if (model_key.startswith("tc") and target_key.startswith("tc")) or (
        model_key.startswith("sc")
        and target_key.startswith("sc")
        or model_key.startswith("pooled")
    ):
        min_gap = int(min_gap * 1.5)

    # Skip if outputs already exist
    if (out_dir / "best.png").exists() and (out_dir / "worst.png").exists():
        print(f"  ✓ {out_dir.name} already exists, skipping")
        return True

    cmd = [
        sys.executable,
        "visualization/visualize_cross_dataset.py",
        "--model-name",
        model_name,
        "--model-version",
        MODEL_VERSION,
        "--target-ann",
        target_ann,
        "--target-img",
        target_img,
        "--out-dir",
        str(out_dir),
        "--top-k",
        str(top_k),
        "--bottom-k",
        str(bottom_k),
        "--min-gap",
        str(min_gap),
    ]

    print(f"  Running viz: {model_key} on {target_key} ...")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=_ENV)

    if result.returncode != 0:
        print(f"  ✗ Failed (exit code {result.returncode})")
        return False

    print(f"  ✓ Saved to {out_dir}")
    return True


def _build_model_specs(model_keys: list[str]) -> list[str]:
    """Build --models args for visualize_features.py (name:ckpt:config:source)."""
    specs = []
    for key in model_keys:
        model_name = MODELS[key][0]
        ckpt = str(Path("checkpoints") / model_name / MODEL_VERSION / "best_model.pth")
        config = str(
            Path("checkpoints")
            / model_name
            / MODEL_VERSION
            / "training_config_full.yaml"
        )
        specs.append(f"{key}:{ckpt}:{config}:{model_name}")
    return specs


def run_features(
    comparisons: list[str] | None = None,
    splits: list[str] | None = None,
) -> None:
    """Run visualize_features.py for each (comparison, split) combination.

    Each run saves .npz projections, PNGs, and silhouette_scores.json
    to results/feature_plots/{comparison}/{split}/.
    """
    comparisons = comparisons or list(FEATURE_COMPARISONS.keys())
    splits = splits or list(FEATURE_SPLITS.keys())

    n_total = len(comparisons) * len(splits)
    n_success, n_fail = 0, 0

    for comp_name in comparisons:
        if comp_name not in FEATURE_COMPARISONS:
            print(f"  Unknown comparison: {comp_name}")
            continue
        comp_models = FEATURE_COMPARISONS[comp_name]
        model_specs = _build_model_specs(comp_models)

        for split_name in splits:
            if split_name not in FEATURE_SPLITS:
                print(f"  Unknown split: {split_name}")
                continue

            out_dir = FEATURE_DIR / comp_name / split_name
            split_cfg = FEATURE_SPLITS[split_name]

            # Skip if output already exists (check for silhouette_scores.json)
            if (out_dir / "silhouette_scores.json").exists():
                print(f"  ✓ {comp_name}/{split_name} already exists, skipping")
                n_success += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "visualization/visualize_features.py",
                "--models",
                *model_specs,
                "--tc-ann",
                *split_cfg["tc_ann"],
                "--tc-img",
                FEATURE_TC_IMG,
                "--sc-ann",
                *split_cfg["sc_ann"],
                "--sc-img",
                FEATURE_SC_IMG,
                "--output-dir",
                str(out_dir),
            ]

            print(f"  Running features: {comp_name}/{split_name} ...")
            print(f"    {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False, env=_ENV)

            if result.returncode != 0:
                print(f"  ✗ Failed (exit code {result.returncode})")
                n_fail += 1
            else:
                print(f"  ✓ Saved to {out_dir}")
                n_success += 1

    print(
        f"\nFeature extraction complete: {n_success}/{n_total} succeeded, {n_fail} failed"
    )


def compile_summary() -> dict:
    """Read all eval JSONs and compile into a single summary dict."""
    summary = {
        "models": {},
        "eval_cells": {},
    }

    # Model metadata
    for key, (model_name, label) in MODELS.items():
        config_path = (
            Path("checkpoints")
            / model_name
            / MODEL_VERSION
            / "training_config_full.yaml"
        )
        summary["models"][key] = {
            "model_name": model_name,
            "label": label,
            "config": str(config_path),
        }

    # Load each eval JSON
    for model_key in MODELS:
        for target_key in TARGETS:
            cell_name = f"{model_key}_on_{target_key}"
            out_path = eval_output_path(model_key, target_key)

            if not out_path.exists():
                print(f"  ? {out_path.name} not found, skipping")
                continue

            with open(out_path) as f:
                data = json.load(f)

            _, target_label = TARGETS[target_key][2], TARGETS[target_key][2]
            model_label = MODELS[model_key][1]

            cell = {
                "model": model_key,
                "model_label": model_label,
                "target": target_key,
                "target_label": target_label,
                "eval_space": data.get("eval_space"),
                "metrics": data.get("metrics", {}),
            }

            # Include grouped metrics if present
            if "grouped_metrics" in data:
                cell["grouped_metrics"] = data["grouped_metrics"]

            summary["eval_cells"][cell_name] = cell

    return summary


# ── Main ──────────────────────────────────────────────────────────────


def main():
    """CLI entry point: run evals, extract features, and compile summary.json."""
    parser = argparse.ArgumentParser(
        description="Run cross-dataset evaluations and compile results summary."
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile summary from existing eval JSONs (no GPU needed)",
    )
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="Skip cross-dataset evaluations",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature extraction and visualization",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Models to evaluate (default: all). Choices: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help=f"Target datasets (default: all). Choices: {list(TARGETS.keys())}",
    )
    parser.add_argument(
        "--comparisons",
        nargs="*",
        default=None,
        help=f"Feature comparisons to run (default: all). Choices: {list(FEATURE_COMPARISONS.keys())}",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help=f"Feature splits to run (default: all). Choices: {list(FEATURE_SPLITS.keys())}",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Generate best/worst prediction visualizations (disabled by default)",
    )
    parser.add_argument(
        "--viz-top-k",
        type=int,
        default=10,
        help="Number of best images to visualize (default: 10)",
    )
    parser.add_argument(
        "--viz-bottom-k",
        type=int,
        default=5,
        help="Number of worst images to visualize (default: 5)",
    )
    parser.add_argument(
        "--viz-min-gap",
        type=int,
        default=10,
        help="Min filename-index gap between selected examples (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even if output already exists",
    )
    args = parser.parse_args()

    model_keys = args.models or list(MODELS.keys())
    target_keys = args.targets or list(TARGETS.keys())

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Cross-dataset evaluations
    if not args.compile_only and not args.skip_evals:
        print(f"Running {len(model_keys)} × {len(target_keys)} evaluations ...\n")
        n_success, n_fail = 0, 0

        for model_key in model_keys:
            if model_key not in MODELS:
                print(f"  Unknown model: {model_key}")
                continue
            for target_key in target_keys:
                if target_key not in TARGETS:
                    print(f"  Unknown target: {target_key}")
                    continue

                if args.force:
                    out = eval_output_path(model_key, target_key)
                    if out.exists():
                        out.unlink()

                ok = run_eval(model_key, target_key)
                if ok:
                    n_success += 1
                else:
                    n_fail += 1

        print(f"\nEvaluation complete: {n_success} succeeded, {n_fail} failed")

    # Stage 2: Feature extraction and visualization
    if not args.compile_only and not args.skip_features:
        print()
        run_features(comparisons=args.comparisons, splits=args.splits)

    # Stage 3: Prediction visualizations (opt-in)
    if not args.compile_only and args.viz:
        print("\nGenerating prediction visualizations ...\n")
        n_success, n_fail = 0, 0

        for model_key in model_keys:
            if model_key not in MODELS:
                continue
            for target_key in target_keys:
                if target_key not in TARGETS:
                    continue

                if args.force:
                    out_dir = VIZ_DIR / f"{model_key}_on_{target_key}"
                    for f in ("best.png", "worst.png", "per_image_scores.csv"):
                        p = out_dir / f
                        if p.exists():
                            p.unlink()

                ok = run_viz(
                    model_key,
                    target_key,
                    top_k=args.viz_top_k,
                    bottom_k=args.viz_bottom_k,
                    min_gap=args.viz_min_gap,
                )
                if ok:
                    n_success += 1
                else:
                    n_fail += 1

        print(f"\nVisualization complete: {n_success} succeeded, {n_fail} failed")

    # Compile summary
    print("\nCompiling summary ...")
    summary = compile_summary()

    summary_path = Path("results/summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\nSummary saved to {summary_path}")
    print(f"  {len(summary['eval_cells'])} eval cells\n")

    # Print a quick mAP table
    header = f"  {'':>20s}"
    for tk in target_keys:
        header += f"  {TARGETS[tk][2]:>10s}"
    print(header)
    print(f"  {'─'*20}" + f"  {'─'*10}" * len(target_keys))

    for mk in model_keys:
        row = f"  {MODELS[mk][1]:>20s}"
        for tk in target_keys:
            cell = summary["eval_cells"].get(f"{mk}_on_{tk}")
            if cell and "mask_mAP" in cell.get("metrics", {}):
                row += f"  {cell['metrics']['mask_mAP']:>10.4f}"
            else:
                row += f"  {'—':>10s}"
        print(row)

    print()


if __name__ == "__main__":
    main()
