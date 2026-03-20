"""
Streamlit demo: Cross-Dataset Underwater Instance Segmentation
Run with: uv run streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from registry import (
    CATEGORY_NAMES_BY_SPACE,
    CKPT_DIR_TO_LABEL,
    DS_COLORS,
    DS_DISPLAY,
    DS_NAMES,
    MODEL_LABELS,
    TC20_NAMES,
)

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Underwater Instance Segmentation — Cross-Dataset Analysis",
    layout="wide",
)

FEATURE_DIR = Path("results/feature_plots")
EVAL_DIR = Path("results/eval")
VIZ_DIR = Path("results/viz/cross_dataset")


def _model_label(checkpoint_path: str, source_ds: str) -> str:
    """Derive a human-readable model label from the checkpoint path."""
    parts = Path(checkpoint_path).parts
    model_dir = parts[1] if len(parts) > 1 else source_ds
    return CKPT_DIR_TO_LABEL.get(model_dir, model_dir)


@st.cache_data
def load_all_eval_cells() -> list[dict]:
    """Scan results/eval/*.json and return a list of eval cell dicts.

    Each dict has: model_label, target_label, metrics (top-level),
    and grouped_metrics (coarse/ternary/binary sub-dicts).
    """
    cells = []
    if not EVAL_DIR.exists():
        return cells
    for p in sorted(EVAL_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        ckpt = data.get("checkpoint", "")
        src = data.get("source_dataset", "?")
        tgt = data.get("target_dataset", "?")
        cells.append({
            "model_label": _model_label(ckpt, src),
            "target_label": f"{DS_DISPLAY.get(tgt, tgt)} test",
            "metrics": data.get("metrics", {}),
            "grouped_metrics": data.get("grouped_metrics", {}),
        })
    return cells


EVAL_CELLS_DATA = load_all_eval_cells()


# ═══════════════════════════════════════════════════════════════════════
# Data loaders (cached)
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data
def load_proj(feature_dir: str, emb_type: str, method: str, model: str):
    """Load projected 2D embeddings (.npz with emb_tc, emb_sc, and
    optionally labels_tc/labels_sc for ROI embeddings)."""
    path = Path(feature_dir) / f"proj_{emb_type}_{method}_{model}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


@st.cache_data
def load_silhouette(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_training_csv(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def discover_training_csvs():
    """Return {display_name: path} for all training_history.csv files."""
    paths = sorted(Path("checkpoints").glob("*/*/training_history.csv"))
    result = {}
    for p in paths:
        name = f"{p.parts[1]} / {p.parts[2]}"
        result[name] = str(p)
    return result


# ═══════════════════════════════════════════════════════════════════════
# App header
# ═══════════════════════════════════════════════════════════════════════

st.title("Cross-Dataset Underwater Instance Segmentation")
st.caption(
    "Do models trained on one underwater trash dataset generalize to another? "
    "This demo presents evaluation results, backbone feature visualizations, "
    "and prediction examples across TrashCan and SeaClear datasets."
)

tab_results, tab_features, tab_predictions, tab_curves = st.tabs(
    [
        "📊 Cross-Dataset Results",
        "🔍 Feature Space",
        "🖼️ Prediction Examples",
        "📈 Training Curves",
    ]
)


# ═══════════════════════════════════════════════════════════════════════
# Tab 1 — Cross-Dataset Results
# ═══════════════════════════════════════════════════════════════════════

with tab_results:
    st.header("Cross-Dataset Evaluation Results")

    if not EVAL_CELLS_DATA:
        st.info("No evaluation results found. Place eval JSONs in results/eval/.")
    else:
        # ── Discover available category spaces and metrics ────────────
        all_spaces = ["primary"]
        for cell in EVAL_CELLS_DATA:
            all_spaces.extend(cell["grouped_metrics"].keys())
        all_spaces = list(dict.fromkeys(all_spaces))  # deduplicate, preserve order

        SPACE_LABELS = {
            "primary": "Primary (native eval space)",
            "coarse": "Coarse (5 classes)",
            "ternary": "Ternary (3 classes)",
            "binary": "Binary (2 classes)",
        }
        METRIC_OPTIONS = {
            "mask_mAP": "Mask mAP",
            "mask_mAP_50": "Mask mAP@50",
            "mask_mAP_75": "Mask mAP@75",
            "box_mAP": "Box mAP",
            "box_mAP_50": "Box mAP@50",
        }

        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            space_sel = st.selectbox(
                "Category space",
                all_spaces,
                format_func=lambda x: SPACE_LABELS.get(x, x),
            )
        with ctrl2:
            metric_sel = st.selectbox(
                "Metric",
                list(METRIC_OPTIONS.keys()),
                format_func=lambda x: METRIC_OPTIONS.get(x, x),
            )

        # ── Build results table for selected space + metric ──────────
        def _get_metric(cell: dict, space: str, metric: str) -> float | None:
            if space == "primary":
                return cell["metrics"].get(metric)
            return cell["grouped_metrics"].get(space, {}).get(metric)

        results: dict[str, dict[str, float | None]] = {}
        for cell in EVAL_CELLS_DATA:
            ml = cell["model_label"]
            tl = cell["target_label"]
            if ml not in results:
                results[ml] = {}
            results[ml][tl] = _get_metric(cell, space_sel, metric_sel)

        all_models = sorted(results.keys())
        all_targets = sorted({t for m in results.values() for t in m})

        models_sel = st.multiselect(
            "Models", all_models, default=all_models, key="results_models",
        )
        if not models_sel:
            st.info("Select at least one model above.")
        else:
            filtered = {m: results[m] for m in models_sel}
            df = pd.DataFrame(filtered).T
            df.index.name = "Model"
            df_display = df.apply(
                lambda col: col.map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "pending"
                ),
            )

            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.subheader("Results table")
                st.dataframe(df_display, width="stretch")
                st.caption("Rows = model; columns = evaluation target.")

            with col_right:
                metric_label = METRIC_OPTIONS.get(metric_sel, metric_sel)
                st.subheader(f"{metric_label} by model")

                fig = go.Figure()
                colors = list(DS_COLORS.values()) + ["#4CAF50", "#9C27B0"]
                for i, target in enumerate(all_targets):
                    vals = [filtered[m].get(target) for m in models_sel]
                    fig.add_trace(
                        go.Bar(
                            name=target,
                            x=[MODEL_LABELS.get(m, m) for m in models_sel],
                            y=vals,
                            marker_color=colors[i % len(colors)],
                            text=[f"{v:.4f}" if v else "" for v in vals],
                            textposition="outside",
                        )
                    )
                y_max = max(
                    (v for m in filtered.values() for v in m.values() if v is not None),
                    default=0.4,
                )
                fig.update_layout(
                    barmode="group",
                    yaxis_title=metric_label,
                    yaxis_range=[0, y_max * 1.2],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=380,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════
# Tab 2 — Feature Space
# ═══════════════════════════════════════════════════════════════════════

with tab_features:
    st.header("Backbone Feature Space Visualization")
    st.markdown(
        "Features extracted from **ResNet50 layer4** "
        "(global average pool for image-level; ROI-aligned 7×7 crop for per-object). "
        "Projected to 2D with t-SNE or UMAP. "
        "A high silhouette score means the model has learned to separate TC from SC — "
        "even within the same object class."
    )

    # ── Discover available comparisons and splits ─────────────────────
    available_comparisons = (
        sorted(
            d.name
            for d in FEATURE_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if FEATURE_DIR.exists()
        else []
    )

    if not available_comparisons:
        st.info(
            "No feature visualizations found. Run compile_results.py to generate them."
        )
    else:
        comp_col, split_col = st.columns(2)
        with comp_col:
            comparison_sel = st.selectbox(
                "Model group",
                available_comparisons,
            )
        comp_dir = FEATURE_DIR / comparison_sel
        available_splits = (
            sorted(
                d.name
                for d in comp_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            if comp_dir.exists()
            else []
        )
        with split_col:
            split_sel = st.selectbox(
                "Data split",
                available_splits,
                index=next(
                    (i for i, s in enumerate(available_splits) if "test" in s), 0
                ),
            )

        feature_subdir = comp_dir / split_sel

        # ── Detect category space from split name ────────────────────
        # e.g. "chunksplit_test_coarse" → "coarse", "chunksplit_val_tc20" → "tc20"
        _cat_space = "tc20"  # default
        for space in ("coarse", "ternary", "binary"):
            if split_sel.endswith(f"_{space}"):
                _cat_space = space
                break
        cat_names = CATEGORY_NAMES_BY_SPACE.get(_cat_space, TC20_NAMES)

        # ── Controls ──────────────────────────────────────────────────
        # Discover which models have .npz projections in this subdir.
        # Filenames are proj_{emb}_{method}_{model_key}.npz where model_key
        # may contain underscores (e.g. pooled_tc20), so we strip the known
        # 4-token prefix to recover it.
        def _extract_model_key(stem: str) -> str:
            parts = stem.split("_", 3)  # ['proj', 'img', 'tsne', 'pooled_tc20']
            return parts[3] if len(parts) == 4 else stem

        available_models = (
            sorted(
                {
                    _extract_model_key(p.stem)
                    for p in feature_subdir.glob("proj_img_*.npz")
                }
            )
            if feature_subdir.exists()
            else list(MODEL_LABELS.keys())
        )

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            models_sel = st.multiselect(
                "Models",
                available_models,
                default=available_models,
                format_func=lambda x: MODEL_LABELS.get(x, x),
                key="features_models",
            )
        with ctrl2:
            method_sel = st.selectbox(
                "Projection", ["tsne", "umap"], format_func=lambda x: x.upper()
            )
        with ctrl3:
            emb_type_sel = st.radio(
                "Embedding type",
                ["img", "roi"],
                format_func=lambda x: (
                    "Image-level (GAP)" if x == "img" else "Per-object (ROI)"
                ),
            )

        # Class filter — lets the user hide classes to declutter the plot
        all_classes = sorted(cat_names.keys())
        if emb_type_sel == "roi":
            class_filter = st.multiselect(
                "Classes to show",
                options=all_classes,
                default=all_classes,
                format_func=lambda x: cat_names.get(x, f"id_{x}"),
            )
        else:
            class_filter = all_classes

        # ── Render one plot per selected model, side by side ──────────
        if not models_sel:
            st.info("Select at least one model above.")
        else:
            sil_all = load_silhouette(str(feature_subdir / "silhouette_scores.json"))

            cols = st.columns(len(models_sel))
            for col, model_key in zip(cols, models_sel):
                data = load_proj(str(feature_subdir), emb_type_sel, method_sel, model_key)
                with col:
                    if data is None:
                        st.info(
                            f"No {method_sel.upper()} data for "
                            f"{MODEL_LABELS.get(model_key, model_key)}."
                        )
                        continue

                    emb_tc = data["emb_tc"]
                    emb_sc = data["emb_sc"]
                    labels_tc = data.get("labels_tc")
                    labels_sc = data.get("labels_sc")

                    # Filter by selected classes (ROI only)
                    if emb_type_sel == "roi" and labels_tc is not None:
                        mask_tc = np.isin(labels_tc, class_filter)
                        mask_sc = np.isin(labels_sc, class_filter) if labels_sc is not None else np.ones(len(emb_sc), dtype=bool)
                        emb_tc = emb_tc[mask_tc]
                        emb_sc = emb_sc[mask_sc]

                    fig = go.Figure()
                    for ds_key, emb in [("tc", emb_tc), ("sc", emb_sc)]:
                        fig.add_trace(
                            go.Scattergl(
                                x=emb[:, 0],
                                y=emb[:, 1],
                                mode="markers",
                                name=DS_NAMES[ds_key],
                                marker=dict(color=DS_COLORS[ds_key], size=3, opacity=0.5),
                                hovertemplate=f"<b>{DS_NAMES[ds_key]}</b><extra></extra>",
                            )
                        )

                    sil_key = f"{'image' if emb_type_sel == 'img' else 'roi'}_{model_key}"
                    ds_sil = sil_all.get(sil_key, {}).get("dataset")
                    sil_str = f"  (sil: {ds_sil:.3f})" if ds_sil is not None else ""

                    fig.update_layout(
                        title=f"{MODEL_LABELS.get(model_key, model_key)}{sil_str}",
                        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        height=450,
                        showlegend=(model_key == models_sel[0]),  # legend on first plot only
                        margin=dict(t=40, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════
# Tab 3 — Prediction Examples
# ═══════════════════════════════════════════════════════════════════════

with tab_predictions:
    st.header("Prediction Examples")
    st.markdown(
        "Best and worst predictions per eval cell, "
        "ranked by per-image greedy IoU matching at 0.5 threshold."
    )

    # Discover eval cells from results/viz/cross_dataset/ subdirectories
    EVAL_CELLS = {}
    if VIZ_DIR.exists():
        for subdir in sorted(VIZ_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            # e.g. "tc_on_tc" → "TC model → TC test"
            parts = subdir.name.split("_on_")
            if len(parts) == 2:
                model_key, target_key = parts
                model_label = {
                    "tc": "TC model",
                    "sc": "SC model",
                    "pooled": "Pooled model",
                    "sc-o": "SC+overlay model",
                }.get(model_key, model_key)
                target_label = {"tc": "TC", "sc": "SC"}.get(target_key, target_key)
                domain = (
                    "in-domain"
                    if model_key.startswith(target_key) or (model_key == "pooled")
                    else "cross-domain"
                )
                EVAL_CELLS[f"{model_label} → {target_label} test ({domain})"] = (
                    subdir.name
                )
    if not EVAL_CELLS:
        # Fallback to expected cells so the UI still shows something
        EVAL_CELLS = {
            "TC model → TC test (in-domain)": "tc_on_tc",
            "TC model → SC test (cross-domain)": "tc_on_sc",
            "SC model → SC test (in-domain)": "sc_on_sc",
            "SC model → TC test (cross-domain)": "sc_on_tc",
            "Pooled model → TC test": "pooled_on_tc",
            "Pooled model → SC test": "pooled_on_sc",
        }

    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        cell_sel = st.selectbox("Evaluation cell", list(EVAL_CELLS.keys()))
    with pred_col2:
        quality_sel = st.radio("Prediction quality", ["best", "worst"], horizontal=True)

    img_path = VIZ_DIR / EVAL_CELLS[cell_sel] / f"{quality_sel}.png"

    if img_path.exists():
        st.image(str(img_path), use_column_width=True)
        quality_word = "highest" if quality_sel == "best" else "lowest"
        st.caption(
            f"**{cell_sel} — {quality_sel} predictions.** "
            f"Images with the {quality_word} per-image match rate "
            f"(fraction of GT boxes with an IoU ≥ 0.5 prediction)."
        )
    else:
        st.info(f"Results pending — {img_path} not found.")


# ═══════════════════════════════════════════════════════════════════════
# Tab 4 — Training Curves
# ═══════════════════════════════════════════════════════════════════════

with tab_curves:
    st.header("Training Curves")

    csv_map = discover_training_csvs()
    if not csv_map:
        st.info("No training_history.csv files found under checkpoints/.")
    else:
        selected_models = st.multiselect(
            "Models to compare",
            options=list(csv_map.keys()),
            default=list(csv_map.keys())[:3],
        )

        metric_options = {
            "Mask mAP": "mask_mAP",
            "Mask mAP@50": "mask_mAP_50",
            "Box mAP": "box_mAP",
            "Loss": "loss",
            "Mask mAP (small objects)": "mask_mAP_small",
        }
        metric_sel = st.selectbox("Metric", list(metric_options.keys()))
        metric_col = metric_options[metric_sel]

        if selected_models:
            fig = go.Figure()
            for name in selected_models:
                df = load_training_csv(csv_map[name])
                if df is None or metric_col not in df.columns:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=df["epoch"],
                        y=df[metric_col],
                        mode="lines",
                        name=name,
                    )
                )
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title=metric_sel,
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Select at least one model above.")
