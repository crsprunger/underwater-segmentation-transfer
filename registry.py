"""Centralized registry of models, datasets, and display constants.

This module is the single source of truth for model short keys, checkpoint
directory names, display labels, dataset paths, and feature comparison
definitions.  Imported by compile_results.py, app.py, and any future
evaluation or visualization scripts.

Nothing in this module depends on PyTorch or any heavy library — it is
pure Python dicts and strings so it can be imported cheaply anywhere.
"""

from src.category_groups import (  # noqa: F401 — re-exported
    TC20_NAMES,
    TC20_CATEGORIES,
    COARSE_CATEGORIES,
    TERNARY_CATEGORIES,
    BINARY_CATEGORIES,
)

# Category name dicts for each evaluation space: {id: name}
COARSE_NAMES = {k: v["name"] for k, v in COARSE_CATEGORIES.items()}
TERNARY_NAMES = {k: v["name"] for k, v in TERNARY_CATEGORIES.items()}
BINARY_NAMES = {k: v["name"] for k, v in BINARY_CATEGORIES.items()}

# Lookup by space name string
CATEGORY_NAMES_BY_SPACE = {
    "tc20": TC20_NAMES,
    "coarse": COARSE_NAMES,
    "ternary": TERNARY_NAMES,
    "binary": BINARY_NAMES,
}

# ═══════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════
#
# Short key → (checkpoint_dir_name, display_label)
#
# The short key is used in filenames (eval JSONs, feature .npz files)
# and as the canonical identifier across scripts.  The checkpoint_dir_name
# is the subdirectory under checkpoints/ that holds the weights.

MODELS = {
    "tc": ("trashcan", "TC model"),
    "sc": ("seaclear", "SC model"),
    "tc_chunksplit": ("trashcan_chunksplit", "TC model"),
    "sc_chunksplit": ("seaclear_chunksplit", "SC model"),
    "tc_chunksplit_tc20": ("trashcan_chunksplit_tc20", "TC (TC20) model"),
    "sc_chunksplit_tc20": ("seaclear_chunksplit_tc20", "SC (TC20) model"),
    "tc_chunksplit_coarse": ("trashcan_chunksplit_coarse", "TC (coarse) model"),
    "sc_chunksplit_coarse": ("seaclear_chunksplit_coarse", "SC (coarse) model"),
    "pooled_tc20": ("pooled_chunksplit_tc20", "Pooled model"),
    "pooled_coarse": ("pooled_chunksplit_coarse", "Pooled (coarse) model"),
    "sc_chunksplit+overlay": ("seaclear_chunksplit_overlay", "SC+overlay model"),
    "sc_chunksplit+overlay_tc20": (
        "seaclear_chunksplit_overlay_tc20",
        "SC+overlay (TC20) model",
    ),
    "sc_chunksplit+grayscale": (
        "seaclear_chunksplit_grayscale0.4",
        "SC+grayscale model",
    ),
    "sc_chunksplit+grayscale_tc20": (
        "seaclear_chunksplit_grayscale0.4_tc20",
        "SC+grayscale (TC20) model",
    ),
}

# Derived lookups
MODEL_LABELS = {key: label for key, (_, label) in MODELS.items()}
"""Short key → display label (e.g. 'tc_chunksplit' → 'TC model')."""

CKPT_DIR_TO_KEY = {ckpt_dir: key for key, (ckpt_dir, _) in MODELS.items()}
"""Checkpoint directory name → short key (e.g. 'trashcan_chunksplit' → 'tc_chunksplit')."""

CKPT_DIR_TO_LABEL = {ckpt_dir: label for _, (ckpt_dir, label) in MODELS.items()}
"""Checkpoint directory name → display label (e.g. 'trashcan_chunksplit' → 'TC model')."""


# ═══════════════════════════════════════════════════════════════════════
# Target datasets (for evaluation)
# ═══════════════════════════════════════════════════════════════════════
#
# Short key → (annotation_json, image_dir, display_label)

TARGETS = {
    "tc_test": (
        "trashcan_data/trashcan_test.json",
        "trashcan_data/images",
        "TC test",
    ),
    "sc_test": (
        "seaclear_data/seaclear_480p_test.json",
        "seaclear_data/images_480p",
        "SC test",
    ),
    "tc_chunksplit_test": (
        "trashcan_data/trashcan_chunksplit_test.json",
        "trashcan_data/images",
        "TC chunksplit test",
    ),
    "sc_chunksplit_test": (
        "seaclear_data/seaclear_480p_chunksplit_test.json",
        "seaclear_data/images_480p",
        "SC chunksplit test",
    ),
}

TARGET_LABELS = {key: label for key, (_, _, label) in TARGETS.items()}
"""Short key → display label (e.g. 'tc_chunksplit_test' → 'TC chunksplit test')."""

# Pretty names for the raw dataset strings found in eval JSONs
DS_DISPLAY = {
    "trashcan": "TC",
    "seaclear": "SC",
    "tc20": "TC20",
}


# ═══════════════════════════════════════════════════════════════════════
# Display / UI constants
# ═══════════════════════════════════════════════════════════════════════

DS_COLORS = {"tc": "#2196F3", "sc": "#FF5722"}
DS_NAMES = {"tc": "TrashCan", "sc": "SeaClear"}


# ═══════════════════════════════════════════════════════════════════════
# Feature visualization configuration
# ═══════════════════════════════════════════════════════════════════════

# Image directories (constant across all splits)
FEATURE_TC_IMG = "trashcan_data/images"
FEATURE_SC_IMG = "seaclear_data/images_480p"

# Annotation files for feature extraction, keyed by split name.
# Each split maps to {tc_ann: [...], sc_ann: [...]}.
FEATURE_SPLITS = {
    "train_tc20": {
        "tc_ann": ["trashcan_data/trashcan_train_tc20.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_train_tc20.json"],
    },
    "val_tc20": {
        "tc_ann": ["trashcan_data/trashcan_val_tc20.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_val_tc20.json"],
    },
    "test_tc20": {
        "tc_ann": ["trashcan_data/trashcan_test_tc20.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_test_tc20.json"],
    },
    "chunksplit_train_tc20": {
        "tc_ann": ["trashcan_data/trashcan_chunksplit_train_tc20.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_chunksplit_train_tc20.json"],
    },
    "chunksplit_val_tc20": {
        "tc_ann": ["trashcan_data/trashcan_chunksplit_val_tc20.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_chunksplit_val_tc20.json"],
    },
    "chunksplit_test_tc20": {
        "tc_ann": ["trashcan_data/trashcan_chunksplit_test_tc20.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_chunksplit_test_tc20.json"],
    },
    "chunksplit_train_coarse": {
        "tc_ann": ["trashcan_data/trashcan_chunksplit_train_coarse.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_chunksplit_train_coarse.json"],
    },
    "chunksplit_val_coarse": {
        "tc_ann": ["trashcan_data/trashcan_chunksplit_val_coarse.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_chunksplit_val_coarse.json"],
    },
    "chunksplit_test_coarse": {
        "tc_ann": ["trashcan_data/trashcan_chunksplit_test_coarse.json"],
        "sc_ann": ["seaclear_data/seaclear_480p_chunksplit_test_coarse.json"],
    },
}

# Comparisons: which model sets to visualize together.
# Each comparison maps to a list of model short keys.
FEATURE_COMPARISONS = {
    "naive_split": ["tc", "sc"],
    "chunksplit": ["tc_chunksplit", "sc_chunksplit", "pooled_tc20"],
    "chunksplit+grayscale": [
        "tc_chunksplit",
        "sc_chunksplit",
        "sc_chunksplit+grayscale",
    ],
    "chunksplit+overlay": [
        "tc_chunksplit",
        "sc_chunksplit",
        "sc_chunksplit+overlay",
    ],
    "chunksplit_tc20-trained": [
        "tc_chunksplit_tc20",
        "sc_chunksplit_tc20",
        "pooled_tc20",
    ],
    "chunksplit_coarse-trained": [
        "tc_chunksplit_coarse",
        "sc_chunksplit_coarse",
        "pooled_coarse",
    ],
}

# Category groupings to evaluate in addition to the primary eval space
CATEGORY_GROUPS = ["coarse", "ternary", "binary"]

# Default model version subdirectory under each checkpoint directory
MODEL_VERSION = "model6v2"
