"""
Unified category mapping module for TrashCan / SeaClear datasets.

Defines the full remapping chain:
  SeaClear original (40 cats) ──SC_TO_TC──> TrashCan original (22 cats)
  TrashCan original (22 cats) ──TC_TO_TC20──> TC20 (20 cats, contiguous 1-20)
  TC20 (20 cats) ──COARSE_MAPPING──> Coarse (5 cats: rov, organic, trash_easy,
                                      trash_entangled, trash_heavy)
  Coarse (5 cats) ─────────────────> Ternary (3 cats: rov, non_trash, trash)
  Ternary (3 cats) ────────────────> Binary (2 cats: non_trash, trash)

Composite mappings (e.g. SC → TC20) are built automatically via _compose().
Direct mappings (e.g. SC → coarse, TC → coarse) are defined explicitly to
avoid losing categories that are excluded from TC20.

Usage:
    from category_groups import get_scheme, remap_results, remap_coco_gt

    # Eval-time: remap predictions and GT to coarse categories
    scheme = get_scheme("coarse")
    grouped_gt = remap_coco_gt(coco_gt, scheme["mapping"], scheme["categories"])
    grouped_preds = remap_results(predictions, scheme["mapping"])

    # Annotation conversion: remap SeaClear annotations to TC20
    scheme = get_scheme("tc20", source="seaclear")
    # scheme["mapping"] maps SC original IDs → TC20 IDs

    # Or go directly from SeaClear to coarse:
    scheme = get_scheme("coarse", source="seaclear")
"""

# ═════════════════════════════════════════════════════════════════════
# Source category spaces
# ═════════════════════════════════════════════════════════════════════

# SeaClear original category ID → TrashCan original category ID
# Categories with no TC equivalent are absent (dropped during conversion)
SC_TO_TC = {
    1: 14,  # can_metal          -> trash_can
    2: 20,  # tarp_plastic       -> trash_tarp
    3: 16,  # container_plastic  -> trash_container
    4: 11,  # bottle_plastic     -> trash_bottle
    6: 2,  # plant              -> plant
    7: 16,  # container_middle_size_metal -> trash_container
    8: 8,  # animal_etc         -> animal_etc
    10: 11,  # bottle_glass       -> trash_bottle
    11: 19,  # wreckage_metal     -> trash_wreckage
    12: 17,  # unknown_instance   -> trash_unknown_instance
    13: 10,  # pipe_plastic       -> trash_pipe
    14: 22,  # net_plastic        -> trash_net
    15: 5,  # animal_shells      -> animal_shells
    16: 21,  # rope_fiber         -> trash_rope
    18: 15,  # cup_plastic        -> trash_cup
    20: 12,  # bag_plastic        -> trash_bag
    22: 9,  # clothing_fiber     -> trash_clothing
    23: 15,  # cup_ceramic        -> trash_cup
    27: 1,  # rov_cable          -> rov
    28: 1,  # rov_tortuga        -> rov
    29: 18,  # branch_wood        -> trash_branch
    31: 13,  # snack_wrapper_plastic -> trash_snack_wrapper
    34: 21,  # rope_plastic       -> trash_rope
    36: 3,  # animal_fish        -> animal_fish
    37: 13,  # snack_wrapper_paper -> trash_snack_wrapper
    38: 1,  # rov_vehicle_leg    -> rov
    39: 1,  # rov_bluerov        -> rov
    40: 4,  # animal_starfish    -> animal_starfish
}

# SeaClear category IDs with no TrashCan equivalent
SC_UNMAPPED = {5, 9, 17, 19, 21, 24, 25, 26, 30, 32, 33, 35}

# TrashCan original IDs excluded from TC20 (no SeaClear equivalent)
TC_EXCLUDED = {6, 7}  # animal_crab, animal_eel

# TrashCan original category ID → TC20 contiguous category ID
TC_TO_TC20 = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9,
    12: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
}

# ═════════════════════════════════════════════════════════════════════
# TC20 — the common evaluation space (20 contiguous categories)
# ═════════════════════════════════════════════════════════════════════

TC20_CATEGORIES = {
    1: {"id": 1, "name": "rov", "supercategory": "rov"},
    2: {"id": 2, "name": "plant", "supercategory": "plant"},
    3: {"id": 3, "name": "animal_fish", "supercategory": "animal_fish"},
    4: {"id": 4, "name": "animal_starfish", "supercategory": "animal_starfish"},
    5: {"id": 5, "name": "animal_shells", "supercategory": "animal_shells"},
    6: {"id": 6, "name": "animal_etc", "supercategory": "animal_etc"},
    7: {"id": 7, "name": "trash_clothing", "supercategory": "trash_clothing"},
    8: {"id": 8, "name": "trash_pipe", "supercategory": "trash_pipe"},
    9: {"id": 9, "name": "trash_bottle", "supercategory": "trash_bottle"},
    10: {"id": 10, "name": "trash_bag", "supercategory": "trash_bag"},
    11: {
        "id": 11,
        "name": "trash_snack_wrapper",
        "supercategory": "trash_snack_wrapper",
    },
    12: {"id": 12, "name": "trash_can", "supercategory": "trash_can"},
    13: {"id": 13, "name": "trash_cup", "supercategory": "trash_cup"},
    14: {"id": 14, "name": "trash_container", "supercategory": "trash_container"},
    15: {
        "id": 15,
        "name": "trash_unknown_instance",
        "supercategory": "trash_unknown_instance",
    },
    16: {"id": 16, "name": "trash_branch", "supercategory": "trash_branch"},
    17: {"id": 17, "name": "trash_wreckage", "supercategory": "trash_wreckage"},
    18: {"id": 18, "name": "trash_tarp", "supercategory": "trash_tarp"},
    19: {"id": 19, "name": "trash_rope", "supercategory": "trash_rope"},
    20: {"id": 20, "name": "trash_net", "supercategory": "trash_net"},
}

TC20_NAMES = {k: v["name"] for k, v in TC20_CATEGORIES.items()}

# Identity mapping for TC20 → TC20
TC20_MAPPING = {i: i for i in range(1, 21)}

# ═════════════════════════════════════════════════════════════════════
# Coarse scheme (5 categories, grouped by extraction difficulty)
# ═════════════════════════════════════════════════════════════════════

COARSE_CATEGORIES = {
    1: {"id": 1, "name": "rov", "supercategory": "rov"},
    2: {"id": 2, "name": "organic", "supercategory": "organic"},
    3: {"id": 3, "name": "trash_easy", "supercategory": "trash"},
    4: {"id": 4, "name": "trash_entangled", "supercategory": "trash"},
    5: {"id": 5, "name": "trash_heavy", "supercategory": "trash"},
}

# TC20 ID → coarse ID
COARSE_MAPPING = {
    # rov
    1: 1,  # rov
    # organic (living things, natural debris)
    2: 2,  # plant
    3: 2,  # animal_fish
    4: 2,  # animal_starfish
    5: 2,  # animal_shells
    6: 2,  # animal_etc
    16: 2,  # trash_branch (natural debris)
    # trash_easy (small, extractable with simple robotic arms)
    8: 3,  # trash_pipe (plastic)
    9: 3,  # trash_bottle
    10: 3,  # trash_bag
    11: 3,  # trash_snack_wrapper
    12: 3,  # trash_can
    13: 3,  # trash_cup
    14: 3,  # trash_container
    # trash_entangled (flexible, risk of snagging)
    7: 4,  # trash_clothing
    18: 4,  # trash_tarp
    19: 4,  # trash_rope
    20: 4,  # trash_net
    # trash_heavy (large/heavy, needs human review)
    15: 5,  # trash_unknown_instance
    17: 5,  # trash_wreckage
}

# ═════════════════════════════════════════════════════════════════════
# Ternary scheme (3 categories: rov / non_trash / trash)
# ═════════════════════════════════════════════════════════════════════

TERNARY_CATEGORIES = {
    1: {"id": 1, "name": "rov", "supercategory": "rov"},
    2: {"id": 2, "name": "non_trash", "supercategory": "non_trash"},
    3: {"id": 3, "name": "trash", "supercategory": "trash"},
}

# TC20 ID → ternary ID
TERNARY_MAPPING = {
    tc20_id: (1 if coarse_id == 1 else 2 if coarse_id == 2 else 3)
    for tc20_id, coarse_id in COARSE_MAPPING.items()
}

# ═════════════════════════════════════════════════════════════════════
# Binary scheme (2 categories: non_trash / trash)
# ═════════════════════════════════════════════════════════════════════

BINARY_CATEGORIES = {
    1: {"id": 1, "name": "non_trash", "supercategory": "non_trash"},
    2: {"id": 2, "name": "trash", "supercategory": "trash"},
}

# TC20 ID → binary ID (rov and organic both map to non_trash)
BINARY_MAPPING = {
    tc20_id: (1 if coarse_id in {1, 2} else 2)
    for tc20_id, coarse_id in COARSE_MAPPING.items()
}

# ═════════════════════════════════════════════════════════════════════
# Composite mappings & registry
# ═════════════════════════════════════════════════════════════════════


def _compose(m1: dict, m2: dict) -> dict:
    """Compose two mappings: m1[x] → m2[m1[x]], dropping unmapped."""
    return {k: m2[v] for k, v in m1.items() if v in m2}


# SC original → TC20
SC_TO_TC20 = _compose(SC_TO_TC, TC_TO_TC20)

# TC original → coarse (direct, includes animal_crab/animal_eel)
TC_TO_COARSE = {
    # rov
    1: 1,  # rov
    # organic
    2: 2,  # plant
    3: 2,  # animal_fish
    4: 2,  # animal_starfish
    5: 2,  # animal_shells
    6: 2,  # animal_crab (not in TC20)
    7: 2,  # animal_eel (not in TC20)
    8: 2,  # animal_etc
    18: 2,  # trash_branch (natural debris)
    # trash_easy
    10: 3,  # trash_pipe (plastic)
    11: 3,  # trash_bottle
    12: 3,  # trash_bag
    13: 3,  # trash_snack_wrapper
    14: 3,  # trash_can
    15: 3,  # trash_cup
    16: 3,  # trash_container
    # trash_entangled
    9: 4,  # trash_clothing
    20: 4,  # trash_tarp
    21: 4,  # trash_rope
    22: 4,  # trash_net
    # trash_heavy
    17: 5,  # trash_unknown_instance
    19: 5,  # trash_wreckage
}

# SC original → coarse (direct, includes all 40 SC categories)
SC_TO_COARSE = {
    # rov
    27: 1,  # rov_cable
    28: 1,  # rov_tortuga
    38: 1,  # rov_vehicle_leg
    39: 1,  # rov_bluerov
    # organic
    6: 2,  # plant
    8: 2,  # animal_etc
    9: 2,  # animal_sponge (not in TC20)
    15: 2,  # animal_shells
    17: 2,  # animal_urchin (not in TC20)
    29: 2,  # branch_wood
    36: 2,  # animal_fish
    40: 2,  # animal_starfish
    # trash_easy
    1: 3,  # can_metal
    3: 3,  # container_plastic
    4: 3,  # bottle_plastic
    7: 3,  # container_middle_size_metal
    10: 3,  # bottle_glass
    13: 3,  # pipe_plastic
    18: 3,  # cup_plastic
    20: 3,  # bag_plastic
    21: 3,  # sanitaries_plastic (not in TC20)
    23: 3,  # cup_ceramic
    24: 3,  # boot_rubber (not in TC20)
    25: 3,  # tire_rubber (not in TC20)
    26: 3,  # jar_glass (not in TC20)
    31: 3,  # snack_wrapper_plastic
    32: 3,  # lid_plastic (not in TC20)
    33: 3,  # cardboard_paper (not in TC20)
    37: 3,  # snack_wrapper_paper
    # trash_entangled
    2: 4,  # tarp_plastic
    14: 4,  # net_plastic
    16: 4,  # rope_fiber
    22: 4,  # clothing_fiber
    34: 4,  # rope_plastic
    35: 4,  # cable_metal (not in TC20)
    # trash_heavy
    5: 5,  # tube_cement (not in TC20)
    11: 5,  # wreckage_metal
    12: 5,  # unknown_instance
    19: 5,  # brick_clay (not in TC20)
    30: 5,  # furniture_wood (not in TC20)
}

# TC original → ternary (direct)
TC_TO_TERNARY = {
    k: (1 if v == 1 else 2 if v == 2 else 3) for k, v in TC_TO_COARSE.items()
}

# SC original → ternary (direct)
SC_TO_TERNARY = {
    k: (1 if v == 1 else 2 if v == 2 else 3) for k, v in SC_TO_COARSE.items()
}

# TC original → binary (direct; rov and organic both map to non_trash)
TC_TO_BINARY = {k: (1 if v in {1, 2} else 2) for k, v in TC_TO_COARSE.items()}

# SC original → binary (direct)
SC_TO_BINARY = {k: (1 if v in {1, 2} else 2) for k, v in SC_TO_COARSE.items()}


# ═════════════════════════════════════════════════════════════════════
# Category space hierarchy (finest → coarsest)
# ═════════════════════════════════════════════════════════════════════

# Lower number = finer granularity. Used to determine the coarsest
# common evaluation space between two category spaces.
SPACE_HIERARCHY = {
    "seaclear": 0,
    "trashcan": 0,
    "tc20": 1,
    "coarse": 2,
    "ternary": 3,
    "binary": 4,
}


def coarsest_common_space(space_a: str, space_b: str) -> str:
    """Return the coarsest (least granular) of two category spaces.

    Both spaces are mapped to the hierarchy, and the one with the higher
    (coarser) rank is returned. If they're at the same level, the canonical
    name is returned (tc20 for the original spaces, or the shared name).
    """
    rank_a = SPACE_HIERARCHY.get(space_a)
    rank_b = SPACE_HIERARCHY.get(space_b)
    if rank_a is None:
        raise ValueError(f"Unknown category space '{space_a}'")
    if rank_b is None:
        raise ValueError(f"Unknown category space '{space_b}'")

    if rank_a >= rank_b:
        # a is coarser or equal
        return space_a if rank_a > rank_b else _canonical_space(space_a, space_b)
    else:
        return space_b


def _canonical_space(a: str, b: str) -> str:
    """When two spaces are at the same hierarchy level, pick the canonical one."""
    if a == b:
        return a  # same space — evaluate natively
    # seaclear and trashcan are both level 0 → tc20 is the common space
    if {a, b} <= {"seaclear", "trashcan", "tc20"}:
        return "tc20"
    return a


def spaces_coarser_than(space: str) -> list[str]:
    """Return all category spaces strictly coarser than the given one."""
    rank = SPACE_HIERARCHY.get(space)
    if rank is None:
        raise ValueError(f"Unknown category space '{space}'")
    return sorted(
        [s for s, r in SPACE_HIERARCHY.items() if r > rank],
        key=lambda s: SPACE_HIERARCHY[s],
    )


# Identity mappings
COARSE_IDENTITY = {i: i for i in COARSE_CATEGORIES}
TERNARY_IDENTITY = {i: i for i in TERNARY_CATEGORIES}
BINARY_IDENTITY = {i: i for i in BINARY_CATEGORIES}


# Registry: (target, source) → (mapping, categories)
# source="tc20" means the mapping expects TC20 IDs as input
# source="trashcan" means it expects original TrashCan IDs
# source="seaclear" means it expects original SeaClear IDs
GROUPING_SCHEMES = {
    # From TC20
    ("tc20", "tc20"): {"mapping": TC20_MAPPING, "categories": TC20_CATEGORIES},
    ("coarse", "tc20"): {"mapping": COARSE_MAPPING, "categories": COARSE_CATEGORIES},
    ("ternary", "tc20"): {"mapping": TERNARY_MAPPING, "categories": TERNARY_CATEGORIES},
    ("binary", "tc20"): {"mapping": BINARY_MAPPING, "categories": BINARY_CATEGORIES},
    # From TrashCan original
    ("tc20", "trashcan"): {"mapping": TC_TO_TC20, "categories": TC20_CATEGORIES},
    ("coarse", "trashcan"): {"mapping": TC_TO_COARSE, "categories": COARSE_CATEGORIES},
    ("ternary", "trashcan"): {
        "mapping": TC_TO_TERNARY,
        "categories": TERNARY_CATEGORIES,
    },
    ("binary", "trashcan"): {"mapping": TC_TO_BINARY, "categories": BINARY_CATEGORIES},
    # From SeaClear original
    ("tc20", "seaclear"): {"mapping": SC_TO_TC20, "categories": TC20_CATEGORIES},
    ("coarse", "seaclear"): {"mapping": SC_TO_COARSE, "categories": COARSE_CATEGORIES},
    ("ternary", "seaclear"): {
        "mapping": SC_TO_TERNARY,
        "categories": TERNARY_CATEGORIES,
    },
    ("binary", "seaclear"): {"mapping": SC_TO_BINARY, "categories": BINARY_CATEGORIES},
    # From coarse
    ("coarse", "coarse"): {"mapping": COARSE_IDENTITY, "categories": COARSE_CATEGORIES},
    ("ternary", "coarse"): {
        "mapping": {
            1: 1,
            2: 2,
            3: 3,
            4: 3,
            5: 3,
        },  # rov→rov, organic→non_trash, trash*→trash
        "categories": TERNARY_CATEGORIES,
    },
    ("binary", "coarse"): {
        "mapping": {i: (1 if i in {1, 2} else 2) for i in COARSE_CATEGORIES},
        "categories": BINARY_CATEGORIES,
    },
    # From ternary
    ("ternary", "ternary"): {
        "mapping": TERNARY_IDENTITY,
        "categories": TERNARY_CATEGORIES,
    },
    ("binary", "ternary"): {
        "mapping": {
            1: 1,
            2: 1,
            3: 2,
        },  # rov→non_trash, non_trash→non_trash, trash→trash
        "categories": BINARY_CATEGORIES,
    },
    # From binary (identity only)
    ("binary", "binary"): {"mapping": BINARY_IDENTITY, "categories": BINARY_CATEGORIES},
}


def get_scheme(target: str, source: str = "tc20") -> dict:
    """Get a category mapping scheme.

    Args:
        target: target category space ("tc20", "coarse", "binary")
        source: source category space ("tc20", "trashcan", "seaclear")

    Returns:
        dict with keys:
            'mapping': source ID → target ID
            'categories': target ID → {"id": ..., "name": ..., "supercategory": ...}
    """
    key = (target, source)
    if key not in GROUPING_SCHEMES:
        available_targets = sorted(set(k[0] for k in GROUPING_SCHEMES))
        available_sources = sorted(set(k[1] for k in GROUPING_SCHEMES))
        raise ValueError(
            f"Unknown scheme (target='{target}', source='{source}'). "
            f"Available targets: {available_targets}, sources: {available_sources}"
        )
    return GROUPING_SCHEMES[key]


def detect_source_space(coco_obj) -> str:
    """Auto-detect the category space of a COCO annotation object.

    Inspects category IDs and names to determine if annotations are in:
      - "tc20": 20 categories, contiguous IDs 1-20, TC20 names
      - "trashcan": original TrashCan (22 cats, includes animal_crab/animal_eel)
      - "seaclear": original SeaClear (IDs go up to 40, SC-style names)
      - "coarse": 5 categories (rov, organic, trash_easy, trash_entangled, trash_heavy)
      - "ternary": 3 categories (rov, non_trash, trash)
      - "binary": 2 categories (non_trash, trash)

    Returns:
        One of "tc20", "trashcan", "seaclear", "coarse", "ternary", "binary"

    Raises:
        ValueError if the category space cannot be determined.
    """
    cats = coco_obj.loadCats(coco_obj.getCatIds())
    cat_ids = set(c["id"] for c in cats)
    cat_names = set(c["name"] for c in cats)
    n_cats = len(cats)

    # Binary: exactly 2 categories named non_trash and trash
    if n_cats == 2 and cat_names == {"non_trash", "trash"}:
        return "binary"

    # Ternary: 3 categories (rov, non_trash, trash)
    ternary_names = {c["name"] for c in TERNARY_CATEGORIES.values()}
    if n_cats == len(TERNARY_CATEGORIES) and cat_names == ternary_names:
        return "ternary"

    # Coarse: 5 categories with coarse names
    coarse_names = {c["name"] for c in COARSE_CATEGORIES.values()}
    if n_cats == len(COARSE_CATEGORIES) and cat_names == coarse_names:
        return "coarse"

    # TC20: exactly 20 categories, IDs 1-20
    if n_cats == 20 and cat_ids == set(range(1, 21)):
        return "tc20"

    # TrashCan original: has IDs 6 and 7 (animal_crab, animal_eel)
    if {6, 7}.issubset(cat_ids) and any(
        "animal_crab" in c["name"] or "animal_eel" in c["name"] for c in cats
    ):
        return "trashcan"

    # SeaClear: IDs go higher than 22, SC-style names
    if max(cat_ids) > 22:
        return "seaclear"

    raise ValueError(
        f"Cannot auto-detect category space: {n_cats} categories, "
        f"IDs {sorted(cat_ids)[:5]}... Names: {sorted(cat_names)[:5]}..."
    )


def available_targets() -> list[str]:
    """List available target category spaces."""
    return sorted(set(k[0] for k in GROUPING_SCHEMES))


def available_sources() -> list[str]:
    """List available source category spaces."""
    return sorted(set(k[1] for k in GROUPING_SCHEMES))


# ═════════════════════════════════════════════════════════════════════
# Remapping utilities (for eval-time and annotation conversion)
# ═════════════════════════════════════════════════════════════════════


def remap_results(results: list[dict], mapping: dict) -> list[dict]:
    """Remap prediction category IDs. Predictions with unmapped IDs are dropped."""
    mapping_keys = set(mapping.keys())
    remapped = []
    for r in results:
        cat_id = int(r["category_id"])
        if cat_id in mapping_keys:
            r = r.copy()
            r["category_id"] = mapping[cat_id]
            remapped.append(r)
    return remapped


def remap_coco_gt(coco_gt, mapping: dict, categories: dict):
    """Create a remapped copy of a COCO ground-truth object.

    Remaps annotation category IDs and replaces the category list.
    Returns a new COCO object (the original is not modified).
    """
    import copy
    import json
    import os
    import tempfile
    from pycocotools.coco import COCO

    dataset = copy.deepcopy(coco_gt.dataset)

    dataset["categories"] = [
        {
            "id": cat["id"],
            "name": cat["name"],
            "supercategory": cat.get("supercategory", cat["name"]),
        }
        for cat in categories.values()
    ]

    mapping_keys = set(mapping.keys())
    remapped_anns = []
    for ann in dataset["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in mapping_keys:
            ann = ann.copy()
            ann["category_id"] = mapping[cat_id]
            remapped_anns.append(ann)
    dataset["annotations"] = remapped_anns

    # pycocotools stores RLE counts as bytes — decode for JSON serialization
    for ann in dataset["annotations"]:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and isinstance(seg.get("counts"), bytes):
            seg["counts"] = seg["counts"].decode("utf-8")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset, f)
        tmp_path = f.name

    remapped_coco = COCO(tmp_path)
    os.unlink(tmp_path)
    return remapped_coco


def remap_annotation_file(
    input_path: str, output_path: str, mapping: dict, categories: dict
) -> dict:
    """Remap a COCO annotation JSON file to a new category space.

    Args:
        input_path: path to input annotation JSON
        output_path: path to write remapped annotation JSON
        mapping: source category ID → target category ID
        categories: target ID → {"id": ..., "name": ..., "supercategory": ...}

    Returns:
        dict with conversion stats (orig_anns, kept_anns, dropped, per_category_counts)
    """
    import json

    with open(input_path) as f:
        data = json.load(f)

    mapping_keys = set(mapping.keys())
    orig_ann_count = len(data["annotations"])

    new_anns = []
    dropped = 0
    cat_counts = {}

    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in mapping_keys:
            dropped += 1
            continue
        ann = ann.copy()
        ann["category_id"] = mapping[cat_id]
        new_anns.append(ann)
        cat_counts[ann["category_id"]] = cat_counts.get(ann["category_id"], 0) + 1

    data["annotations"] = new_anns
    data["categories"] = [
        {
            "id": cat["id"],
            "name": cat["name"],
            "supercategory": cat.get("supercategory", cat["name"]),
        }
        for cat in sorted(categories.values(), key=lambda c: c["id"])
    ]

    with open(output_path, "w") as f:
        json.dump(data, f)

    cat_names = {cat["id"]: cat["name"] for cat in categories.values()}
    return {
        "orig_anns": orig_ann_count,
        "kept_anns": len(new_anns),
        "dropped": dropped,
        "per_category_counts": {
            cat_names.get(k, f"id_{k}"): v for k, v in sorted(cat_counts.items())
        },
    }
