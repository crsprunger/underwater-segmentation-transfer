"""Core training library for cross-dataset underwater instance segmentation.

Modules:
    training_config  - Dataclass holding all hyperparameters and paths
    dataset          - COCO-format dataset, transforms, and batch collation
    augmentation     - Fourier Style Randomization and copy-paste augmentation
    model            - Mask R-CNN construction with custom heads and regularization
    evaluation       - COCO evaluation loop and test-time augmentation (TTA)
    checkpointing    - Atomic save/load, signal handling, history export
    train            - Training loop entry point
    category_groups  - Category space hierarchy and remapping logic
"""