"""Centralized configuration for Mask R-CNN training.

All hyperparameters, paths, and feature flags live in a single dataclass
so that every training run is fully reproducible from its serialized config.
The config is saved as YAML inside each checkpoint directory.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Complete specification of a training run.

    Organized into 10 sections covering system paths, optimization,
    architecture, augmentations, regularization, TTA, and logging.
    Defaults match the baseline TrashCan experiment; override via YAML.
    """

    # ==========================================
    #  --------- 1. SYSTEM & PATHS -----------
    # ==========================================
    train_ann_path: str = ""  # path to COCO-format train annotation JSON
    val_ann_path: str = ""  # path to COCO-format val annotation JSON
    train_img_dir: str = ""  # directory containing training images
    val_img_dir: str = ""  # directory containing validation images
    checkpoint_dir: str = ""
    num_workers: int = 8
    use_torch_compile: bool = False

    # ==========================================
    #  ------ 2. TRAINING HYPERPARAMETERS ------
    # ==========================================
    num_epochs: int = 100
    batch_size: int = 8
    accumulation_steps: int = 1  # Effective batch = batch_size * accumulation_steps
    use_amp: bool = True  # Mixed precision
    clip_grad_norm: float = 5.0  # 0.0 disables clipping

    # ==========================================
    #  -------- 3. OPTIMIZER & SCHEDULER -------
    # ==========================================
    optimizer: str = "sgd"  # "sgd" or "adamw"
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    adam_betas: tuple = (0.9, 0.999)

    lr_scheduler_type: str = "cosine"  # "multistep" or "cosine"
    lr_milestones: tuple = (66, 88)
    lr_gamma: float = 0.1
    warmup_epochs: int = 1
    warmup_factor: float = 0.001
    cosine_eta_min: float = 0.0

    # ==========================================
    #  --------- 4. MODEL ARCHITECTURE ---------
    # ==========================================
    # num_classes: number of object classes + 1 background (TrashCan=23
    #   SeaClear=41, TC20=21, coarse=6, etc.).
    num_classes: int = 23
    # pretrained: True uses the pretrained weights from the COCO dataset; False
    #   uses the ImageNet weights for the backbone body only.
    pretrained: bool = True
    # use_v1_arch: True yields an architecture closer to Detectron2's for the
    #   sake of comparison, but the v2 architecture is expected to perform
    #   better in general.
    use_v1_arch: bool = False  # False = v2 (4-conv+1-FC box head)
    # use_group_norm: replace BatchNorm2d with GroupNorm in FPN and ROI heads
    use_group_norm: bool = True
    gn_num_groups: int = 32

    # freeze_backbone_stages: Torchvision's Mask R-CNN ResNet50 + FPN backbone
    #   freezes all backbone stages except the last 3 (layer2, layer3, layer4)
    #   by default, which our setting does not override, so setting a value
    #   less than 2 has the same effect as setting it to 2.
    #   (Note: the TV default here matches Detectron2's default FREEZE_AT=2 →
    #   freeze stem + layer1)
    freeze_backbone_stages: int = 2
    # freeze_backbone_bn: freeze the backbone batch normalization layers; this
    #   shouldn't be set to False for datasets as small as TrashCan or SeaClear.
    freeze_backbone_bn: bool = True
    backbone_lr_factor: float = 1.0  # < 1.0 = lower LR for backbone
    backbone_wd_factor: float = 1.0  # < 1.0 = lower WD for backbone

    # ==========================================
    #  ------ 5. RPN & PROPOSAL TUNING ---------
    # ==========================================
    # Custom anchor generator: we use these settings to override the default
    #   anchor generator, which uses anchor sizes from 32 to 512, because our
    #   datasets have lots of small objects.
    use_custom_anchor_generator: bool = True
    custom_anchor_sizes: tuple = ((16,), (32,), (64,), (128,), (256,))
    custom_aspect_ratios: tuple = ((0.5, 1.0, 2.0),)

    # Custom RPN foreground IoU threshold: lower value to better catch smaller objects.
    use_custom_rpn_fg_iou_threshold: bool = False
    custom_rpn_fg_iou_threshold: float = 0.5  # Lower value to catch smaller objects

    # Custom positive proposal fractions: the default values below are the
    #   Torchvision default values. Increasing the box positive fraction gives
    #   the box heads more positive examples to learn from (helpful if recall
    #   is low), while decreasing it gives more hard negatives (helpful if
    #   precision is low). The RPN fraction works similarly but at the region
    #   proposal stage.
    use_custom_positive_proposal_fractions: bool = False
    custom_rpn_positive_fraction: float = 0.5
    custom_box_positive_fraction: float = 0.25

    # ==========================================
    #  ------ 6. IMAGE SIZING & BASE DATA ------
    # ==========================================
    # Multi-scale training: short side sampled from min_size, long side capped at max_size
    min_size: tuple = (256, 272, 288, 304, 320)
    max_size: int = 512

    # ==========================================
    #  --------- 7. DATA AUGMENTATIONS ---------
    # ==========================================
    use_photometric_distort: bool = False
    use_random_erasing: bool = False

    use_random_grayscale: bool = False
    random_grayscale_prob: float = 0.2

    use_fourier_style_randomization: bool = False
    fsr_prob: float = 0.3  # probability of applying per image
    fsr_beta: float = 0.01  # low-frequency disk radius (fraction of spectral width)
    fsr_noise_prob: float = (
        0.15  # fraction of FSR applications that use noise instead of style bank
    )
    fsr_style_bank_dir: str = (
        ""  # directory of images to use as style donors (any format)
    )
    fsr_style_bank_spectra: str = ""  # path to precomputed .npz with 'amplitudes' array
    fsr_style_bank_max_images: int = 200  # max images to load from style bank dir

    # Copy-paste augmentation:
    use_copy_paste: bool = False
    copy_paste_prob: float = 0.5
    copy_paste_max_objects: int = 3
    copy_paste_scale_ranges: tuple = ((0.3, 0.8), (0.8, 1.2))
    copy_paste_scale_range_weights: tuple = (0.7, 0.3)
    copy_paste_rotation_range: tuple = (-45, 45)
    copy_paste_min_crop_side_length: int = 24
    copy_paste_min_object_area: int = 512
    copy_paste_color_match: bool = True
    copy_paste_exclude_cats: (
        tuple
    ) = ()  # category IDs to never paste (e.g. (1,) for ROV)

    # Text overlay augmentation:
    text_overlay_min_object_area: int = 32  # min mask area after text mask subtraction

    # ==========================================
    #  ----------- 8. REGULARIZATION -----------
    # ==========================================

    # Note: these regularization techniques are not used in our published experiments thus far.

    # dropout_rate: dropout rate for the box head, injected after the FC+ReLU
    #   and before the predictor.
    dropout_rate: float = 0.0
    # mask_dropout_rate: spatial dropout (Dropout2d) rate for the mask head,
    #   injected after the mask head convs and before the mask predictor.
    mask_dropout_rate: float = 0.0
    label_smoothing: float = 0.0

    # ==========================================
    #  ------- 9. TEST-TIME AUGMENTATION -------
    # ==========================================
    tta_enabled: bool = False
    tta_flip: bool = True
    tta_scales: tuple = (1.0,)
    tta_rotation: tuple = (0.0,)
    tta_grayscale: bool = False
    tta_clahe: bool = False
    tta_nms_threshold: float = 0.5
    tta_soft_nms: bool = False
    tta_mask_avg: bool = True

    # ==========================================
    #  -------- 10. LOGGING & DEBUGGING --------
    # ==========================================
    print_freq: int = 50
    verbose: bool = True
    log_grad_norm: bool = True
