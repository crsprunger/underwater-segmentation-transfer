"""Mask R-CNN model construction and configuration.

Contains model building with custom heads, GroupNorm replacement,
backbone freezing, dropout injection, label smoothing, and
parameter group construction for differential learning rates.
"""

import torch
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models.detection.roi_heads as roi_heads

from src.training_config import TrainingConfig


def _replace_bn_with_gn(module: torch.nn.Module, num_groups: int = 32) -> None:
    """Recursively replace BatchNorm2d with GroupNorm in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            num_channels = child.num_features
            # num_groups must evenly divide num_channels; reduce if needed
            groups = num_groups
            while num_channels % groups != 0:
                groups -= 1
            setattr(module, name, torch.nn.GroupNorm(groups, num_channels))
        else:
            _replace_bn_with_gn(child, num_groups)


def build_model(cfg: TrainingConfig) -> torch.nn.Module:
    """Build Mask R-CNN with COCO pretrained backbone, replace heads for use with target dataset.

    use_v1_arch=False (default) uses maskrcnn_resnet50_fpn_v2 (4-conv+1-FC box head).
    use_v1_arch=True uses maskrcnn_resnet50_fpn (2-FC box head, matches Detectron2), which defaults to aligned=False in ROI Align; we patch both poolers to aligned=True.
    """

    kwargs = {}
    kwargs["min_size"] = cfg.min_size
    kwargs["max_size"] = cfg.max_size
    if cfg.use_custom_positive_proposal_fractions:
        kwargs["rpn_positive_fraction"] = cfg.custom_rpn_positive_fraction
        kwargs["box_positive_fraction"] = cfg.custom_box_positive_fraction
    if cfg.use_custom_rpn_fg_iou_threshold:
        kwargs["rpn_fg_iou_threshold"] = cfg.custom_rpn_fg_iou_threshold

    if cfg.use_v1_arch:
        if cfg.pretrained:
            model = maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                **kwargs,
            )
        else:
            model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
                num_classes=cfg.num_classes,
                **kwargs,
            )
        # Patch ROI Align to use aligned=True (matches Detectron2; fixes half-pixel offset)
        model.roi_heads.box_roi_pool.aligned = True
        model.roi_heads.mask_roi_pool.aligned = True
    else:
        if cfg.pretrained:
            model = maskrcnn_resnet50_fpn_v2(
                weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                **kwargs,
            )
        else:
            model = maskrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
                num_classes=cfg.num_classes,
                **kwargs,
            )

    if cfg.use_custom_anchor_generator:
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        rpn_anchor_generator = AnchorGenerator(
            sizes=cfg.custom_anchor_sizes,
            aspect_ratios=cfg.custom_aspect_ratios * len(cfg.custom_anchor_sizes),
        )
        model.rpn.anchor_generator = rpn_anchor_generator

    if cfg.pretrained:
        # Replace box predictor (always needed when pretrained — heads are COCO 91-class)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)
        # Replace mask predictor
        in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels, dim_reduced, cfg.num_classes
        )

    if cfg.use_group_norm:
        # Replace BN with GN in FPN and ROI heads (not backbone — preserve ImageNet BN stats)
        _replace_bn_with_gn(model.backbone.fpn, cfg.gn_num_groups)
        _replace_bn_with_gn(model.roi_heads, cfg.gn_num_groups)

    if cfg.freeze_backbone_bn:
        # Freeze backbone BN: fix running stats at ImageNet values, don't train scale/bias
        for module in model.backbone.body.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)

    # Completely freeze early backbone stages (see training_config.py for more details)
    stages_to_freeze = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    for stage in stages_to_freeze[: cfg.freeze_backbone_stages + 1]:
        stage_module = getattr(model.backbone.body, stage, None)
        if stage_module is not None:
            for param in stage_module.parameters():
                param.requires_grad_(False)

    # Inject dropout after box head FC+ReLU, before the predictor
    if cfg.dropout_rate > 0:
        model.roi_heads.box_head.append(torch.nn.Dropout(p=cfg.dropout_rate))

    # Inject spatial dropout (Dropout2d) after mask head convs, before mask predictor
    if cfg.mask_dropout_rate > 0:
        model.roi_heads.mask_head.append(torch.nn.Dropout2d(p=cfg.mask_dropout_rate))

    if cfg.label_smoothing > 0:
        # Safety check: Prevent double-patching if build_model is called multiple times
        if not hasattr(roi_heads, "_original_fastrcnn_loss"):
            roi_heads._original_fastrcnn_loss = roi_heads.fastrcnn_loss

        def smooth_fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets
        ):
            # Calculate the standard box loss using the original function
            _, box_loss = roi_heads._original_fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )

            # Recompute the classification loss with configured label smoothing applied
            labels_cat = torch.cat(labels, dim=0)
            cls_loss = torch.nn.functional.cross_entropy(
                class_logits, labels_cat, label_smoothing=cfg.label_smoothing
            )
            return cls_loss, box_loss

        # Apply the patch
        roi_heads.fastrcnn_loss = smooth_fastrcnn_loss

    return model


def get_param_groups(model, cfg):
    """Construct optimizer parameter groups with differential learning rates.

    Separates parameters into up to four groups:
      - Non-backbone params with weight decay
      - Non-backbone params without weight decay (biases, norms)
      - Backbone body params with scaled weight decay and LR
      - Backbone body biases/norms with scaled LR, no weight decay (this will be empty if freeze_backbone_bn is True, which should generally be the case)

    This provides the option to, for example, follow the Detectron2 convention
    of lower LR for pretrained backbone layers to preserve learned low-level
    features while allowing the task-specific heads to adapt freely.
    """
    param_groups = []
    norm_types = (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)
    norm_param_ids = set()
    for module in model.modules():
        if isinstance(module, norm_types):
            for param in module.parameters(recurse=False):
                norm_param_ids.add(id(param))
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or id(param) in norm_param_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    use_backbone_factors = (
        cfg.backbone_lr_factor != 1.0 or cfg.backbone_wd_factor != 1.0
    )
    if use_backbone_factors:
        backbone_body_ids = {
            id(p) for n, p in model.named_parameters() if n.startswith("backbone.body")
        }
        bb_decay = [p for p in decay_params if id(p) in backbone_body_ids]
        bb_no_decay = [p for p in no_decay_params if id(p) in backbone_body_ids]
        other_decay = [p for p in decay_params if id(p) not in backbone_body_ids]
        other_no_decay = [p for p in no_decay_params if id(p) not in backbone_body_ids]
        bb_lr = cfg.lr * cfg.backbone_lr_factor
        bb_wd = cfg.weight_decay * cfg.backbone_wd_factor
        param_groups = [
            {"params": other_decay, "weight_decay": cfg.weight_decay},
            {"params": other_no_decay, "weight_decay": 0.0},
            {"params": bb_decay, "weight_decay": bb_wd, "lr": bb_lr},
            {"params": bb_no_decay, "weight_decay": 0.0, "lr": bb_lr},
        ]
        print(
            f"Backbone body: lr={bb_lr:.2e}, wd={bb_wd:.2e} "
            f"(factors: lr={cfg.backbone_lr_factor}, wd={cfg.backbone_wd_factor})"
        )
    else:
        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    return param_groups
