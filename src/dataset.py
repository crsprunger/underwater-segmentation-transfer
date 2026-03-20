"""Dataset and data loading for Mask R-CNN training.

Contains the COCO-format dataset class, batch collation with copy-paste
augmentation, and data transformation pipeline construction.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.io import decode_image
from pycocotools.coco import COCO
import torchvision.transforms.v2 as T

from src.training_config import TrainingConfig
from src.augmentation import FourierStyleRandomization, _in_batch_copy_paste


class CocoDataset(torch.utils.data.Dataset):
    """COU dataset in COCO format, producing targets for torchvision Mask R-CNN.

    Category IDs are remapped: COU 0-23 → torchvision 1-24 (0 = background).
    """

    def __init__(
        self,
        coco_ann_path: str,
        img_dir: str,
        transforms=None,
        cfg: TrainingConfig = None,
    ):
        self.coco = COCO(coco_ann_path)
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.img_ids = sorted(self.coco.getImgIds())
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]
        if not img_path.exists():
            img_path = self.img_dir / Path(img_info["file_name"]).name
        img = decode_image(str(img_path), mode=torchvision.io.ImageReadMode.RGB)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        ann_ids = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            mask_arr = self.coco.annToMask(ann)
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(mask_arr)
            areas.append(ann.get("area", float(mask_arr.sum())))
            iscrowd.append(ann.get("iscrowd", 0))
            ann_ids.append(ann["id"])

        # If text overlays exist on disk, load the overlaid image instead
        # and subtract the text mask from annotation masks
        text_mask_path = self.img_dir / "text_masks" / (img_path.stem + ".png")
        if text_mask_path.exists():
            text_mask = cv2.imread(str(text_mask_path), cv2.IMREAD_GRAYSCALE) > 0
            min_area = self.cfg.text_overlay_min_object_area if self.cfg else 32
            filtered_boxes = []
            filtered_labels = []
            filtered_masks = []
            filtered_areas = []
            filtered_iscrowd = []
            filtered_ann_ids = []
            for j in range(len(masks)):
                m = masks[j].copy()
                m[text_mask] = 0
                if m.sum() >= min_area:
                    # Recompute bounding box from remaining mask
                    mys, mxs = np.where(m > 0)
                    filtered_boxes.append(
                        [
                            float(mxs.min()),
                            float(mys.min()),
                            float(mxs.max() + 1),
                            float(mys.max() + 1),
                        ]
                    )
                    filtered_labels.append(labels[j])
                    filtered_masks.append(m)
                    filtered_areas.append(float(m.sum()))
                    filtered_iscrowd.append(iscrowd[j])
                    filtered_ann_ids.append(ann_ids[j])
            boxes = filtered_boxes
            labels = filtered_labels
            masks = filtered_masks
            areas = filtered_areas
            iscrowd = filtered_iscrowd
            ann_ids = filtered_ann_ids

        img = tv_tensors.Image(img)
        canvas_size = F.get_size(img)

        h, w = img_info["height"], img_info["width"]
        if len(boxes) == 0:
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format="XYXY",
                    canvas_size=canvas_size,
                ),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": tv_tensors.Mask(torch.zeros((0, h, w), dtype=torch.uint8)),
                "image_id": img_id,
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_height": h,
                "image_width": w,
                "ann_ids": ann_ids,
            }
        else:
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    format="XYXY",
                    canvas_size=canvas_size,
                ),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": tv_tensors.Mask(
                    torch.stack(
                        [
                            torch.from_numpy(m) if isinstance(m, np.ndarray) else m
                            for m in masks
                        ]
                    ),
                    dtype=torch.uint8,
                ),
                "image_id": img_id,
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                "image_height": h,
                "image_width": w,
                "ann_ids": ann_ids,
            }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class BatchCollator:
    """Custom collate function that optionally applies in-batch copy-paste.

    Standard PyTorch collation would stack images into a single tensor,
    but Mask R-CNN expects a list of individually-sized tensors (the model's
    GeneralizedRCNNTransform handles padding internally).  This collator
    unzips the batch and, when copy-paste augmentation is enabled, pastes
    object instances across images within the same batch to increase the
    effective number of annotated objects per image.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        # 1. Standard unzip
        images, targets = tuple(zip(*batch))

        # 2. Apply in-batch copy paste if training and enabled
        if self.cfg.use_copy_paste:
            images, targets = _in_batch_copy_paste(images, targets, self.cfg)

        return images, targets


def get_transforms(train: bool, cfg: TrainingConfig = None):
    """Build the torchvision v2 transform pipeline for training or inference.

    Training pipeline (order matters for correctness):
      1. Random horizontal flip
      2. Optional Fourier Style Randomization (domain-adaptive color transfer)
      3. Optional photometric distortion (brightness, contrast, hue, saturation)
      4. Optional random grayscale conversion
      5. Float conversion and [0,1] scaling
      6. Optional random erasing
      7. Bounding box sanitization (remove degenerate boxes from augmentation)

    Inference pipeline: float conversion only.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        if cfg is not None and cfg.use_fourier_style_randomization:
            transforms.append(FourierStyleRandomization(cfg))
        if cfg is not None and cfg.use_photometric_distort:
            transforms.append(T.RandomPhotometricDistort())
        if cfg is not None and cfg.use_random_grayscale:
            transforms.append(T.RandomGrayscale(p=cfg.random_grayscale_prob))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train and cfg is not None and cfg.use_random_erasing:
        transforms.append(
            T.RandomErasing(p=0.5, scale=(0.01, 0.05))
        )  # Max 5% of image area
    if train:
        transforms.append(T.SanitizeBoundingBoxes())
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
