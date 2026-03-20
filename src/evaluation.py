"""Evaluation and test-time augmentation for Mask R-CNN.

Contains COCO-format evaluation using pycocotools and a TTA wrapper
that generates augmented copies, runs inference, de-augments predictions,
and merges results via NMS.
"""

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
from pycocotools import mask as mask_util
from pycocotools.cocoeval import COCOeval

from src.training_config import TrainingConfig


@torch.no_grad()
def evaluate(model, data_loader) -> dict:
    """Evaluate on val set using COCO API. Returns dict of mAP metrics."""
    model.eval()
    device = next(model.parameters()).device

    coco_gt = data_loader.dataset.coco
    coco_results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"]
            w, h = target["image_width"], target["image_height"]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"]

            if masks.shape[-2] != h or masks.shape[-1] != w:
                scale = float(min(w, h)) / float(min(masks.shape[-2], masks.shape[-1]))
                masks = (
                    (
                        torch.nn.functional.interpolate(
                            masks, size=(h, w), mode="bilinear", align_corners=False
                        )[:, 0]
                        > 0.5
                    )
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
            else:
                scale = 1.0
                masks = (masks[:, 0] > 0.5).to(torch.uint8).cpu().numpy()
            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                binary_mask = masks[i]
                rle = mask_util.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(labels[i]),
                        "bbox": [
                            float(x1) * scale,
                            float(y1) * scale,
                            float(x2 - x1) * scale,
                            float(y2 - y1) * scale,
                        ],
                        "score": float(scores[i]),
                        "segmentation": rle,
                    }
                )

    if not coco_results:
        print("  WARNING: No predictions generated.")
        return {
            "mask_mAP": 0.0,
            "mask_mAP_50": 0.0,
            "mask_mAP_75": 0.0,
            "box_mAP": 0.0,
            "box_mAP_50": 0.0,
            "box_mAP_75": 0.0,
        }

    coco_dt = coco_gt.loadRes(coco_results)
    metrics = {}
    for iou_type in ["bbox", "segm"]:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        prefix = "box" if iou_type == "bbox" else "mask"
        metrics[f"{prefix}_mAP"] = coco_eval.stats[0]
        metrics[f"{prefix}_mAP_50"] = coco_eval.stats[1]
        metrics[f"{prefix}_mAP_75"] = coco_eval.stats[2]
        metrics[f"{prefix}_mAP_small"] = coco_eval.stats[3]

    model.train()
    return metrics


class TTAWrapper(torch.nn.Module):
    """Wraps a Mask R-CNN model to perform test-time augmentation.

    Generates augmented copies of each image, runs inference on each,
    de-augments predictions, and merges via NMS.
    """

    def __init__(self, model, cfg: TrainingConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def _soft_nms(
        self, boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001
    ):
        """Soft-NMS: decay scores of overlapping boxes instead of removing them."""
        from torchvision.ops import box_iou

        order = scores.argsort(descending=True)
        boxes = boxes[order]
        scores = scores[order].clone()
        keep = []
        while len(scores) > 0:
            keep.append(0)
            if len(scores) == 1:
                break
            ious = box_iou(boxes[:1], boxes[1:])[0]
            # Gaussian decay
            scores[1:] *= torch.exp(-(ious**2) / sigma)
            # Remove low-score boxes
            mask = scores[1:] > score_threshold
            boxes = torch.cat([boxes[1:][mask]])
            scores = torch.cat([scores[1:][mask]])
            if len(scores) == 0:
                break
        return torch.tensor(keep, dtype=torch.long)

    def _merge_predictions(self, all_preds, img_h, img_w):
        """Merge predictions from multiple augmented views via NMS."""
        from torchvision.ops import nms

        if not all_preds:
            return {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
                "masks": torch.zeros(0, 1, img_h, img_w),
            }

        boxes = torch.cat([p["boxes"] for p in all_preds])
        scores = torch.cat([p["scores"] for p in all_preds])
        labels = torch.cat([p["labels"] for p in all_preds])
        masks = torch.cat([p["masks"] for p in all_preds])

        if len(boxes) == 0:
            return {"boxes": boxes, "scores": scores, "labels": labels, "masks": masks}

        # Per-class NMS (offset trick: shift boxes by class to avoid cross-class suppression)
        max_coord = boxes.max() + 1
        class_offsets = labels.float() * max_coord
        shifted_boxes = boxes + class_offsets[:, None]

        if self.cfg.tta_soft_nms:
            keep = self._soft_nms(shifted_boxes, scores, self.cfg.tta_nms_threshold)
        else:
            keep = nms(shifted_boxes, scores, self.cfg.tta_nms_threshold)

        result = {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "masks": masks[keep],
        }

        # Optional: average masks for highly overlapping predictions of same class
        if self.cfg.tta_mask_avg and len(keep) > 1:
            result["masks"] = self._average_overlapping_masks(
                result["boxes"],
                result["labels"],
                result["masks"],
                all_preds,
                boxes,
                labels,
                masks,
            )

        return result

    def _average_overlapping_masks(
        self,
        kept_boxes,
        kept_labels,
        kept_masks,
        all_preds,
        all_boxes,
        all_labels,
        all_masks,
    ):
        """For each kept detection, average its mask with highly overlapping same-class masks."""
        from torchvision.ops import box_iou

        iou_thresh = 0.5
        ious = box_iou(kept_boxes, all_boxes)
        averaged = kept_masks.clone().float()
        for i in range(len(kept_boxes)):
            same_class = all_labels == kept_labels[i]
            high_iou = ious[i] > iou_thresh
            match = same_class & high_iou
            if match.sum() > 1:
                averaged[i] = all_masks[match].float().mean(dim=0)
        return averaged

    def _run_one(self, model, image):
        """Run inference on a single image tensor. Returns prediction dict."""
        outputs = model([image])
        return outputs[0]

    def _flip_pred(self, pred, img_w):
        """De-augment a horizontally flipped prediction."""
        boxes = pred["boxes"].clone()
        boxes[:, [0, 2]] = img_w - boxes[:, [2, 0]]
        masks = pred["masks"].flip(-1)  # flip width dimension
        return {**pred, "boxes": boxes, "masks": masks}

    def _scale_pred(self, pred, scale, orig_h, orig_w):
        """De-augment a scaled prediction back to original size."""
        boxes = pred["boxes"] / scale
        # Resize masks back
        masks = torch.nn.functional.interpolate(
            pred["masks"], size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        return {**pred, "boxes": boxes, "masks": masks}

    def _rotate_pred(self, pred, angle, orig_h, orig_w):
        """De-augment a rotated prediction."""

        # Rotate masks back
        masks = F.rotate(pred["masks"], -angle)
        # Crop/pad masks to original size
        _, _, mh, mw = masks.shape
        if mh != orig_h or mw != orig_w:
            masks = torch.nn.functional.interpolate(
                masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
        # Recompute boxes from de-rotated masks
        boxes = self._masks_to_boxes(masks)
        return {**pred, "boxes": boxes, "masks": masks}

    def _masks_to_boxes(self, masks):
        """Extract tight bounding boxes from mask predictions."""
        n = masks.shape[0]
        boxes = torch.zeros(n, 4, device=masks.device)
        for i in range(n):
            m = masks[i, 0] > 0.5
            if not m.any():
                continue
            ys = torch.where(m.any(dim=1))[0]
            xs = torch.where(m.any(dim=0))[0]
            boxes[i] = torch.tensor([xs[0], ys[0], xs[-1] + 1, ys[-1] + 1])
        return boxes

    @torch.no_grad()
    def forward(self, images):
        """Run TTA inference on a list of images."""
        self.model.eval()
        results = []
        for image in images:
            _, img_h, img_w = image.shape
            all_preds = []

            # Always run original
            pred = self._run_one(self.model, image)
            all_preds.append(pred)

            # Horizontal flip
            if self.cfg.tta_flip:
                flipped = image.flip(-1)
                pred_f = self._run_one(self.model, flipped)
                all_preds.append(self._flip_pred(pred_f, img_w))

            # Multi-scale
            for scale in self.cfg.tta_scales:
                if abs(scale - 1.0) < 1e-3:
                    continue  # skip identity scale
                sh = int(img_h * scale)
                sw = int(img_w * scale)
                scaled = torch.nn.functional.interpolate(
                    image.unsqueeze(0).float(),
                    size=(sh, sw),
                    mode="bilinear",
                    align_corners=False,
                )[0].to(image.dtype)
                pred_s = self._run_one(self.model, scaled)
                all_preds.append(self._scale_pred(pred_s, scale, img_h, img_w))

                # Also flip the scaled version
                if self.cfg.tta_flip:
                    pred_sf = self._run_one(self.model, scaled.flip(-1))
                    pred_sf = self._flip_pred(pred_sf, sw)
                    all_preds.append(self._scale_pred(pred_sf, scale, img_h, img_w))

            # Rotation
            for angle in self.cfg.tta_rotation:
                if abs(angle) < 0.1:
                    continue
                rotated = F.rotate(image, angle, expand=False)
                pred_r = self._run_one(self.model, rotated)
                all_preds.append(self._rotate_pred(pred_r, angle, img_h, img_w))

            # Grayscale
            if self.cfg.tta_grayscale:
                gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
                gray_3ch = gray.unsqueeze(0).expand(3, -1, -1).contiguous()
                pred_g = self._run_one(self.model, gray_3ch)
                all_preds.append(pred_g)
                if self.cfg.tta_flip:
                    pred_gf = self._run_one(self.model, gray_3ch.flip(-1))
                    all_preds.append(self._flip_pred(pred_gf, img_w))

            # CLAHE
            if self.cfg.tta_clahe:
                img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                clahe_tensor = (
                    torch.from_numpy(clahe_img.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .to(image.device)
                )
                pred_c = self._run_one(self.model, clahe_tensor)
                all_preds.append(pred_c)
                if self.cfg.tta_flip:
                    pred_cf = self._run_one(self.model, clahe_tensor.flip(-1))
                    all_preds.append(self._flip_pred(pred_cf, img_w))

            merged = self._merge_predictions(all_preds, img_h, img_w)
            results.append(merged)
        return results
