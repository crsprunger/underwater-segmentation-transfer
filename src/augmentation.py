"""Data augmentation for Mask R-CNN training.

Contains Fourier Style Randomization (FSR) for domain adaptation
and in-batch copy-paste augmentation with occlusion handling.
"""

import glob as glob_module
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from src.training_config import TrainingConfig


class FourierStyleRandomization:
    """Training-time augmentation that replaces low-frequency Fourier amplitude
    with that of a random style bank image, forcing the model to rely on
    phase (structure) rather than amplitude (style/color).

    Supports two style bank sources (set one or both):
      - fsr_style_bank_dir: directory of images; spectra computed lazily and cached
      - fsr_style_bank_spectra: precomputed .npz with 'amplitudes' (N, H, W, 3),
        'disk_amplitudes' (N, disk_pixels, 3), or 'mean_amplitude' (H, W, 3)

    When neither is set (or with probability fsr_noise_prob), replaces the disk
    with scaled Gaussian noise instead.
    """

    def __init__(self, cfg: TrainingConfig):
        self.prob = cfg.fsr_prob
        self.beta = cfg.fsr_beta
        self.noise_prob = cfg.fsr_noise_prob
        self.style_bank_dir = cfg.fsr_style_bank_dir
        self.style_bank_spectra_path = cfg.fsr_style_bank_spectra
        self.max_images = cfg.fsr_style_bank_max_images
        self._style_amplitudes = None  # lazy-loaded list of (H, W, 3) amplitude arrays
        self._disk_donors = (
            None  # lazy-loaded list of (disk_pixels, 3) arrays (disk-only mode)
        )
        self._disk_meta = None  # (H, W, beta) from disk-only .npz
        self._loaded = False

    def _load_style_bank(self):
        """Lazy-load style bank amplitude spectra."""
        if self._loaded:
            return
        self._loaded = True
        self._style_amplitudes = []
        self._disk_donors = []

        # Load from precomputed .npz (supports full, disk-only, and mean formats)
        if self.style_bank_spectra_path:
            data = np.load(self.style_bank_spectra_path)
            if "disk_amplitudes" in data:
                # Disk-only format: (N, disk_pixels, 3)
                disk_beta = float(data["disk_beta"])
                disk_h = int(data["target_h"])
                disk_w = int(data["target_w"])
                if self.beta > disk_beta:
                    print(
                        f"[FSR] Warning: fsr_beta ({self.beta}) > stored disk_beta "
                        f"({disk_beta}); pixels outside stored disk will be zero. "
                        f"Re-run compute_fda_spectrum.py with --disk-beta >= {self.beta}"
                    )
                self._disk_meta = (disk_h, disk_w, disk_beta)
                disk_amps = data["disk_amplitudes"].astype(np.float32)
                for i in range(disk_amps.shape[0]):
                    self._disk_donors.append(disk_amps[i])
                print(
                    f"[FSR] Loaded {len(self._disk_donors)} disk-only spectra "
                    f"(beta={disk_beta}, {disk_amps.shape[1]} pixels/image) "
                    f"from {self.style_bank_spectra_path}"
                )
            elif "amplitudes" in data:
                # Per-image full style bank format: (N, H, W, 3)
                amps = data["amplitudes"].astype(np.float32)
                for i in range(amps.shape[0]):
                    self._style_amplitudes.append(amps[i])
                print(
                    f"[FSR] Loaded {len(self._style_amplitudes)} precomputed spectra "
                    f"from {self.style_bank_spectra_path}"
                )
            elif "mean_amplitude" in data:
                # Mean spectrum format (from FDA): treat as a single donor
                self._style_amplitudes.append(data["mean_amplitude"].astype(np.float32))
                print(
                    f"[FSR] Loaded mean spectrum as single donor "
                    f"from {self.style_bank_spectra_path}"
                )
            else:
                print(
                    f"[FSR] Warning: {self.style_bank_spectra_path} has no recognized "
                    f"key ('disk_amplitudes', 'amplitudes', 'mean_amplitude'); ignoring"
                )

        # Load from image directory
        if self.style_bank_dir:
            style_dir = Path(self.style_bank_dir)
            paths = sorted(
                glob_module.glob(str(style_dir / "*.jpg"))
                + glob_module.glob(str(style_dir / "*.png"))
                + glob_module.glob(str(style_dir / "*.jpeg"))
            )
            if len(paths) > self.max_images:
                rng = np.random.RandomState(42)
                paths = list(rng.choice(paths, self.max_images, replace=False))
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                amp = np.stack(
                    [
                        np.abs(np.fft.fftshift(np.fft.fft2(img[:, :, c])))
                        for c in range(3)
                    ],
                    axis=-1,
                )
                self._style_amplitudes.append(amp)
            print(
                f"[FSR] Computed spectra from {len(paths)} images in {self.style_bank_dir} "
                f"(total style bank: {len(self._style_amplitudes)})"
            )

        if not self._style_amplitudes and not self._disk_donors:
            print("[FSR] No style bank configured; will use noise-only mode")

    @staticmethod
    def _make_low_freq_mask(h: int, w: int, beta: float) -> np.ndarray:
        """Create a circular mask for the low-frequency region of the spectrum."""
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * beta)
        radius = max(radius, 1)
        Y, X = np.ogrid[:h, :w]
        mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius**2
        return mask  # (H, W) bool

    def __call__(self, img, target=None):
        if random.random() > self.prob:
            return (img, target) if target is not None else img

        self._load_style_bank()

        # Convert tensor to numpy float [0, 1] for FFT
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            if img.dtype == torch.uint8:
                img_np = img.permute(1, 2, 0).numpy().astype(np.float32) / 255.0
            else:
                img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.asarray(img, dtype=np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0

        h, w, c = img_np.shape

        # Decide: noise or style bank donor
        has_donors = bool(self._style_amplitudes or self._disk_donors)
        use_noise = not has_donors or random.random() < self.noise_prob

        # Compute FFT of source image
        amp_channels = []
        phase_channels = []
        for ch in range(c):
            f = np.fft.fft2(img_np[:, :, ch])
            f_shift = np.fft.fftshift(f)
            amp_channels.append(np.abs(f_shift))
            phase_channels.append(np.angle(f_shift))

        mask = self._make_low_freq_mask(h, w, self.beta)

        if use_noise:
            # Replace low-freq amplitude with scaled Gaussian noise
            # Scale to match the mean amplitude in the disk region
            for ch in range(c):
                disk_mean = amp_channels[ch][mask].mean()
                disk_std = amp_channels[ch][mask].std()
                noise = np.abs(np.random.normal(disk_mean, disk_std, size=mask.sum()))
                amp_channels[ch][mask] = noise
        elif self._disk_donors:
            # Disk-only mode: donor is (disk_pixels, 3) for a specific (H, W, beta)
            donor_disk = random.choice(self._disk_donors)
            disk_h, disk_w, disk_beta = self._disk_meta
            if disk_h == h and disk_w == w and abs(disk_beta - self.beta) < 1e-9:
                # Exact match — assign directly
                for ch in range(c):
                    amp_channels[ch][mask] = donor_disk[:, ch]
            else:
                # Reconstruct sparse full spectrum, resize, re-extract
                donor_full = np.zeros((disk_h, disk_w, 3), dtype=np.float32)
                stored_mask = self._make_low_freq_mask(disk_h, disk_w, disk_beta)
                for ch in range(3):
                    donor_full[:, :, ch][stored_mask] = donor_disk[:, ch]
                if disk_h != h or disk_w != w:
                    donor_full = cv2.resize(
                        donor_full, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                for ch in range(c):
                    amp_channels[ch][mask] = donor_full[:, :, ch][mask]
        else:
            # Full spectrum mode: donor is (H, W, 3)
            donor_amp = random.choice(self._style_amplitudes)
            if donor_amp.shape[0] != h or donor_amp.shape[1] != w:
                donor_amp = cv2.resize(
                    donor_amp, (w, h), interpolation=cv2.INTER_LINEAR
                )
            for ch in range(c):
                amp_channels[ch][mask] = donor_amp[:, :, ch][mask]

        # Reconstruct image from modified amplitude + original phase
        out_channels = []
        for ch in range(c):
            f_modified = amp_channels[ch] * np.exp(1j * phase_channels[ch])
            f_ishift = np.fft.ifftshift(f_modified)
            img_back = np.fft.ifft2(f_ishift).real
            out_channels.append(img_back)

        img_out = np.stack(out_channels, axis=-1).clip(0, 1).astype(np.float32)

        # Convert back to original format
        if is_tensor:
            if img.dtype == torch.uint8:
                img_out = torch.from_numpy((img_out * 255).astype(np.uint8)).permute(
                    2, 0, 1
                )
            else:
                img_out = torch.from_numpy(img_out).permute(2, 0, 1)
        return (img_out, target) if target is not None else img_out


def _in_batch_copy_paste(images, targets, cfg):
    """
    Applies copy-paste augmentation using donors from the current batch.
    Operates purely on PyTorch tensors to maximize CPU worker speed.
    """
    B = len(images)
    if B < 2:
        return images, targets

    new_images = list(images)
    new_targets = list(targets)

    for i in range(B):
        if random.random() > cfg.copy_paste_prob:
            continue

        # 1. Pick a random donor from the same batch
        donor_idx = random.choice([j for j in range(B) if j != i])
        d_img = images[donor_idx]
        d_tgt = targets[donor_idx]

        if len(d_tgt["boxes"]) == 0:
            continue

        # 2. Clone the target image and annotations so we can modify them
        img = new_images[i].clone()
        tgt = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in new_targets[i].items()
        }
        _, h_tgt, w_tgt = img.shape

        # 3. Pick 1-3 random objects from the donor
        n_paste = min(
            random.randint(1, cfg.copy_paste_max_objects), len(d_tgt["boxes"])
        )
        indices = random.sample(range(len(d_tgt["boxes"])), n_paste)

        for idx in indices:
            # Get donor object data
            x1, y1, x2, y2 = d_tgt["boxes"][idx].int().tolist()
            obj_mask = d_tgt["masks"][idx]  # (H, W)
            obj_label = d_tgt["labels"][idx]

            # Skip excluded categories (e.g. ROV)
            if cfg.copy_paste_exclude_cats and int(obj_label) in cfg.copy_paste_exclude_cats:
                continue

            # Crop
            crop_mask = obj_mask[y1:y2, x1:x2].unsqueeze(0)  # (1, oh, ow)
            crop_pixels = d_img[:, y1:y2, x1:x2]  # (3, oh, ow)

            _, oh, ow = crop_mask.shape
            if (
                oh < cfg.copy_paste_min_crop_side_length
                or ow < cfg.copy_paste_min_crop_side_length
            ):
                continue

            # Scale Jitter
            # Bias heavily toward shrinking the donor object to create artificial "small" targets
            random_value = random.random()
            scale_range_weights = np.cumsum(cfg.copy_paste_scale_range_weights)
            scale_range_weights = scale_range_weights / scale_range_weights[-1]
            for scale_range, weight_psum in zip(
                cfg.copy_paste_scale_ranges, scale_range_weights.tolist()
            ):
                if random_value < weight_psum:
                    scale = random.uniform(scale_range[0], scale_range[1])
                    break
            new_h, new_w = max(4, int(oh * scale)), max(4, int(ow * scale))
            crop_pixels = F.resize(
                crop_pixels, [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR
            )
            crop_mask = F.resize(
                crop_mask, [new_h, new_w], interpolation=F.InterpolationMode.NEAREST
            )

            # Rotation Jitter
            angle = random.uniform(*cfg.copy_paste_rotation_range)
            if abs(angle) > 1:
                crop_pixels = F.rotate(crop_pixels, angle, expand=True)
                crop_mask = F.rotate(crop_mask, angle, expand=True)
                new_h, new_w = crop_mask.shape[1], crop_mask.shape[2]

            # Random Placement
            if new_h >= h_tgt or new_w >= w_tgt:
                continue
            py = random.randint(0, h_tgt - new_h)
            px = random.randint(0, w_tgt - new_w)

            # Color match: Reinhard transfer (match mean/std per channel)
            # Use the target region as the reference for color statistics
            if cfg.copy_paste_color_match:
                tgt_region = img[:, py : py + new_h, px : px + new_w]  # (3, nh, nw)
                obj_pixels = crop_mask[0] > 0  # (nh, nw)
                if obj_pixels.sum() > 1:
                    for c in range(3):
                        src_vals = crop_pixels[c][obj_pixels]
                        tgt_vals = tgt_region[c].float()
                        src_mean, src_std = src_vals.mean(), src_vals.std().clamp(
                            min=1e-6
                        )
                        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std().clamp(
                            min=1e-6
                        )
                        crop_pixels[c][obj_pixels] = (
                            (src_vals - src_mean) * (tgt_std / src_std) + tgt_mean
                        ).clamp(0, 1)

            # Paste pixels
            mask_3ch = crop_mask.expand(3, -1, -1)
            img[:, py : py + new_h, px : px + new_w] = torch.where(
                mask_3ch > 0, crop_pixels, img[:, py : py + new_h, px : px + new_w]
            )

            # Create full-size mask
            full_mask = torch.zeros(
                (h_tgt, w_tgt), dtype=crop_mask.dtype, device=crop_mask.device
            )
            full_mask[py : py + new_h, px : px + new_w] = crop_mask[0]

            # --- Occlusion Handling ---
            # Subtract pasted mask from existing objects, recompute boxes
            valid_indices = []
            new_boxes = []
            for m_idx in range(len(tgt["masks"])):
                tgt["masks"][m_idx] = tgt["masks"][m_idx] & (full_mask == 0)
                remaining = tgt["masks"][m_idx]
                if remaining.sum() >= cfg.copy_paste_min_object_area:
                    # Recompute bounding box from remaining mask
                    mys, mxs = torch.where(remaining > 0)
                    new_boxes.append(
                        [
                            mxs.min().float(),
                            mys.min().float(),
                            (mxs.max() + 1).float(),
                            (mys.max() + 1).float(),
                        ]
                    )
                    valid_indices.append(m_idx)

            # Filter occluded objects and update boxes
            if valid_indices:
                tgt["boxes"] = torch.tensor(new_boxes, device=tgt["boxes"].device)
            else:
                tgt["boxes"] = tgt["boxes"][valid_indices]
            tgt["labels"] = tgt["labels"][valid_indices]
            tgt["masks"] = tgt["masks"][valid_indices]
            tgt["area"] = tgt["area"][valid_indices]
            tgt["iscrowd"] = tgt["iscrowd"][valid_indices]

            # Compute new bounding box
            ys, xs = torch.where(full_mask > 0)
            if len(ys) < cfg.copy_paste_min_object_area:
                continue
            bx1, by1, bx2, by2 = (
                xs.min().float(),
                ys.min().float(),
                (xs.max() + 1).float(),
                (ys.max() + 1).float(),
            )

            # Append new object
            tgt["boxes"] = torch.cat(
                (
                    tgt["boxes"],
                    torch.tensor([[bx1, by1, bx2, by2]], device=tgt["boxes"].device),
                )
            )
            tgt["labels"] = torch.cat(
                (tgt["labels"], torch.tensor([obj_label], device=tgt["labels"].device))
            )
            tgt["masks"] = torch.cat((tgt["masks"], full_mask.unsqueeze(0)))
            tgt["area"] = torch.cat(
                (
                    tgt["area"],
                    torch.tensor([full_mask.sum().float()], device=tgt["area"].device),
                )
            )
            tgt["iscrowd"] = torch.cat(
                (tgt["iscrowd"], torch.tensor([0], device=tgt["iscrowd"].device))
            )

        new_images[i] = img
        new_targets[i] = tgt

    return tuple(new_images), tuple(new_targets)
