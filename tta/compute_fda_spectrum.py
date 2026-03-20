"""
Precompute Fourier amplitude spectra for a set of images.

Output modes:
  --mode mean    → Single mean amplitude spectrum (for FDA test-time preprocessing)
  --mode bank    → Per-image or clustered spectra in one file (for FSR training augmentation)
  --mode both    → Both of the above (requires --bank-output)

Options:
  --half         → Store amplitudes as float16 to halve file size
  --cluster N    → K-means cluster the spectra (on the low-freq disk) into N
                   groups and save per-cluster means instead of per-image spectra.
                   Clusters with fewer than --min-cluster-size members are merged
                   into their nearest neighbor.

Usage:
    # Mean spectrum for FDA test-time preprocessing
    python compute_fda_spectrum.py \
        --img-dir seaclear_data/images_480p \
        --output fda_spectra/seaclear_480p_mean.npz \
        --mode mean --max-images 200

    # Style bank for FSR training augmentation
    python compute_fda_spectrum.py \
        --img-dir style_bank_images/ \
        --output fda_spectra/style_bank.npz \
        --mode bank --max-images 500

    # Clustered style bank (large image set → compact representation)
    python compute_fda_spectrum.py \
        --img-dir style_bank_images/ \
        --output fda_spectra/style_bank_clustered.npz \
        --mode bank --max-images 1000 --cluster 20 --half

    # Both at once
    python compute_fda_spectrum.py \
        --img-dir seaclear_data/images_480p \
        --output fda_spectra/seaclear_480p_mean.npz \
        --bank-output fda_spectra/seaclear_480p_bank.npz \
        --mode both --max-images 200
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def _make_low_freq_mask(h: int, w: int, beta: float) -> np.ndarray:
    """Create a circular mask for the low-frequency region of the spectrum."""
    cy, cx = h // 2, w // 2
    radius = max(int(min(h, w) * beta), 1)
    Y, X = np.ogrid[:h, :w]
    return ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2


def cluster_spectra(
    amplitudes: np.ndarray,
    n_clusters: int,
    beta: float = 0.05,
    min_cluster_size: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Cluster amplitude spectra using k-means on the low-frequency disk.

    Args:
        amplitudes: (N, H, W, 3) per-image amplitude spectra
        n_clusters: target number of clusters
        beta: disk radius for extracting clustering features
        min_cluster_size: clusters smaller than this are merged into nearest neighbor
        seed: random seed for k-means

    Returns:
        (K, H, W, 3) cluster centroid spectra (K <= n_clusters)
    """
    from sklearn.cluster import KMeans

    N, H, W, C = amplitudes.shape
    n_clusters = min(n_clusters, N)

    # Extract low-frequency disk region as clustering features
    mask = _make_low_freq_mask(H, W, beta)
    disk_pixels = np.count_nonzero(mask)
    print(f"[Cluster] Disk: beta={beta}, radius={max(int(min(H, W) * beta), 1)}px, "
          f"{disk_pixels} pixels ({100 * disk_pixels / (H * W):.1f}% of spectrum)")

    # Flatten disk region per image: (N, disk_pixels * 3)
    features = np.zeros((N, disk_pixels * C), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            features[i, c * disk_pixels:(c + 1) * disk_pixels] = amplitudes[i, :, :, c][mask]

    # Normalize features for better k-means behavior
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True).clip(min=1e-8)
    features_norm = (features - feat_mean) / feat_std

    print(f"[Cluster] Running k-means: {N} spectra → {n_clusters} clusters ...")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features_norm)

    # Count cluster sizes and identify small clusters
    cluster_ids, counts = np.unique(labels, return_counts=True)
    print(f"[Cluster] Cluster sizes: {sorted(counts.tolist(), reverse=True)}")

    # Merge small clusters into nearest neighbor
    small = cluster_ids[counts < min_cluster_size]
    if len(small) > 0:
        large = cluster_ids[counts >= min_cluster_size]
        if len(large) == 0:
            # All clusters are small — just keep them all
            print(f"[Cluster] Warning: all clusters have <{min_cluster_size} members; "
                  f"keeping all {len(cluster_ids)} clusters")
        else:
            large_centers = km.cluster_centers_[large]
            for s in small:
                # Find nearest large cluster
                dists = np.linalg.norm(large_centers - km.cluster_centers_[s], axis=1)
                nearest = large[np.argmin(dists)]
                n_merged = (labels == s).sum()
                labels[labels == s] = nearest
                print(f"[Cluster] Merged cluster {s} ({n_merged} members) → cluster {nearest}")

    # Compute per-cluster mean spectra (using full spectra, not just disk)
    final_ids = np.unique(labels)
    centroids = []
    for cid in final_ids:
        members = amplitudes[labels == cid]
        centroids.append(members.mean(axis=0))
        print(f"[Cluster] Final cluster {cid}: {len(members)} members")

    result = np.stack(centroids, axis=0).astype(np.float32)
    print(f"[Cluster] Output: {result.shape[0]} cluster centroids")
    return result


def compute_spectra(img_dir: Path, max_images: int = 200, seed: int = 42,
                    native_fft: bool = False) -> dict:
    """Compute per-image Fourier amplitude spectra and their mean.

    By default, images are resized to a common (median) resolution before FFT.
    With native_fft=True, FFT is computed at each image's native resolution and
    the resulting amplitude spectrum is resized to a common target (max resolution
    across all images). This preserves low-frequency information that would be
    lost by downscaling images before FFT.

    Args:
        img_dir: directory containing images (jpg/png/jpeg)
        max_images: max images to sample (for speed / memory)
        seed: random seed for sampling
        native_fft: if True, compute FFT at native resolution then resize spectrum

    Returns:
        dict with keys:
            'amplitudes': (N, H, W, 3) float32 per-image amplitude spectra
            'mean_amplitude': (H, W, 3) float64 mean amplitude spectrum
            'target_h': int, height of output spectra
            'target_w': int, width of output spectra
            'n_images': int, number of images used
            'img_dir': str, source directory
    """
    paths = sorted(
        glob.glob(str(img_dir / "*.jpg"))
        + glob.glob(str(img_dir / "*.png"))
        + glob.glob(str(img_dir / "*.jpeg"))
    )
    print(f"Found {len(paths)} images in {img_dir}")

    if len(paths) == 0:
        raise ValueError(f"No images found in {img_dir}")

    if len(paths) > max_images:
        rng = np.random.RandomState(seed)
        paths = list(rng.choice(paths, max_images, replace=False))
        print(f"Sampled {max_images} images (seed={seed})")

    # First pass: find target dimensions
    heights, widths = [], []
    for p in paths[:min(50, len(paths))]:
        img = cv2.imread(p)
        if img is not None:
            heights.append(img.shape[0])
            widths.append(img.shape[1])

    if native_fft:
        target_h = int(np.max(heights))
        target_w = int(np.max(widths))
        print(f"Target dimensions: {target_h} x {target_w} (max of first {len(heights)} images)")
        print(f"  FFT computed at native resolution, spectra resized to target")
    else:
        target_h = int(np.median(heights))
        target_w = int(np.median(widths))
        print(f"Target dimensions: {target_h} x {target_w} (median of first {len(heights)} images)")

    # Second pass: compute per-image amplitude spectra
    all_amps = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if native_fft:
            # FFT at native resolution, then resize the amplitude spectrum
            amp = np.stack(
                [np.abs(np.fft.fftshift(np.fft.fft2(img[:, :, c]))) for c in range(3)],
                axis=-1,
            )
            if amp.shape[0] != target_h or amp.shape[1] != target_w:
                amp = cv2.resize(amp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Resize image first, then FFT
            img = cv2.resize(img, (target_w, target_h))
            amp = np.stack(
                [np.abs(np.fft.fftshift(np.fft.fft2(img[:, :, c]))) for c in range(3)],
                axis=-1,
            )

        all_amps.append(amp)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(paths)} images ...")

    if not all_amps:
        raise ValueError(f"No valid images found in {img_dir}")

    amplitudes = np.stack(all_amps, axis=0).astype(np.float32)  # (N, H, W, 3)
    mean_amp = amplitudes.mean(axis=0).astype(np.float64)  # (H, W, 3)
    print(f"Computed spectra from {len(all_amps)} images")

    return {
        "amplitudes": amplitudes,
        "mean_amplitude": mean_amp,
        "target_h": target_h,
        "target_w": target_w,
        "n_images": len(all_amps),
        "img_dir": str(img_dir),
    }


def save_mean(result: dict, out_path: Path, half: bool = False):
    """Save mean amplitude spectrum (for FDA test-time preprocessing)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_amp = result["mean_amplitude"]
    if half:
        mean_amp = mean_amp.astype(np.float16)
    np.savez_compressed(
        str(out_path),
        mean_amplitude=mean_amp,
        target_h=result["target_h"],
        target_w=result["target_w"],
        n_images=result["n_images"],
        img_dir=result["img_dir"],
    )
    print(f"\nSaved mean spectrum to {out_path}")
    print(f"  Shape: {mean_amp.shape}, dtype: {mean_amp.dtype}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")


def save_bank(result: dict, out_path: Path, half: bool = False,
              n_clusters: int = 0, cluster_beta: float = 0.05,
              min_cluster_size: int = 3, seed: int = 42,
              disk_only: bool = False, disk_beta: float = 0.0):
    """Save per-image or clustered amplitude spectra (for FSR training augmentation).

    If disk_only=True, extracts only the low-frequency disk region and saves it
    in a compact format: 'disk_amplitudes' (N, disk_pixels, 3) plus 'disk_beta'
    so the loader can reconstruct the mask.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    amplitudes = result["amplitudes"]

    if n_clusters > 0:
        amplitudes = cluster_spectra(
            amplitudes,
            n_clusters=n_clusters,
            beta=cluster_beta,
            min_cluster_size=min_cluster_size,
            seed=seed,
        )

    H = int(result["target_h"])
    W = int(result["target_w"])

    if disk_only:
        beta = disk_beta if disk_beta > 0 else 0.01  # default to typical FSR beta
        mask = _make_low_freq_mask(H, W, beta)
        N = amplitudes.shape[0]
        disk_pixels = np.count_nonzero(mask)
        # Extract disk region: (N, disk_pixels, 3)
        disk_amps = np.zeros((N, disk_pixels, 3), dtype=np.float32)
        for i in range(N):
            for c in range(3):
                disk_amps[i, :, c] = amplitudes[i, :, :, c][mask]

        if half:
            disk_amps = disk_amps.astype(np.float16)

        np.savez_compressed(
            str(out_path),
            disk_amplitudes=disk_amps,
            disk_beta=np.float64(beta),
            target_h=H,
            target_w=W,
            n_images=result["n_images"],
            img_dir=result["img_dir"],
        )
        print(f"\nSaved disk-only style bank to {out_path}")
        print(f"  Shape: {disk_amps.shape}, dtype: {disk_amps.dtype}")
        print(f"  Disk: beta={beta}, {disk_pixels} pixels per image")
        print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")
    else:
        if half:
            amplitudes = amplitudes.astype(np.float16)

        np.savez_compressed(
            str(out_path),
            amplitudes=amplitudes,
            target_h=H,
            target_w=W,
            n_images=result["n_images"],
            img_dir=result["img_dir"],
        )
        print(f"\nSaved style bank to {out_path}")
        print(f"  Shape: {amplitudes.shape}, dtype: {amplitudes.dtype}")
        print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Fourier amplitude spectra for FDA / FSR."
    )
    parser.add_argument(
        "--img-dir", required=True, type=str,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output path for the primary .npz file"
    )
    parser.add_argument(
        "--mode", choices=["mean", "bank", "both"], default="mean",
        help="What to output: 'mean' (FDA), 'bank' (FSR), or 'both'"
    )
    parser.add_argument(
        "--bank-output", type=str, default="",
        help="Output path for the style bank .npz (required if --mode both)"
    )
    parser.add_argument(
        "--max-images", type=int, default=200,
        help="Maximum number of images to sample (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Store amplitudes as float16 to reduce file size"
    )
    parser.add_argument(
        "--cluster", type=int, default=0, metavar="N",
        help="K-means cluster spectra into N groups and save cluster means "
             "instead of per-image spectra (bank mode only)"
    )
    parser.add_argument(
        "--cluster-beta", type=float, default=0.05,
        help="Disk radius for clustering features (default: 0.05). "
             "Larger than FSR beta since we want to capture enough "
             "style signal for meaningful clusters."
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=3,
        help="Clusters with fewer members are merged into nearest neighbor (default: 3)"
    )
    parser.add_argument(
        "--disk-only", action="store_true",
        help="Store only the low-frequency disk pixels instead of full spectra "
             "(much smaller files; bank mode only)"
    )
    parser.add_argument(
        "--disk-beta", type=float, default=0.0,
        help="Disk radius for --disk-only extraction (default: uses fsr_beta=0.01). "
             "Must be >= the fsr_beta you plan to use at training time."
    )
    parser.add_argument(
        "--native-fft", action="store_true",
        help="Compute FFT at each image's native resolution, then resize the "
             "amplitude spectrum to the common target (max resolution in the set). "
             "Preserves low-frequency information from high-res images."
    )
    args = parser.parse_args()

    if args.mode == "both" and not args.bank_output:
        parser.error("--bank-output is required when --mode both")

    if args.cluster > 0 and args.mode == "mean":
        parser.error("--cluster only applies to bank mode (use --mode bank or --mode both)")

    if args.disk_only and args.mode == "mean":
        parser.error("--disk-only only applies to bank mode (use --mode bank or --mode both)")

    result = compute_spectra(
        Path(args.img_dir),
        max_images=args.max_images,
        seed=args.seed,
        native_fft=args.native_fft,
    )

    out_path = Path(args.output)
    if args.mode == "mean":
        save_mean(result, out_path, half=args.half)
    elif args.mode == "bank":
        save_bank(result, out_path, half=args.half,
                  n_clusters=args.cluster, cluster_beta=args.cluster_beta,
                  min_cluster_size=args.min_cluster_size, seed=args.seed,
                  disk_only=args.disk_only, disk_beta=args.disk_beta)
    else:  # both
        save_mean(result, out_path, half=args.half)
        save_bank(result, Path(args.bank_output), half=args.half,
                  n_clusters=args.cluster, cluster_beta=args.cluster_beta,
                  min_cluster_size=args.min_cluster_size, seed=args.seed,
                  disk_only=args.disk_only, disk_beta=args.disk_beta)


if __name__ == "__main__":
    main()
