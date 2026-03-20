"""
Add synthetic CCTV-style text overlays to images, mimicking TrashCan's
text overlay style.

Typical TrashCan overlays:
- 1-2 lines of text at top (timestamp, camera ID, coordinates)
- 1-2 lines of text at bottom (status info, depth, etc.)
- Brand logo region in bottom-right corner
- White text with black outline (stroke)
- Some configurations have semi-transparent black background boxes

This script processes images in-place or to an output directory, drawing
random overlay configurations sampled from a set of templates.

Usage:
    python add_text_overlay.py \
        --input-dir seaclear_data/images_480p \
        --output-dir seaclear_data/images_480p_textoverlay

    # Preview mode: show a few samples without saving
    python add_text_overlay.py \
        --input-dir seaclear_data/images_480p \
        --output-dir seaclear_data/images_480p_textoverlay \
        --preview 5
"""

import argparse
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ═══════════════════════════════════════════════════════════════════════
# Text generation helpers
# ═══════════════════════════════════════════════════════════════════════


def random_timestamp(base=None):
    if base is None:
        base = datetime(2019, 1, 1) + timedelta(days=random.randint(0, 1500))
    t = base + timedelta(seconds=random.randint(0, 86400))
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%b-%Y %H:%M:%S",
    ]
    s = t.strftime(random.choice(formats))
    if "%f" in s:
        s = s[:-4]  # trim microseconds to 2 digits
    if random.random() < 0.7:
        for _ in range(random.randint(0, 4)):
            s += f"     {random.uniform(0, 3000):.1f}"
    return s


def random_camera_id():
    templates = [
        lambda: f"CAM-{random.randint(1,12):02d}",
        lambda: f"Camera {random.randint(1,8)}",
        lambda: f"CH{random.randint(1,16):02d}",
        lambda: f"DVR {random.randint(1,4)} Ch{random.randint(1,8)}",
    ]
    return random.choice(templates)()


def random_coordinates():
    lat = random.uniform(-60, 60)
    lon = random.uniform(-180, 180)
    depth = random.uniform(10, 4000)
    templates = [
        lambda: f"{lat:+.4f} {lon:+.4f}",
        lambda: f"LAT {lat:.3f} LON {lon:.3f}",
        lambda: f"Depth: {depth:.1f}m",
        lambda: f"D={depth:.0f}m  {lat:.2f}N {abs(lon):.2f}{'W' if lon<0 else 'E'}",
    ]
    return random.choice(templates)()


def random_status_line():
    templates = [
        lambda: f"ALT: {random.uniform(0, 500):.1f}m  HDG: {random.randint(0,359)}",
        lambda: f"Frame {random.randint(1,99999):05d}",
    ]
    return random.choice(templates)()


def random_brand():
    brands = ["©JAMSTEC"]
    return random.choice(brands)


def random_info_line1():
    templates = [
        lambda: (
            f"HD:    {random.uniform(0, 500):.1f}    {random.uniform(-50, 50):.1f}"
            if random.random() < 0.3
            else f"HD:    {random.uniform(0, 500):.1f}    {random.uniform(-50, 50):.1f}    D: {random.uniform(0, 3000):.1f}   Do: {random.uniform(0, 10):.1f}"
        ),
    ]
    return random.choice(templates)()


def random_info_line2():
    templates = [
        lambda: (
            f"OCD:   {random.uniform(-50, 50):.1f}    {random.uniform(-50, 50):.1f}"
            if random.random() < 0.3
            else f"OCD:   {random.uniform(0, 500):.1f}    {random.uniform(-50, 50):.1f}    S: {random.uniform(0, 50):.1f}  T: {random.uniform(0, 10):.1f}"
        ),
    ]
    return random.choice(templates)()


# ═══════════════════════════════════════════════════════════════════════
# Overlay configuration templates
# ═══════════════════════════════════════════════════════════════════════


def generate_overlay_config():
    """Generate a random overlay configuration (text lines + positions).

    Returns a dict describing what to draw. Each config is meant to be
    shared across all frames of a "virtual video" — but for SC images
    we just assign one config per image (or per chunk if desired).
    """
    config = {
        "top_lines": [],
        "bottom_lines": [],
        "brand": None,
        "has_bg_box": random.random() < 0.3,  # ~30% have background boxes
        "bg_alpha": random.uniform(0.1, 0.4),
        "font_size": random.choice([12, 13, 14]),
        "outline_width": random.choice([1]),
    }

    # Top: 1-2 lines (timestamp, camera ID, coordinates)
    top_options = [random_coordinates, random_coordinates]
    n_top = random.choice([0, 1])
    chosen_top = random.sample(top_options, n_top)
    chosen_top = [random_timestamp] + chosen_top
    for fn in chosen_top:
        config["top_lines"].append(fn())

    # Bottom: 0-2 lines (status, coordinates, depth)
    # n_bottom = random.choice([1, 2, 3])
    # bottom_options = [random_status_line, random_info_line1, random_info_line2]
    # if n_bottom > 0:
    #     chosen_bottom = random.sample(
    #         bottom_options, min(n_bottom, len(bottom_options))
    #     )
    #     for fn in chosen_bottom:
    #         print(f"adding bottom line {fn}")
    #         config["bottom_lines"].append(fn())
    chosen_bottom = []
    if random.random() < 0.2:
        chosen_bottom.append(random_status_line)
    if random.random() < 0.6:
        chosen_bottom.append(random_info_line1)
    if random.random() < 0.4:
        chosen_bottom.append(random_info_line2)
    for fn in chosen_bottom:
        config["bottom_lines"].append(fn())
    # config["bottom_lines"] = [random_info_line1, random_info_line2]

    # Brand logo text in bottom-right (~60% of configs)
    if random.random() < 0.7:
        config["brand"] = random_brand()

    return config


# ═══════════════════════════════════════════════════════════════════════
# Drawing
# ═══════════════════════════════════════════════════════════════════════


def draw_text_with_outline(
    draw,
    xy,
    text,
    font,
    fill=(255, 255, 255),
    outline_fill=(0, 0, 0),
    outline_width=1,
    mask_draw=None,
):
    """Draw text with an outline (stroke) effect.

    If mask_draw is provided, also renders the same text (with outline)
    onto the mask as white pixels for pixel-accurate text masking.
    """
    x, y = xy
    # Draw outline
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_fill)
            if mask_draw is not None:
                mask_draw.text((x + dx, y + dy), text, font=font, fill=255)
    # Draw main text
    draw.text((x, y), text, font=font, fill=fill)
    if mask_draw is not None:
        mask_draw.text((x, y), text, font=font, fill=255)


def apply_overlay(
    img: Image.Image, config: dict, timestamp_override: str = None
) -> tuple[Image.Image, np.ndarray]:
    """Apply a text overlay configuration to a PIL Image.

    Args:
        img: Input PIL Image (RGB).
        config: Overlay config from generate_overlay_config().
        timestamp_override: If provided, replace any timestamp with this
            (useful for making timestamps vary per frame).

    Returns:
        (overlaid_image, text_mask): PIL Image with overlay drawn, and a
        binary uint8 mask (H, W) where 255 = text pixel. The mask can be
        used to subtract text regions from annotation masks.
    """
    img = img.copy()
    w, h = img.size

    # Grayscale image for pixel-accurate text mask
    mask_img = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask_img)

    # Try to use a monospace font; fall back to default
    font = None
    brand_font = None
    mono_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    mono_font_path = random.choice(mono_fonts)
    try:
        font = ImageFont.truetype(mono_font_path, config["font_size"])
        brand_font = ImageFont.truetype(mono_font_path, config["font_size"] * 2)
    except (OSError, IOError):
        font = ImageFont.load_default(config["font_size"])
        brand_font = ImageFont.load_default(config["font_size"] * 2)

    draw = ImageDraw.Draw(img, "RGBA")

    margin_x = 8
    margin_y = 5
    line_spacing = config["font_size"] + 4

    outline_w = config["outline_width"]

    bg_color = (
        (0, 0, 0, int(255 * config["bg_alpha"])) if config["has_bg_box"] else None
    )
    pad_x = 4  # horizontal padding around each text segment
    pad_y = 6  # vertical padding around each text segment

    def draw_line_with_bg(x, y, text):
        """Draw a line of text, splitting on runs of 2+ spaces.
        Each segment gets its own tight background box (if bg enabled).
        Also marks the text_mask for each segment."""
        import re

        segments = re.split(r"( {2,})", text)

        cursor_x = x
        for seg in segments:
            if not seg or seg.isspace():
                # Advance cursor by the width of the whitespace
                if seg:
                    seg_bbox = draw.textbbox((0, 0), seg, font=font)
                    cursor_x += seg_bbox[2] - seg_bbox[0]
                continue

            seg_bbox = draw.textbbox((0, 0), seg, font=font)
            seg_w = seg_bbox[2] - seg_bbox[0]
            seg_h = seg_bbox[3] - seg_bbox[1]

            if bg_color:
                draw.rectangle(
                    [
                        cursor_x - pad_x,
                        y,
                        cursor_x + seg_w + pad_x,
                        y + seg_h + pad_y,
                    ],
                    fill=bg_color,
                )

            draw_text_with_outline(
                draw,
                (cursor_x, y),
                seg,
                font,
                outline_width=outline_w,
                mask_draw=mask_draw,
            )
            cursor_x += seg_w

    # Draw top lines (left-aligned)
    y = margin_y
    for line in config["top_lines"]:
        draw_line_with_bg(margin_x, y, line)
        y += line_spacing

    # Draw bottom lines (left-aligned)
    n_bottom_total = len(config["bottom_lines"]) + (1 if config["brand"] else 0)
    y = h - margin_y - n_bottom_total * line_spacing
    for line in config["bottom_lines"]:
        draw_line_with_bg(margin_x, y, line)
        y += line_spacing

    # Draw brand in bottom-right
    if config["brand"]:
        bbox = draw.textbbox((0, 0), config["brand"], font=brand_font)
        text_w = bbox[2] - bbox[0]
        bx = w - text_w - margin_x // 2
        by = h - line_spacing * 1.4 - margin_y // 2
        draw_text_with_outline(
            draw,
            (bx, by),
            config["brand"],
            brand_font,
            outline_width=outline_w,
            mask_draw=mask_draw,
        )

        # Convert mask to numpy, dilate to expand by a few pixels
    text_mask = np.array(mask_img, dtype=np.uint8)
    dilate_px = config.get("mask_dilate", 5)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    return img, text_mask


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Add synthetic CCTV-style text overlays to images."
    )
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output image directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quality", type=int, default=95, help="JPEG save quality")
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="If >0, save only this many samples (for quick preview)",
    )
    parser.add_argument(
        "--config-per",
        choices=["image", "chunk"],
        default="image",
        help="Generate one overlay config per image or per chunk of images",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Chunk size for config-per=chunk mode",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=4,
        help="Pixels to dilate text mask (default: 3)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "text_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if args.preview > 0:
        image_files = image_files[: args.preview]

    print(f"Processing {len(image_files)} images")
    print(f"  Config mode: {args.config_per}")
    if args.config_per == "chunk":
        print(f"  Chunk size: {args.chunk_size}")
    print(f"  Text masks: {mask_dir}")

    config = None
    for i, img_path in enumerate(image_files):
        # Generate config based on mode
        if args.config_per == "image":
            config = generate_overlay_config()
            config["mask_dilate"] = args.mask_dilate
        elif args.config_per == "chunk":
            if i % args.chunk_size == 0:
                config = generate_overlay_config()
                config["mask_dilate"] = args.mask_dilate

        img = Image.open(img_path).convert("RGB")
        img_out, text_mask = apply_overlay(img, config)

        out_path = output_dir / img_path.name
        img_out.save(out_path, quality=args.quality)

        mask_path = mask_dir / (img_path.stem + ".png")
        cv2.imwrite(str(mask_path), text_mask)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(image_files)} ...")

    print(f"Done. Output: {output_dir}")
    print(f"  Text masks: {mask_dir}")


if __name__ == "__main__":
    main()
