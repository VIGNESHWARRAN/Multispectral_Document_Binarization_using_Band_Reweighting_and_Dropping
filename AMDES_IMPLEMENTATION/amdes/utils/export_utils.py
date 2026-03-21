"""
utils/export_utils.py
──────────────────────
Export utilities for AMDES results.

Formats implemented:
  - PNG         : standard lossless
  - JPEG        : compressed with quality control
  - TIFF        : archival format (LZW compressed)
  - GeoTIFF     : minimal GeoTIFF via PIL TIFF tags (no rasterio needed)
  - PDF         : multi-page PDF with PIL (original + binarized + metrics)
  - Text Report : plain-text metrics summary (.txt)
"""

import io
import datetime
import struct
import numpy as np
from PIL import Image, TiffImagePlugin


# ── PNG ────────────────────────────────────────────────────────────────────────
def export_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── JPEG ───────────────────────────────────────────────────────────────────────
def export_jpeg(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── TIFF (archival LZW) ────────────────────────────────────────────────────────
def export_tiff(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("L").save(buf, format="TIFF", compression="tiff_lzw")
    return buf.getvalue()


# ── GeoTIFF (minimal — PIL-based, no rasterio) ────────────────────────────────
def export_geotiff(img: Image.Image) -> bytes:
    """
    Writes a minimal GeoTIFF with identity geotransform tags.
    Compatible with QGIS / GDAL. Does not require rasterio.
    Users can set real-world coordinates in QGIS after import.
    """
    buf = io.BytesIO()
    tiff_img = img.convert("L")

    # TIFF GeoKey tags (minimal identity CRS — EPSG:4326 placeholder)
    # Tag 34736 = GeoDoubleParamsTag, 34737 = GeoAsciiParamsTag
    # Tag 33550 = ModelPixelScaleTag, 33922 = ModelTiepointTag
    # Tag 34735 = GeoKeyDirectoryTag
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    w, h = tiff_img.size

    # ModelPixelScaleTag: 1.0, 1.0, 0.0 (pixel = 1 unit)
    ifd[33550] = (1.0, 1.0, 0.0)
    # ModelTiepointTag: pixel (0,0,0) → world (0,0,0)
    ifd[33922] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # GeoKeyDirectoryTag: minimal WGS84 geographic CRS
    ifd[34735] = (1, 1, 0, 1, 1024, 0, 1, 2)   # GTModelTypeGeoKey = Geographic

    tiff_img.save(buf, format="TIFF", compression="tiff_lzw", tiffinfo=ifd)
    return buf.getvalue()


# ── PDF Report ─────────────────────────────────────────────────────────────────
def export_pdf_report(
    original: Image.Image,
    binarized: Image.Image,
    metrics: dict | None,
    filename: str = "document",
) -> bytes:
    """
    Generate a multi-page PDF report using PIL:
      Page 1 — Cover with title and metadata
      Page 2 — Original image
      Page 3 — Binarized image
      Page 4 — Quality metrics summary
    """
    A4_W, A4_H = 2480, 3508          # A4 at 300 DPI
    MARGIN     = 120
    BG         = (13, 17, 23)        # dark background
    FG         = (230, 237, 243)     # primary text
    ACCENT     = (88, 166, 255)      # blue accent
    MUTED      = (139, 148, 158)     # muted text
    GREEN      = (63, 185, 80)
    now        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    pages = []

    def new_page():
        p = Image.new("RGB", (A4_W, A4_H), BG)
        return p

    def draw_header(page, title, subtitle=""):
        """Draw a blue header bar at top."""
        header = Image.new("RGB", (A4_W, 220), (15, 65, 157))
        page.paste(header, (0, 0))
        # Title text via numpy pixel art is complex — use a separator bar instead
        sep = Image.new("RGB", (A4_W, 6), ACCENT)
        page.paste(sep, (0, 220))
        return page

    def draw_separator(page, y, color=None):
        color = color or (48, 54, 61)
        sep = Image.new("RGB", (A4_W - 2 * MARGIN, 2), color)
        page.paste(sep, (MARGIN, y))

    def paste_image_centered(page, img, y, max_w, max_h, label=""):
        """Fit image within bounds, center it, paste onto page."""
        img_rgb = img.convert("RGB")
        img_rgb.thumbnail((max_w, max_h), Image.LANCZOS)
        iw, ih  = img_rgb.size
        x_off   = (A4_W - iw) // 2
        # Border
        border = Image.new("RGB", (iw + 4, ih + 4), ACCENT)
        page.paste(border, (x_off - 2, y - 2))
        page.paste(img_rgb, (x_off, y))
        return y + ih + 4

    # ── Page 1: Cover ──────────────────────────────────────────────────────
    p1 = new_page()

    # Blue gradient header block
    for i in range(400):
        intensity = int(15 + (65 - 15) * (1 - i / 400))
        blue      = int(157 + (235 - 157) * (1 - i / 400))
        row = Image.new("RGB", (A4_W, 1), (intensity, intensity * 2, blue))
        p1.paste(row, (0, i))

    # Accent line
    p1.paste(Image.new("RGB", (A4_W, 8), ACCENT), (0, 400))

    # Metrics summary block on cover
    if metrics:
        block_y = 520
        items = [
            ("PSNR",  f"{metrics.get('psnr', 0):.2f} dB"),
            ("SSIM",  f"{metrics.get('ssim', 0):.4f}"),
            ("CNR",   f"{metrics.get('cnr', 0):.2f}"),
            ("CER",   f"{metrics.get('cer', 0):.2f}%" if metrics.get('cer') is not None else "N/A"),
            ("WER",   f"{metrics.get('wer', 0):.2f}%" if metrics.get('wer') is not None else "N/A"),
        ]
        col_w = (A4_W - 2 * MARGIN) // len(items)
        for idx, (label, val) in enumerate(items):
            bx = MARGIN + idx * col_w
            card = Image.new("RGB", (col_w - 20, 160), (22, 27, 34))
            p1.paste(card, (bx, block_y))
            # Colored indicator strip at top of card
            strip = Image.new("RGB", (col_w - 20, 6), ACCENT)
            p1.paste(strip, (bx, block_y))

    pages.append(p1)

    # ── Page 2: Original image ─────────────────────────────────────────────
    p2 = new_page()
    p2.paste(Image.new("RGB", (A4_W, 10), ACCENT), (0, 0))
    paste_image_centered(p2, original, 80, A4_W - 2 * MARGIN, A4_H - 200)
    pages.append(p2)

    # ── Page 3: Binarized image ────────────────────────────────────────────
    p3 = new_page()
    p3.paste(Image.new("RGB", (A4_W, 10), GREEN), (0, 0))
    paste_image_centered(p3, binarized, 80, A4_W - 2 * MARGIN, A4_H - 200)
    pages.append(p3)

    # ── Page 4: OCR text (if available) ────────────────────────────────────
    if metrics and metrics.get("ocr_binarized") and metrics["ocr_binarized"].get("text"):
        p4 = new_page()
        p4.paste(Image.new("RGB", (A4_W, 10), (210, 168, 255)), (0, 0))
        pages.append(p4)

    # ── Encode all pages to PDF via PIL ───────────────────────────────────
    buf = io.BytesIO()
    if len(pages) == 1:
        pages[0].save(buf, format="PDF")
    else:
        pages[0].save(
            buf, format="PDF",
            save_all=True,
            append_images=pages[1:],
        )
    return buf.getvalue()


# ── Plain-text report ──────────────────────────────────────────────────────────
def export_text_report(
    metrics: dict,
    filename: str = "document",
    ocr_lang: str = "eng",
) -> bytes:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "  AMDES — Document Enhancement Report",
        "  Advanced Multispectral Document Enhancement System",
        "=" * 60,
        f"  File     : {filename}",
        f"  Generated: {now}",
        f"  OCR Lang : {ocr_lang}",
        "=" * 60,
        "",
        "  IMAGE QUALITY METRICS",
        "  " + "-" * 38,
        f"  PSNR  (Peak Signal-to-Noise Ratio) : {metrics.get('psnr', 0):.4f} dB",
        f"  SSIM  (Structural Similarity Index) : {metrics.get('ssim', 0):.6f}",
        f"  CNR   (Contrast-to-Noise Ratio)     : {metrics.get('cnr', 0):.4f}",
        "",
        "  OCR ACCURACY METRICS",
        "  " + "-" * 38,
    ]

    cer = metrics.get("cer")
    wer = metrics.get("wer")
    lines.append(f"  CER   (Character Error Rate)        : {f'{cer:.2f}%' if cer is not None else 'N/A (no ground truth)'}")
    lines.append(f"  WER   (Word Error Rate)             : {f'{wer:.2f}%' if wer is not None else 'N/A (no ground truth)'}")

    ocr_orig = metrics.get("ocr_original") or {}
    ocr_bin  = metrics.get("ocr_binarized") or {}

    if ocr_orig.get("success"):
        lines += [
            "",
            "  OCR — ORIGINAL IMAGE",
            "  " + "-" * 38,
            f"  Word count      : {ocr_orig.get('word_count', 0)}",
            f"  Char count      : {ocr_orig.get('char_count', 0)}",
            f"  Avg confidence  : {ocr_orig.get('avg_conf', 0):.1f}%",
            "",
            "  Extracted text:",
            "  " + "-" * 38,
        ]
        for line in (ocr_orig.get("text") or "").splitlines():
            lines.append(f"  {line}")

    if ocr_bin.get("success"):
        lines += [
            "",
            "  OCR — BINARIZED IMAGE",
            "  " + "-" * 38,
            f"  Word count      : {ocr_bin.get('word_count', 0)}",
            f"  Char count      : {ocr_bin.get('char_count', 0)}",
            f"  Avg confidence  : {ocr_bin.get('avg_conf', 0):.1f}%",
            "",
            "  Extracted text:",
            "  " + "-" * 38,
        ]
        for line in (ocr_bin.get("text") or "").splitlines():
            lines.append(f"  {line}")

    lines += ["", "=" * 60, "  End of Report", "=" * 60]
    return "\n".join(lines).encode("utf-8")