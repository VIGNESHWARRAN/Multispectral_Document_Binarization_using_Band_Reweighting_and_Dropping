"""
utils/image_utils.py
─────────────────────
Shared image helpers used by both the backend pipeline and Streamlit pages.
"""

import io
import numpy as np
from PIL import Image


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode any PIL image as PNG bytes."""
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def bytes_to_pil(data: bytes) -> Image.Image:
    """Decode raw bytes → PIL Image."""
    return Image.open(io.BytesIO(data))


def pil_to_grayscale_array(img: Image.Image) -> np.ndarray:
    """PIL Image → 2-D uint8 numpy array (grayscale)."""
    return np.array(img.convert("L"), dtype=np.uint8)


def image_stats(img: Image.Image) -> dict:
    """
    Return basic statistics for a PIL image.

    Returns:
        dict with keys: mean, std, ink_pct, width, height, mode
    """
    arr = pil_to_grayscale_array(img)
    w, h = img.size
    ink_pct = float((arr < 128).sum() / arr.size * 100)
    return {
        "mean":    float(arr.mean()),
        "std":     float(arr.std()),
        "ink_pct": round(ink_pct, 2),
        "width":   w,
        "height":  h,
        "mode":    img.mode,
    }


def resize_if_needed(img: Image.Image, max_dim: int = 4096) -> Image.Image:
    """
    Downscale image proportionally if either dimension exceeds max_dim.
    Avoids sending huge images to the model.
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
