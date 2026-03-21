"""
backend/routes/binarize.py
───────────────────────────
POST /binarize — accepts an image upload, returns binarized PNG.
"""

import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from backend.pipeline import run_binarization
from config.settings import ALLOWED_MIME_TYPES, MAX_UPLOAD_MB
from utils.image_utils import pil_to_png_bytes, resize_if_needed

router = APIRouter()
logger = logging.getLogger("AMDES.Routes")


@router.post("/binarize")
async def binarize(file: UploadFile = File(...)):
    """
    Upload a document image → returns a binarized grayscale PNG.

    Binarization priority:
      1. Local model (if USE_LOCAL_MODEL=true)
      2. HuggingFace Inference API
      3. Otsu fallback (always available)
    """
    # ── Validate MIME type ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'. Use PNG, JPG, or TIFF.",
        )

    raw = await file.read()

    # ── Validate file size ────────────────────────────────────────────────────
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_MB} MB.",
        )

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    original_size = img.size
    logger.info(f"Processing: {file.filename} | {original_size} | {img.mode}")

    # Resize very large images before sending to model (avoids OOM / timeouts)
    img = resize_if_needed(img, max_dim=4096)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    result = run_binarization(img)

    # Restore original resolution if we resized earlier
    if result.size != original_size:
        result = result.resize(original_size, Image.LANCZOS)

    return Response(content=pil_to_png_bytes(result), media_type="image/png")
