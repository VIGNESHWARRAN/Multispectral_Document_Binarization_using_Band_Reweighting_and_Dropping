"""
AMDES — FastAPI Backend
Binarization via Hugging Face Inference API (placeholder until local model is ready).

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=hf_your_token_here
    uvicorn main:app --reload --port 8000

Later: set USE_LOCAL_MODEL=true and place model.h5 here to switch to your own weights.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
from PIL import Image, ImageFilter
import io, os, logging
import requests as req

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AMDES")

app = FastAPI(title="AMDES Binarization API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Config ───────────────────────────────────────────────────────────────────
# Get free token at: https://huggingface.co/settings/tokens (read-only is enough)
HF_TOKEN        = os.getenv("HF_TOKEN", "")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
MODEL_PATH      = os.getenv("MODEL_PATH", "model.h5")

# HF models to try in order (free, no GPU needed via Inference API)
HF_MODELS = [
    "ioclab/ribose-binarization",          # document binarization — best fit
    "saifhassan/document-binarization",    # backup document model
]

local_model = None

# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global local_model
    if USE_LOCAL_MODEL:
        try:
            import tensorflow as tf
            local_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✅ Local model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"⚠️  Local model not loaded: {e} — will use HF API")
    else:
        logger.info("Mode: HuggingFace Inference API")

    if not HF_TOKEN:
        logger.warning("⚠️  HF_TOKEN not set. Free tier works but may rate-limit. "
                       "Set it via: export HF_TOKEN=hf_xxxx")


# ─── Binarization Methods ─────────────────────────────────────────────────────

def binarize_via_hf(png_bytes: bytes) -> Image.Image:
    """Call HuggingFace Inference API — tries each model in HF_MODELS list."""
    headers = {"Content-Type": "image/png"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    for model_id in HF_MODELS:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        try:
            logger.info(f"→ Calling HF model: {model_id}")
            resp = req.post(url, headers=headers, data=png_bytes, timeout=45)

            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                if "image" in ct:
                    img = Image.open(io.BytesIO(resp.content)).convert("L")
                    logger.info(f"✅ HF success via {model_id}")
                    return img

            if resp.status_code == 503:
                # Model cold-starting — tell user to retry
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"The HuggingFace model '{model_id}' is warming up (cold start). "
                        "Please wait 20–30 seconds and try again."
                    )
                )

            logger.warning(f"HF {model_id} → {resp.status_code}: {resp.text[:150]}")

        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"HF {model_id} failed: {e}")

    raise RuntimeError("All HF models failed")


def binarize_local_model(img: Image.Image) -> Image.Image:
    """Use locally saved Keras model (.h5)."""
    SIZE = (256, 256)
    gray = img.convert("L").resize(SIZE)
    arr  = np.array(gray, dtype=np.float32) / 255.0
    arr  = arr[np.newaxis, ..., np.newaxis]
    pred = local_model.predict(arr)
    out  = (pred[0, ..., 0] > 0.5).astype(np.uint8) * 255
    return Image.fromarray(out, mode="L").resize(img.size, Image.NEAREST)


def binarize_otsu_fallback(img: Image.Image) -> Image.Image:
    """
    Pure-Python Otsu thresholding — used only if both local model and HF API fail.
    Produces clean binary output without any external dependencies.
    """
    gray = img.convert("L")
    arr  = np.array(gray, dtype=np.float32)

    # Compute Otsu optimal threshold
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total    = arr.size
    sum_all  = np.dot(np.arange(256), hist)
    sum_bg   = w0 = 0.0
    best_t   = best_var = 0

    for t in range(256):
        w0 += hist[t]
        w1  = total - w0
        if w0 == 0 or w1 == 0:
            continue
        sum_bg += t * hist[t]
        mu0 = sum_bg / w0
        mu1 = (sum_all - sum_bg) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, t

    binary = ((arr > best_t) * 255).astype(np.uint8)
    result = Image.fromarray(binary, mode="L")
    # Median filter to remove salt-and-pepper noise
    result = result.filter(ImageFilter.MedianFilter(size=3))
    logger.info(f"Otsu fallback used (threshold={best_t})")
    return result


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "AMDES Binarization API",
        "mode": "local_model" if local_model else "huggingface_api",
        "hf_token_set": bool(HF_TOKEN),
        "hf_models": HF_MODELS,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "local_model_loaded": local_model is not None,
        "hf_token_set": bool(HF_TOKEN),
    }

@app.post("/binarize")
async def binarize(file: UploadFile = File(...)):
    """
    Upload a document image → returns binarized grayscale PNG.

    Priority order:
      1. Local Keras model  (if USE_LOCAL_MODEL=true and model.h5 present)
      2. HuggingFace API    (ioclab/ribose-binarization)
      3. Otsu fallback      (if HF API is down or unavailable)
    """
    ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/tif"}
    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'. Use PNG, JPG, or TIFF."
        )

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    original_size = img.size
    logger.info(f"Processing: {file.filename} | {original_size} | {img.mode}")

    # 1 — Local model
    if local_model is not None:
        try:
            result = binarize_local_model(img)
            return _png_resp(result)
        except Exception as e:
            logger.error(f"Local model error: {e}")

    # 2 — HuggingFace API
    try:
        png_buf = io.BytesIO()
        img.convert("RGB").save(png_buf, format="PNG")
        result = binarize_via_hf(png_buf.getvalue())
        result = result.resize(original_size, Image.LANCZOS)
        return _png_resp(result)
    except HTTPException:
        raise   # 503 cold-start — let client see the message
    except Exception as e:
        logger.warning(f"HF failed, using Otsu: {e}")

    # 3 — Otsu fallback
    result = binarize_otsu_fallback(img)
    return _png_resp(result)


def _png_resp(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
