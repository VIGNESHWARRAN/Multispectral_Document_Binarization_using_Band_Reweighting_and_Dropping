"""
backend/pipeline.py
────────────────────
Manages model loading and the priority fallback chain.

Priority order
──────────────
1. LocalModel    — your trained weights  (enabled via USE_LOCAL_MODEL=true)
2. HuggingFaceModel — HF Inference API
3. OtsuFallback  — always available, no dependencies

To add a new model:
  1. Implement BinarizationModel in backend/models/
  2. Import it here
  3. Add it to the chain in _build_chain()
"""

import logging
from PIL import Image
from fastapi import HTTPException

from config.settings import USE_LOCAL_MODEL
from backend.models.huggingface  import HuggingFaceModel
from backend.models.otsu_fallback import OtsuFallback

logger = logging.getLogger("AMDES.Pipeline")

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_local_model:   object | None = None
_hf_model:      HuggingFaceModel | None = None
_otsu_fallback: OtsuFallback | None = None


def _get_local_model():
    global _local_model
    if _local_model is None and USE_LOCAL_MODEL:
        from backend.models.local_model import LocalModel
        _local_model = LocalModel()
    return _local_model


def _get_hf_model() -> HuggingFaceModel:
    global _hf_model
    if _hf_model is None:
        _hf_model = HuggingFaceModel()
    return _hf_model


def _get_otsu() -> OtsuFallback:
    global _otsu_fallback
    if _otsu_fallback is None:
        _otsu_fallback = OtsuFallback()
    return _otsu_fallback


def load_models():
    """
    Called at startup — eager-loads the local model if configured.
    HF and Otsu are lazy-loaded on first use (they need no startup work).
    """
    if USE_LOCAL_MODEL:
        try:
            _get_local_model()
        except Exception as e:
            logger.warning(f"⚠️  Local model failed to load: {e}. HF API will be used.")
    else:
        logger.info("Local model disabled (USE_LOCAL_MODEL=false). Using HF API + Otsu fallback.")


def run_binarization(img: Image.Image) -> Image.Image:
    """
    Run binarization through the priority chain.

    Args:
        img: Input PIL Image (any mode)

    Returns:
        Binarized PIL Image ("L" mode)

    Raises:
        HTTPException(503) if HF model is cold-starting (let client retry)
    """
    original_size = img.size

    # ── 1. Local model ────────────────────────────────────────────────────────
    local = _get_local_model()
    if local is not None:
        try:
            logger.info(f"Running {local.name}")
            result = local.predict(img)
            return result.resize(original_size, Image.LANCZOS)
        except Exception as e:
            logger.error(f"Local model error: {e}. Falling back to HF API.")

    # ── 2. HuggingFace API ────────────────────────────────────────────────────
    try:
        hf = _get_hf_model()
        logger.info(f"Running {hf.name}")
        result = hf.predict(img)
        return result.resize(original_size, Image.LANCZOS)
    except HTTPException:
        raise   # 503 cold-start — propagate to client
    except Exception as e:
        logger.warning(f"HF model error: {e}. Falling back to Otsu.")

    # ── 3. Otsu fallback ──────────────────────────────────────────────────────
    otsu = _get_otsu()
    logger.info(f"Running {otsu.name}")
    return otsu.predict(img)
