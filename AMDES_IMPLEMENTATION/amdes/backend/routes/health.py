"""
backend/routes/health.py
─────────────────────────
/health and / endpoints.
"""

from fastapi import APIRouter
from config.settings import USE_LOCAL_MODEL, HF_TOKEN, HF_MODELS
from backend.pipeline import _get_local_model   # private but safe to introspect here

router = APIRouter()


@router.get("/")
def root():
    local = _get_local_model()
    return {
        "service":            "AMDES Binarization API",
        "version":            "1.0.0",
        "local_model_active": local is not None,
        "use_local_model":    USE_LOCAL_MODEL,
        "hf_token_set":       bool(HF_TOKEN),
        "hf_models":          HF_MODELS,
    }


@router.get("/health")
def health():
    local = _get_local_model()
    return {
        "status":             "ok",
        "local_model_loaded": local is not None,
        "hf_token_set":       bool(HF_TOKEN),
    }
