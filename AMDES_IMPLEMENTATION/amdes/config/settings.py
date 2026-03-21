"""
config/settings.py
──────────────────
Single source of truth for all AMDES configuration.

Load order:
  1. .env file (via python-dotenv)
  2. Environment variables
  3. Hardcoded defaults

Both the FastAPI backend and the Streamlit frontend import from here.
"""

import os
from dotenv import load_dotenv

# Load .env from repo root (works whether you run from root or a subdir)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root, ".env"), override=False)


# ─── Auth0 ────────────────────────────────────────────────────────────────────
AUTH0_DOMAIN        = os.getenv("AUTH0_DOMAIN",        "YOUR_TENANT.us.auth0.com")
AUTH0_CLIENT_ID     = os.getenv("AUTH0_CLIENT_ID",     "YOUR_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
AUTH0_CALLBACK      = os.getenv("AUTH0_CALLBACK",      "http://localhost:8501")

# ─── Backend / Frontend ────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ─── HuggingFace ──────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")

# HF model IDs tried in order — first success wins
HF_MODELS = [
    "ioclab/ribose-binarization",        # document binarization — best fit
    "saifhassan/document-binarization",  # backup
]

# ─── Local Model ──────────────────────────────────────────────────────────────
USE_LOCAL_MODEL: bool = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
MODEL_PATH: str       = os.getenv("MODEL_PATH", "model.h5")
MODEL_TYPE: str       = os.getenv("MODEL_TYPE", "keras")   # "keras" | "onnx"

# ─── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_MB: int = 200
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/tif",
}
