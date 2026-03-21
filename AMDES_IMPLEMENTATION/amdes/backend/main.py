"""
backend/main.py
────────────────
FastAPI application entry point.

Run:
    uvicorn backend.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import load_models
from backend.routes.health import router as health_router
from backend.routes.binarize import router as binarize_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("AMDES")

app = FastAPI(
    title="AMDES Binarization API",
    description="Advanced Multispectral Document Enhancement System — backend",
    version="1.0.0",
)

# Allow all origins (tighten this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(binarize_router)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    logger.info("AMDES API starting up…")
    load_models()
    logger.info("AMDES API ready.")
