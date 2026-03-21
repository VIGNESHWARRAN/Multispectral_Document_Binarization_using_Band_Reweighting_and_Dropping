"""
backend/models/huggingface.py
──────────────────────────────
Calls the HuggingFace Inference API for binarization.

Models are tried in order from config.settings.HF_MODELS.
Raises HTTPException(503) on cold-start so the frontend can prompt retry.
Raises RuntimeError if all models fail.
"""

import io
import logging

import requests
from fastapi import HTTPException
from PIL import Image

from backend.models.base import BinarizationModel
from config.settings import HF_TOKEN, HF_MODELS

logger = logging.getLogger("AMDES.HuggingFace")

_TIMEOUT_SECS = 45


class HuggingFaceModel(BinarizationModel):
    """Wraps HuggingFace Inference API — no local weights required."""

    def predict(self, img: Image.Image) -> Image.Image:
        # Encode image as PNG bytes
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        headers = {"Content-Type": "image/png"}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        for model_id in HF_MODELS:
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            try:
                logger.info(f"→ HF Inference API: {model_id}")
                resp = requests.post(url, headers=headers, data=png_bytes,
                                     timeout=_TIMEOUT_SECS)

                if resp.status_code == 200:
                    ct = resp.headers.get("content-type", "")
                    if "image" in ct:
                        result = Image.open(io.BytesIO(resp.content)).convert("L")
                        logger.info(f"✅ HF success: {model_id}")
                        return result
                    logger.warning(f"HF {model_id} returned non-image content-type: {ct}")

                elif resp.status_code == 503:
                    # Cold start — tell the user to retry after a few seconds
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"HuggingFace model '{model_id}' is warming up. "
                            "Please wait 20–30 seconds and try again."
                        ),
                    )

                else:
                    logger.warning(
                        f"HF {model_id} → HTTP {resp.status_code}: {resp.text[:200]}"
                    )

            except HTTPException:
                raise   # 503 bubbles up to caller
            except Exception as e:
                logger.warning(f"HF {model_id} request error: {e}")

        raise RuntimeError("All HuggingFace models failed. Falling back to Otsu.")

    @property
    def name(self) -> str:
        return f"HuggingFaceModel(models={HF_MODELS})"
