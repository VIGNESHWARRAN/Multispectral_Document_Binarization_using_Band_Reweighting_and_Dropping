"""
backend/models/otsu_fallback.py
────────────────────────────────
Pure-Python Otsu thresholding — zero external dependencies.

Used as the last resort when both the local model and HF API are unavailable.
Produces clean binary output via the classic Otsu optimal threshold algorithm,
followed by a median filter to suppress salt-and-pepper noise.
"""

import logging

import numpy as np
from PIL import Image, ImageFilter

from backend.models.base import BinarizationModel

logger = logging.getLogger("AMDES.OtsuFallback")


class OtsuFallback(BinarizationModel):
    """
    Classic Otsu binarization with median noise removal.

    Characteristics:
      - No model weights, no network calls — always available
      - Works well for documents with clear ink/background contrast
      - Degrades on heavily degraded or low-contrast images (use CNN for those)
    """

    def predict(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        arr  = np.array(gray, dtype=np.float32)

        threshold = self._otsu_threshold(arr)
        binary    = ((arr > threshold) * 255).astype(np.uint8)

        result = Image.fromarray(binary, mode="L")
        result = result.filter(ImageFilter.MedianFilter(size=3))

        logger.info(f"Otsu fallback used (threshold={threshold})")
        return result

    @staticmethod
    def _otsu_threshold(arr: np.ndarray) -> int:
        """Compute Otsu's optimal binarization threshold."""
        hist, _  = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        total    = arr.size
        sum_all  = np.dot(np.arange(256, dtype=np.float64), hist)
        sum_bg   = 0.0
        w0       = 0.0
        best_t   = 0
        best_var = 0.0

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
                best_var = var
                best_t   = t

        return best_t

    @property
    def name(self) -> str:
        return "OtsuFallback"
