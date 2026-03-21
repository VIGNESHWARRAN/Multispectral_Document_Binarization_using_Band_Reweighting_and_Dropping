"""
utils/wavelet.py
─────────────────
Wavelet-based document enhancement — FR-3.2
Implements multi-level DWT enhancement using PyWavelets.

Wavelet families: Haar, Daubechies (db4/db6/db8), Biorthogonal (bior3.3/bior4.4)
Thresholding   : Hard, Soft, Adaptive, Automatic (BayesShrink)
"""

import numpy as np
from PIL import Image

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


# ── Wavelet name map ───────────────────────────────────────────────────────────
WAVELET_MAP = {
    "Haar":                  "haar",
    "Daubechies db4":        "db4",
    "Daubechies db6":        "db6",
    "Daubechies db8":        "db8",
    "Biorthogonal bior3.3":  "bior3.3",
    "Biorthogonal bior4.4":  "bior4.4",
}


def _bayes_threshold(coeffs: np.ndarray) -> float:
    """BayesShrink automatic threshold estimation."""
    sigma = np.median(np.abs(coeffs)) / 0.6745 + 1e-10
    s2    = np.var(coeffs)
    return float(sigma ** 2 / max(np.sqrt(max(s2 - sigma ** 2, 0)), 1e-10))


def _apply_threshold(coeffs: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "hard":
        return np.where(np.abs(coeffs) >= threshold, coeffs, 0.0)
    elif mode == "soft":
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)
    elif mode in ("adaptive", "automatic", "bayesshrink"):
        t = _bayes_threshold(coeffs)
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - t, 0.0)
    else:
        return np.where(np.abs(coeffs) >= threshold, coeffs, 0.0)


def enhance_wavelet(
    img: Image.Image,
    wavelet: str = "Haar",
    levels: int = 3,
    threshold_mode: str = "Automatic",
    threshold_value: float = 0.05,
    enhance_contrast: bool = True,
) -> dict:
    """
    Apply multi-level wavelet decomposition enhancement.

    Args:
        img              : Input PIL Image
        wavelet          : Wavelet family name (key from WAVELET_MAP)
        levels           : Decomposition levels (1–5)
        threshold_mode   : Hard / Soft / Adaptive / Automatic
        threshold_value  : Manual threshold (used when mode = Hard or Soft)
        enhance_contrast : Stretch contrast of reconstructed image

    Returns dict with:
        enhanced      : PIL Image (L mode)
        wavelet_used  : str
        levels_used   : int
        threshold_mode: str
        subbands      : list of (cH, cV, cD) images per level (for visualisation)
    """
    if not PYWT_AVAILABLE:
        # Fallback: simple unsharp mask using numpy
        return _fallback_enhance(img)

    wname = WAVELET_MAP.get(wavelet, wavelet)
    gray  = np.array(img.convert("L"), dtype=np.float64) / 255.0
    H, W  = gray.shape

    # Clamp levels to what PyWavelets supports for this image size
    max_levels = pywt.dwt_max_level(min(H, W), wname)
    levels     = min(levels, max_levels)

    # ── Forward DWT ───────────────────────────────────────────────────────
    coeffs = pywt.wavedec2(gray, wname, level=levels)
    # coeffs[0] = approx (LL), coeffs[1..] = (cH, cV, cD) tuples

    # ── Threshold detail coefficients ────────────────────────────────────
    new_coeffs = [coeffs[0]]   # keep approximation unchanged
    subbands   = []
    for detail_tuple in coeffs[1:]:
        cH, cV, cD = detail_tuple
        cH2 = _apply_threshold(cH, threshold_value, threshold_mode)
        cV2 = _apply_threshold(cV, threshold_value, threshold_mode)
        cD2 = _apply_threshold(cD, threshold_value, threshold_mode)
        new_coeffs.append((cH2, cV2, cD2))

        # Collect subband visualisations
        def _vis(c):
            cc = np.abs(c)
            mn, mx = cc.min(), cc.max()
            if mx > mn:
                cc = (cc - mn) / (mx - mn) * 255
            return Image.fromarray(cc.astype(np.uint8), mode="L")
        subbands.append((_vis(cH), _vis(cV), _vis(cD)))

    # ── Inverse DWT ───────────────────────────────────────────────────────
    reconstructed = pywt.waverec2(new_coeffs, wname)
    reconstructed = reconstructed[:H, :W]   # crop to original size (padding artefacts)

    # ── Post-processing ───────────────────────────────────────────────────
    rec = np.clip(reconstructed, 0.0, 1.0)

    if enhance_contrast:
        # Stretch contrast to use full [0,1] range
        lo, hi = rec.min(), rec.max()
        if hi > lo:
            rec = (rec - lo) / (hi - lo)

    enhanced = Image.fromarray((rec * 255).astype(np.uint8), mode="L")

    return {
        "enhanced":       enhanced,
        "wavelet_used":   wname,
        "levels_used":    levels,
        "threshold_mode": threshold_mode,
        "subbands":       subbands,
    }


def _fallback_enhance(img: Image.Image) -> dict:
    """Unsharp mask fallback when PyWavelets is unavailable."""
    import cv2
    arr    = np.array(img.convert("L"))
    blur   = cv2.GaussianBlur(arr, (0, 0), 3)
    sharp  = cv2.addWeighted(arr, 1.5, blur, -0.5, 0)
    result = Image.fromarray(sharp, mode="L")
    return {
        "enhanced":       result,
        "wavelet_used":   "fallback (unsharp mask)",
        "levels_used":    0,
        "threshold_mode": "N/A",
        "subbands":       [],
    }