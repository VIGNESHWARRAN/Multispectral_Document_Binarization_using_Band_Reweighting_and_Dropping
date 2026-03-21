"""
utils/tesseract_utils.py
─────────────────────────
Tesseract path auto-detection for Windows + Linux + Mac.
Import this instead of calling pytesseract directly.
"""

import os
import sys
import shutil
import pytesseract
from PIL import Image


# ── Auto-detect Tesseract on Windows ─────────────────────────────────────────
_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
    r"C:\tools\tesseract\tesseract.exe",
]

def _configure_tesseract():
    """Set pytesseract.tesseract_cmd to the correct path."""
    # Already on PATH
    if shutil.which("tesseract"):
        return True, None

    if sys.platform == "win32":
        for path in _WINDOWS_PATHS:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True, None
        return False, (
            "Tesseract not found. Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and add to PATH, or place at: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )

    if sys.platform == "darwin":
        for path in ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True, None
        return False, "Install Tesseract: brew install tesseract"

    return False, "Install Tesseract: sudo apt-get install tesseract-ocr"


_TESSERACT_OK, _TESSERACT_ERR = _configure_tesseract()


def is_available() -> bool:
    return _TESSERACT_OK


def get_error() -> str:
    return _TESSERACT_ERR or ""


def ocr_image(img: Image.Image, lang: str = "eng") -> dict:
    """
    Run OCR. Returns dict with text, word_count, char_count, avg_conf, success.
    """
    if not _TESSERACT_OK:
        return {
            "text": "", "word_count": 0, "char_count": 0,
            "avg_conf": 0.0, "success": False,
            "error": _TESSERACT_ERR,
        }
    try:
        import numpy as np
        data = pytesseract.image_to_data(
            img.convert("RGB"), lang=lang,
            output_type=pytesseract.Output.DICT,
        )
        words = [w for w in data["text"] if w.strip()]
        confs = [int(c) for c, w in zip(data["conf"], data["text"])
                 if w.strip() and int(c) >= 0]
        avg_conf  = float(np.mean(confs)) if confs else 0.0
        full_text = pytesseract.image_to_string(img.convert("RGB"), lang=lang).strip()
        return {
            "text":       full_text,
            "word_count": len(words),
            "char_count": len(full_text.replace(" ", "").replace("\n", "")),
            "avg_conf":   round(avg_conf, 1),
            "success":    True,
        }
    except Exception as e:
        return {
            "text": "", "word_count": 0, "char_count": 0,
            "avg_conf": 0.0, "success": False, "error": str(e),
        }