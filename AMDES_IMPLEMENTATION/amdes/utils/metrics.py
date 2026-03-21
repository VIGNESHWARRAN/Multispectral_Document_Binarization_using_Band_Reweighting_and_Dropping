"""utils/metrics.py"""

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils.tesseract_utils import ocr_image, is_available as tesseract_ok


def compute_psnr(original: Image.Image, binarized: Image.Image) -> float:
    orig = np.array(original.convert("L"), dtype=np.float64)
    bin_ = np.array(binarized.convert("L").resize(original.size, Image.LANCZOS), dtype=np.float64)
    try:
        return float(peak_signal_noise_ratio(orig, bin_, data_range=255))
    except Exception:
        return 0.0


def compute_ssim(original: Image.Image, binarized: Image.Image) -> float:
    orig = np.array(original.convert("L"), dtype=np.float64)
    bin_ = np.array(binarized.convert("L").resize(original.size, Image.LANCZOS), dtype=np.float64)
    try:
        score, _ = structural_similarity(orig, bin_, full=True, data_range=255)
        return float(score)
    except Exception:
        return 0.0


def compute_cnr(binarized: Image.Image) -> float:
    arr = np.array(binarized.convert("L"), dtype=np.float64)
    ink = arr[arr < 128]
    bg  = arr[arr >= 128]
    if len(ink) == 0 or len(bg) == 0:
        return 0.0
    diff = abs(ink.mean() - bg.mean())
    std  = np.sqrt((ink.std() ** 2 + bg.std() ** 2) / 2 + 1e-8)
    return float(diff / std)


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def compute_cer(ref: str, hyp: str) -> float:
    ref_ = list(ref.replace(" ", "").replace("\n", ""))
    hyp_ = list(hyp.replace(" ", "").replace("\n", ""))
    if not ref_:
        return 0.0
    return round(_edit_distance(ref_, hyp_) / len(ref_) * 100, 2)


def compute_wer(ref: str, hyp: str) -> float:
    ref_ = ref.lower().split()
    hyp_ = hyp.lower().split()
    if not ref_:
        return 0.0
    return round(_edit_distance(ref_, hyp_) / len(ref_) * 100, 2)


def compute_all_metrics(
    original: Image.Image,
    binarized: Image.Image,
    run_ocr_flag: bool = True,
    ocr_lang: str = "eng",
) -> dict:
    result = {
        "psnr": compute_psnr(original, binarized),
        "ssim": compute_ssim(original, binarized),
        "cnr":  compute_cnr(binarized),
        "ocr_available": tesseract_ok(),
    }

    if run_ocr_flag:
        ocr_orig = ocr_image(original, lang=ocr_lang)
        ocr_bin  = ocr_image(binarized, lang=ocr_lang)
        result["ocr_original"]  = ocr_orig
        result["ocr_binarized"] = ocr_bin
        result["cer"] = compute_cer(ocr_orig.get("text",""), ocr_bin.get("text",""))
        result["wer"] = compute_wer(ocr_orig.get("text",""), ocr_bin.get("text",""))
    else:
        result["ocr_original"]  = None
        result["ocr_binarized"] = None
        result["cer"] = None
        result["wer"] = None

    return result