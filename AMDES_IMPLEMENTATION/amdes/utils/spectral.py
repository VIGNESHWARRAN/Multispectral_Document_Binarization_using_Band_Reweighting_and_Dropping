"""utils/spectral.py"""

import numpy as np
from PIL import Image


def image_to_bands(img: Image.Image) -> np.ndarray:
    """PIL Image → (H, W, 3) float32, normalised [0,1]. Resized to max 256px to keep memory sane."""
    # Downsample for processing — spectral unmixing works on pixel statistics, not spatial detail
    MAX_DIM = 256
    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def bands_to_image(bands: np.ndarray, target_size=None) -> Image.Image:
    if bands.ndim == 3:
        gray = bands.mean(axis=2)
    else:
        gray = bands
    gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
    img  = Image.fromarray(gray, mode="L")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def flatten_bands(arr: np.ndarray) -> np.ndarray:
    H, W, B = arr.shape
    return arr.reshape(-1, B)   # (N, B) where N = H*W, small because of downsample


def unflatten(pixels: np.ndarray, H: int, W: int) -> np.ndarray:
    return pixels.reshape(H, W, -1)


# ── VCA ───────────────────────────────────────────────────────────────────────

def vca(pixels: np.ndarray, n_endmembers: int = 3, seed: int = 42) -> np.ndarray:
    """Vertex Component Analysis — works on (N, B) pixels."""
    rng = np.random.default_rng(seed)
    N, B = pixels.shape
    n_endmembers = min(n_endmembers, B, N)

    mean = pixels.mean(axis=0)
    Xc   = pixels - mean

    # SVD on (B, B) covariance — safe because B=3 for RGB
    _, _, Vt = np.linalg.svd(Xc.T @ Xc / N, full_matrices=False)
    Xp = Xc @ Vt[:n_endmembers].T   # (N, n_endmembers)
    Xa = np.hstack([Xp, np.ones((N, 1), dtype=np.float32)])  # (N, n_endmembers+1)

    endmember_idx = []
    u = rng.standard_normal(n_endmembers + 1).astype(np.float32)
    u /= np.linalg.norm(u)

    for _ in range(n_endmembers):
        proj = Xa @ u
        idx  = int(np.argmax(np.abs(proj)))
        endmember_idx.append(idx)

        v    = Xa[idx].copy()
        v   /= (np.linalg.norm(v) + 1e-10)
        # Project out v from u
        u    = u - np.dot(u, v) * v
        norm = np.linalg.norm(u)
        u    = u / norm if norm > 1e-10 else rng.standard_normal(n_endmembers + 1).astype(np.float32)
        u   /= np.linalg.norm(u)

    return pixels[endmember_idx]


# ── N-FINDR ────────────────────────────────────────────────────────────────────

def nfindr(pixels: np.ndarray, n_endmembers: int = 3,
           max_iter: int = 200, seed: int = 42) -> np.ndarray:
    """N-FINDR — works on downsampled (N, B) pixels."""
    rng = np.random.default_rng(seed)
    N, B = pixels.shape
    n_endmembers = min(n_endmembers, B, N)

    mean = pixels.mean(axis=0)
    Xc   = pixels - mean
    _, _, Vt = np.linalg.svd(Xc.T @ Xc / N, full_matrices=False)
    Xr = (Xc @ Vt[:n_endmembers - 1].T).astype(np.float32)   # (N, K-1)

    idx = rng.choice(N, n_endmembers, replace=False)
    E   = Xr[idx].copy()

    def _vol(E):
        M      = np.ones((n_endmembers, n_endmembers), dtype=np.float32)
        M[:, :-1] = E
        return abs(float(np.linalg.det(M)))

    vol = _vol(E)
    for _ in range(max_iter):
        improved = False
        for i in range(n_endmembers):
            sample = rng.choice(N, min(200, N), replace=False)
            for j in sample:
                E_new    = E.copy()
                E_new[i] = Xr[j]
                v_new    = _vol(E_new)
                if v_new > vol + 1e-10:
                    E[i] = Xr[j]; idx[i] = j; vol = v_new; improved = True; break
        if not improved:
            break

    return pixels[idx]


# ── NNLS ───────────────────────────────────────────────────────────────────────

def nnls_unmix(pixels: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """NNLS abundance estimation — batched for speed."""
    from scipy.optimize import nnls as scipy_nnls
    N, B = pixels.shape
    K    = endmembers.shape[0]
    A    = endmembers.T.astype(np.float64)
    out  = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        x, _ = scipy_nnls(A, pixels[i].astype(np.float64))
        s = x.sum()
        out[i] = (x / s if s > 1e-10 else x).astype(np.float32)
    return out


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_spectral_unmixing(
    img: Image.Image,
    algorithm: str = "VCA",
    n_endmembers: int = 3,
) -> dict:
    """
    Run spectral unmixing. Image is downsampled to 256px max for processing.
    Abundance maps are upsampled back to original size for display.
    """
    original_size = img.size
    n_endmembers = min(n_endmembers, 3)  # RGB has 3 bands max
    bands  = image_to_bands(img)          # downsampled, (h, w, 3)
    H, W, B = bands.shape
    pixels = flatten_bands(bands)         # (N, 3) — N is at most 256*256 = 65536

    algo = algorithm.upper().replace("-","").replace("_","").replace(" ","")
    if "NFINDR" in algo or "FINDR" in algo:
        endmembers = nfindr(pixels, n_endmembers)
        algo_name  = "N-FINDR"
    else:
        endmembers = vca(pixels, n_endmembers)
        algo_name  = "VCA" if "VCA" in algo else "NNLS (VCA endmembers)"

    abundances = nnls_unmix(pixels, endmembers)   # (N, K)
    abu_map    = unflatten(abundances, H, W)       # (H, W, K)

    # Per-endmember abundance images, upsampled to original size
    abu_images = []
    for k in range(n_endmembers):
        layer = (abu_map[:, :, k] * 255).clip(0, 255).astype(np.uint8)
        abu_img = Image.fromarray(layer, mode="L").resize(original_size, Image.LANCZOS)
        abu_images.append(abu_img)

    ink_idx   = int(np.argmin(endmembers.mean(axis=1)))
    ink_layer = abu_images[ink_idx]

    return {
        "endmembers":       endmembers,
        "abundances":       abu_map,
        "abundance_images": abu_images,
        "ink_layer":        ink_layer,
        "algorithm":        algo_name,
        "n_endmembers":     n_endmembers,
        "ink_idx":          ink_idx,
    }