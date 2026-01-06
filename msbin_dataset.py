from pathlib import Path
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def _read_png_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing file: {path}")
    return img

def pad_to_size(img2d: np.ndarray, size: int) -> np.ndarray:
    h, w = img2d.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return img2d
    # pad bottom and right (constant 0 is OK for inputs; labels too)
    return np.pad(img2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

def group_books(image_dir: Path):
    # Your naming: BT5_0.png ... BT5_11.png  (bookId_wavelengthId)
    files = sorted(image_dir.glob("*.png"))
    books = {}
    for p in files:
        m = re.match(r"^(.+)_([0-9]+)$", p.stem)  # key, wl
        if not m:
            continue
        key = m.group(1)     # e.g., "BT5"
        wl = int(m.group(2)) # 0..11
        books.setdefault(key, {})[wl] = p

    keys = []
    for k, d in books.items():
        if all(i in d for i in range(12)):
            keys.append(k)
    keys.sort()
    return books, keys

class MSBinDibcoPatchDataset(Dataset):
    """
    Uses MSBin's pre-generated DIBCO-style binary labels:
      data/<split>/dibco_labels/fg_1/<key>.png  (or fg_2)
    """
    def __init__(self, msbin_root: str, split="train", fg_type=1,
                 patch_size=128, stride=128, use_white_only=False):
        self.root = Path(msbin_root)
        self.split = split
        self.fg_type = fg_type
        self.patch_size = patch_size
        self.stride = stride
        self.use_white_only = use_white_only

        self.image_dir = self.root / split / "images"
        fg_folder = "fg_1" if fg_type == 1 else "fg_2"
        self.label_dir = self.root / split / "dibco_labels" / fg_folder

        self.books, self.keys = group_books(self.image_dir)

        # Build patch index
        self.index = []
        for key in self.keys:
            img0 = _read_png_gray(self.books[key][0])
            H, W = img0.shape
            if H < patch_size or W < patch_size:
                continue

            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.index.append((key, y, x))

        if len(self.index) == 0:
            raise RuntimeError(
                f"Dataset empty. Check paths:\n"
                f" images: {self.image_dir}\n"
                f" labels: {self.label_dir}\n"
                f"Example images: {list(self.image_dir.glob('*.png'))[:5]}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, y, x = self.index[idx]

        wl_ids = [0] if self.use_white_only else list(range(12))
        chans = []
        for wl in wl_ids:
            # --- image ---
            img = _read_png_gray(self.books[key][wl])
            crop = img[y:y+self.patch_size, x:x+self.patch_size]
            crop = pad_to_size(crop, self.patch_size).astype(np.float32) / 255.0
            chans.append(crop)

        x_img = np.stack(chans, axis=0)  # C,H,W

        # DIBCO label is already binary (0/255). Convert to 0/1.
        lbl_path = self.label_dir / f"{key}.png"
        lbl = _read_png_gray(lbl_path)
        y_crop = lbl[y:y+self.patch_size, x:x+self.patch_size]
        y_crop = pad_to_size(y_crop, self.patch_size)
        y_mask = (y_crop > 127).astype(np.float32)


        return torch.from_numpy(x_img), torch.from_numpy(y_mask).unsqueeze(0)
