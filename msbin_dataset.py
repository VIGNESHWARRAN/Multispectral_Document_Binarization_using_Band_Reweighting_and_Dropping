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

def _read_png_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"Missing file: {path}")
    return img

def pad_to_size(img: np.ndarray, size: int) -> np.ndarray:
    if img.ndim == 2:
        h, w = img.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h == 0 and pad_w == 0:
            return img
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    if img.ndim == 3:
        h, w, c = img.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h == 0 and pad_w == 0:
            return img
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

    raise ValueError("pad_to_size expects 2D or 3D array")

def group_books(image_dir: Path):
    files = sorted(image_dir.glob("*.png"))
    books = {}
    for p in files:
        m = re.match(r"^(.+)_([0-9]+)$", p.stem)
        if not m:
            continue
        key = m.group(1)       # BookId_PageId
        wl = int(m.group(2))   # 0..11
        books.setdefault(key, {})[wl] = p

    keys = [k for k, d in books.items() if all(i in d for i in range(12))]
    keys.sort()
    return books, keys


class MSBinDibcoPatchDataset(Dataset):
    """
    MSBIN patch dataset using OFFICIAL color-coded labels/ (not dibco_labels).
    UR is treated as background, matching evaluation definition. [file:594]
    Supports filtering by page keys via keys_txt.
    """
    def __init__(
        self,
        msbin_root: str,
        split="train",
        fg_type=1,
        patch_size=256,
        stride=256,
        use_white_only=False,
        min_fg_frac=0.002,
        max_patches_per_page=None,
        keys_txt: str | None = None,
    ):
        self.root = Path(msbin_root)
        self.split = split
        self.fg_type = int(fg_type)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.use_white_only = bool(use_white_only)
        self.min_fg_frac = float(min_fg_frac)
        self.max_patches_per_page = max_patches_per_page

        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"   # <-- official labels/

        self.books, all_keys = group_books(self.image_dir)

        if keys_txt is not None:
            wanted = {k.strip() for k in Path(keys_txt).read_text().splitlines() if k.strip()}
            self.keys = [k for k in all_keys if k in wanted]
        else:
            self.keys = all_keys

        self.index = []
        for key in self.keys:
            img0 = _read_png_gray(self.books[key][0])
            H, W = img0.shape
            if H < patch_size or W < patch_size:
                continue

            gt = _read_png_bgr(self.label_dir / f"{key}.png")

            page_patches = []
            for yy in range(0, H - patch_size + 1, stride):
                for xx in range(0, W - patch_size + 1, stride):
                    gt_patch = gt[yy:yy+patch_size, xx:xx+patch_size]
                    fg = self._gt_to_fgmask(gt_patch)
                    if fg.mean() < self.min_fg_frac:
                        continue
                    page_patches.append((key, yy, xx))

            if self.max_patches_per_page is not None and len(page_patches) > self.max_patches_per_page:
                step = max(1, len(page_patches) // self.max_patches_per_page)
                page_patches = page_patches[::step][:self.max_patches_per_page]

            self.index.extend(page_patches)

        if len(self.index) == 0:
            raise RuntimeError(
                f"Dataset empty. Check paths:\n"
                f" images: {self.image_dir}\n"
                f" labels: {self.label_dir}\n"
                f" keys_txt: {keys_txt}\n"
                f"Try lowering --min_fg_frac.\n"
            )

    def _gt_to_fgmask(self, gt_bgr: np.ndarray) -> np.ndarray:
        gt_bgr = pad_to_size(gt_bgr, self.patch_size)

        is_fg1 = (gt_bgr[:, :, 0] == 255) & (gt_bgr[:, :, 1] == 255) & (gt_bgr[:, :, 2] == 255)  # white
        is_fg2 = (gt_bgr[:, :, 0] == 122) & (gt_bgr[:, :, 1] == 122) & (gt_bgr[:, :, 2] == 122)  # gray
        is_ur  = (gt_bgr[:, :, 0] == 255) & (gt_bgr[:, :, 1] == 0)   & (gt_bgr[:, :, 2] == 0)    # BGR blue

        y = is_fg1.astype(np.float32) if self.fg_type == 1 else is_fg2.astype(np.float32)
        y[is_ur] = 0.0  # UR excluded -> background [file:594]
        return y

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, y0, x0 = self.index[idx]

        wl_ids = [0] if self.use_white_only else list(range(12))
        chans = []
        for wl in wl_ids:
            img = _read_png_gray(self.books[key][wl])
            crop = img[y0:y0+self.patch_size, x0:x0+self.patch_size]
            crop = pad_to_size(crop, self.patch_size).astype(np.float32) / 255.0
            chans.append(crop)
        x_img = np.stack(chans, axis=0)  # C,H,W

        gt = _read_png_bgr(self.label_dir / f"{key}.png")
        gt_crop = gt[y0:y0+self.patch_size, x0:x0+self.patch_size]
        y_mask = self._gt_to_fgmask(gt_crop)

        return torch.from_numpy(x_img), torch.from_numpy(y_mask).unsqueeze(0)


group_pages = group_books
