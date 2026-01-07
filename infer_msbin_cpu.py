# infer_keys.py
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

from msbin_dataset import group_books, _read_png_gray
from unet_small import UNetSmall

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--msbin_root", type=str, required=True)
    p.add_argument("--split", type=str, default="train")  # val lives inside train/
    p.add_argument("--keys_txt", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--white_only", action="store_true")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--band_reweight", action="store_true")
    return p.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    msbin_root = Path(args.msbin_root)
    image_dir = msbin_root / args.split / "images"
    pages, all_keys = group_books(image_dir)

    wanted = [k.strip() for k in Path(args.keys_txt).read_text().splitlines() if k.strip()]
    wanted = [k for k in wanted if k in pages]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    in_ch = 1 if args.white_only else 12
    model = UNetSmall(in_ch=in_ch, base=16, band_reweight=args.band_reweight, band_drop_p=0.0).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for key in tqdm(wanted, desc=f"Infer keys ({args.split})"):
        wl_ids = [0] if args.white_only else list(range(12))
        chans = []
        for wl in wl_ids:
            img = _read_png_gray(pages[key][wl]).astype(np.float32) / 255.0
            chans.append(img)
        x = np.stack(chans, axis=0)  # C,H,W

        C, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        x_pad = np.pad(x, ((0,0),(0,pad_h),(0,pad_w)), mode="reflect")

        xt = torch.from_numpy(x_pad).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
        probs = probs[:H, :W]

        pred = (probs < args.thr).astype(np.uint8) * 255
        cv2.imwrite(str(outdir / f"{key}.png"), pred)

if __name__ == "__main__":
    main()
