# train_unet.py
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from msbin_dataset import MSBinDibcoPatchDataset
from unet_small import UNetSmall

def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (1.0 - (num / den)).mean()

def save_ckpt(path, model, opt, epoch, best_score, extra=None):
    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def load_ckpt(path, model, opt):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if "torch_rng" in ckpt: torch.set_rng_state(ckpt["torch_rng"])
    if "py_rng" in ckpt: random.setstate(ckpt["py_rng"])
    if "np_rng" in ckpt: np.random.set_state(ckpt["np_rng"])
    start_epoch = ckpt.get("epoch", 1)
    best_score = ckpt.get("best_score", -1e18)
    return start_epoch, best_score

def log_band_weights_csv(out_csv: Path, epoch: int, weights_1d: np.ndarray):
    header = ["epoch"] + [f"band_{i}" for i in range(len(weights_1d))]
    row = [epoch] + [float(x) for x in weights_1d.tolist()]
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--msbin_root", type=str, required=True)
    p.add_argument("--fg_type", type=int, default=1)

    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--min_fg_frac", type=float, default=0.002)
    p.add_argument("--max_patches_per_page", type=int, default=None)

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--white_only", action="store_true")
    p.add_argument("--outdir", type=str, default="runs_cpu/unet")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--band_reweight", action="store_true")
    p.add_argument("--band_drop_p", type=float, default=0.0)

    p.add_argument("--pos_weight", type=float, default=15.0)  # important
    p.add_argument("--bce_w", type=float, default=0.7)
    p.add_argument("--dice_w", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)

    return p.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    last_path = outdir / "last.pt"
    best_path = outdir / "best.pt"
    band_csv = outdir / "band_weights.csv"

    train_ds = MSBinDibcoPatchDataset(
        msbin_root=args.msbin_root,
        split="train",
        fg_type=args.fg_type,
        patch_size=args.patch,
        stride=args.stride,
        use_white_only=args.white_only,
        min_fg_frac=args.min_fg_frac,
        max_patches_per_page=args.max_patches_per_page,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    in_ch = 1 if args.white_only else 12
    model = UNetSmall(
        in_ch=in_ch,
        base=16,
        band_reweight=args.band_reweight,
        band_drop_p=args.band_drop_p,
    ).to(device)

    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))


    start_epoch = 1
    best_score = -1e18  # higher is better; we’ll use -train_loss as score for now

    if args.resume and last_path.exists():
        start_epoch, best_score = load_ckpt(last_path, model, opt)
        print(f"Resuming from {last_path} at epoch {start_epoch}, best_score={best_score:.4f}")
    else:
        print("Starting fresh training.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):

                logits = model(x)
                loss = args.bce_w * bce(logits, y) + args.dice_w * dice_loss_with_logits(logits, y)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            running += loss.item()

        avg = running / max(1, len(train_loader))
        score = -avg

        save_ckpt(last_path, model, opt, epoch + 1, best_score)

        if score > best_score:
            best_score = score
            save_ckpt(best_path, model, opt, epoch + 1, best_score)

        if args.band_reweight and getattr(model, "last_band_weights", None) is not None:
            w_mean = model.last_band_weights.mean(dim=0).cpu().numpy()
            log_band_weights_csv(band_csv, epoch, w_mean)

        print(f"Epoch {epoch}: train_loss={avg:.4f} best_score={best_score:.4f} saved={outdir}")

if __name__ == "__main__":
    main()
