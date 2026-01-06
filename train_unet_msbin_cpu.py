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

def save_ckpt(path, model, opt, epoch, best_loss):
    ckpt = {
        "epoch": epoch,  # next epoch to run
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        # RNG states for reproducibility when resuming
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
    }
    torch.save(ckpt, path)

def load_ckpt(path, model, opt):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    # restore RNG
    if "torch_rng" in ckpt: torch.set_rng_state(ckpt["torch_rng"])
    if "py_rng" in ckpt: random.setstate(ckpt["py_rng"])
    if "np_rng" in ckpt: np.random.set_state(ckpt["np_rng"])
    start_epoch = ckpt.get("epoch", 1)
    best_loss = ckpt.get("best_loss", 1e18)
    return start_epoch, best_loss
def log_band_weights_csv(out_csv: Path, epoch: int, weights_1d: np.ndarray):
    # weights_1d shape: (C,)
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
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--white_only", action="store_true")
    p.add_argument("--outdir", type=str, default="runs_cpu/unet")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band_reweight", action="store_true")
    p.add_argument("--band_drop_p", type=float, default=0.0)

    return p.parse_args()

def main():
    args = get_args()
    device = "cpu"
    

    # seeds (helps reproducibility; resume restores RNG too)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    last_path = outdir / "last.pt"
    best_path = outdir / "best.pt"
    band_csv = outdir / "band_weights.csv"


    train_ds = MSBinDibcoPatchDataset(
        msbin_root=args.msbin_root, split="train", fg_type=args.fg_type,
        patch_size=args.patch, stride=args.stride, use_white_only=args.white_only
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=False
    )

    in_ch = 1 if args.white_only else 12
    model = UNetSmall(in_ch=in_ch, base=16,band_reweight=args.band_reweight,band_drop_p=args.band_drop_p).to(device)

    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_loss = 1e18

    # ---- RESUME ----
    if args.resume and last_path.exists():
        start_epoch, best_loss = load_ckpt(last_path, model, opt)
        print(f"Resuming from {last_path} at epoch {start_epoch}, best_loss={best_loss:.4f}")
    else:
        print("Starting fresh training.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()

        avg = running / max(1, len(train_loader))

        # Save "last" every epoch so you can resume
        save_ckpt(last_path, model, opt, epoch + 1, best_loss)

        # Save best checkpoint separately
        if avg < best_loss:
            best_loss = avg
            save_ckpt(best_path, model, opt, epoch + 1, best_loss)
        # ---- Band weight logging (epoch-level) ----
        if args.band_reweight and getattr(model, "last_band_weights", None) is not None:
            w_mean = model.last_band_weights.mean(dim=0).cpu().numpy()  # (C,)
            log_band_weights_csv(band_csv, epoch, w_mean)
            w_str = ", ".join([f"{v:.3f}" for v in w_mean.tolist()])
            print(f"[Band weights mean @epoch {epoch}] {w_str}")
        else:
            if args.band_reweight:
                print("Warning: band_reweight enabled but model.last_band_weights is None. Update unet_small.py to store last_band_weights.")


        print(f"Epoch {epoch}: train_loss={avg:.4f} best_loss={best_loss:.4f} saved={outdir}")

if __name__ == "__main__":
    main()
