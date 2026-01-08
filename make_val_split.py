# make_val_split.py
import argparse
from pathlib import Path
import random
from msbin_dataset import group_books

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--msbin_root", type=str, required=True)
    p.add_argument("--val_n", type=int, default=10)      # 10 pages is fine
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="splits")
    return p.parse_args()

def main():
    args = get_args()
    msbin_root = Path(args.msbin_root)

    image_dir = msbin_root / "train" / "images"
    _, keys = group_books(image_dir)

    rng = random.Random(args.seed)
    rng.shuffle(keys)

    val_keys = sorted(keys[:args.val_n])
    train_keys = sorted(keys[args.val_n:])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "val_keys.txt").write_text("\n".join(val_keys) + "\n")
    (outdir / "train_keys.txt").write_text("\n".join(train_keys) + "\n")

    print(f"Total keys: {len(keys)}")
    print(f"Train keys: {len(train_keys)} -> {outdir/'train_keys.txt'}")
    print(f"Val keys:   {len(val_keys)} -> {outdir/'val_keys.txt'}")

if __name__ == "__main__":
    main()
