import argparse
from pathlib import Path
import subprocess
import shutil

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--msbin_root", type=str, required=True)
    p.add_argument("--val_keys", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--fg_type", type=int, default=1)
    p.add_argument("--band_reweight", action="store_true")
    return p.parse_args()

def main():
    args = get_args()

    outdir = Path(args.outdir)
    pred_dir = outdir / f"val_pred_thr{args.thr:.2f}".replace(".", "")
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1) inference on val keys (full pages)
    cmd_infer = [
        "python", "infer_keys.py",
        "--msbin_root", args.msbin_root,
        "--split", "train",
        "--keys_txt", args.val_keys,
        "--ckpt", args.ckpt,
        "--outdir", str(pred_dir),
        "--thr", str(args.thr),
    ]
    if args.band_reweight:
        cmd_infer.append("--band_reweight")
    subprocess.check_call(cmd_infer)

    # 2) official MSBin evaluation on those predictions
    gt_dir = str(Path(args.msbin_root) / "test" / "labels")  # not used for val
    # For validation we must evaluate against TRAIN labels (val lives in train/)
    gt_dir = str(Path(args.msbin_root) / "train" / "labels")

    cmd_eval = [
        "python", str(Path(args.msbin_root).parent / "code" / "binar_eval.py"),
        gt_dir,
        str(pred_dir),
        str(pred_dir / "results.csv"),
        "-fg_type", str(args.fg_type),
    ]
    subprocess.check_call(cmd_eval)

    print("Wrote:", pred_dir / "results.csv")

if __name__ == "__main__":
    main()
