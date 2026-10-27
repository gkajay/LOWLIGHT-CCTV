"""Train the custom CNN from scratch on the LOL training pairs.

    cd backend
    python -m train.train_cnn --data ../LOLdataset

Loss = L1 + 0.2 * (1 - SSIM). Checkpoints are chosen on images held out from
our485, so eval15 stays a clean test set.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app import config
from app.models.cnn import LowLightCNN

from .common import History, PairDataset, evaluate, pick_device, split_train_val
from .losses import SSIMLoss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.LOL_DIR, help="folder containing our485/ and eval15/")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=256, help="random crop size for training")
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--val-count", type=int, default=20, help="images held out of our485 for checkpoint selection")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=config.WEIGHTS_DIR / "cnn.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    train_names, val_names = split_train_val(args.data, args.val_count, args.seed)
    low_dir, high_dir = args.data / "our485" / "low", args.data / "our485" / "high"

    train_ds = PairDataset(low_dir, high_dir, train_names, crop=args.crop, augment=True)
    val_ds = PairDataset(low_dir, high_dir, val_names) if val_names else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    print(f"device={device}  train={len(train_ds)}  val={len(val_names)}  epochs={args.epochs}")

    model = LowLightCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    l1, ssim_loss = torch.nn.L1Loss(), SSIMLoss().to(device)

    history = History("cnn", {"model": "Custom CNN", "loss": f"L1 + {args.ssim_weight} x (1 - SSIM)",
                              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                              "train_images": len(train_ds), "val_images": len(val_names), "device": str(device)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_psnr, start = -1.0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            pred = model(low)
            loss = l1(pred, high) + args.ssim_weight * ssim_loss(pred, high)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * low.size(0)
            count += low.size(0)
        sched.step()

        entry = {"epoch": epoch, "loss": total / count, "seconds": round(time.time() - start, 1)}
        if val_ds is not None:
            model.eval()
            entry["val_psnr"], entry["val_ssim"] = evaluate(model, val_ds, device)
            if entry["val_psnr"] > best_psnr:
                best_psnr = entry["val_psnr"]
                torch.save(model.state_dict(), args.out)
                entry["best"] = True
        else:
            torch.save(model.state_dict(), args.out)
        history.add(**entry)
        extra = f"  val PSNR {entry['val_psnr']:.2f} dB  SSIM {entry['val_ssim']:.4f}{'  *' if entry.get('best') else ''}" if val_ds else ""
        print(f"epoch {epoch:3d}/{args.epochs}  loss {entry['loss']:.4f}{extra}")

    torch.save(model.state_dict(), args.out.with_name(args.out.stem + "_last.pth"))
    print(f"\nDone. Best weights: {args.out}")


if __name__ == "__main__":
    main()
