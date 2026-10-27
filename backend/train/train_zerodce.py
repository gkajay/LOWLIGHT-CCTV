"""Train Zero-DCE++ from scratch (no pretrained checkpoint, no reference images needed).

    cd backend
    python -m train.train_zerodce --data ../LOLdataset

Zero-DCE++ is zero-reference: it learns from the low-light images alone using
four hand-designed losses (spatial consistency, exposure, colour constancy and
illumination smoothness). Reference images are only used to pick the best
checkpoint on images held out from our485.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app import config
from app.models.zerodce import ZeroDCEPP, enhance_padded, init_weights

from .common import History, PairDataset, evaluate, pick_device, split_train_val
from .losses import ColorConstancyLoss, ExposureLoss, IlluminationSmoothnessLoss, SpatialConsistencyLoss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.LOL_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--scale-factor", type=int, default=1, help="downscale used while training (1 = full resolution)")
    ap.add_argument("--tv-weight", type=float, default=200.0)
    ap.add_argument("--exp-weight", type=float, default=10.0)
    ap.add_argument("--col-weight", type=float, default=5.0)
    ap.add_argument("--spa-weight", type=float, default=1.0)
    ap.add_argument("--val-count", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=config.WEIGHTS_DIR / "zerodce.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    train_names, val_names = split_train_val(args.data, args.val_count, args.seed)
    low_dir, high_dir = args.data / "our485" / "low", args.data / "our485" / "high"

    train_ds = PairDataset(low_dir, None, train_names, crop=args.crop, augment=True, paired=False)
    val_ds = PairDataset(low_dir, high_dir, val_names) if val_names else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print(f"device={device}  train={len(train_ds)}  val={len(val_names)}  epochs={args.epochs}")

    model = ZeroDCEPP(scale_factor=args.scale_factor).to(device)
    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    l_col, l_spa = ColorConstancyLoss().to(device), SpatialConsistencyLoss().to(device)
    l_exp, l_tv = ExposureLoss().to(device), IlluminationSmoothnessLoss().to(device)

    history = History("zerodce", {"model": "Zero-DCE++", "loss": "spatial + exposure + colour + smoothness (zero-reference)",
                                  "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                                  "train_images": len(train_ds), "val_images": len(val_names), "device": str(device)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_psnr, start = -1.0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for low in loader:
            low = low.to(device)
            enhanced, curves = model(low)
            loss = (args.tv_weight * l_tv(curves)
                    + args.spa_weight * l_spa(low, enhanced).mean()
                    + args.col_weight * l_col(enhanced).mean()
                    + args.exp_weight * l_exp(enhanced))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            total += loss.item() * low.size(0)
            count += low.size(0)

        entry = {"epoch": epoch, "loss": total / count, "seconds": round(time.time() - start, 1)}
        if val_ds is not None:
            model.eval()
            train_scale, model.scale_factor = model.scale_factor, config.ZDCE_SCALE  # validate the way it is deployed
            entry["val_psnr"], entry["val_ssim"] = evaluate(lambda x: enhance_padded(model, x), val_ds, device)
            model.scale_factor = train_scale
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
