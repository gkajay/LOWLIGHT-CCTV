"""Train the GAN: the custom CNN is the generator, a conditional PatchGAN is the discriminator.

    cd backend
    python -m train.train_gan --data ../LOLdataset

By default the generator starts from the trained CNN (weights/cnn.pth), so train
the CNN first. The GAN then fine-tunes it with an adversarial loss that rewards
images the discriminator cannot tell apart from real bright photos.

Generator loss     = L1 + 0.2 * (1 - SSIM) + adv_weight * adversarial loss
Discriminator loss = least-squares GAN (real -> 1, generated -> 0)

Files written to backend/weights/:
    gan.pth       generator after the LAST epoch (this is the one the web app uses)
    gan_best.pth  generator with the best validation PSNR (usually less "GAN-like")
eval15 is never used here, so it stays a clean test set.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app import config
from app.engine import _extract_state_dict
from app.models.cnn import LowLightCNN
from app.models.discriminator import PatchDiscriminator

from .common import History, PairDataset, evaluate, pick_device, split_train_val
from .losses import SSIMLoss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.LOL_DIR, help="folder containing our485/ and eval15/")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--g-lr", type=float, default=1e-4, help="generator learning rate")
    ap.add_argument("--d-lr", type=float, default=1e-4, help="discriminator learning rate")
    ap.add_argument("--adv-weight", type=float, default=0.01, help="weight of the adversarial term in the generator loss")
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--val-count", type=int, default=20)
    ap.add_argument("--init-from", type=Path, default=config.CNN_WEIGHTS, help="CNN weights that initialise the generator")
    ap.add_argument("--scratch", action="store_true", help="train the generator from random weights (less stable)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=config.WEIGHTS_DIR / "gan.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    train_names, val_names = split_train_val(args.data, args.val_count, args.seed)
    low_dir, high_dir = args.data / "our485" / "low", args.data / "our485" / "high"

    train_ds = PairDataset(low_dir, high_dir, train_names, crop=args.crop, augment=True)
    val_ds = PairDataset(low_dir, high_dir, val_names) if val_names else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    G = LowLightCNN().to(device)
    init_note = "random weights"
    if not args.scratch:
        if args.init_from.exists():
            G.load_state_dict(_extract_state_dict(torch.load(args.init_from, map_location="cpu")))
            init_note = f"initialised from {args.init_from.name}"
        else:
            print(f"WARNING: {args.init_from} not found. Training the generator from scratch. "
                  f"Train the CNN first (python -m train.train_cnn) for a more stable GAN.")
    D = PatchDiscriminator().to(device)
    print(f"device={device}  train={len(train_ds)}  val={len(val_names)}  epochs={args.epochs}  generator: {init_note}")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    l1, mse, ssim_loss = nn.L1Loss(), nn.MSELoss(), SSIMLoss().to(device)

    history = History("gan", {
        "model": "GAN", "loss": f"generator: L1 + {args.ssim_weight} x (1 - SSIM) + {args.adv_weight} x adversarial (LSGAN)",
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.g_lr,
        "train_images": len(train_ds), "val_images": len(val_names), "device": str(device),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_path = args.out.with_name(args.out.stem + "_best.pth")
    best_psnr, start = -1.0, time.time()

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        sum_g = sum_rec = sum_d = 0.0
        count = 0
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            fake = G(low)

            # --- discriminator: real bright image -> 1, generated image -> 0
            d_real = D(low, high)
            d_fake = D(low, fake.detach())
            loss_d = 0.5 * (mse(d_real, torch.ones_like(d_real)) + mse(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # --- generator: match the reference AND fool the discriminator
            d_out = D(low, fake)
            adv = mse(d_out, torch.ones_like(d_out))
            rec = l1(fake, high) + args.ssim_weight * ssim_loss(fake, high)
            loss_g = rec + args.adv_weight * adv
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            n = low.size(0)
            sum_g += loss_g.item() * n
            sum_rec += rec.item() * n
            sum_d += loss_d.item() * n
            count += n

        entry = {"epoch": epoch, "loss": sum_g / count, "rec_loss": sum_rec / count,
                 "d_loss": sum_d / count, "seconds": round(time.time() - start, 1)}
        if val_ds is not None:
            G.eval()
            entry["val_psnr"], entry["val_ssim"] = evaluate(G, val_ds, device)
            if entry["val_psnr"] > best_psnr:
                best_psnr = entry["val_psnr"]
                torch.save(G.state_dict(), best_path)
                entry["best"] = True
        torch.save(G.state_dict(), args.out)  # last epoch
        history.add(**entry)
        extra = f"  val PSNR {entry['val_psnr']:.2f} dB  SSIM {entry['val_ssim']:.4f}{'  *' if entry.get('best') else ''}" if val_ds else ""
        print(f"epoch {epoch:3d}/{args.epochs}  G {entry['loss']:.4f}  D {entry['d_loss']:.4f}{extra}")

    print(f"\nDone. Generator (last epoch): {args.out}\nBest by validation PSNR:      {best_path}")


if __name__ == "__main__":
    main()
