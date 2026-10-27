"""Shared helpers for the training scripts: dataset, train/val split, evaluation, history."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app import config
from app.metrics import psnr_ssim

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> list[str]:
    return sorted(p.name for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def split_train_val(lol_dir: Path, val_count: int, seed: int) -> tuple[list[str], list[str]]:
    """Hold `val_count` images out of our485 for checkpoint selection.
    eval15 is never touched during training, so it stays a clean test set."""
    names = list_images(lol_dir / "our485" / "low")
    if not names:
        raise SystemExit(f"No images found in {lol_dir / 'our485' / 'low'}. Pass --data pointing at the LOLdataset folder.")
    rng = random.Random(seed)
    rng.shuffle(names)
    val_count = min(val_count, max(len(names) - 1, 0))
    return sorted(names[val_count:]), sorted(names[:val_count])


class PairDataset(Dataset):
    """Low-light / normal-light pairs held in memory as uint8 arrays."""

    def __init__(self, low_dir: Path, high_dir: Path, names: list[str],
                 crop: int | None = None, augment: bool = False, paired: bool = True):
        self.crop, self.augment, self.paired = crop, augment, paired
        self.low = [np.asarray(Image.open(Path(low_dir) / n).convert("RGB")) for n in names]
        self.high = [np.asarray(Image.open(Path(high_dir) / n).convert("RGB")) for n in names] if paired else None

    def __len__(self) -> int:
        return len(self.low)

    @staticmethod
    def _tensor(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, i: int):
        lo = self.low[i]
        hi = self.high[i] if self.paired else None
        if self.crop:
            h, w = lo.shape[:2]
            c = min(self.crop, h, w)
            y, x = random.randint(0, h - c), random.randint(0, w - c)
            lo = lo[y:y + c, x:x + c]
            hi = hi[y:y + c, x:x + c] if hi is not None else None
        if self.augment:
            if random.random() < 0.5:
                lo = lo[:, ::-1]
                hi = hi[:, ::-1] if hi is not None else None
            if random.random() < 0.5:
                lo = lo[::-1]
                hi = hi[::-1] if hi is not None else None
            k = random.randint(0, 3) if lo.shape[0] == lo.shape[1] else 0
            if k:
                lo = np.rot90(lo, k)
                hi = np.rot90(hi, k) if hi is not None else None
        return (self._tensor(lo), self._tensor(hi)) if self.paired else self._tensor(lo)


@torch.no_grad()
def evaluate(forward, dataset: PairDataset, device: torch.device) -> tuple[float, float]:
    """Mean PSNR / SSIM of forward(low) against high over a dataset of full images."""
    psnrs, ssims = [], []
    for i in range(len(dataset)):
        low, high = dataset[i]
        out = forward(low.unsqueeze(0).to(device)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        p, s = psnr_ssim(high.permute(1, 2, 0).numpy(), out)
        psnrs.append(p)
        ssims.append(s)
    return float(np.mean(psnrs)), float(np.mean(ssims))


class History:
    """Per-epoch log written to backend/data/history_<key>.json (shown in the web UI)."""

    def __init__(self, key: str, meta: dict):
        self.path = config.DATA_DIR / f"history_{key}.json"
        self.data = {"meta": meta, "epochs": []}

    def add(self, **entry) -> None:
        self.data["epochs"].append(entry)
        self.path.write_text(json.dumps(self.data, indent=2))
