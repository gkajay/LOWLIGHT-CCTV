"""Central configuration. Every path can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# Weights produced by train/train_cnn.py, train/train_zerodce.py and train/train_gan.py.
# lowlight_model.pth (your earlier CNN weights) is accepted as a fallback.
CNN_WEIGHTS = Path(os.getenv("CNN_WEIGHTS") or _first_existing(WEIGHTS_DIR / "cnn.pth", WEIGHTS_DIR / "lowlight_model.pth"))
ZDCE_WEIGHTS = Path(os.getenv("ZDCE_WEIGHTS") or WEIGHTS_DIR / "zerodce.pth")
# Generator of the GAN (same architecture as the CNN), written by train/train_gan.py.
GAN_WEIGHTS = Path(os.getenv("GAN_WEIGHTS") or WEIGHTS_DIR / "gan.pth")

# Folder that contains our485/ and eval15/ (the LOL dataset).
_lol = os.getenv("LOL_DIR")
LOL_DIR = Path(_lol) if _lol else _first_existing(BASE_DIR / "LOLdataset", BASE_DIR.parent / "LOLdataset")

# Uploads larger than this (longest side, px) are downscaled before enhancement.
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))
# Zero-DCE++ downscale factor used at inference time (12 is the paper's default).
ZDCE_SCALE = int(os.getenv("ZDCE_SCALE", "12"))

FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
