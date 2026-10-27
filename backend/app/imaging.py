import io

import numpy as np
from PIL import Image, ImageOps


def load_rgb(data: bytes, max_side: int | None = None) -> tuple[np.ndarray, bool]:
    """Decode bytes -> float32 RGB array in [0, 1]. Returns (array, was_downscaled)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    resized = False
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        resized = True
    return np.asarray(img, dtype=np.float32) / 255.0, resized


def resize_to(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
