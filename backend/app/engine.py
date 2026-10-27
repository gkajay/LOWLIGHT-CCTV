"""Loads the two trained models and runs inference on this machine (no external services)."""
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .models.cnn import LowLightCNN
from .models.zerodce import ZeroDCEPP, enhance_padded

MODELS = {
    "cnn": {"label": "Custom CNN", "weights": lambda: config.CNN_WEIGHTS},
    "zerodce": {"label": "Zero-DCE++", "weights": lambda: config.ZDCE_WEIGHTS},
    "gan": {"label": "GAN", "weights": lambda: config.GAN_WEIGHTS},
}

TRAIN_MODULE = {"cnn": "train_cnn", "zerodce": "train_zerodce", "gan": "train_gan"}
# The GAN generator has the same architecture as the CNN, so it reuses LowLightCNN.
CNN_LIKE = ("cnn", "gan")


def _extract_state_dict(obj) -> dict:
    """Accept a raw state_dict or a checkpoint dict; strip a DataParallel 'module.' prefix."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    return {k.removeprefix("module."): v for k, v in obj.items()}


class Engine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, torch.nn.Module] = {}
        self._errors: dict[str, str] = {}

    # ------------------------------------------------------------------ loading
    def _build(self, key: str) -> torch.nn.Module:
        path: Path = MODELS[key]["weights"]()
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path.name}. Train the model first "
                f"(python -m train.{TRAIN_MODULE[key]}) or place the file in backend/weights/."
            )
        model = LowLightCNN() if key in CNN_LIKE else ZeroDCEPP(scale_factor=config.ZDCE_SCALE)
        state = _extract_state_dict(torch.load(path, map_location="cpu"))
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model

    def get(self, key: str) -> torch.nn.Module:
        if key not in MODELS:
            raise KeyError(key)
        if key not in self._models:
            try:
                self._models[key] = self._build(key)
                self._errors.pop(key, None)
                self._warmup(key)
            except Exception as exc:  # reported through /api/health
                self._errors[key] = str(exc)
                raise
        return self._models[key]

    def reload(self) -> None:
        """Forget cached models so freshly trained weights are picked up."""
        self._models.clear()
        self._errors.clear()

    def status(self) -> dict:
        out = {}
        for key, meta in MODELS.items():
            path: Path = meta["weights"]()
            loaded = True
            try:
                self.get(key)
            except Exception:
                loaded = False
            out[key] = {
                "label": meta["label"],
                "weights_file": path.name,
                "weights_found": path.exists(),
                "ready": loaded,
                "error": None if loaded else self._errors.get(key),
            }
        return out

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _forward(self, key: str, x: torch.Tensor) -> torch.Tensor:
        model = self.get(key)
        return model(x) if key in CNN_LIKE else enhance_padded(model, x)

    def _warmup(self, key: str) -> None:
        self._forward(key, torch.zeros(1, 3, 48, 48, device=self.device))

    def run(self, key: str, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Enhance a float32 HxWx3 image in [0, 1]. Returns (image, milliseconds)."""
        self.get(key)
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = self._forward(key, x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        out = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return out, ms
