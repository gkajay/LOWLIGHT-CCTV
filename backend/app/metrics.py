import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr_ssim(reference: np.ndarray, image: np.ndarray) -> tuple[float, float]:
    """PSNR (dB) and SSIM for two float32 RGB images in [0, 1] with identical shape.

    The same function and settings are used for every model so results are comparable.
    """
    if reference.shape != image.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {image.shape}")
    psnr = float(peak_signal_noise_ratio(reference, image, data_range=1.0))
    ssim = float(structural_similarity(reference, image, channel_axis=2, data_range=1.0))
    return min(psnr, 99.0), ssim


def mean_brightness(image: np.ndarray) -> float:
    """Mean luminance in [0, 1]."""
    return float((0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).mean())
