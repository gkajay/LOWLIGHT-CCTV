import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- CNN (supervised)
class SSIMLoss(nn.Module):
    """1 - SSIM, differentiable, Gaussian 11x11 window."""

    def __init__(self, channels: int = 3, size: int = 11, sigma: float = 1.5):
        super().__init__()
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
        self.register_buffer("window", window)
        self.channels, self.pad = channels, size // 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        conv = lambda t: F.conv2d(t, self.window, padding=self.pad, groups=self.channels)
        mu_x, mu_y = conv(x), conv(y)
        sxx = conv(x * x) - mu_x ** 2
        syy = conv(y * y) - mu_y ** 2
        sxy = conv(x * y) - mu_x * mu_y
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
        return 1.0 - ssim.mean()


# ------------------------------------------- Zero-DCE++ (zero-reference, no targets)
class ColorConstancyLoss(nn.Module):
    """Keeps the mean of the R, G and B channels close to each other."""

    def forward(self, x):
        mr, mg, mb = torch.split(torch.mean(x, dim=(2, 3), keepdim=True), 1, dim=1)
        d_rg, d_rb, d_gb = (mr - mg) ** 2, (mr - mb) ** 2, (mb - mg) ** 2
        return torch.sqrt(d_rg ** 2 + d_rb ** 2 + d_gb ** 2 + 1e-12)


class SpatialConsistencyLoss(nn.Module):
    """Preserves local contrast: neighbouring-region differences must match the input's."""

    def __init__(self):
        super().__init__()
        kernels = {
            "left": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            "up": [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            "down": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
        for name, k in kernels.items():
            self.register_buffer(name, torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3))
        self.pool = nn.AvgPool2d(4)

    def forward(self, original, enhanced):
        o = self.pool(torch.mean(original, 1, keepdim=True))
        e = self.pool(torch.mean(enhanced, 1, keepdim=True))
        total = 0.0
        for name in ("left", "right", "up", "down"):
            k = getattr(self, name)
            total = total + (F.conv2d(o, k, padding=1) - F.conv2d(e, k, padding=1)) ** 2
        return total


class ExposureLoss(nn.Module):
    """Pushes the average brightness of each 16x16 patch toward a well-exposed level."""

    def __init__(self, patch_size: int = 16, target: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = target

    def forward(self, x):
        return torch.mean((self.pool(torch.mean(x, 1, keepdim=True)) - self.target) ** 2)


class IlluminationSmoothnessLoss(nn.Module):
    """Total-variation penalty on the curve-parameter map so neighbouring pixels get similar curves."""

    def forward(self, a):
        b, _, h, w = a.shape
        h_tv = ((a[:, :, 1:, :] - a[:, :, :-1, :]) ** 2).sum()
        w_tv = ((a[:, :, :, 1:] - a[:, :, :, :-1]) ** 2).sum()
        count_h = (h - 1) * w * a.shape[1]
        count_w = h * (w - 1) * a.shape[1]
        return 2 * (h_tv / count_h + w_tv / count_w) / b
