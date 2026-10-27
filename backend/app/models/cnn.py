import torch
import torch.nn as nn


class LowLightCNN(nn.Module):
    """Custom 4-layer CNN: direct low-light -> enhanced image mapping.

    Architecture is identical to the original project's model.py, so weights
    trained earlier (weights/lowlight_model.pth) still load.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
