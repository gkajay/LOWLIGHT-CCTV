import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator.

    It looks at the dark input together with a bright image (real or generated)
    and outputs a map of scores, one per image patch: high means "looks like a
    real bright photo of this scene", low means "looks generated". It is used
    only during training and is not needed to enhance images.
    """

    def __init__(self, in_channels: int = 6, base: int = 32):
        super().__init__()

        def block(cin: int, cout: int, stride: int, norm: bool = True):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, 2, norm=False),
            *block(base, base * 2, 2),
            *block(base * 2, base * 4, 2),
            nn.Conv2d(base * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, low: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([low, image], dim=1))
