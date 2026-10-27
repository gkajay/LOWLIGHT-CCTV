import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise conv followed by a 1x1 pointwise conv (the 'CSDN' block of Zero-DCE++)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEPP(nn.Module):
    """Zero-DCE++: predicts per-pixel curve parameters and applies the curve 8 times.

        LE(I) = I + A * (I^2 - I)

    `scale_factor` > 1 runs the curve-estimation network on a downscaled copy of
    the image (fast) and upsamples the parameter map. It can be changed at
    runtime because it does not affect the weights.
    """

    def __init__(self, scale_factor: int = 1, iterations: int = 8, channels: int = 32):
        super().__init__()
        self.scale_factor = scale_factor
        self.iterations = iterations
        self.relu = nn.ReLU(inplace=True)
        c = channels
        self.e_conv1 = DepthwiseSeparableConv(3, c)
        self.e_conv2 = DepthwiseSeparableConv(c, c)
        self.e_conv3 = DepthwiseSeparableConv(c, c)
        self.e_conv4 = DepthwiseSeparableConv(c, c)
        self.e_conv5 = DepthwiseSeparableConv(c * 2, c)
        self.e_conv6 = DepthwiseSeparableConv(c * 2, c)
        self.e_conv7 = DepthwiseSeparableConv(c * 2, 3)

    def forward(self, x):
        h, w = x.shape[2:]
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, size=(max(h // self.scale_factor, 1), max(w // self.scale_factor, 1)),
                                   mode="bilinear", align_corners=False)

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor != 1:
            x_r = F.interpolate(x_r, size=(h, w), mode="bilinear", align_corners=True)

        enhanced = x
        for _ in range(self.iterations):
            enhanced = enhanced + x_r * (enhanced * enhanced - enhanced)
        return enhanced, x_r


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.no_grad()
def enhance_padded(model: ZeroDCEPP, x: torch.Tensor) -> torch.Tensor:
    """Run the model on any image size: pad to a multiple of scale_factor, then crop back."""
    s = model.scale_factor
    h, w = x.shape[2:]
    ph, pw = (-h) % s, (-w) % s
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    out, _ = model(x)
    return out[:, :, :h, :w]
