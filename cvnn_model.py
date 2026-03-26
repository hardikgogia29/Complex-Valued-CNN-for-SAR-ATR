"""
cvnn_model.py — Complex-Valued CNN for SAR Target Classification
================================================================
v4: 2-call optimized ComplexConv2d and ComplexLinear.

The v3 forward pass used 4 separate F.conv2d / F.linear calls per layer:
    out_r = F.conv2d(z.real, wr) - F.conv2d(z.imag, wi)
    out_i = F.conv2d(z.real, wi) + F.conv2d(z.imag, wr)

This version reduces to 2 calls by stacking the real and imaginary
channels of the input and constructing compound weight matrices:

    z_stack        = cat([z.real, z.imag], dim=1)   # (B, 2*C_in, H, W)
    W_real_part    = cat([wr, -wi],        dim=1)   # (C_out, 2*C_in, k, k)
    W_imag_part    = cat([wi,  wr],        dim=1)   # (C_out, 2*C_in, k, k)

    out_r = F.conv2d(z_stack, W_real_part)          # 1 call
    out_i = F.conv2d(z_stack, W_imag_part)          # 1 call

Proof of equivalence:
    F.conv2d(cat([z.real, z.imag]), cat([wr, -wi]))
    = conv(z.real, wr) + conv(z.imag, -wi)
    = conv(z.real, wr) - conv(z.imag,  wi)   ✓  (= out_r)

    F.conv2d(cat([z.real, z.imag]), cat([wi,  wr]))
    = conv(z.real, wi) + conv(z.imag,  wr)   ✓  (= out_i)

This is mathematically identical to v3. The parameters (weight, bias)
are stored identically as complex64 nn.Parameters with the same Kaiming
initialisation. The only change is the forward computation graph.

Speedup: 2x fewer kernel launches per layer -> ~2x faster per epoch.

At 72s/epoch (measured on T4 with v3, batch=128):
    v4 estimate: ~36s/epoch
    200 epochs SGD + APGD: ~7.0 hours (fits in Kaggle 9-hour session)

All other classes (Cardioid, ComplexDropout, ComplexAvgPool2d,
ComplexGlobalAvgPool2d, CV_CNN) are identical to v3.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Kaiming initialiser for complex parameters (unchanged from v3)
# ──────────────────────────────────────────────────────────────────────────────

def _complex_kaiming_uniform_(tensor: torch.Tensor, fan_in: int) -> torch.Tensor:
    """
    Complex Kaiming uniform: each component (real, imag) uses
    bound = sqrt(3) * sqrt(2/fan_in) / sqrt(2)
    so that E[|w|^2] = 2/fan_in  (matches real Kaiming variance).
    """
    bound = math.sqrt(3.0) * math.sqrt(2.0 / fan_in) / math.sqrt(2.0)
    with torch.no_grad():
        tensor.real.uniform_(-bound, bound)
        tensor.imag.uniform_(-bound, bound)
    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# ComplexConv2d — 2-call optimized (v4)
# ──────────────────────────────────────────────────────────────────────────────

class ComplexConv2d(nn.Module):
    """
    2-D convolution with complex64 weight and bias. 2-call forward.

    Parameters are identical to v3 (complex64 nn.Parameters, same init).
    Forward uses stacked-channel trick to reduce 4 conv calls to 2:

        z_stack = cat([z.real, z.imag], dim=1)        # (B, 2*C_in, H, W)
        out_r   = conv2d(z_stack, cat([wr, -wi], 1))  # real part
        out_i   = conv2d(z_stack, cat([wi,  wr], 1))  # imag part
        out     = complex(out_r, out_i) + bias
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size,
                        dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        _complex_kaiming_uniform_(self.weight, fan_in)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        wr = self.weight.real   # (out, in, k, k)
        wi = self.weight.imag   # (out, in, k, k)
        s, p = self.stride, self.padding

        # Stack input channels: (B, 2*in, H, W)
        z_stack = torch.cat([z.real, z.imag], dim=1)

        # 2 conv calls instead of 4
        out_r = F.conv2d(z_stack, torch.cat([wr, -wi], dim=1), None, s, p)
        out_i = F.conv2d(z_stack, torch.cat([wi,  wr], dim=1), None, s, p)

        out = torch.complex(out_r, out_i)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out


# ──────────────────────────────────────────────────────────────────────────────
# ComplexLinear — 2-call optimized (v4)
# ──────────────────────────────────────────────────────────────────────────────

class ComplexLinear(nn.Module):
    """
    Linear layer with complex64 weight and bias. 2-call forward.

    Parameters are identical to v3 (complex64 nn.Parameters, same init).
    Forward uses stacked-feature trick to reduce 4 linear calls to 2:

        z_stack = cat([z.real, z.imag], dim=-1)          # (..., 2*in)
        out_r   = linear(z_stack, cat([wr, -wi], dim=1)) # real part
        out_i   = linear(z_stack, cat([wi,  wr], dim=1)) # imag part
        out     = complex(out_r, out_i) + bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        _complex_kaiming_uniform_(self.weight, self.in_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        wr = self.weight.real   # (out, in)
        wi = self.weight.imag   # (out, in)

        # Stack input features: (..., 2*in)
        z_stack = torch.cat([z.real, z.imag], dim=-1)

        # 2 linear calls instead of 4
        out_r = F.linear(z_stack, torch.cat([wr, -wi], dim=1), None)
        out_i = F.linear(z_stack, torch.cat([wi,  wr], dim=1), None)

        out = torch.complex(out_r, out_i)

        if self.bias is not None:
            out = out + self.bias

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks (identical to v3)
# ──────────────────────────────────────────────────────────────────────────────

class Cardioid(nn.Module):
    """f(z) = 0.5 * (1 + cos(arg(z))) * z"""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scale = 0.5 * (1.0 + z.real / (z.abs().clamp(min=1e-8)))
        return scale * z


class ComplexDropout(nn.Module):
    """Shared-mask dropout for complex tensors."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return z
        mask = torch.ones(z.shape, dtype=torch.float32, device=z.device)
        mask = F.dropout(mask, p=self.p, training=True)
        return z * mask.to(z.dtype)


class ComplexAvgPool2d(nn.Module):
    """2-D average pooling for complex tensors."""
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.pool(z.real), self.pool(z.imag))


class ComplexGlobalAvgPool2d(nn.Module):
    """Global average pool: (B, C, H, W) -> (B, C)."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.mean(dim=(-2, -1))


# ──────────────────────────────────────────────────────────────────────────────
# CV_CNN (identical to v3)
# ──────────────────────────────────────────────────────────────────────────────

class CV_CNN(nn.Module):
    """
    Complex-Valued CNN for SAR ATR.
    Input:  (B, 1, H, W) complex64
    Output: (B, num_classes) float32  — |complex logits|
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.3,
                 img_size: int = 128):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = ComplexConv2d(1,   32,  kernel_size=3, padding=1)
        self.conv2 = ComplexConv2d(32,  64,  kernel_size=3, padding=1)
        self.conv3 = ComplexConv2d(64,  64,  kernel_size=3, padding=1)
        self.conv4 = ComplexConv2d(64,  128, kernel_size=3, padding=1)
        self.conv5 = ComplexConv2d(128, 128, kernel_size=3, padding=1)

        self.pool    = ComplexAvgPool2d(kernel_size=2, stride=2)
        self.act     = Cardioid()
        self.dropout = ComplexDropout(p=dropout_p)

        self.gap = ComplexGlobalAvgPool2d()
        self.fc1 = ComplexLinear(128, 256)
        self.fc2 = ComplexLinear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)

        x = self.act(self.conv3(x))
        x = self.pool(x)

        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.pool(x)

        x = self.gap(x)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x.abs()

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            'complex_params'        : total,
            'real_equivalent_params': total * 2,
        }
