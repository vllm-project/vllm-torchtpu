"""
quant.py
--------
Weight quantization simulation module for uniform integer quantization (RTN, GPTQ).
Provides clamp and scaling functions for simulated low-bit precision benchmarks.
"""

import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    """Applies uniform integer quantization and de-quantization to tensor x."""
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    """
    Quantization module that calculates per-channel or per-tensor scales and zero-points
    for simulated integer weight quantization.
    """

    def __init__(self, shape=1):
        """Initializes quantization buffers for scale and zero-point."""
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self,
                  bits,
                  perchannel=False,
                  sym=True,
                  mse=False,
                  norm=2.4,
                  grid=100,
                  maxshrink=.8,
                  grouprows=1):
        """Configures quantization bit-width, symmetry, and channel granularity."""
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.grouprows = grouprows

    def find_params(self, x, weight=False):
        """Calculates optimal scale and zero-point parameters for tensor x."""
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) > 1 else 1
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)

    def quantize(self, x):
        """Applies configured quantization to tensor x."""
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        """Returns True if quantization is active (maxq > 0)."""
        return self.maxq > 0

    def ready(self):
        """Returns True if scale parameters have been initialized."""
        return torch.all(self.scale != 0)
