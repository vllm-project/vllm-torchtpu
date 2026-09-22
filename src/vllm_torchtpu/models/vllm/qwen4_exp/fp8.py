# SPDX-License-Identifier: Apache-2.0
"""Lossless checkpoint-block packing for Qwen4Exp W8A16 GMM.

Checkpoint axes are [expert, output, input], with 128x128 scale blocks.
GMM axes are [expert, input, output]; gate/up are contiguous HBM halves.
Only the existing GMM performs the 128-lane VMEM interleave. No dequantization
or requantization occurs here. Scale values are widened exactly to FP32.
"""
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertGeometry:
    tp_size: int = 1
    tp_rank: int = 0
    candidate: str = "A"

    def __post_init__(self):
        if (self.tp_size not in (1, 2, 4, 8)
                or not 0 <= self.tp_rank < self.tp_size
                or self.candidate not in ("A", "B")):
            raise ValueError(
                "Expected TP1/2/4/8, a valid rank, and A/B packing")

    @property
    def logical_size(self):
        return 640 // self.tp_size

    @property
    def start(self):
        return self.tp_rank * self.logical_size

    @property
    def base(self):
        return self.start // 128 * 128 if self.candidate == "A" else self.start

    @property
    def physical_size(self):
        if self.candidate == "B":
            return math.ceil(self.logical_size / 128) * 128
        return max(
            math.ceil(((r + 1) * self.logical_size -
                       r * self.logical_size // 128 * 128) / 128) * 128
            for r in range(self.tp_size))

    @property
    def down_block(self):
        return 128 if self.candidate == "A" else math.gcd(
            128, self.logical_size)


def pack_expert_projection(
        weight: torch.Tensor, scale: torch.Tensor, projection: str,
        geometry: ExpertGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one or more complete CPU checkpoint experts, preserving FP8 bits.

    The leading expert dimension permits a one-expert streaming loader; the
    global intermediate coordinates remain intact until this function slices.
    """
    if projection not in ("gate", "up", "down"):
        raise ValueError(f"Unknown expert projection: {projection}")
    shape = (2560, 640) if projection == "down" else (640, 2560)
    if (weight.device.type != "cpu" or scale.device.type != "cpu"
            or weight.dtype != torch.float8_e4m3fn or weight.ndim != 3
            or weight.shape[1:] != shape or scale.shape
            != (weight.shape[0], shape[0] // 128, shape[1] // 128)
            or scale.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError(
            "Expected full CPU FP8 experts and 128x128 checkpoint scales")
    g = geometry
    coords = torch.arange(g.physical_size, device="cpu") + g.base
    valid = (coords >= g.start) & (coords < g.start + g.logical_size)
    safe = coords.clamp(0, 639)
    # Index uint8: CPU float8 advanced indexing is not implemented in Torch.
    bits = weight.view(torch.uint8)
    if projection != "down":
        packed = bits[:, safe, :].transpose(1, 2).contiguous()
        packed[..., ~valid] = 0
        scales = scale[:, safe // 128, :].transpose(1, 2).unsqueeze(2).float()
        scales[..., ~valid] = 1
    else:
        packed = bits[:, :, safe].transpose(1, 2).contiguous()
        packed[:, ~valid, :] = 0
        blocks = ((g.base + torch.arange(g.physical_size // g.down_block,
                                         device="cpu") * g.down_block) //
                  128).clamp(0, 4)
        scales = scale[:, torch.arange(2560, device="cpu") // 128, :]
        scales = scales[:, :, blocks].transpose(1, 2).unsqueeze(2).float()
    return packed.view(torch.float8_e4m3fn), scales.contiguous()
