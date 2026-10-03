"""Torch normalization helpers with FP32 accumulation."""

import torch


def rms_norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype)
