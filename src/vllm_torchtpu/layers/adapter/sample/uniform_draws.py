# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for in-program uniform draws (``kernels/uniform_draws.py``)."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.uniform_draws import initial_state
from vllm_torchtpu.kernels.uniform_draws import uniform_draws as _jax_uniform_draws

__all__ = ["initial_state", "uniform_draws"]

# Registered at import: the op is called from inside compiled programs.
_op = pallas.jax_op("pallas::uniform_draws", _jax_uniform_draws)


@_op.register_fake
def _fake_uniform_draws(state: torch.Tensor, rows: int, cols: int, count: int):
    return (
        state.new_empty((rows, cols), dtype=torch.float32),
        state.new_empty((count,), dtype=torch.float32),
        torch.empty_like(state),
    )


def _mix64(z: torch.Tensor) -> torch.Tensor:
    """splitmix64's finalizer on int64 (wrapping multiplies, logical shifts)."""
    z = (z ^ ((z >> 30) & 0x3FFFFFFFF)) * -4658895280553007687
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFF)) * -7723592293110705685
    return z ^ ((z >> 31) & 0x1FFFFFFFF)


def _hashed_uniform_draws(
    state: torch.Tensor, rows: int, cols: int, count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The same contract off TPU (CPU tests), from a counter hash of the state."""
    words = state.to(torch.int64) & 0xFFFFFFFF
    key = _mix64((words[0] | (words[1] << 32)) ^ _mix64(words[2] | (words[3] << 32)))
    counters = torch.arange(rows * cols + count, device=state.device)
    bits = (_mix64(counters ^ key) >> 40) & 0xFFFFFF
    draws = bits.to(torch.float32) * (1.0 / (1 << 24))
    step = torch.tensor([0, 0, 1, 0], dtype=state.dtype, device=state.device)
    return draws[: rows * cols].view(rows, cols), draws[rows * cols :], state + step


def uniform_draws(
    state: torch.Tensor, rows: int, cols: int, count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """F32 [rows, cols] and [count] uniforms in [0, 1) drawn from ``state``
    (int32 [4], ``initial_state``), and the state to draw from next."""
    if state.device.type != "tpu":
        return _hashed_uniform_draws(state, rows, cols, count)
    return _op(state, rows, cols, count)
