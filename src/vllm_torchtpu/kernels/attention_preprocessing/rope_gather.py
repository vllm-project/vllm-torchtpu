# SPDX-License-Identifier: Apache-2.0
"""RoPE rows for the attention-preprocessing kernels.

The model's ``cos_sin_cache`` (epic's ``_tpu_rope_cache``) is
``[max_position, rotary_dim]`` FP32 with ``[cos | sin]`` packed side by side.
Every layer of one forward step uses the same positions, so the caller gathers
the ``[tokens, rotary_dim]`` rows once per step (``gather_rows``) and every
kernel reads them as an ordinary tiled input. An in-kernel DMA gather was
measured slower: about 0.1 us per contiguous run, so decode batches and
non-contiguous positions cost up to 2-3x the kernel itself.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.attention_preprocessing.serial import _LANE, _lane_gather


def gather_rows(cos_sin_cache: jax.Array, positions: jax.Array) -> jax.Array:
    """``[tokens, rotary_dim]`` rows for ``positions``, clamped to the table."""
    clamped = jnp.clip(positions, 0, cos_sin_cache.shape[0] - 1)
    return jnp.take(cos_sin_cache, clamped, axis=0)


def coefficients(cos_sin):
    """``[rows, rotary_dim]`` packed rows -> 128-lane cos and sign-folded sin.

    The rotated channels are the tail of the last lane block; NoPE lanes get
    the identity (cos 1, sin 0). Pair partners are lanes ``i`` and ``i ^ 1``;
    the GPT-J sign (negative on even lanes) is folded into sin, which is exact.
    """
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    lane = jnp.arange(_LANE)
    channel = lane - (_LANE - rotary_dim)
    is_rot = channel >= 0
    freq = jnp.maximum(channel, 0) >> 1
    sign = jnp.where((lane & 1) == 0, -1.0, 1.0)
    cos = jnp.where(is_rot, _lane_gather(cos_sin, freq), 1.0)
    sin = jnp.where(is_rot, _lane_gather(cos_sin, half + freq), 0.0) * sign
    return cos, sin


def check_rope_rows(rope_rows, tokens, head_dim):
    if rope_rows.ndim != 2 or rope_rows.dtype != jnp.float32:
        raise ValueError("rope_rows must be FP32 [tokens, rotary_dim]")
    if rope_rows.shape[0] != tokens:
        raise ValueError("rope_rows needs one row per token")
    rotary_dim = rope_rows.shape[1]
    if rotary_dim % 2 or not 0 < rotary_dim <= _LANE or rotary_dim >= head_dim:
        raise ValueError("Rotary dim must be even, at most one lane block")


def split_cos_sin(cos_sin_cache, positions):
    """Reference helper: gathered ``(cos, sin)`` halves, clamped like gather_rows."""
    rows = gather_rows(cos_sin_cache, positions)
    half = cos_sin_cache.shape[1] // 2
    return rows[:, :half], rows[:, half:]
