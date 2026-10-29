# SPDX-License-Identifier: Apache-2.0
"""Typed Torch bridges for the fused DeepSeek-V4.1 compressor."""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.attention_preprocessing import compress_rope
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_ops: dict[tuple[int, int, float], Any] = {}


def _run(x, weight, gamma, rows, has_prev, valid, state, write, read, *, ratio, eps):
    if x.shape[0] == 0:
        return (
            jnp.zeros((0, 512), jnp.bfloat16),
            jnp.zeros((0, 128), jnp.int32),
            jnp.zeros((0, 1), jnp.float32),
            state,
        )
    return compress_rope.fused_compress_rope(
        x,
        weight,
        gamma,
        rows,
        has_prev,
        valid,
        state,
        write,
        read,
        ratio=ratio,
        eps=eps,
    )


def _bridge(ratio: int, banks: int, eps: float = 1e-6):
    if (ratio, banks) not in ((1, 0), (2, 1), (2, 2)):
        raise ValueError("Use ratio 1 without state, or ratio 2 with one/two banks")
    run = functools.partial(_run, ratio=ratio, eps=eps)
    if ratio == 1:

        def ratio1(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            latent, words, scales, _ = run(
                x, weight, gamma, rows, has_prev, valid, None, None, None
            )
            return latent, words, scales

        return ratio1
    if banks == 1:

        def ratio2_one(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
            state: jax.Array,
            write: jax.Array,
            read: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            latent, words, scales, updated = run(
                x, weight, gamma, rows, has_prev, valid, state, write, read
            )
            return latent, words, scales, updated

        return ratio2_one

    def ratio2_two(
        x: jax.Array,
        weight: jax.Array,
        gamma: jax.Array,
        rows: jax.Array,
        has_prev: jax.Array,
        valid: jax.Array,
        state_low: jax.Array,
        state_high: jax.Array,
        write: jax.Array,
        read: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        latent, words, scales, updated = run(
            x,
            weight,
            gamma,
            rows,
            has_prev,
            valid,
            (state_low, state_high),
            write,
            read,
        )
        assert isinstance(updated, tuple)
        return latent, words, scales, updated[0], updated[1]

    return ratio2_two


def _fake(ratio, banks):
    def fake(x, weight, gamma, rows, has_prev, valid, *state_and_ring):
        latent = x.new_empty((x.shape[0], 512), dtype=torch.bfloat16)
        words = x.new_empty((x.shape[0], 128), dtype=torch.int32)
        scales = x.new_empty((x.shape[0], 1), dtype=torch.float32)
        states = tuple(torch.empty_like(a) for a in state_and_ring[:banks])
        return latent, words, scales, *states

    return fake


def get_compress_rope_op(ratio: int, banks: int = 0, *, eps: float = 1e-6) -> Any:
    """Return the pre-registered ratio/bank/epsilon-specific operation."""
    function = _bridge(ratio, banks, float(eps))
    key = ratio, banks, float(eps)
    if key in _ops:
        return _ops[key]
    eps_tag = float(eps).hex().replace(".", "_").replace("+", "p").replace("-", "m")
    op = pallas.jax_op(
        f"pallas::dsv41_compress_rope_r{ratio}_b{banks}_e{eps_tag}",
        function,
        mesh=get_pallas_op_mesh(),
        donate_argnums=tuple(range(6, 6 + banks)),
        input_partition_specs=cast(Any, (P(),) * (6 if ratio == 1 else 8 + banks)),
    )
    op.register_fake(_fake(ratio, banks))
    _ops[key] = op
    return op
