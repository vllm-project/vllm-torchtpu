# SPDX-License-Identifier: Apache-2.0
"""Typed Torch bridges for the fused DeepSeek-V4.1 compressor."""

from typing import Any, cast

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.attention_preprocessing import compress_rope
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_ops: dict[tuple[int, int], Any] = {}


def _run(
    x,
    weight,
    gamma,
    rows,
    has_prev,
    valid,
    *,
    ratio,
    state=None,
    write=None,
    read=None,
):
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
    )


def _bridge(ratio: int, banks: int):
    if ratio == 1:

        def ratio1(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            latent, words, scales, _ = _run(
                x, weight, gamma, rows, has_prev, valid, ratio=1
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
            latent, words, scales, new_state = _run(
                x,
                weight,
                gamma,
                rows,
                has_prev,
                valid,
                ratio=2,
                state=state,
                write=write,
                read=read,
            )
            assert isinstance(new_state, jax.Array)
            return latent, words, scales, new_state

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
        latent, words, scales, new_state = _run(
            x,
            weight,
            gamma,
            rows,
            has_prev,
            valid,
            ratio=2,
            state=(state_low, state_high),
            write=write,
            read=read,
        )
        assert isinstance(new_state, tuple)
        return latent, words, scales, new_state[0], new_state[1]

    return ratio2_two


def _fake(ratio, banks):
    def fake(x, weight, gamma, rows, has_prev, valid, *state_and_ring):
        latent = x.new_empty((x.shape[0], 512))
        words = x.new_empty((x.shape[0], 128), dtype=torch.int32)
        scales = x.new_empty((x.shape[0], 1), dtype=torch.float32)
        if ratio == 1:
            return latent, words, scales
        states = tuple(x.new_empty(a.shape) for a in state_and_ring[:banks])
        return latent, words, scales, *states

    return fake


def get_compress_rope_op(ratio: int, banks: int = 0) -> Any:
    """Return the pre-registered ratio/bank-specific compressor operation."""
    if ratio == 1:
        banks = 0
    if ratio not in (1, 2) or banks not in (0, 1, 2):
        raise ValueError(
            "Compressor bridge supports ratio 1 or ratio 2 with one/two banks"
        )
    key = ratio, banks
    if key in _ops:
        return _ops[key]
    function = _bridge(ratio, banks)
    tensor_count = 6 if ratio == 1 else 8 + banks
    op = pallas.jax_op(
        f"pallas::dsv41_compress_rope_r{ratio}_b{banks}",
        function,
        mesh=get_pallas_op_mesh(),
        donate_argnums=tuple(range(6, 6 + banks)),
        input_partition_specs=cast(Any, (P(),) * tensor_count),
    )
    op.register_fake(_fake(ratio, banks))
    _ops[key] = op
    return op
