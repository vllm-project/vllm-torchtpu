# SPDX-License-Identifier: Apache-2.0
"""Torch ops for the fused DeepSeek-V4.1 attention-preprocessing kernels.

``pallas.jax_op`` requires every parameter and the return of a bridged
function to be an annotated array, so each kernel mode gets its own explicit
signature (no ``*args``, keyword parameters or ``Any`` return); epsilon and
the compression ratio are bound per op.
"""

import functools
import inspect
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_rope as compress_kernel,
)
from vllm_torchtpu.kernels.attention_preprocessing import qa_rope as qa_kernel
from vllm_torchtpu.kernels.attention_preprocessing import qb_rope as qb_kernel
from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_HEAD_DIM = 512
_ops: dict[str, Any] = {}


def _get(name, build, donate=()):
    """``pallas::dsv41_<name>``, registered from ``build() -> (fn, fake)`` once."""
    op = _ops.get(name)
    if op is None:
        function, fake = build()
        op = pallas.jax_op(
            f"pallas::dsv41_{name}",
            function,
            mesh=get_pallas_op_mesh(),
            donate_argnums=donate,
            input_partition_specs=cast(
                Any, (P(),) * len(inspect.signature(function).parameters)
            ),
        )
        op.register_fake(fake)
        _ops[name] = op
    return op


def _tag(eps: float) -> str:
    return float(eps).hex().replace(".", "_").replace("+", "p").replace("-", "m")


def rope_rows() -> Any:
    """``(cos_sin_cache, positions) -> [T, rotary_dim]`` FP32 rows, clamped."""
    return _get(
        "rope_rows",
        lambda: (
            rope_gather.gather_rows,
            lambda cache, pos: cache.new_empty((pos.shape[0], cache.shape[1])),
        ),
    )


def _qa(x, weight, scale, gamma_qr, gamma_kv, rows, *, eps, emit_qr):
    if not x.shape[0]:  # empty dummy batches
        return (
            jnp.zeros((0, gamma_qr.shape[0]), jnp.bfloat16),
            jnp.zeros((0, gamma_kv.shape[0]), jnp.bfloat16),
        )
    return qa_kernel.fused_qa_kv_norm_rope(
        x, weight, scale, gamma_qr, gamma_kv, rows, eps=eps, emit_qr=emit_qr
    )


def _qa_bridge(emit_qr: bool, eps: float):
    run = functools.partial(_qa, eps=eps, emit_qr=emit_qr)
    if emit_qr:

        def qa_kv_norm_rope(
            x: jax.Array,
            weight: jax.Array,
            weight_scale: jax.Array,
            gamma_qr: jax.Array,
            gamma_kv: jax.Array,
            rope_rows: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return run(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows)

        return qa_kv_norm_rope

    def kv_norm_rope(
        x: jax.Array,
        weight: jax.Array,
        weight_scale: jax.Array,
        gamma_qr: jax.Array,
        gamma_kv: jax.Array,
        rope_rows: jax.Array,
    ) -> jax.Array:
        return run(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows)[1]

    return kv_norm_rope


def _qa_fake(emit_qr):
    def fake(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows):
        kv = x.new_empty((x.shape[0], gamma_kv.shape[0]))
        return (x.new_empty((x.shape[0], gamma_qr.shape[0])), kv) if emit_qr else kv

    return fake


def qa_rope(emit_qr: bool, eps: float) -> Any:
    """``(x, w, scale, gamma_qr, gamma_kv, rope_rows) -> (qr, kv)`` or ``kv``.

    KV leaves rotated; ``emit_qr=False`` projects only the KV columns.
    """
    return _get(
        f"qa_kv_norm_rope{'' if emit_qr else '_kv'}_e{_tag(eps)}",
        lambda: (_qa_bridge(emit_qr, eps), _qa_fake(emit_qr)),
    )


def _qb(
    qr: jax.Array, weight: jax.Array, weight_scale: jax.Array, rope_rows: jax.Array
) -> jax.Array:
    if not qr.shape[0]:
        return jnp.zeros((0, weight.shape[1] // _HEAD_DIM, _HEAD_DIM), jnp.bfloat16)
    return qb_kernel.fused_qb_rope(qr, weight, weight_scale, rope_rows)


def _qb_fake(qr, weight, weight_scale, rope_rows):
    return qr.new_empty((qr.shape[0], weight.shape[1] // _HEAD_DIM, _HEAD_DIM))


def qb_rope() -> Any:
    """``(qr, w, scale, rope_rows) -> q [T, heads, 512]`` with RoPE applied."""
    return _get("qb_rope", lambda: (_qb, _qb_fake))


def _compress(ratio, eps, x, weight, gamma, rows, has_prev, valid, *state_and_ring):
    if not x.shape[0]:
        state = state_and_ring[0] if state_and_ring else None
        return (
            jnp.zeros((0, 512), jnp.bfloat16),
            jnp.zeros((0, 128), jnp.int32),
            jnp.zeros((0, 1), jnp.float32),
            state,
        )
    return compress_kernel.fused_compress_rope(
        x, weight, gamma, rows, has_prev, valid, *state_and_ring, ratio=ratio, eps=eps
    )


def _compress_bridge(ratio: int, banks: int, eps: float):
    run = functools.partial(_compress, ratio, eps)
    if ratio == 1:

        def compress_rope_ratio1(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rope_rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            latent, words, scales, _ = run(x, weight, gamma, rope_rows, has_prev, valid)
            return latent, words, scales

        return compress_rope_ratio1
    if banks == 1:

        def compress_rope_ratio2(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rope_rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
            state: jax.Array,
            ring_write: jax.Array,
            ring_read: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return run(
                x,
                weight,
                gamma,
                rope_rows,
                has_prev,
                valid,
                state,
                ring_write,
                ring_read,
            )

        return compress_rope_ratio2

    def compress_rope_ratio2_banked(
        x: jax.Array,
        weight: jax.Array,
        gamma: jax.Array,
        rope_rows: jax.Array,
        has_prev: jax.Array,
        valid: jax.Array,
        state_low: jax.Array,
        state_high: jax.Array,
        ring_write: jax.Array,
        ring_read: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        latent, words, scales, state = run(
            x,
            weight,
            gamma,
            rope_rows,
            has_prev,
            valid,
            (state_low, state_high),
            ring_write,
            ring_read,
        )
        assert isinstance(state, tuple)
        low, high = state
        return latent, words, scales, low, high

    return compress_rope_ratio2_banked


def _compress_fake(banks):
    def fake(x, weight, gamma, rope_rows, has_prev, valid, *state_and_ring):
        latent = x.new_empty((x.shape[0], 512), dtype=torch.bfloat16)
        words = x.new_empty((x.shape[0], 128), dtype=torch.int32)
        scales = x.new_empty((x.shape[0], 1), dtype=torch.float32)
        states = tuple(torch.empty_like(s) for s in state_and_ring[:banks])
        return latent, words, scales, *states

    return fake


def compress_rope(ratio: int, banks: int, eps: float) -> Any:
    """``(x, w, gamma, rope_rows, has_prev, valid[, *state, write, read])``.

    Returns ``(latent, CSA2 words, scales, *updated state banks)``; ratio 2
    updates its ``banks`` compressor state pools in place (donated).
    """
    if (ratio, banks) not in ((1, 0), (2, 1), (2, 2)):
        raise ValueError("Use ratio 1 without state, or ratio 2 with one/two banks")
    return _get(
        f"compress_rope_r{ratio}_b{banks}_e{_tag(eps)}",
        lambda: (_compress_bridge(ratio, banks, eps), _compress_fake(banks)),
        donate=tuple(range(6, 6 + banks)),
    )
