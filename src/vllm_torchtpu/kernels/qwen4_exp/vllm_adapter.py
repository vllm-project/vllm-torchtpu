# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue


@functools.cache
def build_gated_residual_op(branches: int, hidden: int, token_tile: int):
    op = pallas.jax_op(
        f"qwen4_exp::gated_residual_b{branches}_h{hidden}_m{token_tile}_v1",
        gated_residual_fn(branches, hidden, token_tile),
    )

    @op.register_fake
    def fake(residual, block, gates):
        return residual.new_empty(residual.shape)

    return op


@functools.cache
def build_full_mix_op(branches: int, hidden: int, eps: float, token_tile: int):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _full_mix(
            x,
            weight,
            down,
            up,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty(x.shape, dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_norm_op(branches: int, hidden: int, eps: float, token_tile: int):
    """Register a local Pallas op for complete prefill token tiles."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(input_tensor: jax.Array, weight: jax.Array) -> jax.Array:
        return _normalize(
            input_tensor,
            weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_norm_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}_v2",
        core,
    )

    @op.register_fake
    def fake(input_tensor, weight):
        return input_tensor.new_empty(input_tensor.shape)

    return op


@functools.cache
def build_up_epilogue_op(branches: int, hidden: int, token_tile: int):
    """Register a local JAX op; callers gate it to whole prefill tiles."""

    def core(mid: jax.Array, norm: jax.Array, weight: jax.Array) -> jax.Array:
        return _up_epilogue(
            mid, norm, weight, branches=branches, hidden=hidden, token_tile=token_tile
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_up_epilogue_c{branches}_h{hidden}_m{token_tile}_v1", core
    )

    @op.register_fake
    def fake(mid, norm, weight):
        return mid.new_empty((mid.shape[0], hidden), dtype=torch.bfloat16)

    return op
