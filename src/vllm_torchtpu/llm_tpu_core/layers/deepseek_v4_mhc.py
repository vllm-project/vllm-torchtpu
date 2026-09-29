# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""DeepSeek-V4 mHC Pallas kernels and custom ops.

Decoupled hardware-adapted mHC execution primitives with strictly zero
dependencies on vLLM.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.mhc import (
    fused_post_pre_kernel,
    post_kernel,
    pre_kernel,
)

ATTN_DATA_AXIS = None
_mhc_op_cache: dict[str, Any] = {}


def _name_float(value: float) -> str:
    """A float rendered so it is safe inside a torch op name."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _gate_suffix(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> str:
    """The gate constants are baked into the traced op, so they name it."""
    return (
        f"_r{_name_float(rms_eps)}_p{_name_float(hc_pre_eps)}"
        f"_s{_name_float(hc_sinkhorn_eps)}"
        f"_a{_name_float(hc_post_mult_value)}_i{sinkhorn_repeat}"
    )


def _mhc_pre_jax(
    residual: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mix GEMM + stream collapse in Pallas, gates/Sinkhorn in XLA."""
    return pre_kernel.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _mhc_post_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """Recombine the sublayer output into the mHC residual streams."""
    return post_kernel.mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _mhc_fused_post_pre_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One layer's post and the next layer's pre as a single kernel."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )


def _fake_pre_outputs(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    device = residual.device
    return (
        torch.empty((*outer_shape, hc_mult, 1), dtype=torch.float32, device=device),
        torch.empty(
            (*outer_shape, hc_mult, hc_mult), dtype=torch.float32, device=device
        ),
        torch.empty((*outer_shape, hidden_size), dtype=residual.dtype, device=device),
    )


def _fake_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del fn, hc_scale, hc_base
    del rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    return _fake_pre_outputs(residual)


def _fake_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    del x, post_layer_mix, comb_res_mix
    return torch.empty_like(residual)


def _fake_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del x, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base
    del rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    return (torch.empty_like(residual), *_fake_pre_outputs(residual))


def _mhc_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    mesh: Any = None,
):
    op_name = "pallas::deepseek_v4_mhc_pre_v1" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_pre_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=mesh,
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_pre)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_post_op(mesh: Any = None):
    """The ``post`` op alone, for callers that settle a deferred post."""
    op_name = "pallas::deepseek_v4_mhc_post_v1"
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            _mhc_post_jax,
            mesh=mesh,
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
            ),
        )
        op.register_fake(_fake_mhc_post)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_fused_post_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    mesh: Any = None,
):
    op_name = "pallas::deepseek_v4_mhc_fused_post_pre_v1" + _gate_suffix(
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_fused_post_pre_jax,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
            ),
            mesh=mesh,
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        op.register_fake(_fake_mhc_fused_post_pre)
        _mhc_op_cache[op_name] = op
    return op


class MHCOps(NamedTuple):
    pre: Any
    fused: Any
    post: Any


def get_mhc_ops(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    mesh: Any = None,
) -> MHCOps:
    return MHCOps(
        pre=_mhc_pre_op(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            mesh=mesh,
        ),
        fused=_mhc_fused_post_pre_op(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            mesh=mesh,
        ),
        post=get_mhc_post_op(mesh=mesh),
    )


__all__ = [
    "MHCOps",
    "get_mhc_ops",
    "get_mhc_post_op",
]
