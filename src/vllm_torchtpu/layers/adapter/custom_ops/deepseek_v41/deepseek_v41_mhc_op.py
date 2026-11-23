# SPDX-License-Identifier: Apache-2.0

"""TorchTPU binding for shifted mHC with fused input RMS."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v41.mhc import shifted_post_pre_rms

_post_pre_rms_op = None


def _unfused(
    x: jax.Array,
    flat: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    pre: jax.Array,
    norm_weight: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
    norm_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The same layer as XLA ops, for shapes the kernel does not take."""
    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import (
        delayed_post_pre_flat_jax,
    )

    residual, next_post, next_comb, value, next_pre = delayed_post_pre_flat_jax(
        x,
        flat,
        post,
        comb,
        weight,
        scale,
        base,
        pre,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        repeats,
    )
    normalized = value.astype(jnp.float32) * jax.lax.rsqrt(
        jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
        + norm_eps
    )
    value = (normalized * norm_weight.astype(jnp.float32)).astype(value.dtype)
    return residual, next_post, next_comb, value, next_pre


def _post_pre_rms_jax(
    x: jax.Array,
    flat: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    pre: jax.Array,
    norm_weight: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
    norm_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens, mult = pre.shape
    dim = flat.shape[1] // mult
    # Every op is per token, so the tokens run in 32-token mix groups, padded
    # when needed: the real rows are bitwise the kernel's rows. Up to one
    # tile (128 rows, whole 8-row tiles) the kernel moves only the real rows.
    # 16-token groups (240 rows: 20.8 us against 43.9 padded) round
    # differently, so they wait for their accuracy check.
    supported = shifted_post_pre_rms.is_supported_shape
    padded = tokens if tokens <= 128 and tokens % 8 == 0 else -(-tokens // 32) * 32
    if not supported(padded, mult, dim):
        return _unfused(
            x,
            flat,
            post,
            comb,
            weight,
            scale,
            base,
            pre,
            norm_weight,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            repeats,
            norm_eps,
        )

    def pad(a: jax.Array) -> jax.Array:
        # Zero rows: the real rows come out bitwise the same as with repeated
        # rows, without edge padding's per-row gathers (~0.14 ms per step).
        widths = [(0, padded - tokens)] + [(0, 0)] * (a.ndim - 1)
        return jnp.pad(a, widths)

    outputs = shifted_post_pre_rms.shifted_post_pre_rms(
        pad(x.reshape(tokens, -1).astype(flat.dtype)),
        pad(flat),
        pad(post),
        pad(comb),
        weight,
        scale,
        base,
        pad(pre),
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_repeat=repeats,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        **_gate_layout(),
    )
    if padded == tokens:
        return outputs
    return cast(Any, tuple(out[:tokens] for out in outputs))


def _gate_layout() -> dict:
    layout: dict = {"early_prep": envs.TPU_DEEPSEEK_V41_MHC_EARLY_PREP}
    if envs.TPU_DEEPSEEK_V41_MHC_RESIDENT_GATES:
        layout["unpadded_gates"] = True
        layout["gate_input_mode"] = shifted_post_pre_rms.GateInputMode.COMPILER
    return layout


def initialize_shifted_mhc() -> None:
    global _post_pre_rms_op
    if _post_pre_rms_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    _post_pre_rms_op = pallas.jax_op(
        "pallas::deepseek_v41_v2_shifted_post_pre_rms",
        _post_pre_rms_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], (P(),) * 9 + (None,) * 6
        ),
    )

    def fake(x, flat, post, comb, weight, scale, base, pre, *args):
        tokens, mult = pre.shape
        return (
            torch.empty_like(flat),
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    _post_pre_rms_op.register_fake(fake)


def get_shifted_post_pre_rms_op() -> Any:
    assert _post_pre_rms_op is not None
    return _post_pre_rms_op
