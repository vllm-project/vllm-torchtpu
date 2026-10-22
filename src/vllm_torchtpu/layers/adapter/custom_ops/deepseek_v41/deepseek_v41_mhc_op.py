# SPDX-License-Identifier: Apache-2.0

"""TorchTPU binding for shifted mHC with fused input RMS."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41.mhc import shifted_post_pre_rms

_post_pre_rms_op = None


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
    moe_fp8: bool = False,
) -> list[jax.Array]:
    tokens, mult = pre.shape
    dim = flat.shape[1] // mult
    if not shifted_post_pre_rms.is_supported_shape(tokens, mult, dim):
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
        quantized = []
        if moe_fp8:
            amax = jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1, keepdims=True)
            moe_scale = amax / 448.0
            valid = (moe_scale > 0) & jnp.isfinite(moe_scale)
            moe_scale = jnp.where(valid, moe_scale, 0.0)
            reciprocal = jnp.where(
                valid, 1.0 / jnp.where(valid, moe_scale, 1.0), 0.0
            ).astype(jnp.bfloat16)
            moe_q = jnp.where(valid, value * reciprocal, 0).astype(jnp.float8_e4m3fn)
            quantized.extend((moe_q, moe_scale))
        return [residual, next_post, next_comb, value, next_pre, *quantized]
    outputs = shifted_post_pre_rms.shifted_post_pre_rms(
        x.reshape(flat.shape[0], -1).astype(flat.dtype),
        flat,
        post,
        comb,
        weight,
        scale,
        base,
        pre,
        rms_eps=rms_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=post_multiplier,
        sinkhorn_repeat=repeats,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        moe_fp8=moe_fp8,
    )
    return list(outputs)


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
        # Runtime accepts PartitionSpec and None for static arguments.
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], (P(),) * 9 + (None,) * 8
        ),
    )

    def fake(
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
        moe_fp8=False,
    ):
        tokens, mult = pre.shape
        value = flat.new_empty((tokens, flat.shape[1] // mult))
        outputs = [
            torch.empty_like(flat),
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            value,
            scale.new_empty((tokens, mult)),
        ]
        for enabled in (moe_fp8,):
            if enabled:
                outputs.extend(
                    [
                        torch.empty_like(value, dtype=torch.float8_e4m3fn),
                        scale.new_empty((tokens, 1), dtype=torch.float32),
                    ]
                )
        return outputs

    _post_pre_rms_op.register_fake(fake)


def get_shifted_post_pre_rms_op() -> Any:
    assert _post_pre_rms_op is not None
    return _post_pre_rms_op
