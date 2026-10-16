# SPDX-License-Identifier: Apache-2.0

"""TorchTPU binding for shifted mHC with fused input RMS."""

from typing import Any

import jax
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return shifted_post_pre_rms.shifted_post_pre_rms(
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
    )


def initialize_shifted_mhc() -> None:
    global _post_pre_rms_op
    if _post_pre_rms_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _post_pre_rms_op = pallas.jax_op(
        "pallas::deepseek_v41_v2_shifted_post_pre_rms",
        _post_pre_rms_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 9 + (None,) * 6,
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
