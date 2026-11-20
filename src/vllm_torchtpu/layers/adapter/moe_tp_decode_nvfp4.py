# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/moe_tp_decode_nvfp4.

Mirrors ``layers/adapter/moe_tp_decode.py`` for ``VllmNvfp4MoEMethod`` layers
behind ``USE_MOE_TP_DECODE_NVFP4_KERNEL`` (default off):
  - At ``n < 16``, pads ``x`` and ``scores`` to ``m = 16``, runs
    ``moe_tp_decode_nvfp4`` (W4A8 grouped FP4 decode kernel + in-kernel
    all-reduce), and slices ``[:n]``.
  - At ``n in (16, 32, 64)``, runs ``moe_tp_decode_nvfp4`` directly.
  - At ``n > 64`` (mixed / prefill buckets), runs ``moe_tp_decode._tp_chain``
    (``fused_moe_func`` -> ``gmm_v2`` W4A16 + online-FP8 shared expert) plus
    ``lax.psum`` on the same K-major FP4 weights and 4D block scales, with zero
    weight duplication.

Supports both EP x TP MoE (e.g. ``EP2xTP4`` via ``ENABLE_EXPERT_PARALLEL=true``
and ``TPU_MOE_EP_TP_SIZE=4``, where ``ep_size * tp_size == n_ranks`` with
linear expert placement) and plain TP MoE (``tp_size == n_ranks``).
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental import moe_tp_decode_nvfp4 as kernel
from vllm_torchtpu.layers.adapter import moe_tp_decode as fp8_decode
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to

logger = init_logger(__name__)

_ENV_FLAG = "USE_MOE_TP_DECODE_NVFP4_KERNEL"
_OP_ATTR = "_moe_tp_decode_nvfp4_op"
_BUFFERS = (
    "_moe_tp_decode_nvfp4_sw13",
    "_moe_tp_decode_nvfp4_ss13",
    "_moe_tp_decode_nvfp4_sw2",
    "_moe_tp_decode_nvfp4_ss2",
    "_moe_tp_decode_nvfp4_bias",
    "_moe_tp_decode_nvfp4_rank",
    "_moe_tp_decode_nvfp4_e0",
)

_OPS: dict[tuple, Any] = {}


@dataclasses.dataclass(frozen=True)
class _OpStatics(fp8_decode._OpStatics):
    split_meta: bool = True
    w2_n_chunk: int = 2048


def moe_tp_decode_nvfp4_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's NVFP4 quant method the fused TP op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def _proposed_routed_sds(
    layer: Any,
    *,
    requant_block: int,
    clamp_per_matrix: bool = True,
) -> (
    tuple[
        jax.ShapeDtypeStruct,
        jax.ShapeDtypeStruct,
        tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
    ]
    | str
):
    """Describe the K-major FP4 weights and 4D scales ``_prepare_moe_weights``
    will emit for ``layer`` before checkpoint tensors are mutated."""
    if requant_block <= 0:
        return f"invalid requant_block {requant_block}"
    w13_u8, w2_u8 = layer.w13_weight.data, layer.w2_weight.data
    experts = int(w13_u8.shape[0])
    hidden = int(w13_u8.shape[-1]) * 2
    inter = int(w2_u8.shape[-1]) * 2
    b13 = min(requant_block, hidden) if clamp_per_matrix else requant_block
    b2 = min(requant_block, inter) if clamp_per_matrix else requant_block
    if (
        hidden % b13 == 0
        and inter % b2 == 0
        and (not clamp_per_matrix or (b13 % 16 == 0 and b2 % 16 == 0))
    ):
        inter_out = inter
    else:
        if hidden % requant_block != 0:
            return (
                f"hidden ({hidden}) is not divisible by weight block ({requant_block})"
            )
        inter_out = align_to(inter, requant_block)
        b13 = b2 = requant_block
    return (
        jax.ShapeDtypeStruct((experts, hidden, 2 * inter_out), jnp.float4_e2m1fn),
        jax.ShapeDtypeStruct((experts, inter_out, hidden), jnp.float4_e2m1fn),
        (
            jax.ShapeDtypeStruct(
                (experts, hidden // b13, 1, 2 * inter_out), jnp.float32
            ),
            jax.ShapeDtypeStruct((experts, inter_out // b2, 1, hidden), jnp.float32),
        ),
    )


def _admit(
    runner: Any,
    n_ranks: int,
    *,
    proposed_routed: tuple[Any, Any, tuple[Any, Any]] | None = None,
    pre_load: bool = False,
) -> fp8_decode._Admitted | str:
    """Admission check for one ``MoERunner``, or a human-readable refusal."""
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod

    def _pre_check() -> str | None:
        if enable_pipelined_collective_and_compute():
            return "the MoE path owns its collectives (TPU_MOE_COLLECTION_CHUNK_SIZE)"
        return None

    def _get_scales(layer: Any) -> tuple[torch.Tensor, torch.Tensor] | str:
        w13_scale = getattr(layer, "w13_weight_scale", None)
        w2_scale = getattr(layer, "w2_weight_scale", None)
        if w13_scale is None or w2_scale is None:
            return "missing w13_weight_scale / w2_weight_scale"
        return (w13_scale, w2_scale)

    return fp8_decode._admit_common(
        runner,
        n_ranks,
        quant_cls=VllmNvfp4MoEMethod,
        pre_check=_pre_check,
        get_scales=_get_scales,
        row_buckets=kernel.ROW_BUCKETS,
        can_use_fn=kernel.can_use_moe_tp_decode_nvfp4,
        no_rows_msg=lambda w13, scales, shared: (
            f"the kernel admits no token count (w13 {list(w13.shape)} "
            f"{w13.dtype}, s13 {list(scales[0].shape)}, "
            f"s2 {list(scales[1].shape)}, shared w13 {list(shared[0].shape)})"
        ),
        build_statics=lambda base, _rnr, _hid, _topk, _scales, _shared: _OpStatics(
            **base,
            split_meta=bool(envs.TPU_MOE_TP_DECODE_SPLIT_META),
        ),
        proposed_routed=proposed_routed,
        pre_load=pre_load,
    )


def precheck_admission(
    layer: Any,
    *,
    requant_block: int,
    clamp_per_matrix: bool = True,
) -> str | None:
    """Return ``None`` if ``layer`` will arm after load-time requantization,
    else a human-readable refusal reason."""
    runner = fp8_decode.routed_experts_runner(layer)
    if runner is None:
        return "routed experts layer has no MoERunner"
    qm = layer.quant_method
    if moe_tp_decode_nvfp4_supported(qm):
        return None
    if fp8_decode.moe_tp_decode_supported(qm):
        return "moe_tp_decode already serves this layer"
    proposed = _proposed_routed_sds(
        layer,
        requant_block=requant_block,
        clamp_per_matrix=clamp_per_matrix,
    )
    if isinstance(proposed, str):
        return proposed
    topo = fp8_decode._topology()
    res = fp8_decode._admit_runner(
        runner,
        topo,
        lambda r, n: _admit(r, n, proposed_routed=proposed, pre_load=True),
        pre_load=True,
    )
    return res if isinstance(res, str) else None


def _op_body(
    statics: _OpStatics,
    n_ranks: int,
    axis: str,
    above_fn: Any = None,
):
    """Per-rank body under ``shard_map`` over ``axis``."""
    return fp8_decode._make_op_body(
        statics,
        n_ranks,
        axis,
        row_buckets=kernel.ROW_BUCKETS,
        can_use_fn=lambda x, scores, bias, *weights: kernel.can_use_moe_tp_decode_nvfp4(
            x,
            scores,
            bias,
            *weights,
            topk=statics.topk,
            n_ranks=n_ranks,
            w2_n_chunk=statics.w2_n_chunk,
        ),
        decode_fn=lambda x, scores, bias, n_valid, rank, *weights_and_e0: (
            kernel.moe_tp_decode_nvfp4(
                x,
                scores,
                bias,
                n_valid,
                rank,
                *weights_and_e0,
                routed_scale=statics.routed_scale,
                topk=statics.topk,
                n_ranks=n_ranks,
                fast=statics.fast,
                hbm_operands=statics.hbm_operands,
                split_meta=statics.split_meta,
                w2_n_chunk=statics.w2_n_chunk,
            )
        ),
        above_fn=above_fn,
    )


def _get_op(mesh: Any, statics: _OpStatics):
    """One ``pallas.jax_op`` per distinct closure, built at arming."""
    return fp8_decode._build_jax_op(
        _OPS,
        mesh,
        statics,
        op_prefix="pallas::moe_tp_decode_nvfp4",
        make_body=_op_body,
    )


def should_arm() -> bool:
    """Whether ``load_model`` should call ``arm`` for NVFP4 MoE layers."""
    return envs.USE_MOE_TP_DECODE_NVFP4_KERNEL


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm every admitted NVFP4 MoE layer of ``model`` and ``draft_model``."""
    return fp8_decode._arm_layers(
        model,
        draft_model,
        is_already_armed=lambda qm: (
            moe_tp_decode_nvfp4_supported(qm) or fp8_decode.moe_tp_decode_supported(qm)
        ),
        admit_fn=_admit,
        get_op_fn=lambda mesh, statics: _get_op(mesh, statics),
        op_attr=_OP_ATTR,
        buffers=_BUFFERS,
        env_flag=_ENV_FLAG,
        log_tag="moe_tp_decode_nvfp4",
        log_obj=logger,
    )


def moe_tp_decode_nvfp4_forward(
    quant_method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    """Forward of an armed NVFP4 layer: reduced ``shared + scale * routed``."""
    return fp8_decode._forward_armed(
        quant_method,
        layer,
        x,
        router_logits,
        op_attr=_OP_ATTR,
        buffers=_BUFFERS,
        scale_attrs=("w13_weight_scale", "w2_weight_scale"),
        log_tag="moe_tp_decode_nvfp4",
    )
