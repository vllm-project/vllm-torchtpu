# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/moe_tp_decode_nvfp4.

Mirrors ``layers/adapter/moe_tp_decode.py`` for ``VllmNvfp4MoEMethod`` layers
behind ``USE_MOE_TP_DECODE_NVFP4_KERNEL`` (default off):
  - At ``n < 16``, pads ``x`` and ``scores`` to ``m = 16``, runs
    ``moe_tp_decode_nvfp4`` (W4A8 grouped FP4 decode kernel + in-kernel
    all-reduce), and slices ``[:n]``.
  - At ``n in (16, 32, 64)``, runs ``moe_tp_decode_nvfp4`` directly.
  - At ``n > 64`` (mixed / prefill buckets), runs
    ``fused_moe_prefill_nvfp4_stage_a`` (W4A16 default, Stage-A whole-expert
    ring + fused online-FP8 shared expert + chunked RS/AG combine) when
    ``FUSED_MOE_PREFILL_NVFP4_KERNEL`` is enabled, the layer and traced
    activations admit it, and ``n`` meets the mode cutoff (``n >= 2048`` for
    ``"w4a16"``, ``n >= 512`` for ``"w4a8"``);
    otherwise runs ``moe_tp_decode._tp_chain`` (``fused_moe_func`` ->
    ``gmm_v2`` W4A16 + online-FP8 shared expert) plus ``lax.psum`` on the same
    K-major FP4 weights and 4D block scales, with zero weight duplication.

Supports both EP x TP MoE (e.g. ``EP2xTP4`` via ``ENABLE_EXPERT_PARALLEL=true``
and ``TPU_MOE_EP_TP_SIZE=4``, where ``ep_size * tp_size == n_ranks`` with
linear expert placement) and plain TP MoE (``tp_size == n_ranks``).
"""

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental import (
    fused_moe_prefill_nvfp4 as prefill_nvfp4,
)
from vllm_torchtpu.kernels.experimental import moe_tp_decode_nvfp4 as kernel
from vllm_torchtpu.layers.adapter import moe_tp_decode as fp8_decode
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ENV_FLAG = "USE_MOE_TP_DECODE_NVFP4_KERNEL"
_OP_ATTR = "_moe_tp_decode_nvfp4_op"
_PREFILL_PROBE_ROWS = 128
_PREFILL_MIN_TOKENS: dict[prefill_nvfp4.ActMode, int] = {
    "w4a16": 2048,
    "w4a8": 512,
}
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
    prefill_act_mode: prefill_nvfp4.ActMode = "w4a16"
    split_meta: bool = True
    w2_n_chunk: int = 2048


def moe_tp_decode_nvfp4_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's NVFP4 quant method the fused TP op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def _admit(runner: Any, n_ranks: int) -> fp8_decode._Admitted | str:
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

    def _build_statics(
        base: dict[str, Any],
        rnr: Any,
        hidden: int,
        topk: int,
        scales: tuple[torch.Tensor, torch.Tensor],
        shared: tuple[torch.Tensor, ...],
    ) -> _OpStatics:
        layer = rnr.routed_experts
        w13_scale, w2_scale = scales
        sw13_t, ss13_t, sw2_t, ss2_t = shared
        prefill_above: int | None = None
        prefill_act_mode: prefill_nvfp4.ActMode = "w4a16"
        if envs.FUSED_MOE_PREFILL_NVFP4_KERNEL:
            cand_act_mode = cast(
                prefill_nvfp4.ActMode, envs.FUSED_MOE_PREFILL_NVFP4_ACT_MODE
            )
            reason = prefill_nvfp4.fused_moe_prefill_nvfp4_unsupported_reason(
                jax.ShapeDtypeStruct((_PREFILL_PROBE_ROWS, hidden), jnp.bfloat16),
                fp8_decode._sds(layer.w13_weight),
                fp8_decode._sds(layer.w2_weight),
                fp8_decode._sds(w13_scale),
                fp8_decode._sds(w2_scale),
                fp8_decode._sds(sw13_t),
                fp8_decode._sds(ss13_t),
                fp8_decode._sds(sw2_t),
                fp8_decode._sds(ss2_t),
                jax.ShapeDtypeStruct((_PREFILL_PROBE_ROWS, topk), jnp.bfloat16),
                jax.ShapeDtypeStruct((_PREFILL_PROBE_ROWS, topk), jnp.int32),
                act_mode=cand_act_mode,
            )
            if reason is None:
                min_tokens = _PREFILL_MIN_TOKENS[cand_act_mode]
                prefill_above = max(kernel.ROW_BUCKETS[-1], min_tokens - 1)
                prefill_act_mode = cand_act_mode
            else:
                logger.warning_once(
                    "moe_tp_decode_nvfp4: FUSED_MOE_PREFILL_NVFP4_KERNEL=1 "
                    "refused prefill (act_mode=%s, w13=%s %s, w2=%s, "
                    "s13=%s, s2=%s, shared_w13=%s): %s",
                    cand_act_mode,
                    tuple(layer.w13_weight.shape),
                    str(layer.w13_weight.dtype),
                    tuple(layer.w2_weight.shape),
                    tuple(w13_scale.shape),
                    tuple(w2_scale.shape),
                    tuple(sw13_t.shape),
                    reason,
                )
        return _OpStatics(
            **base,
            prefill_above=prefill_above,
            prefill_act_mode=prefill_act_mode,
            split_meta=bool(envs.TPU_MOE_TP_DECODE_SPLIT_META),
        )

    return fp8_decode._admit_common(
        runner,
        n_ranks,
        quant_cls=VllmNvfp4MoEMethod,
        pre_check=_pre_check,
        get_scales=_get_scales,
        row_buckets=kernel.ROW_BUCKETS,
        can_use_fn=kernel.can_use_moe_tp_decode_nvfp4,
        no_rows_msg=lambda layer, scales, shared: (
            f"the kernel admits no token count (w13 {list(layer.w13_weight.shape)} "
            f"{layer.w13_weight.dtype}, s13 {list(scales[0].shape)}, "
            f"s2 {list(scales[1].shape)}, shared w13 {list(shared[0].shape)})"
        ),
        build_statics=_build_statics,
    )


def _prefill_chain(
    x,
    scores,
    bias,
    is_pad,
    w13,
    w2,
    s13,
    s2,
    sw13,
    ss13,
    sw2,
    ss2,
    st: _OpStatics,
    axis: str,
    e0,
):
    """Reduced layer output via ``fused_moe_prefill_nvfp4_stage_a``."""
    logger.info(
        "moe_tp_decode_nvfp4: fused MoE prefill kernel (%s) traced for %d tokens",
        st.prefill_act_mode,
        x.shape[0],
    )
    w, ids = fp8_decode._route(x, scores, bias, is_pad, st)
    return prefill_nvfp4.fused_moe_prefill_nvfp4_stage_a(
        x,
        w13,
        w2,
        s13,
        s2,
        sw13,
        ss13,
        sw2,
        ss2,
        None,
        w,
        ids.astype(jnp.int32),
        axis_name=axis,
        expert_start=e0.astype(jnp.int32) if st.use_ep else None,
        routed_scale=st.routed_scale,
        act_mode=st.prefill_act_mode,
        w2_n_chunk=st.w2_n_chunk,
    )


def _op_body(statics: _OpStatics, n_ranks: int, axis: str):
    """Per-rank body under ``shard_map`` over ``axis``."""

    def _above(
        x,
        scores,
        bias,
        is_pad,
        w13,
        w2,
        s13,
        s2,
        sw13,
        ss13,
        sw2,
        ss2,
        *,
        statics,
        axis,
        e0,
        prequant,
    ):
        del prequant
        if not prefill_nvfp4.can_use_fused_moe_prefill_nvfp4(
            x,
            w13,
            w2,
            s13,
            s2,
            sw13,
            ss13,
            sw2,
            ss2,
            jax.ShapeDtypeStruct((x.shape[0], statics.topk), jnp.bfloat16),
            jax.ShapeDtypeStruct((x.shape[0], statics.topk), jnp.int32),
            act_mode=statics.prefill_act_mode,
            w2_n_chunk=statics.w2_n_chunk,
        ):
            partial = fp8_decode._tp_chain(
                x,
                scores,
                bias,
                is_pad,
                w13,
                w2,
                s13,
                s2,
                sw13,
                ss13,
                sw2,
                ss2,
                statics,
                e0.reshape(()),
            )
            return jax.lax.psum(partial, axis)
        return _prefill_chain(
            x,
            scores,
            bias,
            is_pad,
            w13,
            w2,
            s13,
            s2,
            sw13,
            ss13,
            sw2,
            ss2,
            statics,
            axis,
            e0.reshape(()),
        )

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
        above_fn=_above,
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
