# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/moe_tp_decode (USE_MOE_TP_DECODE_KERNEL).

Under MoE tensor parallelism vLLM runs the shared expert, scales and adds the
routed output, and all-reduces the sum over the TP group. For an armed layer
one torch op does all of it: at the token counts the kernel admits (16, 32 and 64)
one Pallas call per rank covers routing, the shared expert, the routed experts
and the all-reduce; every other count runs the usual TP path (router top-k,
fused_moe_func, the shared expert's linears) and a psum in the same exported
program. The token count is static there, so one torch op serves every bucket.
The op's output is reduced, so arming also takes the shared expert and the
routed scale off the runner, and VllmMoERunner._fused_output_is_reduced turns
the runner's all-reduce off for that layer.

`arm` runs once at the end of TPUModelRunner.load_model, when the weights,
the MTP drafter and the token-padding mask exist. It is collective over the TP
group: every rank must arm the same layers (a rank that skips the kernel would
leave its peers waiting in its barrier), so the ranks compare an arming
signature and raise on a mismatch, then run the kernel's all-reduce once on
known inputs through the op's mesh and rank operand.

Expert range: with --enable-expert-parallel and TPU_MOE_EP_TP_SIZE (EP between
groups of consecutive ranks, TP inside one), a rank holds its group's block of
experts at its TP slice of I. The kernel then takes the block's first global
expert id (runtime data) and computes those experts only; routing still runs
over every expert and the in-kernel all-reduce still sums every rank of the
op's mesh (the whole EP x TP world), whose partials together cover every
(expert, column) once. Other token counts run the layer's EP path
(fused_moe_func with use_ep and experts_start: SparseCore ragged gather, local
grouped GEMMs, ragged gather-reduce) plus the shared expert and a psum, so
prefill and mixed batches keep the EP x TP routed kernels.

Prefill kernel: with FUSED_MOE_PREFILL_KERNEL=1, a TP-expert layer (no EP)
runs the steps of more than FUSED_MOE_DECODE_KERNEL_MAX_TOKENS tokens through
kernels/experimental/fused_moe_prefill (routed and shared experts, combine and
all-reduce) instead of the usual path and a psum: the same kernel and the same
cut-over as under FUSED_MOE_DECODE_KERNEL, which this op stands down for.
FUSED_MOE_PREFILL_KERNEL_V2 swaps in its stage-A build (per-row activation
scales) where the layer's experts fit whole.
"""

import dataclasses
import faulthandler
import hashlib
import weakref
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.experimental import moe_tp_decode as kernel
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill,
    plan_blocks,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel_v2 import (  # noqa: E501
    fused_moe_prefill_stage_a,
)
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.kernels.router_topk import select as router_select
from vllm_torchtpu.kernels.router_topk import select_bias as router_select_bias
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OP_ATTR = "_moe_tp_decode_op"
# TPU_FUSED_ADD_RMSNORM_QUANT_MOE: the op variant that also takes the norm's fp8
# rows and scales (set by rmsnorm.arm_prequant_moe), and the slot it reads them
# from on the FusedMoE layer.
_OP_PREQUANT_ATTR = "_moe_tp_decode_op_prequant"
PREQUANT_MOE_ATTR = "_tpu_prequant_moe_slot"
# Non-persistent buffers of an armed layer: the shared expert's tensors (the
# parameters themselves, not views: a view handed to the op is re-materialized
# on every call), the f32 correction bias and this rank's mesh index.
_BUFFERS = (
    "_moe_tp_decode_sw13",
    "_moe_tp_decode_ss13",
    "_moe_tp_decode_sw2",
    "_moe_tp_decode_ss2",
    "_moe_tp_decode_bias",
    "_moe_tp_decode_rank",
    "_moe_tp_decode_e0",
)
# A wrong rank or mesh hangs the self-test's all-reduce: dump every stack and
# exit instead.
SELFTEST_TIMEOUT_S = 300
_SUM_FLOOR = 1e-20
# fused_moe_prefill quantizes x per 512 columns; the hidden size must divide.
_PREFILL_Q_BLOCK = 512

_OPS: dict[tuple, Any] = {}
_SELFTEST_OPS: dict[all_reduce.AllReduceSpec, Any] = {}
_RANK_BUFFER: torch.Tensor | None = None
# TPU_MOE_TP_DECODE_SHARED_META: one experts_start buffer per (device, value).
_E0_BUFFERS: dict[tuple[str, int], torch.Tensor] = {}
_SELFTESTED_BUCKETS: set[tuple[int, int]] = set()


@dataclasses.dataclass(frozen=True)
class _Topology:
    mesh: Any
    mesh_index: int


@dataclasses.dataclass(frozen=True)
class _OpStatics:
    topk: int
    routed_scale: float
    activation: str
    use_sparse_core: bool
    onehot_threshold: int
    # Expert range: weights hold a linear block of experts per EP group, and
    # the psum path runs the EP routed path (use_ep, experts_start).
    use_ep: bool = False
    # TPU_MOE_ROUTER_TOPK_BIAS: the psum path picks experts as the layer does
    # (router_topk.select_bias; same ids and weights as select on scores+bias).
    router_bias: bool = False
    # FUSED_MOE_PREFILL_KERNEL on a TP-expert layer: token counts above this
    # run the fused MoE prefill kernel instead of the psum path. None = off.
    prefill_above: int | None = None
    # TPU_MOE_TP_DECODE_FAST: the kernel's fast schedule (same output bits).
    fast: bool = False
    # TPU_MOE_TP_DECODE_HBM_OPERANDS: the kernel's 8 weight operands constrained
    # to HBM (it DMAs them itself; no XLA VMEM staging copy). Same bits.
    hbm_operands: bool = False


@dataclasses.dataclass(frozen=True)
class _Admitted:
    shared: tuple[Any, ...]  # sw13, ss13, sw2, ss2
    rows: tuple[int, ...]  # the token counts the kernel admits
    hidden: int
    statics: _OpStatics
    experts_start: int  # first global expert of this rank's block (0 under TP)


_RUNNER_REF_ATTR = "_tpu_moe_runner_ref"


def bind_routed_experts_runner(runner: Any) -> None:
    """Link ``runner.routed_experts`` back to ``runner`` via a weakref so
    load-time weight hooks can run the same runner-level admission check as
    ``arm()`` without creating an ``nn.Module`` cycle."""
    routed = getattr(runner, "routed_experts", None)
    if routed is not None:
        setattr(routed, _RUNNER_REF_ATTR, weakref.ref(runner))


def routed_experts_runner(layer: Any) -> Any | None:
    """Return the ``MoERunner`` owning ``layer``, or ``None`` if unbound."""
    ref = getattr(layer, _RUNNER_REF_ATTR, None)
    return None if ref is None else ref()


def moe_tp_decode_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's quant method the op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def _sds(t: torch.Tensor | jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    if isinstance(t, jax.ShapeDtypeStruct):
        return t
    shape = tuple(t.shape)
    if t.dtype == torch.float4_e2m1fn_x2 and shape:
        shape = (*shape[:-1], shape[-1] * 2)
    return jax.ShapeDtypeStruct(shape, pallas.pallas.TORCH_TO_JAX_DTYPE_MAP[t.dtype])


def _topology() -> _Topology | str:
    """The op's 1-D mesh over the TP group and this rank's index in it, or a
    refusal reason. Collective: build_ep_mesh gathers device ids."""
    try:
        tp = parallel_state.get_tp_group()
    except (AssertionError, AttributeError, RuntimeError):
        return "the TP group is not initialized"
    ep = get_ep_group()
    if ep is None:
        return "no EP group (the TP group has one device)"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    ep_ranks = tuple(int(r) for r in ep.ranks)
    if tp_ranks != ep_ranks:
        return f"TP ranks {tp_ranks} != EP ranks {ep_ranks} (needs dp 1, pcp 1)"
    try:
        world = torch.distributed.get_world_size()
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        return "torch.distributed is not initialized"
    if len(tp_ranks) != world:
        return f"the TP group has {len(tp_ranks)} of {world} ranks"
    mesh = build_ep_mesh()
    mesh_index = ep_mesh_index()
    assert mesh is not None and mesh_index is not None  # the EP group exists
    return _Topology(mesh=mesh, mesh_index=mesh_index)


def _unflipped_fp8_linear_sds(
    lin: Any,
) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct] | str:
    """Predict post-load ``(weight, weight_scale)`` shapes/dtypes for a
    shared-expert ``VllmFp8LinearMethodTPU`` whose ``process_weights_after_loading``
    has not run yet (``MoERunner`` registers ``routed_experts`` before
    ``_shared_experts``)."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import _get_linear_quant_config

    qm = lin.quant_method
    quant_cfg = getattr(qm, "quant_config", None)
    online_fp8 = bool(getattr(qm, "online_fp8", False))
    linear_quant_cfg = getattr(qm, "_linear_quant_config", None)
    if quant_cfg is None and not online_fp8 and linear_quant_cfg is None:
        return "a shared-expert weight is not K-major"
    if getattr(lin, "weight", None) is None or lin.weight.dim() != 2:
        return "a shared-expert weight is not K-major"
    n_out, k_in = (int(d) for d in lin.weight.shape)
    is_fp8_ckpt = bool(getattr(quant_cfg, "is_checkpoint_fp8_serialized", False))
    if not is_fp8_ckpt and online_fp8:
        policy = getattr(qm, "online_fp8_policy", None)
        if lin.weight.dtype != torch.bfloat16 or policy is None:
            return "the shared expert is not FP8 with per-channel scales"
        if policy.weight_scheme == "per_block":
            return "the shared expert is not FP8 with per-channel scales"
        scale_shape = (1,) if policy.weight_scheme == "per_tensor" else (n_out,)
    else:
        if linear_quant_cfg is None:
            linear_quant_cfg = _get_linear_quant_config(
                getattr(qm, "linear_config", None)
            )
        _name, requant_dtype, requant_block_size, blockwise_kernel = linear_quant_cfg
        if requant_block_size is not None and k_in < requant_block_size:
            blockwise_kernel = False
        if requant_dtype != torch.float8_e4m3fn or blockwise_kernel:
            return "the shared expert is not FP8 with per-channel scales"
        scale_shape = (n_out,)
    return (
        jax.ShapeDtypeStruct((k_in, n_out), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct(scale_shape, jnp.float32),
    )


def _shared_slice(runner: Any, *, pre_load: bool = False) -> tuple[Any, ...] | str:
    """This rank's shared-expert tensors (w13 [D, 2Fs], s13 [2Fs], w2 [Fs, D],
    s2 [D]), or a refusal reason."""
    from vllm_torchtpu.layers.adapter.linear_common import (
        KEEP_VLLM_LAYOUT_ATTR,
        WEIGHT_FLIPPED_ATTR,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    if runner.shared_experts is None:
        return "no shared expert"
    mlp = runner.shared_experts._layer
    gate_up = getattr(mlp, "gate_up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if gate_up is None or down is None:
        return "the shared expert has no gate_up_proj / down_proj"
    out: list[Any] = []
    for lin in (gate_up, down):
        if not isinstance(lin.quant_method, VllmFp8LinearMethodTPU):
            return "a shared-expert linear is not VllmFp8LinearMethodTPU"
        flipped = getattr(lin, WEIGHT_FLIPPED_ATTR, False)
        if getattr(lin, KEEP_VLLM_LAYOUT_ATTR, False) or (not flipped and not pre_load):
            return "a shared-expert weight is not K-major"
        if lin.bias is not None:
            return "a shared-expert linear has a bias"
        if flipped:
            if lin.weight.dtype != torch.float8_e4m3fn or lin.weight_scale.dim() != 1:
                return "the shared expert is not FP8 with per-channel scales"
            out.extend((lin.weight.data, lin.weight_scale.data))
        else:
            predicted = _unflipped_fp8_linear_sds(lin)
            if isinstance(predicted, str):
                return predicted
            out.extend(predicted)
    if down.reduce_results:
        return "the shared expert reduces its own output"
    return tuple(out)


def _admit_common(
    runner: Any,
    n_ranks: int,
    *,
    quant_cls: type,
    pre_check: Callable[[], str | None],
    get_scales: Callable[[Any], tuple[Any, Any] | str],
    row_buckets: tuple[int, ...],
    can_use_fn: Callable[..., bool],
    no_rows_msg: Callable[[Any, tuple[Any, Any], tuple[Any, ...]], str],
    build_statics: Callable[
        [
            dict[str, Any],
            Any,
            int,
            int,
            tuple[Any, Any],
            tuple[Any, ...],
        ],
        _OpStatics | str,
    ],
    proposed_routed: tuple[Any, Any, tuple[Any, Any]] | None = None,
    pre_load: bool = False,
) -> _Admitted | str:
    """Shared admission checks across TP decode MoE adapters."""
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )

    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

    layer = runner.routed_experts
    qm = layer.quant_method
    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    if type(qm) is not quant_cls:
        return f"quant method {type(qm).__name__} is not {quant_cls.__name__}"
    if qm._tpu_activation_str != "silu":
        return f"activation {qm._tpu_activation_str!r} is not silu"
    if qm._use_adaptive_fused_moe or fused_moe_ep_supported(qm):
        return "a fused EP MoE kernel serves this layer"
    reason = pre_check()
    if reason is not None:
        return reason
    if par.use_ep:
        # Expert range over EP x TP weights: the partials of all ep x tp ranks
        # meet in the kernel's all-reduce over the op's n_ranks mesh.
        if par.ep_size * par.tp_size != n_ranks:
            return (
                f"needs EP x TP MoE over {n_ranks} ranks "
                f"(ep {par.ep_size}, tp {par.tp_size})"
            )
        if getattr(layer, "expert_placement_strategy", "linear") != "linear":
            return "EP placement is not linear"
        local_experts = (
            proposed_routed[0].shape[0]
            if proposed_routed is not None
            else layer.w13_weight.shape[0]
        )
        if local_experts * par.ep_size != layer.global_num_experts:
            return (
                f"{local_experts} local experts x ep {par.ep_size} != "
                f"{layer.global_num_experts} experts"
            )
        experts_start = moe_routing.get_experts_start(layer)
        assert experts_start is not None  # use_ep
    elif par.tp_size != n_ranks:
        return f"needs TP MoE over {n_ranks} ranks (ep {par.ep_size}, tp {par.tp_size})"
    else:
        experts_start = 0
    if par.dp_size != 1 or par.pcp_size != 1 or cfg.is_sequence_parallel:
        return "needs dp 1, pcp 1 and no sequence-parallel MoE"
    if par.enable_eplb:
        return "EPLB is on"
    if runner.gate is None or runner.shared_expert_gate is not None:
        return "needs the runner to hold the gate and no shared-expert gate"
    if runner.routed_input_transform is not None:
        return "a routed input transform is set"
    if runner.routed_output_transform is not None:
        return "a routed output transform is set"
    if cfg.skip_final_all_reduce:
        return "skip_final_all_reduce is set"
    if isinstance(runner.router, ZeroExpertRouter):
        return "zero-expert router"
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "the routed experts have biases"
    grouped = (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    )
    if layer.scoring_func != "sigmoid" or not layer.renormalize or grouped:
        return "routing is not sigmoid + renormalized ungrouped top-k"
    if layer.e_score_correction_bias is None:
        return "no e_score_correction_bias"
    if layer.custom_routing_function is not None:
        return "custom routing function"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return "routing simulation is on"
    if moe_routing._ROUTER_TOPK != "rowmax":
        return f"TPU_MOE_ROUTER_TOPK={moe_routing._ROUTER_TOPK} (needs rowmax)"
    if layer.routed_scaling_factor != 1.0:
        return f"layer routed_scaling_factor {layer.routed_scaling_factor} != 1"
    if not kernel.exact_in_bf16(runner.routed_scaling_factor):
        return f"routed_scaling_factor {runner.routed_scaling_factor} is not bf16"
    shared = _shared_slice(runner, pre_load=True) if pre_load else _shared_slice(runner)
    if isinstance(shared, str):
        return shared
    if proposed_routed is not None:
        w13, w2, scales = proposed_routed
    else:
        w13, w2 = layer.w13_weight, layer.w2_weight
        scales = get_scales(layer)
        if isinstance(scales, str):
            return scales
    hidden = int(_sds(w13).shape[1])
    experts = layer.global_num_experts
    topk = int(cfg.experts_per_token)
    weights = (
        layer.e_score_correction_bias,
        w13,
        w2,
        *scales,
        *shared,
    )
    rows = tuple(
        m
        for m in row_buckets
        if can_use_fn(
            jax.ShapeDtypeStruct((m, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((m, experts), jnp.float32),
            *(_sds(t) for t in weights),
            topk=topk,
            n_ranks=n_ranks,
        )
    )
    if not rows:
        return no_rows_msg(w13, scales, shared)
    base_statics = dict(
        topk=topk,
        routed_scale=float(runner.routed_scaling_factor),
        activation=qm._tpu_activation_str,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_threshold=resolve_onehot_permute_threshold(),
        use_ep=bool(par.use_ep),
        router_bias=bool(moe_routing._ROUTER_TOPK_BIAS),
        fast=bool(envs.TPU_MOE_TP_DECODE_FAST),
    )
    statics = build_statics(base_statics, runner, hidden, topk, scales, shared)
    if isinstance(statics, str):
        return statics
    return _Admitted(
        shared=shared,
        rows=rows,
        hidden=hidden,
        statics=statics,
        experts_start=int(experts_start),
    )


def _admit(runner, n_ranks: int) -> _Admitted | str:
    """What arming needs for this runner, or why it keeps the usual path."""
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    def _pre_check() -> str | None:
        if envs.FUSED_MOE_DECODE_KERNEL:
            return "FUSED_MOE_DECODE_KERNEL is set; moe_tp_decode stands down"
        # Same test as apply_monolithic's owns_collectives: those paths run
        # their own collectives, so the layer keeps them.
        if enable_pipelined_collective_and_compute() or envs.TPU_MOE_FP8_ALL_GATHER:
            return (
                "the MoE path owns its collectives "
                "(TPU_MOE_COLLECTION_CHUNK_SIZE or TPU_MOE_FP8_ALL_GATHER)"
            )
        return None

    return _admit_common(
        runner,
        n_ranks,
        quant_cls=VllmFp8MoEMethodTPU,
        pre_check=_pre_check,
        get_scales=lambda layer: (
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
        ),
        row_buckets=kernel.ROW_BUCKETS,
        can_use_fn=kernel.can_use_moe_tp_decode,
        no_rows_msg=lambda w13, _scales, shared: (
            f"the kernel admits no token count (w13 {list(w13.shape)} "
            f"{w13.dtype}, shared w13 {list(shared[0].shape)})"
        ),
        build_statics=lambda base, rnr, hidden, _topk, _scales, _shared: _OpStatics(
            **base,
            prefill_above=(
                envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS
                if envs.FUSED_MOE_PREFILL_KERNEL
                and not rnr.moe_config.moe_parallel_config.use_ep
                and hidden % _PREFILL_Q_BLOCK == 0
                else None
            ),
        ),
    )


def _route(x, scores, bias, is_pad, st):
    """select_experts as the unarmed path runs it (ids from the biased scores,
    renormalized unbiased weights in bf16), then
    zero_routing_weights_for_padding: (weights, ids)."""
    bias = bias.astype(jnp.float32)
    if st.router_bias:
        w, ids = router_select_bias(scores, bias[None, :], st.topk)
    else:
        _, ids = router_select(scores + bias[None, :], st.topk)
        w = jnp.take_along_axis(scores, ids, axis=-1)
    total = jnp.sum(w, axis=-1, keepdims=True)
    # torch.clamp(min=1e-20) keeps NaN; a fused jnp.maximum may not (XLA CPU).
    total = jnp.where(jnp.isnan(total), total, jnp.maximum(total, _SUM_FLOOR))
    w = (w / total).astype(x.dtype)
    w = jnp.where(is_pad[:, None], jnp.zeros_like(w), w)
    return w, ids


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
    st,
    axis,
    x_q=None,
    x_scale=None,
):
    """The layer's output, summed over `axis`, from the fused MoE prefill
    kernel (FUSED_MOE_PREFILL_KERNEL) on this op's routing, as
    fused_moe_decode's prefill path calls it. GLM-5.2's shared expert is
    ungated; routed_scale is folded into the route weights by the kernel.
    FUSED_MOE_PREFILL_KERNEL_V2: the stage-A build (per-row activation scales)
    where the experts fit whole. `x_q` / `x_scale` (TPU_FUSED_ADD_RMSNORM_QUANT_MOE):
    x's per-row fp8 rows and scales from the norm, which the stage-A build
    takes in place of its own quantization; the default build (512-column
    scales) quantizes x itself."""
    prefill = fused_moe_prefill
    if envs.FUSED_MOE_PREFILL_KERNEL_V2 and (
        plan_blocks(x.shape[0], x.shape[1], w2.shape[1], sw2.shape[0], w13.dtype)
        is None
    ):
        prefill = fused_moe_prefill_stage_a
    # Trace-time, once per token bucket: the evidence that the kernel serves it.
    logger.info(
        "moe_tp_decode: fused MoE prefill kernel%s traced for %d tokens",
        " (V2, stage A)" if prefill is fused_moe_prefill_stage_a else "",
        x.shape[0],
    )
    w, ids = _route(x, scores, bias, is_pad, st)
    prequant = {}
    if x_q is not None and prefill is fused_moe_prefill_stage_a:
        prequant = {"x_q": x_q, "x_scale": x_scale}
    return prefill(
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
        expert_start=None,
        routed_scale=st.routed_scale,
        **prequant,
    )


def _tp_chain(
    x, scores, bias, is_pad, w13, w2, s13, s2, sw13, ss13, sw2, ss2, st, e0=None
):
    """This rank's partial as the unarmed path computes it: _route,
    fused_moe_func (under EP with experts_start e0, so the SparseCore EP
    dispatch and combine), the shared expert's two quantized linears around
    SiluAndMul, and shared + scale * routed in bf16."""
    w, ids = _route(x, scores, bias, is_pad, st)
    routed = fused_moe_func(
        x,
        w13,
        w2,
        s13,
        s2,
        None,
        None,
        w,
        ids,
        e0 if st.use_ep else None,
        topk=st.topk,
        activation=st.activation,
        use_ep=st.use_ep,
        use_sparse_core=st.use_sparse_core,
        onehot_moe_permute_threshold=st.onehot_threshold,
        rhs_quant_dtype=None,
        skip_padded_tokens=True,  # admission requires TPU_MOE_SKIP_PADDED_TOKENS
    )
    gate_up = xla_quantized_matmul(x, sw13, ss13)
    fs = sw2.shape[0]
    act = jax.nn.silu(gate_up[:, :fs].astype(jnp.float32)).astype(x.dtype)
    shared = xla_quantized_matmul(act * gate_up[:, fs:], sw2, ss2)
    return shared + routed * st.routed_scale


def _make_op_body(
    statics: _OpStatics,
    n_ranks: int,
    axis: str,
    *,
    row_buckets: tuple[int, ...],
    can_use_fn: Callable[..., bool],
    decode_fn: Callable[..., jax.Array],
    above_fn: Callable[..., jax.Array] | None = None,
):
    """Per-rank body under shard_map over `axis`: the fused decode kernel at
    admitted token counts, `above_fn` above `statics.prefill_above` when set,
    else this rank's partial from `_tp_chain` and a psum."""

    def body(
        x,
        scores,
        bias,
        pad_mask,
        w13,
        w2,
        s13,
        s2,
        sw13,
        ss13,
        sw2,
        ss2,
        rank,
        e0,
        *prequant,
    ):
        n = x.shape[0]  # static: torch_tpu exports one program per token count
        is_pad = pad_mask[:n]  # replicated, so n_valid agrees on every rank
        n_valid = n - jnp.sum(is_pad.astype(jnp.int32))
        weights = (w13, w2, s13, s2, sw13, ss13, sw2, ss2)
        m = row_buckets[0]
        if n < m:
            # A decode bucket below the kernel's smallest row bucket
            # (TPU_TOKEN_BUCKET_EXTRA=4,8): run the kernel on `m` rows, the
            # extra rows trailing padding. The kernel skips rows >= n_valid
            # (admission requires TPU_MOE_SKIP_PADDED_TOKENS) and every row is
            # independent, so the first n rows are bit for bit the m-row
            # bucket's; the plain-XLA fallback below costs far more.
            xp = jnp.pad(x, ((0, m - n), (0, 0)))
            sp = jnp.pad(scores, ((0, m - n), (0, 0)))
            if can_use_fn(xp, sp, bias, *weights):
                return decode_fn(
                    xp, sp, bias, n_valid, rank.reshape(()), *weights, e0.reshape(())
                )[:n]
        if can_use_fn(x, scores, bias, *weights):
            return decode_fn(
                x, scores, bias, n_valid, rank.reshape(()), *weights, e0.reshape(())
            )
        if (
            above_fn is not None
            and statics.prefill_above is not None
            and n > statics.prefill_above
        ):
            return above_fn(
                x,
                scores,
                bias,
                is_pad,
                *weights,
                statics=statics,
                axis=axis,
                e0=e0,
                prequant=prequant,
            ).astype(x.dtype)
        partial = _tp_chain(x, scores, bias, is_pad, *weights, statics, e0.reshape(()))
        return lax.psum(partial, axis)

    return body


def _op_body(statics: _OpStatics, n_ranks: int, axis: str):
    """The op's per-rank body under shard_map over `axis`: the fused kernel at
    the token counts it admits, else this rank's partial from the usual path
    and a psum."""

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
        del e0
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
            *prequant,
        )

    return _make_op_body(
        statics,
        n_ranks,
        axis,
        row_buckets=kernel.ROW_BUCKETS,
        can_use_fn=lambda x, scores, bias, *weights: kernel.can_use_moe_tp_decode(
            x, scores, bias, *weights, topk=statics.topk, n_ranks=n_ranks
        ),
        decode_fn=lambda x, scores, bias, n_valid, rank, *weights_and_e0: (
            kernel.moe_tp_decode(
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
            )
        ),
        above_fn=_above,
    )


def _build_jax_op(
    cache: dict[tuple, Any],
    mesh: Any,
    statics: _OpStatics,
    *,
    op_prefix: str,
    make_body: Callable[[Any, int, str], Any],
    prequant: bool = False,
):
    key = (mesh, statics, prequant)
    op = cache.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    # Operands as moe_tp_decode*_forward passes them; bias and the padding mask
    # are replicated, everything else is per rank.
    specs = (P(axis), P(axis), P(), P()) + (P(axis),) * (12 if prequant else 10)
    body = make_body(statics, int(mesh.size), axis)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    def run(*args):
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        x: jax.Array,
        scores: jax.Array,
        bias: jax.Array,
        pad_mask: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        sw13: jax.Array,
        ss13: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        rank: jax.Array,
        e0: jax.Array,
    ) -> jax.Array:
        return run(
            x, scores, bias, pad_mask, w13, w2, s13, s2, sw13, ss13, sw2, ss2, rank, e0
        )

    def fn_prequant(
        x: jax.Array,
        scores: jax.Array,
        bias: jax.Array,
        pad_mask: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        sw13: jax.Array,
        ss13: jax.Array,
        sw2: jax.Array,
        ss2: jax.Array,
        rank: jax.Array,
        e0: jax.Array,
        x_q: jax.Array,
        x_scale: jax.Array,
    ) -> jax.Array:
        return run(
            x,
            scores,
            bias,
            pad_mask,
            w13,
            w2,
            s13,
            s2,
            sw13,
            ss13,
            sw2,
            ss2,
            rank,
            e0,
            x_q,
            x_scale,
        )

    name = op_prefix + (f"_{len(cache)}" if cache else "")
    # torch_tpu annotates the partition specs as tuples of axis-name tuples, but
    # jax_op takes PartitionSpecs at runtime, as main's other call sites pass them.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name,
        fn_prequant if prequant else fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    # vLLM traces with a dynamic token dimension; rows in == rows out.
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    cache[key] = op
    return op


def _get_op(mesh, statics: _OpStatics, prequant: bool = False):
    """One jax_op per distinct closure; built at arming, never in a trace.
    `prequant`: the variant with two more per-rank operands, x_q [T, h/128,
    128] fp8 and x_scale [T] f32 (TPU_FUSED_ADD_RMSNORM_QUANT_MOE)."""
    return _build_jax_op(
        _OPS,
        mesh,
        statics,
        op_prefix="pallas::moe_tp_decode",
        make_body=_op_body,
        prequant=prequant,
    )


def _selftest_op(mesh, spec: all_reduce.AllReduceSpec):
    op = _SELFTEST_OPS.get(spec)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    specs = (P(axis), P(axis), P())
    call = all_reduce.standalone(spec)

    def body(x, rank, n_valid):
        meta = jnp.stack([n_valid.reshape(()), rank.reshape(())]).astype(jnp.int32)
        return call(meta, x)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    def fn(x: jax.Array, rank: jax.Array, n_valid: jax.Array) -> jax.Array:
        args = (x, rank, n_valid)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # PartitionSpecs, as in _get_op (torch_tpu's annotation is narrower).
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::moe_tp_decode_selftest_{len(_SELFTEST_OPS)}",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _SELFTEST_OPS[spec] = op
    return op


def _rank_tensor(device: torch.device, mesh_index: int) -> torch.Tensor:
    """This rank's mesh index as int32 [1, 1] data (one tensor per process), so
    every rank compiles the same program."""
    global _RANK_BUFFER
    if _RANK_BUFFER is None:
        _RANK_BUFFER = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    return _RANK_BUFFER


def _e0_tensor(device: torch.device, experts_start: int) -> torch.Tensor:
    """experts_start as int32 [1, 1] data. TPU_MOE_TP_DECODE_SHARED_META: one
    tensor per value, shared by every layer, so each layer's meta stack is the
    same HLO value and CSE keeps one; else a fresh tensor per layer."""
    if not envs.TPU_MOE_TP_DECODE_SHARED_META:
        return torch.tensor([[experts_start]], dtype=torch.int32, device=device)
    key = (str(device), int(experts_start))
    t = _E0_BUFFERS.get(key)
    if t is None:
        t = torch.tensor([[experts_start]], dtype=torch.int32, device=device)
        _E0_BUFFERS[key] = t
    return t


def _selftest(topo: _Topology, buckets, device) -> list[tuple[int, int, int]]:
    """Run the kernel's all-reduce alone for every armed (rows, hidden) and row
    path through the op's mesh and rank operand; returns this rank's failing
    (rows, hidden, n_valid). Collective: every rank makes the same calls."""
    n_ranks = int(topo.mesh.size)
    rank = _rank_tensor(device, topo.mesh_index)
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for m, h in buckets:
            spec = all_reduce.make_spec(n_ranks, m, h)
            op = _selftest_op(topo.mesh, spec)
            x = all_reduce.selftest_inputs(spec)
            mine = torch.from_numpy(x[topo.mesh_index]).to(torch.bfloat16).to(device)
            for nv in all_reduce.selftest_row_counts(spec):
                n_valid = torch.tensor([nv], dtype=torch.int32, device=device)
                out = op(mine, rank, n_valid).to(torch.float32).cpu().numpy()
                if not all_reduce.selftest_ok(spec, x, out, nv):
                    failures.append((m, h, nv))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def _gather(obj) -> list:
    """all_gather_object over the TP group's CPU group."""
    tp = parallel_state.get_tp_group()
    if tp.world_size == 1:
        return [obj]
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def _check_ranks_agree(entry: dict) -> None:
    """Raise unless every TP rank arms the same layers with the same statics
    and, when any layer is armed, the mesh indices are 0..n-1 once each (the
    kernel addresses peers, and orders its f32 sum, by mesh index)."""
    seen = _gather(entry)
    problems = []
    if any(s["sig"] != seen[0]["sig"] for s in seen):
        problems.append("arming signatures differ")
    indices = sorted(s["mesh_index"] for s in seen)
    if any(s["armed"] for s in seen) and indices != list(range(len(seen))):
        problems.append(f"mesh indices {indices} are not 0..{len(seen) - 1}")
    if problems:
        raise RuntimeError(
            f"moe_tp_decode: TP ranks disagree ({'; '.join(problems)}); their "
            f"programs and the kernel's barrier would not match. Per rank: {seen}"
        )


def should_arm() -> bool:
    """Whether load_model should call `arm` at all.

    USE_MOE_TP_DECODE_KERNEL is on by default, and without
    TPU_MOE_SKIP_PADDED_TOKENS every layer refuses (the op reads n_valid from
    the padding mask), so the default skips arming and its collectives there.
    An explicit setting still arms and logs why each layer was refused.
    """
    if not envs.USE_MOE_TP_DECODE_KERNEL:
        return False
    return envs.TPU_MOE_SKIP_PADDED_TOKENS or envs.is_set("USE_MOE_TP_DECODE_KERNEL")


def _admit_runner(
    runner: Any,
    topo: _Topology | str,
    admit_fn: Callable[[Any, int], _Admitted | str],
    *,
    pre_load: bool = False,
) -> _Admitted | str:
    """Run topology, padding-mask, and layer admission gates for ``runner``."""
    if isinstance(topo, str):
        return topo
    if not envs.TPU_MOE_SKIP_PADDED_TOKENS or (
        not pre_load and token_padding._padding_state is None
    ):
        return "needs TPU_MOE_SKIP_PADDED_TOKENS=1 (n_valid comes from its mask)"
    return admit_fn(runner, int(topo.mesh.size))


def _arm_layers(
    model: torch.nn.Module,
    draft_model: torch.nn.Module | None,
    *,
    is_already_armed: Callable[[Any], bool],
    admit_fn: Callable[[Any, int], _Admitted | str],
    get_op_fn: Callable[[Any, Any], Any],
    op_attr: str,
    buffers: tuple[str, ...],
    env_flag: str,
    log_tag: str,
    log_obj: Any,
) -> int:
    """Shared arm loop across TP decode MoE adapters."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    topo = _topology()  # collective: first, on every rank
    target = [m for m in model.modules() if isinstance(m, MoERunner)]
    known = {id(r) for r in target}
    drafter = (
        []
        if draft_model is None
        else [
            m
            for m in draft_model.modules()
            if isinstance(m, MoERunner) and id(m) not in known
        ]
    )
    plan: list[tuple[Any, _Admitted]] = []
    refused: dict[str, int] = {}
    for runner in target + drafter:
        if is_already_armed(runner._quant_method):
            continue
        res = _admit_runner(runner, topo, admit_fn)
        if isinstance(res, str):
            refused[res] = refused.get(res, 0) + 1
        else:
            plan.append((runner, res))
    buckets = sorted({(m, adm.hidden) for _, adm in plan for m in adm.rows})
    entries = [(r.layer_name, repr(adm.statics)) for r, adm in plan]
    sig = hashlib.sha256(
        repr((entries, len(target), len(drafter), buckets)).encode()
    ).hexdigest()[:16]
    mesh_index = -1 if isinstance(topo, str) else topo.mesh_index
    _check_ranks_agree(dict(sig=sig, mesh_index=mesh_index, armed=len(plan)))
    uncovered = [b for b in buckets if b not in _SELFTESTED_BUCKETS]
    if plan and uncovered:
        assert not isinstance(topo, str)
        device = plan[0][0].routed_experts.w13_weight.device
        failures = _selftest(topo, uncovered, device)
        seen = _gather(dict(mesh_index=topo.mesh_index, failures=failures))
        if any(s["failures"] for s in seen):
            raise RuntimeError(
                f"{log_tag}: the all-reduce self-test failed (per rank: mesh "
                f"index, failing (rows, hidden, n_valid)): {seen}"
            )
        _SELFTESTED_BUCKETS.update(uncovered)
    for runner, adm in plan:
        assert not isinstance(topo, str)
        layer = runner.routed_experts
        # An alias of the f32 parameter: to() and contiguous() are no-ops.
        bias = layer.e_score_correction_bias.detach().to(torch.float32).contiguous()
        rank = _rank_tensor(layer.w13_weight.device, topo.mesh_index)
        # [1, 1] like the rank: per-rank data, one program on every rank.
        e0 = _e0_tensor(layer.w13_weight.device, adm.experts_start)
        for name, tensor in zip(buffers, (*adm.shared, bias, rank, e0)):
            layer.register_buffer(name, tensor, persistent=False)
        statics = adm.statics
        # TPU_MOE_TP_DECODE_HBM_OPERANDS: "drafter" (MTP layer only) or "all".
        mode = envs.TPU_MOE_TP_DECODE_HBM_OPERANDS
        if mode not in ("", "0", "drafter", "all"):
            raise ValueError(
                f"TPU_MOE_TP_DECODE_HBM_OPERANDS={mode!r}: use '', 'drafter' or 'all'"
            )
        if mode == "all" or (mode == "drafter" and id(runner) not in known):
            statics = dataclasses.replace(statics, hbm_operands=True)
        setattr(layer.quant_method, op_attr, get_op_fn(topo.mesh, statics))
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        runner.routed_scaling_factor = 1.0
    armed_drafter = sum(1 for r, _ in plan if id(r) not in known)
    # On by default for FP8, so a model it does not serve is the common case there.
    log = log_obj.info if plan or not envs.is_set(env_flag) else log_obj.warning
    log(
        f"{log_tag}: armed %d/%d target and %d/%d drafter MoE layers (mesh "
        "index %d, kernel token counts %s, sig %s); refused: %s",
        len(plan) - armed_drafter,
        len(target),
        armed_drafter,
        len(drafter),
        mesh_index,
        sorted({m for m, _ in buckets}),
        sig,
        refused or "none",
    )
    return len(plan)


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm every admitted MoE layer of `model` and of `draft_model` (the MTP
    drafter); returns how many were armed. Collective over the TP group."""
    return _arm_layers(
        model,
        draft_model,
        is_already_armed=moe_tp_decode_supported,
        admit_fn=_admit,
        get_op_fn=lambda mesh, statics: _get_op(mesh, statics),
        op_attr=_OP_ATTR,
        buffers=_BUFFERS,
        env_flag="USE_MOE_TP_DECODE_KERNEL",
        log_tag="moe_tp_decode",
        log_obj=logger,
    )


def _forward_armed(
    quant_method: Any,
    layer: Any,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    op_attr: str,
    buffers: tuple[str, ...],
    scale_attrs: tuple[str, str],
    log_tag: str,
    op_prequant_attr: str | None = None,
) -> torch.Tensor:
    padding = token_padding._padding_state
    assert padding is not None, f"{log_tag} needs the token-padding mask"
    sw13, ss13, sw2, ss2, bias, rank, e0 = (getattr(layer, n) for n in buffers)
    # The usual path's scoring function; everything after it is the op.
    scores = moe_routing._apply_scoring_fn(layer.scoring_func, router_logits)
    op = getattr(quant_method, op_attr)
    extra: tuple = ()
    if op_prequant_attr is not None:
        # TPU_FUSED_ADD_RMSNORM_QUANT_MOE: the norm that produced `x` filled the
        # slot only on its Pallas (prefill-sized) branch, so this is not a
        # token-count branch of its own; the rows are the stage-A kernel's own
        # quantization, bit for bit.
        slot = getattr(layer, PREQUANT_MOE_ATTR, None)
        prequant = None if slot is None else slot.value
        if slot is not None and prequant is not None:
            slot.value = None
            op = getattr(quant_method, op_prequant_attr)
            extra = tuple(prequant)
    s13_attr, s2_attr = scale_attrs
    return op(
        x,
        scores,
        bias,
        padding.local_padding_mask,
        layer.w13_weight,
        layer.w2_weight,
        getattr(layer, s13_attr),
        getattr(layer, s2_attr),
        sw13,
        ss13,
        sw2,
        ss2,
        rank,
        e0,
        *extra,
    )


def moe_tp_decode_forward(
    quant_method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    """Forward of an armed layer: shared + routed_scaling_factor * routed,
    already all-reduced over the TP group."""
    return _forward_armed(
        quant_method,
        layer,
        x,
        router_logits,
        op_attr=_OP_ATTR,
        buffers=_BUFFERS,
        scale_attrs=("w13_weight_scale_inv", "w2_weight_scale_inv"),
        log_tag="moe_tp_decode",
        op_prequant_attr=_OP_PREQUANT_ATTR,
    )


def arm_prequant_op(layer: Any) -> int | None:
    """TPU_FUSED_ADD_RMSNORM_QUANT_MOE: build (at load, outside Dynamo) the
    prequant variant of an armed layer's op. Returns the token count above
    which its prefill kernel runs (`prefill_above`), or None when the layer is
    not armed, has no prefill kernel, or FUSED_MOE_PREFILL_KERNEL_V2 is off
    (only the stage-A build takes per-row rows: nothing to pair)."""
    if not envs.FUSED_MOE_PREFILL_KERNEL_V2:
        return None
    op = getattr(layer.quant_method, _OP_ATTR, None)
    if op is None:
        return None
    entry = next(((k, v) for k, v in _OPS.items() if v is op), None)
    if entry is None:
        return None
    (mesh, statics, prequant), _ = entry
    if prequant or statics.prefill_above is None:
        return None
    setattr(layer.quant_method, _OP_PREQUANT_ATTR, _get_op(mesh, statics, True))
    return statics.prefill_above
