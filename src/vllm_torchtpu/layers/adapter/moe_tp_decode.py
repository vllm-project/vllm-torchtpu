# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for kernels/experimental/moe_tp_decode (USE_MOE_TP_DECODE_KERNEL).

Under MoE tensor parallelism vLLM runs the shared expert, scales and adds the
routed output, and all-reduces the sum over the TP group. For an armed layer
one torch op does all of it: at the token counts the kernel admits (16, 32 and 64)
one Pallas call per rank covers routing, the shared expert, the routed experts
and the all-reduce (counts below 16, and with TPU_MOE_TP_DECODE_PAD_ROWS counts
between two of them, run it on the next count with trailing padding rows);
every other count runs the usual TP path (router top-k, fused_moe_func, the
shared expert's linears) and a psum in the same exported program. The token
count is static there, so one torch op serves every bucket. The op's output
is reduced, so arming also takes the shared expert and the routed scale off
the runner, and VllmMoERunner._fused_output_is_reduced turns the runner's
all-reduce off for that layer.

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
from vllm_torchtpu.kernels.experimental import rotated_collectives
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    combine_scatter,
    fused_moe_prefill,
    plan_blocks,
    row_all_gather,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel_v2 import (  # noqa: E501
    _quantize_rows,
    fused_moe_prefill_stage_a,
    gather_routes,
    gather_rows,
    own_rows,
)
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.kernels.router_topk import select as router_select
from vllm_torchtpu.kernels.router_topk import select_bias as router_select_bias
from vllm_torchtpu.layers.adapter import moe_routing, small_all_reduce_op, token_padding
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
# TPU_PREFILL_SEQ_PARALLEL: the op variants that take and return this core's
# rows of the step (set by arm_token_split_op), and the slot on the FusedMoE
# layer through which the decoder layer asks for them.
_OP_SPLIT_ATTR = "_moe_tp_decode_op_split"
_OP_SPLIT_PREQUANT_ATTR = "_moe_tp_decode_op_split_prequant"
TOKEN_SPLIT_MOE_ATTR = "_tpu_token_split_moe_slot"
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
# TPU_MOE_PREFILL_ROTATED_COLLECTIVES: (chip coords, core on chip) per mesh
# index of each op mesh the rotated collectives run over (None: XLA's).
_ROTATED_PLACES: dict[Any, list | None] = {}
_SELFTEST_DONE = False


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
    # TPU_MOE_TP_DECODE_AG_PAIRS: the all-reduce paths that exchange at least
    # this many rows all-gather by chip pairs (0: none). Same bits.
    ag_pairs: int = 0
    # TPU_MOE_TP_DECODE_EXCHANGE_PIECES: the all-reduce paths that exchange at
    # least this many rows exchange by lane-group piece (reduce-scatter and
    # all-gather overlap; 0: none). Same bits.
    exchange_pieces: int = 0
    # TPU_MOE_TP_DECODE_PAD_ROWS: a token count between two kernel row counts
    # runs the kernel on the next one (see _kernel_rows). Same bits per row.
    pad_rows: bool = False


@dataclasses.dataclass(frozen=True)
class _Admitted:
    shared: tuple[torch.Tensor, ...]  # sw13, ss13, sw2, ss2
    rows: tuple[int, ...]  # the token counts the kernel admits
    hidden: int
    statics: _OpStatics
    experts_start: int  # first global expert of this rank's block (0 under TP)


def moe_tp_decode_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's quant method the op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def _sds(t: torch.Tensor) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        tuple(t.shape), pallas.pallas.TORCH_TO_JAX_DTYPE_MAP[t.dtype]
    )


def _topology() -> _Topology | str:
    """The op's 1-D mesh over the TP group and this rank's index in it, or a
    refusal reason. Collective: build_ep_mesh gathers device ids."""
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if ep is None:
        return "no EP group (the TP group has one device)"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    ep_ranks = tuple(int(r) for r in ep.ranks)
    if tp_ranks != ep_ranks:
        return f"TP ranks {tp_ranks} != EP ranks {ep_ranks} (needs dp 1, pcp 1)"
    world = torch.distributed.get_world_size()
    if len(tp_ranks) != world:
        return f"the TP group has {len(tp_ranks)} of {world} ranks"
    mesh = build_ep_mesh()
    mesh_index = ep_mesh_index()
    assert mesh is not None and mesh_index is not None  # the EP group exists
    return _Topology(mesh=mesh, mesh_index=mesh_index)


def _shared_slice(runner) -> tuple[torch.Tensor, ...] | str:
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
    for lin in (gate_up, down):
        if not isinstance(lin.quant_method, VllmFp8LinearMethodTPU):
            return "a shared-expert linear is not VllmFp8LinearMethodTPU"
        flipped = getattr(lin, WEIGHT_FLIPPED_ATTR, False)
        if not flipped or getattr(lin, KEEP_VLLM_LAYOUT_ATTR, False):
            return "a shared-expert weight is not K-major"
        if lin.bias is not None:
            return "a shared-expert linear has a bias"
        if lin.weight.dtype != torch.float8_e4m3fn or lin.weight_scale.dim() != 1:
            return "the shared expert is not FP8 with per-channel scales"
    if down.reduce_results:
        return "the shared expert reduces its own output"
    return (
        gate_up.weight.data,
        gate_up.weight_scale.data,
        down.weight.data,
        down.weight_scale.data,
    )


def _admit(runner, n_ranks: int) -> _Admitted | str:
    """What arming needs for this runner, or why it keeps the usual path."""
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )

    from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    layer = runner.routed_experts
    qm = layer.quant_method
    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    if type(qm) is not VllmFp8MoEMethodTPU:
        return f"quant method {type(qm).__name__} is not VllmFp8MoEMethodTPU"
    if qm._tpu_activation_str != "silu":
        return f"activation {qm._tpu_activation_str!r} is not silu"
    if qm._use_adaptive_fused_moe or fused_moe_ep_supported(qm):
        return "a fused EP MoE kernel serves this layer"
    if envs.FUSED_MOE_DECODE_KERNEL:
        return "FUSED_MOE_DECODE_KERNEL is set; moe_tp_decode stands down"
    # Same test as apply_monolithic's owns_collectives: those paths run their own
    # collectives, so the layer keeps them.
    if enable_pipelined_collective_and_compute() or envs.TPU_MOE_FP8_ALL_GATHER:
        return (
            "the MoE path owns its collectives "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE or TPU_MOE_FP8_ALL_GATHER)"
        )
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
        if layer.w13_weight.shape[0] * par.ep_size != layer.global_num_experts:
            return (
                f"{layer.w13_weight.shape[0]} local experts x ep {par.ep_size} != "
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
    shared = _shared_slice(runner)
    if isinstance(shared, str):
        return shared
    hidden = layer.w13_weight.shape[1]
    experts = layer.global_num_experts
    topk = int(cfg.experts_per_token)
    weights = (
        layer.e_score_correction_bias,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        *shared,
    )
    rows = tuple(
        m
        for m in kernel.ROW_BUCKETS
        if kernel.can_use_moe_tp_decode(
            jax.ShapeDtypeStruct((m, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((m, experts), jnp.float32),
            *(_sds(t) for t in weights),
            topk=topk,
            n_ranks=n_ranks,
        )
    )
    if not rows:
        return (
            f"the kernel admits no token count (w13 {list(layer.w13_weight.shape)} "
            f"{layer.w13_weight.dtype}, shared w13 {list(shared[0].shape)})"
        )
    statics = _OpStatics(
        topk=topk,
        routed_scale=float(runner.routed_scaling_factor),
        activation=qm._tpu_activation_str,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_threshold=resolve_onehot_permute_threshold(),
        use_ep=bool(par.use_ep),
        router_bias=bool(moe_routing._ROUTER_TOPK_BIAS),
        prefill_above=(
            envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS
            if envs.FUSED_MOE_PREFILL_KERNEL
            and not par.use_ep
            and hidden % _PREFILL_Q_BLOCK == 0
            else None
        ),
        fast=bool(envs.TPU_MOE_TP_DECODE_FAST),
        ag_pairs=int(envs.TPU_MOE_TP_DECODE_AG_PAIRS),
        exchange_pieces=int(envs.TPU_MOE_TP_DECODE_EXCHANGE_PIECES),
        pad_rows=bool(envs.TPU_MOE_TP_DECODE_PAD_ROWS),
    )
    return _Admitted(
        shared=shared,
        rows=rows,
        hidden=hidden,
        statics=statics,
        experts_start=int(experts_start),
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


def _kernel_rows(n: int, pad_rows: bool) -> int:
    """The row count the kernel runs an n-row call on: n when the kernel has
    that size; the smallest size above n when n is below the smallest size,
    or, with pad_rows, between two sizes; otherwise n (the usual path)."""
    larger = [m for m in kernel.ROW_BUCKETS if m > n]
    if n in kernel.ROW_BUCKETS or not larger:
        return n
    if n < kernel.ROW_BUCKETS[0] or pad_rows:
        return larger[0]
    return n


def _op_body(statics: _OpStatics, n_ranks: int, axis: str):
    """The op's per-rank body under shard_map over `axis`: the fused kernel at
    the token counts it admits, else this rank's partial from the usual path
    and a psum."""

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
        m = _kernel_rows(n, statics.pad_rows)
        if m != n:
            # A decode bucket the kernel has no size for (4 and 8 from
            # TPU_TOKEN_BUCKET_EXTRA, below its smallest size; 48, between
            # sizes, with TPU_MOE_TP_DECODE_PAD_ROWS): run the kernel on `m`
            # rows, the extra rows trailing padding. The kernel skips rows >=
            # n_valid (admission requires TPU_MOE_SKIP_PADDED_TOKENS) and every
            # row is independent, so the first n rows are bit for bit the m-row
            # bucket's; the plain-XLA fallback below costs far more.
            xp = jnp.pad(x, ((0, m - n), (0, 0)))
            sp = jnp.pad(scores, ((0, m - n), (0, 0)))
            if kernel.can_use_moe_tp_decode(
                xp, sp, bias, *weights, topk=statics.topk, n_ranks=n_ranks
            ):
                return kernel.moe_tp_decode(
                    xp,
                    sp,
                    bias,
                    n_valid,
                    rank.reshape(()),
                    *weights,
                    e0.reshape(()),
                    routed_scale=statics.routed_scale,
                    topk=statics.topk,
                    n_ranks=n_ranks,
                    fast=statics.fast,
                    hbm_operands=statics.hbm_operands,
                    ag_pairs=statics.ag_pairs,
                    exchange_pieces=statics.exchange_pieces,
                )[:n]
        if kernel.can_use_moe_tp_decode(
            x, scores, bias, *weights, topk=statics.topk, n_ranks=n_ranks
        ):
            return kernel.moe_tp_decode(
                x,
                scores,
                bias,
                n_valid,
                rank.reshape(()),
                *weights,
                e0.reshape(()),
                routed_scale=statics.routed_scale,
                topk=statics.topk,
                n_ranks=n_ranks,
                fast=statics.fast,
                hbm_operands=statics.hbm_operands,
                ag_pairs=statics.ag_pairs,
                exchange_pieces=statics.exchange_pieces,
            )
        if statics.prefill_above is not None and n > statics.prefill_above:
            return _prefill_chain(
                x, scores, bias, is_pad, *weights, statics, axis, *prequant
            ).astype(x.dtype)
        partial = _tp_chain(x, scores, bias, is_pad, *weights, statics, e0.reshape(()))
        return lax.psum(partial, axis)

    return body


def _split_body(statics: _OpStatics, n_ranks: int, axis: str, rotated=None):
    """The op's per-rank body for TPU_PREFILL_SEQ_PARALLEL: x, scores and the
    optional x_q / x_scale hold this core's rows of the step (`own_rows`), and
    so does the output. Each core routes and quantizes its own rows, the fp8
    rows, scales and routes are all-gathered, and the fused MoE prefill kernel
    runs as in _prefill_chain, without the all-gather after its
    reduce-scatter. Every row's routing, quantization and sum are the ones
    _op_body computes for it. `rotated` (TPU_MOE_PREFILL_ROTATED_COLLECTIVES):
    the peer tables (all-gather, reduce-scatter, and the fused combine's or None
    without TPU_MOE_PREFILL_COMBINE_SCATTER) of the rotated collectives
    over the op's mesh, numpy int32; the core's mesh index comes from `rank`."""

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
        del e0
        n = x.shape[0] * n_ranks
        if statics.prefill_above is None or n <= statics.prefill_above:
            raise ValueError(
                f"the split MoE op serves only the prefill kernel's token counts, "
                f"not {n}"
            )
        if plan_blocks(n, x.shape[1], w2.shape[1], sw2.shape[0], w13.dtype) is not None:
            raise NotImplementedError(
                "the split MoE op needs the stage-A prefill build (whole experts)"
            )
        logger.info(
            "moe_tp_decode: split fused MoE prefill kernel traced for %d tokens", n
        )
        # The rank operand, not lax.axis_index: its partition-id is refused by
        # XLA's SPMD partitioner in the model's whole-graph compile.
        is_pad = own_rows(pad_mask[:n], rank.reshape(()), n_ranks)
        w, ids = _route(x, scores, bias, is_pad, statics)
        x_q, x_scale = prequant if prequant else _quantize_rows(x, w13.dtype)
        core_major = envs.TPU_MOE_PREFILL_GATHER_ONCE
        if core_major:
            w, ids, x_scale = gather_routes(w, ids, x_scale, axis)
            if rotated is not None and row_all_gather.supports(x_q.shape, x_q.dtype):
                x_q = row_all_gather.all_gather(
                    x_q, rank.reshape(1), jnp.asarray(rotated[0]), axis=axis
                )
            else:
                x_q = lax.all_gather(x_q, axis, axis=0, tiled=True)
        else:
            x_q, x_scale, w, ids = (
                gather_rows(a, axis) for a in (x_q, x_scale, w, ids.astype(jnp.int32))
            )
        return fused_moe_prefill_stage_a(
            None,
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
            ids,
            axis_name=axis,
            expert_start=None,
            routed_scale=statics.routed_scale,
            x_q=x_q,
            x_scale=x_scale,
            scatter=True,
            core_major=core_major,
            rank=rank.reshape(1) if rotated is not None and core_major else None,
            rs_table=rotated[1] if rotated is not None and core_major else None,
            cs_table=rotated[2] if rotated is not None and core_major else None,
        ).astype(x.dtype)

    return body


def _get_op(mesh, statics: _OpStatics, prequant: bool = False, split: bool = False):
    """One jax_op per distinct closure; built at arming, never in a trace.
    `prequant`: the variant with two more per-rank operands, x_q [T, h/128,
    128] fp8 and x_scale [T] f32 (TPU_FUSED_ADD_RMSNORM_QUANT_MOE). `split`:
    the variant over this core's rows of the step (`_split_body`)."""
    key = (mesh, statics, prequant, split)
    op = _OPS.get(key)
    if op is not None:
        return op
    axis = mesh.axis_names[0]
    # Operands as moe_tp_decode_forward passes them; bias and the padding mask
    # are replicated, everything else is per rank.
    specs = (P(axis), P(axis), P(), P()) + (P(axis),) * (12 if prequant else 10)
    if split:
        body = _split_body(statics, int(mesh.size), axis, _rotated_tables(mesh))
    else:
        body = _op_body(statics, int(mesh.size), axis)

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

    name = "pallas::moe_tp_decode" + (f"_{len(_OPS)}" if _OPS else "")
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
    _OPS[key] = op
    return op


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


def _selftest(
    topo: _Topology, buckets, device, ag_pairs: int = 0, pieces: int = 0
) -> list[tuple[int, int, int]]:
    """Run the kernel's all-reduce alone for every armed (rows, hidden) and row
    path through the op's mesh and rank operand, with the kernel's chip-pair
    and piece settings; returns this rank's failing (rows, hidden, n_valid).
    Collective: every rank makes the same calls."""
    n_ranks = int(topo.mesh.size)
    rank = _rank_tensor(device, topo.mesh_index)
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for m, h in buckets:
            spec = all_reduce.make_spec(n_ranks, m, h, ag_pairs, pieces)
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


def _chip_places(topo: _Topology) -> list[tuple[tuple[int, ...], int]]:
    """(chip coords, core on chip) at each index of the op's mesh, gathered
    over the TP group (small_all_reduce_op._chip_places). Collective."""
    return small_all_reduce_op._chip_places(
        topo.mesh, topo.mesh_index, _gather, who="moe_tp_decode"
    )


def _rotated_places(topo: _Topology) -> list | None:
    """TPU_MOE_PREFILL_ROTATED_COLLECTIVES: (chip coords, core on chip) per
    index of the op's mesh when it is a 2x2x2 slice of 2-core chips, else None
    with the reason logged (XLA's collectives). Collective: every rank gathers
    every rank's device, and all get the same answer."""
    if not envs.TPU_MOE_PREFILL_GATHER_ONCE:
        raise ValueError(
            "TPU_MOE_PREFILL_ROTATED_COLLECTIVES needs TPU_MOE_PREFILL_GATHER_ONCE=1"
        )
    return small_all_reduce_op.rotated_places(
        topo.mesh,
        topo.mesh_index,
        "TPU_MOE_PREFILL_ROTATED_COLLECTIVES",
        "the split MoE op's fp8 row all-gather and last reduce-scatter",
        _gather,
    )


def _rotated_tables(mesh) -> tuple | None:
    """The rotated collectives' peer tables (all-gather, reduce-scatter) over
    `mesh`, numpy int32, when arming found it a 2x2x2 slice; else None."""
    places = _ROTATED_PLACES.get(mesh)
    if places is None:
        return None
    return (
        rotated_collectives.peer_table(places, "all_gather"),
        rotated_collectives.peer_table(places, "reduce_scatter"),
        combine_scatter.scatter_table(places)
        if envs.TPU_MOE_PREFILL_COMBINE_SCATTER
        else None,
    )


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


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm every admitted MoE layer of `model` and of `draft_model` (the MTP
    drafter); returns how many were armed. Collective over the TP group."""
    global _SELFTEST_DONE
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
    plan, refused = [], {}
    for runner in target + drafter:
        if moe_tp_decode_supported(runner._quant_method):
            continue
        if isinstance(topo, str):
            res = topo
        elif (
            not envs.TPU_MOE_SKIP_PADDED_TOKENS or token_padding._padding_state is None
        ):
            res = "needs TPU_MOE_SKIP_PADDED_TOKENS=1 (n_valid comes from its mask)"
        else:
            res = _admit(runner, int(topo.mesh.size))
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
    ag_pairs = int(envs.TPU_MOE_TP_DECODE_AG_PAIRS)
    pieces = int(envs.TPU_MOE_TP_DECODE_EXCHANGE_PIECES)
    if plan and ag_pairs > 0:
        assert not isinstance(topo, str)
        places = _chip_places(topo)
        all_reduce.check_chip_pairs(places)
        ids = [int(d.id) for d in topo.mesh.devices.flat]
        logger.info(
            "moe_tp_decode: chip-pair all-gather pairs (mesh indices: device "
            "ids @ chip coords): %s",
            "; ".join(
                f"{i},{i + 1}: {ids[i]},{ids[i + 1]} @ {places[i][0]}"
                for i in range(0, len(ids), 2)
            ),
        )
    if plan and envs.TPU_MOE_PREFILL_ROTATED_COLLECTIVES:
        assert not isinstance(topo, str)
        if topo.mesh not in _ROTATED_PLACES:
            _ROTATED_PLACES[topo.mesh] = _rotated_places(topo)
    if plan and not _SELFTEST_DONE:
        assert not isinstance(topo, str)
        device = plan[0][0].routed_experts.w13_weight.device
        failures = _selftest(topo, buckets, device, ag_pairs, pieces)
        seen = _gather(dict(mesh_index=topo.mesh_index, failures=failures))
        if any(s["failures"] for s in seen):
            raise RuntimeError(
                "moe_tp_decode: the all-reduce self-test failed (per rank: mesh "
                f"index, failing (rows, hidden, n_valid)): {seen}"
            )
        _SELFTEST_DONE = True
    for runner, adm in plan:
        assert not isinstance(topo, str)
        layer = runner.routed_experts
        # An alias of the f32 parameter: to() and contiguous() are no-ops.
        bias = layer.e_score_correction_bias.detach().to(torch.float32).contiguous()
        rank = _rank_tensor(layer.w13_weight.device, topo.mesh_index)
        # [1, 1] like the rank: per-rank data, one program on every rank.
        e0 = _e0_tensor(layer.w13_weight.device, adm.experts_start)
        for name, tensor in zip(_BUFFERS, (*adm.shared, bias, rank, e0)):
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
        setattr(layer.quant_method, _OP_ATTR, _get_op(topo.mesh, statics))
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        runner.routed_scaling_factor = 1.0
    armed_drafter = sum(1 for r, _ in plan if id(r) not in known)
    # On by default, so a model it does not serve is the common case there.
    log = (
        logger.info
        if plan or not envs.is_set("USE_MOE_TP_DECODE_KERNEL")
        else logger.warning
    )
    log(
        "moe_tp_decode: armed %d/%d target and %d/%d drafter MoE layers (mesh "
        "index %d, kernel token counts %s, chip-pair all-gather %s, piece "
        "exchange %s, sig %s); refused: %s",
        len(plan) - armed_drafter,
        len(target),
        armed_drafter,
        len(drafter),
        mesh_index,
        sorted({m for m, _ in buckets}),
        f"from {ag_pairs} rows" if ag_pairs else "off",
        f"from {pieces} rows" if pieces else "off",
        sig,
        refused or "none",
    )
    return len(plan)


def moe_tp_decode_forward(
    quant_method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    """Forward of an armed layer: shared + routed_scaling_factor * routed,
    already all-reduced over the TP group."""
    padding = token_padding._padding_state
    assert padding is not None, "moe_tp_decode needs the token-padding mask"
    sw13, ss13, sw2, ss2, bias, rank, e0 = (getattr(layer, n) for n in _BUFFERS)
    # The usual path's scoring function; everything after it is the op.
    scores = moe_routing._apply_scoring_fn(layer.scoring_func, router_logits)
    # TPU_FUSED_ADD_RMSNORM_QUANT_MOE: the norm that produced `x` filled the slot
    # only on its Pallas (prefill-sized) branch, so this is not a token-count
    # branch of its own; the rows are the stage-A kernel's own quantization,
    # bit for bit.
    slot = getattr(layer, PREQUANT_MOE_ATTR, None)
    prequant = None if slot is None else slot.value
    # TPU_PREFILL_SEQ_PARALLEL: the decoder layer fills this slot when x holds
    # this core's rows of the step; the output then holds them too.
    split_slot = getattr(layer, TOKEN_SPLIT_MOE_ATTR, None)
    split = split_slot is not None and split_slot.value is not None
    if split:
        assert split_slot is not None
        split_slot.value = None
    op = getattr(quant_method, _OP_SPLIT_ATTR if split else _OP_ATTR)
    extra: tuple = ()
    if slot is not None and prequant is not None:
        slot.value = None
        op = getattr(
            quant_method, _OP_SPLIT_PREQUANT_ATTR if split else _OP_PREQUANT_ATTR
        )
        extra = tuple(prequant)
    return op(
        x,
        scores,
        bias,
        padding.local_padding_mask,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        sw13,
        ss13,
        sw2,
        ss2,
        rank,
        e0,
        *extra,
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
    (mesh, statics, prequant, split), _ = entry
    if prequant or split or statics.prefill_above is None:
        return None
    setattr(layer.quant_method, _OP_PREQUANT_ATTR, _get_op(mesh, statics, True))
    return statics.prefill_above


def arm_token_split_op(layer: Any, slot: Any) -> tuple[Any, int, torch.Tensor] | None:
    """TPU_PREFILL_SEQ_PARALLEL: build (at load, outside Dynamo) the split
    variants of an armed layer's op and hand the layer `slot`, which the
    decoder layer fills to ask for them. Returns the op's mesh, the token
    count above which its prefill kernel runs and this rank's mesh index
    operand (int32 [1, 1]), or None when the layer is not armed or has no
    stage-A prefill kernel (FUSED_MOE_PREFILL_KERNEL_V2)."""
    if not envs.FUSED_MOE_PREFILL_KERNEL_V2:
        return None
    op = getattr(layer.quant_method, _OP_ATTR, None)
    if op is None:
        return None
    entry = next(((k, v) for k, v in _OPS.items() if v is op), None)
    if entry is None:
        return None
    (mesh, statics, _, _), _ = entry
    if statics.prefill_above is None or statics.use_ep:
        return None
    method = layer.quant_method
    setattr(method, _OP_SPLIT_ATTR, _get_op(mesh, statics, split=True))
    if getattr(method, _OP_PREQUANT_ATTR, None) is not None:
        setattr(method, _OP_SPLIT_PREQUANT_ATTR, _get_op(mesh, statics, True, True))
    setattr(layer, TOKEN_SPLIT_MOE_ATTR, slot)
    return mesh, statics.prefill_above, layer._moe_tp_decode_rank
