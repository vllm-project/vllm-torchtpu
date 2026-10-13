# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused EP MoE v3 under tensor-parallel attention (MOE_FUSED_EP_KERNEL_IMPL=v3).

With tensor-parallel attention over the EP group (TP = EP = every rank, DP 1,
as GLM-5.2 serves) every rank holds all T tokens and the MoE must return all of
them, and GLM-5.2's router (sigmoid + e_score_correction_bias, renormalized
top-k, routed scale on the runner) is not v3's softmax selector. One op per
armed layer runs the whole MoE, as moe_tp_decode does under TP MoE:

- T >= MOE_FUSED_EP_V3_MIN_TOKENS: production's routing; mesh index m takes rows
  [m T/16, (m+1) T/16) (TokenReplicaLayout); v3 with routing=(ids, weights);
  one all-gather restores every row; the shared expert's TP slice all-reduced.
- smaller T (decode): production's EP path and its single all-reduce.

T is static per exported program. The op's output is reduced, so arming takes
the shared expert and the routed scale off the runner. `arm` runs at the end of
load_model and is collective over the TP group.
"""

import dataclasses
import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import jax_op
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    ep_token_replica_groups,
    get_ep_group,
)
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout
from vllm_torchtpu.kernels.fused_moe.v3 import (
    MESH_AXIS,
    WEIGHT_BLOCK_DEFAULT,
    Config,
    fused_ep_moe,
)
from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.kernels.router_topk import select as router_select
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.fused_moe_ep_v3 import (
    TILE_ROWS,
    WEIGHT_FORMAT,
    _unsupported_layer_reason,
    _unsupported_weights_reason,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OP_ATTR = "_fused_moe_ep_v3_tp_op"
# Non-persistent buffers of an armed layer: the shared expert's tensors, the
# f32 correction bias and this rank's mesh index.
_BUFFERS = (
    "_fused_moe_ep_v3_tp_sw13",
    "_fused_moe_ep_v3_tp_ss13",
    "_fused_moe_ep_v3_tp_sw2",
    "_fused_moe_ep_v3_tp_ss2",
    "_fused_moe_ep_v3_tp_bias",
    "_fused_moe_ep_v3_tp_rank",
)
# Without an activation block v3 holds whole experts in VMEM, which GLM-5.2's
# 16 local experts of 6144 x 2048 per rank do not fit (86.8 MiB of a 62.7 MiB
# budget at two slots); a 512-value block lets it stream each expert in column
# blocks (two slots of 1024 x 256 at that shape).
STREAM_ACTIVATION_BLOCK = 512
_SUM_FLOOR = 1e-20

_OPS: dict[tuple, Any] = {}
_RANK_BUFFER: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class _Topology:
    mesh: Any
    mesh_index: int
    rank_order: tuple[
        int, ...
    ]  # the EP rank whose expert block sits at each mesh index
    groups: tuple[tuple[int, ...], ...]  # token replica groups in mesh coordinates


@dataclasses.dataclass(frozen=True)
class _OpStatics:
    topk: int
    routed_scale: float
    activation: str
    use_sparse_core: bool
    onehot_threshold: int
    local_experts: int
    min_tokens: int
    config: Config


@dataclasses.dataclass(frozen=True)
class _Admitted:
    shared: tuple[torch.Tensor, ...]  # sw13, ss13, sw2, ss2
    statics: _OpStatics


def fused_moe_ep_v3_tp_supported(quant_method: Any) -> bool:
    """Whether arming gave this layer's quant method the op."""
    return getattr(quant_method, _OP_ATTR, None) is not None


def tp_attention_over_ep() -> bool:
    """Whether this deployment is tensor-parallel attention over the EP group:
    the TP group is every rank and the MoE is expert-parallel over the same
    ranks (DP 1). Read off the config: fused_moe_ep_v3's deployment check calls
    it at weight load, where a draft model's load may have narrowed the TP
    group coordinator."""
    from vllm.config import get_current_vllm_config

    par = get_current_vllm_config().parallel_config
    return (
        par.tensor_parallel_size > 1
        and par.data_parallel_size == 1
        and par.pipeline_parallel_size == 1
        and par.enable_expert_parallel
    )


def _topology() -> _Topology | str:
    """The EP mesh, this rank's mesh index, the EP rank order and the token
    replica groups, or a refusal reason. Collective (device ids, groups)."""
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if ep is None:
        return "no EP group"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    ep_ranks = tuple(int(r) for r in ep.ranks)
    if tp_ranks != ep_ranks:
        return f"TP ranks {tp_ranks} != EP ranks {ep_ranks} (needs dp 1, pcp 1)"
    world = torch.distributed.get_world_size()
    if len(tp_ranks) != world:
        return f"the TP group has {len(tp_ranks)} of {world} ranks"
    mesh = build_ep_mesh()
    mesh_index, order = ep_mesh_index(), ep_rank_order()
    assert mesh is not None and mesh_index is not None and order is not None
    if MESH_AXIS not in mesh.axis_names:
        return (
            f"the EP mesh's axes are {mesh.axis_names}; v3 splits experts over "
            f"{MESH_AXIS!r}"
        )
    groups = ep_token_replica_groups()
    if len(groups) != 1:
        return f"token replica groups {groups}: v3 here needs one group of every rank"
    return _Topology(
        mesh=mesh, mesh_index=mesh_index, rank_order=tuple(order), groups=groups
    )


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


def _v3_config(
    hidden: int, inter: int, local_experts: int, topk: int
) -> tuple[Config, str]:
    """Config.for_shape, or with STREAM_ACTIVATION_BLOCK when its experts do not
    fit VMEM whole, and v3's weight plan for the log. ValueError if neither."""
    config = Config.for_shape(
        hidden=hidden, inter=inter, experts_per_shard=local_experts, top_k=topk
    )

    def plan(cfg):
        def fit(token_block):
            return v3_layer.expert_buffers_that_fit(
                local_experts,
                TILE_ROWS,
                hidden,
                inter,
                config=cfg,
                weight_format=WEIGHT_FORMAT,
                weight_block=WEIGHT_BLOCK_DEFAULT,
                has_w1_bias=False,
                has_w2_bias=False,
                no_gate=False,
                activation_block=cfg.activation_block,
                result_rows=cfg.result_rows,
                token_block=token_block,
                token_rows=cfg.token_rows,
            )

        got = fit(cfg.token_rows_block)
        # A streamed expert takes one scale per token row, as fused_ep_moe plans it.
        return fit(0) if got[2] and cfg.token_block is None else got

    try:
        slots, prefetch, block, rows = plan(config)
    except ValueError:
        if config.activation_block:
            raise
        config = dataclasses.replace(config, activation_block=STREAM_ACTIVATION_BLOCK)
        slots, prefetch, block, rows = plan(config)
    stream = f"stream {block} columns x {rows} rows" if block else "whole experts"
    return config, f"{stream}, {slots} weight slots (prefetch {prefetch})"


def _admit(runner, topo: _Topology) -> _Admitted | str:
    """What arming needs for this runner, or why it keeps the usual path."""
    from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
        ZeroExpertRouter,
    )

    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    layer = runner.routed_experts
    qm = layer.quant_method
    cfg = runner.moe_config
    par = cfg.moe_parallel_config
    n_ranks = int(topo.mesh.size)
    if type(qm) is not VllmFp8MoEMethodTPU:
        return f"quant method {type(qm).__name__} is not VllmFp8MoEMethodTPU"
    activation = qm._tpu_activation_str
    if activation is None:
        return "process_weights_after_loading did not run for this layer"
    if enable_pipelined_collective_and_compute() or envs.TPU_MOE_FP8_ALL_GATHER:
        return (
            "the MoE path owns its collectives "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE or TPU_MOE_FP8_ALL_GATHER)"
        )
    if not par.use_ep or par.ep_size != n_ranks:
        return (
            f"needs EP MoE over {n_ranks} ranks (ep {par.ep_size}, use_ep {par.use_ep})"
        )
    if par.dp_size != 1 or par.pcp_size != 1 or cfg.is_sequence_parallel:
        return "needs dp 1, pcp 1 and no sequence-parallel MoE"
    if par.enable_eplb:
        return "EPLB is on"
    if runner.gate is None or runner.shared_expert_gate is not None:
        return "needs the runner to hold the gate and no shared-expert gate"
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
    ):
        return "a routed input or output transform is set"
    if cfg.skip_final_all_reduce:
        return "skip_final_all_reduce is set"
    if isinstance(runner.router, ZeroExpertRouter):
        return "zero-expert router"
    grouped = (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    )
    if layer.scoring_func != "sigmoid" or not layer.renormalize or grouped:
        return "routing is not sigmoid + renormalized ungrouped top-k"
    if layer.e_score_correction_bias is None:
        return "no e_score_correction_bias"
    if layer.custom_routing_function is not None or layer.apply_router_weight_on_input:
        return "custom routing function or router weight on the input"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return "routing simulation is on"
    if moe_routing._ROUTER_TOPK != "rowmax":
        return f"TPU_MOE_ROUTER_TOPK={moe_routing._ROUTER_TOPK} (needs rowmax)"
    if layer.routed_scaling_factor != 1.0:
        return f"layer routed_scaling_factor {layer.routed_scaling_factor} != 1"
    w1 = layer.w13_weight
    local_experts, hidden, inter = w1.shape[0], w1.shape[1], w1.shape[2] // 2
    reason = _unsupported_layer_reason(
        layer, local_experts, n_ranks
    ) or _unsupported_weights_reason(layer, local_experts, hidden, inter, activation)
    if reason is not None:
        return reason
    # v3 reads expert m * local + j as mesh index m's j-th expert; the op relabels
    # the router's ids from EP-rank order, so each EP rank's block must be its
    # contiguous experts.
    ep_rank = topo.rank_order[topo.mesh_index]
    if moe_routing.get_experts_start(layer) != ep_rank * local_experts:
        return (
            f"EP rank {ep_rank}'s first expert is "
            f"{moe_routing.get_experts_start(layer)}, not {ep_rank} x {local_experts}"
        )
    shared = _shared_slice(runner)
    if isinstance(shared, str):
        return shared
    topk = int(cfg.experts_per_token)
    try:
        config, plan = _v3_config(hidden, inter, local_experts, topk)
    except ValueError as ex:
        return f"v3 has no VMEM plan for this shape: {ex}"
    statics = _OpStatics(
        topk=topk,
        routed_scale=float(runner.routed_scaling_factor),
        activation=activation,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_threshold=resolve_onehot_permute_threshold(),
        local_experts=local_experts,
        min_tokens=int(envs.MOE_FUSED_EP_V3_MIN_TOKENS),
        config=config,
    )
    logger.info_once("Fused EP MoE v3 (TP attention): %s; %s", plan, config)
    return _Admitted(shared=shared, statics=statics)


def _route(scores, bias, is_pad, topk, dtype):
    """Production's routing on sigmoid scores: ids from the biased scores,
    renormalized unbiased weights in bf16, padded rows' weights zero."""
    _, ids = router_select(scores + bias.astype(jnp.float32)[None, :], topk)
    w = jnp.take_along_axis(scores, ids, axis=-1)
    total = jnp.sum(w, axis=-1, keepdims=True)
    # torch.clamp(min=1e-20) keeps NaN; a fused jnp.maximum may not (XLA CPU).
    total = jnp.where(jnp.isnan(total), total, jnp.maximum(total, _SUM_FLOOR))
    w = (w / total).astype(dtype)
    return ids, jnp.where(is_pad[:, None], jnp.zeros_like(w), w)


def _shared_expert(x, sw13, ss13, sw2, ss2):
    """This rank's TP slice of the shared expert: a bf16 partial, unreduced."""
    gate_up = xla_quantized_matmul(x, sw13, ss13)
    fs = sw2.shape[0]
    act = jax.nn.silu(gate_up[:, :fs].astype(jnp.float32)).astype(x.dtype)
    return xla_quantized_matmul(act * gate_up[:, fs:], sw2, ss2)


def _channel_scale(s):
    """[E, 1, ..., N] per-channel GMM scales as v3's [E, N]."""
    return s.reshape(s.shape[0], s.shape[-1])


def _build_fn(topo: _Topology, statics: _OpStatics):
    """The op's function over global (mesh-stacked) operands."""
    mesh, axis = topo.mesh, topo.mesh.axis_names[0]
    n_ranks = int(mesh.size)
    st = statics
    local = st.local_experts
    permuted = topo.rank_order != tuple(range(n_ranks))
    # Mesh index of each EP rank: v3 reads expert m * local + j as mesh index
    # m's j-th expert, while the router numbers experts by EP rank.
    mesh_of_rank = jnp.asarray(
        sorted(range(n_ranks), key=topo.rank_order.__getitem__), jnp.int32
    )
    per_rank = P(axis)
    rep = P()
    gmm_kw = dict(
        topk=st.topk,
        activation=st.activation,
        use_ep=True,
        use_sparse_core=st.use_sparse_core,
        onehot_moe_permute_threshold=st.onehot_threshold,
        skip_padded_tokens=True,  # admission requires TPU_MOE_SKIP_PADDED_TOKENS
    )
    # The torch bridge exports the op as one jitted function: its SPMD partitioner
    # refuses the layer's own inner jit, a side-effecting custom call without a
    # sharding and lax.axis_index, so the mesh index goes in as data
    # (fused_moe_ep_v3.py).
    v3_kw = dict(
        top_k=st.topk,
        renormalize=True,  # not read with routing=
        mesh=mesh,
        tile_rows=TILE_ROWS,
        weight_format=WEIGHT_FORMAT,
        activation="silu",
        config=st.config,
        experts_per_shard=local,
        jit=False,
        side_effects=False,
    )

    def ep_body(
        x, scores, bias, pad_mask, w13, w2, s13, s2, sw13, ss13, sw2, ss2, start, rank
    ):
        n = x.shape[0]
        is_pad = pad_mask[:n]  # replicated, so every rank zeroes the same rows
        ids, w = _route(scores, bias, is_pad, st.topk, x.dtype)
        routed = fused_moe_func(
            x, w13, w2, s13, s2, None, None, w, ids, start.reshape(()), **gmm_kw
        )
        shared = _shared_expert(x, sw13, ss13, sw2, ss2)
        return lax.psum(shared + routed * st.routed_scale, axis)

    def v3_front(layout):
        def body(x, scores, bias, pad_mask, sw13, ss13, sw2, ss2, rank):
            n = x.shape[0]
            is_pad = pad_mask[:n]
            ids, w = _route(scores, bias, is_pad, st.topk, x.dtype)
            if permuted:
                ids = mesh_of_rank[ids // local] * local + ids % local
            ids = jnp.where(
                is_pad[:, None], jnp.full_like(ids, -1), ids
            )  # route nowhere
            me = rank.reshape(())
            shared = _shared_expert(x, sw13, ss13, sw2, ss2)
            return (
                layout.split(x, me),
                layout.split(ids, me, fill_value=-1),
                layout.split(w, me),
                shared,
            )

        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(per_rank, per_rank, rep, rep) + (per_rank,) * 5,
            out_specs=(per_rank,) * 4,
            check_vma=False,
        )

    def v3_back(layout):
        def body(y, shared):
            return lax.psum(shared, axis) + layout.restore(y, axis) * st.routed_scale

        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(per_rank,) * 2,
            out_specs=per_rank,
            check_vma=False,
        )

    specs = (per_rank, per_rank, rep, rep) + (per_rank,) * 10
    ep_path = jax.shard_map(
        ep_body, mesh=mesh, in_specs=specs, out_specs=per_rank, check_vma=False
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
        start: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        args = (
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
            start,
            rank,
        )
        args = tuple(
            jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs)
        )
        (
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
            start,
            rank,
        ) = args
        n = (
            x.shape[0] // n_ranks
        )  # static: torch_tpu exports one program per token count
        if n < st.min_tokens:
            return ep_path(*args)
        layout = TokenReplicaLayout.create(ep=n_ranks, input_rows=n, groups=topo.groups)
        x_l, ids_l, w_l, shared = v3_front(layout)(
            x, scores, bias, pad_mask, sw13, ss13, sw2, ss2, rank
        )
        s13_c, s2_c = _channel_scale(s13), _channel_scale(s2)
        routing = (ids_l, w_l)
        y = fused_ep_moe(
            x_l, w13, w2, s13_c, s2_c, None, routing=routing, shard_index=rank, **v3_kw
        )
        return v3_back(layout)(y, shared)

    return fn, specs


def _get_op(topo: _Topology, statics: _OpStatics):
    """One jax_op per distinct closure; built at arming, never in a trace."""
    key = (topo, statics)
    op = _OPS.get(key)
    if op is not None:
        return op
    fn, specs = _build_fn(topo, statics)
    name = "pallas::fused_moe_ep_v3_tp" + (f"_{len(_OPS)}" if _OPS else "")
    # torch_tpu's jax_op stub types the specs as tuple[tuple[str, ...], ...],
    # but they go to NamedSharding, which needs a PartitionSpec (as in
    # fused_moe_ep_v3.py).
    # pyrefly: ignore[no-matching-overload]
    op = jax_op(
        name,
        fn,
        mesh=topo.mesh,
        input_partition_specs=specs,
        output_partition_specs=P(topo.mesh.axis_names[0]),
    )
    # vLLM traces with a dynamic token dimension; rows in == rows out.
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    _OPS[key] = op
    return op


def _rank_tensor(device: torch.device, mesh_index: int) -> torch.Tensor:
    """This rank's mesh index as int32 [1, 1] data (one tensor per process), so
    every rank compiles the same program."""
    global _RANK_BUFFER
    if _RANK_BUFFER is None:
        _RANK_BUFFER = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    return _RANK_BUFFER


def _gather(obj) -> list:
    """all_gather_object over the TP group's CPU group."""
    tp = parallel_state.get_tp_group()
    if tp.world_size == 1:
        return [obj]
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def _check_ranks_agree(entry: dict) -> None:
    """Raise unless every TP rank arms the same layers with the same statics:
    v3's exchange and the all-reduces are collectives every rank must join."""
    seen = _gather(entry)
    if any(s["sig"] != seen[0]["sig"] for s in seen):
        raise RuntimeError(
            f"fused_moe_ep_v3_tp: TP ranks arm different layers; their programs "
            f"and v3's exchange would not match. Per rank: {seen}"
        )


def arm(model: torch.nn.Module, draft_model: torch.nn.Module | None = None) -> int:
    """Arm every admitted MoE layer of `model` and of `draft_model` (the MTP
    drafter); returns how many were armed. Collective over the TP group."""
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
        if fused_moe_ep_v3_tp_supported(runner._quant_method):
            continue
        if isinstance(topo, str):
            res = topo
        elif (
            not envs.TPU_MOE_SKIP_PADDED_TOKENS or token_padding._padding_state is None
        ):
            res = "needs TPU_MOE_SKIP_PADDED_TOKENS=1 (padded rows come from its mask)"
        else:
            res = _admit(runner, topo)
        if isinstance(res, str):
            refused[res] = refused.get(res, 0) + 1
        else:
            plan.append((runner, res))
    entries = [(r.layer_name, repr(adm.statics)) for r, adm in plan]
    sig = hashlib.sha256(
        repr((entries, len(target), len(drafter))).encode()
    ).hexdigest()[:16]
    _check_ranks_agree(dict(sig=sig, armed=len(plan)))
    for runner, adm in plan:
        assert not isinstance(topo, str)
        layer = runner.routed_experts
        # An alias of the f32 parameter: to() and contiguous() are no-ops.
        bias = layer.e_score_correction_bias.detach().to(torch.float32).contiguous()
        rank = _rank_tensor(layer.w13_weight.device, topo.mesh_index)
        for name, tensor in zip(_BUFFERS, (*adm.shared, bias, rank)):
            layer.register_buffer(name, tensor, persistent=False)
        setattr(layer.quant_method, _OP_ATTR, _get_op(topo, adm.statics))
        runner._shared_experts = None
        runner._forward_entry = runner._select_forward()
        runner.routed_scaling_factor = 1.0
    armed_drafter = sum(1 for r, _ in plan if id(r) not in known)
    log = logger.info if plan else logger.warning
    log(
        "fused_moe_ep_v3_tp: armed %d/%d target and %d/%d drafter MoE layers "
        "(v3 at >= %d tokens, production EP below; mesh index %s, sig %s); refused: %s",
        len(plan) - armed_drafter,
        len(target),
        armed_drafter,
        len(drafter),
        int(envs.MOE_FUSED_EP_V3_MIN_TOKENS),
        None if isinstance(topo, str) else topo.mesh_index,
        sig,
        refused or "none",
    )
    return len(plan)


def fused_moe_ep_v3_tp_forward(
    quant_method: Any, layer: Any, x: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    """Forward of an armed layer: shared + routed_scaling_factor * routed,
    already reduced over the TP group, on every rank's rows."""
    padding = token_padding._padding_state
    assert padding is not None, "fused_moe_ep_v3_tp needs the token-padding mask"
    sw13, ss13, sw2, ss2, bias, rank = (getattr(layer, n) for n in _BUFFERS)
    # Production's scoring op; everything after it is the op.
    scores = moe_routing._apply_scoring_fn(layer.scoring_func, router_logits)
    return getattr(quant_method, _OP_ATTR)(
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
        layer._experts_start.reshape(1, 1),
        rank,
    )
