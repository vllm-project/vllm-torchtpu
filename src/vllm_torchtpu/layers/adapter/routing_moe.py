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
"""Torch bridge for tokamax's routing MoE kernel (USE_ROUTING_MOE=1).

tokamax's routing_moe (tokamax/_src/ops/experimental/routing_moe) all-gathers
the tokens over the EP axis; each device then gathers its experts' token rows
by index, runs gate/up and down, and sends each result row straight to the
device that owns the token, which sums the token's top-k rows weighted by the
routing weights. No device builds a full [tokens, hidden] partial sum and there
is no reduce-scatter.

Like the fused EP path (fused_moe_ep.py), the whole layer is one op over the EP
mesh that takes this rank's own tokens and returns this rank's own tokens, built
at weight load. Ranks of one TP group hold the same token rows; each passes the
kernel its own share of them (TokenReplicaLayout), so every token is routed
once, and the group all-gathers the results. The experts are picked here as
moe_router.select_experts picks them: score, top-k, renormalize.

ROUTING_MOE_REMOTE_SCATTER=0 selects the kernel's local mode instead of the remote
scatter above: each device sums its experts' rows into a partial result for
every token, and a reduce-scatter combines the partial results.

ROUTING_MOE_FP8_COMBINE=1 has routing_gmm write its expert-output rows in fp8
with per-lane f32 scales (tokamax's lane-strided format, `scatter_format`)
instead of bf16, and output_reduce dequantizes them while summing each token's
top-k rows. With remote scatter this halves the bytes sent to the token owners;
in local mode the rows stay on the device, so it saves only HBM traffic. The
rows are quantized, so the output is close to, not equal to, bf16 combine.

ROUTING_MOE_FP8_DISPATCH=1 all-gathers the tokens in fp8 with one f32 scale per
token instead of bf16 (as fused_moe_ep's rowquant), and dequantizes them before
tokamax's layer, for token buckets where each device routes at least
ROUTING_MOE_FP8_DISPATCH_MIN_ROWS rows (decode buckets keep the bf16 gather).
Per-row scales keep every array 2-D: a per-column-block scale needs a
[tokens, blocks, 128] view, which on TPU's tiled layouts is a physical relayout
of the whole gathered array (measured: costlier than the bf16 gather).
tokamax's own sequence-parallel path gathers bf16 rows (its input_pack and
routing_gmm take no quantized x), so this path gathers itself and runs
tokamax's per-device layer (`_routing_moe_local`) with the rows replicated,
keeping remote scatter's local-token output (no output collective).

ROUTING_MOE_GMM_PLAN passes tokamax RoutingGmmPlan overrides (tiling, x
quantization block, accumulation dtype, ...), optionally per token count.
"""

import dataclasses
import json
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    ep_token_replica_groups,
)
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ROUTING_MOE_OP_NAME = "pallas::routing_moe"

# The attribute prebuild's op is recorded under, on the quant method that owns
# the layer (one per MoE layer), as fused_moe_ep.FUSED_MOE_EP_OP_ATTR.
ROUTING_MOE_OP_ATTR = "_routing_moe_op"

# One op per distinct closure, shared by the layers that agree on it, as in
# fused_moe_ep._OPS.
_OPS: dict[tuple, Any] = {}

# This rank's index in the EP mesh, as a device tensor. The kernel takes it as
# data: `lax.axis_index` lowers to `partition-id`, which torch_tpu's jax.export
# bridge rejects (see fused_moe_ep._RANK_BUFFER).
_RANK_BUFFER: torch.Tensor | None = None

# fp8 combine rows pack 4 fp8 values per 32-bit lane across 128 lanes, so the
# hidden size must fill whole 512-element chunks (tokamax rejects others).
_FP8_COMBINE_CHUNK = 512

_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)


def routing_moe_supported(owner: Any) -> bool:
    """Whether prebuild_routing_moe armed this layer's quant method."""
    return getattr(owner, ROUTING_MOE_OP_ATTR, None) is not None


def unsupported_reason(layer) -> str | None:
    """Why this integration can't run the layer, or None if it can.

    Only what the integration itself doesn't handle; the kernel raises for what
    it doesn't support.
    """
    if not layer.moe_config.moe_parallel_config.use_ep:
        return "the experts are not expert-parallel"
    if layer.scoring_func not in ("softmax", "sigmoid"):
        return f"routing here doesn't score with {layer.scoring_func}"
    if layer.e_score_correction_bias is not None:
        return "the score correction bias isn't wired up here"
    if float(layer.routed_scaling_factor) != 1.0:
        return "routed_scaling_factor isn't wired up here"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing isn't wired up here"
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases aren't passed to the kernel here"
    hidden = layer.w13_weight.shape[1]
    if envs.ROUTING_MOE_FP8_COMBINE and hidden % _FP8_COMBINE_CHUNK:
        return (
            f"ROUTING_MOE_FP8_COMBINE needs a hidden size divisible by "
            f"{_FP8_COMBINE_CHUNK}, not {hidden}"
        )
    if envs.ROUTING_MOE_FP8_DISPATCH and not envs.ROUTING_MOE_REMOTE_SCATTER:
        return "ROUTING_MOE_FP8_DISPATCH needs remote scatter"
    return None


def _route(
    gating: jax.Array, topk: int, renormalize: bool, scoring_fn: str
) -> tuple[jax.Array, jax.Array]:
    """moe_router.select_experts for the scoring functions admitted above."""
    gating = gating.astype(jnp.float32)
    if scoring_fn == "softmax":
        scores = jax.nn.softmax(gating, axis=-1)
    else:
        scores = jax.nn.sigmoid(gating)
    weights, ids = lax.top_k(scores, topk)
    if renormalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return weights, ids.astype(jnp.int32)


def _relabel_table(order: tuple[int, ...], local_experts: int) -> list[int]:
    """Kernel expert id of each global expert id.

    Mesh index m holds the experts of EP rank order[m], and the kernel reads
    expert id e as mesh index e // local_experts.
    """
    index = {rank: m for m, rank in enumerate(order)}
    return [
        index[g // local_experts] * local_experts + g % local_experts
        for g in range(len(order) * local_experts)
    ]


def _scatter_format(fp8_combine: bool) -> Any | None:
    """tokamax's `RoutingMoEPlan.scatter_format`: routing_gmm's output rows
    (what remote scatter sends to the token owners) in fp8 with one f32 scale
    per lane for ROUTING_MOE_FP8_COMBINE, else None (bf16 rows)."""
    if not fp8_combine:
        return None
    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.common import (
        out_format as tk_out_format,
    )

    return tk_out_format.ActivationFormat(
        dtype=jnp.float8_e4m3fn, quant=tk_out_format.LaneStrided()
    )


def parse_gmm_plan(text: str) -> tuple:
    """ROUTING_MOE_GMM_PLAN as a hashable spec: ((max_rows, fields), ...),
    first match wins, max_rows None for the default. Fields are tokamax
    RoutingGmmPlan field names; an unknown name raises when the plan is built.
    """
    if not text.strip():
        return ()
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"ROUTING_MOE_GMM_PLAN must be a JSON object: {text!r}")

    def frozen(fields: dict) -> tuple:
        if not isinstance(fields, dict):
            raise ValueError(f"ROUTING_MOE_GMM_PLAN fields must be objects: {fields!r}")
        return tuple(sorted(fields.items()))

    if "by_rows" not in raw and "default" not in raw:
        return ((None, frozen(raw)),)
    unknown = set(raw) - {"by_rows", "default"}
    if unknown:
        raise ValueError(f"ROUTING_MOE_GMM_PLAN: unknown keys {sorted(unknown)}")
    spec = tuple(
        (int(max_rows), frozen(fields))
        for max_rows, fields in sorted(raw.get("by_rows", []), key=lambda e: e[0])
    )
    if "default" in raw:
        spec += ((None, frozen(raw["default"])),)
    return spec


def gmm_plan_fields(spec: tuple, rows: int) -> dict | None:
    """The plan fields `spec` (parse_gmm_plan) gives a device contributing
    `rows` token rows, or None for tokamax's heuristics."""
    for max_rows, fields in spec:
        if max_rows is None or rows <= max_rows:
            return dict(fields)
    return None


def _quantize_rows(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """[t, H] -> fp8 [t, H] and one f32 scale per row, [t, 1]."""
    xf = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
    scale = jnp.where(amax > 0, amax / _FP8_MAX, 1.0)
    return (xf * (1.0 / scale)).astype(jnp.float8_e4m3fn), scale


def _dequantize_rows(q: jax.Array, scale: jax.Array, dtype) -> jax.Array:
    return (q.astype(jnp.float32) * scale).astype(dtype)


def _routing_moe_jax(
    mesh,
    x: jax.Array,
    weights: jax.Array,
    ids: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    ep_shard_idx: jax.Array,
    *,
    activation: str,
    remote_scatter: bool,
    fp8_combine: bool,
    fp8_dispatch: bool,
    gmm_plan: tuple,
    fp8_dispatch_min_rows: int = 0,
) -> jax.Array:
    """tokamax routing_moe over the EP mesh on this rank's routed rows (x, top-k
    weights and ids sharded on axis 0); returns this rank's rows. Pure JAX, so
    it lowers off TPU (with an abstract mesh) as well as inside the op."""
    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.gmm import plan as tk_gmm_plan

    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.layer import config as tk_config

    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.layer import modular as tk_modular

    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.layer import plan as tk_plan

    ep = mesh.shape[EP_AXIS_NAME]
    rows = x.shape[0] // ep
    fields = gmm_plan_fields(gmm_plan, rows)
    plan = None if fields is None else tk_gmm_plan.RoutingGmmPlan(**fields)
    moe_plan = tk_plan.RoutingMoEPlan(
        remote_scatter=remote_scatter, scatter_format=_scatter_format(fp8_combine)
    )
    if not (fp8_dispatch and rows >= fp8_dispatch_min_rows):
        return tk_modular.routing_moe(
            x,
            weights,
            ids,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            mesh=mesh,
            activation=activation,
            tp_axis=None,
            ep_axis=EP_AXIS_NAME,
            use_sp=True,
            moe_plan=moe_plan,
            gmm_plan=plan,
            ep_shard_idx=ep_shard_idx,
        )

    # fp8 dispatch: tokamax's layer with the rows replicated (use_sp=False), so
    # it gathers nothing itself; with remote scatter its per-device result is
    # already this rank's tokens, so the replicated layout's output all-gather
    # is dropped.
    class _LocalOutput(tk_config.RoutingMoEConfig):
        @property
        def output_collective(self):
            return tk_config.OutputCollective.NONE

    base = tk_config.RoutingMoEConfig.create(
        tp_axis=None,
        ep_axis=EP_AXIS_NAME,
        use_sp=False,
        reduce_output=True,
        activation=activation,
        plan=tk_plan.ResolvedMoEPlan.create(moe_plan, x_dtype=x.dtype),
        mesh=mesh,
        w1_scale_shape=w1_scale.shape,
        w2_scale_shape=w2_scale.shape,
    )
    cfg = _LocalOutput(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    )

    def local(x_l, w_l, ids_l, w1_l, w2_l, s1_l, s2_l, idx_l):
        q, scale = _quantize_rows(x_l)
        q = lax.all_gather(q, EP_AXIS_NAME, axis=0, tiled=True)
        scale = lax.all_gather(scale, EP_AXIS_NAME, axis=0, tiled=True)
        ids_all = lax.all_gather(ids_l, EP_AXIS_NAME, axis=0, tiled=True)
        # _routing_moe_local slices this rank's weight rows back out.
        w_all = lax.all_gather(w_l, EP_AXIS_NAME, axis=0, tiled=True)
        return tk_modular._routing_moe_local(
            _dequantize_rows(q, scale, x_l.dtype),
            w_all,
            ids_all,
            w1_l,
            w2_l,
            s1_l,
            s2_l,
            None,
            None,
            idx_l,
            cfg=cfg,
            gmm_plan=plan,
        )

    rows_spec = PartitionSpec(EP_AXIS_NAME)
    w_spec = PartitionSpec(EP_AXIS_NAME, None, None)
    s_spec = PartitionSpec(EP_AXIS_NAME, None, None, None)
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(rows_spec,) * 3 + (w_spec, w_spec, s_spec, s_spec, rows_spec),
        out_specs=rows_spec,
        check_vma=False,
    )(x, weights, ids, w1, w2, w1_scale, w2_scale, ep_shard_idx)


def _build_op(
    mesh,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    activation: str,
    mesh_expert_order: tuple[int, ...] | None,
    replica_groups: tuple[tuple[int, ...], ...],
    remote_scatter: bool,
    *,
    fp8_combine: bool = False,
    fp8_dispatch: bool = False,
    gmm_plan: tuple = (),
    fp8_dispatch_min_rows: int = 0,
):
    key = (
        topk,
        renormalize,
        scoring_fn,
        activation,
        mesh_expert_order,
        replica_groups,
        remote_scatter,
        fp8_combine,
        fp8_dispatch,
        gmm_plan,
        fp8_dispatch_min_rows,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached
    # tokamax is imported inside _routing_moe_jax, where USE_ROUTING_MOE has
    # asked for the kernel: this module is imported for every model.
    # A distinct torch op name per closure: two ops registered under one name
    # would share a dispatch entry (see fused_moe_ep._build_op).
    name = ROUTING_MOE_OP_NAME if not _OPS else f"{ROUTING_MOE_OP_NAME}_{len(_OPS)}"
    ep = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    # Every operand is annotated jax.Array: torch_tpu infers static argnums
    # from the signature and refuses an unannotated one.
    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        layout = TokenReplicaLayout.create(
            ep=ep, input_rows=x.shape[0] // ep, groups=replica_groups
        )
        relabel = (
            None
            if mesh_expert_order is None
            else jnp.asarray(
                _relabel_table(mesh_expert_order, w1.shape[0] // ep), jnp.int32
            )
        )

        def split_and_route(x_l, gating_l, rank_l):
            me = rank_l[0, 0]
            weights, ids = _route(
                layout.split(gating_l, me), topk, renormalize, scoring_fn
            )
            if relabel is not None:
                ids = relabel[ids]
            return layout.split(x_l, me), weights, ids

        x, weights, ids = jax.shard_map(
            split_and_route,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )(x, gating, rank)
        out = _routing_moe_jax(
            mesh,
            x,
            weights,
            ids,
            w1,
            w2,
            w1_scale,
            w2_scale,
            rank.reshape(-1),
            activation=activation,
            remote_scatter=remote_scatter,
            fp8_combine=fp8_combine,
            fp8_dispatch=fp8_dispatch,
            gmm_plan=gmm_plan,
            fp8_dispatch_min_rows=fp8_dispatch_min_rows,
        )
        return jax.shard_map(
            lambda out_l: layout.restore(out_l, EP_AXIS_NAME),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_vma=False,
        )(out)

    op = jax_op(
        name,
        moe,
        mesh=mesh,
        # x, w1, w2, w1_scale, w2_scale, gating, rank; the output is sharded on
        # axis 0 like x.
        input_partition_specs=(spec,) * 7,
        output_partition_specs=spec,
    )

    # The op returns the rows it was given, so the output shape is known
    # without exporting (as fused_moe_ep's fake).
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def _refusal(reason: str) -> ValueError:
    return ValueError(
        f"routing_moe can't run this MoE layer: {reason}. "
        "Unset USE_ROUTING_MOE to use another MoE kernel."
    )


def prebuild_routing_moe(
    layer, topk: int, renormalize: bool, activation: str
) -> Any | None:
    """Build the routing MoE op for one layer, or None if USE_ROUTING_MOE is
    off. Raises for a layer it can't run rather than falling back to another
    backend.

    Called from process_weights_after_loading: resolving the mesh gathers from
    every rank, which a compiled forward can't do.
    """
    if not envs.USE_ROUTING_MOE:
        return None
    if envs.USE_MOE_FUSED_EP_KERNEL:
        raise ValueError("USE_ROUTING_MOE and USE_MOE_FUSED_EP_KERNEL are exclusive")
    reason = unsupported_reason(layer)
    if reason is not None:
        raise _refusal(reason)
    mesh = build_ep_mesh()
    order = ep_rank_order()
    if mesh is None or order is None:
        raise _refusal("there is no EP group")
    ep = mesh.shape[EP_AXIS_NAME]
    local_experts = layer.w13_weight.shape[0]
    if layer.global_num_experts != local_experts * ep:
        raise _refusal(
            f"{layer.global_num_experts} experts don't split evenly over "
            f"{ep} EP ranks of {local_experts}"
        )

    global _RANK_BUFFER
    _RANK_BUFFER = torch.tensor(
        [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
    )
    order = tuple(order)
    replica_groups = ep_token_replica_groups(
        is_sequence_parallel=layer.moe_config.moe_parallel_config.is_sequence_parallel
    )
    logger.info_once(
        "[MoE]: routing_moe over %d devices, %d experts each, token replica "
        "groups %s; %s scatter, %s combine rows, %s dispatch rows (from %d rows "
        "per device); gmm plan %s.",
        ep,
        local_experts,
        replica_groups,
        "remote" if envs.ROUTING_MOE_REMOTE_SCATTER else "local",
        "fp8" if envs.ROUTING_MOE_FP8_COMBINE else "bf16",
        "fp8" if envs.ROUTING_MOE_FP8_DISPATCH else "bf16",
        envs.ROUTING_MOE_FP8_DISPATCH_MIN_ROWS,
        envs.ROUTING_MOE_GMM_PLAN or "tokamax heuristics",
    )
    return _build_op(
        mesh,
        topk,
        renormalize,
        layer.scoring_func,
        activation,
        None if order == tuple(range(ep)) else order,
        replica_groups,
        envs.ROUTING_MOE_REMOTE_SCATTER,
        fp8_combine=envs.ROUTING_MOE_FP8_COMBINE,
        fp8_dispatch=envs.ROUTING_MOE_FP8_DISPATCH,
        gmm_plan=parse_gmm_plan(envs.ROUTING_MOE_GMM_PLAN),
        fp8_dispatch_min_rows=envs.ROUTING_MOE_FP8_DISPATCH_MIN_ROWS,
    )


def routing_moe(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer through the routing MoE op prebuild recorded on
    `owner`, the layer's quant method.

    hidden_states [tokens, hidden] and router_logits [tokens, experts] are this
    rank's; w1 [E, H, 2I] (gate | up), w2 [E, I, H] and their scales
    [E, blocks, 1, N] are this rank's experts. Returns this rank's
    [tokens, hidden].
    """
    op = getattr(owner, ROUTING_MOE_OP_ATTR)
    return op(hidden_states, w1, w2, w1_scale, w2_scale, router_logits, _RANK_BUFFER)
