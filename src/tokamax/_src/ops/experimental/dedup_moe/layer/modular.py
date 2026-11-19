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
# ==============================================================================
r"""Model-facing entry point of the De-Dup MoE layer.

A model's MoE layer calls

    topology = BoardTopology.uniform(num_devices=ep, board_size=bs)  # optional
    cfg = DedupMoeConfig(ep=ep, hidden=D, num_experts=E, top_k=K, ffn=F,
                         topology=topology)
    prepared = prepare_weights(mesh, w1, w2, w1_scale, w2_scale, cfg=cfg)
    y = dedup_moe(x, router_logits, prepared, cfg)   # inside jax.jit

and always runs ONE kernel configuration per (topology, t bucket, weight
format), chosen statically at trace time by `cfg.policy` (layer/plan.py):

  `auto` (default): a measured mode table where one covers the point, and
      the formula config (`plan.derive_config`) everywhere else.
  `rules`: always `plan.derive_config`.
  `<mode name>`: pins a mode (mode_registry.MODES).

Without a topology every policy runs the direct wave dispatch. All policies
run the same arithmetic and give bit-identical outputs.

Everything here is static at trace time: t_local comes from x's shape and is
rounded up to a t bucket, and the layer is `monolithic.dedup_ep_moe` with
the plan's keyword arguments.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.layer import config as layer_config
from tokamax._src.ops.experimental.dedup_moe.layer import monolithic
from tokamax._src.ops.experimental.dedup_moe.layer import plan as layer_plan

AXIS = layer_config.AXIS

# Re-export config and plan symbols so `layer.modular` is a complete surface.
DEFAULT_T_BUCKETS = layer_config.DEFAULT_T_BUCKETS
REP_BUCKETS = monolithic.REP_BUCKETS
build_rep_lut = monolithic.build_rep_lut
UnifiedRules = layer_config.UnifiedRules
DedupMoeConfig = layer_config.DedupMoeConfig
Config = layer_config.Config
PreparedMoe = layer_config.PreparedMoe
PreparedWeights = layer_config.PreparedMoe
UnifiedPlan = layer_plan.UnifiedPlan
plan = layer_plan.plan
compiler_options = layer_plan.compiler_options

PreparedMoe.rep_lut = property(  # type: ignore[attr-defined]
    lambda self: (
        self.orig_to_slot
        if (
            self.orig_to_slot is not None
            and getattr(self.orig_to_slot, "ndim", 1) >= 2
        )
        else None
    )
)


def prepare_weights(
    mesh: Any,
    w1: Any,
    w2: Any,
    w1_scale: Any = None,
    w2_scale: Any = None,
    *,
    cfg: DedupMoeConfig,
    ws1: Any = None,
    ws2: Any = None,
    placement: Sequence[int] | Any | None = None,
    rep_lut: Any | None = None,
    rep_offsets: Any | None = None,
    rep_slots: Any | None = None,
    logical_to_physical_map: Any | None = None,
) -> PreparedMoe:
  """One-time offline step: weights into slot order plus placement/replica maps.

  Applies the slot -> original expert mapping (`placement`, default
  cfg.placement) to the expert axis of w1, w2 and their scales, and keeps
  the maps `modular.dedup_moe` uses to route logical expert selections to
  physical slots (including quota-balanced replica LUT routing when experts
  are replicated via `lpt_rep` / `hca_rep`). Identity without a placement.

  Args:
    mesh: One-axis EP mesh; expert stacks are sharded on it.
    w1: [E, D, 2F] expert gate+up weights (original expert order).
    w2: [E, F, D].
    w1_scale: [E, 2F] f32 per-channel scales (fp8), or None (bf16).
    w2_scale: [E, D] or None.
    cfg: The layer config.
    ws1: Optional shared expert [D, 2Fs] (replicated).
    ws2: Optional shared expert [Fs, D].
    placement: [E_slots] slot -> original expert id, or None for cfg.placement.
    rep_lut: Optional [E, nb, REP_BUCKETS] int32 quota-balanced replica LUT.
    rep_offsets: Optional [E + 1] int32 CSR offsets into `rep_slots`.
    rep_slots: Optional [N_slots] int32 physical replica slots per expert.
    logical_to_physical_map: Optional [E], [E, B], or [E, nb, B] int32 map from
      logical expert id to physical slot id.

  Returns:
    The prepared weights.

  Raises:
    ValueError: If cfg.include_shared_expert does not match ws1 and ws2, or if
      the placement / replica maps have invalid shapes or indices.
  """
  if cfg.include_shared_expert != (ws1 is not None):
    raise ValueError("cfg.include_shared_expert must match passing ws1/ws2")
  e_total = cfg.num_experts
  p_raw = cfg.placement if placement is None else placement
  has_rep_args = (
      rep_lut is not None
      or (rep_offsets is not None and rep_slots is not None)
      or logical_to_physical_map is not None
  )
  if p_raw is None and not has_rep_args:
    return layer_config.prepare_weights(
        mesh,
        w1,
        w2,
        w1_scale,
        w2_scale,
        cfg=cfg,
        ws1=ws1,
        ws2=ws2,
        placement=None,
    )
  shard = NamedSharding(mesh, P(AXIS))
  rep = NamedSharding(mesh, P())
  ident = jnp.arange(e_total, dtype=jnp.int32)
  if p_raw is not None:
    p = jnp.asarray(p_raw, dtype=jnp.int32)
    if p.ndim != 1:
      raise ValueError(f"placement must be 1-D; got {p.shape}")
    if len(p) % cfg.ep != 0:
      raise ValueError(
          f"placement length {len(p)} must be divisible by ep={cfg.ep}"
      )
    if bool(jnp.any((p < 0) | (p >= e_total))):
      raise ValueError(f"placement entries must be in [0, {e_total}); got {p}")
    is_perm = p.shape == (e_total,) and bool(
        jnp.array_equal(jnp.sort(p), ident)
    )
    if is_perm and not has_rep_args:
      return layer_config.prepare_weights(
          mesh,
          w1,
          w2,
          w1_scale,
          w2_scale,
          cfg=cfg,
          ws1=ws1,
          ws2=ws2,
          placement=tuple(int(i) for i in p.tolist()),
      )
    if bool(jnp.array_equal(p, ident)):
      w1_s, w2_s, w1_scale_s, w2_scale_s = w1, w2, w1_scale, w2_scale
      s2o = None
    else:
      idx = jax.device_put(p, rep)
      take = jax.jit(lambda w, i: jnp.take(w, i, axis=0), out_shardings=shard)
      w1_s, w2_s, w1_scale_s, w2_scale_s = (
          None if w is None else take(w, idx)
          for w in (w1, w2, w1_scale, w2_scale)
      )
      s2o = idx
  else:
    p = ident
    w1_s, w2_s, w1_scale_s, w2_scale_s = w1, w2, w1_scale, w2_scale
    s2o = None

  if logical_to_physical_map is not None:
    lut_arr = jnp.asarray(logical_to_physical_map, dtype=jnp.int32)
  elif rep_lut is not None:
    lut_arr = jnp.asarray(rep_lut, dtype=jnp.int32)
  else:
    board_size = (
        cfg.topology.board_size
        if cfg.topology is not None
        else min(8, cfg.ep)
    )
    lut_arr = jnp.asarray(
        build_rep_lut(
            p,
            ep=cfg.ep,
            e_total=e_total,
            board_size=board_size,
            rep_offsets=rep_offsets,
            rep_slots=rep_slots,
        ),
        dtype=jnp.int32,
    )
  o2s = jax.device_put(lut_arr, rep)
  return PreparedMoe(
      w1_s, w2_s, w1_scale_s, w2_scale_s, ws1, ws2, s2o, o2s, mesh
  )

# A pad row of the gating logits: 0 at the row's K chosen (local) experts,
# this elsewhere, so softmax top-k picks exactly those K.
_PAD_LOGIT = -1.0e4


def _pad_expert_ids(me, n_pad: int, *, g_local: int, board: int, topk: int):
  """[n_pad, K] expert ids for pad rows: this device's experts first.

  Pad j takes local experts (j + i) % g_local for i < g_local, then the
  experts of the next devices on the same board, so pad rows send no rows
  off the board (none off the device when g_local >= K).

  Args:
    me: This device's index (traced).
    n_pad: Pad rows.
    g_local: Experts per device.
    board: Devices per board (ep without a topology).
    topk: K.

  Returns:
    [n_pad, K] int32 expert ids.

  Raises:
    ValueError: If K exceeds the experts on one board.
  """
  j = jnp.arange(n_pad, dtype=jnp.int32)[:, None]
  i = jnp.arange(topk, dtype=jnp.int32)[None, :]
  dev_in_board = me % board
  hop = i // g_local
  dev = (me - dev_in_board) + (dev_in_board + hop) % board
  if topk > g_local * board:
    raise ValueError(f"top_k={topk} > experts on one board")
  return dev * g_local + (j + i) % g_local


def dedup_moe(
    x: jax.Array,
    router_logits: jax.Array,
    prepared: PreparedMoe,
    cfg: DedupMoeConfig,
    *,
    topk_ids: jax.Array | None = None,
    kernel_options: Mapping[str, Any] | None = None,
    combine_options: Mapping[str, Any] | None = None,
) -> jax.Array:
  """One MoE layer: [T, D] tokens -> [T, D], T = ep * t_local.

  jit-friendly; every decision is static (from shapes and cfg). x, the
  router logits and topk_ids are token-sharded over the mesh axis (each
  device holds its t_local rows).

  Args:
    x: [T, D] bf16 activations.
    router_logits: [T, E] router logits in ORIGINAL expert order (softmax top-k
      with renormalization).
    prepared: From prepare_weights.
    cfg: The layer config.
    topk_ids: Optional [T, K] expert ids (original order) chosen by the caller;
      the weights are then the softmax scores at those ids.
    kernel_options: Optional mapping of `FusedKernelConfig` options forwarded to
      `monolithic.dedup_ep_moe`.
    combine_options: Optional mapping of `output_reduce.Config` options
      forwarded to `monolithic.dedup_ep_moe`.

  Returns:
    [T, D], sharded like x.

  Raises:
    ValueError: If the tokens do not split over the mesh axis.
  """
  ep = cfg.ep
  n, _ = x.shape
  if n % ep:
    raise ValueError(f"{n} tokens do not split over ep={ep}")
  t_local = n // ep
  p = plan(cfg, t_local)
  return _run_plan(
      x,
      router_logits,
      prepared,
      cfg,
      p,
      t_local,
      topk_ids,
      kernel_options=kernel_options,
      combine_options=combine_options,
  )


def _run_plan(
    x,
    gating,
    prepared,
    cfg,
    p,
    t_local,
    topk_ids,
    *,
    kernel_options=None,
    combine_options=None,
):
  """Pads to the bucket, remaps to slot order, runs the plan, unpads."""
  mesh, ep, k = prepared.mesh, cfg.ep, cfg.top_k
  num_slots = int(prepared.w1.shape[0])
  tb, g_local = p.t_bucket, num_slots // ep
  board = cfg.topology.board_size if cfg.topology is not None else ep
  n_pad = tb - t_local
  l2p = prepared.orig_to_slot
  has_l2p = l2p is not None
  has_rep_lut = has_l2p and getattr(l2p, "ndim", 1) >= 2
  # When canonical_routing is True or None (default), `monolithic.dedup_ep_moe`
  # scores `gating` in original expert order and remaps the selected top-k
  # logical expert ids to physical slots via `logical_to_physical_map`, making
  # any expert placement or replication bit-for-bit identical to unplaced
  # weights without extra full-logit gathers or sigmoid conversions.
  use_in_kernel_l2p = has_l2p and (
      has_rep_lut or cfg.canonical_routing is not False
  )
  remap_logits = (
      prepared.slot_to_orig is not None and not use_in_kernel_l2p
  )
  # ------------------------------------------------------------------
  # Step 1: Optional legacy logit remap and bucket pad.
  # ------------------------------------------------------------------
  if n_pad or remap_logits:
    has_ids = topk_ids is not None
    need_s2o_for_pad = (
        n_pad > 0
        and use_in_kernel_l2p
        and not has_rep_lut
        and prepared.slot_to_orig is not None
    )

    def prep(x_l, g_l, *rest):
      ids_l = rest[0] if has_ids else None
      maps = rest[1 if has_ids else 0 :]
      if remap_logits:
        s2o, o2s = maps[0], maps[1]
        g_l = jnp.take(g_l, s2o, axis=1)
        if ids_l is not None:
          ids_l = jnp.take(o2s, ids_l.astype(jnp.int32), axis=0)
      if n_pad:
        me = jax.lax.axis_index(AXIS)
        pad_slot_ids = _pad_expert_ids(
            me, n_pad, g_local=g_local, board=board, topk=k
        )
        if need_s2o_for_pad:
          pad_ids = jnp.take(maps[0], pad_slot_ids, axis=0)
        elif has_rep_lut:
          pad_ids = jnp.broadcast_to(
              jnp.arange(k, dtype=jnp.int32)[None, :], (n_pad, k)
          )
        else:
          pad_ids = pad_slot_ids
        x_l = jnp.concatenate(
            [x_l, jnp.zeros((n_pad, x_l.shape[1]), x_l.dtype)], axis=0
        )
        g_pad = jnp.full((n_pad, g_l.shape[1]), _PAD_LOGIT, g_l.dtype)
        g_pad = g_pad.at[jnp.arange(n_pad)[:, None], pad_ids].set(0.0)
        g_l = jnp.concatenate([g_l, g_pad], axis=0)
        if ids_l is not None:
          ids_l = jnp.concatenate([ids_l.astype(jnp.int32), pad_ids], axis=0)
      return (x_l, g_l) + ((ids_l,) if has_ids else ())

    args = (x, gating) + ((topk_ids,) if has_ids else ())
    specs = (P(AXIS),) * len(args)
    if remap_logits:
      args += (prepared.slot_to_orig, prepared.orig_to_slot)
      specs += (P(), P())
    elif need_s2o_for_pad:
      args += (prepared.slot_to_orig,)
      specs += (P(),)
    out = jax.shard_map(
        prep,
        mesh=mesh,
        in_specs=specs,
        out_specs=(P(AXIS),) * (3 if has_ids else 2),
        check_vma=False,
    )(*args)
    x, gating = out[0], out[1]
    topk_ids = out[2] if has_ids else None

  # ------------------------------------------------------------------
  # Step 2: Run the monolithic layer with the plan's arguments.
  # ------------------------------------------------------------------
  kwargs = dict(p.kwargs)
  if use_in_kernel_l2p:
    kwargs["logical_to_physical_map"] = l2p
  sc_kwargs = dict(kwargs.pop("sc_kwargs", None) or {})
  layer_fn = monolithic.dedup_ep_moe
  if p.impl == "product_scdma_cfg":
    # Lazy: the SparseCore sender needs tpu_sc only when it is used.
    from tokamax._src.ops.experimental.dedup_moe.dispatch import sc_sender  # pylint: disable=g-import-not-at-top

    if cfg.sc_return:
      sc_kwargs["sc_forward"] = True
      sc_kwargs["sc_return"] = True
    if cfg.sc_combine:
      sc_kwargs["sc_forward"] = True
      sc_kwargs["sc_combine"] = True
    layer_fn = sc_sender.wrap_layer(layer_fn, **sc_kwargs)
  elif p.impl != "product":
    raise ValueError(f"unknown impl {p.impl!r}")
  if topk_ids is not None:
    kwargs["hash_topk_ids"] = topk_ids
  if prepared.ws1 is not None:
    kwargs.update(ws1=prepared.ws1, ws2=prepared.ws2)
  if kernel_options is not None:
    kwargs["kernel_options"] = kernel_options
  if combine_options is not None:
    kwargs["combine_options"] = combine_options
  # The tuned auto-policy De-Dup schedule only with cfg.tuned_auto, so the
  # default auto plan keeps its program.
  auto_dedup = (
      cfg.tuned_auto
      and cfg.policy == "auto"
      and kwargs.get("dedup_topology") is not None
  )
  ramp = -1.0 if auto_dedup and cfg.ramp_ratio == 2.0 else cfg.ramp_ratio
  hoist = "w01" if auto_dedup and cfg.stage_hoist == "none" else cfg.stage_hoist
  lookahead = 1 if auto_dedup and not cfg.fwd_lookahead else cfg.fwd_lookahead
  if cfg.direct_order != "ring" or cfg.wave_order != "rows" or ramp != 2.0:
    kwargs.update(
        direct_order=cfg.direct_order,
        wave_order=cfg.wave_order,
        ramp_ratio=ramp,
        topology_board_size=cfg.scheduling_board_size,
    )
  if hoist != "none" or lookahead:
    kwargs.update(stage_hoist=hoist, dedup_fwd_lookahead=lookahead)
  if cfg.trace_scopes:
    kwargs.update(dedup_trace_scopes=True)
  if cfg.return_order != "ring":
    kwargs.update(
        return_order=cfg.return_order,
        topology_board_size=cfg.scheduling_board_size,
    )
  if cfg.combine_dtype is not None:
    kwargs["combine_dtype"] = cfg.combine_dtype
  if cfg.push_from_vmem is not None:
    kwargs["push_from_vmem"] = cfg.push_from_vmem
  use_ddc = cfg.dedup_combine or (
      auto_dedup and cfg.combine_dtype == "bf16" and tb >= 512
  )
  if use_ddc:
    kwargs.update(
        dedup_combine=True,
        topology_board_size=cfg.scheduling_board_size,
    )
  if auto_dedup and g_local >= 16 and tb >= 512:
    kwargs["dedup_min_fanout"] = 0
  y = layer_fn(
      x,
      prepared.w1,
      prepared.w2,
      prepared.w1_scale,
      prepared.w2_scale,
      gating,
      topk=k,
      renormalize=True,
      mesh=mesh,
      capacity=cfg.capacity,
      weight_format=arg.WeightFormat(cfg.weight_format),
      **kwargs,
  )

  # ------------------------------------------------------------------
  # Step 3: Slice off bucket padding rows on each shard.
  # ------------------------------------------------------------------
  if n_pad:
    y = jax.shard_map(
        lambda y_l: y_l[:t_local],
        mesh=mesh,
        in_specs=(P(AXIS),),
        out_specs=P(AXIS),
        check_vma=False,
    )(y)
  return y


def make_layer(cfg: DedupMoeConfig, mesh: Any, shared_weights=(None, None)):
  """Returns `layer(x, w1, w2, w1s, w2s, gating, hash_ids)` running dedup_moe.

  The weights and hash_ids are already in slot order (no placement remap).

  Args:
    cfg: The layer config.
    mesh: One-axis EP mesh.
    shared_weights: (ws1, ws2) replicated shared expert, or (None, None).

  Returns:
    The layer function.
  """

  def layer(x, w1, w2, w1s, w2s, gating, hash_ids):
    prepared = PreparedMoe(
        w1, w2, w1s, w2s, shared_weights[0], shared_weights[1], None, None, mesh
    )
    return dedup_moe(x, gating, prepared, cfg, topk_ids=hash_ids)

  return layer
