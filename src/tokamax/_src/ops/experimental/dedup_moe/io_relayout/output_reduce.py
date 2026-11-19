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

"""Arrival combine: weighted top-k reduction of the arrived expert rows.

`arrivals[pos]` in XLA is an offloaded gather whose output is written to HBM
at every size, so the gathered rows would make a full round trip through HBM
before the weighted sum reads them, followed by a relayout copy because
[t, lane blocks, 128] and [t, hidden] do not share a tiling.
Here each arrival row lands in VMEM as one DMA and is reduced where it lands,
so the combine moves the rows in and the tokens out and nothing else.

The row DMAs are emitted BESIDE the reduction rather than in a loop of their
own: they are scalar-unit work against a body that runs the vector unit, so
in one basic block the scheduler packs them into slots the reduction leaves
empty.

What the schedule costs is the DMA descriptor issue, not the arithmetic, so
the two things worth doing to it are shortening the scalar sequence a row
needs and keeping both DMA queues fed.

This is a variant of `fused_moe.kernel`'s combine kernel that also covers
small (decode) token counts with a single-parity schedule.
"""

import dataclasses
import threading
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel

HIDDEN_LANE_BLOCK = fused_host.HIDDEN_LANE_BLOCK
U32_SUBLANE_TILE = fused_host.U32_SUBLANE_TILE
align_up = fused_host.align_up
pow2_shift = fused_host.pow2_shift

# Supported modes for folding the arrival_scales gather into the combine kernel.
FUSE_SCALE_GATHER_MODES = ("off", "vmem", "smem")
# Leave headroom below XLA's ~614 KiB callee SMEM budget when staging
# `arrival_scales` in SMEM.
COMBINE_SMEM_BUDGET_BYTES = 580 * 1024


def _normalize_fuse_scale_gather(mode: Any) -> str:
  """Normalizes `fuse_scale_gather` knob values to 'off' | 'vmem' | 'smem'."""
  if mode in (False, None, "off"):
    return "off"
  if mode is True or mode == "vmem":
    return "vmem"
  if mode == "smem":
    return "smem"
  raise ValueError(
      f"fuse_scale_gather must be a bool or one of {FUSE_SCALE_GATHER_MODES};"
      f" got {mode!r}"
  )


def _smem_padded_bytes(shape: tuple[int, ...]) -> int:
  if len(shape) <= 1:
    n = shape[0] if shape else 1
    return 4 * align_up(n, HIDDEN_LANE_BLOCK)
  lead = 1
  for d in shape[:-2]:
    lead *= d
  return (
      4
      * lead
      * align_up(shape[-2], U32_SUBLANE_TILE)
      * align_up(shape[-1], HIDDEN_LANE_BLOCK)
  )


def _maybe_annotate_no_hazard(ref: Any, *, enabled: bool = True) -> Any:
  """Annotates a VMEM ref with `no_hazard_no_deps=True` when lowering to TPU."""
  ctx = getattr(jax.config, "pallas_tpu_interpret_mode_context_manager", None)
  is_interp = (ctx is not None and ctx.value is not None) or hasattr(
      pl.pallas_call, "__wrapped__"
  )
  if not enabled or is_interp:
    return ref
  return pltpu.annotate(ref, no_hazard_no_deps=True)

# The pipelined schedule's constants, shared with `fused_moe.kernel`: tokens
# per staged step, tokens per output relayout group, tokens per reduction
# chunk, the smallest token count the pipelined schedule applies to, the
# buffer rotation depth, and the steps of row lead.
COMBINE_TOKENS = fused_kernel.COMBINE_TOKENS
COMBINE_GROUP = fused_kernel.COMBINE_GROUP
COMBINE_CHUNK = fused_kernel.COMBINE_CHUNK
COMBINE_MIN_TOKENS = fused_kernel.COMBINE_MIN_TOKENS
COMBINE_PARITIES = fused_kernel.COMBINE_PARITIES
COMBINE_LEAD = fused_kernel.COMBINE_LEAD


def _combine_is_pipelined(t_local, tb):
  """Whether `t_local` and `tb` satisfy the 4-parity pipelined combine schedule."""
  if t_local < COMBINE_MIN_TOKENS or tb < COMBINE_GROUP:
    return False
  steps = t_local // tb
  return (
      not t_local % tb
      and not tb % COMBINE_GROUP
      and steps > COMBINE_LEAD
      and steps >= COMBINE_PARITIES
      and not steps % COMBINE_PARITIES
  )


def combine_step_tokens(t_local):
  """Returns the tokens a combine step stages.

  Tokens a combine step stages, or 0 where this path does not apply.

  Above COMBINE_MIN_TOKENS, returns a step width `tb` that satisfies the
  4-parity pipelined schedule (`_combine_is_pipelined`). For smaller batch
  sizes (`8 <= t_local < COMBINE_MIN_TOKENS`, `t_local % 8 == 0`), returns a
  single-parity step width `tb <= COMBINE_TOKENS` dividing `t_local` so the
  Pallas combine kernel runs at decode batch sizes too (no XLA gather
  fallback at any t bucket).

  Args:
    t_local: Tokens per shard.

  Returns:
    The step width `tb`, or 0 where the Pallas combine does not apply.
  """
  if t_local < U32_SUBLANE_TILE or t_local % U32_SUBLANE_TILE:
    return 0
  tb = COMBINE_TOKENS
  while tb >= COMBINE_GROUP:
    if _combine_is_pipelined(t_local, tb):
      return tb
    tb //= 2
  tb = min(COMBINE_TOKENS, t_local)
  while tb >= U32_SUBLANE_TILE:
    if not t_local % tb:
      return tb
    tb //= 2
  return 0


@dataclasses.dataclass(frozen=True)
class Config:
  """Static configuration for the arrival combine kernel."""

  t_local: int
  topk: int
  hidden: int
  lane_blocks: int
  wire_dtype: jax.typing.DTypeLike
  out_dtype: jax.typing.DTypeLike
  tb: int
  pipelined: bool
  steps: int
  gt: int
  groups: int
  chunk_w: int
  parities: int
  lanes: int = HIDDEN_LANE_BLOCK
  evenly_distribute_critical_nodes: bool = False
  annotate_no_hazard: bool = False
  fuse_scale_gather: str = "off"
  decode_double_buffer: bool = False
  mirror_sublanes: int = 0
  mirror_tile_m: int = 0
  recv_rows: int = 0

  @classmethod
  def create(
      cls,
      *,
      t_local: int,
      topk: int,
      hidden: int,
      lane_blocks: int,
      wire_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
      evenly_distribute_critical_nodes: bool = False,
      annotate_no_hazard: bool = False,
      fuse_scale_gather: Any = False,
      decode_double_buffer: bool = False,
      mirror_sublanes: int = 0,
      mirror_tile_m: int = 0,
      recv_rows: int = 0,
  ):
    """Derives the combine kernel's step, group and chunk sizes.

    Args:
      t_local: Tokens per shard.
      topk: Experts per token.
      hidden: Model width.
      lane_blocks: 128-lane blocks per row.
      wire_dtype: Element type of the arrived rows.
      out_dtype: Element type of the combined output.
      evenly_distribute_critical_nodes: Compiler scheduling hint.
      annotate_no_hazard: Annotate VMEM staging refs hazard-free.
      fuse_scale_gather: 'off'/False, 'vmem'/True, or 'smem' (auto-falls back
        to 'vmem' if `arrival_scales` exceeds `COMBINE_SMEM_BUDGET_BYTES`).
      decode_double_buffer: Double-buffer the non-pipelined decode loop when
        `steps >= 2`.
      mirror_sublanes: Height of `arrival_scales` when fusing scale gather.
      mirror_tile_m: Width of `arrival_scales` when fusing scale gather.
      recv_rows: Height of `arrivals` for in-kernel clamping when fusing scale
        gather.

    Returns:
      The Config.
    """
    tb = combine_step_tokens(t_local)
    pipelined = _combine_is_pipelined(t_local, tb)
    steps = t_local // tb
    if (
        bool(decode_double_buffer)
        and (not pipelined)
        and steps == 1
        and tb >= 2 * COMBINE_GROUP
    ):
      # Split single-step decode buckets (e.g. t=32 -> 2x16, t=48 -> 3x16,
      # t=64 -> 2x32) so input row DMAs and output stores overlap reduction.
      start = ((tb // 2) // COMBINE_GROUP) * COMBINE_GROUP
      for cand in range(start, COMBINE_GROUP - 1, -COMBINE_GROUP):
        if t_local % cand == 0:
          tb = cand
          steps = t_local // tb
          break
    gt = COMBINE_GROUP if pipelined else min(COMBINE_GROUP, tb)
    groups = tb // gt
    chunk_w = COMBINE_CHUNK if pipelined else min(COMBINE_CHUNK, tb)
    ddb = bool(decode_double_buffer) and (not pipelined) and steps >= 2
    parities = COMBINE_PARITIES if pipelined else (2 if ddb else 1)
    fsg = _normalize_fuse_scale_gather(fuse_scale_gather)
    if mirror_sublanes <= 0 or mirror_tile_m <= 0:
      fsg = "off"
    elif fsg == "smem":
      scales_rows = align_up(mirror_sublanes, U32_SUBLANE_TILE)
      smem_bytes = _smem_padded_bytes(
          (scales_rows, mirror_tile_m)
      ) + 3 * _smem_padded_bytes((t_local * topk,))
      if smem_bytes > COMBINE_SMEM_BUDGET_BYTES:
        fsg = "vmem"
    return cls(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=lane_blocks,
        wire_dtype=wire_dtype,
        out_dtype=out_dtype,
        tb=tb,
        pipelined=pipelined,
        steps=steps,
        gt=gt,
        groups=groups,
        chunk_w=chunk_w,
        parities=parities,
        evenly_distribute_critical_nodes=bool(
            evenly_distribute_critical_nodes
        ),
        annotate_no_hazard=bool(annotate_no_hazard),
        fuse_scale_gather=fsg,
        decode_double_buffer=ddb,
        mirror_sublanes=int(mirror_sublanes),
        mirror_tile_m=int(mirror_tile_m),
        recv_rows=int(recv_rows),
    )

  @property
  def kernel_name(self) -> str:
    tags = ""
    if self.fuse_scale_gather == "vmem":
      tags += "_fsgv"
    elif self.fuse_scale_gather == "smem":
      tags += "_fsgs"
    if self.decode_double_buffer:
      tags += "_ddb"
    if self.evenly_distribute_critical_nodes:
      tags += "_edcn"
    if self.annotate_no_hazard:
      tags += "_nh"
    return f"dedup_moe_combine_t{self.t_local}_k{self.topk}_b{self.tb}{tags}"


def _build_combine_kernel(
    *,
    t_local,
    topk,
    hidden,
    lane_blocks,
    wire_dtype,
    out_dtype,
    evenly_distribute_critical_nodes=False,
    annotate_no_hazard=False,
    fuse_scale_gather=False,
    decode_double_buffer=False,
    mirror_sublanes=0,
    mirror_tile_m=0,
    recv_rows=0,
    has_order_after=False,
):
  """out[i] = sum_k coef[i, k] * arrivals[pos[i * topk + k]].

  `pos` is a scalar-prefetch table of arrival rows, already clamped into the
  buffer (or clamped in-kernel when `fuse_scale_gather != 'off'`); `coef` is
  the arrival row's scale folded with the router weight and rides SMEM beside
  it (or is gathered in-kernel from `arrival_scales` via a single async HBM DMA
  overlapped with step 0's arrival row DMAs). The result is [t_local, hidden]
  in the layer's own output layout.

  An arrival row is [lane blocks, 128] -- one register -- while the output
  is [tokens, hidden], which is that layout transposed: a token's hidden
  axis lies along sublanes in the row and along lanes in the output. The sum
  therefore runs in the ROW's layout and only its bf16 result is relaid out,
  so the sublane shuffle sees one bf16 token per group instead of topk
  packed-f8 rows, and the live accumulator is one token wide.

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.
    hidden: Model width.
    lane_blocks: 128-lane blocks per row.
    wire_dtype: Element type of the arrived rows.
    out_dtype: Element type of the output.
    evenly_distribute_critical_nodes: Compiler scheduling hint.
    annotate_no_hazard: Annotate VMEM staging refs hazard-free.
    fuse_scale_gather: 'off'/False, 'vmem'/True, or 'smem'.
    decode_double_buffer: Double-buffer non-pipelined decode loop.
    mirror_sublanes: Height of `arrival_scales` when fusing scale gather.
    mirror_tile_m: Width of `arrival_scales` when fusing scale gather.
    recv_rows: Height of `arrivals` for in-kernel clamping.
    has_order_after: Whether an HBM ordering dependency input is passed.

  Returns:
    `fn(arrivals, coef, pos, order_after=None)` (when
    `fuse_scale_gather == 'off'`) or `fn(arrivals, topk_weights, pos,
    arrival_scales, mirror_pos, order_after=None)` returning [t_local, hidden].
  """
  cfg = Config.create(
      t_local=t_local,
      topk=topk,
      hidden=hidden,
      lane_blocks=lane_blocks,
      wire_dtype=wire_dtype,
      out_dtype=out_dtype,
      evenly_distribute_critical_nodes=evenly_distribute_critical_nodes,
      annotate_no_hazard=annotate_no_hazard,
      fuse_scale_gather=fuse_scale_gather,
      decode_double_buffer=decode_double_buffer,
      mirror_sublanes=mirror_sublanes,
      mirror_tile_m=mirror_tile_m,
      recv_rows=recv_rows,
  )
  tb = cfg.tb
  pipelined = cfg.pipelined
  steps = cfg.steps
  gt = cfg.gt
  groups = cfg.groups
  chunk_w = cfg.chunk_w
  parities = cfg.parities
  lanes = cfg.lanes
  fsg = cfg.fuse_scale_gather
  fused_scales = fsg != "off"
  if fused_scales:
    tile_shift = pow2_shift(cfg.mirror_tile_m, "the mirror's lane count")
    mirror_len = cfg.mirror_sublanes * cfg.mirror_tile_m
    scales_buf_rows = align_up(cfg.mirror_sublanes, U32_SUBLANE_TILE)
  else:
    tile_shift = 0
    mirror_len = 0
    scales_buf_rows = 0

  def kernel(*args):
    args_it = iter(args)
    pos_sm = next(args_it)
    if fused_scales:
      mpos_sm = next(args_it)
      w_sm = next(args_it)
      coef_sm = None
      arr_hbm = next(args_it)
      scales_hbm = next(args_it)
    else:
      mpos_sm = None
      w_sm = None
      coef_sm = next(args_it)
      arr_hbm = next(args_it)
      scales_hbm = None
    if has_order_after:
      next(args_it)  # Ordering dependency only; never read.
    out_hbm = next(args_it)
    rows_vm = _maybe_annotate_no_hazard(
        next(args_it), enabled=cfg.annotate_no_hazard
    )
    tile_vm = next(args_it)
    out_vm = _maybe_annotate_no_hazard(
        next(args_it), enabled=cfg.annotate_no_hazard
    )
    row_sems = next(args_it)
    out_sems = next(args_it)
    if fused_scales:
      scales_buf = next(args_it)
      scl_sem = next(args_it)
      scl_copy = pltpu.make_async_copy(
          scales_hbm,
          scales_buf.at[pl.ds(0, cfg.mirror_sublanes)],
          scl_sem,
      )
    else:
      scales_buf = None
      scl_copy = None

    def start_row(base, parity, i, k):
      # Alternate queues by slot: this kernel is one 4 KiB descriptor per
      # arrival row and an in-order queue retires descriptors at a fixed
      # rate, which for a row this small is slower than the row's own
      # bytes. Both queues signal the one semaphore the step waits on,
      # which counts bytes and not descriptors.
      row_idx = pos_sm[(base + i) * topk + k]
      if fused_scales and cfg.recv_rows > 0:
        row_idx = jnp.minimum(row_idx, jnp.int32(cfg.recv_rows - 1))
      pltpu.make_async_copy(
          arr_hbm.at[row_idx],
          rows_vm.at[parity, k * tb + i],
          row_sems.at[parity],
      ).start(priority=k % 2)

    use_vmem_tile_mul = (
        fsg == "vmem"
        and cfg.mirror_tile_m == lanes
        and lane_blocks % U32_SUBLANE_TILE == 0
    )

    def load_coef_tile(idx):
      assert mpos_sm is not None and w_sm is not None and scales_buf is not None
      scl = jnp.minimum(mpos_sm[idx], jnp.int32(mirror_len - 1))
      sub = jnp.right_shift(scl, tile_shift)
      lane = jnp.bitwise_and(scl, cfg.mirror_tile_m - 1)
      sub_a = pl.multiple_of(jnp.bitwise_and(sub, -U32_SUBLANE_TILE), U32_SUBLANE_TILE)
      sub_off = sub - sub_a
      tile = scales_buf[pl.ds(sub_a, U32_SUBLANE_TILE), :]
      s0 = jnp.take_along_axis(
          tile,
          jnp.broadcast_to(sub_off, (U32_SUBLANE_TILE, cfg.mirror_tile_m)),
          axis=0,
      )
      s1 = jnp.take_along_axis(
          s0,
          jnp.broadcast_to(lane, (U32_SUBLANE_TILE, cfg.mirror_tile_m)),
          axis=1,
      )
      return s1 * w_sm[idx]

    def reduce_token(base, parity, i):
      """One token's weighted sum, in the arrival rows' own layout."""
      acc: Any = None
      for k in range(topk):
        idx = (base + i) * topk + k
        if use_vmem_tile_mul:
          coef_tile = load_coef_tile(idx)
          coef = jnp.concatenate(
              [coef_tile] * (lane_blocks // U32_SUBLANE_TILE), axis=0
          )
        elif fsg == "smem":
          assert (
              mpos_sm is not None
              and w_sm is not None
              and scales_buf is not None
          )
          scl = jnp.minimum(mpos_sm[idx], jnp.int32(mirror_len - 1))
          sub = jnp.right_shift(scl, tile_shift)
          lane = jnp.bitwise_and(scl, cfg.mirror_tile_m - 1)
          coef = scales_buf[sub, lane] * w_sm[idx]
        elif fsg == "vmem":
          coef = load_coef_tile(idx)[0, 0]
        else:
          assert coef_sm is not None
          coef = coef_sm[idx]
        term = rows_vm[parity, k * tb + i].astype(jnp.float32) * coef
        acc = term if acc is None else acc + term
      tile_vm[i] = acc.astype(out_dtype)

    if scl_copy is not None:
      scl_copy.start()

    if not pipelined:
      if cfg.decode_double_buffer:
        # Prologue: issue step 0's arrival row DMAs into parity 0.
        for i in range(tb):
          for k in range(topk):
            start_row(0, 0, i, k)
        for step in range(steps):
          base = step * tb
          parity = step & 1
          npar = 1 - parity
          pltpu.make_async_copy(
              rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
          ).wait()
          if step == 0 and scl_copy is not None:
            scl_copy.wait()
          if step >= parities:
            pltpu.make_async_copy(
                out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
            ).wait()
          if step + 1 < steps:
            nxt_base = (step + 1) * tb

            def one_chunk_ddb(
                chunk, carry, base=base, nxt_base=nxt_base, parity=parity, npar=npar
            ):
              for u in range(chunk_w):
                i = chunk * chunk_w + u
                for k in range(topk):
                  start_row(nxt_base, npar, i, k)
                reduce_token(base, parity, i)
              return carry

            lax.fori_loop(0, tb // chunk_w, one_chunk_ddb, jnp.int32(0))
          else:
            def one_chunk_last(chunk, carry, base=base, parity=parity):
              for u in range(chunk_w):
                reduce_token(base, parity, chunk * chunk_w + u)
              return carry

            lax.fori_loop(0, tb // chunk_w, one_chunk_last, jnp.int32(0))
          for g in range(groups):
            out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[
                pl.ds(g * gt, gt)
            ].reshape(gt, hidden)
          pltpu.make_async_copy(
              out_vm.at[parity],
              out_hbm.at[pl.ds(step * tb, tb)],
              out_sems.at[parity],
          ).start()
        for p in range(parities):
          pltpu.make_async_copy(
              out_vm.at[p], out_vm.at[p], out_sems.at[p]
          ).wait()
        return

      for step in range(steps):
        base = step * tb
        # Step 1: Issue arrival row DMAs for the current step.
        for i in range(tb):
          for k in range(topk):
            start_row(base, 0, i, k)
        # Step 2: Wait for arrival row DMAs (and step 0's scale mirror DMA).
        pltpu.make_async_copy(
            rows_vm.at[0], rows_vm.at[0], row_sems.at[0]
        ).wait()
        if step == 0 and scl_copy is not None:
          scl_copy.wait()

        # Step 3: Reduce top-k arrival rows for each token in chunks.
        def one_chunk_small(chunk, carry, base=base):
          for u in range(chunk_w):
            reduce_token(base, 0, chunk * chunk_w + u)
          return carry

        lax.fori_loop(0, tb // chunk_w, one_chunk_small, jnp.int32(0))
        # Step 4: Transpose reduced token groups into output VMEM layout.
        for g in range(groups):
          out_vm[0, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
              gt, hidden
          )
        # Step 5: Store output step from VMEM to HBM and wait.
        pltpu.make_async_copy(
            out_vm.at[0],
            out_hbm.at[pl.ds(step * tb, tb)],
            out_sems.at[0],
        ).start()
        pltpu.make_async_copy(out_vm.at[0], out_vm.at[0], out_sems.at[0]).wait()
      return

    def one_step(step, parity):
      # Step 1: Compute next prefetch batch offset and parity.
      # The step's rows were started a step ago; the next step's are
      # started beside this one's reduction, so their descriptor issue
      # rides in slots the vector work leaves empty. The last step
      # re-starts its own rows rather than branching -- the copy is never
      # read and is drained below.
      nxt = jnp.minimum(step + COMBINE_LEAD, steps - 1) * tb
      base = step * tb
      npar = (parity + COMBINE_LEAD) % COMBINE_PARITIES
      # Step 2: Wait for the input DMA of this parity and the output DMA of
      # the reused parity.
      pltpu.make_async_copy(
          rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
      ).wait()
      if scl_copy is not None:
        @pl.when(step == 0)
        def _():
          scl_copy.wait()

      @pl.when(step >= COMBINE_PARITIES)
      def _():
        pltpu.make_async_copy(
            out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
        ).wait()

      # Step 3: Reduce top-k arrival rows while issuing prefetch DMAs for `nxt`.
      def one_chunk(chunk, carry):
        for u in range(COMBINE_CHUNK):
          i = chunk * COMBINE_CHUNK + u
          for k in range(topk):
            start_row(nxt, npar, i, k)
          reduce_token(base, parity, i)
        return carry

      lax.fori_loop(0, tb // COMBINE_CHUNK, one_chunk, jnp.int32(0))
      # Step 4: Relayout reduced token groups into `[gt, hidden]` output VMEM.
      for g in range(groups):
        out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
            gt, hidden
        )
      # Step 5: Start async output DMA from VMEM to HBM.
      pltpu.make_async_copy(
          out_vm.at[parity],
          out_hbm.at[pl.ds(step * tb, tb)],
          out_sems.at[parity],
      ).start()

    def step_pair(pair, carry):
      for parity in range(COMBINE_PARITIES):
        one_step(pair * COMBINE_PARITIES + parity, parity)
      return carry

    # Prologue: Prefetch first COMBINE_LEAD steps of arrival rows.
    for p in range(COMBINE_LEAD):
      for i in range(tb):
        for k in range(topk):
          start_row(p * tb, p, i, k)
    # Body: Main pipelined reduction loop over 4-parity groups.
    lax.fori_loop(0, steps // COMBINE_PARITIES, step_pair, jnp.int32(0))
    # Epilogue: The last COMBINE_LEAD steps re-start the final step's rows into
    # the buffers the loop no longer reads; drain them, and the stores.
    for p in range(COMBINE_LEAD):
      pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
    for p in range(COMBINE_PARITIES):
      pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

  hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
  scratch_shapes: list[Any] = [
      pltpu.VMEM((parities, topk * tb, lane_blocks, lanes), wire_dtype),
      pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
      pltpu.VMEM((parities, tb, hidden), out_dtype),
      pltpu.SemaphoreType.DMA((parities,)),
      pltpu.SemaphoreType.DMA((parities,)),
  ]
  if fused_scales:
    if fsg == "smem":
      scratch_shapes.append(
          pltpu.SMEM((scales_buf_rows, cfg.mirror_tile_m), jnp.float32)
      )
    else:
      scratch_shapes.append(
          pltpu.VMEM((scales_buf_rows, cfg.mirror_tile_m), jnp.float32)
      )
    scratch_shapes.append(pltpu.SemaphoreType.DMA(()))
  call = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3 if fused_scales else 2,
          in_specs=([hbm, hbm] if fused_scales else [hbm])
          + ([hbm] if has_order_after else []),
          out_specs=hbm,
          scratch_shapes=tuple(scratch_shapes),
          grid=(),
      ),
      out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=fused_host.vmem_limit(),
          disable_bounds_checks=True,
          flags=(
              {"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True}
              if cfg.evenly_distribute_critical_nodes
              else None
          ),
      ),
      name=cfg.kernel_name,
  )

  if fused_scales:
    def fn_fused(
        arrivals, topk_weights, pos, arrival_scales, mirror_pos,
        order_after=None,
    ):
      if (
          pos.shape != (t_local, topk)
          or mirror_pos.shape != (t_local, topk)
          or topk_weights.shape != (t_local, topk)
      ):
        raise ValueError(
            "the combine takes token-major "
            f"[{t_local}, {topk}] tables, got pos {pos.shape}, "
            f"mirror_pos {mirror_pos.shape}, weights {topk_weights.shape}"
        )
      extra = (order_after,) if has_order_after else ()
      return call(
          pos.astype(jnp.int32).reshape(-1),
          mirror_pos.astype(jnp.int32).reshape(-1),
          topk_weights.astype(jnp.float32).reshape(-1),
          arrivals,
          arrival_scales,
          *extra,
      )

    return fn_fused

  def fn(arrivals, coef, pos, order_after=None):
    # Both tables flatten TOKEN-major, so a token's topk entries are one
    # contiguous SMEM run and its 2 * topk scalar reads share a single
    # address computation. That is also the order the plan builds them in,
    # so neither flatten is a transpose.
    if pos.shape != (t_local, topk) or coef.shape != (t_local, topk):
      raise ValueError(
          "the combine takes token-major "
          f"[{t_local}, {topk}] tables, got pos "
          f"{pos.shape} and coef {coef.shape}"
      )
    extra = (order_after,) if has_order_after else ()
    return call(
        pos.astype(jnp.int32).reshape(-1), coef.reshape(-1), arrivals, *extra
    )

  return fn


_COMBINE_CACHE = {}
_COMBINE_CACHE_LOCK = threading.Lock()


def build_combine_kernel(**kwargs):
  """Memoizing front door (as fused_kernel.kernel.build_dedup_moe_kernel)."""
  key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
  fn = _COMBINE_CACHE.get(key)
  if fn is None:
    with _COMBINE_CACHE_LOCK:
      fn = _COMBINE_CACHE.get(key)
      if fn is None:
        fn = _build_combine_kernel(**kwargs)
        _COMBINE_CACHE[key] = fn
  return fn


def combine_arrivals(
    arrivals,
    arrival_scales,
    pos,
    mirror_pos,
    topk_weights,
    out_dtype,
    *,
    order_after: Any = None,
):
  """Combine arrivals [rows, lane blocks, 128] into [t_local, hidden].

  This is the layer's combine: every routed row the fused kernel pushed to
  this shard is weighted by its router weight and summed into its token.
  Each of a token's topk selections lands as its own arrival row (the
  combine is not de-duplicated), so `pos` [t_local, topk] names one
  arrival row per selection slot.

  Two tables index two different objects, so each owns its own clamp.
  `pos` is the arrival ROW: a row of `arrivals`, and on the kernel path
  below it is a DMA address. `mirror_pos` is a position in the wire's scale
  mirror, which is [sublanes, tile_m] because a transport can only address
  the mirror a sublane at a time and a sublane holds tile_m f32 whatever is
  written in it -- so a whole source TILE's scales are one sublane and the
  mirror position is NOT the arrival row. An unquantized (bf16) wire
  carries no scales: the kernel moves no mirror then, and neither table is
  read for it.

  Wherever t_local is a multiple of 8 (every t bucket) the row gather and
  the weighted sum are one Mosaic call (combine_step_tokens): XLA
  answers `arrivals[rows]` with an offloaded gather whose output is in HBM
  at any size, so the rows would be written out and read straight back
  before the sum sees them. Any other t_local takes the fused XLA epilogue
  below.

  Both paths fold the arrival row's scale and the router weight into one
  coefficient over K * t_local values, so each of the K * t_local * hidden
  elements is multiplied once rather than twice.

  Args:
    arrivals: [rows, lane blocks, 128] arrived rows.
    arrival_scales: [sublanes, tile_m] f32 scale mirror, or None.
    pos: [t_local, topk] arrival row of each selection.
    mirror_pos: [t_local, topk] scale-mirror position of each selection.
    topk_weights: [t_local, topk] router weights.
    out_dtype: Output element type.
    order_after: Optional HBM array to order the combine pallas_call after.

  Returns:
    [t_local, hidden] combined tokens.
  """
  t_local, topk = pos.shape
  recv_rows, lane_blocks, lanes = arrivals.shape
  hidden = lane_blocks * lanes
  has_scales = arrival_scales is not None and arrivals.dtype != jnp.bfloat16
  # Passed only when set, so the default build key is unchanged.
  order_kw: dict[str, Any] = (
      {} if order_after is None else {"has_order_after": True}
  )
  order_arg: dict[str, Any] = (
      {} if order_after is None else {"order_after": order_after}
  )
  # Respect `fuse_scale_gather` configured via `combine_options` (which wraps
  # `build_combine_kernel` in a `functools.partial` in `monolithic.py`).
  partial_kw = getattr(build_combine_kernel, "keywords", None) or {}
  fuse_mode = _normalize_fuse_scale_gather(
      partial_kw.get("fuse_scale_gather", False)
  )
  if combine_step_tokens(t_local) and has_scales and fuse_mode != "off":
    assert arrival_scales is not None
    mirror_sublanes, tile_m = arrival_scales.shape
    pow2_shift(tile_m, "the mirror's lane count")
    combine = build_combine_kernel(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals.dtype,
        out_dtype=out_dtype,
        mirror_sublanes=mirror_sublanes,
        mirror_tile_m=tile_m,
        recv_rows=recv_rows,
        **order_kw,
    )
    return combine(
        arrivals, topk_weights, pos, arrival_scales, mirror_pos, **order_arg
    )
  # A gather clamps an out-of-range index to the last one; a DMA does not,
  # and a sublane index clamped on its own would pair the last sublane with
  # the wrong lane. Neither position is ever negative.
  rows = jnp.minimum(pos, recv_rows - 1)
  if not has_scales:
    # Unquantized wire rows: there is no row scale to apply.
    coef = topk_weights.astype(jnp.float32)
  else:
    assert arrival_scales is not None
    mirror_sublanes, tile_m = arrival_scales.shape
    pow2_shift(tile_m, "the mirror's lane count")
    scl = jnp.minimum(mirror_pos, mirror_sublanes * tile_m - 1)
    row_scales = jnp.take(arrival_scales.reshape(-1), scl, axis=0)
    coef = row_scales * topk_weights.astype(jnp.float32)
  if combine_step_tokens(t_local):
    combine = build_combine_kernel(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=lane_blocks,
        wire_dtype=arrivals.dtype,
        out_dtype=out_dtype,
        **order_kw,
    )
    return combine(arrivals, coef, rows, **order_arg)
  rows_fp8 = arrivals[pos.T.reshape(-1)]  # f8 [K*t, lane blocks, 128]
  coef_k = coef.T.reshape(-1)  # k-major [K*t_local]
  terms = []
  for k in range(topk):
    slot_rows = lax.slice(
        rows_fp8, (k * t_local, 0, 0), ((k + 1) * t_local, lane_blocks, lanes)
    )
    slot_coef = lax.slice(coef_k, (k * t_local,), ((k + 1) * t_local,))
    terms.append(slot_rows.astype(jnp.float32) * slot_coef[:, None, None])
  total = terms[0]
  for term in terms[1:]:
    total = total + term
  return total.astype(out_dtype).reshape(t_local, hidden)


GATEWAY_COMBINE_COLLECTIVE_ID = 2


def gateway_step_tokens(t_local: int, tb: int | None = None) -> int:
  """Tokens per gateway combine step: `tb`, or the combine's default.

  The tables (`build_gateway_combine_tables`) and the kernel
  (`combine_gateway_arrivals`) are striped by step, so both must be given the
  same `tb`.

  Args:
    t_local: Tokens per shard.
    tb: Requested step width, or None for `combine_step_tokens(t_local)`.

  Returns:
    The step width.
  """
  if tb is None:
    return combine_step_tokens(t_local) or t_local
  if (
      tb <= 0
      or t_local % tb
      or tb % U32_SUBLANE_TILE
      or (tb > COMBINE_GROUP and tb % COMBINE_GROUP)
  ):
    raise ValueError(
        f"gateway combine tb={tb} must divide t_local={t_local}, be a multiple"
        f" of {U32_SUBLANE_TILE}, and of {COMBINE_GROUP} above it"
    )
  return tb


def gateway_combine_vmem_bytes(
    *,
    t_local: int,
    topk: int,
    hidden: int,
    ep: int,
    board_size: int,
    wire_dtype: jax.typing.DTypeLike,
    out_dtype: jax.typing.DTypeLike,
    tb: int | None = None,
) -> int:
  """VMEM scratch the (TensorCore) gateway combine kernel allocates."""
  tb = gateway_step_tokens(t_local, tb)
  num_boards = ep // board_size
  total_steps = num_boards * (t_local // tb)
  parities = (
      COMBINE_PARITIES
      if total_steps >= COMBINE_PARITIES and not total_steps % COMBINE_PARITIES
      else 2
  )
  wire = jnp.dtype(wire_dtype).itemsize
  out = jnp.dtype(out_dtype).itemsize
  rows = parities * topk * tb * hidden * wire
  tile = tb * hidden * out
  remote = 2 * (num_boards - 1) * t_local * hidden * out
  own = t_local * hidden * out
  return rows + tile + remote + own


def _onehot_lookup(onehot, table, nbytes):
  """`onehot @ table` for a non-negative int32 table < 2**(8 * nbytes).

  Splits the table into bytes (exact in bf16) so the lookup is one MXU
  matmul rather than a 4-D VPU reduce or an offloaded gather.
  """
  planes = jnp.stack(
      [((table >> (8 * i)) & 0xFF).astype(jnp.bfloat16) for i in range(nbytes)],
      axis=2,
  )  # [B, n_sub, nbytes, E_b]
  got = jnp.round(
      jnp.einsum(
          "bswe,bspe->bswp",
          onehot.astype(jnp.bfloat16),
          planes,
          preferred_element_type=jnp.float32,
      )
  ).astype(jnp.int32)
  out = got[..., 0]
  for i in range(1, nbytes):
    out = out | (got[..., i] << (8 * i))
  return out


def shard_gateway_return_tables(
    routing, expert_rows, me, *, e_total: int, ep: int, board_size: int
):
  """Builds shard `me`'s gateway-staged return push tables and counts.

  Under De-Dup Combine (`dedup_combine=True`), expert `e_local` on device `me`
  (`my_b = me // board_size`) pushes its output rows for source `s` not across
  the inter-board cable to `s`, but over the intra-board links of `my_b` to
  gateway `gw(s, my_b) = my_b * board_size + (s % board_size)`.

  Args:
    routing: `ShardRoutingTables` from `build_routing_tables_sharded`.
    expert_rows: `[g_local]` int32 aligned row counts per local expert.
    me: This shard's index.
    e_total: Total experts `E`.
    ep: Expert-parallel width.
    board_size: Devices per board.

  Returns:
    `(gw_push_dst, gw_counts, gw_recv_base_flat)`:
      `gw_push_dst`: `[g_local, ep]` int32 destination offsets in `ROWBLK`
        blocks inside `gw(s, my_b)`'s arrival buffer.
      `gw_counts`: `[N_COUNTS, ep]` int32 kernel count table for `me`.
      `gw_recv_base_flat`: `[ep, e_total]` int32 per-gateway arrival row offsets
        ordered by `(source_board m, expert_on_board e_b)`.
  """
  g_local = e_total // ep
  num_boards = ep // board_size
  experts_per_board = board_size * g_local
  me = jnp.asarray(me, jnp.int32)
  my_b = lax.div(me, jnp.int32(board_size))
  my_local = lax.rem(me, jnp.int32(board_size))

  # `run_rows_aligned` is `[e_total, ep]`. View as `[b, e_b, m, k]` where
  # `e_global = b * experts_per_board + e_b` and `s = m * board_size + k`.
  aligned_all = routing.run_rows_aligned.astype(jnp.int32)
  aligned_4d = aligned_all.reshape(
      num_boards, experts_per_board, num_boards, board_size
  )
  # Gateway `gw = b * board_size + k` receives runs from board `b` for sources
  # `s_m = m * board_size + k`, laid out in `(m, e_b)` order.
  rows_at_gw = aligned_4d.transpose(0, 3, 2, 1).reshape(ep, e_total)
  gw_recv_base_flat = jnp.cumsum(rows_at_gw, axis=1) - rows_at_gw
  gw_recv_base = gw_recv_base_flat.reshape(
      num_boards, board_size, num_boards, board_size, g_local
  )
  my_gw_dst = lax.dynamic_slice(
      gw_recv_base,
      (my_b, 0, 0, my_local, 0),
      (1, board_size, num_boards, 1, g_local),
  )[0, :, :, 0, :]
  my_gw_dst_rows = my_gw_dst.transpose(2, 1, 0).reshape(g_local, ep)
  gw_push_dst = jnp.right_shift(my_gw_dst_rows, fused_host.ROWBLK_SHIFT).astype(
      jnp.int32
  )

  all_run_rows = routing.run_rows.astype(jnp.int32)
  true_rows = lax.dynamic_slice(all_run_rows, (me * g_local, 0), (g_local, ep))
  aligned = lax.dynamic_slice(aligned_all, (me * g_local, 0), (g_local, ep))
  is_self_gw = (
      lax.rem(jnp.arange(ep, dtype=jnp.int32), jnp.int32(board_size))
      == my_local
  ).astype(jnp.int32)
  other_gw = 1 - is_self_gw
  true_4d = all_run_rows.reshape(
      num_boards, experts_per_board, num_boards, board_size
  )
  gw_recv_rows_all = true_4d.sum(axis=(1, 2)).reshape(ep)
  gw_recv_aligned_all = rows_at_gw.sum(axis=1)
  mine = (jnp.arange(ep, dtype=jnp.int32) == me).astype(jnp.int32)
  first = (jnp.arange(g_local, dtype=jnp.int32) == 0).astype(jnp.int32)[:, None]
  active = (expert_rows > 0).astype(jnp.int32)[:, None]
  # Scale-mirror sublanes: a run owns the same source tiles whichever device
  # it lands on, so the direct layout's `run_tiles` count holds here too.
  all_tiles = routing.run_tiles.astype(jnp.int32)
  tiles = lax.dynamic_slice(all_tiles, (me * g_local, 0), (g_local, ep))
  gw_recv_tiles_all = (
      all_tiles.reshape(num_boards, experts_per_board, num_boards, board_size)
      .sum(axis=(1, 2))
      .reshape(ep)
  )

  def over_experts(term):
    return term.sum(axis=0, keepdims=True)

  gw_counts = jnp.concatenate(
      [
          over_experts(true_rows * other_gw[None, :]),
          over_experts(
              (gw_recv_rows_all * mine)[None, :] * first
              - true_rows * is_self_gw[None, :]
          ),
          over_experts(aligned * other_gw[None, :]),
          over_experts(
              (gw_recv_aligned_all * mine)[None, :] * first
              - aligned * is_self_gw[None, :]
          ),
          over_experts(active * mine[None, :]),
          over_experts(tiles * other_gw[None, :]),
          over_experts(
              (gw_recv_tiles_all * mine)[None, :] * first
              - tiles * is_self_gw[None, :]
          ),
          over_experts(tiles * is_self_gw[None, :]),
      ],
      axis=0,
  ).astype(jnp.int32)
  return gw_push_dst, gw_counts, gw_recv_base_flat


def build_gateway_combine_tables(
    gw_recv_base_flat: jax.Array,
    pairs_g: jax.Array,
    weights_g: jax.Array,
    me: jax.Array,
    *,
    e_total: int,
    ep: int,
    board_size: int,
    t_local: int,
    topk: int,
    block: int,
    sentinel_row: int,
    tb: int | None = None,
    mirror_tile_m: int = 0,
    run_start_aligned: jax.Array | None = None,
):
  """Builds gateway `me`'s compacted `(pos, coef, meta)` combine tables.

  Orders the `num_boards` mirror sources `s_m = m * board_size + (me % bs)` with
  the `num_boards - 1` remote boards first (`m = (my_b + 1 + p) % num_boards`)
  and `me`'s own board last (`m = my_b`), compacts each token's live
  board-`my_b` slots to `0 .. cnt - 1`, clamps inactive slots to `sentinel_row`,
  and computes per-step active-plane bounds `step_max_k`.

  With `mirror_tile_m`, also returns each slot's position in the gateway's
  scale mirror for an FP8 return wire. Run `(m, e_b)` of the gateway's buffer
  owns mirror sublanes from `(gw_recv_base >> log2(tile_m)) + 2 * q`, with
  `q = m * experts_per_board + e_b`, its first row on lane
  `run_start_aligned[e, s_m] % tile_m` (the direct layout's rule with the
  gateway's base and run index); inactive slots get position 0.

  Args:
    gw_recv_base_flat: `[ep, e_total]` int32 from `shard_gateway_return_tables`.
    pairs_g: `[ep, t_local * topk]` int32 global expert selections.
    weights_g: `[ep, t_local * topk]` float32 router weights.
    me: This shard's index.
    e_total: Total experts `E`.
    ep: Expert-parallel width.
    board_size: Devices per board.
    t_local: Tokens per shard.
    topk: Experts per token.
    block: Routing histogram block size.
    sentinel_row: Valid arrival-buffer row index used for inactive slots.
    tb: Tokens per combine step (`gateway_step_tokens`); pass the kernel the
      same value.
    mirror_tile_m: Scale-mirror lane count, or 0 for no mirror table.
    run_start_aligned: `[e_total, ep]` routing `run_start_aligned`; required
      with `mirror_tile_m`.

  Returns:
    `(pos_sorted, coef_sorted, meta)` where `pos_sorted` and `coef_sorted` have
    shape `[num_boards * t_local, topk]` and `meta` has shape
    `[2 + total_steps]`, plus `mirror_sorted` (like `pos_sorted`) with
    `mirror_tile_m`.
  """
  num_boards = ep // board_size
  experts_per_board = board_size * (e_total // ep)
  me = jnp.asarray(me, jnp.int32)
  my_b = lax.div(me, jnp.int32(board_size))
  my_local = lax.rem(me, jnp.int32(board_size))
  e_lo = my_b * jnp.int32(experts_per_board)
  e_hi = e_lo + jnp.int32(experts_per_board)
  my_gw_base = lax.dynamic_slice(
      gw_recv_base_flat, (me, 0), (1, num_boards * experts_per_board)
  )[0].reshape(num_boards, experts_per_board)

  bins_b = jnp.arange(experts_per_board, dtype=jnp.int32)
  sentinel = jnp.int32(sentinel_row)
  tb = gateway_step_tokens(t_local, tb)
  steps_per_board = t_local // tb
  remote_boards = num_boards - 1

  p_iota = jnp.arange(num_boards, dtype=jnp.int32)
  m_vec = lax.rem(my_b + 1 + p_iota, jnp.int32(num_boards))
  s_vec = m_vec * jnp.int32(board_size) + my_local
  pairs_all = jnp.take(pairs_g, s_vec, axis=0)
  weights_all = jnp.take(weights_g, s_vec, axis=0)
  base_all = jnp.take(my_gw_base, m_vec, axis=0)
  on_b = (pairs_all >= e_lo) & (pairs_all < e_hi)
  e_b = jnp.where(on_b, pairs_all - e_lo, -1)

  win = fused_host.rank_window(block, t_local * topk)
  n_sub = (t_local * topk) // win
  blocks_all = e_b.reshape(num_boards, n_sub, win)
  oh_all = blocks_all[:, :, :, None] == bins_b[None, None, None, :]
  hist_all = oh_all.astype(jnp.int32).sum(axis=2)
  off_all = jnp.cumsum(hist_all, axis=1) - hist_all
  eq_all = blocks_all[:, :, :, None] == blocks_all[:, :, None, :]
  tri_w = jnp.tril(jnp.ones((win, win), dtype=jnp.bool_), k=-1)
  rank_all = jnp.sum(
      (eq_all & tri_w[None, None, :, :]).astype(jnp.int32), axis=3
  )
  # Arrival rows index the table; three bytes past 65,535.
  base_pos_all = _onehot_lookup(
      oh_all, off_all + base_all[:, None, :], 2 if sentinel_row < 65536 else 3
  )
  raw_pos_all = (base_pos_all + rank_all).reshape(num_boards, t_local, topk)
  on_b_3d = on_b.reshape(num_boards, t_local, topk)
  pos_all = jnp.where(
      on_b_3d, jnp.minimum(raw_pos_all, sentinel), sentinel
  ).astype(jnp.int32)
  coef_all = jnp.where(
      on_b_3d, weights_all.reshape(num_boards, t_local, topk), 0.0
  ).astype(jnp.float32)
  planes = [pos_all, coef_all]
  if mirror_tile_m:
    if run_start_aligned is None:
      raise ValueError("mirror_tile_m needs run_start_aligned")
    shift = pow2_shift(mirror_tile_m, "the mirror's lane count")
    q_all = m_vec[:, None] * jnp.int32(experts_per_board) + bins_b[None, :]
    # run_start_aligned[e_lo + e_b, s_vec[p]] -> [num_boards, experts_per_board]
    rs = lax.dynamic_slice(
        run_start_aligned.astype(jnp.int32),
        (e_lo, 0),
        (experts_per_board, ep),
    )
    rs_all = jnp.take(rs, s_vec, axis=1).T
    mirror_table = (
        off_all
        + (
            (jnp.right_shift(base_all, shift) + 2 * q_all) * mirror_tile_m
            + (rs_all & (mirror_tile_m - 1))
        )[:, None, :]
    )
    # Mirror positions stay below 2**24 at every supported shape.
    raw_mirror = (_onehot_lookup(oh_all, mirror_table, 3) + rank_all).reshape(
        num_boards, t_local, topk
    )
    planes.append(jnp.where(on_b_3d, raw_mirror, 0).astype(jnp.int32))
  # Compact active board-`my_b` slots `0 .. cnt - 1` to the front of `[topk]`
  # without a sort.
  on_i = on_b_3d.astype(jnp.int32)
  cnt_tok = jnp.sum(on_i, axis=2)  # [num_boards, t_local]
  rank_on = jnp.cumsum(on_i, axis=2) - on_i  # [num_boards, t_local, topk]
  off_i = 1 - on_i
  rank_off = jnp.cumsum(off_i, axis=2) - off_i
  slot_dst = jnp.where(
      on_b_3d, rank_on, cnt_tok[:, :, None] + rank_off
  )  # [num_boards, t_local, topk]
  k_out = jnp.arange(topk, dtype=jnp.int32)[None, None, None, :]
  sel_k = slot_dst[:, :, :, None] == k_out  # [num_boards, t_local, k_in, k_out]
  max_k_all = jnp.max(
      cnt_tok.reshape(num_boards, steps_per_board, tb),
      axis=2,
  )

  def compact_and_stripe(plane):
    zero = jnp.zeros((), plane.dtype)
    s = jnp.sum(jnp.where(sel_k, plane[:, :, :, None], zero), axis=2)
    rem = (
        s[:remote_boards]
        .reshape(remote_boards, steps_per_board, tb, topk)
        .transpose(1, 0, 2, 3)
        .reshape(remote_boards * t_local, topk)
    )
    return jnp.concatenate([rem, s[remote_boards]], axis=0)

  out = [compact_and_stripe(p) for p in planes]
  max_k_rem_striped = max_k_all[:remote_boards].T.reshape(-1)
  step_max_k = jnp.concatenate(
      [max_k_rem_striped, max_k_all[remote_boards]], axis=0
  )
  meta = jnp.concatenate([me[None], my_b[None], step_max_k]).astype(jnp.int32)
  if mirror_tile_m:
    return out[0], out[1], meta, out[2]
  return out[0], out[1], meta


def _build_gateway_combine_kernel(
    *,
    t_local: int,
    topk: int,
    hidden: int,
    lane_blocks: int,
    ep: int,
    board_size: int,
    wire_dtype: jax.typing.DTypeLike,
    out_dtype: jax.typing.DTypeLike,
    sc_combine: bool = False,
    sc_subcores: int = 4,
    has_order_after: bool = False,
    tb: int | None = None,
):
  """Fused Pallas gateway combine + mirror-peer return + home sum kernel."""
  tb = gateway_step_tokens(t_local, tb)
  steps_per_board = t_local // tb
  gt = min(COMBINE_GROUP, tb)
  groups = tb // gt
  chunk_w = min(COMBINE_CHUNK, tb)
  lanes = HIDDEN_LANE_BLOCK
  num_boards = ep // board_size
  remote_boards = num_boards - 1
  remote_steps = remote_boards * steps_per_board
  total_steps = num_boards * steps_per_board
  pipelined_gw = (
      total_steps >= COMBINE_PARITIES and total_steps % COMBINE_PARITIES == 0
  )
  parities = COMBINE_PARITIES if pipelined_gw else 2
  lead = COMBINE_LEAD if pipelined_gw else 1

  def kernel(*refs):
    absent: Any = None
    it = iter(refs)
    pos_sm = next(it)
    coef_sm = next(it)
    meta_sm = next(it)
    arr_hbm = next(it)
    if sc_combine:
      send_rem_hbm = next(it)
      recv_rem_hbm = next(it)
      comb_recv_sem = next(it)
      sc_comb_sems = next(it)
    else:
      send_rem_hbm = recv_rem_hbm = comb_recv_sem = sc_comb_sems = absent
    if has_order_after:
      order_after_hbm = next(it)
      del order_after_hbm
    out_hbm = next(it)
    rows_vm = next(it)
    tile_vm = next(it)
    if sc_combine:
      send_vm = recv_rem_vm = remote_send_sem = remote_recv_sem = absent
      own_vm = next(it)
      row_sems = next(it)
      local_sem = next(it)
      rem_vm = next(it)
      rem_sems = next(it)
    else:
      rem_vm = rem_sems = absent
      send_vm = next(it)
      recv_rem_vm = next(it)
      own_vm = next(it)
      row_sems = next(it)
      remote_send_sem = next(it)
      remote_recv_sem = next(it)
      local_sem = next(it)

    # Inactive slots of a fetched plane point at the arrival buffer's last row
    # with coefficient 0 (build_gateway_combine_tables). No expert writes that
    # row, so it is zeroed here: uninitialized contents times 0 is NaN when
    # they are NaN or infinity. The write overlaps the barrier handshake and
    # has landed before the first row fetch starts.
    sentinel_idx = arr_hbm.shape[0] - 1
    rows_vm[0, pl.ds(0, 1)] = jnp.zeros((1, lane_blocks, lanes), wire_dtype)
    pltpu.make_async_copy(
        rows_vm.at[0, pl.ds(0, 1)],
        arr_hbm.at[pl.ds(sentinel_idx, 1)],
        local_sem,
    ).start()

    bsem = pltpu.get_barrier_semaphore()
    for i in range(ep):
      pl.semaphore_signal(bsem, inc=1, device_id=(jnp.int32(i),))
    pl.semaphore_wait(bsem, ep)

    pltpu.make_async_copy(
        rows_vm.at[0, pl.ds(0, 1)],
        rows_vm.at[0, pl.ds(0, 1)],
        local_sem,
    ).wait()

    me = meta_sm[0]
    my_b = meta_sm[1]
    my_local = lax.rem(me, jnp.int32(board_size))

    def start_step_rows(step, static_parity):
      base = step * tb
      max_k = meta_sm[2 + step]
      for k in range(topk):

        @pl.when(k < max_k)
        def _(k=k):
          for i in range(tb):
            pltpu.make_async_copy(
                arr_hbm.at[pos_sm[(base + i) * topk + k]],
                rows_vm.at[static_parity, k * tb + i],
                row_sems.at[static_parity],
            ).start(priority=k % 2)

    def wait_step_rows(step, static_parity):
      # One static-size wait per fetched plane: max_k * tb rows in all.
      max_k = meta_sm[2 + step]
      for k in range(topk):

        @pl.when(k < max_k)
        def _(k=k):
          pltpu.make_async_copy(
              rows_vm.at[static_parity, pl.ds(k * tb, tb)],
              rows_vm.at[static_parity, pl.ds(k * tb, tb)],
              row_sems.at[static_parity],
          ).wait()

    def reduce_step(step, static_parity):
      base = step * tb
      max_k = jnp.clip(meta_sm[2 + step], 0, topk)

      def reduce_zero():
        tile_vm[:] = jnp.zeros((tb, lane_blocks, lanes), out_dtype)

      def make_reduce_k(num_k: int):
        def _fn():
          def one_chunk(chunk, carry):
            for u in range(chunk_w):
              i = chunk * chunk_w + u
              acc: Any = None
              for k in range(num_k):
                term = (
                    rows_vm[static_parity, k * tb + i].astype(jnp.float32)
                    * coef_sm[(base + i) * topk + k]
                )
                acc = term if acc is None else acc + term
              tile_vm[i] = acc.astype(out_dtype)
            return carry

          lax.fori_loop(0, tb // chunk_w, one_chunk, jnp.int32(0))

        return _fn

      branches = [reduce_zero] + [make_reduce_k(k) for k in range(1, topk + 1)]
      lax.switch(max_k, branches)

    def run_step_parity(step, static_parity):
      nxt_step = step + lead
      nxt_parity = (static_parity + lead) % parities
      wait_step_rows(step, static_parity)

      @pl.when(nxt_step < total_steps)
      def _():
        start_step_rows(nxt_step, nxt_parity)

      reduce_step(step, static_parity)

      @pl.when(step < remote_steps)
      def _():
        if sc_combine:
          pltpu.make_async_copy(
              tile_vm,
              send_rem_hbm.at[pl.ds(step * tb, tb)],
              local_sem,
          ).start()
          pltpu.make_async_copy(
              tile_vm,
              send_rem_hbm.at[pl.ds(step * tb, tb)],
              local_sem,
          ).wait()
          for sub in range(sc_subcores):
            pl.semaphore_signal(
                sc_comb_sems.at[step],
                inc=1,
                device_id={"cae_core": 0, "cae_sub": sub},
            )
        else:
          for g in range(groups):
            send_vm[pl.ds(step * tb + g * gt, gt)] = tile_vm[
                pl.ds(g * gt, gt)
            ].reshape(gt, hidden)
          s_step = lax.div(step, jnp.int32(remote_boards))
          peer_idx = lax.rem(step, jnp.int32(remote_boards))
          dst_b = lax.rem(my_b + 1 + peer_idx, jnp.int32(num_boards))
          dst_dev = dst_b * jnp.int32(board_size) + my_local
          dst_row = (
              peer_idx * jnp.int32(steps_per_board) + s_step
          ) * jnp.int32(tb)
          pltpu.make_async_remote_copy(
              src_ref=send_vm.at[pl.ds(step * tb, tb)],
              dst_ref=recv_rem_vm.at[pl.ds(dst_row, tb)],
              send_sem=remote_send_sem,
              recv_sem=remote_recv_sem.at[s_step],
              device_id=(dst_dev,),
          ).start()

      @pl.when(step >= remote_steps)
      def _():
        s_step = step - jnp.int32(remote_steps)
        if sc_combine:
          for g in range(groups):
            own_vm[pl.ds(s_step * tb + g * gt, gt)] = tile_vm[
                pl.ds(g * gt, gt)
            ].reshape(gt, hidden)
        else:
          for p in range(remote_boards):
            src_row = (p * steps_per_board + s_step) * tb
            pltpu.make_async_copy(
                recv_rem_vm.at[pl.ds(src_row, tb)],
                recv_rem_vm.at[pl.ds(src_row, tb)],
                remote_recv_sem.at[s_step],
            ).wait()
          for g in range(groups):
            own_vm[pl.ds(s_step * tb + g * gt, gt)] = tile_vm[
                pl.ds(g * gt, gt)
            ].reshape(gt, hidden)
          acc = own_vm[pl.ds(s_step * tb, tb)].astype(jnp.float32)
          for p in range(remote_boards):
            src_row = (p * steps_per_board + s_step) * tb
            acc = acc + recv_rem_vm[pl.ds(src_row, tb)].astype(jnp.float32)
          own_vm[pl.ds(s_step * tb, tb)] = acc.astype(out_dtype)
          pltpu.make_async_copy(
              own_vm.at[pl.ds(s_step * tb, tb)],
              out_hbm.at[pl.ds(s_step * tb, tb)],
              local_sem,
          ).start()

    for p in range(lead):
      start_step_rows(jnp.int32(p), p)

    if total_steps % parities == 0:

      def step_group(grp, carry):
        base_step = grp * parities
        for p in range(parities):
          run_step_parity(base_step + p, p)
        return carry

      lax.fori_loop(0, total_steps // parities, step_group, jnp.int32(0))
    else:

      def step_body(step, carry):
        parity = lax.rem(step, jnp.int32(parities))
        for p in range(parities):

          @pl.when(parity == p)
          def _(p=p):
            run_step_parity(step, p)

        return carry

      lax.fori_loop(0, total_steps, step_body, jnp.int32(0))

    if sc_combine:
      pltpu.make_async_copy(
          recv_rem_hbm, recv_rem_hbm, comb_recv_sem
      ).wait()

      def sum_step_sc(s_step, carry):
        # Remote partials are out_dtype and one per remote board, so they
        # stage in their own buffer rather than the wire-typed row parities.
        for p in range(remote_boards):
          src_row = (p * steps_per_board + s_step) * tb
          pltpu.make_async_copy(
              recv_rem_hbm.at[pl.ds(src_row, tb)],
              rem_vm.at[p],
              rem_sems.at[p],
          ).start()
        rem_sum: Any = None
        for p in range(remote_boards):
          pltpu.make_async_copy(
              rem_vm.at[p],
              rem_vm.at[p],
              rem_sems.at[p],
          ).wait()
          part = rem_vm[p].astype(jnp.float32)
          rem_sum = part if rem_sum is None else rem_sum + part
        tile_vm[:] = rem_sum.astype(out_dtype)
        for g in range(groups):
          off = s_step * tb + g * gt
          own_vm[pl.ds(off, gt)] = (
              own_vm[pl.ds(off, gt)].astype(jnp.float32)
              + tile_vm[pl.ds(g * gt, gt)]
              .reshape(gt, hidden)
              .astype(jnp.float32)
          ).astype(out_dtype)
        pltpu.make_async_copy(
            own_vm.at[pl.ds(s_step * tb, tb)],
            out_hbm.at[pl.ds(s_step * tb, tb)],
            local_sem,
        ).start()
        return carry

      lax.fori_loop(0, steps_per_board, sum_step_sc, jnp.int32(0))
      pltpu.make_async_copy(own_vm, own_vm, local_sem).wait()
    else:
      pltpu.make_async_copy(send_vm, send_vm, remote_send_sem).wait()
      pltpu.make_async_copy(own_vm, own_vm, local_sem).wait()

    for i in range(ep):
      pl.semaphore_signal(bsem, inc=1, device_id=(jnp.int32(i),))
    pl.semaphore_wait(bsem, ep)

  hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
  out_shape = jax.ShapeDtypeStruct((t_local, hidden), out_dtype)
  if sc_combine:
    # SparseCore Pallas is imported only here: older JAX releases lack it.
    from jax.experimental.pallas import tpu_sc as plsc  # pylint: disable=g-import-not-at-top

    scm_comb = plsc.VectorSubcoreMesh(
        core_axis_name="cae_core",
        subcore_axis_name="cae_sub",
        num_cores=1,
        num_subcores=sc_subcores,
    )
    in_specs = [
        hbm,
        hbm,
        hbm,
        pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        pl.BlockSpec(memory_space=pltpu.SEMAPHORE @ scm_comb),
    ] + ([hbm] if has_order_after else [])
    scratch_shapes = (
        pltpu.VMEM((parities, topk * tb, lane_blocks, lanes), wire_dtype),
        pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
        pltpu.VMEM((t_local, hidden), out_dtype),
        pltpu.SemaphoreType.DMA((parities,)),
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((remote_boards, tb, lane_blocks, lanes), out_dtype),
        pltpu.SemaphoreType.DMA((remote_boards,)),
    )
  else:
    in_specs = [hbm] + ([hbm] if has_order_after else [])
    scratch_shapes = (
        pltpu.VMEM((parities, topk * tb, lane_blocks, lanes), wire_dtype),
        pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
        pltpu.VMEM((remote_boards * t_local, hidden), out_dtype),
        pltpu.VMEM((remote_boards * t_local, hidden), out_dtype),
        pltpu.VMEM((t_local, hidden), out_dtype),
        pltpu.SemaphoreType.DMA((parities,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((steps_per_board,)),
        pltpu.SemaphoreType.DMA,
    )
  sc_sfx = f"_sc{sc_subcores}" if sc_combine else ""
  call = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=in_specs,
          out_specs=hbm,
          scratch_shapes=scratch_shapes,
          grid=(),
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(
          collective_id=GATEWAY_COMBINE_COLLECTIVE_ID,
          vmem_limit_bytes=fused_host.vmem_limit(),
          disable_bounds_checks=True,
      ),
      name=(
          f"dedup_gw_combine_t{t_local}_k{topk}_b{tb}_ep{ep}_bs{board_size}"
          f"{sc_sfx}"
      ),
  )

  def fn(
      arrivals,
      coef_sorted,
      pos_sorted,
      meta,
      send_rem_buf=None,
      recv_rem_buf=None,
      comb_recv_sem=None,
      sc_comb_sems=None,
      order_after=None,
  ):
    extra = (
        (send_rem_buf, recv_rem_buf, comb_recv_sem, sc_comb_sems)
        if sc_combine
        else ()
    ) + ((order_after,) if has_order_after else ())
    return call(
        pos_sorted.astype(jnp.int32).reshape(-1),
        coef_sorted.astype(jnp.float32).reshape(-1),
        meta.astype(jnp.int32).reshape(-1),
        arrivals,
        *extra,
    )

  return fn


_GW_COMBINE_CACHE = {}
_GW_COMBINE_CACHE_LOCK = threading.Lock()


def build_gateway_combine_kernel(**kwargs):
  """Memoizing front door for `_build_gateway_combine_kernel`."""
  key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
  fn = _GW_COMBINE_CACHE.get(key)
  if fn is None:
    with _GW_COMBINE_CACHE_LOCK:
      fn = _GW_COMBINE_CACHE.get(key)
      if fn is None:
        fn = _build_gateway_combine_kernel(**kwargs)
        _GW_COMBINE_CACHE[key] = fn
  return fn


def combine_gateway_arrivals(
    arrivals: jax.Array,
    pos_sorted: jax.Array,
    coef_sorted: jax.Array,
    meta: jax.Array,
    *,
    t_local: int,
    topk: int,
    ep: int,
    board_size: int,
    out_dtype: jax.typing.DTypeLike,
    sc_combine: bool = False,
    sc_subcores: int = 4,
    send_rem_buf: jax.Array | None = None,
    recv_rem_buf: jax.Array | None = None,
    comb_recv_sem: Any = None,
    sc_comb_sems: Any = None,
    order_after: Any = None,
    tb: int | None = None,
    arrival_scales: jax.Array | None = None,
    mirror_pos_sorted: jax.Array | None = None,
) -> jax.Array:
  """Combines gateway-staged arrivals `[recv_rows, lane_blocks, 128]` into `[t_local, hidden]`.

  Every device of the `ep` axis must call it in the same program: the kernel
  barriers across all of them and pushes each board's partial sums to the
  token's home device.

  Args:
    arrivals: `[recv_rows, lane_blocks, 128]` gateway arrival buffer. Its last
      row is the tables' `sentinel_row`; the kernel overwrites it with zeros.
    pos_sorted: `[num_boards * t_local, topk]` from
      `build_gateway_combine_tables`.
    coef_sorted: Same shape, router weights (0 on inactive slots).
    meta: `[2 + num_boards * t_local // tb]` int32 `[me, my_b, step_max_k..]`.
    t_local: Tokens per shard.
    topk: Experts per token.
    ep: Expert-parallel width.
    board_size: Devices per board.
    out_dtype: Element type of the partials and the output.
    sc_combine: Sum the remote partials on SparseCore (needs `tpu_sc`).
    sc_subcores: SparseCore vector subcores for `sc_combine`.
    send_rem_buf: `sc_combine` HBM send buffer,
      `[remote_boards * t_local, lane_blocks, 128]` of `out_dtype`.
    recv_rem_buf: `sc_combine` HBM receive buffer, same shape.
    comb_recv_sem: `sc_combine` receive semaphore.
    sc_comb_sems: `sc_combine` per-step SparseCore semaphores.
    order_after: Optional HBM array to order the pallas_call after.
    tb: Tokens per step; must match `build_gateway_combine_tables`'s.
    arrival_scales: `[sublanes, tile_m]` f32 scale mirror of a quantized wire,
      or None (bf16 wire).
    mirror_pos_sorted: `[num_boards * t_local, topk]` mirror positions from
      `build_gateway_combine_tables(mirror_tile_m=tile_m)`; required with
      `arrival_scales`.

  Returns:
    `[t_local, hidden]` combined tokens of this device.
  """
  _, lane_blocks, lanes = arrivals.shape
  hidden = lane_blocks * lanes
  num_boards = ep // board_size
  step_tokens = gateway_step_tokens(t_local, tb)
  want_pos = (num_boards * t_local, topk)
  want_meta = (2 + num_boards * (t_local // step_tokens),)
  if pos_sorted.shape != want_pos or coef_sorted.shape != want_pos:
    raise ValueError(
        f"gateway combine tables must be {want_pos}, got pos"
        f" {pos_sorted.shape} and coef {coef_sorted.shape}"
    )
  if meta.shape != want_meta:
    raise ValueError(
        f"gateway combine meta must be {want_meta} for tb={step_tokens}, got"
        f" {meta.shape}; pass the tables and the kernel the same tb"
    )
  if arrival_scales is not None and arrivals.dtype != jnp.bfloat16:
    if mirror_pos_sorted is None:
      raise ValueError("arrival_scales needs mirror_pos_sorted")
    mirror_sublanes, tile_m = arrival_scales.shape
    pow2_shift(tile_m, "the mirror's lane count")
    scl = jnp.minimum(mirror_pos_sorted, mirror_sublanes * tile_m - 1)
    row_scales = jnp.take(arrival_scales.reshape(-1), scl, axis=0)
    # Inactive slots (coefficient 0) may name a mirror lane no run wrote.
    coef_sorted = jnp.where(
        coef_sorted != 0, coef_sorted.astype(jnp.float32) * row_scales, 0.0
    )
  kw: dict[str, Any] = {}
  if tb is not None:
    kw["tb"] = step_tokens
  combine = build_gateway_combine_kernel(
      t_local=t_local,
      topk=topk,
      hidden=hidden,
      lane_blocks=lane_blocks,
      ep=ep,
      board_size=board_size,
      wire_dtype=arrivals.dtype,
      out_dtype=out_dtype,
      sc_combine=sc_combine,
      sc_subcores=sc_subcores,
      has_order_after=order_after is not None,
      **kw,
  )
  return combine(
      arrivals,
      coef_sorted,
      pos_sorted,
      meta,
      send_rem_buf=send_rem_buf,
      recv_rem_buf=recv_rem_buf,
      comb_recv_sem=comb_recv_sem,
      sc_comb_sems=sc_comb_sems,
      order_after=order_after,
  )
