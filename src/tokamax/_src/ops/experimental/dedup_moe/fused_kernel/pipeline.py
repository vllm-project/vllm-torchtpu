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

"""Main kernel module for orchestrating DMA and computation pipeline.

Separates the stateful prologue, inner per-expert tile pipeline,
progressive/VMEM
return pushes, and epilogue transport drain from the stateless expert FFN
compute (shared with `fused_moe.kernel`) and the memory reference helpers
(`memory_ref.py`).
"""

import contextlib
import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import config
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import memory_ref
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import plan as plan_lib
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import tg_fused
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import kernel as fused_kernel


def make_dedup_moe_kernel_body(cfg: config.FusedKernelConfig):
  """Builds the Pallas kernel body for a resolved config.

  Args:
    cfg: The resolved FusedKernelConfig.

  Returns:
    The kernel function `kernel(*refs)` that pl.pallas_call runs.
  """
  g_local = cfg.g_local
  hidden = cfg.hidden
  ep = cfg.ep
  weight_format = cfg.weight_format
  act_fn = cfg.act_fn
  has_w1_bias = cfg.has_w1_bias
  has_w2_bias = cfg.has_w2_bias
  waves = cfg.waves
  dedup_slots = cfg.dedup_slots
  wait_for_exchange = cfg.wait_for_exchange
  tc_dedup = cfg.tc_dedup
  sc_ret = cfg.sc_ret
  sc_ret_sub = cfg.sc_return_subcores - 1
  fwd_max_runs = cfg.fwd_max_runs
  nb = cfg.nb

  def scope(name: str):
    """A device trace region (cfg.trace_scopes), else a no-op context."""
    if cfg.trace_scopes:
      return jax.named_scope(name)
    return contextlib.nullcontext()

  wave_bounds_base = cfg.wave_bounds_base
  form = cfg.form
  skip_scale_mirror = cfg.skip_scale_mirror
  lane_blocks = cfg.lane_blocks
  has_scales = cfg.has_scales
  has_act_scale = cfg.has_act_scale
  has_ls_vm = cfg.has_ls_vm
  prefetch_distance = cfg.prefetch_distance
  nbuf = cfg.nbuf
  prefetches = cfg.prefetches
  tile_m = cfg.tile_m
  tile_blocks = cfg.tile_blocks
  tile_block_shift = cfg.tile_block_shift
  tile_shift = cfg.tile_shift
  height_rungs = cfg.height_rungs
  push_from_vmem = cfg.push_from_vmem
  progressive_push = cfg.progressive_push
  ls_window_rows = cfg.ls_window_rows
  gather_tiles = cfg.gather_tiles
  gather_payload_rows = cfg.gather_payload_rows
  gather_window_rows = cfg.gather_window_rows
  annotate_no_hazard = cfg.annotate_no_hazard
  defer_remote_push = cfg.defer_remote_push
  branchless_gather_window = cfg.branchless_gather_window
  loop_ep_push = cfg.loop_ep_push
  epilogue_barrier = cfg.epilogue_barrier
  early_weight_refill = cfg.early_weight_refill and cfg.prefetches
  fused_tg = cfg.fuse_tg_assemble
  lhs_issue_unroll = cfg.lhs_issue_unroll
  pp_lag = cfg.progressive_push_lag
  prologue_full_prefetch = cfg.prologue_full_prefetch and cfg.prefetches
  fuse_tg_copy = cfg.fuse_tg_copy
  split_weight_waits = (
      cfg.split_weight_waits and cfg.weight_format != arg.WeightFormat.BF16
  )
  prime_before_ship = cfg.prime_before_ship and push_from_vmem
  overlap_prologue_sync = cfg.overlap_prologue_sync
  chunked_epilogue = cfg.chunked_epilogue
  hoist_single_wave_wait = (
      overlap_prologue_sync
      and cfg.wait_for_exchange
      and cfg.waves == 1
      and not cfg.tc_dedup
  )

  rowblk = fused_host.ROWBLK
  lanes = fused_host.HIDDEN_LANE_BLOCK
  scale_sublanes = max(1, tile_m // lanes)
  scale_cols = tile_m // scale_sublanes
  out_parities = fused_host.OUT_PARITIES
  weight_dma_priority = config.WEIGHT_DMA_PRIORITY

  def kernel(*refs):
    # Step 1: Unpack SMEM scalar-prefetch tables, HBM operands, and
    # VMEM/DMA scratch refs.
    it = iter(refs)
    absent: Any = None

    def take(present: bool = True) -> Any:
      """The next ref when the build has it, else None."""
      return next(it) if present else absent

    # Block-unit transport tables, from shard_transport_tables_in_blocks.
    # Three of that builder's tables are the same array as another it
    # already ships, so each arrives once and the duplicates are derived
    # here instead of being prefetched again:
    #   push_src == contrib_off  -- a push reads its source at the offset
    #     the commit wrote it to;
    #   push_len == commit_len   -- `region_rows` IS `run_rows_aligned`
    #     reshaped, so slicing either at this shard gives one array;
    #   recv_row_off == push_dst << ROWBLK_SHIFT -- `recv_base` is a cumsum
    #     of ALIGNED run lengths, so the row offset is the block offset in
    #     rows.
    commit_start_sm, commit_len_sm, contrib_off_sm, push_dst_sm = (
        next(it) for _ in range(config.N_TRANSPORT_TABLES)
    )
    # Row-unit push tables, from shard_push_tables_in_rows.
    (true_rows_sm,) = (next(it) for _ in range(config.N_PUSH_TABLES))
    # (rows, base) i32 [G] row-unit tables from shard_expert_slabs.
    expert_rows_sm, expert_base_sm = (
        next(it) for _ in range(config.N_SLAB_TABLES)
    )
    # visit_sm[visit_i] is the real expert at compacted step visit_i.
    visit_sm = next(it)
    # The five counts, each still spread over the expert-parallel axis.
    counts_sm = next(it)
    # This shard's expert-parallel index, as data rather than `lax.axis_index`.
    rank_sm = next(it)
    if tc_dedup:
      stage_tab_sm = next(it)
      fwd_tab_sm = next(it)
      fwd_cnt_sm = next(it)
    else:
      stage_tab_sm = fwd_tab_sm = fwd_cnt_sm = absent
    tg_runs_sm = take(fused_tg)
    tg_meta_sm = take(fused_tg)

    def count(row):
      """Close one count's reduction on the scalar core."""
      total = counts_sm[row, 0]
      for d in range(1, ep):
        total = total + counts_sm[row, d]
      return total

    n_visit = count(fused_host.COUNT_VISITS)
    token_gather_hbm = take(not fused_tg)
    lhs_hbm = next(it)
    disp_tab_hbm = take(wait_for_exchange)
    ls_hbm = take(has_ls_vm and not fused_tg)
    w1_hbm, w2_hbm = (next(it) for _ in range(2))
    w1s_hbm = take(has_scales)
    w2s_hbm = take(has_scales)
    w1b_hbm = take(has_w1_bias)
    w2b_hbm = take(has_w2_bias)
    stage_hbm = take(tc_dedup)
    tg_land_hbm = take(fused_tg)
    if sc_ret:
      recv_hbm = next(it)
      contrib_hbm = next(it)
    wave_sems = take(wait_for_exchange)
    stage_sems = take(tc_dedup)
    tg_sem = take(fused_tg and wait_for_exchange)
    if sc_ret:
      ret_recv_sem = next(it)
      sc_ret_sems = next(it)
      rscl_hbm = next(it)
      cscl_hbm = absent
    else:
      ret_recv_sem = sc_ret_sems = absent
      recv_hbm = next(it)
      rscl_hbm = next(it)
      contrib_hbm = next(it)
      cscl_hbm = next(it)
    if branchless_gather_window:
      token_gather_sm = next(it)
      token_gather_sm0 = token_gather_sm1 = absent
    else:
      token_gather_sm0, token_gather_sm1 = (next(it) for _ in range(2))
      token_gather_sm = absent
    lhs_vm, w1_vm, w2_vm = (next(it) for _ in range(3))
    w1s_vm = take(has_scales)
    w2s_vm = take(has_scales)
    w1b_vm = take(has_w1_bias)
    w2b_vm = take(has_w2_bias)
    ls_vm = take(has_ls_vm)
    out_vm = next(it)
    oscl_vm = next(it)
    mid_q_vm = take(chunked_epilogue)
    down_vm = take(chunked_epilogue)
    sinv_vm = take(chunked_epilogue)
    # Parity-buffered staging refs are explicitly guarded by DMA semaphores
    # (`lhs_sems`, `send_sems` / `commit_sems`, `send_scl_sems` /
    # `commit_scl_sems`). Annotate them with `no_hazard_no_deps=True` so Mosaic
    # does not insert false alias dependencies between tile t+1's DMAs and
    # tile t's compute.
    lhs_vm = memory_ref.maybe_annotate_no_hazard(
        lhs_vm, enabled=annotate_no_hazard
    )
    if has_ls_vm:
      ls_vm = memory_ref.maybe_annotate_no_hazard(
          ls_vm, enabled=annotate_no_hazard
      )
    out_vm = memory_ref.maybe_annotate_no_hazard(
        out_vm, enabled=annotate_no_hazard
    )
    oscl_vm = memory_ref.maybe_annotate_no_hazard(
        oscl_vm, enabled=annotate_no_hazard
    )
    if branchless_gather_window:
      token_gather_sems = next(it)
      token_gather_sem0 = token_gather_sem1 = absent
    else:
      token_gather_sem0, token_gather_sem1 = (next(it) for _ in range(2))
      token_gather_sems = absent
    lhs_sems, w1_sems, w2_sems = (next(it) for _ in range(3))
    w1s_sems = take(has_scales)
    w2s_sems = take(has_scales)
    cp_sem = next(it)
    if sc_ret:
      commit_sems = next(it)
      commit_scl_sems = next(it)
      send_sems = recv_sem = send_scl_sems = recv_scl_sem = absent
      send_sem = send_scl_sem = mehop_sem = mehop_scl_sem = absent
    elif push_from_vmem:
      send_sems, recv_sem = (next(it) for _ in range(2))
      send_scl_sems, recv_scl_sem = (next(it) for _ in range(2))
      commit_sems = commit_scl_sems = absent
      mehop_sem = mehop_scl_sem = absent
    else:
      commit_sems, send_sem, recv_sem = (next(it) for _ in range(3))
      commit_scl_sems, send_scl_sem, recv_scl_sem = (next(it) for _ in range(3))
      mehop_sem = next(it)
      mehop_scl_sem = next(it)
    disp_tab_sm, waves_done_sm = (
        take(wait_for_exchange),
        take(wait_for_exchange),
    )
    if tc_dedup:
      fwd_done_sm = next(it)
      fwd_send_sem = next(it)
    else:
      fwd_done_sm = fwd_send_sem = absent
    if fused_tg:
      tg_land_vm = next(it)
      tg_vm = next(it)
      tg_scale_vm = take(has_ls_vm)
      tg_sm = next(it)
      tg_sems = next(it)
      # The tile's scale window is a VMEM -> VMEM copy out of the image.
      ls_hbm = tg_scale_vm
    else:
      tg_land_vm = tg_vm = tg_scale_vm = tg_sm = tg_sems = absent

    me = rank_sm[0, 0]

    def sync(src, dst):
      """Copy `src` into `dst` and wait for it before returning."""
      copy = pltpu.make_async_copy(src, dst, cp_sem)
      copy.start()
      copy.wait()

    def w1_copy(expert, slot):
      """The gate/up weight slab of `expert`, into weight slot `slot`."""
      return memory_ref.slot_copy(w1_hbm.at[expert], w1_vm, w1_sems, slot)

    def w2_copy(expert, slot):
      """The down weight slab of `expert`, into weight slot `slot`."""
      return memory_ref.slot_copy(w2_hbm.at[expert], w2_vm, w2_sems, slot)

    def w1s_copy(expert, slot):
      """The gate/up weight-scale row of `expert`, into weight slot `slot`."""
      return pltpu.make_async_copy(
          w1s_hbm.at[pl.ds(expert, 1)],
          w1s_vm.at[pl.ds(slot, 1)],
          w1s_sems.at[slot],
      )

    def w2s_copy(expert, slot):
      """The down weight-scale row of `expert`, into weight slot `slot`."""
      return pltpu.make_async_copy(
          w2s_hbm.at[pl.ds(expert, 1)],
          w2s_vm.at[pl.ds(slot, 1)],
          w2s_sems.at[slot],
      )

    def weight_starts(expert, slot):
      """Start every buffer slot `slot` holds for `expert`."""
      if has_scales:
        w1s_copy(expert, slot).start(priority=weight_dma_priority)
        w2s_copy(expert, slot).start(priority=weight_dma_priority)
      w1_copy(expert, slot).start(priority=weight_dma_priority)
      w2_copy(expert, slot).start(priority=weight_dma_priority)

    def weight_waits(expert, slot):
      """Wait for every buffer slot `slot` holds for `expert`."""
      w1_copy(expert, slot).wait()
      if has_scales:
        w1s_copy(expert, slot).wait()
      if not split_weight_waits:
        w2_copy(expert, slot).wait()
        if has_scales:
          w2s_copy(expert, slot).wait()

    def weight_waits_w2(expert, slot):
      """Wait for `w2` (and `w2s`) when `split_weight_waits` deferred it."""
      w2_copy(expert, slot).wait()
      if has_scales:
        w2s_copy(expert, slot).wait()

    def weight_refill(visit_i):
      """Start the weights of visit `visit_i + prefetch_distance`, if any.

      The refilled slot is (visit_i + distance) % nbuf == (visit_i - 1) % nbuf
      (nbuf == distance + 1), the slot of the expert visited just before
      `visit_i`. Both call sites run after that expert's last matmul, so the
      slot has no live reader. When `prologue_full_prefetch` is enabled,
      `visit_sm[prefetch_distance]` was already started in `prologue()`, so
      `visit_i == 0` skips refilling slot `prefetch_distance`.

      Args:
        visit_i: The visit step the refill runs ahead of.
      """
      cond = visit_i + prefetch_distance < n_visit
      if prologue_full_prefetch:
        cond = jnp.logical_and(visit_i > 0, cond)

      @pl.when(cond)
      def _():
        refill_slot = lax.rem(visit_i + prefetch_distance, jnp.int32(nbuf))
        ahead = visit_sm[visit_i + prefetch_distance]
        weight_starts(ahead, refill_slot)

    def prologue():
      """Land the resident tables and start the first weight refills."""
      if has_w1_bias:
        sync(w1b_hbm, w1b_vm)
      if has_w2_bias:
        sync(w2b_hbm, w2b_vm)
      n_prologue = min(nbuf if prologue_full_prefetch else prefetch_distance, g_local)
      for b in range(n_prologue):

        @pl.when(jnp.int32(b) < n_visit)
        def _(b=b):
          weight_starts(visit_sm[b], b)

    def forward_staged_wave(w):
      """De-Dup dispatch: forward_staged_wave_body for `w`."""
      assert stage_tab_sm is not None
      assert fwd_tab_sm is not None
      assert fwd_cnt_sm is not None
      assert stage_sems is not None
      memory_ref.forward_staged_wave_body(
          w,
          me=me,
          stage_tab_sm=stage_tab_sm,
          fwd_tab_sm=fwd_tab_sm,
          fwd_cnt_sm=fwd_cnt_sm,
          stage_sems=stage_sems,
          stage_hbm=stage_hbm,
          lhs_hbm=lhs_hbm,
          wave_sems=wave_sems,
          fwd_send_sem=fwd_send_sem,
          num_slots=dedup_slots,
          num_waves=waves,
          num_boards=nb,
          fwd_max_runs=fwd_max_runs,
      )

    def forward_staged_through(last):
      """De-Dup dispatch: forward every wave up to `last` not yet forwarded."""
      assert fwd_done_sm is not None
      first = fwd_done_sm[0]

      def body(w, c):
        forward_staged_wave(w)
        return c

      with scope("comm_fwd_gateway"):
        lax.fori_loop(first, last + 1, body, jnp.int32(0))
      fwd_done_sm[0] = jnp.maximum(first, last + 1)

    def wait_wave(w):
      """Wait for all of wave ``w``'s rows."""
      rows = disp_tab_sm[3 * ep * waves + w]

      @pl.when(rows > 0)
      def _():
        memory_ref.rows_wait(wave_sems.at[w], lhs_hbm, rows)

    def wait_waves_through(last):
      """Wait every wave up to ``last`` not yet waited."""
      if tc_dedup:
        forward_staged_through(last)
      first = waves_done_sm[0]

      def body(w, c):
        wait_wave(w)
        return c

      with scope("comm_wait_dispatch"):
        lax.fori_loop(first, last + 1, body, jnp.int32(0))
      waves_done_sm[0] = jnp.maximum(first, last + 1)

    # Prologue: Load biases, start initial weight prefetches, sync the
    # dispatch table, execute the all-pairs barrier, and kick off the wave 0
    # gateway forwards.
    prologue()
    disp_copy = None
    if wait_for_exchange:
      if overlap_prologue_sync:
        disp_copy = pltpu.make_async_copy(disp_tab_hbm, disp_tab_sm, cp_sem)
        disp_copy.start()
      else:
        sync(disp_tab_hbm, disp_tab_sm)
      if not hoist_single_wave_wait:
        waves_done_sm[0] = jnp.int32(0)
    if tc_dedup:
      assert fwd_done_sm is not None
      fwd_done_sm[0] = jnp.int32(0)
    tg_to_smem = None
    if fused_tg:
      # After the first weight refills went out (prologue) and before the
      # barrier and wave 0's wait: the assembly runs under the weight stream.
      tg_to_smem = tg_fused.assemble(
          ep=ep,
          n_runs=ep * g_local,
          n_tokens=cfg.tg_n_tokens,
          runs_sm=tg_runs_sm,
          meta_sm=tg_meta_sm,
          land_hbm=tg_land_hbm,
          tg_sem=tg_sem,
          land_vm=tg_land_vm,
          tg_vm=tg_vm,
          scale_vm=tg_scale_vm,
          tg_sm=tg_sm,
          tg_sems=tg_sems,
          fuse_copy=fuse_tg_copy,
          wait_to_smem=not overlap_prologue_sync,
          fast_short_runs=overlap_prologue_sync,
      )
    with scope("sync_start_barrier"):
      memory_ref.all_pairs_barrier(ep)
    if disp_copy is not None:
      disp_copy.wait()
    if tg_to_smem is not None and overlap_prologue_sync:
      tg_to_smem.wait()
    if hoist_single_wave_wait:
      with scope("comm_wait_dispatch"):
        wait_wave(jnp.int32(0))
    if tc_dedup:
      # Kick wave 0's gateway forwards immediately after the barrier so wave 0's
      # forwarded rows are in flight before `prime_expert(visit_sm[0])`
      # waits on `wave_sems[0]`.
      forward_staged_through(jnp.int32(0))

    # ---- ragged tile machinery ----
    def gather_dma_coords(logical_base):
      """Aligned HBM base and in-window offset for a slab row."""
      alignment = jnp.int32(fused_host.TOKEN_GATHER_DMA_ALIGNMENT)
      shift = lax.rem(logical_base, alignment)
      dma_base = pl.multiple_of(
          logical_base - shift, fused_host.TOKEN_GATHER_DMA_ALIGNMENT
      )
      return dma_base, shift

    def issue_gather_window(logical_base, slot):
      """Start one fixed token-index window in SMEM slot ``slot``."""
      if fused_tg:
        return  # The whole token image is already in SMEM.
      dma_base, _ = gather_dma_coords(logical_base)
      if branchless_gather_window:
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm.at[
                pl.ds(slot * gather_window_rows, gather_window_rows)
            ],
            token_gather_sems.at[slot],
        ).start(priority=0)
        return

      @pl.when(slot == 0)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm0,
            token_gather_sem0,
        ).start(priority=0)

      @pl.when(slot == 1)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm1,
            token_gather_sem1,
        ).start(priority=0)

    def wait_gather_window(logical_base, slot):
      """Wait for the fixed token-index window in ``slot``."""
      if fused_tg:
        return
      dma_base, _ = gather_dma_coords(logical_base)
      if branchless_gather_window:
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm.at[
                pl.ds(slot * gather_window_rows, gather_window_rows)
            ],
            token_gather_sems.at[slot],
        ).wait()
        return

      @pl.when(slot == 0)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm0,
            token_gather_sem0,
        ).wait()

      @pl.when(slot == 1)
      def _():
        pltpu.make_async_copy(
            token_gather_hbm.at[pl.ds(dma_base, gather_window_rows)],
            token_gather_sm1,
            token_gather_sem1,
        ).wait()

    def lhs_issue_tile(row_base, live_blocks, slot, gather_slot, gather_offset):
      """Issue one tile's activations using a resident index window."""

      def issue_rows(token_of):
        """Start live_blocks * rowblk row DMAs; token_of(k) is row k's token.

        By default a fori_loop over the live blocks. cfg.lhs_issue_unroll
        (tuned path) issues them from a lax.switch over live_blocks whose
        branches are fully unrolled; the same DMAs either way.
        """
        if lhs_issue_unroll:

          def issue_n(nb):
            for k in range(nb * rowblk):
              pltpu.make_async_copy(
                  lhs_hbm.at[token_of(k)],
                  lhs_vm.at[slot, k],
                  lhs_sems.at[slot],
              ).start()

          lax.switch(
              jnp.clip(live_blocks, 0, tile_blocks),
              [functools.partial(issue_n, nb) for nb in range(tile_blocks + 1)],
          )
          return

        def issue_block(i, _):
          """Fetch rowblk token rows of block i."""
          for r in range(rowblk):
            pltpu.make_async_copy(
                lhs_hbm.at[token_of(i * rowblk + r)],
                lhs_vm.at[slot, i * rowblk + r],
                lhs_sems.at[slot],
            ).start()
          return _

        lax.fori_loop(0, live_blocks, issue_block, jnp.int32(0))

      if fused_tg:
        # Token rows by their slab rows.
        issue_rows(lambda k: tg_fused.token_at(tg_sm, row_base + k))
      elif branchless_gather_window:
        window_offset = gather_slot * gather_window_rows + gather_offset
        issue_rows(lambda k: token_gather_sm[window_offset + k])
      else:

        @pl.when(gather_slot == 0)
        def _():
          issue_rows(lambda k: token_gather_sm0[gather_offset + k])

        @pl.when(gather_slot == 1)
        def _():
          issue_rows(lambda k: token_gather_sm1[gather_offset + k])

      if has_ls_vm:
        pltpu.make_async_copy(
            ls_hbm.at[pl.ds(row_base // lanes, ls_window_rows)],
            ls_vm.at[slot],
            lhs_sems.at[slot],
        ).start()

    def lhs_ready_tile(live_rows, slot):
      """Wait the token rows and the scale window of one tile."""
      pltpu.make_async_copy(
          lhs_vm.at[slot, pl.ds(0, live_rows)],
          lhs_vm.at[slot, pl.ds(0, live_rows)],
          lhs_sems.at[slot],
      ).wait()
      if has_ls_vm:
        pltpu.make_async_copy(
            ls_vm.at[slot], ls_vm.at[slot], lhs_sems.at[slot]
        ).wait()

    def _mirror_base(e):
      """Sublane expert e's tiles start at in the contribution mirror."""
      return jnp.right_shift(expert_base_sm[e], tile_shift) + 2 * e

    def _dst_run(e, d):
      """The run index of (e, d) at its destination's arrival mirror.

      Direct: the global expert me * g_local + e. De-Dup Combine: at the
      board gateway the runs are q = m * (bs * g_local) + e_b, m = d's board
      (the token owner's) and e_b = (me % bs) * g_local + e (the expert on the
      gateway's board), so a gateway's mirror keeps the direct layout over its
      own run table.
      """
      if not cfg.dedup_combine:
        return me * g_local + e
      bs_i32 = jnp.int32(cfg.return_board_size)
      return (
          lax.div(jnp.int32(d), bs_i32) * (bs_i32 * g_local)
          + lax.rem(me, bs_i32) * g_local
          + e
      )

    def _run_tiles(e, d, region_blocks):
      """A run's mirror sublanes: (source, destination, count)."""
      start = commit_start_sm[e, d]
      lo = jnp.right_shift(start, tile_block_shift)
      hi = jnp.right_shift(start + region_blocks - 1, tile_block_shift)
      dst = jnp.right_shift(push_dst_sm[e, d], tile_block_shift) + 2 * (
          _dst_run(e, d)
      )
      return (
          (_mirror_base(e) + lo) * scale_sublanes,
          dst * scale_sublanes,
          (hi - lo + 1) * scale_sublanes,
      )

    def wait_commits(parity, live_blocks):
      """Wait one tile's commits on `parity`: data rows, then scales."""
      memory_ref.rows_wait(
          commit_sems.at[parity], contrib_hbm, live_blocks * rowblk
      )
      if not skip_scale_mirror:
        memory_ref.rows_wait(
            commit_scl_sems.at[parity], cscl_hbm, scale_sublanes
        )

    def wait_shipped(parity, rows, subs):
      """Wait until the wire has read `parity`'s staging slot."""

      @pl.when(rows > 0)
      def _():
        memory_ref.rows_wait(send_sems.at[parity], contrib_hbm, rows)

      if not skip_scale_mirror:

        @pl.when(subs > 0)
        def _():
          memory_ref.rows_wait(send_scl_sems.at[parity], cscl_hbm, subs)

    def prime_expert(e, lhs_slot, wslot, window_started, visit_i):
      """Prime expert ``e``'s metadata and first activation tile."""
      if early_weight_refill:
        # Before any blocking wait below (wave arrivals, the gather window),
        # so the HBM weight stream never drains while this shard waits for
        # the dispatch.
        weight_refill(visit_i)
      if wait_for_exchange and not hoist_single_wave_wait:
        last = jnp.int32(0)
        for w in range(waves - 1):
          last = last + (visit_i >= disp_tab_sm[wave_bounds_base + w]).astype(
              jnp.int32
          )
        if tc_dedup and cfg.fwd_lookahead:
          # Forward the next waves' staged rows now, ahead of this gateway's
          # own visit of those waves, so its peers' rows are not held behind
          # its compute. Forwarding wave w before waiting wave w still holds.
          forward_staged_through(
              jnp.minimum(last + cfg.fwd_lookahead, waves - 1)
          )
        wait_waves_through(last)
      rows = expert_rows_sm[e]
      slab_base = expert_base_sm[e]

      @pl.when(jnp.logical_not(window_started))
      def _():
        issue_gather_window(slab_base, wslot)

      wait_gather_window(slab_base, wslot)
      _, shift = gather_dma_coords(slab_base)
      lhs_issue_tile(
          slab_base, jnp.minimum(rows, tile_m) // rowblk, lhs_slot, wslot, shift
      )

      @pl.when(rows > gather_payload_rows)
      def _():
        issue_gather_window(
            slab_base + gather_payload_rows, jnp.int32(1) - wslot
        )

    def expert_tiles(e, visit_i, carry):
      """One expert step: a fori_loop over its [tile_m, H] tiles."""
      # Step 1: Resolve weight slot, wait for current expert's weights, and
      # issue weight prefetch for `visit_i + prefetch_distance`.
      slot = lax.rem(visit_i, jnp.int32(nbuf))
      if not prefetches:
        weight_starts(e, slot)
      weight_waits(e, slot)

      if prefetches and not early_weight_refill:
        weight_refill(visit_i)

      rows = expert_rows_sm[e]
      slab_base = expert_base_sm[e]
      n_tiles = -(-rows // tile_m)
      wslot = carry[-1]
      n_groups = -(-n_tiles // gather_tiles)
      next_wslot = lax.rem(wslot + n_groups, jnp.int32(2))

      def commit_tile(parity, t, tile_block_base, live_blocks):
        """Commit the tile's intersection with each (expert, dest) run."""

        def commit_dest(d):
          run_start = commit_start_sm[e, d]
          run_len = commit_len_sm[e, d]
          lo = jnp.maximum(run_start, tile_block_base)
          hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
          overlap = jnp.maximum(hi - lo, 0)
          lo = jnp.minimum(lo, hi)
          src_block = lo - tile_block_base
          dst_block = contrib_off_sm[e, d] + (lo - run_start)

          @pl.when(overlap > 0)
          def _(
              src_block=src_block,
              dst_block=dst_block,
              overlap=overlap,
              parity=parity,
          ):
            pltpu.make_async_copy(
                out_vm.at[parity, pl.ds(src_block * rowblk, overlap * rowblk)],
                contrib_hbm.at[pl.ds(dst_block * rowblk, overlap * rowblk)],
                commit_sems.at[parity],
            ).start()

        if loop_ep_push:
          lax.fori_loop(
              0, ep, lambda d, c: (commit_dest(d), c)[1], jnp.int32(0)
          )
        else:
          for d in range(ep):
            commit_dest(d)

        if not skip_scale_mirror:
          pltpu.make_async_copy(
              oscl_vm.at[parity],
              cscl_hbm.at[
                  pl.ds((_mirror_base(e) + t) * scale_sublanes, scale_sublanes)
              ],
              commit_scl_sems.at[parity],
          ).start()

      def ship_tile(parity, t, tile_block_base, live_blocks):
        """Ship the tile straight out of `out_vm`, no slab in between."""

        def ship_dest(d, acc, static_self=None):
          rows_out, subs_out = acc
          run_start = commit_start_sm[e, d]
          run_len = commit_len_sm[e, d]
          lo = jnp.maximum(run_start, tile_block_base)
          hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
          overlap = jnp.maximum(hi - lo, 0)
          lo = jnp.minimum(lo, hi)
          src_row = (lo - tile_block_base) * rowblk
          into_run = (lo - run_start) * rowblk
          dst_row = push_dst_sm[e, d] * rowblk + into_run
          true_here = jnp.clip(
              true_rows_sm[e, d] - into_run, 0, overlap * rowblk
          )
          if cfg.dedup_combine:
            # De-Dup Combine: push to this board's gateway for source d,
            # gw(d, my_b) = my_b * bs + d % bs, over intra-board links.
            bs_i32 = jnp.int32(cfg.return_board_size)
            d_local = lax.rem(jnp.int32(d), bs_i32)
            d_i32 = lax.div(me, bs_i32) * bs_i32 + d_local
            is_me = d_local == lax.rem(me, bs_i32)
            static_self = None
          else:
            d_i32 = jnp.int32(d)
            is_me = d_i32 == me
          dst_sub = (
              jnp.right_shift(push_dst_sm[e, d], tile_block_shift)
              + 2 * _dst_run(e, d)
              + t
              - jnp.right_shift(run_start, tile_block_shift)
          ) * scale_sublanes

          if static_self is not True:
            rem_cond = (
                true_here > 0
                if static_self is False
                else memory_ref.and_nonempty(jnp.logical_not(is_me), true_here)
            )

            @pl.when(rem_cond)
            def _(
                src_row=src_row,
                dst_row=dst_row,
                true_here=true_here,
                parity=parity,
                d_i32=d_i32,
            ):
              pltpu.make_async_remote_copy(
                  src_ref=out_vm.at[parity, pl.ds(src_row, true_here)],
                  dst_ref=recv_hbm.at[pl.ds(dst_row, true_here)],
                  send_sem=send_sems.at[parity],
                  recv_sem=recv_sem,
                  device_id=(d_i32,),
              ).start()

          if static_self is not False:
            me_cond = (
                overlap > 0
                if static_self is True
                else memory_ref.and_nonempty(is_me, overlap)
            )

            @pl.when(me_cond)
            def _(
                src_row=src_row, dst_row=dst_row, overlap=overlap, parity=parity
            ):
              pltpu.make_async_copy(
                  out_vm.at[parity, pl.ds(src_row, overlap * rowblk)],
                  recv_hbm.at[pl.ds(dst_row, overlap * rowblk)],
                  send_sems.at[parity],
              ).start()

          if not skip_scale_mirror:
            if static_self is not True:
              rem_scl_cond = (
                  overlap > 0
                  if static_self is False
                  else memory_ref.and_nonempty(jnp.logical_not(is_me), overlap)
              )

              @pl.when(rem_scl_cond)
              def _(dst_sub=dst_sub, parity=parity, d_i32=d_i32):
                pltpu.make_async_remote_copy(
                    src_ref=oscl_vm.at[parity],
                    dst_ref=rscl_hbm.at[pl.ds(dst_sub, scale_sublanes)],
                    send_sem=send_scl_sems.at[parity],
                    recv_sem=recv_scl_sem,
                    device_id=(d_i32,),
                ).start()

            if static_self is not False:
              me_scl_cond = (
                  overlap > 0
                  if static_self is True
                  else memory_ref.and_nonempty(is_me, overlap)
              )

              @pl.when(me_scl_cond)
              def _(dst_sub=dst_sub, parity=parity):
                pltpu.make_async_copy(
                    oscl_vm.at[parity],
                    rscl_hbm.at[pl.ds(dst_sub, scale_sublanes)],
                    send_scl_sems.at[parity],
                ).start()

          if static_self is False:
            rows_out = rows_out + true_here
          elif static_self is True:
            rows_out = rows_out + overlap * rowblk
          else:
            rows_out = rows_out + jnp.where(is_me, overlap * rowblk, true_here)
          if not skip_scale_mirror:
            subs_out = subs_out + jnp.where(overlap > 0, scale_sublanes, 0)
          return rows_out, subs_out

        acc = (jnp.int32(0), jnp.int32(0))
        if loop_ep_push:
          return lax.fori_loop(0, ep, ship_dest, acc)
        dests = push_dests()
        if (
            prime_before_ship
            and not cfg.dedup_combine
            and cfg.return_order in ("rot", "board_stripe")
        ):
          for d in dests[:-1]:
            acc = ship_dest(d, acc, static_self=False)
          return ship_dest(dests[-1], acc, static_self=True)
        for d in dests:
          acc = ship_dest(d, acc)
        return acc

      def tile_body(t, carried):
        """Compute tile t, stage it for the wire and ship it."""
        tile_count = carried[0]
        row_base = slab_base + t * tile_m
        live_rows = jnp.minimum(rows - t * tile_m, tile_m)
        live_blocks = live_rows // rowblk
        tile_block_base = t * tile_blocks
        parity = lax.rem(tile_count, jnp.int32(out_parities))

        # Step 1: Guard `out_vm[parity]` reuse and issue progressive pushes.
        if push_from_vmem:
          wait_shipped(
              parity,
              jnp.where(parity == 0, carried[1], carried[2]),
              jnp.where(parity == 0, carried[3], carried[4]),
          )
        else:
          pending = jnp.where(parity == 0, carried[1], carried[2])

          @pl.when(pending > 0)
          def _():
            wait_commits(parity, pending)

          if progressive_push and not defer_remote_push and pp_lag == 1:
            # Push the runs ending in tile t - 1 now, after
            # waiting that tile's commits (the other parity), rather than
            # those ending in t - 2.
            prev_pending = jnp.where(parity == 0, carried[2], carried[1])

            @pl.when(t >= 1)
            def _():
              @pl.when(prev_pending > 0)
              def _():
                wait_commits(jnp.int32(1) - parity, prev_pending)

              push_runs_ending_in(e, t - 1)

          elif progressive_push and not defer_remote_push:

            @pl.when(t >= 2)
            def _():
              push_runs_ending_in(e, t - 2)

        # Step 2: Wait for current tile's LHS rows/scales and prefetch tile t+1.
        lhs_ready_tile(live_rows, parity)
        other_parity = jnp.int32(1) - parity

        def issue_next_tile():
          """Issue tile t+1's activations into the other slot."""
          next_t = t + 1
          next_group = next_t // gather_tiles
          next_within = lax.rem(next_t, jnp.int32(gather_tiles))
          next_group_slot = lax.rem(wslot + next_group, jnp.int32(2))
          next_group_base = slab_base + next_group * gather_payload_rows
          _, next_shift = gather_dma_coords(next_group_base)
          next_rows = jnp.minimum(rows - next_t * tile_m, tile_m)
          lhs_issue_tile(
              slab_base + next_t * tile_m,
              next_rows // rowblk,
              other_parity,
              next_group_slot,
              next_shift + next_within * tile_m,
          )

        @pl.when(t + 1 < n_tiles)
        def _():
          next_t = t + 1
          next_group = next_t // gather_tiles
          next_within = lax.rem(next_t, jnp.int32(gather_tiles))
          next_group_slot = lax.rem(wslot + next_group, jnp.int32(2))
          next_group_base = slab_base + next_group * gather_payload_rows

          @pl.when(next_within == 0)
          def _():
            wait_gather_window(next_group_base, next_group_slot)
            following = next_group + 1

            @pl.when(following * gather_tiles < n_tiles)
            def _():
              issue_gather_window(
                  slab_base + following * gather_payload_rows,
                  jnp.int32(1) - next_group_slot,
              )

          issue_next_tile()

        @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
        def _():
          issue_gather_window(expert_base_sm[visit_sm[visit_i + 1]], next_wslot)

        # Step 3: Run stateless expert FFN compute at the active tail-height
        # rung.
        def compute_tile(height):
          """Run the FFN on the tile's leading `height` rows."""
          full = height == tile_m
          act_rows = (
              lhs_vm[parity] if full else lhs_vm[parity, pl.ds(0, height)]
          ).reshape(height, hidden)
          ls_col = None
          if has_ls_vm:
            ls_col = fused_kernel._tile_row_scales(  # pylint: disable=protected-access
                ls_vm[parity], row_base, height
            )
          act_scales = ls_col if has_act_scale else None
          w1b_row = w1b_vm[pl.ds(e, 1), :] if has_w1_bias else None
          if chunked_epilogue:
            w1s_row = w1s_vm[pl.ds(slot, 1), :]
            acc1 = lax.dot_general(
                act_rows,
                w1_vm[slot],
                fused_kernel.DN,
                preferred_element_type=jnp.float32,
            )
            mid_q, mid_scale = fused_kernel._intermediate_rows(  # pylint: disable=protected-access
                acc1,
                act_scales,
                cfg.inter,
                w1s=w1s_row,
                act_fn=act_fn,
                w1b=w1b_row,
            )
            if split_weight_waits:

              @pl.when(t == 0)
              def _():
                weight_waits_w2(e, slot)

            if full:
              mid_q_vm[...] = mid_q
            else:
              mid_q_vm[pl.ds(0, height)] = mid_q
            chunk_lb = plan_lib.DOWN_CHUNK_LANE_BLOCKS
            sub_lb = plan_lib.DOWN_SUBCHUNK_LANE_BLOCKS
            sub_n = sub_lb * lanes
            n_sub = chunk_lb // sub_lb
            n_chunks = lane_blocks // chunk_lb
            amax = None
            for c in range(n_chunks):
              # Compute 2D sub-chunks (N=512 -> 128 f32 VREGS) and concatenate
              # in 2D BEFORE reshaping to 3D (height, 16, 128) so concat has
              # zero sublane vrot and the 3D store lowers to ss:$8 scatter.
              down_parts_2d = []
              for sub in range(n_sub):
                n0 = (c * n_sub + sub) * sub_n
                mid_q_in = mid_q_vm[...] if full else mid_q_vm[pl.ds(0, height)]
                acc2_s = lax.dot_general(
                    mid_q_in,
                    w2_vm[slot, :, pl.ds(n0, sub_n)],
                    fused_kernel.DN,
                    preferred_element_type=jnp.float32,
                )
                down_s = acc2_s if mid_scale is None else acc2_s * mid_scale
                if has_scales:
                  down_s = down_s * w2s_vm[pl.ds(slot, 1), pl.ds(n0, sub_n)]
                if has_w2_bias:
                  down_s = down_s + w2b_vm[pl.ds(e, 1), pl.ds(n0, sub_n)]
                down_bf16_s = down_s.astype(jnp.bfloat16)
                chunk_amax = jnp.max(
                    jnp.abs(down_bf16_s), axis=-1, keepdims=True
                )
                amax = (
                    chunk_amax
                    if amax is None
                    else jnp.maximum(amax, chunk_amax)
                )
                down_parts_2d.append(down_bf16_s)
              down_bf16_c = (
                  down_parts_2d[0]
                  if n_sub == 1
                  else jnp.concatenate(down_parts_2d, axis=1)
              )
              down_3d_c = down_bf16_c.reshape(height, chunk_lb, lanes)
              if full:
                down_vm[c] = down_3d_c
              else:
                down_vm[c, pl.ds(0, height)] = down_3d_c
            wire_scales, wire_sinv, _ = fused_kernel._row_scale(amax)  # pylint: disable=protected-access
            sinv_f32 = jnp.broadcast_to(
                wire_sinv.astype(jnp.bfloat16).astype(jnp.float32),
                (height, lanes),
            )
            if full:
              sinv_vm[...] = sinv_f32
            else:
              sinv_vm[pl.ds(0, height)] = sinv_f32
            row_step = 16
            n_row_steps = height // row_step
            use_halves = lane_blocks % 32 == 0
            n_out_chunks = lane_blocks // 32 if use_halves else 1

            def store_chunked_parity(static_parity):
              if full:
                oscl_vm[static_parity] = wire_scales.reshape(
                    scale_sublanes, scale_cols
                )
              elif scale_sublanes == 1 or height <= lanes:
                oscl_vm[static_parity, 0:1, pl.ds(0, height)] = (
                    wire_scales.reshape(1, height)
                )
              else:
                oscl_vm[static_parity, pl.ds(0, height // lanes), :] = (
                    wire_scales.reshape(height // lanes, lanes)
                )

              def row_body(rb, _):
                r0 = rb * row_step
                s_blk = sinv_vm[pl.ds(r0, row_step)]
                s_8 = lax.broadcast_in_dim(
                    s_blk, (row_step, 8, lanes), (0, 2)
                )
                s_16 = jnp.concatenate([s_8, s_8], axis=1).astype(arg.BF16)
                if use_halves:
                  mask_u32 = jnp.where(
                      s_8 > 0.0, jnp.uint32(0xFFFFFFFF), jnp.uint32(0)
                  )
                  for h_i in range(n_out_chunks):
                    d0 = down_vm[2 * h_i, pl.ds(r0, row_step)]
                    d1 = down_vm[2 * h_i + 1, pl.ds(r0, row_step)]
                    q_raw = jnp.concatenate(
                        [d0 * s_16, d1 * s_16], axis=1
                    ).astype(arg.FP8)
                    q_u32 = pltpu.bitcast(q_raw, jnp.uint32) & mask_u32
                    out_vm[
                        static_parity,
                        pl.ds(r0, row_step),
                        pl.ds(32 * h_i, 32),
                        :,
                    ] = pltpu.bitcast(q_u32, arg.FP8)
                else:
                  s_3d = (
                      s_16
                      if n_chunks == 1
                      else jnp.concatenate([s_16] * n_chunks, axis=1)
                  )
                  usable_3d = s_3d > 0
                  d_all = (
                      down_vm[0, pl.ds(r0, row_step)]
                      if n_chunks == 1
                      else jnp.concatenate(
                          [
                              down_vm[c, pl.ds(r0, row_step)]
                              for c in range(n_chunks)
                          ],
                          axis=1,
                      )
                  )
                  q_all = jnp.where(
                      usable_3d, d_all * s_3d, jnp.zeros_like(d_all)
                  ).astype(arg.FP8)
                  out_vm[static_parity, pl.ds(r0, row_step)] = q_all
                return jnp.int32(0)

              lax.fori_loop(0, n_row_steps, row_body, jnp.int32(0))

            @pl.when(parity == 0)
            def _():
              store_chunked_parity(0)

            @pl.when(parity == 1)
            def _():
              store_chunked_parity(1)

            return
          if weight_format == arg.WeightFormat.BF16:
            acc2, mid_scale = fused_kernel.expert_ffn_bf16(
                act_rows,
                w1_vm[slot],
                w2_vm[slot],
                act_fn=act_fn,
                w1b=w1b_row,
            )
          elif split_weight_waits:
            w1s_row = w1s_vm[pl.ds(slot, 1), :]
            acc1 = lax.dot_general(
                act_rows,
                w1_vm[slot],
                fused_kernel.DN,
                preferred_element_type=jnp.float32,
            )
            mid_q, mid_scale = fused_kernel._intermediate_rows(  # pylint: disable=protected-access
                acc1,
                act_scales,
                cfg.inter,
                w1s=w1s_row,
                act_fn=act_fn,
                w1b=w1b_row,
            )

            @pl.when(t == 0)
            def _():
              weight_waits_w2(e, slot)

            acc2 = lax.dot_general(
                mid_q,
                w2_vm[slot],
                fused_kernel.DN,
                preferred_element_type=jnp.float32,
            )
          else:
            acc2, mid_scale = fused_kernel.expert_ffn_fp8(
                act_rows,
                act_scales,
                w1_vm[slot],
                w2_vm[slot],
                w1s_vm[pl.ds(slot, 1), :],
                act_fn=act_fn,
                w1b=w1b_row,
            )
          wire_rows, wire_scales = fused_kernel.wire_row_and_scale(
              acc2,
              mid_scale,
              w2s=w2s_vm[pl.ds(slot, 1), :] if has_scales else None,
              w2b=w2b_vm[pl.ds(e, 1), :] if has_w2_bias else None,
              wire_dtype=cfg.wire_dtype or form.wire_dtype,
              tile_m=height,
          )
          wire_rows_staged = wire_rows.reshape(height, lane_blocks, lanes)

          def store_parity(static_parity):
            """Stage the tile's rows and scales in one out_vm slot."""
            if full:
              out_vm[static_parity] = wire_rows_staged
              oscl_vm[static_parity] = wire_scales.reshape(
                  scale_sublanes, scale_cols
              )
            else:
              out_vm[static_parity, pl.ds(0, height)] = wire_rows_staged
              if scale_sublanes == 1 or height <= lanes:
                oscl_vm[static_parity, 0:1, pl.ds(0, height)] = (
                    wire_scales.reshape(1, height)
                )
              else:
                oscl_vm[static_parity, pl.ds(0, height // lanes), :] = (
                    wire_scales.reshape(height // lanes, lanes)
                )

          @pl.when(parity == 0)
          def _():
            store_parity(0)

          @pl.when(parity == 1)
          def _():
            store_parity(1)

        if len(height_rungs) == 1:
          compute_tile(tile_m)
        else:
          for rung_i, rung in enumerate(height_rungs):
            below = height_rungs[rung_i - 1] if rung_i else 0

            @pl.when(jnp.logical_and(live_rows > below, live_rows <= rung))
            def _(rung=rung):
              compute_tile(rung)

        # Step 4: Ship the output tile directly from VMEM or commit it to
        # `contrib_hbm`. When `defer_remote_push` is enabled, remote DMAs are
        # issued after tile t+1's local loads and tile t's compute (and for
        # `push_from_vmem`, after priming the next visited expert on the last
        # tile) so the remote-DMA global memory barrier does not stall local
        # loads or compute.
        if push_from_vmem:
          if defer_remote_push or prime_before_ship:

            @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
            def _():
              nxt = visit_sm[visit_i + 1]
              prime_expert(
                  nxt,
                  other_parity,
                  next_wslot,
                  jnp.bool_(True),
                  visit_i=visit_i + 1,
              )

          rows_out, subs_out = ship_tile(
              parity, t, tile_block_base, live_blocks
          )
          return (
              tile_count + 1,
              jnp.where(parity == 0, rows_out, carried[1]),
              jnp.where(parity == 1, rows_out, carried[2]),
              jnp.where(parity == 0, subs_out, carried[3]),
              jnp.where(parity == 1, subs_out, carried[4]),
              wslot,
          )

        if progressive_push and defer_remote_push:

          @pl.when(t >= 2)
          def _():
            push_runs_ending_in(e, t - 2)

        commit_tile(parity, t, tile_block_base, live_blocks)

        if progressive_push and not defer_remote_push and pp_lag == 1:
          # Tile t - 1's commits were waited in step 1 (t >= 1).
          keep_other = jnp.where(
              t >= 1,
              jnp.int32(0),
              jnp.where(parity == 0, carried[2], carried[1]),
          )
          return (
              tile_count + 1,
              jnp.where(parity == 0, live_blocks, keep_other),
              jnp.where(parity == 1, live_blocks, keep_other),
              wslot,
          )
        return (
            tile_count + 1,
            jnp.where(parity == 0, live_blocks, carried[1]),
            jnp.where(parity == 1, live_blocks, carried[2]),
            wslot,
        )

      carry = lax.fori_loop(0, n_tiles, tile_body, carry)

      # Cross-expert lookahead: prime the next visited expert's first
      # activation tile before this expert's commit drain and push (when
      # `push_from_vmem` and `defer_remote_push` are both active, this already
      # ran inside the last tile before `ship_tile`).
      if not (push_from_vmem and (defer_remote_push or prime_before_ship)):
        tiles_done = carry[0]

        @pl.when(visit_i + 1 < n_visit)
        def _():
          nxt = visit_sm[visit_i + 1]
          prime_expert(
              nxt,
              lax.rem(tiles_done, jnp.int32(out_parities)),
              next_wslot,
              jnp.bool_(True),
              visit_i=visit_i + 1,
          )

      return carry[:-1] + (next_wslot,)

    def drain_commits(carried):
      """Drain both parities' pending commits, then zero the counts."""
      tile_count, pending0, pending1, wslot = carried

      @pl.when(pending0 > 0)
      def _():
        wait_commits(0, pending0)

      @pl.when(pending1 > 0)
      def _():
        wait_commits(1, pending1)

      return (tile_count, jnp.int32(0), jnp.int32(0), wslot)

    def drain_shipped(carried):
      """Wait out whatever the last tile on each parity put on the wire."""
      for p in range(out_parities):
        wait_shipped(p, carried[1 + p], carried[3 + p])

    @pl.when(n_visit > 0)
    def _():
      prime_expert(
          visit_sm[0],
          jnp.int32(0),
          jnp.int32(0),
          jnp.bool_(False),
          visit_i=jnp.int32(0),
      )

    def run_last_tile(e, d):
      """The expert-local tile holding run (e, d)'s last block."""
      return jnp.right_shift(
          commit_start_sm[e, d] + commit_len_sm[e, d] - 1, tile_block_shift
      )

    def push_run(e, d, cond=None):
      """Push run (e, d): its remote rows and scale mirror, or the own-dest hop."""
      region_blocks = commit_len_sm[e, d]
      src_block = contrib_off_sm[e, d]
      dst_block = push_dst_sm[e, d]
      if cfg.dedup_combine:
        # De-Dup Combine: push to this board's gateway for source d,
        # gw(d, my_b) = my_b * bs + d % bs, over intra-board links.
        bs_i32 = jnp.int32(cfg.return_board_size)
        d_local = lax.rem(jnp.int32(d), bs_i32)
        d_i32 = lax.div(me, bs_i32) * bs_i32 + d_local
        is_me = d_local == lax.rem(me, bs_i32)
      else:
        d_i32 = jnp.int32(d)
        is_me = d_i32 == me
      rem_pred = jnp.logical_not(is_me)
      me_pred = is_me
      if cond is not None:
        rem_pred = jnp.logical_and(cond, rem_pred)
        me_pred = jnp.logical_and(cond, me_pred)

      @pl.when(memory_ref.and_nonempty(rem_pred, region_blocks))
      def _():
        true_rows = true_rows_sm[e, d]
        # `region_blocks > 0` iff `true_rows > 0` (`region_rows` is `true_rows`
        # rounded up to `rowblk`), so no nested `pl.when(true_rows > 0)` is
        # needed.
        pltpu.make_async_remote_copy(
            src_ref=contrib_hbm.at[pl.ds(src_block * rowblk, true_rows)],
            dst_ref=recv_hbm.at[pl.ds(dst_block * rowblk, true_rows)],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(d_i32,),
        ).start()
        if skip_scale_mirror:
          return
        src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
        pltpu.make_async_remote_copy(
            src_ref=cscl_hbm.at[pl.ds(src_sub, n_sub)],
            dst_ref=rscl_hbm.at[pl.ds(dst_sub, n_sub)],
            send_sem=send_scl_sem,
            recv_sem=recv_scl_sem,
            device_id=(d_i32,),
        ).start()

      @pl.when(memory_ref.and_nonempty(me_pred, region_blocks))
      def _():
        pltpu.make_async_copy(
            contrib_hbm.at[pl.ds(src_block * rowblk, region_blocks * rowblk)],
            recv_hbm.at[pl.ds(dst_block * rowblk, region_blocks * rowblk)],
            mehop_sem,
        ).start()
        if skip_scale_mirror:
          return
        src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
        pltpu.make_async_copy(
            cscl_hbm.at[pl.ds(src_sub, n_sub)],
            rscl_hbm.at[pl.ds(dst_sub, n_sub)],
            mehop_scl_sem,
        ).start()

    def push_dests():
      """The return push destinations, in cfg.return_order."""
      if cfg.return_order == "ring":
        return list(range(ep))
      if cfg.return_order == "rot":
        return [lax.rem(me + i, jnp.int32(ep)) for i in range(1, ep + 1)]
      bs = cfg.return_board_size
      n_boards = ep // bs
      my_board = lax.div(me, jnp.int32(bs))
      my_local = lax.rem(me, jnp.int32(bs))
      dests = []
      for k in range(bs):
        for b in list(range(1, n_boards)) + [0]:
          if b == 0 and k == 0:
            continue
          board = lax.rem(my_board + b, jnp.int32(n_boards))
          dests.append(board * bs + lax.rem(my_local + k, jnp.int32(bs)))
      return dests + [me]

    def push_runs_ending_in(e, tile):
      """Progressive push: expert e's runs whose last tile is `tile`."""
      if loop_ep_push:
        lax.fori_loop(
            0,
            ep,
            lambda d, c: (push_run(e, d, cond=run_last_tile(e, d) == tile), c)[
                1
            ],
            jnp.int32(0),
        )
      else:
        for d in push_dests():
          push_run(e, d, cond=run_last_tile(e, d) == tile)

    def push_expert(e, first_tile=None):
      """Push expert e's remote regions and hop its own-dest region."""
      if loop_ep_push:
        lax.fori_loop(
            0,
            ep,
            lambda d, c: (
                push_run(
                    e,
                    d,
                    cond=(
                        None
                        if first_tile is None
                        else run_last_tile(e, d) >= first_tile
                    ),
                ),
                c,
            )[1],
            jnp.int32(0),
        )
      else:
        for d in push_dests():
          cond = (
              None if first_tile is None else run_last_tile(e, d) >= first_tile
          )
          push_run(e, d, cond=cond)

    # Body: Main visit loop over active local experts.
    def visit_step(visit_i, carry):
      """One step of the visit list: compute, then drain and push."""
      e = visit_sm[visit_i]
      with scope("compute_expert"):
        carry = expert_tiles(e, visit_i, carry)
      if push_from_vmem:
        return carry
      with scope("comm_push_return"):
        carry = drain_commits(carry)
        if sc_ret:
          pl.semaphore_signal(
              sc_ret_sems.at[visit_i],
              inc=1,
              device_id={"cae_core": 0, "cae_sub": sc_ret_sub},
          )
        elif progressive_push:
          n_tiles = -(-expert_rows_sm[e] // tile_m)
          push_expert(
              e, first_tile=n_tiles - (2 if defer_remote_push else pp_lag)
          )
        else:
          push_expert(e)
      return carry

    carry0 = tuple(jnp.int32(0) for _ in range(6 if push_from_vmem else 4))
    carry_end = lax.fori_loop(0, n_visit, visit_step, carry0)

    # Epilogue: Drain remaining wave waits, gateway forward sends, return
    # pushes, and synchronize all shards before returning.
    def drain_transport():
      """Consume the pushes and the commits the per-tile waits left."""
      if sc_ret:
        if cfg.dedup_combine:
          bs = cfg.return_board_size
          my_local = lax.rem(me, jnp.int32(bs))
          self_rows = jnp.int32(0)
          for m in range(ep // bs):
            s_m = m * bs + my_local
            for g in range(g_local):
              self_rows = self_rows + jnp.where(
                  commit_len_sm[g, s_m] > 0, true_rows_sm[g, s_m], 0
              )
        else:
          self_rows = jnp.int32(0)
          for g in range(g_local):
            self_rows = self_rows + jnp.where(
                commit_len_sm[g, me] > 0, true_rows_sm[g, me], 0
            )
        memory_ref.rows_wait(
            ret_recv_sem,
            recv_hbm,
            count(fused_host.COUNT_RECV_ROWS) + self_rows,
        )
        return
      if push_from_vmem:
        drain_shipped(carry_end)
        memory_ref.rows_wait(
            recv_sem, recv_hbm, count(fused_host.COUNT_RECV_ROWS)
        )
        if not skip_scale_mirror:
          memory_ref.rows_wait(
              recv_scl_sem,
              rscl_hbm,
              count(fused_host.COUNT_RECV_MIRROR) * scale_sublanes,
          )
        return
      memory_ref.rows_wait(
          send_sem, contrib_hbm, count(fused_host.COUNT_SEND_ROWS)
      )
      memory_ref.rows_wait(
          recv_sem, recv_hbm, count(fused_host.COUNT_RECV_ROWS)
      )
      if not skip_scale_mirror:
        memory_ref.rows_wait(
            send_scl_sem,
            cscl_hbm,
            count(fused_host.COUNT_SEND_MIRROR) * scale_sublanes,
        )
        memory_ref.rows_wait(
            recv_scl_sem,
            rscl_hbm,
            count(fused_host.COUNT_RECV_MIRROR) * scale_sublanes,
        )
      if cfg.dedup_combine:
        bs = cfg.return_board_size
        my_local = lax.rem(me, jnp.int32(bs))
        self_blocks = jnp.int32(0)
        for m in range(ep // bs):
          s_m = m * bs + my_local
          for g in range(g_local):
            self_blocks = self_blocks + commit_len_sm[g, s_m]
      else:
        self_blocks = commit_len_sm[0, me]
        for g in range(1, g_local):
          self_blocks = self_blocks + commit_len_sm[g, me]
      memory_ref.rows_wait(mehop_sem, recv_hbm, self_blocks * rowblk)
      if not skip_scale_mirror:
        memory_ref.rows_wait(
            mehop_scl_sem,
            rscl_hbm,
            count(fused_host.COUNT_SELF_MIRROR) * scale_sublanes,
        )

    if wait_for_exchange and not hoist_single_wave_wait:
      wait_waves_through(jnp.int32(waves - 1))
    if tc_dedup:
      fwd_row = (2 * nb + dedup_slots) * waves
      fwd_remote_total = stage_tab_sm[fwd_row]
      for w in range(1, waves):
        fwd_remote_total = fwd_remote_total + stage_tab_sm[fwd_row + w]

      @pl.when(fwd_remote_total > 0)
      def _():
        with scope("comm_fwd_send_drain"):
          memory_ref.rows_wait_send(fwd_send_sem, lhs_hbm, fwd_remote_total)

    with scope("comm_return_drain"):
      drain_transport()
    if epilogue_barrier:
      with scope("sync_end_barrier"):
        memory_ref.all_pairs_barrier(ep)

  return kernel
