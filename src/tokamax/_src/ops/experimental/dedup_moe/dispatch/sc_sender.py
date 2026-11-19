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
"""SparseCore (CAE) wave dispatch for the De-Dup MoE layer.

The wave dispatch (wave_dispatch.py) issues its token-row DMAs from a
TensorCore (TC) Pallas "start" kernel (tc_sender.ragged_a2a_start_waves):
per wave and destination a head (chunk_rows rows) and a tail DMA from the
packed send buffer into the destination's receive slot, each landing on the
destination's wave semaphore recv_sems[w], which the fused expert kernel waits
on (byte counts). This module moves those row DMAs to the chip's SparseCore
(on TPU8i the SparseCore runs the CAE, the Collectives Acceleration Engine),
split over `num_subcores` vector subcores:

  1. TC        list kernel: all-pairs barrier, the token-gather list DMAs
               (small), and allocation of the TC semaphores recv_sems[W] and
               tg_sem (and the De-Dup stage semaphores) that the TC consumers
               wait on
  2. SC        send kernel: every (wave, destination) head / tail run of the
               send rows -> recv on the destination, signalling the
               DESTINATION'S TC semaphore recv_sems[w] (device_id={axis: d,
               tc: 0}), then the De-Dup stage runs to the remote gateways;
               runs after (1) by data dependency, so after the barrier. Two
               forms:
                 contiguous  XLA (TC) packs send = rows[perm] first, and the
                             SC copies each run in one DMA;
                 gather      no TC pack: the SC copies each send row straight
                             from the layer's rows, one DMA per row, reading
                             perm through an SMEM window
  3. TC        unchanged: routing tables, tg_assemble (waits tg_sem), the
               fused expert kernel (waits recv_sems[w] per wave, FFN, result
               push), combine
  4. SC / TC   drains: an SC kernel waits on each subcore's SC send semaphore,
               a TC kernel on the list send semaphore (both thread the fused
               kernel's arrivals through, as ragged_a2a_send_done does)

With `sc_forward` (De-Dup only) the gateway K4 forwards also move to the
SparseCore: the stage runs count on the gateway SparseCore's semaphores, and
one SC kernel (sc_forward_start) running beside the fused expert kernel
forwards each stage run as soon as it lands, then drains its forwards and
the SC sends (so the drain in (4) is TC only); the fused kernel is built
without its forwards (dedup_sc_forward).

The swap is done by `use_sc_dispatch()`, which replaces
tc_sender.ragged_a2a_start_waves / ragged_a2a_send_done while the
layer is traced (wave_dispatch calls them through the module), so
layer/monolithic.py, wave_dispatch.py and fused_kernel/kernel.py are unchanged.
wave_dispatch.start / finish(sc_sender=True) (DEDUP_MOE_SC_SENDER=1) call the
same start and drain directly.

Requirements found by the SparseCore gating benchmarks: only VectorSubcoreMesh
SC kernels compile on TPU8I (ScalarSubcoreMesh: "Found custom kernel targeting
kScScalarSubcore on a unsupported platform"); one vector subcore drives remote
DMAs at the TC's rate; both TensorCores' SC programs of a chip run concurrently.
"""

import contextlib
import functools
import importlib
import inspect
import sys
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dispatch import tc_sender as dispatch_kernels

SC_CORE = "cae_core"
SC_SUB = "cae_sub"
TC_AXIS = "cae_tc"
LIST_COLLECTIVE_ID = dispatch_kernels.A2A_COLLECTIVE_ID
SMEM_PAD = 128  # int32 words: SMEM tables are padded to whole 512 B rows.
PERM_BUF_WORDS = SMEM_PAD  # 128 words (512 B) aligned window
# Vector subcores the SC sends are split over (the measured configuration).
NUM_SUBCORES = 4
# fwd_tab words per SMEM window of the SC forward kernel (128 runs).
FWD_WIN_WORDS = 4 * SMEM_PAD
MAX_SC_RET_VISITS = 16
MAX_SC_COMB_STEPS = 64

# XLA options the SC -> TC semaphore DMAs need (jax
# tests/pallas/tpu_pallas_mpmd_test.py MpmdAsyncTest; xla_tpu_use_tc_device_
# shape_on_sc keeps the SC kernel's HBM operands in the TC layouts, so XLA
# inserts no relayout copy between the SC writes and the TC readers).
COMPILER_OPTIONS = {
    "xla_mosaic_unsafe_allow_multicore_remote_dma": "true",
    "xla_tpu_use_tc_device_shape_on_sc": "true",
}



def _plsc() -> Any:
  """`jax.experimental.pallas.tpu_sc`, imported on first use.

  The import is deferred so this module imports (and the TensorCore dispatch
  runs) on a JAX without Pallas SparseCore.

  Returns:
    The tpu_sc module.
  """
  return importlib.import_module("jax.experimental.pallas.tpu_sc")


def _ensure_sc_compat() -> None:
  """Applies the JAX compatibility patches the SC kernels need (idempotent)."""
  from tokamax._src.ops.experimental.dedup_moe.dispatch import sc_compat  # pylint: disable=g-import-not-at-top

  sc_compat.ensure_sc_compat()


_TC_SC_MESH_PATCHED = False
_TC_CROSS_CORE_SEM_PATCHED = False
_MPMD_ARRAY_WRITES_PATCHED = False


def allow_tc_sc_mesh_pair() -> None:
  """Idempotently registers `(TensorCoreMesh, VectorSubcoreMesh)` in `mpmd._VALID_MESH_PAIRS`."""
  global _TC_SC_MESH_PATCHED
  if _TC_SC_MESH_PATCHED:
    return
  mpmd_mod: Any = getattr(pl, "mpmd", None)
  valid_pairs = getattr(mpmd_mod, "_VALID_MESH_PAIRS", None)
  pair = (pltpu.TensorCoreMesh, _plsc().VectorSubcoreMesh)
  if (
      mpmd_mod is not None
      and valid_pairs is not None
      and pair not in valid_pairs
  ):
    mpmd_mod._VALID_MESH_PAIRS = (*valid_pairs, pair)  # pylint: disable=protected-access
  _TC_SC_MESH_PATCHED = True


def allow_tc_cross_core_semaphores() -> None:
  """Idempotently allows `pl.pallas_call` on TensorCore to accept `@ VectorSubcoreMesh` semaphores.

  `pl.pallas_call` lowers at `jax.jit` compile time (outside the tracing
  context), where `lowering.lower_jaxpr_into_unpipelined_module`,
  `_check_block_mappings`, and `lower_jaxpr_to_pipelined_module` call
  `tpu_core.memory_space_to_tpu_memory_space(ms)` without `core_type`. When `ms`
  is `MemorySpace.SEMAPHORE @ VectorSubcoreMesh`,
  `memory_space_to_tpu_memory_space` checks `ms.core_type != CoreType.TC` and
  raises unless converted to `MemorySpace.SEMAPHORE` at those validation sites.
  The inner `semaphore_signal_p` lowering still inspects the original
  `aval.memory_space` (a `MeshMemorySpace`) to resolve `{SC_CORE: 0, SC_SUB:
  sub}` to the SC vector subcore IDs.
  """
  global _TC_CROSS_CORE_SEM_PATCHED
  if _TC_CROSS_CORE_SEM_PATCHED:
    return
  tpu_core: Any = sys.modules["jax._src.pallas.mosaic.core"]
  tc_lowering: Any = sys.modules["jax._src.pallas.mosaic.lowering"]
  orig_fn = tpu_core.memory_space_to_tpu_memory_space
  mesh_ms_cls = getattr(pl, "MeshMemorySpace", ())

  def _patched_ms_to_tpu(x, core_type=None):
    if (
        core_type is None
        and isinstance(x, mesh_ms_cls)
        and x.memory_space == pltpu.MemorySpace.SEMAPHORE
    ):
      frame = inspect.currentframe()
      caller = frame.f_back if frame is not None else None
      if caller is not None and caller.f_code.co_name in (
          "lower_jaxpr_into_unpipelined_module",
          "_check_block_mappings",
          "lower_jaxpr_to_pipelined_module",
      ):
        return pltpu.MemorySpace.SEMAPHORE
    return orig_fn(x, core_type=core_type)

  tpu_core.memory_space_to_tpu_memory_space = _patched_ms_to_tpu
  tc_lowering.tpu_core.memory_space_to_tpu_memory_space = _patched_ms_to_tpu
  _TC_CROSS_CORE_SEM_PATCHED = True


def allow_mpmd_array_writes() -> None:
  """Idempotently allows `pl.kernel` (`mpmd_map`) to signal/write shared value operands.

  `sc_forward_start` and `sc_ragged_a2a_start_waves(sc_forward=True)` pass
  shared buffers (`recv`, `stage`) and cross-core semaphores (`_sem_value`)
  by value with `out_type=()` so XLA adds no false data dependency between
  `cae_sc_forward` on the SparseCore and `fused_ep_moe_v2` on the TensorCore,
  letting them execute concurrently. A JAX change added a `WriteEffect`
  check in `mpmd._mpmd_map_abstract_eval` that rejects `WriteEffect` on
  non-`AbstractRef` inputs unless `index in input_output_aliases`. Wrapping
  `_mpmd_map_abstract_eval` to treat non-Ref inputs as aliased for that check
  preserves the by-value concurrent lowering without modifying `out_type`.
  """
  global _MPMD_ARRAY_WRITES_PATCHED
  if _MPMD_ARRAY_WRITES_PATCHED:
    return
  mpmd_mod: Any = sys.modules.get("jax._src.pallas.mpmd") or getattr(
      pl, "mpmd", None
  )
  state_mod: Any = sys.modules.get("jax._src.state.types")
  if mpmd_mod is None or state_mod is None:
    return
  orig_eval = mpmd_mod._mpmd_map_abstract_eval  # pylint: disable=protected-access
  abstract_ref_cls = state_mod.AbstractRef

  def _patched_eval(*in_avals, input_output_aliases, **kwargs):
    extra_aliases = dict(input_output_aliases)
    for idx, aval in enumerate(in_avals):
      if not isinstance(aval, abstract_ref_cls) and idx not in extra_aliases:
        extra_aliases[idx] = -1
    return orig_eval(*in_avals, input_output_aliases=extra_aliases, **kwargs)

  mpmd_mod._mpmd_map_abstract_eval = _patched_eval  # pylint: disable=protected-access
  mpmd_mod.mpmd_map_p.def_effectful_abstract_eval(_patched_eval)
  _MPMD_ARRAY_WRITES_PATCHED = True


def sc_mesh(num_subcores: int = NUM_SUBCORES) -> Any:
  """Returns the SparseCore vector-subcore mesh the sender kernels run on.

  Args:
    num_subcores: Vector subcores to use on the core.

  Returns:
    A one-core plsc.VectorSubcoreMesh.
  """
  return _plsc().VectorSubcoreMesh(
      core_axis_name=SC_CORE,
      subcore_axis_name=SC_SUB,
      num_cores=1,
      num_subcores=num_subcores,
  )


def _subcore_slice(lo, m, sub, num_subcores: int):
  """Subcore `sub`'s contiguous slice `[sub_lo, sub_lo + sub_m)` of `[lo, lo + m)`."""
  m = jnp.maximum(jnp.int32(0), m)
  if num_subcores == 1:
    return lo, m
  nsub = jnp.int32(num_subcores)
  s0 = lax.div(sub * m, nsub)
  s1 = lax.div((sub + 1) * m, nsub)
  return lo + s0, s1 - s0


def subcore_row_totals(
    disp_tab: jax.Array,
    me: jax.Array | int,
    *,
    ep: int,
    waves: int,
    num_subcores: int = NUM_SUBCORES,
    stage_send_tab: jax.Array | None = None,
) -> jax.Array:
  """Returns the rows each SparseCore subcore sends.

  Args:
    disp_tab: [4 * ep, W] dispatch table.
    me: This shard's index.
    ep: Expert-parallel width.
    waves: W.
    num_subcores: Vector subcores sharing the sends.
    stage_send_tab: Optional De-Dup stage table; its stage runs are counted too.

  Returns:
    [num_subcores] i32 rows sent per subcore across all waves, head and tail
    passes, and stage runs.

  Raises:
    ValueError: If `ep` is not divisible by `num_subcores`.
  """
  if ep % num_subcores != 0:
    raise ValueError(
        f"ep={ep} must be divisible by num_subcores={num_subcores}"
    )
  ii = lax.rem(jnp.asarray(me, jnp.int32) + jnp.arange(ep, dtype=jnp.int32), ep)
  cnt = jnp.sum(disp_tab[ep + ii, :waves].astype(jnp.int32), axis=1)
  tot = jnp.sum(cnt.reshape(ep // num_subcores, num_subcores), axis=0)
  if stage_send_tab is not None:
    nb = stage_send_tab.shape[0] // 5
    cnt_s = stage_send_tab[nb : 2 * nb, :waves].astype(jnp.int32)
    sub_idx = jnp.arange(num_subcores, dtype=jnp.int32)[:, None, None]
    c_m = jnp.maximum(0, cnt_s)
    s0 = (sub_idx * c_m[None, :, :]) // num_subcores
    s1 = ((sub_idx + 1) * c_m[None, :, :]) // num_subcores
    tot = tot + jnp.sum(s1 - s0, axis=(1, 2))
  return tot


def tc_mesh() -> pltpu.TensorCoreMesh:
  """Returns the one-core TensorCore mesh used beside the SparseCore kernels.

  Returns:
    A pltpu.TensorCoreMesh.
  """
  return pltpu.TensorCoreMesh(axis_name=TC_AXIS, num_cores=1)


def _shard_map_axis() -> str:
  names = jax.sharding.get_abstract_mesh().axis_names
  if len(names) != 1:
    raise ValueError(
        f"SC dispatch must run inside a single-axis shard_map; got {names}"
    )
  return names[0]


def _pad_words(x: jax.Array) -> jax.Array:
  x = jnp.reshape(x, (-1,)).astype(jnp.int32)
  n = x.shape[0]
  return jnp.pad(x, (0, -n % SMEM_PAD))


# ------------------------------------------------------------------------------
# (1) TC list kernel: barrier, list DMAs, TC semaphore allocation.
# ------------------------------------------------------------------------------


def _tc_list_kernel(
    list_tab_hbm,
    tg_lists,
    tg_land,
    recv_sems,
    tg_sem,
    list_send_sem,
    *extra,
    ep,
    list_rows,
    axis,
):
  """All-pairs barrier, then each (me -> d) token-gather list DMA."""
  del recv_sems  # allocated here (outputs), signalled by the SC send kernel
  list_tab = extra[-1]
  pltpu.sync_copy(list_tab_hbm, list_tab)
  me = lax.axis_index(axis)
  bsem = pltpu.get_barrier_semaphore()
  for i in range(ep):
    pl.semaphore_signal(bsem, 1, device_id={axis: i})
  pl.semaphore_wait(bsem, ep)
  for i in range(ep):
    d = lax.rem(me + i, ep)
    n_list = list_tab[d]
    src = tg_lists.at[pl.ds(d * list_rows, n_list)]
    dst = tg_land.at[pl.ds(me * list_rows, n_list)]
    if i == 0:

      @pl.when(n_list > 0)
      def _(src=src, dst=dst):
        pltpu.make_async_copy(src, dst, tg_sem).start()

    else:

      @pl.when(n_list > 0)
      def _(src=src, dst=dst, d=d):
        pltpu.make_async_remote_copy(
            src, dst, list_send_sem, tg_sem, device_id={axis: d}
        ).start()


def _tc_list_kernel_sc_stage(
    list_tab_hbm,
    tg_lists,
    tg_land,
    *rest,
    n_sc_sems=1,
    ep,
    list_rows,
    axis,
):
  """_tc_list_kernel with the gateway SC semaphores threaded as operands."""
  # sc_forward / sc_return / sc_combine: the SparseCore semaphores are threaded
  # through so every device allocates them before this kernel's all-pairs
  # barrier.
  recv_sems, tg_sem, list_send_sem, *extra = rest[n_sc_sems:]
  _tc_list_kernel(
      list_tab_hbm,
      tg_lists,
      tg_land,
      recv_sems,
      tg_sem,
      list_send_sem,
      *extra,
      ep=ep,
      list_rows=list_rows,
      axis=axis,
  )


def _sc_sem_alloc_kernel(*sems):
  """Allocates `sems` (its outputs) on the SparseCore vector subcores."""
  del sems


def _sem_value(sem, mesh):
  """A semaphore array as a plain operand constrained to `mesh`'s semaphores."""
  # Passed by value, a semaphore adds no XLA data edge between the kernels
  # that share it, so the SC forward kernel and the fused TC kernel stay
  # concurrent.
  state_primitives = sys.modules["jax._src.state.primitives"]
  return state_primitives.with_memory_space_constraint(
      sem, pltpu.SEMAPHORE @ mesh
  )


# ------------------------------------------------------------------------------
# (2) SC send kernels.
# ------------------------------------------------------------------------------


def _stage_device(axis, gw, sig, stage_sc_subcores):
  """device_id of a stage DMA's receive semaphore on gateway `gw`."""
  # stage_sc_subcores == 0: the gateway TensorCore's stage_sems (TC forwards).
  # Otherwise (sc_forward) the gateway SparseCore subcore that owns served slot
  # `sig`: sig % stage_sc_subcores.
  if not stage_sc_subcores:
    return {axis: gw, TC_AXIS: 0}
  owner = (
      jnp.int32(0)
      if stage_sc_subcores == 1
      else lax.rem(sig, jnp.int32(stage_sc_subcores))
  )
  return {axis: gw, SC_CORE: 0, SC_SUB: owner}


def _stage_subcore_piece(n, cs, head, sub, num_subcores):
  """Subcore `sub`'s `(lo, m)` piece of one `(b, w)` stage run in this pass."""
  sub_lo, sub_n = _subcore_slice(jnp.int32(0), n, sub, num_subcores)
  lo = sub_lo + (jnp.int32(0) if head else cs)
  m = jnp.minimum(cs, sub_n) if head else sub_n - cs
  return lo, m


def _sc_send_kernel(
    tab_hbm,
    send,
    recv,
    recv_sems,
    *rest,
    ep,
    waves,
    chunk_rows,
    axis,
    num_subcores=NUM_SUBCORES,
    has_stage=False,
    num_boards=0,
    board_size=8,
    stage_sc_subcores=0,
    stage_all_remote=False,
):
  """Issues every (wave, destination) head / tail run of `send` -> `recv`."""
  if has_stage:
    stage_ref, stage_sems, send_sem, tab = rest
  else:
    send_sem, tab = rest
    stage_ref = stage_sems = None
  pltpu.sync_copy(tab_hbm, tab)
  me = lax.axis_index(axis)
  sub = lax.axis_index(SC_SUB) if num_subcores > 1 else jnp.int32(0)
  cs = jnp.int32(chunk_rows)

  def run(d, w, head, dev):
    n = tab[(ep + d) * waves + w]
    lo = jnp.int32(0) if head else cs
    m = jnp.minimum(cs, n) if head else n - cs
    src_off = tab[d * waves + w] + lo
    dst_off = tab[(2 * ep + d) * waves + w] + lo

    @pl.when(m > 0)
    def _():
      pltpu.make_async_remote_copy(
          send.at[pl.ds(src_off, m)],
          recv.at[pl.ds(dst_off, m)],
          send_sem,
          recv_sems.at[w],
          device_id=dev,
      ).start()

  for w in range(waves):
    for head in (True, False):

      def stage_pass(w=w, head=head):
        if not has_stage:
          return
        assert stage_ref is not None
        assert stage_sems is not None
        sbase = 3 * ep * waves
        my_board = lax.div(me, jnp.int32(board_size))
        for bi in range(1, num_boards):
          b = lax.rem(my_board + bi, num_boards)
          n = tab[sbase + (num_boards + b) * waves + w]
          s_off = tab[sbase + b * waves + w]
          d_off = tab[sbase + (2 * num_boards + b) * waves + w]
          gw = tab[sbase + 3 * num_boards * waves + b]
          sig = tab[sbase + (3 * num_boards * waves + num_boards) + b]
          lo, m = _stage_subcore_piece(n, cs, head, sub, num_subcores)

          @pl.when(m > 0)
          def _(s_off=s_off, d_off=d_off, lo=lo, m=m, gw=gw, sig=sig, w=w):
            pltpu.make_async_remote_copy(
                send.at[pl.ds(s_off + lo, m)],
                stage_ref.at[pl.ds(d_off + lo, m)],
                send_sem,
                stage_sems.at[sig * waves + w],
                device_id=_stage_device(axis, gw, sig, stage_sc_subcores),
            ).start()

      if num_subcores == 1:
        run(me, w, head, {TC_AXIS: 0})
      else:

        @pl.when(sub == 0)
        def _(w=w, head=head):
          run(me, w, head, {TC_AXIS: 0})

        @pl.when(sub > 0)
        def _(w=w, head=head):
          d0 = (
              (me // board_size) * board_size + lax.rem(me + sub, board_size)
              if stage_all_remote
              else lax.rem(me + sub, ep)
          )
          run(d0, w, head, {axis: d0, TC_AXIS: 0})

      n_dir_peers = board_size if stage_all_remote else ep

      def body(k, c, w=w, head=head):
        d = (
            (me // board_size) * board_size
            + lax.rem(me + k * num_subcores + sub, board_size)
            if stage_all_remote
            else lax.rem(me + k * num_subcores + sub, ep)
        )
        run(d, w, head, {axis: d, TC_AXIS: 0})
        return c

      lax.fori_loop(1, n_dir_peers // num_subcores, body, jnp.int32(0))
      stage_pass()


def _sc_gather_send_kernel(
    tab_hbm,
    perm_hbm,
    rows,
    recv,
    recv_sems,
    *rest,
    ep,
    waves,
    chunk_rows,
    axis,
    num_subcores=NUM_SUBCORES,
    has_stage=False,
    num_boards=0,
    board_size=8,
    stage_sc_subcores=0,
    stage_all_remote=False,
):
  """_sc_send_kernel without the XLA pack: one DMA per row, from `rows`."""
  if has_stage:
    stage_ref, stage_sems, send_sem, tab, buf = rest
  else:
    send_sem, tab, buf = rest
    stage_ref = stage_sems = None
  pltpu.sync_copy(tab_hbm, tab)
  me = lax.axis_index(axis)
  sub = lax.axis_index(SC_SUB) if num_subcores > 1 else jnp.int32(0)
  cs = jnp.int32(chunk_rows)

  def gather_piece(src_off, dst_off, m, dst_ref, dst_sem, dev):
    m = jnp.maximum(jnp.int32(0), m)
    k0 = lax.div(src_off, jnp.int32(SMEM_PAD))
    k1 = lax.div(src_off + m + (SMEM_PAD - 1), jnp.int32(SMEM_PAD))
    n_steps = jnp.where(m > 0, k1 - k0, jnp.int32(0))

    def gather_step(s, c):
      blk_off = (k0 + s) * SMEM_PAD
      lo_off = jnp.maximum(src_off, blk_off)
      hi_off = jnp.minimum(src_off + m, blk_off + SMEM_PAD)
      cnt = hi_off - lo_off
      shift = lo_off - blk_off
      dst_base = dst_off + (lo_off - src_off)
      pltpu.sync_copy(perm_hbm.at[pl.ds(blk_off, SMEM_PAD)], buf)

      def gather_row(r, rc):
        idx = buf[shift + r]
        pltpu.make_async_remote_copy(
            rows.at[pl.ds(idx, 1)],
            dst_ref.at[pl.ds(dst_base + r, 1)],
            send_sem,
            dst_sem,
            device_id=dev,
        ).start()
        return rc

      lax.fori_loop(0, cnt, gather_row, jnp.int32(0))
      return c

    lax.fori_loop(0, n_steps, gather_step, jnp.int32(0))

  def run(d, w, head, dev):
    n = tab[(ep + d) * waves + w]
    lo = jnp.int32(0) if head else cs
    m = jnp.minimum(cs, n) if head else n - cs
    gather_piece(
        tab[d * waves + w] + lo,
        tab[(2 * ep + d) * waves + w] + lo,
        m,
        recv,
        recv_sems.at[w],
        dev,
    )

  for w in range(waves):
    for head in (True, False):

      def stage_pass(w=w, head=head):
        if not has_stage:
          return
        assert stage_ref is not None
        assert stage_sems is not None
        sbase = 3 * ep * waves
        my_board = lax.div(me, jnp.int32(board_size))
        for bi in range(1, num_boards):
          b = lax.rem(my_board + bi, num_boards)
          n = tab[sbase + (num_boards + b) * waves + w]
          s_off = tab[sbase + b * waves + w]
          d_off = tab[sbase + (2 * num_boards + b) * waves + w]
          gw = tab[sbase + 3 * num_boards * waves + b]
          sig = tab[sbase + (3 * num_boards * waves + num_boards) + b]
          lo, m = _stage_subcore_piece(n, cs, head, sub, num_subcores)
          gather_piece(
              s_off + lo,
              d_off + lo,
              m,
              stage_ref,
              stage_sems.at[sig * waves + w],
              _stage_device(axis, gw, sig, stage_sc_subcores),
          )

      if num_subcores == 1:
        run(me, w, head, {TC_AXIS: 0})
      else:

        @pl.when(sub == 0)
        def _(w=w, head=head):
          run(me, w, head, {TC_AXIS: 0})

        @pl.when(sub > 0)
        def _(w=w, head=head):
          d0 = (
              (me // board_size) * board_size + lax.rem(me + sub, board_size)
              if stage_all_remote
              else lax.rem(me + sub, ep)
          )
          run(d0, w, head, {axis: d0, TC_AXIS: 0})

      n_dir_peers = board_size if stage_all_remote else ep

      def body(k, c, w=w, head=head):
        d = (
            (me // board_size) * board_size
            + lax.rem(me + k * num_subcores + sub, board_size)
            if stage_all_remote
            else lax.rem(me + k * num_subcores + sub, ep)
        )
        run(d, w, head, {axis: d, TC_AXIS: 0})
        return c

      lax.fori_loop(1, n_dir_peers // num_subcores, body, jnp.int32(0))
      stage_pass()


# ------------------------------------------------------------------------------
# Drop-in replacements for tc_sender.ragged_a2a_start_waves / send_done.
# ------------------------------------------------------------------------------


def sc_ragged_a2a_start_waves(
    send,
    recv,
    disp_tab,
    me,
    *,
    ep,
    waves,
    chunk_rows,
    rows,
    perm,
    list_tab,
    tg_lists,
    tg_land,
    stage_send_tab=None,
    stage=None,
    num_boards=0,
    num_slots=0,
    board_size=8,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
    stage_hoist="none",
    pack_parity=1,
    gather=False,
    num_subcores=NUM_SUBCORES,
    sc_forward=False,
    sc_return=False,
    sc_combine=False,
    stage_all_remote=False,
):
  """Starts the wave exchange from the SparseCore (direct and De-Dup).

  SC version of tc_sender.ragged_a2a_start_waves (direct and De-Dup).

  Returns the 8-tuple `fut` (send_sem, recv_sems, send, recv, list_send_sem,
  tg_sem, tg_lists, tg_land) or 10-tuple when `stage is not None`
  (..., stage_sems, stage). gather: the per-row gather send (no TC pack);
  otherwise contiguous sends from the TC-packed rows.

  sc_forward (De-Dup): the stage runs count on the gateway SparseCore's
  semaphores (subcore sigma % num_subcores) for sc_forward_start, and `fut`
  gets an 11th entry, those semaphores. The TC stage_sems (fut[8]) are still
  allocated for the fused kernel's operand list and are never signalled.

  Args:
    send: Send buffer (packed rows, or the token rows for the gather send).
    recv: Uninitialized receive buffer.
    disp_tab: [4 * ep, W] dispatch table.
    me: This shard's index.
    ep: Expert-parallel width.
    waves: W.
    chunk_rows: Rows per DMA chunk.
    rows: This shard's token rows.
    perm: Local token of each packed send row.
    list_tab: Token-gather list table.
    tg_lists: Token-gather lists to send.
    tg_land: Landing buffer of the lists.
    stage_send_tab: De-Dup stage table, or None.
    stage: Gateway stage buffer, or None.
    num_boards: Boards (De-Dup).
    num_slots: Stage slots per gateway (De-Dup).
    board_size: Devices per board (De-Dup).
    direct_order: Ignored on SC (kept for signature compatibility with
      wave_dispatch.start).
    stripe_boards: Ignored on SC.
    stripe_board_size: Ignored on SC.
    stage_hoist: Ignored on SC.
    pack_parity: Ignored on SC (no TC pack semaphores).
    gather: Use the per-row gather send instead of contiguous sends.
    num_subcores: Vector subcores.
    sc_forward: Count the stage runs on the gateway SparseCore's semaphores.
    sc_return: Allocate SparseCore / TensorCore semaphores for SC return pushes.
    sc_combine: Allocate SparseCore / TensorCore semaphores for SC gateway
      combine optical sends.
    stage_all_remote: Whether all remote-board tokens are staged (min_fanout <=
      1), so direct sends only visit intra-board peers.

  Returns:
    The `fut` tuple described above.

  Raises:
    ValueError: If the arguments are inconsistent.
  """
  del me, direct_order, stripe_boards, stripe_board_size, stage_hoist
  del pack_parity
  _ensure_sc_compat()
  has_stage = stage is not None
  if (sc_forward or sc_return or sc_combine) and not has_stage:
    raise ValueError("sc_forward/sc_return/sc_combine need a De-Dup stage")
  if has_stage != (stage_send_tab is not None):
    raise ValueError("pass stage_send_tab and stage together or neither")
  if has_stage and (num_boards < 2 or num_slots < 1):
    raise ValueError(
        "stage needs num_boards >= 2 and num_slots >= 1; got"
        f" {num_boards}, {num_slots}"
    )
  if ep % num_subcores != 0:
    raise ValueError(
        f"ep={ep} must be divisible by num_subcores={num_subcores}"
    )
  if board_size > 0:
    eff_board_size = board_size
  elif num_boards > 0:
    eff_board_size = ep // num_boards
  else:
    eff_board_size = 8
  hbm = functools.partial(
      pltpu.with_memory_space_constraint, memory_space=pltpu.HBM
  )
  send, recv = hbm(send), hbm(recv)
  if gather:
    rows = hbm(rows)
  tg_lists, tg_land = hbm(tg_lists), hbm(tg_land)
  if has_stage:
    assert stage is not None
    stage = hbm(stage)
  axis = _shard_map_axis()
  tcm, scm = tc_mesh(), sc_mesh(num_subcores)
  list_rows = tg_lists.shape[0] // ep

  # Pack on the TC (XLA gather; the TC start kernel does it with per-row
  # DMAs). Padding entries of perm are never sent. With `gather` there is no
  # TC pack.
  if gather:
    packed = rows
  else:
    packed = jnp.take(rows, perm.astype(jnp.int32), axis=0, mode="clip")
    if packed.shape != send.shape:
      raise ValueError(f"packed {packed.shape} != send {send.shape}")
    packed = pltpu.with_memory_space_constraint(packed, pltpu.HBM)

  # (1) TC list kernel (allocates stage_sems before barrier when has_stage).
  tab_parts = [disp_tab[: 3 * ep, :waves].reshape(-1).astype(jnp.int32)]
  if has_stage:
    assert stage_send_tab is not None
    tab_parts.append(
        stage_send_tab[: 3 * num_boards, :waves].reshape(-1).astype(jnp.int32)
    )
    tab_parts.append(
        stage_send_tab[3 * num_boards : 5 * num_boards, 0]
        .reshape(-1)
        .astype(jnp.int32)
    )
  tab3 = _pad_words(jnp.concatenate(tab_parts, axis=0))
  lists_ref = jax.new_ref(tg_lists, memory_space=pltpu.HBM)
  land_ref = jax.new_ref(tg_land, memory_space=pltpu.HBM)
  recv_ref = jax.new_ref(recv, memory_space=pltpu.HBM)
  list_smem = pltpu.SMEM((SMEM_PAD * pl.cdiv(ep + 1, SMEM_PAD),), jnp.int32)
  list_out_type = [
      pltpu.SemaphoreType.DMA((waves,)) @ tcm,
      pltpu.SemaphoreType.DMA(()) @ tcm,
      pltpu.SemaphoreType.DMA(()) @ tcm,
  ]
  if has_stage:
    list_out_type.append(pltpu.SemaphoreType.DMA((num_slots * waves,)) @ tcm)
  if sc_return:
    list_out_type.append(pltpu.SemaphoreType.DMA(()) @ tcm)
  if sc_combine:
    list_out_type.append(pltpu.SemaphoreType.DMA(()) @ tcm)
  list_params = pltpu.CompilerParams(
      collective_id=LIST_COLLECTIVE_ID,
      skip_device_barrier=True,
      has_side_effects=True,
  )
  sc_params = pltpu.CompilerParams(
      has_side_effects=True, skip_device_barrier=True
  )
  sc_stage_sems = None
  sc_ret_sems = None
  ret_recv_sem = None
  sc_comb_sems = None
  comb_recv_sem = None
  allow_tc_sc_mesh_pair()
  allow_tc_cross_core_semaphores()
  allow_mpmd_array_writes()
  if sc_forward:
    # The gateway SparseCore's stage semaphores (and optional return/combine
    # semaphores), allocated before the list kernel's barrier (its operands):
    # no stage, return, or combine DMA can reach a peer before they exist there.
    sc_alloc_types = [pltpu.SemaphoreType.DMA((num_slots * waves,)) @ scm]
    if sc_return:
      sc_alloc_types.append(
          pltpu.SemaphoreType.REGULAR((MAX_SC_RET_VISITS,)) @ scm
      )
    if sc_combine:
      sc_alloc_types.append(
          pltpu.SemaphoreType.REGULAR((MAX_SC_COMB_STEPS,)) @ scm
      )
    alloc_sfx = ("_ret" if sc_return else "") + ("_comb" if sc_combine else "")
    if len(sc_alloc_types) == 1:
      sc_alloc_outs = (
          pl.kernel(
              _sc_sem_alloc_kernel,
              out_type=sc_alloc_types[0],
              mesh=scm,
              compiler_params=sc_params,
              name=f"cae_sc_fwd_stage_sems_ep{ep}_w{waves}",
          )(),
      )
    else:
      sc_alloc_outs = pl.kernel(
          _sc_sem_alloc_kernel,
          out_type=tuple(sc_alloc_types),
          mesh=scm,
          compiler_params=sc_params,
          name=f"cae_sc_fwd_stage_sems_ep{ep}_w{waves}{alloc_sfx}",
      )()
    sc_sem_refs = [
        jax.new_ref(s, memory_space=pltpu.SEMAPHORE @ scm)
        for s in sc_alloc_outs
    ]
    list_outs = pl.kernel(
        functools.partial(
            _tc_list_kernel_sc_stage,
            n_sc_sems=len(sc_sem_refs),
            ep=ep,
            list_rows=list_rows,
            axis=axis,
        ),
        out_type=tuple(list_out_type),
        mesh=tcm,
        scratch_types=[list_smem],
        compiler_params=list_params,
        name=f"cae_list_start_ep{ep}_dedup_scfwd{alloc_sfx}",
    )(_pad_words(list_tab), lists_ref, land_ref, *sc_sem_refs)
    sc_frozen = [_sem_value(jax.freeze(r), scm) for r in sc_sem_refs]
    sc_stage_sems = sc_frozen[0]
    sc_idx = 1
    if sc_return:
      sc_ret_sems = sc_frozen[sc_idx]
      sc_idx += 1
    if sc_combine:
      sc_comb_sems = sc_frozen[sc_idx]
  else:
    list_outs = pl.kernel(
        functools.partial(
            _tc_list_kernel, ep=ep, list_rows=list_rows, axis=axis
        ),
        out_type=tuple(list_out_type),
        mesh=tcm,
        scratch_types=[list_smem],
        compiler_params=list_params,
        name=f"cae_list_start_ep{ep}{'_dedup' if has_stage else ''}",
    )(_pad_words(list_tab), lists_ref, land_ref)
  if has_stage:
    recv_sems, tg_sem, list_send_sem, stage_sems, *extra_tc_sems = list_outs
    tc_idx = 0
    if sc_return:
      ret_recv_sem = extra_tc_sems[tc_idx]
      tc_idx += 1
    if sc_combine:
      comb_recv_sem = extra_tc_sems[tc_idx]
  else:
    recv_sems, tg_sem, list_send_sem = list_outs
    stage_sems = None
  tg_lists = jax.freeze(lists_ref)
  tg_land = jax.freeze(land_ref)

  # (2) SC send kernel.
  rsem_ref = jax.new_ref(recv_sems, memory_space=pltpu.SEMAPHORE @ tcm)
  if has_stage and sc_forward:
    stage_ref = jax.new_ref(stage, memory_space=pltpu.HBM)
    ssem_stage_ref = None
    stage_args = (stage_ref, sc_stage_sems)
  elif has_stage:
    stage_ref = jax.new_ref(stage, memory_space=pltpu.HBM)
    ssem_stage_ref = jax.new_ref(stage_sems, memory_space=pltpu.SEMAPHORE @ tcm)
    stage_args = (stage_ref, ssem_stage_ref)
  else:
    stage_ref = ssem_stage_ref = None
    stage_args = ()
  sub_sfx = "" if num_subcores == 1 else f"_sub{num_subcores}"
  dedup_sfx = (
      ("_dedup" if has_stage else "")
      + ("_scfwd" if sc_forward else "")
      + ("_bg" if has_stage and stage_all_remote else "")
  )
  stage_kw = dict(
      has_stage=has_stage,
      num_boards=num_boards,
      board_size=eff_board_size,
      stage_sc_subcores=num_subcores if sc_forward else 0,
      stage_all_remote=bool(has_stage and stage_all_remote),
  )
  if gather:
    perm_pad = _pad_words(jnp.pad(perm.astype(jnp.int32), (0, PERM_BUF_WORDS)))
    send_sem = pl.kernel(
        functools.partial(
            _sc_gather_send_kernel,
            ep=ep,
            waves=waves,
            chunk_rows=chunk_rows,
            axis=axis,
            num_subcores=num_subcores,
            **stage_kw,
        ),
        out_type=pltpu.SemaphoreType.DMA(()) @ scm,
        mesh=scm,
        scratch_types=[
            pltpu.SMEM((tab3.shape[0],), jnp.int32),
            pltpu.SMEM((PERM_BUF_WORDS,), jnp.int32),
        ],
        compiler_params=sc_params,
        name=f"cae_sc_gather_send_ep{ep}_w{waves}{dedup_sfx}{sub_sfx}",
    )(tab3, perm_pad, rows, recv_ref, rsem_ref, *stage_args)
  else:
    send_sem = pl.kernel(
        functools.partial(
            _sc_send_kernel,
            ep=ep,
            waves=waves,
            chunk_rows=chunk_rows,
            axis=axis,
            num_subcores=num_subcores,
            **stage_kw,
        ),
        out_type=pltpu.SemaphoreType.DMA(()) @ scm,
        mesh=scm,
        scratch_types=[pltpu.SMEM((tab3.shape[0],), jnp.int32)],
        compiler_params=sc_params,
        name=f"cae_sc_send_ep{ep}_w{waves}{dedup_sfx}{sub_sfx}",
    )(tab3, packed, recv_ref, rsem_ref, *stage_args)
  recv = jax.freeze(recv_ref)
  recv_sems = jax.freeze(rsem_ref)
  if has_stage:
    assert stage_ref is not None
    stage = jax.freeze(stage_ref)
    if ssem_stage_ref is not None:
      stage_sems = jax.freeze(ssem_stage_ref)
  packed, recv = hbm(packed), hbm(recv)
  tg_lists, tg_land = hbm(tg_lists), hbm(tg_land)
  if has_stage:
    assert stage is not None
    stage = hbm(stage)
    extra_fut = ()
    if sc_forward:
      extra_fut += (sc_stage_sems,)
    if sc_return:
      extra_fut += (sc_ret_sems, ret_recv_sem)
    if sc_combine:
      extra_fut += (sc_comb_sems, comb_recv_sem)
    return (
        send_sem,
        recv_sems,
        packed,
        recv,
        list_send_sem,
        tg_sem,
        tg_lists,
        tg_land,
        stage_sems,
        stage,
    ) + extra_fut
  return (
      send_sem,
      recv_sems,
      packed,
      recv,
      list_send_sem,
      tg_sem,
      tg_lists,
      tg_land,
  )


# ------------------------------------------------------------------------------
# (3) De-Dup gateway forwards issued from the SparseCore (sc_forward).
# ------------------------------------------------------------------------------


def _sc_rows_wait(sem, ref, rows, cap):
  """Waits `rows` rows of `ref`'s row size on DMA semaphore `sem` (SC side)."""
  capv = jnp.int32(cap)

  def body(_, c):
    pltpu.make_async_copy(
        ref.at[pl.ds(0, cap)], ref.at[pl.ds(0, cap)], sem
    ).wait()
    return c

  lax.fori_loop(0, lax.div(rows, capv), body, jnp.int32(0))
  rem = lax.rem(rows, capv)

  @pl.when(rem > 0)
  def _():
    pltpu.make_async_copy(
        ref.at[pl.ds(0, rem)], ref.at[pl.ds(0, rem)], sem
    ).wait()


def sc_forward_body(
    *,
    me,
    sub,
    stage_tab,
    cnt_tab,
    cnt_base,
    fwd_hbm,
    win,
    misc,
    stage,
    recv,
    recv_sems,
    stage_sems,
    send_sem,
    copy_fn,
    count_local,
    waves,
    num_slots,
    num_boards,
    fwd_max_runs,
    num_subcores,
):
  """The gateway's K4 forwards (memory_ref.forward_staged_wave_body), all waves.

  For each issue wave w, the subcore that owns served slot sigma
  (sigma % num_subcores, the subcore the senders' stage DMAs count on) issues
  the slot-0 runs of (sigma, w) -- rows staged in earlier waves, already
  waited here -- then waits the stage run of (sigma, w) byte-exact on
  `stage_sems` and issues its slot-1 runs. Each run is one copy
  `stage[stage_row:+rows] -> recv[dst_row:+rows]` on its target, counted on
  the target's `recv_sems[w + (dst >> FWD_DST_SHIFT)]` exactly as the TC path
  does. The forward table stays in HBM (`fwd_hbm`, flat, padded by
  win.shape[0] words) and streams through the SMEM window `win`; the stage
  table is `stage_tab` and the run counts start at `cnt_tab[cnt_base]`.
  Then drains this subcore's forward sends on `send_sem`.

  Shared by the SparseCore kernel (_sc_forward_kernel) and the CPU
  interpret-mode harness (dedup_moe_test), which run it with their own
  `copy_fn(src, dst, sem, d, is_me)`; `count_local` says whether a local
  (d == me) copy also counts on `send_sem`.

  Args:
    me: This device (traced int32).
    sub: This subcore (traced int32; 0 with one subcore).
    stage_tab: SMEM flat stage table.
    cnt_tab: SMEM words holding the flat fwd_cnt at `cnt_base`.
    cnt_base: Word offset of fwd_cnt in `cnt_tab`.
    fwd_hbm: Flat fwd_tab words in HBM, padded by win.shape[0].
    win: SMEM window (a multiple of SMEM_PAD words) of fwd_hbm.
    misc: SMEM scratch: [0] window base word, [1] rows counted on send_sem.
    stage: The gateway stage buffer.
    recv: The receive buffer (the fused kernel's lhs).
    recv_sems: [W] receive-wave semaphores.
    stage_sems: [S * W] stage semaphores.
    send_sem: Send semaphore of the forwards.
    copy_fn: Starts one forward copy.
    count_local: Local copies count on send_sem too.
    waves: W.
    num_slots: S.
    num_boards: nb.
    fwd_max_runs: Forward-table depth per (sigma, w).
    num_subcores: Vector subcores of the mesh.
  """
  fields = dedup_plan_mod.FWD_FIELDS
  fwd_slots = dedup_plan_mod.FWD_SLOTS
  win_words = win.shape[0]
  if win_words % SMEM_PAD:
    raise ValueError(
        f"window of {win_words} words: want a multiple of {SMEM_PAD}"
    )
  cap = stage.shape[0]
  misc[0] = jnp.int32(-(1 << 30))
  misc[1] = jnp.int32(0)

  def issue_run(off, w):
    s_row = win[off + dedup_plan_mod.FWD_STAGE_ROW]
    m = win[off + dedup_plan_mod.FWD_ROWS]
    d_field = win[off + dedup_plan_mod.FWD_DST]
    d_row = win[off + dedup_plan_mod.FWD_DST_ROW]
    d = d_field & dedup_plan_mod.FWD_DST_MASK
    sem = recv_sems.at[
        w + jnp.right_shift(d_field, dedup_plan_mod.FWD_DST_SHIFT)
    ]
    src = stage.at[pl.ds(s_row, m)]
    dst = recv.at[pl.ds(d_row, m)]
    is_me = d == me

    @pl.when(jnp.logical_and(is_me, m > 0))
    def _():
      copy_fn(src, dst, sem, d, True)

    @pl.when(jnp.logical_and(jnp.logical_not(is_me), m > 0))
    def _():
      copy_fn(src, dst, sem, d, False)

    counted = m if count_local else jnp.where(is_me, 0, m)
    misc[1] = misc[1] + jnp.maximum(counted, 0)

  def issue_slot(sig, w, slot):
    base = cnt_base + (sig * waves + w) * fwd_slots
    first = jnp.int32(0)
    for k in range(slot):
      first = first + cnt_tab[base + k]
    r0 = (sig * waves + w) * fwd_max_runs + first
    r_end = r0 + cnt_tab[base + slot]

    def cond(r):
      return r < r_end

    def body(r):
      pos = r * fields

      # The window holds whole runs (its base is SMEM_PAD-aligned and a run is
      # `fields` words), so a run is in it iff its first word is.
      @pl.when((pos < misc[0]) | (pos >= misc[0] + win_words))
      def _():
        wbase = lax.div(pos, jnp.int32(SMEM_PAD)) * SMEM_PAD
        pltpu.sync_copy(fwd_hbm.at[pl.ds(wbase, win_words)], win)
        misc[0] = wbase

      wbase = misc[0]
      hi = jnp.minimum(r_end, lax.div(wbase + win_words, jnp.int32(fields)))

      def one(k, c):
        issue_run((r + k) * fields - wbase, w)
        return c

      lax.fori_loop(0, hi - r, one, jnp.int32(0))
      return hi  # > r: the window always holds run r.

    lax.while_loop(cond, body, r0)

  def owned(sig, fn):
    if num_subcores == 1:
      fn()
    else:
      pl.when(sub == sig % num_subcores)(fn)

  for w in range(waves):
    for sig in range(num_slots):
      owned(sig, functools.partial(issue_slot, sig, w, 0))
    for sig in range(num_slots):

      def staged(sig=sig, w=w):
        n = stage_tab[(2 * num_boards + sig) * waves + w]

        @pl.when(n > 0)
        def _():
          _sc_rows_wait(stage_sems.at[sig * waves + w], stage, n, cap)

        issue_slot(sig, w, 1)

      owned(sig, staged)
  _sc_rows_wait(send_sem, stage, misc[1], cap)


def sc_return_body(
    *,
    me,
    sub,
    tab,
    ret_base,
    contrib,
    recv_ret,
    sc_ret_sems,
    ret_recv_sem,
    ret_send_sem,
    axis,
    ep,
    g_local,
    num_subcores,
):
  """Issues per-expert return pushes on subcore `num_subcores - 1` as TC commits each expert."""
  ret_sub = num_subcores - 1
  ep_g = g_local * ep
  dev_base = ret_base + 1
  src_base = dev_base + ep
  dst_base = src_base + ep_g
  len_base = dst_base + ep_g

  @pl.when(sub == ret_sub)
  def _():
    n_visit = tab[ret_base]

    def visit_step(visit_i, sent_acc):
      pl.semaphore_wait(sc_ret_sems.at[visit_i], 1)
      v_off = visit_i * ep

      def dest_step(i, acc):
        idx = v_off + i
        s_row = tab[src_base + idx]
        d_row = tab[dst_base + idx]
        m = tab[len_base + idx]
        t_dev = tab[dev_base + i]
        is_me = t_dev == me
        src = contrib.at[pl.ds(s_row, m)]
        dst = recv_ret.at[pl.ds(d_row, m)]

        @pl.when(jnp.logical_and(is_me, m > 0))
        def _():
          pltpu.make_async_remote_copy(
              src, dst, ret_send_sem, ret_recv_sem, device_id={TC_AXIS: 0}
          ).start()

        @pl.when(jnp.logical_and(jnp.logical_not(is_me), m > 0))
        def _():
          pltpu.make_async_remote_copy(
              src,
              dst,
              ret_send_sem,
              ret_recv_sem,
              device_id={axis: t_dev, TC_AXIS: 0},
          ).start()

        return acc + jnp.maximum(m, 0)

      return lax.fori_loop(0, ep, dest_step, sent_acc)

    sent_total = lax.fori_loop(0, n_visit, visit_step, jnp.int32(0))

    @pl.when(sent_total > 0)
    def _():
      _sc_rows_wait(ret_send_sem, contrib, sent_total, contrib.shape[0])


def sc_gw_combine_body(
    *,
    me,
    sub,
    send_rem,
    recv_rem,
    sc_comb_sems,
    comb_recv_sem,
    comb_send_sem,
    axis,
    ep,
    board_size,
    t_local,
    tb,
    num_subcores,
):
  """Issues board-striped inter-board optical gateway combine DMAs across all SC subcores."""
  num_boards = ep // board_size
  remote_boards = num_boards - 1
  steps_per_board = t_local // tb
  remote_steps = remote_boards * steps_per_board
  sub_tb = tb // num_subcores
  my_b = lax.div(me, jnp.int32(board_size))
  my_local = lax.rem(me, jnp.int32(board_size))
  sub_off = sub * jnp.int32(sub_tb)

  def comb_step(step, _):
    pl.semaphore_wait(sc_comb_sems.at[step], 1)
    s_step = lax.div(step, jnp.int32(remote_boards))
    peer_idx = lax.rem(step, jnp.int32(remote_boards))
    dst_b = lax.rem(my_b + 1 + peer_idx, jnp.int32(num_boards))
    dst_dev = dst_b * jnp.int32(board_size) + my_local
    src_row = step * jnp.int32(tb) + sub_off
    dst_row = (
        peer_idx * jnp.int32(steps_per_board) + s_step
    ) * jnp.int32(tb) + sub_off
    pltpu.make_async_remote_copy(
        send_rem.at[pl.ds(src_row, sub_tb)],
        recv_rem.at[pl.ds(dst_row, sub_tb)],
        comb_send_sem,
        comb_recv_sem,
        device_id={axis: dst_dev, TC_AXIS: 0},
    ).start()
    return jnp.int32(0)

  lax.fori_loop(0, remote_steps, comb_step, jnp.int32(0))
  _sc_rows_wait(
      comb_send_sem,
      send_rem,
      jnp.int32(remote_steps * sub_tb),
      send_rem.shape[0],
  )


def _sc_forward_kernel(
    tab_hbm,
    fwd_hbm,
    stage,
    recv,
    recv_sems,
    stage_sems,
    send,
    sender_sem,
    *rest,
    waves,
    num_slots,
    num_boards,
    fwd_max_runs,
    axis,
    num_subcores,
    cnt_base,
    tot_base,
    has_return=False,
    ret_base=0,
    ep=0,
    g_local=0,
    has_combine=False,
    board_size=8,
    t_local=0,
    comb_tb=128,
):
  """sc_forward_body (+ optional sc_return_body & sc_gw_combine_body) on SparseCore."""
  absent: Any = None
  it = iter(rest)
  if has_return:
    contrib = next(it)
    recv_ret = next(it)
    sc_ret_sems = next(it)
    ret_recv_sem = next(it)
  else:
    contrib = recv_ret = sc_ret_sems = ret_recv_sem = absent
  if has_combine:
    send_rem = next(it)
    recv_rem = next(it)
    sc_comb_sems = next(it)
    comb_recv_sem = next(it)
  else:
    send_rem = recv_rem = sc_comb_sems = comb_recv_sem = absent
  tab = next(it)
  win = next(it)
  misc = next(it)
  send_sem = next(it)
  ret_send_sem = next(it) if has_return else absent
  comb_send_sem = next(it) if has_combine else absent

  pltpu.sync_copy(tab_hbm, tab)
  me = lax.axis_index(axis)
  sub = lax.axis_index(SC_SUB) if num_subcores > 1 else jnp.int32(0)

  def copy_fn(src, dst, sem, d, is_me):
    dev = {TC_AXIS: 0} if is_me else {axis: d, TC_AXIS: 0}
    pltpu.make_async_remote_copy(src, dst, send_sem, sem, device_id=dev).start()

  sc_forward_body(
      me=me,
      sub=sub,
      stage_tab=tab,
      cnt_tab=tab,
      cnt_base=cnt_base,
      fwd_hbm=fwd_hbm,
      win=win,
      misc=misc,
      stage=stage,
      recv=recv,
      recv_sems=recv_sems,
      stage_sems=stage_sems,
      send_sem=send_sem,
      copy_fn=copy_fn,
      count_local=True,
      waves=waves,
      num_slots=num_slots,
      num_boards=num_boards,
      fwd_max_runs=fwd_max_runs,
      num_subcores=num_subcores,
  )
  if has_return:
    sc_return_body(
        me=me,
        sub=sub,
        tab=tab,
        ret_base=ret_base,
        contrib=contrib,
        recv_ret=recv_ret,
        sc_ret_sems=sc_ret_sems,
        ret_recv_sem=ret_recv_sem,
        ret_send_sem=ret_send_sem,
        axis=axis,
        ep=ep,
        g_local=g_local,
        num_subcores=num_subcores,
    )
  if has_combine:
    sc_gw_combine_body(
        me=me,
        sub=sub,
        send_rem=send_rem,
        recv_rem=recv_rem,
        sc_comb_sems=sc_comb_sems,
        comb_recv_sem=comb_recv_sem,
        comb_send_sem=comb_send_sem,
        axis=axis,
        ep=ep,
        board_size=board_size,
        t_local=t_local,
        tb=comb_tb,
        num_subcores=num_subcores,
    )
  _sc_rows_wait(sender_sem, send, tab[tot_base + sub], send.shape[0])


def sc_forward_start(
    wd,
    *,
    ep,
    num_subcores=NUM_SUBCORES,
    ret_tab=None,
    contrib_buf=None,
    recv_ret_buf=None,
    g_local=0,
    send_rem_buf=None,
    recv_rem_buf=None,
    board_size=8,
    t_local=0,
    comb_tb=128,
):
  """Starts the gateway's K4 forwards (+ optional return pushes & combine) on SparseCore."""
  _ensure_sc_compat()
  fut, smem_ops = wd.fut, wd.smem_ops
  if len(fut) < 11 or smem_ops is None or wd.stage_send_tab is None:
    raise ValueError(
        "sc_forward_start needs sc_ragged_a2a_start_waves(sc_forward=True)"
        " and the gateway tables"
    )
  sender_sem, recv_sems, send, recv = fut[0], fut[1], fut[2], fut[3]
  stage, sc_stage_sems = fut[9], fut[10]
  has_return = ret_tab is not None
  has_combine = send_rem_buf is not None
  f_idx = 11
  if has_return:
    sc_ret_sems, ret_recv_sem = fut[f_idx], fut[f_idx + 1]
    f_idx += 2
  else:
    sc_ret_sems = ret_recv_sem = None
  if has_combine:
    sc_comb_sems, comb_recv_sem = fut[f_idx], fut[f_idx + 1]
  else:
    sc_comb_sems = comb_recv_sem = None
  num_slots, waves, fwd_max_runs, fields = smem_ops.fwd_tab.shape
  if fields != dedup_plan_mod.FWD_FIELDS:
    raise ValueError(f"fwd_tab {smem_ops.fwd_tab.shape}: want 4 fields")
  if sc_stage_sems.shape != (num_slots * waves,):
    raise ValueError(
        f"SC stage sems {sc_stage_sems.shape} != (S * W,) ="
        f" ({num_slots * waves},)"
    )
  num_boards = wd.stage_send_tab.shape[0] // 5
  if num_boards < 2 or ep % num_boards:
    raise ValueError(
        f"De-Dup forwards need ep divisible by num_boards >= 2; got ep={ep},"
        f" num_boards={num_boards}"
    )
  axis = _shard_map_axis()
  stage_flat = smem_ops.stage_tab.astype(jnp.int32).reshape(-1)
  cnt_flat = smem_ops.fwd_cnt.astype(jnp.int32).reshape(-1)
  tot_per_sub = subcore_row_totals(
      wd.disp_tab,
      lax.axis_index(axis),
      ep=ep,
      waves=wd.disp_tab.shape[1],
      num_subcores=num_subcores,
      stage_send_tab=wd.stage_send_tab,
  ).astype(jnp.int32)
  tab_parts = [stage_flat, cnt_flat, tot_per_sub]
  ret_base = stage_flat.shape[0] + cnt_flat.shape[0] + tot_per_sub.shape[0]
  if has_return:
    tab_parts.append(ret_tab.astype(jnp.int32).reshape(-1))
  tab = _pad_words(jnp.concatenate(tab_parts))
  fwd = _pad_words(
      jnp.pad(
          smem_ops.fwd_tab.astype(jnp.int32).reshape(-1), (0, FWD_WIN_WORDS)
      )
  )
  tcm, scm = tc_mesh(), sc_mesh(num_subcores)
  sub_sfx = "" if num_subcores == 1 else f"_sub{num_subcores}"
  orch_sfx = ("_ret" if has_return else "") + ("_comb" if has_combine else "")
  allow_tc_sc_mesh_pair()
  allow_tc_cross_core_semaphores()
  allow_mpmd_array_writes()
  extra_args = ()
  extra_scratch = []
  if has_return:
    extra_args += (
        contrib_buf,
        recv_ret_buf,
        _sem_value(sc_ret_sems, scm),
        _sem_value(ret_recv_sem, tcm),
    )
    extra_scratch.append(pltpu.SemaphoreType.DMA(()))
  if has_combine:
    extra_args += (
        send_rem_buf,
        recv_rem_buf,
        _sem_value(sc_comb_sems, scm),
        _sem_value(comb_recv_sem, tcm),
    )
    extra_scratch.append(pltpu.SemaphoreType.DMA(()))
  pl.kernel(
      functools.partial(
          _sc_forward_kernel,
          waves=waves,
          num_slots=num_slots,
          num_boards=num_boards,
          fwd_max_runs=fwd_max_runs,
          axis=axis,
          num_subcores=num_subcores,
          cnt_base=stage_flat.shape[0],
          tot_base=stage_flat.shape[0] + cnt_flat.shape[0],
          has_return=has_return,
          ret_base=ret_base,
          ep=ep,
          g_local=g_local,
          has_combine=has_combine,
          board_size=board_size,
          t_local=t_local,
          comb_tb=comb_tb,
      ),
      out_type=(),
      mesh=scm,
      scratch_types=[
          pltpu.SMEM((tab.shape[0],), jnp.int32),
          pltpu.SMEM((FWD_WIN_WORDS,), jnp.int32),
          pltpu.SMEM((SMEM_PAD,), jnp.int32),
          pltpu.SemaphoreType.DMA(()),
          *extra_scratch,
      ],
      compiler_params=pltpu.CompilerParams(
          skip_device_barrier=True, has_side_effects=True
      ),
      name=(
          f"cae_sc_forward_ep{ep}_w{waves}_s{num_slots}_r{fwd_max_runs}"
          f"{orch_sfx}{sub_sfx}"
      ),
  )(
      tab,
      fwd,
      stage,
      recv,
      _sem_value(recv_sems, tcm),
      _sem_value(sc_stage_sems, scm),
      send,
      _sem_value(sender_sem, scm),
      *extra_args,
  )


def _sc_drain_kernel(
    tot_hbm, send, send_sem, carry, tot, *, cap, num_subcores=NUM_SUBCORES
):
  """Waits for `tot[sub]` rows (16 KiB each) sent on each subcore's SC send semaphore."""
  del carry  # threaded through: orders the drain after its producer
  pltpu.sync_copy(tot_hbm, tot)
  sub = lax.axis_index(SC_SUB) if num_subcores > 1 else jnp.int32(0)
  total = tot[sub]
  capv = jnp.int32(cap)

  def body(_, c):
    pltpu.make_async_copy(
        send.at[pl.ds(0, cap)], send.at[pl.ds(0, cap)], send_sem
    ).wait()
    return c

  lax.fori_loop(0, lax.div(total, capv), body, jnp.int32(0))
  rem = lax.rem(total, capv)

  @pl.when(rem > 0)
  def _():
    pltpu.make_async_copy(
        send.at[pl.ds(0, rem)], send.at[pl.ds(0, rem)], send_sem
    ).wait()


def _tc_list_drain_kernel(
    list_tab, tg_lists, list_send_sem, carry, carry_out, *, ep, axis
):
  del carry, carry_out
  total = list_tab[ep]

  @pl.when(total > 0)
  def _():
    pltpu.make_async_remote_copy(
        tg_lists.at[pl.ds(0, total)],
        tg_lists.at[pl.ds(0, total)],
        list_send_sem,
        list_send_sem,
        device_id={axis: jnp.int32(0)},
    ).wait_send()


def sc_ragged_a2a_send_done(
    send_sem,
    send,
    disp_tab,
    carry,
    *,
    ep,
    list_tab,
    tg_lists,
    list_send_sem,
    stage_send_tab=None,
    num_subcores=NUM_SUBCORES,
    sc_forward=False,
):
  """Drains the SC sends (SC kernel) and the list sends (TC kernel).

  Each subcore drains its own row total (subcore_row_totals: its direct runs
  and its share of every stage run). `carry` is pinned to HBM. sc_forward:
  the SC forward kernel (sc_forward_start) already drained the SC sends, so
  only the TC list drain runs.

  Args:
    send_sem: The SC sender's send semaphore.
    send: The packed send buffer.
    disp_tab: [4 * ep, W] dispatch table.
    carry: Value threaded through both drains.
    ep: Expert-parallel width.
    list_tab: Token-gather list table.
    tg_lists: Token-gather lists.
    list_send_sem: The list send semaphore.
    stage_send_tab: Optional De-Dup stage table.
    num_subcores: Vector subcores sharing the sends.
    sc_forward: Whether gateway forwards ran on the SparseCore.

  Returns:
    `carry` (threaded through both drains).
  """
  _ensure_sc_compat()
  axis = _shard_map_axis()
  if sc_forward:
    return _tc_list_drain(
        pltpu.with_memory_space_constraint(carry, pltpu.HBM),
        list_tab,
        tg_lists,
        list_send_sem,
        ep=ep,
        axis=axis,
    )
  scm = sc_mesh(num_subcores)
  waves = disp_tab.shape[1]
  if num_subcores == 1 and stage_send_tab is None:
    tot_per_sub = jnp.sum(disp_tab[ep : 2 * ep, :waves].astype(jnp.int32))[None]
  else:
    tot_per_sub = subcore_row_totals(
        disp_tab,
        lax.axis_index(axis),
        ep=ep,
        waves=waves,
        num_subcores=num_subcores,
        stage_send_tab=stage_send_tab,
    )
  carry = pltpu.with_memory_space_constraint(carry, pltpu.HBM)
  carry_ref = jax.new_ref(carry, memory_space=pltpu.HBM)
  ssem_ref = jax.new_ref(send_sem, memory_space=pltpu.SEMAPHORE @ scm)
  sub_sfx = "" if num_subcores == 1 else f"_sub{num_subcores}"
  pl.kernel(
      functools.partial(
          _sc_drain_kernel, cap=send.shape[0], num_subcores=num_subcores
      ),
      mesh=scm,
      scratch_types=[pltpu.SMEM((SMEM_PAD,), jnp.int32)],
      compiler_params=pltpu.CompilerParams(
          has_side_effects=True, skip_device_barrier=True
      ),
      name=f"cae_sc_send_done_ep{ep}{sub_sfx}",
  )(_pad_words(tot_per_sub), send, ssem_ref, carry_ref)
  carry = jax.freeze(carry_ref)
  carry = pltpu.with_memory_space_constraint(carry, pltpu.HBM)
  return _tc_list_drain(
      carry, list_tab, tg_lists, list_send_sem, ep=ep, axis=axis
  )


def _tc_list_drain(carry, list_tab, tg_lists, list_send_sem, *, ep, axis):
  """The TC list-send drain of sc_ragged_a2a_send_done (threads `carry`)."""
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  out = pl.pallas_call(
      functools.partial(_tc_list_drain_kernel, ep=ep, axis=axis),
      out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
      in_specs=[smem, anyspec, semspec, anyspec],
      out_specs=anyspec,
      input_output_aliases={3: 0},
      name=f"cae_list_done_ep{ep}",
  )(list_tab.astype(jnp.int32), tg_lists, list_send_sem, carry)
  return pltpu.with_memory_space_constraint(out, pltpu.HBM)


def _layer_module():
  """Lazy import of `layer.monolithic` to avoid any import-order coupling."""
  return importlib.import_module(
      "tokamax._src.ops.experimental.dedup_moe.layer.monolithic"
  )


def clear_layer_cache() -> None:
  """Drops layer/monolithic.py's cached jitted shard_maps.

  dedup_ep_moe caches its jitted shard_map under the layer's arguments;
  a trace made before a patch (or with another patch) would otherwise be
  reused, silently running the unpatched / another variant's program.
  """
  layer = _layer_module()
  with layer._LAYER_SM_CACHE_LOCK:  # pylint: disable=protected-access
    layer._LAYER_SM_CACHE.clear()  # pylint: disable=protected-access


@contextlib.contextmanager
def use_sc_dispatch(
    gather=False,
    num_subcores=NUM_SUBCORES,
    sc_forward=False,
    sc_return=False,
    sc_combine=False,
    stage_all_remote=False,
):
  """Routes wave_dispatch's start/drain through the SparseCore.

  sc_forward: the De-Dup gateway forwards too (sc_forward_start; the layer
  needs dedup_sc_forward=True, which wrap_layer sets).
  sc_return: also issue per-expert return pushes on SparseCore.
  sc_combine: also issue gateway combine optical sends on SparseCore.
  stage_all_remote: whether all remote-board tokens are staged (min_fanout <=
    1), so direct sends only visit intra-board peers.

  Args:
    gather: Use the per-row gather send.
    num_subcores: Vector subcores.
    sc_forward: Also issue the De-Dup gateway forwards from the SparseCore.
    sc_return: Also issue per-expert return pushes from the SparseCore.
    sc_combine: Also issue gateway combine optical sends from the SparseCore.
    stage_all_remote: Whether all remote-board tokens are staged.

  Yields:
    Nothing; the SparseCore dispatch is active inside the context.
  """
  layer = _layer_module()
  saved = (
      dispatch_kernels.ragged_a2a_start_waves,
      dispatch_kernels.ragged_a2a_send_done,
      layer._sc_forward_start,  # pylint: disable=protected-access
  )
  start_kw = {}
  done_kw = {}
  if sc_forward:
    start_kw["sc_forward"] = True
    done_kw["sc_forward"] = True
  if sc_return:
    start_kw["sc_return"] = True
  if sc_combine:
    start_kw["sc_combine"] = True
  if stage_all_remote:
    start_kw["stage_all_remote"] = True
  clear_layer_cache()
  dispatch_kernels.ragged_a2a_start_waves = functools.partial(
      sc_ragged_a2a_start_waves,
      gather=gather,
      num_subcores=num_subcores,
      **start_kw,
  )
  dispatch_kernels.ragged_a2a_send_done = functools.partial(
      sc_ragged_a2a_send_done,
      num_subcores=num_subcores,
      **done_kw,
  )
  if sc_forward:
    layer._sc_forward_start = functools.partial(  # pylint: disable=protected-access
        sc_forward_start, num_subcores=num_subcores
    )
  try:
    yield
  finally:
    (
        dispatch_kernels.ragged_a2a_start_waves,
        dispatch_kernels.ragged_a2a_send_done,
        layer._sc_forward_start,  # pylint: disable=protected-access
    ) = saved
    clear_layer_cache()


def wrap_layer(
    fn,
    gather=False,
    num_subcores=NUM_SUBCORES,
    sc_forward=False,
    sc_return=False,
    sc_combine=False,
):
  """Wraps a layer function so it traces with the SparseCore dispatch.

  Args:
    fn: The layer function, such as layer.monolithic.dedup_ep_moe.
    gather: Use the per-row gather send.
    num_subcores: Vector subcores.
    sc_forward: Also issue the De-Dup gateway forwards from the SparseCore (sets
      the layer's dedup_sc_forward=True).
    sc_return: Also issue per-expert return pushes from the SparseCore (sets
      dedup_sc_return=True).
    sc_combine: Also issue gateway combine optical sends from the SparseCore
      (sets dedup_sc_combine=True).

  Returns:
    A function with `fn`'s signature.
  """

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    if sc_forward or sc_return or sc_combine:
      kwargs = dict(kwargs)
      if sc_forward:
        kwargs.setdefault("dedup_sc_forward", True)
      if sc_return:
        kwargs.setdefault("dedup_sc_return", True)
      if sc_combine:
        kwargs.setdefault("dedup_sc_combine", True)
    stage_all_remote = int(kwargs.get("dedup_min_fanout", 2)) <= 1
    with use_sc_dispatch(
        gather=gather,
        num_subcores=num_subcores,
        sc_forward=sc_forward,
        sc_return=sc_return,
        sc_combine=sc_combine,
        stage_all_remote=stage_all_remote,
    ):
      return fn(*args, **kwargs)

  return wrapped
