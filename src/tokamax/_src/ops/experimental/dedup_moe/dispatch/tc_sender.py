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
"""A wave-major ragged all-to-all as a pair of split-phase Pallas kernels.

XLA builds an async ragged all-to-all, but its scheduler places start and done
back to back and `AsyncCollectiveMerger` turns the pair back into one sync op,
so nothing can run under the exchange. Splitting it into custom calls -- a
start kernel that only ISSUES the remote DMAs and returns their semaphores, and
consumers that wait on them -- puts the overlap under the caller's control
(see the Pallas TPU async design note in the JAX documentation).

The exchange is issued in `waves` groups: the caller lays the dispatch plan out
by expert group, and the start kernel issues group 0 to every destination, then
group 1, and so on. Every row of group w lands on the receiver's semaphore
`recv_sems[w]`, so a receiver can start computing on group 0
while later groups are still in flight. The fused MoE kernel does exactly that:
it takes the receive semaphores as an operand and waits for group w only before
its first expert in group w.

Table: the [4 * ep, waves] int32 dispatch plan. Rows (k * ep + d, w): k=0 send
offset, k=1 send rows, k=2 offset in d's receive buffer; row 3 * ep holds the
rows this shard receives per group, row 3 * ep + 1 col 0 the total rows it
sends to OTHER shards. A sender signals the receiver's semaphore for group w,
so the receiver's count for group w is the sum over sources -- exactly row
3 * ep -- and its waits are byte-exact (every semaphore returns to zero).

Invariants:
  * Senders only `wait_send` (never `wait`), once, after the consumer kernel
    (`ragged_a2a_send_done`): every outgoing DMA had the whole kernel to
    finish, and `wait_send` never blocks on a peer.
  * The start kernel opens with an all-pairs barrier, so no shard writes a
    peer's receive buffer before that peer is executing the same program
    point (its buffer is live there; SPMD buffer assignment is identical).
  * `send` and `recv` are threaded through the kernels with
    input_output_aliases, and the send drain threads a value the caller uses,
    so XLA can neither free, reuse nor dead-code-eliminate them while DMAs
    are in flight.
  * Every exchange buffer is pinned to HBM, operands and outputs both, so
    memory-space assignment cannot stage it through VMEM around these calls.

With De-Dup, the start kernel also packs and sends each (source, remote
board) stage run to that board's gateway device; boards are contiguous
blocks of `board_size` devices on the mesh axis.
"""

import functools
import itertools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan

# Distinct from the fused kernel's COLLECTIVE_ID (0): its own barrier semaphore.
A2A_COLLECTIVE_ID = 1
ROUTING_AG_COLLECTIVE_ID = 3


def _all_pairs_barrier(ep):
  sem = pltpu.get_barrier_semaphore()
  for i in range(ep):
    pl.semaphore_signal(sem, inc=1, device_id=(jnp.int32(i),))
  pl.semaphore_wait(sem, ep)


def _head_tail_piece(n, cs, head):
  """Rows [lo, lo + m) of an n-row run sent in this pass.

  The head pass sends the first `cs` rows and the tail pass the rest.

  Args:
    n: Rows of the run.
    cs: Rows of the head piece (chunk_rows).
    head: Whether this is the head pass.

  Returns:
    (lo, m): the first row and the row count of this pass's piece.
  """
  lo = jnp.int32(0) if head else cs
  m = jnp.minimum(cs, n) if head else n - cs
  return lo, m


DIRECT_ORDERS = ("ring", "board_stripe", "board_stripe_split")
# De-Dup stage-send placement (EXPERIMENTAL): "none" issues each (wave,
# head/tail) pass's stage runs at the head of that pass (default); "all" packs
# and issues every wave's stage runs before any direct run; "w01" hoists waves
# 0 and 1 only.
STAGE_HOISTS = ("none", "all", "w01")
# Pack semaphores of the packed start kernel: 1 packs and sends each step in
# turn on one semaphore; 2 packs step i + 1 on the other parity while step i
# waits and sends.
PACK_PARITIES = (1, 2)


def _direct_dest_groups(
    me, *, ep, direct_order, stripe_boards, stripe_board_size
):
  """The direct-send destinations, as groups packed and issued in order.

  Each entry is (d, is_local): d the destination shard (traced) and is_local
  whether d == me is known statically (a local copy instead of a remote DMA).

    ring:               one group, d = (me + i) % ep (XLA a2a's rotation).
    board_stripe:       one group: the remote boards first, striped board-major
                        (remote board inner, slot offset outer) so every
                        (my board -> remote board) link gets a DMA early, then
                        the intra-board peers, own shard first.
    board_stripe_split: the same order as two groups, remote then intra-board,
                        so the remote rows are packed and issued before the
                        intra-board rows are packed.

  Args:
    me: This shard's index (traced).
    ep: Expert-parallel width.
    direct_order: One of DIRECT_ORDERS.
    stripe_boards: Boards (board-major orders).
    stripe_board_size: Devices per board (board-major orders).

  Returns:
    A list of groups, each a list of (d, is_local).
  """
  if direct_order == "ring":
    return [[(lax.rem(me + i, ep), i == 0) for i in range(ep)]]
  nb, bs = stripe_boards, stripe_board_size
  my_b = me // bs
  my_s = lax.rem(me, bs)
  remote = [
      (lax.rem(my_b + bi, nb) * bs + lax.rem(my_s + so, bs), False)
      for so, bi in itertools.product(range(bs), range(1, nb))
  ]
  intra = [(my_b * bs + lax.rem(my_s + i, bs), i == 0) for i in range(bs)]
  if direct_order == "board_stripe":
    return [remote + intra]
  return [remote, intra]


def _check_direct_order(direct_order, ep, stripe_boards, stripe_board_size):
  if direct_order not in DIRECT_ORDERS:
    raise ValueError(f"direct_order must be one of {DIRECT_ORDERS}")
  if direct_order != "ring" and (
      stripe_boards < 2
      or stripe_board_size < 1
      or stripe_boards * stripe_board_size != ep
  ):
    raise ValueError(
        f"direct_order={direct_order!r} needs >= 2 boards that tile ep={ep};"
        f" got {stripe_boards} x {stripe_board_size}"
    )


def _start_waves_kernel(
    tab,
    me_ref,
    send_in,
    recv_in,
    *refs,
    ep,
    waves,
    chunk_rows,
    list_rows,
    has_stage=False,
    num_boards=0,
    board_size=0,
    row_send=False,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
    stage_hoist="none",
    pack_parity=1,
):
  """Start kernel body: pack and issue the lists, stage runs and waves."""
  del send_in, recv_in  # aliased to send_out / recv_out
  refs = list(refs)
  perm_sm = refs.pop(0)
  rows = refs.pop(0)
  list_tab = refs.pop(0)
  tg_lists_in = refs.pop(0)
  tg_land_in = refs.pop(0)
  del tg_lists_in, tg_land_in
  if has_stage:
    stage_send_tab = refs.pop(0)
    stage_in = refs.pop(0)
    del stage_in
  else:
    stage_send_tab = None
  send_sem = refs.pop(0)
  recv_sems = refs.pop(0)
  send_out = refs.pop(0)
  recv_out = refs.pop(0)
  list_send_sem = refs.pop(0)
  tg_sem = refs.pop(0)
  tg_lists_out = refs.pop(0)
  tg_land_out = refs.pop(0)
  if has_stage:
    stage_sems = refs.pop(0)
    stage_out = refs.pop(0)
  else:
    stage_sems = stage_out = None
  if row_send:
    refs.pop(0)  # rows, aliased: kept alive for the drain kernel
  pack_sem = refs.pop(0)  # [2] DMA semaphores under pack_parity == 2
  assert not refs

  me = me_ref[0]
  ep_is_pow2 = (ep & (ep - 1)) == 0

  def peer_id(i):
    if ep_is_pow2:
      return jnp.bitwise_and(me + i, jnp.int32(ep - 1))
    return lax.rem(me + i, ep)

  _all_pairs_barrier(ep)
  # Issue the token-gather list DMAs before wave 0's activation rows so the
  # lists land before the rows that the receiver's token gather indexes.
  for i in range(ep):
    d = peer_id(i)
    n_list = list_tab[d]
    src_list = tg_lists_out.at[pl.ds(d * list_rows, n_list)]
    dst_list = tg_land_out.at[pl.ds(me * list_rows, n_list)]
    if i == 0:

      @pl.when(n_list > 0)
      def _(src_list=src_list, dst_list=dst_list):
        pltpu.make_async_copy(src_list, dst_list, tg_sem).start()

    else:

      @pl.when(n_list > 0)
      def _(src_list=src_list, dst_list=dst_list, d=d):
        pltpu.make_async_remote_copy(
            src_ref=src_list,
            dst_ref=dst_list,
            send_sem=list_send_sem,
            recv_sem=tg_sem,
            device_id=(d,),
        ).start()

  cs = jnp.int32(chunk_rows)

  def pack_rows(base, m, psem=None):
    """Pack send rows [base, base + m) from `rows` by `perm` on `psem`."""
    psem = pack_sem if psem is None else psem

    def _pack_one(r, _):
      idx = base + r
      pltpu.make_async_copy(
          rows.at[pl.ds(perm_sm[idx], 1)],
          send_out.at[pl.ds(idx, 1)],
          psem,
      ).start()
      return jnp.int32(0)

    lax.fori_loop(0, m, _pack_one, jnp.int32(0))

  def pack_stage(w, head, psem=None):
    # Pack the De-Dup stage runs of this (w, head) pass into send_out
    # and returns the rows started on `psem`.
    assert stage_send_tab is not None
    m_stage = jnp.int32(0)
    for bi in range(1, num_boards):
      b = lax.rem(me // board_size + bi, num_boards)
      n = stage_send_tab[num_boards + b, w]
      lo, m = _head_tail_piece(n, cs, head)
      m = jnp.maximum(jnp.int32(0), m)
      pack_rows(stage_send_tab[b, w] + lo, m, psem)
      m_stage = m_stage + m
    return m_stage

  def pack_direct(w, head, dests=None, psem=None):
    # Wave-pipelined local pack of this (w, head) pass's direct runs to
    # `dests` (default: the ring) into send_out, so no XLA `rows[perm]`
    # gather runs before the kernel, and returns the rows started on
    # `psem`. Packed positions come from the table, so the destination
    # order changes no data.
    m_dir = jnp.int32(0)
    for i in range(ep if dests is None else len(dests)):
      d = peer_id(i) if dests is None else dests[i][0]
      n = tab[ep + d, w]
      lo, m = _head_tail_piece(n, cs, head)
      m = jnp.maximum(jnp.int32(0), m)
      pack_rows(tab[d, w] + lo, m, psem)
      m_dir = m_dir + m
    return m_dir

  def wait_pack(m_rows, psem=None):
    psem = pack_sem if psem is None else psem

    @pl.when(m_rows > 0)
    def _():
      pltpu.make_async_copy(
          send_out.at[pl.ds(0, m_rows)],
          send_out.at[pl.ds(0, m_rows)],
          psem,
      ).wait()

  def send_stage(w, head):
    # Issue the De-Dup stage runs to the remote gateways. A stage run never
    # targets this shard's own board (`b != me // board_size`), so every
    # stage DMA is a remote copy on `send_sem` signalling
    # `stage_sems.at[sigma * W + w]` on gateway `gw`; the head and tail
    # pieces share it.
    assert stage_send_tab is not None
    assert stage_out is not None
    assert stage_sems is not None
    for bi in range(1, num_boards):
      b = lax.rem(me // board_size + bi, num_boards)
      n = stage_send_tab[num_boards + b, w]
      gw = stage_send_tab[3 * num_boards + b, w]
      sig = stage_send_tab[4 * num_boards + b, w]
      lo, m = _head_tail_piece(n, cs, head)
      src = send_out.at[pl.ds(stage_send_tab[b, w] + lo, m)]
      dst = stage_out.at[pl.ds(stage_send_tab[2 * num_boards + b, w] + lo, m)]

      @pl.when(m > 0)
      def _(src=src, dst=dst, gw=gw, sig=sig, w=w):
        pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst,
            send_sem=send_sem,
            recv_sem=stage_sems.at[sig * waves + w],
            device_id=(gw,),
        ).start()

  def send_direct(w, head, dests=None):
    # Destination order rotated by `me` (as XLA's own a2a does) so the
    # shards do not all hit peer 0 first, or `dests` (_direct_dest_groups).
    for i in range(ep if dests is None else len(dests)):
      if dests is None:
        d, is_local = peer_id(i), i == 0
      else:
        d, is_local = dests[i]
      n = tab[ep + d, w]
      lo, m = _head_tail_piece(n, cs, head)
      src = send_out.at[pl.ds(tab[d, w] + lo, m)]
      dst = recv_out.at[pl.ds(tab[2 * ep + d, w] + lo, m)]
      if is_local:  # d == me: a local copy on this shard's own slot

        @pl.when(m > 0)
        def _(src=src, dst=dst, w=w):
          pltpu.make_async_copy(src, dst, recv_sems.at[w]).start()

      else:

        @pl.when(m > 0)
        def _(src=src, dst=dst, d=d, w=w):
          pltpu.make_async_remote_copy(
              src_ref=src,
              dst_ref=dst,
              send_sem=send_sem,
              recv_sem=recv_sems.at[w],
              device_id=(d,),
          ).start()

  t_local = rows.shape[0]
  per_dest_perm = perm_sm.ndim == 2

  def send_direct_rows(w, head=None, dests=None):
    # Row send: each row goes from `rows` straight to its receive row, with
    # no pack into send_out and no wait for the pack. Same rows, same
    # receive rows and the same semaphore byte counts as pack + send.
    # Destinations in ring order, or `dests` (_direct_dest_groups).
    for i in range(ep if dests is None else len(dests)):
      if dests is None:
        d, is_local = peer_id(i), i == 0
      else:
        d, is_local = dests[i]
      n = tab[ep + d, w]
      if head is None:
        lo = jnp.int32(0)
        m = jnp.maximum(jnp.int32(0), n)
      else:
        lo, m = _head_tail_piece(n, cs, head)
        m = jnp.maximum(jnp.int32(0), m)
      dst0 = tab[2 * ep + d, w] + lo
      base = (dst0 - me * t_local) if per_dest_perm else (tab[d, w] + lo)
      if is_local:

        def local_row(r, c, base=base, dst0=dst0, d=d, w=w):
          tok = perm_sm[d, base + r] if per_dest_perm else perm_sm[base + r]
          pltpu.make_async_copy(
              rows.at[pl.ds(tok, 1)],
              recv_out.at[pl.ds(dst0 + r, 1)],
              recv_sems.at[w],
          ).start()
          return c

        lax.fori_loop(0, m, local_row, jnp.int32(0))
      else:

        def remote_row(r, c, base=base, dst0=dst0, d=d, w=w):
          tok = perm_sm[d, base + r] if per_dest_perm else perm_sm[base + r]
          pltpu.make_async_remote_copy(
              src_ref=rows.at[pl.ds(tok, 1)],
              dst_ref=recv_out.at[pl.ds(dst0 + r, 1)],
              send_sem=send_sem,
              recv_sem=recv_sems.at[w],
              device_id=(d,),
          ).start()
          return c

        lax.fori_loop(0, m, remote_row, jnp.int32(0))

  if row_send:
    assert not has_stage
    row_dests = None
    if direct_order != "ring":
      row_dests = [
          dest
          for group in _direct_dest_groups(
              me,
              ep=ep,
              direct_order=direct_order,
              stripe_boards=stripe_boards,
              stripe_board_size=stripe_board_size,
          )
          for dest in group
      ]
    for w in range(waves):
      if t_local <= 96:
        send_direct_rows(w, dests=row_dests)
      else:
        for head in (True, False):
          send_direct_rows(w, head, dests=row_dests)
    return

  hoist = has_stage and stage_hoist != "none"
  if pack_parity == 1 and direct_order == "ring" and not hoist:
    for w in range(waves):
      # The first chunk_rows rows of every destination, then the rest of
      # each destination's run in one copy: every peer's first piece is in
      # flight early, and a wave costs at most 2 * ep copies however
      # skewed it is (chunking every run cost a loop body per chunk of the
      # busiest destination, times ep).
      for head in (True, False):
        # Stage rows take 2 hops (sender -> gateway -> target), so each
        # (w, head) pass packs and issues its cross-board stage runs
        # FIRST, at the head of the DMA queue, and only then the ep 1-hop
        # direct runs.
        if has_stage:
          wait_pack(pack_stage(w, head))
          send_stage(w, head)
        wait_pack(pack_direct(w, head))
        send_direct(w, head)
    return

  # The same passes as a list of (pack, send) steps, with the De-Dup stage
  # steps of the hoisted waves first and the direct runs in `dest_groups`.
  if direct_order == "ring":
    dest_groups = [None]
  else:
    dest_groups = _direct_dest_groups(
        me,
        ep=ep,
        direct_order=direct_order,
        stripe_boards=stripe_boards,
        stripe_board_size=stripe_board_size,
    )
  if not hoist:
    hoisted = ()
  elif stage_hoist == "all":
    hoisted = tuple(range(waves))
  else:
    hoisted = tuple(range(min(2, waves)))
  steps = []
  for w in hoisted:
    for head in (True, False):
      steps.append((
          functools.partial(pack_stage, w, head),
          functools.partial(send_stage, w, head),
      ))
  for w in range(waves):
    for head in (True, False):
      if has_stage and w not in hoisted:
        steps.append((
            functools.partial(pack_stage, w, head),
            functools.partial(send_stage, w, head),
        ))
      for dests in dest_groups:
        steps.append((
            functools.partial(pack_direct, w, head, dests),
            functools.partial(send_direct, w, head, dests),
        ))
  if pack_parity == 2:
    # Pack step i + 1 on the other parity's semaphore while step i waits
    # for its pack and sends.
    m_cur = steps[0][0](psem=pack_sem.at[0])
    for i, (_, send_fn) in enumerate(steps):
      par = i % 2
      m_nxt = (
          steps[i + 1][0](psem=pack_sem.at[1 - par])
          if i + 1 < len(steps)
          else jnp.int32(0)
      )
      wait_pack(m_cur, pack_sem.at[par])
      send_fn()
      m_cur = m_nxt
  else:
    for pack_fn, send_fn in steps:
      wait_pack(pack_fn())
      send_fn()


def ragged_a2a_start_waves(
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
    board_size=0,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
    stage_hoist="none",
    pack_parity=1,
):
  """Issue the wave-major exchange; see `_start_waves_call`."""
  return _start_waves_call(
      send,
      recv,
      disp_tab,
      me,
      ep=ep,
      waves=waves,
      chunk_rows=chunk_rows,
      rows=rows,
      perm=perm,
      list_tab=list_tab,
      tg_lists=tg_lists,
      tg_land=tg_land,
      stage_send_tab=stage_send_tab,
      stage=stage,
      num_boards=num_boards,
      num_slots=num_slots,
      board_size=board_size,
      direct_order=direct_order,
      stripe_boards=stripe_boards,
      stripe_board_size=stripe_board_size,
      stage_hoist=stage_hoist,
      pack_parity=pack_parity,
  )


def ragged_a2a_start_waves_row_send(
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
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
):
  """`ragged_a2a_start_waves` with row send (direct dispatch only).

  Every row is DMA'd from `rows` straight to its receive row, skipping the
  pack into `send` and its wait. Returns the `fut` tuple plus `rows` as one
  more element, which goes to `ragged_a2a_send_done_row_send`.
  """
  return _start_waves_call(
      send,
      recv,
      disp_tab,
      me,
      ep=ep,
      waves=waves,
      chunk_rows=chunk_rows,
      rows=rows,
      perm=perm,
      list_tab=list_tab,
      tg_lists=tg_lists,
      tg_land=tg_land,
      row_send=True,
      direct_order=direct_order,
      stripe_boards=stripe_boards,
      stripe_board_size=stripe_board_size,
  )


def _start_waves_fast_kernel(
    *refs,
    ep,
    g_local,
    t_local,
    topk,
    waves,
    nw,
    list_rows,
    has_scale,
    has_wave_of,
    has_wb_in,
    compute_visit_me,
    has_wave_bounds,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
):
  """Fused SMEM plan + send-list build + row-send start kernel body."""
  it = iter(refs)
  me_ref = next(it)
  send_in = next(it)
  recv_in = next(it)
  rows = next(it)
  tg_lists_in = next(it)
  tg_land_in = next(it)
  del send_in, recv_in, tg_lists_in, tg_land_in
  pairs_me_sm = next(it)
  hist_me_sm = next(it)
  recv_cnt_sm = next(it)
  scale_bits_sm = next(it) if has_scale else None
  wave_of_sm = next(it) if has_wave_of else None
  wave_bounds_me_sm = next(it) if has_wb_in else None
  rows_by_dest_me_sm = next(it) if compute_visit_me else None
  send_sem = next(it)
  recv_sems = next(it)
  send_out = next(it)
  recv_out = next(it)
  del send_out
  list_send_sem = next(it)
  tg_sem = next(it)
  tg_lists_out = next(it)
  tg_land_out = next(it)
  rows_keep = next(it)
  del rows_keep
  disp_tab_out = next(it)
  list_tab_out = next(it)
  visit_me_out = next(it) if compute_visit_me else None
  rest = list(it)
  pack_sem = rest.pop()
  perm_sm = rest.pop(0)

  start_list_dmas, wait_list_dmas = source_plan.emit_decode_plan_and_lists_body(
      me_ref=me_ref,
      pairs_me_sm=pairs_me_sm,
      hist_me_sm=hist_me_sm,
      recv_cnt_sm=recv_cnt_sm,
      scale_bits_sm=scale_bits_sm,
      wave_of_sm=wave_of_sm,
      wave_bounds_me_sm=wave_bounds_me_sm,
      rows_by_dest_me_sm=rows_by_dest_me_sm,
      disp_tab_sm=disp_tab_out,
      perm_sm=perm_sm,
      list_tab_sm=list_tab_out,
      visit_me_sm=visit_me_out,
      tg_lists_hbm=tg_lists_out,
      scratch_refs=rest,
      dma_sem=pack_sem,
      ep=ep,
      g_local=g_local,
      t_local=t_local,
      topk=topk,
      waves=waves,
      nw=nw,
      list_rows_per_dest=list_rows,
      per_dest_perm=True,
      write_headers=False,
      has_wave_bounds=has_wave_bounds,
  )
  # Overlap local SMEM -> HBM list copies with the all-pairs ICI barrier.
  start_list_dmas()
  _all_pairs_barrier(ep)
  wait_list_dmas()

  me = me_ref[0]
  ep_is_pow2 = (ep & (ep - 1)) == 0

  def peer_id(i):
    if ep_is_pow2:
      return jnp.bitwise_and(me + i, jnp.int32(ep - 1))
    return lax.rem(me + i, ep)

  for i in range(ep):
    d = peer_id(i)
    n_list = list_tab_out[d]
    src_list = tg_lists_out.at[pl.ds(d * list_rows, n_list)]
    dst_list = tg_land_out.at[pl.ds(me * list_rows, n_list)]
    if i == 0:

      @pl.when(n_list > 0)
      def _(src_list=src_list, dst_list=dst_list):
        pltpu.make_async_copy(src_list, dst_list, tg_sem).start()

    else:

      @pl.when(n_list > 0)
      def _(src_list=src_list, dst_list=dst_list, d=d):
        pltpu.make_async_remote_copy(
            src_ref=src_list,
            dst_ref=dst_list,
            send_sem=list_send_sem,
            recv_sem=tg_sem,
            device_id=(d,),
        ).start()

  me_base = me * jnp.int32(t_local)
  row_dests = None
  if direct_order != "ring":
    row_dests = [
        dest
        for group in _direct_dest_groups(
            me,
            ep=ep,
            direct_order=direct_order,
            stripe_boards=stripe_boards,
            stripe_board_size=stripe_board_size,
        )
        for dest in group
    ]
  for w in range(waves):
    for i in range(ep):
      if row_dests is None:
        d, is_local = peer_id(i), i == 0
      else:
        d, is_local = row_dests[i]
      n = disp_tab_out[ep + d, w]
      m = jnp.maximum(jnp.int32(0), n)
      dst0 = disp_tab_out[2 * ep + d, w]
      base = jnp.int32(0) if waves == 1 else (dst0 - me_base)
      if is_local:

        def local_row(r, c, base=base, dst0=dst0, d=d, w=w):
          tok = perm_sm[d, base + r]
          pltpu.make_async_copy(
              rows.at[pl.ds(tok, 1)],
              recv_out.at[pl.ds(dst0 + r, 1)],
              recv_sems.at[w],
          ).start()
          return c

        lax.fori_loop(0, m, local_row, jnp.int32(0))
      else:

        def remote_row(r, c, base=base, dst0=dst0, d=d, w=w):
          tok = perm_sm[d, base + r]
          pltpu.make_async_remote_copy(
              src_ref=rows.at[pl.ds(tok, 1)],
              dst_ref=recv_out.at[pl.ds(dst0 + r, 1)],
              send_sem=send_sem,
              recv_sem=recv_sems.at[w],
              device_id=(d,),
          ).start()
          return c

        lax.fori_loop(0, m, remote_row, jnp.int32(0))


def ragged_a2a_start_waves_row_send_fast(
    send,
    recv,
    rows,
    tg_lists,
    tg_land,
    pairs_me,
    hist_me,
    recv_cnt,
    me,
    *,
    ep,
    g_local,
    waves,
    scale_bits=None,
    wave_of=None,
    wave_bounds_me=None,
    rows_by_dest_me=None,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
):
  """Fuses decode SMEM plan + send-list build into the row-send start kernel.

  Builds `disp_tab`, scratch `perm`, `list_tab`, packed send lists (DMA'd to
  `tg_lists` during the barrier), and (when `waves == 1` with `rows_by_dest_me`)
  `visit_me`, then launches list and direct-row DMAs directly from SMEM.

  `direct_order` (DIRECT_ORDERS, with `stripe_boards` x `stripe_board_size`
  tiling ep) orders the row DMAs' destinations; same data.

  Returns:
    `(fut, rows_keep, disp_tab, list_tab, visit_me)`.

  Raises:
    ValueError: If the arguments are inconsistent.
  """
  _check_direct_order(direct_order, ep, stripe_boards, stripe_board_size)
  t_local, topk = pairs_me.shape
  nw = 1 if scale_bits is None else 2
  list_rows = tg_lists.shape[0] // ep
  compute_visit_me = rows_by_dest_me is not None
  if compute_visit_me and waves != 1:
    raise ValueError("rows_by_dest_me visit_me fast path requires waves == 1")
  if waves > 1 and wave_of is None:
    raise ValueError("wave_of is required when waves > 1")
  has_wave_bounds = (
      (wave_bounds_me is not None or compute_visit_me) and ep >= 3
  )

  hbm = functools.partial(
      pltpu.with_memory_space_constraint, memory_space=pltpu.HBM
  )
  send, recv, rows = hbm(send), hbm(recv), hbm(rows)
  tg_lists, tg_land = hbm(tg_lists), hbm(tg_land)

  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)

  args = [
      jnp.reshape(jnp.asarray(me, jnp.int32), (1,)),
      send,
      recv,
      rows,
      tg_lists,
      tg_land,
      pairs_me.astype(jnp.int32),
      hist_me.astype(jnp.int32).reshape(ep * g_local),
      recv_cnt.astype(jnp.int32).reshape(waves),
  ]
  in_specs = [
      smem,
      anyspec,
      anyspec,
      anyspec,
      anyspec,
      anyspec,
      smem,
      smem,
      smem,
  ]
  has_scale = scale_bits is not None
  if has_scale:
    args.append(scale_bits.astype(jnp.int32))
    in_specs.append(smem)
  has_wave_of = waves > 1
  if has_wave_of:
    assert wave_of is not None
    args.append(wave_of.astype(jnp.int32))
    in_specs.append(smem)
  has_wb_in = wave_bounds_me is not None and not compute_visit_me
  if has_wb_in:
    assert wave_bounds_me is not None
    args.append(wave_bounds_me.astype(jnp.int32).reshape(waves))
    in_specs.append(smem)
  if compute_visit_me:
    assert rows_by_dest_me is not None
    args.append(rows_by_dest_me.astype(jnp.int32))
    in_specs.append(smem)

  aliases = {1: 2, 2: 3, 4: 6, 5: 7, 3: 8}
  out_shape = [
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA((waves,)),
      jax.ShapeDtypeStruct(send.shape, send.dtype),
      jax.ShapeDtypeStruct(recv.shape, recv.dtype),
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA(()),
      jax.ShapeDtypeStruct(tg_lists.shape, tg_lists.dtype),
      jax.ShapeDtypeStruct(tg_land.shape, tg_land.dtype),
      jax.ShapeDtypeStruct(rows.shape, rows.dtype),
      jax.ShapeDtypeStruct((4 * ep, waves), jnp.int32),
      jax.ShapeDtypeStruct((ep + 1,), jnp.int32),
  ]
  out_specs = [
      semspec,
      semspec,
      anyspec,
      anyspec,
      semspec,
      semspec,
      anyspec,
      anyspec,
      anyspec,
      smem,
      smem,
  ]
  if compute_visit_me:
    out_shape.append(jax.ShapeDtypeStruct((g_local,), jnp.int32))
    out_specs.append(smem)

  scratch_shapes = (
      (pltpu.SMEM((ep, t_local), jnp.int32),)
      + source_plan.decode_fast_plan_scratch_shapes(
          ep=ep,
          g_local=g_local,
          t_local=t_local,
          topk=topk,
          waves=waves,
          nw=nw,
          per_dest_perm=True,
          write_headers=False,
          compute_visit_me=compute_visit_me,
      )
      + (pltpu.SemaphoreType.DMA(()),)
  )

  outs = pl.pallas_call(
      functools.partial(
          _start_waves_fast_kernel,
          ep=ep,
          g_local=g_local,
          t_local=t_local,
          topk=topk,
          waves=waves,
          nw=nw,
          list_rows=list_rows,
          has_scale=has_scale,
          has_wave_of=has_wave_of,
          has_wb_in=has_wb_in,
          compute_visit_me=compute_visit_me,
          has_wave_bounds=has_wave_bounds,
          direct_order=direct_order,
          stripe_boards=stripe_boards,
          stripe_board_size=stripe_board_size,
      ),
      out_shape=tuple(out_shape),
      in_specs=in_specs,
      out_specs=tuple(out_specs),
      scratch_shapes=scratch_shapes,
      input_output_aliases=aliases,
      compiler_params=pltpu.CompilerParams(collective_id=A2A_COLLECTIVE_ID),
      name=(
          f"wave_a2a_start_ep{ep}_w{waves}_rows_fast"
          f"{'' if direct_order == 'ring' else '_' + direct_order}"
      ),
  )(*args)
  fut = tuple(outs[:8])
  rows_keep = outs[8]
  disp_tab = outs[9]
  list_tab = outs[10]
  visit_me = outs[11] if compute_visit_me else None
  return fut, rows_keep, disp_tab, list_tab, visit_me


def _start_waves_call(
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
    board_size=0,
    row_send=False,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
    stage_hoist="none",
    pack_parity=1,
):
  """Issue the wave-major exchange, its token-gather lists and De-Dup stage.

  send [rows, ...] is an uninitialized send buffer that the kernel packs
  from `rows` by `perm`, wave by wave; recv (uninitialized HBM) is the
  receive buffer; disp_tab is the [4 * ep, waves] plan described in the
  module docstring and me this shard's index. Each (me -> d) token-gather
  list goes into `tg_land` on `tg_sem`. When `stage_send_tab` and `stage`
  are supplied (De-Dup), also packs and DMAs each (me -> remote board b)
  stage run into the gateway's `stage` buffer, on
  `stage_sems[sigma * waves + w]`. With `row_send` (direct dispatch only),
  every row is DMA'd from `rows` straight to its receive row, skipping the
  pack into `send` and its wait; `rows` is then returned as one more output
  that the caller passes to `ragged_a2a_send_done_row_send`.

  Args:
    send: Uninitialized send buffer the kernel packs from `rows`.
    recv: Uninitialized receive buffer in HBM.
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
    row_send: Send each row from `rows` directly (no pack), direct only.
    direct_order: Direct-send destination order (DIRECT_ORDERS; EXPERIMENTAL).
    stripe_boards: Boards of the board-major direct orders.
    stripe_board_size: Devices per board of the board-major direct orders.
    stage_hoist: De-Dup stage-send placement (STAGE_HOISTS; EXPERIMENTAL).
    pack_parity: Pack semaphores (PACK_PARITIES); 2 overlaps the pack of
      each step with the previous step's wait and send. Packed path only.

  Returns:
    The `fut` tuple of semaphores and buffers the receivers and the drain wait
    on.

  Raises:
    ValueError: If the arguments are inconsistent.
  """
  if stage_hoist not in STAGE_HOISTS:
    raise ValueError(f"stage_hoist must be one of {STAGE_HOISTS}")
  if pack_parity not in PACK_PARITIES:
    raise ValueError(f"pack_parity must be one of {PACK_PARITIES}")
  _check_direct_order(direct_order, ep, stripe_boards, stripe_board_size)
  if row_send:
    # Row send has no pack and no stage; only the destination order applies.
    stage_hoist, pack_parity = "none", 1
  hbm = functools.partial(
      pltpu.with_memory_space_constraint, memory_space=pltpu.HBM
  )
  send, recv = hbm(send), hbm(recv)
  tg_lists = hbm(tg_lists)
  tg_land = hbm(tg_land)
  if stage is not None:
    stage = hbm(stage)
  has_stage = stage is not None
  if has_stage != (stage_send_tab is not None):
    raise ValueError("pass stage_send_tab and stage together or neither")
  if has_stage and (num_boards < 1 or num_slots < 1 or board_size < 1):
    raise ValueError(
        "has_stage requires num_boards, num_slots and board_size >= 1"
    )
  list_rows = tg_lists.shape[0] // ep
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  in_specs = [smem, smem, anyspec, anyspec, smem, anyspec]
  args = [
      disp_tab.astype(jnp.int32),
      jnp.reshape(me, (1,)).astype(jnp.int32),
      send,
      recv,
      perm.astype(jnp.int32),
      rows,
  ]
  aliases = {2: 2, 3: 3}
  out_shape = [
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA((waves,)),
      jax.ShapeDtypeStruct(send.shape, send.dtype),
      jax.ShapeDtypeStruct(recv.shape, recv.dtype),
  ]
  out_specs = [semspec, semspec, anyspec, anyspec]
  base_in = len(in_specs)
  base_out = len(out_shape)
  in_specs.extend([smem, anyspec, anyspec])
  args.extend([list_tab.astype(jnp.int32), tg_lists, tg_land])
  out_shape.extend([
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA(()),
      jax.ShapeDtypeStruct(tg_lists.shape, tg_lists.dtype),
      jax.ShapeDtypeStruct(tg_land.shape, tg_land.dtype),
  ])
  out_specs.extend([semspec, semspec, anyspec, anyspec])
  aliases[base_in + 1] = base_out + 2
  aliases[base_in + 2] = base_out + 3
  if has_stage:
    assert stage_send_tab is not None
    assert stage is not None
    base_in = len(in_specs)
    base_out = len(out_shape)
    in_specs.extend([smem, anyspec])
    args.extend([stage_send_tab.astype(jnp.int32), stage])
    out_shape.extend([
        pltpu.SemaphoreType.DMA((num_slots * waves,)),
        jax.ShapeDtypeStruct(stage.shape, stage.dtype),
    ])
    out_specs.extend([semspec, anyspec])
    aliases[base_in + 1] = base_out + 1
  if row_send:
    if has_stage:
      raise ValueError("row_send does not support De-Dup stage runs")
    # rows is the DMA source of every remote row: alias it to an output
    # the drain kernel takes, so XLA keeps it alive until the drain.
    args[5] = hbm(rows)
    aliases[5] = len(out_shape)
    out_shape.append(jax.ShapeDtypeStruct(rows.shape, rows.dtype))
    out_specs.append(anyspec)
  return pl.pallas_call(
      functools.partial(
          _start_waves_kernel,
          ep=ep,
          waves=waves,
          chunk_rows=chunk_rows,
          list_rows=list_rows,
          has_stage=has_stage,
          num_boards=num_boards,
          board_size=board_size,
          row_send=row_send,
          direct_order=direct_order,
          stripe_boards=stripe_boards,
          stripe_board_size=stripe_board_size,
          stage_hoist=stage_hoist,
          pack_parity=pack_parity,
      ),
      out_shape=tuple(out_shape),
      in_specs=in_specs,
      out_specs=tuple(out_specs),
      scratch_shapes=(
          pltpu.SemaphoreType.DMA((2,) if pack_parity == 2 else ()),
      ),
      input_output_aliases=aliases,
      compiler_params=pltpu.CompilerParams(collective_id=A2A_COLLECTIVE_ID),
      name=(
          f"wave_a2a_start_ep{ep}_w{waves}{'_dedup' if has_stage else ''}"
          f"{'_rows' if row_send else ''}"
          f"{'' if direct_order == 'ring' else '_' + direct_order}"
          f"{'_hoist' + stage_hoist if has_stage and stage_hoist != 'none' else ''}"
          f"{'_pp2' if pack_parity == 2 else ''}"
      ),
  )(*args)


def _send_done_kernel(
    tab,
    send,
    send_sem,
    list_tab,
    tg_lists,
    list_send_sem,
    *rest,
    ep,
    axis=None,
):
  """Drain kernel body: wait for this shard's outgoing rows and lists."""
  # rest: [rows_keep,] carry, carry_out. rows_keep (row send) and carry are
  # pass-throughs that keep their buffers alive and order the drain.
  del rest
  total = tab[3 * ep + 1, 0]
  # Any peer works: the drain only waits on this shard's own send semaphore
  # (upstream passes mesh coordinate (0,)). The standalone drain kernel
  # (ragged_a2a_send_done) names the shard_map axis instead, because JAX
  # lowers `wait_send` with the real device id against the kernel's mesh,
  # and Pallas gives a kernel its mesh
  # only when the kernel has a named-axis or comms effect. That kernel only
  # waits, and a DMA with a tuple id adds neither, so the tuple failed to
  # lower ("got 1 ids for a 0D mesh"); the dict form adds the named-axis
  # effect. Same wait, same semaphore. axis=None keeps the tuple for callers
  # that inline this body into a kernel that already communicates.
  peer = (jnp.int32(0),) if axis is None else {axis: jnp.int32(0)}

  @pl.when(total > 0)
  def _():
    # One byte-exact wait for every row this shard sent to other shards
    # (all on the one send semaphore); `wait_send` never blocks on peers.
    pltpu.make_async_remote_copy(
        src_ref=send.at[pl.ds(0, total)],
        dst_ref=send.at[pl.ds(0, total)],
        send_sem=send_sem,
        recv_sem=send_sem,
        device_id=peer,
    ).wait_send()

  list_total = list_tab[ep]

  @pl.when(list_total > 0)
  def _():
    pltpu.make_async_remote_copy(
        src_ref=tg_lists.at[pl.ds(0, list_total)],
        dst_ref=tg_lists.at[pl.ds(0, list_total)],
        send_sem=list_send_sem,
        recv_sem=list_send_sem,
        device_id=peer,
    ).wait_send()


def _shard_map_axis():
  """The single mesh axis of the enclosing shard_map (the layer's)."""
  names = jax.sharding.get_abstract_mesh().axis_names
  if len(names) != 1:
    raise ValueError(
        "ragged_a2a_send_done must run inside the layer's single-axis"
        f" shard_map; the current mesh axes are {names}"
    )
  return names[0]


def ragged_a2a_send_done(
    send_sem,
    send,
    disp_tab,
    carry,
    *,
    ep,
    list_tab,
    tg_lists,
    list_send_sem,
):
  """Drain the start kernel's outgoing DMAs; see `_send_done_call`."""
  return _send_done_call(
      send_sem,
      send,
      disp_tab,
      carry,
      ep=ep,
      list_tab=list_tab,
      tg_lists=tg_lists,
      list_send_sem=list_send_sem,
  )


def ragged_a2a_send_done_row_send(
    send_sem,
    send,
    disp_tab,
    carry,
    *,
    ep,
    list_tab,
    tg_lists,
    list_send_sem,
    rows_keep,
):
  """`ragged_a2a_send_done` that also keeps the row-send source alive."""
  return _send_done_call(
      send_sem,
      send,
      disp_tab,
      carry,
      ep=ep,
      list_tab=list_tab,
      tg_lists=tg_lists,
      list_send_sem=list_send_sem,
      rows_keep=rows_keep,
  )


def _send_done_call(
    send_sem,
    send,
    disp_tab,
    carry,
    *,
    ep,
    list_tab,
    tg_lists,
    list_send_sem,
    rows_keep=None,
):
  """Drain the start kernel's outgoing DMAs; returns `carry` (aliased).

  `carry` must be a value the caller USES (e.g. the fused kernel's
  arrivals): threading it through makes the drain impossible to
  dead-code-eliminate and orders it after `carry`'s producer. Without it
  the drain was eliminated, so the send semaphore and send buffer died
  right after the start kernel and XLA reused them while the DMAs were
  still in flight (the fused kernel's semaphores ended nonzero and garbage
  rows were sent).

  `carry` (and so the aliased output) is pinned to HBM. Unconstrained,
  memory-space assignment can stage the arrivals through VMEM around this
  call, a full extra copy in and out.

  The start kernel sends De-Dup stage rows on the same send semaphore and
  counts them in disp_tab[3 * ep + 1, 0], so this one wait covers them.

  Args:
    send_sem: The start kernel's send semaphore.
    send: The send buffer.
    disp_tab: [4 * ep, W] dispatch table.
    carry: A value the caller uses, threaded through and aliased.
    ep: Expert-parallel width.
    list_tab: Token-gather list table.
    tg_lists: The sent lists.
    list_send_sem: The lists' send semaphore.
    rows_keep: The row-send source rows (start kernel's extra output), kept
      alive until this drain; None without row send.

  Returns:
    `carry`, aliased and pinned to HBM.
  """
  carry = pltpu.with_memory_space_constraint(carry, pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  in_specs = [smem, anyspec, semspec, smem, anyspec, semspec]
  args = [
      disp_tab.astype(jnp.int32),
      send,
      send_sem,
      list_tab.astype(jnp.int32),
      tg_lists,
      list_send_sem,
  ]
  if rows_keep is not None:
    in_specs.append(anyspec)
    args.append(pltpu.with_memory_space_constraint(rows_keep, pltpu.HBM))
  in_specs.append(anyspec)
  args.append(carry)
  return pl.pallas_call(
      functools.partial(_send_done_kernel, ep=ep, axis=_shard_map_axis()),
      out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
      in_specs=in_specs,
      out_specs=anyspec,
      input_output_aliases={len(args) - 1: 0},
      name=f"wave_a2a_send_done_ep{ep}",
  )(*args)


def _routing_ag_start_kernel(
    me_ref,
    packed_in,
    out_in,
    send_sem,
    recv_sem,
    packed_keep,
    out_ref,
    *,
    ep,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
):
  """Issues async all-gather DMAs of `packed_keep` `[1, n_sub, 128]`."""
  del packed_in, out_in
  me = me_ref[0]
  _all_pairs_barrier(ep)
  src = packed_keep.at[pl.ds(0, 1)]
  dst_me = out_ref.at[pl.ds(me, 1)]
  pltpu.make_async_copy(src, dst_me, recv_sem).start()
  if direct_order != "ring":
    dests = [
        dest
        for group in _direct_dest_groups(
            me,
            ep=ep,
            direct_order=direct_order,
            stripe_boards=stripe_boards,
            stripe_board_size=stripe_board_size,
        )
        for dest in group
    ]
    for d, is_local in dests:
      if not is_local:
        pltpu.make_async_remote_copy(
            src_ref=src,
            dst_ref=dst_me,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(d,),
        ).start()
  else:
    ep_is_pow2 = (ep & (ep - 1)) == 0
    for i in range(1, ep):
      d = (
          jnp.bitwise_and(me + i, jnp.int32(ep - 1))
          if ep_is_pow2
          else lax.rem(me + i, ep)
      )
      pltpu.make_async_remote_copy(
          src_ref=src,
          dst_ref=dst_me,
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id=(d,),
      ).start()


def _routing_ag_done_kernel(
    out_in,
    packed_keep,
    send_sem,
    recv_sem,
    out_ref,
    *,
    ep,
    axis=None,
):
  """Waits for incoming routing slices and drains outgoing routing DMAs."""
  del out_in, packed_keep
  pltpu.make_async_copy(
      out_ref.at[pl.ds(0, ep)],
      out_ref.at[pl.ds(0, ep)],
      recv_sem,
  ).wait()
  if ep > 1:
    peer = (jnp.int32(0),) if axis is None else {axis: jnp.int32(0)}
    pltpu.make_async_remote_copy(
        src_ref=out_ref.at[pl.ds(0, ep - 1)],
        dst_ref=out_ref.at[pl.ds(0, ep - 1)],
        send_sem=send_sem,
        recv_sem=send_sem,
        device_id=peer,
    ).wait_send()


def routing_ag_start(
    packed_tile,
    out_tile,
    me,
    *,
    ep,
    direct_order="ring",
    stripe_boards=0,
    stripe_board_size=0,
):
  """Starts split-phase Pallas routing all-gather over `[1, n_sub, 128]`."""
  _check_direct_order(direct_order, ep, stripe_boards, stripe_board_size)
  hbm = functools.partial(
      pltpu.with_memory_space_constraint, memory_space=pltpu.HBM
  )
  packed_tile = hbm(packed_tile)
  out_tile = hbm(out_tile)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  return pl.pallas_call(
      functools.partial(
          _routing_ag_start_kernel,
          ep=ep,
          direct_order=direct_order,
          stripe_boards=stripe_boards,
          stripe_board_size=stripe_board_size,
      ),
      out_shape=(
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA(()),
          jax.ShapeDtypeStruct(packed_tile.shape, packed_tile.dtype),
          jax.ShapeDtypeStruct(out_tile.shape, out_tile.dtype),
      ),
      in_specs=[smem, anyspec, anyspec],
      out_specs=(semspec, semspec, anyspec, anyspec),
      input_output_aliases={1: 2, 2: 3},
      compiler_params=pltpu.CompilerParams(
          collective_id=ROUTING_AG_COLLECTIVE_ID
      ),
      name=(
          f"routing_ag_start_ep{ep}"
          f"{'' if direct_order == 'ring' else '_' + direct_order}"
      ),
  )(jnp.reshape(jnp.asarray(me, jnp.int32), (1,)), packed_tile, out_tile)


def routing_ag_done(ag_fut, *, ep):
  """Completes `routing_ag_start` and returns `[ep, n_sub, 128]` in HBM."""
  send_sem, recv_sem, packed_keep, out_tile = ag_fut
  hbm = functools.partial(
      pltpu.with_memory_space_constraint, memory_space=pltpu.HBM
  )
  out_tile = hbm(out_tile)
  packed_keep = hbm(packed_keep)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
  return pl.pallas_call(
      functools.partial(_routing_ag_done_kernel, ep=ep, axis=_shard_map_axis()),
      out_shape=jax.ShapeDtypeStruct(out_tile.shape, out_tile.dtype),
      in_specs=[anyspec, anyspec, semspec, semspec],
      out_specs=anyspec,
      input_output_aliases={0: 0},
      name=f"routing_ag_done_ep{ep}",
  )(out_tile, packed_keep, send_sem, recv_sem)
