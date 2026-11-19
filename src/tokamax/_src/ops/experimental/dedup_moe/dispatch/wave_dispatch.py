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
"""Wave dispatch: the token exchange of the De-Dup MoE layer, in expert waves.

Instead of an activation all-gather, which delivers every token to every
shard, each shard sends every (token, destination shard) pair with a hit
once, through the split-phase Pallas ragged all-to-all in
tc_sender.py.

Every shard orders its local experts fewest rows first, from the per-expert
counts the routing exchange already all-gathers, so senders and receivers
agree on the order without exchanging it, and cuts that order into W groups
("waves") whose modeled compute grows 1 : 2 : 4 : ... (wave_plan). The
exchange is laid out and issued wave by wave, and the fused kernel waits for
an expert's wave only before that expert's first tile: the smallest experts
start soonest and the rest of the exchange lands under their matmuls. The
routing tables are built while the exchange is in flight.

With De-Dup (dedup_tables), a token needed by several devices of a remote
board is sent once to that board's gateway, which forwards it inside the
fused kernel (dedup_plan).

Every buffer is sized for the worst case (all tokens on one shard), so no
routing drops a token.
"""

from typing import NamedTuple
import os

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dispatch import source_plan
from tokamax._src.ops.experimental.dedup_moe.dispatch import tc_sender as dispatch_kernels
from tokamax._src.ops.experimental.fused_moe import host as fused_host

ROWBLK = fused_host.ROWBLK
align_up = fused_host.align_up

# Rows in the first copy of each destination's share of a wave; the rest of
# the share follows in one copy (tc_sender._start_waves_kernel).
CHUNK_ROWS = 8
# Direct dispatch sends each row from the token rows to its receive row,
# without the pack into the send buffer (tc_sender row_send). Opt-in until
# measured: DEDUP_MOE_ROW_SEND=1, or row_send=True on `start`.
_ROW_SEND = os.environ.get("DEDUP_MOE_ROW_SEND", "0") == "1"
# Row send only up to this many tokens per shard, the packed path above it
# (DEDUP_MOE_ROW_SEND_MAX_T, or row_send_max_t on `start`); unset: every t.
_ROW_SEND_MAX_T = os.environ.get("DEDUP_MOE_ROW_SEND_MAX_T", "")
# EXPERIMENTAL scheduling knobs, all off by default; the `start` kwargs of the
# same names override them.
#   DEDUP_MOE_PACK_PARITY=2: overlap each pack with the previous send
#     (tc_sender pack_parity).
#   DEDUP_MOE_DIRECT_ORDER: tc_sender.DIRECT_ORDERS (packed and row-send
#     paths); board-major orders use
#     DEDUP_MOE_DIRECT_BOARD_SIZE (0: the De-Dup topology's, else
#     min(MAX_BOARD_SIZE, ep)).
#   DEDUP_MOE_WAVE_ORDER: "rows" or "xboard" (wave_plan hist_remote).
#   DEDUP_MOE_RAMP_RATIO: wave_plan ramp_ratio (<= 0: [1, 2, 2, 3] shares).
#   DEDUP_MOE_STAGE_HOIST: tc_sender.STAGE_HOISTS (De-Dup).
#   DEDUP_MOE_COMPACT_PLAN=1: dedup_plan's single-sort stage order, packed
#     forward-run sort and compact sender perm, and source_plan fw_only.
#   DEDUP_MOE_MIN_FANOUT: overrides min_fanout in `start`,
#     `materialize_smem_ops` and `forward_runs_needed`; 0 selects
#     dedup_plan's board-gathered stage order.
#   DEDUP_MOE_SC_SENDER=1: SparseCore sends (sc_sender; needs tpu_sc).
_PACK_PARITY = int(os.environ.get("DEDUP_MOE_PACK_PARITY", "1"))
_DIRECT_ORDER = os.environ.get("DEDUP_MOE_DIRECT_ORDER", "ring")
_DIRECT_BOARD_SIZE = int(os.environ.get("DEDUP_MOE_DIRECT_BOARD_SIZE", "0"))
_WAVE_ORDER = os.environ.get("DEDUP_MOE_WAVE_ORDER", "rows")
_RAMP_RATIO_ENV = os.environ.get("DEDUP_MOE_RAMP_RATIO", "")
_STAGE_HOIST = os.environ.get("DEDUP_MOE_STAGE_HOIST", "none")
COMPACT_PLAN = os.environ.get("DEDUP_MOE_COMPACT_PLAN", "0") == "1"
_MIN_FANOUT_ENV = os.environ.get("DEDUP_MOE_MIN_FANOUT", "")
SC_SENDER = os.environ.get("DEDUP_MOE_SC_SENDER", "0") == "1"
# Early dispatch (direct, W=1 only): plan and issue the exchange from this
# shard's own top-k, before the routing all-gather, which then overlaps the
# dispatch DMAs in flight (start_early / finish_early). Opt-in:
# DEDUP_MOE_EARLY_DISPATCH=1.
_EARLY_DISPATCH = os.environ.get("DEDUP_MOE_EARLY_DISPATCH", "0") == "1"
_ASYNC_ROUTING_AG = os.environ.get("DEDUP_MOE_ASYNC_ROUTING_AG", "0") == "1"
_ASYNC_ROUTING_AG_MAX_T = int(
    os.environ.get("DEDUP_MOE_ASYNC_ROUTING_AG_MAX_T", "96")
)


def early_dispatch_enabled(*, waves, dedup):
  """Whether the layer issues the exchange before the routing all-gather."""
  return _EARLY_DISPATCH and waves == 1 and not dedup


def async_routing_ag_enabled(*, waves, dedup, t_local):
  """Whether the layer overlaps split-phase routing AG with W=1 local prep."""
  return (
      _ASYNC_ROUTING_AG
      and waves == 1
      and not dedup
      and (_ASYNC_ROUTING_AG_MAX_T <= 0 or t_local <= _ASYNC_ROUTING_AG_MAX_T)
  )


class WaveDispatch(NamedTuple):
  """An exchange in flight and the plan it was laid out by.

  fut: tc_sender.ragged_a2a_start_waves's (send_sem, recv_sems,
    send, recv, list_send_sem, tg_sem, tg_lists, tg_land[, stage_sems,
    stage]). The fused kernel reads its rows from `recv` and waits on
    `recv_sems`; `tg_assemble` reads `tg_land` and waits on `tg_sem`;
    `finish` drains `send_sem` and `list_send_sem`.
  disp_tab [4 * ep, waves] i32: the dispatch plan (see tc_sender);
    row wave_bounds_row(ep) carries this shard's wave cuts.
  visit_order [ep, g_local] i32: every shard's local experts in visit
    order (waves ascending, fewest rows first), empty experts last.
  list_tab [ep + 1] i32: token-gather list DMA row counts per destination
    plus the remote total at index `ep`.
  stage_send_tab [5 * nb, waves] i32 | None: the sender's De-Dup stage DMA
    table.
  smem_ops SmemOperands | None: the gateway's stage_tab, fwd_tab and
    fwd_cnt for the fused kernel (De-Dup; materialize_smem_ops).
  dplan SourcePlan | None: the sender's De-Dup dispatch plan arrays.
  fwd_schedule FwdSchedule | None: the De-Dup forward schedule.
  wave_of [ep, g_local] i32: the wave of every shard's local experts.
  """

  fut: tuple[jax.Array, ...]
  disp_tab: jax.Array
  visit_order: jax.Array
  list_tab: jax.Array | None = None
  stage_send_tab: jax.Array | None = None
  smem_ops: dedup_plan_mod.SmemOperands | None = None
  dplan: dedup_plan_mod.SourcePlan | None = None
  fwd_schedule: dedup_plan_mod.FwdSchedule | None = None
  wave_of: jax.Array | None = None
  rows_keep: jax.Array | None = None

  @property
  def send_buf(self):
    return self.fut[2]

  @property
  def tg_sem(self):
    return self.fut[5] if len(self.fut) > 4 else None

  @property
  def tg_land(self):
    return self.fut[7] if len(self.fut) > 4 else None

  @property
  def stage_sems(self):
    return self.fut[8] if len(self.fut) > 8 else None

  @property
  def stage(self):
    return self.fut[9] if len(self.fut) > 8 else None


def ascending_expert_order(hist, *, ep, g_local):
  """Orders each shard's local experts fewest rows first, empties last.

  Args:
    hist: [ep * g_local] i32 rows per global expert, identical on every shard.
    ep: Expert-parallel width.
    g_local: Experts per shard.

  Returns:
    (pos, order): pos [ep, g_local] i32 is the visit position of local expert e
    on shard d; order [ep, g_local] i32 is the local expert at visit position p,
    so order[me] is a drop-in visit list. Ties break by ascending index.
  """
  h = hist.reshape(ep, g_local).astype(jnp.int32)
  idx = jnp.arange(g_local, dtype=jnp.int32)
  # 2**24 > any row count; 2**24 * g_local + g_local stays below 2**31.
  key = jnp.where(h > 0, jnp.minimum(h, 2**24 - 1), 2**24) * g_local + idx
  pos = jnp.sum((key[:, None, :] < key[:, :, None]).astype(jnp.int32), axis=2)
  order = jnp.sum(
      jnp.where(pos[:, None, :] == idx[None, :, None], idx[None, None, :], 0),
      axis=2,
  )
  return pos, order


# Ramp waves: each shard's experts, fewest rows first, cut into W waves of
# VARIABLE size so the waves' modeled compute grows RAMP_RATIO-fold per wave
# (1 : 2 : 4 : ...), at least one expert per wave: a small first wave is a
# short pipeline fill. The fused kernel reads each wave's end position from
# row `wave_bounds_row(ep)` of disp_tab. An expert's modeled compute is
# tau + c * rows (a fixed cost per visited expert plus a cost per row),
# here in units of c as the integer EXPERT_COST_ROWS + rows, so every shard
# derives the same waves bit for bit (the cost model's ramp wave
# assignment). The waves change
# only when rows land and which expert is computed when, never what is
# computed.
RAMP_RATIO = 2
EXPERT_COST_ROWS = 129  # round(tau / c) of the fitted cost model.
# Fixed per-expert cost of the [1, 2, 2, 3] schedule (ramp_ratio <= 0).
ECR_EXPERT_COST_ROWS = 227


def _resolve_ramp_ratio(ramp_ratio):
  """`ramp_ratio`, or DEDUP_MOE_RAMP_RATIO / RAMP_RATIO when None."""
  if ramp_ratio is not None:
    return ramp_ratio
  return float(_RAMP_RATIO_ENV) if _RAMP_RATIO_ENV else RAMP_RATIO


def _resolve_min_fanout(min_fanout):
  """DEDUP_MOE_MIN_FANOUT when set, else `min_fanout`."""
  return int(_MIN_FANOUT_ENV) if _MIN_FANOUT_ENV else min_fanout


def _resolve_compact_plan(compact_plan):
  return COMPACT_PLAN if compact_plan is None else bool(compact_plan)


def _wave_shares(ramp_ratio, waves):
  """Integer modeled-compute shares of the waves (wave_plan)."""
  if float(ramp_ratio) <= 0.0:
    share = [1] + [2] * max(waves - 2, 0) + ([3] if waves >= 2 else [])
    return share[:waves]
  if float(ramp_ratio) == int(ramp_ratio):
    return [int(ramp_ratio) ** w for w in range(waves)]
  return [round(float(ramp_ratio) ** w * 1024) for w in range(waves)]


def wave_bounds_row(ep):
  """Returns the disp_tab row that carries the wave bounds.

  The disp_tab row that carries the wave bounds.

  disp_tab rows 3 * ep + 2 .. 4 * ep - 1 are zero padding
  (source_plan.dispatch_table, dedup_plan.dedup_disp_tab), so the bounds
  need no kernel operand of their own; hence ep >= 3.

  Args:
    ep: Expert-parallel width (>= 3).

  Returns:
    3 * ep + 2, a zero-padding row of the dispatch table.
  """
  return 3 * ep + 2


def _rank_by(keys, *, g_local):
  """Per-shard rank of each local expert under lexicographic `keys`.

  Ties break by ascending expert index. A compare-reduce over [ep, g, g]
  rather than a sort.

  Args:
    keys: [ep, g_local] i32 arrays, most significant first.
    g_local: Experts per shard.

  Returns:
    (pos, order) like ascending_expert_order.
  """
  idx = jnp.arange(g_local, dtype=jnp.int32)
  # before[d, e, e2]: e2 precedes e.
  before = idx[None, None, :] < idx[None, :, None]
  for k in reversed(keys):
    a, b = k[:, None, :], k[:, :, None]
    before = (a < b) | ((a == b) & before)
  pos = jnp.sum(before.astype(jnp.int32), axis=2)
  order = jnp.sum(
      jnp.where(pos[:, None, :] == idx[None, :, None], idx[None, None, :], 0),
      axis=2,
  )
  return pos, order


def wave_plan(
    hist,
    *,
    ep,
    g_local,
    waves,
    ramp_ratio=RAMP_RATIO,
    expert_cost_rows=EXPERT_COST_ROWS,
    hist_remote=None,
):
  """Computes every shard's visit order, expert waves and wave bounds (ramp waves).

  Args:
    hist: [ep * g_local] i32 rows per global expert, identical on every shard.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    waves: W.
    ramp_ratio: Growth of each wave's modeled compute (0.0 for ECR comm+compute
      weighted [1, 2, 2, 3] schedule, -1.0 for pure ECR compute [1, 2, 2, 3]).
    expert_cost_rows: Fixed per-expert cost, in rows. A non-integer ramp_ratio
      uses integer shares round(ramp_ratio**w * 1024).
    hist_remote: EXPERIMENTAL. Optional [ep * g_local] i32 rows per global
      expert that come from another board. When given (and W > 1), the experts
      are ordered by hist_remote / (expert_cost_rows + rows) ascending (the
      fewest cross-board rows per unit of modeled compute first) instead of by
      rows, so the early waves compute on intra-board rows while the
      cross-board rows are in flight.

  Returns:
    (visit, wave_of, bounds): visit [ep, g_local] i32 local expert at each visit
    position (waves contiguous and ascending, fewest rows first inside a wave,
    empty experts last); wave_of [ep, g_local] i32 each local expert's wave (W -
    1 for empty experts); bounds [ep, W] i32 nonempty experts in waves 0..w.
    With at most W nonempty experts, each gets its own wave.
  """
  h = hist.reshape(ep, g_local).astype(jnp.int32)
  nonempty = h > 0
  m = jnp.sum(nonempty.astype(jnp.int32), axis=1)  # [ep]
  eff_cost_rows = (
      ECR_EXPERT_COST_ROWS if float(ramp_ratio) <= 0.0 else expert_cost_rows
  )
  rkey = None
  hr_i32 = None
  if waves == 1 or hist_remote is None:
    pos, asc = ascending_expert_order(hist, ep=ep, g_local=g_local)
  else:
    hr_i32 = hist_remote.reshape(ep, g_local).astype(jnp.int32)
    ratio = hr_i32.astype(jnp.float32) / (eff_cost_rows + h).astype(
        jnp.float32
    )
    rkey = jnp.minimum((ratio * 2**20).astype(jnp.int32), 2**20)
    pos, asc = _rank_by(
        [jnp.where(nonempty, 0, 1).astype(jnp.int32), rkey, h],
        g_local=g_local,
    )
  if waves == 1:
    wave_of = jnp.zeros((ep, g_local), jnp.int32)
    return asc, wave_of, m[:, None].astype(jnp.int32)
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  use_pos_mask = jax.default_backend() == "tpu"
  pos_mask = (
      pos[:, None, :] == g_iota[None, :, None] if use_pos_mask else None
  )
  if float(ramp_ratio) == 0.0 and hr_i32 is not None:
    cost = jnp.where(nonempty, eff_cost_rows + h + (hr_i32 >> 1), 0)
  else:
    cost = jnp.where(nonempty, eff_cost_rows + h, 0)  # [ep, g] i32
  if pos_mask is not None:
    cost_asc = jnp.sum(jnp.where(pos_mask, cost[:, None, :], 0), axis=2)
  else:
    cost_asc = jnp.take_along_axis(cost, asc, axis=1)
  cs = jnp.cumsum(cost_asc, axis=1)  # cs[:, c - 1]: the first c experts
  total = cs[:, -1]
  share = _wave_shares(ramp_ratio, waves)
  share_sum = sum(share)
  cum_share = [sum(share[: w + 1]) for w in range(waves)]
  big = jnp.int32(2**30)
  cand = g_iota + 1  # experts in waves 0..w
  prev = jnp.zeros((ep,), jnp.int32)
  cuts = []
  for w in range(waves - 1):
    hi = m - (waves - 1 - w)
    valid = (cand[None, :] >= prev[:, None] + 1) & (
        cand[None, :] <= hi[:, None]
    )
    # |cs * S - total * cum_share[w]| = S * |cs - target_w|, exact.
    err = jnp.abs(cs * share_sum - (total * cum_share[w])[:, None])
    c = jnp.argmin(jnp.where(valid, err, big), axis=1).astype(jnp.int32) + 1
    cuts.append(c)
    prev = c
  if cuts:
    cut = jnp.stack(cuts, axis=1)  # [ep, W - 1]
    wave_asc = jnp.sum(
        (cut[:, None, :] <= g_iota[None, :, None]).astype(jnp.int32), axis=2
    )
  else:
    wave_asc = jnp.zeros((ep, g_local), jnp.int32)
  if pos_mask is not None:
    wave_of = jnp.sum(jnp.where(pos_mask, wave_asc[:, :, None], 0), axis=1)
  else:
    wave_of = jnp.take_along_axis(wave_asc, pos, axis=1)
  few = (m <= waves)[:, None]
  wave_of = jnp.where(few, jnp.minimum(pos, waves - 1), wave_of)
  wave_of = jnp.where(nonempty, wave_of, waves - 1).astype(jnp.int32)
  if use_pos_mask:
    visit = asc
  else:
    in_wave = [h] if rkey is None else [rkey, h]
    _, visit = _rank_by(
        [jnp.where(nonempty, wave_of, waves), *in_wave], g_local=g_local
    )
  cnt = jnp.sum(
      (
          (wave_of[:, :, None] == w_iota[None, None, :]) & nonempty[:, :, None]
      ).astype(jnp.int32),
      axis=1,
  )
  return visit, wave_of, jnp.cumsum(cnt, axis=1).astype(jnp.int32)


def dispatch_plan(
    pairs_g,
    me,
    *,
    ep,
    t_local,
    g_local,
    waves,
    send_alloc,
    wave_of,
):
  """Reference dispatch plan for shard `me` (tests only).

  The original dense wave plan, kept as the oracle source_plan is checked
  against (source_plan_test, dedup_plan_test); the layer builds its plan
  with source_plan.

  Every (token, dest) pair with a hit is sent ONCE, in its first-hit wave.
  The receive buffer on dest d is laid out by (wave, source), tokens
  ascending within each segment.

  Args:
    pairs_g: [n_tokens, topk] i32 global expert ids in mesh order (dest = id //
      g_local).
    me: This shard's index.
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    send_alloc: Rows of the send buffer.
    wave_of: [ep, g_local] i32 wave of each shard's local experts.

  Returns:
    (perm [send_alloc] i32: local token row of each packed send row;
    tab [4 * ep, waves] i32: send offset / send rows / dest offset per
    (d, w), rows received per wave (row 3 ep), total rows sent to other
    shards (row 3 ep + 1, col 0); posmap [n_tokens] i32: receive-buffer row of
    global token t on `me`, 0 where `me` does not need t).
  """
  n_tokens, topk = pairs_g.shape
  # The token axis stays minor in every array below. With topk minor, a
  # [n_tokens, topk] int32 array fills topk of a vreg's 128 lanes, and the
  # relayouts and elementwise ops on it cost more than the plan's arithmetic.
  pk = pairs_g.T.astype(jnp.int32)  # [topk, n_tokens]
  dest = pk // g_local
  table = wave_of.astype(jnp.int32)  # [ep, g_local]
  # The wave of every pair is table[dest, local expert]. Not jnp.take (an
  # XLA TPU gather over every pair is slow), nor a compare over all E
  # experts. A one-hot of the local expert
  # times the table on the MXU gives each pair its wave on every shard, and
  # a compare over the ep shards keeps its own. The entries are below
  # 2 * waves, so bf16 holds them exactly.
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  local = (pk - dest * g_local).reshape(1, topk * n_tokens)
  onehot = (local == g_iota[:, None]).astype(
      jnp.bfloat16
  )  # [g_local, topk*n_tokens]
  by_dest = jnp.dot(
      table.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
  )  # [ep, topk*n_tokens]
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  wave = (
      jnp.sum(
          jnp.where(
              dest.reshape(1, topk * n_tokens) == d_iota[:, None], by_dest, 0.0
          ),
          axis=0,
      )
      .astype(jnp.int32)
      .reshape(topk, n_tokens)
  )
  # First-hit wave of each token on each shard; `waves` = no hit.
  fw = jnp.min(
      jnp.where(
          dest[:, None, :] == d_iota[None, :, None], wave[:, None, :], waves
      ),
      axis=0,
  )  # [dest, n_tokens]
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  oh = (
      (fw[:, None, :] == w_iota[None, :, None])
      .astype(jnp.int32)
      .reshape(ep, waves, ep, t_local)
  )  # [dest, w, src, t]
  cnt = oh.sum(axis=3)  # [dest, w, src]
  # Receive layout on every dest: segments in (w, src) order.
  seg = cnt.reshape(ep, waves * ep)
  base = (jnp.cumsum(seg, axis=1) - seg).reshape(ep, waves, ep)  # [d, w, s]
  dst_off = lax.dynamic_index_in_dim(base, me, axis=2, keepdims=False)  # [d, w]
  send_cnt = lax.dynamic_index_in_dim(cnt, me, axis=2, keepdims=False)  # [d, w]
  flat = send_cnt.reshape(-1)
  send_off = (jnp.cumsum(flat) - flat).reshape(ep, waves)
  recv_cnt = lax.dynamic_index_in_dim(cnt, me, axis=0, keepdims=False).sum(1)
  remote = (
      send_cnt.sum()
      - lax.dynamic_index_in_dim(send_cnt, me, axis=0, keepdims=False).sum()
  )
  col = jnp.arange(waves, dtype=jnp.int32)
  extra = jnp.concatenate(
      [
          recv_cnt.astype(jnp.int32)[None, :],
          jnp.where(col == 0, remote, 0)[None, :].astype(jnp.int32),
          jnp.zeros((ep - 2, waves), jnp.int32),
      ],
      axis=0,
  )
  tab = jnp.concatenate([send_off, send_cnt, dst_off, extra], axis=0)
  # Exclusive prefix counts along the token axis as a matmul with a
  # strictly upper triangular ones matrix (0/1 in bf16, exact f32 sums),
  # which is faster than XLA's reduce-window cumsum on TPU.
  tri = jnp.triu(jnp.ones((t_local, t_local), jnp.bfloat16), k=1)

  def ranks(x):  # [rows, t_local] 0/1 -> exclusive prefix count per row
    return jnp.dot(
        x.astype(jnp.bfloat16), tri, preferred_element_type=jnp.float32
    ).astype(jnp.int32)

  # Sender: packed rows ordered (d, w, t).
  my = lax.dynamic_index_in_dim(oh, me, axis=2, keepdims=False)  # [d, w, t]
  pos = send_off[:, :, None] + ranks(my.reshape(ep * waves, t_local)).reshape(
      ep, waves, t_local
  )
  pos_dt = jnp.sum(my * pos, axis=1)  # [d, t]
  valid = my.sum(axis=1) > 0
  t_idx = jnp.broadcast_to(
      jnp.arange(t_local, dtype=jnp.int32)[None, :], (ep, t_local)
  )
  # Valid send positions are unique and dense in [0, total), so sorting the
  # (d, t) candidates by position puts each slot's token in place; the
  # slots past the total are never sent.
  _, perm = lax.sort_key_val(
      jnp.where(valid, pos_dt, jnp.int32(2**30)).reshape(-1), t_idx.reshape(-1)
  )
  perm = perm[:send_alloc]
  # Receiver: where each needed token landed on me.
  mine = lax.dynamic_index_in_dim(oh, me, axis=0, keepdims=False)  # [w, s, t]
  base_me = lax.dynamic_index_in_dim(base, me, axis=0, keepdims=False)  # [w, s]
  posr = base_me[:, :, None] + ranks(mine.reshape(waves * ep, t_local)).reshape(
      waves, ep, t_local
  )
  posmap = jnp.sum(mine * posr, axis=0).reshape(n_tokens)
  return perm, tab, posmap


def _uninit_hbm(shape, dtype):
  """An HBM buffer with no defined contents, at no cost.

  A Pallas call whose body never writes its output hands XLA a fresh,
  unfilled buffer. The ragged all-to-all only defines the rows it lands and
  the kernel reads only those, so a zero fill (which grows with T and EP)
  would be pure waste.

  The XLA buffer is also pinned to HBM (the out_specs memory space only
  types the kernel's ref). Unpinned, memory-space assignment may place the
  buffer in VMEM and then copy its undefined contents to HBM for the start
  kernel. `DATAFLOW_SIDE_EFFECTING` blocks XLA CSE from deduplicating buffers
  across layers (which would otherwise pin them across the whole model and
  insert HBM-to-HBM copies before `input_output_aliases`) while still allowing
  DCE and unconstrained scheduling.

  Args:
    shape: Buffer shape.
    dtype: Buffer element type.

  Returns:
    The uninitialized HBM buffer.
  """
  side_effect_type = getattr(pltpu, "SideEffectType", None)
  has_side_effects = (
      side_effect_type.DATAFLOW_SIDE_EFFECTING
      if side_effect_type is not None
      else True
  )
  return pl.pallas_call(
      lambda o_ref: None,
      out_shape=pltpu.HBM(shape, dtype),
      out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
      compiler_params=pltpu.CompilerParams(has_side_effects=has_side_effects),
      name="uninit_recv",
  )()


def _fwd_schedule(fwd_max_runs):
  """The De-Dup forward schedule: per target wave, lookahead 0."""
  return dedup_plan_mod.FwdSchedule(per_target_wave=True, max_runs=fwd_max_runs)


WAVE_ORDERS = ("rows", "xboard")


def _sched_board_size(board_size, dedup_tables, ep):
  """Board size of the board-aware scheduling options (`start`)."""
  if dedup_tables is not None:
    return dedup_tables.board_size
  if not board_size:
    board_size = _DIRECT_BOARD_SIZE
  if not board_size:
    board_size = min(dedup_plan_mod.MAX_BOARD_SIZE, ep)
  return int(board_size)


def _sc_sender_module():
  """sc_sender, imported on first use (it needs Pallas TPU SparseCore)."""
  from tokamax._src.ops.experimental.dedup_moe.dispatch import sc_sender  # pylint: disable=g-import-not-at-top

  return sc_sender


def start(
    rows,
    pairs,
    rows_by_dest,
    me,
    *,
    ep,
    g_local,
    waves,
    scale_bits=None,
    dedup_tables=None,
    min_fanout=2,
    fwd_max_runs=dedup_plan_mod.MAX_FWD_RUNS,
    rank_method="matmul",
    list_pack_mode=None,
    row_send=None,
    decode_fast_plan=None,
    direct_order=None,
    wave_order=None,
    ramp_ratio=None,
    board_size=None,
    stage_hoist=None,
    pack_parity=None,
    compact_plan=None,
    sc_sender=None,
    row_send_max_t=None,
    pairs_me=None,
    hist_me=None,
    recv_cnt_override=None,
):
  """Plan shard `me`'s exchange and issue it; returns a WaveDispatch.

  rows [t_local, lane blocks, 128]: this shard's token rows as they go on
    the wire (fp8 where the weight format quantizes activations).
  pairs [T, topk] i32: every shard's selected expert ids, in mesh order.
  rows_by_dest [ep, ep * g_local] i32: the all-gathered per-expert pair
    counts, identical on every shard.
  scale_bits [t_local] i32 | None: this shard's activation-scale bits when
    rows are fp8 (carried as plane 1 of the token-gather lists).
  dedup_tables TopologyTables | None: when supplied (De-Dup), stages rows
    needed by >= `min_fanout` devices of a remote board once to that board's
    gateway, which forwards them inside the fused kernel. Each forwarded
    row counts in its target's own first wave (per-target-wave
    forwarding), and the gateway's forward tables hold `fwd_max_runs` runs
    per (served source, wave); the caller builds them after the start
    kernel (materialize_smem_ops) so they overlap the dispatch DMAs.

  The token-gather lists are built without negative-index wraps
  (source_plan.build_send_lists) and every exchange buffer (send, recv,
  lists, landing, stage) is pinned to HBM, so memory-space assignment adds
  no VMEM round trips around the start and drain kernels.

  Args:
    rows: [t_local, lane blocks, 128] this shard's token rows as they go on the
      wire.
    pairs: [T, topk] i32 every shard's selected expert ids, in mesh order.
    rows_by_dest: [ep, ep * g_local] i32 all-gathered per-expert pair counts.
    me: This shard's index.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    waves: W.
    scale_bits: Optional [t_local] i32 activation-scale bits when rows are fp8.
    dedup_tables: TopologyTables to enable De-Dup, or None.
    min_fanout: Devices of a remote board that must need a token for it to be
      staged.
    fwd_max_runs: Gateway forward runs per (served source, wave).
    rank_method: `matmul` or `cumsum` (source_plan.exclusive_ranks).
    list_pack_mode: source_plan.build_send_lists `send_list_gather`, or None.
    row_send: Send rows straight from `rows` (direct dispatch only); None
      reads DEDUP_MOE_ROW_SEND.
    decode_fast_plan: source_plan decode fast-plan mode; None reads
      DEDUP_MOE_DECODE_FAST_PLAN.
    direct_order: EXPERIMENTAL direct-send destination order
      (tc_sender.DIRECT_ORDERS); None reads DEDUP_MOE_DIRECT_ORDER.
    wave_order: EXPERIMENTAL. "rows" (fewest rows first) or "xboard" (fewest
      cross-board rows per unit of modeled compute first); None reads
      DEDUP_MOE_WAVE_ORDER.
    ramp_ratio: Growth of each wave's modeled compute (wave_plan); None reads
      DEDUP_MOE_RAMP_RATIO, else RAMP_RATIO.
    board_size: Devices per board for the board-aware options; the De-Dup
      tables' board size when dedup_tables is given. None or 0 reads
      DEDUP_MOE_DIRECT_BOARD_SIZE, and 0 there means min(MAX_BOARD_SIZE, ep).
    stage_hoist: EXPERIMENTAL De-Dup stage-send placement
      (tc_sender.STAGE_HOISTS); None reads DEDUP_MOE_STAGE_HOIST.
    pack_parity: Pack semaphores of the packed start kernel
      (tc_sender.PACK_PARITIES); None reads DEDUP_MOE_PACK_PARITY.
    compact_plan: De-Dup plan variant (dedup_plan `compact_plan`) and source_plan
      `fw_only`; None reads DEDUP_MOE_COMPACT_PLAN.
    sc_sender: Issue the packed exchange from the SparseCore
      (sc_sender.sc_ragged_a2a_start_waves); None reads DEDUP_MOE_SC_SENDER.
      `finish` must get the same value.
    row_send_max_t: Row send only when t_local <= this (the packed path
      above it); None reads DEDUP_MOE_ROW_SEND_MAX_T, and unset there means
      every t.

  Returns:
    A WaveDispatch.

  Raises:
    ValueError: If ep < 3 or the scheduling options are inconsistent.
  """
  if ep < 3:
    raise ValueError(f"wave dispatch needs ep >= 3; got ep={ep}")
  t_local = rows.shape[0]
  if pairs_me is not None:
    n_tokens, topk = ep * t_local, pairs_me.shape[1]
  elif pairs is not None:
    if pairs.shape[0] == ep and t_local > 1:
      n_tokens, topk = ep * t_local, pairs.shape[1] // t_local
      pairs_me = lax.dynamic_index_in_dim(
          pairs.astype(jnp.int32),
          jnp.asarray(me, jnp.int32),
          axis=0,
          keepdims=False,
      ).reshape(t_local, topk)
    else:
      n_tokens, topk = pairs.shape
  else:
    raise ValueError("Either pairs or pairs_me must be provided.")
  send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
  min_fanout = _resolve_min_fanout(min_fanout)
  compact_plan = _resolve_compact_plan(compact_plan)
  ramp_ratio = _resolve_ramp_ratio(ramp_ratio)
  direct_order = _DIRECT_ORDER if direct_order is None else direct_order
  wave_order = _WAVE_ORDER if wave_order is None else wave_order
  stage_hoist = _STAGE_HOIST if stage_hoist is None else stage_hoist
  pack_parity = _PACK_PARITY if pack_parity is None else int(pack_parity)
  sc_sender = SC_SENDER if sc_sender is None else bool(sc_sender)
  sched_bs = _sched_board_size(board_size, dedup_tables, ep)
  board_aware = direct_order != "ring" or wave_order != "rows"
  if board_aware and (ep % sched_bs or ep // sched_bs < 2):
    raise ValueError(
        f"direct_order={direct_order!r} / wave_order={wave_order!r} need >= 2"
        f" boards of board_size devices; got ep={ep}, board_size={sched_bs}"
    )
  if wave_order not in WAVE_ORDERS:
    raise ValueError(
        f"wave_order must be one of {WAVE_ORDERS}; got {wave_order!r}"
    )
  # A De-Dup caller sizing its forward tables with forward_runs_needed must
  # pass it the same wave_order / ramp_ratio (it re-derives the waves).
  hist_remote = None
  if wave_order == "xboard":
    src_board = jnp.arange(ep, dtype=jnp.int32) // sched_bs
    exp_board = jnp.arange(ep * g_local, dtype=jnp.int32) // g_local // sched_bs
    hist_remote = jnp.sum(
        jnp.where(src_board[:, None] != exp_board[None, :], rows_by_dest, 0),
        axis=0,
    )
  sched_kw = {}
  if ramp_ratio != RAMP_RATIO:
    sched_kw["ramp_ratio"] = ramp_ratio
  if hist_remote is not None:
    sched_kw["hist_remote"] = hist_remote
  send_kw = {}
  if direct_order != "ring":
    send_kw.update(
        direct_order=direct_order,
        stripe_boards=ep // sched_bs,
        stripe_board_size=sched_bs,
    )
  # The row-send start kernels take the destination order only.
  row_kw = dict(send_kw)
  if stage_hoist != "none" and dedup_tables is not None:
    send_kw["stage_hoist"] = stage_hoist
  if pack_parity != 1:
    send_kw["pack_parity"] = pack_parity
  start_waves = (
      _sc_sender_module().sc_ragged_a2a_start_waves
      if sc_sender
      else dispatch_kernels.ragged_a2a_start_waves
  )
  if row_send is None:
    row_send = _ROW_SEND
  if row_send_max_t is None and _ROW_SEND_MAX_T:
    row_send_max_t = int(_ROW_SEND_MAX_T)
  row_send = (
      bool(row_send)
      and dedup_tables is None
      and (row_send_max_t is None or t_local <= row_send_max_t)
  )
  nw = 1 if scale_bits is None else 2
  fast_mode = (
      source_plan._DECODE_FAST_PLAN  # pylint: disable=protected-access
      if decode_fast_plan is None
      else int(decode_fast_plan)
  )
  if fast_mode > 0 and dedup_tables is None and pairs is not None:
    use_mode2 = fast_mode >= 2 and row_send
    per_dest_perm = row_send
    compute_visit_me = waves == 1
    if source_plan.decode_fast_plan_supported(
        ep=ep,
        g_local=g_local,
        t_local=t_local,
        topk=topk,
        waves=waves,
        nw=nw,
        per_dest_perm=per_dest_perm,
        send_alloc=send_alloc,
        write_headers=False,
        compute_visit_me=compute_visit_me,
        has_wave_bounds=True,
    ):
      me_i32 = jnp.asarray(me, jnp.int32)
      if waves == 1:
        order = None
        wave_of = None
        wave_bounds_me = None
        rows_by_dest_me = lax.dynamic_slice(
            rows_by_dest.astype(jnp.int32), (0, me_i32 * g_local), (ep, g_local)
        )
      else:
        hist = rows_by_dest.sum(axis=0)
        order, wave_of, wave_bounds = wave_plan(
            hist, ep=ep, g_local=g_local, waves=waves, **sched_kw
        )
        wave_bounds_me = lax.dynamic_index_in_dim(
            wave_bounds, me, axis=0, keepdims=False
        )
        rows_by_dest_me = None
      recv_cnt = source_plan.receiver_counts(
          pairs,
          me,
          ep=ep,
          t_local=t_local,
          g_local=g_local,
          waves=waves,
          wave_of=wave_of,
      )
      if hist_me is None:
        hist_me = lax.dynamic_index_in_dim(
            rows_by_dest, me, axis=0, keepdims=False
        )
      tg_shape = (
          ep * source_plan.list_rows(t_local, topk, nw),
          source_plan.LANES,
      )
      send = _uninit_hbm((send_alloc, *rows.shape[1:]), rows.dtype)
      recv = _uninit_hbm((align_up(n_tokens, ROWBLK), *rows.shape[1:]), rows.dtype)
      tg_lists_init = _uninit_hbm(tg_shape, jnp.int32)
      tg_land = _uninit_hbm(tg_shape, jnp.int32)
      if use_mode2:
        if pairs_me is None:
          pairs_me = lax.dynamic_slice(
              pairs.astype(jnp.int32), (me_i32 * t_local, 0), (t_local, topk)
          )
        fut, rows_keep, disp_tab, list_tab, visit_me = (
            dispatch_kernels.ragged_a2a_start_waves_row_send_fast(
                send,
                recv,
                rows,
                tg_lists_init,
                tg_land,
                pairs_me,
                hist_me,
                recv_cnt,
                me,
                ep=ep,
                g_local=g_local,
                waves=waves,
                scale_bits=scale_bits,
                wave_of=wave_of,
                wave_bounds_me=wave_bounds_me,
                rows_by_dest_me=rows_by_dest_me,
                **row_kw,
            )
        )
        if waves == 1:
          assert visit_me is not None
          order = jnp.broadcast_to(visit_me[None, :], (ep, g_local))
        assert order is not None
        return WaveDispatch(
            fut,
            disp_tab,
            order,
            list_tab,
            None,
            None,
            None,
            None,
            wave_of,
            rows_keep,
        )
      disp_tab, perm, list_tab, tg_lists, visit_me = (
          source_plan.decode_fast_plan(
              pairs,
              me,
              hist_me,
              recv_cnt,
              ep=ep,
              t_local=t_local,
              g_local=g_local,
              waves=waves,
              scale_bits=scale_bits,
              wave_of=wave_of,
              wave_bounds_me=wave_bounds_me,
              rows_by_dest_me=rows_by_dest_me,
              per_dest_perm=per_dest_perm,
              send_alloc=send_alloc,
              write_headers=False,
              tg_lists_buf=tg_lists_init,
              pairs_me=pairs_me,
          )
      )
      if waves == 1:
        assert visit_me is not None
        order = jnp.broadcast_to(visit_me[None, :], (ep, g_local))
      assert order is not None
      if row_send:
        fut = dispatch_kernels.ragged_a2a_start_waves_row_send(
            send,
            recv,
            disp_tab,
            me,
            ep=ep,
            waves=waves,
            chunk_rows=CHUNK_ROWS,
            rows=rows,
            perm=perm,
            list_tab=list_tab,
            tg_lists=tg_lists,
            tg_land=tg_land,
            **row_kw,
        )
        fut, rows_keep = tuple(fut[:-1]), fut[-1]
      else:
        fut = start_waves(
            send,
            recv,
            disp_tab,
            me,
            ep=ep,
            waves=waves,
            chunk_rows=CHUNK_ROWS,
            rows=rows,
            perm=perm,
            list_tab=list_tab,
            tg_lists=tg_lists,
            tg_land=tg_land,
            **send_kw,
        )
        rows_keep = None
      return WaveDispatch(
          fut,
          disp_tab,
          order,
          list_tab,
          None,
          None,
          None,
          None,
          wave_of,
          rows_keep,
      )
  if waves == 1 and dedup_tables is None:
    me_i32 = jnp.asarray(me, jnp.int32)
    h_me = lax.dynamic_slice(
        rows_by_dest.astype(jnp.int32), (0, me_i32 * g_local), (ep, g_local)
    ).sum(axis=0)
    _, order_me = ascending_expert_order(h_me, ep=1, g_local=g_local)
    order = jnp.broadcast_to(order_me, (ep, g_local))
    wave_of = None
    wave_bounds_me = jnp.sum((h_me > 0).astype(jnp.int32), keepdims=True)
  else:
    hist = rows_by_dest.sum(axis=0)
    order, wave_of, wave_bounds = wave_plan(
        hist, ep=ep, g_local=g_local, waves=waves, **sched_kw
    )
    wave_bounds_me = lax.dynamic_index_in_dim(
        wave_bounds, me, axis=0, keepdims=False
    )
  plan_p1 = source_plan.source_plan(
      pairs if pairs_me is None else pairs_me,
      me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      waves=waves,
      wave_of=wave_of,
      rank_method=rank_method,
      local=pairs_me is not None,
      **({"fw_only": True} if compact_plan and dedup_tables is not None else {}),
  )
  # Receiver per-wave counts on `me`. Under De-Dup's per-target-wave
  # forwarding every row `me` needs (direct or gateway-forwarded) counts in
  # `me`'s own first-hit wave, so the counts are the direct send's.
  if recv_cnt_override is not None:
    recv_cnt = recv_cnt_override
  else:
    recv_cnt = source_plan.receiver_counts(
        pairs,
        me,
        ep=ep,
        t_local=t_local,
        g_local=g_local,
        waves=waves,
        wave_of=wave_of,
    )
  if dedup_tables is None:
    fwd_schedule = None
    plan = plan_p1
    disp_tab = source_plan.dispatch_table(
        plan, me, recv_cnt, t_local=t_local, wave_bounds_me=wave_bounds_me
    )
    if row_send and t_local <= source_plan._SORT_FREE_MAX_T:  # pylint: disable=protected-access
      perm = source_plan.send_perm_by_dest(plan)
    else:
      perm = source_plan.send_perm(plan, send_alloc=send_alloc)
    stage_send_tab = None
    dplan = None
    stage_buf = None
    num_boards = 0
    num_slots = 0
    board_size = 0
  else:
    fwd_schedule = _fwd_schedule(fwd_max_runs)
    num_boards = dedup_tables.num_boards
    num_slots = dedup_tables.num_slots
    board_size = dedup_tables.board_size
    dplan = dedup_plan_mod.dedup_plan(
        plan_p1.fw,
        me,
        tables=dedup_tables,
        num_waves=waves,
        min_fanout=min_fanout,
        need=plan_p1.need,
        schedule=fwd_schedule,
        compute_fwd_runs=False,
        **({"compact_plan": True} if compact_plan else {}),
    )
    disp_tab = dedup_plan_mod.dedup_disp_tab(dplan, recv_cnt)
    disp_tab = disp_tab.at[wave_bounds_row(ep)].set(wave_bounds_me)
    perm = dplan.perm[:send_alloc]
    stage_send_tab = jnp.concatenate(
        [
            dplan.stage_send_off,
            dplan.cnt_stage,
            dplan.stage_dst,
            jnp.broadcast_to(dplan.gw[:, None], (num_boards, waves)),
            jnp.broadcast_to(dplan.sigma[:, None], (num_boards, waves)),
        ],
        axis=0,
    ).astype(jnp.int32)
    stage_buf = _uninit_hbm(
        (align_up(num_slots * t_local, ROWBLK), *rows.shape[1:]), rows.dtype
    )
    plan = plan_p1._replace(
        cnt_dir=dplan.cnt_dir,
        slot_off=dplan.slot_off,
        row_on_dest=jnp.maximum(dplan.row_on_dest, 0),
        send_off=dplan.send_off,
        send_cnt=dplan.cnt_dir,
    )
  if hist_me is None:
    hist_me = lax.dynamic_index_in_dim(rows_by_dest, me, axis=0, keepdims=False)
  tg_lists = source_plan.build_send_lists(
      pairs if pairs_me is None else pairs_me,
      plan,
      me,
      hist_me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      scale_bits=scale_bits,
      pack_mode=list_pack_mode,
      local=pairs_me is not None,
      include_headers=not source_plan._FAST_GLUE,  # pylint: disable=protected-access
  )
  n_per_dest = source_plan.list_lengths(hist_me, ep=ep, g_local=g_local)
  dma_rows = source_plan.list_dma_rows(n_per_dest, nw)
  remote_list_rows = dma_rows.sum() - lax.dynamic_index_in_dim(
      dma_rows, me, axis=0, keepdims=False
  )
  list_tab = jnp.concatenate([dma_rows, remote_list_rows[None]]).astype(
      jnp.int32
  )
  send = _uninit_hbm((send_alloc, *rows.shape[1:]), rows.dtype)
  # A shard receives each token at most once (in fixed per-source slots).
  recv = _uninit_hbm((align_up(n_tokens, ROWBLK), *rows.shape[1:]), rows.dtype)
  tg_land = _uninit_hbm(tg_lists.shape, tg_lists.dtype)
  if row_send:
    fut = dispatch_kernels.ragged_a2a_start_waves_row_send(
        send,
        recv,
        disp_tab,
        me,
        ep=ep,
        waves=waves,
        chunk_rows=CHUNK_ROWS,
        rows=rows,
        perm=perm,
        list_tab=list_tab,
        tg_lists=tg_lists,
        tg_land=tg_land,
        **row_kw,
    )
  else:
    fut = start_waves(
        send,
        recv,
        disp_tab,
        me,
        ep=ep,
        waves=waves,
        chunk_rows=CHUNK_ROWS,
        rows=rows,
        perm=perm,
        list_tab=list_tab,
        tg_lists=tg_lists,
        tg_land=tg_land,
        stage_send_tab=stage_send_tab,
        stage=stage_buf,
        num_boards=num_boards,
        num_slots=num_slots,
        board_size=board_size,
        **send_kw,
    )
  rows_keep = None
  if row_send:
    fut, rows_keep = tuple(fut[:-1]), fut[-1]
  return WaveDispatch(
      fut,
      disp_tab,
      order,
      list_tab,
      stage_send_tab,
      None,
      dplan,
      fwd_schedule,
      wave_of,
      rows_keep,
  )


def start_early(
    rows,
    pairs_me,
    hist_me,
    me,
    *,
    ep,
    g_local,
    n_tokens,
    scale_bits=None,
    rank_method="matmul",
    list_pack_mode=None,
    row_send=None,
):
  """Issues shard `me`'s direct W=1 exchange from its own routing alone.

  With one wave and fixed per-source slots, every send row, receive row and
  token-gather list depends only on this shard's [t_local, K] selection and
  its per-expert counts. Two disp_tab entries need the exchanged routing:
  the rows `me` receives (row 3 * ep) and its nonempty-expert count (row
  wave_bounds_row(ep)). The start kernel reads neither, so they are left 0
  here and `finish_early` fills them in after the routing all-gather.

  Args:
    rows: [t_local, lane blocks, 128] this shard's token rows on the wire.
    pairs_me: [t_local, topk] i32 this shard's selected expert ids, in mesh
      order (the rows it contributes to the routing all-gather).
    hist_me: [ep * g_local] i32 this shard's per-expert pair counts.
    me: This shard's index.
    ep: Expert-parallel width (>= 3).
    g_local: Experts per shard.
    n_tokens: Tokens across all shards (receive buffer height).
    scale_bits: Optional [t_local] i32 activation-scale bits (fp8 rows).
    rank_method: `matmul` or `cumsum`.
    list_pack_mode: source_plan.build_send_lists `send_list_gather`, or None.
    row_send: Send rows straight from `rows`; None reads DEDUP_MOE_ROW_SEND.

  Returns:
    A WaveDispatch whose disp_tab rows 3 * ep and wave_bounds_row(ep) and
    visit_order are placeholders until `finish_early`.

  Raises:
    ValueError: If ep < 3.
  """
  if ep < 3:
    raise ValueError(f"wave dispatch needs ep >= 3; got ep={ep}")
  t_local, topk = pairs_me.shape
  waves = 1
  plan = source_plan.source_plan(
      pairs_me,
      me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      waves=waves,
      wave_of=None,
      rank_method=rank_method,
      local=True,
  )
  zero_w = jnp.zeros((waves,), jnp.int32)
  disp_tab = source_plan.dispatch_table(
      plan, me, zero_w, t_local=t_local, wave_bounds_me=zero_w
  )
  send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
  if row_send is None:
    row_send = _ROW_SEND
  row_send = bool(row_send)
  if row_send and t_local <= source_plan._SORT_FREE_MAX_T:  # pylint: disable=protected-access
    perm = source_plan.send_perm_by_dest(plan)
  else:
    perm = source_plan.send_perm(plan, send_alloc=send_alloc)
  nw = 1 if scale_bits is None else 2
  tg_lists = source_plan.build_send_lists(
      pairs_me,
      plan,
      me,
      hist_me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      scale_bits=scale_bits,
      pack_mode=list_pack_mode,
      local=True,
      include_headers=not source_plan._FAST_GLUE,  # pylint: disable=protected-access
  )
  n_per_dest = source_plan.list_lengths(hist_me, ep=ep, g_local=g_local)
  dma_rows = source_plan.list_dma_rows(n_per_dest, nw)
  remote_list_rows = dma_rows.sum() - lax.dynamic_index_in_dim(
      dma_rows, me, axis=0, keepdims=False
  )
  list_tab = jnp.concatenate([dma_rows, remote_list_rows[None]]).astype(
      jnp.int32
  )
  send = _uninit_hbm((send_alloc, *rows.shape[1:]), rows.dtype)
  recv = _uninit_hbm((align_up(n_tokens, ROWBLK), *rows.shape[1:]), rows.dtype)
  tg_land = _uninit_hbm(tg_lists.shape, tg_lists.dtype)
  if row_send:
    fut = dispatch_kernels.ragged_a2a_start_waves_row_send(
        send,
        recv,
        disp_tab,
        me,
        ep=ep,
        waves=waves,
        chunk_rows=CHUNK_ROWS,
        rows=rows,
        perm=perm,
        list_tab=list_tab,
        tg_lists=tg_lists,
        tg_land=tg_land,
    )
    fut, rows_keep = tuple(fut[:-1]), fut[-1]
  else:
    fut = dispatch_kernels.ragged_a2a_start_waves(
        send,
        recv,
        disp_tab,
        me,
        ep=ep,
        waves=waves,
        chunk_rows=CHUNK_ROWS,
        rows=rows,
        perm=perm,
        list_tab=list_tab,
        tg_lists=tg_lists,
        tg_land=tg_land,
    )
    rows_keep = None
  order = jnp.zeros((ep, g_local), jnp.int32)
  return WaveDispatch(
      fut, disp_tab, order, list_tab, None, None, None, None, None, rows_keep
  )


def finish_early(wd, pairs, rows_by_dest, me, *, ep, g_local, t_local=None):
  """Fills in the routing-dependent parts of a `start_early` dispatch.

  Args:
    wd: The WaveDispatch from `start_early`.
    pairs: [T, topk] or compact [ep, t_local * topk] i32 exchanged expert ids,
      in mesh order.
    rows_by_dest: [ep, ep * g_local] i32 exchanged per-expert pair counts.
    me: This shard's index.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    t_local: Optional tokens per shard (required when `pairs` is compact).

  Returns:
    `wd` with the received-row count and wave bound in disp_tab and the
    visit order filled in: the same values `start` builds at W=1.
  """
  if t_local is None:
    t_local = pairs.shape[0] // ep
  me_i32 = jnp.asarray(me, jnp.int32)
  h_me = lax.dynamic_slice(
      rows_by_dest.astype(jnp.int32), (0, me_i32 * g_local), (ep, g_local)
  ).sum(axis=0)
  _, order_me = ascending_expert_order(h_me, ep=1, g_local=g_local)
  order = jnp.broadcast_to(order_me, (ep, g_local))
  wave_bounds_me = jnp.sum((h_me > 0).astype(jnp.int32), keepdims=True)
  recv_cnt = source_plan.receiver_counts(
      pairs,
      me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      waves=1,
      wave_of=None,
  )
  tab = wd.disp_tab
  wb = wave_bounds_row(ep)
  disp_tab = jnp.concatenate(
      [
          tab[: 3 * ep],
          recv_cnt.astype(jnp.int32)[None, :],
          tab[3 * ep + 1 : wb],
          wave_bounds_me.astype(jnp.int32)[None, :],
          tab[wb + 1 :],
      ],
      axis=0,
  )
  return wd._replace(disp_tab=disp_tab, visit_order=order)


def w1_dest_counts(pairs_me, *, ep, g_local):
  """Returns `[ep]` i32 counts of local tokens targeting each destination."""
  pk = pairs_me.astype(jnp.int32).T
  dest = pk // g_local
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  return jnp.sum(
      jnp.any(dest[:, None, :] == d_iota[None, :, None], axis=0).astype(
          jnp.int32
      ),
      axis=1,
  )


def routing_ag_start(packed, me, *, ep, direct_order=None, board_size=None):
  """Starts split-phase Pallas routing all-gather of 1D `packed` i32."""
  direct_order = _DIRECT_ORDER if direct_order is None else direct_order
  sched_bs = _sched_board_size(board_size, None, ep)
  ag_kw = {}
  if direct_order != "ring":
    ag_kw.update(
        direct_order=direct_order,
        stripe_boards=ep // sched_bs,
        stripe_board_size=sched_bs,
    )
  width = packed.shape[0]
  tile_words = ROWBLK * source_plan.LANES
  n_sub = max(ROWBLK, -(-width // tile_words) * ROWBLK)
  pad_w = n_sub * source_plan.LANES - width
  packed_padded = jnp.pad(packed, (0, pad_w)) if pad_w else packed
  packed_tile = packed_padded.reshape(
      1, n_sub, source_plan.LANES
  ).astype(jnp.int32)
  out_tile = _uninit_hbm((ep, n_sub, source_plan.LANES), jnp.int32)
  return dispatch_kernels.routing_ag_start(
      packed_tile, out_tile, me, ep=ep, **ag_kw
  )


def routing_ag_done(ag_fut, *, ep, width):
  """Completes `routing_ag_start` and returns `[ep, width]` i32."""
  out_tile = dispatch_kernels.routing_ag_done(ag_fut, ep=ep)
  out_2d = out_tile.reshape(ep, -1)
  return out_2d[:, :width] if out_2d.shape[1] != width else out_2d


class PreparedLocalW1(NamedTuple):
  """Local W=1 dispatch arrays built while routing all-gather is in flight."""

  disp_tab_base: jax.Array
  perm: jax.Array
  tg_lists: jax.Array
  list_tab: jax.Array


def prepare_local_w1(
    pairs_me,
    hist_me,
    me,
    *,
    ep,
    g_local,
    scale_bits=None,
    rank_method="matmul",
    list_pack_mode=None,
    row_send=None,
    row_send_max_t=None,
) -> PreparedLocalW1:
  """Builds local W=1 dispatch plan and send lists without exchanged routing."""
  if ep < 3:
    raise ValueError(f"wave dispatch needs ep >= 3; got ep={ep}")
  t_local, topk = pairs_me.shape
  plan = source_plan.source_plan(
      pairs_me,
      me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      waves=1,
      wave_of=None,
      rank_method=rank_method,
      local=True,
  )
  zero_w = jnp.zeros((1,), jnp.int32)
  disp_tab_base = source_plan.dispatch_table(
      plan, me, zero_w, t_local=t_local, wave_bounds_me=zero_w
  )
  send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
  if row_send is None:
    row_send = _ROW_SEND
  if row_send_max_t is None and _ROW_SEND_MAX_T:
    row_send_max_t = int(_ROW_SEND_MAX_T)
  row_send = bool(row_send) and (
      row_send_max_t is None or t_local <= row_send_max_t
  )
  if row_send and t_local <= source_plan._SORT_FREE_MAX_T:  # pylint: disable=protected-access
    perm = source_plan.send_perm_by_dest(plan)
  else:
    perm = source_plan.send_perm(plan, send_alloc=send_alloc)
  nw = 1 if scale_bits is None else 2
  tg_lists = source_plan.build_send_lists(
      pairs_me,
      plan,
      me,
      hist_me,
      ep=ep,
      t_local=t_local,
      g_local=g_local,
      scale_bits=scale_bits,
      pack_mode=list_pack_mode,
      local=True,
      include_headers=not source_plan._FAST_GLUE,  # pylint: disable=protected-access
  )
  n_per_dest = source_plan.list_lengths(hist_me, ep=ep, g_local=g_local)
  dma_rows = source_plan.list_dma_rows(n_per_dest, nw).astype(jnp.int32)
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  remote_list_rows = jnp.sum(jnp.where(d_iota != me, dma_rows, 0))
  list_tab = jnp.where(
      jnp.arange(ep + 1, dtype=jnp.int32) < ep,
      jnp.pad(dma_rows, (0, 1)),
      remote_list_rows,
  )
  return PreparedLocalW1(disp_tab_base, perm, tg_lists, list_tab)


def launch_prepared_w1(
    prepared: PreparedLocalW1,
    rows,
    rows_by_dest,
    me,
    *,
    ep,
    g_local,
    n_tokens,
    topk,
    recv_cnt=None,
    pairs=None,
    row_send=None,
    row_send_max_t=None,
    direct_order=None,
    board_size=None,
    pack_parity=None,
    sc_sender=None,
    **unused_kw,
) -> WaveDispatch:
  """Completes `disp_tab` and launches W=1 wave dispatch from `prepared`."""
  del unused_kw
  t_local = rows.shape[0]
  send_alloc = align_up(t_local * min(topk, ep), ROWBLK)
  direct_order = _DIRECT_ORDER if direct_order is None else direct_order
  pack_parity = _PACK_PARITY if pack_parity is None else int(pack_parity)
  sc_sender = SC_SENDER if sc_sender is None else bool(sc_sender)
  sched_bs = _sched_board_size(board_size, None, ep)
  send_kw = {}
  if direct_order != "ring":
    send_kw.update(
        direct_order=direct_order,
        stripe_boards=ep // sched_bs,
        stripe_board_size=sched_bs,
    )
  row_kw = dict(send_kw)
  if pack_parity != 1:
    send_kw["pack_parity"] = pack_parity
  if row_send is None:
    row_send = _ROW_SEND
  if row_send_max_t is None and _ROW_SEND_MAX_T:
    row_send_max_t = int(_ROW_SEND_MAX_T)
  row_send = bool(row_send) and (
      row_send_max_t is None or t_local <= row_send_max_t
  )
  me_i32 = jnp.asarray(me, jnp.int32)
  h_me = lax.dynamic_slice(
      rows_by_dest.astype(jnp.int32), (0, me_i32 * g_local), (ep, g_local)
  ).sum(axis=0)
  _, order_me = ascending_expert_order(h_me, ep=1, g_local=g_local)
  order = jnp.broadcast_to(order_me, (ep, g_local))
  wave_bounds_me = jnp.sum((h_me > 0).astype(jnp.int32), keepdims=True)
  if recv_cnt is None:
    assert pairs is not None
    recv_cnt = source_plan.receiver_counts(
        pairs,
        me,
        ep=ep,
        t_local=t_local,
        g_local=g_local,
        waves=1,
        wave_of=None,
    )
  tab = prepared.disp_tab_base
  wb = wave_bounds_row(ep)
  row = lax.broadcasted_iota(jnp.int32, tab.shape, 0)
  disp_tab = jnp.where(
      row == 3 * ep,
      recv_cnt.astype(jnp.int32)[None, :],
      jnp.where(row == wb, wave_bounds_me.astype(jnp.int32)[None, :], tab),
  )
  send = _uninit_hbm((send_alloc, *rows.shape[1:]), rows.dtype)
  recv = _uninit_hbm((align_up(n_tokens, ROWBLK), *rows.shape[1:]), rows.dtype)
  tg_land = _uninit_hbm(prepared.tg_lists.shape, prepared.tg_lists.dtype)
  if row_send:
    fut = dispatch_kernels.ragged_a2a_start_waves_row_send(
        send,
        recv,
        disp_tab,
        me,
        ep=ep,
        waves=1,
        chunk_rows=CHUNK_ROWS,
        rows=rows,
        perm=prepared.perm,
        list_tab=prepared.list_tab,
        tg_lists=prepared.tg_lists,
        tg_land=tg_land,
        **row_kw,
    )
    fut, rows_keep = tuple(fut[:-1]), fut[-1]
  else:
    start_waves = (
        _sc_sender_module().sc_ragged_a2a_start_waves
        if sc_sender
        else dispatch_kernels.ragged_a2a_start_waves
    )
    fut = start_waves(
        send,
        recv,
        disp_tab,
        me,
        ep=ep,
        waves=1,
        chunk_rows=CHUNK_ROWS,
        rows=rows,
        perm=prepared.perm,
        list_tab=prepared.list_tab,
        tg_lists=prepared.tg_lists,
        tg_land=tg_land,
        **send_kw,
    )
    rows_keep = None
  return WaveDispatch(
      fut,
      disp_tab,
      order,
      prepared.list_tab,
      None,
      None,
      None,
      None,
      None,
      rows_keep,
  )


def materialize_smem_ops(
    wd: WaveDispatch,
    pairs: jax.Array,
    me,
    *,
    t_local: int,
    waves: int,
    dedup_tables,
    min_fanout: int = 2,
    compact_plan: bool | None = None,
) -> WaveDispatch:
  """Builds the gateway SMEM operands (`stage_tab`, `fwd_tab`, `fwd_cnt`).

  Args:
    wd: The WaveDispatch returned by `start`.
    pairs: [T, topk] i32 expert ids in mesh order.
    me: This shard's index.
    t_local: Tokens per shard.
    waves: W.
    dedup_tables: TopologyTables, or None without De-Dup.
    min_fanout: As in `start`.
    compact_plan: As in `start`.

  Returns:
    `wd` with `smem_ops` filled in, or `wd` unchanged without De-Dup or when
    already built.
  """
  if dedup_tables is None or wd.smem_ops is not None:
    return wd
  assert wd.wave_of is not None
  assert wd.fwd_schedule is not None
  assert wd.dplan is not None
  fw_board = dedup_plan_mod.served_fw(
      pairs,
      wd.wave_of,
      me,
      tables=dedup_tables,
      t=t_local,
      num_waves=waves,
  )
  gplan = dedup_plan_mod.gateway_plan(
      fw_board,
      me,
      tables=dedup_tables,
      num_waves=waves,
      min_fanout=_resolve_min_fanout(min_fanout),
      schedule=wd.fwd_schedule,
      **({"compact_plan": True} if _resolve_compact_plan(compact_plan) else {}),
  )
  smem_ops = dedup_plan_mod.pack_smem_operands(wd.dplan, gplan)
  return wd._replace(smem_ops=smem_ops)


def forward_runs_needed(
    pairs,
    *,
    ep,
    g_local,
    waves,
    t_local,
    dedup_tables,
    min_fanout=2,
    wave_order=None,
    ramp_ratio=None,
):
  """The most forward runs any (gateway, served source, wave) needs.

  Per-target-wave runs have no static bound (dedup_plan.FwdSchedule), so a
  caller that knows its routing sizes the layer's dedup_fwd_max_runs with
  this before building the layer: every gateway's own `gateway_plan` (same
  stage selection, stage order and run split as `start`), for all ep
  gateways.

  Args:
    pairs: [T, topk] i32 global expert ids in mesh order, as `start` takes.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    waves: W.
    t_local: Tokens per shard.
    dedup_tables: The De-Dup topology tables.
    min_fanout: As `start`.
    wave_order: As `start`.
    ramp_ratio: As `start`.

  Returns:
    i32 scalar.
  """
  min_fanout = _resolve_min_fanout(min_fanout)
  ramp_ratio = _resolve_ramp_ratio(ramp_ratio)
  wave_order = _WAVE_ORDER if wave_order is None else wave_order
  pairs = jnp.asarray(pairs, jnp.int32)
  hist = jnp.bincount(pairs.reshape(-1), length=ep * g_local)
  sched_kw = {}
  if ramp_ratio != RAMP_RATIO:
    sched_kw["ramp_ratio"] = ramp_ratio
  if wave_order == "xboard":
    bs = dedup_tables.board_size
    src_board = (jnp.arange(pairs.shape[0], dtype=jnp.int32) // t_local) // bs
    exp_board = pairs // g_local // bs
    remote = (exp_board != src_board[:, None]).reshape(-1)
    sched_kw["hist_remote"] = jnp.bincount(
        pairs.reshape(-1), weights=remote.astype(jnp.int32), length=ep * g_local
    ).astype(jnp.int32)
  _, wave_of, _ = wave_plan(
      hist, ep=ep, g_local=g_local, waves=waves, **sched_kw
  )
  schedule = _fwd_schedule(dedup_plan_mod.MAX_FWD_RUNS)

  def one(gw):
    fws = dedup_plan_mod.served_fw(
        pairs, wave_of, gw, tables=dedup_tables, t=t_local, num_waves=waves
    )
    fwd_cnt = dedup_plan_mod.gateway_forward_runs_peak(
        fws,
        gw,
        tables=dedup_tables,
        num_waves=waves,
        min_fanout=min_fanout,
        schedule=schedule,
    )
    return jnp.max(fwd_cnt)

  return jnp.max(lax.map(one, jnp.arange(ep, dtype=jnp.int32)))


def finish(wd, arrivals, *, ep, sc_sender=None):
  """Drain this shard's outgoing copies; returns `arrivals` (aliased).

  Called after the fused kernel, which gave the copies the whole kernel to
  finish. Threading the kernel's arrivals through orders the drain after
  the kernel and keeps it from being dead-code-eliminated
  (tc_sender.ragged_a2a_send_done), and keeps them in HBM.

  Args:
    wd: The WaveDispatch returned by `start`.
    arrivals: The fused kernel's arrivals.
    ep: Expert-parallel width.
    sc_sender: As `start` (the SparseCore drain); None reads
      DEDUP_MOE_SC_SENDER.

  Returns:
    `arrivals`, aliased.
  """
  send_sem, _, send, _, list_send_sem, _, tg_lists, *_ = wd.fut
  sc_sender = SC_SENDER if sc_sender is None else bool(sc_sender)
  if sc_sender and wd.rows_keep is None:
    return _sc_sender_module().sc_ragged_a2a_send_done(
        send_sem,
        send,
        wd.disp_tab,
        arrivals,
        ep=ep,
        list_tab=wd.list_tab,
        tg_lists=tg_lists,
        list_send_sem=list_send_sem,
        stage_send_tab=wd.stage_send_tab,
    )
  if wd.rows_keep is not None:
    return dispatch_kernels.ragged_a2a_send_done_row_send(
        send_sem,
        send,
        wd.disp_tab,
        arrivals,
        ep=ep,
        list_tab=wd.list_tab,
        tg_lists=tg_lists,
        list_send_sem=list_send_sem,
        rows_keep=wd.rows_keep,
    )
  return dispatch_kernels.ragged_a2a_send_done(
      send_sem,
      send,
      wd.disp_tab,
      arrivals,
      ep=ep,
      list_tab=wd.list_tab,
      tg_lists=tg_lists,
      list_send_sem=list_send_sem,
  )
