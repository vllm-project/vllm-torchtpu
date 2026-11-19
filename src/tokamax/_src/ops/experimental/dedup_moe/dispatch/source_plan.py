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
"""Sender-local wave dispatch plan and token-gather lists.

The reference dispatch plan (`wave_dispatch.dispatch_plan`) evaluates the
expert-wave plan for every
global token on every shard (O(K * ep * T) for the first-hit waves and an
[ep, W, ep, t] one-hot) and lays the receive buffer out (wave, source)-major,
which needs a prefix over all sources and a `posmap [T]` per shard.

This module evaluates the same plan for ONE source s over its own t tokens
(convention D1 of dedup_plan.py) and lays every receive buffer out in fixed
per-source slots (D2):
rows [s * t, (s + 1) * t) of every destination's receive buffer belong to
source s, (wave, token) ordered inside the slot. Every receive row is then
known to the source alone, so the source also tells each destination where its
rows landed: one int32 list per (source -> destination), already in the
destination's slab order, behind a 128-word header.
`tg_assemble.py` turns the landed lists into the fused kernel's token-gather
table. Which (token, destination) pairs are sent, and in which wave, is
exactly the reference plan's; only where they land changes.

Everything here is pure jnp built from compares, MXU one-hot products, matmul
ranks, one sort and dynamic slices: no XLA scatter and no element gather.
"""

import math
import os
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.fused_moe import host as fused_host

align_up = fused_host.align_up

LANES = 128

# The header block in front of every (source -> destination) list. One
# block goes to EVERY destination, even with no entries, so a receiver always
# waits for a static ep * 512 B of headers plus the bodies.
HDR_WORDS = LANES
MAX_WAVES = 16
HDR_N = 0  # entries in the list, n(s -> d)
HDR_SRC = 1  # the source shard s
HDR_WAVES = 2  # W
HDR_NW = 3  # int32 words per entry (planes in the body)
HDR_SEND_OFF = 8  # [MAX_WAVES] s's send offset of (d, w)
HDR_CNT_DIR = HDR_SEND_OFF + MAX_WAVES  # [MAX_WAVES] rows of (d, w) in slot s
HDR_N_FWD = (
    HDR_CNT_DIR + MAX_WAVES
)  # [MAX_WAVES] De-Dup: forwarded rows (unused)
HDR_STAGE_SEND_OFF = HDR_N_FWD + MAX_WAVES  # [MAX_WAVES] De-Dup (unused)
HDR_CNT_STAGE = HDR_STAGE_SEND_OFF + MAX_WAVES  # [MAX_WAVES] De-Dup (unused)
assert HDR_CNT_STAGE + MAX_WAVES <= HDR_WORDS

# Exclusive ranks along t use one [t, t] strictly upper triangular matmul up
# to this many tokens and RANK_BLOCK-wide blocks above it (a full tri is t^2
# bf16: 32 MiB at t = 4096).
RANK_FULL_MAX = 1024
RANK_BLOCK = 512
RANK_METHODS = ("matmul", "cumsum")
SEND_LIST_GATHER_MODES = (
    "loop",
    "slice",
    "vec",
    "flat",
    "gather",
    "vmap",
    "shift",
    "pallas",
    "scatter_set",
    "compact",
)
LIST_PACK_MODES = SEND_LIST_GATHER_MODES


def list_capacity(t_local, topk):
  """Returns the most entries one (source -> destination) list can hold.

  Lmax: the most entries one (source -> destination) list can hold.

  Every pair of the source can compute on one destination. That is t * K
  and not t * min(K, g): hash-routed ids may repeat an expert within a
  token, and each repeat is a pair of its own (dropless).

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.

  Returns:
    t_local * topk rounded up to whole 128-word rows.
  """
  return align_up(t_local * topk, LANES)


def list_row_words(t_local, topk, nw):
  """Returns the words of one source's row of the list buffers.

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.
    nw: Planes per entry (1, or 2 with scale bits).

  Returns:
    The header words plus `nw` planes of `list_capacity` words.
  """
  return HDR_WORDS + nw * list_capacity(t_local, topk)


def list_rows(t_local, topk, nw):
  """Returns one source's row of the list buffers in [*, 128] rows.

  Args:
    t_local: Tokens per shard.
    topk: Experts per token.
    nw: Planes per entry.

  Returns:
    list_row_words(...) // 128.
  """
  return list_row_words(t_local, topk, nw) // LANES


def list_dma_rows(n, nw):
  """Returns the [*, 128] rows one list DMA moves.

  Args:
    n: Entries in the list.
    nw: Planes per entry.

  Returns:
    One header row plus `nw` planes of ceil(n / 128) rows.
  """
  return 1 + nw * ((n + (LANES - 1)) // LANES)


class SourcePlan(NamedTuple):
  """Source s's half of the expert-wave dispatch plan.

  dest [K, t] i32: destination shard of every pair (pk // g).
  wave_pair [K, t] i32: the wave of every pair on its destination.
  fw [ep, t] i32: first-hit wave of each token on each shard (W: no hit).
  need [ep, t] bool: fw < W.
  oh_dir [ep, W, t] i32: fw == w, the rows s sends directly in wave w.
  cnt_dir [ep, W] i32: rows per (d, w).
  slot_off [ep, W] i32: where (d, w)'s rows start inside slot s on d.
  rank_dir [ep, W, t] i32: exclusive rank of each token within (d, w).
  row_on_dest [ep, t] i32: the receive row of token t on d,
    s * t + slot_off[d, fw] + rank (0 where d does not need t).
  send_off [ep, W] i32: where (d, w)'s rows start in s's send buffer.
  send_cnt [ep, W] i32: rows s sends per (d, w) (= cnt_dir without De-Dup).
  """

  dest: jax.Array
  wave_pair: jax.Array
  fw: jax.Array
  need: jax.Array
  oh_dir: jax.Array
  cnt_dir: jax.Array
  slot_off: jax.Array
  rank_dir: jax.Array
  row_on_dest: jax.Array
  send_off: jax.Array
  send_cnt: jax.Array


def exclusive_ranks(x, *, method="matmul"):
  """Returns the exclusive prefix count of a 0/1 array along its last axis.

  [rows, t] 0/1 -> exclusive prefix count along t, i32.

  A strictly upper triangular ones matmul (0/1 in bf16, exact f32 sums), as
  the reference dispatch_plan; above RANK_FULL_MAX tokens the tri is RANK_BLOCK
  wide and the block totals are prefixed separately. `method="cumsum"` uses a
  direct integer prefix sum along the last axis.

  Args:
    x: [rows, t] 0/1 values.
    method: `matmul` (default) or `cumsum`.

  Returns:
    [rows, t] i32 exclusive prefix counts.
  """
  if method not in RANK_METHODS:
    raise ValueError(f"method={method!r} is not one of {RANK_METHODS}")
  if method == "cumsum":
    xi = x.astype(jnp.int32)
    return jnp.cumsum(xi, axis=1) - xi
  rows, t = x.shape
  if t <= RANK_FULL_MAX:
    tri = jnp.triu(jnp.ones((t, t), jnp.bfloat16), k=1)
    return jnp.dot(
        x.astype(jnp.bfloat16), tri, preferred_element_type=jnp.float32
    ).astype(jnp.int32)
  nb = -(-t // RANK_BLOCK)
  tp = nb * RANK_BLOCK
  xp = jnp.pad(x.astype(jnp.int32), ((0, 0), (0, tp - t))).reshape(
      rows, nb, RANK_BLOCK
  )
  tri = jnp.triu(jnp.ones((RANK_BLOCK, RANK_BLOCK), jnp.bfloat16), k=1)
  within = jnp.einsum(
      "rnb,bc->rnc",
      xp.astype(jnp.bfloat16),
      tri,
      preferred_element_type=jnp.float32,
  ).astype(jnp.int32)
  tot = xp.sum(axis=2)
  before = jnp.cumsum(tot, axis=1) - tot
  return (within + before[:, :, None]).reshape(rows, tp)[:, :t]


def source_plan(
    pairs_g,
    s,
    *,
    ep,
    t_local,
    g_local,
    waves,
    wave_of,
    staged=None,
    rank_method="matmul",
    fw_only: bool = False,
    local=False,
):
  """Builds source s's dispatch plan: per-pair waves and D2 receive rows.

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order, or [t_local, K] of
      source s alone when `local`.
    s: Source shard (int or traced scalar).
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    wave_of: [ep, g] i32 wave of every shard's local experts
      (wave_dispatch.wave_plan).
    staged: Optional [ep, t] bool De-Dup mask of (dest, token) pairs taken out
      of the direct rows.
    rank_method: `matmul` (default) or `cumsum` for exclusive_ranks.
    fw_only: When True (De-Dup caller that replaces cnt_dir/slot_off/row_on_dest
      with dedup_plan's), skip the direct-only oh_dir/exclusive_ranks work; those
      fields are empty.
    local: `pairs_g` holds only source s's rows.

  Returns:
    A SourcePlan; its per-pair wave, first-hit wave, send rows and send offsets
    match the reference dispatch plan.

  Raises:
    ValueError: If waves exceeds MAX_WAVES (the list header width).
  """
  _, topk = pairs_g.shape
  if waves > MAX_WAVES:
    raise ValueError(f"waves={waves} > {MAX_WAVES} (list header width)")
  t = t_local
  s = jnp.asarray(s, jnp.int32)
  # The token axis stays minor, as in the reference plan: [K, t], not [t, K].
  if local:
    pk = pairs_g.astype(jnp.int32).T
  else:
    pk = lax.dynamic_slice(
        pairs_g.astype(jnp.int32), (s * t, 0), (t, topk)
    ).T
  dest = pk // g_local
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  if waves == 1:
    # Single-wave direct dispatch: every expert is in wave 0, so a token needs
    # shard d in wave 0 iff any of its topk pairs targets d.
    wave_pair = jnp.zeros_like(dest)
    hit = jnp.any(dest[:, None, :] == d_iota[None, :, None], axis=0)
    fw = jnp.where(hit, jnp.int32(0), jnp.int32(1))
  else:
    table = wave_of.astype(jnp.int32)
    # The MXU lookup (one-hot of the local expert times the table, then a
    # select by destination), over this source's t tokens instead of T.
    g_iota = jnp.arange(g_local, dtype=jnp.int32)
    local = (pk - dest * g_local).reshape(1, topk * t)
    onehot = (local == g_iota[:, None]).astype(jnp.bfloat16)  # [g, K * t]
    by_dest = jnp.dot(
        table.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
    )  # [ep, K * t]
    wave_pair = (
        jnp.sum(
            jnp.where(
                dest.reshape(1, topk * t) == d_iota[:, None], by_dest, 0.0
            ),
            axis=0,
        )
        .astype(jnp.int32)
        .reshape(topk, t)
    )
    fw = jnp.min(
        jnp.where(
            dest[:, None, :] == d_iota[None, :, None],
            wave_pair[:, None, :],
            waves,
        ),
        axis=0,
    )  # [ep, t]
  need = fw < waves
  if fw_only:
    empty = jnp.zeros((0,), jnp.int32)
    return SourcePlan(
        dest=dest,
        wave_pair=wave_pair,
        fw=fw,
        need=need,
        oh_dir=empty,
        cnt_dir=empty,
        slot_off=empty,
        rank_dir=empty,
        row_on_dest=empty,
        send_off=empty,
        send_cnt=empty,
    )
  if waves == 1:
    oh_dir = need[:, None, :].astype(jnp.int32)
    if staged is not None:
      oh_dir = jnp.where(staged[:, None, :], 0, oh_dir)
    cnt_dir = oh_dir.sum(axis=2)  # [ep, 1]
    slot_off = jnp.zeros((ep, 1), jnp.int32)
    rank_dir = exclusive_ranks(
        oh_dir.reshape(ep, t), method=rank_method
    ).reshape(ep, 1, t)
    direct = need if staged is None else (oh_dir[:, 0, :] > 0)
    row_on_dest = jnp.where(direct, s * t + rank_dir[:, 0, :], 0)
  else:
    w_iota = jnp.arange(waves, dtype=jnp.int32)
    oh_dir = (fw[:, None, :] == w_iota[None, :, None]).astype(jnp.int32)
    if staged is not None:
      oh_dir = jnp.where(staged[:, None, :], 0, oh_dir)
    cnt_dir = oh_dir.sum(axis=2)  # [ep, W]
    slot_off = jnp.cumsum(cnt_dir, axis=1) - cnt_dir
    rank_dir = exclusive_ranks(
        oh_dir.reshape(ep * waves, t), method=rank_method
    ).reshape(ep, waves, t)
    direct = oh_dir.sum(axis=1) > 0
    row_on_dest = jnp.where(
        direct,
        s * t + jnp.sum(oh_dir * (slot_off[:, :, None] + rank_dir), axis=1),
        0,
    )
  if _FAST_GLUE_V2 and waves == 1 and ep <= 64:
    e_iota = jnp.arange(ep, dtype=jnp.int32)
    send_off = jnp.sum(
        jnp.where(
            e_iota[None, :, None] < e_iota[:, None, None],
            cnt_dir[None, :, :],
            0,
        ),
        axis=1,
    )
  else:
    flat = cnt_dir.reshape(-1)
    send_off = (jnp.cumsum(flat) - flat).reshape(ep, waves)
  return SourcePlan(
      dest=dest,
      wave_pair=wave_pair,
      fw=fw,
      need=need,
      oh_dir=oh_dir,
      cnt_dir=cnt_dir,
      slot_off=slot_off,
      rank_dir=rank_dir,
      row_on_dest=row_on_dest,
      send_off=send_off,
      send_cnt=cnt_dir,
  )


def send_perm(plan, *, send_alloc, method=None):
  """Returns the local token of each packed send row.

  Args:
    plan: The SourcePlan of s = me.
    send_alloc: Rows of the send buffer.
    method: `sort` (a key-value sort of the ep * t candidates) or `onehot`
      (each slot selects the one candidate whose position it is: a compare
      over [send_alloc, ep * t], no sort). None: `onehot` when t <=
      `_SORT_FREE_MAX_T`, else `sort`. Rows past the sent total differ (0
      vs unsent tokens) and are never read.

  Returns:
    [send_alloc] i32 tokens in (dest, wave, token) order, the reference plan's
    `perm`.
  """
  ep, _, t = plan.oh_dir.shape
  if method is None:
    method = "onehot" if t <= _SORT_FREE_MAX_T else "sort"
  if method not in ("sort", "onehot"):
    raise ValueError(f"method={method!r} is not sort or onehot")
  pos = plan.send_off[:, :, None] + plan.rank_dir
  pos_dt = jnp.sum(plan.oh_dir * pos, axis=1)  # [d, t]
  valid = plan.oh_dir.sum(axis=1) > 0
  t_idx = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (ep, t))
  if method == "onehot":
    key = jnp.where(valid, pos_dt, -1).reshape(-1)
    slot = jnp.arange(send_alloc, dtype=jnp.int32)
    return jnp.sum(
        jnp.where(key[None, :] == slot[:, None], t_idx.reshape(1, -1), 0),
        axis=1,
    ).astype(jnp.int32)
  tok_bits = max(1, (t - 1).bit_length())
  if ep * t <= (1 << (30 - tok_bits)):
    packed = jnp.where(
        valid,
        (pos_dt << tok_bits) | t_idx,
        jnp.int32(1 << 30) | t_idx,
    ).reshape(-1)
    perm = lax.sort(packed, dimension=0, is_stable=False) & jnp.int32(
        (1 << tok_bits) - 1
    )
    return perm[:send_alloc]
  _, perm = lax.sort_key_val(
      jnp.where(valid, pos_dt, jnp.int32(2**30)).reshape(-1),
      t_idx.reshape(-1),
      is_stable=False,
  )
  return perm[:send_alloc]


def send_perm_by_dest(plan):
  """Returns `[ep, t_local]` local tokens per destination in slot order.

  Inverts `slot_off + rank_dir` within each destination (`[ep, t_local, t_local]`
  instead of `[send_alloc, ep * t_local]`). For each destination `d`, wave `w`
  starts at `slot_off[d, w]` inside row `d`.

  Args:
    plan: The SourcePlan of s = me.

  Returns:
    [ep, t_local] i32 local token indices per destination slot.
  """
  _, waves, t = plan.oh_dir.shape
  if waves == 1:
    pos_dt = plan.rank_dir[:, 0, :]
    valid = plan.oh_dir[:, 0, :] > 0
  else:
    pos = plan.slot_off[:, :, None] + plan.rank_dir
    pos_dt = jnp.sum(plan.oh_dir * pos, axis=1)
    valid = plan.oh_dir.sum(axis=1) > 0
  key = jnp.where(valid, pos_dt, -1)
  slot = jnp.arange(t, dtype=jnp.int32)
  if _FAST_GLUE_V2 and t <= 256:
    eq_bf16 = (key[:, None, :] == slot[None, :, None]).astype(jnp.bfloat16)
    return jnp.dot(
        eq_bf16, slot.astype(jnp.bfloat16), preferred_element_type=jnp.float32
    ).astype(jnp.int32)
  return jnp.sum(
      jnp.where(
          key[:, None, :] == slot[None, :, None],
          slot[None, None, :],
          0,
      ),
      axis=2,
  ).astype(jnp.int32)


def remote_rows(plan, me):
  """Returns the rows s = me sends to other shards.

  Args:
    plan: The SourcePlan of s = me.
    me: This shard's index.

  Returns:
    The remote total (disp_tab row 3 ep + 1, column 0).
  """
  mine = lax.dynamic_index_in_dim(plan.send_cnt, me, axis=0, keepdims=False)
  return plan.send_cnt.sum() - mine.sum()


def receiver_counts(pairs_g, me, *, ep, t_local, g_local, waves, wave_of):
  """Returns the rows `me` receives per wave.

  [W] i32: rows `me` receives per wave, sum over s of cnt_dir_s[me].

  Recomputed from the exchanged routing: only table row `me`
  is looked up, through the same kind of one-hot, so the cost is one
  [g, K * T] one-hot (the reference plan also built it) plus O(K * T), against
  its
  [ep, K * T] lookup and [K, ep, T] compare. The headers carry the same
  counts (HDR_CNT_DIR); switching the kernel's row 3 ep to them is 0c work.

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order.
    me: This shard's index.
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    wave_of: [ep, g] i32 wave of every shard's local experts.

  Returns:
    [W] i32: the sum over sources s of cnt_dir_s[me].
  """
  is_compact = pairs_g.shape[0] == ep and t_local > 1
  if is_compact:
    topk = pairs_g.shape[1] // t_local
    if topk * t_local != pairs_g.shape[1]:
      raise ValueError(
          f"Compact pairs_g {pairs_g.shape} not divisible by t_local={t_local}."
      )
    n_tokens = ep * t_local
  else:
    n_tokens, topk = pairs_g.shape
  me = jnp.asarray(me, jnp.int32)
  g_is_pow2 = g_local > 0 and (g_local & (g_local - 1)) == 0
  first = me * g_local
  if is_compact and (waves == 1 or g_local <= 31):
    # Operate directly on compact `[ep, t_local * topk]` from the routing
    # all-gather so XLA never materializes a padded `[ep * t_local, topk]`
    # HBM copy.
    p = pairs_g.astype(jnp.int32)
    if g_is_pow2:
      g_shift = g_local.bit_length() - 1
      on_me = jnp.right_shift(p, g_shift) == me
    else:
      on_me = (p >= first) & (p < first + g_local)
    if waves == 1:
      tok_hit = lax.reduce_window(
          on_me.astype(jnp.int32),
          jnp.int32(0),
          lax.bitwise_or,
          (1, topk),
          (1, topk),
          "VALID",
      )
      return jnp.sum(tok_hit).reshape(1).astype(jnp.int32)
    table = wave_of.astype(jnp.int32)
    w_iota = jnp.arange(waves, dtype=jnp.int32)
    row = lax.dynamic_index_in_dim(table, me, axis=0, keepdims=False)  # [g]
    g_bit = jnp.left_shift(jnp.int32(1), jnp.arange(g_local, dtype=jnp.int32))
    cum_mask = lax.reduce(
        jnp.where(
            (row[None, :] >= 0) & (row[None, :] <= w_iota[:, None]),
            g_bit[None, :],
            jnp.int32(0),
        ),
        jnp.int32(0),
        lax.bitwise_or,
        (1,),
    )  # [W]
    if g_is_pow2:
      safe_local = jnp.bitwise_and(p, jnp.int32(g_local - 1))
    else:
      safe_local = jnp.where(on_me, p - first, jnp.int32(0))
    pair_bit = jnp.where(
        on_me, jnp.left_shift(jnp.int32(1), safe_local), jnp.int32(0)
    )
    tok_mask = lax.reduce_window(
        pair_bit,
        jnp.int32(0),
        lax.bitwise_or,
        (1, topk),
        (1, topk),
        "VALID",
    )  # [ep, t_local]
    cum_cnt = jnp.sum(
        ((tok_mask[None, :, :] & cum_mask[:, None, None]) != 0).astype(
            jnp.int32
        ),
        axis=(1, 2),
    )
    return jnp.diff(cum_cnt, prepend=jnp.int32(0)).astype(jnp.int32)
  pk = (
      pairs_g.reshape(n_tokens, topk).T.astype(jnp.int32)
      if is_compact
      else pairs_g.T.astype(jnp.int32)
  )  # [K, T]
  if g_is_pow2:
    g_shift = g_local.bit_length() - 1
    on_me = jnp.right_shift(pk, g_shift) == me
  else:
    on_me = (pk >= first) & (pk < first + g_local)
  if waves == 1:
    return jnp.sum(jnp.any(on_me, axis=0).astype(jnp.int32), keepdims=True)
  table = wave_of.astype(jnp.int32)
  w_iota = jnp.arange(waves, dtype=jnp.int32)
  if g_local <= 31:
    # Bitmask reduction over the <=31 local experts avoids an [g, K*T] bf16
    # one-hot and MXU dot: cum_cnt[w] counts tokens with first-hit wave <= w.
    row = lax.dynamic_index_in_dim(table, me, axis=0, keepdims=False)  # [g]
    g_bit = jnp.left_shift(jnp.int32(1), jnp.arange(g_local, dtype=jnp.int32))
    cum_mask = lax.reduce(
        jnp.where(
            (row[None, :] >= 0) & (row[None, :] <= w_iota[:, None]),
            g_bit[None, :],
            jnp.int32(0),
        ),
        jnp.int32(0),
        lax.bitwise_or,
        (1,),
    )  # [W]
    if g_is_pow2:
      safe_local = jnp.bitwise_and(pk, jnp.int32(g_local - 1))
    else:
      safe_local = jnp.where(on_me, pk - first, jnp.int32(0))
    pair_bit = jnp.where(
        on_me, jnp.left_shift(jnp.int32(1), safe_local), jnp.int32(0)
    )
    tok_mask = lax.reduce(pair_bit, jnp.int32(0), lax.bitwise_or, (0,))  # [T]
    cum_cnt = jnp.sum(
        ((tok_mask[None, :] & cum_mask[:, None]) != 0).astype(jnp.int32),
        axis=1,
    )
    return jnp.diff(cum_cnt, prepend=jnp.int32(0)).astype(jnp.int32)
  row = lax.dynamic_index_in_dim(table, me, axis=0, keepdims=True)  # [1, g]
  g_iota = jnp.arange(g_local, dtype=jnp.int32)
  # Pairs computing elsewhere fall outside [0, g): an all-zero column.
  local = (pk - first).reshape(1, topk * n_tokens)
  onehot = (local == g_iota[:, None]).astype(jnp.bfloat16)  # [g, K * T]
  wave = (
      jnp.dot(
          row.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
      )
      .astype(jnp.int32)
      .reshape(topk, n_tokens)
  )
  fw_me = jnp.min(jnp.where(on_me, wave, waves), axis=0)  # [T]
  return jnp.sum((fw_me[None, :] == w_iota[:, None]).astype(jnp.int32), axis=1)


def dispatch_table(plan, me, recv_cnt, *, t_local, wave_bounds_me=None):
  """Builds the [4 * ep, W] dispatch table of shard `me`.

  The [4 * ep, W] disp_tab of dispatch_kernels, from plan (s = me).

  Rows as the reference plan's: send offset, send rows, destination offset, then
  rows
  received per wave (row 3 ep) and the remote total (row 3 ep + 1, col 0).
  Only the destination offset changes: (d, w) lands in slot me on d (D2).
  Built by concatenation, not .at[].set, so there is no XLA scatter.

  Args:
    plan: The SourcePlan of s = me.
    me: This shard's index.
    recv_cnt: [W] rows `me` receives per wave.
    t_local: Tokens per shard.
    wave_bounds_me: Optional [W] wave bounds of shard `me` placed at row
      `3 * ep + 2`.

  Returns:
    [4 * ep, W] i32 disp_tab.

  Raises:
    ValueError: If ep < 2 (or ep < 3 when `wave_bounds_me` is passed).
  """
  ep, waves = plan.cnt_dir.shape
  if ep < 2:
    raise ValueError("wave dispatch needs ep >= 2")
  dst_off = jnp.asarray(me, jnp.int32) * t_local + plan.slot_off
  col = jnp.arange(waves, dtype=jnp.int32)
  extra_rows = [
      recv_cnt.astype(jnp.int32)[None, :],
      jnp.where(col == 0, remote_rows(plan, me), 0)[None, :].astype(jnp.int32),
  ]
  if wave_bounds_me is not None:
    if ep < 3:
      raise ValueError("wave_bounds_me in dispatch_table needs ep >= 3")
    extra_rows.append(wave_bounds_me.astype(jnp.int32)[None, :])
    if ep > 3:
      extra_rows.append(jnp.zeros((ep - 3, waves), jnp.int32))
  else:
    extra_rows.append(jnp.zeros((ep - 2, waves), jnp.int32))
  extra = jnp.concatenate(extra_rows, axis=0)
  return jnp.concatenate([plan.send_off, plan.send_cnt, dst_off, extra], axis=0)


def list_lengths(hist_s, *, ep, g_local):
  """Returns the entries of each (s -> d) list.

  Args:
    hist_s: [ep * g] i32 per-expert pair counts of source s.
    ep: Expert-parallel width.
    g_local: Experts per shard.

  Returns:
    [ep] i32 n(s -> d).
  """
  return hist_s.reshape(ep, g_local).sum(axis=1).astype(jnp.int32)


def list_headers(plan, me, n, *, nw):
  """Returns the header block of each (me -> d) list.

  Args:
    plan: The SourcePlan of s = me.
    me: This shard's index.
    n: [ep] entries per list.
    nw: Planes per entry.

  Returns:
    [ep, 128] i32 header blocks.
  """
  ep, waves = plan.cnt_dir.shape

  def col(v):
    return jnp.broadcast_to(jnp.asarray(v, jnp.int32), (ep,))[:, None]

  pad = jnp.zeros((ep, MAX_WAVES - waves), jnp.int32)
  return jnp.concatenate(
      [
          n.astype(jnp.int32)[:, None],
          col(me),
          col(waves),
          col(nw),
          jnp.zeros((ep, HDR_SEND_OFF - 4), jnp.int32),
          plan.send_off,
          pad,
          plan.cnt_dir,
          pad,
          # De-Dup fields (n_fwd, stage send offsets, stage counts) and spare.
          jnp.zeros((ep, HDR_WORDS - HDR_N_FWD), jnp.int32),
      ],
      axis=1,
  )


def unique_scatter_rows(row, valid, slab_len):
  """Returns unique scatter rows so `.at[].set(unique_indices=True)` is valid.

  Valid entries keep `row` (which must already be distinct in `[0, slab_len)`);
  invalid entries are assigned distinct indices `[slab_len, slab_len + n_pad)`.

  Args:
    row: [N] i32 destination indices.
    valid: [N] bool validity mask.
    slab_len: Valid destination range `[0, slab_len)`.

  Returns:
    [N] i32 strictly unique indices in `[0, slab_len + N)`.
  """
  pad_rank = jnp.cumsum((~valid).astype(jnp.int32)) - 1
  return jnp.where(valid, row.astype(jnp.int32), slab_len + pad_rank)


# Build per-destination list bodies with one batched window read / Pallas copy
# instead of `ep` dynamic slices (TODO(b/581886436)).
_GATHER_SEND_LISTS = os.environ.get("DEDUP_MOE_GATHER_SEND_LISTS", "1") != "0"
_SEND_LIST_GATHER = os.environ.get("DEDUP_MOE_SEND_LIST_GATHER", "vec")
_PALLAS_INTERPRET: Any = None
# Up to this t_local, send_perm and the send-list ordering are sort free
# (compare-and-select over the few decode pairs instead of an XLA sort).
# Same bits as the sorts on every row that is read. 0 (default) keeps the
# sorts; DEDUP_MOE_SORT_FREE_MAX_T=96 covers decode.
_SORT_FREE_MAX_T = int(os.environ.get("DEDUP_MOE_SORT_FREE_MAX_T", "0"))
# Decode SMEM fast plan (direct dispatch only, TODO(b/581886436)):
#   0: off (default).
#   1: standalone SMEM Pallas kernel (`decode_fast_plan`).
#   2: fused at the head of `tc_sender` row-send start kernel (falls back to 1
#      when row_send is off).
_DECODE_FAST_PLAN = int(os.environ.get("DEDUP_MOE_DECODE_FAST_PLAN", "0"))
_DECODE_FAST_PLAN_MAX_T = int(
    os.environ.get("DEDUP_MOE_DECODE_FAST_PLAN_MAX_T", "96")
)
_DECODE_FAST_PLAN_MAX_SMEM_BYTES = 384 * 1024
_FAST_GLUE_V2 = os.environ.get("DEDUP_MOE_FAST_GLUE_V2", "0") == "1"
_FAST_GLUE = _FAST_GLUE_V2 or (
    os.environ.get("DEDUP_MOE_FAST_GLUE", "0") == "1"
)


def _sort_pairs(operands, *, t_local, keep_keys=True, use_mxu=None):
  """lax.sort(operands, num_keys=1) for unique int32 keys, maybe sort free.

  Up to `_SORT_FREE_MAX_T`, each key's rank is the count of smaller keys
  (an [M, M] compare) and each output slot selects the operand whose rank
  it is (an [M, M] select-sum): exact, since the keys are unique.

  Args:
    operands: (keys, *values), [M] int32 each; keys unique.
    t_local: Tokens per shard (picks the method).
    keep_keys: Whether `ordered[0]` must hold the sorted keys.
    use_mxu: Optional bool to select byte-sliced bf16 MXU select-sum.

  Returns:
    The operands in ascending key order (`ordered[0]` is `keys` unchanged when
    `keep_keys` is False on the sort-free path).
  """
  if t_local > _SORT_FREE_MAX_T:
    return lax.sort(operands, num_keys=1, is_stable=False)
  keys = operands[0]
  m = keys.shape[0]
  rank = jnp.sum((keys[None, :] < keys[:, None]).astype(jnp.int32), axis=1)
  hit = rank[None, :] == jnp.arange(m, dtype=jnp.int32)[:, None]  # [slot, p]
  vals = operands if keep_keys else operands[1:]
  if use_mxu is None:
    use_mxu = _FAST_GLUE
  if use_mxu and vals and all(op.dtype == jnp.int32 for op in vals):
    shifts = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
    packed_u = jnp.stack([op.astype(jnp.uint32) for op in vals], axis=1)
    mat = (
        jnp.bitwise_and(
            jnp.right_shift(packed_u[:, :, None], shifts[None, None, :]),
            jnp.uint32(0xFF),
        )
        .reshape(m, 4 * len(vals))
        .astype(jnp.bfloat16)
    )
    res = jnp.dot(
        hit.astype(jnp.bfloat16), mat, preferred_element_type=jnp.float32
    ).astype(jnp.uint32)
    sorted_vals = tuple(
        (
            res[:, 4 * idx]
            | jnp.left_shift(res[:, 4 * idx + 1], 8)
            | jnp.left_shift(res[:, 4 * idx + 2], 16)
            | jnp.left_shift(res[:, 4 * idx + 3], 24)
        ).astype(jnp.int32)
        for idx in range(len(vals))
    )
    return sorted_vals if keep_keys else (keys, *sorted_vals)
  sorted_vals = tuple(
      jnp.sum(jnp.where(hit, op[None, :], 0), axis=1).astype(op.dtype)
      for op in vals
  )
  return sorted_vals if keep_keys else (keys, *sorted_vals)


def _flat_list_bodies(planes, start, n, *, lmax, nw):
  """Builds `[ep, nw * lmax]` list bodies with one `jnp.take` and no loop."""
  width = nw * lmax
  n_ = n[:, None]
  j = jnp.arange(width, dtype=jnp.int32)[None, :]
  p0 = jnp.pad(planes[0], (0, width)).astype(jnp.int32)
  len0 = p0.shape[0]
  o0 = jnp.clip(start, 0, len0 - width)
  idx = o0[:, None] + j
  live = j < n_
  flat = p0
  if nw == 2:
    stride = (n + (LANES - 1)) // LANES * LANES
    p1 = jnp.pad(planes[1], (lmax, width)).astype(jnp.int32)
    o1 = jnp.clip(start - stride + lmax, 0, p1.shape[0] - width)
    rel = j - stride[:, None]
    sel1 = (rel >= 0) & (rel < n_)
    idx = jnp.where(sel1, len0 + o1[:, None] + j, idx)
    live = live | sel1
    flat = jnp.concatenate([p0, p1])
  got = jnp.take(flat, idx, axis=0, mode="clip")
  return jnp.where(live, got, 0).astype(jnp.int32)


def _shift_windows(plane, offsets, width):
  """Returns `[R, width]` where row `r` is `plane[offsets[r] : offsets[r] + width]`."""
  rows, length = offsets.shape[0], plane.shape[0]
  x = jnp.broadcast_to(plane[None, :], (rows, length))
  k = 1
  while k < length:
    shifted = jnp.concatenate(
        [x[:, k:], jnp.zeros((rows, k), x.dtype)], axis=1
    )
    x = jnp.where(((offsets & k) != 0)[:, None], shifted, x)
    k *= 2
  return x[:, :width]


def _shift_list_bodies(planes, start, n, *, lmax, nw):
  """Builds `[ep, nw * lmax]` list bodies via barrel-shifted windows."""
  width = nw * lmax
  n_ = n[:, None]
  j = jnp.arange(width, dtype=jnp.int32)[None, :]
  p0 = jnp.pad(planes[0], (0, width)).astype(jnp.int32)
  w0 = _shift_windows(p0, jnp.clip(start, 0, p0.shape[0] - width), width)
  body = jnp.where(j < n_, w0, 0)
  if nw == 2:
    stride = (n + (LANES - 1)) // LANES * LANES
    p1 = jnp.pad(planes[1], (lmax, width)).astype(jnp.int32)
    o1 = jnp.clip(start - stride + lmax, 0, p1.shape[0] - width)
    w1 = _shift_windows(p1, o1, width)
    rel = j - stride[:, None]
    body = jnp.where((rel >= 0) & (rel < n_), w1, body)
  return body.astype(jnp.int32)


_SUB = 8
_VFRONT = 8
_VBACK = 16


def _vec_sublanes_from(win, k):
  out = win[0:_SUB]
  for jj in range(1, _SUB + 1):
    out = jnp.where(k == jj, win[jj : jj + _SUB], out)
  return out


def _vec_copy_run(src_vm, img_vm, src, n, dst):
  """Copies `img[dst + i] = src[src + i]` for `i < n` over `[rows, 128]` refs."""
  delta = src - dst
  q = jnp.right_shift(delta, 7)
  rem = jnp.bitwise_and(delta, LANES - 1)
  t_first = jnp.right_shift(dst, 10)
  t_last = jnp.right_shift(dst + n - 1, 10)
  sub = lax.broadcasted_iota(jnp.int32, (_SUB, LANES), 0)
  lane = lax.broadcasted_iota(jnp.int32, (_SUB, LANES), 1)

  def tile(ti, carry):
    r0 = pl.multiple_of(ti * _SUB, _SUB)
    q0 = r0 + q
    qa = pl.multiple_of(jnp.bitwise_and(q0, -_SUB), _SUB)
    sh = q0 - qa
    win = src_vm[pl.ds(qa, 2 * _SUB), :]
    a = _vec_sublanes_from(win, sh)
    b = _vec_sublanes_from(win, sh + 1)
    shift = LANES - rem
    vals = jnp.where(
        lane < LANES - rem, pltpu.roll(a, shift, 1), pltpu.roll(b, shift, 1)
    )
    w = (r0 + sub) * LANES + lane
    live = (w >= dst) & (w < dst + n)
    cur = img_vm[pl.ds(r0, _SUB), :]
    img_vm[pl.ds(r0, _SUB), :] = jnp.where(live, vals, cur)
    return carry

  lax.fori_loop(t_first, t_last + 1, tile, 0)


def _vec_list_bodies(
    planes, start, n, *, lmax, nw, header_words: int = 0
):
  """Builds `[ep, nw * lmax]` (or `[ep * list_rows, 128]`) lists in VMEM."""
  width = nw * lmax
  if width % LANES or header_words % LANES:
    raise ValueError(
        f"list width {width} and header {header_words} must be multiples of"
        f" {LANES}"
    )
  ep = start.shape[0]
  row_stride = header_words + width
  out_rows = -(-(ep * row_stride // LANES) // _SUB) * _SUB
  d = jnp.arange(ep, dtype=jnp.int32)
  p0 = planes[0].astype(jnp.int32)
  len0 = p0.shape[0]
  srcs, dsts = [start], [d * row_stride + header_words]
  flat = p0
  if nw == 2:
    stride = (n + (LANES - 1)) // LANES * LANES
    flat = jnp.concatenate([p0, planes[1].astype(jnp.int32)])
    srcs.append(len0 + start)
    dsts.append(d * row_stride + header_words + stride)
  runs = jnp.stack(
      [
          jnp.concatenate(srcs) + _VFRONT * LANES,
          jnp.concatenate([n] * nw),
          jnp.concatenate(dsts),
      ]
  ).astype(jnp.int32)
  n_runs = runs.shape[1]
  rows = -(-flat.shape[0] // LANES)
  rows = -(-rows // _SUB) * _SUB
  src2d = jnp.pad(
      flat,
      (_VFRONT * LANES, (rows + _VBACK) * LANES - flat.shape[0]),
  ).reshape(_VFRONT + rows + _VBACK, LANES)
  if not _FAST_GLUE_V2:
    need = _VFRONT + out_rows + rows + 2 * _SUB
    if src2d.shape[0] < need:
      src2d = jnp.pad(src2d, ((0, need - src2d.shape[0]), (0, 0)))

  def kernel(runs_sm, src_vm, out_vm):
    zero = jnp.zeros((_SUB, LANES), jnp.int32)

    def clear(i, c):
      out_vm[pl.ds(pl.multiple_of(i * _SUB, _SUB), _SUB), :] = zero
      return c

    lax.fori_loop(0, out_rows // _SUB, clear, 0)

    def run(r, c):
      cnt = runs_sm[1, r]

      @pl.when(cnt > 0)
      def _():
        _vec_copy_run(src_vm, out_vm, runs_sm[0, r], cnt, runs_sm[2, r])

      return c

    lax.fori_loop(0, n_runs, run, 0)

  out = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((out_rows, LANES), jnp.int32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(),
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
      ),
      interpret=_PALLAS_INTERPRET or False,
      name="dedup_send_list_bodies",
  )(runs, src2d)
  if header_words:
    return out.reshape(-1)[: ep * row_stride].reshape(
        ep * (row_stride // LANES), LANES
    )
  return out.reshape(-1)[: ep * width].reshape(ep, width)


def _dma_windows(flat, offsets, width):
  """Copies `[R, width]` windows from `flat` at `offsets` via Pallas async DMAs."""
  rows = offsets.shape[0]

  def kernel(off_ref, src_ref, out_ref, sem):
    def copy(r):
      return pltpu.make_async_copy(
          src_ref.at[pl.ds(off_ref[r], width)], out_ref.at[r], sem
      )

    def start_copy(r, c):
      copy(r).start()
      return c

    def wait_copy(r, c):
      copy(r).wait()
      return c

    lax.fori_loop(0, rows, start_copy, 0)
    lax.fori_loop(0, rows, wait_copy, 0)

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((rows, width), jnp.int32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(),
          in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          scratch_shapes=[pltpu.SemaphoreType.DMA(())],
      ),
      interpret=_PALLAS_INTERPRET or False,
      name="dedup_send_list_windows",
  )(offsets.astype(jnp.int32), flat.astype(jnp.int32))


def _pallas_list_bodies(planes, start, n, *, lmax, nw):
  """Builds `[ep, nw * lmax]` list bodies with Pallas DMA window copies."""
  width = nw * lmax
  n_ = n[:, None]
  j = jnp.arange(width, dtype=jnp.int32)[None, :]
  ep = start.shape[0]
  p0 = jnp.pad(planes[0], (0, width)).astype(jnp.int32)
  len0 = p0.shape[0]
  offs = [jnp.clip(start, 0, len0 - width)]
  flat = p0
  stride = (n + (LANES - 1)) // LANES * LANES
  if nw == 2:
    p1 = jnp.pad(planes[1], (lmax, width)).astype(jnp.int32)
    offs.append(len0 + jnp.clip(start - stride + lmax, 0, p1.shape[0] - width))
    flat = jnp.concatenate([p0, p1])
  w = _dma_windows(flat, jnp.concatenate(offs), width)
  body = jnp.where(j < n_, w[:ep], 0)
  if nw == 2:
    rel = j - stride[:, None]
    body = jnp.where((rel >= 0) & (rel < n_), w[ep:], body)
  return body.astype(jnp.int32)


def _gather_list_bodies(planes, start, n, *, lmax, nw):
  """Builds `[ep, nw * lmax]` list bodies with `jax.vmap(lax.dynamic_slice)`."""
  width = nw * lmax
  n_ = n[:, None]
  j = jnp.arange(width, dtype=jnp.int32)[None, :]

  def windows(plane, offsets):
    return jax.vmap(
        lambda o: lax.dynamic_slice(
            plane, (o,), (width,), allow_negative_indices=False
        )
    )(offsets)

  p0 = jnp.pad(planes[0], (0, width))
  body = jnp.where(j < n_, windows(p0, start), 0)
  if nw == 2:
    stride = (n + (LANES - 1)) // LANES * LANES
    p1 = jnp.pad(planes[1], (lmax, width))
    w1 = windows(p1, start - stride + lmax)
    rel = j - stride[:, None]
    body = jnp.where((rel >= 0) & (rel < n_), w1, body)
  return body.astype(jnp.int32)


def _compact_list_bodies(planes, start, n, *, lmax, nw):
  """Builds `[ep, nw * lmax]` list bodies with stacked slices and block masks.

  One dynamic slice per destination over the stacked planes, then plane 1 is
  placed `align_up(n, 128)` words in by a select over 128-word blocks.
  """
  ep = start.shape[0]
  lane = jnp.arange(lmax, dtype=jnp.int32)[None, :]  # [1, lmax]
  live = lane < n[:, None]  # [ep, lmax]
  if nw == 1:
    p0 = jnp.pad(planes[0], (0, lmax))
    segs0 = jnp.stack([
        lax.dynamic_slice(
            p0, (start[d],), (lmax,), allow_negative_indices=False
        )
        for d in range(ep)
    ])
    return jnp.where(live, segs0, 0)
  p01 = jnp.stack(
      [jnp.pad(planes[0], (0, lmax)), jnp.pad(planes[1], (0, lmax))], axis=0
  )  # [2, K*t + lmax]
  segs01 = jnp.stack(
      [
          lax.dynamic_slice(
              p01, (0, start[d]), (2, lmax), allow_negative_indices=False
          )
          for d in range(ep)
      ],
      axis=1,
  )  # [2, ep, lmax]
  segs01 = jnp.where(live[None, :, :], segs01, 0)
  segs0, segs1 = segs01[0], segs01[1]
  n_blocks = lmax // LANES
  s0_blk = segs0.reshape(ep, n_blocks, LANES)
  s1_blk = segs1.reshape(ep, n_blocks, LANES)
  s0_pad = jnp.pad(s0_blk, ((0, 0), (0, n_blocks), (0, 0)))
  stride_blk = (n + (LANES - 1)) // LANES  # [ep] in [0, n_blocks]
  blk_idx = jnp.arange(2 * n_blocks, dtype=jnp.int32)[None, :]
  rel = blk_idx - stride_blk[:, None]  # [ep, 2*n_blocks]
  k_iota = jnp.arange(n_blocks, dtype=jnp.int32)[None, None, :]
  shifted_s1 = jnp.sum(
      jnp.where(
          (rel[:, :, None] == k_iota)[:, :, :, None],
          s1_blk[:, None, :, :],
          0,
      ),
      axis=2,
  )
  return jnp.where(
      (blk_idx < stride_blk[:, None])[:, :, None], s0_pad, shifted_s1
  ).reshape(ep, 2 * lmax)


def build_send_lists(
    pairs_g,
    plan,
    me,
    hist_me,
    *,
    ep,
    t_local,
    g_local,
    scale_bits=None,
    send_list_gather=None,
    pack_mode=None,
    local=False,
    include_headers: bool = True,
):
  """Builds the token-gather lists shard `me` sends.

  The token-gather lists shard `me` sends, [ep * list_rows, 128] i32.

  Row block d (list_rows rows) is the (me -> d) list: the header, then the
  entries of me's pairs on d's experts in d's slab order -- local expert
  ascending, then pair index (t * K + k), exactly how the run (expert, me)
  is laid out in d's slab -- one plane per word: word 0 is the receive row
  of the pair's token on d (plan.row_on_dest); word 1, with scale_bits [t]
  (fp8 rows), the token's activation-scale bits. Planes are
  align_up(n, 128) apart, so ONE DMA of list_dma_rows(n, nw) rows carries
  the whole list. Words past n in a plane are zero and never read.

  One sort of the t * K pairs keyed by (expert, pair) plus list-body
  assembly (`send_list_gather`: `vec`, `flat`, `gather`/`vmap`, `shift`,
  `pallas`, `scatter_set`, `compact`, or `loop`/`slice`); the receive row rides the sort
  as an operand, selected per pair by a compare over the ep destinations (no
  gather). hist_me [ep * g] i32 is me's per-expert pair count (rows_by_dest[me]).

  The per-destination slices pass allow_negative_indices=False: their
  starts are never negative, and JAX's negative-index wrap stages three
  scalar HLO ops (add, compare, select) per destination that XLA runs one
  by one outside any fusion, a cost that grows with EP. Same bits.

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order.
    plan: The SourcePlan of s = me.
    me: This shard's index.
    hist_me: [ep * g] i32 me's per-expert pair counts (rows_by_dest[me]).
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    scale_bits: Optional [t] i32 activation-scale bits (fp8 rows), sent as plane
      1.
    send_list_gather: Optional override in `SEND_LIST_GATHER_MODES`. When None,
      uses `_SEND_LIST_GATHER` if `_GATHER_SEND_LISTS` is True else `loop`.
    pack_mode: Alias for `send_list_gather`.
    local: `pairs_g` holds only shard `me`'s [t_local, K] rows.

  Returns:
    [ep * list_rows, 128] i32 lists, row block d being the (me -> d) list.

  Raises:
    ValueError: If the list sort key would overflow int32 or the gather mode is
      invalid.
  """
  mode = send_list_gather if send_list_gather is not None else pack_mode
  if mode is not None and mode not in SEND_LIST_GATHER_MODES:
    raise ValueError(
        f"send_list_gather={mode!r} is not one of {SEND_LIST_GATHER_MODES}"
    )
  if mode is None:
    mode = _SEND_LIST_GATHER if _GATHER_SEND_LISTS else "loop"
    if mode == "vec" and jax.default_backend() == "cpu" and _PALLAS_INTERPRET is None:
      mode = "loop"
  _, topk = pairs_g.shape
  t = t_local
  me = jnp.asarray(me, jnp.int32)
  nw = 1 if scale_bits is None else 2
  lmax = list_capacity(t, topk)
  if local:
    pk = pairs_g.astype(jnp.int32).T
  else:
    pk = lax.dynamic_slice(
        pairs_g.astype(jnp.int32), (me * t, 0), (t, topk)
    ).T
  d_iota = jnp.arange(ep, dtype=jnp.int32)
  val = jnp.sum(
      jnp.where(
          plan.dest[:, None, :] == d_iota[None, :, None],
          plan.row_on_dest[None, :, :],
          0,
      ),
      axis=1,
  )  # [K, t]
  pair = (
      jnp.arange(t, dtype=jnp.int32)[None, :] * topk
      + jnp.arange(topk, dtype=jnp.int32)[:, None]
  )
  # Unique per pair, below (E + 1) * t * K. An id outside [0, E) is in no
  # destination's count (n below), so it sorts past every list instead of
  # shifting the segments.
  e_total = ep * g_local
  if (e_total + 1) * t * topk >= 2**31:
    raise ValueError(
        f"list sort key (E + 1) * t * K = {(e_total + 1) * t * topk} "
        "does not fit int32"
    )
  valid = (pk >= 0) & (pk < e_total)
  key = jnp.where(valid, pk, e_total) * (t * topk) + pair
  operands = [key.reshape(-1), val.reshape(-1)]
  if scale_bits is not None:
    sb = scale_bits.astype(jnp.int32)
    if sb.ndim == 1:
      sb = jnp.broadcast_to(sb[None, :], (topk, t))
    elif sb.shape == (t, topk):
      sb = sb.T
    else:
      raise ValueError(
          f"scale_bits must be [t_local] or [t_local, topk], got {sb.shape}"
      )
    operands.append(sb.reshape(-1))
  ordered = _sort_pairs(
      tuple(operands), t_local=t, keep_keys=(mode == "scatter_set")
  )
  n = list_lengths(hist_me, ep=ep, g_local=g_local)
  if _FAST_GLUE_V2 and ep <= 64:
    start = jnp.sum(
        jnp.where(d_iota[None, :] < d_iota[:, None], n[None, :], 0), axis=1
    )
  else:
    start = jnp.cumsum(n) - n
  if mode == "scatter_set":
    m_pairs = t * topk
    pk_sorted = ordered[0] // m_pairs
    live_entry = pk_sorted < e_total
    d_sorted = jnp.where(live_entry, pk_sorted // g_local, 0)
    d_mask = d_sorted[:, None] == d_iota[None, :]
    start_of_entry = jnp.sum(jnp.where(d_mask, start[None, :], 0), axis=1)
    pos_in_d = jnp.arange(m_pairs, dtype=jnp.int32) - start_of_entry
    body_words = nw * lmax
    total_body_words = ep * body_words
    idx0 = d_sorted * body_words + pos_in_d
    val0 = jnp.where(live_entry, ordered[1], 0)
    if nw == 1:
      u_idx = unique_scatter_rows(idx0, live_entry, total_body_words)
      flat_bodies = (
          jnp.zeros((total_body_words + m_pairs,), jnp.int32)
          .at[u_idx]
          .set(val0, unique_indices=True)[:total_body_words]
      )
    else:
      stride = (n + (LANES - 1)) // LANES * LANES
      stride_of_entry = jnp.sum(jnp.where(d_mask, stride[None, :], 0), axis=1)
      idx1 = d_sorted * body_words + stride_of_entry + pos_in_d
      val1 = jnp.where(live_entry, ordered[2], 0)
      all_idx = jnp.concatenate([idx0, idx1])
      all_live = jnp.concatenate([live_entry, live_entry])
      all_val = jnp.concatenate([val0, val1])
      u_idx = unique_scatter_rows(all_idx, all_live, total_body_words)
      flat_bodies = (
          jnp.zeros((total_body_words + 2 * m_pairs,), jnp.int32)
          .at[u_idx]
          .set(all_val, unique_indices=True)[:total_body_words]
      )
    bodies_arr = flat_bodies.reshape(ep, body_words)
  elif mode == "compact":
    bodies_arr = _compact_list_bodies(
        ordered[1:], start, n, lmax=lmax, nw=nw
    ).astype(jnp.int32)
  elif mode not in ("loop", "slice"):
    if mode == "vec" and _FAST_GLUE_V2:
      planes = list(ordered[1:])
    else:
      planes = [jnp.pad(p, (0, lmax)) for p in ordered[1:]]
    if mode == "vec" and not include_headers:
      return _vec_list_bodies(
          planes, start, n, lmax=lmax, nw=nw, header_words=HDR_WORDS
      )
    gather_fn = {
        "flat": _flat_list_bodies,
        "pallas": _pallas_list_bodies,
        "shift": _shift_list_bodies,
        "vec": _vec_list_bodies,
        "gather": _gather_list_bodies,
        "vmap": _gather_list_bodies,
    }.get(mode, _gather_list_bodies)
    bodies_arr = gather_fn(planes, start, n, lmax=lmax, nw=nw)
  else:
    # Padded so a slice never runs off the end (dynamic_slice would clamp
    # its start, shifting the entries).
    planes = [jnp.pad(p, (0, lmax)) for p in ordered[1:]]
    lane = jnp.arange(lmax, dtype=jnp.int32)
    bodies = []
    for d in range(ep):
      live = lane < n[d]
      segs = [
          jnp.where(
              live,
              lax.dynamic_slice(
                  p, (start[d],), (lmax,), allow_negative_indices=False
              ),
              0,
          )
          for p in planes
      ]
      body = segs[0]
      if nw == 2:
        stride = (n[d] + (LANES - 1)) // LANES * LANES
        body = lax.dynamic_update_slice(
            jnp.concatenate([segs[0], jnp.zeros((lmax,), jnp.int32)]),
            segs[1],
            (stride,),
            allow_negative_indices=False,
        )
      bodies.append(body)
    bodies_arr = jnp.stack(bodies)
  blocks = jnp.concatenate(
      [list_headers(plan, me, n, nw=nw), bodies_arr], axis=1
  )  # [ep, list_row_words]
  return blocks.reshape(ep * list_rows(t, topk, nw), LANES)


def _packed_list_rows(ep, t_local, topk, nw, write_headers=False):
  """Tight upper bound on live 128-word rows across all `ep` send lists."""
  m_pairs = t_local * topk
  k_max = min(ep, m_pairs)
  plane_rows = k_max + max(0, m_pairs - k_max) // LANES
  body_rows = max(1, nw * plane_rows)
  return (ep + body_rows) if write_headers else body_rows


def decode_fast_plan_scratch_shapes(
    *,
    ep,
    g_local,
    t_local,
    topk,
    waves,
    nw,
    per_dest_perm=True,
    write_headers=False,
    compute_visit_me=False,
):
  """Returns the ordered SMEM scratch shapes for `emit_decode_plan_and_lists_body`."""
  e_total = ep * g_local
  packed_rows = _packed_list_rows(
      ep, t_local, topk, nw, write_headers=write_headers
  )
  shapes = [
      pltpu.SMEM((packed_rows, LANES), jnp.int32),  # lists_sm
      pltpu.SMEM((ep, t_local), jnp.int32),  # row_on_dest_sm
      pltpu.SMEM((e_total,), jnp.int32),  # cursor_sm
      pltpu.SMEM((ep,), jnp.int32),  # seen_stamp_sm
      pltpu.SMEM((ep,), jnp.int32),  # body_row_off_sm
      pltpu.SMEM((ep,), jnp.int32),  # stride_words_sm
  ]
  if waves > 1:
    shapes.extend([
        pltpu.SMEM((ep,), jnp.int32),  # seen_idx_sm
        pltpu.SMEM((t_local, topk), jnp.int32),  # tok_d_sm
        pltpu.SMEM((t_local, topk), jnp.int32),  # tok_w_sm
        pltpu.SMEM((t_local,), jnp.int32),  # tok_nd_sm
        pltpu.SMEM((ep, waves), jnp.int32),  # cur_slot_sm
    ])
    if not per_dest_perm:
      shapes.append(pltpu.SMEM((ep, waves), jnp.int32))  # cur_send_sm
  else:
    if not per_dest_perm:
      shapes.append(pltpu.SMEM((ep, t_local), jnp.int32))  # perm_2d_sm
    if compute_visit_me:
      shapes.append(pltpu.SMEM((g_local,), jnp.int32))  # key_sm
  return tuple(shapes)


def decode_fast_plan_smem_bytes(
    *,
    ep,
    g_local,
    t_local,
    topk,
    waves,
    nw,
    per_dest_perm=True,
    send_alloc=0,
    write_headers=False,
    compute_visit_me=False,
):
  """Returns total int32 SMEM bytes (inputs + outputs + scratch) for fast plan."""
  e_total = ep * g_local
  scratch = decode_fast_plan_scratch_shapes(
      ep=ep,
      g_local=g_local,
      t_local=t_local,
      topk=topk,
      waves=waves,
      nw=nw,
      per_dest_perm=per_dest_perm,
      write_headers=write_headers,
      compute_visit_me=compute_visit_me,
  )
  elems = sum(math.prod(s.shape) for s in scratch)
  # Inputs: me, pairs_me, hist_me, recv_cnt, optional scale/wave_of/bounds/rbd.
  elems += 1 + t_local * topk + e_total + waves
  if nw == 2:
    elems += t_local * topk
  if waves > 1:
    elems += ep * g_local + waves
  elif compute_visit_me:
    elems += ep * g_local + g_local
  # Outputs: disp_tab, list_tab, perm.
  elems += 4 * ep * waves + (ep + 1)
  elems += (ep * t_local) if per_dest_perm else send_alloc
  return elems * 4


def decode_fast_plan_supported(
    *,
    ep,
    g_local,
    t_local,
    topk,
    waves,
    nw,
    per_dest_perm=True,
    send_alloc=0,
    write_headers=False,
    compute_visit_me=False,
    has_wave_bounds=True,
):
  """Returns whether the SMEM decode fast plan fits and is legal for this shape."""
  if t_local <= 0 or t_local > _DECODE_FAST_PLAN_MAX_T:
    return False
  if waves < 1 or waves > MAX_WAVES:
    return False
  if ep < (3 if has_wave_bounds else 2):
    return False
  if compute_visit_me and (waves != 1 or g_local > 64):
    return False
  smem_bytes = decode_fast_plan_smem_bytes(
      ep=ep,
      g_local=g_local,
      t_local=t_local,
      topk=topk,
      waves=waves,
      nw=nw,
      per_dest_perm=per_dest_perm,
      send_alloc=send_alloc,
      write_headers=write_headers,
      compute_visit_me=compute_visit_me,
  )
  return smem_bytes <= _DECODE_FAST_PLAN_MAX_SMEM_BYTES


def emit_decode_plan_and_lists_body(
    *,
    me_ref,
    pairs_me_sm,
    hist_me_sm,
    recv_cnt_sm,
    scale_bits_sm,
    wave_of_sm,
    wave_bounds_me_sm,
    rows_by_dest_me_sm,
    disp_tab_sm,
    perm_sm,
    list_tab_sm,
    visit_me_sm,
    tg_lists_hbm,
    scratch_refs,
    dma_sem,
    ep,
    g_local,
    t_local,
    topk,
    waves,
    nw,
    list_rows_per_dest,
    per_dest_perm=True,
    send_alloc=0,
    write_headers=False,
    has_wave_bounds=True,
):
  """Builds disp_tab, perm, list_tab[, visit_me] and packed send lists in SMEM.

  Shared kernel body for standalone `decode_fast_plan` (Mode 1) and the fused
  `tc_sender` start kernel (Mode 2). Returns `(start_list_dmas, wait_list_dmas)`
  so callers can overlap the local SMEM -> HBM list copy with an ICI barrier.
  """
  e_total = ep * g_local
  me = me_ref[0]
  g_is_pow2 = g_local > 0 and (g_local & (g_local - 1)) == 0
  g_shift = g_local.bit_length() - 1
  g_mask = g_local - 1
  compute_visit_me = visit_me_sm is not None

  def _div_g(e):
    if g_is_pow2:
      return jnp.right_shift(e, jnp.int32(g_shift))
    return lax.div(e, jnp.int32(g_local))

  def _mod_g(e):
    if g_is_pow2:
      return jnp.bitwise_and(e, jnp.int32(g_mask))
    return e - lax.div(e, jnp.int32(g_local)) * jnp.int32(g_local)

  it = iter(scratch_refs)
  lists_sm = next(it)
  row_on_dest_sm = next(it)
  cursor_sm = next(it)
  seen_stamp_sm = next(it)
  body_row_off_sm = next(it)
  stride_words_sm = next(it)
  if waves > 1:
    seen_idx_sm = next(it)
    tok_d_sm = next(it)
    tok_w_sm = next(it)
    tok_nd_sm = next(it)
    cur_slot_sm = next(it)
    cur_send_sm = None if per_dest_perm else next(it)
    perm_2d_sm = None
    key_sm = None
  else:
    seen_idx_sm = tok_d_sm = tok_w_sm = tok_nd_sm = None
    cur_slot_sm = cur_send_sm = None
    perm_2d_sm = None if per_dest_perm else next(it)
    key_sm = next(it) if compute_visit_me else None
  assert next(it, None) is None

  # Clear disp_tab, seen_stamp (to -1), and 1D perm (when !per_dest_perm).
  def _clear_disp(r, _):
    for w in range(waves):
      disp_tab_sm[r, w] = jnp.int32(0)
    return jnp.int32(0)

  lax.fori_loop(0, 4 * ep, _clear_disp, jnp.int32(0))

  def _clear_seen(d, _):
    seen_stamp_sm[d] = jnp.int32(-1)
    return jnp.int32(0)

  lax.fori_loop(0, ep, _clear_seen, jnp.int32(0))

  if not per_dest_perm:
    def _clear_perm(i, _):
      perm_sm[i] = jnp.int32(0)
      return jnp.int32(0)

    lax.fori_loop(0, send_alloc, _clear_perm, jnp.int32(0))

  # Optional W == 1 visit order + wave_bounds_me from rows_by_dest_me.
  nonempty_me = jnp.int32(0)
  if compute_visit_me:
    assert waves == 1 and rows_by_dest_me_sm is not None and key_sm is not None

    def _h_loop(g, nonempty):
      def _s_loop(s, acc):
        return acc + rows_by_dest_me_sm[s, g]

      hg = lax.fori_loop(0, ep, _s_loop, jnp.int32(0))
      is_pos = hg > jnp.int32(0)
      val = jnp.where(
          is_pos,
          jnp.minimum(hg, jnp.int32(2**24 - 1)),
          jnp.int32(2**24),
      )
      key_sm[g] = val * jnp.int32(g_local) + g
      return nonempty + is_pos.astype(jnp.int32)

    nonempty_me = lax.fori_loop(0, g_local, _h_loop, jnp.int32(0))

    def _rank_loop(g, _):
      kg = key_sm[g]

      def _cmp_loop(g2, r):
        return r + (key_sm[g2] < kg).astype(jnp.int32)

      rank = lax.fori_loop(0, g_local, _cmp_loop, jnp.int32(0))
      visit_me_sm[rank] = g
      return jnp.int32(0)

    lax.fori_loop(0, g_local, _rank_loop, jnp.int32(0))

  # Pass 1: per-destination counts, offsets, row_on_dest, and send perm.
  me_base = me * jnp.int32(t_local)
  if waves == 1:
    perm_target = perm_sm if per_dest_perm else perm_2d_sm
    assert perm_target is not None

    def _p1_w1_tok(t, _):
      def _p1_w1_k(k, _):
        e = pairs_me_sm[t, k]
        valid = (e >= jnp.int32(0)) & (e < jnp.int32(e_total))

        @pl.when(valid)
        def _():
          d = _div_g(e)
          is_new = seen_stamp_sm[d] != t

          @pl.when(is_new)
          def _():
            seen_stamp_sm[d] = t
            sl = disp_tab_sm[ep + d, 0]
            disp_tab_sm[ep + d, 0] = sl + jnp.int32(1)
            row_on_dest_sm[d, t] = me_base + sl
            perm_target[d, sl] = t

        return jnp.int32(0)

      lax.fori_loop(0, topk, _p1_w1_k, jnp.int32(0))
      return jnp.int32(0)

    lax.fori_loop(0, t_local, _p1_w1_tok, jnp.int32(0))

    def _p1b_w1_dest(d, carry):
      send_cur, remote_tot = carry
      cnt = disp_tab_sm[ep + d, 0]
      disp_tab_sm[d, 0] = send_cur
      disp_tab_sm[2 * ep + d, 0] = me_base
      if not per_dest_perm:
        assert perm_2d_sm is not None

        def _copy_perm(r, _):
          perm_sm[send_cur + r] = perm_2d_sm[d, r]
          return jnp.int32(0)

        lax.fori_loop(0, cnt, _copy_perm, jnp.int32(0))
      next_send = send_cur + cnt
      next_remote = remote_tot + jnp.where(d != me, cnt, jnp.int32(0))
      return (next_send, next_remote)

    _, remote_tot = lax.fori_loop(
        0, ep, _p1b_w1_dest, (jnp.int32(0), jnp.int32(0))
    )
    disp_tab_sm[3 * ep, 0] = recv_cnt_sm[0]
    disp_tab_sm[3 * ep + 1, 0] = remote_tot
    if has_wave_bounds:
      if compute_visit_me:
        disp_tab_sm[3 * ep + 2, 0] = nonempty_me
      elif wave_bounds_me_sm is not None:
        disp_tab_sm[3 * ep + 2, 0] = wave_bounds_me_sm[0]
  else:
    assert wave_of_sm is not None
    assert seen_idx_sm is not None and tok_d_sm is not None
    assert tok_w_sm is not None and tok_nd_sm is not None
    assert cur_slot_sm is not None
    wave_of_ref = wave_of_sm
    seen_idx_ref = seen_idx_sm
    tok_d_ref = tok_d_sm
    tok_w_ref = tok_w_sm
    tok_nd_ref = tok_nd_sm
    cur_slot_ref = cur_slot_sm

    def _p1a_tok(t, _):
      tok_nd_ref[t] = jnp.int32(0)

      def _p1a_k(k, _):
        e = pairs_me_sm[t, k]
        valid = (e >= jnp.int32(0)) & (e < jnp.int32(e_total))

        @pl.when(valid)
        def _():
          d = _div_g(e)
          g = _mod_g(e)
          w = wave_of_ref[d, g]
          is_new = seen_stamp_sm[d] != t

          @pl.when(is_new)
          def _():
            nd = tok_nd_ref[t]
            seen_stamp_sm[d] = t
            seen_idx_ref[d] = nd
            tok_d_ref[t, nd] = d
            tok_w_ref[t, nd] = w
            tok_nd_ref[t] = nd + jnp.int32(1)

          idx = seen_idx_ref[d]
          tok_w_ref[t, idx] = jnp.minimum(tok_w_ref[t, idx], w)

        return jnp.int32(0)

      lax.fori_loop(0, topk, _p1a_k, jnp.int32(0))
      nd_t = tok_nd_ref[t]

      def _p1a_cnt(j, _):
        d = tok_d_ref[t, j]
        w = tok_w_ref[t, j]
        disp_tab_sm[ep + d, w] = disp_tab_sm[ep + d, w] + jnp.int32(1)
        return jnp.int32(0)

      lax.fori_loop(0, nd_t, _p1a_cnt, jnp.int32(0))
      return jnp.int32(0)

    lax.fori_loop(0, t_local, _p1a_tok, jnp.int32(0))

    def _p1b_dest(d, carry):
      send_cur, remote_tot = carry
      slot_cur = jnp.int32(0)
      is_remote = d != me
      for w in range(waves):
        cnt = disp_tab_sm[ep + d, w]
        disp_tab_sm[d, w] = send_cur
        disp_tab_sm[2 * ep + d, w] = me_base + slot_cur
        cur_slot_ref[d, w] = slot_cur
        if cur_send_sm is not None:
          cur_send_sm[d, w] = send_cur
        send_cur = send_cur + cnt
        slot_cur = slot_cur + cnt
        remote_tot = remote_tot + jnp.where(is_remote, cnt, jnp.int32(0))
      return (send_cur, remote_tot)

    _, remote_tot = lax.fori_loop(
        0, ep, _p1b_dest, (jnp.int32(0), jnp.int32(0))
    )
    for w in range(waves):
      disp_tab_sm[3 * ep, w] = recv_cnt_sm[w]
      disp_tab_sm[3 * ep + 1, w] = remote_tot if w == 0 else jnp.int32(0)
      if has_wave_bounds and wave_bounds_me_sm is not None:
        disp_tab_sm[3 * ep + 2, w] = wave_bounds_me_sm[w]

    def _p1c_tok(t, _):
      nd_t = tok_nd_ref[t]

      def _p1c_j(j, _):
        d = tok_d_ref[t, j]
        w = tok_w_ref[t, j]
        sl = cur_slot_ref[d, w]
        cur_slot_ref[d, w] = sl + jnp.int32(1)
        row_on_dest_sm[d, t] = me_base + sl
        if per_dest_perm:
          perm_sm[d, sl] = t
        else:
          assert cur_send_sm is not None
          so = cur_send_sm[d, w]
          cur_send_sm[d, w] = so + jnp.int32(1)
          perm_sm[so] = t
        return jnp.int32(0)

      lax.fori_loop(0, nd_t, _p1c_j, jnp.int32(0))
      return jnp.int32(0)

    lax.fori_loop(0, t_local, _p1c_tok, jnp.int32(0))

  # Pass 2a: prefix-sum hist_me per destination into packed SMEM list offsets.
  def _p2a_dest(d, carry):
    pack_row, remote_list_rows = carry
    d_base = d * jnp.int32(g_local)

    def _g_prefix(g, n_d):
      e = d_base + g
      cursor_sm[e] = n_d
      return n_d + hist_me_sm[e]

    n_d = lax.fori_loop(0, g_local, _g_prefix, jnp.int32(0))
    n_plane_rows = jnp.right_shift(n_d + jnp.int32(LANES - 1), jnp.int32(7))
    n_body_rows = jnp.int32(nw) * n_plane_rows
    dma_rows_d = jnp.int32(1) + n_body_rows
    list_tab_sm[d] = dma_rows_d
    remote_list_rows = remote_list_rows + jnp.where(
        d != me, dma_rows_d, jnp.int32(0)
    )
    stride_words_sm[d] = jnp.left_shift(n_plane_rows, jnp.int32(7))
    body_row_off_sm[d] = pack_row
    if write_headers:
      def _zero_hdr(lane, _):
        lists_sm[pack_row, lane] = jnp.int32(0)
        return jnp.int32(0)

      lax.fori_loop(0, LANES, _zero_hdr, jnp.int32(0))
      lists_sm[pack_row, HDR_N] = n_d
      lists_sm[pack_row, HDR_SRC] = me
      lists_sm[pack_row, HDR_WAVES] = jnp.int32(waves)
      lists_sm[pack_row, HDR_NW] = jnp.int32(nw)
      for w in range(waves):
        lists_sm[pack_row, HDR_SEND_OFF + w] = disp_tab_sm[d, w]
        lists_sm[pack_row, HDR_CNT_DIR + w] = disp_tab_sm[ep + d, w]

      def _zero_body_row(br, _):
        rr = pack_row + jnp.int32(1) + br

        def _zero_lane(lane, _):
          lists_sm[rr, lane] = jnp.int32(0)
          return jnp.int32(0)

        lax.fori_loop(0, LANES, _zero_lane, jnp.int32(0))
        return jnp.int32(0)

      lax.fori_loop(0, n_body_rows, _zero_body_row, jnp.int32(0))
      body_start_word = jnp.left_shift(pack_row + jnp.int32(1), jnp.int32(7))
      next_pack_row = pack_row + dma_rows_d
    else:
      body_start_word = jnp.left_shift(pack_row, jnp.int32(7))
      next_pack_row = pack_row + n_body_rows

    def _g_abs(g, _):
      e = d_base + g
      cursor_sm[e] = body_start_word + cursor_sm[e]
      return jnp.int32(0)

    lax.fori_loop(0, g_local, _g_abs, jnp.int32(0))
    return (next_pack_row, remote_list_rows)

  _, remote_list_rows = lax.fori_loop(
      0, ep, _p2a_dest, (jnp.int32(0), jnp.int32(0))
  )
  list_tab_sm[ep] = remote_list_rows

  # Pass 2b: scatter all (t, k) pairs in ascending (t, k) order into lists_sm.
  scale_is_1d = scale_bits_sm is not None and scale_bits_sm.ndim == 1

  def _p2b_tok(t, _):
    sb_t = scale_bits_sm[t] if scale_is_1d else jnp.int32(0)

    def _p2b_k(k, _):
      e = pairs_me_sm[t, k]
      valid = (e >= jnp.int32(0)) & (e < jnp.int32(e_total))

      @pl.when(valid)
      def _():
        d = _div_g(e)
        wpos = cursor_sm[e]
        cursor_sm[e] = wpos + jnp.int32(1)
        r0 = jnp.right_shift(wpos, jnp.int32(7))
        l0 = jnp.bitwise_and(wpos, jnp.int32(LANES - 1))
        lists_sm[r0, l0] = row_on_dest_sm[d, t]
        if nw == 2:
          assert scale_bits_sm is not None
          sb = sb_t if scale_is_1d else scale_bits_sm[t, k]
          wpos1 = wpos + stride_words_sm[d]
          r1 = jnp.right_shift(wpos1, jnp.int32(7))
          l1 = jnp.bitwise_and(wpos1, jnp.int32(LANES - 1))
          lists_sm[r1, l1] = sb

      return jnp.int32(0)

    lax.fori_loop(0, topk, _p2b_k, jnp.int32(0))
    return jnp.int32(0)

  lax.fori_loop(0, t_local, _p2b_tok, jnp.int32(0))

  # Pass 2c: local SMEM -> HBM copies of each destination's live list rows.
  def start_list_dmas():
    def _start_d(d, _):
      if write_headers:
        n_rows = list_tab_sm[d]
        r0 = body_row_off_sm[d]
        pltpu.make_async_copy(
            lists_sm.at[pl.ds(r0, n_rows)],
            tg_lists_hbm.at[pl.ds(d * list_rows_per_dest, n_rows)],
            dma_sem,
        ).start()
      else:
        n_body = list_tab_sm[d] - jnp.int32(1)

        @pl.when(n_body > jnp.int32(0))
        def _():
          r0 = body_row_off_sm[d]
          pltpu.make_async_copy(
              lists_sm.at[pl.ds(r0, n_body)],
              tg_lists_hbm.at[pl.ds(d * list_rows_per_dest + 1, n_body)],
              dma_sem,
          ).start()

      return jnp.int32(0)

    lax.fori_loop(0, ep, _start_d, jnp.int32(0))

  def wait_list_dmas():
    def _wait_d(d, _):
      if write_headers:
        n_rows = list_tab_sm[d]
        r0 = body_row_off_sm[d]
        pltpu.make_async_copy(
            lists_sm.at[pl.ds(r0, n_rows)],
            tg_lists_hbm.at[pl.ds(d * list_rows_per_dest, n_rows)],
            dma_sem,
        ).wait()
      else:
        n_body = list_tab_sm[d] - jnp.int32(1)

        @pl.when(n_body > jnp.int32(0))
        def _():
          r0 = body_row_off_sm[d]
          pltpu.make_async_copy(
              lists_sm.at[pl.ds(r0, n_body)],
              tg_lists_hbm.at[pl.ds(d * list_rows_per_dest + 1, n_body)],
              dma_sem,
          ).wait()

      return jnp.int32(0)

    lax.fori_loop(0, ep, _wait_d, jnp.int32(0))

  return start_list_dmas, wait_list_dmas


def decode_fast_plan(
    pairs_g,
    me,
    hist_me,
    recv_cnt,
    *,
    ep,
    t_local,
    g_local,
    waves,
    scale_bits=None,
    wave_of=None,
    wave_bounds_me=None,
    rows_by_dest_me=None,
    per_dest_perm=True,
    send_alloc=None,
    write_headers=False,
    tg_lists_buf=None,
    pairs_me=None,
):
  """Builds `(disp_tab, perm, list_tab, tg_lists, visit_me)` in one SMEM kernel.

  Replaces `source_plan`, `send_perm`/`send_perm_by_dest`, `dispatch_table`,
  `build_send_lists`, `list_tab`, and (for `waves == 1`) `wave_plan` at decode
  shapes (`t_local <= _DECODE_FAST_PLAN_MAX_T`).

  Args:
    pairs_g: [T, K] i32 global expert ids in mesh order.
    me: This shard's index.
    hist_me: [ep * g_local] i32 per-expert pair counts of shard `me`.
    recv_cnt: [W] i32 rows `me` receives per wave (`receiver_counts`).
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    g_local: Experts per shard.
    waves: W.
    scale_bits: Optional [t_local] or [t_local, K] i32 activation-scale bits.
    wave_of: [ep, g_local] i32 expert waves (required when `waves > 1`).
    wave_bounds_me: Optional [W] i32 wave cuts for row `3 * ep + 2`.
    rows_by_dest_me: Optional [ep, g_local] i32 counts for `me`'s experts;
      when passed with `waves == 1`, computes `visit_me` and `wave_bounds_me`
      inside the kernel.
    per_dest_perm: Whether `perm` is `[ep, t_local]` (row_send) or
      `[send_alloc]` (packed send).
    send_alloc: Length of 1D `perm` when `per_dest_perm=False`.
    write_headers: Whether to populate list headers and zero trailing words.
    tg_lists_buf: Optional pre-allocated `[ep * list_rows, 128]` i32 HBM buffer
      aliased to the `tg_lists` output.

  Returns:
    `(disp_tab, perm, list_tab, tg_lists, visit_me)` (`visit_me` is None unless
    `rows_by_dest_me` is passed with `waves == 1`).
  """
  me_i32 = jnp.asarray(me, jnp.int32)
  if pairs_me is not None:
    pairs_me = pairs_me.astype(jnp.int32)
    topk = pairs_me.shape[1]
  elif pairs_g.shape[0] == ep and t_local > 1:
    topk = pairs_g.shape[1] // t_local
    pairs_me = lax.dynamic_index_in_dim(
        pairs_g.astype(jnp.int32), me_i32, axis=0, keepdims=False
    ).reshape(t_local, topk)
  else:
    _, topk = pairs_g.shape
    pairs_me = lax.dynamic_slice(
        pairs_g.astype(jnp.int32), (me_i32 * t_local, 0), (t_local, topk)
    )
  nw = 1 if scale_bits is None else 2
  if not per_dest_perm and send_alloc is None:
    raise ValueError("send_alloc is required when per_dest_perm=False")
  if waves > 1 and wave_of is None:
    raise ValueError("wave_of is required when waves > 1")
  compute_visit_me = rows_by_dest_me is not None
  if compute_visit_me and waves != 1:
    raise ValueError("rows_by_dest_me visit_me fast path requires waves == 1")
  has_wave_bounds = (
      (wave_bounds_me is not None or compute_visit_me) and ep >= 3
  )
  list_rows_per_dest = list_rows(t_local, topk, nw)
  tg_shape = (ep * list_rows_per_dest, LANES)

  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  anyspec = pl.BlockSpec(memory_space=pl.ANY)
  args = [
      jnp.reshape(me_i32, (1,)),
      pairs_me,
      hist_me.astype(jnp.int32).reshape(ep * g_local),
      recv_cnt.astype(jnp.int32).reshape(waves),
  ]
  in_specs = [smem, smem, smem, smem]
  has_scale = scale_bits is not None
  if has_scale:
    sb = scale_bits.astype(jnp.int32)
    if sb.ndim not in (1, 2) or sb.shape[0] != t_local:
      raise ValueError(
          f"scale_bits must be [t_local] or [t_local, topk], got {sb.shape}"
      )
    args.append(sb)
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
  has_buf = tg_lists_buf is not None
  aliases = {}
  if has_buf:
    if (
        _PALLAS_INTERPRET is None
        and getattr(pl.pallas_call, "__qualname__", "") == "pallas_call"
    ):
      tg_lists_buf = pltpu.with_memory_space_constraint(tg_lists_buf, pltpu.HBM)
    aliases[len(args)] = 3
    args.append(tg_lists_buf)
    in_specs.append(anyspec)

  perm_shape = (ep, t_local) if per_dest_perm else (send_alloc,)
  out_shape = [
      jax.ShapeDtypeStruct((4 * ep, waves), jnp.int32),
      jax.ShapeDtypeStruct(perm_shape, jnp.int32),
      jax.ShapeDtypeStruct((ep + 1,), jnp.int32),
      jax.ShapeDtypeStruct(tg_shape, jnp.int32),
  ]
  out_specs = [smem, smem, smem, anyspec]
  if compute_visit_me:
    out_shape.append(jax.ShapeDtypeStruct((g_local,), jnp.int32))
    out_specs.append(smem)

  scratch_shapes = decode_fast_plan_scratch_shapes(
      ep=ep,
      g_local=g_local,
      t_local=t_local,
      topk=topk,
      waves=waves,
      nw=nw,
      per_dest_perm=per_dest_perm,
      write_headers=write_headers,
      compute_visit_me=compute_visit_me,
  ) + (pltpu.SemaphoreType.DMA(()),)

  def _kernel(*refs):
    it = iter(refs)
    me_ref = next(it)
    pairs_me_sm = next(it)
    hist_me_sm = next(it)
    recv_cnt_sm = next(it)
    scale_bits_sm = next(it) if has_scale else None
    wave_of_sm = next(it) if has_wave_of else None
    wave_bounds_me_sm = next(it) if has_wb_in else None
    rows_by_dest_me_sm = next(it) if compute_visit_me else None
    if has_buf:
      _ = next(it)  # aliased to tg_lists_hbm
    disp_tab_sm = next(it)
    perm_sm = next(it)
    list_tab_sm = next(it)
    tg_lists_hbm = next(it)
    visit_me_sm = next(it) if compute_visit_me else None
    rest = list(it)
    dma_sem = rest.pop()
    start_dmas, wait_dmas = emit_decode_plan_and_lists_body(
        me_ref=me_ref,
        pairs_me_sm=pairs_me_sm,
        hist_me_sm=hist_me_sm,
        recv_cnt_sm=recv_cnt_sm,
        scale_bits_sm=scale_bits_sm,
        wave_of_sm=wave_of_sm,
        wave_bounds_me_sm=wave_bounds_me_sm,
        rows_by_dest_me_sm=rows_by_dest_me_sm,
        disp_tab_sm=disp_tab_sm,
        perm_sm=perm_sm,
        list_tab_sm=list_tab_sm,
        visit_me_sm=visit_me_sm,
        tg_lists_hbm=tg_lists_hbm,
        scratch_refs=rest,
        dma_sem=dma_sem,
        ep=ep,
        g_local=g_local,
        t_local=t_local,
        topk=topk,
        waves=waves,
        nw=nw,
        list_rows_per_dest=list_rows_per_dest,
        per_dest_perm=per_dest_perm,
        send_alloc=send_alloc or 0,
        write_headers=write_headers,
        has_wave_bounds=has_wave_bounds,
    )
    start_dmas()
    wait_dmas()

  outs = pl.pallas_call(
      _kernel,
      out_shape=tuple(out_shape),
      in_specs=in_specs,
      out_specs=tuple(out_specs),
      scratch_shapes=scratch_shapes,
      input_output_aliases=aliases,
      interpret=_PALLAS_INTERPRET or False,
      name=f"dedup_decode_fast_plan_ep{ep}_w{waves}",
  )(*args)
  disp_tab, perm, list_tab, tg_lists = outs[:4]
  visit_me = outs[4] if compute_visit_me else None
  return disp_tab, perm, list_tab, tg_lists, visit_me


def slab_row_on_dest(
    local_blocks,
    block_hist,
    run_start_aligned,
    expert_base,
    me,
    *,
    ep,
    g_local,
    shard_stride,
    t_local,
    topk,
):
  """Returns the destination slab row each of shard me's pairs computes on.

  [K, t] i32: the row of its destination's own slab each of shard me's
  pairs computes on (for oracles).

  expert_base[e] - owner(e) * stride + run_start_aligned[e, me] + the
  pair's rank among me's pairs of expert e, the rank being fused_host's
  my_off + my_rank (as `pos` in build_routing_tables_sharded).

  Args:
    local_blocks: Routing blocks of me's pairs.
    block_hist: Per-block expert histogram.
    run_start_aligned: [E, ep] aligned run starts.
    expert_base: [E] slab base of each expert.
    me: This shard's index.
    ep: Expert-parallel width.
    g_local: Experts per shard.
    shard_stride: Slab rows per shard.
    t_local: Tokens per shard.
    topk: Experts per token.

  Returns:
    [K, t] i32 slab rows (for oracles).
  """
  e_total = ep * g_local
  bins = jnp.arange(e_total, dtype=jnp.int32)
  my_off = jnp.cumsum(block_hist, axis=0) - block_hist
  my_run_start = lax.dynamic_slice(
      run_start_aligned, (0, jnp.asarray(me, jnp.int32)), (e_total, 1)
  ).reshape(e_total)
  base = expert_base - (bins // g_local) * shard_stride + my_run_start
  block = local_blocks.shape[1]
  rows = fused_host._lookup_per_pair(  # pylint: disable=protected-access
      my_off + base[None, :], local_blocks, bins
  ) + fused_host._rank_within_block(  # pylint: disable=protected-access
      local_blocks, block
  )
  return rows.reshape(t_local, topk).T


def build_shard_routing_tables(
    topk_idx,
    me,
    *,
    e_total,
    ep,
    t_local,
    block,
    tile_m,
    shard_stride,
    all_gather_rows=None,
    gathered=None,
    include_mirror_pos=True,
    include_slab_row=False,
):
  """Builds ShardRoutingTables without dead `slab_row` work and with fused lookup.

  When `tg_assemble` builds `token_gather` and `scale_slab` from the sender
  lists, the global `slab_row` pass of `fused_host.build_routing_tables_sharded`
  (over all `T * K` pairs) is unused. In addition, `pos_table` and
  `mirror_table` share the `[blocks_per_dest, block, E]` expert-match mask
  `my_blocks == bins`, so computing `pos` and `mirror_pos` from a single mask
  (or skipping `mirror_pos` on an unquantized wire) halves the 3D lookup work
  while keeping every field bit-identical to
  `fused_host.build_routing_tables_sharded`.

  Args:
    topk_idx: `[T, K]` expert ids, or None when `gathered` is supplied.
    me: This shard's index.
    e_total: Total routed experts (`E`).
    ep: Expert-parallel width.
    t_local: Tokens per shard.
    block: Pair block size for the local histogram.
    tile_m: Tile height `c`.
    shard_stride: Slab rows per shard.
    all_gather_rows: Collective callback when `gathered` is None.
    gathered: `fused_host.GatheredPairs` from the routing exchange.
    include_mirror_pos: Whether to compute `mirror_pos` (needed for quantized
      wires).
    include_slab_row: Whether to compute `slab_row` across all `T * K` pairs.

  Returns:
    A `fused_host.ShardRoutingTables` instance.
  """
  if include_slab_row:
    return fused_host.build_routing_tables_sharded(
        topk_idx,
        me,
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=tile_m,
        shard_stride=shard_stride,
        all_gather_rows=all_gather_rows,
        gathered=gathered,
    )
  if (gathered is None) == (all_gather_rows is None):
    raise ValueError("pass exactly one of all_gather_rows and gathered")
  num_tokens, topk = (
      topk_idx.shape
      if gathered is None
      else (gathered.n_tokens, gathered.topk)
  )
  n = num_tokens * topk
  g_local = e_total // ep
  fused_host._refuse_bad_plan_args(  # pylint: disable=protected-access
      num_tokens,
      topk,
      e_total=e_total,
      ep=ep,
      t_local=t_local,
      block=block,
      tile_m=tile_m,
  )
  if shard_stride % tile_m:
    raise ValueError(
        f"the per-shard slab stride {shard_stride} is not a "
        f"whole number of {tile_m}-row tiles"
    )
  n_blocks = n // block
  blocks_per_dest = (t_local * topk) // block
  me = jnp.asarray(me, dtype=jnp.int32)
  bins = jnp.arange(e_total, dtype=jnp.int32)
  if gathered is None:
    assert all_gather_rows is not None
    expert_blocks = (
        topk_idx.reshape(-1).astype(jnp.int32).reshape(n_blocks, block)
    )
    my_blocks = lax.dynamic_slice(
        expert_blocks, (me * blocks_per_dest, 0), (blocks_per_dest, block)
    )
    my_hist = fused_host._block_hist(  # pylint: disable=protected-access
        my_blocks, bins
    )
    rows_by_dest = all_gather_rows(my_hist.sum(axis=0, keepdims=True))
  else:
    my_blocks, my_hist = gathered.local_blocks, gathered.block_hist
    rows_by_dest = gathered.rows_by_dest
  my_off = jnp.cumsum(my_hist, axis=0) - my_hist

  run_rows = rows_by_dest.T
  run_rows_aligned = (run_rows + (fused_host.ROWBLK - 1)) & jnp.int32(
      -fused_host.ROWBLK
  )
  run_start_aligned = jnp.cumsum(run_rows_aligned, axis=1) - run_rows_aligned
  expert_rows_aligned = run_rows_aligned.sum(axis=1)
  rows_by_shard = expert_rows_aligned.reshape(ep, g_local)
  local_base = jnp.cumsum(rows_by_shard, axis=1) - rows_by_shard
  expert_base = (
      local_base + (jnp.arange(ep, dtype=jnp.int32)[:, None] * shard_stride)
  ).reshape(e_total)
  region_rows = run_rows_aligned.reshape(ep, g_local, ep)
  rows_per_dest = region_rows.transpose(2, 0, 1).reshape(ep, ep * g_local)
  recv_base = (jnp.cumsum(rows_per_dest, axis=1) - rows_per_dest).reshape(
      ep, ep, g_local
  )
  recv_rows = rows_per_dest.sum(axis=1)

  my_recv_base = lax.dynamic_slice(
      recv_base, (me, 0, 0), (1, ep, g_local)
  ).reshape(e_total)
  pos_table = my_off + my_recv_base[None, :]
  my_rank = fused_host._rank_within_block(  # pylint: disable=protected-access
      my_blocks, block
  )
  on_bin = my_blocks[:, :, None] == bins[None, None, :]
  pos = (
      jnp.sum(jnp.where(on_bin, pos_table[:, None, :], 0), axis=-1) + my_rank
  ).reshape(t_local, topk)

  run_tiles = fused_host.mirror_run_tiles(
      run_start_aligned, run_rows_aligned, tile_m
  )
  if include_mirror_pos:
    shift = fused_host.pow2_shift(tile_m, "tile_m")
    my_run_start = lax.dynamic_slice(
        run_start_aligned, (0, me), (e_total, 1)
    ).reshape(e_total)
    mirror_table = (
        my_off
        + (
            (
                jnp.right_shift(my_recv_base, shift)
                + 2 * jnp.arange(e_total, dtype=jnp.int32)
            )
            * tile_m
            + (my_run_start & (tile_m - 1))
        )[None, :]
    )
    mirror_pos = (
        jnp.sum(jnp.where(on_bin, mirror_table[:, None, :], 0), axis=-1)
        + my_rank
    ).reshape(t_local, topk)
  else:
    mirror_pos = jnp.zeros((t_local, topk), dtype=jnp.int32)

  slab_row = jnp.zeros((0,), dtype=jnp.int32)
  return fused_host.ShardRoutingTables(
      pos=pos,
      mirror_pos=mirror_pos,
      run_tiles=run_tiles,
      slab_row=slab_row,
      run_rows=run_rows,
      run_rows_aligned=run_rows_aligned,
      run_start_aligned=run_start_aligned,
      region_rows=region_rows,
      recv_base=recv_base,
      recv_rows=recv_rows,
      expert_rows_aligned=expert_rows_aligned,
      expert_base=expert_base,
      rows_alloc=ep * shard_stride,
      n_tokens=num_tokens,
      topk=topk,
  )
