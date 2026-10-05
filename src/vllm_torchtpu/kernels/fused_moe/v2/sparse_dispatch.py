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
"""Experimental sparse, expert-wave token dispatch for fused EP MoE v2.

`fused_ep_moe_v2(..., sparse_waves=W)` with W > 0 replaces the activation
all-gather, which delivers every token to every shard, with a sparse exchange:
each shard sends every (token, destination shard) pair with a hit once, through
the split-phase Pallas ragged all-to-all in async_dispatch.py.

Every shard orders its local experts fewest rows first, from the per-expert
counts the routing exchange already all-gathers, so senders and receivers
agree on the order without exchanging it. The exchange is laid out and issued
in W groups ("waves") of experts in that order, and the fused kernel, built
with the same sparse_waves, waits for an expert's wave only before that
expert's first tile: the smallest experts start soonest and the rest of the
exchange lands under their matmuls. The routing tables are built while the
exchange is in flight.

The exchange takes as long as the hottest shard's ingress, where the
all-gather moves a fixed T rows to every shard, so this pays off when routing
is balanced and loses under strong skew; nothing switches between the two at
run time. Every buffer is sized for the worst case (all tokens on one shard),
so no routing drops a token.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2 import async_dispatch
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    ROWBLK,
    align_up,
    sparse_experts_per_wave,
)

# Rows in the first copy of each destination's share of a wave; the rest of
# the share follows in one copy (async_dispatch._start_waves_kernel).
CHUNK_ROWS = 8


class SparseDispatch(NamedTuple):
    """An exchange in flight and the plan it was laid out by.

    fut: async_dispatch.ragged_a2a_start_waves's (send_sem, recv_sems, send,
      recv). The fused kernel reads its rows from `recv` and waits on
      `recv_sems`; `finish` drains `send_sem`.
    disp_tab [4 * ep, waves] i32: the dispatch plan (see async_dispatch).
    visit_order [ep, g_local] i32: every shard's local experts, fewest rows
      first, empty experts last.
    posmap [T] i32: the receive-buffer row of every global token on this shard.
    """

    fut: tuple
    disp_tab: jax.Array
    visit_order: jax.Array
    posmap: jax.Array


def ascending_expert_order(hist, *, ep, g_local):
    """Each shard's local experts, fewest rows first, empties last.

    hist [ep * g_local] i32: rows per global expert (the exchanged routing
    histogram, identical on every shard, so every sender and receiver derives
    the same order). Returns
      pos [ep, g_local] i32: the visit position of local expert e on shard d;
      order [ep, g_local] i32: the local expert at visit position p on d --
        on `me`, order[me] is a drop-in visit list (the nonempty experts come
        first, so its first n_visit entries are exactly the visited ones).
    Ties break by ascending index, as in expert_visit_list.
    """
    h = hist.reshape(ep, g_local).astype(jnp.int32)
    idx = jnp.arange(g_local, dtype=jnp.int32)
    # 2**24 > any row count; 2**24 * g_local + g_local stays below 2**31.
    key = jnp.where(h > 0, jnp.minimum(h, 2**24 - 1), 2**24) * g_local + idx
    pos = jnp.sum((key[:, None, :] < key[:, :, None]).astype(jnp.int32), axis=2)
    order = jnp.sum(
        jnp.where(pos[:, None, :] == idx[None, :, None], idx[None, None, :], 0), axis=2
    )
    return pos, order


def stream_groups(ep, waves):
    """Source-shard group sizes for streaming a shard's busiest expert.

    Wave w carries group w of that expert's source shards, counted from the
    destination in mesh order: (ep/8, ep/8, ep/4, ep/2) at 4 waves. The
    groups double so the first ones land early, under the other experts'
    matmuls, and senders spread their share of it over the waves because
    each destination counts from itself. () where there is nothing to
    stream: one wave, or an ep the halving does not divide.
    """
    if waves < 2 or ep % (1 << (waves - 1)):
        return ()
    sizes = [ep >> (waves - w) for w in range(1, waves)]
    return (sizes[0], *sizes)


def stream_split(hist, order, *, ep, g_local):
    """The local expert each shard streams: its last visited, the busiest.

    hist [ep * g_local] i32 and order as ascending_expert_order's. Returns
    split [ep] i32. On a shard with no rows at all it names an empty expert,
    which no pair references.
    """
    h = hist.reshape(ep, g_local)
    last = jnp.maximum(jnp.sum((h > 0).astype(jnp.int32), axis=1) - 1, 0)
    idx = jnp.arange(g_local, dtype=jnp.int32)
    return jnp.sum(jnp.where(idx[None, :] == last[:, None], order, 0), axis=1)


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
    split=None,
    stream=(),
):
    """Sparse expert-wave dispatch plan for shard `me`.

    pairs_g [T, K] i32 global expert ids in mesh order (dest = id // g_local).
    wave_of [ep, g_local] i32 is the wave of each shard's local experts (its
    visit position // host.sparse_experts_per_wave, see
    `ascending_expert_order`). stream (stream_groups; () is off) instead
    puts each shard's expert split [ep] on wave g for source group g,
    counted from that shard in mesh order.
    Every (token, dest) pair with a hit is sent ONCE, in its first-hit wave.
    The receive buffer on dest d is laid out by (wave, source), tokens
    ascending within each segment. Returns:
      perm [send_alloc] i32: local token row of each packed send row;
      tab [4 * ep, waves] i32: send offset / send rows / dest offset per
        (d, w), rows received per wave (row 3 ep), total rows sent to other
        shards (row 3 ep + 1, col 0);
      posmap [T] i32: receive-buffer row of global token t on `me` (0 where
        `me` does not need t, which no live slab row reads).
    """
    T, K = pairs_g.shape
    # The token axis stays minor in every array below. With topk minor, a
    # [T, K] int32 array fills K of a vreg's 128 lanes, and the relayouts and
    # elementwise ops on it cost more than the plan's arithmetic.
    pk = pairs_g.T.astype(jnp.int32)  # [K, T]
    dest = pk // g_local
    table = wave_of.astype(jnp.int32)  # [ep, g_local]
    if stream:
        if split is None:
            raise ValueError("stream needs split, each shard's streamed expert")
        # A streamed expert's entry carries `waves` on top, so the one
        # lookup below also tells which pairs are streamed.
        streamed = jnp.arange(g_local, dtype=jnp.int32)[None, :] == split[:, None]
        table = table + jnp.where(streamed, waves, 0)
    # The wave of every pair is table[dest, local expert]. Not jnp.take: XLA's
    # TPU gather of 98K pairs from a 384-entry table cost 612 us on DSV4; nor
    # a compare over all E experts: 36 us. A one-hot of the local expert
    # times the table on the MXU gives each pair its wave on every shard, and
    # a compare over the ep shards keeps its own. The entries are below
    # 2 * waves, so bf16 holds them exactly.
    g_iota = jnp.arange(g_local, dtype=jnp.int32)
    local = (pk - dest * g_local).reshape(1, K * T)
    onehot = (local == g_iota[:, None]).astype(jnp.bfloat16)  # [g_local, K*T]
    by_dest = jnp.dot(
        table.astype(jnp.bfloat16), onehot, preferred_element_type=jnp.float32
    )  # [ep, K*T]
    d_iota = jnp.arange(ep, dtype=jnp.int32)
    wave = (
        jnp.sum(
            jnp.where(dest.reshape(1, K * T) == d_iota[:, None], by_dest, 0.0), axis=0
        )
        .astype(jnp.int32)
        .reshape(K, T)
    )
    if stream:
        # The source group of each streamed pair, counted from its
        # destination: the destination's own rows come first.
        src = jnp.arange(T, dtype=jnp.int32)[None, :] // t_local
        rel = lax.rem(src - dest + ep, ep)
        group = jnp.zeros_like(rel)
        bound = 0
        for size in stream[:-1]:
            bound += size
            group = group + (rel >= bound).astype(jnp.int32)
        wave = jnp.where(wave >= waves, group, wave)
    # First-hit wave of each token on each shard; `waves` = no hit.
    fw = jnp.min(
        jnp.where(dest[:, None, :] == d_iota[None, :, None], wave[:, None, :], waves),
        axis=0,
    )  # [dest, T]
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
    extra = jnp.zeros((ep, waves), jnp.int32)
    extra = extra.at[0].set(recv_cnt).at[1, 0].set(remote)
    tab = jnp.concatenate([send_off, send_cnt, dst_off, extra], axis=0)
    # Exclusive prefix counts along the token axis as a matmul with a
    # strictly upper triangular ones matrix (0/1 in bf16, exact f32 sums):
    # XLA's reduce-window cumsum cost 36 us each at 1024 tokens per rank.
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
    posmap = jnp.sum(mine * posr, axis=0).reshape(T)
    return perm, tab, posmap


def _uninit_hbm(shape, dtype):
    """An HBM buffer with no defined contents, at no cost.

    A Pallas call whose body never writes its output hands XLA a fresh,
    unfilled buffer. The ragged all-to-all only defines the rows it lands and
    the kernel reads only those, so a zero fill (31 us at T=16K rows, and it
    grows with EP) would be pure waste.
    """
    return pl.pallas_call(
        lambda o_ref: None,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        name="uninit_recv",
    )()


def start(rows, pairs, rows_by_dest, me, *, ep, g_local, waves, stream=()):
    """Plan shard `me`'s exchange and issue it; returns a SparseDispatch.

    rows [t_local, lane blocks, 128]: this shard's token rows as they go on
      the wire (fp8 where the weight format quantizes activations).
    pairs [T, topk] i32: every shard's selected expert ids, in mesh order.
    rows_by_dest [ep, ep * g_local] i32: the all-gathered per-expert pair
      counts, identical on every shard.
    stream: stream_groups(ep, waves) to spread each shard's busiest expert
      over the waves, or () for whole-expert waves.
    """
    t_local = rows.shape[0]
    n_tokens, topk = pairs.shape
    hist = rows_by_dest.sum(axis=0)
    pos, order = ascending_expert_order(hist, ep=ep, g_local=g_local)
    # The streamed expert is the last visited, so it stays in place and
    # the others fill the waves before the last.
    split = stream_split(hist, order, ep=ep, g_local=g_local) if stream else None
    perm, disp_tab, posmap = dispatch_plan(
        pairs,
        me,
        ep=ep,
        t_local=t_local,
        g_local=g_local,
        waves=waves,
        # A token is sent to at most min(topk, ep) shards.
        send_alloc=align_up(t_local * min(topk, ep), ROWBLK),
        wave_of=pos // sparse_experts_per_wave(g_local, waves, stream=bool(stream)),
        split=split,
        stream=stream,
    )
    # A shard receives each token at most once.
    recv = _uninit_hbm((align_up(n_tokens, ROWBLK), *rows.shape[1:]), rows.dtype)
    fut = async_dispatch.ragged_a2a_start_waves(
        rows[perm], recv, disp_tab, me, ep=ep, waves=waves, chunk_rows=CHUNK_ROWS
    )
    return SparseDispatch(fut, disp_tab, order, posmap)


def token_gather(slab_row, posmap, *, topk, ragged_stride):
    """host.shard_token_gather for the sparse layout.

    Scatters each routed pair's RECEIVE row, rather than its token number,
    onto the slab row it computes on (slab_row, from host.local_slab_rows;
    pairs of other shards fall outside the slab and are dropped). posmap
    repeated topk times is the per-pair value, so no T * topk-element gather
    is needed.
    """
    n_tokens = posmap.shape[0]
    return jnp.clip(
        jnp.zeros((ragged_stride,), jnp.int32)
        .at[slab_row]
        .add(jnp.repeat(posmap, topk), mode="drop"),
        0,
        n_tokens - 1,
    )


def finish(sparse, arrivals, *, ep):
    """Drain this shard's outgoing copies; returns `arrivals` (aliased).

    Called after the fused kernel, which gave the copies the whole kernel to
    finish. Threading the kernel's arrivals through orders the drain after
    the kernel and keeps it from being dead-code-eliminated
    (async_dispatch.ragged_a2a_send_done).
    """
    send_sem, _, send, _ = sparse.fut
    return async_dispatch.ragged_a2a_send_done(
        send_sem, send, sparse.disp_tab, arrivals, ep=ep
    )
