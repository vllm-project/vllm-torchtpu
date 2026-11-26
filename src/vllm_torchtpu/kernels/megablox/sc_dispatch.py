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
"""SparseCore dispatch of the capacity exchange's fp8 token rows.

The capacity exchange sends each token row that reaches a member once, the
receiver copies its routes' rows into route order, and only then can the
expert kernel start. Here a SparseCore kernel sends one row per route straight
to its row of the receiver's route order (the order of `local_route_plan`):
local expert by local expert ("waves"), each wave signalling its own DMA
semaphore in the receiver's TensorCore memory. The SC kernel exits as soon as
its DMAs are issued, so a TensorCore kernel can compute on the waves that
have landed while later ones are in flight.

Protocol, per call:
- The receive buffer and the wave semaphores are outputs of the dispatch
  kernel, so they are a member's only once its kernel runs: a barrier over
  the members (barrier semaphore with a collective id) keeps peers from
  writing earlier.
- A count-based barrier is exact only if every member hears from every other
  member in every call: one that expects nothing from a peer could otherwise
  finish, enter the next call and signal its barrier while the peer still
  counts this one. A member that has no route for a peer therefore sends it
  one sentinel row on the extra semaphore ``num_waves``, into a row past the
  routes' rows.
- Mosaic puts a DMA's send semaphore on its source's core type, so the send
  semaphore is an SC semaphore, output by the dispatch kernel and waited for
  every row sent by `sc_drain`, which also holds the source rows alive.
- The consumer waits ``group_sizes[e]`` rows on wave semaphore ``e`` and
  ``sentinels`` rows on semaphore ``num_waves``, one row's bytes per row.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

I32 = jnp.int32
# Header words of a send table ([#sends, 0, ...]); the sends follow, so the
# table stays whole 16-word SMEM rows.
TABLE_HEADER = 16
# A send packs (wave, destination - this member mod members, source token,
# landing row) high to low, so that sorting the words orders the sends.
_LAND_BITS = 14
_SRC_BITS = 6
_DST_BITS = 4
_WAVE_BITS = 4
_SRC_SHIFT = _LAND_BITS
_DST_SHIFT = _SRC_SHIFT + _SRC_BITS
_WAVE_SHIFT = _DST_SHIFT + _DST_BITS
_NO_SEND = (1 << 31) - 1
# Mosaic's tensor core axis of the wave semaphores' memory.
_TC_AXIS = "tc"


class SendPlan(NamedTuple):
    """This member's sends of one SC dispatch, and what it receives."""

    table: jax.Array  # [TABLE_HEADER + M] int32: #sends, then packed sends
    group_sizes: jax.Array  # [L] int32: rows each wave lands here
    sentinels: jax.Array  # [1] int32: sentinel rows landing here


def sc_dispatch_supported(
    *, num_members: int, num_tokens: int, local_num_experts: int, recv_rows: int
) -> bool:
    """Whether a send packs into one word: sentinel rows included, the
    landing rows, tokens, members and waves fit their fields."""
    return (
        recv_rows + num_members <= 1 << _LAND_BITS
        and num_tokens <= 1 << _SRC_BITS
        and num_members <= 1 << _DST_BITS
        and local_num_experts + 1 <= 1 << _WAVE_BITS
    )


def recv_buffer_rows(recv_rows: int, num_members: int) -> int:
    """Rows of the receive buffer: ``recv_rows`` route rows, then one sentinel
    row per member, rounded up to whole 16-row groups."""
    return pl.cdiv(recv_rows + num_members, 16) * 16


@functools.partial(
    jax.jit,
    static_argnames=("local_num_experts", "tp_size", "skip_padded", "recv_rows"),
)
def sc_send_plan(
    words,
    ep_rank,
    fits,
    *,
    local_num_experts: int,
    tp_size: int,
    skip_padded: bool,
    recv_rows: int,
) -> SendPlan:
    """The send table of this member's SC dispatch, from the gathered routing.

    ``words`` [D, K, S] int32: member s's route k of token t, expert id in the
    high 16 bits, bf16 weight in the low 16 (as `local_route_plan` takes it).
    ``ep_rank`` [1]: this rank's EP rank, DP position ``ep_rank // tp_size``.
    ``fits`` [1] (0: the step overflows, nothing is sent but sentinels).
    ``skip_padded``: a route with a zero or non-finite weight is not computed.

    A route of this member's token t goes to the member owning its expert on
    this TP rank when that member computes it (as `local_route_plan` decides),
    landing at the route's position in that member's route order: (local
    expert, source member, token), a token naming an expert twice taking
    consecutive rows in k order. Sends are wave-major (local expert), then by
    destination starting after this member, then by token.
    """
    num_members, topk, num_tokens = words.shape
    num_local = local_num_experts
    if not sc_dispatch_supported(
        num_members=num_members,
        num_tokens=num_tokens,
        local_num_experts=num_local,
        recv_rows=recv_rows,
    ):
        raise ValueError(
            f"sc_send_plan does not take {num_members} members, {num_tokens} "
            f"tokens, {num_local} local experts, {recv_rows} receive rows"
        )
    columns = num_members * num_local
    ep = ep_rank[0]
    me = ep // tp_size
    ids = words >> 16
    rank = ids // num_local
    computed = (
        (ids >= 0)
        & (rank < num_members * tp_size)
        & (rank % tp_size == ep % tp_size)
        & (fits[0] != 0)
    )
    if skip_padded:
        # A zero (subnormals flush to zero) or non-finite bf16 weight.
        exponent = words & 0x7F80
        computed = computed & (exponent != 0) & (exponent != 0x7F80)
    # (destination member, local expert) of each route; `columns`: none.
    col = jnp.where(computed, (rank // tp_size) * num_local + ids % num_local, columns)

    column = jnp.arange(columns, dtype=I32)
    per_member = jnp.sum(col[..., None] == column, axis=(1, 2), dtype=I32)  # [D, C]
    total = per_member.sum(axis=0).reshape(num_members, num_local)
    earlier = jnp.arange(num_members, dtype=I32)[:, None] < me
    before = jnp.sum(jnp.where(earlier, per_member, 0), axis=0)
    start = (jnp.cumsum(total, axis=1, dtype=I32) - total).reshape(-1)
    base = start + before
    mine_here = per_member.reshape(num_members, num_members, num_local)
    from_member = lax.dynamic_index_in_dim(mine_here, me, axis=1, keepdims=False)
    group_sizes = lax.dynamic_index_in_dim(total, me, keepdims=False)
    sentinels = jnp.sum(from_member.sum(axis=1) == 0, dtype=I32)[None]

    # This member's routes in (token, k) order.
    mine = lax.dynamic_index_in_dim(col, me, keepdims=False).T.reshape(-1)
    n = mine.shape[0]
    order = jnp.arange(n, dtype=I32)
    within = jnp.sum(
        (mine[None, :] == mine[:, None]) & (order[None, :] < order[:, None]),
        axis=1,
        dtype=I32,
    )
    land = (
        jnp.sum(jnp.where(mine[:, None] == column, base, 0), axis=1, dtype=I32) + within
    )
    dest = mine // num_local
    rot = (dest - me) % num_members
    send = jnp.where(
        mine < columns,
        jnp.left_shift(mine % num_local, _WAVE_SHIFT)
        | jnp.left_shift(rot, _DST_SHIFT)
        | jnp.left_shift(order // topk, _SRC_SHIFT)
        | land,
        _NO_SEND,
    )
    sends_to = jnp.sum(
        dest[:, None] == jnp.arange(num_members, dtype=I32), axis=0, dtype=I32
    )
    peer = jnp.arange(num_members, dtype=I32)
    sentinel = jnp.where(
        sends_to == 0,
        jnp.left_shift(num_local, _WAVE_SHIFT)
        | jnp.left_shift((peer - me) % num_members, _DST_SHIFT)
        | (recv_rows + me),
        _NO_SEND,
    )
    send = jnp.concatenate([send, sentinel])
    # The sends' words are distinct: each one's rank is its table slot.
    slot = jnp.sum(send[None, :] < send[:, None], axis=1, dtype=I32)
    width = pl.cdiv(send.shape[0], TABLE_HEADER) * TABLE_HEADER
    live = send != _NO_SEND
    body = jnp.sum(
        jnp.where(
            live[:, None] & (slot[:, None] == jnp.arange(width, dtype=I32)),
            send[:, None],
            0,
        ),
        axis=0,
        dtype=I32,
    )
    header = jnp.zeros((TABLE_HEADER,), I32).at[0].set(jnp.sum(live, dtype=I32))
    return SendPlan(jnp.concatenate([header, body]), group_sizes, sentinels)


def _sc_mesh():
    return plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1)


def sc_dispatch(
    src_words,
    table,
    *,
    axis: str,
    num_members: int,
    num_waves: int,
    recv_rows: int,
    collective_id: int,
):
    """Sends this member's rows by its `sc_send_plan` table; call inside
    shard_map on every member of mesh axis ``axis`` (``num_members``), same
    sizes on all.

    ``src_words`` [S, *row] uint32 (in HBM): the source rows, token-major
    (`fp8_source_rows`). Returns ``(recv, wave_sems, send_sem)``: the receive
    buffer [`recv_buffer_rows`, *row] (route rows from 0, sentinel rows from
    ``recv_rows``), ``num_waves + 1`` TC DMA semaphores (the last one counts
    sentinel rows) and the SC DMA semaphore of this member's sends, for
    `sc_drain`. The rows may still be in flight when this returns: a consumer
    waits for them on the wave semaphores.
    """
    sc_mesh = _sc_mesh()
    tc_mesh = pltpu.create_tensorcore_mesh(_TC_AXIS, num_cores=1)

    def body(src_hbm, tbl_hbm, recv, wave_sems, send_sems, tbl_s, tbl_sem):
        tbl_copy = pltpu.make_async_copy(tbl_hbm, tbl_s, tbl_sem)
        tbl_copy.start()
        me = lax.axis_index(axis)
        if num_members > 1:
            barrier = pltpu.get_barrier_semaphore()
            for k in range(1, num_members):
                peer = lax.rem(me + k, num_members)
                pl.semaphore_signal(barrier, 1, device_id={axis: peer})
            pl.semaphore_wait(barrier, num_members - 1)
        tbl_copy.wait()

        @pl.loop(0, tbl_s[0])
        def _(i):
            word = tbl_s[TABLE_HEADER + i]
            srl = lax.shift_right_logical
            land = word & ((1 << _LAND_BITS) - 1)
            src = srl(word, _SRC_SHIFT) & ((1 << _SRC_BITS) - 1)
            rot = srl(word, _DST_SHIFT) & ((1 << _DST_BITS) - 1)
            wave = srl(word, _WAVE_SHIFT)
            pltpu.async_remote_copy(
                src_hbm.at[src],
                recv.at[land],
                send_sems.at[0],
                wave_sems.at[wave],
                device_id={axis: lax.rem(me + rot, num_members), _TC_AXIS: 0},
            )

    return pl.kernel(
        body,
        out_type=(
            pltpu.HBM(
                (recv_buffer_rows(recv_rows, num_members), *src_words.shape[1:]),
                src_words.dtype,
            ),
            pltpu.SemaphoreType.DMA((num_waves + 1,)) @ tc_mesh,
            pltpu.SemaphoreType.DMA((1,)) @ sc_mesh,
        ),
        mesh=sc_mesh,
        scratch_types=[pltpu.SMEM(table.shape, I32), pltpu.SemaphoreType.DMA(())],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name="sc_dispatch",
    )(src_words, table)


def _wait_rows(n, wait, limit):
    """Waits ``n`` (dynamic) rows by static power-of-two waits ``wait(b)`` of
    at most min(64, limit) rows."""
    big = 1
    while big * 2 <= min(64, limit):
        big *= 2

    @pl.loop(0, n // big)
    def _(i):
        wait(big)

    b = big // 2
    while b:
        pl.when((n & b) != 0)(functools.partial(wait, b))
        b //= 2


def sc_drain(send_sem, src_words, table, *, axis: str):
    """Waits for every row this member's `sc_dispatch` over mesh axis
    ``axis`` sent (``table``'s count) from ``src_words``, which it keeps alive
    until then. Returns a copy of ``table``: a value for a later op to take,
    so that XLA keeps this kernel and the source rows."""
    sc_mesh = _sc_mesh()
    sem_ref = jax.new_ref(send_sem, memory_space=pltpu.SEMAPHORE @ sc_mesh)

    def body(sem_r, src_hbm, tbl_hbm, out_hbm, tbl_s, tbl_sem):
        pltpu.async_copy(tbl_hbm, tbl_s, tbl_sem).wait()

        def wait(b):
            # A descriptor of b rows: its wait counts b rows' bytes.
            rows = src_hbm.at[pl.ds(0, b)]
            pltpu.make_async_remote_copy(
                rows, rows, sem_r.at[0], sem_r.at[0], device_id={axis: 0}
            ).wait_send()

        _wait_rows(tbl_s[0], wait, src_words.shape[0])
        pltpu.async_copy(tbl_s, out_hbm, tbl_sem).wait()

    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct(table.shape, I32),
        mesh=sc_mesh,
        scratch_types=[pltpu.SMEM(table.shape, I32), pltpu.SemaphoreType.DMA(())],
        name="sc_dispatch_drain",
    )(sem_ref, src_words, table)
