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
"""Placement-aware two-shot all-reduce of a small bf16 [rows, hidden] block.

Two levers over all_reduce's flat two-shot, either or both:

    ordered issue  Every core issues its DMAs slowest latency class first
                   (class = ICI hops and the number of die-0 endpoints, priced
                   by ``LATENCY_US``), round-robin across its chip's first-hop
                   links, from a per-core peer table built at trace time from
                   the devices' coords and core_on_chip. Receivers count each
                   sender on its own semaphore.
    D2D-first      Phase 0: the die-0 core of each chip pushes its block to its
                   die-1 twin over the intra-chip link; the twin sums the two in
                   float32. Phase 1: one-shot reduce-scatter of the float32
                   partials among the C = N / 2 die-1 cores, chunk = hidden / C.
                   Phase 2: each owner rounds its chunk once and pushes it to
                   all N - 1 other cores, its own die 0 included.

The table arrives as data, one int32 row per core in SMEM (``placement_meta``),
with the core's rank in slot 0, as all_reduce takes its rank under the torch
bridge. Sums are float32 in a fixed order and rounded once -- core order for the
flat form; die 0 + die 1, then the chips in owner order, for D2D-first -- and
each owner's chunk is computed once and copied, so every core's output is
bitwise the same.

``drops`` (fault injection and removal arms, timing-only): the meta may mark
DMAs as skipped; the sender does not issue them and the receiver does not wait
for them, so the receiving slot keeps whatever it held.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    wait_bytes_of,
)

COLLECTIVE_ID = 5

# One-way latency of a 4 KiB remote DMA in us: a + b * (hops - 1) + c * die0
# endpoints between chips, same_chip between the two cores of one chip
# (tpu7x 2x2x1 ping-pong, device time, ). Only the
# order it induces matters to the kernel.
LATENCY_US = dict(a=1.706, b=0.467, c=0.446, same_chip=0.892)


class Core(NamedTuple):
    coords: tuple
    die: int  # core_on_chip


def cores_of(devices):
    """The (chip coords, core_on_chip) of each device, in mesh order."""
    return [Core(tuple(int(c) for c in d.coords), int(d.core_on_chip)) for d in devices]


def _hops(cores, i, j):
    dims = [max(c.coords[k] for c in cores) + 1 for k in range(3)]
    h = 0
    for k in range(3):
        dk = abs(cores[i].coords[k] - cores[j].coords[k])
        h += min(dk, dims[k] - dk) if dims[k] > 2 else dk
    return h


def _latency(cores, i, j, latency):
    h = _hops(cores, i, j)
    if h == 0:
        return latency["same_chip"]
    d0 = int(cores[i].die == 0) + int(cores[j].die == 0)
    return latency["a"] + latency["b"] * (h - 1) + latency["c"] * d0


def _first_link(cores, i, j):
    """The dimension of the first ICI hop from i to j under dimension-order
    routing (x, then y, then z); 3 for the intra-chip link."""
    for k in range(3):
        if cores[i].coords[k] != cores[j].coords[k]:
            return k
    return 3


def issue_order(cores, me, peers, latency):
    """``peers`` slowest latency class first; within a class, round-robin over
    first-hop links, continuing from the link the previous DMA left on. Ties
    break on j ^ me, so on a regular layout every rank's order is the same
    XOR pattern of its rank."""
    classes = {}
    for j in sorted(peers, key=lambda j: j ^ me):
        classes.setdefault(_latency(cores, me, j, latency), []).append(j)
    order, last = [], -1
    for lat in sorted(classes, reverse=True):
        by_link = {}
        for j in classes[lat]:
            by_link.setdefault(_first_link(cores, me, j), []).append(j)
        links = sorted(by_link)
        start = next((p for p, link in enumerate(links) if link > last), 0)
        links = links[start:] + links[:start]
        while any(by_link.values()):
            for link in links:
                if by_link[link]:
                    order.append(by_link[link].pop(0))
                    last = link
    return order


def d2d_layout(cores):
    """(twin per core, owners): the die-1 cores in chip order own the chunks."""
    n = len(cores)
    if n % 2:
        raise ValueError(f"{n} cores: D2D-first needs both cores of every chip")
    chip_of = {}
    for i, c in enumerate(cores):
        chip_of.setdefault(c.coords, {})[c.die] = i
    if any(sorted(v) != [0, 1] for v in chip_of.values()):
        raise ValueError("D2D-first needs both cores of every chip in the mesh")
    twin = [chip_of[c.coords][1 - c.die] for c in cores]
    owners = [chip_of[k][1] for k in sorted(chip_of, key=lambda k: chip_of[k][1])]
    return twin, owners


def _d2d_offsets(n):
    """Slot offsets in a D2D meta row, and its width."""
    c = n // 2
    off = dict(rank=0, die1=1, twin=2, slot=3, send0=4, owners=5)
    off["order1"] = off["owners"] + c
    off["order2"] = off["order1"] + c - 1
    off["expect0"] = off["order2"] + n - 1
    off["expect1"] = off["expect0"] + 1
    off["expect2"] = off["expect1"] + c
    off["width"] = off["expect2"] + c
    return off


def meta_width(n, d2d):
    return _d2d_offsets(n)["width"] if d2d else 2 * n


def placement_meta(cores, *, d2d, ordered, latency=LATENCY_US, drop=()):
    """int32 [n, meta_width] rows, row r for the core of mesh index r.

    flat:  [rank, order (n - 1), expect (n)]; order lists the peers in issue
           order, -1 for a dropped DMA; expect[j] = 1 when the core waits for
           core j's data.
    d2d:   [rank, is_die1, twin, slot, send0, owners (C), order1 (C - 1),
            order2 (n - 1), expect0, expect1 (C), expect2 (C)]; slot is the
           owner index of a die-1 core (its twin's for die 0), send0 whether
           die 0 sends its phase-0 block, order1 the chunks a die-1 core sends
           in phase 1, order2 the cores it sends its own chunk to in phase 2.

    ``drop``: (src, dst) mesh-index pairs whose DMAs are skipped, in every
    phase where src sends to dst.
    """
    n = len(cores)
    drop = set(drop)

    def arrange(me, peers):
        return issue_order(cores, me, peers, latency) if ordered else list(peers)

    rows = np.zeros((n, meta_width(n, d2d)), np.int32)
    if not d2d:
        for me in range(n):
            order = arrange(me, [j for j in range(n) if j != me])
            order = [-1 if (me, j) in drop else j for j in order]
            expect = [int(j != me and (j, me) not in drop) for j in range(n)]
            rows[me] = [me, *order, *expect]
        return rows
    twin, owners = d2d_layout(cores)
    c = len(owners)
    slot_of = {o: s for s, o in enumerate(owners)}
    for me in range(n):
        die1 = cores[me].die == 1
        slot = slot_of[me] if die1 else slot_of[twin[me]]
        send0 = int(not die1 and (me, twin[me]) not in drop)
        order1, order2, expect1 = [-1] * (c - 1), [-1] * (n - 1), [0] * c
        if die1:
            peers1 = arrange(me, [o for o in owners if o != me])
            order1 = [-1 if (me, o) in drop else slot_of[o] for o in peers1]
            peers2 = arrange(me, [j for j in range(n) if j != me])
            order2 = [-1 if (me, j) in drop else j for j in peers2]
            expect1 = [int(o != me and (o, me) not in drop) for o in owners]
        expect2 = [int(o != me and (o, me) not in drop) for o in owners]
        expect0 = int(die1 and (twin[me], me) not in drop)
        rows[me] = [
            me,
            int(die1),
            twin[me],
            slot,
            send0,
            *owners,
            *order1,
            *order2,
            expect0,
            *expect1,
            *expect2,
        ]
    return rows


def _flat(x, n, axis, drops):
    rows, hidden = x.shape
    own = hidden // n

    def kernel(meta, x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2):
        me = meta[0]

        def order(k):
            return meta[1 + k]

        def expect(j):
            return meta[n + j] != 0

        def send_to(k, src, dst, send, recv):
            j = order(k)

            def go():
                pltpu.make_async_remote_copy(
                    src_ref=src(j),
                    dst_ref=dst,
                    send_sem=send,
                    recv_sem=recv.at[me],
                    device_id={axis: j},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

            pl.when(j >= 0)(go) if drops else go()

        def wait_from(j, buf, recv):
            def go():
                wait_bytes_of(buf.at[j], recv.at[j])

            pl.when(expect(j))(go) if drops else pl.when(jnp.int32(j) != me)(go)

        def drain(buf, send):
            if drops:
                for k in range(n - 1):
                    pl.when(order(k) >= 0)(lambda: wait_bytes_of(buf.at[0], send))
            else:
                wait_bytes_of(buf.at[pl.ds(0, n - 1)], send)

        barrier(n, axis, True)
        for j in range(n):
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        for k in range(n - 1):  # round 1: owner j's columns into slot `me`
            send_to(k, lambda j: mine.at[j], slices.at[me], send1, recv1)
        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        for j in range(n):
            wait_from(j, slices, recv1)
        acc = slices[0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j].astype(jnp.float32)
        gathered[pl.ds(me, 1)] = acc.astype(gathered.dtype)[None]
        for k in range(n - 1):  # round 2: my summed columns into slot `me`
            send_to(k, lambda j: gathered.at[me], gathered.at[me], send2, recv2)
        for j in range(n):
            wait_from(j, gathered, recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        drain(slices, send1)
        drain(gathered, send2)

    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((n,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((n,)),
    ]
    return kernel, scratch, f"placement_all_reduce_flat_r{rows}"


def _d2d(x, n, axis, drops):
    rows, hidden = x.shape
    c = n // 2
    own = hidden // c
    off = _d2d_offsets(n)

    def kernel(
        meta,
        x_ref,
        o_ref,
        twin_buf,
        mine,
        slices,
        gathered,
        send0,
        recv0,
        send1,
        recv1,
        send2,
        recv2,
    ):
        me, die1, twin, slot = meta[0], meta[1] != 0, meta[2], meta[3]
        die0 = jnp.logical_not(die1)

        def remote(src, dst, send, recv, j):
            pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=send,
                recv_sem=recv,
                device_id={axis: j},
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

        def kept(flag, fn):
            """fn, or fn under the meta flag when the meta may carry drops."""
            pl.when(flag)(fn) if drops else fn()

        barrier(n, axis, True)

        @pl.when(die0)
        def _phase0():  # my block to my twin
            kept(
                meta[off["send0"]] != 0,
                lambda: remote(x_ref, twin_buf, send0, recv0, twin),
            )

        @pl.when(die1)
        def _phases_1_2():
            kept(meta[off["expect0"]] != 0, lambda: wait_bytes_of(twin_buf, recv0))
            p = twin_buf[...].astype(jnp.float32) + x_ref[...].astype(jnp.float32)
            for q in range(c):
                mine[q] = p[:, q * own : (q + 1) * own]
            for k in range(c - 1):  # phase 1: chunk q into slot `slot` of owner q
                q = meta[off["order1"] + k]
                kept(
                    q >= 0,
                    lambda q=q: remote(
                        mine.at[q],
                        slices.at[slot],
                        send1,
                        recv1.at[slot],
                        meta[off["owners"] + q],
                    ),
                )
            slices[pl.ds(slot, 1)] = mine[pl.ds(slot, 1)]
            for s in range(c):
                flag = (
                    (meta[off["expect1"] + s] != 0) if drops else (jnp.int32(s) != slot)
                )
                pl.when(flag)(lambda s=s: wait_bytes_of(slices.at[s], recv1.at[s]))
            acc = slices[0]
            for s in range(1, c):
                acc = acc + slices[s]
            gathered[pl.ds(slot, 1)] = acc.astype(gathered.dtype)[None]
            for k in range(n - 1):  # phase 2: my chunk into slot `slot` everywhere
                j = meta[off["order2"] + k]
                kept(
                    j >= 0,
                    lambda j=j: remote(
                        gathered.at[slot], gathered.at[slot], send2, recv2.at[slot], j
                    ),
                )

        for s in range(c):  # every core: the other owners' chunks
            flag = (
                (meta[off["expect2"] + s] != 0)
                if drops
                else (die0 | (jnp.int32(s) != slot))
            )
            pl.when(flag)(lambda s=s: wait_bytes_of(gathered.at[s], recv2.at[s]))
        for s in range(c):
            o_ref[:, pl.ds(s * own, own)] = gathered[s]

        @pl.when(die0)
        def _drain0():
            kept(meta[off["send0"]] != 0, lambda: wait_bytes_of(x_ref, send0))

        @pl.when(die1)
        def _drain12():
            if drops:
                for k in range(c - 1):
                    pl.when(meta[off["order1"] + k] >= 0)(
                        lambda: wait_bytes_of(slices.at[0], send1)
                    )
                for k in range(n - 1):
                    pl.when(meta[off["order2"] + k] >= 0)(
                        lambda: wait_bytes_of(gathered.at[0], send2)
                    )
            else:
                wait_bytes_of(slices.at[pl.ds(0, c - 1)], send1)
                wait_bytes_of(gathered.at[pl.ds(0, c)], send2)
                wait_bytes_of(gathered.at[pl.ds(0, c - 1)], send2)

    scratch = [
        pltpu.VMEM((rows, hidden), x.dtype),
        pltpu.VMEM((c, rows, own), jnp.float32),
        pltpu.VMEM((c, rows, own), jnp.float32),
        pltpu.VMEM((c, rows, own), x.dtype),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((c,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((c,)),
    ]
    return kernel, scratch, f"placement_all_reduce_d2d_r{rows}"


def small_all_reduce_placement(
    x, meta, *, num_cores, axis, d2d, drops=False, collective_id=COLLECTIVE_ID
):
    """The sum of ``x`` over the ``num_cores`` cores of mesh axis ``axis``,
    inside a shard_map over that axis.

    Args:
      x: [rows, hidden], this core's block; hidden a multiple of
        num_cores * LANES.
      meta: int32 [meta_width(num_cores, d2d)], this core's row of
        ``placement_meta`` (its rank in slot 0), as data.
      d2d: the D2D-first form; else the flat two-shot.
      drops: the meta may mark skipped DMAs (fault injection, removal arms).
    """
    rows, hidden = x.shape
    if hidden % (num_cores * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {num_cores} cores x {LANES} lanes"
        )
    build = _d2d if d2d else _flat
    kernel, scratch, name = build(x, num_cores, axis, drops)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(meta.astype(jnp.int32), x)


# --- D2D-first, direct form ---------------------------------------------------
#
# The same three phases with no staging copies and fewer semaphores: x and the
# output stay where XLA put them (pl.ANY); die 0 DMAs x straight to its twin,
# phase 1 DMAs column windows of the chip partial, phase 2 DMAs each owner's
# reduced chunk straight into every core's output. Phase 1 has one byte-counted
# receive semaphore in mesh order, and per-sender semaphores waited fastest
# sender first (accumulating as they land) when ordered; phase 2 has one
# byte-counted semaphore. The barrier is asymmetric: die 0 signals the C owners
# and waits for its twin only, die 1 signals its twin and the other owners and
# waits for all N - 1 signals that target it, so a die-0 core starts phase 0 as
# soon as its twin has entered the kernel. One barrier per call is still
# enough: a core counts only signals of the call it is in, because no core can
# signal call k + 1 before finishing call k, a die-0 core finishes call k only
# after phase 2 of every owner, and an owner reaches phase 2 only after every
# die-1 core passed barrier k (phase 1) and its twin passed it (phase 0).
#
# By default the chip partial is rounded to x.dtype once after phase 0, halving
# phase-1 bytes at the cost of one extra rounding per chip partial (outputs are
# no longer within the flat form's tolerance; still bitwise equal on every
# core). exact_partials keeps them float32: one rounding, as the flat form, but
# float32 sums in chip-pair order, so not bitwise the flat form's in general.


def _direct_offsets(n):
    c = n // 2
    off = dict(rank=0, die1=1, twin=2, slot=3, send0=4, expect0=5, owners=6)
    off["order1"] = off["owners"] + c
    off["wait1"] = off["order1"] + c - 1
    off["order2"] = off["wait1"] + c - 1
    off["width"] = off["order2"] + n - 1
    return off


def d2d_direct_meta(cores, *, ordered, latency=LATENCY_US, skip_phase0=()):
    """int32 [n, W] rows: [rank, die1, twin, slot, send0, expect0, owners (C),
    order1 (C - 1), wait1 (C - 1), order2 (n - 1)]. order1: the chunks a die-1
    core sends in phase 1 (issue order); wait1: the senders' slots in the order
    their chunks are awaited (fastest first when ordered); order2: the cores it
    sends its reduced chunk to. ``skip_phase0``: chip slots whose phase-0 DMA
    is skipped (removal arm, positive control)."""
    n = len(cores)
    twin, owners = d2d_layout(cores)
    c = len(owners)
    slot_of = {o: s for s, o in enumerate(owners)}
    rows = np.zeros((n, _direct_offsets(n)["width"]), np.int32)
    for me in range(n):
        die1 = cores[me].die == 1
        slot = slot_of[me] if die1 else slot_of[twin[me]]
        order1, wait1, order2 = [0] * (c - 1), [0] * (c - 1), [0] * (n - 1)
        if die1:
            peers1 = [o for o in owners if o != me]
            peers2 = [j for j in range(n) if j != me]
            if ordered:
                issue1 = issue_order(cores, me, peers1, latency)
                arrive1 = sorted(
                    peers1, key=lambda o: (_latency(cores, o, me, latency), o ^ me)
                )
                peers2 = issue_order(cores, me, peers2, latency)
            else:
                issue1 = arrive1 = peers1
            order1 = [slot_of[o] for o in issue1]
            wait1 = [slot_of[o] for o in arrive1]
            order2 = peers2
        rows[me] = [
            me,
            int(die1),
            twin[me],
            slot,
            int(not die1 and slot not in skip_phase0),
            int(die1 and slot not in skip_phase0),
            *owners,
            *order1,
            *wait1,
            *order2,
        ]
    return rows


def d2d_direct_patterns(meta):
    """The direct form's table as rank-relative XOR patterns (order1 and wait1
    on the slot, order2 on the rank), for a kernel that takes only its rank;
    ValueError unless the layout is regular (twin = r ^ 1, die = r & 1, owner
    of chunk q = 2q + 1) and every die-1 row has the same patterns."""
    n = meta.shape[0]
    off = _direct_offsets(n)
    c = n // 2
    pats = None
    for r in range(n):
        row = [int(v) for v in meta[r]]
        if (
            row[off["twin"]] != r ^ 1
            or row[off["die1"]] != (r & 1)
            or (row[off["owners"] : off["order1"]] != [2 * q + 1 for q in range(c)])
        ):
            raise ValueError(f"rank {r}: irregular core layout")
        if not r & 1:
            continue
        slot = row[off["slot"]]
        p = (
            tuple(q ^ slot for q in row[off["order1"] : off["wait1"]]),
            tuple(s ^ slot for s in row[off["wait1"] : off["order2"]]),
            tuple(j ^ r for j in row[off["order2"] : off["width"]]),
        )
        if pats not in (None, p):
            raise ValueError(f"rank {r}: order is not a rank-relative XOR pattern")
        pats = p
    return pats


def _d2d_direct(x, n, axis, *, ordered, exact, sym_barrier, patterns, skip_phase0):
    rows, hidden = x.shape
    c = n // 2
    own = hidden // c
    pdt = jnp.float32 if exact else x.dtype
    off = _direct_offsets(n)

    def columns(ref, q):
        return ref.at[:, pl.ds(pl.multiple_of(q * own, LANES), own)]

    def kernel(
        table,
        x_any,
        o_any,
        twin_buf,
        xbuf,
        partial,
        slices,
        red,
        send0,
        recv0,
        lsem,
        send1,
        rs_recv,
        send2,
        ag_recv,
        osem,
    ):
        me = table[0]
        if patterns is None:
            die1 = table[off["die1"]] != 0
            twin, slot = table[off["twin"]], table[off["slot"]]
            send0_on = table[off["send0"]] != 0
            expect0_on = table[off["expect0"]] != 0

            def owner(q):
                return table[off["owners"] + q]

            def order1(k):
                return table[off["order1"] + k]

            def wait1(k):
                return table[off["wait1"] + k]

            def order2(k):
                return table[off["order2"] + k]
        else:
            p1, w1, p2 = patterns
            die1 = (me & 1) == 1
            twin, slot = me ^ 1, me >> 1
            kept = jnp.bool_(True)
            for s in skip_phase0:
                kept = kept & (slot != s)
            send0_on = jnp.logical_not(die1) & kept
            expect0_on = die1 & kept

            def owner(q):
                return 2 * q + 1

            def order1(k):
                return slot ^ p1[k]

            def wait1(k):
                return slot ^ w1[k]

            def order2(k):
                return me ^ p2[k]

        die0 = jnp.logical_not(die1)

        def remote(src, dst, send, rsem, j):
            pltpu.make_async_remote_copy(
                src_ref=src,
                dst_ref=dst,
                send_sem=send,
                recv_sem=rsem,
                device_id={axis: j},
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

        def signal(sem, j):
            pl.semaphore_signal(
                sem, 1, device_id={axis: j}, device_id_type=pl.DeviceIdType.MESH
            )

        if sym_barrier:
            barrier(n, axis, True)
        else:
            bsem = pltpu.get_barrier_semaphore()

            @pl.when(die0)
            def _():
                for q in range(c):
                    signal(bsem, owner(q))
                pl.semaphore_wait(bsem, 1)

            @pl.when(die1)
            def _():
                signal(bsem, twin)
                for q in range(c):
                    pl.when(jnp.int32(q) != slot)(lambda q=q: signal(bsem, owner(q)))
                pl.semaphore_wait(bsem, n - 1)

        @pl.when(die0 & send0_on)
        def _phase0():  # my block straight to my twin
            remote(x_any, twin_buf, send0, recv0, twin)

        @pl.when(die1)
        def _phases_1_2():
            mine = pltpu.make_async_copy(x_any, xbuf, lsem)
            mine.start()
            mine.wait()
            pl.when(expect0_on)(lambda: wait_bytes_of(twin_buf, recv0))
            partial[...] = (
                twin_buf[...].astype(jnp.float32) + xbuf[...].astype(jnp.float32)
            ).astype(pdt)
            for k in range(c - 1):  # phase 1: chunk q into slot `slot` of owner q
                q = order1(k)
                remote(
                    columns(partial, q),
                    slices.at[slot],
                    send1,
                    rs_recv.at[slot] if ordered else rs_recv,
                    owner(q),
                )
            keep = pltpu.make_async_copy(columns(partial, slot), slices.at[slot], lsem)
            keep.start()
            keep.wait()
            if ordered:  # accumulate as the chunks land, fastest sender first
                acc = slices[slot].astype(jnp.float32)
                for k in range(c - 1):
                    s = wait1(k)
                    wait_bytes_of(slices.at[s], rs_recv.at[s])
                    acc = acc + slices[s].astype(jnp.float32)
            else:  # chips in owner order
                wait_bytes_of(slices.at[pl.ds(0, c - 1)], rs_recv)
                acc = slices[0].astype(jnp.float32)
                for s in range(1, c):
                    acc = acc + slices[s].astype(jnp.float32)
            red[...] = acc.astype(red.dtype)
            pltpu.make_async_copy(red, columns(o_any, slot), osem).start()
            for k in range(n - 1):  # phase 2: my chunk into every core's output
                remote(red, columns(o_any, slot), send2, ag_recv, order2(k))

        def chunks(m):  # a view of m [rows, own] chunks of x.dtype, for byte counts
            return twin_buf.at[:, pl.ds(0, m * own)]

        @pl.when(die0)
        def _():
            wait_bytes_of(chunks(c), ag_recv)
            pl.when(send0_on)(lambda: wait_bytes_of(twin_buf, send0))

        @pl.when(die1)
        def _():
            wait_bytes_of(chunks(c - 1), ag_recv)
            wait_bytes_of(red, osem)
            wait_bytes_of(slices.at[pl.ds(0, c - 1)], send1)
            wait_bytes_of(chunks(c), send2)
            wait_bytes_of(chunks(c - 1), send2)

    scratch = [
        pltpu.VMEM((rows, hidden), x.dtype),  # twin_buf
        pltpu.VMEM((rows, hidden), x.dtype),  # xbuf
        pltpu.VMEM((rows, hidden), pdt),  # partial
        pltpu.VMEM((c, rows, own), pdt),  # slices
        pltpu.VMEM((rows, own), x.dtype),  # red
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((c,)) if ordered else pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]
    return kernel, scratch, f"placement_all_reduce_d2dz_r{rows}"


def small_all_reduce_d2d_direct(
    x,
    table,
    *,
    num_cores,
    axis,
    ordered,
    exact_partials=False,
    symmetric_barrier=False,
    patterns=None,
    skip_phase0=(),
    collective_id=COLLECTIVE_ID,
):
    """D2D-first, direct form (see above), inside a shard_map over ``axis``.

    Args:
      x: [rows, hidden], this core's block; hidden a multiple of
        num_cores / 2 * LANES.
      table: this core's row of ``d2d_direct_meta`` (int32, SMEM), or with
        ``patterns`` (``d2d_direct_patterns`` of that meta) its rank alone
        (int32 [1]), the order then compiled in.
      ordered: the meta's order is slowest class first (per-sender phase-1
        semaphores, awaited fastest first); else one phase-1 semaphore.
      exact_partials, symmetric_barrier: float32 chip partials, the full
        barrier (probe knobs, see above).
      skip_phase0: with ``patterns``, chip slots whose phase 0 is skipped
        (timing-only); with a meta row, put them in the meta instead.
    """
    rows, hidden = x.shape
    if hidden % (num_cores // 2 * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {num_cores // 2} chips x {LANES} lanes"  # noqa: E501
        )
    kernel, scratch, name = _d2d_direct(
        x,
        num_cores,
        axis,
        ordered=ordered,
        exact=exact_partials,
        sym_barrier=symmetric_barrier,
        patterns=patterns,
        skip_phase0=tuple(skip_phase0),
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(table.astype(jnp.int32), x)


# --- flat two-shot, issue order compiled in -------------------------------------
#
# A multi-word SMEM operand is copied into SMEM on every call (0.53 us per call
# on tpu7x 2x2x1, ), as all_reduce's note on its rank
# operand warns, and per-sender semaphores cost another 0.14 us. This form
# takes the rank alone (int32 [1], as all_reduce) and compiles the issue order
# in as two rank-relative XOR patterns, one per die: peer k of rank r is
# r ^ pattern[r & 1][k]. Receives count on one semaphore per round, as
# all_reduce. Same sums as all_reduce's two-shot, so outputs are bitwise its.


def ordered_patterns(cores, latency=LATENCY_US, reverse=False):
    """(pattern for die 0, pattern for die 1) of ``issue_order`` (fastest class
    first with ``reverse``, a falsification arm); ValueError unless every
    rank's order is r ^ pattern[die(r)] with die(r) = r & 1."""
    n = len(cores)
    pats = {}
    for r in range(n):
        order = issue_order(cores, r, [j for j in range(n) if j != r], latency)
        if reverse:
            order = order[::-1]
        p = tuple(j ^ r for j in order)
        if cores[r].die != (r & 1) or pats.setdefault(cores[r].die, p) != p:
            raise ValueError(
                f"rank {r}: issue order is not a rank-relative XOR pattern"
            )
    return pats[0], pats[1]


def small_all_reduce_ordered(
    x, rank, *, num_cores, axis, patterns, collective_id=COLLECTIVE_ID
):
    """all_reduce's two-shot with every core issuing its DMAs in the order
    ``patterns`` (``ordered_patterns`` of the mesh's cores) gives, inside a
    shard_map over ``axis``; ``rank`` is int32 [1], this core's index."""
    rows, hidden = x.shape
    n = num_cores
    if hidden % (n * LANES):
        raise ValueError(
            f"hidden {hidden} is not a multiple of {n} cores x {LANES} lanes"
        )
    own = hidden // n
    p0, p1 = patterns

    def kernel(
        rank_ref, x_ref, o_ref, mine, slices, gathered, send1, recv1, send2, recv2
    ):
        me = rank_ref[0]  # noqa: F841
        die1 = (me & 1) == 1

        def peer(k):
            return jnp.where(die1, me ^ p1[k], me ^ p0[k])

        def send_round(src, dst, send, recv):
            for k in range(n - 1):
                j = peer(k)
                pltpu.make_async_remote_copy(
                    src_ref=src(j),
                    dst_ref=dst,
                    send_sem=send,
                    recv_sem=recv,
                    device_id={axis: j},
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        barrier(n, axis, True)
        for j in range(n):  # slot j: the columns core j owns
            mine[j] = x_ref[:, pl.ds(j * own, own)]
        send_round(lambda j: mine.at[j], slices.at[me], send1, recv1)
        slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
        acc = slices[0].astype(jnp.float32)
        for j in range(1, n):
            acc = acc + slices[j].astype(jnp.float32)
        gathered[pl.ds(me, 1)] = acc.astype(gathered.dtype)[None]
        send_round(lambda j: gathered.at[me], gathered.at[me], send2, recv2)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)
        for j in range(n):
            o_ref[:, pl.ds(j * own, own)] = gathered[j]
        wait_bytes_of(slices.at[pl.ds(0, n - 1)], send1)
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], send2)

    scratch = [
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        pltpu.VMEM((n, rows, own), x.dtype),
        *[pltpu.SemaphoreType.DMA] * 4,
    ]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec()],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"placement_all_reduce_ordered_r{rows}",
    )(rank.astype(jnp.int32), x)
