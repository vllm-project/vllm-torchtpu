# SPDX-License-Identifier: Apache-2.0
"""Fused EP16/TP2 prefill MoE for 512- through 8192-token windows.

Schedule:
  1. Router/latent/shared projections overlap the incremental Top16, route sort,
     route exchange, packed latent ring and shared-output reduction.
  2. Compact routing metadata moves to SMEM. GU/SiTU and the first down column
     share a pipeline; later down columns overlap token combine and reduction.
  3. The reduced VMEM shard is normalized, projected up and added to the shared
     expert and residual outputs.

The factory at the end is the only production entry point. The small-window, 4K
and 8K schedules are selected by window size; 8K skew overflow retains every route via
an uncapped fallback. Python generators split work at tracing time, allowing
independent vector/communication work between MXU panels without extra calls.
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from .. import prefill_megakernel as k

_RMW_BATCH = 2
_RMW_DUMMY_SLOTS = 16
_LOCAL_EXPERTS = 56  # EP16/TP2: a chip's two ranks share 56 experts
_MIDDLE = 1536  # each rank's half of an expert's intermediate width
# Local routes a launch keeps in SMEM (expert-major token ids and weights):
# all of them up to 4K tokens; from 8K, a launch in which any EP group holds
# more takes the uncapped fallback on every rank.
_COMPACT_ROUTES = 65536


def _route_capacity(tokens):
    """The compact SMEM route capacity of one ``tokens`` launch."""
    return min(tokens * 16, _COMPACT_ROUTES)


def _sort_route_steps(keys, probabilities, token_ids=None, *, size, bits=0):
    """Yield after each 4096-route window of the in-place bitonic network.

    Keys encode (local expert, original route position), so all live keys are
    unique and the result has deterministic expert-major/token-major order.
    Probability values follow exactly the same swaps as their keys.
    """
    window, rows = 4096, 32
    assert size >= 2 * window and size & (size - 1) == 0
    shape = (rows, 128)
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    positions = row * 128 + lane
    mask = (1 << bits) - 1

    def view(ref, ordinal):
        return ref.at[pl.ds(ordinal * rows, rows), :]

    def partner(value, distance):
        if distance >= 1024:
            # Reorder register-sized chunks without a gather-index constant.
            return jnp.concatenate(
                [
                    value[
                        8 * (i ^ (distance // 1024)) : 8 * (i ^ (distance // 1024)) + 8
                    ]
                    for i in range(rows // 8)
                ],
                axis=0,
            )
        if distance >= 128:
            shape3 = (rows // 8, 8, 128)
            index = jax.lax.broadcasted_iota(jnp.int32, shape3, 1)
            return jnp.take_along_axis(
                value.reshape(shape3),
                index ^ (distance // 128),
                axis=1,
                mode="promise_in_bounds",
            ).reshape(shape)
        return jnp.take_along_axis(
            value, lane ^ distance, axis=1, mode="promise_in_bounds"
        )

    def apply_in_window(x, p, ordinal, steps):
        global_position = positions + ordinal * window
        for distance, group in steps:
            y, q = partner(x, distance), partner(p, distance)
            descending = (global_position & group) != 0
            upper = (global_position & distance) != 0
            take_min = ~(descending ^ upper)
            keep = ~((x < y) ^ take_min)
            x, p = jnp.where(keep, x, y), jnp.where(keep, p, q)
        return x, p

    pending = []

    def flush():
        if not pending:
            return
        for ordinal in range(size // window):
            kr, pr = view(keys, ordinal), view(probabilities, ordinal)
            x, p = apply_in_window(kr[...], pr[...], ordinal, pending)
            kr[...], pr[...] = x, p
            yield
        pending.clear()

    for stage in range(1, size.bit_length()):
        group = 1 << stage
        for shift in reversed(range(stage)):
            distance = 1 << shift
            if distance < window:
                pending.append((distance, group))
                continue
            yield from flush()
            if distance == window:
                tail_steps = [(1 << s, group) for s in reversed(range(shift))]
                is_final = (group == size) and (token_ids is not None)
                for ordinal in range(size // (2 * window)):
                    low, high = 2 * ordinal, 2 * ordinal + 1
                    kl, kh = view(keys, low), view(keys, high)
                    plow, phigh = view(probabilities, low), view(probabilities, high)
                    x, y, p, q = kl[...], kh[...], plow[...], phigh[...]
                    ascending = (low * window & group) == 0
                    keep = ~((x < y) ^ ascending)
                    xl, xh = jnp.where(keep, x, y), jnp.where(keep, y, x)
                    pl_v, ph_v = jnp.where(keep, p, q), jnp.where(keep, q, p)
                    xl, pl_v = apply_in_window(xl, pl_v, low, tail_steps)
                    if is_final:
                        view(token_ids, low)[...] = (xl & mask) // 16
                    else:
                        kl[...] = xl
                    plow[...] = pl_v
                    yield
                    xh, ph_v = apply_in_window(xh, ph_v, high, tail_steps)
                    if is_final:
                        view(token_ids, high)[...] = (xh & mask) // 16
                    else:
                        kh[...] = xh
                    phigh[...] = ph_v
                    yield
                break
            stride = distance // window
            for ordinal in range(size // (2 * window)):
                low = ordinal // stride * (2 * stride) + ordinal % stride
                high = low + stride
                kl, kh = view(keys, low), view(keys, high)
                plow, phigh = view(probabilities, low), view(probabilities, high)
                x, y, p, q = kl[...], kh[...], plow[...], phigh[...]
                ascending = (low * window & group) == 0
                keep = ~((x < y) ^ ascending)
                kl[...], kh[...] = jnp.where(keep, x, y), jnp.where(keep, y, x)
                plow[...], phigh[...] = jnp.where(keep, p, q), jnp.where(keep, q, p)
                yield
    yield from flush()


class _Top16:
    """Incremental biased sigmoid Top16; ties choose the lowest expert ID."""

    def __init__(self, logits, bias):
        self.raw = 1.0 / (1.0 + jnp.exp(-logits))
        self.scores = self.raw + bias
        self.experts = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        self.selected: list[jax.Array] = []
        self.probabilities: list[jax.Array] = []
        self.index = 0

    def step(self):
        best = jnp.max(self.scores, axis=1, keepdims=True)
        winner = jnp.min(
            jnp.where(self.scores == best, self.experts, self.scores.shape[1]),
            axis=1,
            keepdims=True,
        )
        chosen = self.experts == winner
        probability = jnp.sum(jnp.where(chosen, self.raw, 0.0), axis=1, keepdims=True)
        self.selected.append(winner)
        self.probabilities.append(probability)
        self.scores = jnp.where(chosen, -jnp.inf, self.scores)
        self.index += 1

    def finish(self, valid):
        while self.index < 16:
            self.step()
        selected = jnp.concatenate(self.selected, axis=1)
        probabilities = jnp.concatenate(self.probabilities, axis=1)
        denominator = jnp.maximum(
            jnp.sum(probabilities, axis=1, keepdims=True), 1e-30
        )
        return selected, jnp.where(valid, probabilities / denominator, 0.0)


def _metadata_steps(
    wire,
    keys,
    probabilities,
    token_ids,
    counts,
    maximum,
    *,
    tokens,
    rank,
    counts_ready,
    local_experts=56,
):
    """Build stable local routing metadata, yielding one prep/sort chunk at a time.

    The all-gathered wire contains 16 expert IDs followed by 16 probability bit
    patterns per token. Every rank counts all expert groups at 8K, making the
    compact-capacity decision uniform before any conditional collective runs.
    """
    ep_groups = 896 // local_experts
    expert_tp = 32 // ep_groups
    bits = (tokens * 16 - 1).bit_length()
    shape = (128, 16) if tokens < 4096 else (16, 128)
    expert_index = jnp.arange(local_experts, dtype=jnp.int32)
    group_index = jnp.arange(ep_groups, dtype=jnp.int32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    total_counts = jnp.zeros((local_experts,), jnp.int32)
    total_groups = jnp.zeros((ep_groups,), jnp.int32)
    for chunk in range(tokens // 128):
        if tokens < 4096:
            records = wire[pl.ds(chunk * 32, 32), :].reshape(128, 32)
            ids = records[:, :16]
            probs = jax.lax.bitcast_convert_type(records[:, 16:], jnp.float32)
        else:
            # Keep IDs and probabilities in separate planes so 128-token
            # chunks need no interleaved-record reshape during metadata prep.
            chunks_per_rank = tokens // 4096
            rank_base = (chunk // chunks_per_rank) * (tokens // 128)
            chunk_base = (chunk % chunks_per_rank) * 16
            ids = wire[pl.ds(rank_base + chunk_base, 16), :]
            probs = jax.lax.bitcast_convert_type(
                wire[pl.ds(rank_base + tokens // 256 + chunk_base, 16), :],
                jnp.float32,
            )
        local = ids - (rank // expert_tp * local_experts)
        valid = (probs != 0) & (local >= 0) & (local < local_experts)
        local = jnp.where(valid, local, local_experts)
        original_position = chunk * 2048 + rows * shape[1] + lanes
        packed = (local << bits) | original_position
        keys[pl.ds(chunk * 16, 16), :] = packed.reshape(16, 128)
        probabilities[pl.ds(chunk * 16, 16), :] = probs.reshape(16, 128)
        # Count routes on the packed 16x128 layout.
        local_flat = local.reshape(16, 128)
        tile_counts = jnp.zeros((local_experts,), jnp.int32)
        for expert in range(local_experts):
            count = jnp.sum((local_flat == expert).astype(jnp.int32))
            tile_counts = jnp.where(expert_index == expert, count, tile_counts)
        total_counts += tile_counts
        if maximum is not None:
            ids_flat = ids.reshape(16, 128)
            live_flat = probs.reshape(16, 128) != 0
            tile_groups = jnp.zeros((ep_groups,), jnp.int32)
            groups = ids_flat // local_experts
            for group in range(ep_groups):
                count = jnp.sum((live_flat & (groups == group)).astype(jnp.int32))
                tile_groups = jnp.where(group_index == group, count, tile_groups)
            total_groups += tile_groups
        yield
    counts[...] = total_counts
    if maximum is not None:
        maximum[...] = jnp.max(total_groups)[None]
    counts_ready()
    yield from _sort_route_steps(
        keys, probabilities, token_ids, size=tokens * 16, bits=bits
    )


def _dense_panels(lhs, weights, buffers, semaphores, *, columns):
    """Stream 256-column BF16 panels, with the next weight DMA in flight.

    The yielded FP32 product is deliberately not stored here. The caller places
    independent routing/vector/communication work before consuming it so the
    compiler can schedule that work alongside the MXU product.
    """

    def source(panel):
        return weights.at[:, pl.ds(panel * 256, 256)]

    pltpu.make_async_copy(source(0), buffers[0], semaphores[0]).start()
    for panel in range(columns // 256):
        bank = panel % 2
        pltpu.make_async_copy(source(panel), buffers[bank], semaphores[bank]).wait()
        if panel + 1 < columns // 256:
            pltpu.make_async_copy(
                source(panel + 1), buffers[1 - bank], semaphores[1 - bank]
            ).start()
        yield (
            panel * 256,
            jnp.matmul(lhs, buffers[bank][...], preferred_element_type=jnp.float32),
        )


def _latent_half_pair_panels(lhs, weights, buffers, semaphores):
    """Pair matching 256-column fragments of the two 1792-wide halves."""

    def source_column(panel):
        return (panel // 2) * 256 + (panel % 2) * 1792

    def copy(panel):
        return pltpu.make_async_copy(
            weights.at[:, pl.ds(source_column(panel), 256)],
            buffers[panel % 4],
            semaphores[panel % 4],
        )

    for panel in range(2):
        copy(panel).start()
    for panel in range(0, 14, 2):
        for j in range(2):
            copy(panel + j).wait()
        for j in range(2):
            if panel + 2 + j < 14:
                copy(panel + 2 + j).start()
        products = [
            jnp.matmul(
                lhs, buffers[(panel + j) % 4][...], preferred_element_type=jnp.float32
            )
            for j in range(2)
        ]
        for j, product in enumerate(products):
            yield source_column(panel + j), product


def _route_gather_steps(payload, output, send, recv, *, rank):
    """Five recursive-doubling VMEM exchanges; one completion counter per round.

    Output ranges are immutable once received. A call-entry collective barrier
    is required, and callers must drain this generator before consuming output.
    """
    rows = payload.shape[0]
    output[pl.ds(rank * rows, rows), :] = payload[...]
    for stage in range(5):
        peers = 1 << stage
        peer = rank ^ peers
        origin = rank // peers * (peers * rows)
        incoming = peer // peers * (peers * rows)
        source = output.at[pl.ds(origin, peers * rows), :]
        destination = output.at[pl.ds(origin, peers * rows), :]
        received = output.at[pl.ds(incoming, peers * rows), :]
        transfer = pltpu.make_async_remote_copy(
            source,
            destination,
            send.at[stage],
            recv.at[stage],
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        transfer.start()
        yield stage
        transfer.wait_send()
        # The local receive address differs from the destination on our peer.
        pltpu.make_async_remote_copy(
            received,
            received,
            send.at[stage],
            recv.at[stage],
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).wait_recv()


def _enter_mesh():
    """Match collective lifetimes before any peer can write another's scratch."""
    barrier = pltpu.get_barrier_semaphore()
    for peer in range(32):
        pl.semaphore_signal(
            barrier, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
        )
    pl.semaphore_wait(barrier, 32)


def _latent_gather_steps(
    tables, payload, received, output, send, recv, credits, stores, *, rank
):
    """Twelve directional streams, with two receive slots per stream.

    Windows up to 4K forward through VMEM, copying the completed receive slot into payload.
    8K forwards immutable final HBM ranges to avoid that copy. The latter still
    returns credits before reusing receive counters; immutable data alone does
    not make reusing an unconsumed semaphore safe.
    """
    rows = payload.shape[0]
    assert rows in (16, 32, 64, 128, 256)
    regions = []
    begin = 0
    for stream in range(12):
        size = rows // 12 + (stream < rows % 12)
        family = stream // 2 % 3
        direction = -1 if stream % 2 == 0 else 1
        position = tables[family, 0, rank]
        target = tables[family, 1, (position + 32 + direction) % 32]
        source_peer = tables[family, 1, (position + 32 - direction) % 32]
        regions.append((begin, size, family, direction, position, target, source_peer))
        begin += size

    def destination(region, step):
        begin, size, family, direction, position, _, _ = region
        owner = tables[family, 1, (position + 64 - direction * step) % 32]
        return output.at[pl.ds(owner * rows + begin, size), :, :]

    def remote(source, target, ss, rs, peer):
        return pltpu.make_async_remote_copy(
            source,
            target,
            ss,
            rs,
            device_id=peer,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )

    if rows <= 128:
        for step in range(32):
            for stream, region in enumerate(regions):
                begin, size, _, _, _, peer, source_peer = region
                piece = payload.at[pl.ds(begin, size), :, :]
                ss, credit, st = send.at[stream], credits.at[stream], stores.at[stream]
                if step:
                    old = received.at[(step - 1) % 2, pl.ds(begin, size), :, :]
                    transfer = remote(
                        piece, old, ss, recv.at[(step - 1) % 2, stream], peer
                    )
                    transfer.wait_send()
                    transfer.wait_recv()
                    pltpu.make_async_copy(
                        piece, destination(region, step - 1), st
                    ).wait()
                    piece[...] = old[...]
                    pl.semaphore_signal(
                        credit,
                        1,
                        device_id=source_peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                dest = destination(region, step)
                store = pltpu.make_async_copy(piece, dest, st)
                store.start()
                if step < 31:
                    if step >= 2:
                        pl.semaphore_wait(credit, 1)
                    future = received.at[step % 2, pl.ds(begin, size), :, :]
                    remote(piece, future, ss, recv.at[step % 2, stream], peer).start()
                else:
                    store.wait()
                    pl.semaphore_wait(credit, 2)
            yield
        return

    own = output.at[pl.ds(rank * rows, rows), :, :]
    seed = pltpu.make_async_copy(payload, own, stores)
    seed.start()
    pending: list[Any] = [None] * 12
    for step in range(32):
        for stream, region in enumerate(regions):
            begin, size, _, _, _, peer, source_peer = region
            if pending[stream] is not None:
                transfer, incoming, completed_step = pending[stream]
                transfer.wait_send()
                remote(
                    incoming,
                    incoming,
                    send.at[stream],
                    recv.at[completed_step % 2, stream],
                    peer,
                ).wait_recv()
                if completed_step + 2 < 31:
                    pl.semaphore_signal(
                        credits.at[stream],
                        1,
                        device_id=source_peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
            if step < 31:
                source = (
                    payload.at[pl.ds(begin, size), :, :]
                    if step == 0
                    else destination(region, step)
                )
                dest = destination(region, step)
                incoming = destination(region, step + 1)
                if step >= 2:
                    pl.semaphore_wait(credits.at[stream], 1)
                transfer = remote(
                    source, dest, send.at[stream], recv.at[step % 2, stream], peer
                )
                transfer.start()
                pending[stream] = transfer, incoming, step
        yield
    seed.wait()


def _latent_ready_init(
    tables, payload, output, send, recv, stores, state, meta, *, rank, stride, offset
):
    """Seed twelve independently progressing streams of a 256-row latent gather.

    Rank r's rows land at ``r * stride + offset`` of ``output`` (``stride``
    rows per rank: 256, or 512 for a 16K window's two passes). Cache neighbor
    IDs and owner row offsets in SMEM once. Each stream tracks its hop and
    receive/forward/complete phase; state[0] counts active streams.
    """
    rows = payload.shape[0]
    assert rows == 256
    for directed in range(6):
        family, direction = directed // 2, -1 if directed % 2 == 0 else 1
        position = tables[family, 0, rank]
        meta[directed * 34] = tables[family, 1, (position + direction) & 31]
        meta[directed * 34 + 1] = tables[family, 1, (position - direction) & 31]

        @pl.loop(0, 32)
        def owners(step, *, family=family, position=position, direction=direction, directed=directed):
            owner = tables[family, 1, (position - direction * step) & 31]
            meta[directed * 34 + 2 + step] = owner * stride + offset

    own = pl.ds(rank * stride + offset, rows)
    pltpu.make_async_copy(payload, output.at[own, :, :], stores).start()
    state[0] = 12
    begin = 0
    for stream in range(12):
        size = rows // 12 + (stream < rows % 12)
        part = pl.ds(begin, size)
        pltpu.make_async_remote_copy(
            payload.at[part, :, :],
            output.at[pl.ds(rank * stride + offset + begin, size), :, :],
            send.at[stream],
            recv.at[0, stream],
            device_id=meta[(stream % 6) * 34],
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).start()
        state[1 + stream] = 1
        state[13 + stream] = 0
        begin += size


def _latent_ready_progress(
    payload, output, send, recv, credits, stores, state, meta, *, rank, stride,
    offset, drain=False,
):
    """Advance ready streams without blocking unrelated compute or streams.

    A receive consumes both DMA completion counts. Credits protect the two
    alternating receive semaphores from reuse before the peer consumes them.
    The final drain completes every stream before routed experts read HBM.
    """
    rows = payload.shape[0]

    def sweep():
        begin = 0
        for stream in range(12):
            size = rows // 12 + (stream < rows % 12)
            start = begin
            begin += size
            base = (stream % 6) * 34
            # TPUv7 DMA semaphore accounting: one unit per 32 logical bytes.
            units = size * 2 * 1792 * 2 // 32
            step = state[1 + stream]

            @pl.when(state[13 + stream] == 0)
            def receive(*, step=step, stream=stream, units=units, base=base, start=start, size=size):
                bank = (step - 1) & 1
                ready = (pl.semaphore_read(send.at[stream]) >= units) & (
                    pl.semaphore_read(recv.at[bank, stream]) >= units
                )

                @pl.when(ready)
                def consume():
                    incoming = output.at[pl.ds(meta[base + 2 + step] + start, size), :, :]
                    # wait_send only consumes its completion count; the HBM view
                    # has the same logical transfer size as the initial VMEM source.
                    transfer = pltpu.make_async_remote_copy(
                        incoming,
                        incoming,
                        send.at[stream],
                        recv.at[bank, stream],
                        device_id=meta[base],
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    transfer.wait_send()
                    transfer.wait_recv()

                    @pl.when(step < 30)
                    def credit():
                        pl.semaphore_signal(
                            credits.at[stream], 1, device_id=meta[base + 1], device_id_type=pl.DeviceIdType.LOGICAL
                        )

                    state[13 + stream] = jnp.where(step == 31, 2, 1)

                    @pl.when(step == 31)
                    def complete():
                        state[0] = state[0] - 1

            @pl.when(state[13 + stream] == 1)
            def forward(*, step=step, stream=stream, base=base, start=start, size=size):
                available = (step < 2) | (pl.semaphore_read(credits.at[stream]) >= 1)

                @pl.when(available)
                def issue():
                    @pl.when(step >= 2)
                    def acquire():
                        pl.semaphore_wait(credits.at[stream], 1)

                    source = output.at[pl.ds(meta[base + 2 + step] + start, size), :, :]
                    pltpu.make_async_remote_copy(
                        source,
                        source,
                        send.at[stream],
                        recv.at[step & 1, stream],
                        device_id=meta[base],
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    ).start()
                    state[1 + stream] = step + 1
                    state[13 + stream] = 0

    if drain:

        def body(i):
            sweep()
            return i + 1

        jax.lax.while_loop(lambda i: state[0] != 0, body, jnp.int32(0))
        own = pl.ds(rank * stride + offset, rows)
        pltpu.make_async_copy(payload, output.at[own, :, :], stores).wait()
    else:

        @pl.when(state[0] != 0)
        def progress():
            sweep()


def _stripe_validity(rows, length, rank):
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    stripe = min(rows, 64)
    return (row // stripe * (32 * stripe) + rank * stripe + row % stripe) < length


def _prologue(
    tables,
    control,
    sources,
    router,
    bias,
    latent_weight,
    shared_gu,
    shared_down,
    shared,
    latent_hbm,
    token_ids,
    probabilities,
    counts,
    maximum,
    *,
    rows,
    pair,
    counts_ready,
    metadata_ready,
):
    """Router, latent projection and shared experts, with incremental collectives.

    ``shared`` is an HBM buffer of the rank's ``rows`` rows: the shared-expert
    output goes there (the epilogue reads it back), so it holds no VMEM during
    the routed experts. The caller retains the metadata outputs across this
    scope. All temporary projection and communication buffers die before
    routed-expert scratch is allocated.

    The dense work runs in passes of at most 256 rows (two for a 16K window's
    512) over the caller's ``pair`` buffer (``[2 * pass rows, hidden]``, our
    half at ``parity * pass rows``): each pass loads its rows from
    ``sources[pass]`` (HBM; None when the caller has already put them in our
    half), sends them to the peer for the shared experts, runs the
    router/Top16 and the latent projection, starts its own latent ring and
    runs the shared experts; the route exchange and the route metadata cover
    every pass's routes and start after the last pass's Top16.
    """
    hidden_size = router.shape[-2]
    passes = 2 if rows == 512 else 1
    part = rows // passes
    big = part == 256  # the 8K streaming schedule (8K and each pass of 16K)
    rank = jax.lax.axis_index("k3_sp")
    parity, peer = rank % 2, rank ^ 1
    family = control[1]
    router = router.at[family]
    bias = bias.at[family]
    latent_weight = latent_weight.at[family]
    shared_gu = shared_gu.at[family]
    shared_down = shared_down.at[family]
    valid = _stripe_validity(rows, control[0], rank)
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(s):
        pair_v = pair
        own = pair_v.at[pl.ds(parity * part, part), :]
        if big:
            buffers = tuple(s["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4))
            router_v = s["router_gu"].at[:, :896]
        else:
            buffers, router_v = s["gu_buffers"], s["router"]
        gate_buffers, up_buffers = buffers[:2], buffers[2:]
        gate_sems, up_sems = s["gu_sems"][:2], s["gu_sems"][2:]

        def input_copy(h):
            return pltpu.make_async_copy(sources[h], own, s["input_sem"])

        # The router's VMEM doubles as the panel buffers: each pass reloads it.
        router_copy = pltpu.make_async_copy(router, router_v, s["gu_sems"][0])
        bias_copy = pltpu.make_async_copy(bias, s["bias"], s["gu_sems"][1])
        if sources[0] is not None:
            input_copy(0).start()
        router_copy.start()
        bias_copy.start()
        _enter_mesh()
        route_steps, metadata_steps = iter(()), iter(())
        route_started = metadata_started = False

        def start_metadata():
            nonlocal metadata_started, metadata_steps
            if not metadata_started:
                metadata_started = True
                metadata_steps = _metadata_steps(
                    s["route_wire"],
                    s["keys"],
                    probabilities,
                    token_ids,
                    counts,
                    maximum,
                    tokens=rows * 32,
                    rank=rank,
                    counts_ready=counts_ready,
                    local_experts=counts.shape[0],
                )

        def metadata_progress():
            for _ in range((2 if rows <= 128 else 5) * passes):
                next(metadata_steps, None)

        rings = []  # big: each pass's 256-row latent ring, progressed together

        def ring(h):
            payload = s["latent_payload"].at[h]
            refs = (
                s["latent_send_sems"][h],
                s["latent_recv_sems"][h],
                s["latent_credits"][h],
                s["latent_stores"][h],
                s["latent_state"][h],
                s["latent_meta"][h],
            )
            send, recv, credits, stores, state, meta = refs
            _latent_ready_init(
                tables, payload, latent_hbm, send, recv, stores, state, meta,
                rank=rank, stride=rows, offset=h * part,
            )

            def progress(drain):
                _latent_ready_progress(
                    payload, latent_hbm, send, recv, credits, stores, state, meta,
                    rank=rank, stride=rows, offset=h * part, drain=drain,
                )

            rings.append(progress)

        latent_steps = iter(())

        def latent_progress(*, drain=False):
            if big:
                for progress in rings:
                    progress(drain)
            elif drain:
                for _ in latent_steps:
                    pass
            else:
                next(latent_steps, None)

        def gu_copy(panel, offset, buffers, sems):
            bank = panel % 2
            return pltpu.make_async_copy(
                shared_gu.at[:, pl.ds(offset + panel * 256, 256)],
                buffers[bank],
                sems[bank],
            )

        def down_copy(panel):
            return pltpu.make_async_copy(
                shared_down.at[:, pl.ds(panel * 256, 256)],
                s["down_buffers"][panel % 2],
                s["down_sems"][panel % 2],
            )

        for h in range(passes):
            last = h == passes - 1
            if h:
                if sources[h] is not None:
                    input_copy(h).start()
                router_copy.start()
            pl.semaphore_signal(
                s["peer_credit"], 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
            )
            if sources[h] is not None:
                input_copy(h).wait()
            pl.semaphore_wait(s["peer_credit"], 1)
            peer_input = pltpu.make_async_remote_copy(
                own,
                own,
                s["peer_send"],
                s["peer_recv"],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            peer_input.start()
            router_copy.wait()
            if not h:
                bias_copy.wait()
            hx = own[...]
            logits = jnp.matmul(hx, router_v[...], preferred_element_type=jnp.float32)
            top = _Top16(logits, s["bias"][...])
            routed = False
            if big:
                latent_buffers = tuple(
                    s["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4)
                )
                latent_products = _latent_half_pair_panels(
                    hx, latent_weight, latent_buffers, s["gu_sems"]
                )
            else:
                latent_products = _dense_panels(
                    hx, latent_weight, gate_buffers, gate_sems, columns=3584
                )
            first_half = None
            for col, product in latent_products:
                if route_started:
                    stage = next(route_steps, None)
                    if rows <= 128 and stage is None:
                        start_metadata()
                if rows <= 128 and metadata_started:
                    metadata_progress()
                if big:
                    fragment = product.astype(jnp.bfloat16)
                    if col < 1792:
                        first_half = fragment
                    else:
                        # Assemble both halves before writing their packed words.
                        assert first_half is not None
                        paired = jnp.concatenate((first_half, fragment), axis=1)
                        s["latent_payload"][h, :, :, pl.ds(col - 1792, 256)] = (
                            paired.reshape(part, 2, 256)
                        )
                else:
                    s["latent"][:, pl.ds(col, 256)] = product.astype(jnp.bfloat16)
                for _ in range(min(2, 16 - top.index)):
                    top.step()
                if top.index == 16 and not routed:
                    routed = True
                    ids, probs = top.finish(valid[h * part : (h + 1) * part])
                    bits = jax.lax.bitcast_convert_type(probs, jnp.int32)
                    if rows < 128:
                        wire = jnp.concatenate((ids, bits), axis=1)
                        s["route_payload"][...] = wire.reshape(rows // 4, 128)
                    else:
                        # Expert IDs then probability bits, token-major (the
                        # pass's tokens are rows h * part / 8 of each plane).
                        span = part // 8
                        s["route_payload"][pl.ds(h * span, span), :] = ids.reshape(
                            span, 128
                        )
                        s["route_payload"][pl.ds(rows // 8 + h * span, span), :] = (
                            bits.reshape(span, 128)
                        )
                    if last:
                        route_steps = _route_gather_steps(
                            s["route_payload"],
                            s["route_wire"],
                            s["route_send"],
                            s["route_recv"],
                            rank=rank,
                        )
                        next(route_steps, None)
                        route_started = True
            if last:
                for _ in route_steps:
                    pass
                start_metadata()
            if not big:
                s["latent_payload"][...] = s["latent"][...].reshape(rows, 2, 1792)
                latent_steps = _latent_gather_steps(
                    tables,
                    s["latent_payload"],
                    s["latent_recv"],
                    latent_hbm,
                    s["latent_send_sems"],
                    s["latent_recv_sems"],
                    s["latent_credits"],
                    s["latent_stores"],
                    rank=rank,
                )
            peer_input.wait_send()
            peer_input.wait_recv()

            gu_copy(0, 0, gate_buffers, gate_sems).start()
            gu_copy(0, 3072, up_buffers, up_sems).start()
            if big:
                ring(h)
            for panel in range(12):
                if not big:
                    latent_progress()
                bank = panel % 2
                gu_copy(panel, 0, gate_buffers, gate_sems).wait()
                gu_copy(panel, 3072, up_buffers, up_sems).wait()
                if panel + 1 < 12:
                    gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
                    gu_copy(panel + 1, 3072, up_buffers, up_sems).start()
                for row in (0, part):
                    lhs = pair_v[pl.ds(row, part), :]
                    gate = jnp.matmul(
                        lhs, gate_buffers[bank][...], preferred_element_type=jnp.float32
                    )
                    up = jnp.matmul(
                        lhs, up_buffers[bank][...], preferred_element_type=jnp.float32
                    )
                    metadata_progress()
                    activated = (4.0 * jnp.tanh(gate / 4.0)) * (
                        1.0 / (1.0 + jnp.exp(-gate))
                    )
                    activated *= 25.0 * jnp.tanh(up / 25.0)
                    s["middle"][pl.ds(row, part), pl.ds(panel * 256, 256)] = (
                        activated.astype(jnp.bfloat16)
                    )
                    if big:
                        latent_progress()

            def shared_store(panel, h=h):
                bank = panel % 2
                return pltpu.make_async_copy(
                    s["shared_stage"][bank],
                    shared.at[pl.ds(h * part, part), pl.ds(panel * 256, 256)],
                    s["shared_sems"][bank],
                )

            def finish_reduction(pending, shared_store=shared_store):
                transfer, bank, panel = pending
                own_sum = s["partial"][bank][pl.ds(parity * part, part), :]
                transfer.wait_send()
                transfer.wait_recv()
                if panel >= 2:
                    shared_store(panel - 2).wait()
                s["shared_stage"][bank][...] = (
                    own_sum + s["peer_y"][bank][...]
                ).astype(jnp.bfloat16)
                shared_store(panel).start()

            down_copy(0).start()
            pending = None
            for panel in range(28):
                bank = panel % 2
                down_copy(panel).wait()
                if panel + 1 < 28:
                    down_copy(panel + 1).start()
                for row in (0, part):
                    product = jnp.matmul(
                        s["middle"][pl.ds(row, part), :],
                        s["down_buffers"][bank][...],
                        preferred_element_type=jnp.float32,
                    )
                    metadata_progress()
                    s["partial"][bank][pl.ds(row, part), :] = product
                    if row == part:
                        if pending is not None:
                            finish_reduction(pending)
                        transfer = pltpu.make_async_remote_copy(
                            s["partial"][bank].at[pl.ds((1 - parity) * part, part), :],
                            s["peer_y"][bank],
                            s["reduce_send"][bank],
                            s["reduce_recv"][bank],
                            device_id=peer,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        transfer.start()
                        pending = transfer, bank, panel
                    if big or row == part:
                        latent_progress()
            finish_reduction(pending)
            for panel in (26, 27):
                shared_store(panel).wait()
        if big:
            for _ in metadata_steps:
                latent_progress()
            # Start compact VMEM-to-SMEM transfers while the latent rings finish.
            metadata_ready()
            latent_progress(drain=True)
        else:
            latent_progress(drain=True)
            for _ in metadata_steps:
                pass

    ring_count = passes if big else 1
    scratch = dict(
        router=vmem((hidden_size, 896), jnp.bfloat16) if not big else None,
        router_gu=vmem((hidden_size, 1024), jnp.bfloat16) if big else None,
        gu_buffers=tuple(vmem((hidden_size, 256), jnp.bfloat16) for _ in range(4))
        if not big
        else (),
        gu_sems=tuple(dma for _ in range(4)),
        bias=vmem((1, 896), jnp.float32),
        input_sem=dma if any(source is not None for source in sources) else None,
        peer_send=dma,
        peer_recv=dma,
        peer_credit=pltpu.SemaphoreType.REGULAR,
        latent=vmem((rows, 3584), jnp.bfloat16) if not big else None,
        route_payload=vmem((rows // 4, 128), jnp.int32),
        route_wire=vmem((rows * 8, 128), jnp.int32),
        route_send=dma((5,)),
        route_recv=dma((5,)),
        keys=vmem((rows * 4, 128), jnp.int32),
        latent_payload=vmem((passes, part, 2, 1792), jnp.bfloat16)
        if big
        else vmem((rows, 2, 1792), jnp.bfloat16),
        latent_recv=vmem((2, rows, 2, 1792), jnp.bfloat16) if not big else None,
        latent_state=tuple(pltpu.SMEM((25,), jnp.int32) for _ in range(ring_count))
        if big
        else None,
        latent_meta=tuple(pltpu.SMEM((6 * 34,), jnp.int32) for _ in range(ring_count))
        if big
        else None,
        latent_send_sems=tuple(dma((12,)) for _ in range(ring_count)) if big else dma((12,)),
        latent_recv_sems=tuple(dma((2, 12)) for _ in range(ring_count))
        if big
        else dma((2, 12)),
        latent_credits=tuple(pltpu.SemaphoreType.REGULAR((12,)) for _ in range(ring_count))
        if big
        else pltpu.SemaphoreType.REGULAR((12,)),
        latent_stores=tuple(dma for _ in range(ring_count)) if big else dma((12,)),
        middle=vmem((2 * part, 3072), jnp.bfloat16),
        down_buffers=tuple(vmem((3072, 256), jnp.bfloat16) for _ in range(2)),
        down_sems=(dma, dma),
        partial=tuple(vmem((2 * part, 256), jnp.float32) for _ in range(2)),
        peer_y=tuple(vmem((part, 256), jnp.float32) for _ in range(2)),
        reduce_send=(dma, dma),
        reduce_recv=(dma, dma),
        shared_stage=tuple(vmem((part, 256), jnp.bfloat16) for _ in range(2)),
        shared_sems=(dma, dma),
    )
    pl.run_scoped(body, scratch)


def _routed_experts(
    latent,
    gu_weights,
    gu_scales,
    rhs,
    scales,
    counts,
    destinations,
    probabilities,
    tables,
    layer,
    offset,
    middle,
    out,
    *,
    tokens,
    probabilities_ready,
):
    """Joint GU/SiTU/first-down pipeline followed by remaining columns.

    Down columns are 1792 wide below 8K, 896 at 8K and 512 at 16K (the
    window's FP32 token sums of one column block fill half of VMEM either
    way); the remaining columns pair expert tiles up to 8K.
    """
    columns = {8192: 896, 16384: 512}.get(tokens, 1792)
    tile = 128 if tokens < 8192 else 256
    assert tokens in (512, 1024, 2048, 4096, 8192, 16384)
    gu_width = gu_weights.shape[-1]
    assert gu_width % 2 == 0
    mid_width = gu_width // 2
    assert rhs.shape[-2] * 8 == mid_width
    unroll = 8 if tokens < 8192 else 32
    gu_unroll = 4 if tokens < 8192 else 8
    first_down_unroll = 2 if tokens < 8192 else 8
    first_tail_unroll = 8
    ring_dtype = jnp.bfloat16
    x = jax.ShapeDtypeStruct((_route_capacity(tokens), mid_width), latent.dtype)
    rows, width = x.shape[0], rhs.shape[-1]
    assert middle.shape == x.shape
    tm, tn, localrows = (tile, columns, tokens // 32)
    packed_half = 512 if tokens == 8192 else tn // 2
    assert width % tn == 0
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
    cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, x.shape[-1]), x.dtype),
        jax.ShapeDtypeStruct(rhs.shape[1:], rhs.dtype),
        jax.ShapeDtypeStruct(scales.shape[1:], scales.dtype),
        None,
        counts,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=False,
        zero_initialize=False,
        fuse_act=None,
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    cfg = dataclasses.replace(
        cfg, tiles=dataclasses.replace(cfg.tiles, tile_m=tm, tile_n=tn)
    )
    sub = cfg.dims.size_lhs_sublane
    max_gm = cfg.dims.size_group + pl.cdiv(rows, tm) - 1
    regions = k._regions(localrows, tn)
    streams = len(regions)
    store_tile = 128
    gu_cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct((rows, 3584), latent.dtype),
        jax.ShapeDtypeStruct(gu_weights.shape[1:], gu_weights.dtype),
        jax.ShapeDtypeStruct(gu_scales.shape[1:], gu_scales.dtype),
        None,
        counts,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=None,
        acc_dtype=None,
        maybe_quantize_lhs=False,
        zero_initialize=False,
        fuse_act="situ:4:25",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    gu_cfg = dataclasses.replace(
        gu_cfg,
        tiles=dataclasses.replace(gu_cfg.tiles, tile_m=tm, tile_n=mid_width, tile_k=3584),
    )
    assert gu_cfg.dims.size_lhs_sublane == sub
    for config in (cfg, gu_cfg):
        assert not config.lhs_cfgs.should_quantize
        assert config.rhs_cfgs.should_dequantize_before_matmul
        assert not config.rhs_cfgs.should_dequantize_after_matmul

    def kernel(
        counts,
        offset,
        layer,
        tables,
        destinations,
        probabilities,
        latent,
        gu,
        w,
        middle,
        out,
        metadata,
        packed,
        sums,
        sends,
        receives,
        credits,
        stores,
        payload,
        rx,
        state,
        ring_meta,
    ):
        w = dataclasses.replace(
            w, weight=w.weight.at[layer[0]], scale=w.scale.at[layer[0]]
        )
        w = dataclasses.replace(w, weight=w.weight.bitcast(jnp.float4_e2m1fn))
        # Gate/up columns are lane-interleaved in memory (gate and up 128-column
        # chunks alternate), so one contiguous tile feeds the fused SiTU.
        gu = dataclasses.replace(
            gu,
            weight=gu.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
            scale=gu.scale.at[layer[0]],
        )
        # Six directed rings: [target, source, 32 local-contribution row bases].
        # One initialization per layer call, reused across all four columns.
        rank = lax.axis_index("k3_sp")
        positions = [tables[f, 0, rank] for f in range(3)]
        peers = []
        for family in range(3):
            position = positions[family]
            peer_neg = tables[family, 1, (position - 1) & 31]
            peer_pos = tables[family, 1, (position + 1) & 31]
            d_neg, d_pos = 2 * family, 2 * family + 1
            ring_meta[d_neg * 34] = peer_neg
            ring_meta[d_neg * 34 + 1] = peer_pos
            ring_meta[d_pos * 34] = peer_pos
            ring_meta[d_pos * 34 + 1] = peer_neg
            peers.append((peer_neg,))
            peers.append((peer_pos,))

        @pl.loop(0, 32)
        def hop(step):
            delta = step + 1
            for family in range(3):
                position = positions[family]
                owner_pos = tables[family, 1, (position + delta) & 31]
                owner_neg = tables[family, 1, (position - delta) & 31]
                ring_meta[(2 * family) * 34 + 2 + step] = (
                    tables[family, 2, owner_pos] * localrows
                )
                ring_meta[(2 * family + 1) * 34 + 2 + step] = (
                    tables[family, 2, owner_neg] * localrows
                )

        active_ng = k.fill_metadata(counts, offset, metadata, cfgs=cfg)

        @pl.when(active_ng == 0)
        def empty_metadata():
            metadata.gm_id_to_group_id[0] = 0
            metadata.gm_id_to_m_offset[0] = 0
            metadata.gm_id_to_m_offset[1] = 0

        ng = jnp.maximum(active_ng, 1)
        (xs, ws), _ = k.generate_block_specs(metadata, cfg)
        state[0] = 0

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(ready, len(peers))

        # Real-prompt tuning: short expert loops need all twelve streams per
        # poll; longer loops alternate six-stream groups to bound polling cost.
        def advance(stop, block, subtotal_vmem, *, drain=False, poll_round=0):
            _ready_reduce(
                tables, subtotal_vmem, out.at[:, pl.ds(block * tn, tn)],
                sends, receives, credits, stores, payload, rx,
                streams=regions, buffers=2, state=state, ring_meta=ring_meta, drain=drain, poll_round=poll_round, sparse_poll=(ng >= 72),
            )

        def emit_subtotals(subtotal_vmem):
            @pl.loop(0, tokens // store_tile)
            def block(b):
                # Trim the aligned-layout padding before the existing ring.
                # The cut at tn=896 is lane aligned, unlike a 448-wide half.
                values = sums[pl.ds(b * store_tile, store_tile)].reshape(
                    store_tile, 2 * packed_half
                )
                subtotal_vmem.at[pl.ds(b * store_tile, store_tile)][...] = values[
                    :, :tn
                ].astype(jnp.bfloat16)

        # Token combine: the GU rows already carry their route weights, so each
        # expert row adds into its token's sums unweighted. A batch issues all
        # loads before its stores; one expert holds a token at most once, so the
        # destinations of a batch differ (padding rows go to the dummy slots).
        def batched_unclamped(source, begin, base_idx, width_b):
            idxs = [base_idx + r for r in range(width_b)]
            dests = [destinations[i_r] for i_r in idxs]
            vals = [
                source.at[pl.ds(i_r - begin, 1)][...].astype(jnp.float32)
                for i_r in idxs
            ]
            olds = [sums.at[pl.ds(d_r, 1)][...] for d_r in dests]
            for r in range(width_b):
                sums.at[pl.ds(dests[r], 1)][...] = olds[r] + vals[r]

        def batched_clamped(
            source, begin, base_idx, end, width_b, *, guard, dummy_base=0
        ):
            idxs = [base_idx + r for r in range(width_b)]
            clamped = [jnp.minimum(idx_r, end - 1) for idx_r in idxs]
            dests = [
                jnp.where(
                    guard & (idxs[r] < end),
                    destinations[clamped[r]],
                    tokens + ((dummy_base + r) % _RMW_DUMMY_SLOTS),
                )
                for r in range(width_b)
            ]
            vals = [
                source.at[pl.ds(c_r - begin, 1)][...].astype(jnp.float32)
                for c_r in clamped
            ]
            olds = [sums.at[pl.ds(d_r, 1)][...] for d_r in dests]
            for r in range(width_b):
                sums.at[pl.ds(dests[r], 1)][...] = olds[r] + vals[r]

        def consume(tile_g, source, first=0, *, loop_unroll=unroll):
            start = metadata.gm_id_to_m_offset[tile_g]
            end = metadata.gm_id_to_m_offset[tile_g + 1]
            begin = start // sub * sub
            start = jnp.minimum(start + first, end)
            step_b = min(_RMW_BATCH, loop_unroll)
            full = (end - start) // loop_unroll

            @pl.loop(0, full)
            def chunk(b):
                base = start + b * loop_unroll
                for u in range(0, loop_unroll, step_b):
                    batched_unclamped(source, begin, base + u, min(step_b, loop_unroll - u))

            tail_start = start + full * loop_unroll
            tail_steps = pl.cdiv(end - tail_start, step_b)

            @pl.loop(0, tail_steps)
            def tail(tb):
                batched_clamped(
                    source,
                    begin,
                    tail_start + tb * step_b,
                    end,
                    step_b,
                    guard=jnp.bool_(True),
                )

        gu_panels = pl.cdiv(gu_width, pltpu.get_tpu_info().mxu_column_size)
        moved_rows = gu_panels * gu_unroll

        def gu_work(g, panel):
            gu_u = gu_unroll
            previous_g = jnp.maximum(g - 1, 0)
            start = metadata.gm_id_to_m_offset[previous_g]
            end = metadata.gm_id_to_m_offset[previous_g + 1]
            begin = start // sub * sub
            previous = packed.at[(g + 1) % 2]
            step_b = min(_RMW_BATCH, gu_u)
            for row in range(0, gu_u, step_b):
                batched_clamped(
                    previous,
                    begin,
                    start + panel * gu_u + row,
                    end,
                    min(step_b, gu_u - row),
                    guard=(g > 0),
                    dummy_base=panel * gu_u + row,
                )

        def inner(x, w, metadata, *, first=False, subtotal_vmem=None, tile=None, poll=True):
            """One expert tile of a down column: the pipeline's grid step, or
            ``tile`` of a paired step. ``poll`` (a static or traced bool) runs
            the reduce ring's progress before the tile."""
            g = pl.program_id(1) if tile is None else tile
            n = 0 if first else pl.program_id(0) + 1
            if not first:

                @pl.when(g == 0)
                def clear():
                    sums[...] = jnp.zeros(sums.shape, jnp.float32)

                def ring_poll():
                    advance(
                        jnp.minimum(32, (g + 1) * 32 // ng), n - 1, subtotal_vmem,
                        poll_round=g if tile is None else g // 2,
                    )

                if poll is True:
                    ring_poll()
                elif poll is not False:
                    pl.when(poll)(ring_poll)
            current = packed.at[g % 2]
            previous = packed.at[(g + 1) % 2]
            previous_g = jnp.maximum(g - 1, 0)
            panel_count = pl.cdiv(tn, pltpu.get_tpu_info().mxu_column_size)
            down_unroll = first_down_unroll if first else unroll
            gu_moved = moved_rows if first else 0

            def overlap_work(panel):
                down_u = down_unroll
                start = metadata.gm_id_to_m_offset[previous_g]
                end = metadata.gm_id_to_m_offset[previous_g + 1]
                begin = start // sub * sub
                base_off = gu_moved + panel * down_u
                step_b = min(_RMW_BATCH, down_u)
                for row in range(0, down_u, step_b):
                    batched_clamped(
                        previous,
                        begin,
                        start + base_off + row,
                        end,
                        min(step_b, down_u - row),
                        guard=(g > 0),
                        dummy_base=panel * down_u + row,
                    )

            if not first:

                @pl.when(g == 0)
                def initialize_previous():
                    previous[...] = jnp.zeros(previous.shape, previous.dtype)

            @pl.when(active_ng > 0)
            def product():
                _down_inner(
                    x,
                    w,
                    current,
                    metadata,
                    cfgs=cfg,
                    overlap_work=overlap_work,
                    group=g,
                )

            @pl.when(g > 0)
            def previous_tail():
                consume(
                    previous_g,
                    previous,
                    gu_moved + panel_count * down_unroll,
                    loop_unroll=first_tail_unroll if first else unroll,
                )

            @pl.when(g == ng - 1)
            def finish():
                consume(g, current, loop_unroll=first_tail_unroll if first else unroll)
                if not first:
                    advance(jnp.int32(32), n - 1, subtotal_vmem, drain=True)
                    emit_subtotals(subtotal_vmem)
                    state[0] = 0

                    @pl.when(n == width // tn - 1)
                    def last():
                        advance(jnp.int32(32), n, subtotal_vmem, drain=True)

        (_, gu_ws), gu_ys = k.generate_block_specs(metadata, gu_cfg)
        gu_ws = _interleaved_gate_up_specs(gu_ws, gu_width)

        @functools.partial(
            pl.run_scoped,
            gu_partial=pltpu.VMEM((sub, mid_width), gu_cfg.out_dtype),
            acts=pltpu.VMEM((2, tm, 2, 1792), latent.dtype),
            act_sems=pltpu.SemaphoreType.DMA((2,)),
            row_weights=pltpu.VMEM((2, tm, 128), jnp.float32),
        )
        def first_column(gu_partial, acts, act_sems, row_weights):
            def buffering(spec):
                return dataclasses.replace(
                    spec, pipeline_mode=pl.Buffered(buffer_count=2)
                )

            joint_gu_ws, joint_down_ws = jax.tree.map(
                buffering, (gu_ws, ws), is_leaf=lambda v: isinstance(v, pl.BlockSpec)
            )

            def active_bounds(g):
                raw_start = metadata.gm_id_to_m_offset[g]
                raw_end = metadata.gm_id_to_m_offset[g + 1]
                begin = raw_start // sub * sub
                return (raw_start, raw_start - begin, raw_end - raw_start)

            def start(g, bank):
                raw_start, first, n_active = active_bounds(g)
                unroll = 8 if tokens < 8192 else 16
                full = (n_active // unroll) * unroll

                def weigh(i):
                    row_weights.at[bank, pl.ds(first + i, 1)][...] = jnp.full(
                        (1, 128), probabilities[raw_start + i], jnp.float32
                    )

                @pl.loop(0, n_active // unroll)
                def row_vec(blk):
                    base = blk * unroll
                    sources = [destinations[raw_start + base + u] for u in range(unroll)]
                    for u in range(unroll):
                        pltpu.make_async_copy(
                            latent.at[pl.ds(sources[u], 1)],
                            acts.at[bank, pl.ds(first + base + u, 1)],
                            act_sems.at[bank],
                        ).start()
                    for u in range(unroll):
                        weigh(base + u)

                @pl.loop(full, n_active)
                def row_rem(i):
                    source = destinations[raw_start + i]
                    pltpu.make_async_copy(
                        latent.at[pl.ds(source, 1)],
                        acts.at[bank, pl.ds(first + i, 1)],
                        act_sems.at[bank],
                    ).start()
                    weigh(i)

            def joint(gw, dw, gy, metadata, gu_partial, acts, act_sems, row_weights):
                g = pl.program_id(1)
                bank = g % 2

                @pl.when(g == 0)
                def initialize_accumulation():
                    # emit_pipeline has already prefetched the first weights.
                    # The <=4K probability DMA can overlap metadata and prefetch;
                    # drain once here before any expert reads its probability.
                    probabilities_ready()
                    sums[...] = jnp.zeros(sums.shape, jnp.float32)
                    packed.at[1][...] = jnp.zeros(packed.at[1].shape, packed.dtype)
                    acts[...] = jnp.zeros(acts.shape, acts.dtype)
                    row_weights[...] = jnp.zeros(row_weights.shape, jnp.float32)

                @pl.when(active_ng > 0)
                def gate_up():
                    @pl.when(g == 0)
                    def initial():
                        start(g, bank)

                    _, first, n_active = active_bounds(g)

                    @pl.when(n_active > 0)
                    def wait_active():
                        active = acts.at[bank, pl.ds(first, n_active)]
                        pltpu.make_async_copy(
                            active, active, act_sems.at[bank]
                        ).wait()

                    @pl.when(g + 1 < ng)
                    def following():
                        start(g + 1, 1 - bank)

                    _gu_inner(
                        _PackedRows(acts.at[bank]),
                        gw,
                        gy,
                        gu_partial,
                        metadata,
                        row_weights.at[bank],
                        cfgs=gu_cfg,
                        overlap_work=lambda panel: gu_work(g, panel),
                    )

                inner(gy, dw, metadata, first=True)

            specs = dict(
                grid=(1, ng, 1),
                in_specs=(joint_gu_ws, joint_down_ws),
                out_specs=gu_ys,
            )
            refs = (gu, w, middle.reshape(-1, sub, mid_width))
            scratches = (metadata, gu_partial, acts, act_sems, row_weights)
            pltpu.emit_pipeline(joint, **specs)(*refs, scratches=scratches)

        @functools.partial(
            pl.run_scoped, subtotal_vmem=pltpu.VMEM((tokens, tn), x.dtype)
        )
        def remaining_columns(subtotal_vmem):
            emit_subtotals(subtotal_vmem)
            state[0] = 0

            def shifted(spec):
                index_map = spec.index_map
                return dataclasses.replace(
                    spec, index_map=lambda n, g, k: index_map(n + 1, g, k)
                )

            shifted_ws = jax.tree.map(
                shifted, ws, is_leaf=lambda v: isinstance(v, pl.BlockSpec)
            )
            rows = middle.reshape(-1, sub, mid_width)
            if tokens == 16384:
                # The 16K token sums leave no room for a second tile's blocks.
                def remaining(x, w, metadata):
                    inner(x, w, metadata, subtotal_vmem=subtotal_vmem)

                pltpu.emit_pipeline(
                    remaining,
                    grid=(width // tn - 1, ng, 1),
                    in_specs=(xs, shifted_ws),
                    out_specs=(),
                )(rows, w, scratches=(metadata,))
                return

            # Two expert tiles per pipeline step (tiles 2h and 2h + 1; an odd
            # count's last step fetches its last tile twice and skips the copy),
            # with one reduce-ring poll per step: half the steps' fixed cost.
            # Two buffers per input: each step's pair plus the next pair in flight.
            def paired(spec, second):
                index_map = spec.index_map

                def mapped(n, h, k):
                    return index_map(n, jnp.minimum(2 * h + second, ng - 1), k)

                return dataclasses.replace(
                    spec, index_map=mapped, pipeline_mode=pl.Buffered(buffer_count=2)
                )

            def pair(specs, second):
                return jax.tree.map(
                    lambda spec: paired(spec, second),
                    specs,
                    is_leaf=lambda v: isinstance(v, pl.BlockSpec),
                )

            def remaining(xa, xb, wa, wb, metadata):
                h = pl.program_id(1)
                inner(xa, wa, metadata, subtotal_vmem=subtotal_vmem, tile=2 * h)

                @pl.when(2 * h + 1 < ng)
                def second():
                    inner(
                        xb, wb, metadata, subtotal_vmem=subtotal_vmem,
                        tile=2 * h + 1, poll=False,
                    )

            pltpu.emit_pipeline(
                remaining,
                grid=(width // tn - 1, (ng + 1) // 2, 1),
                in_specs=(
                    pair(xs, 0), pair(xs, 1), pair(shifted_ws, 0), pair(shifted_ws, 1)
                ),
                out_specs=(),
            )(rows, rows, w, w, scratches=(metadata,))

    def run(**scratch):
        kernel(
            counts,
            offset,
            layer,
            tables,
            destinations,
            probabilities,
            latent,
            k.WeightsRef(weight=gu_weights, scale=gu_scales, bias=None),
            k.WeightsRef(weight=rhs, scale=scales, bias=None),
            middle,
            out,
            **scratch,
        )

    pl.run_scoped(
        run,
        metadata=k.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
        ),
        packed=pltpu.VMEM((2, tm, 2, packed_half), x.dtype),
        sums=pltpu.VMEM((tokens + _RMW_DUMMY_SLOTS, 2, packed_half), jnp.float32),
        sends=pltpu.SemaphoreType.DMA((streams,)),
        receives=pltpu.SemaphoreType.DMA((2, streams)),
        credits=pltpu.SemaphoreType.REGULAR((streams,)),
        stores=pltpu.SemaphoreType.DMA,
        payload=pltpu.VMEM((localrows, tn), ring_dtype),
        rx=pltpu.VMEM((2, localrows, tn), ring_dtype),
        state=pltpu.SMEM((1 + 2 * streams,), jnp.int32),
        ring_meta=pltpu.SMEM((6 * 34,), jnp.int32),
    )


class _PackedRows:
    """Load only the static row bucket selected by the production GMM."""

    def __init__(self, ref):
        self.ref = ref

    def reshape(self, rows, width):
        assert rows == -1 and width == 2 * self.ref.shape[-1]
        return self

    def __getitem__(self, rows):
        return self.ref[rows].reshape(-1, 2 * self.ref.shape[-1])


def _interleaved_gate_up_specs(specs, width):
    """Block specs of a lane-interleaved gate/up weight: the GMM tiles the
    activated output (``tile_n`` columns), each tile reads ``2 * tile_n``
    contiguous weight columns."""
    return k.WeightsRef(
        weight=dataclasses.replace(
            specs.weight, block_shape=(*specs.weight.block_shape[:-1], width)
        ),
        scale=dataclasses.replace(
            specs.scale, block_shape=(*specs.scale.block_shape[:-1], width)
        ),
        bias=None,
    )


_DOWN_K_STEP = 768  # rows of the 256-row down tile's weights dequantized at once


def _dequantized(rhs, cfgs, rows, cols):
    """BF16 weights of ``rhs`` rows ``rows`` and columns ``cols`` (slices)."""
    block = cfgs.rhs_cfgs.quant_block_size
    part = dataclasses.replace(
        rhs,
        weight=rhs.weight.at[rows, cols],
        scale=rhs.scale.at[pl.ds(rows.start // block, rows.size // block), cols],
    )
    weights = part.get_weight().astype(cfgs.lhs_cfgs.dtype)
    scales = part.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
    shape = weights.shape
    return (weights.reshape(shape[0] // block, block, shape[1]) * scales).reshape(shape)


def _product_panels(lhs, rhs, *, cfgs, bucket_rows, overlap_work):
    """Yield ``(panel, FP32 product)`` per MXU column panel of one expert tile.

    Both routed projections cover their complete reduction dimension in one
    tile. The 128-row tile dequantizes its whole weight tile once. The 256-row
    gate/up tile dequantizes per panel; the 256-row down tile reduces its
    1536 rows in 768-row steps, each dequantized whole, and accumulates in
    FP32. The callback consumes rows of the previous expert tile between the
    last step's MXU panels; it must not read this product's result.
    """
    assert cfgs.dims.size_k == cfgs.tiles.tile_k
    assert not cfgs.rhs_cfgs.has_bias
    columns = pltpu.get_tpu_info().mxu_column_size
    size_k = cfgs.tiles.tile_k
    activations = lhs.reshape(-1, size_k)[:bucket_rows]
    width = rhs.weight.shape[-1]
    panels = [(first, min(columns, width - first)) for first in range(0, width, columns)]
    if cfgs.tiles.tile_m == 256 and cfgs.fuse_act is None:
        k_step = min(size_k, _DOWN_K_STEP)
        accumulators = [None] * len(panels)
        for k0 in range(0, size_k, k_step):
            rows = pl.ds(k0, k_step)
            weights = _dequantized(rhs, cfgs, rows, pl.ds(0, width))
            last_step = k0 + k_step == size_k
            for index, (first, panel_width) in enumerate(panels):
                product = jnp.matmul(
                    activations[:, k0:k0 + k_step],
                    weights[:, first:first + panel_width],
                    preferred_element_type=jnp.float32,
                ).astype(cfgs.acc_dtype)
                if last_step:
                    overlap_work(index)
                total = accumulators[index]
                accumulators[index] = product if total is None else total + product
        for index, accumulator in enumerate(accumulators):
            yield index, accumulator
        return
    whole = None
    if cfgs.tiles.tile_m != 256:
        whole = _dequantized(rhs, cfgs, pl.ds(0, size_k), pl.ds(0, width))
    for index, (first, panel_width) in enumerate(panels):
        if whole is None:
            panel_weights = _dequantized(
                rhs, cfgs, pl.ds(0, size_k), pl.ds(first, panel_width)
            )
        else:
            panel_weights = whole[:, first:first + panel_width]
        product = jnp.matmul(
            activations, panel_weights, preferred_element_type=jnp.float32
        ).astype(cfgs.acc_dtype)
        overlap_work(index)
        yield index, product


def _expert_product(lhs, rhs, *, cfgs, bucket_rows, overlap_work):
    """The complete product of ``_product_panels``."""
    panels = _product_panels(
        lhs, rhs, cfgs=cfgs, bucket_rows=bucket_rows, overlap_work=overlap_work
    )
    return jnp.concatenate([product for _, product in panels], axis=1)


def _gu_inner(lhs, rhs, output, partial, metadata, row_weights, *, cfgs, overlap_work):
    """GU/SiTU with the production row buckets and adjacent-expert carry.

    Each 256-column weight panel holds a gate and an up 128-column chunk
    (lane-interleaved layout), so its product is activated, scaled by the
    rows' route weights (``row_weights``, lane-broadcast; the down projection's
    rows then combine unweighted) and stored as 128 intermediate columns at
    once. Expert boundaries need not align with the 16-row BF16 sublane. Mask
    outside the active expert, then preserve the preceding expert's partial
    first row.
    """
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    first, last = start - aligned, end - aligned
    lanes = pltpu.get_tpu_info().num_lanes

    def product(bucket_index):
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        row = lax.broadcasted_iota(jnp.int32, (rows, lanes), 0)
        live = (first <= row) & (row < last)
        weights = row_weights[:rows, :1]
        rows_out = output.reshape(-1, cfgs.tiles.tile_n)
        for panel, value in _product_panels(
            lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
        ):
            value = k.apply_act_fn(value, cfgs.fuse_act) * weights
            rows_out[:rows, pl.ds(panel * lanes, lanes)] = jnp.where(
                live, value, 0
            ).astype(output.dtype)
        zeros = jnp.zeros_like(partial)
        output[0] += jnp.where(group == 0, zeros, partial[...])
        last_row = last // cfgs.dims.size_lhs_sublane
        partial[...] = jnp.where(
            last % cfgs.dims.size_lhs_sublane == 0, zeros, output[last_row]
        )

    # The 256-row tile shares the 128-row product to bound instruction size.
    first_bucket = 1 if cfgs.tiles.tile_m == 256 else 0
    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(first_bucket, cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(
        jnp.maximum(
            jnp.maximum(last - 1, 0) // cfgs.tiles.bucket_base - first_bucket, 0
        ),
        branches,
    )


def _down_inner(lhs, rhs, output, metadata, *, cfgs, overlap_work, group):
    """Down projection stored as packed rows for token-wise VALU combine.

    Each valid expert row is consumed individually, so this output needs no
    adjacent-expert carry. Rows arrive weighted (see ``_gu_inner``).
    """
    assert cfgs.fuse_act is None
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    last = end - aligned

    # GU already masks inactive rows; consume() visits only live routes.
    def product(bucket_index):
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        value = _expert_product(
            lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
        )
        if cfgs.tiles.tile_n == 896:
            # Keep the half boundary on a complete 128-lane block.
            # Only the final 128 elements are padding; the consumer trims
            # them once per subtotal before communicating the 896 columns.
            value = value.astype(output.dtype)
            padded = jnp.concatenate(
                (value, jnp.zeros((rows, 128), output.dtype)), axis=1
            )
            output[:rows] = padded.reshape(rows, 2, 512)
        else:
            output[:rows] = value.astype(output.dtype).reshape(
                rows, 2, cfgs.tiles.tile_n // 2
            )

    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(jnp.maximum(last - 1, 0) // cfgs.tiles.bucket_base, branches)


def _reduce_step(
    tables_ref,
    x_ref,
    o_ref,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    payload,
    received,
    *,
    streams,
    buffers,
    step,
):
    rank = lax.axis_index("k3_sp")
    descriptors = []
    positions = [tables_ref[f, 0, rank] for f in range(3)]

    def ring_coords(pos, family):
        peer = tables_ref[family, 1, pos]
        return (peer,)

    @pl.when(step == 0)
    def enter():
        peers = [
            ring_coords((positions[f] + direction) % 32, jnp.int32(f))
            for f in range(3)
            for direction in (-1, 1)
        ]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
        cols = pl.ds(col_begin, width)
        family = stream // 2 % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = ring_coords((position + direction) % 32, jnp.int32(family))
        source = ring_coords((position - direction) % 32, jnp.int32(family))
        owner_mesh = ring_coords(
            (position - direction * (step + 1)) % 32, jnp.int32(family)
        )[0]
        owner = tables_ref[family, 2, owner_mesh]
        rows = pl.ds(begin, size)
        source_rows = pl.ds(owner * o_ref.shape[0] + begin, size)
        descriptors.append((target, source, rows, cols, source_rows))
    for stream, (target, source, rows, cols, source_rows) in enumerate(descriptors):

        def local_value(rows=rows, cols=cols, source_rows=source_rows):
            return (
                x_ref.reshape(x_ref.shape[0], o_ref.shape[1])
                .at[source_rows, cols][...]
                .astype(payload.dtype)
            )

        def remote(index, rows=rows, cols=cols, stream=stream, target=target):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step > 0)
        def wait(remote=remote):
            remote(step - 1).wait()

        @pl.when(step == 0)
        def seed(rows=rows, cols=cols, local_value=local_value):
            payload.at[rows, cols][...] = local_value()

        @pl.when(step > 0)
        def add(
            rows=rows, cols=cols, stream=stream, source=source, local_value=local_value
        ):
            payload.at[rows, cols][...] = (
                local_value().astype(jnp.float32)
                + received.at[(step - 1) % buffers, rows, cols][...].astype(jnp.float32)
            ).astype(payload.dtype)
            pl.semaphore_signal(
                credits.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < 31)
        def forward(remote=remote, stream=stream):
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[stream], 1)

            remote(step).start()

        @pl.when(step == 31)
        def finish(rows=rows, cols=cols, stream=stream):
            pl.semaphore_wait(credits.at[stream], buffers)

    @pl.when(step == 31)
    def store_complete_shard():
        output = pltpu.make_async_copy(payload, o_ref, store_sem)
        output.start()
        output.wait()


def _ready_reduce(
    tables_ref,
    x_ref,
    o_ref,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    payload,
    received,
    *,
    streams,
    buffers,
    state,
    ring_meta,
    drain=False,
    poll_round=0,
    sparse_poll=True,
):
    """Same 12-stream protocol with cached descriptors and cheap failed polls.

    ring_meta has six directed rings, each with neighbor mesh IDs followed by
    32 source-row bases. SMEM offsets, not physical VMEM pointers, are cached.
    Neither a failed receive nor a failed credit poll needs a payload address.
    """
    assert buffers == 2
    count = len(streams)

    def remote(stream, rows, cols, target, bank):
        return pltpu.make_async_remote_copy(
            payload.at[rows, cols],
            received.at[bank, rows, cols],
            send_sems.at[stream],
            recv_sems.at[bank, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    @pl.when(state[0] == 0)
    def initialize():
        for stream, region in enumerate(streams):
            begin, size, col_begin, width = region
            base = (stream % 6) * 34
            rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
            source_rows = pl.ds(pl.multiple_of(ring_meta[base + 2], o_ref.shape[0]) + begin, size)
            target = (ring_meta[base],)
            payload.at[rows, cols][...] = x_ref.at[source_rows, cols][...].astype(payload.dtype)
            remote(stream, rows, cols, target, jnp.int32(0)).start()
            state[1 + stream] = 1
            state[1 + count + stream] = 0
        state[0] = count + 1

    def sweep(selected=None):
        for stream, region in enumerate(streams):
            if selected is not None and (stream // 6) != selected:
                continue
            begin, size, col_begin, width = region
            base = (stream % 6) * 34
            rows, cols = pl.ds(begin, size), pl.ds(col_begin, width)
            units = size * width * jnp.dtype(payload.dtype).itemsize // 32
            step = state[1 + stream]

            @pl.when(state[1 + count + stream] == 0)
            def receive(
                *, step=step, stream=stream, units=units, base=base, rows=rows, cols=cols, begin=begin, size=size
            ):
                bank = (step - 1) & 1
                ready = (pl.semaphore_read(send_sems.at[stream]) >= units) & (
                    pl.semaphore_read(recv_sems.at[bank, stream]) >= units
                )

                @pl.when(ready)
                def reduce():
                    target = (ring_meta[base],)
                    remote(stream, rows, cols, target, bank).wait()
                    source_rows = pl.ds(pl.multiple_of(ring_meta[base + 2 + step], o_ref.shape[0]) + begin, size)
                    payload.at[rows, cols][...] = (
                        x_ref.at[source_rows, cols][...].astype(jnp.float32)
                        + received.at[bank, rows, cols][...].astype(jnp.float32)
                    ).astype(payload.dtype)
                    pl.semaphore_signal(
                        credits.at[stream], 1, device_id=(ring_meta[base + 1],), device_id_type=pl.DeviceIdType.MESH
                    )
                    state[1 + count + stream] = jnp.where(step == 31, 2, 1)

            @pl.when(state[1 + count + stream] == 1)
            def forward(*, step=step, stream=stream, rows=rows, cols=cols, base=base):
                available = (step < buffers) | (pl.semaphore_read(credits.at[stream]) >= 1)

                @pl.when(available)
                def send():
                    @pl.when(step >= buffers)
                    def acquire():
                        pl.semaphore_wait(credits.at[stream], 1)

                    remote(stream, rows, cols, (ring_meta[base],), step & 1).start()
                    state[1 + stream] = step + 1
                    state[1 + count + stream] = 0

            @pl.when(state[1 + count + stream] == 2)
            def finish(*, stream=stream):
                @pl.when(pl.semaphore_read(credits.at[stream]) >= buffers)
                def release():
                    pl.semaphore_wait(credits.at[stream], buffers)
                    state[1 + count + stream] = 3
                    state[0] = state[0] - 1

    if drain:

        def pending(i):
            return state[0] != 1

        def iterate(i):
            sweep()
            return i + 1

        jax.lax.while_loop(pending, iterate, jnp.int32(0))
        output = pltpu.make_async_copy(payload, o_ref, store_sem)
        output.start()
        output.wait()
    else:

        @pl.when(state[0] != 1)
        def progress():
            # Alternate two six-stream groups only for long expert loops.
            for group in range(2):

                @pl.when(((poll_round & 1) == group) | ~sparse_poll)
                def selected(*, group=group):
                    sweep(group)


def _epilogue(control, reduced, norm_weight, up_weight, prefix, output):
    """Normalize routed output and project up with three weight/prefix banks.

    ``output`` holds the shared-expert rows on entry (see ``_prologue``); each
    panel reads them back before it overwrites them with the layer output.
    Reduced rows remain in VMEM. Preserve the BF16 boundaries after
    normalization, after latent-up, and after adding the shared expert.
    """
    rows, width = reduced.shape
    family = control[1]
    norm_weight = norm_weight.at[family, :, :width]
    up_weight = up_weight.at[family]
    valid = _stripe_validity(rows, control[0], jax.lax.axis_index("k3_sp"))
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(s):
        copy = pltpu.make_async_copy(norm_weight, s["norm"], s["norm_sem"])
        copy.start()
        copy.wait()

        def weight_copy(panel):
            return pltpu.make_async_copy(
                up_weight.at[:, pl.ds(panel * 512, 512)],
                s["weights"][panel % 3],
                s["weight_sems"][panel % 3],
            )

        def prefix_copy(panel):
            return pltpu.make_async_copy(
                prefix.at[:, pl.ds(panel * 512, 512)],
                s["tiles"][panel % 3],
                s["prefix_sems"][panel % 3],
            )

        def shared_copy(panel):
            return pltpu.make_async_copy(
                output.at[:, pl.ds(panel * 512, 512)],
                s["shared"][panel % 3],
                s["shared_sems"][panel % 3],
            )

        def output_copy(panel):
            return pltpu.make_async_copy(
                s["tiles"][panel % 3],
                output.at[:, pl.ds(panel * 512, 512)],
                s["output_sems"][panel % 3],
            )

        # Hide the first weight/residual loads behind the RMS reduction, which
        # runs on blocks of at most 64 rows to bound its FP32 temporaries.
        for panel in (0, 1):
            weight_copy(panel).start()
            prefix_copy(panel).start()
            shared_copy(panel).start()
        step = min(64, rows)
        for block in range(rows // step):
            block_rows = pl.ds(block * step, step)
            routed = reduced[block_rows, :].astype(jnp.float32)
            mean = jnp.sum(routed * routed, axis=1, keepdims=True) / width
            inverse = jax.lax.rsqrt(mean + 1e-5)
            s["normalized"][block_rows, :] = (
                (routed * inverse) * s["norm"][...]
            ).astype(jnp.bfloat16)
        for panel in range(14):
            bank = panel % 3
            weight_copy(panel).wait()
            prefix_copy(panel).wait()
            shared_copy(panel).wait()
            if panel + 2 < 14:
                if panel >= 1:
                    output_copy(panel - 1).wait()
                weight_copy(panel + 2).start()
                prefix_copy(panel + 2).start()
                shared_copy(panel + 2).start()
            product = jnp.matmul(
                s["normalized"][...],
                s["weights"][bank][...],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            value = product + s["shared"][bank][...]
            value = jnp.where(valid, value, 0)
            s["tiles"][bank][...] = s["tiles"][bank][...] + value
            output_copy(panel).start()
        for panel in (11, 12, 13):
            output_copy(panel).wait()

    pl.run_scoped(
        body,
        dict(
            norm=vmem((1, width), jnp.float32),
            norm_sem=dma,
            normalized=vmem((rows, width), jnp.bfloat16),
            weights=tuple(vmem((width, 512), jnp.bfloat16) for _ in range(3)),
            tiles=tuple(vmem((rows, 512), jnp.bfloat16) for _ in range(3)),
            shared=tuple(vmem((rows, 512), jnp.bfloat16) for _ in range(3)),
            weight_sems=(dma, dma, dma),
            prefix_sems=(dma, dma, dma),
            shared_sems=(dma, dma, dma),
            output_sems=(dma, dma, dma),
        ),
    )


class _FlatMetadata:
    """Scalar expert-row access to the 2D SMEM layout used by metadata DMA.

    Keeping DMA source and destination two-dimensional avoids an unsupported
    1D VMEM memref reshape. Only scalar reads are needed by expert pipelines.
    """

    def __init__(self, ref):
        # Both views remain 2D; scalar consumers use one contiguous index.
        self.ref = ref.reshape(1, ref.shape[0] * ref.shape[1])

    def __getitem__(self, index):
        return self.ref[0, index]


def _layer_kernel(
    tables,
    control,
    hidden,
    router,
    bias,
    latent_down,
    shared_gu,
    shared_down,
    expert_gu,
    gu_scales,
    expert_down,
    down_scales,
    norm,
    latent_up,
    prefix,
    output,
    latent,
    middle,
    fallback_ids=None,
    fallback_probs=None,
    fallback_counts=None,
    maximum=None,
    route_counts=None,
    mixture_refs=None,
):
    """One launch owns all compute, metadata and communication on the fast path.

    Large projection scratch has a shorter lifetime than expert scratch. Only
    the scalar routing metadata survives that boundary in VMEM/SMEM; the
    shared-expert rows wait in ``output``. From 8K all ranks take the same
    overflow branch, which reads them from there; no routes are dropped to fit
    SMEM.
    """
    rows, hidden_size = prefix.shape
    tokens = rows * 32
    local_experts = expert_gu.shape[1]
    capacity = _route_capacity(tokens)
    shape = (tokens // 8, 128)
    compact = tokens * 16 > capacity  # the compact routes are a prefix of the sort
    part = rows // (2 if rows == 512 else 1)  # the prologue's pass rows
    passes = rows // part
    vmem, smem, dma = pltpu.VMEM, pltpu.SMEM, pltpu.SemaphoreType.DMA
    mixture = None
    sources = tuple(
        None if hidden is None else hidden.at[pl.ds(h * part, part)]
        for h in range(passes)
    )
    if mixture_refs is not None:
        blocks_ref, mlp_res_ref, ffn_norm_ref = mixture_refs
        mix_tile = min(128, part)
        # Pass 0 is mixed into the prologue's VMEM rows; later passes into their
        # rows of ``output`` (free until their shared experts), which the
        # prologue then loads, so no mixture scratch lives inside the prologue.
        sources = (None,) + tuple(
            output.at[pl.ds(h * part, part)] for h in range(1, passes)
        )

        def mixture(h, *, own=None, hbm=None):
            """Residual mixture + RMSNorm of pass ``h``'s rows into VMEM ``own``
            or HBM ``hbm``."""
            rows_h = pl.ds(h * part, part)
            store = hbm is not None

            def run(*mix_s):
                mix_s = list(mix_s)
                output_bufs = mix_s.pop(2) if store else None
                output_sems = mix_s.pop(-2) if store else None
                (
                    prefix_bufs,
                    block_buf,
                    numerator_buf,
                    row_max_buf,
                    row_sum_buf,
                    folded_buf,
                    norm_buf,
                    prefix_sems,
                    block_sem,
                    weight_sems,
                ) = mix_s
                k.mixture_norm_tiled_body(
                    control[2],
                    prefix.at[rows_h],
                    blocks_ref.at[:, rows_h],
                    mlp_res_ref,
                    ffn_norm_ref,
                    hbm,
                    prefix_bufs,
                    block_buf,
                    output_bufs,
                    numerator_buf,
                    row_max_buf,
                    row_sum_buf,
                    folded_buf,
                    norm_buf,
                    prefix_sems,
                    block_sem,
                    output_sems,
                    weight_sems,
                    tile=mix_tile,
                    eps=1e-5,
                    output_vmem_ref=own,
                )

            pl.run_scoped(
                run,
                *k.mixture_norm_tiled_scratch_shapes(
                    hidden_size, mix_tile, store_output=store
                ),
            )

    def body(s):
        s["offset"][0] = 0
        # Dense projections retain global family indices. Resident expert banks
        # correspond to first/group/final programs with 3/8/1 MoE families.
        family = k.expert_family_slot(expert_gu.shape[0], control[1])
        s["family"][0] = family

        def count_copy():
            return pltpu.make_async_copy(s["counts_v"], s["counts"], s["count_sem"])

        def maximum_copy():
            return pltpu.make_async_copy(maximum, s["maximum"], s["max_sem"])

        def counts_ready():
            count_copy().start()
            if maximum is not None:
                maximum_copy().start()

        def probability_copy(probs_v):
            return pltpu.make_async_copy(
                probs_v.at[: capacity // 128, :],
                s["probs"],
                s["prob_sem"],
            )

        def prepare(ids_v, probs_v, pair):
            indices = pltpu.make_async_copy(
                ids_v.at[: capacity // 128, :], s["ids"], s["index_sem"]
            )

            def metadata_ready():
                indices.start()
                probability_copy(probs_v).start()

            if mixture is not None:
                parity = jax.lax.axis_index("k3_sp") % 2
                mixture(0, own=pair.at[pl.ds(parity * part, part), :])
                for h in range(1, passes):
                    mixture(h, hbm=sources[h])
            _prologue(
                tables,
                control,
                sources,
                router,
                bias,
                latent_down,
                shared_gu,
                shared_down,
                output,
                latent,
                ids_v,
                probs_v,
                s["counts_v"],
                maximum,
                rows=rows,
                pair=pair,
                counts_ready=counts_ready,
                metadata_ready=metadata_ready,
            )
            count_copy().wait()
            if route_counts is not None:
                for e in range(local_experts):
                    route_counts[e] = s["counts"][e]
            if maximum is not None:
                maximum_copy().wait()
            if tokens < 8192:
                metadata_ready()
            if maximum is not None:

                @pl.when(s["maximum"][0] > capacity)
                def export_overflow():
                    for source, dest in (
                        (ids_v, fallback_ids),
                        (probs_v, fallback_probs),
                        (s["counts_v"], fallback_counts),
                    ):
                        copy = pltpu.make_async_copy(source, dest, s["export_sem"])
                        copy.start()
                        copy.wait()

            indices.wait()
            if compact:
                probability_copy(probs_v).wait()

        pair = vmem((2 * part, hidden_size), jnp.bfloat16)
        if compact:
            # The compact path completes both metadata DMAs before expert work.
            # Reclaim the full probability array for expert scratch.
            pl.run_scoped(
                prepare, vmem(shape, jnp.int32), vmem(shape, jnp.float32), pair
            )
        else:
            pl.run_scoped(
                lambda ids, pair: prepare(ids, s["probs_v"], pair),
                vmem(shape, jnp.int32),
                pair,
            )

        def probabilities_ready():
            if not compact:
                probability_copy(s["probs_v"]).wait()

        def routed_and_epilogue(reduced):
            _routed_experts(
                latent,
                expert_gu,
                gu_scales,
                expert_down,
                down_scales,
                s["counts"],
                _FlatMetadata(s["ids"]),
                _FlatMetadata(s["probs"]),
                tables,
                s["family"],
                s["offset"],
                middle,
                reduced,
                tokens=tokens,
                probabilities_ready=probabilities_ready,
            )
            _epilogue(control, reduced, norm, latent_up, prefix, output)

        def fast():
            # The reduced rows' VMEM is not held through the prologue.
            pl.run_scoped(routed_and_epilogue, vmem((rows, 3584), jnp.bfloat16))

        if maximum is None:
            fast()
        else:

            @pl.when(s["maximum"][0] <= capacity)
            def compact_path():
                fast()

    pl.run_scoped(
        body,
        dict(
            probs_v=vmem(shape, jnp.float32) if not compact else None,
            counts_v=vmem((local_experts,), jnp.int32),
            counts=smem((local_experts,), jnp.int32),
            maximum=smem((1,), jnp.int32) if maximum is not None else None,
            ids=smem((capacity // 128, 128), jnp.int32),
            probs=smem((capacity // 128, 128), jnp.float32),
            family=smem((1,), jnp.int32),
            offset=smem((1,), jnp.int32),
            count_sem=dma,
            max_sem=dma if maximum is not None else None,
            index_sem=dma,
            prob_sem=dma,
            export_sem=dma if maximum is not None else None,
        ),
    )


_COMPOUND_PARTS: dict[int, tuple[int, ...]] = {
    6144: (4096, 2048),
    10240: (8192, 2048),
    12288: (8192, 4096),
    14336: (8192, 4096, 2048),
}
# A 16K launch whose routes overflow the compact capacity redoes the layer as
# two 8K launches (each with its own overflow path).
_OVERFLOW_PARTS: dict[int, tuple[int, ...]] = {16384: (8192, 8192)}


def make_indexed_moe_sp_ring(mesh, *, seq_len=8192):
    """Build the fused prefill MoE for a supported serving window."""
    return _build_indexed_moe_sp_ring(mesh, seq_len=seq_len, with_mixture=False)


def make_indexed_mixture_moe_sp_ring(mesh, *, seq_len=8192):
    """Build the fused residual-mixture + RMSNorm + prefill MoE."""
    return _build_indexed_moe_sp_ring(mesh, seq_len=seq_len, with_mixture=True)


def _build_indexed_moe_sp_ring(mesh, *, seq_len=8192, with_mixture=False):
    """Build the fused EP16/TP2 prefill MoE for a supported serving window.

    Inputs follow the prefill layer interface: rank-sharded normalized hidden
    states (or residual blocks + mixture/norm weights when ``with_mixture`` is
    True), residual prefix, resident weight families, valid length and family
    index. 512..8K and 16K windows run as one launch; the other compound
    windows chain 8K/4K/2K in-place launches over contiguous 64-row SP stripe
    slices.
    """
    if seq_len in _COMPOUND_PARTS:
        return _make_moe_compound(
            mesh, seq_len, _COMPOUND_PARTS[seq_len], with_mixture=with_mixture
        )
    if seq_len not in (512, 1024, 2048, 4096, 8192, 16384):
        raise ValueError(
            "Fused prefill MoE supports 512, 1024, and multiples of 2048 up to 16384"
        )
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("Fused prefill MoE requires a flat 32-rank k3_sp mesh")
    from .layer import routed_experts_prepared

    local_experts = _LOCAL_EXPERTS
    rows = seq_len // 32
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(32)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shape = (seq_len // 8, 128)
    bf16 = jnp.bfloat16
    workspace_shapes = [
        ((rows, 7168), bf16),
        ((seq_len, 2, 1792), bf16),
        ((_route_capacity(seq_len), _MIDDLE), bf16),
    ]
    output_specs = [hbm] * 3
    can_overflow = seq_len * 16 > _route_capacity(seq_len)
    if can_overflow:
        # Below 8K every route fits the compact capacity, so omit the
        # overflow outputs there.
        workspace_shapes += [
            (shape, jnp.int32),
            (shape, jnp.float32),
            ((local_experts,), jnp.int32),
            ((1,), jnp.int32),
        ]
        output_specs += [hbm] * 3 + [pl.BlockSpec()]
    counting = k._config().prefill_route_counts
    if counting:
        workspace_shapes += [((local_experts,), jnp.int32)]
        output_specs += [pl.BlockSpec(memory_space=pltpu.SMEM)]

    def body(*refs):
        refs = list(refs)
        mix_refs = None
        if with_mixture:
            blocks_ref = refs[2]
            refs[2] = None
            ffn_norm_ref = refs.pop(16)
            mlp_res_ref = refs.pop(15)
            mix_refs = (blocks_ref, mlp_res_ref, ffn_norm_ref)
        if counting:
            return _layer_kernel(
                *refs[:-1], route_counts=refs[-1], mixture_refs=mix_refs
            )
        return _layer_kernel(*refs, mixture_refs=mix_refs)

    num_hbm_inputs = 15 if with_mixture else 13
    kernel_name = (
        f"fused_mixture_moe_pallas_{seq_len}"
        if with_mixture
        else f"fused_moe_pallas_{seq_len}"
    )
    kernel = pl.pallas_call(
        body,
        out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * num_hbm_inputs,
            out_specs=tuple(output_specs),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=False,
        ),
        name=kernel_name,
    )

    def local(*args):
        if with_mixture:
            prefix, blocks, mlp_res, ffn_norm, active, families, length, slot = args
            control = jnp.concatenate(
                (length.reshape(1), slot.reshape(1), active.reshape(1))
            )
            first_arg = blocks[0]
            tail_args = (mlp_res[0], ffn_norm[0])
        else:
            hidden, prefix, families, length, slot = args
            control = jnp.concatenate((length.reshape(1), slot.reshape(1)))
            first_arg = hidden[0]
            tail_args = ()
        w = {name: value[0] for name, value in families.items()}
        outputs = kernel(
            jnp.asarray(tables),
            control,
            first_arg,
            w["router"],
            w["router_bias"],
            w["latent_down"],
            w["shared_gu"],
            w["shared_down"],
            w["expert_gu"],
            w["expert_gus"],
            w["expert_down"],
            w["expert_ds"],
            w["latent_norm"],
            w["latent_up"],
            prefix[0],
            *tail_args,
        )
        routes = None
        if counting:
            *outputs, routes = outputs
            routes = routes[None]
        if not can_overflow:
            return (outputs[0][None], routes) if counting else outputs[0][None]
        # Without the compact path's epilogue, the output holds the shared rows.
        final, latent, _, ids, probs, counts, maximum = outputs
        shared = final

        def overflow():
            if fallback is not None:
                redone = fallback(*args)
                return (redone[0] if counting else redone)[0]
            reduced = routed_experts_prepared(
                latent,
                ids.reshape(-1),
                probs.reshape(-1),
                counts,
                w["expert_gu"],
                w["expert_gus"],
                w["expert_down"],
                w["expert_ds"],
                reduce_tables=tables,
                expert_layer=slot,
                global_route_max=maximum[0],
            )
            selected = {name: w[name][slot[0]] for name in ("latent_norm", "latent_up")}
            return k._moe_sp_tail(
                reduced,
                shared,
                prefix[0],
                length,
                selected,
                axis_name="k3_sp",
                eps=1e-5,
            )

        result = jax.lax.cond(
            maximum[0] <= _route_capacity(seq_len), lambda: final, overflow
        )[None]
        return (result, routes) if counting else result

    fallback = None
    if seq_len in _OVERFLOW_PARTS:
        fallback = _compound_local(
            mesh, seq_len, _OVERFLOW_PARTS[seq_len], with_mixture=with_mixture
        )
    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=_layer_specs(with_mixture),
            out_specs=(P("k3_sp"), P("k3_sp")) if counting else P("k3_sp"),
            check_vma=False,
        )
    )


def _layer_specs(with_mixture):
    """shard_map input specs of a MoE layer op's arguments."""
    if with_mixture:  # prefix, blocks, mlp_res, ffn_norm, active, families, length, slot
        return (P("k3_sp"),) * 4 + (P(), P("k3_sp"), P(), P())
    return (P("k3_sp"), P("k3_sp"), P("k3_sp"), P(), P())


def _make_moe_compound(
    mesh, seq_len: int, part_tokens: tuple[int, ...], *, with_mixture: bool = False
):
    """The jitted compound-window MoE op (see ``_compound_local``)."""
    counting = k._config().prefill_route_counts
    return jax.jit(
        jax.shard_map(
            _compound_local(mesh, seq_len, part_tokens, with_mixture=with_mixture),
            mesh=mesh,
            in_specs=_layer_specs(with_mixture),
            out_specs=(P("k3_sp"), P("k3_sp")) if counting else P("k3_sp"),
            check_vma=False,
        )
    )


def _compound_local(
    mesh, seq_len: int, part_tokens: tuple[int, ...], *, with_mixture: bool = False
):
    """The per-rank MoE of a compound window (6K, 10K, 12K, 14K; 16K redoing an
    overflowed launch) as in-place 8K/4K/2K launches.

    Every 2048-token SP block corresponds to a contiguous 64-row slice on each
    rank. Each launch reads and writes its ``part_rows`` slice of the
    ``[window_rows, 7168]`` stream (and ``[8, window_rows, 7168]`` blocks when
    ``with_mixture`` is True) in place by aliasing the output buffer, avoiding
    any HBM split or concatenation.
    """
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("Fused prefill MoE requires a flat 32-rank k3_sp mesh")
    if sum(part_tokens) != seq_len:
        raise ValueError(f"Compound MoE parts {part_tokens} do not sum to {seq_len}")
    from .layer import routed_experts_prepared

    window_rows = seq_len // 32
    local_experts = _LOCAL_EXPERTS
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(32)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    bf16 = jnp.bfloat16
    counting = k._config().prefill_route_counts
    inputs = 15 if with_mixture else 13  # after the two scalar-prefetch operands

    tok_offsets: list[int] = []
    row_offsets: list[int] = []
    acc_tok = 0
    for p_tok in part_tokens:
        tok_offsets.append(acc_tok)
        row_offsets.append(acc_tok // 32)
        acc_tok += p_tok
    part_rows = tuple(p_tok // 32 for p_tok in part_tokens)
    can_overflow_parts = tuple(t * 16 > _route_capacity(t) for t in part_tokens)

    def make_kernel(part_idx: int):
        p_tok = part_tokens[part_idx]
        p_rows = part_rows[part_idx]
        r_off = row_offsets[part_idx]
        aliased = part_idx > 0
        can_overflow_part = can_overflow_parts[part_idx]
        p_shape = (p_tok // 8, 128)
        workspace_shapes = [
            ((window_rows, 7168), bf16),
            ((p_tok, 2, 1792), bf16),
            ((_route_capacity(p_tok), _MIDDLE), bf16),
        ]
        output_specs = [hbm] * 3
        if can_overflow_part:
            workspace_shapes += [
                (p_shape, jnp.int32),
                (p_shape, jnp.float32),
                ((local_experts,), jnp.int32),
                ((1,), jnp.int32),
            ]
            output_specs += [hbm] * 3 + [pl.BlockSpec()]
        if counting:
            workspace_shapes += [((local_experts,), jnp.int32)]
            output_specs += [pl.BlockSpec(memory_space=pltpu.SMEM)]
        kernel_name = (
            f"fused_mixture_moe_pallas_{p_tok}"
            if with_mixture
            else f"fused_moe_pallas_{p_tok}"
        )

        def body(*refs):
            refs = list(refs)
            if aliased:
                del refs[2 + inputs]  # previous launch's output, aliased below
            own = pl.ds(r_off, p_rows)
            mix_refs = None
            if with_mixture:
                blocks_ref = refs[2].at[:, own]
                refs[2] = None
                refs[14] = refs[14].at[own]
                refs[2 + inputs] = refs[2 + inputs].at[own]
                ffn_norm_ref = refs.pop(16)
                mlp_res_ref = refs.pop(15)
                mix_refs = (blocks_ref, mlp_res_ref, ffn_norm_ref)
            else:
                for index in (2, 1 + inputs, 2 + inputs):  # hidden, prefix, output
                    refs[index] = refs[index].at[own]
            if counting:
                return _layer_kernel(
                    *refs[:-1], route_counts=refs[-1], mixture_refs=mix_refs
                )
            return _layer_kernel(*refs, mixture_refs=mix_refs)

        return pl.pallas_call(
            body,
            out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=(hbm,) * (inputs + aliased),
                out_specs=tuple(output_specs),
            ),
            input_output_aliases={2 + inputs: 0} if aliased else {},
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                vmem_limit_bytes=64 * 1024**2,
                disable_bounds_checks=True,
                shape_invariant_numerics=False,
            ),
            name=kernel_name,
        )

    kernels = tuple(make_kernel(i) for i in range(len(part_tokens)))
    def local(*args):
        if with_mixture:
            prefix, blocks, mlp_res, ffn_norm, active, families, length, slot = args
            first_arg = blocks[0]
            tail_args = (mlp_res[0], ffn_norm[0])
        else:
            hidden, prefix, families, length, slot = args
            active = None
            first_arg = hidden[0]
            tail_args = ()
        w = {name: value[0] for name, value in families.items()}
        operands = (
            first_arg,
            w["router"],
            w["router_bias"],
            w["latent_down"],
            w["shared_gu"],
            w["shared_down"],
            w["expert_gu"],
            w["expert_gus"],
            w["expert_down"],
            w["expert_ds"],
            w["latent_norm"],
            w["latent_up"],
            prefix[0],
            *tail_args,
        )
        lengths = tuple(
            jnp.clip(length - t_off, 0, p_tok)
            for p_tok, t_off in zip(part_tokens, tok_offsets, strict=True)
        )
        parts_out, routes, previous = [], None, ()
        for idx, kernel in enumerate(kernels):
            if with_mixture:
                assert active is not None
                control = jnp.concatenate(
                    (lengths[idx].reshape(1), slot.reshape(1), active.reshape(1))
                )
            else:
                control = jnp.concatenate((lengths[idx].reshape(1), slot.reshape(1)))
            outputs = kernel(jnp.asarray(tables), control, *operands, *previous)
            if counting:
                *outputs, counted = outputs
                routes = counted if routes is None else routes + counted
            parts_out.append(outputs)
            previous = (outputs[0],)
        final = parts_out[-1][0]
        if not any(can_overflow_parts):
            if counting:
                assert routes is not None
                return final[None], routes[None]
            return final[None]

        def overflow(idx: int):
            r_off, p_rows = row_offsets[idx], part_rows[idx]
            # The part's rows of the output hold its shared rows (no epilogue).
            _, latent, _, ids, probs, counts, maximum = parts_out[idx]
            shared = final[r_off : r_off + p_rows]
            reduced = routed_experts_prepared(
                latent,
                ids.reshape(-1),
                probs.reshape(-1),
                counts,
                w["expert_gu"],
                w["expert_gus"],
                w["expert_down"],
                w["expert_ds"],
                reduce_tables=tables,
                expert_layer=slot,
                global_route_max=maximum[0],
            )
            selected = {name: w[name][slot[0]] for name in ("latent_norm", "latent_up")}
            return k._moe_sp_tail(
                reduced,
                shared,
                prefix[0][r_off : r_off + p_rows],
                lengths[idx],
                selected,
                axis_name="k3_sp",
                eps=1e-5,
            )

        fits = [
            (parts_out[i][6][0] <= _route_capacity(part_tokens[i]))
            if can_overflow_parts[i]
            else jnp.bool_(True)
            for i in range(len(part_tokens))
        ]
        all_fit = fits[0]
        for fit in fits[1:]:
            all_fit = all_fit & fit

        def fallback():
            slices = [
                jax.lax.cond(
                    fits[i],
                    lambda r=row_offsets[i], pr=part_rows[i]: final[r : r + pr],
                    lambda i=i: overflow(i),
                )
                if can_overflow_parts[i]
                else final[row_offsets[i] : row_offsets[i] + part_rows[i]]
                for i in range(len(part_tokens))
            ]
            return jnp.concatenate(slices, axis=0)

        result = jax.lax.cond(all_fit, lambda: final, fallback)[None]
        if counting:
            assert routes is not None
            return result, routes[None]
        return result

    return local
