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


def _sort_route_steps(keys, probabilities, *, size):
    """Yield after each 4096-route window of the in-place bitonic network.

    Keys encode (local expert, original route position), so all live keys are
    unique and the result has deterministic expert-major/token-major order.
    Probability values follow exactly the same swaps as their keys.
    """
    window, rows = 4096, 32
    assert size >= window and size & (size - 1) == 0
    shape = (rows, 128)
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    positions = row * 128 + lane

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

    pending = []

    def flush():
        if not pending:
            return
        for ordinal in range(size // window):
            kr, pr = view(keys, ordinal), view(probabilities, ordinal)
            x, p = kr[...], pr[...]
            global_position = positions + ordinal * window
            for distance, group in pending:
                y, q = partner(x, distance), partner(p, distance)
                descending = (global_position & group) != 0
                upper = (global_position & distance) != 0
                take_min = ~(descending ^ upper)
                keep = ~((x < y) ^ take_min)
                x, p = jnp.where(keep, x, y), jnp.where(keep, p, q)
            kr[...], pr[...] = x, p
            yield
        pending.clear()

    for stage in range(1, size.bit_length()):
        for shift in reversed(range(stage)):
            distance, group = 1 << shift, 1 << stage
            if distance < window:
                pending.append((distance, group))
                continue
            yield from flush()
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
        rows, experts = logits.shape
        self.raw = 1.0 / (1.0 + jnp.exp(-logits))
        self.scores = self.raw + bias
        self.experts = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        self.slots = jax.lax.broadcasted_iota(jnp.int32, (rows, 16), 1)
        self.selected = jnp.zeros((rows, 16), jnp.int32)
        self.probabilities = jnp.zeros((rows, 16), jnp.float32)
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
        self.selected = jnp.where(self.slots == self.index, winner, self.selected)
        self.probabilities = jnp.where(
            self.slots == self.index, probability, self.probabilities
        )
        self.scores = jnp.where(chosen, -jnp.inf, self.scores)
        self.index += 1

    def finish(self, valid):
        while self.index < 16:
            self.step()
        denominator = jnp.maximum(
            jnp.sum(self.probabilities, axis=1, keepdims=True), 1e-30
        )
        return self.selected, jnp.where(valid, self.probabilities / denominator, 0.0)


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
):
    """Build stable local routing metadata, yielding one prep/sort chunk at a time.

    The all-gathered wire contains 16 expert IDs followed by 16 probability bit
    patterns per token. Every rank counts all 16 expert groups at 8K, making the
    compact-capacity decision uniform before any conditional collective runs.
    """
    bits = (tokens * 16 - 1).bit_length()
    shape = (128, 16) if tokens < 4096 else (16, 128)
    expert_index = jnp.arange(56, dtype=jnp.int32)
    group_index = jnp.arange(16, dtype=jnp.int32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    total_counts = jnp.zeros((56,), jnp.int32)
    total_groups = jnp.zeros((16,), jnp.int32)
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
        local = ids - (rank // 2 * 56)
        valid = (probs != 0) & (local >= 0) & (local < 56)
        local = jnp.where(valid, local, 56)
        original_position = chunk * 2048 + rows * shape[1] + lanes
        packed = (local << bits) | original_position
        keys[pl.ds(chunk * 16, 16), :] = packed.reshape(16, 128)
        probabilities[pl.ds(chunk * 16, 16), :] = probs.reshape(16, 128)
        # Count routes on the packed 16x128 layout.
        local_flat = local.reshape(16, 128)
        ids_flat = ids.reshape(16, 128)
        live_flat = probs.reshape(16, 128) != 0
        tile_counts = jnp.zeros((56,), jnp.int32)
        for expert in range(56):
            count = jnp.sum((local_flat == expert).astype(jnp.int32))
            tile_counts = jnp.where(expert_index == expert, count, tile_counts)
        total_counts += tile_counts
        if tokens == 8192:
            tile_groups = jnp.zeros((16,), jnp.int32)
            groups = ids_flat // 56
            for group in range(16):
                count = jnp.sum((live_flat & (groups == group)).astype(jnp.int32))
                tile_groups = jnp.where(group_index == group, count, tile_groups)
            total_groups += tile_groups
        yield
    counts[...] = total_counts
    if maximum is not None:
        maximum[...] = jnp.max(total_groups)[None]
    counts_ready()
    yield from _sort_route_steps(keys, probabilities, size=tokens * 16)

    @pl.loop(0, tokens * 16 // 4096)
    def extract(chunk):
        index = pl.ds(chunk * 32, 32)
        token_ids[index, :] = (keys[index, :] & ((1 << bits) - 1)) // 16


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


def _latent_ready_init(tables, payload, output, send, recv, stores, state, meta, *, rank):
    """Seed twelve independently progressing streams of the 8K latent gather.

    Cache neighbor IDs and owner row offsets in SMEM once. Each stream tracks
    its hop and receive/forward/complete phase; state[0] counts active streams.
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
            meta[directed * 34 + 2 + step] = owner * rows

    pltpu.make_async_copy(payload, output.at[pl.ds(rank * rows, rows), :, :], stores).start()
    state[0] = 12
    begin = 0
    for stream in range(12):
        size = rows // 12 + (stream < rows % 12)
        part = pl.ds(begin, size)
        pltpu.make_async_remote_copy(
            payload.at[part, :, :],
            output.at[pl.ds(rank * rows + begin, size), :, :],
            send.at[stream],
            recv.at[0, stream],
            device_id=meta[(stream % 6) * 34],
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).start()
        state[1 + stream] = 1
        state[13 + stream] = 0
        begin += size


def _latent_ready_progress(
    payload, output, send, recv, credits, stores, state, meta, *, rank, drain=False
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
        pltpu.make_async_copy(payload, output.at[pl.ds(rank * rows, rows), :, :], stores).wait()
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
    hidden,
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
    counts_ready,
    metadata_ready,
):
    """Router, latent projection and shared experts, with incremental collectives.

    The caller retains shared/metadata outputs across this scope. All temporary
    projection and communication buffers die before routed-expert scratch is
    allocated. The initial HBM input load targets our half of the peer buffer;
    both the router and shared expert use this same copy.
    """
    rows, hidden_size = hidden.shape
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
        own = s["pair"].at[pl.ds(parity * rows, rows), :]
        copy = pltpu.make_async_copy(hidden, own, s["input_sem"])
        copy.start()
        copy.wait()
        if rows <= 128:
            copy = pltpu.make_async_copy(router, s["router"], s["input_sem"])
            copy.start()
            copy.wait()
        copy = pltpu.make_async_copy(bias, s["bias"], s["input_sem"])
        copy.start()
        copy.wait()
        _enter_mesh()
        pl.semaphore_signal(
            s["peer_credit"], 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
        )
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
        if rows == 256:
            buffers = tuple(s["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4))
            router_v = s["router_gu"].at[:, :896]
            copy = pltpu.make_async_copy(router, router_v, s["gu_sems"][0])
            copy.start()
            copy.wait()
        else:
            buffers, router_v = s["gu_buffers"], s["router"]
        gate_buffers, up_buffers = buffers[:2], buffers[2:]
        gate_sems, up_sems = s["gu_sems"][:2], s["gu_sems"][2:]
        hx = own[...]
        logits = jnp.matmul(hx, router_v[...], preferred_element_type=jnp.float32)
        top = _Top16(logits, s["bias"][...])
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
                )

        def metadata_progress():
            for _ in range(2 if rows <= 128 else 6):
                next(metadata_steps, None)

        if rows == 256:
            latent_buffers = tuple(
                s["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4)
            )
            latent_products = _latent_half_pair_panels(
                hx, latent_weight, latent_buffers, s["gu_sems"]
            )
        else:
            latent_products = _dense_panels(
                hx, latent_weight, gate_buffers, gate_sems, columns=4096
            )
        first_half = None
        for col, product in latent_products:
            if route_started:
                stage = next(route_steps, None)
                if rows <= 128 and stage is None:
                    start_metadata()
            if rows <= 128 and metadata_started:
                metadata_progress()
            if rows == 256:
                fragment = product.astype(jnp.bfloat16)
                if col < 1792:
                    first_half = fragment
                else:
                    # Assemble both halves before writing their packed words.
                    assert first_half is not None
                    paired = jnp.concatenate((first_half, fragment), axis=1)
                    s["latent_payload"][:, :, pl.ds(col - 1792, 256)] = paired.reshape(
                        rows, 2, 256
                    )
            else:
                s["latent"][:, pl.ds(col, 256)] = product.astype(jnp.bfloat16)
            for _ in range(min(2, 16 - top.index)):
                top.step()
            if top.index == 16 and not route_started:
                ids, probs = top.finish(valid)
                if rows < 128:
                    wire = jnp.concatenate(
                        (ids, jax.lax.bitcast_convert_type(probs, jnp.int32)), axis=1
                    )
                else:
                    wire = jnp.concatenate(
                        (
                            ids.reshape(rows // 8, 128),
                            jax.lax.bitcast_convert_type(probs, jnp.int32).reshape(
                                rows // 8, 128
                            ),
                        ),
                        axis=0,
                    )
                s["route_payload"][...] = wire.reshape(rows // 4, 128)
                route_steps = _route_gather_steps(
                    s["route_payload"],
                    s["route_wire"],
                    s["route_send"],
                    s["route_recv"],
                    rank=rank,
                )
                next(route_steps, None)
                route_started = True
        for _ in route_steps:
            pass
        start_metadata()
        if rows <= 128:
            s["latent_payload"][...] = s["latent"][:, :3584].reshape(rows, 2, 1792)
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
        if rows == 256:
            latent_tick = 0
            def latent_progress(*, drain=False):
                nonlocal latent_tick
                if latent_tick == 0:
                    _latent_ready_init(tables, s["latent_payload"], latent_hbm,
                        s["latent_send_sems"], s["latent_recv_sems"], s["latent_stores"],
                        s["latent_state"], s["latent_meta"], rank=rank)
                else:
                    _latent_ready_progress(s["latent_payload"], latent_hbm,
                        s["latent_send_sems"], s["latent_recv_sems"], s["latent_credits"],
                        s["latent_stores"], s["latent_state"], s["latent_meta"], rank=rank,
                        drain=drain)
                latent_tick += 1
        else:
            def latent_progress(*, drain=False):
                if drain:
                    for _ in latent_steps:
                        pass
                else:
                    next(latent_steps, None)
        peer_input.wait_send()
        peer_input.wait_recv()

        def gu_copy(panel, offset, buffers, sems):
            bank = panel % 2
            return pltpu.make_async_copy(
                shared_gu.at[:, pl.ds(offset + panel * 256, 256)],
                buffers[bank],
                sems[bank],
            )

        gu_copy(0, 0, gate_buffers, gate_sems).start()
        gu_copy(0, 3072, up_buffers, up_sems).start()
        if rows == 256:
            latent_progress()
        for panel in range(12):
            if rows <= 128:
                latent_progress()
            bank = panel % 2
            gu_copy(panel, 0, gate_buffers, gate_sems).wait()
            gu_copy(panel, 3072, up_buffers, up_sems).wait()
            if panel + 1 < 12:
                gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
                gu_copy(panel + 1, 3072, up_buffers, up_sems).start()
            for row in (0, rows):
                lhs = s["pair"][pl.ds(row, rows), :]
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
                s["middle"][pl.ds(row, rows), pl.ds(panel * 256, 256)] = (
                    activated.astype(jnp.bfloat16)
                )
                if rows == 256:
                    latent_progress()

        def down_copy(panel):
            return pltpu.make_async_copy(
                shared_down.at[:, pl.ds(panel * 256, 256)],
                s["down_buffers"][panel % 2],
                s["down_sems"][panel % 2],
            )

        def finish_reduction(pending):
            transfer, bank, panel = pending
            own = s["partial"][bank][pl.ds(parity * rows, rows), :]
            transfer.wait_send()
            transfer.wait_recv()
            shared[:, pl.ds(panel * 256, 256)] = (own + s["peer_y"][bank][...]).astype(
                jnp.bfloat16
            )

        down_copy(0).start()
        pending = None
        for panel in range(28):
            bank = panel % 2
            down_copy(panel).wait()
            if panel + 1 < 28:
                down_copy(panel + 1).start()
            for row in (0, rows):
                product = jnp.matmul(
                    s["middle"][pl.ds(row, rows), :],
                    s["down_buffers"][bank][...],
                    preferred_element_type=jnp.float32,
                )
                metadata_progress()
                s["partial"][bank][pl.ds(row, rows), :] = product
                if row == rows:
                    if pending is not None:
                        finish_reduction(pending)
                    transfer = pltpu.make_async_remote_copy(
                        s["partial"][bank].at[pl.ds((1 - parity) * rows, rows), :],
                        s["peer_y"][bank],
                        s["reduce_send"][bank],
                        s["reduce_recv"][bank],
                        device_id=peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                    transfer.start()
                    pending = transfer, bank, panel
                if rows == 256 or row == rows:
                    latent_progress()
        finish_reduction(pending)
        if rows == 256:
            for _ in metadata_steps:
                latent_progress()
            # Start compact VMEM-to-SMEM transfers while the latent ring finishes.
            metadata_ready()
            latent_progress(drain=True)
        else:
            latent_progress(drain=True)
            for _ in metadata_steps:
                pass

    scratch = dict(
        pair=vmem((2 * rows, hidden_size), jnp.bfloat16),
        router=vmem((hidden_size, 896), jnp.bfloat16) if rows <= 128 else None,
        router_gu=vmem((hidden_size, 1024), jnp.bfloat16) if rows == 256 else None,
        gu_buffers=tuple(vmem((hidden_size, 256), jnp.bfloat16) for _ in range(4))
        if rows <= 128
        else (),
        gu_sems=tuple(dma for _ in range(4)),
        bias=vmem((1, 896), jnp.float32),
        input_sem=dma,
        peer_send=dma,
        peer_recv=dma,
        peer_credit=pltpu.SemaphoreType.REGULAR,
        latent=vmem((rows, 4096), jnp.bfloat16) if rows <= 128 else None,
        route_payload=vmem((rows // 4, 128), jnp.int32),
        route_wire=vmem((rows * 8, 128), jnp.int32),
        route_send=dma((5,)),
        route_recv=dma((5,)),
        keys=vmem((rows * 4, 128), jnp.int32),
        latent_payload=vmem((rows, 2, 1792), jnp.bfloat16),
        latent_recv=vmem((2, rows, 2, 1792), jnp.bfloat16) if rows <= 128 else None,
        latent_state=pltpu.SMEM((25,), jnp.int32) if rows == 256 else None,
        latent_meta=pltpu.SMEM((6 * 34,), jnp.int32) if rows == 256 else None,
        latent_send_sems=dma((12,)),
        latent_recv_sems=dma((2, 12)),
        latent_credits=pltpu.SemaphoreType.REGULAR((12,)),
        latent_stores=dma((12,)) if rows <= 128 else dma,
        middle=vmem((2 * rows, 3072), jnp.bfloat16),
        down_buffers=tuple(vmem((3072, 256), jnp.bfloat16) for _ in range(2)),
        down_sems=(dma, dma),
        partial=tuple(vmem((2 * rows, 256), jnp.float32) for _ in range(2)),
        peer_y=tuple(vmem((rows, 256), jnp.float32) for _ in range(2)),
        reduce_send=(dma, dma),
        reduce_recv=(dma, dma),
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
    """Joint GU/SiTU/first-down pipeline followed by remaining columns."""
    columns = 1792 if tokens < 8192 else 896
    tile = 128 if tokens < 8192 else 256
    unroll = 8 if tokens < 8192 else 32
    gu_unroll = 4 if tokens < 8192 else 8
    first_down_unroll = 2 if tokens < 8192 else 8
    metadata_row_capacity = 0 if tokens < 8192 else 32768
    ring_dtype = jnp.float32
    assert tokens in (512, 1024, 2048, 4096, 8192)
    x = jax.ShapeDtypeStruct((tokens * 16, 1536), latent.dtype)
    rows, width = (metadata_row_capacity or x.shape[0], rhs.shape[-1])
    assert rows <= x.shape[0]
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
        tiles=dataclasses.replace(gu_cfg.tiles, tile_m=tm, tile_n=1536, tile_k=3584),
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
        cast,
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
        gu = dataclasses.replace(
            gu,
            weight=gu.weight.at[layer[0]].bitcast(jnp.float4_e2m1fn),
            scale=gu.scale.at[layer[0]],
        )
        gu = k.FusedWeightsRef(gate=gu, up=jax.tree.map(lambda r: r.at[..., 1536:], gu))
        # Six directed rings: [target, source, 32 local-contribution row bases].
        # One initialization per layer call, reused across all four columns.
        rank = lax.axis_index("k3_sp")
        for directed in range(6):
            family, direction = directed // 2, -1 if directed % 2 == 0 else 1
            position = tables[family, 0, rank]
            ring_meta[directed * 34] = tables[family, 1, (position + direction) & 31]
            ring_meta[directed * 34 + 1] = tables[family, 1, (position - direction) & 31]
            @pl.loop(0, 32)
            def hop(step, *, family=family, position=position, direction=direction, directed=directed):
                owner_mesh = tables[family, 1, (position - direction * (step + 1)) & 31]
                owner = tables[family, 2, owner_mesh]
                ring_meta[directed * 34 + 2 + step] = owner * localrows

        active_ng = k.fill_metadata(counts, offset, metadata, cfgs=cfg)

        @pl.when(active_ng == 0)
        def empty_metadata():
            metadata.gm_id_to_group_id[0] = 0
            metadata.gm_id_to_m_offset[0] = 0
            metadata.gm_id_to_m_offset[1] = 0

        ng = jnp.maximum(active_ng, 1)
        (xs, ws), _ = k.generate_block_specs(metadata, cfg)
        state[0] = 0

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
                cast[...] = values[:, :tn].astype(jnp.bfloat16)
                subtotal_vmem.at[pl.ds(b * store_tile, store_tile)][...] = cast[...]

        def route(i):
            return (destinations[i], probabilities[i])

        def consume(tile_g, source, first=0, *, loop_unroll=unroll):
            start = metadata.gm_id_to_m_offset[tile_g]
            end = metadata.gm_id_to_m_offset[tile_g + 1]
            begin = start // sub * sub
            start = jnp.minimum(start + first, end)

            def update(i):
                dest, weight = route(i)
                value = source.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
                target = sums.at[pl.ds(dest, 1)]
                target[...] = target[...] + value * weight

            full = (end - start) // loop_unroll

            @pl.loop(0, full)
            def chunk(b):
                for u in range(loop_unroll):
                    update(start + b * loop_unroll + u)

            @pl.loop(start + full * loop_unroll, end)
            def tail(i):
                update(i)

        gu_panels = pl.cdiv(3072, pltpu.get_tpu_info().mxu_column_size)
        moved_rows = gu_panels * gu_unroll

        def gu_work(g, panel):
            previous_g = jnp.maximum(g - 1, 0)
            start = metadata.gm_id_to_m_offset[previous_g]
            end = metadata.gm_id_to_m_offset[previous_g + 1]
            begin = start // sub * sub
            for row in range(gu_unroll):
                offset = panel * gu_unroll + row
                i = jnp.minimum(start + offset, end - 1)
                dest, prob = route(i)
                weight = jnp.where((g > 0) & (start + offset < end), prob, 0)
                previous = packed.at[(g + 1) % 2]
                value = previous.at[pl.ds(i - begin, 1)][...].astype(jnp.float32)
                target = sums.at[pl.ds(dest, 1)]
                target[...] = target[...] + value * weight

        def inner(x, w, metadata, *, first=False, subtotal_vmem=None):
            g, n = (pl.program_id(1), 0 if first else pl.program_id(0) + 1)
            if not first:

                @pl.when(g == 0)
                def clear():
                    sums[...] = jnp.zeros(sums.shape, jnp.float32)

            if not first:
                advance(jnp.minimum(32, (g + 1) * 32 // ng), n - 1, subtotal_vmem, poll_round=g)
            current = packed.at[g % 2]
            previous = packed.at[(g + 1) % 2]
            previous_g = jnp.maximum(g - 1, 0)
            panel_count = pl.cdiv(tn, pltpu.get_tpu_info().mxu_column_size)
            down_unroll = first_down_unroll if first else unroll

            def overlap_work(panel):
                start = metadata.gm_id_to_m_offset[previous_g]
                end = metadata.gm_id_to_m_offset[previous_g + 1]
                begin = start // sub * sub

                def work():
                    for row in range(down_unroll):
                        offset = (
                            (moved_rows if first else 0) + panel * down_unroll + row
                        )
                        i = jnp.minimum(start + offset, end - 1)
                        dest, prob = route(i)
                        weight = jnp.where((g > 0) & (start + offset < end), prob, 0)
                        value = previous.at[pl.ds(i - begin, 1)][...].astype(
                            jnp.float32
                        )
                        target = sums.at[pl.ds(dest, 1)]
                        target[...] = target[...] + value * weight

                work()

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
                )

            @pl.when(g > 0)
            def previous_tail():
                consume(
                    previous_g,
                    previous,
                    (moved_rows if first else 0) + panel_count * down_unroll,
                    loop_unroll=8 if first else unroll,
                )

            @pl.when(g == ng - 1)
            def finish():
                consume(g, current, loop_unroll=8 if first else unroll)
                if not first:
                    advance(jnp.int32(32), n - 1, subtotal_vmem, drain=True)
                    emit_subtotals(subtotal_vmem)
                    state[0] = 0

                    @pl.when(n == width // tn - 1)
                    def last():
                        advance(jnp.int32(32), n, subtotal_vmem, drain=True)

        (_, gu_ws), gu_ys = k.generate_block_specs(metadata, gu_cfg)
        gu_ws = k.FusedWeightsRef(gate=gu_ws, up=gu_ws)

        @functools.partial(
            pl.run_scoped,
            gu_partial=pltpu.VMEM((sub, 1536), gu_cfg.out_dtype),
            acts=pltpu.VMEM((2, tm, 2, 1792), latent.dtype),
            act_sems=pltpu.SemaphoreType.DMA((2,)),
        )
        def first_column(gu_partial, acts, act_sems):
            def buffering(spec):
                return dataclasses.replace(
                    spec, pipeline_mode=pl.Buffered(buffer_count=2)
                )

            joint_gu_ws, joint_down_ws = jax.tree.map(
                buffering, (gu_ws, ws), is_leaf=lambda v: isinstance(v, pl.BlockSpec)
            )

            def bounds(g):
                begin = metadata.gm_id_to_m_offset[g] // sub * sub
                end = pl.cdiv(metadata.gm_id_to_m_offset[g + 1], sub) * sub
                return (begin, end - begin)

            def start(g, bank):
                begin, nrows = bounds(g)

                @pl.loop(0, nrows)
                def row(i):
                    source = destinations[begin + i]
                    pltpu.make_async_copy(
                        latent.at[pl.ds(source, 1)],
                        acts.at[bank, pl.ds(i, 1)],
                        act_sems.at[bank],
                    ).start()

            def joint(gw, dw, gy, metadata, gu_partial, acts, act_sems):
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

                @pl.when(active_ng > 0)
                def gate_up():
                    @pl.when(g == 0)
                    def initial():
                        start(g, bank)

                    _, nrows = bounds(g)
                    active = acts.at[bank, pl.ds(0, nrows)]
                    pltpu.make_async_copy(active, active, act_sems.at[bank]).wait()

                    @pl.when(g + 1 < ng)
                    def following():
                        start(g + 1, 1 - bank)

                    _gu_inner(
                        _PackedRows(acts.at[bank]),
                        gw,
                        gy,
                        gu_partial,
                        metadata,
                        cfgs=gu_cfg,
                        overlap_work=lambda panel: gu_work(g, panel),
                    )

                inner(gy, dw, metadata, first=True)

            specs = dict(
                grid=(1, ng, 1),
                in_specs=(joint_gu_ws, joint_down_ws),
                out_specs=gu_ys,
            )
            refs = (gu, w, middle.reshape(-1, sub, 1536))
            scratches = (metadata, gu_partial, acts, act_sems)
            if tokens == 8192:
                # The 8K schedule already fills VMEM; retain its compact buffers.
                pltpu.emit_pipeline(joint, **specs)(*refs, scratches=scratches)
                return

            # Skew the physical GU weight/scale row stride to reduce concurrent
            # VMEM load conflicts. Logical tiles and arithmetic stay unchanged.
            pipeline, allocate = pltpu.emit_pipeline_with_allocations(joint, **specs)
            allocations = allocate(*refs)
            logical_gu = allocations[0]

            def is_buffer(value):
                return hasattr(value, "window_ref")

            def pad(buffer):
                window = buffer.window_ref
                if window is None:
                    return buffer
                return dataclasses.replace(
                    buffer,
                    window_ref=pltpu.VMEM(
                        (*window.shape[:-1], window.shape[-1] + 256), window.dtype
                    ),
                )

            padded_gu = jax.tree.map(pad, logical_gu, is_leaf=is_buffer)

            def run_pipeline(buffers):
                def trim(buffer, original):
                    if buffer.window_ref is None:
                        return buffer
                    return dataclasses.replace(
                        buffer,
                        window_ref=buffer.window_ref.at[
                            ..., pl.ds(0, original.window_ref.shape[-1])
                        ],
                    )

                selected = (
                    jax.tree.map(trim, buffers[0], logical_gu, is_leaf=is_buffer),
                    *buffers[1:],
                )
                pipeline(
                    *refs,
                    allocations=selected,
                    scratches=scratches,
                )

            pl.run_scoped(run_pipeline, (padded_gu, *allocations[1:]))

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

            def remaining(x, w, metadata):
                inner(x, w, metadata, subtotal_vmem=subtotal_vmem)

            pltpu.emit_pipeline(
                remaining,
                grid=(width // tn - 1, ng, 1),
                in_specs=(xs, shifted_ws),
                out_specs=(),
            )(middle.reshape(-1, sub, 1536), w, scratches=(metadata,))

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
        sums=pltpu.VMEM((tokens, 2, packed_half), jnp.float32),
        cast=pltpu.VMEM((store_tile, tn), x.dtype),
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


def _column_rhs(rhs, first, size):
    """Slice references before dequantization to bound panel temporaries."""

    def slice_ref(ref, first, size):
        assert ref.scale is not None
        return dataclasses.replace(
            ref,
            weight=ref.weight.at[:, pl.ds(first, size)],
            scale=ref.scale.at[..., pl.ds(first, size)],
        )

    if isinstance(rhs, k.FusedWeightsRef):
        # Gate/up columns alternate in lane-sized chunks. A complete pair
        # maps to half as many columns in each underlying reference.
        pair = 2 * pltpu.get_tpu_info().num_lanes
        assert first % pair == 0 and size % pair == 0
        return k.FusedWeightsRef(
            slice_ref(rhs.gate, first // 2, size // 2),
            slice_ref(rhs.up, first // 2, size // 2),
        )
    return slice_ref(rhs, first, size)


def _expert_product(lhs, rhs, *, cfgs, bucket_rows, overlap_work):
    """Dequantize MXFP4 weights, panel-locally for the 8K row tile.

    Both routed projections cover their complete reduction dimension in one
    tile. The callback consumes rows of the previous expert tile between MXU
    panels; it must not read this product's result.
    """
    assert cfgs.dims.size_k == cfgs.tiles.tile_k
    assert not cfgs.rhs_cfgs.has_bias
    columns = pltpu.get_tpu_info().mxu_column_size
    activations = lhs.reshape(-1, cfgs.tiles.tile_k)[:bucket_rows]
    width = (
        2 * rhs.gate.weight.shape[-1]
        if isinstance(rhs, k.FusedWeightsRef)
        else rhs.weight.shape[-1]
    )
    block = cfgs.rhs_cfgs.quant_block_size
    # The 8K window uses a 256-row tile. Smaller windows use 128 rows;
    # their whole-tile conversion schedules better than panel-local loads.
    panel_local = cfgs.tiles.tile_m == 256
    weights = None
    if not panel_local:
        weights = rhs.get_weight()
        scales = rhs.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
        weights = weights.astype(cfgs.lhs_cfgs.dtype).reshape(
            cfgs.num_quant_blocks_per_tile_k, block, width
        )
        weights = (weights * scales).reshape(cfgs.tiles.tile_k, width)
    panels = []
    for first in range(0, width, columns):
        panel_width = min(columns, width - first)
        if panel_local:
            panel_rhs = _column_rhs(rhs, first, panel_width)
            panel_weights = panel_rhs.get_weight().astype(cfgs.lhs_cfgs.dtype).reshape(
                cfgs.num_quant_blocks_per_tile_k, block, panel_width
            )
            scales = panel_rhs.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
            panel_weights = (panel_weights * scales).reshape(
                cfgs.tiles.tile_k, panel_width
            )
        else:
            assert weights is not None
            panel_weights = weights[:, first:first + panel_width]
        accumulator = jnp.zeros((bucket_rows, panel_width), cfgs.acc_dtype)
        product = jnp.matmul(
            activations, panel_weights, preferred_element_type=jnp.float32
        ).astype(cfgs.acc_dtype)
        overlap_work(first // columns)
        accumulator += product
        panels.append(accumulator)
    return jnp.concatenate(panels, axis=1)


def _gu_inner(lhs, rhs, output, partial, metadata, *, cfgs, overlap_work):
    """GU/SiTU with the production row buckets and adjacent-expert carry.

    Expert boundaries need not align with the 16-row BF16 sublane. Mask outside
    the active expert, then preserve the preceding expert's partial first row.
    """
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    first, last = start - aligned, end - aligned

    def product(bucket_index):
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        value = _expert_product(
            lhs, rhs, cfgs=cfgs, bucket_rows=rows, overlap_work=overlap_work
        )
        value = k.apply_act_fn(value, cfgs.fuse_act)
        row = lax.broadcasted_iota(jnp.int32, value.shape, 0)
        value = jnp.where((first <= row) & (row < last), value, 0)
        output.reshape(-1, cfgs.tiles.tile_n)[:rows] = value.astype(output.dtype)
        zeros = jnp.zeros_like(partial)
        output[0] += jnp.where(group == 0, zeros, partial[...])
        last_row = last // cfgs.dims.size_lhs_sublane
        partial[...] = jnp.where(
            last % cfgs.dims.size_lhs_sublane == 0, zeros, output[last_row]
        )

    # At 8K, share the 128-row product for both small buckets.
    # Keep route order, trading padded work for fewer instruction overlays.
    first_bucket = 1 if cfgs.tiles.tile_m == 256 else 0
    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(first_bucket, cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(jnp.maximum(last // cfgs.tiles.bucket_base - first_bucket, 0), branches)


def _down_inner(lhs, rhs, output, metadata, *, cfgs, overlap_work):
    """Down projection stored as packed rows for token-wise VALU combine.

    Each valid expert row is consumed individually, so this output needs no
    adjacent-expert carry. BF16 rounding precedes FP32 routing weighting.
    """
    assert cfgs.fuse_act is None
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    aligned = start - start % cfgs.dims.size_lhs_sublane
    last = end - aligned

    # GU already masks inactive rows; consume() visits only live routes.
    def product(bucket_index):
        rows = cfgs.tiles.bucket_base * (bucket_index + 1)
        with jax.named_scope(f"bm{rows}_fragment"):
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
    lax.switch(last // cfgs.tiles.bucket_base, branches)


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
                local_value() + received.at[(step - 1) % buffers, rows, cols][...]
            )
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
        peers = [(ring_meta[d * 34],) for d in range(6)]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(ready, len(peers))

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
            units = size * width * 4 // 32
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
                        x_ref.at[source_rows, cols][...].astype(payload.dtype) + received.at[bank, rows, cols][...]
                    )
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


def _epilogue(control, reduced, shared, norm_weight, up_weight, prefix, output):
    """Normalize routed output and project up with three weight/prefix banks.

    Reduced FP32 rows remain in VMEM. Preserve the BF16 boundaries after
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
        panels = 7168 // 512

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

        def output_copy(panel):
            return pltpu.make_async_copy(
                s["tiles"][panel % 3],
                output.at[:, pl.ds(panel * 512, 512)],
                s["output_sems"][panel % 3],
            )

        # Hide the first weight/residual loads behind the RMS reduction.
        for panel in (0, 1):
            weight_copy(panel).start()
            prefix_copy(panel).start()
        routed = reduced[...]
        mean = jnp.sum(routed * routed, axis=1, keepdims=True) / width
        inverse = jax.lax.rsqrt(mean + 1e-5)
        s["normalized"][...] = ((routed * inverse) * s["norm"][...]).astype(
            jnp.bfloat16
        )
        for panel in range(panels):
            bank = panel % 3
            weight_copy(panel).wait()
            prefix_copy(panel).wait()
            if panel + 2 < panels:
                if panel >= 1:
                    output_copy(panel - 1).wait()
                weight_copy(panel + 2).start()
                prefix_copy(panel + 2).start()
            product = jnp.matmul(
                s["normalized"][...],
                s["weights"][bank][...],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            value = product + shared[:, pl.ds(panel * 512, 512)]
            value = jnp.where(valid, value, 0)
            s["tiles"][bank][...] = s["tiles"][bank][...] + value
            output_copy(panel).start()
        for panel in range(panels - 3, panels):
            output_copy(panel).wait()

    pl.run_scoped(
        body,
        dict(
            norm=vmem((1, width), jnp.float32),
            norm_sem=dma,
            normalized=vmem((rows, width), jnp.bfloat16),
            weights=tuple(vmem((width, 512), jnp.bfloat16) for _ in range(3)),
            tiles=tuple(vmem((rows, 512), jnp.bfloat16) for _ in range(3)),
            weight_sems=(dma, dma, dma),
            prefix_sems=(dma, dma, dma),
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
    fallback_shared=None,
    maximum=None,
    route_counts=None,
):
    """One launch owns all compute, metadata and communication on the fast path.

    Large projection scratch has a shorter lifetime than expert scratch. Only
    shared output and the scalar routing metadata survive that boundary. At 8K
    all ranks take the same overflow branch; no routes are dropped to fit SMEM.
    """
    rows = hidden.shape[0]
    tokens = rows * 32
    capacity = tokens * 16 if tokens < 8192 else 32768
    shape = (tokens // 8, 128)
    vmem, smem, dma = pltpu.VMEM, pltpu.SMEM, pltpu.SemaphoreType.DMA

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
            if tokens == 8192:
                maximum_copy().start()

        def probability_copy(probs_v):
            return pltpu.make_async_copy(
                probs_v.at[: capacity // 128, :],
                s["probs"],
                s["prob_sem"],
            )

        def prepare(ids_v, probs_v):
            indices = pltpu.make_async_copy(
                ids_v.at[: capacity // 128, :], s["ids"], s["index_sem"])
            def metadata_ready():
                indices.start()
                probability_copy(probs_v).start()
            _prologue(
                tables,
                control,
                hidden,
                router,
                bias,
                latent_down,
                shared_gu,
                shared_down,
                s["shared"],
                latent,
                ids_v,
                probs_v,
                s["counts_v"],
                maximum,
                counts_ready=counts_ready,
                metadata_ready=metadata_ready,
            )
            count_copy().wait()
            if route_counts is not None:
                for e in range(56):
                    route_counts[e] = s["counts"][e]
            if tokens == 8192:
                maximum_copy().wait()
            if tokens != 8192:
                metadata_ready()
            if tokens == 8192:

                @pl.when(s["maximum"][0] > capacity)
                def export_overflow():
                    for source, dest in (
                        (ids_v, fallback_ids),
                        (probs_v, fallback_probs),
                        (s["counts_v"], fallback_counts),
                        (s["shared"], fallback_shared),
                    ):
                        copy = pltpu.make_async_copy(source, dest, s["export_sem"])
                        copy.start()
                        copy.wait()

            indices.wait()
            if tokens == 8192:
                probability_copy(probs_v).wait()

        if tokens == 8192:
            # The compact path completes both metadata DMAs before expert work.
            # Reclaim the full probability array (512 KiB) for expert scratch.
            pl.run_scoped(prepare, vmem(shape, jnp.int32), vmem(shape, jnp.float32))
        else:
            pl.run_scoped(
                lambda ids: prepare(ids, s["probs_v"]), vmem(shape, jnp.int32)
            )

        def probabilities_ready():
            if tokens < 8192:
                probability_copy(s["probs_v"]).wait()

        def fast():
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
                s["reduced"],
                tokens=tokens,
                probabilities_ready=probabilities_ready,
            )
            _epilogue(
                control, s["reduced"], s["shared"], norm, latent_up, prefix, output
            )

        if tokens < 8192:
            fast()
        else:

            @pl.when(s["maximum"][0] <= capacity)
            def compact():
                fast()

    pl.run_scoped(
        body,
        dict(
            shared=vmem((rows, 7168), jnp.bfloat16),
            reduced=vmem((rows, 3584), jnp.float32),
            probs_v=vmem(shape, jnp.float32) if tokens < 8192 else None,
            counts_v=vmem((56,), jnp.int32),
            counts=smem((56,), jnp.int32),
            maximum=smem((1,), jnp.int32) if tokens == 8192 else None,
            ids=smem((capacity // 128, 128), jnp.int32),
            probs=smem((capacity // 128, 128), jnp.float32),
            family=smem((1,), jnp.int32),
            offset=smem((1,), jnp.int32),
            count_sem=dma,
            max_sem=dma if tokens == 8192 else None,
            index_sem=dma,
            prob_sem=dma,
            export_sem=dma,
        ),
    )


def make_indexed_moe_sp_ring(mesh, *, seq_len=8192):
    """Build the fused EP16/TP2 prefill MoE for a supported serving window.

    Inputs follow the prefill layer interface: rank-sharded normalized hidden
    states, residual prefix, resident weight families, valid length and family
    index. All fast-path operations execute in one Pallas kernel. Rare 8K route
    overflow uses the uncapped production expert path with the same metadata.
    """
    if seq_len not in (512, 1024, 2048, 4096, 8192):
        raise ValueError(
            "Fused prefill MoE supports 512, 1024, 2048, 4096 and 8192 token windows"
        )
    if k._config().moe_reduce_dtype != "float32":
        raise ValueError(
            "Fused prefill MoE requires FP32 reduction; use the dispatcher for BF16"
        )
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("Fused prefill MoE requires a flat 32-rank k3_sp mesh")
    from .layer import routed_experts_prepared

    rows = seq_len // 32
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(32)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shape = (seq_len // 8, 128)
    bf16 = jnp.bfloat16
    workspace_shapes = [
        ((rows, 7168), bf16),
        ((seq_len, 2, 1792), bf16),
        ((seq_len * 16, 1536), bf16),
    ]
    output_specs = [hbm] * 3
    if seq_len == 8192:
        # Only 8K can overflow compact SMEM routing capacity. At <=4K every
        # route fits, so omit the unused overflow outputs and maximum DMA.
        workspace_shapes += [
            (shape, jnp.int32),
            (shape, jnp.float32),
            ((56,), jnp.int32),
            ((rows, 7168), bf16),
            ((1,), jnp.int32),
        ]
        output_specs += [hbm] * 4 + [pl.BlockSpec()]
    counting = k._config().prefill_route_counts
    body = _layer_kernel
    if counting:
        workspace_shapes += [((56,), jnp.int32)]
        output_specs += [pl.BlockSpec(memory_space=pltpu.SMEM)]

        def body(*refs):
            return _layer_kernel(*refs[:-1], route_counts=refs[-1])
    kernel = pl.pallas_call(
        body,
        out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * 13,
            out_specs=tuple(output_specs),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=False,
        ),
        name=f"fused_moe_pallas_{seq_len}",
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P(), P()),
        out_specs=(P("k3_sp"), P("k3_sp")) if counting else P("k3_sp"),
        check_vma=False,
    )
    def local(hidden, prefix, families, length, slot):
        w = {name: value[0] for name, value in families.items()}
        control = jnp.concatenate((length.reshape(1), slot.reshape(1)))
        outputs = kernel(
            jnp.asarray(tables),
            control,
            hidden[0],
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
        )
        if counting:
            *outputs, routes = outputs
            routes = routes[None]
        if seq_len < 8192:
            return (outputs[0][None], routes) if counting else outputs[0][None]
        final, latent, _, ids, probs, counts, shared, maximum = outputs

        def overflow():
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
                shared_reduced=True,
            )

        result = jax.lax.cond(maximum[0] <= 32768, lambda: final, overflow)[None]
        return (result, routes) if counting else result

    return jax.jit(local)
