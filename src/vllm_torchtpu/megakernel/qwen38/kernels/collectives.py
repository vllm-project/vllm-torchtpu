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
"""Collectives shared by the production layerstack and hardware qualification."""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

F32 = jnp.float32
# Columns of one packet block moved between a source or destination and the packet.
BLOCK = 1024


def canonical_sum(source, rank_offset):
    """Sum an XOR-addressed inbox in absolute rank order on every receiver.

    Inbox slot ``peer`` holds absolute rank ``rank_offset ^ peer``. An explicit
    common order preserves replicated FP32 results before the BF16 boundary.
    The indexed inbox dimension is outside the two tiled matrix dimensions.
    """
    total = jnp.zeros(source.shape[1:], F32)
    for absolute_rank in range(source.shape[0]):
        total += source[rank_offset ^ absolute_rank, ...].astype(F32)
    return total


def _wait_bytes(view, semaphore):
    # A DMA semaphore counts bytes: one wait on a view spanning every payload
    # of a fan-out consumes all of their completions.
    tpu.make_async_copy(view, view, semaphore).wait()


def hierarchical_sum(
    src, iteration, rank, local, remote, sends, recvs, *, compact, id_type, defer_send=False
):
    """FP32 local-eight then host-group allgather/sum in absolute rank order.

    A C1 row is reshaped into the inboxes' aligned [packet rows, cols] packet,
    without arithmetic changes. Double-buffered inboxes preserve the existing
    decode protocol. BF16 inboxes send each partial and each host sum rounded
    to BF16; every sum still accumulates in FP32.
    Each phase's transfers share one of four send and four receive semaphores,
    slot 2 * phase + iteration parity, drained by byte-counted waits. A peer
    runs at most one call ahead, so parity slots cannot mix two calls'
    completions. With ``defer_send`` the host-phase sends stay pending until
    ``drain_hierarchical_sum`` for this iteration.
    """
    rows, h = src.shape
    local_size, hosts = local.shape[1], remote.shape[1]
    packet_rows, packet_cols = local.shape[-2:]

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

    slot = iteration % 2
    received = local.at[slot, pl.ds(1, local_size - 1)] if local_size > 1 else None
    hosted = remote.at[slot, pl.ds(1, hosts - 1)] if hosts > 1 else None
    live_rows = packet_rows * packet_cols // h if compact else rows
    if compact and (not 1 <= live_rows <= rows or live_rows * h != packet_rows * packet_cols):
        raise ValueError("Compact packet must hold complete live rows")
    packet = src[:live_rows, :].reshape(packet_rows, packet_cols) if compact else src[...]
    local[slot, 0, ...] = packet.astype(local.dtype)
    for peer in range(local_size - 1, 0, -1):
        tpu.make_async_remote_copy(
            local.at[slot, 0],
            local.at[slot, peer],
            sends.at[slot],
            recvs.at[slot],
            device_id=peer_id(rank ^ peer),
            device_id_type=id_type,
        ).start()
    if received is not None:
        _wait_bytes(received, recvs.at[slot])
    remote[slot, 0, ...] = canonical_sum(
        local.at[slot], rank % local_size
    ).astype(remote.dtype)
    for peer in range(hosts - 1, 0, -1):
        tpu.make_async_remote_copy(
            remote.at[slot, 0],
            remote.at[slot, peer],
            sends.at[2 + slot],
            recvs.at[2 + slot],
            device_id=peer_id(rank ^ (peer * local_size)),
            device_id_type=id_type,
        ).start()
    if hosted is not None:
        _wait_bytes(hosted, recvs.at[2 + slot])
    result = canonical_sum(remote.at[slot], rank // local_size)
    if received is not None:
        _wait_bytes(received, sends.at[slot])
    if hosted is not None and not defer_send:
        _wait_bytes(hosted, sends.at[2 + slot])
    return jnp.pad(result.reshape(live_rows, h), ((0, rows - live_rows), (0, 0))) if compact else result


def drain_hierarchical_sum(iteration, remote, sends):
    """Wait the host-phase sends that hierarchical_sum(defer_send=True) left pending."""
    if remote.shape[1] > 1:
        slot = iteration % 2
        _wait_bytes(remote.at[slot, pl.ds(1, remote.shape[1] - 1)], sends.at[2 + slot])


def scattered_sum(
    src, iteration, rank, scratch, *, id_type, defer_send=False, parts=1, deliver=None
):
    """hierarchical_sum's exact FP32 result with reduce-scatter/all-gather traffic.

    Local rank j owns packet column chunk j. It sums the eight local chunks in
    absolute rank order, exchanges that partial with its host-group peers, sums
    those in absolute host order and returns the chunk to every local peer.
    Each element sees hierarchical_sum's additions, while a rank sends about
    2.1 packets instead of 10. ``scratch`` holds a ``stage`` [packet rows, cols]
    packet, double-buffered ``scatter`` [2, local, packet rows, chunk], ``host``
    [2, hosts, packet rows, chunk] and ``gathered`` [2, packet rows, cols] inboxes,
    and 6 * ``parts`` send and receive semaphores (one per phase, piece and iteration
    parity: slot 2 * (phase * parts + piece) + parity).
    A BF16 ``gathered`` carries each owner's rounded chunk, which equals the
    caller's own rounding of the FP32 sum; pass one only when the caller rounds
    the result to BF16. BF16 ``stage``, ``scatter`` and ``host`` inboxes also send
    the partials and host sums rounded to BF16, which changes the result.
    ``parts`` > 1 moves every chunk as that many column pieces: all local sends
    start at once, and each piece's host exchange and gather start as soon as its
    own inbound bytes arrive, so later pieces' transfers overlap earlier pieces'
    phases. Sums and their order are unchanged. ``defer_send`` leaves the
    gather-phase sends to ``drain_scattered_sum``.
    Live rows filling several packet rows sit in the packet as row blocks side by side;
    every element still meets the same partials in the same order. Their sum can then
    go to ``deliver(rows, columns, value)`` in [packet rows, BLOCK] FP32 blocks instead
    of being returned as one value.
    """
    rows, h = src.shape
    _, local_size, _, chunk = scratch["scatter"].shape
    hosts = scratch["host"].shape[1]
    packet_rows, packet_cols = scratch["stage"].shape
    live_rows = packet_rows * packet_cols // h
    piece = chunk // parts
    if (not 1 <= live_rows <= rows or live_rows * h != packet_rows * packet_cols
            or chunk * local_size != packet_cols or piece * parts != chunk or piece % 128):
        raise ValueError("Scattered packet must hold complete live rows in aligned pieces")
    blocks = live_rows // packet_rows if live_rows % packet_rows == 0 else 0
    if deliver is not None and (not blocks or h % BLOCK):
        raise ValueError("Delivered sums require live rows filling whole packet rows")
    sends, recvs = scratch["sends"], scratch["recvs"]

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

    def exchange(sources, destinations, peers, semaphore):
        for source, destination, peer in zip(sources, destinations, peers, strict=True):
            tpu.make_async_remote_copy(
                source,
                destination,
                sends.at[semaphore],
                recvs.at[semaphore],
                device_id=peer_id(peer),
                device_id_type=id_type,
            ).start()

    slot = iteration % 2
    owner = rank % local_size
    stage, scatter, host, gathered = (
        scratch[name] for name in ("stage", "scatter", "host", "gathered")
    )

    def semaphore(phase, part):
        return 2 * (phase * parts + part) + slot

    def columns(chunk_owner, part):
        return pl.ds(pl.multiple_of(chunk_owner * chunk + part * piece, 128), piece)

    def own(part):
        return pl.ds(part * piece, piece)

    def scattered(part):
        return scatter.at[slot, pl.ds(1, local_size - 1), :, own(part)]

    def hosted(part):
        return host.at[slot, pl.ds(1, hosts - 1), :, own(part)]

    if blocks > 1:

        @pl.loop(0, packet_cols // BLOCK)
        def fill(index):
            source_rows, source_columns = _block(index, packet_rows, h)
            stage[:, pl.ds(pl.multiple_of(index * BLOCK, BLOCK), BLOCK)] = src[
                source_rows, source_columns
            ].astype(stage.dtype)
    else:
        stage[...] = src[:live_rows, :].reshape(packet_rows, packet_cols).astype(stage.dtype)
    scatter[slot, 0, ...] = stage[:, pl.ds(pl.multiple_of(owner * chunk, 128), chunk)]
    # Local peer rank ^ peer owns chunk owner ^ peer; its inbox slot peer is this rank.
    local_peers = range(local_size - 1, 0, -1)
    host_peers = range(hosts - 1, 0, -1)
    for part in range(parts):
        exchange(
            [stage.at[:, columns(owner ^ peer, part)] for peer in local_peers],
            [scatter.at[slot, peer, :, own(part)] for peer in local_peers],
            [rank ^ peer for peer in local_peers],
            semaphore(0, part),
        )
    for part in range(parts):
        _wait_bytes(scattered(part), recvs.at[semaphore(0, part)])
        host[slot, 0, :, own(part)] = canonical_sum(
            scatter.at[slot, :, :, own(part)], owner
        ).astype(host.dtype)
        exchange(
            [host.at[slot, 0, :, own(part)] for _ in host_peers],
            [host.at[slot, peer, :, own(part)] for peer in host_peers],
            [rank ^ (peer * local_size) for peer in host_peers],
            semaphore(1, part),
        )
    for part in range(parts):
        _wait_bytes(hosted(part), recvs.at[semaphore(1, part)])
        gathered[slot, :, columns(owner, part)] = canonical_sum(
            host.at[slot, :, :, own(part)], rank // local_size
        ).astype(gathered.dtype)
        exchange(
            [gathered.at[slot, :, columns(owner, part)] for _ in local_peers],
            [gathered.at[slot, :, columns(owner, part)] for _ in local_peers],
            [rank ^ peer for peer in local_peers],
            semaphore(2, part),
        )
    for part in range(parts):
        _wait_bytes(_collected(gathered, slot, local_size, piece), recvs.at[semaphore(2, part)])
    result = None
    if deliver is not None:

        @pl.loop(0, packet_cols // BLOCK)
        def give(index):
            packet_columns = pl.ds(pl.multiple_of(index * BLOCK, BLOCK), BLOCK)
            deliver(*_block(index, packet_rows, h), gathered[slot, :, packet_columns].astype(F32))
    elif blocks > 1:
        result = jnp.concatenate(
            [gathered[slot, :, pl.ds(block * h, h)] for block in range(blocks)]
        ).astype(F32)
    else:
        result = gathered[slot, ...].astype(F32).reshape(live_rows, h)
    # The next call rewrites stage; host and gathered alternate by parity.
    for part in range(parts):
        _wait_bytes(scattered(part), sends.at[semaphore(0, part)])
        _wait_bytes(hosted(part), sends.at[semaphore(1, part)])
        if not defer_send:
            _wait_bytes(_collected(gathered, slot, local_size, piece), sends.at[semaphore(2, part)])
    if result is None:
        return None
    return jnp.pad(result, ((0, rows - live_rows), (0, 0)))


def _block(index, packet_rows, h):
    # Rows and columns of a source block: packet column block ``index`` of row blocks
    # laid side by side.
    per_row_block = h // BLOCK
    rows = pl.ds(pl.multiple_of(index // per_row_block * packet_rows, packet_rows), packet_rows)
    return rows, pl.ds(pl.multiple_of(index % per_row_block * BLOCK, BLOCK), BLOCK)


def _collected(gathered, slot, local_size, piece):
    # A view with the byte count of one piece gathered from every other local rank.
    return gathered.at[slot, :, pl.ds(0, (local_size - 1) * piece)]


def drain_scattered_sum(iteration, scratch, parts=1):
    """Wait the gather-phase sends that scattered_sum(defer_send=True) left pending."""
    slot = iteration % 2
    _, local_size, _, chunk = scratch["scatter"].shape
    for part in range(parts):
        _wait_bytes(
            _collected(scratch["gathered"], slot, local_size, chunk // parts),
            scratch["sends"].at[2 * (2 * parts + part) + slot],
        )


def paired_sum(src, iteration, rank, scratch, *, id_type, deliver, defer_send=False, parts=1):
    """scattered_sum with each chip's two cores (rank ^ 1) reduced first, for wide packets.

    The cores swap FP32 packet halves on chip and each adds the pair in rank order,
    rounding the pair sum once to the wire dtype. The ranks of one core position then
    reduce-scatter that half over the host's chips, exchange host sums across hosts as in
    scattered_sum, gather the chunks back over the chips, and the pair swaps finished
    halves. Off-chip bytes in the host-local phases halve. Each host sum adds one rounded
    pair sum per chip instead of one rounded partial per rank, so the result differs from
    scattered_sum's.
    Live rows fill whole packet rows (see scattered_sum); every chunk lies in one row
    block, so each core reads its FP32 pieces straight from ``src``. ``scratch`` holds a
    ``stage`` [packet rows, cols / 2] own-half inbox, double-buffered ``scatter``
    [2, chips, packet rows, chunk] and ``host`` [2, hosts, packet rows, chunk] inboxes, a
    BF16 ``gathered`` [2, packet rows, cols], an FP32 ``pin`` [chips, packet rows, chunk]
    inbox for the sibling's half (a sibling reaches the next call only after this call has
    consumed it), and 10 * ``parts`` send and receive semaphores (slot 2 * (phase * parts +
    piece) + parity). ``defer_send`` leaves the final pair swap's sends to
    ``drain_paired_sum``.
    """
    rows, h = src.shape
    _, chips, packet_rows, chunk = scratch["scatter"].shape
    hosts = scratch["host"].shape[1]
    stage, scatter, host, gathered, pin = (
        scratch[name] for name in ("stage", "scatter", "host", "gathered", "pin")
    )
    local_size = 2 * chips
    packet_cols = gathered.shape[-1]
    live_rows = packet_rows * packet_cols // h
    piece = chunk // parts
    if (live_rows % packet_rows or live_rows > rows or chunk * local_size != packet_cols
            or h % chunk or piece * parts != chunk or piece % 128 or h % BLOCK):
        raise ValueError("Paired packet must hold whole row blocks in aligned chunk pieces")
    sends, recvs = scratch["sends"], scratch["recvs"]

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

    def exchange(sources, destinations, peers, semaphore):
        for source, destination, peer in zip(sources, destinations, peers, strict=True):
            tpu.make_async_remote_copy(
                source,
                destination,
                sends.at[semaphore],
                recvs.at[semaphore],
                device_id=peer_id(peer),
                device_id_type=id_type,
            ).start()

    slot = iteration % 2
    core, chip = rank % 2, rank % local_size // 2
    owner = core * chips + chip
    host_index = rank // local_size

    def semaphore(phase, part):
        return 2 * (phase * parts + part) + slot

    def columns(chunk_owner, part):
        return pl.ds(pl.multiple_of(chunk_owner * chunk + part * piece, 128), piece)

    def own(part):
        return pl.ds(part * piece, piece)

    def partial(chunk_owner, part):
        # A chunk piece of the packet, read from the FP32 row block that holds it.
        start = chunk_owner * chunk + part * piece
        block = pl.ds(pl.multiple_of(start // h * packet_rows, packet_rows), packet_rows)
        return src.at[block, pl.ds(pl.multiple_of(start % h, 128), piece)]

    def paired(part):
        return pin.at[:, :, own(part)]

    def chipped(part):
        return scatter.at[slot, pl.ds(1, chips - 1), :, own(part)]

    def hosted(part):
        return host.at[slot, pl.ds(1, hosts - 1), :, own(part)]

    def collected(count, part):
        # A gathered view with the byte count of ``count`` inbound pieces.
        return gathered.at[slot, :, pl.ds(0, count * piece)]

    other = (1 - core) * chips
    for part in range(parts):
        exchange(
            [partial(other + q, part) for q in range(chips)],
            [pin.at[q, :, own(part)] for q in range(chips)],
            [rank ^ 1] * chips,
            semaphore(0, part),
        )
    # Chip peer rank ^ (2 * peer) owns this core's chunk chip ^ peer.
    chip_peers = range(chips - 1, 0, -1)
    host_peers = range(hosts - 1, 0, -1)
    for part in range(parts):
        _wait_bytes(paired(part), recvs.at[semaphore(0, part)])
        even = core == 0
        for q in range(chips):
            mine = partial(core * chips + q, part)[...]
            theirs = pin[q, :, own(part)]
            pair = jnp.zeros(mine.shape, F32)
            pair += jnp.where(even, mine, theirs)
            pair += jnp.where(even, theirs, mine)
            stage[:, columns(q, part)] = pair.astype(stage.dtype)
        scatter[slot, 0, :, own(part)] = stage[:, columns(chip, part)]
        exchange(
            [stage.at[:, columns(chip ^ peer, part)] for peer in chip_peers],
            [scatter.at[slot, peer, :, own(part)] for peer in chip_peers],
            [rank ^ (2 * peer) for peer in chip_peers],
            semaphore(1, part),
        )
    for part in range(parts):
        _wait_bytes(chipped(part), recvs.at[semaphore(1, part)])
        host[slot, 0, :, own(part)] = canonical_sum(
            scatter.at[slot, :, :, own(part)], chip
        ).astype(host.dtype)
        exchange(
            [host.at[slot, 0, :, own(part)] for _ in host_peers],
            [host.at[slot, peer, :, own(part)] for peer in host_peers],
            [rank ^ (peer * local_size) for peer in host_peers],
            semaphore(2, part),
        )
    for part in range(parts):
        _wait_bytes(hosted(part), recvs.at[semaphore(2, part)])
        gathered[slot, :, columns(owner, part)] = canonical_sum(
            host.at[slot, :, :, own(part)], host_index
        ).astype(gathered.dtype)
        exchange(
            [gathered.at[slot, :, columns(owner, part)] for _ in chip_peers],
            [gathered.at[slot, :, columns(owner, part)] for _ in chip_peers],
            [rank ^ (2 * peer) for peer in chip_peers],
            semaphore(3, part),
        )
    for part in range(parts):
        _wait_bytes(collected(chips - 1, part), recvs.at[semaphore(3, part)])
        mine = [columns(core * chips + q, part) for q in range(chips)]
        exchange(
            [gathered.at[slot, :, block] for block in mine],
            [gathered.at[slot, :, block] for block in mine],
            [rank ^ 1] * chips,
            semaphore(4, part),
        )
    for part in range(parts):
        _wait_bytes(collected(chips, part), recvs.at[semaphore(4, part)])

    @pl.loop(0, packet_cols // BLOCK)
    def give(index):
        packet_columns = pl.ds(pl.multiple_of(index * BLOCK, BLOCK), BLOCK)
        deliver(*_block(index, packet_rows, h), gathered[slot, :, packet_columns].astype(F32))

    for part in range(parts):
        _wait_bytes(paired(part), sends.at[semaphore(0, part)])
        _wait_bytes(chipped(part), sends.at[semaphore(1, part)])
        _wait_bytes(hosted(part), sends.at[semaphore(2, part)])
        _wait_bytes(collected(chips - 1, part), sends.at[semaphore(3, part)])
        if not defer_send:
            _wait_bytes(collected(chips, part), sends.at[semaphore(4, part)])


def drain_paired_sum(iteration, scratch, parts=1):
    """Wait the pair-swap sends that paired_sum(defer_send=True) left pending."""
    slot = iteration % 2
    piece = scratch["scatter"].shape[-1] // parts
    for part in range(parts):
        _wait_bytes(
            scratch["gathered"].at[slot, :, pl.ds(0, scratch["scatter"].shape[1] * piece)],
            scratch["sends"].at[2 * (4 * parts + part) + slot],
        )


def broadcast_sum(src, iteration, rank, scratch, *, id_type, defer_send=False):
    """scattered_sum's BF16-gathered result in two phases instead of three.

    After the same local reduce-scatter, every owner sends its rounded host-local
    chunk sum to all other ranks, and each rank adds the host sums of every chunk
    itself in absolute host order. A rank then sends 31 chunks in its second phase
    instead of 10 over two, which pays only while chunks are small. Like a BF16
    ``gathered`` inbox, the FP32 sum is returned rounded to BF16; pass it only when
    the caller rounds the result to BF16. ``scratch``
    holds scattered_sum's ``stage`` and ``scatter``, a ``sums`` [2, hosts, packet
    rows, cols] inbox in the wire dtype, and four send and four receive semaphores
    (slot 2 * phase + parity); ``defer_send`` leaves the broadcast sends to
    ``drain_broadcast_sum``.
    """
    rows, h = src.shape
    _, local_size, packet_rows, chunk = scratch["scatter"].shape
    sums = scratch["sums"]
    hosts = sums.shape[1]
    packet_cols = scratch["stage"].shape[1]
    live_rows = packet_rows * packet_cols // h
    if (not 1 <= live_rows <= rows or live_rows * h != packet_rows * packet_cols
            or chunk * local_size != packet_cols or chunk % 128):
        raise ValueError("Broadcast packet must hold complete live rows in aligned chunks")
    sends, recvs = scratch["sends"], scratch["recvs"]

    def peer_id(peer):
        return peer if id_type == pl.DeviceIdType.LOGICAL else (peer,)

    def send(source, destination, peer, semaphore):
        tpu.make_async_remote_copy(
            source,
            destination,
            sends.at[semaphore],
            recvs.at[semaphore],
            device_id=peer_id(peer),
            device_id_type=id_type,
        ).start()

    def columns(chunk_owner):
        return pl.ds(pl.multiple_of(chunk_owner * chunk, 128), chunk)

    slot = iteration % 2
    owner, host = rank % local_size, rank // local_size
    stage, scatter = scratch["stage"], scratch["scatter"]
    stage[...] = src[:live_rows, :].reshape(packet_rows, packet_cols).astype(stage.dtype)
    scatter[slot, 0, ...] = stage[:, columns(owner)]
    scattered = scatter.at[slot, pl.ds(1, local_size - 1)]
    # Local peer rank ^ peer owns chunk owner ^ peer; its inbox slot peer is this rank.
    for peer in range(local_size - 1, 0, -1):
        send(stage.at[:, columns(owner ^ peer)], scatter.at[slot, peer], rank ^ peer, slot)
    _wait_bytes(scattered, recvs.at[slot])
    own = sums.at[slot, host, :, columns(owner)]
    own[...] = canonical_sum(scatter.at[slot], owner).astype(sums.dtype)
    # Every rank stores a sum at its own (host, chunk) place in each peer's inbox.
    for peer in range(local_size * hosts - 1, 0, -1):
        send(own, own, rank ^ peer, 2 + slot)
    for view in _broadcast_views(sums, slot, local_size, chunk):
        _wait_bytes(view, recvs.at[2 + slot])
    total = jnp.zeros((packet_rows, packet_cols), F32)
    for absolute in range(hosts):
        total += sums[slot, absolute, ...].astype(F32)
    result = total.astype(jnp.bfloat16).astype(F32).reshape(live_rows, h)
    # The next call rewrites stage; sums alternate by parity.
    _wait_bytes(scattered, sends.at[slot])
    if not defer_send:
        drain_broadcast_sum(iteration, scratch)
    return jnp.pad(result, ((0, rows - live_rows), (0, 0)))


def _broadcast_views(sums, slot, local_size, chunk):
    # Views whose byte counts add up to the chunks of every other rank.
    hosts = sums.shape[1]
    return (
        sums.at[slot, pl.ds(0, hosts - 1)],
        sums.at[slot, 0, :, pl.ds(0, (local_size - 1) * chunk)],
    )


def drain_broadcast_sum(iteration, scratch):
    """Wait the broadcast sends that broadcast_sum(defer_send=True) left pending."""
    slot = iteration % 2
    _, local_size, _, chunk = scratch["scatter"].shape
    for view in _broadcast_views(scratch["sums"], slot, local_size, chunk):
        _wait_bytes(view, scratch["sends"].at[2 + slot])
