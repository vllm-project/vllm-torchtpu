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
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from ... import config as k3_config
from .. import prefill_megakernel as k
from .. import rank_loader
from . import metadata_compact, routed_16k, shared_down_16k, w4a8_16k

_COMPACT_8K_CAPACITY = 65536
_COMPACT_16K_CAPACITY = 98304
_RMW_BATCH = 2
_RMW_DUMMY_SLOTS = 16
_FP8_MAX = 448.0
_FP8_SCALE_EPS = 1e-12
_FP8_DTYPE = jnp.float8_e4m3fn

MOE_DENSE_FP8_FAMILIES = ("latent_down", "shared_gu", "shared_down", "latent_up")
MOE_DENSE_FP8_SCALES = (
    "latent_down_scale",
    "shared_gu_scale",
    "shared_down_scale",
    "latent_up_scale",
)


def quantize_moe_dense_weight_fp8(
    weight: jax.Array, *, slice_cols: int | None = None
) -> tuple[jax.Array, jax.Array]:
    w_src = weight[..., :slice_cols] if slice_cols is not None else weight
    w_f32 = w_src.astype(jnp.float32)
    max_abs = jnp.max(jnp.abs(w_f32), axis=-2, keepdims=True)
    scale = jnp.maximum(max_abs, _FP8_SCALE_EPS) / _FP8_MAX
    w_fp8 = jnp.clip(w_f32 / scale, -_FP8_MAX, _FP8_MAX).astype(_FP8_DTYPE)
    return w_fp8, scale.astype(jnp.float32)


def _quantize_rows_to_fp8_vmem(
    src_ref, dst_fp8_ref, dst_scale_ref, *, rows: int, row_offset: int = 0
):
    row_chunk = min(rows, 128)
    for r0 in range(0, rows, row_chunk):
        rc = min(row_chunk, rows - r0)
        x_f32 = src_ref[pl.ds(row_offset + r0, rc), :].astype(jnp.float32)
        rmax = jnp.max(jnp.abs(x_f32), axis=1, keepdims=True)
        scale = jnp.maximum(rmax, _FP8_SCALE_EPS) / _FP8_MAX
        inv_scale = 1.0 / scale
        dst_fp8_ref[pl.ds(row_offset + r0, rc), :] = (x_f32 * inv_scale).astype(
            _FP8_DTYPE
        )
        dst_scale_ref[pl.ds(row_offset + r0, rc), :] = jnp.broadcast_to(
            scale, (rc, 128)
        )


def _compact_capacity(tokens: int) -> int:
    if tokens < 8192:
        return tokens * 16
    if tokens == 8192:
        return _COMPACT_8K_CAPACITY
    return _COMPACT_16K_CAPACITY


def sparse_poll_threshold(*, split_ring: bool) -> int:
    return 144 if split_ring else 72


def _sort_route_steps(
    keys, probabilities, token_ids=None, *, size, bits=0, keep_half=False
):
    window, rows = (8192, 64) if size >= 262144 else (4096, 32)
    assert size >= 2 * window and size & (size - 1) == 0
    shape = (rows, 128)
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
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        return jnp.take_along_axis(
            value, lane ^ distance, axis=1, mode="promise_in_bounds"
        )

    def apply_in_window(x, p, ordinal, steps):
        row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        global_position = row * 128 + lane + ordinal * window
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
                pair_limit = (
                    size // (4 * window)
                    if (keep_half and group == size and distance < size // 2)
                    else size // (2 * window)
                )
                for ordinal in range(pair_limit):
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
            pair_limit = (
                size // (4 * window)
                if (keep_half and group == size and distance < size // 2)
                else size // (2 * window)
            )
            drop_high = keep_half and group == size and distance == size // 2
            for ordinal in range(pair_limit):
                low = ordinal // stride * (2 * stride) + ordinal % stride
                high = low + stride
                kl, kh = view(keys, low), view(keys, high)
                plow, phigh = view(probabilities, low), view(probabilities, high)
                x, y, p, q = kl[...], kh[...], plow[...], phigh[...]
                ascending = (low * window & group) == 0
                keep = ~((x < y) ^ ascending)
                kl[...], plow[...] = jnp.where(keep, x, y), jnp.where(keep, p, q)
                if not drop_high:
                    kh[...], phigh[...] = jnp.where(keep, y, x), jnp.where(keep, q, p)
                yield
    yield from flush()


class _Top16:
    """Incremental biased sigmoid Top16; ties choose the lowest expert ID."""

    def __init__(self, logits, bias):
        self.raw = 1.0 / (1.0 + jnp.exp(-logits))
        self.scores = self.raw + bias
        self.selected: list[jax.Array] = []
        self.probabilities: list[jax.Array] = []
        self.index = 0

    def step(self):
        experts = jax.lax.broadcasted_iota(jnp.int32, self.scores.shape, 1)
        best = jnp.max(self.scores, axis=1, keepdims=True)
        winner = jnp.min(
            jnp.where(self.scores == best, experts, self.scores.shape[1]),
            axis=1,
            keepdims=True,
        )
        chosen = experts == winner
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
        denominator = jnp.maximum(jnp.sum(probabilities, axis=1, keepdims=True), 1e-30)
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
    chunks = tokens // 128
    count_batch = min(8, chunks)
    total_counts = jnp.zeros((local_experts,), jnp.int32)
    total_groups = jnp.zeros((ep_groups,), jnp.int32)
    local_batch: list[jax.Array] = []
    ids_batch: list[jax.Array] = []
    live_batch: list[jax.Array] = []
    for chunk in range(chunks):
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
        local_batch.append(local.reshape(16, 128))
        if maximum is not None:
            ids_batch.append(ids.reshape(16, 128))
            live_batch.append(probs.reshape(16, 128) != 0)
        if len(local_batch) == count_batch:
            local_flat = jnp.concatenate(local_batch, axis=0)
            tile_counts = jnp.zeros((local_experts,), jnp.int32)
            for expert in range(local_experts):
                count = jnp.sum((local_flat == expert).astype(jnp.int32))
                tile_counts = jnp.where(expert_index == expert, count, tile_counts)
            total_counts += tile_counts
            local_batch.clear()
            if maximum is not None:
                groups = jnp.concatenate(ids_batch, axis=0) // local_experts
                live_flat = jnp.concatenate(live_batch, axis=0)
                tile_groups = jnp.zeros((ep_groups,), jnp.int32)
                for group in range(ep_groups):
                    count = jnp.sum((live_flat & (groups == group)).astype(jnp.int32))
                    tile_groups = jnp.where(group_index == group, count, tile_groups)
                total_groups += tile_groups
                ids_batch.clear()
                live_batch.clear()
        yield
    counts[...] = total_counts
    if maximum is not None:
        maximum[...] = jnp.max(total_groups)[None]
    counts_ready()
    yield from _sort_route_steps(
        keys,
        probabilities,
        token_ids,
        size=tokens * 16,
        bits=bits,
        keep_half=(tokens == 8192 and maximum is None),
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


def _latent_half_pair_panels(
    lhs_ref, weights, buffers, semaphores, *, prestarted: bool = False
):
    """Pair matching 256-column fragments of the two 1792-wide halves."""
    rows = lhs_ref.shape[0]
    row_tile = min(rows, 256)

    def source_column(panel):
        return (panel // 2) * 256 + (panel % 2) * 1792

    def copy(panel):
        return pltpu.make_async_copy(
            weights.at[:, pl.ds(source_column(panel), 256)],
            buffers[panel % 4],
            semaphores[panel % 4],
        )

    if not prestarted:
        for panel in range(2):
            copy(panel).start()
    for panel in range(0, 14, 2):
        for j in range(2):
            copy(panel + j).wait()
        for j in range(2):
            if panel + 2 + j < 14:
                copy(panel + 2 + j).start()
        products = [
            jnp.concatenate(
                [
                    jnp.matmul(
                        lhs_ref[pl.ds(r0, row_tile), :],
                        buffers[(panel + j) % 4][...],
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)
                    for r0 in range(0, rows, row_tile)
                ],
                axis=0,
            )
            for j in range(2)
        ]
        for j, product in enumerate(products):
            yield source_column(panel + j), product


def _dense_panels_fp8(
    lhs_fp8, lhs_scale, weights_fp8, weights_scale, buffers, semaphores, *, columns
):
    def source(panel):
        return weights_fp8.at[:, pl.ds(panel * 256, 256)]

    pltpu.make_async_copy(source(0), buffers[0], semaphores[0]).start()
    for panel in range(columns // 256):
        bank = panel % 2
        pltpu.make_async_copy(source(panel), buffers[bank], semaphores[bank]).wait()
        if panel + 1 < columns // 256:
            pltpu.make_async_copy(
                source(panel + 1), buffers[1 - bank], semaphores[1 - bank]
            ).start()
        w_sc = weights_scale[:, pl.ds(panel * 256, 256)]
        raw = jnp.matmul(
            lhs_fp8, buffers[bank][...], preferred_element_type=jnp.float32
        )
        yield panel * 256, raw * lhs_scale * w_sc


def _latent_half_pair_panels_fp8(
    lhs_fp8_ref, lhs_scale_ref, weights_fp8, weights_scale, buffers, semaphores
):
    rows = lhs_fp8_ref.shape[0]
    row_tile = min(rows, 256)

    def source_column(panel):
        return (panel // 2) * 256 + (panel % 2) * 1792

    def copy(panel):
        return pltpu.make_async_copy(
            weights_fp8.at[:, pl.ds(source_column(panel), 256)],
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
        products = []
        for j in range(2):
            col = source_column(panel + j)
            w_sc = weights_scale[:, pl.ds(col, 256)]
            parts = []
            for r0 in range(0, rows, row_tile):
                raw = jnp.matmul(
                    lhs_fp8_ref[pl.ds(r0, row_tile), :],
                    buffers[(panel + j) % 4][...],
                    preferred_element_type=jnp.float32,
                )
                x_sc = lhs_scale_ref[pl.ds(r0, row_tile), :1]
                parts.append((raw * x_sc * w_sc).astype(jnp.bfloat16))
            products.append(jnp.concatenate(parts, axis=0))
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


def _await_partner_scope(ready, peer):
    pl.semaphore_signal(
        ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.LOGICAL
    )
    pl.semaphore_wait(ready, 1)


def _latent_gather_steps(
    tables, payload, received, output, send, recv, credits, stores, *, rank
):
    """Twelve directional streams, with two receive slots per stream.

    Windows up to 4K forward through VMEM, copying the completed receive slot into
    payload.
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
        regions.append((begin, size, family, direction))
        begin += size

    def stream_peers(family, direction):
        position = tables[family, 0, rank]
        target = tables[family, 1, (position + 32 + direction) % 32]
        source_peer = tables[family, 1, (position + 32 - direction) % 32]
        return position, target, source_peer

    def destination(region, step, position):
        begin, size, family, direction = region
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
                begin, size, family, direction = region
                position, peer, source_peer = stream_peers(family, direction)
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
                        piece, destination(region, step - 1, position), st
                    ).wait()
                    piece[...] = old[...]
                    pl.semaphore_signal(
                        credit,
                        1,
                        device_id=source_peer,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )
                dest = destination(region, step, position)
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
            begin, size, family, direction = region
            position, peer, source_peer = stream_peers(family, direction)
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
                    else destination(region, step, position)
                )
                dest = destination(region, step, position)
                incoming = destination(region, step + 1, position)
                if step >= 2:
                    pl.semaphore_wait(credits.at[stream], 1)
                transfer = remote(
                    source, dest, send.at[stream], recv.at[step % 2, stream], peer
                )
                transfer.start()
                pending[stream] = transfer, incoming, step
        yield
    seed.wait()


_HALF_LATENT_META_STRIDE = 20


def _latent_half_stream_layout(
    rows: int, n_ranks: int
) -> tuple[tuple[int, int, int, int, int], ...]:
    row_sizes = [rows // 3 + (family < rows % 3) for family in range(3)]
    row_begins = (0, row_sizes[0], row_sizes[0] + row_sizes[1])
    layout = []
    for stream in range(6):
        family = stream // 2
        direction = 1 if stream % 2 == 0 else -1
        max_hop = n_ranks // 2 if direction == 1 else (n_ranks // 2 - 1)
        layout.append(
            (family, direction, row_begins[family], row_sizes[family], max_hop)
        )
    return tuple(layout)


def _latent_half_init(
    tables, payload, output, send, recv, stores, state, meta, *, rank, n_ranks=32
):
    rows = payload.shape[0]
    assert rows in (256, 512)
    assert n_ranks >= 4 and (n_ranks & (n_ranks - 1)) == 0
    rank_mask = n_ranks - 1
    layout = _latent_half_stream_layout(rows, n_ranks)
    for stream, (family, direction, _, _, max_hop) in enumerate(layout):
        base = stream * _HALF_LATENT_META_STRIDE
        position = tables[family, 0, rank]
        meta[base] = tables[family, 1, (position + direction) & rank_mask]
        meta[base + 1] = tables[family, 1, (position - direction) & rank_mask]

        @pl.loop(0, max_hop + 1)
        def owners(
            step,
            *,
            family=family,
            position=position,
            direction=direction,
            base=base,
            rank_mask=rank_mask,
        ):
            owner = tables[family, 1, (position - direction * step) & rank_mask]
            meta[base + 2 + step] = owner * rows

    pltpu.make_async_copy(
        payload, output.at[pl.ds(rank * rows, rows), :, :], stores
    ).start()
    state[0] = 6
    for stream, (_, _, begin, size, _) in enumerate(layout):
        part = pl.ds(begin, size)
        pltpu.make_async_remote_copy(
            payload.at[part, :, :],
            output.at[pl.ds(rank * rows + begin, size), :, :],
            send.at[stream],
            recv.at[0, stream],
            device_id=meta[stream * _HALF_LATENT_META_STRIDE],
            device_id_type=pl.DeviceIdType.LOGICAL,
        ).start()
        state[1 + stream] = 1
        state[7 + stream] = 0


def _latent_half_progress(
    payload,
    output,
    send,
    recv,
    credits,
    stores,
    state,
    meta,
    *,
    rank,
    n_ranks=32,
    drain=False,
):
    rows = payload.shape[0]
    row_bytes = payload.shape[1] * payload.shape[2] * jnp.dtype(payload.dtype).itemsize
    layout = _latent_half_stream_layout(rows, n_ranks)

    def sweep():
        for stream, (_, _, start, size, max_hop) in enumerate(layout):
            base = stream * _HALF_LATENT_META_STRIDE
            # TPUv7 DMA semaphore accounting: one unit per 32 logical bytes.
            units = size * row_bytes // 32
            step = state[1 + stream]

            @pl.when(state[7 + stream] == 0)
            def receive(
                *,
                step=step,
                stream=stream,
                units=units,
                base=base,
                start=start,
                size=size,
                max_hop=max_hop,
            ):
                bank = (step - 1) & 1
                ready = (pl.semaphore_read(send.at[stream]) >= units) & (
                    pl.semaphore_read(recv.at[bank, stream]) >= units
                )

                @pl.when(ready)
                def consume():
                    incoming = output.at[
                        pl.ds(meta[base + 2 + step] + start, size), :, :
                    ]
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

                    @pl.when(step < max_hop - 1)
                    def credit():
                        pl.semaphore_signal(
                            credits.at[stream],
                            1,
                            device_id=meta[base + 1],
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )

                    state[7 + stream] = jnp.where(step == max_hop, 2, 1)

                    @pl.when(step == max_hop)
                    def complete():
                        state[0] = state[0] - 1

            @pl.when(state[7 + stream] == 1)
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
                    state[7 + stream] = 0

    if drain:

        def body(i):
            sweep()
            return i + 1

        jax.lax.while_loop(lambda i: state[0] != 0, body, jnp.int32(0))
        pltpu.make_async_copy(
            payload, output.at[pl.ds(rank * rows, rows), :, :], stores
        ).wait()
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
    pair_buf=None,
    shared_in_hbm=False,
    dense_scales=None,
    replica_table=None,
    replica_slots: int = 0,
    latent_scales_hbm=None,
    defer_shared_down: bool = False,
):
    """Router, latent projection and shared experts, with incremental collectives.

    The caller retains shared/metadata outputs across this scope. All temporary
    projection and communication buffers die before routed-expert scratch is
    allocated. The initial HBM input load targets our half of the peer buffer;
    both the router and shared expert use this same copy.
    """
    if hidden is None:
        assert pair_buf is not None
        rows, hidden_size = pair_buf.shape[0] // 2, pair_buf.shape[1]
    else:
        rows, hidden_size = hidden.shape
    rank = jax.lax.axis_index("k3_sp")
    parity, peer = rank % 2, rank ^ 1
    family = control[1]
    router = router.at[family]
    bias = bias.at[family]
    latent_weight = latent_weight.at[family]
    shared_gu = shared_gu.at[family]
    shared_down = shared_down.at[family]
    dense_fp8 = dense_scales is not None
    w4a8_latent = latent_scales_hbm is not None
    if w4a8_latent:
        assert not dense_fp8 and rows == 512
        assert latent_hbm.dtype == w4a8_16k.LATENT_E4M3_DTYPE
        assert latent_hbm.shape[1:] == w4a8_16k.LATENT_E4M3_TAIL_SHAPE
        assert latent_scales_hbm.shape == w4a8_16k.LATENT_SCALES_SHAPE
        assert latent_scales_hbm.dtype == jnp.float32
    if defer_shared_down:
        assert not dense_fp8 and shared_in_hbm and rows == 512
        assert shared.shape == (rows, shared_down_16k.SHARED_HIDDEN_16K)
    if dense_scales is not None:
        latent_weight_scale = dense_scales[0].at[family]
        shared_gu_scale = dense_scales[1].at[family]
        shared_down_scale = dense_scales[2].at[family]
    else:
        latent_weight_scale = None
        shared_gu_scale = None
        shared_down_scale = None
    valid = _stripe_validity(rows, control[0], rank)
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(s):
        pair_v = pair_buf if pair_buf is not None else s["pair"]
        own = pair_v.at[pl.ds(parity * rows, rows), :]
        use_compact_metadata = rows >= 256
        metadata_steps = iter(())
        metadata_started = False
        latent_progress_fn: Any = None

        def start_metadata():
            nonlocal metadata_started, metadata_steps
            if not use_compact_metadata and not metadata_started:
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
            if use_compact_metadata:
                return
            steps_per_call = 2 if rows <= 128 else (5 if rows == 256 else 6)
            for _ in range(steps_per_call):
                next(metadata_steps, None)

        def run_front_and_gu(fs):
            nonlocal latent_progress_fn

            def gu_copy(panel, offset, bufs, sems):
                bank = panel % 2
                return pltpu.make_async_copy(
                    shared_gu.at[:, pl.ds(offset + panel * 256, 256)],
                    bufs[bank],
                    sems[bank],
                )

            if rows >= 256:
                buffers = tuple(
                    fs["router_gu"].at[:, pl.ds(i * 256, 256)] for i in range(4)
                )
                router_v = fs["router_gu"].at[:, :896]
                gate_buffers, up_buffers = buffers[:2], buffers[2:]
                gate_sems, up_sems = fs["gu_sems"][:2], fs["gu_sems"][2:]
                copy = (
                    pltpu.make_async_copy(hidden, own, s["input_sem"])
                    if pair_buf is None
                    else None
                )
                if copy is not None:
                    copy.start()
                pltpu.make_async_copy(
                    latent_weight.at[:, pl.ds(0, 256)], buffers[0], fs["gu_sems"][0]
                ).start()
                pltpu.make_async_copy(
                    latent_weight.at[:, pl.ds(256 if w4a8_latent else 1792, 256)],
                    buffers[1],
                    fs["gu_sems"][1],
                ).start()
                _enter_mesh()
                pl.semaphore_signal(
                    s["peer_credit"],
                    1,
                    device_id=peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                if copy is not None:
                    copy.wait()
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
                router_copy = pltpu.make_async_copy(router, router_v, fs["gu_sems"][0])
                bias_copy = pltpu.make_async_copy(bias, fs["bias"], fs["gu_sems"][1])

                def start_router_bias() -> None:
                    router_copy.start()
                    bias_copy.start()

                if w4a8_latent:
                    w4a8_16k.compute_dynamic_latent_e4m3(
                        own,
                        latent_weight,
                        buffers,
                        fs["gu_sems"],
                        s["latent_payload"],
                        fs["lat_scale_payload"],
                        prestarted=True,
                        on_panels_done=start_router_bias,
                    )
                else:
                    latent_products = _latent_half_pair_panels(
                        own, latent_weight, buffers, fs["gu_sems"], prestarted=True
                    )
                    first_half = None
                    for col, product in latent_products:
                        fragment = product.astype(jnp.bfloat16)
                        if col < 1792:
                            first_half = fragment
                        else:
                            assert first_half is not None
                            paired = jnp.concatenate((first_half, fragment), axis=1)
                            s["latent_payload"][:, :, pl.ds(col - 1792, 256)] = (
                                paired.reshape(rows, 2, 256)
                            )
                    start_router_bias()
                latent_tick = 0

                def latent_progress(*, drain=False):
                    nonlocal latent_tick
                    if latent_tick == 0:
                        _latent_half_init(
                            tables,
                            s["latent_payload"],
                            latent_hbm,
                            s["latent_send_sems"],
                            s["latent_recv_sems"],
                            s["latent_stores"],
                            s["latent_state"],
                            s["latent_meta"],
                            rank=rank,
                        )
                    else:
                        _latent_half_progress(
                            s["latent_payload"],
                            latent_hbm,
                            s["latent_send_sems"],
                            s["latent_recv_sems"],
                            s["latent_credits"],
                            s["latent_stores"],
                            s["latent_state"],
                            s["latent_meta"],
                            rank=rank,
                            drain=drain,
                        )
                    latent_tick += 1

                latent_progress_fn = latent_progress
                latent_progress_fn()
                if w4a8_latent:
                    scale_steps = _route_gather_steps(
                        fs["lat_scale_payload"],
                        s["lat_scales_v"],
                        s["scale_send"],
                        s["scale_recv"],
                        rank=rank,
                    )
                    next(scale_steps, None)
                else:
                    scale_steps = iter(())
                router_copy.wait()
                bias_copy.wait()
                latent_progress_fn()
                if w4a8_latent:
                    next(scale_steps, None)
                bias_v = fs["bias"][...]
                top_tile = min(rows, 256)
                logits_tiles = [
                    jnp.matmul(
                        own[pl.ds(r0, top_tile), :],
                        router_v[...],
                        preferred_element_type=jnp.float32,
                    )
                    for r0 in range(0, rows, top_tile)
                ]
                tops = [_Top16(lg, bias_v) for lg in logits_tiles]
                latent_progress_fn()
                peer_input.wait_send()
                peer_input.wait_recv()
                gu_copy(0, 0, gate_buffers, gate_sems).start()
                gu_copy(0, 3072, up_buffers, up_sems).start()
                latent_progress_fn()
                if w4a8_latent:
                    next(scale_steps, None)
                route_steps = iter(())
                route_started = False
                scale_out_started = False
                gu_row_tile = min(rows, 256)
                top_steps_per_panel = 4 if defer_shared_down else 2
                for panel in range(12):
                    bank = panel % 2
                    gu_copy(panel, 0, gate_buffers, gate_sems).wait()
                    gu_copy(panel, 3072, up_buffers, up_sems).wait()
                    if panel + 1 < 12:
                        gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
                        gu_copy(panel + 1, 3072, up_buffers, up_sems).start()
                    if (
                        w4a8_latent
                        and not scale_out_started
                        and next(scale_steps, None) is None
                    ):
                        pltpu.make_async_copy(
                            s["lat_scales_v"],
                            latent_scales_hbm,
                            s["scale_out_sem"],
                        ).start()
                        scale_out_started = True
                    if route_started:
                        next(route_steps, None)
                    if defer_shared_down:
                        mid_stage = fs["mid_stage"]
                        mid_stage_sems = fs["mid_stage_sems"]
                        for half in range(2):
                            slot = 2 * panel + half
                            s_bank = slot % 2
                            if slot >= 2:
                                prev_panel, prev_half = divmod(slot - 2, 2)
                                shared_down_16k.stage_middle_half_copy(
                                    mid_stage[s_bank],
                                    shared,
                                    mid_stage_sems[s_bank],
                                    half=prev_half,
                                    panel=prev_panel,
                                ).wait()
                            for r_sub in range(0, rows, gu_row_tile):
                                row = half * rows + r_sub
                                lhs = pair_v[pl.ds(row, gu_row_tile), :]
                                gate = jnp.matmul(
                                    lhs,
                                    gate_buffers[bank][...],
                                    preferred_element_type=jnp.float32,
                                )
                                up = jnp.matmul(
                                    lhs,
                                    up_buffers[bank][...],
                                    preferred_element_type=jnp.float32,
                                )
                                activated = (4.0 * jnp.tanh(gate / 4.0)) * (
                                    1.0 / (1.0 + jnp.exp(-gate))
                                )
                                activated *= 25.0 * jnp.tanh(up / 25.0)
                                mid_stage[s_bank][pl.ds(r_sub, gu_row_tile), :] = (
                                    activated.astype(jnp.bfloat16)
                                )
                            shared_down_16k.stage_middle_half_copy(
                                mid_stage[s_bank],
                                shared,
                                mid_stage_sems[s_bank],
                                half=half,
                                panel=panel,
                            ).start()
                            latent_progress_fn()
                    else:
                        for row in range(0, 2 * rows, gu_row_tile):
                            lhs = pair_v[pl.ds(row, gu_row_tile), :]
                            gate = jnp.matmul(
                                lhs,
                                gate_buffers[bank][...],
                                preferred_element_type=jnp.float32,
                            )
                            up = jnp.matmul(
                                lhs,
                                up_buffers[bank][...],
                                preferred_element_type=jnp.float32,
                            )
                            activated = (4.0 * jnp.tanh(gate / 4.0)) * (
                                1.0 / (1.0 + jnp.exp(-gate))
                            )
                            activated *= 25.0 * jnp.tanh(up / 25.0)
                            s["middle"][
                                pl.ds(row, gu_row_tile), pl.ds(panel * 256, 256)
                            ] = activated.astype(jnp.bfloat16)
                            if row % rows == 0:
                                latent_progress_fn()
                    for top in tops:
                        for _ in range(min(top_steps_per_panel, 16 - top.index)):
                            top.step()
                    if tops[-1].index == 16 and not route_started:
                        ids_parts, probs_parts = [], []
                        for t_idx, top in enumerate(tops):
                            p_ids, p_probs = top.finish(
                                valid[t_idx * top_tile : (t_idx + 1) * top_tile]
                            )
                            ids_parts.append(p_ids)
                            probs_parts.append(p_probs)
                        ids = jnp.concatenate(ids_parts, axis=0)
                        probs = jnp.concatenate(probs_parts, axis=0)
                        wire = jnp.concatenate(
                            (
                                ids.reshape(rows // 8, 128),
                                jax.lax.bitcast_convert_type(probs, jnp.int32).reshape(
                                    rows // 8, 128
                                ),
                            ),
                            axis=0,
                        )
                        fs["route_payload"][...] = wire.reshape(rows // 4, 128)
                        route_steps = _route_gather_steps(
                            fs["route_payload"],
                            s["route_wire"],
                            s["route_send"],
                            s["route_recv"],
                            rank=rank,
                        )
                        next(route_steps, None)
                        route_started = True
                if defer_shared_down:
                    for tail_slot in (22, 23):
                        t_panel, t_half = divmod(tail_slot, 2)
                        t_bank = tail_slot % 2
                        shared_down_16k.stage_middle_half_copy(
                            fs["mid_stage"][t_bank],
                            shared,
                            fs["mid_stage_sems"][t_bank],
                            half=t_half,
                            panel=t_panel,
                        ).wait()
                if w4a8_latent and not scale_out_started:
                    for _ in scale_steps:
                        pass
                    pltpu.make_async_copy(
                        s["lat_scales_v"], latent_scales_hbm, s["scale_out_sem"]
                    ).start()
                for _ in route_steps:
                    latent_progress_fn()
                return

            buffers, router_v = fs["gu_buffers"], fs["router"]
            copy = (
                pltpu.make_async_copy(hidden, own, s["input_sem"])
                if pair_buf is None
                else None
            )
            router_copy = pltpu.make_async_copy(router, router_v, fs["gu_sems"][0])
            bias_copy = pltpu.make_async_copy(bias, fs["bias"], fs["gu_sems"][1])
            if copy is not None:
                copy.start()
            router_copy.start()
            bias_copy.start()
            _enter_mesh()
            pl.semaphore_signal(
                s["peer_credit"],
                1,
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            if copy is not None:
                copy.wait()
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
            bias_copy.wait()
            gate_buffers, up_buffers = buffers[:2], buffers[2:]
            gate_sems, up_sems = fs["gu_sems"][:2], fs["gu_sems"][2:]
            bias_v = fs["bias"][...]
            top_tile = min(rows, 256)
            logits_tiles = [
                jnp.matmul(
                    own[pl.ds(r0, top_tile), :],
                    router_v[...],
                    preferred_element_type=jnp.float32,
                )
                for r0 in range(0, rows, top_tile)
            ]
            tops = [_Top16(lg, bias_v) for lg in logits_tiles]
            route_steps = iter(())
            route_started = False

            latent_products = _dense_panels(
                own[...], latent_weight, gate_buffers, gate_sems, columns=3584
            )
            for col, product in latent_products:
                if route_started:
                    stage = next(route_steps, None)
                    if stage is None:
                        start_metadata()
                if metadata_started:
                    metadata_progress()
                fs["latent"][:, pl.ds(col, 256)] = product.astype(jnp.bfloat16)
                for top in tops:
                    for _ in range(min(2, 16 - top.index)):
                        top.step()
                if tops[-1].index == 16 and not route_started:
                    ids_parts, probs_parts = [], []
                    for t_idx, top in enumerate(tops):
                        p_ids, p_probs = top.finish(
                            valid[t_idx * top_tile : (t_idx + 1) * top_tile]
                        )
                        ids_parts.append(p_ids)
                        probs_parts.append(p_probs)
                    ids = jnp.concatenate(ids_parts, axis=0)
                    probs = jnp.concatenate(probs_parts, axis=0)
                    if rows < 128:
                        wire = jnp.concatenate(
                            (ids, jax.lax.bitcast_convert_type(probs, jnp.int32)),
                            axis=1,
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
                    fs["route_payload"][...] = wire.reshape(rows // 4, 128)
                    route_steps = _route_gather_steps(
                        fs["route_payload"],
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
            s["latent_payload"][...] = fs["latent"][...].reshape(rows, 2, 1792)
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

            def latent_progress(*, drain=False):
                if drain:
                    for _ in latent_steps:
                        pass
                else:
                    next(latent_steps, None)

            latent_progress_fn = latent_progress
            peer_input.wait_send()
            peer_input.wait_recv()
            gu_copy(0, 0, gate_buffers, gate_sems).start()
            gu_copy(0, 3072, up_buffers, up_sems).start()
            gu_row_tile = min(rows, 256)
            for panel in range(12):
                latent_progress_fn()
                bank = panel % 2
                gu_copy(panel, 0, gate_buffers, gate_sems).wait()
                gu_copy(panel, 3072, up_buffers, up_sems).wait()
                if panel + 1 < 12:
                    gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
                    gu_copy(panel + 1, 3072, up_buffers, up_sems).start()
                for row in range(0, 2 * rows, gu_row_tile):
                    lhs = pair_v[pl.ds(row, gu_row_tile), :]
                    gate = jnp.matmul(
                        lhs, gate_buffers[bank][...], preferred_element_type=jnp.float32
                    )
                    up = jnp.matmul(
                        lhs, up_buffers[bank][...], preferred_element_type=jnp.float32
                    )
                    if row % rows == 0:
                        metadata_progress()
                    activated = (4.0 * jnp.tanh(gate / 4.0)) * (
                        1.0 / (1.0 + jnp.exp(-gate))
                    )
                    activated *= 25.0 * jnp.tanh(up / 25.0)
                    s["middle"][pl.ds(row, gu_row_tile), pl.ds(panel * 256, 256)] = (
                        activated.astype(jnp.bfloat16)
                    )

        def run_fp8_front_and_gu(fp8_outer):
            nonlocal latent_progress_fn
            top_tile = min(rows, 256)
            gu_cols = 256 if rows >= 512 else 512
            n_gu_panels = 3072 // gu_cols

            def run_router(rs):
                copy = (
                    pltpu.make_async_copy(hidden, own, s["input_sem"])
                    if pair_buf is None
                    else None
                )
                router_copy = pltpu.make_async_copy(router, rs["router"], rs["sems"][0])
                bias_copy = pltpu.make_async_copy(bias, rs["bias"], rs["sems"][1])
                if copy is not None:
                    copy.start()
                router_copy.start()
                bias_copy.start()
                _enter_mesh()
                pl.semaphore_signal(
                    s["peer_credit"],
                    1,
                    device_id=peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                if copy is not None:
                    copy.wait()
                pl.semaphore_wait(s["peer_credit"], 1)
                pltpu.make_async_remote_copy(
                    own,
                    own,
                    s["peer_send"],
                    s["peer_recv"],
                    device_id=peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                ).start()
                router_copy.wait()
                bias_copy.wait()
                bias_v = rs["bias"][...]
                logits_tiles = [
                    jnp.matmul(
                        own[pl.ds(r0, top_tile), :],
                        rs["router"][...],
                        preferred_element_type=jnp.float32,
                    )
                    for r0 in range(0, rows, top_tile)
                ]
                tops = [_Top16(lg, bias_v) for lg in logits_tiles]
                for _ in range(16):
                    for top in tops:
                        top.step()
                ids_parts, probs_parts = [], []
                for t_idx, top in enumerate(tops):
                    p_ids, p_probs = top.finish(
                        valid[t_idx * top_tile : (t_idx + 1) * top_tile]
                    )
                    ids_parts.append(p_ids)
                    probs_parts.append(p_probs)
                ids = jnp.concatenate(ids_parts, axis=0)
                probs = jnp.concatenate(probs_parts, axis=0)
                if rows < 128:
                    wire = jnp.concatenate(
                        (ids, jax.lax.bitcast_convert_type(probs, jnp.int32)),
                        axis=1,
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
                fp8_outer["route_payload"][...] = wire.reshape(rows // 4, 128)

            pl.run_scoped(
                run_router,
                dict(
                    router=vmem((hidden_size, 896), jnp.bfloat16),
                    bias=vmem((1, 896), jnp.float32),
                    sems=(dma, dma),
                ),
            )

            def run_latent_and_gu(fs):
                nonlocal latent_progress_fn
                assert latent_weight_scale is not None
                assert shared_gu_scale is not None
                peer_input = pltpu.make_async_remote_copy(
                    own,
                    own,
                    s["peer_send"],
                    s["peer_recv"],
                    device_id=peer,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                ld_sc_copy = pltpu.make_async_copy(
                    latent_weight_scale, fs["ld_sc"], fs["sc_sems"][0]
                )
                gu_sc_copy = pltpu.make_async_copy(
                    shared_gu_scale, fs["gu_sc"], fs["sc_sems"][1]
                )
                ld_sc_copy.start()
                gu_sc_copy.start()
                route_steps = _route_gather_steps(
                    fp8_outer["route_payload"],
                    s["route_wire"],
                    s["route_send"],
                    s["route_recv"],
                    rank=rank,
                )
                next(route_steps, None)
                latent_buffers = tuple(
                    fs["gu_fp8"].at[:, pl.ds(i * 256, 256)] for i in range(4)
                )
                if rows >= 32:
                    _quantize_rows_to_fp8_vmem(
                        pair_v,
                        fs["pair_fp8"],
                        fs["pair_sc"],
                        rows=rows,
                        row_offset=parity * rows,
                    )
                    own_fp8 = fs["pair_fp8"].at[pl.ds(parity * rows, rows), :]
                    own_sc = fs["pair_sc"].at[pl.ds(parity * rows, rows), :]
                    ld_sc_copy.wait()
                    if rows >= 256:
                        latent_products = _latent_half_pair_panels_fp8(
                            own_fp8,
                            own_sc,
                            latent_weight,
                            fs["ld_sc"],
                            latent_buffers,
                            fs["gu_sems"],
                        )
                    else:
                        latent_products = _dense_panels_fp8(
                            own_fp8[...],
                            own_sc[:, :1],
                            latent_weight,
                            fs["ld_sc"],
                            latent_buffers[:2],
                            fs["gu_sems"][:2],
                            columns=3584,
                        )
                else:
                    own_x = own[...].astype(jnp.float32)
                    own_rmax = jnp.max(jnp.abs(own_x), axis=1, keepdims=True)
                    own_sc_val = jnp.maximum(own_rmax, _FP8_SCALE_EPS) / _FP8_MAX
                    own_fp8_val = (own_x * (1.0 / own_sc_val)).astype(_FP8_DTYPE)
                    ld_sc_copy.wait()
                    latent_products = _dense_panels_fp8(
                        own_fp8_val,
                        own_sc_val,
                        latent_weight,
                        fs["ld_sc"],
                        latent_buffers[:2],
                        fs["gu_sems"][:2],
                        columns=3584,
                    )
                first_half = None
                for col, product in latent_products:
                    stage = next(route_steps, None)
                    if rows <= 128 and stage is None:
                        start_metadata()
                    if rows <= 128 and metadata_started:
                        metadata_progress()
                    if rows >= 256:
                        fragment = product.astype(jnp.bfloat16)
                        if col < 1792:
                            first_half = fragment
                        else:
                            assert first_half is not None
                            paired = jnp.concatenate((first_half, fragment), axis=1)
                            s["latent_payload"][:, :, pl.ds(col - 1792, 256)] = (
                                paired.reshape(rows, 2, 256)
                            )
                    else:
                        fs["latent"][:, pl.ds(col, 256)] = product.astype(jnp.bfloat16)
                if rows >= 256:
                    latent_tick = 0

                    def latent_progress(*, drain=False):
                        nonlocal latent_tick
                        if latent_tick == 0:
                            _latent_half_init(
                                tables,
                                s["latent_payload"],
                                latent_hbm,
                                s["latent_send_sems"],
                                s["latent_recv_sems"],
                                s["latent_stores"],
                                s["latent_state"],
                                s["latent_meta"],
                                rank=rank,
                            )
                        else:
                            _latent_half_progress(
                                s["latent_payload"],
                                latent_hbm,
                                s["latent_send_sems"],
                                s["latent_recv_sems"],
                                s["latent_credits"],
                                s["latent_stores"],
                                s["latent_state"],
                                s["latent_meta"],
                                rank=rank,
                                drain=drain,
                            )
                        latent_tick += 1

                    latent_progress_fn = latent_progress
                    latent_progress()
                    for _ in route_steps:
                        latent_progress()
                    start_metadata()
                else:
                    for _ in route_steps:
                        pass
                    start_metadata()
                    s["latent_payload"][...] = fs["latent"][...].reshape(rows, 2, 1792)
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

                    def latent_progress(*, drain=False):
                        if drain:
                            for _ in latent_steps:
                                pass
                        else:
                            next(latent_steps, None)

                    latent_progress_fn = latent_progress

                peer_input.wait_send()
                peer_input.wait_recv()
                gate_buffers = tuple(
                    fs["gu_fp8"].at[:, pl.ds(i * gu_cols, gu_cols)] for i in range(2)
                )
                up_buffers = tuple(
                    fs["gu_fp8"].at[:, pl.ds((i + 2) * gu_cols, gu_cols)]
                    for i in range(2)
                )
                gate_sems, up_sems = fs["gu_sems"][:2], fs["gu_sems"][2:]

                def gu_copy(panel, offset, bufs, sems):
                    bank = panel % 2
                    return pltpu.make_async_copy(
                        shared_gu.at[:, pl.ds(offset + panel * gu_cols, gu_cols)],
                        bufs[bank],
                        sems[bank],
                    )

                gu_copy(0, 0, gate_buffers, gate_sems).start()
                gu_copy(0, 3072, up_buffers, up_sems).start()
                if rows >= 32:
                    _quantize_rows_to_fp8_vmem(
                        pair_v,
                        fs["pair_fp8"],
                        fs["pair_sc"],
                        rows=rows,
                        row_offset=(1 - parity) * rows,
                    )
                else:
                    _quantize_rows_to_fp8_vmem(
                        pair_v,
                        fs["pair_fp8"],
                        fs["pair_sc"],
                        rows=2 * rows,
                        row_offset=0,
                    )
                gu_sc_copy.wait()
                gu_row_tile = min(2 * rows, min(max(rows, 32), 256))
                for row in range(0, 2 * rows, gu_row_tile):
                    s["mid_max"][pl.ds(row, gu_row_tile), :] = jnp.full(
                        (gu_row_tile, 128), _FP8_SCALE_EPS, jnp.float32
                    )
                if rows >= 256:
                    latent_progress_fn()
                for panel in range(n_gu_panels):
                    if rows <= 128:
                        latent_progress_fn()
                        latent_progress_fn()
                    bank = panel % 2
                    gu_copy(panel, 0, gate_buffers, gate_sems).wait()
                    gu_copy(panel, 3072, up_buffers, up_sems).wait()
                    if panel + 1 < n_gu_panels:
                        gu_copy(panel + 1, 0, gate_buffers, gate_sems).start()
                        gu_copy(panel + 1, 3072, up_buffers, up_sems).start()
                    g_sc = fs["gu_sc"][:, pl.ds(panel * gu_cols, gu_cols)]
                    u_sc = fs["gu_sc"][:, pl.ds(3072 + panel * gu_cols, gu_cols)]
                    for row in range(0, 2 * rows, gu_row_tile):
                        lhs = fs["pair_fp8"][pl.ds(row, gu_row_tile), :]
                        x_sc = fs["pair_sc"][pl.ds(row, gu_row_tile), :1]
                        gate = (
                            jnp.matmul(
                                lhs,
                                gate_buffers[bank][...],
                                preferred_element_type=jnp.float32,
                            )
                            * x_sc
                            * g_sc
                        )
                        up = (
                            jnp.matmul(
                                lhs,
                                up_buffers[bank][...],
                                preferred_element_type=jnp.float32,
                            )
                            * x_sc
                            * u_sc
                        )
                        if gu_cols == 512 or row % rows == 0:
                            metadata_progress()
                        if rows <= 256:
                            metadata_progress()
                        activated = (4.0 * jnp.tanh(gate / 4.0)) * (
                            1.0 / (1.0 + jnp.exp(-gate))
                        )
                        activated *= 25.0 * jnp.tanh(up / 25.0)
                        s["middle"][
                            pl.ds(row, gu_row_tile), pl.ds(panel * gu_cols, gu_cols)
                        ] = activated.astype(jnp.bfloat16)
                        pmax = jnp.max(jnp.abs(activated), axis=1, keepdims=True)
                        cur_max = s["mid_max"][pl.ds(row, gu_row_tile), :]
                        s["mid_max"][pl.ds(row, gu_row_tile), :] = jnp.maximum(
                            cur_max, pmax
                        )
                        if rows >= 256 and (gu_cols == 512 or row % rows == 0):
                            latent_progress_fn()

            pl.run_scoped(
                run_latent_and_gu,
                dict(
                    gu_fp8=vmem((hidden_size, 4 * gu_cols), _FP8_DTYPE),
                    pair_fp8=vmem((2 * rows, hidden_size), _FP8_DTYPE),
                    pair_sc=vmem((2 * rows, 128), jnp.float32),
                    ld_sc=vmem((1, 3584), jnp.float32),
                    gu_sc=vmem((1, 6144), jnp.float32),
                    gu_sems=tuple(dma for _ in range(4)),
                    sc_sems=(dma, dma),
                    latent=vmem((rows, 3584), jnp.bfloat16) if rows <= 128 else None,
                ),
            )

        if dense_fp8:
            pl.run_scoped(
                run_fp8_front_and_gu,
                dict(
                    route_payload=vmem((rows // 4, 128), jnp.int32),
                ),
            )
        else:
            pl.run_scoped(
                run_front_and_gu,
                dict(
                    router=vmem((hidden_size, 896), jnp.bfloat16)
                    if rows <= 128
                    else None,
                    router_gu=vmem((hidden_size, 1024), jnp.bfloat16)
                    if rows >= 256
                    else None,
                    gu_buffers=tuple(
                        vmem((hidden_size, 256), jnp.bfloat16) for _ in range(4)
                    )
                    if rows <= 128
                    else (),
                    gu_sems=tuple(dma for _ in range(4)),
                    bias=vmem((1, 896), jnp.float32),
                    latent=vmem((rows, 3584), jnp.bfloat16) if rows <= 128 else None,
                    route_payload=vmem((rows // 4, 128), jnp.int32),
                    lat_scale_payload=vmem((rows // 128, 128), jnp.float32)
                    if w4a8_latent
                    else None,
                    mid_stage=tuple(vmem((rows, 256), jnp.bfloat16) for _ in range(2))
                    if defer_shared_down
                    else (),
                    mid_stage_sems=(dma, dma) if defer_shared_down else (),
                ),
            )

        def advance_latent_stage():
            latent_progress_fn()

        replica_scratch = (
            metadata_compact.ReplicaRoutingScratch(
                group_loads=s["rep_loads"],
                my_drop_experts=s["rep_drop_e"],
                my_add_experts=s["rep_add_e"],
                my_add_slots=s["rep_add_slot"],
                my_counts=s["rep_cnts"],
            )
            if replica_table is not None
            else None
        )
        if defer_shared_down:
            metadata_compact.run_sequential_compact_metadata(
                s["route_wire"],
                s["keys"],
                probabilities,
                token_ids,
                counts,
                maximum,
                s["metadata_cursor"],
                tokens=rows * 32,
                rank=rank,
                counts_ready=counts_ready,
                local_experts=counts.shape[0],
                on_poll=advance_latent_stage,
                replica_table=replica_table,
                replica_slots=replica_slots,
                replica_scratch=replica_scratch,
            )
            metadata_ready()
            if w4a8_latent:
                pltpu.make_async_copy(
                    s["lat_scales_v"], latent_scales_hbm, s["scale_out_sem"]
                ).wait()
            latent_progress_fn(drain=True)
            return

        compact_schedule = (
            metadata_compact.build_compact_metadata_schedule(
                s["route_wire"],
                s["keys"],
                probabilities,
                token_ids,
                counts,
                maximum,
                s["metadata_cursor"],
                tokens=rows * 32,
                rank=rank,
                counts_ready=counts_ready,
                local_experts=counts.shape[0],
                on_tail_stage=advance_latent_stage,
                replica_table=replica_table,
                replica_slots=replica_slots,
                replica_scratch=replica_scratch,
            )
            if use_compact_metadata
            else None
        )
        compact_step_idx = [0]

        def compact_metadata_step():
            if compact_schedule is None:
                return
            idx = compact_step_idx[0]
            if idx < len(compact_schedule.steps):
                compact_step_idx[0] = idx + 1
                compact_schedule.steps[idx]()

        def run_down(ds):
            _await_partner_scope(s["down_ready"], peer)
            shared_pending = [False, False]

            def down_copy(panel):
                return pltpu.make_async_copy(
                    shared_down.at[:, pl.ds(panel * 256, 256)],
                    ds["down_buffers"][panel % 2],
                    ds["down_sems"][panel % 2],
                )

            def finish_reduction(pending):
                transfer, bank, panel = pending
                own_part = ds["partial"][bank][pl.ds(parity * rows, rows), :]
                transfer.wait_send()
                transfer.wait_recv()
                reduced_panel = (own_part + ds["peer_y"][bank][...]).astype(
                    jnp.bfloat16
                )
                if not shared_in_hbm:
                    shared[:, pl.ds(panel * 256, 256)] = reduced_panel
                    return
                s_bank = panel % 2
                if shared_pending[s_bank]:
                    pltpu.make_async_copy(
                        ds["shared_tiles"][s_bank],
                        shared.at[:, pl.ds((panel - 2) * 256, 256)],
                        ds["shared_sems"][s_bank],
                    ).wait()
                ds["shared_tiles"][s_bank][...] = reduced_panel
                pltpu.make_async_copy(
                    ds["shared_tiles"][s_bank],
                    shared.at[:, pl.ds(panel * 256, 256)],
                    ds["shared_sems"][s_bank],
                ).start()
                shared_pending[s_bank] = True

            down_copy(0).start()
            pending = None
            down_row_tile = min(rows, 256)
            for panel in range(28):
                bank = panel % 2
                down_copy(panel).wait()
                if panel + 1 < 28:
                    down_copy(panel + 1).start()
                for row in range(0, 2 * rows, down_row_tile):
                    product = jnp.matmul(
                        s["middle"][pl.ds(row, down_row_tile), :],
                        ds["down_buffers"][bank][...],
                        preferred_element_type=jnp.float32,
                    )
                    if row % rows == 0:
                        metadata_progress()
                    ds["partial"][bank][pl.ds(row, down_row_tile), :] = product
                    compact_metadata_step()
                    if row + down_row_tile == 2 * rows:
                        if pending is not None:
                            finish_reduction(pending)
                        transfer = pltpu.make_async_remote_copy(
                            ds["partial"][bank].at[pl.ds((1 - parity) * rows, rows), :],
                            ds["peer_y"][bank],
                            ds["reduce_send"][bank],
                            ds["reduce_recv"][bank],
                            device_id=peer,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        transfer.start()
                        pending = transfer, bank, panel
                    if (rows >= 256 and row % rows == 0) or (
                        row + down_row_tile == 2 * rows
                    ):
                        latent_progress_fn()
            finish_reduction(pending)
            if shared_in_hbm:
                for tail_panel in (26, 27):
                    s_bank = tail_panel % 2
                    pltpu.make_async_copy(
                        ds["shared_tiles"][s_bank],
                        shared.at[:, pl.ds(tail_panel * 256, 256)],
                        ds["shared_sems"][s_bank],
                    ).wait()

        def run_down_fp8(ds):
            assert shared_down_scale is not None
            _await_partner_scope(s["down_ready"], peer)
            shared_pending = [False, False]

            def down_copy(panel):
                return pltpu.make_async_copy(
                    shared_down.at[:, pl.ds(panel * 512, 512)],
                    ds["down_buffers"][panel % 2],
                    ds["down_sems"][panel % 2],
                )

            def finish_reduction(pending):
                transfer, bank, panel = pending
                own_part = ds["partial"][bank][pl.ds(parity * rows, rows), :]
                transfer.wait_send()
                transfer.wait_recv()
                reduced_panel = (own_part + ds["peer_y"][bank][...]).astype(
                    jnp.bfloat16
                )
                if not shared_in_hbm:
                    shared[:, pl.ds(panel * 512, 512)] = reduced_panel
                    return
                s_bank = panel % 2
                if shared_pending[s_bank]:
                    pltpu.make_async_copy(
                        ds["shared_tiles"][s_bank],
                        shared.at[:, pl.ds((panel - 2) * 512, 512)],
                        ds["shared_sems"][s_bank],
                    ).wait()
                ds["shared_tiles"][s_bank][...] = reduced_panel
                pltpu.make_async_copy(
                    ds["shared_tiles"][s_bank],
                    shared.at[:, pl.ds(panel * 512, 512)],
                    ds["shared_sems"][s_bank],
                ).start()
                shared_pending[s_bank] = True

            down_sc_copy = pltpu.make_async_copy(
                shared_down_scale, ds["down_sc"], ds["sc_sem"]
            )
            down_sc_copy.start()
            down_copy(0).start()
            q_chunk = min(2 * rows, 128)
            for r0 in range(0, 2 * rows, q_chunk):
                sc = s["mid_max"][pl.ds(r0, q_chunk), :1] / _FP8_MAX
                ds["mid_sc"][pl.ds(r0, q_chunk), :] = jnp.broadcast_to(
                    sc, (q_chunk, 128)
                )
                inv_sc = 1.0 / sc
                ds["mid_fp8"][pl.ds(r0, q_chunk), :] = (
                    s["middle"][pl.ds(r0, q_chunk), :].astype(jnp.float32) * inv_sc
                ).astype(_FP8_DTYPE)
            down_sc_copy.wait()
            pending = None
            down_row_tile = min(2 * rows, min(max(rows, 32), 256))
            for panel in range(14):
                bank = panel % 2
                down_copy(panel).wait()
                if panel + 1 < 14:
                    down_copy(panel + 1).start()
                d_sc = ds["down_sc"][:, pl.ds(panel * 512, 512)]
                for row in range(0, 2 * rows, down_row_tile):
                    m_sc = ds["mid_sc"][pl.ds(row, down_row_tile), :1]
                    product = (
                        jnp.matmul(
                            ds["mid_fp8"][pl.ds(row, down_row_tile), :],
                            ds["down_buffers"][bank][...],
                            preferred_element_type=jnp.float32,
                        )
                        * m_sc
                        * d_sc
                    )
                    metadata_progress()
                    if rows <= 256:
                        metadata_progress()
                    ds["partial"][bank][pl.ds(row, down_row_tile), :] = product
                    compact_metadata_step()
                    if row + down_row_tile == 2 * rows:
                        if pending is not None:
                            finish_reduction(pending)
                        transfer = pltpu.make_async_remote_copy(
                            ds["partial"][bank].at[pl.ds((1 - parity) * rows, rows), :],
                            ds["peer_y"][bank],
                            ds["reduce_send"][bank],
                            ds["reduce_recv"][bank],
                            device_id=peer,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )
                        transfer.start()
                        pending = transfer, bank, panel
                    if rows >= 256:
                        latent_progress_fn()
                    elif row + down_row_tile == 2 * rows:
                        latent_progress_fn()
                        latent_progress_fn()
            finish_reduction(pending)
            if shared_in_hbm:
                for tail_panel in (12, 13):
                    s_bank = tail_panel % 2
                    pltpu.make_async_copy(
                        ds["shared_tiles"][s_bank],
                        shared.at[:, pl.ds(tail_panel * 512, 512)],
                        ds["shared_sems"][s_bank],
                    ).wait()

        if dense_fp8:
            pl.run_scoped(
                run_down_fp8,
                dict(
                    mid_fp8=vmem((2 * rows, 3072), _FP8_DTYPE),
                    mid_sc=vmem((2 * rows, 128), jnp.float32),
                    down_sc=vmem((1, 7168), jnp.float32),
                    sc_sem=dma,
                    down_buffers=tuple(vmem((3072, 512), _FP8_DTYPE) for _ in range(2)),
                    down_sems=(dma, dma),
                    partial=tuple(vmem((2 * rows, 512), jnp.float32) for _ in range(2)),
                    peer_y=tuple(vmem((rows, 512), jnp.float32) for _ in range(2)),
                    reduce_send=(dma, dma),
                    reduce_recv=(dma, dma),
                    shared_tiles=tuple(
                        vmem((rows, 512), jnp.bfloat16) for _ in range(2)
                    )
                    if shared_in_hbm
                    else (),
                    shared_sems=(dma, dma) if shared_in_hbm else (),
                ),
            )
        else:
            pl.run_scoped(
                run_down,
                dict(
                    down_buffers=tuple(
                        vmem((3072, 256), jnp.bfloat16) for _ in range(2)
                    ),
                    down_sems=(dma, dma),
                    partial=tuple(vmem((2 * rows, 256), jnp.float32) for _ in range(2)),
                    peer_y=tuple(vmem((rows, 256), jnp.float32) for _ in range(2)),
                    reduce_send=(dma, dma),
                    reduce_recv=(dma, dma),
                    shared_tiles=tuple(
                        vmem((rows, 256), jnp.bfloat16) for _ in range(2)
                    )
                    if shared_in_hbm
                    else (),
                    shared_sems=(dma, dma) if shared_in_hbm else (),
                ),
            )
        if compact_schedule is not None:
            while compact_step_idx[0] < len(compact_schedule.steps):
                compact_metadata_step()
                latent_progress_fn()
            compact_schedule.tail()
            # Start compact VMEM-to-SMEM transfers while the latent ring finishes.
            metadata_ready()
            if w4a8_latent:
                pltpu.make_async_copy(
                    s["lat_scales_v"], latent_scales_hbm, s["scale_out_sem"]
                ).wait()
            latent_progress_fn(drain=True)
        else:
            latent_progress_fn(drain=True)
            for _ in metadata_steps:
                pass

    rep_base_le = counts.shape[0] - replica_slots if replica_table is not None else 0
    rep_ep_groups = 896 // rep_base_le if replica_table is not None else 0
    rep_max_drops = (
        min(rep_base_le, rep_ep_groups * replica_slots)
        if replica_table is not None
        else 0
    )
    scratch = dict(
        pair=vmem((2 * rows, hidden_size), jnp.bfloat16) if pair_buf is None else None,
        input_sem=dma if pair_buf is None else None,
        peer_send=dma,
        peer_recv=dma,
        peer_credit=pltpu.SemaphoreType.REGULAR,
        down_ready=None if defer_shared_down else pltpu.SemaphoreType.REGULAR,
        route_wire=vmem((rows * 8, 128), jnp.int32),
        route_send=dma((5,)),
        route_recv=dma((5,)),
        keys=vmem((rows * 4, 128), jnp.int32),
        metadata_cursor=pltpu.SMEM((1,), jnp.int32) if rows >= 256 else None,
        rep_loads=pltpu.SMEM((rep_ep_groups,), jnp.int32)
        if replica_table is not None
        else None,
        rep_drop_e=pltpu.SMEM((rep_max_drops,), jnp.int32)
        if replica_table is not None
        else None,
        rep_add_e=pltpu.SMEM((replica_slots,), jnp.int32)
        if replica_table is not None
        else None,
        rep_add_slot=pltpu.SMEM((replica_slots,), jnp.int32)
        if replica_table is not None
        else None,
        rep_cnts=pltpu.SMEM((2,), jnp.int32) if replica_table is not None else None,
        latent_payload=vmem((rows, *latent_hbm.shape[1:]), latent_hbm.dtype),
        latent_recv=vmem((2, rows, 2, 1792), jnp.bfloat16) if rows <= 128 else None,
        latent_state=pltpu.SMEM((13,), jnp.int32) if rows >= 256 else None,
        latent_meta=pltpu.SMEM((6 * _HALF_LATENT_META_STRIDE,), jnp.int32)
        if rows >= 256
        else None,
        latent_send_sems=dma((6 if rows >= 256 else 12,)),
        latent_recv_sems=dma((2, 6 if rows >= 256 else 12)),
        latent_credits=pltpu.SemaphoreType.REGULAR((6 if rows >= 256 else 12,)),
        latent_stores=dma((12,)) if rows <= 128 else dma,
        lat_scales_v=vmem(w4a8_16k.LATENT_SCALES_SHAPE, jnp.float32)
        if w4a8_latent
        else None,
        scale_send=dma((5,)) if w4a8_latent else None,
        scale_recv=dma((5,)) if w4a8_latent else None,
        scale_out_sem=dma if w4a8_latent else None,
        middle=None if defer_shared_down else vmem((2 * rows, 3072), jnp.bfloat16),
        mid_max=vmem((2 * rows, 128), jnp.float32) if dense_fp8 else None,
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
    subtotal_tail_hbm=None,
    latent_scales_hbm=None,
    deferred_shared_hbm=None,
    deferred_shared_down_hbm=None,
    last_ring_overlap_fn: Callable[[Callable[[], None]], None] | None = None,
):
    """Joint GU/SiTU/first-down pipeline followed by remaining columns."""
    if tokens == routed_16k.TOKENS_16K:
        routed_16k.routed_experts_16k(
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
            probabilities_ready=probabilities_ready,
            subtotal_hbm=subtotal_tail_hbm,
            compact_capacity=_compact_capacity(tokens),
            rmw_batch=_RMW_BATCH,
            rmw_dummy_slots=_RMW_DUMMY_SLOTS,
            latent_scales_hbm=latent_scales_hbm,
            deferred_shared_hbm=deferred_shared_hbm,
            deferred_shared_down_hbm=deferred_shared_down_hbm,
            last_ring_overlap_fn=last_ring_overlap_fn,
        )
        return
    gu_width = gu_weights.shape[-1]
    assert gu_width % 2 == 0
    mid_width = gu_width // 2
    assert rhs.shape[-2] * 8 == mid_width
    ep8_8k = tokens == 8192 and mid_width == 768
    columns = 1792 if (tokens < 8192 or ep8_8k) else 896
    sums_dtype = jnp.bfloat16 if ep8_8k else jnp.float32
    tile = 128 if tokens < 8192 else 256
    assert tokens in (512, 1024, 2048, 4096, 8192)
    unroll = 8 if tokens < 8192 else (16 if mid_width == 768 else 32)
    gu_unroll = 4 if tokens < 8192 else 8
    first_down_unroll = 2 if tokens < 8192 else 8
    first_tail_unroll = 8 if tokens < 8192 else (16 if mid_width == 768 else 8)
    metadata_row_capacity = 0 if tokens < 8192 else _compact_capacity(tokens)
    ring_dtype = jnp.bfloat16
    x = jax.ShapeDtypeStruct((tokens * 16, mid_width), latent.dtype)
    rows, width = (metadata_row_capacity or x.shape[0], rhs.shape[-1])
    assert rows <= x.shape[0]
    tm, tn, localrows = (tile, columns, tokens // 32)
    packed_half = 512 if tn == 896 else tn // 2
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
    regions = k._regions(localrows, tn)  # noqa: SLF001 -- existing ring helper
    regions_half = (
        k._regions(localrows, tn // 2)  # noqa: SLF001 -- existing ring helper
        if ep8_8k
        else None
    )
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
        tiles=dataclasses.replace(
            gu_cfg.tiles, tile_m=tm, tile_n=mid_width, tile_k=3584
        ),
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
        state,
        ring_meta,
        moved_smem,
        phase_sems,
        phase_smem,
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
        gu = k.FusedWeightsRef(
            gate=gu, up=jax.tree.map(lambda r: r.at[..., mid_width:], gu)
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
        phase_smem[0] = 0
        phase_smem[1] = 0

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

        def signal_ring_peers():
            sem = phase_sems.at[phase_smem[0] & 1]
            for peer in peers:
                pl.semaphore_signal(
                    sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )

        def wait_ring_peers():
            slot = phase_smem[0]
            pl.semaphore_wait(phase_sems.at[slot & 1], len(peers))
            phase_smem[0] = slot + 1

        def sync_ring_peers():
            signal_ring_peers()
            wait_ring_peers()

        # Real-prompt tuning: short expert loops need all twelve streams per
        # poll; longer loops alternate six-stream groups to bound polling cost.
        def advance_slice(
            x_slice,
            o_slice,
            payload_buf,
            rx_buf,
            ring_regions,
            *,
            drain=False,
            poll_round=0,
        ):
            _ready_reduce(
                tables,
                x_slice,
                o_slice,
                sends,
                receives,
                credits,
                stores,
                payload_buf,
                rx_buf,
                streams=ring_regions,
                buffers=2,
                state=state,
                ring_meta=ring_meta,
                drain=drain,
                poll_round=poll_round,
                sparse_poll=(ng >= sparse_poll_threshold(split_ring=ep8_8k)),
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

        def route(i):
            return (destinations[i], probabilities[i])

        def batched_unclamped(source, begin, base_idx, width_b):
            idxs = [base_idx + r for r in range(width_b)]
            routes = [route(i_r) for i_r in idxs]
            if sums_dtype == jnp.bfloat16:
                vals = [source.at[pl.ds(i_r - begin, 1)][...] for i_r in idxs]
                olds = [sums.at[pl.ds(d_r, 1)][...] for d_r, _ in routes]
                accs = [
                    olds[r] + vals[r] * routes[r][1].astype(jnp.bfloat16)
                    for r in range(width_b)
                ]
            else:
                vals = [
                    source.at[pl.ds(i_r - begin, 1)][...].astype(jnp.float32)
                    for i_r in idxs
                ]
                olds = [sums.at[pl.ds(d_r, 1)][...] for d_r, _ in routes]
                accs = [olds[r] + vals[r] * routes[r][1] for r in range(width_b)]
            for r in range(width_b):
                sums.at[pl.ds(routes[r][0], 1)][...] = accs[r]

        def batched_clamped(
            source, begin, base_idx, end, width_b, *, guard, dummy_base=0
        ):
            idxs = [base_idx + r for r in range(width_b)]
            actives = [guard & (idx_r < end) for idx_r in idxs]
            clamped = [jnp.minimum(idx_r, end - 1) for idx_r in idxs]
            routes = [route(c_r) for c_r in clamped]
            dests = [
                jnp.where(
                    actives[r],
                    routes[r][0],
                    tokens + ((dummy_base + r) % _RMW_DUMMY_SLOTS),
                )
                for r in range(width_b)
            ]
            weights = [jnp.where(actives[r], routes[r][1], 0.0) for r in range(width_b)]
            if sums_dtype == jnp.bfloat16:
                vals = [source.at[pl.ds(c_r - begin, 1)][...] for c_r in clamped]
                olds = [sums.at[pl.ds(d_r, 1)][...] for d_r in dests]
                accs = [
                    olds[r] + vals[r] * weights[r].astype(jnp.bfloat16)
                    for r in range(width_b)
                ]
            else:
                vals = [
                    source.at[pl.ds(c_r - begin, 1)][...].astype(jnp.float32)
                    for c_r in clamped
                ]
                olds = [sums.at[pl.ds(d_r, 1)][...] for d_r in dests]
                accs = [olds[r] + vals[r] * weights[r] for r in range(width_b)]
            for r in range(width_b):
                sums.at[pl.ds(dests[r], 1)][...] = accs[r]

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
                    batched_unclamped(
                        source, begin, base + u, min(step_b, loop_unroll - u)
                    )

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
        gu_u_by_bucket = (4, 8, 12, 16) if ep8_8k else (gu_unroll,) * 4
        first_down_u_by_bucket = (2, 4, 8, 12) if ep8_8k else (first_down_unroll,) * 4
        rem_down_u_by_bucket = (4, 8, 14, 18) if ep8_8k else (unroll,) * 4

        def gu_work(g, panel, bucket_index=0):
            gu_u = gu_u_by_bucket[bucket_index]
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
            if ep8_8k and panel == gu_panels - 1:
                moved_smem[0] = gu_panels * gu_u

        def inner(
            x,
            w,
            metadata,
            *,
            first=False,
            subtotal_vmem=None,
            payload_buf=None,
            rx_buf=None,
        ):
            g, n = (pl.program_id(1), 0 if first else pl.program_id(0) + 1)
            if not first:
                assert subtotal_vmem is not None
                assert payload_buf is not None and rx_buf is not None

                @pl.when(g == 0)
                def clear():
                    sums[...] = jnp.zeros(sums.shape, sums_dtype)

                if ep8_8k:
                    half_tn = tn // 2
                    mid_g = ng // 2

                    @pl.when(g < mid_g)
                    def poll_h0():
                        advance_slice(
                            subtotal_vmem.at[:, pl.ds(0, half_tn)],
                            out.at[:, pl.ds(0, half_tn)],
                            payload_buf,
                            rx_buf,
                            regions_half,
                            poll_round=g,
                        )

                    @pl.when(g == mid_g)
                    def switch_h0_h1():
                        advance_slice(
                            subtotal_vmem.at[:, pl.ds(0, half_tn)],
                            out.at[:, pl.ds(0, half_tn)],
                            payload_buf,
                            rx_buf,
                            regions_half,
                            drain=True,
                        )
                        state[0] = 0
                        advance_slice(
                            subtotal_vmem.at[:, pl.ds(half_tn, half_tn)],
                            out.at[:, pl.ds(half_tn, half_tn)],
                            payload_buf,
                            rx_buf,
                            regions_half,
                            poll_round=0,
                        )

                    @pl.when(g > mid_g)
                    def poll_h1():
                        advance_slice(
                            subtotal_vmem.at[:, pl.ds(half_tn, half_tn)],
                            out.at[:, pl.ds(half_tn, half_tn)],
                            payload_buf,
                            rx_buf,
                            regions_half,
                            poll_round=g - mid_g,
                        )
                else:
                    advance_slice(
                        subtotal_vmem,
                        out.at[:, pl.ds((n - 1) * tn, tn)],
                        payload_buf,
                        rx_buf,
                        regions,
                        poll_round=g,
                    )
            current = packed.at[g % 2]
            previous = packed.at[(g + 1) % 2]
            previous_g = jnp.maximum(g - 1, 0)
            panel_count = pl.cdiv(tn, pltpu.get_tpu_info().mxu_column_size)
            down_unroll = first_down_unroll if first else unroll
            gu_moved = (
                moved_smem[0] if (first and ep8_8k) else (moved_rows if first else 0)
            )

            def overlap_work(panel, bucket_index=0):
                down_u = (first_down_u_by_bucket if first else rem_down_u_by_bucket)[
                    bucket_index
                ]
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
                if ep8_8k and panel == panel_count - 1:
                    moved_smem[0] = gu_moved + panel_count * down_u

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
                    moved_smem[0]
                    if ep8_8k
                    else ((moved_rows if first else 0) + panel_count * down_unroll),
                    loop_unroll=first_tail_unroll if first else unroll,
                )

            @pl.when(g == ng - 1)
            def finish():
                consume(g, current, loop_unroll=first_tail_unroll if first else unroll)
                if not first:
                    assert subtotal_vmem is not None
                    assert payload_buf is not None and rx_buf is not None
                    if ep8_8k:
                        half_tn = tn // 2
                        advance_slice(
                            subtotal_vmem.at[:, pl.ds(half_tn, half_tn)],
                            out.at[:, pl.ds(half_tn, half_tn)],
                            payload_buf,
                            rx_buf,
                            regions_half,
                            drain=True,
                        )
                        state[0] = 0
                    else:
                        advance_slice(
                            subtotal_vmem,
                            out.at[:, pl.ds((n - 1) * tn, tn)],
                            payload_buf,
                            rx_buf,
                            regions,
                            drain=True,
                        )
                        emit_subtotals(subtotal_vmem)
                        state[0] = 0

                        @pl.when(n == width // tn - 1)
                        def last():
                            advance_slice(
                                subtotal_vmem,
                                out.at[:, pl.ds(n * tn, tn)],
                                payload_buf,
                                rx_buf,
                                regions,
                                drain=True,
                            )

        (_, gu_ws), gu_ys = k.generate_block_specs(metadata, gu_cfg)
        gu_ws = k.FusedWeightsRef(gate=gu_ws, up=gu_ws)

        @functools.partial(
            pl.run_scoped,
            gu_partial=pltpu.VMEM((sub, mid_width), gu_cfg.out_dtype),
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

            def active_bounds(g):
                raw_start = metadata.gm_id_to_m_offset[g]
                raw_end = metadata.gm_id_to_m_offset[g + 1]
                begin = raw_start // sub * sub
                return (raw_start, raw_start - begin, raw_end - raw_start)

            def start(g, bank):
                raw_start, first, n_active = active_bounds(g)
                unroll = 16 if tokens >= 8192 else 8
                full = (n_active // unroll) * unroll

                @pl.loop(0, n_active // unroll)
                def row_vec(blk):
                    base = blk * unroll
                    sources = [
                        destinations[raw_start + base + u] for u in range(unroll)
                    ]
                    for u in range(unroll):
                        pltpu.make_async_copy(
                            latent.at[pl.ds(sources[u], 1)],
                            acts.at[bank, pl.ds(first + base + u, 1)],
                            act_sems.at[bank],
                        ).start()

                @pl.loop(full, n_active)
                def row_rem(i):
                    source = destinations[raw_start + i]
                    pltpu.make_async_copy(
                        latent.at[pl.ds(source, 1)],
                        acts.at[bank, pl.ds(first + i, 1)],
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
                    sums[...] = jnp.zeros(sums.shape, sums_dtype)
                    packed.at[1][...] = jnp.zeros(packed.at[1].shape, packed.dtype)
                    acts[...] = jnp.zeros(acts.shape, acts.dtype)

                @pl.when(active_ng > 0)
                def gate_up():
                    @pl.when(g == 0)
                    def initial():
                        start(g, bank)

                    _, first, n_active = active_bounds(g)

                    @pl.when(n_active > 0)
                    def wait_active():
                        active = acts.at[bank, pl.ds(first, n_active)]
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
                        overlap_work=lambda panel, b_idx=0: gu_work(g, panel, b_idx),
                    )

                inner(gy, dw, metadata, first=True)

            specs = dict(
                grid=(1, ng, 1),
                in_specs=(joint_gu_ws, joint_down_ws),
                out_specs=gu_ys,
            )
            refs = (gu, w, middle.reshape(-1, sub, mid_width))
            scratches = (metadata, gu_partial, acts, act_sems)
            if tokens == 8192 and mid_width != 768:
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

        ring_tn = tn // 2 if ep8_8k else tn

        @functools.partial(
            pl.run_scoped, subtotal_vmem=pltpu.VMEM((tokens, tn), x.dtype)
        )
        def remaining_columns(subtotal_vmem):
            emit_subtotals(subtotal_vmem)
            state[0] = 0

            def shifted(spec):
                index_map = spec.index_map
                updates = {"index_map": lambda n, g, k: index_map(n + 1, g, k)}
                if ep8_8k:
                    updates["pipeline_mode"] = pl.Buffered(buffer_count=2)
                return dataclasses.replace(spec, **updates)

            shifted_ws = jax.tree.map(
                shifted, ws, is_leaf=lambda v: isinstance(v, pl.BlockSpec)
            )

            @functools.partial(
                pl.run_scoped,
                payload_buf=pltpu.VMEM((localrows, ring_tn), ring_dtype),
                rx_buf=pltpu.VMEM((2, localrows, ring_tn), ring_dtype),
            )
            def run_rem(payload_buf, rx_buf):
                sync_ring_peers()

                def remaining(x, w, metadata):
                    inner(
                        x,
                        w,
                        metadata,
                        subtotal_vmem=subtotal_vmem,
                        payload_buf=payload_buf,
                        rx_buf=rx_buf,
                    )

                pltpu.emit_pipeline(
                    remaining,
                    grid=(width // tn - 1, ng, 1),
                    in_specs=(xs, shifted_ws),
                    out_specs=(),
                )(middle.reshape(-1, sub, mid_width), w, scratches=(metadata,))

            if ep8_8k:
                emit_subtotals(subtotal_vmem)
                state[0] = 0

                @functools.partial(
                    pl.run_scoped,
                    payload_full=pltpu.VMEM((localrows, tn), ring_dtype),
                    rx_full=pltpu.VMEM((2, localrows, tn), ring_dtype),
                )
                def run_last(payload_full, rx_full):
                    for stream in range(streams):
                        base = (stream % 6) * 34
                        pl.semaphore_signal(
                            credits.at[stream],
                            1,
                            device_id=(ring_meta[base + 1],),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                    for stream in range(streams):
                        pl.semaphore_wait(credits.at[stream], 1)
                    advance_slice(
                        subtotal_vmem,
                        out.at[:, pl.ds(tn, tn)],
                        payload_full,
                        rx_full,
                        regions,
                        drain=True,
                    )

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
        sums=pltpu.VMEM((tokens + _RMW_DUMMY_SLOTS, 2, packed_half), sums_dtype),
        sends=pltpu.SemaphoreType.DMA((streams,)),
        receives=pltpu.SemaphoreType.DMA((2, streams)),
        credits=pltpu.SemaphoreType.REGULAR((streams,)),
        stores=pltpu.SemaphoreType.DMA,
        state=pltpu.SMEM((1 + 2 * streams,), jnp.int32),
        ring_meta=pltpu.SMEM((6 * 34,), jnp.int32),
        moved_smem=pltpu.SMEM((1,), jnp.int32),
        phase_sems=pltpu.SemaphoreType.REGULAR((2,)),
        phase_smem=pltpu.SMEM((2,), jnp.int32),
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
    panel_local = cfgs.tiles.tile_m == 256 and (cfgs.tiles.tile_k > 768 or width > 1792)
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
            panel_weights = (
                panel_rhs.get_weight()
                .astype(cfgs.lhs_cfgs.dtype)
                .reshape(cfgs.num_quant_blocks_per_tile_k, block, panel_width)
            )
            scales = panel_rhs.get_scale(replicate_size=block).astype(
                cfgs.lhs_cfgs.dtype
            )
            panel_weights = (panel_weights * scales).reshape(
                cfgs.tiles.tile_k, panel_width
            )
        else:
            assert weights is not None
            panel_weights = weights[:, first : first + panel_width]
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
            lhs,
            rhs,
            cfgs=cfgs,
            bucket_rows=rows,
            overlap_work=lambda panel: overlap_work(panel, bucket_index),
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

    # At EP16 (tile_n=1536, 12 GU panels), share the 128-row product to bound
    # instruction size; at EP8 (tile_n=768, 6 GU panels), keep the 64-row bucket.
    first_bucket = 1 if (cfgs.tiles.tile_m == 256 and cfgs.tiles.tile_n > 768) else 0
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
        value = _expert_product(
            lhs,
            rhs,
            cfgs=cfgs,
            bucket_rows=rows,
            overlap_work=lambda panel: overlap_work(panel, bucket_index),
        )
        packed_width = 2 * output.shape[-1]
        if packed_width > cfgs.tiles.tile_n:
            value = value.astype(output.dtype)
            padded = jnp.concatenate(
                (
                    value,
                    jnp.zeros((rows, packed_width - cfgs.tiles.tile_n), output.dtype),
                ),
                axis=1,
            )
            output[:rows] = padded.reshape(rows, 2, output.shape[-1])
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

    def unpack_region(region):
        if len(region) == 6:
            begin, size, src_bank, src_col, dst_col, width = region
            return (
                begin,
                size,
                src_bank,
                pl.ds(src_col, width),
                pl.ds(dst_col, width),
                width,
            )
        begin, size, col_begin, width = region
        cols = pl.ds(col_begin, width)
        return begin, size, None, cols, cols, width

    def read_x(source_rows, src_bank, src_cols):
        if src_bank is None:
            return x_ref.at[source_rows, src_cols][...]
        return x_ref.at[source_rows, src_bank, src_cols][...]

    @pl.when(state[0] == 0)
    def initialize():
        for stream, region in enumerate(streams):
            begin, size, src_bank, src_cols, cols, _ = unpack_region(region)
            base = (stream % 6) * 34
            rows = pl.ds(begin, size)
            source_rows = pl.ds(
                pl.multiple_of(ring_meta[base + 2], o_ref.shape[0]) + begin, size
            )
            target = (ring_meta[base],)
            payload.at[rows, cols][...] = read_x(
                source_rows, src_bank, src_cols
            ).astype(payload.dtype)
            remote(stream, rows, cols, target, jnp.int32(0)).start()
            state[1 + stream] = 1
            state[1 + count + stream] = 0
        state[0] = count + 1

    def sweep(selected=None):
        for stream, region in enumerate(streams):
            if selected is not None and (stream // 6) != selected:
                continue
            begin, size, src_bank, src_cols, cols, width = unpack_region(region)
            base = (stream % 6) * 34
            rows = pl.ds(begin, size)
            units = size * width * jnp.dtype(payload.dtype).itemsize // 32
            step = state[1 + stream]

            @pl.when(state[1 + count + stream] == 0)
            def receive(
                *,
                step=step,
                stream=stream,
                units=units,
                base=base,
                rows=rows,
                cols=cols,
                src_bank=src_bank,
                src_cols=src_cols,
                begin=begin,
                size=size,
            ):
                bank = (step - 1) & 1
                ready = (pl.semaphore_read(send_sems.at[stream]) >= units) & (
                    pl.semaphore_read(recv_sems.at[bank, stream]) >= units
                )

                @pl.when(ready)
                def reduce():
                    target = (ring_meta[base],)
                    remote(stream, rows, cols, target, bank).wait()
                    source_rows = pl.ds(
                        pl.multiple_of(ring_meta[base + 2 + step], o_ref.shape[0])
                        + begin,
                        size,
                    )
                    payload.at[rows, cols][...] = (
                        read_x(source_rows, src_bank, src_cols).astype(jnp.float32)
                        + received.at[bank, rows, cols][...].astype(jnp.float32)
                    ).astype(payload.dtype)
                    pl.semaphore_signal(
                        credits.at[stream],
                        1,
                        device_id=(ring_meta[base + 1],),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    state[1 + count + stream] = jnp.where(step == 31, 2, 1)

            @pl.when(state[1 + count + stream] == 1)
            def forward(*, step=step, stream=stream, rows=rows, cols=cols, base=base):
                available = (step < buffers) | (
                    pl.semaphore_read(credits.at[stream]) >= 1
                )

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


def _epilogue(
    control,
    reduced,
    shared,
    norm_weight,
    up_weight,
    prefix,
    output,
    *,
    staged_hbm=False,
    up_scale=None,
):
    rows, width = reduced.shape
    family = control[1]
    norm_weight = norm_weight.at[family, :, :width]
    up_weight = up_weight.at[family]
    dense_fp8 = up_scale is not None
    up_scale_layer = up_scale.at[family] if up_scale is not None else None
    valid = _stripe_validity(rows, control[0], jax.lax.axis_index("k3_sp"))
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def body(s):
        copy = pltpu.make_async_copy(norm_weight, s["norm"], s["norm_sem"])
        copy.start()
        sc_copy = (
            pltpu.make_async_copy(up_scale_layer, s["up_sc"], s["sc_sem"])
            if up_scale_layer is not None
            else None
        )
        if sc_copy is not None:
            sc_copy.start()
        if staged_hbm:
            reduced_copy = pltpu.make_async_copy(
                reduced, s["reduced_v"], s["stage_sems"][0]
            )
            shared_copy = pltpu.make_async_copy(
                shared, s["shared_v"], s["stage_sems"][1]
            )
            reduced_copy.start()
            shared_copy.start()
            reduced_copy.wait()
            reduced_ref = s["reduced_v"]
            shared_ref = s["shared_v"]
        else:
            shared_copy = None
            reduced_ref = reduced
            shared_ref = shared
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
        norm_v = s["norm"][...]
        norm_chunk = min(rows, 32)
        for r0 in range(0, rows, norm_chunk):
            routed = reduced_ref.at[pl.ds(r0, norm_chunk)][...].astype(jnp.float32)
            mean = jnp.sum(routed * routed, axis=1, keepdims=True) / width
            inverse = jax.lax.rsqrt(mean + 1e-5)
            normed = (routed * inverse) * norm_v
            if dense_fp8:
                rmax = jnp.max(jnp.abs(normed), axis=1, keepdims=True)
                rscale = jnp.maximum(rmax, _FP8_SCALE_EPS) / _FP8_MAX
                s["normalized"].at[pl.ds(r0, norm_chunk)][...] = (
                    normed / rscale
                ).astype(_FP8_DTYPE)
                s["r_sc"][pl.ds(r0, norm_chunk), :] = jnp.broadcast_to(
                    rscale, (norm_chunk, 128)
                )
            else:
                s["normalized"].at[pl.ds(r0, norm_chunk)][...] = normed.astype(
                    jnp.bfloat16
                )
        if sc_copy is not None:
            sc_copy.wait()
        if shared_copy is not None:
            shared_copy.wait()
        row_tile = min(rows, 256)
        for panel in range(14):
            bank = panel % 3
            weight_copy(panel).wait()
            prefix_copy(panel).wait()
            if panel + 2 < 14:
                if panel >= 1:
                    output_copy(panel - 1).wait()
                weight_copy(panel + 2).start()
                prefix_copy(panel + 2).start()
            if dense_fp8:
                w_sc = s["up_sc"][:, pl.ds(panel * 512, 512)]
                for r0 in range(0, rows, row_tile):
                    raw = jnp.matmul(
                        s["normalized"][pl.ds(r0, row_tile), :],
                        s["weights"][bank][...],
                        preferred_element_type=jnp.float32,
                    )
                    r_sc = s["r_sc"][pl.ds(r0, row_tile), :1]
                    product = (raw * r_sc * w_sc).astype(jnp.bfloat16)
                    value = (
                        product
                        + shared_ref[pl.ds(r0, row_tile), pl.ds(panel * 512, 512)]
                    )
                    value = jnp.where(valid[r0 : r0 + row_tile, :], value, 0)
                    s["tiles"][bank][pl.ds(r0, row_tile), :] = (
                        s["tiles"][bank][pl.ds(r0, row_tile), :] + value
                    )
            else:
                product = jnp.matmul(
                    s["normalized"][...],
                    s["weights"][bank][...],
                    preferred_element_type=jnp.float32,
                ).astype(jnp.bfloat16)
                value = product + shared_ref[:, pl.ds(panel * 512, 512)]
                value = jnp.where(valid, value, 0)
                s["tiles"][bank][...] = s["tiles"][bank][...] + value
            output_copy(panel).start()
        for panel in (11, 12, 13):
            output_copy(panel).wait()

    weight_dtype = _FP8_DTYPE if dense_fp8 else jnp.bfloat16
    scratch = dict(
        norm=vmem((1, width), jnp.float32),
        norm_sem=dma,
        up_sc=vmem((1, 7168), jnp.float32) if dense_fp8 else None,
        sc_sem=dma if dense_fp8 else None,
        r_sc=vmem((rows, 128), jnp.float32) if dense_fp8 else None,
        normalized=vmem((rows, width), weight_dtype),
        weights=tuple(vmem((width, 512), weight_dtype) for _ in range(3)),
        tiles=tuple(vmem((rows, 512), jnp.bfloat16) for _ in range(3)),
        weight_sems=(dma, dma, dma),
        prefix_sems=(dma, dma, dma),
        output_sems=(dma, dma, dma),
    )
    if staged_hbm:
        scratch["reduced_v"] = vmem((rows, width), jnp.bfloat16)
        scratch["shared_v"] = vmem((rows, 7168), jnp.bfloat16)
        scratch["stage_sems"] = (dma, dma)
    pl.run_scoped(body, scratch)


def stage_b_last_ring_overlap(
    next_active_scalar,
    blocks_hbm,
    next_folded_hbm,
    b_num_hbm,
    b_stats_hbm,
    progress: Callable[[], None],
) -> None:
    rows, hidden = blocks_hbm.shape[1], blocks_hbm.shape[2]
    tile = min(128, rows)
    v_chunk = min(32, tile)
    n_tiles = rows // tile
    n_slots = min(2, n_tiles)
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA

    def run(s):
        c_fw = pltpu.make_async_copy(next_folded_hbm, s["folded"], s["fw_sem"])
        c_fw.start()

        def load_blk(t_idx, blk_idx, slot):
            return pltpu.make_async_copy(
                blocks_hbm.at[blk_idx, pl.ds(t_idx * tile, tile), :],
                s["b_slots"].at[slot],
                s["b_sems"].at[slot],
            )

        def store_num(t_idx, n_slot):
            return pltpu.make_async_copy(
                s["num_slots"].at[n_slot],
                b_num_hbm.at[pl.ds(t_idx * tile, tile), :],
                s["num_sems"].at[n_slot],
            )

        @pl.when(next_active_scalar > 0)
        def pref_initial():
            load_blk(0, 0, 0).start()

        c_fw.wait()
        progress()

        @pl.loop(0, n_tiles)
        def tile_loop(t):
            n_slot = t % n_slots
            if n_tiles >= 2:

                @pl.when(t >= 2)
                def wait_prev_num():
                    store_num(t - 2, n_slot).wait()

            @pl.when(next_active_scalar == 0)
            def zero_tile():
                z_chunk = jnp.zeros((v_chunk, hidden), jnp.float32)
                for off in range(0, tile, v_chunk):
                    s["num_slots"].at[n_slot, pl.ds(off, v_chunk), :][...] = z_chunk
                t_sl = pl.ds(t * tile, tile)
                s["stats"].at[t_sl, 0:1][...] = jnp.full((tile, 1), -1e30, jnp.float32)
                s["stats"].at[t_sl, 1:2][...] = jnp.zeros((tile, 1), jnp.float32)
                progress()

            @pl.loop(0, next_active_scalar)
            def blk_loop(blk):
                b_slot = blk % 2

                @pl.when(blk + 1 < next_active_scalar)
                def pref_next_blk():
                    load_blk(t, blk + 1, 1 - b_slot).start()

                load_blk(t, blk, b_slot).wait()
                progress()
                fold_v = s["folded"][...]
                for off in range(0, tile, v_chunk):
                    l_sl = pl.ds(off, v_chunk)
                    g_sl = pl.ds(t * tile + off, v_chunk)
                    vals = s["b_slots"].at[b_slot, l_sl, :][...].astype(jnp.float32)
                    sc = jnp.sum(vals * fold_v, axis=1, keepdims=True) * lax.rsqrt(
                        jnp.mean(vals * vals, axis=1, keepdims=True) + 1e-5
                    )

                    @pl.when(blk == 0)
                    def init_blk(*, l_sl=l_sl, g_sl=g_sl, vals=vals, sc=sc):
                        s["num_slots"].at[n_slot, l_sl, :][...] = vals
                        s["stats"].at[g_sl, 0:1][...] = sc
                        s["stats"].at[g_sl, 1:2][...] = jnp.ones(
                            (v_chunk, 1), jnp.float32
                        )

                    @pl.when(blk > 0)
                    def accum_blk(*, l_sl=l_sl, g_sl=g_sl, vals=vals, sc=sc):
                        om = s["stats"].at[g_sl, 0:1][...]
                        nm = jnp.maximum(om, sc)
                        os_ = jnp.exp(om - nm)
                        ns_ = jnp.exp(sc - nm)
                        s["num_slots"].at[n_slot, l_sl, :][...] = (
                            s["num_slots"].at[n_slot, l_sl, :][...] * os_ + vals * ns_
                        )
                        s["stats"].at[g_sl, 1:2][...] = (
                            s["stats"].at[g_sl, 1:2][...] * os_ + ns_
                        )
                        s["stats"].at[g_sl, 0:1][...] = nm

                    if tile >= 64 and off == 32:
                        progress()

            if n_tiles >= 2:

                @pl.when((t + 1 < n_tiles) & (next_active_scalar > 0))
                def pref_next_tile_b0():
                    load_blk(t + 1, 0, 0).start()

            store_num(t, n_slot).start()
            progress()

        c_st = pltpu.make_async_copy(s["stats"], b_stats_hbm, s["st_sem"])
        c_st.start()
        if n_tiles >= 2:
            store_num(n_tiles - 2, (n_tiles - 2) % n_slots).wait()
        store_num(n_tiles - 1, (n_tiles - 1) % n_slots).wait()
        c_st.wait()

    pl.run_scoped(
        run,
        dict(
            folded=vmem((1, hidden), jnp.float32),
            fw_sem=dma,
            b_slots=vmem((2, tile, hidden), jnp.bfloat16),
            b_sems=dma((2,)),
            num_slots=vmem((n_slots, tile, hidden), jnp.float32),
            num_sems=dma((n_slots,)),
            stats=vmem((rows, 128), jnp.float32),
            st_sem=dma,
        ),
    )


def epilogue_merge_next_norm(
    control,
    reduced,
    shared,
    norm_weight,
    up_weight,
    prefix,
    output,
    *,
    staged_hbm: bool,
    b_num_hbm,
    b_stats_hbm,
    next_folded_hbm,
    next_norm_hbm,
    next_normed_out,
) -> None:
    rows, width = reduced.shape
    hidden = prefix.shape[1]
    family = control[1]
    norm_w = norm_weight.at[family, :, :width]
    up_w = up_weight.at[family]
    valid = _stripe_validity(rows, control[0], lax.axis_index("k3_sp"))
    vmem, dma = pltpu.VMEM, pltpu.SemaphoreType.DMA
    tile = min(128, rows)
    v_chunk = min(32, tile)
    norm_chunk = min(32, rows)
    n_tiles = rows // tile
    n_slots = min(2, n_tiles)

    def body(s):
        c_nw = pltpu.make_async_copy(norm_w, s["norm"], s["norm_sem"])
        c_px = pltpu.make_async_copy(prefix, s["next_px"], s["px_sem"])
        c_st = pltpu.make_async_copy(b_stats_hbm, s["stats"], s["next_sems"][0])
        c_nf = pltpu.make_async_copy(next_folded_hbm, s["folded"], s["next_sems"][1])
        c_nn = pltpu.make_async_copy(next_norm_hbm, s["next_norm"], s["next_sems"][2])
        c_nw.start()
        c_px.start()
        c_st.start()
        c_nf.start()
        c_nn.start()
        if staged_hbm:
            c_red = pltpu.make_async_copy(reduced, s["reduced_v"], s["stage_sems"][0])
            c_sh = pltpu.make_async_copy(shared, s["shared_v"], s["stage_sems"][1])
            c_red.start()
            c_sh.start()
            c_red.wait()
            reduced_ref = s["reduced_v"]
            shared_ref = s["shared_v"]
        else:
            c_sh = None
            reduced_ref = reduced
            shared_ref = shared
        c_nw.wait()

        def weight_copy(panel: int):
            return pltpu.make_async_copy(
                up_w.at[:, pl.ds(panel * 512, 512)],
                s["weights"][panel % 3],
                s["weight_sems"][panel % 3],
            )

        def output_copy(panel: int):
            return pltpu.make_async_copy(
                s["next_px"].at[:, pl.ds(panel * 512, 512)],
                output.at[:, pl.ds(panel * 512, 512)],
                s["output_sems"][panel % 3],
            )

        def load_b_num(t_idx, slot):
            return pltpu.make_async_copy(
                b_num_hbm.at[pl.ds(t_idx * tile, tile), :],
                s["num_slots"].at[slot],
                s["num_sems"].at[slot],
            )

        def store_next_out(t_idx, slot):
            return pltpu.make_async_copy(
                s["out_slots"].at[slot],
                next_normed_out.at[pl.ds(t_idx * tile, tile), :],
                s["next_out_sems"].at[slot],
            )

        for panel in (0, 1):
            weight_copy(panel).start()

        norm_v = s["norm"][...]
        for r0 in range(0, rows, norm_chunk):
            routed = reduced_ref.at[pl.ds(r0, norm_chunk)][...].astype(jnp.float32)
            inv = lax.rsqrt(
                jnp.sum(routed * routed, axis=1, keepdims=True) / width + 1e-5
            )
            s["normalized"].at[pl.ds(r0, norm_chunk)][...] = (
                (routed * inv) * norm_v
            ).astype(jnp.bfloat16)

        if c_sh is not None:
            c_sh.wait()
        c_px.wait()
        load_b_num(0, 0).start()
        if n_tiles >= 2:
            load_b_num(1, 1).start()

        for panel in range(14):
            bank = panel % 3
            cols = pl.ds(panel * 512, 512)
            weight_copy(panel).wait()
            if panel + 2 < 14:
                if panel >= 1:
                    output_copy(panel - 1).wait()
                weight_copy(panel + 2).start()
            prod = jnp.matmul(
                s["normalized"][...],
                s["weights"][bank][...],
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
            val = jnp.where(valid, prod + shared_ref[:, cols], 0)
            s["next_px"][:, cols] = s["next_px"][:, cols] + val
            output_copy(panel).start()

        c_st.wait()
        c_nf.wait()
        c_nn.wait()
        fold_v = s["folded"][...]
        nn_v = s["next_norm"][...].astype(jnp.float32)

        @pl.loop(0, n_tiles)
        def merge_tile(t):
            slot = t % n_slots
            load_b_num(t, slot).wait()
            if n_tiles >= 2:

                @pl.when(t >= 2)
                def wait_prev_out():
                    store_next_out(t - 2, slot).wait()

            for off in range(0, tile, v_chunk):
                l_sl = pl.ds(off, v_chunk)
                g_sl = pl.ds(t * tile + off, v_chunk)
                px_f = s["next_px"].at[g_sl, :][...].astype(jnp.float32)
                p_sc = jnp.sum(px_f * fold_v, axis=1, keepdims=True) * lax.rsqrt(
                    jnp.mean(px_f * px_f, axis=1, keepdims=True) + 1e-5
                )
                om = s["stats"].at[g_sl, 0:1][...]
                bs = s["stats"].at[g_sl, 1:2][...]
                nm = jnp.maximum(om, p_sc)
                os_ = jnp.exp(om - nm)
                ps_ = jnp.exp(p_sc - nm)
                num = s["num_slots"].at[slot, l_sl, :][...] * os_ + px_f * ps_
                den = bs * os_ + ps_
                mixed = (num * (1.0 / den)).astype(jnp.bfloat16).astype(jnp.float32)
                rvar = lax.rsqrt(jnp.mean(mixed * mixed, axis=1, keepdims=True) + 1e-5)
                s["out_slots"].at[slot, l_sl, :][...] = (mixed * rvar * nn_v).astype(
                    jnp.bfloat16
                )
            if n_tiles >= 2:

                @pl.when(t + 2 < n_tiles)
                def pref_next_num():
                    load_b_num(t + 2, slot).start()

            store_next_out(t, slot).start()

        for panel in (11, 12, 13):
            output_copy(panel).wait()
        if n_tiles >= 2:
            store_next_out(n_tiles - 2, (n_tiles - 2) % n_slots).wait()
        store_next_out(n_tiles - 1, (n_tiles - 1) % n_slots).wait()

    merge_scratch: dict[str, Any] = dict(
        norm=vmem((1, width), jnp.float32),
        norm_sem=dma,
        normalized=vmem((rows, width), jnp.bfloat16),
        weights=tuple(vmem((width, 512), jnp.bfloat16) for _ in range(3)),
        weight_sems=(dma, dma, dma),
        output_sems=(dma, dma, dma),
        next_px=vmem((rows, hidden), jnp.bfloat16),
        px_sem=dma,
        stats=vmem((rows, 128), jnp.float32),
        folded=vmem((1, hidden), jnp.float32),
        next_norm=vmem((1, hidden), jnp.bfloat16),
        next_sems=(dma, dma, dma),
        num_slots=vmem((n_slots, tile, hidden), jnp.float32),
        num_sems=dma((n_slots,)),
        out_slots=vmem((n_slots, tile, hidden), jnp.bfloat16),
        next_out_sems=dma((n_slots,)),
    )
    if staged_hbm:
        merge_scratch["reduced_v"] = vmem((rows, width), jnp.bfloat16)
        merge_scratch["shared_v"] = vmem((rows, hidden), jnp.bfloat16)
        merge_scratch["stage_sems"] = (dma, dma)
    pl.run_scoped(body, merge_scratch)


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
    mixture_refs=None,
    shared_hbm=None,
    reduced_hbm=None,
    subtotal_tail_hbm=None,
    dense_scales=None,
    replica_table=None,
    replica_slots: int = 0,
    latent_scales_hbm=None,
    next_mixture_refs=None,
):
    """One launch owns all compute, metadata and communication on the fast path.

    Large projection scratch has a shorter lifetime than expert scratch. Only
    shared output and the scalar routing metadata survive that boundary. At 8K
    all ranks take the same overflow branch; no routes are dropped to fit SMEM.
    """
    rows, hidden_size = prefix.shape if hidden is None else hidden.shape
    tokens = rows * 32
    local_experts = expert_gu.shape[1]
    capacity = _compact_capacity(tokens)
    shape = (tokens // 8, 128)
    stage_hbm = shared_hbm is not None and reduced_hbm is not None
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
            if maximum is not None:
                maximum_copy().start()

        def probability_copy(probs_v):
            return pltpu.make_async_copy(
                probs_v.at[: capacity // 128, :],
                s["probs"],
                s["prob_sem"],
            )

        stream_shared_hbm = stage_hbm and tokens >= 16384
        defer_shared_down = (
            stream_shared_hbm
            and tokens == routed_16k.TOKENS_16K
            and (dense_scales is None)
        )
        assert not (defer_shared_down and fallback_shared is not None)

        def prepare(prep):
            ids_v = prep["ids_v"]
            probs_v = prep["probs_v"] if tokens >= 8192 else s["probs_v"]
            pair_v = prep.get("pair_v")
            shared_dest = (
                shared_hbm
                if stream_shared_hbm
                else (prep["shared_v"] if stage_hbm else s["shared"])
            )
            indices = pltpu.make_async_copy(
                ids_v.at[: capacity // 128, :], s["ids"], s["index_sem"]
            )

            def metadata_ready():
                indices.start()
                probability_copy(probs_v).start()

            if mixture_refs is not None:
                assert pair_v is not None
                blocks_ref, mlp_res_ref, ffn_norm_ref = mixture_refs
                mix_tile = min(128, rows)
                parity = jax.lax.axis_index("k3_sp") % 2
                own = pair_v.at[pl.ds(parity * rows, rows), :]
                mix_specs = k.mixture_norm_tiled_scratch_shapes(
                    hidden_size, mix_tile, store_output=False
                )

                def run_mix(*mix_s):
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
                        prefix,
                        blocks_ref,
                        mlp_res_ref,
                        ffn_norm_ref,
                        None,
                        prefix_bufs,
                        block_buf,
                        None,
                        numerator_buf,
                        row_max_buf,
                        row_sum_buf,
                        folded_buf,
                        norm_buf,
                        prefix_sems,
                        block_sem,
                        None,
                        weight_sems,
                        tile=mix_tile,
                        eps=1e-5,
                        output_vmem_ref=own,
                    )

                pl.run_scoped(run_mix, *mix_specs)

            _prologue(
                tables,
                control,
                hidden,
                router,
                bias,
                latent_down,
                shared_gu,
                shared_down,
                shared_dest,
                latent,
                ids_v,
                probs_v,
                s["counts_v"],
                maximum,
                counts_ready=counts_ready,
                metadata_ready=metadata_ready,
                pair_buf=pair_v,
                shared_in_hbm=stream_shared_hbm,
                dense_scales=dense_scales[:3] if dense_scales is not None else None,
                replica_table=replica_table,
                replica_slots=replica_slots,
                latent_scales_hbm=latent_scales_hbm,
                defer_shared_down=defer_shared_down,
            )
            stage_copy = None
            if stage_hbm and not stream_shared_hbm:
                stage_copy = pltpu.make_async_copy(
                    shared_dest, shared_hbm, s["stage_sem"]
                )
                stage_copy.start()
            count_copy().wait()
            if route_counts is not None:
                for e in range(route_counts.shape[0]):
                    route_counts[e] = s["counts"][e]
            if maximum is not None:
                maximum_copy().wait()
            if tokens < 8192:
                metadata_ready()
            if maximum is not None:

                @pl.when(s["maximum"][0] > capacity)
                def export_overflow():
                    export_pairs = [
                        (ids_v, fallback_ids),
                        (probs_v, fallback_probs),
                        (s["counts_v"], fallback_counts),
                    ]
                    if fallback_shared is not None:
                        export_pairs.append((shared_dest, fallback_shared))
                    for source, dest in export_pairs:
                        copy = pltpu.make_async_copy(source, dest, s["export_sem"])
                        copy.start()
                        copy.wait()

            indices.wait()
            if tokens >= 8192:
                probability_copy(probs_v).wait()
            if stage_copy is not None:
                stage_copy.wait()

        prep_spec = dict(ids_v=vmem(shape, jnp.int32))
        if tokens >= 8192:
            # The compact path completes both metadata DMAs before expert work.
            prep_spec["probs_v"] = vmem(shape, jnp.float32)
        if mixture_refs is not None:
            prep_spec["pair_v"] = vmem((2 * rows, hidden_size), jnp.bfloat16)
        if stage_hbm and not stream_shared_hbm:
            prep_spec["shared_v"] = vmem((rows, 7168), jnp.bfloat16)
        pl.run_scoped(prepare, prep_spec)

        def probabilities_ready():
            if tokens < 8192:
                probability_copy(s["probs_v"]).wait()

        def fast():
            routed_out = reduced_hbm if stage_hbm else s["reduced"]
            last_ring_overlap_fn = None
            if next_mixture_refs is not None:
                assert mixture_refs is not None and dense_scales is None
                blocks_ref, _, _ = mixture_refs
                (
                    next_folded_ref,
                    next_norm_ref,
                    next_normed_out,
                    b_num_hbm,
                    b_stats_hbm,
                ) = next_mixture_refs
                if tokens == routed_16k.TOKENS_16K:

                    def overlap_stage_b(progress: Callable[[], None]) -> None:
                        stage_b_last_ring_overlap(
                            control[3],
                            blocks_ref,
                            next_folded_ref,
                            b_num_hbm,
                            b_stats_hbm,
                            progress,
                        )

                    last_ring_overlap_fn = overlap_stage_b

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
                routed_out,
                tokens=tokens,
                probabilities_ready=probabilities_ready,
                subtotal_tail_hbm=subtotal_tail_hbm,
                latent_scales_hbm=latent_scales_hbm,
                deferred_shared_hbm=shared_hbm if defer_shared_down else None,
                deferred_shared_down_hbm=(
                    shared_down.at[control[1]] if defer_shared_down else None
                ),
                last_ring_overlap_fn=last_ring_overlap_fn,
            )
            if next_mixture_refs is not None:
                assert mixture_refs is not None
                blocks_ref_ep, _, _ = mixture_refs
                (
                    next_folded_ep,
                    next_norm_ep,
                    next_normed_ep,
                    b_num_ep,
                    b_stats_ep,
                ) = next_mixture_refs
                if tokens != routed_16k.TOKENS_16K:
                    stage_b_last_ring_overlap(
                        control[3],
                        blocks_ref_ep,
                        next_folded_ep,
                        b_num_ep,
                        b_stats_ep,
                        lambda: None,
                    )
                epilogue_merge_next_norm(
                    control,
                    routed_out,
                    shared_hbm if stage_hbm else s["shared"],
                    norm,
                    latent_up,
                    prefix,
                    output,
                    staged_hbm=stage_hbm,
                    b_num_hbm=b_num_ep,
                    b_stats_hbm=b_stats_ep,
                    next_folded_hbm=next_folded_ep,
                    next_norm_hbm=next_norm_ep,
                    next_normed_out=next_normed_ep,
                )
            else:
                _epilogue(
                    control,
                    routed_out,
                    shared_hbm if stage_hbm else s["shared"],
                    norm,
                    latent_up,
                    prefix,
                    output,
                    staged_hbm=stage_hbm,
                    up_scale=dense_scales[3] if dense_scales is not None else None,
                )

        if maximum is None:
            fast()
        else:

            @pl.when(s["maximum"][0] <= capacity)
            def compact():
                fast()

            if defer_shared_down:

                @pl.when(s["maximum"][0] > capacity)
                def materialize_overflow_shared():
                    rank = jax.lax.axis_index("k3_sp")
                    shared_down_16k.run_deferred_shared_down_from_hbm(
                        shared_hbm,
                        shared_down.at[control[1]],
                        s["overflow_down_ready"],
                        rows=rows,
                        parity=rank % 2,
                        peer=rank ^ 1,
                    )

    defer_overflow_shared = (
        stage_hbm and tokens == routed_16k.TOKENS_16K and (dense_scales is None)
    )
    pl.run_scoped(
        body,
        dict(
            shared=None if stage_hbm else vmem((rows, 7168), jnp.bfloat16),
            reduced=None if stage_hbm else vmem((rows, 3584), jnp.bfloat16),
            stage_sem=dma if (stage_hbm and tokens < 16384) else None,
            probs_v=vmem(shape, jnp.float32) if tokens < 8192 else None,
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
            overflow_down_ready=pltpu.SemaphoreType.REGULAR
            if (defer_overflow_shared and maximum is not None)
            else None,
        ),
    )


_COMPOUND_PARTS: dict[int, tuple[int, ...]] = {
    6144: (4096, 2048),
    10240: (8192, 2048),
    12288: (8192, 4096),
    14336: (8192, 4096, 2048),
    16384: (8192, 8192),
}


def make_indexed_moe_sp_ring(mesh, *, seq_len=8192):
    return _build_indexed_moe_sp_ring(mesh, seq_len=seq_len, with_mixture=False)


def supports_fused_next_mixture(*, seq_len: int) -> bool:
    cfg = k3_config.active()
    return (
        seq_len == routed_16k.TOKENS_16K
        and cfg.prefill_fused_moe
        and cfg.prefill_moe_16k_monolithic
        and cfg.prefill_expert_parallel in (8, 16)
        and not cfg.prefill_moe_dense_fp8
    )


def make_indexed_mixture_moe_sp_ring(
    mesh, *, seq_len=8192, with_next_mixture: bool = False
):
    return _build_indexed_moe_sp_ring(
        mesh,
        seq_len=seq_len,
        with_mixture=True,
        with_next_mixture=with_next_mixture,
    )


def _build_indexed_moe_sp_ring(
    mesh,
    *,
    seq_len=8192,
    with_mixture=False,
    with_next_mixture: bool = False,
):
    if with_next_mixture and not with_mixture:
        raise ValueError("with_next_mixture requires with_mixture=True")
    cfg = k3_config.active()
    if with_next_mixture and cfg.prefill_moe_dense_fp8:
        raise ValueError("with_next_mixture does not support prefill_moe_dense_fp8")
    if seq_len in _COMPOUND_PARTS and not (
        seq_len == 16384 and cfg.prefill_moe_16k_monolithic
    ):
        if with_next_mixture:
            raise ValueError(
                "with_next_mixture requires a monolithic prefill MoE window, got "
                f"{seq_len}"
            )
        return _make_moe_compound(
            mesh, seq_len, _COMPOUND_PARTS[seq_len], with_mixture=with_mixture
        )
    if seq_len not in (512, 1024, 2048, 4096, 8192, 16384):
        raise ValueError(
            "Fused prefill MoE supports 512, 1024, and multiples of 2048 up to 16384"
        )
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("Fused prefill MoE requires a flat 32-rank k3_sp mesh")
    ep = cfg.prefill_expert_parallel
    if ep not in (8, 16):
        raise ValueError(
            "Fused prefill MoE requires prefill_expert_parallel in (8, 16)"
        )
    from .layer import routed_experts_prepared

    base_local_experts = 896 // ep
    replica_slots = (
        rank_loader.prefill_replica_slots()
        if (cfg.prefill_replica_plan and ep == 16 and seq_len in (8192, 16384))
        else 0
    )
    use_replicas = replica_slots > 0
    local_experts = base_local_experts + replica_slots
    num_scalar_prefetch = 2 + int(use_replicas)
    mid_width = 3072 // (32 // ep)
    rows = seq_len // 32
    capacity = _compact_capacity(seq_len)
    middle_rows = capacity if seq_len == 16384 else seq_len * 16
    split_subtotal_16k = seq_len == 16384
    w4a8_16k_enabled = seq_len == 16384 and ep == 16 and cfg.prefill_moe_w4a8
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(32)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    shape = (seq_len // 8, 128)
    bf16 = jnp.bfloat16
    latent_spec = (
        ((seq_len, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE), w4a8_16k.LATENT_E4M3_DTYPE)
        if w4a8_16k_enabled
        else ((seq_len, 2, 1792), bf16)
    )
    workspace_shapes = [
        ((rows, 7168), bf16),
        latent_spec,
        ((middle_rows, mid_width), bf16),
    ]
    output_specs = [hbm] * 3
    can_overflow = (seq_len == 8192 and ep >= 8) or seq_len == 16384
    stage_outputs_hbm = (seq_len == 8192 and ep == 8) or seq_len == 16384
    has_fallback_shared = can_overflow and not stage_outputs_hbm
    if can_overflow:
        workspace_shapes += [
            (shape, jnp.int32),
            (shape, jnp.float32),
            ((local_experts,), jnp.int32),
        ]
        output_specs += [hbm] * 3
        if has_fallback_shared:
            workspace_shapes += [((rows, 7168), bf16)]
            output_specs += [hbm]
        workspace_shapes += [((1,), jnp.int32)]
        output_specs += [pl.BlockSpec()]
    if stage_outputs_hbm:
        workspace_shapes += [((rows, 7168), bf16), ((rows, 3584), bf16)]
        output_specs += [hbm, hbm]
    if split_subtotal_16k:
        workspace_shapes += [((seq_len, routed_16k.SUBTOTAL_16K_COLS), bf16)]
        output_specs += [hbm]
    if w4a8_16k_enabled:
        workspace_shapes += [(w4a8_16k.LATENT_SCALES_SHAPE, jnp.float32)]
        output_specs += [hbm]
    if with_next_mixture:
        workspace_shapes += [
            ((rows, 7168), bf16),
            ((rows, 7168), jnp.float32),
            ((rows, 128), jnp.float32),
        ]
        output_specs += [hbm] * 3
    counting = cfg.prefill_route_counts
    dense_fp8 = cfg.prefill_moe_dense_fp8
    if counting:
        workspace_shapes += [((base_local_experts,), jnp.int32)]
        output_specs += [smem]

    def body(*refs):
        refs = list(refs)
        replica_table_ref = refs.pop(2) if use_replicas else None
        for idx in (8, 9, 10, 11):
            if refs[idx].shape[1] < local_experts:
                raise ValueError(
                    f"Expert tensor slot count {refs[idx].shape[1]} < required "
                    f"{local_experts}"
                )
            if refs[idx].shape[1] != local_experts:
                refs[idx] = refs[idx].at[:, pl.ds(0, local_experts)]
        next_folded_ref = None
        next_norm_ref = None
        if with_next_mixture:
            next_norm_ref = refs.pop(18)
            next_folded_ref = refs.pop(17)
        mix_refs = None
        if with_mixture:
            blocks_ref = refs[2]
            refs[2] = None
            ffn_norm_ref = refs.pop(16)
            mlp_res_ref = refs.pop(15)
            mix_refs = (blocks_ref, mlp_res_ref, ffn_norm_ref)
        dense_scales = None
        if dense_fp8:
            dense_scales = tuple(refs[15:19])
            del refs[15:19]
        route_counts = refs.pop() if counting else None
        next_mix_refs = None
        if with_next_mixture:
            b_stats_hbm = refs.pop()
            b_num_hbm = refs.pop()
            next_normed_out = refs.pop()
            next_mix_refs = (
                next_folded_ref,
                next_norm_ref,
                next_normed_out,
                b_num_hbm,
                b_stats_hbm,
            )
        latent_scales_hbm = refs.pop() if w4a8_16k_enabled else None
        subtotal_tail_hbm = refs.pop() if split_subtotal_16k else None
        reduced_hbm = refs.pop() if stage_outputs_hbm else None
        shared_hbm = refs.pop() if stage_outputs_hbm else None
        maximum_ref = refs.pop() if can_overflow else None
        fallback_shared_ref = refs.pop() if has_fallback_shared else None
        fallback_counts_ref = refs.pop() if can_overflow else None
        fallback_probs_ref = refs.pop() if can_overflow else None
        fallback_ids_ref = refs.pop() if can_overflow else None
        return _layer_kernel(
            *refs,
            fallback_ids=fallback_ids_ref,
            fallback_probs=fallback_probs_ref,
            fallback_counts=fallback_counts_ref,
            fallback_shared=fallback_shared_ref,
            maximum=maximum_ref,
            route_counts=route_counts,
            mixture_refs=mix_refs,
            shared_hbm=shared_hbm,
            reduced_hbm=reduced_hbm,
            subtotal_tail_hbm=subtotal_tail_hbm,
            dense_scales=dense_scales,
            replica_table=replica_table_ref,
            replica_slots=replica_slots,
            latent_scales_hbm=latent_scales_hbm,
            next_mixture_refs=next_mix_refs,
        )

    chunk_routed_kernel = None
    if seq_len == 16384:
        chunk_out_shapes = [
            ((rows, 3584), bf16),
            ((capacity, mid_width), bf16),
        ]
        chunk_out_specs = [hbm, hbm]
        if split_subtotal_16k:
            chunk_out_shapes.append(((seq_len, routed_16k.SUBTOTAL_16K_COLS), bf16))
            chunk_out_specs.append(hbm)

        def chunk_body(
            tables_ref,
            family_ref,
            offset_ref,
            counts_ref,
            ids_ref,
            probs_ref,
            latent_ref,
            gu_ref,
            gus_ref,
            down_ref,
            ds_ref,
            *tail_refs,
        ):
            if gu_ref.shape[1] != local_experts:
                gu_ref = gu_ref.at[:, pl.ds(0, local_experts)]
                gus_ref = gus_ref.at[:, pl.ds(0, local_experts)]
                down_ref = down_ref.at[:, pl.ds(0, local_experts)]
                ds_ref = ds_ref.at[:, pl.ds(0, local_experts)]
            if w4a8_16k_enabled:
                latent_scales_ref, out_ref, middle_ref, *extra_refs = tail_refs
            else:
                latent_scales_ref = None
                out_ref, middle_ref, *extra_refs = tail_refs
            tail_ref = extra_refs[0] if split_subtotal_16k else None
            _enter_mesh()
            _routed_experts(
                latent_ref,
                gu_ref,
                gus_ref,
                down_ref,
                ds_ref,
                counts_ref,
                _FlatMetadata(ids_ref),
                _FlatMetadata(probs_ref),
                tables_ref,
                family_ref,
                offset_ref,
                middle_ref,
                out_ref,
                tokens=seq_len,
                probabilities_ready=lambda: None,
                subtotal_tail_hbm=tail_ref,
                latent_scales_hbm=latent_scales_ref,
            )

        chunk_routed_kernel = pl.pallas_call(
            chunk_body,
            out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in chunk_out_shapes),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=6,
                in_specs=(hbm,) * (6 if w4a8_16k_enabled else 5),
                out_specs=tuple(chunk_out_specs),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                vmem_limit_bytes=64 * 1024**2,
                disable_bounds_checks=True,
                shape_invariant_numerics=False,
            ),
            name="fused_moe_routed_chunk_16384",
        )

    num_hbm_inputs = (
        (15 if with_mixture else 13)
        + (2 if with_next_mixture else 0)
        + (4 if dense_fp8 else 0)
    )
    kernel_name = (
        f"fused_mixture_next_moe_pallas_{seq_len}"
        if with_next_mixture
        else f"fused_mixture_moe_pallas_{seq_len}"
        if with_mixture
        else f"fused_moe_pallas_{seq_len}"
    )
    kernel = pl.pallas_call(
        body,
        out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
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

    overflow_next_norm_op = (
        k.make_mixture_norm_tiled(seq_len=rows, tile=min(128, rows), eps=1e-5)
        if (with_next_mixture and can_overflow)
        else None
    )

    if with_next_mixture:
        in_specs = (
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
            P(),
            P("k3_sp"),
            P(),
            P(),
            P("k3_sp"),
            P("k3_sp"),
            P(),
        )
    elif with_mixture:
        in_specs = (
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
            P(),
            P("k3_sp"),
            P(),
            P(),
        )
    else:
        in_specs = (P("k3_sp"), P("k3_sp"), P("k3_sp"), P(), P())

    moe_out_spec = (P("k3_sp"), P("k3_sp")) if counting else P("k3_sp")
    out_specs = (moe_out_spec, P("k3_sp")) if with_next_mixture else moe_out_spec

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def local(*args):
        if with_next_mixture:
            (
                prefix,
                blocks,
                mlp_res,
                ffn_norm,
                active,
                families,
                length,
                slot,
                next_folded,
                next_norm,
                next_active,
            ) = args
            control = jnp.concatenate(
                (
                    length.reshape(1),
                    slot.reshape(1),
                    active.reshape(1),
                    next_active.reshape(1),
                )
            )
            first_arg = blocks[0]
            tail_args = (mlp_res[0], ffn_norm[0], next_folded[0], next_norm[0])
        elif with_mixture:
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
        dense_scale_args = (
            tuple(w[name] for name in MOE_DENSE_FP8_SCALES) if dense_fp8 else ()
        )
        replica_args = (w["prefill_replica_table"][slot[0]],) if use_replicas else ()
        outputs = list(
            kernel(
                jnp.asarray(tables),
                control,
                *replica_args,
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
                *dense_scale_args,
            )
        )
        routes = outputs.pop()[None] if counting else None
        next_normed = None
        if with_next_mixture:
            outputs.pop()
            outputs.pop()
            next_normed = outputs.pop()
        latent_scales = outputs.pop() if w4a8_16k_enabled else None
        if split_subtotal_16k:
            outputs.pop()
        staged_shared = None
        if stage_outputs_hbm:
            outputs.pop()
            staged_shared = outputs.pop()
        if not can_overflow:
            moe_result = (outputs[0][None], routes) if counting else outputs[0][None]
            if with_next_mixture:
                assert next_normed is not None
                return moe_result, next_normed[None]
            return moe_result
        maximum = outputs.pop()
        shared = outputs.pop() if has_fallback_shared else staged_shared
        assert shared is not None
        final, latent, _, ids, probs, counts = outputs

        def overflow():
            if seq_len == 16384:
                assert chunk_routed_kernel is not None
                chunk_rows = capacity // 128
                max_chunks = (seq_len * 16 + capacity - 1) // capacity
                pad_rows = max_chunks * chunk_rows - ids.shape[0]
                ids_pad = jnp.pad(ids, ((0, pad_rows), (0, 0))) if pad_rows > 0 else ids
                probs_pad = (
                    jnp.pad(probs, ((0, pad_rows), (0, 0))) if pad_rows > 0 else probs
                )
                ends = jnp.cumsum(counts, dtype=jnp.int32)
                starts = ends - counts
                family_slot = k.expert_family_slot(
                    w["expert_gu"].shape[0], slot[0]
                ).reshape(1)
                zero_offset = jnp.zeros((1,), jnp.int32)
                num_chunks = (maximum[0] + capacity - 1) // capacity

                def add_chunk(q, acc):
                    low = q * capacity
                    high = low + capacity
                    chunk_counts = jnp.maximum(
                        0, jnp.minimum(ends, high) - jnp.maximum(starts, low)
                    )
                    chunk_ids = jax.lax.dynamic_slice(
                        ids_pad, (q * chunk_rows, 0), (chunk_rows, 128)
                    )
                    chunk_probs = jax.lax.dynamic_slice(
                        probs_pad, (q * chunk_rows, 0), (chunk_rows, 128)
                    )
                    chunk_scale_args = (latent_scales,) if w4a8_16k_enabled else ()
                    chunk_out = chunk_routed_kernel(
                        jnp.asarray(tables),
                        family_slot,
                        zero_offset,
                        chunk_counts,
                        chunk_ids,
                        chunk_probs,
                        latent,
                        w["expert_gu"],
                        w["expert_gus"],
                        w["expert_down"],
                        w["expert_ds"],
                        *chunk_scale_args,
                    )[0]
                    return acc + chunk_out.astype(jnp.float32)

                reduced = jax.lax.fori_loop(
                    0, num_chunks, add_chunk, jnp.zeros((rows, 3584), jnp.float32)
                )
            else:
                extra_slots = w["expert_gu"].shape[1] - local_experts
                counts_for_prep = (
                    jnp.pad(counts, (0, extra_slots)) if extra_slots > 0 else counts
                )
                reduced = routed_experts_prepared(
                    latent,
                    ids.reshape(-1),
                    probs.reshape(-1),
                    counts_for_prep,
                    w["expert_gu"],
                    w["expert_gus"],
                    w["expert_down"],
                    w["expert_ds"],
                    reduce_tables=tables,
                    expert_layer=slot,
                    global_route_max=maximum[0],
                )
            latent_up_slot = w["latent_up"][slot[0]]
            if dense_fp8:
                latent_up_slot = (
                    latent_up_slot.astype(jnp.float32) * w["latent_up_scale"][slot[0]]
                ).astype(jnp.bfloat16)
            selected = {
                "latent_norm": w["latent_norm"][slot[0]],
                "latent_up": latent_up_slot,
            }
            return k._moe_sp_tail(  # noqa: SLF001 -- existing prefill_megakernel helper
                reduced,
                shared,
                prefix[0],
                length,
                selected,
                axis_name="k3_sp",
                eps=1e-5,
                shared_reduced=True,
            )

        if with_next_mixture:
            assert overflow_next_norm_op is not None and next_normed is not None

            def overflow_with_next():
                ovf_final = overflow()
                ovf_next = overflow_next_norm_op(
                    ovf_final, blocks[0], next_folded[0], next_norm[0], next_active
                )
                return ovf_final, ovf_next

            final_pair, next_pair = jax.lax.cond(
                maximum[0] <= capacity,
                lambda: (final, next_normed),
                overflow_with_next,
            )
            moe_result = (final_pair[None], routes) if counting else final_pair[None]
            return moe_result, next_pair[None]

        result = jax.lax.cond(maximum[0] <= capacity, lambda: final, overflow)[None]
        return (result, routes) if counting else result

    return jax.jit(local)


def _make_moe_compound(
    mesh, seq_len: int, part_tokens: tuple[int, ...], *, with_mixture: bool = False
):
    if mesh.size != 32 or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("Fused prefill MoE requires a flat 32-rank k3_sp mesh")
    cfg = k3_config.active()
    ep = cfg.prefill_expert_parallel
    if ep not in (8, 16):
        raise ValueError(
            "Fused prefill MoE requires prefill_expert_parallel in (8, 16)"
        )
    if sum(part_tokens) != seq_len:
        raise ValueError(f"Compound MoE parts {part_tokens} do not sum to {seq_len}")
    from .layer import routed_experts_prepared

    window_rows = seq_len // 32
    base_local_experts = 896 // ep
    configured_replica_slots = (
        rank_loader.prefill_replica_slots()
        if (cfg.prefill_replica_plan and ep == 16)
        else 0
    )
    part_replica_slots = tuple(
        configured_replica_slots if p_tok in (8192, 16384) else 0
        for p_tok in part_tokens
    )
    mid_width = 3072 // (32 // ep)
    tables = k.ring_tables(list(mesh.devices.flat), tuple(range(32)))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    bf16 = jnp.bfloat16
    counting = cfg.prefill_route_counts
    dense_fp8 = cfg.prefill_moe_dense_fp8
    inputs = (15 if with_mixture else 13) + (4 if dense_fp8 else 0)

    tok_offsets: list[int] = []
    row_offsets: list[int] = []
    acc_tok = 0
    for p_tok in part_tokens:
        tok_offsets.append(acc_tok)
        row_offsets.append(acc_tok // 32)
        acc_tok += p_tok
    part_rows = tuple(p_tok // 32 for p_tok in part_tokens)
    can_overflow_parts = tuple(p_tok == 8192 and ep >= 8 for p_tok in part_tokens)

    def make_kernel(part_idx: int):
        p_tok = part_tokens[part_idx]
        p_rows = part_rows[part_idx]
        r_off = row_offsets[part_idx]
        aliased = part_idx > 0
        can_overflow_part = can_overflow_parts[part_idx]
        stage_outputs_hbm = p_tok == 8192 and ep == 8
        has_fallback_shared = can_overflow_part and not stage_outputs_hbm
        p_replica_slots = part_replica_slots[part_idx]
        p_use_replicas = p_replica_slots > 0
        p_local_experts = base_local_experts + p_replica_slots
        p_num_scalar = 2 + int(p_use_replicas)
        p_shape = (p_tok // 8, 128)
        workspace_shapes = [
            ((window_rows, 7168), bf16),
            ((p_tok, 2, 1792), bf16),
            ((p_tok * 16, mid_width), bf16),
        ]
        output_specs = [hbm] * 3
        if can_overflow_part:
            workspace_shapes += [
                (p_shape, jnp.int32),
                (p_shape, jnp.float32),
                ((p_local_experts,), jnp.int32),
            ]
            output_specs += [hbm] * 3
            if has_fallback_shared:
                workspace_shapes += [((p_rows, 7168), bf16)]
                output_specs += [hbm]
            workspace_shapes += [((1,), jnp.int32)]
            output_specs += [pl.BlockSpec()]
        if stage_outputs_hbm:
            workspace_shapes += [((p_rows, 7168), bf16), ((p_rows, 3584), bf16)]
            output_specs += [hbm, hbm]
        if counting:
            workspace_shapes += [((base_local_experts,), jnp.int32)]
            output_specs += [pl.BlockSpec(memory_space=pltpu.SMEM)]
        kernel_name = (
            f"fused_mixture_moe_pallas_{p_tok}"
            if with_mixture
            else f"fused_moe_pallas_{p_tok}"
        )

        def body(*refs):
            refs = list(refs)
            replica_table_ref = refs.pop(2) if p_use_replicas else None
            if aliased:
                del refs[2 + inputs]
            own = pl.ds(r_off, p_rows)
            for idx in (8, 9, 10, 11):  # expert_gu, expert_gus, expert_down, expert_ds
                if refs[idx].shape[1] < p_local_experts:
                    raise ValueError(
                        f"Expert tensor slot count {refs[idx].shape[1]} < required "
                        f"{p_local_experts}"
                    )
                if refs[idx].shape[1] != p_local_experts:
                    refs[idx] = refs[idx].at[:, pl.ds(0, p_local_experts)]
            refs[14] = refs[14].at[own]
            refs[2 + inputs] = refs[2 + inputs].at[own]
            mix_refs = None
            if with_mixture:
                blocks_ref = refs[2].at[:, own]
                refs[2] = None
                ffn_norm_ref = refs.pop(16)
                mlp_res_ref = refs.pop(15)
                mix_refs = (blocks_ref, mlp_res_ref, ffn_norm_ref)
            else:
                refs[2] = refs[2].at[own]
            dense_scales = None
            if dense_fp8:
                dense_scales = tuple(refs[15:19])
                del refs[15:19]
            route_counts = refs.pop() if counting else None
            reduced_hbm = refs.pop() if stage_outputs_hbm else None
            shared_hbm = refs.pop() if stage_outputs_hbm else None
            maximum_ref = refs.pop() if can_overflow_part else None
            fallback_shared_ref = refs.pop() if has_fallback_shared else None
            fallback_counts_ref = refs.pop() if can_overflow_part else None
            fallback_probs_ref = refs.pop() if can_overflow_part else None
            fallback_ids_ref = refs.pop() if can_overflow_part else None
            return _layer_kernel(
                *refs,
                fallback_ids=fallback_ids_ref,
                fallback_probs=fallback_probs_ref,
                fallback_counts=fallback_counts_ref,
                fallback_shared=fallback_shared_ref,
                maximum=maximum_ref,
                route_counts=route_counts,
                mixture_refs=mix_refs,
                shared_hbm=shared_hbm,
                reduced_hbm=reduced_hbm,
                dense_scales=dense_scales,
                replica_table=replica_table_ref,
                replica_slots=p_replica_slots,
            )

        return pl.pallas_call(
            body,
            out_shape=tuple(jax.ShapeDtypeStruct(s, d) for s, d in workspace_shapes),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=p_num_scalar,
                in_specs=(hbm,) * (inputs + aliased),
                out_specs=tuple(output_specs),
            ),
            input_output_aliases={p_num_scalar + inputs: 0} if aliased else {},
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                vmem_limit_bytes=64 * 1024**2,
                disable_bounds_checks=True,
                shape_invariant_numerics=False,
            ),
            name=kernel_name,
        )

    kernels = tuple(make_kernel(i) for i in range(len(part_tokens)))
    in_specs = (
        (
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
            P(),
            P("k3_sp"),
            P(),
            P(),
        )
        if with_mixture
        else (P("k3_sp"), P("k3_sp"), P("k3_sp"), P(), P())
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(P("k3_sp"), P("k3_sp")) if counting else P("k3_sp"),
        check_vma=False,
    )
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
        dense_scale_args = (
            tuple(w[name] for name in MOE_DENSE_FP8_SCALES) if dense_fp8 else ()
        )
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
            *dense_scale_args,
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
            rep_args = (
                (w["prefill_replica_table"][slot[0]],)
                if part_replica_slots[idx] > 0
                else ()
            )
            outputs = list(
                kernel(jnp.asarray(tables), control, *rep_args, *operands, *previous)
            )
            if counting:
                counted = outputs.pop()
                routes = counted if routes is None else routes + counted
            if part_tokens[idx] == 8192 and ep == 8:
                outputs.pop()
                staged_shared = outputs.pop()
                if can_overflow_parts[idx]:
                    maximum_out = outputs.pop()
                    outputs.extend((staged_shared, maximum_out))
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
            _, latent, _, ids, probs, counts, shared, maximum = parts_out[idx]
            p_local_experts = base_local_experts + part_replica_slots[idx]
            extra_slots = w["expert_gu"].shape[1] - p_local_experts
            counts_for_prep = (
                jnp.pad(counts, (0, extra_slots)) if extra_slots > 0 else counts
            )
            reduced = routed_experts_prepared(
                latent,
                ids.reshape(-1),
                probs.reshape(-1),
                counts_for_prep,
                w["expert_gu"],
                w["expert_gus"],
                w["expert_down"],
                w["expert_ds"],
                reduce_tables=tables,
                expert_layer=slot,
                global_route_max=maximum[0],
            )
            latent_up_slot = w["latent_up"][slot[0]]
            if dense_fp8:
                latent_up_slot = (
                    latent_up_slot.astype(jnp.float32) * w["latent_up_scale"][slot[0]]
                ).astype(jnp.bfloat16)
            selected = {
                "latent_norm": w["latent_norm"][slot[0]],
                "latent_up": latent_up_slot,
            }
            return k._moe_sp_tail(  # noqa: SLF001 -- existing prefill_megakernel helper
                reduced,
                shared,
                prefix[0][r_off : r_off + p_rows],
                lengths[idx],
                selected,
                axis_name="k3_sp",
                eps=1e-5,
                shared_reduced=True,
            )

        fits = [
            (parts_out[i][7][0] <= _COMPACT_8K_CAPACITY)
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

    return jax.jit(local)
