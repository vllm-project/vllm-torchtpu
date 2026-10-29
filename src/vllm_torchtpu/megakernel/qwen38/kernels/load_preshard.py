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
"""Load a lossless NVFP4 checkpoint by reading each host-owned cell once.

The published manifest is the commit marker. Record and file checksums, native
shapes, and complete mathematical coverage are verified while streaming into
final donated device buffers; there is no second resident model or disk cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import multiprocessing
import os
import re
import shutil
import tempfile
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import numpy as np

from . import MODEL_ID, MODEL_REVISION, Config
from .layout import Layout, memory_plan
from .preshard import (
    ATOM_WIDTH,
    EXPERT_BLOCK,
    MAX_HEADER_BYTES,
    ExpertTile,
    _geometry,
    host_cells,
    owner_rank,
    read_records,
    records_to_tile,
    split_tile,
    tile_updates,
)

MANIFEST_NAME = "preshard-manifest.json"
FORMAT = "qwen-preshard-nvfp4-v1"
EXPERT_FAMILIES = frozenset(("egu", "egu_scale", "edw", "edw_scale", "eparams"))
_HEX256 = re.compile(r"[0-9a-f]{64}\Z")
STREAM_BUFFER_BYTES = 1 << 20
CONVERTED_CELL_ROOT = "/dev/shm"


def dense_extents(layout):
    """Expected native dense updates, excluding alignment rows supplied by zeros."""
    c, specs = layout.config, layout.record_specs
    expected = {}

    def add(name, ordinal=None, shape=None):
        if ordinal is None:
            extent, starts = specs[name].shape, (0,) * len(specs[name].shape)
        else:
            extent = (1, *(specs[name].shape[1:] if shape is None else shape))
            starts = (ordinal, *((0,) * (len(extent) - 1)))
        expected[name, starts] = tuple(extent)

    for name in ("emb", "lmw", "fnorm"):
        add(name)
    for layer in range(c.layers):
        for name in ("norm_in", "norm_post", "router", "shared_gate", "gu", "dw"):
            add(name, layer)
    for ordinal, _ in enumerate(c.linear_layers):
        for name in ("qkvz", "outw", "convw", "onorm", "lparams"):
            add(name, ordinal)
        for name in ("alog", "dtb"):
            add(name, ordinal, (layout.local_v_heads,))
    for ordinal, _ in enumerate(c.full_layers):
        for name in ("qwkv", "ow", "qn", "kn", "fparams"):
            add(name, ordinal)
    return expected


def logical_hosts(local_ranks):
    """Use mesh logical ranks, never Slurm task IDs or physical device IDs."""
    ranks = set(local_ranks)
    if not ranks or any(type(rank) is not int or rank not in range(32) for rank in ranks):
        raise ValueError("Pre-sharded loading requires TP32 logical ranks")
    hosts = sorted({rank // 8 for rank in ranks})
    if ranks != {8 * host + lane for host in hosts for lane in range(8)}:
        raise ValueError("Pre-sharded loading requires complete contiguous groups of 8 ranks")
    return hosts


class _HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha256 = hashlib.sha256()
        self.size = 0

    def readinto(self, target):
        count = self.stream.readinto(target)
        if count:
            self.sha256.update(memoryview(target).cast("B")[:count])
            self.size += count
        return count

    def read(self, count):
        value = self.stream.read(count)
        self.sha256.update(value)
        self.size += len(value)
        return value


class PreshardedCheckpoint:
    """Validated complete model metadata with bounded, single-pass record readers."""

    def __init__(self, directory, config=None):
        self.root = Path(directory)
        if any(part.endswith(".partial") for part in self.root.parts):
            raise ValueError("Pre-sharded staging directories must be published before loading")
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError("Pre-sharded checkpoint is incomplete: published manifest is missing")
        if manifest_path.stat().st_size > 4 << 20:
            raise ValueError("Pre-sharded manifest exceeds its metadata limit")
        self.manifest = m = json.loads(manifest_path.read_text())
        if not isinstance(m, dict) or m.get("format") != FORMAT:
            raise ValueError("Unsupported pre-sharded checkpoint format")
        if m.get("complete_model") is not True:
            raise ValueError("Production loading requires a complete pre-sharded model")
        for name, expected in (
            ("tp", 32),
            ("router_tp", 4),
            ("atom_width", ATOM_WIDTH),
        ):
            if m.get(name) != expected:
                raise ValueError(f"Unsupported pre-sharded {name}")
        if m.get("source_model") != MODEL_ID or m.get("source_revision") != MODEL_REVISION:
            raise ValueError("Pre-sharded checkpoint must preserve the pinned native NVFP4 source")
        if not isinstance(m.get("binding_sha256"), str) or not _HEX256.fullmatch(
            m["binding_sha256"]
        ):
            raise ValueError("Pre-sharded checkpoint omits its source/plan binding")
        config_path = self.root / "config.json"
        if config_path.stat().st_size > 1 << 20:
            raise ValueError("Pre-sharded config exceeds its metadata limit")
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != m.get("config_sha256"):
            raise ValueError("Pre-sharded config checksum mismatch")
        self.config = Config.from_checkpoint(self.root)
        if config is not None and self.config != config:
            raise ValueError("Checkpoint config differs from the requested layout")
        if self.config.quantization != "nvfp4":
            raise ValueError("Pre-sharded checkpoint requires native NVFP4")
        Layout(self.config, tp=32, expert_tp=16, router_tp=4)
        if m.get("expert_block") != min(EXPERT_BLOCK, self.config.experts // 8):
            raise ValueError("Unsupported pre-sharded expert_block")
        # Validate every advertised object before any device allocation. The full
        # SHA256 is then verified during the same sequential pass that loads it.
        self.cells, self.dense, paths = {}, {}, set()
        dense_layout = Layout(self.config)
        extents = dense_extents(dense_layout)
        dense_count = len(extents)
        dense_bytes = sum(
            math.prod(shape) * np.dtype(dense_layout.weight_specs[name].dtype).itemsize
            for (name, _), shape in extents.items()
        )
        experts, channels = self.config.experts // 4, self.config.moe_intermediate // 2
        tiles = experts // m["expert_block"] * (channels // ATOM_WIDTH)
        expert_bytes = (
            experts * channels * self.config.dim * 27 // 16
            + experts * (channels // ATOM_WIDTH) * 24
        )
        counts = {
            "dense_shards": (dense_count, dense_bytes),
            "expert_cells": (tiles * 7, expert_bytes),
        }
        for kind, target, fields in (
            ("expert_cells", self.cells, ("layer", "expert_quarter", "channel_half")),
            ("dense_shards", self.dense, ("rank",)),
        ):
            entries = m.get(kind)
            if not isinstance(entries, list):
                raise ValueError(f"Pre-sharded manifest omits {kind}")
            for entry in entries:
                if not isinstance(entry, dict) or any(
                    type(entry.get(k)) is not int for k in fields
                ):
                    raise ValueError(f"Invalid pre-sharded {kind} coordinates")
                key = tuple(entry[k] for k in fields)
                if key in target:
                    raise ValueError(f"Duplicate pre-sharded {kind} entry")
                name = entry.get("path")
                if not isinstance(name, str) or not name:
                    raise ValueError("Invalid pre-sharded object path")
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or name in paths:
                    raise ValueError("Pre-sharded object paths must be unique and relative")
                path = self.root / relative
                if not path.resolve().is_relative_to(self.root.resolve()):
                    raise ValueError("Pre-sharded object path escapes its checkpoint")
                paths.add(name)
                if not isinstance(entry.get("sha256"), str) or not _HEX256.fullmatch(
                    entry["sha256"]
                ):
                    raise ValueError("Invalid pre-sharded object checksum")
                for field in ("size", "records", "payload_bytes"):
                    if type(entry.get(field)) is not int or entry[field] <= 0:
                        raise ValueError(f"Invalid pre-sharded object {field}")
                if (entry["records"], entry["payload_bytes"]) != counts[kind]:
                    raise ValueError(f"Pre-sharded {kind} record/payload coverage mismatch")
                if not path.is_file() or path.stat().st_size != entry["size"]:
                    raise ValueError(f"Missing or truncated pre-sharded object: {name}")
                target[key] = entry
        expected_cells = {
            (layer, e, i) for layer in range(self.config.layers) for e in range(4) for i in range(2)
        }
        if set(self.cells) != expected_cells or set(self.dense) != {(r,) for r in range(32)}:
            raise ValueError("Pre-sharded manifest has incomplete or unexpected model coverage")
        if m.get("selected_layers") != list(range(self.config.layers)) or m.get(
            "selected_ranks"
        ) != list(range(32)):
            raise ValueError("Pre-sharded selected layers/ranks do not cover the complete model")

    def _records(self, entry, max_record_bytes, coordinates):
        # Buffered framing reads avoid separate FUSE calls for each tag/length/
        # JSON header. Large readinto payloads still stream into bounded arrays.
        with (self.root / entry["path"]).open("rb", buffering=STREAM_BUFFER_BYTES) as stream:
            report = {}
            yield from read_records(
                stream,
                expected_model=self.manifest["source_model"],
                expected_revision=self.manifest["source_revision"],
                max_record_bytes=max_record_bytes,
                report=report,
                expected_sha256=entry["sha256"],
                expected_metadata={
                    "binding_sha256": self.manifest["binding_sha256"],
                    "config_sha256": self.manifest["config_sha256"],
                    **coordinates,
                },
            )
            if any(
                report.get(key) != entry[key]
                for key in ("records", "payload_bytes", "size", "sha256")
            ):
                raise ValueError(
                    f"Pre-sharded object manifest checksum/count mismatch: {entry['path']}"
                )

    def iter_dense(self, rank, layout, *, max_record_bytes=128 << 20):
        if layout.config != self.config or layout.tp != 32 or layout.router_tp != 4:
            raise ValueError("Dense pre-shards require matching TP32 and router TP4")
        expected = dense_extents(layout)
        for record in self._records(
            self.dense[rank,], max_record_bytes, {"kind": "dense", "rank": rank}
        ):
            metadata = record.metadata
            if (
                not isinstance(metadata, dict)
                or set(metadata) != {"rank", "offsets"}
                or metadata["rank"] != rank
            ):
                raise ValueError("Dense pre-shard rank/offset metadata mismatch")
            offsets = metadata["offsets"]
            if not isinstance(offsets, list) or any(type(x) is not int for x in offsets):
                raise ValueError("Dense pre-shard offsets must be integers")
            key = record.name, tuple(offsets)
            shape = expected.pop(key, None)
            if shape is None or record.array.shape != shape:
                raise ValueError("Dense pre-shard has duplicate, missing, or invalid extent")
            if record.array.dtype != np.dtype(layout.weight_specs[record.name].dtype):
                raise ValueError("Dense pre-shard dtype mismatch")
            yield record.name, tuple(offsets), record.array
        if expected:
            raise ValueError("Dense pre-shard has incomplete mathematical coverage")

    def iter_cell(self, layer, expert_quarter, channel_half, *, max_record_bytes=128 << 20):
        c = self.config
        elo, ehi = expert_quarter * (c.experts // 4), (expert_quarter + 1) * (c.experts // 4)
        clo, chi = (
            channel_half * (c.moe_intermediate // 2),
            (channel_half + 1) * (c.moe_intermediate // 2),
        )
        block = min(EXPERT_BLOCK, c.experts // 8)
        # Reject an oversized header before allocating its payload. A valid
        # expert record is at most one gate/down atom (8 MiB in production).
        max_record_bytes = min(max_record_bytes, block * c.dim * ATOM_WIDTH // 2)
        expected = {
            (e, channel): min(block, ehi - e)
            for e in range(elo, ehi, block)
            for channel in range(clo, chi, ATOM_WIDTH)
        }
        records = []
        params = {}
        for record in self._records(
            self.cells[layer, expert_quarter, channel_half],
            max_record_bytes,
            {
                "kind": "expert",
                "layer": layer,
                "expert_quarter": expert_quarter,
                "channel_half": channel_half,
            },
        ):
            records.append(record)
            if len(records) != 7:
                continue
            tile = records_to_tile(records, config=c)
            records = []
            count = expected.pop((tile.expert_start, tile.channel_start), None)
            if tile.layer != layer or count != tile.expert_count:
                raise ValueError("Expert pre-shard has duplicate, missing, or invalid tile extent")
            calibration = tile.arrays["eparams"].tobytes()
            previous = params.setdefault(tile.expert_start, calibration)
            if previous != calibration:
                raise ValueError("Expert calibration scales differ between channel atoms")
            yield tile
        if records or expected:
            raise ValueError("Expert pre-shard has incomplete mathematical coverage")


class PrefetchedIterator:
    """One bounded CPU item ahead, with deterministic file/thread cleanup.

    The producer waits for an empty slot *before* advancing the source. Thus
    there is one consumer item and at most one queued OR being-read item, rather
    than a full queue plus another producer allocation. All device work remains
    on the consumer thread. Cancellation waits for the current bounded read to
    finish; an already-blocking filesystem syscall cannot be safely interrupted.
    """

    _EMPTY = object()

    def __init__(self, iterable, *, max_item_bytes, size_of):
        if max_item_bytes <= 0:
            raise ValueError("Prefetch item limit must be positive")
        self._source = iter(iterable)
        self._limit = max_item_bytes
        self._size_of = size_of
        self._condition = threading.Condition()
        self._item = self._EMPTY
        self._done = self._cancelled = False
        self._failure = None
        self.thread = None

    def __enter__(self):
        if self.thread is not None:
            raise RuntimeError("Checkpoint prefetch iterators cannot be reused")
        self.thread = threading.Thread(
            target=self._produce, name="qwen-checkpoint-prefetch", daemon=True
        )
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.thread is None:
            raise RuntimeError("Use checkpoint prefetch as a context manager")
        with self._condition:
            self._condition.wait_for(
                lambda: self._item is not self._EMPTY or self._done or self._cancelled
            )
            if self._cancelled:
                raise StopIteration
            if self._item is not self._EMPTY:
                item, self._item = self._item, self._EMPTY
                self._condition.notify_all()
                return item
            if self._failure is not None:
                raise self._failure
            raise StopIteration

    def _produce(self):
        failure = None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._item is self._EMPTY or self._cancelled)
                    if self._cancelled:
                        break
                try:
                    item = next(self._source)
                except StopIteration:
                    break
                if self._size_of(item) > self._limit:
                    raise ValueError("Checkpoint item exceeds bounded prefetch limit")
                with self._condition:
                    if self._cancelled:
                        break
                    self._item = item
                    self._condition.notify_all()
                del item
        except BaseException as error:
            failure = error
        finally:
            try:
                close = getattr(self._source, "close", None)
                if close is not None:
                    close()
            except BaseException as error:
                if failure is None:
                    failure = error
            with self._condition:
                self._failure, self._done = failure, True
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        if self.thread is not None:
            self.thread.join()
        else:
            close = getattr(self._source, "close", None)
            if close is not None:
                close()
        # Drop unread arrays and exception tracebacks only after the producer
        # has closed its source; queued DMA inputs have independent ownership.
        self._item, self._source, self._failure = self._EMPTY, None, None


def staging_plan(layout, max_staging_bytes):
    """Conservative live host/device payload bound, excluding resident weights.

    Reserve two item-sized host arrays (consumer + one prefetched item), two
    retained host/device transfer pairs, one stream buffer and bounded framing.
    GCSFuse's independently managed read-ahead is outside this process budget.
    """
    payload_limit = max_staging_bytes // 8
    specs = layout.weight_specs
    largest_dense = max(
        math.prod(shape) * np.dtype(specs[name].dtype).itemsize
        for (name, _), shape in dense_extents(layout).items()
    )
    largest_atom = min(EXPERT_BLOCK, layout.config.experts // 8) * (
        layout.config.dim * ATOM_WIDTH * 27 // 16 + 6 * 4
    )
    explicit_bytes = (
        2 * max(largest_dense, largest_atom)
        + 4 * payload_limit
        + STREAM_BUFFER_BYTES
        + 8 * MAX_HEADER_BYTES
    )
    if layout.fast:
        from .precision import CONVERSION_SCRATCH_BYTES

        # Conversion outputs retain at most one native-sized atom alongside
        # bounded per-expert arithmetic scratch and the existing DMA queue.
        explicit_bytes += CONVERSION_SCRATCH_BYTES + largest_atom
    if largest_dense > payload_limit or explicit_bytes > max_staging_bytes:
        raise ValueError("Pre-sharded read/prefetch/transfer staging exceeds its bounded budget")
    return {
        "payload_limit": payload_limit,
        "largest_dense": largest_dense,
        "largest_atom": largest_atom,
        "stream_buffer_bytes": STREAM_BUFFER_BYTES,
        "prefetch_slots": 1,
        "explicit_staging_bytes": explicit_bytes,
    }


class DonatedUpdateQueue:
    """Bounded host/device transfers retaining source arrays through completion."""

    def __init__(self, buffers, devices, *, payload_limit, log=None, max_inflight=2):
        from .load import _DonatedSliceUpdater

        if max_inflight not in (1, 2) or payload_limit <= 0:
            raise ValueError("Loader transfer queue must have one or two bounded entries")
        self.buffers = buffers
        self.devices = devices
        self.payload_limit = payload_limit
        self.max_inflight = max_inflight
        self.pending = deque()
        self.updaters = {
            rank: _DonatedSliceUpdater(device, log=log) for rank, device in devices.items()
        }

    def _complete_one(self):
        completed = self.pending.popleft()
        completed[1].block_until_ready()
        del completed

    def submit(self, rank, name, offsets, value):
        if value.nbytes > self.payload_limit:
            raise ValueError("Pre-sharded update exceeds staging budget")
        key = rank, name
        # Do not retain a subsequently donated/deleted destination handle in
        # the completion queue. Drain any earlier update of this destination.
        while len(self.pending) >= self.max_inflight or any(
            item[0] == key for item in self.pending
        ):
            self._complete_one()
        device = self.devices[rank]
        chunk = jax.device_put(value[None], device)
        starts = jax.device_put(np.asarray((0, *offsets), np.int32), device)
        try:
            result = self.updaters[rank](self.buffers[rank][name], chunk, starts)
        except BaseException:
            # Compile/alias/capacity failures occur after enqueueing the H2D
            # copies. Keep the caller's source alive until those copies finish.
            jax.block_until_ready((chunk, starts))
            raise
        self.buffers[rank][name] = result
        self.pending.append((key, result, chunk, starts, value))

    def drain(self):
        failure = None
        while self.pending:
            try:
                self._complete_one()
            except BaseException as error:
                # A failed completion must not abandon other in-flight inputs.
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure


def conversion_workers():
    """Expert-cell worker processes, leaving cores for this process's transfers."""
    return max(1, min(64, len(os.sched_getaffinity(0)) - 8))


_CELL_SOURCE = None


def _set_cell_source(source):
    global _CELL_SOURCE
    _CELL_SOURCE = source


def _convert_cell(
    layer, quarter, half, expert_tp, fast, max_record_bytes, path, overlap=0, copies_only=False
):
    """Read, verify and convert one expert cell in a worker process.

    Final-layout updates are written back to back into ``path`` on host shared
    memory; the returned table locates them. Every piece keeps its calibration
    bytes so the loader can repeat the cross-cell consistency check. With
    ``overlap``, each group's leading experts also yield copies for the successor
    group's slots past its own experts (calibration None); ``copies_only`` skips
    the owners' updates.
    """
    source, config = _CELL_SOURCE, _CELL_SOURCE.config
    local_experts, _ = _geometry(config, expert_tp)
    groups = 32 // expert_tp
    update_fn = tile_updates
    if fast:
        from .precision import fast_tile_updates

        update_fn = fast_tile_updates
    pieces, offset = [], 0
    with open(path, "wb") as stream:

        def write(piece, rank, slot_offset=0):
            nonlocal offset
            updates = []
            for name, offsets, value in update_fn(
                piece, rank=rank, expert_tp=expert_tp, config=config, include_params=not slot_offset
            ):
                if slot_offset and name == "eparams":
                    continue
                offsets = (offsets[0], offsets[1] + slot_offset, *offsets[2:])
                value = np.ascontiguousarray(value)
                stream.write(value.reshape(-1).view(np.uint8).data)
                updates.append((name, tuple(offsets), value.dtype, value.shape, offset))
                offset += value.nbytes
            return updates

        for tile in source.iter_cell(layer, quarter, half, max_record_bytes=max_record_bytes):
            for piece in split_tile(tile, expert_tp=expert_tp, config=config):
                rank = owner_rank(piece.expert_start, piece.channel_start, expert_tp, config=config)
                if not copies_only:
                    calibration = piece.arrays["eparams"].tobytes()
                    pieces.append((rank, piece.expert_start, calibration, write(piece, rank)))
                copied = min(piece.expert_count, overlap - piece.expert_start % local_experts)
                if copied > 0:
                    lead = ExpertTile(
                        piece.layer,
                        piece.expert_start,
                        piece.channel_start,
                        piece.channel_width,
                        {name: value[:copied] for name, value in piece.arrays.items()},
                    )
                    successor = (rank // expert_tp + 1) % groups * expert_tp + rank % expert_tp
                    pieces.append(
                        (successor, piece.expert_start, None, write(lead, rank, local_experts))
                    )
    return pieces


def _map_cell(path):
    """Map a converted cell read-only and unlink it; pages live until the last view."""
    with open(path, "rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    os.remove(path)
    return mapped


def load_presharded(
    checkpoint, mesh, layout, *, max_staging_bytes=1 << 30, log=None, workers=None
):
    """Read each owned stream once and donate updates to all local final buffers.

    Dense shards use one CPU reader that prefetches at most one bounded record.
    Expert cells are read, verified and converted by ``workers`` spawned
    processes (default ``conversion_workers()``), at most ``workers`` plus one
    layer of cells ahead, into host shared memory. Updates are submitted in the
    serial loader's order. Two transfers may be outstanding. Staging accounts
    for the prefetched item, current item, retained host/device pairs and the
    1 MiB stream buffer. GCSFuse read-ahead buffers are managed separately.
    """
    from .load import _global_arrays, _zero_device, device_record

    if layout.tp != 32 or mesh.size != 32 or tuple(mesh.axis_names) != ("tp",):
        raise ValueError("Pre-sharded loading requires the TP32 mesh")
    if layout.expert_tp not in (1, 4, 8, 16) or layout.router_tp != 4:
        raise ValueError("Pre-sharded loading supports expert TP1/4/8/16 and router TP4")
    if max_staging_bytes < 1 << 20:
        raise ValueError("max_staging_bytes must be at least 1 MiB")
    source = PreshardedCheckpoint(checkpoint, layout.config)
    specs = layout.weight_specs
    local = {
        rank: device
        for rank, device in enumerate(mesh.devices.flat)
        if device.process_index == jax.process_index()
    }
    hosts = logical_hosts(local)
    staging = staging_plan(layout, max_staging_bytes)
    payload_limit = staging["payload_limit"]
    if log:
        log(
            f"Pre-sharded staging: {json.dumps(staging, sort_keys=True)}; GCSFuse read-ahead is separate"
        )
    physical = memory_plan(layout.config, layout, 1, 1)["physical_weight_bytes_per_device_estimate"]
    for device in local.values():
        stats = device.memory_stats() or {}
        if "bytes_limit" in stats:
            available = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
            if physical + max_staging_bytes > available:
                raise MemoryError(
                    f"Device {device.id}: pre-sharded weights/staging exceed available HBM"
                )
    buffers = {}
    for rank, device in local.items():
        if log:
            log(
                f"Allocating pre-sharded rank {rank}: {physical / 2**30:.2f} GiB estimated physical weights"
            )
        buffers[rank] = {name: _zero_device(spec, device) for name, spec in specs.items()}
        jax.block_until_ready(buffers[rank])
    queue = DonatedUpdateQueue(buffers, local, payload_limit=payload_limit, log=log)

    try:
        for rank in local:
            with PrefetchedIterator(
                source.iter_dense(rank, layout, max_record_bytes=payload_limit),
                max_item_bytes=staging["largest_dense"],
                size_of=lambda item: item[2].nbytes,
            ) as records:
                for name, offsets, value in records:
                    queue.submit(rank, name, offsets, device_record(layout, name, value))
            if log:
                log(f"Rank {rank}: dense pre-shard verified")
        workers = conversion_workers() if workers is None else workers
        # With expert overlap a host also reads its predecessor host's cells, for the
        # copies its first group holds; the cells verify only as whole objects.
        cells = [
            (layer, quarter, half, copies)
            for layer in range(layout.config.layers)
            for host in hosts
            for copies, owner in ((False, host), (True, (host - 1) % 4))[
                : 2 if layout.expert_overlap else 1
            ]
            for quarter, half in host_cells(layout.expert_tp, owner, config=layout.config)
        ]
        per_layer = len(cells) // layout.config.layers
        scratch = tempfile.mkdtemp(
            prefix="qwen-load-",
            dir=CONVERTED_CELL_ROOT if os.path.isdir(CONVERTED_CELL_ROOT) else None,
        )
        pool = ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_set_cell_source,
            initargs=(source,),
        )
        if log:
            log(f"Pre-sharded expert cells: {workers} conversion processes, scratch {scratch}")
        pending, remaining = deque(), iter(cells)

        def submit_cell():
            cell = next(remaining, None)
            if cell is not None:
                *coordinates, copies = cell
                path = os.path.join(scratch, "-".join(map(str, cell)))
                pending.append((path, pool.submit(
                    _convert_cell, *coordinates, layout.expert_tp, layout.fast, payload_limit,
                    path, layout.expert_overlap, copies,
                )))

        try:
            for _ in range(workers + per_layer):
                submit_cell()
            for layer in range(layout.config.layers):
                seen_params, written_params = {}, set()
                for _ in range(per_layer):
                    path, future = pending.popleft()
                    pieces = future.result()
                    mapped = _map_cell(path)
                    submit_cell()
                    for rank, expert_start, calibration, updates in pieces:
                        if calibration is None:
                            # A copy for a successor group on another host is that
                            # host's to read.
                            if rank in local:
                                for name, offsets, dtype, shape, offset in updates:
                                    value = np.frombuffer(
                                        mapped, dtype, math.prod(shape), offset
                                    ).reshape(shape)
                                    queue.submit(rank, name, offsets, value)
                            continue
                        if rank not in local:
                            raise ValueError("Owned pre-shard cell contains a nonlocal logical rank")
                        key = rank, expert_start
                        if seen_params.setdefault(key, calibration) != calibration:
                            raise ValueError("Expert calibration scales differ across cells")
                        # The same expert's scalar table is shared by all its atoms.
                        # Repeated updates are bounded and bit-identical; skipping
                        # after its first atom avoids unnecessary tiny transfers.
                        include_params = key not in written_params
                        written_params.add(key)
                        for name, offsets, dtype, shape, offset in updates:
                            if name == "eparams" and not include_params:
                                continue
                            value = np.frombuffer(
                                mapped, dtype, math.prod(shape), offset
                            ).reshape(shape)
                            queue.submit(rank, name, offsets, value)
                    # Views keep the mapping alive until their transfers complete.
                    pieces = mapped = value = None
                queue.drain()
                if log:
                    log(
                        f"Pre-sharded layer {layer + 1}/{layout.config.layers}: owned cells verified and resident"
                    )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            shutil.rmtree(scratch, ignore_errors=True)
        queue.drain()
    finally:
        queue.drain()
    for rank, updater in queue.updaters.items():
        if log:
            verified = sum(
                record["alias_bytes"] is not None for _, record in updater.variants.values()
            )
            log(
                f"Rank {rank} resident; {verified}/{len(updater.variants)} update variants have verified destination aliases"
            )
    return _global_arrays(mesh, specs, buffers)
