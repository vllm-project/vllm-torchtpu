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
"""CPU prototype for a lossless, layout-independent NVFP4 expert checkpoint.

This module does not modify or integrate with the production loader. A canonical
cell covers E/4 experts and I/2 intermediate channels; every supported host reads
two complete cells. Atoms are at most 16 experts by 128 intermediate channels.
The container is sequential and self-validating, with no backward writes.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import BinaryIO

import ml_dtypes
import numpy as np

from . import Config

FORMAT_VERSION = 1
MAGIC = b"QWENPS01"
ATOM_WIDTH = 128
EXPERT_BLOCK = 16
IO_CHUNK_BYTES = 8 << 20
MAX_HEADER_BYTES = 1 << 20
DEFAULT_CONFIG = Config(quantization="nvfp4")
FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "float32": np.dtype(np.float32),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3fn": FP8,
}
_TILE_NAMES = (
    "egu_gate",
    "egu_up",
    "egu_gate_scale",
    "egu_up_scale",
    "edw",
    "edw_scale",
    "eparams",
)


def _geometry(config, expert_tp):
    if config.quantization != "nvfp4" or expert_tp not in (1, 4, 8, 16):
        raise ValueError("Expected native NVFP4 and expert_tp in (1, 4, 8, 16)")
    if config.experts % (32 if expert_tp == 1 else 8) or config.moe_intermediate % (
        16 * ATOM_WIDTH
    ):
        raise ValueError("Expert dimensions must support every TP32 expert layout")
    if config.dim % 16:
        raise ValueError("Hidden dimensions must preserve complete NVFP4 groups")
    return config.experts * expert_tp // 32, config.moe_intermediate // expert_tp


def owner_rank(expert, channel, expert_tp, *, config=DEFAULT_CONFIG):
    """Return the TP32 logical rank owning one expert/intermediate coordinate."""
    local_experts, width = _geometry(config, expert_tp)
    if not 0 <= expert < config.experts or not 0 <= channel < config.moe_intermediate:
        raise ValueError("Expert or intermediate coordinate is outside the model")
    return expert // local_experts * expert_tp + channel // width


def host_cells(expert_tp, host, *, config=DEFAULT_CONFIG):
    """Return (expert-quarter, intermediate-half) cell IDs for logical host 0..3."""
    _geometry(config, expert_tp)
    if host not in range(4):
        raise ValueError("Logical host must be in [0, 4)")
    if expert_tp in (1, 4, 8):
        return ((host, 0), (host, 1))
    first = 2 * (host // 2)
    return ((first, host % 2), (first + 1, host % 2))


@dataclass(frozen=True)
class ExpertTile:
    layer: int
    expert_start: int
    channel_start: int
    channel_width: int
    arrays: Mapping[str, np.ndarray]

    @property
    def expert_count(self):
        return self.arrays["eparams"].shape[0]

    @property
    def payload_bytes(self):
        return sum(value.nbytes for value in self.arrays.values())


def _validate_batch(batch, config, expert_start):
    if set(batch) != {"gate", "up", "down", "gate_scale", "up_scale", "down_scale", "params"}:
        raise ValueError("Native batch must contain all three weights, scales and params")
    params = np.asarray(batch["params"])
    if params.ndim != 2 or params.shape[1] != 6 or params.dtype != np.float32:
        raise ValueError("Native params must have shape [experts, 6] and FP32 dtype")
    count = params.shape[0]
    if (
        not 0 < count <= EXPERT_BLOCK
        or not 0 <= expert_start < expert_start + count <= config.experts
    ):
        raise ValueError("Expert batch exceeds its bounded size or model range")
    if expert_start // (config.experts // 8) != (expert_start + count - 1) // (config.experts // 8):
        raise ValueError("Expert batch crosses a common ownership boundary")
    if not np.all(np.isfinite(params) & (params > 0)):
        raise ValueError("Native calibrated scales must be finite and positive")
    for name in ("gate", "up", "down"):
        n, k = (
            (config.dim, config.moe_intermediate)
            if name == "down"
            else (config.moe_intermediate, config.dim)
        )
        for suffix, divisor, dtype in (("", 2, np.dtype(np.uint8)), ("_scale", 16, FP8)):
            array = np.asarray(batch[name + suffix])
            if array.shape != (count, n, k // divisor) or array.dtype != dtype:
                raise ValueError(f"Wrong native shape or dtype for {name + suffix}")
    return count


def pack_expert_tile(
    batch, *, layer, expert_start, channel_start, channel_width=ATOM_WIDTH, config=DEFAULT_CONFIG
):
    """Transpose native packed bytes/scales without unpacking or quantizing.

    The input contains at most 16 complete experts: 432 MiB at production width.
    Its matrices have the checkpoint's [E,N,K/2] and [E,N,K/16] shapes. A returned
    tile is independently contiguous and about 27 MiB for a full expert batch.
    """
    _geometry(config, 4)
    _validate_batch(batch, config, expert_start)
    if not 0 <= layer < config.layers:
        raise ValueError("Layer is outside the model")
    if (
        channel_width != ATOM_WIDTH
        or channel_start % ATOM_WIDTH
        or not 0 <= channel_start < channel_start + channel_width <= config.moe_intermediate
    ):
        raise ValueError("Expected an aligned, complete 128-channel atom")
    lo, hi = channel_start, channel_start + channel_width
    arrays = {}
    for name in ("gate", "up"):
        arrays["egu_" + name] = np.ascontiguousarray(batch[name][:, lo:hi, :].transpose(0, 2, 1))
        arrays["egu_" + name + "_scale"] = np.ascontiguousarray(
            batch[name + "_scale"][:, lo:hi, :].transpose(0, 2, 1)
        )
    arrays["edw"] = np.ascontiguousarray(batch["down"][:, :, lo // 2 : hi // 2].transpose(0, 2, 1))
    arrays["edw_scale"] = np.ascontiguousarray(
        batch["down_scale"][:, :, lo // 16 : hi // 16].transpose(0, 2, 1)
    )
    arrays["eparams"] = np.array(batch["params"], copy=True, order="C")
    return ExpertTile(layer, expert_start, channel_start, channel_width, arrays)


def iter_expert_tiles(batch, *, layer, expert_start, config=DEFAULT_CONFIG):
    """Yield independent bounded atoms, retaining no previously yielded tile."""
    for start in range(0, config.moe_intermediate, ATOM_WIDTH):
        yield pack_expert_tile(
            batch, layer=layer, expert_start=expert_start, channel_start=start, config=config
        )


def _validate_tile(tile, config):
    _geometry(config, 4)
    if set(tile.arrays) != set(_TILE_NAMES):
        raise ValueError("Packed tile has missing or unexpected arrays")
    params = tile.arrays["eparams"]
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError("Packed params must have shape [experts, 6]")
    count, h, width = params.shape[0], config.dim, tile.channel_width
    if not 0 < count <= EXPERT_BLOCK or width != ATOM_WIDTH or tile.channel_start % width:
        raise ValueError("Packed tile has invalid atom dimensions")
    if not 0 <= tile.layer < config.layers:
        raise ValueError("Packed layer is outside the model")
    first_owner = owner_rank(tile.expert_start, tile.channel_start, 4, config=config)
    last_owner = owner_rank(
        tile.expert_start + count - 1, tile.channel_start + width - 1, 4, config=config
    )
    if first_owner != last_owner:
        raise ValueError("Packed tile crosses a common ownership boundary")
    expected = {
        "egu_gate": ((count, h // 2, width), np.dtype(np.uint8)),
        "egu_up": ((count, h // 2, width), np.dtype(np.uint8)),
        "egu_gate_scale": ((count, h // 16, width), FP8),
        "egu_up_scale": ((count, h // 16, width), FP8),
        "edw": ((count, width // 2, h), np.dtype(np.uint8)),
        "edw_scale": ((count, width // 16, h), FP8),
        "eparams": ((count, 6), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in expected.items():
        if tile.arrays[name].shape != shape or tile.arrays[name].dtype != dtype:
            raise ValueError(f"Wrong packed shape or dtype for {name}")
    if not np.all(np.isfinite(params) & (params > 0)):
        raise ValueError("Packed calibrated scales must be finite and positive")


def split_tile(tile, *, expert_tp, config=DEFAULT_CONFIG):
    """View a format-v1 atom as rank-owned pieces, including EP32 boundaries."""
    local_experts, _ = _geometry(config, expert_tp)
    _validate_tile(tile, config)
    offset = 0
    while offset < tile.expert_count:
        start = tile.expert_start + offset
        count = min(tile.expert_count - offset, local_experts - start % local_experts)
        yield ExpertTile(
            tile.layer,
            start,
            tile.channel_start,
            tile.channel_width,
            {name: value[offset : offset + count] for name, value in tile.arrays.items()},
        )
        offset += count


def tile_updates(tile, *, rank, expert_tp, config=DEFAULT_CONFIG, include_params=True):
    """Yield (family, offsets, array) updates for one rank; nonowners yield none.

    Offsets match qwen.load._rank_chunks, excluding the leading global-rank axis.
    Gate/up are separate updates so wider final shards retain the right ordering.
    """
    local_experts, width = _geometry(config, expert_tp)
    if rank not in range(32):
        raise ValueError("Logical rank must be in [0, 32)")
    _validate_tile(tile, config)
    owner = owner_rank(tile.expert_start, tile.channel_start, expert_tp, config=config)
    last_owner = owner_rank(
        tile.expert_start + tile.expert_count - 1,
        tile.channel_start + tile.channel_width - 1,
        expert_tp,
        config=config,
    )
    if owner != last_owner:
        raise ValueError("Packed tile crosses a rank ownership boundary")
    if owner != rank:
        return
    e = tile.expert_start - rank // expert_tp * local_experts
    j = tile.channel_start - rank % expert_tp * width
    for half, name in enumerate(("gate", "up")):
        yield "egu", (tile.layer, e, 0, half * width + j), tile.arrays["egu_" + name][None]
        yield (
            "egu_scale",
            (tile.layer, e, 0, half * width + j),
            tile.arrays["egu_" + name + "_scale"][None],
        )
    yield "edw", (tile.layer, e, j // 2, 0), tile.arrays["edw"][None]
    yield "edw_scale", (tile.layer, e, j // 16, 0), tile.arrays["edw_scale"][None]
    if include_params:
        params = np.zeros((1, tile.expert_count, 8, 128), np.float32)
        params[0, :, 0, :6] = tile.arrays["eparams"]
        yield "eparams", (tile.layer, e, 0, 0), params


@dataclass(frozen=True)
class PackedRecord:
    name: str
    array: np.ndarray
    metadata: Mapping | None = None


def tile_records(tile):
    """Expose one atom as seven records, including its compact calibration table."""
    metadata = {
        "layer": tile.layer,
        "expert_start": tile.expert_start,
        "channel_start": tile.channel_start,
        "channel_width": tile.channel_width,
    }
    for name in _TILE_NAMES:
        yield PackedRecord(name, tile.arrays[name], metadata)


def records_to_tile(records, *, config=DEFAULT_CONFIG):
    records = list(records)
    if len(records) != len(_TILE_NAMES) or {r.name for r in records} != set(_TILE_NAMES):
        raise ValueError("Expected exactly one complete packed expert tile")
    metadata = records[0].metadata
    if not isinstance(metadata, dict) or set(metadata) != {
        "layer",
        "expert_start",
        "channel_start",
        "channel_width",
    }:
        raise ValueError("Packed tile metadata is incomplete")
    if any(record.metadata != metadata for record in records):
        raise ValueError("Packed records describe different tiles")
    if any(type(value) is not int for value in metadata.values()):
        raise ValueError("Packed tile coordinates must be integers")
    tile = ExpertTile(**metadata, arrays={r.name: r.array for r in records})
    _validate_tile(tile, config)
    return tile


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_all(stream, data):
    view = memoryview(data).cast("B")
    while view:
        count = stream.write(view)
        if count is None or count <= 0:
            raise OSError("Packed stream made no write progress")
        view = view[count:]


def _read_into(stream, target):
    view = memoryview(target).cast("B")
    while view:
        count = stream.readinto(view)
        if count is None or count <= 0:
            raise EOFError("Truncated packed checkpoint")
        view = view[count:]


def _read_exact(stream, count):
    data = bytearray(count)
    _read_into(stream, data)
    return data


class RecordWriter:
    """Incremental sequential writer; several streams may share one source batch."""

    def __init__(self, stream, *, source_model, source_revision, metadata=None):
        if not source_model or not source_revision:
            raise ValueError("Source model and immutable revision are required")
        self.stream = stream
        self.header = {
            "format_version": FORMAT_VERSION,
            "source_model": source_model,
            "source_revision": source_revision,
            "metadata": dict(metadata or {}),
        }
        encoded = _json_bytes(self.header)
        if len(encoded) > MAX_HEADER_BYTES:
            raise ValueError("Packed header exceeds its size limit")
        self.digest = hashlib.sha256()
        self.count = self.payload_bytes = self.size = 0
        self.finished = False
        self._emit(MAGIC)
        self._emit(struct.pack("<I", len(encoded)))
        self._emit(encoded)

    def _emit(self, data):
        _write_all(self.stream, data)
        self.digest.update(data)
        self.size += len(data)

    def write(self, record):
        if self.finished:
            raise ValueError("Packed writer is already complete")
        array = np.asarray(record.array)
        dtype = _DTYPES.get(array.dtype.name)
        if dtype != array.dtype or dtype is None or not array.flags.c_contiguous:
            raise ValueError(
                "Packed records need a supported native dtype and C-contiguous payload"
            )
        if not isinstance(record.name, str) or not record.name:
            raise ValueError("Packed record needs a nonempty name")
        info = _json_bytes(
            {
                "name": record.name,
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "nbytes": array.nbytes,
                "metadata": dict(record.metadata or {}),
            }
        )
        if len(info) > MAX_HEADER_BYTES:
            raise ValueError("Packed record header exceeds its size limit")
        self._emit(b"R")
        self._emit(struct.pack("<I", len(info)))
        self._emit(info)
        payload_hash = hashlib.sha256()
        payload = memoryview(array.reshape(-1).view(np.uint8))
        for start in range(0, len(payload), IO_CHUNK_BYTES):
            part = payload[start : start + IO_CHUNK_BYTES]
            self._emit(part)
            payload_hash.update(part)
        self._emit(payload_hash.digest())
        self.count += 1
        self.payload_bytes += array.nbytes

    def finish(self):
        if self.finished:
            raise ValueError("Packed writer is already complete")
        stream_digest = self.digest.digest()
        footer = b"F" + struct.pack("<Q", self.count) + stream_digest
        _write_all(self.stream, footer)
        self.digest.update(footer)
        self.size += len(footer)
        self.finished = True
        return {
            **self.header,
            "records": self.count,
            "payload_bytes": self.payload_bytes,
            "size": self.size,
            "sha256": self.digest.hexdigest(),
            "content_sha256": stream_digest.hex(),
        }


def write_records(
    stream: BinaryIO,
    records: Iterable[PackedRecord],
    *,
    source_model: str,
    source_revision: str,
    metadata: Mapping | None = None,
):
    """Write contiguous records sequentially and return full-file checksum metadata.

    This function does not close or flush the caller's stream. Production callers
    must publish only after close, object validation and a completed manifest.
    """
    writer = RecordWriter(
        stream, source_model=source_model, source_revision=source_revision, metadata=metadata
    )
    for record in records:
        writer.write(record)
    return writer.finish()


def read_records(
    stream: BinaryIO,
    *,
    expected_model=None,
    expected_revision=None,
    max_record_bytes=256 << 20,
    expected_metadata=None,
    expected_sha256=None,
    report=None,
):
    """Yield verified records; exhausting the iterator verifies file completion.

    Consumers must exhaust this iterator before publishing or accepting a file.
    Each allocation is bounded by max_record_bytes; callers control how many
    yielded arrays remain live. No filesystem seek or whole-file buffering occurs.

    An optional, initially empty ``report`` dict receives verified counts and the
    whole-file SHA256 only after the footer, EOF, and optional expected_sha256
    comparison pass. Deriving that digest continues the existing stream hash
    with the 41-byte footer; payload bytes need no third hash or second read.
    """
    if report is not None and (not isinstance(report, dict) or report):
        raise ValueError("Completion report must be an initially empty dict")
    if max_record_bytes < 1:
        raise ValueError("Record limit must be positive")
    digest = hashlib.sha256()
    size = payload_bytes = 0

    def take(count):
        nonlocal size
        data = _read_exact(stream, count)
        digest.update(data)
        size += len(data)
        return data

    def header():
        size = struct.unpack("<I", take(4))[0]
        if size > MAX_HEADER_BYTES:
            raise ValueError("Packed header exceeds its size limit")
        result = json.loads(take(size))
        if not isinstance(result, dict):
            raise ValueError("Packed header must be an object")
        return result

    if bytes(take(len(MAGIC))) != MAGIC:
        raise ValueError("Unrecognized packed checkpoint format")
    info = header()
    if info.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported packed format version")
    if not info.get("source_model") or not info.get("source_revision"):
        raise ValueError("Packed checkpoint omits source identity")
    for key, expected in (("source_model", expected_model), ("source_revision", expected_revision)):
        if expected is not None and info[key] != expected:
            raise ValueError(f"Packed checkpoint {key} mismatch")
    if expected_metadata is not None and info.get("metadata") != dict(expected_metadata):
        raise ValueError("Packed checkpoint source/config metadata mismatch")
    count = 0
    while True:
        tag = bytes(_read_exact(stream, 1))
        size += 1
        if tag == b"F":
            count_bytes = _read_exact(stream, 8)
            expected_count = struct.unpack("<Q", count_bytes)[0]
            expected_digest = bytes(_read_exact(stream, 32))
            size += 40
            if count != expected_count or digest.digest() != expected_digest:
                raise ValueError("Packed checkpoint footer checksum/count mismatch")
            if stream.read(1):
                raise ValueError("Unexpected bytes after packed completion footer")
            content_sha256 = digest.hexdigest()
            digest.update(tag + count_bytes + expected_digest)
            file_sha256 = digest.hexdigest()
            if expected_sha256 is not None and file_sha256 != expected_sha256:
                raise ValueError("Packed checkpoint whole-file checksum mismatch")
            if report is not None:
                report.update(
                    {
                        "sha256": file_sha256,
                        "content_sha256": content_sha256,
                        "size": size,
                        "records": count,
                        "payload_bytes": payload_bytes,
                    }
                )
            return
        if tag != b"R":
            raise ValueError("Invalid packed record tag")
        digest.update(tag)
        info = header()
        dtype = _DTYPES.get(info.get("dtype"))
        shape = info.get("shape")
        if (
            dtype is None
            or not isinstance(shape, list)
            or any(type(d) is not int or d < 0 for d in shape)
            or len(shape) > 16
            or not isinstance(info.get("name"), str)
            or not info["name"]
            or not isinstance(info.get("metadata", {}), dict)
        ):
            raise ValueError("Invalid packed record metadata")
        nbytes = math.prod(shape) * dtype.itemsize
        if type(info.get("nbytes")) is not int or nbytes != info["nbytes"]:
            raise ValueError("Packed record extent does not match its shape")
        if nbytes > max_record_bytes:
            raise ValueError("Packed record exceeds its bounded read limit")
        array = np.empty(tuple(shape), dtype)
        payload = memoryview(array.reshape(-1).view(np.uint8))
        payload_hash = hashlib.sha256()
        for start in range(0, nbytes, IO_CHUNK_BYTES):
            part = payload[start : start + IO_CHUNK_BYTES]
            _read_into(stream, part)
            digest.update(part)
            payload_hash.update(part)
            size += len(part)
        if payload_hash.digest() != bytes(take(32)):
            raise ValueError("Packed record payload checksum mismatch")
        count += 1
        payload_bytes += nbytes
        yield PackedRecord(info["name"], array, info.get("metadata", {}))
