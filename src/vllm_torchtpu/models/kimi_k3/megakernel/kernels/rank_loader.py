# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/rank_loader.py
"""Streaming per-rank readers for the TP32 pre-shards (decode and prefill).

Both megakernels read the same pre-sharded rank directories
(``<presharded>/rankNN/{index.json, dense.bin, expert_*.bin}``); the prefill
additionally reads the SP projection cache
(``latent_down.bin``, ``latent_up.bin``, ``shared_{gu,down}_{0,1}.bin``).
Seeks and mmap page faults are slow on network storage, so reads are mostly
large sequential ``readinto`` calls on separate handles, straight into the
numpy array the caller keeps (the regrouped EP16 expert parts slice
memory-mapped shards instead).

The iterators yield ``(group, name, array)`` host arrays for one rank, in
the per-rank layout the kernels consume (no leading rank axis), without
importing ``torch``, kernel modules, or ``common.pool_alias``.
"""

from __future__ import annotations

import importlib
import json
import mmap
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, TypedDict, cast

import ml_dtypes
import numpy as np

from ..common.constants import (
    EXPERT_FAMILIES,
    GIB_BYTES,
    HEAD_DIM,
    MIB_BYTES,
    NUM_LAYERS,
    TP_SIZE,
)
from ..common.paths import require_local_path
from ..config import active as _config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path

_READ_CHUNK_BYTES = 64 * MIB_BYTES
_SP_PAIR_SIZE = 2
_SP_MIN_NDIM = 2
# Legacy combined k_projection stores 9 * HEAD_DIM (1152) Q/K/V columns,
# HEAD_DIM (128) k_gate columns, and 4 * HEAD_DIM (512) trailing columns (1792 total).
LEGACY_KDA_GATE_START = 9 * HEAD_DIM
_LEGACY_KDA_GATE_END = LEGACY_KDA_GATE_START + HEAD_DIM
_LEGACY_KDA_PROJ_WIDTH = _LEGACY_KDA_GATE_END + 4 * HEAD_DIM
PREFILL_COMMON = (
    "attn_res",
    "attn_norm",
    "mlp_res",
    "ffn_norm",
    "dense_gu",
    "dense_down",
)
PREFILL_IO = ("embedding", "lm_head", "output_res", "final_norm")
PREFILL_KDA = (
    "k_gate",
    "k_fb",
    "k_projection",
    "k_conv",
    "k_a_log",
    "k_dt",
    "k_norm",
    "k_o",
)
PREFILL_MLA = (
    "m_qa",
    "m_qb",
    "m_ka",
    "m_kb",
    "m_gate",
    "m_qnorm",
    "m_knorm",
    "m_o",
)
PREFILL_MOE_DENSE = ("router", "router_bias", "latent_norm")


def _part_label(family: str, index: int, count: int) -> str:
    """Return canonical ``family@index/count`` part key."""
    return f"{family}@{index}/{count}"


class FamilySpec(TypedDict, total=False):
    """Per-family metadata entry inside a rank pre-shard ``index.json``."""

    file: str
    offset: int
    files: list[str]
    layer_ranges: list[list[int]]
    shape: list[int]
    dtype: str


class RankIndex(TypedDict, total=False):
    """Top-level ``index.json`` manifest for one rank pre-shard directory."""

    format: int
    rank: int
    layers: int
    families: dict[str, FamilySpec]


def _validate_rank(rank: int) -> int:
    """Validate a TP rank index in ``[0, TP_SIZE)``."""
    if isinstance(rank, bool) or not 0 <= rank < TP_SIZE:
        msg = f"rank must be in [0, {TP_SIZE}), got {rank!r}"
        raise ValueError(msg)
    return rank


def _validate_attention_dp(dp: int) -> int:
    """Validate an Attention-DP degree dividing ``TP_SIZE``."""
    if isinstance(dp, bool) or dp <= 0 or TP_SIZE % dp != 0:
        msg = f"attention_dp must be a positive divisor of {TP_SIZE}, got {dp!r}"
        raise ValueError(msg)
    return dp


def _attention_dp_weights() -> dict[str, tuple[str, int | None]]:
    """Load the Attention-DP weight sharding table lazily from decode_megakernel."""
    module = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.kernels.decode_megakernel"
    )
    return cast(
        "dict[str, tuple[str, int | None]]",
        module.ATTENTION_DP_WEIGHTS,
    )


def _dtype(name: str | np.dtype[np.generic] | type[np.generic]) -> np.dtype[np.generic]:
    """Resolve a checkpoint dtype name or NumPy dtype into a NumPy dtype."""
    if name == "bfloat16":
        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(name)


MIN_STREAM_BYTES = 256 * MIB_BYTES


def _default_streams() -> int:
    """Return configured concurrent stream count per contiguous file range."""
    return _config().read_streams


# Host bytes one rank may hold in flight while loading. Eight ranks share a
# host; larger budgets can push a NUMA node into direct reclaim and evict the
# page cache that keeps restarts warm.
def _default_inflight_bytes() -> int:
    """Return configured host staging byte budget while streaming shards."""
    return int(_config().load_inflight_gb * GIB_BYTES)


def _read_range(path: Path, offset: int, view: memoryview) -> None:
    """Read a byte range from path at offset into view."""
    with path.open("rb", buffering=0) as handle:
        handle.seek(offset)
        position = 0
        while position < len(view):
            count = handle.readinto(view[position : position + _READ_CHUNK_BYTES])
            if not count:
                msg = f"{path}: short read at {offset + position}"
                raise EOFError(msg)
            position += count


def read_into(
    path: Path, offset: int, out: np.ndarray, streams: int | None = None
) -> np.ndarray:
    """Fill ``out`` (C-contiguous) from ``path`` at ``offset``.

    Uses ``streams`` handles (default ``KernelConfig.read_streams``), each
    reading its own contiguous part sequentially. Network storage throughput
    scales with handles, not with threads sharing one.
    """
    view = memoryview(out.reshape(-1).view(np.uint8))
    streams = max(
        1,
        min(
            streams if streams is not None else _default_streams(),
            len(view) // MIN_STREAM_BYTES,
        ),
    )
    if streams == 1:
        _read_range(path, offset, view)
        return out
    step = -(-len(view) // streams)
    step = -(-step // _READ_CHUNK_BYTES) * _READ_CHUNK_BYTES
    parts = [(start, view[start : start + step]) for start in range(0, len(view), step)]
    with ThreadPoolExecutor(len(parts)) as pool:
        list(pool.map(lambda part: _read_range(path, offset + part[0], part[1]), parts))
    return out


def read_array(
    path: Path, offset: int, shape: Sequence[int], dtype: str | np.dtype[np.generic]
) -> np.ndarray:
    """Allocate an array of shape and dtype and fill it from path at offset."""
    resolved = _dtype(dtype)
    return read_into(path, offset, np.empty(tuple(shape), resolved))


class RankDir:
    """One rank's pre-shard directory and its index."""

    def __init__(
        self, presharded: Path | str, rank: int, layers: int = NUM_LAYERS
    ) -> None:
        """Initialize a RankDir reader and validate its index metadata."""
        _validate_rank(rank)
        root = require_local_path(presharded, label="presharded")
        self.path = root / f"rank{rank:02d}"
        self.index: RankIndex = json.loads(
            (self.path / "index.json").read_text(encoding="utf-8")
        )
        if self.index.get("rank") != rank or self.index.get("layers") != layers:
            msg = f"unexpected pre-shard index in {self.path}"
            raise ValueError(msg)
        self.families: dict[str, FamilySpec] = self.index["families"]
        if not self.dense_names():
            msg = f"pre-shard index in {self.path} contains no dense families"
            raise ValueError(msg)

    def spec(self, name: str) -> FamilySpec:
        """Return the index specification entry for family name."""
        return self.families[name]

    def dense_names(self) -> list[str]:
        """Return all family names stored in dense.bin."""
        return [
            name
            for name, spec in self.families.items()
            if spec.get("file") == "dense.bin"
        ]

    def read_dense(self, name: str) -> np.ndarray:
        """Read one dense weight family from dense.bin."""
        spec = self.families[name]
        return read_array(
            self.path / "dense.bin", spec["offset"], spec["shape"], spec["dtype"]
        )

    def expert_parts(
        self, name: str
    ) -> Iterator[tuple[str, int, Callable[[], np.ndarray]]]:
        """Yield ``(label, nbytes, reader)`` per expert shard file in layer order."""
        spec = self.families[name]
        shape = tuple(spec["shape"])
        dtype = _dtype(spec["dtype"])
        per_layer = int(np.prod(shape[1:])) * dtype.itemsize
        count = len(spec["files"])
        for index, (filename, (lo, hi)) in enumerate(
            zip(spec["files"], spec["layer_ranges"], strict=True)
        ):

            def reader(
                filename: str = filename, lo: int = lo, hi: int = hi
            ) -> np.ndarray:
                """Read the layer slice stored in one expert binary shard file."""
                return read_into(
                    self.path / filename,
                    0,
                    np.empty((hi - lo, *shape[1:]), dtype),
                )

            yield (
                _part_label(name, index, count),
                (hi - lo) * per_layer,
                reader,
            )

    def regrouped_expert_parts(
        self,
        name: str,
        quarter_dirs: Sequence[RankDir],
        part: int,
        parts: int,
    ) -> Iterator[tuple[str, int, Callable[[], np.ndarray]]]:
        """Yield an expert family regrouped from EP8 x TP4 shards into EP16 x TP2 parts.

        Reads experts ``part`` of ``parts`` equal slices of the EP8 group's 112,
        over the TP4 quarters in ``quarter_dirs`` (two: half the intermediate,
        EP16 x TP2) -- gate quarters then up quarters along N, down and its
        scales along K.
        """
        spec = self.families[name]
        shape = tuple(spec["shape"])  # [layers, 112, ...] per quarter
        dtype = _dtype(spec["dtype"])
        experts = shape[1] // parts
        count = len(spec["files"])
        nbytes = (
            int(np.prod((experts, *shape[2:]))) * dtype.itemsize * len(quarter_dirs)
        )
        for index, (filename, (lo, hi)) in enumerate(
            zip(spec["files"], spec["layer_ranges"], strict=True)
        ):

            def reader(
                filename: str = filename, lo: int = lo, hi: int = hi
            ) -> np.ndarray:
                """Memory-map and join EP8 quarter files into an EP16 x TP2 slice."""
                quarters = []
                slice_shape = (hi - lo, *shape[1:])
                elem_count = int(np.prod(slice_shape))
                for rank_dir in quarter_dirs:
                    with (
                        (rank_dir.path / filename).open("rb") as handle,
                        mmap.mmap(
                            handle.fileno(), 0, access=mmap.ACCESS_READ
                        ) as memory_map,
                    ):
                        quarters.append(
                            np.frombuffer(memory_map, dtype=dtype, count=elem_count)
                            .reshape(slice_shape)[
                                :, part * experts : (part + 1) * experts
                            ]
                            .copy()
                        )
                if name in ("expert_gu", "expert_gus"):
                    half = shape[-1] // 2
                    return np.concatenate(
                        [
                            *(q[..., :half] for q in quarters),
                            *(q[..., half:] for q in quarters),
                        ],
                        axis=-1,
                    )
                return np.concatenate(quarters, axis=2)

            yield _part_label(name, index, count), (hi - lo) * nbytes, reader


def _read_dense_families(rank_dir: RankDir) -> dict[str, np.ndarray]:
    """Read every dense family of rank_dir in one sequential pass over dense.bin."""
    spec_of = {name: rank_dir.families[name] for name in rank_dir.dense_names()}
    order = sorted(spec_of, key=lambda n: spec_of[n]["offset"])
    end = _dense_bytes(rank_dir)
    raw = read_into(rank_dir.path / "dense.bin", 0, np.empty(end, np.uint8))
    result: dict[str, np.ndarray] = {}
    for name in order:
        spec = spec_of[name]
        dtype = _dtype(spec["dtype"])
        result[name] = (
            raw[spec["offset"] : spec["offset"] + _family_bytes(spec)]
            .view(dtype)
            .reshape(spec["shape"])
        )
    return result


def split_kda_projection(
    families: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Split ``k_gate`` out of an older pre-shard's ``k_projection [.., 1792]``."""
    combined = families.get("k_projection")
    if (
        combined is not None
        and combined.shape[-1] == _LEGACY_KDA_PROJ_WIDTH
        and "k_gate" not in families
    ):
        families["k_gate"] = np.ascontiguousarray(
            combined[..., LEGACY_KDA_GATE_START:_LEGACY_KDA_GATE_END]
        )
        families["k_projection"] = np.ascontiguousarray(
            np.concatenate(
                (
                    combined[..., :LEGACY_KDA_GATE_START],
                    combined[..., _LEGACY_KDA_GATE_END:],
                ),
                axis=-1,
            )
        )
    return families


def _budgeted[T](
    jobs: Iterable[tuple[str, int, Callable[[], T]]],
    *,
    budget: int | None = None,
    workers: int = 8,
) -> Iterator[tuple[str, T]]:
    """Run ``(key, nbytes, fn)`` jobs on threads within ``budget`` bytes in flight.

    One job always may run. Yields ``(key, result)`` in job order; a job's bytes
    are released when the caller resumes after it.
    """
    resolved_budget = _default_inflight_bytes() if budget is None else budget
    job_list = list(jobs)
    with ThreadPoolExecutor(workers) as pool:
        # (key, nbytes, future) in submission order
        pending: list[tuple[str, int, Future[T]]] = []
        inflight = 0
        next_job = 0

        def fill() -> None:
            """Submit queued jobs while total in-flight bytes stay within budget."""
            nonlocal inflight, next_job
            while next_job < len(job_list):
                key, nbytes, fn = job_list[next_job]
                if pending and inflight + nbytes > resolved_budget:
                    return
                pending.append((key, nbytes, pool.submit(fn)))
                inflight += nbytes
                next_job += 1

        fill()
        while pending:
            key, nbytes, future = pending.pop(0)
            yield key, future.result()
            inflight -= nbytes
            fill()


def _dense_bytes(rank_dir: RankDir) -> int:
    """Return total byte span of dense.bin for rank_dir."""
    specs = [rank_dir.families[n] for n in rank_dir.dense_names()]
    return max(spec["offset"] + _family_bytes(spec) for spec in specs)


def _family_bytes(spec: FamilySpec) -> int:
    """Return byte size of one weight family from its index spec."""
    return int(np.prod(spec["shape"])) * _dtype(spec["dtype"]).itemsize


def _single_entry_job(
    group: str, name: str, reader: Callable[[], np.ndarray]
) -> Callable[[], list[tuple[str, str, np.ndarray]]]:
    """Wrap an array reader into a single-item ``(group, name, array)`` job."""

    def run() -> list[tuple[str, str, np.ndarray]]:
        """Read one array and package it with its group and family name."""
        return [(group, name, reader())]

    return run


def _decode_dense_entries(rank_dir: RankDir) -> list[tuple[str, str, np.ndarray]]:
    """Read and split decode dense families for one rank."""
    families = split_kda_projection(_read_dense_families(rank_dir))
    return [("decode", name, family) for name, family in families.items()]


def iter_decode_rank(
    presharded: Path | str,
    rank: int,
    *,
    layers: int = NUM_LAYERS,
    attention_dp: int = 1,
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield the decode megakernel's weights for ``rank`` (MXFP4 experts as stored).

    Yields ``("decode", name, array)``: the four packed expert families
    (largest first, one array per file, named ``family@index/count``, layer
    ranges in order -- the caller joins them),
    then every dense family (``k_projection`` split into ``k_gate`` and the
    rest), with at most ``KernelConfig.load_inflight_gb`` of host buffers in flight.
    With ``attention_dp`` > 1 the head-sharded attention families hold the
    rank's held TP32 shards (see :func:`attention_dp_families`).
    """
    _validate_attention_dp(attention_dp)
    rank_dir = RankDir(presharded, rank, layers)
    expert_order = sorted(
        EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n))
    )
    jobs: list[tuple[str, int, Callable[[], list[tuple[str, str, np.ndarray]]]]] = [
        (part_key, nbytes, _single_entry_job("decode", part_key, reader))
        for name in expert_order
        for part_key, nbytes, reader in rank_dir.expert_parts(name)
    ]
    jobs.append(
        (
            "dense",
            _dense_bytes(rank_dir),
            lambda: _decode_dense_entries(rank_dir),
        )
    )
    for key, entries in _budgeted(jobs):
        if key == "dense" and attention_dp > 1:
            families = {name: array for _group, name, array in entries}
            entries.clear()
            families.update(
                attention_dp_families(families, presharded, rank, layers, attention_dp)
            )
            for name, array in families.items():
                yield "decode", name, array
        else:
            yield from entries


def attention_dp_families(
    families: dict[str, np.ndarray],
    presharded: Path | str,
    rank: int,
    layers: int,
    dp: int,
) -> dict[str, np.ndarray]:
    """Assemble Attention-DP head-sharded weight families for ``rank``.

    Concatenates or stacks TP32 shards ``rank % (32 / dp) + k * (32 / dp)`` in
    order ``k`` per ``decode_megakernel.ATTENTION_DP_WEIGHTS``, reading peer
    shards from their ranks' pre-shards.
    """
    _validate_rank(rank)
    _validate_attention_dp(dp)
    attention_dp_weights = _attention_dp_weights()
    group = TP_SIZE // dp
    shards = []
    for shard in range(dp):
        source = rank % group + shard * group
        if source == rank:
            shards.append(families)
            continue
        peer = RankDir(presharded, source, layers)
        names = [name for name in attention_dp_weights if name in peer.families]
        shards.append(
            split_kda_projection({name: peer.read_dense(name) for name in names})
        )
    held: dict[str, np.ndarray] = {}
    for name, (_kind, axis) in attention_dp_weights.items():
        parts = [shard[name] for shard in shards]
        if axis is None:  # "stack"
            held[name] = np.stack(parts, axis=1)
        else:
            held[name] = np.concatenate(parts, axis=1 + axis)
    return held


def _sequence_parallel_output(
    own: np.ndarray, peer_dir: RankDir, name: str, rank: int
) -> np.ndarray:
    """Build prefill TP32 output projection from the chip pair's decode shards."""
    if own.ndim < _SP_MIN_NDIM or own.shape[1] % _SP_PAIR_SIZE != 0:
        msg = (
            f"sequence-parallel output {name!r} must have at least 2 dims "
            f"and even row dimension, got shape {own.shape}"
        )
        raise ValueError(msg)
    _pair, part = divmod(rank, _SP_PAIR_SIZE)
    half_rows = own.shape[1] // _SP_PAIR_SIZE
    rows = slice(part * half_rows, (part + 1) * half_rows)
    peer = peer_dir.read_dense(name)
    halves = (own, peer) if rank % _SP_PAIR_SIZE == 0 else (peer, own)
    return np.ascontiguousarray(np.concatenate([h[:, rows] for h in halves], axis=-1))


def _prefill_dense_entries(
    rank_dir: RankDir, peer_dir: RankDir, rank: int
) -> list[tuple[str, str, np.ndarray]]:
    """Read and organize prefill dense families across io, common, kda, mla, and moe."""
    families = split_kda_projection(_read_dense_families(rank_dir))
    sp_outputs = {"k_o": "kda", "m_o": "mla"}
    entries: list[tuple[str, str, np.ndarray]] = []
    groups = (
        ("io", PREFILL_IO),
        ("common", PREFILL_COMMON),
        ("kda", PREFILL_KDA),
        ("mla", PREFILL_MLA),
        ("moe", PREFILL_MOE_DENSE),
    )
    for group_name, names in groups:
        for name in names:
            family = families[name]
            if sp_outputs.get(name) == group_name:
                family = _sequence_parallel_output(family, peer_dir, name, rank)
            entries.append((group_name, name, family))
    return entries


def iter_prefill_rank(
    presharded: Path | str,
    projection_cache: Path | str,
    rank: int,
    *,
    layers: int = NUM_LAYERS,
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield the prefill stack's weights for ``rank``.

    Yields ``(group, name, array)`` with ``group`` in io/common/kda/mla/moe
    (per-rank arrays, without the rank axis); expert families first, per
    file, as EP16 x TP2 parts regrouped from the EP8 group's shards.
    """
    cache = require_local_path(projection_cache, label="projection_cache")
    rank_dir = RankDir(presharded, rank, layers)
    peer_dir = RankDir(presharded, rank ^ 1, layers)
    layout = json.loads((cache / "layout.json").read_text(encoding="utf-8"))
    if layout.get("layers") != layers:
        msg = f"unexpected projection cache {cache}"
        raise ValueError(msg)

    def projection(name: str) -> np.ndarray:
        """Read a cached prefill projection family array from disk."""
        spec = layout["families"][name]
        return read_array(cache / spec["file"], 0, spec["shape"], spec["dtype"])

    parity = rank % 2

    def cached(
        target_name: str, source: str
    ) -> tuple[int, Callable[[], list[tuple[str, str, np.ndarray]]]]:
        """Return byte size and job closure for a cached prefill family."""
        spec: FamilySpec = layout["families"][source]
        return (
            _family_bytes(spec),
            _single_entry_job("moe", target_name, lambda: projection(source)),
        )

    # EP16 x TP2: 56 experts, half the intermediate (TP2 within the chip),
    # regrouped from the EP8 group's quarters (same host).
    group = 4 * (rank // 4)
    quarters, part, parts = (
        (2 * (rank % 2), 2 * (rank % 2) + 1),
        (rank // 2) % 2,
        2,
    )
    quarter_dirs = [RankDir(presharded, group + q, layers) for q in quarters]
    jobs: list[tuple[str, int, Callable[[], list[tuple[str, str, np.ndarray]]]]] = [
        (part_key, nbytes, _single_entry_job("moe", part_key, reader))
        for name in sorted(
            EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n))
        )
        for part_key, nbytes, reader in rank_dir.regrouped_expert_parts(
            name, quarter_dirs, part, parts
        )
    ]
    dense_bytes = _dense_bytes(rank_dir) + sum(
        _family_bytes(peer_dir.spec(name))
        for name in ("k_o", "m_o")
        if name in peer_dir.families
    )
    jobs.append(
        (
            "dense",
            dense_bytes,
            lambda: _prefill_dense_entries(rank_dir, peer_dir, rank),
        )
    )
    for name, source in (
        ("latent_down", "latent_down"),
        ("latent_up", "latent_up"),
        ("shared_gu", f"shared_gu_{parity}"),
        ("shared_down", f"shared_down_{parity}"),
    ):
        nbytes, fn = cached(name, source)
        jobs.append((name, nbytes, fn))
    for _key, entries in _budgeted(jobs):
        yield from entries
