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
the per-rank layout the kernels consume (no leading rank axis). They are
independent of JAX so a vLLM worker can upload them through torch.
"""

from __future__ import annotations

import dataclasses
import json
import mmap
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ml_dtypes
import numpy as np

from ..common.constants import EXPERT_FAMILIES as EXPERT_FAMILIES
from ..config import active as _config
from .decode_megakernel import ATTENTION_DP_WEIGHTS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

CHUNK = 64 << 20
_LEGACY_KDA_PROJ_WIDTH = 1792
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


@dataclasses.dataclass(frozen=True)
class ExpertPartSpec:
    """Descriptor for one sharded expert binary part in layer order."""

    family: str
    index: int
    count: int

    @property
    def label(self) -> str:
        """Return canonical ``family@index/count`` part key."""
        return f"{self.family}@{self.index}/{self.count}"


def _dtype(name: str | np.dtype[Any]) -> np.dtype[Any]:
    """Resolve a checkpoint dtype name or NumPy dtype into a NumPy dtype."""
    if name == "bfloat16":
        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(name)


MIN_STREAM_BYTES = 256 << 20


def _default_streams() -> int:
    """Return configured concurrent stream count per contiguous file range."""
    return _config().read_streams


def _default_inflight_bytes() -> int:
    """Return configured host staging byte budget while streaming shards."""
    return int(_config().load_inflight_gb * (1 << 30))


def _read_range(path: Path, offset: int, view: memoryview) -> None:
    """Read a byte range from path at offset into view."""
    with path.open("rb", buffering=0) as handle:
        handle.seek(offset)
        position = 0
        while position < len(view):
            count = handle.readinto(view[position : position + CHUNK])
            if not count:
                raise EOFError(f"{path}: short read at {offset + position}")
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
    step = -(-step // CHUNK) * CHUNK
    parts = [(start, view[start : start + step]) for start in range(0, len(view), step)]
    with ThreadPoolExecutor(len(parts)) as pool:
        list(pool.map(lambda part: _read_range(path, offset + part[0], part[1]), parts))
    return out


def read_array(
    path: Path, offset: int, shape: Sequence[int], dtype: str | np.dtype[Any]
) -> np.ndarray:
    """Allocate an array of shape and dtype and fill it from path at offset."""
    resolved = _dtype(dtype)
    return read_into(path, offset, np.empty(tuple(shape), resolved))


class RankDir:
    """One rank's pre-shard directory and its index."""

    def __init__(self, presharded: Path | str, rank: int, layers: int = 93) -> None:
        """Initialize a RankDir reader and validate its index metadata."""
        self.path = Path(presharded) / f"rank{rank:02d}"
        self.index: dict[str, Any] = json.loads((self.path / "index.json").read_text())
        if self.index.get("rank") != rank or self.index.get("layers") != layers:
            raise ValueError(f"unexpected pre-shard index in {self.path}")
        self.families: dict[str, dict[str, Any]] = self.index["families"]

    def spec(self, name: str) -> dict[str, Any]:
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
        """Yield ``(label, nbytes, reader)`` per expert shard file in layer order.

        ``(label, nbytes, reader)`` per file of an expert family, in layer
        order; ``label`` is ``name@index/count`` (see ``join_parts``).
        """
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
                ExpertPartSpec(name, index, count).label,
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
        """Yield an expert family regrouped from EP8/TP4 shards into EP16/TP2 parts.

        Reads experts ``part`` of ``parts`` equal slices of the EP8 group's 112,
        over the TP4 quarters in ``quarter_dirs`` (two: half the intermediate,
        EP16/TP2) -- gate quarters then up quarters along N, down and its
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
                """Memory-map and join EP8 quarter files into an EP16/TP2 slice."""
                quarters = []
                slice_shape = (hi - lo, *shape[1:])
                elem_count = int(np.prod(slice_shape))
                for rank_dir in quarter_dirs:
                    with (
                        (rank_dir.path / filename).open("rb") as handle,
                        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm,
                    ):
                        array = np.frombuffer(
                            mm, dtype=dtype, count=elem_count
                        ).reshape(slice_shape)
                        quarters.append(
                            np.array(array[:, part * experts : (part + 1) * experts])
                        )
                        del array
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

            yield ExpertPartSpec(name, index, count).label, (hi - lo) * nbytes, reader


def _read_dense_families(rank_dir: RankDir) -> dict[str, np.ndarray]:
    """Read every dense family of rank_dir in one sequential pass over dense.bin."""
    spec_of = {name: rank_dir.families[name] for name in rank_dir.dense_names()}
    order = sorted(spec_of, key=lambda n: spec_of[n]["offset"])
    end = max(
        spec_of[n]["offset"]
        + int(np.prod(spec_of[n]["shape"])) * _dtype(spec_of[n]["dtype"]).itemsize
        for n in order
    )
    raw = read_into(rank_dir.path / "dense.bin", 0, np.empty(end, np.uint8))
    result: dict[str, np.ndarray] = {}
    for name in order:
        spec = spec_of[name]
        dtype = _dtype(spec["dtype"])
        count = int(np.prod(spec["shape"]))
        result[name] = (
            raw[spec["offset"] : spec["offset"] + count * dtype.itemsize]
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
        families["k_gate"] = np.ascontiguousarray(combined[..., 1152:1280])
        families["k_projection"] = np.ascontiguousarray(
            np.concatenate((combined[..., :1152], combined[..., 1280:]), axis=-1)
        )
    return families


# Host bytes one rank may hold in flight while loading. Eight ranks share a
# host; larger budgets can push a NUMA node into direct reclaim and evict the
# page cache that keeps restarts warm.
def _budgeted(
    jobs: Iterable[tuple[str, int, Callable[[], Any]]],
    *,
    budget: int | None = None,
    workers: int = 8,
) -> Iterator[tuple[str, Any]]:
    """Run ``(key, nbytes, fn)`` jobs on threads within ``budget`` bytes in flight."""
    resolved_budget = _default_inflight_bytes() if budget is None else budget
    job_list = list(jobs)
    with ThreadPoolExecutor(workers) as pool:
        # (key, nbytes, future) in submission order
        pending: list[tuple[str, int, Future[Any]]] = []
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
    return max(
        s["offset"] + int(np.prod(s["shape"])) * _dtype(s["dtype"]).itemsize
        for s in specs
    )


def _family_bytes(spec: dict[str, Any]) -> int:
    """Return byte size of one weight family from its index spec."""
    return int(np.prod(spec["shape"])) * _dtype(spec["dtype"]).itemsize


def iter_decode_rank(
    presharded: Path | str,
    rank: int,
    *,
    layers: int = 93,
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
    rank_dir = RankDir(presharded, rank, layers)
    expert_order = sorted(
        EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n))
    )
    jobs: list[tuple[str, int, Callable[[], Any]]] = [
        job for name in expert_order for job in rank_dir.expert_parts(name)
    ]
    jobs.append(
        ("dense", _dense_bytes(rank_dir), lambda: _read_dense_families(rank_dir))
    )
    for key, value in _budgeted(jobs):
        if key == "dense":
            families = split_kda_projection(value)
            if attention_dp > 1:
                families.update(
                    attention_dp_families(
                        families, presharded, rank, layers, attention_dp
                    )
                )
            for name, family in families.items():
                yield "decode", name, family
        else:
            yield "decode", key, value


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
    group = 32 // dp
    shards = []
    for shard in range(dp):
        source = rank % group + shard * group
        if source == rank:
            shards.append(families)
            continue
        peer = RankDir(presharded, source, layers)
        names = [name for name in ATTENTION_DP_WEIGHTS if name in peer.families]
        shards.append(
            split_kda_projection({name: peer.read_dense(name) for name in names})
        )
    held: dict[str, np.ndarray] = {}
    for name, (_kind, axis) in ATTENTION_DP_WEIGHTS.items():
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
    _pair, part = divmod(rank, 2)
    half_rows = own.shape[1] // 2
    rows = slice(part * half_rows, (part + 1) * half_rows)
    peer = peer_dir.read_dense(name)
    halves = (own, peer) if rank % 2 == 0 else (peer, own)
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
    layers: int = 93,
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield the prefill stack's weights for ``rank``.

    Yields ``(group, name, array)`` with ``group`` in io/common/kda/mla/moe
    (per-rank arrays, without the rank axis); expert families first, per
    file, as EP16/TP2 parts regrouped from the EP8 group's shards.
    """
    rank_dir = RankDir(presharded, rank, layers)
    peer_dir = RankDir(presharded, rank ^ 1, layers)
    cache = Path(projection_cache)
    layout = json.loads((cache / "layout.json").read_text())
    if layout.get("layers") != layers:
        raise ValueError(f"unexpected projection cache {cache}")

    def projection(name: str) -> np.ndarray:
        """Read a cached prefill projection family array from disk."""
        spec = layout["families"][name]
        return read_array(cache / spec["file"], 0, spec["shape"], spec["dtype"])

    parity = rank % 2

    def cached(name: str) -> tuple[int, Callable[[], np.ndarray]]:
        """Return byte size and reader closure for a cached prefill family."""
        spec = layout["families"][name]
        return (
            int(np.prod(spec["shape"])) * _dtype(spec["dtype"]).itemsize,
            lambda: projection(name),
        )

    # EP16/TP2: 56 experts, half the intermediate (TP2 within the chip),
    # regrouped from the EP8 group's quarters (same host).
    group = 4 * (rank // 4)
    quarters, part, parts = (
        (2 * (rank % 2), 2 * (rank % 2) + 1),
        (rank // 2) % 2,
        2,
    )
    quarter_dirs = [RankDir(presharded, group + q, layers) for q in quarters]
    jobs: list[tuple[str, int, Callable[[], Any]]] = [
        job
        for name in sorted(
            EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n))
        )
        for job in rank_dir.regrouped_expert_parts(name, quarter_dirs, part, parts)
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
        nbytes, fn = cached(source)
        jobs.append((name, nbytes, fn))
    for key, value in _budgeted(jobs):
        if key != "dense":
            yield "moe", key, value
            continue
        for group_name, name, family in value:
            yield group_name, name, family
