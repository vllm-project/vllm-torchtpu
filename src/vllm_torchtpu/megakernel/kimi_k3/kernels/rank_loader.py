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

import functools
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config

CHUNK = 64 << 20
EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
PREFILL_COMMON = ("attn_res", "attn_norm", "mlp_res", "ffn_norm", "dense_gu", "dense_down")
PREFILL_IO = ("embedding", "lm_head", "output_res", "final_norm")
PREFILL_KDA = ("k_gate", "k_fb", "k_projection", "k_conv", "k_a_log", "k_dt", "k_norm", "k_o")
PREFILL_MLA = ("m_qa", "m_qb", "m_ka", "m_kb", "m_gate", "m_qnorm", "m_knorm", "m_o")
PREFILL_MOE_DENSE = ("router", "router_bias", "latent_norm")


def _dtype(name: str) -> np.dtype:
    if name == "bfloat16":
        import ml_dtypes
        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(name)


STREAMS = _config().read_streams  # handles per contiguous range
MIN_STREAM_BYTES = 256 << 20


def _read_range(path: Path, offset: int, view: memoryview) -> None:
    with open(path, "rb", buffering=0) as handle:
        handle.seek(offset)
        position = 0
        while position < len(view):
            count = handle.readinto(view[position:position + CHUNK])
            if not count:
                raise EOFError(f"{path}: short read at {offset + position}")
            position += count


def read_into(path: Path, offset: int, out: np.ndarray, streams: int | None = None) -> np.ndarray:
    """Fill ``out`` (C-contiguous) from ``path`` at ``offset``: ``streams``
    handles (default ``KernelConfig.read_streams``), each reading its own contiguous
    part sequentially. Network storage throughput scales with handles, not
    with threads sharing one."""
    view = memoryview(out.reshape(-1).view(np.uint8))
    streams = max(1, min(streams or STREAMS, len(view) // MIN_STREAM_BYTES))
    if streams == 1:
        _read_range(path, offset, view)
        return out
    step = -(-len(view) // streams)
    step = -(-step // CHUNK) * CHUNK
    parts = [(start, view[start:start + step]) for start in range(0, len(view), step)]
    with ThreadPoolExecutor(len(parts)) as pool:
        list(pool.map(lambda part: _read_range(path, offset + part[0], part[1]), parts))
    return out


def read_array(path: Path, offset: int, shape, dtype) -> np.ndarray:
    return read_into(path, offset, np.empty(tuple(shape), _dtype(dtype) if isinstance(dtype, str) else dtype))


class RankDir:
    """One rank's pre-shard directory and its index."""

    def __init__(self, presharded, rank: int, layers: int = 93):
        self.path = Path(presharded) / f"rank{rank:02d}"
        self.index = json.loads((self.path / "index.json").read_text())
        if self.index.get("rank") != rank or self.index.get("layers") != layers:
            raise ValueError(f"unexpected pre-shard index in {self.path}")
        self.families = self.index["families"]

    def spec(self, name):
        return self.families[name]

    def dense_names(self):
        return [name for name, spec in self.families.items() if spec.get("file") == "dense.bin"]

    def read_dense(self, name) -> np.ndarray:
        spec = self.families[name]
        return read_array(self.path / "dense.bin", spec["offset"], spec["shape"], spec["dtype"])

    def read_dense_all(self) -> dict[str, np.ndarray]:
        """Every dense family in one sequential pass over ``dense.bin``."""
        spec_of = {name: self.families[name] for name in self.dense_names()}
        order = sorted(spec_of, key=lambda n: spec_of[n]["offset"])
        end = max(spec_of[n]["offset"] + int(np.prod(spec_of[n]["shape"])) * _dtype(spec_of[n]["dtype"]).itemsize
                  for n in order)
        raw = read_into(self.path / "dense.bin", 0, np.empty(end, np.uint8))
        result = {}
        for name in order:
            spec = spec_of[name]
            dtype = _dtype(spec["dtype"])
            count = int(np.prod(spec["shape"]))
            result[name] = raw[spec["offset"]:spec["offset"] + count * dtype.itemsize].view(dtype).reshape(spec["shape"])
        return result

    def read_dense_layers(self, name, lo, hi) -> np.ndarray:
        """Layers [lo, hi) of a dense family with a leading layer axis."""
        spec = self.families[name]
        shape, dtype = tuple(spec["shape"]), _dtype(spec["dtype"])
        per_layer = int(np.prod(shape[1:])) * dtype.itemsize
        return read_array(self.path / "dense.bin", spec["offset"] + lo * per_layer,
                          (hi - lo, *shape[1:]), dtype)

    def expert_layers(self, name, lo, hi, quarter_dirs) -> np.ndarray:
        """Layers [lo, hi) of an expert family, all of the EP8 group's
        experts over the TP4 quarters in ``quarter_dirs`` (gate quarters then
        up quarters along N; down and its scales along K)."""
        spec = self.families[name]
        shape, dtype = tuple(spec["shape"]), _dtype(spec["dtype"])
        per_layer = int(np.prod(shape[1:])) * dtype.itemsize
        quarters = []
        for rank_dir in quarter_dirs:
            parts = []
            for filename, (first, last) in zip(spec["files"], spec["layer_ranges"]):
                a, b = max(lo, first), min(hi, last)
                if a < b:
                    parts.append(read_array(rank_dir.path / filename, (a - first) * per_layer,
                                            (b - a, *shape[1:]), dtype))
            quarters.append(np.concatenate(parts) if len(parts) > 1 else parts[0])
        if name in ("expert_gu", "expert_gus"):
            half = shape[-1] // 2
            return np.concatenate([q[..., :half] for q in quarters]
                                  + [q[..., half:] for q in quarters], axis=-1)
        return np.concatenate(quarters, axis=2)

    def expert_parts(self, name):
        """``(label, nbytes, reader)`` per file of an expert family, in layer
        order; ``label`` is ``name@index/count`` (see ``join_parts``)."""
        spec = self.families[name]
        shape = tuple(spec["shape"])
        dtype = _dtype(spec["dtype"])
        per_layer = int(np.prod(shape[1:])) * dtype.itemsize
        count = len(spec["files"])
        for index, (filename, (lo, hi)) in enumerate(zip(spec["files"], spec["layer_ranges"])):
            def reader(filename=filename, lo=lo, hi=hi):
                return read_into(self.path / filename, 0, np.empty((hi - lo, *shape[1:]), dtype))
            yield f"{name}@{index}/{count}", (hi - lo) * per_layer, reader

    def regrouped_expert_parts(self, name, quarter_dirs, part, parts):
        """An expert family regrouped from EP8/TP4 shards, per file like
        ``expert_parts``: experts ``part`` of ``parts`` equal slices of the EP8
        group's 112, over the TP4 quarters in ``quarter_dirs`` (two: half the
        intermediate, EP16/TP2) -- gate quarters then up quarters along N,
        down and its scales along K."""
        spec = self.families[name]
        shape = tuple(spec["shape"])  # [layers, 112, ...] per quarter
        dtype = _dtype(spec["dtype"])
        experts = shape[1] // parts
        count = len(spec["files"])
        nbytes = int(np.prod((experts,) + shape[2:])) * dtype.itemsize * len(quarter_dirs)
        for index, (filename, (lo, hi)) in enumerate(zip(spec["files"], spec["layer_ranges"])):
            def reader(filename=filename, lo=lo, hi=hi):
                quarters = []
                for rank_dir in quarter_dirs:
                    array = np.memmap(rank_dir.path / filename, dtype, "r", shape=(hi - lo, *shape[1:]))
                    quarters.append(np.array(array[:, part * experts:(part + 1) * experts]))
                if name in ("expert_gu", "expert_gus"):
                    half = shape[-1] // 2
                    return np.concatenate([q[..., :half] for q in quarters]
                                          + [q[..., half:] for q in quarters], axis=-1)
                return np.concatenate(quarters, axis=2)
            yield f"{name}@{index}/{count}", (hi - lo) * nbytes, reader


def split_kda_projection(families: dict) -> dict:
    """Split ``k_gate`` out of an older pre-shard's ``k_projection [.., 1792]``."""
    combined = families.get("k_projection")
    if combined is not None and combined.shape[-1] == 1792 and "k_gate" not in families:
        families["k_gate"] = np.ascontiguousarray(combined[..., 1152:1280])
        families["k_projection"] = np.ascontiguousarray(
            np.concatenate((combined[..., :1152], combined[..., 1280:]), axis=-1))
    return families


# Host bytes one rank may hold in flight while loading. Eight ranks share a
# host; larger budgets can push a NUMA node into direct reclaim and evict the
# page cache that keeps restarts warm.
INFLIGHT_BYTES = int(_config().load_inflight_gb * 2**30)


def _budgeted(jobs, *, budget: int = INFLIGHT_BYTES, workers: int = 8):
    """Run ``(key, nbytes, fn)`` jobs on threads with at most ``budget`` bytes
    in flight (one job always may run), yielding ``(key, result)`` in job
    order; a job's bytes are released when the caller resumes after it."""
    jobs = list(jobs)
    with ThreadPoolExecutor(workers) as pool:
        pending: list = []  # (key, nbytes, future) in submission order
        inflight = 0
        next_job = 0

        def fill():
            nonlocal inflight, next_job
            while next_job < len(jobs):
                key, nbytes, fn = jobs[next_job]
                if pending and inflight + nbytes > budget:
                    return
                pending.append((key, nbytes, pool.submit(fn)))
                inflight += nbytes
                next_job += 1

        fill()
        while pending:
            key, nbytes, future = pending.pop(0)
            result = future.result()
            yield key, result
            del result
            inflight -= nbytes
            fill()


def _dense_bytes(rank_dir: RankDir) -> int:
    specs = [rank_dir.families[n] for n in rank_dir.dense_names()]
    return max(s["offset"] + int(np.prod(s["shape"])) * _dtype(s["dtype"]).itemsize for s in specs)


def _family_bytes(spec) -> int:
    return int(np.prod(spec["shape"])) * _dtype(spec["dtype"]).itemsize


def _expert_jobs(rank_dir: RankDir):
    """Expert families largest first, one job per file."""
    experts = sorted(EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n)))
    for name in experts:
        yield from rank_dir.expert_parts(name)


def iter_decode_rank(presharded, rank: int, *, layers: int = 93,
                     attention_dp: int = 1) -> Iterator[tuple[str, str, np.ndarray]]:
    """The decode megakernel's weights for ``rank`` (MXFP4 experts as stored).

    Yields ``("decode", name, array)``: the four packed expert families
    (largest first, one array per file, named ``family@index/count``, layer
    ranges in order -- the caller joins them),
    then every dense family (``k_projection`` split into ``k_gate`` and the
    rest), with at most ``KernelConfig.load_inflight_gb`` of host buffers in flight.
    With ``attention_dp`` > 1 the head-sharded attention families hold the
    rank's held TP32 shards (see :func:`attention_dp_families`).
    """
    rank_dir = RankDir(presharded, rank, layers)
    jobs: list[tuple] = list(_expert_jobs(rank_dir))
    jobs.append(("dense", _dense_bytes(rank_dir), rank_dir.read_dense_all))
    for key, value in _budgeted(jobs):
        if key == "dense":
            families = split_kda_projection(value)
            if attention_dp > 1:
                families.update(attention_dp_families(families, presharded, rank, layers, attention_dp))
            for name, family in families.items():
                yield "decode", name, family
        else:
            yield "decode", key, value


def attention_dp_families(families: dict, presharded, rank: int, layers: int, dp: int) -> dict:
    """Attention DP's head-sharded families for ``rank``: its TP32 shards
    ``rank % (32 / dp) + k * (32 / dp)`` in order ``k``, concatenated or
    stacked per ``decode_megakernel.ATTENTION_DP_WEIGHTS``, the other shards
    read from their ranks' pre-shards."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels.decode_megakernel import (
        ATTENTION_DP_WEIGHTS,
    )

    group = 32 // dp
    shards = []
    for shard in range(dp):
        source = rank % group + shard * group
        if source == rank:
            shards.append(families)
            continue
        peer = RankDir(presharded, source, layers)
        names = [name for name in ATTENTION_DP_WEIGHTS if name in peer.families]
        shards.append(split_kda_projection({name: peer.read_dense(name) for name in names}))
    held = {}
    for name, (kind, axis) in ATTENTION_DP_WEIGHTS.items():
        parts = [shard[name] for shard in shards]
        if axis is None:  # "stack"
            held[name] = np.stack(parts, axis=1)
        else:
            held[name] = np.concatenate(parts, axis=1 + axis)
    return held


def _sequence_parallel_output(own: np.ndarray, peer_dir: RankDir, name: str, rank: int) -> np.ndarray:
    """Prefill TP32 output projection from the chip pair's decode shards."""
    pair, part = divmod(rank, 2)
    rows = slice(part * 384, (part + 1) * 384)
    peer = peer_dir.read_dense(name)
    halves = (own, peer) if rank % 2 == 0 else (peer, own)
    return np.ascontiguousarray(np.concatenate([h[:, rows] for h in halves], axis=-1))


def iter_prefill_rank(presharded, projection_cache, rank: int, *,
                      layers: int = 93, span: tuple[int, int] | None = None,
                      ) -> Iterator[tuple[str, str, np.ndarray]]:
    """The prefill stack's weights for ``rank``.

    Yields ``(group, name, array)`` with ``group`` in io/common/kda/mla/moe
    (per-rank arrays, without the rank axis); expert families first, per
    file, as EP16/TP2 parts regrouped from the EP8 group's shards. With
    ``span`` = (first, last), only those layers' slots of every family and no
    vocabulary or layer-0 FFN shards (a layer probe).
    """
    if span is not None:
        yield from _iter_prefill_rank_span(presharded, projection_cache, rank, *span, layers=layers)
        return
    rank_dir = RankDir(presharded, rank, layers)
    peer_dir = RankDir(presharded, rank ^ 1, layers)
    cache = Path(projection_cache)
    layout = json.loads((cache / "layout.json").read_text())
    if layout.get("layers") != layers:
        raise ValueError(f"unexpected projection cache {cache}")

    def projection(name):
        spec = layout["families"][name]
        return read_array(cache / spec["file"], 0, spec["shape"], spec["dtype"])

    parity = rank % 2

    def cached(name):
        spec = layout["families"][name]
        return (int(np.prod(spec["shape"])) * _dtype(spec["dtype"]).itemsize,
                lambda: projection(name))

    # EP16/TP2: 56 experts, half the intermediate (TP2 within the chip),
    # regrouped from the EP8 group's quarters (same host).
    group = 4 * (rank // 4)
    quarters, part, parts = (2 * (rank % 2), 2 * (rank % 2) + 1), (rank // 2) % 2, 2
    quarter_dirs = [RankDir(presharded, group + q, layers) for q in quarters]
    jobs: list[tuple] = [
        job for name in sorted(EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n)))
        for job in rank_dir.regrouped_expert_parts(name, quarter_dirs, part, parts)]
    jobs.append(("dense", _dense_bytes(rank_dir), rank_dir.read_dense_all))
    for name, source in (("latent_down", "latent_down"), ("latent_up", "latent_up"),
                         ("shared_gu", f"shared_gu_{parity}"), ("shared_down", f"shared_down_{parity}")):
        nbytes, fn = cached(source)
        jobs.append((name, nbytes, fn))
    for key, value in _budgeted(jobs):
        if key != "dense":
            yield "moe", key, value
            continue
        families = split_kda_projection(value)
        del value
        for name in PREFILL_IO:
            yield "io", name, families[name]
        for name in PREFILL_COMMON:
            yield "common", name, families[name]
        for name in PREFILL_KDA:
            family = families[name]
            if name == "k_o":
                family = _sequence_parallel_output(family, peer_dir, "k_o", rank)
            yield "kda", name, family
        for name in PREFILL_MLA:
            family = families[name]
            if name == "m_o":
                family = _sequence_parallel_output(family, peer_dir, "m_o", rank)
            yield "mla", name, family
        for name in PREFILL_MOE_DENSE:
            yield "moe", name, families[name]
        del families


# A pipeline stage of the prefill: one host's 8 cores run a contiguous run
# of whole 4-layer groups. Stage rank j holds the 12 heads of TP32 ranks
# 4j..4j+3 (as four 3-head groups, each one TP32 rank's shard and layout) and,
# EP4 x TP2, its chip's two EP8 groups' 224 experts over the TP4 quarters of
# its core parity.
STAGE_RANKS = 8
STAGE_HEAD_GROUPS = 32 // STAGE_RANKS


def stage_slots(first: int, last: int) -> dict[str, tuple[int, int]]:
    """The family slots of layers [first, last): ``kda``, ``mla``, ``moe``
    and ``layer`` (per-layer families) as [lo, hi) ranges. MLA layers are
    3, 7, ..., 91 and 92; layer 0 has the dense FFN, layers 1..92 the MoE."""
    mla = [i for i in range(first, last) if i % 4 == 3 or i == 92]
    kda = [i for i in range(first, last) if i not in mla]

    def span(slots):
        return (slots[0], slots[-1] + 1) if slots else (0, 0)

    return {
        "layer": (first, last),
        "kda": span([i - (i + 1) // 4 for i in kda]),
        "mla": span([i // 4 if i < 92 else 23 for i in mla]),
        "moe": span([i - 1 for i in range(max(first, 1), last)]),
    }


def iter_prefill_stage_rank(presharded, projection_cache, core: int, first: int, last: int,
                            *, layers: int = 93) -> Iterator[tuple[str, str, np.ndarray]]:
    """The prefill stage weights of layers [first, last) for stage rank
    ``core`` (0..7), like :func:`iter_prefill_rank`: head-sharded attention
    families stacked as ``[L, 4, ...]`` head groups (KDA output projections
    ``[L, 1536, 7168]``, the groups' rows in order; the MLA output gate and
    projection of all 96 heads, ``[L, 7168, 12288]`` and ``[L, 12288, 7168]``
    in TP32 rank order), EP4 x TP2 experts
    ``[L, 224, ...]`` (per expert file, as ``family@index/count`` parts),
    the vocabulary shards of the first (embedding) and the last (LM head)
    stage. Reads run on threads with at most ``KernelConfig.load_inflight_gb``
    in flight."""
    slots = stage_slots(first, last)
    every = [RankDir(presharded, rank, layers) for rank in range(STAGE_HEAD_GROUPS * STAGE_RANKS)]
    heads = every[STAGE_HEAD_GROUPS * core:STAGE_HEAD_GROUPS * (core + 1)]
    own = heads[0]
    cache = Path(projection_cache)
    layout = json.loads((cache / "layout.json").read_text())
    if layout.get("layers") != layers:
        raise ValueError(f"unexpected projection cache {cache}")
    chip, parity = divmod(core, 2)
    quarter_dirs = [[RankDir(presharded, 4 * group + q, layers) for q in (2 * parity, 2 * parity + 1)]
                    for group in (2 * chip, 2 * chip + 1)]
    jobs: list[tuple] = []

    def layer_bytes(spec, count, copies=1):
        return int(np.prod(spec["shape"][1:])) * _dtype(spec["dtype"]).itemsize * count * copies

    def cached(name, lo, hi):
        spec = layout["families"][name]
        shape, dtype = tuple(spec["shape"]), _dtype(spec["dtype"])
        per_layer = int(np.prod(shape[1:])) * dtype.itemsize
        return read_array(cache / spec["file"], lo * per_layer, (hi - lo, *shape[1:]), dtype)

    moe_lo, moe_hi = slots["moe"]
    if moe_hi > moe_lo:
        for name in sorted(EXPERT_FAMILIES, key=lambda n: -_family_bytes(own.spec(n))):
            spec = own.spec(name)
            # Four-layer pieces within each expert file: bounded host buffers.
            ranges = [(c, min(c + 4, min(moe_hi, b))) for a, b in spec["layer_ranges"]
                      for c in range(max(moe_lo, a), min(moe_hi, b), 4)]
            for index, (a, b) in enumerate(ranges):
                def experts(name=name, a=a, b=b):
                    return np.concatenate(
                        [own.expert_layers(name, a, b, dirs) for dirs in quarter_dirs], axis=1)
                jobs.append((("moe", f"{name}@{index}/{len(ranges)}"),
                             layer_bytes(spec, b - a, 4), experts))
        for name, source in (("latent_down", "latent_down"), ("latent_up", "latent_up"),
                             ("shared_gu", f"shared_gu_{parity}"),
                             ("shared_down", f"shared_down_{parity}")):
            jobs.append((("moe", name), layer_bytes(layout["families"][source], moe_hi - moe_lo),
                         functools.partial(cached, source, moe_lo, moe_hi)))
        for name in PREFILL_MOE_DENSE:
            jobs.append((("moe", name), layer_bytes(own.spec(name), moe_hi - moe_lo),
                         functools.partial(own.read_dense_layers, name, moe_lo, moe_hi)))

    def head_groups(name, lo, hi, axis, dirs=heads):
        parts = []
        for rank_dir in dirs:
            part = rank_dir.read_dense_layers(name, lo, hi)
            if name == "k_projection" and part.shape[-1] == 1792:
                part = np.concatenate((part[..., :1152], part[..., 1280:]), axis=-1)
            if name == "k_o":
                rank = rank_dir.index["rank"]
                peer = every[rank ^ 1].read_dense_layers(name, lo, hi)
                halves = (part, peer) if rank % 2 == 0 else (peer, part)
                rows = slice((rank % 2) * 384, (rank % 2 + 1) * 384)
                part = np.concatenate([h[:, rows] for h in halves], axis=-1)
            parts.append(part)
        if axis is None:
            return np.ascontiguousarray(np.stack(parts, axis=1))
        return np.ascontiguousarray(np.concatenate(parts, axis=axis))

    def every_output(lo, hi):
        # A chip pair's two decode shards side by side are its six heads' rows.
        pairs = [np.concatenate([d.read_dense_layers("m_o", lo, hi) for d in every[r:r + 2]], axis=-1)
                 for r in range(0, len(every), 2)]
        return np.ascontiguousarray(np.concatenate(pairs, axis=1))

    kda_lo, kda_hi = slots["kda"]
    if kda_hi > kda_lo:
        def k_gate():
            return np.ascontiguousarray(
                own.read_dense_layers("k_projection", kda_lo, kda_hi)[..., 1152:1280])
        jobs.append((("kda", "k_gate"), layer_bytes(own.spec("k_projection"), kda_hi - kda_lo), k_gate))
        for name in PREFILL_KDA:
            if name != "k_gate":
                jobs.append((("kda", name), layer_bytes(own.spec(name), kda_hi - kda_lo, 8),
                             functools.partial(head_groups, name, kda_lo, kda_hi,
                                               1 if name == "k_o" else None)))
    mla_lo, mla_hi = slots["mla"]
    if mla_hi > mla_lo:
        for name in PREFILL_MLA:
            copies = 4
            if name in ("m_qa", "m_ka", "m_qnorm", "m_knorm"):
                fn, copies = functools.partial(own.read_dense_layers, name, mla_lo, mla_hi), 1
            elif name == "m_gate":
                # Every head: the stage MLA gates and projects its own rows.
                fn, copies = functools.partial(head_groups, name, mla_lo, mla_hi, -1, every), len(every)
            elif name == "m_o":
                fn, copies = functools.partial(every_output, mla_lo, mla_hi), len(every)
            else:
                fn = functools.partial(head_groups, name, mla_lo, mla_hi, -1)
            jobs.append((("mla", name), layer_bytes(own.spec(name), mla_hi - mla_lo, copies), fn))
    for name in ("attn_res", "attn_norm", "mlp_res", "ffn_norm"):
        jobs.append((("common", name), layer_bytes(own.spec(name), last - first),
                     functools.partial(own.read_dense_layers, name, first, last)))
    if first == 0:
        def dense_gu():
            # Each shard's gate and up halves, regrouped as all gates then all ups.
            shards = [np.split(d.read_dense("dense_gu"), 2, axis=-1) for d in heads]
            return np.concatenate([gate for gate, _ in shards] + [up for _, up in shards], axis=-1)
        jobs.append((("common", "dense_gu"), 4 * _family_bytes(own.spec("dense_gu")), dense_gu))
        for name, group in (("dense_down", "common"), ("embedding", "io")):
            jobs.append(((group, name), 4 * _family_bytes(own.spec(name)), functools.partial(
                lambda n: np.concatenate([d.read_dense(n) for d in heads], axis=0), name)))
    if last == layers:
        jobs.append((("io", "lm_head"), 4 * _family_bytes(own.spec("lm_head")),
                     lambda: np.concatenate([d.read_dense("lm_head") for d in heads], axis=-1)))
        for name in ("output_res", "final_norm"):
            jobs.append((("io", name), _family_bytes(own.spec(name)),
                         functools.partial(own.read_dense, name)))
    for (group, name), value in _budgeted(jobs):
        yield group, name, value


def _iter_prefill_rank_span(presharded, projection_cache, rank: int, first: int, last: int,
                            *, layers: int) -> Iterator[tuple[str, str, np.ndarray]]:
    """:func:`iter_prefill_rank`'s families for layers [first, last) only."""
    slots = stage_slots(first, last)
    rank_dir = RankDir(presharded, rank, layers)
    peer_dir = RankDir(presharded, rank ^ 1, layers)
    cache = Path(projection_cache)
    layout = json.loads((cache / "layout.json").read_text())
    parity = rank % 2
    moe_lo, moe_hi = slots["moe"]

    def cached(name, lo, hi):
        spec = layout["families"][name]
        shape, dtype = tuple(spec["shape"]), _dtype(spec["dtype"])
        per_layer = int(np.prod(shape[1:])) * dtype.itemsize
        return read_array(cache / spec["file"], lo * per_layer, (hi - lo, *shape[1:]), dtype)

    if moe_hi > moe_lo:
        group = 4 * (rank // 4)
        quarter_dirs = [RankDir(presharded, group + q, layers)
                        for q in (2 * parity, 2 * parity + 1)]
        part = (rank // 2) % 2
        for name in EXPERT_FAMILIES:
            experts = rank_dir.expert_layers(name, moe_lo, moe_hi, quarter_dirs)
            half = experts.shape[1] // 2
            yield "moe", name, np.ascontiguousarray(experts[:, part * half:(part + 1) * half])
        for name, source in (("latent_down", "latent_down"), ("latent_up", "latent_up"),
                             ("shared_gu", f"shared_gu_{parity}"),
                             ("shared_down", f"shared_down_{parity}")):
            yield "moe", name, cached(source, moe_lo, moe_hi)
        for name in PREFILL_MOE_DENSE:
            yield "moe", name, rank_dir.read_dense_layers(name, moe_lo, moe_hi)

    def sp_output(name, lo, hi):
        own = rank_dir.read_dense_layers(name, lo, hi)
        peer = peer_dir.read_dense_layers(name, lo, hi)
        halves = (own, peer) if rank % 2 == 0 else (peer, own)
        rows = slice(parity * 384, (parity + 1) * 384)
        return np.ascontiguousarray(np.concatenate([h[:, rows] for h in halves], axis=-1))

    kda_lo, kda_hi = slots["kda"]
    if kda_hi > kda_lo:
        projection = rank_dir.read_dense_layers("k_projection", kda_lo, kda_hi)
        if projection.shape[-1] == 1792:
            yield "kda", "k_gate", np.ascontiguousarray(projection[..., 1152:1280])
            projection = np.concatenate((projection[..., :1152], projection[..., 1280:]), axis=-1)
        for name in PREFILL_KDA:
            if name == "k_gate":
                continue
            if name == "k_projection":
                yield "kda", name, np.ascontiguousarray(projection)
            elif name == "k_o":
                yield "kda", name, sp_output(name, kda_lo, kda_hi)
            else:
                yield "kda", name, rank_dir.read_dense_layers(name, kda_lo, kda_hi)
    mla_lo, mla_hi = slots["mla"]
    if mla_hi > mla_lo:
        for name in PREFILL_MLA:
            if name == "m_o":
                yield "mla", name, sp_output(name, mla_lo, mla_hi)
            else:
                yield "mla", name, rank_dir.read_dense_layers(name, mla_lo, mla_hi)
    for name in ("attn_res", "attn_norm", "mlp_res", "ffn_norm"):
        yield "common", name, rank_dir.read_dense_layers(name, first, last)
