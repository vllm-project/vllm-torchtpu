# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming per-rank readers for the TP32 pre-shards (decode and prefill).

Both megakernels read the same pre-sharded rank directories
(``<presharded>/rankNN/{index.json, dense.bin, expert_*.bin}``); the prefill
additionally reads the SP projection cache
(``latent_down.bin``, ``latent_up.bin``, ``shared_{gu,down}_{0,1}.bin``).
Seeks and mmap page faults are slow on network storage, so reads are mostly
large sequential ``readinto`` calls on separate handles, straight into the
numpy array the caller keeps (the regrouped EP16/EP32 expert parts slice
memory-mapped shards instead).

The iterators yield ``(group, name, array)`` host arrays for one rank, in
the per-rank layout the kernels consume (no leading rank axis). They are
independent of JAX so a vLLM worker can upload them through torch, and
:func:`to_device` builds the kernels' rank-leading global arrays from them.
"""

from __future__ import annotations

import json
import os
import threading
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
        group's 112, over the TP4 quarters in ``quarter_dirs`` (all four:
        whole experts, EP32; two: half the intermediate, EP16/TP2) -- gate
        quarters then up quarters along N, down and its scales along K."""
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

    def read_expert_family(self, name) -> np.ndarray:
        """One expert family ``[92, ...]`` from its files, one reader per file."""
        spec = self.families[name]
        shape = tuple(spec["shape"])
        dtype = _dtype(spec["dtype"])
        out = np.empty(shape, dtype)
        with ThreadPoolExecutor(len(spec["files"])) as pool:
            list(pool.map(lambda item: read_into(self.path / item[0], 0, out[item[1][0]:item[1][1]]),
                          zip(spec["files"], spec["layer_ranges"])))
        return out


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


def _expert_jobs(rank_dir: RankDir, split: bool):
    """Expert families largest first, whole or (``split``) one job per file."""
    experts = sorted(EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n)))
    for name in experts:
        if split:
            yield from rank_dir.expert_parts(name)
        else:
            yield name, _family_bytes(rank_dir.spec(name)), lambda n=name: rank_dir.read_expert_family(n)


HOT_LOCAL = 112  # EP8 local experts; hot copies follow
FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                      np.float32)


MAX_COPY_SLOTS = 16  # copy slot j is local expert id 112 + j; the expert loop's lane masks hold ids < 128


def _plan_path(path):
    if not os.path.isabs(path) and not os.path.exists(path):
        # A bare name: one of the plans shipped in kimi_k3/data.
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", path)
    return path


def _check_slots(path, experts):
    if experts.ndim != 3 or experts.shape[1] != 8 or not 1 <= experts.shape[2] <= MAX_COPY_SLOTS:
        raise ValueError(f"plan {path}: experts must be [layers, 8, S] with 1 <= S <= {MAX_COPY_SLOTS}")


def hot_plan(rank: int, layers: int):
    """This rank's hot copy slots, [layers - 1, S] local ids of its own EP group (-1 empty), or None.
    Fails closed when the plan places another group's expert here (home-only copies)."""
    path = _config().kd_hot_plan
    if not path:
        return None
    path = _plan_path(path)
    experts = np.load(path)["experts"][: layers - 1]
    _check_slots(path, experts)
    group = rank // 4
    mine = experts[:, group, :]
    if not np.all((mine < 0) | (mine // HOT_LOCAL == group)):
        raise ValueError(f"hot plan {path}: EP group {group} slots hold experts of other groups (home-only loader)")
    return np.where(mine < 0, -1, mine % HOT_LOCAL).astype(np.int32), experts


def _family_slice(rank_dir, name, layer, expert):
    """One expert's entry of a family at one layer, memory-mapped from its file."""
    spec = rank_dir.families[name]
    shape = tuple(spec["shape"])
    dtype = _dtype(spec["dtype"])
    for filename, (lo, hi) in zip(spec["files"], spec["layer_ranges"]):
        if lo <= layer < hi:
            array = np.memmap(rank_dir.path / filename, dtype, "r", shape=(hi - lo, *shape[1:]))
            return np.array(array[layer - lo, expert])
    raise KeyError((name, layer))


def _fp8_words(packed, scales):
    """MXFP4 [K/8, N] u32 + E8M0 [K/32, N] -> fp8 e4m3 rows packed 4 per u32 word ([K/4, N], row 4r + b in
    byte b), FP32 column scales [N], and whether the re-expression was exact."""
    import ml_dtypes

    k8, columns = packed.shape
    nibbles = np.stack([(packed >> (4 * i)) & 15 for i in range(8)], axis=1).reshape(k8 * 8, columns)
    values = FP4_VALUES[nibbles]
    exponents = scales.astype(np.int32)
    emax = exponents.max(axis=0)
    relative = np.ldexp(np.float32(1), np.repeat(exponents - emax[None, :], 32, axis=0)).astype(np.float32)
    scaled = values * relative
    fp8 = scaled.astype(ml_dtypes.float8_e4m3fn)
    exact = bool(np.array_equal(fp8.astype(np.float32), scaled))
    words = fp8.view(np.uint8).reshape(k8 * 2, 4, columns).transpose(0, 2, 1).copy().view("<u4").reshape(k8 * 2, columns)
    column = np.ldexp(np.float32(1), emax - 127).astype(np.float32)
    return words, column, exact


def _extend_hot(rank_dir, name, part, lo, slots):
    """Append S hot copies (two entries each) to one layer-range part of an expert family."""
    n_layers, _, rows, columns = part.shape
    count = slots.shape[1]
    extra = np.zeros((n_layers, 2 * count, rows, columns), part.dtype)
    inexact = 0
    for i in range(n_layers):
        for j, expert in enumerate(slots[lo + i]):
            if expert < 0:
                continue
            if name in ("expert_gu", "expert_down"):
                scale_name = "expert_gus" if name == "expert_gu" else "expert_ds"
                words, _, exact = _fp8_words(part[i, expert], _family_slice(rank_dir, scale_name, lo + i, expert))
                extra[i, 2 * j : 2 * j + 2] = words.reshape(2, rows, columns)
                inexact += not exact
            else:
                exponents = part[i, expert].astype(np.int32)
                column = np.ldexp(np.float32(1), exponents.max(axis=0) - 127).astype(np.float32)
                extra[i, 2 * j, :4] = column.view(np.uint8).reshape(columns, 4).T
    if inexact:
        raise ValueError(f"{name}: {inexact} hot copies are not exact in fp8 (E8M0 spread too wide)")
    return np.concatenate([part, extra], axis=1)


def _hot_table(latent_norm, experts):
    """The hot-copy table in latent_norm's zero padding lanes (as f32): 3584 + i expert id,
    3584 + 8S + i holder group, 3584 + 16S + i local slot id, i = g * S + j."""
    count = experts.shape[2]
    table = latent_norm.copy()
    for g in range(8):
        for j in range(count):
            i = g * count + j
            table[:, 0, 3584 + i] = experts[:, g, j]
            table[:, 0, 3584 + 8 * count + i] = np.where(experts[:, g, j] < 0, -1, g)
            table[:, 0, 3584 + 16 * count + i] = np.where(experts[:, g, j] < 0, -1, HOT_LOCAL + j)
    return table


def replica_plan(rank: int, layers: int):
    """MXFP4 replicas (KernelConfig.kd_replica_plan): this rank's EP group's slots as global expert ids
    [layers - 1, S] (-1 empty), the whole plan [layers - 1, 8, S] and the dispatch order [layers - 1, 8 S],
    or None. Fails closed on malformed plans."""
    path = _config().kd_replica_plan
    if not path:
        return None
    path = _plan_path(path)
    plan = np.load(path)
    experts = plan["experts"][: layers - 1].astype(np.int64)
    _check_slots(path, experts)
    order = plan["order"][: layers - 1].astype(np.int64)
    count = experts.shape[2]
    if order.shape != (experts.shape[0], 8 * count) or any(
        sorted(order[layer].tolist()) != list(range(8 * count)) for layer in range(order.shape[0])
    ):
        raise ValueError(f"replica plan {path}: order must permute the 8 S slots of every layer")
    for layer in range(experts.shape[0]):
        used = experts[layer][experts[layer] >= 0]
        if used.max(initial=-1) >= 8 * HOT_LOCAL or len(set(used.tolist())) != used.size:
            raise ValueError(f"replica plan {path}: layer {layer} has invalid or repeated experts")
    return experts[:, rank // 4, :], experts, order


def _extend_replicas(rank_dir, presharded, name, part, lo, slots, rank):
    """Append S MXFP4 replica entries per layer to one layer-range part of an expert family: slot j holds this
    rank's TP4 quarter of global expert slots[layer, j], read from the owning group's rank directory."""
    n_layers = part.shape[0]
    count = slots.shape[1]
    extra = np.zeros((n_layers, count, *part.shape[2:]), part.dtype)
    sources = {}
    for i in range(n_layers):
        for j, expert in enumerate(slots[lo + i]):
            if expert < 0:
                continue
            owner = 4 * (int(expert) // HOT_LOCAL) + rank % 4
            if owner == rank:
                extra[i, j] = part[i, int(expert) % HOT_LOCAL]
                continue
            if owner not in sources:
                sources[owner] = RankDir(presharded, owner, rank_dir.index["layers"])
            extra[i, j] = _family_slice(sources[owner], name, lo + i, int(expert) % HOT_LOCAL)
    return np.concatenate([part, extra], axis=1)


def _replica_table(latent_norm, experts, order):
    """The replica table in latent_norm's zero padding lanes (as f32), entries in dispatch order:
    3584 + i expert id, 3584 + 8S + i holder group, 3584 + 16S + i local slot id (112 + j)."""
    count = experts.shape[2]
    entries = 8 * count
    table = latent_norm.copy()
    if np.any(table[:, 0, 3584:3584 + 3 * entries] != 0):
        raise ValueError("latent_norm padding lanes are not zero")
    for layer in range(experts.shape[0]):
        for i, flat in enumerate(order[layer]):
            g, j = divmod(int(flat), count)
            expert = experts[layer, g, j]
            table[layer, 0, 3584 + i] = expert
            table[layer, 0, 3584 + entries + i] = g if expert >= 0 else -1
            table[layer, 0, 3584 + 2 * entries + i] = HOT_LOCAL + j if expert >= 0 else -1
    return table


def iter_decode_rank(presharded, rank: int, *, layers: int = 93, split_experts: bool = False,
                     attention_dp: int = 1) -> Iterator[tuple[str, str, np.ndarray]]:
    """The decode megakernel's weights for ``rank`` (MXFP4 experts as stored).

    Yields ``("decode", name, array)``: the four packed expert families
    (largest first; with ``split_experts`` one array per file, named
    ``family@index/count``, layer ranges in order -- the caller joins them),
    then every dense family (``k_projection`` split into ``k_gate`` and the
    rest), with at most ``KernelConfig.load_inflight_gb`` of host buffers in flight.
    With ``attention_dp`` > 1 the head-sharded attention families hold the
    rank's held TP32 shards (see :func:`attention_dp_families`).
    """
    rank_dir = RankDir(presharded, rank, layers)
    plan = hot_plan(rank, layers)
    replicas = replica_plan(rank, layers)
    jobs = list(_expert_jobs(rank_dir, split_experts))
    jobs.append(("dense", _dense_bytes(rank_dir), rank_dir.read_dense_all))
    for key, value in _budgeted(jobs):
        if plan is not None and key != "dense":
            family, _, part = key.partition("@")
            index = int(part.split("/")[0]) if part else 0
            lo = rank_dir.families[family]["layer_ranges"][index][0] if part else 0
            value = _extend_hot(rank_dir, family, value, lo, plan[0])
        if replicas is not None and key != "dense":
            family, _, part = key.partition("@")
            index = int(part.split("/")[0]) if part else 0
            lo = rank_dir.families[family]["layer_ranges"][index][0] if part else 0
            value = _extend_replicas(rank_dir, presharded, family, value, lo, replicas[0], rank)
        if key == "dense":
            families = split_kda_projection(value)
            if plan is not None:
                families["latent_norm"] = _hot_table(families["latent_norm"], plan[1])
            if replicas is not None:
                families["latent_norm"] = _replica_table(families["latent_norm"], replicas[1], replicas[2])
            if attention_dp > 1:
                families.update(attention_dp_families(families, presharded, rank, layers, attention_dp))
            for name, family in families.items():
                yield "decode", name, family
        else:
            yield "decode", key, value


def attention_dp_families(families: dict, presharded, rank: int, layers: int, dp: int) -> dict:
    """Attention DP's head-sharded families for ``rank``: its TP32 shards
    ``rank % (32 / dp) + k * (32 / dp)`` in order ``k``, concatenated or
    stacked as ``decode_megakernel.to_attention_dp`` lays them out, the other
    shards read from their ranks' pre-shards."""
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


def iter_prefill_rank(presharded, projection_cache, rank: int, *, layers: int = 93,
                      split_experts: bool = False) -> Iterator[tuple[str, str, np.ndarray]]:
    """The prefill stack's weights for ``rank``.

    Yields ``(group, name, array)`` with ``group`` in io/common/kda/mla/moe
    (per-rank arrays, without the rank axis); expert families first
    (``split_experts``: per file, as in :func:`iter_decode_rank`).
    """
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

    ep = _config().prefill_expert_parallel
    jobs: list[tuple] = []
    if ep in (16, 32):  # regrouped from the EP8 group's quarters (same host)
        group = 4 * (rank // 4)
        if ep == 32:  # 28 whole experts
            quarters, part, parts = range(4), rank % 4, 4
        else:  # 56 experts, half the intermediate (TP2 within the chip)
            quarters, part, parts = (2 * (rank % 2), 2 * (rank % 2) + 1), (rank // 2) % 2, 2
        quarter_dirs = [RankDir(presharded, group + q, layers) for q in quarters]
        jobs += [job for name in sorted(EXPERT_FAMILIES, key=lambda n: -_family_bytes(rank_dir.spec(n)))
                 for job in rank_dir.regrouped_expert_parts(name, quarter_dirs, part, parts)]
    else:
        jobs += list(_expert_jobs(rank_dir, split_experts))
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


def to_device(mesh, iterator_for_rank, *, axis=None, ranks_in_flight: int = 8):
    """Global rank-leading JAX arrays ``{group: {name: [32, ...]}}`` from a
    per-rank iterator, for this process's devices of ``mesh``."""
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axis = axis or mesh.axis_names[0]
    devices = list(mesh.devices.flat)
    local = [rank for rank, device in enumerate(devices) if device.process_index == jax.process_index()]
    per_rank: dict[int, dict] = {}
    lock = threading.Lock()

    def load(rank):
        result: dict = {}
        for group, name, value in iterator_for_rank(rank):
            array = jax.device_put(value[None], devices[rank])
            array.block_until_ready()
            result.setdefault(group, {})[name] = array
            del value
        with lock:
            per_rank[rank] = result

    with ThreadPoolExecutor(min(ranks_in_flight, len(local))) as pool:
        list(pool.map(load, local))
    sharding = NamedSharding(mesh, P(axis))
    sample = per_rank[local[0]]
    return {
        group: {
            name: jax.make_array_from_single_device_arrays(
                (len(devices), *sample[group][name].shape[1:]), sharding,
                [per_rank[rank][group][name] for rank in local])
            for name in names
        }
        for group, names in sample.items()
    }
