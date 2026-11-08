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
"""Block-major HBM staging for multi-shape KV offload.

A multi-shape KV cache (MiniMax M3: per layer an MSA K/V array, an index-key
array and the drafter's K/V) is many separate device allocations. Raiden moves
one DMA per registered array per block, so a block of ~120 small pages costs
~120 DMAs of a few KB each, and the host<->HBM copies run at ~1 GB/s per rank,
bound by the per-DMA cost.

With staging, Raiden is handed one array instead: an HBM staging buffer of
`rows` block-major rows, each holding every page of one block back to back.
Every block then moves as one DMA of the whole row. The pages move between the
KV arrays and the staging rows on device:

- store: `pack` gathers the blocks' pages into staging rows; the save fence
  covers it; then Raiden saves the rows (device ids = staging rows).
- load: Raiden loads into staging rows; once the load lands, `unpack` scatters
  the rows into the destination blocks, in the step that releases the
  request, so the request's first forward runs after it.

Rows are owned by the scheduler (`StagedTransferScheduler`) and returned only
when nothing can still touch them: a load's rows once every rank reports its
unpack retired, a store's rows once its save finished, a cancelled store's
rows once every rank reports its pack retired.

The host block becomes one contiguous row, a layout different from the
per-array one, so the offload namespace is salted when staging is on.
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import jax

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_offload.base import OffloadKey

logger = init_logger(__name__)

# kv_connector_extra_config key: staging rows per rank (0 or absent: off).
STAGING_BLOCKS_KEY = "staging_blocks"

# A staging row is `(row_bytes // _LANES, _LANES)` int8. Rows are padded to a
# whole number of (32, 128) int8 tiles, so the row is exactly its physical
# bytes and dim 0 is the outermost physical dimension, as Raiden requires
# for block offsetting.
_LANES = 128
_ROW_ALIGN_BYTES = 32 * _LANES

# Pair counts of the compiled pack/unpack programs: 8, 32, 128, ...
_BUCKET_MIN = 8
_BUCKET_GROWTH = 4

# Internal ids of the staged parts handed to the manager; above any base job
# id (vLLM counts those from 0).
_PART_ID_BASE = 1 << 48


def parse_staging_blocks(extra_config: Mapping[str, Any]) -> int:
    """`kv_connector_extra_config["staging_blocks"]`: rows, 0 when off."""
    value = extra_config.get(STAGING_BLOCKS_KEY)
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{STAGING_BLOCKS_KEY} must be an integer, got {value!r}")
    try:
        rows = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{STAGING_BLOCKS_KEY} must be an integer, got {value!r}"
        ) from None
    if rows < 0 or str(rows) != str(value).strip():
        raise ValueError(
            f"{STAGING_BLOCKS_KEY} must be a non-negative integer, got {value!r}"
        )
    return rows


def padded_row_bytes(payload_bytes: int) -> int:
    """A staging row holding `payload_bytes` of pages, tile-padded."""
    return -(-payload_bytes // _ROW_ALIGN_BYTES) * _ROW_ALIGN_BYTES


_CONNECTOR_NAME = "TPURaidenOffloadingConnector"


def staging_extra_config(vllm_config: Any) -> Mapping[str, Any] | None:
    """The extra config of the Raiden offload connector, top level or a child
    of a multi-connector; None without one."""
    kv_tc = getattr(vllm_config, "kv_transfer_config", None)
    if kv_tc is None:
        return None
    extra = kv_tc.kv_connector_extra_config or {}
    if kv_tc.kv_connector == _CONNECTOR_NAME:
        return extra
    for child in extra.get("connectors", None) or ():
        if isinstance(child, Mapping) and child.get("kv_connector") == _CONNECTOR_NAME:
            return child.get("kv_connector_extra_config") or {}
    return None


def staged_layer_page_bytes(
    kv_cache_specs: Mapping[str, Any], skip_draft_kv: bool
) -> dict[str, int]:
    """Per offloaded layer, its page bytes: what one staging row holds.

    `kv_cache_specs` maps layer names to leaf specs. The drafter's layers
    (non-MLA attention with a sliding window next to MLA caches, see
    `raiden_store.draft_kv_layer_names`) are left out with `skip_draft_kv`.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

    has_mla = any(isinstance(s, MLAAttentionSpec) for s in kv_cache_specs.values())
    out: dict[str, int] = {}
    for name, spec in kv_cache_specs.items():
        if (
            skip_draft_kv
            and has_mla
            and isinstance(spec, AttentionSpec)
            and not isinstance(spec, MLAAttentionSpec)
            and getattr(spec, "sliding_window", None) is not None
        ):
            continue
        out[name] = int(spec.page_size_bytes)
    return out


def staging_hbm_reserve_bytes(
    vllm_config: Any, kv_cache_specs: Mapping[str, Any]
) -> int:
    """HBM the staging rows take on this worker: 0 when staging is off.

    The rows are allocated after the memory profile, so the KV cache budget
    must leave them free (`TPUModelRunner.post_profile_hbm_reserves`).
    `kv_cache_specs` is the runner's per-layer spec map; the worker checks
    the rows it allocates against the same figure.
    """
    extra = staging_extra_config(vllm_config)
    if extra is None:
        return 0
    rows = parse_staging_blocks(extra)
    if not rows:
        return 0
    skip = str(extra.get("skip_draft_kv", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    pages = staged_layer_page_bytes(kv_cache_specs, skip)
    return rows * padded_row_bytes(sum(pages.values()))


def bucket_len(n: int) -> int:
    """Smallest compiled pair count >= n."""
    assert n > 0
    size = _BUCKET_MIN
    while size < n:
        size *= _BUCKET_GROWTH
    return size


def bucket_lengths(max_pairs: int) -> list[int]:
    """Every compiled pair count a call of up to `max_pairs` pairs can use."""
    out = [_BUCKET_MIN]
    while out[-1] < max_pairs:
        out.append(out[-1] * _BUCKET_GROWTH)
    return out


def pad_pairs(src: Sequence[int], dst: Sequence[int]) -> tuple[list[int], list[int]]:
    """Pads index pairs to a bucket by repeating the first pair.

    A repeated pair writes the same bytes to the same place again, so the
    padded copy has exactly the effect of the unpadded one (no pad target
    such as the null block is ever written).
    """
    assert len(src) == len(dst) and src, (len(src), len(dst))
    n = bucket_len(len(src))
    pad = n - len(src)
    return list(src) + [src[0]] * pad, list(dst) + [dst[0]] * pad


@dataclass(frozen=True)
class PoolGeometry:
    """One registered KV array: `num_blocks * split` rows of `shape[1:]`."""

    shape: tuple[int, ...]
    dtype: str
    itemsize: int

    def split(self, num_blocks: int) -> int:
        assert self.shape[0] % num_blocks == 0, (self.shape, num_blocks)
        return self.shape[0] // num_blocks

    def page_bytes(self, num_blocks: int) -> int:
        return self.split(num_blocks) * math.prod(self.shape[1:]) * self.itemsize


@dataclass(frozen=True)
class StagingLayout:
    """Where each array's page sits in a staging row."""

    num_blocks: int
    rows: int
    pools: tuple[PoolGeometry, ...]

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError(f"staging needs at least one row, got {self.rows}")
        if not self.pools:
            raise ValueError("staging needs at least one KV array")
        for pool in self.pools:
            if pool.shape[0] % self.num_blocks:
                raise ValueError(
                    f"KV array of shape {pool.shape} does not hold whole "
                    f"blocks of {self.num_blocks}"
                )

    @functools.cached_property
    def page_bytes(self) -> tuple[int, ...]:
        return tuple(pool.page_bytes(self.num_blocks) for pool in self.pools)

    @functools.cached_property
    def offsets(self) -> tuple[int, ...]:
        out, acc = [], 0
        for page in self.page_bytes:
            out.append(acc)
            acc += page
        return tuple(out)

    @property
    def payload_bytes(self) -> int:
        return sum(self.page_bytes)

    @property
    def row_bytes(self) -> int:
        """Bytes of one staging row: the pages plus tile padding."""
        return padded_row_bytes(self.payload_bytes)

    @property
    def staging_shape(self) -> tuple[int, int, int]:
        return (self.rows, self.row_bytes // _LANES, _LANES)

    def signature(self) -> str:
        """Short stable digest of the geometry (op names, logs)."""
        material = repr(
            (self.num_blocks, self.rows, [(p.shape, p.dtype) for p in self.pools])
        )
        return hashlib.sha256(material.encode()).hexdigest()[:12]


# -- device programs (pure JAX, CPU-testable) ----------------------------------


def _expand(blocks, split: int):
    """Block ids -> the `split` consecutive array rows of each block."""
    import jax.numpy as jnp

    if split == 1:
        return blocks
    offsets = jnp.arange(split, dtype=blocks.dtype)
    return (blocks[:, None] * split + offsets).reshape(-1)


def _to_bytes(x, n: int):
    """`(n * split, *page)` of any dtype -> `(n, page_bytes)` int8."""
    import jax
    import jax.numpy as jnp

    # A 1-byte dtype bitcasts in place; a wider one gains a trailing dim of
    # its bytes (little-endian), which the reshape folds into the row.
    return jax.lax.bitcast_convert_type(x, jnp.int8).reshape(n, -1)


def _from_bytes(b, n: int, pool: PoolGeometry, split: int):
    """`(n, page_bytes)` int8 -> `(n * split, *page)` in the array's dtype."""
    import jax
    import jax.numpy as jnp

    dtype = jnp.dtype(pool.dtype)
    shape = (n * split,) + tuple(pool.shape[1:])
    if pool.itemsize > 1:
        shape += (pool.itemsize,)
    return jax.lax.bitcast_convert_type(b.reshape(shape), dtype)


def make_pack_fn(layout: StagingLayout) -> Callable:
    """`(staging, pools, blocks, rows) -> staging'`: row `rows[i]` of the
    staging buffer gets every page of block `blocks[i]`, in layout order."""
    import jax.numpy as jnp

    def pack(staging, pools, blocks, rows):
        n = blocks.shape[0]
        parts = []
        for pool, geom in zip(pools, layout.pools):
            split = geom.split(layout.num_blocks)
            parts.append(_to_bytes(pool[_expand(blocks, split)], n))
        pad = layout.row_bytes - layout.payload_bytes
        if pad:
            parts.append(jnp.zeros((n, pad), jnp.int8))
        data = jnp.concatenate(parts, axis=1).reshape((n,) + layout.staging_shape[1:])
        return staging.at[rows].set(data)

    return pack


def make_unpack_fn(layout: StagingLayout) -> Callable:
    """`(pools, staging, rows, blocks) -> pools'`: block `blocks[i]` of every
    array gets its page back from staging row `rows[i]`."""

    def unpack(pools, staging, rows, blocks):
        n = rows.shape[0]
        data = staging[rows].reshape(n, layout.row_bytes)
        out = []
        for pool, geom, off, page in zip(
            pools, layout.pools, layout.offsets, layout.page_bytes
        ):
            split = geom.split(layout.num_blocks)
            vals = _from_bytes(data[:, off : off + page], n, geom, split)
            out.append(pool.at[_expand(blocks, split)].set(vals))
        return out

    return unpack


# -- worker side: the staging buffer and its torch_tpu programs ----------------


def _torch_dtype_name(dtype: torch.dtype) -> str:
    """The JAX dtype name of a torch dtype (`torch.float8_e4m3fn` ->
    `float8_e4m3fn`)."""
    return str(dtype).removeprefix("torch.")


def layout_from_tensors(
    pools: Sequence[torch.Tensor], num_blocks: int, rows: int
) -> StagingLayout:
    return StagingLayout(
        num_blocks=num_blocks,
        rows=rows,
        pools=tuple(
            PoolGeometry(
                shape=tuple(int(d) for d in pool.shape),
                dtype=_torch_dtype_name(pool.dtype),
                itemsize=pool.element_size(),
            )
            for pool in pools
        ),
    )


@functools.cache
def _device_programs(layout: StagingLayout):
    """The compiled pack and unpack of one layout (torch_tpu only)."""
    import jax
    import torch
    from torch_tpu._internal import pallas

    pack_fn = make_pack_fn(layout)
    unpack_fn = make_unpack_fn(layout)
    tag = layout.signature()

    def _pack(
        staging: jax.Array, pools: list[jax.Array], blocks: jax.Array, rows: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return pack_fn(staging, pools, blocks, rows), rows[0]

    def _unpack(
        pools: list[jax.Array], staging: jax.Array, rows: jax.Array, blocks: jax.Array
    ) -> tuple[list[jax.Array], jax.Array]:
        return unpack_fn(pools, staging, rows, blocks), blocks[0]

    pack_op = pallas.jax_op(
        f"pallas::raiden_staging_pack_{tag}", _pack, donate_argnums=(0,)
    )
    unpack_op = pallas.jax_op(
        f"pallas::raiden_staging_unpack_{tag}", _unpack, donate_argnums=(0,)
    )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def run_pack(staging, pools, blocks, rows):
        # Donation + copy_ writeback, aliased in place (as `copy_pool_blocks`).
        new_staging, marker = pack_op(staging, pools, blocks, rows)
        staging.copy_(new_staging)
        return marker

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def run_unpack(pools, staging, rows, blocks):
        new_pools, marker = unpack_op(pools, staging, rows, blocks)
        for pool, new_pool in zip(pools, new_pools):
            pool.copy_(new_pool)
        return marker

    return run_pack, run_unpack


class StagingBuffer:
    """Worker-side staging rows and the programs that fill and drain them.

    `pools` are the KV arrays in their native shape and dtype (one per
    storage, in registration order); the programs take them as arguments,
    never a reinterpreting view, so torch_tpu hands over the buffers
    themselves.
    """

    def __init__(
        self,
        pools: Sequence[torch.Tensor],
        *,
        num_blocks: int,
        rows: int,
        device: torch.device,
        programs: Callable[[StagingLayout], tuple[Callable, Callable]] | None = None,
    ):
        import torch

        self.layout = layout_from_tensors(pools, num_blocks, rows)
        self.pools = list(pools)
        self.device = device
        self.tensor = torch.zeros(
            self.layout.staging_shape, dtype=torch.int8, device=device
        )
        self._programs = programs or _device_programs

    @property
    def row_bytes(self) -> int:
        return self.layout.row_bytes

    def _indices(self, values: list[int]) -> torch.Tensor:
        import torch

        return torch.tensor(values, dtype=torch.int32).to(
            self.device, non_blocking=True
        )

    def pack(self, blocks: Sequence[int], rows: Sequence[int]) -> None:
        """Staging row `rows[i]` <- every page of block `blocks[i]`."""
        if not blocks:
            return
        assert len(set(rows)) == len(rows), "a pack writes each row once"
        import torch

        blocks_p, rows_p = pad_pairs(blocks, rows)
        run_pack, _ = self._programs(self.layout)
        with torch.no_grad():
            run_pack(
                self.tensor, self.pools, self._indices(blocks_p), self._indices(rows_p)
            )

    def unpack(self, rows: Sequence[int], blocks: Sequence[int]) -> None:
        """Every page of block `blocks[i]` <- staging row `rows[i]`."""
        if not rows:
            return
        assert len(set(blocks)) == len(blocks), "an unpack writes each block once"
        self._run_unpack(*pad_pairs(rows, blocks))

    def _run_unpack(self, rows: list[int], blocks: list[int]) -> None:
        import torch

        _, run_unpack = self._programs(self.layout)
        with torch.no_grad():
            run_unpack(
                self.pools, self.tensor, self._indices(rows), self._indices(blocks)
            )

    def precompile(self) -> None:
        """Compiles every bucket while the caches are quiescent.

        Packs block 0 into rows `0..k-1`, then unpacks those rows into block
        0: the unpack writes back the bytes the pack just read, so no cache
        content changes.
        """
        for n in bucket_lengths(self.layout.rows):
            k = min(n, self.layout.rows)
            self.pack([0] * k, list(range(k)))
            rows, blocks = pad_pairs(list(range(k)), [0] * k)
            self._run_unpack(rows, blocks)


# -- scheduler side: rows, parts, acks -----------------------------------------


class StagingRows:
    """Free list of staging rows (any rows will do; contiguity buys nothing,
    the host slots are scattered anyway)."""

    def __init__(self, num_rows: int):
        assert num_rows > 0
        self.num_rows = num_rows
        self._free: list[int] = list(range(num_rows - 1, -1, -1))
        self._taken: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def take(self, n: int) -> list[int] | None:
        assert 0 < n <= self.num_rows, (n, self.num_rows)
        if n > len(self._free):
            return None
        rows = [self._free.pop() for _ in range(n)]
        self._taken.update(rows)
        return rows

    def give(self, rows: Sequence[int]) -> None:
        for row in rows:
            assert row in self._taken, f"staging row {row} returned twice"
            self._taken.remove(row)
            self._free.append(row)


@dataclass
class _Part:
    part_id: int
    base_job_id: int
    is_store: bool
    keys: list[OffloadKey]
    device_block_ids: list[int]
    rows: list[int] | None = None
    # store: queued -> fencing -> launched; cancelled while fencing.
    # load: queued -> launched -> unpacking.
    state: str = "queued"
    acks: int = 0


@dataclass
class _BaseJob:
    job_id: int
    is_store: bool
    req_id: str | None = None
    # Parts not resolved yet (queued, fencing, launched).
    outstanding: set[int] = field(default_factory=set)
    failed: bool = False


@dataclass
class StagedStepOutput:
    """What one scheduler step ships to the workers and echoes to vLLM."""

    # part id -> (device blocks, rows) to pack before the part's save fence.
    packs: dict[int, tuple[list[int], list[int]]] = field(default_factory=dict)
    # part id -> (rows, device blocks) to unpack; acked as `unpacked_jobs`.
    unpacks: dict[int, tuple[list[int], list[int]]] = field(default_factory=dict)
    finished_store_job_ids: list[int] = field(default_factory=list)
    finished_load_jobs: dict[int, str] = field(default_factory=dict)
    failed_load_req_ids: set[str] = field(default_factory=set)


class StagedTransferScheduler:
    """Scheduler-side owner of the staging rows.

    Splits vLLM's store and load jobs into parts of at most `max_part_rows`
    blocks, gives each part rows, drives the manager with the part's rows as
    the device block ids, and turns part completions back into base-job
    completions once all of a job's parts resolved.

    `manager` is the `RaidenOffloadingManager` (only `submit_load_job`,
    `submit_store_job`, `poll_finished_jobs`, `on_store_job_cancelled` and
    `abandon_load` are used).
    """

    def __init__(self, manager: Any, num_rows: int, num_workers: int):
        self._manager = manager
        self.rows = StagingRows(num_rows)
        self._num_workers = num_workers
        # Half the rows at most, so a load can always start while stores hold
        # the rest.
        self.max_part_rows = max(1, num_rows // 2)
        self._next_part_id = _PART_ID_BASE
        self._parts: dict[int, _Part] = {}
        self._jobs: dict[int, _BaseJob] = {}
        self._load_queue: deque[int] = deque()
        self._store_queue: deque[int] = deque()
        # Unpinned loads: failed by the manager at once, no rows needed.
        self._unpinned: list[_Part] = []

    # -- intake ------------------------------------------------------------

    def is_part(self, job_id: int) -> bool:
        return job_id in self._parts

    def owns_job(self, job_id: int) -> bool:
        return job_id in self._jobs

    def _split(
        self,
        job_id: int,
        is_store: bool,
        keys: list[OffloadKey],
        device_block_ids: list[int],
    ) -> list[_Part]:
        assert len(keys) == len(device_block_ids), (len(keys), len(device_block_ids))
        parts = []
        for start in range(0, len(keys), self.max_part_rows):
            part = _Part(
                part_id=self._next_part_id,
                base_job_id=job_id,
                is_store=is_store,
                keys=keys[start : start + self.max_part_rows],
                device_block_ids=device_block_ids[start : start + self.max_part_rows],
            )
            self._next_part_id += 1
            self._parts[part.part_id] = part
            parts.append(part)
        return parts

    def add_store_job(
        self, job_id: int, keys: list[OffloadKey], device_block_ids: list[int]
    ) -> None:
        job = _BaseJob(job_id=job_id, is_store=True)
        self._jobs[job_id] = job
        for part in self._split(job_id, True, keys, device_block_ids):
            job.outstanding.add(part.part_id)
            self._store_queue.append(part.part_id)

    def add_load_job(
        self,
        job_id: int,
        keys: list[OffloadKey],
        device_block_ids: list[int],
        req_id: str,
        pinned: bool = True,
    ) -> None:
        job = _BaseJob(job_id=job_id, is_store=False, req_id=req_id)
        self._jobs[job_id] = job
        if not pinned:
            # prepare_load holds nothing: one part, failed by the manager
            # without touching the store.
            part = _Part(
                part_id=self._next_part_id,
                base_job_id=job_id,
                is_store=False,
                keys=keys,
                device_block_ids=device_block_ids,
                state="launched",
            )
            self._next_part_id += 1
            self._parts[part.part_id] = part
            job.outstanding.add(part.part_id)
            self._manager.submit_load_job(
                part.part_id, keys, device_block_ids, req_id, pinned=False
            )
            return
        for part in self._split(job_id, False, keys, device_block_ids):
            job.outstanding.add(part.part_id)
            self._load_queue.append(part.part_id)

    # -- flush ---------------------------------------------------------------

    def flush(self, job_ids: set[int], out: StagedStepOutput) -> None:
        """Source blocks of these store jobs are about to be reused.

        A part not yet packed is cancelled. A packed part (its pack was
        shipped in an earlier step) is cancelled too; its rows come back once
        every rank reports the pack retired. A launched part keeps going: its
        save reads the staging rows, not the reused blocks, so nothing waits.
        """
        for job_id in job_ids:
            job = self._jobs.get(job_id)
            if job is None or not job.is_store:
                continue
            for part_id in list(job.outstanding):
                part = self._parts[part_id]
                if part.state == "queued":
                    self._store_queue.remove(part_id)
                    del self._parts[part_id]
                elif part.state == "fencing":
                    part.state = "cancelled"
                else:
                    continue
                self._manager.on_store_job_cancelled(part.keys)
                job.outstanding.discard(part_id)
            self._maybe_finish(job, out)

    # -- per step ------------------------------------------------------------

    def step(self, out: StagedStepOutput) -> None:
        """Launch what fits, then collect what finished."""
        self._pump(out)
        for finished in self._manager.poll_finished_jobs():
            self._on_part_finished(finished, out)
        # Rows freed by failures above may let more start this step.
        self._pump(out)

    def _pump(self, out: StagedStepOutput) -> None:
        while self._load_queue:
            part = self._parts[self._load_queue[0]]
            rows = self.rows.take(len(part.keys))
            if rows is None:
                # Loads first: stores wait until the head load fits.
                return
            self._load_queue.popleft()
            part.rows = rows
            part.state = "launched"
            job = self._jobs[part.base_job_id]
            self._manager.submit_load_job(part.part_id, part.keys, rows, job.req_id)
        while self._store_queue:
            part = self._parts[self._store_queue[0]]
            rows = self.rows.take(len(part.keys))
            if rows is None:
                return
            self._store_queue.popleft()
            part.rows = rows
            part.state = "fencing"
            out.packs[part.part_id] = (list(part.device_block_ids), list(rows))

    def _on_part_finished(self, finished: Any, out: StagedStepOutput) -> None:
        part = self._parts.get(finished.job_id)
        assert part is not None, f"manager finished unknown job {finished.job_id}"
        job = self._jobs[part.base_job_id]
        job.outstanding.discard(part.part_id)
        if part.is_store:
            # The save no longer reads the rows, success or not.
            self._release(part)
        elif finished.success and not job.failed:
            part.state = "unpacking"
            assert part.rows is not None
            out.unpacks[part.part_id] = (list(part.rows), list(part.device_block_ids))
        else:
            # Nothing reads a failed part's rows; every transfer into them
            # resolved before the manager reported it.
            self._release(part)
            if not finished.success:
                job.failed = True
                self._abandon_queued(job)
        self._maybe_finish(job, out)

    def _abandon_queued(self, job: _BaseJob) -> None:
        """A failed load's parts that never launched: give back their pins."""
        for part_id in list(job.outstanding):
            part = self._parts[part_id]
            if part.state != "queued":
                continue
            self._load_queue.remove(part_id)
            job.outstanding.discard(part_id)
            del self._parts[part_id]
            self._manager.abandon_load(part.keys)

    def _maybe_finish(self, job: _BaseJob, out: StagedStepOutput) -> None:
        if job.outstanding:
            return
        del self._jobs[job.job_id]
        if job.is_store:
            out.finished_store_job_ids.append(job.job_id)
        else:
            assert job.req_id is not None
            out.finished_load_jobs[job.job_id] = job.req_id
            if job.failed:
                out.failed_load_req_ids.add(job.req_id)

    def _release(self, part: _Part) -> None:
        if part.rows is not None:
            self.rows.give(part.rows)
            part.rows = None
        self._parts.pop(part.part_id, None)

    # -- worker acks ---------------------------------------------------------

    def on_fenced(self, fenced: Mapping[int, int]) -> None:
        """Pack fences retired on `count` more ranks; launch the saves of the
        parts every rank has packed."""
        for part_id, count in fenced.items():
            part = self._parts.get(part_id)
            if part is None or part.state not in ("fencing", "cancelled"):
                continue
            part.acks += count
            assert part.acks <= self._num_workers
            if part.acks < self._num_workers:
                continue
            if part.state == "cancelled":
                self._release(part)
                continue
            part.state = "launched"
            part.acks = 0
            assert part.rows is not None
            self._manager.submit_store_job(part.part_id, part.keys, list(part.rows))

    def on_unpacked(self, unpacked: Mapping[int, int]) -> None:
        """Unpacks retired on `count` more ranks; free fully retired rows."""
        for part_id, count in unpacked.items():
            part = self._parts.get(part_id)
            if part is None or part.state != "unpacking":
                continue
            part.acks += count
            assert part.acks <= self._num_workers
            if part.acks == self._num_workers:
                self._release(part)

    def has_pending_work(self) -> bool:
        return bool(self._parts or self._jobs)
