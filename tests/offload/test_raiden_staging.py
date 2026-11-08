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
"""Staged multi-shape KV offload (`raiden_staging`), on CPU.

1. Exactness: a block saved and loaded through staging rows lands with the
   same bytes as through today's one-slot-per-array path, for every array
   shape MiniMax M3 registers (MSA K/V int4 G=1 / G=4 with folded scales,
   fp8 token-sharded, bf16 index keys, fp8 drafter K/V with several kernel
   rows per block), at DCP2 and DCP8 per-rank geometries. Raiden is modelled
   as a raw byte copy between a device row and a host slot. Fault controls
   show the comparison catches a corrupted row, a shifted page offset and a
   swapped array order.
2. Scheduler: rows are the device ids Raiden sees, a load ships its unpack
   in the step it is echoed finished, rows come back only after every rank
   acks, failures give back rows and pins, a flushed store is never drained.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from tests.offload import test_raiden_store_offload as base
from tests.offload.test_raiden_store_offload import (
    FakeKVCacheStore,
    _key,
    make_manager,
    seed_host_entries,
    sub_hashes,
)
from vllm_torchtpu.offload import raiden_staging as rs

NUM_BLOCKS = 12


def _minimax_geometries(tokens: int) -> list[tuple[tuple[int, ...], str]]:
    """Per-rank arrays of one MiniMax M3 layer pair, `tokens` per block."""
    return [
        # MSA K/V int4, G=1, scales folded in one extra word row.
        ((NUM_BLOCKS, tokens // 8 + 1, 4, 256), "uint8"),
        # MSA K/V int4, G=4.
        ((NUM_BLOCKS, tokens // 8 + 1, 4, 1024), "uint8"),
        # Index keys int4 without folded scales.
        ((NUM_BLOCKS, tokens // 8, 4, 128), "uint8"),
        # MSA K/V fp8, token-sharded (4 tokens per 32-bit word).
        ((NUM_BLOCKS, tokens // 4, 4, 512), "float8_e4m3fn"),
        # Index keys bf16.
        ((NUM_BLOCKS, tokens, 128), "bfloat16"),
        # Drafter K/V fp8, two kernel rows per scheduler block.
        ((NUM_BLOCKS * 2, tokens // 2, 2, 1, 128), "float8_e4m3fn"),
    ]


def _random_pool(shape, dtype, rng) -> np.ndarray:
    raw = rng.integers(
        0,
        256,
        size=(
            int(np.prod(shape)) * np.dtype(getattr(ml_dtypes, dtype, dtype)).itemsize,
        ),
        dtype=np.uint8,
    )
    return raw.view(getattr(ml_dtypes, dtype, np.dtype(dtype))).reshape(shape)


def _layout(pools: list[np.ndarray], rows: int) -> rs.StagingLayout:
    return rs.StagingLayout(
        num_blocks=NUM_BLOCKS,
        rows=rows,
        pools=tuple(
            rs.PoolGeometry(shape=p.shape, dtype=str(p.dtype), itemsize=p.itemsize)
            for p in pools
        ),
    )


def _block_bytes(pool: np.ndarray, block: int) -> bytes:
    split = pool.shape[0] // NUM_BLOCKS
    return pool[block * split : (block + 1) * split].tobytes()


class _LegacyPath:
    """Today's path: one host slot per array per block, raw page copies."""

    def __init__(self, pools: list[np.ndarray]):
        self.pools = pools
        self.host: dict[int, list[bytes]] = {}

    def save(self, blocks, slots):
        for block, slot in zip(blocks, slots):
            self.host[slot] = [_block_bytes(p, block) for p in self.pools]

    def load(self, dst_pools, slots, blocks):
        out = []
        for i, pool in enumerate(dst_pools):
            pool = pool.copy()
            split = pool.shape[0] // NUM_BLOCKS
            for slot, block in zip(slots, blocks):
                page = np.frombuffer(self.host[slot][i], dtype=pool.dtype)
                pool[block * split : (block + 1) * split] = page.reshape(
                    (split,) + pool.shape[1:]
                )
            out.append(pool)
        return out


class _StagedPath:
    """Staging rows; Raiden copies a whole row's raw bytes to a host slot."""

    def __init__(self, pools: list[np.ndarray], rows: int, layout=None):
        self.layout = layout or _layout(pools, rows)
        self.pack = jax.jit(rs.make_pack_fn(self.layout))
        self.unpack = jax.jit(rs.make_unpack_fn(self.layout))
        self.staging = jnp.zeros(self.layout.staging_shape, jnp.int8)
        self.host: dict[int, bytes] = {}

    def save(self, pools, blocks, slots, rows):
        b, r = rs.pad_pairs(blocks, rows)
        self.staging = self.pack(
            self.staging,
            [jnp.asarray(p) for p in pools],
            jnp.asarray(b, jnp.int32),
            jnp.asarray(r, jnp.int32),
        )
        staging = np.asarray(self.staging)
        for row, slot in zip(rows, slots):
            self.host[slot] = staging[row].tobytes()

    def load(self, dst_pools, slots, blocks, rows):
        staging = np.asarray(self.staging).copy()
        for row, slot in zip(rows, slots):
            staging[row] = np.frombuffer(self.host[slot], np.int8).reshape(
                staging.shape[1:]
            )
        self.staging = jnp.asarray(staging)
        r, b = rs.pad_pairs(rows, blocks)
        out = self.unpack(
            [jnp.asarray(p) for p in dst_pools],
            self.staging,
            jnp.asarray(r, jnp.int32),
            jnp.asarray(b, jnp.int32),
        )
        return [np.asarray(p) for p in out]


class TestStagingExactness(unittest.TestCase):
    def _pools(self, tokens, seed=0):
        rng = np.random.default_rng(seed)
        return [_random_pool(s, d, rng) for s, d in _minimax_geometries(tokens)]

    def _assert_same_bytes(self, got, want):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(g.dtype, w.dtype)
            self.assertEqual(g.shape, w.shape)
            self.assertEqual(g.tobytes(), w.tobytes())

    def _round_trip(self, tokens, src_blocks, dst_blocks, rows_cap=8):
        pools = self._pools(tokens)
        # Destination: other content everywhere, so an unwritten page shows.
        dst = self._pools(tokens, seed=1)
        slots = [100 + i for i in range(len(src_blocks))]
        legacy = _LegacyPath(pools)
        legacy.save(src_blocks, slots)
        want = legacy.load(dst, slots, dst_blocks)

        staged = _StagedPath(pools, rows_cap)
        rows = list(range(len(src_blocks)))[::-1]
        staged.save(pools, src_blocks, slots, rows)
        # Load through different rows than the save used.
        load_rows = [(r + 3) % rows_cap for r in range(len(src_blocks))]
        got = staged.load(dst, slots, dst_blocks, load_rows)
        return pools, dst, want, got

    def test_matches_legacy_dcp2(self):
        # DCP2: 64 tokens per rank-local block.
        _, dst, want, got = self._round_trip(64, [3, 7, 1], [5, 9, 2])
        self._assert_same_bytes(got, want)

    def test_matches_legacy_dcp8(self):
        # DCP8: 16 tokens per rank-local block.
        _, _, want, got = self._round_trip(16, [11, 4], [6, 10])
        self._assert_same_bytes(got, want)

    def test_single_block_and_bucket_padding(self):
        # One pair pads to a bucket of 8 by repeating itself.
        pools, dst, want, got = self._round_trip(64, [2], [8])
        self._assert_same_bytes(got, want)
        # Pages of every other block, the null block 0 included, untouched.
        for g, d in zip(got, dst):
            for block in range(NUM_BLOCKS):
                if block != 8:
                    self.assertEqual(_block_bytes(g, block), _block_bytes(d, block))

    def test_loaded_pages_equal_source_pages(self):
        pools, _, _, got = self._round_trip(64, [3, 7, 1], [5, 9, 2])
        for g, p in zip(got, pools):
            for s, d in zip([3, 7, 1], [5, 9, 2]):
                self.assertEqual(_block_bytes(g, d), _block_bytes(p, s))

    def test_row_layout_is_pages_back_to_back(self):
        pools = self._pools(64)
        staged = _StagedPath(pools, 4)
        staged.save(pools, [5], [0], [2])
        row = staged.host[0]
        layout = staged.layout
        self.assertEqual(len(row), layout.row_bytes)
        self.assertEqual(layout.row_bytes % (32 * 128), 0)
        for pool, off, page in zip(pools, layout.offsets, layout.page_bytes):
            self.assertEqual(row[off : off + page], _block_bytes(pool, 5))
        self.assertEqual(
            row[layout.payload_bytes :], bytes(layout.row_bytes - layout.payload_bytes)
        )

    # -- fault controls: the comparison above can fail ----------------------

    def test_fault_corrupted_host_row_is_caught(self):
        pools = self._pools(64)
        dst = self._pools(64, seed=1)
        legacy = _LegacyPath(pools)
        legacy.save([3], [0])
        want = legacy.load(dst, [0], [5])
        staged = _StagedPath(pools, 4)
        staged.save(pools, [3], [0], [0])
        row = bytearray(staged.host[0])
        row[staged.layout.offsets[4] + 7] ^= 0x01
        staged.host[0] = bytes(row)
        got = staged.load(dst, [0], [5], [1])
        self.assertNotEqual(got[4].tobytes(), want[4].tobytes())
        for i in (0, 1, 2, 3, 5):
            self.assertEqual(got[i].tobytes(), want[i].tobytes())

    def test_fault_shifted_offset_is_caught(self):
        pools = self._pools(64)
        dst = self._pools(64, seed=1)
        legacy = _LegacyPath(pools)
        legacy.save([3], [0])
        want = legacy.load(dst, [0], [5])
        staged = _StagedPath(pools, 4)
        staged.save(pools, [3], [0], [0])
        # Unpack with page offsets one byte off.
        bad = _StagedPath(pools, 4)
        bad.host = staged.host
        layout = bad.layout
        object.__setattr__(
            layout, "offsets", tuple(o + 1 if o else o for o in layout.offsets)
        )
        bad.unpack = jax.jit(rs.make_unpack_fn(layout))
        got = bad.load(dst, [0], [5], [1])
        self.assertNotEqual([g.tobytes() for g in got], [w.tobytes() for w in want])

    def test_fault_swapped_array_order_is_caught(self):
        pools = self._pools(64)
        dst = self._pools(64, seed=1)
        staged = _StagedPath(pools, 4)
        staged.save(pools, [3], [0], [0])
        # A reader that orders the two MSA arrays the other way round.
        bad = _StagedPath([pools[1], pools[0], *pools[2:]], 4)
        bad.host = staged.host
        got = bad.load([dst[1], dst[0], *dst[2:]], [0], [5], [1])
        self.assertNotEqual(_block_bytes(got[0], 5), _block_bytes(pools[1], 3))
        self.assertNotEqual(_block_bytes(got[1], 5), _block_bytes(pools[0], 3))


class TestStagingHelpers(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(rs.parse_staging_blocks({}), 0)
        self.assertEqual(rs.parse_staging_blocks({"staging_blocks": 128}), 128)
        self.assertEqual(rs.parse_staging_blocks({"staging_blocks": "64"}), 64)
        for bad in (-1, "x", True, 1.5, "1.5"):
            with self.assertRaises(ValueError):
                rs.parse_staging_blocks({"staging_blocks": bad})

    def test_buckets(self):
        self.assertEqual(rs.bucket_len(1), 8)
        self.assertEqual(rs.bucket_len(9), 32)
        self.assertEqual(rs.bucket_lengths(100), [8, 32, 128])
        self.assertEqual(
            rs.pad_pairs([4, 5], [1, 2]), ([4, 5] + [4] * 6, [1, 2] + [1] * 6)
        )

    def test_rows_never_double_allocated(self):
        rows = rs.StagingRows(4)
        a = rows.take(3)
        self.assertIsNone(rows.take(2))
        b = rows.take(1)
        self.assertEqual(sorted(a + b), [0, 1, 2, 3])
        rows.give(a)
        with self.assertRaises(AssertionError):
            rows.give(a)


def _torch_programs(layout):
    """Stand-in for the torch_tpu programs: the same JAX functions, run on
    CPU over byte views of the torch tensors, written back in place."""

    def to_jax(t: torch.Tensor):
        name = str(t.dtype).removeprefix("torch.")
        dtype = getattr(ml_dtypes, name, None) or np.dtype(name)
        raw = t.contiguous().view(torch.uint8).numpy()
        return jnp.asarray(raw.view(dtype).reshape(t.shape))

    def write_back(t: torch.Tensor, arr) -> None:
        raw = np.array(arr).view(np.uint8).reshape(t.view(torch.uint8).shape)
        t.view(torch.uint8).copy_(torch.from_numpy(raw))

    pack_fn = rs.make_pack_fn(layout)
    unpack_fn = rs.make_unpack_fn(layout)

    def run_pack(staging, pools, blocks, rows):
        new = pack_fn(
            to_jax(staging),
            [to_jax(p) for p in pools],
            jnp.asarray(blocks.numpy()),
            jnp.asarray(rows.numpy()),
        )
        write_back(staging, new)

    def run_unpack(pools, staging, rows, blocks):
        new = unpack_fn(
            [to_jax(p) for p in pools],
            to_jax(staging),
            jnp.asarray(rows.numpy()),
            jnp.asarray(blocks.numpy()),
        )
        for p, n in zip(pools, new):
            write_back(p, n)

    return run_pack, run_unpack


class TestStagingBuffer(unittest.TestCase):
    def _pools(self, seed):
        g = torch.Generator().manual_seed(seed)
        shapes = [
            ((NUM_BLOCKS, 9, 4, 256), torch.uint8),
            ((NUM_BLOCKS, 16, 4, 512), torch.float8_e4m3fn),
            ((NUM_BLOCKS, 64, 128), torch.bfloat16),
            ((NUM_BLOCKS * 2, 32, 2, 1, 128), torch.float8_e4m3fn),
        ]
        out = []
        for shape, dtype in shapes:
            n = int(np.prod(shape)) * torch.empty(0, dtype=dtype).element_size()
            raw = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
            out.append(raw.view(dtype).reshape(shape))
        return out

    def test_pack_unpack_moves_exact_pages(self):
        src = self._pools(0)
        buf = rs.StagingBuffer(
            src,
            num_blocks=NUM_BLOCKS,
            rows=16,
            device=torch.device("cpu"),
            programs=_torch_programs,
        )
        self.assertEqual(buf.tensor.shape, buf.layout.staging_shape)
        buf.pack([4, 9, 2], [10, 0, 5])
        # Another engine's pools receive the rows (Raiden moved the bytes).
        dst = self._pools(1)
        before = [p.clone() for p in dst]
        other = rs.StagingBuffer(
            dst,
            num_blocks=NUM_BLOCKS,
            rows=16,
            device=torch.device("cpu"),
            programs=_torch_programs,
        )
        other.tensor.copy_(buf.tensor)
        other.unpack([10, 0, 5], [1, 6, 11])
        for s, d, b in zip(src, dst, before):
            split = s.shape[0] // NUM_BLOCKS
            for sb, db in ((4, 1), (9, 6), (2, 11)):
                self.assertTrue(
                    torch.equal(
                        d[db * split : (db + 1) * split].view(torch.uint8),
                        s[sb * split : (sb + 1) * split].view(torch.uint8),
                    )
                )
            for blk in set(range(NUM_BLOCKS)) - {1, 6, 11}:
                self.assertTrue(
                    torch.equal(
                        d[blk * split : (blk + 1) * split].view(torch.uint8),
                        b[blk * split : (blk + 1) * split].view(torch.uint8),
                    )
                )

    def test_precompile_changes_nothing(self):
        pools = self._pools(0)
        before = [p.clone() for p in pools]
        calls = []

        def programs(layout):
            run_pack, run_unpack = _torch_programs(layout)

            def pack(staging, p, blocks, rows):
                calls.append(("pack", blocks.numel()))
                run_pack(staging, p, blocks, rows)

            def unpack(p, staging, rows, blocks):
                calls.append(("unpack", rows.numel()))
                run_unpack(p, staging, rows, blocks)

            return pack, unpack

        buf = rs.StagingBuffer(
            pools,
            num_blocks=NUM_BLOCKS,
            rows=40,
            device=torch.device("cpu"),
            programs=programs,
        )
        buf.precompile()
        self.assertEqual(
            calls,
            [
                ("pack", 8),
                ("unpack", 8),
                ("pack", 32),
                ("unpack", 32),
                ("pack", 128),
                ("unpack", 128),
            ],
        )
        for p, b in zip(pools, before):
            self.assertTrue(torch.equal(p.view(torch.uint8), b.view(torch.uint8)))


# -- scheduler side ------------------------------------------------------------


def _staged(store=None, rows=4, num_workers=2):
    store = store or FakeKVCacheStore()
    manager = make_manager(store, device_block_size_factor=1, num_blocks=64)
    return store, manager, rs.StagedTransferScheduler(manager, rows, num_workers)


def _step(staged):
    out = rs.StagedStepOutput()
    staged.step(out)
    return out


class TestStagedLoads(unittest.TestCase):
    def _prepared(self, store, manager, n):
        keys = [_key(i) for i in range(n)]
        for k in keys:
            seed_host_entries(store, k, 1)
        spec = manager.prepare_load(keys, SimpleNamespace(req_id="r"))
        return keys, spec

    def test_load_uses_rows_unpacks_on_finish_frees_after_all_acks(self):
        store, manager, staged = _staged(rows=4)
        keys, spec = self._prepared(store, manager, 2)
        staged.add_load_job(7, keys, [30, 31], "r", pinned=spec.pinned)
        out = _step(staged)
        (load,) = store.calls_named("load")
        rows = load.args[1]
        self.assertEqual(sorted(rows), sorted(set(rows)))
        self.assertTrue(all(0 <= r < 4 for r in rows))
        self.assertEqual(out.unpacks, {})
        self.assertEqual(out.finished_load_jobs, {})

        store.complete_load(load.args[0])
        out = _step(staged)
        ((part_id, (unpack_rows, blocks)),) = out.unpacks.items()
        self.assertEqual((unpack_rows, blocks), (rows, [30, 31]))
        self.assertEqual(out.finished_load_jobs, {7: "r"})
        self.assertEqual(out.failed_load_req_ids, set())
        self.assertEqual(staged.rows.num_free, 2)
        staged.on_unpacked({part_id: 1})
        self.assertEqual(staged.rows.num_free, 2)
        staged.on_unpacked({part_id: 1})
        self.assertEqual(staged.rows.num_free, 4)
        self.assertEqual(store.pinned_hashes(), {})
        self.assertFalse(staged.has_pending_work())

    def test_long_load_runs_in_parts_and_finishes_once(self):
        store, manager, staged = _staged(rows=4)  # parts of 2
        keys, spec = self._prepared(store, manager, 5)
        staged.add_load_job(1, keys, [10, 11, 12, 13, 14], "r")
        _step(staged)
        loads = store.calls_named("load")
        self.assertEqual([len(c.args[0]) for c in loads], [2, 2])
        finished = {}
        unpacked = {}
        for _ in range(6):
            for c in store.calls_named("load"):
                if any(h in store._loading for h in c.args[0]):
                    store.complete_load(c.args[0])
            out = _step(staged)
            finished.update(out.finished_load_jobs)
            for pid, (rows, blocks) in out.unpacks.items():
                unpacked[pid] = blocks
                staged.on_unpacked({pid: 2})
        self.assertEqual(finished, {1: "r"})
        self.assertEqual(
            sorted(b for v in unpacked.values() for b in v), [10, 11, 12, 13, 14]
        )
        self.assertEqual(staged.rows.num_free, 4)
        self.assertEqual(store.pinned_hashes(), {})

    def test_failed_part_frees_rows_and_abandons_queued_parts(self):
        store, manager, staged = _staged(rows=2)  # parts of 1
        keys, spec = self._prepared(store, manager, 4)
        staged.add_load_job(1, keys, [10, 11, 12, 13], "r")
        _step(staged)
        loads = store.calls_named("load")
        self.assertEqual(len(loads), 2)
        store.complete_load(loads[0].args[0], success=False)
        store.complete_load(loads[1].args[0])
        out = _step(staged)
        self.assertEqual(out.finished_load_jobs, {1: "r"})
        self.assertEqual(out.failed_load_req_ids, {"r"})
        self.assertEqual(out.unpacks, {})
        self.assertEqual(staged.rows.num_free, 2)
        # The queued parts never launched and hold no pins.
        self.assertEqual(len(store.calls_named("load")), 2)
        self.assertEqual(store.pinned_hashes(), {})

    def test_unpinned_load_fails_without_rows(self):
        store, manager, staged = _staged(rows=2)
        staged.add_load_job(3, [_key(1)], [9], "r", pinned=False)
        out = _step(staged)
        self.assertEqual(out.finished_load_jobs, {3: "r"})
        self.assertEqual(out.failed_load_req_ids, {"r"})
        self.assertEqual(store.calls_named("load"), [])
        self.assertEqual(staged.rows.num_free, 2)

    def test_remote_part_reads_into_rows(self):
        store, manager, staged = _staged(rows=4)
        key = _key(5)
        (h,) = sub_hashes(key, 1)
        store.storage_entries.add(h)
        manager.lookup(key, SimpleNamespace(req_id="r"))
        spec = manager.prepare_load([key], SimpleNamespace(req_id="r"))
        staged.add_load_job(2, [key], [20], "r", pinned=spec.pinned)
        _step(staged)
        (read,) = store.calls_named("read_remote")
        self.assertTrue(0 <= read.args[2][0] < 4)
        store.complete_remote_read([h])
        out = _step(staged)
        self.assertEqual(list(out.unpacks.values()), [(read.args[2], [20])])


class TestStagedStores(unittest.TestCase):
    def test_store_packs_then_saves_rows_after_all_fences(self):
        store, manager, staged = _staged(rows=4)
        keys = [_key(1), _key(2)]
        manager.prepare_store(keys, SimpleNamespace(req_id="r"))
        staged.add_store_job(5, keys, [40, 41])
        out = _step(staged)
        ((part_id, (blocks, rows)),) = out.packs.items()
        self.assertEqual(blocks, [40, 41])
        self.assertEqual(store.calls_named("insert"), [])
        staged.on_fenced({part_id: 1})
        self.assertEqual(store.calls_named("insert"), [])
        staged.on_fenced({part_id: 1})
        (insert,) = store.calls_named("insert")
        self.assertEqual([s.device_block_id for s in insert.args[1]], rows)
        (save,) = store.calls_named("save")
        store.complete_save(save.args[0])
        out = _step(staged)
        self.assertEqual(out.finished_store_job_ids, [5])
        self.assertEqual(staged.rows.num_free, 4)

    def test_loads_take_rows_before_stores(self):
        store, manager, staged = _staged(rows=2)
        staged.add_store_job(5, [_key(1)], [40])
        key = _key(9)
        seed_host_entries(store, key, 1)
        manager.prepare_load([key], SimpleNamespace(req_id="r"))
        staged.add_load_job(6, [key], [50], "r")
        out = _step(staged)
        self.assertEqual(len(store.calls_named("load")), 1)
        self.assertEqual(len(out.packs), 1)
        self.assertEqual(staged.rows.num_free, 0)

    def test_flush_cancels_unlaunched_and_never_drains_launched(self):
        store, manager, staged = _staged(rows=2)  # parts of 1
        keys = [_key(i) for i in range(3)]
        manager.prepare_store(keys, SimpleNamespace(req_id="r"))
        staged.add_store_job(5, keys, [40, 41, 42])
        out = _step(staged)
        (p_launch, p_fence) = sorted(out.packs)
        staged.on_fenced({p_launch: 2})
        self.assertEqual(len(store.calls_named("save")), 1)
        # Part 3 is still queued (no rows); part 2 is fencing.
        with patch.object(manager, "drain_jobs") as drain:
            out = rs.StagedStepOutput()
            staged.flush({5}, out)
        drain.assert_not_called()
        self.assertEqual(out.finished_store_job_ids, [])
        self.assertNotIn(keys[1], manager._inflight_store_keys)
        self.assertNotIn(keys[2], manager._inflight_store_keys)
        # The fencing part's rows return only once its pack retired everywhere.
        self.assertEqual(staged.rows.num_free, 0)
        staged.on_fenced({p_fence: 2})
        self.assertEqual(staged.rows.num_free, 1)
        self.assertEqual(len(store.calls_named("save")), 1)
        # The launched save completes normally, then the job is finished.
        (save,) = store.calls_named("save")
        store.complete_save(save.args[0])
        out = _step(staged)
        self.assertEqual(out.finished_store_job_ids, [5])
        self.assertEqual(staged.rows.num_free, 2)

    def test_store_created_and_flushed_same_step(self):
        store, manager, staged = _staged(rows=2)
        staged.add_store_job(5, [_key(1)], [40])
        out = rs.StagedStepOutput()
        staged.flush({5}, out)
        staged.step(out)
        self.assertEqual(out.finished_store_job_ids, [5])
        self.assertEqual(out.packs, {})
        self.assertEqual(staged.rows.num_free, 2)


class TestStagedConnector(unittest.TestCase):
    """The connector's staged build_connector_meta and ack routing."""

    _make_scheduler = base.TestSchedulerFenceFlow._make_scheduler
    _stock_meta = base.TestSchedulerFenceFlow._stock_meta
    _store_job = base.TestSchedulerFenceFlow._store_job
    _load_job = base.TestSchedulerFenceFlow._load_job
    _build_meta = base.TestSchedulerFenceFlow._build_meta

    def _make_staged_scheduler(self, rows=4):
        store = FakeKVCacheStore()
        manager = make_manager(store, device_block_size_factor=1, num_blocks=64)
        sched = self._make_scheduler(manager)
        sched._staged = rs.StagedTransferScheduler(manager, rows, 2)
        return store, manager, sched

    def test_meta_ships_packs_as_fences_and_unpacks_with_finish(self):
        from vllm_torchtpu.offload.raiden_connector import (
            RaidenOffloadingWorkerMetadata,
        )

        store, manager, sched = self._make_staged_scheduler()
        key = _key(3)
        seed_host_entries(store, key, 1)
        manager.prepare_load([key], SimpleNamespace(req_id="r2"))
        manager.prepare_store([_key(1)], SimpleNamespace(req_id="r1"))
        meta = self._build_meta(
            sched,
            self._stock_meta(
                store_jobs={1: self._store_job([_key(1)], [40], "r1")},
                load_jobs={2: self._load_job([key], [50], "r2")},
            ),
        )
        self.assertEqual(meta.fence_job_ids, set(meta.staging_packs))
        self.assertEqual(len(meta.staging_packs), 1)
        (load,) = store.calls_named("load")
        store.complete_load(load.args[0])
        meta = self._build_meta(sched, self._stock_meta())
        self.assertEqual(meta.finished_load_jobs, {2: "r2"})
        ((pid, (rows, blocks)),) = meta.staging_unpacks.items()
        self.assertEqual((rows, blocks), (load.args[1], [50]))
        # Worker acks reach the staging scheduler.
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
            OffloadingConnectorScheduler,
        )

        out = SimpleNamespace(
            kv_connector_worker_meta=RaidenOffloadingWorkerMetadata(
                unpacked_jobs={pid: 2}
            )
        )
        free = sched._staged.rows.num_free
        with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
            sched.update_connector_output(out)
        self.assertEqual(sched._staged.rows.num_free, free + 1)

    def test_worker_meta_aggregates_unpacked(self):
        from vllm_torchtpu.offload.raiden_connector import (
            RaidenOffloadingWorkerMetadata,
        )

        a = RaidenOffloadingWorkerMetadata(unpacked_jobs={1: 1})
        b = RaidenOffloadingWorkerMetadata(unpacked_jobs={1: 1, 2: 1})
        self.assertEqual(a.aggregate(b).unpacked_jobs, {1: 2, 2: 1})


class TestStagedWorkerConnector(unittest.TestCase):
    def _conn(self):
        from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

        conn = object.__new__(TPURaidenOffloadingConnector)
        conn.connector_worker = MagicMock()
        conn._fenced_jobs = {}
        conn._completed_jobs = {}
        conn._pool_sync_tensors = [MagicMock()]
        conn._pending_fences = []
        conn._unpacked_jobs = {}
        conn._pending_unpacks = []
        conn._staging = MagicMock()
        return conn

    def _meta(self, **kwargs):
        from vllm_torchtpu.offload.raiden_connector import (
            RaidenOffloadingConnectorMetadata,
        )

        return RaidenOffloadingConnectorMetadata(load_jobs={}, store_jobs={}, **kwargs)

    def test_unpack_before_echo_and_pack_before_fence(self):
        conn = self._conn()
        order = []
        conn._staging.unpack.side_effect = lambda r, b: order.append(("unpack", r, b))
        conn._staging.pack.side_effect = lambda b, r: order.append(("pack", b, r))
        conn._connector_metadata = self._meta(
            fence_job_ids={9},
            staging_packs={9: ([40], [1])},
            staging_unpacks={8: ([0, 3], [50, 51])},
            finished_load_jobs={2: "r"},
        )
        events = iter(["unpack-ev", "fence-ev"])
        with (
            patch(
                "vllm_torchtpu.offload.raiden_connector._enqueue_pool_writes",
                side_effect=lambda t: order.append("enqueue"),
            ),
            patch(
                "vllm_torchtpu.offload.raiden_connector._record_fence_event",
                side_effect=lambda: (order.append("record"), next(events))[1],
            ),
            patch(
                "vllm_torchtpu.offload.raiden_connector._event_is_retired",
                return_value=False,
            ),
        ):
            results = conn.get_transfer_results(set())
        self.assertEqual(results.finished_recving, {"r"})
        self.assertEqual(
            order,
            [
                ("unpack", [0, 3], [50, 51]),
                "enqueue",
                "record",
                "enqueue",
                ("pack", [40], [1]),
                "enqueue",
                "record",
            ],
        )
        self.assertEqual(conn._pending_unpacks, [({8}, "unpack-ev")])
        self.assertEqual(conn._unpacked_jobs, {})
        conn._connector_metadata = self._meta()
        with patch(
            "vllm_torchtpu.offload.raiden_connector._event_is_retired",
            return_value=True,
        ):
            conn.get_transfer_results(set())
        self.assertEqual(conn._unpacked_jobs, {8: 1})
        self.assertEqual(conn._fenced_jobs, {9: 1})
        meta = conn.build_connector_worker_meta()
        self.assertEqual(meta.unpacked_jobs, {8: 1})
        self.assertEqual(conn._unpacked_jobs, {})


class TestStagedNamespaceAndSpec(unittest.TestCase):
    def test_namespace_salted_only_when_staged(self):
        from vllm_torchtpu.offload.raiden_store import derive_offload_namespace

        cfg = SimpleNamespace(
            model_config=SimpleNamespace(model="m", revision=None, quantization=None),
            cache_config=SimpleNamespace(prefix_caching_hash_algo="sha256"),
        )
        kw = dict(
            kernel_block_size=64,
            per_block_shape={(1, 1, 8)},
            kv_dtype={torch.uint8},
            device_block_size=64,
            world_size=8,
            num_kv_cache_groups=1,
            num_kv_cache_tensors=3,
        )
        plain = derive_offload_namespace(cfg, **kw)
        self.assertEqual(plain, derive_offload_namespace(cfg, staged_rows=False, **kw))
        self.assertNotEqual(
            plain, derive_offload_namespace(cfg, staged_rows=True, **kw)
        )

    def test_spec_off_by_default_and_refused_for_single_shape(self):
        self.assertEqual(base._build_spec({}).staging_blocks, 0)
        with self.assertRaisesRegex(ValueError, "multi-shape"):
            base._build_spec({"staging_blocks": 8})


if __name__ == "__main__":
    unittest.main()


# -- HBM budget: the staging rows are reserved before the KV pool is sized -----


def _mla(page_width: int, dtype=torch.uint8):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=page_width, dtype=dtype
    )


def _draft():
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    return SlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        sliding_window=128,
    )


_SPECS = {
    "l0.kv": _mla(640),
    "l0.index": _mla(80),
    "l1.kv": _mla(640),
    "l1.index": _mla(80),
    "draft.attn": _draft(),
}


def _offload_config(extra, connector="TPURaidenOffloadingConnector"):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector, kv_connector_extra_config=extra
        )
    )


class TestStagingHbmReserve(unittest.TestCase):
    def test_exact_bytes(self):
        payload = sum(s.page_size_bytes for s in _SPECS.values())
        row = rs.padded_row_bytes(payload)
        self.assertGreaterEqual(row, payload)
        self.assertEqual(row % (32 * 128), 0)
        cfg = _offload_config({"staging_blocks": 128})
        self.assertEqual(rs.staging_hbm_reserve_bytes(cfg, _SPECS), 128 * row)

    def test_skip_draft_kv_leaves_drafter_out(self):
        payload = sum(s.page_size_bytes for n, s in _SPECS.items() if n != "draft.attn")
        cfg = _offload_config({"staging_blocks": 8, "skip_draft_kv": True})
        self.assertEqual(
            rs.staging_hbm_reserve_bytes(cfg, _SPECS), 8 * rs.padded_row_bytes(payload)
        )

    def test_child_of_multi_connector(self):
        cfg = _offload_config(
            {
                "connectors": [
                    {"kv_connector": "Other"},
                    {
                        "kv_connector": "TPURaidenOffloadingConnector",
                        "kv_connector_extra_config": {"staging_blocks": 4},
                    },
                ]
            },
            connector="TPUMultiConnector",
        )
        payload = sum(s.page_size_bytes for s in _SPECS.values())
        self.assertEqual(
            rs.staging_hbm_reserve_bytes(cfg, _SPECS), 4 * rs.padded_row_bytes(payload)
        )

    def test_zero_when_off(self):
        for cfg in (
            SimpleNamespace(kv_transfer_config=None),
            _offload_config({}),
            _offload_config({"staging_blocks": 0}),
            _offload_config({"staging_blocks": 64}, connector="OffloadingConnector"),
        ):
            self.assertEqual(rs.staging_hbm_reserve_bytes(cfg, _SPECS), 0)

    def test_matches_the_rows_the_worker_allocates(self):
        """The reserve is the bytes of the staging buffer built from arrays
        of these specs' pages."""
        num_blocks = 6
        pools = [
            torch.zeros((num_blocks, s.page_size_bytes), dtype=torch.int8)
            for s in _SPECS.values()
        ]
        layout = rs.layout_from_tensors(pools, num_blocks, 32)
        cfg = _offload_config({"staging_blocks": 32})
        self.assertEqual(
            rs.staging_hbm_reserve_bytes(cfg, _SPECS), layout.rows * layout.row_bytes
        )


def _runner(extra):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = object.__new__(TPUModelRunner)
    runner.vllm_config = (
        _offload_config(extra)
        if extra is not None
        else (SimpleNamespace(kv_transfer_config=None))
    )
    runner.get_kv_cache_spec = lambda: dict(_SPECS)
    runner.lora_slot_write_hbm_bytes = lambda: 0
    runner.pp_handoff_hbm_reserve_bytes = lambda: 0
    return runner


def _reserves(extra):
    with patch(
        "vllm_torchtpu.utils.estimate_kv_connector_hbm_reserve", return_value=1000
    ):
        return _runner(extra).post_profile_hbm_reserves()


def test_runner_reserves_staging_bytes_exactly():
    staged = _reserves({"staging_blocks": 128})
    plain = _reserves({})
    payload = sum(s.page_size_bytes for s in _SPECS.values())
    assert staged.kv_connector - plain.kv_connector == 128 * rs.padded_row_bytes(
        payload
    )
    assert plain.kv_connector == 1000
    assert _reserves(None).kv_connector == 1000


def test_kv_pool_shrinks_by_exactly_the_staging_bytes(monkeypatch):
    from tests.worker import test_tpu_worker as tw
    from vllm_torchtpu import utils

    def available(extra):
        worker = tw._ep_dp_worker(
            monkeypatch,
            dp_size=1,
            expert_parallel=False,
            local=40 * utils.GBYTES,
            peer=40 * utils.GBYTES,
        )
        runner = _runner(extra)
        worker.model_runner.post_profile_hbm_reserves.side_effect = (
            runner.post_profile_hbm_reserves
        )
        with patch(
            "vllm_torchtpu.utils.estimate_kv_connector_hbm_reserve", return_value=0
        ):
            return worker.determine_available_memory()

    payload = sum(s.page_size_bytes for s in _SPECS.values())
    staging_bytes = 128 * rs.padded_row_bytes(payload)
    unset = available({})
    assert unset == 40 * utils.GBYTES
    assert available(None) == unset
    assert available({"staging_blocks": 0}) == unset
    assert unset - available({"staging_blocks": 128}) == staging_bytes


def test_worker_refuses_rows_the_budget_did_not_reserve():
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    num_blocks = 6
    specs = dict(_SPECS)
    pools = [
        torch.zeros((num_blocks, s.page_size_bytes), dtype=torch.int8)
        for s in specs.values()
    ]
    conn = object.__new__(TPURaidenOffloadingConnector)
    conn._staging_blocks = 16
    group = SimpleNamespace(
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs),
        layer_names=list(specs),
    )
    conn.connector_worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group])
    )
    # Reserved for fewer rows than the worker is about to allocate.
    conn._vllm_config = _offload_config({"staging_blocks": 8})
    with (
        patch("vllm_torchtpu.offload.raiden_connector.StagingBuffer") as buf,
        pytest.raises(RuntimeError, match="uncounted"),
    ):
        conn._init_staging(pools, num_blocks)
    buf.assert_not_called()


# -- logging: proof in the server log that the staged path ran -----------------


def test_init_log_reports_rows_bytes_reserve_and_arrays():
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    num_blocks = 6
    specs = dict(_SPECS)
    pools = [
        torch.zeros((num_blocks, s.page_size_bytes), dtype=torch.int8)
        for s in specs.values()
    ]
    conn = object.__new__(TPURaidenOffloadingConnector)
    conn._staging_blocks = 16
    group = SimpleNamespace(
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs),
        layer_names=list(specs),
    )
    conn.connector_worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group])
    )
    conn._vllm_config = _offload_config({"staging_blocks": 16})
    real = rs.StagingBuffer

    def cpu_buffer(*args, **kwargs):
        return real(*args, programs=_torch_programs, **kwargs)

    with (
        patch("vllm_torchtpu.offload.raiden_connector.StagingBuffer", cpu_buffer),
        patch("vllm_torchtpu.utils.synchronize_tensors"),
        patch("vllm_torchtpu.offload.raiden_connector.logger") as log,
    ):
        canonical = conn._init_staging(pools, num_blocks)
    payload = sum(s.page_size_bytes for s in specs.values())
    row = rs.padded_row_bytes(payload)
    assert canonical.page_size_bytes == row
    (call,) = [c for c in log.info.call_args_list if "staged offload on" in c.args[0]]
    rows, row_bytes, payload_bytes, alloc_mib, reserved_mib, arrays, _, _ = call.args[
        1:
    ]
    assert (rows, row_bytes, payload_bytes, arrays) == (16, row, payload, len(pools))
    assert alloc_mib == reserved_mib == 16 * row / (1 << 20)


def test_periodic_stats_log():
    now = [0.0]
    store, manager, _ = _staged(rows=4)
    staged = rs.StagedTransferScheduler(
        manager, 4, 2, row_bytes=1 << 20, log_interval_s=60.0, clock=lambda: now[0]
    )
    keys = [_key(i) for i in range(2)]
    for k in keys:
        seed_host_entries(store, k, 1)
    manager.prepare_load(keys, SimpleNamespace(req_id="r"))
    staged.add_load_job(1, keys, [30, 31], "r")
    skeys = [_key(10)]
    manager.prepare_store(skeys, SimpleNamespace(req_id="s"))
    staged.add_store_job(2, skeys, [40])
    with patch("vllm_torchtpu.offload.raiden_staging.logger") as log:
        out = _step(staged)
        (part_id,) = out.packs
        staged.on_fenced({part_id: 2})
        (load,) = store.calls_named("load")
        now[0] = 0.5
        store.complete_load(load.args[0])
        _step(staged)
        log.info.assert_not_called()  # within the interval
        now[0] = 61.0
        _step(staged)
        (call,) = log.info.call_args_list
        args = call.args[1:]
        assert args[1:6] == (1, 2, 2.0, 0, 2 * (1 << 20) / 0.5 / 1e9)
        assert args[6:9] == (1, 1, 1.0)
        assert args[9] == 0
        # Nothing moved in the next interval: no line.
        now[0] = 130.0
        _step(staged)
        assert len(log.info.call_args_list) == 1
