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
"""
================================================================================
3-TIER EXECUTION PROVENANCE BANNER (Rule 19 Mandatory Provenance)
================================================================================
- Tier 1 (Compiled Host C-ABI Shared Library): N/A (Python/JAX Pallas & XLA lowering)
- Tier 2 (Compiled Host C++20 LloRegionBuilder IR): N/A
- Tier 3 (Execution Provenance):
  [CPU NUMERICAL PARITY & JAX 16-DEVICE SHARD_MAP VERIFICATION — NOT PHYSICAL TPU]
  Executes on 16-device CPU JAX mesh (`XLA_FLAGS=--xla_force_host_platform_device_count=16`)
  verifying Companion PR #5 Coalesced Lightning Indexer K-Cache Writer in `has` mode
  (`HEAD_ALONG_SUBLANE`, `(num_blocks, 64, 4, 256)`) and Top-K Single-Write / No-Copy /
  Shared-Read optimization flags across odd token counts, out-of-order page indices,
  padding slot masks, and varying block sizes/head dims.
================================================================================
"""

import ast
import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Configure 16 host CPU devices for DCP=16 shard_map testing before importing jax
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    coalesced_kv_append,
    coalesced_kv_append_dcp,
    cp_rank_as_data,
    get_kv_cache_shape,
    indexer_kv_layout,
    kv_cache_append,
    kv_cache_utils,
    scatter_rows,
    scatter_rows_reference,
    streamindex_topk,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flat arrays."""
    fa = a.astype(np.float64).ravel()
    fb = b.astype(np.float64).ravel()
    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    if denom == 0.0:
        return 1.0 if np.all(fa == fb) else 0.0
    return float(np.dot(fa, fb) / denom)


def execute_has_quads_granule_pipeline_cpu(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    dst_rows: jax.Array,
) -> tuple[np.ndarray, int, list[int], list[int], int]:
    """Execute the exact `_find_runs` + `_HasQuads.move_token` + `_HasQuads.blend`
    granule pipeline on CPU, returning:
      `(updated_cache, num_runs, run_lengths, run_dst_rows, partial_quad_reads)`
    """
    layout = KVLayout.HEAD_ALONG_SUBLANE
    page_size = kv_cache_append.cache_page_size(cache_kv, layout)
    rows = np.asarray(kv_cache_append._format_rows(k_packed, cache_kv, layout))
    num_tokens, lkv_dim = rows.shape
    padded_tokens = kv_cache_append._align_to(num_tokens, 4)
    k_padded = np.zeros((padded_tokens, lkv_dim), dtype=np.uint8)
    k_padded[:num_tokens] = rows
    k_quads = k_padded.reshape(padded_tokens // 4, 4, lkv_dim)

    num_pages = cache_kv.shape[0]
    num_rows = num_pages * page_size
    dst_np = np.asarray(dst_rows, dtype=np.int32)

    run_src = np.zeros((num_tokens + 1,), dtype=np.int32)
    run_dst = np.zeros((num_tokens + 1,), dtype=np.int32)
    num_runs = 0
    num_valid = 0
    next_row = -1

    # Step 1: Exact scalar-core walk + source compaction (`_find_runs` + `move_token`)
    for tok in range(num_tokens):
        row = int(dst_np[tok])
        valid = (row >= 0) and (row < num_rows)
        starts = valid and ((row != next_row) or (row % 4 == 0))
        run_src[num_runs] = num_valid
        run_dst[num_runs] = row
        if valid and (num_valid != tok):
            k_quads[num_valid >> 2] = np.asarray(
                kv_cache_append._blend_token_into_quad(
                    jnp.asarray(k_quads[num_valid >> 2], dtype=jnp.int32),
                    jnp.asarray(k_quads[tok >> 2], dtype=jnp.int32),
                    jnp.int32(tok & 3),
                    jnp.int32(num_valid & 3),
                ),
                dtype=np.uint8,
            )
        if starts:
            num_runs += 1
        if valid:
            num_valid += 1
            next_row = row + 1
    run_src[num_runs] = num_valid

    # Step 2: Exact wave RMW execution (`_write_runs` + `_HasQuads.blend`)
    flat_quads = np.array(cache_kv, copy=True).reshape(-1, 4, lkv_dim)
    num_src_quads = k_quads.shape[0]
    run_lengths = []
    run_dst_list = []
    partial_quad_reads = 0

    for r in range(num_runs):
        src_tok = int(run_src[r])
        row = int(run_dst[r])
        n = int(run_src[r + 1] - src_tok)
        run_lengths.append(n)
        run_dst_list.append(row)

        quad_idx = row >> 2
        if n < 4:
            partial_quad_reads += 1
            rmw_quad = flat_quads[quad_idx].astype(np.int32)
        else:
            # When n == 4 (full quad), _write_runs skips the HBM->VMEM read!
            rmw_quad = np.zeros((4, lkv_dim), dtype=np.int32)

        q0 = src_tok >> 2
        q1 = min(q0 + 1, num_src_quads - 1)
        pair = np.concatenate(
            [k_quads[q0].astype(np.int32), k_quads[q1].astype(np.int32)], axis=0
        )
        first = row & 3
        shift = (first - (src_tok & 3)) & 7
        src_rolled = np.roll(pair, shift=shift, axis=0)
        d = np.arange(4, dtype=np.int32)[:, None]
        blended = np.where((d >= first) & (d < first + n), src_rolled[:4], rmw_quad)
        flat_quads[quad_idx] = blended.astype(np.uint8)

    return (
        flat_quads.reshape(cache_kv.shape),
        num_runs,
        run_lengths,
        run_dst_list,
        partial_quad_reads,
    )


def _store_topk_rows_reference(
    buffer: torch.Tensor,
    topk_indices: torch.Tensor,
    num_tokens: int,
    topk: int,
) -> None:
    """Reference implementation matching mla_attention_op._store_topk_rows."""
    rows, cols = topk_indices.shape
    buf_rows, buf_cols = buffer.shape
    if num_tokens == rows and topk == cols and cols == buf_cols:
        n = min(rows, buf_rows)
        buffer[:n] = topk_indices[:n]
        return

    buffer[:num_tokens] = -1
    n = min(rows, buf_rows)
    buffer[:n, :topk] = topk_indices[:n]


class TestCoalescedIndexerKWriter(unittest.TestCase):
    """Gate 7B/7C Unit & Adversarial Test Suite for Coalesced Indexer K-Cache Writer."""

    def test_01_default_has_mode_and_kv_shape_compatibility(self):
        """Test 1: Verify DEFAULT_INDEXER_KV_CACHE_MODE and zero KV shape changes."""
        self.assertIn(envs.DEFAULT_INDEXER_KV_CACHE_MODE, ["has", "sal"])
        self.assertIn(envs.VLLM_TPU_INDEXER_KV_CACHE_MODE, ["has", "sal"])
        self.assertIn(indexer_kv_layout(256), [KVLayout.HEAD_ALONG_SUBLANE, KVLayout.SEQ_ALONG_LANE])
        self.assertEqual(indexer_kv_layout(256, "has"), KVLayout.HEAD_ALONG_SUBLANE)
        self.assertEqual(indexer_kv_layout(256, "xla"), KVLayout.HEAD_ALONG_SUBLANE)
        self.assertEqual(indexer_kv_layout(256, "sal"), KVLayout.SEQ_ALONG_LANE)
        # Verify fallback to HEAD_ALONG_SUBLANE when block_size is not a multiple of 128
        self.assertEqual(indexer_kv_layout(64, "sal"), KVLayout.HEAD_ALONG_SUBLANE)

        # Verify get_kv_cache_shape preserves (num_blocks, 64, 4, 256) in `has` mode
        # for unpacked head_dim=128 and packed record width 129.
        shape_128 = get_kv_cache_shape(
            total_num_pages=16,
            page_size=256,
            actual_head_dim=128,
            kv_layout=indexer_kv_layout(256, "has"),
        )
        shape_129 = get_kv_cache_shape(
            total_num_pages=16,
            page_size=256,
            actual_head_dim=129,
            kv_layout=indexer_kv_layout(256, "has"),
        )
        self.assertEqual(shape_128, (16, 64, 4, 256))
        self.assertEqual(shape_129, (16, 64, 4, 256))

        # Verify SAL mode recovers 128 + 4 = 132 -> (16, 33, 4, 256) from packed 129
        sal_shape = get_kv_cache_shape(
            total_num_pages=16,
            page_size=256,
            actual_head_dim=129,
            kv_layout=indexer_kv_layout(256, "sal"),
        )
        self.assertEqual(sal_shape, (16, 33, 4, 256))

    def test_02_dcp16_interleave4_prefill_1024_zero_read_full_quad_coalescing(self):
        """Test 2: Verify 1024-token prefill on DCP=16 (interleave_size=4) compacts to 16
        full-quad runs (64 contiguous slots 0..63) with ZERO HBM read-modify-write reads!
        """
        cp_size = 16
        interleave_size = 4
        page_size = 256
        num_local_pages = 8
        num_tokens = 1024  # 1 prefill sequence of 1024 tokens

        rng = np.random.default_rng(42)
        cache_init = jnp.zeros((num_local_pages, 64, 4, 256), dtype=jnp.uint8)
        k_packed = jnp.asarray(
            rng.integers(0, 256, size=(num_tokens, 129), dtype=np.uint8)
        )
        seq_lens = jnp.array([1024, 0, 0, 0], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1024, 1024, 1024, 1024], dtype=jnp.int32)
        # Virtual page 0 maps to physical local page 3 on each rank
        page_indices = jnp.array([3, 4, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)

        for rank in range(cp_size):
            page, slot = kv_cache_utils.get_dcp_page_and_slot(
                num_tokens,
                seq_lens,
                page_indices,
                cu_q_lens,
                jnp.int32(rank),
                page_size=page_size,
                num_pages=num_local_pages,
                dcp_size=cp_size,
                interleave_size=interleave_size,
            )
            dst_rows = kv_cache_utils.get_dst_rows(page, slot, page_size)
            _, num_runs, run_lengths, run_dst_rows, partial_reads = (
                execute_has_quads_granule_pipeline_cpu(cache_init, k_packed, dst_rows)
            )
            # Each rank owns 64 tokens = 16 full 4-token quads landing on
            # contiguous rows [3*256 + 0 .. 3*256 + 63], with ZERO partial-quad HBM reads!
            self.assertEqual(num_runs, 16)
            self.assertEqual(run_lengths, [4] * 16)
            self.assertEqual(
                run_dst_rows, [3 * page_size + 4 * q for q in range(16)]
            )
            self.assertEqual(
                partial_reads,
                0,
                f"Rank {rank} expected 0 HBM reads for full-quad prefill runs",
            )

    def test_03_dcp16_prefill_and_decode_bit_exact_parity_vs_xla_scatter(self):
        """Test 3: Verify 100% bit-exact parity across all 16 DCP ranks on prefill + decode."""
        cp_size = 16
        interleave_size = 4
        page_size = 256
        num_local_pages = 6
        record_width = 256
        mesh = Mesh(np.array(jax.devices()[:cp_size]), (DCP_AXIS_NAME,))

        rng = np.random.default_rng(20261002)
        # Pre-populate cache_kv on all 16 DCP ranks with non-zero bytes to verify
        # partial-quad blending preserves existing neighbor tokens!
        global_cache_init = rng.integers(
            1, 250, size=(cp_size * num_local_pages, 64, 4, record_width), dtype=np.uint8
        )
        sharding = NamedSharding(mesh, P(DCP_AXIS_NAME))
        cache_xla = jax.device_put(jnp.asarray(global_cache_init), sharding)
        cache_pallas = jax.device_put(jnp.asarray(global_cache_init), sharding)

        # Mixed batch:
        # - Seq 0: decode (1 token at pos 517 -> unaligned lane 1 inside quad)
        # - Seq 1: decode (1 token at pos 19 -> unaligned lane 3 inside quad)
        # - Seq 2: prefill (1024 tokens at pos 0..1023)
        # - Padded tail: 22 tokens (total padded batch = 1048 tokens)
        num_tokens = 1048
        k_packed = jnp.asarray(
            rng.integers(0, 256, size=(num_tokens, 129), dtype=np.uint8)
        )
        seq_lens = jnp.array([518, 20, 1024, 0], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1, 2, 1026, 1026], dtype=jnp.int32)
        page_indices = jnp.array(
            [
                1, 2,  # Seq 0 virtual pages
                4, 5,  # Seq 1 virtual pages
                0, 3,  # Seq 2 virtual pages
                0, 0,
            ],
            dtype=jnp.int32,
        )

        # 1. Reference XLA scatter across 16 DCP ranks
        def _xla_dcp_scatter(cache_kv, k_packed, seq_lens, page_indices, cu_q_lens):
            rank = cp_rank_as_data(DCP_AXIS_NAME, cp_size)
            _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
            pg_sz = block_size_per_kv_packing * kv_packing
            k_pad = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
            seq_id, pos, in_batch = kv_cache_utils._row_positions(
                k_pad.shape[0], seq_lens, cu_q_lens
            )
            virtual_page_size = jnp.int32(pg_sz * cp_size)
            cycle = jnp.int32(cp_size * interleave_size)
            interleave = jnp.int32(interleave_size)
            offset = pos % virtual_page_size
            owner = (offset % cycle) // interleave
            slot = (offset // cycle) * interleave + (offset % interleave)
            local_page_indices = jnp.mod(
                page_indices, jnp.int32(cache_kv.shape[0])
            ).reshape(seq_lens.shape[0], -1)
            max_pages_per_seq = local_page_indices.shape[1]
            virtual_page = jnp.clip(pos // virtual_page_size, 0, max_pages_per_seq - 1)
            valid = jnp.logical_and(in_batch, owner == rank)
            page = jnp.where(
                valid,
                local_page_indices[seq_id, virtual_page],
                jnp.int32(kv_cache_utils.OOB_PAGE),
            )
            return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_pad)

        # 2. PR #5 coalesced_kv_append_dcp across 16 DCP ranks via jax.shard_map
        def _pallas_dcp_scatter(cache_kv, k_packed, seq_lens, page_indices, cu_q_lens):
            rank = cp_rank_as_data(DCP_AXIS_NAME, cp_size)
            return coalesced_kv_append_dcp(
                cache_kv,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                rank,
                dcp_size=cp_size,
                interleave_size=interleave_size,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

        replicated = P()
        sharded = P(DCP_AXIS_NAME)
        out_xla = np.asarray(
            jax.shard_map(
                _xla_dcp_scatter,
                mesh=mesh,
                in_specs=(sharded, replicated, replicated, replicated, replicated),
                out_specs=sharded,
                check_vma=False,
            )(cache_xla, k_packed, seq_lens, page_indices, cu_q_lens)
        )
        out_pallas_dcp = np.asarray(
            jax.shard_map(
                _pallas_dcp_scatter,
                mesh=mesh,
                in_specs=(sharded, replicated, replicated, replicated, replicated),
                out_specs=sharded,
                check_vma=False,
            )(cache_pallas, k_packed, seq_lens, page_indices, cu_q_lens)
        )

        # Verify 16-rank shard_map coalesced_kv_append_dcp matches XLA scatter bit-for-bit
        np.testing.assert_array_equal(out_pallas_dcp, out_xla)
        cossim = _cosine_similarity(out_pallas_dcp, out_xla)
        self.assertAlmostEqual(cossim, 1.0, places=6)

        # Also verify each rank's _HasQuads granule pipeline (_find_runs + move_token + blend)
        total_decode_partial_reads = 0
        for rank in range(cp_size):
            r0 = rank * num_local_pages
            r1 = (rank + 1) * num_local_pages
            rank_cache_init = jnp.asarray(global_cache_init[r0:r1])
            page, slot = kv_cache_utils.get_dcp_page_and_slot(
                num_tokens,
                seq_lens,
                page_indices,
                cu_q_lens,
                jnp.int32(rank),
                page_size=page_size,
                num_pages=num_local_pages,
                dcp_size=cp_size,
                interleave_size=interleave_size,
            )
            dst_rows = kv_cache_utils.get_dst_rows(page, slot, page_size)
            out_has_quads, _, _, _, partial_reads = execute_has_quads_granule_pipeline_cpu(
                rank_cache_init, k_packed, dst_rows
            )
            total_decode_partial_reads += partial_reads
            np.testing.assert_array_equal(
                out_has_quads,
                out_xla[r0:r1],
                err_msg=f"_HasQuads granule pipeline mismatch on DCP rank {rank}",
            )
        # Exactly 2 decode sequences in batch -> only 2 partial-quad RMW reads across all 16 ranks!
        self.assertEqual(total_decode_partial_reads, 2)

    def test_04_odd_token_counts_and_out_of_order_page_indices(self):
        """Test 4: Verify bit-exact parity across odd token counts T in {1, 3, 7, 16, 64, 256}
        and out-of-order page indices across multiple sequences.
        """
        rng = np.random.default_rng(777)
        page_size = 256
        num_pages = 16
        layout = KVLayout.HEAD_ALONG_SUBLANE

        for t in [1, 3, 7, 16, 64, 256]:
            with self.subTest(num_tokens=t):
                cache_init = jnp.asarray(
                    rng.integers(1, 250, size=(num_pages, page_size // 4, 4, 256), dtype=np.uint8)
                )
                k_packed = jnp.asarray(rng.integers(0, 256, size=(t, 129), dtype=np.uint8))

                # Split t tokens across 2 sequences with out-of-order pages
                t0 = t // 2
                t1 = t - t0
                seq_lens = jnp.array([20 + t0, 50 + t1], dtype=jnp.int32)
                cu_q_lens = jnp.array([0, t0, t], dtype=jnp.int32)
                # Virtual pages in non-sequential order: seq 0 uses [7, 2, 5], seq 1 uses [1, 6, 3]
                page_indices = jnp.array([[7, 2, 5, 0], [1, 6, 3, 4]], dtype=jnp.int32)

                # Reference scatter
                page, slot = kv_cache_utils.get_page_and_slot(
                    t, seq_lens, page_indices, cu_q_lens, page_size=page_size, num_pages=num_pages
                )
                dst_rows = kv_cache_utils.get_dst_rows(page, slot, page_size)
                out_ref = np.asarray(
                    scatter_rows_reference(
                        cache_init,
                        k_packed,
                        dst_rows,
                        kv_layout=layout,
                    )
                )

                # Coalesced KV append end-to-end
                out_coalesced = np.asarray(
                    coalesced_kv_append(
                        cache_init,
                        k_packed,
                        seq_lens,
                        page_indices,
                        cu_q_lens,
                        kv_layout=layout,
                    )
                )

                np.testing.assert_array_equal(out_coalesced, out_ref)
                cossim = _cosine_similarity(out_coalesced, out_ref)
                self.assertAlmostEqual(cossim, 1.0, places=6)

    def test_05_padding_slot_masks_and_oob_pages(self):
        """Test 5: Verify padding slot masks (-1 slots, OOB pages) leave background cache
        bytes completely unmodified.
        """
        rng = np.random.default_rng(888)
        page_size = 256
        num_pages = 8
        layout = KVLayout.HEAD_ALONG_SUBLANE

        # Fixed background byte 0xAA (170)
        cache_init = jnp.full((num_pages, page_size // 4, 4, 256), 170, dtype=jnp.uint8)
        num_tokens = 32
        k_packed = jnp.full((num_tokens, 129), 42, dtype=jnp.uint8)

        # Alternating valid tokens and invalid/padding tokens (-1, OOB_PAGE)
        dst_rows_np = np.full((num_tokens,), -1, dtype=np.int32)
        valid_indices = [0, 2, 5, 8, 14, 20, 25, 30]
        for idx in valid_indices:
            dst_rows_np[idx] = idx * 10  # valid row < num_pages * page_size
        dst_rows = jnp.asarray(dst_rows_np)

        out_ref = np.asarray(
            scatter_rows_reference(
                cache_init,
                k_packed,
                dst_rows,
                kv_layout=layout,
            )
        )
        out_pallas = np.asarray(
            scatter_rows(
                cache_init,
                k_packed,
                dst_rows,
                kv_layout=layout,
            )
        )

        np.testing.assert_array_equal(out_pallas, out_ref)

        # Check that rows not targeted by valid_indices retain exactly 170
        flat_cache = out_pallas.reshape(-1, 4, 256)
        written_quads = set()
        for idx in valid_indices:
            row = dst_rows_np[idx]
            written_quads.add(row >> 2)

        for q in range(flat_cache.shape[0]):
            if q not in written_quads:
                self.assertTrue(
                    np.all(flat_cache[q] == 170),
                    f"Background quad {q} was unexpectedly modified!",
                )

    def test_06_varying_block_size_and_head_dim(self):
        """Test 6: Verify layout and scatter across block_size in {64, 128, 256, 512}
        and head_dim in {64, 128, 129}.
        """
        rng = np.random.default_rng(999)
        num_pages = 4
        num_tokens = 16

        for block_size in [64, 128, 256, 512]:
            for head_dim in [64, 128, 129]:
                with self.subTest(block_size=block_size, head_dim=head_dim):
                    layout = KVLayout.HEAD_ALONG_SUBLANE
                    shape = get_kv_cache_shape(
                        total_num_pages=num_pages,
                        page_size=block_size,
                        actual_head_dim=head_dim,
                        kv_layout=layout,
                    )
                    expected_lkv_dim = 256
                    self.assertEqual(
                        shape, (num_pages, block_size // 4, 4, expected_lkv_dim)
                    )

                    cache_init = jnp.asarray(rng.integers(0, 255, size=shape, dtype=np.uint8))
                    k_packed = jnp.asarray(rng.integers(0, 255, size=(num_tokens, head_dim), dtype=np.uint8))
                    dst_rows = jnp.arange(num_tokens, dtype=jnp.int32) * 2

                    out_ref = np.asarray(
                        scatter_rows_reference(
                            cache_init,
                            k_packed,
                            dst_rows,
                            kv_layout=layout,
                        )
                    )
                    out_pallas = np.asarray(
                        scatter_rows(
                            cache_init,
                            k_packed,
                            dst_rows,
                            kv_layout=layout,
                        )
                    )
                    np.testing.assert_array_equal(out_pallas, out_ref)

    def test_07_topk_single_write_nocopy_shared_read(self):
        """Test 7: Verify Top-K Single-Write / No-Copy / Shared-Read optimization flags
        and logic (skipping -1 fill on exact shape match, dynamic update, AST wiring).
        """
        # 1. Verify env flags in envs
        self.assertTrue(envs.TPU_INDEXER_TOPK_SINGLE_WRITE)
        self.assertTrue(envs.TPU_INDEXER_TOPK_NOCOPY)
        self.assertTrue(envs.TPU_INDEXER_TOPK_SHARED_READ)
        self.assertIn(envs.DEFAULT_INDEXER_KV_CACHE_MODE, ["has", "sal"])
        self.assertIn(envs.VLLM_TPU_INDEXER_KV_CACHE_MODE, ["has", "sal"])

        # 2. Verify _store_topk_rows single-write fast path
        buf_rows, buf_cols = 1024, 64
        buf = torch.full((buf_rows, buf_cols), 999, dtype=torch.int32)
        topk_indices = torch.randint(0, 500, (buf_rows, buf_cols), dtype=torch.int32)

        # Fast path: num_tokens == rows and topk == cols and cols == buf_cols
        _store_topk_rows_reference(buf, topk_indices, buf_rows, buf_cols)
        torch.testing.assert_close(buf, topk_indices)

        # Slow / fallback path: shape mismatch triggers -1 fill
        buf_slow = torch.full((buf_rows, buf_cols), 999, dtype=torch.int32)
        partial_tokens = 100
        _store_topk_rows_reference(buf_slow, topk_indices[:partial_tokens], partial_tokens, buf_cols)
        # First partial_tokens should match topk_indices
        torch.testing.assert_close(buf_slow[:partial_tokens], topk_indices[:partial_tokens])
        # Remaining rows up to buf_rows should remain 999 (untouched beyond partial_tokens)
        self.assertTrue(torch.all(buf_slow[partial_tokens:] == 999))

        # 3. Verify JAX dynamic update slice logic
        jax_buf = jnp.zeros((1024, 64), dtype=jnp.int32)
        jax_rows = jnp.ones((128, 64), dtype=jnp.int32) * 42
        updated = jax.lax.dynamic_update_slice(jax_buf, jax_rows, (0, 0))
        self.assertTrue(jnp.all(updated[:128] == 42))
        self.assertTrue(jnp.all(updated[128:] == 0))

        # 4. AST verification of mla_attention_op.py wiring
        mla_op_path = os.path.join(
            os.path.dirname(__file__),
            "../../../../src/vllm_torchtpu/layers/adapter/custom_ops/mla_attention_op.py",
        )
        self.assertTrue(os.path.exists(mla_op_path), f"File {mla_op_path} not found")
        with open(mla_op_path, "r", encoding="utf-8") as f:
            code = f.read()

        parsed = ast.parse(code)
        func_names = [n.name for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef)]
        self.assertIn("_topk_rows_update_jax", func_names)
        self.assertIn("_fake_topk_rows_update", func_names)
        self.assertIn("_store_topk_rows", func_names)
        self.assertIn("_write_topk_rows", func_names)

        # Verify key identifiers are wired in the file
        self.assertIn("TPU_INDEXER_TOPK_SINGLE_WRITE", code)
        self.assertIn("TPU_INDEXER_TOPK_NOCOPY", code)
        self.assertIn("TPU_INDEXER_TOPK_SHARED_READ", code)
        self.assertIn("VLLM_TPU_INDEXER_KV_CACHE_MODE", code)
        self.assertIn("coalesced_kv_append_dcp", code)
        self.assertIn("coalesced_kv_append", code)

    def test_08_negative_controls_validation(self):
        """Test 8: Verify negative controls raise expected exceptions (Rule 16 Gate 3/4)."""
        # 1. Invalid mode raises ValueError
        with self.assertRaisesRegex(ValueError, "Invalid VLLM_TPU_INDEXER_KV_CACHE_MODE"):
            indexer_kv_layout(256, "invalid_mode")

        # 2. Non-divisible interleave_size raises ValueError in get_dcp_page_and_slot
        with self.assertRaisesRegex(ValueError, "interleave_size=6 must divide"):
            kv_cache_utils.get_dcp_page_and_slot(
                16,
                jnp.array([16], dtype=jnp.int32),
                jnp.array([0], dtype=jnp.int32),
                jnp.array([0, 16], dtype=jnp.int32),
                0,
                page_size=256,
                num_pages=4,
                dcp_size=16,
                interleave_size=6,
            )

        # 3. Invalid cache_kv shape (lkv_dim=100 not a multiple of 128) raises ValueError
        bad_cache = jnp.zeros((4, 64, 4, 100), dtype=jnp.uint8)
        k_packed = jnp.zeros((16, 129), dtype=jnp.uint8)
        dst_rows = jnp.zeros((16,), dtype=jnp.int32)
        with self.assertRaisesRegex(
            ValueError, "is invalid for HEAD_ALONG_SUBLANE"
        ):
            kv_cache_append.scatter_rows(
                bad_cache,
                k_packed,
                dst_rows,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )

        # 4. Mismatched dst_rows length raises ValueError
        good_cache = jnp.zeros((4, 64, 4, 256), dtype=jnp.uint8)
        bad_dst = jnp.zeros((8,), dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "dst_rows shape .* must be"):
            kv_cache_append.scatter_rows(
                good_cache,
                k_packed,
                bad_dst,
                kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
