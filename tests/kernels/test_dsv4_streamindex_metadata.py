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
"""Tests for batch_tile_idx, bq_idx, and bkv_idx metadata (GDN v3 style)."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata


class MetadataTest(parameterized.TestCase):

    @parameterized.named_parameters(
        (
            "standard_decode",
            256,
            16,
            40704,
        ),
        (
            "batched_decode_bs4",
            256,
            64,
            36608,
        ),
        (
            "prefill_chunk",
            1,
            64,
            42240,
        ),
        (
            "ragged_prefill_batch",
            64,
            32,
            41472,
        ),
        (
            "small_batch",
            4,
            8,
            42240,
        ),
        (
            "long_context_64k",
            128,
            128,
            36736,
        ),
    )
    def test_generate_max_steps(
        self,
        num_seqs: int,
        max_pages_per_seq: int,
        expected_max_steps: int,
    ):
        max_steps = metadata.generate_max_steps(
            num_seqs=num_seqs,
            max_pages_per_seq=max_pages_per_seq,
        )
        self.assertEqual(max_steps, expected_max_steps)
        self.assertGreater(max_steps, 0)
        self.assertEqual(max_steps % 128, 0)

    def test_one_sequence_num_steps_one(self):
        """Test metadata for 1 sequence with 1 tile."""
        seq_lens = jnp.array([128], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)
        start_seq_idx = 0
        end_seq_idx = 1
        bq_sz = 16
        bkv_sz = 128
        compression_ratio = 1
        static_q_len = 1
        seq_batch_size = 1

        meta = metadata.compute_batched_seq_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=start_seq_idx,
            end_seq_idx=end_seq_idx,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=1,
            page_size=128,
            max_num_tokens=1,
            compression_ratio=compression_ratio,
            static_q_len=static_q_len,
            seq_batch_size=seq_batch_size,
        )

        self.assertEqual(int(meta.num_steps[0]), 1)

    def test_compute_batched_seq_metadata_decode(self):
        seq_lens = jnp.array([128, 256, 512, 64], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        start_seq_idx = 0
        end_seq_idx = 4
        bq_sz = 16
        bkv_sz = 64
        compression_ratio = 2
        static_q_len = 1
        seq_batch_size = 2

        meta = metadata.compute_batched_seq_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=start_seq_idx,
            end_seq_idx=end_seq_idx,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=8,
            page_size=32,
            max_num_tokens=4,
            compression_ratio=compression_ratio,
            static_q_len=static_q_len,
            seq_batch_size=seq_batch_size,
        )

        # Check start_seq_idx is batch_tile_idx per tile
        np.testing.assert_array_equal(
            np.array(meta.start_seq_idx)[:6], [0, 0, 2, 2, 2, 2])

        # For tile 0 (seq 0..1): max(64, 128) = 128 -> nbkv=2 (bkv 0..1)
        # For tile 1 (seq 2..3): max(256, 32) = 256 -> nbkv=4 (bkv 0..3)
        # Total tiles = 2 + 4 = 6
        self.assertEqual(int(meta.num_steps[0]), 6)
        np.testing.assert_array_equal(
            np.array(meta.batch_tile_idx)[:6], [0, 0, 2, 2, 2, 2])
        np.testing.assert_array_equal(
            np.array(meta.bq_idx)[:6], [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            np.array(meta.bkv_idx)[:6], [0, 1, 0, 1, 2, 3])

    def test_compute_per_seq_metadata_prefill(self):
        seq_lens = jnp.array([128, 256], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 64, 256], dtype=jnp.int32)
        start_seq_idx = 0
        end_seq_idx = 2
        bq_sz = 32
        bkv_sz = 64
        compression_ratio = 1

        meta = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=start_seq_idx,
            end_seq_idx=end_seq_idx,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=4,
            page_size=64,
            max_num_tokens=256,
            compression_ratio=compression_ratio,
            static_q_len=None,
        )

        np.testing.assert_array_equal(
            np.array(meta.start_seq_idx)[:4], [0, 0, 0, 0])

        # seq 0: q_len=64 -> nbq=2 (bq 0..1), kv_len=128 -> nbkv=2 (bkv 0..1) => 4 tiles
        # seq 1: q_len=192 -> nbq=6 (bq 0..5), kv_len=256 -> nbkv=4 (bkv 0..3) => 24 tiles
        # Total tiles = 4 + 24 = 28
        self.assertEqual(int(meta.num_steps[0]), 28)
        # Check seq 0 tiles (first 4)
        np.testing.assert_array_equal(
            np.array(meta.batch_tile_idx)[:4], [0, 0, 0, 0])
        np.testing.assert_array_equal(np.array(meta.bq_idx)[:4], [0, 0, 1, 1])
        np.testing.assert_array_equal(np.array(meta.bkv_idx)[:4], [0, 1, 0, 1])

    def test_mixed_config_and_print_schedule(self):
        """Test config with B=13 from streamindex_topk_test.py."""
        S_list = [128, 64, 32, 256, 128, 128, 64, 32, 256, 128, 32, 64, 256]
        seq_lens = jnp.array(S_list, dtype=jnp.int32)
        cu_q_lens = jnp.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 266, 268, 276], dtype=jnp.int32)
        bq_sz = 64
        bkv_p = 2
        page_size = 64
        bkv_sz = page_size * bkv_p  # 128
        comp_ratio = 1

        print("\n" + "=" * 60)
        print("TEST CASE 11 (B=13) METADATA & GENERATED SCHEDULE")
        print("=" * 60)

        # Pass 1: Decode Batched (seq 0 to 8 with seq_batch_size = 4)
        meta_decode_batch = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=0,
            end_seq_idx=8,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=4,
            page_size=page_size,
            max_num_tokens=276,
            compression_ratio=comp_ratio,
            static_q_len=1,
            seq_batch_size=4,
        )

        # Pass 2: Decode Remainder (seq 8 to 10 with seq_batch_size = 1)
        meta_decode_rem = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=8,
            end_seq_idx=10,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=4,
            page_size=page_size,
            max_num_tokens=276,
            compression_ratio=comp_ratio,
            static_q_len=1,
            seq_batch_size=1,
        )

        # Pass 3: Mixed / Prefill (seq 10 to 13 with seq_batch_size = 1)
        meta_mixed = metadata.compute_metadata(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            start_seq_idx=10,
            end_seq_idx=13,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            pages_per_seq=4,
            page_size=page_size,
            max_num_tokens=276,
            compression_ratio=comp_ratio,
            static_q_len=None,
            seq_batch_size=1,
        )
        # ============================================================
        # META_DECODE_BATCH (Batched Decode: seqs 0..7)
        # ============================================================
        # Pipeline Step (all_steps):              [   0,     1,     2   ]
        # Owner Tile:                             |--- Tile 0 ----|  |T 2|
        # Covered Sequences:                      |-- seqs 0..3 --|  |4..7|
        # ------------------------------------------------------------
        # Tile's Starting Step:                       0      0      2
        # Local Step in Tile (tile_step_id):          0      1      0
        # ------------------------------------------------------------
        # Query Block Index (bq_idx):                 0      0      0
        # KV Block Index (bkv_idx):                   0      1      0
        # ============================================================
        # --- GENERATED TILE SCHEDULE ---
        # Tile 00: BatchedDecode (seqs 0..3) [start_seq=0] | bq_idx=0 | bkv_idx=0
        # Tile 01: BatchedDecode (seqs 0..3) [start_seq=0] | bq_idx=0 | bkv_idx=1
        # Tile 02: BatchedDecode (seqs 4..7) [start_seq=4] | bq_idx=0 | bkv_idx=0
        # Tile 03: DecodeRemainder (seq 8) [start_seq=8]  | bq_idx=0 | bkv_idx=0
        # Tile 04: DecodeRemainder (seq 8) [start_seq=8]  | bq_idx=0 | bkv_idx=1
        # Tile 05: DecodeRemainder (seq 9) [start_seq=9]  | bq_idx=0 | bkv_idx=0
        # Tile 06: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=0 | bkv_idx=0
        # Tile 07: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=1 | bkv_idx=0
        # Tile 08: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=2 | bkv_idx=0
        # Tile 09: Prefill/Mixed (seq 10) [start_seq=10]  | bq_idx=3 | bkv_idx=0
        # Tile 10: Prefill/Mixed (seq 11) [start_seq=11]  | bq_idx=0 | bkv_idx=0
        # Tile 11: Prefill/Mixed (seq 12) [start_seq=12]  | bq_idx=0 | bkv_idx=0
        # Tile 12: Prefill/Mixed (seq 12) [start_seq=12]  | bq_idx=0 | bkv_idx=1
        # ============================================================
        self.assertEqual(int(meta_decode_batch.num_steps[0]), 3)
        self.assertEqual(int(meta_decode_rem.num_steps[0]), 3)
        self.assertEqual(int(meta_mixed.num_steps[0]), 7)

        # Full Tile Schedule Summary
        print("\n--- GENERATED TILE SCHEDULE ---")
        schedule = []
        tile_count = 0

        for meta, label in [
            (meta_decode_batch, "BatchedDecode"),
            (meta_decode_rem, "DecodeRemainder"),
            (meta_mixed, "Prefill/Mixed"),
        ]:
            num_t = int(meta.num_steps[0])
            for p in range(num_t):
                s_idx = int(meta.batch_tile_idx[p])
                start_s = int(meta.start_seq_idx[p])
                bq = int(meta.bq_idx[p])
                bkv = int(meta.bkv_idx[p])
                seq_str = (f"seqs {s_idx}..{s_idx + 3}"
                           if "Batched" in label else f"seq {s_idx}")
                schedule.append((
                    tile_count,
                    f"{label} ({seq_str}) [start_seq={start_s}]",
                    bq,
                    bkv,
                ))
                tile_count += 1

        for tile_id, phase, bq, bkv in schedule:
            print(
                f"Tile {tile_id:02d}: {phase:<38s} | bq_idx={bq} | bkv_idx={bkv}"
            )

        print("=" * 60 + "\n")


if __name__ == "__main__":
    absltest.main()
