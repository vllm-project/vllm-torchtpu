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
"""Byte-exact TPU test for the combined native-SC gather."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest

from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout, native_sc_gather

pytestmark = pytest.mark.tpu_test


class NativeScGatherTest(absltest.TestCase):
    def test_combined_gather_is_byte_exact(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        pages, page_size = 128, 256
        cache_tokens = pages * page_size
        query_tokens, topk = 128, 256
        count = query_tokens * topk
        rng = np.random.default_rng(20260830)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 0x7E, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = native_sc_cache_layout.pack_nope(nope_u8)
        rope_i32 = native_sc_cache_layout.pack_rope_banded(rope_u8)
        flat_nope = native_sc_cache_layout.flatten_nope_rows(nope_i32)
        flat_rope_u8 = rope_u8.reshape(-1, 128)

        patterns = {
            "uniform_unsorted": rng.integers(0, cache_tokens, count, dtype=np.int32),
            "repeated": np.full(count, 37, np.int32),
            "strided": (np.arange(count, dtype=np.int32) * 257) % cache_tokens,
        }
        for name, flat_indices in patterns.items():
            with self.subTest(name=name):
                indices = flat_indices.reshape(query_tokens, topk)
                gathered_nope, gathered_rope = native_sc_gather.dsa_gather_native_sc(
                    jnp.asarray(nope_i32),
                    jnp.asarray(rope_i32),
                    jnp.asarray(indices),
                    out_size=count,
                    atoms_per_batch=16,
                )
                gathered_nope, gathered_rope = jax.block_until_ready(
                    (gathered_nope, gathered_rope)
                )

                np.testing.assert_array_equal(
                    np.asarray(gathered_nope), flat_nope[flat_indices]
                )
                expected_rope = native_sc_cache_layout.pack_selected_rope_for_tc(
                    flat_rope_u8[flat_indices]
                )
                np.testing.assert_array_equal(np.asarray(gathered_rope), expected_rope)

    def test_live_bounded_gather_is_byte_exact_on_live_rows(self):
        """The decode bucket (16 tokens x topk 2048) with a traced live count.

        Only the rows below ``num_live_indices`` are promised; the rest are
        left unwritten. Whole-token counts (``live_multiple`` = topk) take the
        unguarded grid; a ragged count exercises the per-owner tail guard.
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        pages, page_size = 64, 1024
        cache_tokens = pages * page_size
        query_tokens, topk = 16, 2048
        count = query_tokens * topk
        rng = np.random.default_rng(20260925)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 0x7E, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = native_sc_cache_layout.pack_nope(nope_u8)
        rope_i32 = native_sc_cache_layout.pack_rope_banded(rope_u8)
        flat_nope = native_sc_cache_layout.flatten_nope_rows(nope_i32)
        flat_rope_u8 = rope_u8.reshape(-1, 128)
        flat_indices = rng.integers(0, cache_tokens, count, dtype=np.int32)
        indices = jnp.asarray(flat_indices.reshape(query_tokens, topk))

        # Zero live rows first (an idle rank's dummy batch): the zero-trip
        # path must start no DMA it does not wait, or every later launch in
        # this process fails.
        cases = [
            (0, topk),
            (0, 1),
            (1 * topk, topk),
            (5 * topk, topk),
            (16 * topk, topk),
            (1000, 1),
        ]
        for num_live, live_multiple in cases:
            with self.subTest(num_live=num_live, live_multiple=live_multiple):
                gathered_nope, gathered_rope = native_sc_gather.dsa_gather_native_sc(
                    jnp.asarray(nope_i32),
                    jnp.asarray(rope_i32),
                    indices,
                    out_size=count,
                    atoms_per_batch=1,
                    num_live_indices=jnp.int32(num_live),
                    live_multiple=live_multiple,
                )
                gathered_nope, gathered_rope = jax.block_until_ready(
                    (gathered_nope, gathered_rope)
                )
                live = flat_indices[:num_live]
                np.testing.assert_array_equal(
                    np.asarray(gathered_nope)[:num_live], flat_nope[live]
                )
                # ROPE rows pack four tokens each; compare whole groups only.
                groups = num_live // native_sc_gather.ROPE_PACKING
                expected_rope = native_sc_cache_layout.pack_selected_rope_for_tc(
                    flat_rope_u8[flat_indices]
                )
                np.testing.assert_array_equal(
                    np.asarray(gathered_rope)[:groups], expected_rope[:groups]
                )

    def test_row_live_gather_is_byte_exact_on_live_atoms(self):
        """``row_live_atoms``: live atoms byte-exact for 512/256-slot rows, whole or
        split over batches, 0/1/all live, alone and under ``num_live_indices``."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")
        pages, page_size = 64, 1024
        rng = np.random.default_rng(20261014)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 0x7E, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8))
        rope_i32 = jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope_u8))
        flat_nope = native_sc_cache_layout.flatten_nope_rows(
            native_sc_cache_layout.pack_nope(nope_u8)
        )
        flat_rope_u8 = rope_u8.reshape(-1, 128)
        atom = native_sc_gather.OCCURRENCES_PER_ATOM

        cases = [
            # (rows, row_len, atoms_per_batch, profile, num_live rows or None)
            (256, 512, 16, "mixed", None),
            (256, 512, 16, "zero", None),
            (256, 512, 16, "full", None),
            (256, 512, 16, "mixed", 128 * 512),
            (512, 256, 16, "mixed", None),
            (64, 512, 1, "mixed", None),
        ]
        for rows, row_len, apb, profile, num_live in cases:
            with self.subTest(rows=rows, row_len=row_len, apb=apb, p=profile):
                row_atoms = row_len // atom
                count = rows * row_len
                flat_indices = rng.integers(0, pages * page_size, count, np.int32)
                if profile == "zero":
                    live = np.zeros(rows, np.int32)
                elif profile == "full":
                    live = np.full(rows, row_atoms, np.int32)
                else:
                    live = rng.integers(0, row_atoms + 1, rows).astype(np.int32)
                    live[:3] = [0, 1, row_atoms]
                kwargs = {}
                if num_live is not None:
                    kwargs = dict(
                        num_live_indices=jnp.int32(num_live), live_multiple=16 * 512
                    )
                nope, rope = jax.block_until_ready(
                    native_sc_gather.dsa_gather_native_sc(
                        nope_i32,
                        rope_i32,
                        jnp.asarray(flat_indices.reshape(rows, row_len)),
                        out_size=count,
                        atoms_per_batch=apb,
                        row_live_atoms=jnp.asarray(live),
                        **kwargs,
                    )
                )
                slot_live = (
                    np.arange(row_len)[None, :] < (live * atom)[:, None]
                ).reshape(-1)
                if num_live is not None:
                    slot_live &= np.arange(count) < num_live
                np.testing.assert_array_equal(
                    np.asarray(nope)[slot_live], flat_nope[flat_indices[slot_live]]
                )
                expected_rope = native_sc_cache_layout.pack_selected_rope_for_tc(
                    flat_rope_u8[flat_indices]
                )
                group_live = slot_live[:: native_sc_gather.ROPE_PACKING]
                np.testing.assert_array_equal(
                    np.asarray(rope)[group_live], expected_rope[group_live]
                )


if __name__ == "__main__":
    absltest.main()
