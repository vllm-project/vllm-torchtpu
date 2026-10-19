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

    def test_packed_rope_gather_matches_the_padded_cache(self):
        """``packed_rope``: a 64-byte-per-token ROPE cache gathers to the same
        rows as the padded one, rope tail zeroed, with and without a live
        bound."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        pages, page_size = 64, 1024
        cache_tokens = pages * page_size
        query_tokens, topk = 16, 2048
        count = query_tokens * topk
        rng = np.random.default_rng(20261018)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 256, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8))
        flat_rope_u8 = rope_u8.reshape(-1, 128)
        packed_rope_i32 = jnp.asarray(
            native_sc_cache_layout.pack_rope_packed(
                flat_rope_u8[:, :64].reshape(pages, page_size, 64)
            )
        )
        self.assertEqual(packed_rope_i32.shape, (pages, page_size // 8, 128))
        flat_indices = rng.integers(0, cache_tokens, count, dtype=np.int32)
        expected_rope = native_sc_cache_layout.pack_selected_rope_for_tc(
            flat_rope_u8[flat_indices]
        )

        for atoms, live_kwargs in (
            (16, {}),
            (1, dict(num_live_indices=jnp.int32(5 * topk), live_multiple=topk)),
        ):
            with self.subTest(atoms=atoms, live=bool(live_kwargs)):
                _, gathered_rope = jax.block_until_ready(
                    native_sc_gather.dsa_gather_native_sc(
                        nope_i32,
                        packed_rope_i32,
                        jnp.asarray(flat_indices.reshape(query_tokens, topk)),
                        out_size=count,
                        atoms_per_batch=atoms,
                        packed_rope=True,
                        **live_kwargs,
                    )
                )
                groups = (
                    count if not live_kwargs else 5 * topk
                ) // native_sc_gather.ROPE_PACKING
                np.testing.assert_array_equal(
                    np.asarray(gathered_rope)[:groups], expected_rope[:groups]
                )

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

    def test_exported_mlir_has_no_allocate_buffer(self):
        """Ensure native-SC gather lowers without AllocateBuffer custom calls.

        When torch_tpu inlines custom-kernel functions across chunks/layers and
        runs mlir::createCSEPass(), zero-operand AllocateBuffer ops with
        has_side_effect=false get CSE-merged into a single buffer, causing XLA
        CopyInsertion to materialize hundreds of HBM-to-HBM copy(%empty) ops.
        """
        from unittest import mock

        from jax._src import tpu_info as src_tpu_info
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        info = src_tpu_info.get_tpu_info_for_chip(src_tpu_info.ChipVersion.TPU_7X, 1)
        nope = jax.ShapeDtypeStruct((1024, 128), jnp.uint32)
        rope = jax.ShapeDtypeStruct((256, 128), jnp.uint32)
        idx = jax.ShapeDtypeStruct((256, 2048), jnp.int32)
        live = jax.ShapeDtypeStruct((), jnp.int32)

        def fn(n, r, i, live_idx):
            return native_sc_gather.dsa_gather_native_sc(
                n,
                r,
                i,
                out_size=256 * 2048,
                atoms_per_batch=16,
                num_live_indices=live_idx,
                live_multiple=16 * 512,
            )

        with (
            mock.patch.object(src_tpu_info, "get_tpu_info", return_value=info),
            mock.patch.object(src_tpu_info, "is_tpu_device", return_value=True),
            mock.patch.object(tpu_info, "get_tpu_info", return_value=info),
            mock.patch.object(tpu_info, "is_tpu_device", return_value=True),
            mock.patch.object(pltpu, "get_tpu_info", return_value=info),
            pl.pallas_export_experimental(dynamic_shapes=False),
        ):
            exported = jax.export.export(jax.jit(fn), platforms=["tpu"])(
                nope, rope, idx, live
            )
            mlir_str = str(exported.mlir_module())

        self.assertNotIn("AllocateBuffer", mlir_str)


if __name__ == "__main__":
    absltest.main()
