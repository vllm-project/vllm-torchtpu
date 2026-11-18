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

import contextlib
import re
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest
from jax import export
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import (
    kernel as sparse_mla_kernel,
)
from vllm_torchtpu.kernels.mla.sparse import (
    native_sc_cache_layout,
    native_sc_gather,
)

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

    def test_should_match_wide_atom_gather_for_linear_and_packed_modes(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")
        pages, page_size = 32, 256
        rng = np.random.default_rng(20261118)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 0x7E, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8))
        rope_padded = jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope_u8))
        flat_rope_u8 = rope_u8.reshape(-1, 128)
        rope_packed = jnp.asarray(
            native_sc_cache_layout.pack_rope_packed(
                flat_rope_u8[:, :64].reshape(pages, page_size, 64)
            )
        )
        flat_nope = native_sc_cache_layout.flatten_nope_rows(
            native_sc_cache_layout.pack_nope(nope_u8)
        )

        cases = [
            (64, 2048, 16, True, True, None),
            (64, 2048, 16, True, False, None),
            (64, 2048, 16, False, True, 48 * 2048),
            (16, 512, 2, True, True, 12 * 512),
        ]
        for rows, topk, apb, packed, linear, num_live in cases:
            with self.subTest(
                rows=rows,
                topk=topk,
                apb=apb,
                packed=packed,
                linear=linear,
                num_live=num_live,
            ):
                count = rows * topk
                flat_indices = rng.integers(0, pages * page_size, count, np.int32)
                idx = jnp.asarray(flat_indices.reshape(rows, topk))
                kwargs = {}
                if num_live is not None:
                    kwargs["num_live_indices"] = jnp.int32(num_live)
                got_nope, got_rope = jax.block_until_ready(
                    native_sc_gather.dsa_gather_native_sc(
                        nope_i32,
                        rope_packed if packed else rope_padded,
                        idx,
                        out_size=count,
                        atoms_per_batch=apb,
                        packed_rope=packed,
                        linear_indices=linear,
                        **kwargs,
                    )
                )
                live_n = count if num_live is None else num_live
                np.testing.assert_array_equal(
                    np.asarray(got_nope)[:live_n], flat_nope[flat_indices[:live_n]]
                )
                expected_rope = native_sc_cache_layout.pack_selected_rope_for_tc(
                    flat_rope_u8[flat_indices]
                )
                groups = live_n // native_sc_gather.ROPE_PACKING
                np.testing.assert_array_equal(
                    np.asarray(got_rope)[:groups], expected_rope[:groups]
                )

    def test_should_select_wide_atom_factor_only_for_pipelined_overlap_batches(self):
        self.assertEqual(
            native_sc_gather._wide_atom_factor(  # noqa: SLF001 -- geometry helper
                16,
                wait_mode="overlap_rope",
                serve_pipelined=True,
                is_dcp_mode=False,
                has_row_live_atoms=False,
            ),
            4,
        )
        self.assertEqual(
            native_sc_gather._wide_atom_factor(  # noqa: SLF001 -- geometry helper
                2,
                wait_mode="overlap_rope",
                serve_pipelined=True,
                is_dcp_mode=False,
                has_row_live_atoms=False,
            ),
            2,
        )
        self.assertEqual(
            native_sc_gather._wide_atom_factor(  # noqa: SLF001 -- geometry helper
                16,
                wait_mode="overlap_rope",
                serve_pipelined=True,
                is_dcp_mode=True,
                has_row_live_atoms=False,
            ),
            1,
        )

    def test_should_reject_compact_rope_when_packed_rope_is_disabled(self):
        nope = jnp.zeros((2, 64, 128), dtype=jnp.uint32)
        rope = jnp.zeros((2, 16, 128), dtype=jnp.uint32)
        idx = jnp.zeros((32, 64), dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "compact_rope requires packed_rope"):
            native_sc_gather.dsa_gather_native_sc(
                nope,
                rope,
                idx,
                out_size=32 * 64,
                atoms_per_batch=1,
                packed_rope=False,
                compact_rope=True,
            )

    def test_should_match_padded_rope_when_compact_rope_packs_eight_tokens_per_row(
        self,
    ):
        if jax.default_backend() != "tpu":
            self.skipTest("Native SC compact rope gather runs on TPU")
        rng = np.random.default_rng(1802)
        pages, page_size = 8, 256
        nope_i32 = jnp.asarray(
            rng.integers(0, 2**32 - 1, (pages, page_size, 128), dtype=np.uint32)
        )
        rope_raw = rng.integers(0, 256, (pages, page_size, 64), dtype=np.uint8)
        rope_packed = jnp.asarray(native_sc_cache_layout.pack_rope_packed(rope_raw))
        flat_nope = np.asarray(nope_i32).reshape(-1, 128)
        flat_rope_u8 = np.pad(rope_raw.reshape(-1, 64), ((0, 0), (0, 64)))

        cases = [
            (32, 64, 1, False, None),
            (16, 512, 2, True, 12 * 512),
            (64, 2048, 16, True, None),
            (64, 2048, 16, False, 48 * 2048),
        ]
        for rows, topk, apb, linear, num_live in cases:
            with self.subTest(rows=rows, topk=topk, apb=apb, linear=linear):
                count = rows * topk
                flat_indices = rng.integers(0, pages * page_size, count, np.int32)
                idx = jnp.asarray(flat_indices.reshape(rows, topk))
                kwargs = {}
                if num_live is not None:
                    kwargs["num_live_indices"] = jnp.int32(num_live)
                got_nope, got_rope = jax.block_until_ready(
                    native_sc_gather.dsa_gather_native_sc(
                        nope_i32,
                        rope_packed,
                        idx,
                        out_size=count,
                        atoms_per_batch=apb,
                        packed_rope=True,
                        compact_rope=True,
                        linear_indices=linear,
                        **kwargs,
                    )
                )
                live_n = count if num_live is None else num_live
                np.testing.assert_array_equal(
                    np.asarray(got_nope)[:live_n], flat_nope[flat_indices[:live_n]]
                )
                got_r = np.asarray(got_rope)
                self.assertEqual(
                    got_r.shape,
                    (count // native_sc_gather.COMPACT_ROPE_PACKING, 128),
                )
                n_atoms = count // native_sc_gather.OCCURRENCES_PER_ATOM
                expanded = np.zeros(
                    (count // native_sc_gather.ROPE_PACKING, 128), dtype=np.uint32
                )
                for a in range(n_atoms):
                    src = got_r[a * 8 : (a + 1) * 8]
                    expanded[a * 16 : a * 16 + 8, :64] = src[:, :64]
                    expanded[a * 16 + 8 : a * 16 + 16, :64] = src[:, 64:]
                expected_rope = native_sc_cache_layout.pack_selected_rope_for_tc(
                    flat_rope_u8[flat_indices]
                )
                groups = live_n // native_sc_gather.ROPE_PACKING
                np.testing.assert_array_equal(expanded[:groups], expected_rope[:groups])

    def test_should_export_compact_rope_attention_with_16_lane_indirect_dmas(self):
        fake_sc = SimpleNamespace(num_cores=2, num_subcores=16, num_lanes=16)
        fake_info = SimpleNamespace(
            sparse_core=fake_sc,
            num_lanes=128,
            vmem_capacity_bytes=64 * 1024 * 1024,
            generation=7,
            get_sublane_tiling=lambda dtype: 32 // jnp.dtype(dtype).itemsize,
        )
        specs = (
            jax.ShapeDtypeStruct((64, 64, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((64, 64, 64), jnp.bfloat16),
            jax.ShapeDtypeStruct((16, 256, 128), jnp.uint32),
            jax.ShapeDtypeStruct((16, 32, 128), jnp.uint32),
            jax.ShapeDtypeStruct((64, 2048), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )

        def _run_attn(qn, qp, kn, kr, idx, qsl, dist):
            return sparse_mla_kernel.sparse_ragged_paged_attention(
                qn,
                kn,
                kr,
                idx,
                None,
                qsl,
                dist,
                k_scale=1.0,
                sm_scale=0.1,
                cache_layout=sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT,
                packed_rope=True,
                q_pe=qp,
                gather_and_attention_chunk_size=64,
                attention_kernel_batch_size=16,
                mixed_body_levers=sparse_mla_kernel.MIXED_BODY_DEFAULT,
            )

        ctx = contextlib.ExitStack()
        if jax.default_backend() != "tpu":
            ctx.enter_context(
                patch.object(pltpu, "get_tpu_info", return_value=fake_info)
            )
            ctx.enter_context(
                patch(
                    "jax._src.pallas.mosaic.tpu_info.get_tpu_info",
                    return_value=fake_info,
                )
            )
        with ctx:
            sparse_mla_kernel.sparse_ragged_paged_attention.clear_cache()
            native_sc_gather.dsa_gather_native_sc.clear_cache()
            if jax.default_backend() == "tpu":
                mlir = export.export(jax.jit(_run_attn), platforms=("tpu",))(
                    *specs
                ).mlir_module()
                self.assertIn("_packedrope_compactrope_w4", mlir)
                bad_widths = re.findall(
                    r"enqueue_indirect_dma[^\n]*vector<(?!16x)\d+xi32>", mlir
                )
                self.assertEqual(bad_widths, [])
            else:
                jaxpr_text = str(jax.make_jaxpr(_run_attn)(*specs))
                self.assertIn("_packedrope_compactrope_w4", jaxpr_text)


if __name__ == "__main__":
    absltest.main()
