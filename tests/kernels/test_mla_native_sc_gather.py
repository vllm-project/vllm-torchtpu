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


@pytest.mark.tpu_test
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
        """A 64-byte-per-token ROPE cache gathers to the same rows as the
        128-byte padded one, rope tail zeroed, derived from cache shapes."""
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

    def test_ring_gather_matches_the_pipeline_gather(self):
        """`ring_slots` DMA-ring serving writes the exact same live bytes as
        `emit_pipeline`, across padded (128B) and packed (64B) RoPE, ring depths
        2 and 4, single/multi-batch owners, whole-round, ragged, and zero-live
        counts."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        pages, page_size = 64, 1024
        cache_tokens = pages * page_size
        query_tokens, topk = 128, 2048
        count = query_tokens * topk
        rng = np.random.default_rng(20261030)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 0x7E, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8))
        flat_rope_u8 = rope_u8.reshape(-1, 128)
        ropes = {
            "padded_128b": jnp.asarray(
                native_sc_cache_layout.pack_rope_banded(rope_u8)
            ),
            "packed_64b": jnp.asarray(
                native_sc_cache_layout.pack_rope_packed(
                    flat_rope_u8[:, :64].reshape(pages, page_size, 64)
                )
            ),
        }
        indices = jnp.asarray(
            rng.integers(0, cache_tokens, count, dtype=np.int32).reshape(
                query_tokens, topk
            )
        )
        cases = [
            (16, 4, None, 1),
            (8, 4, None, 1),
            (8, 2, None, 1),
            (16, 4, 0, 16 * topk),
            (16, 4, 16 * topk, 16 * topk),
            (16, 4, 64 * topk, 16 * topk),
            (16, 4, 20 * topk, 4 * topk),
            (8, 4, 0, 4 * topk),
            (8, 4, 4 * topk, 4 * topk),
            (8, 4, 20 * topk, 4 * topk),
        ]
        for rope_kind, rope_i32 in ropes.items():
            for atoms, ring_slots, num_live, live_multiple in cases:
                with self.subTest(
                    rope_kind=rope_kind,
                    atoms=atoms,
                    ring_slots=ring_slots,
                    num_live=num_live,
                ):
                    if num_live is None:

                        def pipe_fn(n, r, idx, _a=atoms):
                            return native_sc_gather.dsa_gather_native_sc(
                                n,
                                r,
                                idx,
                                out_size=count,
                                atoms_per_batch=_a,
                                ring_slots=0,
                            )

                        def ring_fn(n, r, idx, _a=atoms, _rs=ring_slots):
                            return native_sc_gather.dsa_gather_native_sc(
                                n,
                                r,
                                idx,
                                out_size=count,
                                atoms_per_batch=_a,
                                ring_slots=_rs,
                            )

                        args = (nope_i32, rope_i32, indices)
                    else:

                        def pipe_fn(n, r, idx, live_n, _a=atoms, _lm=live_multiple):
                            return native_sc_gather.dsa_gather_native_sc(
                                n,
                                r,
                                idx,
                                out_size=count,
                                atoms_per_batch=_a,
                                num_live_indices=live_n,
                                live_multiple=_lm,
                                ring_slots=0,
                            )

                        def ring_fn(
                            n,
                            r,
                            idx,
                            live_n,
                            _a=atoms,
                            _rs=ring_slots,
                            _lm=live_multiple,
                        ):
                            return native_sc_gather.dsa_gather_native_sc(
                                n,
                                r,
                                idx,
                                out_size=count,
                                atoms_per_batch=_a,
                                num_live_indices=live_n,
                                live_multiple=_lm,
                                ring_slots=_rs,
                            )

                        args = (nope_i32, rope_i32, indices, jnp.int32(num_live))

                    jit_pipe = jax.jit(pipe_fn)
                    jit_ring = jax.jit(ring_fn)
                    pipe_ir = str(jit_pipe.lower(*args).compiler_ir())
                    ring_ir = str(jit_ring.lower(*args).compiler_ir())
                    self.assertIn("_overlap_rope_pipe", pipe_ir)
                    self.assertNotIn(f"_ring{ring_slots}", pipe_ir)
                    self.assertIn(f"_ring{ring_slots}", ring_ir)
                    ref_nope, ref_rope = jax.block_until_ready(jit_pipe(*args))
                    got_nope, got_rope = jax.block_until_ready(jit_ring(*args))
                    rows = count if num_live is None else num_live
                    groups = rows // native_sc_gather.ROPE_PACKING
                    np.testing.assert_array_equal(
                        np.asarray(got_nope)[:rows], np.asarray(ref_nope)[:rows]
                    )
                    np.testing.assert_array_equal(
                        np.asarray(got_rope)[:groups], np.asarray(ref_rope)[:groups]
                    )

    def test_ring_gather_stress_is_byte_exact_across_launches(self):
        """Many back-to-back DMA-ring launches with varied live counts and
        fresh indices. A timing-dependent slot, semaphore or index-buffer race
        shows up intermittently, so one bit-exact run is not enough: launch
        every iteration before syncing, then check each against numpy."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        pages, page_size = 64, 1024
        cache_tokens = pages * page_size
        query_tokens, topk = 64, 2048
        count = query_tokens * topk
        rng = np.random.default_rng(20261101)
        nope_u8 = rng.integers(0, 256, (pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(0, 0x7E, (pages, page_size // 4, 4, 128), np.uint8)
        rope_u8[..., 64:] = 0
        nope_i32 = native_sc_cache_layout.pack_nope(nope_u8)
        flat_nope = native_sc_cache_layout.flatten_nope_rows(nope_i32)
        flat_rope_u8 = rope_u8.reshape(-1, 128)
        rope_i32 = jnp.asarray(
            native_sc_cache_layout.pack_rope_packed(
                flat_rope_u8[:, :64].reshape(pages, page_size, 64)
            )
        )
        nope_i32 = jnp.asarray(nope_i32)

        ring_fn = jax.jit(
            lambda n, r, idx, live_n: native_sc_gather.dsa_gather_native_sc(
                n,
                r,
                idx,
                out_size=count,
                atoms_per_batch=16,
                num_live_indices=live_n,
                live_multiple=1,
                ring_slots=4,
            )
        )
        probe = (nope_i32, rope_i32, jnp.zeros((query_tokens, topk), jnp.int32))
        self.assertIn("_ring4", str(ring_fn.lower(*probe, jnp.int32(0)).compiler_ir()))

        # Live counts span zero, sub-round ragged tails, and whole rounds.
        live_counts = [0, count, 1, 63, 65, 1000, 4 * topk, 20 * topk + 17]
        live_counts += rng.integers(0, count + 1, 42).tolist()
        launches = []
        for num_live in live_counts:
            flat_indices = rng.integers(0, cache_tokens, count, dtype=np.int32)
            out = ring_fn(
                nope_i32,
                rope_i32,
                jnp.asarray(flat_indices.reshape(query_tokens, topk)),
                jnp.int32(num_live),
            )
            launches.append((num_live, flat_indices, out))
        for i, (num_live, flat_indices, (got_nope, got_rope)) in enumerate(launches):
            with self.subTest(iteration=i, num_live=num_live):
                live = flat_indices[:num_live]
                np.testing.assert_array_equal(
                    np.asarray(got_nope)[:num_live], flat_nope[live]
                )
                groups = num_live // native_sc_gather.ROPE_PACKING
                if not groups:
                    continue
                np.testing.assert_array_equal(
                    np.asarray(got_rope)[:groups],
                    native_sc_cache_layout.pack_selected_rope_for_tc(
                        flat_rope_u8[live[: groups * native_sc_gather.ROPE_PACKING]]
                    ),
                )


class NativeScGatherCpuTest(absltest.TestCase):
    def test_exported_mlir_has_no_allocate_buffer(self):
        """Ensure native-SC gather lowers without AllocateBuffer custom calls
        for both 64B (16-word) and 128B (32-word) SparseCore RoPE shapes across
        pipeline and DMA-ring modes."""
        from unittest import mock

        from jax._src import tpu_info as src_tpu_info
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        info = src_tpu_info.get_tpu_info_for_chip(src_tpu_info.ChipVersion.TPU_7X, 1)
        nope = jax.ShapeDtypeStruct((1024, 128), jnp.uint32)
        idx = jax.ShapeDtypeStruct((256, 2048), jnp.int32)
        live = jax.ShapeDtypeStruct((), jnp.int32)

        for ring_slots in (0, 4):

            def fn(n, r, i, live_idx, _rs=ring_slots):
                return native_sc_gather.dsa_gather_native_sc(
                    n,
                    r,
                    i,
                    out_size=256 * 2048,
                    atoms_per_batch=16,
                    num_live_indices=live_idx,
                    live_multiple=16 * 512,
                    ring_slots=_rs,
                )

            for rope_rows in (128, 256):
                rope = jax.ShapeDtypeStruct((rope_rows, 128), jnp.uint32)
                with (
                    self.subTest(ring_slots=ring_slots, rope_rows=rope_rows),
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
                if ring_slots > 0:
                    self.assertIn(f"_ring{ring_slots}", mlir_str)
                    self.assertNotIn("_overlap_rope_pipe", mlir_str)
                else:
                    self.assertIn("_overlap_rope_pipe", mlir_str)
                    self.assertNotIn("_ring4", mlir_str)

    def test_ring_fallback_warns_and_rejects_non_16_lane_sc(self):
        """Verify 16-lane check and warning_once when ring falls back."""
        from unittest import mock

        from jax._src import tpu_info as src_tpu_info
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        nope = jax.ShapeDtypeStruct((1024, 128), jnp.uint32)
        rope = jax.ShapeDtypeStruct((256, 128), jnp.uint32)

        # 1. v6e has an 8-lane SparseCore -> explicit ValueError.
        v6e_info = src_tpu_info.get_tpu_info_for_chip(
            src_tpu_info.ChipVersion.TPU_V6E, 1
        )
        with (
            mock.patch.object(pltpu, "get_tpu_info", return_value=v6e_info),
            self.assertRaisesRegex(ValueError, "16-lane SparseCore"),
        ):
            native_sc_gather.dsa_gather_native_sc(
                jnp.zeros((4, 256, 128), jnp.uint32),
                jnp.zeros((4, 64, 128), jnp.uint32),
                jnp.zeros((16, 128), jnp.int32),
                out_size=16 * 128,
                ring_slots=4,
            )

        # 2. Clamped non-power-of-two atoms_per_batch (6 = 12288 // (32 * 64) > 4),
        #    excessive VMEM scratch (ring_slots=16, atoms=32), and non-gen7 16-lane
        #    SparseCore chips warn and fall back to emit_pipeline.
        v7x_info = src_tpu_info.get_tpu_info_for_chip(
            src_tpu_info.ChipVersion.TPU_7X, 1
        )
        fallback_cases = [
            ("non_pow2_atoms", v7x_info, 6 * 32 * 64, 16, 4),
            ("vmem_budget", v7x_info, 32 * 32 * 64, 32, 16),
        ]
        if hasattr(src_tpu_info.ChipVersion, "TPU_8I"):
            tpu8i_info = src_tpu_info.get_tpu_info_for_chip(
                src_tpu_info.ChipVersion.TPU_8I, 1
            )
            fallback_cases.append(("unvalidated_gen", tpu8i_info, 16 * 4 * 64, 16, 4))

        for label, chip_info, out_rows, atoms, slots in fallback_cases:
            idx = jax.ShapeDtypeStruct((out_rows // 64, 64), jnp.int32)

            def fn(n, r, i, _o=out_rows, _a=atoms, _s=slots):
                return native_sc_gather.dsa_gather_native_sc(
                    n, r, i, out_size=_o, atoms_per_batch=_a, ring_slots=_s
                )

            with (
                self.subTest(case=label),
                mock.patch.object(src_tpu_info, "get_tpu_info", return_value=chip_info),
                mock.patch.object(src_tpu_info, "is_tpu_device", return_value=True),
                mock.patch.object(tpu_info, "get_tpu_info", return_value=chip_info),
                mock.patch.object(tpu_info, "is_tpu_device", return_value=True),
                mock.patch.object(pltpu, "get_tpu_info", return_value=chip_info),
                mock.patch.object(native_sc_gather.logger, "warning_once") as warn_mock,
                pl.pallas_export_experimental(dynamic_shapes=False),
            ):
                exported = jax.export.export(jax.jit(fn), platforms=["tpu"])(
                    nope, rope, idx
                )
                mlir_str = str(exported.mlir_module())
            warn_mock.assert_called_once()
            self.assertIn("_overlap_rope_pipe", mlir_str)
            self.assertNotIn(f"_ring{slots}", mlir_str)

    def test_rejects_invalid_ring_slots(self):
        nope = jnp.zeros((4, 256, 128), jnp.uint32)
        rope = jnp.zeros((4, 64, 128), jnp.uint32)
        idx = jnp.zeros((16, 128), jnp.int32)
        for bad_slots in (-1, 1, 3, 6):
            with (
                self.subTest(bad_slots=bad_slots),
                self.assertRaisesRegex(ValueError, "ring_slots"),
            ):
                native_sc_gather.dsa_gather_native_sc(
                    nope, rope, idx, out_size=16 * 128, ring_slots=bad_slots
                )

    def test_rejects_mismatched_or_unsupported_rope_row_widths(self):
        """Negative controls: invalid rope_cache shapes fail before lowering."""
        nope = jnp.zeros((4, 256, 128), jnp.uint32)
        idx = jnp.zeros((16, 64), jnp.int32)
        for bad_rope_shape, match in (
            # Not divisible across the 1024 nope tokens (14848 % 1024 == 512).
            ((4, 29, 128), "must divide"),
            # 16 tokens/row -> 8 words/token (not a multiple of 16 lanes).
            ((4, 16, 128), "derived rope_words"),
            # 2 tokens/row -> 64 words/token (4 * 64 > 128).
            ((4, 128, 128), "derived rope_words"),
        ):
            with (
                self.subTest(bad_rope_shape=bad_rope_shape),
                self.assertRaisesRegex(ValueError, match),
            ):
                native_sc_gather.dsa_gather_native_sc(
                    nope,
                    jnp.zeros(bad_rope_shape, jnp.uint32),
                    idx,
                    out_size=16 * 64,
                )


if __name__ == "__main__":
    absltest.main()
