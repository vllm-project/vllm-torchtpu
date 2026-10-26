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

    def test_row_offset_gathers_the_offset_rows(self):
        """``row_offset`` (GLM5_MKD_STACK_C): per-layer indices + an in-kernel
        row offset gather exactly what ``indices + row_offset`` gathers from the
        merged stack, for every live row; offset 0 equals no offset; a wrong
        offset (positive control) gathers other rows."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        n_layers, pages, page_size = 3, 32, 256
        layer_tokens = pages * page_size
        query_tokens, topk = 32, 2048
        count = query_tokens * topk
        rng = np.random.default_rng(20261023)
        nope_u8 = rng.integers(0, 256, (n_layers * pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(
            0, 0x7E, (n_layers * pages, page_size // 4, 4, 128), np.uint8
        )
        rope_u8[..., 64:] = 0
        nope_i32 = jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8))
        rope_i32 = jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope_u8))
        indices = jnp.asarray(
            rng.integers(0, layer_tokens, count, dtype=np.int32).reshape(
                query_tokens, topk
            )
        )
        num_live = 20 * topk + 3 * 64
        groups = num_live // native_sc_gather.ROPE_PACKING

        def gather(idx, **kw):
            return jax.block_until_ready(
                native_sc_gather.dsa_gather_native_sc(
                    nope_i32,
                    rope_i32,
                    idx,
                    out_size=count,
                    atoms_per_batch=16,
                    num_live_indices=jnp.int32(num_live),
                    live_multiple=64,
                    **kw,
                )
            )

        for row in range(n_layers):
            with self.subTest(row=row):
                off = row * layer_tokens
                want_nope, want_rope = gather(indices + off)
                got_nope, got_rope = gather(indices, row_offset=off)
                np.testing.assert_array_equal(
                    np.asarray(got_nope)[:num_live], np.asarray(want_nope)[:num_live]
                )
                np.testing.assert_array_equal(
                    np.asarray(got_rope)[:groups], np.asarray(want_rope)[:groups]
                )
                wrong_nope, _ = gather(
                    indices, row_offset=((row + 1) % n_layers) * layer_tokens
                )
                self.assertFalse(
                    np.array_equal(
                        np.asarray(wrong_nope)[:num_live],
                        np.asarray(want_nope)[:num_live],
                    ),
                    "positive control: a wrong row offset must gather other rows",
                )

    def test_row_offset_every_live_form_and_kv_write(self):
        """``row_offset`` on the forms the stacked-cache DCP path uses besides
        ``num_live_indices``: ``live_vec_offset`` (the fused decode prep's
        operand, offset in a 16-lane operand of its own) with and without
        ``kv_write``, and ``num_live_indices`` with ``kv_write``. Per-layer
        indices and per-layer write rows + an in-kernel offset == ``indices +
        row_offset`` and ``dst + row_offset`` through the plain kernel: every
        live gathered row and both whole caches (the write lands in the
        layer's row of the stack and the gather reads it back). A wrong
        offset (positive control) moves the writes and the gathered rows."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("native-SC gather requires a TPU SparseCore")

        n_layers, pages, page_size = 3, 16, 256
        layer_tokens = pages * page_size
        query_tokens, topk = 16, 2048
        count = query_tokens * topk
        rng = np.random.default_rng(20261024)
        nope_u8 = rng.integers(0, 256, (n_layers * pages, page_size, 4, 128), np.uint8)
        rope_u8 = rng.integers(
            0, 0x7E, (n_layers * pages, page_size // 4, 4, 128), np.uint8
        )
        rope_u8[..., 64:] = 0
        caches = (
            jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8)),
            jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope_u8)),
        )
        # Per-layer write rows (some skipped), planted at the start of the
        # indices so the gather reads them back first.
        dst = rng.choice(layer_tokens, query_tokens, replace=False).astype(np.int32)
        dst[[3, 9]] = -1
        flat = rng.integers(0, layer_tokens, count, dtype=np.int32)
        flat[: query_tokens * 8 : 8] = np.where(
            dst >= 0, dst, flat[: query_tokens * 8 : 8]
        )
        nope_rows = jnp.asarray(
            rng.integers(0, 2**32, (query_tokens, 128), np.uint64), jnp.uint32
        )
        rope_rows = jnp.asarray(
            rng.integers(0, 2**32, (query_tokens, 32), np.uint64), jnp.uint32
        )
        num_live = 9 * topk
        groups = num_live // native_sc_gather.ROPE_PACKING
        _, _, _, occ, nb = native_sc_gather.live_vec_geometry(count, 1)
        live_b = min(-(-num_live // occ), nb)
        meta = np.zeros((8, 128), np.int32)
        meta[0, :16] = live_b
        meta[0, 16:] = -(-live_b // 32)

        def gather(indices, form, write_rows=None, **kw):
            if form == "vec_offset":
                operand = jnp.concatenate(
                    [indices.reshape(count // 128, 128), jnp.asarray(meta)], 0
                )
                live = dict(live_vec_offset=count, linear_indices=True)
            else:
                operand = indices.reshape(query_tokens, topk)
                live = dict(num_live_indices=jnp.int32(num_live))
            kv = None
            if write_rows is not None:
                kv = native_sc_gather.pack_write_rows(
                    jnp.asarray(write_rows), nope_rows, rope_rows
                )
            return jax.block_until_ready(
                native_sc_gather.dsa_gather_native_sc(
                    *caches,
                    operand,
                    out_size=count,
                    atoms_per_batch=1,
                    live_multiple=topk,
                    kv_write=kv,
                    **live,
                    **kw,
                )
            )

        def same(a, b):
            n = [
                np.array_equal(
                    np.asarray(a[0])[:num_live], np.asarray(b[0])[:num_live]
                ),
                np.array_equal(np.asarray(a[1])[:groups], np.asarray(b[1])[:groups]),
            ]
            n += [
                np.array_equal(np.asarray(x), np.asarray(y))
                for x, y in zip(a[2:], b[2:])
            ]
            return n

        for form, write in (
            ("vec_offset", False),
            ("vec_offset", True),
            ("num_live", True),
        ):
            for row in range(n_layers):
                with self.subTest(form=form, write=write, row=row):
                    off = row * layer_tokens
                    ref_dst = np.where(dst >= 0, dst + off, dst) if write else None
                    want = gather(jnp.asarray(flat + off), form, ref_dst)
                    got = gather(
                        jnp.asarray(flat), form, dst if write else None, row_offset=off
                    )
                    self.assertEqual(same(got, want), [True] * len(want))
                    bad = ((row + 1) % n_layers) * layer_tokens
                    wrong = gather(
                        jnp.asarray(flat), form, dst if write else None, row_offset=bad
                    )
                    res = same(wrong, want)
                    self.assertFalse(res[0], "positive control: wrong offset gathered")
                    if write:
                        self.assertFalse(res[2], "positive control: wrong offset wrote")

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
