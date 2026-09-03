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
"""Insert-layout tests for the sparse-MLA KV cache writers.

Pure jnp scatter, no Pallas: these run on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout, KVCacheType, SparseMLAKVCacheSpec,
    update_sparse_mla_kv_cache)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16
WORD_BYTES = 4

# Layouts pinned explicitly rather than read from the environment: these
# tests check the exact geometry dsa_gather is written against.
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, ROPE_DIM, KV_PACKING)


def _empty(spec: SparseMLAKVCacheSpec) -> jax.Array:
    return jnp.zeros(spec.shape, spec.jax_dtype)


def _token_bytes(cache: jax.Array, spec: SparseMLAKVCacheSpec) -> np.ndarray:
    """Decode any layout to `[num_pages, page_size, head_dim]` uint8.

    Decoding is written from the layout contract, not from the writer: a
    sparsecore word `w` holds byte `b` of feature band `b`, so recovering
    token-major bytes needs the band axis transposed back in front of the
    feature axis. A writer that packed in any other order fails the
    comparisons below.
    """
    arr = np.asarray(cache)
    rows = arr.reshape(spec.num_pages, spec.page_size, -1)
    if spec.layout is KVCacheLayout.TENSORCORE:
        # nope [P, S, 4, 128] and rope [P, S // 4, 4, 128] are both already
        # byte images -- the flatten above is the whole decode.
        return rows
    bands = rows.view(np.uint8).reshape(*rows.shape, WORD_BYTES)
    return bands.transpose(0, 1, 3, 2).reshape(spec.num_pages, spec.page_size,
                                               spec.head_dim)


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


class UpdateSparseMLAKvCacheTest(parameterized.TestCase):
    """Insert-layout unit test; pure jnp scatter, runs on CPU too."""

    @parameterized.named_parameters(
        dict(testcase_name="nope_sc_rope_tc",
             nope_layout=KVCacheLayout.SPARSECORE,
             rope_layout=KVCacheLayout.TENSORCORE),
        dict(testcase_name="nope_sc_rope_sc",
             nope_layout=KVCacheLayout.SPARSECORE,
             rope_layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="nope_tc_rope_sc",
             nope_layout=KVCacheLayout.TENSORCORE,
             rope_layout=KVCacheLayout.SPARSECORE),
        dict(testcase_name="nope_tc_rope_tc",
             nope_layout=KVCacheLayout.TENSORCORE,
             rope_layout=KVCacheLayout.TENSORCORE),
    )
    def test_rows_land_at_page_slot(self, nope_layout, rope_layout):
        """Every layout combination puts a token's bytes at the same
        (page, slot); only the address arithmetic differs."""
        nope_spec = SparseMLAKVCacheSpec.create(KVCacheType.NOPE, nope_layout,
                                                TOTAL_PAGES, PAGE_SIZE,
                                                LKV_DIM, KV_PACKING)
        rope_spec = SparseMLAKVCacheSpec.create(KVCacheType.ROPE, rope_layout,
                                                TOTAL_PAGES, PAGE_SIZE,
                                                ROPE_DIM, KV_PACKING)
        rng = np.random.default_rng(7)
        q_lens, seq_lens = [5, 3], [37, 3]  # seq 0 appends across a page edge
        total = sum(q_lens)
        kv_c = _quantize_fp8(
            rng.standard_normal((total, LKV_DIM)).astype(np.float32), 1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total, ROPE_DIM)).astype(np.float32), 1.0)
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, :2] = [5, 2]
        block_tables[1, 0] = 9

        nope_cache, rope_cache = update_sparse_mla_kv_cache(
            _empty(nope_spec),
            _empty(rope_spec),
            kv_c,
            k_pe,
            jnp.asarray(seq_lens, jnp.int32),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray([0, 5, 8], jnp.int32),
            nope_spec=nope_spec,
            rope_spec=rope_spec)

        # Every layout is token-major once decoded to [page, slot, bytes].
        nope_rows = _token_bytes(nope_cache, nope_spec)
        rope_rows = _token_bytes(rope_cache, rope_spec)
        exp_nope = np.asarray(jax.lax.bitcast_convert_type(kv_c, jnp.uint8))
        exp_rope = np.asarray(jax.lax.bitcast_convert_type(k_pe, jnp.uint8))
        # seq 0: positions 32..36 -> page 2 slots 0..4; seq 1: page 9 slots 0..2.
        placements = [(2, i, i) for i in range(5)]
        placements += [(9, i, 5 + i) for i in range(3)]
        touched = np.zeros((TOTAL_PAGES, PAGE_SIZE), bool)
        for page, slot, token in placements:
            np.testing.assert_array_equal(nope_rows[page, slot],
                                          exp_nope[token])
            np.testing.assert_array_equal(rope_rows[page, slot, :ROPE_DIM],
                                          exp_rope[token])
            # Pad tail is never written (zero fill).
            self.assertEqual(
                int(np.count_nonzero(rope_rows[page, slot, ROPE_DIM:])), 0)
            touched[page, slot] = True
        self.assertEqual(int(np.count_nonzero(nope_rows[~touched])), 0)
        self.assertEqual(int(np.count_nonzero(rope_rows[~touched])), 0)

    def test_padded_batch_is_dropped(self):
        """Padded tokens (valid=False) must be dropped, never written."""
        rng = np.random.default_rng(42)
        total_tokens, valid_tokens = 16, 6
        kv_c = _quantize_fp8(
            rng.standard_normal((total_tokens, LKV_DIM)).astype(np.float32),
            1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total_tokens, ROPE_DIM)).astype(np.float32),
            1.0)
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, 0] = 1
        block_tables[1, 0] = 2

        nope_cache, rope_cache = update_sparse_mla_kv_cache(
            _empty(NOPE_SPEC),
            _empty(ROPE_SPEC),
            kv_c,
            k_pe,
            jnp.asarray([4, 2], jnp.int32),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray([0, 4, valid_tokens], jnp.int32),
            nope_spec=NOPE_SPEC,
            rope_spec=ROPE_SPEC)

        nope_rows = _token_bytes(nope_cache, NOPE_SPEC).reshape(-1, LKV_DIM)
        rope_rows = _token_bytes(rope_cache,
                                 ROPE_SPEC).reshape(-1, ROPE_SPEC.head_dim)
        # Exactly the valid tokens landed, nothing else.
        self.assertEqual(int(nope_rows.any(axis=1).sum()), valid_tokens)
        self.assertEqual(int(rope_rows.any(axis=1).sum()), valid_tokens)

    @parameterized.named_parameters(
        dict(testcase_name="nope",
             cache_type=KVCacheType.NOPE,
             head_dim=LKV_DIM),
        dict(testcase_name="rope",
             cache_type=KVCacheType.ROPE,
             head_dim=ROPE_DIM),
    )
    def test_sparsecore_words_are_little_endian_pack4(self, cache_type,
                                                      head_dim):
        """Each sparsecore word is pack4 of the token's four byte bands."""
        specs = {
            layout:
            SparseMLAKVCacheSpec.create(cache_type, layout, TOTAL_PAGES,
                                        PAGE_SIZE, head_dim, KV_PACKING)
            for layout in KVCacheLayout
        }
        is_nope = cache_type is KVCacheType.NOPE
        rng = np.random.default_rng(11)
        num_tokens = 6
        kv_c = _quantize_fp8(
            rng.standard_normal((num_tokens, LKV_DIM)).astype(np.float32), 1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((num_tokens, ROPE_DIM)).astype(np.float32),
            1.0)
        values = kv_c if is_nope else k_pe
        block_tables = np.full((1, PAGES_PER_SEQ), 3, np.int32)

        caches = {}
        for layout, spec in specs.items():
            nope_spec = spec if is_nope else NOPE_SPEC
            rope_spec = ROPE_SPEC if is_nope else spec
            nope, rope = update_sparse_mla_kv_cache(
                _empty(nope_spec),
                _empty(rope_spec),
                kv_c,
                k_pe,
                jnp.asarray([num_tokens], jnp.int32),
                jnp.asarray(block_tables.reshape(-1)),
                jnp.asarray([0, num_tokens], jnp.int32),
                nope_spec=nope_spec,
                rope_spec=rope_spec)
            caches[layout] = nope if is_nope else rope

        # Reference: pack the four bands of each written token by hand.
        sc_spec = specs[KVCacheLayout.SPARSECORE]
        sc_words = np.asarray(caches[KVCacheLayout.SPARSECORE]).reshape(
            TOTAL_PAGES, PAGE_SIZE, -1)
        raw = np.asarray(jax.lax.bitcast_convert_type(values, jnp.uint8))
        padded = np.zeros((num_tokens, sc_spec.head_dim), np.uint8)
        padded[:, :raw.shape[1]] = raw
        bands = padded.reshape(num_tokens, WORD_BYTES, -1).astype(np.uint32)
        expected = (bands[:, 0] | (bands[:, 1] << 8) | (bands[:, 2] << 16)
                    | (bands[:, 3] << 24))
        for token in range(num_tokens):
            np.testing.assert_array_equal(sc_words[3, token], expected[token])

        # ...and the two layouts hold the same bytes, differing only in how
        # the 32-bit packing is expressed.
        np.testing.assert_array_equal(
            _token_bytes(caches[KVCacheLayout.SPARSECORE],
                         specs[KVCacheLayout.SPARSECORE]),
            _token_bytes(caches[KVCacheLayout.TENSORCORE],
                         specs[KVCacheLayout.TENSORCORE]))
