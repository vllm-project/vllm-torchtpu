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
"""Correctness tests for the indexer K cache writer (`kv_cache_append`).

Every test compares the whole cache, bit for bit, against a golden that does
not use the kernel. Caches start random, so bytes the writer must leave alone
are checked too. Two goldens:

  - `_has_append_4d` / `_sal_append_4d`: today's XLA append expressions. Used
    for well-formed batches, to show the writer is a drop-in replacement.
  - `_reference`: a per-token NumPy address oracle feeding
    `scatter_rows_reference` (XLA scatter, out-of-range rows dropped). Used for
    everything else: padding, invalid positions and pages, and DCP.

Performance lives in `scripts/benchmark_indexer_kv_cache_append.py`.
"""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_append,
    kv_cache_utils,
)

_KV_PACKING = 4
_HAS = "head_along_sublane"
_SAL = "seq_along_lane"
# Every correctness test runs once per layout.
_LAYOUTS = (_HAS, _SAL)


def _xla_page_and_slot(num_tokens, seq_lens, page_indices, cu_q_lens, page_size):
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
    q_len = cu_q_lens[seq + 1] - cu_q_lens[seq]
    pos = seq_lens[seq] - q_len + (tok - cu_q_lens[seq])
    page = jnp.where(
        tok < cu_q_lens[-1],
        page_indices.reshape(seq_lens.shape[0], -1)[seq, pos // page_size],
        jnp.int32(2**30),  # Batch padding: dropped by the scatter.
    )
    return page, pos % page_size


def _has_append_4d(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    """`HEAD_ALONG_SUBLANE`: `cache.at[page, slot // 4, slot % 4].set(k)`."""
    _, quads_per_page, _, lkv_dim = cache_kv.shape
    width = k_packed.shape[-1]
    k = (
        jnp.pad(k_packed, ((0, 0), (0, lkv_dim - width)))
        if width < lkv_dim
        else k_packed[:, :lkv_dim]
    )
    page, slot = _xla_page_and_slot(
        k.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        quads_per_page * _KV_PACKING,
    )
    return cache_kv.at[page, slot // _KV_PACKING, slot % _KV_PACKING].set(k)


def _sal_append_4d(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    """`SEQ_ALONG_LANE`: `cache.at[page, :, :, slot].set(k_record)`."""
    _, sublane_groups, _, page_size = cache_kv.shape
    head_dim = (sublane_groups - 1) * _KV_PACKING
    rec_dim = sublane_groups * _KV_PACKING
    width = k_packed.shape[-1]
    if width == rec_dim:
        k = k_packed
    elif width <= head_dim + 1:
        k = jnp.pad(k_packed, ((0, 0), (0, rec_dim - width)))
    else:  # Keep the head bytes and the first scale byte.
        k = jnp.pad(k_packed[:, : head_dim + 1], ((0, 0), (0, _KV_PACKING - 1)))
    page, slot = _xla_page_and_slot(
        k.shape[0], seq_lens, page_indices, cu_q_lens, page_size
    )
    return cache_kv.at[page, :, :, slot].set(k.reshape(-1, sublane_groups, _KV_PACKING))


_TODAYS_APPEND = {
    _HAS: _has_append_4d,
    _SAL: _sal_append_4d,
}


def _in_layout(cache_has: jax.Array, layout: str) -> jax.Array:
    """`cache_has` in `layout`, via the reader's own conversion."""
    if layout == _HAS:
        return cache_has
    return st_lib.convert_cache_to_seq_along_lane(cache_has, actual_head_dim=128)


def _random_cache(
    rng: np.random.Generator, num_pages: int, page_size: int, lkv_dim=256
) -> jax.Array:
    """Random `HEAD_ALONG_SUBLANE` cache `[num_pages, page_size // 4, 4, W]`."""
    return jnp.asarray(
        rng.integers(
            0, 256, size=(num_pages, page_size // 4, 4, lkv_dim), dtype=np.uint8
        )
    )


def _random_rows(
    rng: np.random.Generator, num_tokens: int, width: int = 129
) -> jax.Array:
    return jnp.asarray(rng.integers(0, 256, size=(num_tokens, width), dtype=np.uint8))


def _cu_q_lens(q_lens) -> jax.Array:
    return jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)


@functools.partial(jax.jit, static_argnames=("layout",), donate_argnums=0)
def _append_jit(cache, k, seq_lens, page_indices, cu_q_lens, *, layout):
    return kv_cache_append.coalesced_kv_append(
        cache, k, seq_lens, page_indices, cu_q_lens, kv_layout=layout
    )


@functools.partial(
    jax.jit,
    static_argnames=("layout", "dcp_size", "interleave_size"),
    donate_argnums=0,
)
def _append_dcp_jit(
    cache,
    k,
    seq_lens,
    page_indices,
    cu_q_lens,
    dcp_rank,
    *,
    layout,
    dcp_size,
    interleave_size,
):
    return kv_cache_append.coalesced_kv_append_dcp(
        cache,
        k,
        seq_lens,
        page_indices,
        cu_q_lens,
        dcp_rank,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
        kv_layout=layout,
    )


def _append(
    cache,
    k,
    seq_lens,
    page_indices,
    cu_q_lens,
    *,
    layout,
    dcp_size=1,
    dcp_rank=0,
    interleave_size=1,
) -> jax.Array:
    """Runs the writer as serving does: jitted, cache donated, rank traced."""
    if dcp_size == 1:
        return _append_jit(
            jnp.copy(cache), k, seq_lens, page_indices, cu_q_lens, layout=layout
        )
    return _append_dcp_jit(
        jnp.copy(cache),
        k,
        seq_lens,
        page_indices,
        cu_q_lens,
        jnp.int32(dcp_rank),
        layout=layout,
        dcp_size=dcp_size,
        interleave_size=interleave_size,
    )


def _numpy_dst_rows(
    num_tokens: int,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    *,
    page_size: int,
    num_pages: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    interleave_size: int = 1,
) -> np.ndarray:
    """Per-token loop oracle for the address layer, written independently of it.

    A token is written iff it belongs to a sequence, its KV position is `>= 0`,
    its page ordinal is inside the page-table row, its page id is valid, and
    (DCP) this rank owns its position.
    """
    seq_lens = np.asarray(seq_lens)
    cu = np.asarray(cu_q_lens)
    num_seqs = seq_lens.shape[0]
    table = np.asarray(page_indices).reshape(num_seqs, -1)
    dst = np.full((num_tokens,), kv_cache_utils.SKIP_ROW, np.int32)
    for s in range(num_seqs):
        q_len = int(cu[s + 1] - cu[s])
        for j in range(q_len):
            tok = int(cu[s]) + j
            pos = int(seq_lens[s]) - q_len + j
            if tok >= num_tokens or pos < 0:
                continue
            if dcp_size == 1:
                ordinal, slot, owned, page_limit = (
                    pos // page_size,
                    pos % page_size,
                    True,
                    num_pages,
                )
            else:
                ordinal, offset = divmod(pos, page_size * dcp_size)
                cycle, within = divmod(offset, dcp_size * interleave_size)
                owned = within // interleave_size == dcp_rank
                slot = cycle * interleave_size + within % interleave_size
                page_limit = num_pages * dcp_size
            if not owned or ordinal >= table.shape[1]:
                continue
            raw_page = int(table[s, ordinal])
            if not 0 <= raw_page < page_limit:
                continue
            dst[tok] = (raw_page % num_pages) * page_size + slot
    return dst


def _reference(
    cache, k, seq_lens, page_indices, cu_q_lens, *, layout, **dcp
) -> jax.Array:
    """XLA-scatter golden driven by the NumPy address oracle."""
    page_size = kv_cache_append.cache_page_size(cache, layout)
    dst_rows = _numpy_dst_rows(
        k.shape[0],
        seq_lens,
        page_indices,
        cu_q_lens,
        page_size=page_size,
        num_pages=cache.shape[0],
        **dcp,
    )
    return kv_cache_append.scatter_rows_reference(
        cache, k, jnp.asarray(dst_rows), layout
    )


def _uniform_batch(
    num_seqs: int,
    tokens_per_seq: int,
    base_seq_len: int,
    unaligned_offset: int = 0,
    width: int = 129,
    page_size: int = 1024,
):
    """`num_seqs` sequences of `tokens_per_seq` new tokens on distinct pages.

    Sequence lengths are staggered around `base_seq_len` so their write
    positions have different phases within a page.
    """
    pages_per_seq = (base_seq_len + page_size - 1) // page_size + 2
    num_pages = max(68, num_seqs * pages_per_seq + 4)
    rng = np.random.default_rng(42 + num_seqs * 100 + tokens_per_seq)
    cache = _random_cache(rng, num_pages, page_size)
    k = _random_rows(rng, num_seqs * tokens_per_seq, width)
    pages = rng.permutation(num_pages)[: num_seqs * pages_per_seq]
    seq_lens = np.maximum(
        tokens_per_seq,
        base_seq_len - (np.arange(num_seqs) * 16) % 64 + unaligned_offset,
    )
    return (
        cache,
        k,
        jnp.asarray(seq_lens, jnp.int32),
        jnp.asarray(pages.reshape(num_seqs, pages_per_seq), jnp.int32),
        _cu_q_lens(np.full(num_seqs, tokens_per_seq)),
    )


# (name, num_seqs, tokens_per_seq, base_seq_len, unaligned_offset, width)
_TODAYS_SHAPES = (
    ("decode_b16", 16, 1, 1024, 0, 129),
    ("decode_b64", 64, 1, 4096, 1, 129),
    ("decode_b128", 128, 1, 4096, 2, 129),
    ("decode_b256", 256, 1, 4096, 3, 129),
    ("prefill_t1024_aligned", 1, 1024, 61440, 0, 129),
    ("prefill_t1024_page_crossing", 1, 1024, 61184, 0, 129),
    ("prefill_t1024_unaligned_1", 1, 1024, 61185, 1, 129),
    ("prefill_t1024_unaligned_2", 1, 1024, 61186, 2, 129),
    ("prefill_t1024_unaligned_3", 1, 1024, 61187, 3, 129),
    ("prefill_t4096_aligned", 1, 4096, 61440, 0, 129),
    ("prefill_t4096_unaligned_1", 1, 4096, 61185, 1, 129),
    ("prefill_b2_t512", 2, 512, 8192, 0, 129),
    ("prefill_b4_t256", 4, 256, 8192, 0, 129),
    ("prefill_b16_t64", 16, 64, 4096, 0, 129),
    ("prefill_b64_t16", 64, 16, 2048, 0, 129),
    # 256-byte input rows (already padded to the HAS record width).
    ("decode_b16_wide_rows", 16, 1, 1024, 0, 256),
    ("prefill_t1024_unaligned_wide_rows", 1, 1024, 61185, 1, 256),
)


class CoalescedKvAppendTest(parameterized.TestCase):
    def _assert_matches_reference(
        self, cache_has, k, seq_lens, page_indices, cu_q_lens, **dcp
    ):
        """Checks the writer against `_reference` in every layout.

        Also checks that the reference changed the cache whenever the oracle has
        a writable token, so a setup that skips everything can't pass vacuously.
        """
        writes_expected = bool(
            np.any(
                _numpy_dst_rows(
                    k.shape[0],
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    page_size=kv_cache_append.cache_page_size(cache_has, _HAS),
                    num_pages=cache_has.shape[0],
                    **dcp,
                )
                >= 0
            )
        )
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            expected = np.asarray(
                _reference(
                    cache,
                    k,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    layout=layout,
                    **dcp,
                )
            )
            self.assertEqual(
                not np.array_equal(expected, np.asarray(cache)),
                writes_expected,
                msg=f"reference writes vs oracle: layout={layout} {dcp}",
            )
            np.testing.assert_array_equal(
                np.asarray(
                    _append(
                        cache,
                        k,
                        seq_lens,
                        page_indices,
                        cu_q_lens,
                        layout=layout,
                        **dcp,
                    )
                ),
                expected,
                err_msg=f"layout={layout} {dcp}",
            )

    # ---------------------------------------------------------------------------
    # Drop-in parity with today's XLA append (R1).
    # ---------------------------------------------------------------------------

    @parameterized.named_parameters(*_TODAYS_SHAPES)
    def test_matches_todays_xla_append(
        self, num_seqs, tokens_per_seq, base_seq_len, unaligned_offset, width
    ):
        cache_has, k, seq_lens, page_indices, cu_q_lens = _uniform_batch(
            num_seqs, tokens_per_seq, base_seq_len, unaligned_offset, width
        )
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            expected = _TODAYS_APPEND[layout](
                jnp.copy(cache), k, seq_lens, page_indices, cu_q_lens
            )
            actual = _append(cache, k, seq_lens, page_indices, cu_q_lens, layout=layout)
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(expected), err_msg=layout
            )

    @parameterized.named_parameters(*_TODAYS_SHAPES)
    def test_sal_equals_reader_conversion_of_has(
        self, num_seqs, tokens_per_seq, base_seq_len, unaligned_offset, width
    ):
        """SAL output is exactly what `streamindex_topk` expects to read.

        `convert_cache_to_seq_along_lane` is the reader's own HAS -> SAL mapping,
        so this checks the SAL writer independently of any SAL golden.
        """
        cache_has, k, seq_lens, page_indices, cu_q_lens = _uniform_batch(
            num_seqs, tokens_per_seq, base_seq_len, unaligned_offset, width
        )
        out_has = _append(cache_has, k, seq_lens, page_indices, cu_q_lens, layout=_HAS)
        out_sal = _append(
            _in_layout(cache_has, _SAL),
            k,
            seq_lens,
            page_indices,
            cu_q_lens,
            layout=_SAL,
        )
        np.testing.assert_array_equal(
            np.asarray(out_sal), np.asarray(_in_layout(out_has, _SAL))
        )

    @parameterized.named_parameters(
        ("decode_b16", 16, 1, 1024, 0),
        ("prefill_t1024_unaligned_1", 1, 1024, 61185, 1),
        ("prefill_b16_t64", 16, 64, 4096, 0),
    )
    def test_comparison_catches_off_by_one_slot_and_wrong_rank(
        self, num_seqs, tokens_per_seq, base_seq_len, unaligned_offset
    ):
        """Positive control (R8): the byte-exact comparison is able to fail.

        Every token written one slot late (`seq_lens + 1`), or by the wrong DCP
        rank, must differ from the golden in every layout.
        """
        cache_has, k, seq_lens, page_indices, cu_q_lens = _uniform_batch(
            num_seqs, tokens_per_seq, base_seq_len, unaligned_offset
        )
        dcp = dict(dcp_size=2, interleave_size=4)
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            args = (k, seq_lens, page_indices, cu_q_lens)
            shifted = (k, seq_lens + 1, page_indices, cu_q_lens)
            self.assertFalse(
                np.array_equal(
                    np.asarray(_append(cache, *shifted, layout=layout)),
                    np.asarray(_reference(cache, *args, layout=layout)),
                ),
                msg=f"one-slot shift not caught: {layout}",
            )
            self.assertFalse(
                np.array_equal(
                    np.asarray(
                        _append(cache, *args, layout=layout, dcp_rank=1, **dcp)
                    ),
                    np.asarray(
                        _reference(cache, *args, layout=layout, dcp_rank=0, **dcp)
                    ),
                ),
                msg=f"wrong DCP rank not caught: {layout}",
            )

    # ---------------------------------------------------------------------------
    # Serving batch shapes.
    # ---------------------------------------------------------------------------

    @parameterized.named_parameters(
        # (name, num_decodes, total_tokens, base_seq_len, prefill_pos_offset)
        ("8k_63x1_plus_961", 63, 1024, 8192, 37),
        ("8k_61x1_plus_963_quad_misaligned", 61, 1024, 8192, 2),
        ("128k_8x1_plus_1016", 8, 1024, 131072, 91),
        ("128k_15x1_plus_1009", 15, 1024, 131072, 5),
    )
    def test_serving_mixed_batch(
        self, num_decodes, total_tokens, base_seq_len, prefill_pos_offset
    ):
        """Steady-state continuous batching: decodes first, then one chunk.

        The chunk starts at token `num_decodes` (not 128-aligned, often not
        4-aligned) with an unrelated position phase, as vLLM schedules it once
        the decodes have used part of the token budget. In `HEAD_ALONG_SUBLANE`
        the chunk alone is ~241 quad runs, i.e. several waves.
        """
        page_size = 1024
        num_seqs = num_decodes + 1
        pages_per_seq = (base_seq_len + prefill_pos_offset) // page_size + 2
        num_pages = num_seqs * pages_per_seq + 4
        rng = np.random.default_rng(1234 + num_decodes)
        page_indices = jnp.asarray(
            rng.permutation(num_pages)[: num_seqs * pages_per_seq].reshape(
                num_seqs, pages_per_seq
            ),
            jnp.int32,
        )
        seq_lens = rng.integers(base_seq_len // 2, base_seq_len, size=num_seqs)
        seq_lens[-1] = base_seq_len + prefill_pos_offset
        q_lens = np.ones(num_seqs, np.int32)
        q_lens[-1] = total_tokens - num_decodes
        self._assert_matches_reference(
            _random_cache(rng, num_pages, page_size),
            _random_rows(rng, total_tokens),
            jnp.asarray(seq_lens, jnp.int32),
            page_indices,
            _cu_q_lens(q_lens),
        )

    def test_mixed_batch_with_misaligned_phases(self):
        """Token index and cache position disagree mod 4 and mod 128, per seq."""
        rng = np.random.default_rng(999)
        self._assert_matches_reference(
            _random_cache(rng, 32, 1024),
            _random_rows(rng, 1028),
            jnp.array([1021, 1535, 2048, 2561], jnp.int32),
            jnp.arange(16, dtype=jnp.int32).reshape(4, 4),
            jnp.array([0, 1, 514, 515, 1028], jnp.int32),
        )

    @parameterized.parameters((2, 0), (2, 7), (4, 0), (4, 3), (8, 0), (8, 125))
    def test_spec_decode_shapes(self, q_len, pos_offset):
        """Spec-decode / MTP: `q_len` in {2, 4, 8} at aligned and unaligned slots."""
        self._assert_matches_reference(
            *_uniform_batch(32, q_len, 1024 + pos_offset, pos_offset)
        )

    @parameterized.parameters((2, 1, 1), (4, 2, 3), (8, 3, 125))
    def test_mtp_step_to_step_token_rewrite(self, q_len, accepted, pos_offset):
        """MTP across consecutive steps (R5): rejected draft slots are overwritten."""
        rng = np.random.default_rng(303 + q_len * 10 + accepted)
        num_seqs, page_size, num_pages = 8, 256, 32
        cache_has = _random_cache(rng, num_pages, page_size)
        page_indices = jnp.arange(num_pages, dtype=jnp.int32).reshape(num_seqs, 4)
        cu_q_lens = _cu_q_lens(np.full(num_seqs, q_len))
        seq_lens_step1 = jnp.full((num_seqs,), 128 + pos_offset + q_len, jnp.int32)
        # Step 2 rewrites rejected draft positions (q_len - accepted) and appends new.
        seq_lens_step2 = seq_lens_step1 - (q_len - accepted) + q_len
        k_step1 = _random_rows(rng, num_seqs * q_len)
        k_step2 = _random_rows(rng, num_seqs * q_len)

        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            actual = _append(
                cache, k_step1, seq_lens_step1, page_indices, cu_q_lens, layout=layout
            )
            actual = _append(
                actual, k_step2, seq_lens_step2, page_indices, cu_q_lens, layout=layout
            )
            expected = _reference(
                cache, k_step1, seq_lens_step1, page_indices, cu_q_lens, layout=layout
            )
            expected = _reference(
                expected,
                k_step2,
                seq_lens_step2,
                page_indices,
                cu_q_lens,
                layout=layout,
            )
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(expected), err_msg=layout
            )

    def test_multi_step_partial_granule_rmw(self):
        """Multi-step partial-granule RMW (R3): neighbor slots survive across steps.

        Step 1 writes slot 1 of quad 0 (and tile 0); Step 2 writes slots 2..3 of
        the same quad/tile. Slot 0 (untouched) and slot 1 (from Step 1) must be
        preserved bit-for-bit after Step 2.
        """
        rng = np.random.default_rng(515)
        cache_has = _random_cache(rng, 4, 256)
        page_indices = jnp.array([[2]], jnp.int32)
        k_step1 = _random_rows(rng, 1)
        k_step2 = _random_rows(rng, 2)

        for layout in _LAYOUTS:
            cache0 = _in_layout(cache_has, layout)
            # Step 1: write 1 token at pos=1 (seq_len=2, q_len=1).
            cache1 = _append(
                cache0,
                k_step1,
                jnp.array([2], jnp.int32),
                page_indices,
                jnp.array([0, 1], jnp.int32),
                layout=layout,
            )
            # Step 2: write 2 tokens at pos=2, 3 (seq_len=4, q_len=2).
            cache2 = _append(
                cache1,
                k_step2,
                jnp.array([4], jnp.int32),
                page_indices,
                jnp.array([0, 2], jnp.int32),
                layout=layout,
            )
            ref1 = _reference(
                cache0,
                k_step1,
                jnp.array([2], jnp.int32),
                page_indices,
                jnp.array([0, 1], jnp.int32),
                layout=layout,
            )
            ref2 = _reference(
                ref1,
                k_step2,
                jnp.array([4], jnp.int32),
                page_indices,
                jnp.array([0, 2], jnp.int32),
                layout=layout,
            )
            np.testing.assert_array_equal(
                np.asarray(cache1), np.asarray(ref1), err_msg=f"{layout} step1"
            )
            np.testing.assert_array_equal(
                np.asarray(cache2), np.asarray(ref2), err_msg=f"{layout} step2"
            )

    @parameterized.parameters((64, 1), (64, 7), (64, 16), (256, 12))
    def test_padded_max_num_seqs(self, max_num_seqs, active_seqs):
        """Inactive sequence slots (`q_len = seq_len = 0`, junk pages) and padding.

        `k_packed` also has padding rows past `cu_q_lens[-1]`.
        """
        pages_per_seq = 4
        num_pages = active_seqs * pages_per_seq + 4
        rng = np.random.default_rng(101 + max_num_seqs + active_seqs)
        q_lens = np.zeros(max_num_seqs, np.int32)
        q_lens[:active_seqs] = rng.choice([1, 4, 16, 128], size=active_seqs)
        seq_lens = np.zeros(max_num_seqs, np.int32)
        seq_lens[:active_seqs] = q_lens[:active_seqs] + rng.integers(
            0, 2048, size=active_seqs
        )
        table = np.full((max_num_seqs, pages_per_seq), 999999, np.int32)
        table[:active_seqs] = np.arange(active_seqs * pages_per_seq).reshape(
            active_seqs, pages_per_seq
        )
        self._assert_matches_reference(
            _random_cache(rng, num_pages, 1024),
            _random_rows(rng, max(int(q_lens.sum()) + 16, 128)),
            jnp.asarray(seq_lens),
            jnp.asarray(table),
            _cu_q_lens(q_lens),
        )

    def test_flat_and_2d_page_tables_agree(self):
        cache_has, k, seq_lens, page_indices, cu_q_lens = _uniform_batch(16, 4, 2048, 3)
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            np.testing.assert_array_equal(
                np.asarray(
                    _append(
                        cache,
                        k,
                        seq_lens,
                        page_indices.reshape(-1),
                        cu_q_lens,
                        layout=layout,
                    )
                ),
                np.asarray(
                    _append(cache, k, seq_lens, page_indices, cu_q_lens, layout=layout)
                ),
                err_msg=layout,
            )

    # ---------------------------------------------------------------------------
    # Tokens that must not be written (R2).
    # ---------------------------------------------------------------------------

    def test_invalid_positions_and_pages_are_skipped(self):
        """Out-of-range pages, `-1` pages, `pos < 0`, and past-the-table ordinals."""
        num_pages = 12
        rng = np.random.default_rng(404)
        cache_has = _random_cache(rng, num_pages, 1024)

        # Decode: seqs 0-2 are valid. Seq 3's position (2999) is past its 2-page
        # table row, so it must not read seq 4's row. Seq 4's page (99999) is out
        # of range, so it must not clamp onto the last page. Seq 5 has pos = -1.
        decode = (
            _random_rows(rng, 6),
            jnp.array([10, 10, 12, 3000, 5, 0], jnp.int32),
            jnp.array(
                [[3, 4], [5, 6], [0, 1], [7, 8], [99999, -1], [9, 10]], jnp.int32
            ),
            jnp.arange(7, dtype=jnp.int32),
        )
        self._assert_matches_reference(cache_has, *decode)
        out = _append(cache_has, *decode, layout=_HAS)
        np.testing.assert_array_equal(
            np.asarray(out[num_pages - 1]), np.asarray(cache_has[num_pages - 1])
        )

        # Multi-token: seq 0 has seq_len 2 < q_len 5, so its first 3 tokens have
        # pos < 0 and the remaining 2 must still be written (and compacted).
        # Seq 3 runs from its valid page into an out-of-range one.
        self._assert_matches_reference(
            cache_has,
            _random_rows(rng, 29),
            jnp.array([2, 20, 20, 2500], jnp.int32),
            jnp.array([[1, 2], [5, 6], [3, 4], [7, 99999]], jnp.int32),
            jnp.array([0, 5, 13, 21, 29], jnp.int32),
        )

    # ---------------------------------------------------------------------------
    # Cache geometry.
    # ---------------------------------------------------------------------------

    @parameterized.parameters(128, 256, 1024)
    def test_page_sizes(self, page_size):
        num_seqs, pages_per_seq = 8, 4
        rng = np.random.default_rng(505 + page_size)
        cache_has = _random_cache(rng, num_seqs * pages_per_seq + 4, page_size)
        page_indices = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
        for q_len, pos_offset in [(128, 0), (1, 13), (37, 19)]:
            self._assert_matches_reference(
                cache_has,
                _random_rows(rng, num_seqs * q_len),
                jnp.full((num_seqs,), page_size + pos_offset, jnp.int32),
                page_indices.reshape(num_seqs, pages_per_seq),
                _cu_q_lens(np.full(num_seqs, q_len)),
            )

    def test_head_dim_64_odd_sublane_groups(self):
        """`head_dim = 64`: HAS rows of 128 B, SAL with 17 (odd) sublane groups."""
        head_dim, page_size, num_pages = 64, 1024, 32
        rng = np.random.default_rng(606)
        cache_has = _random_cache(rng, num_pages, page_size, lkv_dim=128)

        def to_sal(c):
            flat = c.reshape(num_pages, page_size, -1)
            rec = jnp.concatenate(
                [
                    flat[:, :, :head_dim],
                    jnp.pad(
                        flat[:, :, head_dim : head_dim + 1], ((0, 0),) * 2 + ((0, 3),)
                    ),
                ],
                axis=-1,
            )
            return rec.swapaxes(1, 2).reshape(
                num_pages, head_dim // 4 + 1, 4, page_size
            )

        page_indices = jnp.arange(num_pages, dtype=jnp.int32).reshape(8, 4)
        for q_len, pos_offset in [(128, 0), (1, 17), (45, 7)]:
            args = (
                _random_rows(rng, 8 * q_len, head_dim + 1),
                jnp.full((8,), 512 + pos_offset, jnp.int32),
                page_indices,
                _cu_q_lens(np.full(8, q_len)),
            )
            out_has = _append(cache_has, *args, layout=_HAS)
            out_sal = _append(to_sal(cache_has), *args, layout=_SAL)
            np.testing.assert_array_equal(
                np.asarray(out_has),
                np.asarray(_reference(cache_has, *args, layout=_HAS)),
            )
            np.testing.assert_array_equal(
                np.asarray(out_sal), np.asarray(to_sal(out_has))
            )

    # ---------------------------------------------------------------------------
    # Waves and DMA ordering (R6).
    # ---------------------------------------------------------------------------

    @parameterized.parameters(0, 1, 2, 3, 4, 5, 6, 7)
    def test_decode_waves_race_stress(self, seed):
        """256 single-token runs: 2 waves of 128 reads sharing one semaphore.

        Pages and slots are random across a large cache, so a blend that consumed
        a granule before its read landed would show up as a mismatch. Each layout
        runs 4 times on one compiled program.
        """
        page_size, num_seqs, pages_per_seq, num_pages = 1024, 256, 2, 640
        rng = np.random.default_rng(9000 + seed)
        cache_has = _random_cache(rng, num_pages, page_size)
        args = (
            _random_rows(rng, num_seqs),
            jnp.asarray(
                rng.integers(1, pages_per_seq * page_size + 1, size=num_seqs),
                jnp.int32,
            ),
            jnp.asarray(
                rng.permutation(num_pages)[: num_seqs * pages_per_seq].reshape(
                    num_seqs, pages_per_seq
                ),
                jnp.int32,
            ),
            jnp.arange(num_seqs + 1, dtype=jnp.int32),
        )
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            expected = np.asarray(_reference(cache, *args, layout=layout))
            for it in range(4):
                np.testing.assert_array_equal(
                    np.asarray(_append(cache, *args, layout=layout)),
                    expected,
                    err_msg=f"layout={layout}, iteration={it}",
                )

    @parameterized.parameters(512, 1024)
    def test_large_batch_many_waves(self, num_seqs):
        """4 and 8 waves of single-token runs on 128-token pages."""
        page_size = 128
        rng = np.random.default_rng(707 + num_seqs)
        self._assert_matches_reference(
            _random_cache(rng, num_seqs, page_size),
            _random_rows(rng, num_seqs),
            jnp.asarray(rng.integers(1, page_size, size=num_seqs), jnp.int32),
            jnp.arange(num_seqs, dtype=jnp.int32).reshape(num_seqs, 1),
            jnp.arange(num_seqs + 1, dtype=jnp.int32),
        )

    def test_no_full_cache_copy_in_hlo(self):
        """The donated cache is updated in place (R6): no op copies or relayouts it."""
        if jax.default_backend() != "tpu":
            self.skipTest("HLO Pallas in-place aliasing check requires TPU backend")
        cache_has, k, seq_lens, page_indices, cu_q_lens = _uniform_batch(64, 1, 4096)
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            hlo = (
                _append_jit.lower(
                    cache, k, seq_lens, page_indices, cu_q_lens, layout=layout
                )
                .compile()
                .as_text()
            )
            cache_type = "u8[" + ",".join(map(str, cache.shape)) + "]"
            for line in hlo.splitlines():
                if (
                    cache_type in line
                    and "= u8[" in line
                    and any(op in line for op in (" copy(", " reshape(", " transpose("))
                ):
                    self.fail(f"Full-cache copy in {layout} HLO: {line}")

    # ---------------------------------------------------------------------------
    # DCP (R4).
    # ---------------------------------------------------------------------------

    @parameterized.parameters(
        (2, 1, 1), (2, 1, 128), (4, 4, 1), (4, 4, 64), (2, 128, 256)
    )
    def test_dcp_parity_all_ranks(self, dcp_size, interleave_size, q_len):
        page_size, num_seqs, pages_per_seq = 1024, 8, 4
        rng = np.random.default_rng(808 + dcp_size * 10 + interleave_size)
        cache_has = _random_cache(rng, num_seqs * pages_per_seq, page_size)
        args = (
            _random_rows(rng, num_seqs * q_len),
            jnp.full((num_seqs,), page_size * dcp_size + q_len + 13, jnp.int32),
            jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32).reshape(
                num_seqs, pages_per_seq
            ),
            _cu_q_lens(np.full(num_seqs, q_len)),
        )
        for rank in range(dcp_size):
            self._assert_matches_reference(
                cache_has,
                *args,
                dcp_size=dcp_size,
                dcp_rank=rank,
                interleave_size=interleave_size,
            )

    @parameterized.named_parameters(
        ("decode_off0", [1, 1, 1, 1], 16, 0),
        ("decode_off1", [1, 1, 1, 1], 16, 1),
        ("decode_off3", [1, 1, 1, 1], 16, 3),
        ("decode_off253", [1, 1, 1, 1], 16, 253),
        ("decode_off255_page_cross", [1, 1, 1, 1], 16, 255),
        ("chunk_off0", [1, 1, 1, 1021], 1024, 0),
        ("chunk_off1", [1, 1, 1, 1021], 1024, 1),
        ("chunk_off3", [1, 1, 1, 1021], 1024, 3),
        ("chunk_off253_page_cross", [1, 1, 1, 1021], 1024, 253),
        ("chunk_off255_page_cross", [1, 1, 1, 1021], 1024, 255),
        ("single_chunk_off253_page_cross", [1024], 1024, 253),
    )
    def test_agentx_dcp16_interleave4(self, active_q_lens, num_tokens, offset):
        """AgentX serving: `page_size=256`, 16 ranks, `interleave_size=4`, 8 seqs.

        Decode (`[1, 1, 1, 1]` padded to 16 tokens) leaves most ranks with no
        tokens; a chunk gives each rank ~64 tokens in groups of 4 at a global
        stride of 64, which compaction must merge so no two runs share a granule.
        Rank 0's first local slot is `offset` into a page (`253 + 4 > 256` crosses
        pages).
        """
        page_size, dcp_size, interleave_size, max_seqs, pages_per_seq = (
            256,
            16,
            4,
            8,
            4,
        )
        rng = np.random.default_rng(2026 + num_tokens + offset)
        q_lens = np.zeros(max_seqs, np.int32)
        q_lens[: len(active_q_lens)] = active_q_lens
        # Rank 0's first local position is `page_size + offset`; later sequences
        # start one interleave chunk later, i.e. on the next rank.
        local = page_size + offset
        base = (local // interleave_size) * dcp_size * interleave_size + (
            local % interleave_size
        )
        seq_lens = np.zeros(max_seqs, np.int32)
        for i, q_len in enumerate(active_q_lens):
            seq_lens[i] = base + i * interleave_size + q_len
        cache_has = _random_cache(rng, max_seqs * pages_per_seq, page_size)
        args = (
            _random_rows(rng, num_tokens),
            jnp.asarray(seq_lens),
            jnp.arange(max_seqs * pages_per_seq, dtype=jnp.int32).reshape(
                max_seqs, pages_per_seq
            ),
            _cu_q_lens(q_lens),
        )
        for rank in (0, 1, 2, 3, 7, 15):
            self._assert_matches_reference(
                cache_has,
                *args,
                dcp_size=dcp_size,
                dcp_rank=rank,
                interleave_size=interleave_size,
            )

    def test_dcp_rank_is_traced_one_program_for_all_ranks(self):
        traces = []

        def append(cache, k, seq_lens, page_indices, cu_q_lens, rank):
            traces.append(None)
            return kv_cache_append.coalesced_kv_append_dcp(
                cache,
                k,
                seq_lens,
                page_indices,
                cu_q_lens,
                rank,
                dcp_size=4,
                interleave_size=4,
            )

        append = jax.jit(append, donate_argnums=0)
        rng = np.random.default_rng(77)
        cache = _random_cache(rng, 16, 256)
        args = (
            _random_rows(rng, 64),
            jnp.array([700, 900], jnp.int32),
            jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
            jnp.array([0, 1, 64], jnp.int32),
        )
        for rank in range(4):
            np.testing.assert_array_equal(
                np.asarray(append(jnp.copy(cache), *args, jnp.int32(rank))),
                np.asarray(
                    _reference(
                        cache,
                        *args,
                        layout=_HAS,
                        dcp_size=4,
                        dcp_rank=rank,
                        interleave_size=4,
                    )
                ),
            )
        self.assertLen(traces, 1)

    # ---------------------------------------------------------------------------
    # The two layers separately + CPU/interpret fallback (R7).
    # ---------------------------------------------------------------------------

    @parameterized.named_parameters(
        ("unsharded", 1, 1),
        ("dcp2_il1", 2, 1),
        ("dcp4_il4", 4, 4),
        ("dcp16_il4", 16, 4),
        ("dcp2_il128", 2, 128),
    )
    def test_address_layer_matches_numpy_oracle(self, dcp_size, interleave_size):
        """`get_(dcp_)page_and_slot` + `get_dst_rows` vs the per-token oracle.

        Includes batch padding, a zero-length sequence, negative positions, a
        sequence running past its page-table row, and -1 / out-of-range pages.
        """
        page_size, num_pages = 256, 16
        q_lens = np.array([1, 0, 300, 7, 1, 40], np.int32)
        seq_lens = jnp.array(
            [5, 0, 1100 * dcp_size, 3, 700 * dcp_size, 90], jnp.int32
        )
        cu_q_lens = _cu_q_lens(q_lens)
        num_tokens = int(q_lens.sum()) + 13  # batch padding
        rng = np.random.default_rng(31 + dcp_size)
        table = rng.integers(0, num_pages * dcp_size, size=(6, 4)).astype(np.int32)
        table[3, 0] = -1
        table[5, 0] = num_pages * dcp_size  # out of range
        for rank in sorted({0, dcp_size - 1, dcp_size // 2}):
            if dcp_size == 1:
                page, slot = kv_cache_utils.get_page_and_slot(
                    num_tokens,
                    seq_lens,
                    jnp.asarray(table),
                    cu_q_lens,
                    page_size=page_size,
                    num_pages=num_pages,
                )
            else:
                page, slot = kv_cache_utils.get_dcp_page_and_slot(
                    num_tokens,
                    seq_lens,
                    jnp.asarray(table),
                    cu_q_lens,
                    jnp.int32(rank),
                    page_size=page_size,
                    num_pages=num_pages,
                    dcp_size=dcp_size,
                    interleave_size=interleave_size,
                )
            np.testing.assert_array_equal(
                np.asarray(kv_cache_utils.get_dst_rows(page, slot, page_size)),
                _numpy_dst_rows(
                    num_tokens,
                    seq_lens,
                    table,
                    cu_q_lens,
                    page_size=page_size,
                    num_pages=num_pages,
                    dcp_size=dcp_size,
                    dcp_rank=rank,
                    interleave_size=interleave_size,
                ),
                err_msg=f"dcp_size={dcp_size}, rank={rank}",
            )

    @parameterized.named_parameters(
        ("unsharded", 1, 1),
        ("dcp2_il1", 2, 1),
        ("dcp4_il4", 4, 4),
        ("dcp16_il4", 16, 4),
    )
    def test_address_layer_matches_mla_kv_cache_utils(
        self, dcp_size, interleave_size
    ):
        """`streamindex_topk.kv_cache_utils` matches `kernels.mla.kv_cache_utils`."""
        try:
            from vllm_torchtpu.kernels.mla import (
                kv_cache_utils as mla_kv_cache_utils,
            )
        except ImportError:
            self.skipTest("requires torch / torch_tpu")

        self.assertEqual(kv_cache_utils.OOB_PAGE, mla_kv_cache_utils.OOB_PAGE)
        self.assertEqual(kv_cache_utils.SKIP_ROW, mla_kv_cache_utils.SKIP_ROW)
        page_size, num_pages = 256, 32
        q_lens = np.array([1, 0, 64, 7, 1, 40], np.int32)
        seq_lens = jnp.array([5, 0, 512, 23, 300, 90], jnp.int32)
        cu_q_lens = _cu_q_lens(q_lens)
        num_tokens = int(q_lens.sum()) + 11
        rng = np.random.default_rng(91 + dcp_size)
        table = jnp.asarray(
            rng.integers(0, num_pages, size=(6, 8), dtype=np.int32)
        )
        if dcp_size == 1:
            page, slot = kv_cache_utils.get_page_and_slot(
                num_tokens,
                seq_lens,
                table,
                cu_q_lens,
                page_size=page_size,
                num_pages=num_pages,
            )
            dst_rows = kv_cache_utils.get_dst_rows(page, slot, page_size)
            mla_dst_rows = mla_kv_cache_utils.get_dst_rows(
                num_tokens, seq_lens, table, cu_q_lens, page_size
            )
            np.testing.assert_array_equal(
                np.asarray(dst_rows), np.asarray(mla_dst_rows)
            )
        else:
            for rank in (0, dcp_size // 2, dcp_size - 1):
                page, slot = kv_cache_utils.get_dcp_page_and_slot(
                    num_tokens,
                    seq_lens,
                    table,
                    cu_q_lens,
                    jnp.int32(rank),
                    page_size=page_size,
                    num_pages=num_pages,
                    dcp_size=dcp_size,
                    interleave_size=interleave_size,
                )
                dst_rows = kv_cache_utils.get_dst_rows(page, slot, page_size)
                mla_page, mla_slot = mla_kv_cache_utils._get_dcp_page_and_slot(
                    num_tokens,
                    seq_lens,
                    table,
                    cu_q_lens,
                    jnp.int32(rank),
                    nope_spec=types.SimpleNamespace(
                        page_size=page_size, num_pages=num_pages
                    ),
                    dcp_size=dcp_size,
                    interleave_size=interleave_size,
                )
                mla_dst_rows = mla_kv_cache_utils._dst_rows(
                    mla_page, mla_slot, page_size
                )
                np.testing.assert_array_equal(
                    np.asarray(dst_rows),
                    np.asarray(mla_dst_rows),
                    err_msg=f"dcp_size={dcp_size}, rank={rank}",
                )

    def test_interpret_fallback_matches_reference(self):
        """`interpret=True` fallback (R7) matches `_reference` in both layouts."""
        cache_has, k, seq_lens, page_indices, cu_q_lens = _uniform_batch(16, 4, 1024, 1)
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            out_unsharded = kv_cache_append.coalesced_kv_append(
                jnp.copy(cache),
                k,
                seq_lens,
                page_indices,
                cu_q_lens,
                kv_layout=layout,
                interpret=True,
            )
            np.testing.assert_array_equal(
                np.asarray(out_unsharded),
                np.asarray(
                    _reference(
                        cache, k, seq_lens, page_indices, cu_q_lens, layout=layout
                    )
                ),
                err_msg=f"unsharded interpret {layout}",
            )
            out_dcp = kv_cache_append.coalesced_kv_append_dcp(
                jnp.copy(cache),
                k,
                seq_lens,
                page_indices,
                cu_q_lens,
                jnp.int32(1),
                dcp_size=4,
                interleave_size=4,
                kv_layout=layout,
                interpret=True,
            )
            np.testing.assert_array_equal(
                np.asarray(out_dcp),
                np.asarray(
                    _reference(
                        cache,
                        k,
                        seq_lens,
                        page_indices,
                        cu_q_lens,
                        layout=layout,
                        dcp_size=4,
                        dcp_rank=1,
                        interleave_size=4,
                    )
                ),
                err_msg=f"dcp interpret {layout}",
            )

    @parameterized.parameters(256, 1024)
    def test_scatter_rows_arbitrary_dst_rows(self, page_size):
        """The writer alone, fed destinations no page table would produce.

        Segments of consecutive rows land on distinct pages at random offsets and
        lengths (1 to a full page), in shuffled order, with skip rows (-1 and
        `>= num_rows`) inside them. Skips inside a segment force compaction; one
        segment crosses from page 22 into page 23 and must split into two runs.
        """
        num_pages = 24
        rng = np.random.default_rng(page_size)
        num_rows = num_pages * page_size
        dst = list(22 * page_size + page_size - 5 + np.arange(10))
        for page in rng.permutation(num_pages - 2)[:18]:
            length = int(rng.choice([1, 2, 3, 5, 127, 128, 129, page_size]))
            length = min(length, page_size)
            start = int(rng.integers(0, page_size - length + 1))
            for row in page * page_size + start + np.arange(length):
                if rng.random() < 0.2:
                    dst.append(int(rng.choice([-1, num_rows, num_rows + 77])))
                dst.append(int(row))
        dst = jnp.asarray(dst, jnp.int32)
        src = _random_rows(rng, dst.shape[0])
        cache_has = _random_cache(rng, num_pages, page_size)
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            actual = jax.jit(
                functools.partial(kv_cache_append.scatter_rows, kv_layout=layout),
                donate_argnums=0,
            )(jnp.copy(cache), src, dst)
            np.testing.assert_array_equal(
                np.asarray(actual),
                np.asarray(
                    kv_cache_append.scatter_rows_reference(cache, src, dst, layout)
                ),
                err_msg=layout,
            )

    def test_scatter_rows_all_skipped_or_empty_is_identity(self):
        cache_has = _random_cache(np.random.default_rng(5), 4, 256)
        for layout in _LAYOUTS:
            cache = _in_layout(cache_has, layout)
            for num_tokens in (9, 0):
                out = kv_cache_append.scatter_rows(
                    jnp.copy(cache),
                    jnp.ones((num_tokens, 129), jnp.uint8),
                    jnp.full((num_tokens,), kv_cache_utils.SKIP_ROW, jnp.int32),
                    layout,
                )
                np.testing.assert_array_equal(
                    np.asarray(out), np.asarray(cache), err_msg=f"{layout} {num_tokens}"
                )

    # ---------------------------------------------------------------------------
    # Reader top-k parity across xla / has / sal (R9).
    # ---------------------------------------------------------------------------

    @parameterized.named_parameters(
        ("decode_b8", [1] * 8, [512, 640, 768, 896, 1024, 1152, 1280, 1408], 8, 8),
        ("mixed_4x1_plus_60", [1, 1, 1, 1, 60], [512, 640, 768, 896, 1024], 4, 5),
    )
    def test_reader_topk_matches_across_xla_has_and_sal(
        self, active_q_lens, active_seq_lens, num_decodes, total_seqs
    ):
        """`streamindex_topk` returns identical top-k indices for xla, has, and sal."""
        if jax.default_backend() != "tpu":
            self.skipTest("streamindex_topk Pallas kernel requires TPU backend")
        page_size, max_seqs, pages_per_seq, topk = 256, 8, 8, 128
        num_pages = max_seqs * pages_per_seq
        num_tokens = int(sum(active_q_lens))
        rng = np.random.default_rng(3141 + num_tokens)

        q_lens = np.zeros(max_seqs, np.int32)
        q_lens[: len(active_q_lens)] = active_q_lens
        seq_lens_np = np.zeros(max_seqs, np.int32)
        seq_lens_np[: len(active_seq_lens)] = active_seq_lens
        seq_lens = jnp.asarray(seq_lens_np, jnp.int32)
        cu_q_lens = _cu_q_lens(q_lens)
        distribution = jnp.asarray(
            [num_decodes, num_decodes, total_seqs], dtype=jnp.int32
        )
        page_indices = jnp.arange(num_pages, dtype=jnp.int32)

        cache_has = _random_cache(rng, num_pages, page_size)
        cache_sal = _in_layout(cache_has, _SAL)
        k_rows = _random_rows(rng, num_tokens)
        q = jnp.asarray(rng.standard_normal((num_tokens, 64, 128)), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((num_tokens, 64)), jnp.bfloat16)

        def score(cache, layout):
            return st_lib.streamindex_topk(
                q,
                w,
                cache,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k=topk,
                compression_ratio=1,
                num_kv_pages_per_block=(2, 2, 2),
                num_queries_per_block=(1, 64, 64),
                kv_layout=layout,
            )

        c_xla = _has_append_4d(
            jnp.copy(cache_has),
            k_rows,
            seq_lens,
            page_indices.reshape(max_seqs, pages_per_seq),
            cu_q_lens,
        )
        c_has = _append(
            cache_has,
            k_rows,
            seq_lens,
            page_indices.reshape(max_seqs, pages_per_seq),
            cu_q_lens,
            layout=_HAS,
        )
        c_sal = _append(
            cache_sal,
            k_rows,
            seq_lens,
            page_indices.reshape(max_seqs, pages_per_seq),
            cu_q_lens,
            layout=_SAL,
        )

        topk_xla = np.asarray(score(c_xla, _HAS))
        topk_has = np.asarray(score(c_has, _HAS))
        topk_sal = np.asarray(score(c_sal, _SAL))

        np.testing.assert_array_equal(topk_has, topk_xla)
        for row in range(num_tokens):
            np.testing.assert_array_equal(
                np.sort(topk_sal[row]),
                np.sort(topk_xla[row]),
                err_msg=f"token row {row}",
            )

    @parameterized.parameters((2, 4), (4, 4))
    def test_dcp_reader_topk_matches_across_xla_has_and_sal(
        self, dcp_size, interleave_size
    ):
        """`streamindex_topk_dcp` matches across xla, has, and sal scatters."""
        if jax.default_backend() != "tpu" or len(jax.devices()) < dcp_size:
            self.skipTest(f"Requires TPU with at least {dcp_size} devices")
        page_size, max_seqs, pages_per_seq, topk = 256, 4, 4, 128
        local_pages = max_seqs * pages_per_seq
        num_tokens = 64
        rng = np.random.default_rng(2718 + dcp_size)

        q_lens = np.array([1, 1, 62, 0], np.int32)
        seq_lens = jnp.array([512, 520, 768, 0], jnp.int32)
        cu_q_lens = _cu_q_lens(q_lens)
        distribution = jnp.array([2, 2, 3], jnp.int32)
        page_indices = jnp.arange(local_pages, dtype=jnp.int32)

        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:dcp_size]), (st_lib.DCP_AXIS_NAME,)
        )
        sharded = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(st_lib.DCP_AXIS_NAME)
        )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        shards_has = [
            _random_cache(np.random.default_rng(100 + r), local_pages, page_size)
            for r in range(dcp_size)
        ]
        shards_sal = [_in_layout(s, _SAL) for s in shards_has]
        global_has = jax.device_put(jnp.concatenate(shards_has, axis=0), sharded)
        global_sal = jax.device_put(jnp.concatenate(shards_sal, axis=0), sharded)

        k_rows = jax.device_put(_random_rows(rng, num_tokens), replicated)
        q = jax.device_put(
            jnp.asarray(rng.standard_normal((num_tokens, 64, 128)), jnp.bfloat16),
            replicated,
        )
        w = jax.device_put(
            jnp.asarray(rng.standard_normal((num_tokens, 64)), jnp.bfloat16),
            replicated,
        )
        seq_lens_r = jax.device_put(seq_lens, replicated)
        page_indices_r = jax.device_put(page_indices, replicated)
        cu_q_lens_r = jax.device_put(cu_q_lens, replicated)
        distribution_r = jax.device_put(distribution, replicated)

        def run_dcp(cache_global, mode: str, layout: str):
            def _scatter(c, k, sl, pi, cu):
                rank = st_lib.cp_rank_as_data(st_lib.DCP_AXIS_NAME, dcp_size)
                if mode == "xla":
                    page, slot = kv_cache_utils.get_dcp_page_and_slot(
                        k.shape[0],
                        sl,
                        pi,
                        cu,
                        rank,
                        page_size=page_size,
                        num_pages=c.shape[0],
                        dcp_size=dcp_size,
                        interleave_size=interleave_size,
                    )
                    k_pad = jnp.pad(k, ((0, 0), (0, c.shape[-1] - k.shape[-1])))
                    return c.at[page, slot // _KV_PACKING, slot % _KV_PACKING].set(
                        k_pad
                    )
                return kv_cache_append.coalesced_kv_append_dcp(
                    c,
                    k,
                    sl,
                    pi,
                    cu,
                    rank,
                    dcp_size=dcp_size,
                    interleave_size=interleave_size,
                    kv_layout=layout,
                )

            p_sharded = jax.sharding.PartitionSpec(st_lib.DCP_AXIS_NAME)
            p_rep = jax.sharding.PartitionSpec()
            updated = jax.shard_map(
                _scatter,
                mesh=mesh,
                in_specs=(p_sharded, p_rep, p_rep, p_rep, p_rep),
                out_specs=p_sharded,
                check_vma=False,
            )(cache_global, k_rows, seq_lens_r, page_indices_r, cu_q_lens_r)
            out = st_lib.streamindex_topk_dcp(
                q,
                w,
                updated,
                seq_lens_r,
                page_indices_r,
                cu_q_lens_r,
                distribution_r,
                mesh=mesh,
                k=topk,
                compression_ratio=1,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                num_kv_pages_per_block=(2, 2, 2),
                num_queries_per_block=(1, 64, 64),
                kv_layout=layout,
            )
            return np.asarray(updated), np.asarray(out)

        c_xla, topk_xla = run_dcp(global_has, "xla", _HAS)
        c_has, topk_has = run_dcp(global_has, "has", _HAS)
        _, topk_sal = run_dcp(global_sal, "sal", _SAL)

        np.testing.assert_array_equal(c_has, c_xla)
        np.testing.assert_array_equal(topk_has, topk_xla)
        for row in range(topk_xla.shape[0]):
            np.testing.assert_array_equal(
                np.sort(topk_sal[row]),
                np.sort(topk_xla[row]),
                err_msg=f"sharded row {row}",
            )

    # ---------------------------------------------------------------------------
    # Argument validation.
    # ---------------------------------------------------------------------------

    def test_invalid_arguments_raise(self):
        cache_has = jnp.zeros((4, 256, 4, 256), jnp.uint8)
        cache_sal = jnp.zeros((4, 33, 4, 1024), jnp.uint8)
        metadata = (
            jnp.array([1], jnp.int32),
            jnp.array([[0]], jnp.int32),
            jnp.array([0, 1], jnp.int32),
        )
        k = jnp.zeros((4, 129), jnp.uint8)

        with self.assertRaisesRegex(ValueError, "SEQ_ALONG_LANE"):
            kv_cache_append.coalesced_kv_append(cache_sal, k, *metadata, kv_layout=_HAS)
        with self.assertRaisesRegex(ValueError, "HEAD_ALONG_SUBLANE"):
            kv_cache_append.coalesced_kv_append(cache_has, k, *metadata, kv_layout=_SAL)
        with self.assertRaisesRegex(ValueError, "dst_rows shape"):
            kv_cache_append.scatter_rows(cache_has, k, jnp.zeros((3,), jnp.int32))
        with self.assertRaisesRegex(ValueError, "at most"):
            kv_cache_append.coalesced_kv_append(
                cache_has, jnp.zeros((65536, 129), jnp.uint8), *metadata
            )
        with self.assertRaisesRegex(ValueError, "VMEM"):
            kv_cache_append.coalesced_kv_append(
                jnp.zeros((4, 256, 4, 1024), jnp.uint8),
                jnp.zeros((32768, 1024), jnp.uint8),
                *metadata,
            )
        with self.assertRaisesRegex(ValueError, "must divide page_size"):
            kv_cache_append.coalesced_kv_append_dcp(
                cache_has, k, *metadata, 0, dcp_size=2, interleave_size=3
            )


if __name__ == "__main__":
    absltest.main()
