# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_SPARSE_MLA_PACKED_ROPE layout pieces that run on CPU: the packed ROPE
cache spec, the writer's token packing against the layout the SparseCore
gather reads, the gather's band transpose and the kv-in-gather write operand.
The gather kernel itself needs a TPU."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import KVCacheLayout, KVCacheType
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout, native_sc_gather


def _rope_spec(packed, layout=KVCacheLayout.SPARSECORE, cache_type=KVCacheType.ROPE):
    return kv_cache_utils.SparseMLAKVCacheSpec.create(
        cache_type, layout, 3, 256, 64, 4, packed_rope=packed
    )


def test_packed_rope_spec_holds_eight_tokens_per_row():
    padded, packed = _rope_spec(False), _rope_spec(True)
    assert padded.token_bytes == 128 and padded.shape == (3, 64, 128)
    assert packed.token_bytes == 64 and packed.tokens_per_row == 8
    assert packed.shape == (3, 32, 128)  # 256 tokens / 8 per 128-word row
    # 640 -> 576 B/token with the 512-byte nope token.
    assert np.prod(packed.shape) * 4 * 2 == np.prod(padded.shape) * 4


@pytest.mark.parametrize(
    "layout, cache_type",
    [
        (KVCacheLayout.TENSORCORE, KVCacheType.ROPE),
        (KVCacheLayout.SPARSECORE, KVCacheType.NOPE),
    ],
)
def test_packed_rope_refuses_other_caches(layout, cache_type):
    with pytest.raises(ValueError, match="packed rope"):
        _rope_spec(True, layout, cache_type)


def test_writer_packing_is_the_layout_the_gather_reads():
    """`scatter`/`tc_scatter` write `pack_tokens(k_pe, 64)` rows of 16 words;
    laid end to end they are `pack_rope_packed`'s eight-token rows."""
    rng = np.random.default_rng(0)
    rope_u8 = rng.integers(0, 256, (16, 64), dtype=np.uint8)
    words = np.asarray(kv_cache_utils.pack_tokens(jnp.asarray(rope_u8), 64))
    assert words.shape == (16, native_sc_gather.PACKED_ROPE_WORDS)
    np.testing.assert_array_equal(
        words.reshape(2, 128), native_sc_cache_layout.pack_rope_packed(rope_u8)
    )


@pytest.mark.parametrize("words", [native_sc_gather.ROPE_WORDS, 16])
def test_transpose_quad_interleaves_four_tokens(words):
    """Band b word w byte t of the output is token t's byte b * words + w,
    for the padded (32-word) and packed (16-word) token alike."""
    rng = np.random.default_rng(words)
    tokens = rng.integers(0, 256, (4, 4 * words), dtype=np.uint8)
    data = [
        jnp.asarray(kv_cache_utils.pack_tokens(jnp.asarray(t[None]), 4 * words))
        for t in tokens
    ]
    out = np.stack([np.asarray(d) for d in native_sc_gather.transpose_quad(data)])
    got = out.reshape(4, words).view(np.uint8).reshape(4, words, 4)
    for b in range(4):
        for w in range(words):
            np.testing.assert_array_equal(got[b, w], tokens[:, b * words + w])


def test_pack_write_rows_takes_sixteen_word_rope_tokens():
    t = 5
    dst = jnp.arange(t, dtype=jnp.int32)
    nope = jnp.ones((t, 128), jnp.uint32)
    rope = jnp.full((t, native_sc_gather.PACKED_ROPE_WORDS), 7, jnp.uint32)
    rows = np.asarray(native_sc_gather.pack_write_rows(dst, nope, rope))
    tp = rows.shape[0] // 3
    rope_part = rows[tp : tp + t]
    assert (rope_part[:, :16] == 7).all() and (rope_part[:, 16:] == 0).all()
    with pytest.raises(ValueError, match="16, 32 or 128"):
        native_sc_gather.pack_write_rows(dst, nope, jnp.zeros((t, 24), jnp.uint32))
