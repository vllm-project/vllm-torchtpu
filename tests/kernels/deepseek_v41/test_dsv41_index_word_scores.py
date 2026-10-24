# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for scoring decode index keys directly from packed word rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import index_word_scores


def _words(rng: np.random.Generator, shape: tuple[int, ...]) -> jax.Array:
    bits = rng.integers(-(2**31), 2**31, size=shape, dtype=np.int32)
    # Mask exponent bits so both 16-bit halves are finite normal BF16 values.
    return jnp.asarray((bits & np.int32(-0x40004001)) | np.int32(0x00800080))


def _split_words(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    def bf16(bits: jax.Array) -> jax.Array:
        return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)

    return bf16(words << 16), bf16(words & jnp.int32(-65536))


def _reference(q: jax.Array, weights: jax.Array, words: jax.Array) -> np.ndarray:
    """Reference XLA path: unpack word halves into `[tokens, 2 * rows, 128]`."""
    groups, rows, _ = words.shape
    tokens = q.shape[0]
    keys = jnp.concatenate(
        _split_words(words.reshape(-1, index_word_scores.ROW_WORDS // 2)), axis=1
    )
    keys = jnp.repeat(
        keys.reshape(groups, index_word_scores.SLOTS * rows, -1),
        tokens // groups,
        axis=0,
    )
    logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
    scores = (jnp.maximum(logits, 0) * weights).sum(1)
    return np.asarray(scores)


@pytest.mark.parametrize(
    "tokens,groups,rows",
    [
        (2, 2, 16),
        (6, 1, 16),
        (4, 2, 16),
        (12, 2, 16),
        (1, 1, 2 * index_word_scores.MAX_BLOCK_ROWS),
    ],
)
def test_score_word_rows_matches_xla(tokens: int, groups: int, rows: int) -> None:
    rng = np.random.default_rng(rows + tokens * 10 + groups)
    words = _words(rng, (groups, rows, index_word_scores.ROW_WORDS))
    q = jnp.asarray(
        rng.standard_normal((tokens, 3, index_word_scores.DIM)), jnp.bfloat16
    )
    weights = jnp.asarray(rng.standard_normal((tokens, 3, 1)), jnp.float32)
    out = index_word_scores.score_word_rows(q, weights, words, interpret=True)
    assert out.shape == (tokens, index_word_scores.SLOTS * rows)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(out), _reference(q, weights, words), rtol=1e-6, atol=1e-5
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens,groups,rows", [(2, 2, 16), (6, 2, 16)])
def test_score_word_rows_tpu(tokens: int, groups: int, rows: int) -> None:
    rng = np.random.default_rng(100 + rows + tokens * 10 + groups)
    words = _words(rng, (groups, rows, index_word_scores.ROW_WORDS))
    q = jnp.asarray(
        rng.standard_normal((tokens, 3, index_word_scores.DIM)), jnp.bfloat16
    )
    weights = jnp.asarray(rng.standard_normal((tokens, 3, 1)), jnp.float32)
    out = index_word_scores.score_word_rows(q, weights, words, interpret=False)
    assert out.shape == (tokens, index_word_scores.SLOTS * rows)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(out), _reference(q, weights, words), rtol=1e-6, atol=1e-5
    )


@pytest.mark.parametrize(
    "q_shape,weights_shape,words_shape",
    [
        ((1, 2, 128), (1, 2, 1), (1, 16, 64)),  # not 128-word rows
        ((1, 2, 64), (1, 2, 1), (1, 16, 128)),  # not 128-element keys
        ((1, 2, 128), (1, 3, 1), (1, 16, 128)),  # mismatched head weights
        ((1, 2, 128), (1, 2, 1), (1, 12, 128)),  # rows not a power of two
        ((1, 2, 128), (1, 2, 1), (1, 4, 128)),  # fewer than 8 rows
        ((3, 2, 128), (3, 2, 1), (2, 16, 128)),  # tokens % groups != 0
        ((2, 2, 128), (2, 2, 1), (4, 16, 128)),  # tokens < groups
        ((2, 2), (2, 2, 1), (1, 16, 128)),  # non-3D query
    ],
)
def test_score_word_rows_rejects_invalid_shapes(
    q_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    words_shape: tuple[int, ...],
) -> None:
    q = jnp.zeros(q_shape, jnp.bfloat16)
    weights = jnp.zeros(weights_shape, jnp.float32)
    words = jnp.zeros(words_shape, jnp.int32)
    with pytest.raises(ValueError):
        index_word_scores.score_word_rows(q, weights, words, interpret=True)


@pytest.mark.parametrize(
    "q_dtype,weights_dtype,words_dtype",
    [
        (jnp.float32, jnp.float32, jnp.int32),
        (jnp.bfloat16, jnp.bfloat16, jnp.int32),
        (jnp.bfloat16, jnp.float32, jnp.uint32),
    ],
)
def test_score_word_rows_rejects_invalid_dtypes(
    q_dtype: jnp.dtype, weights_dtype: jnp.dtype, words_dtype: jnp.dtype
) -> None:
    q = jnp.zeros((2, 2, index_word_scores.DIM), q_dtype)
    weights = jnp.zeros((2, 2, 1), weights_dtype)
    words = jnp.zeros((1, 16, index_word_scores.ROW_WORDS), words_dtype)
    with pytest.raises(ValueError):
        index_word_scores.score_word_rows(q, weights, words, interpret=True)
