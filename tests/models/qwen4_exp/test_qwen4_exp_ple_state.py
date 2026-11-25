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
"""Tests for the Qwen4Exp PLE state kept in the unified KV pool."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

# The pure-JAX core behind PLEStateCache; the op wrapper itself is
# covered end to end by test_qwen4_exp_tiny_e2e.py.
from vllm_torchtpu.models.qwen4_exp.ops.ple_state import (
    _pages_per_block,
    _read_states,
    _write_states,
    read_ple_state,
    write_ple_state,
)
from vllm_torchtpu.models.qwen4_exp.ple import ragged_tail

# Two 16-token pages per block of a [pages, 16, 1, 4, 128] fp8 pool: 16 KiB
# per block, enough for a 9 x 512 bf16 window and two token ids.
_POOL_SHAPE = (18, 16, 1, 4, 128)
_PAGES_PER_BLOCK = 3
_WINDOW = (9, 512)


def _random_pool(dtype: jnp.dtype) -> jnp.ndarray:
    """A pool of _POOL_SHAPE in ``dtype`` with random bytes."""
    itemsize = jnp.dtype(dtype).itemsize
    shape = (*_POOL_SHAPE[:-1], _POOL_SHAPE[-1] * itemsize)
    raw = np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8)
    return jnp.asarray(raw).view(dtype)


_POOL_DTYPES = pytest.mark.parametrize(
    "pool_dtype", [jnp.float8_e4m3fn, jnp.bfloat16], ids=["fp8_pool", "bf16_pool"]
)


def _write_three_states(pool):
    """Writes 3 states to blocks [3, 5, 0]; row 2 owns no block (null block 0)."""
    rng = np.random.default_rng(1)
    window = jnp.asarray(rng.standard_normal((3, *_WINDOW)), jnp.bfloat16)
    context = jnp.asarray([[7, 8], [9, 10], [11, 12]], jnp.int32)
    blocks = jnp.asarray([3, 5, 0], jnp.int32)
    new_pool = _write_states(
        pool, blocks, window, context, pages_per_block=_PAGES_PER_BLOCK
    )
    return new_pool, blocks, window, context


def _read(pool, blocks):
    return _read_states(
        pool,
        blocks,
        pages_per_block=_PAGES_PER_BLOCK,
        window_shape=_WINDOW,
        window_dtype=jnp.bfloat16,
        context_len=2,
    )


@_POOL_DTYPES
def test_round_trip_is_bit_exact(pool_dtype) -> None:
    new_pool, blocks, window, context = _write_three_states(_random_pool(pool_dtype))
    got_window, got_context = _read(new_pool, blocks[:2])
    np.testing.assert_array_equal(
        np.asarray(got_window, np.float32), np.asarray(window[:2], np.float32)
    )
    np.testing.assert_array_equal(np.asarray(got_context), np.asarray(context[:2]))


@_POOL_DTYPES
def test_write_touches_only_the_given_blocks(pool_dtype) -> None:
    pool = _random_pool(pool_dtype)
    new_pool, *_ = _write_three_states(pool)
    assert new_pool.shape == pool.shape and new_pool.dtype == pool.dtype
    num_blocks = _POOL_SHAPE[0] // _PAGES_PER_BLOCK
    before = np.asarray(pool.view(jnp.uint8)).reshape(num_blocks, -1)
    after = np.asarray(new_pool.view(jnp.uint8)).reshape(num_blocks, -1)
    for block in (1, 2, 4):
        np.testing.assert_array_equal(after[block], before[block])


def test_mismatched_state_shapes_are_rejected() -> None:
    pool = _random_pool(jnp.bfloat16)
    blocks = jnp.asarray([3, 5], jnp.int32)
    window = jnp.zeros((2, *_WINDOW), jnp.bfloat16)
    with pytest.raises(ValueError, match="ngram_context"):
        _write_states(
            pool,
            blocks,
            window,
            jnp.zeros((2, 2), jnp.float32),
            pages_per_block=_PAGES_PER_BLOCK,
        )
    with pytest.raises(ValueError, match="conv_window"):
        _write_states(
            pool,
            blocks,
            window[:1],
            jnp.zeros((2, 2), jnp.int32),
            pages_per_block=_PAGES_PER_BLOCK,
        )


def test_state_larger_than_a_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="state block has"):
        _read_states(
            _random_pool(jnp.float8_e4m3fn),
            jnp.asarray([1], jnp.int32),
            pages_per_block=1,
            window_shape=_WINDOW,
            window_dtype=jnp.bfloat16,
            context_len=2,
        )


def test_pages_per_block_requires_whole_pages_of_a_5d_pool() -> None:
    assert _pages_per_block(_POOL_SHAPE, 48, is_lbhnc=False) == 3
    assert _pages_per_block((18, 4, 32, 4, 16), 48, is_lbhnc=True) == 3
    with pytest.raises(ValueError, match="multiple"):
        _pages_per_block(_POOL_SHAPE, 24)
    with pytest.raises(NotImplementedError, match="5-D"):
        _pages_per_block((6, 2, 16, 1, 4, 128), 32)


def _tail(rows, history, chunks, ends):
    """Reference: per request, the last len(history[r]) rows of history[r]
    followed by the first ends[r] rows of its chunk."""
    out = []
    for r, (start, _) in enumerate(chunks):
        seq = torch.cat([history[r], rows[start : start + ends[r]]])
        out.append(seq[-history.shape[1] :])
    return torch.stack(out)


@pytest.mark.parametrize("ends_kind", ["query_lens", "checkpoint_offsets"])
def test_ragged_tail_is_last_history_rows_then_chunk_prefix(ends_kind) -> None:
    torch.manual_seed(0)
    # Chunks of 5, 0, 1 and 7 tokens, then padding up to 16 tokens.
    chunks = [(0, 5), (5, 5), (5, 6), (6, 13)]
    query_start_loc = torch.tensor([0, 5, 5, 6, 13])
    rows = torch.randn(16, 3)
    history = torch.randn(4, 4, 3)
    query_lens = torch.tensor([5, 0, 1, 7])
    ends = query_lens if ends_kind == "query_lens" else torch.tensor([2, 0, 0, 6])
    torch.testing.assert_close(
        ragged_tail(rows, history, query_start_loc, ends),
        _tail(rows, history, chunks, ends.tolist()),
        rtol=0,
        atol=0,
    )


def test_ple_state_ops_trace_with_symbolic_fake_tensors() -> None:
    """Static jax_op registrations trace cleanly under symbolic FakeTensorMode."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        num_reqs = shape_env.create_unbacked_symint()
        pool = torch.empty(_POOL_SHAPE, dtype=torch.bfloat16)
        block_ids = torch.empty((num_reqs,), dtype=torch.int32)
        window, context = read_ple_state(
            pool,
            block_ids,
            block_size=48,
            is_lbhnc=False,
            window_shape=_WINDOW,
            window_dtype=torch.bfloat16,
            context_len=2,
        )
        assert window.shape == (num_reqs, *_WINDOW)
        assert window.dtype == torch.bfloat16
        assert context.shape == (num_reqs, 2)
        assert context.dtype == torch.int32
        write_ple_state(
            pool,
            block_ids,
            window,
            context,
            block_size=48,
            is_lbhnc=False,
        )
