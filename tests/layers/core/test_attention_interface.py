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
"""Tests for the parts of `attention_interface.py` that can run without a
real TPU kernel.

Most of `attention_interface.py` is glue around Pallas kernels: it reshapes
inputs, picks kernel parameters, checks arguments, and then hands off to a
kernel. The kernels themselves are tested in their own files under
`tests/kernels/`. This file covers the glue:

* writing new tokens into the KV cache (`update_cache`), which is ordinary
  array reshaping and scattering with no kernel involved;
* the SMEM-overflow guard around paged attention, which decides whether to
  call the kernel once or split the batch into smaller pieces; and
* the up-front argument checks that raise a clear error before any kernel
  is ever called.

Because none of these tests dispatch a kernel, they run the same way on any
TPU generation and on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import attention_interface

# Sparse-MLA head dimensions. These are the only sizes the fp8 gather kernel
# accepts (see the assert in `sparse_mla_attention`), so the layout test below
# has to use them to get past the size check and reach the layout check.
LKV_DIM = 512
ROPE_DIM = 64
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
# Two cache specs in the TensorCore layout and one in the SparseCore layout,
# so a test can hand `sparse_mla_attention` a mismatched pair.
NOPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.NOPE, KVCacheLayout.TENSORCORE, 16, 32, LKV_DIM, KV_PACKING
)
ROPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE, KVCacheLayout.TENSORCORE, 16, 32, ROPE_DIM, KV_PACKING
)
ROPE_SC_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE, KVCacheLayout.SPARSECORE, 16, 32, ROPE_DIM, KV_PACKING
)


class UpdateCacheTest(parameterized.TestCase):
    """`update_cache` writes freshly computed keys or values into the paged
    KV cache.

    The cache is laid out as `(kv_heads, num_blocks, block_size, head_dim)`.
    The function has two very different code paths depending on whether it
    is handling a prefill (many tokens per sequence, written a whole block
    at a time) or a decode step (one token per sequence, written one slot at
    a time). Each test below builds a small cache, calls the function, and
    compares the result to the same update written out by hand.
    """

    def test_decode_writes_one_token_per_sequence(self):
        """Decode: each sequence contributes one new token, and it lands at
        the flat slot index given for that sequence. Every other slot in the
        cache must be left untouched."""
        K, L, S, H = 2, 3, 4, 5
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(2 * K * H, dtype=jnp.float32).reshape(2, K, 1, H)
        indices = jnp.array([1, 5], dtype=jnp.int32)

        updated = attention_interface.update_cache(False, cache, indices, operand)

        flat = updated.reshape(K, L * S, H)
        for b in range(2):
            for k in range(K):
                np.testing.assert_array_equal(flat[k, indices[b]], operand[b, k, 0])
        # Everything else stays zero.
        mask = jnp.ones((L * S,), dtype=bool).at[indices].set(False)
        for k in range(K):
            np.testing.assert_array_equal(flat[k, mask], 0)

    def test_prefill_writes_whole_blocks(self):
        """Prefill: the new tokens for a sequence are split into block-sized
        chunks and each chunk is written to the block it was assigned. With
        T=4 tokens and a block size of 2 that is two blocks."""
        B, K, T, H = 1, 2, 4, 3
        S = 2
        L = 4
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(B * K * T * H, dtype=jnp.float32).reshape(B, K, T, H)
        indices = jnp.array([0, 1], dtype=jnp.int32)

        updated = attention_interface.update_cache(True, cache, indices, operand)

        expected = cache.at[:, indices, :, :].set(
            jnp.swapaxes(operand, 0, 1).reshape(K, B * T // S, S, H)
        )
        np.testing.assert_array_equal(updated, expected)

    def test_prefill_sliding_window_keeps_only_the_tail(self):
        """Prefill with a sliding window: when a prompt is longer than the
        window, only the most recent `sliding_window` tokens are kept. The
        earlier tokens must not be written to the cache at all."""
        B, K, T, H = 1, 2, 6, 3
        sliding_window = 4
        S = 2
        L = 4
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(B * K * T * H, dtype=jnp.float32).reshape(B, K, T, H)
        indices = jnp.array([0, 1], dtype=jnp.int32)

        updated = attention_interface.update_cache(
            True,
            cache,
            indices,
            operand,
            prefill_seq_len=jnp.array(T),
            sliding_window=sliding_window,
        )

        # start = max(0, T - sliding_window) = 2: only the last 4 tokens land
        # in the cache, not the first two.
        tail = operand[:, :, T - sliding_window :, :]
        expected = cache.at[:, indices, :, :].set(
            jnp.swapaxes(tail, 0, 1).reshape(K, sliding_window // S, S, H)
        )
        np.testing.assert_array_equal(updated, expected)


class PagedAttentionGuardedSmemTest(parameterized.TestCase):
    """`paged_attention_with_guarded_smem` protects the paged attention
    kernel from running out of SMEM.

    The kernel keeps every sequence's page table in SMEM, which is small
    and fixed in size. When the total number of page indices is under
    `MAX_ALLOWED_PAGE_INDICES_N`, the function just calls the kernel once.
    When it is over, the function splits the batch into equal-sized smaller
    batches, runs the kernel on each one, and stitches the outputs back
    together.

    The real kernel is replaced with a stub that records how it was called
    and returns `q + 1`. The splitting logic does not depend on what the
    kernel computes, so the stub is enough to check it.
    """

    def test_small_batch_calls_kernel_directly(self):
        """A batch under the limit is passed straight to the kernel in a
        single call, with the page table unchanged."""
        calls = []

        def fake_kernel(q, k_pages, v_pages, lengths, page_indices):
            calls.append(page_indices.shape)
            return q + 1.0

        q = jnp.zeros((4, 2, 8))
        lengths = jnp.zeros((4,), jnp.int32)
        page_indices = jnp.zeros((4, 2), jnp.int32)

        out = attention_interface.paged_attention_with_guarded_smem(
            fake_kernel, q, q, q, lengths, page_indices
        )

        self.assertEqual(calls, [(4, 2)])
        np.testing.assert_array_equal(out, jnp.ones_like(q))

    def test_oversized_batch_splits_into_minibatches(self):
        """A batch over the limit is split up: the kernel is called more
        than once, each time on a slice of the page table, and the combined
        output has the same shape as the input.

        This runs with `jax.disable_jit()`. The split is plain Python (a
        `for` loop and some reshapes), which is all this test is checking.
        Compiling the split for a real TPU at the size needed to trigger it
        is a separate concern for the kernel owners, and on some hosts it
        currently crashes the XLA compiler.
        """
        calls = []

        def fake_kernel(q, k_pages, v_pages, lengths, page_indices):
            calls.append(page_indices.shape)
            return q + 1.0

        max_n = attention_interface.MAX_ALLOWED_PAGE_INDICES_N
        batch_size = 4
        blocks_per_seq = max_n // batch_size + 1  # size > max_n
        self.assertGreater(batch_size * blocks_per_seq, max_n)

        q = jnp.zeros((batch_size, 1, 1))
        lengths = jnp.zeros((batch_size,), jnp.int32)
        page_indices = jnp.zeros((batch_size, blocks_per_seq), jnp.int32)

        with jax.disable_jit():
            out = attention_interface.paged_attention_with_guarded_smem(
                fake_kernel, q, q, q, lengths, page_indices
            )

        self.assertGreater(len(calls), 1)
        self.assertEqual(out.shape, q.shape)
        np.testing.assert_array_equal(out, jnp.ones_like(q))


class ShardedRaggedPagedAttentionValidationTest(parameterized.TestCase):
    """Argument checks in `sharded_ragged_paged_attention` that fail fast,
    before the function builds the sharded kernel call."""

    def test_attention_sink_requires_hd64(self):
        """Attention sinks are only implemented in the head_dim=64 kernel
        variant. Passing one with any other head size must raise a clear
        `NotImplementedError` rather than a confusing kernel failure. The
        check happens before `mesh` or `kv_cache` are read, so both can be
        placeholders here."""
        head_dim = 32  # != 64, so use_hd64 is False
        q = jnp.zeros((4, 2, head_dim))
        kv_lens = jnp.zeros((1,), jnp.int32)
        page_indices = jnp.zeros((1,), jnp.int32)
        cu_q_lens = jnp.zeros((1,), jnp.int32)
        distribution = jnp.zeros((3,), jnp.int32)
        attention_sink = jnp.zeros((2,))

        with self.assertRaisesRegex(NotImplementedError, "head_dim==64"):
            attention_interface.sharded_ragged_paged_attention(
                None,  # mesh: unused before the raise
                q,
                q,
                q,
                jnp.zeros((1,)),  # kv_cache: unused before the raise
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sink,
                sm_scale=1.0,
            )


class SparseMlaAttentionLayoutValidationTest(parameterized.TestCase):
    """Argument checks in `sparse_mla_attention` that fail fast, before the
    function builds the sharded kernel call."""

    def test_mismatched_layouts_raise(self):
        """The NoPE and RoPE caches must use the same memory layout (both
        TensorCore or both SparseCore). Handing the function one of each
        must raise a `ValueError` naming both layouts. All the array
        arguments here are zeros of the right shape; only the two layout
        specs matter for this check."""
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))

        with self.assertRaisesRegex(ValueError, "matching NoPE and RoPE layouts"):
            attention_interface.sparse_mla_attention(
                jnp.zeros((1, 2, LKV_DIM), jnp.bfloat16),
                jnp.zeros((1, 2, ROPE_DIM), jnp.bfloat16),
                jnp.zeros((1, LKV_DIM), jnp.float8_e4m3fn),
                jnp.zeros((1, ROPE_DIM), jnp.float8_e4m3fn),
                jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
                jnp.zeros(ROPE_SC_SPEC.shape, ROPE_SC_SPEC.jax_dtype),
                jnp.zeros((1, 1), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((4,), jnp.int32),
                jnp.zeros((2,), jnp.int32),
                jnp.zeros((3,), jnp.int32),
                mesh,
                NOPE_SPEC,
                ROPE_SC_SPEC,
            )
