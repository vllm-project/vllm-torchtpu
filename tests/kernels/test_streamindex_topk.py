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
"""Tests for the DeepSeek-V4 StreamIndex Top-K kernel."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

# isort: off
from vllm_torchtpu.kernels.deepseek_v4.compress_norm_rope import (
    quantize_fp8_ue8m0,
    shared_indexer_cache_shape,
    unpack_indexer_kv_cache,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
# isort: on

# The kernel zero-pads the query head dim up to a full 128-lane group and reads
# the UE8M0 block scale from the byte right after it, so a packed indexer record
# is always `[fp8 x PADDED_HEAD_DIM | scale | pad]`, whatever the real head dim.
PADDED_HEAD_DIM = 128
# One scale per key vector.
QUANT_BLOCK = PADDED_HEAD_DIM
# uint8 lanes per 32-bit word in the MLA paged-cache layout.
PACKING = 4

# fp8 keys plus a TPU matmul put the kernel's scores within a few bf16 ULPs of a
# NumPy fp32 reference, so scores get the same tolerance the compressor test
# uses for its dequantized outputs.
_RTOL = 2e-2
_ATOL = 2e-2


def _pack_indexer_cache(keys):
    """fp8-quantize ``keys`` and pack them into the shared uint8 KV cache.

    Args:
      keys: ``[num_pages, page_size, head_dim]`` fp32 compressed KV keys.

    Returns:
      ``(cache, dequantized)``. ``cache`` is the paged ``uint8`` buffer the
      kernel reads; ``dequantized`` is what it sees after the fp8 round trip.
      The reference has to score against the latter -- scoring the fp32 input
      would fold quantization error into the expected ranking.
    """
    num_pages, page_size, head_dim = keys.shape
    if page_size % PACKING:
        raise ValueError(
            f"page_size {page_size} must be a multiple of {PACKING}")

    padded = np.zeros((num_pages * page_size, PADDED_HEAD_DIM), np.float32)
    padded[:, :head_dim] = keys.reshape(-1, head_dim)
    fp8, scale = quantize_fp8_ue8m0(jnp.asarray(padded), QUANT_BLOCK)

    shape = shared_indexer_cache_shape(num_pages, page_size, PADDED_HEAD_DIM,
                                       QUANT_BLOCK)
    record = np.zeros((num_pages * page_size, shape[-1]), np.uint8)
    record[:, :PADDED_HEAD_DIM] = np.asarray(
        jax.lax.bitcast_convert_type(fp8, jnp.uint8))
    record[:, PADDED_HEAD_DIM:PADDED_HEAD_DIM + 1] = np.asarray(
        jax.lax.bitcast_convert_type(scale, jnp.uint8))
    cache = record.reshape(shape)

    # Read the record back through the shipped unpacker instead of reusing
    # `fp8`/`scale` directly, so the reference is grounded in the same layout
    # contract the kernel implements.
    got_fp8, got_scale = unpack_indexer_kv_cache(jnp.asarray(cache),
                                                 PADDED_HEAD_DIM, QUANT_BLOCK)
    dequantized = (np.asarray(got_fp8).astype(np.float32) *
                   np.asarray(got_scale).astype(np.float32))
    dequantized = dequantized.reshape(num_pages, page_size,
                                      PADDED_HEAD_DIM)[..., :head_dim]
    return cache, dequantized


def _reference_scores(q, weights, keys, block_table, q_lens, seq_lens,
                      cu_q_lens, comp_ratio):
    """Exact indexer score of every compressed KV position, ``-inf`` if masked.

    ``score[t, s] = sum_h relu(q[t, h] . key[s]) * weights[t, h]`` for every
    compressed position ``s`` that both exists (``s < seq_len // comp_ratio``)
    and is causally visible to token ``t``.
    """
    num_tokens, _, head_dim = q.shape
    page_size = keys.shape[1]
    max_kv = block_table.shape[1] * page_size
    scores = np.full((num_tokens, max_kv), -np.inf, np.float32)

    for seq, (q_len, seq_len) in enumerate(zip(q_lens, seq_lens)):
        if q_len == 0:
            continue
        kv_len = seq_len // comp_ratio
        seq_keys = keys[block_table[seq]].reshape(-1, head_dim)
        for i in range(q_len):
            token = int(cu_q_lens[seq]) + i
            # Queries are the tail of the sequence, so this token sits at
            # absolute (uncompressed) position `seq_len - q_len + i` and sees
            # compressed positions up to `q_pos // comp_ratio`.
            q_pos = seq_len - q_len + i
            limit = min(kv_len, q_pos // comp_ratio + 1)
            if limit <= 0:
                continue
            inner = seq_keys[:limit] @ q[token].T  # [limit, num_heads]
            scores[token, :limit] = (np.maximum(inner, 0.0) *
                                     weights[token]).sum(-1)
    return scores


def _make_inputs(q_lens,
                 seq_lens,
                 comp_ratio,
                 page_size,
                 pages_per_seq,
                 num_heads,
                 head_dim,
                 signed_weights=False,
                 seed=0):
    """Build one full kernel invocation plus its exact reference scores."""
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_tokens = int(sum(q_lens))
    assert len(seq_lens) == num_seqs
    for q_len, seq_len in zip(q_lens, seq_lens):
        assert q_len <= seq_len, "a query cannot precede its own sequence"
        assert seq_len // comp_ratio <= pages_per_seq * page_size, (
            "page table too small for the compressed sequence")

    q = rng.standard_normal((num_tokens, num_heads, head_dim),
                            dtype=np.float32)
    lo, hi = (-1.5, 1.5) if signed_weights else (0.25, 1.75)
    weights = rng.uniform(lo, hi, (num_tokens, num_heads)).astype(np.float32)

    # Give every sequence a shuffled, disjoint set of physical pages so a bug in
    # the page walk cannot be hidden by sequential layout.
    num_pages = num_seqs * pages_per_seq
    block_table = rng.permutation(num_pages).astype(np.int32).reshape(
        num_seqs, pages_per_seq)
    keys = rng.standard_normal((num_pages, page_size, head_dim),
                               dtype=np.float32)
    cache, deq_keys = _pack_indexer_cache(keys)

    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    # `distribution` requires the decode-only sequences to lead the batch.
    num_decodes = 0
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
        num_decodes += 1

    scores = _reference_scores(q, weights, deq_keys, block_table, q_lens,
                               seq_lens, cu_q_lens, comp_ratio)
    inputs = {
        "q": q,
        "indexer_weights": weights,
        "cache_kv": cache,
        "seq_lens": np.asarray(seq_lens, np.int32),
        "page_indices": block_table.reshape(-1),
        "cu_q_lens": cu_q_lens,
        "distribution": np.array([num_decodes, num_decodes, num_seqs],
                                 np.int32),
    }
    return inputs, scores


def _run(inputs, k, comp_ratio, bkv_p, bq_sz):
    """Invoke the kernel; returns the jax array so callers can inspect it."""
    return streamindex_topk(
        **{
            name: jnp.asarray(value)
            for name, value in inputs.items()
        },
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )


def _assert_topk(actual, scores, k):
    """Check the kernel's indices against the exact reference scores.

    The kernel promises the ``k`` highest-scoring compressed KV positions per
    token, padded with ``-1`` once a token runs out of visible positions. It
    does *not* promise an order -- selection packs winners in increasing KV
    position, and nothing downstream depends on best-first -- so the check is on
    the selected set. Equal scores may be broken either way, so the comparison
    is on the selected *scores* rather than on the indices, except when a token
    has at most ``k`` visible positions, where the index set is fully determined
    and is compared exactly.
    """
    num_tokens, max_kv = scores.shape
    assert actual.shape == (num_tokens, k), actual.shape
    assert actual.dtype == np.int32, actual.dtype

    for token in range(num_tokens):
        row = actual[token]
        visible = np.isfinite(scores[token])
        n_visible = int(visible.sum())
        n_kept = int(np.count_nonzero(row >= 0))

        assert np.all(row[:n_kept] >= 0) and np.all(row[n_kept:] < 0), (
            f"token {token}: -1 padding is not a suffix: {row}")
        assert n_kept == min(
            k,
            n_visible), (f"token {token}: kept {n_kept} positions, expected "
                         f"{min(k, n_visible)}")
        if n_kept == 0:
            continue

        kept = row[:n_kept]
        assert kept.max() < max_kv, (
            f"token {token}: index {kept.max()} is outside the {max_kv} KV "
            "positions the page table can address")
        assert len(np.unique(kept)) == n_kept, (
            f"token {token}: duplicate indices in {kept}")

        if n_visible <= k:
            # Everything visible has to come back, so the set is exact and the
            # comparison does not depend on score arithmetic at all.
            np.testing.assert_array_equal(
                np.sort(kept), np.flatnonzero(visible),
                f"token {token}: not every visible position was reported")

        # Sorted on both sides: this pins the set the kernel picked, not the
        # order it emitted them in, which is deliberately unspecified.
        expected = np.sort(scores[token])[::-1][:n_kept]
        np.testing.assert_allclose(
            np.sort(scores[token][kept])[::-1],
            expected,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"token {token}: selected scores are not the top {n_kept}")


class StreamIndexTopKTest(parameterized.TestCase):

    @parameterized.named_parameters(
        # Single decode step; k covers every visible position.
        dict(testcase_name="_decode",
             q_lens=(1, ),
             seq_lens=(512, ),
             comp_ratio=4,
             page_size=128,
             pages_per_seq=4,
             num_heads=4,
             head_dim=128,
             k=128,
             bkv_p=1,
             bq_sz=1),
        # Six decode sequences of differing length: four go through the batched
        # decode kernel (seq_batch_size=4), the last two one at a time. The
        # batch runs as many KV blocks as its longest member, so the shorter
        # ones also exercise the length mask.
        dict(testcase_name="_decode_batched",
             q_lens=(1, 1, 1, 1, 1, 1),
             seq_lens=(512, 768, 1024, 640, 896, 512),
             comp_ratio=4,
             page_size=128,
             pages_per_seq=4,
             num_heads=4,
             head_dim=128,
             k=128,
             bkv_p=1,
             bq_sz=1),
        # Prefill with two query blocks, and k well under the number of visible
        # positions so the running Top-K actually has to discard candidates.
        dict(testcase_name="_prefill",
             q_lens=(8, ),
             seq_lens=(1024, ),
             comp_ratio=4,
             page_size=64,
             pages_per_seq=8,
             num_heads=8,
             head_dim=128,
             k=64,
             bkv_p=2,
             bq_sz=4),
        # Leading decode sequence followed by two prefills (the mixed kernel).
        dict(testcase_name="_mixed",
             q_lens=(1, 5, 3),
             seq_lens=(384, 512, 256),
             comp_ratio=2,
             page_size=64,
             pages_per_seq=8,
             num_heads=4,
             head_dim=128,
             k=128,
             bkv_p=2,
             bq_sz=4,
             signed_weights=True),
        # A zero-token sequence in the middle, and k larger than the page table
        # can ever address so the result is padded back out to k.
        dict(testcase_name="_empty_seq_and_k_padding",
             q_lens=(2, 0, 3),
             seq_lens=(256, 0, 128),
             comp_ratio=1,
             page_size=64,
             pages_per_seq=4,
             num_heads=4,
             head_dim=128,
             k=512,
             bkv_p=2,
             bq_sz=2),
        # head_dim below one lane group: the kernel zero-pads the query to 128
        # and the packed record keeps the scale byte at the padded offset.
        dict(testcase_name="_narrow_head_dim",
             q_lens=(4, ),
             seq_lens=(512, ),
             comp_ratio=2,
             page_size=32,
             pages_per_seq=8,
             num_heads=2,
             head_dim=64,
             k=64,
             bkv_p=4,
             bq_sz=2),
        # Single head, no compression, one query per block.
        dict(testcase_name="_single_head",
             q_lens=(3, ),
             seq_lens=(256, ),
             comp_ratio=1,
             page_size=32,
             pages_per_seq=8,
             num_heads=1,
             head_dim=128,
             k=32,
             bkv_p=4,
             bq_sz=1),
    )
    def test_matches_reference(self,
                               q_lens,
                               seq_lens,
                               comp_ratio,
                               page_size,
                               pages_per_seq,
                               num_heads,
                               head_dim,
                               k,
                               bkv_p,
                               bq_sz,
                               signed_weights=False,
                               seed=0):
        self.assertEqual((page_size * bkv_p) % 128, 0,
                         "bkv_sz must be a multiple of 128 for TPU DMA")

        inputs, scores = _make_inputs(q_lens=q_lens,
                                      seq_lens=seq_lens,
                                      comp_ratio=comp_ratio,
                                      page_size=page_size,
                                      pages_per_seq=pages_per_seq,
                                      num_heads=num_heads,
                                      head_dim=head_dim,
                                      signed_weights=signed_weights,
                                      seed=seed)
        actual = np.asarray(_run(inputs, k, comp_ratio, bkv_p, bq_sz))
        _assert_topk(actual, scores, k)

    def test_page_table_fragmentation_is_respected(self):
        """Permuting a sequence's pages permutes its Top-K the same way.

        The kernel reports indices in *compressed sequence* space, so remapping
        which physical page backs each logical block must leave the answer
        unchanged. This is what a paged KV cache guarantees and what a mistake
        in the page walk breaks.
        """
        kwargs = dict(q_lens=(4, ),
                      seq_lens=(512, ),
                      comp_ratio=2,
                      page_size=64,
                      pages_per_seq=4,
                      num_heads=4,
                      head_dim=128,
                      seed=3)
        inputs, scores = _make_inputs(**kwargs)
        baseline = np.asarray(
            _run(inputs, k=64, comp_ratio=2, bkv_p=2, bq_sz=4))

        # Move every logical block to a different physical page, carrying its
        # contents along, and re-run.
        page_indices = inputs["page_indices"]
        shuffled = np.roll(page_indices, 1)
        cache = np.asarray(inputs["cache_kv"]).copy()
        cache[shuffled] = np.asarray(inputs["cache_kv"])[page_indices]
        moved = dict(inputs, cache_kv=cache, page_indices=shuffled)

        actual = np.asarray(_run(moved, k=64, comp_ratio=2, bkv_p=2, bq_sz=4))
        np.testing.assert_array_equal(actual, baseline)
        _assert_topk(actual, scores, k=64)

    @parameterized.named_parameters(
        dict(testcase_name="_decode", q_lens=(1, 1, 1), k=2048),
        dict(testcase_name="_prefill", q_lens=(6, ), k=512),
    )
    def test_lowering_shape(self, q_lens, k):
        """Trace-only check: the kernel is never executed, so no TPU needed."""
        inputs, _ = _make_inputs(q_lens=q_lens,
                                 seq_lens=tuple(256 for _ in q_lens),
                                 comp_ratio=2,
                                 page_size=64,
                                 pages_per_seq=4,
                                 num_heads=4,
                                 head_dim=128,
                                 seed=1)
        fn = functools.partial(streamindex_topk,
                               k=k,
                               compression_ratio=2,
                               num_kv_pages_per_block=2,
                               num_queries_per_block=4)
        out = jax.eval_shape(
            fn, **{
                name: jnp.asarray(v)
                for name, v in inputs.items()
            })

        self.assertEqual(out.shape, (sum(q_lens), k))
        self.assertEqual(out.dtype, jnp.int32)

    def test_runs_on_tpu(self):
        """Executes on TPU and confirms the backend really is TPU."""
        inputs, scores = _make_inputs(q_lens=(1, 1, 1, 1, 6),
                                      seq_lens=(512, 640, 768, 512, 1024),
                                      comp_ratio=4,
                                      page_size=128,
                                      pages_per_seq=4,
                                      num_heads=4,
                                      head_dim=128,
                                      seed=7)
        out = _run(inputs, k=128, comp_ratio=4, bkv_p=1, bq_sz=4)
        out.block_until_ready()
        self.assertEqual(out.devices().pop().platform, "tpu")
        _assert_topk(np.asarray(out), scores, k=128)


if __name__ == "__main__":
    absltest.main()
