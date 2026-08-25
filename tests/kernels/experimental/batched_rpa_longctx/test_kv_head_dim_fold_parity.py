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
"""On-device parity for the `TPU_RPA_FOLD_KV_HEAD_DIM` KV-cache layout.

The fold moves bytes; it does not change values or quantisation. What it does
change is how many tokens of KV fit in VMEM at once, so left to itself the
wrapper stages a bigger chunk on the folded layout. Those are two separate
contracts and the tests keep them apart:

  * same block sizes  -> the outputs are bit-exact, and so is the cache;
  * self-sized blocks -> the chunking differs, which reassociates the
    online-softmax accumulation, so the outputs agree to bf16 rounding.

The shapes here are the ones the fold exists for: one KV head per rank (TP8 on
a 2-KV-head model) at head_dim 256 with an fp8 cache.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)

_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def derived_blocks(fold, *, num_q_heads, num_kv_heads, head_dim, q_dtype,
                   kv_dtype, page_size, total_q_tokens, num_page_indices,
                   sm_scale):
    """The block sizes the wrapper would choose for itself on this layout."""
    model_cfgs = configs.ModelConfigs(num_q_heads=num_q_heads,
                                      num_kv_heads=num_kv_heads,
                                      head_dim=head_dim,
                                      mask_value=-1e30,
                                      sm_scale=sm_scale)
    serve_cfgs = configs.ServingConfigs(
        num_seqs=1,
        page_size=page_size,
        total_q_tokens=total_q_tokens,
        num_page_indices=num_page_indices,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
        dtype_out=q_dtype,
        fold_kv_head_dim=fold,
    )
    return wrapper.calculate_block_sizes(model_cfgs, serve_cfgs,
                                         _VMEM_LIMIT_BYTES)


def _build_cache(fold,
                 target_k,
                 target_v,
                 kv_len,
                 total_pages,
                 page_size,
                 num_q_heads,
                 num_kv_heads,
                 head_dim,
                 q_dtype,
                 kv_dtype,
                 page_indices,
                 sm_scale,
                 blocks=None):
    """Prime the cache by letting the kernel write the whole sequence in."""
    cache_shape = wrapper.get_kv_cache_shape(total_pages,
                                             page_size,
                                             num_kv_heads,
                                             head_dim,
                                             kv_dtype,
                                             fold_kv_head_dim=fold)
    _, cache = wrapper.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), q_dtype),
        jnp.asarray(target_k, kv_dtype),
        jnp.asarray(target_v, kv_dtype),
        jnp.zeros(cache_shape, kv_dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=blocks[0] if blocks else None,
        prefill_block_sizes=blocks[1] if blocks else None,
        fold_kv_head_dim=fold,
    )
    return cache


# num_kv_heads 1 is the folding case; 2 is the DP rank, where the fold is a
# no-op and the two runs must produce the identical cache tensor as well.
@pytest.mark.parametrize("num_kv_heads, expect_smaller_page", [(1, True),
                                                               (2, False)])
@pytest.mark.parametrize("kv_len, q_len, distribution", [
    (32, 16, (0, 0, 1)),
    (33, 1, (1, 1, 1)),
    (4096, 1, (1, 1, 1)),
])
def test_folded_cache_matches_unfolded(kv_len, q_len, distribution,
                                       num_kv_heads, expect_smaller_page):
    rng = np.random.default_rng(0)
    head_dim, num_q_heads, page_size = 256, 32, 256
    dtype, kv_dtype = jnp.bfloat16, jnp.float8_e4m3fn
    total_pages = max(8, -(-kv_len // page_size) + 1)
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k = r(kv_len, num_kv_heads, head_dim)
    target_v = r(kv_len, num_kv_heads, head_dim)
    query = r(q_len, num_q_heads, head_dim)
    new_k = target_k[kv_len - q_len:kv_len]
    new_v = target_v[kv_len - q_len:kv_len]

    # Both runs get the *unfolded* choice, so the KV chunking is identical and
    # only the byte placement differs. It is always valid for the folded
    # layout too: folding shrinks the KV buffer, it never grows it.
    blocks = derived_blocks(False,
                            num_q_heads=num_q_heads,
                            num_kv_heads=num_kv_heads,
                            head_dim=head_dim,
                            q_dtype=dtype,
                            kv_dtype=kv_dtype,
                            page_size=page_size,
                            total_q_tokens=q_len,
                            num_page_indices=total_pages,
                            sm_scale=sm_scale)

    def run(fold):
        cache = _build_cache(fold, target_k, target_v, kv_len, total_pages,
                             page_size, num_q_heads, num_kv_heads, head_dim,
                             dtype, kv_dtype, page_indices, sm_scale, blocks)
        out, new_cache = wrapper.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(new_k, kv_dtype),
            jnp.asarray(new_v, kv_dtype),
            cache,
            jnp.array([kv_len], jnp.int32),
            page_indices,
            jnp.array([0, q_len], jnp.int32),
            jnp.asarray(distribution, jnp.int32),
            sm_scale=sm_scale,
            decode_block_sizes=blocks[0],
            prefill_block_sizes=blocks[1],
            fold_kv_head_dim=fold,
        )
        return np.asarray(out.astype(jnp.float32)), new_cache

    out_unfolded, cache_unfolded = run(False)
    out_folded, cache_folded = run(True)

    if expect_smaller_page:
        assert cache_folded.size * 2 == cache_unfolded.size
    else:
        assert cache_folded.shape == cache_unfolded.shape

    # Same chunking: identical values in a different place, read back in the
    # same order and dequantised by the same code, so bit-exact.
    np.testing.assert_array_equal(out_folded, out_unfolded)
    assert np.isfinite(out_folded).all()


def test_self_sized_blocks_stage_a_bigger_chunk_and_agree_to_bf16():
    """Left to size its own blocks, the folded run stages more KV per step.

    That is part of the win -- half the bytes per token means nearly twice the
    tokens fit the same VMEM budget (decode `bkv_sz` 1792 -> 3328 at this
    shape). It also reassociates the online-softmax accumulation once the
    sequence spans more than one chunk, so the outputs agree to bf16 rounding
    rather than exactly. The stored bytes are unaffected;
    `test_folded_writeback_lands_the_same_kv` pins those.
    """
    rng = np.random.default_rng(0)
    kv_len, q_len = 4096, 1
    head_dim, num_kv_heads, num_q_heads, page_size = 256, 1, 32, 256
    dtype, kv_dtype = jnp.bfloat16, jnp.float8_e4m3fn
    total_pages = max(8, -(-kv_len // page_size) + 1)
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k = r(kv_len, num_kv_heads, head_dim)
    target_v = r(kv_len, num_kv_heads, head_dim)
    query = r(q_len, num_q_heads, head_dim)

    def derived_decode_bkv(fold):
        decode, _ = derived_blocks(fold,
                                   num_q_heads=num_q_heads,
                                   num_kv_heads=num_kv_heads,
                                   head_dim=head_dim,
                                   q_dtype=dtype,
                                   kv_dtype=kv_dtype,
                                   page_size=page_size,
                                   total_q_tokens=q_len,
                                   num_page_indices=total_pages,
                                   sm_scale=sm_scale)
        return decode.bkv_sz

    assert derived_decode_bkv(True) > derived_decode_bkv(False)

    def run(fold):
        cache = _build_cache(fold,
                             target_k,
                             target_v,
                             kv_len,
                             total_pages,
                             page_size,
                             num_q_heads,
                             num_kv_heads,
                             head_dim,
                             dtype,
                             kv_dtype,
                             page_indices,
                             sm_scale,
                             blocks=None)
        out, _ = wrapper.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(target_k[kv_len - q_len:kv_len], kv_dtype),
            jnp.asarray(target_v[kv_len - q_len:kv_len], kv_dtype),
            cache,
            jnp.array([kv_len], jnp.int32),
            page_indices,
            jnp.array([0, q_len], jnp.int32),
            jnp.array([1, 1, 1], jnp.int32),
            sm_scale=sm_scale,
            fold_kv_head_dim=fold,
        )
        return np.asarray(out.astype(jnp.float32))

    out_unfolded, out_folded = run(False), run(True)
    # bf16 has ~3 decimal digits; the observed spread here is ~4e-4 on values
    # of order 1e-2.
    np.testing.assert_allclose(out_folded, out_unfolded, rtol=0, atol=2e-3)
    assert np.isfinite(out_folded).all()


def test_folded_writeback_lands_the_same_kv():
    """The cache the kernel wrote must decode back to the unfolded one.

    Guards the writeback half of the flag specifically: a load-only fold would
    still pass an attention-output comparison on the first step and corrupt the
    cache for every step after.
    """
    rng = np.random.default_rng(1)
    kv_len, head_dim, num_kv_heads, num_q_heads, page_size = 512, 256, 1, 32, 256
    dtype, kv_dtype = jnp.bfloat16, jnp.float8_e4m3fn
    total_pages = kv_len // page_size + 1
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k = r(kv_len, num_kv_heads, head_dim)
    target_v = r(kv_len, num_kv_heads, head_dim)

    caches = {
        fold:
        np.asarray(
            _build_cache(fold, target_k, target_v, kv_len, total_pages,
                         page_size, num_q_heads, num_kv_heads, head_dim, dtype,
                         kv_dtype, page_indices,
                         head_dim**-0.5).astype(jnp.float32))
        for fold in (False, True)
    }

    # Unfolded: (pages, page_size, 1, 4, 256), live slots 0 (K) and 1 (V).
    # Folded:   (pages, page_size, 1, 4, 128), slots [K_lo, K_hi, V_lo, V_hi].
    unfolded = caches[False]
    folded = caches[True]
    np.testing.assert_array_equal(
        folded[:, :, 0, 0:2, :].reshape(total_pages, page_size, head_dim),
        unfolded[:, :, 0, 0, :])
    np.testing.assert_array_equal(
        folded[:, :, 0, 2:4, :].reshape(total_pages, page_size, head_dim),
        unfolded[:, :, 0, 1, :])
