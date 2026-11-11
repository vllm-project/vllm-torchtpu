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
"""`msa_decode_attention` against `msa_attention_lse` (the DCP attend kernel)
on paged decode inputs: bf16 and 4-bit caches, one and two KV heads per cache
row, transposed and row-major score tiles."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention_lse,
    msa_decode_attention,
    msa_fp4,
)

D = 128
HG = 16
TOPK = 16
PAGE = 128


def _inputs(num_heads, fp4, seed=0, num_seqs=2, q_len=8, extra_tokens=8):
    """Paged decode inputs: `num_seqs` verify groups of `q_len` tokens and
    `extra_tokens` padding tokens, page keys ``seq * pages_per_seq + block``."""
    rng = np.random.default_rng(seed)
    pages_per_seq = 6
    num_pages = num_seqs * pages_per_seq + 1
    table = (rng.permutation(num_pages - 1) + 1).astype(np.int32)
    num_tokens = num_seqs * q_len + extra_tokens
    selected = np.full((num_tokens, num_heads, TOPK), -1, np.int32)
    local_page = np.full((num_tokens,), -1, np.int32)
    local_limit = np.zeros((num_tokens,), np.int32)
    for b in range(num_seqs):
        for t in range(q_len):
            tok = b * q_len + t
            own = pages_per_seq - 1 - (t % 2)  # the last two blocks hold the group
            local_page[tok] = b * pages_per_seq + own
            local_limit[tok] = 1 + (t * 13) % PAGE
            for g in range(num_heads):
                picks = rng.choice(pages_per_seq, size=3, replace=False)
                keys = sorted({b * pages_per_seq + own, *(b * pages_per_seq + picks)})
                selected[tok, g, : len(keys)] = keys
    # A token with no page at all (its output must be 0, lse -inf).
    selected[1] = -1
    local_page[1] = -1
    q = jnp.asarray(rng.standard_normal((num_tokens, num_heads * HG, D)), jnp.bfloat16)
    width = num_heads * 2 * D
    if fp4:
        nib = rng.integers(0, 16, size=(num_pages, PAGE, width), dtype=np.uint8)
        nib = np.where(nib == 8, 0, nib).astype(np.uint8)
        kv = msa_fp4.pack_rows(jnp.asarray(nib))
        scale = rng.uniform(0.02, 0.2, (num_pages, msa_fp4.scale_rows(num_heads), PAGE))
        scale[0] = np.nan  # an unwritten page's scales read as 0
        scale = jnp.asarray(scale, jnp.float32)
    else:
        kv = jnp.asarray(
            rng.standard_normal((num_pages, PAGE // 2, 2, width)), jnp.bfloat16
        )
        scale = None
    return dict(
        q=q,
        kv_cache=kv,
        selected=jnp.asarray(selected),
        local_page=jnp.asarray(local_page),
        local_limit=jnp.asarray(local_limit),
        page_table=jnp.asarray(table),
        scale_cache=scale,
    )


def _reference(inp, sm_scale):
    return msa_attention_lse.msa_sparse_attention_lse_heads(
        inp["q"],
        inp["kv_cache"],
        inp["selected"],
        inp["local_page"],
        inp["local_limit"],
        inp["page_table"],
        sm_scale=sm_scale,
        block_q=8,
        scale_cache=inp["scale_cache"],
        interpret=True,
        head_major=True,
    )


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("transposed", [True, False])
def test_matches_lse_kernel(fp4, num_heads, transposed):
    sm_scale = 1.0 / np.sqrt(D)
    inp = _inputs(num_heads, fp4)
    out_ref, lse_ref = _reference(inp, sm_scale)
    out, lse = msa_decode_attention.msa_decode_attention_lse(
        inp["q"],
        inp["kv_cache"],
        inp["selected"],
        inp["local_page"],
        inp["local_limit"],
        inp["page_table"],
        sm_scale=sm_scale,
        scale_cache=inp["scale_cache"],
        transposed=transposed,
        tiles_per_step=2,
        interpret=True,
    )
    assert out.shape == out_ref.shape and lse.shape == lse_ref.shape
    out, out_ref = np.asarray(out, np.float32), np.asarray(out_ref, np.float32)
    lse, lse_ref = np.asarray(lse), np.asarray(lse_ref)
    np.testing.assert_array_equal(np.isfinite(lse), np.isfinite(lse_ref))
    assert not np.isfinite(lse[:, 1]).any() and not out[:, 1].any()
    fin = np.isfinite(lse_ref)
    # Same pages and masks; only the order of f32 reductions may differ.
    np.testing.assert_allclose(lse[fin], lse_ref[fin], rtol=0, atol=1e-5)
    np.testing.assert_allclose(out, out_ref, rtol=0, atol=2e-2 * np.abs(out_ref).max())
    assert np.mean(out == out_ref) > 0.95
