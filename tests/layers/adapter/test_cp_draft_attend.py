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
"""The drafter's paged window-attention kernel against its XLA reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.adapter.cp_draft_attention import (
    attend_block,
    attend_cached,
    attend_cached_reference,
)


@pytest.mark.parametrize(
    "dcp, kv_heads, dtype, scale",
    [
        (2, 1, jnp.float8_e4m3fn, None),
        (2, 1, jnp.float8_e4m3fn, 0.5),
        (8, 4, jnp.float8_e4m3fn, None),
        (8, 4, jnp.bfloat16, None),
    ],
)
def test_attend_cached_matches_reference(dcp, kv_heads, dtype, scale):
    rng = np.random.default_rng(0)
    packing = 32 // jax.dtypes.itemsize_bits(dtype)
    words = -(-2 * kv_heads // packing)
    cache = jnp.asarray(
        rng.standard_normal((64, 128, words, packing, 128)), jnp.float32
    ).astype(dtype)
    # Long, short, partial-block and empty (padding) requests.
    seq_lens = jnp.asarray([3000, 9, 700, 130, 0], jnp.int32)
    q_lens = jnp.asarray([8, 8, 5, 8, 0], jnp.int32)
    cu_q_lens = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(q_lens)])
    tables = jnp.asarray(rng.integers(1, 64, (5, 32)), jnp.int32)
    q = jnp.asarray(rng.standard_normal((40, 16, 128)), jnp.float32).astype(
        jnp.bfloat16
    )
    kwargs = dict(
        kv_heads=kv_heads,
        max_query_len=8,
        sliding_window=512,
        sm_scale=0.088,
        dcp=dcp,
        k_scale=scale,
        v_scale=scale,
        interpret=True,
    )
    for rank in range(dcp):
        args = (q, cache, seq_lens, tables, cu_q_lens)
        want_out, want_lse = attend_cached_reference(
            *args, rank=jnp.int32(rank), **kwargs
        )
        got_out, got_lse = attend_cached(*args, rank=jnp.int32(rank), **kwargs)
        np.testing.assert_allclose(
            np.asarray(got_out, np.float32),
            np.asarray(want_out, np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(got_lse), np.asarray(want_lse), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("dcp, kv_heads", [(2, 1), (8, 4)])
def test_uniform_blocks_match_the_general_layout(dcp, kv_heads):
    """A draft forward's rows (`uniform_blocks`: 7 per request, padding rows
    after them): the reshaped paths equal the general ones, a request without
    cached tokens (the draft's padding requests) included."""
    rng = np.random.default_rng(1)
    block, num_reqs, num_tokens = 7, 4, 32
    words = -(-2 * kv_heads // 4)
    cache = jnp.asarray(
        rng.standard_normal((64, 128, words, 4, 128)), jnp.float32
    ).astype(jnp.float8_e4m3fn)
    seq_lens = jnp.asarray([3000, 7, 700, 135], jnp.int32)
    cu_q_lens = jnp.arange(num_reqs + 1, dtype=jnp.int32) * block
    tables = jnp.asarray(rng.integers(1, 64, (num_reqs, 32)), jnp.int32)
    q = jnp.asarray(rng.standard_normal((num_tokens, 16, 128)), jnp.bfloat16)
    key = jnp.asarray(rng.standard_normal((num_tokens, kv_heads, 128)), jnp.bfloat16)
    value = jnp.asarray(rng.standard_normal((num_tokens, kv_heads, 128)), jnp.bfloat16)
    for rank in range(dcp):
        outs = [
            attend_cached(
                q, cache, seq_lens, tables, cu_q_lens, kv_heads=kv_heads,
                max_query_len=block, sliding_window=512, sm_scale=0.088, dcp=dcp,
                rank=jnp.int32(rank), interpret=True, uniform_blocks=uniform,
            )
            for uniform in (False, True)
        ]
        for got, want in zip(outs[1], outs[0]):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    blocks = [
        attend_block(
            q, key, value, seq_lens, cu_q_lens, max_query_len=block,
            sm_scale=0.088, causal=False, uniform_blocks=uniform,
        )
        for uniform in (False, True)
    ]
    for got, want in zip(blocks[1], blocks[0]):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
