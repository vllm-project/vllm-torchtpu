# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Where the prefill's attention of a step with cached prefixes finds its keys."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.megakernel.qwen38.attention import (
    attention_path,
    history_pages,
    local_attention,
    pool_keys,
    reference_attention,
)

PAGE, CORES = 128, 8
SPAN = PAGE * CORES


def test_pages_follow_each_request_in_order_and_flat_positions_hold_its_positions():
    tables = np.zeros((4, 3), np.int32)
    tables[0] = [7, 8, 9]
    tables[1] = [3, 4, 0]
    kv_lens = jnp.asarray([2 * SPAN + 5, SPAN, 0, 0])
    distribution = jnp.asarray([0, 2, 2])
    pages, first, needed, limits = history_pages(
        kv_lens, tables, distribution, PAGE, CORES, 6
    )
    # Three blocks of the first request, one of the second, then null pages.
    assert list(np.asarray(pages)) == [7, 8, 9, 3, 0, 0]
    assert list(np.asarray(first))[:2] == [0, 3 * SPAN]
    assert int(needed) == 4
    # Positions of each run's block the request holds: its last block 5.
    assert list(np.asarray(limits)) == [2 * SPAN + 5, SPAN + 5, 5, SPAN, 0, 0]


def test_the_path_takes_the_smallest_size_that_holds_the_pages():
    q_lens = jnp.asarray([0, 100, 100])  # cu_q_lens of one 100-token request
    distribution = jnp.asarray([0, 1, 1])

    def path(kv_len, pages):
        kv_lens = jnp.asarray([kv_len, 0])
        return int(attention_path(kv_lens, q_lens, distribution, pages, (64, 128)))

    assert path(100, 1) == 0  # no cached prefix: the step's own keys
    assert path(5000, 5) == 1
    assert path(100000, 98) == 2
    assert path(200000, 196) == 3  # too many: the streaming kernel


def test_pool_keys_are_rows_of_each_heads_keys_then_values():
    heads, dim = 2, 16
    k = np.random.default_rng(0).standard_normal((PAGE, heads, dim)).astype(np.float32)
    v = np.random.default_rng(1).standard_normal((PAGE, heads, dim)).astype(np.float32)
    # The cache's page: [2 * heads (each head's key then value), dim / 4, 4, page].
    page = np.concatenate([k, v], axis=-1).reshape(PAGE, 2 * heads, dim // 4, 4)
    cache = jnp.asarray(page.transpose(1, 2, 3, 0)[None], jnp.float8_e4m3fn)
    rows = np.asarray(pool_keys(cache, jnp.asarray([0])), np.float32)
    expect = np.concatenate([k.reshape(PAGE, -1), v.reshape(PAGE, -1)], axis=1)
    np.testing.assert_array_equal(
        rows, np.asarray(jnp.asarray(expect, jnp.float8_e4m3fn), np.float32)
    )


def test_attention_over_pooled_keys_matches_the_reference():
    # One request of 3 blocks: 2,560 cached positions, then 512 new ones (the
    # queries); its keys in history_pages order, dealt a page per core.
    hq, hkv, dim, cached, new = 2, 1, 128, 2560, 512
    total = cached + new
    blocks = -(-total // SPAN)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((new, hq, dim)).astype(np.float32)
    k = rng.standard_normal((blocks * SPAN, hkv * dim)).astype(np.float32)
    v = rng.standard_normal((blocks * SPAN, hkv * dim)).astype(np.float32)
    flat = np.arange(blocks * SPAN)
    owner, slot = (flat // PAGE) % CORES, (flat // SPAN) * PAGE + flat % PAGE
    order = np.argsort(owner * blocks * PAGE + slot)  # gathered row -> position
    positions = cached + np.arange(new)
    reference = reference_attention(
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(k, jnp.float8_e4m3fn),
        jnp.asarray(v, jnp.float8_e4m3fn),
        jnp.asarray(positions),
        jnp.zeros(new, jnp.int32),
        sm_scale=dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
    )
    out = local_attention(
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(k[order], jnp.float8_e4m3fn),
        jnp.asarray(v[order], jnp.float8_e4m3fn),
        jnp.asarray(positions, jnp.int32),
        jnp.zeros(new, jnp.int32),
        sm_scale=dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
        cores=CORES,
        interleave=PAGE,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(reference, np.float32), atol=0.05
    )
