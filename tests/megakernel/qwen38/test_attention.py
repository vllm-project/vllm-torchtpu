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
"""The gathered-KV causal attention kernel (``megakernel.qwen38.attention``)
in TPU interpret mode on the host."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.megakernel.qwen38.attention import (  # noqa: E402
    CHUNK,
    has_history,
    local_attention,
    reference_attention,
    rope_tables,
)
from vllm_torchtpu.megakernel.qwen38.reference import rms, rope  # noqa: E402


@pytest.mark.parametrize("core", [0, 1])
def test_local_attention_matches_reference_over_packed_requests(core):
    q_heads, kv_heads, dim, cores = 8, 2, 128, 2
    tokens = 8 * CHUNK
    # Two requests ([0, 1000) and [1000, 1900)), then padding; the core's rows
    # are flat chunks core, core + cores, ...
    start = np.where(np.arange(tokens) < 1000, 0, 1000)
    flat = np.concatenate(
        [np.arange(c * CHUNK, (c + 1) * CHUNK) for c in range(core, 8, cores)]
    )
    q_start = np.where(flat < 1900, start[flat], flat + 1)
    rng = np.random.default_rng(core)
    q = jnp.asarray(rng.standard_normal((flat.size, q_heads, dim)), jnp.bfloat16)
    kv = rng.standard_normal((2, tokens, kv_heads * dim)) * 4
    k, v = (jnp.asarray(x, jnp.float32).astype(jnp.float8_e4m3fn) for x in kv)
    scales = dict(sm_scale=dim**-0.5, k_scale=0.25, v_scale=0.5)
    flat, q_start = jnp.asarray(flat, jnp.int32), jnp.asarray(q_start, jnp.int32)
    want = reference_attention(q, k, v, flat, q_start, **scales)

    # The kernel reads the keys core-major, as gathered.
    def core_major(x):
        return x.reshape(4, cores, CHUNK, -1).transpose(1, 0, 2, 3).reshape(tokens, -1)

    got = local_attention(
        q,
        core_major(k),
        core_major(v),
        flat,
        q_start,
        cores=cores,
        interpret=True,
        **scales,
    )
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(want, np.float32), atol=2e-2, rtol=1e-2
    )
    assert np.all(np.asarray(got)[np.asarray(q_start > flat)] == 0)


def test_has_history_counts_active_requests_only():
    cu = jnp.asarray([0, 100, 300, 300], jnp.int32)
    fresh = jnp.asarray([100, 200, 7], jnp.int32)  # the third slot is inactive
    assert not bool(has_history(fresh, cu, jnp.asarray([0, 2, 2], jnp.int32)))
    cached = jnp.asarray([100, 250, 0], jnp.int32)
    assert bool(has_history(cached, cu, jnp.asarray([0, 2, 2], jnp.int32)))


def test_local_attention_normalizes_and_rotates_the_projected_queries():
    q_heads, kv_heads, dim, rotary, eps = 8, 2, 256, 64, 1e-6
    tokens = 4 * CHUNK
    rng = np.random.default_rng(3)
    flat = np.arange(tokens)
    positions = jnp.asarray(rng.integers(0, 100_000, tokens), jnp.int32)
    projected = jnp.asarray(
        rng.standard_normal((tokens, q_heads, 2 * dim)) * 3, jnp.bfloat16
    )
    norm = jnp.asarray(rng.standard_normal(dim) * 0.1, jnp.bfloat16)
    kv = rng.standard_normal((2, tokens, kv_heads * dim))
    k, v = (jnp.asarray(x, jnp.float32).astype(jnp.float8_e4m3fn) for x in kv)
    flat, start = jnp.asarray(flat, jnp.int32), jnp.zeros(tokens, jnp.int32)
    scales = dict(sm_scale=dim**-0.5, k_scale=1.0, v_scale=1.0)

    query = rope(rms(projected[..., :dim], norm, eps), positions, rotary, 1e7)
    want = local_attention(query, k, v, flat, start, interpret=True, **scales)
    got = local_attention(
        projected,
        k,
        v,
        flat,
        start,
        q_norm=norm,
        rope=rope_tables(positions, rotary, 1e7),
        rotary_dim=rotary,
        eps=eps,
        interpret=True,
        **scales,
    )
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(want, np.float32), atol=2e-2, rtol=1e-2
    )
