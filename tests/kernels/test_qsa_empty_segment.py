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
"""Empty decode segments skip work while retaining output rows and cache owners."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qsa_review_reference import check_reference

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa


def _stub_inputs(counts, distribution):
    tokens = max(sum(counts) + 3, 8)
    starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
    q = jnp.arange(tokens * 128, dtype=jnp.float32).reshape(tokens, 1, 128) / 32
    gate = jnp.broadcast_to(jnp.arange(tokens)[:, None, None] % 3, q.shape)
    cache = jnp.arange(3 * 128 * 2 * 128, dtype=jnp.float32).reshape(3, 128, 1, 2, 128)
    return (
        q,
        gate,
        jnp.zeros((tokens, 16), jnp.int32),
        jnp.full(tokens, 17, jnp.int32),
        jnp.arange(tokens, dtype=jnp.int32)[:, None] * 3 + 1,
        cache,
        jnp.asarray(starts),
        jnp.asarray(distribution, jnp.int32),
    )


@pytest.mark.parametrize(
    "counts,distribution",
    [
        ([3, 5, 0], [0, 0, 2]),
        ([0, 0, 5, 0], [1, 2, 3]),
        ([1, 0, 1], [3, 3, 3]),
        ([3, 2, 0], [0, 2, 2]),
        ([1, 3, 5, 0], [1, 2, 3]),
        ([3, 5, 2], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
    ],
)
def test_empty_segment_runtime_route_and_owner(monkeypatch, counts, distribution):
    args = _stub_inputs(counts, distribution)
    observed = []

    def result(q, gate, blocks, lengths, table, cache, *a, initial_output=None, **kw):
        inactive = jnp.zeros_like(q) if initial_output is None else initial_output
        values = q + 2 * gate + 3 * table[:, :1, None]
        return jnp.where((lengths > 0)[:, None, None], values, inactive), cache

    def decode(*a, **kw):
        jax.debug.callback(lambda: observed.append(1), ordered=True)
        return result(*a, **kw)

    monkeypatch.setattr(qsa_rpa, "_prefill_attention", result)
    monkeypatch.setattr(qsa_rpa, "rpa_attention", decode)
    before = np.asarray(args[5]).tobytes()

    def run(q, gate, blocks, lengths, table, cache, starts, dist):
        return qsa_rpa.qsa_attention(
            q,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            request_distribution=dist,
            kv_heads=1,
            ratio=4,
        )

    compiled = jax.jit(run, donate_argnums=(5,)).lower(*args).compile()
    output, cache = compiled(*args)
    jax.block_until_ready((output, cache))
    jax.effects_barrier()
    prefix, active = int(args[6][distribution[1]]), int(args[6][distribution[2]])
    assert observed == ([1] if prefix > 0 else [])
    expected = (
        np.asarray(args[0])
        + 2 * np.asarray(args[1])
        + 3 * np.asarray(args[4])[:, :1, None]
    )
    expected[active:] = 0
    np.testing.assert_array_equal(output, expected)
    assert np.asarray(cache).tobytes() == before
    memory = compiled.memory_analysis()
    assert memory is not None and memory.alias_size_in_bytes >= len(before)


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("regime", ["prefill", "verify", "mixed"])
@pytest.mark.parametrize("history", [False, True])
def test_native_empty_segment_matches_attention_reference(
    monkeypatch, fp8, regime, history
):
    tokens, heads, dim, page = 128, 4, 128, 128
    rng = np.random.default_rng(238833)
    starts = np.array([0, 37, 64, 64], np.int32)
    context = np.array([63, 129, 0] if history else [0, 0, 0], np.int32)
    request = np.minimum(
        np.searchsorted(starts[1:], np.arange(tokens), side="right"), 2
    )
    lengths = np.where(
        np.arange(tokens) < 64,
        context[request] + np.arange(tokens) - starts[request] + 1,
        0,
    ).astype(np.int32)
    distribution = np.array(
        {"prefill": [0, 0, 2], "verify": [0, 2, 2], "mixed": [0, 1, 2]}[regime],
        np.int32,
    )
    request_tables = np.array([[2, 4, 6, 8], [1, 3, 5, 7], [0, 0, 0, 0]], np.int32)
    q = jnp.asarray(rng.normal(0, 0.2, (tokens, heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 1, q.shape), jnp.bfloat16)
    key = jnp.asarray(rng.normal(0, 0.2, (9 * page, 1, dim)), jnp.bfloat16)
    value = jnp.asarray(rng.normal(0, 0.2, key.shape), jnp.bfloat16)
    packed = qsa_rpa.pack_rows(key, value, fp8=fp8)
    cache = packed.reshape(9, page, *packed.shape[1:])
    groups = np.arange(48, dtype=np.int32)[None]
    blocks = jnp.asarray(np.where(groups < lengths[:, None] // 4, groups, -1))
    args = (
        q,
        gate,
        blocks,
        jnp.asarray(lengths),
        jnp.asarray(request_tables[request]),
        cache,
        jnp.asarray(starts),
        jnp.asarray(distribution),
    )
    before = np.asarray(cache).tobytes()
    observed = []
    original = qsa_rpa.rpa_attention

    def decode(*a, **kw):
        result = original(*a, **kw)
        jax.debug.callback(lambda: observed.append(1), ordered=True)
        return result

    monkeypatch.setattr(qsa_rpa, "rpa_attention", decode)

    def run(q, gate, blocks, lengths, table, cache, starts, dist):
        return qsa_rpa.qsa_attention(
            q,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            request_distribution=dist,
            kv_heads=1,
            ratio=4,
        )

    compiled = jax.jit(run, donate_argnums=(5,)).lower(*args).compile()
    output, next_cache = compiled(*args)
    jax.block_until_ready((output, next_cache))
    jax.effects_barrier()
    assert observed == ([] if regime == "prefill" else [1])
    assert np.asarray(next_cache).tobytes() == before
    memory = compiled.memory_analysis()
    assert memory is not None and memory.alias_size_in_bytes >= len(before)
    reference_args = (*args[:5], next_cache, args[6])
    check_reference(
        reference_args, output, kv_heads=1, rows=np.arange(64), ratio=4, page_size=page
    )
    assert np.linalg.norm(np.asarray(output, np.float32)) > 0
