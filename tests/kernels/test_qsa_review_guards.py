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
"""Request-local routing, empty ABIs, and public SMEM capacity boundaries."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa, qsa_state
from vllm_torchtpu.kernels.sparse_gqa_decode import batching


@pytest.fixture
def info(monkeypatch):
    value = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: value)
    return value


@pytest.mark.parametrize(
    "tokens,prefill,reverse,idle,verify",
    [
        (257, 2, False, 0, False),
        (512, 257, False, 0, False),
        (512, 2, True, 17, False),
        (512, 257, True, 17, False),
        (1024, 257, False, 17, True),
    ],
)
def test_mixed_attention_routes_real_request_lengths(
    monkeypatch, info, tokens, prefill, reverse, idle, verify
):
    from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
    from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive

    counts = [1] * 255 + [prefill]
    if verify:
        counts[253:255] = [4, 4]
    counts += [0] * idle
    starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
    owners = np.zeros(tokens, np.int32)
    expected_route = {}
    for request, count in enumerate(counts):
        for row in range(int(starts[request]), int(starts[request + 1])):
            owners[row] = len(counts) - request if reverse else request + 1
            expected_route[row] = "decode" if request < 255 else "prefill"
    rows = np.arange(tokens, dtype=np.int32)
    query = jnp.broadcast_to(
        jnp.asarray(rows, jnp.bfloat16)[:, None, None], (tokens, 1, 128)
    )
    gate = jnp.broadcast_to(
        jnp.asarray(rows % 7, jnp.bfloat16)[:, None, None], query.shape
    )
    lengths = jnp.where(jnp.arange(tokens) < starts[-1], 65, 0).astype(jnp.int32)
    table = jnp.asarray(owners[:, None], jnp.int32)
    seen = []

    def attend(kind, q, g, blocks, lens, pages, cache, *args, **kw):
        assert kw["kv_heads"] == 1 and kw["ratio"] == 4
        for row, length in zip(np.asarray(blocks[:, 0]), np.asarray(lens)):
            if length:
                seen.append((int(row), kind))
        result = (
            q.astype(jnp.float32) + pages[:, :1, None] + g.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        inactive = kw["initial_output"] if kind == "decode" else 0
        return jnp.where((lens > 0)[:, None, None], result, inactive), cache

    monkeypatch.setattr(
        qsa_rpa, "_rpa_attention_batch", lambda *a, **kw: attend("decode", *a, **kw)
    )
    monkeypatch.setattr(
        qsa_rpa, "rpa_attention", lambda *a, **kw: attend("decode", *a, **kw)
    )
    monkeypatch.setattr(
        tc, "block_sparse_gqa_attention", lambda *a, **kw: attend("prefill", *a, **kw)
    )
    monkeypatch.setattr(
        adaptive, "is_supported_by_adaptive_prefill", lambda *a, **kw: False
    )
    pool = jnp.zeros((1, 128, 1, 2, 128), jnp.bfloat16)
    with jax.disable_jit():
        output, owner = qsa_rpa.qsa_attention(
            query,
            gate,
            jnp.broadcast_to(
                jnp.arange(tokens, dtype=jnp.int32)[:, None], (tokens, 16)
            ),
            lengths,
            table,
            pool,
            jnp.asarray(starts),
            request_distribution=jnp.array(
                [253 if verify else 255, 255, 256], jnp.int32
            ),
            kv_heads=1,
            ratio=4,
        )
    assert owner is pool
    assert sorted(seen) == sorted(expected_route.items())
    expected = (
        query.astype(jnp.float32) + table[:, :1, None] + gate.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    expected = jnp.where((lengths > 0)[:, None, None], expected, 0)
    np.testing.assert_array_equal(output, expected)
    assert all(
        count != 1 or expected_route[int(starts[r])] == "decode"
        for r, count in enumerate(counts)
    )


@pytest.mark.parametrize(
    "writer", [pooled_qsa._write_main_cache_universal, pooled_qsa.write_main_cache]
)
def test_empty_main_write_returns_before_pallas(monkeypatch, writer):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty write launched Pallas")

    monkeypatch.setattr(pl, "pallas_call", forbidden)
    pool = jnp.zeros((2, 128, 1, 2, 128), jnp.bfloat16)
    empty = jnp.zeros(0, jnp.int32)
    assert (
        writer(
            pool, pool.reshape(-1, 1, 2, 128)[:0], empty, empty, empty.astype(jnp.bool_)
        )
        is pool
    )


def test_empty_pooled_step_returns_before_reshape_or_producer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty step entered producer/Pallas")

    monkeypatch.setattr(pooled_qsa, "produce", forbidden)
    monkeypatch.setattr(pl, "pallas_call", forbidden)
    pool = jnp.zeros((2, 128, 1, 2, 128), jnp.bfloat16)
    producer = (
        jnp.zeros((0, 256), jnp.bfloat16),
        jnp.zeros((0, 128), jnp.bfloat16),
        jnp.zeros((0, 128), jnp.bfloat16),
        jnp.zeros((0, 256), jnp.bfloat16),
        jnp.zeros(128, jnp.bfloat16),
        jnp.zeros(128, jnp.bfloat16),
        jnp.zeros(128, jnp.bfloat16),
        jnp.zeros(128, jnp.bfloat16),
        jnp.zeros(0, jnp.int32),
        jnp.zeros((8, 128), jnp.bfloat16),
    )
    metadata = (
        jnp.zeros(1, jnp.int32),
        jnp.zeros(0, jnp.int32),
        jnp.zeros(0, jnp.int32),
        jnp.zeros(0, jnp.int32),
        jnp.zeros(0, jnp.int32),
    )
    index_pool = jnp.zeros((2, 16, 1, 2, 128), jnp.bfloat16)
    result = pooled_qsa.compact_pooled_qsa_step(
        *producer,
        pool,
        index_pool,
        *metadata,
        request_distribution=jnp.zeros(3, jnp.int32),
        compress_ratio=4,
        token_topk=64,
        norm_eps=1e-6,
        rotary_dim=2,
    )
    assert result[0] is pool
    assert result[1] is index_pool
    assert result[2].shape == (0, 128) and result[2].dtype == jnp.bfloat16
    assert result[3].shape == (0, 67) and result[3].dtype == jnp.int32


def test_state_capacity_has_explicit_error_before_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid state reached Pallas")

    monkeypatch.setattr(pl, "pallas_call", forbidden)
    pool = jnp.zeros((2, 1, 1, 2, 128), jnp.bfloat16)
    with pytest.raises(
        ValueError, match="state regions exceed the physical page capacity"
    ):
        qsa_state.write_ring_and_tags(
            pool,
            jnp.zeros((1, 256), jnp.bfloat16),
            jnp.zeros((1, 256), jnp.int32),
            jnp.array([1], jnp.int32),
        )


@pytest.mark.parametrize("columns", [0, -1])
def test_rpa_rejects_nonpositive_width(columns):
    with pytest.raises(ValueError, match="positive"):
        batching.rpa_query_batch_size(
            columns,
            query=jax.ShapeDtypeStruct((1, 4, 128), jnp.bfloat16),
            cache=jax.ShapeDtypeStruct((1, 128, 1, 2, 128), jnp.bfloat16),
            kv_heads=1,
            ratio=4,
        )


@pytest.mark.parametrize("lanes,columns", [(128, 513), (64, 41000)])
def test_rpa_one_row_exact_capacity_boundary(monkeypatch, lanes, columns):
    # 128 lanes, ratio4: the actual minimum batch8 schedule occupies 834052 B.
    schedule_bytes = 834052 if lanes == 128 else 417028
    required = schedule_bytes + 4 + 4 * (columns + 4)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=lanes,
            smem_capacity_bytes=required,
            vmem_capacity_bytes=32 * 1024 * 1024,
        ),
    )
    assert (
        batching.rpa_query_batch_size(
            columns,
            query=jax.ShapeDtypeStruct((1, 4, 128), jnp.bfloat16),
            cache=jax.ShapeDtypeStruct((1, 128, 1, 2, 128), jnp.bfloat16),
            kv_heads=1,
            ratio=4,
        )
        == 1
    )
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=lanes,
            smem_capacity_bytes=required - 1,
            vmem_capacity_bytes=32 * 1024 * 1024,
        ),
    )
    with pytest.raises(ValueError, match="One QSA page-table row.*SMEM capacity"):
        batching.rpa_query_batch_size(
            columns,
            query=jax.ShapeDtypeStruct((1, 4, 128), jnp.bfloat16),
            cache=jax.ShapeDtypeStruct((1, 128, 1, 2, 128), jnp.bfloat16),
            kv_heads=1,
            ratio=4,
        )


@pytest.mark.parametrize("prefill", [False, True])
def test_pure_batches_have_only_one_live_segment(monkeypatch, info, prefill):
    tokens = 8192 if prefill else 512
    counts = [3072, 5120, 0] if prefill else [1] * 256 + [0] * 17
    starts = jnp.asarray(np.r_[0, np.cumsum(counts)], jnp.int32)
    q = jnp.ones((tokens, 1, 128), jnp.bfloat16)
    seen = []
    pool = jnp.zeros((1, 128, 1, 2, 128), jnp.bfloat16)

    def run(kind, query, gate, ids, lengths, table, owner, *args, **kw):
        seen.extend((kind, int(row)) for row in np.flatnonzero(np.asarray(lengths)))
        inactive = kw["initial_output"] if kind == "decode" else 0
        return jnp.where((lengths > 0)[:, None, None], query, inactive), owner

    monkeypatch.setattr(
        qsa_rpa, "_prefill_attention", lambda *a, **kw: run("prefill", *a, **kw)
    )
    monkeypatch.setattr(
        qsa_rpa, "rpa_attention", lambda *a, **kw: run("decode", *a, **kw)
    )
    with jax.disable_jit():
        result, owner = qsa_rpa.qsa_attention(
            q,
            q,
            jnp.zeros((tokens, 16), jnp.int32),
            jnp.ones(tokens, jnp.int32),
            jnp.zeros((tokens, 1), jnp.int32),
            pool,
            starts,
            request_distribution=jnp.array(
                [0, 0, 2] if prefill else [256, 256, 256], jnp.int32
            ),
            kv_heads=1,
            ratio=4,
        )
    live = int(starts[-1])
    assert seen == [("prefill" if prefill else "decode", row) for row in range(live)]
    np.testing.assert_array_equal(result[:live], q[:live])
    np.testing.assert_array_equal(result[live:], 0)
    assert owner is pool


@pytest.mark.parametrize(
    "tokens,counts,distribution",
    [
        (13, [1, 3, 5, 0], [1, 2, 3]),
        (3, [1, 2], [1, 1, 2]),
        (5, [0, 0, 3, 0], [0, 0, 3]),
        (7, [0, 0, 0], [0, 0, 0]),
    ],
)
def test_suffix_keeps_original_rows_and_prefix_tail_drops(
    monkeypatch, tokens, counts, distribution
):
    starts = np.r_[0, np.cumsum(counts)].astype(np.int32)
    query = jnp.arange(tokens * 128, dtype=jnp.float32).reshape(tokens, 1, 128)
    gate = (jnp.arange(tokens, dtype=jnp.float32) % 5 + 1)[:, None, None]
    gate = jnp.broadcast_to(gate, query.shape)
    table = (jnp.arange(tokens, dtype=jnp.int32) * 3 + 11)[:, None]
    blocks = jnp.broadcast_to(jnp.arange(tokens)[:, None], (tokens, 16))
    lengths = jnp.where(jnp.arange(tokens) < starts[-1], 17, 0).astype(jnp.int32)
    pool = jnp.zeros((1, 128, 1, 2, 128), jnp.bfloat16)
    calls = []

    def attend(q, g, ids, lens, pages, owner, *args, **kw):
        calls.append(len(q))
        result = q + 2 * g + 3 * pages[:, :1, None]
        inactive = kw.pop("initial_output", 0)
        return jnp.where((lens > 0)[:, None, None], result, inactive), owner

    def prefill(q, g, ids, lens, pages, owner, local_starts, **kw):
        assert q is query and g is gate and ids is blocks and pages is table
        prefix = int(starts[distribution[1]])
        active = int(starts[distribution[2]])
        np.testing.assert_array_equal(local_starts, np.clip(starts, prefix, active))
        return attend(q, g, ids, lens, pages, owner, **kw)

    monkeypatch.setattr(qsa_rpa, "_prefill_attention", prefill)
    monkeypatch.setattr(qsa_rpa, "rpa_attention", attend)
    monkeypatch.setattr(batching, "rpa_query_batch_size", lambda *a, **kw: 3)
    with jax.disable_jit():
        output, owner = qsa_rpa.qsa_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            pool,
            jnp.asarray(starts),
            request_distribution=jnp.asarray(distribution, jnp.int32),
            kv_heads=1,
            ratio=4,
        )
    expected = np.zeros(query.shape, np.float32)
    for request in range(distribution[2]):
        for row in range(int(starts[request]), int(starts[request + 1])):
            expected[row] = (
                np.asarray(query[row])
                + 2 * np.asarray(gate[row])
                + 3 * np.asarray(table[row, 0])
            )
    np.testing.assert_array_equal(output, expected)
    assert owner is pool
    assert all(size == tokens for size in calls)
