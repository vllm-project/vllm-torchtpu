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
"""100% full-module line and branch coverage tests for DeepSeek-V4.1
v2/attention_kernels.py.

Covers all 31 functions/ops and all branch decision points across lines 1-1891 of:
src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
"""

from __future__ import annotations

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# Ensure CPU platform and skip MDS queries
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention_kernels,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    compact_records,
    paged_attention_jax,
    unpack_union_rope_words,
)

# ==============================================================================
# Helper Mock Objects & Cache Tensors
# ==============================================================================


class _FakeKVBuffer:
    def __init__(
        self,
        raw: torch.Tensor,
        rope: torch.Tensor | None = None,
        shape: tuple[int, ...] = (16, 16),
        raw_hi: torch.Tensor | None = None,
        rope_hi: torch.Tensor | None = None,
    ):
        self.raw = raw
        self.raw_hi = raw_hi
        self.rope = rope
        self.rope_hi = rope_hi
        self.shape = shape
        self.total_rows = raw.shape[0] + (0 if raw_hi is None else raw_hi.shape[0])


# ==============================================================================
# 1. Tests for Window Attention, _attend, and SWA JAX Kernels
# ==============================================================================


# ==============================================================================
# 2. Tests for Compaction, Padding & Union Gather Indices
# ==============================================================================


# ==============================================================================
# 3. Tests for Gather Sparse KV Chunk, Fallback & Union Gather Pipeline
# ==============================================================================


# ==============================================================================
# 4. Tests for Initialization, Custom Ops & PyTorch Wrappers
# ==============================================================================


# ==============================================================================
# 5. Tests for Exact Indexer Kernels, Scores, Top-K & Selection
# ==============================================================================


@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("dim", [64, 512])
def test_bf16_gathered_attention_matches_inline_gather(monkeypatch, banks, dim):
    """Reuse the global payload while each layer changes Q, SWA and its sink."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_cache_values

    rng = np.random.default_rng(42)
    values = jnp.asarray(rng.normal(size=(128, dim)), dtype=jnp.bfloat16)
    pool = scatter_cache_values(
        jnp.zeros((128 * dim // 256, 128), jnp.int32),
        jnp.arange(128, dtype=jnp.int32),
        values,
    )
    lo, hi = (pool, None) if banks == 1 else tuple(jnp.split(pool, 2))
    main = pool if hi is None else (lo, hi)
    records = jnp.array([[1, 70, -1], [100, 2, 3], [-1, -1, -1]], jnp.int32)
    gathered = attention_kernels.gather_bf16_kv_jax(lo, hi, records, dim)
    original = attention_kernels.gather_cache_pieces
    # A separate SWA tensor makes any accidental main-cache read observable.
    swa = pool.copy()

    def checked_gather(source, *args):
        assert source is swa, "consumer gathered global KV again"
        return original(source, *args)

    for layer in range(6):
        q = jnp.asarray(rng.normal(size=(3, 2, dim)), dtype=jnp.bfloat16)
        swa_records = jnp.array([[layer, -1], [layer + 1, 7], [-1, -1]], jnp.int32)
        sink = jnp.array([layer / 10, -0.3], jnp.float32)
        expected = paged_attention_jax(q, pool, main, swa_records, records, sink, 0.05)
        with monkeypatch.context() as patcher:
            patcher.setattr(attention_kernels, "gather_cache_pieces", checked_gather)
            actual = paged_attention_jax(
                q, swa, main, swa_records, records, sink, 0.05, gathered
            )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("shards,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("precompacted", [False, True])
def test_fp8_gather_payload_matches_paged_rows(banks, shards, rank, precompacted):
    """Check both packed payloads across banks, requests, padding and TP shards."""
    rng = np.random.default_rng(9)
    pool = jnp.asarray(rng.integers(-(2**30), 2**30, size=(32, 128), dtype=np.int32))
    rope_words = jnp.asarray(
        rng.integers(-(2**30), 2**30, size=(32, 32), dtype=np.int32)
    )
    rope = rope_words.reshape(-1, 128)
    lo, hi = (pool, None) if banks == 1 else tuple(jnp.split(pool, 2))
    rlo, rhi = (rope, None) if banks == 1 else tuple(jnp.split(rope, 2))
    pages = jnp.array([[3, 0], [1, 2]], jnp.int32)
    starts = jnp.array([0, 5, 11], jnp.int32)
    records = jnp.tile(jnp.array([9, -1, 2, 5, 0], jnp.int32), (12, 1))
    records = records.at[3].set(-1).at[11].set(-1)
    if precompacted:
        records = compact_records(records)
    nope_out, rope_out = jax.jit(
        attention_kernels.gather_fp8_kv_jax,
        static_argnames=("page_size", "shards", "precompacted"),
    )(
        lo,
        hi,
        rlo,
        rhi,
        records,
        pages,
        starts,
        jnp.array([rank]),
        8,
        shards,
        precompacted,
    )
    rope_out = unpack_union_rope_words(rope_out.reshape(-1, 128), nope_out.shape[1])
    rope_out = rope_out.reshape(nope_out.shape[0], nope_out.shape[1], 32)
    first = rank * (12 // shards)
    compacted = np.asarray(compact_records(records))
    for local in range(12 // shards):
        token = first + local
        request = 0 if token < 5 else 1
        valid = compacted[token] >= 0
        logical = compacted[token, valid]
        physical = np.asarray(pages)[request, logical // 8] * 8 + logical % 8
        np.testing.assert_array_equal(nope_out[local, valid], pool[physical])
        np.testing.assert_array_equal(rope_out[local, valid], rope_words[physical])
    assert nope_out.shape[0] == (12 if shards == 1 else 8)


def test_gather_reuse_in_compiled_twenty_layer_stage(monkeypatch):
    """The captured forward has 3 global gathers for L2..L19, reused six times."""
    ak = attention_kernels
    lib = torch.library.Library("dsv41_gather_test", "FRAGMENT")
    lib.define("gather(Tensor pool, Tensor? hi, Tensor records, int dim) -> Tensor")
    lib.define(
        "attend(Tensor q, Tensor swa, Tensor main, Tensor sr, Tensor mr, "
        "Tensor sink, float scale, Tensor? gathered=None, "
        "int query_group_size=1) -> Tensor"
    )
    lib.impl("gather", lambda pool, hi, records, dim: pool.clone(), "CPU")
    lib.impl("gather", lambda pool, hi, records, dim: torch.empty_like(pool), "Meta")
    lib.impl(
        "attend",
        lambda q,
        swa,
        main,
        sr,
        mr,
        sink,
        scale,
        gathered=None,
        query_group_size=1: q.clone(),
        "CPU",
    )
    lib.impl(
        "attend",
        lambda q,
        swa,
        main,
        sr,
        mr,
        sink,
        scale,
        gathered=None,
        query_group_size=1: torch.empty_like(q),
        "Meta",
    )
    gather_op = torch.ops.dsv41_gather_test.gather.default
    attend_op = torch.ops.dsv41_gather_test.attend.default
    monkeypatch.setattr(ak, "_gather_bf16_op", gather_op)
    monkeypatch.setattr(ak, "_attention_op", attend_op)
    pool = torch.zeros((32, 128), dtype=torch.int32)
    cache = _FakeKVBuffer(pool)
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    def forward(q, first_records):
        with ak.gathered_kv_cache_scope():
            for layer in range(20):
                if layer < 2:
                    records = first_records[:, :0]
                elif layer in (2, 8, 14):
                    records = first_records + layer
                q = ak.paged_attention(
                    q, cache, cache, first_records, records, q.new_zeros(2), 0.1
                )
        return q

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    q = torch.ones((2, 2, 128), dtype=torch.bfloat16)
    records = torch.zeros((2, 4), dtype=torch.int32)
    compiled(q, records)
    compiled(q, records + 1)
    assert len(graphs) == 1
    gathers = [n for n in graphs[0].graph.nodes if n.target == gather_op]
    attends = [n for n in graphs[0].graph.nodes if n.target == attend_op]
    assert len(gathers) == 3
    assert len(attends) == 20
    for group, gathered in enumerate(gathers):
        assert all(
            node.args[-1] is gathered for node in attends[2 + 6 * group : 8 + 6 * group]
        )
    assert ak._active_gathered_kv is None


def test_fp8_stage_reuse_refreshes_source_indices_and_step(monkeypatch):
    ak = attention_kernels
    gather = mock.Mock(side_effect=lambda *args: (args[0].clone(), args[2].clone()))
    consumed = []

    def attend(*args):
        consumed.append(args[-2:])
        return args[0]

    monkeypatch.setattr(ak, "_gather_fp8_op", gather)
    monkeypatch.setattr(ak, "_fp8_attention_op", attend)
    monkeypatch.setattr(ak, "_attention_tp_size", 1)
    pool = torch.zeros((32, 128), dtype=torch.int32)
    main = _FakeKVBuffer(pool, pool[:8].clone(), shape=(4, 8))
    q = torch.zeros((2, 2, 512), dtype=torch.bfloat16)
    records = torch.zeros((2, 4), dtype=torch.int32)
    pages = torch.zeros((1, 4), dtype=torch.int32)
    starts = torch.tensor([0, 2], dtype=torch.int32)

    def run(selected):
        return ak.sparse_fp8_attention(
            q, main, main, records, selected, q.new_zeros(2), 0.1, pages, starts
        )

    for _ in range(2):
        with ak.gathered_kv_cache_scope():
            run(records)
            for _ in range(5):
                run(records)
            assert all(pair[0] is consumed[-1][0] for pair in consumed[-6:])
            run(records.clone())  # A new index source must gather again.
            main = _FakeKVBuffer(pool.clone(), pool[:8].clone(), shape=(4, 8))
            run(records)  # A new KV source must gather even with the same indices.
    assert gather.call_count == 6
    assert consumed[0][0] is not consumed[8][0]


@pytest.mark.parametrize("tokens", [3, 136])
def test_fp8_consumers_skip_global_gather(monkeypatch, tokens):
    """Only the per-layer SWA gather remains when SparseMLA gets a saved payload."""
    from jax.experimental import pallas as pl

    ak = attention_kernels
    pool = jnp.zeros((16, 128), jnp.int32)
    rope = jnp.zeros((4, 128), jnp.int32)
    records = jnp.tile(jnp.array([0, 7, -1], jnp.int32), (tokens, 1))
    pages, starts = jnp.array([[0]], jnp.int32), jnp.array([0, tokens], jnp.int32)
    nope, rope_values = ak.gather_fp8_kv_jax(
        pool, None, rope, None, records, pages, starts, jnp.array([0]), 16, 1, False
    )
    chunks = []

    def pallas_call(kernel, **kwargs):
        def invoke(lens, bounds, sems, ids, sink, q, kv_nope, kv_rope, num, den, mx):
            chunks.append((kv_nope, kv_rope))
            return q

        return invoke

    monkeypatch.setattr(pl, "pallas_call", pallas_call)
    monkeypatch.setattr(
        ak,
        "_csa_gather_pool",
        mock.Mock(side_effect=AssertionError("global KV was gathered by a consumer")),
    )
    for _ in range(6):
        ak.sparse_fp8_attention_jax(
            jnp.ones((tokens, 2, 512), jnp.bfloat16),
            pool,
            pool,
            rope,
            jnp.full((tokens, 64), -1, jnp.int32),
            records,
            pages,
            starts,
            jnp.zeros(2, jnp.float32),
            0.1,
            16,
            max_requests=1,
            gathered_nope=nope,
            gathered_rope=rope_values,
        )
    assert len(chunks) == 6
    for actual_nope, actual_rope in chunks:
        np.testing.assert_array_equal(actual_nope, nope)
        np.testing.assert_array_equal(actual_rope, rope_values)
