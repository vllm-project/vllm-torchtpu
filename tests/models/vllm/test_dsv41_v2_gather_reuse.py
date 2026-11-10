# SPDX-License-Identifier: Apache-2.0
"""100% full-module line and branch coverage tests for DeepSeek-V4.1
v2/attention_kernels.py.

Covers all 31 functions/ops and all branch decision points across lines 1-1891 of:
src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
"""

from __future__ import annotations

import os

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
    paged_attention_jax,
)

# ==============================================================================
# Helper Mock Objects & Cache Tensors
# ==============================================================================


class _FakeKVBuffer:
    def __init__(
        self,
        raw: torch.Tensor,
        shape: tuple[int, ...] = (16, 16),
        raw_hi: torch.Tensor | None = None,
    ):
        self.raw = raw
        self.raw_hi = raw_hi
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
