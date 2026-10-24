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
"""QSA schedule/backend forwarding and padded request causality."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)
from vllm_torchtpu.kernels.qwen4_exp import decode_indexer, pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata


@dataclass(frozen=True)
class ReferenceTopK:
    """Exact test selector; uses actual StreamIndex scores in TPU tests."""

    reverse: bool = False
    overlaps_scoring: ClassVar[bool] = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        assert scheduling_group_id is None
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None] < row_lengths[:, None], scores, -jnp.inf
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(jnp.isfinite(values), ids, -1)
        if self.reverse:
            count = jnp.sum(ids >= 0, axis=1)
            order = jnp.maximum(count[:, None] - 1 - jnp.arange(k)[None], 0)
            ids = jnp.where(
                jnp.arange(k)[None] < count[:, None],
                jnp.take_along_axis(ids, order, axis=1),
                -1,
            )
            values = jnp.where(
                jnp.arange(k)[None] < count[:, None],
                jnp.take_along_axis(values, order, axis=1),
                -jnp.inf,
            )
        return (
            (ids, jax.lax.bitcast_convert_type(values, jnp.int32))
            if return_scores
            else ids
        )


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize(
    "tokens,requests", [(4, 8), (8, 8), (128, 8), (512, 8), (9, 9)]
)
def test_shared_schedule_forwards_backend_and_causal_metadata(
    monkeypatch, pooled, tokens, requests
):
    seen = []
    backend = ReferenceTopK()
    counts = np.zeros(requests, np.int32)
    counts[:2] = [1, min(tokens - 1, 5)]
    starts = jnp.asarray(np.r_[0, counts.cumsum()], jnp.int32)
    context = jnp.asarray([7, 11] + [0] * (requests - 2), jnp.int32)
    req, logical, valid = _token_metadata(tokens, starts, context, jnp.arange(requests))
    lens = jnp.where(valid, logical + 1, 0)
    query = jnp.ones((tokens, 2, 128), jnp.bfloat16)
    table = jnp.arange(requests * 2, dtype=jnp.int32).reshape(requests, 2)

    def record(q, weights, cache, lengths, pages, cu, distribution, **kwargs):
        assert kwargs["topk_backend"] is backend
        seen.append((np.asarray(lengths), np.asarray(cu), kwargs))
        # Independently reconstruct each query's visible complete groups.
        rq = jnp.minimum(
            jnp.sum(jnp.arange(len(q))[:, None] >= cu[None, 1:], 1), len(lengths) - 1
        )
        if kwargs["num_queries_per_block"] == 1:
            visible = lengths // 4
        else:
            visible = jnp.maximum(
                (lengths[rq] - (cu[rq + 1] - jnp.arange(len(q))) + 1) // 4, 0
            )
            visible = jnp.where(jnp.arange(len(q)) < cu[-1], visible, 0)
            assert kwargs["components"].schedule.causal_position_offset == 3
        return jnp.where(
            jnp.arange(16)[None] < visible[:, None], jnp.arange(16)[None], -1
        )

    monkeypatch.setattr(decode_indexer, "streamindex_topk", record)
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)
    with jax.disable_jit():
        if pooled:
            got = pooled_qsa.select_blocks(
                query,
                jnp.zeros((requests * 2, 128, 1, 2, 128), jnp.bfloat16),
                lens,
                table,
                req,
                starts,
                context,
                ratio=4,
                block_topk=16,
                topk_backend=backend,
            )
        else:
            got = decode_indexer.select_decode_blocks(
                query,
                jnp.zeros((requests * 2, 32, 1, 128), jnp.bfloat16),
                lens,
                table,
                req,
                starts=starts,
                context=context,
                block_topk=16,
                topk_backend=backend,
            )
    expected = np.full((tokens, 16), -1, np.int32)
    for request, count in enumerate(counts):
        for local in range(count):
            row = int(starts[request]) + local
            n = min(16, (int(context[request]) + local + 1) // 4)
            expected[row, :n] = np.arange(n)
    np.testing.assert_array_equal(got, expected)
    assert all(
        (c[2]["num_queries_per_block"] != 1) == (tokens > requests) for c in seen
    )
    if tokens == 512:
        assert seen[0][2]["chunk_tokens"] == 512


def test_default_is_original_sparsecore_and_jit_backend_identity(monkeypatch):
    traces = []

    def record(q, weights, cache, lengths, pages, cu, distribution, **kwargs):
        backend = kwargs["topk_backend"]
        traces.append(backend)
        return jnp.full(
            (len(q), kwargs["k"]),
            1 if isinstance(backend, SparseCoreTopK) else 2,
            jnp.int32,
        )

    monkeypatch.setattr(decode_indexer, "streamindex_topk", record)
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)
    fn = decode_indexer.select_decode_blocks
    fn.clear_cache()
    args = (
        jnp.zeros((1, 1, 128), jnp.bfloat16),
        jnp.zeros((1, 32, 1, 128), jnp.bfloat16),
        jnp.array([4], jnp.int32),
        jnp.zeros((1, 1), jnp.int32),
        jnp.zeros((1,), jnp.int32),
    )
    np.testing.assert_array_equal(fn(*args), 1)
    np.testing.assert_array_equal(fn(*args, topk_backend=ReferenceTopK()), 2)
    np.testing.assert_array_equal(fn(*args), 1)
    assert traces == [SparseCoreTopK(), ReferenceTopK()]
    fn.clear_cache()


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens,requests", [(4, 8), (8, 8), (128, 8), (512, 8)])
@pytest.mark.parametrize("alternate", [False, True])
def test_native_schedule_matches_independent_visibility(tokens, requests, alternate):
    rng = np.random.default_rng(1964)
    # Positive, distinct integer keys and positive queries avoid near-boundary
    # numerical ties while testing selection beyond the512-winner budget.
    counts = np.zeros(requests, np.int32)
    counts[:2] = [max(1, tokens // 3), tokens - max(1, tokens // 3)]
    starts = jnp.asarray(np.r_[0, counts.cumsum()], jnp.int32)
    context = jnp.asarray([2560, 2571] + [0] * (requests - 2), jnp.int32)
    req, logical, valid = _token_metadata(tokens, starts, context, jnp.arange(requests))
    lens = jnp.where(valid, logical + 1, 0)
    groups_per_request = 1024
    table = rng.permutation(requests * 32).reshape(requests, 32).astype(np.int32)
    cache = np.zeros((requests * 32, 32, 1, 128), np.float32)
    order = rng.permutation(groups_per_request).astype(np.float32) + 1
    for r in range(requests):
        cache[table[r], :, 0, 0] = order.reshape(32, 32)
    cache = jnp.asarray(cache, jnp.bfloat16)
    before = np.asarray(cache).copy()
    q = jnp.zeros((tokens, 2, 128), jnp.bfloat16).at[:, :, 0].set(1)
    kwargs = dict(topk_backend=ReferenceTopK()) if alternate else {}
    ids = np.asarray(
        decode_indexer.select_decode_blocks(
            q,
            cache,
            lens,
            jnp.asarray(table),
            req,
            starts=starts,
            context=context,
            **kwargs,
        )
    )
    for row in range(tokens):
        r = int(req[row])
        n = int(lens[row]) // 4
        physical = before[table[r], :, 0, 0].reshape(-1).astype(np.float32)
        # BF16 storage can tie larger integers: compare selected score multiset
        # and eligibility independently, without assuming tie winner ordering.
        live = ids[row][ids[row] >= 0]
        assert len(live) == min(n, 512)
        assert len(set(live)) == len(live)
        assert np.all(live < n)
        np.testing.assert_array_equal(
            np.sort(physical[live]), np.sort(physical[:n])[-512:]
        )
        np.testing.assert_array_equal(ids[row, len(live) :], -1)
    np.testing.assert_array_equal(cache, before)


def test_native_factories_accept_default_and_bound_backend_without_backend_init():
    import functools

    import torch
    from jax._src import xla_bridge
    from torch_tpu._internal import pallas

    initialized = torch.tpu.is_initialized()
    clients = tuple(xla_bridge._backends)
    for name, fn in [
        ("compact_pooled", pooled_qsa.compact_pooled_qsa_step),
    ]:
        for label, core in [
            ("default", fn),
            ("alternate", functools.partial(fn, topk_backend=ReferenceTopK())),
        ]:
            op = pallas.jax_op(f"qsa_routing_schema::{name}_{label}", core)
            assert "topk_backend" not in str(op._schema)
    assert torch.tpu.is_initialized() == initialized
    assert tuple(xla_bridge._backends) == clients


@pytest.mark.parametrize("tokens", [1, 4])
def test_pooled_step_forwards_geometry_and_backend_before_attention(
    monkeypatch, tokens
):
    backend = ReferenceTopK()
    seen = []

    def select(query, pool, lengths, table, req, starts, context, **kwargs):
        seen.append(
            (kwargs["topk_backend"], starts, context, kwargs["vmem_limit_bytes"])
        )
        return jnp.full((len(query), 16), -1, jnp.int32)

    # Record just the indexer boundary. The native lifecycle test separately
    # executes the real cache I/O and full attention on TPU.
    monkeypatch.setattr(pooled_qsa, "select_blocks", select)
    monkeypatch.setattr(pooled_qsa, "write_main_cache", lambda pool, *a: pool)

    def write_compressed(pool, *args, logical_block_size):
        assert logical_block_size == 128
        return pool

    monkeypatch.setattr(pooled_qsa, "write_compressed", write_compressed)
    monkeypatch.setattr(pooled_qsa, "write_ring_and_tags", lambda pool, *a: pool)
    monkeypatch.setattr(
        pooled_qsa,
        "read_ring_and_tags",
        lambda pool, ids, *a: (
            jnp.zeros((len(ids), 8 * 128), jnp.bfloat16),
            jnp.zeros((len(ids), 16), jnp.int32),
        ),
    )
    monkeypatch.setattr(
        pooled_qsa,
        "qsa_attention",
        lambda q, gate, blocks, lengths, table, pool, starts, **kw: (
            jnp.zeros_like(q),
            pool,
        ),
    )
    from types import SimpleNamespace

    monkeypatch.setattr(
        pooled_qsa.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=16 * 1024**2),
    )
    norm = jnp.zeros((128,), jnp.bfloat16)
    starts = jnp.array([0, tokens, tokens], jnp.int32)
    context = jnp.zeros((2,), jnp.int32)
    pooled_qsa.compact_pooled_qsa_step(
        jnp.ones((tokens, 256), jnp.bfloat16),
        jnp.ones((tokens, 128), jnp.bfloat16),
        jnp.ones((tokens, 128), jnp.bfloat16),
        jnp.ones((tokens, 256), jnp.bfloat16),
        norm,
        norm,
        norm,
        norm,
        jnp.arange(tokens, dtype=jnp.int32),
        jnp.tile(jnp.array([[1, 0]], jnp.bfloat16), (8, 1)),
        jnp.zeros((4, 128, 1, 2, 128), jnp.bfloat16),
        jnp.zeros((4, 16, 1, 2, 128), jnp.bfloat16),
        starts,
        context,
        jnp.array([3, 0], jnp.int32),
        jnp.array([[1], [0]], jnp.int32),
        jnp.array([[2], [0]], jnp.int32),
        compress_ratio=4,
        token_topk=64,
        norm_eps=1e-6,
        rotary_dim=2,
        topk_backend=backend,
    )
    assert len(seen) == 1
    assert seen[0][0] is backend and seen[0][1] is starts and seen[0][2] is context
    assert seen[0][3] == 16 * 1024**2
