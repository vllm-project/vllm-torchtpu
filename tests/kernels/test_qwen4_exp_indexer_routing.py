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

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)
from vllm_torchtpu.kernels.qwen4_exp import decode_indexer, pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing
from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata
from vllm_torchtpu.kernels.qwen4_exp.resident_topk import QwenSparseCoreTopK


@pytest.mark.parametrize("case", ["wide", "chip", "batch", "topk", "carrier"])
def test_decode_reader_view_preserves_page_ids_and_native_keys(monkeypatch, case):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import tuned_block_sizes

    chip = pltpu.ChipVersion.TPU_V5P if case == "chip" else pltpu.ChipVersion.TPU_7X
    monkeypatch.setattr(
        tuned_block_sizes.pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(chip, 1),
    )
    batch = 9 if case == "batch" else 16
    k = 16 if case == "topk" else 512
    native = (jnp.arange(4 * 416 * 128) % 113).astype(jnp.bfloat16)
    cache = native.reshape(4, 208, 2, 128)
    if case == "carrier":
        cache = jax.lax.bitcast_convert_type(cache, jnp.uint8)
        cache = cache.transpose(0, 1, 2, 4, 3).reshape(4, 208, 4, 128)
    pages = jnp.arange(32 * 41, dtype=jnp.int32).reshape(32, 41) % 4
    requests = jnp.arange(batch, dtype=jnp.int32)[::-1]
    calls = []

    def inspect(q, weights, read_cache, lengths, read_pages, cu, dist, **kwargs):
        reader = kwargs["components"].keys
        reader.validate(read_cache, 128, KVLayout.HEAD_ALONG_SUBLANE)
        calls.append(reader.packing)
        np.testing.assert_array_equal(read_pages, pages[requests].reshape(-1))
        if case != "carrier":
            np.testing.assert_array_equal(read_cache.reshape(-1), native)
            assert read_cache.shape[1] * reader.packing == 416
        np.testing.assert_array_equal(cu, jnp.arange(len(q) + 1, dtype=jnp.int32))
        np.testing.assert_array_equal(dist, jnp.full((3,), len(q), jnp.int32))
        return jnp.full((len(q), k), -1, jnp.int32)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", inspect)
    # Inspect concrete reader operands inside the runtime scoring branch.
    info = pltpu.get_tpu_info_for_chip(chip, 1)
    monkeypatch.setattr(
        tuned_block_sizes,
        "_hardware",
        lambda: tuned_block_sizes.DmaHardware.from_tpu_info(info),
    )
    with jax.disable_jit():
        decode_indexer.select_index_blocks(
            jnp.zeros((batch, 4, 128), jnp.bfloat16),
            cache,
            jnp.full((batch,), 65536, jnp.int32),
            pages,
            requests,
            starts=None,
            context=None,
            ratio=4,
            block_topk=k,
        )
    assert calls == [info.get_sublane_tiling(jnp.bfloat16) if case != "carrier" else 2]


@pytest.mark.parametrize("case", ["decode", "prefill", "padding", "backend", "chip"])
def test_sidecar_page_coalescing_preserves_owned_key_addresses(monkeypatch, case):
    """Large pages must preserve physical ownership and exclude sidecar padding."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import tuned_block_sizes

    chip = pltpu.ChipVersion.TPU_V5P if case == "chip" else pltpu.ChipVersion.TPU_7X
    monkeypatch.setattr(
        tuned_block_sizes.pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(chip, 1),
    )
    physical_rows = 448 if case == "padding" else 416
    pool = (jnp.arange(4 * physical_rows * 128) % 127).astype(jnp.bfloat16)
    pool = pool.reshape(4, physical_rows // 2, 1, 2, 128)
    requests = 8 if case == "prefill" else 16
    table = jnp.asarray(np.random.default_rng(52).integers(4, size=(requests, 41)))
    req = jnp.arange(16, dtype=jnp.int32)[::-1] % requests
    columns = jnp.array([0, 31, 32, 415, 416, 17055])
    observed = []

    def read(q, cache, lengths, pages, request_ids, **kwargs):
        page_size = cache.shape[1] * 2
        observed.append(page_size)
        keys = cache.reshape(-1, page_size, 128)
        ids = pages[request_ids[:, None], columns[None] // page_size]
        return keys[ids, columns[None] % page_size]

    monkeypatch.setattr(pooled_qsa, "select_index_blocks", read)
    actual = pooled_qsa.select_blocks(
        jnp.zeros((16, 4, 128), jnp.bfloat16),
        pool,
        jnp.full((16,), 67584, jnp.int32),
        table,
        req,
        jnp.arange(requests + 1, dtype=jnp.int32),
        jnp.zeros((requests,), jnp.int32),
        ratio=4,
        block_topk=512,
        topk_backend=ReferenceTopK() if case == "backend" else None,
        logical_block_size=1664,
        request_distribution=jnp.full((3,), requests, jnp.int32),
    )
    expected = pool.reshape(4, physical_rows, 128)[
        table[req[:, None], columns[None] // 416], columns[None] % 416
    ]
    np.testing.assert_array_equal(actual, expected)
    assert observed == [416 if case in ("decode", "chip") else 32]


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


@pytest.mark.parametrize("namespace", [None, 7])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize(
    "tokens,requests", [(4, 8), (8, 8), (128, 8), (512, 8), (9, 9)]
)
def test_shared_schedule_forwards_backend_and_causal_metadata(
    monkeypatch, pooled, tokens, requests, namespace
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
        assert kwargs["scheduling_group_namespace"] == namespace
        if cu is None:
            assert distribution is None
            cu = jnp.arange(len(q) + 1, dtype=jnp.int32)
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
            assert kwargs["components"] == qsa_indexing()
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
                request_distribution=jnp.array([1, 1, 2], jnp.int32),
                topk_backend=backend,
                scheduling_group_namespace=namespace,
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
                request_distribution=jnp.array([1, 1, 2], jnp.int32),
                topk_backend=backend,
                scheduling_group_namespace=namespace,
            )
    expected = np.full((tokens, 16), -1, np.int32)
    for request, count in enumerate(counts):
        for local in range(count):
            row = int(starts[request]) + local
            n = min(16, (int(context[request]) + local + 1) // 4)
            expected[row, :n] = np.arange(n)
    np.testing.assert_array_equal(got, expected)
    decode_rows = sum(
        np.count_nonzero(lens > 0)
        for lens, cu, kw in seen
        if kw["num_queries_per_block"] == 1
    )
    prefill_rows = sum(
        int(cu[-1]) for lens, cu, kw in seen if kw["num_queries_per_block"] != 1
    )
    if tokens <= requests:
        assert decode_rows == sum(counts)
        assert prefill_rows == 0
    else:
        assert decode_rows == sum(int(n) for n in counts if n == 1)
        assert prefill_rows == sum(int(n) for n in counts if n > 1)


def test_default_is_qwen_sparsecore_and_jit_backend_identity(monkeypatch):
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
    jax.clear_caches()
    args = (
        jnp.zeros((1, 1, 128), jnp.bfloat16),
        jnp.zeros((1, 32, 1, 128), jnp.bfloat16),
        jnp.array([4096], jnp.int32),
        jnp.zeros((1, 1), jnp.int32),
        jnp.zeros((1,), jnp.int32),
    )
    np.testing.assert_array_equal(fn(*args), 1)
    np.testing.assert_array_equal(fn(*args, topk_backend=ReferenceTopK()), 2)
    np.testing.assert_array_equal(fn(*args), 1)
    assert traces == [QwenSparseCoreTopK(), ReferenceTopK()]
    jax.clear_caches()


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
            request_distribution=jnp.array(
                [1 if tokens == 4 else 0, 1 if tokens == 4 else 0, 2], jnp.int32
            ),
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

    from jax._src import xla_bridge
    from torch_tpu._internal import pallas
    from torch_tpu._internal.device._device_module import (
        TpuDeviceModule as device_module,
    )

    initialized = device_module.is_initialized()
    clients = tuple(xla_bridge._backends)
    for name, fn in [
        ("compact_pooled", pooled_qsa.compact_pooled_qsa_step),
    ]:
        for label, core in [
            ("default", fn),
            ("alternate", functools.partial(fn, topk_backend=ReferenceTopK())),
            ("namespace", functools.partial(fn, scheduling_group_namespace=37)),
        ]:
            op = pallas.jax_op(f"qsa_routing_schema::{name}_{label}", core)
            assert "topk_backend" not in str(op._schema)
    assert device_module.is_initialized() == initialized
    assert tuple(xla_bridge._backends) == clients


@pytest.mark.parametrize("tokens", [1, 4])
def test_pooled_step_forwards_geometry_and_backend_before_attention(
    monkeypatch, tokens
):
    backend = ReferenceTopK()
    fixture_rotary_dim = 2
    seen = []

    def select(query, pool, lengths, table, req, starts, context, **kwargs):
        seen.append(
            (
                kwargs["topk_backend"],
                starts,
                context,
                kwargs["vmem_limit_bytes"],
                kwargs["scheduling_group_namespace"],
            )
        )
        return jnp.full((len(query), 16), -1, jnp.int32)

    # Record just the indexer boundary. The native lifecycle test separately
    # executes the real cache I/O and full attention on TPU.
    monkeypatch.setattr(pooled_qsa, "select_blocks", select)
    monkeypatch.setattr(pooled_qsa, "write_main_cache", lambda pool, *a: pool)

    def write_compressed(
        pool,
        *args,
        logical_block_size,
        batch_pairs,
        trig_rows,
        trig_cache,
        positions,
        rotary_dim,
    ):
        assert rotary_dim == fixture_rotary_dim
        assert logical_block_size == 128
        assert not batch_pairs
        assert trig_rows is None and trig_cache is None and positions is None
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
        rotary_dim=fixture_rotary_dim,
        request_distribution=jnp.array(
            [1 if tokens == 1 else 0, 1 if tokens == 1 else 0, 1], jnp.int32
        ),
        topk_backend=backend,
        scheduling_group_namespace=11,
    )
    assert len(seen) == 1
    assert seen[0][0] is backend and seen[0][1] is starts and seen[0][2] is context
    assert seen[0][3] == 16 * 1024**2
    assert seen[0][4] == 11


def test_cached_decode_selector_separates_namespace_and_backend(monkeypatch):
    traces = []

    def record(q, weights, cache, lengths, pages, cu, distribution, **kwargs):
        traces.append((kwargs["scheduling_group_namespace"], kwargs["topk_backend"]))
        return jnp.full((len(q), kwargs["k"]), len(traces), jnp.int32)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", record)
    monkeypatch.setattr(decode_indexer, "decode_query_chunk_size", lambda _: 4)
    fn = decode_indexer.select_decode_blocks
    jax.clear_caches()
    args = (
        jnp.zeros((1, 1, 128), jnp.bfloat16),
        jnp.zeros((1, 32, 1, 128), jnp.bfloat16),
        jnp.array([4], jnp.int32),
        jnp.zeros((1, 1), jnp.int32),
        jnp.zeros((1,), jnp.int32),
    )
    backend = ReferenceTopK()
    for ns, selector, expected in [
        (3, backend, 1),
        (4, backend, 2),
        (3, backend, 1),
        (3, SparseCoreTopK(), 3),
    ]:
        np.testing.assert_array_equal(
            fn(*args, topk_backend=selector, scheduling_group_namespace=ns), expected
        )
    assert traces == [(3, backend), (4, backend), (3, SparseCoreTopK())]
    jax.clear_caches()


def test_production_compact_factory_exposes_compiler_owned_namespace(monkeypatch):
    import torch
    from torch_tpu._internal import pallas

    from vllm_torchtpu.compilation.tpu_compiler import _bind_scheduling_group_namespaces
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    registrations = []
    original = pallas.jax_op

    def register(name, function, **kwargs):
        op = original(name, function, **kwargs)
        registrations.append((op, kwargs))
        return op

    monkeypatch.setattr(pallas, "jax_op", register)
    qsa_op.build_compact_pooled_qsa_op(
        topk_backend=ReferenceTopK(), topk_backend_key="namespace-contract-test-v1"
    )
    op, options = registrations[0]
    assert options["donate_argnums"] == (10, 11)
    schema = op._opoverload._schema
    namespace_index = next(
        i
        for i, argument in enumerate(schema.arguments)
        if argument.name == "scheduling_group_namespace"
    )
    assert schema.arguments[namespace_index].default_value == 0
    graph = torch.fx.Graph()
    args = tuple(
        graph.placeholder(arg.name) for arg in schema.arguments[:namespace_index]
    )
    kwargs = {
        arg.name: graph.placeholder(arg.name)
        for arg in schema.arguments[namespace_index + 1 :]
    }
    first = graph.call_function(op._opoverload, args, kwargs)
    second = graph.call_function(op._opoverload, args, kwargs)
    graph.output((first, second))
    module = torch.fx.GraphModule({}, graph)
    _bind_scheduling_group_namespaces(module)
    assert [
        n.kwargs["scheduling_group_namespace"]
        for n in module.graph.nodes
        if n.op == "call_function"
    ] == [1, 2]
    assert first.kwargs["ring_capacity"] is kwargs["ring_capacity"]
    assert second.kwargs["ring_capacity"] is kwargs["ring_capacity"]


@pytest.mark.parametrize("mode", ["default", "injected", "explicit"])
def test_prefill_prefix_capacity_preserves_functional_visibility(monkeypatch, mode):
    import importlib

    stream = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    widths = []

    def pallas_call(kernel, **kwargs):
        shape = kwargs["out_shape"].shape

        def score(*args):
            values = jnp.broadcast_to(
                jnp.arange(shape[1] * 128, dtype=jnp.float32),
                (shape[0], shape[1] * 128),
            )
            return jax.lax.bitcast_convert_type(values, jnp.int32).reshape(shape)

        return score

    original = ReferenceTopK.select

    def select(self, bits, k, row_lengths, *, scheduling_group_id, return_scores):
        jax.debug.callback(lambda: widths.append(bits.shape[1]), ordered=True)
        return original(
            ReferenceTopK(),
            bits,
            k,
            row_lengths,
            scheduling_group_id=None,
            return_scores=return_scores,
        )

    # Real adapter, metadata, JIT, ranking and visibility; hardware score/SC
    # bodies alone are replaced by monotonic float32 scores and CPU selection.
    monkeypatch.setattr(stream.pl, "pallas_call", pallas_call)
    monkeypatch.setattr(ReferenceTopK, "select", select)
    monkeypatch.setattr(QwenSparseCoreTopK, "select", select)
    jax.clear_caches()
    backend = None if mode == "default" else ReferenceTopK()
    prefix_capacity = 256 if mode == "explicit" else None
    for history in (0, 4096):
        widths.clear()
        ids = decode_indexer.select_index_blocks(
            jnp.ones((1024, 2, 128), jnp.bfloat16),
            jnp.zeros((129, 16, 2, 128), jnp.bfloat16),
            jnp.full((1024,), 1024 + history, jnp.int32),
            jnp.tile(jnp.arange(128, dtype=jnp.int32), (2, 1)),
            jnp.zeros((1024,), jnp.int32),
            starts=jnp.array([0, 1024, 1024], jnp.int32),
            context=jnp.array([history, 0], jnp.int32),
            ratio=4,
            block_topk=128,
            scheduling_group_namespace=5,
            request_distribution=jnp.array([0, 0, 1], jnp.int32),
            topk_backend=backend,
            prefix_capacity=prefix_capacity,
        )
        ids.block_until_ready()
        visible = (history + np.arange(1, 1025)) // 4
        slots = np.arange(128)[None, :]
        expected = np.where(
            slots < np.minimum(visible, 128)[:, None],
            np.maximum(visible - 128, 0)[:, None] + slots,
            -1,
        )
        np.testing.assert_array_equal(np.sort(ids, axis=1), np.sort(expected, axis=1))
        np.testing.assert_array_equal(np.asarray(ids) >= 0, expected >= 0)
        assert widths == [1024 if history == 0 and mode != "injected" else 4096] * 2


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.uint8, jnp.float8_e4m3fn])
@pytest.mark.parametrize("page_size", [32, 416])
@pytest.mark.parametrize("decode_only", [False, True])
def test_ragged_prefill_view_preserves_carrier_and_page_addresses(
    monkeypatch, dtype, page_size, decode_only
):
    """Both traced branches retain every byte and physical page mapping."""
    from jax.experimental.pallas import tpu as pltpu

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    packing = 2 if dtype == jnp.bfloat16 else 4
    storage = jnp.arange(4 * (page_size // 2) * packing * 128) % 113
    cache = storage.astype(jnp.bfloat16 if packing == 2 else jnp.uint8)
    if dtype == jnp.float8_e4m3fn:
        cache = jax.lax.bitcast_convert_type(cache, dtype)
    cache = cache.reshape(4, page_size // 2, packing, 128)
    table = jnp.array([[3, 1], [2, 0]], jnp.int32)
    starts = jnp.array([0, 1, 2] if decode_only else [0, 8, 16], jnp.int32)
    distribution = jnp.array([2, 2, 2] if decode_only else [0, 0, 2], jnp.int32)
    prefill_calls = []

    def record(q, weights, read_cache, lengths, pages, cu, dist, **kwargs):
        kwargs["components"].keys.validate(read_cache, 128, KVLayout.HEAD_ALONG_SUBLANE)
        if kwargs["num_queries_per_block"] != 1 and not isinstance(
            pages, jax.core.Tracer
        ):
            prefill_calls.append(1)
            actual = read_cache[pages.reshape(2, -1)].reshape(2, -1)
            expected = cache[table].reshape(2, -1)
            if dtype == jnp.float8_e4m3fn:
                actual = jax.lax.bitcast_convert_type(actual, jnp.uint8)
                expected = jax.lax.bitcast_convert_type(expected, jnp.uint8)
            np.testing.assert_array_equal(actual, expected)
            assert read_cache.shape[1] * 2 == 32
        return jnp.full((len(q), 16), -1, jnp.int32)

    monkeypatch.setattr(decode_indexer, "streamindex_topk", record)

    def run(query):
        return decode_indexer.select_index_blocks(
            query,
            cache,
            jnp.full((16,), 96, jnp.int32),
            table,
            jnp.arange(16, dtype=jnp.int32) % 2,
            starts=starts,
            # Exceed the 16-group budget so the reader is actually exercised.
            context=jnp.full((2,), 80, jnp.int32),
            request_distribution=distribution,
            ratio=4,
            block_topk=16,
        )

    query = jnp.zeros((16, 4, 128), jnp.bfloat16)
    # lax.cond must trace the prefill branch even for an all-decode batch.
    assert jax.eval_shape(run, query).shape == (16, 16)
    with jax.disable_jit():
        np.testing.assert_array_equal(run(query), -np.ones((16, 16), np.int32))
    assert bool(prefill_calls) == (not decode_only)


@pytest.mark.parametrize("qwen", [False, True])
@pytest.mark.parametrize("return_scores", [False, True])
def test_resident_policy_is_local_to_qwen(monkeypatch, qwen, return_scores):
    import importlib

    backend_module = importlib.import_module(SparseCoreTopK.__module__)
    qwen_module = importlib.import_module(QwenSparseCoreTopK.__module__)
    calls = []

    def shared_selector(scores, k, **kwargs):
        calls.append((False, kwargs))
        return scores

    def qwen_selector(scores, k, **kwargs):
        calls.append((True, kwargs))
        return scores

    monkeypatch.setattr(backend_module, "sparsecore_topk", shared_selector)
    monkeypatch.setattr(qwen_module, "qwen_sparsecore_topk", qwen_selector)
    backend = QwenSparseCoreTopK() if qwen else SparseCoreTopK()
    scores = jnp.zeros((3, 16), jnp.int32)
    lengths = jnp.array([0, 7, 16], jnp.int32)
    assert (
        backend.select(
            scores, 16, lengths, scheduling_group_id=7, return_scores=return_scores
        )
        is scores
    )
    assert calls[0][0] is qwen
    assert "try_resident_prefix" not in calls[0][1]
    assert calls[0][1]["row_lengths"] is lengths
    assert calls[0][1]["write_empty_rows"] is True
    assert calls[0][1]["scheduling_group_id"] == 7
    assert calls[0][1]["return_scores"] is return_scores
    assert backend.overlaps_scoring is True
