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
"""Scheduling identities must belong to compiled call sites, not JIT traces."""

import functools
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.fx as fx


@torch.library.custom_op("streamindex_scheduling_test::identity", mutates_args=())
def identity(x: torch.Tensor, scheduling_group_namespace: int = 0) -> torch.Tensor:
    return x.clone()


@identity.register_fake
def _fake_identity(x, scheduling_group_namespace=0):
    return torch.empty_like(x)


def _graph():
    graph = fx.Graph()
    value = graph.placeholder("x")
    value = graph.call_function(identity._opoverload, (value,))
    value = graph.call_function(identity._opoverload, (value, 99))
    value = graph.call_function(
        identity._opoverload, (value,), {"scheduling_group_namespace": 99}
    )
    graph.output(value)
    return fx.GraphModule(torch.nn.Module(), graph)


def _namespaces(module):
    return [
        node.args[1]
        if len(node.args) > 1
        else node.kwargs["scheduling_group_namespace"]
        for node in module.graph.nodes
        if node.op == "call_function" and node.target == identity._opoverload
    ]


def test_call_sites_bind_defaults_positional_and_keywords():
    from vllm_torchtpu.compilation.tpu_compiler import _bind_scheduling_group_namespaces

    first, second = _graph(), _graph()
    for graph in (second, first, second):
        _bind_scheduling_group_namespaces(graph)
        assert _namespaces(graph) == [1, 2, 3]
    assert first.code == second.code
    torch.testing.assert_close(first(torch.ones(2)), torch.ones(2))


def test_repeated_nested_graph_owns_distinct_namespaces_without_weight_copy():
    from vllm_torchtpu.compilation.tpu_compiler import _bind_scheduling_group_namespaces

    child = _graph()
    child.register_parameter("bias", torch.nn.Parameter(torch.ones(2)))
    first_op = next(node for node in child.graph.nodes if node.op == "call_function")
    with child.graph.inserting_before(first_op):
        bias = child.graph.get_attr("bias")
        value = child.graph.call_function(torch.add, (first_op.args[0], bias))
    first_op.args = (value,)
    child.recompile()
    root = torch.nn.Module()
    root.child = child
    graph = fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_module("child", (x,))
    second = graph.call_module("child", (first,))
    graph.output(second)
    module = fx.GraphModule(root, graph)
    _bind_scheduling_group_namespaces(module)
    calls = [node for node in module.graph.nodes if node.op == "call_module"]
    assert calls[0].target != calls[1].target
    assert _namespaces(module.get_submodule(calls[0].target)) == [1, 2, 3]
    assert _namespaces(module.get_submodule(calls[1].target)) == [4, 5, 6]
    assert module.get_submodule(calls[0].target).bias is child.bias
    assert module.get_submodule(calls[1].target).bias is child.bias
    code = module.code
    _bind_scheduling_group_namespaces(module)
    assert module.code == code
    torch.testing.assert_close(module(torch.ones(2)), torch.full((2,), 3.0))


def test_load_binds_the_same_call_sites_before_replay(monkeypatch):
    from vllm.config.utils import Range

    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    compiler = TpuCompilerAdaptor()
    graph = _graph()
    original = graph.code
    seen = []

    def replay(path, prepared, was_wrapped):
        seen.append(_namespaces(prepared))
        return lambda *args: args

    monkeypatch.setattr(compiler, "_replay", replay)
    compiler.load(("key", "unused", False), graph, [], 0, Range(1, 1))
    assert seen == [[1, 2, 3]]
    assert graph.code == original


def test_pairing_is_disjoint_for_different_chunk_counts():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import (
        scoped_scheduling_group_id,
    )

    groups = [
        {scoped_scheduling_group_id(namespace, pair) for pair in range(count)}
        for namespace, count in enumerate((1, 15, 2, 511, 3, 127), 1)
    ]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert all(value > 0 and value % 2 == 0 for group in groups for value in group)
    for bad in ((0, 0), (-1, 0), (1, -1)):
        with pytest.raises(ValueError):
            scoped_scheduling_group_id(*bad)


def test_changed_call_site_binding_invalidates_exported_runtime_cache(
    monkeypatch, tmp_path
):
    from tests.compilation.test_tpu_compiler import TestTpuCompilerCache
    from vllm_torchtpu.compilation import tpu_compiler

    source = tmp_path / "src/vllm_torchtpu/compilation/tpu_compiler.py"
    source.parent.mkdir(parents=True)
    source.write_text("# first call-site binding\n")
    monkeypatch.setattr(tpu_compiler, "__file__", str(source))
    config = TestTpuCompilerCache._config()
    first = tpu_compiler.compute_tpu_compilation_hash(config)
    source.write_text("# changed call-site binding\n")
    assert tpu_compiler.compute_tpu_compilation_hash(config) != first


def fixture_inputs(tokens):
    """Strictly ordered BF16 keys, fragmented pages, two live and six idle slots."""
    pages = 80
    rng = np.random.default_rng(42)
    tables = np.zeros((8, pages), np.int32)
    tables[:2] = rng.permutation(np.arange(1, 2 * pages + 1)).reshape(2, pages)
    values = (np.arange(pages * 128, dtype=np.uint16) + 0x2000).copy()
    cache_words = np.zeros((2 * pages + 1, 128, 256), np.uint16)
    cache_words[:, :, 128] = 0x3F80
    cache_words[0] = 0x3E80
    for request in range(2):
        for logical, physical in enumerate(tables[request]):
            cache_words[physical, :, 0] = values[logical * 128 : (logical + 1) * 128]
    cache = torch.from_numpy(cache_words.copy()).view(torch.bfloat16)
    cache = cache.reshape(2 * pages + 1, 128, 1, 256)
    query = torch.zeros((tokens, 4, 128), dtype=torch.bfloat16)
    query[..., 0] = 1
    counts = np.array([tokens // 2, tokens - tokens // 2] + [0] * 6, np.int32)
    starts = np.r_[0, counts.cumsum()].astype(np.int32)
    context = np.array([4096, 2048] + [0] * 6, np.int32)
    requests = np.repeat(np.arange(8), counts).astype(np.int32)
    lengths = np.concatenate(
        [
            context[request] + np.arange(1, count + 1)
            for request, count in enumerate(counts)
        ]
    ).astype(np.int32)
    return tuple(
        [query, cache]
        + [
            torch.from_numpy(x.copy())
            for x in (lengths, tables, requests, starts, context)
        ]
    )


def check_selected(actual, lengths):
    """Positive query/key products give strict ordering by logical key ID."""
    actual = np.asarray(actual)
    assert actual.shape == (len(lengths), 512)
    visible = np.asarray(lengths)
    assert np.all(visible > 512)
    expected = visible[:, None] - 512 + np.arange(512)[None, :]
    np.testing.assert_array_equal(np.sort(actual, axis=1), expected)


@functools.cache
def native_selector(annotated=True):
    """Real selector with the same compiler-owned schema contract as QSA."""
    import inspect

    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas

    from tests.kernels.test_streamindex_topk_backend import (
        DenseKeyReader,
        PrefixSchedule,
    )
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
        StreamIndexComponents,
    )
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
        SparseCoreTopK,
    )

    @dataclass(frozen=True)
    class IdentitySchedule(PrefixSchedule):
        def positions(self, positions):
            return positions

        def visible_lengths(self, lengths):
            return lengths

    components = StreamIndexComponents(
        DenseKeyReader(scale_gain=1.0), IdentitySchedule()
    )
    # This lets the same native test reproduce the old cached-body collision;
    # old source did not expose the namespace keyword at the StreamIndex API.
    namespace_supported = (
        "scheduling_group_namespace" in inspect.signature(streamindex_topk).parameters
    )

    @dataclass(frozen=True)
    class Unannotated(SparseCoreTopK):
        overlaps_scoring = False

    backend = None if annotated else Unannotated()

    def core(
        query: jax.Array,
        cache: jax.Array,
        lengths: jax.Array,
        table: jax.Array,
        requests: jax.Array,
        starts: jax.Array,
        context: jax.Array,
        scheduling_group_namespace: int = 0,
    ) -> jax.Array:
        options = (
            {"scheduling_group_namespace": scheduling_group_namespace}
            if namespace_supported
            else {}
        )
        return streamindex_topk(
            query,
            jnp.ones(query.shape[:2], jnp.float32),
            cache,
            context + jnp.diff(starts),
            table.reshape(-1),
            starts,
            jnp.array([0, 0, 2], jnp.int32),
            k=512,
            compression_ratio=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=(1, 128, 128),
            chunk_tokens=512 if len(query) % 512 == 0 else None,
            components=components,
            topk_backend=SparseCoreTopK() if backend is None else backend,
            **options,
        )

    op = pallas.jax_op(f"streamindex_scheduling_test::select_{annotated}", core)

    @op.register_fake
    def fake(
        query,
        cache,
        lengths,
        table,
        requests,
        starts,
        context,
        scheduling_group_namespace=0,
    ):
        return query.new_empty((len(query), 512), dtype=torch.int32)

    return op


def selector_chain(op, layers=2):
    def chain(query, cache, lengths, table, requests, starts, context):
        for layer in range(layers):
            selected = op(query, cache, lengths, table, requests, starts, context)
            if layer + 1 < layers:
                query = query + (
                    (selected.sum(dim=1) % 7).to(query.dtype)[:, None, None] / 1024
                )
        return selected

    return chain


@pytest.mark.parametrize("annotated", [True, False])
def test_native_registration_fake_and_compiler_binding_without_device(annotated):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.compilation.tpu_compiler import _bind_scheduling_group_namespaces

    op = native_selector(annotated)
    seen = []

    def backend(graph, args):
        _bind_scheduling_group_namespaces(graph)
        seen.extend(
            node.kwargs["scheduling_group_namespace"]
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target == op._opoverload
        )
        return graph.forward

    with FakeTensorMode() as mode:
        args = tuple(mode.from_tensor(x) for x in fixture_inputs(1024))
        answer = torch.compile(
            selector_chain(op, 3), backend=backend, fullgraph=True, dynamic=False
        )(*args)
        assert answer.shape == (1024, 512) and answer.dtype == torch.int32
    assert seen == [1, 2, 3]


@pytest.mark.tpu_test
def test_cached_native_calls_across_buckets_and_resets():
    from vllm.config.utils import Range

    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

    op = native_selector()
    compiler = TpuCompilerAdaptor()

    def backend(graph, args):
        result, _ = compiler.compile(graph, args, {}, Range(1, 1))
        assert result is not None
        return result

    for tokens in (128, 1024, 1536, 1024, 128, 1536):
        torch._dynamo.reset()
        cpu = fixture_inputs(tokens)
        args = tuple(x.to("tpu") for x in cpu)
        fn = torch.compile(
            selector_chain(op, 3), backend=backend, fullgraph=True, dynamic=False
        )
        result = fn(*args).cpu().numpy()
        check_selected(result, cpu[2].numpy())
        config.reset_scheduling_group_ids()
        np.testing.assert_array_equal(fn(*args).cpu().numpy(), result)
        torch.testing.assert_close(
            args[1].cpu().view(torch.uint16), cpu[1].view(torch.uint16), rtol=0, atol=0
        )
