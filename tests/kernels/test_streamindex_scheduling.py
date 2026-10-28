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
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest
import torch
import torch.fx as fx

from tests.kernels.streamindex_scheduling_helpers import fixture_inputs


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


def test_prebound_outer_aliases_keep_inner_namespaces_independent():
    from vllm_torchtpu.compilation.scheduling_groups import (
        _bind_scheduling_group_namespaces,
    )

    inner = _graph()
    inner.register_parameter("bias", torch.nn.Parameter(torch.ones(2)))
    first = next(node for node in inner.graph.nodes if node.op == "call_function")
    with inner.graph.inserting_before(first):
        bias = inner.graph.get_attr("bias")
        value = inner.graph.call_function(torch.add, (first.args[0], bias))
    first.args = (value,)
    inner.recompile()
    root = torch.nn.Module()
    root.inner = inner
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_module("inner", (x,)))
    outer = fx.GraphModule(root, graph)
    _bind_scheduling_group_namespaces(outer)

    root = torch.nn.Module()
    root.outer = outer
    graph = fx.Graph()
    x = graph.placeholder("x")
    x = graph.call_module("outer", (x,))
    graph.output(graph.call_module("outer", (x,)))
    module = fx.GraphModule(root, graph)
    _bind_scheduling_group_namespaces(module)
    outers = [
        cast(fx.GraphModule, module.get_submodule(str(node.target)))
        for node in module.graph.nodes
        if node.op == "call_module"
    ]
    inners = [
        [
            cast(fx.GraphModule, child.get_submodule(str(node.target)))
            for node in child.graph.nodes
            if node.op == "call_module"
        ]
        for child in outers
    ]
    assert [[_namespaces(child) for child in group] for group in inners] == [
        [[1, 2, 3]],
        [[4, 5, 6]],
    ]
    assert {id(child) for child in inners[0]}.isdisjoint(
        id(child) for child in inners[1]
    )
    assert all(child.bias is inner.bias for group in inners for child in group)
    torch.testing.assert_close(module(torch.ones(2)), torch.full((2,), 3.0))


def test_nested_aliases_share_tensors_and_rebind_without_new_modules():
    from vllm_torchtpu.compilation.scheduling_groups import (
        _bind_scheduling_group_namespaces,
    )

    leaf = _graph()
    leaf.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
    leaf.register_buffer("bias", torch.ones(2))
    first = next(node for node in leaf.graph.nodes if node.op == "call_function")
    with leaf.graph.inserting_before(first):
        weight = leaf.graph.get_attr("weight")
        bias = leaf.graph.get_attr("bias")
        value = leaf.graph.call_function(torch.add, (first.args[0], weight))
        value = leaf.graph.call_function(torch.add, (value, bias))
    first.args = (value,)
    leaf.recompile()

    def repeated(child):
        root = torch.nn.Module()
        root.left = child
        root.right = child
        graph = fx.Graph()
        x = graph.placeholder("x")
        x = graph.call_module("left", (x,))
        x = graph.call_module("right", (x,))
        graph.output(x)
        return fx.GraphModule(root, graph)

    outer = repeated(leaf)
    module = repeated(outer)
    _bind_scheduling_group_namespaces(module)
    bodies = [
        cast(fx.GraphModule, child._modules[str(inner.target)])
        for node in module.graph.nodes
        if node.op == "call_module"
        for child in (cast(fx.GraphModule, module._modules[str(node.target)]),)
        for inner in child.graph.nodes
        if inner.op == "call_module"
    ]
    assert [_namespaces(child) for child in bodies] == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    assert len({id(child) for child in bodies}) == 4
    assert module._modules["left"] is outer
    assert bodies[0] is leaf
    assert all(
        child.weight is leaf.weight and child.bias is leaf.bias for child in bodies
    )
    before = [
        (id(child), tuple(child._modules), child.code) for child in module.modules()
    ]
    _bind_scheduling_group_namespaces(module)
    assert [
        (id(child), tuple(child._modules), child.code) for child in module.modules()
    ] == before
    torch.testing.assert_close(module(torch.ones(2)), torch.full((2,), 9.0))


@pytest.mark.parametrize("kind", ["module", "parameter", "buffer", "attribute"])
def test_clone_names_preserve_existing_attributes(kind):
    from vllm_torchtpu.compilation.scheduling_groups import (
        _bind_scheduling_group_namespaces,
    )

    root = torch.nn.Module()
    root.child = _graph()
    graph = fx.Graph()
    x = graph.placeholder("x")
    x = graph.call_module("child", (x,))
    x = graph.call_module("child", (x,))
    graph.output(x)
    module = fx.GraphModule(root, graph)
    collisions = {}
    for name in ("_scheduling_child_1", "_scheduling_child_2"):
        if kind == "module":
            value = torch.nn.Identity()
            module.add_module(name, value)
            registry = module._modules
        elif kind == "parameter":
            value = torch.nn.Parameter(torch.ones(2))
            module.register_parameter(name, value)
            registry = module._parameters
        elif kind == "buffer":
            value = torch.ones(2)
            module.register_buffer(name, value)
            registry = module._buffers
        else:
            value = object()
            module.__dict__[name] = value
            registry = module.__dict__
        collisions[name] = value
    _bind_scheduling_group_namespaces(module)
    assert all(registry[name] is value for name, value in collisions.items())
    calls = [node for node in module.graph.nodes if node.op == "call_module"]
    assert calls[0].target == "child"
    assert calls[1].target not in collisions
    assert _namespaces(module._modules[str(calls[1].target)]) == [4, 5, 6]
    torch.testing.assert_close(module(torch.ones(2)), torch.ones(2))


def test_dotted_module_aliases_receive_distinct_namespaces():
    from vllm_torchtpu.compilation.scheduling_groups import (
        _bind_scheduling_group_namespaces,
    )

    root = torch.nn.Module()
    holder = torch.nn.Module()
    child = _graph()
    holder.add_module("left", child)
    holder.add_module("right", child)
    root.add_module("holder", holder)
    graph = fx.Graph()
    x = graph.placeholder("x")
    x = graph.call_module("holder.left", (x,))
    x = graph.call_module("holder.right", (x,))
    graph.output(x)
    module = fx.GraphModule(root, graph)
    _bind_scheduling_group_namespaces(module)
    assert _namespaces(cast(fx.GraphModule, module.get_submodule("holder.left"))) == [
        1,
        2,
        3,
    ]
    second = next(node for node in module.graph.nodes if node.name == "holder_right")
    assert _namespaces(module._modules[str(second.target)]) == [4, 5, 6]


def test_export_round_trip_preserves_bound_call_sites():
    from vllm_torchtpu.compilation.scheduling_groups import (
        _bind_scheduling_group_namespaces,
    )

    program = torch.export.export(_graph(), (torch.ones(2),))
    _bind_scheduling_group_namespaces(program.graph_module)
    stream = io.BytesIO()
    torch.export.save(program, stream)
    stream.seek(0)
    restored = torch.export.load(stream)
    assert _namespaces(restored.graph_module) == [1, 2, 3]
    _bind_scheduling_group_namespaces(restored.graph_module)
    assert _namespaces(restored.graph_module) == [1, 2, 3]
    torch.testing.assert_close(restored.module()(torch.ones(2)), torch.ones(2))


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


@pytest.mark.parametrize("edited", ["binder", "compiler"])
def test_changed_call_site_binding_invalidates_exported_runtime_cache(
    monkeypatch, tmp_path, edited
):
    from vllm.config import VllmConfig

    from tests.compilation.test_tpu_compiler import TestTpuCompilerCache
    from vllm_torchtpu.compilation import tpu_compiler

    compiler = tmp_path / "src/vllm_torchtpu/compilation/tpu_compiler.py"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("# compiler implementation\n")
    binder = compiler.with_name("scheduling_groups.py")
    binder.write_text("# first call-site binding\n")
    monkeypatch.setattr(tpu_compiler, "__file__", str(compiler))
    config = cast(VllmConfig, TestTpuCompilerCache._config())
    first = tpu_compiler.compute_tpu_compilation_hash(config)
    if edited == "compiler":
        compiler.write_text("# unrelated compiler edit\n")
        assert tpu_compiler.compute_tpu_compilation_hash(config) == first
    else:
        binder.write_text("# changed call-site binding\n")
        assert tpu_compiler.compute_tpu_compilation_hash(config) != first


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


def selector_chain(
    op: Callable[..., torch.Tensor], layers: int = 2
) -> Callable[..., torch.Tensor]:
    def chain(query, cache, lengths, table, requests, starts, context):
        selected: torch.Tensor | None = None
        for layer in range(layers):
            selected = op(query, cache, lengths, table, requests, starts, context)
            if layer + 1 < layers:
                query = query + (
                    (selected.sum(dim=1) % 7).to(query.dtype)[:, None, None] / 1024
                )
        assert selected is not None
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
