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
"""StreamIndex namespace specialization through TorchTPU's JAX export path."""

import hashlib
import json
import re
import time
from dataclasses import dataclass

import pytest
import torch
import torch.fx as fx

from tests.kernels.streamindex_scheduling_helpers import fixture_inputs


def register_selector(name, traces, *, chunk_tokens=512):
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
    backend = SparseCoreTopK()

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
        traces.append(scheduling_group_namespace)
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
            chunk_tokens=chunk_tokens,
            components=components,
            topk_backend=backend,
            scheduling_group_namespace=scheduling_group_namespace,
        )

    # Keep the native default fake implementation: it performs real JAX export.
    return pallas.jax_op(name, core)


def selector_graph(op, callsites):
    graph = fx.Graph()
    args = tuple(
        graph.placeholder(name)
        for name in (
            "query",
            "cache",
            "lengths",
            "table",
            "requests",
            "starts",
            "context",
        )
    )
    calls = tuple(graph.call_function(op._opoverload, args) for _ in range(callsites))
    graph.output(calls)
    return fx.GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize("callsites,heads", [(1, 4), (2, 4), (12, 4), (21, 64)])
@pytest.mark.parametrize("namespace_mode", ["same", "distinct", "zero"])
def test_native_export_specializes_only_for_distinct_namespaces(
    callsites, heads, namespace_mode, monkeypatch, record_property, tmp_path
):
    import jax
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch_tpu._internal.pallas.pallas import JaxCallable

    from vllm_torchtpu.compilation.tpu_compiler import _bind_scheduling_group_namespaces

    traces, exports, callables, lowerings = [], [], [], []
    original_init = JaxCallable.__init__
    original_lower = jax.stages.Traced.lower

    def observe_lower(instance, *args, **kwargs):
        start = time.perf_counter()
        result = original_lower(instance, *args, **kwargs)
        lowerings.append(time.perf_counter() - start)
        return result

    monkeypatch.setattr(jax.stages.Traced, "lower", observe_lower)

    def observe_init(instance, *args, **kwargs):
        original_init(instance, *args, **kwargs)
        callables.append(instance)
        original_export = instance.exported

        def observe_export(*args, **kwargs):
            start = time.perf_counter()
            result = original_export(*args, **kwargs)
            elapsed = time.perf_counter() - start
            module = result.mlir_module()
            exports.append((elapsed, module))
            return result

        monkeypatch.setattr(instance, "exported", observe_export)

    monkeypatch.setattr(JaxCallable, "__init__", observe_init)
    op = register_selector(
        f"streamindex_scheduling_cost::select_{callsites}_{heads}_{namespace_mode}",
        traces,
    )
    module = selector_graph(op, callsites)
    start = time.perf_counter()
    _bind_scheduling_group_namespaces(module)
    binder_s = time.perf_counter() - start
    nodes = [node for node in module.graph.nodes if node.op == "call_function"]
    assert [node.kwargs["scheduling_group_namespace"] for node in nodes] == list(
        range(1, callsites + 1)
    )
    if namespace_mode != "distinct":
        namespace = 0 if namespace_mode == "zero" else 1
        for node in nodes:
            node.kwargs = {"scheduling_group_namespace": namespace}
        module.recompile()

    inputs = list(fixture_inputs(1024))
    inputs[0] = inputs[0][:, :1].expand(-1, heads, -1).contiguous()
    target = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    with jax.sharding.use_abstract_mesh(target), FakeTensorMode() as mode:
        args = tuple(mode.from_tensor(value) for value in inputs)
        start = time.perf_counter()
        first = module(*args)
        first_s = time.perf_counter() - start
        first_traces = list(traces)
        first_exports = len(exports)
        start = time.perf_counter()
        second = module(*args)
        repeat_s = time.perf_counter() - start
    expected = (
        list(range(1, callsites + 1))
        if namespace_mode == "distinct"
        else [0 if namespace_mode == "zero" else 1]
    )
    assert first_traces == expected
    assert traces == first_traces
    assert len(lowerings) == len(exports) == 2 * callsites
    assert len(first) == len(second) == callsites
    assert all(value.shape == (1024, 512) for value in first + second)
    assert len(callables) == 1
    unique_modules = {
        hashlib.sha256(text.encode()).hexdigest(): text for _, text in exports
    }
    assert len(unique_modules) == len(expected)
    body_counts = []
    for digest, text in unique_modules.items():
        (tmp_path / f"{digest}.mlir").write_text(text)
        body_counts.append(
            len(re.findall(r"func.func private @streamindex_topk(?:\(|_)", text))
        )
    assert body_counts == [1] * len(expected)
    metrics = {
        "callsites": callsites,
        "heads": heads,
        "tokens": 1024,
        "namespace_mode": namespace_mode,
        "backend": "SparseCoreTopK",
        "binder_s": binder_s,
        "first_export_graph_s": first_s,
        "repeat_export_graph_s": repeat_s,
        "jax_core_traces": len(traces),
        "first_export_invocations": first_exports,
        "repeat_export_invocations": len(exports) - first_exports,
        "export_trace_lower_s": sum(elapsed for elapsed, _ in exports),
        "jax_lower_invocations": len(lowerings),
        "jax_lower_s": sum(lowerings),
        "unique_export_modules": len(unique_modules),
        "streamindex_bodies_per_unique_module": body_counts,
        "native_registered_executable_cache_entries": len(callables[0].output_shapes),
        "target": "TPU7x abstract mesh; CPU host export, no XLA compile",
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    record_property("metrics", json.dumps(metrics))
