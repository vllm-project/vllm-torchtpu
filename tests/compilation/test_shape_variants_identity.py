# SPDX-License-Identifier: Apache-2.0
"""Bucket hand-over checks module-state identity, not only shapes.

t128: two traces of the backbone took the HC down weight and the kept up
transpose (both bf16 [320, 10240]) in opposite orders; the shape signature
matched, so the hand-over would have bound the two exchanged. Placeholders
lifted from module state carry their path in their name; a different order
of state now refuses (or warns, with VLLM_TPU_SHAPE_VARIANT_IDENTITY_WARN=1).
"""

import pytest
import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.compilation import shape_variants

KEY = "artifact_compile_range_64_64_subgraph_1"
DOWN = "l_self_modules_model_modules_layers_modules_3_modules_hc_modules_down_parameters_weight_"
UP_T = "l_self_modules_model_modules_layers_modules_3_modules_hc_buffers__up_t_"


def _graph(order: tuple[str, ...]) -> fx.GraphModule:
    """x @ a.T @ b with placeholders ``order``; a and b are same-shaped state."""
    mode = FakeTensorMode()
    meta = {
        "x": mode.from_tensor(torch.zeros(4, 8)),
        "a": mode.from_tensor(torch.zeros(8, 8)),
        "b": mode.from_tensor(torch.zeros(8, 8)),
        "getitem_3": mode.from_tensor(torch.zeros(4, 8)),
    }
    graph = fx.Graph()
    nodes = {}
    for name in order:
        target = {"a": DOWN, "b": UP_T}.get(name, name)
        nodes[name] = graph.placeholder(target)
        nodes[name].meta["example_value"] = meta[name]
    h = graph.call_function(torch.matmul, (nodes["x"], nodes["a"]))
    graph.output(graph.call_function(torch.matmul, (h, nodes["b"])))
    return fx.GraphModule(torch.nn.Module(), graph)


def _handed(first: fx.GraphModule, second: fx.GraphModule):
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, first, first)
    return session.get(None, KEY, second)


def test_same_state_order_is_handed_over():
    assert _handed(_graph(("x", "a", "b")), _graph(("x", "a", "b"))) is not None


def test_same_shaped_state_swapped_is_refused():
    with pytest.raises(shape_variants.ShapeSpecializationError, match="_up_t"):
        _handed(_graph(("x", "a", "b")), _graph(("x", "b", "a")))


def test_swap_warns_with_the_fallback_env(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_SHAPE_VARIANT_IDENTITY_WARN", "1")
    assert _handed(_graph(("x", "a", "b")), _graph(("x", "b", "a"))) is not None


def test_layer_index_is_not_identity():
    # One compiled range serves many layers: layer 3's and layer 7's same
    # weight are the same state for the hand-over.
    first = _graph(("x", "a", "b"))
    second = _graph(("x", "a", "b"))
    for node in second.graph.nodes:
        if node.op == "placeholder":
            node.target = node.target.replace("_modules_3_", "_modules_7_")
    assert _handed(first, second) is not None


def test_activations_match_by_shape_only():
    # Trace-local activation names differ between traces.
    first = _graph(("x", "a", "b"))
    second = _graph(("x", "a", "b"))
    for node in second.graph.nodes:
        if node.target == "x":
            node.target = "getitem_12"
    assert _handed(first, second) is not None


def test_replay_signature_carries_the_state_order():
    assert shape_variants.graph_signature(
        _graph(("x", "a", "b"))
    ) != shape_variants.graph_signature(_graph(("x", "b", "a")))
