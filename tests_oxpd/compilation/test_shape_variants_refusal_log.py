# SPDX-License-Identifier: Apache-2.0
"""A refused hand-over logs both traces' SymInt and tensor sources before raising.

T37 / T38 (2026-11-11) refused on one SymInt placeholder at a different
position; the error text carries only (dtype, shape) signatures. The REFUSED
line names each SymInt's source (None = no single sourced root, which is what
stops the permutation), so the next refusal localizes itself.
"""

import logging

import pytest
import torch
from torch import fx
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_torchtpu.compilation import shape_variants

KEY = "artifact_compile_range_64_64_subgraph_0"


def _graph(order, n_source):
    env = ShapeEnv()
    mode = FakeTensorMode(shape_env=env)
    root = env.create_symbol(4, TensorPropertySource(LocalSource(n_source), TensorProperty.SIZE, 0))
    meta = {
        "x": mode.from_tensor(torch.zeros(4, 8)),
        "y": mode.from_tensor(torch.zeros(4, 8)),
        "n": env.create_symintnode(root, hint=4),
    }
    graph = fx.Graph()
    nodes = {}
    for name in order:
        nodes[name] = graph.placeholder(name)
        nodes[name].meta["example_value"] = meta[name]
    graph.output(graph.call_function(torch.add, (graph.call_function(torch.mul, (nodes["x"], nodes["n"])), nodes["y"])))
    return fx.GraphModule(torch.nn.Module(), graph)


@pytest.fixture(autouse=True)
def _capture(caplog):
    name = shape_variants.logger.name  # resolve the lazy logger first
    lg = logging.getLogger(name)
    saved = (lg.level, lg.propagate)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.addHandler(caplog.handler)
    try:
        yield
    finally:
        lg.removeHandler(caplog.handler)
        lg.setLevel(saved[0])
        lg.propagate = saved[1]


def test_a_refusal_logs_both_traces_symint_sources(caplog):
    session = shape_variants.BucketExecutables()
    first = _graph(("x", "n", "y"), "input_ids")
    session.put(None, KEY, first, first)
    with pytest.raises(shape_variants.ShapeSpecializationError, match="inputs differ"):
        session.get(None, KEY, _graph(("x", "y", "n"), "positions"))
    lines = [r.getMessage() for r in caplog.records if "[shape-variants] REFUSED" in r.getMessage()]
    assert len(lines) == 1, lines
    assert KEY in lines[0] and "input_ids" in lines[0] and "positions" in lines[0], lines[0]


def test_a_permuted_hand_over_logs_no_refusal(caplog):
    session = shape_variants.BucketExecutables()
    first = _graph(("x", "n", "y"), "input_ids")
    session.put(None, KEY, first, first)
    assert session.get(None, KEY, _graph(("x", "y", "n"), "input_ids")) is not None
    assert not [r for r in caplog.records if "REFUSED" in r.getMessage()]
