# SPDX-License-Identifier: Apache-2.0
"""Same-shape AMBIGUITY count on bucket hand-over (diagnostics only).

The source-order check verifies sourced inputs by name. A positional mix-up
can only exchange inputs of one (dtype, shape), and a lone unsourced input in
such a group is forced into its place, so the unverified inputs are exactly
the unsourced ones in a group with two or more unsourced members (for example
two split-boundary activations of one shape). The check counts and logs them.
"""

import logging

import pytest
import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.compilation import shape_variants

KEY = "artifact_compile_range_16_16_subgraph_0"
Q = "l_self_modules_norm_query_parameters_weight_"
SHAPES = {"x": (4, 8), Q: (8,), "add_1": (8,), "add_2": (8,), "mul_3": (8,)}


def _graph(order: tuple[str, ...]) -> fx.GraphModule:
    """Sum of x and every [8] input, with placeholders in ``order``."""
    mode = FakeTensorMode()
    graph = fx.Graph()
    nodes = {}
    for name in order:
        nodes[name] = graph.placeholder(name)
        nodes[name].meta["example_value"] = mode.from_tensor(torch.zeros(SHAPES[name]))
    out = nodes["x"]
    for name in order:
        if name != "x":
            out = graph.call_function(torch.add, (out, nodes[name]))
    graph.output(out)
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


def _hand(order):
    session = shape_variants.BucketExecutables()
    first = _graph(order)
    session.put(None, KEY, first, first)
    handed = session.get(None, KEY, _graph(order))
    return session, handed, first


def _lines(caplog, tag):
    return [r.getMessage() for r in caplog.records if tag in r.getMessage()]


def test_two_unsourced_same_shape_inputs_are_ambiguous(caplog):
    session, handed, first = _hand(("x", "add_1", "add_2"))
    assert handed is first  # behaviour unchanged
    lines = _lines(caplog, "SOURCE-ORDER AMBIGUOUS")
    assert len(lines) == 1 and KEY in lines[0] and "2 unsourced input(s)" in lines[0], lines
    assert (session.source_ambiguous, session.source_ambiguous_inputs) == (1, 2)
    assert session.source_shared_inputs == 2


def test_a_lone_unsourced_input_beside_a_sourced_one_is_not_ambiguous(caplog):
    session, _, _ = _hand(("x", Q, "add_1"))
    assert _lines(caplog, "SOURCE-ORDER AMBIGUOUS") == []
    assert (session.source_ambiguous, session.source_ambiguous_inputs) == (0, 0)
    # The looser count still lists it: it shares [8] with the sourced input.
    assert session.source_shared_inputs == 1


def test_unsourced_inputs_of_distinct_shapes_are_not_ambiguous(caplog):
    session, _, _ = _hand(("x", "add_1"))
    assert _lines(caplog, "SOURCE-ORDER AMBIGUOUS") == []
    assert (session.source_ambiguous, session.source_shared_inputs) == (0, 0)


def test_ambiguous_inputs_skips_symints_and_counts_groups():
    f8 = ("torch.float32", (8,))
    sig = (("torch.float32", (4, 8)), "SymInt", f8, f8, f8)
    srcs = (None, None, None, Q, None)
    ambiguous, shared = shape_variants.ambiguous_inputs(sig, srcs)
    assert ambiguous == [(2, f8), (4, f8)]
    assert shared == [(2, f8), (4, f8)]


def test_warmup_reports_the_ambiguity_counts(caplog):
    with shape_variants.warmup() as session:
        session.put(None, KEY, _graph(("x", "add_1", "add_2", "mul_3")), object())
        session.get(None, KEY, _graph(("x", "add_1", "add_2", "mul_3")))
    summary = _lines(caplog, "SOURCE-ORDER CHECK")
    assert summary == [
        "[shape-variants] SOURCE-ORDER CHECK handovers=1 compared=0 blind=1 "
        "mismatches=0 traces=0 ambiguous=1 ambiguous_inputs=3 shared_inputs=3"
    ], summary
