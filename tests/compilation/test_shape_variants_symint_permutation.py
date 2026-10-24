# SPDX-License-Identifier: Apache-2.0
"""Bucket hand-over between traces whose inputs differ only in SymInt order.

Two traces of a split backbone can take the same inputs with a SymInt
placeholder at a different position (FX orders placeholders by first use).
The hand-over matches SymInts by sympy symbol and permutes the arguments;
any other difference still refuses.
"""

import sympy
import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_torchtpu.compilation import shape_variants

KEY = "artifact_compile_range_4_4_subgraph_0"


def _symint(env: ShapeEnv, name: str, hint: int) -> torch.SymInt:
    symbol = sympy.Symbol(name, integer=True, positive=True)
    return env.create_symintnode(symbol, hint=hint)


def _graph(
    order: tuple[str, ...], *, y_dtype=torch.float32, n_name="s72"
) -> fx.GraphModule:
    """A graph computing x * n + y with its placeholders in ``order``.

    Each graph gets its own ShapeEnv, as each Dynamo trace does; the token
    symbol keeps its name across them.
    """
    env = ShapeEnv()
    mode = FakeTensorMode(shape_env=env)
    meta = {
        "x": mode.from_tensor(torch.zeros(4, 8)),
        "y": mode.from_tensor(torch.zeros(4, 8, dtype=y_dtype)),
        "n": _symint(env, n_name, 4),
    }
    graph = fx.Graph()
    nodes = {}
    for name in order:
        nodes[name] = graph.placeholder(name)
        nodes[name].meta["example_value"] = meta[name]
    out = graph.call_function(
        torch.add,
        (graph.call_function(torch.mul, (nodes["x"], nodes["n"])), nodes["y"]),
    )
    graph.output(out)
    return fx.GraphModule(torch.nn.Module(), graph)


def _args(graph: fx.GraphModule, x, y, n):
    values = {"x": x, "y": y, "n": n}
    return [values[node.name] for node in graph.graph.nodes if node.op == "placeholder"]


def test_permuted_symint_is_handed_over_and_matches_a_fresh_compile():
    first = _graph(("x", "n", "y"))
    second = _graph(("x", "y", "n"))
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, first, first)

    adopted = session.get(None, KEY, second)
    assert adopted is not None and adopted is not first

    x, y = torch.randn(4, 8), torch.randn(4, 8)
    got = adopted(*_args(second, x, y, 3))
    fresh = second(*_args(second, x, y, 3))
    assert torch.equal(got, fresh)


def test_identical_signature_hands_over_the_same_runnable():
    first = _graph(("x", "n", "y"))
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, first, first)
    assert session.get(None, KEY, _graph(("x", "n", "y"))) is first


def test_a_real_input_difference_still_refuses():
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, _graph(("x", "n", "y")), lambda *a: None)
    try:
        session.get(None, KEY, _graph(("x", "y", "n"), y_dtype=torch.bfloat16))
    except shape_variants.ShapeSpecializationError as exc:
        assert "inputs differ" in str(exc)
    else:
        raise AssertionError("a dtype difference was handed over")


def test_a_symbol_only_one_trace_has_still_refuses():
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, _graph(("x", "n", "y")), lambda *a: None)
    try:
        session.get(None, KEY, _graph(("x", "y", "n"), n_name="s80"))
    except shape_variants.ShapeSpecializationError:
        pass
    else:
        raise AssertionError("SymInts with different symbols were matched")
