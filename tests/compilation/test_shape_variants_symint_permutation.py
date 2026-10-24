# SPDX-License-Identifier: Apache-2.0
"""Bucket hand-over between traces whose inputs differ only in SymInt order.

Two traces of a split backbone can take the same inputs with a SymInt
placeholder at a different position (FX orders placeholders by first use).
The hand-over matches SymInts by the Dynamo source of their root symbol and
permutes the arguments; any other difference still refuses.
"""

import torch
from torch import fx
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_torchtpu.compilation import shape_variants

KEY = "artifact_compile_range_4_4_subgraph_0"


def _size_source(name: str) -> TensorPropertySource:
    return TensorPropertySource(LocalSource(name), TensorProperty.SIZE, 0)


def _graph(
    order: tuple[str, ...],
    *,
    y_dtype=torch.float32,
    n_source="input_ids",
    via=None,
) -> fx.GraphModule:
    """A graph computing x * n + y with its placeholders in ``order``.

    Each graph gets its own ShapeEnv, as each Dynamo trace does. ``n`` is the
    size of ``n_source``; with ``via``, it is first allocated for that other
    source and replaced by the ``n_source`` symbol, as Dynamo does when a guard
    equates two sizes.
    """
    env = ShapeEnv()
    mode = FakeTensorMode(shape_env=env)
    root = env.create_symbol(4, _size_source(n_source))
    symbol = root
    if via is not None:
        symbol = env.create_symbol(4, _size_source(via))
        env._set_replacement(symbol, root, "guard equated the two sizes")
    meta = {
        "x": mode.from_tensor(torch.zeros(4, 8)),
        "y": mode.from_tensor(torch.zeros(4, 8, dtype=y_dtype)),
        "n": env.create_symintnode(symbol, hint=4),
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


def _handed(first: fx.GraphModule, second: fx.GraphModule):
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, first, first)
    return session.get(None, KEY, second)


def test_permuted_symint_is_handed_over_and_matches_a_fresh_compile():
    second = _graph(("x", "y", "n"))
    adopted = _handed(_graph(("x", "n", "y")), second)
    assert adopted is not None

    x, y = torch.randn(4, 8), torch.randn(4, 8)
    # Without the permutation this would compute x * y + n.
    assert torch.equal(
        adopted(*_args(second, x, y, 3)), second(*_args(second, x, y, 3))
    )


def test_a_replaced_symbol_resolves_to_its_root_source():
    # The new trace allocated the size for another tensor first; a guard then
    # replaced it by the input_ids size. Its symbol name differs, its root
    # source does not.
    second = _graph(("x", "y", "n"), via="positions")
    adopted = _handed(_graph(("x", "n", "y")), second)
    x, y = torch.randn(4, 8), torch.randn(4, 8)
    assert torch.equal(
        adopted(*_args(second, x, y, 5)), second(*_args(second, x, y, 5))
    )


def test_identical_signature_hands_over_the_same_runnable():
    first = _graph(("x", "n", "y"))
    assert _handed(first, _graph(("x", "n", "y"))) is first


def _refused(second: fx.GraphModule) -> bool:
    try:
        _handed(_graph(("x", "n", "y")), second)
    except shape_variants.ShapeSpecializationError as exc:
        return "inputs differ" in str(exc)
    return False


def test_a_real_input_difference_still_refuses():
    assert _refused(_graph(("x", "y", "n"), y_dtype=torch.bfloat16))


def test_a_symint_from_another_source_still_refuses():
    assert _refused(_graph(("x", "y", "n"), n_source="positions"))
