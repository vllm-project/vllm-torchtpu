# SPDX-License-Identifier: Apache-2.0
"""Source-order check on bucket hand-over (diagnostics only).

A handed-over runnable takes its tensor inputs by position, and the hand-over
is accepted on a (dtype, shape) signature. Two traces that first use two
same-shape tensors in opposite order therefore match, and the adopting trace
would feed each tensor to the other's placeholder. The check names every
sourced tensor placeholder and logs ``SOURCE-ORDER MISMATCH`` when a hand-over
would bind one out of place. It never changes what the hand-over returns.
"""

import logging

import pytest
import torch
from torch import fx
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_torchtpu.compilation import shape_variants

KEY = "artifact_compile_range_16_16_subgraph_0"
Q = "l_self_modules_norm_query_parameters_weight_"
K = "l_self_modules_norm_key_parameters_weight_"
MISMATCH = "SOURCE-ORDER MISMATCH"


def _graph(order: tuple[str, ...], *, with_n: bool = False) -> fx.GraphModule:
    """x * q + k (q and k both float32 [8]) with placeholders in ``order``.

    ``n`` (a SymInt, the size of input_ids) is multiplied in when present.
    """
    env = ShapeEnv()
    mode = FakeTensorMode(shape_env=env)
    meta = {
        "x": mode.from_tensor(torch.zeros(4, 8)),
        Q: mode.from_tensor(torch.zeros(8)),
        K: mode.from_tensor(torch.zeros(8)),
        "add_1": mode.from_tensor(torch.zeros(8)),
        "add_2": mode.from_tensor(torch.zeros(8)),
    }
    if with_n:
        root = env.create_symbol(
            4, TensorPropertySource(LocalSource("input_ids"), TensorProperty.SIZE, 0)
        )
        meta["n"] = env.create_symintnode(root, hint=4)
    graph = fx.Graph()
    nodes = {}
    for name in order:
        nodes[name] = graph.placeholder(name)
        nodes[name].meta["example_value"] = meta[name]
    a, b = (Q, K) if Q in nodes else ("add_1", "add_2")
    out = graph.call_function(
        torch.add, (graph.call_function(torch.mul, (nodes["x"], nodes[a])), nodes[b])
    )
    if with_n:
        out = graph.call_function(torch.mul, (out, nodes["n"]))
    graph.output(out)
    return fx.GraphModule(torch.nn.Module(), graph)


def _args(graph: fx.GraphModule, values: dict):
    return [values[node.name] for node in graph.graph.nodes if node.op == "placeholder"]


def _hand(first: fx.GraphModule, second: fx.GraphModule):
    session = shape_variants.BucketExecutables()
    session.put(None, KEY, first, first)
    return session, session.get(None, KEY, second)


def _mismatch_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if MISMATCH in r.getMessage()]


@pytest.fixture(autouse=True)
def _capture(caplog):
    # vllm_torchtpu loggers are "vllm.<module>", and vLLM's "vllm" logger does
    # not propagate to the root logger caplog listens on: attach directly, and
    # stop propagation meanwhile so no record is captured twice.
    name = shape_variants.logger.name  # resolve the lazy logger (configures vLLM logging first)
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


def test_swapped_same_shape_sources_log_a_mismatch_and_hand_over_unchanged(caplog):
    first, second = _graph(("x", Q, K)), _graph(("x", K, Q))
    session, handed = _hand(first, second)
    # Behaviour unchanged: the earlier trace's runnable, exactly as before.
    assert handed is first
    lines = _mismatch_lines(caplog)
    assert len(lines) == 1, lines
    assert KEY in lines[0] and Q in lines[0] and K in lines[0]
    # Both source lists are printed.
    assert "compiled-for sources:" in lines[0] and "given sources:" in lines[0]
    assert (session.source_handovers, session.source_compared) == (1, 1)
    assert (session.source_blind, session.source_mismatches) == (0, 1)


def test_the_flagged_hand_over_is_numerically_wrong():
    # What the warning is about: the adopting trace feeds k where q belongs.
    first, second = _graph(("x", Q, K)), _graph(("x", K, Q))
    _, handed = _hand(first, second)
    g = torch.Generator().manual_seed(0)
    values = {
        "x": torch.randn(4, 8, generator=g),
        Q: torch.randn(8, generator=g),
        K: torch.randn(8, generator=g),
    }
    native = second(*_args(second, values))
    assert not torch.allclose(handed(*_args(second, values)), native)


def test_same_order_logs_nothing(caplog):
    first, second = _graph(("x", Q, K)), _graph(("x", Q, K))
    session, handed = _hand(first, second)
    assert handed is first
    assert _mismatch_lines(caplog) == []
    assert (session.source_compared, session.source_mismatches) == (1, 0)


def test_unsourced_intermediates_are_not_compared(caplog):
    # Op-order names differ between traces for the same value; never flagged.
    first, second = _graph(("x", "add_1", "add_2")), _graph(("x", "add_2", "add_1"))
    session, handed = _hand(first, second)
    assert handed is first
    assert _mismatch_lines(caplog) == []
    # "x" is unsourced too, so this hand-over is blind, and counted as such.
    assert (session.source_compared, session.source_blind) == (0, 1)


def test_symint_permuted_hand_over_is_checked_too(caplog):
    first = _graph(("x", "n", Q, K), with_n=True)
    second = _graph(("x", K, Q, "n"), with_n=True)
    session, handed = _hand(first, second)
    assert handed is not None and handed is not first  # the permuting adapter
    lines = _mismatch_lines(caplog)
    assert len(lines) == 1 and Q in lines[0] and K in lines[0], lines
    assert session.source_mismatches == 1


def test_symint_moved_alone_is_not_a_mismatch(caplog):
    first = _graph(("x", "n", Q, K), with_n=True)
    second = _graph(("x", Q, K, "n"), with_n=True)
    session, handed = _hand(first, second)
    assert handed is not None
    assert _mismatch_lines(caplog) == []
    assert (session.source_compared, session.source_mismatches) == (1, 0)


def test_warmup_reports_the_counts(caplog):
    with shape_variants.warmup() as session:
        session.put(None, KEY, _graph(("x", Q, K)), object())
        session.get(None, KEY, _graph(("x", K, Q)))
        session.get(None, KEY, _graph(("x", Q, K)))
    summary = [r.getMessage() for r in caplog.records if "SOURCE-ORDER CHECK" in r.getMessage()]
    assert summary == [
        "[shape-variants] SOURCE-ORDER CHECK handovers=2 compared=2 blind=0 "
        "mismatches=1 traces=0 ambiguous=0 ambiguous_inputs=0 shared_inputs=0"
    ], summary


def test_source_order_diffs_ignores_symint_positions():
    f8 = ("float32", (8,))
    sig_old = ("x", "SymInt", f8, f8)
    sig_new = ("x", f8, "SymInt", f8)
    # The SymInt moved; the non-SymInt order (x, Q, K) is the same -> nothing out of place.
    assert shape_variants.source_order_diffs(
        (sig_old, (None, None, Q, K)), (sig_new, (None, Q, None, K))
    ) == []
    # Q and K trade places among the non-SymInt inputs -> both positions flagged.
    assert shape_variants.source_order_diffs(
        (sig_old, (None, None, Q, K)), (sig_new, (None, K, None, Q))
    ) == [(1, Q, K), (2, K, Q)]


def test_detector_is_outside_the_compile_cache_key():
    # Compiled-program identity: shape_variants.py is not a hashed cache-key file.
    from pathlib import Path

    from vllm_torchtpu.compilation import tpu_compiler

    repo_root = Path(tpu_compiler.__file__).resolve().parents[2]
    files = {
        p.relative_to(repo_root).as_posix()
        for p in tpu_compiler._iter_runtime_cache_key_files(repo_root)
    }
    assert files, "no cache-key files found"
    assert "vllm_torchtpu/compilation/shape_variants.py" not in files
