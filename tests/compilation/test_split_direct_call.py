# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SPLIT_DIRECT_CALL: a split executable and its pieces run their
forward directly, and take nn.Module's call path whenever a hook could run."""

import operator

import pytest
import torch
import torch.fx

from vllm_torchtpu.compilation import split_direct_call


class _Leaf:
    """Stands in for a compiled executable: returns its inputs reordered."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return (args[1] * 2, args[0] + 1)


class _Wrapper(torch.nn.Module):
    """TorchTPU's _WrapperModule."""

    def __init__(self, submod, unwrap_singleton_tuple: bool):
        super().__init__()
        self.submod = submod
        self.unwrap_singleton_tuple = unwrap_singleton_tuple

    def forward(self, *args):
        x = self.submod(*args)
        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
            return x[0]
        return x


class _Split:
    """TorchTPU's _SplitCompiledExecutable, as far as the patch reads it."""

    def __init__(self, gm):
        self._split_gm = gm
        self.recompile_fn = None
        self.handshakes = 0

    def _maybe_handshake_and_recompile(self):
        if self.recompile_fn is None:
            return
        self.handshakes += 1

    def __call__(self, *args, **kwargs):
        self._maybe_handshake_and_recompile()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        return self._split_gm(*args)


def _split_of(leaf):
    graph = torch.fx.Graph()
    a, b = graph.placeholder("arg0_1"), graph.placeholder("arg1_1")
    call = graph.call_module("compiled_submod_0", (b, a))
    graph.output(
        (
            graph.call_function(operator.getitem, (call, 0)),
            graph.call_function(operator.getitem, (call, 1)),
        )
    )
    root = torch.nn.Module()
    root.add_module("compiled_submod_0", _Wrapper(leaf, False))
    return _Split(torch.fx.GraphModule(root, graph))


@pytest.fixture
def classes():
    class Split(_Split):
        pass

    class Wrapper(_Wrapper):
        pass

    split_direct_call.install(Split, Wrapper)
    return Split, Wrapper


def _patched_split(classes, leaf):
    split_cls, wrapper_cls = classes
    exe = _split_of(leaf)
    exe.__class__ = split_cls
    exe._split_gm.compiled_submod_0.__class__ = wrapper_cls
    return exe


def test_direct_call_returns_what_the_module_path_returns(classes, monkeypatch):
    leaf = _Leaf()
    exe = _patched_split(classes, leaf)
    x, y = torch.arange(3), torch.arange(3) + 10
    ref = _split_of(_Leaf())(x, y)
    called = []
    monkeypatch.setattr(
        torch.nn.Module, "_call_impl", lambda *a, **k: called.append(1) or 1 / 0
    )
    for got in (exe(x, y), exe([x, y])):
        assert all(torch.equal(g, r) for g, r in zip(got, ref))
    assert called == [] and leaf.calls == 2


def test_hooks_and_handshakes_take_the_usual_path(classes):
    leaf = _Leaf()
    exe = _patched_split(classes, leaf)
    x, y = torch.arange(3), torch.arange(3) + 10
    seen = []
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda m, args, out: seen.append(type(m).__name__)
    )
    try:
        exe(x, y)
    finally:
        handle.remove()
    assert len(seen) == 2  # the GraphModule and the piece
    exe.recompile_fn = lambda: None
    exe(x, y)
    assert exe.handshakes == 1
    local = exe._split_gm.compiled_submod_0.register_forward_pre_hook(
        lambda m, args: seen.append("pre")
    )
    exe.recompile_fn = None
    exe(x, y)
    local.remove()
    assert seen[-1] == "pre" and leaf.calls == 3


def test_lazy_graph_module_first_call_takes_the_usual_path(classes):
    """split_module builds a lazy GraphModule inside Dynamo: it has no
    _wrapped_call until its first call compiles it."""
    from torch.fx._lazy_graph_module import _use_lazy_graph_module
    from torch.fx.passes.split_module import split_module

    class Two(torch.nn.Module):
        def forward(self, a, b):
            return a + b, a * b

    with _use_lazy_graph_module(True):
        split_gm = split_module(torch.fx.symbolic_trace(Two()), Two(), lambda n: 0)
        split_gm.recompile()
    split_cls, _ = classes
    exe = _Split(split_gm)
    exe.__class__ = split_cls
    x, y = torch.arange(3), torch.arange(3) + 10
    assert getattr(type(split_gm), "_wrapped_call", None) is None
    for _ in range(2):
        got = exe(x, y)
        assert torch.equal(got[0], x + y) and torch.equal(got[1], x * y)
    assert type(split_gm)._wrapped_call.cls_call is None


def test_a_different_split_call_is_refused():
    class Split(_Split):
        def __call__(self, *args, **kwargs):
            self.resolve()
            return self._split_gm(*args)

    with pytest.raises(RuntimeError, match="not the code this patch replays"):
        split_direct_call.install(Split, _Wrapper)


def test_a_piece_with_its_own_call_is_refused():
    class Split(_Split):
        pass

    class Wrapper(_Wrapper):
        def __call__(self, *args):
            return self.forward(*args)

    with pytest.raises(RuntimeError, match="has its own __call__"):
        split_direct_call.install(Split, Wrapper)


def test_installs_on_torch_tpu():
    split_compiler = pytest.importorskip("torch_tpu._internal.compile.split_compiler")
    split_cls = split_compiler._SplitCompiledExecutable
    wrapper_cls = split_compiler._WrapperModule
    split_call = split_cls.__call__
    try:
        split_direct_call.install(split_cls, wrapper_cls)
        assert split_cls.__call__._tpu_direct
    finally:  # leave the classes as the rest of the session expects them
        split_cls.__call__ = split_call
        if "__call__" in vars(wrapper_cls):
            del wrapper_cls.__call__
