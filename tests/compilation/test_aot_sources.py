# SPDX-License-Identifier: Apache-2.0
"""A cold start and an AOT-load start must hash the same traced source files."""

import hashlib
import importlib.util
import inspect
import sys
from unittest.mock import patch

import pytest
import torch
import vllm.compilation.wrapper as wrapper_mod
from torch._dynamo.aot_compile_types import SerializableCallable
from torch._dynamo.symbolic_convert import InliningInstructionTranslator

from vllm_torchtpu.compilation.aot_sources import record_inline_attempts

HELPER = """
def scale(table, key):
    return table[key]
"""


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _code_hash(files):
    # vLLM's VllmBackend.__call__: sorted paths, each followed by its content.
    content = []
    for path in sorted(files):
        if path == "<string>":
            continue
        content.append(path)
        with open(path) as f:
            content.append(f.read())
    return hashlib.sha256("\n".join(content).encode()).hexdigest()


class _Graph(SerializableCallable):
    """Stands in for vLLM's VllmSerializableFunction; never saved here."""

    def __init__(self, graph_module):
        self.graph_module = graph_module

    def __call__(self, *args, **kwargs):
        return self.graph_module(*args, **kwargs)

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        raise NotImplementedError

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        raise NotImplementedError


class _Wrapper:
    """Calls torch's AOT compile the way vLLM's wrapper does."""

    def __init__(self, fn):
        self._compiled_callable = torch.compile(
            fn, fullgraph=True, backend=lambda gm, _: _Graph(gm)
        )

    def aot_compile(self, *args, **kwargs):
        return self._compiled_callable.aot_compile((args, kwargs))


@pytest.fixture
def traced(tmp_path, monkeypatch):
    """A forward that calls into a helper whose inlining raises KeyError."""
    helper_path = tmp_path / "aot_sources_helper.py"
    helper_path.write_text(HELPER)
    helper = _import(helper_path, "aot_sources_helper")

    def forward(x):
        try:
            return x * helper.scale({}, "missing")
        except KeyError:
            return x + 1

    monkeypatch.setattr(wrapper_mod, "TorchCompileWithNoGuardsWrapper", _Wrapper)
    torch._dynamo.reset()
    yield forward, str(helper_path)
    torch._dynamo.reset()
    sys.modules.pop("aot_sources_helper", None)


def _trace(forward):
    """Return (cold-start files, AOT artifact) for one AOT compile of forward."""
    cold = {forward.__code__.co_filename}
    inline_call = InliningInstructionTranslator.inline_call_

    def vllm_inline_call(tx):  # vllm/compilation/decorators.py, cold start
        cold.add(tx.f_code.co_filename)
        return inline_call(tx)

    with patch.object(InliningInstructionTranslator, "inline_call_", vllm_inline_call):
        compiled = wrapper_mod.TorchCompileWithNoGuardsWrapper(forward).aot_compile(
            torch.ones(4)
        )
    return cold, compiled


def _warm_files(compiled):
    # vllm/compilation/decorators.py _verify_source_unchanged, AOT-load start.
    return {
        inspect.getfile(sys.modules[source.module])
        for source in compiled.source_info().inlined_sources
    }


def test_without_the_record_a_failed_inline_splits_the_cache_dir(traced):
    forward, helper_path = traced
    cold, compiled = _trace(forward)
    assert helper_path in cold
    assert helper_path not in _warm_files(compiled)
    assert _code_hash(cold) != _code_hash(_warm_files(compiled))


def test_cold_and_aot_load_starts_hash_the_same_files(traced):
    forward, helper_path = traced
    with patch.object(_Wrapper, "aot_compile", _Wrapper.aot_compile):
        record_inline_attempts()
        cold, compiled = _trace(forward)
    warm = _warm_files(compiled)
    assert helper_path in cold
    assert warm == cold
    assert _code_hash(warm) == _code_hash(cold)


def test_a_changed_traced_source_changes_the_hash(traced):
    forward, helper_path = traced
    with patch.object(_Wrapper, "aot_compile", _Wrapper.aot_compile):
        record_inline_attempts()
        cold, compiled = _trace(forward)
    before = _code_hash(_warm_files(compiled))
    with open(helper_path, "a") as f:
        f.write("# edited\n")
    assert _code_hash(_warm_files(compiled)) != before
    assert _code_hash(cold) != before
    recorded = {
        inspect.getfile(sys.modules[s.module]): s.content
        for s in compiled.source_info().inlined_sources
    }
    # The artifact's own source check on load now sees the edit as well.
    with open(helper_path) as f:
        assert recorded[helper_path] != f.read()
