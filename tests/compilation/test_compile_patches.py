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
"""Unit tests for the vLLM compilation patches applied for per-layer compile."""

from types import SimpleNamespace

import pytest
import torch
import vllm.compilation.wrapper as wrapper_mod
from vllm.compilation.piecewise_backend import PiecewiseBackend
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

from vllm_torchtpu import (
    _patch_vllm_compile_prefix_isolation,
    _patch_vllm_piecewise_backend,
    _patch_vllm_reset_compile_wrapper,
)


class _Wrapped(torch.nn.Module, TorchCompileWithNoGuardsWrapper):
    """A decorated layer: nn.Module plus the compile wrapper, as vLLM builds it.

    `TorchCompileWithNoGuardsWrapper.__init__` needs a live vllm_config, so only
    the nn.Module half is initialized here; `isinstance` is what the patches use.
    """

    def __init__(self, prefix: str | None = None) -> None:
        torch.nn.Module.__init__(self)
        if prefix is not None:
            self.prefix = prefix


class _OtherWrapped(_Wrapped):
    """A second decorated class without a prefix."""


class TestResetCompileWrapperRecursion:
    """`reset_compile_wrapper` must reach wrappers nested below the model."""

    @pytest.fixture
    def calls(self):
        """Install the patch over a recording stub; restore the module after."""
        original = wrapper_mod.reset_compile_wrapper
        seen = []
        wrapper_mod.reset_compile_wrapper = lambda m: seen.append(m)
        _patch_vllm_reset_compile_wrapper()
        yield seen
        wrapper_mod.reset_compile_wrapper = original

    def test_resets_every_nested_wrapper(self, calls):
        # The shape that matters: wrappers are per-layer children, and vLLM's
        # own reset unwraps only one level before giving up.
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([_Wrapped() for _ in range(3)])
        model.model.head = _Wrapped()

        wrapper_mod.reset_compile_wrapper(model)

        assert len(calls) == 4
        assert all(isinstance(m, _Wrapped) for m in calls)

    def test_wrapper_passed_directly_is_reset_once(self, calls):
        layer = _Wrapped()
        wrapper_mod.reset_compile_wrapper(layer)
        assert calls == [layer]

    def test_falls_back_to_the_model_when_no_wrapper_exists(self, calls):
        # Preserves upstream behavior for models that are not compiled per layer.
        model = torch.nn.Module()
        model.model = torch.nn.Module()

        wrapper_mod.reset_compile_wrapper(model)

        assert calls == [model]

    def test_none_is_ignored(self, calls):
        wrapper_mod.reset_compile_wrapper(None)
        assert calls == []

    def test_reinstalling_does_not_stack(self, calls):
        _patch_vllm_reset_compile_wrapper()
        wrapper_mod.reset_compile_wrapper(_Wrapped())
        assert len(calls) == 1


class TestPiecewiseBackendDispatch:
    """Bucket dispatch when no argument is a SymInt."""

    @pytest.fixture(autouse=True)
    def patched(self):
        original = PiecewiseBackend.__call__
        _patch_vllm_piecewise_backend()
        yield
        PiecewiseBackend.__call__ = original

    @staticmethod
    def _backend(sym_shape_indices):
        """A backend with two compiled buckets, keyed by the shape asked for."""
        entries = {
            n: SimpleNamespace(
                compiled=True, compile_range=(n, n), runnable=lambda *a, n=n, **k: n
            )
            for n in (16, 128)
        }
        return SimpleNamespace(
            sym_shape_indices=sym_shape_indices,
            range_entries=entries,
            compile_sizes=[16, 128],
            compile_ranges=[(16, 16), (128, 128)],
            _find_range_for_shape=entries.get,
        )

    def test_resolves_bucket_from_tensor_dim_0(self):
        # vLLM derives sym_shape_indices from args that *are* SymInts; a layer
        # taking only tensors leaves it empty, and upstream then assumes a
        # single compiled bucket. Dim 0 carries the token count instead.
        backend = self._backend(sym_shape_indices=[])
        chosen = PiecewiseBackend.__call__(backend, torch.zeros(128, 4))
        assert chosen == 128

    def test_sym_shape_index_still_wins_when_present(self):
        backend = self._backend(sym_shape_indices=[1])
        chosen = PiecewiseBackend.__call__(backend, torch.zeros(128, 4), 16)
        assert chosen == 16

    def test_unknown_shape_is_rejected(self):
        backend = self._backend(sym_shape_indices=[])
        with pytest.raises(AssertionError):
            PiecewiseBackend.__call__(backend, torch.zeros(999, 4))


class TestCompilePrefixIsolation:
    """Each instance must get its own on-disk compile-cache identity."""

    @pytest.fixture
    def install(self):
        """Patch over a recorder of the compile_prefix forwarded to __init__.

        Each call starts again from the unpatched recorder, as a new process
        does, and returns the emptied list the recorder appends to.
        """
        original = TorchCompileWithNoGuardsWrapper.__init__
        seen = []

        def recorder(self, compile_prefix="", is_encoder=False):
            seen.append(compile_prefix)

        def install():
            seen.clear()
            TorchCompileWithNoGuardsWrapper.__init__ = recorder
            _patch_vllm_compile_prefix_isolation()
            return seen

        yield install
        TorchCompileWithNoGuardsWrapper.__init__ = original

    @pytest.fixture
    def prefixes(self, install):
        """Prefixes recorded under one installed patch."""
        return install()

    @staticmethod
    def _build(*layers):
        for layer in layers:
            TorchCompileWithNoGuardsWrapper.__init__(layer)

    @staticmethod
    def _fallback(cls, index):
        return f"{cls.__module__}.{cls.__qualname__}_{index}"

    def test_uses_self_prefix(self, prefixes):
        layer = _Wrapped(prefix="model.layers.7")
        TorchCompileWithNoGuardsWrapper.__init__(layer)
        assert prefixes == ["model.layers.7"]

    def test_distinct_prefixes_stay_distinct(self, prefixes):
        for i in (0, 1):
            TorchCompileWithNoGuardsWrapper.__init__(
                _Wrapped(prefix=f"model.layers.{i}")
            )
        assert prefixes == ["model.layers.0", "model.layers.1"]

    def test_missing_prefix_falls_back_to_class_and_index(self, prefixes):
        self._build(_Wrapped(), _Wrapped(prefix="model.layers.3"), _Wrapped())
        assert prefixes == [
            self._fallback(_Wrapped, 0),
            "model.layers.3",
            self._fallback(_Wrapped, 1),
        ]

    def test_fallback_index_counts_per_class(self, prefixes):
        # Another class built in between does not shift _Wrapped's names.
        self._build(_Wrapped(), _OtherWrapped(), _Wrapped())
        assert prefixes == [
            self._fallback(_Wrapped, 0),
            self._fallback(_OtherWrapped, 0),
            self._fallback(_Wrapped, 1),
        ]

    def test_fallback_names_repeat_after_a_restart(self, install):
        runs = []
        for _ in range(2):
            seen = install()
            self._build(_Wrapped(), _Wrapped(prefix="model.layers.3"), _Wrapped())
            runs.append(list(seen))
        assert runs[0] == runs[1]
        assert len(set(runs[0])) == 3

    def test_explicit_prefix_is_left_alone(self, prefixes):
        TorchCompileWithNoGuardsWrapper.__init__(
            _Wrapped(prefix="ignored"), compile_prefix="caller"
        )
        assert prefixes == ["caller"]

    def test_encoder_keeps_the_shared_prefix(self, prefixes):
        TorchCompileWithNoGuardsWrapper.__init__(_Wrapped(), is_encoder=True)
        assert prefixes == [""]
