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
"""Tests for `TpuMonoSpec.load_tensor` and `observe_model_loader`."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

import vllm_torchtpu
from tests.megakernel.adapter.conftest import FAKE_TEMPLATE
from vllm_torchtpu.megakernel import adapter as mkr
from vllm_torchtpu.megakernel.adapter import (
    constants as adapter_constants,
)
from vllm_torchtpu.megakernel.adapter import (
    hooks,
    salt,
    testkit,
)
from vllm_torchtpu.megakernel.adapter.hooks import observe_model_loader
from vllm_torchtpu.megakernel.adapter.salt import resolve_plugin_hook
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.runtime import contract
from vllm_torchtpu.megakernel.core.runtime.contract import TpuMonoSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class _DefaultLoader:
    """Stub matching vLLM streaming model loaders."""

    def __init__(self, pairs: list[tuple[str, torch.Tensor]]) -> None:
        """Initializes the instance."""
        self._pairs = pairs

    def load_weights(self, model: object, _cfg: object = None) -> None:
        """Stream checkpoint pairs directly into model.load_weights."""
        if hasattr(model, "load_weights"):
            model.load_weights(iter(self._pairs))


class _ModelWithNestedLoad:
    """Model whose `load_weights` delegates once to `self.load_weights`."""

    def __init__(self) -> None:
        """Initializes the instance."""
        self.loaded: list[tuple[str, torch.Tensor]] = []
        self._nested = False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consume weights, exercising nested self-call re-entrancy."""
        items = list(weights)
        if not self._nested:
            self._nested = True
            return self.load_weights(items)
        self.loaded.extend(items)
        return {k for k, _ in items}


class _RunaiStyleLoader:
    """Stub matching `RunaiModelStreamerLoader.load_weights`."""

    def __init__(self, pairs: list[tuple[str, torch.Tensor]]) -> None:
        """Initializes the instance."""
        self._pairs = pairs

    def load_weights(self, model: _ModelWithNestedLoad, _cfg: object = None) -> None:
        """Stream iterator directly into `model.load_weights`."""
        model.load_weights(iter(self._pairs))


class _FailingClassLoader:
    """Loader with class-level `load_weights` yielding a failing key."""

    def load_weights(self, model: object, _cfg: object = None) -> None:
        """Stream pairs to model."""
        pairs = [("ok", torch.ones((1,))), ("boom", torch.ones((1,)))]
        if hasattr(model, "load_weights"):
            model.load_weights(iter(pairs))


class _SingleWeightLoader:
    """Stub loader yielding one weight."""

    def load_weights(self, model: object, _cfg: object = None) -> None:
        """Stream one pair."""
        if hasattr(model, "load_weights"):
            model.load_weights(iter([("w", torch.ones((2,)))]))


class _StubRunnerLoader:
    """Minimal `BaseModelLoader`-shaped loader for runner wiring checks."""

    def load_weights(self, model: object, _cfg: object = None) -> None:
        """Stream weights into model."""
        pairs = [
            ("embed.weight", torch.ones((2, 2))),
            ("lm_head.weight", torch.ones((2, 2))),
        ]
        if hasattr(model, "load_weights"):
            model.load_weights(iter(pairs))

    def load_model(self, *, vllm_config: object, model_config: object) -> list[str]:
        """Consume `load_weights` as `BaseModelLoader` does."""
        _ = (vllm_config, model_config)
        model = _ModelWithNestedLoad()
        self.load_weights(model)
        return [k for k, _ in model.loaded]


@pytest.fixture(autouse=True)
def _clean_mk_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear `VLLM_TPU_MEGAKERNEL` before each test."""
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


def _register_plugin(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    spec: TpuMonoSpec,
) -> None:
    """Register an in-memory plugin module exposing `SPEC`."""
    module = ModuleType(name)
    monkeypatch.setattr(module, "SPEC", spec, raising=False)
    monkeypatch.setitem(sys.modules, name, module)


def _vllm_cfg(plugin: str | None, options: dict[str, object] | None = None) -> object:
    """Build a minimal `VllmConfig`-shaped namespace for plugin resolution."""
    mk: object = (
        False if plugin is None else {"plugin": plugin, "options": dict(options or {})}
    )
    return SimpleNamespace(
        additional_config={"megakernel": mk},
        model_config=SimpleNamespace(hf_config=None),
    )


def _sample_pairs() -> tuple[
    list[tuple[str, torch.Tensor]], list[tuple[str, tuple[int, ...], object]]
]:
    """Return checkpoint `(name, tensor)` pairs and expected hook observations."""
    w0 = torch.ones((2, 4), dtype=torch.float32)
    w1 = torch.zeros((4,), dtype=torch.uint8)
    pairs = [("layers.0.experts.w13_weight", w0), ("layers.0.experts.scale", w1)]
    expected: list[tuple[str, tuple[int, ...], object]] = [
        ("layers.0.experts.w13_weight", (2, 4), "tp32"),
        ("layers.0.experts.scale", (4,), "tp32"),
    ]
    return pairs, expected


def _recording_cfg(
    monkeypatch: pytest.MonkeyPatch,
    pkg: str,
    seen: list[tuple[str, tuple[int, ...], object]],
) -> object:
    """Register a test plugin recording streamed weight metadata."""

    def _load_tensor(
        name: str,
        tensor: torch.Tensor,
        options: dict[str, object],
    ) -> str:
        """Record observed weight metadata and return ignored sentinel."""
        seen.append((name, tuple(tensor.shape), options.get("tag")))
        return "ignored-sentinel"

    _register_plugin(
        monkeypatch,
        pkg,
        TpuMonoSpec("obs", "1", create=".a:c", load_tensor=_load_tensor),
    )
    return _vllm_cfg(pkg, {"tag": "tp32"})


def test_spec_accepts_load_tensor() -> None:
    """`TpuMonoSpec` stores `None`, a lazy `'module:attr'` string, or a callable."""
    default_spec = TpuMonoSpec("m", "1", create=".a:c")
    assert default_spec.load_tensor is None

    lazy_spec = TpuMonoSpec("m", "1", create=".a:c", load_tensor=".weights:load_tensor")
    assert lazy_spec.load_tensor == ".weights:load_tensor"

    def _hook(name: str, tensor: object, options: dict[str, object]) -> None:
        """No-op hook."""
        _ = (name, tensor, options)

    fn_spec = TpuMonoSpec("m", "1", create=".a:c", load_tensor=_hook)
    assert fn_spec.load_tensor is _hook


def test_load_tensor_stream_passes_unchanged_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_tensor_stream` yields unchanged pairs while calling `load_tensor`."""
    seen: list[tuple[str, tuple[int, ...], object]] = []
    cfg = _recording_cfg(monkeypatch, "obs_stream_pkg", seen)
    pairs, expected = _sample_pairs()

    streamed = list(hooks.load_tensor_stream(cfg, pairs))
    assert streamed == pairs
    assert seen == expected


def test_observe_model_loader_wraps_default_load_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap `DefaultModelLoader.load_weights` and restore on exit."""
    seen: list[tuple[str, tuple[int, ...], object]] = []
    cfg = _recording_cfg(monkeypatch, "obs_default_pkg", seen)
    pairs, expected = _sample_pairs()

    default_loader = _DefaultLoader(pairs)
    model = _ModelWithNestedLoad()
    with mkr.observe_model_loader(cfg, default_loader):
        default_loader.load_weights(model, None)
    assert model.loaded == pairs
    assert "load_weights" not in vars(default_loader)
    assert "load_weights" not in vars(model)
    assert seen == expected


def test_observe_model_loader_wraps_runai_load_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`observe_model_loader` observes `RunaiModelStreamerLoader.load_weights` once."""
    seen: list[tuple[str, tuple[int, ...], object]] = []
    cfg = _recording_cfg(monkeypatch, "obs_runai_pkg", seen)
    pairs, expected = _sample_pairs()

    runai_loader = _RunaiStyleLoader(pairs)
    model = _ModelWithNestedLoad()
    with mkr.observe_model_loader(cfg, runai_loader):
        runai_loader.load_weights(model, None)
    assert model.loaded == pairs
    assert "load_weights" not in vars(runai_loader)
    assert "load_weights" not in vars(model)
    assert seen == expected


def test_observe_model_loader_restores_loader_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-stream errors restore both class methods and prior instance attributes."""

    def _failing_load_tensor(
        name: str,
        tensor: object,
        options: dict[str, object],
    ) -> None:
        """Raise on `boom` to verify loader restoration."""
        _ = (tensor, options)
        if name == "boom":
            msg = "hook failed"
            raise RuntimeError(msg)

    _register_plugin(
        monkeypatch,
        "obs_err_pkg",
        TpuMonoSpec("obs_err", "1", create=".a:c", load_tensor=_failing_load_tensor),
    )
    cfg = _vllm_cfg("obs_err_pkg")

    cls_loader = _FailingClassLoader()
    model = _ModelWithNestedLoad()
    with (
        pytest.raises(RuntimeError, match="hook failed"),
        mkr.observe_model_loader(cfg, cls_loader),
    ):
        cls_loader.load_weights(model)
    assert "load_weights" not in vars(cls_loader)
    assert "load_weights" not in vars(model)

    def _prior_load_weights(model: object, _cfg: object = None) -> None:
        """Instance-level weight loader."""
        if hasattr(model, "load_weights"):
            model.load_weights([("boom", torch.ones((1,)))])

    ns_loader = SimpleNamespace(load_weights=_prior_load_weights)
    model2 = _ModelWithNestedLoad()
    with (
        pytest.raises(RuntimeError, match="hook failed"),
        mkr.observe_model_loader(cfg, ns_loader),
    ):
        ns_loader.load_weights(model2)
    assert ns_loader.load_weights is _prior_load_weights
    assert "load_weights" not in vars(model2)


def test_observe_model_loader_public_exports_and_unknown_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `load_tensor_stream` export and unknown hook error."""
    _register_plugin(
        monkeypatch,
        "obs_pub_pkg",
        TpuMonoSpec("obs_pub", "1", create=".a:c"),
    )
    cfg = _vllm_cfg("obs_pub_pkg")
    assert mkr.load_tensor_stream is hooks.load_tensor_stream
    resolve_hook: Callable[..., object] = salt.resolve_plugin_hook
    with pytest.raises(ValueError, match="Unknown plugin hook"):
        resolve_hook(cfg, "unknown_hook")


def test_observe_model_loader_is_noop_without_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled megakernel or `load_tensor=None` leaves `model_loader` untouched."""
    _register_plugin(
        monkeypatch,
        "obs_none_pkg",
        TpuMonoSpec("obs_none", "1", create=".a:c"),
    )
    loader = _SingleWeightLoader()
    expected_single = 1
    for cfg in (_vllm_cfg(None), _vllm_cfg("obs_none_pkg")):
        assert salt.resolve_plugin_hook(cfg, "load_tensor") is None
        with mkr.observe_model_loader(cfg, loader):
            assert "load_weights" not in vars(loader)
            model = _ModelWithNestedLoad()
            loader.load_weights(model)
            assert len(model.loaded) == expected_single


def test_check_spec_resolves_lazy_load_tensor(
    mk_fake_template: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate lazy `load_tensor` strings and bad targets in `check_plugin_spec`."""
    _ = mk_fake_template
    template_mod = importlib.import_module(FAKE_TEMPLATE)
    adapter_mod = importlib.import_module(f"{FAKE_TEMPLATE}.adapter")

    def _noop_load_tensor(
        name: str,
        tensor: object,
        options: dict[str, object],
    ) -> None:
        """Accept the 3-argument `load_tensor(name, tensor, options)` contract."""
        _ = (name, tensor, options)

    monkeypatch.setitem(vars(adapter_mod), "load_tensor", _noop_load_tensor)
    good_spec = TpuMonoSpec(
        "good_lt",
        "1",
        create=".adapter:create",
        load_tensor=".adapter:load_tensor",
    )
    bad_spec = TpuMonoSpec(
        "bad_lt",
        "1",
        create=".adapter:create",
        load_tensor=".adapter:missing_fn",
    )
    monkeypatch.setitem(vars(template_mod), "GOOD_SPEC", good_spec)
    monkeypatch.setitem(vars(template_mod), "BAD_SPEC", bad_spec)

    checked = testkit.check_plugin_spec(f"{FAKE_TEMPLATE}:GOOD_SPEC")
    assert checked.load_tensor == ".adapter:load_tensor"
    resolved = salt.resolve_plugin_hook(
        _vllm_cfg(f"{FAKE_TEMPLATE}:GOOD_SPEC"),
        "load_tensor",
    )
    assert resolved is not None
    fn, opts = resolved
    assert fn is _noop_load_tensor
    fn("w", torch.ones((1,)), opts)
    with pytest.raises(MegakernelConfigError, match="can't resolve load_tensor"):
        testkit.check_plugin_spec(f"{FAKE_TEMPLATE}:BAD_SPEC")


def test_runner_load_model_observes_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TPUModelRunner.load_model` wraps `load_model` in `observe_model_loader`."""
    runner_path = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    src = runner_path.read_text(encoding="utf-8")
    load_fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    obs_with_nodes = [
        w
        for w in ast.walk(load_fn)
        if isinstance(w, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "observe_model_loader"
            and len(item.context_expr.args) == len(("cfg", "loader"))
            for item in w.items
        )
    ]
    expected_with_blocks = 1
    assert len(obs_with_nodes) == expected_with_blocks
    body_src = ast.unparse(obs_with_nodes[0])
    assert "model_loader.load_model" in body_src

    names: list[str] = []

    def _record_name(
        name: str,
        tensor: object,
        options: dict[str, object],
    ) -> None:
        """Append each streamed weight name."""
        _ = (tensor, options)
        names.append(name)

    _register_plugin(
        monkeypatch,
        "obs_runner_pkg",
        TpuMonoSpec("obs_runner", "1", create=".a:c", load_tensor=_record_name),
    )
    cfg = _vllm_cfg("obs_runner_pkg")
    stub_loader = _StubRunnerLoader()
    with mkr.observe_model_loader(cfg, stub_loader):
        loaded = stub_loader.load_model(vllm_config=cfg, model_config=None)
    assert loaded == ["embed.weight", "lm_head.weight"]
    assert names == ["embed.weight", "lm_head.weight"]
    assert "load_weights" not in vars(stub_loader)


def test_adapter_constants_module_and_explicit_hook_dispatch() -> None:
    """Validate adapter/constants.py exports and explicit hook dispatch."""
    assert hasattr(adapter_constants, "DRIVER_EXPORTS")
    assert hasattr(adapter_constants, "JAX_BRIDGE_EXPORTS")
    assert hasattr(adapter_constants, "SUPPRESS_ATTR")
    assert adapter_constants.SUPPRESS_ATTR == "_megakernel_suppress_depth"

    assert not inspect.isgeneratorfunction(observe_model_loader)

    cfg = _vllm_cfg("test_spec")
    with pytest.raises(ValueError, match="Unknown plugin hook"):
        resolve_plugin_hook(cfg, "unknown_hook_name")


def test_observe_real_upstream_default_model_loader_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pure-stdlib `hooks` import and real upstream `DefaultModelLoader`."""
    hooks_tree = ast.parse(Path(hooks.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(hooks_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "salt" not in imported_modules
    assert adapter_constants.HOOK_LOAD_TENSOR == "load_tensor"
    assert adapter_constants.HOOK_LAYOUT_REQUEST == "layout_request"
    assert hasattr(contract, "LoadTensorHook")

    if "vllm" not in sys.modules and not os.environ.get("VLLM_PLUGINS"):
        monkeypatch.setenv("VLLM_PLUGINS", "torchtpu")

    vllm_cfg_mod = importlib.import_module("vllm.config")
    load_cfg_mod = importlib.import_module("vllm.config.load")
    default_loader_mod = importlib.import_module(
        "vllm.model_executor.model_loader.default_loader"
    )
    real_base_cls = default_loader_mod.DefaultModelLoader

    pairs, expected = _sample_pairs()

    def _get_weights_iterator(
        _self: object,
        source: object,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield in-memory checkpoint pairs without disk I/O."""
        _ = source
        return iter(pairs)

    in_memory_loader_cls = type(
        "_InMemoryDefaultLoader",
        (real_base_cls,),
        {"_get_weights_iterator": _get_weights_iterator},
    )

    class _TargetModule(nn.Module):
        """Minimal `nn.Module` recording streamed weights."""

        def __init__(self) -> None:
            """Initializes the instance."""
            super().__init__()
            self.loaded: list[tuple[str, torch.Tensor]] = []

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> set[str]:
            """Consume streamed weights."""
            items = list(weights)
            self.loaded.extend(items)
            return {name for name, _ in items}

    monkeypatch.setattr(
        vllm_cfg_mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_expert_parallel=False,
                enable_ep_weight_filter=False,
                enable_eplb=False,
            )
        ),
    )
    load_config = load_cfg_mod.LoadConfig(
        model_loader_extra_config={"enable_weights_track": False}
    )
    real_loader = in_memory_loader_cls(load_config)
    target_model = _TargetModule()
    model_cfg = SimpleNamespace(
        quantization=None,
        is_moe=False,
        model="stub-qwen38",
        revision=None,
    )

    seen: list[tuple[str, tuple[int, ...], object]] = []
    plugin_cfg = _recording_cfg(monkeypatch, "obs_real_default_loader_pkg", seen)
    with mkr.observe_model_loader(plugin_cfg, real_loader):
        real_loader.load_weights(target_model, model_cfg)

    assert target_model.loaded == pairs
    assert seen == expected
    assert "load_weights" not in real_loader.__dict__
    assert "load_weights" not in target_model.__dict__
