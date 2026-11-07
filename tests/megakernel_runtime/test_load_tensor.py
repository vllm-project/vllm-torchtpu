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
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import vllm_torchtpu
from vllm_torchtpu import megakernel_runtime as mkr
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec
from vllm_torchtpu.megakernel_runtime import hooks, salt, testkit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@pytest.fixture(autouse=True)
def _clean_mk_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear `VLLM_TPU_MEGAKERNEL` before each test."""
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


def _install_obs_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    body: str,
) -> str:
    """Write a temporary plugin package under `tmp_path` and register cleanup."""
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return name


def _vllm_cfg(plugin: str | None, options: dict[str, object] | None = None) -> object:
    """Build a minimal `VllmConfig`-shaped namespace for plugin resolution."""
    if plugin is None:
        mk: object = False
    else:
        mk = {"plugin": plugin, "options": dict(options or {})}
    return SimpleNamespace(
        additional_config={"megakernel": mk},
        model_config=SimpleNamespace(hf_config=None),
    )


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


def test_observe_model_loader_shows_each_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_all_weights` and `load_weights` stream unchanged pairs to `load_tensor`."""
    pkg = _install_obs_plugin(
        tmp_path,
        monkeypatch,
        "obs_stream_pkg",
        textwrap.dedent(
            """\
            from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

            SEEN = []

            def load_tensor(name, tensor, options):
                SEEN.append((name, tuple(tensor.shape), options.get("tag")))
                return "ignored-sentinel"

            SPEC = TpuMonoSpec("obs", "1", create=".a:c", load_tensor=load_tensor)
            """
        ),
    )
    try:
        mod = importlib.import_module(pkg)
        cfg = _vllm_cfg(pkg, {"tag": "tp32"})
        w0 = torch.ones((2, 4), dtype=torch.float32)
        w1 = torch.zeros((4,), dtype=torch.uint8)
        pairs = [("layers.0.experts.w13_weight", w0), ("layers.0.experts.scale", w1)]

        streamed = list(hooks.load_tensor_stream(cfg, pairs))
        assert streamed == pairs
        assert mod.SEEN == [
            ("layers.0.experts.w13_weight", (2, 4), "tp32"),
            ("layers.0.experts.scale", (4,), "tp32"),
        ]
        mod.SEEN.clear()

        class _DefaultLoader:
            """Stub matching `DefaultModelLoader.get_all_weights`."""

            def get_all_weights(
                self, _model_cfg: object, _model: object
            ) -> Iterator[tuple[str, torch.Tensor]]:
                """Yield raw checkpoint pairs."""
                yield from pairs

        default_loader = _DefaultLoader()
        with mkr.observe_model_loader(cfg, default_loader):
            out = list(default_loader.get_all_weights(None, None))
        assert out == pairs
        assert "get_all_weights" not in vars(default_loader)
        assert mod.SEEN == [
            ("layers.0.experts.w13_weight", (2, 4), "tp32"),
            ("layers.0.experts.scale", (4,), "tp32"),
        ]
        mod.SEEN.clear()

        class _ModelWithNestedLoad:
            """Model whose `load_weights` also delegates to `self.load_weights` once."""

            def __init__(self) -> None:
                """Initialize loaded record."""
                self.loaded: list[tuple[str, torch.Tensor]] = []
                self._nested = False

            def load_weights(
                self, weights: Iterable[tuple[str, torch.Tensor]]
            ) -> set[str]:
                """Consume weights, exercising nested self-call re-entrancy."""
                items = list(weights)
                if not self._nested:
                    self._nested = True
                    return self.load_weights(items)
                self.loaded.extend(items)
                return {k for k, _ in items}

        class _RunaiStyleLoader:
            """Stub matching `RunaiModelStreamerLoader.load_weights`."""

            def load_weights(self, model: _ModelWithNestedLoad, _cfg: object) -> None:
                """Stream iterator directly into `model.load_weights`."""
                model.load_weights(iter(pairs))

        runai_loader = _RunaiStyleLoader()
        model = _ModelWithNestedLoad()
        with mkr.observe_model_loader(cfg, runai_loader):
            runai_loader.load_weights(model, None)
        assert model.loaded == pairs
        assert "load_weights" not in vars(runai_loader)
        assert "load_weights" not in vars(model)
        assert mod.SEEN == [
            ("layers.0.experts.w13_weight", (2, 4), "tp32"),
            ("layers.0.experts.scale", (4,), "tp32"),
        ]
    finally:
        sys.modules.pop(pkg, None)


def test_observe_model_loader_restores_loader_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-stream errors restore both class methods and prior instance attributes."""
    pkg = _install_obs_plugin(
        tmp_path,
        monkeypatch,
        "obs_err_pkg",
        textwrap.dedent(
            """\
            from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

            def load_tensor(name, tensor, options):
                if name == "boom":
                    raise RuntimeError("hook failed")

            SPEC = TpuMonoSpec(
                "obs_err", "1", create=".a:c", load_tensor=load_tensor
            )
            """
        ),
    )
    try:
        cfg = _vllm_cfg(pkg)

        class _ClassLoader:
            """Loader with class-level `get_all_weights`."""

            def get_all_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
                """Yield one good weight then trigger hook error."""
                yield "ok", torch.ones((1,))
                yield "boom", torch.ones((1,))

        cls_loader = _ClassLoader()
        with (
            pytest.raises(RuntimeError, match="hook failed"),
            mkr.observe_model_loader(cfg, cls_loader),
        ):
            list(cls_loader.get_all_weights())
        assert "get_all_weights" not in vars(cls_loader)

        def _prior_get_all() -> Iterator[tuple[str, torch.Tensor]]:
            """Instance-level weight generator."""
            yield "boom", torch.ones((1,))

        ns_loader = SimpleNamespace(get_all_weights=_prior_get_all)
        with (
            pytest.raises(RuntimeError, match="hook failed"),
            mkr.observe_model_loader(cfg, ns_loader),
        ):
            list(ns_loader.get_all_weights())
        assert ns_loader.get_all_weights is _prior_get_all
    finally:
        sys.modules.pop(pkg, None)


def test_observe_model_loader_is_noop_without_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled megakernel or `load_tensor=None` leaves `model_loader` untouched."""
    pkg = _install_obs_plugin(
        tmp_path,
        monkeypatch,
        "obs_none_pkg",
        textwrap.dedent(
            """\
            from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

            SPEC = TpuMonoSpec("obs_none", "1", create=".a:c")
            """
        ),
    )
    try:

        class _Loader:
            """Stub loader."""

            def get_all_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
                """Yield one pair."""
                yield "w", torch.ones((2,))

        loader = _Loader()
        for cfg in (_vllm_cfg(None), _vllm_cfg(pkg)):
            assert salt.resolve_plugin_hook(cfg, "load_tensor") is None
            with mkr.observe_model_loader(cfg, loader):
                assert "get_all_weights" not in vars(loader)
                assert len(list(loader.get_all_weights())) == 1
    finally:
        sys.modules.pop(pkg, None)


def test_check_spec_resolves_lazy_load_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate lazy `load_tensor` strings and bad targets in check_plugin_spec."""
    monkeypatch.syspath_prepend(str(tmp_path))
    good_pkg = tmp_path / "mk_good_lt_pkg"
    good_pkg.mkdir()
    (good_pkg / "__init__.py").write_text(
        textwrap.dedent(
            """\
            from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

            SPEC = TpuMonoSpec(
                "good_lt",
                "1",
                create=".adapter:create",
                load_tensor=".adapter:load_tensor",
            )
            """
        ),
        encoding="utf-8",
    )
    (good_pkg / "adapter.py").write_text(
        "def create(view, options):\n    return None\n"
        "def load_tensor(name, tensor, options):\n    return None\n",
        encoding="utf-8",
    )
    bad_pkg = tmp_path / "mk_bad_lt_pkg"
    bad_pkg.mkdir()
    (bad_pkg / "__init__.py").write_text(
        textwrap.dedent(
            """\
            from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

            SPEC = TpuMonoSpec(
                "bad_lt",
                "1",
                create=".adapter:create",
                load_tensor=".adapter:missing_fn",
            )
            """
        ),
        encoding="utf-8",
    )
    (bad_pkg / "adapter.py").write_text(
        "def create(view, options):\n    return None\n",
        encoding="utf-8",
    )
    importlib.invalidate_caches()
    try:
        checked = testkit.check_plugin_spec("mk_good_lt_pkg")
        assert checked.load_tensor == ".adapter:load_tensor"
        with pytest.raises(MegakernelConfigError, match="can't resolve load_tensor"):
            testkit.check_plugin_spec("mk_bad_lt_pkg")
    finally:
        for prefix in ("mk_good_lt_pkg", "mk_bad_lt_pkg"):
            for mod_name in [
                m for m in sys.modules if m == prefix or m.startswith(f"{prefix}.")
            ]:
                sys.modules.pop(mod_name, None)


def test_runner_load_model_observes_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`TPUModelRunner.load_model` wraps `load_model` in `observe_model_loader`."""
    runner_path = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    src = runner_path.read_text(encoding="utf-8")
    load_fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    expected_ctx = (
        "megakernel_runtime.observe_model_loader(self.vllm_config, model_loader)"
    )
    with_nodes = [
        w
        for w in ast.walk(load_fn)
        if isinstance(w, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr) == expected_ctx
            for item in w.items
        )
    ]
    assert len(with_nodes) == 1
    body_src = ast.unparse(with_nodes[0])
    assert "model_loader.load_model" in body_src
    assert "load_model(" in body_src

    pkg = _install_obs_plugin(
        tmp_path,
        monkeypatch,
        "obs_runner_pkg",
        textwrap.dedent(
            """\
            from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

            NAMES = []

            def load_tensor(name, tensor, options):
                NAMES.append(name)

            SPEC = TpuMonoSpec(
                "obs_runner", "1", create=".a:c", load_tensor=load_tensor
            )
            """
        ),
    )
    try:
        mod = importlib.import_module(pkg)
        cfg = _vllm_cfg(pkg)

        class _StubLoader:
            """Minimal `BaseModelLoader`-shaped loader."""

            def get_all_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
                """Stream two checkpoint weights."""
                yield "embed.weight", torch.ones((2, 2))
                yield "lm_head.weight", torch.ones((2, 2))

            def load_model(
                self, *, vllm_config: object, model_config: object
            ) -> list[str]:
                """Consume `get_all_weights` as `DefaultModelLoader` does."""
                _ = (vllm_config, model_config)
                return [name for name, _ in self.get_all_weights()]

        stub_loader = _StubLoader()
        with mkr.observe_model_loader(cfg, stub_loader):
            loaded = stub_loader.load_model(vllm_config=cfg, model_config=None)
        assert loaded == ["embed.weight", "lm_head.weight"]
        assert mod.NAMES == ["embed.weight", "lm_head.weight"]
        assert "get_all_weights" not in vars(stub_loader)
    finally:
        sys.modules.pop(pkg, None)
