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
"""Runner-hook tests for `megakernel_runtime`: mesh order, suppression, step context."""

from __future__ import annotations

import logging
from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import vllm_torchtpu.megakernel_runtime.driver as driver_mod

# The order a runner mesh may list 8 devices in (ICI ring order).
_RING_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)


@pytest.fixture(autouse=True)
def _std_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the driver's lazy vLLM logger to `logging` when vLLM is absent."""
    try:
        driver_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            driver_mod.logger, "_logger", logging.getLogger(driver_mod.logger._name)
        )


def _devices(n: int) -> list[Any]:
    devs = jax.devices()
    if len(devs) < n:
        pytest.skip(f"needs {n} JAX devices, found {len(devs)}")
    return list(devs[:n])


def _ids(mesh: Any) -> list[int]:
    return [int(d.id) for d in np.asarray(mesh.devices).flat]


def _fake_pallas(mesh: Any, calls: list[dict[str, Any]] | None = None) -> Any:
    def get_pallas_mesh(**kwargs: Any) -> Any:
        if calls is not None:
            calls.append(kwargs)
        return mesh

    return lambda: get_pallas_mesh


def test_tp_mesh_prefers_pallas_mesh_over_permuted_runner_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devs = _devices(8)
    pallas_mesh = Mesh(np.asarray(devs), ("tp",))
    runner_mesh = Mesh(np.asarray([devs[i] for i in _RING_ORDER]), ("tp",))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        driver_mod, "_get_pallas_mesh_fn", _fake_pallas(pallas_mesh, calls)
    )
    mesh = driver_mod._build_tp_rank_mesh(8, fallback_mesh=runner_mesh)
    assert mesh is pallas_mesh
    assert _ids(mesh) == [int(d.id) for d in devs]
    assert _ids(mesh) != _ids(runner_mesh)
    assert calls == [{"axis_names": ("tp",)}]


@pytest.mark.parametrize("bad", ["size", "axes", "raises"])
def test_tp_mesh_with_torch_tpu_fails_closed(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    devs = _devices(8)
    runner_mesh = Mesh(np.asarray(devs), ("tp",))
    if bad == "raises":

        def get_pallas_mesh(**_kwargs: Any) -> Any:
            raise RuntimeError("no Pallas runtime")

        monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: get_pallas_mesh)
    else:
        pallas_mesh = (
            Mesh(np.asarray(devs[:4]), ("tp",))
            if bad == "size"
            else Mesh(np.asarray(devs), ("x",))
        )
        monkeypatch.setattr(
            driver_mod, "_get_pallas_mesh_fn", _fake_pallas(pallas_mesh)
        )
    # Never falls back to the runner mesh once torch_tpu is present.
    with pytest.raises(driver_mod.MeshDecline) as exc_info:
        driver_mod._build_tp_rank_mesh(8, fallback_mesh=runner_mesh)
    res = exc_info.value.result
    assert not res.eligible
    want = "pallas_mesh_unavailable" if bad == "raises" else "pallas_mesh_mismatch"
    assert res.code == want and res.message


def test_tp_mesh_without_torch_tpu_uses_runner_mesh_then_jax_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devs = _devices(8)
    monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: None)
    tp_mesh = Mesh(np.asarray(devs), ("tp",))
    assert driver_mod._build_tp_rank_mesh(8, fallback_mesh=tp_mesh) is tp_mesh

    ring_2d = Mesh(np.asarray([devs[i] for i in _RING_ORDER]).reshape(2, 4), ("a", "b"))
    mesh = driver_mod._build_tp_rank_mesh(8, fallback_mesh=ring_2d)
    assert mesh is not None and tuple(mesh.axis_names) == ("tp",)
    assert _ids(mesh) == [int(devs[i].id) for i in _RING_ORDER]

    mesh = driver_mod._build_tp_rank_mesh(4, fallback_mesh=tp_mesh)
    assert mesh is not None and tuple(mesh.axis_names) == ("tp",)
    assert _ids(mesh) == [int(d.id) for d in devs[:4]]


def test_get_pallas_mesh_fn_is_none_without_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch_tpu", None)
    assert driver_mod._get_pallas_mesh_fn() is None


# --- Suppression -------------------------------------------------------------


def test_suppressed_nests_and_restores_on_error() -> None:
    from types import SimpleNamespace
    from unittest import mock

    from vllm_torchtpu import megakernel_runtime as mkr

    runner = SimpleNamespace()
    assert not mkr.is_suppressed(runner) and not mkr.is_suppressed(None)
    with mkr.suppressed(runner):
        assert mkr.is_suppressed(runner)
        with mkr.suppressed(runner):
            assert mkr.is_suppressed(runner)
        assert mkr.is_suppressed(runner)
    assert not mkr.is_suppressed(runner)
    with pytest.raises(RuntimeError, match="boom"), mkr.suppressed(runner):
        raise RuntimeError("boom")
    assert not mkr.is_suppressed(runner)
    # Scoped to one runner; attribute-synthesizing mocks are never suppressed.
    other = SimpleNamespace()
    with mkr.suppressed(runner):
        assert not mkr.is_suppressed(other)
    assert not mkr.is_suppressed(mock.MagicMock())


def _bare_driver(monkeypatch: pytest.MonkeyPatch, *, eligible: bool = True) -> Any:
    """A `MegakernelDriver` with only the state `try_forward` touches."""
    import torch

    from vllm_torchtpu.megakernel import runtime as rt

    drv = object.__new__(driver_mod.MegakernelDriver)
    drv._prepared_step = None
    drv.ab_calibrator = rt.MKABCalibrator(enabled=False, period_s=0.0)
    drv.backbone = object()
    calls: list[str] = []

    def can_run(*_a: Any, **_k: Any) -> bool:
        calls.append("can_run")
        drv._prepared_step = ("prepared",)
        return eligible

    def run(input_ids: Any, positions: Any, **_k: Any) -> Any:
        calls.append("run")
        return torch.full((int(positions.shape[-1]), 4), 7.0)

    monkeypatch.setattr(drv, "can_run", can_run, raising=False)
    monkeypatch.setattr(
        driver_mod.MegakernelDriver,
        "__call__",
        lambda self, *a, **k: run(*a, **k),
    )
    drv.calls = calls
    return drv


def test_try_forward_never_engages_while_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu import megakernel_runtime as mkr
    from vllm_torchtpu.megakernel_runtime import StepContext

    drv = _bare_driver(monkeypatch)
    runner = SimpleNamespace()
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    with mkr.suppressed(runner):
        drv._prepared_step = ("stale",)
        assert drv.try_forward(runner, StepContext(ids, pos)) == (False, None)
        assert drv._prepared_step is None
    assert drv.calls == []
    hit, out = drv.try_forward(runner, StepContext(ids, pos))
    assert hit and tuple(out.shape) == (2, 4)
    assert drv.calls == ["can_run", "run"]


def test_runner_dummy_run_forward_is_suppressed() -> None:
    """`TPUModelRunner._dummy_run` calls `forward_model` inside `suppressed(self)`."""
    import ast
    from pathlib import Path

    import vllm_torchtpu

    src = (Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run"
    )

    def _is_suppress(item: ast.withitem) -> bool:
        call = item.context_expr
        return (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "megakernel_runtime.suppressed"
            and [ast.unparse(a) for a in call.args] == ["self"]
        )

    def _calls_forward(node: ast.AST) -> bool:
        return any(
            isinstance(c, ast.Call) and ast.unparse(c.func) == "self.forward_model"
            for c in ast.walk(node)
        )

    fwd_calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "self.forward_model"
    ]
    assert fwd_calls, "_dummy_run no longer calls self.forward_model"
    guarded = [
        w
        for w in ast.walk(fn)
        if isinstance(w, ast.With)
        and any(_is_suppress(i) for i in w.items)
        and _calls_forward(w)
    ]
    n_guarded = sum(
        1
        for w in guarded
        for c in ast.walk(w)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "self.forward_model"
    )
    assert n_guarded == len(fwd_calls)


# --- Step context ------------------------------------------------------------


def test_try_forward_unpacks_step_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.megakernel_runtime import StepContext

    drv = _bare_driver(monkeypatch)
    seen: dict[str, Any] = {}

    def can_run(input_ids: Any, positions: Any, **kw: Any) -> bool:
        seen.update(input_ids=input_ids, positions=positions, **kw)
        return False

    monkeypatch.setattr(drv, "can_run", can_run, raising=False)
    runner = SimpleNamespace()
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    emb, it = object(), object()
    ctx = StepContext(ids, pos, inputs_embeds=emb, intermediate_tensors=it)
    assert drv.try_forward(runner, ctx) == (False, None)
    assert seen["input_ids"] is ids and seen["positions"] is pos
    assert seen["inputs_embeds"] is emb and seen["intermediate_tensors"] is it
    assert seen["runner"] is runner


@pytest.mark.parametrize("model_kwargs", [None, {}, {"extra": 1}])
def test_try_forward_declines_unconsumed_model_kwargs(
    monkeypatch: pytest.MonkeyPatch, model_kwargs: Any
) -> None:
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.megakernel_runtime import StepContext

    drv = _bare_driver(monkeypatch)
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    ctx = StepContext(ids, pos, model_kwargs=model_kwargs)
    hit, out = drv.try_forward(SimpleNamespace(), ctx)
    if model_kwargs:
        assert (hit, out) == (False, None) and drv.calls == []
    else:
        assert hit and drv.calls == ["can_run", "run"]


class _FakeIntermediateTensors:
    pass


def _runner_forward_model() -> Any:
    """`TPUModelRunner.forward_model`; compiled from source without torch_tpu."""
    try:
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    except ImportError:
        pass
    else:
        return TPUModelRunner.forward_model
    import ast
    import textwrap
    from pathlib import Path

    import vllm_torchtpu
    from vllm_torchtpu import megakernel_runtime

    path = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    src = path.read_text(encoding="utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "forward_model"
    )
    fn_src = textwrap.dedent(ast.get_source_segment(src, fn) or "")
    ns: dict[str, Any] = {
        "megakernel_runtime": megakernel_runtime,
        "IntermediateTensors": _FakeIntermediateTensors,
    }
    exec(compile(fn_src, str(path), "exec"), ns)
    return ns["forward_model"]


def _raw_outputs() -> list[Any]:
    import torch

    h = torch.arange(6.0).reshape(3, 2)
    aux = [torch.ones(3, 2)]
    return [h, [h], (h,), (h, aux), [h, aux], _FakeIntermediateTensors()]


@pytest.mark.parametrize("raw_idx", range(6))
def test_runner_normalizes_megakernel_output_like_stock(raw_idx: int) -> None:
    """A megakernel step's output goes through the stock normalization."""
    from types import SimpleNamespace

    import torch

    forward_model = _runner_forward_model()
    raw = _raw_outputs()[raw_idx]
    stock_calls: list[dict[str, Any]] = []

    def model(**kw: Any) -> Any:
        stock_calls.append(kw)
        return raw

    class _Driver:
        def __init__(self, hit: bool) -> None:
            self.hit, self.ctx = hit, None

        def try_forward(self, runner: Any, ctx: Any) -> tuple[bool, Any]:
            self.ctx = ctx
            return (True, raw) if self.hit else (False, None)

    ids = torch.zeros((3,), dtype=torch.int32)
    pos = torch.zeros((3,), dtype=torch.int32)
    stock = forward_model(SimpleNamespace(_megakernel=None, model=model), ids, pos)
    assert len(stock_calls) == 1
    drv = _Driver(hit=True)
    mk = forward_model(SimpleNamespace(_megakernel=drv, model=model), ids, pos)
    assert len(stock_calls) == 1, "a megakernel hit must not run the stock model"
    assert drv.ctx.input_ids is ids and drv.ctx.positions is pos
    assert type(mk) is type(stock) and len(mk) == len(stock) == 2
    assert mk[0] is stock[0]
    assert mk[1] is stock[1]


def test_runner_declined_step_passes_model_kwargs_to_stock() -> None:
    from types import SimpleNamespace

    import torch

    forward_model = _runner_forward_model()
    stock_calls: list[dict[str, Any]] = []
    h = torch.zeros(2, 2)

    def model(**kw: Any) -> Any:
        stock_calls.append(kw)
        return h

    class _Decline:
        ctx: Any = None

        def try_forward(self, runner: Any, ctx: Any) -> tuple[bool, Any]:
            self.ctx = ctx
            return False, None

    drv = _Decline()
    runner = SimpleNamespace(_megakernel=drv, model=model)
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    it = object()
    out = forward_model(
        runner, ids, pos, intermediate_tensors=it, model_kwargs={"extra": 1}
    )
    assert out[0] is h and out[1] is None
    assert drv.ctx.model_kwargs == {"extra": 1}
    assert drv.ctx.intermediate_tensors is it
    assert stock_calls == [
        dict(
            input_ids=ids,
            positions=pos,
            inputs_embeds=None,
            extra=1,
            intermediate_tensors=it,
        )
    ]
