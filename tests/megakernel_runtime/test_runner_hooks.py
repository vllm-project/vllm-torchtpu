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
    assert driver_mod._build_tp_rank_mesh(8, fallback_mesh=runner_mesh) is None


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
