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
"""`ProgramDriver` internals: the TP-rank mesh and the bind agreement before
warmup."""

from __future__ import annotations

import logging
from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import vllm_torchtpu.megakernel.adapter.agreement as agreement_mod
import vllm_torchtpu.megakernel.adapter.driver as driver_mod


@pytest.fixture(autouse=True)
def _std_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the driver's lazy vLLM logger to `logging` when vLLM is absent."""
    try:
        driver_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            driver_mod.logger, "_logger", logging.getLogger(driver_mod.logger._name)
        )


_RING_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)


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


_TORCH_TPU_MODULES = ("torch_tpu", "torch_tpu._internal", "torch_tpu._internal.pallas")


def _fake_torch_tpu(monkeypatch: pytest.MonkeyPatch, get_pallas_mesh: Any) -> None:
    """An importable `torch_tpu` whose `_internal.pallas` has `get_pallas_mesh`
    (every level is set, so a real, already imported `torch_tpu` is shadowed)."""
    import sys
    import types

    for name in _TORCH_TPU_MODULES:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(
        sys.modules["torch_tpu._internal.pallas"],
        "get_pallas_mesh",
        get_pallas_mesh,
        raising=False,
    )


def _never_called(**_kwargs: Any) -> Any:
    raise AssertionError("torch_tpu's get_pallas_mesh called off TPU")


def test_get_pallas_mesh_fn_is_none_without_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setattr(driver_mod.jax, "default_backend", lambda: "tpu")
    for name in _TORCH_TPU_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    assert driver_mod._get_pallas_mesh_fn() is None


def test_get_pallas_mesh_fn_is_torch_tpus_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch_tpu(monkeypatch, _never_called)
    monkeypatch.setattr(driver_mod.jax, "default_backend", lambda: "tpu")
    assert driver_mod._get_pallas_mesh_fn() is _never_called


def test_cpu_backend_ignores_an_installed_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU process (`JAX_PLATFORMS=cpu`, as CI's unit lane) with `torch_tpu`
    installed builds its mesh from JAX's CPU devices, never torch_tpu's."""
    assert jax.default_backend() == "cpu"
    _fake_torch_tpu(monkeypatch, _never_called)
    assert driver_mod._get_pallas_mesh_fn() is None
    mesh = driver_mod._build_tp_rank_mesh(1)
    assert tuple(mesh.axis_names) == ("tp",) and mesh.devices.size == 1


class _FakeTPGroup:
    """Stands in for the TP group's CPU process group: `all_reduce(MAX)` folds
    in the votes of the other ranks, which the test sets."""

    def __init__(self, peer_votes: list[int], peer_tokens: list[int] | None = None):
        self.peer_votes = peer_votes
        self.peer_tokens = peer_tokens
        self.calls = 0

    def all_reduce_max(self, t: Any, group: Any) -> None:
        assert group is self
        self.calls += 1
        mine = int(t[1])
        toks = [mine] if self.peer_tokens is None else [mine, *self.peer_tokens]
        if self.peer_tokens is None:
            toks = [mine] * (1 + len(self.peer_votes))
        t[0] = max([int(t[0]), *self.peer_votes])
        t[1] = max(toks)
        t[2] = max(-x for x in toks)


@pytest.fixture
def fake_tp_group(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(
        peer_votes: list[int], peer_tokens: list[int] | None = None
    ) -> _FakeTPGroup:
        g = _FakeTPGroup(peer_votes, peer_tokens)
        monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: g)
        monkeypatch.setattr(agreement_mod, "_all_reduce_max", g.all_reduce_max)
        return g

    return make


def _vote(code: str) -> int:
    return driver_mod.DECLINE_VOTE_CODES.index(code) + 1


def _bind_driver(bind: Any) -> Any:
    from types import SimpleNamespace

    from vllm_torchtpu.megakernel.adapter.plugin import RunnerView

    program = SimpleNamespace(display_name="test")
    drv = driver_mod.ProgramDriver(program, view=RunnerView(None))
    drv._bound = True
    drv._steps = {(1, 8): None, (4, 4): None}
    drv.bind = bind
    drv.warmed = []
    drv.warmup = lambda runner: drv.warmed.append(runner)
    return drv


def test_bind_and_warmup_peer_decline_skips_warmup(fake_tp_group: Any) -> None:
    from types import SimpleNamespace

    g = fake_tp_group([_vote("conv_state_dtype_unsupported")])
    drv = _bind_driver(lambda runner: None)
    runner = SimpleNamespace(_megakernel=drv)
    drv.bind_and_warmup(runner)
    assert g.calls == 1 and drv.warmed == [] and not drv._bound
    assert isinstance(runner._megakernel, driver_mod.DeclinedDriver)
    assert (
        runner._megakernel.decline.code == "peer_declined:conv_state_dtype_unsupported"
    )

    g = fake_tp_group([0])
    drv = _bind_driver(lambda runner: None)
    runner = SimpleNamespace(_megakernel=drv)
    drv.bind_and_warmup(runner)
    assert g.calls == 1 and drv.warmed == [runner] and runner._megakernel is drv


def test_bind_and_warmup_votes_before_raising(
    fake_tp_group: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rank whose bind raises still joins the agreement (as `bind_error`)."""
    from types import SimpleNamespace

    votes: list[int] = []
    g = fake_tp_group([0])
    orig = g.all_reduce_max

    def spy(t: Any, group: Any) -> None:
        votes.append(int(t[0]))
        orig(t, group)

    monkeypatch.setattr(agreement_mod, "_all_reduce_max", spy)

    def bind(runner: Any) -> None:
        raise ValueError("internal bug")

    drv = _bind_driver(bind)
    with pytest.raises(ValueError, match="internal bug"):
        drv.bind_and_warmup(SimpleNamespace(_megakernel=drv))
    assert votes == [_vote("bind_error")] and drv.warmed == []


def test_bind_and_warmup_agrees_on_the_bucket_set(fake_tp_group: Any) -> None:
    from types import SimpleNamespace

    # bind_and_warmup agrees on the compiled bucket set before warmup.
    g = fake_tp_group([0], peer_tokens=[driver_mod.buckets_token([(1, 8)])])
    drv = _bind_driver(lambda runner: None)
    runner = SimpleNamespace(_megakernel=drv)
    drv.bind_and_warmup(runner)
    assert g.calls == 1 and drv.warmed == []
    assert runner._megakernel.decline.code == "bucket_set_mismatch"


def test_bind_and_warmup_agreement_error_propagates_after_one_collective(
    fake_tp_group: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the agreement itself raises (e.g. a gloo timeout), the rank raises
    after that one collective instead of voting again in a cleanup path."""
    from types import SimpleNamespace

    calls: list[int] = []

    def boom(t: Any, group: Any) -> None:
        calls.append(int(t[0]))
        raise RuntimeError("gloo timeout")

    fake_tp_group([0])
    monkeypatch.setattr(agreement_mod, "_all_reduce_max", boom)
    drv = _bind_driver(lambda runner: None)
    with pytest.raises(RuntimeError, match="gloo timeout"):
        drv.bind_and_warmup(SimpleNamespace(_megakernel=drv))
    assert calls == [0] and drv.warmed == []
