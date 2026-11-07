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
"""`ProgramDriver` internals: the TP-rank mesh, `try_forward` routing and
the bind agreement before warmup."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod
import vllm_torchtpu.megakernel_runtime.driver as driver_mod
from vllm_torchtpu.megakernel.runtime.stats import BatchShape


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


def test_get_pallas_mesh_fn_is_none_without_torch_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch_tpu", None)
    assert driver_mod._get_pallas_mesh_fn() is None


def _step_runner(
    num_reqs: int | None = None,
    qsl: list[int] | None = None,
    *,
    offsets: bool = False,
    comb_key: tuple[Any, int, int] | None = None,
    comb_cpu: list[int] | None = None,
) -> Any:
    """The runner attributes `_inspect_step_detail` reads during a step."""
    return SimpleNamespace(
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=num_reqs),
        mamba_slot_read_offsets=torch.zeros(4, dtype=torch.int32) if offsets else None,
        _cached_comb_key=comb_key,
        _combined_request_distribution_cpu=(
            None if comb_cpu is None else torch.tensor(comb_cpu, dtype=torch.int32)
        ),
        query_start_loc_np=None if qsl is None else np.asarray(qsl, dtype=np.int32),
        query_start_loc_cpu=None,
    )


def _i32(*vals: int) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int32)


_NO_MD = SimpleNamespace()


@pytest.mark.parametrize(
    "runner, attn_md, m0, want",
    [
        # The runner decides: its request count and host query_start_loc.
        (_step_runner(3, [0, 1, 2, 3]), _NO_MD, _NO_MD, BatchShape(True, 3, 1, "ok")),
        # A multi-token step without slot read offsets is a prefill.
        (
            _step_runner(2, [0, 4, 8]),
            _NO_MD,
            _NO_MD,
            BatchShape(False, 2, 0, "stock_prefill"),
        ),
        # Slot read offsets: the combined distribution (decode, total) from
        # the cached key; equal counts serve the verify step.
        (
            _step_runner(2, [0, 4, 8], offsets=True, comb_key=("k", 2, 2)),
            _NO_MD,
            _NO_MD,
            BatchShape(True, 2, 4, "ok"),
        ),
        (
            _step_runner(2, [0, 5, 9], offsets=True, comb_key=("k", 0, 2)),
            _NO_MD,
            _NO_MD,
            BatchShape(False, 2, 0, "stock_prefill"),
        ),
        (
            _step_runner(2, [0, 4, 9], offsets=True, comb_key=("k", 1, 2)),
            _NO_MD,
            _NO_MD,
            BatchShape(False, 2, 0, "stock_mixed"),
        ),
        # ... or from the CPU distribution tensor (indices 3 and 5), with the
        # request count taken from it.
        (
            _step_runner(None, [0, 4, 8], offsets=True, comb_cpu=[0, 0, 0, 2, 0, 2]),
            _NO_MD,
            _NO_MD,
            BatchShape(True, 2, 4, "ok"),
        ),
        # Without a host query_start_loc the distribution alone can decline.
        (
            _step_runner(None, None, offsets=True, comb_key=("k", 1, 2)),
            _NO_MD,
            _NO_MD,
            BatchShape(False, 2, 0, "stock_mixed"),
        ),
        # CPU attention metadata (no runner): the request distribution, the
        # sequence count and query_start_loc_cpu.
        (
            None,
            SimpleNamespace(
                request_distribution=_i32(3, 0, 3),
                num_seqs=3,
                query_start_loc_cpu=_i32(0, 1, 2, 3),
            ),
            _NO_MD,
            BatchShape(True, 3, 1, "ok"),
        ),
        # The recurrent layer's distribution comes first.
        (
            None,
            SimpleNamespace(request_distribution=_i32(2, 0, 2), num_seqs=2),
            SimpleNamespace(mamba_request_distribution=_i32(1, 0, 2)),
            BatchShape(False, 2, 0, "stock_mixed"),
        ),
        # A CPU query_start_loc tensor serves as the host copy.
        (
            None,
            SimpleNamespace(num_seqs=2, query_start_loc=_i32(0, 1, 2)),
            _NO_MD,
            BatchShape(True, 2, 1, "ok"),
        ),
        # No host query_start_loc anywhere fails closed.
        (
            _step_runner(2, None),
            SimpleNamespace(num_seqs=2),
            _NO_MD,
            BatchShape(False, 0, 0, "stock_no_qsl"),
        ),
    ],
    ids=[
        "runner_decode",
        "runner_multi_token_without_offsets",
        "offsets_comb_key_verify",
        "offsets_comb_key_prefill",
        "offsets_comb_key_mixed",
        "offsets_comb_cpu_tensor",
        "offsets_without_query_start_loc",
        "metadata_decode",
        "metadata_recurrent_distribution_first",
        "metadata_query_start_loc_tensor",
        "no_host_query_start_loc",
    ],
)
def test_inspect_step_detail(runner: Any, attn_md: Any, m0: Any, want: Any) -> None:
    assert driver_mod._inspect_step_detail(attn_md, m0, runner) == want


def _bare_driver(monkeypatch: pytest.MonkeyPatch, *, eligible: bool = True) -> Any:
    """A `ProgramDriver` with only the state `try_forward` touches."""
    import torch

    from vllm_torchtpu.megakernel import runtime as rt

    drv = object.__new__(driver_mod.ProgramDriver)
    drv._pending = None
    drv.ab_calibrator = rt.MKABCalibrator(enabled=False, period_s=0.0)
    drv._backbone = None
    calls: list[str] = []

    def route(*_a: Any, **_k: Any) -> bool:
        calls.append("can_run")
        drv._pending = ("prepared",)
        return eligible

    def run(input_ids: Any, positions: Any, **_k: Any) -> Any:
        calls.append("run")
        return torch.full((int(positions.shape[-1]), 4), 7.0)

    monkeypatch.setattr(drv, "_route", route, raising=False)
    monkeypatch.setattr(
        driver_mod.ProgramDriver,
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
        drv._pending = ("stale",)
        assert drv.try_forward(runner, StepContext(ids, pos)) == (False, None)
        assert drv._pending is None
    assert drv.calls == []
    hit, out = drv.try_forward(runner, StepContext(ids, pos))
    assert hit and tuple(out.shape) == (2, 4)
    assert drv.calls == ["can_run", "run"]


def test_try_forward_unpacks_step_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.megakernel_runtime import StepContext

    drv = _bare_driver(monkeypatch)
    seen: dict[str, Any] = {}

    def route(ctx: Any, runner: Any) -> bool:
        seen.update(ctx=ctx, runner=runner)
        return False

    monkeypatch.setattr(drv, "_route", route, raising=False)
    runner = SimpleNamespace()
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    emb, it = object(), object()
    ctx = StepContext(ids, pos, inputs_embeds=emb, intermediate_tensors=it)
    assert drv.try_forward(runner, ctx) == (False, None)
    # The runner's step context reaches the routing decision unchanged.
    assert seen["ctx"] is ctx and seen["runner"] is runner
    assert (ctx.input_ids, ctx.positions) == (ids, pos)
    assert ctx.inputs_embeds is emb and ctx.intermediate_tensors is it


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

    from vllm_torchtpu.megakernel_runtime.plugin import RunnerView

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
