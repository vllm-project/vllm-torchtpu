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
"""Step routing: the batch shape a step is inspected as, `try_forward`
(suppression, the step context, unconsumed model inputs), the routing
decision with its declines, the `can_run` / `__call__` hand-off, the
text-token promotion, the A/B calibrator's arms, and the runner shapes a
bucket is warmed up with."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.adapter import stepping
from vllm_torchtpu.megakernel.adapter.hooks import StepContext
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    RunnerShape,
    StepDecision,
)
from vllm_torchtpu.megakernel.core.runtime.stats import (
    BatchShape,
    MKABCalibrator,
    StepMixTracker,
)


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
    assert stepping._inspect_step_detail(attn_md, m0, runner) == want


def _bare_driver(monkeypatch: pytest.MonkeyPatch, *, eligible: bool = True) -> Any:
    """A `StepRouting` with only the state `try_forward` touches."""
    import torch

    from vllm_torchtpu.megakernel.core import runtime as rt

    class _Bare(stepping.StepRouting):
        pass

    drv = object.__new__(_Bare)
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
    monkeypatch.setattr(_Bare, "__call__", lambda self, *a, **k: run(*a, **k))
    drv.calls = calls
    return drv


def test_try_forward_never_engages_while_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.megakernel import adapter as mkr
    from vllm_torchtpu.megakernel.adapter import StepContext

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

    from vllm_torchtpu.megakernel.adapter import StepContext

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

    from vllm_torchtpu.megakernel.adapter import StepContext

    drv = _bare_driver(monkeypatch)
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    ctx = StepContext(ids, pos, model_kwargs=model_kwargs)
    hit, out = drv.try_forward(SimpleNamespace(), ctx)
    if model_kwargs:
        assert (hit, out) == (False, None) and drv.calls == []
    else:
        assert hit and drv.calls == ["can_run", "run"]


def test_runner_shapes_without_paddings() -> None:
    shapes = stepping.runner_shapes(
        Bucket(4, 16), (), (), ckpt_window=1, max_pages_per_seq=9
    )
    assert shapes == [RunnerShape(16, 4, 9, 4)]
    shapes = stepping.runner_shapes(
        Bucket(1, 64), (16, 32), ((4, 2), (1, 8)), ckpt_window=3, max_pages_per_seq=9
    )
    assert shapes == [RunnerShape(32, 4, 2, 3), RunnerShape(32, 1, 8, 3)]


# --- routing through a stub program ---------------------------------------------

_MK = StepDecision(True, "mk_t1_b4", 3, 1, 4)
_INELIGIBLE = StepDecision(False, "stock_big", 3, 1)


class _Program:
    """Records `decide` / `forward`; `decide` returns `decision`."""

    def __init__(self, decision: StepDecision | None) -> None:
        self.decision, self.calls = decision, []

    def decide(self, ctx: StepContext, fwd: Any) -> StepDecision | None:
        self.calls.append(("decide", ctx))
        return self.decision

    def forward(self, decision: Any, ctx: StepContext, fwd: Any) -> torch.Tensor:
        self.calls.append(("forward", decision, fwd.num_tokens))
        return ctx.input_ids.float() * 10


class _Driver(stepping.StepRouting):
    """The state a driver sets for `StepRouting`, around a stub program."""

    def __init__(
        self,
        program: _Program,
        *,
        ab: bool = False,
        bound: bool = True,
        backbone: Any = None,
        text_embeds: bool = False,
    ) -> None:
        self.program, self.tp_rank, self._display_name = program, 0, "Stub"
        self._backbone, self._text_embeds = backbone, text_embeds
        self._steps, self._bound = {(1, 4): lambda x: x}, bound
        self._pending = self._inspect = self._inspect_runner = None
        self.step_mix_tracker = StepMixTracker(period_s=3600.0)
        self.ab_calibrator = MKABCalibrator(enabled=ab, period_s=3600.0)


def _stock_runner(**kw: Any) -> Any:
    """A runner whose stock model returns `-input_ids` and records its calls."""
    calls: list[dict[str, Any]] = []

    def model(**k: Any) -> torch.Tensor:
        calls.append(k)
        return -k["input_ids"].float()

    return SimpleNamespace(
        model=model, stock_calls=calls, device=torch.device("cpu"), **kw
    )


@pytest.fixture
def fwd_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(stepping, "get_forward_context", lambda: ctx)


_IDS = torch.tensor([5, 6, 7], dtype=torch.int32)
_POS = torch.zeros(3, dtype=torch.int32)


@pytest.mark.parametrize("ab", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "not_bound",
        "no_input_ids",
        "inputs_embeds",
        "intermediate_tensors",
        "aux_hidden_state_layers",
        "position_ids_override",
    ],
)
def test_route_declines_before_decide(fwd_ctx: None, case: str, ab: bool) -> None:
    """Each early decline: stock, no `decide`, nothing pending or counted, and
    no A/B timing (`last_decision` stays `stock_early`)."""
    prog = _Program(_MK)
    drv = _Driver(
        prog,
        ab=ab,
        bound=case != "not_bound",
        backbone=(
            SimpleNamespace(aux_hidden_state_layers=(2,))
            if case == "aux_hidden_state_layers"
            else None
        ),
    )
    bctx = SimpleNamespace(
        position_ids_override=_POS if case == "position_ids_override" else None
    )
    runner = _stock_runner(_attn_metadata_builder_ctx=bctx)
    ctx = StepContext(
        None if case in ("no_input_ids", "inputs_embeds") else _IDS,
        _POS,
        inputs_embeds=torch.zeros(3, 2) if case == "inputs_embeds" else None,
        intermediate_tensors=object() if case == "intermediate_tensors" else None,
    )
    drv._pending = ("stale",)
    assert drv.try_forward(runner, ctx) == (False, None)
    assert drv._pending is None
    assert drv.ab_calibrator.last_decision == ("stock_early", "stock")
    assert prog.calls == [] and runner.stock_calls == []
    assert drv.step_mix_tracker.counts == {}


@pytest.mark.parametrize("ab", [False, True])
def test_route_decide_none_is_stock_and_uncounted(fwd_ctx: None, ab: bool) -> None:
    prog = _Program(None)
    drv = _Driver(prog, ab=ab)
    runner = _stock_runner()
    assert drv.try_forward(runner, StepContext(_IDS, _POS)) == (False, None)
    assert [c[0] for c in prog.calls] == ["decide"]
    assert drv._pending is None and runner.stock_calls == []
    assert drv.ab_calibrator.last_decision == ("stock_early", "stock")
    assert drv.step_mix_tracker.counts == {}


@pytest.mark.parametrize("ab", [False, True])
def test_route_ineligible_is_counted_and_timed_on_the_stock_arm(
    fwd_ctx: None, ab: bool
) -> None:
    """An ineligible step counts under its reason. With A/B on, `try_forward`
    runs and times the stock model itself."""
    prog = _Program(_INELIGIBLE)
    drv = _Driver(prog, ab=ab)
    runner = _stock_runner()
    hit, out = drv.try_forward(runner, StepContext(_IDS, _POS))
    assert [c[0] for c in prog.calls] == ["decide"]
    assert drv._pending is None
    assert drv.ab_calibrator.last_decision == ("stock_big", "stock")
    assert drv.step_mix_tracker.counts == {"stock_big": 1}
    if ab:
        assert hit and torch.equal(out, -_IDS.float())
        assert runner.stock_calls == [
            dict(input_ids=_IDS, positions=_POS, inputs_embeds=None)
        ]
        assert set(drv.ab_calibrator._cumulative["stock_big_p3"]) == {"stock"}
    else:
        assert (hit, out) == (False, None) and runner.stock_calls == []


def test_eligible_step_serves_on_the_program(fwd_ctx: None) -> None:
    prog = _Program(_MK)
    drv = _Driver(prog)
    runner = _stock_runner()
    hit, out = drv.try_forward(runner, StepContext(_IDS, _POS))
    assert hit and torch.equal(out, _IDS.float() * 10)
    assert [c[0] for c in prog.calls] == ["decide", "forward"]
    assert prog.calls[1][1:] == (_MK, 3)
    # `__call__` consumed the step `_route` left pending.
    assert drv._pending is None and runner.stock_calls == []
    assert drv.ab_calibrator.last_decision == ("mk_t1_b4", "mk")
    assert drv.step_mix_tracker.counts == {"mk_t1_b4": 1}


def test_can_run_leaves_the_step_pending_for_call(fwd_ctx: None) -> None:
    prog = _Program(_MK)
    drv = _Driver(prog)
    runner = _stock_runner()
    assert drv.can_run(_IDS, _POS, runner=runner)
    decision, fwd, ctx = drv._pending
    assert decision is _MK and ctx.input_ids is _IDS and fwd.num_tokens is None
    out = drv(_IDS, _POS, num_tokens=2, runner=runner)
    assert torch.equal(out, _IDS.float() * 10)
    assert prog.calls[-1] == ("forward", _MK, 2) and drv._pending is None
    # Without a pending step, `__call__` runs the program with no decision.
    drv(_IDS, _POS, runner=runner)
    assert prog.calls[-1] == ("forward", None, 3)
    # A later decline drops a step left pending.
    assert drv.can_run(_IDS, _POS, runner=runner)
    assert not drv.can_run(_IDS, _POS, inputs_embeds=torch.zeros(3, 2), runner=runner)
    assert drv._pending is None


def test_ab_alternates_the_arms_of_an_eligible_key(fwd_ctx: None) -> None:
    """A/B on: the first eligible step runs the program (`mk` arm), the next
    one the stock model (`stock` arm); both are timed under the key."""
    prog = _Program(_MK)
    drv = _Driver(prog, ab=True)
    runner = _stock_runner()
    hit, out = drv.try_forward(runner, StepContext(_IDS, _POS))
    assert hit and torch.equal(out, _IDS.float() * 10)
    assert drv.ab_calibrator.last_decision == ("mk_t1_b4", "mk")
    hit, out = drv.try_forward(runner, StepContext(_IDS, _POS))
    assert hit and torch.equal(out, -_IDS.float())
    assert drv.ab_calibrator.last_decision == ("mk_t1_b4", "stock")
    assert drv._pending is None
    assert [c[0] for c in prog.calls] == ["decide", "forward", "decide"]
    assert len(runner.stock_calls) == 1
    assert drv.step_mix_tracker.counts == {"mk_t1_b4_ab_mk": 1, "mk_t1_b4_ab_stock": 1}
    assert set(drv.ab_calibrator._cumulative["mk_t1_b4_p3"]) == {"mk", "stock"}


@pytest.mark.parametrize(
    ("text_embeds", "token_ids", "served"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_text_embeds_step_is_served_from_its_token_ids(
    fwd_ctx: None, text_embeds: bool, token_ids: bool, served: bool
) -> None:
    """With `_text_embeds`, a step the runner embedded from text tokens alone
    is served from `ctx.token_ids`; otherwise `inputs_embeds` sends it to the
    stock model."""
    prog = _Program(_MK)
    drv = _Driver(prog, text_embeds=text_embeds)
    ctx = StepContext(
        None,
        _POS,
        inputs_embeds=torch.zeros(3, 2),
        token_ids=_IDS if token_ids else None,
    )
    hit, out = drv.try_forward(_stock_runner(), ctx)
    if served:
        assert hit and torch.equal(out, _IDS.float() * 10)
        seen = prog.calls[0][1]
        assert seen.input_ids is _IDS and seen.inputs_embeds is None
    else:
        assert (hit, out) == (False, None) and prog.calls == []
