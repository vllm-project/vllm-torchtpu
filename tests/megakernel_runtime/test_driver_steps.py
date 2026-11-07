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
(suppression, the step context, unconsumed model inputs) and the runner
shapes a bucket is warmed up with."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.runtime.contract import Bucket, RunnerShape
from vllm_torchtpu.megakernel.runtime.stats import BatchShape
from vllm_torchtpu.megakernel_runtime import stepping


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

    from vllm_torchtpu.megakernel import runtime as rt

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


def test_runner_shapes_without_paddings() -> None:
    shapes = stepping.runner_shapes(
        Bucket(4, 16), (), (), ckpt_window=1, max_pages_per_seq=9
    )
    assert shapes == [RunnerShape(16, 4, 9, 4)]
    shapes = stepping.runner_shapes(
        Bucket(1, 64), (16, 32), ((4, 2), (1, 8)), ckpt_window=3, max_pages_per_seq=9
    )
    assert shapes == [RunnerShape(32, 4, 2, 3), RunnerShape(32, 1, 8, 3)]
