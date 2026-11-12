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
"""The runner's megakernel step and sequence parallelism.

A step the megakernel serves records sequence parallelism off; a stock step
records the step's sequence-parallel form, unless a megakernel program that
serves TP mode has loaded (it resharded the stock MoE experts to TP), in which
case every stock step and its default padding mask keep the rows replicated.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_torchtpu.layers.adapter import sequence_parallel
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Driver:
    def __init__(self, hit: bool, mode: str = "ep") -> None:
        self.hit = hit
        self.program = SimpleNamespace(mode=mode)
        self.active_seen: list[bool] = []

    def try_forward(self, runner: Any, ctx: Any) -> tuple[bool, Any]:
        self.active_seen.append(sequence_parallel.is_active())
        return (True, torch.ones(2, 2)) if self.hit else (False, None)


@pytest.fixture
def sp_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every step is sequence-parallel (the switch on, threshold 0)."""
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    monkeypatch.setattr(sequence_parallel, "min_rows", lambda *_a: 0)
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    sequence_parallel.set_active(True)
    yield
    sequence_parallel.set_active(False)


def _runner(driver: Any, replicated: bool) -> Any:
    seen: list[bool] = []

    def model(**_kw: Any) -> Any:
        seen.append(sequence_parallel.is_active())
        return torch.zeros(2, 2)

    runner = SimpleNamespace(
        _megakernel=driver, _megakernel_rows_replicated=replicated, model=model
    )
    runner.stock_active = seen
    return runner


def _forward(runner: Any) -> Any:
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    return TPUModelRunner.forward_model(runner, ids, pos)


def test_a_megakernel_step_records_sequence_parallelism_off(sp_on: None) -> None:
    drv = _Driver(hit=True, mode="tp")
    runner = _runner(drv, replicated=True)
    out, aux = _forward(runner)
    assert torch.equal(out, torch.ones(2, 2)) and aux is None
    assert runner.stock_active == [] and not sequence_parallel.is_active()


@pytest.mark.parametrize(
    "driver, replicated, want",
    [
        (None, False, True),  # no megakernel: the stack's form
        ("ep", False, True),  # an EP megakernel declining the step
        ("tp", True, False),  # a TP megakernel resharded the experts
    ],
)
def test_a_stock_step_keeps_rows_replicated_after_a_tp_megakernel(
    sp_on: None, driver: Any, replicated: bool, want: bool
) -> None:
    drv = None if driver is None else _Driver(hit=False, mode=driver)
    runner = _runner(drv, replicated=replicated)
    _forward(runner)
    assert runner.stock_active == [want]


def test_the_default_padding_mask_follows_the_rows(sp_on: None) -> None:
    """The padding mask is this rank's slice only for a sliced step."""
    updates: list[tuple[int, int]] = []
    state = SimpleNamespace(update=lambda v, n: updates.append((v, n)))
    # 14 of 32 rows valid; TP rank 3 of 8 holds rows 12..15 of a sliced step
    for replicated, want in ((False, (2, 4)), (True, (14, 32))):
        runner = SimpleNamespace(
            _token_padding_state=state,
            _megakernel_rows_replicated=replicated,
            _last_sequence_layout_plan=None,
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
        )
        with pytest.MonkeyPatch.context() as mp:
            import vllm.distributed as vd

            mp.setattr(vd, "get_tensor_model_parallel_rank", lambda: 3)
            TPUModelRunner._token_padding_update(runner, 32, num_valid_tokens=14)
        assert updates[-1] == want


def test_load_records_replicated_rows_for_a_tp_program() -> None:
    """`load_model` sets the flag from the created driver's program (source)."""
    import inspect

    src = inspect.getsource(TPUModelRunner.load_model)
    create = src.index("self._megakernel = megakernel_adapter.maybe_create(self)")
    flag = src.index("self._megakernel_rows_replicated = (", create)
    assert '"mode", None) == "tp"' in src[flag : flag + 200]
    init = inspect.getsource(tpu_runner.TPUModelRunner.__init__)
    assert "self._megakernel_rows_replicated = False" in init
