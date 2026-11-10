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
"""The runner side of row-gated sequence-parallel MoE.

The communicator and the MoE layers read `sequence_parallel.is_active()` to
decide whether a step's rows are sliced over TP. Nothing but the runner sets
it, so if `forward_model` stopped recording each step's form, every step
would silently take one form and row gating would be inert. These tests pin
that wiring and the per-rank padding mask that goes with a sliced step.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.distributed

from vllm_torchtpu.layers.adapter import sequence_parallel
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.cpu_test

THRESHOLD = 512


@pytest.fixture
def sp_on(monkeypatch):
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    monkeypatch.setattr(sequence_parallel, "min_rows", lambda *a: THRESHOLD)


@pytest.mark.parametrize("rows,sliced", [(8, False), (511, False), (512, True)])
def test_forward_model_records_the_form_of_each_step(sp_on, monkeypatch, rows, sliced):
    """The model runs with `is_active()` equal to the step's form, whatever
    the previous step left behind."""
    monkeypatch.setattr(sequence_parallel, "_ACTIVE", not sliced)
    seen = []

    def model(**kwargs):
        seen.append(sequence_parallel.is_active())
        return torch.zeros(rows, 4)

    runner = SimpleNamespace(model=model)
    positions = torch.arange(rows)
    out, aux = TPUModelRunner.forward_model(runner, positions, positions)
    assert seen == [sliced]
    assert out.shape == (rows, 4) and aux is None


def test_forward_model_leaves_sequence_parallel_off_when_disabled(monkeypatch):
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: False)
    monkeypatch.setattr(sequence_parallel, "_ACTIVE", True)
    runner = SimpleNamespace(model=lambda **kwargs: torch.zeros(4096, 4))
    positions = torch.arange(4096)
    TPUModelRunner.forward_model(runner, positions, positions)
    assert sequence_parallel.is_active() is False


def _padding_runner(tp, max_tokens=1024):
    return SimpleNamespace(
        _token_padding_state=TokenPaddingState.create(max_tokens, device="cpu"),
        _last_sequence_layout_plan=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


def _mask(runner, rows):
    return runner._token_padding_state.get_local_padding_mask(rows).tolist()


@pytest.mark.parametrize("tp_rank,rank_valid", [(0, 128), (1, 128), (2, 44), (3, 0)])
def test_a_sliced_step_masks_this_ranks_slice(sp_on, monkeypatch, tp_rank, rank_valid):
    """512 rows over TP4: each rank owns 128 of them; the first 300 are real."""
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(
        runner, 512, num_valid_tokens=300, rows_sliced=True
    )
    assert _mask(runner, 128) == [i >= rank_valid for i in range(128)]
    assert runner._token_padding_state.get_local_valid_tokens().item() == rank_valid


def test_an_sp_step_writes_the_full_mask_until_a_model_follows_the_form(
    sp_on, monkeypatch
):
    """No model follows each step's form yet, so a sequence-parallel step of
    512 rows (at the 512-row threshold) still gets main's full mask: a path
    that reads every row must not see other ranks' real rows as padding."""
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 2)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(runner, 512, num_valid_tokens=300)
    assert _mask(runner, 512) == [i >= 300 for i in range(512)]
    assert runner._token_padding_state.get_local_valid_tokens().item() == 300


@pytest.mark.parametrize("tp_rank,rank_valid", [(0, 128), (2, 44), (3, 0)])
def test_a_model_that_follows_the_form_gets_the_slice_by_default(
    sp_on, monkeypatch, tp_rank, rank_valid
):
    """Once the model follows each step's form (the Qwen3.5 model patch), the
    default decision is the target forward's: a 512-row sequence-parallel step
    is sliced, so each rank's mask is its own slice."""
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(runner, 512, num_valid_tokens=300)
    assert _mask(runner, 128) == [i >= rank_valid for i in range(128)]
    assert runner._token_padding_state.get_local_valid_tokens().item() == rank_valid


def test_a_small_step_of_a_model_that_follows_the_form_masks_every_row(
    sp_on, monkeypatch
):
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 3)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(runner, 16, num_valid_tokens=5)
    assert _mask(runner, 16) == [i >= 5 for i in range(16)]


def test_a_replicated_step_masks_every_row(sp_on, monkeypatch):
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 3)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(runner, 16, num_valid_tokens=5)
    assert _mask(runner, 16) == [i >= 5 for i in range(16)]


def test_an_explicit_replicated_decision_overrides_the_row_count(sp_on, monkeypatch):
    """The drafter passes its own decision: a replicated forward of a
    sliced-size step must not mark real rows of other ranks as padding."""
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 1)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(
        runner, 512, num_valid_tokens=300, rows_sliced=False
    )
    assert _mask(runner, 512) == [i >= 300 for i in range(512)]
