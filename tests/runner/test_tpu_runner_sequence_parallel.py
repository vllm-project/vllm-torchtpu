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
from vllm.config import ParallelConfig

from vllm_torchtpu import _patch_sequence_parallel_moe_admission, envs
from vllm_torchtpu.layers.adapter import sequence_parallel, token_padding
from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.cpu_test

THRESHOLD = 512


@pytest.fixture
def sp_on(monkeypatch):
    monkeypatch.setattr(sequence_parallel, "enabled", lambda: True)
    monkeypatch.setattr(sequence_parallel, "min_rows", lambda *a: THRESHOLD)


@pytest.fixture
def admission(monkeypatch):
    """vLLM's ParallelConfig under the TPU's sequence-parallel MoE admission
    with TPU_MOE_SEQUENCE_PARALLEL=1: admitted with expert parallelism and
    TP > 1 only."""
    monkeypatch.setattr(
        ParallelConfig,
        "use_sequence_parallel_moe",
        ParallelConfig.use_sequence_parallel_moe,
    )
    monkeypatch.setattr(ParallelConfig, "_tpu_no_sp_moe_patch", False, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    _patch_sequence_parallel_moe_admission()


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


def _padding_runner(tp, max_tokens=1024, parallel_config=None):
    return SimpleNamespace(
        _token_padding_state=TokenPaddingState.create(max_tokens, device="cpu"),
        _last_sequence_layout_plan=None,
        parallel_config=parallel_config
        or SimpleNamespace(tensor_parallel_size=tp, use_sequence_parallel_moe=True),
    )


def _mask(runner, rows):
    return runner._token_padding_state.get_local_padding_mask(rows).tolist()


@pytest.mark.parametrize("tp_rank,rank_valid", [(0, 128), (1, 128), (2, 44), (3, 0)])
def test_a_sliced_step_masks_this_ranks_slice(monkeypatch, tp_rank, rank_valid):
    """512 rows over TP4: each rank owns 128 of them; the first 300 are real."""
    monkeypatch.setattr(tpu_runner, "get_tensor_model_parallel_rank", lambda: tp_rank)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(
        runner, 512, num_valid_tokens=300, rows_sliced=True
    )
    assert _mask(runner, 128) == [i >= rank_valid for i in range(128)]
    assert runner._token_padding_state.get_local_valid_tokens().item() == rank_valid


@pytest.mark.parametrize("rows,valid,tp_rank", [(512, 300, 2), (16, 5, 3)])
def test_the_full_mask_is_written_unless_the_rows_are_sliced(
    monkeypatch, rows, valid, tp_rank
):
    """By default a step gets the full mask on every rank: a path that reads
    every row must not see other ranks' real rows as padding."""
    monkeypatch.setattr(tpu_runner, "get_tensor_model_parallel_rank", lambda: tp_rank)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(runner, rows, num_valid_tokens=valid)
    assert _mask(runner, rows) == [i >= valid for i in range(rows)]
    assert runner._token_padding_state.get_local_valid_tokens().item() == valid


@pytest.mark.parametrize("tp_rank", range(4))
@pytest.mark.parametrize("enable_expert_parallel", [False, True])
def test_the_moe_drops_exactly_the_padding_rows(
    sp_on, admission, monkeypatch, enable_expert_parallel, tp_rank
):
    """512 rows over TP4, the first 300 real, once the model follows each
    step's form (the Qwen3.5 model patch). The default decision is the target
    forward's, and vLLM's decoder layers slice the rows only where
    sequence-parallel MoE is admitted: with expert parallelism each rank's MoE
    reads its own 128 rows, without it every rank's MoE reads all 512. Either
    way the MoE keeps the routing weights of exactly the real rows."""
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(tpu_runner, "get_tensor_model_parallel_rank", lambda: tp_rank)
    parallel_config = ParallelConfig(
        tensor_parallel_size=4,
        enable_expert_parallel=enable_expert_parallel,
        distributed_executor_backend="mp",
    )
    runner = _padding_runner(tp=4, parallel_config=parallel_config)
    TPUModelRunner._token_padding_update(runner, 512, num_valid_tokens=300)
    first, rows = (128 * tp_rank, 128) if enable_expert_parallel else (0, 512)
    with token_padding.use_padding_state(runner._token_padding_state):
        _, weights = token_padding.zero_routing_weights_for_padding(
            torch.zeros(rows, 2, dtype=torch.int32),
            torch.ones(rows, 2),
            is_local_tensor=True,
        )
    kept = [first + i for i in range(rows) if bool(weights[i].all())]
    assert kept == [row for row in range(first, first + rows) if row < 300]
    assert runner._token_padding_state.get_local_valid_tokens().item() == len(kept)


def test_an_explicit_replicated_decision_overrides_the_slice(sp_on, monkeypatch):
    """rows_sliced=False is the caller's decision for a replicated forward of
    a step the target would slice: other ranks' real rows must not read as
    padding."""
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(tpu_runner, "get_tensor_model_parallel_rank", lambda: 1)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(
        runner, 512, num_valid_tokens=300, rows_sliced=False
    )
    assert _mask(runner, 512) == [i >= 300 for i in range(512)]
    assert runner._token_padding_state.get_local_valid_tokens().item() == 300


def test_a_small_step_of_a_model_that_follows_the_form_masks_every_row(
    sp_on, monkeypatch
):
    monkeypatch.setattr(sequence_parallel, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(tpu_runner, "get_tensor_model_parallel_rank", lambda: 3)
    runner = _padding_runner(tp=4)
    TPUModelRunner._token_padding_update(runner, 16, num_valid_tokens=5)
    assert _mask(runner, 16) == [i >= 5 for i in range(16)]
