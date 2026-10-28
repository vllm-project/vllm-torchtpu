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
"""Protect asynchronous store snapshots, chunk coverage and padding semantics."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.distributed.kv_transfer import kv_save_nan_check
from vllm_torchtpu.distributed.kv_transfer.kv_save_nan_check import (
    KVSaveNanCheck,
    begin_kv_save_nan_check,
    wait_for_save_with_nan_check,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    SequenceLayoutPlan,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


@pytest.fixture
def events(monkeypatch):
    events = []

    class Event:
        ready = False
        synchronized = False

        def __init__(self):
            events.append(self)

        def record(self):
            assert not self.synchronized

        def query(self):
            return self.ready

        def synchronize(self):
            assert self.ready, "The store consumer must wait for copy readiness"
            self.synchronized = True

    monkeypatch.setattr(kv_save_nan_check, "TpuEvent", Event)
    return events


def plan(size, valid, token_slice=None, mapping=None):
    return SequenceLayoutPlan(
        descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        token_slice=token_slice or slice(0, size),
        global_num_tokens=valid,
        global_padded_num_tokens=size,
        local_num_tokens=valid,
        local_padded_num_tokens=size,
        _packed_to_request_major_token_indices=mapping,
    )


def test_earlier_chunk_and_request_mapping_survive_reused_host_buffers(events):
    check = KVSaveNanCheck()
    query = np.array([0, 2, 3])
    hidden = torch.tensor([[0.0], [float("nan")], [1.0], [float("nan")]])
    check.record(hidden, ("bad", "good"), query, plan(4, 3))
    query[:] = 0
    check.record(torch.ones(4, 1), ("later",), np.array([0, 3]), plan(4, 3))
    assert not check.is_ready()
    assert not any(event.synchronized for event in events)
    for event in events:
        event.ready = True
    assert check.is_ready()
    assert check.bad_request_ids() == {"bad"}
    assert all(event.synchronized for event in events)
    assert check.bad_request_ids() == {"bad"}


@pytest.mark.parametrize(
    "bad_row,expected",
    [
        (0, {"b"}),
        (1, set()),
        (2, {"a"}),
    ],
)
def test_pcp_repacked_tokens_ignore_padding(events, bad_row, expected):
    check = KVSaveNanCheck()
    mapping = np.array([0, 2, -1, 1, 3, 4, -1, -1])
    hidden = torch.ones(4, 2)
    hidden[bad_row, 0] = float("nan")
    check.record(
        hidden, ("a", "b"), np.array([0, 2, 5]), plan(8, 5, slice(1, 5), mapping)
    )
    events[0].ready = True
    assert check.bad_request_ids() == expected


def test_steps_hold_independent_copy_events_and_verdicts(events):
    bad, good = KVSaveNanCheck(), KVSaveNanCheck()
    for check, value in [(bad, float("nan")), (good, 0.0)]:
        check.record(torch.tensor([[value]]), ("same",), np.array([0, 1]), plan(1, 1))
    events[1].ready = True
    assert good.is_ready() and not bad.is_ready()
    assert good.bad_request_ids() == set()
    events[0].ready = True
    assert bad.bad_request_ids() == {"same"}


def test_multi_connector_only_passes_snapshot_to_opted_in_stores():
    class Plain:
        calls = 0

        def wait_for_save(self):
            self.calls += 1

    class Checking:
        needs_kv_save_nan_check = False
        snapshot = None
        calls = 0

        def wait_for_save(self, nan_check=None):
            self.snapshot = nan_check
            self.calls += 1

    plain, checking, disabled = Plain(), Checking(), Checking()
    multi = object.__new__(MultiConnector)
    multi._connectors = [plain, checking, disabled]
    assert begin_kv_save_nan_check(multi) is None
    checking.needs_kv_save_nan_check = True
    check = begin_kv_save_nan_check(multi)
    assert check is not None
    wait_for_save_with_nan_check(multi, check)
    assert plain.calls == 1 and checking.snapshot is check
    assert disabled.calls == 1 and disabled.snapshot is None


def test_intermediate_tensors_records_nans_across_activations(events):
    check = KVSaveNanCheck()
    chunk_0 = IntermediateTensors(
        {
            "hidden_states": torch.tensor([[float("nan")], [1.0]]),
            "residual": torch.ones(2, 1),
        }
    )
    chunk_1 = IntermediateTensors(
        {
            "hidden_states": torch.ones(2, 1),
            "residual": torch.tensor([[1.0], [float("nan")]]),
            "topk_indices": torch.zeros((2, 1), dtype=torch.int32),
        }
    )
    check.record(chunk_0, ("bad_hidden", "good_0"), np.array([0, 1, 2]), plan(2, 2))
    check.record(chunk_1, ("good_1", "bad_residual"), np.array([0, 1, 2]), plan(2, 2))
    for event in events:
        event.ready = True
    assert check.bad_request_ids() == {"bad_hidden", "bad_residual"}


@pytest.mark.parametrize("pp_is_last", [True, False])
def test_runner_execute_model_and_sample_tokens_forward_step_nan_check(
    events, pp_is_last
):
    class OptInConnector:
        needs_kv_save_nan_check = True
        snapshot = None

        def wait_for_save(self, nan_check=None):
            self.snapshot = nan_check

        def get_transfer_results(self, _finished_req_ids):
            return KVConnectorTransferResults(finished_sending={"r0"})

        def get_block_ids_with_load_errors(self):
            return set()

        def build_connector_worker_meta(self):
            return None

        def get_kv_connector_stats(self):
            return None

        def clear_connector_metadata(self):
            pass

    connector = OptInConnector()
    wave = MagicMock()
    wave.step.return_value = contextlib.nullcontext()
    wave.start_forward.return_value = IntermediateTensors(
        {"hidden_states": torch.zeros(2, 2), "residual": torch.zeros(2, 2)}
    )

    chunk_outputs = [
        (
            torch.tensor([[1.0, 2.0], [float("nan"), 1.0]])
            if pp_is_last
            else IntermediateTensors(
                {
                    "hidden_states": torch.ones(2, 2),
                    "residual": torch.tensor([[1.0, 2.0], [float("nan"), 1.0]]),
                }
            )
        ),
        (
            torch.tensor([[1.0, 2.0], [float("nan"), 1.0]])
            if pp_is_last
            else IntermediateTensors(
                {
                    "hidden_states": torch.tensor([[1.0, 2.0], [float("nan"), 1.0]]),
                    "residual": torch.ones(2, 2),
                }
            )
        ),
    ]
    valid_tokens_per_chunk = [2, 1]

    runner = SimpleNamespace(
        _pp_wave=None if pp_is_last else wave,
        _pp_is_first=True,
        _pp_is_last=pp_is_last,
        _pp_pending_scheduler_output=None,
        _pp_pending_kv_save_nan_check=None,
        execute_model_state=None,
        _update_states=lambda _so: None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _mtp_prefill_only=False,
        _reorder_batch_for_rpa=lambda _so: (0, 0),
        _count_input_chunks=lambda _so: (2, 1),
        _dp_coordinated_step=lambda *_args, **_kwargs: (2, 3),
        _dp_num_tokens_across_dp=lambda _n: None,
        supports_mm_inputs=False,
        phase_based_profiler=None,
        _is_async_drafter=False,
        _greedy_top_tokens_this_step=lambda: False,
        vllm_config=SimpleNamespace(),
        maybe_setup_kv_connector=lambda *_args, **_kwargs: None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            max_num_logprobs=None,
            all_greedy=True,
            num_tokens_no_spec=[2, 1],
            token_ids_cpu=np.zeros((2, 8), dtype=np.int32),
        ),
        requests={
            "r0": SimpleNamespace(
                num_computed_tokens=0, num_tokens=2, output_token_ids=[]
            ),
            "r1": SimpleNamespace(
                num_computed_tokens=0, num_tokens=1, output_token_ids=[]
            ),
        },
        _run_dp_dummy_chunk=MagicMock(),
        query_start_loc_np=np.zeros(4, dtype=np.int32),
        _last_sequence_layout_plan=None,
        position_ids=torch.zeros(2, dtype=torch.int32),
        _flush_mamba_state_seed_copies=lambda: None,
        _reset_read_offsets_for_new_requests=lambda *_args: None,
        input_ids=torch.zeros(2, dtype=torch.int32),
        _apply_async_token_substitution=lambda ids, *_args: ids,
        _record_input_staging_fence=lambda: None,
        _get_model_inputs=lambda ids, _mm: (ids, None),
        _token_padding_update=lambda _n: None,
        _pp_take_topk_indices=lambda *_args: None,
        _pp_outgoing_tensors=lambda tensors, _rows: tensors,
        mesh=None,
        _kv_cache_bundle=None,
        _stage_timing=MagicMock(),
        is_pooling_model=False,
        sequence_layout_planner=SimpleNamespace(
            maybe_select_logits_hidden_states=lambda hs, _plan, _idx: hs[:1]
        ),
        compute_logits_from_hidden_states=lambda _hs: torch.zeros(1, 8),
        _attn_metadata_builder_ctx=SimpleNamespace(
            mamba_state_indices=None, unified_mamba_state_indices=None
        ),
        mamba_slot_read_offsets=None,
        speculative_config=None,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda _logits: torch.tensor([7]),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
    )
    runner._stage_timing.stage.return_value = contextlib.nullcontext()
    runner._stage_timing.take_samples.return_value = ([], [])

    def prepare_inputs(_so, start_index, _num_decode, _num_windowed):
        valid = valid_tokens_per_chunk[start_index]
        runner.query_start_loc_np[:2] = [0, valid]
        runner._last_sequence_layout_plan = plan(2, valid)
        empty = torch.empty(0, dtype=torch.int32)
        return (
            None,
            torch.tensor([valid - 1]),
            1,
            1,
            start_index + 1,
            empty,
            empty,
            None,
            None,
            None,
        )

    runner._prepare_inputs = prepare_inputs
    runner.forward_model = MagicMock(side_effect=[(out, None) for out in chunk_outputs])
    for name in (
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
        "_sample_tokens_pp_intermediate_stage",
        "sample_tokens",
        "execute_model",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        num_scheduled_tokens={"r0": 2, "r1": 1},
        finished_req_ids=set(),
        has_structured_output_requests=False,
    )

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.set_forward_context",
            side_effect=lambda *_a, **_kw: contextlib.nullcontext(),
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
            side_effect=lambda *_a, **_kw: contextlib.nullcontext(),
        ),
        patch.object(
            AsyncTPUCopyState,
            "from_device_chunks",
            side_effect=lambda chunks, lengths, spec_logprobs=None: AsyncTPUCopyState(
                chunks_cpu=chunks,
                chunk_real_lens=lengths,
                copy_ready_event=MagicMock(),
                spec_logprobs=spec_logprobs,
            ),
        ),
    ):
        assert runner.execute_model(scheduler_output) is None
        output = runner.sample_tokens(None)

    runner._run_dp_dummy_chunk.assert_called_once_with(2)
    assert connector.snapshot is not None
    assert len(events) == 2
    for event in events:
        event.ready = True
    assert connector.snapshot.bad_request_ids() == {"r0"}
    assert output.kv_connector_output.finished_sending == {"r0"}
