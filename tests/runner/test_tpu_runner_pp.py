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
"""Pipeline-parallel paths of TPUModelRunner."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_torchtpu.distributed.pp_ring import PPRing
from vllm_torchtpu.distributed.pp_wave import PPWave
from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(**attrs):
    runner = MagicMock(spec=TPUModelRunner)
    runner._stage_timing = MagicMock()
    runner.device = torch.device("cpu")
    runner.model_config = MagicMock()
    runner.model_config.dtype = torch.bfloat16
    runner._pp_intermediate_template = None
    runner._pp_topk_buffer = None
    runner._pp_pending_kv_save_nan_check = None
    for name in (
        "forward_model",
        "_pp_intermediate_tensors",
        "_pp_take_topk_indices",
        "_pp_outgoing_tensors",
        "_sample_tokens_pp_intermediate_stage",
        "sample_tokens",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


class TestForwardModel:
    def test_single_stage_call_is_unchanged(self):
        runner = _runner()
        runner.model = MagicMock(return_value=torch.zeros(4, 8))
        ids, pos = torch.zeros(4, dtype=torch.int32), torch.zeros(4)

        out, aux = runner.forward_model(ids, pos)

        runner.model.assert_called_once_with(
            input_ids=ids, positions=pos, inputs_embeds=None
        )
        assert out.shape == (4, 8) and aux is None

    def test_passes_intermediate_tensors_and_returns_them_unwrapped(self):
        runner = _runner()
        produced = IntermediateTensors({"hidden_states": torch.zeros(4, 8)})
        runner.model = MagicMock(return_value=produced)
        consumed = IntermediateTensors({"hidden_states": torch.ones(4, 8)})
        ids, pos = torch.zeros(4, dtype=torch.int32), torch.zeros(4)

        out, aux = runner.forward_model(ids, pos, intermediate_tensors=consumed)

        runner.model.assert_called_once_with(
            input_ids=ids,
            positions=pos,
            inputs_embeds=None,
            intermediate_tensors=consumed,
        )
        assert out is produced and aux is None


class TestIntermediateBuffers:
    def _model(self):
        model = MagicMock()
        model.make_empty_intermediate_tensors.return_value = IntermediateTensors(
            {
                "hidden_states": torch.zeros(1, 8, dtype=torch.bfloat16),
                "residual": torch.zeros(1, 8, dtype=torch.bfloat16),
            }
        )
        return model

    def test_buffers_take_the_token_count_and_cache_the_template(self):
        runner = _runner(model=self._model())

        zeros = runner._pp_intermediate_tensors(16, zeros=True)
        empties = runner._pp_intermediate_tensors(32, zeros=False)

        assert set(zeros.tensors) == {"hidden_states", "residual"}
        assert all(
            t.shape == (16, 8) and t.dtype == torch.bfloat16
            for t in zeros.tensors.values()
        )
        assert torch.equal(zeros["hidden_states"], torch.zeros(16, 8).bfloat16())
        assert all(t.shape == (32, 8) for t in empties.tensors.values())
        runner.model.make_empty_intermediate_tensors.assert_called_once_with(
            batch_size=1, dtype=torch.bfloat16, device=runner.device
        )


class TestTopkHandoff:
    def _runner_with_buffer(self):
        buffer = torch.zeros(64, 4, dtype=torch.int32)
        return _runner(_pp_topk_buffer=buffer), buffer

    def test_incoming_indices_land_in_the_buffer_and_leave_the_arguments(self):
        runner, buffer = self._runner_with_buffer()
        received = torch.full((16, 4), 7, dtype=torch.int32)
        tensors = IntermediateTensors(
            {
                "hidden_states": torch.zeros(16, 8),
                "residual": torch.zeros(16, 8),
                "topk_indices": received,
            }
        )

        runner._pp_take_topk_indices(tensors, 16)

        # the model's forward never sees it
        assert set(tensors.tensors) == {"hidden_states", "residual"}
        assert torch.equal(buffer[:16], received)
        # rows this forward does not cover are untouched
        assert torch.equal(buffer[16:], torch.zeros(48, 4, dtype=torch.int32))

    def test_a_stage_handed_no_indices_is_refused(self):
        runner, _ = self._runner_with_buffer()
        tensors = IntermediateTensors({"hidden_states": torch.zeros(16, 8)})

        with pytest.raises(RuntimeError, match="no top-k indices"):
            runner._pp_take_topk_indices(tensors, 16)


class TestIntermediateStageSampleTokens:
    def test_without_a_pending_step_returns_empty(self):
        runner = _runner(
            _pp_pending_scheduler_output=None,
            _pp_pending_kv_save_nan_check=object(),
        )

        assert (
            runner._sample_tokens_pp_intermediate_stage() is EMPTY_MODEL_RUNNER_OUTPUT
        )
        assert runner._pp_pending_kv_save_nan_check is None

    def test_reports_kv_connector_progress_for_the_pending_step(self):
        scheduler_output = object()
        nan_check = object()
        runner = _runner(
            _pp_pending_scheduler_output=scheduler_output,
            _pp_pending_kv_save_nan_check=nan_check,
        )
        runner.get_finished_kv_transfers.return_value = (
            {"r1"},
            None,
            set(),
            None,
            set(),
            None,
            None,
        )

        output = runner._sample_tokens_pp_intermediate_stage()

        runner.maybe_wait_for_kv_save.assert_called_once_with(nan_check)
        runner.get_finished_kv_transfers.assert_called_once_with(scheduler_output)
        assert output is not EMPTY_MODEL_RUNNER_OUTPUT
        assert output.kv_connector_output.finished_sending == {"r1"}
        assert runner._pp_pending_scheduler_output is None
        assert runner._pp_pending_kv_save_nan_check is None

    def test_quiet_connector_returns_empty(self):
        runner = _runner(_pp_pending_scheduler_output=object())
        runner.get_finished_kv_transfers.return_value = (
            None,
            None,
            set(),
            None,
            set(),
            None,
            None,
        )

        assert (
            runner._sample_tokens_pp_intermediate_stage() is EMPTY_MODEL_RUNNER_OUTPUT
        )

    def test_sample_tokens_on_an_intermediate_stage_takes_the_passthrough(self):
        runner = _runner(
            execute_model_state=None,
            _pp_is_last=False,
            _pp_pending_scheduler_output=None,
        )

        assert runner.sample_tokens(None) is EMPTY_MODEL_RUNNER_OUTPUT

    def test_sample_tokens_without_state_on_the_last_stage_is_a_noop(self):
        runner = _runner(execute_model_state=None, _pp_is_last=True)

        assert runner.sample_tokens(None) is None


TEMPLATE = {
    "hidden_states": ((8,), torch.bfloat16),
    "residual": ((8,), torch.bfloat16),
}


class TestHandOffConstruction:
    def _runner(self):
        runner = SimpleNamespace(
            _add_pipeline_chunk_buckets=lambda: None,
            parallel_config=SimpleNamespace(pipeline_parallel_size=4),
            _pp_wave=None,
            _pp_intermediate_template=None,
            _pp_topk_buffer=None,
            max_num_tokens=64,
            device=torch.device("cpu"),
            enforce_eager=True,
        )

        def intermediate_tensors(num_tokens, zeros):
            runner._pp_intermediate_template = TEMPLATE

        runner._pp_intermediate_tensors = intermediate_tensors
        return runner

    @pytest.fixture
    def built(self, monkeypatch):
        built = []

        class Fake:
            def __init__(self, *args, **kwargs):
                built.append((type(self).__name__, args, kwargs))

            def warmup(self):
                built.append("warmup")

        monkeypatch.setattr(tpu_runner, "PPRing", type("PPRing", (Fake,), {}))
        monkeypatch.setattr(tpu_runner, "PPWave", type("PPWave", (Fake,), {}))
        return built

    def test_the_ring_flag_builds_the_ring_with_its_depth(self, monkeypatch, built):
        monkeypatch.setenv("TPU_PP_HANDOFF", "ring")
        monkeypatch.setenv("TPU_PP_RING_DEPTH", "3")
        runner = self._runner()

        TPUModelRunner.capture_model(runner)

        device = torch.device("cpu")
        assert built == [("PPRing", (device, 64, TEMPLATE), {"depth": 3}), "warmup"]
        assert type(runner._pp_wave).__name__ == "PPRing"

    def test_the_default_builds_the_wave(self, monkeypatch, built):
        monkeypatch.delenv("TPU_PP_HANDOFF", raising=False)
        runner = self._runner()

        TPUModelRunner.capture_model(runner)

        device = torch.device("cpu")
        assert built == [("PPWave", (device, 64, TEMPLATE), {}), "warmup"]


class TestRingForwardRows:
    def _runner(self, scheduled, *, max_reqs=8, most_model_len=None, most_reqs=None):
        req_ids = [f"r{i}" for i in range(len(scheduled))]
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(num_reqs=len(scheduled), req_ids=req_ids),
            most_model_len=most_model_len,
            num_reqs_max_model_len=max_reqs,
            num_reqs_most_model_len=most_reqs,
            num_tokens_paddings=[16, 32, 64, 128],
            _dp_target_bucket=None,
            sequence_layout_planner=AllSequenceLayoutPlanner(),
        )
        for name in ("_pp_forward_rows", "_count_input_chunks"):
            setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens=dict(zip(req_ids, scheduled)),
            total_num_scheduled_tokens=sum(scheduled),
        )
        return runner, scheduler_output

    def test_one_forward_pads_to_its_bucket(self):
        runner, scheduler_output = self._runner([20, 5])
        assert runner._pp_forward_rows(scheduler_output) == [32]

    def test_requests_beyond_a_forward_split_like_the_chunk_count(self):
        runner, scheduler_output = self._runner([1] * 5, max_reqs=2)
        assert runner._pp_forward_rows(scheduler_output) == [16, 16, 16]
        assert runner._count_input_chunks(scheduler_output)[0] == 3

    def test_a_long_request_takes_the_max_model_len_split(self):
        runner, scheduler_output = self._runner(
            [100, 1, 1], max_reqs=2, most_model_len=8, most_reqs=4
        )
        # [100, 1] in max-model-len chunks of two, then [1] in a short one
        assert runner._pp_forward_rows(scheduler_output) == [128, 16]
        assert runner._count_input_chunks(scheduler_output)[0] == 2

    def test_the_dp_bucket_raises_every_forward(self):
        runner, scheduler_output = self._runner([20, 5])
        runner._dp_target_bucket = 64
        assert runner._pp_forward_rows(scheduler_output) == [64]


class _Stop(Exception):
    pass


class TestRingStepPosting:
    """execute_model posts a step's receives before preparing any forward."""

    def _run(self, wave, *, first=False, tokens=37):
        def stop():
            raise _Stop

        runner = SimpleNamespace(
            execute_model_state=None,
            _update_states=lambda _so: None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
            _mtp_prefill_only=False,
            _reorder_batch_for_rpa=lambda _so: (0, 0),
            _count_input_chunks=lambda _so: (2, 1),
            _dp_coordinated_step=lambda *_a, **_kw: (None, 2),
            _pp_wave=wave,
            _pp_is_first=first,
            _pp_forward_rows=lambda _so: [32, 16],
            supports_mm_inputs=False,
            phase_based_profiler=None,
            _is_async_drafter=False,
            # the first call after the posting point
            _greedy_top_tokens_this_step=stop,
        )
        scheduler_output = SimpleNamespace(
            total_num_scheduled_tokens=tokens, scheduled_encoder_inputs={}
        )
        with patch.object(tpu_runner, "has_kv_transfer_group", return_value=False):
            if tokens == 0:
                return TPUModelRunner.execute_model(runner, scheduler_output)
            with pytest.raises(_Stop):
                TPUModelRunner.execute_model(runner, scheduler_output)
        return None

    def test_a_later_stage_posts_every_forward_of_the_step(self):
        ring = MagicMock(spec=PPRing)
        self._run(ring)
        ring.begin_step.assert_called_once_with([32, 16])

    def test_the_first_stage_posts_nothing(self):
        ring = MagicMock(spec=PPRing)
        self._run(ring, first=True)
        ring.begin_step.assert_not_called()

    def test_a_step_without_tokens_posts_nothing(self):
        ring = MagicMock(spec=PPRing)
        assert self._run(ring, tokens=0) is EMPTY_MODEL_RUNNER_OUTPUT
        ring.begin_step.assert_not_called()

    def test_the_wave_has_no_step_to_begin(self):
        # The wave has no begin_step: reaching for it would raise.
        wave = MagicMock(spec=PPWave)
        self._run(wave)
        assert wave.step.call_count == 1


class TestRingPushAndSettle:
    def test_push_and_settle_hand_nothing_on(self, monkeypatch):
        from vllm.distributed import parallel_state

        def no_transfer(*args, **kwargs):
            raise AssertionError("the ring transferred something")

        monkeypatch.setattr(torch.distributed, "isend", no_transfer)
        monkeypatch.setattr(torch.distributed, "irecv", no_transfer)
        monkeypatch.setattr(
            parallel_state,
            "get_pp_group",
            lambda: SimpleNamespace(rank_in_group=1, world_size=4, ranks=[0, 1, 2, 3]),
        )
        ring = PPRing(torch.device("cpu"), 64, TEMPLATE)
        runner = _runner(_pp_wave=ring)
        for name in ("pp_push", "pp_settle"):
            getattr(TPUModelRunner, name)(runner)
        assert ring.transfers == 0 and not ring._sends
