# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel paths of TPUModelRunner."""

from unittest.mock import MagicMock

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(**attrs):
    runner = MagicMock(spec=TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.model_config = MagicMock()
    runner.model_config.dtype = torch.bfloat16
    runner._pp_intermediate_template = None
    for name in ("forward_model", "_pp_intermediate_tensors",
                 "_sample_tokens_pp_intermediate_stage", "sample_tokens"):
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

        runner.model.assert_called_once_with(input_ids=ids,
                                             positions=pos,
                                             inputs_embeds=None)
        assert out.shape == (4, 8) and aux is None

    def test_passes_intermediate_tensors_and_returns_them_unwrapped(self):
        runner = _runner()
        produced = IntermediateTensors({"hidden_states": torch.zeros(4, 8)})
        runner.model = MagicMock(return_value=produced)
        consumed = IntermediateTensors({"hidden_states": torch.ones(4, 8)})
        ids, pos = torch.zeros(4, dtype=torch.int32), torch.zeros(4)

        out, aux = runner.forward_model(ids,
                                        pos,
                                        intermediate_tensors=consumed)

        runner.model.assert_called_once_with(input_ids=ids,
                                             positions=pos,
                                             inputs_embeds=None,
                                             intermediate_tensors=consumed)
        assert out is produced and aux is None


class TestIntermediateBuffers:

    def _model(self):
        model = MagicMock()
        model.make_empty_intermediate_tensors.return_value = (
            IntermediateTensors({
                "hidden_states":
                torch.zeros(1, 8, dtype=torch.bfloat16),
                "residual":
                torch.zeros(1, 8, dtype=torch.bfloat16),
            }))
        return model

    def test_buffers_take_the_token_count_and_cache_the_template(self):
        runner = _runner(model=self._model())

        zeros = runner._pp_intermediate_tensors(16, zeros=True)
        empties = runner._pp_intermediate_tensors(32, zeros=False)

        assert set(zeros.tensors) == {"hidden_states", "residual"}
        assert all(t.shape == (16, 8) and t.dtype == torch.bfloat16
                   for t in zeros.tensors.values())
        assert torch.equal(zeros["hidden_states"],
                           torch.zeros(16, 8).bfloat16())
        assert all(t.shape == (32, 8) for t in empties.tensors.values())
        runner.model.make_empty_intermediate_tensors.assert_called_once_with(
            batch_size=1, dtype=torch.bfloat16, device=runner.device)


class TestIntermediateStageSampleTokens:

    def test_without_a_pending_step_returns_empty(self):
        runner = _runner(_pp_pending_scheduler_output=None)

        assert runner._sample_tokens_pp_intermediate_stage() is \
            EMPTY_MODEL_RUNNER_OUTPUT

    def test_reports_kv_connector_progress_for_the_pending_step(self):
        scheduler_output = object()
        runner = _runner(_pp_pending_scheduler_output=scheduler_output)
        runner.get_finished_kv_transfers.return_value = ({"r1"}, None, None,
                                                         set(), None, None)

        output = runner._sample_tokens_pp_intermediate_stage()

        runner.maybe_wait_for_kv_save.assert_called_once_with()
        runner.get_finished_kv_transfers.assert_called_once_with(
            scheduler_output)
        assert output is not EMPTY_MODEL_RUNNER_OUTPUT
        assert output.kv_connector_output.finished_sending == {"r1"}
        assert runner._pp_pending_scheduler_output is None

    def test_quiet_connector_returns_empty(self):
        runner = _runner(_pp_pending_scheduler_output=object())
        runner.get_finished_kv_transfers.return_value = (None, None, None,
                                                         set(), None, None)

        assert runner._sample_tokens_pp_intermediate_stage() is \
            EMPTY_MODEL_RUNNER_OUTPUT

    def test_sample_tokens_on_an_intermediate_stage_takes_the_passthrough(
            self):
        runner = _runner(execute_model_state=None,
                         _pp_is_last=False,
                         _pp_pending_scheduler_output=None)

        assert runner.sample_tokens(None) is EMPTY_MODEL_RUNNER_OUTPUT

    def test_sample_tokens_without_state_on_the_last_stage_is_a_noop(self):
        runner = _runner(execute_model_state=None, _pp_is_last=True)

        assert runner.sample_tokens(None) is None
