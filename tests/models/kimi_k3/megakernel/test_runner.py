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

"""Asynchronous megakernel responses preserve worker state and KV metadata."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.runner import MegakernelModelRunner


@pytest.mark.parametrize("output_rank", [False, True])
@pytest.mark.parametrize("kv_transfer", [False, True])
def test_async_sample_tokens_preserves_local_state_and_transfer_metadata(
    monkeypatch, output_rank, kv_transfer
):
    import torch
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_torchtpu.models.kimi_k3.megakernel import runner
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

    tokens = torch.tensor([[101], [202]], dtype=torch.int32)
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[tokens],
        chunk_real_lens=[2],
        copy_ready_event=None,
        completed=True,
    )
    monkeypatch.setattr(
        AsyncTPUCopyState, "from_device_chunks", lambda *args: copy_state
    )
    monkeypatch.setattr(runner, "has_kv_transfer_group", lambda: kv_transfer)
    connector_output = KVConnectorOutput(expected_finished_count=32)
    scheduled = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1})
    instance = SimpleNamespace(
        execute_model_state=runner.MegakernelStepState(scheduled, tokens, {0: 0, 1: 0}),
        input_batch=SimpleNamespace(
            num_reqs=2, req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}
        ),
        requests={
            request_id: SimpleNamespace(num_computed_tokens=10, num_tokens=11)
            for request_id in ("a", "b")
        },
        scheduler_config=SimpleNamespace(async_scheduling=True),
        _is_output_rank=output_rank,
        _modify_prev_results=lambda: None,
        maybe_wait_for_kv_save=lambda: None,
        _build_kv_connector_output=lambda _: connector_output,
        _update_placeholder=lambda *args: {"a": 0, "b": 1},
    )
    result = MegakernelModelRunner.sample_tokens(instance).get_output()
    assert result.kv_connector_output is connector_output
    assert instance._pre_async_results.req_ids == ["a", "b"]
    assert instance._pre_async_results.copy_state is copy_state
    assert copy_state.sampled_token_ids_cpu.tolist() == [[101], [202]]
    if output_rank or not kv_transfer:
        assert result.req_ids == ["a", "b"]
        assert result.sampled_token_ids == [[101], [202]]
    else:
        assert result.req_ids == []
        assert result.sampled_token_ids == []
