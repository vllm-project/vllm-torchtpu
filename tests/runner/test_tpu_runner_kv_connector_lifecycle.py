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
"""Order of the KV connector calls around the drafter in sample_tokens."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)


@pytest.mark.parametrize("num_tokens", [0, 1])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_execute_model_starts_transfer_before_draining_previous_results(
    num_tokens, inline, is_async
):
    events = []
    output = object()

    class ForwardReached(Exception):
        pass

    def forward(*_args):
        events.append("forward")
        raise ForwardReached

    runner = SimpleNamespace(
        execute_model_state=None,
        _mtp_diagnostic_step=None,
        vllm_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(async_scheduling=is_async),
        _update_states=lambda _: events.append("update"),
        maybe_setup_kv_connector=MagicMock(
            side_effect=lambda *_args, **_kwargs: events.append("transfer")
        ),
        _flush_disjoint_async_results=lambda: events.append("flush"),
        _pcp_enabled=False,
        _pp_wave=None,
        _reorder_batch_for_rpa=forward,
        kv_connector_no_forward=lambda *_args: output,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        scheduled_spec_decode_tokens={},
    )
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.set_forward_context",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.dist_utils.get_raiden_inline_load",
            return_value=inline,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True
        ),
    ):
        if num_tokens:
            with pytest.raises(ForwardReached):
                TPUModelRunner.execute_model(runner, scheduler_output)
        else:
            assert TPUModelRunner.execute_model(runner, scheduler_output) is output

    runner.maybe_setup_kv_connector.assert_called_once_with(
        scheduler_output,
        after_forward=False,
        wait_for_completion=inline,
        report_completion=num_tokens == 0 or not inline,
    )
    assert events == (
        ["update", "transfer"]
        + (["flush"] if is_async else [])
        + (["forward"] if num_tokens else [])
    )


def _recording_connector(events: list[str]) -> MagicMock:
    """A connector that logs save, poll and clear, and refuses any of them
    once this step's metadata has been cleared."""
    connector = MagicMock()
    connector.metadata_bound = True

    def logged(name, result=None):
        def call(*_args):
            assert connector.metadata_bound, f"{name} after the metadata clear"
            events.append(name)
            return result

        return call

    def clear():
        logged("clear")()
        connector.metadata_bound = False

    connector.wait_for_save.side_effect = logged("save")
    connector.get_finished.side_effect = logged("poll", ({"sent"}, {"loaded"}))
    connector.clear_connector_metadata.side_effect = clear
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.build_connector_worker_meta.return_value = None
    connector.get_kv_connector_stats.return_value = None
    return connector


def _fake_runner(method, is_async, verify, propose):
    """The runner state sample_tokens reads for one request sampling token 7
    (drafting token 8 when async)."""
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 1},
        finished_req_ids=set(),
        has_structured_output_requests=False,
        num_invalid_spec_tokens={},
    )
    async_source = ([torch.tensor([7, 8])], {0: 0}, None, {"r0": 1})
    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            scheduler_output=scheduler_output,
            draft_chunks=[],
            logits_list=[torch.zeros(1, 16)],
            num_reqs_list=[1],
            spec_decode_metadata_list=[object() if verify else None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            kv_save_nan_check=None,
            mtp_tensor_diagnostics=[],
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=1,
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            all_greedy=True,
            num_tokens_no_spec=[1],
            token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
        ),
        requests={
            "r0": SimpleNamespace(
                num_computed_tokens=0, num_tokens=1, output_token_ids=[]
            )
        },
        speculative_config=(
            None
            if method is None
            else SimpleNamespace(method=method, num_speculative_tokens=1)
        ),
        scheduler_config=SimpleNamespace(async_scheduling=is_async),
        vllm_config=SimpleNamespace(kv_transfer_config=None, speculative_config=None),
        _is_async_drafter=method == "mtp",
        _hidden_nan_detector=None,
        _kv_cache_nan_detector=None,
        is_pooling_model=False,
        drafter=SimpleNamespace(),
        spec_decode_manager=SimpleNamespace(propose_draft_token_ids=propose),
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda _logits: torch.tensor([7]),
        _sample_spec_verify_chunk=lambda *_args: torch.tensor([[7, -1]]),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _modify_prev_results=lambda: None,
        _sync_replicated_drafts_across_tp=lambda drafts: drafts,
        _assemble_async_spec_substitution=lambda *_args: async_source,
        _assemble_async_prefill_bootstrap=lambda *_args: async_source,
        _update_placeholder=lambda *_args: {"r0": 0},
        _stage_timing=MagicMock(),
        _nan_checker=None,
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    runner._async_drafter = runner.drafter
    for name in ("maybe_wait_for_kv_save", "get_finished_kv_transfers"):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _host_copy(chunks, lengths, spec_logprobs=None):
    return AsyncTPUCopyState(
        chunks_cpu=chunks,
        chunk_real_lens=lengths,
        copy_ready_event=MagicMock(),
        spec_logprobs=spec_logprobs,
    )


@pytest.mark.parametrize(
    "method,is_async,verify",
    [
        (None, False, False),
        (None, True, False),
        ("mtp", False, False),
        ("mtp", True, False),
        ("mtp", False, True),
        ("mtp", True, True),
        ("ngram", False, False),
    ],
)
@pytest.mark.parametrize("with_connector", [False, True])
def test_device_drafter_runs_before_kv_connector_save(
    method, is_async, verify, with_connector
):
    events: list[str] = []

    def propose(**kwargs):
        if method == "ngram":
            assert kwargs["sampled_token_ids"] == [[7]]
        events.append("draft")
        return torch.tensor([[8]]) if is_async else None

    runner = _fake_runner(method, is_async, verify, propose)
    connector = _recording_connector(events)

    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
            return_value=with_connector,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=_host_copy),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)

    assert isinstance(output, AsyncTPUModelRunnerOutput) == is_async
    if is_async:
        output = output.get_output()
    assert output.sampled_token_ids == [[7]]

    if not with_connector:
        assert events == ([] if method is None else ["draft"])
        assert output.kv_connector_output is None
        return
    # The device drafter writes the draft layers' KV into the blocks the
    # target just filled, so the connector may save them only afterwards.
    # ngram drafts on the host and writes no KV.
    assert (
        events
        == {
            None: ["save", "poll", "clear"],
            "mtp": ["draft", "save", "poll", "clear"],
            "ngram": ["save", "poll", "clear", "draft"],
        }[method]
    )
    assert output.kv_connector_output.finished_sending == {"sent"}
    assert output.kv_connector_output.finished_recving == {"loaded"}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_mtp_producer_output_waits_for_post_draft_kv(is_async, verify):
    events = []

    def propose(**kwargs):
        events.append("draft")
        return torch.tensor([[8]]) if is_async else None

    runner = _fake_runner("mtp", is_async, verify, propose)
    runner.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    event = MagicMock()
    event.record.side_effect = lambda: events.append("record")
    event.synchronize.side_effect = lambda: events.append("wait")
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=_host_copy),
        patch("vllm_torchtpu.runner.tpu_runner.torch.tpu.Event", return_value=event),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
        if is_async:
            assert events == ["draft", "record"]
            output = output.get_output()
    events.append("scheduler_output")
    assert output.sampled_token_ids == [[7]]
    assert events == ["draft", "record", "wait", "scheduler_output"]


def test_pp_prefill_mtp_hidden_outputs_are_enqueued_before_token_fence(monkeypatch):
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    events = []
    runner = _fake_runner("mtp", False, False, MagicMock())
    runner.vllm_config.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    hidden = torch.tensor([[1.0, 2.0]])
    drafter = MagicMock(spec=Eagle3Proposer)
    drafter.prefill_only.side_effect = lambda *args: events.append("mtp") or [hidden]
    runner._async_drafter = drafter
    runner.execute_model_state.scheduler_output.kv_connector_metadata = None
    connector = _recording_connector(events)
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")

    def enqueue(tensors, *, wait):
        assert len(tensors) == 1 and tensors[0] is hidden
        assert wait is False
        events.append("enqueue_mtp")

    event = MagicMock()
    event.record.side_effect = lambda: events.append("record")
    event.synchronize.side_effect = lambda: events.append("wait")
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=lambda: event), raising=False
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
            "vllm_torchtpu.runner.tpu_runner.synchronize_tensors", side_effect=enqueue
        ),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=_host_copy),
        patch(
            "vllm_torchtpu.spec_decode.prefill_only.handoff_seed_tokens",
            side_effect=lambda tokens, *args: tokens,
        ),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    assert output.sampled_token_ids == [[7]]
    assert events == ["mtp", "enqueue_mtp", "record", "save", "poll", "clear", "wait"]
    runner.spec_decode_manager.propose_draft_token_ids.assert_not_called()
