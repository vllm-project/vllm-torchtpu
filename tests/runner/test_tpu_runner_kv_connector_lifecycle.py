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


def _fake_runner(method, is_async, verify, propose, fence=False):
    """The runner state sample_tokens reads for one request sampling token 7
    (drafting token 8 when async)."""
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"r0": 1},
        finished_req_ids=set(),
        has_structured_output_requests=False,
    )
    async_source = ([torch.tensor([7, 8])], {0: 0}, None, {"r0": 1})
    runner = SimpleNamespace(
        _pp_wave=None,
        execute_model_state=SimpleNamespace(
            scheduler_output=scheduler_output,
            draft_chunks=[],
            logits_list=[torch.zeros(1, 16)],
            num_reqs_list=[1],
            spec_decode_metadata_list=[object() if verify else None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
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
        _is_async_drafter=method == "mtp",
        _mtp_prefill_only=False,
        _fence_spec_kv_under_token_copy=fence,
        _is_dspark_pd_producer=False,
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
def test_spec_kv_fence_runs_after_drafter_before_connector_save(is_async):
    """A Stage-3 producer that registers KV blocks ahead of the finish
    enqueues the drafter's graph through its draft-token output and re-arms
    the sampled-token copy's ready event behind it, after the drafter and
    before the connector save, so the first token leaving the device proves
    the step's KV is written."""
    events: list[str] = []

    def propose(**kwargs):
        events.append("draft")
        return torch.tensor([[8]]) if is_async else None

    def fence(tensors, *, wait):
        # The drafter's device output (the draft tokens) roots the enqueue.
        assert len(tensors) == 1 and tensors[0].tolist() == [[8]]
        assert wait is False
        events.append("fence")

    def host_copy(chunks, lengths, spec_logprobs=None):
        state = _host_copy(chunks, lengths, spec_logprobs)
        state.fence_pending_device_work = lambda: events.append("rearm")
        return state

    runner = _fake_runner("mtp", is_async, False, propose, fence=True)
    connector = _recording_connector(events)
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group",
            return_value=True,
        ),
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group",
            return_value=connector,
        ),
        patch("vllm_torchtpu.runner.tpu_runner.synchronize_tensors", side_effect=fence),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)

    if is_async:
        output = output.get_output()
    assert output.sampled_token_ids == [[7]]
    expected = ["draft", "fence", "rearm"] if is_async else ["draft", "rearm"]
    assert events == expected + ["save", "poll", "clear"]
