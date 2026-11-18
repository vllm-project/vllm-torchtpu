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
"""Helpers for CPU tests of the runner's output path: the runner state that
`TPUModelRunner.sample_tokens` reads (also for one plain greedy step, as the
min_tokens and penalties tests drive it), a stand-in for the async copy of its
sampled tokens and logprobs, and references for the logprobs it returns."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm import SamplingParams
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


def _sample_tokens_runner(logits_list, num_reqs_list, **overrides):
    """The runner state `TPUModelRunner.sample_tokens` reads for a synchronous
    step without speculative decoding, logprobs, grammar or KV connector.
    Chunk `i` holds `num_reqs_list[i]` requests and their logits
    `logits_list[i]`. The requests are r0, r1, ... in batch order, each with
    one scheduled token and no output yet, and they sample greedily (the
    argmax). Keyword arguments replace runner attributes; tests change the
    nested state (`execute_model_state`, `input_batch`, `requests`) on the
    returned runner. With `scheduler_config` overridden to async scheduling,
    the step parks its tokens through the runner's own `_update_placeholder`
    and returns before the copy is waited on."""
    req_ids = [f"r{i}" for i in range(sum(num_reqs_list))]
    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={r: 1 for r in req_ids},
                finished_req_ids=set(),
                has_structured_output_requests=False,
            ),
            draft_chunks=[],
            logits_list=list(logits_list),
            num_reqs_list=list(num_reqs_list),
            spec_decode_metadata_list=[None] * len(logits_list),
            mamba_state_indices_list=None,
            mamba_boundary_plans=None,
            pooler_output_list=[],
            top_tokens=False,
            kv_save_nan_check=None,
            prompt_logprobs_chunks=[],
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=len(req_ids),
            req_ids=list(req_ids),
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            all_greedy=True,
            no_top_k=True,
            no_top_p=True,
            num_tokens_no_spec=[1] * len(req_ids),
            token_ids_cpu=np.zeros((len(req_ids), 8), dtype=np.int32),
            generators={},
        ),
        requests={
            r: SimpleNamespace(
                req_id=r, num_computed_tokens=0, num_tokens=1, output_token_ids=[]
            )
            for r in req_ids
        },
        num_prompt_logprobs={},
        _pre_async_results=None,
        _pp_handoff=None,
        _dp_step_other_decode=False,
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _is_async_drafter=False,
        _is_dspark_pd_producer=False,
        _fence_spec_kv_under_token_copy=False,
        _adaptive_verification=None,
        _thinking_budget=None,
        is_pooling_model=False,
        drafter=SimpleNamespace(),
        spec_decode_manager=SimpleNamespace(
            propose_draft_token_ids=lambda **_kwargs: None
        ),
        _min_tokens_batch_state=lambda: {},
        _penalties_batch_state=lambda: False,
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda x: torch.argmax(x, dim=-1),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _modify_prev_results=lambda: None,
        _sync_replicated_drafts_across_tp=lambda drafts: drafts,
        _stage_timing=MagicMock(),
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    for name in (
        "_sample_chunk_logits",
        "_sample_plain_chunk",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
        "_build_kv_connector_output",
        "_update_placeholder",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _host_copy_state(chunks, lengths, **kwargs):
    """Stand in for `AsyncTPUCopyState.from_device_chunks` in CPU tests: the
    chunks are already host tensors and the ready event is a mock. Keyword
    arguments reach `AsyncTPUCopyState` unchanged, so no stand-in needs an
    edit when the method gains one."""
    return AsyncTPUCopyState(
        chunks_cpu=chunks,
        chunk_real_lens=lengths,
        copy_ready_event=MagicMock(),
        **kwargs,
    )


def _plain_step_runner(num_output, logits, req_id="r0"):
    """The runner state sample_tokens reads for one greedy request in a plain
    (non-speculative, synchronous) step, picking the argmax of what it gets,
    with the runner's own min_tokens and penalty methods. With `num_output`,
    request `req_id` has that many outputs, min_tokens 4 and stop token 3. No
    request uses penalties; set the InputBatch penalty arrays and
    `no_penalties=False` to turn them on (see test_penalties.py)."""
    runner = _sample_tokens_runner(
        [logits], [1], _min_tokens=min_tokens.MinTokensTracker()
    )
    batch = runner.input_batch
    batch.num_tokens_no_spec = np.array([1])
    batch.num_prompt_tokens = np.array([1])
    batch.no_penalties = True
    batch.repetition_penalties_cpu = np.ones(1, dtype=np.float32)
    batch.presence_penalties_cpu = np.zeros(1, dtype=np.float32)
    batch.frequency_penalties_cpu = np.zeros(1, dtype=np.float32)
    # The staged H2D uploads _apply_penalties makes (host tensors on CPU).
    runner.device = torch.device("cpu")
    runner._registered_h2d_sources = set()
    runner._free_h2d_sources = {}
    runner._unfenced_h2d_sources = []
    if num_output is not None:
        runner._min_tokens.add(req_id, SamplingParams(min_tokens=4, stop_token_ids=[3]))
        runner.requests[req_id] = SimpleNamespace(
            num_computed_tokens=0, num_tokens=1, output_token_ids=[0] * num_output
        )
    for name in (
        "_sample_chunk_logits",
        "_sample_plain_chunk",
        "_mask_min_tokens",
        "_min_tokens_batch_state",
        "_penalties_batch_state",
        "_penalties_chunk",
        "_apply_penalties",
        "_settle_draft_placeholders_for_min_tokens",
        "_num_output_tokens",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _run_sample_tokens(runner):
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False
        ),
        patch.object(
            AsyncTPUCopyState, "from_device_chunks", side_effect=_host_copy_state
        ),
        torch.compiler.set_stance("force_eager"),
    ):
        return TPUModelRunner.sample_tokens(runner, None)


def _blocking_logprobs_lists(chunks):
    """Reference for the logprobs of a step without drafts, built the way
    sample_tokens built them with a blocking copy before the copy was
    deferred: each `(tensors, num_rows)` chunk sliced to its first
    `num_rows` rows, `.tolists()`, then concatenated in chunk order."""
    lists = [
        LogprobsTensors(
            t.logprob_token_ids[:n],
            t.logprobs[:n],
            t.selected_token_ranks[:n],
            t.cu_num_generated_tokens,
        ).tolists()
        for t, n in chunks
    ]
    return LogprobsLists(
        logprob_token_ids=np.concatenate([lp.logprob_token_ids for lp in lists]),
        logprobs=np.concatenate([lp.logprobs for lp in lists]),
        sampled_token_ranks=np.concatenate([lp.sampled_token_ranks for lp in lists]),
    )


def _assert_same_logprobs_lists(actual, expected):
    """Same rows, values and dtypes, one generated token per request."""
    assert actual.cu_num_generated_tokens is None
    assert expected.cu_num_generated_tokens is None
    for field in ("logprob_token_ids", "logprobs", "sampled_token_ranks"):
        a, e = getattr(actual, field), getattr(expected, field)
        assert a.dtype == e.dtype, field
        assert np.array_equal(a, e), field
