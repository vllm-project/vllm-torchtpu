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
`TPUModelRunner.sample_tokens` reads, a stand-in for the async copy of its
sampled tokens and logprobs, and references for the logprobs it returns."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

from vllm_torchtpu.runner import min_tokens, repetition_penalty
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


def _sample_tokens_runner(logits_list, num_reqs_list, **overrides):
    """The runner state `TPUModelRunner.sample_tokens` reads for a synchronous
    step without speculative decoding, logprobs, grammar, `repetition_penalty`
    or KV connector.
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
        _repetition_penalty=SimpleNamespace(begin_step=lambda _batch: False),
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


def plain_step_runner(logits: torch.Tensor) -> SimpleNamespace:
    """`_sample_tokens_runner` state for one chunk with the greedy request
    `r0`, with the runner's own `min_tokens` and `repetition_penalty` trackers
    and methods and neither set on the request. Tests change the fields they
    need."""
    runner = _sample_tokens_runner(
        [logits],
        [1],
        _min_tokens=min_tokens.MinTokensTracker(),
        _repetition_penalty=repetition_penalty.RepetitionPenaltyTracker(
            logits.shape[1],
            max(8, logits.shape[0]),
            async_scheduling=False,
            speculative_decoding=False,
        ),
    )
    request = runner.requests["r0"]
    request.prompt_token_ids = None
    batch = runner.input_batch
    batch.repetition_penalties_reqs = set()
    batch.repetition_penalties_cpu = np.ones(1, dtype=np.float32)
    batch.req_output_token_ids = [request.output_token_ids]
    for name in (
        "_mask_min_tokens",
        "_min_tokens_batch_state",
        "_settle_draft_placeholders_for_min_tokens",
        "_num_output_tokens",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def run_sample_tokens(runner):
    """`TPUModelRunner.sample_tokens` on `runner`, eager, with the D2H copy of
    the sampled tokens replaced by a host copy."""
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
