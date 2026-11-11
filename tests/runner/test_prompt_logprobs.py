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
"""Prompt logprobs through TPUModelRunner._prompt_logprobs_chunk and emit.

The runner's own compute_selected_logits and gather_logprobs run eagerly on
CPU over a linear LM head, and every step's result is checked against a
float64 log-softmax of the full prompt.
"""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import prompt_logprobs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

HIDDEN, VOCAB = 8, 32
PROMPTS = {
    "a": [5, 9, 2, 30, 7, 7, 1],
    "b": [4, 4, 8, 3, 0],
    "c": [11, 6, 6, 21],
}
# "b" asks for none.
NUM_PROMPT_LOGPROBS = {"a": 2, "c": 0}
_GEN = torch.Generator().manual_seed(0)
WEIGHT = torch.randn(VOCAB, HIDDEN, generator=_GEN)
# The hidden state at every prompt position.
HIDDEN_STATES = {
    r: torch.randn(len(p), HIDDEN, generator=_GEN) for r, p in PROMPTS.items()
}


def _runner(max_logprobs=3, **overrides):
    runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda h: h @ WEIGHT.T),
        is_pooling_model=False,
        model_config=SimpleNamespace(max_logprobs=max_logprobs),
        max_num_reqs=8,
        device=torch.device("cpu"),
        requests={
            req_id: SimpleNamespace(
                prompt_token_ids=prompt, in_progress_prompt_logprobs_cpu=None
            )
            for req_id, prompt in PROMPTS.items()
        },
        num_prompt_logprobs=dict(NUM_PROMPT_LOGPROBS),
        input_batch=SimpleNamespace(req_ids=[], num_computed_tokens_cpu=None),
        query_start_loc_np=None,
        _prompt_logprobs_unsupported=None,
    )
    runner.compute_selected_logits = functools.partial(
        TPUModelRunner.compute_selected_logits, runner
    )
    runner.gather_logprobs = functools.partial(TPUModelRunner.gather_logprobs, runner)
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _run_steps(runner, schedule):
    """Run each step's chunks (each a dict req_id -> scheduled tokens, in batch
    order) and emit; return what each step completed."""
    computed = dict.fromkeys(PROMPTS, 0)
    completed_per_step = []
    for step in schedule:
        runner.input_batch.req_ids = [r for chunk in step for r in chunk]
        runner.input_batch.num_computed_tokens_cpu = np.array(
            [computed[r] for r in runner.input_batch.req_ids]
        )
        chunks, start_index = [], 0
        for chunk_tokens in step:
            req_ids = list(chunk_tokens)
            qsl = np.concatenate([[0], np.cumsum([chunk_tokens[r] for r in req_ids])])
            # Padding rows are NaN, so scoring one fails the value checks.
            chunk_hidden = torch.full((int(qsl[-1]) + 3, HIDDEN), float("nan"))
            for i, req_id in enumerate(req_ids):
                first = computed[req_id]
                chunk_hidden[qsl[i] : qsl[i + 1]] = HIDDEN_STATES[req_id][
                    first : first + chunk_tokens[req_id]
                ]
            runner.query_start_loc_np = qsl
            with torch.compiler.set_stance("force_eager"):
                chunk = TPUModelRunner._prompt_logprobs_chunk(
                    runner,
                    SimpleNamespace(num_scheduled_tokens=chunk_tokens),
                    chunk_hidden,
                    start_index,
                    len(req_ids),
                )
            if chunk is not None:
                chunks.append(chunk)
            start_index += len(req_ids)
        completed_per_step.append(
            prompt_logprobs.emit(chunks, runner.requests, runner.num_prompt_logprobs)
        )
        for chunk_tokens in step:
            for req_id, num_tokens in chunk_tokens.items():
                computed[req_id] += num_tokens
    return completed_per_step


def _expected(req_id, num_logprobs):
    """Token ids, logprobs and ranks of every prompt row, in float64."""
    prompt = torch.tensor(PROMPTS[req_id])
    logits = (HIDDEN_STATES[req_id][:-1] @ WEIGHT.T).double()
    ids = torch.cat(
        [prompt[1:, None], logits.topk(num_logprobs, dim=-1).indices], dim=1
    )
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, ids)
    ranks = (logits >= logits.gather(-1, prompt[1:, None])).sum(-1)
    return ids, logprobs, ranks


# -1 is --max-logprobs -1: gather_logprobs keeps no top-k, so a request gets
# the prompt token's logprob alone.
@pytest.mark.parametrize("max_logprobs", [3, -1])
@pytest.mark.parametrize(
    "schedule, completing_step",
    [
        pytest.param([[{"a": 7, "b": 5, "c": 4}]], 0, id="whole_prompts"),
        # Every row is scored in step 0; vLLM returns them with the next chunk.
        pytest.param([[{"a": 6, "c": 3}], [{"a": 1, "c": 1}]], 1, id="one_token_short"),
        # Chunked prefill, with a step split into two forward chunks
        # (start_index > 0 for the second).
        pytest.param(
            [[{"b": 5, "a": 3}, {"c": 2}], [{"a": 4}, {"c": 2}]],
            1,
            id="two_chunks_per_step",
        ),
    ],
)
def test_prompt_logprobs_match_log_softmax(schedule, completing_step, max_logprobs):
    """Row i holds the logprob of prompt token i + 1 from hidden state i, the
    top-k and the rank; returned once, when the prompt completes."""
    runner = _runner(max_logprobs)

    completed_per_step = _run_steps(runner, schedule)

    for step, completed in enumerate(completed_per_step):
        expected = set(NUM_PROMPT_LOGPROBS) if step == completing_step else set()
        assert set(completed) == expected
    assert runner.num_prompt_logprobs == {}
    for req_id, num_logprobs in NUM_PROMPT_LOGPROBS.items():
        assert runner.requests[req_id].in_progress_prompt_logprobs_cpu is None
        result = completed_per_step[completing_step][req_id]
        top_k = min(num_logprobs, max(max_logprobs, 0))
        ids, logprobs, ranks = _expected(req_id, top_k)
        assert result.logprobs.shape == (len(PROMPTS[req_id]) - 1, top_k + 1)
        assert torch.equal(result.logprob_token_ids.long(), ids)
        torch.testing.assert_close(
            result.logprobs.double(), logprobs, rtol=0, atol=1e-5
        )
        assert torch.equal(result.selected_token_ranks.long(), ranks)


@pytest.mark.parametrize(
    "flag, fix",
    [
        ("dp_vocab_logits", "TPU_DP_VOCAB_PARALLEL"),
        ("selected_logits_hidden_states", "--prefill-context-parallel-size 1"),
        ("lora", "--enable-lora"),
    ],
)
def test_prompt_logprobs_are_skipped_where_unsupported(flag, fix):
    """The reason names the fix, and the step does no device work and returns
    no prompt logprobs instead of stopping the engine."""
    flags = dict.fromkeys(
        ["dp_vocab_logits", "selected_logits_hidden_states", "lora"], False
    )
    reason = prompt_logprobs.unsupported_reason(**{**flags, flag: True})
    assert reason is not None and fix in reason
    runner = _runner(_prompt_logprobs_unsupported=reason, compute_selected_logits=None)

    assert _run_steps(runner, [[{"a": 7, "c": 4}]]) == [{}]
