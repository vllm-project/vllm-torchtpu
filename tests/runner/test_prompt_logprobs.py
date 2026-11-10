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

HIDDEN, VOCAB, MAX_LOGPROBS = 8, 32, 3
PROMPTS = {
    "a": [5, 9, 2, 30, 7, 7, 1],
    "b": [4, 4, 8, 3, 0],
    "c": [11, 6, 6, 21],
}
# "b" asks for none.
NUM_PROMPT_LOGPROBS = {"a": 2, "c": 0}


def _runner(**overrides):
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(VOCAB, HIDDEN, generator=gen)
    runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda h: h @ weight.T),
        is_pooling_model=False,
        model_config=SimpleNamespace(max_logprobs=MAX_LOGPROBS),
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
        _dp_vocab_logits=False,
        speculative_config=None,
        sequence_layout_planner=SimpleNamespace(
            uses_selected_logits_hidden_states=False
        ),
        weight=weight,
    )
    runner.compute_selected_logits = functools.partial(
        TPUModelRunner.compute_selected_logits, runner
    )
    runner.gather_logprobs = functools.partial(TPUModelRunner.gather_logprobs, runner)
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _run_steps(runner, hidden, schedule):
    """Run each step's chunk (req_id -> scheduled tokens) and emit; return the
    request ids each step completed."""
    computed = dict.fromkeys(PROMPTS, 0)
    completed_per_step = []
    for step in schedule:
        req_ids = list(step)
        qsl = np.concatenate([[0], np.cumsum([step[r] for r in req_ids])])
        # Padding rows hold noise, so reading one shows up in the values.
        chunk_hidden = torch.randn(int(qsl[-1]) + 3, HIDDEN)
        for i, req_id in enumerate(req_ids):
            first = computed[req_id]
            chunk_hidden[qsl[i] : qsl[i + 1]] = hidden[req_id][
                first : first + step[req_id]
            ]
        runner.input_batch.req_ids = req_ids
        runner.input_batch.num_computed_tokens_cpu = np.array(
            [computed[r] for r in req_ids]
        )
        runner.query_start_loc_np = qsl
        with torch.compiler.set_stance("force_eager"):
            chunk = TPUModelRunner._prompt_logprobs_chunk(
                runner,
                SimpleNamespace(num_scheduled_tokens=step),
                chunk_hidden,
                0,
                len(req_ids),
            )
        completed = prompt_logprobs.emit(
            [chunk] if chunk is not None else [],
            runner.requests,
            runner.num_prompt_logprobs,
        )
        completed_per_step.append(completed)
        for req_id in req_ids:
            computed[req_id] += step[req_id]
    return completed_per_step


@pytest.mark.parametrize(
    "schedule, completing_step",
    [
        pytest.param([{"a": 7, "b": 5, "c": 4}], 0, id="whole_prompts"),
        pytest.param([{"a": 3, "b": 5, "c": 2}, {"a": 4, "c": 2}], 1, id="chunked"),
        # Every row is scored in step 0; vLLM returns them with the next chunk.
        pytest.param([{"a": 6, "c": 3}, {"a": 1, "c": 1}], 1, id="one_token_short"),
    ],
)
def test_prompt_logprobs_match_log_softmax(schedule, completing_step):
    """Row i holds the logprob of prompt token i + 1 from hidden state i, the
    top-k and the rank; returned once, when the prompt completes."""
    runner = _runner()
    gen = torch.Generator().manual_seed(1)
    hidden = {r: torch.randn(len(p), HIDDEN, generator=gen) for r, p in PROMPTS.items()}

    completed_per_step = _run_steps(runner, hidden, schedule)

    for step, completed in enumerate(completed_per_step):
        expected = set(NUM_PROMPT_LOGPROBS) if step == completing_step else set()
        assert set(completed) == expected
    assert runner.num_prompt_logprobs == {}
    for req_id, num_logprobs in NUM_PROMPT_LOGPROBS.items():
        assert runner.requests[req_id].in_progress_prompt_logprobs_cpu is None
        result = completed_per_step[completing_step][req_id]
        prompt = torch.tensor(PROMPTS[req_id])
        logits = (hidden[req_id][:-1] @ runner.weight.T).double()
        logprobs = torch.log_softmax(logits, dim=-1)
        num_rows = len(prompt) - 1
        assert result.logprobs.shape == (num_rows, num_logprobs + 1)
        expected_ids = torch.cat(
            [prompt[1:, None], logits.topk(num_logprobs, dim=-1).indices], dim=1
        )
        assert torch.equal(result.logprob_token_ids.long(), expected_ids)
        torch.testing.assert_close(
            result.logprobs.double(),
            logprobs.gather(-1, expected_ids),
            rtol=0,
            atol=1e-5,
        )
        target_logits = logits.gather(-1, prompt[1:, None])
        assert torch.equal(
            result.selected_token_ranks.long(), (logits >= target_logits).sum(-1)
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"_dp_vocab_logits": True}, "TPU_DP_VOCAB_PARALLEL"),
        ({"speculative_config": object()}, "speculative decoding"),
        (
            {
                "sequence_layout_planner": SimpleNamespace(
                    uses_selected_logits_hidden_states=True
                )
            },
            "prefill context parallelism",
        ),
    ],
)
def test_prompt_logprobs_raise_where_unsupported(overrides, message):
    """Unsupported configurations raise only when a request asks."""
    hidden = {r: torch.zeros(len(p), HIDDEN) for r, p in PROMPTS.items()}
    with pytest.raises(NotImplementedError, match=message):
        _run_steps(_runner(**overrides), hidden, [{"a": 7}])
    assert _run_steps(_runner(**overrides), hidden, [{"b": 5}]) == [{}]


def test_prompt_logprobs_beyond_max_logprobs_raise():
    """vLLM stores prompt_logprobs=-1 as the vocabulary size."""
    hidden = {r: torch.zeros(len(p), HIDDEN) for r, p in PROMPTS.items()}
    runner = _runner(num_prompt_logprobs={"a": VOCAB})
    with pytest.raises(NotImplementedError, match="max_logprobs"):
        _run_steps(runner, hidden, [{"a": 7}])
