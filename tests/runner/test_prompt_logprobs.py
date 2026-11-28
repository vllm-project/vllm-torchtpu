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
from vllm.distributed import parallel_state

from vllm_torchtpu.runner import prompt_logprobs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

HIDDEN, VOCAB = 8, 32
PROMPTS = {
    "a": [5, 9, 2, 30, 7, 7, 1],
    "b": [4, 4, 8, 3, 0],
    "c": [11, 6, 6, 21],
    # Long enough for more than one 16-row logits call.
    "d": list(range(3, 22)),
}
# "b" asks for none; "d" only in the DP lockstep test.
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
        model_config=SimpleNamespace(
            max_logprobs=max_logprobs,
            logprobs_mode="raw_logprobs",
            get_hidden_size=lambda: HIDDEN,
        ),
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
        _dp_lockstep_enabled=lambda: False,
        _dp_target_bucket=None,
        _dp_step_num_chunks=0,
        _dp_step_prompt_logprob_rows=0,
        _hidden_states_dtype=torch.float32,
        _pp_is_last=True,
    )
    for name in (
        "compute_selected_logits",
        "gather_logprobs",
        "_prompt_logprob_rows_per_call",
        "_dp_prompt_logprob_calls",
        "_dp_prompt_logprob_dummies",
    ):
        setattr(runner, name, functools.partial(getattr(TPUModelRunner, name), runner))
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _run_steps(runner, schedule, num_tokens_padded=None):
    """Plan each step's chunks (each a dict req_id -> scheduled tokens, in batch
    order), score them and emit; return what each step completed."""
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
            chunk_hidden = torch.full(
                (num_tokens_padded or int(qsl[-1]) + 3, HIDDEN), float("nan")
            )
            for i, req_id in enumerate(req_ids):
                first = computed[req_id]
                chunk_hidden[qsl[i] : qsl[i + 1]] = HIDDEN_STATES[req_id][
                    first : first + chunk_tokens[req_id]
                ]
            runner.query_start_loc_np = qsl
            chunk = TPUModelRunner._prompt_logprobs_chunk(
                runner,
                SimpleNamespace(num_scheduled_tokens=chunk_tokens),
                start_index,
                len(req_ids),
            )
            if chunk is not None:
                chunks.append((chunk, chunk_hidden))
            start_index += len(req_ids)
        with torch.compiler.set_stance("force_eager"):
            TPUModelRunner._score_prompt_logprobs(runner, chunks)
        completed_per_step.append(
            prompt_logprobs.emit(
                [chunk for chunk, _ in chunks],
                runner.requests,
                runner.num_prompt_logprobs,
            )
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
        ("selected_logits_hidden_states", "--prefill-context-parallel-size 1"),
        ("lora", "--enable-lora"),
    ],
)
def test_prompt_logprobs_are_skipped_where_unsupported(flag, fix):
    """The reason names the fix, and the step does no device work and returns
    no prompt logprobs instead of stopping the engine."""
    flags = dict.fromkeys(["selected_logits_hidden_states", "lora"], False)
    reason = prompt_logprobs.unsupported_reason(**{**flags, flag: True})
    assert reason is not None and fix in reason
    runner = _runner(_prompt_logprobs_unsupported=reason, compute_selected_logits=None)

    assert _run_steps(runner, [[{"a": 7, "c": 4}]]) == [{}]


def test_prompt_logprobs_are_unsupported_with_the_pcp_manager():
    """USE_TPU_PCP_MANAGER leaves no sequence layout planner; the reason comes
    from the PCP manager, which selects the logits rows itself."""
    runner = SimpleNamespace(
        pcp_manager=SimpleNamespace(uses_selected_logits_hidden_states=True),
        sequence_layout_planner=None,
        lora_config=None,
    )
    runner.get_pcp_planner = functools.partial(TPUModelRunner.get_pcp_planner, runner)

    reason = TPUModelRunner._prompt_logprobs_unsupported_reason(runner)

    assert reason is not None and "prefill context parallelism" in reason


@pytest.mark.parametrize("rows_per_chunk", [[9], [1, 1], [8, 1], [7, 7, 7], [0, 5]])
def test_max_logits_calls_bounds_every_split(rows_per_chunk):
    calls = sum(-(-rows // 8) for rows in rows_per_chunk)
    bound = prompt_logprobs.max_logits_calls(
        sum(rows_per_chunk), 8, len(rows_per_chunk)
    )
    assert calls <= bound
    assert prompt_logprobs.max_logits_calls(0, 8, 3) == 0


def test_dp_lockstep_engines_run_the_same_logits_calls():
    """Under DP lockstep every engine must issue the same logits collectives:
    an engine scoring prompt rows, a busy engine with none and an idle engine
    run the same calls at the same shapes, and the scoring engine's prompt
    logprobs are unchanged."""
    bucket = 32
    num_prompt_logprobs = {"d": 1, "a": 2}
    # What _dp_coordinated_step all-reduces: the scoring engine's 18 + 6 rows,
    # in calls of 16, the last with 8 real rows.
    dp_state = {
        "max_num_reqs": 16,
        "_dp_lockstep_enabled": lambda: True,
        "_dp_target_bucket": bucket,
        "_dp_step_num_chunks": 1,
        "_dp_step_prompt_logprob_rows": 24,
    }
    calls = {}

    def recording(name, runner):
        calls[name] = []

        def compute_selected_logits(hidden, indices):
            calls[name].append((tuple(hidden.shape), tuple(indices.shape)))
            return TPUModelRunner.compute_selected_logits(runner, hidden, indices)

        runner.compute_selected_logits = compute_selected_logits
        return runner

    scoring = recording(
        "scoring", _runner(num_prompt_logprobs=dict(num_prompt_logprobs), **dp_state)
    )
    (completed,) = _run_steps(scoring, [[{"d": 19, "b": 5, "a": 7}]], bucket)
    busy = recording("busy", _runner(num_prompt_logprobs={}, **dp_state))
    assert _run_steps(busy, [[{"b": 5}]], bucket) == [{}]
    idle = recording(
        "idle",
        _runner(
            _execute_dp_mm_encoder=lambda: None,
            _run_dp_dummy_chunk=lambda _bucket: None,
            _is_async_drafter=False,
            **dp_state,
        ),
    )
    with torch.compiler.set_stance("force_eager"):
        TPUModelRunner._run_dp_idle_pairing(idle, bucket, 1)

    expected_calls = [((bucket, HIDDEN), (16,))] * 2
    assert calls["scoring"] == calls["busy"] == calls["idle"] == expected_calls
    for req_id, num_logprobs in num_prompt_logprobs.items():
        ids, logprobs, _ = _expected(req_id, num_logprobs)
        assert torch.equal(completed[req_id].logprob_token_ids.long(), ids)
        torch.testing.assert_close(
            completed[req_id].logprobs.double(), logprobs, rtol=0, atol=1e-5
        )


def test_every_dp_rank_learns_the_largest_prompt_row_count(monkeypatch):
    """The prompt rows ride _dp_coordinated_step's all-reduce (here the MAX of
    what two simulated ranks send), so every rank derives the same bound."""

    def step(rank, rows, all_reduce):
        runner = object.__new__(TPUModelRunner)
        runner._dp_lockstep_enabled = lambda: True
        runner.num_tokens_paddings = [16]
        group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=None)
        monkeypatch.setattr(parallel_state, "get_dp_group", lambda: group)
        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        runner._dp_coordinated_step(16, 1, local_prompt_logprob_rows=rows)
        return runner

    sent = []
    for rank, rows in enumerate([9, 0]):
        step(rank, rows, lambda t, op=None, group=None: sent.append(t.clone()))
    reduced = torch.stack(sent).amax(dim=0)
    runners = [
        step(rank, rows, lambda t, op=None, group=None: t.copy_(reduced))
        for rank, rows in enumerate([9, 0])
    ]
    assert [r._dp_step_prompt_logprob_rows for r in runners] == [9, 9]


@pytest.mark.parametrize(
    "lockstep, pp_is_last, rows",
    [(True, True, 18 + 3), (False, True, 0), (True, False, 0)],
)
def test_dp_local_prompt_logprob_rows(lockstep, pp_is_last, rows):
    """d scores 18 rows, a finishes its prompt (rows 3-5) and b asks for none;
    only the last pipeline stage of a lockstep engine counts them."""
    runner = _runner(
        _dp_lockstep_enabled=lambda: lockstep,
        _pp_is_last=pp_is_last,
        num_prompt_logprobs={"d": 1, "a": 2},
        input_batch=SimpleNamespace(
            num_reqs=3,
            req_ids=["d", "b", "a"],
            num_computed_tokens_cpu=np.array([0, 0, 3]),
        ),
    )
    step = SimpleNamespace(num_scheduled_tokens={"d": 19, "b": 5, "a": 4})
    assert TPUModelRunner._dp_local_prompt_logprob_rows(runner, step) == rows


def test_last_logits_call_pads_to_its_own_bucket_outside_lockstep():
    """Outside DP lockstep, 24 rows in calls of 16 end with an 8-row call."""
    sizes = []
    runner = _runner(max_num_reqs=16, num_prompt_logprobs={"d": 1, "a": 2})

    def compute_selected_logits(hidden, indices):
        sizes.append(indices.shape[0])
        return TPUModelRunner.compute_selected_logits(runner, hidden, indices)

    runner.compute_selected_logits = compute_selected_logits
    _run_steps(runner, [[{"d": 19, "a": 7}]], 32)
    assert sizes == [16, 8]


@pytest.mark.parametrize(
    ("logprobs_mode", "unnormalized"),
    [
        ("raw_logprobs", False),
        ("processed_logprobs", False),
        ("raw_logits", True),
        ("processed_logits", True),
    ],
)
def test_prompt_logprobs_honor_logprobs_mode(logprobs_mode, unnormalized):
    """Prompt tokens skip sampling processors, so processed_* equals raw_* and
    *_logits returns unnormalized prompt logits."""
    runner = _runner(max_logprobs=3)
    runner.model_config.logprobs_mode = logprobs_mode

    (completed,) = _run_steps(runner, [[{"a": 7, "b": 5, "c": 4}]])

    for req_id, num_logprobs in NUM_PROMPT_LOGPROBS.items():
        result = completed[req_id]
        prompt = torch.tensor(PROMPTS[req_id])
        logits = (HIDDEN_STATES[req_id][:-1] @ WEIGHT.T).double()
        ids = torch.cat(
            [prompt[1:, None], logits.topk(num_logprobs, dim=-1).indices], dim=1
        )
        ref = (logits if unnormalized else torch.log_softmax(logits, dim=-1)).gather(
            -1, ids
        )
        assert torch.equal(result.logprob_token_ids.long(), ids)
        torch.testing.assert_close(result.logprobs.double(), ref, rtol=0, atol=1e-5)
