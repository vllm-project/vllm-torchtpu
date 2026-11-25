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
"""
End-to-end tests for per-request ``thinking_token_budget`` on TPU.

A real reasoning model served with ``reasoning_parser="qwen3"`` exercises what
the CPU tests in tests/runner/test_thinking_budget.py cannot: vLLM deriving
the reasoning marker token IDs from the tokenizer
(``ReasoningConfig.initialize_token_ids``), and the compiled forcing kernels
running through ``TPUModelRunner.sample_tokens`` at real logits shapes, on plain
decode and on ngram speculative decoding.

Mirrors upstream vLLM's
tests/entrypoints/openai/chat_completion/test_thinking_token_budget.py: count
the generated tokens between ``<think>`` and the first ``</think>`` after it.

Run with: pytest tests/entrypoints/llm/test_thinking_budget.py -v
"""

import weakref

import pytest
from tpu_release import wait_for_tpu_release
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "Qwen/Qwen3-0.6B"
# Qwen3's <think> / </think>, the qwen3 reasoning parser's start and end
# markers (one token each).
THINK_START_ID = 151667
THINK_END_ID = 151668

# The ngram prompt repeats a sentence the model tends to quote while it
# thinks, so prompt lookup proposes drafts inside the thinking block.
MESSAGES = [
    [{"role": "user", "content": "What is 1+1? Be concise."}],
    [
        {
            "role": "user",
            "content": "Repeat this sentence twice: the quick brown fox jumps "
            "over the lazy dog.",
        }
    ],
]
BUDGETS = (0, 5, 12)
# Room for the largest budget, the forced </think> and an answer.
MAX_TOKENS = 64

NGRAM_CONFIG = {
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}


def _thinking_span(token_ids: list[int]) -> tuple[int, int | None] | None:
    """(tokens after the last ``<think>`` before the first ``</think>``, index
    of that ``</think>`` or None if thinking never closed), or None if there
    is no ``<think>``."""
    starts = [i for i, tok in enumerate(token_ids) if tok == THINK_START_ID]
    if not starts:
        return None
    first = starts[-1] + 1
    for i in range(first, len(token_ids)):
        if token_ids[i] == THINK_END_ID:
            return i - first, i
    return len(token_ids) - first, None


@pytest.fixture(scope="module", params=["plain", "ngram"])
def llm(request):
    # ngram is sync-only: vLLM rejects ngram with async scheduling.
    spec_kwargs = (
        {"speculative_config": NGRAM_CONFIG, "async_scheduling": False}
        if request.param == "ngram"
        else {}
    )
    # Prompts are ~20 tokens and at most 4 run at once, so no step needs more
    # than 128 tokens.
    llm = LLM(
        model=MODEL_NAME,
        reasoning_parser="qwen3",
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        **spec_kwargs,
    )
    # pytest caches the fixture, so hand out a weakref.proxy to let the engine
    # be garbage collected at teardown.
    yield weakref.proxy(llm)

    # The other configuration's engine starts right after this teardown.
    llm.llm_engine.engine_core.shutdown()
    del llm
    cleanup_dist_env_and_memory()
    wait_for_tpu_release()


def test_marker_token_ids(llm: LLM):
    """The constants above are what ``ReasoningConfig.initialize_token_ids``
    derives from the qwen3 parser's marker strings."""
    tokenizer = llm.get_tokenizer()
    assert tokenizer.encode("<think>", add_special_tokens=False) == [THINK_START_ID]
    assert tokenizer.encode("</think>", add_special_tokens=False) == [THINK_END_ID]


def test_budget_closes_thinking_and_answer_continues(llm: LLM):
    """Greedy requests with ``thinking_token_budget`` think for exactly the
    budget, then emit ``</think>`` and keep answering; unbudgeted requests in
    the same batch are the control showing the model would think longer."""
    conversations = []
    params = []
    for messages in MESSAGES:
        for budget in (*BUDGETS, None):
            conversations.append(messages)
            params.append(
                SamplingParams(
                    temperature=0, max_tokens=MAX_TOKENS, thinking_token_budget=budget
                )
            )
    outputs = llm.chat(
        conversations,
        params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": True},
    )
    assert len(outputs) == len(params)

    for output, sampling_params in zip(outputs, params):
        budget = sampling_params.thinking_token_budget
        assert output.prompt_token_ids is not None
        token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
        span = _thinking_span(token_ids)
        assert span is not None, f"no <think> in {output.outputs[0].text!r}"
        num_thinking, end_idx = span
        if budget is None:
            # Negative control: without a budget the model thinks past the
            # largest budget, so the budgeted requests are evidence.
            assert num_thinking > max(BUDGETS), output.outputs[0].text
            continue
        assert end_idx is not None, (
            f"budget {budget}: thinking never closed: {output.outputs[0].text!r}"
        )
        assert num_thinking == budget, (
            f"budget {budget}: {num_thinking} thinking tokens: "
            f"{output.outputs[0].text!r}"
        )
        assert len(token_ids) > end_idx + 1, (
            f"budget {budget}: nothing generated after </think>"
        )
