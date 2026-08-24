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
End-to-end tests for structured (guided) decoding on TPU, without
speculative decoding.

The grammar compilation, FSM advance and bitmask generation all happen in
vLLM core on the host; the TPU runner's only job is applying the packed
int32 bitmask to the logits before sampling. These tests check that job
across the shapes it must handle:
- every constraint type the runner sees identically (choice / json / regex);
- mixed batches (structured + unstructured requests side by side);
- greedy and non-greedy sampling;
- multi-chunk execution (the per-step batch split TPU does when the
  page-table SMEM budget caps requests per forward) -- the row-misalignment
  regression, where a later chunk's structured requests silently lost their
  masks.

For structured outputs combined with speculative decoding, see
test_speculative_decoding.py.

Run with: pytest tests/entrypoints/llm/test_structured_output.py -v
"""

import json
import os
import re
import weakref

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.sampling_params import StructuredOutputsParams

MODEL_NAME = "Qwen/Qwen3-0.6B"

CHOICES = ["Positive", "Negative"]
CHOICE_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=8,
    structured_outputs=StructuredOutputsParams(choice=CHOICES),
)
CHOICE_PROMPT = "Sentiment of 'what a great movie': "

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
    },
    "required": ["name", "age"],
}
JSON_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=64,
    structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA),
)
JSON_PROMPT = ("Generate a JSON object with a name (string) and an age "
               "(integer): ")

REGEX_PATTERN = r"[0-9]{3}-[0-9]{4}"
REGEX_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=16,
    structured_outputs=StructuredOutputsParams(regex=REGEX_PATTERN),
)
REGEX_PROMPT = "A random US phone number without area code: "

UNCONSTRAINED_PARAMS = SamplingParams(temperature=0, max_tokens=16)
UNCONSTRAINED_PROMPT = "The capital of France is"


def _assert_choice(text: str) -> None:
    assert text in CHOICES, f"choice constraint violated: {text!r}"


def _assert_json(text: str) -> None:
    parsed = json.loads(text)
    assert isinstance(parsed, dict), text
    assert isinstance(parsed.get("name"), str), text
    assert isinstance(parsed.get("age"), int), text


def _assert_regex(text: str) -> None:
    assert re.fullmatch(REGEX_PATTERN,
                        text), (f"regex constraint violated: {text!r}")


@pytest.fixture(scope="module")
def llm():
    # The multi-chunk test sends a Python function to the TPU workers via
    # collective_rpc, which needs pickle fallback in vLLM's IPC encoder;
    # the engine (core process) must be spawned with it already set.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=256,
        max_num_seqs=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


def test_choice_greedy(llm: LLM):
    outputs = llm.generate([CHOICE_PROMPT] * 4, CHOICE_PARAMS)
    assert len(outputs) == 4
    for output in outputs:
        _assert_choice(output.outputs[0].text)


def test_json_schema_greedy(llm: LLM):
    outputs = llm.generate([JSON_PROMPT] * 4, JSON_PARAMS)
    assert len(outputs) == 4
    for output in outputs:
        _assert_json(output.outputs[0].text)


def test_regex_greedy(llm: LLM):
    outputs = llm.generate([REGEX_PROMPT] * 4, REGEX_PARAMS)
    assert len(outputs) == 4
    for output in outputs:
        _assert_regex(output.outputs[0].text)


def test_mixed_batch(llm: LLM):
    """Structured and unstructured requests in one batch: each structured
    request gets its own mask row, unstructured rows stay untouched (a
    misaligned scatter shows up as either a violated constraint or an
    unconstrained request going silent/garbled)."""
    prompts = [
        CHOICE_PROMPT, UNCONSTRAINED_PROMPT, JSON_PROMPT, UNCONSTRAINED_PROMPT
    ]
    params = [
        CHOICE_PARAMS, UNCONSTRAINED_PARAMS, JSON_PARAMS, UNCONSTRAINED_PARAMS
    ]
    outputs = llm.generate(prompts, params)
    _assert_choice(outputs[0].outputs[0].text)
    _assert_json(outputs[2].outputs[0].text)
    for i in (1, 3):
        assert outputs[i].outputs[0].text, "unconstrained request went silent"


def test_choice_non_greedy(llm: LLM):
    """The mask guarantee must hold under random sampling too (masked
    logits are -inf, so a banned token's probability is exactly zero)."""
    params = SamplingParams(
        temperature=0.7,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(choice=CHOICES),
    )
    outputs = llm.generate([CHOICE_PROMPT] * 4, params)
    for output in outputs:
        _assert_choice(output.outputs[0].text)


def _force_multi_chunk_cap(self, cap: int = 2):
    runner = self.model_runner
    caps = (runner.num_reqs_max_model_len, runner.num_reqs_most_model_len)
    runner.num_reqs_max_model_len = cap
    if runner.num_reqs_most_model_len is not None:
        runner.num_reqs_most_model_len = cap
    return caps


def _restore_chunk_cap(self, caps):
    runner = self.model_runner
    runner.num_reqs_max_model_len, runner.num_reqs_most_model_len = caps


def test_multi_chunk_row_alignment(llm: LLM):
    """Regression for the multi-chunk bitmask misalignment: the scatter
    used to slice the first `num_reqs` rows of the global buffer for every
    chunk, so from the second chunk on structured requests silently lost
    their masks (and could steal other requests' masks).

    Cap=2 with a 4-request batch forces two chunks per step; structured
    requests are placed in both chunks."""
    saved = llm.llm_engine.collective_rpc(_force_multi_chunk_cap, args=(2, ))
    try:
        prompts = [
            CHOICE_PROMPT, UNCONSTRAINED_PROMPT, CHOICE_PROMPT, JSON_PROMPT
        ]
        params = [
            CHOICE_PARAMS, UNCONSTRAINED_PARAMS, CHOICE_PARAMS, JSON_PARAMS
        ]
        outputs = llm.generate(prompts, params)
        _assert_choice(outputs[0].outputs[0].text)
        _assert_choice(outputs[2].outputs[0].text)
        _assert_json(outputs[3].outputs[0].text)
        assert outputs[1].outputs[0].text, "unconstrained request went silent"
    finally:
        llm.llm_engine.collective_rpc(_restore_chunk_cap, args=(saved[0], ))
