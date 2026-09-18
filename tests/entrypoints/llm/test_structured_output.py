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
- a recursive JSON schema (cyclic $ref), which keeps a grammar stack open for
  the whole generation instead of closing after a few tokens;
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

# Shared between the schema and its assertions below, so the two cannot
# drift apart.
AST_DESCRIPTION_PATTERN = r"[A-Za-z0-9_ -]{5,50}"
AST_OPERATORS = ["ADD", "SUB", "MUL", "DIV", "POW", "MOD"]
AST_SUB_OPERATORS = ["ADD", "SUB", "MUL"]

# SubTree references itself, so the grammar compiler has to handle a cyclic
# $ref rather than inlining a fixed number of levels. The top-level operands
# are SubTree refs rather than anyOf[number, SubTree]: with the union, greedy
# decoding takes the cheapest branch and emits a flat "operands": [1, 2],
# which would leave the $ref path untested on every run.
COMPLEX_AST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ExpressionTree",
    "type": "object",
    "properties": {
        "operator": {
            "type": "string",
            "enum": AST_OPERATORS,
        },
        "description": {
            "type": "string",
            "pattern": f"^{AST_DESCRIPTION_PATTERN}$",
        },
        "operands": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/SubTree"
            },
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "required": ["operator", "description", "operands"],
    "$defs": {
        "SubTree": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "enum": AST_SUB_OPERATORS
                },
                "operands": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["operator", "operands"],
        }
    },
}
# Guaranteed nesting makes the output longer than the flat JSON test's, and
# recursion means the model chooses when to stop. Budget accordingly: the
# fixture caps max_model_len at 256, and the prompt is ~15 tokens.
COMPLEX_AST_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=192,
    structured_outputs=StructuredOutputsParams(json=COMPLEX_AST_SCHEMA),
)
COMPLEX_AST_PROMPT = "Generate a math expression AST in JSON format: "

UNCONSTRAINED_PARAMS = SamplingParams(temperature=0, max_tokens=16)
UNCONSTRAINED_PROMPT = "The capital of France is"


def _assert_choice(text: str) -> None:
    assert text in CHOICES, f"choice constraint violated: {text!r}"


def _assert_json(text: str) -> None:
    parsed = json.loads(text)
    assert isinstance(parsed, dict), text
    assert isinstance(parsed.get("name"), str), text
    assert isinstance(parsed.get("age"), int), text


def _assert_number(value: object, text: str) -> None:
    # bool is a subclass of int, so a bare isinstance(value, (int, float))
    # would accept a JSON `true` as a number -- precisely the kind of leak a
    # dropped bitmask produces, and the one this suite exists to catch.
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"operand must be a number: {value!r} in {text!r}")


def _assert_sub_tree(node: object, text: str) -> None:
    assert isinstance(node, dict), f"expected a SubTree object in {text!r}"
    assert node.get("operator") in AST_SUB_OPERATORS, text
    operands = node.get("operands")
    assert isinstance(operands, list) and len(operands) == 2, text
    for op in operands:
        # SubTree.operands is anyOf[number, SubTree], so recurse on objects.
        if isinstance(op, dict):
            _assert_sub_tree(op, text)
        else:
            _assert_number(op, text)


def _assert_complex_ast_json(text: str) -> None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # A recursive schema can run long. Distinguish "the mask let through
        # invalid JSON" from "generation was cut off mid-object", instead of
        # surfacing a bare JSONDecodeError.
        raise AssertionError(
            f"output is not valid JSON ({exc}) -- truncated at max_tokens? "
            f"raw: {text!r}") from exc
    assert isinstance(parsed, dict), text
    assert parsed.get("operator") in AST_OPERATORS, text
    desc = parsed.get("description")
    assert isinstance(desc, str), text
    assert re.fullmatch(
        AST_DESCRIPTION_PATTERN,
        desc), (f"invalid description pattern: {desc!r} in {text!r}")
    operands = parsed.get("operands")
    assert isinstance(operands, list) and len(operands) == 2, text
    # The schema pins the top-level operands to SubTree refs, so a bare
    # number here means the $ref branch of the grammar was not honoured.
    for op in operands:
        _assert_sub_tree(op, text)


def _is_complex_ast_json(text: str) -> bool:
    """Whether `text` satisfies the complex AST schema.

    Only the AssertionErrors raised by _assert_complex_ast_json are treated as
    "does not satisfy the schema"; anything else propagates, so a broken helper
    cannot masquerade as a schema violation.
    """
    try:
        _assert_complex_ast_json(text)
    except AssertionError:
        return False
    return True


def _assert_regex(text: str) -> None:
    assert re.fullmatch(REGEX_PATTERN,
                        text), (f"regex constraint violated: {text!r}")


@pytest.fixture(scope="module")
def llm():
    # `pytest.MonkeyPatch.context()`, not the `monkeypatch` fixture: that
    # fixture is function-scoped and requesting it here raises ScopeMismatch.
    # Writing straight to os.environ instead would leave
    # VLLM_ALLOW_INSECURE_SERIALIZATION set for every later test module in the
    # session, masking any test that checks vLLM rejects insecure
    # serialization.
    with pytest.MonkeyPatch.context() as mp:
        # The multi-chunk test sends a Python function to the TPU workers via
        # collective_rpc, which needs pickle fallback in vLLM's IPC encoder;
        # the engine (core process) must be spawned with it already set.
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
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

        # Inside the context, so the engine is still torn down with the same
        # environment it was spawned under.
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


def test_complex_ast_json_schema_greedy(llm: LLM):
    """A recursive schema is the longest constraint the runner has to carry.

    test_json_schema_greedy closes its object in a handful of tokens; this
    one holds a grammar stack open across the whole generation, so the
    bitmask changes shape on nearly every step instead of settling. That is
    the regime where a stale or misapplied mask actually shows up.
    """
    outputs = llm.generate([COMPLEX_AST_PROMPT] * 4, COMPLEX_AST_PARAMS)
    assert len(outputs) == 4
    for output in outputs:
        _assert_complex_ast_json(output.outputs[0].text)


def test_complex_ast_json_schema_unconstrained_negative(llm: LLM):
    """Negative control for test_complex_ast_json_schema_greedy.

    Same prompt, same token budget, no grammar. If the model satisfied the
    recursive AST schema on its own, the positive test would pass whether or
    not the bitmask was applied at all -- it would be measuring the prompt,
    not the runner. Asserting the unconstrained output violates the schema is
    what makes the positive test evidence.
    """
    params = SamplingParams(
        temperature=COMPLEX_AST_PARAMS.temperature,
        max_tokens=COMPLEX_AST_PARAMS.max_tokens,
    )
    outputs = llm.generate([COMPLEX_AST_PROMPT] * 4, params)
    assert len(outputs) == 4
    texts = [output.outputs[0].text for output in outputs]
    assert not any(_is_complex_ast_json(text) for text in texts), (
        "unconstrained generation already satisfies the AST schema, so "
        f"test_complex_ast_json_schema_greedy proves nothing: {texts!r}")


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


def test_more_requests_than_max_num_seqs(llm: LLM):
    """Regression for the bitmask row order under batch turnover.

    Every other test in this file submits at most `max_num_seqs` requests, so
    each request keeps the `InputBatch` slot it was admitted into for its
    whole life and the scheduler's request order happens to coincide with the
    runner's batch order. Bitmask rows are emitted in scheduler order and
    consumed by batch index, so that coincidence hides any mix-up between the
    two.

    Submitting more requests than the batch can hold removes it:
    `InputBatch.condense()` backfills a finished request's slot with the last
    request in the batch, after which bitmask row i no longer belongs to
    batch row i.

    The four constraints are interleaved so that requests finish at staggered
    steps -- which is what drives the backfill -- and so that a row landing on
    the wrong request carries not just a different grammar position but a
    different grammar, making the violation unmissable.
    """
    kinds = [
        (CHOICE_PROMPT, CHOICE_PARAMS, _assert_choice),
        (REGEX_PROMPT, REGEX_PARAMS, _assert_regex),
        (JSON_PROMPT, JSON_PARAMS, _assert_json),
        (COMPLEX_AST_PROMPT, COMPLEX_AST_PARAMS, _assert_complex_ast_json),
    ]
    # 4x the fixture's max_num_seqs, so the batch turns over several times.
    selected = [kinds[i % len(kinds)] for i in range(16)]

    outputs = llm.generate([prompt for prompt, _, _ in selected],
                           [params for _, params, _ in selected])

    assert len(outputs) == len(selected)
    for i, (output, (_, _, check)) in enumerate(zip(outputs, selected)):
        completion = output.outputs[0]
        # A mask from the wrong request lets through a token the grammar
        # cannot accept; vLLM then fails to advance the FSM and terminates
        # the request with finish_reason "error".
        assert completion.finish_reason != "error", (
            f"request {i} was terminated mid-generation: {completion.text!r}")
        check(completion.text)
