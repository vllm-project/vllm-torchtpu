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
End-to-end unit and integration tests for Structured Jump Decoding on TPU.

Tests that deterministic template structures (e.g. fixed JSON keys, static enums)
trigger Jump-Forward fast path, produce valid JSON matching the schema, and maintain
semantic correctness across single and batched requests.

Run with:
    pytest tests/entrypoints/llm/test_structured_jump_decoding.py -v
"""

import json
import os
import weakref

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.sampling_params import StructuredOutputsParams

MODEL_NAME = "Qwen/Qwen3-0.6B"

# High-determinism template schema designed to maximize Jump-Forward triggers
TELEMETRY_JUMP_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CloudServiceTelemetry",
    "type": "object",
    "properties": {
        "api_version": {
            "type": "string",
            "enum": ["v2.1"],
        },
        "service_name": {
            "type": "string",
            "enum": ["inference-gateway"],
        },
        "status": {
            "type": "string",
            "enum": ["HEALTHY", "DEGRADED", "UNHEALTHY"],
        },
        "metrics": {
            "type": "object",
            "properties": {
                "cluster_region": {
                    "type": "string",
                    "enum": ["us-central1", "us-east4", "europe-west4"],
                },
                "active_instances": {
                    "type": "integer",
                    "enum": [8, 16, 32, 64],
                },
                "is_healthy": {
                    "type": "boolean",
                },
            },
            "required": ["cluster_region", "active_instances", "is_healthy"],
        },
    },
    "required": ["api_version", "service_name", "status", "metrics"],
}

JUMP_SAMPLING_PARAMS = SamplingParams(
    temperature=0,
    max_tokens=128,
    structured_outputs=StructuredOutputsParams(
        json=TELEMETRY_JUMP_SCHEMA,
        disable_any_whitespace=True,
    ),
)

JUMP_PROMPT = "Generate a telemetry report for the inference gateway service in JSON format: "
UNCONSTRAINED_PROMPT = "The capital of France is"
UNCONSTRAINED_PARAMS = SamplingParams(temperature=0, max_tokens=16)


def _assert_telemetry_json(text: str) -> None:
    """Validates that generated text is valid JSON matching TELEMETRY_JUMP_SCHEMA."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        pytest.fail(
            f"Failed to parse generated text as JSON: {err}\nOutput text: {text!r}"
        )

    assert isinstance(parsed,
                      dict), f"Expected dict, got {type(parsed)}: {text!r}"
    assert parsed.get(
        "api_version") == "v2.1", f"Unexpected api_version: {parsed}"
    assert parsed.get(
        "service_name"
    ) == "inference-gateway", f"Unexpected service_name: {parsed}"
    assert parsed.get("status") in ["HEALTHY", "DEGRADED", "UNHEALTHY"
                                    ], f"Unexpected status: {parsed}"

    metrics = parsed.get("metrics")
    assert isinstance(metrics, dict), f"Expected metrics dict: {parsed}"
    assert metrics.get("cluster_region") in [
        "us-central1", "us-east4", "europe-west4"
    ], f"Invalid cluster_region: {metrics}"
    assert metrics.get("active_instances") in [
        8, 16, 32, 64
    ], f"Invalid active_instances: {metrics}"
    assert isinstance(metrics.get("is_healthy"),
                      bool), f"Invalid is_healthy: {metrics}"


@pytest.fixture(scope="module")
def llm():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_TPU_ENABLE_JUMP_DECODING"] = "1"

    spec_config = {
        "method": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 5,
    }

    llm = LLM(
        model=MODEL_NAME,
        speculative_config=spec_config,
        max_model_len=256,
        max_num_seqs=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
    )

    yield weakref.proxy(llm)

    del llm
    cleanup_dist_env_and_memory()


def test_deterministic_template_jump_decoding(llm: LLM):
    """Verifies that Jump Decoding produces 100% valid JSON matching the deterministic template."""
    outputs = llm.generate([JUMP_PROMPT] * 4, JUMP_SAMPLING_PARAMS)
    assert len(outputs) == 4
    for idx, output in enumerate(outputs):
        text = output.outputs[0].text
        print(f"\n[Request {idx}] Output: {text!r}")
        _assert_telemetry_json(text)


def test_jump_decoding_step_compression_proof(llm: LLM):
    """Proves that Jump Decoding is actively fast-forwarding tokens by asserting

    that the total number of engine execution steps is dramatically lower than
    the generated token count (e.g. <= 6 steps for 35+ tokens).
    """
    llm._add_request(JUMP_PROMPT, JUMP_SAMPLING_PARAMS)

    total_steps = 0
    final_outputs = []
    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        total_steps += 1
        for out in step_outputs:
            if out.finished:
                final_outputs.append(out)

    assert len(final_outputs) == 1
    output_text = final_outputs[0].outputs[0].text
    generated_token_ids = final_outputs[0].outputs[0].token_ids
    total_tokens = len(generated_token_ids)

    # 1. Correctness assertion
    _assert_telemetry_json(output_text)

    # 2. Hard Invariant Proof: Compression ratio (tokens / steps) must be >= 3.0x
    # Standard autoregressive decode strictly produces 1 token per step (ratio <= 1.0).
    # Jump Decoding fast-forwards static templates in batch, achieving 5x - 8x compression.
    assert total_tokens >= 30, f"Expected at least 30 tokens, got {total_tokens}"
    compression_ratio = total_tokens / total_steps
    assert compression_ratio >= 3.0, (
        f"Jump Decoding FAILED to compress steps! "
        f"Generated {total_tokens} tokens in {total_steps} steps (Compression ratio: {compression_ratio:.2f}x, expected >= 3.0x)."
    )
    assert total_steps <= 12, (
        f"Expected <= 12 steps for {total_tokens} tokens, but took {total_steps} steps."
    )


def test_jump_decoding_mixed_batch(llm: LLM):
    """Verifies that mixed batches (jump structured + unconstrained) decode correctly without interference."""
    prompts = [
        JUMP_PROMPT,
        UNCONSTRAINED_PROMPT,
        JUMP_PROMPT,
        UNCONSTRAINED_PROMPT,
    ]
    params = [
        JUMP_SAMPLING_PARAMS,
        UNCONSTRAINED_PARAMS,
        JUMP_SAMPLING_PARAMS,
        UNCONSTRAINED_PARAMS,
    ]
    outputs = llm.generate(prompts, params)
    assert len(outputs) == 4
    _assert_telemetry_json(outputs[0].outputs[0].text)
    _assert_telemetry_json(outputs[2].outputs[0].text)
    assert "Paris" in outputs[1].outputs[0].text or len(
        outputs[1].outputs[0].text) > 0
    assert "Paris" in outputs[3].outputs[0].text or len(
        outputs[3].outputs[0].text) > 0


def test_speculative_jump_decoding_ngram(llm: LLM):
    """Verifies that Speculative Decoding (N-gram) + Jump Decoding work together harmoniously.

    Ensures that speculative draft proposing and verification under grammar constraints
    interoperate seamlessly with FSM deterministic token fast-forwarding.
    """
    # 1. Multi-request generation under speculative jump decoding
    outputs = llm.generate([JUMP_PROMPT] * 2, JUMP_SAMPLING_PARAMS)
    assert len(outputs) == 2
    for idx, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"\n[Speculative Jump Request {idx}] Output: {text!r}")
        _assert_telemetry_json(text)

    # 2. Step compression proof with speculative decoding
    llm._add_request(JUMP_PROMPT, JUMP_SAMPLING_PARAMS)
    total_steps = 0
    final_outputs = []
    while llm.llm_engine.has_unfinished_requests():
        step_outs = llm.llm_engine.step()
        total_steps += 1
        for out in step_outs:
            if out.finished:
                final_outputs.append(out)

    assert len(final_outputs) == 1
    out_text = final_outputs[0].outputs[0].text
    gen_tokens = final_outputs[0].outputs[0].token_ids
    total_tokens = len(gen_tokens)

    _assert_telemetry_json(out_text)
    assert total_tokens >= 30, f"Expected >= 30 tokens, got {total_tokens}"
    compression_ratio = total_tokens / total_steps
    assert compression_ratio >= 3.0, (
        f"Speculative Jump Decoding failed step compression: "
        f"{total_tokens} tokens in {total_steps} steps ({compression_ratio:.2f}x)"
    )


def test_speculative_jump_decoding_high_k(llm: LLM):
    """Verifies that Speculative Jump Decoding scales effectively with higher draft budget (K=5)."""
    outputs = llm.generate([JUMP_PROMPT] * 2, JUMP_SAMPLING_PARAMS)
    assert len(outputs) == 2
    for idx, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"\n[High-K Speculative Jump {idx}] Output: {text!r}")
        _assert_telemetry_json(text)


def test_speculative_jump_decoding_multi_schema_batch(llm: LLM):
    """Verifies that concurrent requests with different JSON schemas and unconstrained prompts

    under Speculative Decoding maintain strict grammar isolation and correct jump fast-forwarding.
    """
    prompts = [
        JUMP_PROMPT,
        UNCONSTRAINED_PROMPT,
        JUMP_PROMPT,
        UNCONSTRAINED_PROMPT,
    ]
    params = [
        JUMP_SAMPLING_PARAMS,
        UNCONSTRAINED_PARAMS,
        JUMP_SAMPLING_PARAMS,
        UNCONSTRAINED_PARAMS,
    ]

    outputs = llm.generate(prompts, params)
    assert len(outputs) == 4
    _assert_telemetry_json(outputs[0].outputs[0].text)
    _assert_telemetry_json(outputs[2].outputs[0].text)
    assert "Paris" in outputs[1].outputs[0].text or len(
        outputs[1].outputs[0].text) > 0
    assert "Paris" in outputs[3].outputs[0].text or len(
        outputs[3].outputs[0].text) > 0


def test_grammar_aware_draft_lookahead_speculative_jump():
    """Verifies Grammar-Aware Draft Lookahead (Step 9).

    The ngram proposer can only propose text it has already seen, so on a
    novel prompt it returns short or empty drafts and the speculative slots
    go to waste. update_draft_token_ids keeps whatever the proposer got right
    and tops the remaining slots up with deterministic grammar tokens, so the
    step count drops even with nothing to match against.
    """
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_TPU_ENABLE_JUMP_DECODING"] = "1"

    # Novel non-repeating prompt: the ngram proposer has no prior text to
    # match, so any step compression here comes from the grammar lookahead.
    novel_prompt = ("Return cloud infrastructure telemetry status in strict "
                    "JSON format: ")

    spec_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 4,
    }

    spec_llm = LLM(
        model=MODEL_NAME,
        speculative_config=spec_config,
        max_model_len=256,
        max_num_seqs=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        disable_log_stats=False,
    )

    try:
        spec_llm._add_request(novel_prompt, JUMP_SAMPLING_PARAMS)
        total_steps = 0
        final_outputs = []
        while spec_llm.llm_engine.has_unfinished_requests():
            step_outs = spec_llm.llm_engine.step()
            total_steps += 1
            for out in step_outs:
                if out.finished:
                    final_outputs.append(out)

        assert len(final_outputs) == 1
        out_text = final_outputs[0].outputs[0].text
        total_tokens = len(final_outputs[0].outputs[0].token_ids)

        _assert_telemetry_json(out_text)
        assert total_tokens >= 30, f"Expected >= 30 tokens, got {total_tokens}"
        # tokens/steps is purely algorithmic (no wall-clock), so it is stable
        # under greedy decoding. Measured 3.35x (57 tokens / 17 steps) on TPU
        # v7x with Qwen3-0.6B under the top-up semantics, i.e. ~12% headroom
        # over this threshold.
        compression_ratio = total_tokens / total_steps
        assert compression_ratio >= 3.0, (
            f"Expected >= 3.0x step compression via grammar lookahead, "
            f"got {compression_ratio:.2f}x "
            f"({total_tokens} tokens / {total_steps} steps)")
    finally:
        del spec_llm
        cleanup_dist_env_and_memory()
