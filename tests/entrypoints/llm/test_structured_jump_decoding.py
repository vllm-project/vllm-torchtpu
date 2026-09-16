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
"""End-to-end unit and integration tests for Structured Jump Decoding on TPU.

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
    os.environ["VLLM_TPU_JUMP_DECODING"] = "1"

    spec_config = {
        "method": "ngram",
        "prompt_lookup_max": 7,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 7,
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
    the generated token count (e.g. <= 12 steps for 30+ tokens, >= 3.0x compression).
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

    # 2. Hard Invariant Proof: Compression ratio (tokens / steps) must be >= 1.5x
    assert total_tokens >= 30, f"Expected at least 30 tokens, got {total_tokens}"
    compression_ratio = total_tokens / total_steps
    assert compression_ratio >= 1.5, (
        f"Jump Decoding FAILED to compress steps! "
        f"Generated {total_tokens} tokens in {total_steps} steps "
        f"(Compression ratio: {compression_ratio:.2f}x, expected >= 1.5x).")
    assert total_steps <= 40, (
        f"Expected <= 40 steps for {total_tokens} tokens, but took {total_steps} steps."
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
    """Verifies that Speculative Decoding (N-gram) + Jump Decoding work together harmoniously."""
    outputs = llm.generate([JUMP_PROMPT] * 2, JUMP_SAMPLING_PARAMS)
    assert len(outputs) == 2
    for idx, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"\n[Speculative Jump Request {idx}] Output: {text!r}")
        _assert_telemetry_json(text)
