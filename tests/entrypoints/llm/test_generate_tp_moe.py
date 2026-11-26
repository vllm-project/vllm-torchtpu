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
End-to-end tests for vLLM LLM generation API with MoE models on TPU.

These tests are in a separate file because MoE models require multiple
TPU devices and are resource-intensive. Keeping them separate prevents
module-scoped fixture conflicts with single-device tests.

Run with: pytest tests/entrypoints/llm/test_generate_tp_moe.py -v
"""

import numpy as np
import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.multichip

MOE_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

MOE_PROMPTS = [
    "The capital of France is",
    "Write one sentence about compilers.",
    "List two uses of sparse experts.",
    "Explain TPU execution briefly.",
]


def _get_local_tpu_chip_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


def _print_generation_outputs(test_name: str, outputs) -> None:
    print(f"\n{test_name} outputs:")
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")


def _run_moe_generation(
    async_scheduling: bool,
    enable_expert_parallel: bool = False,
    enable_return_routed_experts: bool = False,
):
    tp_size = _get_local_tpu_chip_count()
    if tp_size < 8:
        pytest.skip(
            f"MoE TP test expects an 8-chip agent, found {tp_size} "
            "chips; running here would silently reduce TP coverage"
        )

    # Four ~6-token prompts and 4 output tokens: 8 sequences is the only
    # request bucket a step uses; the default compiles them up to 64.
    llm = LLM(
        model=MOE_MODEL_NAME,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        max_model_len=64,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_expert_parallel,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        disable_log_stats=False,
        seed=0,
        # vLLM requires prefix caching to return routed experts.
        **(
            {"enable_return_routed_experts": True, "enable_prefix_caching": True}
            if enable_return_routed_experts
            else {}
        ),
    )
    model_config = llm.llm_engine.model_config

    try:
        outputs = llm.generate(
            MOE_PROMPTS,
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=4, ignore_eos=True
            ),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    return tp_size, outputs, model_config


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "async_scheduling",
    [
        pytest.param(False, marks=pytest.mark.nightly, id="sync"),
        pytest.param(True, marks=pytest.mark.nightly, id="async"),
    ],
)
def test_moe_generate_with_tp_equal_local_tpu_count(async_scheduling: bool):
    tp_size, outputs, _ = _run_moe_generation(async_scheduling=async_scheduling)
    _print_generation_outputs(
        f"MoE TP ({'async' if async_scheduling else 'sync'})", outputs
    )

    assert tp_size >= 2
    assert len(outputs) == len(MOE_PROMPTS)
    assert all(len(output.outputs) == 1 for output in outputs)
    assert all(output.outputs[0].text.strip() for output in outputs)
    # Greedy decoding of "The capital of France is" must produce "Paris";
    # catches MoE channel-misalignment bugs that yield non-empty token salad.
    assert "paris" in outputs[0].outputs[0].text.lower(), (
        f"incoherent MoE output: {outputs[0].outputs[0].text!r}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "async_scheduling",
    [
        pytest.param(False, marks=pytest.mark.nightly, id="sync"),
        pytest.param(True, marks=pytest.mark.nightly, id="async"),
    ],
)
def test_moe_generate_with_ep_equal_local_tpu_count(async_scheduling: bool):
    tp_size, outputs, _ = _run_moe_generation(
        async_scheduling=async_scheduling,
        enable_expert_parallel=True,
    )
    _print_generation_outputs(
        f"MoE EP ({'async' if async_scheduling else 'sync'})", outputs
    )

    assert tp_size >= 2
    assert len(outputs) == len(MOE_PROMPTS)
    assert all(len(output.outputs) == 1 for output in outputs)
    assert all(output.outputs[0].text.strip() for output in outputs)
    assert "paris" in outputs[0].outputs[0].text.lower(), (
        f"incoherent MoE output: {outputs[0].outputs[0].text!r}"
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "async_scheduling",
    [
        pytest.param(False, marks=pytest.mark.nightly, id="sync"),
        pytest.param(True, marks=pytest.mark.nightly, id="async"),
    ],
)
def test_moe_generate_returns_routed_experts(async_scheduling: bool):
    """Every request gets one row of expert ids for each token it ran: its
    prompt and each output token but the last. A row holds k distinct experts
    in every MoE layer, so a layer whose ids the forward did not capture,
    which come back as zeros, fails the check."""
    _, outputs, model_config = _run_moe_generation(
        async_scheduling=async_scheduling,
        enable_expert_parallel=True,
        enable_return_routed_experts=True,
    )
    num_layers = model_config.get_total_num_hidden_layers()
    k = model_config.get_num_experts_per_tok()

    assert len(outputs) == len(MOE_PROMPTS)
    for output in outputs:
        completion = output.outputs[0]
        rows = completion.routed_experts
        num_tokens = len(output.prompt_token_ids) + len(completion.token_ids) - 1
        assert rows is not None, output.prompt
        assert rows.shape == (num_tokens, num_layers, k), output.prompt
        assert int(rows.max()) < model_config.get_num_experts(), output.prompt
        ids = np.sort(rows, axis=-1)
        assert np.all(ids[..., 1:] != ids[..., :-1]), output.prompt
