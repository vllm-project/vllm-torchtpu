"""
End-to-end tests for vLLM LLM generation API with MoE models on TPU.

These tests are in a separate file because MoE models require multiple
TPU devices and are resource-intensive. Keeping them separate prevents
module-scoped fixture conflicts with single-device tests.

Run with: pytest tests/entrypoints/llm/test_generate_tp_moe.py -v
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

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


def _run_moe_generation(async_scheduling: bool,
                        enable_expert_parallel: bool = False):
    tp_size = _get_local_tpu_chip_count()
    if tp_size < 2:
        pytest.skip("MoE distributed test requires at least 2 local TPU chips")

    llm = LLM(
        model=MOE_MODEL_NAME,
        max_num_batched_tokens=64,
        max_model_len=64,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_expert_parallel,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        disable_log_stats=False,
        seed=0,
    )

    try:
        outputs = llm.generate(
            MOE_PROMPTS,
            sampling_params=SamplingParams(temperature=0.0,
                                           max_tokens=4,
                                           ignore_eos=True),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    return tp_size, outputs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("async_scheduling", [
    pytest.param(False, marks=pytest.mark.nightly, id="sync"),
    pytest.param(True, id="async"),
])
def test_moe_generate_with_tp_equal_local_tpu_count(async_scheduling: bool):
    tp_size, outputs = _run_moe_generation(async_scheduling=async_scheduling)
    _print_generation_outputs(
        f"MoE TP ({'async' if async_scheduling else 'sync'})", outputs)

    assert tp_size >= 2
    assert len(outputs) == len(MOE_PROMPTS)
    assert all(len(output.outputs) == 1 for output in outputs)
    assert all(output.outputs[0].text.strip() for output in outputs)
    # Greedy decoding of "The capital of France is" must produce "Paris";
    # catches MoE channel-misalignment bugs that yield non-empty token salad.
    assert "paris" in outputs[0].outputs[0].text.lower(), \
        f"incoherent MoE output: {outputs[0].outputs[0].text!r}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("async_scheduling", [
    pytest.param(False, marks=pytest.mark.nightly, id="sync"),
    pytest.param(True, id="async"),
])
def test_moe_generate_with_ep_equal_local_tpu_count(async_scheduling: bool):
    tp_size, outputs = _run_moe_generation(
        async_scheduling=async_scheduling,
        enable_expert_parallel=True,
    )
    _print_generation_outputs(
        f"MoE EP ({'async' if async_scheduling else 'sync'})", outputs)

    assert tp_size >= 2
    assert len(outputs) == len(MOE_PROMPTS)
    assert all(len(output.outputs) == 1 for output in outputs)
    assert all(output.outputs[0].text.strip() for output in outputs)
    assert "paris" in outputs[0].outputs[0].text.lower(), \
        f"incoherent MoE output: {outputs[0].outputs[0].text!r}"
