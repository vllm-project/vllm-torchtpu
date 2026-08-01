"""
End-to-end tests for vLLM LLM generation API with tensor parallelism on TPU.

These tests are in a separate file from test_generate.py because TP tests
require all TPU devices, which conflicts with the single-device LLM fixture
in that file (module-scoped fixtures coexist within the same module).

Run with: pytest tests/entrypoints/llm/test_generate_tp.py -v
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "Qwen/Qwen3-0.6B"


def _get_local_tpu_chip_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


@pytest.mark.parametrize("async_scheduling", [
    pytest.param(False, marks=pytest.mark.nightly, id="sync"),
    pytest.param(True, id="async"),
])
def test_generate_with_tp_equal_local_tpu_count(async_scheduling: bool):
    tp_size = _get_local_tpu_chip_count()

    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=64,
        max_model_len=64,
        tensor_parallel_size=tp_size,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        disable_log_stats=False,
    )

    try:
        outputs = llm.generate(
            ["Tensor parallel test prompt."],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    assert tp_size >= 1
    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
