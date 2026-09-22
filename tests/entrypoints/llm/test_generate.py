"""
End-to-end tests for vLLM LLM generation API on TPU.

This module tests the LLM.generate() API with various configurations using
the Qwen3-0.6B model with a single TPU device.

Tests include:
- Multiple sampling parameters
- Priority queue handling
- Max model length enforcement
- Logprobs serialization
- Statistics logging

For TP tests, see test_generate_tp.py.
For MoE tests, see test_generate_tp_moe.py.

Run with: pytest tests/entrypoints/llm/test_generate.py -v
"""

import weakref

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.exceptions import VLLMValidationError

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]

GREEDY_SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=4)


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=64,
        max_model_len=64,
        tensor_parallel_size=1,
        # TODO: Reduce GPU memory utilization after KV cache optimization.
        # For reference, this value is set to 0.10 in the equivalent test in vllm.
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        disable_log_stats=False,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


def test_multiple_sampling_params(llm: LLM):
    """Test multiple sampling parameters."""
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=1),
        SamplingParams(temperature=0.7, max_tokens=2),
        SamplingParams(temperature=0.0, max_tokens=3),
        SamplingParams(temperature=0.7, max_tokens=4),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(
        VLLMValidationError,
        match=r"lengths of prompts .* and params .* must be the same",
    ):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(PROMPTS, sampling_params=GREEDY_SAMPLING_PARAMS)
    assert len(PROMPTS) == len(outputs)


def test_non_greedy_sampling(llm: LLM):
    sampling_params = SamplingParams(temperature=0.7, max_tokens=4)
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)


def test_multiple_priority(llm: LLM):
    # Generate works when priority is None
    outputs = llm.generate(
        PROMPTS, sampling_params=GREEDY_SAMPLING_PARAMS, priority=None
    )
    assert len(PROMPTS) == len(outputs)

    # Generate works when length of priority is same as the len(PROMPTS)
    outputs = llm.generate(
        PROMPTS, sampling_params=GREEDY_SAMPLING_PARAMS, priority=[0] * len(PROMPTS)
    )
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the length of priority does not match the length of prompts
    with pytest.raises(
        VLLMValidationError,
        match=r"lengths of prompts .* and priority .* must be the same",
    ):
        outputs = llm.generate(
            PROMPTS,
            sampling_params=GREEDY_SAMPLING_PARAMS,
            priority=[0] * (len(PROMPTS) - 1),
        )

    # Exception raised, if the priority list is empty
    with pytest.raises(
        VLLMValidationError,
        match=r"lengths of prompts .* and priority .* must be the same",
    ):
        outputs = llm.generate(
            PROMPTS, sampling_params=GREEDY_SAMPLING_PARAMS, priority=[]
        )


def test_max_model_len(llm: LLM):
    max_model_len = llm.model_config.max_model_len
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_model_len + 10)
    outputs = llm.generate(PROMPTS, sampling_params)
    for output in outputs:
        num_total_tokens = len(output.prompt_token_ids) + len(
            output.outputs[0].token_ids
        )
        # Total tokens must not exceed max_model_len.
        # It can be less if generation finishes due to other reasons (e.g., EOS)
        # before reaching the absolute model length limit.
        assert num_total_tokens <= max_model_len


def test_logprobs(llm: LLM):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)
    for output in outputs:
        assert output.outputs[0].logprobs is not None
        assert len(output.outputs[0].logprobs) > 0


def test_log_stats(llm: LLM):
    outputs = llm.generate(PROMPTS, sampling_params=GREEDY_SAMPLING_PARAMS)
    assert all(output.metrics is not None for output in outputs)
