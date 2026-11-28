# SPDX-License-Identifier: Apache-2.0
"""TpuPlatform.validate_request: unsupported sampling options are 400s."""

import pytest
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        (dict(temperature=0.9, seed=1), "seed"),
        (dict(temperature=0.0, logprobs=-1), "logprobs"),
    ],
)
def test_unsupported_options_are_client_errors(kwargs, parameter):
    with pytest.raises(VLLMValidationError) as e:
        TpuPlatform.validate_request(None, SamplingParams(**kwargs))
    # A VLLMClientError is what AsyncLLM passes through as a 400.
    assert isinstance(e.value, VLLMClientError)
    assert e.value.parameter == parameter


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(temperature=0.0),
        dict(temperature=0.9),
        dict(temperature=0.0, seed=1),  # greedy ignores the seed
        dict(temperature=0.0, logprobs=5),
        dict(temperature=0.9, logprobs=0),
        # Prompt logprobs are not computed on TPU, but vLLM sets them
        # implicitly on chat requests with echo, which work today.
        dict(temperature=0.0, prompt_logprobs=0),
        dict(temperature=0.0, prompt_logprobs=5),
    ],
)
def test_supported_requests_pass(kwargs):
    TpuPlatform.validate_request(None, SamplingParams(**kwargs))


def test_pooling_requests_pass():
    TpuPlatform.validate_request(None, PoolingParams())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("top_logprobs", [None, 5])
def test_chat_echo_passes(stream, top_logprobs):
    """Chat echo sets prompt_logprobs without the client asking for it."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        echo=True,
        stream=stream,
        logprobs=top_logprobs is not None,
        top_logprobs=top_logprobs or 0,
    )
    params = request.to_sampling_params(16, {})
    assert params.prompt_logprobs == (top_logprobs or 0)
    TpuPlatform.validate_request(None, params)
