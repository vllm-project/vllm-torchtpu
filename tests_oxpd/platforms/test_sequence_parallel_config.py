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
"""Tests for sequence-parallel MoE platform compilation configuration."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def configure_sp_moe():
    """Stub optional dependencies and yield _configure_sequence_parallel_moe."""
    stubs = {
        mod: MagicMock()
        for mod in (
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.compile",
            "torch_tpu._internal.utils",
            "portpicker",
        )
        if mod not in sys.modules
    }
    with patch.dict(sys.modules, stubs):
        from vllm_torchtpu.platforms.tpu_platform import (
            _configure_sequence_parallel_moe,
        )

        yield _configure_sequence_parallel_moe


@pytest.mark.parametrize(
    "tp,original,expected",
    [
        (8, [8, 16, 24, 32, 48, 64, 128, 256], [16, 32, 64, 128, 256]),
        (4, [4, 8, 16, 24, 32, 64], [8, 16, 32, 64]),
        (2, [2, 4, 8, 16, 32], [4, 8, 16, 32]),
    ],
)
def test_sequence_parallel_compile_sizes(configure_sp_moe, tp, original, expected):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    compilation_config = SimpleNamespace(compile_sizes=list(original))
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    configure_sp_moe(vllm_config)
    assert compilation_config.compile_sizes == expected


def test_sequence_parallel_rejects_empty_compile_sizes(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    compilation_config = SimpleNamespace(compile_sizes=[8, 24, 48])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    with pytest.raises(ValueError, match="left no valid compile sizes"):
        configure_sp_moe(vllm_config)


def test_sequence_parallel_rejects_largest_size_dropped(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    # 48 is dropped (48 // 8 = 6, not power-of-two 2, 4 or multiple of 8)
    compilation_config = SimpleNamespace(compile_sizes=[16, 32, 48])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )
    with pytest.raises(ValueError, match="dropped largest compile size 48"):
        configure_sp_moe(vllm_config)


@pytest.mark.parametrize("arch", ["DeepseekV3ForCausalLM", "Qwen3NextForCausalLM"])
def test_sequence_parallel_rejects_unvalidated_arch(configure_sp_moe, arch):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=[arch])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=None,
    )
    with pytest.raises(
        ValueError,
        match="validated only for Qwen3.5 MoE",
    ):
        configure_sp_moe(vllm_config)


def test_sequence_parallel_admits_mtp(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=SimpleNamespace(method="mtp"),
    )
    configure_sp_moe(vllm_config)
    assert vllm_config.compilation_config.compile_sizes == [16, 32, 64]


@pytest.mark.parametrize("method", ["eagle", "ngram"])
def test_sequence_parallel_rejects_non_mtp_speculative(configure_sp_moe, method):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=SimpleNamespace(method=method),
    )
    with pytest.raises(ValueError, match="covers only the MTP drafter"):
        configure_sp_moe(vllm_config)


def test_sequence_parallel_rejects_pipeline_parallel(configure_sp_moe):
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
    )
    model_config = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(compile_sizes=[16, 32, 64]),
        speculative_config=None,
    )
    with pytest.raises(
        ValueError,
        match="does not support pipeline parallelism",
    ):
        configure_sp_moe(vllm_config)
