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


@pytest.fixture(scope="module")
def validate_dense_sp(configure_sp_moe):
    """Yields `_validate_dense_sequence_parallel` with platform stubs active."""
    del configure_sp_moe
    from vllm_torchtpu.platforms.tpu_platform import _validate_dense_sequence_parallel

    return _validate_dense_sequence_parallel


def _gemma4_vllm_config(
    *, enabled: bool = True, **overrides: object
) -> SimpleNamespace:
    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    model = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4_text"),
    )
    cfg = SimpleNamespace(
        additional_config={"tpu_dense_sequence_parallel": enabled} if enabled else {},
        parallel_config=parallel,
        model_config=model,
        lora_config=None,
        speculative_config=None,
    )
    for key, val in overrides.items():
        if hasattr(parallel, key):
            setattr(parallel, key, val)
        elif key == "model_type":
            model.hf_text_config.model_type = val
        else:
            setattr(cfg, key, val)
    return cfg


def test_dense_sp_platform_validation_admits_valid_and_ngram(validate_dense_sp) -> None:
    from vllm.config import LoRAConfig

    # Flag off: skips validation even at TP=1.
    validate_dense_sp(_gemma4_vllm_config(enabled=False, tensor_parallel_size=1))

    # Flag on with model_config=None (model-less VllmConfig()): returns cleanly.
    validate_dense_sp(_gemma4_vllm_config(model_config=None))

    validate_dense_sp(_gemma4_vllm_config())
    validate_dense_sp(
        _gemma4_vllm_config(speculative_config=SimpleNamespace(method="ngram"))
    )

    with pytest.raises(ValueError, match="tpu_dense_sequence_parallel"):
        validate_dense_sp(
            _gemma4_vllm_config(
                additional_config={"tpu_dense_sequence_parallel": "true"}
            )
        )

    with pytest.raises(NotImplementedError, match="LoRA"):
        validate_dense_sp(_gemma4_vllm_config(lora_config=LoRAConfig(max_loras=1)))


def test_dense_sp_platform_warns_once_on_non_gemma4(
    monkeypatch: pytest.MonkeyPatch, validate_dense_sp
) -> None:
    from vllm_torchtpu.platforms import tpu_platform

    warn = MagicMock()
    monkeypatch.setattr(tpu_platform.logger, "warning_once", warn)
    validate_dense_sp(
        _gemma4_vllm_config(model_type="qwen3_5_text", tensor_parallel_size=1)
    )
    warn.assert_called_once()


@pytest.mark.parametrize(
    ("overrides", "exc_type", "match"),
    [
        ({"tensor_parallel_size": 1}, ValueError, "tensor_parallel_size > 1"),
        (
            {"pipeline_parallel_size": 2},
            NotImplementedError,
            "pipeline_parallel_size=2",
        ),
        (
            {"prefill_context_parallel_size": 2},
            NotImplementedError,
            "prefill_context_parallel_size=2",
        ),
        (
            {"decode_context_parallel_size": 2},
            NotImplementedError,
            "decode_context_parallel_size=2",
        ),
        (
            {"speculative_config": SimpleNamespace(method="mtp")},
            NotImplementedError,
            "'mtp'",
        ),
        (
            {"speculative_config": SimpleNamespace(method="eagle3")},
            NotImplementedError,
            "'eagle3'",
        ),
    ],
)
def test_dense_sp_platform_rejects_unsupported_configs(
    validate_dense_sp,
    overrides: dict[str, object],
    exc_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc_type, match=match):
        validate_dense_sp(_gemma4_vllm_config(**overrides))


def test_hier_rs_platform_validation_and_env_forwarding(
    monkeypatch: pytest.MonkeyPatch, validate_dense_sp
) -> None:
    from vllm_torchtpu import envs
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    assert "TPU_DENSE_SP_HIER_RS" in TpuPlatform.additional_env_vars

    monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "fp8")
    with pytest.raises(ValueError, match="tpu_dense_sequence_parallel"):
        validate_dense_sp(_gemma4_vllm_config(enabled=False))
