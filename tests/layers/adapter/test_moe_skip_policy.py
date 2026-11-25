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
"""Tests for BF16 MoE padded-token skipping policy resolution."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import moe_skip_policy as msp
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
)


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    for name in (
        "TORCH_TPU_DP_SIZE",
        "TPU_MOE_COLLECTION_CHUNK_SIZE",
        "USE_MOE_TP_DECODE_KERNEL",
        "FUSED_MOE_DECODE_KERNEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _supported_config(**parallel_overrides):
    parallel = {
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        quant_config=VllmUnquantizedConfig(),
        parallel_config=SimpleNamespace(**parallel),
        speculative_config=None,
        lora_config=None,
    )


def test_supported_config_enables_skipping():
    policy = msp.resolve_bf16_moe_skip_policy(_supported_config())
    assert policy == msp.Bf16MoeSkipPolicy(enabled=True)


@pytest.mark.parametrize("config", [_supported_config(), None])
def test_flag_off_is_stock_without_reason(monkeypatch, config):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0")
    policy = msp.resolve_bf16_moe_skip_policy(config)
    assert policy == msp.Bf16MoeSkipPolicy(enabled=False)


def test_no_current_config_runs_stock_with_reason():
    policy = msp.resolve_bf16_moe_skip_policy(None)
    assert policy.enabled is False
    assert policy.fallback_reason == "no current VllmConfig"


@pytest.mark.parametrize(
    "missing",
    [
        "enable_expert_parallel",
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_missing_parallel_field_fails_loudly(missing):
    """A missing ParallelConfig field raises instead of defaulting to a size
    that would admit skipping."""
    config = _supported_config()
    delattr(config.parallel_config, missing)
    with pytest.raises(AttributeError, match=missing):
        msp.resolve_bf16_moe_skip_policy(config)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda c, mp: setattr(c, "quant_config", object()), "quantized model"),
        (
            lambda c, mp: setattr(c.parallel_config, "enable_expert_parallel", True),
            "expert parallelism",
        ),
        (
            lambda c, mp: setattr(c.parallel_config, "data_parallel_size", 2),
            "data_parallel_size > 1",
        ),
        (
            lambda c, mp: setattr(c.parallel_config, "pipeline_parallel_size", 2),
            "pipeline_parallel_size > 1",
        ),
        (
            lambda c, mp: setattr(
                c.parallel_config, "prefill_context_parallel_size", 2
            ),
            "prefill_context_parallel_size > 1",
        ),
        (
            lambda c, mp: setattr(c.parallel_config, "decode_context_parallel_size", 2),
            "decode_context_parallel_size > 1",
        ),
        (
            lambda c, mp: mp.setenv("TPU_MOE_COLLECTION_CHUNK_SIZE", "16"),
            "pipelined MoE collective (TPU_MOE_COLLECTION_CHUNK_SIZE)",
        ),
        (
            lambda c, mp: setattr(c, "speculative_config", SimpleNamespace()),
            "speculative decoding",
        ),
        (lambda c, mp: setattr(c, "lora_config", SimpleNamespace()), "LoRA"),
    ],
)
def test_unsupported_config_falls_back_to_stock(monkeypatch, mutate, expected):
    config = _supported_config()
    mutate(config, monkeypatch)
    policy = msp.resolve_bf16_moe_skip_policy(config)
    assert policy == msp.Bf16MoeSkipPolicy(enabled=False, fallback_reason=expected)


def test_dp_without_ep_falls_back_to_stock(monkeypatch):
    """For DP without EP the worker sets data_parallel_size to 1, so the DP
    size comes from TORCH_TPU_DP_SIZE (utils.get_dp_size)."""
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "2")
    policy = msp.resolve_bf16_moe_skip_policy(_supported_config(data_parallel_size=1))
    assert policy == msp.Bf16MoeSkipPolicy(
        enabled=False, fallback_reason="data_parallel_size > 1"
    )


@pytest.mark.parametrize(
    "name", ["USE_MOE_TP_DECODE_KERNEL", "FUSED_MOE_DECODE_KERNEL"]
)
def test_fused_moe_decode_kernels_keep_skipping(monkeypatch, name):
    """Both fused MoE decode kernels admit only FP8 MoE layers, so they never
    serve a BF16 MoE layer and do not turn skipping off."""
    monkeypatch.setenv(name, "1")
    policy = msp.resolve_bf16_moe_skip_policy(_supported_config())
    assert policy == msp.Bf16MoeSkipPolicy(enabled=True)
