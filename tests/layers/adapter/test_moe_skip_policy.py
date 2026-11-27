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


def _supported_parallel_config(**overrides):
    cfg = {
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _resolve(**overrides):
    kwargs = {
        "skip_flag": True,
        "parallel_config": _supported_parallel_config(),
        "pipelined_collective": False,
        "sequence_parallel": False,
        "fused_decode_kernel": False,
        "has_speculative": False,
        "has_lora": False,
        "is_unquantized_model": True,
    }
    kwargs.update(overrides)
    return msp.resolve_bf16_moe_skip_policy(**kwargs)


@pytest.mark.parametrize(
    "parallel_overrides, extra_overrides",
    [
        ({}, {}),
        ({"enable_expert_parallel": True}, {}),
        ({"enable_expert_parallel": True}, {"sequence_parallel": True}),
        ({"enable_expert_parallel": True, "data_parallel_size": 4}, {}),
        (
            {"enable_expert_parallel": True, "data_parallel_size": 4},
            {"pipelined_collective": True},
        ),
    ],
    ids=["tp_only", "tp_ep", "tp_ep_seq_parallel", "dp_ep", "dp_ep_pipelined"],
)
def test_resolve_policy_supported_enables_skipping(parallel_overrides, extra_overrides):
    policy = _resolve(
        parallel_config=_supported_parallel_config(**parallel_overrides),
        **extra_overrides,
    )
    assert policy.enabled is True
    assert policy.fallback_reason is None


def test_resolve_policy_flag_off_is_stock_without_reason():
    policy = _resolve(skip_flag=False)
    assert policy.enabled is False
    assert policy.fallback_reason is None


@pytest.mark.parametrize(
    "missing",
    [
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_resolve_policy_fails_loudly_on_missing_parallel_field(missing):
    """A missing ParallelConfig field raises instead of defaulting to a size
    that would admit skipping."""
    cfg = _supported_parallel_config()
    delattr(cfg, missing)
    with pytest.raises(AttributeError, match=missing):
        _resolve(parallel_config=cfg)


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        ({"is_unquantized_model": False}, "quantized model"),
        (
            {"parallel_config": _supported_parallel_config(pipeline_parallel_size=2)},
            "pipeline_parallel_size",
        ),
        (
            {
                "parallel_config": _supported_parallel_config(
                    prefill_context_parallel_size=2
                )
            },
            "prefill_context_parallel_size",
        ),
        (
            {
                "parallel_config": _supported_parallel_config(
                    decode_context_parallel_size=2
                )
            },
            "decode_context_parallel_size",
        ),
        (
            {"sequence_parallel": True, "pipelined_collective": True},
            "sequence-parallel",
        ),
        ({"fused_decode_kernel": True}, "fused MoE decode"),
        ({"has_speculative": True}, "speculative"),
        ({"has_lora": True}, "LoRA"),
    ],
)
def test_resolve_policy_unsupported_falls_back_to_stock(overrides, expected_fragment):
    """Unsupported topologies and features fall back to the stock contract
    before model load / prebuild."""
    policy = _resolve(**overrides)
    assert policy.enabled is False
    assert policy.fallback_reason is not None
    assert expected_fragment in policy.fallback_reason


def test_bf16_policy_scope_nests_and_restores():
    assert msp.get_active_bf16_policy() is None
    outer = msp.Bf16MoeSkipPolicy(enabled=True)
    inner = msp.Bf16MoeSkipPolicy(enabled=False, fallback_reason="test")
    with msp.bf16_policy_scope(outer):
        assert msp.get_active_bf16_policy() is outer
        with msp.bf16_policy_scope(inner):
            assert msp.get_active_bf16_policy() is inner
        assert msp.get_active_bf16_policy() is outer
    assert msp.get_active_bf16_policy() is None


def test_bf16_policy_scope_restores_on_error():
    policy = msp.Bf16MoeSkipPolicy(enabled=True)
    with pytest.raises(RuntimeError, match="boom"), msp.bf16_policy_scope(policy):
        raise RuntimeError("boom")
    assert msp.get_active_bf16_policy() is None
