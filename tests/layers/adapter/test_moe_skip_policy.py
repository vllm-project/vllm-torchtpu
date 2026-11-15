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
"""Tests for BF16 MoeSkip policy resolution."""

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
        "fused_decode_kernel": False,
        "has_speculative": False,
        "has_lora": False,
        "is_unquantized_model": True,
    }
    kwargs.update(overrides)
    return msp.resolve_bf16_moe_skip_policy(**kwargs)


def test_resolve_policy_supported_enables_count_mode():
    policy = _resolve()
    assert policy.enabled is True
    assert policy.padding_rows == "count"
    assert policy.fallback_reason is None
    assert policy.as_dict() == {
        "enabled": True,
        "padding_rows": "count",
        "fallback_reason": None,
    }


def test_resolve_policy_flag_off_is_stock_without_reason():
    policy = _resolve(skip_flag=False)
    assert policy.enabled is False
    assert policy.padding_rows is None
    assert policy.fallback_reason is None


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        ({"is_unquantized_model": False}, "non-unquantized"),
        (
            {
                "parallel_config": _supported_parallel_config(
                    enable_expert_parallel=True
                )
            },
            "expert",
        ),
        (
            {"parallel_config": _supported_parallel_config(data_parallel_size=2)},
            "data_parallel_size",
        ),
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
        ({"pipelined_collective": True}, "pipelined"),
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
    assert policy.padding_rows is None
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
