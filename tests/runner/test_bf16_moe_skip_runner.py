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
"""Runner-side tests for BF16 MoE padded-token skipping."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

from vllm_torchtpu.layers.adapter import moe_skip_policy as msp
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.quantization import unquantized
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod,
)
from vllm_torchtpu.runner import tpu_runner as runner_module
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer


class _Mask:
    def __init__(self):
        self.updates: list[tuple[int, int]] = []

    def update(self, valid: int, padded: int) -> None:
        self.updates.append((valid, padded))


class _OtherMoEMethod(FusedMoEMethodBase):
    def create_weights(self, *args, **kwargs):
        raise AssertionError("not called")

    def get_fused_moe_quant_config(self, layer):
        raise AssertionError("not called")


def _runner():
    runner = object.__new__(TPUModelRunner)
    runner._token_padding_state = _Mask()
    runner._bf16_moe_skip_policy = msp.Bf16MoeSkipPolicy(enabled=False)
    runner._last_sequence_layout_plan = None
    return runner


def _step(runner, valid: int, padded: int) -> None:
    runner._last_sequence_layout_plan = SimpleNamespace(
        kind=runner_module.SequenceLayoutKind.ALL, local_num_tokens=valid
    )
    runner._token_padding_update(padded)


def _supported_runner(monkeypatch):
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    monkeypatch.delenv("TPU_MOE_COLLECTION_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("USE_MOE_TP_DECODE_KERNEL", raising=False)
    monkeypatch.delenv("FUSED_MOE_DECODE_KERNEL", raising=False)

    runner = _runner()
    runner.vllm_config = SimpleNamespace(quant_config=VllmUnquantizedConfig())
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
    )
    runner.speculative_config = None
    runner.drafter = None
    runner.lora_config = None
    return runner


def test_every_step_updates_padding_state_including_full_buckets():
    """Single graph: every step, full buckets included, writes the padding
    state (mask and, with skipping on, local_valid_tokens) before forward."""
    runner = _runner()
    steps = [(15, 16), (16, 16), (255, 256), (256, 256), (32, 32)]
    for valid, padded in steps:
        _step(runner, valid, padded)
    assert runner._token_padding_state.updates == steps


def test_resolve_bf16_policy_supported_enables_skipping(monkeypatch):
    runner = _supported_runner(monkeypatch)
    policy = runner._resolve_bf16_moe_skip_policy()
    assert policy.enabled is True
    assert policy.fallback_reason is None


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r, mp: mp.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "0"), None),
        (
            lambda r, mp: setattr(r.vllm_config, "quant_config", object()),
            "non-unquantized",
        ),
        (
            lambda r, mp: setattr(r.parallel_config, "enable_expert_parallel", True),
            "expert",
        ),
        (
            lambda r, mp: setattr(r.parallel_config, "data_parallel_size", 2),
            "data_parallel_size",
        ),
        (
            lambda r, mp: setattr(r.parallel_config, "pipeline_parallel_size", 2),
            "pipeline_parallel_size",
        ),
        (
            lambda r, mp: setattr(
                r.parallel_config, "prefill_context_parallel_size", 2
            ),
            "prefill_context_parallel_size",
        ),
        (
            lambda r, mp: setattr(r.parallel_config, "decode_context_parallel_size", 2),
            "decode_context_parallel_size",
        ),
        (lambda r, mp: mp.setenv("TPU_MOE_COLLECTION_CHUNK_SIZE", "16"), "pipelined"),
        (lambda r, mp: mp.setenv("USE_MOE_TP_DECODE_KERNEL", "1"), "fused MoE decode"),
        (lambda r, mp: mp.setenv("FUSED_MOE_DECODE_KERNEL", "1"), "fused MoE decode"),
        (lambda r, mp: setattr(r, "drafter", object()), "speculative"),
        (lambda r, mp: setattr(r, "lora_config", object()), "LoRA"),
    ],
)
def test_resolve_bf16_policy_falls_back_early(monkeypatch, mutate, expected):
    runner = _supported_runner(monkeypatch)
    mutate(runner, monkeypatch)
    policy = runner._resolve_bf16_moe_skip_policy()
    assert policy.enabled is False
    if expected is None:
        assert policy.fallback_reason is None
    else:
        assert policy.fallback_reason is not None and expected in policy.fallback_reason


def test_finalize_policy_disarms_dense_and_mixed_models(monkeypatch):
    runner = _supported_runner(monkeypatch)
    policy = runner._resolve_bf16_moe_skip_policy()
    assert policy.enabled is True

    # Dense model: no MoE layers -> disable policy before TokenPaddingState.
    runner._bf16_moe_skip_policy = policy
    runner.model = SimpleNamespace(modules=lambda: [SimpleNamespace()])
    runner._finalize_bf16_moe_skip_policy_for_model()
    assert runner._bf16_moe_skip_policy.enabled is False

    # Mixed MoE model: disarm VllmUnquantizedFusedMoEMethod instances before trace.
    runner._bf16_moe_skip_policy = policy
    disarmed: list[object] = []
    bf16_method = object.__new__(VllmUnquantizedFusedMoEMethod)
    bf16_method._skip_padded_tokens = True
    bf16_method._tpu_activation_str = None
    orig_disable = VllmUnquantizedFusedMoEMethod.disable_moe_skip

    def _spy_disable(self, layer=None):
        disarmed.append(layer)
        return orig_disable(self, layer)

    monkeypatch.setattr(VllmUnquantizedFusedMoEMethod, "disable_moe_skip", _spy_disable)
    bf16_mod = SimpleNamespace(quant_method=bf16_method)
    other_mod = SimpleNamespace(quant_method=object.__new__(_OtherMoEMethod))
    runner.model = SimpleNamespace(modules=lambda: [bf16_mod, other_mod])
    runner._finalize_bf16_moe_skip_policy_for_model()
    assert runner._bf16_moe_skip_policy.enabled is False
    assert disarmed == [bf16_mod]
    assert bf16_method._skip_padded_tokens is False


def test_finalize_policy_keeps_pure_bf16_moe_model(monkeypatch):
    runner = _supported_runner(monkeypatch)
    policy = runner._resolve_bf16_moe_skip_policy()
    runner._bf16_moe_skip_policy = policy
    bf16_method = object.__new__(VllmUnquantizedFusedMoEMethod)
    bf16_method._skip_padded_tokens = True
    runner.model = SimpleNamespace(
        modules=lambda: [SimpleNamespace(quant_method=bf16_method)]
    )
    runner._finalize_bf16_moe_skip_policy_for_model()
    assert runner._bf16_moe_skip_policy is policy


def test_draft_moe_layers_follow_the_runner_policy(monkeypatch):
    """Flag on with an MTP / EAGLE drafter, in load_model order.

    Speculative decoding turns the policy off, so TokenPaddingState tracks no
    valid-token count. The drafter loads after the target's policy scope has
    closed; its BF16 MoE layers must still take the runner's policy and run
    the stock path instead of reading the missing count.
    """
    runner = _supported_runner(monkeypatch)
    drafter = object.__new__(Eagle3Proposer)
    runner.drafter = drafter
    runner.speculative_config = SimpleNamespace()
    runner.model = object()
    monkeypatch.setattr(
        unquantized.UnquantizedFusedMoEMethod, "__init__", lambda self, moe: None
    )
    moe_cfg = SimpleNamespace(
        experts_per_token=2,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_ep=False),
    )

    runner._bf16_moe_skip_policy = runner._resolve_bf16_moe_skip_policy()
    policy = runner._bf16_moe_skip_policy
    assert policy.enabled is False
    assert policy.fallback_reason == "speculative decoding"

    with msp.bf16_policy_scope(policy):
        target_method = VllmUnquantizedFusedMoEMethod(moe_cfg)
    assert target_method._skip_padded_tokens is False
    runner._finalize_bf16_moe_skip_policy_for_model()
    assert runner._bf16_moe_skip_policy is policy

    draft_methods: list[VllmUnquantizedFusedMoEMethod] = []

    def _load_draft(target_model):
        assert target_model is runner.model
        assert msp.get_active_bf16_policy() is policy
        draft_methods.append(VllmUnquantizedFusedMoEMethod(moe_cfg))

    monkeypatch.setattr(drafter, "load_model", _load_draft, raising=False)
    runner._load_async_drafter()
    assert msp.get_active_bf16_policy() is None
    assert [m._skip_padded_tokens for m in draft_methods] == [False]

    state = token_padding.TokenPaddingState.create(
        16, "cpu", track_valid_tokens=runner._bf16_moe_skip_policy.enabled
    )
    assert state.local_valid_tokens is None

    calls: list[dict] = []
    weights = torch.full((4, 2), 0.5, dtype=torch.float32)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    monkeypatch.setattr(
        unquantized.moe_routing, "route", lambda layer, x, logits: (weights, ids)
    )
    monkeypatch.setattr(
        unquantized, "enable_pipelined_collective_and_compute", lambda: False
    )
    monkeypatch.setattr(
        unquantized,
        "fused_moe_gmm",
        lambda **kwargs: (calls.append(kwargs), kwargs["hidden_states"])[1],
    )
    draft_layer = SimpleNamespace(
        w13_weight=torch.zeros(4, 128, 256, dtype=torch.bfloat16),
        w2_weight=torch.zeros(4, 128, 128, dtype=torch.bfloat16),
        _experts_start=torch.zeros((), dtype=torch.int32),
        moe_config=moe_cfg,
    )
    draft_method = draft_methods[0]
    draft_method._tpu_activation_str = "silu"
    token_padding.set_padding_state(state)
    try:
        draft_method._forward_monolithic_tpu(
            draft_layer,
            torch.ones(4, 128, dtype=torch.bfloat16),
            torch.zeros(4, 4, dtype=torch.bfloat16),
        )
    finally:
        token_padding.set_padding_state(None)
    assert len(calls) == 1
    assert calls[0]["skip_padded_tokens"] is False
    assert "row_info" not in calls[0]
