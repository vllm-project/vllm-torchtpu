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
"""Tests for BF16 MoE padded-token skipping in VllmUnquantizedFusedMoEMethod.

- Skipping on: skip_padded_tokens=True, PaddedRowFilter.ROW_COUNT and
  row_info=local_valid_tokens.
- Skipping off (flag off, an unsupported configuration, or a layer built
  outside the runner's policy scope): skip_padded_tokens=False, without
  padded_row_filter or row_info.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import fused_moe as fused_moe_module
from vllm_torchtpu.layers.adapter import moe_skip_policy, token_padding
from vllm_torchtpu.layers.adapter.quantization import unquantized as module
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import PaddedRowFilter

_TOPK = 2


def _moe_cfg(use_ep: bool = False):
    return SimpleNamespace(
        experts_per_token=_TOPK,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
    )


def _method(*, skip_padded_tokens: bool) -> VllmUnquantizedFusedMoEMethod:
    method = VllmUnquantizedFusedMoEMethod.__new__(VllmUnquantizedFusedMoEMethod)
    method._tpu_activation_str = "silu"
    method._skip_padded_tokens = skip_padded_tokens
    return method


def _dummy_layer(use_ep: bool = False):
    return SimpleNamespace(
        w13_weight=torch.zeros(4, 128, 256, dtype=torch.bfloat16),
        w2_weight=torch.zeros(4, 128, 128, dtype=torch.bfloat16),
        _experts_start=torch.zeros((), dtype=torch.int32),
        moe_config=_moe_cfg(use_ep=use_ep),
    )


@pytest.fixture
def capture_forward(monkeypatch):
    weights = torch.tensor(
        [[0.6, 0.4], [0.0, 1.0], [0.7, 0.3], [0.8, 0.2]], dtype=torch.float32
    )
    ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32)
    zero_calls: list[bool] = []
    fused_calls: list[dict] = []
    pipelined_calls: list[dict] = []

    monkeypatch.setattr(
        module.moe_routing, "route", lambda layer, x, logits: (weights, ids)
    )
    monkeypatch.setattr(
        token_padding,
        "zero_routing_weights_for_padding",
        lambda *args, **kwargs: (zero_calls.append(True), (ids, weights))[1],
    )
    monkeypatch.setattr(
        module, "enable_pipelined_collective_and_compute", lambda: False
    )

    def _rec_fused(**kwargs):
        fused_calls.append(kwargs)
        return kwargs["hidden_states"]

    def _rec_pipelined(**kwargs):
        pipelined_calls.append(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(module, "fused_moe_gmm", _rec_fused)
    monkeypatch.setattr(module, "pipelined_fused_moe_gmm", _rec_pipelined)
    return SimpleNamespace(
        weights=weights,
        ids=ids,
        zero_calls=zero_calls,
        fused_calls=fused_calls,
        pipelined_calls=pipelined_calls,
    )


def _run_forward(method: VllmUnquantizedFusedMoEMethod, layer=None):
    x = torch.ones(4, 128, dtype=torch.bfloat16)
    logits = torch.zeros(4, 4, dtype=torch.bfloat16)
    return method._forward_monolithic_tpu(layer or _dummy_layer(), x, logits)


def test_count_forward_passes_count_and_does_not_zero_weights(capture_forward):
    """With skipping on, the forward passes the row-count filter and
    local_valid_tokens without touching routing weights (so padded rows stay
    non-zero and real zero-gate rows keep stock dispatch)."""
    state = token_padding.TokenPaddingState.create(16, "cpu", track_valid_tokens=True)
    state.update(2, 4)
    token_padding.set_padding_state(state)
    try:
        method = _method(skip_padded_tokens=True)
        _run_forward(method)
    finally:
        token_padding.set_padding_state(None)

    assert capture_forward.zero_calls == []
    assert len(capture_forward.fused_calls) == 1
    call = capture_forward.fused_calls[0]
    assert call["skip_padded_tokens"] is True
    assert call["padded_row_filter"] is PaddedRowFilter.ROW_COUNT
    assert call["row_info"] is state.local_valid_tokens
    assert int(call["row_info"].item()) == 2
    # Padded rows keep their original non-zero weights (exclusion is by count).
    assert torch.equal(call["topk_weights"], capture_forward.weights)


def test_stock_contract_when_disabled_or_fallback(capture_forward, monkeypatch):
    """With skipping off, the forward passes skip_padded_tokens=False and omits
    padded_row_filter and row_info, also when pipelined_fused_moe_gmm runs."""
    state = token_padding.TokenPaddingState.create(16, "cpu", track_valid_tokens=True)
    state.update(1, 4)
    token_padding.set_padding_state(state)
    try:
        method = _method(skip_padded_tokens=False)
        _run_forward(method)

        monkeypatch.setattr(
            module, "enable_pipelined_collective_and_compute", lambda: True
        )
        _run_forward(method)
    finally:
        token_padding.set_padding_state(None)

    assert capture_forward.zero_calls == []
    assert len(capture_forward.fused_calls) == 1
    assert len(capture_forward.pipelined_calls) == 1
    for call in (capture_forward.fused_calls[0], capture_forward.pipelined_calls[0]):
        assert call["skip_padded_tokens"] is False
        assert "padded_row_filter" not in call
        assert "row_info" not in call


def test_count_mode_raises_if_local_valid_tokens_missing(capture_forward):
    state = token_padding.TokenPaddingState.create(16, "cpu", track_valid_tokens=False)
    token_padding.set_padding_state(state)
    try:
        method = _method(skip_padded_tokens=True)
        with pytest.raises(RuntimeError, match="track_valid_tokens=False"):
            _run_forward(method)
    finally:
        token_padding.set_padding_state(None)


def test_init_consumes_runner_policy_scope(monkeypatch):
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod, "__init__", lambda self, moe: None
    )
    enabled_policy = moe_skip_policy.Bf16MoeSkipPolicy(enabled=True)
    with moe_skip_policy.bf16_policy_scope(enabled_policy):
        m = VllmUnquantizedFusedMoEMethod(_moe_cfg())
    assert m._skip_padded_tokens is True

    disabled_policy = moe_skip_policy.Bf16MoeSkipPolicy(
        enabled=False, fallback_reason="data_parallel_size > 1"
    )
    with moe_skip_policy.bf16_policy_scope(disabled_policy):
        m_off = VllmUnquantizedFusedMoEMethod(_moe_cfg())
    assert m_off._skip_padded_tokens is False


def test_init_outside_runner_scope_runs_stock(monkeypatch):
    """Only the runner's policy turns skipping on: with the flag set, a layer
    built outside bf16_policy_scope still runs the stock path."""
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod, "__init__", lambda self, moe: None
    )
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    assert moe_skip_policy.get_active_bf16_policy() is None
    m = VllmUnquantizedFusedMoEMethod(_moe_cfg())
    assert m._skip_padded_tokens is False


@pytest.mark.parametrize(
    "skip, expected_keys",
    [
        (False, [(_TOPK, "silu", False, None, False, PaddedRowFilter.ZERO_WEIGHT)]),
        (True, [(_TOPK, "silu", False, None, True, PaddedRowFilter.ROW_COUNT)]),
    ],
)
def test_process_weights_after_loading_prebuilds_expected_kernels(
    monkeypatch, skip, expected_keys
):
    """process_weights_after_loading prebuilds only the kernels needed by the
    resolved policy via the real prebuild_fused_moe_kernel cache key."""

    class _FakeCache(dict):
        def __init__(self):
            super().__init__()
            self.lookups: list[tuple] = []

        def get(self, key, default=None):
            self.lookups.append(key)
            return lambda *args, **kwargs: None

    cache = _FakeCache()
    monkeypatch.setattr(fused_moe_module, "_fused_moe_kernel_cache", cache)

    class _Experts(SimpleNamespace):
        pass

    monkeypatch.setattr(module, "RoutedExperts", _Experts)
    monkeypatch.setattr(
        module.moe_routing, "register_experts_start_buffer", lambda *a, **k: None
    )

    method = _method(skip_padded_tokens=skip)
    method.moe = SimpleNamespace(has_bias=False)
    method._set_tpu_activation = lambda layer: "silu"

    layer = _Experts(
        w13_weight=torch.nn.Parameter(
            torch.zeros(2, 256, 128, dtype=torch.bfloat16), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(2, 128, 128, dtype=torch.bfloat16), requires_grad=False
        ),
        moe_config=_moe_cfg(),
    )
    method.process_weights_after_loading(layer)
    assert cache.lookups == expected_keys


def test_disable_moe_skip_disarms_and_prebuilds_stock_kernel(monkeypatch):
    prebuild_calls: list[dict] = []
    monkeypatch.setattr(
        module,
        "prebuild_fused_moe_kernel",
        lambda **kwargs: prebuild_calls.append(kwargs),
    )
    method = _method(skip_padded_tokens=True)
    method.disable_moe_skip(_dummy_layer())
    assert method._skip_padded_tokens is False
    assert [call["skip_padded_tokens"] for call in prebuild_calls] == [False]
    assert "padded_row_filter" not in prebuild_calls[0]
