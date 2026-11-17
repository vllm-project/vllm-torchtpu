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

- Skipping on: skip_padded_tokens=True and row_info=local_valid_tokens (the
  row-count filter).
- Skipping off (flag off, an unsupported configuration, or a layer built
  outside the runner's policy scope): skip_padded_tokens=False and
  row_info=None (the stock kernel).
- On the TPU, the compiled forward reads local_valid_tokens when it runs, so
  one program serves every count the runner writes.
"""

from types import SimpleNamespace

import pytest
import torch
import torch._dynamo.utils

from vllm_torchtpu.layers.adapter import fused_moe as fused_moe_module
from vllm_torchtpu.layers.adapter import moe_skip_policy, token_padding
from vllm_torchtpu.layers.adapter.quantization import unquantized as module
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
)

_TOPK = 2


def _moe_cfg(use_ep: bool = False):
    return SimpleNamespace(
        experts_per_token=_TOPK,
        has_bias=False,
        is_sequence_parallel=False,
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
        module.moe_routing, "route", lambda layer, x, logits, **kwargs: (weights, ids)
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
    assert call["row_info"] is state.local_valid_tokens
    assert int(call["row_info"].item()) == 2
    # Padded rows keep their original non-zero weights (exclusion is by count).
    assert torch.equal(call["topk_weights"], capture_forward.weights)


def test_stock_contract_when_disabled_or_fallback(capture_forward, monkeypatch):
    """With skipping off, the forward passes skip_padded_tokens=False and
    row_info=None, also when pipelined_fused_moe_gmm runs."""
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
        assert call["row_info"] is None


def test_count_mode_asserts_if_local_valid_tokens_missing(capture_forward):
    state = token_padding.TokenPaddingState.create(16, "cpu", track_valid_tokens=False)
    token_padding.set_padding_state(state)
    try:
        method = _method(skip_padded_tokens=True)
        with pytest.raises(AssertionError, match="tracks no valid tokens"):
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


def test_init_logs_the_fallback_reason(monkeypatch):
    """A BF16 MoE layer logs why the flag is not applied; nothing is logged
    with the flag off, with skipping on, or outside the runner's scope."""
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod, "__init__", lambda self, moe: None
    )
    logged: list[tuple] = []
    monkeypatch.setattr(
        module.logger, "warning_once", lambda msg, *args: logged.append(args)
    )
    for policy in (
        moe_skip_policy.Bf16MoeSkipPolicy(enabled=False),
        moe_skip_policy.Bf16MoeSkipPolicy(enabled=True),
    ):
        with moe_skip_policy.bf16_policy_scope(policy):
            VllmUnquantizedFusedMoEMethod(_moe_cfg())
    VllmUnquantizedFusedMoEMethod(_moe_cfg())
    assert logged == []

    fallback = moe_skip_policy.Bf16MoeSkipPolicy(
        enabled=False, fallback_reason="expert parallelism"
    )
    with moe_skip_policy.bf16_policy_scope(fallback):
        VllmUnquantizedFusedMoEMethod(_moe_cfg())
    assert logged == [("expert parallelism",)]


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
        # (topk, activation, use_ep, rhs_quant_dtype, skip_padded_tokens,
        #  topk_weights_float32, has_row_info, return_rows)
        (False, [(_TOPK, "silu", False, None, False, False, False, False)]),
        (True, [(_TOPK, "silu", False, None, True, False, True, False)]),
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


def _top_logits(hidden_states, gating_output, topk, renormalize):
    """Route each row to its topk largest logits, gated by their values.

    Selection only, no arithmetic, so the stock and the row-count programs
    route identically.
    """
    weights, ids = torch.topk(gating_output, topk, dim=-1)
    return weights.to(hidden_states.dtype), ids.to(torch.int32)


def _dynamo_graphs() -> int:
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def _xla_cache_entries() -> int:
    return len(torch.tpu._get_cache_stats().per_entry_stats)


@pytest.mark.tpu_test
def test_one_compiled_forward_reads_each_steps_row_count():
    """row_info enters the compiled forward through
    token_padding.get_local_valid_tokens(), not as an argument. One row-count
    program, compiled for 16 rows, serves a warmup step (count 0) and then
    counts 3, 16 and 5, which TokenPaddingState.update writes into the same
    device tensor before each step, as the runner does. Each step's real rows
    match the stock program bit for bit and its padded rows are zero, and no
    step after the warmup traces or compiles again, so no count is baked in."""
    num_tokens, experts, hidden, inter, topk = 16, 16, 2048, 768, 8
    dev = torch.device("tpu")
    gen = torch.Generator().manual_seed(0)
    w13 = (torch.randn(experts, hidden, 2 * inter, generator=gen) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(experts, inter, hidden, generator=gen) * 0.02).to(torch.bfloat16)
    layer = SimpleNamespace(
        w13_weight=w13.to(dev),
        w2_weight=w2.to(dev),
        _experts_start=None,
        moe_config=SimpleNamespace(
            experts_per_token=topk, moe_parallel_config=SimpleNamespace(use_ep=False)
        ),
        custom_routing_function=_top_logits,
        renormalize=False,
        # Set by VllmMoERunner; moe_routing.route reads it.
        capture_fn=None,
    )
    stock_method = _method(skip_padded_tokens=False)
    count_method = _method(skip_padded_tokens=True)
    for method in (stock_method, count_method):
        # _method skips __init__, but Dynamo reads the nn.Module state (the
        # method is a vLLM CustomOp).
        torch.nn.Module.__init__(method)
        # The kernel process_weights_after_loading prebuilds for this method.
        fused_moe_module.prebuild_fused_moe_kernel(
            topk=topk,
            activation="silu",
            use_ep=False,
            skip_padded_tokens=method._skip_padded_tokens,
            has_row_info=method._skip_padded_tokens,
        )

    def stock_forward(x, logits):
        return stock_method._forward_monolithic_tpu(layer, x, logits)

    def count_forward(x, logits):
        return count_method._forward_monolithic_tpu(layer, x, logits)

    compile_opts = dict(backend="tpu", fullgraph=True, dynamic=False)
    stock_forward = torch.compile(stock_forward, **compile_opts)
    count_forward = torch.compile(count_forward, **compile_opts)

    state = token_padding.TokenPaddingState.create(
        num_tokens, dev, track_valid_tokens=True
    )
    token_padding.set_padding_state(state)
    compiled = []
    try:
        for num_valid in (0, 3, 16, 5):
            # Padded rows get large hidden states and, like every row,
            # non-zero gates.
            x = torch.randn(num_tokens, hidden, generator=gen)
            x[num_valid:] *= 50
            logits = torch.rand(num_tokens, experts, generator=gen) * 0.25 + 0.05
            x, logits = x.to(torch.bfloat16).to(dev), logits.to(dev)
            state.update(num_valid, num_tokens)
            ref = stock_forward(x, logits).cpu()
            out = count_forward(x, logits).cpu()
            # The warmup step traces and compiles both programs; the later
            # steps reuse them.
            compiled.append((_dynamo_graphs(), _xla_cache_entries()))
            assert compiled[-1] == compiled[0], (num_valid, compiled)
            assert torch.equal(out[:num_valid], ref[:num_valid]), num_valid
            assert torch.equal(out[num_valid:], torch.zeros_like(out[num_valid:]))
            if num_valid < num_tokens:
                assert ref[num_valid:].float().abs().max() > 0
    finally:
        token_padding.set_padding_state(None)
