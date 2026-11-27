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

- The layer resolves the policy from the VllmConfig it is built under; a
  draft model built by Eagle3Proposer takes the speculative-decoding fallback.
- Skipping on: skip_padded_tokens=True and row_info=the token padding mask.
- Skipping off (flag off, an unsupported configuration, or no padding state):
  skip_padded_tokens=False and row_info=None (the stock kernel).
- On the TPU, the compiled forward reads the mask when it runs, so one program
  serves every padding the runner writes.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch._dynamo.utils

from vllm_torchtpu.layers.adapter import fused_moe as fused_moe_module
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.quantization import unquantized as module
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod,
)

_TOPK = 2


def _moe_cfg(use_ep: bool = False):
    return SimpleNamespace(
        experts_per_token=_TOPK,
        has_bias=False,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
    )


def _vllm_config(**overrides):
    """A VllmConfig stand-in with every field the policy reads."""
    config = SimpleNamespace(
        quant_config=VllmUnquantizedConfig(),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
        lora_config=None,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.fixture
def build_method(monkeypatch):
    """Construct the method (the upstream __init__ skipped) under a config."""
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod, "__init__", lambda self, moe: None
    )
    for name in ("TORCH_TPU_DP_SIZE", "TPU_MOE_COLLECTION_CHUNK_SIZE"):
        monkeypatch.delenv(name, raising=False)

    def build(config, *, flag: bool) -> VllmUnquantizedFusedMoEMethod:
        monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1" if flag else "0")
        monkeypatch.setattr(module, "get_current_vllm_config_or_none", lambda: config)
        return VllmUnquantizedFusedMoEMethod(_moe_cfg())

    return build


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


def test_skip_forward_passes_padding_mask_and_keeps_weights(capture_forward):
    """With skipping on, the forward passes the token padding mask as row_info
    without touching routing weights (so padded rows stay non-zero and real
    zero-gate rows keep stock dispatch)."""
    state = token_padding.TokenPaddingState.create(16, "cpu")
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
    # The whole buffer the runner writes each step, not a copy or a slice.
    assert call["row_info"] is state.local_padding_mask
    assert call["row_info"][:4].tolist() == [False, False, True, True]
    # Padded rows keep their original non-zero weights (exclusion is by row).
    assert torch.equal(call["topk_weights"], capture_forward.weights)


def test_stock_contract_when_disabled_or_fallback(capture_forward, monkeypatch):
    """With skipping off, the forward passes skip_padded_tokens=False and
    row_info=None, also when pipelined_fused_moe_gmm runs."""
    state = token_padding.TokenPaddingState.create(16, "cpu")
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


def test_skip_without_padding_state_runs_stock(capture_forward):
    """The runner sets the padding state whenever the flag is on; a forward
    run without one has no rows to skip and keeps the stock contract."""
    token_padding.set_padding_state(None)
    _run_forward(_method(skip_padded_tokens=True))
    call = capture_forward.fused_calls[0]
    assert call["skip_padded_tokens"] is False
    assert call["row_info"] is None


@pytest.mark.parametrize(
    "config, flag, expected",
    [
        (_vllm_config(), True, True),
        (_vllm_config(), False, False),
        (_vllm_config(speculative_config=SimpleNamespace()), True, False),
        (None, True, False),
    ],
)
def test_init_resolves_policy_from_current_vllm_config(
    build_method, config, flag, expected
):
    assert build_method(config, flag=flag)._skip_padded_tokens is expected


def test_init_logs_the_fallback_reason(build_method, monkeypatch):
    """A BF16 MoE layer logs why the flag is not applied; nothing is logged
    with the flag off or with skipping on."""
    logged: list[tuple] = []
    monkeypatch.setattr(
        module.logger, "warning_once", lambda msg, *args: logged.append(args)
    )
    build_method(_vllm_config(), flag=False)
    build_method(None, flag=False)
    build_method(_vllm_config(), flag=True)
    assert logged == []

    ep = SimpleNamespace(**vars(_vllm_config().parallel_config))
    ep.enable_expert_parallel = True
    build_method(_vllm_config(parallel_config=ep), flag=True)
    build_method(None, flag=True)
    assert logged == [("expert parallelism",), ("no current VllmConfig",)]


def test_draft_moe_layers_take_the_speculative_fallback(monkeypatch):
    """Flag on, a BF16 target and an MTP drafter: Eagle3Proposer builds the
    draft under a copy of the target's config, which keeps speculative_config,
    so the draft's BF16 MoE layers run the stock path like the target's."""
    import vllm_torchtpu.spec_decode.eagle3 as e3

    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    for name in ("TORCH_TPU_DP_SIZE", "TPU_MOE_COLLECTION_CHUNK_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod, "__init__", lambda self, moe: None
    )
    logged: list[tuple] = []
    monkeypatch.setattr(
        module.logger, "warning_once", lambda msg, *args: logged.append(args)
    )
    target_config = _vllm_config(
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_tensor_parallel_size=1,
            draft_model_config=SimpleNamespace(runner_type=None),
        ),
        load_config=object(),
        compilation_config=SimpleNamespace(inductor_compile_config={}),
    )
    target_config.parallel_config.tensor_parallel_size = 1
    proposer = e3.Eagle3Proposer(
        runner=mock.MagicMock(
            _pcp_enabled=False, _mtp_enabled=True, _mtp_prefill_only=False
        ),
        vllm_config=target_config,
    )
    proposer.runner = SimpleNamespace(mesh=object())
    draft_methods: list[VllmUnquantizedFusedMoEMethod] = []

    class _Loader:
        def load_model(self, *, vllm_config, model_config):
            # A draft MoE layer's quant method, built as vLLM builds it: under
            # the draft's config, set by the real set_current_vllm_config.
            draft_methods.append(VllmUnquantizedFusedMoEMethod(_moe_cfg()))
            return object()

    monkeypatch.setattr(e3, "get_model_loader", lambda _cfg: _Loader())
    monkeypatch.setattr(e3, "set_model_tag", lambda _tag: contextlib.nullcontext())
    monkeypatch.setattr(
        e3, "set_vllm_model_wrapper_context", lambda **_kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(e3, "_force_draft_tp1", lambda: contextlib.nullcontext())

    proposer._load_draft_model()

    assert proposer.draft_vllm_config is not target_config
    assert [m._skip_padded_tokens for m in draft_methods] == [False]
    assert logged == [("speculative decoding",)]


@pytest.mark.parametrize(
    "skip, expected_keys",
    [
        # (topk, activation, use_ep, rhs_quant_dtype, skip_padded_tokens,
        #  topk_weights_float32, has_row_info)
        (False, [(_TOPK, "silu", False, None, False, False, False)]),
        (True, [(_TOPK, "silu", False, None, True, False, True)]),
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

    Selection only, no arithmetic, so the stock and the skipping programs
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
    """row_info enters the compiled forward as the token padding mask read
    from TokenPaddingState, not as an argument. One skipping program, compiled
    for 16 rows, serves a warmup step (0 valid rows) and then 3, 16 and 5
    valid rows, which TokenPaddingState.update writes into the same device
    mask before each step, as the runner does. Each step's real rows match the
    stock program bit for bit and its padded rows are zero, and no step after
    the warmup traces or compiles again, so no padding is baked in."""
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
    skip_method = _method(skip_padded_tokens=True)
    for method in (stock_method, skip_method):
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

    def skip_forward(x, logits):
        return skip_method._forward_monolithic_tpu(layer, x, logits)

    compile_opts = dict(backend="tpu", fullgraph=True, dynamic=False)
    stock_forward = torch.compile(stock_forward, **compile_opts)
    skip_forward = torch.compile(skip_forward, **compile_opts)

    # Longer than the step, like the runner's buffer (max_num_tokens rows).
    state = token_padding.TokenPaddingState.create(2 * num_tokens, dev)
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
            out = skip_forward(x, logits).cpu()
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
