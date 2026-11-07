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
"""The exporter's draft model steps are copies of the TPU runner's DFlash /
DSpark proposer's; each must agree with the runner's on the same draft model."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import draft_model
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import DSparkProposer

_H, _NQ, _NKV, _HD, _L, _V, _R = 16, 4, 2, 4, 3, 40, 3


class _RecordingImpl:
    """An attention impl that records what each call would write."""

    def __init__(self, writes: list[tuple[Any, ...]]) -> None:
        self.writes = writes

    def forward(self, *, layer, query, key, value, kv_cache, attn_metadata):
        self.writes.append((layer.name, query, key, value, kv_cache, attn_metadata))


class _KNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(_HD) + 0.5)
        self.variance_epsilon = 1e-6


def _rotary(positions, q, k):
    angle = positions.to(torch.float32).view(-1, 1, 1) * 0.1
    return q * torch.cos(angle) + 1.0, k * torch.cos(angle) - k * torch.sin(angle)


class _SelfAttn(nn.Module):
    def __init__(self, i: int, writes: list[tuple[Any, ...]]) -> None:
        super().__init__()
        self.head_dim, self.num_kv_heads, self.num_heads = _HD, _NKV, _NQ
        self.q_size = _NQ * _HD
        self.qkv_proj = nn.Linear(_H, (_NQ + 2 * _NKV) * _HD, bias=True)
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.qkv_proj.quant_method = UnquantizedLinearMethod()
        self.k_norm = _KNorm()
        self.rotary_emb = _rotary
        self.attn = SimpleNamespace(
            name=f"layer{i}", impl=_RecordingImpl(writes), kv_cache=f"kv{i}"
        )


class _Backbone(nn.Module):
    def __init__(self, writes: list[tuple[Any, ...]]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module() for _ in range(_L)])
        for i, layer in enumerate(self.layers):
            layer.self_attn = _SelfAttn(i, writes)
        self.fc = nn.Linear(2 * _H, _H, bias=False)
        self.hidden_norm = nn.LayerNorm(_H)
        self.embed = nn.Embedding(_V, _H)
        self.mix = nn.Linear(_H, _H)

    def forward(self, input_ids, positions):
        x = self.embed(input_ids.to(torch.int64))
        x = x + positions.to(x.dtype).unsqueeze(-1) * 0.01
        # Rows of a block see each other (non-causal), as the draft does.
        return self.mix(x) + x.view(-1, 1, _H).mean(0)


class _Draft(nn.Module):
    """A DSpark-shaped draft: what the proposer and the copies call."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[tuple[Any, ...]] = []
        self.model = _Backbone(self.writes)
        self.head = nn.Linear(_H, _V, bias=False)
        self.markov_w1 = nn.Embedding(_V, _R)
        self.markov_w2 = nn.Linear(_R, _V, bias=False)
        draft_model.bind_fused_kv_weight(self)

    def combine_hidden_states(self, hidden):
        return self.model.fc(hidden)

    def compute_draft_logits(self, hidden):
        return self.head(hidden)

    def markov_embed(self, token_ids):
        return self.markov_w1(token_ids.to(torch.int64))

    def markov_bias(self, embed):
        return self.markov_w2(embed) * 4.0

    def map_draft_to_target(self, ids):
        return ids


@pytest.fixture
def draft() -> _Draft:
    torch.manual_seed(0)
    return _Draft()


def test_fused_kv_weight_matches_vllms_stacking(draft: _Draft) -> None:
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    vllm_side = SimpleNamespace(hidden_norm=draft.model.hidden_norm)
    attns = [layer.self_attn for layer in draft.model.layers]
    DFlashQwen3Model._build_context_kv_buffers(vllm_side, attns, has_bias=True)
    torch.testing.assert_close(draft.model._fused_kv_weight, vllm_side._fused_kv_weight)
    torch.testing.assert_close(draft.model._fused_kv_bias, vllm_side._fused_kv_bias)


def test_precompute_context_kv_matches_the_tpu_runner(draft: _Draft) -> None:
    aux = [torch.randn(5, _H), torch.randn(5, _H)]
    positions = torch.tensor([3, 4, 5, 9, 10])
    metadata = tuple(f"md{i}" for i in range(_L))

    runner = SimpleNamespace(draft_model=draft)
    inspect.unwrap(DFlashProposer._tpu_precompute_and_update_kv_cache)(
        runner, aux, positions, metadata
    )
    expected, draft.writes[:] = list(draft.writes), []
    draft_model.precompute_context_kv(draft, aux, positions, metadata)

    assert len(draft.writes) == len(expected) == _L
    for got, want in zip(draft.writes, expected, strict=True):
        assert got[0] == want[0] and got[4:] == want[4:]
        for a, b in zip(got[1:4], want[1:4], strict=True):
            torch.testing.assert_close(a, b)


def test_layout_draft_block_matches_the_tpu_runner() -> None:
    base_tokens = torch.tensor([7, 8, 31], dtype=torch.int32)
    base_pos = torch.tensor([[12], [62], [0]], dtype=torch.int32)
    runner = SimpleNamespace(_max_model_len=64)
    expected = DFlashProposer._layout_draft_block(
        runner, base_tokens, base_pos, 4, 12, 31
    )
    actual = draft_model.layout_draft_block(base_tokens, base_pos, 4, 12, 31, 64)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("from_anchor", [True, False])
def test_dspark_forward_and_sample_matches_the_tpu_runner(
    draft: _Draft, from_anchor: bool
) -> None:
    block = draft_model.query_block_size(3, from_anchor)
    ids, positions, _ = draft_model.layout_draft_block(
        torch.tensor([5, 9], dtype=torch.int32),
        torch.tensor([[3], [7]], dtype=torch.int32),
        block,
        2 * block,
        mask_token_id=39,
        max_model_len=64,
    )
    proposer = SimpleNamespace(
        draft_model=draft, sample_from_anchor=from_anchor, use_confidence_head=False
    )
    runner_fn = inspect.unwrap(DSparkProposer._dflash_forward_and_sample)
    expected, _ = runner_fn(proposer, ids, positions, block)
    actual = draft_model.dspark_forward_and_sample(
        draft, ids, positions, block, from_anchor
    )
    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected)


def _batched_rpa_impl() -> Any:
    """A `--attention-backend CUSTOM` attention impl, with its class's kernel."""
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
    impl.decode_query_size = 8
    impl.initialize_kernel = lambda *args, **kwargs: None
    return impl


def _runner_bound_draft_impl(draft_tp: int) -> Any:
    """A draft attention impl after `TPUModelRunner._initialize_attention_kernels`."""
    from unittest.mock import MagicMock, patch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    impl = _batched_rpa_impl()
    runner = MagicMock(spec=TPUModelRunner)
    runner._attention_kernels_initialized = False
    runner.model = None
    runner.mesh = MagicMock()
    runner.vllm_config = MagicMock()
    runner.drafter = SimpleNamespace(_draft_attn_layer_names={"draft.0"})
    runner.speculative_config = SimpleNamespace(draft_tensor_parallel_size=draft_tp)
    with (
        patch(
            "vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config",
            return_value={"draft.0": SimpleNamespace(impl=impl)},
        ),
        patch("vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context"),
    ):
        TPUModelRunner._initialize_attention_kernels(runner)
    return impl


@pytest.mark.parametrize("draft_tp", [1, 2])
def test_draft_attention_kernel_matches_the_tpu_runner(draft_tp: int) -> None:
    """A tp=1 draft runs the local ragged paged attention kernel, as in the
    runner, even under the batched-kernel backend, which rejects the
    non-causal attention of DSpark and DFlash drafts."""
    expected = _runner_bound_draft_impl(draft_tp)
    actual = _batched_rpa_impl()
    draft_model.bind_draft_attention_kernel(actual, draft_tp)

    for attr in ("decode_query_size", "_kernel_entry", "_kernel_op_prefix"):
        assert getattr(actual, attr) == getattr(expected, attr), attr
    if draft_tp == 1:
        assert "use_causal_mask" in inspect.signature(actual._kernel_entry).parameters
        assert actual._kernel_entry.__name__ == "_pallas_rpa_kernel_local"
