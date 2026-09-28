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
"""Unit tests for DeepSeek-V4-specific MoE routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.adapter.moe_routing import _hash_moe_select, select_experts

GROUPED_TOPK_PATH = (
    "vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk"
)


def _grouped_layer(**overrides):
    layer = MagicMock()
    layer.use_grouped_topk = True
    layer.num_expert_group = 2
    layer.topk_group = 1
    layer.routed_scaling_factor = 1.0
    layer.e_score_correction_bias = None
    layer.hash_indices_table = None
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def test_grouped_softmax_delegates_to_vllm_grouped_topk():
    """Legacy grouped routing (DSV2/DSV3) must use vLLM's own grouped_topk."""
    hidden = torch.randn(4, 8)
    logits = torch.randn(4, 8)
    expected = (torch.rand(4, 2), torch.randint(0, 8, (4, 2)))
    with patch(GROUPED_TOPK_PATH, return_value=expected) as mock_grouped:
        weights, ids = select_experts(
            hidden,
            logits,
            topk=2,
            renormalize=True,
            scoring_fn="softmax",
            layer=_grouped_layer(),
        )
    mock_grouped.assert_called_once()
    assert torch.equal(ids, expected[1].to(torch.int32))


def test_classic_topk_sqrtsoftplus_scoring():
    """DSV4 classic top-k selection with sqrtsoftplus scoring and
    e_score_correction_bias."""
    hidden = torch.randn(2, 4)
    logits = torch.zeros(2, 4)
    bias = torch.tensor([0.0, 0.5, 0.0, 1.0])
    layer = MagicMock()
    layer.use_grouped_topk = False
    layer.e_score_correction_bias = bias
    layer.routed_scaling_factor = 1.0
    layer.hash_indices_table = None

    weights, ids = select_experts(
        hidden,
        logits,
        topk=2,
        renormalize=False,
        scoring_fn="sqrtsoftplus",
        layer=layer,
    )
    assert set(ids[0].tolist()) == {1, 3}
    raw = torch.sqrt(F.softplus(torch.zeros(())))
    torch.testing.assert_close(weights[0], torch.full((2,), raw.item()))


def test_classic_topk_sqrtsoftplus_renormalize_and_scaling():
    """DSV4 classic top-k with renormalize=True and routed_scaling_factor != 1.0."""
    hidden = torch.randn(2, 4)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    layer = MagicMock()
    layer.use_grouped_topk = False
    layer.e_score_correction_bias = None
    layer.routed_scaling_factor = 2.5
    layer.hash_indices_table = None

    weights, ids = select_experts(
        hidden, logits, topk=2, renormalize=True, scoring_fn="sqrtsoftplus", layer=layer
    )
    assert weights.shape == (2, 2)
    assert ids.shape == (2, 2)
    # With renormalize=True and scaling=2.5, sum across topk weights per token must
    # equal 2.5
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.full((2,), 2.5, dtype=weights.dtype)
    )


def test_classic_topk_multi_token_batch():
    """DSV4 classic top-k over a multi-token batch (16 tokens, 64 experts, topk=8)."""
    num_tokens, num_experts, topk = 16, 64, 8
    hidden = torch.randn(num_tokens, 128)
    logits = torch.randn(num_tokens, num_experts)
    bias = torch.randn(num_experts)
    layer = MagicMock()
    layer.use_grouped_topk = False
    layer.e_score_correction_bias = bias
    layer.routed_scaling_factor = 1.0
    layer.hash_indices_table = None

    weights, ids = select_experts(
        hidden,
        logits,
        topk=topk,
        renormalize=True,
        scoring_fn="sqrtsoftplus",
        layer=layer,
    )
    assert weights.shape == (num_tokens, topk)
    assert ids.shape == (num_tokens, topk)
    assert ids.dtype == torch.int32
    assert weights.dtype == hidden.dtype
    # Check that weights per token are normalized to sum to 1.0
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(num_tokens, dtype=weights.dtype)
    )


def test_hash_moe_select_routes_by_token_id():
    table = torch.tensor([[0, 1], [2, 3], [1, 2]])  # token id -> expert ids
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    input_ids = torch.tensor([2, 0])
    weights, ids = _hash_moe_select(
        scores, table, input_ids, renormalize=False, routed_scaling_factor=1.0
    )
    assert ids[0].tolist() == [1, 2]
    assert ids[1].tolist() == [0, 1]
    torch.testing.assert_close(weights[0], scores[0, [1, 2]])
    torch.testing.assert_close(weights[1], scores[1, [0, 1]])


def test_select_experts_hash_table_takes_priority():
    """A layer with hash_indices_table must never reach topk routing."""
    hidden = torch.randn(2, 4)
    logits = torch.randn(2, 4)
    layer = _grouped_layer(hash_indices_table=torch.tensor([[0, 1], [2, 3], [1, 2]]))
    with patch(GROUPED_TOPK_PATH) as mock_grouped:
        weights, ids = select_experts(
            hidden,
            logits,
            topk=2,
            renormalize=True,
            scoring_fn="sigmoid",
            layer=layer,
            input_ids=torch.tensor([1, 2]),
        )
    mock_grouped.assert_not_called()
    assert ids[0].tolist() == [2, 3]
    assert ids[1].tolist() == [1, 2]
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, dtype=weights.dtype))


def test_non_grouped_plain_path_matches_manual_topk():
    """Without a layer, behavior must match plain scored top-k (non-DSV4)."""
    hidden = torch.randn(3, 6)
    logits = torch.randn(3, 6)
    weights, ids = select_experts(
        hidden, logits, topk=2, renormalize=False, scoring_fn="softmax", layer=None
    )
    expected_w, expected_i = torch.topk(
        torch.softmax(logits.float(), dim=-1), k=2, dim=-1
    )
    assert torch.equal(ids, expected_i.to(torch.int32))
    torch.testing.assert_close(weights, expected_w.to(weights.dtype))


@pytest.mark.cpu_test
def test_dsv4_forward_lets_vllm_runner_apply_gate_once():
    """The runner owns the gate; DSv4 must forward tokens without a second GEMM."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE

    class Gate(torch.nn.Linear):
        tid2eid = torch.tensor([[0], [1]])
        calls = 0

        def forward(self, x):
            self.calls += 1
            return super().forward(x), None

    class CPUExperts:
        # Replace only the accelerator expert kernel; the runner's gate,
        # dispatch and output handling are the real upstream implementation.
        quant_method = SimpleNamespace(
            is_monolithic=True,
            skip_forward_padding=True,
            has_unpadded_output=False,
            moe_kernel=None,
        )

        def _ensure_moe_quant_config_init(self):
            pass

        def forward_monolithic(self, *, x, router_logits, input_ids=None):
            return x * router_logits[:, :1] + input_ids[:, None]

    model = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(model)
    model.gate = Gate(2, 2, bias=False)
    model.gate.weight.data.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    moe_config = SimpleNamespace(
        hidden_dim=2,
        tp_size=1,
        dp_size=1,
        ep_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
    )
    with set_current_vllm_config(config):
        model.experts = MoERunner(
            layer_name="model.layers.0.ffn.experts",
            moe_config=moe_config,
            router=None,
            routed_experts=CPUExperts(),
            gate=model.gate,
        )
    with set_forward_context(None, config):
        output = model(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]), input_ids=torch.tensor([1, 0])
        )
    torch.testing.assert_close(output, torch.tensor([[3.0, 5.0], [18.0, 24.0]]))
    assert model.gate.calls == 1
