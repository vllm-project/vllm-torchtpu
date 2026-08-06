"""Unit tests for DeepSeek-V4-specific MoE routing (classic top-k sqrtsoftplus, hash-routed experts)."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.vllm.moe_routing import (_hash_moe_select,
                                                   select_experts)

GROUPED_TOPK_PATH = ("vllm.model_executor.layers.fused_moe.router."
                     "grouped_topk_router.grouped_topk")


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
        weights, ids = select_experts(hidden,
                                      logits,
                                      topk=2,
                                      renormalize=True,
                                      scoring_fn="softmax",
                                      layer=_grouped_layer())
    mock_grouped.assert_called_once()
    assert torch.equal(ids, expected[1].to(torch.int32))


def test_classic_topk_sqrtsoftplus_scoring():
    """DSV4 classic top-k selection with sqrtsoftplus scoring and e_score_correction_bias."""
    hidden = torch.randn(2, 4)
    logits = torch.zeros(2, 4)
    bias = torch.tensor([0.0, 0.5, 0.0, 1.0])
    layer = MagicMock()
    layer.use_grouped_topk = False
    layer.e_score_correction_bias = bias
    layer.routed_scaling_factor = 1.0
    layer.hash_indices_table = None

    weights, ids = select_experts(hidden,
                                  logits,
                                  topk=2,
                                  renormalize=False,
                                  scoring_fn="sqrtsoftplus",
                                  layer=layer)
    assert set(ids[0].tolist()) == {1, 3}
    raw = torch.sqrt(F.softplus(torch.zeros(())))
    torch.testing.assert_close(weights[0], torch.full((2, ), raw.item()))


def test_classic_topk_sqrtsoftplus_renormalize_and_scaling():
    """DSV4 classic top-k with renormalize=True and routed_scaling_factor != 1.0."""
    hidden = torch.randn(2, 4)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    layer = MagicMock()
    layer.use_grouped_topk = False
    layer.e_score_correction_bias = None
    layer.routed_scaling_factor = 2.5
    layer.hash_indices_table = None

    weights, ids = select_experts(hidden,
                                  logits,
                                  topk=2,
                                  renormalize=True,
                                  scoring_fn="sqrtsoftplus",
                                  layer=layer)
    assert weights.shape == (2, 2)
    assert ids.shape == (2, 2)
    # With renormalize=True and scaling=2.5, sum across topk weights per token must equal 2.5
    torch.testing.assert_close(weights.sum(dim=-1),
                               torch.full((2, ), 2.5, dtype=weights.dtype))


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

    weights, ids = select_experts(hidden,
                                  logits,
                                  topk=topk,
                                  renormalize=True,
                                  scoring_fn="sqrtsoftplus",
                                  layer=layer)
    assert weights.shape == (num_tokens, topk)
    assert ids.shape == (num_tokens, topk)
    assert ids.dtype == torch.int32
    assert weights.dtype == hidden.dtype
    # Check that weights per token are normalized to sum to 1.0
    torch.testing.assert_close(weights.sum(dim=-1),
                               torch.ones(num_tokens, dtype=weights.dtype))


def test_hash_moe_select_routes_by_token_id():
    table = torch.tensor([[0, 1], [2, 3], [1, 2]])  # token id -> expert ids
    scores = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ])
    input_ids = torch.tensor([2, 0])
    weights, ids = _hash_moe_select(scores,
                                    table,
                                    input_ids,
                                    renormalize=False,
                                    routed_scaling_factor=1.0)
    assert ids[0].tolist() == [1, 2]
    assert ids[1].tolist() == [0, 1]
    torch.testing.assert_close(weights[0], scores[0, [1, 2]])
    torch.testing.assert_close(weights[1], scores[1, [0, 1]])


def test_select_experts_hash_table_takes_priority():
    """A layer with hash_indices_table must never reach topk routing."""
    hidden = torch.randn(2, 4)
    logits = torch.randn(2, 4)
    layer = _grouped_layer(
        hash_indices_table=torch.tensor([[0, 1], [2, 3], [1, 2]]))
    with patch(GROUPED_TOPK_PATH) as mock_grouped:
        weights, ids = select_experts(hidden,
                                      logits,
                                      topk=2,
                                      renormalize=True,
                                      scoring_fn="sigmoid",
                                      layer=layer,
                                      input_ids=torch.tensor([1, 2]))
    mock_grouped.assert_not_called()
    assert ids[0].tolist() == [2, 3]
    assert ids[1].tolist() == [1, 2]
    torch.testing.assert_close(weights.sum(dim=-1),
                               torch.ones(2, dtype=weights.dtype))


def test_non_grouped_plain_path_matches_manual_topk():
    """Without a layer, behavior must match plain scored top-k (non-DSV4)."""
    hidden = torch.randn(3, 6)
    logits = torch.randn(3, 6)
    weights, ids = select_experts(hidden,
                                  logits,
                                  topk=2,
                                  renormalize=False,
                                  scoring_fn="softmax",
                                  layer=None)
    expected_w, expected_i = torch.topk(torch.softmax(logits.float(), dim=-1),
                                        k=2,
                                        dim=-1)
    assert torch.equal(ids, expected_i.to(torch.int32))
    torch.testing.assert_close(weights, expected_w.to(weights.dtype))
