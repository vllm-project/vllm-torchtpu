"""Unit tests for DeepSeek-V4-specific MoE routing (sqrtsoftplus grouped top-k, hash-routed experts)."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.vllm.moe_routing import (_hash_moe_select,
                                                   _sqrtsoftplus_grouped_topk,
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
    """Non-sqrtsoftplus grouped routing must use vLLM's own grouped_topk."""
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


def test_grouped_sqrtsoftplus_bypasses_vllm_grouped_topk():
    """sqrtsoftplus must take the DSV4-only path (vLLM's raises ValueError)."""
    hidden = torch.randn(4, 8)
    logits = torch.randn(4, 8)
    with patch(GROUPED_TOPK_PATH) as mock_grouped:
        weights, ids = select_experts(hidden,
                                      logits,
                                      topk=2,
                                      renormalize=True,
                                      scoring_fn="sqrtsoftplus",
                                      layer=_grouped_layer())
    mock_grouped.assert_not_called()
    assert weights.shape == (4, 2)
    assert ids.shape == (4, 2)
    assert ids.dtype == torch.int32


def test_sqrtsoftplus_grouped_topk_reference_math():
    """Check group masking + scoring against an independent computation."""
    # 2 tokens, 4 experts in 2 groups of 2; keep 1 group, then top-2 experts.
    logits = torch.tensor([
        [4.0, 3.0, 0.0, 0.1],  # group 0 dominates
        [0.0, 0.1, 5.0, 2.0],  # group 1 dominates
    ])
    weights, ids = _sqrtsoftplus_grouped_topk(logits,
                                              topk=2,
                                              renormalize=False,
                                              num_expert_group=2,
                                              topk_group=1)
    scores = torch.sqrt(F.softplus(logits))
    assert set(ids[0].tolist()) == {0, 1}
    assert set(ids[1].tolist()) == {2, 3}
    expected0 = scores[0, ids[0].long()]
    expected1 = scores[1, ids[1].long()]
    torch.testing.assert_close(weights[0], expected0)
    torch.testing.assert_close(weights[1], expected1)


def test_sqrtsoftplus_grouped_topk_bias_selects_but_does_not_weight():
    """e_score_correction_bias steers selection; weights come from raw scores."""
    logits = torch.zeros(1, 4)  # uniform scores: bias fully decides selection
    bias = torch.tensor([0.0, 0.5, 0.0, 1.0])
    weights, ids = _sqrtsoftplus_grouped_topk(logits,
                                              topk=2,
                                              renormalize=False,
                                              num_expert_group=1,
                                              topk_group=1,
                                              e_score_correction_bias=bias)
    assert set(ids[0].tolist()) == {1, 3}
    raw = torch.sqrt(F.softplus(torch.zeros(())))
    torch.testing.assert_close(weights[0], torch.full((2, ), raw.item()))


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
