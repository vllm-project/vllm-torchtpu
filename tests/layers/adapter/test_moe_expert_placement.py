# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time expert placement (TPU_MOE_EXPERT_PLACEMENT). No TPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs, model_loader_patches
from vllm_torchtpu.layers.adapter import moe_expert_placement as mp
from vllm_torchtpu.layers.adapter import moe_placement_glm52 as table

EXPERTS = 256


@pytest.fixture
def glm52(monkeypatch):
    monkeypatch.setattr(
        envs, "TPU_MOE_EXPERT_PLACEMENT", "glm52-ep16-prefill", raising=False
    )
    mp.placement.cache_clear()
    yield mp.placement()
    mp.placement.cache_clear()


def test_unset_keeps_checkpoint_order(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", "", raising=False)
    mp.placement.cache_clear()
    try:
        assert mp.placement() == {}
        name = "model.layers.40.mlp.experts.7.gate_proj.weight"
        assert mp.renamed(name) == name
        gate = torch.randn(EXPERTS, 8)
        assert mp.reordered("model.layers.40.mlp.gate.weight", gate) is gate
    finally:
        mp.placement.cache_clear()


def test_unknown_table_raises(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_EXPERT_PLACEMENT", "nope", raising=False)
    mp.placement.cache_clear()
    try:
        with pytest.raises(ValueError, match="no such table"):
            mp.placement()
    finally:
        mp.placement.cache_clear()


def test_unbalanced_table_raises():
    with pytest.raises(ValueError, match="each of 2 ranks"):
        mp._renumbering("0001", 2)


def test_each_expert_lands_on_its_table_rank(glm52):
    per_rank = EXPERTS // table.RANKS
    assert sorted(glm52) == sorted(table.LAYERS)
    for layer, ranks in table.LAYERS.items():
        new_of_old, old_of_new = glm52[layer]
        assert sorted(new_of_old) == list(range(EXPERTS))
        for old in range(EXPERTS):
            assert old_of_new[new_of_old[old]] == old
            assert new_of_old[old] // per_rank == int(ranks[old], 16)


def test_expert_names_are_renumbered(glm52):
    new = glm52[40][0][7]
    assert (
        mp.renamed("model.layers.40.mlp.experts.7.gate_proj.weight")
        == f"model.layers.40.mlp.experts.{new}.gate_proj.weight"
    )
    for name in (
        "model.layers.78.mlp.experts.7.gate_proj.weight",  # the drafter layer
        "model.layers.2.mlp.gate_proj.weight",  # a dense layer
        "model.layers.40.mlp.shared_experts.down_proj.weight",
    ):
        assert mp.renamed(name) == name


@pytest.mark.parametrize("layer", [3, 40, 77])
def test_router_picks_the_same_experts_with_the_same_weights(glm52, layer):
    generator = torch.Generator().manual_seed(layer)
    hidden, tokens = 64, 512
    gate = torch.randn(EXPERTS, hidden, generator=generator).to(torch.bfloat16)
    bias = torch.randn(EXPERTS, generator=generator) * 0.1
    x = torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16)
    new_gate = mp.reordered(f"model.layers.{layer}.mlp.gate.weight", gate)
    new_bias = mp.reordered(
        f"model.layers.{layer}.mlp.gate.e_score_correction_bias", bias
    )
    shared = f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight"
    assert mp.reordered(shared, gate) is gate

    def route(g, b):
        scores = torch.sigmoid((x @ g.t()).float())
        _, ids = torch.topk(scores + b, 8)
        weights = scores.gather(-1, ids)
        return ids, weights / weights.sum(-1, keepdim=True)

    ids, weights = route(gate, bias)
    new_ids, new_weights = route(new_gate, new_bias)
    old_ids = torch.tensor(glm52[layer][1])[new_ids]
    before, after = ids.sort(-1), old_ids.sort(-1)
    assert torch.equal(before.values, after.values)
    assert torch.equal(
        weights.gather(-1, before.indices), new_weights.gather(-1, after.indices)
    )


def test_weight_filter_keeps_each_rank_its_table_experts(glm52):
    per_rank = EXPERTS // table.RANKS
    for rank in range(table.RANKS):
        local = set(range(per_rank * rank, per_rank * (rank + 1)))
        for layer, ranks in table.LAYERS.items():
            kept = {
                e
                for e in range(EXPERTS)
                if not model_loader_patches._should_skip(
                    mp.renamed(
                        f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"
                    ),
                    local,
                )
            }
            assert kept == {e for e in range(EXPERTS) if int(ranks[e], 16) == rank}


def _moe(experts=EXPERTS, ep=16, use_ep=True):
    """A decoder layer whose mlp is a DeepseekV2MoE exposing only its MoE
    configuration, as the served MoE runner does."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    class _MoE(DeepseekV2MoE):
        def __init__(self, experts):
            torch.nn.Module.__init__(self)
            self.experts = experts

    config = SimpleNamespace(
        num_experts=experts,
        ep_size=ep,
        moe_parallel_config=SimpleNamespace(use_ep=use_ep),
    )
    return SimpleNamespace(mlp=_MoE(SimpleNamespace(moe_config=config)))


def _model(layers):
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_layer_check_on_a_served_shaped_model(glm52):
    dense = [SimpleNamespace(mlp=torch.nn.Identity()) for _ in range(3)]
    layers = dense + [_moe() for _ in range(75)]
    mp._check_layers(_model(layers))
    for bad in (_moe(ep=8), _moe(experts=384), _moe(use_ep=False)):
        with pytest.raises(ValueError, match="linear EP") as error:
            mp._check_layers(_model(layers[:40] + [bad] + layers[41:]))
        assert str(error.value).startswith(
            "TPU_MOE_EXPERT_PLACEMENT=glm52-ep16-prefill: "
        )
        assert str(error.value).endswith(
            "; unset TPU_MOE_EXPERT_PLACEMENT for this model or parallel configuration"
        )
    with pytest.raises(ValueError, match="no entry for MoE layer"):
        mp._check_layers(_model(layers + [_moe()]))
    with pytest.raises(ValueError, match="which has no MoE"):
        mp._check_layers(
            _model(dense + [SimpleNamespace(mlp=torch.nn.Identity())] + layers[4:])
        )


def test_layer_check_refuses_a_model_shorter_than_the_table(glm52):
    dense = [SimpleNamespace(mlp=torch.nn.Identity()) for _ in range(3)]
    deepseek_v3 = dense + [_moe() for _ in range(58)]
    with pytest.raises(ValueError, match="names layers up to 77; the model has 61"):
        mp._check_layers(_model(deepseek_v3))


def _install_over(monkeypatch, load_weights):
    from vllm.model_executor.models import deepseek_v2

    model_class = deepseek_v2.DeepseekV2ForCausalLM
    monkeypatch.setattr(model_class, "load_weights", load_weights)
    monkeypatch.setattr(model_class, "_tpu_expert_placement", False, raising=False)
    monkeypatch.setattr(
        model_loader_patches, "_should_skip", model_loader_patches._should_skip
    )
    monkeypatch.setattr(mp, "_check_layers", lambda model: None)
    monkeypatch.setattr(mp, "_model_accepted", False)
    monkeypatch.setattr(mp, "_loading_accepted_model", False)
    mp.install()
    return model_class


def _filtered(weights, local):
    """The checkpoint as the EP weight filter streams it: lazily, while the
    model's load_weights consumes it."""
    for name, tensor in weights:
        if not model_loader_patches._should_skip(name, local):
            yield name, tensor


def test_install_renumbers_the_loaded_weights(glm52, monkeypatch):
    loaded = []
    model_class = _install_over(monkeypatch, lambda self, w: loaded.extend(w))
    mp.install()  # a second install does not wrap twice

    gate = torch.arange(EXPERTS, dtype=torch.float32)
    weights = [
        ("model.layers.40.mlp.experts.7.down_proj.weight", torch.zeros(1)),
        ("model.layers.40.mlp.gate.weight", gate),
    ]
    model_class.load_weights(object(), _filtered(weights, {glm52[40][0][7]}))
    assert (
        loaded[0][0]
        == f"model.layers.40.mlp.experts.{glm52[40][0][7]}.down_proj.weight"
    )
    assert loaded[1][1].tolist() == glm52[40][1]


def test_filter_leaves_a_later_load_as_it_is(glm52, monkeypatch):
    model_class = _install_over(monkeypatch, lambda self, w: list(w))
    model_class.load_weights(object(), iter([]))
    # A DeepSeek-V3 drafter's MTP layer, loaded after the target by its own
    # load_weights, which does not renumber.
    local = set(range(16))
    kept = {
        e
        for e in range(EXPERTS)
        if not model_loader_patches._should_skip(
            f"model.layers.61.mlp.experts.{e}.down_proj.weight", local
        )
    }
    assert kept == local


def test_filter_refuses_a_model_the_table_was_not_checked_against(glm52, monkeypatch):
    from vllm.model_executor.models import deepseek_v2

    monkeypatch.setattr(deepseek_v2.DeepseekV2ForCausalLM, "load_weights", None)
    monkeypatch.setattr(
        deepseek_v2.DeepseekV2ForCausalLM,
        "_tpu_expert_placement",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        model_loader_patches, "_should_skip", model_loader_patches._should_skip
    )
    monkeypatch.setattr(mp, "_model_accepted", False)
    monkeypatch.setattr(mp, "_loading_accepted_model", False)
    mp.install()
    # A Qwen3-MoE-style name, loaded without DeepseekV2ForCausalLM.load_weights.
    name = "model.layers.40.mlp.experts.7.down_proj.weight"
    with pytest.raises(ValueError, match="not a DeepseekV2ForCausalLM") as error:
        model_loader_patches._should_skip(name, set(range(16)))
    assert str(error.value).startswith("TPU_MOE_EXPERT_PLACEMENT=glm52-ep16-prefill: ")
