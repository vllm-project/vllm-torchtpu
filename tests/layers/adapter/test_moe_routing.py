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
"""Unit tests for vllm_torchtpu.layers.adapter.moe_routing helpers."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import moe_routing


def _make_layer(*, use_ep, ep_size=1, ep_rank=0, global_num_experts=0):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
            ep_size=ep_size,
            ep_rank=ep_rank,
        ),
        global_num_experts=global_num_experts,
    )


# (ep_size, global_num_experts) -- mix of divisible and non-divisible cases,
# plus the production sizes we actually ship (Qwen3-30B: 128, Qwen3-480B: 160).
@pytest.mark.parametrize(
    "ep_size,global_num_experts",
    [
        (2, 8),
        (2, 9),
        (4, 18),
        (4, 33),
        (8, 128),
        (8, 160),
    ],
)
def test_get_experts_start_matches_determine_expert_map(ep_size, global_num_experts):
    """Parity against vLLM's linear-placement contract for every rank."""
    determine_expert_map = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.expert_map_manager"
    ).determine_expert_map
    for ep_rank in range(ep_size):
        _, expert_map, _ = determine_expert_map(
            ep_size=ep_size,
            ep_rank=ep_rank,
            global_num_experts=global_num_experts,
            expert_placement_strategy="linear",
        )
        owned = (expert_map >= 0).nonzero(as_tuple=True)[0]
        expected = int(owned[0].item()) if owned.numel() else 0
        actual = moe_routing.get_experts_start(
            _make_layer(
                use_ep=True,
                ep_size=ep_size,
                ep_rank=ep_rank,
                global_num_experts=global_num_experts,
            )
        )
        assert actual == expected, (
            f"ep_size={ep_size} ep_rank={ep_rank} "
            f"global_num_experts={global_num_experts}: "
            f"expected experts_start={expected}, got {actual}"
        )


def test_get_experts_start_returns_none_without_ep():
    assert moe_routing.get_experts_start(_make_layer(use_ep=False)) is None


def test_validate_linear_ep_placement_accepts_linear():
    moe_routing.validate_linear_ep_placement(
        SimpleNamespace(expert_placement_strategy="linear")
    )


def test_validate_linear_ep_placement_requires_declared_strategy():
    # An object that never declares its placement must fail loudly, not be
    # assumed linear.
    with pytest.raises(AttributeError):
        moe_routing.validate_linear_ep_placement(SimpleNamespace())


def test_validate_linear_ep_placement_rejects_round_robin():
    with pytest.raises(NotImplementedError, match="linear EP placement"):
        moe_routing.validate_linear_ep_placement(
            SimpleNamespace(expert_placement_strategy="round_robin")
        )


def _routing_layer(topk=10, renormalize=True):
    return SimpleNamespace(
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        renormalize=renormalize,
        scoring_func="softmax",
        custom_routing_function=None,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(experts_per_token=topk),
    )


def test_router_topk_env_defaults_to_rowmax():
    import vllm_torchtpu.envs as envs

    assert envs.environment_variables["TPU_MOE_ROUTER_TOPK"]() == "rowmax"


def test_router_topk_env_rejects_unknown_value(monkeypatch):
    import vllm_torchtpu.envs as envs

    monkeypatch.setenv("TPU_MOE_ROUTER_TOPK", "approx")
    with pytest.raises(ValueError, match="TPU_MOE_ROUTER_TOPK"):
        envs.environment_variables["TPU_MOE_ROUTER_TOPK"]()


def test_select_experts_rowmax_matches_sort(monkeypatch):
    """The gate is behaviour-preserving on tie-free data, end to end."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    hidden = torch.randn(64, 8, dtype=torch.bfloat16)
    logits = torch.randn(64, 512)
    layer = _routing_layer()

    def run():
        return moe_routing.select_experts(
            hidden_states=hidden,
            router_logits=logits,
            topk=10,
            renormalize=True,
            scoring_fn="softmax",
            layer=layer,
        )

    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "sort")
    ref_w, ref_i = run()
    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "rowmax")
    got_w, got_i = run()

    assert got_w.dtype == ref_w.dtype and got_i.dtype == torch.int32
    assert torch.equal(got_w, ref_w)
    assert torch.equal(got_i, ref_i.to(torch.int32))


def test_router_topk_op_is_registered_at_import():
    """The op must exist before the first call: a lazy, lock-guarded
    registration is untraceable by Dynamo and aborts the MoE compile."""
    import torch

    from vllm_torchtpu.layers.adapter import router_topk

    assert router_topk._op is not None
    assert hasattr(torch.ops.pallas, "moe_router_topk")


@pytest.mark.parametrize("impl", ["sort", "rowmax"])
def test_routing_path_compiles_fullgraph(monkeypatch, impl):
    """No graph break, no Dynamo error, on the path vLLM compiles.

    The tensors have to be on device: torch_tpu owns the default compile
    backend, so CPU inputs abort the compile before the router is reached.
    """
    torch = pytest.importorskip("torch")
    torch._dynamo.reset()
    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", impl)
    layer = _routing_layer()
    hidden = torch.randn(16, 8, dtype=torch.bfloat16).to("tpu")
    logits = torch.randn(16, 512).to("tpu")

    def run(h, r):
        return moe_routing.select_experts(
            hidden_states=h,
            router_logits=r,
            topk=10,
            renormalize=True,
            scoring_fn="softmax",
            layer=layer,
        )

    # backend="eager" on purpose: this test asserts the *Python* path traces
    # without a graph break, which is device-independent. Leaving the backend
    # unset makes _default_backend_selector pick the TPU backend on a TPU host
    # and try to compile these synthetic CPU tensors for device, which fails
    # for both impls and has nothing to do with what the test checks.
    compiled = torch.compile(run, fullgraph=True, backend="eager", dynamic=False)
    got_w, got_i = compiled(hidden, logits)
    ref_w, ref_i = run(hidden, logits)
    assert torch.equal(got_w, ref_w) and torch.equal(got_i, ref_i)


@pytest.mark.parametrize(
    "field",
    [
        "custom_routing_function",
        "scoring_func",
        "e_score_correction_bias",
        "routed_scaling_factor",
    ],
)
def test_route_requires_declared_routing_fields(monkeypatch, field):
    import torch

    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "sort")
    layer = _routing_layer(topk=1)
    delattr(layer, field)
    with pytest.raises(AttributeError, match=field):
        moe_routing.route(layer, torch.ones(1, 2), torch.tensor([[1.0, 2.0]]))


def test_route_scales_topk_weights_by_the_layer_factor(monkeypatch):
    import torch

    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "sort")
    hidden = torch.ones(2, 4)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    unit = _routing_layer(topk=2)
    scaled = _routing_layer(topk=2)
    scaled.routed_scaling_factor = 2.5
    unit_weights, unit_ids = moe_routing.route(unit, hidden, logits)
    scaled_weights, scaled_ids = moe_routing.route(scaled, hidden, logits)
    assert torch.equal(unit_ids, scaled_ids)
    torch.testing.assert_close(scaled_weights, unit_weights * 2.5)


def test_route_hash_table_scales_by_the_layer_factor():
    import torch

    layer = _routing_layer(topk=2, renormalize=False)
    layer.hash_indices_table = torch.tensor([[0, 1], [2, 3]])
    layer.routed_scaling_factor = 2.0
    hidden = torch.ones(2, 4)
    # Uniform logits score every expert 0.25, so the factor is the only
    # thing shaping the weights.
    logits = torch.zeros(2, 4)
    weights, ids = moe_routing.route(
        layer, hidden, logits, input_ids=torch.tensor([1, 0])
    )
    assert ids.tolist() == [[2, 3], [0, 1]]
    torch.testing.assert_close(weights, torch.full((2, 2), 0.5))
