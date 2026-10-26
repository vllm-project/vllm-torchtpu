# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for KimiMoE latent AllReduce overlap and 2-chunk prefill pipelining."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu.models.vllm.kimi_k3 import moe as kimi_moe


class _Projection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor):
        return self.fn(x), None


def test_kimi_moe_init_acquires_latent_allreduce_and_intra_chip_shared_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When TP=32 and EP=32 with TPU_K3_MOE_LATENT_AR_OVERLAP=1, KimiMoE wires
    intra-chip TP2 for shared_experts and TP32 for latent routed AllReduce."""
    monkeypatch.setenv("TPU_K3_MOE_LATENT_AR_OVERLAP", "1")
    monkeypatch.setattr(kimi_moe.envs, "TPU_K3_SP_PREFILL", False)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True)
    )
    monkeypatch.setattr(
        kimi_moe, "get_current_vllm_config_or_none", lambda: vllm_config
    )
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size", lambda: 32)
    intra_group = SimpleNamespace(world_size=2, rank_in_group=0)
    tp_group = SimpleNamespace(world_size=32, rank_in_group=0)
    monkeypatch.setattr(kimi_moe, "get_intra_chip_group", lambda: intra_group)
    monkeypatch.setattr(kimi_moe, "get_tp_group", lambda: tp_group)

    mlp_kwargs: dict = {}

    class FakeMLP(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            mlp_kwargs.update(kwargs)

    class FakeGate(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = SimpleNamespace(
                quant_method=SimpleNamespace(apply_monolithic=lambda *a: a[1])
            )
            self.moe_config = SimpleNamespace()

    monkeypatch.setattr(kimi_moe, "KimiMLP", FakeMLP)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "make_latent_projection", FakeGate)
    monkeypatch.setattr(kimi_moe, "RMSNorm", FakeGate)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", lambda **kwargs: FakeRunner())

    config = KimiLinearConfig(
        hidden_size=32,
        routed_expert_hidden_size=16,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=1024,
    )
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")
    assert layer.shared_group is None
    assert layer.prefill_group is tp_group
    assert mlp_kwargs["tp_group"] is None


def test_kimi_moe_init_defaults_to_tp32_shared_experts_and_fused_moe_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """By default when TPU_K3_SP_PREFILL=False, KimiMoE keeps shared_experts at
    TP=32 (shared_group=None) and pipe_chunks=1, delegating forward() to
    self.experts (FusedMoEFactory / MoERunner)."""
    monkeypatch.delenv("TPU_K3_MOE_LATENT_AR_OVERLAP", raising=False)
    monkeypatch.delenv("TPU_K3_MOE_PIPE_CHUNKS", raising=False)
    monkeypatch.setattr(kimi_moe.envs, "TPU_K3_SP_PREFILL", False)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True)
    )
    monkeypatch.setattr(
        kimi_moe, "get_current_vllm_config_or_none", lambda: vllm_config
    )
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size", lambda: 32)
    intra_group = SimpleNamespace(world_size=2, rank_in_group=0)
    tp_group = SimpleNamespace(world_size=32, rank_in_group=0)
    monkeypatch.setattr(kimi_moe, "get_intra_chip_group", lambda: intra_group)
    monkeypatch.setattr(kimi_moe, "get_tp_group", lambda: tp_group)

    mlp_kwargs: dict = {}
    runner_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class FakeMLP(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            mlp_kwargs.update(kwargs)

    class FakeGate(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x[:, :8], None

    class FakeRunner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = SimpleNamespace(
                quant_method=SimpleNamespace(apply_monolithic=lambda *a: a[1])
            )
            self.moe_config = SimpleNamespace()

        def forward(
            self, *, hidden_states: torch.Tensor, router_logits: torch.Tensor
        ) -> torch.Tensor:
            runner_calls.append(
                (tuple(hidden_states.shape), tuple(router_logits.shape))
            )
            return hidden_states * 2.0

    monkeypatch.setattr(kimi_moe, "KimiMLP", FakeMLP)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "make_latent_projection", FakeGate)
    monkeypatch.setattr(kimi_moe, "RMSNorm", FakeGate)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", lambda **kwargs: FakeRunner())

    config = KimiLinearConfig(
        hidden_size=32,
        routed_expert_hidden_size=16,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=1024,
    )
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")
    assert layer.shared_group is None
    assert layer.prefill_group is None
    assert layer.pipe_chunks == 1
    assert mlp_kwargs["tp_group"] is None
    assert mlp_kwargs["reduce_results"] is False

    x = torch.ones((4, 32), dtype=torch.float32)
    out = layer(x, sequence_parallel=False)
    assert runner_calls == [((4, 32), (4, 8))]
    torch.testing.assert_close(out, x * 2.0)


def test_kimi_moe_schedules_shared_experts_between_allreduce_and_routed_up_proj() -> None:
    """shared_experts must launch immediately after group.all_reduce(output) and
    before routed_expert_norm / routed_expert_up_proj so XLA overlaps shared_experts
    compute with the latent AllReduce."""
    events: list[tuple[str, tuple[int, ...]]] = []

    class Group:
        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            events.append(("all_reduce_latent", tuple(x.shape)))
            return x * 2.0

    model = kimi_moe.KimiMoE.__new__(kimi_moe.KimiMoE)
    nn.Module.__init__(model)
    model.shared_group = SimpleNamespace(world_size=2)
    model.prefill_group = Group()
    model.pipe_chunks = 2
    model.gate = _Projection(
        lambda x: (events.append(("gate", tuple(x.shape))) or x[:, :4])
    )
    model.routed_expert_down_proj = _Projection(
        lambda x: (events.append(("down_proj", tuple(x.shape))) or x[:, :2])
    )
    model.routed_expert_norm = lambda x: (
        events.append(("routed_norm", tuple(x.shape))) or (x * 1.5)
    )
    model.routed_expert_up_proj = _Projection(
        lambda x: (events.append(("up_proj", tuple(x.shape))) or x.repeat(1, 2))
    )

    def shared_fn(x: torch.Tensor, *, sequence_parallel: bool = False) -> torch.Tensor:
        assert not sequence_parallel
        events.append(("shared_experts", tuple(x.shape)))
        return x * 0.25

    model.shared_experts = shared_fn
    model.experts = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                apply_monolithic=lambda layer, latent, logits: (
                    events.append(("routed_gmm", tuple(latent.shape)))
                    or (latent + logits[:, :2])
                )
            )
        )
    )

    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    out = model(x, sequence_parallel=False)

    assert [e[0] for e in events] == [
        "gate",
        "down_proj",
        "routed_gmm",
        "all_reduce_latent",
        "shared_experts",
        "routed_norm",
        "up_proj",
    ]
    # Latent AllReduce operates on [T, 2] (latent dim), not [T, 4] (hidden dim)
    assert events[3][1] == (8, 2)
    expected_routed = ((x[:, :2] + x[:, :2]) * 2.0 * 1.5).repeat(1, 2)
    expected = expected_routed + x * 0.25
    torch.testing.assert_close(out, expected)


def test_kimi_moe_tp32_shared_experts_ping_pong_allreduce_overlap() -> None:
    """When shared_group is None (TP=32 shared_experts, reduce_results=False)
    and prefill_group is set (TPU_K3_MOE_LATENT_AR_OVERLAP=1), _forward_expert_parallel
    issues:
      1) group.all_reduce(routed_latent) [T, 2]
      2) shared_experts(hidden_states) [T, 4] (overlapped with [T, 2] AllReduce)
      3) group.all_reduce(shared_output) [T, 4]
      4) routed_expert_norm + routed_expert_up_proj (overlapped with [T, 4] AllReduce)."""
    events: list[tuple[str, tuple[int, ...]]] = []

    class Group:
        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            tag = "all_reduce_latent" if x.shape[-1] == 2 else "all_reduce_shared"
            events.append((tag, tuple(x.shape)))
            return x * 2.0

    model = kimi_moe.KimiMoE.__new__(kimi_moe.KimiMoE)
    nn.Module.__init__(model)
    model.shared_group = None
    model.prefill_group = Group()
    model.pipe_chunks = 1
    model.gate = _Projection(
        lambda x: (events.append(("gate", tuple(x.shape))) or x[:, :4])
    )
    model.routed_expert_down_proj = _Projection(
        lambda x: (events.append(("down_proj", tuple(x.shape))) or x[:, :2])
    )
    model.routed_expert_norm = lambda x: (
        events.append(("routed_norm", tuple(x.shape))) or (x * 1.5)
    )
    model.routed_expert_up_proj = _Projection(
        lambda x: (events.append(("up_proj", tuple(x.shape))) or x.repeat(1, 2))
    )

    def shared_fn(x: torch.Tensor, *, sequence_parallel: bool = False) -> torch.Tensor:
        assert not sequence_parallel
        events.append(("shared_experts", tuple(x.shape)))
        return x * 0.25

    model.shared_experts = shared_fn
    model.experts = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                apply_monolithic=lambda layer, latent, logits: (
                    events.append(("routed_gmm", tuple(latent.shape)))
                    or (latent + logits[:, :2])
                )
            )
        )
    )

    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    out = model(x, sequence_parallel=False)

    assert [e[0] for e in events] == [
        "gate",
        "down_proj",
        "routed_gmm",
        "all_reduce_latent",
        "shared_experts",
        "all_reduce_shared",
        "routed_norm",
        "up_proj",
    ]
    assert events[3] == ("all_reduce_latent", (8, 2))
    assert events[5] == ("all_reduce_shared", (8, 4))
    expected_routed = ((x[:, :2] + x[:, :2]) * 2.0 * 1.5).repeat(1, 2)
    expected_shared = (x * 0.25) * 2.0
    torch.testing.assert_close(out, expected_routed + expected_shared)


@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_kimi_moe_two_chunk_prefill_pipelining_order_and_parity(
    num_tokens: int,
) -> None:
    """For non-SP prefill T >= 4096, 2-chunk pipelining must issue both chunks'
    down_proj + routed_gmm + all_reduce before shared_experts and before either
    chunk's norm + up_proj, while matching unchunked numerical output."""
    events: list[tuple[str, int]] = []

    class Group:
        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            events.append(("all_reduce", x.shape[0]))
            return x * 1.25

    w_down = torch.randn(8, 4)
    w_up = torch.randn(4, 8)
    w_gate = torch.randn(8, 6)

    def make_model(pipe_chunks: int) -> kimi_moe.KimiMoE:
        model = kimi_moe.KimiMoE.__new__(kimi_moe.KimiMoE)
        nn.Module.__init__(model)
        model.shared_group = SimpleNamespace(world_size=2)
        model.prefill_group = Group()
        model.pipe_chunks = pipe_chunks
        model.gate = _Projection(
            lambda x: (events.append(("gate", x.shape[0])) or (x @ w_gate))
        )
        model.routed_expert_down_proj = _Projection(
            lambda x: (events.append(("down_proj", x.shape[0])) or (x @ w_down))
        )
        model.routed_expert_norm = lambda x: (
            events.append(("routed_norm", x.shape[0]))
            or (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))
        )
        model.routed_expert_up_proj = _Projection(
            lambda x: (events.append(("up_proj", x.shape[0])) or (x @ w_up))
        )
        model.shared_experts = lambda x, *, sequence_parallel=False: (
            events.append(("shared_experts", x.shape[0])) or (torch.tanh(x) * 0.3)
        )
        model.experts = SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(
                    apply_monolithic=lambda layer, latent, logits: (
                        events.append(("routed_gmm", latent.shape[0]))
                        or (latent * torch.sigmoid(logits[:, :4]))
                    )
                )
            )
        )
        return model

    x = torch.randn(num_tokens, 8)
    unchunked = make_model(pipe_chunks=1)(x, sequence_parallel=False)

    events.clear()
    chunked = make_model(pipe_chunks=2)(x, sequence_parallel=False)

    half = num_tokens // 2
    assert events == [
        ("gate", half),
        ("down_proj", half),
        ("routed_gmm", half),
        ("all_reduce", half),
        ("gate", half),
        ("down_proj", half),
        ("routed_gmm", half),
        ("all_reduce", half),
        ("shared_experts", num_tokens),
        ("routed_norm", half),
        ("up_proj", half),
        ("routed_norm", half),
        ("up_proj", half),
    ]
    torch.testing.assert_close(chunked, unchunked, rtol=1e-5, atol=1e-5)
