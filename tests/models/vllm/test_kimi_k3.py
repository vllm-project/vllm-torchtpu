# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the TPU Kimi model."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

import vllm_torchtpu.models.vllm.kimi_k3 as kimi
import vllm_torchtpu.models.vllm.kimi_k3.attention as kimi_attention
import vllm_torchtpu.models.vllm.kimi_k3.layers as kimi_layers
import vllm_torchtpu.models.vllm.kimi_k3.model as kimi_model
import vllm_torchtpu.models.vllm.kimi_k3.moe as kimi_moe
import vllm_torchtpu.models.vllm.kimi_k3.sequence_parallel as kimi_sp
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.vllm.custom_ops import \
    kda_attention_op as kimi_custom_ops
from vllm_torchtpu.models.vllm.kimi_k3 import (KimiDeltaAttention,
                                               KimiLinearForCausalLM,
                                               KimiModel,
                                               MultiHeadLatentAttention)
from vllm_torchtpu.models.vllm.kimi_k3.layers import SituAndMul


def test_kimi_architectures_are_registered() -> None:
    architectures = (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    )
    previous = {
        architecture: ModelRegistry.models.get(architecture)
        for architecture in architectures
    }
    try:
        register_layers()
        for architecture in architectures:
            registered = ModelRegistry.models[architecture]
            assert registered.load_model_cls() is KimiLinearForCausalLM
    finally:
        for architecture, registered in previous.items():
            if registered is None:
                ModelRegistry.models.pop(architecture, None)
            else:
                ModelRegistry.models[architecture] = registered


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul(linear_beta: float | None) -> None:
    activation = SituAndMul(beta=4.0, linear_beta=linear_beta)
    inputs = torch.tensor([[-3.0, 0.5, 7.0, -30.0, 2.0, 40.0]])
    gate, up = inputs.chunk(2, dim=-1)
    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = (up if linear_beta is None else linear_beta *
                   torch.tanh(up / linear_beta))

    torch.testing.assert_close(activation(inputs), expected_gate * expected_up)


def test_checkpoint_name_mapping_is_conditional_for_mla() -> None:
    common_mapper = KimiLinearForCausalLM.hf_to_vllm_mapper
    assert common_mapper.apply_list([
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    ]) == [
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    ]

    fused_mapper = common_mapper | KimiLinearForCausalLM.fused_mla_mapper
    weight = torch.empty(1)
    [(name, weight)] = list(
        fused_mapper.apply([
            ("language_model.layers.1.self_attn.q_a_proj.weight", weight),
        ]))
    assert name == "model.layers.1.self_attn.fused_qkv_a_proj.weight"
    assert weight.shard_id == 0


def test_expert_weight_loader_uses_current_fused_moe_names() -> None:
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=1,
        q_lora_rank=None,
        tie_word_embeddings=False,
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    moe = model.model.layers[0].block_sparse_moe = nn.Module()
    moe.experts = nn.Module()
    moe.experts.routed_experts = nn.Module()
    parameter = nn.Parameter(torch.empty(1))
    moe.experts.routed_experts.w13_weight = parameter

    calls = []

    def weight_loader(param, loaded_weight, weight_name, **kwargs):
        calls.append((param, loaded_weight, weight_name, kwargs))

    set_weight_attrs(parameter, {"weight_loader": weight_loader})
    loaded_weight = torch.ones(1)
    loaded = model.load_weights([
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", loaded_weight),
    ])

    parameter_name = (
        "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight")
    assert loaded == {parameter_name}
    assert calls == [(
        parameter,
        loaded_weight,
        parameter_name,
        {
            "expert_id": 0,
            "shard_id": "w1"
        },
    )]


def test_mxfp4_expert_weight_loader_rebinds_packed_checkpoint_name() -> None:
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=1,
        q_lora_rank=None,
        tie_word_embeddings=False,
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    moe = model.model.layers[0].block_sparse_moe = nn.Module()
    moe.experts = nn.Module()
    moe.experts.routed_experts = nn.Module()
    parameter = nn.Parameter(torch.empty(1))
    moe.experts.routed_experts.w13_weight = parameter

    calls = []

    def weight_loader(param, loaded_weight, weight_name, **kwargs):
        calls.append((param, loaded_weight, weight_name, kwargs))

    set_weight_attrs(parameter, {"weight_loader": weight_loader})
    loaded_weight = torch.ones(1, dtype=torch.uint8)
    loaded = model.load_weights([
        ("language_model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
         loaded_weight),
    ])

    parameter_name = (
        "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight")
    assert loaded == {parameter_name}
    assert calls == [(
        parameter,
        loaded_weight,
        parameter_name,
        {
            "expert_id": 0,
            "shard_id": "w1"
        },
    )]


def test_small_experts_use_current_padded_weight_location(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeMLP(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = nn.Module()
            self.routed_experts.w13_weight = nn.Parameter(torch.ones(1))
            self.routed_experts.w2_weight = nn.Parameter(torch.ones(1))
            self.moe_config = SimpleNamespace(
                intermediate_size_per_partition_unpadded=None)

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 8)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "KimiMLP", FakeMLP)
    monkeypatch.setattr(kimi_moe, "FusedMoE", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=1024,
    )
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["intermediate_size"] == 2048
    assert "activation_situ_beta" not in fused_moe_args
    assert "activation_situ_linear_beta" not in fused_moe_args
    assert layer.experts.moe_config.activation_situ_beta is None
    assert layer.experts.moe_config.activation_situ_linear_beta is None
    assert layer.experts.moe_config.intermediate_size_per_partition_unpadded == 128
    assert not layer.experts.routed_experts.w13_weight.any()
    assert not layer.experts.routed_experts.w2_weight.any()


def test_expert_parallelism_keeps_native_intermediate_size(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = nn.Module()
            self.moe_config = SimpleNamespace()

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
        enable_expert_parallel=True))
    monkeypatch.setattr(kimi_moe, "get_current_vllm_config_or_none",
                        lambda: vllm_config)
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 8)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "FusedMoE", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=1024,
    )
    kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["intermediate_size"] == 1024


def test_situ_moe_uses_tpu_activation_descriptor(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeGate(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = nn.Module()
            self.moe_config = SimpleNamespace()

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 1)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "FusedMoE", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=128,
    )
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["activation"] == "silu"
    assert layer.experts.routed_experts.activation == "situ"
    assert layer.experts.moe_config.activation_situ_beta == 4.0
    assert layer.experts.moe_config.activation_situ_linear_beta == 25.0


def test_latent_moe_reduces_before_norm(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class SquareNorm(nn.Module):

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states.square()

    class IdentityLinear(nn.Module):

        def forward(self, hidden_states: torch.Tensor):
            return hidden_states, None

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 2)
    monkeypatch.setattr(kimi_moe, "tensor_model_parallel_all_reduce",
                        lambda hidden_states: hidden_states + 1)
    transform = kimi_moe.KimiRoutedOutputTransform(SquareNorm(),
                                                   IdentityLinear())
    hidden_states = torch.tensor([[2.0, 3.0]])

    # The division compensates for DefaultMoERunner's final TP reduction.
    torch.testing.assert_close(transform(hidden_states),
                               (hidden_states + 1).square() / 2)


def test_latent_moe_skips_legacy_tp_reduction_with_sequence_parallel(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class SquareNorm(nn.Module):

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states.square()

    class IdentityLinear(nn.Module):

        def forward(self, hidden_states: torch.Tensor):
            return hidden_states, None

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 2)

    def unexpected_all_reduce(hidden_states):
        raise AssertionError("SP output was already reduced-scattered")

    monkeypatch.setattr(kimi_moe, "tensor_model_parallel_all_reduce",
                        unexpected_all_reduce)
    transform = kimi_moe.KimiRoutedOutputTransform(SquareNorm(),
                                                   IdentityLinear(),
                                                   use_sequence_parallel=True)
    hidden_states = torch.tensor([[2.0, 3.0]])

    torch.testing.assert_close(transform(hidden_states),
                               hidden_states.square())


def test_kimi_mlp_replicates_weights_with_sequence_parallel(
        monkeypatch: pytest.MonkeyPatch) -> None:
    constructor_args: list[tuple[str, bool]] = []

    class FakeLinear(nn.Module):

        def __init__(self, *args, disable_tp=False, **kwargs) -> None:
            super().__init__()
            del args, kwargs
            constructor_args.append((type(self).__name__, disable_tp))

        def forward(self, hidden_states):
            return hidden_states, None

    class FakeColumn(FakeLinear):
        pass

    class FakeRow(FakeLinear):
        pass

    monkeypatch.setattr(kimi_layers, "MergedColumnParallelLinear", FakeColumn)
    monkeypatch.setattr(kimi_layers, "RowParallelLinear", FakeRow)
    monkeypatch.setattr(kimi_layers, "SiluAndMul", nn.Identity)
    kimi_layers.KimiMLP(
        16,
        32,
        "silu",
        use_sequence_parallel=True,
    )

    assert constructor_args == [("FakeColumn", True), ("FakeRow", True)]


def test_latent_moe_enables_fused_moe_sequence_parallel(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeLinear(nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, hidden_states):
            return hidden_states, None

    class FakeMLP(FakeLinear):
        pass

    class FakeNorm(FakeLinear):

        def forward(self, hidden_states):
            return hidden_states

    class FakeGate(FakeLinear):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.e_score_correction_bias = None

    class FakeRunner(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = SimpleNamespace(activation=None)
            self.moe_config = SimpleNamespace()

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
        enable_expert_parallel=True))
    monkeypatch.setattr(kimi_moe, "get_current_vllm_config_or_none",
                        lambda: vllm_config)
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size",
                        lambda: 4)
    monkeypatch.setattr(kimi_moe, "GateLinear", FakeGate)
    monkeypatch.setattr(kimi_moe, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_moe, "RMSNorm", FakeNorm)
    monkeypatch.setattr(kimi_moe, "KimiMLP", FakeMLP)
    monkeypatch.setattr(kimi_moe, "FusedMoE", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=128,
        routed_expert_hidden_size=8,
        latent_moe_use_norm=True,
    )
    layer = kimi_moe.KimiMoE(
        config,
        quant_config=None,
        prefix="moe",
        use_sequence_parallel=True,
    )

    assert fused_moe_args["is_sequence_parallel"] is True
    transform = fused_moe_args["routed_output_transform"]
    assert transform.use_sequence_parallel is True
    assert layer.shared_experts is not None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({
            "pipeline_parallel_size": 2
        }, False),
        ({
            "enable_expert_parallel": False
        }, False),
        ({
            "tensor_parallel_size": 1
        }, False),
        ({
            "data_parallel_size": 1
        }, False),
    ],
)
def test_kimi_sequence_parallel_activation(overrides, expected) -> None:
    values = dict(
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        tensor_parallel_size=4,
        data_parallel_size=2,
    )
    values.update(overrides)
    assert kimi_sp.is_kimi_sequence_parallel(
        SimpleNamespace(**values)) is expected


def test_sequence_parallel_helpers_pad_shard_and_gather(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(kimi_sp, "get_tp_group",
                        lambda: SimpleNamespace(device_communicator=None))
    monkeypatch.setattr(kimi_sp, "get_tensor_model_parallel_world_size",
                        lambda: 2)
    monkeypatch.setattr(kimi_sp, "sequence_parallel_chunk",
                        lambda values: values[2:4])
    monkeypatch.setattr(
        kimi_sp,
        "tensor_model_parallel_all_gather",
        lambda values, dim: calls.append(
            ("gather", values.shape, dim)) or values.repeat(2, 1),
    )
    monkeypatch.setattr(
        kimi_sp,
        "tensor_model_parallel_reduce_scatter",
        lambda values, dim: calls.append(
            ("scatter", values.shape, dim)) or values[:values.shape[0] // 2],
    )
    values = torch.arange(10).view(5, 2)

    torch.testing.assert_close(kimi_sp.shard(values), values[2:4])
    assert kimi_sp.all_gather(values[:2]).shape == (4, 2)
    scattered = kimi_sp.reduce_scatter(values)
    assert scattered.shape == (3, 2)
    assert calls == [("gather", torch.Size([2, 2]), 0),
                     ("scatter", torch.Size([6, 2]), 0)]


def test_sequence_parallel_shard_pads_kimi_k3_token_axis(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen_shapes = []

    def chunk(values):
        seen_shapes.append(values.shape)
        padded = torch.cat([values, torch.zeros_like(values[:1])])
        return padded[3:]

    monkeypatch.setattr(kimi_sp, "sequence_parallel_chunk", chunk)
    values = torch.arange(10).view(5, 1, 2)

    result = kimi_sp.shard(values)

    expected = torch.cat([values[3:], torch.zeros_like(values[:1])])
    torch.testing.assert_close(result, expected)
    assert seen_shapes == [torch.Size([5, 2])]


def test_sequence_parallel_reduce_scatter_pads_kimi_k3_token_axis(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen_shapes = []
    communicator = SimpleNamespace(
        custom_reduce_scatter=lambda values: seen_shapes.append(
            values.shape) or values[:values.shape[0] // 2])
    monkeypatch.setattr(
        kimi_sp, "get_tp_group",
        lambda: SimpleNamespace(device_communicator=communicator))
    monkeypatch.setattr(kimi_sp, "get_tensor_model_parallel_world_size",
                        lambda: 2)
    values = torch.arange(10).view(5, 1, 2)

    result = kimi_sp.reduce_scatter(values)

    assert result.shape == (3, 1, 2)
    assert seen_shapes == [torch.Size([6, 2])]


def test_kimi_attention_residual_buffer_uses_sharded_token_count(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class CheckLayer(nn.Module):

        def forward(self, positions, hidden_states, block_residuals):
            assert block_residuals.shape == (hidden_states.shape[0], 0,
                                             hidden_states.shape[1])
            return hidden_states, block_residuals

    class IdentityResidual(nn.Module):

        def forward(self, hidden_states, block_residuals):
            return hidden_states

    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    model.attn_res_block_size = 4
    model.use_sequence_parallel = True
    model.layers = nn.ModuleList([CheckLayer()])
    model.output_attn_res = IdentityResidual()
    model.norm = nn.Identity()
    monkeypatch.setattr(kimi_model, "sp_shard", lambda values: values[:2])
    monkeypatch.setattr(kimi_model, "sp_all_gather", lambda values: values)

    values = torch.randn(5, 8)
    result = KimiModel.forward(model,
                               None,
                               torch.arange(5),
                               inputs_embeds=values)

    assert result.shape == (2, 8)


def test_decoder_sequence_parallel_collective_order(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Identity(nn.Module):

        def forward(self, values):
            return values

    class Attention(nn.Module):

        def forward(self, positions, values):
            calls.append(("attention", values.shape[0], positions.shape[0]))
            return values + 10

    layer = kimi_model.KimiDecoderLayer.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.attn_res_block_size = None
    layer.input_layernorm = Identity()
    layer.self_attn = Attention()
    layer.post_attention_layernorm = Identity()
    layer.is_moe = False
    layer.mlp = Identity()

    def gather(values):
        calls.append(("gather", values.shape[0]))
        return torch.cat((values, values[:1]), dim=0)

    def reduce_scatter(values):
        calls.append(("scatter", values.shape[0]))
        return values[:2]

    monkeypatch.setattr(kimi_model, "sp_all_gather", gather)
    monkeypatch.setattr(kimi_model, "sp_reduce_scatter", reduce_scatter)
    hidden_states = torch.ones(2, 4)
    output, block_residuals = layer(torch.arange(3),
                                    hidden_states,
                                    block_residuals=None)

    assert block_residuals is None
    assert calls == [("gather", 2), ("attention", 3, 3), ("scatter", 3)]
    torch.testing.assert_close(output, torch.full_like(output, 24))


def test_model_has_no_public_cache_or_pp_api() -> None:
    parameters = inspect.signature(KimiLinearForCausalLM.forward).parameters
    assert "cache" not in parameters
    assert "use_cache" not in parameters
    assert not supports_pp(KimiLinearForCausalLM)
    assert not hasattr(kimi, "KimiK3Cache")
    assert not hasattr(kimi, "MLACache")


def test_model_is_registered_for_torch_compile() -> None:
    assert TorchCompileWithNoGuardsWrapper in KimiModel.__bases__


def test_kimi_linear_mla_keeps_its_checkpoint_layout(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeLinear(nn.Module):

        def __init__(self, input_size, output_size, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.output_size = (sum(output_size) if isinstance(
                output_size, list) else output_size)
            self.weight = nn.Parameter(
                torch.empty(self.output_size, input_size))

        def forward(self, inputs):
            return inputs.new_zeros(inputs.shape[0], self.output_size), None

    class FakeMLAWrapper(nn.Module):

        def __init__(self, *args) -> None:
            super().__init__()
            self.mla_modules = args[8]

        def forward(self, positions, hidden_states):
            return hidden_states

    monkeypatch.setattr(kimi_attention, "MultiHeadLatentAttentionWrapper",
                        FakeMLAWrapper)
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MergedColumnParallelLinear",
                        FakeLinear)
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_world_size",
                        lambda: 1)
    config = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
    )
    # This unit test does not have a model_config, so skip the platform-level
    # normalization that requires one. The layer still gets a complete
    # VllmConfig, including the compilation config used by RMSNorm.
    monkeypatch.setattr(current_platform, "check_and_update_config",
                        lambda _: None)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = MultiHeadLatentAttention(
            config,
            vllm_config,
            prefix="model.layers.0.self_attn",
        )

    assert layer.q_proj is not None
    assert layer.kv_a_proj_with_mqa is not None
    assert layer.fused_qkv_a_proj is None
    hidden_states = torch.randn(2, config.hidden_size)
    torch.testing.assert_close(
        layer(torch.arange(2), hidden_states),
        hidden_states,
    )


class _Projection(nn.Module):

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        values = torch.arange(
            inputs.shape[0] * self.output_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return values.view(inputs.shape[0], self.output_size), None


class _IdentityProjection(nn.Module):

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return inputs, None


class _ConvWeight(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 1, 3))


def test_kda_forward_dispatches_to_both_custom_ops() -> None:
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.q_proj = _Projection(4)
    layer.k_proj = _Projection(4)
    layer.v_proj = _Projection(4)
    layer.f_a_proj = _Projection(2)
    layer.f_b_proj = _Projection(4)
    layer.b_proj = _Projection(2)
    layer.g_a_proj = _Projection(2)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False

    sconv_cache = torch.zeros(1, 8, 3)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
    )
    layer._metadata = lambda: metadata

    calls: list[str] = []

    def sconv_op(mixed_qkv, *args):
        calls.append("sconv")
        assert args[0] is sconv_cache
        return mixed_qkv

    def kda_op(mixed_qkv, *args):
        calls.append("kda")
        assert args[3] is recurrent_cache
        return mixed_qkv[:, :4].view(-1, 2, 2)

    layer.sconv_op = sconv_op
    layer.kda_op = kda_op

    output = layer(torch.arange(2), torch.ones(2, 4))
    assert calls == ["sconv", "kda"]
    torch.testing.assert_close(output, torch.arange(8).view(2, 4).float())


def test_kda_custom_ops_compile_as_one_full_graph(
        monkeypatch: pytest.MonkeyPatch) -> None:

    def jax_op(name, function, donate_argnums=()):
        del function, donate_argnums
        if "sconv" in name:

            def implementation(
                mixed_qkv: torch.Tensor,
                conv_state: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del q_weight, k_weight, v_weight
                del query_start_loc, state_indices, seq_lens
                return conv_state + 1, mixed_qkv.clone()

        else:

            def implementation(
                mixed_qkv: torch.Tensor,
                raw_gate: torch.Tensor,
                beta: torch.Tensor,
                output_gate: torch.Tensor,
                recurrent_state: torch.Tensor,
                a_log: torch.Tensor,
                dt_bias: torch.Tensor,
                norm_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del raw_gate, beta, output_gate, a_log, dt_bias, norm_weight
                del query_start_loc, state_indices, seq_lens
                num_heads, head_dim = recurrent_state.shape[1:3]
                output = mixed_qkv[:, :num_heads * head_dim]
                return (output.view(-1, num_heads,
                                    head_dim).clone(), recurrent_state + 1)

        return torch.library.custom_op(name, mutates_args=())(implementation)

    monkeypatch.setattr(kimi_custom_ops.pallas, "jax_op", jax_op)
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.q_proj = _Projection(4)
    layer.k_proj = _Projection(4)
    layer.v_proj = _Projection(4)
    layer.f_a_proj = _Projection(2)
    layer.f_b_proj = _Projection(4)
    layer.b_proj = _Projection(2)
    layer.g_a_proj = _Projection(2)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.sconv_op = kimi_custom_ops.build_kimi_sconv_op(
        "test_compile",
        kernel_size=3,
        state_dim_first=True,
    )
    layer.kda_op = kimi_custom_ops.build_kimi_kda_op(
        "test_compile",
        lower_bound=None,
        eps=1e-5,
    )

    sconv_cache = torch.zeros(1, 12, 2)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
    )
    layer._metadata = lambda: metadata
    hidden_states = torch.ones(2, 4)
    positions = torch.arange(2)

    expected = layer(positions, hidden_states)
    sconv_cache.zero_()
    recurrent_cache.zero_()
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    actual = compiled(positions, hidden_states)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(sconv_cache, torch.ones_like(sconv_cache))
    torch.testing.assert_close(recurrent_cache,
                               torch.ones_like(recurrent_cache))
