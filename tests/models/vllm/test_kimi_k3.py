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
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.kimi_k3.common.mm_preprocess import navit_resize_image
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

import vllm_torchtpu.models.vllm.kimi_k3 as kimi
import vllm_torchtpu.models.vllm.kimi_k3.attention as kimi_attention
import vllm_torchtpu.models.vllm.kimi_k3.moe as kimi_moe
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.vllm.custom_ops import \
    kda_attention_op as kimi_custom_ops
from vllm_torchtpu.models.vllm.kimi_k3 import (KimiDeltaAttention,
                                               KimiK3ForConditionalGeneration,
                                               KimiLinearForCausalLM,
                                               KimiModel,
                                               MultiHeadLatentAttention)
from vllm_torchtpu.models.vllm.kimi_k3.layers import SituAndMul


def test_kimi_architectures_are_registered() -> None:
    architectures = {
        "KimiLinearForCausalLM": KimiLinearForCausalLM,
        "KimiK3ForConditionalGeneration": KimiK3ForConditionalGeneration,
    }
    previous = {
        architecture: ModelRegistry.models.get(architecture)
        for architecture in architectures
    }
    previous_config = MODELS_CONFIG_MAP.get("KimiK3ForConditionalGeneration")
    try:
        register_layers()
        for architecture, expected in architectures.items():
            registered = ModelRegistry.models[architecture]
            assert registered.load_model_cls() is expected

        quantization_config = {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
        }
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                quantization_config=quantization_config.copy()),
            hf_text_config=SimpleNamespace(
                quantization_config=quantization_config.copy()),
            model_arch_config=SimpleNamespace(
                quantization_config=quantization_config.copy()),
        )
        config_handler = MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"]
        config_handler.verify_and_update_model_config(model_config)
        assert all(
            config.quantization_config["quant_method"] == "compressed-tensors"
            for config in (
                model_config.hf_config,
                model_config.hf_text_config,
                model_config.model_arch_config,
            ))
    finally:
        if previous_config is None:
            MODELS_CONFIG_MAP.pop("KimiK3ForConditionalGeneration", None)
        else:
            MODELS_CONFIG_MAP[
                "KimiK3ForConditionalGeneration"] = previous_config
        for architecture, registered in previous.items():
            if registered is None:
                ModelRegistry.models.pop(architecture, None)
            else:
                ModelRegistry.models[architecture] = registered


def test_kimi_k3_multimodal_interface_and_weight_mapping() -> None:
    assert KimiK3ForConditionalGeneration.supports_multimodal
    assert KimiK3ForConditionalGeneration.supports_encoder_tp_data
    assert (KimiK3ForConditionalGeneration.get_placeholder_str(
        "image", 0) == "<|kimi_image_placeholder|>")
    with pytest.raises(ValueError, match="Unsupported modality"):
        KimiK3ForConditionalGeneration.get_placeholder_str("video", 0)

    assert KimiK3ForConditionalGeneration.hf_to_vllm_mapper.apply_list([
        "language_model.layers.1.self_attn.q_proj.weight",
        "language_model.model.embed_tokens.weight",
        "mm_projector.proj.0.weight",
        "mm_projector.proj.2.weight",
        "vision_tower.encoder.blocks.0.norm1.weight",
    ]) == [
        "language_model.model.layers.1.self_attn.q_proj.weight",
        "language_model.model.embed_tokens.weight",
        "mm_projector.linear_1.weight",
        "mm_projector.linear_2.weight",
        "vision_tower.encoder.blocks.0.norm1.weight",
    ]


def test_kimi_k3_navit_resize_aligns_to_merged_patches() -> None:
    resized = navit_resize_image(
        width=400,
        height=80,
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=65536,
        patch_limit_on_one_side=512,
        fixed_output_tokens=None,
    )
    assert (resized["new_width"] + resized["pad_width"]) % 28 == 0
    assert (resized["new_height"] + resized["pad_height"]) % 28 == 0
    assert resized["num_tokens"] == 45


def test_kimi_k3_scatter_free_embedding_merge() -> None:
    model = KimiK3ForConditionalGeneration.__new__(
        KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model._has_oov_mm_tokens = False
    language_model = nn.Module()
    language_model.embed_input_ids = lambda ids: torch.stack(  # type: ignore[method-assign]
        (ids.float(), ids.float() + 10),
        dim=-1)
    model.language_model = language_model
    model._language_model_names = ["language_model"]

    input_ids = torch.tensor([1, 2, 3, 4])
    is_multimodal = torch.tensor([False, True, True, False])
    media = [torch.tensor([[20.0, 21.0], [30.0, 31.0]])]
    actual = model.embed_input_ids(input_ids,
                                   media,
                                   is_multimodal=is_multimodal)
    expected = torch.tensor([[1.0, 11.0], [20.0, 21.0], [30.0, 31.0],
                             [4.0, 14.0]])
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="1 multimodal tokens to 2"):
        model.embed_input_ids(input_ids, [media[0][:1]],
                              is_multimodal=is_multimodal)


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

    parameter_name = "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"
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
        (
            "language_model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
            loaded_weight,
        ),
    ])

    parameter_name = "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"
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


@pytest.mark.parametrize(
    ("min_per_partition", "expected_intermediate", "expected_unpadded"),
    [(None, 1024, None), (256, 2048, 128)],
)
def test_small_experts_use_current_padded_weight_location(
    monkeypatch: pytest.MonkeyPatch,
    min_per_partition: int | None,
    expected_intermediate: int,
    expected_unpadded: int | None,
) -> None:

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
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
    config = KimiLinearConfig(
        hidden_size=16,
        hidden_act="silu",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=1024,
    )
    if min_per_partition is not None:
        config.min_moe_intermediate_per_partition = min_per_partition
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["intermediate_size"] == expected_intermediate
    assert fused_moe_args["activation_situ_beta"] is None
    assert fused_moe_args["activation_situ_linear_beta"] is None
    assert (layer.experts.moe_config.intermediate_size_per_partition_unpadded
            == expected_unpadded)
    expect_zero = expected_unpadded is not None
    assert bool(layer.experts.routed_experts.w13_weight.any()) != expect_zero
    assert bool(layer.experts.routed_experts.w2_weight.any()) != expect_zero


def test_expert_parallelism_keeps_native_intermediate_size(
    monkeypatch: pytest.MonkeyPatch, ) -> None:

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
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
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
    monkeypatch: pytest.MonkeyPatch, ) -> None:

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
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", fake_fused_moe)
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
    kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert fused_moe_args["activation"] == "situ"
    assert fused_moe_args["activation_situ_beta"] == 4.0
    assert fused_moe_args["activation_situ_linear_beta"] == 25.0


def test_latent_moe_transform_leaves_reduction_to_vllm_runner() -> None:

    class SquareNorm(nn.Module):

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states.square()

    class IdentityLinear(nn.Module):

        def forward(self, hidden_states: torch.Tensor):
            return hidden_states, None

    transform = kimi_moe.KimiRoutedOutputTransform(SquareNorm(),
                                                   IdentityLinear())
    hidden_states = torch.tensor([[2.0, 3.0]])

    # vLLM v0.27 reduces the routed latent output before invoking this
    # nonlinear transform. Reducing again here underweights it at TP > 1.
    torch.testing.assert_close(transform(hidden_states),
                               hidden_states.square())


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
    monkeypatch: pytest.MonkeyPatch, ) -> None:

    class FakeLinear(nn.Module):

        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
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


def test_mla_fuses_output_gate_with_lora_a_projections(
        monkeypatch: pytest.MonkeyPatch) -> None:

    class FakeLinear(nn.Module):

        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
            del kwargs
            self.output_size = (sum(output_size) if isinstance(
                output_size, list) else output_size)
            self.weight = nn.Parameter(
                torch.empty(self.output_size, input_size))

    class FakeMLAWrapper(nn.Module):

        def __init__(self, *args) -> None:
            super().__init__()
            self.mla_modules = args[8]

    monkeypatch.setattr(kimi_attention, "MultiHeadLatentAttentionWrapper",
                        FakeMLAWrapper)
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MixedParallelMergedLinear",
                        FakeLinear)
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_world_size",
                        lambda: 1)
    config = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
        mla_use_output_gate=True,
    )
    monkeypatch.setattr(current_platform, "check_and_update_config",
                        lambda _: None)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = MultiHeadLatentAttention(
            config,
            vllm_config,
            prefix="model.layers.0.self_attn",
        )

    assert layer.fused_qkv_a_proj.output_size == 8 + (8 + 2) + 4 * 4
    assert layer.g_proj is None
    assert layer.mla_attn.mla_modules.gate_is_fused is True


@pytest.mark.parametrize("use_parameter_hook", [False, True],
                         ids=["legacy-loader", "v2-parameter-hook"])
def test_mixed_parallel_merged_linear_loads_replicated_and_tp_slices(
        monkeypatch: pytest.MonkeyPatch, use_parameter_hook: bool) -> None:
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_world_size",
                        lambda: 2)
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_rank",
                        lambda: 1)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank",
                        lambda: 0)
    monkeypatch.setattr(parameter_module,
                        "get_tensor_model_parallel_world_size", lambda: 1)
    layer = kimi_attention.MixedParallelMergedLinear(
        2,
        [4, 3],
        [False, True],
        bias=False,
        quant_config=None,
        prefix="mixed",
    )
    column_weight = torch.arange(8, dtype=layer.weight.dtype).view(4, 2)
    replicated_weight = torch.arange(6, dtype=layer.weight.dtype).view(3,
                                                                       2) + 100

    loader = (layer.weight.weight_loader
              if use_parameter_hook else layer.weight_loader)
    loader(layer.weight, column_weight, 0)
    loader(layer.weight, replicated_weight, 1)

    expected = torch.cat((column_weight[2:], replicated_weight), dim=0)
    torch.testing.assert_close(layer.weight, expected)


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
    layer.fused_qkvb_proj = _Projection(14)
    layer.fused_fa_ga_proj = _Projection(4)
    layer.f_a_proj = None
    layer.g_proj = None
    layer.local_projection_size = 4
    layer.num_heads = 2
    layer.head_dim = 2
    layer.f_b_proj = _Projection(4)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    # The ops take the fused conv weight, built once at load time.
    layer.conv_size, layer.num_heads, layer.head_dim = 3, 2, 2
    KimiDeltaAttention.process_weights_after_loading(layer,
                                                     act_dtype=torch.bfloat16)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.use_naive_kda = False

    sconv_cache = torch.zeros(1, 8, 3)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    layer._metadata = lambda: metadata

    calls: list[str] = []

    def dispatched_kda_op(mixed_qkv, *args):
        calls.append("dispatched")
        # It owns both caches: the convolution is fused into the decode kernel,
        # so there is no separate convolution op to hand the conv cache to.
        assert args[3] is sconv_cache
        assert args[4] is recurrent_cache
        return mixed_qkv[:, :4].view(-1, 2, 2)

    def unexpected(name):

        def op(*args, **kwargs):
            raise AssertionError(f"{name} must not run for this batch")

        return op

    layer.dispatched_kda_op = dispatched_kda_op
    # Neither of these is reachable without VLLM_TPU_USE_NAIVE_KDA, and a
    # separate `sconv_op` in front of the dispatched op would convolve twice.
    layer.sconv_op = unexpected("sconv_op")
    layer.chunk_kda_op = unexpected("chunk_kda_op")
    layer.kda_op = unexpected("kda_op")

    output = layer(torch.arange(2), torch.ones(2, 4))
    assert calls == ["dispatched"]
    torch.testing.assert_close(
        output,
        torch.tensor([[0, 1, 2, 3], [14, 15, 16, 17]], dtype=torch.float32),
    )


def test_kda_custom_ops_compile_as_one_full_graph(
    monkeypatch: pytest.MonkeyPatch, ) -> None:

    def jax_op(name, function, donate_argnums=()):
        del function, donate_argnums
        if "dispatched" in name:

            def implementation(
                mixed_qkv: torch.Tensor,
                raw_gate: torch.Tensor,
                beta: torch.Tensor,
                output_gate: torch.Tensor,
                conv_state: torch.Tensor,
                recurrent_state: torch.Tensor,
                conv_weight: torch.Tensor,
                a_log: torch.Tensor,
                dt_bias: torch.Tensor,
                norm_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
                distribution: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                del raw_gate, beta, output_gate, a_log, dt_bias, norm_weight
                del conv_weight, distribution
                del query_start_loc, state_indices, seq_lens
                num_heads, head_dim = recurrent_state.shape[1:3]
                output = mixed_qkv[:, :num_heads * head_dim]
                return (
                    output.view(-1, num_heads, head_dim).clone(),
                    conv_state + 1,
                    recurrent_state + 1,
                )

        elif "sconv" in name:

            def implementation(
                mixed_qkv: torch.Tensor,
                conv_state: torch.Tensor,
                conv_weight: torch.Tensor,
                query_start_loc: torch.Tensor,
                state_indices: torch.Tensor,
                seq_lens: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del conv_weight
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
                return (
                    output.view(-1, num_heads, head_dim).clone(),
                    recurrent_state + 1,
                )

        return torch.library.custom_op(name, mutates_args=())(implementation)

    monkeypatch.setattr(kimi_custom_ops.pallas, "jax_op", jax_op)
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.fused_qkvb_proj = _Projection(14)
    layer.fused_fa_ga_proj = _Projection(4)
    layer.f_a_proj = None
    layer.g_proj = None
    layer.local_projection_size = 4
    layer.num_heads = 2
    layer.head_dim = 2
    layer.f_b_proj = _Projection(4)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    # The ops take the fused conv weight, built once at load time.
    layer.conv_size, layer.num_heads, layer.head_dim = 3, 2, 2
    KimiDeltaAttention.process_weights_after_loading(layer,
                                                     act_dtype=torch.bfloat16)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.use_naive_kda = False
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
    layer.chunk_kda_op = kimi_custom_ops.build_kimi_chunk_kda_op(
        "test_compile",
        lower_bound=None,
        eps=1e-5,
    )
    layer.dispatched_kda_op = kimi_custom_ops.build_kimi_dispatched_kda_op(
        "test_compile",
        lower_bound=None,
        eps=1e-5,
        state_dim_first=True,
    )

    sconv_cache = torch.zeros(1, 12, 2)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
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
