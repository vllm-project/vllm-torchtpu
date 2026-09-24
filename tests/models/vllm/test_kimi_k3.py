# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the TPU Kimi model."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.fx.experimental._config as fx_config
from torch import nn
from torch._dynamo import mark_dynamic
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.kimi_k3.common.mm_preprocess import navit_resize_image
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

import vllm_torchtpu.models.vllm.kimi_k3 as kimi
import vllm_torchtpu.models.vllm.kimi_k3.attention as kimi_attention
import vllm_torchtpu.models.vllm.kimi_k3.model as kimi_model
import vllm_torchtpu.models.vllm.kimi_k3.moe as kimi_moe
from vllm_torchtpu.compilation.shape_variants import trace_shape_env, unsupported_reason
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as kimi_custom_ops
from vllm_torchtpu.models.vllm.kimi_k3 import (
    KimiDeltaAttention,
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiModel,
    MultiHeadLatentAttention,
)
from vllm_torchtpu.models.vllm.kimi_k3.layers import AttentionResidual, SituAndMul


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    # These model tests use CPU tensors and stubbed custom ops. Dynamo still
    # queries registered devices; keep it from opening a distributed TPU
    # client merely to inspect the current device while tracing CPU code.
    #
    # torch.tpu is registered by importing torch_tpu, which the cpu_test job
    # never does: tests/conftest.py imports it only under --use-tpu, and
    # importing it here would claim the accelerator in the pytest parent and
    # lock out every spawned child. With no device module registered there is
    # nothing for Dynamo to query, so there is nothing to stub either.
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return
    monkeypatch.setattr(tpu, "is_available", lambda: False)
    monkeypatch.setattr(tpu, "current_device", lambda: 0)
    monkeypatch.setattr(tpu, "device_count", lambda: 0)
    monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)


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
            hf_config=SimpleNamespace(quantization_config=quantization_config.copy()),
            hf_text_config=SimpleNamespace(
                quantization_config=quantization_config.copy()
            ),
            model_arch_config=SimpleNamespace(
                quantization_config=quantization_config.copy()
            ),
        )
        config_handler = MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"]
        config_handler.verify_and_update_model_config(model_config)
        assert all(
            config.quantization_config["quant_method"] == "compressed-tensors"
            for config in (
                model_config.hf_config,
                model_config.hf_text_config,
                model_config.model_arch_config,
            )
        )
    finally:
        if previous_config is None:
            MODELS_CONFIG_MAP.pop("KimiK3ForConditionalGeneration", None)
        else:
            MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = previous_config
        for architecture, registered in previous.items():
            if registered is None:
                ModelRegistry.models.pop(architecture, None)
            else:
                ModelRegistry.models[architecture] = registered


def test_kimi_k3_multimodal_interface_and_weight_mapping() -> None:
    assert KimiK3ForConditionalGeneration.supports_multimodal
    assert KimiK3ForConditionalGeneration.supports_encoder_tp_data
    assert (
        KimiK3ForConditionalGeneration.get_placeholder_str("image", 0)
        == "<|kimi_image_placeholder|>"
    )
    with pytest.raises(ValueError, match="Unsupported modality"):
        KimiK3ForConditionalGeneration.get_placeholder_str("video", 0)

    assert KimiK3ForConditionalGeneration.hf_to_vllm_mapper.apply_list(
        [
            "language_model.layers.1.self_attn.q_proj.weight",
            "language_model.model.embed_tokens.weight",
            "mm_projector.proj.0.weight",
            "mm_projector.proj.2.weight",
            "vision_tower.encoder.blocks.0.norm1.weight",
        ]
    ) == [
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
    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model._has_oov_mm_tokens = False
    language_model = nn.Module()
    language_model.embed_input_ids = lambda ids: torch.stack(  # type: ignore[method-assign]
        (ids.float(), ids.float() + 10), dim=-1
    )
    model.language_model = language_model
    model._language_model_names = ["language_model"]

    input_ids = torch.tensor([1, 2, 3, 4])
    is_multimodal = torch.tensor([False, True, True, False])
    media = [torch.tensor([[20.0, 21.0], [30.0, 31.0]])]
    actual = model.embed_input_ids(input_ids, media, is_multimodal=is_multimodal)
    expected = torch.tensor([[1.0, 11.0], [20.0, 21.0], [30.0, 31.0], [4.0, 14.0]])
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="1 multimodal tokens to 2"):
        model.embed_input_ids(input_ids, [media[0][:1]], is_multimodal=is_multimodal)


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul(linear_beta: float | None) -> None:
    activation = SituAndMul(beta=4.0, linear_beta=linear_beta)
    inputs = torch.tensor([[-3.0, 0.5, 7.0, -30.0, 2.0, 40.0]])
    gate, up = inputs.chunk(2, dim=-1)
    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = (
        up if linear_beta is None else linear_beta * torch.tanh(up / linear_beta)
    )

    torch.testing.assert_close(activation(inputs), expected_gate * expected_up)


def test_checkpoint_name_mapping_is_conditional_for_mla() -> None:
    common_mapper = KimiLinearForCausalLM.hf_to_vllm_mapper
    assert common_mapper.apply_list(
        [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        ]
    ) == [
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    ]

    fused_mapper = common_mapper | KimiLinearForCausalLM.fused_mla_mapper
    weight = torch.empty(1)
    [(name, weight)] = list(
        fused_mapper.apply(
            [
                ("language_model.layers.1.self_attn.q_a_proj.weight", weight),
            ]
        )
    )
    assert name == "model.layers.1.self_attn.fused_qkv_a_proj.weight"
    assert weight.shard_id == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_weight_loader_preserves_lm_head_skip_and_streaming(tied) -> None:
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=None, q_lora_rank=None, tie_word_embeddings=tied
    )
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(2, 3)
    # The production constructor creates a separate head even with the flag
    # enabled; native alias detection alone cannot replace its skip policy.
    model.lm_head = nn.Linear(3, 2, bias=False)
    model.lm_head_extra = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.fill_(-1)
    embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    head = embedding + 10
    extra = embedding + 20

    def checkpoint():
        yield "language_model.model.embed_tokens.weight", embedding
        yield "language_model.lm_head.weight", head
        # Verify dense checkpoint tensors are consumed before advancing the
        # stream, rather than first being retained in a full-checkpoint list.
        torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
        if tied:
            yield "language_model.lm_head.extra.unexpected.weight", head
        yield "language_model.lm_head_extra.weight", extra

    loaded = model.load_weights(checkpoint())
    expected = {"model.embed_tokens.weight", "lm_head_extra.weight"}
    if not tied:
        expected.add("lm_head.weight")
    assert loaded == expected
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.lm_head_extra.weight, extra)
    torch.testing.assert_close(
        model.lm_head.weight, torch.full_like(head, -1) if tied else head
    )
    with pytest.raises(ValueError, match="not_in_model"):
        model.load_weights([("not_in_model.weight", head)])


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
    loaded = model.load_weights(
        [
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", loaded_weight),
        ]
    )

    parameter_name = "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"
    assert loaded == {parameter_name}
    assert calls == [
        (
            parameter,
            loaded_weight,
            parameter_name,
            {"expert_id": 0, "shard_id": "w1"},
        )
    ]


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
    loaded = model.load_weights(
        [
            (
                "language_model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
                loaded_weight,
            ),
        ]
    )

    parameter_name = "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"
    assert loaded == {parameter_name}
    assert calls == [
        (
            parameter,
            loaded_weight,
            parameter_name,
            {"expert_id": 0, "shard_id": "w1"},
        )
    ]


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
                intermediate_size_per_partition_unpadded=None
            )

    fused_moe_args = {}

    def fake_fused_moe(**kwargs):
        fused_moe_args.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size", lambda: 8)
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
    assert (
        layer.experts.moe_config.intermediate_size_per_partition_unpadded
        == expected_unpadded
    )
    expect_zero = expected_unpadded is not None
    assert bool(layer.experts.routed_experts.w13_weight.any()) != expect_zero
    assert bool(layer.experts.routed_experts.w2_weight.any()) != expect_zero


def test_expert_parallelism_keeps_native_intermediate_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True)
    )
    monkeypatch.setattr(
        kimi_moe, "get_current_vllm_config_or_none", lambda: vllm_config
    )
    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size", lambda: 8)
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size", lambda: 1)
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

    transform = kimi_moe.KimiRoutedOutputTransform(SquareNorm(), IdentityLinear())
    hidden_states = torch.tensor([[2.0, 3.0]])

    # vLLM v0.27 reduces the routed latent output before invoking this
    # nonlinear transform. Reducing again here underweights it at TP > 1.
    torch.testing.assert_close(transform(hidden_states), hidden_states.square())


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = SimpleNamespace(all_reduce=lambda x: x)
    monkeypatch.setattr(kimi_attention, "get_tp_group", lambda: group)

    class FakeLinear(nn.Module):
        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
            del kwargs
            self.output_size = (
                sum(output_size) if isinstance(output_size, list) else output_size
            )
            self.weight = nn.Parameter(torch.empty(self.output_size, input_size))

        def forward(self, inputs):
            return inputs.new_zeros(inputs.shape[0], self.output_size), None

    class FakeMLAWrapper(nn.Module):
        def __init__(
            self, *args, gate_is_fused, non_causal_multi_token_decode=False
        ) -> None:
            super().__init__()
            self.mla_modules = args[8]
            self.non_causal_multi_token_decode = non_causal_multi_token_decode

        def forward(self, positions, hidden_states):
            return hidden_states

    monkeypatch.setattr(
        kimi_attention, "MultiHeadLatentAttentionWrapper", FakeMLAWrapper
    )
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MergedColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
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
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
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
    assert layer.mla_attn.non_causal_multi_token_decode is False
    hidden_states = torch.randn(2, config.hidden_size)
    torch.testing.assert_close(
        layer(torch.arange(2), hidden_states),
        hidden_states,
    )


def test_mla_fuses_output_gate_with_lora_a_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
            del kwargs
            self.output_size = (
                sum(output_size) if isinstance(output_size, list) else output_size
            )
            self.weight = nn.Parameter(torch.empty(self.output_size, input_size))

    class FakeMLAWrapper(nn.Module):
        def __init__(
            self, *args, gate_is_fused, non_causal_multi_token_decode=False
        ) -> None:
            super().__init__()
            self.mla_modules = args[8]
            self.gate_is_fused = gate_is_fused
            self.non_causal_multi_token_decode = non_causal_multi_token_decode

    monkeypatch.setattr(
        kimi_attention, "MultiHeadLatentAttentionWrapper", FakeMLAWrapper
    )
    monkeypatch.setattr(kimi_attention, "ColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "MixedParallelMergedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(kimi_attention, "RowParallelLinear", FakeLinear)
    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
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
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = MultiHeadLatentAttention(
            config,
            vllm_config,
            prefix="model.layers.0.self_attn",
        )

    assert layer.fused_qkv_a_proj.output_size == 8 + (8 + 2) + 4 * 4
    assert layer.g_proj is None
    assert layer.mla_attn.gate_is_fused is True
    assert layer.mla_attn.non_causal_multi_token_decode is False


@pytest.mark.parametrize(
    "use_parameter_hook", [False, True], ids=["legacy-loader", "v2-parameter-hook"]
)
def test_mixed_parallel_merged_linear_loads_replicated_and_tp_slices(
    monkeypatch: pytest.MonkeyPatch, use_parameter_hook: bool
) -> None:
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    layer = kimi_attention.MixedParallelMergedLinear(
        2,
        [4, 3],
        [False, True],
        bias=False,
        quant_config=None,
        prefix="mixed",
    )
    column_weight = torch.arange(8, dtype=layer.weight.dtype).view(4, 2)
    replicated_weight = torch.arange(6, dtype=layer.weight.dtype).view(3, 2) + 100

    loader = layer.weight.weight_loader if use_parameter_hook else layer.weight_loader
    loader(layer.weight, column_weight, 0)
    loader(layer.weight, replicated_weight, 1)

    expected = torch.cat((column_weight[2:], replicated_weight), dim=0)
    torch.testing.assert_close(layer.weight, expected)


def _mla_config(**overrides) -> KimiLinearConfig:
    """A small, valid NoPE MLA config. Tests override one field at a time
    to hit a specific rejection in `MultiHeadLatentAttention.__init__`."""
    kwargs = dict(
        hidden_size=16,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
    )
    kwargs.update(overrides)
    return KimiLinearConfig(**kwargs)


def test_mla_requires_nope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The TPU MLA layer only implements the NoPE variant. A config that
    asks for the RoPE-inside-MLA variant is rejected up front."""
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
    config = _mla_config(mla_use_nope=False)
    with pytest.raises(ValueError, match="NoPE MLA only"):
        MultiHeadLatentAttention(config, VllmConfig(), prefix="a")


def test_mla_rejects_incomplete_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four MLA head dimensions must be set. Leaving one as `None`
    produces an error that lists every field, so the user can see which
    one is missing."""
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
    config = _mla_config(kv_lora_rank=None)
    with pytest.raises(ValueError, match="Incomplete Kimi K3 MLA config"):
        MultiHeadLatentAttention(config, VllmConfig(), prefix="b")


def test_mla_rejects_heads_not_divisible_by_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attention heads are split evenly across tensor-parallel ranks, so
    4 heads cannot be spread over 3 ranks."""
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 3
    )
    config = _mla_config(num_attention_heads=4)
    with pytest.raises(ValueError, match="divisible by TP size"):
        MultiHeadLatentAttention(config, VllmConfig(), prefix="c")


def test_mixed_parallel_merged_linear_rejects_length_mismatch() -> None:
    """`MixedParallelMergedLinear` takes one output size and one
    replicate/shard flag per fused output. Two sizes with only one flag is
    a caller bug and is rejected."""
    with pytest.raises(ValueError, match="sharding mode"):
        kimi_attention.MixedParallelMergedLinear(
            2, [4, 3], [False], bias=False, quant_config=None, prefix="mismatch"
        )


def test_mixed_parallel_merged_linear_rejects_tp_indivisible_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An output marked as column-parallel is split across TP ranks, so its
    size must divide evenly. Here the first output (size 3, sharded) does
    not divide by TP=2; the second (size 4, replicated) is never split and
    would have been fine."""
    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 2
    )
    with pytest.raises(ValueError, match="must divide TP"):
        kimi_attention.MixedParallelMergedLinear(
            2,
            [3, 4],
            [False, True],
            bias=False,
            quant_config=None,
            prefix="indivisible",
        )


def test_kda_state_dtype_reads_model_and_cache_config() -> None:
    """`kda_state_dtype` returns the storage dtypes for a KDA layer's two
    state tensors. The conv state is always fp32 (the fused conv1d kernel
    needs it), regardless of the model dtype or `mamba_cache_dtype`; the
    recurrent state follows vLLM's usual KDA rule."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    conv_dtype, recurrent_dtype = kimi_attention.kda_state_dtype(vllm_config)
    assert conv_dtype == torch.float32
    assert recurrent_dtype == torch.float32


def test_load_a_log_flattens_legacy_layout_and_shards_by_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_load_a_log` is the weight loader for the KDA `A_log` parameter.
    Older checkpoints store it as `[1, 1, H, 1]` and newer ones as `[H]`;
    either way each TP rank keeps only its own slice of the H heads. With
    H=4 and a 2-head local parameter, rank 1 gets heads 2 and 3."""
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_rank", lambda: 1)
    param = torch.empty(2)
    loaded = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)

    kimi_attention._load_a_log(param, loaded)

    torch.testing.assert_close(param, torch.tensor([2.0, 3.0]))


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


def test_kda_forward_dispatches_to_both_custom_ops(monkeypatch) -> None:
    group = SimpleNamespace(all_reduce=lambda x: x)
    monkeypatch.setattr(kimi_attention, "get_tp_group", lambda: group)

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
    KimiDeltaAttention.process_weights_after_loading(layer, act_dtype=torch.bfloat16)
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
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
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

    layer.dispatched_kda_op = dispatched_kda_op

    output = layer(torch.arange(2), torch.ones(2, 4))
    assert calls == ["dispatched"]
    torch.testing.assert_close(
        output,
        torch.tensor([[0, 1, 2, 3], [14, 15, 16, 17]], dtype=torch.float32),
    )


def test_kda_forward_dispatches_to_pooled_op_on_singleton_cache(monkeypatch) -> None:
    """A one-tensor kv_cache means the unified pool: the layer must take the
    pooled op and never the dense dispatched op."""
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.fused_qkvb_proj = _Projection(14)
    layer.fused_fa_ga_proj = _Projection(4)
    layer.f_a_proj = None
    layer.g_proj = None
    layer.local_projection_size = 4
    layer.f_b_proj = _Projection(4)
    layer.g_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    layer.conv_size, layer.num_heads, layer.head_dim = 3, 2, 2
    KimiDeltaAttention.process_weights_after_loading(layer, act_dtype=torch.bfloat16)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
    layer.use_naive_kda = False

    pool = torch.zeros(2, 4, 2, 8)
    layer.kv_cache = [pool]
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        mamba_slot_read_offsets=None,
    )
    layer._metadata = lambda: metadata

    calls: list[str] = []

    def pooled_kda_op(mixed_qkv, *args):
        calls.append("pooled")
        assert args[3] is pool
        return mixed_qkv[:, :4].view(-1, 2, 2)

    def unexpected(name):
        def op(*args, **kwargs):
            raise AssertionError(f"{name} must not run for the pooled cache")

        return op

    monkeypatch.setattr(
        kimi_attention, "get_tp_group", lambda: SimpleNamespace(all_reduce=lambda x: x)
    )
    layer.pooled_kda_op = pooled_kda_op
    layer.dispatched_kda_op = unexpected("dispatched_kda_op")
    layer.sconv_op = unexpected("sconv_op")
    layer.chunk_kda_op = unexpected("chunk_kda_op")
    layer.kda_op = unexpected("kda_op")

    output = layer(torch.arange(2), torch.ones(2, 4))
    assert calls == ["pooled"]
    torch.testing.assert_close(
        output,
        torch.tensor([[0, 1, 2, 3], [14, 15, 16, 17]], dtype=torch.float32),
    )


def _kda_config(**linear_attn_overrides) -> KimiLinearConfig:
    """A small, valid Kimi K3 config with one KDA layer. Tests override
    fields of `linear_attn_config` to hit a specific branch or rejection in
    `KimiDeltaAttention.__init__`."""
    linear_attn_config = dict(
        head_dim=4,
        num_heads=2,
        short_conv_kernel_size=3,
        gate_lower_bound=None,
        use_full_rank_gate=False,
        kda_layers=[1],
        full_attn_layers=[],
    )
    linear_attn_config.update(linear_attn_overrides)
    return KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        linear_attn_config=linear_attn_config,
    )


def _patch_kda_construction_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap out the heavy dependencies of `KimiDeltaAttention.__init__`.

    The real constructor builds vLLM tensor-parallel linear layers (which
    need an initialized distributed group) and registers two Pallas custom
    ops (which register with `torch.library` by name). Neither is what a
    constructor test is checking, so both are replaced: the linear layers
    with a minimal module that just records its output size, and the op
    builders with a stub that returns a no-op callable. This is the same
    approach the `MultiHeadLatentAttention` constructor tests above use.
    """

    class FakeLinear(nn.Module):
        def __init__(self, input_size, output_size, *args, **kwargs) -> None:
            super().__init__()
            del args
            del kwargs
            self.output_size = (
                sum(output_size) if isinstance(output_size, list) else output_size
            )
            self.weight = nn.Parameter(torch.empty(self.output_size, input_size))

    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(kimi_attention, name, FakeLinear)
    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(kimi_attention, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        kimi_attention,
        "build_kimi_dispatched_kda_op",
        lambda *a, **k: (lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        kimi_attention,
        "build_kimi_pooled_kda_op",
        lambda *a, **k: (lambda *args, **kwargs: None),
    )


def test_kda_construction_wires_projections_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real `KimiDeltaAttention.__init__` end to end for the default
    (fused-gate) config and check what it built: the local head count after
    TP, the fused gate projection in place of the separate ones, the per-layer
    state shapes the cache manager will allocate, and that the layer registered
    itself under its prefix in the static forward context.

    The existing forward tests below skip `__init__` with `__new__`, so this is
    the only test that runs the constructor itself."""
    _patch_kda_construction_deps(monkeypatch)
    config = _kda_config()
    vllm_config = VllmConfig()

    with set_current_vllm_config(vllm_config):
        layer = kimi_attention.KimiDeltaAttention(
            config, vllm_config, prefix="model.layers.0.kda"
        )

    assert layer.num_heads == 2
    assert layer.head_dim == 4
    assert layer.fused_fa_ga_proj is not None
    assert layer.f_a_proj is None
    assert layer.g_proj is None
    assert layer.get_state_shape() == ((2, 3, 2, 4), (2, 4, 4))
    assert (
        vllm_config.compilation_config.static_forward_context["model.layers.0.kda"]
        is layer
    )


def test_kda_construction_full_rank_gate_builds_separate_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `use_full_rank_gate=True` the layer builds separate `f_a_proj`
    and `g_proj` projections instead of the fused `fused_fa_ga_proj`, and
    drops `g_b_proj`. This is the other branch of the constructor."""
    _patch_kda_construction_deps(monkeypatch)
    config = _kda_config(use_full_rank_gate=True, gate_lower_bound=1.0)
    vllm_config = VllmConfig()

    with set_current_vllm_config(vllm_config):
        layer = kimi_attention.KimiDeltaAttention(
            config, vllm_config, prefix="model.layers.0.kda"
        )

    assert layer.f_a_proj is not None
    assert layer.g_proj is not None
    assert layer.fused_fa_ga_proj is None
    assert layer.g_b_proj is None


def test_kda_construction_rejects_duplicate_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each layer registers itself in the static forward context under its
    prefix. Building a second layer with the same prefix would silently
    overwrite the first, so it is rejected."""
    _patch_kda_construction_deps(monkeypatch)
    config = _kda_config()
    vllm_config = VllmConfig()

    with set_current_vllm_config(vllm_config):
        kimi_attention.KimiDeltaAttention(
            config, vllm_config, prefix="model.layers.0.kda"
        )
        with pytest.raises(ValueError, match="Duplicate layer name"):
            kimi_attention.KimiDeltaAttention(
                config, vllm_config, prefix="model.layers.0.kda"
            )


@pytest.mark.parametrize("num_spec_tokens", [0, 5, 7])
def test_kda_construction_passes_speculative_window_to_op(
    monkeypatch: pytest.MonkeyPatch, num_spec_tokens: int
) -> None:
    """Ordinary decode and speculative decode pass their window to the op."""
    _patch_kda_construction_deps(monkeypatch)
    config = _kda_config()
    vllm_config = VllmConfig()
    vllm_config.speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_spec_tokens)
        if num_spec_tokens
        else None
    )
    received = {}

    def build_op(prefix, **kwargs):
        received.update(kwargs)
        return lambda *args, **kwargs: None

    monkeypatch.setattr(kimi_attention, "build_kimi_dispatched_kda_op", build_op)
    with set_current_vllm_config(vllm_config):
        layer = kimi_attention.KimiDeltaAttention(
            config, vllm_config, prefix="model.layers.0.kda"
        )

    assert layer.num_spec_tokens == num_spec_tokens
    assert received["num_spec_tokens"] == num_spec_tokens


def test_kda_construction_rejects_missing_linear_attn_config() -> None:
    """A Kimi config without `linear_attn_config` has no KDA layers to
    describe, so the KDA layer cannot be built from it."""
    config = KimiLinearConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        linear_attn_config=None,
    )
    fake_vllm_config = SimpleNamespace(speculative_config=None)

    with pytest.raises(ValueError, match="requires linear_attn_config"):
        kimi_attention.KimiDeltaAttention(config, fake_vllm_config, prefix="y")


def test_kda_construction_rejects_heads_not_divisible_by_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KDA heads are split evenly across tensor-parallel ranks, so 2 heads
    cannot be spread over 3 ranks."""
    monkeypatch.setattr(
        kimi_attention, "get_tensor_model_parallel_world_size", lambda: 3
    )
    config = _kda_config(num_heads=2)
    fake_vllm_config = SimpleNamespace(speculative_config=None)

    with pytest.raises(ValueError, match="divisible by TP size"):
        kimi_attention.KimiDeltaAttention(config, fake_vllm_config, prefix="z")


class _ConstantProjection(nn.Module):
    """Returns a tensor filled with one value, so a test can tell which
    projection produced a given argument."""

    def __init__(self, value: float, output_size: int) -> None:
        super().__init__()
        self.value = value
        self.output_size = output_size

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return inputs.new_full((inputs.shape[0], self.output_size), self.value), None


def _kda_metadata() -> SimpleNamespace:
    """Minimal per-layer metadata for a one-sequence, two-token step."""
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
    )


def _failing_op(name: str):
    """A KDA op stand-in that fails the test if `forward` ever calls it."""

    def op(*args, **kwargs):
        raise AssertionError(f"{name} must not run on this path")

    return op


def _make_bare_kda_layer(use_full_rank_gate: bool = False):
    """Build a `KimiDeltaAttention` without running `__init__`.

    Same approach as the forward-dispatch tests above: create the object
    with `__new__` and set just the attributes `forward` reads, using small
    stand-in projections. `use_full_rank_gate` picks which of the two gate
    layouts to install.

    The layer comes back with valid metadata, no cache bound, and both KDA
    ops set to fail the test if called. That way a test that expects the
    zeros path proves it got there because of the cache check, and a test
    that expects a dispatch has to bind a cache and install a real stub.
    """
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.fused_qkvb_proj = _Projection(14)
    layer.local_projection_size = 4
    layer.conv_size, layer.num_heads, layer.head_dim = 3, 2, 2
    layer.f_b_proj = _Projection(4)
    layer.o_proj = _IdentityProjection()
    layer.q_conv1d = _ConvWeight()
    layer.k_conv1d = _ConvWeight()
    layer.v_conv1d = _ConvWeight()
    KimiDeltaAttention.process_weights_after_loading(layer, act_dtype=torch.bfloat16)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = use_full_rank_gate
    if use_full_rank_gate:
        layer.f_a_proj = _Projection(2)
        layer.g_proj = _ConstantProjection(7.0, 4)
        layer.fused_fa_ga_proj = None
        layer.g_b_proj = None
    else:
        layer.fused_fa_ga_proj = _Projection(4)
        layer.f_a_proj = None
        layer.g_proj = None
        layer.g_b_proj = _Projection(4)
    layer._metadata = _kda_metadata
    layer.kv_cache = None
    layer.dispatched_kda_op = _failing_op("dispatched_kda_op")
    layer.pooled_kda_op = _failing_op("pooled_kda_op")
    return layer


@pytest.mark.parametrize(
    "setup",
    ["no_metadata", "no_cache", "empty_cache"],
)
def test_kda_forward_returns_zeros_when_it_cannot_run_the_op(
    setup: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`forward` has three reasons to skip the KDA op and return zeros: no
    per-layer metadata, no cache bound yet, or a cache bound with no
    storage (what the runner hands out during profiling). Each case is set
    up so only that one condition holds, and both ops are tripwires, so the
    test fails if `forward` reaches a dispatch.
    """
    monkeypatch.setattr(
        kimi_attention, "get_tp_group", lambda: SimpleNamespace(all_reduce=lambda x: x)
    )
    layer = _make_bare_kda_layer()
    if setup == "no_metadata":
        layer._metadata = lambda: None
        layer.kv_cache = (torch.zeros(1, 8, 3), torch.zeros(1, 2, 2, 2))
    elif setup == "no_cache":
        layer.kv_cache = None
    else:
        layer.kv_cache = (torch.empty(0), torch.empty(0))

    output = layer(torch.arange(2), torch.ones(2, 4))

    torch.testing.assert_close(output, torch.zeros(2, 4))


def test_kda_forward_full_rank_gate_feeds_separate_gates_to_the_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `use_full_rank_gate=True`, `forward` computes the output gate
    from `g_proj` directly (not from a `g_b_proj` over a split of a fused
    projection) and the recurrence gate from `f_b_proj(f_a_proj(x))`. A
    two-tensor cache is bound so `forward` reaches the dispatched op, and
    the op stub checks it was handed exactly those two tensors."""
    monkeypatch.setattr(
        kimi_attention, "get_tp_group", lambda: SimpleNamespace(all_reduce=lambda x: x)
    )
    layer = _make_bare_kda_layer(use_full_rank_gate=True)
    metadata = layer._metadata()
    layer._metadata = lambda: metadata
    sconv_cache = torch.zeros(1, 8, 3)
    recurrent_cache = torch.zeros(1, 2, 2, 2)
    layer.kv_cache = (sconv_cache, recurrent_cache)
    hidden_states = torch.ones(2, 4)
    received: dict[str, torch.Tensor] = {}

    def dispatched_kda_op(mixed_qkv, raw_gate, beta, output_gate, *args):
        received["raw_gate"] = raw_gate
        received["output_gate"] = output_gate
        assert args[0] is sconv_cache
        assert args[1] is recurrent_cache
        assert args[-2] is metadata.request_distribution
        assert args[-1] is metadata.mamba_state_indices
        return mixed_qkv[:, :4].view(-1, 2, 2)

    layer.dispatched_kda_op = dispatched_kda_op

    layer(torch.arange(2), hidden_states)

    # g_proj is the constant-7 stub; nothing else in the layer produces 7s.
    torch.testing.assert_close(received["output_gate"], torch.full((2, 4), 7.0))
    f_a, _ = layer.f_a_proj(hidden_states)
    expected_raw_gate, _ = layer.f_b_proj(f_a)
    torch.testing.assert_close(received["raw_gate"], expected_raw_gate)


def test_metadata_returns_none_without_a_forward_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_metadata` returns `None` when there is no active forward context at
    all, which is the case outside of a model forward pass."""
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.prefix = "p"
    monkeypatch.setattr(kimi_attention, "is_forward_context_available", lambda: False)

    assert layer._metadata() is None


def test_metadata_returns_none_for_a_non_dict_attn_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_metadata` also returns `None` when a forward context exists but its
    `attn_metadata` is not the per-layer dict the runner normally provides."""
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.prefix = "p"
    monkeypatch.setattr(kimi_attention, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        kimi_attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata="not-a-dict"),
    )

    assert layer._metadata() is None


def test_metadata_rejects_wrong_layer_metadata_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the per-layer entry exists but is not an `AttentionMetadata`, the
    layer raises rather than trying to read fields off an unknown object."""
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.prefix = "p"
    monkeypatch.setattr(kimi_attention, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        kimi_attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"p": "wrong-type"}),
    )

    with pytest.raises(TypeError, match="incompatible attention metadata"):
        layer._metadata()


def test_core_attention_pooled_rejects_pcp_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unified-pool KDA path does not support prefill-context-parallel
    streaming metadata and says so up front."""
    layer = _make_bare_kda_layer()
    monkeypatch.setattr(
        kimi_attention, "is_pcp_streaming_attention_metadata", lambda md: True
    )
    metadata = SimpleNamespace(mamba_slot_read_offsets=None)

    with pytest.raises(NotImplementedError, match="PCP streaming prefill"):
        layer._core_attention_pooled(None, None, None, None, None, metadata)


def test_core_attention_pooled_rejects_speculative_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mamba_slot_read_offsets` is only set for speculative-decode verify
    steps, which the pooled path cannot handle (it keeps one state per block,
    not the per-window checkpoints verify needs)."""
    layer = _make_bare_kda_layer()
    monkeypatch.setattr(
        kimi_attention, "is_pcp_streaming_attention_metadata", lambda md: False
    )
    metadata = SimpleNamespace(mamba_slot_read_offsets=torch.zeros(1))

    with pytest.raises(NotImplementedError, match="Speculative decoding"):
        layer._core_attention_pooled(None, None, None, None, None, metadata)


def test_core_attention_pooled_requires_state_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pooled op gathers and scatters state by `mamba_state_indices`;
    metadata without them is an error, not a silent no-op."""
    layer = _make_bare_kda_layer()
    monkeypatch.setattr(
        kimi_attention, "is_pcp_streaming_attention_metadata", lambda md: False
    )
    metadata = SimpleNamespace(mamba_slot_read_offsets=None, mamba_state_indices=None)

    with pytest.raises(RuntimeError, match="requires mamba_state_indices"):
        layer._core_attention_pooled(None, None, None, None, None, metadata)


def test_kda_custom_ops_compile_as_one_full_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = SimpleNamespace(all_reduce=lambda x: x)
    monkeypatch.setattr(kimi_attention, "get_tp_group", lambda: group)

    def jax_op(name, function, donate_argnums=()):
        del function, donate_argnums
        assert "dispatched" in name

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
            window_distribution: torch.Tensor,
            slot_read_offsets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del raw_gate, beta, output_gate, a_log, dt_bias, norm_weight
            del conv_weight, distribution, window_distribution
            del slot_read_offsets
            del query_start_loc, state_indices, seq_lens
            num_heads, head_dim = recurrent_state.shape[1:3]
            output = mixed_qkv[:, : num_heads * head_dim]
            return (
                output.view(-1, num_heads, head_dim).clone(),
                conv_state + 1,
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
    KimiDeltaAttention.process_weights_after_loading(layer, act_dtype=torch.bfloat16)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2))
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(4))
    layer.use_full_rank_gate = False
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
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
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
    torch.testing.assert_close(recurrent_cache, torch.ones_like(recurrent_cache))


@pytest.mark.parametrize("num_blocks", [0, 8], ids=["prefix-only", "with-blocks"])
def test_attention_residual_matches_reference(num_blocks: int) -> None:
    """The layer computes a softmax-weighted sum over the residual slots:
    RMS-normalize each slot, project to a scalar score, softmax over slots,
    mix. The reference below states that directly in fp32.
    """
    torch.manual_seed(0)
    num_tokens, hidden, eps = 33, 64, 1e-5
    # Stubbed like the other layer tests here: RMSNorm/ReplicatedLinear are
    # CustomOps whose constructors need a VllmConfig context, and constructing
    # one is exactly what the TPU device parse makes impossible off-TPU. The
    # fold only reads the two weights, so stub them with plain Parameters.
    module = AttentionResidual.__new__(AttentionResidual)
    nn.Module.__init__(module)
    module.eps = eps
    module.norm = SimpleNamespace(
        weight=nn.Parameter(torch.randn(hidden, dtype=torch.bfloat16) * 0.5 + 1)
    )
    module.proj = SimpleNamespace(
        weight=nn.Parameter(torch.randn(1, hidden, dtype=torch.bfloat16) * 0.1)
    )
    module.process_weights_after_loading()

    prefix_sum = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    block_residuals = torch.randn(num_tokens, num_blocks, hidden, dtype=torch.bfloat16)

    values = torch.cat((block_residuals, prefix_sum.unsqueeze(-2)), dim=-2).float()
    normed = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    scores = normed @ (module.norm.weight.float() * module.proj.weight[0].float())
    probabilities = scores.softmax(dim=-1)
    expected = (probabilities.unsqueeze(-1) * values).sum(dim=-2).to(prefix_sum.dtype)

    actual = module(prefix_sum, block_residuals)

    assert actual.shape == (num_tokens, hidden)
    assert actual.dtype == prefix_sum.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def _attention_residual_stub(hidden: int, eps: float) -> AttentionResidual:
    # Same stubbing as test_attention_residual_matches_reference: the fold
    # only reads the two weights.
    module = AttentionResidual.__new__(AttentionResidual)
    nn.Module.__init__(module)
    module.eps = eps
    module.norm = SimpleNamespace(
        weight=nn.Parameter(torch.randn(hidden, dtype=torch.bfloat16) * 0.5 + 1)
    )
    module.proj = SimpleNamespace(
        weight=nn.Parameter(torch.randn(1, hidden, dtype=torch.bfloat16) * 0.1)
    )
    module.process_weights_after_loading()
    return module


def test_attention_residual_single_token_matches_batched_rows() -> None:
    """One token mixes the same way whether it is alone or in a batch."""
    torch.manual_seed(0)
    module = _attention_residual_stub(hidden=64, eps=1e-5)
    prefix_sum = torch.randn(5, 64, dtype=torch.bfloat16)
    block_residuals = torch.randn(5, 8, 64, dtype=torch.bfloat16)

    batched = module(prefix_sum, block_residuals)
    for row in range(5):
        alone = module(prefix_sum[row : row + 1], block_residuals[row : row + 1])
        torch.testing.assert_close(alone, batched[row : row + 1])


def test_attention_residual_single_token_trace_is_its_own_bucket() -> None:
    """The one-token structure must not leak into other token buckets.

    The layer decides its formulation from the token count, so the trace made
    for one token has to carry a guard that refuses every other compile size;
    shape_variants then gives those buckets their own trace.
    """
    torch.manual_seed(0)
    module = _attention_residual_stub(hidden=64, eps=1e-5)
    graphs = []

    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    prefix_sum = torch.randn(1, 64, dtype=torch.bfloat16)
    block_residuals = torch.randn(1, 8, 64, dtype=torch.bfloat16)
    mark_dynamic(prefix_sum, 0)
    mark_dynamic(block_residuals, 0)
    torch._dynamo.reset()
    # The TPU platform switches to size-oblivious shapes when compile_sizes
    # has 1; that is the mode the bucket-1 trace runs under.
    with fx_config.patch(backed_size_oblivious=True):
        compiled = torch.compile(module, backend=capture, fullgraph=True, dynamic=False)
        compiled(prefix_sum, block_residuals)

    shape_env = trace_shape_env(graphs[0])
    assert unsupported_reason(shape_env, 1) is None
    assert unsupported_reason(shape_env, 2) is not None
    assert unsupported_reason(shape_env, 512) is not None


@pytest.mark.parametrize(
    "field",
    [
        "mla_use_output_gate",
        "activation_situ_beta",
        "activation_situ_linear_beta",
        "attn_res_block_size",
    ],
)
def test_kimi_model_requires_declared_optional_config_fields(field):
    config = KimiLinearConfig()
    delattr(config, field)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config))
    with pytest.raises(AttributeError, match=field):
        KimiModel(vllm_config=vllm_config)


@pytest.mark.parametrize("use_norm", [False, True])
def test_latent_moe_norm_follows_the_config_flag(
    monkeypatch: pytest.MonkeyPatch, use_norm: bool
) -> None:
    class Fake(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(kimi_moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kimi_moe, "GateLinear", Fake)
    monkeypatch.setattr(kimi_moe, "KimiMLP", Fake)
    monkeypatch.setattr(kimi_moe, "make_latent_projection", Fake)
    monkeypatch.setattr(kimi_moe, "FusedMoEFactory", lambda **kwargs: Fake())
    norm_args: list[tuple] = []

    class FakeNorm(nn.Module):
        def __init__(self, *args) -> None:
            super().__init__()
            norm_args.append(args)

    monkeypatch.setattr(kimi_moe, "RMSNorm", FakeNorm)
    config = KimiLinearConfig(
        hidden_size=16,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=32,
        routed_expert_hidden_size=8,
        latent_moe_use_norm=use_norm,
    )
    layer = kimi_moe.KimiMoE(config, quant_config=None, prefix="moe")

    assert isinstance(layer.routed_expert_down_proj, Fake)
    assert isinstance(layer.routed_expert_up_proj, Fake)
    if use_norm:
        assert isinstance(layer.routed_expert_norm, FakeNorm)
        assert norm_args == [(8, config.rms_norm_eps)]
    else:
        assert layer.routed_expert_norm is None
        assert norm_args == []


@pytest.mark.parametrize("attn_res_block_size", [None, 4])
def test_kimi_model_wires_declared_optional_features(
    monkeypatch: pytest.MonkeyPatch, attn_res_block_size: int | None
) -> None:
    class Recorder(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    for name in (
        "MultiHeadLatentAttention",
        "KimiDeltaAttention",
        "KimiMoE",
        "KimiMLP",
        "RMSNorm",
        "AttentionResidual",
        "VocabParallelEmbedding",
    ):
        monkeypatch.setattr(kimi_model, name, Recorder)
    config = KimiLinearConfig(
        num_hidden_layers=2,
        hidden_act="situ",
        activation_situ_beta=1.5,
        activation_situ_linear_beta=0.5,
        attn_res_block_size=attn_res_block_size,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    model = kimi_model.KimiModel(vllm_config=vllm_config)

    uses_residual_blocks = attn_res_block_size is not None
    assert model.attn_res_block_size == attn_res_block_size
    assert hasattr(model, "output_attn_res") is uses_residual_blocks
    assert len(model.layers) == 2
    for layer in model.layers:
        assert isinstance(layer.self_attn, Recorder)
        assert layer.attn_res_block_size == attn_res_block_size
        assert hasattr(layer, "self_attention_res") is uses_residual_blocks
        assert hasattr(layer, "mlp_res") is uses_residual_blocks
        assert layer.mlp.kwargs["situ_beta"] == 1.5
        assert layer.mlp.kwargs["situ_linear_beta"] == 0.5


@pytest.mark.parametrize("transposed", [False, True])
def test_dummy_router_is_fixed_random_and_layer_specific(monkeypatch, transposed):
    config = SimpleNamespace(load_config=SimpleNamespace(load_format="dummy"))
    monkeypatch.setattr(kimi_moe, "get_current_vllm_config_or_none", lambda: config)
    module = kimi_moe.KimiMoE.__new__(kimi_moe.KimiMoE)
    nn.Module.__init__(module)
    module.prefix = "model.layers.1.block_sparse_moe"
    module.gate = nn.Module()
    module.gate.input_size = 128
    module.gate.e_score_correction_bias = nn.Parameter(torch.zeros(896))
    shape = (128, 896) if transposed else (896, 128)
    module.gate.weight = nn.Parameter(torch.zeros(shape))
    module.initialize_dummy_router()
    first = module.gate.e_score_correction_bias.detach().clone()
    first_weight = module.gate.weight.detach().clone()
    assert 0.015 < first.std().item() < 0.025
    assert 0.95 < first_weight.std().item() * 128**0.5 < 1.05
    assert torch.all(first != 0)
    module.initialize_dummy_router()
    torch.testing.assert_close(
        first, module.gate.e_score_correction_bias, rtol=0, atol=0
    )
    torch.testing.assert_close(first_weight, module.gate.weight, rtol=0, atol=0)
    module.prefix = "model.layers.2.block_sparse_moe"
    module.initialize_dummy_router()
    assert not torch.equal(first, module.gate.e_score_correction_bias)
    assert not torch.equal(first_weight, module.gate.weight)
    checkpoint_bias = module.gate.e_score_correction_bias.detach().clone()
    checkpoint_weight = module.gate.weight.detach().clone()
    config.load_config.load_format = "safetensors"
    module.initialize_dummy_router()
    torch.testing.assert_close(
        checkpoint_bias, module.gate.e_score_correction_bias, rtol=0, atol=0
    )
    torch.testing.assert_close(checkpoint_weight, module.gate.weight, rtol=0, atol=0)


def test_attention_residual_bridge_fake_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.models.vllm.kimi_k3.layers import _build_attention_residual_op

    op = _build_attention_residual_op(1e-5)
    assert op is _build_attention_residual_op(1e-5)
    with FakeTensorMode():
        prefix = torch.empty(256, 7168, dtype=torch.bfloat16)
        history = torch.empty(256, 8, 7168, dtype=torch.bfloat16)
        weight = torch.empty(7168, dtype=torch.float32)
        output = op(prefix, history, weight)
        assert output.shape == prefix.shape
        assert output.dtype == prefix.dtype


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_chip_shared_expert_matches_full_projection(sequence_parallel):
    """Pair TP reproduces a full FFN and returns the owner's token slice."""
    from vllm_torchtpu.models.vllm.kimi_k3.layers import KimiMLP

    torch.manual_seed(73)
    x = torch.randn(8, 4)
    gate = torch.randn(6, 4)
    up = torch.randn(6, 4)
    down = torch.randn(4, 6)
    contributions = []
    for rank in range(2):
        g = x @ gate.chunk(2)[rank].T
        u = x @ up.chunk(2)[rank].T
        contributions.append(
            (torch.nn.functional.silu(g) * u) @ down.chunk(2, dim=1)[rank].T
        )
    reference = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T

    def check_rank(rank):
        class Group:
            def all_gather(self, local, dim):
                assert dim == 0
                torch.testing.assert_close(local, x.chunk(2)[rank])
                return x

            def all_reduce(self, partial):
                torch.testing.assert_close(partial, contributions[rank])
                return partial + contributions[1 - rank]

            def reduce_scatter(self, partial, dim):
                assert dim == 0
                return self.all_reduce(partial).chunk(2)[rank]

        class Projection(nn.Module):
            def forward(self, value):
                return torch.cat(
                    (value @ gate.chunk(2)[rank].T, value @ up.chunk(2)[rank].T), dim=-1
                ), None

        module = KimiMLP.__new__(KimiMLP)
        nn.Module.__init__(module)
        module.tp_group = Group()
        module.sp_group = None
        module.gate_up_proj = Projection()
        module.act_fn = lambda value: (
            torch.nn.functional.silu(value.chunk(2, -1)[0]) * value.chunk(2, -1)[1]
        )
        module.down_proj = SimpleNamespace(
            quant_method=SimpleNamespace(
                apply=lambda layer, value, bias: value @ down.chunk(2, 1)[rank].T
            )
        )
        actual = module(
            x.chunk(2)[rank] if sequence_parallel else x,
            sequence_parallel=sequence_parallel,
        )
        expected = reference.chunk(2)[rank] if sequence_parallel else reference
        torch.testing.assert_close(actual, expected)

    for rank in range(2):
        check_rank(rank)


@pytest.mark.parametrize("gather_mode", ["plain", "joint", "wide_ids"])
def test_sp_moe_reduces_latent_before_norm_and_up_projection(monkeypatch, gather_mode):
    from vllm_torchtpu.models.vllm.kimi_k3 import moe as module

    calls = []

    class Group:
        def all_gather(self, x, dim):
            calls.append(("gather", x.shape, x.dtype))
            return x.repeat(32, 1)

        def reduce_scatter(self, x, dim):
            calls.append(("scatter", x.shape, x.dtype))
            return x[:1] * 32

    class Projection(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x), None

    def experts(layer, latent, weights, ids):
        assert latent.shape == (32, 2)
        torch.testing.assert_close(
            weights, torch.full((32, 2), 0.5, dtype=latent.dtype)
        )
        assert ids.dtype == torch.int32
        torch.testing.assert_close(
            ids, torch.tensor([[3, 9]], dtype=torch.int32).repeat(32, 1)
        )
        return latent * 2

    model = module.KimiMoE.__new__(module.KimiMoE)
    nn.Module.__init__(model)
    group = module.FusedPrefillCollectives.__new__(module.FusedPrefillCollectives)
    group.all_gather = Group().all_gather
    group.reduce_scatter = Group().reduce_scatter

    def gather_joint(latent, weights, ids):
        calls.append(("joint", latent.shape, latent.dtype))
        return tuple(a.repeat(32, 1) for a in (latent, weights, ids))

    group._moe_gather_op = gather_joint
    model.prefill_group = group
    model.gate = Projection(lambda x: torch.zeros(x.shape[0], 896))
    model.gate.output_size = 65536 if gather_mode == "wide_ids" else 896
    model.routed_expert_down_proj = Projection(lambda x: x[:, :2])
    model.routed_expert_norm = lambda x: x / x.square().mean(-1, keepdim=True).sqrt()
    model.routed_expert_up_proj = Projection(lambda x: x.repeat(1, 2))
    model.shared_experts = None
    model.experts = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(apply_with_routing=experts)
        )
    )
    monkeypatch.setattr(
        module.moe_routing,
        "route",
        lambda *args: (torch.full((1, 2), 0.5), torch.tensor([[3, 9]])),
    )
    dtype = torch.float32 if gather_mode == "plain" else torch.bfloat16
    x = torch.tensor([[1.0, 3.0, 4.0, 6.0]], dtype=dtype)
    actual = model._forward_expert_parallel(x, True)
    expected = (x[:, :2] / (x[:, :2].square().mean(-1, keepdim=True).sqrt())).repeat(
        1, 2
    )
    torch.testing.assert_close(actual, expected)
    expected_calls = (
        ["joint", "scatter"] if gather_mode == "joint" else ["gather"] * 3 + ["scatter"]
    )
    assert [entry[0] for entry in calls] == expected_calls
    assert calls[-1][1] == (32, 2)
    assert all(entry[1][-1] == 2 for entry in calls)


def test_sp_prefill_trace_accepts_dynamic_token_dimension():
    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import FusedPrefillCollectives
    from vllm_torchtpu.models.vllm.kimi_k3.model import KimiModel

    # Exercise the real model's token slicing and the collective shape guard,
    # without attention/weights or a TPU client. The compiler must retain a
    # divisibility guard without forcing mark_dynamic's token dimension
    # to the initial bucket size.
    group = FusedPrefillCollectives.__new__(FusedPrefillCollectives)
    group.ag = lambda x: x.repeat(32, 1)
    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    model.sp_prefill = True
    model.sp_group = SimpleNamespace(
        rank_in_group=0, world_size=32, all_gather=group.all_gather
    )
    model.aux_hidden_state_layers = ()
    model.attn_res_block_size = 12
    model.output_attn_res = _attention_residual_stub(hidden=3584, eps=1e-5)
    model.layers = nn.ModuleList()
    model.norm = nn.Identity()
    graphs = []

    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    hidden = torch.zeros(8192, 3584, dtype=torch.bfloat16)
    mark_dynamic(hidden, 0)
    torch._dynamo.reset()
    with fx_config.patch(backed_size_oblivious=True):
        fn = torch.compile(
            model.forward, backend=capture, fullgraph=True, dynamic=False
        )
        out = fn(None, torch.arange(8192), inputs_embeds=hidden)
    assert out.shape == hidden.shape
    env = trace_shape_env(graphs[0])
    assert unsupported_reason(env, 8192) is None
    for size in (2048, 4096, 16384):
        assert unsupported_reason(env, size) is None
    for size in (1, 8, 2049, 8191):
        assert unsupported_reason(env, size) is not None


@pytest.mark.parametrize("flipped", [False, True])
def test_attention_project_rs_preserves_linear_layout(flipped):
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import (
        FusedPrefillCollectives,
        _build_project_op,
    )

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    tables = np.zeros((3, 3, 32), np.int32)
    op = _build_project_op(mesh, tables, 29000)
    assert _build_project_op(mesh, tables, 29000) is op
    group = FusedPrefillCollectives.__new__(FusedPrefillCollectives)
    group.project_rs = op
    calls = []

    class Projection:
        bias = None

        def __call__(self, x):
            calls.append("linear")
            return x.new_empty((x.shape[0], 7168)), None

    def scatter(x, dim):
        calls.append("scatter")
        return x.new_empty((x.shape[0] // 32, x.shape[1]))

    group.reduce_scatter = scatter
    projection = Projection()
    setattr(projection, WEIGHT_FLIPPED_ATTR, flipped)
    with FakeTensorMode():
        projection.weight = torch.empty(
            (384, 7168) if flipped else (7168, 384), dtype=torch.bfloat16
        )
        x = torch.empty((8192, 384), dtype=torch.bfloat16)
        result = group.project_reduce_scatter(x, projection)
        assert result.shape == (256, 7168)
        assert result.dtype == torch.bfloat16
    assert calls == ([] if flipped else ["linear", "scatter"])


def test_kda_prefill_packing_preserves_projection_and_reload():
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import FusedPrefillCollectives

    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.prefill_group = FusedPrefillCollectives.__new__(FusedPrefillCollectives)
    layer.use_full_rank_gate = True
    layer.num_heads, layer.head_dim = 3, 128
    generator = torch.Generator().manual_seed(19)
    for name, width in [("fused_qkvb_proj", 1155), ("g_proj", 384), ("f_a_proj", 128)]:
        projection = nn.Module()
        projection.weight = nn.Parameter(
            torch.randn((7168, width), generator=generator, dtype=torch.bfloat16) * 0.01
        )
        setattr(projection, WEIGHT_FLIPPED_ATTR, True)
        setattr(layer, name, projection)
    x = torch.randn((2, 7168), generator=generator)
    for reload in (False, True):
        if reload:
            with torch.no_grad():
                layer.fused_qkvb_proj.weight.add_(0.02)
        layer._pack_prefill_projection()
        output = x @ layer.packed_prefill_weight.float()
        qkvb = x @ layer.fused_qkvb_proj.weight.float()
        torch.testing.assert_close(output[:, :1155], qkvb)
        assert torch.count_nonzero(output[:, 1155:1280]) == 0
        torch.testing.assert_close(
            output[:, 1280:1664], x @ layer.g_proj.weight.float()
        )
        torch.testing.assert_close(output[:, 1664:], x @ layer.f_a_proj.weight.float())
    layer.use_full_rank_gate = False
    layer._pack_prefill_projection()
    assert layer.packed_prefill_weight is None


def test_gather_projection_bridge_has_rank_local_output():
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import (
        _build_gather_project_op,
    )

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    tables = np.zeros((3, 3, 32), np.int32)
    op = _build_gather_project_op(mesh, tables, 29001)
    assert _build_gather_project_op(mesh, tables, 29001) is op
    with FakeTensorMode():
        result = op(
            torch.empty((256, 7168), dtype=torch.bfloat16),
            torch.empty((7168, 1792), dtype=torch.bfloat16),
        )
        assert result.shape == (8192, 1792)
        assert result.dtype == torch.bfloat16


def test_moe_gather_bridge_shapes_and_registration_cache():
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import _build_moe_gather_op

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    tables = np.zeros((3, 3, 32), np.int32)
    op = _build_moe_gather_op(mesh, tables, 29003)
    assert _build_moe_gather_op(mesh, tables, 29003) is op
    with FakeTensorMode():
        latent, weights, ids = op(
            torch.empty((256, 3584), dtype=torch.bfloat16),
            torch.empty((256, 16), dtype=torch.bfloat16),
            torch.empty((256, 16), dtype=torch.int32),
        )
        assert latent.shape == (8192, 3584)
        assert latent.dtype == torch.bfloat16
        assert weights.shape == ids.shape == (8192, 16)
        assert weights.dtype == torch.bfloat16
        assert ids.dtype == torch.int32


def _trace_input_signature(fn, args):
    from vllm_torchtpu.compilation.shape_variants import graph_signature

    graphs = []

    def capture(graph, _inputs):
        graphs.append(graph)
        # Inspect the real Dynamo graph without executing its TPU custom ops.
        return lambda *_args: (torch.empty(0),)

    for arg in args:
        mark_dynamic(arg, 0)
    torch._dynamo.reset()
    with fx_config.patch(backed_size_oblivious=True):
        torch.compile(fn, backend=capture, fullgraph=True, dynamic=False)(*args)
    assert len(graphs) == 1
    return graph_signature(graphs[0])


def test_packed_kda_keeps_parameter_abi_across_token_buckets(monkeypatch):
    import jax
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import (
        FusedPrefillCollectives,
        _build_gather_project_op,
        _build_project_op,
    )

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    tables = np.zeros((3, 3, 32), np.int32)
    group = FusedPrefillCollectives.__new__(FusedPrefillCollectives)
    group.gather_project = _build_gather_project_op(mesh, tables, 29002)
    group.project_rs = _build_project_op(mesh, tables, 29003)
    group.base = SimpleNamespace(all_reduce=lambda x: x)
    monkeypatch.setattr(kimi_attention, "get_tp_group", lambda: group)

    class Projection(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(inputs, outputs, dtype=torch.bfloat16)
            )
            self.bias = None
            setattr(self, WEIGHT_FLIPPED_ATTR, True)

        def forward(self, x):
            return x @ self.weight, None

    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.prefill_group = group
    layer.use_full_rank_gate = True
    layer.num_heads, layer.head_dim, layer.local_projection_size = 3, 128, 384
    layer.register_buffer(
        "packed_prefill_weight", torch.empty(7168, 1792, dtype=torch.bfloat16)
    )
    layer.f_b_proj = Projection(128, 384)
    layer.o_proj = Projection(384, 7168)
    layer._metadata = lambda: None
    layer.kv_cache = None

    def forward(x):
        sp = x.shape[0] >= 8192
        if sp:
            x = x.narrow(0, 0, 256)
        return layer(None, x, sequence_parallel=sp)

    signatures = [
        _trace_input_signature(forward, (torch.empty(n, 7168, dtype=torch.bfloat16),))
        for n in (2048, 8192)
    ]
    assert signatures[0] == signatures[1]


def test_mla_keeps_weight_cache_abi_across_token_buckets(monkeypatch):
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla

    @torch.library.custom_op("kimi_abi_test::latent_attention", mutates_args=())
    def latent_attention(
        cache: torch.Tensor,
        q: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kr: torch.Tensor,
        lens: torch.Tensor,
        blocks: torch.Tensor,
        starts: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        return q.clone()

    latent_attention.register_fake(lambda cache, q, *args: torch.empty_like(q))
    layer = mla.VllmTPUMLAAttention.__new__(mla.VllmTPUMLAAttention)
    nn.Module.__init__(layer)
    layer.layer_name = "test_mla_parameter_abi"
    layer.calculate_kv_scales = False
    layer.num_heads, layer.qk_nope_head_dim = 3, 128
    layer.qk_rope_head_dim, layer.kv_lora_rank, layer.v_head_dim = 64, 512, 128
    layer.scale = 192**-0.5
    layer.kv_cache_quantized_dtype = None
    layer._q_scale_float = layer._k_scale_float = layer._v_scale_float = 1.0
    layer.W_UK_T = nn.Parameter(torch.empty(3, 128, 512, dtype=torch.bfloat16))
    layer.W_UV = nn.Parameter(torch.empty(3, 512, 128, dtype=torch.bfloat16))
    layer.impl = PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)
    layer.mla_op = latent_attention
    cache = torch.empty((64, 128, 2, 640), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        seq_lens=torch.zeros(8, dtype=torch.int32),
        block_tables=torch.zeros(320, dtype=torch.int32),
        query_start_loc=torch.zeros(9, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    monkeypatch.setattr(
        mla, "get_attention_context", lambda name: (metadata, None, cache, None)
    )

    def forward(x):
        # The production wrapper computes these activations before entering
        # attention. Keep them internal to the trace, as in the full model.
        q, qr, kv, kr = x.split([384, 192, 512, 64], dim=-1)
        return layer(
            (q.reshape(-1, 3, 128), qr.reshape(-1, 3, 64)), kv, kr.reshape(-1, 1, 64)
        )

    signatures = [
        _trace_input_signature(forward, (torch.empty(n, 1152, dtype=torch.bfloat16),))
        for n in (2048, 8192)
    ]
    assert signatures[0] == signatures[1]


def test_mla_preprocess_keeps_prefill_parameter_abi(monkeypatch):
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.models.vllm.kimi_k3 import collective_ops

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    pairs = tuple((2 * i, 2 * i + 1) for i in range(16))
    monkeypatch.setattr(
        collective_ops, "mla_chip_layout", lambda devices, owners: (pairs, (), ())
    )
    pack, op = collective_ops._build_mla_ops(mesh, tuple(range(32)), 29005, 1e-5)
    assert collective_ops._build_mla_ops(mesh, tuple(range(32)), 29005, 1e-5) == (
        pack,
        op,
    )
    with FakeTensorMode():
        weight = torch.empty(7168, 2496, dtype=torch.bfloat16)
        gate = pack(weight)
        assert gate.shape == (7168, 6144)
        for sp, rows in ((True, 256), (False, 2048), (False, 1)):
            out = op(
                torch.empty(rows, 7168, dtype=torch.bfloat16),
                weight,
                gate,
                torch.empty(1536, dtype=torch.bfloat16),
                torch.empty(512, dtype=torch.bfloat16),
                sp,
            )
            expected_rows = rows * 32 if sp else rows
            assert [x.shape for x in out] == [
                (expected_rows, n) for n in (1536, 512, 64, 384)
            ]

    class Preprocess(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(7168, 2496, dtype=torch.bfloat16))
            self.register_buffer("gate", torch.empty(7168, 6144, dtype=torch.bfloat16))
            self.qnorm = nn.Parameter(torch.empty(1536, dtype=torch.bfloat16))
            self.kvnorm = nn.Parameter(torch.empty(512, dtype=torch.bfloat16))

        def forward(self, x):
            sp = x.shape[0] >= 8192
            if sp:
                x = x.narrow(0, 0, 256)
            return op(x, self.weight, self.gate, self.qnorm, self.kvnorm, sp)[0]

    model = Preprocess()
    signatures = [
        _trace_input_signature(model, (torch.empty(n, 7168, dtype=torch.bfloat16),))
        for n in (2048, 8192)
    ]
    assert signatures[0] == signatures[1]


def test_mla_tp2_gate_buffer_refresh_preserves_checkpoint_parameters():
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
    from vllm_torchtpu.models.vllm.kimi_k3.collective_ops import FusedPrefillCollectives

    layer = kimi_attention.MultiHeadLatentAttention.__new__(
        kimi_attention.MultiHeadLatentAttention
    )
    nn.Module.__init__(layer)
    layer.mla_gate_is_fused = True
    layer.fused_qkv_a_proj = nn.Module()
    layer.fused_qkv_a_proj.weight = nn.Parameter(
        torch.empty(7168, 2496, dtype=torch.bfloat16, device="meta")
    )
    setattr(layer.fused_qkv_a_proj, WEIGHT_FLIPPED_ATTR, True)
    layer.q_a_layernorm = SimpleNamespace(variance_epsilon=1e-5)
    layer.kv_a_layernorm = SimpleNamespace(variance_epsilon=1e-5)
    layer.mla_attn = nn.Module()
    layer.mla_attn.register_buffer("kimi_gate_weight", None, persistent=False)
    group = FusedPrefillCollectives.__new__(FusedPrefillCollectives)
    calls = []

    def op(*args):
        return args

    def pack(weight):
        calls.append(weight)
        return torch.tensor([len(calls)], dtype=torch.bfloat16)

    group.mla_ops = lambda eps: (pack, op)
    layer.prefill_group = group
    original = layer.fused_qkv_a_proj.weight
    for expected in (1, 2):
        layer.prepare_sp_mla_gate()
        assert layer.mla_attn.kimi_gate_weight.item() == expected
        assert layer.mla_attn.kimi_preprocess is op
        assert layer.fused_qkv_a_proj.weight is original
    assert all(weight is original for weight in calls)
    assert not any("kimi_gate_weight" in key for key in layer.state_dict())


def test_mla_wrapper_uses_global_rows_after_local_preprocessing():
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    wrapper = VllmTPUMultiHeadLatentAttentionWrapper.__new__(
        VllmTPUMultiHeadLatentAttentionWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(0))
    wrapper.kimi_gate_weight = torch.empty(0)
    wrapper.q_a_layernorm = SimpleNamespace(weight=torch.empty(0))
    wrapper.kv_a_layernorm = SimpleNamespace(weight=torch.empty(0))
    wrapper.num_heads = 3
    wrapper.v_head_dim = wrapper.qk_nope_head_dim = 128
    wrapper.qk_rope_head_dim = 64
    wrapper.qk_head_dim = 192
    wrapper.rotary_emb = None
    wrapper.is_sparse = False
    wrapper.o_proj = object()
    calls = []

    def preprocess(x, *args):
        assert x.shape[0] == 2 and args[-1] is True
        return tuple(
            torch.zeros(64, n, dtype=torch.bfloat16) for n in (1536, 512, 64, 384)
        )

    wrapper.kimi_preprocess = preprocess
    wrapper.q_b_proj = lambda q: (torch.zeros(q.shape[0], 576, dtype=q.dtype), None)

    def attend(q, kv, rope, *, output_shape, **kwargs):
        assert q[0].shape == (64, 3, 128)
        assert kv.shape == (64, 512) and rope.shape == (64, 1, 64)
        assert output_shape == (64, 384)
        return torch.ones(output_shape, dtype=torch.bfloat16)

    wrapper.mla_attn = attend
    group = SimpleNamespace(
        project_reduce_scatter=lambda x, w: calls.append(x) or x[:2]
    )
    result = wrapper(
        torch.arange(64),
        torch.empty(2, 7168, dtype=torch.bfloat16),
        prefill_group=group,
    )
    assert result.shape == (2, 384)
    torch.testing.assert_close(
        calls[0], torch.full((64, 384), 0.5, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize(
    "tokens", [0, 1, 8, 31, 32, 64, 96, 1024, 1056, 2048, 4096, 8192, 16384]
)
@pytest.mark.parametrize("rank", [0, 7, 31])
@pytest.mark.parametrize("capture_aux", [False, True])
def test_sp_prefill_divisible_shapes_preserve_token_ownership(
    tokens, rank, capture_aux
):
    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    model.sp_prefill = True
    model.aux_hidden_state_layers = (1,) if capture_aux else ()
    model.attn_res_block_size = None
    model.norm = nn.Identity()
    hidden = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
    positions = torch.arange(tokens)
    expected_sp = tokens > 0 and tokens % 32 == 0
    calls = []

    class Layer(nn.Module):
        def forward(self, pos, value, residual, *, sequence_parallel):
            assert sequence_parallel == expected_sp
            assert pos is positions
            expected = hidden.reshape(32, -1, 4)[rank] if expected_sp else hidden
            torch.testing.assert_close(value, expected)
            return value + 1, residual

    def gather(value, dim):
        calls.append(dim)
        torch.testing.assert_close(value, hidden.reshape(32, -1, 4)[rank] + 1)
        return hidden + 1

    model.sp_group = SimpleNamespace(
        world_size=32, rank_in_group=rank, all_gather=gather
    )
    model.layers = nn.ModuleList([Layer()])
    actual = model.forward(None, positions, inputs_embeds=hidden)
    if capture_aux:
        actual, auxiliary = actual
        assert len(auxiliary) == 1
        torch.testing.assert_close(auxiliary[0], hidden + 1)
    torch.testing.assert_close(actual, hidden + 1)
    assert calls == ([0] * (2 if capture_aux else 1) if expected_sp else [])
