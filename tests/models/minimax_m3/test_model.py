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
"""MiniMax M3 registration, config checks and checkpoint loading on CPU."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationMode,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

import vllm_torchtpu.models.minimax_m3.attention as attention_mod
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.quantization import _MODEL_QUANT_CONFIGS
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4MoEMethod
from vllm_torchtpu.models.minimax_m3.hooks import TPUMiniMaxM3Config
from vllm_torchtpu.models.minimax_m3.model import MiniMaxM3SparseForCausalLM
from vllm_torchtpu.models.minimax_m3.quant import (
    MiniMaxM3Mxfp8LinearMethod,
    MiniMaxM3Nvfp4MoEMethod,
    MiniMaxM3QuantConfig,
    dequantize_mxfp8,
)

HIDDEN = 64
HEAD_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 4
INTER = 64
EXPERTS = 4
LAYERS = 2  # layer 0 dense MLP, layer 1 MoE
_LM = "language_model.model"


def _text_config() -> MiniMaxM3TextConfig:
    return MiniMaxM3TextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        dense_intermediate_size=INTER,
        shared_intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=96,
        max_position_embeddings=1024,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        rope_theta=10000000,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        num_local_experts=EXPERTS,
        num_experts_per_tok=2,
        n_shared_experts=1,
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        moe_layer_freq=[0, 1],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": HEAD_DIM,
            "sparse_num_index_heads": NUM_KV_HEADS,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_score_type": "max",
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_attention_freq": [1] * LAYERS,
            "sparse_disable_index_value": [1] * LAYERS,
        },
    )


def _quant_dict() -> dict:
    return {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "group_size": 16,
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
        "ignored_layers": [
            f"{_LM}.layers.1.block_sparse_moe.gate",
            "language_model.lm_head",
            f"{_LM}.embed_tokens",
            "vision_tower",
        ],
    }


def test_minimax_m3_is_registered() -> None:
    register_layers()
    for arch in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ):
        assert ModelRegistry.models[arch].load_model_cls() is (
            MiniMaxM3SparseForCausalLM
        )
        assert MODELS_CONFIG_MAP[arch] is TPUMiniMaxM3Config
    assert _MODEL_QUANT_CONFIGS["mxfp8"] is MiniMaxM3QuantConfig


def _hook_config(**overrides) -> SimpleNamespace:
    text_config = _text_config()
    values = dict(
        tensor_parallel_size=4,
        cache_dtype="auto",
        block_size=16,
        user_specified_block_size=False,
        speculative_config=None,
        indexer_kv_dtype="auto",
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
        disable_hybrid_kv_cache_manager=None,
    )
    values.update(overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        cache_config=SimpleNamespace(
            block_size=values["block_size"],
            user_specified_block_size=values["user_specified_block_size"],
            cache_dtype=values["cache_dtype"],
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=values["tensor_parallel_size"],
            pipeline_parallel_size=1,
            prefill_context_parallel_size=values["prefill_context_parallel_size"],
            decode_context_parallel_size=values["decode_context_parallel_size"],
            cp_kv_cache_interleave_size=values["cp_kv_cache_interleave_size"],
        ),
        speculative_config=values["speculative_config"],
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=values["disable_hybrid_kv_cache_manager"]
        ),
        attention_config=AttentionConfig(
            indexer_kv_dtype=values["indexer_kv_dtype"],
        ),
    )


@pytest.mark.parametrize(
    "cache_dtype,indexer_kv_dtype",
    [("auto", "auto"), ("fp8", "auto"), ("fp8", "fp8"), ("bfloat16", "bf16")],
)
def test_config_pins_block_size_to_sparse_block(cache_dtype, indexer_kv_dtype) -> None:
    config = _hook_config(cache_dtype=cache_dtype, indexer_kv_dtype=indexer_kv_dtype)
    TPUMiniMaxM3Config.verify_and_update_config(config)
    assert config.cache_config.block_size == 128
    assert config.cache_config.user_specified_block_size


@pytest.mark.parametrize(
    "overrides,error",
    [
        (dict(tensor_parallel_size=2), NotImplementedError),
        (dict(cache_dtype="fp8_e5m2"), NotImplementedError),
        (dict(cache_dtype="fp8", indexer_kv_dtype="bf16"), NotImplementedError),
        (
            dict(speculative_config=SimpleNamespace(method="eagle3")),
            NotImplementedError,
        ),
        (dict(block_size=64, user_specified_block_size=True), ValueError),
        (dict(indexer_kv_dtype="fp8"), NotImplementedError),
        (dict(prefill_context_parallel_size=2), NotImplementedError),
        # DCP needs whole sparse blocks per rank, splits KV-head replicas only,
        # and runs the DSpark draft only in the unified KV cache group.
        (
            dict(
                tensor_parallel_size=8,
                decode_context_parallel_size=2,
                cp_kv_cache_interleave_size=64,
            ),
            ValueError,
        ),
        # DCP must divide TP (wider groups up to TP span several KV heads).
        (dict(tensor_parallel_size=8, decode_context_parallel_size=3), ValueError),
        (
            dict(
                tensor_parallel_size=8,
                decode_context_parallel_size=2,
                speculative_config=SimpleNamespace(method="dspark"),
                disable_hybrid_kv_cache_manager=False,
            ),
            NotImplementedError,
        ),
    ],
)
def test_config_rejects_unsupported(overrides, error) -> None:
    with pytest.raises(error):
        TPUMiniMaxM3Config.verify_and_update_config(_hook_config(**overrides))


@pytest.mark.parametrize("dcp", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 128])
def test_config_dcp_keeps_whole_blocks_per_rank(interleave, dcp) -> None:
    config = _hook_config(
        tensor_parallel_size=8,
        decode_context_parallel_size=dcp,
        cp_kv_cache_interleave_size=interleave,
    )
    TPUMiniMaxM3Config.verify_and_update_config(config)
    assert config.parallel_config.cp_kv_cache_interleave_size == 128


def test_config_dcp_accepts_dspark(monkeypatch) -> None:
    import vllm_torchtpu.models.minimax_m3.dspark as dspark_mod

    seen = []
    monkeypatch.setattr(dspark_mod, "configure_dspark_draft", seen.append)
    speculative_config = SimpleNamespace(method="dspark")
    config = _hook_config(
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
        speculative_config=speculative_config,
        disable_hybrid_kv_cache_manager=True,
    )
    TPUMiniMaxM3Config.verify_and_update_config(config)
    assert seen == [speculative_config]
    assert config.parallel_config.cp_kv_cache_interleave_size == 128


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_msa_cache_spec_under_dcp(dcp, monkeypatch) -> None:
    """A rank's page bytes do not change with DCP, but the spec is
    DCP-sharded: the scheduler's block (and so each page's tokens) grows
    dcp-fold, and a request needs 1 / dcp of the pages per rank."""
    from vllm.platforms import current_platform
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")

    vllm_config = VllmConfig()
    vllm_config.cache_config = CacheConfig(block_size=128)
    vllm_config.parallel_config.decode_context_parallel_size = dcp
    with set_current_vllm_config(vllm_config):
        caches = [
            attention_mod.MiniMaxM3Cache(256, torch.bfloat16, prefix=f"l.kv{dcp}"),
            attention_mod.MiniMaxM3Cache(128, torch.bfloat16, prefix=f"l.idx{dcp}"),
        ]
    for cache, width in zip(caches, (256, 128), strict=True):
        spec = cache.get_kv_cache_spec(vllm_config)
        assert spec.dcp_sharded
        assert spec.block_size == 128
        assert spec.page_size_bytes == 128 * width * 2
        assert spec.max_num_blocks_per_req(vllm_config, 128 * 1024) == 1024 // dcp
        manager = FullAttentionManager(
            spec,
            block_pool=SimpleNamespace(null_block=None),
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=128 * dcp,
            dcp_world_size=dcp,
        )
        assert manager.block_size == 128 * dcp


def test_dequantize_mxfp8_applies_power_of_two_block_scales() -> None:
    weight = torch.tensor([[1.0, -2.0] * 32]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127 + 3, 127 - 1]], dtype=torch.uint8)
    out = dequantize_mxfp8(weight, scale)
    assert out.dtype == torch.bfloat16
    expected = torch.tensor([[8.0, -16.0] * 16 + [0.5, -1.0] * 16])
    torch.testing.assert_close(out.float(), expected)


def test_quant_config_rejects_other_schemes() -> None:
    config = _quant_dict() | {"moe_quant_algo": "MXFP4"}
    with pytest.raises(NotImplementedError):
        MiniMaxM3QuantConfig.from_config(config)


def _mxfp8(rows: int, cols: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    weight = torch.randn(rows, cols, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 130, (rows, cols // 32), generator=gen)
    return {"weight": weight, "weight_scale_inv": scale.to(torch.uint8)}


def _nvfp4(rows: int, cols: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        "weight_packed": torch.randint(0, 256, (rows, cols // 2), generator=gen).to(
            torch.uint8
        ),
        "weight_scale": torch.rand(rows, cols // 16, generator=gen).to(
            torch.float8_e4m3fn
        ),
        "weight_global_scale": torch.tensor(4.0 + torch.rand(1, generator=gen).item()),
        "input_global_scale": torch.tensor(1.0),
    }


def _checkpoint() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    ckpt: dict[str, torch.Tensor] = {
        "language_model.lm_head.weight": torch.randn(96, HIDDEN),
        f"{_LM}.embed_tokens.weight": torch.randn(96, HIDDEN),
        f"{_LM}.norm.weight": torch.randn(HIDDEN),
        "vision_tower.blocks.0.attn.qkv.weight": torch.randn(4, 4),
        "multi_modal_projector.linear_1.weight": torch.randn(4, 4),
    }
    attn_shapes = {
        "q_proj": NUM_HEADS * HEAD_DIM,
        "k_proj": NUM_KV_HEADS * HEAD_DIM,
        "v_proj": NUM_KV_HEADS * HEAD_DIM,
        "index_q_proj": NUM_KV_HEADS * HEAD_DIM,
        "index_k_proj": HEAD_DIM,
    }
    for layer in range(LAYERS):
        p = f"{_LM}.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            ckpt[f"{p}.{norm}.weight"] = torch.randn(HIDDEN)
        for norm in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            ckpt[f"{p}.self_attn.{norm}.weight"] = torch.randn(HEAD_DIM)
        for proj, rows in attn_shapes.items():
            for k, t in _mxfp8(rows, HIDDEN, gen).items():
                ckpt[f"{p}.self_attn.{proj}.{k}"] = t
        for k, t in _mxfp8(HIDDEN, NUM_HEADS * HEAD_DIM, gen).items():
            ckpt[f"{p}.self_attn.o_proj.{k}"] = t
        mlp = f"{p}.mlp" if layer == 0 else f"{p}.block_sparse_moe.shared_experts"
        for proj in ("gate_proj", "up_proj"):
            for k, t in _mxfp8(INTER, HIDDEN, gen).items():
                ckpt[f"{mlp}.{proj}.{k}"] = t
        for k, t in _mxfp8(HIDDEN, INTER, gen).items():
            ckpt[f"{mlp}.down_proj.{k}"] = t
    moe = f"{_LM}.layers.1.block_sparse_moe"
    ckpt[f"{moe}.gate.weight"] = torch.randn(EXPERTS, HIDDEN)
    ckpt[f"{moe}.e_score_correction_bias"] = torch.randn(EXPERTS)
    for e in range(EXPERTS):
        for w, (rows, cols) in {
            "w1": (INTER, HIDDEN),
            "w3": (INTER, HIDDEN),
            "w2": (HIDDEN, INTER),
        }.items():
            for k, t in _nvfp4(rows, cols, gen).items():
                ckpt[f"{moe}.experts.{e}.{w}.{k}"] = t
    return ckpt


class _ModelConfigStub(SimpleNamespace):
    """Model config for layer construction; unset optional features are off."""

    def __getattr__(self, name: str) -> None:
        if name.startswith("__"):
            raise AttributeError(name)
        return None


@pytest.fixture
def tiny_model(request, monkeypatch):
    from vllm.platforms import current_platform

    # CPU CI has no TPU device registered with PyTorch: keep the Gloo group
    # and parameter allocations on CPU.
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    request.getfixturevalue("dist_ctx")
    # The MSA op is a Pallas op; it needs the worker's TPU runtime.
    monkeypatch.setattr(attention_mod, "get_msa_op", lambda *args, **kwargs: None)
    text_config = _text_config()
    quant_config = MiniMaxM3QuantConfig.from_config(_quant_dict())
    configure_quant_config(quant_config, MiniMaxM3SparseForCausalLM)
    vllm_config = VllmConfig()
    vllm_config.model_config = _ModelConfigStub(
        hf_config=text_config,
        hf_text_config=text_config,
        dtype=torch.bfloat16,
        head_dtype=None,
        max_model_len=1024,
    )
    vllm_config.cache_config = CacheConfig(block_size=128)
    vllm_config.scheduler_config = SchedulerConfig(
        max_model_len=1024,
        is_encoder_decoder=False,
        max_num_seqs=1,
        max_num_batched_tokens=128,
    )
    vllm_config.quant_config = quant_config
    monkeypatch.setattr(MiniMaxM3QuantConfig, "vllm_config", vllm_config)
    vllm_config.compilation_config.mode = CompilationMode.NONE
    resolve_kv_cache_layout(vllm_config, [["LBNHC", "LBHNC"]])
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            model = MiniMaxM3SparseForCausalLM(vllm_config=vllm_config, prefix="")
    finally:
        torch.set_default_dtype(torch.float32)
    return model, vllm_config


def test_checkpoint_loads_and_mxfp8_dequantizes(tiny_model, monkeypatch) -> None:
    model, vllm_config = tiny_model
    ckpt = _checkpoint()
    attn = model.model.layers[0].self_attn
    assert isinstance(attn.qkv_proj.quant_method, MiniMaxM3Mxfp8LinearMethod)
    moe = model.model.layers[1].block_sparse_moe
    assert isinstance(moe.gate.quant_method, UnquantizedLinearMethod)
    experts = moe.experts.routed_experts
    assert isinstance(experts.quant_method, MiniMaxM3Nvfp4MoEMethod)

    # Both MSA caches are model-owned rows: [K | V] and index keys.
    specs = {
        name: cache.get_kv_cache_spec(vllm_config)
        for name, cache in (("kv", attn.kv_cache), ("index", attn.index_cache))
    }
    assert (specs["kv"].head_size, specs["index"].head_size) == (2 * HEAD_DIM, HEAD_DIM)
    assert {spec.dtype for spec in specs.values()} == {torch.bfloat16}
    static_forward_context = vllm_config.compilation_config.static_forward_context
    assert static_forward_context["model.layers.0.self_attn.attn"] is attn.kv_cache

    loaded = model.load_weights(ckpt.items())
    params = {name for name, _ in model.named_parameters()}
    assert loaded == params

    # Experts: compressed-tensors names land in the ModelOpt parameters, and
    # the global scale is inverted before the shared NVFP4 preparation.
    e1 = f"{_LM}.layers.1.block_sparse_moe.experts.1"
    torch.testing.assert_close(
        experts.w13_weight[1],
        torch.cat([ckpt[f"{e1}.w1.weight_packed"], ckpt[f"{e1}.w3.weight_packed"]]),
    )
    seen = {}
    monkeypatch.setattr(
        VllmNvfp4MoEMethod,
        "process_weights_after_loading",
        lambda self, layer: seen.update(scale_2=layer.w13_weight_scale_2.clone()),
    )
    experts.quant_method.process_weights_after_loading(experts)
    torch.testing.assert_close(
        seen["scale_2"][1],
        1.0
        / torch.stack(
            [ckpt[f"{e1}.w1.weight_global_scale"], ckpt[f"{e1}.w3.weight_global_scale"]]
        ),
    )

    # Linears: the fused qkv holds [q | k | v | index_q | index_k] dequantized.
    attn.qkv_proj.quant_method.process_weights_after_loading(attn.qkv_proj)
    p = f"{_LM}.layers.0.self_attn"
    expected = torch.cat(
        [
            dequantize_mxfp8(
                ckpt[f"{p}.{proj}.weight"], ckpt[f"{p}.{proj}.weight_scale_inv"]
            )
            for proj in ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj")
        ]
    )
    torch.testing.assert_close(attn.qkv_proj.weight, expected.T)
    assert not hasattr(attn.qkv_proj, "weight_scale")
    x = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
    out, _ = attn.qkv_proj(x)
    torch.testing.assert_close(out, x @ expected.T)


def test_vllm_gqa_dcp_cap_is_lifted_for_registered_models(monkeypatch) -> None:
    """vLLM caps GQA DCP at tp // num_kv_heads (2 for M3 at TP8). For a
    registered architecture the cap check sees the capped size, so DCP 8
    passes it; other checks and other models are unchanged."""
    from vllm.config.model import ModelConfig

    from vllm_torchtpu.distributed import dcp as dcp_mod

    seen = []
    monkeypatch.setattr(
        ModelConfig,
        "verify_with_parallel_config",
        lambda self, pc: seen.append(pc.decode_context_parallel_size),
    )
    monkeypatch.setattr(dcp_mod, "_KV_HEAD_SPANNING_DCP_ARCHITECTURES", set())
    dcp_mod._patch_vllm_gqa_dcp_limit()
    dcp_mod.register_kv_head_spanning_dcp_architectures("MiniMaxM3SparseForCausalLM")

    def model(arch, use_mla=False):
        return SimpleNamespace(
            architectures=[arch],
            use_mla=use_mla,
            get_total_num_kv_heads=lambda: NUM_KV_HEADS,
        )

    pc = SimpleNamespace(decode_context_parallel_size=8, tensor_parallel_size=8)
    ModelConfig.verify_with_parallel_config(model("MiniMaxM3SparseForCausalLM"), pc)
    ModelConfig.verify_with_parallel_config(model("OtherForCausalLM"), pc)
    ModelConfig.verify_with_parallel_config(
        model("MiniMaxM3SparseForCausalLM", use_mla=True), pc
    )
    assert seen == [2, 8, 8]
    assert pc.decode_context_parallel_size == 8  # the caller's config is untouched


@pytest.mark.parametrize("unit,expected", [(None, 128), (64, 64)])
def test_config_dcp_defaults_prefix_match_unit_to_the_sparse_block(
    unit, expected
) -> None:
    """Under DCP the manager block is 128 * dcp tokens; the hook makes prefix
    hits land on sparse blocks unless --prefix-match-unit is given."""
    config = _hook_config(tensor_parallel_size=8, decode_context_parallel_size=8)
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = unit
    TPUMiniMaxM3Config.verify_and_update_config(config)
    assert config.cache_config.prefix_match_unit == expected
