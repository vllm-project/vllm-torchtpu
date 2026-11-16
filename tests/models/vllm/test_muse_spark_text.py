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
"""CPU and TPU unit tests for the Muse Spark Open text model on vLLM v0.33.0."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.platforms import current_platform
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs.muse_spark import (
    MuseSparkAudioConfig,
    MuseSparkConfig,
    MuseSparkTextConfig,
    MuseSparkVisionConfig,
)
from vllm.utils.torch_utils import set_default_torch_dtype

import vllm_torchtpu.envs as tpu_envs
from vllm_torchtpu.layers.adapter.custom_ops import post_norm_moe_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.muse_spark import compat
from vllm_torchtpu.models.vllm.muse_spark import model as muse_model_mod
from vllm_torchtpu.models.vllm.muse_spark.config import (
    TPUMuseSparkConfig,
    normalize_muse_spark_hf_quant_config,
)
from vllm_torchtpu.models.vllm.muse_spark.model import (
    MuseSparkForConditionalGeneration,
    TPUMuseSparkAttention,
    TPUMuseSparkDecoderLayer,
    TPUMuseSparkForCausalLM,
    TPUMuseSparkModel,
    TPUMuseSparkMoE,
)


def make_tiny_text_config_dict(
    *,
    num_hidden_layers: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    moe_hidden_size: int = 32,
    moe_intermediate_size: int = 48,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 16,
    num_local_experts: int = 4,
    num_experts_per_tok: int = 2,
    vocab_size: int = 256,
    mlp_layer_types: list[str] | None = None,
) -> dict[str, Any]:
    layer_types = [
        "sliding_attention" if i == 0 else "full_attention"
        for i in range(num_hidden_layers)
    ]
    layer_rope_theta = [
        500_000.0 if t == "sliding_attention" else 0.0 for t in layer_types
    ]
    if mlp_layer_types is None:
        mlp_layer_types = [
            "dense" if i == 0 else "sparse" for i in range(num_hidden_layers)
        ]
    return {
        "model_type": "muse_spark_text",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_hidden_size": moe_hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 131_072,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "hidden_act": "silu",
        "attention_bias": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 500_000.0},
        "sliding_window": 2048,
        "layer_types": layer_types,
        "mlp_layer_types": mlp_layer_types,
        "layer_rope_theta": layer_rope_theta,
        "num_local_experts": num_local_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_shared_experts": 0,
        "routed_scaling_factor": 1.0,
        "output_multiplier": 0.17677669529663687,
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 5.4730064863875,
        "residual_gate_temperature": 0.3,
        "tie_word_embeddings": False,
    }


def make_tiny_omni_config_dict(**text_overrides: Any) -> dict[str, Any]:
    text_dict = make_tiny_text_config_dict(**text_overrides)
    hidden = text_dict["hidden_size"]
    return {
        "model_type": "muse_spark",
        "architectures": ["MuseSparkForConditionalGeneration"],
        "image_token_id": 250,
        "video_token_id": 251,
        "audio_token_id": 252,
        "out_hidden_size": 128,
        "projector_hidden_size": 64,
        "projector_hidden_act": "gelu",
        "text_config": text_dict,
        "vision_config": {
            "model_type": "muse_spark_vision",
            "patch_size": 14,
            "pos_emb_height": 8,
            "pos_emb_width": 8,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "hidden_act": "gelu",
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0},
            "patch_temporal": 2,
            "merge_size": 2,
            "max_position_embeddings": 64,
            "layer_norm_eps": 1e-5,
            "layer_types": ["window_attention", "full_attention"],
        },
        "audio_config": {
            "model_type": "muse_spark_audio",
            "patch_sizes": [16, 6, 5, 4],
            "hidden_sizes": [16, 32, 32, 64],
            "depths": [1, 1, 1, 1],
            "num_hidden_layers": 4,
            "mlp_ratio": 2.0,
            "conv_kernel_size": 3,
            "hidden_act": "gelu",
            "rms_norm_eps": 1e-6,
            "out_hidden_size": hidden,
        },
    }


def _use_cpu_platform(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
    request.getfixturevalue("dist_ctx")


def make_tiny_vllm_config(
    tmp_path: Path,
    *,
    omni: bool = True,
    quant_config: Any = None,
    device: str = "cpu",
    max_model_len: int = 4096,
    limit_mm_per_prompt: dict[str, int] | None = None,
    **text_overrides: Any,
) -> VllmConfig:
    cfg_dict = (
        make_tiny_omni_config_dict(**text_overrides)
        if omni
        else (
            make_tiny_text_config_dict(**text_overrides)
            | {"architectures": ["MuseSparkForConditionalGeneration"]}
        )
    )
    model_dir = tmp_path / ("omni_model" if omni else "text_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(cfg_dict))
    mc_kwargs: dict[str, Any] = {
        "model": str(model_dir),
        "tokenizer_mode": "skip",
        "dtype": torch.bfloat16,
        "max_model_len": max_model_len,
    }
    if limit_mm_per_prompt is not None:
        mc_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
    model_config = ModelConfig(**mc_kwargs)
    vc = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=128, cache_dtype="auto"),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device=device),
        quant_config=quant_config
        if quant_config is not None
        else VllmUnquantizedConfig(),
    )
    VllmQuantConfig.set_configs(vc)
    return vc


def build_synthetic_hf_weights(
    text_cfg: MuseSparkTextConfig, *, include_towers: bool = False
) -> dict[str, torch.Tensor]:
    """Generate synthetic HF-named Muse Spark weights for OMNI_WEIGHTS_MAPPER."""
    torch.manual_seed(123)
    h = text_cfg.hidden_size
    inter = text_cfg.intermediate_size
    mh = text_cfg.moe_hidden_size
    mi = text_cfg.moe_intermediate_size
    nq = text_cfg.num_attention_heads * text_cfg.head_dim
    nkv = text_cfg.num_key_value_heads * text_cfg.head_dim
    e = text_cfg.num_local_experts
    vocab = text_cfg.vocab_size

    weights: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(
            vocab, h, dtype=torch.bfloat16
        )
        * 0.02,
        "model.language_model.norm.weight": torch.zeros(h, dtype=torch.float32),
        "lm_head.weight": torch.randn(vocab, h, dtype=torch.bfloat16) * 0.02,
    }
    for li in range(text_cfg.num_hidden_layers):
        p = f"model.language_model.layers.{li}"
        weights[f"{p}.input_layernorm.weight"] = torch.zeros(h, dtype=torch.float32)
        weights[f"{p}.self_attn.q_proj.weight"] = (
            torch.randn(nq, h, dtype=torch.bfloat16) * 0.02
        )
        weights[f"{p}.self_attn.k_proj.weight"] = (
            torch.randn(nkv, h, dtype=torch.bfloat16) * 0.02
        )
        weights[f"{p}.self_attn.v_proj.weight"] = (
            torch.randn(nkv, h, dtype=torch.bfloat16) * 0.02
        )
        weights[f"{p}.self_attn.gate_proj.weight"] = (
            torch.randn(nq, h, dtype=torch.bfloat16) * 0.02
        )
        weights[f"{p}.self_attn.o_proj.weight"] = (
            torch.randn(h, nq, dtype=torch.bfloat16) * 0.02
        )
        weights[f"{p}.post_attention_residual_gate.gate"] = torch.zeros(
            h, dtype=torch.float32
        )
        weights[f"{p}.pre_feedforward_layernorm.weight"] = torch.zeros(
            h, dtype=torch.float32
        )
        weights[f"{p}.post_feedforward_layernorm.weight"] = torch.zeros(
            h, dtype=torch.float32
        )
        weights[f"{p}.post_feedforward_residual_gate.gate"] = torch.zeros(
            h, dtype=torch.float32
        )
        if text_cfg.mlp_layer_types[li] == "sparse":
            weights[f"{p}.mlp.gate.weight"] = (
                torch.randn(e, h, dtype=torch.float32) * 0.02
            )
            weights[f"{p}.mlp.gate.e_score_correction_bias"] = torch.zeros(
                e, dtype=torch.float32
            )
            weights[f"{p}.mlp.pre_expert_proj.weight"] = (
                torch.randn(mh, h, dtype=torch.bfloat16) * 0.02
            )
            weights[f"{p}.mlp.pre_expert_norm.weight"] = torch.zeros(
                mh, dtype=torch.float32
            )
            weights[f"{p}.mlp.experts.gate_up_proj"] = (
                torch.randn(e, 2 * mi, mh, dtype=torch.bfloat16) * 0.02
            )
            weights[f"{p}.mlp.experts.down_proj"] = (
                torch.randn(e, mh, mi, dtype=torch.bfloat16) * 0.02
            )
            weights[f"{p}.mlp.experts.post_expert_norm.weight"] = torch.zeros(
                mh, dtype=torch.float32
            )
            weights[f"{p}.mlp.post_expert_proj.weight"] = (
                torch.randn(h, mh, dtype=torch.bfloat16) * 0.02
            )
        else:
            weights[f"{p}.mlp.gate_proj.weight"] = (
                torch.randn(inter, h, dtype=torch.bfloat16) * 0.02
            )
            weights[f"{p}.mlp.up_proj.weight"] = (
                torch.randn(inter, h, dtype=torch.bfloat16) * 0.02
            )
            weights[f"{p}.mlp.down_proj.weight"] = (
                torch.randn(h, inter, dtype=torch.bfloat16) * 0.02
            )

    if include_towers:
        weights["model.vision_tower.layers.0.attn.q_proj.weight"] = torch.ones(4, 4)
        weights["model.vision_adapter.fc1.weight"] = torch.ones(4, 4)
        weights["model.vision_projection.weight"] = torch.ones(4, 4)
        weights["model.audio_tower.layers.0.conv_blocks.0.norm.weight"] = torch.ones(4)
    return weights


def test_upstream_muse_imports() -> None:
    """Upstream vLLM v0.33.0 muse_spark modules import cleanly on CPU/TPU images."""
    import vllm.models.muse_spark  # noqa: F401
    import vllm.models.muse_spark.amd.model  # noqa: F401
    import vllm.models.muse_spark.common.omni  # noqa: F401


def test_register_models_is_lazy() -> None:
    """register_models() registers string paths without heavy imports."""
    code = (
        "import sys, vllm_torchtpu.models.vllm as m; m.register_models();"
        "bad=[k for k in sys.modules if k.startswith(("
        "'vllm_torchtpu.models.vllm.muse_spark.model',"
        "'vllm.models.muse_spark',"
        "'vllm_torchtpu.layers.adapter.custom_ops.post_norm_moe_op'))];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_text_model_imports_no_batch_invariant() -> None:
    """Muse Spark model imports no vllm_torchtpu batch_invariant/det modules."""
    bad = [
        k
        for k in sys.modules
        if k.startswith("vllm_torchtpu")
        and ("batch_invariant" in k or "det_collectives" in k)
    ]
    assert not bad, f"Unexpected batch-invariant modules loaded: {bad}"


def test_upstream_config_parses_and_no_registry_shadowing() -> None:
    """Upstream MuseSparkConfig parses release fields without registry shadowing."""
    register_models()
    assert _CONFIG_REGISTRY["muse_spark"] is MuseSparkConfig
    assert _CONFIG_REGISTRY["muse_spark_text"] is MuseSparkTextConfig
    assert _CONFIG_REGISTRY["muse_spark_vision"] is MuseSparkVisionConfig
    assert _CONFIG_REGISTRY["muse_spark_audio"] is MuseSparkAudioConfig

    cfg = MuseSparkConfig(**make_tiny_omni_config_dict())
    assert isinstance(cfg.text_config, MuseSparkTextConfig)
    assert cfg.get_text_config() is cfg.text_config


def test_compat_apply_idempotent_and_rmsnorm_off_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compat.apply() routes non-CUDA tensors through symbolic-shape-safe norms."""
    import torch.nn.functional as F
    from vllm.models.muse_spark.common import layers as up_layers

    compat.apply()
    compat.apply()  # idempotent
    monkeypatch.setattr(
        up_layers,
        "fused_rmsnorm_fwd",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Triton called off CUDA")),
    )
    norm = up_layers.MuseSpark1xRMSNorm(
        hidden_size=16, eps=1e-5, has_weight=True, zero_centered_gamma=True
    )
    norm.weight.data.zero_()
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(norm(x), norm.forward_native(x))

    # MuseSpark1xRMSNormWithInputScale matches F.rms_norm without calling it,
    # and traces cleanly under dynamic batch shapes.
    in_norm = up_layers.MuseSpark1xRMSNormWithInputScale(
        hidden_size=16, eps=1e-5, has_weight=True, zero_centered_gamma=True
    )
    in_norm.weight.data.uniform_(-0.2, 0.2)
    res = torch.randn(3, 16, dtype=torch.bfloat16)
    xf = (x.float() + res.float()) * (in_norm.weight.float() + 1.0)
    ref_out = F.rms_norm(xf, (16,), None, 1e-5).to(torch.bfloat16)
    ref_res = (x.float() + res.float()).to(torch.bfloat16)

    monkeypatch.setattr(
        F,
        "rms_norm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("F.rms_norm called")),
    )
    got_out, got_res = in_norm(x, residual=res)
    torch.testing.assert_close(got_out, ref_out)
    torch.testing.assert_close(got_res, ref_res)

    dyn_norm = torch.compile(
        lambda t: in_norm(t), backend="eager", dynamic=True, fullgraph=True
    )
    assert dyn_norm(x).shape == (3, 16)
    assert dyn_norm(torch.randn(7, 16, dtype=torch.bfloat16)).shape == (7, 16)


def test_modelopt_exclusions_union() -> None:
    """normalize_muse_spark_hf_quant_config unions exclusions into ignore."""
    hf = SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "exclude_modules": ["layers.0.", "self_attn"],
            "modules_to_not_convert": ["self_attn", "lm_head"],
        }
    )
    normalize_muse_spark_hf_quant_config(hf)
    assert hf.quantization_config["ignore"] == ["layers.0.", "self_attn", "lm_head"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda vc, _mp: setattr(vc.parallel_config, "data_parallel_size", 2),
            "data_parallel_size=1",
        ),
        (
            lambda vc, _mp: (
                setattr(vc.parallel_config, "data_parallel_size", 2),
                setattr(vc.parallel_config, "enable_expert_parallel", True),
            ),
            "data_parallel_size=1",
        ),
        (
            lambda vc, _mp: setattr(
                vc.parallel_config, "prefill_context_parallel_size", 2
            ),
            "PCP/DCP",
        ),
        (
            lambda vc, _mp: setattr(
                vc.parallel_config, "decode_context_parallel_size", 2
            ),
            "PCP/DCP",
        ),
        (
            lambda vc, _mp: setattr(vc.parallel_config, "pipeline_parallel_size", 2),
            "pipeline_parallel_size=1",
        ),
        (
            lambda vc, _mp: (
                setattr(vc.parallel_config, "tensor_parallel_size", 8),
                setattr(vc.parallel_config, "enable_expert_parallel", False),
            ),
            "enable-expert-parallel",
        ),
        (
            lambda vc, mp: mp.setattr(
                ParallelConfig,
                "use_sequence_parallel_moe",
                property(lambda _self: True),
            ),
            "sequence-parallel",
        ),
        (lambda vc, _mp: setattr(vc.parallel_config, "enable_eplb", True), "EPLB"),
        (
            lambda vc, _mp: setattr(vc, "lora_config", LoRAConfig(max_lora_rank=8)),
            "LoRA",
        ),
        (
            lambda vc, _mp: setattr(vc.cache_config, "cache_dtype", "fp8"),
            "auto/bfloat16 KV cache",
        ),
    ],
)
def test_config_rejects_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Any,
    match: str,
) -> None:
    vc = make_tiny_vllm_config(tmp_path)
    vc.cache_config.user_specified_block_size = False
    vc.cache_config.block_size = 16  # type: ignore[assignment]
    mutator(vc, monkeypatch)
    with pytest.raises(NotImplementedError, match=match):
        TPUMuseSparkConfig.verify_and_update_config(vc)


def test_config_sets_default_block_size_and_rejects_expert_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vc = make_tiny_vllm_config(tmp_path)
    vc.cache_config.user_specified_block_size = False
    vc.cache_config.block_size = 16  # type: ignore[assignment]
    TPUMuseSparkConfig.verify_and_update_config(vc)
    assert vc.cache_config.block_size == 128

    monkeypatch.setattr(tpu_envs, "TPU_EXPERT_PLACEMENT", "/tmp/placement.json")
    with pytest.raises(NotImplementedError, match="TPU_EXPERT_PLACEMENT"):
        TPUMuseSparkConfig.verify_and_update_config(vc)


def test_quant_guard_accepts_tpu_and_rejects_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    _use_cpu_platform(monkeypatch, request)
    assert set(TPUMuseSparkForCausalLM.supported_quantization_configs) == {
        VllmNvfp4Config,
        VllmFp8Config,
        VllmUnquantizedConfig,
    }

    class _FakeQuarkConfig:
        packed_modules_mapping: dict[str, list[str]] = {}

        @staticmethod
        def get_name() -> str:
            return "quark"

        @staticmethod
        def apply_vllm_mapper(_mapper: Any) -> None:
            return None

    vc = make_tiny_vllm_config(tmp_path, omni=False, quant_config=_FakeQuarkConfig())
    with (
        set_current_vllm_config(vc),
        pytest.raises(NotImplementedError, match="MuseSpark1x on TPU only supports"),
    ):
        TPUMuseSparkForCausalLM(vllm_config=vc)


def test_model_compile_flag_preserves_single_wrapper() -> None:
    """TPUMuseSparkModel re-enables @support_torch_compile without double-wrapping."""
    from vllm.models.muse_spark.amd import model as up

    assert up.MuseSpark1xModel._ignore_compile_vllm is True
    assert TPUMuseSparkModel._ignore_compile_vllm is False
    assert issubclass(TPUMuseSparkModel, TorchCompileWithNoGuardsWrapper)
    assert TPUMuseSparkModel.__mro__.count(TorchCompileWithNoGuardsWrapper) == 1


def test_kv_layout_matches_neox_fp32_and_attention_output_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """K cache keeps HF/NEOX fp32 RoPE layout; scores match GPT-J interleave."""
    _use_cpu_platform(monkeypatch, request)
    vc = make_tiny_vllm_config(tmp_path, omni=False, max_model_len=131_072)
    with set_current_vllm_config(vc):
        layer = TPUMuseSparkDecoderLayer(vc, prefix="model.layers.0")

    attn = layer.self_attn
    assert isinstance(attn, TPUMuseSparkAttention)
    assert attn.rotary_emb is not None
    assert attn.rotary_emb.is_neox_style is True
    assert attn.rotary_emb.cos_sin_cache.dtype == torch.float32

    torch.manual_seed(55)
    positions = torch.tensor([0, 1, 4095, 131071], dtype=torch.long)
    qkv = torch.randn(4, 2 * (attn.q_size + attn.kv_size), dtype=torch.bfloat16)

    q_out, k_out, v_out, og_out = attn._qk_norm_rope_kv_insert(qkv, positions)
    assert og_out is not None and v_out.shape == k_out.shape

    # Independent NEOX fp32 reference on the same unpermuted Q/K rows.
    q_raw, k_raw, v_raw, _ = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    assert attn.qk_norm is not None
    k_normed = (
        attn.qk_norm(k_raw.reshape(-1, attn.head_dim), out_dtype=torch.float32)
        .reshape(-1, attn.kv_size)
        .to(v_raw.dtype)
    )
    q_normed = (
        attn.qk_norm(q_raw.reshape(-1, attn.head_dim), out_dtype=torch.float32)
        .reshape(-1, attn.q_size)
        .to(v_raw.dtype)
    )
    ref_rope_neox = get_rope(
        head_size=attn.head_dim,
        max_position=131_072,
        is_neox_style=True,
        rope_parameters={"rope_theta": 500_000.0, "rope_type": "default"},
        dtype=torch.float32,
    )
    q_ref32, k_ref32 = ref_rope_neox(positions, q_normed.float(), k_normed.float())
    assert k_ref32 is not None
    assert torch.equal(k_out, k_ref32.to(v_raw.dtype))
    assert torch.equal(q_out, q_ref32.to(v_raw.dtype))

    # Verify mathematical dot-product invariance against upstream's interleaved
    # Q/K row order + GPT-J (is_neox_style=False) RoPE in fp32.
    def _interleave_heads(t: torch.Tensor) -> torch.Tensor:
        return (
            t.view(t.shape[0], -1, 2, attn.head_dim // 2).transpose(2, 3).reshape_as(t)
        )

    ref_rope_gptj = get_rope(
        head_size=attn.head_dim,
        max_position=131_072,
        is_neox_style=False,
        rope_parameters={"rope_theta": 500_000.0, "rope_type": "default"},
        dtype=torch.float32,
    )
    q_gptj, k_gptj = ref_rope_gptj(
        positions,
        _interleave_heads(q_normed).float(),
        _interleave_heads(k_normed).float(),
    )
    assert k_gptj is not None
    q_neox_h = q_ref32.view(4, attn.num_heads, attn.head_dim)
    k_neox_h = k_ref32.view(4, attn.num_kv_heads, attn.head_dim).repeat_interleave(
        attn.num_heads // attn.num_kv_heads, dim=1
    )
    q_gptj_h = q_gptj.view(4, attn.num_heads, attn.head_dim)
    k_gptj_h = k_gptj.view(4, attn.num_kv_heads, attn.head_dim).repeat_interleave(
        attn.num_heads // attn.num_kv_heads, dim=1
    )
    scores_neox = torch.einsum("thd,shd->hts", q_neox_h, k_neox_h)
    scores_gptj = torch.einsum("thd,shd->hts", q_gptj_h, k_gptj_h)
    torch.testing.assert_close(scores_neox, scores_gptj, rtol=1e-5, atol=1e-5)


def test_real_loader_sequence_binds_op_after_quant_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Regression test for F014: load_weights does not call PWAL prematurely."""
    _use_cpu_platform(monkeypatch, request)
    register_models()
    model_cls = ModelRegistry.models[
        "MuseSparkForConditionalGeneration"
    ].load_model_cls()
    assert issubclass(model_cls, TPUMuseSparkForCausalLM)
    assert MODELS_CONFIG_MAP["MuseSparkForConditionalGeneration"] is TPUMuseSparkConfig

    events: list[str] = []

    def _spy_quant_pwal(self: VllmUnquantizedFusedMoEMethod, layer: nn.Module) -> None:
        events.append("quant_pwal")
        self._tpu_activation_str = "silu"

    monkeypatch.setattr(
        VllmUnquantizedFusedMoEMethod, "process_weights_after_loading", _spy_quant_pwal
    )
    monkeypatch.setattr(
        post_norm_moe_op,
        "get_post_norm_moe_op",
        lambda _static: (
            events.append("get_op"),
            lambda h, *a: torch.zeros(
                (h.shape[0], a[3].shape[0]), dtype=torch.float32, device=h.device
            ),
        )[1],
    )

    vc = make_tiny_vllm_config(
        tmp_path,
        omni=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )
    with set_current_vllm_config(vc):
        model = model_cls(vllm_config=vc)

    moe = model.model.layers[1].feed_forward
    assert isinstance(moe, TPUMuseSparkMoE)
    assert isinstance(moe.pooler.quant_method, VllmUnquantizedLinearMethod)
    assert isinstance(moe.unpooler.quant_method, VllmUnquantizedLinearMethod)

    weights = build_synthetic_hf_weights(
        vc.model_config.hf_config.text_config, include_towers=True
    )
    loaded = model.load_weights(weights.items())
    expected_params = {name for name, _ in model.named_parameters()} | {
        "model.layers.1.feed_forward.expert_bias"
    }
    assert loaded == expected_params

    # Before PWAL runs, no fused MoE op is bound and forward fails closed.
    assert moe._impl is None
    assert events == []
    with pytest.raises(RuntimeError, match="no implementation bound"):
        moe(torch.zeros((2, moe.hidden_size), dtype=torch.bfloat16))

    # Running vLLM's real post-load sequence runs quant_pwal BEFORE bind_fused_op.
    process_weights_after_loading(model, vc.model_config, torch.device("cpu"))
    assert events == ["quant_pwal", "get_op"]
    assert moe._impl is not None

    # router_w_kmajor is materialized K-major [H, E] in FP32 for the fused op.
    assert moe.router_w_kmajor is not None
    assert moe.router_w_kmajor.shape == (moe.hidden_size, moe.n_routed_experts)
    assert moe.router_w_kmajor.dtype == torch.float32

    out = moe(torch.zeros((3, moe.hidden_size), dtype=torch.bfloat16))
    assert out.shape == (3, moe.hidden_size)
    assert out.dtype == torch.bfloat16


def test_router_flipped_weight_handled_in_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """D20: bind_fused_op respects WEIGHT_FLIPPED_ATTR when router.weight is [H, E]."""
    _use_cpu_platform(monkeypatch, request)
    monkeypatch.setattr(muse_model_mod, "build_post_norm_moe", lambda *a, **k: None)
    vc = make_tiny_vllm_config(tmp_path, omni=True)
    with set_current_vllm_config(vc):
        model = MuseSparkForConditionalGeneration(vllm_config=vc)
    moe = model.model.layers[1].feed_forward
    assert isinstance(moe, TPUMuseSparkMoE)

    expected_kmajor = torch.arange(
        moe.hidden_size * moe.n_routed_experts, dtype=torch.float32
    ).reshape(moe.hidden_size, moe.n_routed_experts)
    moe.router.weight = nn.Parameter(expected_kmajor.clone(), requires_grad=False)
    setattr(moe.router, WEIGHT_FLIPPED_ATTR, True)
    moe.bind_fused_op()
    assert torch.equal(moe.router_w_kmajor, expected_kmajor)


def _build_nvfp4_hf_weights_from_model(
    model: nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    """Build synthetic ModelOpt NVFP4 HF-named tensors matching ``model``."""
    ckpt: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if ".self_attn.qkv_proj." in name:
            prefix_l, suffix = name.rsplit(".qkv_proj.", 1)
            layer_idx = int(prefix_l.split(".layers.")[1].split(".")[0])
            qkv = model.model.layers[layer_idx].self_attn.qkv_proj  # type: ignore[index]
            qs = qkv.num_heads * qkv.head_size
            kvs = qkv.num_kv_heads * qkv.head_size
            chunks = torch.split(torch.zeros_like(param), [qs, kvs, kvs, qs], dim=0)
            base = f"model.language_model.{prefix_l.removeprefix('model.')}"
            for sub, chunk in zip(("q_proj", "k_proj", "v_proj", "gate_proj"), chunks):
                ckpt.append((f"{base}.{sub}.{suffix}", chunk))
        elif ".feed_forward.gate_up_proj." in name:
            prefix_l, suffix = name.rsplit(".gate_up_proj.", 1)
            base = f"model.language_model.{prefix_l.removeprefix('model.')}"
            if suffix in ("weight", "weight_scale") and param.ndim >= 2:
                c0, c1 = param.chunk(2, dim=0)
                ckpt.append((f"{base}.gate_proj.{suffix}", torch.zeros_like(c0)))
                ckpt.append((f"{base}.up_proj.{suffix}", torch.zeros_like(c1)))
            else:
                ckpt.append((f"{base}.gate_proj.{suffix}", torch.tensor(1.0)))
                ckpt.append((f"{base}.up_proj.{suffix}", torch.tensor(1.0)))
        elif ".feed_forward.experts." in name:
            hf_name = name.replace("model.", "model.language_model.", 1)
            hf_name = hf_name.replace(
                ".feed_forward.experts.routed_experts.", ".mlp.experts."
            )
            hf_name = hf_name.replace(
                ".w13_weight_scale_2", ".gate_up_proj_weight_scale_2"
            )
            hf_name = hf_name.replace(".w13_weight_scale", ".gate_up_proj_weight_scale")
            hf_name = hf_name.replace(".w13_input_scale", ".gate_up_proj_input_scale")
            hf_name = hf_name.replace(".w13_weight", ".gate_up_proj")
            hf_name = hf_name.replace(".w2_weight_scale_2", ".down_proj_weight_scale_2")
            hf_name = hf_name.replace(".w2_weight_scale", ".down_proj_weight_scale")
            hf_name = hf_name.replace(".w2_input_scale", ".down_proj_input_scale")
            hf_name = hf_name.replace(".w2_weight", ".down_proj")
            if "scale" in name:
                val = (
                    torch.ones_like(param, dtype=torch.float32).to(torch.float8_e4m3fn)
                    if param.dtype == torch.float8_e4m3fn
                    else torch.ones_like(param)
                )
                ckpt.append((hf_name, val))
            else:
                ckpt.append((hf_name, torch.zeros_like(param)))
        else:
            hf_name = (
                name.replace("model.", "model.language_model.", 1)
                if name.startswith("model.")
                else name
            )
            hf_name = hf_name.replace(".feed_forward.", ".mlp.").replace(
                ".router.", ".gate."
            )
            ckpt.append(
                (
                    hf_name,
                    torch.ones_like(param)
                    if "scale" in name
                    else torch.zeros_like(param),
                )
            )
    ckpt.append(
        (
            "model.language_model.layers.1.mlp.gate.e_score_correction_bias",
            torch.zeros(4, dtype=torch.float32),
        )
    )
    return ckpt


def test_tiny_muse_nvfp4_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """ModelOpt NVFP4 init -> load_weights covers every routed expert tensor."""
    _use_cpu_platform(monkeypatch, request)
    qcfg_dict = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "exclude_modules": [
            "layers.0.",
            "*self_attn*",
            "lm_head",
            "*router*",
            "*pooler*",
            "*unpooler*",
        ],
    }
    cfg_dict = make_tiny_omni_config_dict()
    cfg_dict["quantization_config"] = dict(qcfg_dict)
    cfg_dict["text_config"]["quantization_config"] = dict(qcfg_dict)
    mdir = tmp_path / "nvfp4"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "config.json").write_text(json.dumps(cfg_dict))

    mc = ModelConfig(
        model=str(mdir),
        tokenizer_mode="skip",
        dtype=torch.bfloat16,
        max_model_len=4096,
    )
    TPUMuseSparkConfig.verify_and_update_model_config(mc)
    qcfg = VllmNvfp4Config.from_config(mc.hf_config.quantization_config)
    vc = VllmConfig(
        model_config=mc,
        cache_config=CacheConfig(block_size=128, cache_dtype="auto"),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device="cpu"),
        quant_config=qcfg,
    )
    VllmQuantConfig.set_configs(vc)

    with set_current_vllm_config(vc):
        model = initialize_model(
            vllm_config=vc, model_class=MuseSparkForConditionalGeneration
        )

    moe = model.model.layers[1].feed_forward
    assert isinstance(moe, TPUMuseSparkMoE)
    routed = moe.experts.routed_experts
    nvfp4_attrs = (
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
    )
    assert routed.w13_weight.dtype == torch.uint8
    for attr in nvfp4_attrs:
        assert getattr(routed, attr).numel() > 0, attr

    ckpt = _build_nvfp4_hf_weights_from_model(model)
    loaded = model.load_weights(ckpt)
    all_params = {name for name, _ in model.named_parameters()}
    assert all_params - loaded == set()


def test_quantization_guards_blockwise_fp8_and_quantized_qkv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Fail fast on blockwise FP8 and quantized self_attn.qkv_proj."""
    _use_cpu_platform(monkeypatch, request)
    block_fp8 = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    vc_fp8 = make_tiny_vllm_config(tmp_path / "bfp8", omni=True, quant_config=block_fp8)
    with (
        set_current_vllm_config(vc_fp8),
        pytest.raises(NotImplementedError, match="blockwise FP8"),
    ):
        MuseSparkForConditionalGeneration(vllm_config=vc_fp8)

    qcfg_dict = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "exclude_modules": ["lm_head", "*router*", "*pooler*", "*unpooler*"],
    }
    cfg_dict = make_tiny_omni_config_dict()
    cfg_dict["quantization_config"] = dict(qcfg_dict)
    cfg_dict["text_config"]["quantization_config"] = dict(qcfg_dict)
    mdir = tmp_path / "nvfp4_qkv"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "config.json").write_text(json.dumps(cfg_dict))
    mc = ModelConfig(
        model=str(mdir),
        tokenizer_mode="skip",
        dtype=torch.bfloat16,
        max_model_len=4096,
    )
    TPUMuseSparkConfig.verify_and_update_model_config(mc)
    vc_qkv = VllmConfig(
        model_config=mc,
        cache_config=CacheConfig(block_size=128, cache_dtype="auto"),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device="cpu"),
        quant_config=VllmNvfp4Config.from_config(mc.hf_config.quantization_config),
    )
    VllmQuantConfig.set_configs(vc_qkv)
    with (
        set_current_vllm_config(vc_qkv),
        pytest.raises(NotImplementedError, match="qkv_proj"),
    ):
        initialize_model(
            vllm_config=vc_qkv, model_class=MuseSparkForConditionalGeneration
        )


@pytest.mark.tpu_test
def test_tiny_muse_bf16_load_finalize_forward_tpu(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Single-chip TPU load -> finalize -> forward smoke test."""
    request.getfixturevalue("dist_ctx")
    vc = make_tiny_vllm_config(
        tmp_path,
        omni=True,
        device="tpu",
        hidden_size=128,
        intermediate_size=256,
        moe_hidden_size=128,
        moe_intermediate_size=128,
        num_local_experts=16,
    )
    with (
        torch.device("tpu"),
        set_default_torch_dtype(torch.bfloat16),
        set_current_vllm_config(vc),
    ):
        model = MuseSparkForConditionalGeneration(vllm_config=vc)
    weights = build_synthetic_hf_weights(vc.model_config.hf_config.text_config)
    model.load_weights((k, v.to("tpu")) for k, v in weights.items())
    process_weights_after_loading(model, vc.model_config, torch.device("tpu"))
    moe = model.model.layers[1].feed_forward
    assert isinstance(moe, TPUMuseSparkMoE)
    out = moe(torch.randn((4, moe.hidden_size), dtype=torch.bfloat16, device="tpu"))
    assert out.shape == (4, moe.hidden_size)
    assert bool(torch.isfinite(out).all().cpu())

    # Verify MuseSpark1xRMSNormWithInputScale traces under dynamic TPU shapes
    # (exercises Dynamo FakeTensor symbolic batch dims on device="tpu") and
    # lowers cleanly through static backend="tpu".
    post_norm = model.model.layers[1].post_ffn_norm
    compiled_dyn = torch.compile(
        lambda t: post_norm(t), backend="eager", dynamic=True, fullgraph=True
    )
    t4 = torch.randn((4, moe.hidden_size), dtype=torch.bfloat16, device="tpu")
    t7 = torch.randn((7, moe.hidden_size), dtype=torch.bfloat16, device="tpu")
    out_dyn4 = compiled_dyn(t4)
    out_dyn7 = compiled_dyn(t7)
    compiled_static = torch.compile(
        lambda t: post_norm(t), backend="tpu", fullgraph=True
    )
    out_static4 = compiled_static(t4)
    assert out_dyn4.shape == (4, moe.hidden_size)
    assert out_dyn7.shape == (7, moe.hidden_size)
    torch.testing.assert_close(out_static4.cpu(), out_dyn4.cpu(), rtol=1e-2, atol=1e-2)
    assert bool(torch.isfinite(out_dyn7).all().cpu())
