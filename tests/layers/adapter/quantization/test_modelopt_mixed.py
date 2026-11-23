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
"""Tests for ModelOpt MIXED_PRECISION dispatch and forward execution on TPU."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED, LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.quantization import (
    fp8,
    get_tpu_quantization_config,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.layers.adapter.quantization.modelopt_mixed import (
    VllmModelOptMixedConfig,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4LinearMethod,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import attach_online_fp8
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

LAYER = "model.layers.0"
# The base quantized_layers has no entries for this layer. Tests that need
# their own layout add entries for it.
LAYER1 = "model.layers.1"

# This copies Qwen3_5ForCausalLMBase.packed_modules_mapping, which maps each
# vLLM fused layer name to the checkpoint shard names quantized_layers uses.
PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}

pytestmark = pytest.mark.usefixtures("tensor_parallel_group")


def _nvfp4(group_size: int, algo: str = "NVFP4") -> dict:
    return {"quant_algo": algo, "group_size": group_size}


def _quant_config(extra_layers: dict | None = None) -> dict:
    quantized_layers = {
        f"{LAYER}.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
        f"{LAYER}.linear_attn.in_proj_z": {"quant_algo": "FP8"},
        f"{LAYER}.linear_attn.in_proj_b": {"quant_algo": "FP8"},
        f"{LAYER}.linear_attn.in_proj_a": {"quant_algo": "FP8"},
        f"{LAYER}.linear_attn.out_proj": {"quant_algo": "FP8"},
        f"{LAYER}.self_attn.q_proj": {"quant_algo": "FP8"},
        f"{LAYER}.self_attn.k_proj": {"quant_algo": "FP8"},
        f"{LAYER}.self_attn.v_proj": {"quant_algo": "FP8"},
        f"{LAYER}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
    }
    quantized_layers.update(extra_layers or {})
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": quantized_layers,
        "ignore": ["mtp*"],
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    }


def _fake_vllm_config(additional_config: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        additional_config=additional_config or {},
    )


def _make_config(extra_layers: dict | None = None) -> VllmModelOptMixedConfig:
    cfg = VllmModelOptMixedConfig.from_config(_quant_config(extra_layers))
    assert isinstance(cfg, VllmModelOptMixedConfig)
    cfg.packed_modules_mapping = dict(PACKED_MODULES)
    cfg.vllm_config = _fake_vllm_config()
    return cfg


def _linear(input_size: int = 64, output_size: int = 32) -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = input_size
    layer.output_size = output_size
    return layer


class _FakeRoutedExperts(RoutedExperts):
    def __init__(
        self,
        num_experts: int = 2,
        proj_names: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
    ) -> None:
        torch.nn.Module.__init__(self)
        self.ckpt_gate_proj_name, self.ckpt_up_proj_name, self.ckpt_down_proj_name = (
            proj_names
        )
        self.moe_config = SimpleNamespace(
            has_bias=False,
            is_act_and_mul=True,
            activation="silu",
            num_logical_experts=num_experts,
            moe_parallel_config=SimpleNamespace(use_ep=False),
        )


def _per_expert_layers(
    spec: dict,
    num_experts: int = 2,
    proj_names: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
) -> dict:
    return {
        f"{LAYER1}.mlp.experts.{e}.{p}": spec
        for e in range(num_experts)
        for p in proj_names
    }


def _cpu_quantized_matmul(x, w, scale, w8a16_max_rows=0):
    result = linear_common._quantized_matmul_jax(
        jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16),
        jnp.asarray(w.float().numpy(), dtype=jnp.float8_e4m3fn),
        jnp.asarray(scale.float().numpy()),
        w8a16_max_rows=w8a16_max_rows,
    )
    return torch.from_numpy(np.asarray(result.astype(jnp.float32)).copy()).to(x.dtype)


def _noop_weight_loader(*_args, **_kwargs) -> None:
    return None


# FP4 E2M1 magnitudes for codes 0-7. Bit 3 of a code is the sign.
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _random_nvfp4_weight(
    gen: torch.Generator, rows: int, cols: int, group_size: int, global_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a packed NVFP4 weight, its E4M3 block scales and its fp32 value."""
    codes = torch.randint(0, 16, (rows, cols), generator=gen, dtype=torch.uint8)
    # ModelOpt packs element 2j in the low nibble and 2j + 1 in the high one.
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    block_scale = (torch.rand(rows, cols // group_size, generator=gen) * 1.5 + 0.5).to(
        torch.float8_e4m3fn
    )
    sign = 1.0 - 2.0 * (codes >> 3).float()
    value = (
        sign
        * _E2M1_VALUES[(codes & 7).long()]
        * block_scale.float().repeat_interleave(group_size, dim=1)
        * global_scale
    )
    return packed, block_scale, value


class TestRegistration:
    def test_name_matches_vllm_detection(self):
        assert VllmModelOptMixedConfig.get_name() == "modelopt_mixed"

    def test_registered_under_tpu_name(self):
        assert get_quantization_config("tpu-modelopt_mixed") is VllmModelOptMixedConfig

    def test_platform_admits_method(self):
        assert "modelopt_mixed" in TpuPlatform.supported_quantization


def _resolve_from_model_dir(
    monkeypatch, model_dir, quantization_config, text_quantization_config=None
):
    from vllm.config import LoadConfig

    # get_tpu_quantization_config stores vllm_config on the class; restore it
    # so later tests don't see this test's stand-in config.
    monkeypatch.setattr(VllmModelOptMixedConfig, "vllm_config", None)
    text_config = (
        None
        if text_quantization_config is None
        else SimpleNamespace(quantization_config=text_quantization_config)
    )
    hf_config = SimpleNamespace(
        quantization_config=quantization_config,
        text_config=text_config,
    )
    model_config = SimpleNamespace(
        model=str(model_dir),
        dtype=torch.bfloat16,
        quantization="modelopt_mixed",
        quantization_config=None,
        hf_config=hf_config,
        hf_overrides={},
        trust_remote_code=False,
        revision=None,
        code_revision=None,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        load_config=LoadConfig(),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        additional_config={},
    )
    resolved = get_tpu_quantization_config(vllm_config)
    assert isinstance(resolved, VllmModelOptMixedConfig)
    return resolved, hf_config


class TestHfQuantConfigFallback:
    @pytest.mark.parametrize(
        ("quantization_config", "text_quantization_config"),
        [
            pytest.param({"quant_method": "modelopt"}, None, id="config_json"),
            pytest.param(None, {"quant_method": "modelopt"}, id="text_config"),
            pytest.param(
                {
                    "quant_method": "modelopt",
                    "quantization": {"quant_algo": "MIXED_PRECISION"},
                },
                None,
                id="nested_quantization_without_layers",
            ),
        ],
    )
    def test_hf_quant_config_json_fallback(
        self, tmp_path, monkeypatch, quantization_config, text_quantization_config
    ):
        hf_quant = {
            "producer": {"name": "modelopt", "version": "0.42.0"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    f"{LAYER}.linear_attn.out_proj": {"quant_algo": "FP8"},
                    f"{LAYER}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
                },
                "exclude_modules": ["mtp*"],
            },
        }
        (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
        resolved, _ = _resolve_from_model_dir(
            monkeypatch, tmp_path, quantization_config, text_quantization_config
        )
        method = resolved.get_quant_method(_linear(), f"{LAYER}.linear_attn.out_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)

    def test_config_json_with_quantized_layers_is_kept(self, tmp_path, monkeypatch):
        # No hf_quant_config.json exists, so the layer map can only come from
        # config.json, which must survive resolution.
        quantization_config = _quant_config()
        resolved, hf_config = _resolve_from_model_dir(
            monkeypatch, tmp_path, quantization_config
        )
        assert hf_config.quantization_config is quantization_config
        method = resolved.get_quant_method(_linear(), f"{LAYER}.linear_attn.out_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)

    def test_hf_quant_config_json_requires_modelopt_producer(self):
        hf_quant = {
            "producer": {"name": "other-tool"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    f"{LAYER}.linear_attn.out_proj": {"quant_algo": "FP8"}
                },
            },
        }
        with pytest.raises(ValueError, match="produced by modelopt"):
            VllmModelOptMixedConfig.from_config(hf_quant)

    def test_hf_quant_config_json_null_producer_raises(self):
        hf_quant = {
            "producer": None,
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    f"{LAYER}.linear_attn.out_proj": {"quant_algo": "FP8"}
                },
            },
        }
        with pytest.raises(ValueError, match="got producer None"):
            VllmModelOptMixedConfig.from_config(hf_quant)


class TestDispatch:
    @pytest.mark.parametrize(
        "prefix",
        [
            f"{LAYER}.linear_attn.out_proj",
            # Every checkpoint shard of these fused layers is FP8.
            f"{LAYER}.linear_attn.in_proj_qkvz",
            f"{LAYER}.linear_attn.in_proj_ba",
            f"{LAYER}.self_attn.qkv_proj",
        ],
    )
    def test_fp8_linear(self, prefix):
        cfg = _make_config()
        method = cfg.get_quant_method(_linear(), prefix)
        assert isinstance(method, VllmFp8LinearMethodTPU)
        # Fused layers load per-shard scales only through weight_loader_v2,
        # which vLLM enables by the method's class name.
        assert type(method).__name__ in WEIGHT_LOADER_V2_SUPPORTED
        assert method.prefix == prefix
        assert method.act_q_static
        assert not method.block_quant
        assert method.quant_config.is_checkpoint_fp8_serialized

    def test_nvfp4_experts_and_heterogeneous_group_sizes(self):
        cfg = _make_config(
            {
                f"{LAYER}.mlp.down": {"quant_algo": "W4A16_NVFP4", "group_size": 32},
                f"{LAYER}.mlp.gate_proj": {"quant_algo": "NVFP4", "group_size": 32},
            }
        )
        expert_method = cfg.get_quant_method(
            _FakeRoutedExperts(), f"{LAYER}.mlp.experts"
        )
        assert isinstance(expert_method, VllmNvfp4MoEMethod)
        assert expert_method.group_size == 16

        w4a16_method = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")
        assert isinstance(w4a16_method, VllmNvfp4LinearMethod)
        assert w4a16_method.group_size == 32

        nvfp4_32_method = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.gate_proj")
        assert isinstance(nvfp4_32_method, VllmNvfp4LinearMethod)
        assert nvfp4_32_method.group_size == 32

    @pytest.mark.parametrize(
        "prefix",
        [
            # These layers are in the checkpoint but absent from quantized_layers.
            f"{LAYER}.mlp.shared_expert.down_proj",
            f"{LAYER}.mlp.shared_expert.gate_up_proj",
            f"{LAYER}.mlp.gate",
            # This layer matches the "mtp*" ignore pattern.
            "mtp.layers.0.self_attn.o_proj",
        ],
    )
    def test_unquantized_linear(self, prefix):
        cfg = _make_config()
        method = cfg.get_quant_method(_linear(), prefix)
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_explicit_none_algo_is_unquantized(self):
        cfg = _make_config({f"{LAYER}.mlp.down": {"quant_algo": "NONE"}})
        method = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_partial_fused_projection_raises_value_error(self):
        cfg = _make_config(
            {
                f"{LAYER}.mlp.shared_expert.gate_proj": {"quant_algo": "FP8"},
                # Omitting up_proj leaves the fused gate_up_proj partly quantized.
            }
        )
        with pytest.raises(ValueError, match="Inconsistent quantization across"):
            cfg.get_quant_method(_linear(), f"{LAYER}.mlp.shared_expert.gate_up_proj")

    @pytest.mark.parametrize(
        ("extra_layers", "group_size"),
        [
            pytest.param(_per_expert_layers(_nvfp4(32)), 32, id="per_expert_keys"),
            pytest.param(
                {
                    f"{LAYER1}.mlp.gate_proj": _nvfp4(32),
                    f"{LAYER1}.mlp.up_proj": _nvfp4(32),
                    f"{LAYER1}.mlp.down_proj": _nvfp4(32),
                    # The experts ignore this entry because it is not directly
                    # under the MoE block.
                    f"{LAYER1}.mlp.shared_expert.down_proj": {"quant_algo": "FP8"},
                },
                32,
                id="expert_projections_under_moe_block",
            ),
            pytest.param(
                {
                    **_per_expert_layers(_nvfp4(16)),
                    f"{LAYER1}.mlp.experts.0.up_proj": {"quant_algo": "NVFP4"},
                    f"{LAYER1}.mlp.experts.1.down_proj": _nvfp4(16, "W4A16_NVFP4"),
                },
                16,
                id="nvfp4_family_per_expert_keys",
            ),
            pytest.param(
                {
                    f"{LAYER1}.mlp.gate_proj": _nvfp4(16),
                    f"{LAYER1}.mlp.up_proj": {"quant_algo": "NVFP4"},
                    f"{LAYER1}.mlp.down_proj": _nvfp4(16, "W4A16_NVFP4"),
                },
                16,
                id="nvfp4_family_expert_projections",
            ),
        ],
    )
    def test_experts_spec_from_projection_keys(self, extra_layers, group_size):
        cfg = _make_config(extra_layers)
        method = cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER1}.mlp.experts")
        assert isinstance(method, VllmNvfp4MoEMethod)
        assert method.group_size == group_size

    def test_experts_use_the_layer_checkpoint_projection_names(self):
        w_names = ("w1", "w3", "w2")
        cfg = _make_config(_per_expert_layers(_nvfp4(16), proj_names=w_names))
        method = cfg.get_quant_method(
            _FakeRoutedExperts(proj_names=w_names), f"{LAYER1}.mlp.experts"
        )
        assert isinstance(method, VllmNvfp4MoEMethod)

    def test_nvfp4_family_fused_projection(self):
        cfg = _make_config(
            {
                f"{LAYER1}.mlp.shared_expert.gate_proj": _nvfp4(16),
                f"{LAYER1}.mlp.shared_expert.up_proj": {"quant_algo": "W4A16_NVFP4"},
            }
        )
        method = cfg.get_quant_method(
            _linear(), f"{LAYER1}.mlp.shared_expert.gate_up_proj"
        )
        assert isinstance(method, VllmNvfp4LinearMethod)
        assert method.group_size == 16

    def test_missing_group_size_uses_nvfp4_default(self):
        # The first NVFP4 entry sets upstream's shared group_size to 32. A layer
        # without its own group_size must still get ModelOpt's default of 16.
        cfg = _make_config(
            {
                f"{LAYER}.mlp.experts": _nvfp4(32),
                f"{LAYER}.mlp.down": {"quant_algo": "NVFP4"},
            }
        )
        experts = cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER}.mlp.experts")
        assert isinstance(experts, VllmNvfp4MoEMethod)
        assert experts.group_size == 32
        linear = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")
        assert isinstance(linear, VllmNvfp4LinearMethod)
        assert linear.group_size == 16

    def test_missing_group_size_uses_top_level_group_size(self):
        raw = _quant_config({f"{LAYER}.mlp.down": {"quant_algo": "NVFP4"}})
        raw["group_size"] = 32
        cfg = VllmModelOptMixedConfig.from_config(raw)
        assert isinstance(cfg, VllmModelOptMixedConfig)
        cfg.packed_modules_mapping = dict(PACKED_MODULES)
        cfg.vllm_config = _fake_vllm_config()
        linear = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")
        assert isinstance(linear, VllmNvfp4LinearMethod)
        assert linear.group_size == 32
        # An entry's own group_size still wins over the top-level value.
        experts = cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER}.mlp.experts")
        assert isinstance(experts, VllmNvfp4MoEMethod)
        assert experts.group_size == 16

    def test_fused_shards_without_packed_modules_mapping(self):
        cfg = _make_config()
        cfg.packed_modules_mapping = {}
        method = cfg.get_quant_method(_linear(), f"{LAYER}.self_attn.qkv_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)

    @pytest.mark.parametrize(
        ("extra_layers", "match"),
        [
            pytest.param(
                {
                    **_per_expert_layers(_nvfp4(16)),
                    f"{LAYER1}.mlp.experts.1.gate_proj": {"quant_algo": "FP8"},
                },
                "Inconsistent quantization across",
                id="per_expert_keys_mixed_algo",
            ),
            pytest.param(
                {
                    f"{LAYER1}.mlp.gate_proj": _nvfp4(16),
                    f"{LAYER1}.mlp.up_proj": _nvfp4(16),
                    f"{LAYER1}.mlp.down_proj": _nvfp4(32),
                },
                "Inconsistent quantization across",
                id="moe_block_projections_mixed_group_size",
            ),
            # Each case below leaves expert weights out of quantized_layers.
            # Those weights are unquantized, so the experts cannot use one
            # quantized method.
            pytest.param(
                {
                    f"{LAYER1}.mlp.gate_proj": _nvfp4(16),
                    f"{LAYER1}.mlp.up_proj": _nvfp4(16),
                },
                "Inconsistent quantization across",
                id="moe_block_projections_missing_down_proj",
            ),
            pytest.param(
                _per_expert_layers(_nvfp4(16), proj_names=("gate_proj", "up_proj")),
                "Inconsistent quantization across",
                id="per_expert_keys_missing_down_proj",
            ),
            pytest.param(
                _per_expert_layers(_nvfp4(16), num_experts=1),
                "Inconsistent quantization across",
                id="per_expert_keys_missing_expert",
            ),
            pytest.param(
                _per_expert_layers(_nvfp4(16), proj_names=("w1", "w3", "w2")),
                "do not match its expert projections",
                id="per_expert_keys_with_other_projection_names",
            ),
        ],
    )
    def test_inconsistent_expert_specs_raise_value_error(self, extra_layers, match):
        cfg = _make_config(extra_layers)
        with pytest.raises(ValueError, match=match):
            cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER1}.mlp.experts")

    def test_quantize_on_load_prefix_wraps_unquantized_linear(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "shared_expert.")
        cfg = _make_config()
        method = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.shared_expert.down_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)
        assert not method.quant_config.is_checkpoint_fp8_serialized
        assert not method.online_fp8

    def test_online_fp8_policy_wraps_unquantized_linear(self):
        cfg = _make_config()
        vllm_cfg = _fake_vllm_config(
            additional_config={
                "tpu_online_fp8": {
                    "enabled": True,
                    "targets": ["*shared_expert*"],
                    "weight_scheme": "per_channel",
                }
            }
        )
        attach_online_fp8(cfg, vllm_cfg)
        method = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.shared_expert.down_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)
        assert method.online_fp8

    def test_weights_mapper_renames_layer_keys_and_online_fp8_targets(self):
        from vllm.model_executor.models.utils import WeightsMapper

        raw = _quant_config(
            {
                key.replace("model.", "backbone.", 1): spec
                for key, spec in _per_expert_layers(_nvfp4(32)).items()
            }
        )
        cfg = VllmModelOptMixedConfig.from_config(raw)
        assert isinstance(cfg, VllmModelOptMixedConfig)
        cfg.packed_modules_mapping = dict(PACKED_MODULES)
        cfg.vllm_config = _fake_vllm_config()
        shared_down = f"{LAYER1}.mlp.shared_expert.down_proj"
        attach_online_fp8(
            cfg,
            _fake_vllm_config(
                additional_config={
                    "tpu_online_fp8": {
                        "enabled": True,
                        "targets": [shared_down.replace("model.", "backbone.", 1)],
                        "weight_scheme": "per_channel",
                    }
                }
            ),
        )
        cfg.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"backbone.": "model."}))

        assert cfg.online_fp8_policy.targets == [shared_down]
        online = cfg.get_quant_method(_linear(), shared_down)
        assert isinstance(online, VllmFp8LinearMethodTPU)
        assert online.online_fp8
        experts = cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER1}.mlp.experts")
        assert isinstance(experts, VllmNvfp4MoEMethod)
        assert experts.group_size == 32

    def test_excluded_experts_are_unquantized(self):
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        cfg = _make_config()
        # The CPU test image has no "tpu" torch device, so name the device.
        with set_current_vllm_config(
            VllmConfig(device_config=DeviceConfig(device="cpu"))
        ):
            method = cfg.get_quant_method(
                _FakeRoutedExperts(), "mtp.layers.0.mlp.experts"
            )
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)

    def test_attention_and_mla_attention_have_no_method(self):
        cfg = _make_config()
        attn = Attention.__new__(Attention)
        torch.nn.Module.__init__(attn)
        assert cfg.get_quant_method(attn, f"{LAYER}.self_attn.attn") is None

        mla = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(mla)
        assert cfg.get_quant_method(mla, f"{LAYER}.self_attn.mla_attn") is None

    def test_unsupported_linear_algo_raises(self):
        cfg = _make_config({f"{LAYER}.mlp.down": {"quant_algo": "MXFP8"}})
        with pytest.raises(NotImplementedError, match="MXFP8"):
            cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")

    def test_unsupported_experts_algo_raises(self):
        cfg = _make_config({f"{LAYER}.mlp.experts": {"quant_algo": "FP8"}})
        with pytest.raises(NotImplementedError, match="FP8"):
            cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER}.mlp.experts")


class TestForwardPass:
    def test_unquantized_and_fp8_linear_forward_cpu(self, monkeypatch):
        monkeypatch.setattr(fp8, "quantized_matmul", _cpu_quantized_matmul)
        cfg = _make_config()
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(4, 64, generator=gen).to(torch.bfloat16)

        unq_layer = _linear()
        unq_method = cfg.get_quant_method(
            unq_layer, f"{LAYER}.mlp.shared_expert.down_proj"
        )
        assert isinstance(unq_method, VllmUnquantizedLinearMethod)
        unq_method.create_weights(
            unq_layer,
            input_size_per_partition=64,
            output_partition_sizes=[32],
            input_size=64,
            output_size=32,
            params_dtype=torch.bfloat16,
            weight_loader=_noop_weight_loader,
        )
        w_bf16 = torch.randn(32, 64, generator=gen).to(torch.bfloat16)
        unq_layer.weight.data.copy_(w_bf16)
        unq_method.process_weights_after_loading(unq_layer)
        out_unq = unq_method.apply(unq_layer, x)
        torch.testing.assert_close(out_unq, x @ w_bf16.T, rtol=1e-2, atol=1e-2)

        # The two partitions get different weight scales so the test covers
        # how a fused layer's per-partition weight_scale is reduced.
        fp8_layer = _linear(input_size=64, output_size=32)
        fp8_method = cfg.get_quant_method(
            fp8_layer, f"{LAYER}.linear_attn.in_proj_qkvz"
        )
        assert isinstance(fp8_method, VllmFp8LinearMethodTPU)
        fp8_method.create_weights(
            fp8_layer,
            input_size_per_partition=64,
            output_partition_sizes=[16, 16],
            input_size=64,
            output_size=32,
            params_dtype=torch.bfloat16,
            weight_loader=_noop_weight_loader,
        )
        w0 = torch.randn(16, 64, generator=gen) * 0.2
        w1 = torch.randn(16, 64, generator=gen) * 0.4
        s0 = w0.abs().amax() / 448.0
        s1 = w1.abs().amax() / 448.0
        fp8_layer.weight.data[:16].copy_((w0 / s0).to(torch.float8_e4m3fn))
        fp8_layer.weight.data[16:].copy_((w1 / s1).to(torch.float8_e4m3fn))
        fp8_layer.weight_scale.data[0] = s0
        fp8_layer.weight_scale.data[1] = s1
        # The checkpoint's static input_scale loads but is unused at runtime.
        assert hasattr(fp8_layer, "input_scale")
        fp8_method.process_weights_after_loading(fp8_layer)
        assert not hasattr(fp8_layer, "input_scale")

        w_ref = torch.cat([w0, w1], dim=0)
        out_fp8 = fp8_method.apply(fp8_layer, x)
        assert out_fp8.shape == (4, 32)
        assert out_fp8.dtype == torch.bfloat16
        ref = x.float() @ w_ref.T
        rel_err = (out_fp8.float() - ref).norm() / ref.norm()
        assert rel_err < 0.08

    @pytest.mark.tpu_test
    def test_fp8_linear_forward_tpu(self):
        if jax.default_backend() != "tpu":
            pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")
        device = torch.device("tpu:0")
        cfg = _make_config()
        gen = torch.Generator().manual_seed(11)
        x = torch.randn(8, 128, generator=gen).to(dtype=torch.bfloat16, device=device)

        fp8_layer = _linear(input_size=128, output_size=256)
        fp8_method = cfg.get_quant_method(fp8_layer, f"{LAYER}.linear_attn.out_proj")
        assert isinstance(fp8_method, VllmFp8LinearMethodTPU)
        with torch.device(device):
            fp8_method.create_weights(
                fp8_layer,
                input_size_per_partition=128,
                output_partition_sizes=[256],
                input_size=128,
                output_size=256,
                params_dtype=torch.bfloat16,
                weight_loader=_noop_weight_loader,
            )
        w_ref = torch.randn(256, 128, generator=gen) * 0.25
        w_scale = w_ref.abs().amax() / 448.0
        fp8_layer.weight.data.copy_(
            (w_ref / w_scale).to(dtype=torch.float8_e4m3fn, device=device)
        )
        fp8_layer.weight_scale.data.fill_(w_scale.item())
        fp8_layer.input_scale.data.fill_(1.0)
        fp8_method.process_weights_after_loading(fp8_layer)

        out_fp8 = fp8_method.apply(fp8_layer, x).cpu().float()
        ref = x.cpu().float() @ w_ref.T
        rel_err = (out_fp8 - ref).norm() / ref.norm()
        assert rel_err < 0.08

    @pytest.mark.tpu_test
    def test_nvfp4_family_fused_linear_forward_tpu(self):
        if jax.default_backend() != "tpu":
            pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()!r}")
        device = torch.device("tpu:0")
        group_size, k, n = 16, 256, 128
        prefix = f"{LAYER1}.mlp.shared_expert"
        cfg = _make_config(
            {
                f"{prefix}.gate_proj": _nvfp4(group_size),
                f"{prefix}.up_proj": _nvfp4(group_size, "W4A16_NVFP4"),
            }
        )
        layer = _linear(input_size=k, output_size=2 * n)
        method = cfg.get_quant_method(layer, f"{prefix}.gate_up_proj")
        assert isinstance(method, VllmNvfp4LinearMethod)
        with torch.device(device):
            method.create_weights(
                layer,
                input_size_per_partition=k,
                output_partition_sizes=[n, n],
                input_size=k,
                output_size=2 * n,
                params_dtype=torch.bfloat16,
                weight_loader=_noop_weight_loader,
            )

        # ModelOpt shares one global scale across the shards of a fused layer.
        gen = torch.Generator().manual_seed(13)
        global_scale = 0.03
        shards = [
            _random_nvfp4_weight(gen, n, k, group_size, global_scale) for _ in range(2)
        ]
        layer.weight.data.copy_(torch.cat([w for w, _, _ in shards]).to(device))
        layer.weight_scale.data.copy_(torch.cat([s for _, s, _ in shards]).to(device))
        layer.weight_scale_2.data.fill_(global_scale)
        # Only the NVFP4 shard has an input_scale in the checkpoint, so the
        # W4A16_NVFP4 shard's entry is never loaded.
        input_scale = layer.input_scale.data.cpu()
        input_scale[0] = 1.0
        layer.input_scale.data.copy_(input_scale.to(device))
        method.process_weights_after_loading(layer)
        assert not hasattr(layer, "input_scale")
        assert not hasattr(layer, "weight_scale_2")

        x = torch.randn(8, k, generator=gen).to(torch.bfloat16)
        out = method.apply(layer, x.to(device)).cpu().float()
        w_ref = torch.cat([value for _, _, value in shards], dim=0)
        ref = x.float() @ w_ref.T
        rel_err = (out - ref).norm() / ref.norm()
        assert rel_err < 0.02
