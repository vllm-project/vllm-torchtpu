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
"""Unit and TPU tests for Mamba3ForCausalLM, Mamba3Model, and Mamba3Config."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, PretrainedConfig
from vllm.config import (
    CacheConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.interfaces import SupportsMambaPrefixCaching
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import get_config
from vllm.v1.kv_cache_interface import MambaSpec

from tests.distributed.tpu_test_utils import run_in_isolated_process
from tests.layers.adapter.mamba_test_utils import (
    MAMBA3_GOLDEN_PROMPT_IDS,
    MAMBA3_MIMO_UPSTREAM_GOLDEN_GREEDY,
    MAMBA3_MIMO_UPSTREAM_GOLDEN_LOGITS,
    MAMBA3_SISO_UPSTREAM_GOLDEN_GREEDY,
    MAMBA3_SISO_UPSTREAM_GOLDEN_LOGITS,
    eval_hf_mamba2_reference,
    eval_torch_mamba3_reference,
    jax_cpu_mesh,
    make_forward_context,
    make_mamba_attn_metadata,
    make_mock_mesh,
    passthrough_jax_op,
    setup_cpu_platform_for_dist,
    write_synthetic_mamba2_checkpoint,
    write_synthetic_mamba3_checkpoint,
)
from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op import VllmMambaMixer3
from vllm_torchtpu.layers.adapter.mamba_mixer3 import MambaMixer3
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.mamba3.config import (
    Mamba3Config,
    TPUMamba3Config,
)
from vllm_torchtpu.models.vllm.mamba3.mamba3 import (
    Mamba3ForCausalLM,
    Mamba3GatedMLP,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture(autouse=True)
def _isolate_mamba3_global_registries():
    """Save and restore global vLLM / HF model and config-parser registries."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.transformers_utils import config as vllm_tf_cfg

    from vllm_torchtpu.models.vllm.mamba3 import config as mamba3_cfg_mod

    saved_models = dict(ModelRegistry.models)
    saved_config_map = dict(MODELS_CONFIG_MAP)
    saved_parsers = dict(vllm_tf_cfg._CONFIG_FORMAT_TO_CONFIG_PARSER)
    saved_flag = mamba3_cfg_mod._MAMBA3_PARSER_REGISTERED
    had_auto_cfg = "mamba3" in CONFIG_MAPPING._extra_content
    saved_auto_cfg = CONFIG_MAPPING._extra_content.get("mamba3")
    try:
        yield
    finally:
        ModelRegistry.models.clear()
        ModelRegistry.models.update(saved_models)
        MODELS_CONFIG_MAP.clear()
        MODELS_CONFIG_MAP.update(saved_config_map)
        vllm_tf_cfg._CONFIG_FORMAT_TO_CONFIG_PARSER.clear()
        vllm_tf_cfg._CONFIG_FORMAT_TO_CONFIG_PARSER.update(saved_parsers)
        mamba3_cfg_mod._MAMBA3_PARSER_REGISTERED = saved_flag
        if had_auto_cfg:
            CONFIG_MAPPING._extra_content["mamba3"] = saved_auto_cfg
        else:
            CONFIG_MAPPING._extra_content.pop("mamba3", None)


@pytest.fixture(autouse=True)
def _cpu_platform_for_unit_tests(request, monkeypatch):
    """Pin `current_platform` to CPU before `dist_ctx` with clean teardown."""
    if request.node.get_closest_marker("tpu_test") is not None:
        return
    setup_cpu_platform_for_dist(request, monkeypatch)


def _make_vllm_config(
    hf_config: Mamba3Config,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> VllmConfig:
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        architecture="Mamba3ForCausalLM",
        dtype=dtype,
        head_dtype=None,
        max_model_len=256,
        quantization=None,
    )
    vllm_config.cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_block_size=256,
        mamba_page_size_padded=None,
        use_replayssm=False,
    )
    vllm_config.speculative_config = None
    vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        pipeline_parallel_size=pp_size,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    vllm_config.compilation_config.mode = CompilationMode.NONE
    return vllm_config


def _build_mamba3_model(
    hf_config: Mamba3Config,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_group: Any = None,
    pp_layer_range: tuple[int, int] | None = None,
    use_real_jax_op: bool = False,
) -> tuple[Mamba3ForCausalLM, VllmConfig]:
    pp_size = 2 if pp_group is not None else 1
    vllm_config = _make_vllm_config(hf_config, tp_size=tp_size, pp_size=pp_size)
    mesh = jax_cpu_mesh() if use_real_jax_op else make_mock_mesh()

    with ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(vllm_config))
        stack.enter_context(
            set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_config)
        )
        if pp_group is not None:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
                    return_value=pp_group,
                )
            )
        if pp_layer_range is not None:
            start_l, end_l = pp_layer_range

            def _fake_make_layers(num_hidden_layers: int, layer_fn: Any, prefix: str):
                layers = torch.nn.ModuleList(
                    [
                        layer_fn(prefix=f"{prefix}.{idx}")
                        if start_l <= idx < end_l
                        else PPMissingLayer()
                        for idx in range(num_hidden_layers)
                    ]
                )
                return start_l, end_l, layers

            stack.enter_context(
                patch(
                    "vllm_torchtpu.models.vllm.mamba3.mamba3.make_layers",
                    side_effect=_fake_make_layers,
                )
            )
        if tp_size > 1:
            for target in (
                "vllm_torchtpu.layers.adapter.mamba_mixer3.get_tensor_model_parallel_world_size",
                "vllm_torchtpu.models.vllm.mamba3.mamba3.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            ):
                stack.enter_context(patch(target, return_value=tp_size))
            for target in (
                "vllm_torchtpu.layers.adapter.mamba_mixer3.get_tensor_model_parallel_rank",
                "vllm_torchtpu.models.vllm.mamba3.mamba3.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            ):
                stack.enter_context(patch(target, return_value=tp_rank))
        if use_real_jax_op:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                    side_effect=passthrough_jax_op,
                )
            )
        else:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                    return_value=MagicMock(),
                )
            )

        model = Mamba3ForCausalLM(vllm_config=vllm_config)
        if use_real_jax_op:
            for decoder_layer in model.backbone.layers:
                if not isinstance(decoder_layer, PPMissingLayer):
                    decoder_layer.mixer.mamba_op = decoder_layer.mixer._build_mamba_op(
                        interpret=True
                    )

    return model, vllm_config


class TestMamba3RegistrationAndConfig:
    def test_mamba3_model_and_config_map_registration(self):
        register_models()
        entry = ModelRegistry.models.get("Mamba3ForCausalLM")
        assert entry is not None, "Mamba3ForCausalLM is not registered"
        assert entry.load_model_cls() is Mamba3ForCausalLM
        assert MODELS_CONFIG_MAP.get("Mamba3ForCausalLM") is TPUMamba3Config

    def test_dedicated_mamba3_config_format_leaves_hf_parser_untouched(self, tmp_path):
        from vllm.transformers_utils.config import HFConfigParser, get_config_parser

        register_models()
        assert type(get_config_parser("hf")) is HFConfigParser
        assert type(get_config_parser("mamba3")).__name__ == "_Mamba3ConfigParser"

        siso_dir = tmp_path / "mamba3-siso-1.5b"
        siso_dir.mkdir()
        siso_json = {
            "d_model": 2048,
            "d_intermediate": 4096,
            "n_layer": 24,
            "vocab_size": 128256,
            "ssm_cfg": {
                "layer": "Mamba3",
                "d_state": 128,
                "expand": 2,
                "headdim": 64,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "A_floor": 0.0001,
                "chunk_size": 64,
                "is_mimo": False,
                "mimo_rank": 4,
                "is_outproj_norm": False,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "tie_embeddings": True,
            "pad_vocab_size_multiple": 16,
        }
        (siso_dir / "config.json").write_text(json.dumps(siso_json))

        raw_dict, _ = PretrainedConfig.get_config_dict(str(siso_dir))
        assert "model_type" not in raw_dict
        assert "architectures" not in raw_dict

        # Dedicated `config_format="mamba3"` parses raw `state-spaces/mamba3-*`,
        # and callable `hf_overrides` runs exactly once inside `get_config`.
        override_calls: list[int] = []

        def _bump_layers(cfg: PretrainedConfig) -> PretrainedConfig:
            override_calls.append(1)
            cfg.num_hidden_layers = 26
            return cfg

        siso_cfg = get_config(
            str(siso_dir),
            trust_remote_code=False,
            config_format="mamba3",
            hf_overrides_fn=_bump_layers,
        )
        assert len(override_calls) == 1
        assert isinstance(siso_cfg, Mamba3Config)
        assert siso_cfg.model_type == "mamba3"
        assert siso_cfg.architectures == ["Mamba3ForCausalLM"]
        assert siso_cfg.hidden_size == 2048
        assert siso_cfg.num_hidden_layers == 26
        assert siso_cfg.d_intermediate == 4096
        assert siso_cfg.num_heads == 64
        assert siso_cfg.head_dim == 64
        assert siso_cfg.state_size == 128
        assert siso_cfg.is_mimo is False
        assert siso_cfg.mimo_rank == 1

        # Explicit `model_type="mamba3"` works directly via `AutoConfig` / `"auto"`.
        explicit_dir = tmp_path / "mamba3-explicit"
        explicit_dir.mkdir()
        (explicit_dir / "config.json").write_text(
            json.dumps(dict(siso_json, model_type="mamba3"))
        )
        assert isinstance(AutoConfig.from_pretrained(str(explicit_dir)), Mamba3Config)

        unaligned = Mamba3Config(vocab_size=100, pad_vocab_size_multiple=16)
        assert unaligned.org_vocab_size == 100
        assert unaligned.vocab_size == 112
        roundtrip = Mamba3Config.from_dict(unaligned.to_dict())
        assert roundtrip.org_vocab_size == 100
        assert roundtrip.vocab_size == 112

        mimo_dir = tmp_path / "mamba3-mimo-1.5b"
        mimo_dir.mkdir()
        mimo_json = dict(siso_json, d_intermediate=3824)
        mimo_json["ssm_cfg"] = dict(
            siso_json["ssm_cfg"], is_mimo=True, mimo_rank=4, chunk_size=16
        )
        (mimo_dir / "config.json").write_text(json.dumps(mimo_json))
        mimo_cfg = get_config(
            str(mimo_dir), trust_remote_code=False, config_format="mamba3"
        )
        assert isinstance(mimo_cfg, Mamba3Config)
        assert mimo_cfg.is_mimo is True
        assert mimo_cfg.mimo_rank == 4
        assert mimo_cfg.d_intermediate == 3824

        # Nested `ssm_cfg` overrides preserve sibling keys (`is_mimo`, `mimo_rank`,
        # `headdim`) and re-sync derived attributes across `get_config`, `update`,
        # and `ModelConfig._apply_dict_overrides` (`setattr(cfg, "ssm_cfg", ...)`):
        mimo_overridden = get_config(
            str(mimo_dir),
            trust_remote_code=False,
            config_format="mamba3",
            hf_overrides_kw={"ssm_cfg": {"d_state": 64}},
        )
        assert isinstance(mimo_overridden, Mamba3Config)
        assert mimo_overridden.state_size == 64
        assert mimo_overridden.ssm_cfg["d_state"] == 64
        assert mimo_overridden.is_mimo is True
        assert mimo_overridden.mimo_rank == 4
        assert mimo_overridden.head_dim == 64
        assert mimo_overridden.num_heads == 64

        # Simulate `ModelConfig._apply_dict_overrides` (`setattr` on `ssm_cfg`):
        mimo_setattr = Mamba3Config.from_dict(mimo_cfg.to_dict())
        mimo_setattr.ssm_cfg = {"d_state": 32, "rope_fraction": 1.0}
        assert mimo_setattr.state_size == 32
        assert mimo_setattr.rope_fraction == 1.0
        assert mimo_setattr.is_mimo is True
        assert mimo_setattr.mimo_rank == 4
        assert mimo_setattr.head_dim == 64
        assert mimo_setattr.ssm_cfg["is_mimo"] is True
        assert mimo_setattr.ssm_cfg["mimo_rank"] == 4

    def test_engine_startup_vllm_config_initializes_mamba_block_size_and_validates(
        self, tmp_path
    ):
        register_models()
        ckpt_dir = tmp_path / "mamba3_startup"
        write_synthetic_mamba3_checkpoint(ckpt_dir, is_mimo=False)

        model_cfg = ModelConfig(
            model=str(ckpt_dir),
            tokenizer=str(ckpt_dir),
            config_format="mamba3",
            max_model_len=128,
            dtype="bfloat16",
        )
        vllm_cfg = VllmConfig(
            model_config=model_cfg,
            cache_config=CacheConfig(enable_prefix_caching=False),
            device_config=DeviceConfig(device="cpu"),
        )
        vllm_cfg.compilation_config.mode = CompilationMode.NONE
        assert vllm_cfg.cache_config.mamba_cache_mode == "none"
        assert vllm_cfg.cache_config.mamba_block_size == 128

        with (
            set_current_vllm_config(vllm_cfg),
            set_vllm_model_wrapper_context(mesh=make_mock_mesh(), vllm_config=vllm_cfg),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=MagicMock(),
            ),
        ):
            model = Mamba3ForCausalLM(vllm_config=vllm_cfg)
            spec = model.backbone.layers[0].mixer.get_kv_cache_spec(vllm_cfg)

        assert isinstance(spec, MambaSpec)
        assert spec.shapes == Mamba3ForCausalLM.get_mamba_state_shape_from_config(
            vllm_cfg
        )
        assert spec.dtypes == Mamba3ForCausalLM.get_mamba_state_dtype_from_config(
            vllm_cfg
        )

        # Prefix caching, quantization, DCP > 1, and replayssm checks:
        with pytest.raises(NotImplementedError, match=r"Prefix caching.*not yet"):
            VllmConfig(
                model_config=model_cfg,
                cache_config=CacheConfig(enable_prefix_caching=True),
                device_config=DeviceConfig(device="cpu"),
            )

        quant_check_cfg = _make_vllm_config(Mamba3Config())
        quant_check_cfg.model_config.quantization = "fp8"
        quant_check_cfg.model_config.is_hybrid = False
        quant_check_cfg.model_config.max_model_len = 128
        with pytest.raises(NotImplementedError, match="Quantized weights"):
            TPUMamba3Config.verify_and_update_config(quant_check_cfg)

        from vllm_torchtpu.layers.adapter.quantization.unquantized import (
            VllmUnquantizedConfig,
        )

        unquant_check_cfg = _make_vllm_config(Mamba3Config())
        unquant_check_cfg.model_config.is_hybrid = False
        unquant_check_cfg.model_config.max_model_len = 128
        unquant_check_cfg.quant_config = VllmUnquantizedConfig()
        TPUMamba3Config.verify_and_update_config(unquant_check_cfg)

        bad_qcfg = _make_vllm_config(Mamba3Config())
        bad_qcfg.model_config.is_hybrid = False
        bad_qcfg.model_config.max_model_len = 128
        bad_qcfg.quant_config = MagicMock()
        with pytest.raises(NotImplementedError, match="Quantized weights"):
            TPUMamba3Config.verify_and_update_config(bad_qcfg)

        dcp_check_cfg = _make_vllm_config(Mamba3Config())
        dcp_check_cfg.model_config.is_hybrid = False
        dcp_check_cfg.parallel_config.decode_context_parallel_size = 2
        with pytest.raises(NotImplementedError, match="Decode context parallelism"):
            TPUMamba3Config.verify_and_update_config(dcp_check_cfg)

        replay_check_cfg = _make_vllm_config(Mamba3Config())
        replay_check_cfg.model_config.is_hybrid = False
        replay_check_cfg.cache_config.use_replayssm = True
        with pytest.raises(NotImplementedError, match="--use-replayssm"):
            TPUMamba3Config.verify_and_update_config(replay_check_cfg)

    def test_state_shape_and_dtype_match_mixer_layer(self):
        for is_mimo in (False, True):
            cfg = Mamba3Config(
                d_model=64,
                n_layer=2,
                vocab_size=256,
                d_intermediate=112 if is_mimo else 128,
                ssm_cfg={
                    "layer": "Mamba3",
                    "d_state": 16,
                    "expand": 2,
                    "headdim": 16,
                    "ngroups": 1,
                    "rope_fraction": 0.5,
                    "is_mimo": is_mimo,
                    "mimo_rank": 4,
                },
            )
            model, vllm_cfg = _build_mamba3_model(cfg)
            mixer = model.backbone.layers[0].mixer
            assert isinstance(mixer, VllmMambaMixer3)
            assert (
                Mamba3ForCausalLM.get_mamba_state_shape_from_config(vllm_cfg)
                == mixer.get_state_shape()
            )
            assert (
                Mamba3ForCausalLM.get_mamba_state_dtype_from_config(vllm_cfg)
                == mixer.get_state_dtype()
            )

        assert SupportsMambaPrefixCaching not in Mamba3ForCausalLM.__mro__
        assert not issubclass(Mamba3ForCausalLM, SupportsMambaPrefixCaching)
        assert not hasattr(Mamba3ForCausalLM, "get_mamba_state_copy_func")


class TestMamba3ArchitectureAndWeights:
    def test_decoder_layer_variants_padded_vocab_masking_and_quant_rejection(self):
        cfg_no_mlp = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=100,
            pad_vocab_size_multiple=16,
            d_intermediate=0,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
            },
        )
        model_no_mlp, vllm_cfg = _build_mamba3_model(cfg_no_mlp)
        assert model_no_mlp.backbone.layers[0].mlp is None
        assert model_no_mlp.backbone.layers[0].norm2 is None
        assert cfg_no_mlp.org_vocab_size == 100
        assert cfg_no_mlp.vocab_size == 112
        assert model_no_mlp.backbone.embedding.org_vocab_size == 112

        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            model_no_mlp.lm_head.weight.copy_(
                torch.randn(model_no_mlp.lm_head.weight.shape, generator=gen)
            )
        h_sample = torch.randn((5, 64), generator=gen)
        logits = model_no_mlp.compute_logits(h_sample)
        assert logits is not None and logits.shape == (5, 112)
        assert torch.isfinite(logits[:, :100]).all()
        assert torch.all(logits[:, 100:] == float("-inf"))

        compiled_logits_fn = torch.compile(
            model_no_mlp.compute_logits, backend="eager", fullgraph=True
        )
        logits_compiled = compiled_logits_fn(h_sample)
        assert logits_compiled is not None and logits_compiled.shape == (5, 112)
        torch.testing.assert_close(logits_compiled[:, :100], logits[:, :100])
        assert torch.all(logits_compiled[:, 100:] == float("-inf"))

        from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config

        fp8_quant = VllmFp8Config()
        with set_current_vllm_config(vllm_cfg):
            with pytest.raises(NotImplementedError, match="Quantized weights"):
                MambaMixer3(
                    hidden_size=64,
                    ssm_state_size=16,
                    intermediate_size=128,
                    num_heads=8,
                    head_dim=16,
                    mimo_rank=1,
                    quant_config=fp8_quant,
                    prefix="layers.quant.mixer",
                )
            with pytest.raises(NotImplementedError, match="Quantized weights"):
                Mamba3GatedMLP(
                    hidden_size=64,
                    intermediate_size=128,
                    quant_config=fp8_quant,
                )

    def test_vllm_unquantized_config_supported_and_fp8_on_load_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """TPUModelRunner sets VllmUnquantizedConfig even for unquantized models."""
        from vllm_torchtpu.layers.adapter.quantization.unquantized import (
            VllmUnquantizedConfig,
            VllmUnquantizedLinearMethod,
        )

        cfg = Mamba3Config(
            d_model=64,
            n_layer=1,
            vocab_size=128,
            d_intermediate=96,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "is_mimo": False,
                "mimo_rank": 1,
            },
        )
        vllm_cfg = _make_vllm_config(cfg)
        vllm_cfg.quant_config = VllmUnquantizedConfig()
        mesh = make_mock_mesh()
        with (
            set_current_vllm_config(vllm_cfg),
            set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_cfg),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op."
                "VllmMambaMixer3._build_mamba_op",
                return_value=MagicMock(),
            ),
        ):
            model = Mamba3ForCausalLM(vllm_config=vllm_cfg)

        layer0 = model.backbone.layers[0]
        assert isinstance(layer0.mlp, Mamba3GatedMLP)
        for linear in (
            layer0.mixer.in_proj,
            layer0.mixer.out_proj,
            layer0.mlp.fc1,
            layer0.mlp.fc2,
        ):
            assert isinstance(linear.quant_method, VllmUnquantizedLinearMethod)

        # Load synthetic weights, run VllmUnquantizedLinearMethod weight transpose,
        # and verify MLP & mixer projections match unquantized F.linear exactly.
        torch.manual_seed(42)
        in_w = torch.randn_like(layer0.mixer.in_proj.weight)
        out_w = torch.randn_like(layer0.mixer.out_proj.weight)
        fc1_w = torch.randn_like(layer0.mlp.fc1.weight)
        fc2_w = torch.randn_like(layer0.mlp.fc2.weight)
        with torch.no_grad():
            layer0.mixer.in_proj.weight.copy_(in_w)
            layer0.mixer.out_proj.weight.copy_(out_w)
            layer0.mlp.fc1.weight.copy_(fc1_w)
            layer0.mlp.fc2.weight.copy_(fc2_w)
            for linear in (
                layer0.mixer.in_proj,
                layer0.mixer.out_proj,
                layer0.mlp.fc1,
                layer0.mlp.fc2,
            ):
                linear.quant_method.process_weights_after_loading(linear)

            x = torch.randn(4, 64)
            got_in, _ = layer0.mixer.in_proj(x)
            torch.testing.assert_close(got_in, torch.nn.functional.linear(x, in_w))
            y = torch.randn(4, 128)
            got_out, _ = layer0.mixer.out_proj(y)
            torch.testing.assert_close(got_out, torch.nn.functional.linear(y, out_w))
            got_fc1, _ = layer0.mlp.fc1(x)
            torch.testing.assert_close(got_fc1, torch.nn.functional.linear(x, fc1_w))
            got_fc2, _ = layer0.mlp.fc2(y)
            torch.testing.assert_close(got_fc2, torch.nn.functional.linear(y, fc2_w))

        # On-load FP8 via QUANTIZE_ON_LOAD_PREFIXES must still be rejected.
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "layers.fp8")
        with set_current_vllm_config(vllm_cfg):
            with pytest.raises(NotImplementedError, match="Quantized weights"):
                MambaMixer3(
                    hidden_size=64,
                    ssm_state_size=16,
                    intermediate_size=128,
                    num_heads=8,
                    head_dim=16,
                    mimo_rank=1,
                    quant_config=VllmUnquantizedConfig(),
                    prefix="layers.fp8.mixer",
                )
            with pytest.raises(NotImplementedError, match="Quantized weights"):
                Mamba3GatedMLP(
                    hidden_size=64,
                    intermediate_size=128,
                    quant_config=VllmUnquantizedConfig(),
                    prefix="layers.fp8.mlp",
                )

    def test_mimo_mlp_intermediate_padding(self):
        # `d_intermediate=96` (not divisible by 128, divisible by 32) pads to `128`!
        cfg_mimo_mlp = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=128,
            d_intermediate=96,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )
        model_mimo, _ = _build_mamba3_model(cfg_mimo_mlp)
        layer0 = model_mimo.backbone.layers[0]
        assert isinstance(layer0.mlp, Mamba3GatedMLP)
        assert layer0.mlp.orig_intermediate_size == 96
        assert layer0.mlp.intermediate_size == 128
        assert isinstance(layer0.mixer.mimo_x, torch.nn.Parameter)
        assert isinstance(layer0.mixer.mimo_o, torch.nn.Parameter)
        assert isinstance(layer0.mixer.mimo_z, torch.nn.Parameter)
        assert not hasattr(layer0.mixer, "Psi")

    def test_load_weights_siso_and_raises_on_unknown_checkpoint_tensor(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=128,
            tie_embeddings=False,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
                "mimo_rank": 4,
                "is_outproj_norm": False,
            },
        )
        model, _ = _build_mamba3_model(cfg)
        mixer = model.backbone.layers[0].mixer
        mlp = model.backbone.layers[0].mlp
        assert mlp is not None
        total_proj = mixer.total_proj_size

        emb_w = torch.randn(64, 32)
        lm_head_w = torch.randn(64, 32)
        b_bias_3d = torch.randn(4, 1, 8)
        c_bias_3d = torch.randn(4, 1, 8)
        fc1_ckpt = torch.randn(256, 32)

        ckpt = [
            ("backbone.embedding.weight", emb_w),
            ("backbone.layers.0.norm.weight", torch.randn(32)),
            ("backbone.layers.0.mixer.in_proj.weight", torch.randn(total_proj, 32)),
            ("backbone.layers.0.mixer.out_proj.weight", torch.randn(32, 64)),
            ("backbone.layers.0.mixer.B_norm.weight", torch.randn(8)),
            ("backbone.layers.0.mixer.C_norm.weight", torch.randn(8)),
            ("backbone.layers.0.mixer.B_bias", b_bias_3d),
            ("backbone.layers.0.mixer.C_bias", c_bias_3d),
            ("backbone.layers.0.mixer.D", torch.randn(4)),
            ("backbone.layers.0.mixer.dt_bias", torch.randn(4)),
            ("backbone.layers.0.norm2.weight", torch.randn(32)),
            ("backbone.layers.0.mlp.fc1.weight", fc1_ckpt),
            ("backbone.layers.0.mlp.fc2.weight", torch.randn(32, 128)),
            ("backbone.norm_f.weight", torch.randn(32)),
            ("lm_head.weight", lm_head_w),
        ]
        loaded = model.load_weights(ckpt)
        assert "backbone.embedding.weight" in loaded
        assert "lm_head.weight" in loaded
        torch.testing.assert_close(mixer.B_bias, b_bias_3d.squeeze(1))
        torch.testing.assert_close(mixer.C_bias, c_bias_3d.squeeze(1))
        torch.testing.assert_close(model.lm_head.weight, lm_head_w)
        # Verify `[y_up, gate] -> [gate, y_up]` swap even at TP=1 when pre-padded!
        torch.testing.assert_close(mlp.fc1.weight[:128], fc1_ckpt[128:])
        torch.testing.assert_close(mlp.fc1.weight[128:], fc1_ckpt[:128])

        with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
            model.load_weights(
                [("backbone.layers.0.mixer.unknown_param", torch.ones(4))]
            )

    def test_pipeline_parallel_missing_layers_and_tied_last_rank_weights(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=2,
            vocab_size=64,
            d_intermediate=0,
            tie_embeddings=True,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
            },
        )
        pp_first = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        pp_last = SimpleNamespace(is_first_rank=False, is_last_rank=True)

        model_first, _ = _build_mamba3_model(
            cfg, pp_group=pp_first, pp_layer_range=(0, 1)
        )
        assert not isinstance(model_first.backbone.embedding, PPMissingLayer)
        assert isinstance(model_first.backbone.layers[1], PPMissingLayer)
        assert isinstance(model_first.backbone.norm_f, PPMissingLayer)
        assert isinstance(model_first.lm_head, PPMissingLayer)
        assert model_first.compute_logits(torch.zeros(2, 32)) is None

        model_last, _ = _build_mamba3_model(
            cfg, pp_group=pp_last, pp_layer_range=(1, 2)
        )
        assert isinstance(model_last.backbone.embedding, PPMissingLayer)
        assert isinstance(model_last.backbone.layers[0], PPMissingLayer)
        assert not isinstance(model_last.backbone.norm_f, PPMissingLayer)
        assert not isinstance(model_last.lm_head, PPMissingLayer)

        emb_w = torch.randn(64, 32)
        d0_w = torch.randn(4)
        d1_w = torch.randn(4)
        norm_f_w = torch.randn(32)
        shared_ckpt = [
            ("backbone.embedding.weight", emb_w),
            ("backbone.layers.0.mixer.D", d0_w),
            ("backbone.layers.1.mixer.D", d1_w),
            ("backbone.norm_f.weight", norm_f_w),
        ]

        loaded_first = model_first.load_weights(shared_ckpt)
        assert "backbone.embedding.weight" in loaded_first
        assert "backbone.layers.0.mixer.D" in loaded_first
        assert "backbone.layers.1.mixer.D" not in loaded_first

        loaded_last = model_last.load_weights(shared_ckpt)
        assert "lm_head.weight" in loaded_last
        assert "backbone.layers.1.mixer.D" in loaded_last
        torch.testing.assert_close(model_last.lm_head.weight, emb_w)

        with patch(
            "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
            return_value=pp_first,
        ):
            inter = model_first(
                torch.tensor([1, 2], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            )
        assert isinstance(inter, IntermediateTensors)
        with patch(
            "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
            return_value=pp_last,
        ):
            out_last = model_last(
                None,
                torch.tensor([0, 1], dtype=torch.long),
                intermediate_tensors=inter,
            )
        assert isinstance(out_last, torch.Tensor) and out_last.shape == (2, 32)

    def test_load_weights_mimo_with_unpadded_mlp_half_swap_and_multiple_of_128(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=96,
            tie_embeddings=True,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )
        model, _ = _build_mamba3_model(cfg)
        mixer = model.backbone.layers[0].mixer
        mlp = model.backbone.layers[0].mlp
        assert mlp is not None and mlp.intermediate_size == 128

        gen = torch.Generator().manual_seed(0)
        mimo_x_w = torch.randn((4, 4, 16), generator=gen)
        mimo_o_w = torch.randn((4, 4, 16), generator=gen)
        mimo_z_w = torch.randn((4, 4, 16), generator=gen)
        y_up_unpadded = torch.randn((96, 32), generator=gen)
        gate_unpadded = torch.randn((96, 32), generator=gen)
        fc1_unpadded = torch.cat([y_up_unpadded, gate_unpadded], dim=0)
        fc2_unpadded = torch.randn((32, 96), generator=gen)

        model.load_weights(
            [
                ("model.embed_tokens.weight", torch.ones(64, 32)),
                ("model.layers.0.mixer.mimo_x", mimo_x_w),
                ("model.layers.0.mixer.mimo_o", mimo_o_w),
                ("model.layers.0.mixer.mimo_z", mimo_z_w),
                ("model.layers.0.mlp_norm.weight", torch.ones(32) * 2.0),
                ("model.layers.0.mlp.fc1.weight", fc1_unpadded),
                ("model.layers.0.mlp.fc2.weight", fc2_unpadded),
            ]
        )
        torch.testing.assert_close(mixer.mimo_x, mimo_x_w)
        torch.testing.assert_close(mixer.mimo_o, mimo_o_w)
        torch.testing.assert_close(mixer.mimo_z, mimo_z_w)
        # First half of `mlp.fc1.weight` is `gate` (`SiluAndMul` gates first half),
        # second half is `y_up`, each padded from 96 -> 128!
        torch.testing.assert_close(mlp.fc1.weight[:96], gate_unpadded)
        assert torch.all(mlp.fc1.weight[96:128] == 0)
        torch.testing.assert_close(mlp.fc1.weight[128:224], y_up_unpadded)
        assert torch.all(mlp.fc1.weight[224:256] == 0)
        torch.testing.assert_close(mlp.fc2.weight[:, :96], fc2_unpadded)
        assert torch.all(mlp.fc2.weight[:, 96:128] == 0)

        # Verify forward matches `fc2_unpadded @ ((x @ y_up.T) * silu(x @ gate.T))`:
        x_in = torch.randn((5, 32), generator=gen)
        expected_mlp_out = (
            (x_in @ y_up_unpadded.T) * torch.nn.functional.silu(x_in @ gate_unpadded.T)
        ) @ fc2_unpadded.T
        torch.testing.assert_close(mlp(x_in), expected_mlp_out, rtol=1e-4, atol=1e-4)

        # Also verify `loaded_shard_id=0` (`y_up`) and `loaded_shard_id=1` (`gate`):
        mlp.fc1.weight.data.zero_()
        mlp.fc1.weight.weight_loader(mlp.fc1.weight, y_up_unpadded, 0)
        mlp.fc1.weight.weight_loader(mlp.fc1.weight, gate_unpadded, 1)
        torch.testing.assert_close(mlp(x_in), expected_mlp_out, rtol=1e-4, atol=1e-4)

    def test_multiprocess_tp_sharding_with_single_group(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=96,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
            },
        )
        model_r1, _ = _build_mamba3_model(cfg, tp_size=2, tp_rank=1)
        mixer_r1 = model_r1.backbone.layers[0].mixer
        mlp_r1 = model_r1.backbone.layers[0].mlp
        assert mlp_r1 is not None and mlp_r1.intermediate_size == 128
        assert mixer_r1.in_proj.weight.shape == (88, 32)

        in_proj_global = torch.arange(158 * 32, dtype=torch.float32).reshape(158, 32)
        b_bias_3d = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 1, 8)
        fc1_padded = torch.arange(256 * 32, dtype=torch.float32).reshape(256, 32)
        fc2_padded = torch.arange(32 * 128, dtype=torch.float32).reshape(32, 128)

        model_r1.load_weights(
            [
                ("backbone.layers.0.mixer.in_proj.weight", in_proj_global),
                ("backbone.layers.0.mixer.B_bias", b_bias_3d),
                ("backbone.layers.0.mlp.fc1.weight", fc1_padded),
                ("backbone.layers.0.mlp.fc2.weight", fc2_padded),
            ]
        )

        expected_in_proj_r1 = torch.cat(
            [
                in_proj_global[32:64],
                in_proj_global[96:128],
                in_proj_global[128:136],
                in_proj_global[136:144],
                in_proj_global[146:148],
                in_proj_global[150:152],
                in_proj_global[154:156],
                in_proj_global[156:158],
            ],
            dim=0,
        )
        torch.testing.assert_close(mixer_r1.in_proj.weight, expected_in_proj_r1)
        torch.testing.assert_close(mixer_r1.B_bias, b_bias_3d.squeeze(1)[2:4])
        # Checkpoint `[y_up, gate]` -> rank 1 gets `[gate[64:128], y_up[64:128]]`:
        expected_fc1_r1 = torch.cat([fc1_padded[192:256], fc1_padded[64:128]], dim=0)
        torch.testing.assert_close(mlp_r1.fc1.weight, expected_fc1_r1)
        torch.testing.assert_close(mlp_r1.fc2.weight, fc2_padded[:, 64:128])


class TestMamba3EndToEndForward:
    @pytest.mark.parametrize(
        ("is_mimo", "seed", "golden_logits", "golden_greedy"),
        [
            (
                False,
                3030,
                MAMBA3_SISO_UPSTREAM_GOLDEN_LOGITS,
                MAMBA3_SISO_UPSTREAM_GOLDEN_GREEDY,
            ),
            (
                True,
                3031,
                MAMBA3_MIMO_UPSTREAM_GOLDEN_LOGITS,
                MAMBA3_MIMO_UPSTREAM_GOLDEN_GREEDY,
            ),
        ],
    )
    def test_checkpoint_load_and_prefill_decode_match_upstream_mamba_ssm_goldens(
        self,
        tmp_path,
        is_mimo: bool,
        seed: int,
        golden_logits: tuple[float, ...],
        golden_greedy: tuple[int, ...],
    ):
        """Verifies SISO & MIMO `Mamba3ForCausalLM` against offline golden logits
        and greedy tokens generated directly from upstream `state-spaces/mamba`."""
        register_models()
        ckpt_dir = tmp_path / f"mamba3_ckpt_{int(is_mimo)}"
        write_synthetic_mamba3_checkpoint(ckpt_dir, is_mimo=is_mimo, seed=seed)

        prompt_ids = list(MAMBA3_GOLDEN_PROMPT_IDS)
        gen_steps = len(golden_greedy)
        ref_prompt_logits, ref_greedy = eval_torch_mamba3_reference(
            ckpt_dir, prompt_ids, gen_steps=gen_steps
        )
        upstream_golden_tensor = torch.tensor(golden_logits, dtype=torch.float32)
        torch.testing.assert_close(
            ref_prompt_logits, upstream_golden_tensor, rtol=1e-5, atol=1e-5
        )
        assert tuple(ref_greedy) == golden_greedy

        hf_cfg = get_config(
            str(ckpt_dir), trust_remote_code=False, config_format="mamba3"
        )
        assert isinstance(hf_cfg, Mamba3Config)
        model, vllm_cfg = _build_mamba3_model(hf_cfg, use_real_jax_op=True)
        sd = load_file(str(ckpt_dir / "model.safetensors"))
        model.load_weights(list(sd.items()))

        aux_shape, ssm_shape = Mamba3ForCausalLM.get_mamba_state_shape_from_config(
            vllm_cfg
        )
        prefixes = [f"backbone.layers.{i}.mixer" for i in range(hf_cfg.n_layer)]
        for layer in model.backbone.layers:
            layer.mixer.kv_cache = (
                torch.zeros((2, *aux_shape), dtype=torch.float32),
                torch.zeros((2, *ssm_shape), dtype=torch.float32),
            )

        meta_pre = make_mamba_attn_metadata(
            mamba_state_indices=torch.tensor([0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, len(prompt_ids)], dtype=torch.int32),
            request_distribution=torch.tensor([0, 1, 1], dtype=torch.int32),
            seq_lens=torch.tensor([len(prompt_ids)], dtype=torch.int32),
        )
        with patch(
            "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context",
            return_value=make_forward_context(prefixes, meta_pre),
        ):
            hidden_pre = model(
                torch.tensor(prompt_ids, dtype=torch.long),
                torch.arange(len(prompt_ids), dtype=torch.long),
            )
            logits_pre = model.compute_logits(hidden_pre)

        assert logits_pre is not None
        assert logits_pre.shape == (len(prompt_ids), 112)
        assert torch.all(logits_pre[:, 100:] == float("-inf"))
        torch.testing.assert_close(
            logits_pre[-1, :100], upstream_golden_tensor, rtol=1e-4, atol=1e-4
        )

        greedy_ids = [int(torch.argmax(logits_pre[-1]).item())]
        for step in range(1, gen_steps):
            cur_len = len(prompt_ids) + step
            meta_dec = make_mamba_attn_metadata(
                mamba_state_indices=torch.tensor([0], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                request_distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
                seq_lens=torch.tensor([cur_len], dtype=torch.int32),
            )
            with patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context",
                return_value=make_forward_context(prefixes, meta_dec),
            ):
                h_dec = model(
                    torch.tensor([greedy_ids[-1]], dtype=torch.long),
                    torch.tensor([cur_len - 1], dtype=torch.long),
                )
                logits_dec = model.compute_logits(h_dec)
            assert logits_dec is not None and logits_dec.shape == (1, 112)
            assert torch.all(logits_dec[:, 100:] == float("-inf"))
            greedy_ids.append(int(torch.argmax(logits_dec[-1]).item()))

        assert tuple(greedy_ids) == golden_greedy


def _run_mamba2_e2e_worker(
    ckpt_dir_str: str, seed: int, prompts: list[list[int]], max_tokens: int
) -> None:
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ckpt_dir = Path(ckpt_dir_str)
    write_synthetic_mamba2_checkpoint(ckpt_dir, seed=seed)
    refs = [
        eval_hf_mamba2_reference(ckpt_dir, prompt_ids, gen_steps=max_tokens)
        for prompt_ids in prompts
    ]
    llm = LLM(
        model=ckpt_dir_str,
        tokenizer=ckpt_dir_str,
        max_model_len=128,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        enable_prefix_caching=False,
        enforce_eager=False,
        async_scheduling=False,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, logprobs=5, ignore_eos=True
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids) for prompt_ids in prompts], params
    )
    assert len(outputs) == len(prompts)
    for idx, (req_out, (ref_logits, expected_tokens)) in enumerate(
        zip(outputs, refs, strict=True)
    ):
        out0 = req_out.outputs[0]
        first_tok = int(out0.token_ids[0])
        assert out0.logprobs is not None and first_tok in out0.logprobs[0]
        step0_lp = float(out0.logprobs[0][first_tok].logprob)
        assert list(out0.token_ids) == expected_tokens, (
            f"Mamba-2 prompt[{idx}] (len={len(prompts[idx])}) token mismatch: "
            f"got {list(out0.token_ids)} vs expected {expected_tokens} "
            f"(step0_lp={step0_lp:.4f})"
        )
        ref_lp = float(
            torch.log_softmax(ref_logits.float(), dim=-1)[expected_tokens[0]].item()
        )
        assert abs(step0_lp - ref_lp) < 0.15, (
            f"Mamba-2 prompt[{idx}] (len={len(prompts[idx])}) step0 logprob drift: "
            f"got {step0_lp:.4f} vs ref {ref_lp:.4f}"
        )


def _run_mamba3_e2e_worker(
    ckpt_dir_str: str,
    is_mimo: bool,
    seed: int,
    prompts: list[list[int]],
    max_tokens: int,
) -> None:
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "1"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ckpt_dir = Path(ckpt_dir_str)
    write_synthetic_mamba3_checkpoint(ckpt_dir, is_mimo=is_mimo, seed=seed)
    refs = [
        eval_torch_mamba3_reference(ckpt_dir, prompt_ids, gen_steps=max_tokens)
        for prompt_ids in prompts
    ]
    llm = LLM(
        model=ckpt_dir_str,
        tokenizer=ckpt_dir_str,
        config_format="mamba3",
        max_model_len=128,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        enable_prefix_caching=False,
        enforce_eager=False,
        async_scheduling=False,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, logprobs=5, ignore_eos=True
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids) for prompt_ids in prompts], params
    )
    assert len(outputs) == len(prompts)
    for idx, (req_out, (ref_logits, expected_tokens)) in enumerate(
        zip(outputs, refs, strict=True)
    ):
        out0 = req_out.outputs[0]
        first_tok = int(out0.token_ids[0])
        assert out0.logprobs is not None and first_tok in out0.logprobs[0]
        step0_lp = float(out0.logprobs[0][first_tok].logprob)
        assert list(out0.token_ids) == expected_tokens, (
            f"Mamba-3 (is_mimo={is_mimo}) prompt[{idx}] (len={len(prompts[idx])}) "
            f"token mismatch: got {list(out0.token_ids)} vs expected "
            f"{expected_tokens} (step0_lp={step0_lp:.4f})"
        )
        ref_lp = float(
            torch.log_softmax(ref_logits.float(), dim=-1)[expected_tokens[0]].item()
        )
        assert abs(step0_lp - ref_lp) < 0.15, (
            f"Mamba-3 (is_mimo={is_mimo}) prompt[{idx}] (len={len(prompts[idx])}) "
            f"step0 logprob drift: got {step0_lp:.4f} vs ref {ref_lp:.4f}"
        )


# 10 varied-length prompts (> 8 usable slots with MIN_NUM_SEQS=8 and
# max_num_seqs=4) interleaved long/short to exercise concurrent slot residency
# and slot reuse without bf16 argmax ties across all 3 synthetic checkpoints.
_E2E_VARIED_PROMPTS: list[list[int]] = [
    [30, 12, 24, 6, 11, 17, 9, 12, 24, 30, 16, 24, 9, 14, 9, 18, 29, 29],
    [20, 19, 28],
    [14, 19, 21, 25, 19, 8, 6, 21, 19, 24, 6, 25, 6, 6, 27, 28],
    [13, 31, 13, 27],
    [11, 22, 5, 13, 17, 31, 17, 5, 19, 18, 9, 28, 13, 23],
    [30, 5, 22, 24, 16],
    [9, 17, 3, 9, 23, 18, 29, 6, 31, 25, 30, 29],
    [10, 23, 26, 22, 13, 6],
    [20, 11, 20, 10, 23, 19, 6, 5, 27, 10],
    [19, 11, 12, 29, 10, 28, 25, 17, 20],
]


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
@pytest.mark.parametrize("variant", ["mamba2", "mamba3_siso", "mamba3_mimo"])
def test_e2e_vllm_llm_serving_mamba2_and_mamba3_parity_tpu(
    tmp_path: Path, variant: str
):
    """End-to-end `vllm.LLM` TPU serving parity with recompilation checks."""
    import functools

    max_tokens = 4

    if variant == "mamba2":
        ckpt_dir = tmp_path / "mamba2_e2e"
        worker = functools.partial(
            _run_mamba2_e2e_worker,
            str(ckpt_dir),
            7001,
            _E2E_VARIED_PROMPTS,
            max_tokens,
        )
    else:
        is_mimo = variant == "mamba3_mimo"
        ckpt_dir = tmp_path / f"mamba3_e2e_{int(is_mimo)}"
        worker = functools.partial(
            _run_mamba3_e2e_worker,
            str(ckpt_dir),
            is_mimo,
            8001 + int(is_mimo),
            _E2E_VARIED_PROMPTS,
            max_tokens,
        )
    run_in_isolated_process(worker, timeout=500)
