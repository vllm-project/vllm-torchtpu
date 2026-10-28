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
"""Reject geometry the selected TPU kernels cannot represent losslessly."""

import pytest

from tests.models.vllm.test_qwen4_exp_parallel import config
from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.qwen4_exp.config import (
    require_unified_pool,
    validate_config,
)


@pytest.mark.parametrize("dtype", ["fp8_per_token_head", "fp8_e5m2"])
def test_admission_rejects_non_cuda_qsa_kv(dtype):
    cfg = config()
    cfg.cache_config.cache_dtype = dtype
    with pytest.raises(ValueError, match="BF16"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("linear_key_head_dim", 64),
        ("linear_key_head_dim", 192),
        ("linear_value_head_dim", 64),
        ("linear_conv_kernel_dim", 1),
    ],
)
def test_admission_rejects_unrepresentable_gdn_head_tiles(field, value):
    cfg = config()
    setattr(cfg.model_config.hf_text_config, field, value)
    with pytest.raises(ValueError, match="GDN"):
        validate_config(cfg)


def test_admission_accepts_upstream_optional_ple_shard_default():
    from transformers import Qwen4ExpTextConfig

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config

    fields = vars(q4exp_text_config())
    del fields["split_ngram_parts"]
    cfg = config()
    cfg.model_config.hf_text_config = Qwen4ExpTextConfig(**fields)
    validate_config(cfg)


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8", "fp8_e4m3"])
def test_admission_accepts_cuda_qsa_kv_modes(dtype):
    cfg = config()
    cfg.cache_config.cache_dtype = dtype
    validate_config(cfg)


@pytest.mark.parametrize(
    "requested,resolved",
    [("auto", "float32"), ("float32", "float32"), ("bfloat16", "bfloat16")],
)
def test_ssm_dtype_uses_checkpoint_default_and_preserves_explicit_override(
    monkeypatch, requested, resolved
):
    import torch
    from vllm.config import CacheConfig

    from vllm_torchtpu.models.vllm.qwen4_exp import config as model_config

    cfg = config()
    cfg.model_config.hf_text_config.mamba_ssm_dtype = "float32"
    cfg.model_config.multimodal_config = None
    cfg.parallel_config.ubatch_size = 1
    cfg.cache_config = CacheConfig(
        block_size=128,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype=requested,
    )
    # Exercise the real upstream normalization and model hook. Capacity
    # planning has separate native-cache tests and does not choose a dtype.
    monkeypatch.setattr(model_config, "native_cache_capacity", lambda cfg, specs: 32)
    model_config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_ssm_cache_dtype == resolved
    expected = torch.float32 if resolved == "float32" else torch.bfloat16
    # Projected convolution history stays BF16 independently of the SSM override.
    assert model_config.gdn_cache_spec(cfg).dtypes == (torch.bfloat16, expected)


@pytest.mark.parametrize(
    "override,expected", [(None, True), (True, True), (False, False)]
)
def test_unified_pool_defaults_on_and_respects_global_override(
    monkeypatch, override, expected
):
    from vllm_torchtpu.gdn_pool_layout import unified_kv_layout_enabled_for_architecture

    if override is None:
        monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    else:
        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", str(int(override)))
    assert unified_kv_layout_enabled_for_architecture("Qwen4ExpForCausalLM") is expected
    if override is False:
        with pytest.raises(ValueError, match="requires the dual-storage pool"):
            require_unified_pool()
        with pytest.raises(ValueError, match="requires the dual-storage pool"):
            validate_config(config())
    else:
        assert require_unified_pool() is None


@pytest.mark.parametrize(
    "option", ["qsa_prefill_backend", "qsa_sparse_backend", "qsa_dense_threshold"]
)
def test_retired_prefill_options_fail_with_migration_message(option):
    cfg = config()
    cfg.additional_config = {"qwen4_exp": {option: "auto"}}
    with pytest.raises(ValueError, match="Remove retired QSA options"):
        validate_config(cfg)


def test_context_capacity_allows_8k_prompt_and_decode():
    cfg = config()
    cfg.model_config.max_model_len = 8192 + 1024
    validate_config(cfg)


def test_context_capacity_follows_model_position_limit():
    cfg = config()
    cfg.model_config.hf_text_config.max_position_embeddings = 8192 + 1024
    cfg.model_config.max_model_len = (
        cfg.model_config.hf_text_config.max_position_embeddings
    )
    validate_config(cfg)
    cfg.model_config.max_model_len += 1
    with pytest.raises(ValueError, match="exceeds model position capacity"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "pooled,prefix,mode,error",
    [
        (False, False, "none", "requires the dual-storage pool"),
        (False, True, "align", "requires the dual-storage pool"),
        (True, True, "align", None),
        (True, False, "none", None),
        (True, False, "align", None),
        (True, True, "none", "mamba_cache_mode='align'"),
    ],
)
def test_prefix_reuse_requires_scheduler_owned_aligned_state(
    monkeypatch, pooled, prefix, mode, error
):
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", str(int(pooled)))
    cfg = config()
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = mode
    if error is None:
        validate_config(cfg)
    else:
        with pytest.raises(ValueError, match=error):
            validate_config(cfg)


@pytest.mark.parametrize("prefix", [False, True])
def test_pool_page_override_respects_upstream_prefix_validation(monkeypatch, prefix):
    from vllm.config import CacheConfig, VllmConfig

    from vllm_torchtpu.models.vllm.qwen4_exp.config import (
        Qwen4ExpForCausalLMConfig,
        TPUQwen4ExpConfig,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    # The fixture already contains normalized text-only HF fields. Exercise
    # our page override and the actual upstream validator, without a loader.
    monkeypatch.setattr(
        Qwen4ExpForCausalLMConfig,
        "verify_and_update_config",
        staticmethod(lambda cfg: None),
    )
    cfg = config()
    cfg.cache_config = CacheConfig(
        block_size=128,
        cache_dtype="fp8_e4m3",
        enable_prefix_caching=prefix,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="bfloat16",
    )
    TPUQwen4ExpConfig.verify_and_update_config(cfg)
    VllmConfig.validate_mamba_block_size(cfg)
    layout, specs = pooled_cache_specs(cfg)
    assert cfg.cache_config.block_size == layout.block_size
    assert cfg.cache_config.mamba_block_size == (layout.block_size if prefix else None)
    assert {spec.page_size_bytes for spec in specs.values()} == {layout.page_bytes}


@pytest.mark.parametrize(
    "architecture,expected",
    [("Qwen3ForCausalLM", False), ("Qwen3_5ForCausalLM", False)],
)
def test_pool_opt_out_remains_available_to_other_models(
    monkeypatch, architecture, expected
):
    from vllm_torchtpu.gdn_pool_layout import unified_kv_layout_enabled_for_architecture

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    assert unified_kv_layout_enabled_for_architecture(architecture) is expected
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
    assert unified_kv_layout_enabled_for_architecture(architecture) is True


def test_qwen4_pool_opt_out_fails_before_adapter_construction(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter

    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    with pytest.raises(ValueError, match="requires the dual-storage pool"):
        Qwen4ExpAdapter(None)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
def test_pd_admits_only_model_owned_fp8_dp8(monkeypatch, role, ssm_dtype):
    from types import SimpleNamespace

    # Override even a materialized lazy env attribute, then restore its absence.
    monkeypatch.setitem(vars(envs), "TPU_KV_RESHARD_TRANSPORT", "raiden")
    cfg = config()
    cfg.cache_config.cache_dtype = "fp8"
    cfg.cache_config.mamba_ssm_cache_dtype = ssm_dtype
    cfg.kv_transfer_config = SimpleNamespace(
        kv_connector="Qwen4ExpRaidenConnector",
        kv_connector_module_path=None,
        kv_role=role,
    )
    validate_config(cfg)
    for transport in ("", "zmq"):
        with monkeypatch.context() as patch:
            patch.setitem(vars(envs), "TPU_KV_RESHARD_TRANSPORT", transport)
            with pytest.raises(ValueError, match="model-owned Raiden connector"):
                validate_config(cfg)
    for admission in ("QWEN35", "KIMIK3", "GLM", "DSV4"):
        with monkeypatch.context() as patch:
            patch.setitem(vars(envs), f"TPU_RAIDEN_{admission}_ADMISSION", True)
            with pytest.raises(ValueError, match="model-owned Raiden connector"):
                validate_config(cfg)
    for section, field, value in [
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 4),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", False),
        ("cache_config", "cache_dtype", "auto"),
        ("cache_config", "mamba_ssm_cache_dtype", "float16"),
        ("cache_config", "enable_prefix_caching", True),
        ("kv_transfer_config", "kv_role", "kv_both"),
        ("kv_transfer_config", "kv_connector", "TPURaidenConnector"),
        ("kv_transfer_config", "kv_connector_module_path", "other.module"),
    ]:
        target = getattr(cfg, section)
        original = getattr(target, field)
        setattr(target, field, value)
        with pytest.raises(ValueError):
            validate_config(cfg)
        setattr(target, field, original)

    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.qwen4_exp.config.host_available_memory",
        lambda: 1 << 60,
    )
    for offload, shared, across in [
        (False, False, True),
        (True, False, True),
        (True, True, False),
    ]:
        cfg.additional_config = {
            "qwen4_exp": {
                "engram_config": {
                    "cpu_offload": offload,
                    "dp_shared_memory": shared,
                    "embedding_across_dp": across,
                }
            }
        }
        validate_config(cfg)

    cfg.speculative_config = SimpleNamespace()
    with pytest.raises(ValueError, match="MTP"):
        validate_config(cfg)
    cfg.speculative_config = None
    monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
    with pytest.raises(ValueError, match="requires the dual-storage pool"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [None, "auto", "0", "128", "256"])
def test_hc_env_override_preserves_s1_admission(monkeypatch, value):
    from vllm.config import CacheConfig

    from vllm_torchtpu.models.vllm.qwen4_exp import config as model_config

    if value is None:
        monkeypatch.delenv("VLLM_TPU_HC_PREFILL_TILE", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", value)
    cfg = config()
    cfg.model_config.multimodal_config = None
    cfg.parallel_config.ubatch_size = 1
    cfg.cache_config = CacheConfig(
        block_size=128,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="bfloat16",
    )
    monkeypatch.setattr(model_config, "native_cache_capacity", lambda cfg, specs: 32)
    model_config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.kv_cache_layout == "BLHNC"
    assert cfg.cache_config.num_gpu_blocks_override == 32
    cfg.model_config.hf_text_config.linear_key_head_dim = 64
    with pytest.raises(ValueError, match="GDN"):
        model_config.TPUQwen4ExpConfig.verify_and_update_config(cfg)


@pytest.mark.parametrize(
    "value", ["true", "false", "-128", "127", "129", "128.0", "atuo"]
)
def test_config_rejects_invalid_hc_debug_override(monkeypatch, value):
    from vllm.config import CacheConfig

    from vllm_torchtpu.models.vllm.qwen4_exp import config as model_config

    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", value)
    cfg = config()
    cfg.cache_config = CacheConfig(mamba_ssm_cache_dtype="bfloat16")
    cfg.model_config.multimodal_config = None
    cfg.parallel_config.ubatch_size = 1
    with pytest.raises(ValueError, match="HC|VLLM_TPU_HC_PREFILL_TILE"):
        model_config.TPUQwen4ExpConfig.verify_and_update_config(cfg)


@pytest.mark.parametrize("value", [True, False, -128, 127])
def test_config_validates_typed_hc_debug_override(monkeypatch, value):
    from vllm.config import CacheConfig

    from vllm_torchtpu.models.vllm.qwen4_exp import config as model_config

    monkeypatch.setitem(
        envs.environment_variables, "VLLM_TPU_HC_PREFILL_TILE", lambda: value
    )
    cfg = config()
    cfg.cache_config = CacheConfig(mamba_ssm_cache_dtype="bfloat16")
    cfg.model_config.multimodal_config = None
    cfg.parallel_config.ubatch_size = 1
    with pytest.raises(ValueError, match="HC prefill"):
        model_config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
