# SPDX-License-Identifier: Apache-2.0
"""Reject geometry the selected TPU kernels cannot represent losslessly."""

import pytest

from tests.models.vllm.test_qwen4_exp_parallel import config
from vllm_torchtpu.models.vllm.qwen4_exp.config import (
    use_unified_pool,
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
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig

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
    "override,expected", [(None, True), (True, True), (False, False)]
)
def test_unified_pool_defaults_on_and_respects_global_override(
    monkeypatch, override, expected
):
    if override is None:
        monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    else:
        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", str(int(override)))
    assert use_unified_pool() is expected


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
        (False, False, "none", None),
        (False, True, "align", "APC only with unified pool"),
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
