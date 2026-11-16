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

from tests.models.qwen4_exp.experimental.qwen38_test_utils import pin_plan
from tests.models.qwen4_exp.experimental.test_parallel import config
from vllm_torchtpu.models.qwen4_exp.experimental.config import (
    use_unified_pool,
    validate_config,
)


@pytest.fixture(autouse=True)
def _plan_off(monkeypatch):
    pin_plan(monkeypatch)


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
    try:
        from transformers import Qwen4ExpTextConfig
    except ImportError:
        from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig

    from tests.models.qwen4_exp.experimental.qwen38_test_utils import (
        q4exp_text_config,
    )

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

    from vllm_torchtpu.models.qwen4_exp.experimental import config as model_config

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
    monkeypatch.setattr(model_config, "use_unified_pool", lambda: False)
    monkeypatch.setattr(model_config, "native_cache_capacity", lambda cfg: 32)
    model_config.TPUQwen4ExpConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_ssm_cache_dtype == resolved
    expected = torch.float32 if resolved == "float32" else torch.bfloat16
    assert model_config.gdn_cache_spec(cfg).dtypes == (torch.float32, expected)


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


@pytest.mark.parametrize("prefix", [False, True])
def test_pool_page_override_respects_upstream_prefix_validation(monkeypatch, prefix):
    from vllm.config import CacheConfig, VllmConfig

    from vllm_torchtpu.models.qwen4_exp.experimental.config import (
        Qwen4ExpForCausalLMConfig,
        TPUQwen4ExpConfig,
    )
    from vllm_torchtpu.models.qwen4_exp.experimental.pool_layout import (
        pooled_cache_specs,
    )

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
    "connector", ["Qwen4ExpRaidenConnector", "TPURaidenOffloadingConnector"]
)
def test_rejects_any_kv_connector(connector):
    from types import SimpleNamespace

    cfg = config()
    validate_config(cfg)
    cfg.kv_transfer_config = SimpleNamespace(kv_connector=connector)
    with pytest.raises(ValueError, match="no KV connector"):
        validate_config(cfg)


def test_get_qsa_kv_cache_bytes_per_block_supports_1arg_and_2arg_vllm(monkeypatch):
    from vllm.v1.core import kv_cache_planning as kv_cache_utils

    from vllm_torchtpu.models.qwen4_exp.experimental.config import (
        get_qsa_kv_cache_bytes_per_block,
        resolve_qsa_kv_cache_layout,
    )

    cfg = config()
    seen = []

    def one_arg(kv_cache_groups):
        seen.append(("1arg", kv_cache_groups))
        return 4096

    def two_arg(kv_cache_groups, kv_cache_layout=None):
        seen.append(("2arg", kv_cache_groups, kv_cache_layout))
        return 8192

    monkeypatch.setattr(kv_cache_utils, "_get_kv_cache_bytes_per_block", one_arg)
    assert get_qsa_kv_cache_bytes_per_block(["g0"], cfg) == 4096
    monkeypatch.setattr(kv_cache_utils, "_get_kv_cache_bytes_per_block", two_arg)
    assert get_qsa_kv_cache_bytes_per_block(["g0"], cfg) == 8192
    assert seen == [
        ("1arg", ["g0"]),
        ("2arg", ["g0"], resolve_qsa_kv_cache_layout(cfg)),
    ]
