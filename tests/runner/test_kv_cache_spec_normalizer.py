import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec

from vllm_torchtpu.kv_cache_spec_normalizer import normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasMLAttentionBackend,
    VllmTPUDeepseekV32IndexerBackend,
)


def test_attention_spec_with_bf16_normalizes_to_fp8_without_inflation():
    # Defect A regression: when input AttentionSpec has dtype=torch.bfloat16 (2 bytes),
    # normalizer must evaluate real_page_size_bytes on the target FP8 dtype (1 byte),
    # avoiding a 2x inflated page_size_padded.
    bf16_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=None,
    )
    assert bf16_spec.real_page_size_bytes == 16384

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": bf16_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    spec = normalized["layer"]

    pallas_page_size = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        16,
        2,
        128,
        torch.float8_e4m3fn,
    )
    assert pallas_page_size == 8192
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == pallas_page_size
    assert spec.page_size_padded < bf16_spec.real_page_size_bytes


def test_mla_attention_spec_with_bf16_normalizes_to_fp8_without_inflation():
    mla_bf16_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=None,
    )
    assert mla_bf16_spec.real_page_size_bytes == 16 * 1 * 576 * 2

    normalized = normalize_kv_cache_specs_for_tpu(
        {"mla_layer": mla_bf16_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    spec = normalized["mla_layer"]
    pallas_mla_page_size = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        16,
        1,
        576,
        torch.float8_e4m3fn,
    )
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == pallas_mla_page_size
    assert spec.page_size_padded < mla_bf16_spec.real_page_size_bytes


def test_attention_spec_uses_tpu_dtype_for_fp8_page_size():
    gpu_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": gpu_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    spec = normalized["layer"]

    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == max(
        PallasAttentionBackend.get_kv_cache_page_size_bytes(
            16,
            2,
            128,
            torch.float8_e4m3fn,
        ),
        gpu_spec.real_page_size_bytes,
    )


def test_mamba_spec_is_left_unchanged():
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=256,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": mamba_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["layer"] is mamba_spec


def test_hybrid_specs_keep_separate_page_size_by_default():
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["attn"].page_size_bytes != normalized["mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded is None


def test_hybrid_specs_use_uniform_page_size_when_unified_enabled():
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
        enable_unified_kv_layout=True,
    )

    assert normalized["attn"].page_size_bytes == normalized["mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized["attn"].page_size_bytes


def test_hybrid_specs_with_smaller_mamba_padded_size_normalizes_safely() -> None:
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=1572864,
    )
    # MambaSpec where page_size_padded was initialized to a smaller value
    # than unpadded size (e.g. from cache_config.mamba_page_size_padded)
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(6, 1, 8192), (32, 128, 128)],
        dtypes=[torch.float32, torch.bfloat16],
        page_size_padded=1146880,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["attn"].page_size_bytes == normalized["mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized["attn"].page_size_bytes
    assert normalized["mamba"].page_size_bytes == 1572864


def test_hybrid_specs_preserve_attention_pallas_padding_workload_a() -> None:
    # Workload A (Qwen 3.5 35B FP8 TP4):
    # Pallas attention kernel requires 786,432 B per page, whereas Mamba unpadded
    # state is 548,864 B. Normalization must preserve the attention Pallas padding.
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(274432,)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )
    assert mamba_spec.page_size_bytes == 548864

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    pallas_expected_page_size = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        1536,
        1,
        128,
        torch.float8_e4m3fn,
    )
    assert pallas_expected_page_size == 786432
    assert normalized["attn"].page_size_bytes == 786432
    assert normalized["attn"].page_size_padded == 786432
    assert normalized["mamba"].page_size_bytes == 786432
    assert normalized["mamba"].page_size_padded == 786432


def test_hybrid_specs_pad_attention_when_mamba_is_larger() -> None:
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=[(500000,)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )
    assert mamba_spec.page_size_bytes == 1000000

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["attn"].page_size_bytes == 1000000
    assert normalized["attn"].page_size_padded == 1000000
    assert normalized["mamba"].page_size_bytes == 1000000
    assert normalized["mamba"].page_size_padded == 1000000


def test_hybrid_specs_mamba_page_uses_pooled_physical_dtypes() -> None:
    # K3 TP32 regression: the KDA layer declares its conv state fp32 (the
    # fused kernel wants fp32 dense caches), but the unified pool stores the
    # conv region in bf16. Sizing the mamba page from the declared dtypes
    # (196608 + 13824 = 210432 B) outgrows the pool's attention-shaped page
    # (207360 B) and fails the materializer's page invariant; the pool
    # physical dtypes keep it at 203520 B so the platform-padded 207360 B
    # holds.
    attention_spec = MLAAttentionSpec(
        block_size=162,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
            162, 1, 576, torch.bfloat16
        ),
    )
    assert attention_spec.page_size_bytes == 207360
    mamba_spec = MambaSpec(
        block_size=32768,
        shapes=[(3, 3, 3, 128), (3, 128, 128)],
        dtypes=[torch.float32, torch.float32],
        page_size_padded=207360,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.bfloat16,
        enable_unified_kv_layout=True,
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["mamba"].page_size_bytes == 207360
    assert normalized["mamba"].page_size_bytes == normalized["attn"].page_size_bytes
    # The retyped spec describes the pool's physical layout: bf16 conv, fp32
    # ssm.
    assert tuple(normalized["mamba"].dtypes) == (torch.bfloat16, torch.float32)


def test_exempt_layers_pass_through_untouched_under_unified_layout() -> None:
    # DeepSeek-V4 custom packed specs where page_size_padded < real_page_size_bytes.
    # If spec.page_size_bytes is called on such an AttentionSpec, it asserts:
    # `assert self.page_size_padded >= self.real_page_size_bytes`.
    # When exempted, this spec must pass through untouched without triggering assertions
    # during unified KV layout normalization.
    ds_v4_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=100,
    )
    assert ds_v4_spec.page_size_padded < ds_v4_spec.real_page_size_bytes

    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=None,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    specs = {
        "attn": attention_spec,
        "mamba": mamba_spec,
        "ds_v4": ds_v4_spec,
    }

    normalized = normalize_kv_cache_specs_for_tpu(
        specs,
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        exempt_layers={"ds_v4"},
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["ds_v4"] is ds_v4_spec
    assert normalized["ds_v4"].dtype == torch.bfloat16
    assert normalized["ds_v4"].page_size_padded == 100
    assert normalized["attn"].page_size_bytes == normalized["mamba"].page_size_bytes
    assert normalized["attn"].dtype == torch.float8_e4m3fn


def test_exempt_attention_does_not_trigger_hybrid_unification_with_mamba() -> None:
    ds_v4_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=100,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "ds_v4": ds_v4_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
        exempt_layers={"ds_v4"},
        attention_backend=PallasAttentionBackend,
    )

    assert normalized["ds_v4"] is ds_v4_spec
    assert normalized["mamba"] is mamba_spec
    assert normalized["mamba"].page_size_padded is None


def test_deepseek_v32_indexer_backend_and_normalizer_modes(monkeypatch, caplog) -> None:
    """VllmTPUDeepseekV32IndexerBackend and normalize_kv_cache_specs_for_tpu
    size the indexer K cache identically to PallasMLAttentionBackend under
    'xla'/'has' (256 B/token), halve to 132 B/token under 'sal', and fall back
    to HAS (256 B/token) with a one-time warning when block_size < 128."""
    import importlib

    from vllm_torchtpu import envs

    st_mod = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    monkeypatch.setattr(st_mod, "_INDEXER_KV_LAYOUT_FALLBACK_LOGGED", set())
    nb, block_size, head_size = 10, 256, 129
    indexer_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        page_size_padded=None,
    )

    # Drop a value a previous test's monkeypatch.setattr undo may have left in the
    # module dict, so `envs.__getattr__` resolves the variable afresh. A bare
    # `delattr(raising=False)` is not enough: `hasattr` is True through
    # `__getattr__`, so it raises on a module dict without the name.
    if "VLLM_TPU_INDEXER_KV_CACHE_MODE" in vars(envs):
        monkeypatch.delattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE")
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    assert envs.environment_variables["VLLM_TPU_INDEXER_KV_CACHE_MODE"]() == "sal"

    for mode in ("xla", "has"):
        monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", mode)
        shape = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
            nb, block_size, 1, head_size, torch.uint8
        )
        mla_shape = PallasMLAttentionBackend.get_kv_cache_shape(
            nb, block_size, 1, head_size, torch.uint8
        )
        assert shape == (nb, block_size // 4, 4, 256)
        assert shape == mla_shape

        page_bytes = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_page_size_bytes(
            block_size, 1, head_size, torch.uint8
        )
        mla_page_bytes = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
            block_size, 1, head_size, torch.uint8
        )
        assert page_bytes == block_size * 256
        assert page_bytes == mla_page_bytes

        norm = normalize_kv_cache_specs_for_tpu(
            {"indexer": indexer_spec},
            torch.float8_e4m3fn,
            attention_backend=PallasAttentionBackend,
        )["indexer"]
        assert norm.dtype == torch.uint8
        assert norm.state_content_bytes == 256
        assert norm.page_size_bytes == block_size * 256

    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    sal_shape = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
        nb, block_size, 1, head_size, torch.uint8
    )
    assert sal_shape == (nb, 33, 4, block_size)
    sal_page_bytes = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_page_size_bytes(
        block_size, 1, head_size, torch.uint8
    )
    assert sal_page_bytes == block_size * 132

    norm_sal = normalize_kv_cache_specs_for_tpu(
        {"indexer": indexer_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )["indexer"]
    assert norm_sal.dtype == torch.uint8
    assert norm_sal.state_content_bytes == 132
    assert norm_sal.page_size_bytes == block_size * 132

    # block_size=64 + 'sal' falls back to HAS (256 B/token) with one-time warning.
    caplog.clear()
    spec_64 = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        page_size_padded=None,
    )
    shape_64 = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
        nb, 64, 1, head_size, torch.uint8
    )
    assert shape_64 == (nb, 16, 4, 256)
    norm_64 = normalize_kv_cache_specs_for_tpu(
        {"indexer": spec_64},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )["indexer"]
    assert norm_64.state_content_bytes == 256
    assert norm_64.page_size_bytes == 64 * 256
    warnings = [
        r.message
        for r in caplog.records
        if "requires block_size to be a positive multiple of 128" in r.message
    ]
    assert len(warnings) == 1


def test_non_indexer_mla_specs_unaffected_by_indexer_kv_cache_mode(monkeypatch) -> None:
    """Sparse-MLA, dense-MLA, and DSv4 specs normalize identically across
    xla, has, and sal."""
    dense_mla = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=None,
    )
    sparse_mla = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    dsv4_packed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        page_size_padded=None,
    )
    specs = {
        "dense_mla": dense_mla,
        "sparse_mla": sparse_mla,
        "dsv4_packed": dsv4_packed,
    }

    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "xla")
    baseline = normalize_kv_cache_specs_for_tpu(
        specs,
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )
    for mode in ("has", "sal"):
        monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", mode)
        current = normalize_kv_cache_specs_for_tpu(
            specs,
            torch.float8_e4m3fn,
            attention_backend=PallasAttentionBackend,
        )
        assert current == baseline


def test_deepseek_v32_indexer_backend_and_spec_normalizer_raiden_glm_fallback(
    monkeypatch, caplog
) -> None:
    """Default 'sal' falls back to HAS shape/page size when Raiden GLM admission
    is enabled, while explicit 'sal' raises ValueError and DSv4 admission alone
    does not downgrade."""
    import importlib
    import logging

    from vllm_torchtpu import envs

    st_mod = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(st_mod, "_INDEXER_KV_LAYOUT_FALLBACK_LOGGED", set())
    # Pin the flags as module attributes rather than through os.environ: the
    # helpers read `envs.<NAME>`, so this stays correct after enable_envs_cache()
    # and regardless of attributes other tests' monkeypatch undo left behind.
    # The mode is pinned both ways, since explicitness is read from os.environ.
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", False)
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_ADMISSION", False)
    monkeypatch.setattr(envs, "TPU_RAIDEN_DSV4_ADMISSION", False)

    nb, block_size, head_size = 8, 256, 129
    indexer_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        page_size_padded=None,
    )

    # 1. Default 'sal' + Raiden GLM admission -> HAS shape (nb, 64, 4, 256)
    #    and 256 B/token.
    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    caplog.clear()
    shape_fallback = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
        nb, block_size, 1, head_size, torch.uint8
    )
    assert shape_fallback == (nb, 64, 4, 256)
    page_fallback = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_page_size_bytes(
        block_size, 1, head_size, torch.uint8
    )
    assert page_fallback == block_size * 256
    norm_fallback = normalize_kv_cache_specs_for_tpu(
        {"indexer": indexer_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )["indexer"]
    assert norm_fallback.state_content_bytes == 256
    assert norm_fallback.page_size_bytes == 65536
    warnings = [
        r.message
        for r in caplog.records
        if "incompatible with Raiden GLM admission" in r.message
    ]
    assert len(warnings) == 1

    # 2. Explicit 'sal' + Raiden GLM admission -> ValueError.
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    with pytest.raises(ValueError, match="Raiden GLM admission"):
        VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
            nb, block_size, 1, head_size, torch.uint8
        )
    with pytest.raises(ValueError, match="Raiden GLM admission"):
        normalize_kv_cache_specs_for_tpu(
            {"indexer": indexer_spec},
            torch.float8_e4m3fn,
            attention_backend=PallasAttentionBackend,
        )

    # 3. DSv4 admission alone (with GLM admission off) -> keeps SAL (132 B/token).
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_ADMISSION", False)
    monkeypatch.setattr(envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    shape_dsv4 = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
        nb, block_size, 1, head_size, torch.uint8
    )
    assert shape_dsv4 == (nb, 33, 4, 256)
    norm_dsv4 = normalize_kv_cache_specs_for_tpu(
        {"indexer": indexer_spec},
        torch.float8_e4m3fn,
        attention_backend=PallasAttentionBackend,
    )["indexer"]
    assert norm_dsv4.state_content_bytes == 132
    assert norm_dsv4.page_size_bytes == 256 * 132
