import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, MambaSpec,
                                        MLAAttentionSpec)

from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasMLAttentionBackend)


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
    )
    spec = normalized["mla_layer"]
    pallas_mla_page_size = (
        PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
            16,
            1,
            576,
            torch.float8_e4m3fn,
        ))
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
    )

    assert normalized["attn"].page_size_bytes != normalized[
        "mamba"].page_size_bytes
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
        enable_unified_kv_layout=True,
    )

    assert normalized["attn"].page_size_bytes == normalized[
        "mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized[
        "attn"].page_size_bytes


def test_hybrid_specs_with_smaller_mamba_padded_size_normalizes_safely(
) -> None:
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
    )

    assert normalized["attn"].page_size_bytes == normalized[
        "mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized[
        "attn"].page_size_bytes
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
        shapes=[(274432, )],
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
        shapes=[(500000, )],
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
    )

    assert normalized["attn"].page_size_bytes == 1000000
    assert normalized["attn"].page_size_padded == 1000000
    assert normalized["mamba"].page_size_bytes == 1000000
    assert normalized["mamba"].page_size_padded == 1000000


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
    )

    assert normalized["ds_v4"] is ds_v4_spec
    assert normalized["ds_v4"].dtype == torch.bfloat16
    assert normalized["ds_v4"].page_size_padded == 100
    assert normalized["attn"].page_size_bytes == normalized[
        "mamba"].page_size_bytes
    assert normalized["attn"].dtype == torch.float8_e4m3fn


def test_exempt_attention_does_not_trigger_hybrid_unification_with_mamba(
) -> None:
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
    )

    assert normalized["ds_v4"] is ds_v4_spec
    assert normalized["mamba"] is mamba_spec
    assert normalized["mamba"].page_size_padded is None
