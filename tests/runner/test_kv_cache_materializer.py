import pytest
import torch
from vllm.v1.kv_cache_interface import (EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.kv_cache_materializer import (
    format_kv_cache_layout_summary, materialize_kv_cache_tensors)


class FakeAttentionBackend:

    @staticmethod
    def get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str,
    ):
        return (num_blocks, block_size, num_kv_heads, 2, head_size)


def make_attention_only_config(num_blocks: int = 4) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.0.self_attn"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            )
        ],
    )


def make_hybrid_config(
    num_blocks: int = 4,
    attn_dtype: torch.dtype = torch.bfloat16,
) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=attn_dtype,
        page_size_padded=page_size,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2, 8), (2, 4, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.0.self_attn", "model.layers.1.mamba"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.mamba"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )


def make_attention_encoder_attention_config(
        num_blocks: int = 4) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    encoder_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.0.self_attn"],
            ),
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=["model.layers.2.self_attn"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.encoder_attn"],
                kv_cache_spec=encoder_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.2.self_attn"],
                kv_cache_spec=attn_spec,
            ),
        ],
    )


def make_hybrid_with_encoder_gap_config(num_blocks: int = 4) -> KVCacheConfig:
    page_size = 16 * 2 * 2 * 8 * torch.bfloat16.itemsize
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    encoder_spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((2, 8), (2, 4, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page_size * num_blocks,
                shared_by=[
                    "model.layers.0.self_attn",
                    "model.layers.2.self_attn",
                    "model.layers.3.mamba",
                ],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.encoder_attn"],
                kv_cache_spec=encoder_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.2.self_attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.3.mamba"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )


def make_attention_groups(cfg: KVCacheConfig) -> list[list[AttentionGroup]]:
    return [[
        AttentionGroup(
            backend=FakeAttentionBackend,
            layer_names=list(group.layer_names),
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=gid,
        )
    ] for gid, group in enumerate(cfg.kv_cache_groups)]


def test_attention_only_allocates_direct_cache_without_raw_backing():
    cfg = make_attention_only_config(num_blocks=4)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group]],
        kernel_block_sizes=[16],
        device=torch.device("cpu"),
        cache_dtype=torch.bfloat16,
    )

    assert materialized.raw_tensors == []
    attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    assert isinstance(attn_cache, torch.Tensor)
    assert attn_cache.shape == (4, 16, 2, 2, 8)
    assert attn_cache.dtype == torch.bfloat16
    assert attn_cache.is_contiguous()


def test_direct_attention_materialization_maps_kernel_block_sizes_after_encoder_only_gap(
):
    cfg = make_attention_encoder_attention_config(num_blocks=4)

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=make_attention_groups(cfg),
        kernel_block_sizes=[16, 8],
        device=torch.device("cpu"),
        cache_dtype=torch.bfloat16,
    )

    assert "model.layers.1.encoder_attn" not in materialized.kv_caches
    first_attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    second_attn_cache = materialized.kv_caches["model.layers.2.self_attn"]
    assert isinstance(first_attn_cache, torch.Tensor)
    assert isinstance(second_attn_cache, torch.Tensor)
    assert first_attn_cache.shape == (4, 16, 2, 2, 8)
    assert second_attn_cache.shape == (8, 8, 2, 2, 8)


def test_hybrid_shared_by_uses_one_raw_backing_tensor(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_KV_CACHE_ALIAS_FALLBACK", "0")
    cfg = make_hybrid_config(num_blocks=4)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )
    mamba_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.1.mamba"],
        kv_cache_spec=cfg.kv_cache_groups[1].kv_cache_spec,
        kv_cache_group_id=1,
    )

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group], [mamba_group]],
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )
    kv_caches = materialized.kv_caches
    raw_tensors = materialized.raw_tensors

    assert len(raw_tensors) == 1
    assert raw_tensors[0].dtype == torch.int8
    assert raw_tensors[0].numel() == cfg.kv_cache_tensors[0].size
    assert raw_tensors[0].shape == (cfg.kv_cache_tensors[0].size, )

    attn_cache = kv_caches["model.layers.0.self_attn"]
    mamba_cache = kv_caches["model.layers.1.mamba"]

    assert isinstance(attn_cache, torch.Tensor)
    assert isinstance(mamba_cache, list)
    assert attn_cache.shape[0] == cfg.num_blocks
    assert mamba_cache[0].shape[0] == cfg.num_blocks
    assert mamba_cache[1].shape[0] == cfg.num_blocks

    raw_storage = raw_tensors[0].untyped_storage().data_ptr()
    assert attn_cache.untyped_storage().data_ptr() == raw_storage
    assert mamba_cache[0].untyped_storage().data_ptr() == raw_storage
    assert mamba_cache[1].untyped_storage().data_ptr() == raw_storage


def test_hybrid_mamba_views_can_use_explicit_block_count(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_KV_CACHE_ALIAS_FALLBACK", "0")
    cfg = make_hybrid_config(num_blocks=4)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )
    mamba_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.1.mamba"],
        kv_cache_spec=cfg.kv_cache_groups[1].kv_cache_spec,
        kv_cache_group_id=1,
    )

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group], [mamba_group]],
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
        mamba_num_blocks=2,
    )

    attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    mamba_cache = materialized.kv_caches["model.layers.1.mamba"]

    assert isinstance(attn_cache, torch.Tensor)
    assert isinstance(mamba_cache, list)
    assert attn_cache.shape[0] == cfg.num_blocks
    assert mamba_cache[0].shape[0] == 2
    assert mamba_cache[1].shape[0] == 2


def test_hybrid_materialization_maps_kernel_block_sizes_after_encoder_only_gap(
        monkeypatch):
    monkeypatch.setenv("TPU_VLLM_KV_CACHE_ALIAS_FALLBACK", "0")
    cfg = make_hybrid_with_encoder_gap_config(num_blocks=4)

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=make_attention_groups(cfg),
        kernel_block_sizes=[16, 8, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )

    assert "model.layers.1.encoder_attn" not in materialized.kv_caches
    first_attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    second_attn_cache = materialized.kv_caches["model.layers.2.self_attn"]
    mamba_cache = materialized.kv_caches["model.layers.3.mamba"]
    assert isinstance(first_attn_cache, torch.Tensor)
    assert isinstance(second_attn_cache, torch.Tensor)
    assert isinstance(mamba_cache, list)
    assert first_attn_cache.shape == (4, 16, 2, 2, 8)
    assert second_attn_cache.shape == (8, 8, 2, 2, 8)
    assert mamba_cache[0].shape[0] == cfg.num_blocks
    assert mamba_cache[1].shape[0] == cfg.num_blocks


@pytest.mark.parametrize("kernel_block_sizes", ([16], [16, 8, 16]))
def test_materialization_rejects_mismatched_kernel_block_size_count(
        kernel_block_sizes):
    cfg = make_attention_encoder_attention_config(num_blocks=4)

    with pytest.raises(ValueError):
        materialize_kv_cache_tensors(
            kv_cache_config=cfg,
            attn_groups=make_attention_groups(cfg),
            kernel_block_sizes=kernel_block_sizes,
            device=torch.device("cpu"),
            cache_dtype=torch.bfloat16,
        )


def test_cache_clone_fallback_detaches_attention_and_mamba_from_raw(
        monkeypatch):
    monkeypatch.delenv("TPU_VLLM_KV_CACHE_ALIAS_FALLBACK", raising=False)
    cfg = make_hybrid_config(num_blocks=4)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )
    mamba_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.1.mamba"],
        kv_cache_spec=cfg.kv_cache_groups[1].kv_cache_spec,
        kv_cache_group_id=1,
    )

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group], [mamba_group]],
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )

    raw_storage = materialized.raw_tensors[0].untyped_storage().data_ptr()
    attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    mamba_cache = materialized.kv_caches["model.layers.1.mamba"]

    assert isinstance(attn_cache, torch.Tensor)
    assert isinstance(mamba_cache, list)
    assert attn_cache.is_contiguous()
    assert attn_cache.untyped_storage().data_ptr() != raw_storage
    assert torch.count_nonzero(attn_cache).item() == 0
    assert mamba_cache[0].is_contiguous()
    assert mamba_cache[1].is_contiguous()
    assert mamba_cache[0].untyped_storage().data_ptr() != raw_storage
    assert mamba_cache[1].untyped_storage().data_ptr() != raw_storage
    assert torch.count_nonzero(mamba_cache[0]).item() == 0
    assert torch.count_nonzero(mamba_cache[1]).item() == 0


def test_cache_clone_fallback_uses_empty_for_fp8_attention(monkeypatch):
    monkeypatch.delenv("TPU_VLLM_KV_CACHE_ALIAS_FALLBACK", raising=False)
    torch_zeros = torch.zeros

    def zeros_without_fp8(*args, **kwargs):
        if kwargs.get("dtype") in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError("fp8 fallback must not require zero_")
        return torch_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros_without_fp8)
    cfg = make_hybrid_config(num_blocks=4, attn_dtype=torch.float8_e4m3fn)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )
    mamba_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.1.mamba"],
        kv_cache_spec=cfg.kv_cache_groups[1].kv_cache_spec,
        kv_cache_group_id=1,
    )

    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group], [mamba_group]],
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="fp8",
    )

    attn_cache = materialized.kv_caches["model.layers.0.self_attn"]
    mamba_cache = materialized.kv_caches["model.layers.1.mamba"]

    assert isinstance(attn_cache, torch.Tensor)
    assert isinstance(mamba_cache, list)
    assert attn_cache.dtype == torch.float8_e4m3fn
    assert attn_cache.shape == (4, 16, 2, 2, 8)
    assert torch.count_nonzero(mamba_cache[0]).item() == 0
    assert torch.count_nonzero(mamba_cache[1]).item() == 0


def test_layout_summary_includes_attention_and_mamba_shapes(monkeypatch):
    monkeypatch.setenv("TPU_VLLM_KV_CACHE_ALIAS_FALLBACK", "0")
    cfg = make_hybrid_config(num_blocks=4)
    attn_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=cfg.kv_cache_groups[0].kv_cache_spec,
        kv_cache_group_id=0,
    )
    mamba_group = AttentionGroup(
        backend=FakeAttentionBackend,
        layer_names=["model.layers.1.mamba"],
        kv_cache_spec=cfg.kv_cache_groups[1].kv_cache_spec,
        kv_cache_group_id=1,
    )
    materialized = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=[[attn_group], [mamba_group]],
        kernel_block_sizes=[16, 16],
        device=torch.device("cpu"),
        cache_dtype="bfloat16",
    )

    summary = format_kv_cache_layout_summary(
        kv_cache_config=cfg,
        kv_caches=materialized.kv_caches,
        raw_tensors=materialized.raw_tensors,
        attn_groups=[[attn_group], [mamba_group]],
    )

    assert "TPU KV cache layout summary:" in summary
    assert "raw[0]: shape=(4096,)" in summary
    assert ("group[0]: spec=FullAttentionSpec backend=FakeAttentionBackend "
            "block_size=16" in summary)
    assert "model.layers.0.self_attn: attention shape=(4, 16, 2, 2, 8)" in summary
    assert "raw_idx=0" in summary
    assert (
        "group[1]: spec=MambaSpec backend=FakeAttentionBackend block_size=16"
        in summary)
    assert "model.layers.1.mamba: mamba states=2" in summary
    assert "state[0]: shape=(4, 2, 8)" in summary
    assert "state[1]: shape=(4, 2, 4, 8)" in summary
    assert "contiguous=False" in summary
    assert "storage_offset_bytes=32" in summary
