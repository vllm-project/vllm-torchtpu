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
"""Tests for topology-aware pure-attention HMA sub-page packing."""

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu.kv_cache_materializer import (
    can_share_attention_cache,
    materialize_shared_attention_cache,
)
from vllm_torchtpu.layers.adapter import attention as attn_module
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasBatchedRPAAttentionBackend,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.tpu_info import get_chip_version


def _build_vllm_cfg(
    uniform_page_bytes: int = 0,
    *,
    max_model_len: int = 8192,
    kv_layout: VllmKVCacheLayout = VllmKVCacheLayout.LBNHC,
    kv_transfer_config: Any = None,
) -> VllmConfig:
    vllm_cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_cfg.cache_config = SimpleNamespace(
        mamba_page_size_padded=uniform_page_bytes,
        kv_cache_layout=kv_layout,
        get_resolved_kv_cache_layout=lambda: kv_layout,
    )
    vllm_cfg.model_config = SimpleNamespace(max_model_len=max_model_len)
    vllm_cfg.kv_transfer_config = kv_transfer_config
    return vllm_cfg


@pytest.fixture(autouse=True)
def _vllm_cfg_and_chip_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v7x-8")
    monkeypatch.setenv("VLLM_TPU_BLOCK_MAJOR_KV", "0")
    get_chip_version.cache_clear()
    with set_current_vllm_config(_build_vllm_cfg()):
        yield
    get_chip_version.cache_clear()


def _make_fake_manager(
    uniform_page_bytes: int,
    *,
    max_model_len: int = 8192,
    max_num_reqs: int = 64,
    unified_kv_layout: bool = False,
    speculative_config: Any = None,
    kv_transfer_config: Any = None,
    pcp_size: int = 1,
    dcp_size: int = 1,
    kv_layout: VllmKVCacheLayout = VllmKVCacheLayout.LBNHC,
) -> KVCacheManager:
    vllm_cfg = _build_vllm_cfg(
        uniform_page_bytes,
        max_model_len=max_model_len,
        kv_layout=kv_layout,
        kv_transfer_config=kv_transfer_config,
    )
    runner = SimpleNamespace(
        _hybrid_uniform_page_size_bytes=uniform_page_bytes,
        _unified_kv_layout=unified_kv_layout,
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        max_model_len=max_model_len,
        max_num_reqs=max_num_reqs,
        vllm_config=vllm_cfg,
        cache_config=vllm_cfg.cache_config,
    )
    return KVCacheManager(cast(Any, runner))


def _gemma4_tp2_specs() -> tuple[dict[str, KVCacheSpec], int, int]:
    swa_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 8, 256, torch.bfloat16
    )
    global_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 2, 512, torch.bfloat16
    )
    assert swa_bytes == 2 * global_bytes
    specs: dict[str, KVCacheSpec] = {
        "layer.0.attn": SlidingWindowSpec(
            block_size=256,
            num_kv_heads=8,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
            page_size_padded=swa_bytes,
        ),
        "layer.5.attn": FullAttentionSpec(
            block_size=256,
            num_kv_heads=2,
            head_size=512,
            dtype=torch.bfloat16,
            page_size_padded=swa_bytes,
        ),
    }
    return specs, swa_bytes, global_bytes


def test_tpu7x_shrinks_oversized_swa_to_match_global_page():
    """TPU_7X TP=2 regime: SWA (1.0 MiB at bs=256) > Global (0.5 MiB at bs=256)
    shrinks SWA block_size 256 -> 128 when SMEM request caps are preserved.
    """
    specs, swa_bytes, global_bytes = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes, max_model_len=20096, max_num_reqs=128)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 128
    assert packed["layer.5.attn"].block_size == 256
    assert packed["layer.0.attn"].page_size_bytes == global_bytes
    assert packed["layer.5.attn"].page_size_bytes == global_bytes
    assert mgr.runner._hybrid_uniform_page_size_bytes == global_bytes
    assert mgr.runner.cache_config.mamba_page_size_padded == global_bytes
    assert can_share_attention_cache(
        list(packed.values()), PallasBatchedRPAAttentionBackend
    )

    pool = materialize_shared_attention_cache(
        layer_specs=[(k, cast(AttentionSpec, v)) for k, v in packed.items()],
        tensor_size=10 * global_bytes,
        num_blocks=10,
        attn_backend=PallasBatchedRPAAttentionBackend,
        device=torch.device("cpu"),
    )
    assert pool.ndim == 4
    assert pool.numel() * pool.element_size() == 10 * global_bytes
    assert pool.shape[1] == 128


def test_tp8_fp8_scales_up_smaller_swa_when_global_page_is_larger():
    """TP=8 FP8 regime: 1 KV head at uint8 pads 2*H_kv=2 -> 4 for 32-bit packing,
    so Global (D=512 -> 0.5 MiB) > SWA (D=256 -> 0.25 MiB). Scales SWA 256 -> 512
    and materializes a 4D shared pool at min_block_size=256.
    """
    swa_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 2, 256, torch.uint8
    )
    global_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 512, torch.uint8
    )
    assert global_bytes == 2 * swa_bytes

    mgr = _make_fake_manager(global_bytes)
    specs: dict[str, KVCacheSpec] = {
        "layer.0.attn": SlidingWindowSpec(
            block_size=256,
            num_kv_heads=2,
            head_size=256,
            dtype=torch.uint8,
            sliding_window=1024,
            page_size_padded=global_bytes,
        ),
        "layer.5.attn": FullAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            page_size_padded=global_bytes,
        ),
    }

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 512
    assert packed["layer.5.attn"].block_size == 256
    assert can_share_attention_cache(
        list(packed.values()), PallasBatchedRPAAttentionBackend
    )

    pool = materialize_shared_attention_cache(
        layer_specs=[(k, cast(AttentionSpec, v)) for k, v in packed.items()],
        tensor_size=10 * global_bytes,
        num_blocks=10,
        attn_backend=PallasBatchedRPAAttentionBackend,
        device=torch.device("cpu"),
    )
    assert pool.ndim == 4
    assert pool.numel() * pool.element_size() == 10 * global_bytes
    assert pool.shape[1] == 256


def test_non_tpu7x_chip_skips_tile_shrink_and_uses_subpage_scale_up(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-TPU_7X chips (e.g. v6e) keep kernel_block_size=256 by scaling FullAttention
    256 -> 512 via KernelBlockTable instead of shrinking SWA's physical tile size.
    """
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    get_chip_version.cache_clear()

    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 512
    assert packed["layer.0.attn"].page_size_bytes == swa_bytes
    assert packed["layer.5.attn"].page_size_bytes == swa_bytes
    assert can_share_attention_cache(
        list(packed.values()), PallasBatchedRPAAttentionBackend
    )


def test_smem_request_cap_guard_falls_back_to_scale_up_at_long_context():
    """At max_model_len=100608 and max_num_reqs=256, shrinking kernel_block_size to 128
    would widen FullAttention's page table to 786 sub-pages (SMEM cap 166 < 256).
    The guard preserves max_num_reqs via scale_up [256, 512].
    """
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes, max_model_len=100608, max_num_reqs=256)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 512
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes


def test_non_divisible_sliding_window_falls_back_to_scale_up():
    """If sliding_window is not divisible by the shrunk block_size (320 % 128 != 0),
    SWA shrink is skipped and FullAttention scales up 256 -> 512 instead.
    """
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    specs["layer.0.attn"] = SlidingWindowSpec(
        block_size=256,
        num_kv_heads=8,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=320,
        page_size_padded=swa_bytes,
    )
    mgr = _make_fake_manager(swa_bytes)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 512


@pytest.mark.parametrize(
    "case",
    [
        "default_pallas_backend",
        "uniform_pages",
        "unified_kv_layout",
        "speculative_config",
        "kv_transfer_config",
        "block_major_kv",
        "pcp",
        "dcp",
        "hnd_layout",
        "mla_mixed",
        "non_multiple_pages",
    ],
)
def test_non_regression_guards_leave_specs_untouched(
    case: str, monkeypatch: pytest.MonkeyPatch
):
    """Every unsupported backend, layout, parallelism, connector, or non-multiple
    page-size ratio returns the original specs untouched with zero side effects.
    """
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    backend_cls: Any = PallasBatchedRPAAttentionBackend
    mgr_kwargs: dict[str, Any] = {}

    if case == "default_pallas_backend":
        backend_cls = PallasAttentionBackend
    elif case == "uniform_pages":
        specs = {"layer.0.attn": specs["layer.0.attn"]}
        mgr = _make_fake_manager(0)
        assert mgr._pack_attention_subpages(specs, backend_cls, 256) == specs
        return
    elif case == "unified_kv_layout":
        mgr_kwargs["unified_kv_layout"] = True
    elif case == "speculative_config":
        mgr_kwargs["speculative_config"] = SimpleNamespace(method="mtp")
    elif case == "kv_transfer_config":
        mgr_kwargs["kv_transfer_config"] = SimpleNamespace(kv_connector="TPUConnector")
    elif case == "block_major_kv":
        monkeypatch.setenv("VLLM_TPU_BLOCK_MAJOR_KV", "1")
    elif case == "pcp":
        mgr_kwargs["pcp_size"] = 2
    elif case == "dcp":
        mgr_kwargs["dcp_size"] = 2
    elif case == "hnd_layout":
        mgr_kwargs["kv_layout"] = VllmKVCacheLayout.LBHNC
    elif case == "mla_mixed":
        specs["layer.9.mla"] = MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            page_size_padded=swa_bytes,
        )
    elif case == "non_multiple_pages":
        # 6 KV heads * 256 = 0.75 MiB vs 4 KV heads * 256 = 0.5 MiB (ratio 3:2)
        p6 = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
            256, 6, 256, torch.bfloat16
        )
        specs = {
            "layer.0.attn": SlidingWindowSpec(
                block_size=256,
                num_kv_heads=6,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
                page_size_padded=p6,
            ),
            "layer.5.attn": FullAttentionSpec(
                block_size=256,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.bfloat16,
                page_size_padded=p6,
            ),
        }
        swa_bytes = p6

    mgr = _make_fake_manager(swa_bytes, **mgr_kwargs)
    packed = mgr._pack_attention_subpages(specs, backend_cls, 256)
    assert packed == specs
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes


def test_shared_cache_materializer_rejects_incompatible_subpage_specs():
    """can_share_attention_cache and materialize_shared_attention_cache reject
    non-multiple block sizes and mismatched tensor byte budgets.
    """
    page_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        384, 2, 512, torch.bfloat16
    )
    bad_specs: list[KVCacheSpec] = [
        SlidingWindowSpec(
            block_size=256,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
            page_size_padded=page_bytes,
        ),
        FullAttentionSpec(
            block_size=384,  # Not a multiple of min_block_size=256
            num_kv_heads=2,
            head_size=512,
            dtype=torch.bfloat16,
            page_size_padded=page_bytes,
        ),
    ]
    assert not can_share_attention_cache(bad_specs, PallasBatchedRPAAttentionBackend)

    with pytest.raises(
        ValueError, match="Shared pool page must contain whole native pages"
    ):
        materialize_shared_attention_cache(
            layer_specs=[
                ("l0", cast(AttentionSpec, bad_specs[0])),
                ("l1", cast(AttentionSpec, bad_specs[1])),
            ],
            tensor_size=4 * page_bytes,
            num_blocks=4,
            attn_backend=PallasBatchedRPAAttentionBackend,
            device=torch.device("cpu"),
        )


def test_pallas_rpa_kernel_impl_4d_pool_rem_stride_for_subpages(
    monkeypatch: pytest.MonkeyPatch,
):
    """4D shared pool reshape divides page_stride by subpage_split so KernelBlockTable's
    expanded sub-page IDs address every sub-page without skipping odd sub-pages, while
    subpage_split=1 preserves page_stride multiplication for padded unified pools.
    """
    captured_metadata: list[Any] = []
    captured_kv_shapes: list[tuple[int, ...]] = []

    def _fake_attention(kv_cache, query, key, value, metadata, *args, **kwargs):
        captured_kv_shapes.append(tuple(kv_cache.shape))
        captured_metadata.append(metadata)
        return kv_cache, query

    monkeypatch.setattr(attn_module, "attention", _fake_attention)

    # 4 physical blocks of 1.0 MiB each (min_block_size=256, packing=2, words=2048).
    # Global layer (2 KV heads, D=512, bf16) has native page 0.5 MiB -> page_stride=2.
    pool_4d = jnp.zeros((4, 256, 2, 2048), dtype=jnp.bfloat16)
    query = jnp.zeros((2, 4, 512), dtype=jnp.bfloat16)
    key = jnp.zeros((2, 2, 512), dtype=jnp.bfloat16)
    value = jnp.zeros((2, 2, 512), dtype=jnp.bfloat16)
    block_tables = jnp.array([[2, 3, 6, 7]], dtype=jnp.int32)
    seq_lens = jnp.array([512], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    # Sub-page packed (subpage_split=2): rem_stride = 2 // 2 = 1 -> tables unchanged.
    new_pool, _ = attn_module._pallas_rpa_kernel_impl(
        pool_4d,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        None,
        None,
        None,
        None,
        mesh=cast(Any, None),
        sliding_window=None,
        skip_kv_update=False,
        rpa_func=None,
        subpage_split=2,
    )
    assert new_pool.shape == pool_4d.shape
    assert captured_kv_shapes[-1] == (8, 256, 2, 2, 512)
    assert jnp.array_equal(captured_metadata[-1].block_tables, block_tables)

    # Padded pool (subpage_split=1): rem_stride = 2 // 1 = 2 -> block_tables * 2.
    unsplit_tables = jnp.array([[1, 3]], dtype=jnp.int32)
    attn_module._pallas_rpa_kernel_impl(
        pool_4d,
        query,
        key,
        value,
        seq_lens,
        unsplit_tables,
        query_start_loc,
        request_distribution,
        None,
        None,
        None,
        None,
        mesh=cast(Any, None),
        sliding_window=None,
        skip_kv_update=False,
        rpa_func=None,
        subpage_split=1,
    )
    assert jnp.array_equal(captured_metadata[-1].block_tables, unsplit_tables * 2)


def test_backend_impl_separates_4d_and_5d_custom_op_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    """Layers sharing identical attention geometry and subpage_split=2 but receiving
    4D shared pools vs 5D unpaired single-layer caches build distinct custom ops.
    """
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    built_ops: list[str] = []

    class _FakeOp:
        def __init__(self, name: str):
            self.name = name

        def register_fake(self, fn):
            return None

        def __call__(self, kv_cache, *args, **kwargs):
            return kv_cache.clone(), args[0].clone()

    def _fake_jax_op(name: str, fn, **kwargs):
        built_ops.append(name)
        return _FakeOp(name)

    monkeypatch.setattr(attn_module.pallas, "jax_op", _fake_jax_op)

    impl = PallasBatchedRPAAttentionBackendImpl(
        num_heads=4,
        head_size=512,
        scale=1.0 / (512**0.5),
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    impl_5d = PallasBatchedRPAAttentionBackendImpl(
        num_heads=4,
        head_size=512,
        scale=1.0 / (512**0.5),
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    with set_vllm_model_wrapper_context(mesh=cast(Any, None)):
        impl.set_subpage_split(2)
        impl.set_kv_cache_ndim(4)
        op_4d = impl._build_rpa_kernel(None, None, None)

        impl_5d.set_subpage_split(2)
        impl_5d.set_kv_cache_ndim(5)
        op_5d = impl_5d._build_rpa_kernel(None, None, None)

    assert op_4d is not op_5d
    assert len(built_ops) == 2
