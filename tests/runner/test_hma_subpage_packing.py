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
"""Tests for pure-attention HMA sub-page packing."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import torch
from vllm import envs as vllm_envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu.kv_cache_materializer import (
    can_share_attention_cache,
    materialize_shared_attention_cache,
)
from vllm_torchtpu.layers.adapter import attention as attn_module
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackend,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.kernel_block_table import (
    KernelBlockTable,
    install_kernel_block_tables,
)
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.tpu_info import get_chip_version


def _build_vllm_cfg(
    uniform_page_bytes: int = 0,
    *,
    block_size: int = 256,
    user_specified_block_size: bool = False,
    max_model_len: int = 8192,
    max_num_reqs: int = 64,
    pcp_size: int = 1,
    dcp_size: int = 1,
    kv_layout: VllmKVCacheLayout = VllmKVCacheLayout.LBNHC,
    kv_transfer_config: KVTransferConfig | None = None,
) -> VllmConfig:
    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    try:
        with pytest.MonkeyPatch.context() as mp:
            if kv_layout in (VllmKVCacheLayout.LBHNC, VllmKVCacheLayout.BLHNC):
                mp.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
            else:
                mp.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)

            cache_cfg = CacheConfig(
                block_size=block_size if user_specified_block_size else None,
                cache_dtype="auto",
            )
            if not user_specified_block_size:
                cache_cfg.block_size = block_size
            assert cache_cfg.user_specified_block_size is user_specified_block_size
            cache_cfg.mamba_page_size_padded = uniform_page_bytes

            vllm_cfg = VllmConfig(
                device_config=DeviceConfig(device="cpu"),
                cache_config=cache_cfg,
                scheduler_config=SchedulerConfig(
                    max_num_seqs=max_num_reqs,
                    max_model_len=max_model_len,
                    is_encoder_decoder=False,
                ),
                parallel_config=ParallelConfig(
                    pipeline_parallel_size=1,
                    tensor_parallel_size=1,
                ),
                attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
            )
            resolve_kv_cache_layout(vllm_cfg, [["LBNHC", "LBHNC"]])
    finally:
        if was_cached:
            vllm_envs.enable_envs_cache()

    vllm_cfg.parallel_config.prefill_context_parallel_size = pcp_size
    vllm_cfg.parallel_config.decode_context_parallel_size = dcp_size
    vllm_cfg.kv_transfer_config = kv_transfer_config
    vllm_cfg.model_config = cast(
        Any,
        SimpleNamespace(max_model_len=max_model_len, get_head_size=lambda: 256),
    )
    assert vllm_cfg.cache_config.get_resolved_kv_cache_layout() == kv_layout
    return vllm_cfg


@pytest.fixture(autouse=True)
def _vllm_cfg_and_chip_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v7x-8")
    monkeypatch.setenv("VLLM_TPU_BLOCK_MAJOR_KV", "0")
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    get_chip_version.cache_clear()
    with set_current_vllm_config(_build_vllm_cfg()):
        yield
    get_chip_version.cache_clear()


def _make_fake_manager(
    uniform_page_bytes: int,
    *,
    block_size: int = 256,
    user_specified_block_size: bool = False,
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
        block_size=block_size,
        user_specified_block_size=user_specified_block_size,
        max_model_len=max_model_len,
        max_num_reqs=max_num_reqs,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        kv_layout=kv_layout,
        kv_transfer_config=kv_transfer_config,
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        block_size=block_size,
        kv_cache_dtype=torch.bfloat16,
        _hybrid_uniform_page_size_bytes=uniform_page_bytes,
        _unified_kv_layout=unified_kv_layout,
        _uniform_mamba_layout=False,
        _mamba_num_blocks=None,
        _is_async_drafter=False,
        drafter=None,
        shared_kv_cache_layers={},
        runner_only_attn_layers=set(),
        speculative_config=speculative_config,
        parallel_config=vllm_cfg.parallel_config,
        scheduler_config=vllm_cfg.scheduler_config,
        model_config=vllm_cfg.model_config,
        max_model_len=max_model_len,
        max_num_reqs=max_num_reqs,
        vllm_config=vllm_cfg,
        cache_config=vllm_cfg.cache_config,
    )
    return KVCacheManager(cast(Any, runner))


def _swa_spec(
    page_size_padded: int,
    *,
    block_size: int = 256,
    num_kv_heads: int = 8,
    head_size: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    sliding_window: int = 1024,
) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        sliding_window=sliding_window,
        page_size_padded=page_size_padded,
    )


def _full_spec(
    page_size_padded: int,
    *,
    block_size: int = 256,
    num_kv_heads: int = 2,
    head_size: int = 512,
    dtype: torch.dtype = torch.bfloat16,
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
    )


def _gemma4_tp2_specs() -> tuple[dict[str, KVCacheSpec], int, int]:
    swa_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 8, 256, torch.bfloat16
    )
    global_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 2, 512, torch.bfloat16
    )
    assert swa_bytes == 2 * global_bytes
    specs: dict[str, KVCacheSpec] = {
        "layer.0.attn": _swa_spec(swa_bytes),
        "layer.5.attn": _full_spec(swa_bytes),
    }
    return specs, swa_bytes, global_bytes


def _make_attention_layer_mock(
    *,
    num_kv_heads: int,
    head_size: int,
    sliding_window: int | None = None,
    kv_sharing_target: str | None = None,
    impl: PallasAttentionBackendImpl | None = None,
) -> MagicMock:
    layer = MagicMock(spec=Attention)
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    layer.sliding_window = sliding_window
    layer.kv_sharing_target_layer_name = kv_sharing_target
    layer.attn_type = AttentionType.DECODER
    layer.is_draft_layer = False
    layer.get_attn_backend.return_value = PallasBatchedRPAAttentionBackend
    if impl is not None:
        layer.impl = impl
    return layer


def _call_rpa_impl(
    kv_cache: jnp.ndarray,
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    block_tables: jnp.ndarray,
    *,
    seq_len: int,
    skip_kv_update: bool = False,
    subpage_split: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_q = query.shape[0]
    return attn_module._pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        jnp.array([seq_len], dtype=jnp.int32),
        block_tables,
        jnp.array([0, num_q], dtype=jnp.int32),
        jnp.array([0, 0, 1], dtype=jnp.int32),
        None,
        None,
        None,
        None,
        mesh=cast(Any, None),
        sliding_window=None,
        skip_kv_update=skip_kv_update,
        rpa_func=None,
        subpage_split=subpage_split,
    )


def _make_rpa_impl(
    *,
    num_heads: int = 4,
    head_size: int = 512,
    num_kv_heads: int = 2,
    sliding_window: int | None = None,
    kv_sharing_target_layer_name: str | None = None,
) -> PallasBatchedRPAAttentionBackendImpl:
    return PallasBatchedRPAAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        kv_sharing_target_layer_name=kv_sharing_target_layer_name,
    )


def test_tpu7x_shrinks_oversized_swa_to_match_global_page():
    """TPU_7X shrinks oversized SWA blocks 256->128 to share a 128-token 4D pool."""
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


def test_shrunk_swa_groups_are_compatible_with_prefix_caching_hash_size():
    """Shrunk [128, 256] groups resolve to scheduler_block_size=256, hash=128."""
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes, max_model_len=20096, max_num_reqs=128)
    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0.attn"], packed["layer.0.attn"]),
            KVCacheGroupSpec(["layer.5.attn"], packed["layer.5.attn"]),
        ],
    )
    vllm_cfg = _build_vllm_cfg(swa_bytes)
    vllm_cfg.cache_config.enable_prefix_caching = True

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_cfg
    )
    assert (scheduler_block_size, hash_block_size) == (256, 128)
    for spec in packed.values():
        assert cast(AttentionSpec, spec).block_size % hash_block_size == 0


def test_user_specified_block_size_skips_shrink_and_uses_scale_up():
    """Explicit `--block-size` skips SWA shrink and uses scale-up [256, 512]."""
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(
        swa_bytes,
        user_specified_block_size=True,
        max_model_len=20096,
        max_num_reqs=128,
    )

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 512
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes
    assert mgr.runner.cache_config.mamba_page_size_padded == swa_bytes


def test_tp8_fp8_scales_up_smaller_swa_when_global_page_is_larger():
    """When full-attention pages exceed SWA pages, SWA scales up 256->512."""
    swa_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 2, 256, torch.uint8
    )
    global_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        256, 1, 512, torch.uint8
    )
    assert global_bytes == 2 * swa_bytes

    mgr = _make_fake_manager(global_bytes)
    specs: dict[str, KVCacheSpec] = {
        "layer.0.attn": _swa_spec(global_bytes, num_kv_heads=2, dtype=torch.uint8),
        "layer.5.attn": _full_spec(global_bytes, num_kv_heads=1, dtype=torch.uint8),
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


def test_non_tpu7x_chip_skips_subpage_packing(monkeypatch: pytest.MonkeyPatch):
    """Non-TPU_7X chips keep the padded baseline layout."""
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    get_chip_version.cache_clear()

    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed is specs
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 256
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes


def test_smem_min_page_size_guard_falls_back_to_scale_up_at_long_context():
    """Long contexts exceeding the SMEM 128-page cap fall back to scale-up."""
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes, max_model_len=100608, max_num_reqs=256)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 512
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes


def test_non_multiple_sliding_window_still_allows_kernel_page_shrink():
    """Sliding windows not divisible by new_bs still shrink cleanly on TPU_7X."""
    specs, swa_bytes, global_bytes = _gemma4_tp2_specs()
    specs["layer.0.attn"] = _swa_spec(swa_bytes, sliding_window=320)
    mgr = _make_fake_manager(swa_bytes)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 128
    assert packed["layer.5.attn"].block_size == 256
    assert mgr.runner._hybrid_uniform_page_size_bytes == global_bytes


def test_get_kv_cache_spec_idempotent_across_repeated_calls():
    """Repeated get_kv_cache_spec() calls produce identical specs and page budgets."""
    _, swa_bytes, global_bytes = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes, max_model_len=20096, max_num_reqs=128)
    layers = {
        "layer.0.attn": _make_attention_layer_mock(
            num_kv_heads=8, head_size=256, sliding_window=1024
        ),
        "layer.5.attn": _make_attention_layer_mock(
            num_kv_heads=2, head_size=512, sliding_window=None
        ),
    }

    with (
        patch.object(
            TpuPlatform,
            "_find_non_ssm_backend",
            return_value=PallasBatchedRPAAttentionBackend,
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config",
            return_value=layers,
        ),
    ):
        first = mgr.get_kv_cache_spec()
        second = mgr.get_kv_cache_spec()

    assert first == second
    assert first["layer.0.attn"].block_size == 128
    assert first["layer.5.attn"].block_size == 256
    assert mgr.runner._hybrid_uniform_page_size_bytes == global_bytes
    assert mgr.runner.cache_config.mamba_page_size_padded == global_bytes


@pytest.mark.parametrize(
    "case",
    [
        "default_pallas_backend",
        "uniform_pages",
        "equal_native_bytes_multi_layer",
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
    case: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Unsupported backends, layouts, parallel modes, or page ratios skip packing."""
    specs, uniform_bytes, _ = _gemma4_tp2_specs()
    backend_cls: Any = PallasBatchedRPAAttentionBackend
    mgr_kwargs: dict[str, Any] = {}

    if case == "default_pallas_backend":
        backend_cls = PallasAttentionBackend
    elif case == "uniform_pages":
        specs = {"layer.0.attn": specs["layer.0.attn"]}
        uniform_bytes = 0
    elif case == "equal_native_bytes_multi_layer":
        specs = {
            "layer.0.attn": specs["layer.0.attn"],
            "layer.1.attn": _swa_spec(uniform_bytes, num_kv_heads=4, head_size=512),
        }
    elif case == "unified_kv_layout":
        mgr_kwargs["unified_kv_layout"] = True
    elif case == "speculative_config":
        mgr_kwargs["speculative_config"] = SimpleNamespace(method="mtp")
    elif case == "kv_transfer_config":
        mgr_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="TPUConnector", kv_role="kv_both"
        )
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
            page_size_padded=uniform_bytes,
        )
    elif case == "non_multiple_pages":
        p6 = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
            256, 6, 256, torch.bfloat16
        )
        specs = {
            "layer.0.attn": _swa_spec(p6, num_kv_heads=6),
            "layer.5.attn": _full_spec(p6, num_kv_heads=4, head_size=256),
        }
        uniform_bytes = p6

    mgr = _make_fake_manager(uniform_bytes, **mgr_kwargs)
    assert mgr._pack_attention_subpages(specs, backend_cls, 256) == specs
    assert mgr.runner._hybrid_uniform_page_size_bytes == uniform_bytes


def test_shared_cache_materializer_rejects_incompatible_subpage_specs():
    """Materializer helpers reject non-multiple block sizes."""
    page_bytes = PallasBatchedRPAAttentionBackend.get_kv_cache_page_size_bytes(
        384, 2, 512, torch.bfloat16
    )
    bad_specs: list[KVCacheSpec] = [
        _swa_spec(page_bytes, num_kv_heads=4),
        _full_spec(page_bytes, block_size=384),
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
    """4D pool reshape scales block_tables by page_stride // subpage_split."""
    captured_metadata: list[Any] = []
    captured_kv_shapes: list[tuple[int, ...]] = []

    def _fake_attention(kv_cache, query, key, value, metadata, *args, **kwargs):
        captured_kv_shapes.append(tuple(kv_cache.shape))
        captured_metadata.append(metadata)
        return kv_cache, query

    monkeypatch.setattr(attn_module, "attention", _fake_attention)

    pool_4d = jnp.zeros((4, 256, 2, 2048), dtype=jnp.bfloat16)
    query = jnp.zeros((2, 4, 512), dtype=jnp.bfloat16)
    key = jnp.zeros((2, 2, 512), dtype=jnp.bfloat16)
    value = jnp.zeros((2, 2, 512), dtype=jnp.bfloat16)
    block_tables = jnp.array([[2, 3, 6, 7]], dtype=jnp.int32)

    new_pool, _ = _call_rpa_impl(
        pool_4d, query, key, value, block_tables, seq_len=512, subpage_split=2
    )
    assert new_pool.shape == pool_4d.shape
    assert captured_kv_shapes[-1] == (8, 256, 2, 2, 512)
    assert jnp.array_equal(captured_metadata[-1].block_tables, block_tables)

    unsplit_tables = jnp.array([[1, 3]], dtype=jnp.int32)
    _call_rpa_impl(
        pool_4d, query, key, value, unsplit_tables, seq_len=512, subpage_split=1
    )
    assert jnp.array_equal(captured_metadata[-1].block_tables, unsplit_tables * 2)


def test_write_then_read_4d_pool_subpage_byte_equivalence(
    monkeypatch: pytest.MonkeyPatch,
):
    """Writing and reading 512 tokens across 2 sub-pages in a 4D pool is bit-exact."""
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    specs["layer.5.attn"] = _full_spec(swa_bytes, block_size=512)
    pool_torch = materialize_shared_attention_cache(
        layer_specs=[(k, cast(AttentionSpec, v)) for k, v in specs.items()],
        tensor_size=4 * swa_bytes,
        num_blocks=4,
        attn_backend=PallasBatchedRPAAttentionBackend,
        device=torch.device("cpu"),
    )
    assert tuple(pool_torch.shape) == (4, 256, 2, 2048)

    tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=512,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[256, 512],
        max_num_blocks=[4, 2],
    )
    install_kernel_block_tables(tables, [256, 256])
    tables.add_row(([1, 3], [2]), 0)
    kernel_row = tables.block_tables[1].get_numpy_array()[0, :2]
    assert kernel_row.tolist() == [4, 5]

    num_tokens = 512
    token_ids = jnp.arange(num_tokens, dtype=jnp.float32)[:, None, None]
    head_ids = jnp.arange(2, dtype=jnp.float32)[None, :, None]
    dim_ids = (jnp.arange(512, dtype=jnp.float32)[None, None, :] % 16.0) / 16.0
    key_512 = (token_ids + 0.25 * head_ids + dim_ids).astype(jnp.bfloat16)
    val_512 = (-token_ids - 0.5 * head_ids - dim_ids).astype(jnp.bfloat16)
    query_512 = jnp.zeros((num_tokens, 4, 512), dtype=jnp.bfloat16)

    read_back_kv: list[tuple[jnp.ndarray, jnp.ndarray]] = []

    def _paged_write_or_read(kv_cache, query, key, value, metadata, *args, **kwargs):
        page_size = kv_cache.shape[1]
        subpage_ids = metadata.block_tables[0]
        if kwargs.get("skip_kv_update", False):
            flat = kv_cache[subpage_ids].reshape(num_tokens, 2, 2, 512)
            read_back_kv.append((flat[:, :, 0, :], flat[:, :, 1, :]))
            return kv_cache, query

        paged = jnp.stack([key, value], axis=2).reshape(
            subpage_ids.shape[0], page_size, 2, 2, 512
        )
        return kv_cache.at[subpage_ids].set(paged), query

    monkeypatch.setattr(attn_module, "attention", _paged_write_or_read)

    pool_4d = jnp.zeros(tuple(pool_torch.shape), dtype=jnp.bfloat16)
    kernel_bt = jnp.asarray(kernel_row[None, :], dtype=jnp.int32)

    written_pool_4d, _ = _call_rpa_impl(
        pool_4d,
        query_512,
        key_512,
        val_512,
        kernel_bt,
        seq_len=num_tokens,
        subpage_split=2,
    )
    assert written_pool_4d.shape == (4, 256, 2, 2048)
    for b in (0, 1, 3):
        assert not bool(jnp.any(written_pool_4d[b] != 0))
    assert bool(jnp.any(written_pool_4d[2, :, :, :1024] != 0))
    assert bool(jnp.any(written_pool_4d[2, :, :, 1024:] != 0))

    _call_rpa_impl(
        written_pool_4d,
        query_512,
        key_512,
        val_512,
        kernel_bt,
        seq_len=num_tokens,
        skip_kv_update=True,
        subpage_split=2,
    )
    assert len(read_back_kv) == 1
    recovered_k, recovered_v = read_back_kv[0]
    assert jnp.array_equal(recovered_k, key_512)
    assert jnp.array_equal(recovered_v, val_512)


def test_default_pallas_backend_impl_rejects_subpage_split_gt_1():
    """Default PallasAttentionBackendImpl raises NotImplementedError on split > 1."""
    impl = PallasAttentionBackendImpl(
        num_heads=4,
        head_size=256,
        scale=256**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    impl.set_subpage_split(2)
    with (
        set_vllm_model_wrapper_context(mesh=cast(Any, None)),
        pytest.raises(NotImplementedError, match="does not support sub-page split"),
    ):
        impl._build_rpa_kernel(None, None, None)


@pytest.fixture
def subpage_glue_env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "vllm_torchtpu.runner.kv_cache_manager.synchronize_tensors",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
        lambda: False,
    )

    _, swa_bytes, _ = _gemma4_tp2_specs()
    vllm_cfg = _build_vllm_cfg(swa_bytes, max_model_len=8192, max_num_reqs=8)
    multi_bt = MultiGroupBlockTable(
        max_num_reqs=8,
        max_num_batched_tokens=512,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[256, 512],
        max_num_blocks=[5, 16],
    )
    recorded_kernel_block_sizes: list[list[int]] = []
    bound_kv_caches: dict[str, torch.Tensor] = {}

    def _may_reinit(cfg: KVCacheConfig, kbs: list[int]) -> None:
        recorded_kernel_block_sizes.append(list(kbs))
        install_kernel_block_tables(multi_bt, kbs)

    def _capture_bind(
        kv_caches: dict[str, torch.Tensor],
        forward_ctx: dict[str, Any],
        runner_kv_caches: list[torch.Tensor],
    ) -> None:
        bound_kv_caches.update(kv_caches)
        runner_kv_caches.extend(kv_caches.values())

    monkeypatch.setattr("vllm_torchtpu.utils.tpu_bind_kv_cache", _capture_bind)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=vllm_cfg,
        parallel_config=vllm_cfg.parallel_config,
        model_config=vllm_cfg.model_config,
        cache_config=vllm_cfg.cache_config,
        scheduler_config=vllm_cfg.scheduler_config,
        kv_cache_dtype=torch.bfloat16,
        use_spmd=False,
        enforce_eager=True,
        speculative_config=None,
        _unified_kv_layout=False,
        _uniform_mamba_layout=False,
        _mamba_num_blocks=None,
        _kv_cache_bundle=None,
        _kv_cache_bundle_layer_index={},
        shared_kv_cache_layers={},
        runner_only_attn_layers=set(),
        kv_caches=[],
        attn_groups=[],
        block_table_cpu=torch.zeros((8, 32), dtype=torch.int32),
        input_batch=SimpleNamespace(block_table=multi_bt),
        may_reinitialize_input_batch=_may_reinit,
        _precompile_dense_kv_cache_block_copies=lambda: None,
        _initialize_attention_kernels=lambda force=False: None,
    )
    return SimpleNamespace(
        runner=runner,
        vllm_cfg=vllm_cfg,
        multi_bt=multi_bt,
        swa_bytes=swa_bytes,
        recorded_kernel_block_sizes=recorded_kernel_block_sizes,
        bound_kv_caches=bound_kv_caches,
    )


def test_initialize_kv_cache_subpage_glue_and_shared_layer_aliases(
    subpage_glue_env: SimpleNamespace,
):
    """initialize_kv_cache wires kernel_block_sizes, 4D/5D pools, and KV aliases."""
    impl_swa = _make_rpa_impl(
        num_heads=8, head_size=256, num_kv_heads=8, sliding_window=1024
    )
    impl_full_shared_region = _make_rpa_impl()
    impl_full_unpaired_5d = _make_rpa_impl()
    impl_full_kv_alias = _make_rpa_impl(kv_sharing_target_layer_name="layer.1")

    subpage_glue_env.vllm_cfg.compilation_config.static_forward_context = {
        "layer.0": _make_attention_layer_mock(
            num_kv_heads=8, head_size=256, sliding_window=1024, impl=impl_swa
        ),
        "layer.1": _make_attention_layer_mock(
            num_kv_heads=2, head_size=512, impl=impl_full_shared_region
        ),
        "layer.2": _make_attention_layer_mock(
            num_kv_heads=2, head_size=512, impl=impl_full_unpaired_5d
        ),
        "layer.3": _make_attention_layer_mock(
            num_kv_heads=2,
            head_size=512,
            kv_sharing_target="layer.1",
            impl=impl_full_kv_alias,
        ),
    }
    subpage_glue_env.runner.shared_kv_cache_layers = {"layer.3": "layer.1"}

    num_blocks = 6
    swa_bytes = subpage_glue_env.swa_bytes
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * swa_bytes,
                layers=["layer.0", "layer.1"],
                layer_stride=0,
                block_stride=swa_bytes,
                offset=0,
            ),
            KVCacheTensor(
                size=num_blocks * swa_bytes,
                layers=["layer.2"],
                layer_stride=0,
                block_stride=swa_bytes,
                offset=num_blocks * swa_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer.0"], kv_cache_spec=_swa_spec(swa_bytes)
            ),
            KVCacheGroupSpec(
                layer_names=["layer.1", "layer.2"],
                kv_cache_spec=_full_spec(swa_bytes, block_size=512),
            ),
        ],
    )

    with patch.object(
        TpuPlatform,
        "_find_non_ssm_backend",
        return_value=PallasBatchedRPAAttentionBackend,
    ):
        KVCacheManager(cast(Any, subpage_glue_env.runner)).initialize_kv_cache(
            kv_cache_config
        )

    multi_bt = subpage_glue_env.multi_bt
    bound_kv_caches = subpage_glue_env.bound_kv_caches
    assert subpage_glue_env.recorded_kernel_block_sizes == [[256, 256]]
    assert not isinstance(multi_bt.block_tables[0], KernelBlockTable)
    assert isinstance(multi_bt.block_tables[1], KernelBlockTable)
    assert multi_bt.block_tables[1].blocks_per_kv_block == 2

    assert bound_kv_caches["layer.0"] is bound_kv_caches["layer.1"]
    assert bound_kv_caches["layer.3"] is bound_kv_caches["layer.1"]
    assert tuple(bound_kv_caches["layer.1"].shape) == (num_blocks, 256, 2, 2048)
    assert tuple(bound_kv_caches["layer.2"].shape) == (
        num_blocks * 2,
        256,
        2,
        2,
        512,
    )

    assert impl_swa._subpage_split == 1
    assert impl_full_shared_region._subpage_split == 2
    assert impl_full_unpaired_5d._subpage_split == 2
    assert impl_full_kv_alias._subpage_split == 2
    assert subpage_glue_env.runner._attention_kernel_block_size == 256
    assert subpage_glue_env.runner._attention_kv_cache_group_id == 1


def test_may_reinitialize_input_batch_uses_max_blocks_across_all_groups():
    """may_reinitialize_input_batch sizes block_table_cpu to max across all groups."""
    _, swa_bytes, _ = _gemma4_tp2_specs()
    max_num_reqs = 8
    max_model_len = 16384
    vllm_cfg = _build_vllm_cfg(
        swa_bytes, max_model_len=max_model_len, max_num_reqs=max_num_reqs
    )
    multi_bt = MultiGroupBlockTable(
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=512,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[256, 512],
        max_num_blocks=[5, 32],
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], _swa_spec(swa_bytes)),
            KVCacheGroupSpec(["layer.1"], _full_spec(swa_bytes, block_size=512)),
        ],
    )

    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vllm_config = vllm_cfg
    runner.max_num_reqs = max_num_reqs
    runner.max_num_local_reqs = max_num_reqs
    runner.max_model_len = max_model_len
    runner.most_model_len = max_model_len
    runner.max_num_tokens = 512
    runner.block_size = 256
    runner.max_num_blocks_per_req = 5
    runner.block_table_cpu = torch.zeros((max_num_reqs, 5), dtype=torch.int32)
    runner.shared_kv_cache_layers = {}
    runner.input_batch = cast(Any, SimpleNamespace(block_table=multi_bt))
    runner._attention_capacity = {}
    runner._decode_device_cache_key = None

    with patch.object(GPUModelRunner, "may_reinitialize_input_batch", autospec=True):
        runner.may_reinitialize_input_batch(kv_cache_config, [256, 256])

    assert isinstance(multi_bt.block_tables[1], KernelBlockTable)
    assert multi_bt.block_tables[0].max_num_blocks_per_req == 5
    assert multi_bt.block_tables[1].max_num_blocks_per_req == 64
    assert runner.max_num_blocks_per_req == 64
    assert tuple(runner.block_table_cpu.shape) == (max_num_reqs, 64)
