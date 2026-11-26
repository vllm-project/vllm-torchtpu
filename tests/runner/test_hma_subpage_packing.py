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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm import envs as vllm_envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.config.attention import AttentionConfig
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from tests.runner.runner_input_test_utils import attention_layer_mock
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kv_cache_materializer import (
    can_share_attention_cache,
    materialize_shared_attention_cache,
)
from vllm_torchtpu.kv_cache_spec_normalizer import resolve_subpage_packing_block_size
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
from vllm_torchtpu.runner.vllm_v1.worker.block_table import MultiGroupBlockTable
from vllm_torchtpu.tpu_info import get_chip_version

BACKEND = PallasBatchedRPAAttentionBackend
# Gemma 4 31B per-rank geometry at TP=2: the SWA page is 2x the global page.
SWA = {"num_kv_heads": 8, "head_size": 256, "sliding_window": 1024}
GLOBAL = {"num_kv_heads": 2, "head_size": 512}


def _page_bytes(block_size: int, layer: dict[str, int], dtype=torch.bfloat16) -> int:
    return BACKEND.get_kv_cache_page_size_bytes(
        block_size, layer["num_kv_heads"], layer["head_size"], dtype
    )


# The shared `vllm_config_context` fixture builds a default `VllmConfig`; these
# tests need the block size, `--block-size` origin, layout and parallel sizes.
def _build_vllm_cfg(
    *,
    block_size: int = 256,
    user_specified_block_size: bool = False,
    max_model_len: int = 4096,
    max_num_reqs: int = 8,
    pcp_size: int = 1,
    dcp_size: int = 1,
    kv_layout: VllmKVCacheLayout = VllmKVCacheLayout.LBNHC,
    kv_transfer_config: KVTransferConfig | None = None,
    speculative_config: Any = None,
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
    vllm_cfg.speculative_config = speculative_config
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


def _make_runner(
    vllm_cfg: VllmConfig,
    *,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
    unified_kv_layout: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        block_size=vllm_cfg.cache_config.block_size,
        kv_cache_dtype=kv_cache_dtype,
        _hybrid_uniform_page_size_bytes=None,
        _subpage_kernel_block_size=None,
        _unified_kv_layout=unified_kv_layout,
        _uniform_mamba_layout=False,
        _mamba_num_blocks=None,
        _kv_cache_bundle=None,
        _kv_cache_bundle_layer_index={},
        _is_async_drafter=False,
        drafter=None,
        use_spmd=False,
        enforce_eager=True,
        shared_kv_cache_layers={},
        runner_only_attn_layers=set(),
        kv_caches=[],
        attn_groups=[],
        block_table_cpu=torch.zeros((8, 32), dtype=torch.int32),
        speculative_config=vllm_cfg.speculative_config,
        parallel_config=vllm_cfg.parallel_config,
        scheduler_config=vllm_cfg.scheduler_config,
        model_config=vllm_cfg.model_config,
        max_model_len=vllm_cfg.model_config.max_model_len,
        max_num_reqs=vllm_cfg.scheduler_config.max_num_seqs,
        vllm_config=vllm_cfg,
        cache_config=vllm_cfg.cache_config,
        _precompile_dense_kv_cache_block_copies=lambda: None,
        _initialize_attention_kernels=lambda force=False: None,
    )


def _make_manager(
    *,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
    unified_kv_layout: bool = False,
    **cfg_kwargs: Any,
) -> KVCacheManager:
    vllm_cfg = _build_vllm_cfg(**cfg_kwargs)
    runner = _make_runner(
        vllm_cfg, kv_cache_dtype=kv_cache_dtype, unified_kv_layout=unified_kv_layout
    )
    return KVCacheManager(cast(Any, runner))


def _layers(
    swa: dict[str, int] = SWA,
    full: dict[str, int] = GLOBAL,
    *,
    num_swa: int = 6,
    num_full: int = 2,
) -> dict[str, MagicMock]:
    layers = {f"swa.{i}": attention_layer_mock(BACKEND, **swa) for i in range(num_swa)}
    layers.update(
        {f"full.{i}": attention_layer_mock(BACKEND, **full) for i in range(num_full)}
    )
    return layers


def _kv_cache_spec(
    mgr: KVCacheManager,
    layers: dict[str, MagicMock],
    backend: type[PallasAttentionBackend] = BACKEND,
) -> dict[str, KVCacheSpec]:
    with (
        patch.object(TpuPlatform, "_find_non_ssm_backend", return_value=backend),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.get_layers_from_vllm_config",
            return_value=layers,
        ),
    ):
        return mgr.get_kv_cache_spec()


def _planned_block_sizes(mgr: KVCacheManager, specs: dict[str, KVCacheSpec]):
    """`{spec type: block_size}` and the common page after vLLM's grouping."""
    groups = get_kv_cache_groups(mgr.runner.vllm_config, dict(specs))
    block_sizes = {
        type(g.kv_cache_spec).__name__: g.kv_cache_spec.block_size for g in groups
    }
    pages = {g.kv_cache_spec.page_size_bytes for g in groups}
    assert len(pages) == 1
    return groups, block_sizes, pages.pop()


def _is_padded_baseline(specs: dict[str, KVCacheSpec], block_size: int = 256) -> bool:
    """Every spec keeps `block_size` and a pinned page, as without packing."""
    return all(
        cast(AttentionSpec, s).block_size == block_size
        and cast(AttentionSpec, s).page_size_padded is not None
        for s in specs.values()
    )


def test_auto_block_size_shrinks_swa_kernel_page():
    """Auto 256 (2048 < max_model_len <= 8192): SWA 128 / global 256, pages unpinned."""
    mgr = _make_manager(max_model_len=4096)
    specs = _kv_cache_spec(mgr, _layers())
    assert all(cast(AttentionSpec, s).block_size == 128 for s in specs.values())
    assert all(cast(AttentionSpec, s).page_size_padded is None for s in specs.values())
    assert mgr.runner._hybrid_uniform_page_size_bytes is None

    groups, block_sizes, page = _planned_block_sizes(mgr, specs)
    assert block_sizes == {"SlidingWindowSpec": 128, "FullAttentionSpec": 256}
    assert page == _page_bytes(256, GLOBAL)
    assert can_share_attention_cache([g.kv_cache_spec for g in groups], BACKEND)

    # The shrunk groups stay compatible with prefix-cache hashing.
    vllm_cfg = mgr.runner.vllm_config
    vllm_cfg.cache_config.enable_prefix_caching = True
    config = get_kv_cache_config_from_groups(vllm_cfg, groups, 8 * page * 2)
    assert resolve_kv_cache_block_sizes(config, vllm_cfg) == (256, 128)


def test_user_specified_block_size_scales_up_from_that_block():
    """Explicit `--block-size 256` is the kernel page: SWA 256 / global 512."""
    mgr = _make_manager(user_specified_block_size=True)
    specs = _kv_cache_spec(mgr, _layers())
    _, block_sizes, page = _planned_block_sizes(mgr, specs)
    assert block_sizes == {"SlidingWindowSpec": 256, "FullAttentionSpec": 512}
    assert page == _page_bytes(256, SWA)


def test_tp8_fp8_scales_up_swa_when_global_page_is_larger():
    """A larger full-attention page is never shrunk; SWA scales up 256 -> 512."""
    swa = {**SWA, "num_kv_heads": 2}
    full = {**GLOBAL, "num_kv_heads": 1}
    assert _page_bytes(256, full, torch.uint8) == 2 * _page_bytes(256, swa, torch.uint8)
    mgr = _make_manager(kv_cache_dtype=torch.uint8)
    specs = _kv_cache_spec(mgr, _layers(swa, full))
    _, block_sizes, _ = _planned_block_sizes(mgr, specs)
    assert block_sizes == {"SlidingWindowSpec": 512, "FullAttentionSpec": 256}


def test_smem_min_page_size_guard_keeps_the_larger_kernel_page():
    """A context too long for 128-token kernel pages in SMEM scales up instead."""
    mgr = _make_manager(max_model_len=100608, max_num_reqs=256)
    specs = _kv_cache_spec(mgr, _layers())
    _, block_sizes, _ = _planned_block_sizes(mgr, specs)
    assert block_sizes == {"SlidingWindowSpec": 256, "FullAttentionSpec": 512}


def test_sliding_window_not_a_multiple_of_the_kernel_page_still_shrinks():
    mgr = _make_manager()
    specs = _kv_cache_spec(mgr, _layers({**SWA, "sliding_window": 320}))
    _, block_sizes, _ = _planned_block_sizes(mgr, specs)
    assert block_sizes == {"SlidingWindowSpec": 128, "FullAttentionSpec": 256}


def test_get_kv_cache_spec_is_idempotent():
    mgr = _make_manager()
    layers = _layers()
    assert _kv_cache_spec(mgr, layers) == _kv_cache_spec(mgr, layers)


def test_non_tpu7x_chip_keeps_padded_layout_and_logs_why(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    get_chip_version.cache_clear()
    mgr = _make_manager()
    with patch("vllm_torchtpu.kv_cache_spec_normalizer.logger.info_once") as info_once:
        specs = _kv_cache_spec(mgr, _layers())
    assert _is_padded_baseline(specs)
    assert mgr.runner._hybrid_uniform_page_size_bytes == _page_bytes(256, SWA)
    reasons = [
        c.args for c in info_once.call_args_list if "is off because" in c.args[0]
    ]
    assert len(reasons) == 1
    assert "CUSTOM backend" in reasons[0][1]


@pytest.mark.parametrize(
    "case",
    [
        "default_pallas_backend",
        "speculative_config",
        "kv_transfer_config",
        "block_major_kv",
        "pcp",
        "dcp",
        "hnd_layout",
        "non_multiple_pages",
        "equal_pages",
        "same_type_full_attention",
        "same_type_sliding_window",
        "mla_layer",
    ],
)
def test_guards_keep_specs_pinned(case: str, monkeypatch: pytest.MonkeyPatch):
    """Unsupported backends, layouts, parallel modes or uniform-type specs skip."""
    backend: Any = BACKEND
    layers: dict[str, Any] = _layers()
    mgr_kwargs: dict[str, Any] = {}
    if case == "default_pallas_backend":
        backend = PallasAttentionBackend
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
    elif case == "non_multiple_pages":
        layers = _layers(
            {**SWA, "num_kv_heads": 6}, {"num_kv_heads": 4, "head_size": 256}
        )
    elif case == "equal_pages":
        layers = _layers(SWA, {"num_kv_heads": 4, "head_size": 512})
    elif case == "same_type_full_attention":
        layers = _layers({"num_kv_heads": 8, "head_size": 256}, GLOBAL)
    elif case == "same_type_sliding_window":
        layers = _layers(SWA, {**GLOBAL, "sliding_window": 1024})
    elif case == "mla_layer":
        layers["mla.0"] = MagicMock(spec=MLAAttention)

    mgr = _make_manager(**mgr_kwargs)
    assert (
        resolve_subpage_packing_block_size(
            cast(Any, layers),
            mgr.runner.vllm_config,
            mgr.runner.kv_cache_dtype,
            backend,
            256,
        )
        is None
    )
    if case != "mla_layer":
        specs = _kv_cache_spec(mgr, layers, backend=backend)
        assert mgr.runner._subpage_kernel_block_size is None
        assert all(
            cast(AttentionSpec, s).block_size == 256
            and cast(AttentionSpec, s).page_size_padded is not None
            for s in specs.values()
        )
        groups = get_kv_cache_groups(mgr.runner.vllm_config, specs)
        assert all(g.kv_cache_spec.block_size == 256 for g in groups)


def test_resolve_runs_backend_helpers_under_the_given_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """An ambient config with another KV layout must not leak into the gate."""
    mgr = _make_manager(max_model_len=4096)
    given = mgr.runner.vllm_config
    seen: list[VllmConfig] = []
    page_size_bytes = BACKEND.get_kv_cache_page_size_bytes

    def _spy(*args: Any, **kwargs: Any) -> int:
        seen.append(get_current_vllm_config())
        return page_size_bytes(*args, **kwargs)

    monkeypatch.setattr(BACKEND, "get_kv_cache_page_size_bytes", staticmethod(_spy))
    with set_current_vllm_config(_build_vllm_cfg(kv_layout=VllmKVCacheLayout.LBHNC)):
        assert (
            resolve_subpage_packing_block_size(
                cast(Any, _layers()), given, mgr.runner.kv_cache_dtype, BACKEND, 256
            )
            == 128
        )
    assert seen
    assert all(config is given for config in seen)


def test_unified_kv_layout_skips_packing():
    mgr = _make_manager(unified_kv_layout=True)
    specs = _kv_cache_spec(mgr, _layers())
    assert all(cast(AttentionSpec, s).block_size == 256 for s in specs.values())


def test_shared_cache_materializer_rejects_non_multiple_block_sizes():
    page_bytes = BACKEND.get_kv_cache_page_size_bytes(384, 2, 512, torch.bfloat16)
    bad_specs: list[AttentionSpec] = [
        SlidingWindowSpec(
            block_size=256,
            num_kv_heads=6,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
            page_size_padded=page_bytes,
        ),
        FullAttentionSpec(
            block_size=384,
            num_kv_heads=2,
            head_size=512,
            dtype=torch.bfloat16,
            page_size_padded=page_bytes,
        ),
    ]
    assert not can_share_attention_cache(bad_specs, BACKEND)
    with pytest.raises(ValueError, match="Incompatible shared KV cache spec"):
        materialize_shared_attention_cache(
            layer_specs=[("l0", bad_specs[0]), ("l1", bad_specs[1])],
            tensor_size=4 * page_bytes,
            num_blocks=4,
            kernel_block_size=256,
            attn_backend=BACKEND,
            device=torch.device("cpu"),
        )


def test_shared_pool_uses_the_kernel_block_size_not_the_region_minimum():
    """A region with no smallest-block layer still gets kernel-block-size pages."""
    page_bytes = _page_bytes(512, GLOBAL)
    region_specs = [
        FullAttentionSpec(
            block_size=512, dtype=torch.bfloat16, page_size_padded=page_bytes, **GLOBAL
        ),
        FullAttentionSpec(
            block_size=512,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            page_size_padded=page_bytes,
        ),
    ]
    assert can_share_attention_cache(region_specs, BACKEND, 256)
    pool = materialize_shared_attention_cache(
        layer_specs=[("l0", region_specs[0]), ("l1", region_specs[1])],
        tensor_size=4 * page_bytes,
        num_blocks=4,
        kernel_block_size=256,
        attn_backend=BACKEND,
        device=torch.device("cpu"),
    )
    assert pool.shape[:2] == (4, 256)
    assert pool.numel() * pool.element_size() == 4 * page_bytes


def _call_rpa_impl(
    kv_cache: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    block_tables: jnp.ndarray,
    *,
    subpage_split: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    num_q = key.shape[0]
    return attn_module._pallas_rpa_kernel_impl(
        kv_cache,
        jnp.zeros((num_q, 4, key.shape[2]), dtype=key.dtype),
        key,
        value,
        jnp.array([num_q], dtype=jnp.int32),
        block_tables,
        jnp.array([0, num_q], dtype=jnp.int32),
        jnp.array([0, 0, 1], dtype=jnp.int32),
        None,
        None,
        None,
        None,
        mesh=cast(Any, None),
        sliding_window=None,
        skip_kv_update=False,
        rpa_func=None,
        subpage_split=subpage_split,
    )


@pytest.mark.parametrize(
    "head_size, subpage_split, scale",
    [
        # Pool page holds 2 native pages of this layer.
        (512, 2, 1),  # KernelBlockTable ids already count both sub-pages.
        (512, 1, 2),  # Manager block ids: scale by the whole factor.
        # Pool page holds 4 native pages; ids count 2 of them per pool page.
        (256, 2, 2),
    ],
)
def test_rpa_kernel_impl_scales_block_tables_by_remaining_factor(
    head_size: int, subpage_split: int, scale: int, monkeypatch: pytest.MonkeyPatch
):
    captured: list[Any] = []

    def _fake_attention(kv_cache, query, key, value, metadata, *args, **kwargs):
        captured.append((tuple(kv_cache.shape), metadata.block_tables))
        return kv_cache, query

    monkeypatch.setattr(attn_module, "attention", _fake_attention)
    pool_4d = jnp.zeros((4, 256, 2, 2048), dtype=jnp.bfloat16)
    kv = jnp.zeros((2, 2, head_size), dtype=jnp.bfloat16)
    block_tables = jnp.array([[1, 3]], dtype=jnp.int32)

    new_pool, _ = _call_rpa_impl(
        pool_4d, kv, kv, block_tables, subpage_split=subpage_split
    )
    assert new_pool.shape == pool_4d.shape
    kernel_shape, kernel_tables = captured[-1]
    assert kernel_shape == (4 * 1024 // head_size, 256, 2, 2, head_size)
    assert jnp.array_equal(kernel_tables, block_tables * scale)


def test_rpa_kernel_impl_rejects_split_that_does_not_divide_the_pool_page():
    pool_4d = jnp.zeros((4, 256, 2, 2048), dtype=jnp.bfloat16)
    kv = jnp.zeros((2, 2, 512), dtype=jnp.bfloat16)
    with pytest.raises(AssertionError, match="not a multiple of subpage_split"):
        _call_rpa_impl(
            pool_4d, kv, kv, jnp.array([[0]], dtype=jnp.int32), subpage_split=4
        )


@pytest.mark.parametrize("subpage", [0, 1])
def test_each_subpage_writes_its_own_half_of_the_pool_block(
    subpage: int, monkeypatch: pytest.MonkeyPatch
):
    """Kernel ids 4 and 5 are the two 256-token halves of manager block 2."""
    page_bytes = _page_bytes(256, SWA)
    pool_torch = materialize_shared_attention_cache(
        layer_specs=[
            (
                "swa",
                SlidingWindowSpec(
                    block_size=256,
                    dtype=torch.bfloat16,
                    page_size_padded=page_bytes,
                    **SWA,
                ),
            ),
            (
                "full",
                FullAttentionSpec(
                    block_size=512,
                    dtype=torch.bfloat16,
                    page_size_padded=page_bytes,
                    **GLOBAL,
                ),
            ),
        ],
        tensor_size=4 * page_bytes,
        num_blocks=4,
        kernel_block_size=256,
        attn_backend=BACKEND,
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

    def _write_pages(kv_cache, query, key, value, metadata, *args, **kwargs):
        ids = metadata.block_tables[0]
        paged = jnp.stack([key, value], axis=2).reshape(
            ids.shape[0], kv_cache.shape[1], 2, 2, 512
        )
        return kv_cache.at[ids].set(paged), query

    monkeypatch.setattr(attn_module, "attention", _write_pages)
    ones = jnp.ones((256, 2, 512), dtype=jnp.bfloat16)
    written, _ = _call_rpa_impl(
        jnp.zeros(tuple(pool_torch.shape), dtype=jnp.bfloat16),
        ones,
        ones,
        jnp.asarray(kernel_row[None, subpage : subpage + 1], dtype=jnp.int32),
        subpage_split=2,
    )
    for block in (0, 1, 3):
        assert not bool(jnp.any(written[block] != 0))
    halves = (written[2, :128], written[2, 128:])
    assert bool(jnp.all(halves[subpage] == 1))
    assert not bool(jnp.any(halves[1 - subpage] != 0))


def test_default_pallas_backend_impl_rejects_subpage_split():
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
        pytest.raises(NotImplementedError, match="does not support sub-page packing"),
    ):
        impl._build_rpa_kernel(None, None, None)


def _rpa_impl(layer: dict[str, int], **kwargs: Any) -> Any:
    return PallasBatchedRPAAttentionBackendImpl(
        num_heads=layer["num_kv_heads"],
        head_size=layer["head_size"],
        scale=layer["head_size"] ** -0.5,
        num_kv_heads=layer["num_kv_heads"],
        alibi_slopes=None,
        sliding_window=layer.get("sliding_window"),
        kv_cache_dtype="auto",
        **kwargs,
    )


def test_batched_rpa_impl_builds_a_distinct_op_per_subpage_split(
    monkeypatch: pytest.MonkeyPatch,
):
    """Changing the split rebuilds the op instead of reusing the unsplit one."""
    built: list[tuple[str, Any]] = []

    def _fake_jax_op(op_name: str, fn: Any, **kwargs: Any) -> MagicMock:
        built.append((op_name, fn))
        return MagicMock()

    monkeypatch.setattr(attn_module.pallas, "jax_op", _fake_jax_op)
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    impl = _rpa_impl(GLOBAL)
    with set_vllm_model_wrapper_context(mesh=cast(Any, None)):
        unsplit = impl._build_rpa_kernel(None, None, None)
        assert impl._build_rpa_kernel(None, None, None) is unsplit
        impl.set_subpage_split(2)
        split = impl._build_rpa_kernel(None, None, None)
        assert impl._build_rpa_kernel(None, None, None) is split
        # Layers with the same config share an op only at the same split.
        assert _rpa_impl(GLOBAL)._build_rpa_kernel(None, None, None) is unsplit
        other = _rpa_impl(GLOBAL)
        other.set_subpage_split(2)
        assert other._build_rpa_kernel(None, None, None) is split

    assert split is not unsplit
    (unsplit_name, unsplit_fn), (split_name, split_fn) = built
    assert unsplit_name != split_name
    assert unsplit_fn.func is split_fn.func is attn_module._pallas_rpa_kernel_batched
    assert "subpage_split" not in unsplit_fn.keywords
    assert split_fn.keywords["subpage_split"] == 2


def _initialize_kv_cache(
    mgr: KVCacheManager,
    kv_cache_config: KVCacheConfig,
    layers: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_num_blocks: list[int],
    shared_kv_cache_layers: dict[str, str] | None = None,
) -> SimpleNamespace:
    monkeypatch.setattr(
        "vllm_torchtpu.runner.kv_cache_manager.synchronize_tensors",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group", lambda: False
    )
    multi_bt = MultiGroupBlockTable(
        max_num_reqs=8,
        max_num_batched_tokens=512,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[
            g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
        ],
        max_num_blocks=max_num_blocks,
    )
    kernel_block_sizes: list[list[int]] = []
    kv_caches: dict[str, torch.Tensor] = {}

    def _may_reinit(cfg: KVCacheConfig, kbs: list[int]) -> None:
        kernel_block_sizes.append(list(kbs))
        install_kernel_block_tables(multi_bt, kbs)

    def _capture_bind(caches, forward_ctx, runner_kv_caches) -> None:
        kv_caches.update(caches)
        runner_kv_caches.extend(caches.values())

    monkeypatch.setattr("vllm_torchtpu.utils.tpu_bind_kv_cache", _capture_bind)
    runner = mgr.runner
    runner.vllm_config.compilation_config.static_forward_context = layers
    if shared_kv_cache_layers is not None:
        runner.shared_kv_cache_layers = shared_kv_cache_layers
    runner.input_batch = SimpleNamespace(block_table=multi_bt)
    runner.may_reinitialize_input_batch = _may_reinit
    with patch.object(TpuPlatform, "_find_non_ssm_backend", return_value=BACKEND):
        mgr.initialize_kv_cache(kv_cache_config)
    return SimpleNamespace(
        runner=runner,
        multi_bt=multi_bt,
        kernel_block_sizes=kernel_block_sizes,
        kv_caches=kv_caches,
    )


@pytest.mark.parametrize(
    "user_specified_block_size, kernel_block_size",
    [(False, 128), (True, 256)],
    ids=["shrink_128_256", "scale_up_256_512"],
)
def test_initialize_kv_cache_from_vllm_planned_config(
    user_specified_block_size: bool,
    kernel_block_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Specs -> vLLM groups and config -> one pool per region with split tables."""
    mgr = _make_manager(user_specified_block_size=user_specified_block_size)
    layers = {
        name: attention_layer_mock(
            BACKEND, impl=_rpa_impl(SWA if name.startswith("swa") else GLOBAL), **attrs
        )
        for name, attrs in (
            [(f"swa.{i}", SWA) for i in range(6)]
            + [(f"full.{i}", GLOBAL) for i in range(2)]
        )
    }
    specs = _kv_cache_spec(mgr, layers)
    assert mgr.runner._subpage_kernel_block_size == kernel_block_size
    vllm_cfg = mgr.runner.vllm_config
    groups, _, page = _planned_block_sizes(mgr, specs)
    num_blocks = 4
    config = get_kv_cache_config_from_groups(vllm_cfg, groups, num_blocks * 2 * page)
    assert config.num_blocks == num_blocks

    env = _initialize_kv_cache(
        mgr,
        config,
        layers,
        monkeypatch,
        max_num_blocks=[4096 // g.kv_cache_spec.block_size for g in groups],
    )
    assert env.kernel_block_sizes == [[kernel_block_size] * len(groups)]
    for group, table in zip(groups, env.multi_bt.block_tables):
        is_full = isinstance(group.kv_cache_spec, FullAttentionSpec)
        assert isinstance(table, KernelBlockTable) is is_full
        if is_full:
            assert table.blocks_per_kv_block == 2

    # Two regions, each holding three SWA layers and one full-attention layer.
    unique = {id(cache): cache for cache in env.kv_caches.values()}
    assert len(unique) == 2
    assert env.kv_caches["swa.0"] is env.kv_caches["full.0"]
    for cache in unique.values():
        assert cache.ndim == 4
        assert cache.shape[:2] == (num_blocks, kernel_block_size)
        assert cache.nbytes == num_blocks * page
    for name, layer in layers.items():
        assert layer.impl._subpage_split == (2 if name.startswith("full") else 1)
    assert env.runner._attention_kernel_block_size == kernel_block_size


def test_initialize_kv_cache_unpaired_layer_and_kv_sharing_alias(
    monkeypatch: pytest.MonkeyPatch,
):
    """A lone layer in a region gets a native 5D cache; KV-sharing layers alias."""
    swa_bytes = _page_bytes(256, SWA)
    mgr = _make_manager(max_model_len=8192, user_specified_block_size=True)
    layers = {
        "layer.0": attention_layer_mock(BACKEND, impl=_rpa_impl(SWA), **SWA),
        "layer.1": attention_layer_mock(BACKEND, impl=_rpa_impl(GLOBAL), **GLOBAL),
        "layer.2": attention_layer_mock(BACKEND, impl=_rpa_impl(GLOBAL), **GLOBAL),
        "layer.3": attention_layer_mock(
            BACKEND,
            impl=_rpa_impl(GLOBAL, kv_sharing_target_layer_name="layer.1"),
            kv_sharing_target_layer_name="layer.1",
            **GLOBAL,
        ),
    }
    _kv_cache_spec(mgr, layers)
    assert mgr.runner._subpage_kernel_block_size == 256
    num_blocks = 6
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
                layer_names=["layer.0"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=256,
                    dtype=torch.bfloat16,
                    page_size_padded=swa_bytes,
                    **SWA,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["layer.1", "layer.2"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=512,
                    dtype=torch.bfloat16,
                    page_size_padded=swa_bytes,
                    **GLOBAL,
                ),
            ),
        ],
    )
    env = _initialize_kv_cache(
        mgr,
        kv_cache_config,
        layers,
        monkeypatch,
        max_num_blocks=[5, 16],
        shared_kv_cache_layers={"layer.3": "layer.1"},
    )
    assert env.kernel_block_sizes == [[256, 256]]
    assert not isinstance(env.multi_bt.block_tables[0], KernelBlockTable)
    assert env.multi_bt.block_tables[1].blocks_per_kv_block == 2

    assert env.kv_caches["layer.0"] is env.kv_caches["layer.1"]
    assert env.kv_caches["layer.3"] is env.kv_caches["layer.1"]
    assert tuple(env.kv_caches["layer.1"].shape) == (num_blocks, 256, 2, 2048)
    assert tuple(env.kv_caches["layer.2"].shape) == (num_blocks * 2, 256, 2, 2, 512)
    assert [layers[f"layer.{i}"].impl._subpage_split for i in range(4)] == [1, 2, 2, 2]
    # Group 1's split table is the widest one the kernel reads.
    assert env.runner._attention_kernel_block_size == 256
    assert env.runner._attention_kv_cache_group_id == 1


def _ref_causal_gqa_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    q_offset: int,
    scale: float,
    sliding_window: int | None,
) -> np.ndarray:
    """Float32 causal (+ optional sliding-window) GQA reference."""
    q_len, num_q_heads, head_dim = query.shape
    kv_len, num_kv_heads, _ = key.shape
    groups = num_q_heads // num_kv_heads
    q = query.astype(np.float32).reshape(q_len, num_kv_heads, groups, head_dim)
    k = key.astype(np.float32)
    v = value.astype(np.float32)
    logits = np.einsum("qkhd,tkd->khqt", q, k) * scale
    q_pos = (q_offset + np.arange(q_len))[:, None]
    kv_pos = np.arange(kv_len)[None, :]
    mask = kv_pos <= q_pos
    if sliding_window is not None:
        mask = mask & ((q_pos - kv_pos) < sliding_window)
    logits = np.where(mask[None, None, :, :], logits, -np.inf)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = np.einsum("khqt,tkd->qkhd", probs, v)
    return out.reshape(q_len, num_q_heads, head_dim)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kernel_block_size, scaled_block_size",
    [(128, 256), (256, 512)],
    ids=["shrink_128_256", "scale_up_256_512"],
)
def test_batched_rpa_on_shared_packed_pool_matches_separate_caches_and_ref(
    kernel_block_size: int,
    scaled_block_size: int,
):
    """Run `_pallas_rpa_kernel_batched` on a shared 4D pool across prefill + decode."""
    tpu_devices = jax.devices("tpu")
    mesh = Mesh(np.asarray(tpu_devices[:1]), ("model",))
    kv_layout = batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE

    sliding_window = kernel_block_size + 64
    swa_cfg = {**SWA, "sliding_window": sliding_window}
    page_bytes = _page_bytes(kernel_block_size, swa_cfg)
    num_blocks = 8
    pool_torch = materialize_shared_attention_cache(
        layer_specs=[
            (
                "swa",
                SlidingWindowSpec(
                    block_size=kernel_block_size,
                    dtype=torch.bfloat16,
                    page_size_padded=page_bytes,
                    **swa_cfg,
                ),
            ),
            (
                "full",
                FullAttentionSpec(
                    block_size=scaled_block_size,
                    dtype=torch.bfloat16,
                    page_size_padded=page_bytes,
                    **GLOBAL,
                ),
            ),
        ],
        tensor_size=num_blocks * page_bytes,
        num_blocks=num_blocks,
        kernel_block_size=kernel_block_size,
        attn_backend=BACKEND,
        device=torch.device("cpu"),
    )
    shared_pool = jnp.zeros(tuple(pool_torch.shape), dtype=jnp.bfloat16)

    tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=scaled_block_size * 2,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[kernel_block_size, scaled_block_size],
        max_num_blocks=[4, 2],
    )
    install_kernel_block_tables(tables, [kernel_block_size, kernel_block_size])
    # Disjoint pool blocks: SWA {1, 4, 6}; full {2, 5} -> kernel {4, 5, 10, 11}.
    tables.add_row(([1, 4, 6], [2, 5]), 0)
    swa_bt = jnp.asarray(tables.block_tables[0].get_numpy_array()[0, :4], jnp.int32)
    full_bt = jnp.asarray(tables.block_tables[1].get_numpy_array()[0, :4], jnp.int32)
    assert full_bt.tolist() == [4, 5, 10, 11]

    sep_swa = jnp.zeros(
        BACKEND.get_kv_cache_shape(
            num_blocks, kernel_block_size, swa_cfg["num_kv_heads"], 256, torch.bfloat16
        ),
        dtype=jnp.bfloat16,
    )
    sep_full = jnp.zeros(
        BACKEND.get_kv_cache_shape(
            num_blocks * 2,
            kernel_block_size,
            GLOBAL["num_kv_heads"],
            512,
            torch.bfloat16,
        ),
        dtype=jnp.bfloat16,
    )

    prefill_len = scaled_block_size + 32
    total_len = prefill_len + 1
    rng = np.random.default_rng(kernel_block_size)

    def _rand_inputs(layer: dict[str, int]):
        kv_h = layer["num_kv_heads"]
        hd = layer["head_size"]
        q = (rng.standard_normal((total_len, kv_h * 2, hd)) * 0.5).astype(np.float32)
        k = (rng.standard_normal((total_len, kv_h, hd)) * 0.5).astype(np.float32)
        v = (rng.standard_normal((total_len, kv_h, hd)) * 0.5).astype(np.float32)
        return (
            jnp.asarray(q, jnp.bfloat16),
            jnp.asarray(k, jnp.bfloat16),
            jnp.asarray(v, jnp.bfloat16),
        )

    swa_qkv = _rand_inputs(swa_cfg)
    full_qkv = _rand_inputs(GLOBAL)

    def _step(cache, qkv, bt, *, q_slice: slice, seq_len: int, decode: bool, sw, split):
        q, k, v = (arr[q_slice] for arr in qkv)
        q_len = q.shape[0]
        dist = jnp.asarray((1, 1, 1) if decode else (0, 0, 1), jnp.int32)
        return attn_module._pallas_rpa_kernel_batched(
            cache,
            q,
            k,
            v,
            jnp.asarray([seq_len], jnp.int32),
            bt,
            jnp.asarray([0, q_len], jnp.int32),
            dist,
            None,
            None,
            None,
            None,
            mesh=mesh,
            sliding_window=sw,
            sm_scale=q.shape[-1] ** -0.5,
            skip_kv_update=False,
            kv_layout=kv_layout,
            subpage_split=split,
        )

    # Interleave SWA and full writes into `shared_pool` across prefill + decode so
    # cross-layer clobbering in the shared 4D buffer would corrupt later reads.
    shared_pool, swa_pre_shared = _step(
        shared_pool,
        swa_qkv,
        swa_bt,
        q_slice=slice(0, prefill_len),
        seq_len=prefill_len,
        decode=False,
        sw=sliding_window,
        split=1,
    )
    shared_pool, full_pre_shared = _step(
        shared_pool,
        full_qkv,
        full_bt,
        q_slice=slice(0, prefill_len),
        seq_len=prefill_len,
        decode=False,
        sw=None,
        split=2,
    )
    shared_pool, swa_dec_shared = _step(
        shared_pool,
        swa_qkv,
        swa_bt,
        q_slice=slice(prefill_len, total_len),
        seq_len=total_len,
        decode=True,
        sw=sliding_window,
        split=1,
    )
    shared_pool, full_dec_shared = _step(
        shared_pool,
        full_qkv,
        full_bt,
        q_slice=slice(prefill_len, total_len),
        seq_len=total_len,
        decode=True,
        sw=None,
        split=2,
    )

    sep_swa, swa_pre_sep = _step(
        sep_swa,
        swa_qkv,
        swa_bt,
        q_slice=slice(0, prefill_len),
        seq_len=prefill_len,
        decode=False,
        sw=sliding_window,
        split=1,
    )
    sep_swa, swa_dec_sep = _step(
        sep_swa,
        swa_qkv,
        swa_bt,
        q_slice=slice(prefill_len, total_len),
        seq_len=total_len,
        decode=True,
        sw=sliding_window,
        split=1,
    )
    sep_full, full_pre_sep = _step(
        sep_full,
        full_qkv,
        full_bt,
        q_slice=slice(0, prefill_len),
        seq_len=prefill_len,
        decode=False,
        sw=None,
        split=1,
    )
    sep_full, full_dec_sep = _step(
        sep_full,
        full_qkv,
        full_bt,
        q_slice=slice(prefill_len, total_len),
        seq_len=total_len,
        decode=True,
        sw=None,
        split=1,
    )

    for shared_out, sep_out, qkv, q_offset, kv_len, sw in (
        (swa_pre_shared, swa_pre_sep, swa_qkv, 0, prefill_len, sliding_window),
        (swa_dec_shared, swa_dec_sep, swa_qkv, prefill_len, total_len, sliding_window),
        (full_pre_shared, full_pre_sep, full_qkv, 0, prefill_len, None),
        (full_dec_shared, full_dec_sep, full_qkv, prefill_len, total_len, None),
    ):
        shared_f32 = np.asarray(shared_out.astype(jnp.float32))
        sep_f32 = np.asarray(sep_out.astype(jnp.float32))
        assert np.abs(shared_f32).max() > 0
        np.testing.assert_array_equal(shared_f32, sep_f32)

        q, k, v = (np.asarray(arr.astype(jnp.float32)) for arr in qkv)
        q_slice = q[q_offset:kv_len]
        ref = _ref_causal_gqa_attention(
            q_slice,
            k[:kv_len],
            v[:kv_len],
            q_offset=q_offset,
            scale=q.shape[-1] ** -0.5,
            sliding_window=sw,
        )
        np.testing.assert_allclose(shared_f32, ref, atol=2e-2, rtol=0)

    packed_5d = shared_pool.reshape(
        num_blocks, kernel_block_size, 1, 2, shared_pool.shape[3]
    )
    shared_swa_pages = attn_module._reshape_packed_kv_cache(packed_5d, sep_swa.shape)
    shared_full_pages = attn_module._reshape_packed_kv_cache(packed_5d, sep_full.shape)
    swa_ids = jnp.asarray([1, 4, 6], jnp.int32)
    full_ids = jnp.asarray([4, 5, 10, 11], jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(shared_swa_pages[swa_ids].astype(jnp.float32)),
        np.asarray(sep_swa[swa_ids].astype(jnp.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(shared_full_pages[full_ids].astype(jnp.float32)),
        np.asarray(sep_full[full_ids].astype(jnp.float32)),
    )
    assert not bool(jnp.any(shared_pool[jnp.asarray([0, 3, 7], jnp.int32)] != 0))


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "kv_layout, pool_mult",
    [
        (batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE, 1),
        (batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE, 2),
        (batched_rpa_configs.KVLayout.SEQ_ALONG_LANE, 2),
    ],
    ids=["nhd_1x", "nhd_2x", "hnd_2x"],
)
def test_batched_rpa_subspace_on_shared_packed_pool_matches_separate_5d(
    kv_layout: batched_rpa_configs.KVLayout,
    pool_mult: int,
):
    """Run `_pallas_rpa_kernel_batched` with Subspace-RPA on a 4D shared pool.

    Covers equal (1x) and unequal (2x) pool pages in HEAD_ALONG_SUBLANE and
    unequal (2x) pool pages in SEQ_ALONG_LANE, checking bit-exact parity
    against a separate 5D cache across prefill and decode.
    """
    from vllm_torchtpu.kernels.experimental.batched_rpa import (
        wrapper as rpa_batched_wrapper,
    )

    tpu_devices = jax.devices("tpu")
    mesh = Mesh(np.asarray(tpu_devices[:1]), ("model",))
    head_dim = 512
    rot_dim = 128
    num_kv_heads = 1
    num_q_heads = 4
    page_size = 128
    num_pool_blocks = 6
    fp8 = jnp.float8_e4m3fn

    native_5d = rpa_batched_wrapper.get_kv_cache_shape(
        total_num_pages=num_pool_blocks * pool_mult,
        page_size=page_size,
        actual_num_kv_heads=num_kv_heads,
        actual_head_dim=head_dim,
        kv_dtype=fp8,
        kv_layout=kv_layout,
        chip_version=get_chip_version(),
        subspace_k_rot_dim=rot_dim,
    )
    packing = 4
    if kv_layout == batched_rpa_configs.KVLayout.SEQ_ALONG_LANE:
        words_per_token = native_5d[1] * native_5d[2] * pool_mult
        pool_shape = (num_pool_blocks, words_per_token, packing, page_size)
    else:
        words_per_token = native_5d[2] * native_5d[4] * pool_mult
        pool_shape = (num_pool_blocks, page_size, packing, words_per_token)

    shared_pool = jax.lax.bitcast_convert_type(
        jnp.zeros(pool_shape, dtype=jnp.uint8), fp8
    )
    sep_cache = jax.lax.bitcast_convert_type(jnp.zeros(native_5d, dtype=jnp.uint8), fp8)

    # Shuffled pool block IDs; with pool_mult=2 and subpage_split=1, the
    # adapter scales pool block IDs by table_scale=2.
    pool_bt = jnp.asarray([4, 1, 5], dtype=jnp.int32)
    sep_bt = pool_bt * pool_mult

    prefill_len = page_size + 24
    total_len = prefill_len + 1
    rng = np.random.default_rng(100 + pool_mult)
    q_all = jnp.asarray(
        rng.standard_normal((total_len, num_q_heads, head_dim)) * 0.5, jnp.bfloat16
    )
    k_f32 = rng.standard_normal((total_len, num_kv_heads, rot_dim)) * 0.5
    v_f32 = rng.standard_normal((total_len, num_kv_heads, head_dim)) * 0.5
    k_scale, v_scale = 0.75, 1.25
    finfo = jnp.finfo(fp8)
    v_all = jnp.clip(
        jnp.asarray(v_f32, jnp.float32) / v_scale, float(finfo.min), float(finfo.max)
    ).astype(fp8)
    if kv_layout == batched_rpa_configs.KVLayout.SEQ_ALONG_LANE:
        k_all = jnp.clip(
            jnp.asarray(k_f32, jnp.float32) / k_scale,
            float(finfo.min),
            float(finfo.max),
        ).astype(fp8)
    else:
        k_all = jnp.asarray(k_f32, jnp.bfloat16)

    def _run(cache, bt, sl: slice, seq_len: int, decode: bool):
        q, k, v = q_all[sl], k_all[sl], v_all[sl]
        dist = jnp.asarray((1, 1, 1) if decode else (0, 0, 1), jnp.int32)
        return attn_module._pallas_rpa_kernel_batched(
            cache,
            q,
            k,
            v,
            jnp.asarray([seq_len], jnp.int32),
            bt,
            jnp.asarray([0, q.shape[0]], jnp.int32),
            dist,
            None,
            None,
            k_scale,
            v_scale,
            mesh=mesh,
            sliding_window=None,
            sm_scale=head_dim**-0.5,
            skip_kv_update=False,
            kv_layout=kv_layout,
            subpage_split=1,
        )

    shared_pool, pre_shared = _run(
        shared_pool, pool_bt, slice(0, prefill_len), prefill_len, False
    )
    shared_pool, dec_shared = _run(
        shared_pool, pool_bt, slice(prefill_len, total_len), total_len, True
    )
    sep_cache, pre_sep = _run(
        sep_cache, sep_bt, slice(0, prefill_len), prefill_len, False
    )
    sep_cache, dec_sep = _run(
        sep_cache, sep_bt, slice(prefill_len, total_len), total_len, True
    )

    for got, want in ((pre_shared, pre_sep), (dec_shared, dec_sep)):
        got_f32 = np.asarray(got.astype(jnp.float32))
        want_f32 = np.asarray(want.astype(jnp.float32))
        assert np.abs(got_f32).max() > 0
        np.testing.assert_array_equal(got_f32, want_f32)

    # Unused pool blocks [0, 2, 3] remain untouched zeros.
    unused = jax.lax.bitcast_convert_type(shared_pool, jnp.uint8)[
        jnp.asarray([0, 2, 3], jnp.int32)
    ]
    assert not bool(jnp.any(unused != 0))
