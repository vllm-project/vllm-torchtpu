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
    # `vllm.envs` memoizes `VLLM_KV_CACHE_LAYOUT` on first read; disable its
    # cache temporarily so `MonkeyPatch` overrides reach `resolve_kv_cache_layout`.
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

            scheduler_cfg = SchedulerConfig(
                max_num_seqs=max_num_reqs,
                max_model_len=max_model_len,
                is_encoder_decoder=False,
            )
            parallel_cfg = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            vllm_cfg = VllmConfig(
                device_config=DeviceConfig(device="cpu"),
                cache_config=cache_cfg,
                scheduler_config=scheduler_cfg,
                parallel_config=parallel_cfg,
                attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
            )
            resolve_kv_cache_layout(vllm_cfg, [["LBNHC", "LBHNC"]])
    finally:
        if was_cached:
            vllm_envs.enable_envs_cache()

    # Apply TPU-specific parallel / connector fields after CPU VllmConfig init so
    # TPU device-count and out-of-tree connector registry checks do not block CPU tests.
    vllm_cfg.parallel_config.prefill_context_parallel_size = pcp_size
    vllm_cfg.parallel_config.decode_context_parallel_size = dcp_size
    vllm_cfg.kv_transfer_config = kv_transfer_config
    vllm_cfg.model_config = cast(
        Any,
        SimpleNamespace(
            max_model_len=max_model_len,
            get_head_size=lambda: 256,
        ),
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


def test_tpu7x_shrinks_oversized_swa_to_match_global_page():
    """TPU_7X TP=2 regime: SWA (1.0 MiB at bs=256) > Global (0.5 MiB at bs=256)
    shrinks SWA block_size 256 -> 128 when block_size is auto-selected and SMEM
    request caps hold.
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


def test_shrunk_swa_groups_are_compatible_with_prefix_caching_hash_size():
    """With prefix caching on, the shrunk [SWA 128, Global 256] groups resolve to
    hash_block_size = gcd(128, 256) = 128 and scheduler_block_size = 256, so
    every group's block size is a multiple of the hash block size (the hybrid
    coordinator's divisibility requirement).
    """
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

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
    """Explicit `--block-size 256` (`user_specified_block_size=True`) must never
    be silently shrunk to kernel_block_size=128; it preserves kernel_block_size=256
    via the [256, 512] scale-up path instead.
    """
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


def test_non_tpu7x_chip_skips_subpage_packing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-TPU_7X chips (e.g. v6e) keep the padded baseline layout: neither the
    kernel page shrink nor the scale-up regime runs.
    """
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
    """At max_model_len=100608 and max_num_seqs=256, `get_tpu_min_page_size`
    returns 256 (since 128-token pages would require 786 pages/req > 512 SMEM cap).
    Shrink is rejected and scale-up [256, 512] preserves max_num_seqs=256.
    """
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    mgr = _make_fake_manager(swa_bytes, max_model_len=100608, max_num_reqs=256)

    packed = mgr._pack_attention_subpages(specs, PallasBatchedRPAAttentionBackend, 256)
    assert packed["layer.0.attn"].block_size == 256
    assert packed["layer.5.attn"].block_size == 512
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes


def test_non_multiple_sliding_window_still_allows_kernel_page_shrink():
    """`BatchedRPA` computes sliding-window bounds per token and vLLM's
    `SlidingWindowManager` supports arbitrary window/block ratios, so a
    `sliding_window` that is not a multiple of `new_bs` (e.g. 320 with 128)
    still shrinks cleanly on TPU_7X.
    """
    specs, swa_bytes, global_bytes = _gemma4_tp2_specs()
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
    assert packed["layer.0.attn"].block_size == 128
    assert packed["layer.5.attn"].block_size == 256
    assert mgr.runner._hybrid_uniform_page_size_bytes == global_bytes


def test_get_kv_cache_spec_idempotent_across_repeated_calls():
    """Repeated `get_kv_cache_spec()` calls recompute `_hybrid_uniform_page_size_bytes`
    before `_pack_attention_subpages`, producing identical specs and page budgets.
    """
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
    """Every unsupported backend, layout, parallelism, connector, or non-multiple
    page-size ratio returns the original specs untouched and logs the skip reason.
    """
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    backend_cls: Any = PallasBatchedRPAAttentionBackend
    mgr_kwargs: dict[str, Any] = {}

    if case == "default_pallas_backend":
        backend_cls = PallasAttentionBackend
    elif case == "uniform_pages":
        specs = {"layer.0.attn": specs["layer.0.attn"]}
        mgr = _make_fake_manager(0)
        with patch("vllm_torchtpu.runner.kv_cache_manager.logger.debug") as dbg:
            assert mgr._pack_attention_subpages(specs, backend_cls, 256) == specs
        assert any(
            "KV sub-page packing skipped:" in (c.args[0] % c.args[1:])
            for c in dbg.call_args_list
        )
        return
    elif case == "equal_native_bytes_multi_layer":
        specs = {
            "layer.0.attn": specs["layer.0.attn"],
            "layer.1.attn": SlidingWindowSpec(
                block_size=256,
                num_kv_heads=4,
                head_size=512,
                dtype=torch.bfloat16,
                sliding_window=1024,
                page_size_padded=swa_bytes,
            ),
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
    with patch("vllm_torchtpu.runner.kv_cache_manager.logger.debug") as dbg:
        packed = mgr._pack_attention_subpages(specs, backend_cls, 256)
    assert packed == specs
    assert mgr.runner._hybrid_uniform_page_size_bytes == swa_bytes
    assert any(
        "KV sub-page packing skipped:" in (c.args[0] % c.args[1:])
        for c in dbg.call_args_list
    )


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


def test_write_then_read_4d_pool_subpage_byte_equivalence(
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end byte-offset proof across `materialize_shared_attention_cache`,
    `KernelBlockTable(split=2)`, and `_pallas_rpa_kernel_impl`:
    writing 512 distinct token K/V vectors into manager block `b = 2` populates
    both 256-token sub-pages `[4, 5]` inside `pool_4d[2]` without touching
    neighboring manager blocks `{0, 1, 3}`, and reading back via `KernelBlockTable`
    recovers all 512 token K/V vectors bit-for-bit.
    """
    specs, swa_bytes, _ = _gemma4_tp2_specs()
    specs["layer.5.attn"] = FullAttentionSpec(
        block_size=512,
        num_kv_heads=2,
        head_size=512,
        dtype=torch.bfloat16,
        page_size_padded=swa_bytes,
    )
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
    # Assign manager block b=2 to FullAttention (group 1).
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
        # kv_cache: (num_subpages=8, page_size=256, 2*H_kv//packing=2, packing=2, D=512)
        page_size = kv_cache.shape[1]
        subpage_ids = metadata.block_tables[0]
        if kwargs.get("skip_kv_update", False):
            gathered_pages = kv_cache[subpage_ids]  # (2, 256, 2, 2, 512)
            flat_tokens = gathered_pages.reshape(num_tokens, 2, 2, 512)
            # Interleaved K/V head packing: axis 2 is head index, axis 3 is (K, V)
            read_back_kv.append((flat_tokens[:, :, 0, :], flat_tokens[:, :, 1, :]))
            return kv_cache, query

        packed_tokens = jnp.stack([key, value], axis=2)  # (512, 2, 2, 512)
        paged_tokens = packed_tokens.reshape(subpage_ids.shape[0], page_size, 2, 2, 512)
        updated = kv_cache.at[subpage_ids].set(paged_tokens)
        return updated, query

    monkeypatch.setattr(attn_module, "attention", _paged_write_or_read)

    pool_4d = jnp.zeros(tuple(pool_torch.shape), dtype=jnp.bfloat16)
    kernel_bt = jnp.asarray(kernel_row[None, :], dtype=jnp.int32)
    seq_lens = jnp.array([num_tokens], dtype=jnp.int32)
    query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    written_pool_4d, _ = attn_module._pallas_rpa_kernel_impl(
        pool_4d,
        query_512,
        key_512,
        val_512,
        seq_lens,
        kernel_bt,
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

    assert written_pool_4d.shape == (4, 256, 2, 2048)
    # Manager blocks 0, 1, 3 are untouched; manager block 2 holds all 512 tokens.
    assert not bool(jnp.any(written_pool_4d[0] != 0))
    assert not bool(jnp.any(written_pool_4d[1] != 0))
    assert not bool(jnp.any(written_pool_4d[3] != 0))
    assert bool(jnp.any(written_pool_4d[2, :, :, :1024] != 0))
    assert bool(jnp.any(written_pool_4d[2, :, :, 1024:] != 0))

    # Read back through a fresh 4D -> 5D reshape using the same KernelBlockTable row.
    attn_module._pallas_rpa_kernel_impl(
        written_pool_4d,
        query_512,
        key_512,
        val_512,
        seq_lens,
        kernel_bt,
        query_start_loc,
        request_distribution,
        None,
        None,
        None,
        None,
        mesh=cast(Any, None),
        sliding_window=None,
        skip_kv_update=True,
        rpa_func=None,
        subpage_split=2,
    )
    assert len(read_back_kv) == 1
    recovered_k, recovered_v = read_back_kv[0]
    assert jnp.array_equal(recovered_k, key_512)
    assert jnp.array_equal(recovered_v, val_512)


def test_default_pallas_backend_impl_rejects_subpage_split_gt_1():
    """`PallasAttentionBackendImpl` (`_pallas_rpa_kernel_default`) does not accept
    `subpage_split`; attempting to build a custom op with `subpage_split > 1`
    raises `NotImplementedError` instead of silently mis-indexing KV pages.
    """
    impl = PallasAttentionBackendImpl(
        num_heads=4,
        head_size=256,
        scale=1.0 / (256**0.5),
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
    """Runner glue test across `KVCacheManager.initialize_kv_cache`:
    verifies `kernel_block_sizes = [256, 256]` from `[SWA(256), Full(512)]`,
    `KernelBlockTable(split=2)` installation on Group 1, 4D shared pool on
    multi-layer Region 0 vs 5D `(num_blocks * 2, 256, ...)` fallback on
    single-layer Region 1, and `set_subpage_split(2)` propagation across both
    owning layers and `shared_kv_cache_layers` aliases.
    """
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
    swa_spec = SlidingWindowSpec(
        block_size=256,
        num_kv_heads=8,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=1024,
        page_size_padded=swa_bytes,
    )
    full_spec = FullAttentionSpec(
        block_size=512,
        num_kv_heads=2,
        head_size=512,
        dtype=torch.bfloat16,
        page_size_padded=swa_bytes,
    )
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
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=swa_spec),
            KVCacheGroupSpec(
                layer_names=["layer.1", "layer.2"], kv_cache_spec=full_spec
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
    """When Group 0 is a narrow `SlidingWindowSpec` (`max_num_blocks_per_req=5`)
    and Group 1 is a sub-page-split `FullAttentionSpec` (`32` manager blocks ->
    `64` kernel blocks via `KernelBlockTable(split=2)`), `TPUModelRunner` must
    size `max_num_blocks_per_req` and `block_table_cpu` to `64`, not Group 0's `5`.
    """
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
            KVCacheGroupSpec(
                layer_names=["layer.0"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=256,
                    num_kv_heads=8,
                    head_size=256,
                    dtype=torch.bfloat16,
                    sliding_window=1024,
                    page_size_padded=swa_bytes,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["layer.1"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=512,
                    num_kv_heads=2,
                    head_size=512,
                    dtype=torch.bfloat16,
                    page_size_padded=swa_bytes,
                ),
            ),
        ],
    )

    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vllm_config = vllm_cfg
    runner.max_num_reqs = max_num_reqs
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
