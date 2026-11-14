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
"""Unit and TPU tests for TPU_RPA_PREFILL_MAX_BKV in batched RPA."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    stitch_utils,
    wrapper,
)


def _mock_tpu_info(
    *,
    generation: int = 7,
    chip_version: str = "v7x",
    mxu_column_size: int = 256,
) -> MagicMock:
    mock = MagicMock()
    mock.generation = generation
    mock.chip_version = chip_version
    mock.num_lanes = 128
    mock.num_sublanes = 8
    mock.mxu_column_size = mxu_column_size
    mock.vmem_capacity_bytes = 64 * 1024 * 1024
    mock.smem_capacity_bytes = 1024 * 1024
    mock.fp8_ops_per_second = 2
    mock.bf16_ops_per_second = 1
    return mock


def _clear_envs_cache() -> None:
    cache_clear = getattr(envs.__getattr__, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def test_env_var_defaults_and_ray_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    _clear_envs_cache()
    monkeypatch.delenv("TPU_RPA_PREFILL_MAX_BKV", raising=False)
    assert envs.TPU_RPA_PREFILL_MAX_BKV == 0
    monkeypatch.setenv("TPU_RPA_PREFILL_MAX_BKV", " 1024 ")
    _clear_envs_cache()
    assert envs.TPU_RPA_PREFILL_MAX_BKV == 1024
    monkeypatch.setenv("TPU_RPA_PREFILL_MAX_BKV", "-512")
    _clear_envs_cache()
    with pytest.raises(ValueError, match="TPU_RPA_PREFILL_MAX_BKV"):
        _ = envs.TPU_RPA_PREFILL_MAX_BKV
    _clear_envs_cache()
    assert "TPU_RPA_PREFILL_MAX_BKV" in TpuPlatform.additional_env_vars


def test_calculate_block_sizes_high_gqa_and_coupled_low_gqa() -> None:
    """Verify:
    1. Flag off preserves exact baseline bq_c_sz=64 on 16:1 GQA (32Q/2KV, hd=256).
    2. Flag on (512/768/1024) grows bkv_sz while keeping bq_c_sz=16 (256 MXU rows)
       and bounding VMEM at large caps (4096 -> <= 1536 on 64 MiB VMEM).
    3. Coupled Step 2 on low-GQA layers (e.g., DCP NEW_TOKENS_ONLY 2Q/1KV) is never
       shrunk below its coupled default when prefill_max_bkv_sz=512.
    4. Chips with mxu_column_size=128 and page_size=256 align bkv_sz to 256.
    """
    high_gqa_model = configs.ModelConfigs(
        num_q_heads=32, num_kv_heads=2, head_dim=256, mask_value=-1e9
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=40,
        page_size=256,
        total_q_tokens=384,
        num_page_indices=40 * 36,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info", return_value=_mock_tpu_info()
    ):
        _, default_prefill = wrapper.calculate_block_sizes(
            high_gqa_model, serve_cfg, vmem_limit
        )
        assert default_prefill == configs.BlockSizes(
            bq_sz=256, bq_c_sz=64, bkv_sz=256, batch_size=1
        )

        _, prefill_512 = wrapper.calculate_block_sizes(
            high_gqa_model, serve_cfg, vmem_limit, prefill_max_bkv_sz=512
        )
        assert prefill_512 == configs.BlockSizes(
            bq_sz=256, bq_c_sz=16, bkv_sz=512, batch_size=1
        )

        _, prefill_768 = wrapper.calculate_block_sizes(
            high_gqa_model, serve_cfg, vmem_limit, prefill_max_bkv_sz=768
        )
        assert prefill_768 == configs.BlockSizes(
            bq_sz=256, bq_c_sz=16, bkv_sz=768, batch_size=1
        )

        _, prefill_4096 = wrapper.calculate_block_sizes(
            high_gqa_model, serve_cfg, vmem_limit, prefill_max_bkv_sz=4096
        )
        assert prefill_4096.bq_sz == 256
        assert prefill_4096.bq_c_sz == 16
        assert 1024 <= prefill_4096.bkv_sz <= 1536

        low_gqa_model = configs.ModelConfigs(
            num_q_heads=2, num_kv_heads=1, head_dim=256, mask_value=-1e9
        )
        _, low_default = wrapper.calculate_block_sizes(
            low_gqa_model, serve_cfg, vmem_limit
        )
        _, low_with_cap = wrapper.calculate_block_sizes(
            low_gqa_model, serve_cfg, vmem_limit, prefill_max_bkv_sz=512
        )
        assert low_with_cap == low_default
        assert low_with_cap.bkv_sz > 512

    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_mock_tpu_info(
            generation=6, chip_version="v6e", mxu_column_size=128
        ),
    ):
        _, v6e_prefill = wrapper.calculate_block_sizes(
            high_gqa_model, serve_cfg, vmem_limit, prefill_max_bkv_sz=768
        )
        assert v6e_prefill.bkv_sz % serve_cfg.page_size == 0

        bad_cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(bq_sz=128, bq_c_sz=16, bkv_sz=384, batch_size=1),
            model=high_gqa_model,
            serve=serve_cfg,
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=vmem_limit,
        )
        with pytest.raises(NotImplementedError, match="aligned to page_size"):
            bad_cfgs.validate_inputs(
                q=jnp.zeros((16, 32, 256), dtype=jnp.bfloat16),
                k=jnp.zeros((16, 2, 256), dtype=jnp.float8_e4m3fn),
                v=jnp.zeros((16, 2, 256), dtype=jnp.float8_e4m3fn),
                kv_cache=jnp.zeros((8, 256, 1, 4, 256), dtype=jnp.float8_e4m3fn),
                kv_lens=jnp.zeros((40,), dtype=jnp.int32),
                page_indices=jnp.zeros((40 * 36,), dtype=jnp.int32),
                cu_q_lens=jnp.zeros((41,), dtype=jnp.int32),
                distribution=jnp.zeros((3,), dtype=jnp.int32),
            )


def test_stitch_guard_uses_prefill_path_in_mixed_mode_when_bkv_exceeds_lanes() -> None:
    model_cfg = configs.ModelConfigs(
        num_q_heads=32, num_kv_heads=2, head_dim=128, mask_value=-1e9
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=4,
        page_size=128,
        total_q_tokens=512,
        num_page_indices=64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info", return_value=_mock_tpu_info()
    ):
        mixed_cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(bq_sz=128, bq_c_sz=16, bkv_sz=512, batch_size=1),
            model=model_cfg,
            serve=serve_cfg,
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=64 * 1024 * 1024,
        )
        decode_cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=512, batch_size=8),
            model=model_cfg,
            serve=serve_cfg,
            mode=configs.RpaCase.DECODE,
            vmem_limit_bytes=64 * 1024 * 1024,
        )
        assert stitch_utils._use_decode_lane_stitch(mixed_cfgs) is False
        assert stitch_utils._use_decode_lane_stitch(decode_cfgs) is True


def test_adapter_and_dcp_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_torchtpu.layers.adapter import attention, cp_attention

    captured_jax_ops: list[tuple[str, object]] = []
    monkeypatch.setattr(attention, "_kernel_registry", {}, raising=False)
    monkeypatch.setattr(attention.PallasAttentionBackendImpl, "_kernel_registry", {})

    def fake_jax_op(name, fn, **kwargs):
        captured_jax_ops.append((name, fn))
        op = MagicMock()
        op.register_fake = MagicMock()
        return op

    monkeypatch.setattr(attention.pallas, "jax_op", fake_jax_op)
    fake_vllm_cfg = MagicMock()
    fake_vllm_cfg.cache_config.get_resolved_kv_cache_layout.return_value = next(
        iter(attention.KV_LAYOUT_BY_VLLM_LAYOUT)
    )
    fake_vllm_cfg.speculative_config = None
    monkeypatch.setattr(attention, "get_current_vllm_config", lambda: fake_vllm_cfg)
    monkeypatch.setattr(
        attention, "get_current_vllm_config_or_none", lambda: fake_vllm_cfg
    )
    monkeypatch.setattr(attention, "unified_block_major_enabled", lambda _: False)
    fake_ctx = MagicMock()
    fake_ctx.vllm_config = fake_vllm_cfg
    fake_ctx.mesh = MagicMock()
    monkeypatch.setattr(attention, "get_vllm_model_wrapper_context", lambda: fake_ctx)

    _clear_envs_cache()
    monkeypatch.setenv("TPU_RPA_PREFILL_MAX_BKV", "512")
    impl_256 = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=32,
        head_size=256,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    assert impl_256.prefill_max_bkv_sz == 512
    impl_256._build_rpa_kernel(None, 1.0, 1.0)
    assert captured_jax_ops[-1][1].keywords.get("prefill_max_bkv_sz") == 512

    impl_hd64 = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=32,
        head_size=64,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    assert impl_hd64.prefill_max_bkv_sz == 0

    impl_base = attention.PallasAttentionBackendImpl(
        num_heads=32,
        head_size=256,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    assert impl_base.prefill_max_bkv_sz == 0

    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(
        cp_attention,
        "_build_dcp_kernel_op",
        lambda name, fn: (name, fn.keywords["prefill_max_bkv_sz"]),
    )
    c0, _ = cp_attention.build_dcp_kernels(
        None, 1.0, None, None, None, None, cp_group_size=2, cp_rank=0
    )
    c1, _ = cp_attention.build_dcp_kernels(
        None,
        1.0,
        None,
        None,
        None,
        None,
        cp_group_size=2,
        cp_rank=0,
        prefill_max_bkv_sz=768,
    )
    assert c0[1] is None
    assert c1[1] == 768
    assert c0[0] != c1[0]
    _clear_envs_cache()


@pytest.mark.tpu_test
@pytest.mark.parametrize("prefill_max_bkv_sz", [512, 768])
@pytest.mark.parametrize(
    "kv_layout,page_size",
    [
        (configs.KVLayout.HEAD_ALONG_SUBLANE, 256),
        (configs.KVLayout.SEQ_ALONG_LANE, 128),
    ],
)
@pytest.mark.parametrize("sliding_window", [None, 384])
def test_tpu_prefill_max_bkv_matches_default_tiles(
    prefill_max_bkv_sz: int,
    kv_layout: configs.KVLayout,
    page_size: int,
    sliding_window: int | None,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    key = jax.random.PRNGKey(7)
    k_q, k_k, k_v, k_cache = jax.random.split(key, 4)

    num_q_heads, num_kv_heads, head_dim = 32, 2, 256
    num_seqs = 8
    pages_per_seq = 48 if page_size == 128 else 24
    num_pages = num_seqs * pages_per_seq

    # Exercise decode, pure prefill (cache_len=0), unaligned intra-page stitch
    # (cache_len=129, 900), and multi-KV-tile writeback (q_len=1100 > bkv_sz=768).
    q_lens = np.array([1, 1, 512, 345, 1100, 0, 0, 0], dtype=np.int32)
    cache_lens = np.array([1200, 2500, 0, 129, 900, 0, 0, 0], dtype=np.int32)
    max_tokens = 2048

    q = (
        jax.random.normal(k_q, (max_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
        * 0.2
    )
    k = (
        jax.random.normal(k_k, (max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        * 0.2
    ).astype(jnp.float8_e4m3fn)
    v = (
        jax.random.normal(k_v, (max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        * 0.2
    ).astype(jnp.float8_e4m3fn)

    cache_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.float8_e4m3fn,
        kv_layout=kv_layout,
        chip_version=pltpu.get_tpu_info().chip_version,
    )
    kv_cache_init = (
        jax.random.normal(k_cache, cache_shape, dtype=jnp.bfloat16) * 0.2
    ).astype(jnp.float8_e4m3fn)

    kv_lens = jnp.asarray(cache_lens + q_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), dtype=jnp.int32)
    distribution = jnp.array([2, 3, 5], dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)

    valid_tokens = int(q_lens.sum())
    out_ref, cache_ref = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=head_dim**-0.5,
        sliding_window=sliding_window,
        k_scale=1.0,
        v_scale=1.0,
        kv_layout=kv_layout,
        prefill_max_bkv_sz=None,
    )
    out_cand, cache_cand = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=head_dim**-0.5,
        sliding_window=sliding_window,
        k_scale=1.0,
        v_scale=1.0,
        kv_layout=kv_layout,
        prefill_max_bkv_sz=prefill_max_bkv_sz,
    )
    np.testing.assert_allclose(
        np.asarray(out_cand[:valid_tokens], dtype=np.float32),
        np.asarray(out_ref[:valid_tokens], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    cand_u8 = np.asarray(cache_cand.view(jnp.uint8))
    ref_u8 = np.asarray(cache_ref.view(jnp.uint8))
    if kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE:
        np.testing.assert_array_equal(cand_u8, ref_u8)
    else:
        # Under SEQ_ALONG_LANE (shape [pages, 2*kv_heads, d_pack, pack, page_size]),
        # unused tail lanes past `kv_len` inside a sequence's last page are padding;
        # every valid token lane and every untouched page must match bit-for-bit.
        mask = np.ones((num_pages, page_size), dtype=bool)
        total_kv = cache_lens + q_lens
        for seq in range(int(distribution[2])):
            kvl = int(total_kv[seq])
            rem = kvl % page_size
            if rem != 0:
                last_page = seq * pages_per_seq + (kvl // page_size)
                mask[last_page, rem:] = False
        mask_5d = mask[:, None, None, None, :]
        np.testing.assert_array_equal(
            np.where(mask_5d, cand_u8, 0),
            np.where(mask_5d, ref_u8, 0),
        )


@pytest.mark.tpu_test
def test_tpu_dcp_return_lse_with_prefill_max_bkv() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    key = jax.random.PRNGKey(19)
    k_q, k_k, k_v, k_cache = jax.random.split(key, 4)

    num_q_heads, num_kv_heads, head_dim = 16, 1, 256
    page_size = 256
    num_seqs = 4
    pages_per_seq = 16
    num_pages = num_seqs * pages_per_seq
    max_tokens = 512

    q = (
        jax.random.normal(k_q, (max_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
        * 0.2
    )
    k = (
        jax.random.normal(k_k, (max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        * 0.2
    ).astype(jnp.float8_e4m3fn)
    v = (
        jax.random.normal(k_v, (max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
        * 0.2
    ).astype(jnp.float8_e4m3fn)
    cache_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.float8_e4m3fn,
        chip_version=pltpu.get_tpu_info().chip_version,
    )
    kv_cache_init = (
        jax.random.normal(k_cache, cache_shape, dtype=jnp.bfloat16) * 0.2
    ).astype(jnp.float8_e4m3fn)

    q_lens = np.array([1, 300, 0, 0], dtype=np.int32)
    cache_lens = np.array([1024, 1536, 0, 0], dtype=np.int32)
    kv_lens = jnp.asarray(cache_lens + q_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), dtype=jnp.int32)
    distribution = jnp.array([1, 1, 2], dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cp_rank = jnp.array([0], dtype=jnp.int32)
    valid_tokens = int(q_lens.sum())

    out_ref, _, lse_ref = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=head_dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
        cp_group_size=2,
        cp_rank=cp_rank,
        attention_scope=configs.AttentionScope.CACHE_ONLY,
        return_lse=True,
        prefill_max_bkv_sz=None,
    )
    out_cand, _, lse_cand = wrapper.ragged_paged_attention(
        q,
        k,
        v,
        kv_cache_init.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=head_dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
        cp_group_size=2,
        cp_rank=cp_rank,
        attention_scope=configs.AttentionScope.CACHE_ONLY,
        return_lse=True,
        prefill_max_bkv_sz=512,
    )
    np.testing.assert_allclose(
        np.asarray(out_cand[:valid_tokens], dtype=np.float32),
        np.asarray(out_ref[:valid_tokens], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(lse_cand[:valid_tokens], dtype=np.float32),
        np.asarray(lse_ref[:valid_tokens], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
