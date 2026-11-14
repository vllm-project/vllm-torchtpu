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
"""Tests for batched_rpa_longctx schedule, live-slot zeroing, and wrapper."""

import contextlib
from unittest import mock

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    kernel,
                                                                    wrapper)


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """Provide TPU7x chip info when tracing/lowering on a CPU host."""

    def tpu7x():
        return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)

    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(tpu_info.registry, {"cpu": tpu7x}):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


@pytest.mark.cpu_test
def test_decode_batch_size_is_static_and_unknown_kwargs_raise():
    """`decode_batch_size` traces as a static argument under `jax.jit` and
    unknown keyword arguments raise `TypeError`."""
    sds = jax.ShapeDtypeStruct
    num_seqs, pages_per_seq, num_tokens = 8, 4, 8
    page_size, num_q_heads, num_kv_heads, head_dim = 128, 4, 1, 128
    dtype = jnp.bfloat16

    with _tpu_info_on_cpu():
        kv_shape = wrapper.get_kv_cache_shape(
            num_seqs * pages_per_seq + 1,
            page_size,
            num_kv_heads,
            head_dim,
            dtype,
            configs.KVLayout.HEAD_ALONG_SUBLANE,
        )
        seen_decode_batch_sizes = []

        def _fake_generate_rpa_metadata(*args, cfgs, **kwargs):
            if cfgs.mode == configs.RpaCase.DECODE:
                seen_decode_batch_sizes.append(cfgs.block.batch_size)
            return jnp.zeros((1, ), dtype=jnp.int32)

        def _fake_rpa_kernel(cu_q_lens, q_offsets, kv_cache_lens, kv_new_lens,
                             page_indices, schedule_hbm, o_hbm_alias_q_hbm,
                             new_kv_hbm, kv_cache, lse_hbm_in, *, cfgs,
                             computer_cls):
            return o_hbm_alias_q_hbm, kv_cache, lse_hbm_in

        with mock.patch.object(
                wrapper.schedule,
                "generate_rpa_metadata",
                side_effect=_fake_generate_rpa_metadata,
        ), mock.patch.object(
                wrapper.kernel,
                "rpa_kernel",
                side_effect=_fake_rpa_kernel,
        ):
            for batch_size in (None, 1, 4):
                traced = wrapper.ragged_paged_attention.trace(
                    sds((num_tokens, num_q_heads, head_dim), dtype),
                    sds((num_tokens, num_kv_heads, head_dim), dtype),
                    sds((num_tokens, num_kv_heads, head_dim), dtype),
                    sds(kv_shape, dtype),
                    sds((num_seqs, ), jnp.int32),
                    sds((num_seqs * pages_per_seq, ), jnp.int32),
                    sds((num_seqs + 1, ), jnp.int32),
                    sds((3, ), jnp.int32),
                    sm_scale=head_dim**-0.5,
                    decode_batch_size=batch_size,
                )
                assert traced is not None
        assert len(seen_decode_batch_sizes) == 3
        assert seen_decode_batch_sizes[1] == 1
        assert seen_decode_batch_sizes[2] > 1

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            wrapper.ragged_paged_attention.trace(
                sds((num_tokens, num_q_heads, head_dim), dtype),
                sds((num_tokens, num_kv_heads, head_dim), dtype),
                sds((num_tokens, num_kv_heads, head_dim), dtype),
                sds(kv_shape, dtype),
                sds((num_seqs, ), jnp.int32),
                sds((num_seqs * pages_per_seq, ), jnp.int32),
                sds((num_seqs + 1, ), jnp.int32),
                sds((3, ), jnp.int32),
                non_existent_option=123,
            )


@pytest.mark.cpu_test
@pytest.mark.parametrize("actual_steps", [0, 1, 2, 3, 5])
def test_zero_live_window_slots_with_nan_prefill(actual_steps):
    """Verify `_zero_live_window_slots` zeroes `[0, min(actual_steps, n_buffer))`
    and that live slots contain no NaNs even when the VMEM window starts full
    of NaNs (`actual_steps < n_buffer`)."""
    n_buffer = 3
    rows_per_slot = 512
    total_rows = n_buffer * rows_per_slot
    num_lanes = 128
    nan_u32 = jnp.uint32(0x7FC00000)

    def _init_kernel(steps_ref, buf_ref, out_ref):
        out_ref[...] = buf_ref[...]
        actual = steps_ref[0]

        @pl.when(actual > 0)
        def _():
            num_live = jnp.minimum(actual, n_buffer)
            kernel._zero_live_window_slots(out_ref, n_buffer, num_live)

    steps = jnp.array([actual_steps], dtype=jnp.int32)
    nan_buf = jnp.full((total_rows, num_lanes), nan_u32, dtype=jnp.uint32)
    out_u32 = pl.pallas_call(
        _init_kernel,
        out_shape=jax.ShapeDtypeStruct(nan_buf.shape, jnp.uint32),
        interpret=True,
    )(steps, nan_buf)

    out_f32 = np.asarray(out_u32).view(np.float32).reshape(
        n_buffer, rows_per_slot, num_lanes)
    live_slots = min(actual_steps, n_buffer)
    if live_slots > 0:
        assert np.all(out_f32[:live_slots] == 0.0)
        assert not np.isnan(out_f32[:live_slots]).any()
    if live_slots < n_buffer:
        assert np.isnan(out_f32[live_slots:]).all()


@pytest.mark.cpu_test
def test_partial_page_and_unlive_nan_slots_do_not_pollute_attention():
    """When VMEM window slots start filled with NaNs and
    `actual_steps < n_buffer` with partial last pages, zeroing only the live
    slots prevents NaN pollution in causal `p @ v` attention and matches the
    full-zero reference."""
    n_buffer = 3
    actual_steps = 2
    rows_per_slot = 512
    num_lanes = 128
    nan_u32 = jnp.uint32(0x7FC00000)

    def _init_live_only(steps_ref, buf_ref, out_ref):
        out_ref[...] = buf_ref[...]
        num_live = jnp.minimum(steps_ref[0], n_buffer)
        kernel._zero_live_window_slots(out_ref, n_buffer, num_live)

    nan_buf = jnp.full((n_buffer * rows_per_slot, num_lanes),
                       nan_u32,
                       dtype=jnp.uint32)
    live_zeroed_u32 = pl.pallas_call(
        _init_live_only,
        out_shape=jax.ShapeDtypeStruct(nan_buf.shape, jnp.uint32),
        interpret=True,
    )(jnp.array([actual_steps], dtype=jnp.int32), nan_buf)

    slots_f32 = (np.asarray(live_zeroed_u32).view(np.float32).reshape(
        n_buffer, rows_per_slot, num_lanes).copy())

    rng = np.random.default_rng(42)
    valid_kv_len = 70
    bkv_sz = 128
    # Simulate partial last page DMA into live slot 0: only the first
    # `valid_kv_len` tokens are written; tail rows in slot 0 remain 0.0 from
    # `_zero_live_window_slots`, while slot 2 (`>= actual_steps`) holds NaN.
    assert np.isnan(slots_f32[2]).all()
    q = rng.standard_normal((1, 8, num_lanes)).astype(np.float32) * 0.5
    k_valid = rng.standard_normal(
        (1, valid_kv_len, num_lanes)).astype(np.float32) * 0.5
    v_valid = rng.standard_normal(
        (1, valid_kv_len, num_lanes)).astype(np.float32) * 0.5

    k_slot0 = slots_f32[0, :bkv_sz].copy()[None, :, :]
    v_slot0 = slots_f32[0, bkv_sz:2 * bkv_sz].copy()[None, :, :]
    k_slot0[:, :valid_kv_len, :] = k_valid
    v_slot0[:, :valid_kv_len, :] = v_valid

    mask = (np.arange(bkv_sz)[None, None, :] < valid_kv_len)
    logits = np.einsum("hqd,hkd->hqk", q, k_slot0) * (num_lanes**-0.5)
    logits = np.where(mask, logits, -1e30)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = np.where(mask, probs, 0.0)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out_live_zero = np.einsum("hqk,hkd->hqd", probs, v_slot0)

    logits_ref = np.einsum("hqd,hkd->hqk", q, k_valid) * (num_lanes**-0.5)
    probs_ref = np.exp(logits_ref - np.max(logits_ref, axis=-1, keepdims=True))
    probs_ref = probs_ref / np.sum(probs_ref, axis=-1, keepdims=True)
    out_ref = np.einsum("hqk,hkd->hqd", probs_ref, v_valid)

    assert np.isfinite(out_live_zero).all()
    np.testing.assert_allclose(out_live_zero, out_ref, rtol=1e-5, atol=1e-5)


def test_tpu_rpa_kernel_nan_prefilled_kv_cache_and_unlive_slots():
    """On TPU, run `ragged_paged_attention` with `actual_steps < n_buffer`,
    partial last pages, and NaN-filled unused KV-cache pages/lanes."""
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend for end-to-end Pallas execution")

    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    num_q_heads, num_kv_heads, head_dim = 4, 2, 128
    page_size, total_pages = 128, 8
    kv_len, q_len = 70, 1
    sm_scale = head_dim**-0.5
    decode_blocks = configs.BlockSizes(bq_sz=1,
                                       bq_c_sz=1,
                                       bkv_sz=128,
                                       batch_size=2,
                                       n_buffer=3)

    q = jnp.asarray(
        rng.standard_normal((q_len, num_q_heads, head_dim)) * 0.5, dtype)
    k_all = rng.standard_normal(
        (kv_len, num_kv_heads, head_dim)).astype(np.float32) * 0.5
    v_all = rng.standard_normal(
        (kv_len, num_kv_heads, head_dim)).astype(np.float32) * 0.5
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    _, clean_cache = wrapper.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), dtype),
        jnp.asarray(k_all, dtype),
        jnp.asarray(v_all, dtype),
        jnp.zeros(cache_shape, dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=decode_blocks,
    )
    nan_cache = clean_cache.at[1:].set(jnp.nan)

    out_clean, _ = wrapper.ragged_paged_attention(
        q,
        jnp.asarray(k_all[-1:], dtype),
        jnp.asarray(v_all[-1:], dtype),
        clean_cache,
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, 1], jnp.int32),
        jnp.array([1, 1, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=decode_blocks,
    )
    out_nan, _ = wrapper.ragged_paged_attention(
        q,
        jnp.asarray(k_all[-1:], dtype),
        jnp.asarray(v_all[-1:], dtype),
        nan_cache,
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, 1], jnp.int32),
        jnp.array([1, 1, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=decode_blocks,
    )
    assert jnp.isfinite(out_nan).all()
    np.testing.assert_allclose(np.asarray(out_nan, np.float32),
                               np.asarray(out_clean, np.float32),
                               atol=1e-2,
                               rtol=0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize("decode_batch_size", [1, 4])
def test_batched_rpa_longctx_traces_without_mocks(kv_layout,
                                                  decode_batch_size):
    """End-to-end trace and TPU lowering of `batched_rpa_longctx` without mocks,
    exercising `pallas_compat` (`BufferedRef`, `when=`, `strict_ordering=`) on
    the pinned JAX build."""
    sds = jax.ShapeDtypeStruct
    num_seqs, pages_per_seq, num_tokens = 4, 4, 8
    page_size, num_q_heads, num_kv_heads, head_dim = 128, 4, 1, 128
    dtype = jnp.bfloat16

    with _tpu_info_on_cpu():
        kv_shape = wrapper.get_kv_cache_shape(
            num_seqs * pages_per_seq + 1,
            page_size,
            num_kv_heads,
            head_dim,
            dtype,
            kv_layout,
        )
        traced = wrapper.ragged_paged_attention.trace(
            sds((num_tokens, num_q_heads, head_dim), dtype),
            sds((num_tokens, num_kv_heads, head_dim), dtype),
            sds((num_tokens, num_kv_heads, head_dim), dtype),
            sds(kv_shape, dtype),
            sds((num_seqs, ), jnp.int32),
            sds((num_seqs * pages_per_seq, ), jnp.int32),
            sds((num_seqs + 1, ), jnp.int32),
            sds((3, ), jnp.int32),
            sm_scale=head_dim**-0.5,
            kv_layout=kv_layout,
            decode_batch_size=decode_batch_size,
        )
        assert traced is not None
        lowered = traced.lower(lowering_platforms=("tpu", ))
        hlo = lowered.as_text()
        assert "tpu.enqueue_dma" in hlo or "tpu_custom_call" in hlo
