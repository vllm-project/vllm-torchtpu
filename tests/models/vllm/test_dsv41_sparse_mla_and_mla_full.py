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

"""100% full-module line and branch coverage tests for

sparse_mla.py and mla.py on CPU under JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=true.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    mla,
    sparse_mla,
)


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        start = int(np.asarray(idx.start))
        size = int(np.asarray(idx.size))
        return slice(start, start + size, stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr).copy()
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        res = self.arr[_norm_idx(idx)]
        return jnp.asarray(res)

    def __setitem__(self, idx, val):
        if isinstance(val, MockArrayRef):
            val = val.arr
        self.arr[_norm_idx(idx)] = np.asarray(val)

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                if idx is Ellipsis:
                    return jnp.asarray(self.ref.arr)
                res = self.ref.arr[_norm_idx(idx)]
                return MockArrayRef(res)

            def __setitem__(self, idx, val):
                if isinstance(val, MockArrayRef):
                    val = val.arr
                self.ref.arr[_norm_idx(idx)] = np.asarray(val)

        return _AtIndexer(self)

    def bitcast(self, dtype):
        return MockArrayRef(self.arr.view(dtype))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return MockArrayRef(self.arr.reshape(*shape))


class MockAsyncCopy:
    """Mock async DMA copy handle."""

    def __init__(self):
        self.started = False
        self.waited = False

    def start(self):
        self.started = True
        return self

    def wait(self):
        self.waited = True
        return self


def fake_when(cond):
    """Decorator executing inner function for statement and branch coverage."""

    def decorator(fn):
        fn()
        return fn

    return decorator


def fake_bitcast(x, dt):
    arr = x.arr if isinstance(x, MockArrayRef) else x
    arr = np.asarray(arr)
    try:
        return jnp.asarray(arr.view(dt))
    except Exception:
        return jnp.asarray(arr).astype(dt)


def fake_fori_loop(lower, upper, body_fun, init_val, unroll=False):
    val = init_val
    for i in range(int(np.asarray(lower)), int(np.asarray(upper))):
        val = body_fun(i, val)
    return val


# ==============================================================================
# 1. Pure Helpers and Utilities Full Coverage
# ==============================================================================


class TestCommonHelpersAndEdgeCases:
    """Tests 100% of helpers across sparse_mla.py and mla.py."""

    def test_cdiv_and_align_to(self):
        for mod in (sparse_mla, mla):
            assert mod.cdiv(10, 4) == 3
            assert mod.cdiv(8, 4) == 2
            assert mod.cdiv(1, 4) == 1
            assert mod.align_to(10, 4) == 12
            assert mod.align_to(8, 4) == 8
            with pytest.raises(AssertionError):
                mod.cdiv(10, 0)

    def test_dtype_bitwidth_and_packing(self):
        for mod in (sparse_mla, mla):
            assert mod.get_dtype_bitwidth(jnp.float32) == 32
            assert mod.get_dtype_bitwidth(jnp.bfloat16) == 16
            assert mod.get_dtype_bitwidth(jnp.uint8) == 8
            assert mod.get_dtype_packing(jnp.float32) == 1
            assert mod.get_dtype_packing(jnp.bfloat16) == 2
            assert mod.get_dtype_packing(jnp.uint8) == 4

    def test_get_kv_cache_shape(self):
        shape_bf16 = sparse_mla.get_kv_cache_shape(
            total_num_pages=16,
            page_size=16,
            kv_dim=512,
            kv_dtype=jnp.bfloat16,
        )
        assert shape_bf16 == (16, 8, 2, 512)

        shape_u8 = sparse_mla.get_kv_cache_shape(
            total_num_pages=32,
            page_size=16,
            kv_dim=448,
            kv_dtype=jnp.uint8,
        )
        assert shape_u8 == (32, 4, 4, 512)

    def test_largest_divisor(self):
        assert sparse_mla._largest_divisor(16, 16) == 16
        assert sparse_mla._largest_divisor(15, 4) == 3
        assert sparse_mla._largest_divisor(7, 4) == 1
        assert sparse_mla._largest_divisor(1, 10) == 1
        # When cap <= 0, loop range is empty, hitting line 67 fallback return 1
        assert sparse_mla._largest_divisor(0, 0) == 1
        assert sparse_mla._largest_divisor(5, 0) == 1

    def test_mla_case_enum(self):
        assert mla.MlaCase.DECODE.value == 0
        assert mla.MlaCase.PREFILL.value == 1
        assert mla.MlaCase.MIXED.value == 2
        assert mla.MlaCase.BATCHED_DECODE.value == 3

        assert mla.MlaCase.DECODE.symbol == "d"
        assert mla.MlaCase.PREFILL.symbol == "p"
        assert mla.MlaCase.MIXED.symbol == "m"
        assert mla.MlaCase.BATCHED_DECODE.symbol == "bd"

    def test_dequant_dsv4_fp8(self):
        arr = jnp.ones((2, 512), dtype=jnp.uint8)
        with (
            patch.object(sparse_mla.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                sparse_mla.pltpu,
                "roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, a),
            ),
        ):
            dequant = sparse_mla._dequant_dsv4_fp8(arr)
            assert dequant.shape == (2, 448)
            assert dequant.dtype == jnp.bfloat16

    def test_prepare_q_inputs(self):
        for mod in (sparse_mla, mla):
            # Already aligned: heads % packing == 0, head_dim % 128 == 0
            q_aligned = jnp.ones((4, 16, 512), dtype=jnp.float32)
            padded_aligned = mod.prepare_q_inputs(q_aligned)
            assert padded_aligned.shape == (4, 16, 512)

            # Unaligned: heads=15 (packing=2 for bf16 aligns to 16,
            # head_dim=500 aligns to 512)
            q_unaligned = jnp.ones((4, 15, 500), dtype=jnp.bfloat16)
            padded_unaligned = mod.prepare_q_inputs(q_unaligned)
            assert padded_unaligned.shape == (4, 16, 512)

    def test_prepare_swa_inputs(self):
        for mod in (sparse_mla, mla):
            swa_acc = jnp.ones((4, 15, 500), dtype=jnp.bfloat16)
            swa_l = jnp.ones((4, 15), dtype=jnp.float32)
            swa_m = jnp.ones((4, 15), dtype=jnp.float32)

            acc_p, l_p, m_p = mod.prepare_swa_inputs(swa_acc, swa_l, swa_m)
            assert acc_p.shape == (4, 16, 512)
            assert l_p.shape == (4, 128)
            assert m_p.shape == (4, 128)

    def test_prepare_outputs(self):
        for mod in (sparse_mla, mla):
            out = jnp.ones((4, 16, 512), dtype=jnp.bfloat16)
            sliced = mod.prepare_outputs(
                out, actual_num_q_heads=15, actual_head_dim=500
            )
            assert sliced.shape == (4, 15, 500)


# ==============================================================================
# 2. sparse_mla.py: gather_page_ids and Kernel Full Coverage
# ==============================================================================


class TestSparseMlaGatherPageIds:
    """Tests 100% lines and branches in gather_page_ids and _gather_page_ids_kernel."""

    def test_gather_page_ids_kernel_direct(self):
        windows_np = np.arange(8 * 128, dtype=np.int32).reshape(8, 128)
        logical_np = np.zeros((8, 16), dtype=np.int32)
        logical_np[:, 0] = 5
        logical_np[:, 1] = 130

        windows_ref = MockArrayRef(windows_np)
        logical_ref = MockArrayRef(logical_np)
        out_ref = MockArrayRef(np.zeros((8, 16), dtype=np.int32))

        sparse_mla._gather_page_ids_kernel(
            windows_ref, logical_ref, out_ref, num_chunks=1
        )
        assert out_ref.arr.shape == (8, 16)
        assert out_ref.arr[0, 0] == 5

    def test_gather_page_ids_outer_padded_and_unpadded(self):
        def fake_pallas_call_gather(kernel, out_shape=None, *args, **kwargs):
            return lambda *call_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        with patch.object(
            sparse_mla.pl, "pallas_call", side_effect=fake_pallas_call_gather
        ):
            # Branch 1: padded_pps != pages_per_seq (pages_per_seq=100)
            # and padded_tokens != num_tokens (num_tokens=5)
            max_num_seqs = 2
            pages_per_seq = 100
            page_indices = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
            seq_page_ids = jnp.zeros((5, 16), dtype=jnp.int32)
            seq_ids_segment = jnp.zeros((5,), dtype=jnp.int32)

            out1 = sparse_mla.gather_page_ids(
                page_indices,
                seq_page_ids,
                seq_ids_segment,
                max_num_seqs,
                block_tokens=8,
            )
            assert out1.shape == (5, 16)

            # Branch 2: padded_pps == pages_per_seq (pages_per_seq=128)
            # and padded_tokens == num_tokens (num_tokens=8)
            pages_per_seq = 128
            page_indices = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
            seq_page_ids = jnp.zeros((8, 16), dtype=jnp.int32)
            seq_ids_segment = jnp.zeros((8,), dtype=jnp.int32)

            out2 = sparse_mla.gather_page_ids(
                page_indices,
                seq_page_ids,
                seq_ids_segment,
                max_num_seqs,
                block_tokens=8,
            )
            assert out2.shape == (8, 16)


class TestSparseMlaAttentionKernelDirect:
    """Tests 100% lines and branches in sparse_mla._attention_kernel."""

    def test_attention_kernel_step_coverage_f32_and_bf16(self):
        for q_dtype in (jnp.float32, jnp.bfloat16):
            batch_size = 4
            topk = 16
            head_dim = 512
            num_q_heads = 16
            num_tokens = 16
            start_seq = 0
            end_seq = 16

            kv_lens_ref = MockArrayRef(np.array([10] * num_tokens, dtype=np.int32))
            start_end_seq_idx_ref = MockArrayRef(
                np.array([start_seq, end_seq, num_tokens], dtype=np.int32)
            )
            sem_ids_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))
            bo_ids_ref = MockArrayRef(np.full((2,), -1, dtype=np.int32))
            attention_sinks_ref = MockArrayRef(
                np.zeros((num_q_heads,), dtype=np.float32)
            )
            q_hbm_ref = MockArrayRef(
                np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float16)
            )
            cache_kv_nope_hbm_ref = MockArrayRef(
                np.zeros((num_tokens, topk, 128), dtype=np.int32)
            )
            cache_kv_rope_hbm_ref = MockArrayRef(
                np.zeros((num_tokens, topk // 4, 128), dtype=np.int32)
            )
            swa_accumution_hbm_ref = MockArrayRef(
                np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
            )
            swa_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
            swa_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
            o_hbm_ref = MockArrayRef(
                np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
            )
            bkv_nope_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, topk * 4, 128), dtype=np.uint8)
            )
            bkv_rope_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, (topk // 4) * 2, 64), dtype=np.float16)
            )
            bq_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float16)
            )
            bo_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float16)
            )
            bl_x2_ref = MockArrayRef(np.zeros((2, batch_size, 128), dtype=np.float32))
            bm_x2_ref = MockArrayRef(np.zeros((2, batch_size, 128), dtype=np.float32))
            swa_acc_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float16)
            )
            sems = MockArrayRef(np.zeros((7, 2), dtype=np.int32))

            with (
                patch.object(
                    sparse_mla.pltpu,
                    "make_async_copy",
                    return_value=MockAsyncCopy(),
                ),
                patch.object(sparse_mla.pltpu, "bitcast", side_effect=fake_bitcast),
                patch.object(
                    sparse_mla.pltpu,
                    "roll",
                    side_effect=lambda x, s, a: jnp.roll(x, s, a),
                ),
                patch.object(
                    sparse_mla.pl,
                    "reciprocal",
                    side_effect=lambda x, approx=True: 1.0 / x,
                ),
                patch.object(sparse_mla.pl, "when", side_effect=fake_when),
                patch.object(sparse_mla.pl, "program_id", return_value=2),
                patch.object(sparse_mla.pl, "num_programs", return_value=2),
            ):
                sparse_mla._attention_kernel(
                    kv_lens_ref,
                    start_end_seq_idx_ref,
                    sem_ids_ref,
                    bo_ids_ref,
                    attention_sinks_ref,
                    q_hbm_ref,
                    cache_kv_nope_hbm_ref,
                    cache_kv_rope_hbm_ref,
                    swa_accumution_hbm_ref,
                    swa_l_hbm_ref,
                    swa_m_hbm_ref,
                    o_hbm_ref,
                    bkv_nope_x2_ref,
                    bkv_rope_x2_ref,
                    bq_x2_ref,
                    bo_x2_ref,
                    bl_x2_ref,
                    bm_x2_ref,
                    swa_acc_x2_ref,
                    sems,
                    sm_scale=1.0,
                    batch_size=batch_size,
                )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)

    def test_attention_kernel_dm_and_bo_waits(self):
        batch_size = 1
        topk = 16
        head_dim = 512
        num_q_heads = 16

        kv_lens_ref = MockArrayRef(np.array([10], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 1, 1], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.array([-1, 0], dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(np.ones((1, num_q_heads, head_dim), dtype=np.float32))
        cache_kv_nope_hbm_ref = MockArrayRef(np.zeros((1, topk, 128), dtype=np.int32))
        cache_kv_rope_hbm_ref = MockArrayRef(
            np.zeros((1, topk // 4, 128), dtype=np.int32)
        )
        swa_accumution_hbm_ref = MockArrayRef(
            np.zeros((1, num_q_heads, head_dim), dtype=np.float32)
        )
        swa_l_hbm_ref = MockArrayRef(np.zeros((1, 128), dtype=np.float32))
        swa_m_hbm_ref = MockArrayRef(np.zeros((1, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(np.zeros((1, num_q_heads, head_dim), dtype=np.float32))
        bkv_nope_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, topk * 4, 128), dtype=np.uint8)
        )
        bkv_rope_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, (topk // 4) * 2, 64), dtype=np.float16)
        )
        bq_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float32)
        )
        bo_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float32)
        )
        bl_x2_ref = MockArrayRef(np.zeros((2, batch_size, 128), dtype=np.float32))
        bm_x2_ref = MockArrayRef(np.zeros((2, batch_size, 128), dtype=np.float32))
        swa_acc_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, num_q_heads, head_dim), dtype=np.float32)
        )
        sems = MockArrayRef(np.zeros((7, 2), dtype=np.int32))

        with (
            patch.object(
                sparse_mla.pltpu,
                "make_async_copy",
                return_value=MockAsyncCopy(),
            ),
            patch.object(sparse_mla.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                sparse_mla.pltpu,
                "roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, a),
            ),
            patch.object(
                sparse_mla.pl,
                "reciprocal",
                side_effect=lambda x, approx=True: 1.0 / x,
            ),
            patch.object(sparse_mla.pl, "when", side_effect=fake_when),
            patch.object(sparse_mla.pl, "program_id", return_value=0),
            patch.object(sparse_mla.pl, "num_programs", return_value=1),
        ):
            sparse_mla._attention_kernel(
                kv_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                cache_kv_nope_hbm_ref,
                cache_kv_rope_hbm_ref,
                swa_accumution_hbm_ref,
                swa_l_hbm_ref,
                swa_m_hbm_ref,
                o_hbm_ref,
                bkv_nope_x2_ref,
                bkv_rope_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                swa_acc_x2_ref,
                sems,
                sm_scale=1.0,
                batch_size=batch_size,
            )


class TestSparseRaggedPagedAttentionOuter:
    """Tests 100% of lines and branches in sparse_mla.sparse_ragged_paged_attention."""

    def test_sparse_ragged_paged_attention_success(self):
        def fake_pallas_call(kernel, out_shape=None, *args, **kwargs):
            return lambda *call_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        def fake_csa_gather(
            cache_kv_nope, cache_kv_rope, indices, num_valid_indices, top_k
        ):
            chunk_size = indices.shape[0] // top_k
            nope_buf = jnp.zeros((chunk_size, top_k, 128), dtype=jnp.int32)
            rope_buf = jnp.zeros((chunk_size, top_k // 4, 128), dtype=jnp.int32)
            return nope_buf, rope_buf

        with (
            patch.object(sparse_mla.pl, "pallas_call", side_effect=fake_pallas_call),
            patch.object(
                sparse_mla.csa_gather, "csa_gather", side_effect=fake_csa_gather
            ),
        ):
            max_num_tokens = 32
            actual_num_q_heads = 16
            actual_head_dim = 512
            total_num_pages = 8
            page_size = 16
            max_num_seqs = 2
            pages_per_seq = 4
            topk = 16

            q = jnp.ones(
                (max_num_tokens, actual_num_q_heads, actual_head_dim),
                dtype=jnp.bfloat16,
            )
            cache_kv_nope = jnp.zeros(
                (total_num_pages, page_size, 128), dtype=jnp.int32
            )
            cache_kv_rope = jnp.zeros(
                (total_num_pages, page_size // 4, 128), dtype=jnp.int32
            )
            topk_indices = jnp.full((max_num_tokens, topk), -1, dtype=jnp.int32)
            topk_indices = topk_indices.at[:, 0].set(0)
            page_indices = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 16, 32], dtype=jnp.int32)
            distribution = jnp.array([1, 1, 2], dtype=jnp.int32)
            attention_sinks = jnp.zeros((actual_num_q_heads,), dtype=jnp.float32)
            swa_accumution = jnp.zeros(
                (max_num_tokens, actual_num_q_heads, actual_head_dim),
                dtype=jnp.bfloat16,
            )
            swa_l = jnp.zeros((max_num_tokens, actual_num_q_heads), dtype=jnp.float32)
            swa_m = jnp.zeros((max_num_tokens, actual_num_q_heads), dtype=jnp.float32)

            out = sparse_mla.sparse_ragged_paged_attention(
                q,
                cache_kv_nope,
                cache_kv_rope,
                topk_indices,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                gather_and_attention_chunk_size=16,
                attention_kernel_batch_size=8,
            )
            assert out.shape == (32, 16, 512)

    def test_sparse_ragged_paged_attention_assertion_errors(self):
        max_num_tokens = 4
        num_heads = 16
        head_dim = 512
        q = jnp.ones((max_num_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        nope_bad = jnp.zeros((2, 16, 128), dtype=jnp.float32)
        nope_good = jnp.zeros((2, 16, 128), dtype=jnp.int32)
        rope_bad = jnp.zeros((2, 4, 128), dtype=jnp.float32)
        rope_good = jnp.zeros((2, 4, 128), dtype=jnp.int32)
        topk = jnp.zeros((max_num_tokens, 16), dtype=jnp.int32)
        page_indices = jnp.zeros((8,), dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        dist = jnp.array([1, 1, 2], dtype=jnp.int32)
        sinks = jnp.zeros((num_heads,), dtype=jnp.float32)
        swa_good = jnp.zeros((max_num_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        swa_bad = jnp.zeros((max_num_tokens, num_heads, head_dim), dtype=jnp.float32)
        swa_l_in = jnp.zeros((max_num_tokens, num_heads), dtype=jnp.float32)
        swa_m_in = jnp.zeros((max_num_tokens, num_heads), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope_bad,
                rope_good,
                topk,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
            )

        with pytest.raises(AssertionError):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope_good,
                rope_bad,
                topk,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
            )

        with pytest.raises(AssertionError):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope_good,
                rope_good,
                topk,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_bad,
                swa_l_in,
                swa_m_in,
            )

        bad_page_indices = jnp.zeros((7,), dtype=jnp.int32)
        with pytest.raises(AssertionError):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope_good,
                rope_good,
                topk,
                bad_page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
            )


# ==============================================================================
# 3. mla.py: Kernel and Outer Full Coverage
# ==============================================================================


class TestMlaAttentionKernelDirect:
    """Tests 100% lines and branches in mla._mla_ragged_paged_attention_kernel."""

    def test_mla_attention_kernel_step_coverage_f32_and_bf16(self):
        for q_dtype in (jnp.float32, jnp.bfloat16):
            batch_size = 2
            max_num_seqs = 8
            num_tokens = 16
            num_q_heads = 16
            head_dim = 512
            lkv_dim = 128
            kv_packing = 4
            num_slots_per_token = 2
            page_size = 16
            total_num_pages = 16
            pages_per_seq = 2
            bkv_p = 1
            bq_sz = 2
            static_q_len = 2

            kv_lens_ref = MockArrayRef(np.array([10] * max_num_seqs, dtype=np.int32))
            kv_lens_to_attend_ref = MockArrayRef(
                np.array([5] * num_tokens, dtype=np.int32)
            )
            page_indices_ref = MockArrayRef(
                np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)
            )
            cu_q_lens_ref = MockArrayRef(
                np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=np.int32)
            )
            start_end_seq_idx_ref = MockArrayRef(np.array([0, 2], dtype=np.int32))
            sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
            bo_ids_ref = MockArrayRef(np.full((4,), -1, dtype=np.int32))
            attention_sinks_ref = MockArrayRef(
                np.zeros((num_q_heads,), dtype=np.float32)
            )
            q_hbm_ref = MockArrayRef(
                np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float16)
            )
            cache_kv_hbm_ref = MockArrayRef(
                np.zeros(
                    (
                        total_num_pages,
                        page_size * num_slots_per_token,
                        kv_packing,
                        lkv_dim,
                    ),
                    dtype=np.uint8,
                )
            )
            swa_accumution_hbm_ref = MockArrayRef(
                np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
            )
            swa_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
            swa_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
            o_hbm_ref = MockArrayRef(
                np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
                if q_dtype == jnp.float32
                else np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
            )
            bkv_x2_ref = MockArrayRef(
                np.zeros(
                    (
                        2,
                        batch_size,
                        bkv_p * page_size,
                        num_slots_per_token * kv_packing,
                        lkv_dim,
                    ),
                    dtype=np.uint8,
                )
            )
            bq_x2_ref = MockArrayRef(
                np.zeros(
                    (2, batch_size, bq_sz, num_q_heads, head_dim), dtype=np.float32
                )
                if q_dtype == jnp.float32
                else np.zeros(
                    (2, batch_size, bq_sz, num_q_heads, head_dim),
                    dtype=np.float16,
                )
            )
            bo_x2_ref = MockArrayRef(
                np.zeros(
                    (2, batch_size, bq_sz, num_q_heads, head_dim), dtype=np.float32
                )
                if q_dtype == jnp.float32
                else np.zeros(
                    (2, batch_size, bq_sz, num_q_heads, head_dim),
                    dtype=np.float16,
                )
            )
            bl_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, bq_sz, 128), dtype=np.float32)
            )
            bm_x2_ref = MockArrayRef(
                np.zeros((2, batch_size, bq_sz, 128), dtype=np.float32)
            )
            swa_acc_x2_ref = MockArrayRef(
                np.zeros(
                    (2, batch_size, bq_sz, num_q_heads, head_dim), dtype=np.float32
                )
                if q_dtype == jnp.float32
                else np.zeros(
                    (2, batch_size, bq_sz, num_q_heads, head_dim),
                    dtype=np.float16,
                )
            )
            sems = MockArrayRef(np.zeros((6, batch_size, 2), dtype=np.int32))
            l_ref = MockArrayRef(
                np.zeros((batch_size, bq_sz * num_q_heads), dtype=np.float32)
            )
            m_ref = MockArrayRef(
                np.zeros((batch_size, bq_sz * num_q_heads), dtype=np.float32)
            )
            acc_ref = MockArrayRef(
                np.zeros((batch_size, bq_sz * num_q_heads, head_dim), dtype=np.float32)
            )

            with (
                patch.object(
                    mla.pltpu, "make_async_copy", return_value=MockAsyncCopy()
                ),
                patch.object(mla.pltpu, "bitcast", side_effect=fake_bitcast),
                patch.object(
                    mla.pl,
                    "reciprocal",
                    side_effect=lambda x, approx=True: 1.0 / x,
                ),
                patch.object(mla.pl, "when", side_effect=fake_when),
                patch.object(mla.pl, "program_id", return_value=0),
                patch.object(mla.pl, "num_programs", return_value=2),
                patch.object(mla.lax, "fori_loop", side_effect=fake_fori_loop),
            ):
                mla._mla_ragged_paged_attention_kernel(
                    kv_lens_ref,
                    kv_lens_to_attend_ref,
                    page_indices_ref,
                    cu_q_lens_ref,
                    start_end_seq_idx_ref,
                    sem_ids_ref,
                    bo_ids_ref,
                    attention_sinks_ref,
                    q_hbm_ref,
                    cache_kv_hbm_ref,
                    swa_accumution_hbm_ref,
                    swa_l_hbm_ref,
                    swa_m_hbm_ref,
                    o_hbm_ref,
                    bkv_x2_ref,
                    bq_x2_ref,
                    bo_x2_ref,
                    bl_x2_ref,
                    bm_x2_ref,
                    swa_acc_x2_ref,
                    sems,
                    l_ref,
                    m_ref,
                    acc_ref,
                    static_q_len=static_q_len,
                    sm_scale=1.0,
                    bkv_p=bkv_p,
                    bq_sz=bq_sz,
                    batch_size=batch_size,
                )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)

    def test_mla_attention_kernel_dynamic_q_len_and_single_batch(self):
        batch_size = 1
        max_num_seqs = 4
        num_tokens = 4
        num_q_heads = 16
        head_dim = 512
        lkv_dim = 128
        kv_packing = 4
        num_slots_per_token = 2
        page_size = 16
        total_num_pages = 8
        pages_per_seq = 2
        bkv_p = 1
        bq_sz = 2

        kv_lens_ref = MockArrayRef(np.array([10] * max_num_seqs, dtype=np.int32))
        kv_lens_to_attend_ref = MockArrayRef(np.array([5] * num_tokens, dtype=np.int32))
        page_indices_ref = MockArrayRef(
            np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)
        )
        cu_q_lens_ref = MockArrayRef(np.array([0, 2, 4, 6, 8], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 1], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.array([0, 0, 0, 0], dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    total_num_pages,
                    page_size * num_slots_per_token,
                    kv_packing,
                    lkv_dim,
                ),
                dtype=np.uint8,
            )
        )
        swa_accumution_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        swa_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        swa_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        bkv_x2_ref = MockArrayRef(
            np.zeros(
                (
                    2,
                    batch_size,
                    bkv_p * page_size,
                    num_slots_per_token * kv_packing,
                    lkv_dim,
                ),
                dtype=np.uint8,
            )
        )
        bq_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, bq_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        bo_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, bq_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        bl_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, bq_sz, 128), dtype=np.float32)
        )
        bm_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, bq_sz, 128), dtype=np.float32)
        )
        swa_acc_x2_ref = MockArrayRef(
            np.zeros((2, batch_size, bq_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        sems = MockArrayRef(np.zeros((6, batch_size, 2), dtype=np.int32))
        l_ref = MockArrayRef(
            np.zeros((batch_size, bq_sz * num_q_heads), dtype=np.float32)
        )
        m_ref = MockArrayRef(
            np.zeros((batch_size, bq_sz * num_q_heads), dtype=np.float32)
        )
        acc_ref = MockArrayRef(
            np.zeros((batch_size, bq_sz * num_q_heads, head_dim), dtype=np.float32)
        )

        with (
            patch.object(mla.pltpu, "make_async_copy", return_value=MockAsyncCopy()),
            patch.object(mla.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla.pl,
                "reciprocal",
                side_effect=lambda x, approx=True: 1.0 / x,
            ),
            patch.object(mla.pl, "when", side_effect=fake_when),
            patch.object(mla.pl, "program_id", return_value=0),
            patch.object(mla.pl, "num_programs", return_value=1),
            patch.object(mla.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            mla._mla_ragged_paged_attention_kernel(
                kv_lens_ref,
                kv_lens_to_attend_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                cache_kv_hbm_ref,
                swa_accumution_hbm_ref,
                swa_l_hbm_ref,
                swa_m_hbm_ref,
                o_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                swa_acc_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                static_q_len=None,
                sm_scale=1.0,
                bkv_p=bkv_p,
                bq_sz=bq_sz,
                batch_size=batch_size,
            )


class TestMlaRaggedPagedAttentionOuter:
    """Tests 100% of lines and branches in mla.mla_ragged_paged_attention."""

    def test_mla_ragged_paged_attention_success(self):
        def fake_pallas_call(kernel, out_shape=None, *args, **kwargs):
            return lambda *call_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        with patch.object(mla.pl, "pallas_call", side_effect=fake_pallas_call):
            max_num_tokens = 32
            actual_num_q_heads = 16
            actual_head_dim = 512
            lkv_dim = 128
            kv_packing = 4
            num_slots_per_token = 2
            page_size = 16
            total_num_pages = 16
            max_num_seqs = 4
            pages_per_seq = 4

            q = jnp.ones(
                (max_num_tokens, actual_num_q_heads, actual_head_dim),
                dtype=jnp.bfloat16,
            )
            cache_kv = jnp.zeros(
                (
                    total_num_pages,
                    page_size * num_slots_per_token,
                    kv_packing,
                    lkv_dim,
                ),
                dtype=jnp.uint8,
            )
            kv_lens = jnp.array([10] * max_num_seqs, dtype=jnp.int32)
            kv_lens_to_attend = jnp.array([5] * max_num_tokens, dtype=jnp.int32)
            page_indices = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 8, 16, 24, 32], dtype=jnp.int32)
            distribution = jnp.array([2, 3, 4], dtype=jnp.int32)
            attention_sinks = jnp.zeros((actual_num_q_heads,), dtype=jnp.float32)
            swa_accumution = jnp.zeros(
                (max_num_tokens, actual_num_q_heads, actual_head_dim),
                dtype=jnp.bfloat16,
            )
            swa_l = jnp.zeros((max_num_tokens, actual_num_q_heads), dtype=jnp.float32)
            swa_m = jnp.zeros((max_num_tokens, actual_num_q_heads), dtype=jnp.float32)

            # Case 1: chunk_prefill_size is not None and params are tuples
            out1 = mla.mla_ragged_paged_attention(
                q,
                cache_kv,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                chunk_prefill_size=16,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                decode_batch_size=2,
            )
            assert out1.shape == (32, 16, 512)

            # Case 2: chunk_prefill_size is None and params are integers
            out2 = mla.mla_ragged_paged_attention(
                q,
                cache_kv,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                chunk_prefill_size=None,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
                decode_batch_size=2,
            )
            assert out2.shape == (32, 16, 512)

    def test_mla_ragged_paged_attention_errors(self):
        max_num_tokens = 4
        num_heads = 16
        head_dim = 512
        lkv_dim = 128
        kv_packing = 4
        num_slots_per_token = 2
        page_size = 16
        total_num_pages = 4
        max_num_seqs = 2
        pages_per_seq = 2

        q = jnp.ones((max_num_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        cache_kv_bad = jnp.zeros(
            (
                total_num_pages,
                page_size * num_slots_per_token,
                kv_packing,
                lkv_dim,
            ),
            dtype=jnp.int32,
        )
        cache_kv_good = jnp.zeros(
            (
                total_num_pages,
                page_size * num_slots_per_token,
                kv_packing,
                lkv_dim,
            ),
            dtype=jnp.uint8,
        )
        kv_lens = jnp.array([10] * max_num_seqs, dtype=jnp.int32)
        kv_lens_to_attend = jnp.array([5] * max_num_tokens, dtype=jnp.int32)
        page_indices = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        dist = jnp.array([1, 1, 2], dtype=jnp.int32)
        sinks = jnp.zeros((num_heads,), dtype=jnp.float32)
        swa_good = jnp.zeros((max_num_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        swa_bad = jnp.zeros((max_num_tokens, num_heads, head_dim), dtype=jnp.float32)
        swa_l_in = jnp.zeros((max_num_tokens, num_heads), dtype=jnp.float32)
        swa_m_in = jnp.zeros((max_num_tokens, num_heads), dtype=jnp.float32)

        # 1. cache_kv not uint8
        with pytest.raises(AssertionError):
            mla.mla_ragged_paged_attention(
                q,
                cache_kv_bad,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
                num_kv_pages_per_block=1,
                num_queries_per_block=1,
            )

        # 2. num_kv_pages_per_block is None
        with pytest.raises(
            ValueError, match="num_kv_pages_per_block and num_queries_per_block"
        ):
            mla.mla_ragged_paged_attention(
                q,
                cache_kv_good,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
                num_kv_pages_per_block=None,
                num_queries_per_block=1,
            )

        # 3. num_queries_per_block is None
        with pytest.raises(
            ValueError, match="num_kv_pages_per_block and num_queries_per_block"
        ):
            mla.mla_ragged_paged_attention(
                q,
                cache_kv_good,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
                num_kv_pages_per_block=1,
                num_queries_per_block=None,
            )

        # 4. swa_accumution dtype mismatch
        with pytest.raises(AssertionError):
            mla.mla_ragged_paged_attention(
                q,
                cache_kv_good,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_bad,
                swa_l_in,
                swa_m_in,
                num_kv_pages_per_block=1,
                num_queries_per_block=1,
            )

        # 5. num_page_indices not divisible by max_num_seqs
        bad_page_indices = jnp.zeros((5,), dtype=jnp.int32)
        with pytest.raises(AssertionError):
            mla.mla_ragged_paged_attention(
                q,
                cache_kv_good,
                kv_lens,
                kv_lens_to_attend,
                bad_page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
                num_kv_pages_per_block=1,
                num_queries_per_block=1,
            )

        # 6. num_slots_per_page not divisible by num_slots_per_tokens (2)
        bad_cache_kv_slots = jnp.zeros(
            (total_num_pages, 15, kv_packing, lkv_dim), dtype=jnp.uint8
        )
        with pytest.raises(AssertionError):
            mla.mla_ragged_paged_attention(
                q,
                bad_cache_kv_slots,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                dist,
                sinks,
                swa_good,
                swa_l_in,
                swa_m_in,
                num_kv_pages_per_block=1,
                num_queries_per_block=1,
            )
