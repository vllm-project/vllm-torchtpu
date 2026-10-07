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

"""100% full-module line and branch coverage tests for:

1. src/vllm_torchtpu/kernels/ragged_paged_attention/v3/util.py
2. src/vllm_torchtpu/kernels/ragged_paged_attention/v3/tuned_block_sizes_hd64.py
3. src/vllm_torchtpu/kernels/ragged_paged_attention/v3/kernel_hd64.py
4. src/vllm_torchtpu/models/vllm/deepseek_v41/common/mhc.py
and verification of v2/cache.py donate_argnums=(0,) in _write_op.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel_hd64 as K64
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    tuned_block_sizes_hd64 as TBS64,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import util as rpa_util
from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache

# ==============================================================================
# Simulation Primitives & Mock Helpers
# ==============================================================================


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
    """Mock buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(self, arr: np.ndarray | jax.Array):
        self.arr = np.asarray(arr)
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        return jnp.asarray(self.arr[_norm_idx(idx)])

    def __setitem__(self, idx, val):
        if isinstance(val, MockArrayRef):
            val = val.arr
        val_arr = np.asarray(val)
        if idx is Ellipsis:
            if self.arr.shape != val_arr.shape:
                val_arr = val_arr.reshape(-1)[: self.arr.size].reshape(self.arr.shape)
            self.arr[...] = val_arr
        else:
            self.arr[_norm_idx(idx)] = val_arr

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                if idx is Ellipsis:
                    return self.ref
                view = self.ref.arr[_norm_idx(idx)]
                return MockArrayRef(view)

            def __setitem__(self, idx, val):
                self.ref[_norm_idx(idx)] = val

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
    """Executes inner function when condition evaluates to True."""

    def decorator(fn):
        execute = True
        try:
            execute = bool(cond)
        except Exception:
            try:
                execute = bool(np.all(cond))
            except Exception:
                execute = True
        if execute:
            fn()
        return fn

    return decorator


def fake_bitcast(x, dt):
    arr = x.arr if isinstance(x, MockArrayRef) else x
    res = jnp.asarray(arr).astype(dt)
    if (
        dt in (jnp.bfloat16, jnp.float16)
        and arr.dtype in (np.uint32, np.int32)
        and res.ndim == 2
        and res.shape[1] == 128
    ):
        res = jnp.repeat(res, 2, axis=0)
    return res


def fake_fori_loop(lower, upper, body_fun, init_val, unroll=False):
    val = init_val
    for i in range(int(np.asarray(lower)), int(np.asarray(upper))):
        val = body_fun(i, val)
    return val


# ==============================================================================
# 1. Tests for ragged_paged_attention/v3/util.py
# ==============================================================================


class TestRpaV3Util:
    """100% full-module line and branch coverage for util.py."""

    def test_cdiv_and_align_to(self):
        assert rpa_util.cdiv(10, 3) == 4
        assert rpa_util.cdiv(9, 3) == 3
        with pytest.raises(AssertionError):
            rpa_util.cdiv(10, 0)

        assert rpa_util.align_to(10, 8) == 16
        assert rpa_util.align_to(16, 8) == 16
        assert rpa_util.align_to(0, 8) == 0

    def test_dtype_bitwidth_and_packing(self):
        assert rpa_util.get_dtype_bitwidth(jnp.float32) == 32
        assert rpa_util.get_dtype_bitwidth(jnp.bfloat16) == 16
        assert rpa_util.get_dtype_bitwidth(jnp.uint8) == 8

        assert rpa_util.get_dtype_packing(jnp.float32) == 1
        assert rpa_util.get_dtype_packing(jnp.bfloat16) == 2
        assert rpa_util.get_dtype_packing(jnp.uint8) == 4

    def test_next_power_of_2(self):
        assert rpa_util.next_power_of_2(1) == 1
        assert rpa_util.next_power_of_2(2) == 2
        assert rpa_util.next_power_of_2(3) == 4
        assert rpa_util.next_power_of_2(15) == 16
        assert rpa_util.next_power_of_2(16) == 16
        assert rpa_util.next_power_of_2(17) == 32
        with pytest.raises(AssertionError):
            rpa_util.next_power_of_2(0)

    def test_get_tpu_version_all_branches(self, monkeypatch):
        # 1. Non-TPU device
        mock_cpu_dev = SimpleNamespace(device_kind="cpu")
        monkeypatch.setattr(jax, "devices", lambda: [mock_cpu_dev])
        assert rpa_util.get_tpu_version() == -1

        # 2. TPU v5e lite (ends with ' lite' and 'e')
        mock_v5e_lite = SimpleNamespace(device_kind="TPU v5e lite")
        monkeypatch.setattr(jax, "devices", lambda: [mock_v5e_lite])
        assert rpa_util.get_tpu_version() == 5

        # 3. TPU v4p (ends with 'p')
        mock_v4p = SimpleNamespace(device_kind="TPU v4p")
        monkeypatch.setattr(jax, "devices", lambda: [mock_v4p])
        assert rpa_util.get_tpu_version() == 4

        # 4. TPU7x
        mock_v7 = SimpleNamespace(device_kind="TPU7x")
        monkeypatch.setattr(jax, "devices", lambda: [mock_v7])
        assert rpa_util.get_tpu_version() == 7

        # 5. TPU v6 (generic TPU vX)
        mock_v6 = SimpleNamespace(device_kind="TPU v6")
        monkeypatch.setattr(jax, "devices", lambda: [mock_v6])
        assert rpa_util.get_tpu_version() == 6


# ==============================================================================
# 2. Tests for ragged_paged_attention/v3/tuned_block_sizes_hd64.py
# ==============================================================================


class TestTunedBlockSizesHd64:
    """100% full-module line and branch coverage for tuned_block_sizes_hd64.py."""

    def test_get_simplified_raw_key_assertions(self):
        with pytest.raises(AssertionError):
            TBS64.get_simplified_raw_key(
                page_size=128,
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.bfloat16,
                actual_num_q_heads=8,
                actual_num_kv_heads=2,
                head_dim=128,
                max_model_len=4096,
            )

        with pytest.raises(AssertionError):
            TBS64.get_simplified_raw_key(
                page_size=128,
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.bfloat16,
                actual_num_q_heads=7,
                actual_num_kv_heads=2,
                head_dim=64,
                max_model_len=4096,
            )

    def test_get_simplified_raw_key_and_lookup_keys(self, monkeypatch):
        monkeypatch.setattr(TBS64, "get_device_name", lambda: "TPU v5e")
        keys = TBS64.get_lookup_keys(
            page_size=128,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=64,
            max_model_len=4096,
        )
        assert keys == (
            "TPU v5e",
            128,
            "q_bfloat16_kv_bfloat16",
            "q_head-8_kv_head-2_head-64",
            4096,
        )

    def test_get_tuned_block_sizes_branches(self, monkeypatch):
        # 1. Version < 4 raises NotImplementedError
        monkeypatch.setattr(TBS64, "get_tpu_version", lambda: 3)
        with pytest.raises(NotImplementedError, match="TPU version must be 4"):
            TBS64.get_tuned_block_sizes(
                jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
            )

        # 2. Case 4 (fallback KeyError)
        monkeypatch.setattr(TBS64, "get_tpu_version", lambda: 4)
        monkeypatch.setattr(TBS64, "get_device_name", lambda: "TPU v4")
        bkv_p, bq = TBS64.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
        )
        assert (bkv_p, bq) == (512 // 128, 32)

        # 3. Case 7 (found in table)
        monkeypatch.setattr(TBS64, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(TBS64, "get_device_name", lambda: "TPU v7")
        bkv_p, bq = TBS64.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
        )
        assert (bkv_p, bq) == (32, 16)

        # 4. Default case (TPU v6e, found in table)
        monkeypatch.setattr(TBS64, "get_tpu_version", lambda: 6)
        monkeypatch.setattr(TBS64, "get_device_name", lambda: "TPU v6e")
        bkv_p, bq = TBS64.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 4096, 32
        )
        assert (bkv_p, bq) == (32, 32)

        # 5. Clamps: pages_per_seq < bkv_p and max_num_tokens < bq
        monkeypatch.setattr(TBS64, "get_tpu_version", lambda: 5)
        monkeypatch.setattr(TBS64, "get_device_name", lambda: "TPU v5e")
        bkv_p, bq = TBS64.get_tuned_block_sizes(
            jnp.bfloat16, jnp.bfloat16, 8, 2, 64, 128, 10, 2
        )
        assert (bkv_p, bq) == (1, 10)


# ==============================================================================
# 3. Tests for ragged_paged_attention/v3/kernel_hd64.py Helpers & Validation
# ==============================================================================


class TestRpaV3KernelHd64HelpersAndValidation:
    """Tests static/dynamic validation and helper functions in kernel_hd64.py."""

    def test_estimates_and_shapes(self):
        smem = K64.get_smem_estimate_bytes(max_num_seqs=4, pages_per_seq=8)
        assert smem > 0

        vmem = K64.get_vmem_estimate_bytes(
            actual_num_kv_heads=2,
            actual_num_q_heads_per_kv_head=4,
            actual_head_dim=64,
            bq_sz=32,
            bkv_sz=256,
            q_dtype=jnp.bfloat16,
            kv_dtype=jnp.bfloat16,
        )
        assert vmem > 0

        with pytest.raises(AssertionError):
            K64.get_vmem_estimate_bytes(2, 4, 128, 32, 256, jnp.bfloat16, jnp.bfloat16)

        shape = K64.get_kv_cache_shape(
            total_num_pages=16,
            page_size=128,
            actual_num_kv_heads=2,
            actual_head_dim=64,
            kv_dtype=jnp.bfloat16,
        )
        assert shape == (16, 128, 1, 2, 128)

        with pytest.raises(AssertionError):
            K64.get_kv_cache_shape(16, 128, 2, 128, jnp.bfloat16)

    def test_merge_kv_and_prepare(self):
        k = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        v = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        merged = K64.merge_kv(k, v)
        assert merged.shape == (4, 1, 2, 128)

        with pytest.raises(AssertionError):
            K64.merge_kv(k, v[:3])

        with pytest.raises(AssertionError):
            K64.merge_kv(k, v.astype(jnp.float32))

        q = jnp.ones((4, 8, 64), dtype=jnp.bfloat16)
        sink = jnp.ones((8,), dtype=jnp.float32)

        # 1. With attention sink
        q_prep, kv_prep, sink_prep = K64.prepare_inputs(q, k, v, sink)
        assert q_prep.shape == (2, 4, 2, 2, 128)
        assert kv_prep.shape == (4, 1, 2, 128)
        assert sink_prep.shape == (2, 4, 128)

        # 2. Without attention sink
        q_prep2, kv_prep2, sink_prep2 = K64.prepare_inputs(q, k, v, None)
        assert sink_prep2 is None

        # 3. Prepare outputs
        out = jnp.ones((2, 4, 2, 2, 128), dtype=jnp.bfloat16)
        prepared_out = K64.prepare_outputs(out, 4, 64)
        assert prepared_out.shape == (4, 8, 64)

    def test_static_validate_inputs_all_errors(self):
        q = jnp.ones((4, 8, 64), dtype=jnp.bfloat16)
        k = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        v = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.ones((8, 16, 1, 2, 128), dtype=jnp.bfloat16)
        kv_lens = jnp.array([16, 16], dtype=jnp.int32)
        page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
        sink = jnp.ones((8,), dtype=jnp.float32)

        # Success case
        K64.static_validate_inputs(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sink,
        )

        # 1. Not 3D
        with pytest.raises(ValueError, match="Expected 3D array"):
            K64.static_validate_inputs(
                q.reshape(32, 64),
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 2. k.shape != v.shape
        with pytest.raises(ValueError, match="Expected k.shape"):
            K64.static_validate_inputs(
                q,
                k,
                v[:3],
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 3. q.shape[0] != k.shape[0]
        with pytest.raises(ValueError, match="Expected q.shape\\[0\\]"):
            K64.static_validate_inputs(
                q[:3],
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 4. q.shape[2] != k.shape[2]
        with pytest.raises(ValueError, match="Expected q.shape\\[2\\]"):
            K64.static_validate_inputs(
                q[:, :, :32],
                k[:, :, :16],
                v[:, :, :16],
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 5. sink shape != (num_q_heads,)
        with pytest.raises(ValueError, match="Expected attention_sink.shape\\[0\\]"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sink[:4],
            )

        # 6. sink dtype != float32
        with pytest.raises(ValueError, match="Expected attention_sink.dtype"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sink.astype(jnp.float16),
            )

        # 7. actual_head_dim != 64
        q_bad_dim = jnp.ones((4, 8, 32), dtype=jnp.bfloat16)
        k_bad_dim = jnp.ones((4, 2, 32), dtype=jnp.bfloat16)
        v_bad_dim = jnp.ones((4, 2, 32), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="Expected actual_head_dim=32 to be 64"):
            K64.static_validate_inputs(
                q_bad_dim,
                k_bad_dim,
                v_bad_dim,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 8. actual_num_q_heads % actual_num_kv_heads != 0
        q_bad_h = jnp.ones((4, 7, 64), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="Expected actual_num_q_heads=7"):
            K64.static_validate_inputs(
                q_bad_h,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 9. actual_head_dim_x2 != 128
        with pytest.raises(ValueError, match="Expected actual_head_dim_x2"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache[..., :64],
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 10. kv_cache dtype != k.dtype
        with pytest.raises(ValueError, match="Expected kv_cache.dtype"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache.astype(jnp.float32),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 11. kv_cache dtype not floating
        k_int = jnp.ones((4, 2, 64), dtype=jnp.int32)
        v_int = jnp.ones((4, 2, 64), dtype=jnp.int32)
        kv_int = jnp.ones((8, 16, 1, 2, 128), dtype=jnp.int32)
        with pytest.raises(ValueError, match="floating point"):
            K64.static_validate_inputs(
                q,
                k_int,
                v_int,
                kv_int,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 12. kv_packing mismatch
        kv_bad_pack = jnp.ones((8, 16, 2, 1, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="does not match"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_bad_pack,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 13. num_kv_heads alignment mismatch
        kv_bad_heads = jnp.ones((8, 16, 2, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="Invalid num_kv_heads"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_bad_heads,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 14. Integer dtypes not int32
        with pytest.raises(ValueError, match="Expected int32 dtype"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens.astype(jnp.int16),
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 15. Non-1D arrays for prefetch
        with pytest.raises(ValueError, match="Expected 1D array"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens.reshape(1, 2),
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 16. page_indices divisibility by max_num_seqs
        with pytest.raises(ValueError, match="divisible by max_num_seqs"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices[:3],
                cu_q_lens,
                distribution,
            )

        # 17. cu_q_lens shape != (max_num_seqs + 1,)
        with pytest.raises(ValueError, match="Expected cu_q_lens.shape"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens[:2],
                distribution,
            )

        # 18. distribution shape != (3,)
        with pytest.raises(ValueError, match="Expected distribution.shape"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution[:2],
            )

        # 19. page_size % kv_packing != 0
        kv_cache_odd = jnp.ones((8, 15, 1, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="divisible by kv_packing"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache_odd,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 20. Non-positive parameters
        with pytest.raises(ValueError, match="sliding_window.*must be positive"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sliding_window=0,
            )

        with pytest.raises(ValueError, match="soft_cap.*must not be 0.0"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                soft_cap=0.0,
            )

        with pytest.raises(ValueError, match="chunk_prefill_size.*must be positive"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                chunk_prefill_size=-1,
            )

        with pytest.raises(
            ValueError, match="num_kv_pages_per_block.*must be positive"
        ):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                num_kv_pages_per_block=0,
            )

        with pytest.raises(ValueError, match="num_queries_per_block.*must be positive"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                num_queries_per_block=-2,
            )

        with pytest.raises(ValueError, match="vmem_limit_bytes.*must be positive"):
            K64.static_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                vmem_limit_bytes=-100,
            )

    def test_dynamic_validate_inputs_all_errors(self):
        q = jnp.ones((4, 8, 64), dtype=jnp.bfloat16)
        k = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        v = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.ones((8, 16, 1, 2, 128), dtype=jnp.bfloat16)
        kv_lens = jnp.array([16, 16], dtype=jnp.int32)
        page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        distribution = jnp.array([1, 2, 2], dtype=jnp.int32)

        # 1. Invalid distribution order
        with pytest.raises(ValueError, match="Invalid distribution"):
            K64.dynamic_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                jnp.array([2, 1, 2], dtype=jnp.int32),
            )

        # 2. k > max_num_seqs
        with pytest.raises(ValueError, match="must be <= max_num_seqs"):
            K64.dynamic_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                jnp.array([1, 2, 5], dtype=jnp.int32),
            )

        # 3. cu_q_lens[k] > max_num_tokens
        with pytest.raises(ValueError, match="must be <= max_num_tokens"):
            K64.dynamic_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                jnp.array([0, 2, 10], dtype=jnp.int32),
                distribution,
            )

        # 4. Require 0 < q_len <= kv_len
        with pytest.raises(ValueError, match="Require 0 < q_len="):
            K64.dynamic_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                jnp.array([1, 16], dtype=jnp.int32),  # q_len=2 > kv_len=1
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 5. page_cnt > pages_per_seq
        with pytest.raises(ValueError, match="Require page_cnt="):
            K64.dynamic_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                jnp.array([100, 16], dtype=jnp.int32),  # page_cnt=7 > 2
                page_indices,
                cu_q_lens,
                distribution,
            )

        # 6. page_idx < 0 or >= total_num_pages
        with pytest.raises(ValueError, match="Require 0 <= page_idx"):
            K64.dynamic_validate_inputs(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                jnp.array([99, 1, 2, 3], dtype=jnp.int32),  # page_idx=99 >= 8
                cu_q_lens,
                distribution,
            )


# ==============================================================================
# 4. Tests for ref_ragged_paged_attention_hd64
# ==============================================================================


class TestRefRaggedPagedAttentionHd64:
    """Tests ref_ragged_paged_attention_hd64 across decoding, prefill, and flags."""

    def test_ref_execution_all_branches(self):
        max_num_tokens = 4
        actual_num_q_heads = 8
        actual_num_kv_heads = 2
        actual_head_dim = 64
        total_num_pages = 4
        page_size = 16

        queries = jnp.ones(
            (max_num_tokens, actual_num_q_heads, actual_head_dim), dtype=jnp.float32
        )
        keys = jnp.ones(
            (max_num_tokens, actual_num_kv_heads, actual_head_dim), dtype=jnp.float32
        )
        values = jnp.ones(
            (max_num_tokens, actual_num_kv_heads, actual_head_dim), dtype=jnp.float32
        )
        kv_cache = jnp.ones((total_num_pages, page_size, 2, 1, 128), dtype=jnp.float32)
        kv_lens = jnp.array([16, 16], dtype=jnp.int32)
        page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
        sink = jnp.ones((actual_num_q_heads,), dtype=jnp.float32)

        # 1. Full features: mask_value=None, sink, scales, soft_cap, sliding_window
        out1, cache1 = K64.ref_ragged_paged_attention_hd64(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sink=sink,
            mask_value=None,
            q_scale=1.5,
            k_scale=0.8,
            v_scale=1.2,
            sliding_window=8,
            soft_cap=50.0,
            skip_kv_update=False,
            use_causal_mask=True,
        )
        assert out1.shape == (max_num_tokens, actual_num_q_heads, actual_head_dim)
        assert cache1.shape == kv_cache.shape

        # 2. skip_kv_update=True, no sink, use_causal_mask=False
        out2, cache2 = K64.ref_ragged_paged_attention_hd64(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sink=None,
            mask_value=-1e9,
            q_scale=None,
            k_scale=None,
            v_scale=None,
            sliding_window=None,
            soft_cap=None,
            skip_kv_update=True,
            use_causal_mask=False,
        )
        assert out2.shape == (max_num_tokens, actual_num_q_heads, actual_head_dim)
        assert cache2.shape == kv_cache.shape

        # 3. Non-floating kv dtype with q_scale (covers branch 131->136)
        with patch.object(K64, "dynamic_validate_inputs", lambda *a, **k: None):
            out3, _ = K64.ref_ragged_paged_attention_hd64(
                queries,
                keys.astype(jnp.int32),
                values.astype(jnp.int32),
                kv_cache.astype(jnp.int32),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                q_scale=1.5,
            )
            assert out3.shape == (max_num_tokens, actual_num_q_heads, actual_head_dim)


# ==============================================================================
# 5. Tests for _ragged_paged_attention_kernel Simulation & Public Call
# ==============================================================================


class TestRpaV3KernelHd64PallasSimulationAndEntrypoint:
    """100% line & branch coverage for Pallas kernel & ragged_paged_attention_hd64."""

    def _make_kernel_refs(
        self,
        q_dtype=np.float32,
        kv_dtype=np.float32,
        bkv_p=2,
        bq_sz=2,
        max_num_seqs=2,
        pages_per_seq=2,
        actual_num_kv_heads=2,
        actual_num_q_heads=4,
    ):
        num_tokens = max_num_seqs * 2
        page_size = 16
        bkv_sz = bkv_p * page_size
        total_num_pages = max_num_seqs * pages_per_seq
        q_packing = rpa_util.get_dtype_packing(q_dtype)
        kv_packing = rpa_util.get_dtype_packing(kv_dtype)
        num_kv_heads = rpa_util.align_to(actual_num_kv_heads, kv_packing)
        num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

        kv_lens_ref = MockArrayRef(np.full((max_num_seqs,), 16, dtype=np.int32))
        page_indices_ref = MockArrayRef(
            np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)
        )
        cu_q_lens_ref = MockArrayRef(np.arange(max_num_seqs + 1, dtype=np.int32) * 2)
        distribution_ref = MockArrayRef(np.array([1, 2, max_num_seqs], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.full((4,), -1, dtype=np.int32))
        bkv_update_ids_ref = MockArrayRef(np.full((6,), -1, dtype=np.int32))

        q_hbm_ref = MockArrayRef(
            np.ones(
                (
                    actual_num_kv_heads,
                    num_tokens,
                    num_q_heads_per_kv_head // q_packing,
                    q_packing,
                    128,
                ),
                dtype=q_dtype,
            )
        )
        kv_hbm_ref = MockArrayRef(
            np.ones(
                (
                    num_tokens,
                    num_kv_heads // kv_packing,
                    kv_packing,
                    128,
                ),
                dtype=kv_dtype,
            )
        )
        kv_cache_hbm_ref = MockArrayRef(
            np.ones(
                (
                    total_num_pages,
                    page_size,
                    num_kv_heads // kv_packing,
                    kv_packing,
                    128,
                ),
                dtype=kv_dtype,
            )
        )
        attention_sink_ref = MockArrayRef(
            np.zeros(
                (actual_num_kv_heads, num_q_heads_per_kv_head, 128),
                dtype=np.float32,
            )
        )
        o_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    actual_num_kv_heads,
                    num_tokens,
                    num_q_heads_per_kv_head // q_packing,
                    q_packing,
                    128,
                ),
                dtype=q_dtype,
            )
        )
        updated_kv_cache_hbm_ref = MockArrayRef(
            np.zeros(
                (
                    total_num_pages,
                    page_size,
                    num_kv_heads // kv_packing,
                    kv_packing,
                    128,
                ),
                dtype=kv_dtype,
            )
        )

        bkv_x2_ref = MockArrayRef(
            np.zeros(
                (
                    2,
                    bkv_sz,
                    num_kv_heads // kv_packing,
                    kv_packing,
                    128,
                ),
                dtype=kv_dtype,
            )
        )
        bq_x2_ref = MockArrayRef(
            np.zeros(
                (
                    2,
                    actual_num_kv_heads,
                    bq_sz,
                    num_q_heads_per_kv_head // q_packing,
                    q_packing,
                    128,
                ),
                dtype=q_dtype,
            )
        )
        bo_x2_ref = MockArrayRef(
            np.zeros(
                (
                    2,
                    actual_num_kv_heads,
                    bq_sz,
                    num_q_heads_per_kv_head // q_packing,
                    q_packing,
                    128,
                ),
                dtype=q_dtype,
            )
        )
        sems = MockArrayRef(np.zeros((4, 2), dtype=np.int32))
        l_ref = MockArrayRef(
            np.zeros(
                (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
                dtype=np.float32,
            )
        )
        m_ref = MockArrayRef(
            np.zeros(
                (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
                dtype=np.float32,
            )
        )
        acc_ref = MockArrayRef(
            np.zeros(
                (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
                dtype=np.float32,
            )
        )

        return (
            kv_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
            distribution_ref,
            sem_ids_ref,
            bo_ids_ref,
            bkv_update_ids_ref,
            q_hbm_ref,
            kv_hbm_ref,
            kv_cache_hbm_ref,
            attention_sink_ref,
            o_hbm_ref,
            updated_kv_cache_hbm_ref,
            bkv_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,
            l_ref,
            m_ref,
            acc_ref,
        )

    def test_kernel_simulation_debug_mode(self):
        refs = self._make_kernel_refs()
        with (
            patch.object(K64.pltpu, "make_async_copy", return_value=MockAsyncCopy()),
            patch.object(K64.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                K64.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(K64.pl, "when", side_effect=fake_when),
            patch.object(K64.pl, "program_id", return_value=0),
            patch.object(K64.pl, "num_programs", return_value=1),
            patch.object(K64.pl, "debug_print", side_effect=lambda *a, **k: None),
            patch.object(K64.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            K64._ragged_paged_attention_kernel(
                *refs,
                sm_scale=1.0,
                sliding_window=None,
                strict_sliding_window=True,
                soft_cap=None,
                mask_value=-1e9,
                q_scale=None,
                k_scale=None,
                v_scale=None,
                skip_kv_update=False,
                chunk_prefill_size=2,
                bkv_p=2,
                bq_sz=2,
                use_causal_mask=True,
                debug_mode=True,
            )

    def test_kernel_simulation_sliding_window_and_scales(self):
        # BF16 dtypes to test unpacking loop in strided_load_bkv and pl.reciprocal
        refs = self._make_kernel_refs(
            q_dtype=np.float16,
            kv_dtype=np.float16,
            actual_num_kv_heads=2,
            actual_num_q_heads=4,
        )
        with (
            patch.object(K64.pltpu, "make_async_copy", return_value=MockAsyncCopy()),
            patch.object(K64.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                K64.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(K64.pl, "when", side_effect=fake_when),
            patch.object(K64.pl, "program_id", return_value=0),
            patch.object(K64.pl, "num_programs", return_value=1),
            patch.object(K64.pl, "debug_print", side_effect=lambda *a, **k: None),
            patch.object(K64.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            # Run with sliding_window, scales, soft_cap, strict_sliding_window=False
            K64._ragged_paged_attention_kernel(
                *refs,
                sm_scale=1.0,
                sliding_window=8,
                strict_sliding_window=False,
                soft_cap=20.0,
                mask_value=-1e9,
                q_scale=1.2,
                k_scale=0.9,
                v_scale=1.1,
                skip_kv_update=True,
                chunk_prefill_size=2,
                bkv_p=2,
                bq_sz=2,
                use_causal_mask=False,
                debug_mode=False,
            )

    def test_kernel_simulation_no_attention_sink_and_strict_sliding(self):
        refs = list(self._make_kernel_refs())
        refs[10] = None  # attention_sink_ref = None
        with (
            patch.object(K64.pltpu, "make_async_copy", return_value=MockAsyncCopy()),
            patch.object(K64.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                K64.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(K64.pl, "when", side_effect=fake_when),
            patch.object(K64.pl, "program_id", return_value=0),
            patch.object(K64.pl, "num_programs", return_value=1),
            patch.object(K64.pl, "debug_print", side_effect=lambda *a, **k: None),
            patch.object(K64.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            K64._ragged_paged_attention_kernel(
                *refs,
                sm_scale=1.0,
                sliding_window=4,
                strict_sliding_window=True,
                soft_cap=None,
                mask_value=-1e9,
                q_scale=None,
                k_scale=None,
                v_scale=None,
                skip_kv_update=False,
                chunk_prefill_size=2,
                bkv_p=2,
                bq_sz=2,
                use_causal_mask=True,
                debug_mode=False,
            )

    def test_kernel_simulation_prefill_mixed_and_prefetch(self):
        refs = self._make_kernel_refs(
            max_num_seqs=4,
            pages_per_seq=2,
            actual_num_kv_heads=2,
            actual_num_q_heads=4,
        )
        distribution_ref = MockArrayRef(np.array([1, 2, 3], dtype=np.int32))
        refs_list = list(refs)
        refs_list[3] = distribution_ref

        with (
            patch.object(K64.pltpu, "make_async_copy", return_value=MockAsyncCopy()),
            patch.object(K64.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                K64.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(K64.pl, "when", side_effect=fake_when),
            patch.object(K64.pl, "num_programs", return_value=3),
            patch.object(K64.pl, "debug_print", side_effect=lambda *a, **k: None),
            patch.object(K64.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            # 1. seq_idx = 1 -> prefill (990) and prefetch next bq/bkv (842, 861)
            with patch.object(K64.pl, "program_id", return_value=1):
                K64._ragged_paged_attention_kernel(
                    *refs_list,
                    sm_scale=1.0,
                    sliding_window=None,
                    strict_sliding_window=True,
                    soft_cap=None,
                    mask_value=-1e9,
                    q_scale=None,
                    k_scale=None,
                    v_scale=None,
                    skip_kv_update=False,
                    chunk_prefill_size=2,
                    bkv_p=2,
                    bq_sz=2,
                    use_causal_mask=True,
                    debug_mode=False,
                )

            # 2. seq_idx = 2 -> mixed (994), static_q_len None (798-799), epilogue
            with patch.object(K64.pl, "program_id", return_value=2):
                K64._ragged_paged_attention_kernel(
                    *refs_list,
                    sm_scale=1.0,
                    sliding_window=None,
                    strict_sliding_window=True,
                    soft_cap=None,
                    mask_value=-1e9,
                    q_scale=None,
                    k_scale=None,
                    v_scale=None,
                    skip_kv_update=False,
                    chunk_prefill_size=2,
                    bkv_p=2,
                    bq_sz=2,
                    use_causal_mask=True,
                    debug_mode=False,
                )

    def test_kernel_simulation_head_break_and_int_kv(self):
        # 1. actual_num_kv_heads=1 with kv_packing=2 (float16) triggers line 903 (break)
        refs_break = self._make_kernel_refs(
            q_dtype=np.float32,
            kv_dtype=np.float16,
            bkv_p=2,
            bq_sz=2,
            max_num_seqs=2,
            pages_per_seq=2,
            actual_num_kv_heads=1,
            actual_num_q_heads=2,
        )
        # 2. kv_dtype=int32 with q_scale triggers branch 407->412
        refs_int = self._make_kernel_refs(
            q_dtype=np.float32,
            kv_dtype=np.int32,
            bkv_p=2,
            bq_sz=2,
            max_num_seqs=2,
            pages_per_seq=2,
            actual_num_kv_heads=2,
            actual_num_q_heads=4,
        )

        with (
            patch.object(K64.pltpu, "make_async_copy", return_value=MockAsyncCopy()),
            patch.object(K64.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                K64.pl, "reciprocal", side_effect=lambda x, approx=True: 1.0 / x
            ),
            patch.object(K64.pl, "when", side_effect=fake_when),
            patch.object(K64.pl, "program_id", return_value=0),
            patch.object(K64.pl, "num_programs", return_value=1),
            patch.object(K64.pl, "debug_print", side_effect=lambda *a, **k: None),
            patch.object(K64.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            # Run 1: line 903 break
            K64._ragged_paged_attention_kernel(
                *refs_break,
                sm_scale=1.0,
                sliding_window=None,
                strict_sliding_window=True,
                soft_cap=None,
                mask_value=-1e9,
                q_scale=None,
                k_scale=None,
                v_scale=None,
                skip_kv_update=False,
                chunk_prefill_size=2,
                bkv_p=2,
                bq_sz=2,
                use_causal_mask=True,
                debug_mode=False,
            )
            # Run 2: branch 407->412 int kv dtype
            K64._ragged_paged_attention_kernel(
                *refs_int,
                sm_scale=1.0,
                sliding_window=None,
                strict_sliding_window=True,
                soft_cap=None,
                mask_value=-1e9,
                q_scale=1.2,
                k_scale=None,
                v_scale=None,
                skip_kv_update=False,
                chunk_prefill_size=2,
                bkv_p=2,
                bq_sz=2,
                use_causal_mask=True,
                debug_mode=False,
            )

    def test_ragged_paged_attention_hd64_outer_call(self, monkeypatch):
        q = jnp.ones((4, 8, 64), dtype=jnp.bfloat16)
        k = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        v = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        kv_cache = jnp.ones((8, 128, 1, 2, 128), dtype=jnp.bfloat16)
        kv_lens = jnp.array([16, 16], dtype=jnp.int32)
        page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4], dtype=jnp.int32)
        distribution = jnp.array([1, 2, 2], dtype=jnp.int32)
        sink = jnp.ones((8,), dtype=jnp.float32)

        def mock_pallas_call(kernel_fn, **kwargs):
            def runner(*args):
                # Return dummy output of shape matching q_prep and updated_kv_cache
                out_shape = (2, 4, 2, 2, 128)
                return jnp.ones(out_shape, dtype=jnp.bfloat16), kv_cache

            return runner

        monkeypatch.setattr(K64.pl, "pallas_call", mock_pallas_call)
        monkeypatch.setattr(TBS64, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(TBS64, "get_device_name", lambda: "TPU v7")

        # 1. Automatic tuning (bq_sz=None) + sink
        out1, c1 = K64.ragged_paged_attention_hd64(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sink=sink,
        )
        assert out1.shape == (4, 8, 64)

        # 2. Manual block sizes and vmem limit specified, no sink
        out2, c2 = K64.ragged_paged_attention_hd64(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sink=None,
            num_kv_pages_per_block=2,
            num_queries_per_block=2,
            vmem_limit_bytes=50 * 1024 * 1024,
        )
        assert out2.shape == (4, 8, 64)


# ==============================================================================
# 6. Tests for deepseek_v41/common/mhc.py & v2/cache.py Donate
# ==============================================================================


class TestCommonMHCAndCacheDonate:
    """100% full-module line & branch coverage for common/mhc.py and cache donate."""

    def test_pre_gates(self):
        tokens = 2
        mult = 4
        dim = 8
        mixes_count = mult * (mult + 2)
        mixes = jnp.ones((tokens, mixes_count), dtype=jnp.float32)
        sqrsum = jnp.ones((tokens, 1), dtype=jnp.float32)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes_count, dtype=jnp.float32)

        pre, post, comb = mhc._pre_gates(
            mixes,
            sqrsum,
            mult,
            dim,
            scale,
            base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=3,
        )
        assert pre.shape == (tokens, mult)
        assert post.shape == (tokens, mult)
        assert comb.shape == (tokens, mult, mult)

    def test_delayed_pre_jax_bfloat16_and_float32(self):
        tokens = 2
        mult = 4
        dim = 8
        mixes = mult * (mult + 2)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 3)

        res = jnp.ones((tokens, mult, dim), dtype=jnp.bfloat16)
        x = jnp.ones((tokens, mult * dim), dtype=jnp.bfloat16)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes, dtype=jnp.float32)
        prev = jnp.ones((tokens, mult), dtype=jnp.float32)

        # 1. FP32 weight path
        fn_f32 = jnp.ones((mixes, mult * dim), dtype=jnp.float32)
        post, comb, val, next_pre = mhc.delayed_pre_jax(
            res, x, fn_f32, scale, base, prev, *constants
        )
        assert post.shape == (tokens, mult, 1)
        assert comb.shape == (tokens, mult, mult)
        assert val.shape == (tokens, dim)
        assert next_pre.shape == (tokens, mult)

        # 2. Packed BF16 weight path
        fn_bf16 = mhc._pack_mix_weight(fn_f32)
        post2, comb2, val2, next_pre2 = mhc.delayed_pre_jax(
            res, x, fn_bf16, scale, base, prev, *constants
        )
        assert post2.shape == (tokens, mult, 1)

    def test_pack_mix_weights_torch(self):
        # 1. pack_mix_weight
        fn = torch.randn(24, 32, dtype=torch.float32)
        packed = mhc.pack_mix_weight(fn)
        assert packed.shape == (32, 128)
        assert packed.dtype == torch.bfloat16

        # 2. pack_mix_weights across mock layers
        # Case A: TPU device and float32 -> packed Parameter
        mock_tpu_fn = MagicMock()
        mock_tpu_fn.device.type = "tpu"
        mock_tpu_fn.dtype = torch.float32

        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.common.mhc.pack_mix_weight"
        ) as mock_pmw:
            mock_pmw.return_value = torch.zeros(32, 128, dtype=torch.bfloat16)
            layer1 = SimpleNamespace(hc_attn_fn=mock_tpu_fn, hc_ffn_fn=mock_tpu_fn)
            mhc.pack_mix_weights([layer1])
            assert isinstance(layer1.hc_attn_fn, torch.nn.Parameter)
            assert isinstance(layer1.hc_ffn_fn, torch.nn.Parameter)
            assert layer1.hc_attn_fn.shape == (32, 128)

        # Case B: CPU device -> no-op
        mock_cpu_fn = MagicMock()
        mock_cpu_fn.device = SimpleNamespace(type="cpu")
        mock_cpu_fn.dtype = torch.float32
        layer2 = SimpleNamespace(hc_attn_fn=mock_cpu_fn, hc_ffn_fn=mock_cpu_fn)
        mhc.pack_mix_weights([layer2])
        assert layer2.hc_attn_fn is mock_cpu_fn

        # Case C: Non-float32 dtype -> no-op
        mock_bf16_fn = MagicMock()
        mock_bf16_fn.device = SimpleNamespace(type="tpu")
        mock_bf16_fn.dtype = torch.bfloat16
        layer3 = SimpleNamespace(hc_attn_fn=mock_bf16_fn, hc_ffn_fn=mock_bf16_fn)
        mhc.pack_mix_weights([layer3])
        assert layer3.hc_attn_fn is mock_bf16_fn

    def test_delayed_pre_flat_jax_pallas_and_xla_branches(self, monkeypatch):
        mult = 4
        dim = 8
        mixes = mult * (mult + 2)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 3)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes, dtype=jnp.float32)
        fn_f32 = jnp.ones((mixes, mult * dim), dtype=jnp.float32)

        # 1. XLA fallback: tokens < 64
        flat_small = jnp.ones((16, mult * dim), dtype=jnp.bfloat16)
        prev_small = jnp.ones((16, mult), dtype=jnp.float32)
        p1, c1, v1, np1 = mhc.delayed_pre_flat_jax(
            flat_small, fn_f32, scale, base, prev_small, *constants
        )
        assert p1.shape == (16, mult, 1)

        # 2. XLA fallback: tokens % 64 != 0
        flat_odd = jnp.ones((65, mult * dim), dtype=jnp.bfloat16)
        prev_odd = jnp.ones((65, mult), dtype=jnp.float32)
        p2, c2, v2, np2 = mhc.delayed_pre_flat_jax(
            flat_odd, fn_f32, scale, base, prev_odd, *constants
        )
        assert p2.shape == (65, mult, 1)

        # 3. XLA fallback: 3 * mixes > 128 (e.g. mult = 8 -> mixes = 80 -> 240 > 128)
        mult_big = 8
        mixes_big = mult_big * (mult_big + 2)
        flat_big = jnp.ones((64, mult_big * dim), dtype=jnp.bfloat16)
        fn_big = jnp.ones((mixes_big, mult_big * dim), dtype=jnp.float32)
        base_big = jnp.zeros(mixes_big, dtype=jnp.float32)
        prev_big = jnp.ones((64, mult_big), dtype=jnp.float32)
        p3, c3, v3, np3 = mhc.delayed_pre_flat_jax(
            flat_big, fn_big, scale, base_big, prev_big, *constants
        )
        assert p3.shape == (64, mult_big, 1)

        # 4. Fused Pallas path with mock_pallas_call
        def mock_pallas_call(kernel, grid, in_specs, out_specs, out_shape, **kwargs):
            for spec in in_specs + out_specs:
                if spec is not None and callable(getattr(spec, "index_map", None)):
                    spec.index_map(0)

            def invoke(flat_in, weight_in, prev_in):
                t_count, w_count = flat_in.shape
                tile_sz = in_specs[0].block_shape[0]
                m_c = prev_in.shape[1]
                d_c = w_count // m_c
                mx_c = m_c * (m_c + 2)
                mix_out = []
                sqr_out = []
                val_out = []
                for i in range(grid[0]):
                    sl = slice(i * tile_sz, (i + 1) * tile_sz)
                    x_ref = MockArrayRef(flat_in[sl])
                    w_ref = MockArrayRef(weight_in)
                    pre_ref = MockArrayRef(prev_in[sl])
                    mix_ref = MockArrayRef(np.zeros((tile_sz, mx_c), dtype=np.float32))
                    sqr_ref = MockArrayRef(np.zeros((tile_sz, 1), dtype=np.float32))
                    val_ref = MockArrayRef(np.zeros((tile_sz, d_c), dtype=np.float32))
                    kernel(x_ref, w_ref, pre_ref, mix_ref, sqr_ref, val_ref)
                    mix_out.append(mix_ref.arr)
                    sqr_out.append(sqr_ref.arr)
                    val_out.append(val_ref.arr)
                return (
                    jnp.concatenate(mix_out, axis=0),
                    jnp.concatenate(sqr_out, axis=0),
                    jnp.concatenate(val_out, axis=0).astype(flat_in.dtype),
                )

            return invoke

        monkeypatch.setattr("jax.experimental.pallas.pallas_call", mock_pallas_call)

        # Case A: tokens = 128 (tile = 128), fn is BF16
        flat_128 = jnp.ones((128, mult * dim), dtype=jnp.bfloat16)
        prev_128 = jnp.ones((128, mult), dtype=jnp.float32)
        fn_bf16 = mhc._pack_mix_weight(fn_f32)
        p128, c128, v128, np128 = mhc.delayed_pre_flat_jax(
            flat_128, fn_bf16, scale, base, prev_128, *constants
        )
        assert p128.shape == (128, mult, 1)

        # Case B: tokens = 64 (tile = 64), fn is FP32
        flat_64 = jnp.ones((64, mult * dim), dtype=jnp.bfloat16)
        prev_64 = jnp.ones((64, mult), dtype=jnp.float32)
        p64, c64, v64, np64 = mhc.delayed_pre_flat_jax(
            flat_64, fn_f32, scale, base, prev_64, *constants
        )
        assert p64.shape == (64, mult, 1)

    def test_post_jax_and_delayed_post_pre_flat_jax(self, monkeypatch):
        tokens = 64
        mult = 4
        dim = 8
        mixes = mult * (mult + 2)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 3)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes, dtype=jnp.float32)
        fn_f32 = jnp.ones((mixes, mult * dim), dtype=jnp.float32)

        x_sub = jnp.ones((tokens, dim), dtype=jnp.bfloat16)
        flat = jnp.ones((tokens, mult * dim), dtype=jnp.bfloat16)
        post_mix = jnp.ones((tokens, mult, 1), dtype=jnp.float32)
        comb_mix = jnp.ones((tokens, mult, mult), dtype=jnp.float32)
        prev = jnp.ones((tokens, mult), dtype=jnp.float32)

        # 1. Direct _post_jax check
        post_out = mhc._post_jax(x_sub, flat, post_mix, comb_mix)
        assert post_out.shape == (tokens, mult * dim)

        # 2. delayed_post_pre_flat_jax XLA fallback path (tokens < 64)
        x_small = jnp.ones((16, dim), dtype=jnp.bfloat16)
        flat_small = jnp.ones((16, mult * dim), dtype=jnp.bfloat16)
        f_s, p_s, c_s, v_s, np_s = mhc.delayed_post_pre_flat_jax(
            x_small,
            flat_small,
            post_mix[:16],
            comb_mix[:16],
            fn_f32,
            scale,
            base,
            prev[:16],
            *constants,
        )
        assert f_s.shape == (16, mult * dim)
        assert p_s.shape == (16, mult, 1)

        # 3. delayed_post_pre_flat_jax fused Pallas path with mock_pallas_call
        def mock_fused_pallas_call(
            kernel, grid, in_specs, out_specs, out_shape, **kwargs
        ):
            for spec in in_specs + out_specs:
                if spec is not None and callable(getattr(spec, "index_map", None)):
                    spec.index_map(0)

            def invoke(sub_in, flat_in, post_in, comb_in, weight_in, prev_in):
                t_count, w_count = flat_in.shape
                tile_sz = in_specs[0].block_shape[0]
                m_c = prev_in.shape[1]
                d_c = w_count // m_c
                mx_c = m_c * (m_c + 2)
                res_out = []
                mix_out = []
                sqr_out = []
                val_out = []
                for i in range(grid[0]):
                    sl = slice(i * tile_sz, (i + 1) * tile_sz)
                    sub_x_ref = MockArrayRef(sub_in[sl])
                    res_ref = MockArrayRef(flat_in[sl])
                    post_ref = MockArrayRef(post_in[sl])
                    comb_ref = MockArrayRef(comb_in[sl])
                    w_ref = MockArrayRef(weight_in)
                    pre_ref = MockArrayRef(prev_in[sl])
                    out_res_ref = MockArrayRef(
                        np.zeros((tile_sz, w_count), dtype=np.float32)
                    )
                    mix_ref = MockArrayRef(np.zeros((tile_sz, mx_c), dtype=np.float32))
                    sqr_ref = MockArrayRef(np.zeros((tile_sz, 1), dtype=np.float32))
                    val_ref = MockArrayRef(np.zeros((tile_sz, d_c), dtype=np.float32))
                    kernel(
                        sub_x_ref,
                        res_ref,
                        post_ref,
                        comb_ref,
                        w_ref,
                        pre_ref,
                        out_res_ref,
                        mix_ref,
                        sqr_ref,
                        val_ref,
                    )
                    res_out.append(out_res_ref.arr)
                    mix_out.append(mix_ref.arr)
                    sqr_out.append(sqr_ref.arr)
                    val_out.append(val_ref.arr)
                return (
                    jnp.concatenate(res_out, axis=0).astype(flat_in.dtype),
                    jnp.concatenate(mix_out, axis=0),
                    jnp.concatenate(sqr_out, axis=0),
                    jnp.concatenate(val_out, axis=0).astype(flat_in.dtype),
                )

            return invoke

        monkeypatch.setattr(
            "jax.experimental.pallas.pallas_call", mock_fused_pallas_call
        )
        f_fused, p_fused, c_fused, v_fused, np_fused = mhc.delayed_post_pre_flat_jax(
            x_sub,
            flat,
            post_mix,
            comb_mix,
            fn_f32,
            scale,
            base,
            prev,
            *constants,
        )
        assert f_fused.shape == (tokens, mult * dim)
        assert p_fused.shape == (tokens, mult, 1)

    def test_initialize_delayed_mhc_and_dispatch(self, monkeypatch):
        monkeypatch.setattr(mhc, "_pre_op", None)
        monkeypatch.setattr(mhc, "_pre_flat_op", None)
        monkeypatch.setattr(mhc, "_post_pre_flat_op", None)

        mock_ctx = SimpleNamespace(mesh=None)
        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
            lambda: mock_ctx,
        )

        class FakeOp:
            def __init__(self, name, fn, *a, **kw):
                self.name = name
                self.fn = fn
                self.fake_fn = None

            def register_fake(self, fn):
                self.fake_fn = fn

            def __call__(self, *args, **kwargs):
                return self.fake_fn(*args, **kwargs)

        monkeypatch.setattr(
            "torch_tpu._internal.pallas.jax_op",
            lambda name, fn, *a, **kw: FakeOp(name, fn, *a, **kw),
        )

        # 1. Initialize delayed mhc
        mhc.initialize_delayed_mhc()
        assert mhc._pre_op is not None
        assert mhc._pre_flat_op is not None
        assert mhc._post_pre_flat_op is not None

        # 2. Idempotency branch
        cached_op = mhc._pre_op
        mhc.initialize_delayed_mhc()
        assert mhc._pre_op is cached_op

        # 3. Test registered fake functions
        residual = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
        x = torch.zeros((2, 32), dtype=torch.bfloat16)
        sub_x = torch.zeros((2, 8), dtype=torch.bfloat16)
        post_mix = torch.zeros((2, 4, 1), dtype=torch.float32)
        comb_mix = torch.zeros((2, 4, 4), dtype=torch.float32)
        fn = torch.zeros((24, 32), dtype=torch.float32)
        scale = torch.zeros(3, dtype=torch.float32)
        base = torch.zeros(24, dtype=torch.float32)
        prev = torch.zeros((2, 4), dtype=torch.float32)

        out1 = mhc._pre_op(residual, x, fn, scale, base, prev)
        assert len(out1) == 4
        out2 = mhc._pre_flat_op(x, fn, scale, base, prev)
        assert len(out2) == 4
        out3 = mhc._post_pre_flat_op(
            sub_x, x, post_mix, comb_mix, fn, scale, base, prev
        )
        assert len(out3) == 5

        # 4. delayed_pre: x is None vs x is not None
        res_flat = mhc.delayed_pre(residual, None, fn, scale, base, prev)
        assert len(res_flat) == 4
        res_x = mhc.delayed_pre(residual, x, fn, scale, base, prev)
        assert len(res_x) == 4

        # 5. delayed_post_pre
        res_post_pre = mhc.delayed_post_pre(
            sub_x, residual, post_mix, comb_mix, fn, scale, base, prev
        )
        assert res_post_pre[0].shape == residual.shape

    def test_pre_post_shifted_and_cache_donate(self, monkeypatch):
        # 1. collapse & _default_pre_mix
        x = torch.ones((2, 4, 8), dtype=torch.bfloat16)
        pre_mix = torch.ones((2, 4), dtype=torch.float32)
        col = mhc.collapse(x, pre_mix)
        assert col.shape == (2, 8)

        def_pre = mhc._default_pre_mix(x, torch.float32)
        assert def_pre.shape == (2, 4)

        # 2. post op & dispatch
        mock_post_op = MagicMock(return_value=torch.ones_like(x))
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op.get_mhc_post_op",
            lambda: mock_post_op,
        )
        mhc.initialize_mhc_post()
        assert mhc._post_op is mock_post_op

        # TPU device branch for post
        x_sub = torch.ones((2, 8), dtype=torch.bfloat16)
        post_m = torch.ones((2, 4, 1), dtype=torch.float32)
        comb_m = torch.ones((2, 4, 4), dtype=torch.float32)

        mock_tpu_res = MagicMock()
        mock_tpu_res.device.type = "tpu"
        res_tpu = mhc.post(x_sub, mock_tpu_res, post_m, comb_m)
        assert res_tpu is mock_post_op.return_value

        # CPU device branch for post
        post_res_cpu = mhc.post(x_sub, x, post_m, comb_m)
        assert post_res_cpu.shape == x.shape

        # 3. pre() dispatch: TPU vs CPU, norm_weight
        fn = torch.zeros((24, 32), dtype=torch.float32)
        scale = torch.zeros(3, dtype=torch.float32)
        base = torch.zeros(24, dtype=torch.float32)
        norm_w = torch.ones(8, dtype=torch.bfloat16)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 3)

        mock_delayed_pre = MagicMock(
            return_value=(
                post_m,
                comb_m,
                torch.zeros((2, 8), dtype=torch.bfloat16),
                pre_mix,
            )
        )
        monkeypatch.setattr(mhc, "delayed_pre", mock_delayed_pre)

        real_arange = torch.arange
        monkeypatch.setattr(mhc.torch, "arange", lambda n, **kw: real_arange(n))
        mock_tpu_res.shape = (2, 4, 8)
        mock_tpu_res.__getitem__ = MagicMock(return_value=torch.zeros(2, 4))

        # TPU path with pre_mix=None
        out_tpu1 = mhc.pre(
            mock_tpu_res, fn, scale, base, *constants, pre_mix=None, norm_weight=norm_w
        )
        assert len(out_tpu1) == 4
        mock_delayed_pre.assert_called_once()

        # TPU path with pre_mix provided
        mock_delayed_pre.reset_mock()
        out_tpu2 = mhc.pre(
            mock_tpu_res, fn, scale, base, *constants, pre_mix=pre_mix, norm_weight=None
        )
        assert len(out_tpu2) == 4

        # CPU branch for pre()
        out_cpu = mhc.pre(
            x, fn, scale, base, *constants, pre_mix=pre_mix, norm_weight=norm_w
        )
        assert len(out_cpu) == 4

        # 4. shifted_post_pre dispatch: stream error, reduce_results, TPU, CPU
        with pytest.raises(ValueError, match="TPU mHC does not use CUDA streams"):
            mhc.shifted_post_pre(
                x_sub, x, post_m, comb_m, fn, scale, base, *constants, stream="err"
            )

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.common.mhc.tensor_model_parallel_all_reduce",
            lambda t: t,
        )

        mock_post_pre_return = (
            x,
            post_m,
            comb_m,
            torch.zeros((2, 8), dtype=torch.bfloat16),
            pre_mix,
        )
        monkeypatch.setattr(
            mhc, "delayed_post_pre", lambda *a, **k: mock_post_pre_return
        )
        monkeypatch.setattr(mhc, "_post_pre_flat_op", MagicMock())

        # TPU branch with capture_aux=True and norm_weight
        r_tpu, p_tpu, c_tpu, v_tpu, np_tpu, aux_tpu = mhc.shifted_post_pre(
            x_sub,
            mock_tpu_res,
            post_m,
            comb_m,
            fn,
            scale,
            base,
            *constants,
            pre_mix=None,
            norm_weight=norm_w,
            capture_aux=True,
            reduce_results=True,
        )
        assert aux_tpu.shape == (2, 8)

        # TPU branch with capture_aux=False and norm_weight=None
        r_tpu2, p_tpu2, c_tpu2, v_tpu2, np_tpu2, aux_tpu2 = mhc.shifted_post_pre(
            x_sub,
            mock_tpu_res,
            post_m,
            comb_m,
            fn,
            scale,
            base,
            *constants,
            pre_mix=pre_mix,
            norm_weight=None,
            capture_aux=False,
            reduce_results=False,
        )
        assert aux_tpu2.shape == (0, 8)

        # CPU branch fallback (or when _post_pre_flat_op is None)
        monkeypatch.setattr(mhc, "_post_pre_flat_op", None)
        r_cpu, p_cpu, c_cpu, v_cpu, np_cpu, aux_cpu = mhc.shifted_post_pre(
            x_sub,
            x,
            post_m,
            comb_m,
            fn,
            scale,
            base,
            *constants,
            pre_mix=pre_mix,
            capture_aux=True,
        )
        assert aux_cpu.shape == (2, 8)

        # 5. v2/cache.py _write_op donate_argnums=(0,) verification
        mock_cache_ctx = SimpleNamespace(mesh=None)
        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
            lambda: mock_cache_ctx,
        )
        captured_jax_ops = {}

        class CacheFakeOp:
            def __init__(self, name, fn, *a, **kw):
                self.name = name
                self.fn = fn
                self.fake_fn = None

            def register_fake(self, fn):
                self.fake_fn = fn

            def __call__(self, *args, **kwargs):
                return self.fake_fn(*args, **kwargs)

        def mock_cache_jax_op(name, fn, *a, **kwargs):
            op = CacheFakeOp(name, fn, *a, **kwargs)
            captured_jax_ops[name] = kwargs
            return op

        monkeypatch.setattr(
            "torch_tpu._internal.pallas.jax_op",
            mock_cache_jax_op,
        )
        monkeypatch.setattr("jax.shard_map", lambda fn, *a, **k: fn)
        monkeypatch.setattr(v2_cache, "_write_op", None)

        v2_cache.initialize_cache_ops()
        assert "pallas::dsv41v2_cache_write" in captured_jax_ops
        assert captured_jax_ops["pallas::dsv41v2_cache_write"].get(
            "donate_argnums"
        ) == (0,)

        # Test _scatter_rows logic with valid slots and drop slots (< 0)
        cache_arr = jnp.zeros((4, 8), dtype=jnp.float32)
        slots = jnp.array([1, -1, 3], dtype=jnp.int32)
        values = jnp.ones((3, 8), dtype=jnp.float32)
        scat = v2_cache._scatter_rows(cache_arr, slots, values)
        assert scat.shape == (4, 8)
        assert float(scat[1, 0]) == 1.0
        assert float(scat[3, 0]) == 1.0
        assert float(scat[0, 0]) == 0.0
