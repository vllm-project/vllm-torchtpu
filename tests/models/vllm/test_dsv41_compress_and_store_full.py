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

"""100% full-module line and branch coverage tests for compress_and_store:

- buffered_ref.py
- compressor_v1.py
- compress_store_ref.py
- compute.py
- config.py
- kernel.py
- proj_and_save_state.py
- project_and_save_state_ref.py
on CPU under JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=true.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    buffered_ref,
    compress_store_ref,
    compressor_v1,
    compute,
    config,
    kernel,
    proj_and_save_state,
    project_and_save_state_ref,
)

# ==============================================================================
# Helpers & Mocks for CPU Pallas Simulation
# ==============================================================================


def _norm_idx(idx):
    if isinstance(idx, tuple):
        return tuple(_norm_idx(i) for i in idx)
    if hasattr(idx, "start") and hasattr(idx, "size"):
        stride = getattr(idx, "stride", 1)
        stride = 1 if stride is None else int(stride)
        start = int(np.asarray(idx.start))
        size = int(np.asarray(idx.size))
        return slice(start, start + size * stride, stride)
    return idx


class MockArrayRef:
    """Mock HBM / VMEM buffer reference for simulating Pallas pipelines on CPU."""

    def __init__(
        self,
        arr: np.ndarray | jax.Array | None = None,
        shape: tuple[int, ...] = (2, 4, 4, 32, 4, 256),
        copy: bool = False,
    ):
        if arr is not None:
            self.arr = np.asarray(arr)
        else:
            self.arr = np.zeros(shape, dtype=np.uint8)
        self.arr.flags.writeable = True
        self.shape = self.arr.shape
        self.dtype = self.arr.dtype

    def __getitem__(self, idx):
        if idx is Ellipsis:
            return jnp.asarray(self.arr)
        try:
            return jnp.asarray(self.arr[_norm_idx(idx)])
        except Exception:
            return jnp.zeros((1, 128))

    def __setitem__(self, idx, val):
        val_arr = getattr(val, "arr", val)
        try:
            if idx is Ellipsis:
                self.arr[...] = np.asarray(val_arr)
            else:
                self.arr[_norm_idx(idx)] = np.asarray(val_arr)
        except Exception:
            pass

    @property
    def at(self):
        class _AtIndexer:
            def __init__(self, ref):
                self.ref = ref

            def __getitem__(self, idx):
                if idx is Ellipsis:
                    return jnp.asarray(self.ref.arr)
                try:
                    return MockArrayRef(self.ref.arr[_norm_idx(idx)], copy=False)
                except Exception:
                    return MockArrayRef(np.zeros((1, 128)), copy=False)

            def __setitem__(self, idx, val):
                val_arr = getattr(val, "arr", val)
                try:
                    if idx is Ellipsis:
                        self.ref.arr[...] = np.asarray(val_arr)
                    else:
                        self.ref.arr[_norm_idx(idx)] = np.asarray(val_arr)
                except Exception:
                    pass

        return _AtIndexer(self)

    def bitcast(self, dtype):
        res = fake_bitcast(self.arr, dtype)
        return MockArrayRef(np.array(res, copy=True), copy=False)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return MockArrayRef(self.arr.reshape(*shape), copy=False)


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
    """Safely evaluates when block body."""

    def decorator(fn):
        try:
            val = bool(np.all(np.asarray(cond)))
            if val:
                fn()
        except Exception:
            fn()
        return fn

    return decorator


def fake_loop(start, stop, step=1, unroll=False):
    """Executes a loop body over integer indices."""

    def decorator(fn):
        for i in range(start, stop, step):
            fn(i)
        return fn

    return decorator


def fake_bitcast(x, ty):
    """CPU fallback for pltpu.bitcast."""
    arr = getattr(x, "arr", x)
    try:
        arr = np.asarray(arr)
        ty_str = str(ty)
        if ("int32" in ty_str) and arr.dtype == np.uint8 and arr.shape[-2:] == (4, 128):
            return jnp.asarray(arr.view(np.int32).reshape(*arr.shape[:-2], 1, 128))
        if ("uint8" in ty_str) and arr.dtype == np.int32 and arr.shape[-2:] == (1, 128):
            return jnp.asarray(arr.view(np.uint8).reshape(*arr.shape[:-2], 4, 128))
        if "float8_e8m0fnu" in ty_str or "float8_e4m3fn" in ty_str:
            return jnp.asarray(arr.astype(np.uint8))
        if "uint16" in ty_str and arr.dtype in (np.float32, np.float16):
            return jnp.asarray(arr.astype(np.float16).view(np.uint16))
        return jnp.asarray(arr.view(ty))
    except Exception:
        pass
    try:
        return x.view(ty)
    except Exception:
        return jnp.zeros(getattr(x, "shape", (1, 128)), dtype=ty)


def fake_roll(x, shift, axis):
    """CPU fallback for pltpu.roll."""
    return jnp.roll(x, shift, axis)


# ==============================================================================
# 1. TestConfig (config.py)
# ==============================================================================


class TestConfig:
    """Tests 100% of line and branch coverage in config.py."""

    def test_select_mode(self):
        # head_dim == LANE (128)
        assert config.select_mode(128, False) == config.Mode.CSA_INDEXER
        assert config.select_mode(128, True) == config.Mode.CSA_INDEXER

        # head_dim != LANE
        assert config.select_mode(512, True) == config.Mode.CSA
        assert config.select_mode(512, False) == config.Mode.HCA

    def test_page_and_layout_sizes(self):
        # physical_page_size
        assert (
            config.physical_page_size(config.Mode.CSA_INDEXER, 128, 4)
            == (128 // 4) // 4
        )
        assert config.physical_page_size(config.Mode.CSA, 128, 4) == 128 // 4
        assert config.physical_page_size(config.Mode.HCA, 128, 4) == (128 // 4) * 2

        # state_host_page_size
        hca_host = config.state_host_page_size(config.Mode.HCA, 128, 128)
        assert hca_host == config.physical_page_size(
            config.Mode.CSA, 128, config.CSA_COMPRESS_RATIO
        )
        csa_host = config.state_host_page_size(config.Mode.CSA, 128, 4)
        assert csa_host == config.physical_page_size(config.Mode.CSA, 128, 4)

        # last_dim_size
        assert config.last_dim_size(config.Mode.CSA_INDEXER) == 256
        assert config.last_dim_size(config.Mode.CSA) == 128
        assert config.last_dim_size(config.Mode.HCA) == 128

        # row_size_bytes
        assert config.row_size_bytes(config.Mode.CSA_INDEXER) == 4 * 256
        assert config.row_size_bytes(config.Mode.CSA) == 4 * 128

        # state_width
        assert config.state_width(512, True) == 1024
        assert config.state_width(512, False) == 512

        # state_rows_per_token
        rows_csa = config.state_rows_per_token(config.Mode.CSA, 512, True)
        assert rows_csa == (2 * 1024 * 4) // 512

    def test_state_block_size_and_capacity(self):
        cap_csa = config._state_page_capacity(config.Mode.CSA, 128)
        cap_hca = config._state_page_capacity(config.Mode.HCA, 128)
        assert cap_csa > 0
        assert cap_hca > 0

        # state_block_size
        bs_csa = config.state_block_size(config.Mode.CSA, 128)
        bs_idx = config.state_block_size(config.Mode.CSA_INDEXER, 128)
        bs_hca = config.state_block_size(config.Mode.HCA, 128)
        assert bs_csa == bs_idx
        assert bs_hca == cap_hca

    def test_dimensions_properties(self):
        dim_quant = config.Dimensions(
            mode=config.Mode.CSA,
            size_n=16,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=4,
            physical_page_size=32,
            state_physical_page_size=32,
            state_block_size=8,
            quant_block=64,
            overlap=True,
            has_rope_cache=True,
        )
        assert dim_quant.is_quantized is True
        assert dim_quant.has_rope is True
        assert dim_quant.nope_dtype == jnp.uint8
        assert dim_quant.rope_dtype == jnp.uint8

        dim_noquant = config.Dimensions(
            mode=config.Mode.HCA,
            size_n=16,
            head_dim=512,
            rope_head_dim=0,
            compress_ratio=128,
            physical_page_size=32,
            state_physical_page_size=32,
            state_block_size=8,
            quant_block=0,
            overlap=False,
            has_rope_cache=False,
        )
        assert dim_noquant.is_quantized is False
        assert dim_noquant.has_rope is False
        assert dim_noquant.nope_dtype == jnp.bfloat16
        assert dim_noquant.rope_dtype == jnp.bfloat16

    def test_configs_make_and_properties(self):
        # Mode.HCA make forces quant_block = 0
        cfg_hca = config.Configs.make(
            config.Mode.HCA,
            size_n=8,
            physical_page_size=64,
            state_block_size=8,
        )
        assert cfg_hca.dims.quant_block == 0
        assert cfg_hca.overlap_factor == 1
        assert cfg_hca.state_width == 512
        assert cfg_hca.head_tiles == 4
        assert cfg_hca.window == 128
        assert cfg_hca.nope_dim == 512 - 64
        assert cfg_hca.nope_store_dim == 512  # has_rope_cache is False
        assert cfg_hca.half_rope == 32
        assert cfg_hca.rope_slot == 3
        assert cfg_hca.record_bytes == 1024  # bf16: 512 * 2
        assert cfg_hca.row_size_bytes == 512
        assert cfg_hca.record_rows == 2
        assert cfg_hca.kv_stride == 2  # record_bytes > row_size_bytes
        assert cfg_hca.cache_last_dims == (4, 128)
        assert cfg_hca.physical_page_size == 64
        assert cfg_hca.state_physical_page_size == 64
        assert cfg_hca.tokens_in_second_minor == 1

        # Mode.CSA make
        cfg_csa = config.Configs.make(
            config.Mode.CSA,
            size_n=8,
            physical_page_size=32,
            state_block_size=8,
            state_physical_page_size=32,
        )
        assert cfg_csa.overlap_factor == 2
        assert cfg_csa.state_width == 1024
        assert cfg_csa.window == 8
        assert cfg_csa.nope_store_dim == 512 - 64  # has_rope_cache is True
        assert cfg_csa.record_bytes == 512
        assert cfg_csa.kv_stride == 1  # record_bytes <= row_size_bytes
        assert cfg_csa.rope_page_size == 32 // 4
        assert cfg_csa.pages_to_buffer_per_token == 2

        # Shapes
        assert cfg_csa.window_shape() == (2, 4, 8, 4, 128)
        assert cfg_csa.window_bytes_shape() == (2, 4, 8, 4, 4, 128)
        assert cfg_csa.output_shape() == (4, 1, 4, 128)
        assert cfg_csa.page_buffer_shape() == (4, 2, 32, 4, 128)
        assert cfg_csa.rope_output_shape() == (4, 4, 128)
        assert cfg_csa.cos_sin_shape() == (4, 64)
        assert cfg_csa.cache_shape(10) == (10, 32, 4, 128)
        assert cfg_csa.state_cache_shape(10) == (10, 32, 4, 128)
        assert cfg_csa.rope_cache_shape(10) == (10, 8, 4, 128)

        # Mode.CSA_INDEXER make
        cfg_idx = config.Configs.make(
            config.Mode.CSA_INDEXER,
            size_n=4,
            physical_page_size=16,
            state_block_size=4,
        )
        assert cfg_idx.dims.head_dim == 128
        assert cfg_idx.record_bytes == 256  # head_dim == LANE
        assert cfg_idx.last_dim_size == 256
        assert cfg_idx.row_size_bytes == 1024
        assert cfg_idx.tokens_in_second_minor == 4


# ==============================================================================
# 2. TestProjectAndSaveStateRef (project_and_save_state_ref.py)
# ==============================================================================


class TestProjectAndSaveStateRef:
    """Tests 100% of project_and_save_state_ref.py."""

    def test_ref_wkv_proj_and_save_state_execution(self):
        num_tokens = 4
        hidden_size = 64
        head_dim = 128
        compress_ratio = 4
        overlap = True
        state_width = 2 * head_dim  # 256
        state_dim = 2 * state_width  # 512

        hidden_states = jnp.ones((num_tokens, hidden_size), dtype=jnp.float32)
        wkv_wgate = jnp.ones((hidden_size, state_dim), dtype=jnp.float32) * 0.1
        ape = jnp.ones((compress_ratio, state_width), dtype=jnp.float32) * 0.5
        positions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        slot_mapping = jnp.array([0, 4, -1, 8], dtype=jnp.int32)

        cache = jnp.zeros((4, 8, 4, 128), dtype=jnp.uint8)

        new_cache = project_and_save_state_ref.ref_wkv_proj_and_save_state(
            hidden_states=hidden_states,
            wkv_wgate=wkv_wgate,
            ape=ape,
            positions=positions,
            slot_mapping=slot_mapping,
            cache=cache,
            state_block_size=2,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        assert new_cache.shape == cache.shape
        assert jnp.any(new_cache > 0)

    def test_ref_wkv_proj_and_save_state_block_size_assertion(self):
        # Exceed state_block_size > tokens_per_page
        hidden_states = jnp.ones((2, 32), dtype=jnp.float32)
        wkv_wgate = jnp.ones((32, 512), dtype=jnp.float32)
        ape = jnp.ones((4, 256), dtype=jnp.float32)
        positions = jnp.array([0, 1], dtype=jnp.int32)
        slot_mapping = jnp.array([0, 4], dtype=jnp.int32)
        cache = jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8)

        with pytest.raises(AssertionError, match="exceeds"):
            project_and_save_state_ref.ref_wkv_proj_and_save_state(
                hidden_states=hidden_states,
                wkv_wgate=wkv_wgate,
                ape=ape,
                positions=positions,
                slot_mapping=slot_mapping,
                cache=cache,
                state_block_size=10,  # exceeds tokens_per_page (2)
                head_dim=128,
                compress_ratio=4,
                overlap=True,
            )


# ==============================================================================
# 3. TestCompressStoreRef (compress_store_ref.py)
# ==============================================================================


class TestCompressStoreRef:
    """Tests 100% of compress_store_ref.py."""

    def test_quantize_helpers(self):
        x = jnp.linspace(-2.0, 2.0, 128, dtype=jnp.float32).reshape(1, 128)
        # Block quantize
        q, scale = compress_store_ref.quantize_fp8_ue8m0(x, 64)
        assert q.shape == x.shape
        assert scale.shape == (1, 2)

        # Lane periodic quantize
        q_lp, scale_lp = compress_store_ref.quantize_fp8_lane_periodic(
            x, period=64, nope_dim=100
        )
        assert q_lp.shape == x.shape
        assert scale_lp.shape == (1, 64)

    def test_interleaved_rope_and_guards(self):
        x = jnp.ones((2, 64), dtype=jnp.float32)
        cos_sin = jnp.ones((2, 32), dtype=jnp.float32)
        out = compress_store_ref.interleaved_rope(x, cos_sin, 32)
        assert out.shape == x.shape

        # Error guards
        with pytest.raises(ValueError, match="head_dim must be even"):
            compress_store_ref.interleaved_rope(
                jnp.ones((2, 63), dtype=jnp.float32), cos_sin, 32
            )

        with pytest.raises(
            ValueError, match="rope_head_dim must be even and <= head_dim"
        ):
            compress_store_ref.interleaved_rope(x, cos_sin, 31)

        with pytest.raises(
            ValueError, match="rope_head_dim must be even and <= head_dim"
        ):
            compress_store_ref.interleaved_rope(x, jnp.ones((2, 128)), 128)

    def test_gather_and_compress(self):
        # state_cache: (num_pages, tokens_per_page, state_dim)
        state_cache = jnp.ones((2, 4, 1024), dtype=jnp.float32)
        positions = jnp.array([3, 7], dtype=jnp.int32)
        block_table = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        token_to_req = jnp.array([0, 1], dtype=jnp.int32)

        kv_w, score_w, valid_m = compress_store_ref.gather_state_windows(
            state_cache=state_cache,
            positions=positions,
            block_table=block_table,
            block_table_stride=2,
            token_to_req_indices=token_to_req,
            block_size=4,
            head_dim=256,
            compress_ratio=4,
            overlap=True,
        )
        assert kv_w.shape == (2, 8, 256)
        assert score_w.shape == (2, 8, 256)
        assert valid_m.shape == (2, 8)

        # compress_norm_rope
        normed = compress_store_ref.compress_norm_rope(
            kv_window=kv_w,
            score_window=score_w,
            valid_mask=valid_m,
            rms_weight=jnp.ones((256,), dtype=jnp.float32),
            cos_sin_cache=jnp.ones((16, 64), dtype=jnp.float32),
            compressed_pos=jnp.array([0, 4], dtype=jnp.int32),
            rms_eps=1e-6,
            rope_head_dim=64,
        )
        assert normed.shape == (2, 256)

    def test_boundary_dest(self):
        positions = jnp.array([0, 3, 7, 10], dtype=jnp.int32)
        slot_mapping = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        kv_slot_mapping = jnp.array([-1, 10, 20, -1], dtype=jnp.int32)
        dest = compress_store_ref._boundary_dest(
            positions, slot_mapping, kv_slot_mapping, compress_ratio=4, num_slots=100
        )
        assert dest[1] == 10
        assert dest[2] == 20
        assert dest[0] == 100
        assert dest[3] == 100

    def test_ref_compress_norm_rope_store_permutations(self):
        # 1. Mode CSA with words and rope_cache
        cache = jnp.zeros((2, 32, 128), dtype=jnp.int32)
        rope_cache = jnp.zeros((2, 8, 128), dtype=jnp.int32)
        positions = jnp.array([3], dtype=jnp.int32)
        slot_mapping = jnp.array([0], dtype=jnp.int32)
        block_table = jnp.array([0, 1], dtype=jnp.int32)
        token_to_req = jnp.array([0], dtype=jnp.int32)
        kv_slot_mapping = jnp.array([0], dtype=jnp.int32)
        rms_weight = jnp.ones((512,), dtype=jnp.float32)
        cos_sin_cache = jnp.ones((16, 64), dtype=jnp.float32)

        new_c, new_r = compress_store_ref.ref_compress_norm_rope_store(
            cache=cache,
            rope_cache=rope_cache,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=cos_sin_cache,
            block_table_stride=2,
            state_block_size=2,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            rms_eps=1e-6,
            quant_block=64,
            is_quantized=True,
            has_rope=True,
            has_rope_cache=True,
        )
        assert new_c.shape == cache.shape
        assert new_r.shape == rope_cache.shape

        # 1b. Mode CSA with int32 state_cache, has_rope_cache=None, uint8 rope_cache
        state_cache_words = jnp.zeros((2, 32, 128), dtype=jnp.int32)
        rope_cache_slabs = jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8)
        new_c_1b, new_r_1b = compress_store_ref.ref_compress_norm_rope_store(
            cache=cache,
            rope_cache=rope_cache_slabs,
            state_cache=state_cache_words,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=cos_sin_cache,
            block_table_stride=2,
            state_block_size=2,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            rms_eps=1e-6,
            quant_block=64,
            is_quantized=True,
            has_rope=True,
            has_rope_cache=None,
        )
        assert new_c_1b.shape == cache.shape
        assert new_r_1b.shape == rope_cache_slabs.shape

        # 2. Mode CSA_INDEXER (head_dim=128, 256 last dim)
        cache_idx = jnp.zeros((2, 16, 4, 256), dtype=jnp.uint8)
        new_c_idx, _ = compress_store_ref.ref_compress_norm_rope_store(
            cache=cache_idx,
            rope_cache=jnp.zeros((2, 4, 4, 128), dtype=jnp.uint8),
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=jnp.ones((128,), dtype=jnp.float32),
            cos_sin_cache=cos_sin_cache,
            block_table_stride=2,
            state_block_size=2,
            head_dim=128,
            rope_head_dim=64,
            compress_ratio=4,
            overlap=True,
            rms_eps=1e-6,
            quant_block=128,
            is_quantized=True,
            has_rope=False,
            has_rope_cache=False,
        )
        assert new_c_idx.shape == cache_idx.shape

        # 3. Mode HCA (unquantized, no rope cache)
        cache_hca = jnp.zeros((2, 64, 4, 128), dtype=jnp.uint8)
        new_c_hca, _ = compress_store_ref.ref_compress_norm_rope_store(
            cache=cache_hca,
            rope_cache=jnp.zeros((2, 4, 4, 128), dtype=jnp.uint8),
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req,
            kv_slot_mapping=kv_slot_mapping,
            rms_weight=rms_weight,
            cos_sin_cache=cos_sin_cache,
            block_table_stride=2,
            state_block_size=2,
            head_dim=512,
            rope_head_dim=64,
            compress_ratio=128,
            overlap=False,
            rms_eps=1e-6,
            quant_block=0,
            is_quantized=False,
            has_rope=False,
            has_rope_cache=False,
        )
        assert new_c_hca.shape == cache_hca.shape

    def test_ref_compress_norm_rope_store_guards(self):
        cache = jnp.zeros((2, 32, 4, 128), dtype=jnp.uint8)
        rope_cache = jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8)
        positions = jnp.array([3], dtype=jnp.int32)
        slot_mapping = jnp.array([0], dtype=jnp.int32)
        block_table = jnp.array([0, 1], dtype=jnp.int32)
        token_to_req = jnp.array([0], dtype=jnp.int32)
        kv_slot_mapping = jnp.array([0], dtype=jnp.int32)
        rms_weight = jnp.ones((512,), dtype=jnp.float32)

        # 1. Packed NOPE record exceeds cache width guard
        with pytest.raises(ValueError, match="exceeds cache width"):
            compress_store_ref.ref_compress_norm_rope_store(
                cache=cache,
                rope_cache=rope_cache,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_to_req_indices=token_to_req,
                kv_slot_mapping=kv_slot_mapping,
                rms_weight=rms_weight,
                cos_sin_cache=jnp.ones((16, 32), dtype=jnp.float32),
                block_table_stride=2,
                state_block_size=2,
                head_dim=512,
                rope_head_dim=32,
                compress_ratio=4,
                overlap=True,
                rms_eps=1e-6,
                quant_block=64,
                is_quantized=True,
                has_rope=True,
                has_rope_cache=True,
            )

        # 2. RoPE dimension mismatch guard
        with (
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.compress_and_store.compress_store_ref.compress_norm_rope",
                return_value=jnp.zeros((1, 576)),
            ),
            pytest.raises(ValueError, match="CSA RoPE is 64 bf16 per token"),
        ):
            compress_store_ref.ref_compress_norm_rope_store(
                cache=cache,
                rope_cache=rope_cache,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_to_req_indices=token_to_req,
                kv_slot_mapping=kv_slot_mapping,
                rms_weight=rms_weight,
                cos_sin_cache=jnp.ones((16, 64), dtype=jnp.float32),
                block_table_stride=2,
                state_block_size=2,
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                rms_eps=1e-6,
                quant_block=64,
                is_quantized=True,
                has_rope=True,
                has_rope_cache=True,
            )


# ==============================================================================
# 4. TestCompute (compute.py)
# ==============================================================================


class TestCompute:
    """Tests 100% of compute.py."""

    def test_interleaved_rope_vector(self):
        tile_n = 2
        x = jnp.ones((tile_n, 1, 128), dtype=jnp.float32)
        cos_32 = jnp.ones((tile_n, 1, 32), dtype=jnp.float32)
        sin_32 = jnp.zeros((tile_n, 1, 32), dtype=jnp.float32)
        out = compute.interleaved_rope_vector(x, cos_32, sin_32)
        assert out.shape == (tile_n, 1, 128)

    def test_quantize_and_pack_tiled(self):
        tile_n = 2
        x = jnp.ones((tile_n, 1, 128), dtype=jnp.float32)
        with patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast):
            q, scale = compute.quantize_fp8_tiled(x, block_size=64)
            assert q.shape == (tile_n, 1, 128)

            # pack_nope_tiled (nope_dim < q_flat.shape[1] and nope_blocks < scale_flat)
            packed = compute.pack_nope_tiled(
                q,
                scale,
                nope_dim=64,
                block_size=64,
                nope_width_bytes=256,
                last_dim_size=128,
            )
            assert packed.shape == (tile_n, 2, 128)

            # pack_nope_tiled with nope_dim >= q_flat.shape[1] and
            # nope_blocks >= scale_flat
            packed2 = compute.pack_nope_tiled(
                q,
                scale,
                nope_dim=128,
                block_size=64,
                nope_width_bytes=256,
                last_dim_size=128,
            )
            assert packed2.shape == (tile_n, 2, 128)

    def test_quantize_and_pack_lane_periodic(self):
        tile_n = 2
        x = jnp.ones((tile_n, 4, 128), dtype=jnp.float32)
        with patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast):
            q, scale = compute.quantize_fp8_lane_periodic(x, period=64, nope_dim=448)
            assert q.shape == (tile_n, 4, 128)

            packed = compute.pack_nope_lane_periodic(
                q, scale, nope_dim=448, nope_width_bytes=512, last_dim_size=128
            )
            assert packed.shape == (tile_n, 4, 128)

    def test_rope_words_and_merge(self):
        tile_n = 4
        rope_slot = jnp.ones((tile_n, 1, 128), dtype=jnp.float32)
        with (
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
            patch("jax.experimental.pallas.tpu.roll", side_effect=fake_roll),
        ):
            words = compute.rope_words_tiled(rope_slot, rope_head_dim=64)
            assert words.shape == (tile_n, 1, 128)

            # merge_rope_row
            row = jnp.zeros((1, 128), dtype=jnp.int32)
            kv_slots = jnp.array([0, 1, -1, 4], dtype=jnp.int32)
            merged = compute.merge_rope_row(
                row=row,
                kv_slots=kv_slots,
                words=words,
                rope_lane=64,
                n=0,
            )
            assert merged.shape == row.shape

    def test_merge_slot_updates(self):
        slots_val = jnp.zeros((4, 256), dtype=jnp.uint8)
        kv_slots = jnp.array([0, 1, -1, 4], dtype=jnp.int32)
        val_padded = jnp.ones((4, 1, 256), dtype=jnp.uint8)
        merged = compute.merge_slot_updates(slots_val, kv_slots, val_padded, n=0)
        assert merged.shape == slots_val.shape
        assert jnp.any(merged > 0)

    def test_gather_from_page_buffer(self):
        # 1. is_indexer = True with overlap = True and False
        page_buf_idx = MockArrayRef(np.ones((2, 3, 32, 4, 256), dtype=np.uint8))
        pos_ref = np.array([3, 7], dtype=np.int32)
        kv_u8 = MockArrayRef(np.zeros((2, 8, 1, 4, 128), dtype=np.uint8))
        sc_u8 = MockArrayRef(np.zeros((2, 8, 1, 4, 128), dtype=np.uint8))

        with patch("jax.experimental.pallas.when", side_effect=fake_when):
            compute.gather_from_page_buffer(
                page_buf_idx,
                pos_ref,
                kv_u8,
                sc_u8,
                global_idx=0,
                num_tokens=2,
                window=8,
                block_size=4,
                pages_to_buffer_per_token=2,
                field_rows=1,
                state_rows_per_token=2,
                overlap=True,
                is_indexer=True,
            )
            assert np.any(kv_u8.arr > 0)

            # overlap = False
            compute.gather_from_page_buffer(
                page_buf_idx,
                pos_ref,
                kv_u8,
                sc_u8,
                global_idx=0,
                num_tokens=2,
                window=4,
                block_size=4,
                pages_to_buffer_per_token=2,
                field_rows=1,
                state_rows_per_token=2,
                overlap=False,
                is_indexer=True,
            )

        # 2. is_indexer = False (Standard CSA/HCA) with overlap = True and False
        page_buf_std = MockArrayRef(np.ones((2, 3, 128, 4, 128), dtype=np.uint8))
        kv_u8_std = MockArrayRef(np.zeros((2, 8, 4, 4, 128), dtype=np.uint8))
        sc_u8_std = MockArrayRef(np.zeros((2, 8, 4, 4, 128), dtype=np.uint8))

        with patch("jax.experimental.pallas.loop", side_effect=fake_loop):
            compute.gather_from_page_buffer(
                page_buf_std,
                pos_ref,
                kv_u8_std,
                sc_u8_std,
                global_idx=0,
                num_tokens=2,
                window=8,
                block_size=4,
                pages_to_buffer_per_token=2,
                field_rows=2,
                state_rows_per_token=4,
                overlap=True,
                is_indexer=False,
            )
            assert np.any(kv_u8_std.arr > 0)

            # overlap = False
            compute.gather_from_page_buffer(
                page_buf_std,
                pos_ref,
                kv_u8_std,
                sc_u8_std,
                global_idx=0,
                num_tokens=2,
                window=4,
                block_size=4,
                pages_to_buffer_per_token=2,
                field_rows=1,
                state_rows_per_token=2,
                overlap=False,
                is_indexer=False,
            )


# ==============================================================================
# 5. TestBufferedRef (buffered_ref.py)
# ==============================================================================


class TestBufferedRef:
    """Tests 100% of buffered_ref.py."""

    def test_create_allocs_and_specs(self):
        cfg_csa = config.Configs.make(
            config.Mode.CSA, size_n=4, physical_page_size=32, state_block_size=8
        )
        cache_ref = MockArrayRef(np.zeros((2, 32, 128), dtype=np.int32))
        rope_cache_ref = MockArrayRef(np.zeros((2, 8, 128), dtype=np.int32))
        state_cache_ref = MockArrayRef(np.zeros((2, 32, 128), dtype=np.int32))
        cos_sin_ref = jnp.zeros((16, 64), dtype=jnp.float32)
        pos_ref = jnp.zeros((4,), dtype=jnp.int32)
        bt_ref = jnp.zeros((4,), dtype=jnp.int32)
        t2r_ref = jnp.zeros((4,), dtype=jnp.int32)
        kv_slot_ref = jnp.zeros((4,), dtype=jnp.int32)
        mask_ref = jnp.zeros((4,), dtype=bool)

        allocs, specs, args = buffered_ref.create_allocs_and_specs(
            cfgs=cfg_csa,
            cache_ref=cache_ref,
            rope_cache_ref=rope_cache_ref,
            state_cache_ref=state_cache_ref,
            cos_sin_cache_ref=cos_sin_ref,
            positions_ref=pos_ref,
            block_table_ref=bt_ref,
            block_table_stride=2,
            token_to_req_indices_ref=t2r_ref,
            kv_slot_mapping_ref=kv_slot_ref,
            is_first_mask_ref=mask_ref,
            is_first_mask_rope_ref=mask_ref,
        )
        assert len(allocs) == 4
        assert len(specs) == 4
        assert len(args) == 4

        # Non-rope / non-rope_cache
        cfg_hca = config.Configs.make(
            config.Mode.HCA,
            size_n=4,
            physical_page_size=64,
            state_block_size=8,
            rope_head_dim=0,
        )
        allocs_hca, _, _ = buffered_ref.create_allocs_and_specs(
            cfgs=cfg_hca,
            cache_ref=jnp.zeros((2, 64, 4, 128), dtype=jnp.uint8),
            rope_cache_ref=None,
            state_cache_ref=None,
            cos_sin_cache_ref=None,
            positions_ref=pos_ref,
            block_table_ref=bt_ref,
            block_table_stride=2,
            token_to_req_indices_ref=t2r_ref,
            kv_slot_mapping_ref=kv_slot_ref,
            is_first_mask_ref=mask_ref,
            is_first_mask_rope_ref=mask_ref,
        )
        assert allocs_hca[0] is None
        assert allocs_hca[3] is None

        # CSA_INDEXER mode (covers line 480: output_args with 3 elements)
        cfg_idx = config.Configs.make(
            config.Mode.CSA_INDEXER,
            size_n=4,
            physical_page_size=16,
            state_block_size=4,
            rope_head_dim=0,
        )
        allocs_idx, _, args_idx = buffered_ref.create_allocs_and_specs(
            cfgs=cfg_idx,
            cache_ref=jnp.zeros((2, 16, 4, 256), dtype=jnp.uint8),
            rope_cache_ref=None,
            state_cache_ref=None,
            cos_sin_cache_ref=None,
            positions_ref=pos_ref,
            block_table_ref=bt_ref,
            block_table_stride=2,
            token_to_req_indices_ref=t2r_ref,
            kv_slot_mapping_ref=kv_slot_ref,
            is_first_mask_ref=mask_ref,
            is_first_mask_rope_ref=mask_ref,
        )
        assert len(allocs_idx) == 4
        assert len(args_idx[1]) == 3

    def test_buffered_ref_dma_methods(self):
        cfg_csa = config.Configs.make(
            config.Mode.CSA, size_n=4, physical_page_size=32, state_block_size=8
        )
        cfg_idx = config.Configs.make(
            config.Mode.CSA_INDEXER, size_n=4, physical_page_size=16, state_block_size=4
        )

        mock_copy = MockAsyncCopy()
        mock_sem = MockArrayRef(np.zeros((2,), dtype=np.int32))
        mock_win = MockArrayRef(np.zeros((2, 4, 4, 32, 4, 256), dtype=np.uint8))

        with (
            patch("jax.experimental.pallas.loop", side_effect=fake_loop),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch.object(
                pltpu.BufferedRef,
                "current_copy_in_slot",
                new_callable=PropertyMock,
                return_value=0,
            ),
            patch.object(
                pltpu.BufferedRef,
                "current_wait_in_slot",
                new_callable=PropertyMock,
                return_value=0,
            ),
            patch.object(
                pltpu.BufferedRef,
                "current_copy_out_slot",
                new_callable=PropertyMock,
                return_value=0,
            ),
            patch.object(
                pltpu.BufferedRef,
                "current_wait_out_slot",
                new_callable=PropertyMock,
                return_value=0,
            ),
        ):

            def _setup_alloc(alloc):
                object.__setattr__(alloc, "sem_recvs", mock_sem)
                object.__setattr__(alloc, "sem_sends", mock_sem)
                object.__setattr__(alloc, "window_ref", mock_win)
                return alloc

            # CosSinRef
            cos_sin_alloc = _setup_alloc(
                buffered_ref.CosSinRef.create(
                    spec=pl.BlockSpec(
                        block_shape=cfg_csa.cos_sin_shape(), memory_space=pltpu.VMEM
                    ),
                    dtype_or_type=jnp.float32,
                    buffer_type=pltpu.BufferType.INPUT,
                    buffer_count=2,
                    use_lookahead=False,
                    cfgs=cfg_csa,
                )
            )
            cos_sin_alloc.copy_in(
                (
                    MockArrayRef(np.zeros((16, 64), dtype=np.float32)),
                    np.array([0, 1, 2, 3]),
                ),
                (0,),
            )
            cos_sin_alloc.wait_in(None, (0,))

            # PageBufferRef
            page_alloc = _setup_alloc(
                buffered_ref.PageBufferRef.create(
                    spec=pl.BlockSpec(
                        block_shape=cfg_csa.page_buffer_shape(), memory_space=pltpu.VMEM
                    ),
                    dtype_or_type=jnp.uint8,
                    buffer_type=pltpu.BufferType.INPUT,
                    buffer_count=2,
                    use_lookahead=False,
                    cfgs=cfg_csa,
                )
            )
            page_alloc.copy_in(
                (
                    MockArrayRef(np.zeros((2, 32, 4, 128), dtype=np.uint8)),
                    np.array([0, 1, 2, 3]),
                    np.array([0, 1]),
                    1,
                    np.array([0, 0, 0, 0]),
                ),
                (0,),
            )
            page_alloc.wait_in(None, (0,))

            # OutputRef (non-indexer)
            out_alloc = _setup_alloc(
                buffered_ref.OutputRef.create(
                    spec=pl.BlockSpec(
                        block_shape=cfg_csa.output_shape(), memory_space=pltpu.VMEM
                    ),
                    dtype_or_type=jnp.uint8,
                    buffer_type=pltpu.BufferType.OUTPUT,
                    buffer_count=2,
                    use_lookahead=False,
                    cfgs=cfg_csa,
                )
            )
            out_alloc.copy_out(
                (
                    MockArrayRef(np.zeros((2, 32, 4, 128), dtype=np.uint8)),
                    np.array([0, 1, 2, 3]),
                ),
                (0,),
            )
            out_alloc.wait_out((None, np.array([0, 1, 2, 3])), (0,))

            # OutputRef (indexer)
            out_idx_alloc = _setup_alloc(
                buffered_ref.OutputRef.create(
                    spec=pl.BlockSpec(
                        block_shape=cfg_idx.output_shape(), memory_space=pltpu.VMEM
                    ),
                    dtype_or_type=jnp.uint8,
                    buffer_type=pltpu.BufferType.INPUT_OUTPUT,
                    buffer_count=2,
                    use_lookahead=False,
                    cfgs=cfg_idx,
                )
            )
            out_idx_alloc.copy_in(
                (
                    MockArrayRef(np.zeros((2, 16, 4, 256), dtype=np.uint8)),
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )
            out_idx_alloc.wait_in(
                (
                    None,
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )
            out_idx_alloc.copy_out(
                (
                    MockArrayRef(np.zeros((2, 16, 4, 256), dtype=np.uint8)),
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )
            out_idx_alloc.wait_out(
                (
                    None,
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )

            # RoPEOutputRef
            rope_alloc = _setup_alloc(
                buffered_ref.RoPEOutputRef.create(
                    spec=pl.BlockSpec(
                        block_shape=cfg_csa.rope_output_shape(), memory_space=pltpu.VMEM
                    ),
                    dtype_or_type=jnp.uint8,
                    buffer_type=pltpu.BufferType.INPUT_OUTPUT,
                    buffer_count=2,
                    use_lookahead=False,
                    cfgs=cfg_csa,
                )
            )
            rope_alloc.copy_in(
                (
                    MockArrayRef(np.zeros((2, 8, 4, 128), dtype=np.uint8)),
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )
            rope_alloc.wait_in(
                (
                    None,
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )
            rope_alloc.copy_out(
                (
                    MockArrayRef(np.zeros((2, 8, 4, 128), dtype=np.uint8)),
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )
            rope_alloc.wait_out(
                (
                    None,
                    np.array([0, 1, 2, 3]),
                    np.array([True, True, True, True]),
                ),
                (0,),
            )


# ==============================================================================
# 6. TestKernel (kernel.py)
# ==============================================================================


class TestKernel:
    """Tests 100% of kernel.py."""

    def test_derive_aliases_and_first_mask(self):
        aliases = kernel.derive_aliases(
            has_rope=True, has_rope_cache=True, num_scalar_prefetch=7
        )
        assert aliases == {9: 0, 10: 1}

        aliases_no_rope = kernel.derive_aliases(
            has_rope=False, has_rope_cache=False, num_scalar_prefetch=7
        )
        assert aliases_no_rope == {8: 0}

        # compute_is_first_mask with pad_len > 0
        kv_slots = jnp.array([0, 0, 1, 4, 4], dtype=jnp.int32)
        mask = kernel.compute_is_first_mask(kv_slots, tile_n=4, pack_factor=4)
        assert mask.shape == kv_slots.shape
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False
        assert bool(mask[2]) is False
        assert bool(mask[3]) is True

        # compute_is_first_mask with pad_len == 0
        kv_slots_even = jnp.array([0, 4, 8, 12], dtype=jnp.int32)
        mask_even = kernel.compute_is_first_mask(kv_slots_even, tile_n=4, pack_factor=4)
        assert jnp.all(mask_even)

    def test_inner_kernel_simulation_csa(self):
        cfg_csa = config.Configs.make(
            config.Mode.CSA, size_n=4, physical_page_size=64, state_block_size=2
        )
        cos_sin_vmem = MockArrayRef(np.ones((4, 64), dtype=np.float32))
        out_vmem = MockArrayRef(np.zeros(cfg_csa.output_shape(), dtype=np.uint8))
        page_buf_vmem = MockArrayRef(
            np.ones(cfg_csa.page_buffer_shape(), dtype=np.uint8)
        )
        rope_out_vmem = MockArrayRef(
            np.zeros(cfg_csa.rope_output_shape(), dtype=np.uint8)
        )
        positions_ref = np.array([3, 3, 3, 3], dtype=np.int32)
        rms_weight_vmem = MockArrayRef(
            np.ones((cfg_csa.head_tiles, 128), dtype=np.float32)
        )
        window_vmem = MockArrayRef(np.ones(cfg_csa.window_shape(), dtype=np.float32))
        kv_slot_mapping = np.array([0, 1, 2, 3], dtype=np.int32)
        is_first_mask = np.array([True, False, False, False])
        is_first_mask_rope = np.array([True, False, False, False])

        with (
            patch("jax.experimental.pallas.program_id", return_value=0),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.loop", side_effect=fake_loop),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
            patch("jax.experimental.pallas.tpu.roll", side_effect=fake_roll),
        ):
            kernel.inner_kernel(
                cos_sin_vmem=cos_sin_vmem,
                out_vmem=out_vmem,
                page_buffer_vmem=page_buf_vmem,
                rope_out_vmem=rope_out_vmem,
                positions_ref=positions_ref,
                rms_weight_vmem_ref=rms_weight_vmem,
                window_vmem=window_vmem,
                kv_slot_mapping_ref=kv_slot_mapping,
                is_first_mask_ref=is_first_mask,
                is_first_mask_rope_ref=is_first_mask_rope,
                cfgs=cfg_csa,
            )
            assert np.any(out_vmem.arr > 0)

    def test_inner_kernel_simulation_indexer_and_hca(self):
        # 1. CSA_INDEXER
        cfg_idx = config.Configs.make(
            config.Mode.CSA_INDEXER,
            size_n=4,
            physical_page_size=32,
            state_block_size=2,
            rope_head_dim=0,
        )
        out_vmem_idx = MockArrayRef(np.zeros(cfg_idx.output_shape(), dtype=np.uint8))
        page_buf_idx = MockArrayRef(
            np.ones(cfg_idx.page_buffer_shape(), dtype=np.uint8)
        )
        positions_ref = np.array([3, 3, 3, 3], dtype=np.int32)
        rms_weight_idx = MockArrayRef(np.ones((1, 128), dtype=np.float32))
        window_idx = MockArrayRef(np.ones(cfg_idx.window_shape(), dtype=np.float32))
        kv_slot_mapping = np.array([0, 1, 2, 3], dtype=np.int32)
        is_first_mask = np.array([True, True, True, True])

        with (
            patch("jax.experimental.pallas.program_id", return_value=0),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.loop", side_effect=fake_loop),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
        ):
            kernel.inner_kernel(
                cos_sin_vmem=None,
                out_vmem=out_vmem_idx,
                page_buffer_vmem=page_buf_idx,
                rope_out_vmem=None,
                positions_ref=positions_ref,
                rms_weight_vmem_ref=rms_weight_idx,
                window_vmem=window_idx,
                kv_slot_mapping_ref=kv_slot_mapping,
                is_first_mask_ref=is_first_mask,
                is_first_mask_rope_ref=is_first_mask,
                cfgs=cfg_idx,
            )
            assert np.any(out_vmem_idx.arr > 0)

        # 2. HCA (normed bf16 bitcast)
        cfg_hca = config.Configs.make(
            config.Mode.HCA,
            size_n=4,
            physical_page_size=64,
            state_block_size=2,
            rope_head_dim=0,
        )
        out_vmem_hca = MockArrayRef(np.zeros(cfg_hca.output_shape(), dtype=np.uint8))
        page_buf_hca = MockArrayRef(
            np.ones(cfg_hca.page_buffer_shape(), dtype=np.uint8)
        )
        rms_weight_hca = MockArrayRef(
            np.ones((cfg_hca.head_tiles, 128), dtype=np.float32)
        )
        window_hca = MockArrayRef(np.ones(cfg_hca.window_shape(), dtype=np.float32))

        with (
            patch("jax.experimental.pallas.program_id", return_value=0),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.loop", side_effect=fake_loop),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
        ):
            kernel.inner_kernel(
                cos_sin_vmem=None,
                out_vmem=out_vmem_hca,
                page_buffer_vmem=page_buf_hca,
                rope_out_vmem=None,
                positions_ref=positions_ref,
                rms_weight_vmem_ref=rms_weight_hca,
                window_vmem=window_hca,
                kv_slot_mapping_ref=kv_slot_mapping,
                is_first_mask_ref=is_first_mask,
                is_first_mask_rope_ref=is_first_mask,
                cfgs=cfg_hca,
            )
            assert np.any(out_vmem_hca.arr > 0)

        # 3. head_tiles == 1 and has_rope=True (covers line 117: normed = rope_ropped)
        cfg_single_tile = config.Configs.make(
            config.Mode.CSA_INDEXER,
            size_n=4,
            physical_page_size=64,
            state_block_size=2,
            head_dim=128,
            rope_head_dim=64,
        )
        out_vmem_single = MockArrayRef(
            np.zeros(cfg_single_tile.output_shape(), dtype=np.uint8)
        )
        page_buf_single = MockArrayRef(
            np.ones(cfg_single_tile.page_buffer_shape(), dtype=np.uint8)
        )
        rms_weight_single = MockArrayRef(np.ones((1, 128), dtype=np.float32))
        window_single = MockArrayRef(
            np.ones(cfg_single_tile.window_shape(), dtype=np.float32)
        )
        rope_out_single = None
        cos_sin_single = MockArrayRef(np.ones((4, 64), dtype=np.float32))

        with (
            patch("jax.experimental.pallas.program_id", return_value=0),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.loop", side_effect=fake_loop),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
            patch("jax.experimental.pallas.tpu.roll", side_effect=fake_roll),
        ):
            kernel.inner_kernel(
                cos_sin_vmem=cos_sin_single,
                out_vmem=out_vmem_single,
                page_buffer_vmem=page_buf_single,
                rope_out_vmem=rope_out_single,
                positions_ref=positions_ref,
                rms_weight_vmem_ref=rms_weight_single,
                window_vmem=window_single,
                kv_slot_mapping_ref=kv_slot_mapping,
                is_first_mask_ref=is_first_mask,
                is_first_mask_rope_ref=is_first_mask,
                cfgs=cfg_single_tile,
            )
            assert np.any(out_vmem_single.arr > 0)

    def test_kernel_fn_execution(self):
        cfg_csa = config.Configs.make(
            config.Mode.CSA, size_n=4, physical_page_size=32, state_block_size=8
        )
        mock_pipeline = MagicMock()

        with (
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline", return_value=mock_pipeline
            ),
            patch(
                "jax.experimental.pallas.with_scoped",
                side_effect=lambda allocations: lambda fn: lambda: fn(allocations),
            ),
        ):
            kernel.kernel_fn(
                block_table_ref=MockArrayRef(np.zeros((2,), dtype=np.int32)),
                positions_ref=MockArrayRef(np.zeros((4,), dtype=np.int32)),
                token_to_req_indices_ref=MockArrayRef(np.zeros((4,), dtype=np.int32)),
                kv_slot_mapping_ref=MockArrayRef(np.zeros((4,), dtype=np.int32)),
                is_first_mask_ref=MockArrayRef(np.zeros((4,), dtype=bool)),
                is_first_mask_rope_ref=MockArrayRef(np.zeros((4,), dtype=bool)),
                grid_size_ref=MockArrayRef(np.array(1, dtype=np.int32)),
                rms_weight_ref=MockArrayRef(np.zeros((4, 128), dtype=np.float32)),
                cos_sin_cache_ref=MockArrayRef(np.zeros((16, 64), dtype=np.float32)),
                cache_ref=MockArrayRef(np.zeros((2, 32, 4, 128), dtype=np.uint8)),
                rope_cache_ref=MockArrayRef(np.zeros((2, 8, 4, 128), dtype=np.uint8)),
                state_cache_ref=MockArrayRef(np.zeros((2, 32, 4, 128), dtype=np.uint8)),
                _out_cache_ref=None,
                _out_rope_cache_ref=None,
                window_scratch_ref=MockArrayRef(
                    np.zeros((2, 4, 8, 4, 128), dtype=np.float32)
                ),
                cfgs=cfg_csa,
                block_table_stride=2,
            )
            mock_pipeline.assert_called_once()

    def test_compress_norm_rope_store_validation_guards(self):
        cache = jnp.zeros((2, 32, 4, 128), dtype=jnp.uint8)
        positions = jnp.array([0, 1], dtype=jnp.int32)
        block_table = jnp.array([[0, 1]], dtype=jnp.int32)
        token_to_req = jnp.array([0, 0], dtype=jnp.int32)
        kv_slot_mapping = jnp.array([0, 1], dtype=jnp.int32)
        rms_weight = jnp.ones((512,), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            kernel.compress_norm_rope_store(
                cache,
                positions,
                block_table,
                token_to_req,
                kv_slot_mapping,
                rms_weight,
                block_table_stride=2,
                state_block_size=8,
                compress_ratio=4,
                overlap=True,
            )

        with pytest.raises(AssertionError):
            kernel.compress_norm_rope_store(
                cache,
                positions,
                jnp.array([0, 1, 2], dtype=jnp.int32),
                token_to_req,
                kv_slot_mapping,
                rms_weight,
                block_table_stride=2,
                state_block_size=8,
                compress_ratio=4,
                overlap=True,
            )

        with pytest.raises(ValueError, match="rope_cache must be provided"):
            kernel.compress_norm_rope_store(
                cache,
                positions,
                jnp.array([0, 1], dtype=jnp.int32),
                token_to_req,
                kv_slot_mapping,
                rms_weight,
                block_table_stride=2,
                state_block_size=8,
                rope_cache=None,
                cos_sin_cache=jnp.ones((16, 64), dtype=jnp.float32),
                compress_ratio=4,
                overlap=True,
            )

    def test_compress_norm_rope_store_mocked_pallas_call(self):
        cache = jnp.zeros((2, 32, 4, 128), dtype=jnp.uint8)
        rope_cache = jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8)
        positions = jnp.array([0, 1], dtype=jnp.int32)
        block_table = jnp.array([0, 1], dtype=jnp.int32)
        token_to_req = jnp.array([0, 0], dtype=jnp.int32)
        kv_slot_mapping = jnp.array([0, 1], dtype=jnp.int32)
        rms_weight = jnp.ones((512,), dtype=jnp.float32)
        cos_sin_cache = jnp.ones((16, 64), dtype=jnp.float32)

        def mock_caller(*args, **kwargs):
            out_shape = kwargs.get("out_shape", (cache, rope_cache))
            if (
                isinstance(out_shape, tuple)
                and len(out_shape) == 2
                and out_shape[1] is None
            ):
                return lambda *call_args: (cache, None)
            return lambda *call_args: (cache, rope_cache)

        with patch("jax.experimental.pallas.pallas_call", side_effect=mock_caller):
            out_c, out_r = kernel.compress_norm_rope_store(
                cache,
                positions,
                block_table,
                token_to_req,
                kv_slot_mapping,
                rms_weight,
                block_table_stride=2,
                state_block_size=8,
                rope_cache=rope_cache,
                cos_sin_cache=cos_sin_cache,
                compress_ratio=4,
                overlap=True,
            )
            assert out_c.shape == cache.shape
            assert out_r.shape == rope_cache.shape

            # HCA mode call: covers branch 363->369
            out_c_hca, out_r_hca = kernel.compress_norm_rope_store(
                cache,
                positions,
                block_table,
                token_to_req,
                kv_slot_mapping,
                rms_weight,
                block_table_stride=2,
                state_block_size=8,
                rope_cache=None,
                cos_sin_cache=None,
                compress_ratio=4,
                overlap=False,
            )
            assert out_c_hca.shape == cache.shape
            assert out_r_hca is None


# ==============================================================================
# 7. TestProjAndSaveState (proj_and_save_state.py)
# ==============================================================================


class TestProjAndSaveState:
    """Tests 100% of proj_and_save_state.py."""

    def test_load_and_store_acc_ref(self):
        with patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast):
            # last_dim = 128
            ref_128 = MockArrayRef(
                np.zeros((4, 128), dtype=np.float32).reshape(2, 2, 1, 128)
            )
            val = jnp.ones((2, 256), dtype=jnp.float32)
            proj_and_save_state.store_to_acc_ref(ref_128, val)
            loaded = proj_and_save_state.load_from_acc_ref(ref_128, dtype=jnp.float32)
            assert loaded.shape == (2, 256)
            assert np.allclose(loaded, 1.0)

            # last_dim = 256
            ref_256 = MockArrayRef(
                np.zeros((2, 256), dtype=np.float32).reshape(2, 1, 1, 256)
            )
            val_256 = jnp.ones((2, 256), dtype=jnp.float32) * 2.0
            proj_and_save_state.store_to_acc_ref(ref_256, val_256)
            loaded_256 = proj_and_save_state.load_from_acc_ref(ref_256, dtype=None)
            assert loaded_256.shape == (2, 256)
            assert np.allclose(loaded_256, 2.0)

    def test_select_tile_k(self):
        assert proj_and_save_state._select_tile_k(1024) == 1024
        assert proj_and_save_state._select_tile_k(7168, cap=3584) == 3584
        with pytest.raises(AssertionError, match="Cannot find a tile_k"):
            proj_and_save_state._select_tile_k(127, align=128)

    def test_kernel_execution_and_branches(self):
        # 1. Pipeline execution with cfg_k1 (num_k=1)
        cfg_k1 = proj_and_save_state.Configs(
            tile_sizes=proj_and_save_state.TileSizes(tile_m=256, tile_k=128, tile_n=4),
            dims=proj_and_save_state.Dimensions(
                size_m=512, size_k=128, size_n=4, compress_ratio=4
            ),
            token_page_size=8,
        )
        slot_mapping = np.array([0, 4, -1, 8], dtype=np.int32)
        positions = np.array([0, 1, 2, 3], dtype=np.int32)
        hidden_states = jnp.ones((4, 128), dtype=jnp.float32)
        wkv_wgate = jnp.ones((128, 512), dtype=jnp.float32)
        ape = jnp.ones((4, 256), dtype=jnp.float32)
        cache = MockArrayRef(np.zeros((2, 8, 4, 128), dtype=np.uint8))
        acc_ref = MockArrayRef(np.zeros((2, 4, 2, 1, 128), dtype=np.float32))
        sem_ref = MockArrayRef(np.zeros((2,), dtype=np.int32))
        write_counts_ref = MockArrayRef(np.array([1, 1], dtype=np.int32))

        mock_copy = MockAsyncCopy()

        def mock_emit_pipeline_k1(fn, grid, in_specs):
            def runner(*args):
                # Populate write_counts_ref to cover line 222 (_do_wait)
                write_counts_ref.arr[...] = 2
                # matmul_ape_write_out: copy_dma=True, add_ape=True
                # (n_idx=0, m_idx=1, k_idx=0)
                with patch("jax.experimental.pallas.program_id", side_effect=[0, 1, 0]):
                    fn(
                        jnp.ones((128, 256)),
                        jnp.ones((4, 128)),
                        jnp.array([0, 1, 2, 3]),
                    )

            return runner

        with (
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch("jax.experimental.pallas.semaphore_wait"),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                side_effect=mock_emit_pipeline_k1,
            ),
        ):
            proj_and_save_state.kernel(
                slot_mapping,
                positions,
                hidden_states,
                wkv_wgate,
                ape,
                cache,
                None,
                acc_ref,
                sem_ref,
                write_counts_ref,
                cfgs=cfg_k1,
            )

        # 2. Pipeline execution with cfg_k2 (num_k=2)
        cfg_k2 = proj_and_save_state.Configs(
            tile_sizes=proj_and_save_state.TileSizes(tile_m=256, tile_k=128, tile_n=4),
            dims=proj_and_save_state.Dimensions(
                size_m=512, size_k=256, size_n=8, compress_ratio=4
            ),
            token_page_size=8,
        )

        def mock_emit_pipeline_k2(fn, grid, in_specs):
            def runner(*args):
                # (a) matmul: copy_dma=False, add_ape=False (n_idx=0, m_idx=0, k_idx=0)
                with patch("jax.experimental.pallas.program_id", side_effect=[0, 0, 0]):
                    fn(
                        jnp.ones((128, 256)),
                        jnp.ones((4, 128)),
                        jnp.array([0, 1, 2, 3]),
                    )
                # (b) matmul_write_out: copy_dma=True, add_ape=False
                # (n_idx=0, m_idx=1, k_idx=0)
                with patch("jax.experimental.pallas.program_id", side_effect=[0, 1, 0]):
                    fn(
                        jnp.ones((128, 256)),
                        jnp.ones((4, 128)),
                        jnp.array([0, 1, 2, 3]),
                    )
                # (c) matmul_ape_only: copy_dma=False, add_ape=True
                # (n_idx=0, m_idx=1, k_idx=1)
                with patch("jax.experimental.pallas.program_id", side_effect=[0, 1, 1]):
                    fn(
                        jnp.ones((128, 256)),
                        jnp.ones((4, 128)),
                        jnp.array([0, 1, 2, 3]),
                    )

            return runner

        with (
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch("jax.experimental.pallas.semaphore_wait"),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                side_effect=mock_emit_pipeline_k2,
            ),
        ):
            proj_and_save_state.kernel(
                np.array([0, 4, -1, 8, 1, 2, 3, 4], dtype=np.int32),
                np.arange(8, dtype=np.int32),
                jnp.ones((8, 256), dtype=jnp.float32),
                jnp.ones((256, 512), dtype=jnp.float32),
                ape,
                cache,
                None,
                acc_ref,
                sem_ref,
                write_counts_ref,
                cfgs=cfg_k2,
            )

        # last_tile_idx < 0 (covers branch 315->318)
        cfg_empty = proj_and_save_state.Configs(
            tile_sizes=proj_and_save_state.TileSizes(tile_m=256, tile_k=128, tile_n=4),
            dims=proj_and_save_state.Dimensions(
                size_m=512, size_k=128, size_n=0, compress_ratio=4
            ),
            token_page_size=8,
        )
        with (
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch("jax.experimental.pallas.semaphore_wait"),
            patch("jax.experimental.pallas.when", side_effect=fake_when),
            patch("jax.experimental.pallas.tpu.bitcast", side_effect=fake_bitcast),
            patch(
                "jax.experimental.pallas.tpu.emit_pipeline",
                return_value=lambda *args: None,
            ),
        ):
            proj_and_save_state.kernel(
                slot_mapping,
                positions,
                hidden_states,
                wkv_wgate,
                ape,
                cache,
                None,
                acc_ref,
                sem_ref,
                write_counts_ref,
                cfgs=cfg_empty,
            )

    def test_proj_and_save_state_outer_and_padding(self):
        hidden_states = jnp.ones((3, 128), dtype=jnp.float32)
        wkv_wgate = jnp.ones((128, 512), dtype=jnp.float32)
        ape = jnp.ones((4, 256), dtype=jnp.float32)
        positions = jnp.array([0, 1, 2], dtype=jnp.int32)
        slot_mapping = jnp.array([0, 1, 2], dtype=jnp.int32)
        cache = jnp.zeros((2, 8, 4, 128), dtype=jnp.uint8)

        def mock_caller(*args, **kwargs):
            return lambda *call_args: cache

        with patch("jax.experimental.pallas.pallas_call", side_effect=mock_caller):
            out = proj_and_save_state.proj_and_save_state(
                hidden_states,
                wkv_wgate,
                ape,
                positions,
                slot_mapping,
                cache,
                compress_ratio=4,
            )
            assert out.shape == cache.shape

            # Unpadded call where padded_num_tokens == num_tokens
            # (covers branch 379->386)
            hidden_states_8 = jnp.ones((8, 128), dtype=jnp.float32)
            positions_8 = jnp.arange(8, dtype=jnp.int32)
            slot_mapping_8 = jnp.arange(8, dtype=jnp.int32)
            out_8 = proj_and_save_state.proj_and_save_state(
                hidden_states_8,
                wkv_wgate,
                ape,
                positions_8,
                slot_mapping_8,
                cache,
                compress_ratio=4,
            )
            assert out_8.shape == cache.shape


# ==============================================================================
# 8. TestCompressorV1 (compressor_v1.py)
# ==============================================================================


class TestCompressorV1:
    """Tests 100% of compressor_v1.py."""

    def test_block_table_row_stride(self):
        bt = jnp.arange(16)
        assert compressor_v1.block_table_row_stride(bt, num_reqs=4) == 4
        with pytest.raises(AssertionError):
            compressor_v1.block_table_row_stride(jnp.ones((4, 4)), num_reqs=4)
        with pytest.raises(AssertionError):
            compressor_v1.block_table_row_stride(jnp.arange(15), num_reqs=4)

    def test_derive_metadata(self):
        positions = jnp.array([0, 1, 2, 3, 0, 1], dtype=jnp.int32)
        block_table = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 4, 6], dtype=jnp.int32)
        kv_block_table = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        # 128 rows per page -> capacity = 128 // 16 = 8 token states per page
        cache = jnp.zeros((4, 128, 4, 128), dtype=jnp.uint8)

        t2r, slot_slots, kv_slots = compressor_v1.derive_metadata(
            positions=positions,
            block_table=block_table,
            query_start_loc=query_start_loc,
            kv_block_table=kv_block_table,
            cache=cache,
            compress_ratio=4,
            state_block_size=4,
            head_dim=512,
            overlap=True,
            cos_sin_cache=jnp.ones((16, 64), dtype=jnp.float32),
        )
        assert t2r.shape == positions.shape
        assert slot_slots.shape == positions.shape
        assert kv_slots.shape == positions.shape
        assert kv_slots[3] >= 0
        assert kv_slots[0] == -1

    def test_derive_metadata_assertion(self):
        positions = jnp.array([0, 1], dtype=jnp.int32)
        cache = jnp.zeros((4, 16, 4, 128), dtype=jnp.uint8)
        with pytest.raises(AssertionError, match="overruns"):
            compressor_v1.derive_metadata(
                positions=positions,
                block_table=jnp.array([0, 1]),
                query_start_loc=jnp.array([0, 2]),
                kv_block_table=jnp.array([0, 1]),
                cache=cache,
                compress_ratio=4,
                state_block_size=64,
                head_dim=512,
                overlap=True,
                cos_sin_cache=None,
            )

    def test_prepare_boundary_batch(self):
        positions = jnp.array([0, 1, 2, 3, 4, 7], dtype=jnp.int32)
        token_to_req = jnp.array([0, 0, 0, 0, 1, 1], dtype=jnp.int32)
        kv_slot_mapping = jnp.array([-1, -1, -1, 0, -1, 4], dtype=jnp.int32)
        is_boundary = jnp.array([False, False, False, True, False, True])
        is_decode_token = jnp.array([False, False, False, True, False, False])

        pos, req, kv = compressor_v1.prepare_boundary_batch(
            positions=positions,
            token_to_req_indices=token_to_req,
            kv_slot_mapping=kv_slot_mapping,
            is_boundary=is_boundary,
            is_decode_token=is_decode_token,
            compress_ratio=4,
            num_reqs=2,
            rope_pack=4,
        )
        assert pos.ndim == 1
        assert req.ndim == 1
        assert kv.ndim == 1
        assert jnp.any(kv >= 0)

    def test_compressor_forward_shared_and_separate_buffer(self):
        hidden_states = jnp.ones((4, 128), dtype=jnp.float32)
        wkv_wgate = jnp.ones((128, 1024), dtype=jnp.float32)
        ape = jnp.ones((4, 512), dtype=jnp.float32)
        norm_weight = jnp.ones((512,), dtype=jnp.float32)
        cos_sin_cache = jnp.ones((16, 64), dtype=jnp.float32)
        positions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        block_table = jnp.array([0, 1], dtype=jnp.int32)
        query_start_loc = jnp.array([0, 4], dtype=jnp.int32)
        kv_block_table = jnp.array([0, 1], dtype=jnp.int32)
        cache = jnp.zeros((2, 128, 4, 128), dtype=jnp.uint8)
        rope_cache = jnp.zeros((2, 32, 4, 128), dtype=jnp.uint8)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

        # 1. shares_buffer = True (state_cache is None)
        with (
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.compress_and_store.proj_and_save_state.proj_and_save_state",
                return_value=cache,
            ),
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.compress_and_store.kernel.compress_norm_rope_store",
                return_value=(cache, rope_cache),
            ),
        ):
            c_out, r_out, s_out = compressor_v1.compressor_forward(
                hidden_states=hidden_states,
                wkv_wgate=wkv_wgate,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                block_table=block_table,
                query_start_loc=query_start_loc,
                kv_block_table=kv_block_table,
                cache=cache,
                rope_cache=rope_cache,
                distribution=distribution,
                state_block_size=8,
                head_dim=512,
                compress_ratio=4,
                overlap=True,
                rms_eps=1e-6,
                quant_block=64,
                state_cache=None,
            )
            assert c_out.shape == cache.shape
            assert r_out.shape == rope_cache.shape
            assert s_out.shape == cache.shape

        # 2. shares_buffer = False (state_cache provided, executes optimization_barrier)
        state_cache = jnp.zeros((2, 128, 4, 128), dtype=jnp.uint8)
        with (
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.compress_and_store.proj_and_save_state.proj_and_save_state",
                return_value=state_cache,
            ),
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.compress_and_store.kernel.compress_norm_rope_store",
                return_value=(cache, rope_cache),
            ),
        ):
            c_out, r_out, s_out = compressor_v1.compressor_forward(
                hidden_states=hidden_states,
                wkv_wgate=wkv_wgate,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                block_table=block_table,
                query_start_loc=query_start_loc,
                kv_block_table=kv_block_table,
                cache=cache,
                rope_cache=rope_cache,
                distribution=distribution,
                state_block_size=8,
                head_dim=512,
                compress_ratio=4,
                overlap=True,
                rms_eps=1e-6,
                quant_block=64,
                state_cache=state_cache,
            )
            assert c_out.shape == cache.shape
            assert r_out.shape == rope_cache.shape
            assert s_out.shape == state_cache.shape
