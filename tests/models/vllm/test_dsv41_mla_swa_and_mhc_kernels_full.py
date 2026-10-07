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

mla_swa.py and mhc/{fused_post_pre_kernel,post_kernel,pre_kernel,utils}.py.
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

from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa
from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    fused_post_pre_kernel,
    post_kernel,
    pre_kernel,
)
from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    utils as mhc_utils,
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
    """Mock buffer reference for simulating Pallas pipelines on CPU."""

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
    arr = np.asarray(arr)
    try:
        res = jnp.asarray(arr.view(dt))
    except Exception:
        res = jnp.asarray(arr).astype(dt)
    if (
        dt == jnp.bfloat16
        and arr.dtype == np.uint8
        and res.ndim == 4
        and res.shape[2] % 2 == 0
        and res.shape[2] > 2
    ):
        res = res[:, :, : res.shape[2] // 2, :]
    return res


def fake_fori_loop(lower, upper, body_fun, init_val, unroll=False):
    val = init_val
    for i in range(int(np.asarray(lower)), int(np.asarray(upper))):
        val = body_fun(i, val)
    return val


# ==============================================================================
# 1. MHC Utils Tests
# ==============================================================================


class TestMhcUtilsFull:
    """Tests 100% full-module line and branch coverage of mhc/utils.py."""

    def test_round_up(self):
        assert mhc_utils.round_up(15, 16) == 16
        assert mhc_utils.round_up(16, 16) == 16
        assert mhc_utils.round_up(17, 16) == 32
        assert mhc_utils.round_up(0, 16) == 0

    def test_select_token_block(self):
        # 1. Without vmem_need
        tb, pad = mhc_utils.select_token_block(10, 64)
        assert tb == 16
        assert pad == 16

        tb, pad = mhc_utils.select_token_block(100, 64)
        assert tb == 64
        assert pad == 128

        # 2. With vmem_need requiring block halving
        def vmem_need(b):
            return 200 if b > 16 else 50

        tb, pad = mhc_utils.select_token_block(
            128, 64, vmem_need=vmem_need, vmem_limit_bytes=100
        )
        assert tb == 16
        assert pad == 128

    def test_pad_to(self):
        a = jnp.ones((4, 8), dtype=jnp.float32)
        b = jnp.zeros((4, 16), dtype=jnp.bfloat16)

        # pad == 0
        res = mhc_utils.pad_to(4, a, b)
        assert res[0].shape == (4, 8)
        assert res[1].shape == (4, 16)

        # pad > 0
        res_pad = mhc_utils.pad_to(6, a, b)
        assert res_pad[0].shape == (6, 8)
        assert res_pad[1].shape == (6, 16)
        assert (res_pad[0][4:] == 0).all()

    def test_trim_to(self):
        a = jnp.ones((6, 8), dtype=jnp.float32)
        b = jnp.zeros((6, 16), dtype=jnp.bfloat16)

        # shape[0] == num_tokens
        res = mhc_utils.trim_to(6, a, b)
        assert res[0].shape == (6, 8)

        # shape[0] > num_tokens
        res_trim = mhc_utils.trim_to(4, a, b)
        assert res_trim[0].shape == (4, 8)
        assert res_trim[1].shape == (4, 16)

    def test_split_fn3(self):
        fn = jnp.ones((8, 64), dtype=jnp.float32) * 1.5
        fn_hi, fn_mid, fn_lo = mhc_utils.split_fn3(fn)
        assert fn_hi.shape == (8, 64)
        assert fn_mid.shape == (8, 64)
        assert fn_lo.shape == (8, 64)
        assert fn_hi.dtype == jnp.bfloat16
        assert fn_mid.dtype == jnp.bfloat16
        assert fn_lo.dtype == jnp.bfloat16

    def test_mhc_pre_gates(self):
        T = 4
        hc_mult = 4
        hidden_size = 64
        hc_mult3 = hc_mult * (hc_mult + 2)  # 24

        mixes = jnp.ones((T, hc_mult3), dtype=jnp.float32) * 0.1
        sqrsum = jnp.ones((T, 1), dtype=jnp.float32) * 10.0
        hc_scale = jnp.ones((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        # sinkhorn_repeat = 1
        pre_mix, post_mix, comb_mix = mhc_utils.mhc_pre_gates(
            mixes,
            sqrsum,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=1,
        )
        assert pre_mix.shape == (T, hc_mult)
        assert post_mix.shape == (T, hc_mult)
        assert comb_mix.shape == (T, hc_mult, hc_mult)

        # sinkhorn_repeat = 3 (exercises sinkhorn iteration loop)
        pre_mix2, post_mix2, comb_mix2 = mhc_utils.mhc_pre_gates(
            mixes,
            sqrsum,
            hc_mult,
            hidden_size,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=3,
        )
        assert comb_mix2.shape == (T, hc_mult, hc_mult)


# ==============================================================================
# 2. MHC Post Kernel Tests
# ==============================================================================


class TestMhcPostKernelFull:
    """Tests 100% full-module line and branch coverage of mhc/post_kernel.py."""

    def test_post_kernel_direct(self):
        tb = 4
        hc_mult = 2
        hidden_size = 8

        x_ref = MockArrayRef(np.ones((tb, hidden_size), dtype=np.float16))
        res_ref = MockArrayRef(
            np.ones((tb, hc_mult * hidden_size), dtype=np.float16) * 0.5
        )
        post_ref = MockArrayRef(np.ones((tb, hc_mult), dtype=np.float32) * 2.0)
        comb_ref = MockArrayRef(
            np.ones((tb, hc_mult * hc_mult), dtype=np.float32) * 0.1
        )
        out_ref = MockArrayRef(np.zeros((tb, hc_mult * hidden_size), dtype=np.float16))

        post_kernel._post_kernel(
            x_ref,
            res_ref,
            post_ref,
            comb_ref,
            out_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
        )
        assert np.isfinite(out_ref.arr).all()
        assert out_ref.arr.shape == (tb, hc_mult * hidden_size)

    def test_mhc_post_2d_assertions_and_execution(self):
        T = 8
        hc_mult = 4
        hidden_size = 32

        x = jnp.zeros((T, hidden_size), dtype=jnp.bfloat16)
        res2d = jnp.zeros((T, hc_mult * hidden_size), dtype=jnp.bfloat16)
        post_layer_mix = jnp.zeros((T, hc_mult), dtype=jnp.float32)
        comb_res_mix = jnp.zeros((T, hc_mult, hc_mult), dtype=jnp.float32)

        # Assertions
        with pytest.raises(AssertionError):
            post_kernel.mhc_post_2d(
                x,
                res2d.astype(jnp.float32),
                post_layer_mix,
                comb_res_mix,
            )
        with pytest.raises(AssertionError):
            post_kernel.mhc_post_2d(
                x,
                res2d,
                post_layer_mix.astype(jnp.bfloat16),
                comb_res_mix,
            )
        with pytest.raises(AssertionError):
            post_kernel.mhc_post_2d(
                x,
                res2d,
                post_layer_mix,
                comb_res_mix.astype(jnp.bfloat16),
            )

        # Successful execution with pallas_call mock
        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        with patch.object(post_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            out = post_kernel.mhc_post_2d(x, res2d, post_layer_mix, comb_res_mix)
            assert out.shape == (T, hc_mult * hidden_size)

            # Test padded tokens branch (T=10 with token_block_size=16)
            out_pad = post_kernel.mhc_post_2d(
                jnp.zeros((10, hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((10, hc_mult * hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((10, hc_mult), dtype=jnp.float32),
                jnp.zeros((10, hc_mult, hc_mult), dtype=jnp.float32),
                token_block_size=16,
            )
            assert out_pad.shape == (10, hc_mult * hidden_size)

            # Test vmem_need execution (tb=32 > SUBLANE=16)
            out_vmem = post_kernel.mhc_post_2d(
                jnp.zeros((32, hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((32, hc_mult * hidden_size), dtype=jnp.bfloat16),
                jnp.zeros((32, hc_mult), dtype=jnp.float32),
                jnp.zeros((32, hc_mult, hc_mult), dtype=jnp.float32),
                token_block_size=32,
            )
            assert out_vmem.shape == (32, hc_mult * hidden_size)

    def test_mhc_post_wrapper(self):
        B, S = 2, 4
        hc_mult = 4
        hidden_size = 32

        x = jnp.zeros((B, S, hidden_size), dtype=jnp.bfloat16)
        residual = jnp.zeros((B, S, hc_mult, hidden_size), dtype=jnp.bfloat16)
        post_layer_mix = jnp.zeros((B, S, hc_mult, 1), dtype=jnp.float32)
        comb_res_mix = jnp.zeros((B, S, hc_mult, hc_mult), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            post_kernel.mhc_post(
                x,
                residual.astype(jnp.float32),
                post_layer_mix,
                comb_res_mix,
            )

        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        with patch.object(post_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            out = post_kernel.mhc_post(x, residual, post_layer_mix, comb_res_mix)
            assert out.shape == (B, S, hc_mult, hidden_size)


# ==============================================================================
# 3. MHC Pre Kernel Tests
# ==============================================================================


class TestMhcPreKernelFull:
    """Tests 100% full-module line and branch coverage of mhc/pre_kernel.py."""

    def test_dot3_and_mixes_kernel_direct(self):
        T = 4
        K = 16
        M = 8

        x = jnp.ones((T, K), dtype=jnp.bfloat16)
        fn_hi = MockArrayRef(np.ones((M, K), dtype=np.float16) * 0.1)
        fn_mid = MockArrayRef(np.ones((M, K), dtype=np.float16) * 0.01)
        fn_lo = MockArrayRef(np.ones((M, K), dtype=np.float16) * 0.001)

        acc = pre_kernel._dot3(x, fn_hi, fn_mid, fn_lo)
        assert acc.shape == (T, M)

        x_ref = MockArrayRef(np.ones((T, K), dtype=np.float16))
        mixes_ref = MockArrayRef(np.zeros((T, M), dtype=np.float32))
        sqrsum_ref = MockArrayRef(np.zeros((T, 1), dtype=np.float32))

        pre_kernel._mixes_kernel(x_ref, fn_hi, fn_mid, fn_lo, mixes_ref, sqrsum_ref)
        assert mixes_ref.arr.shape == (T, M)
        assert sqrsum_ref.arr.shape == (T, 1)

    def test_mixes_collapse_kernel_direct(self):
        T = 4
        hc_mult = 2
        hidden_size = 8
        hc_hidden = hc_mult * hidden_size  # 16
        hc_mult3 = hc_mult * (hc_mult + 2)  # 8

        x_ref = MockArrayRef(np.ones((T, hc_hidden), dtype=np.float16))
        fn_hi_ref = MockArrayRef(np.ones((hc_mult3, hc_hidden), dtype=np.float16))
        fn_mid_ref = MockArrayRef(np.zeros((hc_mult3, hc_hidden), dtype=np.float16))
        fn_lo_ref = MockArrayRef(np.zeros((hc_mult3, hc_hidden), dtype=np.float16))
        sc_ref = MockArrayRef(np.ones((1, 3), dtype=np.float32))
        hb_ref = MockArrayRef(np.zeros((1, hc_mult3), dtype=np.float32))
        mixes_ref = MockArrayRef(np.zeros((T, hc_mult3), dtype=np.float32))
        sqrsum_ref = MockArrayRef(np.zeros((T, 1), dtype=np.float32))
        layer_ref = MockArrayRef(np.zeros((T, hidden_size), dtype=np.float16))

        pre_kernel._mixes_collapse_kernel(
            x_ref,
            fn_hi_ref,
            fn_mid_ref,
            fn_lo_ref,
            sc_ref,
            hb_ref,
            mixes_ref,
            sqrsum_ref,
            layer_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
        )
        assert np.isfinite(mixes_ref.arr).all()
        assert np.isfinite(sqrsum_ref.arr).all()
        assert np.isfinite(layer_ref.arr).all()
        assert layer_ref.arr.shape == (T, hidden_size)

    def test_mhc_pre_mixes_assertions_and_execution(self):
        T = 8
        hc_mult = 4
        hidden_size = 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x2d = jnp.zeros((T, hc_hidden), dtype=jnp.bfloat16)
        fn = jnp.zeros((hc_mult3, hc_hidden), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes(x2d.astype(jnp.float32), fn)
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes(x2d, fn.astype(jnp.bfloat16))
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes(x2d, jnp.zeros((hc_mult3, hc_hidden + 1)))

        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            mixes, sqrsum = pre_kernel.mhc_pre_mixes(x2d, fn)
            assert mixes.shape == (T, hc_mult3)
            assert sqrsum.shape == (T, 1)

            # Exercise _vmem_need (tb=32 > SUBLANE=16)
            m32, s32 = pre_kernel.mhc_pre_mixes(
                jnp.zeros((32, hc_hidden), dtype=jnp.bfloat16),
                fn,
                token_block_size=32,
            )
            assert m32.shape == (32, hc_mult3)
            assert s32.shape == (32, 1)

    def test_mhc_pre_mixes_collapse_assertions_and_execution(self):
        T = 8
        hc_mult = 4
        hidden_size = 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x2d = jnp.zeros((T, hc_hidden), dtype=jnp.bfloat16)
        fn = jnp.zeros((hc_mult3, hc_hidden), dtype=jnp.float32)
        hc_scale = jnp.zeros((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes_collapse(
                x2d.astype(jnp.float32),
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes_collapse(
                x2d,
                fn.astype(jnp.bfloat16),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre_mixes_collapse(
                x2d,
                jnp.zeros((hc_mult3, hc_hidden + 1)),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )

        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            mixes, sqrsum, layer2d = pre_kernel.mhc_pre_mixes_collapse(
                x2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
            assert mixes.shape == (T, hc_mult3)
            assert sqrsum.shape == (T, 1)
            assert layer2d.shape == (T, hidden_size)

            # Exercise _vmem_need (tb=32 > SUBLANE=16)
            m32, s32, l32 = pre_kernel.mhc_pre_mixes_collapse(
                jnp.zeros((32, hc_hidden), dtype=jnp.bfloat16),
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                token_block_size=32,
            )
            assert m32.shape == (32, hc_mult3)
            assert s32.shape == (32, 1)
            assert l32.shape == (32, hidden_size)

    def test_mhc_pre_wrapper(self):
        B, S = 2, 4
        hc_mult = 4
        hidden_size = 32
        hc_mult3 = hc_mult * (hc_mult + 2)

        residual = jnp.zeros((B, S, hc_mult, hidden_size), dtype=jnp.bfloat16)
        fn = jnp.zeros((hc_mult3, hc_mult * hidden_size), dtype=jnp.float32)
        hc_scale = jnp.zeros((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre(
                residual.astype(jnp.float32),
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )
        with pytest.raises(AssertionError):
            pre_kernel.mhc_pre(
                residual,
                fn.astype(jnp.bfloat16),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )

        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call):
            post_mix, comb_mix, layer = pre_kernel.mhc_pre(
                residual,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )
            assert post_mix.shape == (B, S, hc_mult, 1)
            assert comb_mix.shape == (B, S, hc_mult, hc_mult)
            assert layer.shape == (B, S, hidden_size)


# ==============================================================================
# 4. MHC Fused Post-Pre Kernel Tests
# ==============================================================================


class TestMhcFusedPostPreKernelFull:
    """Tests 100% line & branch coverage of mhc/fused_post_pre_kernel.py."""

    def test_fused_kernel_direct_highest_and_default(self):
        tb = 4
        hc_mult = 2
        hidden_size = 8
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x_ref = MockArrayRef(np.ones((tb, hidden_size), dtype=np.float16))
        res_ref = MockArrayRef(np.ones((tb, hc_hidden), dtype=np.float16))
        post_ref = MockArrayRef(np.ones((tb, hc_mult), dtype=np.float32))
        comb_ref = MockArrayRef(np.ones((tb, hc_mult * hc_mult), dtype=np.float32))
        fn_hi_ref = MockArrayRef(np.ones((hc_mult3, hc_hidden), dtype=np.float16))
        fn_mid_ref = MockArrayRef(
            np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.1
        )
        fn_lo_ref = MockArrayRef(
            np.ones((hc_mult3, hc_hidden), dtype=np.float16) * 0.01
        )
        sc_ref = MockArrayRef(np.ones((1, 3), dtype=np.float32))
        hb_ref = MockArrayRef(np.zeros((1, hc_mult3), dtype=np.float32))
        newres_ref = MockArrayRef(np.zeros((tb, hc_hidden), dtype=np.float16))
        mixes_ref = MockArrayRef(np.zeros((tb, hc_mult3), dtype=np.float32))
        sqrsum_ref = MockArrayRef(np.zeros((tb, 1), dtype=np.float32))
        layer_ref = MockArrayRef(np.zeros((tb, hidden_size), dtype=np.float16))

        # 1. gemm_precision == "highest" (exercises lines 78-83)
        fused_post_pre_kernel._fused_kernel(
            x_ref,
            res_ref,
            post_ref,
            comb_ref,
            fn_hi_ref,
            fn_mid_ref,
            fn_lo_ref,
            sc_ref,
            hb_ref,
            newres_ref,
            mixes_ref,
            sqrsum_ref,
            layer_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision="highest",
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
        )
        assert np.isfinite(newres_ref.arr).all()
        assert np.isfinite(mixes_ref.arr).all()
        assert np.isfinite(layer_ref.arr).all()

        # 2. gemm_precision == "default" (skips lines 78-83)
        fused_post_pre_kernel._fused_kernel(
            x_ref,
            res_ref,
            post_ref,
            comb_ref,
            fn_hi_ref,
            fn_mid_ref,
            fn_lo_ref,
            sc_ref,
            hb_ref,
            newres_ref,
            mixes_ref,
            sqrsum_ref,
            layer_ref,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            gemm_precision="default",
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
        )
        assert np.isfinite(newres_ref.arr).all()

    def test_fused_post_pre_mixes_assertions_and_execution(self):
        T = 8
        hc_mult = 4
        hidden_size = 32
        hc_hidden = hc_mult * hidden_size
        hc_mult3 = hc_mult * (hc_mult + 2)

        x2d = jnp.zeros((T, hidden_size), dtype=jnp.bfloat16)
        res2d = jnp.zeros((T, hc_hidden), dtype=jnp.bfloat16)
        post2d = jnp.zeros((T, hc_mult), dtype=jnp.float32)
        comb2d = jnp.zeros((T, hc_mult * hc_mult), dtype=jnp.float32)
        fn = jnp.zeros((hc_mult3, hc_hidden), dtype=jnp.float32)
        hc_scale = jnp.zeros((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        # Assertions
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d.astype(jnp.float32),
                res2d,
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                res2d.astype(jnp.float32),
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                res2d,
                post2d.astype(jnp.bfloat16),
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                res2d,
                post2d,
                comb2d.astype(jnp.bfloat16),
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                res2d,
                post2d,
                comb2d,
                fn.astype(jnp.bfloat16),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                jnp.zeros((T + 1, hc_hidden), dtype=jnp.bfloat16),
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                res2d,
                post2d,
                comb2d,
                jnp.zeros((hc_mult3, hc_hidden + 1)),
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
            )
        with pytest.raises(AssertionError):
            fused_post_pre_kernel.fused_post_pre_mixes(
                x2d,
                res2d,
                post2d,
                comb2d,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                gemm_precision="invalid_precision",
            )

        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(
            fused_post_pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call
        ):
            new_res, mixes, sqrsum, layer2d = (
                fused_post_pre_kernel.fused_post_pre_mixes(
                    x2d,
                    res2d,
                    post2d,
                    comb2d,
                    fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    gemm_precision="highest",
                )
            )
            assert new_res.shape == (T, hc_hidden)
            assert mixes.shape == (T, hc_mult3)
            assert sqrsum.shape == (T, 1)
            assert layer2d.shape == (T, hidden_size)

    def test_mhc_fused_post_pre_wrapper(self):
        B, S = 2, 4
        hc_mult = 4
        hidden_size = 32
        hc_mult3 = hc_mult * (hc_mult + 2)

        x = jnp.zeros((B, S, hidden_size), dtype=jnp.bfloat16)
        residual = jnp.zeros((B, S, hc_mult, hidden_size), dtype=jnp.bfloat16)
        post_layer_mix = jnp.zeros((B, S, hc_mult), dtype=jnp.float32)
        comb_res_mix = jnp.zeros((B, S, hc_mult, hc_mult), dtype=jnp.float32)
        fn = jnp.zeros((hc_mult3, hc_mult * hidden_size), dtype=jnp.float32)
        hc_scale = jnp.zeros((3,), dtype=jnp.float32)
        hc_base = jnp.zeros((hc_mult3,), dtype=jnp.float32)

        with pytest.raises(AssertionError):
            fused_post_pre_kernel.mhc_fused_post_pre(
                x,
                residual.astype(jnp.float32),
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                2,
            )

        def fake_pallas_call(fn, out_shape=None, *args, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(
            fused_post_pre_kernel.pl, "pallas_call", side_effect=fake_pallas_call
        ):
            new_res, post_mix, comb_mix, layer_in = (
                fused_post_pre_kernel.mhc_fused_post_pre(
                    x,
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    1e-6,
                    2.0,
                    2,
                )
            )
            assert new_res.shape == (B, S, hc_mult, hidden_size)
            assert post_mix.shape == (B, S, hc_mult, 1)
            assert comb_mix.shape == (B, S, hc_mult, hc_mult)
            assert layer_in.shape == (B, S, hidden_size)


# ==============================================================================
# 5. MLA SWA Helpers and Outer Execution Tests
# ==============================================================================


class TestMlaSwaHelpersAndOuter:
    """Tests helpers and outer functions in mla_swa.py."""

    def test_cdiv_and_align_to(self):
        assert mla_swa.cdiv(10, 4) == 3
        assert mla_swa.cdiv(8, 4) == 2
        assert mla_swa.align_to(10, 4) == 12
        assert mla_swa.align_to(8, 4) == 8
        with pytest.raises(AssertionError):
            mla_swa.cdiv(10, 0)

    def test_dtype_bitwidth_and_packing(self):
        assert mla_swa.get_dtype_bitwidth(jnp.float32) == 32
        assert mla_swa.get_dtype_bitwidth(jnp.bfloat16) == 16
        assert mla_swa.get_dtype_bitwidth(jnp.uint8) == 8
        assert mla_swa.get_dtype_packing(jnp.float32) == 1
        assert mla_swa.get_dtype_packing(jnp.bfloat16) == 2
        assert mla_swa.get_dtype_packing(jnp.uint8) == 4

    def test_mla_case_enum(self):
        assert mla_swa.MlaCase.DECODE.value == 0
        assert mla_swa.MlaCase.PREFILL.value == 1
        assert mla_swa.MlaCase.MIXED.value == 2

        assert mla_swa.MlaCase.DECODE.symbol == "d"
        assert mla_swa.MlaCase.PREFILL.symbol == "p"
        assert mla_swa.MlaCase.MIXED.symbol == "m"

    def test_prepare_q_inputs(self):
        q = jnp.zeros((8, 15, 100), dtype=jnp.bfloat16)
        out = mla_swa.prepare_q_inputs(q)
        # packing for bf16 is 2 -> align_to(15, 2) = 16
        # align_to(100, 128) = 128
        assert out.shape == (8, 16, 128)

    def test_prepare_kv_inputs(self):
        with pytest.raises(AssertionError):
            mla_swa.prepare_kv_inputs(jnp.zeros((4, 128), dtype=jnp.float32))
        with pytest.raises(AssertionError):
            mla_swa.prepare_kv_inputs(jnp.zeros((4, 100), dtype=jnp.bfloat16))

        kv = jnp.ones((4, 256), dtype=jnp.bfloat16)
        out = mla_swa.prepare_kv_inputs(kv)
        assert out.shape == (4, 512)
        assert out.dtype == jnp.uint8

    def test_prepare_outputs(self):
        out = jnp.zeros((8, 16, 128), dtype=jnp.bfloat16)
        trimmed = mla_swa.prepare_outputs(out, 15, 100)
        assert trimmed.shape == (8, 15, 100)

    def test_get_physical_layout(self):
        cache_kv_shape = (16, 32, 4, 128)
        layout = mla_swa.get_physical_layout(
            cache_kv_shape,
            jnp.bfloat16,
            head_dim=512,
            max_num_seqs=4,
            num_page_indices=16,
        )
        assert len(layout) == 6
        assert layout[0] == 16  # total_num_pages
        assert layout[1] == 4  # kv_packing
        assert layout[2] == 128  # lkv_dim
        assert layout[5] == 4  # pages_per_seq

    def test_run_mla_batched_decode_kernel(self):
        def fake_pallas_call(*args, out_shape=None, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(mla_swa.pl, "pallas_call", side_effect=fake_pallas_call):
            T = 8
            H = 16
            D = 256
            q = jnp.zeros((T, H, D), dtype=jnp.bfloat16)
            new_kv = jnp.zeros((T, 4, 128), dtype=jnp.uint8)
            cache_kv = jnp.zeros((4, 16, 4, 128), dtype=jnp.uint8)
            kv_lens = jnp.array([10, 10, 10, 10], dtype=jnp.int32)
            page_indices = jnp.zeros((16,), dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
            in_output = jnp.zeros((T, H, D), dtype=jnp.bfloat16)
            in_l = jnp.zeros((T, 128), dtype=jnp.float32)
            in_m = jnp.zeros((T, 128), dtype=jnp.float32)
            attention_sinks = jnp.zeros((H,), dtype=jnp.float32)

            out, up_kv, l_val, m_val = mla_swa.run_mla_batched_decode_kernel(
                q,
                new_kv,
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
                start_seq_idx=jnp.array(0),
                end_seq_idx=jnp.array(4),
                in_output=in_output,
                in_l=in_l,
                in_m=in_m,
                attention_sinks=attention_sinks,
                sm_scale=1.0,
                sliding_window=64,
                logical_page_size=16,
                vmem_limit_bytes=100 * 1024 * 1024,
                unnormalized_output=False,
            )
            assert out.shape == q.shape
            assert up_kv.shape == cache_kv.shape
            assert l_val is not None
            assert m_val is not None

    def test_mla_sliding_window_ragged_paged_attention_outer(self):
        T = 8
        H = 16
        D = 256
        max_num_seqs = 4
        pages_per_seq = 4

        q = jnp.zeros((T, H, D), dtype=jnp.bfloat16)
        new_kv = jnp.zeros((T, D), dtype=jnp.bfloat16)
        cache_kv = jnp.zeros((16, 16, 4, 128), dtype=jnp.uint8)
        kv_lens = jnp.array([10] * max_num_seqs, dtype=jnp.int32)
        page_indices = jnp.zeros((max_num_seqs * pages_per_seq,), dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
        distribution = jnp.array([2, 3, 4], dtype=jnp.int32)
        attention_sinks = jnp.zeros((H,), dtype=jnp.float32)

        # 1. Validation errors
        with pytest.raises(ValueError):
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=64,
                logical_page_size=16,
                num_kv_pages_per_block=None,
                num_queries_per_block=16,
            )
        with pytest.raises(ValueError):
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=64,
                logical_page_size=16,
                num_kv_pages_per_block=1,
                num_queries_per_block=None,
            )

        # 2. Assertions
        with pytest.raises(AssertionError):
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv.astype(jnp.float32),
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=64,
                logical_page_size=16,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
            )
        with pytest.raises(AssertionError):
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_kv.astype(jnp.float32),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=64,
                logical_page_size=16,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
            )
        with pytest.raises(AssertionError):
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=64,
                logical_page_size=100,  # exceeds phys_tokens_per_page
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
            )
        with pytest.raises(AssertionError):
            mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_kv,
                kv_lens,
                jnp.zeros((15,), dtype=jnp.int32),  # not divisible by 4
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=64,
                logical_page_size=16,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
            )

        def fake_pallas_call(*args, out_shape=None, **kwargs):
            return lambda *call_args: tuple(
                jnp.zeros(s.shape, dtype=s.dtype) for s in out_shape
            )

        with patch.object(mla_swa.pl, "pallas_call", side_effect=fake_pallas_call):
            # Case 1: sliding_window <= bkv_sz (batched decode)
            # and chunk_prefill_size is not None
            out1, up_kv1, l1, m1 = mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_kv,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=16,
                logical_page_size=16,
                num_kv_pages_per_block=1,  # int
                num_queries_per_block=16,  # int
                chunk_prefill_size=4,
                unnormalized_output=False,
            )
            assert out1.shape == (T, H, D)
            assert l1.shape == (T, H)
            assert m1.shape == (T, H)

            # Case 2: sliding_window > bkv_sz (run_mla_kernel DECODE),
            # chunk_prefill_size is None, tuple params, non_causal_block
            cache_int32 = jnp.zeros((16, 16, 128), dtype=jnp.int32)
            out2, up_kv2, l2, m2 = mla_swa.mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache_int32,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                sliding_window=100,
                logical_page_size=16,
                num_kv_pages_per_block=(1, 1, 1),
                num_queries_per_block=(16, 16, 16),
                chunk_prefill_size=None,
                unnormalized_output=True,
                non_causal_block=True,
            )
            assert out2.shape == (T, H, D)


# ==============================================================================
# 6. MLA SWA Kernel Direct Tests
# ==============================================================================


class TestMlaSwaKernelDirect:
    """Tests _mla_sliding_window_ragged_paged_attention_kernel directly."""

    def test_kernel_assertions(self):
        max_num_seqs = 2
        pages_per_seq = 2
        num_tokens = 4
        num_q_heads = 16
        head_dim = 128
        total_num_pages = 4
        phys_page_size = 16

        kv_lens_ref = MockArrayRef(np.array([10, 10], dtype=np.int32))
        page_indices_ref = MockArrayRef(
            np.zeros((max_num_seqs * pages_per_seq,), dtype=np.int32)
        )
        cu_q_lens_ref = MockArrayRef(np.array([0, 2, 4], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 2], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.zeros((4,), dtype=np.int32))
        bkv_update_ids_ref = MockArrayRef(np.zeros((6,), dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        new_kv_hbm_ref = MockArrayRef(np.zeros((num_tokens, 2, 128), dtype=np.uint8))
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 4, 128), dtype=np.uint8)
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        in_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        in_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 4, 128), dtype=np.uint8)
        )
        l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        bkv_x2_ref = MockArrayRef(np.zeros((2, 16, 2, 128), dtype=np.uint8))
        bq_x2_ref = MockArrayRef(
            np.zeros((2, 2, num_q_heads, head_dim), dtype=np.float16)
        )
        bo_x2_ref = MockArrayRef(
            np.zeros((2, 2, num_q_heads, head_dim), dtype=np.float16)
        )
        bl_x2_ref = MockArrayRef(np.zeros((2, 2, 128), dtype=np.float32))
        bm_x2_ref = MockArrayRef(np.zeros((2, 2, 128), dtype=np.float32))
        sems = MockArrayRef(np.zeros((6, 2), dtype=np.int32))
        l_ref = MockArrayRef(np.zeros((2 * num_q_heads,), dtype=np.float32))
        m_ref = MockArrayRef(np.zeros((2 * num_q_heads,), dtype=np.float32))
        acc_ref = MockArrayRef(np.zeros((2 * num_q_heads, head_dim), dtype=np.float32))

        # 1. q_hbm_ref.shape != o_hbm_ref.shape
        bad_o = MockArrayRef(
            np.zeros((num_tokens, num_q_heads + 1, head_dim), dtype=np.float16)
        )
        with pytest.raises(AssertionError):
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                bkv_update_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                bad_o,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                static_q_len=1,
                sm_scale=1.0,
                sliding_window=16,
                logical_page_size=16,
                unnormalized_output=False,
                non_causal_block=False,
                q_compute_block_size=None,
                bkv_p=1,
                bq_sz=2,
            )

        # 2. sliding_window <= 0
        with pytest.raises(AssertionError):
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                bkv_update_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                static_q_len=1,
                sm_scale=1.0,
                sliding_window=0,
                logical_page_size=16,
                unnormalized_output=False,
                non_causal_block=False,
                q_compute_block_size=None,
                bkv_p=1,
                bq_sz=2,
            )

    def test_kernel_simulation_decode_and_non_causal(self):
        num_tokens = 4
        num_q_heads = 128
        head_dim = 128
        total_num_pages = 4
        phys_page_size = 16
        bkv_sz = 16
        bq_sz = 2

        kv_lens_ref = MockArrayRef(np.array([10, 12], dtype=np.int32))
        page_indices_ref = MockArrayRef(np.array([0, 1, 2, 3], dtype=np.int32))
        cu_q_lens_ref = MockArrayRef(np.array([0, 2, 4], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 1], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.zeros((4,), dtype=np.int32))
        bkv_update_ids_ref = MockArrayRef(np.zeros((6,), dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        new_kv_hbm_ref = MockArrayRef(np.zeros((num_tokens, 2, 128), dtype=np.uint8))
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 2, 128), dtype=np.uint8)
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        in_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        in_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 2, 128), dtype=np.uint8)
        )
        l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        bkv_x2_ref = MockArrayRef(np.zeros((2, bkv_sz, 2, 128), dtype=np.uint8))
        bq_x2_ref = MockArrayRef(
            np.zeros((2, bq_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        bo_x2_ref = MockArrayRef(
            np.zeros((2, bq_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        bl_x2_ref = MockArrayRef(np.zeros((2, bq_sz, 128), dtype=np.float32))
        bm_x2_ref = MockArrayRef(np.zeros((2, bq_sz, 128), dtype=np.float32))
        sems = MockArrayRef(np.zeros((6, 2), dtype=np.int32))
        l_ref = MockArrayRef(np.zeros((bq_sz * num_q_heads,), dtype=np.float32))
        m_ref = MockArrayRef(np.zeros((bq_sz * num_q_heads,), dtype=np.float32))
        acc_ref = MockArrayRef(
            np.zeros((bq_sz * num_q_heads, head_dim), dtype=np.float32)
        )

        with (
            patch.object(
                mla_swa.pltpu, "make_async_copy", return_value=MockAsyncCopy()
            ),
            patch.object(mla_swa.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla_swa.pl,
                "reciprocal",
                side_effect=lambda x, approx=True: 1.0 / x,
            ),
            patch.object(mla_swa.pl, "when", side_effect=fake_when),
            patch.object(mla_swa.pl, "program_id", return_value=0),
            patch.object(mla_swa.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            # Run 1: single_bkv_block = True, static_q_len = 1, non_causal_block = True
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                bkv_update_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                static_q_len=1,
                sm_scale=1.0,
                sliding_window=15,
                logical_page_size=16,
                unnormalized_output=False,
                non_causal_block=True,
                q_compute_block_size=None,
                bkv_p=1,
                bq_sz=bq_sz,
            )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)

    def test_load_with_init_branches(self):
        import types

        load_code = None
        for const in (
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel.__code__.co_consts
        ):
            if hasattr(const, "co_name") and const.co_name == "flash_attention":
                for sub_const in const.co_consts:
                    if (
                        hasattr(sub_const, "co_name")
                        and sub_const.co_name == "load_with_init"
                    ):
                        load_code = sub_const
                        break
        assert load_code is not None

        def make_cell(val):
            return (lambda: val).__closure__[0]

        ref = jnp.zeros((4,), dtype=jnp.float32)

        # single_bkv_block = True (line 182)
        fn_true = types.FunctionType(
            load_code,
            mla_swa.__dict__,
            "load_with_init",
            None,
            (make_cell(0), make_cell(True)),
        )
        res_true = fn_true(ref, 2.0)
        assert jnp.all(res_true == 2.0)

        # single_bkv_block = False, bkv_idx = 0 (line 184 true branch)
        fn_false0 = types.FunctionType(
            load_code,
            mla_swa.__dict__,
            "load_with_init",
            None,
            (make_cell(0), make_cell(False)),
        )
        res_false0 = fn_false0(ref, 2.0)
        assert jnp.all(res_false0 == 2.0)

        # single_bkv_block = False, bkv_idx = 1 (line 184 false branch)
        fn_false1 = types.FunctionType(
            load_code,
            mla_swa.__dict__,
            "load_with_init",
            None,
            (make_cell(1), make_cell(False)),
        )
        res_false1 = fn_false1(ref, 2.0)
        assert jnp.all(res_false1 == 0.0)

    def test_kernel_simulation_multi_bkv_dynamic_q_len_and_unnormalized(self):
        max_num_seqs = 2
        pages_per_seq = 4
        num_tokens = 4
        num_q_heads = 16  # not divisible by 128 -> tests row by row copy
        head_dim = 128
        total_num_pages = 8
        phys_page_size = 16
        bkv_sz = 16
        bq_sz = 2

        kv_lens_ref = MockArrayRef(np.array([40, 40], dtype=np.int32))
        page_indices_ref = MockArrayRef(
            np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)
        )
        cu_q_lens_ref = MockArrayRef(np.array([0, 2, 4], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 2], dtype=np.int32))
        sem_ids_ref = MockArrayRef(np.zeros((3,), dtype=np.int32))
        bo_ids_ref = MockArrayRef(np.zeros((4,), dtype=np.int32))
        bkv_update_ids_ref = MockArrayRef(np.zeros((6,), dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float32)
        )
        new_kv_hbm_ref = MockArrayRef(np.zeros((num_tokens, 2, 128), dtype=np.uint8))
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 4, 128), dtype=np.uint8)
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
        )
        in_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        in_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
        )
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 4, 128), dtype=np.uint8)
        )
        l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        bkv_x2_ref = MockArrayRef(np.zeros((2, bkv_sz, 2, 128), dtype=np.uint8))
        bq_x2_ref = MockArrayRef(
            np.zeros((2, bq_sz, num_q_heads, head_dim), dtype=np.float32)
        )
        bo_x2_ref = MockArrayRef(
            np.zeros((2, bq_sz, num_q_heads, head_dim), dtype=np.float32)
        )
        bl_x2_ref = MockArrayRef(np.zeros((2, bq_sz, 128), dtype=np.float32))
        bm_x2_ref = MockArrayRef(np.zeros((2, bq_sz, 128), dtype=np.float32))
        sems = MockArrayRef(np.zeros((6, 2), dtype=np.int32))
        l_ref = MockArrayRef(np.zeros((bq_sz * num_q_heads,), dtype=np.float32))
        m_ref = MockArrayRef(np.zeros((bq_sz * num_q_heads,), dtype=np.float32))
        acc_ref = MockArrayRef(
            np.zeros((bq_sz * num_q_heads, head_dim), dtype=np.float32)
        )

        with (
            patch.object(
                mla_swa.pltpu, "make_async_copy", return_value=MockAsyncCopy()
            ),
            patch.object(mla_swa.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla_swa.pl,
                "reciprocal",
                side_effect=lambda x, approx=True: 1.0 / x,
            ),
            patch.object(mla_swa.pl, "when", side_effect=fake_when),
            patch.object(mla_swa.pl, "program_id", return_value=0),
            patch.object(mla_swa.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            # Run 2a: non_causal_block = False
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                bkv_update_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                static_q_len=None,
                sm_scale=1.0,
                sliding_window=64,
                logical_page_size=16,
                unnormalized_output=True,
                non_causal_block=False,
                q_compute_block_size=1,
                bkv_p=1,
                bq_sz=bq_sz,
            )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)

            # Run 2b: non_causal_block = True (exercises line 610)
            mla_swa._mla_sliding_window_ragged_paged_attention_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                sem_ids_ref,
                bo_ids_ref,
                bkv_update_ids_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_x2_ref,
                bq_x2_ref,
                bo_x2_ref,
                bl_x2_ref,
                bm_x2_ref,
                sems,
                l_ref,
                m_ref,
                acc_ref,
                static_q_len=None,
                sm_scale=1.0,
                sliding_window=64,
                logical_page_size=16,
                unnormalized_output=True,
                non_causal_block=True,
                q_compute_block_size=1,
                bkv_p=1,
                bq_sz=bq_sz,
            )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)


# ==============================================================================
# 7. MLA Batched Decode Kernel Direct Tests
# ==============================================================================


class TestMlaBatchedDecodeKernelDirect:
    """Tests _mla_batched_decode_kernel directly."""

    def test_batched_decode_kernel_loop_cnt_1_bf16(self):
        num_tokens = 2
        num_q_heads = 128
        head_dim = 128
        total_num_pages = 4
        phys_page_size = 16
        bkv_sz = 16
        seq_batch_sz = 2

        kv_lens_ref = MockArrayRef(np.array([10, 10], dtype=np.int32))
        page_indices_ref = MockArrayRef(np.array([0, 1, 2, 3], dtype=np.int32))
        cu_q_lens_ref = MockArrayRef(np.array([0, 1, 2], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 2], dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        new_kv_hbm_ref = MockArrayRef(np.zeros((num_tokens, 2, 128), dtype=np.uint8))
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 2, 128), dtype=np.uint8)
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        in_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        in_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float16)
        )
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 2, 128), dtype=np.uint8)
        )
        l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))

        bkv_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, bkv_sz, 2, 128), dtype=np.uint8)
        )
        q_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        o_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_q_heads, head_dim), dtype=np.float16)
        )
        l_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, 128), dtype=np.float32)
        )
        m_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, 128), dtype=np.float32)
        )
        sems = MockArrayRef(np.zeros((2, seq_batch_sz + 4), dtype=np.int32))

        with (
            patch.object(
                mla_swa.pltpu, "make_async_copy", return_value=MockAsyncCopy()
            ),
            patch.object(mla_swa.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla_swa.pl,
                "reciprocal",
                side_effect=lambda x, approx=True: 1.0 / x,
            ),
            patch.object(mla_swa.pl, "when", side_effect=fake_when),
            patch.object(mla_swa.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            mla_swa._mla_batched_decode_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_batch_x2_vmem,
                q_batch_x2_vmem,
                o_batch_x2_vmem,
                l_batch_x2_vmem,
                m_batch_x2_vmem,
                sems,
                sm_scale=1.0,
                sliding_window=16,
                logical_page_size=16,
                unnormalized_output=False,
                bkv_p=1,
                seq_batch_sz=seq_batch_sz,
            )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)

    def test_batched_decode_kernel_loop_cnt_2_f32_unnormalized(self):
        max_num_seqs = 4
        pages_per_seq = 2
        num_tokens = 4
        num_q_heads = 16  # not 128 -> tests padding of l/m
        head_dim = 128
        total_num_pages = 8
        phys_page_size = 16
        bkv_sz = 16
        seq_batch_sz = 2  # num_batches = 2 -> loop_cnt > 1

        kv_lens_ref = MockArrayRef(np.array([10, 10, 10, 10], dtype=np.int32))
        page_indices_ref = MockArrayRef(
            np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)
        )
        cu_q_lens_ref = MockArrayRef(np.array([0, 1, 2, 3, 4], dtype=np.int32))
        start_end_seq_idx_ref = MockArrayRef(np.array([0, 4], dtype=np.int32))
        attention_sinks_ref = MockArrayRef(np.zeros((num_q_heads,), dtype=np.float32))
        q_hbm_ref = MockArrayRef(
            np.ones((num_tokens, num_q_heads, head_dim), dtype=np.float32)
        )
        new_kv_hbm_ref = MockArrayRef(np.zeros((num_tokens, 4, 128), dtype=np.uint8))
        cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 4, 128), dtype=np.uint8)
        )
        in_output_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
        )
        in_l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        in_m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        o_hbm_ref = MockArrayRef(
            np.zeros((num_tokens, num_q_heads, head_dim), dtype=np.float32)
        )
        updated_cache_kv_hbm_ref = MockArrayRef(
            np.zeros((total_num_pages, phys_page_size, 4, 128), dtype=np.uint8)
        )
        l_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))
        m_hbm_ref = MockArrayRef(np.zeros((num_tokens, 128), dtype=np.float32))

        bkv_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, bkv_sz, 4, 128), dtype=np.uint8)
        )
        q_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_q_heads, head_dim), dtype=np.float32)
        )
        o_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, num_q_heads, head_dim), dtype=np.float32)
        )
        l_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, 128), dtype=np.float32)
        )
        m_batch_x2_vmem = MockArrayRef(
            np.zeros((2, seq_batch_sz, 128), dtype=np.float32)
        )
        sems = MockArrayRef(np.zeros((2, seq_batch_sz + 4), dtype=np.int32))

        with (
            patch.object(
                mla_swa.pltpu, "make_async_copy", return_value=MockAsyncCopy()
            ),
            patch.object(mla_swa.pltpu, "bitcast", side_effect=fake_bitcast),
            patch.object(
                mla_swa.pl,
                "reciprocal",
                side_effect=lambda x, approx=True: 1.0 / x,
            ),
            patch.object(mla_swa.pl, "when", side_effect=fake_when),
            patch.object(mla_swa.lax, "fori_loop", side_effect=fake_fori_loop),
        ):
            mla_swa._mla_batched_decode_kernel(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                start_end_seq_idx_ref,
                attention_sinks_ref,
                q_hbm_ref,
                new_kv_hbm_ref,
                cache_kv_hbm_ref,
                in_output_hbm_ref,
                in_l_hbm_ref,
                in_m_hbm_ref,
                o_hbm_ref,
                updated_cache_kv_hbm_ref,
                l_hbm_ref,
                m_hbm_ref,
                bkv_batch_x2_vmem,
                q_batch_x2_vmem,
                o_batch_x2_vmem,
                l_batch_x2_vmem,
                m_batch_x2_vmem,
                sems,
                sm_scale=1.0,
                sliding_window=16,
                logical_page_size=16,
                unnormalized_output=True,
                bkv_p=1,
                seq_batch_sz=seq_batch_sz,
            )
            assert o_hbm_ref.arr.shape == (num_tokens, num_q_heads, head_dim)
