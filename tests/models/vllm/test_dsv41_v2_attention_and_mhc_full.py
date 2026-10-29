# SPDX-License-Identifier: Apache-2.0
"""100% full-module line and branch coverage suite for DeepSeek-V4.1 .

Covers:
- `vllm_torchtpu.models.vllm.deepseek_v41.common.mhc` (lines 1-455)
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.attention` (lines 1-1606)
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure CPU backend and skip MDS query before importing torch/jax
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn


# ==============================================================================
# 1. Tests for common/mhc.py (Full 100% Coverage)
# ==============================================================================
class TestCommonMHCFull:
    """Full-module coverage tests for common/mhc.py (lines 1-455)."""

    def test_pre_gates(self):
        tokens = 4
        mult = 4
        dim = 16
        mixes_count = mult * (mult + 2)
        mixes = jnp.ones((tokens, mixes_count), dtype=jnp.float32)
        sqrsum = jnp.ones((tokens, 1), dtype=jnp.float32)
        hc_scale = jnp.ones(3, dtype=jnp.float32)
        hc_base = jnp.zeros(mixes_count, dtype=jnp.float32)
        pre_mix, post_mix, comb_mix = mhc._pre_gates(
            mixes, sqrsum, mult, dim, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 5
        )
        assert pre_mix.shape == (tokens, mult)
        assert post_mix.shape == (tokens, mult)
        assert comb_mix.shape == (tokens, mult, mult)

    def test_truncate_bf16_and_pack_mix_weight_torch_and_jax(self):
        fn_torch = torch.randn(24, 64, dtype=torch.float32)
        packed_torch = mhc.pack_mix_weight(fn_torch)
        assert packed_torch.shape == (64, 128)
        assert packed_torch.dtype == torch.bfloat16

        fn_jax = jnp.asarray(fn_torch.numpy())
        packed_jax = mhc._pack_mix_weight(fn_jax)
        assert packed_jax.shape == (64, 128)
        assert packed_jax.dtype == jnp.bfloat16

    def test_pack_mix_weights_mutation(self):
        # Case 1: TPU device and float32 -> packed
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

        # Case 2: CPU device -> no-op
        mock_cpu_fn = MagicMock()
        mock_cpu_fn.device = SimpleNamespace(type="cpu")
        mock_cpu_fn.dtype = torch.float32
        layer2 = SimpleNamespace(hc_attn_fn=mock_cpu_fn, hc_ffn_fn=mock_cpu_fn)
        mhc.pack_mix_weights([layer2])
        assert layer2.hc_attn_fn is mock_cpu_fn

        # Case 3: Non-float32 dtype -> no-op
        mock_bf16_fn = MagicMock()
        mock_bf16_fn.device = SimpleNamespace(type="tpu")
        mock_bf16_fn.dtype = torch.bfloat16
        layer3 = SimpleNamespace(hc_attn_fn=mock_bf16_fn, hc_ffn_fn=mock_bf16_fn)
        mhc.pack_mix_weights([layer3])
        assert layer3.hc_attn_fn is mock_bf16_fn

    def test_packed_mixes_and_delayed_pre_jax(self):
        tokens, mult, dim = 2, 4, 8
        mixes = mult * (mult + 2)
        residual = jnp.ones((tokens, mult, dim), dtype=jnp.bfloat16)
        x = jnp.ones((tokens, mult * dim), dtype=jnp.bfloat16)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes, dtype=jnp.float32)
        prev_pre = jnp.ones((tokens, mult), dtype=jnp.float32)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 5)

        # Branch 1: fn is BF16 packed weight -> calls _packed_mixes
        fn_bf16 = jnp.ones((mult * dim, 128), dtype=jnp.bfloat16)
        post1, comb1, val1, npre1 = mhc.delayed_pre_jax(
            residual, x, fn_bf16, scale, base, prev_pre, *constants
        )
        assert post1.shape == (tokens, mult, 1)
        assert comb1.shape == (tokens, mult, mult)
        assert val1.shape == (tokens, dim)
        assert npre1.shape == (tokens, mult)

        # Branch 2: fn is FP32 unpacked weight -> calls jnp.matmul
        fn_f32 = jnp.ones((mixes, mult * dim), dtype=jnp.float32)
        post2, comb2, val2, npre2 = mhc.delayed_pre_jax(
            residual, x, fn_f32, scale, base, prev_pre, *constants
        )
        assert post2.shape == (tokens, mult, 1)

    def test_flat_pre_sums_and_delayed_pre_flat_jax(self, monkeypatch):
        # 1. tokens < 16 -> takes delayed_pre_jax path
        mult, dim = 4, 8
        mixes = mult * (mult + 2)
        flat_small = jnp.ones((8, mult * dim), dtype=jnp.bfloat16)
        fn_f32 = jnp.ones((mixes, mult * dim), dtype=jnp.float32)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes, dtype=jnp.float32)
        prev_pre_small = jnp.ones((8, mult), dtype=jnp.float32)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 5)

        p1, c1, v1, np1 = mhc.delayed_pre_flat_jax(
            flat_small, fn_f32, scale, base, prev_pre_small, *constants
        )
        assert p1.shape == (8, mult, 1)

        # 2. tokens >= 16 and tokens % 8 == 0 with mock pallas_call to execute kernel
        captured_kernel_calls = []

        class MockRef:
            def __init__(self, arr):
                self.arr = arr
                self.dtype = arr.dtype

            def __getitem__(self, _):
                return self.arr

            def __setitem__(self, _, val):
                self.arr = val

        def mock_pallas_call(
            kernel, grid, in_specs, out_specs, out_shape, compiler_params
        ):
            captured_kernel_calls.append(compiler_params)
            for spec in (in_specs or []) + (out_specs or []):
                if spec is not None and callable(getattr(spec, "index_map", None)):
                    spec.index_map(0)

            def invoke(flat_in, weight_in, pre_in):
                t_count, width = flat_in.shape
                tile_sz = in_specs[0].block_shape[0]
                m_count = pre_in.shape[1]
                d = width // m_count
                mx = m_count * (m_count + 2)

                mix_buf = []
                sqr_buf = []
                val_buf = []

                for i in range(grid[0]):
                    x_ref = MockRef(flat_in[i * tile_sz : (i + 1) * tile_sz])
                    w_ref = MockRef(weight_in)
                    pre_ref = MockRef(pre_in[i * tile_sz : (i + 1) * tile_sz])
                    mix_ref = MockRef(jnp.zeros((tile_sz, mx), dtype=jnp.float32))
                    sqr_ref = MockRef(jnp.zeros((tile_sz, 1), dtype=jnp.float32))
                    val_ref = MockRef(jnp.zeros((tile_sz, d), dtype=flat_in.dtype))

                    kernel(x_ref, w_ref, pre_ref, mix_ref, sqr_ref, val_ref)

                    mix_buf.append(mix_ref.arr)
                    sqr_buf.append(sqr_ref.arr)
                    val_buf.append(val_ref.arr)

                return (
                    jnp.concatenate(mix_buf, axis=0),
                    jnp.concatenate(sqr_buf, axis=0),
                    jnp.concatenate(val_buf, axis=0),
                )

            return invoke

        monkeypatch.setattr("jax.experimental.pallas.pallas_call", mock_pallas_call)

        # Case A: tokens = 128 (tokens % 128 == 0 -> tile = 128), fn is BF16
        flat_128 = jnp.ones((128, mult * dim), dtype=jnp.bfloat16)
        fn_bf16 = jnp.ones((mult * dim, 128), dtype=jnp.bfloat16)
        prev_128 = jnp.ones((128, mult), dtype=jnp.float32)

        p128, c128, v128, np128 = mhc.delayed_pre_flat_jax(
            flat_128, fn_bf16, scale, base, prev_128, *constants
        )
        assert p128.shape == (128, mult, 1)
        assert len(captured_kernel_calls) == 1

        # Case B: tokens = 64, 96, 256, 1024, fn is FP32
        for t_pallas in (64, 96, 256, 1024):
            flat_p = jnp.ones((t_pallas, mult * dim), dtype=jnp.bfloat16)
            prev_p = jnp.ones((t_pallas, mult), dtype=jnp.float32)
            pp, _, _, _ = mhc.delayed_pre_flat_jax(
                flat_p, fn_f32, scale, base, prev_p, *constants
            )
            assert pp.shape == (t_pallas, mult, 1)
        assert len(captured_kernel_calls) == 5

        # Case C: tokens = 16, 24, 144, 192, 288, 384, 512 take XLA path
        for t_case in (16, 24, 144, 192, 288, 384, 512):
            flat_t = jnp.ones((t_case, mult * dim), dtype=jnp.bfloat16)
            prev_t = jnp.ones((t_case, mult), dtype=jnp.float32)
            pt, _, _, _ = mhc.delayed_pre_flat_jax(
                flat_t, fn_bf16, scale, base, prev_t, *constants
            )
            assert pt.shape == (t_case, mult, 1)
        assert len(captured_kernel_calls) == 5

        # Case D: mult != 4 in _collapse_flat (e.g. mult = 2)
        flat_m2 = jnp.ones((8, 2 * dim), dtype=jnp.bfloat16)
        fn_m2 = jnp.ones((2 * 4, 2 * dim), dtype=jnp.float32)
        base_m2 = jnp.zeros(2 * 4, dtype=jnp.float32)
        prev_m2 = jnp.ones((8, 2), dtype=jnp.float32)
        pm2, _, _, _ = mhc.delayed_pre_flat_jax(
            flat_m2, fn_m2, scale, base_m2, prev_m2, *constants
        )
        assert pm2.shape == (8, 2, 1)

    def test_flat_post_pre_sums_and_delayed_post_pre_flat_jax(self, monkeypatch):
        mult, dim = 4, 8
        mixes = mult * (mult + 2)
        x_small = jnp.ones((8, dim), dtype=jnp.bfloat16)
        flat_small = jnp.ones((8, mult * dim), dtype=jnp.bfloat16)
        post_small = jnp.ones((8, mult, 1), dtype=jnp.float32)
        comb_small = jnp.ones((8, mult, mult), dtype=jnp.float32)
        fn_f32 = jnp.ones((mixes, mult * dim), dtype=jnp.float32)
        scale = jnp.ones(3, dtype=jnp.float32)
        base = jnp.zeros(mixes, dtype=jnp.float32)
        prev_pre_small = jnp.ones((8, mult), dtype=jnp.float32)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 5)

        # 0. Direct _post_jax check
        post_out = mhc._post_jax(x_small, flat_small, post_small, comb_small)
        assert post_out.shape == (8, mult * dim)

        # 1. tokens < 64 -> takes _post_and_collapse_jax + delayed_pre_jax path
        f1, p1, c1, v1, np1 = mhc.delayed_post_pre_flat_jax(
            x_small,
            flat_small,
            post_small,
            comb_small,
            fn_f32,
            scale,
            base,
            prev_pre_small,
            *constants,
        )
        assert f1.shape == (8, mult * dim)
        assert p1.shape == (8, mult, 1)

        # 1b. mult != 4 in _post_and_collapse_jax (e.g. mult = 2)
        fm2, pm2, _, _, _ = mhc.delayed_post_pre_flat_jax(
            x_small,
            jnp.ones((8, 2 * dim), dtype=jnp.bfloat16),
            jnp.ones((8, 2, 1), dtype=jnp.float32),
            jnp.ones((8, 2, 2), dtype=jnp.float32),
            jnp.ones((2 * 4, 2 * dim), dtype=jnp.float32),
            scale,
            jnp.zeros(2 * 4, dtype=jnp.float32),
            jnp.ones((8, 2), dtype=jnp.float32),
            *constants,
        )
        assert fm2.shape == (8, 2 * dim)
        assert pm2.shape == (8, 2, 1)

        # 2. Pallas path (tokens in (64, 96, 128, 256) or >= 1024) with mock pallas_call
        captured_kernel_calls = []

        class MockRef:
            def __init__(self, arr):
                self.arr = arr
                self.dtype = arr.dtype

            def __getitem__(self, idx):
                return self.arr if idx is Ellipsis else self.arr[idx]

            def __setitem__(self, idx, val):
                if idx is Ellipsis:
                    self.arr = val
                else:
                    self.arr = self.arr.at[idx].set(val)

        def mock_pallas_call(
            kernel, grid, in_specs, out_specs, out_shape, compiler_params
        ):
            captured_kernel_calls.append(compiler_params)
            for spec in (in_specs or []) + (out_specs or []):
                if spec is not None and callable(getattr(spec, "index_map", None)):
                    spec.index_map(0)

            def invoke(sub_x_in, flat_in, post_in, comb_in, weight_in, pre_in):
                _, width = flat_in.shape
                tile_sz = in_specs[0].block_shape[0]
                m_count = pre_in.shape[1]
                d = width // m_count
                mx = m_count * (m_count + 2)

                res_buf = []
                mix_buf = []
                sqr_buf = []
                val_buf = []

                for i in range(grid[0]):
                    sl = slice(i * tile_sz, (i + 1) * tile_sz)
                    sub_x_ref = MockRef(sub_x_in[sl])
                    res_ref = MockRef(flat_in[sl])
                    post_ref = MockRef(post_in[sl])
                    comb_ref = MockRef(comb_in[sl])
                    w_ref = MockRef(weight_in)
                    pre_ref = MockRef(pre_in[sl])
                    out_res_ref = MockRef(
                        jnp.zeros((tile_sz, width), dtype=flat_in.dtype)
                    )
                    mix_ref = MockRef(jnp.zeros((tile_sz, mx), dtype=jnp.float32))
                    sqr_ref = MockRef(jnp.zeros((tile_sz, 1), dtype=jnp.float32))
                    val_ref = MockRef(jnp.zeros((tile_sz, d), dtype=flat_in.dtype))

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

                    res_buf.append(out_res_ref.arr)
                    mix_buf.append(mix_ref.arr)
                    sqr_buf.append(sqr_ref.arr)
                    val_buf.append(val_ref.arr)

                return (
                    jnp.concatenate(res_buf, axis=0),
                    jnp.concatenate(mix_buf, axis=0),
                    jnp.concatenate(sqr_buf, axis=0),
                    jnp.concatenate(val_buf, axis=0),
                )

            return invoke

        monkeypatch.setattr("jax.experimental.pallas.pallas_call", mock_pallas_call)

        # Case A: tokens = 128 (tile = 128), fn is BF16
        x_128 = jnp.ones((128, dim), dtype=jnp.bfloat16)
        flat_128 = jnp.ones((128, mult * dim), dtype=jnp.bfloat16)
        post_128 = jnp.ones((128, mult, 1), dtype=jnp.float32)
        comb_128 = jnp.ones((128, mult, mult), dtype=jnp.float32)
        fn_bf16 = jnp.ones((mult * dim, 128), dtype=jnp.bfloat16)
        prev_128 = jnp.ones((128, mult), dtype=jnp.float32)

        f128, p128, c128, v128, np128 = mhc.delayed_post_pre_flat_jax(
            x_128,
            flat_128,
            post_128,
            comb_128,
            fn_bf16,
            scale,
            base,
            prev_128,
            *constants,
        )
        assert f128.shape == (128, mult * dim)
        assert p128.shape == (128, mult, 1)
        assert len(captured_kernel_calls) == 1

        # Case B: tokens = 64, 96, 256, 1024, fn is FP32
        for t_pallas in (64, 96, 256, 1024):
            fp, pp, _, _, _ = mhc.delayed_post_pre_flat_jax(
                jnp.ones((t_pallas, dim), dtype=jnp.bfloat16),
                jnp.ones((t_pallas, mult * dim), dtype=jnp.bfloat16),
                jnp.ones((t_pallas, mult, 1), dtype=jnp.float32),
                jnp.ones((t_pallas, mult, mult), dtype=jnp.float32),
                fn_f32,
                scale,
                base,
                jnp.ones((t_pallas, mult), dtype=jnp.float32),
                *constants,
            )
            assert fp.shape == (t_pallas, mult * dim)
            assert pp.shape == (t_pallas, mult, 1)
        assert len(captured_kernel_calls) == 5

        # Case C: tokens = 16, 24, 144, 192, 288, 384, 512 take XLA path
        for t_case in (16, 24, 144, 192, 288, 384, 512):
            ft, pt, _, _, _ = mhc.delayed_post_pre_flat_jax(
                jnp.ones((t_case, dim), dtype=jnp.bfloat16),
                jnp.ones((t_case, mult * dim), dtype=jnp.bfloat16),
                jnp.ones((t_case, mult, 1), dtype=jnp.float32),
                jnp.ones((t_case, mult, mult), dtype=jnp.float32),
                fn_bf16,
                scale,
                base,
                jnp.ones((t_case, mult), dtype=jnp.float32),
                *constants,
            )
            assert ft.shape == (t_case, mult * dim)
            assert pt.shape == (t_case, mult, 1)
        assert len(captured_kernel_calls) == 5

    def test_initialize_delayed_mhc_and_fakes(self, monkeypatch):
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

        # 1. First initialization
        mhc.initialize_delayed_mhc()
        assert mhc._pre_op is not None
        assert mhc._pre_flat_op is not None
        assert mhc._post_pre_flat_op is not None

        # 2. Idempotency branch
        cached_pre_op = mhc._pre_op
        mhc.initialize_delayed_mhc()
        assert mhc._pre_op is cached_pre_op

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
        assert out1[0].shape == (2, 4, 1)
        assert out1[1].shape == (2, 4, 4)
        assert out1[2].shape == (2, 8)
        assert out1[3].shape == (2, 4)

        out2 = mhc._pre_flat_op(x, fn, scale, base, prev)
        assert len(out2) == 4
        assert out2[0].shape == (2, 4, 1)
        assert out2[1].shape == (2, 4, 4)
        assert out2[2].shape == (2, 8)
        assert out2[3].shape == (2, 4)

        out3 = mhc._post_pre_flat_op(
            sub_x, x, post_mix, comb_mix, fn, scale, base, prev
        )
        assert len(out3) == 5
        assert out3[0].shape == (2, 32)
        assert out3[1].shape == (2, 4, 1)
        assert out3[2].shape == (2, 4, 4)
        assert out3[3].shape == (2, 8)
        assert out3[4].shape == (2, 4)

    def test_delayed_pre_dispatch(self, monkeypatch):
        mock_pre = MagicMock(return_value="pre_called")
        mock_flat = MagicMock(return_value="flat_called")
        mock_post_pre = MagicMock(
            return_value=(torch.zeros((2, 32)), "p", "c", "v", "np")
        )
        monkeypatch.setattr(mhc, "_pre_op", mock_pre)
        monkeypatch.setattr(mhc, "_pre_flat_op", mock_flat)
        monkeypatch.setattr(mhc, "_post_pre_flat_op", mock_post_pre)

        residual = torch.zeros((2, 4, 8))
        fn = torch.zeros((24, 32))
        scale = torch.zeros(3)
        base = torch.zeros(24)
        prev = torch.zeros((2, 4))

        # Case 1: x is None -> calls _pre_flat_op
        res_none = mhc.delayed_pre(residual, None, fn, scale, base, prev)
        assert res_none == "flat_called"
        mock_flat.assert_called_once()
        mock_pre.assert_not_called()

        # Case 2: x is not None -> calls _pre_op
        mock_flat.reset_mock()
        res_x = mhc.delayed_pre(residual, "x_tensor", fn, scale, base, prev)
        assert res_x == "pre_called"
        mock_pre.assert_called_once()

        # Case 3: delayed_post_pre -> calls _post_pre_flat_op and unflattens residual
        res_pp = mhc.delayed_post_pre(
            torch.zeros((2, 8)), residual, "pm", "cm", fn, scale, base, prev
        )
        assert res_pp[0].shape == (2, 4, 8)
        assert res_pp[1:] == ("p", "c", "v", "np")
        mock_post_pre.assert_called_once()

    def test_initialize_mhc_post_and_post_dispatch(self, monkeypatch):
        mock_post_op = MagicMock(return_value="tpu_post_op")
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op.get_mhc_post_op",
            lambda: mock_post_op,
        )
        mhc.initialize_mhc_post()
        assert mhc._post_op is mock_post_op

        # 1. TPU device path
        mock_tpu_res = MagicMock()
        mock_tpu_res.device.type = "tpu"
        res_tpu = mhc.post("x", mock_tpu_res, "post_mix", "comb_mix")
        assert res_tpu == "tpu_post_op"
        mock_post_op.assert_called_once_with("x", mock_tpu_res, "post_mix", "comb_mix")

        # 2. CPU device path
        x = torch.randn(2, 8)
        residual = torch.randn(2, 4, 8)
        post_mix = torch.randn(2, 4, 1)
        comb_mix = torch.randn(2, 4, 4)
        res_cpu = mhc.post(x, residual, post_mix, comb_mix)
        assert res_cpu.shape == residual.shape

    def test_collapse(self):
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        pre_mix = torch.randn(2, 4)
        collapsed = mhc.collapse(x, pre_mix)
        assert collapsed.shape == (2, 8)
        assert collapsed.dtype == torch.bfloat16

    def test_pre_tpu_and_torch_paths(self, monkeypatch):
        residual = torch.randn(2, 4, 8)
        fn = torch.randn(24, 32)
        scale = torch.ones(3)
        base = torch.zeros(24)

        # 1. CPU path without norm
        p1, c1, v1, np1 = mhc.pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 5)
        assert p1.shape == (2, 4, 1)
        assert v1.shape == (2, 8)

        # 2. CPU path with norm_weight
        norm_w = torch.ones(8)
        p2, c2, v2, np2 = mhc.pre(
            residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 5, norm_weight=norm_w
        )
        assert v2.shape == (2, 8)

        # 3. TPU path
        mock_delayed_pre = MagicMock(return_value=("p", "c", torch.zeros(2, 8), "np"))
        monkeypatch.setattr(mhc, "delayed_pre", mock_delayed_pre)

        mock_tpu_res = MagicMock()
        mock_tpu_res.device.type = "tpu"
        mock_tpu_res.shape = (2, 4, 8)
        mock_tpu_res.__getitem__ = MagicMock(return_value=torch.zeros(2, 4))
        real_arange = torch.arange
        monkeypatch.setattr(mhc.torch, "arange", lambda n, **kw: real_arange(n))

        # Sub-case A: pre_mix is None (first layer stream)
        out_tpu1 = mhc.pre(
            mock_tpu_res, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 5, pre_mix=None
        )
        assert out_tpu1[0] == "p"
        mock_delayed_pre.assert_called_once()

        # Sub-case B: pre_mix is not None
        mock_delayed_pre.reset_mock()
        out_tpu2 = mhc.pre(
            mock_tpu_res,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            5,
            pre_mix=torch.zeros(2, 4),
        )
        assert out_tpu2[0] == "p"
        mock_delayed_pre.assert_called_once()

    def test_shifted_post_pre(self, monkeypatch):
        # 1. stream is not None -> ValueError
        with pytest.raises(ValueError, match="TPU mHC does not use CUDA streams"):
            mhc.shifted_post_pre(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                5,
                stream="cuda_stream",
            )

        # 2. CPU / unfused fallback: reduce_results=True and capture_aux=True
        monkeypatch.setattr(mhc, "_post_pre_flat_op", None)
        monkeypatch.setattr(mhc, "tensor_model_parallel_all_reduce", lambda t: t * 2)
        monkeypatch.setattr(mhc, "post", lambda x, res, p_mix, c_mix: res)
        monkeypatch.setattr(
            mhc,
            "pre",
            lambda res, fn, sc, bs, r_eps, p_eps, s_eps, pm, sr, **kw: (
                "p",
                "c",
                "v",
                "np",
            ),
        )
        x = torch.ones(2, 4, 8)
        res = torch.ones(2, 4, 8)
        out = mhc.shifted_post_pre(
            x,
            res,
            "pmix",
            "cmix",
            "fn",
            "sc",
            "bs",
            1e-6,
            1e-6,
            1e-6,
            2.0,
            5,
            capture_aux=True,
            reduce_results=True,
        )
        assert len(out) == 6
        assert out[0].shape == (2, 4, 8)
        assert out[5].shape == (2, 8)  # aux is residual.mean(1)

        # 3. CPU / unfused fallback: reduce_results=False and capture_aux=False
        out2 = mhc.shifted_post_pre(
            x,
            res,
            "pmix",
            "cmix",
            "fn",
            "sc",
            "bs",
            1e-6,
            1e-6,
            1e-6,
            2.0,
            5,
            capture_aux=False,
            reduce_results=False,
        )
        assert out2[5].shape == (0, 8)

        # 4. TPU fused delayed_post_pre path
        monkeypatch.setattr(mhc, "_post_pre_flat_op", "initialized")
        mock_dpp = MagicMock(
            return_value=(
                torch.ones(2, 4, 8),
                "p_tpu",
                "c_tpu",
                torch.ones(2, 8),
                "np_tpu",
            )
        )
        monkeypatch.setattr(mhc, "delayed_post_pre", mock_dpp)
        mock_tpu_res = MagicMock()
        mock_tpu_res.device.type = "tpu"
        mock_tpu_res.shape = (2, 4, 8)
        mock_tpu_res.__getitem__ = MagicMock(return_value=torch.zeros(2, 4))
        real_arange = torch.arange
        monkeypatch.setattr(mhc.torch, "arange", lambda n, **kw: real_arange(n))

        sc = torch.ones(3, dtype=torch.float32)
        norm_w = torch.ones(8, dtype=torch.float32)
        out_tpu1 = mhc.shifted_post_pre(
            torch.ones(2, 8),
            mock_tpu_res,
            "pmix",
            "cmix",
            "fn",
            sc,
            "bs",
            1e-6,
            1e-6,
            1e-6,
            2.0,
            5,
            pre_mix=None,
            norm_weight=norm_w,
            capture_aux=True,
        )
        assert out_tpu1[1] == "p_tpu"
        assert out_tpu1[3].shape == (2, 8)
        assert out_tpu1[5].shape == (2, 8)
        mock_dpp.assert_called_once()

        mock_dpp.reset_mock()
        out_tpu2 = mhc.shifted_post_pre(
            torch.ones(2, 8),
            mock_tpu_res,
            "pmix",
            "cmix",
            "fn",
            sc,
            "bs",
            1e-6,
            1e-6,
            1e-6,
            2.0,
            5,
            pre_mix=torch.zeros(2, 4),
            norm_weight=None,
            capture_aux=False,
        )
        assert out_tpu2[1] == "p_tpu"
        assert out_tpu2[5].shape == (0, 8)
        mock_dpp.assert_called_once()


# ==============================================================================
# 2. Tests for v2/attention.py Helper Functions
# ==============================================================================
class TestAttentionHelperFunctions:
    """Tests for standalone functions and cache addressing in v2/attention.py."""

    def test_select_candidate_blocks(self):
        # 1. Invalid parameter validations
        with pytest.raises(ValueError, match="invalid candidate block parameters"):
            v2_attn.select_candidate_blocks(torch.zeros(2, 10), [10, 10], -1, 4)
        with pytest.raises(ValueError, match="invalid candidate block parameters"):
            v2_attn.select_candidate_blocks(torch.zeros(2, 10), [10, 10], 2, 0)

        # 2. width == 0
        empty = v2_attn.select_candidate_blocks(torch.zeros(2, 0), [0, 0], 2, 4)
        assert empty.shape == (2, 0)

        # 3. Normal candidate block selection
        logits = torch.randn(32)
        lengths = 24
        keep = v2_attn.select_candidate_blocks(
            logits, lengths, topk_blocks=3, block_size=4
        )
        assert keep.shape == (32,)
        assert keep.dtype == torch.bool

    def test_select_compressed_indices(self):
        # 1. topk < 0 -> ValueError
        with pytest.raises(ValueError, match="topk must be nonnegative"):
            v2_attn.select_compressed_indices(torch.zeros(2, 10), [10, 10], -1)

        # 2. Without candidates
        scores = torch.tensor([1.0, 3.0, 2.0, -torch.inf])
        lengths = 3
        idx = v2_attn.select_compressed_indices(scores, lengths, topk=2)
        assert idx.shape == (2,)
        assert idx[0] == 1
        assert idx[1] == 2

        # 3. With candidates mask
        candidates = torch.tensor([True, False, True, True])
        idx_cand = v2_attn.select_compressed_indices(
            scores, lengths, topk=2, candidates=candidates
        )
        assert idx_cand.shape == (2,)

    def test_sparse_attention(self):
        # 1. kv.shape[1] == 0 or indices.shape[-1] == 0 -> returns zeros_like
        q = torch.randn(2, 4, 8, 16)
        kv_empty = torch.randn(2, 0, 16)
        indices = torch.zeros((2, 4, 0), dtype=torch.int32)
        sink = torch.zeros(8)
        assert torch.equal(
            v2_attn.sparse_attention(q, kv_empty, sink, indices, 0.25),
            torch.zeros_like(q),
        )

        # 2. q.dtype == torch.bfloat16 (multi-block online softmax GEMM loop)
        q_bf16 = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
        kv_bf16 = torch.randn(2, 128, 16, dtype=torch.bfloat16)
        indices_bf16 = torch.randint(0, 128, (2, 4, 80), dtype=torch.int32)
        out_bf16 = v2_attn.sparse_attention(q_bf16, kv_bf16, sink, indices_bf16, 0.25)
        assert out_bf16.shape == q_bf16.shape
        assert out_bf16.dtype == torch.bfloat16

        # 3. q.dtype == torch.float32 (softmax path)
        q_f32 = torch.randn(2, 4, 8, 16, dtype=torch.float32)
        kv_f32 = torch.randn(2, 16, 16, dtype=torch.float32)
        indices_f32 = torch.randint(0, 16, (2, 4, 8), dtype=torch.int32)
        out_f32 = v2_attn.sparse_attention(q_f32, kv_f32, sink, indices_f32, 0.25)
        assert out_f32.shape == q_f32.shape
        assert out_f32.dtype == torch.float32

    def test_cache_slots(self):
        # Empty seq_lens -> returns full_like(positions, -1)
        meta_empty = SimpleNamespace(seq_lens=torch.empty(0, dtype=torch.int32))
        pos = torch.arange(8)
        cache = torch.empty(4, 16)
        slots = v2_attn._cache_slots(cache, meta_empty, pos)
        assert torch.all(slots == -1)

        # Non-empty seq_lens with ring=True and ring=False
        meta = SimpleNamespace(
            seq_lens=torch.tensor([4, 4]),
            block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8]),
        )
        s_normal = v2_attn._cache_slots(cache, meta, pos, compress_ratio=1, ring=False)
        assert s_normal.shape == pos.shape
        s_ring = v2_attn._cache_slots(cache, meta, pos, compress_ratio=2, ring=True)
        assert s_ring.shape == pos.shape

    def test_save_compress_norm(self, monkeypatch):
        dim = 8
        norm_w = torch.ones(dim)
        kv_score = torch.randn(10, dim * 2)
        positions = torch.arange(10)
        slot_mapping = torch.arange(10)
        query_start_loc = torch.tensor([0, 5, 10])
        token_to_req = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        latent = torch.zeros(10, dim)

        # 1. ratio != 1 and ratio != 2 -> ValueError
        with pytest.raises(ValueError, match="V4.1 compression ratio must be 1 or 2"):
            v2_attn.save_compress_norm(
                kv_score,
                positions,
                None,
                slot_mapping,
                query_start_loc,
                token_to_req,
                norm_w,
                1e-6,
                3,
                latent,
            )

        # 2. ratio == 1
        v2_attn.save_compress_norm(
            kv_score,
            positions,
            None,
            slot_mapping,
            query_start_loc,
            token_to_req,
            norm_w,
            1e-6,
            1,
            latent,
        )
        assert latent.shape == (10, dim)

        # 3. ratio == 2
        state_cache = torch.zeros(4, 16, dim * 2)
        monkeypatch.setattr(v2_attn, "write_cache", lambda *a, **kw: None)
        v2_attn.save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slot_mapping,
            query_start_loc,
            token_to_req,
            norm_w,
            1e-6,
            2,
            latent,
        )
        assert latent.shape == (10, dim)

        # 4. token_to_req_indices is None (exercises 181->183 False branch)
        v2_attn.save_compress_norm(
            kv_score,
            positions,
            None,
            slot_mapping,
            query_start_loc,
            None,
            norm_w,
            1e-6,
            1,
            latent,
        )

        # 5. slot_mapping is None (exercises 179->181 False branch)
        with pytest.raises(TypeError):
            v2_attn.save_compress_norm(
                kv_score,
                positions,
                None,
                None,
                query_start_loc,
                None,
                norm_w,
                1e-6,
                1,
                latent,
            )

    def test_cache_backends_and_indexer_fp4(self):
        # 1. indexer_uses_fp4
        cfg_fp8 = SimpleNamespace(
            attention_config=SimpleNamespace(resolve_indexer_kv_dtype=lambda x: "fp8")
        )
        assert v2_attn.indexer_uses_fp4(cfg_fp8) is False

        cfg_other = SimpleNamespace(
            attention_config=SimpleNamespace(resolve_indexer_kv_dtype=lambda x: "fp4")
        )
        with pytest.raises(ValueError, match="TPU V4.1 indexer requires bf16 or fp8"):
            v2_attn.indexer_uses_fp4(cfg_other)

        # 2. AttentionBackend
        assert v2_attn.AttentionBackend.get_name() == "TORCH_TPU_DEEPSEEK_V41"
        assert v2_attn.AttentionBackend.get_kv_cache_shape(10, 16, 8, 64) == (
            10,
            8,
            16,
            64,
        )

        # 3. CompressorStateCache.get_attn_backend
        csc = object.__new__(v2_attn.CompressorStateCache)
        assert csc.get_attn_backend() is v2_attn.AttentionBackend

        # 4. SWACache
        swa_cache = object.__new__(v2_attn.SWACache)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention.DeepseekV4SWACache.__init__"
        ) as mock_super_swa:
            v2_attn.SWACache.__init__(
                swa_cache, cache_config=SimpleNamespace(block_size=32)
            )
            mock_super_swa.assert_called_once()

        # 5. IndexerCache
        idx_cache = object.__new__(v2_attn.IndexerCache)
        assert idx_cache.get_attn_backend() is v2_attn.AttentionBackend
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_cache.get_current_vllm_config",
                return_value=SimpleNamespace(
                    cache_config=SimpleNamespace(),
                    attention_config=SimpleNamespace(
                        resolve_indexer_kv_dtype=lambda _: "bf16"
                    ),
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(index_head_dim=128)
                    ),
                ),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention.upstream.DeepseekV4IndexerCache.__init__"
            ) as mock_super_idx,
        ):
            v2_attn.IndexerCache.__init__(idx_cache)
            mock_super_idx.assert_called_once()

    def test_gather_query_rows_and_read_rows(self, monkeypatch):
        # _gather_query_rows with padding != 0
        monkeypatch.setattr(
            v2_attn, "tensor_model_parallel_all_gather", lambda x, dim=0: x
        )
        rows_unpadded = torch.randn(4, 100)
        res1 = v2_attn._gather_query_rows(rows_unpadded)
        assert res1.shape == (4, 100)

        # _gather_query_rows with padding == 0
        rows_padded = torch.randn(4, 128)
        res2 = v2_attn._gather_query_rows(rows_padded)
        assert res2.shape == (4, 128)

        # _read_rows
        cache = torch.randn(10, 32, 16)
        table = torch.randint(0, 10, (2, 4), dtype=torch.int32)
        requests = torch.tensor([0, 1])
        logical = torch.tensor([[0, 1], [2, 3]])
        out_read = v2_attn._read_rows(cache, table, requests, logical)
        assert out_read.shape == (2, 2, 16)


# ==============================================================================
# 3. Tests for DeepseekCompressor
# ==============================================================================
class TestDeepseekCompressorFull:
    """Tests for DeepseekCompressor project, init, and forward in v2/attention.py."""

    def test_deepseek_compressor_project_and_init(self):
        comp = object.__new__(v2_attn.DeepseekCompressor)
        torch.nn.Module.__init__(comp)
        comp.fused_wkv_wgate = SimpleNamespace(weight=torch.randn(16, 32))

        # 1. Project with not flipped
        h = torch.randn(4, 32)
        p1 = comp.project(h)
        assert p1.shape == (4, 16)

        # 2. Project with flipped
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        setattr(comp.fused_wkv_wgate, WEIGHT_FLIPPED_ATTR, True)
        comp.fused_wkv_wgate.weight = torch.randn(32, 16)
        p2 = comp.project(h)
        assert p2.shape == (4, 16)

        # 3. __init__
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention.compressor_upstream.DeepseekCompressor.__init__"
        ) as mock_comp_init:
            v2_attn.DeepseekCompressor.__init__(comp)
            mock_comp_init.assert_called_once()

    def test_deepseek_compressor_forward(self, monkeypatch):
        comp = object.__new__(v2_attn.DeepseekCompressor)
        torch.nn.Module.__init__(comp)

        # Forward when metadata is not dict -> returns None
        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention.get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        )
        assert comp.forward(torch.randn(4, 16), torch.arange(4)) is None


# ==============================================================================
# 4. Tests for DeepseekV41Attention Methods & Properties
# ==============================================================================
class TestDeepseekV41AttentionMethods:
    """Tests for DeepseekV41Attention methods, properties, and projection hooks."""

    def test_padded_q_heads_and_mqa_runtime_error(self):
        assert v2_attn.DeepseekV41Attention.get_padded_num_q_heads(16) == 16
        attn = object.__new__(v2_attn.DeepseekV41Attention)
        with pytest.raises(
            RuntimeError, match="TPU sparse attention is dispatched by _prepare_tpu"
        ):
            attn.forward_mqa(None, None, None, None)

    def test_kv_rope_op_and_rotate(self):
        attn = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)

        # 1. kv_rope_op cached instance
        mock_rope_inst = MagicMock(return_value="tpu_rope_result")
        attn._kv_rope_op_instance = mock_rope_inst
        assert attn.kv_rope_op is mock_rope_inst

        # 1b. kv_rope_op uncached instance (exercises lines 653-657)
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        attn_uncached = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn_uncached)
        with patch.object(
            VllmDeepseekV4MLAAttention,
            "kv_rope_op",
            property(lambda self: "upstream_rope_op"),
        ):
            assert attn_uncached.kv_rope_op == "upstream_rope_op"

        # 2. _rotate with TPU / non-CPU device
        mock_tpu_val = MagicMock()
        mock_tpu_val.device.type = "tpu"
        attn._tpu_rope_cache = "mock_tpu_rope_cache"
        res_tpu = attn._rotate(mock_tpu_val, "pos", inverse=False)
        assert res_tpu == "tpu_rope_result"
        mock_rope_inst.assert_called_once_with(
            mock_tpu_val, "pos", "mock_tpu_rope_cache"
        )

        # 3. _rotate with CPU device (rotary fallback)
        attn.compress_ratio = 1
        attn._config = {
            "qk_rope_head_dim": 8,
            "max_position_embeddings": 256,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "compress_rope_theta": 10000.0,
        }
        val_cpu = torch.randn(4, 8)
        pos = torch.arange(4)
        res_cpu = attn._rotate(val_cpu, pos, inverse=False)
        assert res_cpu.shape == val_cpu.shape

        # 4. _rotate with inverse=True
        res_inv = attn._rotate(val_cpu, pos, inverse=True)
        assert res_inv.shape == val_cpu.shape

    def test_split_qkv_and_norm(self):
        attn = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.q_lora_rank = 16
        attn.q_norm = lambda x: x * 2
        attn.kv_norm = lambda x: x * 3

        qr_kv = torch.ones(4, 32)
        q, null, kv = attn._split_qkv_and_norm(qr_kv)
        assert torch.all(q == 2)
        assert null is None
        assert torch.all(kv == 3)

    def test_o_proj_methods(self):
        attn = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.n_local_heads = 4
        attn.head_dim = 16
        attn._tpu_rope_cache = "rope_cache"
        attn.wo_a = SimpleNamespace(weight="w", weight_scale="s")
        attn.wo_b = lambda x: (x * 2,)

        # Test _o_proj
        mock_o_proj = MagicMock(return_value=torch.ones(2, 4, 16))
        attn._o_proj_op_instance = mock_o_proj
        out = torch.ones(2, 4, 16, dtype=torch.bfloat16)
        pos = torch.arange(2)

        res = attn._o_proj(out, pos)
        assert torch.all(res == 2)

        # Test uncached o_proj_op (lines 1558-1564)
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        attn._o_proj_op_instance = None
        with patch.object(
            VllmDeepseekV4MLAAttention,
            "o_proj_op",
            property(lambda self: "upstream_o_proj_op"),
        ):
            assert attn.o_proj_op == "upstream_o_proj_op"

        # Test _assert_o_proj_supported (lines 1566-1571)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v4.attention.VllmDeepseekV4MLAAttention._assert_o_proj_supported"
        ) as mock_assert:
            attn._assert_o_proj_supported()
            mock_assert.assert_called_once_with(attn)

    def test_select_indices_with_candidate_blocks(self):
        attn = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn._config = {"candidate_topk_blocks": 2, "index_topk": 4}
        attn.candidate_block_size = 4
        attn.candidate_block_buffer = torch.full((4, 4), -1, dtype=torch.int32)

        scores = torch.randn(1, 1, 32)
        lengths = torch.tensor([16])
        logical = torch.arange(32)[None, :]

        # 1. Candidate write path (candidate_write = True)
        mock_indexer_op_write = SimpleNamespace(
            candidate_blocks=True, candidate_write=True
        )
        attn.indexer = SimpleNamespace(indexer_op=mock_indexer_op_write)

        idx_write = attn._select_indices(scores, lengths, logical, 1)
        assert idx_write.shape == (1, 4)

        # 2. Candidate read path (candidate_write = False)
        mock_indexer_op_read = SimpleNamespace(
            candidate_blocks=True, candidate_write=False
        )
        attn.indexer = SimpleNamespace(indexer_op=mock_indexer_op_read)

        idx_read = attn._select_indices(scores, lengths, logical, 1)
        assert idx_read.shape == (1, 4)

        # 3. lengths.max() <= index_topk branch
        lengths_small = torch.tensor([2])
        idx_small = attn._select_indices(scores, lengths_small, logical, 1)
        assert idx_small.shape == (1, 4)

        # 4. candidate_blocks is None (exercises 1498->1548 False branch)
        mock_indexer_op_none = SimpleNamespace(candidate_blocks=None)
        attn.indexer = SimpleNamespace(indexer_op=mock_indexer_op_none)
        idx_none = attn._select_indices(scores, lengths, logical, 1)
        assert idx_none.shape == (1, 4)

    def test_process_weights_after_loading(self, monkeypatch):
        attn = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.attn_sink = torch.randn(8)
        attn._full_attn_sink = torch.empty(8)
        attn._init_local_q_weights = MagicMock()
        attn._assert_o_proj_supported = MagicMock()
        attn.wo_a = SimpleNamespace(
            weight=SimpleNamespace(device=SimpleNamespace(type="cpu"))
        )

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v4.attention.VllmDeepseekV4MLAAttention.process_weights_after_loading",
            lambda self, act: None,
        )

        # 1. tp_size == 1
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
        attn.process_weights_after_loading(torch.bfloat16)
        assert torch.equal(attn._full_attn_sink, attn.attn_sink.float())

        # 2. tp_size > 1
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(
            v2_attn, "tensor_model_parallel_all_gather", lambda t, dim=0: t
        )
        attn.process_weights_after_loading(torch.bfloat16)
        assert torch.equal(attn._full_attn_sink, attn.attn_sink.float())
