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

"""Tests of the library: norm, quant, rope and layouts.

norm, quant and rope are checked against vLLM's formulas. Each layout
primitive runs alone in a small Pallas kernel gridded over token tiles, on
values that are distinct per element.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import (
    layouts,
    norm,
    quant,
    rope,
)

TILE = 256
ROWS = 128  # Rows per store, like the kernels' matmul_rows.


def _distinct(shape, dtype=jnp.float32):
    """Random values: a misplaced row, lane or head is a mismatch."""
    key = jax.random.key(np.prod(shape))
    return jax.random.normal(key, shape, jnp.float32).astype(dtype)


def _tiled(kernel, x, out_shape, num_tokens):
    """Runs kernel(x_ref, o_ref) per TILE tokens of x [T, ...]."""

    def spec(shape):
        n = len(shape)
        return pl.BlockSpec((TILE, *shape[1:]), lambda i: (i,) + (0,) * (n - 1))

    return pl.pallas_call(
        kernel,
        grid=(num_tokens // TILE,),
        in_specs=[spec(x.shape)],
        out_specs=spec(out_shape.shape),
        out_shape=out_shape,
    )(x)


def _assert_equal(got, want):
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(want, np.float32)
    )


def _vllm_rms_norm(x, weight, eps):
    # ir.ops.rms_norm.
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf**2, -1, keepdims=True) + eps)
    if weight is not None:
        xf = xf.astype(weight.dtype) * weight
    return xf.astype(x.dtype)


def _vllm_rope(x, cos, sin, rotary_dim):
    # RotaryEmbedding.forward_static: rotate x[..., :rotary_dim] NeoX-style.
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = jnp.split(rot, 2, -1)
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return jnp.concatenate([out, rest], -1)


class _ReferenceTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        ks = jax.random.split(jax.random.key(0), 2)
        self.x = jax.random.normal(ks[0], (64, 4, 512), jnp.bfloat16)
        self.w = 1 + 0.1 * jax.random.normal(ks[1], (512,), jnp.bfloat16)


class NormTest(_ReferenceTest):
    def test_rms(self):
        _assert_equal(
            norm.norm(configs.Norm.RMS, self.x, self.w, 1e-6),
            _vllm_rms_norm(self.x, self.w, 1e-6),
        )

    def test_rms_one_plus_w(self):
        # GemmaRMSNorm.forward_native passes weight.float() + 1.
        w = self.w - 1
        _assert_equal(
            norm.norm(configs.Norm.RMS_ONE_PLUS_W, self.x, w, 1e-6),
            _vllm_rms_norm(self.x, w.astype(jnp.float32) + 1.0, 1e-6),
        )

    def test_rms_noscale(self):
        _assert_equal(
            norm.norm(configs.Norm.RMS_NOSCALE, self.x, None, 1e-6),
            _vllm_rms_norm(self.x, None, 1e-6),
        )

    def test_fused_add_rms(self):
        # ir.ops.fused_add_rms_norm: f32 add, bf16 residual, norm of the f32 sum.
        residual = self.x[::-1]
        xf = self.x.astype(jnp.float32) + residual.astype(jnp.float32)
        got, got_residual = norm.add_norm(
            configs.Norm.RMS, self.x, residual, self.w, 1e-6
        )
        _assert_equal(got_residual, xf.astype(jnp.bfloat16))
        want = xf * jax.lax.rsqrt(jnp.mean(xf**2, -1, keepdims=True) + 1e-6)
        _assert_equal(got, (want.astype(jnp.bfloat16) * self.w).astype(jnp.bfloat16))

    def test_f32_weight(self):
        k, block = 2048, 1024
        w = _distinct((1, k), jnp.bfloat16)
        got = norm.f32_weight(configs.Norm.RMS_ONE_PLUS_W, w, block)
        self.assertEqual(got.shape, (k // block, 1, block))
        self.assertEqual(got.dtype, jnp.float32)
        _assert_equal(got.reshape(1, k), w.astype(jnp.float32) + 1.0)

    def test_packed_bf16_weight(self):
        k, block = 2048, 1024
        w = _distinct((1, k), jnp.bfloat16)
        got = np.asarray(norm.packed_bf16_weight(configs.Norm.RMS, w, block))
        self.assertEqual(got.shape, (k // block, 1, block))
        bits = np.asarray(w).view(np.uint16).astype(np.uint32).reshape(got.shape)
        np.testing.assert_array_equal(got, bits | (bits << 16))


class QuantTest(_ReferenceTest):
    def test_quantize_fp8_per_token(self):
        # vllm_torchtpu quantized_matmul/util.py quantize_block(x, axis=-1).
        x = self.x.at[0].set(0)
        abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
        scale = abs_max / jnp.finfo(quant.F8).max.astype(jnp.float32)
        scale = jnp.where(scale == 0, 1.0, scale)
        want = (x / scale).clip(-quant.F8_MAX, quant.F8_MAX).astype(quant.F8)
        got, got_scale = quant.quantize_fp8_per_token(x)
        _assert_equal(got_scale, scale)
        _assert_equal(got, want)


class RopeTest(_ReferenceTest):
    def test_partial_rope(self):
        pos = jnp.arange(64)
        cos, sin = rope.cos_sin_default(pos, 64, 1e7)
        x = self.x[..., :256]
        _assert_equal(
            rope.rope(configs.Rope.neox(64), x, cos[:, None], sin[:, None]),
            _vllm_rope(x, cos[:, None], sin[:, None], 64),
        )

    def test_proportional_rope(self):
        # Gemma4RotaryEmbedding: full-width rotate_half with identity padding.
        pos = jnp.arange(64) * 97
        cos, sin = rope.cos_sin_proportional(pos, 512, 0.25, 1e6)
        self.assertEqual(cos.shape, (64, 64))

        def pad(t, v):
            return jnp.pad(t, ((0, 0), (0, 192)), constant_values=v)

        full_cos, full_sin = pad(cos, 1), pad(sin, 0)
        _assert_equal(
            rope.rope(configs.Rope(64, 256), self.x, cos[:, None], sin[:, None]),
            _vllm_rope(self.x, full_cos[:, None], full_sin[:, None], 512),
        )

    def test_yarn_matches_formula(self):
        # gpt-oss-120b: low/high correction dims are 8.09 and 17.39.
        pos = jnp.array([0, 1, 4096, 131071])
        cos, sin = rope.cos_sin_yarn(pos, 64, 150000.0, 32.0, 4096, truncate=False)
        mscale = 0.1 * np.log(32.0) + 1.0
        np.testing.assert_allclose(np.asarray(cos[0], np.float32), mscale, rtol=1e-2)
        np.testing.assert_array_equal(np.asarray(sin[0], np.float32), 0)
        # Highest-frequency pair is extrapolated: angle = pos * 1.
        np.testing.assert_allclose(
            np.asarray(sin[1, 0], np.float32), mscale * np.sin(1.0), rtol=1e-2
        )


class LayoutsTest(parameterized.TestCase):
    def test_pack_heads_bf16(self):
        lo, hi = _distinct((8, 128), jnp.bfloat16), _distinct((8, 128))[::-1]
        hi = hi.astype(jnp.bfloat16)
        packed = np.asarray(layouts.pack_heads([lo, hi]))

        def bits(a):
            return np.asarray(a).view(np.uint16).astype(np.uint32)

        np.testing.assert_array_equal(packed, bits(lo) | (bits(hi) << 16))

    @parameterized.parameters(1536, 512)
    def test_strided_store_column_blocks(self, width):
        """Column blocks of (T, W) f32 -> compact (T, 1, W): mixed_qkv."""
        t, block = 512, 256
        x = _distinct((t, width))

        def kernel(x_ref, o_ref):
            for m in range(0, TILE, ROWS):
                rows = pl.ds(m, ROWS)
                for c in range(0, width, block):
                    layouts.strided_store(
                        o_ref, m, 1, x_ref[rows, pl.ds(c, block)], col=c
                    )

        out = jax.ShapeDtypeStruct((t, 1, width), jnp.float32)
        _assert_equal(_tiled(kernel, x, out, t), x.reshape(t, 1, width))

    @parameterized.parameters(8, 4)
    def test_store_head_pair(self, num_heads):
        """Head pairs of f32 (T, H * 128) -> head-major bf16 (T, H, 128): z."""
        t, d = 512, 128
        x = _distinct((t, num_heads * d))

        def kernel(x_ref, o_ref):
            for m in range(0, TILE, ROWS):
                rows = pl.ds(m, ROWS)
                for p in range(num_heads // 2):
                    val = x_ref[rows, pl.ds(2 * d * p, 2 * d)]
                    layouts.store_head_pair(o_ref, rows, p, val)

        out = jax.ShapeDtypeStruct((t, num_heads, d), jnp.bfloat16)
        want = x.astype(jnp.bfloat16).reshape(t, num_heads, d)
        _assert_equal(_tiled(kernel, x, out, t), want)

    def test_strided_store_rows(self):
        """f32 (T, 128) -> lane-padded (T, 1, 128): b and a."""
        t = 512
        x = _distinct((t, 128))

        def kernel(x_ref, o_ref):
            for m in range(0, TILE, ROWS):
                layouts.strided_store(o_ref, m, 1, x_ref[pl.ds(m, ROWS)])

        out = jax.ShapeDtypeStruct((t, 1, 128), jnp.float32)
        _assert_equal(_tiled(kernel, x, out, t), x.reshape(t, 1, 128))


if __name__ == "__main__":
    absltest.main()
