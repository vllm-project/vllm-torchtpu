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
"""`quantized_matmul_heads` with the DeepSeek-V4 q norm + RoPE epilogue
against the projection and `qnorm_rope_flat` it fuses."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.interpreters import mlir
from jax._src.pallas.mosaic import primitives as tpu_primitives

from vllm_torchtpu.kernels.deepseek_v4 import rope
from vllm_torchtpu.kernels.deepseek_v4.rope import quantized_matmul_heads
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

FP8 = jnp.float8_e4m3fn
EPS = 1e-6
MAX_POSITION = 4096


def _pack_elementwise_on_cpu(*xs, packed_dtype):
    lo, hi = (
        jax.lax.bitcast_convert_type(x.astype(packed_dtype), jnp.uint16).astype(
            jnp.uint32
        )
        for x in xs
    )
    return lo | (hi << 16)


# Interpret mode has no CPU rule for the TPU pack primitive.
mlir.register_lowering(
    tpu_primitives.pack_elementwise_p,
    mlir.lower_fun(_pack_elementwise_on_cpu, multiple_results=False),
    platform="cpu",
)


def _operands(m: int, k: int, n: int):
    keys = jax.random.split(jax.random.key(m * 131 + k), 6)
    x = jax.random.normal(keys[0], (m, k), jnp.float32)
    x = x * jnp.exp(jax.random.uniform(keys[1], (m, 1), minval=-8, maxval=8))
    x = x.at[0].set(0).astype(jnp.bfloat16)
    w_q = jax.random.normal(keys[2], (k, n), jnp.float32).astype(FP8)
    w_s = jax.random.uniform(keys[3], (n,), jnp.float32, 1e-3, 1e-2)
    positions = jax.random.randint(keys[4], (m,), 0, MAX_POSITION, jnp.int32)
    cos_sin_cache = jax.random.uniform(keys[5], (MAX_POSITION, 64), jnp.float32, -1, 1)
    return x, w_q, w_s, positions, cos_sin_cache


def _fused(x, w_q, w_s, positions, cos_sin_cache, head_dim, interpret=False):
    return quantized_matmul_heads(
        x,
        w_q,
        w_s,
        head_dim,
        functools.partial(rope.qnorm_rope_head, eps=EPS),
        rope.cos_sin_lanes,
        positions,
        cos_sin_cache,
        interpret=interpret,
    )


def _as_bits(a: jax.Array) -> np.ndarray:
    return np.asarray(a).view(np.uint16)


# Several row blocks, weight tiles and epilogue row chunks, and one of each.
@pytest.mark.parametrize(
    "m,k,n,head_dim,block_rows,epilogue_rows",
    [(128, 256, 8192, 256, 64, 32), (64, 128, 8192, 512, 64, 128)],
)
def test_quantized_matmul_heads_matches_unfused(
    monkeypatch, m, k, n, head_dim, block_rows, epilogue_rows
):
    monkeypatch.setattr(rope, "HEADS_BLOCK_ROWS", block_rows)
    monkeypatch.setattr(rope, "HEADS_MIN_ROWS", 32)
    monkeypatch.setattr(rope, "HEADS_EPILOGUE_ROWS", epilogue_rows)
    x, w_q, w_s, positions, cos_sin_cache = _operands(m, k, n)
    got = _fused(x, w_q, w_s, positions, cos_sin_cache, head_dim, interpret=True)
    q = xla_quantized_matmul(x, w_q, w_s).reshape(m, n // head_dim, head_dim)
    cos, sin = rope.cos_sin_lanes(cos_sin_cache[positions])
    want = rope.qnorm_rope_head(
        q.astype(jnp.float32), cos[:, None], sin[:, None], eps=EPS
    ).astype(jnp.bfloat16)
    assert got.shape == want.shape and got.dtype == want.dtype
    # XLA:CPU may sum the norm's squares in another order than the kernel.
    mismatched = np.mean(_as_bits(got) != _as_bits(want))
    assert mismatched < 1e-5, mismatched
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-2
    )


@pytest.mark.parametrize("m,block_rows", [(128, 32), (64, 64)])
def test_quantized_matmul_heads_rotates_kv(monkeypatch, m, block_rows):
    monkeypatch.setattr(rope, "HEADS_BLOCK_ROWS", block_rows)
    monkeypatch.setattr(rope, "HEADS_MIN_ROWS", 32)
    k, n, head_dim = 128, 8192, 512
    x, w_q, w_s, positions, cos_sin_cache = _operands(m, k, n)
    kv = jax.random.normal(jax.random.key(7), (m, head_dim), jnp.float32)
    kv = kv.astype(jnp.bfloat16)
    want_q = _fused(x, w_q, w_s, positions, cos_sin_cache, head_dim, interpret=True)
    got_q, got_kv = quantized_matmul_heads(
        x,
        w_q,
        w_s,
        head_dim,
        functools.partial(rope.qnorm_rope_head, eps=EPS),
        rope.cos_sin_lanes,
        positions,
        cos_sin_cache,
        kv=kv,
        kv_fn=rope.rope_lanes,
        interpret=True,
    )
    np.testing.assert_array_equal(_as_bits(got_q), _as_bits(want_q))
    cos, sin = rope.cos_sin_lanes(cos_sin_cache[positions])
    tail = rope.rope_lanes(kv[:, -128:].astype(jnp.float32), cos, sin)
    want_kv = kv.at[:, -128:].set(tail.astype(jnp.bfloat16))
    np.testing.assert_array_equal(_as_bits(got_kv), _as_bits(want_kv))


@pytest.mark.tpu_test
@pytest.mark.parametrize("m", [128, 1024])
def test_quantized_matmul_heads_is_bit_identical_on_tpu(m):
    if jax.default_backend() != "tpu":
        pytest.fail(f"needs the TPU backend, got {jax.default_backend()}")
    k, n, head_dim = 1024, 32768, 512
    x, w_q, w_s, positions, cos_sin_cache = _operands(m, k, n)

    @jax.jit
    def unfused(x, w_q, w_s, positions, cos_sin_cache):
        q = xla_quantized_matmul(x, w_q, w_s, quantize_in_one_pass=True)
        return rope.qnorm_rope_flat(
            q, positions, cos_sin_cache, num_heads=n // head_dim, eps=EPS
        )

    got = jax.jit(_fused, static_argnums=5)(
        x, w_q, w_s, positions, cos_sin_cache, head_dim
    )
    want = unfused(x, w_q, w_s, positions, cos_sin_cache)
    np.testing.assert_array_equal(_as_bits(got), _as_bits(want))


def _fused_rope_quant(x, w_q, w_s, positions, cos_sin_cache, interpret=False):
    return quantized_matmul_heads(
        x,
        w_q,
        w_s,
        rope.LANE,
        rope.rope_lanes,
        rope.cos_sin_lanes,
        positions,
        cos_sin_cache,
        quant_dtype=FP8,
        interpret=interpret,
    )


# Several row blocks, weight tiles and epilogue passes, and one of each.
@pytest.mark.parametrize(
    "m,k,n,block_rows,pass_rows",
    [(256, 256, 8192, 128, 32), (64, 128, 4096, 64, 64)],
)
def test_quantized_matmul_heads_rope_quant_matches_unfused(
    monkeypatch, m, k, n, block_rows, pass_rows
):
    monkeypatch.setattr(rope, "HEADS_BLOCK_ROWS", block_rows)
    monkeypatch.setattr(rope, "HEADS_MIN_ROWS", 32)
    monkeypatch.setattr(rope, "HEADS_QUANT_PASS_ROWS", pass_rows)
    x, w_q, w_s, positions, cos_sin_cache = _operands(m, k, n)
    got_q, got_s = _fused_rope_quant(
        x, w_q, w_s, positions, cos_sin_cache, interpret=True
    )
    q = xla_quantized_matmul(x, w_q, w_s).reshape(m, n // rope.LANE, rope.LANE)
    cos, sin = rope.cos_sin_lanes(cos_sin_cache[positions])
    y = rope.rope_lanes(q.astype(jnp.float32), cos[:, None], sin[:, None])
    want_q, want_s = rope._quantize_lanes(y, FP8)
    assert got_q.shape == want_q.shape and got_q.dtype == want_q.dtype
    assert got_s.shape == want_s.shape[:-1]
    np.testing.assert_array_equal(
        np.asarray(got_q).view(np.uint8), np.asarray(want_q).view(np.uint8)
    )
    # XLA:CPU may divide by the constant dtype max another way than the kernel.
    np.testing.assert_allclose(got_s, want_s[..., 0], rtol=1e-6)


@pytest.mark.tpu_test
@pytest.mark.parametrize("m", [128, 1024])
def test_quantized_matmul_heads_rope_quant_is_bit_identical_on_tpu(m):
    if jax.default_backend() != "tpu":
        pytest.fail(f"needs the TPU backend, got {jax.default_backend()}")
    k, n = 1024, 8192
    x, w_q, w_s, positions, cos_sin_cache = _operands(m, k, n)

    @jax.jit
    def unfused(x, w_q, w_s, positions, cos_sin_cache):
        q = xla_quantized_matmul(x, w_q, w_s, quantize_in_one_pass=True)
        return rope.rope_quant_flat(
            q, positions, cos_sin_cache, num_heads=n // rope.LANE, quant_dtype=FP8
        )

    got = jax.jit(_fused_rope_quant)(x, w_q, w_s, positions, cos_sin_cache)
    want = unfused(x, w_q, w_s, positions, cos_sin_cache)
    for g, w, bits in zip(got, want, (np.uint8, np.uint32), strict=True):
        np.testing.assert_array_equal(
            np.asarray(g).view(bits), np.asarray(w).view(bits)
        )


def test_signed_sin_rotation_is_bit_identical():
    keys = jax.random.split(jax.random.key(3), 3)
    x = jax.random.normal(keys[0], (64, 4, 128), jnp.float32)
    x = x * jnp.exp(jax.random.uniform(keys[1], (64, 4, 1), minval=-30, maxval=30))
    cos_sin = jax.random.uniform(keys[2], (64, 64), jnp.float32, -1, 1)
    cos, sin = rope.cos_sin_lanes(cos_sin)
    cos_s, signed_sin = rope.cos_signed_sin_lanes(cos_sin)
    want = rope.rope_lanes(x, cos[:, None], sin[:, None])
    got = rope.rope_lanes_signed(x, cos_s[:, None], signed_sin[:, None])
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint32), np.asarray(want).view(np.uint32)
    )
