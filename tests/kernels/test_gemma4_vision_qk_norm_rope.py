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
"""Correctness and speed of the fused Gemma-4 vision Q/K norm + RoPE kernel.

The golden is ``qk_norm_rope.reference``: unfused XLA with HF's structure
(per-head RMSNorm, then split into RoPE chunks, rotate_half, concat). Run this
file directly on a TPU host to also print a timing comparison.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gemma4_vision import qk_norm_rope as lib

pytestmark = pytest.mark.tpu_test

NUM_HEADS = 16
HEAD_DIM = 72
# 8 images x 2520 patches, the vision encoder batch in the serving trace.
SERVING_TOKENS = 8 * 2520


def _inputs(tokens: int, seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.normal(keys[0], (tokens, NUM_HEADS * HEAD_DIM), jnp.bfloat16)
    angle = jax.random.uniform(keys[1], (tokens, HEAD_DIM), jnp.float32, 0.0, 6.3)
    cos = jnp.cos(angle).astype(jnp.bfloat16)
    sin = jnp.sin(angle).astype(jnp.bfloat16)
    scale = (1.0 + 0.1 * jax.random.normal(keys[2], (HEAD_DIM,))).astype(jnp.float32)
    return x, cos, sin, scale


def _run_both(tokens: int, seed: int = 0):
    x, cos, sin, scale = _inputs(tokens, seed)
    kwargs = dict(num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    got = lib.qk_norm_rope(
        x,
        lib.tile_cos_sin(cos, NUM_HEADS),
        lib.tile_cos_sin(sin, NUM_HEADS),
        scale,
        **kwargs,
    )
    want = jax.jit(functools.partial(lib.reference, **kwargs))(x, cos, sin, scale)
    return np.asarray(got, np.float32), np.asarray(want, np.float32)


@pytest.mark.parametrize("tokens", [8, 256, 1000, SERVING_TOKENS])
def test_matches_reference(tokens):
    got, want = _run_both(tokens)
    max_diff = float(np.max(np.abs(got - want)))
    print(f"tokens={tokens} max_abs_diff={max_diff:.3e}")
    # Both compute in float32 and round once to bf16; allow one bf16 ulp of
    # rounding difference from a different summation order.
    np.testing.assert_allclose(got, want, rtol=2**-7, atol=2**-7)


def test_rotation_direction_is_hf_rotate_half():
    # Unit cos = 0, sin = 1 isolates rotate_half: out = rotate_half(norm(x)).
    tokens = 8
    x, _, _, scale = _inputs(tokens)
    zeros = jnp.zeros((tokens, HEAD_DIM), jnp.bfloat16)
    ones = jnp.ones((tokens, HEAD_DIM), jnp.bfloat16)
    kwargs = dict(num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    got = lib.qk_norm_rope(
        x,
        lib.tile_cos_sin(zeros, NUM_HEADS),
        lib.tile_cos_sin(ones, NUM_HEADS),
        scale,
        **kwargs,
    )
    want = lib.reference(x, zeros, ones, scale, **kwargs)
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=2**-7,
        atol=2**-7,
    )


def _bench(fn, args, iters: int = 100) -> float:
    out = fn(*args)
    out.block_until_ready()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    out.block_until_ready()
    return (time.perf_counter() - start) / iters * 1e6


if __name__ == "__main__":
    kwargs = dict(num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    for tokens in (8, 1000, SERVING_TOKENS):
        got, want = _run_both(tokens)
        print(
            f"[tpu] tokens={tokens} max_abs_diff="
            f"{float(np.max(np.abs(got - want))):.3e}"
        )
    x, cos, sin, scale = _inputs(SERVING_TOKENS)
    ref = jax.jit(functools.partial(lib.reference, **kwargs))
    ref_us = _bench(ref, (x, cos, sin, scale))
    # cos/sin are tiled once per forward and shared by every layer's Q and K.
    tiled = (x, lib.tile_cos_sin(cos, NUM_HEADS), lib.tile_cos_sin(sin, NUM_HEADS))
    for block in (128, 256, 512, 1024):
        fused_b = functools.partial(lib.qk_norm_rope, block_tokens=block, **kwargs)
        us = _bench(fused_b, (*tiled, scale))
        print(
            f"[tpu] norm+rope per call, tokens={SERVING_TOKENS}: "
            f"xla_unfused={ref_us:.1f} us, fused_kernel(block={block})={us:.1f} us, "
            f"speedup={ref_us / us:.2f}x"
        )
