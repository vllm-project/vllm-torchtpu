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
"""CPU tests of the r10 gmm_v2 keys gmm_fp8_w8a8 / gmm_scale_lookahead
(`gmm_v2.set_r10_options`).

gmm_v2's ref reshapes do not run in this JAX's Pallas interpreter, so the
programs are compared as TPU lowerings (tpu7x abstract mesh) and the fp8 x
fp8 tile arithmetic is checked on its helpers against an f64 reference.
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.megablox import gmm_v2 as G

TPU7X = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
K, N, E = 512, 256, 4


@pytest.fixture(autouse=True)
def _reset():
    G.set_r10_options()
    jax.clear_caches()
    yield
    G.set_r10_options()
    jax.clear_caches()


def _lowered(size_m: int, fuse_act=None) -> str:
    sd = jax.ShapeDtypeStruct
    n = 2 * N if fuse_act else N

    def f(lhs, rhs, sizes, scale):
        return G.gmm_v2(
            lhs, rhs, sizes, rhs_scale=scale, maybe_quantize_lhs=False, fuse_act=fuse_act
        )

    jax.clear_caches()
    with use_abstract_mesh(TPU7X):
        text = (
            jax.jit(f)
            .trace(
                sd((size_m, K), jnp.bfloat16),
                sd((E, K, n), jnp.float4_e2m1fn),
                sd((E,), jnp.int32),
                sd((E, K // 128, 1, n), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    return hashlib.sha1(text.encode()).hexdigest()


@pytest.mark.parametrize("size_m", [64, 1024, 4096])
def test_keys_off_keep_the_program(size_m):
    base = _lowered(size_m)
    G.set_r10_options(False, False)
    assert _lowered(size_m) == base


def test_w8a8_changes_prefill_programs_only():
    prefill, decode = _lowered(4096), _lowered(1024)
    G.set_r10_options(fp8_w8a8=True)
    assert _lowered(4096) != prefill
    assert _lowered(1024) == decode  # <= FP8_DECODE_MAX_M rows: untouched


def test_scale_lookahead_changes_decode_programs():
    decode = _lowered(64)
    G.set_r10_options(scale_lookahead=True)
    assert _lowered(64) != decode


def _cfgs(size_m):
    with use_abstract_mesh(TPU7X):
        lhs = jnp.zeros((size_m, K), jnp.bfloat16)
        rhs = jnp.zeros((E, K, N), jnp.float4_e2m1fn)
        scale = jnp.zeros((E, K // 128, 1, N), jnp.float32)
        return G.make_gmm_configs(
            lhs,
            rhs,
            scale,
            None,
            jnp.zeros((E,), jnp.int32),
            jnp.zeros((1,), jnp.int32),
            tile_info=G.calculate_tiling,
            vmem_limit_bytes=int(64 * 1024 * 1024 * 0.9),
            out_dtype=None,
            acc_dtype=None,
            maybe_quantize_lhs=False,
            zero_initialize=False,
        )


def test_w8a8_dispatch_and_tiles():
    cfgs = _cfgs(4096)
    with use_abstract_mesh(TPU7X):
        assert not any(cfgs.uses_fp8_w8a8(b) for b in cfgs.tiles.bucket_sizes)
        G.set_r10_options(fp8_w8a8=True)
        # Tile choice does not depend on the key.
        assert _cfgs(4096).tiles == cfgs.tiles
        big = [b for b in cfgs.tiles.bucket_sizes if b > G.FP8_RHS_MAX_BUCKET_M]
        assert big and all(cfgs.uses_fp8_w8a8(b) for b in big)
        assert not cfgs.uses_fp8_w8a8(G.FP8_RHS_MAX_BUCKET_M)
        small = _cfgs(2048)
        assert not any(small.uses_fp8_w8a8(b) for b in small.tiles.bucket_sizes)


def test_scale_buffering_and_vmem_estimate():
    lhs = G.InputConfigs(quant_dtype=None, quant_block_size=K, dtype=jnp.bfloat16)
    rhs = G.InputConfigs(
        quant_dtype=jnp.float4_e2m1fn,
        quant_block_size=128,
        dtype=jnp.float4_e2m1fn,
        has_scale=True,
    )

    def est(tm):
        return G.gmm_vmem_estimate(
            tm, 256, 1024, lhs_cfgs=lhs, rhs_cfgs=rhs, fuse_act=None
        )

    assert G._scale_buffering(32) is None
    off = {tm: est(tm) for tm in (32, 128, 256)}
    G.set_r10_options(scale_lookahead=True)
    assert G._scale_buffering(32).buffer_count == 3
    assert G._scale_buffering(256).buffer_count == 2
    scale_bytes = 1024 // 128 * 256 * 4
    assert est(32) == off[32] + scale_bytes
    assert est(128) == off[128] + scale_bytes
    assert est(256) == off[256]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_w8a8_tile_arithmetic_vs_f64(seed):
    """One tile as the kernel computes it under gmm_fp8_w8a8: rhs fp4 with
    per-128 block scales requantized to e4m3 per column, lhs to e4m3 per row,
    one full-depth matmul, scaled at the end. e4m3 keeps 3 mantissa bits
    (relative step 2^-4 = 6.25%), so each product carries ~3% error per
    operand; summed over K=512 random-sign terms the output error is a few
    percent of the output RMS. Tolerance: relative RMS error < 6% (r10's
    measured output error on the model); the default bf16 path must sit
    closer to the reference."""
    rng = np.random.default_rng(seed)
    m, qbs = 128, 128
    w = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    blocks = w.reshape(K // qbs, qbs, N)
    block_scale = np.abs(blocks).max(axis=1, keepdims=True) / 6.0
    w4 = jnp.asarray(blocks / block_scale).astype(jnp.float4_e2m1fn).reshape(K, N)
    scale = jnp.asarray(block_scale.astype(np.float32))  # [K/qbs, 1, N]
    x = jnp.asarray(rng.standard_normal((m, K)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    w_true = (
        np.asarray(w4.astype(jnp.float32)).reshape(K // qbs, qbs, N)
        * block_scale
    ).reshape(K, N).astype(np.float64)
    ref = np.asarray(x.astype(jnp.float32)).astype(np.float64) @ w_true

    w8, col_scale = G.requantize_fp8_colscale(w4, scale, qbs, jnp.bfloat16)
    x8, row_scale = G.quantize_rows_fp8(x)
    assert w8.dtype == x8.dtype == jnp.float8_e4m3fn
    acc = jnp.matmul(x8, w8, preferred_element_type=jnp.float32)
    got = np.asarray(acc * col_scale * row_scale, np.float64)

    rel = np.sqrt(np.mean((got - ref) ** 2) / np.mean(ref**2))
    assert rel < 0.06, rel
    # The default (bf16 dequantize-before) path for comparison.
    w_bf16 = (w4.astype(jnp.bfloat16).reshape(K // qbs, qbs, N) * scale.astype(
        jnp.bfloat16
    )).reshape(K, N)
    base = np.asarray(
        jnp.matmul(x, w_bf16, preferred_element_type=jnp.float32), np.float64
    )
    rel_base = np.sqrt(np.mean((base - ref) ** 2) / np.mean(ref**2))
    assert rel_base < rel  # the lossy key costs precision, as documented
