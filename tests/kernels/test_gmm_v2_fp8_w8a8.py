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
"""CPU tests of TPU_GMM_FP8_W8A8 (gmm_v2 prefill buckets run fp8 x fp8).

Numerics: the in-kernel requantization (fp4 per-128-K-block scales -> e4m3
per-column scales, bf16 lhs -> e4m3 per-row scales) in a Pallas interpret
kernel against a jnp reference. Selection: which buckets / calls take it,
key-off tiles unchanged, the k1 body fallback. Lowering: the full gmm_v2
kernel lowers for tpu7x with the switch on and only prefill calls change.
"""

import base64
import dataclasses
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.megablox import gmm_v2 as G

ENV = "TPU_GMM_FP8_W8A8"
F4 = jnp.float4_e2m1fn
H, FE, LOCAL = 6144, 3072, 16  # MiniMax-M3.1-Flash per-core MoE (EP8)
VMEM = int(64 * 1024 * 1024 * 0.9)


def _fp4_weights(key, k, n, qbs=128):
    w = jax.random.normal(key, (k, n), jnp.float32)
    wb = w.reshape(k // qbs, qbs, n)
    scale = jnp.max(jnp.abs(wb), axis=1, keepdims=True) / float(jnp.finfo(F4).max)
    wq = (wb / scale).astype(F4).reshape(k, n)
    return wq, scale.astype(jnp.float32)  # [k, n] fp4, [k // qbs, 1, n] f32


def _dequant(wq, scale, qbs=128):
    k, n = wq.shape
    return (wq.astype(jnp.float32).reshape(k // qbs, qbs, n) * scale).reshape(k, n)


def _w8a8_kernel(lhs_ref, w_ref, s_ref, out_ref, *, qbs, mxu):
    # Same steps as gmm_v2's _matmul on a TPU_GMM_FP8_W8A8 bucket.
    rhs, col_scale = G.requantize_fp8_colscale(
        w_ref[...], s_ref[...], qbs, lhs_ref.dtype
    )
    lhs, row_scale = G.quantize_rows_fp8(lhs_ref[...])
    accs = []
    for start_n in range(0, rhs.shape[1], mxu):
        acc = jnp.matmul(
            lhs, rhs[:, start_n : start_n + mxu], preferred_element_type=jnp.float32
        )
        accs.append(acc * (col_scale[:, start_n : start_n + mxu] * row_scale))
    out_ref[...] = jnp.concatenate(accs, axis=1)


@pytest.mark.parametrize("m,k,n", [(64, 256, 256), (320, 512, 512)])
def test_w8a8_kernel_math_interpret(m, k, n):
    k1, k2 = jax.random.split(jax.random.key(m))
    lhs = jax.random.normal(k1, (m, k), jnp.float32).astype(jnp.bfloat16)
    lhs = lhs.at[3].set(0)  # all-zero row: scale 0, output 0 (no NaN)
    wq, scale = _fp4_weights(k2, k, n)
    out = pl.pallas_call(
        lambda *r: _w8a8_kernel(*r, qbs=128, mxu=256),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        interpret=True,
    )(lhs, wq, scale)
    out = np.asarray(out)
    ref = np.asarray(lhs.astype(jnp.float32) @ _dequant(wq, scale))
    assert np.isfinite(out).all() and np.all(out[3] == 0)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    # Lossy by design (~6% on model outputs); e4m3 x e4m3 on random data.
    assert 0 < rel < 0.08, rel

    # The kernel matches the same math traced outside Pallas.
    rhs8, cs = G.requantize_fp8_colscale(wq, scale, 128, jnp.bfloat16)
    lhs8, rs = G.quantize_rows_fp8(lhs)
    jref = np.asarray(
        jnp.matmul(lhs8, rhs8, preferred_element_type=jnp.float32) * cs * rs
    )
    np.testing.assert_allclose(out, jref, rtol=1e-5, atol=1e-5 * np.abs(jref).max())


def test_requantize_colscale_roundtrip():
    wq, scale = _fp4_weights(jax.random.key(1), 512, 256)
    w8, cs = G.requantize_fp8_colscale(wq, scale, 128, jnp.bfloat16)
    assert w8.dtype == jnp.float8_e4m3fn and cs.shape == (1, 256)
    back = np.asarray(w8.astype(jnp.float32) * cs)
    ref = np.asarray(_dequant(wq, scale))
    # The block with the column's largest scale maps fp4 max to e4m3 max.
    assert np.abs(w8.astype(jnp.float32)).max() <= float(jnp.finfo(w8.dtype).max)
    assert np.linalg.norm(back - ref) / np.linalg.norm(ref) < 0.05


def test_quantize_rows():
    x = jax.random.normal(jax.random.key(2), (32, 256), jnp.float32)
    x = x.astype(jnp.bfloat16).at[0].set(0)
    x8, rs = G.quantize_rows_fp8(x)
    assert x8.dtype == jnp.float8_e4m3fn and rs.shape == (32, 1)
    back = np.asarray(x8.astype(jnp.float32) * rs)
    ref = np.asarray(x.astype(jnp.float32))
    assert np.all(back[0] == 0)
    assert np.linalg.norm(back - ref) / np.linalg.norm(ref) < 0.05


# --- selection ------------------------------------------------------------


@pytest.fixture
def tpu7x(monkeypatch):
    monkeypatch.setattr(
        G.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            num_lanes=128,
            mxu_column_size=256,
            vmem_capacity_bytes=64 * 1024 * 1024,
            is_matmul_supported=lambda a, b: True,
            get_sublane_tiling=lambda dtype: 16,
        ),
    )
    for name in (ENV, G._PREFILL_TILES_ENV, G._DECODE_W13_ENV, G._K1_BODY_ENV):
        monkeypatch.delenv(name, raising=False)
    G._tile_env.cache_clear()
    G._k1_body_env.cache_clear()
    yield monkeypatch
    G._tile_env.cache_clear()
    G._k1_body_env.cache_clear()


def _call(size_m, gate_up):
    k, n = (H, 2 * FE) if gate_up else (FE, H)
    dims = G.Dimensions(
        size_m=size_m,
        size_k=k,
        size_n=n,
        size_group=LOCAL,
        size_lhs_group=LOCAL,
        size_lhs_sublane=16,
    )
    lhs = G.InputConfigs(quant_dtype=None, quant_block_size=k, dtype=jnp.bfloat16)
    rhs = G.InputConfigs(quant_dtype=F4, quant_block_size=128, dtype=F4, has_scale=True)
    act = "swigluoai" if gate_up else None
    tiles = G.calculate_tiling(dims, lhs, rhs, VMEM, act)
    return G.GmmConfigs(
        tiles=tiles,
        dims=dims,
        lhs_cfgs=lhs,
        rhs_cfgs=rhs,
        out_dtype=jnp.dtype(jnp.bfloat16),
        acc_dtype=jnp.dtype(jnp.float32),
        zero_init=True,
        fuse_act=act,
    )


# Tiles of main before this switch existed (key-off must keep them):
# (tile_m, tile_k, tile_n, buckets) per (size_m, gate_up).
_MAIN_TILES = {
    (128, True): (32, 6144, 1536, (16, 32)),
    (2048, True): (128, 6144, 1536, (16, 32, 48, 128)),
    (2049, True): (256, 6144, 1536, (64, 128, 192, 256)),
    (8192, True): (256, 6144, 1536, (64, 128, 192, 256)),
    (128, False): (32, 3072, 6144, (16, 32)),
    (2048, False): (128, 3072, 6144, (16, 32, 48, 128)),
    (2049, False): (256, 3072, 6144, (64, 128, 192, 256)),
    (8192, False): (256, 3072, 6144, (64, 128, 192, 256)),
}


def _key(t):
    return (t.tile_m, t.tile_k, t.tile_n, t.bucket_sizes)


@pytest.mark.parametrize("size_m,gate_up", sorted(_MAIN_TILES))
def test_key_off_tiles_and_buckets_unchanged(tpu7x, size_m, gate_up):
    cfgs = _call(size_m, gate_up)
    assert _key(cfgs.tiles) == _MAIN_TILES[(size_m, gate_up)]
    assert cfgs.tiles.rhs_buffer_count == 3
    assert not any(cfgs.uses_fp8_w8a8(b) for b in cfgs.tiles.bucket_sizes)
    assert G._w8a8_buckets(cfgs) == ()
    assert not any("w8a8" in k for k in G.get_metadata(cfgs))


@pytest.mark.parametrize("gate_up", [True, False])
def test_key_on_prefill_buckets(tpu7x, gate_up):
    tpu7x.setenv(ENV, "1")
    cfgs = _call(8192, gate_up)
    # A fifth 64-row bucket for prefill chunks; all take fp8 x fp8.
    assert cfgs.tiles.bucket_sizes == (64, 128, 192, 256, 320)
    assert G._w8a8_buckets(cfgs) == cfgs.tiles.bucket_sizes
    assert G.get_metadata(cfgs)["fp8_w8a8_buckets"] == str(cfgs.tiles.bucket_sizes)
    # Decode-sized calls are not covered (tiles and feeds unchanged).
    for size_m in (128, 2048):
        dec = _call(size_m, gate_up)
        assert _key(dec.tiles) == _MAIN_TILES[(size_m, gate_up)]
        assert G._w8a8_buckets(dec) == ()


def test_key_on_composes_with_prefill_tiles(tpu7x):
    tpu7x.setenv(ENV, "1")
    tpu7x.setenv(G._PREFILL_TILES_ENV, "256:32/128/256")
    cfgs = _call(8192, True)
    # The explicit tile table wins; buckets <= 48 rows keep the fp8 rhs feed.
    assert cfgs.tiles.bucket_sizes == (32, 128, 256)
    assert cfgs.fp8_rhs_dtype(32) is not None and not cfgs.uses_fp8_w8a8(32)
    assert G._w8a8_buckets(cfgs) == (128, 256)


def test_needs_fp4_rhs_and_fp8_mxu(tpu7x):
    tpu7x.setenv(ENV, "1")
    cfgs = _call(8192, True)
    no_fp8 = SimpleNamespace(
        num_lanes=128,
        mxu_column_size=256,
        vmem_capacity_bytes=64 << 20,
        is_matmul_supported=lambda a, b: a != jnp.float8_e4m3fn.dtype,
        get_sublane_tiling=lambda dtype: 16,
    )
    tpu7x.setattr(G.pltpu, "get_tpu_info", lambda: no_fp8)
    assert not cfgs.uses_fp8_w8a8(256)
    tpu7x.setattr(
        G.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            **{**vars(no_fp8), "is_matmul_supported": lambda a, b: True}
        ),
    )
    int8_rhs = G.InputConfigs(
        quant_dtype=jnp.int8, quant_block_size=128, dtype=jnp.int8, has_scale=True
    )
    assert not dataclasses.replace(cfgs, rhs_cfgs=int8_rhs).uses_fp8_w8a8(256)


@pytest.mark.parametrize("mode", ["hoist", "pipe2ci", "roll"])
def test_key_on_overrides_k1_chunk_modes(tpu7x, mode):
    tpu7x.setenv(G._K1_BODY_ENV, mode)
    assert G._k1_body_mode(_call(8192, True)) == mode  # key-off: unchanged
    tpu7x.setenv(ENV, "1")
    assert G._k1_body_mode(_call(8192, True)) == "single"
    # Decode-sized calls are not covered: whatever main picks there.
    tpu7x.delenv(ENV)
    off = G._k1_body_mode(_call(2048, True))
    tpu7x.setenv(ENV, "1")
    assert G._k1_body_mode(_call(2048, True)) == off


# --- lowering (tpu7x, no device) -----------------------------------------


def _lowered_body(size_m, fuse_act):
    from jax._src.interpreters import mlir
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
    from jaxlib.mlir import ir

    k, n, e = 1024, 1024, 4
    args = (
        jax.ShapeDtypeStruct((size_m, k), jnp.bfloat16),
        jax.ShapeDtypeStruct((e, k, n), F4),
        jax.ShapeDtypeStruct((e,), jnp.int32),
        jax.ShapeDtypeStruct((e, k // 128, 1, n), jnp.float32),
    )

    def f(lhs, w, gs, s):
        return G.gmm_v2(lhs, w, gs, rhs_scale=s, rhs_quant_dtype=F4, fuse_act=fuse_act)

    jax.clear_caches()  # gmm_v2 is jitted; the switch is read at trace time
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        txt = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    body = re.search(r"\\22body\\22: \\22([A-Za-z0-9+/=]+)\\22", txt).group(1)
    from jaxlib.mosaic.python import tpu as tpu_dialect

    with mlir.make_ir_context() as ctx:
        tpu_dialect.register_dialect(ctx)
        ctx.allow_unregistered_dialects = True
        mod = ir.Module.parse(base64.b64decode(body))
        return mod.operation.get_asm(enable_debug_info=False)


@pytest.mark.parametrize("fuse_act", [None, "swigluoai"])
def test_lowering_tpu7x(monkeypatch, fuse_act):
    pytest.importorskip("jaxlib.mlir")
    for name in (ENV, G._PREFILL_TILES_ENV, G._DECODE_W13_ENV, G._K1_BODY_ENV):
        monkeypatch.delenv(name, raising=False)
    G._tile_env.cache_clear()
    G._k1_body_env.cache_clear()
    pre_off, dec_off = _lowered_body(4096, fuse_act), _lowered_body(512, fuse_act)
    monkeypatch.setenv(ENV, "1")
    pre_on, dec_on = _lowered_body(4096, fuse_act), _lowered_body(512, fuse_act)
    assert dec_on == dec_off  # decode-sized: same kernel
    assert pre_on != pre_off
    # Prefill: e4m3 x e4m3 matmuls appear only with the switch on.
    f8 = r"vector<[0-9x]+xf8E4M3FN>"
    fp8_mm = re.compile(rf"tpu\.matmul.*\({f8}, {f8}")
    assert fp8_mm.search(pre_on) and not fp8_mm.search(pre_off)
