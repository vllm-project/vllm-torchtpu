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
"""CPU race-checked interpreter and TPU tests for VMEM gather-fused MoE kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import gather_gmm as G
from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_gated_act
from vllm_torchtpu.layers.core import fused_moe_gmm

RACE_INTERPRET = pltpu.InterpretParams(detect_races=True)


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("TPU device required")


def _pack_rows(x: jax.Array) -> jax.Array:
    """Pack BF16 ``[n, hidden]`` into uint32 ``[n * hidden // 256, 128]``."""
    n, h = x.shape
    s = G.packed_sublanes(h)
    bits = jax.lax.bitcast_convert_type(x.reshape(n, s, 2, 128), jnp.uint16)
    bits = bits.astype(jnp.uint32)
    return (bits[:, :, 0, :] | (bits[:, :, 1, :] << 16)).reshape(n * s, 128)


def _quantize(block: jax.Array) -> tuple[jax.Array, jax.Array]:
    amax = jnp.max(jnp.abs(block), axis=1, keepdims=True)
    scale = amax / float(jnp.finfo(jnp.float8_e4m3fn).max)
    inv = jnp.where(scale == 0, 0, 1 / scale)
    return (block * inv).astype(jnp.float8_e4m3fn), scale.astype(jnp.bfloat16)


def _expert_rows(xs, w1, w1s, w2, w2s, act):
    """Per-row ``gmm_v2`` reference arithmetic for one expert's routed rows."""
    if xs.shape[0] == 0:
        return jnp.zeros((0, w2.shape[1]), jnp.bfloat16)
    h = xs.shape[1]
    n1 = w1.shape[1]
    acc = jnp.zeros((xs.shape[0], n1), jnp.bfloat16)
    for k0 in range(0, h, G.QUANT_BLOCK):
        k1 = min(h, k0 + G.QUANT_BLOCK)
        q, sc = _quantize(xs[:, k0:k1])
        for n0 in range(0, n1, 256):
            prod = jnp.dot(
                q, w1[k0:k1, n0 : n0 + 256], preferred_element_type=jnp.float32
            )
            prod = (
                prod.astype(jnp.bfloat16)
                * sc
                * w1s[:, n0 : n0 + 256].astype(jnp.bfloat16)
            )
            acc = acc.at[:, n0 : n0 + 256].add(prod)
    inter = n1 // 2
    mid = apply_gated_act(acc[:, :inter], acc[:, inter:], act)
    out = jnp.zeros((xs.shape[0], w2.shape[1]), jnp.bfloat16)
    for k0 in range(0, inter, G.QUANT_BLOCK):
        k1 = min(inter, k0 + G.QUANT_BLOCK)
        q, sc = _quantize(mid[:, k0:k1])
        prod = jnp.dot(q, w2[k0:k1], preferred_element_type=jnp.float32)
        out = out + prod.astype(jnp.bfloat16) * sc * w2s.astype(jnp.bfloat16)
    return out


def _reference(x, token_ids, group_sizes, w1, w1s, w2, w2s, act):
    xs = x[token_ids]
    outs, start = [], 0
    for e, n in enumerate(np.asarray(group_sizes)):
        count = int(n)
        outs.append(
            _expert_rows(xs[start : start + count], w1[e], w1s[e], w2[e], w2s[e], act)
        )
        start += count
    if not outs:
        return jnp.zeros((0, x.shape[1]), jnp.bfloat16)
    return jnp.concatenate(outs, axis=0)


def _routing(key, tokens, experts, topk, skew, weights_dtype=jnp.bfloat16):
    logits = jax.random.normal(key, (tokens, experts)) + skew * jnp.arange(experts)
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    flat = ids.reshape(-1)
    order = jnp.argsort(flat, stable=True)
    group_sizes = jnp.bincount(flat, length=experts).astype(jnp.int32)
    positions = jnp.argsort(order).reshape(tokens, topk).astype(jnp.int32)
    return (
        (order // topk).astype(jnp.int32),
        group_sizes,
        positions,
        weights.astype(weights_dtype),
    )


def _weights(key, experts, hidden, inter):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = (jax.random.normal(k1, (experts, hidden, 2 * inter)) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    w2 = (jax.random.normal(k2, (experts, inter, hidden)) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    w1s = jax.random.uniform(k3, (experts, 1, 2 * inter), minval=0.01, maxval=0.03)
    w2s = jax.random.uniform(k4, (experts, 1, hidden), minval=0.01, maxval=0.03)
    return w1, w1s, w2, w2s


def test_pack_rows_roundtrip():
    x = jax.random.normal(jax.random.PRNGKey(0), (6, 768)).astype(jnp.bfloat16)
    packed = _pack_rows(x)
    assert packed.shape == (6 * 3, 128) and packed.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(G.unpack_rows(packed, 6)), np.asarray(x))


@pytest.mark.parametrize("tokens", [48, 1088])
def test_quantize_pack_rows_layout(tokens):
    hidden = 768
    with jax.default_device(jax.devices("cpu")[0]):
        x = (jax.random.normal(jax.random.PRNGKey(4), (tokens, hidden)) * 3).astype(
            jnp.bfloat16
        )
        x = x.at[5].set(0)
        packed = G.quantize_pack_rows(x, tile_tokens=16, interpret=RACE_INTERPRET)
        rows = np.asarray(packed).reshape(tokens, 4, 128)
        want_scaled, want_scales = [], []
        for k0 in range(0, hidden, G.QUANT_BLOCK):
            scaled, scale = jax.jit(G._scale_rows)(x[:, k0 : k0 + G.QUANT_BLOCK])
            want_scaled.append(scaled)
            want_scales.append(scale.astype(jnp.float32))
        got_scaled = G.unpack_rows(jnp.asarray(rows[:, :3].reshape(-1, 128)), tokens)
        np.testing.assert_array_equal(
            np.asarray(got_scaled), np.asarray(jnp.concatenate(want_scaled, axis=1))
        )
        got_scales = rows[:, 3, :2].view(np.float32)
        np.testing.assert_array_equal(
            got_scales, np.asarray(jnp.concatenate(want_scales, axis=1))
        )
        assert not rows[:, 3, 2:].any()


def test_select_tile_m_adapts_to_vmem_budget():
    limit_64m = int((64 << 20) * G.VMEM_LIMIT_FRACTION)
    limit_32m = int((32 << 20) * G.VMEM_LIMIT_FRACTION)

    assert G.select_tile_m(2816, 384, vmem_limit_bytes=limit_64m) == 640
    assert G.select_tile_m(2816, 1536, vmem_limit_bytes=limit_64m) == 640
    shrunk = G.select_tile_m(2816, 2048, vmem_limit_bytes=limit_64m)
    assert shrunk is not None and 128 <= shrunk < 640
    assert G.select_tile_m(2816, 3072, vmem_limit_bytes=limit_64m) is None

    shrunk_32m = G.select_tile_m(2816, 768, vmem_limit_bytes=limit_32m)
    assert shrunk_32m is not None and 128 <= shrunk_32m < 640

    v7x_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    v7x_limit = int(v7x_info.vmem_capacity_bytes * G.VMEM_LIMIT_FRACTION)
    assert G.select_tile_m(2816, 1536, vmem_limit_bytes=v7x_limit) == 640
    assert G.select_tile_m(2816, 32768, vmem_limit_bytes=v7x_limit) is None


@pytest.mark.parametrize(
    "tokens,experts,topk,tile_m,skew,inter",
    [
        (64, 8, 2, 32, 0.0, 128),  # >= 4 tiles across experts, partial tail tiles
        (48, 8, 4, 32, 0.5, 128),  # skewed routing: empty and multi-tile experts
        (48, 4, 2, 32, 3.0, 128),  # one expert takes 3+ tiles, reusing DMA slots
        (64, 8, 2, 64, 0.0, 640),  # intermediate spanning > 1 quantization block
    ],
)
def test_gather_fused_moe_matches_reference(tokens, experts, topk, tile_m, skew, inter):
    hidden = 768
    with jax.default_device(jax.devices("cpu")[0]):
        kx, kr, kw = jax.random.split(jax.random.PRNGKey(1), 3)
        x = (jax.random.normal(kx, (tokens, hidden)) * 2.0).astype(jnp.bfloat16)
        token_ids, group_sizes, _, _ = _routing(kr, tokens, experts, topk, skew)
        w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)
        out = G.gather_fused_moe(
            G.quantize_pack_rows(x, tile_tokens=16, interpret=RACE_INTERPRET),
            token_ids,
            group_sizes,
            w1,
            w1s,
            w2,
            w2s,
            act="gelu_tanh",
            tile_m=tile_m,
            height_step=tile_m // 2,
            interpret=RACE_INTERPRET,
        )
        rows = tokens * topk
        got = G.unpack_rows(out, rows)
        want = _reference(x, token_ids, group_sizes, w1, w1s, w2, w2s, "gelu_tanh")
        assert np.isfinite(np.asarray(want, np.float32)).all()
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("weights_dtype", [jnp.bfloat16, jnp.float32])
def test_gather_combine_matches_reference(weights_dtype):
    tokens, topk, hidden = 64, 4, 768  # 4 tiles of 16 tokens -> DMA slot reuse
    with jax.default_device(jax.devices("cpu")[0]):
        kr, ky = jax.random.split(jax.random.PRNGKey(2))
        _, _, positions, weights = _routing(
            kr, tokens, 16, topk, 0.0, weights_dtype=weights_dtype
        )
        y = jax.random.normal(ky, (tokens * topk, hidden)).astype(jnp.bfloat16)
        out = G.gather_combine(
            _pack_rows(y),
            positions,
            weights,
            hidden_size=hidden,
            tile_tokens=16,
            interpret=RACE_INTERPRET,
        )
        assert out.dtype == jnp.bfloat16
        gathered = (
            y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
        )
        acc = jnp.zeros((tokens, hidden), jnp.float32)
        for k in range(topk):
            acc = acc + gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(out), np.asarray(acc.astype(jnp.bfloat16))
        )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("weights_dtype", [jnp.bfloat16, jnp.float32])
def test_gather_combine_in_chunks_matches_one_call(masked, weights_dtype):
    tokens, topk, hidden, tt = 80, 4, 768, 16
    with jax.default_device(jax.devices("cpu")[0]):
        kr, ky, km = jax.random.split(jax.random.PRNGKey(8), 3)
        _, _, positions, weights = _routing(
            kr, tokens, 16, topk, 0.0, weights_dtype=weights_dtype
        )
        if masked:
            positions = jnp.where(
                jax.random.bernoulli(km, 0.7, (tokens, topk)), positions, -1
            )
        y = _pack_rows(
            jax.random.normal(ky, (tokens * topk, hidden)).astype(jnp.bfloat16)
        )
        kwargs = dict(
            hidden_size=hidden, tile_tokens=tt, masked=masked, interpret=RACE_INTERPRET
        )
        want = G.gather_combine(y, positions, weights, **kwargs)
        got = G.gather_combine_in_chunks(
            y, positions, weights, max_routes=2 * tt * topk + 1, **kwargs
        )
        assert got.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_gather_combine_masked_routes():
    """Verify negative positions, empty tiles, stale NaNs, and spare-slot DMAs."""
    tokens, topk, hidden, tt = 64, 4, 768, 16  # 4 tiles: slots 0, 1, 0, 1
    with jax.default_device(jax.devices("cpu")[0]):
        kr, ky, km = jax.random.split(jax.random.PRNGKey(6), 3)
        _, _, positions, weights = _routing(
            kr, tokens, 16, topk, 0.0, weights_dtype=jnp.float32
        )
        valid = jax.random.bernoulli(km, 0.45, (tokens, topk))
        # Tile 0 route (0, 0) reads a NaN row; tile 1 is completely masked (0 valid);
        # tile 2 route (32, 0) reuses slot 0 while masked; tile 3 has partial padding.
        valid = valid.at[0, 0].set(True)
        valid = valid.at[tt : 2 * tt, :].set(False)
        valid = valid.at[2 * tt, 0].set(False)
        valid = valid.at[3 * tt : 4 * tt, :].set(False).at[3 * tt, :3].set(True)
        y = jax.random.normal(ky, (tokens * topk, hidden)).astype(jnp.bfloat16)
        y = y.at[positions[0, 0]].set(jnp.nan)
        out = G.gather_combine(
            _pack_rows(y),
            jnp.where(valid, positions, -1),
            weights,
            hidden_size=hidden,
            tile_tokens=tt,
            masked=True,
            interpret=RACE_INTERPRET,
        )
        gathered = (
            y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
        )
        acc = jnp.zeros((tokens, hidden), jnp.float32)
        for k in range(topk):
            prod = gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
            acc = acc + jnp.where(valid[:, k : k + 1], prod, 0.0)
        got = np.asarray(out, np.float32)
        assert np.isnan(got[0]).all() and np.isfinite(got[1:]).all()
        np.testing.assert_array_equal(got[tt : 2 * tt], 0.0)
        np.testing.assert_allclose(
            got[1:],
            np.asarray(acc.astype(jnp.bfloat16), np.float32)[1:],
            rtol=2**-7,
            atol=1e-6,
        )


@pytest.mark.parametrize(
    "mode", ["unmasked", "ep_masked", "ep_skip_padded", "ep_all_remote"]
)
def test_routed_moe_end_to_end_cpu(mode):
    tokens, experts, topk, hidden, inter = 64, 8, 4, 768, 128
    with jax.default_device(jax.devices("cpu")[0]):
        kx, kr, kw = jax.random.split(jax.random.PRNGKey(7), 3)
        x = jax.random.normal(kx, (tokens, hidden)).astype(jnp.bfloat16)
        w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)

        if mode == "unmasked":
            token_ids, group_sizes, positions, weights = _routing(
                kr, tokens, experts, topk, 0.3, weights_dtype=jnp.float32
            )
            masked = False
            valid = jnp.ones((tokens, topk), dtype=jnp.bool_)
        else:
            logits = jax.random.normal(kr, (tokens, 2 * experts))
            weights, global_ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
            weights = weights.astype(jnp.float32)
            if mode == "ep_all_remote":
                ids = jnp.full((tokens, topk), -1, dtype=jnp.int32)
            else:
                ids = jnp.where(global_ids < experts, global_ids, -1).astype(jnp.int32)
            skip_padded = mode == "ep_skip_padded"
            if skip_padded:
                weights = weights.at[-8:].set(0.0)
            valid = (ids >= 0) & (weights != 0) if skip_padded else ids >= 0
            flat_valid = valid.reshape(-1)
            keys = jnp.where(flat_valid, ids.reshape(-1), experts)
            order = jnp.argsort(keys, stable=True)
            group_sizes = jnp.bincount(keys, length=experts + 1)[:experts].astype(
                jnp.int32
            )
            positions = jnp.argsort(order).reshape(tokens, topk).astype(jnp.int32)
            token_ids = (order // topk).astype(jnp.int32)
            masked = True

        rows = G.gather_fused_moe(
            G.quantize_pack_rows(x, tile_tokens=16, interpret=RACE_INTERPRET),
            token_ids,
            group_sizes,
            w1,
            w1s,
            w2,
            w2s,
            act="gelu_tanh",
            tile_m=32,
            height_step=16,
            interpret=RACE_INTERPRET,
        )
        out = G.gather_combine(
            rows,
            jnp.where(valid, positions, -1) if masked else positions,
            weights,
            hidden_size=hidden,
            tile_tokens=16,
            masked=masked,
            interpret=RACE_INTERPRET,
        )
        assert out.dtype == jnp.bfloat16
        if mode == "ep_all_remote":
            np.testing.assert_array_equal(np.asarray(out), 0)
            return

        n_valid = int(group_sizes.sum())
        y = _reference(
            x, token_ids[:n_valid], group_sizes, w1, w1s, w2, w2s, "gelu_tanh"
        )
        y = jnp.concatenate(
            [y, jnp.full((token_ids.shape[0] - n_valid, hidden), jnp.nan, y.dtype)],
            axis=0,
        )
        gathered = (
            y[positions.reshape(-1)].reshape(tokens, topk, hidden).astype(jnp.float32)
        )
        acc = jnp.zeros((tokens, hidden), jnp.float32)
        for k in range(topk):
            prod = gathered[:, k] * weights[:, k : k + 1].astype(jnp.float32)
            acc = acc + jnp.where(valid[:, k : k + 1], prod, 0.0)
        np.testing.assert_allclose(
            np.asarray(out, np.float32),
            np.asarray(acc.astype(jnp.bfloat16), np.float32),
            rtol=2**-7,
            atol=1e-6,
        )


@pytest.mark.parametrize("tokens", [2048, 20480])
def test_lowers_for_tpu7x(tokens):
    """Verify AOT TPU7x lowering for single-call and >128Ki chunked combine."""
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    e, h, inter, topk = 32, 1024, 256, 8
    f8 = jnp.float8_e4m3fn
    hs = jax.ShapeDtypeStruct((tokens, h), jnp.bfloat16)
    w1 = jax.ShapeDtypeStruct((e, h, 2 * inter), f8)
    w2 = jax.ShapeDtypeStruct((e, inter, h), f8)
    w1s = jax.ShapeDtypeStruct((e, 1, 1, 2 * inter), jnp.float32)
    w2s = jax.ShapeDtypeStruct((e, 1, 1, h), jnp.float32)
    tw = jax.ShapeDtypeStruct((tokens, topk), jnp.float32)
    ti = jax.ShapeDtypeStruct((tokens, topk), jnp.int32)
    with jax.sharding.use_abstract_mesh(mesh):
        hlo = (
            jax.jit(
                lambda a, b, c, d, e_s, f, g: fused_moe_gmm.gather_gmm_moe(
                    a,
                    b,
                    c,
                    d,
                    e_s,
                    f,
                    g,
                    topk=topk,
                    activation="silu",
                    skip_padded_tokens=False,
                    masked=True,
                )
            )
            .trace(hs, w1, w2, w1s, w2s, tw, ti)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    assert "gather_fused_moe" in hlo
    assert "gather_combine" in hlo


@pytest.mark.tpu_test
@pytest.mark.parametrize("topk_weights_fp32", [False, True])
def test_tpu_gather_gmm_unmasked_matches_gmm(monkeypatch, topk_weights_fp32):
    _require_tpu()
    tokens, experts, topk, hidden, inter = 1024, 16, 4, 1024, 256
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(11), 3)
    x = jax.random.normal(kx, (tokens, hidden)).astype(jnp.bfloat16)
    logits = jax.random.normal(kr, (tokens, experts))
    weights, ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    weights = weights.astype(jnp.float32)
    ids = ids.astype(jnp.int32)
    w1, w1s, w2, w2s = _weights(kw, experts, hidden, inter)
    w1s_4d = jnp.expand_dims(w1s, 2)
    w2s_4d = jnp.expand_dims(w2s, 2)

    monkeypatch.setattr(
        fused_moe_gmm.envs, "TPU_MOE_TOPK_WEIGHTS_FLOAT32", topk_weights_fp32
    )
    monkeypatch.setattr(fused_moe_gmm, "_GATHER_GMM", False)
    fused_moe_gmm.fused_moe_func.clear_cache()
    want = fused_moe_gmm.fused_moe_func(
        x,
        w1,
        w2,
        w1s_4d,
        w2s_4d,
        None,
        None,
        weights,
        ids,
        topk=topk,
        activation="silu",
        use_ep=False,
        use_sparse_core=False,
    )

    monkeypatch.setattr(fused_moe_gmm, "_GATHER_GMM", True)
    fused_moe_gmm.fused_moe_func.clear_cache()
    try:
        hlo = fused_moe_gmm.fused_moe_func.lower(
            x,
            w1,
            w2,
            w1s_4d,
            w2s_4d,
            None,
            None,
            weights,
            ids,
            topk=topk,
            activation="silu",
            use_ep=False,
            use_sparse_core=False,
        ).as_text()
        got = fused_moe_gmm.fused_moe_func(
            x,
            w1,
            w2,
            w1s_4d,
            w2s_4d,
            None,
            None,
            weights,
            ids,
            topk=topk,
            activation="silu",
            use_ep=False,
            use_sparse_core=False,
        )
    finally:
        fused_moe_gmm.fused_moe_func.clear_cache()

    assert "gather_fused_moe" in hlo
    assert got.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("skip_padded", [False, True])
def test_tpu_gather_gmm_masked_ep(monkeypatch, skip_padded):
    _require_tpu()
    tokens, local_experts, topk, hidden, inter = 1024, 16, 4, 1024, 256
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(12), 3)
    x = jax.random.normal(kx, (tokens, hidden)).astype(jnp.bfloat16)
    logits = jax.random.normal(kr, (tokens, 2 * local_experts))
    weights, global_ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    weights = weights.astype(jnp.float32)
    if skip_padded:
        weights = weights.at[-64:].set(0.0)
    experts_start = jnp.array(local_experts // 2, dtype=jnp.int32)
    w1, w1s, w2, w2s = _weights(kw, local_experts, hidden, inter)
    w1s_4d = jnp.expand_dims(w1s, 2)
    w2s_4d = jnp.expand_dims(w2s, 2)

    monkeypatch.setattr(fused_moe_gmm.envs, "TPU_MOE_TOPK_WEIGHTS_FLOAT32", True)
    monkeypatch.setattr(fused_moe_gmm, "_GATHER_GMM", False)
    fused_moe_gmm.fused_moe_func.clear_cache()
    want = fused_moe_gmm.fused_moe_func(
        x,
        w1,
        w2,
        w1s_4d,
        w2s_4d,
        None,
        None,
        weights,
        global_ids.astype(jnp.int32),
        experts_start=experts_start,
        topk=topk,
        activation="gelu_tanh",
        use_ep=True,
        use_sparse_core=False,
        skip_padded_tokens=skip_padded,
    )

    monkeypatch.setattr(fused_moe_gmm, "_GATHER_GMM", True)
    fused_moe_gmm.fused_moe_func.clear_cache()
    try:
        got = fused_moe_gmm.fused_moe_func(
            x,
            w1,
            w2,
            w1s_4d,
            w2s_4d,
            None,
            None,
            weights,
            global_ids.astype(jnp.int32),
            experts_start=experts_start,
            topk=topk,
            activation="gelu_tanh",
            use_ep=True,
            use_sparse_core=False,
            skip_padded_tokens=skip_padded,
        )
    finally:
        fused_moe_gmm.fused_moe_func.clear_cache()

    assert got.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(got, np.float32)).all()
    if skip_padded:
        np.testing.assert_array_equal(np.asarray(got[-64:]), 0)
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("masked", [False, True])
def test_tpu_gather_gmm_chunked_combine_over_128ki_routes(monkeypatch, masked):
    """Verify >128Ki routes execute chunked gather_combine on TPU."""
    _require_tpu()
    tokens, local_experts, topk, hidden, inter = 20480, 16, 8, 512, 128
    assert tokens * topk > 128 * 1024
    kx, kr, kw = jax.random.split(jax.random.PRNGKey(13), 3)
    x = jax.random.normal(kx, (tokens, hidden)).astype(jnp.bfloat16)
    total_experts = 2 * local_experts if masked else local_experts
    logits = jax.random.normal(kr, (tokens, total_experts))
    weights, global_ids = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), topk)
    weights = weights.astype(jnp.float32)
    experts_start = jnp.array(0, dtype=jnp.int32) if masked else None
    w1, w1s, w2, w2s = _weights(kw, local_experts, hidden, inter)
    w1s_4d = jnp.expand_dims(w1s, 2)
    w2s_4d = jnp.expand_dims(w2s, 2)

    monkeypatch.setattr(fused_moe_gmm.envs, "TPU_MOE_TOPK_WEIGHTS_FLOAT32", True)
    monkeypatch.setattr(fused_moe_gmm, "_GATHER_GMM", False)
    fused_moe_gmm.fused_moe_func.clear_cache()
    want = fused_moe_gmm.fused_moe_func(
        x,
        w1,
        w2,
        w1s_4d,
        w2s_4d,
        None,
        None,
        weights,
        global_ids.astype(jnp.int32),
        experts_start=experts_start,
        topk=topk,
        activation="silu",
        use_ep=masked,
        use_sparse_core=False,
    )

    monkeypatch.setattr(fused_moe_gmm, "_GATHER_GMM", True)
    fused_moe_gmm.fused_moe_func.clear_cache()
    try:
        hlo = fused_moe_gmm.fused_moe_func.lower(
            x,
            w1,
            w2,
            w1s_4d,
            w2s_4d,
            None,
            None,
            weights,
            global_ids.astype(jnp.int32),
            experts_start=experts_start,
            topk=topk,
            activation="silu",
            use_ep=masked,
            use_sparse_core=False,
        ).as_text()
        got = fused_moe_gmm.fused_moe_func(
            x,
            w1,
            w2,
            w1s_4d,
            w2s_4d,
            None,
            None,
            weights,
            global_ids.astype(jnp.int32),
            experts_start=experts_start,
            topk=topk,
            activation="silu",
            use_ep=masked,
            use_sparse_core=False,
        )
    finally:
        fused_moe_gmm.fused_moe_func.clear_cache()

    assert "gather_fused_moe" in hlo
    assert got.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(want, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
