# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""FUSED_MOE_PREFILL_KERNEL_V2 (stage A, per-row activation scales), one device:
against its own default build, the same op bit for bit -- weight ring depth,
explicit-MXU matmuls with the per-N-tile down epilogue, the inline row gather,
256-row tiles and the flat combine pointer all restructure the issue, not the
arithmetic -- and close to the default fused MoE prefill kernel, which scales
activations per 512 columns instead of per row."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import fused_moe_prefill
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as v2,
)

# GLM-5.2's per-rank MoE slice at TP16 (i = i_s = 128, top-8, routed scale
# 2.5, ungated shared expert) with fewer experts and a smaller hidden size.
H, INTER, INTER_S, E, TOPK = 1024, 128, 128, 64, 8
ROUTED_SCALE = 2.5
F8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module")
def setup():
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    tpu = jax.devices()[0]
    if "7" not in tpu.device_kind:
        pytest.skip("the explicit-MXU path needs MXU accumulators (v7x)")
    ks = jax.random.split(jax.random.key(5), 4)

    def q(w, axis):
        s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
        return (w / s).astype(F8), s

    w1, w1s = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
    w2, w2s = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
    sw1, sw1s = q(jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0)
    sw2, sw2s = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
    weights = (
        w1,
        w2,
        w1s[:, :, None, :],
        w2s[:, :, None, :],
        sw1,
        sw1s.reshape(-1),
        sw2,
        sw2s.reshape(-1),
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("d",))
    return weights, mesh


def _routing(m, seed, hot):
    rng = np.random.default_rng(seed)
    g = rng.random((m, E))
    if hot:  # one expert in every token: multi-tile visits and tails
        g[:, 7] = 2.0
    ids = np.argsort(-g, axis=1)[:, :TOPK].astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    tw[-3:] = 0.0  # padded rows
    return jnp.asarray(tw, jnp.bfloat16), jnp.asarray(ids)


def _run(mesh, fn, *args):
    return jax.jit(
        jax.shard_map(
            fn, mesh=mesh, in_specs=(P(),) * len(args), out_specs=P(), check_vma=False
        )
    )(*args)


@pytest.mark.parametrize("m,hot", [(256, False), (1024, True), (4096, True)])
def test_v2_matches_its_default_build_bit_for_bit(setup, m, hot):
    weights, mesh = setup
    x = jax.random.normal(jax.random.key(m), (m, H)).astype(jnp.bfloat16)
    tw, ti = _routing(m, m, hot)

    def body(op, **kw):
        def f(x, tw, ti, *w):
            w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
            return op(
                x,
                w1,
                w2,
                w1s,
                w2s,
                sw1,
                sw1s,
                sw2,
                sw2s,
                None,
                tw,
                ti,
                axis_name="d",
                routed_scale=ROUTED_SCALE,
                **kw,
            )

        return f

    ref = _run(mesh, body(v2.fused_moe_prefill), x, tw, ti, *weights)
    got = _run(mesh, body(v2.fused_moe_prefill, **v2.STAGE_A), x, tw, ti, *weights)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # Per-row against per-512-column activation scales: close, not equal
    # (v7x, j-a45913f7: relative L2 0.0525-0.0531 at 256 / 1024 / 4096 tokens).
    blk = np.asarray(_run(mesh, body(fused_moe_prefill), x, tw, ti, *weights))
    got = np.asarray(got).astype(np.float32)
    blk = blk.astype(np.float32)
    rel = np.linalg.norm(got - blk) / np.linalg.norm(blk)
    assert rel < 7e-2, rel


@pytest.mark.parametrize("m,hot", [(256, False), (1024, True)])
def test_sum_overlap_adds_match_the_serial_sum_bit_for_bit(setup, m, hot):
    """TPU_MOE_PREFILL_SUM_IN_KERNEL_OVERLAP_ADDS moves each tile's row adds into
    the next tile's gate/up K-tiles; every token still gets the same rows added in
    the same order."""
    weights, mesh = setup
    assert v2.sum_columns(m, H, INTER, INTER_S) == H
    x = jax.random.normal(jax.random.key(m + 2), (m, H)).astype(jnp.bfloat16)
    tw, ti = _routing(m, m + 2, hot)

    def body(**kw):
        def f(x, tw, ti, *w):
            w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
            return v2.fused_moe_prefill(
                x,
                w1,
                w2,
                w1s,
                w2s,
                sw1,
                sw1s,
                sw2,
                sw2s,
                None,
                tw,
                ti,
                axis_name="d",
                routed_scale=ROUTED_SCALE,
                sum_in_kernel_max_tokens=m,
                **v2.STAGE_A,
                **kw,
            )

        return f

    serial = _run(mesh, body(), x, tw, ti, *weights)
    overlapped = _run(mesh, body(sum_overlap_adds=True), x, tw, ti, *weights)
    np.testing.assert_array_equal(np.asarray(overlapped), np.asarray(serial))


@pytest.mark.parametrize("m,hot", [(1024, False), (2048, True), (4096, True)])
def test_resident_tokens_match_token_windows_bit_for_bit(setup, m, hot):
    """TPU_MOE_PREFILL_ROW_TOKENS_RESIDENT_MAX_TOKENS: the stage-A build that reads each
    route's token from the token table resident in SMEM returns exactly what the
    build that copies a window of the table before every tile returns."""
    weights, mesh = setup
    x = jax.random.normal(jax.random.key(m + 3), (m, H)).astype(jnp.bfloat16)
    tw, ti = _routing(m, m + 3, hot)

    def body(**kw):
        def f(x, tw, ti, *w):
            w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
            return v2.fused_moe_prefill(
                x,
                w1,
                w2,
                w1s,
                w2s,
                sw1,
                sw1s,
                sw2,
                sw2s,
                None,
                tw,
                ti,
                axis_name="d",
                routed_scale=ROUTED_SCALE,
                **v2.STAGE_A,
                **kw,
            )

        return f

    ref = _run(mesh, body(), x, tw, ti, *weights)
    got = _run(mesh, body(row_tokens_resident_max_tokens=m), x, tw, ti, *weights)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("m,hot", [(1024, True), (2048, True), (4096, True)])
def test_resident_tokens_beside_the_in_kernel_sum_bit_for_bit(setup, m, hot):
    """The in-kernel sum up to 1024 tokens with overlapped adds and 128-row tiles up
    to 4096 tokens, with and without the resident token table up to 4096 tokens: the
    one-pass build already reads every token from its resident table and is the same
    build, and the slab builds above it read the resident table; each output is the
    one without the resident switch, bit for bit."""
    weights, mesh = setup
    x = jax.random.normal(jax.random.key(m + 4), (m, H)).astype(jnp.bfloat16)
    tw, ti = _routing(m, m + 4, hot)

    def body(**kw):
        def f(x, tw, ti, *w):
            w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
            return v2.fused_moe_prefill(
                x,
                w1,
                w2,
                w1s,
                w2s,
                sw1,
                sw1s,
                sw2,
                sw2s,
                None,
                tw,
                ti,
                axis_name="d",
                routed_scale=ROUTED_SCALE,
                **dict(v2.STAGE_A, tile_min=8192),
                sum_in_kernel_max_tokens=1024,
                sum_overlap_adds=True,
                **kw,
            )

        return f

    ref = _run(mesh, body(), x, tw, ti, *weights)
    got = _run(mesh, body(row_tokens_resident_max_tokens=4096), x, tw, ti, *weights)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("m", [512, 4096])
def test_norm_quant_moe_feeds_stage_a_bit_for_bit(setup, m):
    """TPU_FUSED_ADD_RMSNORM_QUANT_MOE: the norm's (x_q, x_scale) are the
    stage-A build's own `_quantize_rows` of the norm output, and the stage-A
    build fed them returns exactly what it returns quantizing y itself."""
    from vllm_torchtpu.kernels import fused_add_rmsnorm as fnorm

    weights, mesh = setup
    x0 = jax.random.normal(jax.random.key(1), (m, H)).astype(jnp.bfloat16)
    r0 = (jax.random.normal(jax.random.key(2), (m, H)) * 0.5).astype(jnp.bfloat16)
    nw = (1.0 + 0.05 * jax.random.normal(jax.random.key(3), (H,))).astype(jnp.bfloat16)
    y_ref, res_ref = fnorm.fused_add_rms_norm(x0, r0, nw, eps=1e-5)
    y, x_q, x_scale, res = fnorm.fused_add_rms_norm_quant_moe(x0, r0, nw, eps=1e-5)
    q_ref, s_ref = jax.jit(lambda y: v2._quantize_rows(y, F8))(y_ref)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(y_ref))
    np.testing.assert_array_equal(np.asarray(res), np.asarray(res_ref))
    np.testing.assert_array_equal(
        np.asarray(x_q).view(np.uint8), np.asarray(q_ref).view(np.uint8)
    )
    np.testing.assert_array_equal(np.asarray(x_scale), np.asarray(s_ref))

    tw, ti = _routing(m, 9, True)

    def stage_a(y, tw, ti, *w, **kw):
        w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
        return v2.fused_moe_prefill_stage_a(
            y,
            w1,
            w2,
            w1s,
            w2s,
            sw1,
            sw1s,
            sw2,
            sw2s,
            None,
            tw,
            ti,
            axis_name="d",
            routed_scale=ROUTED_SCALE,
            **kw,
        )

    def fed(y, x_q, x_scale, tw, ti, *w):
        return stage_a(y, tw, ti, *w, x_q=x_q, x_scale=x_scale)

    np.testing.assert_array_equal(
        np.asarray(_run(mesh, fed, y, x_q, x_scale, tw, ti, *weights)),
        np.asarray(_run(mesh, stage_a, y_ref, tw, ti, *weights)),
    )


@pytest.fixture(scope="module")
def setup_tp8():
    """The 2x2x1 twin's TP8 slice: per-rank intermediate 256 (gate|up 512 columns),
    shared 256."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    inter = inter_s = 256
    ks = jax.random.split(jax.random.key(8), 4)

    def q(w, axis):
        s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
        return (w / s).astype(F8), s

    w1, w1s = q(jax.random.normal(ks[0], (E, H, 2 * inter)) * 0.02, 1)
    w2, w2s = q(jax.random.normal(ks[1], (E, inter, H)) * 0.02, 1)
    sw1, sw1s = q(jax.random.normal(ks[2], (H, 2 * inter_s)) * 0.02, 0)
    sw2, sw2s = q(jax.random.normal(ks[3], (inter_s, H)) * 0.02, 0)
    weights = (
        w1,
        w2,
        w1s[:, :, None, :],
        w2s[:, :, None, :],
        sw1,
        sw1s.reshape(-1),
        sw2,
        sw2s.reshape(-1),
    )
    return weights, Mesh(np.asarray(jax.devices()[:1]), ("d",))


@pytest.mark.parametrize("m", [256, 4096])
def test_v2_unsupported_shape_falls_back_bit_exact(setup_tp8, m):
    """FUSED_MOE_PREFILL_KERNEL_V2 on a shape the explicit-MXU body does not handle
    (TP8: i = 256) runs the per-row default build instead of asserting, bit for
    bit; fed precomputed _quantize_rows output too."""
    weights, mesh = setup_tp8
    assert not v2.xmxu_supported(H, 256, 256)
    x = jax.random.normal(jax.random.key(m + 1), (m, H)).astype(jnp.bfloat16)
    tw, ti = _routing(m, m + 1, True)

    def body(op, **kw):
        def f(x, tw, ti, *w):
            w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
            return op(
                x,
                w1,
                w2,
                w1s,
                w2s,
                sw1,
                sw1s,
                sw2,
                sw2s,
                None,
                tw,
                ti,
                axis_name="d",
                routed_scale=ROUTED_SCALE,
                **kw,
            )

        return f

    ref = _run(mesh, body(v2.fused_moe_prefill), x, tw, ti, *weights)
    got = _run(mesh, body(v2.fused_moe_prefill, **v2.STAGE_A), x, tw, ti, *weights)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    x_q, x_scale = jax.jit(lambda y: v2._quantize_rows(y, F8))(x)
    fed = _run(
        mesh,
        body(v2.fused_moe_prefill, x_q=x_q, x_scale=x_scale, **v2.STAGE_A),
        x,
        tw,
        ti,
        *weights,
    )
    np.testing.assert_array_equal(np.asarray(fed), np.asarray(ref))
