# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.mhc import shifted_post_pre_rms as mhc

M, D = 4, 5120


def _inputs(tokens: int, seed: int = 0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 8)
    x = (jax.random.normal(ks[0], (tokens, D)) * 0.5).astype(jnp.bfloat16)
    flat = (jax.random.normal(ks[1], (tokens, M * D)) * 0.5).astype(jnp.bfloat16)
    post = jax.nn.sigmoid(jax.random.normal(ks[2], (tokens, M, 1))) * 2.0
    comb = jax.nn.softmax(jax.random.normal(ks[3], (tokens, M, M)), axis=-1)
    pre = jax.nn.sigmoid(jax.random.normal(ks[4], (tokens, M))) + 1e-6
    weight = mhc.pack_mix_weight(jax.random.normal(ks[5], (M * (M + 2), M * D)) * 0.01)
    scale = jnp.array([0.1, 0.2, 0.3], jnp.float32)
    base = jax.random.normal(ks[6], (M * (M + 2),)) * 0.1
    norm = (1.0 + 0.1 * jax.random.normal(ks[7], (D,))).astype(jnp.bfloat16)
    return x, flat, post, comb, pre, weight, scale, base, norm


def _layers(layout: dict, x, flat, post, comb, pre, weight, scale, base, norm):
    """Three chained layers, as the decoder passes residual and gates on."""
    outs = []
    for _ in range(3):
        flat, post, comb, x, pre = mhc.shifted_post_pre_rms(
            x, flat, post, comb, weight, scale, base, pre, norm_weight=norm, **layout
        )
        outs += [flat, post, comb, x, pre]
    return outs


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
@pytest.mark.parametrize("tokens", [96, 288, 2048])
def test_resident_gates_are_bitwise_identical(tokens):
    args = _inputs(tokens)
    resident = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.COMPILER,
    }
    ref = jax.jit(lambda *a: _layers({}, *a))(*args)
    got = jax.jit(lambda *a: _layers(resident, *a))(*args)
    for r, g in zip(ref, got, strict=True):
        assert r.shape == g.shape and r.dtype == g.dtype
        np.testing.assert_array_equal(
            np.ascontiguousarray(r).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
@pytest.mark.parametrize("tokens", [96, 288, 2048])
def test_early_prep_is_bitwise_identical(tokens):
    args = _inputs(tokens)
    resident = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.COMPILER,
    }
    ref = jax.jit(lambda *a: _layers(resident, *a))(*args)
    got = jax.jit(lambda *a: _layers({**resident, "early_prep": True}, *a))(*args)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(
            np.ascontiguousarray(r).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
def test_t240_matches_unfused_fallback_within_rounding(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_mhc_op as op,
    )

    x, flat, post, comb, pre, weight, scale, base, norm = _inputs(240)
    consts = (1e-20, 1e-6, 1e-6, 2.0, 20, 1e-6)

    def layer(*args):
        return op._post_pre_rms_jax(*args, *consts)

    args = (x, flat, post, comb, weight, scale, base, pre, norm)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_MHC_SHIFTED_T16", "0")
    ref = jax.jit(layer)(*args)
    monkeypatch.setenv("TPU_DEEPSEEK_V41_MHC_SHIFTED_T16", "1")
    got = jax.jit(layer)(*args)
    # residual, value: one BF16 rounding step; gates: FP32 reassociation.
    for r, g, rtol in zip(ref, got, (2.0**-7, 1e-4, 1e-4, 2.0**-6, 1e-4)):
        r, g = np.asarray(r, np.float32), np.asarray(g, np.float32)
        bound = np.maximum(np.abs(r), np.abs(g)) * rtol + 1e-6
        assert np.all(np.abs(r - g) <= bound)


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
def test_t40_pads_to_a_kernel_tile_and_matches_fallback(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_mhc_op as op,
    )

    x, flat, post, comb, pre, weight, scale, base, norm = _inputs(48)
    consts = (1e-20, 1e-6, 1e-6, 2.0, 20, 1e-6)

    def layer(*args):
        return op._post_pre_rms_jax(*args, *consts)

    def rows(n):
        return (x[:n], flat[:n], post[:n], comb[:n], weight, scale, base, pre[:n], norm)

    monkeypatch.setenv("TPU_DEEPSEEK_V41_MHC_SHIFTED_T16", "1")
    full = jax.jit(layer)(*rows(48))
    got = jax.jit(layer)(*rows(40))
    # Rows are independent: the 40 live rows equal the 48-row kernel's.
    for f, g in zip(full, got, strict=True):
        assert g.shape[0] == 40
        np.testing.assert_array_equal(
            np.ascontiguousarray(f[:40]).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )
    monkeypatch.setenv("TPU_DEEPSEEK_V41_MHC_SHIFTED_T16", "0")
    ref = jax.jit(layer)(*rows(40))
    for r, g, rtol in zip(ref, got, (2.0**-7, 1e-4, 1e-4, 2.0**-6, 1e-4)):
        r, g = np.asarray(r, np.float32), np.asarray(g, np.float32)
        bound = np.maximum(np.abs(r), np.abs(g)) * rtol + 1e-6
        assert np.all(np.abs(r - g) <= bound)


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
@pytest.mark.parametrize("tokens", [16, 48, 96, 240, 288, 2048])
def test_fused_gates_are_bitwise_identical(tokens):
    args = _inputs(tokens)
    resident = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.COMPILER,
        "early_prep": True,
    }
    fused = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.HBM,
        "early_prep": True,
        "fused_gates": True,
    }
    ref = jax.jit(lambda *a: _layers(resident, *a))(*args)
    got = jax.jit(lambda *a: _layers(fused, *a))(*args)
    for r, g in zip(ref, got, strict=True):
        assert r.shape == g.shape and r.dtype == g.dtype
        np.testing.assert_array_equal(
            np.ascontiguousarray(r).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
def test_one_gate_block_matches_the_wide_form_bitwise():
    # Gate columns are independent: a 128-column block (one vreg per comb
    # entry) must equal the same columns of a 256-column [M,M,N] block.
    ks = jax.random.split(jax.random.PRNGKey(3), 3)
    mix = jax.random.normal(ks[0], (M * (M + 2), 128)) * 30.0
    square = jnp.abs(jax.random.normal(ks[1], (1, 128))) * (M * D) + 1.0
    scale = jnp.array([0.1, 0.2, 0.3], jnp.float32)
    base = jax.random.normal(ks[2], (M * (M + 2),)) * 0.5
    narrow = jax.jit(lambda *a: mhc._pre_gates(*a, hidden_size=D))(
        mix, square, scale, base
    )
    wide = jax.jit(lambda *a: mhc._pre_gates(*a, hidden_size=D))(
        jnp.tile(mix, (1, 2)), jnp.tile(square, (1, 2)), scale, base
    )
    for n, w in zip(narrow, wide, strict=True):
        np.testing.assert_array_equal(
            np.asarray(n).view(np.uint32), np.asarray(w)[:128].view(np.uint32)
        )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
@pytest.mark.parametrize("tokens", [256, 384])
def test_multi_tile_rows_match_single_tiles(tokens):
    # Rows are independent: each 128-token tile of a multi-tile call (even
    # and odd tile counts) must equal that 128-row block run as its own call.
    x, flat, post, comb, pre, weight, scale, base, norm = _inputs(tokens)
    layout = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.HBM,
        "early_prep": True,
        "fused_gates": True,
    }

    def seam(x, flat, post, comb, pre):
        return mhc.shifted_post_pre_rms(
            x, flat, post, comb, weight, scale, base, pre, norm_weight=norm, **layout
        )

    run = jax.jit(seam)
    full = run(x, flat, post, comb, pre)
    for start in range(0, tokens, 128):
        rows = slice(start, start + 128)
        block = run(x[rows], flat[rows], post[rows], comb[rows], pre[rows])
        for f, b in zip(full, block, strict=True):
            np.testing.assert_array_equal(
                np.ascontiguousarray(f[rows]).view(np.uint8),
                np.ascontiguousarray(b).view(np.uint8),
            )


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU7x mHC kernel")
def test_chunked_weight_projection_is_bitwise_one_dot(monkeypatch):
    # The 16-token seam projects per MXU K step as its weight chunks land;
    # summed in order, the steps must give exactly the one-dot projection.
    args = _inputs(16)
    layout = {
        "unpadded_gates": True,
        "gate_input_mode": mhc.GateInputMode.HBM,
        "early_prep": True,
        "fused_gates": True,
    }
    assert mhc._WEIGHT_CHUNKS > 1
    got = jax.jit(lambda *a: _layers(layout, *a))(*args)
    monkeypatch.setattr(mhc, "_WEIGHT_CHUNKS", 1)
    ref = jax.jit(lambda *a: _layers(layout, *a))(*args)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(
            np.ascontiguousarray(r).view(np.uint8),
            np.ascontiguousarray(g).view(np.uint8),
        )
