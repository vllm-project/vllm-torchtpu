# SPDX-License-Identifier: Apache-2.0
"""Prefill HC at served step sizes (6,400 / 3,200 tokens): odd tile counts in
hc_full, and the fused mix that keeps ``norm`` on chip (interpret mode)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import (
    extend_down,
    hc_prefill_mix,
    prefill_token_tile_ok,
)
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 128, 64, 1e-6, 128
INTERPRET = pltpu.InterpretParams()


def _arrays(tokens, seed=41):
    rng = np.random.default_rng(seed)
    width = BRANCHES * HIDDEN
    shapes = ((tokens, width), (width,), (LOWRANK, width), (width, LOWRANK), (BRANCHES, width))
    scales = (1.0, 0.3, 0.05, 0.1, 0.05)
    return tuple(
        jnp.asarray(rng.normal(size=s) * c, jnp.bfloat16) for s, c in zip(shapes, scales)
    )


def _bits(a):
    return np.asarray(a).view(np.uint16 if a.dtype == jnp.bfloat16 else np.uint32)


def _reference(x, weight, down, up, inject):
    """The prefill path's staged BF16 contract (hc_norm/hc_full) in plain JAX,
    plus the XLA injection gate the served path computes from ``norm``."""
    values = x.astype(jnp.float32).reshape(-1, BRANCHES, HIDDEN)
    norm = values * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + EPS)
    norm = norm.astype(jnp.bfloat16).reshape(x.shape).astype(jnp.float32)
    scaled = (norm * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    norm = (norm + scaled.astype(jnp.float32)).astype(jnp.bfloat16)
    mid = (norm.astype(jnp.float32) @ down.astype(jnp.float32).T).astype(jnp.bfloat16)
    mid = jax.nn.silu((mid / BRANCHES).astype(jnp.bfloat16).astype(jnp.float32))
    mid = mid.astype(jnp.bfloat16)
    logits = (mid.astype(jnp.float32) @ up.astype(jnp.float32).T).astype(jnp.bfloat16)
    weighted = jax.nn.sigmoid(logits.astype(jnp.float32)) * norm.astype(jnp.float32)
    mixed = (weighted.reshape(-1, BRANCHES, HIDDEN).sum(1) / BRANCHES).astype(jnp.bfloat16)
    inj = (norm.astype(jnp.float32) @ inject.astype(jnp.float32).T).astype(jnp.bfloat16)
    injection = 2.0 * jax.nn.sigmoid(inj.astype(jnp.float32) / BRANCHES)
    return mixed, norm, injection


def _close_bf16(actual, expected, name):
    a = np.asarray(actual, np.float32)
    e = np.asarray(expected, np.float32)
    assert np.isfinite(a).all(), name
    assert np.linalg.norm(a - e) <= 1e-2 * np.linalg.norm(e), name
    assert (a != e).mean() <= 0.02, (name, (a != e).mean())


full = functools.partial(
    _full_mix, branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, interpret=INTERPRET
)
fused = functools.partial(
    hc_prefill_mix, branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, interpret=INTERPRET
)


@pytest.mark.parametrize("odd_tiles", [3, 5])
def test_full_mix_odd_tile_count_matches_even_rows(odd_tiles):
    """An odd tile count (6,400 = 25 x 256) runs its last tile on parity 0:
    each row must equal the same row of an even-count call (rows are
    independent), and the contract reference."""
    x, weight, down, up, inject = _arrays((odd_tiles + 1) * TILE)
    rows = odd_tiles * TILE
    mixed_e, norm_e = full(x, weight, down, up)
    mixed_o, norm_o = full(x[:rows], weight, down, up)
    np.testing.assert_array_equal(_bits(mixed_o), _bits(mixed_e[:rows]))
    np.testing.assert_array_equal(_bits(norm_o), _bits(norm_e[:rows]))
    ref_mixed, ref_norm, _ = _reference(x[:rows], weight, down, up, inject)
    _close_bf16(mixed_o, ref_mixed, "mixed")
    _close_bf16(norm_o, ref_norm, "norm")


def test_full_mix_rejects_single_and_partial_tiles():
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN
    for tokens in (TILE, 3 * TILE + 64):
        args = (
            shape((tokens, width), jnp.bfloat16),
            shape((width,), jnp.bfloat16),
            shape((LOWRANK, width), jnp.bfloat16),
            shape((width, LOWRANK), jnp.bfloat16),
        )
        with pytest.raises(ValueError, match="token tiles"):
            jax.eval_shape(full, *args)


def test_extend_down_pads_to_128_rows():
    _, _, down, _, inject = _arrays(TILE)
    ext = extend_down(down, inject)
    assert ext.shape == (128, BRANCHES * HIDDEN)
    np.testing.assert_array_equal(_bits(ext[:LOWRANK]), _bits(down))
    np.testing.assert_array_equal(_bits(ext[LOWRANK : LOWRANK + BRANCHES]), _bits(inject))
    assert not np.asarray(ext[LOWRANK + BRANCHES :], np.float32).any()


@pytest.mark.parametrize("tiles", [2, 3])
def test_fused_mix_matches_full_mix_and_injection(tiles):
    """``mixed`` equals hc_full's bit for bit (same staged math, norm never
    leaves VMEM); the injection gate equals the served XLA gate computed from
    hc_full's ``norm`` up to the logits' summation order."""
    x, weight, down, up, inject = _arrays(tiles * TILE, seed=7 + tiles)
    mixed, injection = fused(x, weight, down, up, inject)
    if tiles >= 2:
        mixed_full, norm_full = full(x, weight, down, up)
        np.testing.assert_array_equal(_bits(mixed), _bits(mixed_full))
        logits = (norm_full.astype(jnp.float32) @ inject.astype(jnp.float32).T).astype(
            jnp.bfloat16
        )
        expected = 2.0 * jax.nn.sigmoid(logits.astype(jnp.float32) / BRANCHES)
        np.testing.assert_allclose(np.asarray(injection), np.asarray(expected), rtol=1e-2, atol=1e-6)
    assert injection.dtype == jnp.float32 and injection.shape == (tiles * TILE, BRANCHES)
    ref_mixed, _, ref_inj = _reference(x, weight, down, up, inject)
    _close_bf16(mixed, ref_mixed, "mixed")
    np.testing.assert_allclose(np.asarray(injection), np.asarray(ref_inj), rtol=1e-2, atol=1e-6)


def test_fused_combine_mix_matches_gated_residual_then_mix():
    """Prologue: h is the gated_residual combine bit for bit; mixed/injection
    are the plain fused mix of h."""
    tokens = 3 * TILE
    prev, weight, down, up, inject = _arrays(tokens, seed=3)
    rng = np.random.default_rng(5)
    block = jnp.asarray(rng.normal(size=(tokens, HIDDEN)), jnp.bfloat16)
    gates = jnp.asarray(rng.random((tokens, BRANCHES)) * 2, jnp.float32)
    gates = gates.at[0].set(jnp.arange(BRANCHES, dtype=jnp.float32) - 1)
    mixed, injection, h = fused(prev, weight, down, up, inject, block=block, prev_injection=gates)
    expected_h = (
        (
            prev.astype(jnp.float32).reshape(tokens, BRANCHES, HIDDEN)
            + block.astype(jnp.float32)[:, None, :] * gates[:, :, None]
        )
        .astype(jnp.bfloat16)
        .reshape(tokens, -1)
    )
    # Bitwise on TPU against gated_residual (same Mosaic expression; the bench
    # checks it). On CPU, XLA's FMA contraction of the reference may differ:
    # allow one bf16 ulp on at most 0.01% of elements.
    ulps = np.abs(_bits(h).astype(np.int32) - _bits(expected_h).astype(np.int32))
    assert ulps.max() <= 1 and (ulps > 0).mean() <= 1e-4, (ulps.max(), (ulps > 0).mean())
    mixed_h, injection_h = fused(h, weight, down, up, inject)
    np.testing.assert_array_equal(_bits(mixed), _bits(mixed_h))
    np.testing.assert_array_equal(_bits(injection), _bits(injection_h))


def test_fused_mix_rejects_partial_tiles():
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN
    args = (
        shape((TILE + 64, width), jnp.bfloat16),
        shape((width,), jnp.bfloat16),
        shape((LOWRANK, width), jnp.bfloat16),
        shape((width, LOWRANK), jnp.bfloat16),
        shape((BRANCHES, width), jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="token tiles"):
        jax.eval_shape(fused, *args)
    assert prefill_token_tile_ok(6400, 256) and prefill_token_tile_ok(3200, 128)
    assert not prefill_token_tile_ok(3200, 256) and not prefill_token_tile_ok(64, 128)


# ---- partial last tile (TPU_QWEN4_HC_PREFILL_PARTIAL_TILE): 3,200 = 12 x 256 + 128


@pytest.mark.parametrize("rows", [2 * TILE + 32, 3 * TILE + 64, 3 * TILE + 96])
def test_full_mix_partial_last_tile_matches_whole_tile_rows(rows):
    """The partial step copies, computes and drains only its rows: every row
    equals the same row of a whole-tile call."""
    x, weight, down, up, _ = _arrays(4 * TILE, seed=21)
    mixed_w, norm_w = full(x, weight, down, up)
    mixed_p, norm_p = full(x[:rows], weight, down, up, partial=True)
    np.testing.assert_array_equal(_bits(mixed_p), _bits(mixed_w[:rows]))
    np.testing.assert_array_equal(_bits(norm_p), _bits(norm_w[:rows]))


def test_full_mix_partial_needs_flag_and_32_row_multiple():
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN

    def args(tokens):
        return (shape((tokens, width), jnp.bfloat16), shape((width,), jnp.bfloat16),
                shape((LOWRANK, width), jnp.bfloat16), shape((width, LOWRANK), jnp.bfloat16))

    with pytest.raises(ValueError, match="token tiles"):
        jax.eval_shape(full, *args(3 * TILE + 64))  # no partial flag
    with pytest.raises(ValueError, match="token tiles"):
        jax.eval_shape(functools.partial(full, partial=True), *args(3 * TILE + 48))  # not % 32
    jax.eval_shape(functools.partial(full, partial=True), *args(3 * TILE + 64))


@pytest.mark.parametrize("rows", [TILE + 64, 3 * TILE + 64])
def test_fused_mix_partial_last_tile_matches_whole_tile_rows(rows):
    x, weight, down, up, inject = _arrays(4 * TILE, seed=23)
    mixed_w, inj_w = fused(x, weight, down, up, inject)
    mixed_p, inj_p = fused(x[:rows], weight, down, up, inject, partial=True)
    np.testing.assert_array_equal(_bits(mixed_p), _bits(mixed_w[:rows]))
    np.testing.assert_array_equal(_bits(inj_p), _bits(inj_w[:rows]))
    rng = np.random.default_rng(29)
    block = jnp.asarray(rng.normal(size=(4 * TILE, HIDDEN)), jnp.bfloat16)
    gates = jnp.asarray(rng.random((4 * TILE, BRANCHES)) * 2, jnp.float32)
    w = fused(x, weight, down, up, inject, block=block, prev_injection=gates)
    p = fused(x[:rows], weight, down, up, inject, block=block[:rows], prev_injection=gates[:rows], partial=True)
    for a, b in zip(p, w):
        np.testing.assert_array_equal(_bits(a), _bits(b[:rows]))
    with pytest.raises(ValueError, match="token tiles"):
        jax.eval_shape(fused, x[:rows], weight, down, up, inject)  # no partial flag


@pytest.mark.parametrize("rows", [2 * TILE + 32, 3 * TILE + 64])
def test_gated_residual_partial_last_tile(rows):
    rng = np.random.default_rng(31)
    x = jnp.asarray(rng.normal(size=(4 * TILE, BRANCHES * HIDDEN)), jnp.bfloat16)
    block = jnp.asarray(rng.normal(size=(4 * TILE, HIDDEN)), jnp.bfloat16)
    gates = jnp.asarray(rng.random((4 * TILE, BRANCHES)) * 2, jnp.float32)
    whole = gated_residual_fn(BRANCHES, HIDDEN, TILE, False, INTERPRET)(x, block, gates)
    part = gated_residual_fn(BRANCHES, HIDDEN, TILE, True, INTERPRET)(x[:rows], block[:rows], gates[:rows])
    np.testing.assert_array_equal(_bits(part), _bits(whole[:rows]))
    with pytest.raises(ValueError, match="full BF16 tiles"):
        gated_residual_fn(BRANCHES, HIDDEN, TILE, False, INTERPRET)(x[:rows], block[:rows], gates[:rows])


# ---- GatedResidual plumbing (the decoder layer's mix -> combine_mix -> combine)


def _module(seed, tile=0):
    torch.manual_seed(seed)
    module = GatedResidual(
        HyperConnectionConfig(
            hc_count=BRANCHES,
            hidden_size=HIDDEN,
            params_dtype=torch.bfloat16,
            hc_lowrank=LOWRANK,
            rms_norm_eps=EPS,
            hc_per_branch_norm=True,
        )
    )
    with torch.no_grad():
        module.hc_norm.weight.normal_(0.0, 0.3)
        module.input_mix_weight_down.weight.normal_(0.0, 0.05)
        module.input_mix_weight_up.weight.normal_(0.0, 0.1)
        module.block_inject_weight.weight.normal_(0.0, 0.05)
    return module


def _to_jax(t):
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy(), jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _to_torch(a):
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def _install_fused(module, calls, odd_tiles, partial=False):
    """Interpret-mode stand-ins for the registered prefill ops (the torch op
    boundary needs torch_tpu); gated_residual's combine is its torch formula."""
    module.prefill_full_tile = TILE
    module._prefill_tile_step = TILE if odd_tiles else 2 * TILE
    module._prefill_partial = partial

    def inj_mix(x, *weights):
        calls.append("mix")
        return tuple(map(_to_torch, fused(*map(_to_jax, (x, *weights)), partial=partial)))

    def combine_mix(residual, block, gate, *weights):
        calls.append("combine_mix")
        outs = fused(
            *map(_to_jax, (residual, *weights)),
            block=_to_jax(block),
            prev_injection=_to_jax(gate),
            partial=partial,
        )
        return tuple(map(_to_torch, outs))

    def combine(residual, block, gates):
        calls.append("combine")
        r = residual.float().unflatten(-1, (BRANCHES, HIDDEN))
        return (r + block.float()[:, None, :] * gates[:, :, None]).flatten(1).bfloat16()

    module._prefill_inj_mix_op = inj_mix
    module._prefill_combine_mix_op = combine_mix
    module._prefill_combine_op = combine
    return module


def _install_full(module, calls, partial=False):
    """The TPU_QWEN4_HC_PREFILL_ODD_TILES path: hc_full (interpret mode), the
    XLA-style injection from ``norm`` (torch), the gated_residual formula."""
    module.prefill_full_tile = TILE
    module._prefill_tile_step = TILE
    module._prefill_partial = partial

    def full_mix(x, *weights):
        calls.append("full")
        return tuple(map(_to_torch, full(*map(_to_jax, (x, *weights)), partial=partial)))

    def combine(residual, block, gates):
        calls.append("combine")
        r = residual.float().unflatten(-1, (BRANCHES, HIDDEN))
        return (r + block.float()[:, None, :] * gates[:, :, None]).flatten(1).bfloat16()

    module._prefill_full_op = full_mix
    module._prefill_combine_op = combine
    return module


@pytest.mark.parametrize("tokens", [3 * TILE, 4 * TILE, 3 * TILE + 64])
def test_layer_plumbing_fused_prefill_matches_odd_tile_path(tokens):
    """attn mix -> combine_mix -> mlp combine, as Qwen4ExpDecoderLayer runs,
    with the fused prefill ops against the hc_full path at the same step; a
    partial last tile (3 x 128 + 64) takes both paths only under the flag."""
    partial = bool(tokens % TILE)
    calls, ref_calls = [], []
    attn = _install_full(_module(11), ref_calls, partial)
    mlp = _install_full(_module(12), ref_calls, partial)
    f_attn = _install_fused(_module(11), calls, odd_tiles=True, partial=partial)
    f_mlp = _install_fused(_module(12), calls, odd_tiles=True, partial=partial)
    torch.manual_seed(13)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    attn_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    mlp_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    outs = []
    with torch.no_grad():
        for a, m in ((attn, mlp), (f_attn, f_mlp)):
            block, residual = a.mix(hyper)
            block2, residual = a.combine_mix(attn_out, residual, m)
            outs.append((block, block2, m.combine(mlp_out, residual)))
    # Engagement: one fused mix, one fused combine+mix, one tiled combine;
    # the reference took hc_full twice and the tiled combine twice.
    assert calls == ["mix", "combine_mix", "combine"], calls
    assert ref_calls == ["full", "combine", "full", "combine"], ref_calls
    (r_block, r_block2, r_out), (block, block2, out) = outs
    # The first mix is hc_full's math bit for bit; after it, the injection
    # gate's summation order (in-kernel vs the XLA matmul) may move a bf16
    # boundary by one ulp on a few elements.
    np.testing.assert_array_equal(block.view(torch.int16).numpy(), r_block.view(torch.int16).numpy())
    for name, got, want in (("mlp mixed", block2, r_block2), ("layer out", out, r_out)):
        _close_bf16(got.float().numpy(), want.float().numpy(), name)


@pytest.fixture
def fake_adapter(monkeypatch):
    """Stand-in for kernels/qwen4_exp/vllm_adapter (it needs torch_tpu): every
    builder returns a named sentinel, so GatedResidual's real constructor runs."""
    import sys
    import types

    built = []
    fake = types.ModuleType("vllm_torchtpu.kernels.qwen4_exp.vllm_adapter")

    def builder(name):
        def build(*args):
            built.append(name)
            return lambda *a: name

        return build

    for name in ("build_gated_residual_op", "build_full_mix_op", "build_norm_op",
                 "build_up_epilogue_op", "build_decode_mix_op", "build_decode_combine_mix_op",
                 "build_prefill_inj_mix_op", "build_prefill_combine_mix_op"):
        setattr(fake, name, builder(name))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    for name in ("TPU_QWEN4_HC_PREFILL_ODD_TILES", "TPU_QWEN4_HC_PREFILL_FUSED_MIX", "TPU_QWEN4_HC_DECODE_FUSED",
                 "TPU_QWEN4_HC_PREFILL_PARTIAL_TILE"):
        monkeypatch.setattr(envs, name, False)
    return built


@pytest.mark.parametrize(
    "flag,tile,expected",
    [
        # (6,400, 8,192, 3,200, one tile): the served default takes even counts only.
        (None, 256, (False, True, False, False)),
        ("TPU_QWEN4_HC_PREFILL_ODD_TILES", 256, (True, True, False, False)),
        ("TPU_QWEN4_HC_PREFILL_ODD_TILES", 128, (True, True, True, False)),
        ("TPU_QWEN4_HC_PREFILL_FUSED_MIX", 128, (True, True, True, False)),
        # partial last tile: 3,200 at 256-row tiles (12 + a 128-row tile)
        ("TPU_QWEN4_HC_PREFILL_PARTIAL_TILE", 256, (True, True, True, False)),
    ],
)
def test_prefill_tile_gate(monkeypatch, fake_adapter, flag, tile, expected):
    if flag:
        monkeypatch.setattr(envs, flag, True)
    m = GatedResidual(_module(1).config, prefill_full_tile=tile)
    got = tuple(m._prefill_full_rows(t) for t in (6400, 8192, 3200, tile))
    assert got == expected, (flag, tile, got)
    if flag == "TPU_QWEN4_HC_PREFILL_PARTIAL_TILE":
        assert m._prefill_partial and not m._prefill_full_rows(3216)  # 32-row multiples only
    else:
        assert not m._prefill_partial
    fused_built = "build_prefill_inj_mix_op" in fake_adapter
    assert fused_built == (flag == "TPU_QWEN4_HC_PREFILL_FUSED_MIX")
    assert (m._prefill_inj_mix_op is not None) == fused_built
    assert (m._prefill_combine_mix_op is not None) == fused_built


def test_fused_mix_not_built_without_inject_or_tile(monkeypatch, fake_adapter):
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_FUSED_MIX", True)
    assert GatedResidual(_module(2).config, prefill_full_tile=0)._prefill_inj_mix_op is None
    cfg = _module(2).config
    no_combine = GatedResidual(cfg, prefill_full_tile=128, use_combine=False)
    assert no_combine._prefill_inj_mix_op is None  # the final mixer has no combine
    assert GatedResidual(cfg, prefill_full_tile=128)._prefill_inj_mix_op is not None


def test_gdn_zero_init_policy(monkeypatch):
    from vllm_torchtpu.layers.core import gdn_attention as G

    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT", False)
    assert G.pooled_gdn_zero_initialize_out(6400, 8, 0)
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT", True)
    assert not G.pooled_gdn_zero_initialize_out(6400, 8, 0)  # prefill bucket
    assert not G.pooled_gdn_zero_initialize_out(3200, 1, 0)
    assert G.pooled_gdn_zero_initialize_out(8, 8, 0)  # decode bucket
    assert G.pooled_gdn_zero_initialize_out(32, 8, 3)  # MTP verify windows
