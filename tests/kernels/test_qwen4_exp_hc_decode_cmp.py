# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_COMPARE: hc_decode_cmp (interpret mode) against hc_decode.

The two kernels share their rounding boundaries; they differ in summation
orders and in the sigmoid's implementation, so outputs agree to the bf16
tolerance of ``_check_bf16`` rather than bit for bit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode_cmp
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_cmp import (
    cmp_applies,
    hc_decode_mix_auto,
    hc_decode_mix_cmp,
)

BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN


def _inputs(seed, tokens, inject):
    keys = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
    block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
    injection = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
    weights = [
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
    ]
    if inject:
        weights.append((0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16))
    return residual, block, injection.astype(jnp.float32), weights


def _check_bf16(actual, expected, name):
    """hc_decode's tolerance against the torch path: a bf16 rounding boundary
    may move by one ulp on a small fraction of elements."""
    actual = np.asarray(jnp.asarray(actual, jnp.float32))
    expected = np.asarray(jnp.asarray(expected, jnp.float32))
    assert np.isfinite(actual).all(), name
    diff = np.abs(actual - expected)
    scale = np.abs(expected).max()
    assert diff.max() <= 2.0**-7 * scale, (name, diff.max(), scale)
    assert np.linalg.norm(diff) <= 1e-3 * np.linalg.norm(expected), name
    assert (diff > 0).mean() <= 0.02, (name, (diff > 0).mean())


def test_cmp_applies():
    assert [t for t in range(1, 600) if cmp_applies(t)] == list(range(16, 257, 16))


@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("prologue", [True, False])
@pytest.mark.parametrize("tokens", [16, 48, 256])
def test_cmp_matches_served_kernel(tokens, prologue, inject):
    residual, block, injection, weights = _inputs(tokens, tokens, inject)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS)
    extra = dict(block=block, prev_injection=injection) if prologue else {}
    interpret = pltpu.InterpretParams(uninitialized_memory="nan")
    ref = hc_decode_mix(residual, *weights, **extra, **static, interpret=interpret)
    out = hc_decode_mix_cmp(residual, *weights, **extra, **static, interpret=interpret)
    assert len(out) == len(ref)
    _check_bf16(out[0], ref[0], "mixed")
    if inject:
        assert out[1].shape == (tokens, BRANCHES) and out[1].dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(out[1]), np.asarray(ref[1]), rtol=2e-2, atol=1e-6)
    else:
        assert out[1] is None and ref[1] is None
    if prologue:
        _check_bf16(out[2], ref[2], "combined")


@pytest.mark.parametrize("tokens", [8, 24, 272])
def test_auto_keeps_served_kernel_outside_cmp_rows(tokens, monkeypatch):
    residual, block, injection, weights = _inputs(tokens, tokens, True)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, max_tile=512)
    extra = dict(block=block, prev_injection=injection)

    def refuse(*args, **kwargs):
        raise AssertionError("cmp kernel at a served row count")

    monkeypatch.setattr(hc_decode_cmp, "hc_decode_mix_cmp", refuse)
    interpret = pltpu.InterpretParams()
    ref = hc_decode_mix(residual, *weights, **extra, **static, interpret=interpret)
    out = hc_decode_mix_auto(residual, *weights, **extra, **static, interpret=interpret)
    for o, r in zip(out, ref):
        np.testing.assert_array_equal(
            np.asarray(jnp.asarray(o, jnp.float32)), np.asarray(jnp.asarray(r, jnp.float32))
        )


def test_cmp_rejects_unsupported_rows():
    shape = jax.ShapeDtypeStruct
    args = (
        shape((24, WIDTH), jnp.bfloat16),
        shape((WIDTH,), jnp.bfloat16),
        shape((LOWRANK, WIDTH), jnp.bfloat16),
        shape((WIDTH, LOWRANK), jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="16-row pieces"):
        jax.eval_shape(
            lambda *a: hc_decode_mix_cmp(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS),
            *args,
        )


@pytest.mark.parametrize("compare", [False, True])
def test_op_names_follow_the_flag(compare, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_COMPARE", compare)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_TILE_ROWS", 512)
    names = []
    monkeypatch.setattr(
        vllm_adapter.pallas,
        "jax_op",
        lambda name, core: names.append(name) or _FakeOp(),
    )
    for build in (vllm_adapter.build_decode_mix_op, vllm_adapter.build_decode_combine_mix_op):
        build.__wrapped__(BRANCHES, HIDDEN, EPS, True)
        build.__wrapped__(BRANCHES, HIDDEN, EPS, False)
    assert len(names) == 4
    assert all(n.endswith("_t512_cmp_v1" if compare else "_t512_v1") for n in names), names


class _FakeOp:
    def register_fake(self, fn):
        return fn


def test_layer_plumbing_with_cmp_matches_unfused_layer():
    """attn mix -> combine_mix -> mlp combine through GatedResidual, as
    Qwen4ExpDecoderLayer runs, with the decode ops running hc_decode_mix_auto
    at a cmp row count."""
    import torch

    from tests.kernels import test_qwen4_exp_hc_decode as served_test

    tokens = 16
    interpret = pltpu.InterpretParams()

    def to_torch(a):
        if a.dtype == jnp.bfloat16:
            return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)
        return torch.from_numpy(np.array(a))

    def to_jax(t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            return served_test._jax(t)
        return jnp.asarray(t.numpy())

    def ops(module):
        static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, max_tile=512, interpret=interpret)

        def mix_op(x, *weights):
            assert cmp_applies(x.shape[0])
            mixed, inj = hc_decode_mix_auto(*map(to_jax, (x, *weights)), **static)
            return to_torch(mixed) if inj is None else (to_torch(mixed), to_torch(inj))

        def combine_mix_op(residual, block, gate, *weights):
            mixed, inj, combined = hc_decode_mix_auto(
                *map(to_jax, (residual, *weights)),
                block=to_jax(block),
                prev_injection=to_jax(gate),
                **static,
            )
            if inj is None:
                return to_torch(mixed), to_torch(combined)
            return to_torch(mixed), to_torch(inj), to_torch(combined)

        module._decode_mix_op = mix_op
        module._decode_combine_mix_op = combine_mix_op
        return module

    attn, mlp = served_test._module(21), served_test._module(22)
    fused_attn, fused_mlp = ops(served_test._module(21)), ops(served_test._module(22))
    torch.manual_seed(23)
    hyper = torch.randn(tokens, WIDTH).to(torch.bfloat16)
    attn_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    mlp_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        outs = []
        for a, m in ((attn, mlp), (fused_attn, fused_mlp)):
            block, residual = a.mix(hyper)
            block2, residual = a.combine_mix(attn_out, residual, m)
            outs.append((block, block2, m.combine(mlp_out, residual)))
    (ref_block, ref_block2, ref_out), (block, block2, out) = outs
    assert fused_attn._decode_fused(hyper)
    served_test._check_bf16(served_test._jax(block), ref_block, "attn mixed")
    served_test._check_bf16(served_test._jax(block2), ref_block2, "mlp mixed")
    served_test._check_bf16(served_test._jax(out), ref_out, "layer output")
