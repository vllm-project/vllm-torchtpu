# SPDX-License-Identifier: Apache-2.0
"""Check HC prefill kernels against the staged BF16 arithmetic contract."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _combine_full_mix, _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue


def inputs(tokens=256):
    rng = np.random.default_rng(41)
    shapes = ((tokens, 512), (512,), (128, 512), (512, 128))
    return tuple((rng.normal(size=s) * 0.1).astype(np.float32) for s in shapes)


def reference(x, weight, down, up):
    # Each branch has two BF16 boundaries before the residual normalization sum.
    values = x.astype(jnp.float32).reshape(-1, 4, 128)
    norm = values * jax.lax.rsqrt(jnp.mean(values**2, -1, keepdims=True) + 1e-6)
    norm = norm.astype(jnp.bfloat16).reshape(x.shape).astype(jnp.float32)
    scaled = (norm * weight.astype(jnp.float32)).astype(jnp.bfloat16)
    norm = (norm + scaled.astype(jnp.float32)).astype(jnp.bfloat16)
    mid = (norm.astype(jnp.float32) @ down.astype(jnp.float32).T).astype(jnp.bfloat16)
    mid = jax.nn.silu(mid.astype(jnp.float32) / 4).astype(jnp.bfloat16)
    logits = (mid.astype(jnp.float32) @ up.astype(jnp.float32).T).astype(jnp.bfloat16)
    weighted = jax.nn.sigmoid(logits.astype(jnp.float32)) * norm.astype(jnp.float32)
    mixed = jnp.mean(weighted.reshape(-1, 4, 128), axis=1).astype(jnp.bfloat16)
    return mixed, norm, mid


@pytest.mark.parametrize("kernel", ["norm", "up", "full"])
def test_hc_rejects_partial_token_tiles(kernel):
    shape = jax.ShapeDtypeStruct
    x, weight, down, up = [shape(a.shape, jnp.bfloat16) for a in inputs(127)]
    if kernel == "norm":
        fn = functools.partial(
            _normalize, branches=4, hidden=128, eps=1e-6, token_tile=128
        )
        args = (x, weight)
    elif kernel == "up":
        fn = functools.partial(_up_epilogue, branches=4, hidden=128, token_tile=128)
        args = (shape((127, 128), jnp.bfloat16), x, up)
    else:
        fn = functools.partial(
            _full_mix, branches=4, hidden=128, eps=1e-6, token_tile=128
        )
        args = (x, weight, down, up)
    with pytest.raises(ValueError, match="token tiles"):
        jax.eval_shape(fn, *args)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [256, 512])
def test_hc_staged_and_full_mix_match_cpu_reference(tokens):
    cpu = jax.devices("cpu")[0]
    tpu = jax.devices("tpu")[0]
    with jax.default_device(cpu):
        arrays = tuple(jnp.asarray(a, jnp.bfloat16) for a in inputs(tokens))
        expected_mix, expected_norm, mid = reference(*arrays)
    x, weight, down, up = tuple(jax.device_put(a, tpu) for a in arrays)
    norm = _normalize(x, weight, branches=4, hidden=128, eps=1e-6, token_tile=128)
    mixed = _up_epilogue(
        jax.device_put(mid, tpu), norm, up, branches=4, hidden=128, token_tile=128
    )
    full_mix, full_norm = _full_mix(
        x, weight, down, up, branches=4, hidden=128, eps=1e-6, token_tile=128
    )
    # Use the existing HC layer's NRMSE/cosine numerical acceptance contract.
    for actual, expected in [
        (norm, expected_norm),
        (mixed, expected_mix),
        (full_norm, expected_norm),
        (full_mix, expected_mix),
    ]:
        a, e = (np.asarray(v, np.float32) for v in (actual, expected))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )


def _bitwise(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a).view(np.uint16), np.asarray(e).view(np.uint16)
        )


def _full_mix_variants_match(device):
    """The transposed-up and the fused combine variants are bit-identical."""
    put = functools.partial(jax.device_put, device=device)
    x, weight, down, up = (put(jnp.asarray(a, jnp.bfloat16)) for a in inputs(512))
    rng = np.random.default_rng(7)
    block = put(jnp.asarray(rng.normal(size=(512, 128)) * 0.1, jnp.bfloat16))
    gates = put(jnp.asarray(rng.uniform(0, 2, size=(512, 4)), jnp.float32))
    static = dict(branches=4, hidden=128, eps=1e-6, token_tile=128)
    expected = _full_mix(x, weight, down, up, **static)
    _bitwise(_full_mix(x, weight, down, up.T, **static, up_t=True), expected)
    combined = gated_residual_fn(4, 128, 128)(x, block, gates)
    mixed, norm = _full_mix(combined, weight, down, up, **static)
    for up_t in (False, True):
        _bitwise(
            _combine_full_mix(
                x, block, gates, weight, down, up.T if up_t else up, **static, up_t=up_t
            ),
            (mixed, norm, combined),
        )
    # Rows are independent: a wider tile changes only the schedule.
    wide = dict(static, token_tile=256)
    _bitwise(_full_mix(x, weight, down, up, **wide), expected)
    _bitwise(
        _combine_full_mix(x, block, gates, weight, down, up, **wide, up_t=False),
        (mixed, norm, combined),
    )


def test_full_mix_variants_are_bit_identical_interpret():
    from jax.experimental import pallas as pl

    real = pl.pallas_call
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "pallas_call", functools.partial(real, interpret=True))
        _full_mix.clear_cache()
        _combine_full_mix.clear_cache()
        try:
            _full_mix_variants_match(jax.devices("cpu")[0])
        finally:
            _full_mix.clear_cache()
            _combine_full_mix.clear_cache()


@pytest.mark.tpu_test
def test_full_mix_variants_are_bit_identical():
    _full_mix_variants_match(jax.devices("tpu")[0])


def _inject_matches(device):
    """With ``inject`` the second output is the XLA gate projection of ``norm``."""
    put = functools.partial(jax.device_put, device=device)
    x, weight, down, up = (put(jnp.asarray(a, jnp.bfloat16)) for a in inputs(512))
    rng = np.random.default_rng(9)
    inject = put(jnp.asarray(rng.normal(size=(4, 512)) * 0.05, jnp.bfloat16))
    block = put(jnp.asarray(rng.normal(size=(512, 128)) * 0.1, jnp.bfloat16))
    gates = put(jnp.asarray(rng.uniform(0, 2, size=(512, 4)), jnp.float32))
    static = dict(branches=4, hidden=128, eps=1e-6, token_tile=128)
    project = jax.jit(
        lambda n: jax.lax.dot_general(n, inject, (((1,), (1,)), ((), ())))
    )
    mixed, norm = _full_mix(x, weight, down, up, **static)
    _bitwise(_full_mix(x, weight, down, up, inject, **static), (mixed, project(norm)))
    mixed, norm, combined = _combine_full_mix(
        x, block, gates, weight, down, up, **static, up_t=False
    )
    _bitwise(
        _combine_full_mix(
            x, block, gates, weight, down, up, inject, **static, up_t=False
        ),
        (mixed, project(norm), combined),
    )


def test_full_mix_inject_interpret():
    from jax.experimental import pallas as pl

    real = pl.pallas_call
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "pallas_call", functools.partial(real, interpret=True))
        _full_mix.clear_cache()
        _combine_full_mix.clear_cache()
        try:
            _inject_matches(jax.devices("cpu")[0])
        finally:
            _full_mix.clear_cache()
            _combine_full_mix.clear_cache()


@pytest.mark.tpu_test
def test_full_mix_inject():
    _inject_matches(jax.devices("tpu")[0])


def _full_mix_norm_variants_match(device):
    """rows64 / bf16ep / bf16ep64 are bit-identical to vpu; gram* stay within
    one bf16 ulp of it (only the variance's f32 summation order changes)."""
    put = functools.partial(jax.device_put, device=device)
    x, weight, down, up = (put(jnp.asarray(a, jnp.bfloat16)) for a in inputs(512))
    rng = np.random.default_rng(11)
    block = put(jnp.asarray(rng.normal(size=(512, 128)) * 0.1, jnp.bfloat16))
    gates = put(jnp.asarray(rng.uniform(0, 2, size=(512, 4)), jnp.float32))
    static = dict(branches=4, hidden=128, eps=1e-6, token_tile=128)
    expected = _full_mix(x, weight, down, up, **static)
    combined_expected = _combine_full_mix(
        x, block, gates, weight, down, up, **static, up_t=False
    )
    for impl in ("rows64", "bf16ep", "bf16ep64"):
        _bitwise(_full_mix(x, weight, down, up, **static, norm_impl=impl), expected)
        _bitwise(
            _combine_full_mix(
                x, block, gates, weight, down, up, **static, up_t=False, norm_impl=impl
            ),
            combined_expected,
        )
    for impl in ("gram", "gram64"):
        mixed, norm = _full_mix(x, weight, down, up, **static, norm_impl=impl)
        np.testing.assert_allclose(
            np.asarray(norm, np.float32), np.asarray(expected[1], np.float32),
            rtol=1e-2, atol=1e-2,
        )
    with pytest.raises(ValueError, match="norm_impl"):
        _full_mix(x, weight, down, up, **static, norm_impl="nope")


def test_full_mix_norm_variants_interpret():
    from jax.experimental import pallas as pl

    real = pl.pallas_call
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "pallas_call", functools.partial(real, interpret=True))
        _full_mix.clear_cache()
        _combine_full_mix.clear_cache()
        try:
            _full_mix_norm_variants_match(jax.devices("cpu")[0])
        finally:
            _full_mix.clear_cache()
            _combine_full_mix.clear_cache()
