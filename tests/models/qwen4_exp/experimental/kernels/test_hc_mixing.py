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
"""Check HC prefill kernels against the staged BF16 arithmetic contract."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.gated_residual import (
    gated_residual_fn,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.hc_full import (
    _combine_full_mix,
    _full_mix,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.hc_norm import _normalize
from vllm_torchtpu.models.qwen4_exp.experimental.kernels.hc_up import _up_epilogue


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


@pytest.mark.parametrize(
    "kernel,tokens",
    # 127 is not a whole tile. The full mix starts its first DMAs before the
    # tile loop, so it also rejects an empty step (0) and one tile of 128 (it
    # double-buffers and needs an even tile count).
    [("norm", 127), ("up", 127), ("full", 127), ("full", 0), ("full", 128)],
)
def test_hc_rejects_partial_token_tiles(kernel, tokens):
    shape = jax.ShapeDtypeStruct
    x, weight, down, up = [shape(a.shape, jnp.bfloat16) for a in inputs(tokens)]
    if kernel == "norm":
        fn = functools.partial(
            _normalize, branches=4, hidden=128, eps=1e-6, token_tile=128
        )
        args = (x, weight)
    elif kernel == "up":
        fn = functools.partial(_up_epilogue, branches=4, hidden=128, token_tile=128)
        args = (shape((tokens, 128), jnp.bfloat16), x, up)
    else:
        fn = functools.partial(
            _full_mix, branches=4, hidden=128, eps=1e-6, token_tile=128
        )
        args = (x, weight, down, up)
    with pytest.raises(ValueError, match="token|tiles"):
        jax.eval_shape(fn, *args)


@pytest.mark.parametrize("token_tile", [16, 48])
def test_hc_combine_rejects_tiles_not_multiple_of_32(token_tile):
    shape = jax.ShapeDtypeStruct
    tokens = 4 * token_tile
    x, weight, down, up = [shape(a.shape, jnp.bfloat16) for a in inputs(tokens)]
    block = shape((tokens, 128), jnp.bfloat16)
    gates = shape((tokens, 4), jnp.float32)
    fn = functools.partial(
        _combine_full_mix,
        branches=4,
        hidden=128,
        eps=1e-6,
        token_tile=token_tile,
        up_t=False,
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        jax.eval_shape(fn, x, block, gates, weight, down, up)


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


def _interpret(mp):
    from jax.experimental import pallas as pl

    mp.setattr(pl, "pallas_call", functools.partial(pl.pallas_call, interpret=True))


def _gated_residual_reference(x, block, gates):
    """bf16(f32(residual_b) + f32(block) * gates[:, b]) per branch, in NumPy."""
    x, block = (np.asarray(a, np.float32) for a in (x, block))
    branches = gates.shape[1]
    out = x.reshape(len(x), branches, -1) + block[:, None] * gates[:, :, None]
    return np.asarray(jnp.asarray(out.reshape(x.shape), jnp.bfloat16))


def _gated_residual_matches(device):
    put = functools.partial(jax.device_put, device=device)
    x, weight, down, up = (put(jnp.asarray(a, jnp.bfloat16)) for a in inputs(256))
    rng = np.random.default_rng(11)
    block = put(jnp.asarray(rng.normal(size=(256, 128)) * 0.1, jnp.bfloat16))
    gates_np = rng.uniform(0, 2, size=(256, 4)).astype(np.float32)
    gates = put(jnp.asarray(gates_np))
    expected = _gated_residual_reference(x, block, gates_np).astype(np.float32)
    gated = gated_residual_fn(4, 128, 128)(x, block, gates)
    # FMA contraction may move an element by one BF16 ulp.
    np.testing.assert_allclose(
        np.asarray(gated, np.float32), expected, rtol=2**-7, atol=1e-6
    )
    static = dict(branches=4, hidden=128, eps=1e-6, token_tile=128, up_t=False)
    _, _, combined = _combine_full_mix(x, block, gates, weight, down, up, **static)
    _bitwise((combined,), (gated,))


def test_gated_residual_matches_numpy_interpret():
    with pytest.MonkeyPatch.context() as mp:
        _interpret(mp)
        _combine_full_mix.clear_cache()
        try:
            _gated_residual_matches(jax.devices("cpu")[0])
        finally:
            _combine_full_mix.clear_cache()


@pytest.mark.tpu_test
def test_gated_residual_matches_numpy():
    _gated_residual_matches(jax.devices("tpu")[0])


def test_full_mix_small_tile_matches_wide_tile_interpret():
    """token_tile < 32 takes the whole-tile normalization branch; rows are
    independent, so it must match the 32-row-chunk branch bit for bit."""
    with pytest.MonkeyPatch.context() as mp:
        _interpret(mp)
        _full_mix.clear_cache()
        try:
            x, weight, down, up = (jnp.asarray(a, jnp.bfloat16) for a in inputs(64))
            static = dict(branches=4, hidden=128, eps=1e-6)
            _bitwise(
                _full_mix(x, weight, down, up, **static, token_tile=16),
                _full_mix(x, weight, down, up, **static, token_tile=32),
            )
        finally:
            _full_mix.clear_cache()
