# SPDX-License-Identifier: Apache-2.0
"""Check HC prefill kernels against the staged BF16 arithmetic contract."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("kernel", ["norm", "up"])
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
    with pytest.raises(ValueError, match="token tiles"):
        jax.eval_shape(fn, *args)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [256, 512])
def test_hc_staged_mix_matches_cpu_reference(tokens):
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
    # Use the existing HC layer's NRMSE/cosine numerical acceptance contract.
    for actual, expected in [
        (norm, expected_norm),
        (mixed, expected_mix),
    ]:
        a, e = (np.asarray(v, np.float32) for v in (actual, expected))
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - e) / np.linalg.norm(e) <= 0.01
        assert (
            np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
            >= 0.999
        )
