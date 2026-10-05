"""Checks for the DeepSeek V4.1 delayed mHC shared by the v1 and v2 paths.

The oracle is vLLM's ``mhc_pre_delayed_torch``. ``delayed_pre_flat_jax``
takes the XLA path below 64 tokens (or off a multiple of 64) and the Pallas
pass otherwise; the Pallas cases need TPU hardware.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc

MULT = 4
MIXES = MULT * (MULT + 2)
CONSTANTS = (1e-6, 1e-6, 1e-6, 2.0, 20)  # rms, pre, sinkhorn eps; post mult; repeats
GATE_RTOL, GATE_ATOL = 5e-3, 1e-5
BF16_RTOL, BF16_ATOL = 2e-2, 2e-2


def _inputs(tokens, dim, seed=0):
    rng = np.random.default_rng(seed)
    residual = torch.from_numpy(rng.standard_normal((tokens, MULT, dim))).bfloat16()
    fn = torch.from_numpy(0.02 * rng.standard_normal((MIXES, MULT * dim))).float()
    scale = torch.from_numpy(1.0 + 0.1 * rng.standard_normal(3)).float()
    base = torch.from_numpy(0.5 * rng.standard_normal(MIXES)).float()
    previous_pre = torch.from_numpy(rng.uniform(0.0, 1.0, (tokens, MULT))).float()
    return residual, fn, scale, base, previous_pre


def _jax(t: torch.Tensor) -> jax.Array:
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.float().numpy()
    return np.asarray(jnp.asarray(a).astype(jnp.float32))


def _check(tokens, dim, packed):
    residual, fn, scale, base, previous_pre = _inputs(tokens, dim)
    weight = mhc.pack_mix_weight(fn) if packed else fn
    got = mhc.delayed_pre_flat_jax(
        _jax(residual.flatten(1)),
        _jax(weight),
        _jax(scale),
        _jax(base),
        _jax(previous_pre),
        *CONSTANTS,
    )
    want = mhc_pre_delayed_torch(
        residual, fn, scale, base, *CONSTANTS, pre_mix=previous_pre
    )
    post, comb, value, new_pre = got
    assert post.shape == tuple(want[0].shape)
    assert value.dtype == jnp.bfloat16
    for g, w in ((post, want[0]), (comb, want[1]), (new_pre, want[3])):
        np.testing.assert_allclose(_np(g), _np(w), rtol=GATE_RTOL, atol=GATE_ATOL)
    np.testing.assert_allclose(_np(value), _np(want[2]), rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.cpu_test
def test_packed_weight_is_exact_split():
    _, fn, *_ = _inputs(1, 256)
    packed = mhc.pack_mix_weight(fn).float()  # [K, 128]: hi | mid | lo | 0
    parts = packed[:, : 3 * MIXES].T.reshape(3, MIXES, -1)
    torch.testing.assert_close(parts.sum(0), fn, rtol=0, atol=0)
    assert torch.count_nonzero(packed[:, 3 * MIXES :]) == 0


@pytest.mark.cpu_test
def test_torch_and_jax_packing_agree():
    _, fn, *_ = _inputs(1, 256)
    want = mhc._pack_mix_weight(_jax(fn))
    np.testing.assert_array_equal(_np(mhc.pack_mix_weight(fn)), _np(want))


@pytest.mark.cpu_test
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 63, 100])
def test_xla_path_matches_torch_reference(tokens, packed):
    """Token counts the Pallas pass does not take."""
    _check(tokens, 256, packed)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens,dim", [(64, 256), (192, 256), (128, 5120)])
def test_pallas_path_matches_torch_reference(tokens, dim, packed):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel needs TPU hardware")
    _check(tokens, dim, packed)
