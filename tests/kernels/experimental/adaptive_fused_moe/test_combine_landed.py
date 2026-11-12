# SPDX-License-Identifier: Apache-2.0
"""MOE_ADAPTIVE_COMBINE_LANDED: the landed combine is bit-identical to the default combine.

Runs both Mosaic combines in Pallas TPU interpret mode, so it needs no TPU.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import kernel as K


@pytest.fixture
def interpret(monkeypatch):
    real = pl.pallas_call
    monkeypatch.setattr(
        K.pl, "pallas_call", functools.partial(real, interpret=pltpu.InterpretParams())
    )
    monkeypatch.setattr(host, "vmem_limit", lambda: 64 << 20)
    monkeypatch.setattr(K, "_COMBINE_CACHE", {})
    yield monkeypatch


def _combine(monkeypatch, landed, arrivals, coef, pos, t_local, topk, hidden):
    monkeypatch.setattr(K, "COMBINE_LANDED", landed)
    monkeypatch.setattr(K, "_COMBINE_CACHE", {})
    fn = K.build_combine_kernel(
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=hidden // host.HIDDEN_LANE_BLOCK,
        wire_dtype=jnp.float8_e4m3fn,
        out_dtype=jnp.bfloat16,
    )
    return np.asarray(jax.jit(fn)(arrivals, coef, pos).astype(jnp.float32))


@pytest.mark.parametrize("t_local,topk,hidden,recv_rows", [(256, 10, 256, 2600), (512, 4, 384, 2100)])
def test_landed_combine_bit_identical(interpret, t_local, topk, hidden, recv_rows):
    assert K.combine_step_tokens(t_local)
    rng = np.random.default_rng(t_local + topk)
    lb = hidden // host.HIDDEN_LANE_BLOCK
    arrivals = jnp.asarray(
        rng.normal(size=(recv_rows, lb, host.HIDDEN_LANE_BLOCK)).astype(np.float32)
    ).astype(jnp.float8_e4m3fn)
    pos = jnp.asarray(rng.integers(0, recv_rows, size=(t_local, topk)), jnp.int32)
    coef = jnp.asarray(rng.normal(size=(t_local, topk)).astype(np.float32))
    base = _combine(interpret, False, arrivals, coef, pos, t_local, topk, hidden)
    landed = _combine(interpret, True, arrivals, coef, pos, t_local, topk, hidden)
    assert np.array_equal(base, landed)
    ref = (
        np.asarray(arrivals.astype(jnp.float32)).reshape(recv_rows, hidden)[np.asarray(pos)]
        * np.asarray(coef)[:, :, None]
    ).sum(1)
    np.testing.assert_allclose(landed, ref, rtol=2e-2, atol=2e-2)


def test_landed_falls_back_when_too_big(interpret):
    interpret.setattr(host, "vmem_limit", lambda: 1 << 20)
    interpret.setattr(K, "COMBINE_LANDED", True)
    fn_args = dict(t_local=256, topk=10, hidden=256, lane_blocks=2,
                   wire_dtype=jnp.float8_e4m3fn, out_dtype=jnp.bfloat16)
    fn = K.build_combine_kernel(**fn_args)
    arrivals = jnp.zeros((4096, 2, 128), jnp.float8_e4m3fn)
    out = jax.jit(fn)(arrivals, jnp.ones((256, 10), jnp.float32), jnp.zeros((256, 10), jnp.int32))
    assert out.shape == (256, 256)
