# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""FUSED_MOE_DECODE_KERNEL_PADDING_MASK in the Pallas interpreter on 8 host
devices: the folded kernel given the real-row count n (`valid_rows`, rank[0, 2])
and rows from n on left as the caller finds them (hidden states and route
weights holding NaN, inf and large values, routes to any expert) returns the
same bits, on every row, as the call without the flag on the masked inputs (those
rows zeroed, with the routes a zero row's router picks, as the decoder layer's
mask and zero_routing_weights_for_padding leave them today).

Needs 8 host devices: JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode_kernel as kmod,
)

N, H = 8, 2560
E, INTER, INTER_S, TOPK = 2, 128, 128, 2
F8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _q(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    return jnp.asarray(w / s).astype(F8), s.astype(np.float32)


def _weights(seed, fp8_shared):
    rng = np.random.default_rng(seed)
    w1, w1_s = _q(rng.normal(size=(N, E, H, 2 * INTER)) * 0.02, 2)
    w2, w2_s = _q(rng.normal(size=(N, E, INTER, H)) * 0.02, 2)
    if fp8_shared:
        sw1, sw1_s = _q(rng.normal(size=(N, H, 2 * INTER_S)) * 0.02, 1)
        sw2, sw2_s = _q(rng.normal(size=(N, INTER_S, H)) * 0.02, 1)
    else:
        sw1 = jnp.asarray(rng.normal(size=(N, H, 2 * INTER_S)) * 0.02, jnp.bfloat16)
        sw2 = jnp.asarray(rng.normal(size=(N, INTER_S, H)) * 0.02, jnp.bfloat16)
        sw1_s = np.ones((N, 1, 2 * INTER_S), np.float32)
        sw2_s = np.ones((N, 1, H), np.float32)
    gate = jnp.asarray(rng.normal(size=(1, H)) * 0.02, jnp.bfloat16)
    return w1, w1_s, w2, w2_s, sw1, sw1_s, sw2, sw2_s, gate


def _rows(tokens, n, seed, garbage):
    """x, tw, ids for `tokens` rows of which the first n are real. garbage:
    rows from n on as the caller passes them with the flag; else as the masks
    leave them (zero rows and weights, a zero row's routes: the lowest ids)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(tokens, H)).astype(np.float32)
    tw = rng.dirichlet(np.ones(TOPK), size=tokens).astype(np.float32)
    ids = np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(tokens)])
    pad = tokens - n
    if garbage:
        junk = rng.normal(size=(pad, H)).astype(np.float32) * 1e4
        junk.flat[:: max(1, junk.size // 37)] = np.nan
        junk.flat[5 :: max(1, junk.size // 29)] = np.inf
        junk.flat[7 :: max(1, junk.size // 31)] = -np.inf
        x[n:] = junk
        tw[n:] = np.where(rng.random((pad, TOPK)) < 0.5, np.nan, 3.0)
        ids[n:] = rng.integers(-N * E, 2 * N * E, size=(pad, TOPK))
    else:
        x[n:] = 0.0
        tw[n:] = 0.0
        ids[n:] = np.arange(TOPK)
    return (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(tw, jnp.bfloat16),
        jnp.asarray(ids, jnp.int32),
    )


def _call(
    mesh,
    tokens,
    n,
    *,
    flag,
    gate_in_kernel=False,
    exact_width=False,
    fp8_shared=False,
    seed=0,
):
    w1, w1_s, w2, w2_s, sw1, sw1_s, sw2, sw2_s, gate = _weights(seed, fp8_shared)
    x, tw, ids = _rows(tokens, n, seed + 1, garbage=flag)
    cols = [np.arange(N), np.arange(N) * E] + ([np.full(N, n)] if flag else [])
    rank = np.stack(cols, 1).astype(np.int32)
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())

    def body(rank, w1, w2, w1_s, w2_s, sw1, sw1_s, sw2, sw2_s, x, gate, tw, ids):
        return kmod.fused_moe_decode(
            rank,
            x,
            w1[0],
            w2[0],
            w1_s[0][:, :, None, :],
            w2_s[0][:, :, None, :],
            sw1[0],
            sw1_s[0],
            sw2[0],
            sw2_s[0],
            gate,
            tw,
            ids,
            n_ranks=N,
            expert_start="rank",
            fold=True,
            gate_in_kernel=gate_in_kernel,
            interpret=True,
            exact_width=exact_width,
            valid_rows=flag,
        )[None]

    per_rank = (w1, w2, w1_s, w2_s, sw1, sw1_s, sw2, sw2_s)
    args = [
        jax.device_put(rank, sh),
        *(jax.device_put(jnp.asarray(a), sh) for a in per_rank),
        *(jax.device_put(a, rep) for a in (x, gate, tw, ids)),
    ]
    out = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("d"),) * 9 + (P(),) * 4,
            out_specs=P("d"),
            check_vma=False,
        )
    )(*args)
    return np.asarray(out).view(np.uint16)


CASES = {
    # The served 16-row decode bucket: XLA gate, BF16 shared, exact width.
    "16 rows, 11 real": dict(tokens=16, n=11, exact_width=True),
    "16 rows, all real": dict(tokens=16, n=16, exact_width=True),
    "16 rows, none real": dict(tokens=16, n=0, exact_width=True),
    "13 rows padded to 16, gate in kernel": dict(tokens=13, n=9, gate_in_kernel=True),
    "FP8 shared expert": dict(tokens=16, n=11, fp8_shared=True),
    "FP8 shared expert, gate in kernel": dict(
        tokens=16, n=5, fp8_shared=True, gate_in_kernel=True
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_flag_is_bitwise_the_masked_call(mesh, monkeypatch, case):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    kw = CASES[case]
    on = _call(mesh, flag=True, **kw)
    off = _call(mesh, flag=False, **kw)
    assert on.shape == off.shape == (N, kw["tokens"], H)
    floats = off.view(jnp.bfloat16).astype(np.float32)
    assert np.isfinite(floats).all()
    np.testing.assert_array_equal(on, off, err_msg=case)


def test_must_fail_control(mesh, monkeypatch):
    """The garbage rows without the flag: they reach the output (NaN)."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w1, w1_s, w2, w2_s, sw1, sw1_s, sw2, sw2_s, gate = _weights(0, False)
    x, tw, ids = _rows(16, 11, 1, garbage=True)
    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())

    def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate, tw, ids):
        return kmod.fused_moe_decode(
            rank, x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :],
            sw1[0], jnp.ones((2 * INTER_S,), jnp.float32), sw2[0],
            jnp.ones((H,), jnp.float32), gate, tw, ids, n_ranks=N,
            expert_start="rank", fold=True, interpret=True,
        )[None]

    args = [
        jax.device_put(rank, sh),
        *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, sw1, sw2)),
        *(jax.device_put(a, rep) for a in (x, gate, tw, ids)),
    ]
    out = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(P("d"),) * 7 + (P(),) * 4,
            out_specs=P("d"), check_vma=False,
        )
    )(*args)
    out = np.asarray(out.astype(jnp.float32))
    assert not np.isfinite(out[:, 11:]).all()
    assert np.isfinite(out[:, :11]).all()


def test_flag_needs_fold_and_the_count_column():
    x = jnp.zeros((16, H), jnp.bfloat16)
    w1 = jnp.zeros((E, H, 2 * INTER), F8)
    w2 = jnp.zeros((E, INTER, H), F8)
    with pytest.raises(ValueError, match="valid_rows needs fold"):
        kmod.fused_moe_decode(
            jnp.zeros((1,), jnp.int32), x, w1, w2, *(None,) * 9, n_ranks=N,
            valid_rows=True,
        )
    with pytest.raises(ValueError, match=r"valid_rows needs rank \[1, 3\]"):
        kmod.fused_moe_decode(
            jnp.zeros((1, 2), jnp.int32), x, w1, w2, *(None,) * 9, n_ranks=N,
            expert_start="rank", fold=True, valid_rows=True,
        )
