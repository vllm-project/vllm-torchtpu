# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused TP MoE + shared expert + in-kernel all-reduce."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import moe_fused_tp as kernel

N, H, INTER, E, TOPK = 8, 1024, 128, 64, 10
F8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module")
def setup():
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("requires 8 TPU devices")
    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    rank = jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh)

    def gen(r):
        ks = jax.random.split(jax.random.fold_in(jax.random.key(3), r[0, 0]), 4)

        def q(w, axis):
            s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
            return (w / s).astype(F8), s

        w1, s1 = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
        w2, s2 = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
        w1s, s1s = q(jax.random.normal(ks[2], (H, 2 * INTER)) * 0.02, 0)
        w2s, s2s = q(jax.random.normal(ks[3], (INTER, H)) * 0.02, 0)
        return w1, w2, s1[:, :, None, :], s2[:, :, None, :], w1s, s1s, w2s, s2s

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(rank)
    wg = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    return mesh, sh, rank, weights, wg


def _ffn(x, w1, s1, w2, s2):
    gu = jnp.dot(x, w1.astype(jnp.float32) * s1)
    return jnp.dot(
        jax.nn.silu(gu[:, :INTER]) * gu[:, INTER:], w2.astype(jnp.float32) * s2
    )


def _routed(x, w1, w2, s1, s2, tw, ti, *, skip_expert=None):
    out = jnp.zeros((x.shape[0], H), jnp.float32)
    for e in range(E):
        coeff = jnp.sum(
            jnp.where(ti == e, tw.astype(jnp.float32), 0.0), 1, keepdims=True
        )
        if e != skip_expert:
            out = out + coeff * _ffn(x, w1[e], s1[e, 0], w2[e], s2[e, 0])
    return out


def _reference(
    rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, skip_expert=None
):
    """The decomposed layer in f32: routed experts + gated shared expert + psum."""
    xf = x.astype(jnp.float32)
    shared = _ffn(xf, w1s, s1s, w2s, s2s)
    shared = jax.nn.sigmoid(jnp.dot(x, wg, preferred_element_type=jnp.float32)) * shared
    routed = _routed(xf, w1, w2, s1, s2, tw, ti, skip_expert=skip_expert)
    return jax.lax.psum(routed + shared, "d").astype(x.dtype)


def _fused(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti):
    return kernel.fused_moe_tp_ar(
        rank.reshape(-1), x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, n_ranks=N
    )


@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("padded", [False, True])
def test_matches_decomposed_layer(setup, m, padded):
    mesh, sh, rank, weights, wg = setup
    rng = np.random.default_rng(m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(m)]).astype(
        np.int32
    )
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    if padded:
        tw[m - 3 :] = 0
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        wg,
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def run(fn):
        out = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )(*args)
        return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)

    def max_rel(a, ref):
        return (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()

    got, ref = run(_fused), run(_reference)
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    # FP8 activations and a BF16 all-reduce against f32 math: a few percent per row.
    assert max_rel(got[0], ref[0]) < 0.08
    # Positive control: a reference missing one routed expert must fail the bound.
    broken = run(functools.partial(_reference, skip_expert=int(ids[0, 0])))
    assert max_rel(got[0], broken[0]) > 0.08
