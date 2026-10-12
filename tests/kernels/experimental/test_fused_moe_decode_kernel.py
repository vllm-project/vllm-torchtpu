# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused MoE decode kernel (routed + shared expert +
in-kernel all-reduce), with the experts TP-sliced (every rank holds all E) or
EP-sharded (rank r holds global experts [r*E, (r+1)*E)); the shared expert is
always a TP slice."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode,
    vmem_bytes,
)
from vllm_torchtpu.layers.adapter import fused_moe_decode as adapter

N, H, INTER, INTER_S, E, TOPK = 8, 1024, 256, 128, 64, 10
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

        w1, w1_scale = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
        w2, w2_scale = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
        shared_w1, shared_w1_scale = q(
            jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0
        )
        shared_w2, shared_w2_scale = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
        return (
            w1,
            w2,
            w1_scale[:, :, None, :],
            w2_scale[:, :, None, :],
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
        )

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(rank)
    shared_gate_weight = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    return mesh, sh, rank, weights, shared_gate_weight


def _ffn(x, w1, w1_scale, w2, w2_scale):
    i = w2.shape[0]
    gu = jnp.dot(x, w1.astype(jnp.float32) * w1_scale)
    return jnp.dot(
        jax.nn.silu(gu[:, :i]) * gu[:, i:], w2.astype(jnp.float32) * w2_scale
    )


def _routed(x, w1, w2, w1_scale, w2_scale, tw, ti, *, skip_expert=None):
    out = jnp.zeros((x.shape[0], H), jnp.float32)
    for e in range(E):
        coeff = jnp.sum(
            jnp.where(ti == e, tw.astype(jnp.float32), 0.0), 1, keepdims=True
        )
        if e != skip_expert:
            out = out + coeff * _ffn(x, w1[e], w1_scale[e, 0], w2[e], w2_scale[e, 0])
    return out


def _reference(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    skip_expert=None,
    ep=False,
):
    """The decomposed layer in f32: routed experts + gated shared expert + psum."""
    if ep:
        ti = ti - rank.reshape(()) * E
    xf = x.astype(jnp.float32)
    shared = _ffn(xf, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)
    shared = (
        jax.nn.sigmoid(
            jnp.dot(x, shared_gate_weight, preferred_element_type=jnp.float32)
        )
        * shared
    )
    routed = _routed(xf, w1, w2, w1_scale, w2_scale, tw, ti, skip_expert=skip_expert)
    return jax.lax.psum(routed + shared, "d").astype(x.dtype)


def _fused(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    ep=False,
):
    return fused_moe_decode(
        rank.reshape(-1),
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        tw,
        ti,
        n_ranks=N,
        expert_start=rank.reshape(()) * E if ep else None,
    )


def _fallback(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    ep=False,
):
    """The adapter's per-rank body with every token count above its cap: the GMM
    MoE + shared expert + psum path that replaces the kernel above it."""
    return adapter._layer(
        jnp.concatenate([rank, rank * E], axis=1),  # [[mesh index, first expert]]
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        tw,
        ti,
        n_ranks=N,
        topk=TOPK,
        expert_parallel=ep,
        max_tokens=1,
    )


@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("path", ["kernel", "gmm_fallback"])
def test_matches_decomposed_layer(setup, m, padded, ep, path):
    if path == "gmm_fallback" and (m != 32 or padded):
        pytest.skip("one fallback case per mode")
    mesh, sh, rank, weights, shared_gate_weight = setup
    rng = np.random.default_rng(m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    n_global = N * E if ep else E
    ids = np.stack(
        [rng.choice(n_global, TOPK, replace=False) for _ in range(m)]
    ).astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    if padded:
        tw[m - 3 :] = 0
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        shared_gate_weight,
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

    got = run(functools.partial(_fused if path == "kernel" else _fallback, ep=ep))
    ref = run(functools.partial(_reference, ep=ep))
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    # FP8 activations and a BF16 all-reduce against f32 math: a few percent per row.
    assert max_rel(got[0], ref[0]) < 0.08
    # Positive control: a reference missing one routed expert must fail the bound.
    # In EP mode the local id of expert ids[0, 0] is only that on its owner rank.
    broken = run(functools.partial(_reference, skip_expert=int(ids[0, 0]) % E, ep=ep))
    assert max_rel(got[0], broken[0]) > 0.08


def test_vmem_bound_matches_build_limits():
    """The arming gate's estimate against the Qwen3.5-397B-A17B-FP8 sweep at 8
    ranks: TP (512 experts, i 128) builds at 512 tokens and not 1024; EP (64
    experts, i 1024) builds at 256 and not 512."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    from jax.experimental.pallas import tpu as pltpu

    have = pltpu.get_tpu_info().vmem_capacity_bytes
    assert vmem_bytes(512, 4096, 128, 128, 512, 8, jnp.float8_e4m3fn) <= have
    assert vmem_bytes(1024, 4096, 128, 128, 512, 8, jnp.float8_e4m3fn) > have
    assert vmem_bytes(256, 4096, 1024, 128, 64, 8, jnp.float8_e4m3fn) <= have
    assert vmem_bytes(512, 4096, 1024, 128, 64, 8, jnp.float8_e4m3fn) > have
