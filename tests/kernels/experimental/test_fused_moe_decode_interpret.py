# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE decode kernel in the Pallas interpreter on 8 host devices, at
the Qwen3.8-Flash EP layout's awkward shapes: a hidden size whose 8-way
all-reduce chunk is not lane aligned (1536 / 8 = 192, padded to 256 lanes), a
routed intermediate that is not a multiple of the 512-column activation block
(640 = 512 + 128) and a BF16 shared expert, gated by sigmoid(x . w_g), its
80-wide TP slice zero-padded to 128 as the adapter arms it. Checked against the
decomposed layer in f32 (routed experts + gated shared expert, psum).

Needs 8 host devices: JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode_kernel as kmod,
)
from vllm_torchtpu.layers.adapter.fused_moe_decode import pad_shared_intermediate

N, H, INTER, INTER_S, INTER_S_PAD, E, TOPK, M = 8, 1536, 640, 80, 128, 4, 2, 16
F8 = jnp.float8_e4m3fn


def _q(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    return jnp.asarray(w / s).astype(F8), s.astype(np.float32)


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _inputs(seed):
    rng = np.random.default_rng(seed)
    w1, w1_s = _q(rng.normal(size=(N, E, H, 2 * INTER)) * 0.02, 2)
    w2, w2_s = _q(rng.normal(size=(N, E, INTER, H)) * 0.02, 2)
    # The shared expert's per-rank TP slice in the unquantized layout:
    # gate_up [H, 2 i_s] (gate | up), down [i_s, H].
    sw1 = jnp.asarray(rng.normal(size=(N, H, 2 * INTER_S)) * 0.02, jnp.bfloat16)
    sw2 = jnp.asarray(rng.normal(size=(N, INTER_S, H)) * 0.02, jnp.bfloat16)
    gate = jnp.asarray(rng.normal(size=(H, 1)) * 0.02, jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(M, H)), jnp.bfloat16)
    ids = np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(M)])
    tw = jnp.asarray(rng.dirichlet(np.ones(TOPK), size=M), jnp.bfloat16)
    return x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, jnp.asarray(ids, jnp.int32)


def _reference(
    x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids, *, skip=None, gated=True
):
    """Sum over ranks of each rank's routed experts and shared-expert slice."""
    xf = x.astype(jnp.float32)
    out = jnp.zeros((M, H), jnp.float32)
    g = jax.nn.sigmoid(jnp.dot(xf, gate.astype(jnp.float32))) if gated else 1.0
    for r in range(N):
        gu = xf @ sw1[r].astype(jnp.float32)
        out += g * ((jax.nn.silu(gu[:, :INTER_S]) * gu[:, INTER_S:]) @ sw2[r])
        for e in range(E):
            if r * E + e == skip:
                continue
            coeff = jnp.sum(
                jnp.where(ids == r * E + e, tw.astype(jnp.float32), 0.0),
                1,
                keepdims=True,
            )
            a = xf @ (w1[r, e].astype(jnp.float32) * w1_s[r, e])
            mid = jax.nn.silu(a[:, :INTER]) * a[:, INTER:]
            out += coeff * (mid @ (w2[r, e].astype(jnp.float32) * w2_s[r, e]))
    return np.asarray(out)


def _torch(a):
    return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)


def _jax(t):
    return jnp.asarray(t.float().numpy(), jnp.bfloat16)


def _max_rel(a, ref):
    return (
        np.linalg.norm(a - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    ).max()


def test_qwen38_flash_shapes_match_decomposed_layer(mesh, monkeypatch):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids = _inputs(0)
    padded = [
        pad_shared_intermediate(_torch(sw1[r]), _torch(sw2[r]), INTER_S_PAD)
        for r in range(N)
    ]
    psw1 = jnp.stack([_jax(p[0]) for p in padded])
    psw2 = jnp.stack([_jax(p[1]) for p in padded])
    assert psw1.shape == (N, H, 2 * INTER_S_PAD) and psw1.dtype == jnp.bfloat16
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())

    def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate, tw, ids):
        return kmod.fused_moe_decode(
            rank.reshape(-1),
            x,
            w1[0],
            w2[0],
            w1_s[0][:, :, None, :],
            w2_s[0][:, :, None, :],
            sw1[0],
            jnp.ones((1, 2 * INTER_S_PAD), jnp.float32),
            sw2[0],
            jnp.ones((1, H), jnp.float32),
            gate,
            tw,
            ids,
            n_ranks=N,
            expert_start=rank.reshape(()) * E,
            interpret=True,
        )[None]

    args = [
        jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh),
        *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
        *(jax.device_put(a, rep) for a in (x, gate, tw, ids)),
    ]
    got = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("d"),) * 7 + (P(),) * 4,
            out_specs=P("d"),
            check_vma=False,
        )
    )(*args)
    got = np.asarray(got.astype(jnp.float32))
    assert got.shape == (N, M, H)
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    ref = _reference(x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids)
    # FP8 routed matmuls and a BF16 all-reduce against f32 math.
    assert _max_rel(got[0], ref) < 0.08
    # Must-fail controls: one routed expert missing, or the shared gate dropped.
    assert _max_rel(got[0], _reference(*_inputs(0), skip=int(ids[0, 0]))) > 0.08
    ungated = _reference(x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids, gated=False)
    assert _max_rel(got[0], ungated) > 0.08
