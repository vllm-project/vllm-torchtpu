# SPDX-License-Identifier: Apache-2.0
"""DEC-FB-2 lever 5b: fused_moe_decode with routed-row compaction (compact_rows=16) is bitwise the dense kernel (CPU).

Both run in the Pallas interpreter on 8 host devices (shard_map, the in-kernel all-reduce included), at the
Qwen3.8-Flash EP layout's awkward per-rank shapes (routed intermediate 640 = 512 + 128, a BF16 shared expert whose
80-wide TP slice is zero-padded to 128, a gated shared expert, EP over 8 ranks), at the served token buckets m 256
(B48 / B64) and m 128 (B32), with the served fold=True prologue. Routing: uniform, skewed, duplicate experts in a
token's top-k, one hot expert past the compact capacity (dense fallback), an expert with exactly R and one with R + 1
rows (the boundary), every token on one expert, and padded rows (tokens < m). A positive control moves one route
weight and must change the output.

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod  # noqa: E402

N, H, INTER, INTER_S, INTER_S_PAD, E, TOPK, R = 8, 1024, 640, 80, 128, 16, 4, 16
F8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _q(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    return jnp.asarray(w / s).astype(F8), s.astype(np.float32)


def _pad_shared(sw1, sw2):
    """gate_up [H, 2 i_s] -> [H, 2 * 128] (gate and up halves zero-padded apart), down [i_s, H] -> [128, H]."""
    g, u = sw1[:, :INTER_S], sw1[:, INTER_S:]
    z = np.zeros((H, INTER_S_PAD - INTER_S), np.float32)
    w1p = np.concatenate([g, z, u, z], axis=1)
    w2p = np.concatenate([sw2, np.zeros((INTER_S_PAD - INTER_S, H), np.float32)], axis=0)
    return jnp.asarray(w1p, jnp.bfloat16), jnp.asarray(w2p, jnp.bfloat16)


def _weights(seed):
    rng = np.random.default_rng(seed)
    w1, w1_s = _q(rng.normal(size=(N, E, H, 2 * INTER)) * 0.02, 2)
    w2, w2_s = _q(rng.normal(size=(N, E, INTER, H)) * 0.02, 2)
    sw = [_pad_shared((rng.normal(size=(H, 2 * INTER_S)) * 0.02).astype(np.float32),
                      (rng.normal(size=(INTER_S, H)) * 0.02).astype(np.float32)) for _ in range(N)]
    psw1 = jnp.stack([a for a, _ in sw])
    psw2 = jnp.stack([b for _, b in sw])
    gate = jnp.asarray(rng.normal(size=(1, H)) * 0.02, jnp.bfloat16)
    return w1, w1_s, w2, w2_s, psw1, psw2, gate


def _routing(case, tokens, seed):
    rng = np.random.default_rng(seed)
    total = N * E
    if case == "skewed":
        p = np.exp(-4.0 * np.arange(total) / total)
        p /= p.sum()
        ids = np.stack([rng.choice(total, TOPK, replace=False, p=p) for _ in range(tokens)])
    else:
        ids = np.stack([rng.choice(total, TOPK, replace=False) for _ in range(tokens)])
    if case == "dups":
        ids[::7, 1] = ids[::7, 0]  # the same expert twice in a token's top-k
        ids[3, :] = 5  # every slot of one token on one expert
    if case == "hot_boundary":
        ids[:, 0] = np.where(np.arange(tokens) % 3 == 0, 21, ids[:, 0])  # expert 21 past capacity (dense body)
        ids[ids == 40] = 41  # clear expert 40, then give it exactly R rows and expert 41 R + 1 (boundary)
        ids[ids == 41] = 42
        ids[:R, 1] = 40
        ids[R:2 * R + 1, 2] = 41
    if case == "all_one":
        ids[:, 0] = 0  # every token on expert 0
    tw = rng.dirichlet(np.ones(TOPK), size=tokens)
    return jnp.asarray(tw, jnp.bfloat16), jnp.asarray(ids, jnp.int32)


def _run(mesh, w, x, tw, ids, compact):
    w1, w1_s, w2, w2_s, psw1, psw2, gate = w
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())

    def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
        return kmod.fused_moe_decode(
            rank, x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :], sw1[0],
            jnp.ones((2 * INTER_S_PAD,), jnp.float32), sw2[0], jnp.ones((H,), jnp.float32), gate_row, tw, ids,
            n_ranks=N, expert_start="rank", fold=True, interpret=True, compact_rows=compact)[None]

    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)
    args = [jax.device_put(rank, sh), *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
            *(jax.device_put(a, rep) for a in (x, gate, tw, ids))]
    out = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"),) * 7 + (P(),) * 4, out_specs=P("d"),
                                check_vma=False))(*args)
    return np.asarray(out).view(np.uint16)


@functools.lru_cache(maxsize=None)
def _w(seed):
    return _weights(seed)


CASES = [(256, "uniform"), (256, "skewed"), (250, "dups"), (256, "hot_boundary"), (128, "uniform"),
         (128, "all_one")]


@pytest.mark.parametrize("tokens,case", CASES, ids=[f"m{t}_{c}" for t, c in CASES])
def test_compact_rows_bitwise_dense(mesh, monkeypatch, tokens, case):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w = _w(0)
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(rng.normal(size=(tokens, H)), jnp.bfloat16)
    tw, ids = _routing(case, tokens, 1 + tokens)
    dense = _run(mesh, w, x, tw, ids, None)
    comp = _run(mesh, w, x, tw, ids, R)
    assert dense.shape == (N, tokens, H)
    assert all(np.array_equal(dense[0], dense[r]) for r in range(N)), "ranks disagree (dense)"
    assert all(np.array_equal(comp[0], comp[r]) for r in range(N)), "ranks disagree (compact)"
    bad = int((dense != comp).sum())
    assert bad == 0, f"{bad} of {dense.size} output elements differ"
    assert (dense[0] != 0).mean() > 0.9  # a real, nonzero output


def test_positive_control_one_route_weight_changes_output(mesh, monkeypatch):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w = _w(0)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(128, H)), jnp.bfloat16)
    tw, ids = _routing("uniform", 128, 6)
    base = _run(mesh, w, x, tw, ids, R)
    tw2 = tw.at[7, 0].set(tw[7, 0] * 2)
    moved = _run(mesh, w, x, tw2, ids, R)
    assert (base != moved).any()
