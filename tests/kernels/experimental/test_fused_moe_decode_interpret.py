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

import functools

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


@pytest.mark.parametrize("tokens", [M, M - 4])
def test_folded_prologue_is_bitwise_the_unfolded_call(mesh, monkeypatch, tokens):
    """fold=True (routing prologue inside the kernel, operands as stored) returns
    exactly the unfolded call's output, with and without row padding."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids = _inputs(1)
    x, tw, ids = x[:tokens], tw[:tokens], ids[:tokens]
    padded = [
        pad_shared_intermediate(_torch(sw1[r]), _torch(sw2[r]), INTER_S_PAD)
        for r in range(N)
    ]
    psw1 = jnp.stack([_jax(p[0]) for p in padded])
    psw2 = jnp.stack([_jax(p[1]) for p in padded])
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    gate_row = gate.reshape(1, H)  # as the linear stores it

    def body(fold, rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
        common = (
            x,
            w1[0],
            w2[0],
            w1_s[0][:, :, None, :],
            w2_s[0][:, :, None, :],
            sw1[0],
        )
        if fold:
            return kmod.fused_moe_decode(
                rank,
                *common,
                jnp.ones((2 * INTER_S_PAD,), jnp.float32),
                sw2[0],
                jnp.ones((H,), jnp.float32),
                gate_row,
                tw,
                ids,
                n_ranks=N,
                expert_start="rank",
                fold=True,
                interpret=True,
            )[None]
        return kmod.fused_moe_decode(
            rank[0, :1],
            *common,
            jnp.ones((1, 2 * INTER_S_PAD), jnp.float32),
            sw2[0],
            jnp.ones((1, H), jnp.float32),
            gate_row.reshape(-1, 1),
            tw,
            ids,
            n_ranks=N,
            expert_start=rank[0, 1],
            interpret=True,
        )[None]

    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)
    args = [
        jax.device_put(rank, sh),
        *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
        *(jax.device_put(a, rep) for a in (x, gate_row, tw, ids)),
    ]
    outs = [
        np.asarray(
            jax.jit(
                jax.shard_map(
                    functools.partial(body, fold),
                    mesh=mesh,
                    in_specs=(P("d"),) * 7 + (P(),) * 4,
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(*args).view(jnp.uint16)
        )
        for fold in (False, True)
    ]
    np.testing.assert_array_equal(outs[1], outs[0])



def _routing_cases(rng, m, real):
    """(name, ids, tw) for the routed-mode checks, rows past `real` padded the
    way TPU_MOE_SKIP_PADDED_TOKENS leaves them (route weights 0): random
    routes; one expert (global 5) owned by a single row; rank 7 with no
    active expert."""
    ids = np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(m)])
    tw = rng.dirichlet(np.ones(TOPK), size=m)
    tw[real:] = 0
    one = np.where(ids == 5, 6, ids)
    one[3, 0] = 5
    idle = ids % (N * E - E)  # global experts 0 .. 27: rank 7 owns none
    return [("random", ids, tw), ("one-row expert", one, tw), ("idle rank", idle, tw)]


def _routed_outputs(mesh, tokens, real, routed_list):
    """Each routed setting's output, as uint16, per routing case."""
    rng = np.random.default_rng(7)
    _, w1, w1_s, w2, w2_s, sw1, sw2, gate, _, _ = _inputs(2)
    x = jnp.asarray(rng.normal(size=(tokens, H)), jnp.bfloat16)
    padded = [
        pad_shared_intermediate(_torch(sw1[r]), _torch(sw2[r]), INTER_S_PAD)
        for r in range(N)
    ]
    psw1 = jnp.stack([_jax(p[0]) for p in padded])
    psw2 = jnp.stack([_jax(p[1]) for p in padded])
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)

    def body(routed, rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
        return kmod.fused_moe_decode(
            rank,
            x,
            w1[0],
            w2[0],
            w1_s[0][:, :, None, :],
            w2_s[0][:, :, None, :],
            sw1[0],
            jnp.ones((2 * INTER_S_PAD,), jnp.float32),
            sw2[0],
            jnp.ones((H,), jnp.float32),
            gate_row,
            tw,
            ids,
            n_ranks=N,
            expert_start="rank",
            fold=True,
            interpret=True,
            routed=routed,
        )[None]

    out = {}
    for name, ids, tw in _routing_cases(rng, tokens, real):
        args = [
            jax.device_put(rank, sh),
            *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
            *(
                jax.device_put(a, rep)
                for a in (
                    x,
                    gate.reshape(1, H),
                    jnp.asarray(tw, jnp.bfloat16),
                    jnp.asarray(ids, jnp.int32),
                )
            ),
        ]
        out[name] = [
            np.asarray(
                jax.jit(
                    jax.shard_map(
                        functools.partial(body, routed),
                        mesh=mesh,
                        in_specs=(P("d"),) * 7 + (P(),) * 4,
                        out_specs=P("d"),
                        check_vma=False,
                    )
                )(*args).view(jnp.uint16)
            )
            for routed in routed_list
        ]
    return out


@pytest.mark.parametrize("tokens, real", [(64, 64), (64, 51), (57, 57)])
@pytest.mark.parametrize("block", [1, 8, 16])
def test_routed_is_bitwise_the_dense_call(mesh, monkeypatch, tokens, real, block):
    """routed=(block, 64) adds each expert's routed blocks into the same rows in
    the same expert order as the dense call, so the output is the dense output
    bit for bit: zero-weight padded rows, rows padded by the kernel (57 tokens),
    an expert with one row, a rank with no active expert. One tile holds every
    block here, so the matmuls see the dense call's row count; the CPU's dot
    may round a row differently at another row count (see the multi-tile test),
    and the TPU kernel bench checks bit-identity there."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    for name, (dense, routed) in _routed_outputs(
        mesh, tokens, real, [None, (block, 64)]
    ).items():
        np.testing.assert_array_equal(routed, dense, err_msg=name)


def test_routed_multi_tile_matches_dense(mesh, monkeypatch):
    """Tiles of 32 rows, so experts that own rows in all 8 blocks run two
    tiles on one ring slot. Must-fail control: a wrong block or coefficient
    shifts whole rows by O(1); the CPU's dot at 32 rows rounds a few
    elements of a row differently from 64 rows, within a bf16 ulp."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    for name, (dense, routed) in _routed_outputs(mesh, 64, 51, [None, (8, 32)]).items():
        d = np.asarray(dense.view(jnp.bfloat16), np.float32)
        r = np.asarray(routed.view(jnp.bfloat16), np.float32)
        scale = np.abs(d).max(axis=-1, keepdims=True)
        assert np.all(np.abs(r - d) <= scale * 2.0**-7), name


def test_routed_block_flags_skip_padded_rows():
    """routed_block_flags flags block b for expert e exactly when a row of b
    has a nonzero coefficient for e; rows with all route weights 0 (padding
    under TPU_MOE_SKIP_PADDED_TOKENS) set nothing, so a block of padded rows
    only is never scheduled for any expert."""
    rng = np.random.default_rng(3)
    m, real, lanes = 208, 181, 128
    coeff = np.zeros((m, lanes), np.float32)
    for t in range(real):
        coeff[t, rng.choice(64, 10, replace=False)] = rng.random(10) + 0.1
    for block in (1, 8, 16):
        flags = np.concatenate(
            [np.asarray(f) for f in kmod.routed_block_flags(jnp.asarray(coeff), block)]
        ).astype(np.int64)
        nblk = m // block
        got = np.array(
            [[(flags[b // 16, e] >> (b % 16)) & 1 for e in range(lanes)] for b in range(nblk)]
        )
        want = (coeff.reshape(nblk, block, lanes) != 0).any(axis=1)
        np.testing.assert_array_equal(got, want.astype(np.int64))
        padded_only = np.arange(nblk) * block >= real
        assert padded_only.any() and not got[padded_only].any()
        assert got[:, 64:].sum() == 0  # lanes past the 64 experts


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("h", [H, 2560])
def test_exact_width_is_bitwise_the_sliced_output(mesh, monkeypatch, fold, h):
    """exact_width=True (lane tiles split evenly over the ranks in both halves of
    the exchange, output [m, h]) equals the padded split sliced to h byte for
    byte. H = 1536: 12 tiles as 2, 2, 2, 2, 1, 1, 1, 1; 2560 (Qwen3.8-Flash):
    20 tiles as 3, 3, 3, 3, 2, 2, 2, 2."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    rng = np.random.default_rng(7)
    w1, w1_s = _q(rng.normal(size=(N, E, h, 2 * INTER)) * 0.02, 2)
    w2, w2_s = _q(rng.normal(size=(N, E, INTER, h)) * 0.02, 2)
    psw1 = jnp.asarray(rng.normal(size=(N, h, 2 * INTER_S_PAD)) * 0.02, jnp.bfloat16)
    psw2 = jnp.asarray(rng.normal(size=(N, INTER_S_PAD, h)) * 0.02, jnp.bfloat16)
    gate_row = jnp.asarray(rng.normal(size=(1, h)) * 0.02, jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(M, h)), jnp.bfloat16)
    ids = jnp.asarray(
        np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(M)]),
        jnp.int32,
    )
    tw = jnp.asarray(rng.dirichlet(np.ones(TOPK), size=M), jnp.bfloat16)
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())

    def body(exact, rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
        common = (x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :])
        if fold:
            return kmod.fused_moe_decode(
                rank,
                *common,
                sw1[0],
                jnp.ones((2 * INTER_S_PAD,), jnp.float32),
                sw2[0],
                jnp.ones((h,), jnp.float32),
                gate_row,
                tw,
                ids,
                n_ranks=N,
                expert_start="rank",
                fold=True,
                interpret=True,
                exact_width=exact,
            )[None]
        return kmod.fused_moe_decode(
            rank[0, :1],
            *common,
            sw1[0],
            jnp.ones((1, 2 * INTER_S_PAD), jnp.float32),
            sw2[0],
            jnp.ones((1, h), jnp.float32),
            gate_row.reshape(-1, 1),
            tw,
            ids,
            n_ranks=N,
            expert_start=rank[0, 1],
            interpret=True,
            exact_width=exact,
        )[None]

    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)
    args = [
        jax.device_put(rank, sh),
        *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
        *(jax.device_put(a, rep) for a in (x, gate_row, tw, ids)),
    ]
    outs = [
        np.asarray(
            jax.jit(
                jax.shard_map(
                    functools.partial(body, exact),
                    mesh=mesh,
                    in_specs=(P("d"),) * 7 + (P(),) * 4,
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(*args).view(jnp.uint16)
        )
        for exact in (False, True)
    ]
    assert outs[0].shape == outs[1].shape == (N, M, h)
    assert np.any(outs[0] != 0)
    np.testing.assert_array_equal(outs[1], outs[0])
