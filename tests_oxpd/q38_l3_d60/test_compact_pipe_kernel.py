# SPDX-License-Identifier: Apache-2.0
"""q38 L3 D60 (lever 5c): fused_moe_decode with compact_pipe (pipelined routed-row compaction) is bitwise lever 5b's
compact_rows (CPU, Pallas interpreter), and within 0.1 % of elements of the dense kernel.

Why 5b is the bitwise reference on CPU (M, L3-D60 diagnosis j-3c653c96 and smoke j-6326f1e2, jax 0.11.2): in this image
5b ITSELF differs from dense in the interpreter in a few rows at m 256 / 224 uniform (63 / 3 elements per rank; one row
by up to 32 bf16 ulps, the size of an fp8 rounding flip), with the same rows whether the prime moves, acc1 is recomputed
in-step or the ring is deeper than the units. The cause is NOT localised: the witness test below finds a standalone
XLA:CPU fp8 dot's rows identical at M 16 and M 256, so "the dot depends on M" is not the explanation. On TPU, MDC1
j-0cd4e9dc and MDC2 j-1e59e928 measured 5b bitwise dense (0 mismatches after 1 and 48 calls, every case, both hosts).
compact_pipe keeps 5b's per-row arithmetic on the same gathered rows, so it must equal 5b bit for bit here; the TPU car
compares every arm with dense on device.

Harness ported from DEC-FB-2's tests/decfb/test_fused_moe_decode_compact.py (lever 5b; credit): 8 host devices under
shard_map (the in-kernel all-reduce included), the Qwen3.8-Flash EP layout's awkward per-rank shapes (routed
intermediate 640 = 512 + 128, a BF16 shared expert whose 80-wide TP slice is zero-padded to 128, a gated shared expert,
EP over 8 ranks), the served fold=True prologue. D60's buckets 160 / 192 / 208 / 224 / 256 plus 5b's cases: uniform,
skewed, duplicate experts in a token's top-k, one hot expert past the compact capacity (dense body) with an expert of
exactly R and one of R + 1 rows, every token on one expert, padded rows (tokens < m), and a "few" routing where each
rank has at most two active experts (fewer ring units than ring slots: the loop's look-ahead past the last unit) and
some ranks none. A positive control moves one route weight; the kernel's name carries -cr_<R>p.

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
Q38_D60_SMOKE=1 runs the smoke subset (marked by the case list below).
"""
from __future__ import annotations

import functools
import os
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod  # noqa: E402

N, H, INTER, INTER_S, INTER_S_PAD, E, TOPK, R = 8, 1024, 640, 80, 128, 16, 4, 16
F8 = jnp.float8_e4m3fn
SMOKE = os.environ.get("Q38_D60_SMOKE") == "1"


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _q(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    return jnp.asarray(w / s).astype(F8), s.astype(np.float32)


def _pad_shared(sw1, sw2):
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
    elif case == "few":
        # ranks 0..3 hold two active experts each (expert ids 3 / 9 of each rank), ranks 4..7 none
        pool = np.array([r * E + o for r in range(4) for o in (3, 9)])
        ids = np.stack([rng.choice(pool, TOPK, replace=False) for _ in range(tokens)])
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


def _fn(mesh, compact, pipe):
    def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
        kw = {"compact_pipe": True} if pipe else {}
        return kmod.fused_moe_decode(
            rank, x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :], sw1[0],
            jnp.ones((2 * INTER_S_PAD,), jnp.float32), sw2[0], jnp.ones((H,), jnp.float32), gate_row, tw, ids,
            n_ranks=N, expert_start="rank", fold=True, interpret=True, compact_rows=compact, **kw)[None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"),) * 7 + (P(),) * 4, out_specs=P("d"),
                                 check_vma=False))


def _args(mesh, w, x, tw, ids):
    w1, w1_s, w2, w2_s, psw1, psw2, gate = w
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)
    return [jax.device_put(rank, sh), *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
            *(jax.device_put(a, rep) for a in (x, gate, tw, ids))]


def _run(mesh, w, x, tw, ids, compact, pipe=False):
    out = _fn(mesh, compact, pipe)(*_args(mesh, w, x, tw, ids))
    return np.asarray(out).view(np.uint16)


@functools.lru_cache(maxsize=None)
def _w(seed):
    return _weights(seed)


# (tokens, case, in the smoke subset)
CASES = [(256, "uniform", True), (224, "uniform", True), (192, "uniform", False), (208, "uniform", False),
         (160, "skewed", False), (256, "skewed", False), (250, "dups", False), (256, "hot_boundary", True),
         (128, "all_one", False), (32, "few", True), (64, "few", False)]
RUN = [(t, c) for t, c, s in CASES if s or not SMOKE]


@pytest.mark.parametrize("tokens,case", RUN, ids=[f"m{t}_{c}" for t, c in RUN])
def test_compact_pipe_bitwise_5b_near_dense(mesh, monkeypatch, tokens, case):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w = _w(0)
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(rng.normal(size=(tokens, H)), jnp.bfloat16)
    tw, ids = _routing(case, tokens, 1 + tokens)
    dense = _run(mesh, w, x, tw, ids, None)
    comp = _run(mesh, w, x, tw, ids, R)
    pipe = _run(mesh, w, x, tw, ids, R, pipe=True)
    assert dense.shape == comp.shape == pipe.shape == (N, tokens, H)
    for name, out in (("dense", dense), ("5b", comp), ("pipe", pipe)):
        assert all(np.array_equal(out[0], out[r]) for r in range(N)), f"ranks disagree ({name})"
    bad5b = int((comp != pipe).sum())
    bad_d = int((dense != pipe).sum())
    bad_5d = int((dense != comp).sum())
    print(f"Q38_D60_KERNEL m{tokens}_{case} pipe_vs_5b {bad5b} pipe_vs_dense {bad_d} 5b_vs_dense {bad_5d} of {dense.size}")
    assert bad5b == 0, f"{bad5b} of {dense.size} output elements differ (pipe vs 5b compact_rows)"
    assert bad_d <= 0.001 * dense.size, f"pipe vs dense: {bad_d} of {dense.size} elements differ (> 0.1 %)"
    assert (dense[0] != 0).mean() > 0.9  # a real, nonzero output


def test_positive_control_one_route_weight_changes_output(mesh, monkeypatch):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w = _w(0)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(192, H)), jnp.bfloat16)
    tw, ids = _routing("uniform", 192, 6)
    base = _run(mesh, w, x, tw, ids, R, pipe=True)
    tw2 = tw.at[7, 0].set(tw[7, 0] * 2)
    moved = _run(mesh, w, x, tw2, ids, R, pipe=True)
    print(f"Q38_D60_KERNEL posctl moved {int((base != moved).sum())}")
    assert (base != moved).any()


def test_name_suffix_and_flag_needs_compact_rows(mesh, monkeypatch):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w = _w(0)
    x = jnp.zeros((64, H), jnp.bfloat16)
    tw, ids = _routing("uniform", 64, 2)
    args = _args(mesh, w, x, tw, ids)
    texts = {k: str(jax.make_jaxpr(_fn(mesh, *k))(*args)) for k in ((None, False), (R, False), (R, True))}
    assert "-cr_" not in texts[(None, False)]
    assert f"-cr_{R}" in texts[(R, False)] and f"-cr_{R}p" not in texts[(R, False)]
    assert f"-cr_{R}p" in texts[(R, True)]
    with pytest.raises(ValueError, match="compact_pipe needs compact_rows"):
        jax.make_jaxpr(_fn(mesh, None, True))(*args)


def _conds_before_occ(text, m):
    """Kernel-body conds printed before the compaction tables' occ matmul (the first f32[m,128] dot_general). The
    routed prime is the kernel body's first top-level cond (pl.when(count > 0)); the printer shows a cond's branches
    elsewhere, so the cond eqn itself marks where the prime is issued (M, diagnosis j-3c653c96: 5b first cond after the
    occ matmul, compact_pipe before it, and after it again with the prime moved back)."""
    hit = re.search(rf"f32\[{m},128\] = dot_general", text)
    assert hit is not None, "no occ matmul found in the kernel jaxpr"
    return text[: hit.start()].count("cond[")


def test_pipe_primes_before_the_tables_5b_after(mesh, monkeypatch):
    """5c (a): the routed weight prime precedes the compaction tables in compact_pipe mode and follows them in 5b."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    w = _w(0)
    x = jnp.zeros((64, H), jnp.bfloat16)
    tw, ids = _routing("uniform", 64, 2)
    args = _args(mesh, w, x, tw, ids)
    n5b = _conds_before_occ(str(jax.make_jaxpr(_fn(mesh, R, False))(*args)), 64)
    npipe = _conds_before_occ(str(jax.make_jaxpr(_fn(mesh, R, True))(*args)), 64)
    print(f"Q38_D60_KERNEL conds before the tables: 5b {n5b} pipe {npipe}")
    assert n5b == 0 and npipe >= 1


def test_witness_xla_cpu_dot_rows_depend_on_m():
    """Descriptive witness (asserts only that it ran): does a standalone XLA:CPU dot round the same rows differently as
    M = 16 instead of M = 256? fp8 x fp8 -> f32, K 512, N 1280 (the kernel's first-matmul block). Smoke j-6326f1e2: 0
    of 327,680 elements differ, so this does not explain 5b's interpreter difference from dense."""
    rng = np.random.default_rng(7)
    a = jnp.asarray(rng.normal(size=(256, 512)), jnp.float32).astype(F8)
    b = jnp.asarray(rng.normal(size=(512, 1280)) * 0.05, jnp.float32).astype(F8)
    full = np.asarray(jnp.matmul(a, b, preferred_element_type=jnp.float32))
    diffs = 0
    for lo in range(0, 256, 16):
        part = np.asarray(jnp.matmul(a[lo:lo + 16], b, preferred_element_type=jnp.float32))
        diffs += int((part.view(np.uint32) != full[lo:lo + 16].view(np.uint32)).sum())
    print(f"Q38_D60_WITNESS xla_cpu dot rows M16 vs M256: {diffs} of {full.size} f32 elements differ "
          f"(jax {jax.__version__}, backend {jax.default_backend()})")
