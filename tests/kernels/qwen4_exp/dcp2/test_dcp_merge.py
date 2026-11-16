# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DCP_MERGE: the reduce-scatter and projected-share merges
against ``qsa_rpa.dcp_combine`` (the all-gather merge), with the group's
collectives simulated: a reduce-scatter is the sum over ranks of block ``d``,
the proj merge's TP all-reduce the sum over ranks of each share's projection.

Inputs follow test_dcp2_combine.py's adversarial rows (one owner, no owner,
extreme LSE gaps, near ties, NaN gates on dead rows) for DCP 2 and 4, and a
rank that owns no key of a row holds NaN there: the new merges must not read
it, ``dcp_combine`` (which multiplies by a zero weight) is fed zeros instead.

On TPU (``DCP2_HARNESS_DEVICE=tpu``) the same functions run through XLA,
whose ``dcp_combine`` may round its weighted sum differently, so both are
held to a float64 merge there.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from dcp2_harness import DEFAULT_TOL, ON_TPU, check_gated

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

H, DH, T = 3, 256, 96


def operands(size, seed):
    rng = np.random.default_rng(seed)
    out = rng.normal(0, 1, (size, T, size * H, DH)).astype(np.float32)
    lse = rng.normal(5, 3, (size, T, size * H)).astype(np.float32)
    lse[1:, 0:8] = -np.inf  # only rank 0 owns keys
    lse[0, 8:16] = -np.inf  # rank 0 owns none
    lse[:, 16:24] = -np.inf  # padding / idle rows: no rank owns a key
    lse[0, 24:32] = 80.0  # extreme gaps: exp underflows on the other ranks
    lse[1:, 24:32] = -80.0
    lse[:, 32:40] = 1e-3 + 1e-7 * np.arange(size)[:, None, None]  # near ties
    gate = rng.normal(0, 2, (T, size * H, DH)).astype(np.float32)
    gate[16:24] = np.nan  # padding rows carry garbage gates
    dead = np.isneginf(lse)
    clean = out.copy()
    clean[dead] = 0
    garbage = out.copy()
    garbage[dead] = np.nan
    return (
        jnp.asarray(clean),
        jnp.asarray(garbage),
        jnp.asarray(lse),
        jnp.asarray(gate, jnp.bfloat16),
    )


def rank_operand(rank):
    return jnp.full((1,), rank, jnp.int32)


def reference(out, lse, gate, size):
    """dcp_combine on every rank: [S][T, H, D] bf16."""
    return [
        qsa_rpa.dcp_combine(out, lse, gate[:, d * H : (d + 1) * H], rank_operand(d))
        for d in range(size)
    ]


def oracle(out, lse, gate, size):
    """Float64 merge, then dcp_combine's gate rounding: [S][T, H, D] bf16."""
    o, l = np.asarray(out, np.float64), np.asarray(lse, np.float64)
    m = l.max(0)
    fin = np.isfinite(m)
    w = np.where(fin[None], np.exp(l - np.where(fin, m, 0)[None]), 0)
    merged = (w[..., None] * o).sum(0) / np.where(w.sum(0) > 0, w.sum(0), 1)[..., None]
    sig = jax.nn.sigmoid(jnp.asarray(gate, jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32)
    gated = (jnp.asarray(merged, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32) * sig)
    gated = jnp.where(jnp.asarray(fin)[..., None], gated.astype(jnp.bfloat16), 0)
    return [gated[:, d * H : (d + 1) * H] for d in range(size)]


def reduce_scatter_merge(out, lse, gate, size, step):
    addends = [step(out[r], lse[r], lse) for r in range(size)]
    for a in addends:
        assert a.shape == (size, T, H, DH)
    got = []
    for d in range(size):
        summed = sum(a[d].astype(jnp.float32) for a in addends)
        got.append(
            qsa_rpa.dcp_merge_finish(summed, lse, gate[:, d * H : (d + 1) * H], rank_operand(d))
        )
    return got


def live_rows(lse, size):
    fin = np.isfinite(np.asarray(lse).max(0))
    return [fin[:, d * H : (d + 1) * H] for d in range(size)]


# bf16 addends (or bf16 shares, summed after the projection) round each rank's
# term, so the error scales with the terms, not with their sum: a merged
# element that cancels is judged at its head vector's largest element.
LOOSE = replace(DEFAULT_TOL, max_ulp=4.0, ulp_floor_ratio=1.0)
# Report only: XLA on TPU may round dcp_combine's weighted sum differently.
REPORT = replace(DEFAULT_TOL, max_ulp=float("inf"))


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_reduce_scatter_merge_matches_combine(size, seed):
    clean, garbage, lse, gate = operands(size, seed)
    want = reference(clean, lse, gate, size)
    exact = oracle(clean, lse, gate, size)
    live = live_rows(lse, size)
    got = reduce_scatter_merge(garbage, lse, gate, size, qsa_rpa.dcp_merge_scale_step)
    reps = []
    for d in range(size):
        assert got[d].dtype == jnp.bfloat16 and got[d].shape == (T, H, DH)
        # Only the order of the f32 sum over ranks differs from dcp_combine.
        reps.append(
            (
                check_gated(
                    got[d], want[d], live[d], REPORT if ON_TPU else DEFAULT_TOL,
                    what="rs vs dcp_combine",
                )["max_ulp"],
                check_gated(got[d], exact[d], live[d], what="rs vs float64")["max_ulp"],
                check_gated(want[d], exact[d], live[d], REPORT, what="combine vs float64")[
                    "max_ulp"
                ],
            )
        )
    print(f"rs size {size}: max ulps (vs combine, rs vs f64, combine vs f64) "
          f"{np.max(reps, axis=0).tolist()}")


@pytest.mark.parametrize("size", [2, 4])
def test_reduce_scatter_merge_bf16_addends(size):
    clean, garbage, lse, gate = operands(size, 2)
    want = reference(clean, lse, gate, size)
    live = live_rows(lse, size)
    got = reduce_scatter_merge(garbage, lse, gate, size, qsa_rpa.dcp_merge_scale_bf16_step)
    worst = max(
        check_gated(got[d], want[d], live[d], LOOSE, what="rs_bf16")["max_ulp"]
        for d in range(size)
    )
    print(f"rs_bf16 size {size}: max {worst} ulps vs combine")


def test_reduce_scatter_merge_needs_every_rank():
    size = 4
    clean, garbage, lse, gate = operands(size, 3)
    want = reference(clean, lse, gate, size)
    addends = [qsa_rpa.dcp_merge_scale(garbage[r], lse[r], lse) for r in range(size)]
    summed = sum(addends[r][0] for r in range(size - 1))  # one rank missing
    got = qsa_rpa.dcp_merge_finish(summed, lse, gate[:, :H], rank_operand(0))
    with pytest.raises(AssertionError):
        check_gated(got, want[0], live_rows(lse, size)[0], LOOSE)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_projected_shares_match_combine_then_projection(size, seed):
    clean, garbage, lse, gate = operands(size, seed)
    want = reference(clean, lse, gate, size)
    live = live_rows(lse, size)
    hidden = 128
    rng = np.random.default_rng(seed + 10)
    weight = jnp.asarray(rng.normal(0, 0.05, (hidden, size * H * DH)), jnp.bfloat16)
    # The served path: each rank's o_proj over its own heads, summed over TP.
    rows = jnp.concatenate([w.reshape(T, H * DH) for w in want], axis=-1)
    hi = jax.lax.Precision.HIGHEST
    expect = np.asarray(
        jnp.dot(rows, weight.T, precision=hi, preferred_element_type=jnp.float32), np.float64
    )
    shares = [qsa_rpa.dcp_merge_proj_share(garbage[r], lse[r], lse, gate) for r in range(size)]
    proj = np.zeros((T, hidden), np.float64)
    for sh in shares:
        assert sh.dtype == jnp.bfloat16 and sh.shape == (T, size * H, DH)
        assert not np.isnan(np.asarray(sh, np.float32)).any()
        proj += np.asarray(
            jnp.dot(sh.reshape(T, -1), weight.T, precision=hi, preferred_element_type=jnp.float32),
            np.float64,
        )
    assert np.all(proj[16:24] == 0)
    scale = np.abs(expect).max(-1, keepdims=True)
    rel = np.abs(proj - expect) / np.where(scale > 0, scale, 1)
    assert rel.max() <= 2.0**-7, rel.max()
    # The group's shares of a rank's heads add up to its gated merge.
    worst = 0.0
    for d in range(size):
        heads = sum(np.asarray(sh[:, d * H : (d + 1) * H], np.float32) for sh in shares)
        worst = max(worst, check_gated(heads, want[d], live[d], LOOSE, what="proj")["max_ulp"])
    print(f"proj size {size}: projection max rel {rel.max():.2e} of the row's largest output; "
          f"summed shares max {worst} ulps vs combine")


def test_check_stats_report_agreement():
    size = 4
    clean, _, lse, gate = operands(size, 4)
    want = reference(clean, lse, gate, size)[1]
    same = np.asarray(qsa_rpa.dcp_merge_check_stats(want, want, lse))
    assert same[0] == 0 and same[3] == 0 and same[4] == 0 and same[6] == 1
    assert same[5] < 1e-5
    off = want.at[0, 0, 0].add(jnp.bfloat16(1.0))
    stats = np.asarray(qsa_rpa.dcp_merge_check_stats(off, want, lse))
    assert stats[0] > 0.5 and stats[3] == pytest.approx(1 / want.size)
    nan = np.asarray(qsa_rpa.dcp_merge_check_stats(want.at[1, 0, 0].set(jnp.nan), want, lse))
    assert nan[4] == 1


def test_jit_compiles():
    size = 4
    clean, garbage, lse, gate = operands(size, 5)
    a = jax.jit(qsa_rpa.dcp_merge_scale_step)(garbage[0], lse[0], lse)
    b = jax.jit(qsa_rpa.dcp_merge_finish_step)(a[0], lse, gate[:, :H], rank_operand(0))
    c = jax.jit(qsa_rpa.dcp_merge_proj_share_step)(garbage[0], lse[0], lse, gate)
    assert a.shape == (size, T, H, DH) and b.shape == (T, H, DH) and c.shape == gate.shape


@pytest.mark.skipif(not ON_TPU, reason="needs TPU devices (DCP2_HARNESS_DEVICE=tpu)")
@pytest.mark.parametrize("tokens", [192, 384])
def test_group_collectives_on_tpu(tokens):
    """The served shape (DCP 4, 12 heads of 256, 192 / 384 rows) through real
    collectives: all-gather + dcp_combine against the LSE all-gather, the
    weighted reduce-scatter (f32 and bf16 addends) and dcp_merge_finish."""
    # Counted here, not in the skip mark: touching the backend while
    # collecting claims the TPU for the pytest process.
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    import inspect
    from functools import partial

    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    try:
        from jax import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map
    params = inspect.signature(shard_map).parameters
    unchecked = {"check_vma": False} if "check_vma" in params else {"check_rep": False}

    size = 4
    global T
    saved, T = T, tokens
    try:
        clean, garbage, lse, gate = operands(size, 7)
    finally:
        T = saved
    exact = oracle(clean, lse, gate, size)
    live = live_rows(lse, size)
    gate_own = jnp.stack([gate[:, d * H : (d + 1) * H] for d in range(size)])
    mesh = Mesh(np.array(jax.devices()[:size]), ("d",))
    spec = (P("d"), P("d"), P("d"))

    def rank():
        return jax.lax.axis_index("d").astype(jnp.int32).reshape(1)

    @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=P("d"), **unchecked)
    def gathered(out, lse_r, g):
        out_pair = jax.lax.all_gather(out[0], "d")
        lse_pair = jax.lax.all_gather(lse_r[0], "d")
        return qsa_rpa.dcp_combine(out_pair, lse_pair, g[0], rank())[None]

    def scattered(step, scatter):
        @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=P("d"), **unchecked)
        def run(out, lse_r, g):
            lse_all = jax.lax.all_gather(lse_r[0], "d")
            addend = step(out[0], lse_r[0], lse_all)
            if scatter:
                summed = jax.lax.psum_scatter(addend, "d", scatter_dimension=0, tiled=False)
            else:
                summed = jax.lax.psum(addend, "d")[jax.lax.axis_index("d")]
            return qsa_rpa.dcp_merge_finish(summed, lse_all, g[0], rank())[None]

        return run

    old = jax.jit(gathered)(clean, lse, gate_own)
    # Under the served LIBTPU_INIT_ARGS (async collective flags) a standalone
    # JAX reduce-scatter can fail with "Unsupported async call"; the network
    # all-reduce then stands in for it (the engine's reduce-scatter itself is
    # covered by VLLM_TPU_QSA_DCP_MERGE_CHECK).
    collective = "psum_scatter"
    try:
        rs = jax.jit(scattered(qsa_rpa.dcp_merge_scale_step, True))(garbage, lse, gate_own)
        rs16 = jax.jit(scattered(qsa_rpa.dcp_merge_scale_bf16_step, True))(garbage, lse, gate_own)
    except jax.errors.JaxRuntimeError as e:
        if "Unsupported async call" not in str(e):
            raise
        collective = "psum"
        rs = jax.jit(scattered(qsa_rpa.dcp_merge_scale_step, False))(garbage, lse, gate_own)
        rs16 = jax.jit(scattered(qsa_rpa.dcp_merge_scale_bf16_step, False))(garbage, lse, gate_own)
    rep = {"collective": collective, "rs_vs_old": 0.0, "rs_vs_f64": 0.0, "old_vs_f64": 0.0,
           "rs_bf16_vs_old": 0.0}
    for d in range(size):
        rep["rs_vs_old"] = max(rep["rs_vs_old"], check_gated(rs[d], old[d], live[d], REPORT)["max_ulp"])
        rep["rs_vs_f64"] = max(rep["rs_vs_f64"], check_gated(rs[d], exact[d], live[d])["max_ulp"])
        rep["old_vs_f64"] = max(
            rep["old_vs_f64"], check_gated(old[d], exact[d], live[d], REPORT)["max_ulp"]
        )
        rep["rs_bf16_vs_old"] = max(
            rep["rs_bf16_vs_old"], check_gated(rs16[d], old[d], live[d], LOOSE)["max_ulp"]
        )
        diff = np.abs(np.asarray(rs[d], np.float32) - np.asarray(old[d], np.float32))
        rep[f"max_abs_rank{d}"] = float(diff.max())
    print(f"collectives {tokens} rows: {rep}")


class _FakePair:
    """all_gather over a simulated group: concatenates every rank's tensor."""

    def __init__(self, per_rank):
        self.per_rank = per_rank

    def all_gather(self, x, dim):
        return torch.cat(self.per_rank, dim=dim)


@pytest.mark.parametrize("rank", [0, 2, 3])
def test_pair_qgate_local_group_matches_the_all_gather(rank):
    """VLLM_TPU_QSA_DCP_LOCAL_QGATE: the locally projected group rows give this
    rank's own block and the same group rows as the all-gather path."""
    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import _pair_qgate

    size, tokens, width = 4, 6, 3 * 2 * 8
    per_rank = [torch.randn(tokens, width).to(torch.bfloat16) for _ in range(size)]
    group = torch.cat(per_rank, dim=-1)
    dcp_rank = torch.full((1,), rank, dtype=torch.int32)
    own, gathered, extra = _pair_qgate(
        per_rank[rank], _FakePair(per_rank), dcp_rank, size, False, {}, None, 8
    )
    assert extra is None
    assert torch.equal(own, per_rank[rank]) and torch.equal(gathered, group)
    own_l, group_l, extra_l = _pair_qgate(
        group, _FakePair(per_rank), dcp_rank, size, True, {"qgate_check": group}, None, 8
    )
    assert extra_l is None
    assert torch.equal(own_l, per_rank[rank]) and torch.equal(group_l, group)


def test_qgate_check_stats_report_agreement():
    rng = np.random.default_rng(7)
    old = jnp.asarray(rng.normal(0, 1, (6, 4 * 3 * 16)), jnp.bfloat16)
    same = np.asarray(qsa_rpa.dcp_qgate_check_stats(old, old, 8))
    assert same[0] == 0 and same[3] == 0 and same[4] == 0 and same[6] == 1
    off = old.at[2, 5].add(jnp.bfloat16(1.0))
    stats = np.asarray(qsa_rpa.dcp_qgate_check_stats(off, old, 8))
    assert stats[0] > 0.5 and stats[3] == pytest.approx(1 / old.size) and stats[2] > 0
