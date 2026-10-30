# SPDX-License-Identifier: Apache-2.0
"""(d) The pair LSE combine: qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank).

Checked against the harness's reference merge on adversarial (out, lse)
pairs: one-sided rows (lse -inf on one rank), rows no rank owns (both -inf:
must give exact 0 and no NaN), extreme lse gaps, and NaN gates on padding
rows. The combine is also exercised end to end by the decode/prefill tests.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import (
    DEFAULT_TOL,
    D,
    check_gated,
    combine_plain_average,
    combine_rank0_only,
    gate_output,
    rank_operand,
    reference_combine,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

H, DH, T = 3, 256, 64


def pairs(seed):
    rng = np.random.default_rng(seed)
    out = rng.normal(0, 1, (D, T, D * H, DH)).astype(np.float32)
    lse = rng.normal(5, 3, (D, T, D * H)).astype(np.float32)
    lse[1, 0:8] = -np.inf  # only rank 0 owns keys
    lse[0, 8:16] = -np.inf  # only rank 1 owns keys
    lse[:, 16:24] = -np.inf  # padding / idle rows: no rank owns a key
    lse[0, 24:32] = 80.0  # extreme gaps: exp underflows on the other side
    lse[1, 24:32] = -80.0
    lse[0, 32:40] = 1e-3  # near ties
    lse[1, 32:40] = 1e-3 + 1e-7
    out[np.isneginf(lse)] = 0  # contract: no owned key -> out = 0
    gate = rng.normal(0, 2, (T, D * H, DH)).astype(np.float32)
    gate[16:24] = np.nan  # padding rows carry garbage gates
    return jnp.asarray(out), jnp.asarray(lse), jnp.asarray(gate, jnp.bfloat16)


def full_merge(out, lse, rank):
    """Float64 merge, the mathematical target."""
    o = np.asarray(out, np.float64)[:, :, rank * H : (rank + 1) * H]
    l = np.asarray(lse, np.float64)[:, :, rank * H : (rank + 1) * H]
    m = l.max(0)
    fin = np.isfinite(m)
    w = np.where(fin[None], np.exp(l - np.where(fin, m, 0)[None]), 0)
    return (w[..., None] * o).sum(0) / np.where(w.sum(0) > 0, w.sum(0), 1)[..., None], fin


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("seed", [0, 1])
def test_dcp_combine_matches_reference(seed, rank):
    out, lse, gate = pairs(seed)
    g = gate[:, rank * H : (rank + 1) * H]
    got = qsa_rpa.dcp_combine(out, lse, g, rank_operand(rank))
    assert got.dtype == jnp.bfloat16 and got.shape == (T, H, DH)
    want = reference_combine(out, lse, g, rank_operand(rank))
    merged, live = full_merge(out, lse, rank)
    oracle = gate_output(jnp.asarray(merged, jnp.float32), g, jnp.asarray(live))
    rep = check_gated(got, want, live, what="dcp_combine vs reference")
    check_gated(got, oracle, live, what="dcp_combine vs float64 merge")
    assert np.all(np.asarray(got, np.float32)[16:24] == 0)
    print(rep)


def test_dcp_combine_empty_everywhere_is_zero():
    out = jnp.zeros((D, 8, D * H, DH), jnp.float32)
    lse = jnp.full((D, 8, D * H), -jnp.inf, jnp.float32)
    gate = jnp.full((8, H, DH), jnp.nan, jnp.bfloat16)
    for rank in range(D):
        got = np.asarray(qsa_rpa.dcp_combine(out, lse, gate, rank_operand(rank)), np.float32)
        assert not np.isnan(got).any() and np.all(got == 0)


def test_dcp_combine_picks_own_heads():
    out, lse, gate = pairs(3)
    for rank in range(D):
        g = gate[:, rank * H : (rank + 1) * H]
        got = qsa_rpa.dcp_combine(out, lse, g, rank_operand(rank))
        other = reference_combine(out, lse, g, rank_operand(1 - rank))
        assert not np.array_equal(np.asarray(got), np.asarray(other))


@pytest.mark.parametrize("fault", [combine_rank0_only, combine_plain_average],
                         ids=lambda f: f.__name__)
def test_combine_fault_controls_are_detected(fault):
    out, lse, gate = pairs(0)
    for rank in range(D):
        g = gate[:, rank * H : (rank + 1) * H]
        merged, live = full_merge(out, lse, rank)
        oracle = gate_output(jnp.asarray(merged, jnp.float32), g, jnp.asarray(live))
        with pytest.raises(AssertionError):
            check_gated(fault(out, lse, g, rank_operand(rank)), oracle, live,
                        DEFAULT_TOL, what=fault.__name__)
