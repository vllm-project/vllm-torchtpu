# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp_prefill against a float64 oracle.

Runs in the Pallas TPU interpreter on CPU (small shapes) and compiled on TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_tp_prefill as mtp

F8 = jnp.float8_e4m3fn


def _on_tpu():
    return jax.devices()[0].platform == "tpu"


def _q(w, axis):
    s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
    return (w / s).astype(F8), s


def make_case(T, E, H, INTER, K, seed=0, skew=False):
    ks = jax.random.split(jax.random.key(seed), 8)
    x = (jax.random.normal(ks[0], (T, H), jnp.float32)).astype(jnp.bfloat16)
    w1, s1 = _q(jax.random.normal(ks[1], (E, H, 2 * INTER)) * 0.05, 1)
    w2, s2 = _q(jax.random.normal(ks[2], (E, INTER, H)) * 0.05, 1)
    w1s, s1s = _q(jax.random.normal(ks[3], (H, 2 * INTER)) * 0.05, 0)
    w2s, s2s = _q(jax.random.normal(ks[4], (INTER, H)) * 0.05, 0)
    wg = (jax.random.normal(ks[5], (H, 1)) * 0.05).astype(jnp.bfloat16)
    logits = jax.random.normal(ks[6], (T, E))
    if skew:
        logits = logits + jnp.linspace(3.0, 0.0, E)[None]
    tw, ti = jax.lax.top_k(jax.nn.softmax(logits, -1), K)
    tw = tw / tw.sum(-1, keepdims=True)
    return dict(
        x=x,
        w1=w1,
        w2=w2,
        s1=s1.reshape(E, 2 * INTER),
        s2=s2.reshape(E, H),
        w1s=w1s,
        w2s=w2s,
        s1s=s1s.reshape(-1),
        s2s=s2s.reshape(-1),
        wg=wg,
        tw=tw.astype(jnp.float32),
        ti=ti.astype(jnp.int32),
    )


def run_kernel(c, n_valid=None, **kw):
    return mtp.moe_tp_prefill(
        c["x"],
        c["w1"],
        c["w2"],
        c["s1"],
        c["s2"],
        c["w1s"],
        c["w2s"],
        c["s1s"],
        c["s2s"],
        c["wg"],
        c["tw"],
        c["ti"],
        n_valid=n_valid,
        interpret=not _on_tpu(),
        **kw,
    )


def _f64(a):
    return np.asarray(jnp.asarray(a).astype(jnp.float32), np.float64)


def oracle(c, n_valid=None):
    f = _f64
    x = f(c["x"])
    E = c["w1"].shape[0]

    def ffn(xx, w1, w2):
        h = xx @ w1
        i = w1.shape[1] // 2
        return (h[:, :i] / (1 + np.exp(-h[:, :i])) * h[:, i:]) @ w2

    out = np.zeros_like(x)
    tw, ti = f(c["tw"]), np.asarray(c["ti"])
    for e in range(E):
        w1 = f(c["w1"][e]) * f(c["s1"][e])[None]
        w2 = f(c["w2"][e]) * f(c["s2"][e])[None]
        wsel = np.sum(np.where(ti == e, tw, 0.0), -1, keepdims=True)
        out += wsel * ffn(x, w1, w2)
    g = 1 / (1 + np.exp(-(x @ f(c["wg"]))))
    out += g * ffn(x, f(c["w1s"]) * f(c["s1s"])[None], f(c["w2s"]) * f(c["s2s"])[None])
    if n_valid is not None:
        out[n_valid:] = 0
    return out


def row_err(y, ref):
    return float(
        np.max(
            np.linalg.norm(y - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        )
    )


CASES = [  # (T, E, H, INTER, K, skew, n_valid)
    (200, 8, 256, 128, 3, False, None),
    (300, 16, 256, 128, 4, True, None),
    (256, 8, 256, 128, 3, False, 131),
]


@pytest.mark.parametrize("T,E,H,INTER,K,skew,n_valid", CASES)
def test_matches_oracle(T, E, H, INTER, K, skew, n_valid):
    c = make_case(T, E, H, INTER, K, skew=skew)
    y = np.asarray(run_kernel(c, n_valid=n_valid).astype(jnp.float32))
    nv = T if n_valid is None else n_valid
    # fp8 activations: a few percent per row against exact math.
    assert row_err(y[:nv], oracle(c, n_valid)[:nv]) < 8e-2
    assert np.all(y[nv:] == 0)


def test_fault_dropped_expert_is_caught():
    """The bound must see a single expert's rows going missing."""
    c = make_case(200, 8, 256, 128, 3)
    y = np.asarray(run_kernel(c).astype(jnp.float32))
    broken = dict(c, w2=c["w2"].at[3].set(jnp.zeros_like(c["w2"][3])))
    assert row_err(y, oracle(broken)) > 8e-2


def test_prequant_and_chunked_scatter_ranks():
    """Pre-quantized x + pre-computed shared_gate + chunked combine with
    single reduce_fn must match the standard moe_tp_prefill output row-for-row."""
    T, n_ranks, n_chunks = 256, 4, 2
    c = make_case(T, 8, 256, 128, 3)
    ref = np.asarray(run_kernel(c, combine_chunks=1).astype(jnp.float32))

    x_q, x_s = mtp.rowquant_rows(c["x"])
    gate = jax.nn.sigmoid(jnp.dot(c["x"], c["wg"], preferred_element_type=jnp.float32))
    rank_outs = []
    for r in range(n_ranks):
        out_r = run_kernel(
            c,
            combine_chunks=n_chunks,
            reduce_fn=lambda y, r=r: y[
                r * (y.shape[0] // n_ranks) : (r + 1) * (y.shape[0] // n_ranks)
            ],
            x_quantized=(x_q, x_s),
            shared_gate=gate,
            scatter_ranks=n_ranks,
        )
        rank_outs.append(np.asarray(out_r.astype(jnp.float32)))
    got = np.concatenate(rank_outs, axis=0)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("T,E,K,n_valid", [(256, 16, 4, 131), (8192, 512, 10, None)])
def test_routing_plan_matches_reference(T, E, K, n_valid):
    """Packed sort + searchsorted routing_plan matches the 3-array sort +
    [E, R] broadcast reference across both <=64K and >64K route regimes."""
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    topk_ids = jax.random.randint(k1, (T, K), 0, E, dtype=jnp.int32)
    row_scale = jax.random.uniform(k2, (T,), dtype=jnp.float32)
    route_valid = jax.random.bernoulli(k3, 0.9, (T, K))

    got = mtp.routing_plan(
        topk_ids, E, row_scale, route_valid=route_valid, n_valid=n_valid
    )

    n_routes = T * K
    e_flat = topk_ids.reshape(-1).astype(jnp.int32)
    iota = jnp.arange(n_routes, dtype=jnp.int32)
    if n_valid is not None:
        e_flat = jnp.where(iota // K < n_valid, e_flat, E)
    e_flat = jnp.where(route_valid.reshape(-1), e_flat, E)
    s_rep = jnp.broadcast_to(row_scale[:, None], (T, K)).reshape(-1)
    _, order, s_sorted = jax.lax.sort((e_flat, iota, s_rep), num_keys=1, is_stable=True)
    _, pos = jax.lax.sort((order, iota), num_keys=1, is_stable=False)
    counts = jnp.sum(
        (e_flat[None, :] == jnp.arange(E, dtype=jnp.int32)[:, None]).astype(jnp.int32),
        axis=1,
    )
    base = jnp.cumsum(counts) - counts
    active = counts > 0
    (visit_e,) = jnp.nonzero(active, size=E + mtp.VISIT_SLACK, fill_value=E)
    visit_e = visit_e.astype(jnp.int32)
    n_rv = jnp.sum(active).astype(jnp.int32)
    counts_x = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    base_x = jnp.concatenate([base, jnp.zeros((1,), jnp.int32)])
    ref = (
        order // K,
        pos.reshape(T, K),
        s_sorted,
        visit_e,
        base_x[visit_e],
        counts_x[visit_e],
        n_rv,
    )
    for g, r in zip(got, ref):
        assert np.array_equal(np.asarray(g), np.asarray(r))


def test_tp_combine_kernel_matches_reference():
    """build_tp_combine_kernel folds pscl[pos] in SMEM identically to XLA gather."""
    k1, k2, k3, k4 = jax.random.split(jax.random.key(7), 4)
    tc, topk, hidden = 256, 4, 256
    slab = 1024
    part = jax.random.normal(k1, (slab, hidden // mtp.LANES, mtp.LANES)).astype(mtp.FP8)
    pscl = jax.random.uniform(k2, (slab,), dtype=jnp.float32)
    coef = jax.random.uniform(k3, (tc, topk), dtype=jnp.float32)
    pos = jax.random.randint(k4, (tc, topk), 0, slab, dtype=jnp.int32)
    fn = mtp.build_tp_combine_kernel(
        t_local=tc,
        topk=topk,
        hidden=hidden,
        lane_blocks=hidden // mtp.LANES,
        scale_len=slab,
        wire_dtype=mtp.FP8,
        out_dtype=jnp.bfloat16,
        interpret=not _on_tpu(),
    )
    got = np.asarray(fn(part, pscl, coef, pos).astype(jnp.float32))
    rows = part.reshape(slab, hidden)[pos].astype(jnp.float32)
    w = coef * pscl[pos]
    ref = np.asarray(
        jnp.sum(rows * w[..., None], axis=1).astype(jnp.bfloat16).astype(jnp.float32)
    )
    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)
