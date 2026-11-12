# SPDX-License-Identifier: Apache-2.0
"""DCP-2 union-walk prefill (VLLM_TPU_QSA_PREFILL_UNION_DCP).

``qsa_union_dense.union_walk_prefill(dcp=..., return_lse=True)`` walks only the
rank's owned selected groups (``dcp_local_selection`` renumbers them into the
rank's owned group space, where the DCP table addresses like a replicated one)
and returns the (out f32 ungated, lse) partials of the DCP block-sparse kernel.

CPU (this VM): the owned-space renumbering against the contract's owner rule,
and trace-level checks of the routing (the flag sends DCP prefill to the
union walk; with it off every jaxpr is unchanged). TPU
(``DCP2_HARNESS_DEVICE=tpu``, also ``python3 test_dcp2_union_walk.py``): per
rank (out, lse) vs the f32 reference and vs the DCP block-sparse kernel,
combined gated output vs the replicated route, fallbacks bitwise equal to the
DCP block-sparse call, and (``--bench``) per-layer timings at an 8K chunk.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    import pytest
except ImportError:  # `python3 file.py` on an image without pytest (exos: no pip network)
    import contextlib
    import types

    def _identity(*args, **kwargs):
        return lambda fn: fn

    @contextlib.contextmanager
    def _raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"did not raise {exc}")

    pytest = types.ModuleType("pytest")
    pytest.mark = types.SimpleNamespace(skipif=_identity, parametrize=_identity)
    pytest.raises = _raises
    pytest.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else _identity())
    sys.modules["pytest"] = pytest  # for test_dcp2_prefill's import in check_union_case

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcp2_harness import (  # noqa: E402
    D,
    ON_TPU,
    TPU_TOL,
    Tolerance,
    check_partial,
    diff_report,
    make_case,
    owner_of_token,
    rank_operand,
    reference_attend,
    reference_metadata,
    selected_token_ids,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

B = 512
RATIO = 4
# The served union walk (c16 za7h args): group 128, KV tile 1024, cap 12288,
# min context 32768, lean. UNION_DCP on top.
UNION_ENV = {
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": "128",
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": "1024",
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": "12288",
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "32768",
    "VLLM_TPU_QSA_PREFILL_LEAN": "1",
    "VLLM_TPU_QSA_PREFILL_PREFIX_PAGES": "1",
    # Correctness cases force the walk (random selections are dense); the
    # bench sets the served alpha / precheck explicitly.
    "VLLM_TPU_QSA_PREFILL_UNION_DCP_ALPHA": "0",
    "VLLM_TPU_QSA_PREFILL_UNION_DCP_PRECHECK": "0",
    "VLLM_TPU_QSA_PREFILL_UNION_DCP_COST": "off",
}


def _set_env(setenv, union_dcp, extra=None):
    for k, v in UNION_ENV.items():
        setenv(k, v)
    setenv("VLLM_TPU_QSA_PREFILL_UNION_DCP", "1" if union_dcp else "0")
    for k, v in (extra or {}).items():
        setenv(k, v)


# ----------------------------------------------------------------------------
# CPU: owned-space renumbering


def _owned_visible_tokens(blocks, lengths, rank, page_size=B, ratio=RATIO):
    """Per row, the global tokens the union walk attends after the renumbering.

    The walk's virtual causal mask keeps a selected (or tail) local group's
    key j iff local token ``4 * id + j < local length`` (ranks are monotone in
    ids); local ids map back to global ones by inverting the renumbering."""
    lb, ll = ud.dcp_local_selection(jnp.asarray(blocks), jnp.asarray(lengths), rank=rank_operand(rank),
                                    world=D, page_size=page_size, ratio=ratio)
    lb, ll = np.asarray(lb), np.asarray(ll)
    g_per_page = page_size // ratio

    def to_global(local):
        return ((local // g_per_page) * D + rank) * g_per_page + local % g_per_page

    rows = []
    for i in range(len(lengths)):
        ids = {int(x) for x in lb[i] if x >= 0} | ({int(ll[i]) // ratio} if ll[i] % ratio else set())
        toks = set()
        for local in ids:
            for j in range(ratio):
                if local * ratio + j < ll[i]:
                    toks.add(int(to_global(local)) * ratio + j)
        rows.append(toks)
    return rows


def test_local_selection_matches_owned_tokens():
    rng = np.random.default_rng(0)
    lengths = np.concatenate([
        np.arange(1, 70),  # first page: rank 1 owns nothing
        rng.integers(1, 6000, 300),
        np.array([0, 0, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049]),
    ]).astype(np.int32)
    from dcp2_harness import random_blocks

    blocks = random_blocks(rng, lengths, RATIO, 96)
    expected = selected_token_ids(blocks, lengths, RATIO)
    for rank in range(D):
        got = _owned_visible_tokens(blocks, lengths, rank)
        empty = 0
        for i in range(len(lengths)):
            want = {int(t) for t in expected[i] if t >= 0 and owner_of_token(t, B) == rank}
            assert got[i] == want, (rank, i, int(lengths[i]), sorted(got[i] ^ want)[:8])
            empty += not want and lengths[i] > 0
        _, kv_lens, _, _ = reference_metadata(blocks, lengths, np.zeros((len(lengths), 8), np.int32),
                                              ratio=RATIO, page_size=B, dcp=(rank, D))
        assert sum(len(t) for t in got) == int(kv_lens.sum())
        if rank == 1:
            assert empty >= 69  # live rows owning nothing are covered


# ----------------------------------------------------------------------------
# Cases (TPU sizes; the union walk needs a step whose longest row is >= 32768)

# (prefix, chunk) per request. "mix": a long chunk past min context, an
# unaligned short piece, and a fresh 200-token piece whose rows own nothing on
# rank 1; then NaN padding rows past starts[-1] up to a 128-row multiple.
CASES = {
    "mix_pad": dict(chunks=[(36001, 3000), (700, 333), (0, 200)], budget=1024, pad_to=128),
    "long_4k": dict(chunks=[(40003, 4096)], budget=1024, pad_to=0),
}


def union_case(rng, chunks, *, budget, pad_to, kv_head_offset=0):
    lengths, req, seq = [], [], []
    for r, (prefix, count) in enumerate(chunks):
        lengths += list(range(prefix + 1, prefix + count + 1))
        req += [r] * count
        seq.append(prefix + count)
    real = len(lengths)
    pad = (-real) % pad_to if pad_to else 0
    nan_rows = list(range(real, real + pad))
    lengths += [0] * pad
    req += [len(chunks) - 1] * pad
    case = make_case(rng, row_lengths=lengths, row_req=req, seq_tokens=seq, budget=budget,
                     nan_rows=nan_rows, kv_head_offset=kv_head_offset)
    # Padding rows sit past the last request (the served prefill layout).
    starts = np.zeros(len(chunks) + 1, np.int32)
    starts[1:] = np.cumsum([c for _, c in chunks])
    case.starts = starts
    return case


def prefill_attend(case, half):
    """``_prefill_attention(dcp, return_lse)`` on query heads ``half`` (None: all)."""

    def attend(q, gate, blocks, lengths, table, cache, **kw):
        return qsa_rpa._prefill_attention(
            q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), cache,
            jnp.asarray(case.starts), return_lse=True, **kw)[:2]

    return attend


def run_rank(case, rank, setenv, union_dcp, extra=None):
    _set_env(setenv, union_dcp, extra)
    out, lse = prefill_attend(case, None)(
        case.q, case.gate, case.blocks, case.lengths, case.dcp_rows, case.rank_pools[rank],
        dcp=(rank_operand(rank), D), **case.opts)
    return np.asarray(out), np.asarray(lse)


def _err(x, ref, live):
    """|x - ref| over live rows, relative to each head vector's max |ref|: (max, p99)."""
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    scale = np.maximum(np.abs(ref).max(-1, keepdims=True), 1e-30)
    e = (np.abs(x - ref) / scale)[np.asarray(live, bool)]
    return (float(e.max(initial=0)), float(np.percentile(e, 99)) if e.size else 0.0)


def _worst_rows(x, ref, case, kv_lens, k=4):
    e = np.abs(np.asarray(x, np.float64) - np.asarray(ref, np.float64)).max(axis=(1, 2))
    top = np.argsort(-e)[:k]
    return [(int(i), round(float(e[i]), 5), int(case.lengths[i]), int(case.row_req[i]), int(kv_lens[i]))
            for i in top]


def check_union_case(case, setenv, *, tol=TPU_TOL):
    """Union-walk DCP partials and gated outputs, judged as the union walk is
    accepted without DCP (test_qwen4_exp_union_walk_tp_rank): error vs the f32
    reference within twice the DCP block-sparse kernel's (max and p99), exact
    support (rows owning nothing: 0 / -inf), lse at the harness tolerance, no
    NaN; and the walk must have run (not bitwise the block-sparse output)."""
    reports = {}
    union, sparse = [], []
    for r in range(D):
        dcp = (rank_operand(r), D)
        u = run_rank(case, r, setenv, True)
        s = run_rank(case, r, setenv, False)
        ref_out, ref_lse = reference_attend(case.q, case.gate, case.blocks, case.lengths,
                                            case.dcp_rows, case.rank_pools[r], dcp=dcp, **case.opts)
        _, kv_lens, _, _ = reference_metadata(case.blocks, case.lengths, case.dcp_rows,
                                              ratio=case.ratio, page_size=B, dcp=(r, D))
        loose = Tolerance(rtol=1.0, atol=1.0, lse_atol=1.0)  # support / NaN checks only
        check_partial(*u, ref_out, ref_lse, kv_lens, loose, what=f"union rank {r} vs reference")
        check_partial(*s, ref_out, ref_lse, kv_lens, loose, what=f"block-sparse rank {r} vs reference")
        live = np.broadcast_to((kv_lens > 0)[:, None], np.asarray(ref_lse).shape)
        eu, es = _err(u[0], ref_out, live), _err(s[0], ref_out, live)
        reports[f"err_vs_ref{r}"] = dict(union=eu, block_sparse=es,
                                         worst_union=_worst_rows(u[0], ref_out, case, kv_lens),
                                         worst_sparse=_worst_rows(s[0], ref_out, case, kv_lens))
        assert eu[0] <= 2 * max(es[0], 1e-6) and eu[1] <= 2 * max(es[1], 1e-6), (
            f"rank {r}: union error {eu} > 2x block-sparse {es}: {reports[f'err_vs_ref{r}']}")
        fin = np.isfinite(np.asarray(ref_lse))
        lu = np.abs(np.asarray(u[1]) - np.asarray(ref_lse))
        ls = np.abs(np.asarray(s[1]) - np.asarray(ref_lse))
        lu, ls = np.where(fin, lu, 0).max(1), np.where(fin, ls, 0).max(1)
        top = np.argsort(-lu)[:4]
        reports[f"lse_err{r}"] = dict(
            union=float(lu.max()), block_sparse=float(ls.max()),
            worst_union=[(int(i), round(float(lu[i]), 5), round(float(ls[i]), 5), int(case.lengths[i]),
                          int(case.row_req[i]), int(kv_lens[i])) for i in top])
        assert lu.max() <= max(2 * ls.max(), tol.lse_atol), f"rank {r} lse: {reports[f'lse_err{r}']}"
        reports[f"walk_ran{r}"] = bool((u[0] != s[0]).any())
        reports[f"rows_owning_nothing{r}"] = int(((kv_lens == 0) & (case.lengths > 0)).sum())
        union.append(u)
        sparse.append(s)
    assert any(reports[f"walk_ran{r}"] for r in range(D)), "union walk output equals block-sparse bitwise"
    from dcp2_harness import combine_lse, gate_output
    from test_dcp2_prefill import merge_pair

    full_out, full_lse = reference_attend(case.q, case.gate, case.blocks, case.lengths, case.full_rows,
                                          case.full_pool, **case.opts)
    live_all = np.isfinite(np.asarray(full_lse))
    for name, parts in (("union", union), ("block_sparse", sparse)):
        out_pair = jnp.stack([jnp.asarray(p[0]) for p in parts])
        lse_pair = jnp.stack([jnp.asarray(p[1]) for p in parts])
        merged = np.concatenate([np.asarray(merge_pair(out_pair, lse_pair, r, case.heads)[0]) for r in range(D)],
                                axis=1)
        gated = np.concatenate([np.asarray(qsa_rpa.dcp_combine(
            out_pair, lse_pair, case.gate[:, r * case.heads:(r + 1) * case.heads], rank_operand(r)), np.float32)
            for r in range(D)], axis=1)
        ref_gated = np.asarray(gate_output(full_out, case.gate, jnp.asarray(live_all)), np.float32)
        assert not np.isnan(gated).any() and np.all(gated[~live_all] == 0), name
        np.testing.assert_array_equal(np.isfinite(np.asarray(combine_lse(lse_pair))), live_all)
        reports[f"merged_err_{name}"] = _err(merged, full_out, live_all)
        reports[f"gated_err_{name}"] = _err(gated, ref_gated, live_all)
    for kind in ("merged", "gated"):
        eu, es = reports[f"{kind}_err_union"], reports[f"{kind}_err_block_sparse"]
        assert eu[0] <= 2 * max(es[0], 1e-6) and eu[1] <= 2 * max(es[1], 1e-6), (kind, eu, es)
    return reports


def check_fallback(case, setenv, extra):
    """A forced fallback (cap / min context) returns the DCP block-sparse call bitwise."""
    reports = {}
    for r in range(D):
        u = run_rank(case, r, setenv, True, extra)
        s = run_rank(case, r, setenv, False, extra)
        assert np.array_equal(u[0], s[0]) and np.array_equal(u[1], s[1]), (
            f"rank {r} fallback differs: out {diff_report(u[0], s[0])} lse {diff_report(u[1], s[1])}")
        reports[f"fallback_bitwise{r}"] = True
    return reports


# ----------------------------------------------------------------------------
# CPU: trace-level routing checks


def _trace_case():
    rng = np.random.default_rng(5)
    # 1024 rows of one request on a 40K-token context (no pools: shapes only).
    rows, seq, page_cols = 1024, 41000, -(-41000 // B)
    q = jnp.zeros((rows, 2 * 3, 256), jnp.bfloat16)
    blocks = jnp.asarray(rng.integers(0, seq // RATIO, (rows, 256)), jnp.int32)
    lengths = jnp.arange(seq - rows + 1, seq + 1, dtype=jnp.int32)
    starts = jnp.asarray([0, rows], jnp.int32)
    full = jnp.zeros((rows, page_cols), jnp.int32)
    dcp_table = jnp.zeros((rows, -(-page_cols // D)), jnp.int32)
    cache = jnp.zeros((2 * page_cols + 4, B, 1, 4, 256), jnp.float8_e4m3fn)
    return q, blocks, lengths, starts, full, dcp_table, cache


def _tpu_info(monkeypatch):
    """Trace Mosaic kernel bodies on a CPU host (TPU7x info, as conftest's ``interpret``)."""
    if ON_TPU:
        return
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _jaxpr(dcp, slot=0):
    q, blocks, lengths, starts, full, dcp_table, cache = _trace_case()
    table = full if dcp is None else dcp_table

    def f(q, blocks, lengths, table, cache, starts):
        return qsa_rpa._prefill_attention(
            q, q, blocks, lengths, table, cache, starts, kv_heads=1, ratio=RATIO,
            k_scale=0.375, v_scale=2.5, **({"kv_head_offset": slot} if slot else {}),
            **({"dcp": (rank_operand(1), D), "return_lse": True} if dcp else {}))

    return str(jax.make_jaxpr(f)(q, blocks, lengths, table, cache, starts))


@pytest.mark.parametrize("slot", [0, 1])
def test_flag_routes_dcp_prefill_to_union_walk(monkeypatch, slot):
    _tpu_info(monkeypatch)
    _set_env(monkeypatch.setenv, True)
    on = _jaxpr(True, slot)
    assert "qsa_union_walk_g128_kv1024_lean_fast_pg_lse" in on
    assert (f"_slot{slot}" in on) == bool(slot)
    _set_env(monkeypatch.setenv, False)
    off = _jaxpr(True, slot)
    assert "qsa_union_walk" not in off
    # The served defaults (alpha 0.2, sampled precheck) trace too.
    _set_env(monkeypatch.setenv, True, {"VLLM_TPU_QSA_PREFILL_UNION_DCP_ALPHA": "0.6",
                                        "VLLM_TPU_QSA_PREFILL_UNION_DCP_PRECHECK": "8",
                                        "VLLM_TPU_QSA_PREFILL_UNION_DCP_COST": "0.15,2.31e-5,0.17,7.2e-3"})
    served = _jaxpr(True, slot)
    assert "qsa_union_walk_g128_kv1024_lean_fast_pg_lse" in served and served != on
    # Variants the DCP walk does not support keep the block-sparse kernel.
    _set_env(monkeypatch.setenv, True, {"VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS": "1"})
    assert "qsa_union_walk" not in _jaxpr(True, slot)


def test_flag_leaves_replicated_route_unchanged(monkeypatch):
    _tpu_info(monkeypatch)
    _set_env(monkeypatch.setenv, False)
    off = _jaxpr(None)
    _set_env(monkeypatch.setenv, True)
    on = _jaxpr(None)
    assert on == off
    assert "qsa_union_walk_g128_kv1024_lean_fast_pg" in on and "fast_pg_lse" not in on


# j-9a871b0e (served shape: top-k 512 groups, c32 env, rank 0): (rows, sampled union groups, owned pages
# touched, forced walk ms, block-sparse ms) per selection model; one sampled 1024-row block per step.
SERVED_SWEEP = {
    "8k_corr10": (8192, 27359, 71, 6.508, 5.699), "8k_tight2": (8192, 7081, 71, 2.797, 5.715),
    "8k_random": (8192, 70718, 71, 14.386, 5.693), "8k_hot0.05": (8192, 3906, 71, 2.013, 5.700),
    "8k_hot0.1": (8192, 7577, 71, 2.711, 5.705), "8k_hot0.2": (8192, 14780, 71, 4.165, 5.701),
    "8k_pages0.1": (8192, 7296, 8, 2.704, 2.065), "8k_pages0.2": (8192, 14464, 15, 4.153, 2.561),
    "8k_pages0.36": (8192, 24703, 25, 5.857, 3.079), "4k_corr10": (4096, 23432, 40, 2.834, 1.631),
    "4k_tight2": (4096, 6990, 40, 1.335, 1.643), "4k_random": (4096, 39753, 40, 4.102, 1.644),
    "4k_hot0.1": (4096, 4109, 40, 0.925, 1.645), "4k_pages0.2": (4096, 5248, 7, 1.117, 0.706),
}


def test_default_cost_model_picks_the_faster_kernel():
    """The default VLLM_TPU_QSA_PREFILL_UNION_DCP_COST on the measured served-shape sweep (in-sample)."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import _union_dcp_cost

    assert envs.VLLM_TPU_QSA_PREFILL_UNION_DCP_COST == "0.15,2.31e-5,0.17,7.2e-3"
    walk_fixed, per_group, sparse_fixed, per_page = _union_dcp_cost()
    for name, (rows, union, pages, walk_ms, sparse_ms) in SERVED_SWEEP.items():
        take_walk = walk_fixed + per_group * union < sparse_fixed + per_page * pages
        assert take_walk == (walk_ms < sparse_ms), (name, take_walk, walk_ms, sparse_ms)
        blocks = rows // 1024  # the model's per-block times, summed, within 25% of the measurement
        assert abs(blocks * (walk_fixed + per_group * union) / walk_ms - 1) < 0.25, name
        assert abs(blocks * (sparse_fixed + per_page * pages) / sparse_ms - 1) < 0.25 or rows == 4096, name


# ----------------------------------------------------------------------------
# TPU

tpu = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")


@tpu
@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tpu_union_dcp_matches(monkeypatch, name, slot):
    spec = CASES[name]
    case = union_case(np.random.default_rng(len(name) + slot), spec["chunks"], budget=spec["budget"],
                      pad_to=spec["pad_to"], kv_head_offset=slot)
    print(name, slot, check_union_case(case, monkeypatch.setenv))


@tpu
@pytest.mark.parametrize("which", ["cap", "min_context"])
def test_tpu_union_dcp_fallback_is_block_sparse(monkeypatch, which):
    spec = CASES["mix_pad"]
    case = union_case(np.random.default_rng(7), spec["chunks"], budget=spec["budget"], pad_to=128)
    extra = ({"VLLM_TPU_QSA_PREFILL_UNION_CAP": "64"} if which == "cap"
             else {"VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "1000000"})
    print(which, check_fallback(case, monkeypatch.setenv, extra))


# ----------------------------------------------------------------------------
# Bench: per-layer union-walk DCP vs block-sparse DCP at an 8K chunk


def model_blocks(rng, lengths, budget, model, ratio=RATIO, page_size=B):
    """Selections of one long chunk. "random": uniform over the row's complete
    groups (dense union). "hot<f>": rows draw from one request-wide pool of a
    fraction f of the groups, spread uniformly (union ~ f, every page touched).
    "pages<f>": the pool is every group of a random fraction f of the pages
    (union ~ f, page-local: block-sparse skips the other pages)."""
    lengths = np.asarray(lengths)
    complete_max = int(lengths.max()) // ratio
    if model == "random":
        from dcp2_harness import random_blocks

        return random_blocks(rng, lengths, ratio, budget)
    frac = float(model.lstrip("hotpages"))
    if model.startswith("hot"):
        pool = np.sort(rng.choice(complete_max, max(1, int(frac * complete_max)), replace=False))
    else:
        per = page_size // ratio
        pages = -(-complete_max // per)
        keep = rng.choice(pages, max(1, int(round(frac * pages))), replace=False)
        pool = np.sort((keep[:, None] * per + np.arange(per)[None]).reshape(-1))
        pool = pool[pool < complete_max]
    blocks = np.full((len(lengths), budget), -1, np.int32)
    for i, n in enumerate(lengths):
        cand = pool[: np.searchsorted(pool, int(n) // ratio)]
        count = min(len(cand), budget)
        if count:
            blocks[i, :count] = rng.choice(cand, count, replace=False)
    return blocks


def _time(fn, iters=10):
    jax.block_until_ready(fn())
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


ARMS = {
    "block_sparse": {"VLLM_TPU_QSA_PREFILL_UNION_DCP": "0"},
    "walk_forced": {"VLLM_TPU_QSA_PREFILL_UNION_DCP_ALPHA": "0", "VLLM_TPU_QSA_PREFILL_UNION_DCP_PRECHECK": "0",
                    "VLLM_TPU_QSA_PREFILL_UNION_DCP_COST": "off"},
    # The served defaults: precheck 8 with the cost model, alpha 0.6 backstop.
    "served": {"VLLM_TPU_QSA_PREFILL_UNION_DCP_ALPHA": "0.6", "VLLM_TPU_QSA_PREFILL_UNION_DCP_PRECHECK": "8",
               "VLLM_TPU_QSA_PREFILL_UNION_DCP_COST": "0.15,2.31e-5,0.17,7.2e-3"},
    "alpha0.2_pc8": {"VLLM_TPU_QSA_PREFILL_UNION_DCP_ALPHA": "0.2", "VLLM_TPU_QSA_PREFILL_UNION_DCP_PRECHECK": "8",
                     "VLLM_TPU_QSA_PREFILL_UNION_DCP_COST": "off"},
}
MODELS = ("random", "hot0.1", "hot0.2", "hot0.36", "pages0.1", "pages0.2", "pages0.36")


def union_fraction(blocks, lengths, table_columns, rank):
    """Owned union / owned visible per 128-row tile (median, max) for this rank."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    lb, ll = ud.dcp_local_selection(jnp.asarray(blocks), jnp.asarray(lengths), rank=rank_operand(rank),
                                    world=D, page_size=B, ratio=RATIO)
    cap = table_columns * B // RATIO
    _, _, usize, _ = union_bits_plan(lb, ll, group=128, union_cap=cap, groups_cap=cap, ratio=RATIO)
    vis = np.asarray(jnp.max(((ll + RATIO - 1) // RATIO).reshape(-1, 128), axis=1))
    f = np.asarray(usize) / np.maximum(vis, 1)
    return round(float(np.median(f)), 3), round(float(f.max()), 3)


def plan_breakdown(case, rank=0):
    """(2): the DCP walk's per-layer plan pieces, timed alone."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    blocks, lengths = jnp.asarray(case.blocks), jnp.asarray(case.lengths)
    table = jnp.asarray(case.dcp_rows)
    cap = table.shape[1] * B // RATIO
    r = rank_operand(rank)
    local = jax.jit(lambda b, l: ud.dcp_local_selection(b, l, rank=r, world=D, page_size=B, ratio=RATIO))
    lb, ll = local(blocks, lengths)
    plan = jax.jit(lambda b, l: union_bits_plan(b, l, group=128, union_cap=12288, groups_cap=cap, ratio=RATIO))
    rows = blocks.shape[0] // 128
    pick = np.arange(0, rows, 8)
    pre = jax.jit(lambda b, l: union_bits_plan(b.reshape(rows, 128, -1)[pick].reshape(-1, b.shape[1]),
                                               l.reshape(rows, 128)[pick].reshape(-1), group=128,
                                               union_cap=12288, groups_cap=cap, ratio=RATIO)[2])
    # Layer-invariant pieces: per-group page rows and per-word pool bases (identity layout).
    words = -(-cap // 32)
    word_start = jnp.arange(words, dtype=jnp.int32) * (32 * RATIO)
    inv = jax.jit(lambda t: t.reshape(-1, 128, t.shape[1])[:, 0][..., jnp.minimum(word_start // B, t.shape[1] - 1)]
                  * B + word_start % B)
    out = dict(local_selection_ms=_time(lambda: local(blocks, lengths)),
               plan_ms=_time(lambda: plan(lb, ll)),
               precheck_stride8_ms=_time(lambda: pre(lb, ll)),
               layer_invariant_ms=_time(lambda: inv(table)))
    return {k: round(v, 3) for k, v in out.items()}


def bench(chunk=8192, context=74000, budget=2048, models=MODELS, ranks=(0,)):
    results = {}
    case = union_case(np.random.default_rng(11), [(context - chunk, chunk)], budget=budget, pad_to=0)
    for model in models:
        case.blocks = model_blocks(np.random.default_rng(12), case.lengths, budget, model)
        if model == "random":
            results["plan_breakdown"] = plan_breakdown(case)
            print("BENCH plan_breakdown", results["plan_breakdown"], flush=True)
        for r in ranks:
            args = (case.q, case.gate, jnp.asarray(case.blocks), jnp.asarray(case.lengths),
                    jnp.asarray(case.dcp_rows), case.rank_pools[r], jnp.asarray(case.starts))
            row = dict(union_frac_median_max=union_fraction(case.blocks, case.lengths, case.dcp_rows.shape[1], r))
            outs = {}
            for arm, extra in ARMS.items():
                for k, v in UNION_ENV.items():
                    os.environ[k] = v
                os.environ["VLLM_TPU_QSA_PREFILL_UNION_DCP"] = "1"
                for k, v in extra.items():
                    os.environ[k] = v
                fn = jax.jit(lambda *a: qsa_rpa._prefill_attention(
                    *a, dcp=(rank_operand(r), D), return_lse=True, **case.opts)[:2])
                row[arm] = round(_time(lambda: fn(*args)), 3)
                outs[arm] = np.asarray(fn(*args)[0])
            row["served_took_walk"] = bool((outs["served"] != outs["block_sparse"]).any())
            row["walk_vs_bs_max_abs"] = float(np.abs(outs["walk_forced"] - outs["block_sparse"]).max())
            row["walk_speedup"] = round(row["block_sparse"] / row["walk_forced"], 3)
            row["served_speedup"] = round(row["block_sparse"] / row["served"], 3)
            row["best_possible"] = round(row["block_sparse"] / min(row["block_sparse"], row["walk_forced"]), 3)
            results[f"{model}_rank{r}"] = row
            print("BENCH", model, r, row, flush=True)
    return results


if __name__ == "__main__":
    # python3 test_dcp2_union_walk.py [--bench]  (DCP2_HARNESS_DEVICE=tpu for the TPU tests)
    class _Env:
        def setenv(self, k, v):
            os.environ[k] = v

    env = _Env()
    test_local_selection_matches_owned_tokens()
    print("PASS local selection", flush=True)
    failures = 0
    if ON_TPU and "--bench-only" not in sys.argv:
        for name in sorted(CASES):
            for slot in (0, 1):
                try:
                    spec = CASES[name]
                    case = union_case(np.random.default_rng(len(name) + slot), spec["chunks"],
                                      budget=spec["budget"], pad_to=spec["pad_to"], kv_head_offset=slot)
                    print("PASS", name, slot, check_union_case(case, env.setenv), flush=True)
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print("FAIL", name, slot, repr(e)[:4000], flush=True)
        for which in ("cap", "min_context"):
            try:
                spec = CASES["mix_pad"]
                case = union_case(np.random.default_rng(7), spec["chunks"], budget=spec["budget"],
                                  pad_to=128)
                extra = ({"VLLM_TPU_QSA_PREFILL_UNION_CAP": "64"} if which == "cap"
                         else {"VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": "1000000"})
                print("PASS fallback", which, check_fallback(case, env.setenv, extra), flush=True)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print("FAIL fallback", which, repr(e)[:4000], flush=True)
    if ON_TPU:
        if "--bench" in sys.argv or "--bench-only" in sys.argv:
            bench()
            print("BENCH shape 4096-row chunk, 40K context", flush=True)
            bench(chunk=4096, context=40000, models=("random", "hot0.2", "hot0.36", "pages0.1", "pages0.36"))
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)
