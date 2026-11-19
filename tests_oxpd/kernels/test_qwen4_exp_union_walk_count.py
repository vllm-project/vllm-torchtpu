# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_UNION_COUNT: the host-side D2 engagement count ("eligible steps").

A step counts when union_walk_prefill was traced for the step's row count (the static dispatch held) and the step's
longest row reaches the min context recorded at trace time. Tracing (make_jaxpr) is enough to record; nothing runs.
"""

import functools
import inspect

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud

PAGE, RATIO = 3200, 4


@pytest.fixture
def v7x(monkeypatch):
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
    monkeypatch.setattr(ud, "_WALK_TRACED", {})
    monkeypatch.setattr(ud, "_ENGAGEMENT", {"steps": 0, "eligible": 0})
    yield


def _trace(tokens, min_context):
    args = (
        jax.ShapeDtypeStruct((tokens, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 24, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 512), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens, 2), jnp.int32),
        jax.ShapeDtypeStruct((4, PAGE, 1, 4, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )
    f = functools.partial(ud.union_walk_prefill, kv_heads=2, ratio=RATIO, group=128, kv_tile=1024,
                          union_cap=12288, min_context=min_context, lean=True)
    jax.make_jaxpr(lambda *a: f(*a)[0])(*args)


def test_eligible_steps_follow_trace_and_min_context(v7x, monkeypatch):
    logged = []
    monkeypatch.setattr(ud, "_engagement_log", lambda msg, *a: logged.append(a))
    _trace(1024, 32768)
    assert ud._WALK_TRACED == {1024: 32768}
    steps = [(1024, 40000), (1024, 1000), (2048, 90000), (1024, 32768)]
    got = [ud.count_union_walk_step(t, longest, every=2) for t, longest in steps]
    # traced bucket + long enough; too short; a bucket the walk was never traced for; exactly the threshold
    assert got == [True, False, False, True]
    assert ud._ENGAGEMENT == {"steps": 4, "eligible": 2}
    assert logged == [(1, 2), (2, 4)]  # every 2 steps: (eligible, steps)


def test_min_context_zero_counts_every_traced_step(v7x):
    _trace(1024, 0)
    assert all(ud.count_union_walk_step(1024, longest, every=0) for longest in (1, 500, 100000))
    assert not ud.count_union_walk_step(4096, 100000, every=0)


# --- the run-time count (count_union_walk_step_runtime): no trace needed, so it holds on compile-cache-hit engines ---

RULE = dict(group=64, min_context=32768, dense_page=False, gathered=False)


@pytest.fixture
def fresh_rt(monkeypatch):
    monkeypatch.setattr(ud, "_WALK_TRACED", {})
    monkeypatch.setattr(ud, "_ENGAGEMENT_RT", {"steps": 0, "eligible": 0, "checked": 0, "disagree": 0})
    logged = []
    monkeypatch.setattr(ud, "_engagement_log", lambda msg, *a: logged.append((msg, a)))
    yield logged


@pytest.mark.parametrize("tokens, longest, kw, want", [
    (6400, 102400, {}, True),
    (1024, 32768, {}, True),                     # both thresholds exactly
    (1024, 32767, {}, False),                    # longest row below min context: the route switch skips the walk
    (960, 102400, {}, False),                    # a multiple of 64 but under 1,024 rows
    (1056, 102400, {}, False),                   # >= 1,024 rows but not a multiple of the group
    (6400, 102400, {"group": 0}, False),         # walk off
    (6400, 102400, {"group": 48}, False),        # a group that does not divide 128
    (6400, 102400, {"dense_page": True}, False),                     # dense pages walk on gathered tiles only
    (6400, 102400, {"dense_page": True, "gathered": True}, True),
    (6400, 0, {"min_context": 0}, True),
])
def test_runtime_rule(tokens, longest, kw, want):
    assert ud.walk_dispatch_eligible(tokens, longest, **{**RULE, **kw}) is want


def test_runtime_count_needs_no_trace(fresh_rt):
    """The compile-cache-hit case: nothing traced, so the trace-based count reads 0 while the run-time one counts."""
    steps = [(6400, 102400), (6400, 1000), (512, 90000), (6400, 40000)]
    got_rt = [ud.count_union_walk_step_runtime(t, n, 2, **RULE) for t, n in steps]
    assert got_rt == [True, False, False, True]
    assert ud._ENGAGEMENT_RT == {"steps": 4, "eligible": 2, "checked": 0, "disagree": 0}
    assert [a[:2] for _, a in fresh_rt] == [(1, 2), (2, 4)]
    assert fresh_rt[-1][1][-1].startswith("not available")
    assert not any(ud.count_union_walk_step(t, n, every=0) for t, n in steps)  # the trace-based count: all 0


def test_runtime_count_checked_against_trace(v7x, fresh_rt):
    """On an engine that traced, every step is checked against the trace record."""
    _trace(1024, 32768)  # group 128
    rule = dict(RULE, group=128)
    for longest in (40000, 1000, 32768):
        ud.count_union_walk_step_runtime(1024, longest, 0, **rule)
    assert ud._ENGAGEMENT_RT == {"steps": 3, "eligible": 2, "checked": 3, "disagree": 0}
    # a rule that drifted from the program (here: another min context) shows as disagreeing steps
    ud.count_union_walk_step_runtime(1024, 20000, 1, **dict(rule, min_context=16384))
    assert ud._ENGAGEMENT_RT["disagree"] == 1
    assert fresh_rt[-1][1][-1] == "1 of 4 checked steps disagree with the trace record"


@pytest.mark.parametrize("rows, heads, kv_heads, dense, gathered", [
    (6400, 24, 2, None, ""), (1024, 24, 2, None, ""), (960, 24, 2, None, ""), (1056, 24, 2, None, ""),
    (6400, 12, 1, None, ""), (6400, 12, 1, 1664, ""), (6400, 12, 1, 1664, "sparsecore"), (960, 12, 1, 1664, "sparsecore"),
])
def test_runtime_rule_matches_prefill_dispatch(v7x, monkeypatch, rows, heads, kv_heads, dense, gathered):
    """The rule's row and dense-page terms against qsa_rpa._prefill_attention's own dispatch."""
    from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp_mod
    from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles, qsa_rpa

    if kv_heads == 1 and "dense_block" not in inspect.signature(qsa_rpa._prefill_attention).parameters:
        # TP2 shapes (12 heads, kv_heads 1, the dense main-KV page) exist only on trees whose dispatch has the TP2 stage and
        # dense_block (T10 / v2). A TP1-only tree (int-6, FR10b PP8) walks only 24 heads x kv_heads 2; its rows are the cases above.
        pytest.skip("TP2 shape on a TP1-only prefill dispatch (no dense_block): not applicable")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_GATHERED", gathered)
    calls = []

    def walk(query, gate, blocks, lengths, table, cache, starts, **kw):
        calls.append("walk")
        return jnp.zeros(query.shape, jnp.bfloat16), cache

    def served(query, gate, blocks, lengths, table, cache, starts, **kw):
        calls.append("served")
        return jnp.zeros(query.shape, jnp.bfloat16), cache

    monkeypatch.setattr(ud, "union_walk_prefill", walk)
    monkeypatch.setattr(bsp_mod, "block_sparse_gqa_attention", served)
    monkeypatch.setattr(prefill_tiles, "get_sparse_prefill_tiles", lambda *a, **k: (256, 1024, None))
    page = PAGE if dense is None else dense
    args = (
        jax.ShapeDtypeStruct((rows, heads, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, heads, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 512), jnp.int32),
        jax.ShapeDtypeStruct((rows,), jnp.int32),
        jax.ShapeDtypeStruct((rows, 2), jnp.int32),
        jax.ShapeDtypeStruct((4, page, 1, 4, 256), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )
    f = functools.partial(qsa_rpa._prefill_attention, kv_heads=kv_heads, ratio=RATIO,
                          **({"dense_block": dense} if dense else {}))
    jax.make_jaxpr(lambda *a: f(*a)[0])(*args)
    want = ud.walk_dispatch_eligible(rows, 10**9, group=64, min_context=32768, dense_page=dense is not None,
                                     gathered=bool(gathered))
    assert calls == ["walk" if want else "served"]
