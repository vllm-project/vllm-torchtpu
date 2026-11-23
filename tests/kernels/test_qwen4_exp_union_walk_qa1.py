# SPDX-License-Identifier: Apache-2.0
"""q38 QA-1 (PREFILL-10 / QA1BUILD): the union walk's bitwise vector diet, all default off.

  mask_view    (VLLM_TPU_QSA_PREFILL_UNION_MASK_VIEW)    the walk tile's select on a [heads, rows, keys] view
  vguard_lean  (VLLM_TPU_QSA_PREFILL_UNION_VGUARD_LEAN)  the value guard keeps only `written` (no isfinite)
  align_gather (VLLM_TPU_QSA_PREFILL_UNION_ALIGN_GATHER) PA-0: the group-aligned layout by an inverse row gather

Checks, with the Pallas kernels in interpret mode off TPU:
  - the env keys reach union_walk_prefill only when set (unset: the call has no new kwarg) and reach the gathered walk
    kernel (kernel names gain _mv / _vg); the kernel refuses them off the gathered walk;
  - each flag alone, all together, and with qk16, are bitwise the flag-off walk on packed unaligned steps with the
    served option set (G64, KV1024, listed / gathered, in_place, gathered_lean, compact pool, lazy share), on a peaked
    softmax, at a group exactly at the union cap, and on the over-cap and min-context fallbacks;
  - the PA-0 source map equals the scatter's layout on random request layouts (empty requests, trailing rows,
    every fill), checked directly;
  - pool garbage the walk never reads (NaN codes past every request's length and on unowned pages) changes nothing;
  - the precondition is real: a NaN in a row the walk does read (word 0 of a request, the padding source) is zeroed
    by the served guard and reaches the output under vguard_lean (a boundary check, not a use case).
The bitwise claims are made with XLA's excess precision off, pinned before jax creates its CPU client.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax._src import xla_bridge  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

_PINNED_BEFORE_INIT = not xla_bridge.backends_are_initialized()

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

RATIO, TOPK, PAGE = 4, 512, 3200
CAP = 8192
SPARE = ((37, 300, 129, 201, 333), (5000, 1500, 4000, 700, 2900))
A64 = ((64, 192, 320, 128, 320), (5000, 1500, 4000, 700, 2900))
WORST = ((129,) * 7, (4000, 100, 2500, 700, 5200, 1900, 3300))
FOUR = ((37, 300, 129, 201), (5000, 1500, 4000, 700))  # <= compact_reqs requests: the compact-pool gathered walk
SERVED = dict(group=64, kv_tile=1024, union_cap=CAP, min_context=0, lean=True, listed="reference",
              gathered="reference", gather_chunks=3, in_place=True, gathered_lean=True, compact_reqs=4,
              share_bits=True, lazy_share=True)
QA1 = dict(mask_view=True, vguard_lean=True, align_gather=True)
NEW_ENV = ("VLLM_TPU_QSA_PREFILL_UNION_MASK_VIEW", "VLLM_TPU_QSA_PREFILL_UNION_VGUARD_LEAN",
           "VLLM_TPU_QSA_PREFILL_UNION_ALIGN_GATHER")
NAN_E4M3 = 0x7F


@pytest.fixture
def kernels(monkeypatch):
    jax.clear_caches()
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bsp.pl, "pallas_call", interpreted)
    for name in NEW_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    jax.clear_caches()


def _packed_step(seed, rows, cached, tokens=1024, q_std=0.5, garbage=False):
    """A ``tokens``-row step of len(rows) requests, each on its own pages of a permuted, finite FP8 pool.

    ``garbage``: every pool slot the walk never reads (past each request's length, and the unowned pages) holds the
    E4M3 NaN code."""
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-(c + n) // PAGE) for n, c in zip(rows, cached))
    pool_pages = pages_per_seq * len(rows) + (2 if garbage else 0)
    assert sum(rows) <= tokens
    starts = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pages_per_seq), np.int32)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)  # exponent < 15: finite
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    if garbage:
        written = np.zeros((pool_pages, PAGE), bool)
    for r, (n, ctx) in enumerate(zip(rows, cached)):
        pages = owner[r * pages_per_seq:(r + 1) * pages_per_seq]
        if garbage:
            for tok in range(ctx + n):
                written[pages[tok // PAGE], tok % PAGE] = True
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            table[t] = pages
            complete = length // RATIO
            if complete:
                local = list(range(max(0, complete - 32), complete))
                rest = rng.permutation(complete)[: TOPK - len(local)]
                pick = np.unique(np.concatenate([local, rest]))[:TOPK]
                blocks[t, : len(pick)] = rng.permutation(pick)
    if garbage:
        raw[~written] = NAN_E4M3
    q = jnp.asarray(rng.normal(0, q_std, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


def _bits(x):
    return np.asarray(x).view(np.uint16)


def _walk(step, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(SERVED)
    params.update(options)
    out, cache = ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO,
                                       **params)
    assert cache is pool
    return np.asarray(out)


def _kernel_names(jaxpr):
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            found.append(str(eqn.params.get("name_and_src_info", eqn.params.get("name"))).split(" ")[0])
        for param in eqn.params.values():
            for item in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    found += _kernel_names(inner)
    return found


def _walk_names(step, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(SERVED)
    params.update(options)
    jaxpr = jax.make_jaxpr(lambda: ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2,
                                                         ratio=RATIO, **params))().jaxpr
    return [n for n in _kernel_names(jaxpr) if n.startswith("qsa_union_walk_g")]


def _kernel_spy(monkeypatch):
    calls = []
    original = ud.union_walk_kernel

    def spy(*args, **kw):
        calls.append(dict(gathered=kw.get("gathered") is not None, mask_view=kw.get("mask_view", False),
                          vguard_lean=kw.get("vguard_lean", False), keys=tuple(sorted(kw))))
        return original(*args, **kw)

    monkeypatch.setattr(ud, "union_walk_kernel", spy)
    return calls


def _prefill_spy(monkeypatch):
    calls = []
    original = ud.union_walk_prefill

    def spy(*args, **kw):
        calls.append(dict(kw))
        return original(*args, **kw)

    monkeypatch.setattr(ud, "union_walk_prefill", spy)
    return calls


def test_excess_precision_is_off_in_this_process():
    assert _STRICT in os.environ["XLA_FLAGS"] and _PINNED_BEFORE_INIT, (os.environ.get("XLA_FLAGS"), _PINNED_BEFORE_INIT)
    x = (1.0 + (np.arange(8 * 128, dtype=np.float32).reshape(8, 128) + 1.0) * 2.0**-12).astype(np.float32)

    def body(v):
        w = v * 3.0
        return w.astype(jnp.bfloat16).astype(jnp.float32) - w

    def kern(x_ref, o_ref):
        o_ref[...] = body(x_ref[...])

    jitted = np.asarray(jax.jit(body)(jnp.asarray(x)))
    interp = np.asarray(pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
                                       interpret=True)(jnp.asarray(x)))
    assert np.any(jitted != 0) and np.any(interp != 0), "the bf16 round trip was elided: excess precision is on"


def test_kernel_refuses_qa1_off_the_gathered_walk():
    q = jnp.zeros((64, 24, 256), jnp.bfloat16)
    common = dict(group=64, union_cap=CAP, kv_tile=1024, fast_walk=True)
    for kw in (dict(mask_view=True), dict(vguard_lean=True)):
        with pytest.raises(ValueError, match="gathered walk only"):
            ud.union_walk_kernel(q, q, q, q, q, q, q, **kw, **common)


def test_env_keys_are_registered_default_off():
    from vllm_torchtpu import envs

    for name in NEW_ENV:
        assert name in envs.environment_variables, name
        os.environ.pop(name, None)
        assert getattr(envs, name) is False, name


def test_names_mark_engagement(kernels):
    step = _packed_step(7, *A64)
    base = _walk_names(step)
    gathered = [n for n in base if "_gathered" in n]
    assert gathered and all(n.endswith("_gathered_glean") for n in gathered), base
    for opts, suffix in ((dict(mask_view=True), "_glean_mv"), (dict(vguard_lean=True), "_glean_vg"),
                         (dict(mask_view=True, vguard_lean=True), "_glean_mv_vg")):
        names = _walk_names(step, **opts)
        g = [n for n in names if "_gathered" in n]
        assert g and all(n.endswith(suffix) for n in g), (opts, names)
        # the compact overflow's listed walk (not a gathered walk) is unchanged
        assert [n for n in names if "_gathered" not in n] == [n for n in base if "_gathered" not in n]
    assert _walk_names(step, align_gather=True) == base  # PA-0 is outside the kernel


def _walk_kernel_jaxprs(step, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(SERVED)
    params.update(options)
    closed = jax.make_jaxpr(lambda: ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2,
                                                          ratio=RATIO, **params))()
    found = []

    def visit(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                name = str(eqn.params.get("name_and_src_info", eqn.params.get("name"))).split(" ")[0]
                if name.startswith("qsa_union_walk_g") and "_gathered" in name:
                    found.append(eqn.params["jaxpr"])
                continue
            for param in eqn.params.values():
                for item in param if isinstance(param, (tuple, list)) else (param,):
                    inner = getattr(item, "jaxpr", item)
                    if hasattr(inner, "eqns"):
                        visit(inner)

    visit(closed.jaxpr)
    return found


def _census(kernel_jaxpr, rows, kv_tile):
    """(is_finite count, count of bool [rows, kv_tile] values made by an equation) over the kernel body."""
    finite, masks = 0, 0
    kernel_jaxpr = getattr(kernel_jaxpr, "jaxpr", kernel_jaxpr)

    def visit(jaxpr):
        nonlocal finite, masks
        for eqn in jaxpr.eqns:
            finite += eqn.primitive.name == "is_finite"
            for v in eqn.outvars:
                aval = getattr(v, "aval", None)
                if aval is not None and getattr(aval, "shape", None) == (rows, kv_tile) and aval.dtype == jnp.bool_:
                    masks += 1
            for param in eqn.params.values():
                for item in param if isinstance(param, (tuple, list)) else (param,):
                    inner = getattr(item, "jaxpr", item)
                    if hasattr(inner, "eqns"):
                        visit(inner)

    visit(kernel_jaxpr)
    return finite, masks


def test_components_engage_in_the_kernel(kernels):
    """The exact components change the kernel body as designed (bitwise outputs cannot show that): mask_view leaves no
    bool [12 * 64, 1024] mask, vguard_lean leaves no isfinite; flag-off keeps both."""
    step = _packed_step(311, *A64)
    rows = 12 * 64
    for opts, want_finite, want_masks in ((dict(), True, True), (dict(mask_view=True), True, False),
                                          (dict(vguard_lean=True), False, True), (QA1, False, False)):
        kernels_ = _walk_kernel_jaxprs(step, compact_reqs=0, **opts)
        assert kernels_, opts
        for kj in kernels_:
            finite, masks = _census(kj, rows, 1024)
            assert (finite > 0) == want_finite and (masks > 0) == want_masks, (opts, finite, masks)


@pytest.mark.parametrize("opts", [dict(mask_view=True), dict(vguard_lean=True), dict(align_gather=True), QA1],
                         ids=["mask_view", "vguard_lean", "align_gather", "qa1"])
@pytest.mark.parametrize("layout,compact", [(FOUR, 4), (SPARE, 0), (A64, 0), (WORST, 0)],
                         ids=["four-compact", "spare", "a64", "worst"])
def test_qa1_is_the_walk(kernels, monkeypatch, opts, layout, compact):
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 64) * 64 for n in rows) > 1024 else 1024
    step = _packed_step(301 + len(rows), rows, cached, tokens=tokens)
    want = _walk(step, compact_reqs=compact)
    assert np.isfinite(want.astype(np.float32)).all() and np.abs(want.astype(np.float32)).max() > 0
    calls = _kernel_spy(monkeypatch)
    jax.clear_caches()
    got = _walk(step, compact_reqs=compact, **opts)
    np.testing.assert_array_equal(_bits(got), _bits(want))
    gathered = [c for c in calls if c["gathered"]]
    assert gathered, calls
    for key in ("mask_view", "vguard_lean"):
        assert all(c[key] == opts.get(key, False) for c in gathered), (key, calls)
    assert not any(c["mask_view"] or c["vguard_lean"] for c in calls if not c["gathered"]), calls


def test_qa1_with_qk16_is_the_qk16_walk(kernels):
    for layout, compact in ((SPARE, 0), (FOUR, 4)):
        step = _packed_step(331, *layout, tokens=2048)
        want = _walk(step, qk16=True, compact_reqs=compact)
        jax.clear_caches()
        np.testing.assert_array_equal(_bits(_walk(step, qk16=True, compact_reqs=compact, **QA1)), _bits(want))
        jax.clear_caches()


def test_qa1_on_a_peaked_softmax(kernels):
    """Large queries make every row's softmax peaked (rescales underflow to zero on later tiles)."""
    step = _packed_step(337, *A64, q_std=8.0)
    want = _walk(step, compact_reqs=0)
    assert np.isfinite(want.astype(np.float32)).all()
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(_walk(step, compact_reqs=0, **QA1)), _bits(want))


def _union_sizes(step, group):
    """Each walked group's union as union_bits_plan counts it: the rows' selected ids plus each row's partial tail group
    (``length // ratio`` when ``length % ratio``); requests start on group boundaries of the walk layout."""
    q, gate, blocks, lengths, table, pool, starts = step
    sizes = []
    st = np.asarray(starts)
    bl, ln = np.asarray(blocks), np.asarray(lengths)
    for r in range(len(st) - 1):
        for g0 in range(st[r], st[r + 1], group):
            g1 = min(g0 + group, st[r + 1])
            ids = {int(b) for b in bl[g0:g1].reshape(-1) if b >= 0}
            ids |= {int(n) // RATIO for n in ln[g0:g1] if n > 0 and n % RATIO}
            sizes.append(len(ids))
    return sizes


def test_qa1_at_the_union_cap_and_on_fallbacks(kernels):
    step = _packed_step(347, *A64)
    cap = max(_union_sizes(step, 64))
    walked = _walk(step, compact_reqs=0)
    jax.clear_caches()
    want = _walk(step, union_cap=cap, compact_reqs=0)  # the largest group sits exactly at the cap: still walked
    np.testing.assert_array_equal(_bits(want), _bits(walked))  # (a fallback would be the served call, not bitwise)
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(_walk(step, union_cap=cap, compact_reqs=0, **QA1)), _bits(want))
    for opts in (dict(union_cap=cap - 1, compact_reqs=0), dict(union_cap=32), dict(min_context=10**6)):
        # one group over the cap / over-cap / min-context: the served call
        jax.clear_caches()
        want = _walk(step, **opts)
        jax.clear_caches()
        np.testing.assert_array_equal(_bits(_walk(step, **opts, **QA1)), _bits(want))


def test_unread_pool_garbage_changes_nothing(kernels):
    for layout, compact in ((SPARE, 0), (FOUR, 4)):
        step = _packed_step(353, *layout, tokens=2048, garbage=True)
        want = _walk(step, compact_reqs=compact)
        assert np.isfinite(want.astype(np.float32)).all()
        jax.clear_caches()
        np.testing.assert_array_equal(_bits(_walk(step, compact_reqs=compact, **QA1)), _bits(want))
        jax.clear_caches()


def test_boundary_a_read_nan_reaches_the_output_under_vguard_lean(kernels):
    """The finite-pool precondition, shown: a NaN value at word 0 of request 1 (the padding source of its groups) is
    zeroed by the served guard and reaches the output under vguard_lean. Not a use case: real pools are finite."""
    q, gate, blocks, lengths, table, pool, starts = _packed_step(359, *A64)
    raw = np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8)).copy()
    page0 = int(np.asarray(table)[int(np.asarray(starts)[1])][0])
    raw[page0, 0, 0, 1, :] = NAN_E4M3  # token 0, packing slot 1 = head 0's value
    step = (q, gate, blocks, lengths, table, jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn),
            starts)
    served = _walk(step, compact_reqs=0)
    assert np.isfinite(served.astype(np.float32)).all()
    jax.clear_caches()
    lean = _walk(step, compact_reqs=0, vguard_lean=True)
    assert not np.isfinite(lean.astype(np.float32)).all()


def _scatter_layout(array, fill, starts, offsets, max_tokens):
    tokens = array.shape[0]
    real = starts[1:] - starts[:-1]
    req = jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right")
    req = jnp.minimum(req, len(real) - 1)
    positions = jnp.arange(tokens) - starts[req]
    valid = jnp.arange(tokens) < starts[-1]
    slots = jnp.where(valid, offsets[req] + positions, max_tokens)
    shape = (max_tokens, *array.shape[1:])
    return jnp.full(shape, fill, array.dtype).at[slots].set(array, mode="drop")


@pytest.mark.parametrize("seed", range(40))
def test_align_gather_source_is_the_scatter(seed):
    rng = np.random.default_rng(1000 + seed)
    nreq = int(rng.integers(1, 9))
    real = rng.integers(0, 300, nreq)
    real[rng.random(nreq) < 0.2] = 0  # empty requests
    align = int(rng.choice([64, 128, 256]))
    tokens = int(real.sum() + rng.integers(0, 200))  # trailing rows past the last request
    tokens = max(tokens, 1)
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(real)]).astype(np.int32))
    padded = (real + align - 1) // align * align
    offsets = jnp.asarray(np.concatenate([[0], np.cumsum(padded)]).astype(np.int32))
    for max_tokens in sorted(m for m in {int(offsets[-1]) + align * k for k in (0, 1, 3)}
                             | {max(align, int(offsets[-1]) - align)} if m > 0):
        source, hit = ud.align_gather_source(starts, offsets, max_tokens)
        for array, fill in ((jnp.asarray(rng.normal(size=(tokens, 3, 5)), jnp.bfloat16), 0),
                            (jnp.asarray(rng.integers(0, 9, (tokens, 7)), jnp.int32), -1),
                            (jnp.asarray(rng.integers(1, 99, (tokens,)), jnp.int32), 0)):
            want = _scatter_layout(array, fill, starts, offsets, max_tokens)
            got = ud.align_rows_gather(array, fill, source, hit)
            assert got.dtype == want.dtype and got.shape == want.shape
            np.testing.assert_array_equal(np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8))


def _dispatch(step):
    q, gate, blocks, lengths, table, pool, starts = step
    out, cache = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO)
    assert cache is pool
    return np.asarray(out)


def test_env_dispatch(kernels, monkeypatch):
    """Unset: union_walk_prefill gets none of the three kwargs. Set: each arrives as True, and the output is the same."""
    step = _packed_step(367, *A64, tokens=1024)
    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_CAP", str(CAP)), ("VLLM_TPU_QSA_PREFILL_LEAN", "1"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_GATHERED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_IN_PLACE", "1"), ("VLLM_TPU_QSA_PREFILL_UNION_GATHERED_LEAN", "1")):
        monkeypatch.setenv(name, value)
    calls = _prefill_spy(monkeypatch)
    want = _dispatch(step)
    assert calls and not any(k in calls[0] for k in ("mask_view", "vguard_lean", "align_gather")), calls
    for name in NEW_ENV:
        monkeypatch.setenv(name, "1")
    calls.clear()
    kcalls = _kernel_spy(monkeypatch)
    jax.clear_caches()
    got = _dispatch(step)
    assert calls and all(calls[0].get(k) is True for k in ("mask_view", "vguard_lean", "align_gather")), calls
    assert any(c["gathered"] and c["mask_view"] and c["vguard_lean"] for c in kcalls), kcalls
    np.testing.assert_array_equal(_bits(got), _bits(want))
