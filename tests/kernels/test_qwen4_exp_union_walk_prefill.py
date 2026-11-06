# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_UNION_WALK: the D2 union walk on packed multi-request prefill steps.

``qsa_union_dense.union_walk_prefill`` aligns each request of a packed step to
query-group boundaries (as ``block_sparse_gqa_attention.run_with_padding``
does), walks each group's union with that group's own page row, and falls
back to the served block-sparse call when a union exceeds the cap or the step
is short. Steps are built as in ``test_qwen4_exp_prefill_q128_packed.py``:
several requests at unaligned boundaries, each with its own cached context
and its own pages of a permuted pool, so a wrong page row, a wrong group's
pool bases or a misaligned request changes the output.

Checks: each request's rows equal the single-request walk bit for bit; the
two fallbacks equal the served call bit for bit (and the walk differs from it
one step past each threshold); the walk is within the float64 2x rule of the
served kernel; the ``_prefill_attention`` dispatch is unchanged with the flag
off and takes the walk with it on. Off TPU the Pallas kernels run in
interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles
from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

RATIO, TOPK, PAGE = 4, 512, 3200
CAP = 8192  # above every union below

# Five requests at unaligned boundaries (aligned to 128 they need 1408 rows: the
# bucket's spare-block layout); 24 trailing bucket rows of length 0.
SPARE = ((37, 300, 129, 201, 333), (5000, 1500, 4000, 700, 2900))
# Four unaligned requests that need exactly the bucket's 1024 rows aligned to 128.
BUCKET = ((37, 300, 129, 201), (5000, 1500, 4000, 700))
# Every request a multiple of 128 rows: the identity layout.
ALIGNED = ((256, 384, 128, 256), (3000, 300, 4500, 1000))
# Seven 129-row requests need 1792 aligned rows: past the spare blocks, the worst-case layout.
WORST = ((129,) * 7, (4000, 100, 2500, 700, 5200, 1900, 3300))


@pytest.fixture
def kernels(monkeypatch):
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
    for name in ("VLLM_TPU_QSA_PREFILL_UNION_WALK", "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT",
                 "VLLM_TPU_QSA_PREFILL_UNION_CAP", "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE",
                 "VLLM_TPU_QSA_PREFILL_LEAN"):
        monkeypatch.delenv(name, raising=False)
    yield
    jax.clear_caches()


def _packed_step(seed, rows, cached, tokens=1024):
    """A ``tokens``-row step of len(rows) requests; each request owns pages of a permuted pool."""
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-(c + n) // PAGE) for n, c in zip(rows, cached))
    pool_pages = pages_per_seq * len(rows)
    assert sum(rows) <= tokens
    starts = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pages_per_seq), np.int32)
    for r, (n, ctx) in enumerate(zip(rows, cached)):
        pages = owner[r * pages_per_seq:(r + 1) * pages_per_seq]
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
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


def _bits(x):
    return np.asarray(x).view(np.uint16)


def _walk(step, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(group=128, kv_tile=1024, union_cap=CAP, min_context=0, lean=True)
    params.update(options)
    out, cache = ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2,
                                       ratio=RATIO, **params)
    assert cache is pool
    return np.asarray(out)


def _served(step):
    """The block_sparse_gqa_attention call qsa_rpa._prefill_attention makes (dcp None, no LSE)."""
    q, gate, blocks, lengths, table, pool, starts = step
    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        q, pool, kv_heads=2, ratio=RATIO, with_query_tile=True
    )
    out, _ = bsp.block_sparse_gqa_attention(
        q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO,
        query_block=query_block, kv_tile=kv_tile,
        **({"query_tile": query_tile} if query_tile else {}),
    )
    return np.asarray(out)


def _request(step, r, group):
    """Request r alone, padded with length-0 rows to a multiple of ``group``, and its page row."""
    q, gate, blocks, lengths, table, pool, starts = step
    a, b = int(starts[r]), int(starts[r + 1])
    n = b - a
    n_pad = -(-n // group) * group

    def pad(x, fill):
        x = np.asarray(x)[a:b]
        return jnp.asarray(np.concatenate([x, np.full((n_pad - n, *x.shape[1:]), fill, x.dtype)]))

    return (pad(q, 0), pad(gate, 0), pad(blocks, -1), pad(lengths, 0), pool,
            jnp.asarray(np.asarray(table)[a])), a, b


def _max_union(step, group):
    """Largest group union of the step's aligned layout (each request's groups alone)."""
    _, _, _, _, table, _, starts = step
    best = 0
    for r in range(len(starts) - 1):
        (_, _, blocks, lengths, _, row), _, _ = _request(step, r, group)
        _, _, usize, _ = union_bits_plan(blocks, lengths, group=group, union_cap=CAP,
                                         groups_cap=table.shape[1] * PAGE // RATIO)
        best = max(best, int(np.asarray(usize).max()))
    return best


@pytest.mark.parametrize(
    "layout,group,kv_tile,lean",
    [
        (SPARE, 128, 1024, True),
        (BUCKET, 128, 1024, True),
        (ALIGNED, 128, 1024, True),
        (WORST, 128, 1024, True),
        (SPARE, 64, 512, False),
    ],
    ids=["spare-g128", "bucket-g128", "aligned-g128", "worst-g128", "spare-g64"],
)
def test_each_request_equals_the_single_request_walk(kernels, layout, group, kv_tile, lean):
    rows, cached = layout
    step = _packed_step(len(rows) + group, rows, cached)
    got = _walk(step, group=group, kv_tile=kv_tile, lean=lean)
    assert np.isfinite(got.astype(np.float32)).all() and np.abs(got.astype(np.float32)).max() > 0
    starts = np.asarray(step[-1])
    # Bucket rows past the last request (length 0) stay zero.
    assert not got[starts[-1]:].astype(np.float32).any()
    unions = []
    for r in range(len(rows)):
        args, a, b = _request(step, r, group)
        jax.clear_caches()
        want, usize = ud.union_walk_attention(*args, group=group, union_cap=CAP, kv_tile=kv_tile,
                                              lean=lean, fast_walk=True)
        unions.append(int(np.asarray(usize).max()))
        np.testing.assert_array_equal(_bits(got[a:b]), _bits(np.asarray(want)[: b - a]), err_msg=f"request {r}")
    # Some union spans several tiles, so the double-buffered walk crosses tile boundaries.
    assert max(unions) * RATIO > 2 * kv_tile, unions


def test_cap_fallback_is_the_served_call(kernels):
    rows, cached = SPARE
    step = _packed_step(11, rows, cached)
    largest = _max_union(step, 128)
    served = _served(step)
    jax.clear_caches()
    over = _walk(step, union_cap=largest - 1)
    jax.clear_caches()
    tiny = _walk(step, union_cap=32)
    jax.clear_caches()
    at = _walk(step, union_cap=largest)
    np.testing.assert_array_equal(_bits(over), _bits(served))
    np.testing.assert_array_equal(_bits(tiny), _bits(served))
    # One group more of cap and the walk runs: its accumulation order differs from the served kernel's.
    assert not np.array_equal(_bits(at), _bits(served))


def test_min_context_fallback_is_the_served_call(kernels):
    rows, cached = SPARE
    step = _packed_step(12, rows, cached)
    longest = int(np.asarray(step[3]).max())
    served = _served(step)
    jax.clear_caches()
    short = _walk(step, min_context=longest + 1)
    jax.clear_caches()
    at = _walk(step, min_context=longest)
    np.testing.assert_array_equal(_bits(short), _bits(served))
    assert not np.array_equal(_bits(at), _bits(served))


def _f64_reference(step, rows_sel):
    """Exact selected-key attention in float64 on ``rows_sel``, each row on its own page row; gated as served."""
    q, gate, blocks, lengths, table, pool, _ = step
    q64, g64 = np.asarray(q, np.float64), np.asarray(gate, np.float64)
    flat = np.asarray(pool.reshape(-1, 4, 256)).astype(np.float64)
    table, blocks, lengths = np.asarray(table), np.asarray(blocks), np.asarray(lengths)
    out = np.zeros((len(rows_sel), 24, 256))
    for n, row in enumerate(rows_sel):
        length = int(lengths[row])
        groups = {int(g) for g in blocks[row] if g >= 0}
        if length % RATIO:
            groups.add(length // RATIO)
        toks = np.asarray(sorted(t for g in groups for t in range(g * RATIO, g * RATIO + RATIO) if t < length))
        kv = flat[table[row][toks // PAGE] * PAGE + toks % PAGE]
        for h in range(24):
            k, v = kv[:, 2 * (h // 12)], kv[:, 2 * (h // 12) + 1]
            s = k @ q64[row, h] / 16.0
            p = np.exp(s - s.max())
            out[n, h] = (p / p.sum()) @ v
    return out / (1.0 + np.exp(-g64[rows_sel]))


def test_walk_within_twice_the_served_error_against_float64(kernels):
    rows, cached = SPARE
    step = _packed_step(13, rows, cached)
    served = _served(step)
    jax.clear_caches()
    got = _walk(step)
    starts = np.asarray(step[-1])
    # Four rows of every request: first, last and two inside.
    rows_sel = np.concatenate([np.linspace(starts[r], starts[r + 1] - 1, 4).astype(int) for r in range(len(rows))])
    ref = _f64_reference(step, rows_sel)
    rms = np.sqrt(np.mean(ref**2, axis=-1, keepdims=True)) + 1e-30

    def err(x):
        e = np.abs(x[rows_sel].astype(np.float32).astype(np.float64) - ref) / rms
        return e.max(), np.quantile(e, 0.99)

    served_max, served_p99 = err(served)
    got_max, got_p99 = err(got)
    assert served_max < 0.1, served_max  # the reference itself is sound
    assert got_max <= 2 * served_max and got_p99 <= 2 * served_p99, (got_max, got_p99, served_max, served_p99)


def _spy(monkeypatch):
    calls = []
    original = ud.union_walk_prefill

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ud, "union_walk_prefill", spy)
    return calls


def _dispatch(step):
    q, gate, blocks, lengths, table, pool, starts = step
    out, cache = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts,
                                            kv_heads=2, ratio=RATIO)
    assert cache is pool
    return np.asarray(out)


def test_flag_off_dispatch_is_the_served_call(kernels, monkeypatch):
    rows, cached = SPARE
    step = _packed_step(14, rows, cached)
    calls = _spy(monkeypatch)
    got = _dispatch(step)
    assert calls == []
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(got), _bits(_served(step)))


def test_flag_on_dispatch_takes_the_walk(kernels, monkeypatch):
    rows, cached = SPARE
    step = _packed_step(15, rows, cached)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_WALK", "128")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "1000")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_CAP", str(CAP))
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_LEAN", "1")
    calls = _spy(monkeypatch)
    got = _dispatch(step)
    assert calls == [dict(kv_heads=2, ratio=RATIO, k_scale=1.0, v_scale=1.0, group=128, kv_tile=1024,
                          union_cap=CAP, min_context=1000, lean=True)]
    jax.clear_caches()
    want = _walk(step, union_cap=CAP, min_context=1000, lean=True)
    np.testing.assert_array_equal(_bits(got), _bits(want))
    # The walk kernel ran, not the fallback.
    jax.clear_caches()
    assert not np.array_equal(_bits(got), _bits(_served(step)))


def test_flag_on_dispatch_keeps_short_steps_and_rejects_bad_groups(kernels, monkeypatch):
    rows, cached = (37, 300, 129), (5000, 1500, 4000)
    step = _packed_step(16, rows, cached, tokens=512)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_WALK", "128")
    calls = _spy(monkeypatch)
    got = _dispatch(step)  # 512 < 1024 tokens: today's path
    assert calls == []
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(got), _bits(_served(step)))
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_WALK", "100")
    with pytest.raises(ValueError, match="divisor of 128"):
        _dispatch(step)
