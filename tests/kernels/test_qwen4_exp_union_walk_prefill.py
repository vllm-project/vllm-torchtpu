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
                 "VLLM_TPU_QSA_PREFILL_LEAN", "VLLM_TPU_QSA_PREFILL_UNION_LISTED",
                 "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS", "VLLM_TPU_QSA_PREFILL_UNION_ALPHA",
                 "VLLM_TPU_QSA_PREFILL_UNION_QK16", "VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES",
                 "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE", "VLLM_TPU_QSA_PREFILL_UNION_GATHERED",
                 "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS"):
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
                          union_cap=CAP, min_context=1000, lean=True, listed=None, share_bits=False,
                          union_alpha=0.0, qk16=False, run_tables=False, lazy_share=False, gathered=None,
                          gather_chunks=4)]
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


# --- relative cap (union_alpha) and the shared-bitset fallback (share_bits): both opt-in -------------------------


@pytest.mark.parametrize("layout", [SPARE, WORST, ALIGNED])
def test_share_bits_fallback_is_the_served_call(kernels, layout):
    """A cap fallback that reuses the walk plan's bitset is bitwise the served call (packed, unaligned steps)."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(21 + len(rows), rows, cached, tokens=tokens)
    served = _served(step)
    jax.clear_caches()
    over = _walk(step, union_cap=_max_union(step, 128) - 1, share_bits=True)
    jax.clear_caches()
    tiny = _walk(step, union_cap=32, share_bits=True)
    np.testing.assert_array_equal(_bits(over), _bits(served))
    np.testing.assert_array_equal(_bits(tiny), _bits(served))


@pytest.mark.parametrize("layout", [SPARE, WORST])
def test_share_bits_walk_is_the_unshared_walk(kernels, layout):
    """Aligning requests to the served query block moves only padding rows: the walk's output is unchanged."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(31 + len(rows), rows, cached, tokens=tokens)
    unshared = _walk(step)
    jax.clear_caches()
    shared = _walk(step, share_bits=True)
    np.testing.assert_array_equal(_bits(shared), _bits(unshared))
    assert not np.array_equal(_bits(shared), _bits(_served(step)))  # the walk ran


def test_union_alpha_falls_back_on_dense_unions(kernels):
    """union_alpha: a union above alpha x the group's visible groups takes the served call; below it, the walk."""
    rows, cached = SPARE
    step = _packed_step(41, rows, cached)
    served = _served(step)
    walk = _walk(step)
    jax.clear_caches()
    dense = _walk(step, union_alpha=0.01)          # every union is above 1% of its visible groups
    jax.clear_caches()
    roomy = _walk(step, union_alpha=1.0)           # no union exceeds all visible groups
    jax.clear_caches()
    dense_shared = _walk(step, union_alpha=0.01, share_bits=True)
    np.testing.assert_array_equal(_bits(dense), _bits(served))
    np.testing.assert_array_equal(_bits(dense_shared), _bits(served))
    np.testing.assert_array_equal(_bits(roomy), _bits(walk))


def _dot_operand_dtypes(jaxpr):
    """The operand dtypes of every dot_general in a jaxpr, kernels (pallas_call bodies) included."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            found.append(tuple(str(v.aval.dtype) for v in eqn.invars))
        for param in eqn.params.values():
            for item in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    found += _dot_operand_dtypes(inner)
    return found


def test_qk16_operands_are_the_fp32_operands():
    """Exhaustive: qk16 hands the MXU the same operand values as the FP32 dot.

    Keys: all 256 FP8 E4M3 codes convert to the same value through BF16 as
    through FP32. Queries: every one of the 65,536 BF16 values times the served
    score scale 2**-4 (head_dim 256, k_scale 1.0: the checkpoint has no KV scale
    tensors) is still a BF16 value, except where FP32 itself turns subnormal.
    """
    codes = jax.lax.bitcast_convert_type(jnp.arange(256, dtype=jnp.uint8), jnp.float8_e4m3fn)
    via32 = np.asarray(codes.astype(jnp.float32))
    via16 = np.asarray(codes.astype(jnp.bfloat16).astype(jnp.float32))
    finite = np.isfinite(via32)
    assert finite.sum() == 254  # E4M3FN: two NaN codes, no infinities
    np.testing.assert_array_equal(via16[finite].view(np.uint32), via32[finite].view(np.uint32))
    assert np.isnan(via16[~finite]).all()

    q = jax.lax.bitcast_convert_type(jnp.arange(65536, dtype=jnp.uint32).astype(jnp.uint16), jnp.bfloat16)
    scaled = q.astype(jnp.float32) * jnp.float32(256**-0.5 * 1.0)
    recast = np.asarray(scaled.astype(jnp.bfloat16).astype(jnp.float32))
    scaled = np.asarray(scaled)
    normal = np.isfinite(scaled) & ((np.abs(scaled) >= 2.0**-126) | (scaled == 0))
    np.testing.assert_array_equal(recast[normal].view(np.uint32), scaled[normal].view(np.uint32))
    # The exceptions: nonzero |q| < 2**-122 (and NaN/inf, which stay NaN/inf).
    tiny = np.isfinite(scaled) & ~normal
    assert np.all(np.abs(np.asarray(q.astype(jnp.float32))[tiny]) < 2.0**-122)


def test_qk16_walk_engaged_and_only_reorders_the_sum(kernels):
    """qk16 runs a BF16 QK dot (PV stays FP32) and changes only the QK accumulation order.

    Off TPU, XLA's CPU BF16 dot sums in another order than its FP32 dot, so the
    outputs are not bitwise here: under 0.1% of outputs move, each by about one
    BF16 unit in the last place (of the ungated output, then the gate's rounding),
    and the float64 2x rule of the FP32 walk holds.
    A precision cut (e.g. BF16 PV probabilities) moves far more outputs. The
    bitwise keep rule is the TPU replay against today's walk.
    """
    rows, cached = SPARE
    step = _packed_step(51, rows, cached)
    plain = _walk(step)
    jax.clear_caches()
    fast = _walk(step, qk16=True)
    a, b = fast.astype(np.float32), plain.astype(np.float32)
    # A move is at most one BF16 unit in the last place of the ungated output plus the gated
    # output's own rounding (2**-6 of the larger output), or of the (row, head) vector's RMS
    # where the output itself is near zero (cancellation).
    rms = np.sqrt(np.mean(b.astype(np.float64) ** 2, axis=-1, keepdims=True)) + 1e-30
    moved = a != b
    drift = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), rms)
    print(f"qk16 vs FP32 walk: {moved.mean():.2e} of outputs moved, max {drift.max():.2e} of max(|out|, RMS)")
    assert moved.mean() < 1e-3, moved.mean()
    assert drift.max() <= 2.0**-6, drift.max()
    starts = np.asarray(step[-1])
    rows_sel = np.concatenate([np.linspace(starts[r], starts[r + 1] - 1, 4).astype(int) for r in range(len(rows))])
    ref = _f64_reference(step, rows_sel)
    rms = np.sqrt(np.mean(ref**2, axis=-1, keepdims=True)) + 1e-30

    def err(x):
        e = np.abs(x[rows_sel].astype(np.float32).astype(np.float64) - ref) / rms
        return e.max(), np.quantile(e, 0.99)

    (plain_max, plain_p99), (fast_max, fast_p99) = err(plain), err(fast)
    assert fast_max <= 2 * plain_max and fast_p99 <= 2 * plain_p99, (fast_max, fast_p99, plain_max, plain_p99)

    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(kv_heads=2, ratio=RATIO, group=128, kv_tile=1024, union_cap=CAP, min_context=0,
                  lean=True)
    traced = {
        flag: _dot_operand_dtypes(jax.make_jaxpr(
            lambda: ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, qk16=flag,
                                          **params)
        )().jaxpr)
        for flag in (False, True)
    }
    assert ("bfloat16", "bfloat16") in traced[True]
    assert ("float32", "float32") in traced[True]  # PV
    assert ("bfloat16", "bfloat16") not in traced[False]


def test_qk16_refuses_an_inexact_score_scale(kernels):
    """A score scale that is not a power of two would round the pre-scaled queries: refused."""
    rows, cached = SPARE
    step = _packed_step(52, rows, cached)
    with pytest.raises(ValueError, match="power-of-two"):
        _walk(step, qk16=True, k_scale=1.1)
    with pytest.raises(ValueError, match="power-of-two"):
        _walk(step, qk16=True, k_scale=3.0)
    _walk(step, qk16=True, k_scale=0.5)  # 2**-5: exact


def test_share_bits_fallback_reuses_the_plan_bitset(kernels, monkeypatch):
    """Engaged: an over-cap step with share_bits runs no served-layout bitset build.

    Output equality alone cannot show this (a rebuilt bitset is the same bits).
    Each bitset build reports its query block when it runs (a callback runs only
    in the branch taken; the min-context path still holds a served call that a
    long step never takes): without share_bits the step runs the plan's
    128-lane build and then the served call's 256-row build; with it, only the
    plan's.
    """
    rows, cached = SPARE
    step = _packed_step(61, rows, cached)
    q, gate, blocks, lengths, table, pool, starts = step
    served_qb = get_sparse_prefill_tiles(q, pool, kv_heads=2, ratio=RATIO)[0]
    assert served_qb == 256
    ran = []
    original = bsp._selection_bitset

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        block = kwargs["query_block"]
        jax.debug.callback(lambda: ran.append(block))
        return out

    monkeypatch.setattr(bsp, "_selection_bitset", recording)
    by_mode = {}
    for share in (False, True):
        jax.clear_caches()
        ran.clear()
        out = _walk(step, union_cap=32, share_bits=share)
        jax.effects_barrier()
        by_mode[share] = sorted(ran)
        if share:
            np.testing.assert_array_equal(_bits(out), _bits(shared_reference))
        else:
            shared_reference = out
    assert by_mode[False] == [128, served_qb], by_mode
    assert by_mode[True] == [128], by_mode


# --- the listed walk on the served path, G64, and the full int-2 stack behind the flags ---------------------------


def _kernel_names(jaxpr):
    """Names of every pallas_call in a jaxpr, cond/switch branches included."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            found.append(str(eqn.params.get("name")))
        for param in eqn.params.values():
            for item in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    found += _kernel_names(inner)
    return found


def _traced_names(step, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(group=128, kv_tile=1024, union_cap=CAP, min_context=0, lean=True)
    params.update(options)
    return _kernel_names(jax.make_jaxpr(lambda: ud.union_walk_prefill(
        q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO, **params))().jaxpr)


@pytest.mark.parametrize("group,layout", [(128, SPARE), (64, SPARE), (64, WORST)])
def test_listed_prefill_is_the_walk(kernels, group, layout):
    """listed="reference": the listed walk on each group's page row, bitwise the walk, and it is what runs."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(71 + group + len(rows), rows, cached, tokens=tokens)
    walk = _walk(step, group=group)
    jax.clear_caches()
    listed = _walk(step, group=group, listed="reference")
    np.testing.assert_array_equal(_bits(listed), _bits(walk))
    assert not np.array_equal(_bits(walk), _bits(_served(step)))  # the walk ran
    names = _traced_names(step, group=group, listed="reference")
    assert any("_listed" in n for n in names), names
    assert not any("_listed" in n for n in _traced_names(step, group=group)), "listed off must not list"


def test_share_bits_at_g64(kernels):
    """share_bits at G64 (four 64-row groups per served 256-row block): the fallback is the served call and the
    walk is the unshared walk."""
    rows, cached = SPARE
    step = _packed_step(81, rows, cached, tokens=2048)
    served = _served(step)
    jax.clear_caches()
    fallback = _walk(step, group=64, union_cap=32, share_bits=True)
    np.testing.assert_array_equal(_bits(fallback), _bits(served))
    jax.clear_caches()
    unshared = _walk(step, group=64)
    jax.clear_caches()
    shared = _walk(step, group=64, share_bits=True)
    np.testing.assert_array_equal(_bits(shared), _bits(unshared))


def test_share_bits_skips_unpaired_served_tiles(kernels, monkeypatch):
    """share_bits on a bucket whose served tiles are unpaired (1,152 rows on 3,200-row pages without
    Q128_FALLBACK: served Q256/KV128, as the 3,200 bucket at int-2) runs the unshared walk and falls back to the
    served call, instead of failing at trace time (int-2 screen j-0b8938d2); it warns once per bucket.

    Engaged: the over-cap step runs the same bitset builds as share_bits off (the plan's, then the served call's
    own), so the skip is real; output alone cannot show it (a shared bitset gives the same bits)."""
    from unittest.mock import MagicMock

    from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles

    monkeypatch.setattr(prefill_tiles, "logger", MagicMock())  # the default-tiles warning (vllm logger)
    monkeypatch.delenv("VLLM_TPU_QSA_PREFILL_Q128_FALLBACK", raising=False)
    monkeypatch.setattr(ud, "_SHARE_BITS_SKIPPED", set())
    logged = []
    monkeypatch.setattr(ud, "_share_bits_skip_log", lambda msg, *args: logged.append(args))
    rows, cached = BUCKET
    step = _packed_step(91, rows, cached, tokens=1152)
    q, _, _, _, _, pool, _ = step
    served_qb, served_kt = get_sparse_prefill_tiles(q, pool, kv_heads=2, ratio=RATIO)
    assert (served_qb, served_kt) == (256, 128) and served_kt not in bsp.PAIRED_KV_TILES
    served = _served(step)
    jax.clear_caches()
    unshared = _walk(step, group=64)
    jax.clear_caches()
    shared = _walk(step, group=64, share_bits=True)
    np.testing.assert_array_equal(_bits(shared), _bits(unshared))
    assert not np.array_equal(_bits(shared), _bits(served))  # the walk ran
    ran = []
    original = bsp._selection_bitset

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        block = kwargs["query_block"]
        jax.debug.callback(lambda: ran.append(block))
        return out

    monkeypatch.setattr(bsp, "_selection_bitset", recording)
    builds = {}
    for share in (False, True):
        jax.clear_caches()
        ran.clear()
        fallback = _walk(step, group=64, union_cap=32, share_bits=share)
        jax.effects_barrier()
        builds[share] = sorted(ran)
        np.testing.assert_array_equal(_bits(fallback), _bits(served))
    assert builds[True] == builds[False] and served_qb in builds[True], builds
    assert logged == [(1152, 64, 256, 128)], logged  # two share_bits traces, one warning


# --- lazy share: walk on group-aligned rows, build the served layout only on a fallback ----------------------------

# Requests that are multiples of 64 rows but not of 256: share_bits aligns them to the served 256-row block (a padded
# layout); lazy share keeps the identity layout at G64.
A64 = ((64, 192, 320, 128, 320), (5000, 1500, 4000, 700, 2900))
# Every request a multiple of 256 rows: share_bits' and lazy share's layouts are both the identity.
A256 = ((256, 512, 256), (3000, 4500, 1000))


@pytest.mark.parametrize("layout", [SPARE, WORST, ALIGNED, A64, A256])
def test_lazy_share_fallback_is_the_served_call(kernels, layout):
    """A lazy-share fallback (the served call on a layout built from the plan's bitset) is bitwise the served call."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(23 + len(rows) + sum(rows) % 7, rows, cached, tokens=tokens)
    served = _served(step)
    jax.clear_caches()
    tiny = _walk(step, group=64, union_cap=32, share_bits=True, lazy_share=True)
    np.testing.assert_array_equal(_bits(tiny), _bits(served))
    jax.clear_caches()
    over = _walk(step, group=64, union_cap=_max_union(step, 64) - 1, share_bits=True, lazy_share=True)
    np.testing.assert_array_equal(_bits(over), _bits(served))


@pytest.mark.parametrize("layout", [SPARE, A64])
def test_lazy_share_walk_is_the_unshared_walk(kernels, layout):
    """The lazy walk aligns to the group (identity layout for group-aligned requests), not the served block, and is
    the unshared walk bit for bit; share_bits without lazy aligns to 256 (recorded at trace time)."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(33 + len(rows), rows, cached, tokens=tokens)
    unshared = _walk(step, group=64)
    jax.clear_caches()
    ud._WALK_ALIGN.clear()
    lazy = _walk(step, group=64, share_bits=True, lazy_share=True)
    assert ud._WALK_ALIGN[tokens] == (64, True), ud._WALK_ALIGN
    np.testing.assert_array_equal(_bits(lazy), _bits(unshared))
    assert not np.array_equal(_bits(lazy), _bits(_served(step)))  # the walk ran
    jax.clear_caches()
    ud._WALK_ALIGN.clear()
    _walk(step, group=64, share_bits=True)
    assert ud._WALK_ALIGN[tokens] == (256, False), ud._WALK_ALIGN


def test_lazy_share_fallback_reuses_the_plan_bitset(kernels, monkeypatch):
    """Engaged: an over-cap lazy-share step runs only the plan's bitset build (no served-layout build), as share_bits
    does; without sharing it also runs the served call's 256-row build."""
    rows, cached = A64
    step = _packed_step(62, rows, cached)
    ran = []
    original = bsp._selection_bitset

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        block = kwargs["query_block"]
        jax.debug.callback(lambda: ran.append(block))
        return out

    monkeypatch.setattr(bsp, "_selection_bitset", recording)
    by_mode = {}
    for mode in ("off", "lazy"):
        jax.clear_caches()
        ran.clear()
        _walk(step, group=64, union_cap=32, share_bits=mode == "lazy", lazy_share=mode == "lazy")
        jax.effects_barrier()
        by_mode[mode] = sorted(ran)
    assert by_mode["off"] == [128, 256], by_mode
    assert by_mode["lazy"] == [128], by_mode


@pytest.mark.parametrize("layout,direct", [(A256, True), (A64, False)])
def test_lazy_share_fallback_layout(kernels, monkeypatch, layout, direct):
    """Engaged: on a step whose requests are all 256-aligned, the lazy fallback runs the served kernel on the original
    rows (no padded layout); on a 64-aligned step it runs on the padded served layout."""
    rows, cached = layout
    step = _packed_step(64 + len(rows), rows, cached)
    ran = []
    original = bsp.block_sparse_gqa_prefill

    def recording(query, *args, **kwargs):
        out = original(query, *args, **kwargs)
        n = query.shape[0]
        jax.debug.callback(lambda: ran.append(n))
        return out

    monkeypatch.setattr(bsp, "block_sparse_gqa_prefill", recording)
    jax.clear_caches()
    out = _walk(step, group=64, union_cap=32, share_bits=True, lazy_share=True)
    jax.effects_barrier()
    tokens = step[0].shape[0]
    assert ran and (all(n == tokens for n in ran) if direct else all(n > tokens for n in ran)), (ran, tokens)
    monkeypatch.setattr(bsp, "block_sparse_gqa_prefill", original)
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(out), _bits(_served(step)))


def test_lazy_share_dispatch(kernels, monkeypatch):
    """VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE reaches union_walk_prefill."""
    rows, cached = A64
    step = _packed_step(63, rows, cached)
    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS", "1"), ("VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE", "1")):
        monkeypatch.setenv(name, value)
    calls = _spy(monkeypatch)
    _dispatch(step)
    assert calls[0]["share_bits"] is True and calls[0]["lazy_share"] is True, calls


# --- the chunked gathered walk on the served path (gathered=...; SparseCore gather on TPU) --------------------------


@pytest.mark.parametrize("layout,chunks", [(SPARE, 1), (SPARE, 4), (WORST, 3), (A64, 4)])
def test_gathered_prefill_is_the_walk(kernels, layout, chunks):
    """gathered="reference": the walk-free kernel on gathered tiles, in chunks of whole query groups, is the G64 walk bit
    for bit on packed unaligned steps, and it is what runs (gathered kernels traced, no listed walk)."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(91 + chunks + len(rows), rows, cached, tokens=tokens)
    walk = _walk(step, group=64)
    jax.clear_caches()
    got = _walk(step, group=64, listed="reference", gathered="reference", gather_chunks=chunks)
    np.testing.assert_array_equal(_bits(got), _bits(walk))
    assert not np.array_equal(_bits(walk), _bits(_served(step)))  # the walk ran
    names = _traced_names(step, group=64, listed="reference", gathered="reference", gather_chunks=chunks)
    assert any("_gathered" in n for n in names) and not any(n.endswith("_listed") for n in names), names


def test_gathered_with_lazy_share_and_fallback(kernels):
    """With lazy share: the gathered walk is still the walk; an over-cap step is still the served call bit for bit."""
    rows, cached = A64
    step = _packed_step(97, rows, cached)
    walk = _walk(step, group=64)
    jax.clear_caches()
    opts = dict(group=64, listed="reference", gathered="reference", gather_chunks=2, share_bits=True, lazy_share=True)
    np.testing.assert_array_equal(_bits(_walk(step, **opts)), _bits(walk))
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(_walk(step, union_cap=32, **opts)), _bits(_served(step)))


def test_gathered_args_are_checked(kernels):
    rows, cached = SPARE
    step = _packed_step(98, rows, cached)
    with pytest.raises(ValueError, match="gathered needs listed"):
        _walk(step, group=64, gathered="reference")
    with pytest.raises(ValueError, match="gathered needs listed"):
        _walk(step, group=64, listed="reference", gathered="reference", run_tables=True)
    with pytest.raises(ValueError, match="gathered: None"):
        _walk(step, group=64, listed="reference", gathered="xla")


def test_gathered_dispatch(kernels, monkeypatch):
    """VLLM_TPU_QSA_PREFILL_UNION_GATHERED / _GATHER_CHUNKS reach union_walk_prefill and give the direct call's bits."""
    rows, cached = A64
    step = _packed_step(99, rows, cached)
    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_GATHERED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS", "2")):
        monkeypatch.setenv(name, value)
    calls = _spy(monkeypatch)
    got = _dispatch(step)
    assert calls[0]["gathered"] == "reference" and calls[0]["gather_chunks"] == 2, calls
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(got), _bits(_walk(step, group=64, union_cap=calls[0]["union_cap"],
                                                         listed="reference", gathered="reference", gather_chunks=2)))


def test_full_stack_dispatch(kernels, monkeypatch):
    """The int-2 flags (G64, listed, share_bits, alpha 0.9, cap 16,384, qk16) reach union_walk_prefill and give the
    direct call's bits; without alpha and qk16 the stack is the plain G64 walk bit for bit (qk16 itself is gated
    bitwise on TPU, i1k2)."""
    rows, cached = SPARE
    step = _packed_step(82, rows, cached, tokens=2048)
    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_CAP", "16384"), ("VLLM_TPU_QSA_PREFILL_LEAN", "1"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS", "1"), ("VLLM_TPU_QSA_PREFILL_UNION_ALPHA", "0.9")):
        monkeypatch.setenv(name, value)
    calls = _spy(monkeypatch)
    got = _dispatch(step)
    assert calls == [dict(kv_heads=2, ratio=RATIO, k_scale=1.0, v_scale=1.0, group=64, kv_tile=1024,
                          union_cap=16384, min_context=0, lean=True, listed="reference", share_bits=True,
                          union_alpha=0.9, qk16=False, run_tables=False, lazy_share=False, gathered=None,
                          gather_chunks=4)]
    stack = dict(group=64, union_cap=16384, listed="reference", share_bits=True)
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(got), _bits(_walk(step, union_alpha=0.9, **stack)))
    # Without alpha (these synthetic unions are near-dense, so alpha 0.9 sends the step to the shared fallback):
    # listed + share_bits + cap 16,384 at G64 is the plain G64 walk bit for bit.
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(_walk(step, **stack)), _bits(_walk(step, group=64, union_cap=16384)))
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_QK16", "1")
    calls.clear()
    jax.clear_caches()
    _dispatch(step)
    assert calls[0]["qk16"] is True
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "bogus")
    with pytest.raises(ValueError, match="listed"):
        _dispatch(step)


@pytest.mark.parametrize("group,layout", [(64, SPARE), (64, WORST), (128, SPARE)])
def test_run_tables_prefill_is_the_walk(kernels, group, layout):
    """run_tables=True (XLA builders): each tile from run tables + precomputed lines, per-group page rows of a packed
    step; bitwise the walk, and the run-table kernel is what runs."""
    rows, cached = layout
    tokens = 2048 if sum(-(-n // 256) * 256 for n in rows) > 1024 else 1024
    step = _packed_step(91 + group + len(rows), rows, cached, tokens=tokens)
    walk = _walk(step, group=group)
    jax.clear_caches()
    runs = _walk(step, group=group, listed="reference", run_tables=True)
    np.testing.assert_array_equal(_bits(runs), _bits(walk))
    names = _traced_names(step, group=group, listed="reference", run_tables=True)
    assert any("_runs" in n for n in names), names
    with pytest.raises(ValueError, match="run_tables needs listed"):
        _walk(step, group=group, run_tables=True)


def test_run_tables_dispatch(kernels, monkeypatch):
    rows, cached = SPARE
    step = _packed_step(93, rows, cached, tokens=2048)
    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_CAP", "16384"), ("VLLM_TPU_QSA_PREFILL_LEAN", "1"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS", "1"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES", "1")):
        monkeypatch.setenv(name, value)
    calls = _spy(monkeypatch)
    got = _dispatch(step)
    assert calls[0]["run_tables"] is True and calls[0]["listed"] == "reference"
    jax.clear_caches()
    np.testing.assert_array_equal(_bits(got), _bits(_walk(step, group=64, union_cap=16384)))
