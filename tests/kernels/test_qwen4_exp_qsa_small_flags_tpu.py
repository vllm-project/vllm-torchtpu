# SPDX-License-Identifier: Apache-2.0
"""Exact QSA decode flags on TPU, bit for bit (QSA_INDEXER_TEST_DEVICE=tpu;
skipped elsewhere).

- VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY / VLLM_TPU_RPA_SCHEDULE_HBM: the served
  decode RPA (compact 512-key tiles of 4 lanes, slot reads, lean/fast fetch,
  max 128 queries) on MTP verify steps (4 rows a request, each row its own
  512-group selection, contexts 2k-200k) under main-KV DCP 2 with the LSE
  (every rank, both paired slots) and replicated: out and LSE bytes equal to
  both flags off.
- VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE with LEAN_FETCH off: the lean and fast
  fetches' bits, and the plain per-page schedule's values.
- VLLM_TPU_QSA_INDEXER_DENSE_STEP / VLLM_TPU_QSA_INDEXER_TAIL_SKIP: the served
  replicated decode indexer (mixed pass, per-token scores, 4 query rows a
  block, 3,328-key blocks on the pool-page table with one DMA per pool page,
  FP8 pool read as BF16 keys 256 wide, max_model_len 262144, null-block table
  tails): the scoring kernel's written score words identical, and the
  SparseCore selection identical, including ties (``zero_keys`` /
  ``zero_query``) and -0.0 head terms (``test_signed_weights``).

Run on a TPU host: ``QSA_INDEXER_TEST_DEVICE=tpu python -m pytest -q -s
tests/kernels/test_qwen4_exp_qsa_small_flags_tpu.py``.
"""

import functools
import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# An environment switch, not a backend probe: collection must not initialize
# JAX (tests/conftest.py).
ON_TPU = os.environ.get("QSA_INDEXER_TEST_DEVICE", "cpu").lower() == "tpu"
pytestmark = pytest.mark.skipif(not ON_TPU, reason="QSA_INDEXER_TEST_DEVICE=tpu only")

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing  # noqa: E402

sit = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)

RATIO, TOPK_GROUPS, PAGE, SLOT = 4, 512, 512, 3
KV_BLOCK, KEYS_PER_PAGE = 3328, 32
MAX_MODEL_LEN = 262144
POOL_PAGES = 4096
D = 256

# --- decode RPA ----------------------------------------------------------------

RPA_ENV = {
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_FUSED_SPARSE_DECODE": "0",
}
DCP = 2
RPA_POOL = 512
Q_HEADS = 6  # the DCP pair's gathered heads attend on every rank


def _rpa_inputs(rng, *, requests, rows, ctx):
    """``requests`` x 4 verify rows padded to ``rows``, each row with its own
    512-group selection (random or clustered), DCP-2 main tables."""
    cols = -(-ctx[1] // PAGE) + 4
    contexts = rng.integers(ctx[0], ctx[1], size=requests)
    table = np.zeros((rows, cols), np.int32)
    lengths = np.zeros((rows,), np.int32)
    blocks = np.full((rows, TOPK_GROUPS), -1, np.int32)
    perm = rng.permutation(RPA_POOL)
    for r in range(requests):
        n_pages = -(-(contexts[r] + 4) // PAGE)
        own = np.resize(np.roll(perm, -r * 37), n_pages)
        for t in range(4):
            i = r * 4 + t
            if i >= rows:
                break
            table[i, :n_pages] = own
            lengths[i] = contexts[r] + t + 1
            visible = lengths[i] // RATIO
            if visible <= TOPK_GROUPS:
                blocks[i, :visible] = np.arange(visible)
            elif r % 2:
                start = rng.integers(0, visible - TOPK_GROUPS + 1)
                blocks[i] = np.sort(start + rng.choice(TOPK_GROUPS, TOPK_GROUPS, False))
            else:
                blocks[i] = rng.choice(visible, TOPK_GROUPS, replace=False)
    return jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


def _rpa_cache(seed):
    keys = jax.random.normal(jax.random.key(seed), (RPA_POOL, PAGE, 1, 4, D), jnp.bfloat16)
    return keys.astype(jnp.float8_e4m3fn)


def _attend(q, g, b, lens, t, cache, offset, rank):
    return qsa_rpa.decode_attention(
        q, g, b, lens, t, cache, kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5,
        **({"dcp": (rank, DCP), "return_lse": True} if rank is not None else {}),
        **({"kv_head_offset": 1} if offset else {}),
    )


RPA_CASES = [(2, 8, (2000, 9000)), (6, 24, (30000, 60000)), (24, 128, (95000, 110000)),
             (32, 128, (2000, 200000)), (17, 96, (150000, 200000))]


@pytest.mark.parametrize("requests,rows,ctx", RPA_CASES)
def test_rpa_flags_same_bits(monkeypatch, requests, rows, ctx):
    for k, v in RPA_ENV.items():
        monkeypatch.setenv(k, v)
    rng = np.random.default_rng(requests * 7 + rows)
    cache = _rpa_cache(1)
    blocks, lengths, table = _rpa_inputs(rng, requests=requests, rows=rows, ctx=ctx)
    q = jnp.asarray(rng.standard_normal((rows, Q_HEADS, D)), jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((rows, Q_HEADS, D)), jnp.bfloat16)
    arms = {"off": (0, 0), "skip16": (16, 0), "skip8": (8, 0), "hbm": (0, 1),
            "both": (16, 1)}
    outs = {}
    for name, (skip, hbm) in arms.items():
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY", str(skip))
        monkeypatch.setenv("VLLM_TPU_RPA_SCHEDULE_HBM", str(hbm))
        jax.clear_caches()
        for offset in (0, 1):
            fn = jax.jit(functools.partial(_attend, offset=offset))
            for rank in (None, *range(DCP)):
                r = None if rank is None else jnp.asarray([rank], jnp.int32)
                res = fn(q, g, blocks, lengths, table, cache, rank=r)
                outs[(name, offset, rank)] = [
                    np.asarray(x, np.float32).tobytes() for x in res[:-1]
                ]
    report = {}
    for name in arms:
        if name == "off":
            continue
        diff = [k for k in outs if k[0] == name and outs[k] != outs[("off", *k[1:])]]
        report[name] = diff
    print("rpa", requests, rows, ctx, report)
    assert all(not v for v in report.values()), report
    assert any(np.frombuffer(v[0], np.float32).any() for v in outs.values())


@pytest.mark.parametrize("requests,rows,ctx", RPA_CASES[:3])
@pytest.mark.parametrize("slot_read", ["0", "1"])
def test_compact_schedule_fetch_paths(monkeypatch, requests, rows, ctx, slot_read):
    """The compact schedule (one entry per lane-tile) with LEAN_FETCH off, as in
    the c1/c4/c8 serve envs: the same bits as with the lean and fast fetches,
    and the values of the plain per-page schedule and fetch (other KV tiles,
    so another summation order)."""
    for k, v in {**RPA_ENV, "VLLM_TPU_QSA_DECODE_SLOT_READ": slot_read}.items():
        monkeypatch.setenv(k, v)
    rng = np.random.default_rng(requests * 11 + rows)
    cache = _rpa_cache(2)
    blocks, lengths, table = _rpa_inputs(rng, requests=requests, rows=rows, ctx=ctx)
    q = jnp.asarray(rng.standard_normal((rows, Q_HEADS, D)), jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((rows, Q_HEADS, D)), jnp.bfloat16)
    # (compact, lean, fast, skip)
    arms = {
        "plain": ("0", "0", "0", "0"),
        "fast": ("1", "1", "1", "0"),
        "lean": ("1", "1", "0", "0"),
        "per_page": ("1", "0", "0", "0"),
        "per_page_skip": ("1", "0", "0", "16"),
    }
    outs = {}
    for name, (compact, lean, fast, skip) in arms.items():
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE", compact)
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_LEAN_FETCH", lean)
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_FAST_FETCH", fast)
        monkeypatch.setenv("VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY", skip)
        jax.clear_caches()
        res = jax.jit(functools.partial(_attend, offset=1, rank=None))(
            q, g, blocks, lengths, table, cache
        )
        outs[name] = np.asarray(res[0], np.float32)
    fast = outs["fast"].tobytes()
    diff = [n for n in arms if n != "plain" and outs[n].tobytes() != fast]
    print("compact fetch paths", requests, rows, ctx, slot_read, diff)
    assert not diff, diff
    assert outs["fast"].any()
    # BF16-level agreement: another KV tile order rounds differently (~1 BF16 ulp).
    np.testing.assert_allclose(outs["per_page"], outs["plain"], rtol=2e-2, atol=4e-3)


# --- decode indexer ------------------------------------------------------------

SERVED_DECODE_ENV = {
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": str(KV_BLOCK),
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
    "VLLM_TPU_QSA_INDEXER_HOIST_META": "1",
    "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": "1",
}

# (context, query rows) per request: 4 rows at served contexts, a request with
# fewer valid groups than the top-k, a 1-row request, an idle seat, and
# contexts whose keys end exactly on a 3,328-key block boundary or one tail
# group past it.
REQUESTS = (
    [(2048 + 37 * i, 4) for i in range(6)]
    + [(32768 + 811 * i, 4) for i in range(6)]
    + [(104000 + 1499 * i, 4) for i in range(6)]
    + [(190000 + 2333 * i, 4) for i in range(5)]
    + [(700, 4), (65000, 1), (50000, 0), (150000, 4)]
    + [(4 * 6656 - 4, 4), (4 * (6656 + 256) - 4, 4), (4 * (6656 + 832) - 4, 4)]
)
ZERO_QUERY_REQUESTS = (1, 7, 13, 19)


@functools.lru_cache(maxsize=2)
def _pool(seed, zero_keys):
    """FP8 unified pool [P, 512, 1, 4, 256] holding BF16 keys (finite, some
    exact +-0); ``zero_keys`` zeroes 7 of 8 kernel pages."""
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    pages = POOL_PAGES * 32
    keys = jax.random.normal(k1, (pages, 16, 2, 256), jnp.float32)
    keys = jnp.where(jax.random.uniform(k2, keys.shape) < 0.01, -0.0, keys)
    if zero_keys:
        keep = jax.random.uniform(k3, (pages, 1, 1, 1)) < 1 / 8
        keys = jnp.where(keep, keys, 0.0)
    bits = jax.lax.bitcast_convert_type(keys.astype(jnp.bfloat16), jnp.uint16)
    lo = (bits & 0xFF).astype(jnp.uint8)
    hi = (bits >> 8).astype(jnp.uint8)
    raw = jnp.stack([lo, hi], axis=3).reshape(pages, 16, 4, 256)
    raw = jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)
    return raw.reshape(POOL_PAGES, PAGE, 1, 4, 256)


def _step(rng, zero_query):
    """One verify step's operands; table entries past a request's pages are
    the null block 0, as in vLLM's block tables."""
    q_lens = np.array([r[1] for r in REQUESTS], np.int32)
    context = np.array([r[0] for r in REQUESTS], np.int32)
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cols = MAX_MODEL_LEN // PAGE
    table = np.stack(
        [rng.choice(np.arange(1, POOL_PAGES), size=cols, replace=False) for _ in REQUESTS]
    ).astype(np.int32)
    for r, (ctx, q_len) in enumerate(REQUESTS):
        table[r, -(-(ctx + q_len) // PAGE):] = 0
    query = rng.standard_normal((int(starts[-1]), 4, 128)).astype(np.float32)
    if zero_query:
        for r in ZERO_QUERY_REQUESTS:
            query[starts[r] : starts[r + 1]] = 0
    return dict(
        query=jnp.asarray(query, jnp.bfloat16),
        lengths=jnp.asarray(context + q_lens),
        table=jnp.asarray(table),
        req=jnp.arange(len(REQUESTS), dtype=jnp.int32),
        starts=jnp.asarray(starts),
        context=jnp.asarray(context),
    )


def _written(step):
    """[rows, 1] count of the score words the kernel writes per row (every
    block of the row's request) and each request's compressed keys."""
    starts = np.asarray(step["starts"])
    kv = (np.asarray(step["context"]) + np.diff(starts)) // RATIO
    blocks = np.maximum(1, -(-kv // KV_BLOCK))
    out = np.zeros((int(starts[-1]), 1), np.int64)
    for r in range(len(kv)):
        out[starts[r] : starts[r + 1]] = blocks[r] * KV_BLOCK
    return out, kv


class _ScoresOnly:
    """Selector that returns the score buffer itself (k = its width)."""

    overlaps_scoring = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        return (score_bits, score_bits) if return_scores else score_bits

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, _ScoresOnly)


def _kernel_scores(step, pool, dense, tail, weights=None):
    """The scoring kernel's whole output buffer for one window, called the way
    pooled_qsa._select_blocks calls it on the pool-page table."""
    table = np.asarray(step["table"])
    expansion = (4, 32, SLOT * 4)
    width_pages = table.shape[1] * 4
    starts = np.asarray(step["starts"])
    seq = np.asarray(step["context"]) + np.diff(starts)
    rows = int(starts[-1])
    if weights is None:
        weights = np.full((rows, 4), 128**-0.5, np.float32)
    fn = functools.partial(
        sit.streamindex_topk,
        k=width_pages * KEYS_PER_PAGE,
        compression_ratio=RATIO,
        num_kv_pages_per_block=(8, KV_BLOCK // KEYS_PER_PAGE, KV_BLOCK // KEYS_PER_PAGE),
        num_queries_per_block=(1, 4, 4),
        decode_req_batch_size=8,
        components=qsa_indexing(
            RATIO - 1, per_token_scores=True, table_expansion=expansion,
            coalesce_entries=True, dense_step=bool(dense), tail_skip=tail,
        ),
        decode_passes=False,
        topk_backend=_ScoresOnly(),
    )
    return np.asarray(jax.jit(fn)(
        step["query"], jnp.asarray(weights), pool.reshape(-1, KEYS_PER_PAGE // 2, 4, 256),
        jnp.asarray(seq), jnp.asarray(table.reshape(-1)), jnp.asarray(starts),
        jnp.asarray([0, 0, len(seq)], jnp.int32),
    ))


def _select(monkeypatch, step, pool, dense, tail):
    """select_blocks on the served path (window, kernel, SparseCore top-k)."""
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_DENSE_STEP", str(dense))
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_TAIL_SKIP", str(tail))

    def run(query, pool, lengths, table, req, starts, context):
        return pooled_qsa.select_blocks(
            query, pool, lengths, table, req, starts, context, ratio=RATIO,
            block_topk=TOPK_GROUPS, slot=SLOT, max_query_len=4,
        )

    return np.asarray(jax.jit(run)(pool=pool, **step))


@pytest.mark.parametrize("dense,tail", [(1, 0), (0, 8), (1, 8), (0, 26)])
@pytest.mark.parametrize("variant", ["random", "zero_keys", "zero_query"])
def test_indexer_flags_same_scores(monkeypatch, variant, dense, tail):
    for name, value in SERVED_DECODE_ENV.items():
        monkeypatch.setenv(name, value)
    rng = np.random.default_rng(7)
    pool = _pool(11, variant == "zero_keys")
    step = _step(rng, variant == "zero_query")
    base_scores = _kernel_scores(step, pool, 0, 0)
    flag_scores = _kernel_scores(step, pool, dense, tail)
    base = _select(monkeypatch, step, pool, 0, 0)
    flagged = _select(monkeypatch, step, pool, dense, tail)

    written, kv = _written(step)
    rows = len(written)
    region = np.arange(base_scores.shape[1])[None, :] < written
    diff = int(((base_scores[:rows] != flag_scores[:rows]) & region).sum())
    tied = 0
    for t in range(rows):
        valid = base_scores[t, : written[t, 0]].view(np.float32)
        valid = valid[np.isfinite(valid)]
        if len(valid) > TOPK_GROUPS:
            thr = np.sort(valid)[::-1][TOPK_GROUPS - 1]
            tied += int(np.sum(valid >= thr) > TOPK_GROUPS)
    report = {
        "written words": int(region.sum()),
        "differing words": diff,
        "selection equal (ordered)": bool(np.array_equal(base, flagged)),
        "rows with a tied top-k boundary": tied,
        "requests with fewer than k valid groups": int(
            np.sum((kv < TOPK_GROUPS) & (np.diff(np.asarray(step["starts"])) > 0))
        ),
    }
    print("indexer", variant, f"dense {dense} tail {tail}", report)
    assert report["written words"] > 0
    assert diff == 0, report
    assert report["selection equal (ordered)"], report
    if variant != "random":
        assert tied > 0, report


def test_signed_weights():
    """Random-sign weights and zero queries, straight on the kernel: head
    terms of -0.0 and negative scores."""
    rng = np.random.default_rng(3)
    pool = _pool(5, False)
    step = _step(rng, True)
    rows = int(step["starts"][-1])
    weights = rng.standard_normal((rows, 4)).astype(np.float32)
    weights[rng.random(weights.shape) < 0.2] = 0.0
    weights[: rows // 2] = -np.abs(weights[: rows // 2])
    base = _kernel_scores(step, pool, 0, 0, weights)
    flagged = _kernel_scores(step, pool, 1, 8, weights)
    written, _ = _written(step)
    region = np.arange(base.shape[1])[None, :] < written
    diff = int(((base[:rows] != flagged[:rows]) & region).sum())
    vals = base[:rows][region].view(np.float32)
    report = {
        "written words": int(region.sum()),
        "differing words": diff,
        "negative scores": int((vals < 0).sum()),
        "zero scores": int((vals == 0).sum()),
    }
    print("signed", report)
    assert diff == 0, report
    assert report["negative scores"] > 0 and report["zero scores"] > 0
