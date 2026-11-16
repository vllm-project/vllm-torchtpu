# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_INDEXER_DENSE_STEP and VLLM_TPU_QSA_INDEXER_TAIL_SKIP on TPU
(skipped elsewhere).

The served decode indexer (StreamIdxTC mixed pass, per-token scores, 4 query
rows a block, 3,328-key blocks) with the flag off and on, on the same inputs,
at the served geometry: FP8 unified pool of 512-token pages read as BF16 keys
256 wide, max_model_len 262144, the replicated call on the pool-page table
and the compressed-DCP call (VLLM_TPU_QSA_COMPRESSED_DCP) on every rank of 4, on
the pool-page table (VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE, the served layout) and,
for ranks 0 and 3, on the expanded kernel-page table.
The checks:

- the scoring kernel's output words (f32 score bits) are identical over every
  position the kernel writes (each row's blocks), with the kernel called as
  pooled_qsa calls it for one window;
- ``select_blocks`` on the served path (window, kernel, SparseCore top-k)
  returns identical selections (under DCP, identical local score bits and
  global group ids).

Contexts 2k / 32k / 104k / 190k plus a 700-token request (fewer valid groups
than the top-k on every rank), a 1-row request, an idle seat, and contexts that
end on a block boundary or one tail group past it; table entries past a
request's pages are the null block 0, as in vLLM's block tables. ``zero_keys``
zeroes 7 of 8 compressed-key kernel pages, so most groups score exactly 0 and
the top-k boundary falls inside ties; ``zero_query`` zeroes four requests'
index queries, so all their groups tie. ``test_signed_weights`` calls the
kernel with random-sign indexer weights and zero queries, where head terms
are -0.0. ``test_tail_skip_same_scores`` checks the tail skip (page groups
of 8 and 26, with and without the dense step) the same way.

Run it on a TPU host with QSA_INDEXER_TEST_DEVICE=tpu, e.g. through the bench
recipe: ``BENCH=script "BENCH_CMD=env QSA_INDEXER_TEST_DEVICE=tpu python -m
pytest -q -s tests/kernels/test_qwen4_exp_indexer_dense_step.py"``.
"""

import dataclasses
import functools
import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# An environment switch, not a backend probe: collection must not initialize
# JAX (tests/conftest.py). The BF16 key reader does not run in interpret mode.
ON_TPU = os.environ.get("QSA_INDEXER_TEST_DEVICE", "cpu").lower() == "tpu"
pytestmark = pytest.mark.skipif(not ON_TPU, reason="QSA_INDEXER_TEST_DEVICE=tpu only")

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing  # noqa: E402

sit = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)

RATIO, TOPK_GROUPS, PAGE, DCP, SLOT = 4, 512, 512, 4, 3
KV_BLOCK, KEYS_PER_PAGE = 3328, 32
MAX_MODEL_LEN = 262144
POOL_PAGES = 4096
SERVED_DECODE_ENV = {
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": str(KV_BLOCK),
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
}


def _context_for_local(keys, rank):
    """A context whose rank-`rank` share of the compressed keys (512-token
    pool pages dealt round-robin over 4 ranks, 128 keys each) is `keys`."""
    kv = keys // 128 * 512 + rank * 128 + keys % 128
    return 4 * kv - 4  # the step's 4 rows make seq_len = 4 * kv


# (context, query rows) per request: 4 rows of each served context, a request
# with fewer valid groups than the top-k, a 1-row request, an idle seat, and
# contexts whose keys end exactly on a 3,328-key block boundary or one tail
# group (8 or 26 pages of 32 keys) past it, for DCP ranks 0 and 3 and for the
# replicated call.
REQUESTS = (
    [(2048 + 37 * i, 4) for i in range(6)]
    + [(32768 + 811 * i, 4) for i in range(6)]
    + [(104000 + 1499 * i, 4) for i in range(6)]
    + [(190000 + 2333 * i, 4) for i in range(5)]
    + [(700, 4), (65000, 1), (50000, 0), (150000, 4)]
    + [(_context_for_local(6656, 0), 4), (_context_for_local(6656 + 256, 0), 4),
       (_context_for_local(6656, 3), 4), (_context_for_local(6656 + 832, 3), 4),
       (4 * 6656 - 4, 4), (4 * (6656 + 256) - 4, 4)]
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


def _step(rng, dcp, zero_query):
    """One verify step's operands: queries, compressed table, starts, context."""
    q_lens = np.array([r[1] for r in REQUESTS], np.int32)
    context = np.array([r[0] for r in REQUESTS], np.int32)
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    span = PAGE * (DCP if dcp else 1)
    cols = MAX_MODEL_LEN // span
    table = np.stack(
        [rng.choice(np.arange(1, POOL_PAGES), size=cols, replace=False) for _ in REQUESTS]
    ).astype(np.int32)
    for r, (ctx, q_len) in enumerate(REQUESTS):
        table[r, -(-(ctx + q_len) // span):] = 0
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


def _written(step, rank):
    """[rows, positions] mask of the score words the kernel writes: every
    block of the row's request."""
    starts = np.asarray(step["starts"])
    seq = np.asarray(step["context"]) + np.diff(starts)
    kv = seq // RATIO
    if rank is not None:
        ic = PAGE // RATIO
        full, rem = kv // (DCP * ic), kv % (DCP * ic)
        kv = full * ic + np.clip(rem - rank * ic, 0, ic)
    blocks = np.maximum(1, -(-kv // KV_BLOCK))
    rows = int(starts[-1])
    out = np.zeros((rows, 1), np.int64)
    for r in range(len(seq)):
        out[starts[r] : starts[r + 1]] = blocks[r] * KV_BLOCK
    return out, kv


@dataclasses.dataclass(frozen=True)
class _ScoresOnly:
    """Selector that returns the score buffer itself (k = its width)."""

    overlaps_scoring = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        return (score_bits, score_bits) if return_scores else score_bits


def _kernel_scores(step, pool, layout, flag, tail, weights=None):
    """The scoring kernel's whole output buffer for one window, called the way
    pooled_qsa._select_blocks / _dcp_topk call it at the served settings."""
    rank, pool_table = layout
    table = np.asarray(step["table"])
    if rank is None or pool_table:
        # Pool-page table: the reader expands each entry into its 4 kernel pages.
        align = KV_BLOCK // KEYS_PER_PAGE // 4 if rank is not None else 1
        pages = np.pad(table, ((0, 0), (0, (-table.shape[1]) % align)))
        expansion = pooled_qsa._compressed_expansion(4, 32, SLOT)
        width_pages = pages.shape[1] * 4
    else:
        local = (table[..., None] * 32 + SLOT * 4 + np.arange(4)).reshape(len(table), -1)
        pages = np.pad(local, ((0, 0), (0, (-local.shape[1]) % (KV_BLOCK // KEYS_PER_PAGE))))
        expansion = (1, 1, 0)
        width_pages = pages.shape[1]
    extra = (
        {} if rank is None
        else dict(cp_size=DCP, cp_rank=jnp.int32(rank), interleave_size=PAGE)
    )
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
            coalesce_entries=True, dense_step=bool(flag), tail_skip=tail,
        ),
        decode_passes=False,
        topk_backend=_ScoresOnly(),
        **extra,
    )
    return np.asarray(jax.jit(fn)(
        step["query"], jnp.asarray(weights), pool.reshape(-1, KEYS_PER_PAGE // 2, 4, 256),
        jnp.asarray(seq), jnp.asarray(pages.reshape(-1)), jnp.asarray(starts),
        jnp.asarray([0, 0, len(seq)], jnp.int32),
    ))


def _select(monkeypatch, step, pool, rank, flag, tail=0):
    """select_blocks on the served path (window, kernel, SparseCore top-k)."""
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_DENSE_STEP", str(flag))
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_TAIL_SKIP", str(tail))
    dcp = None if rank is None else (jnp.asarray([rank], jnp.int32), DCP)

    def run(query, pool, lengths, table, req, starts, context):
        return pooled_qsa.select_blocks(
            query, pool, lengths, table, req, starts, context, ratio=RATIO,
            block_topk=TOPK_GROUPS, slot=SLOT, max_query_len=4, dcp=dcp,
        )

    return jax.tree.map(np.asarray, jax.jit(run)(pool=pool, **step))


def _bits(x):
    x = np.asarray(x)
    return x.view(np.uint32) if x.dtype.itemsize == 4 else x


# (DCP rank or None for the replicated call, DCP pool-page table)
LAYOUTS = [(None, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 0), (3, 0)]
LAYOUT_IDS = ["rep", "dcp0", "dcp1", "dcp2", "dcp3", "dcp0-pages", "dcp3-pages"]


@pytest.mark.parametrize("variant", ["random", "zero_keys", "zero_query"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_dense_step_same_scores(monkeypatch, variant, layout):
    _compare(monkeypatch, variant, layout, 1, 0)


@pytest.mark.parametrize("flag,tail", [(1, 26), (0, 8)])
@pytest.mark.parametrize("variant", ["random", "zero_keys"])
@pytest.mark.parametrize("layout", [LAYOUTS[i] for i in (0, 1, 4, 5)],
                         ids=[LAYOUT_IDS[i] for i in (0, 1, 4, 5)])
def test_tail_skip_same_scores(monkeypatch, variant, layout, flag, tail):
    _compare(monkeypatch, variant, layout, flag, tail)


def _compare(monkeypatch, variant, layout, flag, tail):
    rank, pool_table = layout
    for name, value in SERVED_DECODE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP", "0" if rank is None else "1")
    monkeypatch.setenv("VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE", str(pool_table))
    rng = np.random.default_rng(7 + (rank or 0))
    pool = _pool(11, variant == "zero_keys")
    step = _step(rng, rank is not None, variant == "zero_query")
    base_scores = _kernel_scores(step, pool, layout, 0, 0)
    dense_scores = _kernel_scores(step, pool, layout, flag, tail)
    base = _select(monkeypatch, step, pool, rank, 0)
    dense = _select(monkeypatch, step, pool, rank, flag, tail)

    written, kv = _written(step, rank)
    rows = len(written)
    cols = np.arange(base_scores.shape[1])[None, :]
    region = cols < written
    words = int(region.sum())
    diff = int(((base_scores[:rows] != dense_scores[:rows]) & region).sum())
    flat_base = base_scores[:rows][region].view(np.float32)
    report = {
        "rows": rows,
        "written words": words,
        "differing words": diff,
        "-inf words": int(np.isneginf(flat_base).sum()),
        "zero words": int((flat_base == 0).sum()),
        "requests with fewer than k valid groups": int(
            np.sum((kv < TOPK_GROUPS) & (np.diff(np.asarray(step["starts"])) > 0))
        ),
    }
    # Rows whose k-th valid score ties across the cut.
    tied = 0
    for t in range(rows):
        valid = base_scores[t, : written[t, 0]].view(np.float32)
        valid = valid[np.isfinite(valid)]
        if len(valid) > TOPK_GROUPS:
            thr = np.sort(valid)[::-1][TOPK_GROUPS - 1]
            tied += int(np.sum(valid >= thr) > TOPK_GROUPS)
    report["rows with a tied top-k boundary"] = tied
    if rank is None:
        same = np.array_equal(np.sort(base, -1), np.sort(dense, -1))
        report["selection equal"] = same
    else:
        same = all(
            np.array_equal(_bits(a), _bits(b)) for a, b in zip(base, dense)
        )
        report["local vals/pos bit-equal"] = same
    print(variant, layout, f"dense {flag} tail {tail}", report)
    assert words > 0
    assert diff == 0, report
    assert same, report
    assert report["requests with fewer than k valid groups"] > 0
    if variant != "random":
        assert tied > 0, report


@pytest.mark.parametrize("rank", [None, 0, 3])
def test_signed_weights(rank):
    """Random-sign weights and zero queries, straight on the kernel: head
    terms of -0.0 and negative scores."""
    rng = np.random.default_rng(3)
    pool = _pool(5, False)
    step = _step(rng, rank is not None, True)
    rows = int(step["starts"][-1])
    weights = rng.standard_normal((rows, 4)).astype(np.float32)
    weights[rng.random(weights.shape) < 0.2] = 0.0
    weights[: rows // 2] = -np.abs(weights[: rows // 2])
    layout = (rank, 1 if rank == 0 else 0)
    base = _kernel_scores(step, pool, layout, 0, 0, weights)
    dense = _kernel_scores(step, pool, layout, 1, 0, weights)
    written, _ = _written(step, rank)
    region = np.arange(base.shape[1])[None, :] < written
    diff = int(((base[:rows] != dense[:rows]) & region).sum())
    vals = base[:rows][region].view(np.float32)
    report = {
        "written words": int(region.sum()),
        "differing words": diff,
        "negative scores": int((vals < 0).sum()),
        "zero scores": int((vals == 0).sum()),
    }
    print("signed", rank, report)
    assert diff == 0, report
    assert report["negative scores"] > 0 and report["zero scores"] > 0
