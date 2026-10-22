# SPDX-License-Identifier: Apache-2.0
"""Row-selected pooled QSA (MTP draft step 0) against the all-token step.

The rows variant must write exactly the same pool (KV, compressed keys, ring
state) and return the all-token result at the requested rows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.pool import write_region

OPTIONS = dict(compress_ratio=4, token_topk=2048, norm_eps=1e-6, rotary_dim=64)


def _inputs(fp8, starts, context, seed=0):
    rng = np.random.default_rng(seed)

    def rand(shape):
        return jnp.asarray(rng.normal(0, 0.3, shape), jnp.bfloat16)

    requests = len(context)
    tokens = 8
    packing = 4 if fp8 else 2
    carrier = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    pool = jnp.zeros((12, 1664, 4 // packing, packing, 256), carrier)
    ring_ids = jnp.array([3, 6, 0], jnp.int32)[:requests]
    ring = rand((requests, 8, 128))
    tags = jnp.zeros((requests, 8, 2), jnp.int32)
    for row in range(requests):
        if context[row]:
            tags = tags.at[row, :4, 0].set(
                context[row] - 1 - (context[row] - 1 - jnp.arange(4)) % 4
            )
    if jax.default_backend() != "cpu":
        write = jax.jit(write_region, static_argnames=("tok0",))
        pool = write(pool, ring.reshape(requests, -1), ring_ids)
        token_bytes = np.prod(pool.shape[2:]) * pool.dtype.itemsize
        tok0 = 8 * 128 * 2 // token_bytes
        pool = write(pool, tags.reshape(requests, -1), ring_ids, tok0=tok0)
    context = jnp.asarray(context, jnp.int32)
    starts = jnp.asarray(starts, jnp.int32)
    req = np.minimum(
        np.searchsorted(np.asarray(starts[1:]), np.arange(tokens), side="right"),
        requests - 1,
    )
    positions = jnp.asarray(
        np.where(
            np.arange(tokens) < int(starts[-1]),
            np.asarray(context)[req] + np.arange(tokens) - np.asarray(starts)[req],
            0,
        ),
        jnp.int32,
    )
    angle = jnp.arange(4096)[:, None] * jnp.arange(32)[None, :] / 8192
    trig = jnp.pad(
        jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], 1), ((0, 0), (0, 64))
    ).astype(jnp.bfloat16)
    main = jnp.array([[2, 7], [5, 9], [0, 0]], jnp.int32)[:requests]
    comp = jnp.array([[1, 8], [4, 10], [0, 0]], jnp.int32)[:requests]
    head = (
        rand((tokens, 24 * 256 * 2)),
        rand((tokens, 2 * 256)),
        rand((tokens, 2 * 256)),
        rand((tokens, 5 * 128)),
        rand((256,)),
        rand((256,)),
        rand((128,)),
        rand((128,)),
        positions,
        trig,
    )
    tail = (
        starts,
        context,
        jnp.pad(ring_ids[:, None], ((0, 0), (0, 31))).reshape(-1),
        main,
        comp,
    )
    return head, pool, tail


def _run_both(fp8, starts, context, rows, **kwargs):
    head, pool, tail = _inputs(fp8, starts, context)
    options = dict(OPTIONS, **kwargs)
    # Fresh closures: retrace per test, under that test's patches.
    full = jax.jit(lambda *args: pooled_qsa.pooled_qsa_step(*args, **options))
    picked = jax.jit(lambda *args: pooled_qsa.pooled_qsa_rows_step(*args, **options))
    full_pool, full_out, full_selected = full(*head, pool, *tail)
    rows = jnp.asarray(rows, jnp.int32)
    rows_pool, rows_out, rows_selected = picked(*head, pool, *tail, rows)
    keep = np.asarray(rows) >= 0
    index = np.maximum(np.asarray(rows), 0)
    want_out = np.where(keep[:, None], np.asarray(full_out, np.float32)[index], 0)
    want_selected = np.where(keep[:, None], np.asarray(full_selected)[index], -1)
    np.testing.assert_array_equal(
        np.asarray(rows_pool).view(np.uint8), np.asarray(full_pool).view(np.uint8)
    )
    return np.asarray(rows_out, np.float32), want_out, rows_selected, want_selected


@pytest.fixture
def rowwise_attention(monkeypatch):
    """CPU: pure-JAX stand-ins for the pool DMA kernels; row-wise attention.

    The stand-in main-cache write stores every valid token's packed KV, so
    the pool comparison checks that the rows step still writes all tokens.
    """
    attended = []

    def write_main_cache(pool, rows, page, logical, valid, **_):
        return pool.at[page, logical % pool.shape[1]].set(
            rows.astype(pool.dtype), mode="drop"
        )

    def read_ring_and_tags(pool, ids, ring_elements, tag_elements):
        return (
            jnp.zeros((len(ids), ring_elements), jnp.bfloat16),
            jnp.zeros((len(ids), tag_elements), jnp.int32),
        )

    def write_compressed(pool, completed, complete, req, logical, *args):
        value = jnp.sum(jnp.where(complete[:, None], completed, 0))
        return pool.at[11, 0].set(value.astype(pool.dtype))

    def write_ring_and_tags(pool, ring, tags, ids):
        return pool.at[11, 1].set(jnp.sum(ring).astype(pool.dtype))

    def select_blocks(query, pool, lengths, table, req, starts, context, **kw):
        # The per-token indexer reads each row's request from `req` and its
        # causal length from `lengths`; the shared-stream one derives both
        # from the ragged batch `starts`/`context`. Any disagreement between
        # the two descriptions shifts the stand-in's blocks.
        score = jnp.sum(query.astype(jnp.float32), axis=(1, 2)) > 0
        token = jnp.arange(len(query), dtype=jnp.int32)
        owner = jnp.searchsorted(starts[1:], token, side="right")
        owner = jnp.minimum(owner, len(context) - 1)
        derived = context[owner] + token - starts[owner] + 1
        mismatch = (derived != lengths).astype(jnp.int32) * 100 + (
            table[owner, 0] != table[req, 0]
        ).astype(jnp.int32) * 200
        base = lengths // kw["ratio"] + 7 * table[req, 0] + score.astype(jnp.int32)
        base = base + mismatch
        blocks = base[:, None] + jnp.arange(kw["block_topk"], dtype=jnp.int32)
        return jnp.where((lengths > 0)[:, None], blocks % 509, -1)

    def qsa_attention(q, gate, blocks, lengths, table, cache, starts, **kw):
        attended.append(len(q))
        value = q.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
        value = value + (blocks[:, :1, None] + table[:, :1, None] + lengths[:, None, None])
        return jnp.where((lengths > 0)[:, None, None], value, 0), cache

    for name, fake in (
        ("write_main_cache", write_main_cache),
        ("read_ring_and_tags", read_ring_and_tags),
        ("write_compressed", write_compressed),
        ("write_ring_and_tags", write_ring_and_tags),
        ("select_blocks", select_blocks),
        ("qsa_attention", qsa_attention),
    ):
        monkeypatch.setattr(pooled_qsa, name, fake)
    return attended


# Two requests: a 3-token verify window, a 4-token prefill chunk, padding.
STARTS, CONTEXT = [0, 3, 7, 7], [1661, 1663, 0]


@pytest.mark.parametrize(
    "rows,attended",
    [
        # Each request's last token plus a padded slot: the decode path.
        ([2, 6, -1], 3),
        # Repeated filler rows, as the proposer pads its loop bucket.
        ([1, 6, 6], 3),
        # More rows than request slots: attend every token, then pick.
        ([2, 6, 6, 6], 8),
    ],
)
@pytest.mark.parametrize("fp8", [False, True])
def test_rows_step_writes_same_pool_and_picks_rows(
    rowwise_attention, fp8, rows, attended
):
    out, want, selected, want_selected = _run_both(fp8, STARTS, CONTEXT, rows)
    np.testing.assert_array_equal(out, want)
    np.testing.assert_array_equal(np.asarray(selected), want_selected)
    # Traced once per program: full step first, then the rows step.
    assert rowwise_attention == [8, attended]


# Three verify windows (K=2) of different lengths filling all request slots.
VERIFY_STARTS, VERIFY_CONTEXT = [0, 3, 6, 8], [1661, 1000, 1663]


@pytest.mark.parametrize("rows", [[2, 5, 7], [2, 5, -1]])
def test_rows_present_their_own_decode_batch(rowwise_attention, rows):
    """Row i is request i's last token, not token i of the step.

    Regression: the rows kept the step's token-level starts/context, which
    the shared-stream indexer (VLLM_TPU_QSA_DECODE_QUERY_BLOCK) uses to map
    query rows to requests. Row 1 then scored as request 0's second token,
    so drafts of every request but the first collapsed under concurrency.
    """
    out, want, selected, want_selected = _run_both(
        False, VERIFY_STARTS, VERIFY_CONTEXT, rows
    )
    np.testing.assert_array_equal(np.asarray(selected), want_selected)
    np.testing.assert_array_equal(out, want)
    assert rowwise_attention == [8, 3]


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [[2, 6, -1], [2, 6, 6, 6]])
@pytest.mark.parametrize("fp8", [False, True])
def test_rows_step_matches_full_step_on_tpu(fp8, rows):
    kwargs = dict(k_scale=0.375, v_scale=2.5) if fp8 else {}
    out, want, selected, want_selected = _run_both(
        fp8, STARTS, CONTEXT, rows, **kwargs
    )
    # The rows take RPA where the all-token step takes block-sparse prefill.
    np.testing.assert_allclose(out, want, rtol=0.05, atol=0.03)
    for got, expected in zip(np.asarray(selected), want_selected):
        np.testing.assert_array_equal(np.sort(got), np.sort(expected))
