# SPDX-License-Identifier: Apache-2.0
"""q38 O1s: request-sharded decode selection under main-KV DCP
(VLLM_TPU_QSA_DECODE_REQUEST_SHARD) against the replicated-selection DCP step.

O1 (incumbent): every TP rank runs ``pooled_qsa_dcp_step``, which scores and
selects every row of the step. O1s: TP rank ``s`` runs
``pooled_qsa_request_select_dcp_step`` for its ``request_shard_width`` rows
only; the ``[TP, T / TP, k]`` blocks are gathered in rank order (the TP
all-gather) and ``pooled_qsa_request_attend_dcp_step`` rebuilds the step's
selection and attends the DCP rank's owned main KV.

The StreamIndex kernel cannot run on CPU. ``streamindex_topk`` is replaced by
``standin_topk``, a per-token reference with the kernel's contract: it reads
each visible compressed key from the cache through the page table (with the
pool-page table expansion), scores it against that token's query, honours the
causal limit from ``seq_lens`` / ``cu_q_lens`` and the decode schedule's
position offset, and returns the top-k set. Everything else is production
code: the decode branch of ``select_blocks`` (shared-stream request windows,
pool-page table, MIXED_ONLY distribution), the cache writes, the DCP page
metadata and RPA routing (with the CPU RPA stand-in). The stand-in returns
the set in an order that depends on the batch (ascending, or reversed on
batches of an odd number of 8-row blocks), as the SparseCore top-k's order
is unspecified; the attention must not care.

Checks per case and DCP rank: pools byte-equal after the step; the partial
attention (out, lse) bit-identical; ``selected`` equal per live row as a set
(and exactly, element for element, when the stand-in keeps one order);
engagement: every live row is scored exactly once across the TP shards and
no shard scores more than its balanced share of requests.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from dcp2_harness import (
    D,
    build_layout,
    from_raw,
    poison_tail_rows,
    project_rank_pool,
    random_fp8,
    rank_operand,
    raw,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

B, DH, H, INDEX_DIM, INDEX_HEADS, RATIO, ROTARY = 512, 256, 3, 128, 4, 4, 64
TOPK = 256  # tokens (64 groups); contexts above 256 tokens exercise real selection
K_SCALE, V_SCALE = 0.375, 2.5
TP = 8
ROWS = 4  # MTP k = 3 verify rows per request (max_query_len)
SCORE_BYTES = 16  # key bytes per score (keeps the [T, G, J] gather small on CPU)


class Recorder:
    """What the stand-in saw: (tokens, live rows, live sequences) per call."""

    def __init__(self):
        self.calls = []
        self.reverse = True


RECORDER = Recorder()
_ENVS_MISSING = object()
STANDIN_ENVS = ("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "VLLM_TPU_QSA_DECODE_KV_BLOCK", "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK",
                "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA",
                "VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY")


def standin_topk(q, weights, cache, seq_lens, page_indices, cu_q_lens, distribution, *, k,
                 compression_ratio, components=None, enable_early_exit=False, **_):
    tokens, seqs = q.shape[0], seq_lens.shape[0]
    lp, pp, base = components.table_expansion if components is not None else (1, 1, 0)
    table = page_indices.reshape(seqs, -1)
    if (lp, pp, base) != (1, 1, 0):
        table = (table[:, :, None] * pp + base + jnp.arange(lp)).reshape(seqs, -1)
    rows = 128 // compression_ratio
    keys = jax.lax.bitcast_convert_type(cache, jnp.uint8).reshape(cache.shape[0], rows, -1)
    keys = keys[..., :SCORE_BYTES].astype(jnp.float32)
    qv = q.astype(jnp.float32).reshape(tokens, -1)[:, :SCORE_BYTES] * weights[:, :1]
    live_seqs = distribution[2]
    t = jnp.arange(tokens, dtype=jnp.int32)
    s = jnp.clip(jnp.searchsorted(cu_q_lens[1:], t, side="right"), 0, seqs - 1)
    live = t < cu_q_lens[live_seqs]
    seq_len = seq_lens[s]
    q_len = cu_q_lens[s + 1] - cu_q_lens[s]
    q_abs = seq_len - q_len + (t - cu_q_lens[s])
    pos = components.schedule.positions(q_abs) if components is not None else q_abs
    max_valid = jnp.minimum(seq_len // compression_ratio - 1, pos // compression_ratio)
    max_valid = jnp.where(live, max_valid, -1)
    groups = table.shape[1] * rows
    g = jnp.arange(groups, dtype=jnp.int32)
    page = jnp.take_along_axis(table[s], jnp.broadcast_to(g // rows, (tokens, groups)), axis=1)
    key = keys[page, jnp.broadcast_to(g % rows, (tokens, groups))]  # [T, G, J]
    score = jnp.einsum("tgj,tj->tg", key, qv)
    visible = g[None, :] <= max_valid[:, None]
    score = jnp.where(visible, score, -jnp.inf)
    order = jnp.lexsort((jnp.broadcast_to(g, score.shape), -score), axis=-1)[:, :k]
    filled = jnp.take_along_axis(score, order, axis=1) > -jnp.inf
    big = jnp.iinfo(jnp.int32).max
    ascending = jnp.sort(jnp.where(filled, order, big), axis=1)
    ascending = jnp.where(ascending == big, -1, ascending).astype(jnp.int32)
    count = jnp.sum(filled, axis=1, keepdims=True)
    i = jnp.arange(k)[None, :]
    reversed_ = jnp.take_along_axis(ascending, jnp.where(i < count, count - 1 - i, i), axis=1)
    early = enable_early_exit & (jnp.max(seq_lens) // compression_ratio <= k)
    # O1's whole-step calls (> 32 rows) and O1s's shard calls (<= 32 rows)
    # then disagree on the order whenever the bucket exceeds 32 tokens.
    flip = RECORDER.reverse and tokens > 32
    out = jnp.where(early | (not flip), ascending, reversed_)
    # Host-side record (also inside the shared stream's lax.scan windows).
    jax.debug.callback(
        lambda rows_, seqs_: RECORDER.calls.append((tokens, int(rows_), int(seqs_))),
        cu_q_lens[live_seqs], live_seqs, ordered=True,
    )
    return out


@pytest.fixture
def standin(monkeypatch):
    # The stand-in reads RECORDER.reverse while tracing, and JAX's trace cache is keyed by the function and the
    # argument types, not by the jax.jit wrapper: a later case with the same shapes (exact_order and B12 share
    # 12 busy / 4 idle seats in a 48-token bucket) would reuse an earlier case's trace and its order switch.
    # Start every test from empty caches so each case traces with its own setting.
    jax.clear_caches()
    RECORDER.calls.clear()
    RECORDER.reverse = True
    monkeypatch.setattr(pooled_qsa, "streamindex_topk", standin_topk)
    saved = {}
    # The served decode indexer flags (FR7 / recipe): shared-stream verify
    # windows, pool-page table, mixed pass only, bounded SparseCore copy.
    for name, value in (
        ("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", 4),
        ("VLLM_TPU_QSA_DECODE_KV_BLOCK", 3328),
        ("VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", 1024),
        ("VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", True),
        ("VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA", True),
        ("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", True),
        ("VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY", True),
    ):
        # q38 L2 O1s-HOIST: set the module attribute directly and remove it on teardown. monkeypatch.setattr's undo
        # writes back the value envs.__getattr__ returned, leaving a frozen module attribute that shadows os.environ
        # for every later test in the session (a later setenv-only flag test then runs the default path; j-3307bbed).
        saved.setdefault(name, envs.__dict__.get(name, _ENVS_MISSING))
        envs.__dict__[name] = value
    yield RECORDER
    for name, old in saved.items():
        if old is _ENVS_MISSING:
            envs.__dict__.pop(name, None)
        else:
            envs.__dict__[name] = old


def make_step(seed, query_lens, contexts, tokens):
    """test_dcp2_op.make_step with the token bucket ``tokens`` (padding rows
    past the live tokens, as the runner's decode buckets)."""
    rng = np.random.default_rng(seed)
    query_lens, contexts = np.asarray(query_lens), np.asarray(contexts)
    requests = len(contexts)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    assert tokens >= starts[-1] and tokens % TP == 0
    seq = np.maximum(contexts + query_lens, 1)
    cols = int(-(-seq.max() // B))
    num_pages = int(sum(-(-seq // B)) * 2 + requests * (cols + 1) + 4)
    layout = build_layout(rng, seq, B, num_pages)
    free = iter(layout.free_pages)
    comp = np.array([[next(free) for _ in range(cols)] for _ in range(requests)], np.int32)
    ring = np.array([next(free) for _ in range(requests)], np.int32)
    keep = sorted({0, *comp.ravel().tolist(), *ring.tolist()})
    full = random_fp8(rng, (num_pages, B, 1, 4, DH))
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(query_lens[r])

    def normal(*shape, scale=1.0):
        return jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16)

    pair_qgate = normal(tokens, D * 2 * H * DH)
    shared = dict(
        key=normal(tokens, DH), value=normal(tokens, DH),
        index_qk=normal(tokens, (INDEX_HEADS + 1) * INDEX_DIM),
        nq=normal(DH, scale=0.1), nk=normal(DH, scale=0.1),
        niq=normal(INDEX_DIM, scale=0.1), nik=normal(INDEX_DIM, scale=0.1),
        positions=jnp.asarray(positions),
        trig=jnp.pad(normal(16384, ROTARY), ((0, 0), (0, 128 - ROTARY))),
    )
    meta = dict(starts=jnp.asarray(starts), context=jnp.asarray(contexts, jnp.int32),
                ring_ids=jnp.asarray(ring), compressed_table=jnp.asarray(comp))
    pools = [from_raw(poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep),
                                       layout, r, seq)) for r in range(D)]
    return layout, pools, pair_qgate, shared, meta, starts


def _args(qg, s, pool, m, table):
    return (qg, s["key"], s["value"], s["index_qk"], s["nq"], s["nk"], s["niq"], s["nik"],
            s["positions"], s["trig"], pool, m["starts"], m["context"], m["ring_ids"],
            jnp.asarray(table), m["compressed_table"])


def _u8(x):
    return jax.lax.bitcast_convert_type(x, jnp.uint8)


def _bits(x):
    x = np.asarray(x, np.float32)
    return x.view(np.uint32)


def run_case(query_lens, contexts, tokens, *, seed=0, main_slot=0, recorder=None):
    layout, pools, qgate, s, m, starts = make_step(seed, query_lens, contexts, tokens)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot, max_query_len=ROWS)
    live = int(starts[-1])
    report = {}
    # Jitted per case (the stand-in's order switch is read while tracing; the standin fixture clears JAX's caches);
    # the Pallas writes then run as compiled interpret loops, not op by op.
    static = ("compress_ratio", "token_topk", "norm_eps", "rotary_dim", "dcp_size", "k_scale",
              "v_scale", "main_slot", "max_query_len")
    o1_step = jax.jit(pooled_qsa.pooled_qsa_dcp_step, static_argnames=static)
    o1s_select = jax.jit(pooled_qsa.pooled_qsa_request_select_dcp_step,
                         static_argnames=static + ("shard_size",))
    o1s_attend = jax.jit(pooled_qsa.pooled_qsa_request_attend_dcp_step,
                         static_argnames=("kv_heads", "compress_ratio", "token_topk",
                                          "dcp_size", "k_scale", "v_scale", "main_slot",
                                          "max_query_len"))
    for r in range(D):
        args = _args(qgate, s, pools[r], m, layout.dcp_table)
        n0 = len(recorder.calls)
        o1 = o1_step(*args, rank_operand(r), dcp_size=D, **opts)
        jax.effects_barrier()
        o1_calls = recorder.calls[n0:]
        n1 = len(recorder.calls)
        blocks = []
        for sh in range(TP):
            res = o1s_select(
                *args, rank_operand(sh), rank_operand(r), shard_size=TP, dcp_size=D, **opts)
            if sh == 0:
                first = res
            else:
                assert bool(jnp.array_equal(_u8(res[0]), _u8(first[0]))), "shard pools differ"
            blocks.append(res[-1])
            del res
        jax.effects_barrier()
        o1s_calls = recorder.calls[n1:]
        pool, q, gate, lengths, req, st, _ = first
        del first
        gathered = jnp.stack(blocks)
        assert gathered.shape == (TP, tokens // TP, TOPK // RATIO)
        o1s = o1s_attend(
            q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"], pool,
            gathered, rank_operand(r), kv_heads=1, compress_ratio=RATIO, token_topk=TOPK,
            dcp_size=D, k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot,
            max_query_len=ROWS)
        # Writes: identical pools.
        assert bool(jnp.array_equal(_u8(o1s[0]), _u8(o1[0]))), f"DCP rank {r}: pool differs"
        # Selection: the same set per live row.
        sel1, sel2 = np.asarray(o1[3]), np.asarray(o1s[3])
        assert sel1.shape == sel2.shape
        for row in range(live):
            a, b = np.sort(sel1[row]), np.sort(sel2[row])
            assert np.array_equal(a, b), f"DCP rank {r} row {row}: selection differs"
        exact_order = bool(np.array_equal(sel1[:live], sel2[:live]))
        # Attention: bit-identical partial output and lse (live rows; padding
        # rows have no keys on either path).
        out1, out2 = _bits(o1[1])[:live], _bits(o1s[1])[:live]
        lse1, lse2 = _bits(o1[2])[:live], _bits(o1s[2])[:live]
        assert np.array_equal(out1, out2), f"DCP rank {r}: partial attention differs"
        assert np.array_equal(lse1, lse2), f"DCP rank {r}: lse differs"
        # Engagement: O1 scores every live row; O1s scores each live row once,
        # across the TP shards, and no shard more than its balanced share.
        assert sum(c[1] for c in o1_calls) == live
        assert len(o1s_calls) >= TP and sum(c[1] for c in o1s_calls) == live, (
            f"O1s scored {sum(c[1] for c in o1s_calls)} rows, want {live}")
        per_shard_rows = [c[0] for c in o1s_calls]
        assert max(per_shard_rows) <= tokens // TP
        uniform = all(int(x) in (0, ROWS) for x in query_lens)
        if uniform:
            busy = int(np.count_nonzero(query_lens))
            want = -(-busy // TP)
            assert max(c[2] for c in o1s_calls) <= max(1, want), (
                f"a shard scored {max(c[2] for c in o1s_calls)} requests, balanced share {want}")
        report[r] = dict(exact_order=exact_order,
                         shard_live_rows=[c[1] for c in o1s_calls],
                         shard_live_seqs=[c[2] for c in o1s_calls])
    return report


def _seats(busy, idle, rng_seed, *, short=False, rows=ROWS):
    """``busy`` verify requests of ``rows`` rows plus ``idle`` idle seats,
    with mixed contexts crossing the DCP pages (B = 512, D * B = 1024)."""
    rng = np.random.default_rng(rng_seed)
    lo, hi = (3, 250) if short else (37, 4100 if busy <= 12 else 1700)
    contexts = rng.integers(lo, hi, busy).tolist() + [0] * idle
    query_lens = [rows] * busy + [0] * idle
    order = rng.permutation(len(contexts))
    return [query_lens[i] for i in order], [contexts[i] for i in order]


# (busy requests, idle seats, token bucket): the served decode buckets are
# 16/32/48/64 (linear, interval 16), then 128 and 256.
CASES = {
    "B1": (1, 3, 16),
    "B3": (3, 1, 16),
    "B4": (4, 4, 16),
    "B5": (5, 3, 32),
    # A padded bucket (48 tokens for 20 live rows): each rank's balanced range
    # (4 rows) is shorter than its static slice (6 rows).
    "B5pad": (5, 3, 48),
    "B8": (8, 8, 32),
    "B12": (12, 4, 48),
    "B28": (28, 4, 128),
    "B48": (48, 16, 256),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_o1s_matches_o1(interpret, rpa_stand_in, standin, case):
    busy, idle, tokens = CASES[case]
    query_lens, contexts = _seats(busy, idle, rng_seed=busy)
    print(case, run_case(query_lens, contexts, tokens, recorder=standin))


def test_o1s_short_contexts_early_exit(interpret, rpa_stand_in, standin):
    """Every row fits the budget (the indexer's early exit and the attend's
    visible-blocks branch) on some shards but not on the whole step."""
    query_lens, contexts = _seats(6, 2, rng_seed=7, short=True)
    contexts[contexts.index(max(contexts))] = 3000
    print(run_case(query_lens, contexts, 32, recorder=standin))


def test_o1s_ragged_rows_and_windows(interpret, rpa_stand_in, standin, monkeypatch):
    """Requests of 1-4 rows (shards cut requests) and a 12-row indexer window
    (the shared stream cuts requests across windows on both paths)."""
    monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 12)
    query_lens = [4, 1, 3, 0, 4, 2, 4, 4, 1, 4]
    contexts = [1500, 777, 2049, 0, 1023, 1024, 3001, 511, 4000, 260]
    print(run_case(query_lens, contexts, 32, seed=3, recorder=standin))


def test_o1s_straddled_requests_keep_their_positions(interpret, rpa_stand_in, standin):
    """Requests cut by a shard boundary (6-row ranges, 4-row requests): the
    later shard's rows must keep their absolute positions. Contexts are 1 mod 4,
    so a position shifted by the 2 rows before the shard changes the row's
    visible groups; the cut requests are short (every visible group selected),
    and one long request keeps the step on the merged-selection path."""
    contexts = [3001] + [41 + 4 * i for i in range(11)] + [0] * 4
    query_lens = [ROWS] * 12 + [0] * 4
    report = run_case(query_lens, contexts, 48, seed=5, recorder=standin)
    assert all(v["shard_live_rows"] == [6] * 8 for v in report.values()), report


def test_o1s_exact_order_when_kernel_order_is_fixed(interpret, rpa_stand_in, standin):
    """With a batch-independent kernel order the selections agree element for
    element too (the MTP draft reads ``selected`` through the shared op)."""
    standin.reverse = False
    query_lens, contexts = _seats(12, 4, rng_seed=12)
    report = run_case(query_lens, contexts, 48, recorder=standin)
    assert all(v["exact_order"] for v in report.values()), report


def test_standin_sets_the_decode_flags(standin):
    """The stand-in fixture's decode flags are in force while it is active (the pair below checks they leave)."""
    assert envs.VLLM_TPU_QSA_DECODE_QUERY_BLOCK == 4 and envs.VLLM_TPU_QSA_INDEXER_MIXED_ONLY is True
    assert all(name in envs.__dict__ for name in STANDIN_ENVS)


def test_standin_leaves_no_frozen_envs(monkeypatch):
    """Runs after test_standin_sets_the_decode_flags (file order): none of the stand-in's names is left as a module
    attribute, and an os.environ change is seen through envs again (j-3307bbed: the old monkeypatch.setattr undo
    froze them, so a later setenv-only test ran the default decode path)."""
    for name in STANDIN_ENVS:
        assert name not in envs.__dict__, f"{name} frozen as a module attribute"
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_QUERY_BLOCK", "8")
    assert envs.VLLM_TPU_QSA_DECODE_QUERY_BLOCK == 8
    monkeypatch.setenv("VLLM_TPU_QSA_INDEXER_MIXED_ONLY", "1")
    assert envs.VLLM_TPU_QSA_INDEXER_MIXED_ONLY is True


def test_request_shard_width_balances_padded_buckets():
    """The live rows split evenly regardless of the bucket's padded tail."""
    def width(live, tokens):
        starts = jnp.asarray([0, live], jnp.int32)
        return int(pooled_qsa.request_shard_width(starts, tokens // TP, TP, ROWS))

    assert width(48 * 4, 256) == 24  # 6 requests per rank, not 8 on six ranks
    assert width(64 * 4, 256) == 32
    assert width(28 * 4, 128) == 16
    assert width(16 * 4, 64) == 8
    assert width(4 * 4, 16) == 2  # capped at the static slice: 2 rows per rank
    assert width(0, 16) == 1
    for live in range(0, 257):
        for tokens in (16, 32, 48, 64, 128, 256):
            if live <= tokens:
                w = width(live, tokens)
                assert 1 <= w <= tokens // TP and TP * w >= live


def test_o1s_jax_ops_register():
    """The O1s jax_ops build with the real torch_tpu (pallas.jax_op checks every
    step argument's annotation at build time) and their fakes are registered."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    select, attend = qsa_op._request_dcp_ops()
    assert select is not None and attend is not None
    assert qsa_op._request_dcp_ops() == (select, attend)  # built once (functools.cache)
    plain = qsa_op._request_plain_dcp_op()  # the prefill-shaped graphs' DCP step (input-list fix)
    assert plain is not None and qsa_op._request_plain_dcp_op() is plain


def test_request_shard_applies():
    """Decode-shaped graphs only (at most max_query_len rows per slot, tokens
    divisible by TP), TP > 1; the module asks with its own graph's shapes."""
    pytest.importorskip("torch_tpu")
    import types

    import torch

    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import request_shard_applies
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    assert request_shard_applies(256, 64, 8, 4)  # B64 verify step
    assert request_shard_applies(16, 64, 8, 4)  # B4 in the 16-token bucket
    assert request_shard_applies(16, 4, 8, 4)
    assert not request_shard_applies(2048, 64, 8, 4)  # a prefill chunk
    assert not request_shard_applies(136, 32, 8, 4)  # more rows than slots x rows
    assert not request_shard_applies(12, 64, 8, 4)  # not divisible by TP
    assert not request_shard_applies(256, 64, 1, 4)  # no TP group
    assert request_shard_applies(64, 64, 8, 1)  # 1-row decode: 64 rows over 64 slots
    assert not request_shard_applies(72, 64, 8, 1)  # 1-row decode: more rows than slots
    meta = types.SimpleNamespace(context_lens=torch.zeros(64, dtype=torch.int32))
    on = types.SimpleNamespace(_request_shard_dcp_op=object(), tp_size=8, max_query_len=4)
    off = types.SimpleNamespace(_request_shard_dcp_op=None, tp_size=8, max_query_len=4)
    assert Qwen4ExpQSA._request_shard_applies(on, torch.zeros(256, 8), meta)
    assert not Qwen4ExpQSA._request_shard_applies(on, torch.zeros(2048, 8), meta)
    assert not Qwen4ExpQSA._request_shard_applies(off, torch.zeros(256, 8), meta)
    assert not Qwen4ExpQSA._request_shard_applies(types.SimpleNamespace(), torch.zeros(256, 8), meta)


def test_o1s_torch_wrapper_plumbing(monkeypatch):
    """build_pooled_qsa_request_shard_dcp_op: the select gets this rank's
    shard_index and shard_size = TP; the attend gets the TP all-gather of every
    rank's blocks as [TP, T / TP, k] (rank order) plus the DCP keywords; the pair
    all-gather / combine finish as the DCP op's."""
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    tokens, k, world = 32, 64, 8
    calls = {}

    def select(*args, shard_size, dcp_size, **kw):
        calls["select"] = dict(shard_index=int(args[16]), dcp_rank=int(args[17]), shard_size=shard_size,
                               dcp_size=dcp_size, qgate_cols=args[0].shape[1], kw=sorted(kw))
        rows = tokens // shard_size
        blocks = torch.full((rows, k), int(args[16]), dtype=torch.int32)
        return (args[10], torch.zeros(tokens, 6, 4), torch.zeros(tokens, 6, 4),
                torch.zeros(tokens, dtype=torch.int32), torch.zeros(tokens, dtype=torch.int32),
                torch.zeros(5, dtype=torch.int32), blocks)

    def attend(q, gate, lengths, req, starts, main_table, context, pool, gathered, dcp_rank, **kw):
        calls["attend"] = dict(gathered=gathered.clone(), dcp_rank=int(dcp_rank), kw=kw)
        return pool, torch.zeros(tokens, 6, 4), torch.zeros(tokens, 6), torch.zeros(tokens, 7, dtype=torch.int32)

    class Group:
        def __init__(self, size, rank):
            self.world_size, self.rank_in_group = size, rank

        def all_gather(self, x, dim=0):
            # Rank r contributes a tensor filled with r (the select stub's blocks are filled with the shard index).
            parts = [torch.full_like(x, r) if x.dtype == torch.int32 else x for r in range(self.world_size)]
            return torch.cat(parts, dim=dim)

    monkeypatch.setattr(qsa_op, "_request_dcp_ops", lambda: (select, attend))
    monkeypatch.setattr(qsa_op, "_dcp_ops", lambda: (None, None, None, lambda o, l, g, r: g))
    monkeypatch.setattr(qsa_op, "_dcp_finish", lambda out, lse, qgate, hd, rank, pair, comb: ("finished", rank))
    op = qsa_op.build_pooled_qsa_request_shard_dcp_op(Group(world, 5), Group(4, 1), 4)
    args = [torch.zeros(tokens, 2 * 3 * 4)] + [torch.zeros(1)] * 3 + [torch.zeros(4)] + [torch.zeros(1)] * 3 \
        + [torch.zeros(tokens, dtype=torch.int64)] + [torch.zeros(1)] + [torch.zeros(10)] \
        + [torch.zeros(5, dtype=torch.int64)] * 5
    out, selected = op(*args, shard_index=torch.tensor([5], dtype=torch.int32),
                       dcp_rank=torch.tensor([1], dtype=torch.int32), kv_heads=1,
                       compress_ratio=4, token_topk=256, norm_eps=1e-6, rotary_dim=64,
                       k_scale=0.5, v_scale=2.0, main_slot=0, max_query_len=4)
    s, a = calls["select"], calls["attend"]
    assert s["shard_index"] == 5 and s["dcp_rank"] == 1 and s["shard_size"] == world and s["dcp_size"] == 4
    assert s["qgate_cols"] == 4 * 2 * 3 * 4  # the pair's heads gathered, as the DCP op
    g = a["gathered"]
    assert tuple(g.shape) == (world, tokens // world, k)
    assert all(bool((g[r] == r).all()) for r in range(world)), "gathered blocks not in TP rank order"
    assert a["dcp_rank"] == 1 and a["kw"]["dcp_size"] == 4 and a["kw"]["kv_heads"] == 1
    assert a["kw"]["max_query_len"] == 4 and a["kw"]["main_slot"] == 0 and a["kw"]["k_scale"] == 0.5
    assert out[0] == "finished" and int(out[1]) == 1  # _dcp_finish with this rank's dcp_rank


def test_o1s_bind_line_logged_on_every_rank(monkeypatch):
    """The per-rank engagement line the G2 reader requires on TP ranks 0-7: plain
    logger.info once per process, whatever the rank (vLLM's info_once defaults to
    scope "local" and would print on the local first rank only)."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    seen = []

    class Rec:
        def info(self, msg, *args):
            seen.append(msg % args)

        def info_once(self, msg, *args, **kw):  # not used for this line
            pass

    monkeypatch.setattr(qsa_mod, "logger", Rec())
    monkeypatch.setattr(qsa_mod, "_Q38_O1S_LOGGED", False)
    qsa_mod._log_o1s_bind(3, 8, 4, 4)
    qsa_mod._log_o1s_bind(3, 8, 4, 4)  # once per process
    assert seen == ["q38 O1s request-sharded decode selection: on (TP rank 3 of 8, main-KV DCP 4, "
                    "rows per request 4)"]


# ---------------------------------------------------------------------------------------------
# One input list for every compile range (G2-O1s j-fa24a1ff: engine init failed on every O1s arm
# with shape_variants.ShapeSpecializationError, "artifact_compile_range_16_16_subgraph_0 was
# compiled by another trace of this model whose inputs differ": the request-shard traces (<= 256
# tokens) took 12 int32 [1] request_shard_index inputs that the 512-token trace did not).
# ---------------------------------------------------------------------------------------------

HEADS_, DH_, HID_, IDX_ = 2, 4, 8, 4  # this rank's heads, head dim, hidden, index dim


class _Captured(Exception):
    """Raised by the capture backend's runnable: the graphs are compiled, never run."""


def _bound_qsa(monkeypatch, flag, tp_rank=5, dcp=4, tp=8):
    """A Qwen4ExpQSA holding only what bind_pool and attend_with_indices read, bound by the real
    bind_pool (stub TP and DCP groups) with the real jax_ops, traced through their fakes."""
    import types
    from functools import partial

    import torch
    import vllm.distributed
    from torch import nn

    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import build_pooled_qsa_op
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    class Group:
        def __init__(self, size, rank):
            self.world_size, self.rank_in_group = size, rank

        def all_gather(self, x, dim=0):
            return torch.cat([x] * self.world_size, dim=dim)

    class Norm(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(n, dtype=torch.bfloat16))

    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: Group(tp, tp_rank))
    monkeypatch.setattr(qsa_mod, "main_kv_dcp_group", lambda size: Group(size, tp_rank % size))
    monkeypatch.setattr(qsa_mod, "_Q38_O1S_LOGGED", True)
    m = qsa_mod.Qwen4ExpQSA.__new__(qsa_mod.Qwen4ExpQSA)
    nn.Module.__init__(m)

    def lin(out):
        return nn.Linear(HID_, out, bias=False, dtype=torch.bfloat16)

    m.q_proj, m.k_proj, m.v_proj = lin(2 * HEADS_ * DH_), lin(DH_), lin(DH_)
    m.indexer = nn.Module()
    m.indexer.index_qk_proj = lin(2 * IDX_)
    m.indexer.q_layernorm, m.indexer.k_layernorm = Norm(IDX_), Norm(IDX_)
    m.q_norm, m.k_norm = Norm(DH_), Norm(DH_)
    m.register_buffer("rope_cache", torch.zeros(4096, 128, dtype=torch.bfloat16), persistent=False)
    m.register_buffer("dcp_rank", torch.full((1,), tp_rank % dcp, dtype=torch.int32), persistent=False)
    m.decode_request_shard = flag
    if flag:  # as __init__ registers it
        m.register_buffer(
            "request_shard_index", torch.full((1,), tp_rank, dtype=torch.int32), persistent=False)
    m.prefix = "model.layers.3.self_attn"
    m.pooled, m.fp8_kv, m.single_graph, m.token_shard = True, False, False, False
    m.tp_size, m.tp_rank, m.main_kv_dcp, m.num_kv_heads, m.max_query_len = tp, tp_rank, dcp, 1, ROWS
    m._op = partial(build_pooled_qsa_op(), compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6,
                    rotary_dim=ROTARY)
    m.bind_pool(torch.zeros(16, 128, dtype=torch.uint8))
    tables = {
        m.prefix + ".raw_ring": torch.zeros(64, dtype=torch.int32),
        m.main_cache_name: torch.zeros(64, 8, dtype=torch.int32),
        m.compressed_cache_name: torch.zeros(64, 8, dtype=torch.int32),
    }
    ctx = types.SimpleNamespace(
        attn_metadata={k: types.SimpleNamespace(block_tables=v) for k, v in tables.items()})
    monkeypatch.setattr(qsa_mod, "get_forward_context", lambda: ctx)
    return m


def _traced_subgraphs(module, buckets):
    """vLLM's piecewise subgraph of every trace ``module`` needs over ``buckets``: Dynamo with the
    token dim dynamic (as support_torch_compile marks it), split by vLLM's own split_graph (no
    splitting ops: one subgraph, submod_0, as on TPU), never run."""
    import torch
    from torch import nn
    from vllm.compilation.backends import split_graph

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata

    class Step(nn.Module):
        def __init__(self, qsa):
            super().__init__()
            self.qsa = qsa

        def forward(self, hidden, positions, query_start_loc, context_lens):
            meta = QSAMetadata(query_start_loc, context_lens, None, None, None)
            return self.qsa.attend_with_indices(hidden, positions, meta)

    graphs = []

    def backend(gm, example_inputs):
        _, items = split_graph(gm, [])
        assert len(items) == 1, [i.submod_name for i in items]
        graphs.append(items[0].graph)

        def run(*args):
            raise _Captured()

        return run

    torch._dynamo.reset()
    compiled = torch.compile(Step(module), backend=backend, fullgraph=True, dynamic=False)
    starts = torch.zeros(65, dtype=torch.int32)
    context = torch.zeros(64, dtype=torch.int32)  # 64 request slots
    for tokens in buckets:
        hidden = torch.zeros(tokens, HID_, dtype=torch.bfloat16)
        positions = torch.zeros(tokens, dtype=torch.int64)
        torch._dynamo.maybe_mark_dynamic(hidden, 0)
        torch._dynamo.maybe_mark_dynamic(positions, 0)
        try:
            with torch.no_grad():
                compiled(hidden, positions, starts, context)
        except Exception as exc:  # the runnable's sentinel, possibly wrapped by Dynamo
            if not isinstance(exc, _Captured) and not isinstance(exc.__cause__, _Captured):
                raise
    torch._dynamo.reset()
    return graphs


def _inputs(graph):
    """(placeholder name, dtype) of a subgraph's tensor inputs, in its input order (SymInts,
    named per trace, are left to shape_variants' own source matching)."""
    import torch

    out = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            continue
        value = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(value, torch.Tensor):
            out.append((node.name, str(value.dtype)))
    return out


def _qsa_ops(graph):
    return sorted({str(n.target).split(".")[1] for n in graph.graph.nodes
                   if n.op == "call_function" and str(n.target).startswith("qwen4_exp.")})


def test_o1s_every_compile_range_takes_one_input_list(monkeypatch):
    """With the flag on, the request-shard trace (16 tokens: decode-shaped) and the plain trace
    (512 tokens: more rows than 64 slots x 4) of one module must give vLLM's subgraph the same
    inputs in the same order, so shape_variants hands buckets between them (_precompile_backbone's
    warmup, on every engine start). Order is checked by name too: a same-shaped pair trading places (shard_index and
    dcp_rank are both int32 [1]) passes the hand-over and binds the wrong operand. O1 (flag off):
    one trace, the plain DCP op, no request_shard_index."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.compilation import shape_variants

    decode, prefill = _traced_subgraphs(_bound_qsa(monkeypatch, True), (16, 512))
    # The hand-over itself (production code), first: the plain trace must adopt the decode
    # trace's bucket. Without the fix this raises the TPU's ShapeSpecializationError.
    tracked = shape_variants.BucketExecutables()
    key = "artifact_compile_range_16_16_subgraph_0"
    runnable = object()
    tracked.put("cache_dir", key, decode, runnable)
    assert tracked.get("cache_dir", key, prefill) is runnable
    # Same inputs by name, in the same order (the hand-over compares dtypes and shapes only).
    names = [n for n, _ in _inputs(decode)]
    assert any("request_shard_index" in n for n in names) and any("dcp_rank" in n for n in names)
    assert _inputs(decode) == _inputs(prefill)
    # Each trace took its own path: request shard at 16 tokens, the carry op at 512.
    assert _qsa_ops(decode) == ["pooled_qsa_request_attend_dcp", "pooled_qsa_request_select_dcp",
                                "qsa_dcp_combine"], _qsa_ops(decode)
    assert _qsa_ops(prefill) == ["pooled_qsa_request_plain_dcp", "qsa_dcp_combine"], _qsa_ops(prefill)

    o1 = _traced_subgraphs(_bound_qsa(monkeypatch, False), (16, 512))
    assert len(o1) == 1, "O1 traced more than once: its graph now depends on the token count"
    assert _qsa_ops(o1[0]) == ["pooled_qsa_dcp", "qsa_dcp_combine"], _qsa_ops(o1[0])
    assert not any("request_shard_index" in n for n, _ in _inputs(o1[0]))


def test_o1s_plain_wrapper_plumbing(monkeypatch):
    """build_pooled_qsa_request_plain_dcp_op: the step gets shard_index at operand 16 and
    dcp_rank at 17 (where the request-shard select takes them), the pair-gathered qgate, and
    the DCP keywords; kv_heads is dropped; the finish is the DCP op's."""
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    tokens, calls = 32, {}

    def step(*args, dcp_size, **kw):
        calls["step"] = dict(n=len(args), shard_index=int(args[16]), dcp_rank=int(args[17]),
                             dcp_size=dcp_size, qgate_cols=args[0].shape[1], kw=sorted(kw))
        return args[10], torch.zeros(tokens, 6, 4), torch.zeros(tokens, 6), torch.zeros(tokens, 7)

    class Pair:
        world_size, rank_in_group = 4, 1

        def all_gather(self, x, dim=0):
            return torch.cat([x] * 4, dim=dim)

    monkeypatch.setattr(qsa_op, "_request_plain_dcp_op", lambda: step)
    monkeypatch.setattr(qsa_op, "_dcp_ops", lambda: (None, None, None, None))
    monkeypatch.setattr(qsa_op, "_dcp_finish", lambda out, lse, qgate, hd, rank, pair, comb: ("finished", rank))
    op = qsa_op.build_pooled_qsa_request_plain_dcp_op(Pair(), 4)
    args = [torch.zeros(tokens, 2 * 3 * 4)] + [torch.zeros(1)] * 3 + [torch.zeros(4)] + [torch.zeros(1)] * 3 \
        + [torch.zeros(tokens, dtype=torch.int64)] + [torch.zeros(1)] + [torch.zeros(10)] \
        + [torch.zeros(5, dtype=torch.int64)] * 5
    out, selected = op(*args, shard_index=torch.tensor([5], dtype=torch.int32),
                       dcp_rank=torch.tensor([1], dtype=torch.int32), kv_heads=1,
                       compress_ratio=4, token_topk=256, norm_eps=1e-6, rotary_dim=64,
                       k_scale=0.5, v_scale=2.0, main_slot=0, max_query_len=4)
    s = calls["step"]
    assert s["n"] == 18 and s["shard_index"] == 5 and s["dcp_rank"] == 1 and s["dcp_size"] == 4
    assert s["qgate_cols"] == 4 * 2 * 3 * 4
    assert "kv_heads" not in s["kw"] and {"compress_ratio", "token_topk", "max_query_len"} <= set(s["kw"])
    assert out[0] == "finished" and int(out[1]) == 1


def test_o1s_plain_step_is_the_dcp_step(interpret, rpa_stand_in, standin):
    """pooled_qsa_request_plain_dcp_step = pooled_qsa_dcp_step with an unused shard_index
    operand placed where the request-shard select takes it: same parameters otherwise, and
    bit-identical outputs (pool, out, lse, selected) whatever the shard index."""
    import inspect

    plain = list(inspect.signature(pooled_qsa.pooled_qsa_request_plain_dcp_step).parameters)
    base = list(inspect.signature(pooled_qsa.pooled_qsa_dcp_step).parameters)
    select = list(inspect.signature(pooled_qsa.pooled_qsa_request_select_dcp_step).parameters)
    i = base.index("dcp_rank")
    assert plain == base[:i] + ["shard_index"] + base[i:]
    assert plain[: plain.index("dcp_rank") + 1] == select[: select.index("dcp_rank") + 1]
    query_lens, contexts = _seats(4, 4, 11)
    layout, pools, qgate, s, m, _ = make_step(11, query_lens, contexts, 16)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=0, max_query_len=ROWS)
    static = tuple(opts) + ("dcp_size",)
    r = 1
    args = _args(qgate, s, pools[r], m, layout.dcp_table)
    o1 = jax.jit(pooled_qsa.pooled_qsa_dcp_step, static_argnames=static)(
        *args, rank_operand(r), dcp_size=D, **opts)
    got = jax.jit(pooled_qsa.pooled_qsa_request_plain_dcp_step, static_argnames=static)(
        *args, rank_operand(6), rank_operand(r), dcp_size=D, **opts)  # TP rank 6: its DCP rank is 2
    assert bool(jnp.array_equal(_u8(got[0]), _u8(o1[0]))), "pool differs"
    for a, b, name in ((got[1], o1[1], "out"), (got[2], o1[2], "lse")):
        assert np.array_equal(_bits(a), _bits(b)), f"{name} differs"
    assert np.array_equal(np.asarray(got[3]), np.asarray(o1[3])), "selected differs"


def test_o1s_target_layers_only(monkeypatch):
    """The request shard is for target QSA layers under main-KV DCP with TP > 1. The MTP draft
    never takes it, also when VLLM_TPU_QSA_MTP_KV_DCP splits the draft's main KV (F2 serving):
    its single graph is traced once for every bucket, so the token-count routing must not run
    there. __init__ decides through the same helper."""
    pytest.importorskip("torch_tpu")
    import inspect

    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    mode = qsa_mod.Qwen4ExpQSA._decode_request_shard_mode
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DECODE_REQUEST_SHARD", True, raising=False)
    assert mode(4, 8, "model.layers.3.self_attn")
    assert mode(2, 8, "model.layers.3.self_attn")
    assert not mode(4, 8, "model.mtp.layers.0.self_attn")  # the draft, F2's MTP DCP on
    assert not mode(4, 8, "mtp.self_attn")
    assert not mode(1, 8, "model.layers.3.self_attn")  # no main-KV DCP
    assert not mode(4, 1, "model.layers.3.self_attn")  # no TP group
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_DECODE_REQUEST_SHARD", False, raising=False)
    assert not mode(4, 8, "model.layers.3.self_attn")  # flag off (O1)
    src = inspect.getsource(qsa_mod.Qwen4ExpQSA.__init__)
    assert "self.decode_request_shard = self._decode_request_shard_mode(" in src


# ---------------------------------------------------------------------------------------------
# q38 L2 (decode plan v1 D21): the O1s min-rows gate, VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS.
# O1s carries a near-fixed cost per step (one XLA all-gather per QSA layer, ~0.18-0.20 ms/step at
# 64 and 256 rows, plus ~0.2 ms of re-base / reassembly glue at 64 rows; M, G2r j-92d0cc68) that its
# indexer saving does not cover at small row counts. Graphs below the gate keep every row on every
# rank through the request-shard module's plain DCP step (= pooled_qsa_dcp + an unused operand,
# test_o1s_plain_step_is_the_dcp_step), so their compile ranges keep one input list.
# ---------------------------------------------------------------------------------------------


def test_request_shard_min_rows_gate():
    """Gate 0 is the ungated routing on every shape; gate 48 keeps the 16- and 32-row buckets on
    the replicated indexer and sends 48 rows and above (decode-shaped) to the request shard."""
    pytest.importorskip("torch_tpu")
    import inspect
    import itertools
    import types

    import torch

    from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import request_shard_applies
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    for t, s, w, q in itertools.product((8, 12, 16, 32, 48, 52, 64, 96, 128, 224, 256, 512, 2048),
                                        (4, 16, 64), (1, 2, 8), (1, 4)):
        ungated = w > 1 and t % w == 0 and t <= s * q
        assert request_shard_applies(t, s, w, q) == ungated, (t, s, w, q)
        assert request_shard_applies(t, s, w, q, 0) == ungated, (t, s, w, q)
        assert request_shard_applies(t, s, w, q, 48) == (ungated and t >= 48), (t, s, w, q)
    assert not request_shard_applies(16, 64, 8, 4, 48)  # c16 bucket
    assert not request_shard_applies(32, 64, 8, 4, 48)  # c32 bucket
    assert request_shard_applies(48, 64, 8, 4, 48)
    assert request_shard_applies(64, 64, 8, 4, 48)
    assert request_shard_applies(256, 64, 8, 4, 48)
    assert not request_shard_applies(2048, 64, 8, 4, 48)  # a prefill chunk stays off
    meta = types.SimpleNamespace(context_lens=torch.zeros(64, dtype=torch.int32))
    gated = types.SimpleNamespace(_request_shard_dcp_op=object(), tp_size=8, max_query_len=4,
                                  request_shard_min_rows=48)
    assert not Qwen4ExpQSA._request_shard_applies(gated, torch.zeros(16, 8), meta)
    assert not Qwen4ExpQSA._request_shard_applies(gated, torch.zeros(32, 8), meta)
    assert Qwen4ExpQSA._request_shard_applies(gated, torch.zeros(48, 8), meta)
    assert Qwen4ExpQSA._request_shard_applies(gated, torch.zeros(64, 8), meta)
    ungated_mod = types.SimpleNamespace(_request_shard_dcp_op=object(), tp_size=8, max_query_len=4)
    assert Qwen4ExpQSA._request_shard_applies(ungated_mod, torch.zeros(16, 8), meta)  # no attribute = 0
    # __init__ reads the env once, next to the request-shard operand (never inside the Dynamo capture).
    src = inspect.getsource(Qwen4ExpQSA.__init__)
    assert "self.request_shard_min_rows = int(" in src
    assert "envs.VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS" in src
    assert "envs.VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS" not in inspect.getsource(
        Qwen4ExpQSA._request_shard_applies)


def test_request_shard_min_rows_env(monkeypatch):
    """Registered, non-negative int, default 0 (no gate)."""
    from vllm_torchtpu import envs as E

    name = "VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS"
    assert name in E.environment_variables
    monkeypatch.delenv(name, raising=False)
    assert E.environment_variables[name]() == 0
    monkeypatch.setenv(name, "48")
    assert E.environment_variables[name]() == 48


def test_o1s_min_rows_gate_compile_ranges(monkeypatch):
    """Gate 48 with the flag on: the 16-row trace takes the plain DCP step, the 64-row trace the
    request shard, the 512-row trace the plain step; all three give vLLM's subgraph the same inputs
    in the same order (shape_variants hand-over), and the hand-over adopts the gated 16-row bucket
    into the 64-row trace. Ungated control: the 16-row trace takes the request shard."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.compilation import shape_variants

    plain = ["pooled_qsa_request_plain_dcp", "qsa_dcp_combine"]
    shard = ["pooled_qsa_request_attend_dcp", "pooled_qsa_request_select_dcp", "qsa_dcp_combine"]
    m = _bound_qsa(monkeypatch, True)
    m.request_shard_min_rows = 48
    graphs = _traced_subgraphs(m, (16, 64, 512))
    assert [_qsa_ops(g) for g in graphs] == [plain, shard, plain], [_qsa_ops(g) for g in graphs]
    assert _inputs(graphs[0]) == _inputs(graphs[1]) == _inputs(graphs[2])
    tracked = shape_variants.BucketExecutables()
    key = "artifact_compile_range_16_16_subgraph_0"
    runnable = object()
    tracked.put("cache_dir", key, graphs[0], runnable)
    assert tracked.get("cache_dir", key, graphs[1]) is runnable
    m0 = _bound_qsa(monkeypatch, True)
    m0.request_shard_min_rows = 0
    ungated = _traced_subgraphs(m0, (16, 512))
    assert _qsa_ops(ungated[0]) == shard, _qsa_ops(ungated[0])


def test_o1s_min_rows_gate_line(monkeypatch):
    """One engagement line per process when the gate is set (every TP rank, plain logger.info);
    nothing when it is 0, so the ungated O1s logs are unchanged."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    seen = []

    class Rec:
        def info(self, msg, *args):
            seen.append(msg % args)

    monkeypatch.setattr(qsa_mod, "logger", Rec())
    monkeypatch.setattr(qsa_mod, "_Q38_O1S_GATE_LOGGED", False)
    qsa_mod._log_o1s_gate(0)
    assert seen == []
    qsa_mod._log_o1s_gate(48)
    qsa_mod._log_o1s_gate(48)
    assert seen == ["q38 O1s min-rows gate: request-sharded selection only on graphs of >= 48 rows "
                    "(VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS)"]


# ---------------------------------------------------------------------------------------------
# q38 L2 (decode plan v1 D27a): VLLM_TPU_QSA_DCP_Q_ONLY_GATHER, the decode DCP ops all-gather only the q halves of
# qgate over the DCP pair. The DCP attention returns ungated partials, so the gathered gate halves are dead data.
# ---------------------------------------------------------------------------------------------


def test_dcp_steps_ignore_the_gathered_gate(interpret, rpa_stand_in, standin):
    """pooled_qsa_dcp_step and O1s's request select + attend give bit-identical (pool, out, lse, selected) when the
    gathered qgate's gate halves are zeros: the DCP attention never reads them (the combine gates with the local
    qgate). Holds for the replicated-selection step and for O1s's request shard (rank 1 of TP 8, DCP rank 1)."""
    query_lens, contexts = _seats(4, 4, 23)
    layout, pools, qgate, s, m, _ = make_step(23, query_lens, contexts, 16)
    tokens = qgate.shape[0]
    zeroed = qgate.reshape(tokens, D * H, 2 * DH).at[..., DH:].set(0).reshape(tokens, -1)
    assert not bool(jnp.array_equal(zeroed, qgate))  # the gate halves really changed
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=0, max_query_len=ROWS)
    static = tuple(opts) + ("dcp_size",)
    r = 1
    step = jax.jit(pooled_qsa.pooled_qsa_dcp_step, static_argnames=static)
    want = step(*_args(qgate, s, pools[r], m, layout.dcp_table), rank_operand(r), dcp_size=D, **opts)
    got = step(*_args(zeroed, s, pools[r], m, layout.dcp_table), rank_operand(r), dcp_size=D, **opts)
    assert bool(jnp.array_equal(_u8(got[0]), _u8(want[0]))), "pool differs"
    for a, b, name in ((got[1], want[1], "out"), (got[2], want[2], "lse")):
        assert np.array_equal(_bits(a), _bits(b)), f"{name} differs"
    assert np.array_equal(np.asarray(got[3]), np.asarray(want[3])), "selected differs"
    sel = jax.jit(pooled_qsa.pooled_qsa_request_select_dcp_step, static_argnames=static + ("shard_size",))
    sw = sel(*_args(qgate, s, pools[r], m, layout.dcp_table), rank_operand(r), rank_operand(r), shard_size=TP,
             dcp_size=D, **opts)
    sg = sel(*_args(zeroed, s, pools[r], m, layout.dcp_table), rank_operand(r), rank_operand(r), shard_size=TP,
             dcp_size=D, **opts)
    for i in (0, 1, 3, 4, 5, 6):  # pool, q, lengths, req, starts, blocks: everything but the gate itself
        assert np.array_equal(np.asarray(_u8(sg[i]) if i == 0 else sg[i]), np.asarray(_u8(sw[i]) if i == 0 else sw[i])), i
    gathered = jnp.stack([sw[6]] * TP)  # every rank's blocks (identity shards suffice for the gate check)
    att_static = ("kv_heads", "compress_ratio", "token_topk", "dcp_size", "k_scale", "v_scale", "main_slot",
                  "max_query_len")
    att = jax.jit(pooled_qsa.pooled_qsa_request_attend_dcp_step, static_argnames=att_static)
    kw = dict(kv_heads=1, compress_ratio=RATIO, token_topk=TOPK, dcp_size=D, k_scale=K_SCALE, v_scale=V_SCALE,
              main_slot=0, max_query_len=ROWS)
    aw = att(sw[1], sw[2], sw[3], sw[4], sw[5], jnp.asarray(layout.dcp_table), m["context"], sw[0], gathered,
             rank_operand(r), **kw)
    ag = att(sg[1], sg[2], sg[3], sg[4], sg[5], jnp.asarray(layout.dcp_table), m["context"], sg[0], gathered,
             rank_operand(r), **kw)
    assert bool(jnp.array_equal(_u8(ag[0]), _u8(aw[0])))
    for a, b in ((ag[1], aw[1]), (ag[2], aw[2])):
        assert np.array_equal(_bits(a), _bits(b))
    assert np.array_equal(np.asarray(ag[3]), np.asarray(aw[3]))


def test_pair_gather_qgate_layout():
    """q-only: the q halves land exactly where the full gather puts them (group head r * heads + h = rank r's head h)
    and the gate halves are zeros; off: the full gather itself. A rank-distinct stub group stands in for the pair."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    class Pair:
        world_size = 4

        def all_gather(self, x, dim=0):
            return torch.cat([x * (r + 1) + r for r in range(self.world_size)], dim=dim)

    tokens, heads, hd = 16, 3, 8
    qgate = torch.randn(tokens, heads * 2 * hd)
    full = qsa_op._pair_gather_qgate(qgate, hd, Pair(), False)
    assert torch.equal(full, Pair().all_gather(qgate, dim=-1))
    qo = qsa_op._pair_gather_qgate(qgate, hd, Pair(), True)
    assert qo.shape == full.shape
    f3, q3 = full.reshape(tokens, 4 * heads, 2 * hd), qo.reshape(tokens, 4 * heads, 2 * hd)
    assert torch.equal(q3[..., :hd], f3[..., :hd])
    assert torch.equal(q3[..., hd:], torch.zeros_like(q3[..., hd:]))


@pytest.mark.parametrize("builder", ["dcp", "draft", "plain", "request"])
def test_q_only_gather_wiring(monkeypatch, builder):
    """With the flag each decode DCP op hands its step a q-only gathered qgate (gate halves zero, q halves those of
    the full gather) and still gates with the local qgate in _dcp_finish; without it, the full gather."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    tokens, heads, hd = 32, 3, 4
    seen = {}

    class Pair:
        world_size, rank_in_group = 4, 1

        def all_gather(self, x, dim=0):
            return torch.cat([x * (r + 1) for r in range(4)], dim=dim)

    class TP:
        world_size, rank_in_group = 8, 1

    def step(*args, **kw):
        seen["qg"] = args[0].clone()
        if builder == "request":
            return (args[10], torch.zeros(tokens, 4 * heads, hd), torch.zeros(tokens, 4 * heads, hd),
                    torch.zeros(tokens, dtype=torch.int32), torch.zeros(tokens, dtype=torch.int32),
                    torch.zeros(5, dtype=torch.int32), torch.zeros(tokens // 8, 7, dtype=torch.int32))
        return args[10], torch.zeros(tokens, 4 * heads, hd), torch.zeros(tokens, 4 * heads), torch.zeros(tokens, 7)

    def attend(*args, **kw):
        return args[7], torch.zeros(tokens, 4 * heads, hd), torch.zeros(tokens, 4 * heads), torch.zeros(tokens, 7)

    def finish(out, lse, qgate, hd_, rank, pair, comb):
        seen["finish_qgate"] = qgate
        return torch.zeros(tokens, heads * hd), rank

    monkeypatch.setattr(qsa_op, "_dcp_finish", finish)
    monkeypatch.setattr(qsa_op, "_dcp_ops", lambda: (step, None, None, None))
    monkeypatch.setattr(qsa_op, "_dcp_draft_ops", lambda: (step, step, step))
    monkeypatch.setattr(qsa_op, "_request_plain_dcp_op", lambda: step)
    monkeypatch.setattr(qsa_op, "_request_dcp_ops", lambda: (step, attend))
    monkeypatch.setattr(qsa_op, "_all_gather_ranks", lambda x, g: torch.stack([x] * 8))
    qgate = torch.randn(tokens, heads * 2 * hd)
    args = [qgate] + [torch.zeros(1)] * 3 + [torch.zeros(hd)] + [torch.zeros(1)] * 3 \
        + [torch.zeros(tokens, dtype=torch.int64)] + [torch.zeros(1)] + [torch.zeros(10)] \
        + [torch.zeros(5, dtype=torch.int64)] * 5
    kw = dict(compress_ratio=4, token_topk=256, norm_eps=1e-6, rotary_dim=64, k_scale=0.5, v_scale=2.0,
              main_slot=0, max_query_len=4)
    for on in (False, True):
        monkeypatch.setenv("VLLM_TPU_QSA_DCP_Q_ONLY_GATHER", "1" if on else "0")
        if builder == "dcp":
            op = qsa_op.build_pooled_qsa_dcp_op(Pair(), 4)
            op(*args, dcp_rank=torch.tensor([1], dtype=torch.int32), **kw)
        elif builder == "draft":
            op = qsa_op.build_pooled_qsa_draft_dcp_op(Pair(), 4, shared=True, with_rows=False)
            op(*args, torch.zeros(tokens, 7, dtype=torch.int64), dcp_rank=torch.tensor([1], dtype=torch.int32), **kw)
        elif builder == "plain":
            op = qsa_op.build_pooled_qsa_request_plain_dcp_op(Pair(), 4)
            op(*args, shard_index=torch.tensor([1], dtype=torch.int32), dcp_rank=torch.tensor([1], dtype=torch.int32),
               kv_heads=1, **kw)
        else:
            op = qsa_op.build_pooled_qsa_request_shard_dcp_op(TP(), Pair(), 4)
            op(*args, shard_index=torch.tensor([1], dtype=torch.int32), dcp_rank=torch.tensor([1], dtype=torch.int32),
               kv_heads=1, **kw)
        g3 = seen["qg"].reshape(tokens, 4 * heads, 2 * hd)
        full = Pair().all_gather(qgate, dim=-1).reshape(tokens, 4 * heads, 2 * hd)
        assert torch.equal(g3[..., :hd], full[..., :hd])
        if on:
            assert torch.equal(g3[..., hd:], torch.zeros_like(g3[..., hd:])), builder
        else:
            assert torch.equal(g3, full), builder
        assert torch.equal(seen["finish_qgate"], qgate)  # the combine still gates with the local qgate


def test_q_only_gather_env():
    from vllm_torchtpu import envs as E

    assert "VLLM_TPU_QSA_DCP_Q_ONLY_GATHER" in E.environment_variables
