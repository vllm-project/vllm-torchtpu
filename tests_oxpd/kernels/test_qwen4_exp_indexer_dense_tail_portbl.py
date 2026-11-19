# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL IDX: VLLM_TPU_QSA_INDEXER_DENSE_STEP and VLLM_TPU_QSA_INDEXER_TAIL_SKIP
(ported from Benliang Wang <benliangw@google.com>, commit 2ce02bb1a on
benliangw/q38fh-integ-pin-5b44702aea; his TPU test is
tests/kernels/test_qwen4_exp_indexer_dense_step.py there) on v8's served decode
indexer: the O1s request-sharded selection of one TP rank (8 ranks,
``request_shard_step``) through ``pooled_qsa.select_blocks`` -> shared-stream
window -> the real StreamIndex scoring kernel (TPU interpret mode on CPU) -> a
JAX top-k standing in for the SparseCore selector, with v8's served indexer env
(4-row query block, 3,328-key block / 6,656-key one-window block, pool-page
table + pool-page DMA, mixed pass only, 128 key lanes, word-strided key reads,
no early exit) on the 8-bit carrier pool.

The KV copies are the REAL ``StreamIndexKVBufferedRef.copy_in`` / ``wait_in``
(the routines the tail skip and the lean addresses change): the interpreter's
loads through ref-level reshape / bitcast transforms use the D24 shim
(test_qwen4_exp_d24_strided_keys, credit), which also counts the KV DMAs the
interpreter executes.

BW: flag off vs on at 80 / 96 / 128 / 192 / 256-row steps (all 8 ranks), the
kernel's whole score buffer (int32 score bits, every row and position) and the
selected group ids, bit for bit. The tail skip alone changes no arithmetic, so
it runs on Gaussian data. The dense step re-associates the 4-head sum relative
to the XLA:CPU interpreter's reduction (see MODEL), so its bitwise legs run on
data where every score is exact in any order ("dyadic": small integers, 64-dim
queries, head weight 1/8). Contexts: shorter than one tail group, shorter than
one block, a partial last block, multi-block, exactly on a 6,656-key block
boundary and one group / one key past it, idle seats, padded buckets, requests
cut across ranks; the 6,656-key one-window block and (forced small windows) the
3,328-key multi-window block; compressed slot 0 and 3.

MODEL (dense step; integer keys and queries with per-head power-of-two weights,
so every head product relu(q.k) * w_h is exact and only the 4-head sum's
association rounds): the flag-off kernel equals the CPU interpreter's sequential
sum from +0.0, and the dense kernel equals (h0 + h2) + (h1 + h3) with
-0.0 -> +0.0, bit for bit on every valid score. That is the order Benliang's
dense step takes from the TPU's sublane reduction of the per-token expression
(his TPU test checks dense == per-token on device; CPU cannot). So on CPU the
dense step differs from flag off by that re-association only; CPU_ULP measures
it on Gaussian data at the served head weight. TPU_DEVICE (run only with
QSA_INDEXER_TEST_DEVICE=tpu) is the device byte check.

ENG: per call, the dense-step score loads (4 strided head-sum loads a step) and
the KV DMAs the interpreter executes; with the keys on the dense loads are > 0
and the KV DMAs drop by the skipped tail groups (> 0); with them off, 0 and 0.
"""

import functools
import importlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as _interp
from jax._src.state import types as _state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_d24_strided_keys import _loads_through_transforms, carrier

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

streamindex = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)
bref_override = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.bref_override"
)

DENSE, TAIL = "VLLM_TPU_QSA_INDEXER_DENSE_STEP", "VLLM_TPU_QSA_INDEXER_TAIL_SKIP"
RATIO, ROWS = 4, 32  # compression ratio; compressed keys per kernel page
BLOCK_TOPK = 512  # token_topk 2048 // 4
POOL_PAGES, TOKENS_PER_PAGE, WIDTH, KEY_DIM = 40, 512, 256, 128
TABLE_COLUMNS = 112  # 57,344 tokens: 14,336 compressed keys per request (every context fits)
SLOTS, TP, MTP_ROWS = 64, 8, 4
TAIL_VALUE = 8  # his served value: 8 kernel pages (256 keys) a group
ONE_WINDOW, KV_BLOCK = 6656, 3328

# v8's served decode indexer env (scratch/decL1/s11/v8nt/cd/record_envs/
# v8nt_record_20261116T021053Z.env, the keys that reach select_blocks).
SERVED = {
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": str(KV_BLOCK),
    "VLLM_TPU_QSA_DECODE_KV_BLOCK_ONE_WINDOW": str(ONE_WINDOW),
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
    "VLLM_TPU_QSA_INDEXER_KEY_LANES": "128",
    "VLLM_TPU_QSA_INDEXER_STRIDED_KEYS": "1",
    "VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT": "1",
}

# Per-head indexer weights replacing select_blocks' uniform q_dim ** -0.5 (MODEL).
WEIGHTS_OVERRIDE = None
# Per call: [full score buffer int32 [rows, positions], row_lengths] per window.
RECORD: list[tuple[np.ndarray, np.ndarray]] = []
# Interpreter events: KV DMAs executed, dense-step head-sum loads executed.
COUNTS = {"kv_dma": 0, "dense_loads": 0}


def _record(bits, lengths):
    RECORD.append((np.asarray(bits).copy(), np.asarray(lengths).copy()))


class RecordingTopK:
    """JAX stand-in for the SparseCore selector (lowest ids win ties) that
    also hands the kernel's whole score buffer to the test."""

    overlaps_scoring = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        jax.debug.callback(_record, score_bits, row_lengths)
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None, :] < row_lengths[:, None], scores, -jnp.inf
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(values > -jnp.inf, ids, -1)
        bits = jax.lax.bitcast_convert_type(values, jnp.int32)
        return (ids, bits) if return_scores else ids

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, RecordingTopK)


def _is_kv_dma(transforms):
    # The real copy_in reads every KV page through the flattened HBM view
    # [pages * 16, 4, 256] of the carrier.
    return any(
        isinstance(t, _state_types.ReshapeTransform) and tuple(t.shape[1:]) == (4, WIDTH)
        for t in transforms
    )


def _is_dense_load(transforms):
    # dense_step_body's token gather: sum_ref.reshape(nsub * 8, 128)[pl.ds(t, nsub, stride=8)]
    return (
        len(transforms) == 2
        and isinstance(transforms[0], _state_types.ReshapeTransform)
        and tuple(transforms[0].shape[1:]) == (128,)
        and not any(isinstance(t, _state_types.BitcastTransform) for t in transforms)
    )


def _counting(read, dma):
    def get(token, device_id, local_core_id, memory_space, buffer_id, transforms, *a, **kw):
        if dma and _is_kv_dma(transforms):
            COUNTS["kv_dma"] += 1
        if not dma and _is_dense_load(transforms):
            COUNTS["dense_loads"] += 1
        return read(token, device_id, local_core_id, memory_space, buffer_id, transforms, *a, **kw)

    return get


@pytest.fixture
def served(monkeypatch):
    """Interpret mode with the real KV copies, v8's served indexer env, the
    recording selector and the interpreter counters."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(
        streamindex.pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    seen = []
    original = _interp.get
    get = _counting(_loads_through_transforms(original, seen), dma=False)
    get.__wrapped__ = _counting(_loads_through_transforms(original.__wrapped__, seen), dma=True)
    monkeypatch.setattr(_interp, "get", get)

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, topk_backend=RecordingTopK())
        if WEIGHTS_OVERRIDE is not None:
            args = list(args)
            args[1] = jnp.broadcast_to(jnp.asarray(WEIGHTS_OVERRIDE, jnp.float32), args[1].shape)
        return streamindex.streamindex_topk(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    for name, value in SERVED.items():
        set_env(monkeypatch, name, value)
    set_env(monkeypatch, DENSE, None)
    set_env(monkeypatch, TAIL, None)
    return monkeypatch


def set_env(monkeypatch, name, value):
    if name in vars(envs):  # a frozen module attribute would shadow the env
        monkeypatch.delattr(envs, name)
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def contexts(n, seed):
    """Request contexts covering the tail-skip and block edges (4 rows each:
    seq_len = context + 4, compressed keys = seq_len // 4)."""
    edges = [
        5, 37, 517, 1003,  # shorter than one 256-key tail group
        3000, 9000, 20001,  # shorter than one 6,656-key block
        4 * 6656 - 4,  # exactly one 6,656-key block
        4 * (6656 + 256) - 4,  # one tail group past it
        4 * (6656 + 1) - 4,  # one key past it
        4 * 3328 - 4, 4 * (3328 + 256) - 4,  # the 3,328-key block edges
        27000, 33333, 41000, 45001,  # multi-block, partial last block
        4 * (2 * 6656) - 4,  # two whole blocks
    ]
    rng = np.random.default_rng(seed)
    extra = rng.integers(0, TABLE_COLUMNS * TOKENS_PER_PAGE - 8, size=n).tolist()
    turn = seed % len(edges)
    return (edges[turn:] + edges[:turn] + extra)[:n]


def step(rows, seed, data="dyadic", ragged=False):
    """One verify step of ``rows`` token rows over SLOTS request slots: requests
    of 4 rows (``ragged``: a few of 1-3 rows), padding rows past the live ones."""
    rng = np.random.default_rng(seed)
    q_lens = []
    live = rows - (4 if rows % 64 else 0)  # a padded bucket at 80 / 112-type sizes
    while sum(q_lens) + 4 <= live:
        q_lens.append(4)
    if ragged:
        q_lens[1], q_lens[5] = 2, 3
        q_lens.append(1)
    q_lens += [0] * (SLOTS - len(q_lens))
    q_lens = np.asarray(q_lens[:SLOTS], np.int32)
    ctx = np.zeros(SLOTS, np.int32)
    n = int((q_lens > 0).sum())
    ctx[q_lens > 0] = contexts(n, seed)
    starts = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    req = np.zeros(rows, np.int32)
    lengths = np.zeros(rows, np.int32)
    for r in range(SLOTS):
        b, e = starts[r], starts[r + 1]
        req[b:e] = r
        lengths[b:e] = ctx[r] + np.arange(1, e - b + 1)
    span = TOKENS_PER_PAGE
    table = rng.integers(1, POOL_PAGES, size=(SLOTS, TABLE_COLUMNS)).astype(np.int32)
    for r in range(SLOTS):
        table[r, -(-(ctx[r] + q_lens[r]) // span):] = 0  # vLLM's null-block tail

    def sample(shape):
        if data in ("integer", "dyadic"):
            return rng.integers(-3, 4, size=shape).astype(np.float32)
        return rng.normal(size=shape).astype(np.float32)

    pool = sample((POOL_PAGES, TOKENS_PER_PAGE, 1, 2, WIDTH))
    query = sample((rows, 4, 64 if data == "dyadic" else KEY_DIM))
    return dict(
        query=jnp.asarray(query, jnp.bfloat16),
        pool=carrier(jnp.asarray(pool, jnp.bfloat16)),
        lengths=jnp.asarray(lengths),
        table=jnp.asarray(table),
        req=jnp.asarray(req),
        starts=jnp.asarray(starts),
        context=jnp.asarray(ctx),
    ), pool


def rank_args(full, rows, index):
    """One TP rank's O1s request-shard operands (pooled_qsa._pooled_qsa)."""
    width = rows // TP
    base, sub_starts, sub_context = pooled_qsa.request_shard_step(
        full["starts"], full["context"], jnp.asarray([index], jnp.int32), width, TP, MTP_ROWS
    )

    def own(x):
        return jax.lax.dynamic_slice_in_dim(x, base, width)

    args = dict(full, query=own(full["query"]), lengths=own(full["lengths"]), req=own(full["req"]),
                starts=sub_starts, context=sub_context)
    return args, int(base)


def run(monkeypatch, args, dense, tail, slot=0):
    """select_blocks on one rank's rows; returns (blocks, windows, counts)."""
    set_env(monkeypatch, DENSE, "1" if dense else None)
    set_env(monkeypatch, TAIL, str(tail) if tail else None)
    RECORD.clear()
    COUNTS.update(kv_dma=0, dense_loads=0)
    jax.clear_caches()
    blocks = np.asarray(
        pooled_qsa.select_blocks(**args, ratio=RATIO, block_topk=BLOCK_TOPK, slot=slot, max_query_len=MTP_ROWS)
    )
    jax.effects_barrier()
    return blocks, list(RECORD), dict(COUNTS)


def compare(monkeypatch, args, dense, tail, slot=0):
    off, off_win, off_n = run(monkeypatch, args, False, 0, slot)
    on, on_win, on_n = run(monkeypatch, args, dense, tail, slot)
    assert off_win and len(off_win) == len(on_win), (len(off_win), len(on_win))
    words = differ = neg = finite = 0
    for (b0, l0), (b1, l1) in zip(off_win, on_win):
        np.testing.assert_array_equal(l0, l1)
        assert b0.shape == b1.shape and b0.dtype == b1.dtype == np.int32
        words += b0.size
        differ += int((b0 != b1).sum())
        f0 = b0.view(np.float32)
        neg += int(np.isneginf(f0).sum())
        finite += int(np.isfinite(f0).sum())
    return dict(off=off, on=on, off_n=off_n, on_n=on_n, words=words, differ=differ, neg=neg,
                finite=finite, off_win=off_win, on_win=on_win)


ROW_COUNTS = [80, 96, 128, 192, 256]
# (dense, tail skip, data): the dense step on order-exact data, the tail skip
# alone on Gaussian data (it changes no arithmetic).
ARMS = [(1, TAIL_VALUE, "dyadic"), (1, 0, "dyadic"), (0, TAIL_VALUE, "normal")]


@pytest.mark.parametrize("rows", ROW_COUNTS)
@pytest.mark.parametrize("dense,tail,data", ARMS)
def test_bitwise_and_engaged_on_served_rank_rows(served, rows, dense, tail, data):
    """BW + ENG: every rank of a ``rows``-row verify step, flag off vs on."""
    full, _ = step(rows, seed=rows, data=data, ragged=rows == 96)
    total = dict(words=0, differ=0, neg=0, finite=0, dense_loads=0, kv_off=0, kv_on=0, sel=0)
    blocks_off, blocks_on = [], []
    for index in range(TP):
        args, _ = rank_args(full, rows, index)
        r = compare(served, args, dense, tail)
        for k in ("words", "differ", "neg", "finite"):
            total[k] += r[k]
        assert r["off_n"]["dense_loads"] == 0
        total["dense_loads"] += r["on_n"]["dense_loads"]
        total["kv_off"] += r["off_n"]["kv_dma"]
        total["kv_on"] += r["on_n"]["kv_dma"]
        total["sel"] += int((r["off"] >= 0).sum())
        np.testing.assert_array_equal(r["on"], r["off"])
        blocks_off.append(r["off"])
        blocks_on.append(r["on"])
        if tail:
            assert r["on_n"]["kv_dma"] <= r["off_n"]["kv_dma"]
        else:
            assert r["on_n"]["kv_dma"] == r["off_n"]["kv_dma"]
    # The step's [T, k] selection rebuilt from the 8 ranks (request_shard_blocks).
    step_off = np.asarray(pooled_qsa.request_shard_blocks(jnp.asarray(np.stack(blocks_off)), full["starts"], MTP_ROWS))
    step_on = np.asarray(pooled_qsa.request_shard_blocks(jnp.asarray(np.stack(blocks_on)), full["starts"], MTP_ROWS))
    np.testing.assert_array_equal(step_on, step_off)
    skipped = total["kv_off"] - total["kv_on"]
    print(f"PBL_IDX_BW rows {rows} dense {dense} tail {tail} data {data}: score words {total['words']} "
          f"differing {total['differ']} (-inf {total['neg']} finite {total['finite']}); selected ids "
          f"{total['sel']} equal; step [T,k] equal; ENG dense loads on {total['dense_loads']} off 0; "
          f"KV DMAs off {total['kv_off']} on {total['kv_on']} skipped {skipped}")
    assert total["differ"] == 0
    assert total["finite"] > 0 and total["neg"] > 0 and total["sel"] > 0
    if dense:
        assert total["dense_loads"] > 0
    else:
        assert total["dense_loads"] == 0
    if tail:
        assert skipped > 0
    else:
        assert skipped == 0


@pytest.mark.parametrize("rows,dense,data", [(96, 1, "dyadic"), (192, 1, "dyadic"), (128, 0, "normal")])
def test_bitwise_multi_window_3328_and_slot3(served, rows, dense, data):
    """The 3,328-key block (forced small windows: a multi-window step keeps
    VLLM_TPU_QSA_DECODE_KV_BLOCK) and compressed slot 3 (lean addresses keep
    the slot's page offset), with the tail skip."""
    served.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 8)
    full, _ = step(rows, seed=rows + 1, data=data)
    differ = words = skipped = loads = 0
    for index in (0, 3, 7):
        args, _ = rank_args(full, rows, index)
        r = compare(served, args, dense, TAIL_VALUE, slot=3)
        assert len(r["off_win"]) > 1, "expected a multi-window step"
        np.testing.assert_array_equal(r["on"], r["off"])
        differ += r["differ"]
        words += r["words"]
        skipped += r["off_n"]["kv_dma"] - r["on_n"]["kv_dma"]
        loads += r["on_n"]["dense_loads"]
    print(f"PBL_IDX_BW3328 rows {rows} dense {dense} data {data} slot 3: score words {words} "
          f"differing {differ}; KV DMAs skipped {skipped}; dense loads {loads}")
    assert differ == 0 and skipped > 0
    assert (loads > 0) == bool(dense)


def _head_products(args, pool, weights, slot, positions):
    """f32 relu(q_h . k) * w_h of every row of the window for the first
    ``positions`` compressed positions: [rows, 4, positions] (exact on integer
    data with power-of-two weights)."""
    q = np.asarray(args["query"].astype(jnp.float32))  # [rows, 4, 128]
    starts = np.asarray(args["starts"])
    keys_pp = pool.reshape(-1, ROWS, WIDTH)[:, :, :KEY_DIM]  # [kernel pages, 32, 128]
    table = np.asarray(args["table"])
    out = np.zeros((len(q), 4, positions), np.float32)
    g = np.arange(positions)
    for r in range(len(starts) - 1):
        b, e = starts[r], starts[r + 1]
        if e <= b:
            continue
        kp = table[r, np.minimum(g // 128, table.shape[1] - 1)] * 32 + slot * 4 + (g % 128) // 32
        st = np.einsum("thd,pd->thp", q[b:e], keys_pp[kp, g % 32])  # exact integers
        out[b:e] = np.maximum(st, np.float32(0)) * np.asarray(weights, np.float32)[None, :, None]
    return out


@pytest.mark.parametrize("sign", ["pos", "neg"])
def test_dense_step_is_exactly_the_tpu_head_order(served, sign):
    """MODEL: flag off = the CPU interpreter's sequential head sum and the dense
    step = (h0 + h2) + (h1 + h3), bit for bit, on every valid score of every
    rank of a 96-row step (head products exact, so only the association rounds).
    "neg": negative weights, so a head with relu(q.k) = 0 gives -0.0 and an
    all-zero score is a sum of four -0.0 terms (the dense step's -0.0 -> +0.0)."""
    rows = 96
    weights = (1.0, 2.0**-20, 2.0**-10, 2.0**-30)
    if sign == "neg":
        weights = tuple(-w for w in weights)
    served.setattr(sys.modules[__name__], "WEIGHTS_OVERRIDE", weights)
    full, pool = step(rows, seed=5, data="integer")
    checked = off_seq = on_tpu = on_off = neg_zero = 0
    for index in range(TP):
        args, _ = rank_args(full, rows, index)
        r = compare(served, args, 1, 0)
        (b0, _), (b1, _) = r["off_win"][0], r["on_win"][0]
        f0, f1 = b0.view(np.float32), b1.view(np.float32)
        p = _head_products(args, pool, weights, 0, f0.shape[1])
        zero = np.float32(0)
        seq = (((zero + p[:, 0]) + p[:, 1]) + p[:, 2]) + p[:, 3]
        tpu = (p[:, 0] + p[:, 2]) + (p[:, 1] + p[:, 3])
        live = tpu[np.isfinite(f0)]
        neg_zero += int(np.signbit(live).sum() - (live < 0).sum())  # -0.0 sums
        tpu = np.where(tpu == 0, zero, tpu)
        valid = np.isfinite(f0)
        assert np.array_equal(valid, np.isfinite(f1))
        checked += int(valid.sum())
        off_seq += int((b0[valid] == seq.view(np.int32)[valid]).sum())
        on_tpu += int((b1[valid] == tpu.view(np.int32)[valid]).sum())
        on_off += int((b0[valid] == b1[valid]).sum())
        np.testing.assert_array_equal(b0[~valid], b1[~valid])
    print(f"PBL_IDX_MODEL {sign} rows {rows}: valid scores {checked}; flag off == sequential "
          f"{off_seq}; dense == (h0+h2)+(h1+h3) {on_tpu}; dense == flag off {on_off} "
          f"(differ by association only: {checked - on_off}); -0.0 pair sums {neg_zero}")
    assert checked > 0
    assert off_seq == checked
    assert on_tpu == checked
    assert on_off < checked  # the weights make the association visible
    if sign == "neg":
        assert neg_zero > 0  # the -0.0 -> +0.0 rule is exercised


def test_dense_step_cpu_ulp_on_normal_data(served):
    """CPU_ULP (measurement): Gaussian data at the served head weight. On the
    CPU interpreter the dense step's (h0 + h2) + (h1 + h3) differs from the
    sequential flag-off sum in the last bits of some scores; report the count
    and the max ULP distance (same-sign finite scores), and the selection
    agreement. On TPU both arms take the same order (TPU_DEVICE checks)."""
    rows = 128
    full, _ = step(rows, seed=11, data="normal")
    words = differ = max_ulp = sel = sel_eq = 0
    for index in range(TP):
        args, _ = rank_args(full, rows, index)
        r = compare(served, args, 1, 0)
        for (b0, _), (b1, _) in zip(r["off_win"], r["on_win"]):
            f0 = b0.view(np.float32)
            valid = np.isfinite(f0)
            np.testing.assert_array_equal(b0[~valid], b1[~valid])
            d = np.abs(b0[valid].astype(np.int64) - b1[valid].astype(np.int64))
            same_sign = (b0[valid] >= 0) == (b1[valid] >= 0)
            assert same_sign.all()
            words += int(valid.sum())
            differ += int((d > 0).sum())
            max_ulp = max(max_ulp, int(d.max(initial=0)))
        sel += int((r["off"] >= 0).sum())
        sel_eq += int(((r["off"] == r["on"]) & (r["off"] >= 0)).sum())
    print(f"PBL_IDX_CPU_ULP rows {rows}: valid scores {words}, differing {differ}, max ulp {max_ulp}; "
          f"selected ids {sel}, same slot and id {sel_eq}")
    assert words > 0 and max_ulp <= 2


@pytest.mark.skipif(
    os.environ.get("QSA_INDEXER_TEST_DEVICE", "cpu").lower() != "tpu",
    reason="QSA_INDEXER_TEST_DEVICE=tpu only (the device byte check)",
)
@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_tpu_device_bitwise_normal_data(monkeypatch, rows):
    """TPU_DEVICE: the same served rank calls on a TPU (no interpret mode),
    Gaussian data at the served head weight, both keys on: the whole score
    buffers and selections must be bit-equal."""

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, topk_backend=RecordingTopK())
        return streamindex.streamindex_topk(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    for name, value in SERVED.items():
        set_env(monkeypatch, name, value)
    full, _ = step(rows, seed=rows + 7, data="normal")
    differ = words = 0
    for index in range(TP):
        args, _ = rank_args(full, rows, index)
        r = compare(monkeypatch, args, 1, TAIL_VALUE)
        np.testing.assert_array_equal(r["on"], r["off"])
        differ += r["differ"]
        words += r["words"]
    print(f"PBL_IDX_TPU rows {rows}: score words {words} differing {differ}")
    assert differ == 0


def _tail_only_bitwise(monkeypatch, rows, tag):
    """The served rank calls with the TAIL SKIP ALONE (dense step OFF, VLLM_TPU_QSA_INDEXER_TAIL_SKIP=TAIL_VALUE) on the
    same Gaussian data as test_tpu_device_bitwise_normal_data: whole score buffers and selections must be bit-equal to
    the base path (both keys off), on every rank, at every served row count."""
    full, _ = step(rows, seed=rows + 7, data="normal")
    differ = words = 0
    for index in range(TP):
        args, _ = rank_args(full, rows, index)
        r = compare(monkeypatch, args, 0, TAIL_VALUE)
        np.testing.assert_array_equal(r["on"], r["off"])
        differ += r["differ"]
        words += r["words"]
    print(f"{tag} rows {rows} dense 0 tail {TAIL_VALUE}: score words {words} differing {differ}")
    assert words > 0
    assert differ == 0


@pytest.mark.skipif(
    os.environ.get("QSA_INDEXER_TEST_DEVICE", "cpu").lower() != "tpu",
    reason="QSA_INDEXER_TEST_DEVICE=tpu only (the device byte check)",
)
@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_tpu_device_bitwise_tail_only_normal_data(monkeypatch, rows):
    """TPU_DEVICE, tail skip alone (q38 HIGHC-HC3CAR, HIGHC-LEAD decider 3b): T93w's prestep ran
    test_tpu_device_bitwise_normal_data with BOTH keys on and read FAIL at 80 / 96 / 128 rows, which does not decide
    the tail skip (the dense step is NB). This is the same device call with the dense step OFF: PASS at every row count
    -> the tail skip is device-bitwise; any FAIL -> the tail skip is NB on device."""

    def scored(*args, **kwargs):
        kwargs.update(enable_early_exit=False, topk_backend=RecordingTopK())
        return streamindex.streamindex_topk(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", scored)
    for name, value in SERVED.items():
        set_env(monkeypatch, name, value)
    _tail_only_bitwise(monkeypatch, rows, "PBL_IDX_TPU_TAIL")


@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_cpu_interpret_tail_only_normal_data(served, rows):
    """The CPU (Pallas interpret) twin of test_tpu_device_bitwise_tail_only_normal_data: the same body and data under
    the served fixture, so the device test's comparison is exercised (and mutation-checked) on every CPU run."""
    _tail_only_bitwise(served, rows, "PBL_IDX_CPU_TAIL")


def test_flags_registered_default_off_and_tagged(monkeypatch):
    """Both keys are registered, default off, and appear in the kernel name only
    when set ("-ds", "-ts8"; the served lowering, as the D24 name test)."""
    assert DENSE in envs.environment_variables and TAIL in envs.environment_variables
    for name in (DENSE, TAIL):
        set_env(monkeypatch, name, None)
    assert envs.environment_variables[DENSE]() is False
    assert envs.environment_variables[TAIL]() == 0
    assert pooled_qsa.indexer_port_kwargs() == {}
    from test_qwen4_exp_d24_strided_keys import _lower_select, _streamidx_kernels

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for name, value in SERVED.items():
        set_env(monkeypatch, name, value)
    names = {}
    for dense, tail in ((0, 0), (1, 0), (0, TAIL_VALUE), (1, TAIL_VALUE)):
        set_env(monkeypatch, DENSE, "1" if dense else None)
        set_env(monkeypatch, TAIL, str(tail) if tail else None)
        jax.clear_caches()
        kernels = _streamidx_kernels(_lower_select(64))
        assert kernels, "no StreamIdx kernel"
        names[(dense, tail)] = kernels
        for k in kernels:
            parts = k.split("-")
            assert ("ds" in parts) == bool(dense), kernels
            assert (f"ts{TAIL_VALUE}" in parts) == bool(tail), kernels
    print("PBL_IDX_NAMES", names)
