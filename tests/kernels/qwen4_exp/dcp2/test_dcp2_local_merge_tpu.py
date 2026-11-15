# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE on TPU (DCP2_HARNESS_DEVICE=tpu; skipped elsewhere).

One verify step of a compressed-DCP QSA layer on every rank of a DCP group of
4 (real owned-group StreamIndex scoring, SparseCore top-k, the pair merge's
SparseCore top-k, cache writers and batched RPA), with the flag off and on, on
the same inputs:

- select step: the same pool bytes and local lists (``vals``, ``pos``); the
  padded score rows change nothing;
- attend step with the local merge (the layers whose selection nobody reads):
  the same pool bytes, attention output and LSE bit for bit as the flag-off
  global merge, on every rank;
- attend step with the global merge under the flag (the layer the MTP draft
  reads, ``global_selection``): ``selected`` byte-identical as well.

Inputs: 15 verify seats (4 rows each, one idle seat) with contexts from 37
tokens (fewer valid groups than the top-k) through partial rank lists to full
ones. ``zero_keys`` zeroes most compressed-key kernel pages, so most groups
score exactly 0 and the boundary falls inside those ties (the position cut
decides); ``zero_query`` zeroes four seats' index queries, so every group of
those rows ties.

Run it in the served decode env, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=env DCP2_HARNESS_DEVICE=tpu python -m pytest -q -s
tests/kernels/qwen4_exp/dcp2/test_dcp2_local_merge_tpu.py"``.
"""

import functools

import dcp2_harness
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_dcp2_op
from dcp2_harness import ON_TPU, from_raw, poison_tail_rows, project_rank_pool, rank_operand, raw
from test_dcp2_op import ROTARY, _args, make_step

from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

DCP = 4  # the served decode's DCP group
RATIO = 4
TOPK = 2048  # tokens: the served 512 compressed groups
K_SCALE, V_SCALE = 0.375, 2.5
SERVED_DECODE_ENV = {
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": "3328",
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_COMPRESSED_DCP": "1",
}
# 37 and 1500: fewer than 512 valid groups; 2050-8100: more than 512 groups,
# rank lists not full; 9000 and up: full rank lists. Positions stay below the
# harness's 16384-row rope table.
CONTEXTS = [37, 1500, 2050, 2600, 3100, 4100, 5000, 6100, 7000, 8100, 9000, 10500, 12000,
            13500, 15000, 16000]
QUERY_LENS = [4] * 9 + [0] + [4] * 6
CASES = {
    # variant, main_slot (two layers share a main KV page), windows, SC overlap
    "random-s0-w1-o0": ("random", 0, 1, 0),
    "random-s1-w2-o2": ("random", 1, 2, 2),
    "zero_keys-s0-w1-o2": ("zero_keys", 0, 1, 2),
    "zero_keys-s1-w2-o0": ("zero_keys", 1, 2, 0),
    "zero_query-s0-w2-o2": ("zero_query", 0, 2, 2),
    "zero_query-s1-w1-o0": ("zero_query", 1, 1, 0),
}


def _zero_most_keys(full, comp, seed):
    """Zero 7 of 8 compressed-key kernel pages (32 groups each) of every
    compressed page; the rest keep their random keys."""
    data = raw(full).copy()
    pages = np.unique(np.asarray(comp))
    view = data[pages].reshape(len(pages), -1, 16 * 4 * data.shape[-1])
    keep = np.random.default_rng(seed).random(view.shape[:2]) < 1 / 8
    view[~keep] = 0
    data[pages] = view.reshape(data[pages].shape)
    return from_raw(data)


def _bits(x):
    x = np.asarray(x)
    return x.view(np.uint32) if x.dtype.itemsize == 4 else raw(x)


def _boundary_stats(vals_pair, pos_pair, lengths):
    """Rows (with keys) that have fewer than k valid candidates, and rows whose
    k-th candidate ties with others across the cut (the position decides)."""
    k = TOPK // RATIO
    vals = np.moveaxis(np.asarray(vals_pair), 0, 1).reshape(vals_pair.shape[1], -1)
    pos = np.moveaxis(np.asarray(pos_pair), 0, 1).reshape(pos_pair.shape[1], -1)
    key = np.asarray(dcp_select.monotone_key(jnp.asarray(vals.view(np.int32))))
    short = tied = 0
    for row in range(len(key)):
        if lengths[row] < RATIO:
            continue
        valid = key[row][pos[row] != dcp_select.EMPTY]
        if len(valid) < k:
            short += 1
            continue
        thr = np.sort(valid)[::-1][k - 1]
        if np.sum(valid > thr) + np.sum(valid == thr) > k:
            tied += 1
    return short, tied


@pytest.mark.parametrize("case", sorted(CASES))
def test_tpu_local_merge_attends_the_same(monkeypatch, case):
    variant, main_slot, windows, overlap = CASES[case]
    monkeypatch.setattr(dcp2_harness, "D", DCP)
    monkeypatch.setattr(test_dcp2_op, "D", DCP)
    for name, value in SERVED_DECODE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLLM_TPU_QSA_DECODE_SC_OVERLAP", str(overlap))
    if windows == 2:
        # 60 rows in two 32-row windows (the 256-row bucket's shape at 128 each).
        monkeypatch.setattr(pooled_qsa, "decode_query_chunk_size", lambda pages: 32)
    layout, full, keep, qgate, s, m, seq, starts = make_step(11 + windows, QUERY_LENS, CONTEXTS)
    if variant == "zero_keys":
        full = _zero_most_keys(full, m["compressed_table"], 5)
    if variant == "zero_query":
        index_qk = s["index_qk"]
        for seat in (1, 4, 11, 15):
            index_qk = index_qk.at[int(starts[seat]) : int(starts[seat + 1])].set(0)
        s = {**s, "index_qk": index_qk}
    pools = [
        poison_tail_rows(project_rank_pool(raw(full), layout, r, keep_pages=keep), layout, r, seq)
        for r in range(DCP)
    ]
    pair_qgate = jnp.concatenate(qgate, axis=1)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=main_slot, max_query_len=4)
    att = dict(kv_heads=1, compress_ratio=RATIO, token_topk=TOPK, k_scale=K_SCALE,
               v_scale=V_SCALE, main_slot=main_slot, max_query_len=4, dcp_size=DCP)
    tokens = int(starts[-1])

    def select(flag):
        monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE", str(flag))
        step = jax.jit(functools.partial(
            pooled_qsa.pooled_qsa_compressed_dcp_select_step, dcp_size=DCP, **opts))
        return [step(*_args(pair_qgate, s, from_raw(pools[r]), m, layout.dcp_table),
                     rank_operand(r)) for r in range(DCP)]

    def attend(selects, local_merge):
        step = jax.jit(functools.partial(
            pooled_qsa.pooled_qsa_compressed_dcp_attend_step, local_merge=local_merge, **att))
        vals_pair = jnp.stack([x[6] for x in selects])
        pos_pair = jnp.stack([x[7] for x in selects])
        out = []
        for r in range(DCP):
            pool, q, gate, lengths, req, st = selects[r][:6]
            out.append([np.asarray(x) for x in step(
                q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"],
                pool, vals_pair, pos_pair, rank_operand(r))])
        return out, vals_pair, pos_pair

    base_sel = select(0)
    base, vals_off, pos_off = attend(base_sel, False)
    flag_sel = select(1)
    local, vals_on, pos_on = attend(flag_sel, True)
    sink, _, _ = attend(flag_sel, False)

    lengths = np.asarray(base_sel[0][3])
    short, tied = _boundary_stats(vals_off, pos_off, lengths)
    kept = [int(np.sum(np.asarray(dcp_select.keep_mask(
        vals_off[r], pos_off[r], *dcp_select.merge_threshold(vals_off, pos_off, TOPK // RATIO))))
        ) for r in range(DCP)]
    filled = [int(np.sum(np.asarray(pos_off[r]) != dcp_select.EMPTY)) for r in range(DCP)]
    report = {"rows": tokens, "short rows": short, "tied-boundary rows": tied,
              "kept/filled per rank": list(zip(kept, filled))}
    failures = []
    for r in range(DCP):
        if not np.array_equal(raw(flag_sel[r][0]), raw(base_sel[r][0])):
            failures.append(f"rank {r}: select pool bytes differ")
        for i, name in ((6, "vals"), (7, "pos")):
            if not np.array_equal(_bits(flag_sel[r][i]), _bits(base_sel[r][i])):
                failures.append(f"rank {r}: local list {name} differs")
        for got, label in ((local, "local"), (sink, "sink")):
            for i, name in ((0, "pool"), (1, "out"), (2, "lse")):
                same = np.array_equal(_bits(got[r][i]), _bits(base[r][i]))
                report[f"rank {r} {label} {name} bit-equal"] = same
                if not same:
                    failures.append(f"rank {r} {label}: {name} differs")
        if not np.array_equal(sink[r][3], base[r][3]):
            failures.append(f"rank {r} sink: selected differs")
        out = base[r][1][:tokens]
        if np.isnan(out).any():
            failures.append(f"rank {r}: NaN in the flag-off output")
        # The local `selected` holds the global one's owned tokens.
        page = layout.page_size
        for t in range(tokens):
            def owned(row):
                ids = row[row >= 0]
                return sorted(ids[(ids // page) % DCP == r].tolist())

            if owned(local[r][3][t]) != owned(base[r][3][t]):
                failures.append(f"rank {r} row {t}: local selected tokens differ")
                break
    print(case, report)
    assert not failures, failures
    assert short > 0, "no row with fewer than k valid candidates"
    if variant != "random":
        assert tied > 0, "no row with ties across the boundary"
    assert any(k < f for k, f in zip(kept, filled)), "the merge dropped nothing"
