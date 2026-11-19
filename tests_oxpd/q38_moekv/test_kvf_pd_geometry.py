# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-MOEKV lever 6: ``VLLM_TPU_QSA_FUSED_KV_WRITE`` (KVF) at PD's served decode geometry.

The fused decode/verify writer (dongruizi e03b38650 + ebbba8253, default off) replaces the main-KV
writer and the compressed-key writer of one QSA layer. Above 64 rows PD's O1 decode today runs both
as serial tiled scatters (``qwen4_exp_main_kv_universal_tiled_scatter``,
``qwen4_exp_compressed_kv_page_tiled_scatter``). The repo's own tests stop at 8 requests, and MM's
production-geometry test (q38mm-dstockb d86b935ec, DSTOCK-B; this file follows its structure) covers
DCP-2 with compressed DCP, paired main slots and <= 128 rows. PD's O1 differs: MAIN_KV_DCP=4,
compressed keys replicated (no COMPRESSED_DCP), whole-row main KV (PACK_MAIN_KV=0 behind PP8
prefill), packed compressed keys (DECODE_QWEN4_EXP_PACK_COMPRESSED_K=1: slots 0-7), 512-row pages, and up to 64 requests x 4 verify rows
(256 tokens). O4 (DP4 x TP2) uses 1664-row pages with no DCP.

The env is the served decode leg's QSA flag set: every VLLM_TPU_QSA_* in DECODE_EXTRA_SERVER_ENV of
the FR7 effective params that DS-DP's G2 decode harness uses (COMPRESSED_RUN_WRITE=1 with its
512-token minimum, so decode keeps the tiled writer; RING_SELECT=1; RING_DMA_BATCH=32; ...), plus the
recipe's DECODE_QUERY_BLOCK=4 and DECODE_KV_BLOCK=3328.

Off and on run ``qsa_cache_writes`` on the same inputs. The pool bytes and the returned (req,
logical, valid) must be identical; the off arm must not call the fused writer and the on arm must
call it exactly once. Pallas interpret mode on CPU with TPU7x chip info, as the repo's own tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa

WIDTH, INDEX_DIM, RATIO, ROTARY = 256, 128, 4, 64
K_SCALE, V_SCALE = 0.375, 2.5
FP8 = jnp.float8_e4m3fn

# Served decode QSA flags (FR7 effective params, DECODE_EXTRA_SERVER_ENV) + recipe server env.
SERVED_ENV = {
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE": "1",
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS": "512",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": "1",
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": "1024",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": "1",
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": "1",
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": "1",
    "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD": "1",
    "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT": "1",
    "VLLM_TPU_QSA_PREFIX_SLICE": "1",
    "VLLM_TPU_QSA_RING_DMA_BATCH": "32",
    "VLLM_TPU_QSA_RING_SELECT": "1",
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": "4",
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": "3328",
}


def _typed(name, text):
    current = getattr(envs, name)
    if isinstance(current, bool):
        return text not in ("0", "", "false", "False")
    if isinstance(current, int):
        return int(text)
    return text


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for name, text in SERVED_ENV.items():
        monkeypatch.setattr(envs, name, _typed(name, text))


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def random_fp8(rng, shape):
    """Finite FP8 E4M3 bytes (|x| in [2^-4, 3.75]), as the DCP-2 harness generates them."""
    sign = rng.integers(0, 2, shape, dtype=np.uint8) << 7
    exp = rng.integers(3, 9, shape, dtype=np.uint8) << 3
    mant = rng.integers(0, 8, shape, dtype=np.uint8)
    return jax.lax.bitcast_convert_type(jnp.asarray(sign | exp | mant), FP8)


def offsets(page_rows, dcp):
    """Context offsets that cover every ring/group phase (mod 4), pair phase (mod 8), page
    crossings inside the 4 verify rows and DCP owner flips (every ``page_rows`` tokens), plus
    filler, for up to 64 requests."""
    out = list(range(8))
    for k in range(1, 2 * max(dcp, 1) + 1):
        edge = k * page_rows
        out += [edge - 3, edge - 2, edge - 1, edge, edge + 1]
    filler = [37, 254, 255, 256, 300, 767, 900, 1000, 1500, 2047, 2500, 3071, 3333, 4000, 4095]
    out += filler
    while len(out) < 64:
        out.append(out[len(out) % len(filler)] + 4096 + 13 * len(out))
    return out[:64]


def make_step(seed, requests, *, page_rows, dcp, rows_per_request=4, idle=(), base_ctx=102400,
              pad_to=0):
    """Operands of one decode/verify step: ``requests`` slots of ``rows_per_request`` rows (0 for
    ``idle`` slots). Main tables are DCP tables (one entry covers dcp * page_rows tokens);
    untouched entries point to page 0 (the null page); touched entries get distinct pages."""
    rng = np.random.default_rng(seed)
    offs = offsets(page_rows, dcp)
    # The runner pads request rows to a power of two in [8, max_num_seqs=64]
    # (tpu_runner._get_req_paddings); padded rows are empty with context 0.
    rows = min(max(8, 1 << (requests - 1).bit_length()), max(64, requests))
    q = np.array([0 if (r in idle or r >= requests) else rows_per_request for r in range(rows)],
                 np.int32)
    contexts = np.array([base_ctx + offs[r] if r < requests else 0 for r in range(rows)], np.int32)
    starts = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
    tokens = max(8, pad_to, -(-int(starts[-1]) // 8) * 8)
    # Main KV entries cover dcp * page_rows tokens; compressed keys are not DCP-split on PD, so
    # their entries cover page_rows tokens (pooled_qsa.compressed_pairs, dcp=None).
    span = max(dcp, 1) * page_rows
    cols = int((contexts.max() + rows_per_request) // page_rows) + 2
    pages = 1 + requests * 6
    free = iter(1 + rng.permutation(pages - 1))
    main = np.zeros((rows, cols), np.int32)
    comp = np.zeros((rows, cols), np.int32)
    ring = np.zeros((rows,), np.int32)
    for r in range(requests):
        last = contexts[r] + rows_per_request - 1
        for c in range(contexts[r] // span, last // span + 1):
            main[r, c] = next(free)
        for c in range(contexts[r] // page_rows, last // page_rows + 1):
            comp[r, c] = next(free)
        ring[r] = next(free)
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r]:starts[r + 1]] = contexts[r] + np.arange(q[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    return dict(
        pool=random_fp8(rng, (pages, page_rows, 1, 4, WIDTH)),
        k=normal(tokens, 1, WIDTH),
        v=normal(tokens, 1, WIDTH),
        raw=normal(tokens, INDEX_DIM),
        positions=jnp.asarray(positions),
        nik=normal(INDEX_DIM),
        trig=jnp.pad(normal(8192, ROTARY), ((0, 0), (0, 128 - ROTARY))),
        starts=jnp.asarray(starts),
        context=jnp.asarray(contexts),
        ring_ids=jnp.asarray(ring),
        main_table=jnp.asarray(main),
        compressed_table=jnp.asarray(comp),
        max_query_len=rows_per_request,
    )


def run_writes(monkeypatch, fused, step, *, rank, dcp, cslot):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", fused)
    calls = []
    original = fused_kv_write.fused_layer_write

    def recording(*args, **kwargs):
        calls.append(len(kwargs["main_rows"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "fused_layer_write", recording)
    jax.clear_caches()
    s = step
    extra = {} if dcp <= 1 else {"dcp": (jnp.asarray([rank], jnp.int32), dcp)}
    out = pooled_qsa.qsa_cache_writes(
        s["pool"], s["k"], s["v"], s["raw"], s["positions"], s["nik"], s["trig"], s["starts"],
        s["context"], s["ring_ids"], s["main_table"], s["compressed_table"],
        compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, k_scale=K_SCALE, v_scale=V_SCALE,
        compressed_slot=cslot, main_slot=-1, max_query_len=s["max_query_len"], **extra,
    )
    return [raw(out[0])] + [np.asarray(x) for x in out[1:]], calls


def check(monkeypatch, step, *, rank, dcp, cslot=0):
    """Returns the number of (page, row) pool rows the step changed."""
    before = raw(step["pool"])
    base, base_calls = run_writes(monkeypatch, False, step, rank=rank, dcp=dcp, cslot=cslot)
    got, calls = run_writes(monkeypatch, True, step, rank=rank, dcp=dcp, cslot=cslot)
    assert base_calls == [], "off arm must not use the fused writer"
    assert len(calls) == 1, f"on arm must use the fused writer once, got {calls}"
    shape = (before.shape[0], before.shape[1], -1)
    assert not np.array_equal(base[0], before), "nothing written"
    diff = np.argwhere((got[0] != base[0]).reshape(shape).any(-1))
    assert len(diff) == 0, f"{len(diff)} pool rows differ; first (page, row) {diff[:3].tolist()}"
    for a, b in zip(got[1:], base[1:]):
        np.testing.assert_array_equal(a, b)
    return int((base[0] != before).reshape(shape).any(-1).sum())


# --- O1: TP8 + DCP-4, 512-row pages ------------------------------------------------------------

@pytest.mark.parametrize("rank,cslot", [(0, 0), (1, 7), (2, 3), (3, 5)])
def test_o1_b64_256_rows(interpret, monkeypatch, rank, cslot):
    """64 requests x 4 verify rows: O1 B64 (and B48's m_256 bucket), every DCP-4 rank, packed
    compressed-key slots (served DECODE_QWEN4_EXP_PACK_COMPRESSED_K=1) from 0 to 7."""
    rows = check(monkeypatch, make_step(101 + rank, 64, page_rows=512, dcp=4), rank=rank, dcp=4,
                 cslot=cslot)
    assert rows >= 16, f"only {rows} pool rows written"


@pytest.mark.parametrize("requests,idle,rank,pad_to", [(32, (), 1, 0), (48, (5, 20, 47), 2, 256), (16, (), 3, 0)])
def test_o1_128_192_64_rows_and_idle(interpret, monkeypatch, requests, idle, rank, pad_to):
    """B32 (128 rows); B48 with idle slots, padded to the m_256 bucket's 256 tokens; and B16
    (64 rows, today's XLA-scatter + per-pair path)."""
    check(monkeypatch, make_step(201 + requests, requests, page_rows=512, dcp=4, idle=idle,
                                 pad_to=pad_to), rank=rank, dcp=4, cslot=3)


@pytest.mark.parametrize("requests", [1, 4])
def test_o1_small_batch(interpret, monkeypatch, requests):
    """B1 / B4 (the K megakernel's range shares ``qsa_cache_writes``)."""
    check(monkeypatch, make_step(301 + requests, requests, page_rows=512, dcp=4), rank=0, dcp=4,
          cslot=5)


def test_o1_draft_like_one_row_per_request(interpret, monkeypatch):
    """MTP draft steps: one row per request (64 rows)."""
    step = make_step(401, 64, page_rows=512, dcp=4, rows_per_request=1)
    check(monkeypatch, step, rank=2, dcp=4, cslot=0)


# --- O4: DP4 x TP2, 1664-row pages, no DCP --------------------------------------------------

@pytest.mark.parametrize("requests", [16, 8])
def test_o4_per_engine(interpret, monkeypatch, requests):
    """Host B64 / B32 on O4 = 16 / 8 requests x 4 rows per engine, with runs crossing pages."""
    check(monkeypatch, make_step(501 + requests, requests, page_rows=1664, dcp=1, base_ctx=99840),
          rank=0, dcp=1, cslot=2)
