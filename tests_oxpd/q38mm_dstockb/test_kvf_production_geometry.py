# SPDX-License-Identifier: Apache-2.0
"""q38mm DSTOCK-B, KVF: ``VLLM_TPU_QSA_FUSED_KV_WRITE`` at the c48 production geometry.

The fused decode/verify writer (e03b38650 + ebbba8253, default off) replaces the main-KV paired-slot
writer and the compressed-key tiled writer of one QSA layer. Its own tests run with the ring options
off, no DCP on the compressed keys and 8-request steps. This file holds it to the served c48 env
(``mm/scratch/host/q38mm_c48_env.txt``): QSA_COMPRESSED_PAIR_SHIFT=1, QSA_RING_SELECT=1,
QSA_RING_DMA_BATCH=32, QSA_COMPRESSED_RUN_WRITE=1 (min 512 tokens, so decode keeps the tiled writer),
MAIN_KV_DCP=2 with COMPRESSED_DCP=1 (both ranks), packed main KV (main_slot 0/1) and packed
compressed keys, FP8 pool [P, 512, 1, 4, 256], max_query_len 4, and 32 / 24 / 16 requests x 4 verify
rows (128 / 96 / 64 tokens) at contexts near 92,672 on every page / pair / group phase.

Off and on run ``qsa_cache_writes`` on the same inputs; the pool bytes and the returned (req, logical,
valid) must be identical, the off arm must not call the fused writer and the on arm must call it once.
Pallas interpret mode on CPU (TPU7x info), as the repo's own fused-writer tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa

B, WIDTH, INDEX_DIM, RATIO, ROTARY, D = 512, 256, 128, 4, 64, 2
K_SCALE, V_SCALE = 0.375, 2.5
FP8 = jnp.float8_e4m3fn

PROD_ENV = {
    "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT": True,
    "VLLM_TPU_QSA_RING_SELECT": True,
    "VLLM_TPU_QSA_RING_DMA_BATCH": 32,
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE": True,
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS": 512,
}


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for name, value in PROD_ENV.items():
        monkeypatch.setattr(envs, name, value)


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def random_fp8(rng, shape):
    """Finite FP8 E4M3 bytes (|x| in [2^-4, 3.75]); the DCP-2 harness's generator."""
    sign = rng.integers(0, 2, shape, dtype=np.uint8) << 7
    exp = rng.integers(3, 9, shape, dtype=np.uint8) << 3
    mant = rng.integers(0, 8, shape, dtype=np.uint8)
    return jax.lax.bitcast_convert_type(jnp.asarray(sign | exp | mant), FP8)


# Context offsets from 92,672 (= 181 x 512) cover every ring/group phase (mod 4), pair phase (mod 8),
# B-page crossings inside the 4 verify rows (509..511 mod 512) and 2B-entry crossings (DCP owner
# flips): requests alternate owners through the offsets.
OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7, 509, 510, 511, 512, 513, 1021, 1022, 1023, 1024, 1025,
           37, 254, 255, 256, 300, 767, 768, 769, 900, 1000, 1500, 1535, 1536, 2047]
BASE_CTX = 92672


def make_step(seed, requests, idle=()):
    """Operands of one decode/verify step: ``requests`` slots, 4 rows each (0 for ``idle`` slots).
    Tables are DCP-2 tables (one entry covers 2B tokens); entries the step never touches point to
    page 0, the null page; the touched entries get distinct random pages."""
    rng = np.random.default_rng(seed)
    query_lens = np.array([0 if r in idle else 4 for r in range(requests)], np.int32)
    contexts = np.array([BASE_CTX + OFFSETS[r % len(OFFSETS)] for r in range(requests)], np.int32)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    tokens = max(8, -(-int(starts[-1]) // 8) * 8)
    span = D * B
    cols = int((contexts.max() + 4) // span) + 2
    pages = 1 + requests * 6
    free = iter(1 + rng.permutation(pages - 1))
    main = np.zeros((requests, cols), np.int32)
    comp = np.zeros((requests, cols), np.int32)
    ring = np.zeros((requests,), np.int32)
    for r in range(requests):
        lo, hi = contexts[r] // span, (contexts[r] + 3) // span
        for c in range(lo, hi + 1):
            main[r, c] = next(free)
            comp[r, c] = next(free)
        ring[r] = next(free)
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r]:starts[r + 1]] = contexts[r] + np.arange(query_lens[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    return dict(
        pool=random_fp8(rng, (pages, B, 1, 4, WIDTH)),
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
    )


def run_writes(monkeypatch, fused, step, *, rank, main_slot, compressed_slot):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", fused)
    calls = []
    original = fused_kv_write.fused_layer_write

    def recording(*args, **kwargs):
        calls.append(len(kwargs["main_rows"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pooled_qsa, "fused_layer_write", recording)
    jax.clear_caches()
    s = step
    dcp = (jnp.asarray([rank], jnp.int32), D)
    out = pooled_qsa.qsa_cache_writes(
        s["pool"], s["k"], s["v"], s["raw"], s["positions"], s["nik"], s["trig"], s["starts"],
        s["context"], s["ring_ids"], s["main_table"], s["compressed_table"],
        compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, k_scale=K_SCALE, v_scale=V_SCALE,
        compressed_slot=compressed_slot, main_slot=main_slot, max_query_len=4,
        dcp=dcp, compressed_dcp=dcp,
    )
    return [raw(out[0])] + [np.asarray(x) for x in out[1:]], calls


def check(monkeypatch, step, *, rank, main_slot, compressed_slot):
    before = raw(step["pool"])
    base, base_calls = run_writes(monkeypatch, False, step, rank=rank, main_slot=main_slot,
                                  compressed_slot=compressed_slot)
    got, calls = run_writes(monkeypatch, True, step, rank=rank, main_slot=main_slot,
                            compressed_slot=compressed_slot)
    assert base_calls == [], "off arm must not use the fused writer"
    assert len(calls) == 1, f"on arm must use the fused writer once, got {calls}"
    assert not np.array_equal(base[0], before), "nothing written"
    diff = np.argwhere((got[0] != base[0]).reshape(got[0].shape[0], got[0].shape[1], -1).any(-1))
    assert len(diff) == 0, f"{len(diff)} pool rows differ; first (page, row) {diff[:3].tolist()}"
    for a, b in zip(got[1:], base[1:]):
        np.testing.assert_array_equal(a, b)
    return int((base[0] != before).reshape(before.shape[0], before.shape[1], -1).any(-1).sum())


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("main_slot,compressed_slot", [(0, 0), (1, 3)])
def test_c48_shape_128_rows(interpret, monkeypatch, rank, main_slot, compressed_slot):
    """32 requests x 4 verify rows = the c48 at-load decode shape."""
    rows = check(monkeypatch, make_step(11 + rank, 32), rank=rank, main_slot=main_slot,
                 compressed_slot=compressed_slot)
    assert rows >= 32, f"only {rows} pool rows written"


@pytest.mark.parametrize("requests,idle", [(24, ()), (16, ()), (32, (3, 17, 30))])
def test_96_64_rows_and_idle_slots(interpret, monkeypatch, requests, idle):
    check(monkeypatch, make_step(23 + requests, requests, idle), rank=0, main_slot=1,
          compressed_slot=3)


def test_draft_like_one_row_per_request(interpret, monkeypatch):
    """MTP draft loop steps: one row per request (32 rows), whole-row main KV (main_slot -1)."""
    step = make_step(41, 32)
    starts = np.arange(33, dtype=np.int32)
    step = dict(step, starts=jnp.asarray(starts), k=step["k"][:32], v=step["v"][:32],
                raw=step["raw"][:32], positions=step["context"])
    check(monkeypatch, step, rank=1, main_slot=-1, compressed_slot=0)
