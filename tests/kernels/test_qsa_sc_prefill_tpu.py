# SPDX-License-Identifier: Apache-2.0
"""TPU accuracy and speed of the SparseCore QSA prefill (VLLM_TPU_QSA_PREFILL_SC).

Runs ``qsa_rpa._prefill_attention`` (our prefill call site) with the flag off
(TensorCore prefill kernel) and on (SparseCore pipeline) at the served
geometry: per-rank 3 query heads / 1 KV head (6 query heads under DCP-2),
D256, FP8 paired main-KV pages of 512 tokens, ratio 4, budget 2048 tokens.

    PYTHONPATH=src:tests/kernels python -m pytest -q -m tpu_test \
        tests/kernels/test_qsa_sc_prefill_tpu.py -s

Set the served QSA prefill env (VLLM_TPU_QSA_PREFILL_LEAN, ..._BITSET_OWNED,
..._LIVE_MASK_PADDED, ...) to time the served TensorCore path.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

RATIO, PAGE, BUDGET, D = 4, 512, 512, 256


def _tpu7x():
    from jax.experimental.pallas import tpu as pltpu

    try:
        return pltpu.get_tpu_info().chip_version == pltpu.ChipVersion.TPU_7X
    except Exception:
        return False


def make_case(tokens, context, heads, *, world=1, seed=0, pattern="uniform"):
    """One prefill chunk of ``tokens`` rows after ``context`` cached tokens.

    Returns query, gate, blocks, lengths, starts and, per DCP rank, the rank's
    (table, cache); world 1 gives the plain layout."""
    rng = np.random.default_rng(seed)
    total = context + tokens
    lengths = np.arange(context + 1, total + 1, dtype=np.int32)
    blocks = np.full((tokens, BUDGET), -1, np.int32)
    for r, n in enumerate(lengths):
        groups = n // RATIO
        count = min(groups, BUDGET)
        if pattern == "recent":
            recent = np.arange(max(0, groups - count // 2), groups)
            rest = rng.choice(max(1, groups - len(recent)), count - len(recent), replace=False)
            pick = np.concatenate([recent, rest])
        else:
            pick = rng.choice(groups, count, replace=False)
        blocks[r, :count] = pick
    global_pages = -(-total // PAGE)
    kv = rng.normal(0, 0.5, (global_pages * PAGE, 1, 4, D)).astype(np.float32)
    ranks = []
    for rank in range(world):
        owned = np.arange(rank, global_pages, world)
        pool = len(owned) + 7
        physical = rng.permutation(pool)[: len(owned)]
        cache = np.zeros((pool, PAGE, 1, 4, D), np.float32)
        for entry, g in enumerate(owned):
            cache[physical[entry]] = kv[g * PAGE : (g + 1) * PAGE]
        cols = len(owned) + 2
        row = np.zeros((cols,), np.int32)
        row[: len(owned)] = physical
        table = np.broadcast_to(row, (tokens, cols)).copy()
        ranks.append((jnp.asarray(table), jnp.asarray(cache, jnp.float8_e4m3fn)))
    query = jnp.asarray(rng.normal(0, 1, (tokens, heads, D)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, (tokens, heads, D)), jnp.bfloat16)
    starts = jnp.asarray([0, 0, tokens, tokens], jnp.int32)
    return query, gate, jnp.asarray(blocks), jnp.asarray(lengths), starts, ranks


def run(monkeypatch, sc, case, *, rank=0, world=1, offset=0, iters=0):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_SC", "1" if sc else "0")
    query, gate, blocks, lengths, starts, ranks = case
    table, cache = ranks[rank]
    dcp = (jnp.asarray([rank], jnp.int32), world) if world > 1 else None

    @jax.jit
    def call(query, gate, blocks, lengths, table, cache, starts):
        return qsa_rpa._prefill_attention(
            query, gate, blocks, lengths, table, cache, starts,
            kv_heads=1, ratio=RATIO, kv_head_offset=offset,
            dcp=dcp, return_lse=dcp is not None,
        )[:-1]

    args = (query, gate, blocks, lengths, table, cache, starts)
    before = np.asarray(cache).view(np.uint8).copy() if not iters else None
    out = jax.block_until_ready(call(*args))
    if before is not None:
        np.testing.assert_array_equal(np.asarray(cache).view(np.uint8), before)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(call(*args))
        times.append((time.perf_counter() - t0) * 1e3)
    out = [np.asarray(x, np.float32) for x in out]
    return out, (float(np.median(times)) if times else None)


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


@pytest.mark.tpu_test
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("context", [700, 20000])
def test_sc_matches_tensorcore(monkeypatch, offset, context):
    if not _tpu7x():
        pytest.skip("SparseCore prefill targets TPU7x")
    case = make_case(1024, context, 3, seed=context + offset)
    (tc,), _ = run(monkeypatch, False, case, offset=offset)
    (sc,), _ = run(monkeypatch, True, case, offset=offset)
    assert np.isfinite(sc).all()
    assert rel(sc, tc) < 1e-2, rel(sc, tc)


@pytest.mark.tpu_test
@pytest.mark.parametrize("context", [700, 20000])
def test_sc_dcp_matches_tensorcore(monkeypatch, context):
    if not _tpu7x():
        pytest.skip("SparseCore prefill targets TPU7x")
    case = make_case(1024, context, 6, world=2, seed=context)
    for rank in range(2):
        (tc_out, tc_lse), _ = run(monkeypatch, False, case, rank=rank, world=2, offset=1)
        (sc_out, sc_lse), _ = run(monkeypatch, True, case, rank=rank, world=2, offset=1)
        np.testing.assert_array_equal(np.isfinite(sc_lse), np.isfinite(tc_lse))
        live = np.isfinite(tc_lse)
        assert np.max(np.abs(sc_lse[live] - tc_lse[live])) < 2e-3
        assert rel(sc_out, tc_out) < 1e-2, rel(sc_out, tc_out)


@pytest.mark.tpu_test
@pytest.mark.parametrize("pattern", ["uniform", "recent"])
@pytest.mark.parametrize("context", [65536, 131072])
@pytest.mark.parametrize("world", [1, 2])
def test_sc_prefill_speed(monkeypatch, pattern, context, world):
    """Prints ms per call (one QSA layer, one 8K chunk); no pass/fail on speed."""
    if not _tpu7x():
        pytest.skip("SparseCore prefill targets TPU7x")
    heads = 3 * world
    case = make_case(8192, context, heads, world=world, pattern=pattern)
    _, tc_ms = run(monkeypatch, False, case, world=world, offset=1, iters=5)
    _, sc_ms = run(monkeypatch, True, case, world=world, offset=1, iters=5)
    print(
        f"\nSC_PREFILL_SPEED world={world} ctx={context} pattern={pattern} "
        f"tc={tc_ms:.2f}ms sc={sc_ms:.2f}ms per layer-chunk"
    )
