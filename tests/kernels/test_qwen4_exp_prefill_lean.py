# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_LEAN: dropping the FP8 key guard keeps sparse prefill bit-identical.

TPU only: the Pallas interpreter cannot run the kernel's bitcast FP8 cache loads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_prefill,
)

TOKENS, PAGES, PAGE, RATIO, TOPK = 32, 4, 128, 4, 12


def _inputs(seed, visible_fill=None):
    rng = np.random.default_rng(seed)
    query = jnp.asarray(rng.normal(size=(TOKENS, 2, 256)), jnp.bfloat16)
    # One request whose last 32 positions are queried; keys past 400 are unwritten.
    lengths = np.arange(400 - TOKENS + 1, 401, dtype=np.int32)
    blocks = np.full((TOKENS, TOPK), -1, np.int32)
    for r, n in enumerate(lengths):
        pick = rng.choice(n // RATIO, size=TOPK - 2, replace=False)
        blocks[r, : TOPK - 2] = np.sort(pick)
    visible = np.zeros(PAGES * PAGE, bool)
    for r, n in enumerate(lengths):
        groups = set(blocks[r][blocks[r] >= 0].tolist())
        if n % RATIO:
            groups.add(n // RATIO)
        for g in groups:
            span = np.arange(g * RATIO, (g + 1) * RATIO)
            visible[span[span < n]] = True
    table = rng.permutation(PAGES).astype(np.int32)
    cache = rng.normal(size=(PAGES, PAGE, 1, 4, 256)).astype(np.float32)
    # Every key and value slot no row can see is NaN: unwritten positions and
    # written positions outside the selections.
    # Logical page i lives in physical page table[i].
    logical = cache[table].reshape(PAGES * PAGE, 1, 4, 256)
    logical[~visible] = np.nan
    if visible_fill is not None:
        logical[np.flatnonzero(visible)[0]] = visible_fill
    cache = logical.reshape(PAGES, PAGE, 1, 4, 256)[np.argsort(table)]
    return (
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(np.broadcast_to(table, (TOKENS, PAGES))),
        jnp.asarray(cache, jnp.float8_e4m3fn),
    )


def _run(monkeypatch, lean, inputs):
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_LEAN", "1" if lean else "0")
    # The flag is read at trace time and the kernel wrapper is jitted: retrace.
    jax.clear_caches()
    out = block_sparse_gqa_prefill(
        *inputs, ratio=RATIO, kv_heads=1, query_block=32, kv_tile=128
    )
    return np.asarray(jax.block_until_ready(out), np.float32)


@pytest.mark.tpu_test
@pytest.mark.parametrize("seed", [0, 1])
def test_lean_key_guard_is_bit_identical(monkeypatch, seed):
    inputs = _inputs(seed)
    base = _run(monkeypatch, False, inputs)
    lean = _run(monkeypatch, True, inputs)
    assert np.isfinite(base).all()
    np.testing.assert_array_equal(lean, base)


@pytest.mark.tpu_test
def test_lean_differs_when_a_visible_key_is_poisoned(monkeypatch):
    # Positive control: the poison reaches the kernel and the guard is live. A
    # NaN visible key and value are read as zeros by the guarded path, while
    # the lean path lets the NaN key into the scores.
    zeroed = _run(monkeypatch, False, _inputs(0, visible_fill=0.0))
    poisoned = _inputs(0, visible_fill=np.nan)
    base = _run(monkeypatch, False, poisoned)
    lean = _run(monkeypatch, True, poisoned)
    print(
        "LEANCTRL base_eq_zeroed", np.array_equal(base, zeroed),
        "base_finite", bool(np.isfinite(base).all()),
        "lean_nan", int(np.isnan(lean).sum()),
        "lean_eq_base", np.array_equal(lean, base),
        "lean_max_abs_diff", float(np.nanmax(np.abs(lean - base))),
    )
    np.testing.assert_array_equal(base, zeroed)
    assert np.isnan(lean).any()
