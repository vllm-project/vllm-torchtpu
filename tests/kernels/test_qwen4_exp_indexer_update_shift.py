# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT: the gather-free indexer update equals the gather path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import qsa


def case(seed, rows_are_slots):
    """Requests with mixed contexts (group phases), one idle request, trailing padding."""
    rng = np.random.default_rng(seed)
    nreq, D, rows, ratio = 6, 16, 8, 4
    lens = rng.integers(0, 40, nreq); lens[2] = 0  # idle request
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    count = int(starts[-1]) + 5  # padding tokens
    context = rng.integers(0, 300, nreq).astype(np.int32); context[0] = 0
    slots = np.where(lens > 0, rng.permutation(nreq), -1).astype(np.int32)
    if rows_are_slots: slots = np.where(lens > 0, np.arange(nreq), -1).astype(np.int32)
    raw = jnp.asarray(rng.normal(size=(count, D)), jnp.bfloat16)
    positions = jnp.asarray(rng.integers(0, 262144, count), jnp.int32)
    ring = jnp.asarray(rng.normal(size=(nreq, rows, D)), jnp.bfloat16)
    tags = jnp.asarray(rng.integers(-1, 262144, (nreq, rows, 2)), jnp.int32)
    nk = jnp.asarray(rng.normal(size=(D,)), jnp.bfloat16)
    trig = jnp.asarray(rng.normal(size=(300000, 16)), jnp.bfloat16)
    return (raw, positions, ring, tags, nk, trig, jnp.asarray(starts), jnp.asarray(context), jnp.asarray(slots), ratio, 1e-6, 8), rows_are_slots

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ras", [False, True])
def test_shift_path_matches_gathers(seed, ras, monkeypatch):
    args, ras = case(seed, ras)
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(envs, "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT", flag)
        outs.append(jax.jit(lambda *a: qsa._indexer_updates(*a[:-3], a[-3], a[-2], a[-1], rows_are_slots=ras), static_argnums=(9, 10, 11))(*args))
    complete = np.asarray(outs[0][1])
    for i, (a, b) in enumerate(zip(*outs)):
        a, b = np.asarray(a), np.asarray(b)
        if i == 0:  # completed keys: only completed rows are used downstream
            a, b = a[complete], b[complete]
        np.testing.assert_array_equal(a.view(np.uint8) if a.dtype == jnp.bfloat16 else a, b.view(np.uint8) if b.dtype == jnp.bfloat16 else b, err_msg=str(i))

def test_fault_ignoring_the_phase_is_caught(monkeypatch):
    args, ras = case(1, False)
    monkeypatch.setattr(qsa, "_select_shift", lambda x, phase, j, ratio: qsa._shift_clip(x, j))
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(envs, "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT", flag)
        outs.append(qsa._indexer_updates(*args[:-3], args[-3], args[-2], args[-1], rows_are_slots=ras))
    c = np.asarray(outs[0][1])
    assert not np.array_equal(np.asarray(outs[0][0])[c].view(np.uint8), np.asarray(outs[1][0])[c].view(np.uint8))
