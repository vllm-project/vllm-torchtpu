# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_STACKED_MAMBA_OPERANDS composes with VLLM_TPU_MK_IO_LEAN=all.

The stacked operands are unstacked in the call (`unstack_meta`) into the per-layer GDN
metadata every consumer reads: `gdn_blocks` (read / write block ids and has-state,
which the conv lever's checkpoint scatter takes as `gdn_hi` / `gdn_rb`). Same block ids
bit for bit, with every IO_LEAN lever and both input-prep forms, and the 33-way stack of
per-layer operands leaves the optimized program.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import gdn_blocks
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import unstack_meta

N_GDN, NS, W, SLOTS = 33, 16, 4, 112509


def _case(seed, live):
    gen = np.random.default_rng(seed)
    G = N_GDN + 4  # the walk's stacks hold unused rows (PLE etc.) as well
    st_stack = gen.integers(0, SLOTS, (G, NS)).astype(np.int32)
    ck_stack = gen.integers(0, 4 * SLOTS, (G, NS, W)).astype(np.int32)
    rows = gen.permutation(G)[:N_GDN].astype(np.int32)
    ro = gen.integers(0, W, SLOTS).astype(np.int32)
    lens = np.where(np.arange(NS) < live, gen.integers(1, 5, NS), 0)
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    seq = (lens + gen.integers(0, 3, NS) * (np.arange(NS) % 2)).astype(np.int32)
    stacked = [st_stack, ck_stack, rows, ro, qsl, seq]
    per_layer = [
        x for o in range(N_GDN) for x in (st_stack[rows[o]], ro, ck_stack[rows[o]], qsl, seq)
    ]
    return [jnp.asarray(x) for x in stacked], [jnp.asarray(x) for x in per_layer]


@pytest.mark.parametrize("lean", ["", "all"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("live", [16, 9, 1])
def test_stacked_meta_gives_the_per_layer_block_ids(monkeypatch, lean, batched, live):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", lean)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", batched)
    stacked, per_layer = _case(live, live)
    meta = unstack_meta(stacked, N_GDN)
    assert len(meta) == len(per_layer)
    for write in (True, False):
        a = jax.jit(lambda *m: gdn_blocks(list(m), NS, write))(*per_layer)
        b = jax.jit(lambda *m: gdn_blocks(unstack_meta(list(m), N_GDN), NS, write))(*stacked)
        for x, y in zip(a, b, strict=True):
            assert x.dtype == y.dtype and np.array_equal(np.asarray(x), np.asarray(y))


def test_stacked_meta_check_catches_swapped_rows():
    """Negative control: two layers' stack rows exchanged change the block ids."""
    stacked, per_layer = _case(3, 16)
    rows = np.asarray(stacked[2]).copy()
    rows[[0, 1]] = rows[[1, 0]]
    bad = stacked[:2] + [jnp.asarray(rows)] + stacked[3:]
    a = gdn_blocks(per_layer, NS, True)
    b = gdn_blocks(unstack_meta(bad, N_GDN), NS, True)
    assert not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b, strict=True))


def _big_concats(hlo: str) -> int:
    """Concatenates / dynamic-update-slice chains of >= N_GDN per-layer pieces."""
    n = 0
    for line in hlo.splitlines():
        if " concatenate(" in line and line.count("%") - 1 >= N_GDN:
            n += 1
    n += len(re.findall(r"dynamic-update-slice\(", hlo)) // N_GDN
    return n


def test_stacked_meta_removes_the_per_layer_stack(monkeypatch):
    """Informational on CPU: with the per-layer operands, `gdn_blocks` stacks 33
    parameters (the served program's DUS chain, xprof j-3d2506a9); with the stacked
    operands that stack is gone (two gathers of the walk's stacks instead)."""
    monkeypatch.setattr(envs, "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS", True)
    stacked, per_layer = _case(0, 16)
    per = jax.jit(lambda *m: gdn_blocks(list(m), NS, True)).lower(*per_layer).compile().as_text()
    stk = (
        jax.jit(lambda *m: gdn_blocks(unstack_meta(list(m), N_GDN), NS, True))
        .lower(*stacked).compile().as_text()
    )
    print("per-layer big stacks", _big_concats(per), "stacked", _big_concats(stk))
    assert _big_concats(per) >= 1
    assert _big_concats(stk) == 0
