# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whole-step post-kernel QSA cache writes vs the stock QSA layers (CPU, Pallas interpret).

Two QSA layers share one unified pool the way the served Qwen3.8 packs it (main KV pages
paired: slots 0 / 1; compressed keys packed: slots 0 / 1). Stock: `pooled_qsa_step` per
layer, in layer order (selection and attention stubbed: they only read the pool). Whole
step: `wholestep_qsa_writes` from the same projections. The pool must be byte-identical
after each of two consecutive verify steps, the second advancing every request by its
accepted count only, so the first step's rejected drafts are overwritten exactly as stock
overwrites them.
"""

from __future__ import annotations

import logging
import sys
import types

try:
    import vllm.logger  # noqa: F401
except ImportError:  # CPU venv without vllm: the kernels only need a logger and the KV layout.
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _logger = types.ModuleType("vllm.logger")

    def _init_logger(name):
        log = logging.getLogger(name)
        log.warning_once = log.info_once = log.debug_once = log.warning
        return log

    _logger.init_logger = _init_logger
    _vllm.envs, _vllm.logger = _envs, _logger
    sys.modules.update({"vllm": _vllm, "vllm.envs": _envs, "vllm.logger": _logger})

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import (
    QSALayerCache,
    wholestep_qsa_writes,
)

PAGES, PAGE_ROWS, WIDTH = 48, 512, 256
HEAD_DIM, INDEX_DIM, HEADS, INDEX_HEADS = 256, 128, 2, 2
RATIO, TOPK, ROTARY, T = 4, 64, 64, 4
LAYERS = [
    QSALayerCache(compressed_slot=0, main_slot=0, k_scale=0.375, v_scale=2.5),
    QSALayerCache(compressed_slot=1, main_slot=1, k_scale=0.5, v_scale=1.25),
]


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True})
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(
        pooled_qsa,
        "select_blocks",
        lambda query, *a, block_topk, **k: jnp.zeros((len(query), block_topk), jnp.int32),
    )
    monkeypatch.setattr(
        pooled_qsa,
        "qsa_attention",
        lambda q, gate, blocks, lengths, table, pool, *a, **k: (
            jnp.zeros(q.shape, jnp.bfloat16),
            pool,
        ),
    )


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def tables(rng, contexts, active):
    """Per-request pages shared by both layers (paired / packed), distinct per request."""
    n, columns = len(contexts), 4
    free = iter(rng.permutation(np.arange(1, PAGES)))
    main = np.zeros((n, columns), np.int32)
    comp = np.zeros((n, columns), np.int32)
    rings = [np.zeros((n, 8), np.int32) for _ in LAYERS]
    for r in range(n):
        if active[r]:
            for c in range((contexts[r] + 2 * T) // PAGE_ROWS + 1):
                main[r, c], comp[r, c] = next(free), next(free)
            for ring in rings:
                ring[r, 0] = next(free)
    return main, comp, rings


def step_inputs(rng, contexts, active):
    lens = np.where(active, T, 0)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    positions = np.zeros(len(contexts) * T, np.int32)
    for r in range(len(contexts)):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(lens[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    rows = len(positions)
    per_layer = [
        dict(
            qgate=normal(rows, 2 * HEADS * HEAD_DIM),
            key=normal(rows, HEAD_DIM),
            value=normal(rows, HEAD_DIM),
            index_qk=normal(rows, (INDEX_HEADS + 1) * INDEX_DIM),
            nq=normal(HEAD_DIM),
            nk=normal(HEAD_DIM),
            niq=normal(INDEX_DIM),
            nik=normal(INDEX_DIM),
        )
        for _ in LAYERS
    ]
    return jnp.asarray(positions), jnp.asarray(starts), per_layer


def stock(pool, per_layer, positions, trig, starts, context, main, comp, rings):
    step = jax.jit(
        pooled_qsa.pooled_qsa_step,
        static_argnames=("compress_ratio", "token_topk", "norm_eps", "rotary_dim", "k_scale",
                         "v_scale", "compressed_slot", "main_slot", "max_query_len"),
    )
    for p, lay, ring in zip(per_layer, LAYERS, rings):
        pool, _, _ = step(
            p["qgate"], p["key"], p["value"], p["index_qk"], p["nq"], p["nk"], p["niq"],
            p["nik"], positions, trig, pool, starts, context, jnp.asarray(ring.reshape(-1)),
            jnp.asarray(main.reshape(-1)), jnp.asarray(comp.reshape(-1)),
            compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
            k_scale=lay.k_scale, v_scale=lay.v_scale, compressed_slot=lay.compressed_slot,
            main_slot=lay.main_slot, max_query_len=T,
        )
    return pool


def whole(pool, per_layer, positions, trig, starts, context, main, comp, rings, num_requests=None):
    fn = jax.jit(
        lambda pool, rows, norms, tabs, positions, trig, starts, context: wholestep_qsa_writes(
            pool, rows, norms, tabs, LAYERS, positions, trig, starts, context,
            compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, max_query_len=T,
            num_requests=num_requests,
        )
    )
    rows = [(p["key"], p["value"], p["index_qk"][:, -INDEX_DIM:]) for p in per_layer]
    norms = [(p["nk"], p["nik"]) for p in per_layer]
    tabs = [
        (jnp.asarray(ring.reshape(-1)), jnp.asarray(main.reshape(-1)), jnp.asarray(comp.reshape(-1)))
        for ring in rings
    ]
    return fn(pool, rows, norms, tabs, positions, trig, starts, context)


# Contexts on every ring / compression-group phase and a page crossing; two idle rows.
CONTEXTS = [37, 0, 509, 0, 1022, 6, 3, 2]
ACTIVE = [True, True, True, False, True, True, False, True]
ACCEPTED = [1, 4, 2, 0, 3, 1, 0, 4]


@pytest.mark.parametrize("pad", [0, 4])
@pytest.mark.parametrize("fused", [False, True])
def test_wholestep_qsa_writes_match_stock_over_two_steps(interpret, monkeypatch, fused, pad):
    """`pad` idle requests past the kernel's request count (the runner's max_num_seqs rows):
    the whole-step writer crops them (`num_requests`) and must still match stock."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", fused)
    rng = np.random.default_rng(7 + fused)
    contexts, active = CONTEXTS + [0] * pad, ACTIVE + [False] * pad
    main, comp, rings = tables(rng, contexts, active)
    pool0 = jnp.asarray(rng.normal(0, 1, (PAGES, PAGE_ROWS, 1, 4, WIDTH)), jnp.float8_e4m3fn)
    trig = jnp.pad(jnp.asarray(rng.normal(0, 1, (4096, ROTARY)), jnp.bfloat16),
                   ((0, 0), (0, 128 - ROTARY)))
    ctx = np.array(contexts, np.int32)
    p_stock = p_whole = pool0
    for _ in range(2):
        positions, starts, per_layer = step_inputs(rng, ctx, active)
        context = jnp.asarray(ctx)
        p_stock = stock(p_stock, per_layer, positions, trig, starts, context, main, comp, rings)
        p_whole = whole(p_whole, per_layer, positions, trig, starts, context, main, comp, rings,
                        num_requests=len(CONTEXTS) if pad else None)
        assert not np.array_equal(raw(p_stock), raw(pool0)), "nothing written"
        np.testing.assert_array_equal(raw(p_whole), raw(p_stock))
        ctx = ctx + np.where(active, ACCEPTED + [0] * pad, 0).astype(np.int32)


def test_check_catches_a_swapped_main_slot(interpret, monkeypatch):
    """Positive control: the two layers' main head slots swapped must change the pool."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", False)
    rng = np.random.default_rng(3)
    main, comp, rings = tables(rng, CONTEXTS, ACTIVE)
    pool0 = jnp.asarray(rng.normal(0, 1, (PAGES, PAGE_ROWS, 1, 4, WIDTH)), jnp.float8_e4m3fn)
    trig = jnp.pad(jnp.asarray(rng.normal(0, 1, (4096, ROTARY)), jnp.bfloat16),
                   ((0, 0), (0, 128 - ROTARY)))
    positions, starts, per_layer = step_inputs(rng, np.array(CONTEXTS, np.int32), ACTIVE)
    context = jnp.asarray(CONTEXTS, jnp.int32)
    want = stock(pool0, per_layer, positions, trig, starts, context, main, comp, rings)
    got = whole(pool0, per_layer[::-1], positions, trig, starts, context, main, comp, rings)
    assert not np.array_equal(raw(got), raw(want))
