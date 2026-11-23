# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-HC on TPU (skipped elsewhere): TPU_QWEN4_HC_PREFILL_FOLD / _DOWN_ASM bit-equal.

Without the fold, a layer's mlp combine is the gated_residual kernel and the next
layer's fused attn mix reads its bf16 result from HBM. With the fold, the same
combine is the reread prologue of that mix. At the served prefill steps (6,400 =
25 tiles; 3,200 = 12 tiles + a 128-row partial) both must give the same combined
streams, mixed output and injection gate bit for bit. The in-register prologue
(reread off, today's attn -> mlp combine-mix) is printed as a descriptive row
(P-DENSE mb5c found a few rows differing). down_asm must equal the plain kernel.

Run on a TPU host through the bench recipe:
``BENCH=script "BENCH_CMD=python -m pytest -q -s tests/kernels/test_qwen4_exp_hc_prefill_fold_tpu.py"``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix


@pytest.fixture(autouse=True)
def _tpu_only():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")


BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 2560, 320, 1e-6, 256
WIDTH = BRANCHES * HIDDEN
fused = functools.partial(
    hc_prefill_mix, branches=BRANCHES, hidden=HIDDEN, eps=EPS, token_tile=TILE, partial=True,
    norm_impl="bf16ep64",
)


def _inputs(seed, rows):
    k = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    prev = (4.0 * jax.random.normal(k[0], (rows, WIDTH))).astype(bf16)
    block = jax.random.normal(k[1], (rows, HIDDEN)).astype(bf16)
    gates = (2.0 * jax.nn.sigmoid(jax.random.normal(k[2], (rows, BRANCHES)))).astype(jnp.float32)
    weights = (
        (0.3 * jax.random.normal(k[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(k[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(k[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(k[6], (BRANCHES, WIDTH))).astype(bf16),
    )
    return prev, block, gates, weights


def _diff(name, got, want):
    g = np.asarray(jnp.asarray(got, jnp.float32))
    w = np.asarray(jnp.asarray(want, jnp.float32))
    rows = int((g != w).reshape(g.shape[0], -1).any(axis=1).sum())
    print(f"PHC TPU {name}: rows differing {rows} of {g.shape[0]}, max|d| {float(np.abs(g - w).max()):.3g}")
    return rows


@pytest.mark.parametrize("rows", [6400, 3200])
def test_tpu_reread_fold_is_bit_equal(rows):
    prev, block, gates, w = _inputs(rows, rows)
    h_ref = jax.jit(gated_residual_fn(BRANCHES, HIDDEN, TILE, partial=True))(prev, block, gates)
    mixed_ref, inj_ref = jax.jit(fused)(h_ref, *w)
    pro = jax.jit(lambda p, b, g, *wt, reread: fused(p, *wt, block=b, prev_injection=g, reread=reread),
                  static_argnames="reread")
    mixed, inj, h = pro(prev, block, gates, *w, reread=True)
    bad = _diff(f"T{rows} rr h", h, h_ref) + _diff(f"T{rows} rr mixed", mixed, mixed_ref)
    bad += _diff(f"T{rows} rr injection", inj, inj_ref)
    m0, i0, _ = pro(prev, block, gates, *w, reread=False)
    _diff(f"T{rows} in-register (descriptive) mixed", m0, mixed_ref)
    _diff(f"T{rows} in-register (descriptive) injection", i0, inj_ref)
    assert bad == 0


@pytest.mark.parametrize("prologue", [False, True])
@pytest.mark.parametrize("rows", [6400, 3200])
def test_tpu_down_asm_is_bit_equal(rows, prologue):
    prev, block, gates, w = _inputs(rows + 1, rows)
    def run(p, b, g, *wt, asm):
        kw = dict(block=b, prev_injection=g) if prologue else {}
        return fused(p, *wt, down_asm=asm, **kw)

    run = jax.jit(run, static_argnames="asm")
    want = run(prev, block, gates, *w, asm=False)
    got = run(prev, block, gates, *w, asm=True)
    bad = sum(_diff(f"T{rows} p{int(prologue)} da {n}", g, x)
              for n, g, x in zip(("mixed", "injection", "h"), got, want))
    assert bad == 0
