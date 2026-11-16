# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_FOLD and _TILE_ROWS on TPU (skipped elsewhere): bit-equal.

Without the fold a layer's mlp combine is an XLA fusion, bf16(f32(residual_b) +
f32(block) * injection[:, b]) per branch (``GatedResidual.combine`` as the decode
program lowers it), and the next fused mix reads its result. With the fold the
same combine is the prologue of the next ``hc_decode_mix`` call. On the same
inputs both must give the same combined streams, mixed output and injection gate
bit for bit, at every decode bucket and with the final mixer's no-injection
variant. Larger row tiles (TPU_QWEN4_HC_DECODE_TILE_ROWS) must give the same
outputs as the 128-row tiling.

Run it on a TPU host, e.g. through the bench recipe:
``BENCH=script "BENCH_CMD=python -m pytest -q -s
tests/kernels/test_qwen4_exp_hc_fold_tpu.py"``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_kernel_tile, hc_decode_mix


@pytest.fixture(autouse=True)
def _tpu_only():
    # At run time: a backend initialized during collection would claim the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")


# Qwen3.8-Flash per-rank shapes (HC is TP-replicated).
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN
# Decode buckets of the served and bench configs below the 512-row cap.
BUCKETS = [16, 64, 96, 112, 128, 144, 160, 192, 208, 224, 240, 256, 320, 384, 448]


@jax.jit
def _xla_combine(residual, block, injection):
    residual = residual.astype(jnp.float32)
    block = block.astype(jnp.float32)
    parts = [
        residual[:, b * HIDDEN : (b + 1) * HIDDEN] + block * injection[:, b : b + 1]
        for b in range(BRANCHES)
    ]
    return jnp.concatenate(parts, axis=-1).astype(jnp.bfloat16)


def _inputs(seed, tokens, inject):
    keys = jax.random.split(jax.random.key(seed), 8)
    bf16 = jnp.bfloat16
    residual = (4.0 * jax.random.normal(keys[0], (tokens, WIDTH))).astype(bf16)
    block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
    injection = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
    weights = [
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
    ]
    if inject:
        inject_weight = 0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))
        weights.append(inject_weight.astype(bf16))
    return residual, block, injection.astype(jnp.float32), weights


def _compare(name, actual, expected):
    actual = np.asarray(jnp.asarray(actual, jnp.float32))
    expected = np.asarray(jnp.asarray(expected, jnp.float32))
    equal = float(np.mean(actual == expected))
    diff = float(np.abs(actual - expected).max())
    return f"{name} bit-equal {equal:.6f} max|d| {diff:.3g}", equal == 1.0


@pytest.mark.parametrize("max_tile", [0, 256])
@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("tokens", BUCKETS)
def test_tpu_folded_combine_is_bit_equal(tokens, inject, max_tile):
    residual, block, injection, weights = _inputs(tokens, tokens, inject)
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, max_tile=max_tile)

    combined_ref = _xla_combine(residual, block, injection)
    mixed_ref, inj_ref = hc_decode_mix(combined_ref, *weights, **static)
    mixed, inj, combined = hc_decode_mix(
        residual, *weights, block=block, prev_injection=injection, **static
    )

    checks = [
        _compare("combined", combined, combined_ref),
        _compare("mixed", mixed, mixed_ref),
    ]
    if inject:
        checks.append(_compare("injection", inj, inj_ref))
    else:
        assert inj is None and inj_ref is None
    print(
        f"fold tokens {tokens} tile {decode_kernel_tile(tokens, max_tile)} "
        f"inject {inject}: " + "; ".join(c[0] for c in checks)
    )
    assert all(c[1] for c in checks), checks


@pytest.mark.parametrize("max_tile", [256, 512])
@pytest.mark.parametrize("prologue", [False, True])
@pytest.mark.parametrize("tokens", [t for t in BUCKETS if t >= 128])
def test_tpu_tile_rows_match_128_tiling(tokens, prologue, max_tile):
    residual, block, injection, weights = _inputs(1000 + tokens, tokens, True)
    extra = dict(block=block, prev_injection=injection) if prologue else {}
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS)
    ref = hc_decode_mix(residual, *weights, **extra, **static)
    out = hc_decode_mix(residual, *weights, **extra, max_tile=max_tile, **static)
    names = ["mixed", "injection", "combined"][: len(ref)]
    checks = [_compare(n, o, r) for n, o, r in zip(names, out, ref)]
    print(
        f"tiles tokens {tokens} tile {decode_kernel_tile(tokens, 0)} -> "
        f"{decode_kernel_tile(tokens, max_tile)} prologue {prologue}: "
        + "; ".join(c[0] for c in checks)
    )
    assert all(c[1] for c in checks), checks
