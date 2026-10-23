# SPDX-License-Identifier: Apache-2.0
"""QSA decode with ``VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE``.

Both layers of a paired main KV page (and unpaired FP8 layers, whose rows hold
the same head slots) present every head slot of the row, with zero queries for
the other slot, so every QSA layer launches the same decode kernel
configuration and its schedule depends only on the step's lengths. On CPU the
kernels are the decode-only references of ``test_qwen4_exp_fused_sparse_decode``;
the RPA route is also lowered for TPU7x to check that the paired slots' decode
schedules are identical (XLA can then compute one per step) and that the empty
mixed pass is gone.
"""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_fused_sparse_decode import (  # noqa: F401
    FLAG,
    Q_HEADS,
    RATIO,
    TPU7X,
    WIDTH,
    _lower,
    _verify_step,
    kernels,
)

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

SHARED = "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE"


def _attend(args, starts, *, fused, shared, **options):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(FLAG, "1" if fused else "0")
        patch.setenv(SHARED, "1" if shared else "0")
        return jax.jit(
            functools.partial(
                qsa_rpa.qsa_attention,
                starts=starts,
                kv_heads=1,
                ratio=RATIO,
                k_scale=0.375,
                v_scale=2.5,
                **options,
            )
        )(*args)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_shared_schedule_matches_per_slot_launch(kernels, fused, offset):
    rng = np.random.default_rng(31 + offset)
    args, starts = _verify_step(rng, [300, 0, 41, 129, 7], 4, 24, 16, 128, 3)
    options = dict(max_query_len=4)
    if offset:
        options["kv_head_offset"] = offset
    expected, _ = _attend(args, starts, fused=fused, shared=False, **options)
    actual, cache = _attend(args, starts, fused=fused, shared=True, **options)
    name, query, other = kernels[-1]
    assert name == ("fused" if fused else "rpa")
    if fused:
        # Fused query layout: (kv_heads, batch, q_per_kv, 1, d).
        assert query[0] == 2 and query[2] == Q_HEADS
    else:
        # Both slots present two KV heads and six queries.
        assert query[1:] == (2 * Q_HEADS, WIDTH) and other[1:] == (2, WIDTH)
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == (24, Q_HEADS, WIDTH)
    assert np.abs(expected).max() > 0
    np.testing.assert_array_equal(actual, expected)
    assert cache.shape == args[5].shape


def test_shared_schedule_mixed_step_prefix(kernels):
    rng = np.random.default_rng(17)
    args, starts = _verify_step(rng, [100, 600, 31], 2, 12, 16, 128, 6)
    # Make the last request a prefill chunk past the decode capacity.
    starts = jnp.asarray([0, 2, 4, 12], jnp.int32)
    for offset in (0, 1):
        options = dict(max_query_len=2, **({"kv_head_offset": 1} if offset else {}))
        expected, _ = _attend(args, starts, fused=False, shared=False, **options)
        actual, _ = _attend(args, starts, fused=False, shared=True, **options)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _schedules(text):
    """Backend configs of the RPA schedules, and the RPA kernel names."""
    calls = [line for line in text.splitlines() if "@tpu_custom_call" in line]
    names = [re.search(r'kernel_name = "([^"]+)"', c).group(1) for c in calls]
    return (
        [c[c.index("backend_config") :] for c, n in zip(calls, names)
         if n == "rpa_metadata_schedule"],
        [n for n in names if n.startswith("RPA")],
    )


def test_paired_slots_lower_identical_decode_schedule(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    monkeypatch.setenv(FLAG, "0")
    rows, columns = 32, 64
    shape = jax.ShapeDtypeStruct
    specs = (
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, Q_HEADS, WIDTH), jnp.bfloat16),
        shape((rows, 512), jnp.int32),
        shape((rows,), jnp.int32),
        shape((rows, columns), jnp.int32),
        shape((4096, 512, 1, 4, WIDTH), jnp.float8_e4m3fn),
        shape((rows // 4 + 1,), jnp.int32),
    )

    def lower(offset):
        options = dict(kv_heads=1, ratio=RATIO, k_scale=0.375, v_scale=2.5)
        if offset:
            options["kv_head_offset"] = offset
        return _schedules(
            _lower(
                functools.partial(qsa_rpa.qsa_attention, max_query_len=4, **options),
                *specs,
            )
        )

    monkeypatch.setenv(SHARED, "0")
    (first, first_kernels), (second, _) = lower(0), lower(1)
    assert len(first) == 2 and any(k.startswith("RPAm") for k in first_kernels)
    assert first[0] != second[0]
    monkeypatch.setenv(SHARED, "1")
    (first, first_kernels), (second, second_kernels) = lower(0), lower(1)
    # One decode schedule and kernel, identical for both slots; no mixed pass.
    assert len(first) == 1 and first == second
    assert first_kernels == second_kernels
    assert all(k.startswith("RPAd") for k in first_kernels)
