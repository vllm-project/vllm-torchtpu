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
"""Two of fused EP MoE v3's Pallas programs, run directly in TPU interpret
mode on CPU against references written out here.

``select_kernel.select_top_k`` (the router's top-k and its renormalized
weights) and ``combine.weighted_sum`` (each token's k-ordered weighted sum of
its expert rows) hold the layer's selection and combine arithmetic and run
as single-device programs. The layer as a whole does not run in the
interpreter: its token exchange returns DMA semaphores from one Pallas
program to the next, and the TPU interpreter cannot return a semaphore from
a program. The layer end to end is ``test_fused_ep_moe_v3.py``, on an
eight-core TPU host.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import combine, select_kernel

pytestmark = pytest.mark.cpu_test

LANES = 128


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


@pytest.fixture
def interpret(monkeypatch):
    """TPU interpret mode with a TPU7x chip record (the programs size their
    VMEM from it)."""
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", _tpu7x)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        yield


def _logits(rows, experts, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, experts)).astype(np.float32)


# ---- select_top_k ----


@pytest.mark.parametrize("renormalize", [False, True], ids=["raw", "renormalized"])
@pytest.mark.parametrize("experts,top_k", [(64, 4), (512, 10)])
def test_select_top_k_matches_lax_top_k(interpret, renormalize, experts, top_k):
    logits = _logits(32, experts)
    weights, indices = select_kernel.select_top_k(
        jnp.asarray(logits), top_k=top_k, block_rows=8, renormalize=renormalize
    )
    values, want_indices = jax.lax.top_k(jnp.asarray(logits), top_k)
    np.testing.assert_array_equal(np.asarray(indices), np.asarray(want_indices))
    if renormalize:
        want = jax.nn.softmax(values, axis=-1)
        np.testing.assert_allclose(np.asarray(weights), want, rtol=1e-6, atol=0)
    else:
        np.testing.assert_array_equal(np.asarray(weights), np.asarray(values))


def test_select_top_k_breaks_ties_toward_the_lower_expert(interpret):
    logits = _logits(8, 64)
    logits[:, 5] = logits[:, 9] = logits[:, 40] = 10.0  # a three-way tie
    _, indices = select_kernel.select_top_k(
        jnp.asarray(logits), top_k=4, block_rows=8, renormalize=True
    )
    np.testing.assert_array_equal(np.asarray(indices)[:, :3], [[5, 9, 40]] * 8)


def test_select_top_k_routes_rows_past_the_valid_count_nowhere(interpret):
    """Rows past valid_tokens, and a row with a NaN logit, get weight zero;
    the other rows are unchanged."""
    logits = _logits(32, 64)
    logits[3, 7] = np.nan
    valid = 21
    weights, indices = select_kernel.select_top_k(
        jnp.asarray(logits),
        top_k=4,
        block_rows=8,
        renormalize=True,
        drop_nan_rows=True,
        valid_tokens=jnp.asarray([valid], jnp.int32),
    )
    weights = np.asarray(weights)
    assert not weights[valid:].any()
    assert not weights[3].any()
    kept = [r for r in range(valid) if r != 3]
    values, want_indices = jax.lax.top_k(jnp.asarray(logits[kept]), 4)
    np.testing.assert_array_equal(np.asarray(indices)[kept], np.asarray(want_indices))
    np.testing.assert_allclose(
        weights[kept], jax.nn.softmax(values, axis=-1), rtol=1e-6, atol=0
    )


# ---- weighted_sum ----


def _pairs(tokens, top_k, hidden, seed=1):
    """Rows in pair order (pair t * top_k + k is token t's slot k), the
    weights with a zero on every token's last slot, and NaN in the rows those
    zero weights point at (a route of weight zero reads a row nobody wrote)."""
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((tokens * top_k, hidden)).astype(np.float32)
    weights = rng.uniform(0.05, 1.0, (tokens, top_k)).astype(np.float32)
    weights[:, -1] = 0.0
    rows.reshape(tokens, top_k, hidden)[:, -1, :] = np.nan
    return rows, weights


def _k_ordered_sum(rows, coef, tokens, top_k, hidden):
    terms = jnp.asarray(rows, jnp.float32).reshape(tokens, top_k, hidden)
    coef = jnp.asarray(coef, jnp.float32)
    acc = None
    for k in range(top_k):
        c = coef[:, k : k + 1]
        term = jnp.where(c != 0, terms[:, k, :] * c, 0.0)
        acc = term if k == 0 else acc + term
    return acc


@pytest.mark.parametrize("fp8_rows", [False, True], ids=["bf16_rows", "fp8_rows"])
def test_weighted_sum_matches_the_k_ordered_sum(interpret, fp8_rows):
    tokens, top_k, hidden = 128, 4, 1024
    rows, weights = _pairs(tokens, top_k, hidden)
    if fp8_rows:
        finite = np.nan_to_num(rows, nan=0.0)
        scales = (np.abs(finite).max(axis=1) / 448.0).astype(np.float32)
        q = jnp.asarray(rows / scales[:, None]).astype(jnp.float8_e4m3fn)
        coef = weights * scales.reshape(tokens, top_k)
        scales_arg = jnp.asarray(scales)
    else:
        q = jnp.asarray(rows).astype(jnp.bfloat16)
        coef = weights
        scales_arg = None
    got = combine.weighted_sum(
        q.reshape(tokens * top_k, hidden // LANES, LANES),
        scales_arg,
        jnp.asarray(weights),
        tokens_per_shard=tokens,
        hidden=hidden,
        out_dtype=jnp.bfloat16,
    )
    want = _k_ordered_sum(q, coef, tokens, top_k, hidden)
    got = np.asarray(got.astype(jnp.float32))
    assert np.isfinite(got).all()
    # A fused multiply-add in one of the two programs can move an f32 sum by
    # one ulp and its bf16 rounding by one step on a few values; a slot
    # mixed up or dropped moves most of them (the control below).
    assert _off_fraction(got, want) < 1e-3


def _off_fraction(got, want):
    """The fraction of values more than one bf16 step from the reference."""
    want = np.asarray(jnp.asarray(want).astype(jnp.bfloat16), np.float32)
    return float((np.abs(got - want) > 2.0**-7 * np.abs(want) + 1e-5).mean())


def test_weighted_sum_rows_are_not_interchangeable(interpret):
    """Control: the same rows with two slots' weights swapped are off the
    reference almost everywhere, so the comparison above sees a slot mix-up."""
    tokens, top_k, hidden = 128, 4, 1024
    rows, weights = _pairs(tokens, top_k, hidden)
    q = jnp.asarray(rows).astype(jnp.bfloat16)
    swapped = weights.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    sums = [
        np.asarray(
            combine.weighted_sum(
                q.reshape(tokens * top_k, hidden // LANES, LANES),
                None,
                jnp.asarray(w),
                tokens_per_shard=tokens,
                hidden=hidden,
                out_dtype=jnp.bfloat16,
            ).astype(jnp.float32)
        )
        for w in (weights, swapped)
    ]
    assert _off_fraction(sums[0], sums[1]) > 0.5
