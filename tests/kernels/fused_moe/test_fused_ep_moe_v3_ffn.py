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
"""Fused EP MoE v3's expert FFN on CPU: its per-expert arithmetic
(``ffn``) against numpy, and the memoized kernel build (``ffn_kernel``).

``ffn.expert_ffn_fp8``, ``ffn.expert_ffn_bf16`` and ``ffn.result_rows`` are
the lines the kernel body runs for each tile: the up projection on fp8 token
rows, silu over the gate times the up half, the intermediate's fp8
requantization, the down projection and the result rows. They run here as
plain JAX against a numpy reference written out step by step, with one
scale per row and one per 128-value block, for the token rows and the
intermediate alike and each on its own.

The build reads ``layout.SHARD_INDEX_OPERAND``: in the torch bridge's mode
the shard's index is one more scalar-prefetch operand. The memoized build
must key on it, or a process that builds in one mode and calls in the other
gets a program with the wrong operand list. The built kernel runs end to end
on TPU in the layer's own test, which comes with the layer.
"""

import importlib

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import ffn, layout

pytestmark = pytest.mark.cpu_test

ffn_kernel = importlib.import_module("vllm_torchtpu.kernels.fused_moe.v3.ffn_kernel")

# The build test_fused_ep_moe_v3.py's small world takes: 16 experts over 8
# shards, 24 rows per shard at top-k 4, hidden 1024, intermediate 128.
BUILD = dict(
    experts_per_shard=2,
    tile_rows=32,
    hidden=1024,
    inter=128,
    num_shards=8,
    weight_format="fp8",
    routed_rows=928,
    result_rows="bf16",
    sorted_rows=True,
)


@pytest.fixture
def tpu7x(monkeypatch):
    """A TPU7x chip record (the build sizes its VMEM from it) and an empty
    build cache."""

    def record():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    monkeypatch.setattr(pltpu, "get_tpu_info", record)
    monkeypatch.setattr(tpu_info, "get_tpu_info", record)
    monkeypatch.setattr(ffn_kernel, "_BUILD_CACHE", {})


def test_a_build_is_reused_within_its_operand_mode_only(tpu7x, monkeypatch):
    monkeypatch.setattr(layout, "SHARD_INDEX_OPERAND", False)
    own_mode = ffn_kernel.ffn_kernel(**BUILD)
    assert ffn_kernel.ffn_kernel(**BUILD) is own_mode
    monkeypatch.setattr(layout, "SHARD_INDEX_OPERAND", True)
    operand_mode = ffn_kernel.ffn_kernel(**BUILD)
    assert operand_mode is not own_mode
    assert ffn_kernel.ffn_kernel(**BUILD) is operand_mode
    monkeypatch.setattr(layout, "SHARD_INDEX_OPERAND", False)
    assert ffn_kernel.ffn_kernel(**BUILD) is own_mode


# ---- the per-expert arithmetic, against numpy ----

ROWS, HIDDEN, INTER, BLOCK = 16, 256, 256, 128
FP8_NP = ml_dtypes.float8_e4m3fn


def _bf16(x):
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _quantize_np(x, block):
    """(fp8 values as f32, f32 scales [rows, n // block or 1]): one scale
    per row (block 0) or per block, the maximum over 448."""
    width = x.shape[1]
    step = block or width
    qs, scales = [], []
    for lo in range(0, width, step):
        part = x[:, lo : lo + step]
        scale = (np.abs(part).max(axis=1, keepdims=True) / np.float32(448.0)).astype(
            np.float32
        )
        ok = np.isfinite(scale) & (scale > 0)
        inverse = np.where(ok, np.float32(1.0) / np.where(ok, scale, 1), 0)
        q = np.where(ok, part * inverse.astype(np.float32), 0).astype(FP8_NP)
        qs.append(q.astype(np.float32))
        scales.append(np.where(ok, scale, 0).astype(np.float32))
    return np.concatenate(qs, axis=1), np.concatenate(scales, axis=1)


def _blocked_dot(lhs, rhs, scales, block):
    if not block:
        return lhs.astype(np.float64) @ rhs.astype(np.float64)
    out = 0
    for b in range(scales.shape[1]):
        cols = slice(b * block, (b + 1) * block)
        part = lhs[:, cols].astype(np.float64) @ rhs[cols].astype(np.float64)
        out = out + part * scales[:, b : b + 1]
    return out


def _reference(rows, row_scales, w1, w2, w1_scales, w2_scales, block, token_block):
    """The expert's result rows, dequantized, written out in numpy (float64
    matmuls): the token rows' scales and the weight scales applied after the
    up projection, silu(gate) * up rounded to bf16, the intermediate
    requantized to fp8, the down projection, its scales, the result rows
    rounded to bf16 and requantized to fp8 with one scale per row."""
    acc1 = _blocked_dot(rows, w1, row_scales, token_block)
    if not token_block:
        acc1 = acc1 * row_scales
    if w1_scales is not None:
        acc1 = acc1 * w1_scales
    acc1 = acc1.astype(np.float32)
    gate, up = acc1[:, :INTER], acc1[:, INTER:]
    mid = _bf16(gate / (1 + np.exp(-gate)) * up)
    mid_q, mid_scales = _quantize_np(mid, block)
    down = _blocked_dot(mid_q, w2, mid_scales, block)
    if not block:
        down = down * mid_scales
    if w2_scales is not None:
        down = down * w2_scales
    q, scales = _quantize_np(_bf16(down.astype(np.float32)), 0)
    return q * scales


def _world(weight_fp8, token_block, seed=0):
    rng = np.random.default_rng(seed)
    x = _bf16(rng.standard_normal((ROWS, HIDDEN)) * rng.uniform(0.2, 3.0, (ROWS, 1)))
    rows, row_scales = _quantize_np(x, token_block)
    w1 = rng.standard_normal((HIDDEN, 2 * INTER)).astype(np.float32) / 8
    w2 = rng.standard_normal((INTER, HIDDEN)).astype(np.float32) / 8
    if weight_fp8:
        s1 = (np.abs(w1).max(axis=0, keepdims=True) / 448).astype(np.float32)
        s2 = (np.abs(w2).max(axis=0, keepdims=True) / 448).astype(np.float32)
        w1 = (w1 / s1).astype(FP8_NP).astype(np.float32)
        w2 = (w2 / s2).astype(FP8_NP).astype(np.float32)
        return rows, row_scales, w1, w2, s1, s2
    return rows, row_scales, _bf16(w1), _bf16(w2), None, None


# (intermediate block, token-row block): one scale per row for both; one
# per block for both (token_block None follows block); and each blocked on
# its own.
BLOCKS = {
    "rows_rows": (0, None),
    "blocks_blocks": (BLOCK, None),
    "rows_intermediate_blocks": (BLOCK, 0),
    "blocked_rows_whole_intermediate": (0, BLOCK),
}


def _expert(weight_fp8, block, token_block, *, swap_halves=False):
    """(the expert's result rows dequantized, the numpy reference)."""
    tokens_blocked = block if token_block is None else token_block
    world = _world(weight_fp8, tokens_blocked)
    rows, row_scales, w1, w2, s1, s2 = world
    if swap_halves:  # a control: the gate and up halves the wrong way round
        w1 = np.concatenate([w1[:, INTER:], w1[:, :INTER]], axis=1)
        if s1 is not None:
            s1 = np.concatenate([s1[:, INTER:], s1[:, :INTER]], axis=1)
    weight_dtype = jnp.float8_e4m3fn if weight_fp8 else jnp.bfloat16
    common = dict(
        activation="silu",
        w1_bias=None,
        no_gate=False,
        block=block,
        token_block=token_block,
        columns_per_pass=128,  # two activation passes over the intermediate
    )
    args = (
        jnp.asarray(rows).astype(jnp.float8_e4m3fn),
        jnp.asarray(row_scales),
        jnp.asarray(w1).astype(weight_dtype),
        jnp.asarray(w2).astype(weight_dtype),
    )
    if weight_fp8:
        acc2, mid_scales = ffn.expert_ffn_fp8(*args, jnp.asarray(s1), **common)
    else:
        acc2, mid_scales = ffn.expert_ffn_bf16(*args, **common)
    w2_scales = None if s2 is None else jnp.asarray(s2)
    q, scales = ffn.result_rows(acc2, mid_scales, w2_scales=w2_scales, w2_bias=None)
    got = np.asarray(q.astype(jnp.float32)) * np.asarray(scales)
    want = _reference(*world, block, tokens_blocked)
    return got, want


def _off_fraction(got, want):
    """The fraction of values more than one fp8 step (an eighth, relative)
    from the reference, with a floor for values near zero."""
    floor = 1e-3 * np.abs(want).max()
    return float((np.abs(got - want) > 0.13 * np.abs(want) + floor).mean())


@pytest.mark.parametrize("weights", ["fp8", "bf16"])
@pytest.mark.parametrize("blocks", list(BLOCKS))
def test_expert_ffn_matches_numpy(weights, blocks):
    """The matmuls add in a different order, so a product can round to the
    neighbouring bf16 or fp8 value; a wrong scale, block or half moves most
    values (the control below)."""
    block, token_block = BLOCKS[blocks]
    got, want = _expert(weights == "fp8", block, token_block)
    assert np.isfinite(got).all()
    assert _off_fraction(got, want) < 0.01


@pytest.mark.parametrize("blocks", list(BLOCKS))
def test_the_comparison_sees_the_halves_swapped(blocks):
    block, token_block = BLOCKS[blocks]
    got, want = _expert(True, block, token_block, swap_halves=True)
    assert _off_fraction(got, want) > 0.5


def test_bf16_result_rows_skip_the_last_rounding():
    """bf16 result rows are the down projection rounded once, with no
    scale."""
    rows, row_scales, w1, w2, s1, s2 = _world(True, 0)
    acc2, mid_scales = ffn.expert_ffn_fp8(
        jnp.asarray(rows).astype(jnp.float8_e4m3fn),
        jnp.asarray(row_scales),
        jnp.asarray(w1).astype(jnp.float8_e4m3fn),
        jnp.asarray(w2).astype(jnp.float8_e4m3fn),
        jnp.asarray(s1),
        activation="silu",
        w1_bias=None,
        no_gate=False,
    )
    out, scales = ffn.result_rows(
        acc2, mid_scales, w2_scales=jnp.asarray(s2), w2_bias=None, dtype=jnp.bfloat16
    )
    assert scales is None and out.dtype == jnp.bfloat16
    want = np.asarray(acc2) * np.asarray(mid_scales) * s2
    np.testing.assert_array_equal(np.asarray(out.astype(jnp.float32)), _bf16(want))
