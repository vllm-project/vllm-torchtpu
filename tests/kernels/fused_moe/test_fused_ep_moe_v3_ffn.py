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
"""Fused EP MoE v3's expert FFN: its per-expert arithmetic (``ffn``) against
numpy and the memoized kernel build (``ffn_kernel``) on CPU, and the built
kernel against numpy on one TPU7x chip.

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
gets a program with the wrong operand list.

On TPU the built kernel runs over the two cores of one chip, one shard each,
on routing tables from ``routing_tables``' reference passes: every routed
row's arrival on its token's shard, through the remote push or the local
hop, is checked against the numpy reference of its expert's FFN, at the
per-shard shape of the Qwen3.5-397B decode layer and at a small shape that
fetches rows through the token windows.
"""

import functools
import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from vllm_torchtpu.kernels.fused_moe.v3 import device, ffn, layout
from vllm_torchtpu.kernels.fused_moe.v3 import routing_tables as tables

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


@pytest.mark.cpu_test
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


def _reference(
    rows, row_scales, w1, w2, w1_scales, w2_scales, block, token_block, bf16_rows=False
):
    """The expert's result rows, dequantized, written out in numpy (float64
    matmuls): the token rows' scales and the weight scales applied after the
    up projection, silu(gate) * up rounded to bf16, the intermediate
    requantized to fp8, the down projection, its scales, the result rows
    rounded to bf16 and (unless `bf16_rows`) requantized to fp8 with one
    scale per row."""
    inter = w2.shape[0]
    acc1 = _blocked_dot(rows, w1, row_scales, token_block)
    if not token_block:
        acc1 = acc1 * row_scales
    if w1_scales is not None:
        acc1 = acc1 * w1_scales
    acc1 = acc1.astype(np.float32)
    gate, up = acc1[:, :inter], acc1[:, inter:]
    mid = _bf16(gate / (1 + np.exp(-gate)) * up)
    mid_q, mid_scales = _quantize_np(mid, block)
    down = _blocked_dot(mid_q, w2, mid_scales, block)
    if not block:
        down = down * mid_scales
    if w2_scales is not None:
        down = down * w2_scales
    down = _bf16(down.astype(np.float32))
    if bf16_rows:
        return down
    q, scales = _quantize_np(down, 0)
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


@pytest.mark.cpu_test
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


@pytest.mark.cpu_test
@pytest.mark.parametrize("blocks", list(BLOCKS))
def test_the_comparison_sees_the_halves_swapped(blocks):
    block, token_block = BLOCKS[blocks]
    got, want = _expert(True, block, token_block, swap_halves=True)
    assert _off_fraction(got, want) > 0.5


@pytest.mark.cpu_test
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


# ---- the built kernel on TPU, against numpy ----


class KernelCase(NamedTuple):
    """One built kernel and the tokens it runs."""

    num_shards: int
    experts_per_shard: int
    hidden: int
    inter: int
    tile_rows: int
    top_k: int
    tokens_per_shard: int
    result_rows: str
    elementwise_dtype: str
    sorted_rows: bool


# Two shards, the two cores of one TPU7x chip, so the runs for the other shard
# go out through the remote push and the own shard's through the local hop.
# The first two are the per-shard shape of the Qwen3.5-397B decode layer (64
# local experts of 4096 x 1024, tile 128, fp8 weights, bf16 element-wise
# products, rows sorted into expert order) with fp8 and with bf16 result
# rows; the third fetches each row through the token windows.
KERNEL_CASES = {
    "measured_fp8_rows": KernelCase(
        2, 64, 4096, 1024, 128, 10, 128, "fp8", "bfloat16", True
    ),
    "measured_bf16_rows": KernelCase(
        2, 64, 4096, 1024, 128, 10, 128, "bf16", "bfloat16", True
    ),
    "small_token_windows": KernelCase(
        2, 4, 512, 256, 32, 4, 64, "fp8", "float32", False
    ),
}


def _kernel_geometry(case):
    """(routed rows, arrival rows) per shard: the layer's bounds for the
    case's tokens."""
    num_experts = case.num_shards * case.experts_per_shard
    routed_rows = tables.routed_rows_bound(
        case.num_shards * case.tokens_per_shard, case.top_k, num_experts, case.tile_rows
    )
    arrival_rows = layout.align_up(
        case.tokens_per_shard * case.top_k + (layout.ROW_BLOCK - 1) * num_experts,
        layout.ROW_BLOCK,
    )
    return routed_rows, arrival_rows


def _kernel_routing(case, seed=3):
    """(top-k indices [tokens, top_k], the reference routing tables). Every
    token routes to the first expert of each shard, so those experts run
    several tiles, to no shard's last expert, so the kernel skips it, and to
    top_k - num_shards others at random."""
    rng = np.random.default_rng(seed)
    shards, local = case.num_shards, case.experts_per_shard
    hot = [s * local for s in range(shards)]
    cold = [s * local + local - 1 for s in range(shards)]
    pool = np.setdiff1d(np.arange(shards * local), hot + cold)
    indices = np.stack(
        [
            rng.permutation(
                np.concatenate(
                    [hot, rng.choice(pool, case.top_k - shards, replace=False)]
                )
            )
            for _ in range(shards * case.tokens_per_shard)
        ]
    ).astype(np.int32)
    routed_rows, _ = _kernel_geometry(case)
    routing = tables.build_routing_tables(
        jnp.asarray(indices),
        num_experts=shards * local,
        num_shards=shards,
        tokens_per_shard=case.tokens_per_shard,
        block=tables.routing_block(case.tokens_per_shard, case.top_k),
        tile_rows=case.tile_rows,
        routed_rows_per_shard=routed_rows,
    )
    return indices, routing


@functools.cache
def _kernel_inputs(*shape):
    """For one shape (KernelCase's first seven fields): the routing, the fp8
    token rows and their scales, the experts' fp8 weights and per-channel
    scales, and every routed pair's reference row (bf16, before the result
    rows' rounding). Each expert's weights are one fp8 matrix with its rows
    rolled by an amount of its own (97 and 89 are odd, so no two experts
    share a roll) and scales of its own, so every expert computes something
    different from one drawn matrix."""
    case = KernelCase(*shape, "fp8", "float32", True)
    shards, local, hidden, inter = shape[:4]
    num_experts = shards * local
    indices, routing = _kernel_routing(case)
    rng = np.random.default_rng(4)
    tokens = indices.shape[0]
    x = _bf16(
        rng.standard_normal((tokens, hidden)) * rng.uniform(0.2, 3.0, (tokens, 1))
    )
    rows, row_scales = _quantize_np(x, 0)
    w1 = rng.standard_normal((hidden, 2 * inter)).astype(np.float32) / 8
    w2 = rng.standard_normal((inter, hidden)).astype(np.float32) / 8
    s1 = (np.abs(w1).max(axis=0, keepdims=True) / 448).astype(np.float32)
    s2 = (np.abs(w2).max(axis=0, keepdims=True) / 448).astype(np.float32)
    w1 = (w1 / s1).astype(FP8_NP)
    w2 = (w2 / s2).astype(FP8_NP)
    w1_all = np.empty((num_experts, hidden, 2 * inter), FP8_NP)
    w2_all = np.empty((num_experts, inter, hidden), FP8_NP)
    for e in range(num_experts):
        w1_all[e] = np.roll(w1, 97 * e, axis=0)
        w2_all[e] = np.roll(w2, 89 * e, axis=0)
    s1_all = (s1 * rng.uniform(0.5, 2.0, (num_experts, 2 * inter))).astype(np.float32)
    s2_all = (s2 * rng.uniform(0.5, 2.0, (num_experts, hidden))).astype(np.float32)
    flat = indices.reshape(-1)
    want = np.zeros((flat.size, hidden), np.float32)
    for e in np.unique(flat):
        pairs = np.flatnonzero(flat == e)
        token = pairs // case.top_k
        want[pairs] = _reference(
            rows[token],
            row_scales[token],
            w1_all[e].astype(np.float32),
            w2_all[e].astype(np.float32),
            s1_all[e : e + 1],
            s2_all[e : e + 1],
            0,
            0,
            bf16_rows=True,
        )
    weights = (w1_all, w2_all, s1_all, s2_all)
    return indices, routing, rows.astype(FP8_NP), row_scales, weights, want


def _kernel_world(case):
    """(every shard's kernel operands stacked on their first axis, each
    routed pair's row in the stacked arrival rows, each pair's reference
    row, every shard's expert rows)."""
    indices, routing, rows, row_scales, weights, want = _kernel_inputs(*case[:7])
    shards, local = case.num_shards, case.experts_per_shard
    num_experts = shards * local
    routed_rows, arrival_rows = _kernel_geometry(case)
    table_rows = layout.align_up(routed_rows, layout.LANES)
    pair_token = np.arange(indices.size) // case.top_k
    per_shard = []
    for s in range(shards):
        split = dict(num_experts=num_experts, num_shards=shards)
        expert_rows, expert_base = tables.shard_expert_rows(routing, s, **split)
        # The token and scale tables the scatter fills: a routed row's token
        # and its row scale's f32 bits, zero where no pair lands.
        local_row = np.asarray(
            tables.local_routed_rows(routing, s, routed_rows_per_shard=routed_rows)
        )
        live = local_row < routed_rows
        token_of_row = np.zeros(table_rows, np.int32)
        token_of_row[local_row[live]] = pair_token[live]
        token_scales = np.zeros(
            layout.scale_table_rows(routed_rows) * layout.LANES, np.int32
        )
        token_scales[local_row[live]] = row_scales[pair_token[live], 0].view(np.int32)
        per_shard.append(
            (
                *tables.shard_transport_tables(routing, s, **split),
                *tables.shard_push_tables(routing, s, **split),
                token_of_row,
                expert_rows,
                expert_base,
                tables.visit_order(expert_rows, local),
                tables.shard_counts(routing, expert_rows, s, **split),
                # Sorted rows are the routed rows in expert order; otherwise
                # every shard's tokens, which the token table indexes.
                rows[token_of_row] if case.sorted_rows else rows,
                token_scales,
            )
        )
    operands = tuple(
        np.concatenate([np.asarray(shard[i]) for shard in per_shard])
        for i in range(len(per_shard[0]))
    ) + tuple(weights)
    arrival_index = (pair_token // case.tokens_per_shard) * arrival_rows + np.asarray(
        routing.arrival_row
    ).reshape(-1)
    expert_rows = np.concatenate([np.asarray(shard[8]) for shard in per_shard])
    return operands, arrival_index, want, expert_rows


def _kernel_program(case):
    """The built kernel over the case's shards, as one jitted function of the
    stacked operands returning the stacked (arrival rows, arrival scales).
    The one place this file builds and runs the kernel."""
    routed_rows, arrival_rows = _kernel_geometry(case)
    run = ffn_kernel.ffn_kernel(
        experts_per_shard=case.experts_per_shard,
        tile_rows=case.tile_rows,
        hidden=case.hidden,
        inter=case.inter,
        num_shards=case.num_shards,
        weight_format="fp8",
        routed_rows=routed_rows,
        result_rows=case.result_rows,
        elementwise_dtype=case.elementwise_dtype,
        sorted_rows=case.sorted_rows,
    )

    def shard(*operands):
        return run(operands[:7], *operands[7:], arrival_rows=arrival_rows)

    mesh = Mesh(np.asarray(jax.devices()[: case.num_shards]), (layout.MESH_AXIS,))
    split = PartitionSpec(layout.MESH_AXIS)
    n_operands = 7 + 11
    program = jax.jit(
        jax.shard_map(
            shard,
            mesh=mesh,
            in_specs=(split,) * n_operands,
            out_specs=(split, split),
            check_vma=False,
        )
    )
    return program, NamedSharding(mesh, split)


def _kernel_arrivals(case, operands):
    """Every shard's arrival rows, [shards * arrival rows, hidden] f32,
    dequantized with their scales for fp8 result rows."""
    program, sharding = _kernel_program(case)
    arrivals, scales = program(*(jax.device_put(a, sharding) for a in operands))
    rows = np.asarray(arrivals.astype(jnp.float32))
    rows = rows[:, : layout.row_lane_blocks(case.hidden)].reshape(-1, case.hidden)
    if case.result_rows == "fp8":
        rows = rows * np.asarray(scales)[:, None]
    return rows


def _row_errors(got, want):
    """Each row's root-mean-square error relative to the row's own root mean
    square."""
    return np.sqrt(np.mean((got - want) ** 2, axis=1) / np.mean(want**2, axis=1))


@pytest.fixture
def on_tpu7x(monkeypatch):
    """Skip unless the chip is a generation the package supports (a fixture,
    so nothing builds a client while pytest collects); an empty build
    cache."""
    if not device.supported_on(jax.devices()[0]):
        pytest.skip("the built FFN kernel needs a TPU7x chip")
    monkeypatch.setattr(ffn_kernel, "_BUILD_CACHE", {})


@pytest.mark.tpu_test
@pytest.mark.parametrize("name", list(KERNEL_CASES))
def test_the_built_kernel_matches_numpy_on_tpu(on_tpu7x, name):
    """Every routed pair's arrival row, on its token's shard, against the
    numpy reference of its expert's FFN on its token's row. The routing
    gives some experts several tiles and leaves one per shard without rows.
    fp8 result rows rounded on scale grids a hair apart and bf16
    element-wise products keep a correct row within a few percent (root
    mean square); a row from the wrong expert, weight slot, token, scale or
    landing row is off by about 140%, as the comparison against the
    references shifted by one pair shows."""
    case = KERNEL_CASES[name]
    if jax.device_count() < case.num_shards:
        pytest.skip(f"needs {case.num_shards} TPU cores")
    operands, arrival_index, want, expert_rows = _kernel_world(case)
    assert expert_rows.max() > case.tile_rows and expert_rows.min() == 0
    got = _kernel_arrivals(case, operands)[arrival_index]
    if case.result_rows == "fp8":
        quantized, scales = _quantize_np(want, 0)
        want = quantized * scales
    assert np.isfinite(got).all()
    errors = _row_errors(got, want)
    assert errors.max() < 0.1, (float(errors.max()), int(errors.argmax()))
    assert np.median(_row_errors(got, np.roll(want, 1, axis=0))) > 0.5
