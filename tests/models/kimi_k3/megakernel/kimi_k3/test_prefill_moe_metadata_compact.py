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

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    metadata_compact,
)


def _pack_route_wire(ids_by_rank: np.ndarray, probs_by_rank: np.ndarray) -> np.ndarray:
    rows = ids_by_rank.shape[1]
    parts = []
    for r in range(32):
        parts.append(ids_by_rank[r].reshape(rows // 8, 128))
        parts.append(probs_by_rank[r].view(np.int32).reshape(rows // 8, 128))
    return np.concatenate(parts, axis=0)


def _reference_routes(
    ids_by_rank: np.ndarray,
    probs_by_rank: np.ndarray,
    *,
    rank: int,
    local_experts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    tokens = ids_by_rank.shape[0] * ids_by_rank.shape[1]
    ids_all = ids_by_rank.reshape(tokens, 16)
    probs_all = probs_by_rank.reshape(tokens, 16)
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896).astype(np.int32)
    ep_groups = 896 // local_experts
    expert_tp = 32 // ep_groups
    first_expert = (rank // expert_tp) * local_experts
    expected_counts = hist_896[first_expert : first_expert + local_experts]
    expected_max = max(
        int(hist_896[g * local_experts : (g + 1) * local_experts].sum())
        for g in range(ep_groups)
    )
    tids_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    for expert in range(first_expert, first_expert + local_experts):
        tok_idx, slot_idx = np.where((ids_all == expert) & live_mask)
        order = np.lexsort((slot_idx, tok_idx))
        tids_list.append(tok_idx[order].astype(np.int32))
        probs_list.append(probs_all[tok_idx[order], slot_idx[order]].astype(np.float32))
    if tids_list:
        return (
            np.concatenate(tids_list),
            np.concatenate(probs_list),
            expected_counts,
            expected_max,
        )
    return (
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        expected_counts,
        expected_max,
    )


@functools.lru_cache(maxsize=16)
def _compile_compact_schedule_interpret(
    tokens: int,
    local_experts: int,
    with_maximum: bool,
):
    total_rows = tokens // 8
    counts_ready_calls = [0]
    step_count_box = [0]

    def kernel(
        wire_ref,
        rank_ref,
        keys_ref,
        probs_ref,
        tids_ref,
        counts_ref,
        max_ref,
        cursor_ref,
        tail_stages_ref,
    ):
        tail_stages_ref[0] = 0

        def _on_counts_ready() -> None:
            counts_ready_calls[0] += 1

        def _on_tail_stage() -> None:
            tail_stages_ref[0] = tail_stages_ref[0] + 1

        max_arg = max_ref if with_maximum else None
        schedule = metadata_compact.build_compact_metadata_schedule(
            wire_ref,
            keys_ref,
            probs_ref,
            tids_ref,
            counts_ref,
            max_arg,
            cursor_ref,
            tokens=tokens,
            rank=rank_ref[0],
            counts_ready=_on_counts_ready,
            local_experts=local_experts,
            on_tail_stage=_on_tail_stage,
        )
        step_count_box[0] = len(schedule.steps)
        for step_fn in schedule.steps:
            step_fn()
        schedule.tail()

    out_shapes = (
        jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        jax.ShapeDtypeStruct((total_rows, 128), jnp.float32),
        jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    fn = jax.jit(pl.pallas_call(kernel, out_shape=out_shapes, interpret=True))
    return fn, counts_ready_calls, step_count_box


def _run_compact_schedule_interpret(
    wire_np: np.ndarray,
    *,
    tokens: int,
    rank: int,
    local_experts: int,
    with_maximum: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None, int, int, int]:
    fn, counts_ready_calls, step_count_box = _compile_compact_schedule_interpret(
        tokens, local_experts, with_maximum
    )
    _, probs_out, tids_out, counts_out, max_out, _, tail_stages = fn(
        jnp.asarray(wire_np), jnp.asarray([rank], dtype=jnp.int32)
    )
    return (
        np.asarray(tids_out),
        np.asarray(probs_out),
        np.asarray(counts_out),
        int(np.asarray(max_out)[0]) if with_maximum else None,
        counts_ready_calls[0],
        step_count_box[0],
        int(np.asarray(tail_stages)[0]),
    )


def _make_routing_inputs(
    tokens: int, local_experts: int, regime: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = tokens // 32
    rng = np.random.default_rng(seed)
    if regime == "balanced":
        ids = rng.integers(0, 896, size=(32, rows, 16), dtype=np.int32)
    elif regime == "empty":
        ids = rng.integers(local_experts, 896, size=(32, rows, 16), dtype=np.int32)
    elif regime == "adv_lane0":
        ids = rng.integers(local_experts, 896, size=(32, rows, 16), dtype=np.int32)
        ids[:, :, :1] = rng.integers(
            0, local_experts, size=(32, rows, 1), dtype=np.int32
        )
    elif regime == "exact_t0":
        ids = rng.integers(local_experts, 896, size=(32, rows, 16), dtype=np.int32)
        ids[:, :, :2] = rng.integers(
            0, local_experts, size=(32, rows, 2), dtype=np.int32
        )
    elif regime == "skew_t1":
        mask = rng.random((32, rows, 16)) < 0.22
        ids = np.where(
            mask,
            rng.integers(0, local_experts, size=(32, rows, 16), dtype=np.int32),
            rng.integers(local_experts, 896, size=(32, rows, 16), dtype=np.int32),
        )
    elif regime == "overflow_t2":
        mask = rng.random((32, rows, 16)) < 0.42
        ids = np.where(
            mask,
            rng.integers(0, local_experts, size=(32, rows, 16), dtype=np.int32),
            rng.integers(local_experts, 896, size=(32, rows, 16), dtype=np.int32),
        )
    elif regime == "collapse_t3":
        ids = rng.integers(0, local_experts, size=(32, rows, 16), dtype=np.int32)
    else:
        raise ValueError(regime)
    probs = rng.uniform(0.01, 0.99, size=(32, rows, 16)).astype(np.float32)
    if regime != "exact_t0":
        probs[-1, -11:, :] = 0.0
    return ids, probs


@pytest.mark.parametrize(
    ("regime", "expect_tail"),
    [
        ("balanced", False),
        ("empty", False),
        ("adv_lane0", False),
        ("exact_t0", False),
        ("skew_t1", True),
        ("overflow_t2", True),
        ("collapse_t3", True),
    ],
)
def test_should_match_exact_routes_across_all_4_tiers_when_8k_ep16(regime, expect_tail):
    tokens, local_experts, rank = 8192, 56, 0
    ids_by_rank, probs_by_rank = _make_routing_inputs(
        tokens, local_experts, regime, seed=20261114 + len(regime)
    )
    wire_np = _pack_route_wire(ids_by_rank, probs_by_rank)

    tids_out, probs_out, counts_out, max_out, ready_calls, step_count, tail_stages = (
        _run_compact_schedule_interpret(
            wire_np,
            tokens=tokens,
            rank=rank,
            local_experts=local_experts,
            with_maximum=True,
        )
    )
    exp_tids, exp_probs, exp_counts, exp_max = _reference_routes(
        ids_by_rank, probs_by_rank, rank=rank, local_experts=local_experts
    )

    assert ready_calls == 1
    assert step_count == 19
    assert (tail_stages > 0) == expect_tail
    assert max_out == exp_max
    np.testing.assert_array_equal(counts_out, exp_counts)
    total_local = int(exp_counts.sum())
    np.testing.assert_array_equal(tids_out.reshape(-1)[:total_local], exp_tids)
    np.testing.assert_array_equal(probs_out.reshape(-1)[:total_local], exp_probs)


@pytest.mark.parametrize("rank", [0, 17, 31])
@pytest.mark.parametrize("with_maximum", [False, True])
def test_should_match_exact_routes_when_8k_ep8_non_zero_ranks_and_optional_max(
    rank, with_maximum
):
    tokens, local_experts = 8192, 112
    ids_by_rank, probs_by_rank = _make_routing_inputs(
        tokens, local_experts, "balanced", seed=20261115 + rank
    )
    wire_np = _pack_route_wire(ids_by_rank, probs_by_rank)

    tids_out, probs_out, counts_out, max_out, ready_calls, step_count, _ = (
        _run_compact_schedule_interpret(
            wire_np,
            tokens=tokens,
            rank=rank,
            local_experts=local_experts,
            with_maximum=with_maximum,
        )
    )
    exp_tids, exp_probs, exp_counts, exp_max = _reference_routes(
        ids_by_rank, probs_by_rank, rank=rank, local_experts=local_experts
    )

    assert ready_calls == 1
    assert step_count == 19
    assert max_out == (exp_max if with_maximum else None)
    np.testing.assert_array_equal(counts_out, exp_counts)
    total_local = int(exp_counts.sum())
    np.testing.assert_array_equal(tids_out.reshape(-1)[:total_local], exp_tids)
    np.testing.assert_array_equal(probs_out.reshape(-1)[:total_local], exp_probs)


@pytest.mark.parametrize(
    ("regime", "expect_tail"),
    [("balanced", False), ("empty", False), ("skew_t1", True)],
)
def test_should_match_exact_routes_when_16k_ep16(regime, expect_tail):
    tokens, local_experts, rank = 16384, 56, 0
    ids_by_rank, probs_by_rank = _make_routing_inputs(
        tokens, local_experts, regime, seed=20261116 + len(regime)
    )
    wire_np = _pack_route_wire(ids_by_rank, probs_by_rank)

    tids_out, probs_out, counts_out, max_out, ready_calls, step_count, tail_stages = (
        _run_compact_schedule_interpret(
            wire_np,
            tokens=tokens,
            rank=rank,
            local_experts=local_experts,
            with_maximum=True,
        )
    )
    exp_tids, exp_probs, exp_counts, exp_max = _reference_routes(
        ids_by_rank, probs_by_rank, rank=rank, local_experts=local_experts
    )

    assert ready_calls == 1
    assert step_count == 49
    assert (tail_stages > 0) == expect_tail
    assert max_out == exp_max
    np.testing.assert_array_equal(counts_out, exp_counts)
    total_local = int(exp_counts.sum())
    np.testing.assert_array_equal(tids_out.reshape(-1)[:total_local], exp_tids)
    np.testing.assert_array_equal(probs_out.reshape(-1)[:total_local], exp_probs)


@pytest.mark.parametrize(
    ("tokens", "local_experts", "with_maximum", "expected_steps"),
    [
        (8192, 56, True, 19),
        (8192, 112, False, 19),
        (16384, 56, True, 49),
        (16384, 112, True, 49),
    ],
)
def test_should_trace_cleanly_for_tpu_pallas_when_8k_or_16k(
    tokens, local_experts, with_maximum, expected_steps
):
    total_rows = tokens // 8
    vmem, smem = pltpu.VMEM, pltpu.SMEM
    step_count_box = [0]

    def body(wire_ref, out_counts_ref, scratch):
        schedule = metadata_compact.build_compact_metadata_schedule(
            wire_ref,
            scratch["keys"],
            scratch["probs"],
            scratch["tids"],
            out_counts_ref,
            scratch["maximum"] if with_maximum else None,
            scratch["cursor"],
            tokens=tokens,
            rank=jnp.int32(0),
            counts_ready=lambda: None,
            local_experts=local_experts,
        )
        step_count_box[0] = len(schedule.steps)
        for step_fn in schedule.steps:
            step_fn()
        schedule.tail()

    scratch_spec = dict(
        keys=vmem((total_rows, 128), jnp.int32),
        probs=vmem((total_rows, 128), jnp.float32),
        tids=vmem((total_rows, 128), jnp.int32),
        maximum=vmem((1,), jnp.int32),
        cursor=smem((1,), jnp.int32),
    )
    fn = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        scratch_shapes=(scratch_spec,),
    )
    jaxpr = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((tokens // 4, 128), jnp.int32))
    assert len(jaxpr.eqns) == 1
    assert step_count_box[0] == expected_steps


def test_should_detect_perturbed_probability_as_positive_control():
    tokens, local_experts, rank = 8192, 56, 0
    ids_by_rank, probs_by_rank = _make_routing_inputs(
        tokens, local_experts, "balanced", seed=20261117
    )
    wire_np = _pack_route_wire(ids_by_rank, probs_by_rank)
    _, probs_out, counts_out, _, _, _, _ = _run_compact_schedule_interpret(
        wire_np,
        tokens=tokens,
        rank=rank,
        local_experts=local_experts,
        with_maximum=True,
    )
    _, exp_probs, exp_counts, _ = _reference_routes(
        ids_by_rank, probs_by_rank, rank=rank, local_experts=local_experts
    )
    total_local = int(exp_counts.sum())
    assert total_local > 0
    perturbed = exp_probs.copy()
    perturbed[0] += np.float32(0.125)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(probs_out.reshape(-1)[:total_local], perturbed)
    np.testing.assert_array_equal(counts_out, exp_counts)
