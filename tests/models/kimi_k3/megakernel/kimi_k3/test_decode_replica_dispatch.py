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

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config, layout
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel as dm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import rank_loader as rl


def _reference_mxfp4_visit_cost(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float32)
    return np.where(c > 0.0, 1.74 + 0.31 * c + 0.67 * (np.ceil(c / 8.0) - 1.0), 0.0)


def _reference_max_straggler_greedy_dispatch(
    winners: np.ndarray,
    table: np.ndarray,
    entries: int,
    local_expert_count: int,
    counts: np.ndarray,
    *,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ep_groups = counts.shape[1] // local_expert_count
    batch_size, k_sel = winners.shape
    r_ids = table[0, :entries].astype(np.int32)
    r_holders = table[0, entries : 2 * entries].astype(np.int32)
    r_slots = table[0, 2 * entries : 3 * entries].astype(np.int32)
    valid = r_ids >= 0

    r_counts = np.zeros(entries, dtype=np.float32)
    for i in range(entries):
        if valid[i]:
            r_counts[i] = float(np.sum(winners == r_ids[i]))
    r_cost = np.where(valid, _reference_mxfp4_visit_cost(r_counts), 0.0)
    r_home = np.where(valid, r_ids // local_expert_count, -1)

    cost_all = _reference_mxfp4_visit_cost(counts[0])
    loads = np.array(
        [
            float(
                np.sum(cost_all[g * local_expert_count : (g + 1) * local_expert_count])
            )
            for g in range(ep_groups)
        ],
        dtype=np.float32,
    )
    used = np.zeros(entries, dtype=bool)
    moved = np.zeros(entries, dtype=bool)

    for _ in range(max_iters):
        max_load = float(np.max(loads))
        g_max = int(np.argmax(loads))
        best_score = 1e9
        best_idx = -1
        for i in range(entries):
            if r_home[i] == g_max and r_cost[i] > 0.0 and not used[i]:
                cg = int(r_holders[i])
                cand_peak = max(
                    float(loads[cg] + r_cost[i]), float(max_load - r_cost[i])
                )
                score = cand_peak + 1e-6 * i
                if score < best_score:
                    best_score = score
                    best_idx = i
        if best_idx >= 0 and best_score < max_load - 1e-4:
            cg = int(r_holders[best_idx])
            eid = int(r_ids[best_idx])
            loads[g_max] -= r_cost[best_idx]
            loads[cg] += r_cost[best_idx]
            used |= r_ids == eid
            moved[best_idx] = True

    to_copy = np.zeros((batch_size, k_sel), dtype=bool)
    holder = np.zeros((batch_size, k_sel), dtype=np.int32)
    slot = np.zeros((batch_size, k_sel), dtype=np.int32)
    for i in range(entries):
        if moved[i]:
            mask = winners == r_ids[i]
            to_copy |= mask
            holder = np.where(mask, r_holders[i], holder)
            slot = np.where(mask, r_slots[i], slot)
    return to_copy, holder, slot, float(np.max(loads))


def _build_synthetic_replica_case(
    seed: int,
    batch_size: int,
    ep_groups: int,
    entries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    num_experts = 896
    local_expert_count = num_experts // ep_groups
    table_width = 256 if 3 * entries <= 256 else 512
    table = np.zeros((1, table_width), dtype=np.float32)

    hot_groups = (0, ep_groups - 1)
    for i in range(entries):
        home_g = hot_groups[i % len(hot_groups)]
        eid = home_g * local_expert_count + (i // len(hot_groups)) % local_expert_count
        holder_g = 1 + (i % (ep_groups - 2))
        table[0, i] = float(eid)
        table[0, entries + i] = float(holder_g)
        table[0, 2 * entries + i] = float(local_expert_count + (i % 4))

    k_sel = 16
    winners = np.zeros((batch_size, k_sel), dtype=np.int32)
    hot_pool = np.unique(table[0, :entries].astype(np.int32))
    cold_pool = np.setdiff1d(np.arange(num_experts, dtype=np.int32), hot_pool)
    for b in range(batch_size):
        n_hot = min(10, len(hot_pool))
        chosen_hot = rng.choice(hot_pool, size=n_hot, replace=False)
        chosen_cold = rng.choice(cold_pool, size=k_sel - n_hot, replace=False)
        winners[b] = np.concatenate([chosen_hot, chosen_cold])

    counts = np.bincount(winners.reshape(-1), minlength=num_experts)[None, :].astype(
        np.float32
    )
    return winners, table, counts, local_expert_count


@pytest.mark.parametrize(
    ("batch_size", "ep_groups", "entries", "seed"),
    [
        (1, 16, 64, 101),
        (4, 16, 48, 102),
        (8, 16, 64, 103),
        (16, 16, 64, 104),
        (8, 8, 32, 105),
        (8, 16, 96, 106),
    ],
)
def test_should_match_independent_greedy_oracle_across_batches_ep_and_chunk_boundaries(
    batch_size: int, ep_groups: int, entries: int, seed: int
) -> None:
    winners, table, counts, local_expert_count = _build_synthetic_replica_case(
        seed, batch_size, ep_groups, entries
    )
    max_iters = 4 if batch_size <= 8 else 8

    to_copy, holder, slot = dm.replica_dispatch(
        jnp.asarray(winners),
        jnp.asarray(table),
        entries,
        local_expert_count,
        jnp.asarray(counts),
    )
    exp_tc, exp_h, exp_sl, _ = _reference_max_straggler_greedy_dispatch(
        winners, table, entries, local_expert_count, counts, max_iters=max_iters
    )

    assert np.any(exp_tc)
    np.testing.assert_array_equal(np.asarray(to_copy), exp_tc)
    np.testing.assert_array_equal(np.asarray(holder), exp_h)
    np.testing.assert_array_equal(np.asarray(slot), exp_sl)

    perturbed_table = table.copy()
    perturbed_table[0, entries : 2 * entries] = (
        perturbed_table[0, entries : 2 * entries].astype(np.int32) + 1
    ) % ep_groups
    wrong_tc, wrong_h, _ = dm.replica_dispatch(
        jnp.asarray(winners),
        jnp.asarray(perturbed_table),
        entries,
        local_expert_count,
        jnp.asarray(counts),
    )
    assert not (
        np.array_equal(np.asarray(wrong_tc), exp_tc)
        and np.array_equal(np.asarray(wrong_h), exp_h)
    )


def test_should_dispatch_real_ep16_plain4_plan_and_reduce_bottleneck_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config,
        "active",
        lambda: config.KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=config.EP16_DEFAULT_EXPERT_PLAN,
            kd_replica_plan=config.EP16_DEFAULT_REPLICA_PLAN,
        ),
    )
    ep16_order = rl.prefill_expert_plan(layout.NUM_LAYERS)
    assert ep16_order is not None
    loaded = rl.replica_plan(0, layout.NUM_LAYERS, expert_order=ep16_order)
    assert loaded is not None
    _, experts_plan, order_plan = loaded
    batch_size, k_sel, num_experts, local_expert_count, entries, slots = (
        8,
        16,
        896,
        56,
        64,
        4,
    )
    rng = np.random.default_rng(2026)
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    probs = ranks ** (-1.05)
    probs /= probs.sum()

    total_reduction = 0.0
    for layer_idx in (1, 11, 21, 41, 61, 81):
        table = np.zeros((1, 256), dtype=np.float32)
        for i, flat in enumerate(order_plan[layer_idx]):
            g, j = divmod(int(flat), slots)
            eid = int(experts_plan[layer_idx, g, j])
            table[0, i] = float(eid)
            table[0, entries + i] = float(g if eid >= 0 else -1)
            table[0, 2 * entries + i] = float(
                local_expert_count + j if eid >= 0 else -1
            )

        hot_eids = [int(x) for x in table[0, :entries] if x >= 0]
        cold_eids = [e for e in range(num_experts) if e not in set(hot_eids)]
        perm = np.array(hot_eids + list(rng.permutation(cold_eids)), dtype=np.int32)
        winners = np.stack(
            [
                rng.choice(perm, size=k_sel, replace=False, p=probs)
                for _ in range(batch_size)
            ],
            axis=0,
        )
        counts = np.bincount(winners.reshape(-1), minlength=num_experts)[
            None, :
        ].astype(np.float32)

        cost_all = _reference_mxfp4_visit_cost(counts[0])
        base_peak = max(
            float(
                np.sum(cost_all[g * local_expert_count : (g + 1) * local_expert_count])
            )
            for g in range(16)
        )
        exp_tc, exp_h, exp_sl, new_peak = _reference_max_straggler_greedy_dispatch(
            winners, table, entries, local_expert_count, counts, max_iters=4
        )
        to_copy, holder, slot = dm.replica_dispatch(
            jnp.asarray(winners),
            jnp.asarray(table),
            entries,
            local_expert_count,
            jnp.asarray(counts),
        )

        np.testing.assert_array_equal(np.asarray(to_copy), exp_tc)
        np.testing.assert_array_equal(np.asarray(holder), exp_h)
        np.testing.assert_array_equal(np.asarray(slot), exp_sl)
        assert new_peak <= base_peak + 1e-5
        total_reduction += base_peak - new_peak

    assert total_reduction > 1.0


def test_should_never_move_negative_sentinel_dead_rows_or_empty_table_slots() -> None:
    local_expert_count = 56
    entries = 4
    winners = np.zeros((8, 16), dtype=np.int32)
    for b in range(6):
        winners[b, :12] = np.arange(0, 12, dtype=np.int32)
        winners[b, 12:] = np.arange(112, 116, dtype=np.int32)
    winners[6, :] = -1
    winners[7, :] = -2
    live_flat = winners[:6].reshape(-1)
    counts = np.bincount(live_flat, minlength=896)[None, :].astype(np.float32)

    table = np.zeros((1, 256), dtype=np.float32)
    table[0, 0:4] = [-1.0, 0.0, 1.0, -1.0]
    table[0, 4:8] = [9.0, 5.0, 6.0, 7.0]
    table[0, 8:12] = [56.0, 57.0, 58.0, 59.0]

    to_copy, holder, slot = dm.replica_dispatch(
        jnp.asarray(winners),
        jnp.asarray(table),
        entries,
        local_expert_count,
        jnp.asarray(counts),
    )
    tc_np = np.asarray(to_copy)
    h_np = np.asarray(holder)
    sl_np = np.asarray(slot)

    assert np.all(tc_np[:6, 0])
    np.testing.assert_array_equal(h_np[:6, 0], 5)
    np.testing.assert_array_equal(sl_np[:6, 0], 57)
    assert not np.any(tc_np[6, :])
    assert not np.any(tc_np[7, :])
    np.testing.assert_array_equal(h_np[6:, :], 0)
    np.testing.assert_array_equal(sl_np[6:, :], 0)


def test_should_move_replicated_expert_at_most_once_to_coolest_copy_holder() -> None:
    local_expert_count = 56
    entries = 3
    winners = np.zeros((4, 16), dtype=np.int32)
    for b in range(4):
        winners[b, :7] = np.arange(0, 7, dtype=np.int32)
        winners[b, 7:13] = np.arange(3 * 56, 3 * 56 + 6, dtype=np.int32)
        winners[b, 13:16] = np.arange(5 * 56, 5 * 56 + 3, dtype=np.int32)
    counts = np.bincount(winners.reshape(-1), minlength=896)[None, :].astype(np.float32)

    table = np.zeros((1, 256), dtype=np.float32)
    table[0, :entries] = [0.0, 0.0, 1.0]
    table[0, entries : 2 * entries] = [3.0, 10.0, 11.0]
    table[0, 2 * entries : 3 * entries] = [56.0, 58.0, 59.0]

    to_copy, holder, slot = dm.replica_dispatch(
        jnp.asarray(winners),
        jnp.asarray(table),
        entries,
        local_expert_count,
        jnp.asarray(counts),
    )
    tc_np = np.asarray(to_copy)
    h_np = np.asarray(holder)
    sl_np = np.asarray(slot)

    assert np.all(tc_np[:, 0])
    np.testing.assert_array_equal(h_np[:, 0], 10)
    np.testing.assert_array_equal(sl_np[:, 0], 58)


def test_should_use_parity_split_dispatch_fallback_when_batch_exceeds_16() -> None:
    batch_size = 24
    local_expert_count = 56
    entries = 1
    other_groups = [g for g in range(1, 16) if g != 4]
    winners = np.zeros((batch_size, 2), dtype=np.int32)
    for b in range(batch_size):
        winners[b, 0] = 0
        winners[b, 1] = other_groups[b % len(other_groups)] * local_expert_count
    counts = np.bincount(winners.reshape(-1), minlength=896)[None, :].astype(np.float32)

    table = np.zeros((1, 256), dtype=np.float32)
    table[0, 0] = 0.0
    table[0, 1] = 4.0
    table[0, 2] = 56.0

    to_copy, holder, slot = dm.replica_dispatch(
        jnp.asarray(winners),
        jnp.asarray(table),
        entries,
        local_expert_count,
        jnp.asarray(counts),
    )
    tc_np = np.asarray(to_copy)
    h_np = np.asarray(holder)
    sl_np = np.asarray(slot)

    expected_odd = (np.arange(batch_size) % 2) == 1
    np.testing.assert_array_equal(tc_np[:, 0], expected_odd)
    np.testing.assert_array_equal(h_np[expected_odd, 0], 4)
    np.testing.assert_array_equal(sl_np[expected_odd, 0], 56)


def test_should_classify_93_layers_identically_for_kda_first_and_mla() -> None:
    mla_set = set(layout.MLA)
    for layer_idx in range(layout.NUM_LAYERS):
        is_delta = (layer_idx % 4 != 3) and (layer_idx != layout.NUM_LAYERS - 1)
        assert is_delta == (layer_idx not in mla_set)
    assert 0 not in mla_set
    assert (layout.NUM_LAYERS - 1) in mla_set
