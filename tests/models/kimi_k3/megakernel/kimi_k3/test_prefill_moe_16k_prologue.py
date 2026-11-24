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

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    fused,
    metadata_compact,
    routed_16k,
    shared_down_16k,
    w4a8_16k,
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
    return (
        np.concatenate(tids_list),
        np.concatenate(probs_list),
        expected_counts,
        expected_max,
    )


def _reference_online_1pass(
    ids_by_rank: np.ndarray,
    probs_by_rank: np.ndarray,
    table_flat: np.ndarray,
    *,
    base_local_experts: int,
    replica_slots: int,
    min_gap: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
    rows = ids_by_rank.shape[1]
    tokens = 32 * rows
    ep_groups = 896 // base_local_experts
    eff_le = base_local_experts + replica_slots
    n_rep = ep_groups * replica_slots
    he_arr = table_flat[:n_rep]
    hg_arr = table_flat[n_rep : 2 * n_rep]
    rg_arr = table_flat[2 * n_rep : 3 * n_rep]
    rs_arr = table_flat[3 * n_rep : 4 * n_rep]

    rank_half = tokens // 256
    ranks_per_win = 128 // rank_half
    n_wins = (tokens // 8) // 128
    group_loads = np.zeros((ep_groups,), dtype=np.int64)
    assigned: list[list[list[tuple[int, float]]]] = [
        [[] for _ in range(eff_le)] for _ in range(ep_groups)
    ]

    for win_idx in range(n_wins):
        r0 = win_idx * ranks_per_win
        r1 = r0 + ranks_per_win
        ids_win = ids_by_rank[r0:r1].reshape(-1)
        probs_win = probs_by_rank[r0:r1].reshape(-1)
        ids_live = np.where(probs_win != 0, ids_win, -1)
        for grp in range(ep_groups):
            lo = grp * base_local_experts
            group_loads[grp] += int(
                np.sum((ids_live >= lo) & (ids_live < lo + base_local_experts))
            )
        moves: dict[int, tuple[int, int]] = {}
        if int(group_loads.max() - group_loads.min()) >= min_gap:
            for k in range(n_rep):
                he = int(he_arr[k])
                if he < 0:
                    continue
                hg, rg = int(hg_arr[k]), int(rg_arr[k])
                gap = int(group_loads[hg] - group_loads[rg])
                if gap < min_gap:
                    continue
                mv = int(np.sum(ids_live == he))
                if 0 < mv <= gap:
                    group_loads[hg] -= mv
                    group_loads[rg] += mv
                    moves[he] = (rg, int(rs_arr[k]))
        base_pos = win_idx * (128 * 128)
        for pos in range(ids_live.size):
            e = int(ids_live[pos])
            if e < 0:
                continue
            tok = (base_pos + pos) // 16
            prob = float(probs_win[pos])
            dest_g, dest_slot = (
                moves[e] if e in moves else divmod(e, base_local_experts)
            )
            assigned[dest_g][dest_slot].append((tok, prob))

    tids_by_g, probs_by_g, counts_by_g = [], [], []
    for g in range(ep_groups):
        counts_by_g.append(
            np.array([len(assigned[g][e]) for e in range(eff_le)], np.int32)
        )
        t_list, p_list = [], []
        for e in range(eff_le):
            for tok, prob in assigned[g][e]:
                t_list.append(tok)
                p_list.append(prob)
        tids_by_g.append(np.array(t_list, dtype=np.int32))
        probs_by_g.append(np.array(p_list, dtype=np.float32))
    return tids_by_g, probs_by_g, counts_by_g, int(group_loads.max())


@functools.lru_cache(maxsize=16)
def _compile_sequential_metadata_interpret(
    tokens: int,
    local_experts: int,
    with_maximum: bool,
    poll_win_stride: int,
    replica_slots: int,
):
    total_rows = tokens // 8
    use_replicas = replica_slots > 0
    base_le = local_experts - replica_slots if use_replicas else local_experts
    ep_groups = 896 // base_le
    n_rep = ep_groups * replica_slots if use_replicas else 0

    def kernel(
        wire_ref,
        rank_ref,
        *rest,
    ):
        if use_replicas:
            rep_table_ref = rest[0]
            (
                keys_ref,
                probs_ref,
                tids_ref,
                counts_ref,
                max_ref,
                cursor_ref,
                stats_ref,
            ) = rest[1:]
        else:
            rep_table_ref = None
            (
                keys_ref,
                probs_ref,
                tids_ref,
                counts_ref,
                max_ref,
                cursor_ref,
                stats_ref,
            ) = rest

        stats_ref[0] = 0
        stats_ref[1] = 0

        def _on_counts_ready() -> None:
            stats_ref[0] = stats_ref[0] + 1

        def _on_poll() -> None:
            stats_ref[1] = stats_ref[1] + 1

        def _body(rep_s) -> None:
            rep_scratch = (
                metadata_compact.ReplicaRoutingScratch(
                    group_loads=rep_s["group_loads"],
                    my_drop_experts=rep_s["drop_e"],
                    my_add_experts=rep_s["add_e"],
                    my_add_slots=rep_s["add_slot"],
                    my_counts=rep_s["cnts"],
                )
                if use_replicas
                else None
            )
            metadata_compact.run_sequential_compact_metadata(
                wire_ref,
                keys_ref,
                probs_ref,
                tids_ref,
                counts_ref,
                max_ref if with_maximum else None,
                cursor_ref,
                tokens=tokens,
                rank=rank_ref[0],
                counts_ready=_on_counts_ready,
                local_experts=local_experts,
                on_poll=_on_poll,
                poll_win_stride=poll_win_stride,
                replica_table=rep_table_ref,
                replica_slots=replica_slots,
                replica_scratch=rep_scratch,
            )

        pl.run_scoped(
            _body,
            dict(
                group_loads=pltpu.SMEM((ep_groups,), jnp.int32)
                if use_replicas
                else None,
                drop_e=pltpu.SMEM((n_rep,), jnp.int32) if use_replicas else None,
                add_e=pltpu.SMEM((n_rep,), jnp.int32) if use_replicas else None,
                add_slot=pltpu.SMEM((n_rep,), jnp.int32) if use_replicas else None,
                cnts=pltpu.SMEM((2,), jnp.int32) if use_replicas else None,
            ),
        )

    out_shapes = (
        jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        jax.ShapeDtypeStruct((total_rows, 128), jnp.float32),
        jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )
    return jax.jit(pl.pallas_call(kernel, out_shape=out_shapes, interpret=True))


def _make_scenario_routes(
    scenario: str,
    *,
    tokens: int = 16384,
    local_experts: int = 56,
    rank: int = 7,
    seed: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = tokens // 32
    ep_groups = 896 // local_experts
    expert_tp = 32 // ep_groups
    first_e = (rank // expert_tp) * local_experts

    ids = rng.integers(0, 896, size=(32, rows, 16), dtype=np.int32)
    probs = rng.uniform(0.05, 0.95, size=(32, rows, 16)).astype(np.float32)

    if scenario == "empty":
        ids[...] = first_e + 3
        probs[...] = 0.0
    elif scenario == "balanced_t0":
        dead = rng.random(ids.shape) < 0.15
        ids[dead] = first_e + 5
        probs[dead] = 0.0
    elif scenario == "exact_t0_boundary":
        ids[...] = (first_e + local_experts) % 896
        probs[...] = 0.25
        for w in range(16):
            r0 = w * 2
            flat_ids = ids[r0 : r0 + 2].reshape(-1)
            flat_probs = probs[r0 : r0 + 2].reshape(-1)
            flat_ids[: 16 * 128] = first_e + (np.arange(16 * 128) % local_experts)
            flat_probs[: 16 * 128] = rng.uniform(0.1, 0.9, size=16 * 128).astype(
                np.float32
            )
    elif scenario == "skew_t1":
        hot = rng.random(ids.shape) < 0.16
        ids[hot] = rng.integers(
            first_e, first_e + local_experts, size=int(hot.sum()), dtype=np.int32
        )
    elif scenario == "overflow_t2":
        hot = rng.random(ids.shape) < 0.38
        ids[hot] = rng.integers(
            first_e, first_e + local_experts, size=int(hot.sum()), dtype=np.int32
        )
    elif scenario == "collapse_t3":
        hot = rng.random(ids.shape) < 0.72
        ids[hot] = rng.integers(
            first_e, first_e + local_experts, size=int(hot.sum()), dtype=np.int32
        )
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return ids, probs


@pytest.mark.parametrize(
    "scenario",
    [
        "balanced_t0",
        "empty",
        "exact_t0_boundary",
        "skew_t1",
        "overflow_t2",
        "collapse_t3",
    ],
)
def test_should_match_reference_routes_across_all_tiers_without_replicas(scenario):
    rank = 7
    local_experts = 56
    ids, probs = _make_scenario_routes(
        scenario, tokens=16384, local_experts=local_experts, rank=rank
    )
    wire = _pack_route_wire(ids, probs)
    call = _compile_sequential_metadata_interpret(16384, local_experts, True, 2, 0)

    _, got_probs, got_tids, got_counts, got_max, got_cursor, got_stats = call(
        jnp.asarray(wire), jnp.asarray([rank], dtype=jnp.int32)
    )
    exp_tids, exp_probs, exp_counts, exp_max = _reference_routes(
        ids, probs, rank=rank, local_experts=local_experts
    )
    total_live = int(exp_counts.sum())

    np.testing.assert_array_equal(np.asarray(got_counts), exp_counts)
    assert int(got_max[0]) == exp_max
    np.testing.assert_array_equal(
        np.asarray(got_tids).reshape(-1)[:total_live], exp_tids
    )
    np.testing.assert_array_equal(
        np.asarray(got_probs).reshape(-1)[:total_live], exp_probs
    )
    assert int(got_stats[0]) == 1
    assert int(got_stats[1]) >= 9
    if scenario == "empty":
        assert int(got_cursor[0]) == 0
    elif scenario == "exact_t0_boundary":
        assert int(got_cursor[0]) == 256
    elif scenario == "skew_t1":
        assert 256 < int(got_cursor[0]) <= 512
    elif scenario == "overflow_t2":
        assert 512 < int(got_cursor[0]) <= 1024
    elif scenario == "collapse_t3":
        assert 1024 < int(got_cursor[0]) <= 2048

    if total_live > 0:
        bad_probs = probs.copy()
        live_idx = np.argwhere(
            (ids >= 3 * local_experts) & (ids < 4 * local_experts) & (probs != 0)
        )[0]
        bad_probs[tuple(live_idx)] = 0.0
        _, _, bad_counts, _ = _reference_routes(
            ids, bad_probs, rank=rank, local_experts=local_experts
        )
        assert not np.array_equal(np.asarray(got_counts), bad_counts)


@pytest.mark.parametrize(
    ("tokens", "local_experts", "with_maximum", "poll_win_stride"),
    [
        (16384, 56, False, 1),
        (16384, 112, True, 2),
        (8192, 56, True, 2),
    ],
)
def test_should_support_optional_maximum_ep8_and_8k_in_sequential_metadata(
    tokens, local_experts, with_maximum, poll_win_stride
):
    rank = 5
    ids, probs = _make_scenario_routes(
        "balanced_t0", tokens=tokens, local_experts=local_experts, rank=rank, seed=202
    )
    wire = _pack_route_wire(ids, probs)
    call = _compile_sequential_metadata_interpret(
        tokens, local_experts, with_maximum, poll_win_stride, 0
    )

    _, got_probs, got_tids, got_counts, got_max, _, got_stats = call(
        jnp.asarray(wire), jnp.asarray([rank], dtype=jnp.int32)
    )
    exp_tids, exp_probs, exp_counts, exp_max = _reference_routes(
        ids, probs, rank=rank, local_experts=local_experts
    )
    total_live = int(exp_counts.sum())

    np.testing.assert_array_equal(np.asarray(got_counts), exp_counts)
    np.testing.assert_array_equal(
        np.asarray(got_tids).reshape(-1)[:total_live], exp_tids
    )
    np.testing.assert_array_equal(
        np.asarray(got_probs).reshape(-1)[:total_live], exp_probs
    )
    if with_maximum:
        assert int(got_max[0]) == exp_max
    assert int(got_stats[0]) == 1
    if poll_win_stride == 1:
        assert int(got_stats[1]) >= (tokens // 8) // 128 + 1


@pytest.mark.parametrize(
    ("base_le", "replica_slots", "hot_prob", "seed"),
    [
        (56, 4, 0.14, 301),
        (56, 4, 0.28, 302),
        (224, 1, 0.45, 303),
    ],
)
def test_should_match_online_1pass_replica_oracle_including_tail_batch_and_rebalance(
    base_le, replica_slots, hot_prob, seed
):
    rng = np.random.default_rng(seed)
    tokens = 16384
    rows = tokens // 32
    ep_groups = 896 // base_le
    expert_tp = 32 // ep_groups
    eff_le = base_le + replica_slots
    n_rep = ep_groups * replica_slots

    he = np.full((n_rep,), -1, dtype=np.int32)
    hg = np.zeros((n_rep,), dtype=np.int32)
    rg = np.zeros((n_rep,), dtype=np.int32)
    rs = np.zeros((n_rep,), dtype=np.int32)
    for s in range(replica_slots):
        idx = (ep_groups - 1) * replica_slots + s
        he[idx] = s
        hg[idx] = 0
        rg[idx] = ep_groups - 1
        rs[idx] = base_le + s
    he[0] = replica_slots
    hg[0] = 0
    rg[0] = 1
    rs[0] = base_le
    table_flat = np.concatenate([he, hg, rg, rs])

    ids = rng.integers(0, 896, size=(32, rows, 16), dtype=np.int32)
    probs = rng.uniform(0.05, 0.95, size=(32, rows, 16)).astype(np.float32)
    hot_mask = rng.random(ids.shape) < hot_prob
    hot_pool = np.arange(0, replica_slots + 2, dtype=np.int32)
    ids[hot_mask] = rng.choice(hot_pool, size=int(hot_mask.sum()))
    dead_mask = rng.random(ids.shape) < 0.08
    ids[dead_mask] = 0
    probs[dead_mask] = 0.0

    wire = _pack_route_wire(ids, probs)
    min_gap = metadata_compact.default_replica_min_gap(tokens)
    exp_tids_by_g, exp_probs_by_g, exp_counts_by_g, exp_max = _reference_online_1pass(
        ids,
        probs,
        table_flat,
        base_local_experts=base_le,
        replica_slots=replica_slots,
        min_gap=min_gap,
    )
    moved_total = int(sum(exp_counts_by_g[g][base_le:].sum() for g in range(ep_groups)))
    assert moved_total > 0

    call = _compile_sequential_metadata_interpret(
        tokens, eff_le, True, 2, replica_slots
    )
    for group in (0, 1, ep_groups - 1):
        rank = group * expert_tp
        _, got_probs, got_tids, got_counts, got_max, _, _ = call(
            jnp.asarray(wire),
            jnp.asarray([rank], dtype=jnp.int32),
            jnp.asarray(table_flat),
        )
        total_live = int(exp_counts_by_g[group].sum())
        np.testing.assert_array_equal(np.asarray(got_counts), exp_counts_by_g[group])
        assert int(got_max[0]) == exp_max
        np.testing.assert_array_equal(
            np.asarray(got_tids).reshape(-1)[:total_live], exp_tids_by_g[group]
        )
        np.testing.assert_array_equal(
            np.asarray(got_probs).reshape(-1)[:total_live], exp_probs_by_g[group]
        )


@pytest.mark.parametrize(("replica_slots", "has_scratch"), [(0, True), (4, False)])
def test_should_reject_invalid_replica_args_in_sequential_metadata(
    replica_slots, has_scratch
):
    dummy = jax.ShapeDtypeStruct((1,), jnp.int32)
    scratch = (
        metadata_compact.ReplicaRoutingScratch(dummy, dummy, dummy, dummy, dummy)
        if has_scratch
        else None
    )
    with pytest.raises(ValueError, match="replica_table requires"):
        metadata_compact.run_sequential_compact_metadata(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            None,
            dummy,
            tokens=16384,
            rank=0,
            counts_ready=lambda: None,
            replica_table=dummy,
            replica_slots=replica_slots,
            replica_scratch=scratch,
        )


def test_should_stage_middle_and_execute_deferred_shared_down_without_aliasing():
    rng = np.random.default_rng(404)
    rows = 128
    inter = shared_down_16k.SHARED_INTER_16K
    hidden = shared_down_16k.SHARED_HIDDEN_16K
    middle_np = np.concatenate(
        [
            rng.normal(0.15, 0.35, size=(rows, inter)).astype(np.float32),
            rng.normal(-0.25, 0.40, size=(rows, inter)).astype(np.float32),
        ],
        axis=0,
    )
    w_down_np = rng.normal(0.0, 0.05, size=(inter, hidden)).astype(np.float32)
    middle_bf16 = jnp.asarray(middle_np, dtype=jnp.bfloat16)
    w_down_bf16 = jnp.asarray(w_down_np, dtype=jnp.bfloat16)

    ref_h0 = jnp.matmul(
        middle_bf16[:rows], w_down_bf16, preferred_element_type=jnp.float32
    )
    ref_h1 = jnp.matmul(
        middle_bf16[rows:], w_down_bf16, preferred_element_type=jnp.float32
    )
    expected_bf16 = (ref_h0 + ref_h1).astype(jnp.bfloat16)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def kernel(mid_in_hbm, w_down_hbm, shared_hbm, poll_cnt_ref):
        poll_cnt_ref[0] = 0

        def _on_poll() -> None:
            poll_cnt_ref[0] = poll_cnt_ref[0] + 1

        def _body(s) -> None:
            for panel in range(12):
                for half in range(2):
                    load = pltpu.make_async_copy(
                        mid_in_hbm.at[
                            pl.ds(half * rows, rows), pl.ds(panel * 256, 256)
                        ],
                        s["tile"],
                        s["load_sem"],
                    )
                    load.start()
                    load.wait()
                    stage = shared_down_16k.stage_middle_half_copy(
                        s["tile"],
                        shared_hbm,
                        s["stage_sem"],
                        half=half,
                        panel=panel,
                    )
                    stage.start()
                    stage.wait()

            shared_down_16k.run_deferred_shared_down_from_hbm(
                shared_hbm,
                w_down_hbm,
                s["down_ready"],
                rows=rows,
                parity=0,
                peer=0,
                on_poll=_on_poll,
            )

        pl.run_scoped(
            _body,
            dict(
                tile=pltpu.VMEM((rows, 256), jnp.bfloat16),
                load_sem=pltpu.SemaphoreType.DMA,
                stage_sem=pltpu.SemaphoreType.DMA,
                down_ready=pltpu.SemaphoreType.REGULAR,
            ),
        )

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm, hbm),
            out_specs=(hbm, pl.BlockSpec(memory_space=pltpu.SMEM)),
        ),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    got_shared, got_polls = call(middle_bf16, w_down_bf16)

    np.testing.assert_array_equal(
        np.asarray(got_shared.astype(jnp.float32)),
        np.asarray(expected_bf16.astype(jnp.float32)),
    )
    assert int(got_polls[0]) == 2 + 2 * shared_down_16k.SHARED_DOWN_NUM_PANELS_16K

    bad_bf16 = ref_h0.astype(jnp.bfloat16)
    assert (
        float(
            jnp.max(
                jnp.abs(got_shared.astype(jnp.float32) - bad_bf16.astype(jnp.float32))
            )
        )
        > 1.0
    )


class _TraceRecorder:
    def __init__(self):
        self.events: list[tuple[str, frozenset[str]]] = []
        self._scopes: list[frozenset[str]] = []

    @contextlib.contextmanager
    def scope(self, names: frozenset[str]):
        self._scopes.append(names)
        try:
            yield
        finally:
            self._scopes.pop()

    def record(self, event: str) -> None:
        self.events.append((event, self._scopes[-1] if self._scopes else frozenset()))


def _install_recorder(monkeypatch) -> _TraceRecorder:
    recorder = _TraceRecorder()
    run_scoped, signal, wait = pl.run_scoped, pl.semaphore_signal, pl.semaphore_wait
    remote_copy = pltpu.make_async_remote_copy

    def recording_run_scoped(body, *types, **kw_types):
        names = set(kw_types)
        for value in types:
            if isinstance(value, dict):
                names |= set(value)

        def scoped_body(*args, **kwargs):
            with recorder.scope(frozenset(names)):
                return body(*args, **kwargs)

        return run_scoped(scoped_body, *types, **kw_types)

    def recording_signal(*args, **kwargs):
        if kwargs.get("device_id") is not None:
            recorder.record("remote_signal")
        return signal(*args, **kwargs)

    def recording_wait(*args, **kwargs):
        recorder.record("wait")
        return wait(*args, **kwargs)

    def recording_remote_copy(*args, **kwargs):
        recorder.record("remote_copy")
        return remote_copy(*args, **kwargs)

    monkeypatch.setattr(pl, "run_scoped", recording_run_scoped)
    monkeypatch.setattr(pl, "semaphore_signal", recording_signal)
    monkeypatch.setattr(pl, "semaphore_wait", recording_wait)
    monkeypatch.setattr(pltpu, "make_async_remote_copy", recording_remote_copy)
    return recorder


def test_should_handshake_chip_partner_before_remote_writes_in_deferred_shared_down(
    monkeypatch,
):
    recorder = _install_recorder(monkeypatch)
    rows = 512
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def kernel(_shared_in_hbm, shared_down_hbm, shared_hbm):
        def _body(s) -> None:
            rank = jax.lax.axis_index("k3_sp")
            shared_down_16k.run_deferred_shared_down_from_hbm(
                shared_hbm,
                shared_down_hbm,
                s["down_ready"],
                rows=rows,
                parity=rank % 2,
                peer=rank ^ 1,
            )

        pl.run_scoped(_body, dict(down_ready=pltpu.SemaphoreType.REGULAR))

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm, hbm),
            out_specs=hbm,
        ),
        input_output_aliases={0: 0},
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    replicated = jax.sharding.PartitionSpec()
    traced = jax.shard_map(
        call,
        mesh=mesh,
        in_specs=(replicated, replicated),
        out_specs=replicated,
        check_vma=False,
    )
    jax.make_jaxpr(traced)(
        jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
        jax.ShapeDtypeStruct((3072, 7168), jnp.bfloat16),
    )

    down_events = [event for event, scope in recorder.events if "peer_y" in scope]
    assert "remote_copy" in down_events
    assert down_events[:2] == ["remote_signal", "wait"]


def test_should_skip_prologue_middle_and_peer_y_when_defer_shared_down_enabled(
    monkeypatch,
):
    monkeypatch.setattr(
        k3_config, "active", lambda: k3_config.KernelConfig(prefill_expert_parallel=16)
    )
    recorder = _install_recorder(monkeypatch)
    seen_allocs: list[dict[str, object]] = []
    orig_run_scoped = pl.run_scoped

    def spy_run_scoped(body, *types, **kw_types):
        for value in types:
            if isinstance(value, dict):
                seen_allocs.append(value)
        return orig_run_scoped(body, *types, **kw_types)

    monkeypatch.setattr(pl, "run_scoped", spy_run_scoped)

    tokens, rows = 16384, 512
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)

    def kernel(
        tables,
        control,
        hidden,
        router,
        bias,
        latent_down,
        shared_gu,
        shared_down,
        latent_out,
        ids_v,
        probs_v,
        counts_v,
        shared_hbm,
    ):
        fused._prologue(  # noqa: SLF001 -- defer_shared_down scope contract is only observable via traced _prologue
            tables,
            control,
            hidden,
            router,
            bias,
            latent_down,
            shared_gu,
            shared_down,
            shared_hbm,
            latent_out,
            ids_v,
            probs_v,
            counts_v,
            None,
            counts_ready=lambda: None,
            metadata_ready=lambda: None,
            shared_in_hbm=True,
            defer_shared_down=True,
        )

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, 2, 1792), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens // 8, 128), jnp.int32),
            jax.ShapeDtypeStruct((tokens // 8, 128), jnp.float32),
            jax.ShapeDtypeStruct((56,), jnp.int32),
            jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * 6,
            out_specs=(hbm, vmem, vmem, vmem, hbm),
        ),
    )
    operands = (
        jax.ShapeDtypeStruct((3, 3, 32), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 7168, 896), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 1, 896), jnp.float32),
        jax.ShapeDtypeStruct((2, 7168, 3584), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 7168, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 3072, 7168), jnp.bfloat16),
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    replicated = jax.sharding.PartitionSpec()
    traced = jax.shard_map(
        call,
        mesh=mesh,
        in_specs=(replicated,) * len(operands),
        out_specs=(replicated,) * 5,
        check_vma=False,
    )
    jax.make_jaxpr(traced)(*operands)

    assert not any("peer_y" in scope for _, scope in recorder.events)
    outer_scratch = [d for d in seen_allocs if "middle" in d]
    assert len(outer_scratch) == 1 and outer_scratch[0]["middle"] is None
    assert outer_scratch[0]["down_ready"] is None
    front_scratch = [d for d in seen_allocs if "mid_stage" in d]
    assert len(front_scratch) == 1 and len(front_scratch[0]["mid_stage"]) == 2


def test_should_preserve_16k_vmem_headroom_and_zero_extra_hbm():
    rows = routed_16k.TOKENS_16K // 32
    assert shared_down_16k.SHARED_STAGED_MIDDLE_COLS_16K == 6144
    assert (
        shared_down_16k.SHARED_STAGED_MIDDLE_COLS_16K
        <= shared_down_16k.SHARED_HIDDEN_16K
    )
    assert shared_down_16k.SHARED_INTER_16K % 128 == 0
    assert shared_down_16k.SHARED_DOWN_PANEL_COLS_16K % 128 == 0

    mib = 1024 * 1024
    middle_reload_bytes = (2 * rows) * shared_down_16k.SHARED_INTER_16K * 2
    panel_cols = shared_down_16k.SHARED_DOWN_PANEL_COLS_16K
    shared_down_scope_bytes = (
        2 * shared_down_16k.SHARED_INTER_16K * panel_cols * 2
        + 2 * (2 * rows) * panel_cols * 4
        + 2 * rows * panel_cols * 4
        + 2 * rows * panel_cols * 2
    )
    assert middle_reload_bytes == 6 * mib
    assert shared_down_scope_bytes == 13 * mib
    outer_ring_last_bytes = int(33.02 * mib)
    assert (
        outer_ring_last_bytes + middle_reload_bytes + shared_down_scope_bytes < 53 * mib
    )


@pytest.mark.parametrize(
    ("shared_shape", "down_shape"),
    [
        ((512, 7168), None),
        (None, (3072, 7168)),
        ((256, 7168), (3072, 7168)),
        ((512, 7168), (3072, 3584)),
    ],
)
def test_should_reject_malformed_deferred_shared_args_in_routed_16k(
    shared_shape, down_shape
):
    latent = jax.ShapeDtypeStruct((16384, 2, 1792), jnp.bfloat16)
    gu_weights = jax.ShapeDtypeStruct((1, 56, 448, 3072), jnp.uint32)
    gu_scales = jax.ShapeDtypeStruct((1, 56, 112, 3072), jnp.uint8)
    rhs = jax.ShapeDtypeStruct((1, 56, 192, 3584), jnp.uint32)
    scales = jax.ShapeDtypeStruct((1, 56, 48, 3584), jnp.uint8)
    counts = jax.ShapeDtypeStruct((56,), jnp.int32)
    middle = jax.ShapeDtypeStruct((98304, 1536), jnp.bfloat16)
    output = jax.ShapeDtypeStruct((512, 3584), jnp.bfloat16)
    subtotal = jax.ShapeDtypeStruct((16384, 1536), jnp.bfloat16)

    with pytest.raises(AssertionError):
        routed_16k.routed_experts_16k(
            latent,
            gu_weights,
            gu_scales,
            rhs,
            scales,
            counts,
            None,
            None,
            None,
            None,
            None,
            middle,
            output,
            probabilities_ready=lambda: None,
            subtotal_hbm=subtotal,
            compact_capacity=98304,
            rmw_batch=128,
            rmw_dummy_slots=16,
            deferred_shared_hbm=(
                jax.ShapeDtypeStruct(shared_shape, jnp.bfloat16)
                if shared_shape is not None
                else None
            ),
            deferred_shared_down_hbm=(
                jax.ShapeDtypeStruct(down_shape, jnp.bfloat16)
                if down_shape is not None
                else None
            ),
        )


@pytest.mark.parametrize(("half", "panel"), [(2, 0), (-1, 0), (1, 12)])
def test_should_reject_out_of_bounds_stage_middle_half_copy(half, panel):
    dummy = jax.ShapeDtypeStruct((512, 256), jnp.bfloat16)
    with pytest.raises(AssertionError):
        shared_down_16k.stage_middle_half_copy(
            dummy, dummy, dummy, half=half, panel=panel
        )


def test_should_invoke_on_panels_done_after_14p_latent_panels_and_before_quant():
    rng = np.random.default_rng(808)
    rows, hidden_size = 128, 256
    own_np = rng.normal(0.0, 0.15, size=(rows, hidden_size)).astype(np.float32)
    own_np[0] *= 25.0
    own_np[1] *= 1e-3
    own_np[2] = 0.0
    w_np = rng.normal(0.0, 0.08, size=(hidden_size, 3584)).astype(np.float32)
    own_bf16 = jnp.asarray(own_np, dtype=jnp.bfloat16)
    w_bf16 = jnp.asarray(w_np, dtype=jnp.bfloat16)

    expected_f32 = jnp.matmul(own_bf16, w_bf16, preferred_element_type=jnp.float32)
    expected_q, expected_s = w4a8_16k.quantize_latent_rows_e4m3_ref(expected_f32)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def body(own_hbm, w_hbm, q_out, s_out, cb_count_ref, pre_quant_scale_ref):
        cb_count_ref[0] = 0

        def inner(s):
            c = pltpu.make_async_copy(own_hbm, s["own"], s["in_sem"])
            c.start()
            c.wait()
            s["lat_s"][...] = jnp.full((rows // 128, 128), -999.0, jnp.float32)

            def _on_panels_done() -> None:
                cb_count_ref[0] = cb_count_ref[0] + 1
                pre_quant_scale_ref[0] = s["lat_s"][0, 0]

            w4a8_16k.compute_dynamic_latent_e4m3(
                s["own"],
                w_hbm,
                s["bufs"],
                s["sems"],
                s["lat_q"],
                s["lat_s"],
                prestarted=False,
                on_panels_done=_on_panels_done,
            )
            cq = pltpu.make_async_copy(s["lat_q"], q_out, s["out_sems"][0])
            cs = pltpu.make_async_copy(s["lat_s"], s_out, s["out_sems"][1])
            cq.start()
            cs.start()
            cq.wait()
            cs.wait()

        pl.run_scoped(
            inner,
            dict(
                own=pltpu.VMEM((rows, hidden_size), jnp.bfloat16),
                bufs=tuple(
                    pltpu.VMEM((hidden_size, 256), jnp.bfloat16) for _ in range(4)
                ),
                sems=tuple(pltpu.SemaphoreType.DMA for _ in range(4)),
                in_sem=pltpu.SemaphoreType.DMA,
                out_sems=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
                lat_q=pltpu.VMEM(
                    (rows, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE), w4a8_16k.LATENT_E4M3_DTYPE
                ),
                lat_s=pltpu.VMEM((rows // 128, 128), jnp.float32),
            ),
        )

    call = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(
                (rows, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE), w4a8_16k.LATENT_E4M3_DTYPE
            ),
            jax.ShapeDtypeStruct((rows // 128, 128), jnp.float32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(hbm, hbm),
            out_specs=(
                hbm,
                hbm,
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ),
        ),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    got_q, got_s, got_cb_count, got_pre_quant_scale = call(own_bf16, w_bf16)

    assert int(got_cb_count[0]) == 1
    assert float(got_pre_quant_scale[0]) == -999.0
    np.testing.assert_allclose(
        np.asarray(got_s).reshape(-1), np.asarray(expected_s), rtol=1e-6, atol=0.0
    )
    np.testing.assert_array_equal(
        np.asarray(got_q.astype(jnp.float32)),
        np.asarray(expected_q.astype(jnp.float32)),
    )
