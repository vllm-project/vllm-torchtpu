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
import json
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.models.kimi_k3.megakernel import runner
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import (
    EP16_DEFAULT_EXPERT_PLAN,
    EP16_DEFAULT_REPLICA_PLAN,
    KernelConfig,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    decode_megakernel as dm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import rank_loader as rl
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import (
    metadata_compact,
)

LAYERS = 3
EXPERT_SHAPES = {
    "expert_gu": (2, 112, 4, 8),
    "expert_gus": (2, 112, 1, 8),
    "expert_down": (2, 112, 6, 4),
    "expert_ds": (2, 112, 2, 4),
}
LAYER_RANGES = ((0, 1), (1, 2))
DATA_DIR = Path(rl.__file__).resolve().parent.parent / "data"


def _pack_route_wire(ids_by_rank: np.ndarray, probs_by_rank: np.ndarray) -> np.ndarray:
    rows = ids_by_rank.shape[1]
    parts = []
    for r in range(32):
        parts.append(ids_by_rank[r].reshape(rows // 8, 128))
        parts.append(probs_by_rank[r].view(np.int32).reshape(rows // 8, 128))
    return np.concatenate(parts, axis=0)


def _reference_online_1pass(
    ids_by_rank: np.ndarray,
    probs_by_rank: np.ndarray,
    table_flat: np.ndarray,
    *,
    base_local_experts: int = 56,
    replica_slots: int = 4,
    min_gap: int | None = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    int,
    np.ndarray,
]:
    rows = ids_by_rank.shape[1]
    tokens = 32 * rows
    ep_groups = 896 // base_local_experts
    eff_le = base_local_experts + replica_slots
    n_rep = ep_groups * replica_slots
    assert table_flat.shape == (4 * n_rep,)
    he_arr = table_flat[:n_rep]
    hg_arr = table_flat[n_rep : 2 * n_rep]
    rg_arr = table_flat[2 * n_rep : 3 * n_rep]
    rs_arr = table_flat[3 * n_rep : 4 * n_rep]

    gap_floor = (
        metadata_compact.default_replica_min_gap(tokens) if min_gap is None else min_gap
    )
    rank_half = tokens // 256
    ranks_per_win = 128 // rank_half
    n_wins = (tokens // 8) // 128

    group_loads = np.zeros((ep_groups,), dtype=np.int64)
    assigned_toks: list[list[list[tuple[int, int, float]]]] = [
        [[] for _ in range(eff_le)] for _ in range(ep_groups)
    ]

    for win_idx in range(n_wins):
        r0 = win_idx * ranks_per_win
        r1 = r0 + ranks_per_win
        ids_win = ids_by_rank[r0:r1].reshape(-1)
        probs_win = probs_by_rank[r0:r1].reshape(-1)
        live = probs_win != 0
        ids_live = np.where(live, ids_win, -1)

        for grp in range(ep_groups):
            lo = grp * base_local_experts
            hi = lo + base_local_experts
            group_loads[grp] += int(np.sum((ids_live >= lo) & (ids_live < hi)))

        moves: dict[int, tuple[int, int]] = {}
        if int(group_loads.max() - group_loads.min()) >= gap_floor:
            for k in range(n_rep):
                hg = int(hg_arr[k])
                rg = int(rg_arr[k])
                gap = int(group_loads[hg] - group_loads[rg])
                if gap < gap_floor:
                    continue
                he = int(he_arr[k])
                if he < 0:
                    continue
                mv = int(np.sum(ids_live == he))
                if 0 < mv <= gap:
                    group_loads[hg] -= mv
                    group_loads[rg] += mv
                    moves[he] = (rg, int(rs_arr[k]))

        base_pos = win_idx * (128 * 128)
        for pos_in_win in range(ids_live.size):
            e = int(ids_live[pos_in_win])
            if e < 0:
                continue
            g_pos = base_pos + pos_in_win
            tok = g_pos // 16
            slot = g_pos % 16
            prob = float(probs_win[pos_in_win])
            if e in moves:
                dest_g, dest_slot = moves[e]
            else:
                dest_g, dest_slot = divmod(e, base_local_experts)
            assigned_toks[dest_g][dest_slot].append((tok, slot, prob))

    tids_by_group: list[np.ndarray] = []
    probs_by_group: list[np.ndarray] = []
    counts_by_group: list[np.ndarray] = []
    for g in range(ep_groups):
        counts = np.array([len(assigned_toks[g][e]) for e in range(eff_le)], np.int32)
        t_list: list[int] = []
        p_list: list[float] = []
        for e in range(eff_le):
            for tok, _slot, prob in assigned_toks[g][e]:
                t_list.append(tok)
                p_list.append(prob)
        tids_by_group.append(np.array(t_list, dtype=np.int32))
        probs_by_group.append(np.array(p_list, dtype=np.float32))
        counts_by_group.append(counts)
    return (
        tids_by_group,
        probs_by_group,
        counts_by_group,
        int(group_loads.max()),
        group_loads,
    )


@functools.lru_cache(maxsize=16)
def _compile_replica_schedule_interpret(
    tokens: int,
    base_local_experts: int,
    replica_slots: int,
    min_gap: int | None,
):
    total_rows = tokens // 8
    ep_groups = 896 // base_local_experts
    eff_le = base_local_experts + replica_slots
    max_drops = min(base_local_experts, ep_groups * replica_slots)
    counts_ready_calls = [0]

    def kernel(
        wire_ref,
        rank_ref,
        table_ref,
        keys_ref,
        probs_ref,
        tids_ref,
        counts_ref,
        max_ref,
        cursor_ref,
        rep_loads_ref,
        rep_drop_ref,
        rep_add_e_ref,
        rep_add_slot_ref,
        rep_cnts_ref,
        tail_stages_ref,
    ):
        tail_stages_ref[0] = 0

        def _on_counts_ready() -> None:
            counts_ready_calls[0] += 1

        def _on_tail_stage() -> None:
            tail_stages_ref[0] = tail_stages_ref[0] + 1

        scratch = metadata_compact.ReplicaRoutingScratch(
            group_loads=rep_loads_ref,
            my_drop_experts=rep_drop_ref,
            my_add_experts=rep_add_e_ref,
            my_add_slots=rep_add_slot_ref,
            my_counts=rep_cnts_ref,
        )
        schedule = metadata_compact.build_compact_metadata_schedule(
            wire_ref,
            keys_ref,
            probs_ref,
            tids_ref,
            counts_ref,
            max_ref,
            cursor_ref,
            tokens=tokens,
            rank=rank_ref[0],
            counts_ready=_on_counts_ready,
            local_experts=eff_le,
            on_tail_stage=_on_tail_stage,
            replica_table=table_ref,
            replica_slots=replica_slots,
            replica_scratch=scratch,
            replica_min_gap=min_gap,
        )
        for step_fn in schedule.steps:
            step_fn()
        schedule.tail()

    out_shapes = (
        jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        jax.ShapeDtypeStruct((total_rows, 128), jnp.float32),
        jax.ShapeDtypeStruct((total_rows, 128), jnp.int32),
        jax.ShapeDtypeStruct((eff_le,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((ep_groups,), jnp.int32),
        jax.ShapeDtypeStruct((max_drops,), jnp.int32),
        jax.ShapeDtypeStruct((replica_slots,), jnp.int32),
        jax.ShapeDtypeStruct((replica_slots,), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    fn = jax.jit(pl.pallas_call(kernel, out_shape=out_shapes, interpret=True))
    return fn, counts_ready_calls


def _run_replica_schedule_interpret(
    wire_np: np.ndarray,
    table_flat: np.ndarray,
    *,
    tokens: int,
    rank: int,
    base_local_experts: int = 56,
    replica_slots: int = 4,
    min_gap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    fn, _ = _compile_replica_schedule_interpret(
        tokens, base_local_experts, replica_slots, min_gap
    )
    out = fn(
        jnp.asarray(wire_np),
        jnp.asarray([rank], dtype=jnp.int32),
        jnp.asarray(table_flat, dtype=jnp.int32),
    )
    probs_out = np.asarray(out[1])
    tids_out = np.asarray(out[2])
    counts_out = np.asarray(out[3])
    max_out = int(np.asarray(out[4])[0])
    tail_stages = int(np.asarray(out[11])[0])
    return tids_out, probs_out, counts_out, max_out, tail_stages


@pytest.fixture(scope="module")
def preshard_32(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("preshard32_prefill_rep")
    expert_raw: dict[tuple[int, str, int], np.ndarray] = {}
    router_raw = np.arange(2 * 4 * 896, dtype=np.float32).reshape(2, 4, 896) + 1.25
    router_bias_raw = np.arange(2 * 1 * 896, dtype=np.float32).reshape(2, 1, 896) - 3.5

    for rank in range(32):
        rank_dir = root / f"rank{rank:02d}"
        rank_dir.mkdir()
        families: dict[str, object] = {}
        dense_shapes: dict[str, tuple[int, ...]] = {
            name: (2, 3) for name in rl.PREFILL_IO + rl.PREFILL_COMMON
        }
        dense_shapes.update({name: (2, 3) for name in rl.PREFILL_KDA + rl.PREFILL_MLA})
        dense_shapes["latent_norm"] = (2, 1, 4096)
        dense_shapes["router"] = (2, 4, 896)
        dense_shapes["router_bias"] = (2, 1, 896)
        dense_shapes["k_o"] = (2, 768, 3)
        dense_shapes["m_o"] = (2, 768, 3)

        blobs = []
        offset = 0
        for name, shape in dense_shapes.items():
            if name == "router":
                arr = router_raw.copy()
            elif name == "router_bias":
                arr = router_bias_raw.copy()
            elif name == "latent_norm":
                arr = np.full(shape, 7.5 + rank, dtype=np.float32)
            else:
                arr = np.full(shape, float(rank + 1), dtype=np.float32)
            families[name] = {
                "file": "dense.bin",
                "offset": offset,
                "shape": list(shape),
                "dtype": "float32",
            }
            blobs.append(arr.tobytes())
            offset += arr.nbytes
        (rank_dir / "dense.bin").write_bytes(b"".join(blobs))

        for name, shape in EXPERT_SHAPES.items():
            files = []
            for part_idx, (lo, hi) in enumerate(LAYER_RANGES):
                part_shape = (hi - lo, *shape[1:])
                base = (
                    (rank + 1) * 1_000_000
                    + (part_idx + 1) * 100_000
                    + np.arange(int(np.prod(part_shape)), dtype=np.int32).reshape(
                        part_shape
                    )
                ).astype(np.float32)
                expert_raw[(rank, name, lo)] = base
                filename = f"{name}_{part_idx}.bin"
                base.tofile(rank_dir / filename)
                files.append(filename)
            families[name] = {
                "shape": list(shape),
                "dtype": "float32",
                "files": files,
                "layer_ranges": [list(r) for r in LAYER_RANGES],
            }
        (rank_dir / "index.json").write_text(
            json.dumps({"rank": rank, "layers": LAYERS, "families": families})
        )

    cache = root / "prefill-sp-cache"
    cache.mkdir()
    cache_families = {}
    for name in (
        "latent_down",
        "latent_up",
        "shared_gu_0",
        "shared_gu_1",
        "shared_down_0",
        "shared_down_1",
    ):
        np.full((2, 2), 0.5, dtype=np.float32).tofile(cache / f"{name}.bin")
        cache_families[name] = {
            "file": f"{name}.bin",
            "shape": [2, 2],
            "dtype": "float32",
        }
    (cache / "layout.json").write_text(
        json.dumps({"layers": LAYERS, "families": cache_families})
    )

    rng = np.random.default_rng(20261113)
    order = np.stack([rng.permutation(896), rng.permutation(896)], axis=0).astype(
        np.int64
    )
    plan_path = root / "ep16_plan.npz"
    np.savez(plan_path, order=order)
    return {
        "root": root,
        "cache": cache,
        "expert_raw": expert_raw,
        "order": order,
        "plan_path": str(plan_path),
    }


def _make_noncolliding_replica_npz(
    path: Path, order: np.ndarray, *, slots: int = 2, leave_empty: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    layers = order.shape[0]
    experts = np.full((layers, 16, slots), -1, dtype=np.int32)
    for layer in range(layers):
        for hg in range(16):
            for s in range(slots):
                if leave_empty and hg == 15 and s == slots - 1:
                    continue
                rg = (hg + 2 + 3 * s) % 16
                phys = hg * 56 + s
                experts[layer, rg, s] = int(order[layer, phys])
    disp = np.stack(
        [np.arange(16 * slots, dtype=np.int32)[::-1] for _ in range(layers)],
        axis=0,
    )
    np.savez(path, experts=experts, order=disp)
    return experts, disp


def test_should_ship_balanced_ep16_plan_and_zero_collision_replica_plan() -> None:
    ep8_order = np.load(DATA_DIR / "plan_ep8_real_hotfirst.npz")["order"]
    ep16_order = np.load(DATA_DIR / EP16_DEFAULT_EXPERT_PLAN)["order"]
    rep = np.load(DATA_DIR / EP16_DEFAULT_REPLICA_PLAN)
    rep_experts = rep["experts"]
    rep_order = rep["order"]

    assert ep16_order.shape == (92, 896)
    assert rep_experts.shape == (92, 16, 4)
    assert rep_order.shape == (92, 64)

    ranks = np.arange(112, dtype=np.float64)
    weight_curves = (
        ("zipf_0.6", (ranks + 1.0) ** (-0.60), 1.005),
        ("shifted_zipf_1.2", (ranks + 2.0) ** (-1.20), 1.010),
        ("exp_10", np.exp(-ranks / 10.0), 1.020),
        ("exp_20", np.exp(-ranks / 20.0), 1.005),
        ("exp_40", np.exp(-ranks / 40.0), 1.005),
        ("linear_80", np.maximum(0.0, 1.0 - ranks / 80.0), 1.005),
    )
    even_totals = {name: 0.0 for name, _, _ in weight_curves}
    odd_totals = {name: 0.0 for name, _, _ in weight_curves}

    for layer in range(92):
        pos_in_block = {
            int(ep8_order[layer, g * 112 + k]): k for g in range(8) for k in range(112)
        }
        for g in range(8):
            even_ranks = [
                pos_in_block[int(e)]
                for e in ep16_order[layer, 2 * g * 56 : (2 * g + 1) * 56]
            ]
            odd_ranks = [
                pos_in_block[int(e)]
                for e in ep16_order[layer, (2 * g + 1) * 56 : (2 * g + 2) * 56]
            ]
            assert even_ranks == sorted(even_ranks)
            assert odd_ranks == sorted(odd_ranks)
            assert sorted(even_ranks + odd_ranks) == list(range(112))
            assert sum(even_ranks) == sum(odd_ranks)
            for octave in range(14):
                lo, hi = 8 * octave, 8 * (octave + 1)
                assert sum(lo <= r < hi for r in even_ranks) == 4
            for name, weights, _ in weight_curves:
                even_totals[name] += float(weights[even_ranks].sum())
                odd_totals[name] += float(weights[odd_ranks].sum())

    for name, _, max_mm in weight_curves:
        mean = 0.5 * (even_totals[name] + odd_totals[name])
        assert max(even_totals[name], odd_totals[name]) / mean <= max_mm, name

    for layer in range(92):
        assert sorted(int(x) for x in rep_order[layer]) == list(range(64))
        inv16 = np.argsort(ep16_order[layer])
        inv8 = np.argsort(ep8_order[layer])
        seen_experts: set[int] = set()
        dest_by_home: dict[int, set[int]] = {hg: set() for hg in range(16)}
        for rg in range(16):
            for s in range(4):
                le = int(rep_experts[layer, rg, s])
                assert 0 <= le < 896
                seen_experts.add(le)
                hg16 = int(inv16[le]) // 56
                hg8 = int(inv8[le]) // 112
                assert hg16 != rg
                assert hg16 // 2 != rg // 2
                assert hg8 != rg // 2
                assert (hg16 % 2) != (rg % 2)
                dest_by_home[hg16].add(rg)
        assert len(seen_experts) == 64
        assert all(len(dests) == 4 for dests in dest_by_home.values())


def test_should_normalize_and_validate_prefill_replica_plan_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("K3_PREFILL_REPLICA_PLAN", raising=False)
    assert (
        envs.environment_variables["K3_PREFILL_REPLICA_PLAN"]()
        == EP16_DEFAULT_REPLICA_PLAN
    )
    monkeypatch.setenv("K3_PREFILL_REPLICA_PLAN", "")
    assert KernelConfig().prefill_replica_plan == ""

    for disabled in ("0", "none", "NONE", "off", "false"):
        cfg = KernelConfig(
            prefill_expert_parallel=16,
            kd_replica_plan=disabled,
            prefill_replica_plan=disabled,
        )
        assert cfg.kd_replica_plan == ""
        assert cfg.prefill_replica_plan == ""

    cfg_prefill_only = KernelConfig(
        prefill_expert_parallel=16,
        decode_expert_parallel=16,
        kd_replica_plan="none",
        prefill_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
    )
    assert cfg_prefill_only.kd_replica_plan == ""
    assert cfg_prefill_only.prefill_replica_plan == EP16_DEFAULT_REPLICA_PLAN

    with pytest.raises(ValueError, match="prefill_expert_parallel=16"):
        KernelConfig(
            prefill_expert_parallel=8,
            prefill_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
        )
    with pytest.raises(ValueError, match="prefill_fused_moe=True"):
        KernelConfig(
            prefill_expert_parallel=16,
            prefill_fused_moe=False,
            prefill_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
        )
    with pytest.raises(ValueError, match="exclusive"):
        KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=8,
            kd_hot_plan="hot.npz",
            prefill_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
        )
    with pytest.raises(ValueError, match="must match"):
        KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            kd_replica_plan="plan_a.npz",
            prefill_replica_plan="plan_b.npz",
        )


def test_should_keep_decode_replicas_off_when_runner_sets_prefill_replica_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[KernelConfig] = []
    monkeypatch.setattr(k3_config, "install", installed.append)
    monkeypatch.setattr(
        runner.k3,
        "build",
        lambda cfg, presharded, draft, *, tp_rank: ("built", cfg.role, tp_rank),
    )
    monkeypatch.setenv("K3_UNIFIED_EP16", "1")
    monkeypatch.setenv("K3_REPLICA_PLAN", "none")
    monkeypatch.setenv("K3_PREFILL_REPLICA_PLAN", EP16_DEFAULT_REPLICA_PLAN)

    fake_vllm_cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["KimiK3ForConditionalGeneration"]
            ),
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=93,
                hidden_size=7168,
            ),
            max_model_len=32768,
        ),
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=32,
            decode_context_parallel_size=32,
            cp_kv_cache_interleave_size=1,
        ),
        cache_config=types.SimpleNamespace(
            block_size=512,
            cache_dtype="fp8",
            mamba_ssm_cache_dtype="auto",
        ),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=16384,
            async_scheduling=True,
        ),
        load_config=types.SimpleNamespace(
            load_format="megakernel",
            model_loader_extra_config={"presharded": "/tmp/unused-preshard"},
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )

    out = runner.build_program(fake_vllm_cfg, tp_rank=0)
    assert out == ("built", "aggregate", 0)
    assert len(installed) == 1
    assert installed[0].decode_expert_parallel == 16
    assert installed[0].prefill_expert_parallel == 16
    assert installed[0].kd_replica_plan == ""
    assert installed[0].prefill_replica_plan == EP16_DEFAULT_REPLICA_PLAN


def test_should_build_prefill_replica_table_and_reject_self_group_collision(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
    tmp_path: Path,
) -> None:
    order = preshard_32["order"]
    assert isinstance(order, np.ndarray)
    rep_path = tmp_path / "prefill_rep_valid.npz"
    rep_experts, rep_order = _make_noncolliding_replica_npz(
        rep_path, order, slots=2, leave_empty=True
    )
    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=str(preshard_32["plan_path"]),
            kd_replica_plan="",
            prefill_replica_plan=str(rep_path),
        ),
    )

    assert rl.prefill_replica_slots() == 2
    table = rl.prefill_replica_table(LAYERS, expert_order=order)
    assert table is not None
    assert table.shape == (LAYERS - 1, 4 * 16 * 2)
    assert table.dtype == np.int32
    inv = np.argsort(order, axis=1)
    n_rep = 32
    for layer in range(LAYERS - 1):
        for k_idx, flat in enumerate(rep_order[layer]):
            rg, slot = divmod(int(flat), 2)
            logical_e = int(rep_experts[layer, rg, slot])
            if logical_e < 0:
                assert int(table[layer, k_idx]) == -1
                assert int(table[layer, n_rep + k_idx]) == 0
                assert int(table[layer, 2 * n_rep + k_idx]) == 0
                assert int(table[layer, 3 * n_rep + k_idx]) == 56
            else:
                phys_e = int(inv[layer, logical_e])
                assert int(table[layer, k_idx]) == phys_e
                assert int(table[layer, n_rep + k_idx]) == phys_e // 56
                assert int(table[layer, 2 * n_rep + k_idx]) == rg
                assert int(table[layer, 3 * n_rep + k_idx]) == 56 + slot

    bad_experts = rep_experts.copy()
    bad_experts[0, 3, 0] = int(order[0, 3 * 56 + 10])
    bad_path = tmp_path / "prefill_rep_self_collide.npz"
    np.savez(bad_path, experts=bad_experts, order=rep_order)
    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            prefill_replica_plan=str(bad_path),
        ),
    )
    with pytest.raises(ValueError, match="replicated onto its own home group"):
        rl.prefill_replica_table(LAYERS, expert_order=order)


@pytest.mark.parametrize("rank", [0, 3, 31])
def test_should_extend_shared_experts_without_touching_latent_norm_when_prefill_only(
    monkeypatch: pytest.MonkeyPatch,
    preshard_32: dict[str, object],
    tmp_path: Path,
    rank: int,
) -> None:
    root = Path(str(preshard_32["root"]))
    cache = Path(str(preshard_32["cache"]))
    order = preshard_32["order"]
    assert isinstance(order, np.ndarray)
    rep_path = tmp_path / f"prefill_only_r{rank}.npz"
    _make_noncolliding_replica_npz(rep_path, order, slots=2, leave_empty=True)

    monkeypatch.setattr(
        k3_config,
        "active",
        lambda: KernelConfig(
            prefill_expert_parallel=16,
            decode_expert_parallel=16,
            prefill_expert_plan=str(preshard_32["plan_path"]),
            kd_replica_plan="",
            prefill_replica_plan=str(rep_path),
        ),
    )

    decode_items = {
        name: arr
        for _, name, arr in rl.iter_decode_rank(
            root, rank, layers=LAYERS, split_experts=False
        )
    }
    np.testing.assert_array_equal(
        decode_items["latent_norm"],
        np.full((LAYERS - 1, 1, 4096), 7.5 + rank, dtype=np.float32),
    )
    standalone_prefill = {
        name: arr
        for _, name, arr in rl.iter_prefill_rank(
            root, cache, rank, layers=LAYERS, split_experts=False
        )
    }
    aggregate_prefill = {
        name: arr
        for _, name, arr in rl.iter_prefill_rank(
            root,
            cache,
            rank,
            layers=LAYERS,
            split_experts=False,
            skip=frozenset(rl.PREFILL_SHARED),
        )
    }

    expected_table = rl.prefill_replica_table(LAYERS, expert_order=order)
    assert expected_table is not None
    np.testing.assert_array_equal(
        standalone_prefill["prefill_replica_table"], expected_table
    )
    np.testing.assert_array_equal(
        aggregate_prefill["prefill_replica_table"], expected_table
    )
    for name in EXPERT_SHAPES:
        assert decode_items[name].shape[1] == 58
        assert standalone_prefill[name].shape[1] == 58
        np.testing.assert_array_equal(decode_items[name], standalone_prefill[name])
        assert name not in aggregate_prefill


@pytest.mark.parametrize(
    ("tokens", "regime", "groups"),
    [
        (8192, "skew_g0", tuple(range(16))),
        (8192, "balanced_skip", (0, 2, 15)),
        (8192, "partial_pad", (0, 2, 15)),
        (16384, "skew_g0", (0, 2, 5, 15)),
    ],
)
@pytest.mark.tpu_test
def test_should_match_online_1pass_oracle_bitwise_across_all_ep16_groups(
    tokens: int,
    regime: str,
    groups: tuple[int, ...],
) -> None:
    rows = tokens // 32
    rng = np.random.default_rng(20261113 + tokens + len(regime))
    ids_by_rank = np.empty((32, rows, 16), dtype=np.int32)
    probs_by_rank = rng.uniform(0.02, 0.25, size=(32, rows, 16)).astype(np.float32)

    if regime == "skew_g0":
        for r in range(32):
            for t in range(rows):
                if r < 18:
                    hot = np.array([0, 1, 2, 3], dtype=np.int32)
                    rest = rng.choice(np.arange(4, 896), size=12, replace=False)
                    ids_by_rank[r, t] = np.concatenate([hot, rest])
                else:
                    ids_by_rank[r, t] = rng.choice(896, size=16, replace=False)
    elif regime == "balanced_skip":
        base = np.arange(16, dtype=np.int32) * 56
        for r in range(32):
            for t in range(rows):
                ids_by_rank[r, t] = base + ((r * rows + t) % 56)
    else:
        for r in range(32):
            for t in range(rows):
                ids_by_rank[r, t] = rng.choice(896, size=16, replace=False)
        probs_by_rank[28:, :, :] = 0.0
        ids_by_rank[28:, :, :] = 0

    n_rep = 64
    table = np.zeros((4 * n_rep,), dtype=np.int32)
    for k in range(n_rep):
        hg = k % 16
        s = k // 16
        rg = (hg + 2 + 3 * s) % 16
        if k in (17, 63):
            table[k] = -1
            table[n_rep + k] = 0
            table[2 * n_rep + k] = 0
            table[3 * n_rep + k] = 56 + s
        else:
            table[k] = hg * 56 + s
            table[n_rep + k] = hg
            table[2 * n_rep + k] = rg
            table[3 * n_rep + k] = 56 + s

    wire_np = _pack_route_wire(ids_by_rank, probs_by_rank)
    ref_tids, ref_probs, ref_counts, ref_max, ref_loads = _reference_online_1pass(
        ids_by_rank, probs_by_rank, table
    )

    total_live_expected = int(np.sum(probs_by_rank != 0))
    assert int(ref_loads.sum()) == total_live_expected
    if regime == "skew_g0":
        pre_g0 = int(np.sum((probs_by_rank != 0) & (ids_by_rank < 56)))
        assert ref_max < pre_g0

    total_checked_routes = 0
    for grp in groups:
        rank = 2 * grp + (grp % 2)
        tids_out, probs_out, counts_out, max_out, _ = _run_replica_schedule_interpret(
            wire_np, table, tokens=tokens, rank=rank
        )
        np.testing.assert_array_equal(counts_out, ref_counts[grp])
        assert max_out == ref_max
        n_grp = int(counts_out.sum())
        total_checked_routes += n_grp
        np.testing.assert_array_equal(tids_out.reshape(-1)[:n_grp], ref_tids[grp])
        np.testing.assert_array_equal(probs_out.reshape(-1)[:n_grp], ref_probs[grp])
        if regime == "balanced_skip":
            assert int(counts_out[56:].sum()) == 0

    if len(groups) == 16:
        assert total_checked_routes == total_live_expected


def test_should_respect_exact_min_gap_boundary_and_reject_invalid_replica_args() -> (
    None
):
    tokens = 8192
    rows = tokens // 32
    probs_by_rank = np.full((32, rows, 16), 0.125, dtype=np.float32)

    base = np.arange(16, dtype=np.int32) * 56 + 10
    excess = 128

    ids_blocked = np.broadcast_to(base[None, None, :], (32, rows, 16)).copy()
    ids_blocked[0, : excess - 1, 2] = 0
    ids_blocked[0, 0, 3] = 0

    ids_moved = np.broadcast_to(base[None, None, :], (32, rows, 16)).copy()
    ids_moved[0, :excess, 2] = 0

    n_rep = 64
    table = np.zeros((4 * n_rep,), dtype=np.int32)
    table[:n_rep] = -1
    table[3 * n_rep :] = 56
    table[0] = 0
    table[n_rep + 0] = 0
    table[2 * n_rep + 0] = 2
    table[3 * n_rep + 0] = 56

    _, _, cnt_blocked, max_blocked, _ = _run_replica_schedule_interpret(
        _pack_route_wire(ids_blocked, probs_by_rank), table, tokens=tokens, rank=4
    )
    assert int(cnt_blocked[56]) == 0
    assert max_blocked == tokens + excess

    _, _, cnt_moved, max_moved, _ = _run_replica_schedule_interpret(
        _pack_route_wire(ids_moved, probs_by_rank), table, tokens=tokens, rank=4
    )
    assert int(cnt_moved[56]) == excess
    assert max_moved == tokens

    with pytest.raises(ValueError, match="replica_slots > 0"):
        metadata_compact.build_compact_metadata_schedule(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            tokens=8192,
            rank=0,
            counts_ready=lambda: None,
            local_experts=56,
            replica_table=object(),
            replica_slots=0,
            replica_scratch=None,
        )


def test_should_keep_decode_ep16_expert_layout_untouched_when_prefill_only() -> None:
    cfg = KernelConfig(
        prefill_expert_parallel=16,
        decode_expert_parallel=16,
        kd_replica_plan="",
        prefill_replica_plan=EP16_DEFAULT_REPLICA_PLAN,
    )
    assert cfg.kd_replica_plan == ""
    assert dm.expert_slot_shape("expert_gu", (56, 448, 3072), halves=2) == (448, 1536)
    assert dm.expert_slot_shape("expert_gu", (60, 448, 3072), halves=2) == (448, 1536)


@pytest.mark.parametrize(
    ("plan", "expect_table"),
    [("", False), (EP16_DEFAULT_REPLICA_PLAN, True)],
)
def test_should_register_prefill_replica_table_name_only_when_plan_enabled(
    monkeypatch: pytest.MonkeyPatch,
    plan: str,
    expect_table: bool,
) -> None:
    cfg = KernelConfig(
        prefill_expert_parallel=16,
        decode_expert_parallel=16,
        kd_replica_plan="",
        prefill_replica_plan=plan,
    )
    monkeypatch.setattr(k3_config, "active", lambda: cfg)
    monkeypatch.setattr(k3_prefill, "active", lambda: cfg)
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 1))

    prog = k3_prefill.PrefillProgram(windows=(8192,), capacity=8192, slots=1)
    assert ("prefill_replica_table" in prog.names["moe"]) is expect_table
