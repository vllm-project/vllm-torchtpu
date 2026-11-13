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
"""Structured launch flag and solve/v2 payload emission for winning serving plans."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, TypedDict, cast

from .chips import CHIPS, JAX_VERSION, ChipSpec, chip_catalog
from .constants import (
    DEEP_PP_THRESHOLD,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_TOP_K,
    KV_DTYPE_BYTES,
    MIN_DIVERSE_TOP_LAYOUTS,
    MIN_TOP_LAYOUTS_FOR_BOUND_BEST,
    MIN_TOP_LAYOUTS_FOR_FASTEST,
    PERCENT_SCALE,
    SOLVE_SCHEMA,
    TOPOLOGY_AXES,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .modules import Dtypes
    from .solver import ParetoPoint, PoolMeshPlan, ServingPlan

_POOL_AXES: tuple[tuple[str, str], ...] = (
    ("TP", "tp"),
    ("DP", "dp"),
    ("EP", "ep"),
    ("PP", "pp"),
    ("PCP", "pcp"),
    ("DCP", "kv_split"),
)

MOE_DISPATCH_NOTES: Final[tuple[str, ...]] = (
    "- 'allreduce': src/vllm_torchtpu/layers/adapter/custom_ops/moe_runner.py:128 "
    "(default when USE_MOE_FUSED_EP_KERNEL=0)",
    "- 'a2a_push': src/vllm_torchtpu/layers/adapter/fused_moe_ep.py:794 "
    "(USE_MOE_FUSED_EP_KERNEL=1)",
    "- 'ag_ep': src/vllm_torchtpu/layers/adapter/pipelined_fused_moe.py:261-301 "
    "(default all-gather + reduce-scatter across EP ranks when "
    "USE_MOE_FUSED_EP_KERNEL=0)",
    "- 'fused_rs': src/vllm_torchtpu/platforms/tpu_platform.py:1565-1625 and "
    "src/vllm_torchtpu/models/vllm/qwen3_next_patch.py:190 "
    "(TPU_MOE_SEQUENCE_PARALLEL=1 USE_MOE_FUSED_EP_KERNEL=1; requires "
    "--enable-expert-parallel, DP=1, PP=1, TP>1 per __init__.py:664)",
)


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Typed launch command emission options for ``point_launch_args``."""

    kv_dtype: str = "auto"
    gpu_memory_utilization: float | None = None
    model: str | None = None
    spec_tokens: int = 0
    mtp_layers: int = 0
    hbm_reserve_gb: float = 0.0


class PointLaunchPayload(TypedDict):
    """Launch CLI arguments and formatted commands for a single Pareto point."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]


class LaunchPayload(TypedDict):
    """Structured vLLM serve CLI payload emitted for one serving plan."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]
    warnings: list[str]
    serving_plan: Mapping[str, object]


class ProvenancePayload(TypedDict):
    """Analytical model provenance metadata in solve JSON payloads."""

    calibrated: bool
    basis: str


class TopKEntry(TypedDict):
    """Ranked top layout entry with role label, throughput margin, and launch."""

    frontier_role: str
    margin_vs_rank1_pct: float
    point: Mapping[str, object]
    launch: PointLaunchPayload


class SolvePayload(TypedDict, total=False):
    """Versioned ``solve/v2`` JSON payload with provenance and Top-K launches."""

    model_layers: int
    chip: str
    scenario: Mapping[str, object]
    mode: str
    recommended: Mapping[str, object] | None
    frontier: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...]
    candidates: tuple[Mapping[str, object], ...]
    gpu_memory_utilization: float
    search: Mapping[str, object] | None
    schema: str
    provenance: ProvenancePayload
    chip_spec: Mapping[str, object]
    model: str | None
    top_k: list[TopKEntry]
    winner_prefix_hit_range: tuple[float, float] | None
    prefix_hit_crossover_rate: float | None
    spec_tokens: int
    mtp_layers: int
    hbm_reserve_gb: float
    target_chips: int
    budget: int
    budget_source: str
    max_instances: int | None
    pins: Mapping[str, object] | None
    chip_budget: Mapping[str, object]
    reference: Mapping[str, object]
    dtypes: Mapping[str, object]
    single_host_fit: dict[str, object]


class CeilingRow(TypedDict):
    """Single Pareto ceiling row exported by ``emit_ceiling_rows``."""

    est_interactivity_tok_s: float
    token_throughput_per_chip: float
    output_throughput_per_chip: float
    est_ttft_ms: float
    est_tpot_ms: float
    bound_tpot_ms: float
    concurrency: int
    total_chips: int
    mode: str
    binding_unit: str


def pool_cli_args(
    pool: PoolMeshPlan,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    spec_tokens: int = 0,
    mtp_layers: int = 0,
) -> list[str]:
    """Translate one ``PoolMeshPlan`` into ``vllm serve`` CLI flags."""
    args = [
        "--tensor-parallel-size",
        str(pool.tp),
        "--data-parallel-size",
        str(pool.dp),
        "--kv-cache-dtype",
        kv_dtype,
    ]
    if pool.pp > 1:
        args.extend(["--pipeline-parallel-size", str(pool.pp)])
    if pool.pcp > 1:
        args.extend(["--prefill-context-parallel-size", str(pool.pcp)])
    if pool.kv_split > 1:
        args.extend(["--decode-context-parallel-size", str(pool.kv_split)])
    if pool.ep > 1 or pool.moe_dispatch == "fused_rs":
        args.append("--enable-expert-parallel")
    if pool.max_model_len is not None:
        args.extend(["--max-model-len", str(pool.max_model_len)])
    if pool.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(pool.max_num_seqs)])
    if pool.kv_offload_gb > 0.0:
        args.extend(["--kv-offloading-size", f"{pool.kv_offload_gb:g}"])
    if gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", f"{gpu_memory_utilization:g}"])
    if spec_tokens > 0 and pool.role != "prefill" and mtp_layers > 0:
        args.extend(
            [
                "--speculative-config",
                f'{{"method":"mtp","num_speculative_tokens":{spec_tokens}}}',
            ]
        )
    return args


def _pool_int(val: object, default: int = 1) -> int:
    """Return integer pool attribute value when present, else default."""
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return default


def format_parallelism(pool: Mapping[str, object]) -> str:
    """Format active parallelism axes joined by ' x ' in canonical order, or TP1."""
    parts = [
        f"{name}{_pool_int(pool.get(key))}"
        for name, key in _POOL_AXES
        if _pool_int(pool.get(key)) > 1
    ]
    return " x ".join(parts) if parts else "TP1"


def pool_label(pool: Mapping[str, object]) -> str:
    """Return a concise pool label.

    Example: ``4x 2x2x2 (32 chips, 8 hosts, 64 cores): TP16 x EP16 x DCP2``.
    """
    instance_count = _pool_int(pool.get("instances"))
    instance_prefix = f"{instance_count}x "
    topology_name = str(pool.get("topology", ""))
    chips_per_instance = _pool_int(pool.get("chips_per_instance"), default=0)
    if chips_per_instance <= 0 and topology_name in TOPOLOGY_AXES:
        topology_dimensions = TOPOLOGY_AXES[topology_name]
        chips_per_instance = (
            topology_dimensions[0] * topology_dimensions[1] * topology_dimensions[2]
        )
    total_chips = chips_per_instance * max(1, instance_count)
    layout_text = format_parallelism(pool)
    if total_chips <= 0:
        return f"{instance_prefix}{topology_name}: {layout_text}"
    hosts = max(1, total_chips // DEFAULT_CHIPS_PER_HOST)
    host_description = "1 host" if hosts == 1 else f"{hosts} hosts"
    cores = total_chips * DEFAULT_CORES_PER_CHIP
    pool_description = f" ({total_chips} chips, {host_description}, {cores} cores)"
    return f"{instance_prefix}{topology_name}{pool_description}: {layout_text}"


def _format_server_cmd(
    cli_args: list[str],
    model: str | None,
    *,
    pp_layer_partition: tuple[int, ...] | None = None,
    moe_dispatch: str = "allreduce",
) -> str:
    """Format launch CLI arguments as a standalone bash command string."""
    joined = shlex.join(cli_args)
    prefix = f"vllm serve {model}" if model else "vllm serve"
    cmd = f"{prefix} {joined}".strip()
    env_parts: list[str] = []
    if moe_dispatch == "a2a_push":
        env_parts.append("USE_MOE_FUSED_EP_KERNEL=1")
    elif moe_dispatch == "fused_rs":
        env_parts.append("TPU_MOE_SEQUENCE_PARALLEL=1 USE_MOE_FUSED_EP_KERNEL=1")
    if pp_layer_partition is not None and len(set(pp_layer_partition)) > 1:
        part_str = ",".join(str(x) for x in pp_layer_partition)
        env_parts.append(f"VLLM_PP_LAYER_PARTITION={part_str}")
    return f"{' '.join(env_parts)} {cmd}" if env_parts else cmd


def _pool_key(pool: PoolMeshPlan | None) -> tuple[object, ...] | None:
    """Group layouts by structural parallelism axes yielding distinct serve CLIs."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.pp_layer_partition,
    )


def _point_struct_key(
    pt: ParetoPoint,
) -> tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]:
    """Build composite structural identity key across mixed or prefill/decode pools."""
    return (pt.mode, _pool_key(pt.prefill_pool), _pool_key(pt.decode_pool))


def _coarse_pool_key(pool: PoolMeshPlan | None) -> tuple[object, ...] | None:
    """Group PoolMeshPlan variants by coarse parallelism archetype."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances > 1,
        pool.pp >= DEEP_PP_THRESHOLD,
        pool.pp > 1,
        pool.pcp > 1,
        pool.kv_split > 1,
        pool.tp > 1,
        pool.dp > 1,
    )


def _coarse_family_key(
    pt: ParetoPoint,
) -> tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]:
    """Build coarse archetype identity key across mixed or prefill/decode pools."""
    return (
        pt.mode,
        _coarse_pool_key(pt.prefill_pool),
        _coarse_pool_key(pt.decode_pool),
    )


def _select_compact_top_layouts(
    ranked: list[ParetoPoint],
    fastest: ParetoPoint,
    k: int,
) -> tuple[ParetoPoint, ...]:
    """Preserve compact Top-K selection when k <= 3."""
    if len(ranked) <= 1 or k < MIN_TOP_LAYOUTS_FOR_FASTEST:
        return tuple(ranked[:k])
    selected = list(ranked[:k])
    f_key = _point_struct_key(fastest)
    sel_keys = [_point_struct_key(p) for p in selected]
    if f_key not in sel_keys:
        selected[-1] = fastest
    elif sel_keys.index(f_key) > 0:
        selected[sel_keys.index(f_key)] = fastest
    selected.sort(key=lambda p: -p.token_throughput_per_chip)
    return tuple(selected)


def _diversity_score(
    pt: ParetoPoint,
    chosen: list[ParetoPoint],
    meta: tuple[float, float, bool],
) -> tuple[int, int, int, int, int, int, float, float]:
    """Score a candidate by frontier status, uncovered archetype axes, and spread."""
    max_tput, max_inter, is_frontier = meta
    seen_modes = {c.mode for c in chosen}
    seen_fams = {_coarse_family_key(c) for c in chosen}
    has_deep_pp = any(
        c.prefill_pool is not None and c.prefill_pool.pp >= DEEP_PP_THRESHOLD
        for c in chosen
    )
    has_pp = any(c.prefill_pool is not None and c.prefill_pool.pp > 1 for c in chosen)
    has_pcp = any(c.prefill_pool is not None and c.prefill_pool.pcp > 1 for c in chosen)
    pref = pt.prefill_pool
    new_mode = int(pt.mode not in seen_modes)
    new_deep_pp = int(
        not has_deep_pp and pref is not None and pref.pp >= DEEP_PP_THRESHOLD
    )
    new_pp = int(not has_pp and pref is not None and pref.pp > 1)
    new_pcp = int(not has_pcp and pref is not None and pref.pcp > 1)
    new_fam = int(_coarse_family_key(pt) not in seen_fams)
    min_dist = min(
        math.hypot(
            (pt.token_throughput_per_chip - c.token_throughput_per_chip) / max_tput,
            (pt.est_interactivity_tok_s - c.est_interactivity_tok_s) / max_inter,
        )
        for c in chosen
    )
    norm_tput = pt.token_throughput_per_chip / max_tput
    return (
        int(is_frontier),
        new_mode,
        new_deep_pp,
        new_pp,
        new_pcp,
        new_fam,
        min_dist,
        norm_tput,
    )


def _select_diverse_top_layouts(
    ranked: list[ParetoPoint],
    fastest: ParetoPoint,
    frontier_keys: set[
        tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]
    ],
    k: int,
) -> tuple[ParetoPoint, ...]:
    """Select Rank-1, Max-Interactivity, and diverse frontier knees up to k."""
    if len(ranked) <= k:
        selected = list(ranked)
    else:
        max_tput = max(p.token_throughput_per_chip for p in ranked) or 1.0
        max_inter = max(p.est_interactivity_tok_s for p in ranked) or 1.0
        chosen: list[ParetoPoint] = [ranked[0]]
        chosen_keys = {_point_struct_key(ranked[0])}
        remaining = list(ranked[1:])
        while remaining and len(chosen) < max(1, k - 1):
            best_pt = max(
                remaining,
                key=lambda p: _diversity_score(
                    p,
                    chosen,
                    (max_tput, max_inter, _point_struct_key(p) in frontier_keys),
                ),
            )
            chosen.append(best_pt)
            chosen_keys.add(_point_struct_key(best_pt))
            remaining = [
                p for p in remaining if _point_struct_key(p) not in chosen_keys
            ]
        if len(chosen) < k and remaining:
            last_pt = max(
                remaining,
                key=lambda p: _diversity_score(
                    p,
                    chosen,
                    (max_tput, max_inter, _point_struct_key(p) in frontier_keys),
                ),
            )
            chosen.append(last_pt)
            chosen_keys.add(_point_struct_key(last_pt))
        selected = chosen
    f_key = _point_struct_key(fastest)
    sel_keys = [_point_struct_key(p) for p in selected]
    if f_key in sel_keys:
        idx_f = sel_keys.index(f_key)
        if idx_f > 0:
            selected[idx_f] = fastest
    elif selected:
        selected[-1] = fastest
    selected.sort(key=lambda p: -p.token_throughput_per_chip)
    return tuple(selected)


def _filter_eligible(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return SLO-met points, preferring `estimate_meets` when any exist."""
    slo_pts = [pt for pt in points if pt.slo_met]
    est_pts = [pt for pt in slo_pts if pt.slo_verdict == "estimate_meets"]
    return est_pts if est_pts else slo_pts


def _best_by_struct_key(
    points: Sequence[ParetoPoint],
) -> dict[
    tuple[str, tuple[object, ...] | None, tuple[object, ...] | None],
    ParetoPoint,
]:
    """Map each structural layout key to its highest-throughput point."""
    best: dict[
        tuple[str, tuple[object, ...] | None, tuple[object, ...] | None],
        ParetoPoint,
    ] = {}
    for pt in points:
        key = _point_struct_key(pt)
        cur = best.get(key)
        if cur is None or (
            pt.token_throughput_per_chip,
            pt.est_interactivity_tok_s,
        ) > (
            cur.token_throughput_per_chip,
            cur.est_interactivity_tok_s,
        ):
            best[key] = pt
    return best


def _select_top_layouts(
    ranked: list[ParetoPoint],
    fastest: ParetoPoint,
    frontier_keys: set[
        tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]
    ],
    k: int,
    slots: int | None = None,
) -> tuple[ParetoPoint, ...]:
    """Run the selector that k calls for: compact for small k, else diverse.

    `slots` is the number of entries to select, k by default. A caller that frees
    one slot passes k - 1 and keeps the selector of k.
    """
    slots = k if slots is None else slots
    if k < MIN_DIVERSE_TOP_LAYOUTS:
        return _select_compact_top_layouts(ranked, fastest, slots)
    return _select_diverse_top_layouts(ranked, fastest, frontier_keys, slots)


def _best_tpot_bound_candidate(
    selected: Sequence[ParetoPoint],
    eligible: Sequence[ParetoPoint],
) -> ParetoPoint | None:
    """Return the lowest time-per-token bound point when no selected layout attains it.

    Layouts are compared, not points: a selected layout that attains the lowest
    bound at any of its eligible points counts. On equal bounds the higher
    estimated throughput wins.
    """
    lowest_bound = min(point.bound_tpot_ms for point in eligible)
    best_points = [point for point in eligible if point.bound_tpot_ms == lowest_bound]
    selected_keys = {_point_struct_key(point) for point in selected}
    if any(_point_struct_key(point) in selected_keys for point in best_points):
        return None
    return max(best_points, key=lambda point: point.token_throughput_per_chip)


def _with_bound_best(
    selected: tuple[ParetoPoint, ...],
    bound_best: ParetoPoint,
    reselect: Callable[[int], tuple[ParetoPoint, ...]],
    k: int,
) -> tuple[ParetoPoint, ...]:
    """Add the bound-best layout to the selection without repeating a layout.

    A selection of exactly k entries gives up the entry the selector picks last:
    the selector runs again with one slot fewer, which keeps rank 1 and the
    fastest layout, and free slots are refilled from the first selection.
    """
    if len(selected) != k:
        result = [*selected, bound_best]
    else:
        result = list(reselect(k - 1))
        layout_keys = {_point_struct_key(point) for point in result}
        for point in (bound_best, *selected):
            if len(result) >= k:
                break
            if _point_struct_key(point) not in layout_keys:
                result.append(point)
                layout_keys.add(_point_struct_key(point))
    return tuple(sorted(result, key=lambda point: -point.token_throughput_per_chip))


def strict_rank(value: float, values: Sequence[float], *, lower_is_better: bool) -> int:
    """Return 1 plus the number of entries strictly better than ``value``."""
    if lower_is_better:
        return 1 + sum(1 for other in values if other < value)
    return 1 + sum(1 for other in values if other > value)


def top_layouts(
    plan: ServingPlan,
    k: int = DEFAULT_TOP_K,
    *,
    near_tie_rel: float = 0.0,
) -> tuple[ParetoPoint, ...]:
    """Select up to k structurally distinct SLO-met layouts spanning Pareto roles."""
    if k <= 0:
        return ()
    eligible = _filter_eligible(plan.candidates)
    if not eligible:
        return ()
    frontier_eligible = _filter_eligible(plan.frontier)
    front_best = _best_by_struct_key(frontier_eligible)
    if len(front_best) >= k and near_tie_rel <= 0.0:
        best_map = front_best
        fastest_pool = frontier_eligible
    else:
        best_map = _best_by_struct_key(eligible)
        # Keep the actual frontier representative for every structural key on frontier.
        best_map.update(front_best)
        fastest_pool = eligible
    front_ranked = sorted(
        (pt for key, pt in best_map.items() if key in front_best),
        key=lambda p: -p.token_throughput_per_chip,
    )
    cand_ranked = sorted(
        (pt for key, pt in best_map.items() if key not in front_best),
        key=lambda p: -p.token_throughput_per_chip,
    )
    ranked = [*front_ranked, *cand_ranked]
    fastest = max(
        fastest_pool,
        key=lambda p: (p.est_interactivity_tok_s, p.token_throughput_per_chip),
    )
    frontier_keys = set(front_best)
    selected = _select_top_layouts(ranked, fastest, frontier_keys, k)

    if near_tie_rel > 0.0 and len(selected) >= k and len(ranked) > k:
        kth_tput = selected[k - 1].token_throughput_per_chip
        sel_keys = {_point_struct_key(p) for p in selected}
        near_ties = [
            p
            for p in ranked[k:]
            if _point_struct_key(p) not in sel_keys
            and (kth_tput - p.token_throughput_per_chip) / max(1e-9, kth_tput)
            <= near_tie_rel
        ]
        if near_ties:
            selected = tuple(
                sorted(
                    [*selected, *near_ties],
                    key=lambda p: -p.token_throughput_per_chip,
                )
            )

    # Rank 1 and the fastest layout keep their slots, so k must leave a third one.
    if k >= MIN_TOP_LAYOUTS_FOR_BOUND_BEST:
        bound_best = _best_tpot_bound_candidate(selected, eligible)
        if bound_best is not None:
            selected = _with_bound_best(
                selected,
                bound_best,
                lambda slots: _select_top_layouts(
                    ranked, fastest, frontier_keys, k, slots
                ),
                k,
            )
    return selected


def top_layout_roles(
    top_pts: tuple[ParetoPoint, ...],
) -> list[tuple[ParetoPoint, str, float]]:
    """Annotate selected Top-K points with role labels and Rank-1 throughput margins."""
    top1_tput = top_pts[0].token_throughput_per_chip if top_pts else 0.0
    fastest_idx = (
        max(
            range(len(top_pts)),
            key=lambda i: (
                top_pts[i].est_interactivity_tok_s,
                top_pts[i].token_throughput_per_chip,
            ),
        )
        if len(top_pts) > 1
        else -1
    )
    lowest_bound = min((pt.bound_tpot_ms for pt in top_pts), default=0.0)
    bound_idx = next(
        (i for i, pt in enumerate(top_pts) if pt.bound_tpot_ms == lowest_bound), -1
    )
    if any(
        top_pts[i].bound_tpot_ms == lowest_bound
        for i in (0, fastest_idx)
        if 0 <= i < len(top_pts)
    ):
        bound_idx = -1
    out: list[tuple[ParetoPoint, str, float]] = []
    for idx, pt in enumerate(top_pts):
        if idx == 0:
            role = "max_throughput"
        elif idx == fastest_idx:
            role = "max_interactivity"
        elif idx == bound_idx:
            role = "min_tpot_bound"
        else:
            role = "balanced_knee"
        margin = round(
            PERCENT_SCALE
            * (pt.token_throughput_per_chip - top1_tput)
            / max(1e-9, top1_tput),
            2,
        )
        out.append((pt, role, margin))
    return out


def point_launch_args(
    pt: ParetoPoint,
    options: LaunchOptions | None = None,
) -> PointLaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for a single ``ParetoPoint``."""
    opts = options or LaunchOptions()
    decode_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=opts.kv_dtype,
        gpu_memory_utilization=opts.gpu_memory_utilization,
        spec_tokens=opts.spec_tokens,
        mtp_layers=opts.mtp_layers,
    )
    prefill_args = (
        pool_cli_args(
            pt.prefill_pool,
            kv_dtype=opts.kv_dtype,
            gpu_memory_utilization=opts.gpu_memory_utilization,
            spec_tokens=opts.spec_tokens,
            mtp_layers=opts.mtp_layers,
        )
        if pt.prefill_pool is not None
        else []
    )
    decode_cmd = _format_server_cmd(
        decode_args,
        opts.model,
        pp_layer_partition=pt.decode_pool.pp_layer_partition,
        moe_dispatch=pt.decode_pool.moe_dispatch,
    )
    if pt.prefill_pool is not None:
        p_pool = pt.prefill_pool
        d_pool = pt.decode_pool
        prefill_cmd = _format_server_cmd(
            prefill_args,
            opts.model,
            pp_layer_partition=p_pool.pp_layer_partition,
            moe_dispatch=p_pool.moe_dispatch,
        )
        p_chips = p_pool.instances * p_pool.chips_per_instance
        d_chips = d_pool.instances * d_pool.chips_per_instance
        commands = [
            f"# provision: prefill {pool_label(asdict(p_pool))} ({p_chips} chips)",
            prefill_cmd,
            f"# provision: decode {pool_label(asdict(d_pool))} ({d_chips} chips)",
            decode_cmd,
            (
                "# note: disaggregated KV transfer / connector flags are not "
                "emitted; configure for your deployment's KV connector"
            ),
        ]
    else:
        inst = pt.decode_pool.instances
        topo = pt.decode_pool.topology
        if inst > 1:
            commands = [
                f"# mixed ({inst}x {topo}): {decode_cmd}",
                decode_cmd,
            ]
        else:
            commands = [decode_cmd]

    if (
        "--max-num-seqs" in decode_args or "--max-num-seqs" in prefill_args
    ) and opts.hbm_reserve_gb == 0.0:
        commands.append(
            "# note: --max-num-seqs is an upper bound on what fits when "
            "--hbm-reserve-gb is 0"
        )

    return {
        "mode": pt.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
    }


def _plan_dict(plan: ServingPlan) -> dict[str, object]:
    """Convert ``ServingPlan`` to dict excluding internal ``sweep_candidates``."""
    data: dict[str, object] = dict(asdict(plan).items())
    data.pop("sweep_candidates", None)
    return data


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
) -> LaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for ``plan``."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    warnings = list(plan.warnings)
    recommended_point = plan.recommended
    if recommended_point is None or not recommended_point.slo_met:
        if not warnings:
            warnings.append("No feasible layout met memory and latency constraints.")
        return {
            "mode": plan.mode,
            "cli_args": [],
            "prefill_cli_args": [],
            "commands": [],
            "warnings": warnings,
            "serving_plan": _plan_dict(plan),
        }

    eff_spec = (
        plan.spec_tokens if plan.spec_tokens > 0 else (plan.scenario.spec_tokens or 0)
    )
    plan_mtp = plan.mtp_layers
    if eff_spec > 0 and plan_mtp <= 0:
        warnings.append(
            "Model declares zero MTP draft layers; omitting --speculative-config."
        )
    launch = point_launch_args(
        recommended_point,
        LaunchOptions(
            kv_dtype=kv_str,
            gpu_memory_utilization=plan.gpu_memory_utilization,
            model=model,
            spec_tokens=eff_spec,
            mtp_layers=plan_mtp,
            hbm_reserve_gb=plan.hbm_reserve_gb,
        ),
    )
    return {
        "mode": launch["mode"],
        "cli_args": launch["cli_args"],
        "prefill_cli_args": launch["prefill_cli_args"],
        "commands": launch["commands"],
        "warnings": warnings,
        "serving_plan": _plan_dict(plan),
    }


def _compute_single_host_fit(plan: ServingPlan) -> dict[str, object]:
    """Compute single-host 2x2x1 fit verdict, capacity, and status."""
    four_chip_points = [
        pt
        for pt in (*plan.frontier, *plan.candidates)
        if pt.total_chips == DEFAULT_CHIPS_PER_HOST
        and pt.decode_pool.topology == "2x2x1"
        and pt.prefill_pool is None
    ]
    if four_chip_points:
        best_pt = max(
            four_chip_points,
            key=lambda p: (p.slo_met, p.token_throughput_per_chip),
        )
        pool = best_pt.decode_pool
        return {
            "feasible": True,
            "status": "FEASIBLE",
            "reason": "Fits single-host 2x2x1 (1 host, 4 chips, 8 TensorCores)",
            "hbm_used_gb": round(pool.hbm_used_gb, 2),
            "hbm_limit_gb": round(pool.hbm_limit_gb, 2),
            "best_parallelism": format_parallelism(asdict(pool)),
        }
    chip_spec = CHIPS.get(plan.chip)
    per_core_bytes = chip_spec.hbm_bytes if chip_spec else 96.0 * (1 << 30)
    limit_gb = round(
        per_core_bytes * plan.gpu_memory_utilization / float(1 << 30),
        2,
    )
    return {
        "feasible": False,
        "status": "REJECTED",
        "reason": (
            "Weight and KV cache exceed 2x2x1 HBM capacity "
            "(1 host, 4 chips, 8 TensorCores)"
        ),
        "hbm_used_gb": 0.0,
        "hbm_limit_gb": limit_gb,
        "best_parallelism": "TP8",
    }


_REF_AXIS_MAP: Final[dict[str, str]] = {
    "TP": "tp",
    "DP": "dp",
    "EP": "ep",
    "PP": "pp",
    "PCP": "pcp",
    "DCP": "kv_split",
}
_EXPECTED_DISAGG_POOL_PARTS: Final[int] = 2


def _parse_single_pool_layout(pool_text: str) -> dict[str, int]:
    """Parse one pool layout string (e.g. 'TP8 x EP8') into an axis dictionary."""
    stripped_pool_text = pool_text.strip()
    if not stripped_pool_text:
        msg = f"empty layout string: {pool_text!r}"
        raise ValueError(msg)
    raw_tokens = re.split(r"\s*[xX]\s*", stripped_pool_text)
    seen_axes: set[str] = set()
    axis_values: dict[str, int] = {
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "pcp": 1,
        "kv_split": 1,
    }
    for raw_token in raw_tokens:
        token = raw_token.strip()
        if not token:
            msg = f"empty axis token in layout: {pool_text!r}"
            raise ValueError(msg)
        match = re.fullmatch(r"([a-zA-Z]+)(\d+)", token)
        if not match:
            msg = f"malformed parallelism axis token: {token!r}"
            raise ValueError(msg)
        axis_name = match.group(1).upper()
        if axis_name not in _REF_AXIS_MAP:
            msg = f"unknown parallelism axis in layout: {token!r}"
            raise ValueError(msg)
        if axis_name in seen_axes:
            msg = f"axis given twice in layout: {token!r}"
            raise ValueError(msg)
        seen_axes.add(axis_name)
        axis_degree = int(match.group(2))
        if axis_degree < 1:
            msg = f"parallelism degree must be at least 1: {token!r}"
            raise ValueError(msg)
        axis_values[_REF_AXIS_MAP[axis_name]] = axis_degree
    return axis_values


def _parse_disagg_layout_parts(
    layout_text: str, pool_parts: Sequence[str]
) -> tuple[str, str]:
    """Extract prefill and decode pool strings from disaggregated layout parts."""
    if len(pool_parts) != _EXPECTED_DISAGG_POOL_PARTS:
        msg = (
            f"disaggregated layout requires two parts separated by '/': {layout_text!r}"
        )
        raise ValueError(msg)
    prefill_part: str | None = None
    decode_part: str | None = None
    for part in pool_parts:
        upper_part = part.upper()
        if upper_part.startswith("P:"):
            if prefill_part is not None:
                msg = f"multiple prefill parts in layout: {part!r}"
                raise ValueError(msg)
            prefill_part = part[2:].strip()
        elif upper_part.startswith("D:"):
            if decode_part is not None:
                msg = f"multiple decode parts in layout: {part!r}"
                raise ValueError(msg)
            decode_part = part[2:].strip()
        else:
            msg = (
                f"unrecognized pool prefix in layout (expected 'P:' or 'D:'): {part!r}"
            )
            raise ValueError(msg)
    if prefill_part is None:
        msg = f"layout needs a prefill part: {layout_text!r}"
        raise ValueError(msg)
    if decode_part is None:
        msg = f"layout needs a decode part: {layout_text!r}"
        raise ValueError(msg)
    return prefill_part, decode_part


def _parse_disagg_layout(
    layout_text: str, stripped_layout: str
) -> dict[str, dict[str, int]]:
    """Parse disaggregated layout text with prefill and decode pools."""
    pool_parts = [part.strip() for part in stripped_layout.split("/")]
    prefill_part, decode_part = _parse_disagg_layout_parts(layout_text, pool_parts)
    return {
        "prefill": _parse_single_pool_layout(prefill_part),
        "decode": _parse_single_pool_layout(decode_part),
    }


def _parse_single_layout(
    layout_text: str, stripped_layout: str
) -> dict[str, dict[str, int]]:
    """Parse single-pool layout text into decode axes."""
    decode_text = stripped_layout
    if decode_text.upper().startswith("D:"):
        decode_text = decode_text[2:].strip()
    elif decode_text.upper().startswith("P:"):
        msg = f"layout needs a decode part: {layout_text!r}"
        raise ValueError(msg)
    return {
        "decode": _parse_single_pool_layout(decode_text),
    }


def parse_layout(layout_text: str) -> dict[str, dict[str, int]]:
    """Parse reference layout specification into decode and optional prefill axes."""
    stripped_layout = layout_text.strip()
    if not stripped_layout:
        msg = f"empty layout text: {layout_text!r}"
        raise ValueError(msg)

    if "/" in stripped_layout:
        return _parse_disagg_layout(layout_text, stripped_layout)
    return _parse_single_layout(layout_text, stripped_layout)


def _pool_matches_axes(pool: PoolMeshPlan, axes: Mapping[str, int]) -> bool:
    """Check if a pool mesh plan matches the exact reference axis values."""
    return (
        pool.tp == axes["tp"]
        and pool.dp == axes["dp"]
        and pool.ep == axes["ep"]
        and pool.pp == axes["pp"]
        and pool.pcp == axes["pcp"]
        and pool.kv_split == axes["kv_split"]
    )


def _candidate_matches_layout(
    candidate: ParetoPoint, parsed_layout: Mapping[str, Mapping[str, int]]
) -> bool:
    """Check if a candidate matches reference decode and prefill pools."""
    if not _pool_matches_axes(candidate.decode_pool, parsed_layout["decode"]):
        return False
    if "prefill" in parsed_layout:
        if candidate.prefill_pool is None:
            return False
        return _pool_matches_axes(candidate.prefill_pool, parsed_layout["prefill"])
    return candidate.prefill_pool is None


def _best_values_by_non_matching_layout(
    candidates: Sequence[ParetoPoint],
    parsed_layout: Mapping[str, Mapping[str, int]],
    value_getter: Callable[[ParetoPoint], float],
    *,
    lower_is_better: bool,
) -> list[float]:
    """Return best metric value per distinct layout for non-matching candidates."""
    best_by_layout_key: dict[
        tuple[str, tuple[object, ...] | None, tuple[object, ...] | None],
        float,
    ] = {}
    for candidate in candidates:
        if not _candidate_matches_layout(candidate, parsed_layout):
            layout_key = _point_struct_key(candidate)
            candidate_value = value_getter(candidate)
            if (
                layout_key not in best_by_layout_key
                or (
                    lower_is_better and candidate_value < best_by_layout_key[layout_key]
                )
                or (
                    not lower_is_better
                    and candidate_value > best_by_layout_key[layout_key]
                )
            ):
                best_by_layout_key[layout_key] = candidate_value
    return list(best_by_layout_key.values())


def reference_layout_report(
    plan: ServingPlan,
    top_points: Sequence[ParetoPoint],
    layout_text: str,
    rejected_by: str,
) -> dict[str, object]:
    """Compile reference layout ranking report across all enumerated candidates."""
    parsed_layout = parse_layout(layout_text)
    if "prefill" in parsed_layout:
        normalized_layout = (
            f"P: {format_parallelism(parsed_layout['prefill'])} / "
            f"D: {format_parallelism(parsed_layout['decode'])}"
        )
    else:
        normalized_layout = format_parallelism(parsed_layout["decode"])

    matching_candidates = [
        candidate
        for candidate in plan.candidates
        if _candidate_matches_layout(candidate, parsed_layout)
    ]
    found = len(matching_candidates) > 0
    distinct_layout_keys = {
        _point_struct_key(candidate) for candidate in plan.candidates
    }
    num_candidates = len(distinct_layout_keys)

    if found:
        best_candidate = max(
            matching_candidates,
            key=lambda candidate: candidate.token_throughput_per_chip,
        )
        reference_bound = min(
            candidate.bound_tpot_ms for candidate in matching_candidates
        )
        other_bounds = _best_values_by_non_matching_layout(
            plan.candidates,
            parsed_layout,
            lambda candidate: candidate.bound_tpot_ms,
            lower_is_better=True,
        )
        rank_by_tpot_bound = strict_rank(
            reference_bound, other_bounds, lower_is_better=True
        )

        reference_throughput = max(
            candidate.token_throughput_per_chip for candidate in matching_candidates
        )
        other_throughputs = _best_values_by_non_matching_layout(
            plan.candidates,
            parsed_layout,
            lambda candidate: candidate.token_throughput_per_chip,
            lower_is_better=False,
        )
        rank_by_estimated_throughput = strict_rank(
            reference_throughput, other_throughputs, lower_is_better=False
        )

        in_top_k = any(
            _candidate_matches_layout(candidate, parsed_layout)
            for candidate in top_points
        )
        candidate_point_dict: Mapping[str, object] | None = asdict(best_candidate)
        final_rejection_reason = ""
    else:
        rank_by_tpot_bound = None
        rank_by_estimated_throughput = None
        in_top_k = False
        candidate_point_dict = None
        final_rejection_reason = rejected_by

    return {
        "layout": normalized_layout,
        "found": found,
        "candidates": num_candidates,
        "rank_by_tpot_bound": rank_by_tpot_bound,
        "rank_by_estimated_throughput": rank_by_estimated_throughput,
        "in_top_k": in_top_k,
        "point": candidate_point_dict,
        "rejected_by": final_rejection_reason,
    }


def build_solve_payload(
    plan: ServingPlan,
    *,
    chip: ChipSpec,
    dtypes: Dtypes | None = None,
    model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    reference: Mapping[str, object] | None = None,
) -> SolvePayload:
    """Build ``solve/v2`` JSON payload with provenance, chip catalog, and Top-K."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    payload: dict[str, object] = _plan_dict(plan)
    payload["single_host_fit"] = _compute_single_host_fit(plan)
    target_chips = (
        plan.recommended.total_chips if plan.recommended is not None else plan.budget
    )
    payload["target_chips"] = target_chips
    payload["budget"] = plan.budget
    payload["budget_source"] = plan.budget_source
    payload["max_instances"] = plan.max_instances
    payload["pins"] = plan.pins
    pins_map = plan.pins or {}
    payload["chip_budget"] = {
        "max_chips": plan.budget,
        "budget_source": plan.budget_source,
        "max_instances": plan.max_instances,
        "prefill_topology": pins_map.get("prefill_topology"),
        "decode_topology": pins_map.get("decode_topology"),
        "pd_ratio": pins_map.get("pd_ratio"),
    }
    payload["schema"] = SOLVE_SCHEMA
    payload["provenance"] = {
        "calibrated": False,
        "basis": (
            "analytical roofline bounds and estimates; est_* values are "
            "modeled estimates, bound_* values are physical lower bounds"
        ),
    }
    chip_dict: dict[str, object] = dict(asdict(chip).items())
    chip_dict["jax_version"] = JAX_VERSION
    chip_dict["catalog"] = chip_catalog(chip)
    payload["chip_spec"] = chip_dict
    payload["model"] = model
    top_pts = top_layouts(plan, top_k)
    eff_spec = (
        plan.spec_tokens if plan.spec_tokens > 0 else (plan.scenario.spec_tokens or 0)
    )
    opts = LaunchOptions(
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
        spec_tokens=eff_spec,
        mtp_layers=plan.mtp_layers,
        hbm_reserve_gb=plan.hbm_reserve_gb,
    )
    top_entries: list[TopKEntry] = [
        {
            "frontier_role": role,
            "margin_vs_rank1_pct": margin,
            "point": asdict(pt),
            "launch": point_launch_args(pt, opts),
        }
        for pt, role, margin in top_layout_roles(top_pts)
    ]
    payload["top_k"] = top_entries
    if reference is not None:
        ref_dict = dict(reference)
        ref_launch: Mapping[str, object] | None = None
        if ref_dict.get("found"):
            parsed_ref = parse_layout(str(ref_dict["layout"]))
            matching = [
                candidate
                for candidate in plan.candidates
                if _candidate_matches_layout(candidate, parsed_ref)
            ]
            if matching:
                best_candidate = max(
                    matching,
                    key=lambda candidate: candidate.token_throughput_per_chip,
                )
                ref_launch = point_launch_args(best_candidate, opts)
        ref_dict["launch"] = ref_launch
        payload["reference"] = ref_dict
    if dtypes is not None:
        payload["dtypes"] = asdict(dtypes)
    return cast("SolvePayload", payload)


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export Pareto ceiling rows leading with interactivity and throughput."""
    return [
        {
            "est_interactivity_tok_s": pt.est_interactivity_tok_s,
            "token_throughput_per_chip": pt.token_throughput_per_chip,
            "output_throughput_per_chip": pt.output_throughput_per_chip,
            "est_ttft_ms": pt.est_ttft_ms,
            "est_tpot_ms": pt.est_tpot_ms,
            "bound_tpot_ms": pt.bound_tpot_ms,
            "concurrency": pt.concurrency,
            "total_chips": pt.total_chips,
            "mode": pt.mode,
            "binding_unit": pt.binding,
        }
        for pt in plan.frontier
        if pt.slo_met
    ]
