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
"""Layout-family grouping, per-concurrency ordering, and reference ranking."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, cast

from .constants import SLO_VERDICT_RULED_OUT
from .schema import (
    _POOL_AXES,
    BoundFamilyEntry,
    ParetoPoint,
    PoolFloorRecord,
    PoolMeshPlan,
    ReferenceLayoutReport,
    ServingPlan,
    SolvePayload,
    SortBy,
    candidate_sort_key,
    effective_slo_verdict,
    format_parallelism,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_REF_AXIS_MAP: Final[dict[str, str]] = dict(_POOL_AXES)
_EXPECTED_DISAGG_POOL_PARTS: Final[int] = 2

__all__ = [
    "attach_reference_report",
    "build_bound_families_by_concurrency",
    "ordered_candidates_by_concurrency",
    "parse_layout",
    "reference_layout_report",
    "strict_rank",
]


def strict_rank(
    value: float,
    values: Sequence[float],
    *,
    lower_is_better: bool,
) -> int:
    """Return 1 plus the number of entries strictly better than ``value``."""
    if lower_is_better:
        return 1 + sum(1 for other in values if other < value)
    return 1 + sum(1 for other in values if other > value)


def _point_tpot_floor_ms(point: ParetoPoint, *, accept_len: float = 1.0) -> float:
    """Return per-token decode TPOT lower bound (ms) available on a candidate point."""
    if point.decode_floor is not None:
        return point.decode_floor.hard_floor_ms / max(1.0, accept_len)
    return point.est_tpot_ms


def _point_ttft_floor_ms(point: ParetoPoint) -> float:
    """Return prefill TTFT reference time (ms) available on a candidate point."""
    return point.est_ttft_ms


def _pool_mesh_family(pool: PoolMeshPlan | None) -> tuple[object, ...] | None:
    """Return mesh family key ignoring KV split and MoE dispatch variant."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
    )


def _bound_family_key(
    point: ParetoPoint,
) -> tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]:
    """Return structural (mode, prefill_mesh_family, decode_mesh_family) key."""
    return (
        point.mode,
        _pool_mesh_family(point.prefill_pool),
        _pool_mesh_family(point.decode_pool),
    )


def _ordered_candidate_key(
    point: ParetoPoint,
    sort_by: SortBy,
) -> tuple[bool, bool, float, float, int, str]:
    """Order candidates within one concurrency via the shared sort key."""
    return candidate_sort_key(point, sort_by)


def _ordered_concurrency_candidates(
    points: Sequence[ParetoPoint],
    *,
    sort_by: SortBy,
    limit: int | None,
) -> tuple[ParetoPoint, ...]:
    """Order all legal candidates at one concurrency and optionally cut to limit."""
    ordered = sorted(
        points,
        key=lambda point: _ordered_candidate_key(point, sort_by),
    )
    if limit is not None:
        return tuple(ordered[:limit])
    return tuple(ordered)


def ordered_candidates_by_concurrency(
    plan: ServingPlan,
    top_k: int | None = None,
) -> dict[int, tuple[ParetoPoint, ...]]:
    """Return ordered legal layouts per concurrency, cutting to ``k`` when set."""
    if (top_k is not None and top_k <= 0) or not plan.candidates:
        return {}
    by_concurrency: dict[int, list[ParetoPoint]] = {}
    for point in plan.candidates:
        by_concurrency.setdefault(point.concurrency, []).append(point)
    return {
        concurrency: _ordered_concurrency_candidates(
            by_concurrency[concurrency],
            sort_by=plan.sort_by,
            limit=top_k,
        )
        for concurrency in sorted(by_concurrency)
    }


def _pool_family_label(pool: PoolMeshPlan) -> str:
    """Format mesh topology and non-DCP parallelism label for one pool."""
    topology = (
        f"{pool.instances}x {pool.topology}" if pool.instances > 1 else pool.topology
    )
    pool_map = dict(asdict(pool))
    pool_map["kv_split"] = 1
    return f"{topology} {format_parallelism(pool_map)}"


def _bound_family_label(point: ParetoPoint) -> tuple[str, ...]:
    """Build human-readable multi-part family label tuple for bound tables."""
    decode_label = _pool_family_label(point.decode_pool)
    if point.prefill_pool is None:
        return (point.mode, decode_label)
    prefill_label = _pool_family_label(point.prefill_pool)
    return (point.mode, f"P:{prefill_label}", f"D:{decode_label}")


@dataclass(frozen=True, slots=True)
class _BoundFamilySummary:
    """Intermediate sortable family summary for one concurrency level."""

    sort_key: tuple[bool, float, float]
    payload: BoundFamilyEntry


def _summarize_bound_family(
    family_label: tuple[str, ...],
    group: Sequence[ParetoPoint],
    *,
    accept_len: float = 1.0,
) -> _BoundFamilySummary:
    """Aggregate one structural family into per-metric bound envelopes.

    ``min_bound_tpot_ms`` and ``min_bound_ttft_ms`` are independent per-metric
    minima over active family members (the modeled TTFT estimate serves as the
    standalone TTFT reference when dedicated prefill bounds are absent).
    """
    active = [
        point
        for point in group
        if effective_slo_verdict(point) != SLO_VERDICT_RULED_OUT
    ]
    any_not_ruled_out = bool(active)
    pool = active if any_not_ruled_out else list(group)
    best_point = min(
        pool,
        key=lambda point: (
            _point_tpot_floor_ms(point, accept_len=accept_len),
            _point_ttft_floor_ms(point),
            -point.token_throughput_per_chip,
        ),
    )
    sorted_members = sorted(
        pool,
        key=lambda point: (
            _point_tpot_floor_ms(point, accept_len=accept_len),
            -point.token_throughput_per_chip,
        ),
    )
    min_bound_tpot = min(
        _point_tpot_floor_ms(point, accept_len=accept_len) for point in pool
    )
    min_bound_ttft = min(_point_ttft_floor_ms(point) for point in pool)
    members: list[dict[str, object]] = [dict(asdict(point)) for point in sorted_members]
    payload: BoundFamilyEntry = {
        "rank": 0,
        "family": family_label,
        "min_bound_tpot_ms": min_bound_tpot,
        "min_bound_ttft_ms": min_bound_ttft,
        "binding": best_point.binding,
        "any_not_ruled_out": any_not_ruled_out,
        "member_count": len(members),
        "members": members,
    }
    return _BoundFamilySummary(
        sort_key=(not any_not_ruled_out, min_bound_tpot, min_bound_ttft),
        payload=payload,
    )


def _summarize_pool_record_family(
    family_label: tuple[str, ...],
    group: Sequence[PoolFloorRecord],
) -> _BoundFamilySummary:
    """Aggregate one single-pool record family into role-specific bound metadata."""
    active = [record for record in group if record.slo_met]
    any_slo_met = bool(active)
    pool = active if any_slo_met else list(group)
    is_decode = pool[0].role == "decode"
    sorted_members = sorted(
        pool,
        key=lambda record: (
            (
                record.tpot_floor_ms
                if is_decode and record.tpot_floor_ms is not None
                else (
                    record.ttft_floor_ms
                    if not is_decode and record.ttft_floor_ms is not None
                    else float("inf")
                )
            ),
            -record.token_throughput_per_chip,
        ),
    )
    best_record = sorted_members[0]
    min_tpot: float | None = (
        min(
            (
                record.tpot_floor_ms
                for record in pool
                if record.tpot_floor_ms is not None
            ),
            default=None,
        )
        if is_decode
        else None
    )
    min_ttft: float | None = (
        min(
            (
                record.ttft_floor_ms
                for record in pool
                if record.ttft_floor_ms is not None
            ),
            default=None,
        )
        if not is_decode
        else None
    )
    primary_bound = (
        min_tpot
        if is_decode and min_tpot is not None
        else (min_ttft if not is_decode and min_ttft is not None else float("inf"))
    )
    members: list[dict[str, object]] = [
        dict(asdict(record)) for record in sorted_members
    ]
    payload: BoundFamilyEntry = {
        "rank": 0,
        "family": family_label,
        "min_bound_tpot_ms": min_tpot,
        "min_bound_ttft_ms": min_ttft,
        "binding": best_record.binding,
        "any_not_ruled_out": any_slo_met,
        "member_count": len(members),
        "members": members,
    }
    return _BoundFamilySummary(
        sort_key=(
            not any_slo_met,
            primary_bound,
            -best_record.token_throughput_per_chip,
        ),
        payload=payload,
    )


def _rank_pool_role_families(
    records: Sequence[PoolFloorRecord],
) -> list[BoundFamilyEntry]:
    """Group and rank single-role pool floor records by their active bound."""
    grouped: dict[tuple[object, ...], list[PoolFloorRecord]] = {}
    labels: dict[tuple[object, ...], tuple[str, ...]] = {}
    for record in records:
        key = (record.role, _pool_mesh_family(record.pool))
        grouped.setdefault(key, []).append(record)
        labels.setdefault(key, (record.role, _pool_family_label(record.pool)))
    summaries = [
        _summarize_pool_record_family(labels[key], group)
        for key, group in grouped.items()
    ]
    summaries.sort(key=lambda item: item.sort_key)
    ranked: list[BoundFamilyEntry] = []
    for rank, item in enumerate(summaries, start=1):
        entry: BoundFamilyEntry = {**item.payload, "rank": rank}
        ranked.append(entry)
    return ranked


def _build_pool_families_by_concurrency(
    records: Sequence[PoolFloorRecord],
) -> dict[str, list[BoundFamilyEntry]]:
    """Group per-pool floor records by (role, mesh_family), ranking roles separately."""
    by_concurrency: dict[int, list[PoolFloorRecord]] = {}
    for record in records:
        by_concurrency.setdefault(record.concurrency, []).append(record)
    result: dict[str, list[BoundFamilyEntry]] = {}
    for concurrency in sorted(by_concurrency):
        conc_records = by_concurrency[concurrency]
        decode_ranked = _rank_pool_role_families(
            [record for record in conc_records if record.role == "decode"]
        )
        prefill_ranked = _rank_pool_role_families(
            [record for record in conc_records if record.role == "prefill"]
        )
        result[str(concurrency)] = [*decode_ranked, *prefill_ranked]
    return result


def build_bound_families_by_concurrency(
    plan: ServingPlan,
) -> dict[str, list[BoundFamilyEntry]]:
    """Group and rank layout families per pool by active lower bound per concurrency."""
    if not plan.candidates:
        pool_records = (*plan.decode_pools, *plan.prefill_pools)
        if not pool_records:
            return {}
        return _build_pool_families_by_concurrency(pool_records)
    accept_len = max(1.0, plan.scenario.accept_len)
    by_concurrency: dict[int, list[ParetoPoint]] = {}
    for point in plan.candidates:
        by_concurrency.setdefault(point.concurrency, []).append(point)
    result: dict[str, list[BoundFamilyEntry]] = {}
    for concurrency in sorted(by_concurrency):
        grouped: dict[tuple[object, ...], list[ParetoPoint]] = {}
        labels: dict[tuple[object, ...], tuple[str, ...]] = {}
        for point in by_concurrency[concurrency]:
            key = _bound_family_key(point)
            grouped.setdefault(key, []).append(point)
            labels.setdefault(key, _bound_family_label(point))
        summaries = [
            _summarize_bound_family(labels[key], group, accept_len=accept_len)
            for key, group in grouped.items()
        ]
        summaries.sort(key=lambda item: item.sort_key)
        ranked_entries: list[BoundFamilyEntry] = []
        for rank, item in enumerate(summaries, start=1):
            entry: BoundFamilyEntry = {**item.payload, "rank": rank}
            ranked_entries.append(entry)
        result[str(concurrency)] = ranked_entries
    return result


def _parse_single_pool_layout(pool_text: str) -> dict[str, int]:
    """Parse one pool layout string (e.g. 'TP8 x EP8') into an axis dictionary."""
    stripped_pool_text = pool_text.strip()
    if not stripped_pool_text:
        msg = f"empty layout string: {pool_text!r}"
        raise ValueError(msg)
    raw_tokens = re.split(r"\s*[xX]\s*", stripped_pool_text)
    seen_axes: set[str] = set()
    axis_values: dict[str, int] = {key: 1 for _, key in _POOL_AXES}
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


def _pool_axis_key(pool: PoolMeshPlan | None) -> tuple[int, ...] | None:
    """Return axis-only tuple matching _candidate_matches_layout granularity."""
    if pool is None:
        return None
    return (pool.tp, pool.dp, pool.ep, pool.pp, pool.pcp, pool.kv_split)


def _point_axis_layout_key(
    point: ParetoPoint,
) -> tuple[str, tuple[int, ...] | None, tuple[int, ...] | None]:
    """Return axis-only (mode, prefill_axes, decode_axes) key for peer grouping."""
    return (
        point.mode,
        _pool_axis_key(point.prefill_pool),
        _pool_axis_key(point.decode_pool),
    )


def _best_values_by_non_matching_layout(
    candidates: Sequence[ParetoPoint],
    parsed_layout: Mapping[str, Mapping[str, int]],
    value_getter: Callable[[ParetoPoint], float],
    *,
    lower_is_better: bool,
) -> list[float]:
    """Return best metric value per distinct axis layout for non-matching peers."""
    best_by_layout_key: dict[
        tuple[str, tuple[int, ...] | None, tuple[int, ...] | None],
        float,
    ] = {}
    for candidate in candidates:
        if not _candidate_matches_layout(candidate, parsed_layout):
            layout_key = _point_axis_layout_key(candidate)
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
) -> ReferenceLayoutReport:
    """Compile reference layout ranking report against same-concurrency peers."""
    parsed_layout = parse_layout(layout_text)
    if "prefill" in parsed_layout:
        normalized_layout = (
            f"P: {format_parallelism(parsed_layout['prefill'])} / "
            f"D: {format_parallelism(parsed_layout['decode'])}"
        )
    else:
        normalized_layout = format_parallelism(parsed_layout["decode"])

    accept_len = max(1.0, plan.scenario.accept_len)
    matching_candidates = [
        candidate
        for candidate in plan.candidates
        if _candidate_matches_layout(candidate, parsed_layout)
    ]
    found = len(matching_candidates) > 0

    if found:
        best_candidate = min(
            matching_candidates,
            key=lambda candidate: candidate_sort_key(candidate, plan.sort_by),
        )
        ruled_out = effective_slo_verdict(best_candidate) == SLO_VERDICT_RULED_OUT
        peer_candidates = [
            candidate
            for candidate in plan.candidates
            if candidate.concurrency == best_candidate.concurrency
        ]
        matching_same_concurrency = [
            candidate
            for candidate in peer_candidates
            if _candidate_matches_layout(candidate, parsed_layout)
        ]
        reference_bound = min(
            _point_tpot_floor_ms(candidate, accept_len=accept_len)
            for candidate in matching_same_concurrency
        )
        other_bounds = _best_values_by_non_matching_layout(
            peer_candidates,
            parsed_layout,
            lambda candidate: _point_tpot_floor_ms(candidate, accept_len=accept_len),
            lower_is_better=True,
        )
        num_candidates = 1 + len(other_bounds)
        rank_by_tpot_bound: int | None = strict_rank(
            reference_bound, other_bounds, lower_is_better=True
        )
        reference_throughput = max(
            candidate.token_throughput_per_chip
            for candidate in matching_same_concurrency
        )
        other_throughputs = _best_values_by_non_matching_layout(
            peer_candidates,
            parsed_layout,
            lambda candidate: candidate.token_throughput_per_chip,
            lower_is_better=False,
        )
        rank_by_estimated_throughput: int | None = strict_rank(
            reference_throughput, other_throughputs, lower_is_better=False
        )
        in_top_k = any(
            _candidate_matches_layout(candidate, parsed_layout)
            for candidate in top_points
        )
        reference_concurrency: int | None = best_candidate.concurrency
        candidate_point_dict: Mapping[str, object] | None = asdict(best_candidate)
        final_rejection_reason = ""
    else:
        num_candidates = len(
            {_point_axis_layout_key(candidate) for candidate in plan.candidates}
        )
        ruled_out = False
        reference_concurrency = None
        rank_by_tpot_bound = None
        rank_by_estimated_throughput = None
        in_top_k = False
        candidate_point_dict = None
        final_rejection_reason = rejected_by

    return {
        "layout": normalized_layout,
        "found": found,
        "ruled_out": ruled_out,
        "concurrency": reference_concurrency,
        "candidates": num_candidates,
        "rank_by_tpot_bound": rank_by_tpot_bound,
        "rank_by_estimated_throughput": rank_by_estimated_throughput,
        "in_top_k": in_top_k,
        "point": candidate_point_dict,
        "rejected_by": final_rejection_reason,
    }


def attach_reference_report(
    payload: SolvePayload,
    reference: Mapping[str, object],
) -> SolvePayload:
    """Attach a physics reference-layout report to a solve payload."""
    updated = dict(payload)
    updated["reference"] = dict(reference)
    return cast("SolvePayload", updated)
