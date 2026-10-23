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
"""Measured-vs-floor gap decomposition and strict sub-floor guardrail."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .chips import ChipSpec
from .step import StepFloor

_NAME_ALIASES: dict[str, str] = {
    "attention": "gqa",
    "attn": "gqa",
    "full_attention": "gqa",
    "rpa": "gqa",
    "linear_attention": "gdn",
    "mamba": "gdn",
    "routed_mlp": "moe",
    "fused_moe": "moe",
    "mlp": "mlp",
    "dense_mlp": "mlp",
    "ffn": "mlp",
    "head": "lm_head",
    "collective": "collectives",
    "collectives": "collectives",
    "allreduce": "collectives",
    "all-reduce": "collectives",
    "all_reduce": "collectives",
    "ar_sum": "collectives",
    "ar_res": "collectives",
    "all-gather": "collectives",
    "reduce-scatter": "collectives",
    "Step time": "step_ms",
    "step_time_ms": "step_ms",
}


@dataclass(frozen=True)
class ModuleGap:
    """Per-module analytical floor versus measured latency."""

    name: str
    hbm_ms: float
    mm_ms: float
    ici_ms: float
    floor_ms: float
    measured_ms: float
    gap_ms: float
    achieved_util: float
    gap_share: float
    status: str


@dataclass(frozen=True)
class StepGapReport:
    """Step-level and module-level comparison of measured latency against StepFloor."""

    modules: tuple[ModuleGap, ...]
    hard_floor_ms: float
    serial_floor_ms: float
    measured_step_ms: float
    unattributed_ms: float
    total_gap_ms: float
    achieved_step_util: float
    status: str
    unmapped_keys: tuple[str, ...] = ()


def _canonical_key(key: str) -> str:
    k = key.strip()
    return _NAME_ALIASES.get(k, _NAME_ALIASES.get(k.lower(), k.lower()))


def normalize_measured(
    raw: Mapping[str, object],
) -> tuple[dict[str, float], float | None]:
    """Extract canonical per-module latencies (ms) and optional total step_ms."""
    source: Mapping[str, object] = raw
    if "decode" in raw and isinstance(raw["decode"], Mapping):
        source = raw["decode"]
    elif "modules" in raw and isinstance(raw["modules"], Mapping):
        source = raw["modules"]

    mod_ms: dict[str, float] = {}
    step_ms: float | None = None
    for k, v in raw.items():
        if isinstance(v, (int, float)) and _canonical_key(str(k)) == "step_ms":
            step_ms = float(v)

    for k, v in source.items():
        if not isinstance(v, (int, float)):
            continue
        canon = _canonical_key(str(k))
        if canon == "step_ms":
            step_ms = float(v)
        else:
            mod_ms[canon] = mod_ms.get(canon, 0.0) + float(v)
    return mod_ms, step_ms


def _extract_unmapped_keys(
    raw: Mapping[str, object], known_names: set[str]
) -> tuple[str, ...]:
    source: Mapping[str, object] = raw
    if "decode" in raw and isinstance(raw["decode"], Mapping):
        source = raw["decode"]
    elif "modules" in raw and isinstance(raw["modules"], Mapping):
        source = raw["modules"]

    unmapped: list[str] = []
    for k, v in source.items():
        if not isinstance(v, (int, float)):
            continue
        canon = _canonical_key(str(k))
        if canon not in known_names and canon != "step_ms":
            unmapped.append(str(k))
    return tuple(unmapped)


def compare_measured(
    sf: StepFloor,
    chip: ChipSpec,
    measured: Mapping[str, object],
    *,
    collective_latency_s: float = 0.0,
) -> StepGapReport:
    """Compare a StepFloor against measured module/step latencies (ms).

    Flags any module or total step whose measured latency falls below the
    strict minimum-expert floor (recomputed with ``strict_hbm_bytes``) as
    ``SUB_FLOOR_MODEL_BUG``.
    """
    mod_map, explicit_step_ms = normalize_measured(measured)
    raw_rows: list[tuple[str, float, float, float, float, float, float]] = []
    positive_gap_sum = 0.0
    sum_measured_mods = 0.0

    comp_mods = [
        m
        for m in sf.modules
        if not (
            m.name == "embed"
            and m.hbm_bytes == 0.0
            and m.fp8_flops == 0.0
            and m.bf16_flops == 0.0
        )
    ]
    known_names = {m.name for m in comp_mods} | {"collectives"}
    unmapped_keys = _extract_unmapped_keys(measured, known_names)

    for m in comp_mods:
        hbm_ms = (m.hbm_bytes / chip.hbm_bw) * 1e3
        strict_hbm_ms = (m.strict_hbm_bytes / chip.hbm_bw) * 1e3
        mm_ms = (m.fp8_flops / chip.fp8_flops + m.bf16_flops / chip.bf16_flops) * 1e3
        ici_ms = (
            (m.ici_bytes / chip.ici_bw) * 1e3
            if chip.ici_bw > 0.0 and not math.isinf(chip.ici_bw)
            else 0.0
        )
        floor_ms = max(hbm_ms, mm_ms)
        strict_floor_ms = max(strict_hbm_ms, mm_ms)
        meas_ms = mod_map.get(m.name, 0.0)
        sum_measured_mods += meas_ms
        gap_ms = meas_ms - floor_ms
        if meas_ms > 0.0 and gap_ms > 0.0:
            positive_gap_sum += gap_ms
        raw_rows.append(
            (m.name, hbm_ms, mm_ms, ici_ms, floor_ms, strict_floor_ms, meas_ms)
        )

    total_hops = sum(m.chain_hops for m in sf.modules)
    chain_s = max(sf.chain_s, total_hops * collective_latency_s + sf.ici_s)
    coll_ici_ms = sf.ici_s * 1e3
    coll_floor_ms = chain_s * 1e3
    coll_meas_ms = mod_map.get("collectives", 0.0)
    sum_measured_mods += coll_meas_ms
    coll_gap_ms = coll_meas_ms - coll_floor_ms
    if coll_meas_ms > 0.0 and coll_gap_ms > 0.0:
        positive_gap_sum += coll_gap_ms
    raw_rows.append(
        (
            "collectives",
            0.0,
            0.0,
            coll_ici_ms,
            coll_floor_ms,
            coll_floor_ms,
            coll_meas_ms,
        )
    )

    measured_step_ms = (
        explicit_step_ms if explicit_step_ms is not None else sum_measured_mods
    )
    unattributed_ms = measured_step_ms - sum_measured_mods
    if unattributed_ms > 0.0:
        positive_gap_sum += unattributed_ms

    out_mods: list[ModuleGap] = []
    any_sub_floor = False
    for (
        name,
        hbm_ms,
        mm_ms,
        ici_ms,
        floor_ms,
        strict_floor_ms,
        meas_ms,
    ) in raw_rows:
        gap_ms = meas_ms - floor_ms
        if meas_ms <= 0.0:
            status = "UNMEASURED"
            util = 0.0
            share = 0.0
        else:
            util = floor_ms / meas_ms if meas_ms > 0.0 else 0.0
            share = (
                max(0.0, gap_ms) / positive_gap_sum if positive_gap_sum > 0.0 else 0.0
            )
            if meas_ms < strict_floor_ms:
                status = "SUB_FLOOR_MODEL_BUG"
                any_sub_floor = True
            else:
                status = "OK"
        out_mods.append(
            ModuleGap(
                name=name,
                hbm_ms=hbm_ms,
                mm_ms=mm_ms,
                ici_ms=ici_ms,
                floor_ms=floor_ms,
                measured_ms=meas_ms,
                gap_ms=gap_ms,
                achieved_util=util,
                gap_share=share,
                status=status,
            )
        )

    hard_floor_ms = max(sf.hard_floor_s, chain_s) * 1e3
    serial_floor_ms = (sf.serial_floor_s - sf.chain_s + chain_s) * 1e3
    total_gap_ms = measured_step_ms - hard_floor_ms
    step_util = hard_floor_ms / measured_step_ms if measured_step_ms > 0.0 else 0.0
    strict_step_hbm_ms = (
        sum(m.strict_hbm_bytes for m in sf.modules) / chip.hbm_bw
    ) * 1e3
    strict_hard_floor_ms = max(strict_step_hbm_ms, sf.mm_s * 1e3, coll_floor_ms)
    if measured_step_ms > 0.0 and measured_step_ms < strict_hard_floor_ms:
        any_sub_floor = True
    step_status = "SUB_FLOOR_MODEL_BUG" if any_sub_floor else "OK"
    return StepGapReport(
        modules=tuple(out_mods),
        hard_floor_ms=hard_floor_ms,
        serial_floor_ms=serial_floor_ms,
        measured_step_ms=measured_step_ms,
        unattributed_ms=unattributed_ms,
        total_gap_ms=total_gap_ms,
        achieved_step_util=step_util,
        status=step_status,
        unmapped_keys=unmapped_keys,
    )
