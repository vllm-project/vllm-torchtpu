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
"""Per-concurrency TPOT floor, interactivity ceiling, and throughput ceiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import UNIT_MICROS_PER_SECOND, UNIT_MILLIS_PER_SECOND
from .modules import Parallel, StepShape
from .step import FloorOptions, SpecShape, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chips import ChipSpec
    from .geometry import ModelGeometry
    from .modules import Dtypes


@dataclass(frozen=True, slots=True)
class ServingRequest:
    """Workload sweep configuration for serving.plan."""

    context: int = 0
    concurrencies: Sequence[int] = ()
    spec_window: int = 0
    acceptance_length: float | None = None
    collective_latency_us: float | None = None


@dataclass(frozen=True, slots=True)
class ConcurrencyRow:
    """Serving ceiling row for one decode concurrency."""

    c: int
    rows: int
    binding: str
    hard_floor_ms: float
    serial_estimate_ms: float
    tpot_floor_ms: float
    interactivity_ceil_tok_s: float
    output_tput_ceil_per_chip: float
    flag: str


def plan(
    geom: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    dt: Dtypes,
    request: ServingRequest,
) -> list[ConcurrencyRow]:
    """Compute ConcurrencyRow ceilings across a list of concurrencies."""
    chips = max(1, parallel.tp * parallel.dp * parallel.pp * parallel.pcp) / float(
        parallel.ranks_per_chip
    )
    if request.spec_window == 0:
        tokens = 1.0
    elif request.acceptance_length is None:
        msg = "acceptance_length is required when spec_window > 0"
        raise ValueError(msg)
    else:
        tokens = float(request.acceptance_length)
    lat_s = (request.collective_latency_us or 0.0) * UNIT_MICROS_PER_SECOND
    opts = FloorOptions(dt=dt, collective_latency_s=lat_s)
    rows_out: list[ConcurrencyRow] = []
    for c in request.concurrencies:
        if request.spec_window > 0:
            sf = spec_cycle_floor(
                geom,
                chip,
                parallel,
                SpecShape(batch=c, context=request.context, k=request.spec_window),
                opts,
            )
        else:
            sf = step_floor(
                geom,
                chip,
                parallel,
                StepShape(batch=c, rows_per_seq=1, context=request.context),
                opts,
            )
        hard_ms = sf.hard_floor_s * UNIT_MILLIS_PER_SECOND
        serial_ms = sf.serial_estimate_s * UNIT_MILLIS_PER_SECOND
        tpot_ms = hard_ms / tokens
        interactivity = UNIT_MILLIS_PER_SECOND / tpot_ms
        output_tput = (c * tokens) / sf.hard_floor_s / chips
        flag = "MEASURE" if sf.binding in ("chain", "ici") else "-"
        rows_out.append(
            ConcurrencyRow(
                c=c,
                rows=c * (request.spec_window + 1),
                binding=sf.binding,
                hard_floor_ms=hard_ms,
                serial_estimate_ms=serial_ms,
                tpot_floor_ms=tpot_ms,
                interactivity_ceil_tok_s=interactivity,
                output_tput_ceil_per_chip=output_tput,
                flag=flag,
            )
        )
    return rows_out
