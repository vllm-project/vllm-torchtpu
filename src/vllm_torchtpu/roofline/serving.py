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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, Unpack

from .chips import topologies_for_chip
from .constants import UNIT_MICRO, UNIT_MILLIS_PER_SECOND
from .modules import Parallel, StepShape
from .step import FloorOptions, SpecShape, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .chips import ChipSpec, SliceTopology
    from .geometry import ModelGeometry
    from .modules import Dtypes

_MIN_ACCEPTANCE_LENGTH = 1.0


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

    concurrency: int
    rows: int
    binding: str
    hard_floor_ms: float
    serial_estimate_ms: float
    tpot_floor_ms: float
    interactivity_ceil_tok_s: float
    output_tput_ceil_per_chip: float
    flag: str


def _resolve_topology(
    total_ranks: int,
    ranks_per_chip: int,
    *,
    chip: ChipSpec,
    topologies: Mapping[str, SliceTopology] | None = None,
) -> SliceTopology | None:
    """Return the smallest chip topology accommodating total_ranks, or None."""
    eff_ranks = max(1, ranks_per_chip)
    needed_chips = math.ceil(float(max(1, total_ranks)) / float(eff_ranks))
    topo_map = topologies if topologies is not None else topologies_for_chip(chip)
    for topology in sorted(
        topo_map.values(), key=lambda t: (t.num_chips, t.num_cores, t.name)
    ):
        if topology.num_chips >= needed_chips and topology.num_cores >= total_ranks:
            return topology
    return None


class _PlanOptions(TypedDict, total=False):
    """Optional keyword arguments for :func:`plan`."""

    topologies: Mapping[str, SliceTopology] | None


def plan(
    geometry: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    dtypes: Dtypes,
    request: ServingRequest,
    **options: Unpack[_PlanOptions],
) -> list[ConcurrencyRow]:
    """Compute ConcurrencyRow ceilings across a list of concurrencies."""
    topologies = options.get("topologies")
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else chip.cores_per_chip
    )
    total_ranks = max(
        1,
        parallel.tp * parallel.dp * parallel.pp * parallel.pcp,
        parallel.ep * parallel.pp,
    )
    chips = float(total_ranks) / float(eff_ranks)
    if request.spec_window < 0:
        msg = f"spec_window must be >= 0, got {request.spec_window}"
        raise ValueError(msg)
    if request.spec_window == 0:
        if (
            request.acceptance_length is not None
            and float(request.acceptance_length) != _MIN_ACCEPTANCE_LENGTH
        ):
            msg = (
                "acceptance_length must be 1.0 when spec_window == 0, "
                f"got {request.acceptance_length}"
            )
            raise ValueError(msg)
        tokens = _MIN_ACCEPTANCE_LENGTH
    elif request.acceptance_length is None:
        msg = "acceptance_length is required when spec_window > 0"
        raise ValueError(msg)
    else:
        tokens = float(request.acceptance_length)
        max_tokens = float(request.spec_window + 1)
        if (
            not math.isfinite(tokens)
            or tokens < _MIN_ACCEPTANCE_LENGTH
            or tokens > max_tokens
        ):
            msg = (
                "acceptance_length must satisfy 1.0 <= acceptance_length "
                f"<= spec_window + 1 ({max_tokens}), got {tokens}"
            )
            raise ValueError(msg)
    lat_s = (request.collective_latency_us or 0.0) * UNIT_MICRO
    topology = _resolve_topology(
        total_ranks, eff_ranks, chip=chip, topologies=topologies
    )
    opts = FloorOptions(dtypes=dtypes, collective_latency_s=lat_s, topology=topology)
    rows_out: list[ConcurrencyRow] = []
    for concurrency in request.concurrencies:
        if concurrency <= 0:
            msg = f"concurrency must be positive (>= 1), got {concurrency}"
            raise ValueError(msg)
        if request.spec_window > 0:
            floor_result = spec_cycle_floor(
                geometry,
                chip,
                parallel,
                SpecShape(
                    batch=concurrency,
                    context=request.context,
                    spec_k=request.spec_window,
                ),
                opts,
            )
        else:
            floor_result = step_floor(
                geometry,
                chip,
                parallel,
                StepShape(batch=concurrency, rows_per_seq=1, context=request.context),
                opts,
            )
        hard_ms = floor_result.hard_floor_s * UNIT_MILLIS_PER_SECOND
        serial_ms = floor_result.serial_estimate_s * UNIT_MILLIS_PER_SECOND
        tpot_ms = hard_ms / tokens
        interactivity = UNIT_MILLIS_PER_SECOND / tpot_ms
        output_tput = (
            (concurrency * max(1, parallel.dp) * tokens)
            / floor_result.hard_floor_s
            / chips
        )
        flag = "MEASURE" if floor_result.binding in ("chain", "ici") else "-"
        rows_out.append(
            ConcurrencyRow(
                concurrency=concurrency,
                rows=concurrency * (request.spec_window + 1),
                binding=floor_result.binding,
                hard_floor_ms=hard_ms,
                serial_estimate_ms=serial_ms,
                tpot_floor_ms=tpot_ms,
                interactivity_ceil_tok_s=interactivity,
                output_tput_ceil_per_chip=output_tput,
                flag=flag,
            )
        )
    return rows_out
