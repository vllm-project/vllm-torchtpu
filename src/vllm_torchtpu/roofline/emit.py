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
"""Export physics candidate ceiling curves from solved serving plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .solver import ServingPlan


class CeilingRow(TypedDict):
    """Single physics candidate row exported by ``emit_ceiling_rows``."""

    est_interactivity_tok_s: float
    token_throughput_per_chip: float
    output_throughput_per_chip: float
    est_ttft_ms: float
    est_tpot_ms: float
    concurrency: int
    total_chips: int
    mode: str
    binding_unit: str
    slo_met: bool
    is_pareto: bool


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export ordered candidate rows covering every legal candidate in ``plan``."""
    return [
        {
            "est_interactivity_tok_s": point.est_interactivity_tok_s,
            "token_throughput_per_chip": point.token_throughput_per_chip,
            "output_throughput_per_chip": point.output_throughput_per_chip,
            "est_ttft_ms": point.est_ttft_ms,
            "est_tpot_ms": point.est_tpot_ms,
            "concurrency": point.concurrency,
            "total_chips": point.total_chips,
            "mode": point.mode,
            "binding_unit": point.binding,
            "slo_met": point.slo_met,
            "is_pareto": point.is_pareto,
        }
        for point in plan.candidates
    ]
