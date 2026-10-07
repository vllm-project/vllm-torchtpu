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
"""Inline baselines, which gate the task they sit next to.

A baseline is either inline (keyed values) or a reference to another
scenario's inline baseline for the same task (`{from: <ref>}`). Baselines
are never inherited through `extends`.
"""

from typing import Annotated

from pydantic import ConfigDict, Discriminator, Field, JsonValue, Tag, model_validator

from scenario.model.common import Matches, Model, NonNegInt, PosInt, RefId, fail

NonNegFloat = Annotated[float, Field(ge=0)]

CellKey = Annotated[
    str,
    Matches(
        "baseline_key",
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        "baseline entries are keyed by cell key, e.g. isl8192_osl1024_c1024",
    ),
]
RunKey = Annotated[
    str,
    Matches(
        "baseline_key",
        r"[A-Za-z0-9][A-Za-z0-9._:-]*",
        "eval baseline entries are keyed by lm_eval task, with _c<N> per "
        "concurrency when lm_eval.concurrency is set",
    ),
]
EvalMetricKey = Annotated[
    str,
    Matches(
        "baseline_key",
        r"[\w@.-]+,[\w-]+",
        "eval metrics are lm_eval's '<metric>,<filter>' keys, e.g. "
        "exact_match,custom-extract",
    ),
]
MetricName = Annotated[
    str,
    Matches(
        "baseline_key",
        r"[A-Za-z_][\w.@-]*",
        "metric names are letters, digits, '_', '.', '@' and '-'",
    ),
]


class PerfCellBaseline(Model):
    """Gated values of one perf cell, in check_regression.py's terms."""

    completed: NonNegInt | None = Field(
        default=None,
        description="Requests completed; a run must complete at least 95% as many.",
    )
    num_prompts: PosInt | None = Field(
        default=None, description="Prompts sent at calibration; not gated."
    )
    median_ttft_ms: NonNegFloat | None = Field(
        default=None,
        description="Median time to first token, ms; gated only when present.",
    )
    median_tpot_ms: NonNegFloat | None = Field(
        default=None, description="Median time per output token, ms."
    )
    total_token_throughput: NonNegFloat | None = Field(
        default=None, description="Input and output tokens per second."
    )
    output_token_throughput: NonNegFloat | None = Field(
        default=None, description="Output tokens per second."
    )
    output_throughput: NonNegFloat | None = Field(
        default=None,
        description="Older name of output_token_throughput, still read by the gate.",
    )

    @model_validator(mode="after")
    def _has_a_metric(self) -> "PerfCellBaseline":
        if all(getattr(self, name) is None for name in type(self).model_fields):
            raise fail("baseline_empty", "a baseline entry needs at least one metric")
        return self


class _KeyedBaseline(Model):
    model_config = ConfigDict(extra="allow")

    source: dict[str, JsonValue] | None = Field(
        default=None,
        alias="_source",
        description="Where the numbers came from; documentation only, the gate "
        "ignores it.",
    )


class PerfBaseline(_KeyedBaseline):
    """Perf gate baseline: cell key -> gated values."""

    __pydantic_extra__: dict[CellKey, PerfCellBaseline] = Field(init=False)


class EvalBaseline(_KeyedBaseline):
    """Eval gate baseline: lm_eval run key -> {"<metric>,<filter>": value}."""

    __pydantic_extra__: dict[RunKey, dict[EvalMetricKey, float]] = Field(init=False)


class AgentxBaseline(_KeyedBaseline):
    """AgentX gate baseline: metric name -> value, for the point's concurrency."""

    __pydantic_extra__: dict[MetricName, float] = Field(init=False)


class BaselineRef(Model):
    """Gate on another scenario's inline baseline for the same task."""

    from_: RefId = Field(
        alias="from",
        description="Reference id of the scenario whose inline baseline for the "
        "same task (and, in a sweep, the same point) gates this one.",
    )


def _baseline_kind(value: object) -> str:
    return "ref" if isinstance(value, dict) and "from" in value else "inline"


PerfBaselineField = Annotated[
    Annotated[PerfBaseline, Tag("inline")] | Annotated[BaselineRef, Tag("ref")],
    Discriminator(_baseline_kind),
]
EvalBaselineField = Annotated[
    Annotated[EvalBaseline, Tag("inline")] | Annotated[BaselineRef, Tag("ref")],
    Discriminator(_baseline_kind),
]
AgentxBaselineField = Annotated[
    Annotated[AgentxBaseline, Tag("inline")] | Annotated[BaselineRef, Tag("ref")],
    Discriminator(_baseline_kind),
]


class PointBaselines(Model):
    """Baselines of one generated sweep point, by task."""

    perf: PerfBaselineField | None = Field(
        default=None, description="The point's tasks.perf baseline."
    )
    agentx: AgentxBaselineField | None = Field(
        default=None, description="The point's tasks.agentx baseline."
    )
    eval: EvalBaselineField | None = Field(
        default=None, description="The point's tasks.eval baseline."
    )
