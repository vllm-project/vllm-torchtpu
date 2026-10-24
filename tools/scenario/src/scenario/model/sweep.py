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
"""Server-side sweeps: one fresh server per point.

A sweep has one of three forms: named `points` with their own overlays,
`zip` axes that line up by position, or `product` axes. Resolution expands
it into points and validates each point as a scenario of its own.
"""

import itertools
import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, JsonValue

from scenario.model.baseline import PointBaselines
from scenario.model.common import KeyPathText, Model, PointName
from scenario.paths import KeyPath, PathSyntaxError, parse_path

AxisValues = Annotated[list[JsonValue], Field(min_length=1)]
Axes = Annotated[dict[KeyPathText, AxisValues], Field(min_length=1)]

POINT_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")
NAME_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


class SweepPoint(Model):
    """A named point: an overlay over the scenario for one server."""

    name: PointName = Field(description="Unique; results and baselines use it.")
    server: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Overlay on server, merged above the scenario for this point.",
    )
    tasks: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Overlay on tasks; also holds this point's baselines.",
    )


class Sweep(Model):
    """Points that each run on a fresh server. Exactly one of points, zip, product."""

    name: str | None = Field(
        default=None,
        description="Point name template for zip and product, e.g. "
        "c{tasks.agentx.concurrency}; each {path} is one of the axes.",
    )
    points: Annotated[list[SweepPoint], Field(min_length=1)] | None = Field(
        default=None, description="Named points, for irregular tables."
    )
    zip: Axes | None = Field(
        default=None,
        description="Dotted path -> values; equal lengths, point i takes value i "
        "of every axis.",
    )
    product: Axes | None = Field(
        default=None,
        description="Dotted path -> values; one point per combination, the "
        "first axis outermost.",
    )
    baselines: dict[PointName, PointBaselines] | None = Field(
        default=None,
        description="Baselines of zip and product points, by point name and "
        "task. Named points carry their own.",
    )

    @property
    def axes(self) -> dict[str, list[JsonValue]]:
        return dict(self.zip or self.product or {})


@dataclass(frozen=True)
class ExpandedPoint:
    """One point of a sweep: a named point's index, or an axis assignment."""

    name: str
    index: int
    assignments: tuple[tuple[KeyPath, JsonValue], ...]


def _name_value(value: JsonValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return str(value)
    raise ValueError(f"a {type(value).__name__} value cannot go into a point name")


def point_name(template: str, values: dict[KeyPath, JsonValue]) -> str:
    """Fill a point name template from one point's axis values.

    Raises:
        ValueError: a placeholder is not an axis, or its value is not a scalar.
    """

    def fill(match: re.Match[str]) -> str:
        text = match.group(1)
        try:
            path = parse_path(text)
        except PathSyntaxError:
            path = None
        if path not in values:
            raise ValueError(f"{{{text}}} is not one of the axes")
        return _name_value(values[path])

    return NAME_PLACEHOLDER.sub(fill, template)


def expand(sweep: Sweep) -> tuple[ExpandedPoint, ...]:
    """The points of a sweep that has passed validation, in run order."""
    if sweep.points is not None:
        return tuple(
            ExpandedPoint(name=point.name, index=index, assignments=())
            for index, point in enumerate(sweep.points)
        )
    axes = [(parse_path(text), values) for text, values in sweep.axes.items()]
    if sweep.zip is not None:
        rows = list(zip(*(values for _, values in axes)))
    else:
        rows = list(itertools.product(*(values for _, values in axes)))
    assert sweep.name is not None
    points = []
    for index, row in enumerate(rows):
        assignments = tuple(
            (path, value) for (path, _), value in zip(axes, row, strict=True)
        )
        points.append(
            ExpandedPoint(
                name=point_name(sweep.name, dict(assignments)),
                index=index,
                assignments=assignments,
            )
        )
    return tuple(points)
