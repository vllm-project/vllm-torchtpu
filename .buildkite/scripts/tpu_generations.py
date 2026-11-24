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
"""The TPU generations CI runs on, and the stamp that tells their steps apart.

A pipeline file in TPU_GENERATION_FILES is written once and goes up once per
TPU generation (bootstrap.py, upload_stages). Each upload reads the
generation's pod shapes as upload-time variables (TpuGeneration.upload_env)
and has the generation's name stamped on its steps (stamp_tpu_generation), so
the file names no generation, and adding one renames no step of the others.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TpuGeneration:
    """A TPU generation: its name and the shapes of its pods.

    The name also tells its steps apart: `key_prefix` goes before every step
    key (Buildkite rejects a duplicate key) and `label_prefix` before every
    label (GitHub names a commit status after its step's label).
    """

    name: str
    shape_single: str
    shape_multi: str

    @property
    def key_prefix(self) -> str:
        return f"{self.name}_"

    @property
    def label_prefix(self) -> str:
        return f"{self.name} "

    def upload_env(self) -> dict[str, str]:
        """The variables a step file reads to fill in SHAPE (either shape)."""
        return {
            "KUBE_SHAPE_SINGLE": self.shape_single,
            "KUBE_SHAPE_MULTI": self.shape_multi,
        }


# Each generation's shapes are kept here alone: upload_models_and_features.sh
# reads them for the nightly model and feature tests (`bootstrap.py tpu-shapes
# <name>`), and pipeline_tests.yml's defaults, for an upload of that file on its
# own, are v7x's; test_tpu_generations.py checks both.
TPU_V7X = TpuGeneration(
    name="v7x",
    shape_single="tpu7x-standard-1t/1x1x1",
    shape_multi="tpu7x-standard-4t/2x2x1",
)
TPU_V6E = TpuGeneration(
    name="v6e",
    shape_single="ct6e-standard-1t/1x1",
    shape_multi="ct6e-standard-8t/2x4",
)
TPU_GENERATIONS_BY_NAME: Mapping[str, TpuGeneration] = {
    gen.name: gen for gen in (TPU_V6E, TPU_V7X)
}
# Pipeline files uploaded once per TPU generation in the listed order (last
# shows/queues first in Buildkite). Files not listed here are uploaded once.
TPU_GENERATION_FILES: Mapping[str, tuple[TpuGeneration, ...]] = {
    ".buildkite/pipeline_tests.yml": (TPU_V6E, TPU_V7X),
}
# The generations those files go up for without TPU_ALL_EXTRA.
DEFAULT_TPU_GENERATIONS: tuple[TpuGeneration, ...] = (TPU_V7X,)
# The generation whose nightly tests-lane steps a verified commit pair needs
# passed. pipeline_verify.yml's nightly report waits on those steps by key, and
# nightly_report.sh asks for the key prefix (`bootstrap.py verified-key-prefix`).
VERIFIED_TPU_GENERATION = TPU_V7X
# The build extras (CI_EXTRAS, bootstrap.py) that change which generations
# those files go up for; a build names one at most. TPU_ALL_EXTRA sends them
# up for every generation they list, side by side, and TPU_V6E_EXTRA for v6e
# alone (not alongside v7x), which leaves out VERIFIED_TPU_GENERATION and so is
# refused on a build that runs the nightly report. The generation brings its
# shapes, so the build names no KUBE_SHAPE_* of its own.
TPU_ALL_EXTRA = "tpu-all"
TPU_V6E_EXTRA = "tpu-v6e"
TPU_GENERATION_EXTRAS = (TPU_ALL_EXTRA, TPU_V6E_EXTRA)


def picked_generations(
    generations: Sequence[TpuGeneration], extras: Collection[str]
) -> tuple[TpuGeneration, ...]:
    """The `generations` a file lists that a build with `extras` runs it on."""
    if TPU_ALL_EXTRA in extras:
        return tuple(generations)
    wanted = (TPU_V6E,) if TPU_V6E_EXTRA in extras else DEFAULT_TPU_GENERATIONS
    return tuple(gen for gen in generations if gen in wanted)


def _own_steps(steps: Sequence[Any]) -> list[dict[str, Any]]:
    """The step objects of `steps`, with each group followed by its steps."""
    found: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        found.append(step)
        if isinstance(step.get("steps"), list):
            found.extend(_own_steps(step["steps"]))
    return found


def stamp_tpu_generation(steps: Sequence[Any], generation: TpuGeneration) -> None:
    """Stamp `generation`'s key and label prefixes on a parsed pipeline's steps.

    `steps` is the `steps` of `pipeline upload --dry-run --format json`, edited
    in place. Every key the pipeline defines takes the key prefix, and so does
    a `depends_on` entry that names one of them; a dependency on a step of
    another upload (the image builds) is left as it is. Every label and group
    name takes the label prefix.

    Raises ValueError for a step whose key, label or depends_on is not of a
    form read here, since a step passed over would go up under another
    generation's name.
    """
    own = _own_steps(steps)
    for step in own:
        for field in ("key", "label", "group"):
            if field in step and not isinstance(step[field], str):
                raise ValueError(
                    f"cannot stamp TPU {generation.name} on a step whose "
                    f"`{field}` is {step[field]!r}"
                )
        # `id`, `identifier` and `name` are Buildkite's aliases of `key` and
        # `label`. The step files use neither, and one that did would keep its
        # other generation's name.
        for alias in ("id", "identifier", "name"):
            if alias in step:
                raise ValueError(
                    f"cannot stamp TPU {generation.name} on a step that "
                    f"uses `{alias}`; use `key` and `label`"
                )
    defined = {step["key"] for step in own if "key" in step}

    def stamped(dependency: Any) -> Any:
        if isinstance(dependency, str):
            if dependency in defined:
                return generation.key_prefix + dependency
            return dependency
        if isinstance(dependency, dict) and isinstance(dependency.get("step"), str):
            return {**dependency, "step": stamped(dependency["step"])}
        raise ValueError(
            f"cannot stamp TPU {generation.name} on depends_on entry {dependency!r}"
        )

    for step in own:
        if "key" in step:
            step["key"] = generation.key_prefix + step["key"]
        for field in ("label", "group"):
            if field in step:
                step[field] = generation.label_prefix + step[field]
        depends_on = step.get("depends_on")
        if isinstance(depends_on, list):
            step["depends_on"] = [stamped(d) for d in depends_on]
        elif depends_on is not None:
            step["depends_on"] = stamped(depends_on)
