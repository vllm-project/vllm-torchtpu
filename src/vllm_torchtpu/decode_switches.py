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
"""Which decode paths are on, resolved from the environment.

A path is a change whose only effect is how fast it runs. Each belongs to one
parent switch, which turns it on; ``TPU_DECODE_FAST_PATHS_OFF`` (a comma list
of path names) turns single paths off again. ``PATHS`` is the one list of names.

Imports only ``envs``, so the resolution runs without a device.
"""

import dataclasses
import os
from collections.abc import Mapping

from vllm_torchtpu import envs

VERIFY_BLOCK = "TPU_BATCHED_RPA_VERIFY_BLOCK"
FAST_PATHS = "TPU_DECODE_FAST_PATHS"
FAST_PATHS_OFF = "TPU_DECODE_FAST_PATHS_OFF"


@dataclasses.dataclass(frozen=True)
class Path:
    """A path: its name in the off list, the variable that turns it on, and
    what it does."""

    name: str
    parent: str
    what: str


PATHS: tuple[Path, ...] = (
    Path(
        "draft_one_query",
        VERIFY_BLOCK,
        "draft passes of one query per sequence run the one-query decode kernel",
    ),
    Path(
        "xla_schedule",
        FAST_PATHS,
        "batched RPA builds its decode schedules with array ops",
    ),
)
PATH_NAMES = tuple(p.name for p in PATHS)


def _flag(environ: Mapping[str, str], name: str) -> bool:
    """The boolean value of ``name``; false when unset or empty."""
    value = environ.get(name)
    if value is None or value.strip() == "":
        return False
    lowered = value.strip().lower()
    if lowered in envs._TRUE_STRINGS:
        return True
    if lowered in envs._FALSE_STRINGS:
        return False
    raise ValueError(
        f"Invalid boolean value '{value}' for {name}. "
        f"Valid options: {envs._TRUE_STRINGS + envs._FALSE_STRINGS} (any case)."
    )


def _off_list(environ: Mapping[str, str]) -> frozenset[str]:
    names = [n.strip() for n in environ.get(FAST_PATHS_OFF, "").split(",")]
    names = [n for n in names if n]
    unknown = [n for n in names if n not in PATH_NAMES]
    if unknown:
        raise ValueError(
            f"{FAST_PATHS_OFF} names unknown path(s) {', '.join(unknown)}; "
            f"valid paths: {', '.join(PATH_NAMES)}."
        )
    return frozenset(names)


def _parent_on(environ: Mapping[str, str], parent: str) -> bool:
    if parent == VERIFY_BLOCK:
        return bool(environ.get(parent, "").strip())
    return _flag(environ, parent)


@dataclasses.dataclass(frozen=True)
class Switches:
    """The resolved switches: the paths that are on."""

    paths: frozenset[str]

    def path(self, name: str) -> bool:
        if name not in PATH_NAMES:
            raise KeyError(f"unknown path {name!r}; valid paths: {PATH_NAMES}")
        return name in self.paths


def resolve(environ: Mapping[str, str]) -> Switches:
    """The switches ``environ`` sets. Refuses an unknown name in the off list."""
    off = _off_list(environ)
    paths = frozenset(
        p.name for p in PATHS if p.name not in off and _parent_on(environ, p.parent)
    )
    return Switches(paths=paths)


def get() -> Switches:
    """The switches of the process environment."""
    return resolve(os.environ)
