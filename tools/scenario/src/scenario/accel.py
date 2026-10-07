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
"""What a slice `<accelerator>/<topology>` provides.

Counts are chips, never TensorCores. GKE machine types and queue names count
TensorCores (`tpu7x-8` is one 2x2x1 host with four chips), so neither is used
here. An accelerator or topology is added with the first lane that runs it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Accelerator:
    name: str
    # Devices vLLM sees per chip: a v7x chip shows its two TensorCores.
    devices_per_chip: int
    # topology -> (hosts, chips per host)
    topologies: Mapping[str, tuple[int, int]]


ACCELERATORS: Mapping[str, Accelerator] = MappingProxyType(
    {
        "tpu7x": Accelerator(
            name="tpu7x",
            devices_per_chip=2,
            topologies=MappingProxyType(
                {
                    "1x1x1": (1, 1),
                    "2x2x1": (1, 4),
                    "2x2x2": (2, 4),
                    "2x2x4": (4, 4),
                    "2x4x4": (8, 4),
                    "4x4x4": (16, 4),
                }
            ),
        ),
    }
)


@dataclass(frozen=True)
class SliceShape:
    accelerator: str
    topology: str
    hosts: int
    chips_per_host: int
    devices_per_host: int

    @property
    def chips(self) -> int:
        return self.hosts * self.chips_per_host

    @property
    def devices(self) -> int:
        return self.hosts * self.devices_per_host

    def __str__(self) -> str:
        return f"{self.accelerator}/{self.topology}"


class SliceError(ValueError):
    """A slice that is malformed or not in the table."""


def parse_slice(text: str) -> SliceShape:
    accelerator, sep, topology = text.partition("/")
    if not sep or not accelerator or not topology:
        raise SliceError(
            f"a slice is <accelerator>/<topology>, e.g. tpu7x/2x2x2; got {text!r}"
        )
    accel = ACCELERATORS.get(accelerator)
    if accel is None:
        raise SliceError(
            f"unknown accelerator {accelerator!r}; known: {', '.join(ACCELERATORS)}"
        )
    shape = accel.topologies.get(topology)
    if shape is None:
        raise SliceError(
            f"{accelerator} has no topology {topology!r}; "
            f"known: {', '.join(accel.topologies)}"
        )
    hosts, chips_per_host = shape
    return SliceShape(
        accelerator=accelerator,
        topology=topology,
        hosts=hosts,
        chips_per_host=chips_per_host,
        devices_per_host=chips_per_host * accel.devices_per_chip,
    )


def known_slices() -> tuple[str, ...]:
    return tuple(
        f"{accel.name}/{topology}"
        for accel in ACCELERATORS.values()
        for topology in accel.topologies
    )
