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
"""Fill a Plan's Refs for one host, from the same addresses the capture used.

The runtime that fills Refs on a real host comes later; this follows the Ref
table in the Plan contract, so the comparison sees what that host would run.
"""

import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

from scenario.plan.types import Command, Concat, EnvItems, Ref, Value

LOOPBACK = "127.0.0.1"
IMAGE = "<image>"


@dataclass(frozen=True)
class Placement:
    """Where a host runs and what it can see of the others."""

    # None for the chipless client host.
    role: str | None
    index: int
    # Plan role -> its hosts' IPv4 addresses and DNS names, in index order.
    ips: Mapping[str, tuple[str, ...]]
    names: Mapping[str, tuple[str | None, ...]]
    results_dir: str
    pulled_model_dir: str
    server_log: str


def fill(value: Value, at: Placement) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Concat):
        return "".join(fill(part, at) for part in value.parts)
    return _ref(value, at)


def _ref(ref: Ref, at: Placement) -> str:
    if ref.kind == "self_ip":
        assert at.role is not None
        return at.ips[at.role][at.index]
    if ref.kind == "head_ip":
        assert at.role is not None
        return at.ips[at.role][0]
    if ref.kind == "host_index":
        return str(at.index)
    if ref.kind == "role_head":
        assert ref.role is not None
        if ref.role == at.role and at.index == 0:
            return LOOPBACK
        name = at.names[ref.role][0]
        assert name is not None, f"role {ref.role} has no DNS name"
        return name
    if ref.kind == "role_ips":
        assert ref.role is not None
        return ref.sep.join(at.ips[ref.role])
    if ref.kind == "results_dir":
        return at.results_dir
    if ref.kind == "pulled_model_dir":
        pulled = pathlib.Path(at.pulled_model_dir)
        if (pulled / "config.json").is_file() or ref.fallback is None:
            return str(pulled)
        return ref.fallback
    if ref.kind == "server_log":
        return at.server_log
    if ref.kind == "image":
        return IMAGE
    raise ValueError(f"unknown Ref kind {ref.kind}")


def fill_argv(command: Command, at: Placement) -> tuple[str, ...]:
    return tuple(fill(value, at) for value in command.argv)


def fill_env(items: EnvItems, at: Placement) -> dict[str, str]:
    return {name: fill(value, at) for name, value in items}
