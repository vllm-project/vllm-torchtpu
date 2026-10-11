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
"""R-PLAN-3: no engine flag or env var is added because of another value.

Deleting an arg removes exactly its own flag from every host and changes
nothing else in the Plan. The args plumbing reads are excluded; each has a
test of its own in test_plan.py.
"""

import copy
import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest
from plan_fixtures import VARIANTS, Data, build
from scenario.plan.argv import render_arg
from scenario.plan.types import Plan, Value

PLUMBING_INPUTS = {
    "tensor-parallel-size",
    "data-parallel-size",
    "prefill-context-parallel-size",
    "pipeline-parallel-size",
    "load-format",
    "max-model-len",
    "api-server-count",
    "trust-remote-code",
    "enable-expert-parallel",
}


def _without(argv: Sequence[Value], tokens: Sequence[str]) -> tuple[Value, ...]:
    width = len(tokens)
    for start in range(len(argv) - width + 1):
        if tuple(argv[start : start + width]) == tuple(tokens):
            return (*argv[:start], *argv[start + width :])
    raise AssertionError(f"{tokens} not in {argv}")


def _strip(plan: Plan, roles: set[str], tokens: list[str]) -> Plan:
    new_roles = []
    for role in plan.roles:
        hosts = []
        for host in role.host_plans:
            if role.name in roles and host.serve is not None:
                serve = dataclasses.replace(
                    host.serve, argv=_without(host.serve.argv, tokens)
                )
                host = dataclasses.replace(host, serve=serve)
            hosts.append(host)
        new_roles.append(dataclasses.replace(role, host_plans=tuple(hosts)))
    return dataclasses.replace(plan, roles=tuple(new_roles))


def _deletions(data: Data) -> list[tuple[list[Any], set[str], str, Any]]:
    """(path to the arg map, roles it reaches, key, value) for each deletable arg."""
    server = data["server"]
    roles = server.get("roles") or {"server": {"args": {}}}
    found = []
    for key, value in server.get("args", {}).items():
        if key in PLUMBING_INPUTS:
            continue
        reached = {
            name for name, role in roles.items() if key not in role.get("args", {})
        }
        found.append((["server", "args"], reached, key, value))
    for name, role in (server.get("roles") or {}).items():
        for key, value in role.get("args", {}).items():
            if key not in PLUMBING_INPUTS and key not in server.get("args", {}):
                found.append((["server", "roles", name, "args"], {name}, key, value))
    return found


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_deleting_an_arg_removes_only_its_flag(
    name: str, variants: dict[str, Data]
) -> None:
    data = variants[name]
    base = build(data)
    deletions = _deletions(data)
    assert deletions, name
    for path, roles, key, value in deletions:
        changed = copy.deepcopy(data)
        node = changed
        for part in path:
            node = node[part]
        del node[key]
        assert build(changed) == _strip(base, roles, render_arg(key, value)), key
