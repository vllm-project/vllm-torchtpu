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
"""Rendered manifests read back as a cluster and the pods would read them.

Each pod's env is the manifest's, with the downward API filled in: the
completion index and a pod IP. Every pod must then run exactly its part of
the Plan, placed by the runtime's own parser. Like pod_target.py it is
imported by name, so checks over a repository's own scenarios can use it.
"""

import string
from collections.abc import Iterator
from typing import Any

from ruamel.yaml import YAML
from scenario.plan import codec
from scenario.plan.types import Plan
from scenario.runtime.placement import Placement, PlacementFlags, place
from scenario.show import render_plan
from scenario.target import ENV_HOST_INDEX, ENV_PLAN, POD_ENV
from scenario.targets.kubernetes import COMPLETION_INDEX, host_name

Doc = dict[str, Any]


def load(text: str, **values: str) -> Doc:
    """The manifest as a document; `values` are substituted first, as `${NAME}`."""
    if values:
        text = string.Template(text).substitute(values)
    return YAML(typ="safe", pure=True).load(text)


def replicated_jobs(doc: Doc) -> dict[str, Doc]:
    return {rjob["name"]: rjob for rjob in doc["spec"]["replicatedJobs"]}


def pod_spec(rjob: Doc) -> Doc:
    return rjob["template"]["spec"]["template"]["spec"]


def container(rjob: Doc) -> Doc:
    (only,) = pod_spec(rjob)["containers"]
    return only


def env_of(rjob: Doc) -> dict[str, Any]:
    """Name -> value, or the valueFrom mapping for the downward API."""
    return {
        e["name"]: e.get("value", e.get("valueFrom")) for e in container(rjob)["env"]
    }


def pods(doc: Doc) -> Iterator[tuple[str, int, dict[str, str]]]:
    """(role, index, env) per pod, with the downward API filled in."""
    for number, (name, rjob) in enumerate(replicated_jobs(doc).items()):
        command = container(rjob)["command"]
        assert command == ["scenario", "run", "--role", name], command
        for index in range(rjob["template"]["spec"]["parallelism"]):
            env = {}
            for entry in container(rjob)["env"]:
                if "value" in entry:
                    env[entry["name"]] = entry["value"]
                    continue
                path = entry["valueFrom"]["fieldRef"]["fieldPath"]
                if path == COMPLETION_INDEX:
                    env[entry["name"]] = str(index)
                else:
                    assert path == "status.podIP", path
                    env[entry["name"]] = f"10.0.{number}.{index + 1}"
            yield name, index, env


def check_pods_run_the_plan(label: str, planned: Plan, doc: Doc, workload: str) -> None:
    """Every pod of `doc` runs its own part of `planned`, and nothing else.

    `workload` is the JobSet name the manifest's host names start with.
    """
    shape = planned.shape()
    found = [(role, index) for role, index, _ in pods(doc)]
    assert found == [
        (role.name, index) for role in shape.roles for index in range(role.hosts)
    ], label
    for role, index, env in pods(doc):
        assert codec.decode(env[ENV_PLAN]) == planned, label
        assert env[ENV_HOST_INDEX] == str(index), label
        placement = place(planned, PlacementFlags(role=role), env)
        assert placement == Placement(
            role=role,
            host_index=index,
            head_address=placement.head_address,
            advertise_ip=env["SCENARIO_ADVERTISE_IP"],
            role_hosts=placement.role_hosts,
        ), label
        if len(shape.roles) > 1:
            assert placement.role_hosts == {
                r.name: tuple(host_name(r.name, i, workload) for i in range(r.hosts))
                for r in shape.roles
            }, label
        rendered = render_plan(codec.decode(env[ENV_PLAN]), role, index)
        assert rendered == render_plan(planned, role, index), label
        assert {n for n in env if n.startswith("SCENARIO_")} <= set(POD_ENV), label
