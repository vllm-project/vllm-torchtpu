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
"""KubernetesTarget: one point of a scenario as one JobSet.

The render is the tool's. Each role is a replicatedJob of Indexed pods, one
per host, with GKE's TPU node selectors and the role's chips; a chipless
driver adds a `client` replicatedJob. Every pod runs `scenario run --role R`
and learns its part from the pod contract env (target.POD_ENV).

A target subclasses KubernetesTarget and fills hooks. Each hook returns data
and none sees the JobSet: names, annotations, env and volumes per role, the
chipless pod's node selector and service account, failure policy,
after-tasks, and how to apply the manifest. render() and submit() are final.
After every hook, verify_unchanged() compares the JobSet with the
target-free render and fails, naming the hook, if anything changed that the
hook may not add to: what pods run, the Plan and its env, resources, node
selectors. It also fails a hook that sets an env name the tool or the
scenario owns, or one the target did not reserve.
"""

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, final

from scenario.plan import codec
from scenario.plan.types import Plan, RoleShape, Shape
from scenario.target import (
    ENV_ADVERTISE_IP,
    ENV_HEAD_ADDRESS,
    ENV_HOST_INDEX,
    ENV_IMAGE,
    ENV_PLAN,
    ENV_RESOLVED,
    ENV_ROLE_HOSTS,
    ENV_TARGET_DATA,
    TOOL_OWNED_ENV,
    AfterTask,
    CpuPod,
    EnvNames,
    JobPolicy,
    StepGroup,
    TargetData,
    TargetInfo,
    Volume,
    ownership_problems,
)

API_VERSION = "jobset.x-k8s.io/v1alpha2"
TPU_RESOURCE = "google.com/tpu"
ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
TPU_LABEL_PREFIX = "cloud.google.com/gke-tpu-"
# Gives each Job of a multi-host role a node pool to itself. A GKE TPU slice is
# one node pool, and the scheduler places pods one at a time, so without it two
# workloads admitted together can each take a host of the same slice and wait.
EXCLUSIVE_TOPOLOGY = "alpha.jobset.sigs.k8s.io/exclusive-topology"
NODE_POOL = "cloud.google.com/gke-nodepool"
COMPLETION_INDEX = "metadata.annotations['batch.kubernetes.io/job-completion-index']"
# GKE taints TPU nodes; a pod that asks for chips needs this to land on them.
TPU_TOLERATION = {
    "key": TPU_RESOURCE,
    "operator": "Equal",
    "value": "present",
    "effect": "NoSchedule",
}
# The chipless driver runs the HTTP client and the router. Requests equal
# limits, so it is Guaranteed on a node it shares.
CLIENT_RESOURCES = {"cpu": "6", "memory": "16Gi", "ephemeral-storage": "20Gi"}
DEFAULT_CONTAINER = "main"
FAIL_ON_EXIT = {
    "rules": [
        {"action": "FailJob", "onExitCodes": {"operator": "NotIn", "values": [0]}}
    ]
}

# Where the image and the workload name go, as text no scalar otherwise holds
# (Unicode private use). serialize() writes the hooks' raw text there after
# escaping, so a `${IMAGE}` that a launcher substitutes survives it.
IMAGE_REF = "\ue000IMAGE\ue001"
NAME_REF = "\ue000WORKLOAD_NAME\ue001"

HOOK_ORDER = (
    "annotations",
    "container_name",
    "env",
    "volumes",
    "cpu_pod",
    "job_policy",
)

_DNS_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUALIFIED = re.compile(
    r"([a-z0-9]([-a-z0-9.]*[a-z0-9])?/)?[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?"
)
_QUANTITY = re.compile(r"[0-9]+(\.[0-9]+)?([KMGTPE]i?|[mkMGTPE])?")
# Text written raw into a quoted YAML scalar.
_RAW_TEXT = re.compile(r"[^\s\"'\\\ue000\ue001]{1,253}")


class RenderError(ValueError):
    """A Plan this target cannot render, or a hook that broke a rule."""


def host_name(role: str, index: int, workload: str = NAME_REF) -> str:
    """The DNS name JobSet gives host `index` of `role` (job index 0)."""
    return f"{workload}-{role}-0-{index}.{workload}"


def default_workload_name(plan: Plan) -> str:
    """scn-<name slug>-<Plan digest>: the same Plan gives the same name."""
    slug = re.sub(r"[^a-z0-9]+", "-", plan.scenario_name.lower()).strip("-")
    slug = slug[:16].strip("-") or "scenario"
    digest = hashlib.sha256(codec.to_json(plan).encode()).hexdigest()[:8]
    return f"scn-{slug}-{digest}"


# The target-free render -----------------------------------------------------


def _json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def render_base(
    plan: Plan,
    *,
    resolved: Mapping[str, Any] | None,
    target_data: TargetData,
) -> dict[str, Any]:
    """The JobSet before any hook: what pods run, and the pod contract env."""
    roles = plan.shape().roles
    driver = next(role.name for role in roles if role.driver)
    encoded = codec.encode(plan)
    resolved_text = codec.encode_data(resolved) if resolved is not None else None
    data_text = codec.encode_data(target_data.to_data())
    role_hosts = None
    if len(roles) > 1:
        role_hosts = _json(
            {r.name: [host_name(r.name, i) for i in range(r.hosts)] for r in roles}
        )
    jobs = []
    for role in roles:
        env: list[dict[str, Any]] = [{"name": ENV_PLAN, "value": encoded}]
        if role.driver and resolved_text is not None:
            env.append({"name": ENV_RESOLVED, "value": resolved_text})
        env += [
            {
                "name": ENV_HOST_INDEX,
                "valueFrom": {"fieldRef": {"fieldPath": COMPLETION_INDEX}},
            },
            {
                "name": ENV_ADVERTISE_IP,
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            },
        ]
        if not role.chipless:
            env.append({"name": ENV_HEAD_ADDRESS, "value": host_name(role.name, 0)})
        if role_hosts is not None:
            env.append({"name": ENV_ROLE_HOSTS, "value": role_hosts})
        env.append({"name": ENV_IMAGE, "value": IMAGE_REF})
        if role.driver:
            env.append({"name": ENV_TARGET_DATA, "value": data_text})
        jobs.append(_replicated_job(role, multi=len(roles) > 1, env=env))
    return {
        "apiVersion": API_VERSION,
        "kind": "JobSet",
        "metadata": {"name": NAME_REF},
        "spec": {
            # A DNS name per pod under a headless service named for the
            # JobSet; host_name() builds them.
            "network": {"enableDNSHostnames": True},
            "successPolicy": {"operator": "All", "targetReplicatedJobs": [driver]},
            "replicatedJobs": jobs,
        },
    }


def _replicated_job(
    role: RoleShape, *, multi: bool, env: list[dict[str, Any]]
) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": DEFAULT_CONTAINER,
        "image": IMAGE_REF,
        "command": ["scenario", "run", "--role", role.name],
    }
    if role.ports:
        container["ports"] = [{"containerPort": port} for port in role.ports]
    if role.chipless:
        container["resources"] = {
            "requests": dict(CLIENT_RESOURCES),
            "limits": dict(CLIENT_RESOURCES),
        }
    else:
        # Only the chips: a TPU pod holds the whole host, and a cpu request
        # can keep a node pool from scaling up.
        chips = str(role.chips_per_host)
        container["resources"] = {
            "limits": {TPU_RESOURCE: chips},
            "requests": {TPU_RESOURCE: chips},
        }
    container["env"] = env
    pod: dict[str, Any] = {"restartPolicy": "Never"}
    if not role.chipless:
        pod["nodeSelector"] = {
            ACCELERATOR_LABEL: role.accelerator,
            TOPOLOGY_LABEL: role.topology,
        }
        pod["tolerations"] = [dict(TPU_TOLERATION)]
    pod["containers"] = [container]
    # Indexed: enableDNSHostnames and GKE's multi-host TPU placement need it.
    job: dict[str, Any] = {
        "parallelism": role.hosts,
        "completions": role.hosts,
        "completionMode": "Indexed",
    }
    if role.chipless:
        # A retried client would run the tasks again against the same servers.
        job["backoffLimit"] = 0
    job["template"] = {"spec": pod}
    template: dict[str, Any] = {}
    if not role.chipless and role.hosts > 1:
        template["metadata"] = {"annotations": {EXCLUSIVE_TOPOLOGY: NODE_POOL}}
    template["spec"] = job
    rjob: dict[str, Any] = {"name": role.name}
    if multi:
        # GKE's TPU webhook sets MEGASCALE_* from the group, so roles sharing
        # one would each wait for the other's slice as a second slice.
        rjob["groupName"] = role.name
    rjob["replicas"] = 1
    rjob["template"] = template
    return rjob


# Hooks: merge their data, then verify ----------------------------------------


@dataclass
class _Added:
    """What the hooks applied so far added, as they returned it."""

    annotations: dict[str, str] = field(default_factory=dict)
    container: str | None = None
    env: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    volumes: dict[str, list[Volume]] = field(default_factory=dict)
    cpu_pod: CpuPod | None = None
    policies: dict[str, JobPolicy] = field(default_factory=dict)
    max_restarts: int | None = None


@dataclass
class _Render:
    target: "KubernetesTarget"
    plan: Plan
    plan_json: str
    shape: Shape
    reserved: EnvNames
    base: dict[str, Any]
    tree: dict[str, Any]
    added: _Added = field(default_factory=_Added)
    applied: list[str] = field(default_factory=list)

    @property
    def who(self) -> str:
        return type(self.target).__name__

    def fail(self, hook: str, problems: Sequence[str]) -> None:
        if problems:
            raise RenderError(
                "; ".join(f"{self.who}.{hook}(): {problem}" for problem in problems)
            )

    def call(self, hook: str, *args: Any) -> Any:
        try:
            return getattr(self.target, hook)(*args)
        except RenderError:
            raise
        except Exception as error:
            raise RenderError(f"{self.who}.{hook}() raised {error!r}") from error

    def items(self, hook: str, *args: Any) -> list[Any]:
        """A hook's list, or a failure naming the hook."""
        found = self.call(hook, *args)
        if isinstance(found, str | bytes) or not isinstance(found, Sequence):
            self.fail(hook, [f"returns {found!r}, not a list"])
        return list(found)

    def jobs(self) -> list[tuple[RoleShape, dict[str, Any]]]:
        by_name = {rjob["name"]: rjob for rjob in self.tree["spec"]["replicatedJobs"]}
        return [(role, by_name[role.name]) for role in self.shape.roles]


def _job(rjob: dict[str, Any]) -> dict[str, Any]:
    return rjob["template"]["spec"]


def _pod(rjob: dict[str, Any]) -> dict[str, Any]:
    return rjob["template"]["spec"]["template"]["spec"]


def _volume(volume: Volume) -> dict[str, Any]:
    empty: dict[str, Any] = {}
    if volume.medium:
        empty["medium"] = volume.medium
    if volume.size_limit is not None:
        empty["sizeLimit"] = volume.size_limit
    return {"name": volume.name, "emptyDir": empty}


def _mount(volume: Volume) -> dict[str, Any]:
    return {"name": volume.name, "mountPath": volume.mount_path}


def _merge_annotations(r: _Render) -> None:
    data = r.call("annotations")
    if not isinstance(data, Mapping):
        r.fail("annotations", [f"returns {data!r}, not a mapping"])
    r.added.annotations = dict(data)
    if data:
        r.tree["metadata"].setdefault("annotations", {}).update(data)


def _merge_container_name(r: _Render) -> None:
    name = r.call("container_name")
    r.added.container = name
    for _, rjob in r.jobs():
        for container in _pod(rjob)["containers"]:
            container["name"] = name


def _merge_env(r: _Render) -> None:
    for role, rjob in r.jobs():
        data = r.call("env", role)
        if not isinstance(data, Mapping):
            r.fail("env", [f"returns {data!r} for role {role.name}, not a mapping"])
        entries = [{"name": name, "value": value} for name, value in data.items()]
        r.added.env[role.name] = copy.deepcopy(entries)
        for container in _pod(rjob)["containers"]:
            container["env"].extend(copy.deepcopy(entries))


def _merge_volumes(r: _Render) -> None:
    for role, rjob in r.jobs():
        volumes = r.items("volumes", role)
        bad = [v for v in volumes if not isinstance(v, Volume)]
        if bad:
            r.fail(
                "volumes", [f"returns {bad[0]!r} for role {role.name}, not a Volume"]
            )
        r.added.volumes[role.name] = volumes
        if not volumes:
            continue
        pod = _pod(rjob)
        pod.setdefault("volumes", []).extend(_volume(v) for v in volumes)
        for container in pod["containers"]:
            container.setdefault("volumeMounts", []).extend(_mount(v) for v in volumes)


def _merge_cpu_pod(r: _Render) -> None:
    data = r.call("cpu_pod")
    if data is None:
        return
    if not isinstance(data, CpuPod):
        r.fail("cpu_pod", [f"returns {data!r}, not a CpuPod or None"])
    r.added.cpu_pod = data
    for role, rjob in r.jobs():
        if not role.chipless:
            continue
        pod = _pod(rjob)
        if data.service_account is not None:
            pod["serviceAccountName"] = data.service_account
        if data.node_selector:
            pod.setdefault("nodeSelector", {}).update(data.node_selector)


def _merge_job_policy(r: _Render) -> None:
    restarts: set[Any] = set()
    for role, rjob in r.jobs():
        policy = r.call("job_policy", role)
        if not isinstance(policy, JobPolicy):
            r.fail("job_policy", [f"returns {policy!r}, not a JobPolicy"])
        r.added.policies[role.name] = policy
        job = _job(rjob)
        if policy.backoff_limit is not None and not role.chipless:
            job["backoffLimit"] = policy.backoff_limit
        if policy.fail_on_exit:
            job["podFailurePolicy"] = copy.deepcopy(FAIL_ON_EXIT)
        if policy.max_restarts is not None:
            restarts.add(policy.max_restarts)
    if len(restarts) > 1:
        r.fail(
            "job_policy",
            [f"roles disagree on the JobSet's max_restarts: {sorted(restarts)}"],
        )
    if restarts:
        r.added.max_restarts = restarts.pop()
        r.tree["spec"]["failurePolicy"] = {"maxRestarts": r.added.max_restarts}


MERGES: dict[str, Callable[[_Render], None]] = {
    "annotations": _merge_annotations,
    "container_name": _merge_container_name,
    "env": _merge_env,
    "volumes": _merge_volumes,
    "cpu_pod": _merge_cpu_pod,
    "job_policy": _merge_job_policy,
}


# verify_unchanged -----------------------------------------------------------


def _annotation_problems(r: _Render) -> list[str]:
    base = r.base["metadata"].get("annotations", {})
    problems = []
    for key, value in r.added.annotations.items():
        if not isinstance(key, str) or not _QUALIFIED.fullmatch(key):
            problems.append(f"annotation key {key!r} is not a qualified name")
        elif not isinstance(value, str):
            problems.append(f"annotation {key} is {value!r}, not a string")
        elif key in base:
            problems.append(f"annotation {key} is the tool's")
    return problems


def _container_problems(r: _Render) -> list[str]:
    name = r.added.container
    if not isinstance(name, str) or not _DNS_LABEL.fullmatch(name):
        return [f"container name {name!r} is not a DNS label"]
    return []


def _env_problems(r: _Render) -> list[str]:
    problems = []
    for role in r.shape.roles:
        seen: set[str] = set()
        for entry in r.added.env.get(role.name, []):
            name, value = entry["name"], entry["value"]
            where = f"on role {role.name}"
            if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
                problems.append(f"env name {name!r} {where} is not an env name")
                continue
            if not isinstance(value, str):
                problems.append(f"sets {name} {where} to {value!r}, not a string")
            elif "$(" in value:
                problems.append(
                    f"sets {name} {where} to {value!r}: Kubernetes expands $(...)"
                )
            if name in seen:
                problems.append(f"sets {name} twice {where}")
            seen.add(name)
            if TOOL_OWNED_ENV.covers(name):
                problems.append(f"sets {name} {where}, which the tool sets")
            elif not r.reserved.covers(name):
                problems.append(
                    f"sets {name} {where}, which its reserved_env() does not list"
                )
    return problems


def _volume_problems(r: _Render) -> list[str]:
    problems = []
    for role in r.shape.roles:
        names: set[str] = set()
        paths: set[str] = set()
        for volume in r.added.volumes.get(role.name, []):
            where = f"volume {volume.name!r} on role {role.name}"
            if not isinstance(volume.name, str) or not _DNS_LABEL.fullmatch(
                volume.name
            ):
                problems.append(f"{where}: the name is not a DNS label")
            if volume.name in names:
                problems.append(f"{where}: the name is used twice")
            names.add(volume.name)
            path = volume.mount_path
            if not isinstance(path, str) or not path.startswith("/"):
                problems.append(f"{where}: mount path {path!r} is not absolute")
            elif path in paths:
                problems.append(f"{where}: mount path {path} is used twice")
            paths.add(path)
            if volume.medium not in ("Memory", ""):
                problems.append(f"{where}: medium {volume.medium!r}")
            size = volume.size_limit
            if size is not None and (
                not isinstance(size, str) or not _QUANTITY.fullmatch(size)
            ):
                problems.append(f"{where}: size limit {size!r} is not a quantity")
            if role.chipless and volume.medium == "Memory" and size is None:
                problems.append(
                    f"{where}: a memory volume on a chipless pod needs a size "
                    "limit, or it can take the node's memory"
                )
    return problems


def _cpu_pod_problems(r: _Render) -> list[str]:
    data = r.added.cpu_pod
    if data is None:
        return []
    problems = []
    for key, value in dict(data.node_selector).items():
        if not isinstance(key, str) or not _QUALIFIED.fullmatch(key):
            problems.append(f"node selector key {key!r} is not a label key")
        elif key.startswith(TPU_LABEL_PREFIX):
            problems.append(f"selects {key} for a pod that holds no chips")
        if not isinstance(value, str):
            problems.append(f"node selector {key} is {value!r}, not a string")
    account = data.service_account
    if account is not None and (
        not isinstance(account, str) or not _DNS_LABEL.fullmatch(account)
    ):
        problems.append(f"service account {account!r} is not a DNS label")
    return problems


def _policy_problems(r: _Render) -> list[str]:
    problems = []
    for role in r.shape.roles:
        policy = r.added.policies.get(role.name)
        if policy is None:
            continue
        for what in ("backoff_limit", "max_restarts"):
            value = getattr(policy, what)
            if value is not None and (type(value) is not int or value < 0):
                problems.append(f"{what} {value!r} for role {role.name}")
        if role.chipless and policy.backoff_limit not in (None, 0):
            problems.append(
                f"backoff_limit {policy.backoff_limit} on the chipless "
                f"{role.name}: a retried driver reruns its tasks"
            )
        if not isinstance(policy.fail_on_exit, bool):
            problems.append(f"fail_on_exit {policy.fail_on_exit!r}")
    return problems


RULES: dict[str, Callable[[_Render], list[str]]] = {
    "annotations": _annotation_problems,
    "container_name": _container_problems,
    "env": _env_problems,
    "volumes": _volume_problems,
    "cpu_pod": _cpu_pod_problems,
    "job_policy": _policy_problems,
}


def _strip_tail(holder: dict[str, Any], key: str, entries: list[Any]) -> None:
    items = holder.get(key)
    if not entries or not isinstance(items, list) or len(items) < len(entries):
        return
    if items[-len(entries) :] == entries:
        del items[-len(entries) :]
        if not items:
            del holder[key]


def _without_additions(r: _Render) -> dict[str, Any]:
    """The JobSet with exactly what the hooks returned taken out again."""
    tree = copy.deepcopy(r.tree)
    added = r.added
    meta = tree.get("metadata", {})
    annotations = meta.get("annotations")
    if isinstance(annotations, dict) and "annotations" not in r.base["metadata"]:
        for key, value in added.annotations.items():
            if annotations.get(key) == value:
                del annotations[key]
        if not annotations:
            del meta["annotations"]
    spec = tree.get("spec", {})
    if added.max_restarts is not None and spec.get("failurePolicy") == {
        "maxRestarts": added.max_restarts
    }:
        del spec["failurePolicy"]
    chipless = {role.name for role in r.shape.roles if role.chipless}
    for rjob in spec.get("replicatedJobs", []):
        name = rjob.get("name")
        try:
            job, pod = _job(rjob), _pod(rjob)
        except (KeyError, TypeError):
            continue
        volumes = added.volumes.get(name, [])
        for container in pod.get("containers", []):
            if added.container is not None and container.get("name") == added.container:
                container["name"] = DEFAULT_CONTAINER
            _strip_tail(container, "env", added.env.get(name, []))
            _strip_tail(container, "volumeMounts", [_mount(v) for v in volumes])
        _strip_tail(pod, "volumes", [_volume(v) for v in volumes])
        cpu = added.cpu_pod
        if cpu is not None and name in chipless:
            if (
                cpu.service_account is not None
                and pod.get("serviceAccountName") == cpu.service_account
            ):
                del pod["serviceAccountName"]
            selector = pod.get("nodeSelector")
            if isinstance(selector, dict):
                for key, value in dict(cpu.node_selector).items():
                    if selector.get(key) == value:
                        del selector[key]
                if not selector:
                    del pod["nodeSelector"]
        policy = added.policies.get(name)
        if policy is not None:
            if (
                policy.backoff_limit is not None
                and name not in chipless
                and job.get("backoffLimit") == policy.backoff_limit
            ):
                del job["backoffLimit"]
            if policy.fail_on_exit and job.get("podFailurePolicy") == FAIL_ON_EXIT:
                del job["podFailurePolicy"]
    return tree


def _difference(a: Any, b: Any, path: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    """The first path where `a` and `b` differ, or None when equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in [*a, *(k for k in b if k not in a)]:
            if key not in a or key not in b:
                return (*path, str(key))
            found = _difference(a[key], b[key], (*path, str(key)))
            if found is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        for index, (x, y) in enumerate(zip(a, b)):
            label = x.get("name") if isinstance(x, dict) else None
            step = f"[{label}]" if isinstance(label, str) else f"[{index}]"
            found = _difference(x, y, (*path, step))
            if found is not None:
                return found
        if len(a) != len(b):
            return (*path, f"[{min(len(a), len(b))}]")
        return None
    return None if a == b else path


_WHAT = (
    (("command", "args", "workingDir", "image", "ports"), "what the pod runs"),
    (("env",), "the pod contract env"),
    (("resources",), "resources"),
    (("nodeSelector", "tolerations", "serviceAccountName", "affinity"), "placement"),
    (("volumes", "volumeMounts"), "volumes"),
    (("network", "successPolicy", "failurePolicy", "dependsOn"), "the JobSet's spec"),
    (("metadata",), "metadata"),
)


def _what(path: tuple[str, ...]) -> str:
    keys = {step for step in path if not step.startswith("[")}
    for names, what in _WHAT:
        if keys & set(names):
            return what
    return "the JobSet's shape"


def verify_unchanged(r: _Render, hook: str) -> None:
    """Fail, naming `hook`, if the hooks so far broke a rule.

    Each applied hook's data is checked against its rule. Then what the hooks
    returned is taken out again, and the rest must equal the target-free
    render: commands, the Plan and the pod contract env, resources, node
    selectors, the JobSet's shape and spec.
    """
    problems = []
    for applied in r.applied:
        problems += RULES[applied](r)
    if codec.to_json(r.plan) != r.plan_json:
        problems.append("changed the Plan")
    path = _difference(_without_additions(r), r.base)
    if path is not None:
        problems.append(f"changed {'.'.join(path).replace('.[', '[')} ({_what(path)})")
    r.fail(hook, problems)


# Serialization -------------------------------------------------------------

_KEY_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "apiVersion",
            "kind",
            "metadata",
            "name",
            "annotations",
            "spec",
            "network",
            "successPolicy",
            "failurePolicy",
            "replicatedJobs",
            "groupName",
            "replicas",
            "parallelism",
            "completions",
            "completionMode",
            "backoffLimit",
            "podFailurePolicy",
            "template",
            "restartPolicy",
            "serviceAccountName",
            "nodeSelector",
            "tolerations",
            "containers",
            "volumes",
            "image",
            "command",
            "ports",
            "resources",
            "env",
            "volumeMounts",
        )
    )
}

# Plain scalars only where YAML 1.1 and 1.2 read the same string. Everything
# else is single-quoted when it can be, else a JSON string; both versions read
# either alike.
_PLAIN = re.compile(r"[A-Za-z_/][A-Za-z0-9_./-]*")
_SPECIAL = {"y", "yes", "n", "no", "true", "false", "on", "off", "null"}
_SINGLE = re.compile(r"[^'\x00-\x1f\x7f-\x9f]*")


def _ordered(node: Any) -> Any:
    if isinstance(node, dict):
        keys = sorted(node, key=lambda k: _KEY_ORDER.get(k, len(_KEY_ORDER)))
        return {key: _ordered(node[key]) for key in keys}
    if isinstance(node, list):
        return [_ordered(item) for item in node]
    return node


def _scalar(value: Any, escape: bool) -> str:
    if value is True or value is False:
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if escape:
            value = value.replace("$", "$$")
        if _PLAIN.fullmatch(value) and value.lower() not in _SPECIAL:
            return value
        if _SINGLE.fullmatch(value):
            return f"'{value}'"
        return json.dumps(value, ensure_ascii=False)
    raise RenderError(f"cannot write {value!r} into a manifest")


def _mapping(
    node: dict[str, Any],
    indent: int,
    lines: list[str],
    escape: bool,
    first: str | None = None,
) -> None:
    pad = " " * indent
    for position, (key, value) in enumerate(node.items()):
        lead = first if position == 0 and first is not None else pad
        name = _scalar(key, escape)
        if isinstance(value, dict) and value:
            lines.append(f"{lead}{name}:")
            _mapping(value, indent + 2, lines, escape)
        elif isinstance(value, list) and value:
            lines.append(f"{lead}{name}:")
            _sequence(value, indent + 2, lines, escape)
        elif isinstance(value, dict | list):
            lines.append(f"{lead}{name}: {'{}' if isinstance(value, dict) else '[]'}")
        else:
            lines.append(f"{lead}{name}: {_scalar(value, escape)}")


def _sequence(items: list[Any], indent: int, lines: list[str], escape: bool) -> None:
    pad = " " * indent
    for item in items:
        if isinstance(item, dict) and item:
            _mapping(item, indent + 2, lines, escape, first=f"{pad}- ")
        elif isinstance(item, dict | list):
            if item:
                raise RenderError("a list directly inside a list")
            lines.append(f"{pad}- {'{}' if isinstance(item, dict) else '[]'}")
        else:
            lines.append(f"{pad}- {_scalar(item, escape)}")


def to_yaml(tree: dict[str, Any], *, escape: bool = False) -> str:
    """Block YAML in a fixed key order, with no line folding."""
    lines: list[str] = []
    _mapping(_ordered(tree), 0, lines, escape)
    return "\n".join(lines) + "\n"


def serialize(tree: dict[str, Any], *, image: str, name: str, escape: bool) -> str:
    """The manifest text: `$` doubled if `escape`, then the image and name."""
    return (
        to_yaml(tree, escape=escape).replace(IMAGE_REF, image).replace(NAME_REF, name)
    )


# The target -----------------------------------------------------------------


class KubernetesTarget:
    """A target whose workloads are JobSets; subclasses fill the hooks.

    The defaults render for any GKE cluster with JobSet: a container named
    main, no annotations, env or volumes, and runs that fail at once
    (backoffLimit 0, FailJob on a non-zero exit, no JobSet restart). apply()
    is the subclass's.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("render", "submit"):
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} overrides {name}(); a target fills hooks "
                    "and the tool renders"
                )

    # Hooks -------------------------------------------------------------

    def info(self) -> TargetInfo:
        return TargetInfo("kubernetes", None, None)

    def reserved_env(self) -> EnvNames:
        """Every name env() and the after-tasks' env can set."""
        return EnvNames(frozenset())

    def image(self, plan: Plan) -> str:
        """The image text the manifest names: the Plan's, or this target's own.

        Raises:
            RenderError: the Plan names no image and the target has none.
        """
        if plan.image is None:
            raise RenderError(
                f"{type(self).__name__} has no image of its own; set server.image "
                "or pass --set server.image=<image URI>"
            )
        return plan.image

    def workload_name(self, plan: Plan) -> str:
        """The JobSet's name, which every pod's DNS name starts with."""
        return default_workload_name(plan)

    def container_name(self) -> str:
        return DEFAULT_CONTAINER

    def annotations(self) -> Mapping[str, str]:
        """Annotations on the JobSet itself."""
        return {}

    def env(self, role: RoleShape) -> Mapping[str, str]:
        """Env for every pod of `role`, beside the pod contract."""
        return {}

    def volumes(self, role: RoleShape) -> Sequence[Volume]:
        return ()

    def cpu_pod(self) -> CpuPod | None:
        """Node selector and service account of the chipless driver's pod."""
        return None

    def job_policy(self, role: RoleShape) -> JobPolicy:
        return JobPolicy(backoff_limit=0, fail_on_exit=True, max_restarts=0)

    def after_tasks(self, shape: Shape) -> Sequence[AfterTask]:
        """Work on the results that the driver host runs (runtime/publish.py)."""
        return ()

    def lm_eval_cache_root(self) -> str | None:
        """Where lm_eval keeps its request cache in the pods; None for none."""
        return None

    def escape_dollars(self) -> bool:
        """Whether `$` is doubled, for a launcher that substitutes ${NAME}."""
        return False

    def shape_problems(self, shape: Shape) -> Sequence[str]:
        """Why this target cannot run a Shape; empty when it can."""
        return ()

    def steps_group(self, shape: Shape) -> StepGroup | None:
        """The CI concurrency group a step running this Shape joins."""
        return None

    def apply(self, manifest: str, plan: Plan) -> int:
        """Run the rendered manifest; the driver's exit code."""
        raise RenderError(
            f"{type(self).__name__} renders manifests and applies none; use the "
            "kubectl target or a fleet's"
        )

    # Final -------------------------------------------------------------

    @final
    def render(self, plan: Plan, resolved: Mapping[str, Any] | None = None) -> str:
        """The manifest submit() applies, byte for byte.

        Raises:
            RenderError: the Plan does not fit this target, or a hook broke a
                rule; the message names the hook.
        """
        return _render(self, plan, resolved)

    @final
    def submit(self, plan: Plan, resolved: Mapping[str, Any] | None = None) -> int:
        return self.apply(self.render(plan, resolved), plan)


class Preview(KubernetesTarget):
    """The tool's render alone: placeholders where a target would decide."""

    def info(self) -> TargetInfo:
        return TargetInfo("none", None, None)

    def image(self, plan: Plan) -> str:
        return "<image>" if plan.image is None else plan.image

    def workload_name(self, plan: Plan) -> str:
        return "<workload-name>"

    def job_policy(self, role: RoleShape) -> JobPolicy:
        return JobPolicy()


def _target_data(r: _Render) -> TargetData:
    info = r.call("info")
    if not isinstance(info, TargetInfo):
        r.fail("info", [f"returns {info!r}, not a TargetInfo"])
    tasks = tuple(r.items("after_tasks", r.shape))
    problems = []
    for task in tasks:
        if not isinstance(task, AfterTask):
            problems.append(f"returns {task!r}, not an AfterTask")
            continue
        for name, _ in task.env:
            if TOOL_OWNED_ENV.covers(name):
                problems.append(f"{task.name} sets {name}, which the tool sets")
            elif not r.reserved.covers(name):
                problems.append(
                    f"{task.name} sets {name}, which its reserved_env() does not list"
                )
    r.fail("after_tasks", problems)
    root = r.call("lm_eval_cache_root")
    data = TargetData(info=info, after_tasks=tasks, lm_eval_cache_root=root)
    try:
        if TargetData.from_data(data.to_data()) != data:
            raise ValueError("it does not survive encoding")
    except ValueError as error:
        raise RenderError(f"{r.who}: the target data is not valid: {error}") from None
    return data


def _text(r: _Render, hook: str) -> str:
    text = r.call(hook, r.plan)
    if not isinstance(text, str) or not _RAW_TEXT.fullmatch(text):
        r.fail(hook, [f"returns {text!r}, which cannot be written unquoted"])
    if codec.to_json(r.plan) != r.plan_json:
        r.fail(hook, ["changed the Plan"])
    return text


def _render(
    target: KubernetesTarget, plan: Plan, resolved: Mapping[str, Any] | None
) -> str:
    who = type(target).__name__
    plan_json = codec.to_json(plan)
    reserved = target.reserved_env()
    if not isinstance(reserved, EnvNames):
        raise RenderError(f"{who}.reserved_env() returns {reserved!r}, not EnvNames")
    problems = ownership_problems(plan, reserved)
    if problems:
        raise RenderError("; ".join(f"{who}: {problem}" for problem in problems))
    shape = plan.shape()
    for what, size in (
        (ENV_PLAN, len(codec.encode(plan))),
        (ENV_RESOLVED, len(codec.encode_data(resolved)) if resolved is not None else 0),
    ):
        if size > codec.MAX_ENCODED_BYTES:
            raise RenderError(
                f"the encoded {what} is {size} bytes, over the "
                f"{codec.MAX_ENCODED_BYTES}-byte limit"
            )
    r = _Render(target, plan, plan_json, shape, reserved, base={}, tree={})
    refused = r.items("shape_problems", shape)
    if refused:
        raise RenderError("; ".join(f"{who}: {problem}" for problem in refused))
    data = _target_data(r)
    r.base = render_base(plan, resolved=resolved, target_data=data)
    r.tree = copy.deepcopy(r.base)
    for hook in HOOK_ORDER:
        MERGES[hook](r)
        r.applied.append(hook)
        verify_unchanged(r, hook)
    image = _text(r, "image")
    name = _text(r, "workload_name")
    escape = target.escape_dollars()
    if not isinstance(escape, bool):
        r.fail("escape_dollars", [f"returns {escape!r}, not a bool"])
    return serialize(r.tree, image=image, name=name, escape=escape)
