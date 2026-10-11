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
"""The CI fleet target's renders against today's hand-written JobSets.

Both sides are reduced to their structure: replicatedJobs, host counts,
groups, exclusive placement, success and failure policy, node selectors,
chips and resources, ports, volumes, and every env name that is not the pod
contract. Commands are left out: what pods run is the Plan's, which the
golden check compares with today's scripts. Every difference is listed below
with its reason; applied to today's structure they must give the render's
exactly. Deleted in M4 with the manifests.
"""

import copy
import pathlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest
from manifest_checks import env_of, load, replicated_jobs
from scenario.plan.build import plan, resolved_payload, select_point
from scenario.resolve import Resolver

ROOT = pathlib.Path(__file__).resolve().parents[2]
TODAY = ROOT / "manifests" / "workloads"
EXCLUSIVE = "alpha.jobset.sigs.k8s.io/exclusive-topology"
NODE_POOL = "cloud.google.com/gke-nodepool"
TOLERATION = {
    "key": "google.com/tpu",
    "operator": "Equal",
    "value": "present",
    "effect": "NoSchedule",
}
VALUES = {"IMAGE": "img", "WORKLOAD_NAME": "wl"}
SLICE = {"MULTIHOST_TOPOLOGY": "2x2x2", "MULTIHOST_HOSTS": "2"}
ABSENT = "<absent>"

pytestmark = pytest.mark.usefixtures("fleet_env")


@dataclass(frozen=True)
class Pair:
    ref: str
    point: str | None
    manifest: str
    values: dict[str, str]
    # Today's replicatedJob name -> the render's.
    renamed: dict[str, str]


PAIRS = {
    "ray-slice": Pair(
        "glm-5.2/fp8-dp16-ep",
        None,
        "ray-multihost-slice.yaml",
        SLICE,
        {"slice": "server"},
    ),
    "mp-slice": Pair(
        "glm-5.2/fp8-dp16-ep-mp",
        None,
        "mp-multihost-slice.yaml",
        SLICE,
        {"slice": "server"},
    ),
    "qwen3.5-1p1d": Pair(
        "qwen3.5-397b/fp8-pcp8-dp8-1p1d",
        None,
        "qwen3.5-397b-1p1d.yaml",
        {},
        {"benchmark": "client"},
    ),
    "kimi-k3-1p1d": Pair(
        "kimi-k3/tp8-dp4-ep-1p1d", None, "kimi-k3-1p1d-4dp8tp.yaml", {}, {}
    ),
    "kimi-k3-agentx-1p1d": Pair(
        "kimi-k3/agentx-1p1d", "c1", "kimi-k3-agentx-1p1d.yaml", {}, {}
    ),
}


def _value(entry: dict[str, Any]) -> Any:
    if "value" in entry:
        return entry["value"]
    return "fieldRef:" + entry["valueFrom"]["fieldRef"]["fieldPath"]


def structure(doc: dict[str, Any], renamed: dict[str, str]) -> dict[str, Any]:
    spec = doc["spec"]
    jobs = {}
    for rjob in spec["replicatedJobs"]:
        job = rjob["template"]["spec"]
        pod = job["template"]["spec"]
        (main,) = pod["containers"]
        jobs[renamed.get(rjob["name"], rjob["name"])] = {
            "groupName": rjob.get("groupName"),
            "replicas": rjob.get("replicas"),
            "dependsOn": rjob.get("dependsOn"),
            "exclusive": (rjob["template"].get("metadata") or {})
            .get("annotations", {})
            .get(EXCLUSIVE),
            "parallelism": job.get("parallelism"),
            "completions": job.get("completions"),
            "completionMode": job.get("completionMode"),
            "backoffLimit": job.get("backoffLimit"),
            "activeDeadlineSeconds": job.get("activeDeadlineSeconds"),
            "successPolicy": job.get("successPolicy"),
            "restartPolicy": pod.get("restartPolicy"),
            "nodeSelector": pod.get("nodeSelector"),
            "serviceAccountName": pod.get("serviceAccountName"),
            "tolerations": pod.get("tolerations"),
            "container": main["name"],
            "image": main["image"],
            "ports": sorted(port["containerPort"] for port in main.get("ports", [])),
            "readinessProbe": main.get("readinessProbe"),
            "resources": main.get("resources"),
            "env": {
                e["name"]: _value(e)
                for e in main.get("env", [])
                if not e["name"].startswith("SCENARIO_")
            },
            "volumes": pod.get("volumes"),
            "volumeMounts": main.get("volumeMounts"),
        }
    success = spec.get("successPolicy")
    if success is not None:
        targets = [renamed.get(n, n) for n in success["targetReplicatedJobs"]]
        success = {**success, "targetReplicatedJobs": targets}
    return {
        "annotations": doc["metadata"].get("annotations"),
        "network": spec.get("network"),
        "successPolicy": success,
        "failurePolicy": spec.get("failurePolicy"),
        "jobs": jobs,
    }


@dataclass(frozen=True)
class Difference:
    path: tuple[str, ...]
    today: Any
    render: Any
    reason: str


DEADLINE = (
    "no activeDeadlineSeconds: the step's TPU_MAX_RUNTIME_SECONDS, else the "
    "fleet's 8 h default, bounds the run, and the launcher writes it into every Job"
)
GENERIC = (
    "restartPolicy Never and the TPU toleration are generic GKE, so every target "
    "renders them; the fleet's pod defaults add the same values today"
)
EXCLUSIVE_ON_ROLE = (
    "exclusive-topology on the multi-host replicatedJob; the launcher's "
    "multiHostJobs adds the same annotation there today"
)
EXCLUSIVE_NOT_ON_JOBSET = (
    "exclusive-topology on each multi-host chip replicatedJob rather than the "
    "JobSet, where it would also cover the chipless client"
)
HEAD_SUCCESS = (
    "no head-only Job successPolicy: followers exit 0 once the driver's status "
    "says done, and the JobSet succeeds with the driver's replicatedJob"
)
JOBSET_SUCCESS = "the success policy always names the driver's replicatedJob"
STATUS_PORT = "8090 is the driver's status endpoint, which followers poll"
ROUTER_PORT = "the router runs on the chipless driver, beside the tasks"
CONTRACT = (
    "the pod contract (SCENARIO_HOST_INDEX, SCENARIO_ADVERTISE_IP, "
    "SCENARIO_HEAD_ADDRESS, SCENARIO_ROLE_HOSTS) replaces the script's own "
    "placement env"
)
PLUMBING = (
    "tool plumbing in the Plan (TPU_ACCELERATOR_TYPE, the per-host DP size, "
    "Raiden's advertise host, the chipless client's env), so no manifest sets it"
)
SCRIPT_INPUT = (
    "inputs of today's entry scripts; the Plan carries the config, the chip count "
    "and the client wiring"
)
UPLOAD_ENV = (
    "the P/D BigQuery row's env is the pd-bigquery after-task's, and whether rows "
    "go up is decided at submit (main, not a PR, BQ_UPLOAD)"
)
RESULTS_ROOT = (
    "ARTIFACTS_DIR is gone: the runtime writes under the target's results root, "
    "perf_eval_results, the same artifact prefix"
)
LOG_ENV = (
    "FORCE_COLOR and TQDM_MININTERVAL came from common.sh through --env; the "
    "target writes them on every pod"
)
DATASETS = (
    "HF_DATASETS_CACHE goes wherever the compile caches do (multi-host and "
    "multi-role pods), so lm-eval datasets land on the HF mount"
)
RAY_TMPDIR = (
    "a role's scratch sets RAY_TMPDIR=/tmp beside TMPDIR=/dev/shm; unused on mp"
)
GROUP = (
    "every replicatedJob of a multi-role JobSet has its own group, single-host "
    "roles included, so one rule covers P/D on one host and on slices"
)
ENGINE_BACKOFF = (
    "chip roles of multi-role plans get backoffLimit 0, as the Kimi pairs and "
    "p8d8 do: a retried engine would rejoin a pair whose other half kept serving"
)
PROBES = (
    "no readiness probes or dependsOn: the driver waits on every role's /health "
    "itself, and the chipless pod starts at once"
)
CLIENT_SIZE = (
    "the chipless pod's size is the tool's fixed cpu 6, 16Gi, 20Gi; the AgentX "
    "client asks for 12, 32Gi, 30Gi and no scenario field states a client's "
    "size yet (F6)"
)
CLIENT_TMPDIR = "the client's TMPDIR=/tmp was the container default restated"


def at(role: str, *path: str) -> tuple[str, ...]:
    return ("jobs", role, *path)


def removed(role: str, reason: str, **env: str) -> list[Difference]:
    return [Difference(at(role, "env", n), v, ABSENT, reason) for n, v in env.items()]


def added(role: str, reason: str, **env: str) -> list[Difference]:
    return [Difference(at(role, "env", n), ABSENT, v, reason) for n, v in env.items()]


def every_pod(role: str, *, chips: bool = True) -> list[Difference]:
    found = added(role, LOG_ENV, FORCE_COLOR="1", TQDM_MININTERVAL="30")
    if chips:
        found += [
            Difference(at(role, "restartPolicy"), None, "Never", GENERIC),
            Difference(at(role, "tolerations"), None, [TOLERATION], GENERIC),
        ]
    return found


def deadline(role: str, seconds: int) -> Difference:
    return Difference(at(role, "activeDeadlineSeconds"), seconds, None, DEADLINE)


POD_IP = "fieldRef:status.podIP"


def slice_differences(**script_env: str) -> list[Difference]:
    return [
        Difference(
            ("successPolicy",),
            None,
            {"operator": "All", "targetReplicatedJobs": ["server"]},
            JOBSET_SUCCESS,
        ),
        Difference(at("server", "exclusive"), None, NODE_POOL, EXCLUSIVE_ON_ROLE),
        deadline("server", 28800),
        Difference(
            at("server", "successPolicy"),
            {"rules": [{"succeededIndexes": "0"}]},
            None,
            HEAD_SUCCESS,
        ),
        Difference(at("server", "ports"), [8000], [8000, 8090], STATUS_PORT),
        *removed(
            "server",
            CONTRACT,
            HEAD_HOST="wl-slice-0-0.wl",
            NUM_HOSTS="2",
            POD_IP=POD_IP,
        ),
        *removed("server", PLUMBING, **script_env),
        *every_pod("server"),
    ]


def pair_engines(**script_env: str) -> list[Difference]:
    """The two four-host Kimi-K3 engines, as both 1P1D manifests write them."""
    found = [
        Difference(
            ("annotations", EXCLUSIVE), NODE_POOL, ABSENT, EXCLUSIVE_NOT_ON_JOBSET
        )
    ]
    for role in ("prefill", "decode"):
        found += [
            Difference(at(role, "exclusive"), None, NODE_POOL, EXCLUSIVE_NOT_ON_JOBSET),
            deadline(role, 28800),
            *removed(
                role,
                CONTRACT,
                ROLE_HOST_PREFIX=f"wl-{role}-0-",
                ROLE_HOST_DOMAIN="wl",
                DECODE_HOST="wl-decode-0-0.wl",
                NUM_HOSTS="4",
                POD_IP=POD_IP,
            ),
            *removed(role, PLUMBING, TPU_ACCELERATOR_TYPE="tpu7x"),
            *removed(role, SCRIPT_INPUT, **script_env),
            *added(role, DATASETS, HF_DATASETS_CACHE="/cache/hf/hub/datasets"),
            *added(role, RAY_TMPDIR, RAY_TMPDIR="/tmp"),
            *every_pod(role),
        ]
    return found


AGENTX_CONFIG = {"K3_PD_CONFIG": "kimi-k3-tp8-dp4-ep-agentx"}
CLIENT_RESOURCES = {"cpu": "6", "memory": "16Gi", "ephemeral-storage": "20Gi"}

DIFFERENCES: dict[str, list[Difference]] = {
    "ray-slice": slice_differences(),
    "mp-slice": slice_differences(DP_LOCAL="8"),
    "qwen3.5-1p1d": [
        *[
            difference
            for role in ("prefill", "decode")
            for difference in (
                Difference(at(role, "groupName"), None, role, GROUP),
                Difference(at(role, "backoffLimit"), None, 0, ENGINE_BACKOFF),
                deadline(role, 36000),
                Difference(
                    at(role, "readinessProbe"),
                    {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 10},
                    None,
                    PROBES,
                ),
                *removed(role, CONTRACT, POD_IP=POD_IP),
                *removed(role, PLUMBING, TPU_RAIDEN_ADVERTISE_HOST=POD_IP),
                *added(role, DATASETS, HF_DATASETS_CACHE="/cache/hf/hub/datasets"),
                *every_pod(role),
            )
        ],
        Difference(at("client", "groupName"), None, "client", GROUP),
        Difference(
            at("client", "dependsOn"),
            [
                {"name": "prefill", "status": "Ready"},
                {"name": "decode", "status": "Ready"},
            ],
            None,
            PROBES,
        ),
        deadline("client", 36000),
        Difference(
            at("client", "ports"), [], [8090, 8100], f"{ROUTER_PORT}; {STATUS_PORT}"
        ),
        *removed("client", CONTRACT, WORKLOAD_NAME="wl"),
        *removed(
            "client",
            UPLOAD_ENV,
            RUN_TYPE="KUBE_DISAGG",
            REPOSITORY="vllm-torchtpu",
            DOCKER_IMAGE="img",
            BQ_UPLOAD="0",
        ),
        *removed(
            "client",
            PLUMBING,
            VLLM_PLUGINS="",
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            PYTHONUNBUFFERED="1",
        ),
        *every_pod("client", chips=False),
    ],
    "kimi-k3-1p1d": [
        *pair_engines(),
        Difference(
            at("prefill", "ports"), [8000, 8100], [8000, 8090, 8100], STATUS_PORT
        ),
        *removed("prefill", RESULTS_ROOT, ARTIFACTS_DIR="perf_eval_results"),
    ],
    "kimi-k3-agentx-1p1d": [
        *pair_engines(**AGENTX_CONFIG, K3_PD_EXTERNAL_CLIENT="1"),
        Difference(at("client", "groupName"), None, "client", GROUP),
        Difference(
            at("client", "dependsOn"),
            [
                {"name": "prefill", "status": "Ready"},
                {"name": "decode", "status": "Ready"},
            ],
            None,
            PROBES,
        ),
        deadline("client", 32400),
        Difference(
            at("client", "ports"), [], [8090, 8100], f"{ROUTER_PORT}; {STATUS_PORT}"
        ),
        *[
            Difference(
                at("client", "resources", kind, name),
                today,
                CLIENT_RESOURCES[name],
                CLIENT_SIZE,
            )
            for kind in ("requests", "limits")
            for name, today in (
                ("cpu", "12"),
                ("memory", "32Gi"),
                ("ephemeral-storage", "30Gi"),
            )
        ],
        *removed("client", CONTRACT, WORKLOAD_NAME="wl"),
        *removed("client", SCRIPT_INPUT, **AGENTX_CONFIG, AGENTX_CHIPS="32"),
        *removed("client", RESULTS_ROOT, ARTIFACTS_DIR="perf_eval_results"),
        *removed(
            "client",
            PLUMBING,
            VLLM_PLUGINS="",
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            PYTHONUNBUFFERED="1",
        ),
        *removed("client", CLIENT_TMPDIR, TMPDIR="/tmp"),
        *every_pod("client", chips=False),
    ],
}


def apply(found: dict[str, Any], differences: list[Difference]) -> dict[str, Any]:
    """Today's structure with each listed difference made, checking its old value."""
    found = copy.deepcopy(found)
    for difference in differences:
        holder = found
        for key in difference.path[:-1]:
            holder = holder[key]
        key = difference.path[-1]
        now = holder.get(key, ABSENT)
        assert now == difference.today, (difference.path, now, difference.today)
        if difference.render == ABSENT:
            del holder[key]
        else:
            holder[key] = difference.render
    return found


def today(pair: Pair) -> dict[str, Any]:
    text = (TODAY / pair.manifest).read_text()
    return load(text, **VALUES, **pair.values)


def rendered(tpu_ci: ModuleType, pair: Pair) -> dict[str, Any]:
    resolver = Resolver(ROOT)
    resolved = select_point(resolver.resolve(resolver.file_of(pair.ref)), pair.point)
    text = tpu_ci.TpuCi().render(plan(resolved), resolved_payload(resolved))
    return load(text, **VALUES)


@pytest.mark.parametrize("name", PAIRS)
def test_render_is_today_with_the_listed_differences(
    tpu_ci: ModuleType, name: str
) -> None:
    pair = PAIRS[name]
    old = structure(today(pair), pair.renamed)
    new = structure(rendered(tpu_ci, pair), {})
    assert apply(old, DIFFERENCES[name]) == new
    assert all(d.reason for d in DIFFERENCES[name])


@pytest.mark.parametrize("name", PAIRS)
def test_host_names_follow_todays_scheme(tpu_ci: ModuleType, name: str) -> None:
    """Pods find each other under the names today's scripts build."""
    pair = PAIRS[name]
    old = replicated_jobs(today(pair))
    new = replicated_jobs(rendered(tpu_ci, pair))
    for before, after in [*pair.renamed.items(), *((n, n) for n in new if n in old)]:
        env, contract = env_of(old[before]), env_of(new[after])
        head = f"wl-{after}-0-0.wl"
        if "SCENARIO_HEAD_ADDRESS" in contract:
            assert contract["SCENARIO_HEAD_ADDRESS"] == head
        if "HEAD_HOST" in env:
            assert env["HEAD_HOST"].replace(f"-{before}-", f"-{after}-") == head
        if "ROLE_HOST_PREFIX" in env:
            hosts = old[before]["template"]["spec"]["parallelism"]
            expected = [
                f"{env['ROLE_HOST_PREFIX']}{i}.{env['ROLE_HOST_DOMAIN']}"
                for i in range(hosts)
            ]
            assert f'"{after}":["{expected[0]}"' in contract["SCENARIO_ROLE_HOSTS"]
            assert all(host in contract["SCENARIO_ROLE_HOSTS"] for host in expected)
        if "DECODE_HOST" in env:
            assert env["DECODE_HOST"] in contract["SCENARIO_ROLE_HOSTS"]
