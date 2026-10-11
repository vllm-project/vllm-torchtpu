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
"""KubernetesTarget: the JobSet render, the pod contract and verify_unchanged.

Each rule of verify_unchanged has a hook that breaks it. A hook can only
return data, so the rules about what pods run are also tested with a merge
step that edits the JobSet the way a broken hook would.
"""

import dataclasses
import pathlib
import string
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from manifest_checks import (
    check_pods_run_the_plan,
    container,
    env_of,
    load,
    pod_spec,
    replicated_jobs,
)
from plan_fixtures import Data, build
from scenario.plan import codec
from scenario.plan.build import plan, resolved_payload
from scenario.plan.types import Plan, RoleShape, Shape
from scenario.resolve import Resolver
from scenario.runtime import placement
from scenario.target import (
    ENV_TARGET_DATA,
    POD_ENV,
    AfterTask,
    CpuPod,
    EnvNames,
    JobPolicy,
    Target,
    TargetData,
    TargetInfo,
    Volume,
)
from scenario.targets import kubernetes
from scenario.targets.kubernetes import (
    COMPLETION_INDEX,
    EXCLUSIVE_TOPOLOGY,
    KubernetesTarget,
    Preview,
    RenderError,
    default_workload_name,
    to_yaml,
)

FIXTURE_REPO = pathlib.Path(__file__).resolve().parent / "data" / "repo"
WORKLOAD = "wl"
IMAGE = "registry.example/vllm@sha256:abc"


class Fleet(KubernetesTarget):
    """A target filling every hook, as a fleet's would."""

    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.applied: list[str] = []

    def _get(self, name: str, default: Any, *args: Any) -> Any:
        value = self.overrides.get(name, default)
        return value(*args) if callable(value) else value

    def info(self) -> TargetInfo:
        return TargetInfo("fleet", "c0ffee", "results")

    def reserved_env(self) -> EnvNames:
        return self._get(
            "reserved_env",
            EnvNames(frozenset({"CACHE", "COLOR", "RUNNER", "QUOTED"}), ("FLEET_",)),
        )

    def image(self, plan: Plan) -> str:
        return self._get("image", "${IMAGE}", plan)

    def workload_name(self, plan: Plan) -> str:
        return "${WORKLOAD_NAME}"

    def container_name(self) -> str:
        return self._get("container_name", "workload")

    def annotations(self) -> Mapping[str, str]:
        return self._get("annotations", {"fleet.example.com/defaults": "standard"})

    def env(self, role: RoleShape) -> Mapping[str, str]:
        default = {"COLOR": "1"} | ({} if role.chipless else {"CACHE": "/cache"})
        return self._get("env", default, role)

    def volumes(self, role: RoleShape) -> Sequence[Volume]:
        default = (
            []
            if role.chipless
            else [Volume("shadow", "/cache/profile", "Memory", "16Gi")]
        )
        return self._get("volumes", default, role)

    def cpu_pod(self) -> CpuPod | None:
        return self._get("cpu_pod", CpuPod({"pool": "cpu"}, "workload-sa"))

    def job_policy(self, role: RoleShape) -> JobPolicy:
        return self._get("job_policy", JobPolicy(), role)

    def after_tasks(self, shape: Shape) -> Sequence[AfterTask]:
        default = [
            AfterTask("keep", ("upload", "{file}"), (("RUNNER", "x"),), "result")
        ]
        return self._get("after_tasks", default, shape)

    def lm_eval_cache_root(self) -> str | None:
        return "/cache/lm_eval"

    def escape_dollars(self) -> bool:
        return True

    def shape_problems(self, shape: Shape) -> Sequence[str]:
        return self._get("shape_problems", (), shape)

    def apply(self, manifest: str, plan: Plan) -> int:
        self.applied.append(manifest)
        return 0


def fill(text: str) -> dict[str, Any]:
    """The manifest after a launcher fills ${IMAGE} and ${WORKLOAD_NAME}."""
    return load(text, IMAGE=IMAGE, WORKLOAD_NAME=WORKLOAD)


def fixture_plans() -> list[tuple[str, Plan, dict[str, Any]]]:
    resolver = Resolver(FIXTURE_REPO)
    found = []
    for file in resolver.scenario_files():
        for resolved in resolver.resolve(file):
            name = f"{resolver.ref_of(file)}:{resolved.point or ''}"
            found.append((name, plan(resolved), resolved_payload(resolved)))
    return found


def test_is_a_target() -> None:
    assert isinstance(Fleet(), Target)
    assert isinstance(Preview(), Target)


def test_single_host_is_a_one_pod_jobset(variants: dict[str, Data]) -> None:
    planned = build(variants["single_host"])
    doc = fill(Fleet().render(planned, {"point": None, "scenario": {"x": 1}}))
    assert doc["apiVersion"] == "jobset.x-k8s.io/v1alpha2"
    assert doc["kind"] == "JobSet"
    assert doc["metadata"] == {
        "name": WORKLOAD,
        "annotations": {"fleet.example.com/defaults": "standard"},
    }
    assert doc["spec"]["network"] == {"enableDNSHostnames": True}
    assert doc["spec"]["successPolicy"] == {
        "operator": "All",
        "targetReplicatedJobs": ["server"],
    }
    assert "failurePolicy" not in doc["spec"]
    (rjob,) = doc["spec"]["replicatedJobs"]
    assert rjob["name"] == "server" and rjob["replicas"] == 1
    assert "groupName" not in rjob
    assert "metadata" not in rjob["template"]
    job = rjob["template"]["spec"]
    assert (job["parallelism"], job["completions"], job["completionMode"]) == (
        1,
        1,
        "Indexed",
    )
    assert "backoffLimit" not in job and "activeDeadlineSeconds" not in job
    pod = pod_spec(rjob)
    assert pod["restartPolicy"] == "Never"
    assert pod["nodeSelector"] == {
        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
        "cloud.google.com/gke-tpu-topology": "2x2x1",
    }
    assert pod["tolerations"] == [
        {
            "key": "google.com/tpu",
            "operator": "Equal",
            "value": "present",
            "effect": "NoSchedule",
        }
    ]
    assert "serviceAccountName" not in pod
    main = container(rjob)
    assert main["name"] == "workload"
    assert main["image"] == IMAGE
    assert main["command"] == ["scenario", "run", "--role", "server"]
    assert "args" not in main and "workingDir" not in main
    assert main["ports"] == [{"containerPort": 8000}]
    assert main["resources"] == {
        "limits": {"google.com/tpu": "4"},
        "requests": {"google.com/tpu": "4"},
    }
    env = env_of(rjob)
    assert list(env) == [
        "SCENARIO_PLAN",
        "SCENARIO_RESOLVED",
        "SCENARIO_HOST_INDEX",
        "SCENARIO_ADVERTISE_IP",
        "SCENARIO_HEAD_ADDRESS",
        "SCENARIO_IMAGE",
        "SCENARIO_TARGET_DATA",
        "COLOR",
        "CACHE",
    ]
    assert codec.decode(env["SCENARIO_PLAN"]) == planned
    assert codec.decode_data(env["SCENARIO_RESOLVED"]) == {
        "point": None,
        "scenario": {"x": 1},
    }
    assert env["SCENARIO_HOST_INDEX"] == {"fieldRef": {"fieldPath": COMPLETION_INDEX}}
    assert env["SCENARIO_ADVERTISE_IP"] == {"fieldRef": {"fieldPath": "status.podIP"}}
    assert env["SCENARIO_HEAD_ADDRESS"] == f"{WORKLOAD}-server-0-0.{WORKLOAD}"
    assert env["SCENARIO_IMAGE"] == IMAGE
    data = TargetData.from_data(codec.decode_data(env["SCENARIO_TARGET_DATA"]))
    assert data == TargetData(
        TargetInfo("fleet", "c0ffee", "results"),
        (AfterTask("keep", ("upload", "{file}"), (("RUNNER", "x"),), "result"),),
        "/cache/lm_eval",
    )
    assert pod["volumes"] == [
        {"name": "shadow", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}
    ]
    assert main["volumeMounts"] == [{"name": "shadow", "mountPath": "/cache/profile"}]


def test_multi_host_role(variants: dict[str, Data]) -> None:
    planned = build(variants["single"])
    doc = fill(Fleet().render(planned))
    (rjob,) = doc["spec"]["replicatedJobs"]
    job = rjob["template"]["spec"]
    assert (job["parallelism"], job["completions"]) == (2, 2)
    assert rjob["template"]["metadata"] == {
        "annotations": {EXCLUSIVE_TOPOLOGY: "cloud.google.com/gke-nodepool"}
    }
    assert pod_spec(rjob)["nodeSelector"]["cloud.google.com/gke-tpu-topology"] == (
        "2x2x2"
    )
    # The driver status endpoint, for the second host.
    assert container(rjob)["ports"] == [
        {"containerPort": 8000},
        {"containerPort": 8090},
    ]
    env = env_of(rjob)
    assert "SCENARIO_ROLE_HOSTS" not in env and "SCENARIO_RESOLVED" not in env
    check_pods_run_the_plan("single", planned, doc, WORKLOAD)


def test_roles_and_a_chipless_client(variants: dict[str, Data]) -> None:
    planned = build(variants["pd"])
    doc = fill(Fleet().render(planned, {"point": None, "scenario": {}}))
    jobs = replicated_jobs(doc)
    assert list(jobs) == ["prefill", "decode", "client"]
    assert [rjob["groupName"] for rjob in jobs.values()] == list(jobs)
    assert doc["spec"]["successPolicy"]["targetReplicatedJobs"] == ["client"]
    for name in ("prefill", "decode"):
        assert "metadata" not in jobs[name]["template"]
        env = env_of(jobs[name])
        assert env["SCENARIO_HEAD_ADDRESS"] == f"{WORKLOAD}-{name}-0-0.{WORKLOAD}"
        assert ENV_TARGET_DATA not in env and "SCENARIO_RESOLVED" not in env
        assert container(jobs[name])["ports"] == [{"containerPort": 8000}]
    client = jobs["client"]
    job = client["template"]["spec"]
    assert job["backoffLimit"] == 0 and job["parallelism"] == 1
    pod = pod_spec(client)
    assert pod["nodeSelector"] == {"pool": "cpu"}
    assert pod["serviceAccountName"] == "workload-sa"
    assert "tolerations" not in pod and "volumes" not in pod
    resources = {"cpu": "6", "memory": "16Gi", "ephemeral-storage": "20Gi"}
    assert container(client)["resources"] == {
        "requests": resources,
        "limits": resources,
    }
    assert container(client)["ports"] == [
        {"containerPort": 8100},
        {"containerPort": 8090},
    ]
    env = env_of(client)
    assert "SCENARIO_HEAD_ADDRESS" not in env
    assert env["COLOR"] == "1" and "CACHE" not in env
    assert {ENV_TARGET_DATA, "SCENARIO_RESOLVED"} <= set(env)
    assert placement.parse_role_hosts(env["SCENARIO_ROLE_HOSTS"]) == {
        "prefill": (f"{WORKLOAD}-prefill-0-0.{WORKLOAD}",),
        "decode": (f"{WORKLOAD}-decode-0-0.{WORKLOAD}",),
        "client": (f"{WORKLOAD}-client-0-0.{WORKLOAD}",),
    }
    check_pods_run_the_plan("pd", planned, doc, WORKLOAD)


def test_multi_host_roles(variants: dict[str, Data]) -> None:
    planned = build(variants["k3_pd"])
    doc = fill(Fleet().render(planned))
    jobs = replicated_jobs(doc)
    assert list(jobs) == ["prefill", "decode"]
    for name, rjob in jobs.items():
        assert rjob["groupName"] == name
        assert rjob["template"]["spec"]["parallelism"] == 4
        assert EXCLUSIVE_TOPOLOGY in rjob["template"]["metadata"]["annotations"]
    assert doc["spec"]["successPolicy"]["targetReplicatedJobs"] == ["prefill"]
    assert ENV_TARGET_DATA in env_of(jobs["prefill"])
    assert ENV_TARGET_DATA not in env_of(jobs["decode"])
    assert container(jobs["prefill"])["ports"] == [
        {"containerPort": 8000},
        {"containerPort": 8100},
        {"containerPort": 8090},
    ]
    check_pods_run_the_plan("k3_pd", planned, doc, WORKLOAD)


def test_every_pod_runs_what_the_plan_says(variants: dict[str, Data]) -> None:
    """Invariance: each pod's own part equals the target-free dry run."""
    plans = [(name, p, r) for name, p, r in fixture_plans()]
    plans += [(f"variant:{n}", build(d), None) for n, d in variants.items()]
    assert len(plans) >= 14
    for name, planned, resolved in plans:
        before = codec.to_json(planned)
        text = Fleet().render(planned, resolved)
        assert codec.to_json(planned) == before, name
        assert Fleet().render(planned, resolved) == text, name
        check_pods_run_the_plan(name, planned, fill(text), WORKLOAD)


def test_pod_contract_names_match_the_runtime() -> None:
    assert POD_ENV == (
        placement.ENV_PLAN,
        placement.ENV_RESOLVED,
        placement.ENV_HOST_INDEX,
        placement.ENV_ADVERTISE_IP,
        placement.ENV_HEAD_ADDRESS,
        placement.ENV_ROLE_HOSTS,
        placement.ENV_IMAGE,
        placement.ENV_TARGET_DATA,
    )


def test_defaults_fail_fast(variants: dict[str, Data]) -> None:
    data = variants["k3_pd"]
    data["server"]["image"] = "registry.example/vllm:1"
    planned = build(data)

    class Plain(KubernetesTarget):
        pass

    target = Plain()
    text = target.render(planned)
    name = default_workload_name(planned)
    assert name.startswith("scn-k3-pd-") and len(name) == len("scn-k3-pd-") + 8
    doc = load(text)
    assert doc["metadata"] == {"name": name}
    for rjob in doc["spec"]["replicatedJobs"]:
        job = rjob["template"]["spec"]
        assert job["backoffLimit"] == 0
        assert job["podFailurePolicy"] == kubernetes.FAIL_ON_EXIT
        assert container(rjob)["name"] == "main"
        assert container(rjob)["image"] == "registry.example/vllm:1"
    assert doc["spec"]["failurePolicy"] == {"maxRestarts": 0}
    env = env_of(doc["spec"]["replicatedJobs"][0])
    assert env["SCENARIO_HEAD_ADDRESS"] == f"{name}-prefill-0-0.{name}"
    with pytest.raises(RenderError, match="applies none"):
        target.submit(planned)
    with pytest.raises(RenderError, match="has no image of its own"):
        target.render(build(variants["pd"]))


def test_preview_shows_placeholders(variants: dict[str, Data]) -> None:
    text = Preview().render(build(variants["pd"]))
    assert "'<image>'" in text
    assert "'<workload-name>-prefill-0-0.<workload-name>'" in text
    assert "podFailurePolicy" not in text and "failurePolicy" not in text


def test_submit_applies_exactly_the_render(variants: dict[str, Data]) -> None:
    planned = build(variants["pd"])
    target = Fleet()
    resolved = {"point": None, "scenario": {"a": 1}}
    assert target.submit(planned, resolved) == 0
    assert target.applied == [target.render(planned, resolved)]


def test_render_and_submit_are_final() -> None:
    with pytest.raises(TypeError, match="overrides render"):

        class Renders(KubernetesTarget):
            def render(self, plan: Plan, resolved: Any = None) -> str:  # type: ignore[misc]
                return ""

    with pytest.raises(TypeError, match="overrides submit"):

        class Submits(KubernetesTarget):
            def submit(self, plan: Plan, resolved: Any = None) -> int:  # type: ignore[misc]
                return 0


def test_dollars_are_escaped_for_a_launcher(variants: dict[str, Data]) -> None:
    """Every `$` doubles, then ${IMAGE} and ${WORKLOAD_NAME} go in raw.

    The scenario's own `$` text is inside the encoded Plan, which has none;
    a target's env value with one comes back from the launcher's
    substitution as written.
    """
    data = variants["single_host"]
    data["server"]["env"] = {"A": "$FOO", "B": "${FOO}", "C": "$$", "D": "$(FOO)"}
    planned = build(data)
    target = Fleet(env={"COLOR": "a$b ${HOME}"})
    text = target.render(planned)
    assert "a$$b $${HOME}" in text and "image: '${IMAGE}'" in text
    hostile = {"FOO": "x", "HOME": "/h", "IMAGE": IMAGE, "WORKLOAD_NAME": WORKLOAD}
    doc = load(string.Template(text).substitute(hostile))
    (rjob,) = doc["spec"]["replicatedJobs"]
    assert env_of(rjob)["COLOR"] == "a$b ${HOME}"
    assert codec.decode(env_of(rjob)["SCENARIO_PLAN"]) == planned


def test_yaml_quotes_what_yaml_1_1_would_misread() -> None:
    tree = {
        "a": ["yes", "on", "No", "4", "1:20", "~", "", "x y", "plain-text_1/2.3", "/a"],
        "b": {"c": True, "d": 3, "e": {}, "f": []},
        "g": ["it's", 'say "hi"', "line\nbreak"],
    }
    text = to_yaml(tree)
    assert text == (
        "a:\n"
        "  - 'yes'\n"
        "  - 'on'\n"
        "  - 'No'\n"
        "  - '4'\n"
        "  - '1:20'\n"
        "  - '~'\n"
        "  - ''\n"
        "  - 'x y'\n"
        "  - plain-text_1/2.3\n"
        "  - /a\n"
        "b:\n"
        "  c: true\n"
        "  d: 3\n"
        "  e: {}\n"
        "  f: []\n"
        "g:\n"
        '  - "it\'s"\n'
        "  - 'say \"hi\"'\n"
        '  - "line\\nbreak"\n'
    )
    assert load(text) == tree


def test_target_env_never_reaches_the_plan(variants: dict[str, Data]) -> None:
    planned = build(variants["k3_pd"])
    doc = fill(Fleet().render(planned))
    for rjob in doc["spec"]["replicatedJobs"]:
        text = codec.to_json(codec.decode(env_of(rjob)["SCENARIO_PLAN"]))
        assert "/cache" not in text and "COLOR" not in text


# Rules, one hook that breaks each ---------------------------------------------


def render_error(target: KubernetesTarget, planned: Plan) -> str:
    with pytest.raises(RenderError) as caught:
        target.render(planned)
    return str(caught.value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"annotations": {"fleet.example.com/x": 1}},
            "Fleet.annotations(): annotation",
        ),
        ({"annotations": {"not a key": "x"}}, "not a qualified name"),
        ({"container_name": "Main_1"}, "Fleet.container_name(): container name"),
        (
            {"env": {"SCENARIO_PLAN": "x"}},
            "Fleet.env(): sets SCENARIO_PLAN on role server, which the tool sets",
        ),
        (
            {"env": {"VLLM_HOST_IP": "x"}},
            "sets VLLM_HOST_IP on role server, which the tool",
        ),
        (
            {"env": {"OTHER": "x"}},
            "sets OTHER on role server, which its reserved_env() does not list",
        ),
        ({"env": {"COLOR": "$(HOME)"}}, "Kubernetes expands $(...)"),
        ({"env": {"COLOR": 1}}, "not a string"),
        (
            {"volumes": [Volume("a", "/x"), Volume("b", "/x")]},
            "Fleet.volumes(): volume 'b' on role server: mount path /x is used twice",
        ),
        ({"volumes": [Volume("a", "x")]}, "mount path 'x' is not absolute"),
        ({"volumes": [Volume("A_1", "/x")]}, "the name is not a DNS label"),
        ({"volumes": [Volume("a", "/x", "Memory", "lots")]}, "is not a quantity"),
        (
            {"job_policy": JobPolicy(backoff_limit=-1)},
            "Fleet.job_policy(): backoff_limit -1",
        ),
        ({"image": "a b"}, "Fleet.image(): returns 'a b'"),
        (
            {"after_tasks": [AfterTask("x", ("y",), (("OTHER", "1"),))]},
            "Fleet.after_tasks(): x sets OTHER, which its reserved_env() does not list",
        ),
        (
            {"after_tasks": [AfterTask("x", ("y",), (("PYTHONUNBUFFERED", "1"),))]},
            "x sets PYTHONUNBUFFERED, which the tool sets",
        ),
        (
            {"reserved_env": EnvNames(frozenset({"TPU_SKIP_MDS_QUERY"}))},
            "Fleet: the target reserves TPU_SKIP_MDS_QUERY, which the tool sets",
        ),
        ({"shape_problems": ["no such pool"]}, "Fleet: no such pool"),
        ({"volumes": None}, "Fleet.volumes(): returns None, not a list"),
        ({"after_tasks": None}, "Fleet.after_tasks(): returns None, not a list"),
    ],
)
def test_single_role_rules(
    variants: dict[str, Data], overrides: dict[str, Any], message: str
) -> None:
    planned = build(variants["single_host"])
    assert message in render_error(Fleet(**overrides), planned)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"volumes": lambda role: [Volume("tmp", "/tmp-x", "Memory")]},
            "volume 'tmp' on role client: a memory volume on a chipless pod needs a "
            "size limit",
        ),
        (
            {"cpu_pod": CpuPod({"cloud.google.com/gke-tpu-topology": "2x2x1"})},
            "Fleet.cpu_pod(): selects cloud.google.com/gke-tpu-topology for a pod "
            "that holds no chips",
        ),
        ({"cpu_pod": CpuPod({}, "Bad_SA")}, "service account 'Bad_SA'"),
        (
            {"job_policy": JobPolicy(backoff_limit=2)},
            "backoff_limit 2 on the chipless client",
        ),
        (
            {
                "job_policy": lambda role: JobPolicy(
                    max_restarts=1 if role.chipless else 0
                )
            },
            "roles disagree on the JobSet's max_restarts: [0, 1]",
        ),
    ],
)
def test_multi_role_rules(
    variants: dict[str, Data], overrides: dict[str, Any], message: str
) -> None:
    planned = build(variants["pd"])
    assert message in render_error(Fleet(**overrides), planned)


def test_the_scenario_may_not_set_a_target_name(variants: dict[str, Data]) -> None:
    data = variants["single_host"]
    data["server"]["env"] = {"CACHE": "/mine"}
    message = render_error(Fleet(), build(data))
    assert message == (
        "Fleet: the scenario sets CACHE, which the target sets; remove it from the "
        "scenario"
    )


def test_a_hook_that_changes_the_plan(variants: dict[str, Data]) -> None:
    planned = build(variants["single_host"])

    def sneaky(plan: Plan) -> str:
        object.__setattr__(plan, "model_name", "other")
        return "${IMAGE}"

    assert (
        render_error(Fleet(image=sneaky), planned) == "Fleet.image(): changed the Plan"
    )


def test_a_hook_that_raises_is_named(variants: dict[str, Data]) -> None:
    def broken(role: RoleShape) -> Mapping[str, str]:
        raise KeyError("x")

    message = render_error(Fleet(env=broken), build(variants["single_host"]))
    assert message == "Fleet.env() raised KeyError('x')"


def edit_after(hook: str, edit: Any) -> Any:
    """A merge step for `hook` that also edits the JobSet."""
    merge = kubernetes.MERGES[hook]

    def broken(r: Any) -> None:
        merge(r)
        edit(r.tree)

    return broken


def first_job(tree: dict[str, Any]) -> dict[str, Any]:
    return tree["spec"]["replicatedJobs"][0]


@pytest.mark.parametrize(
    ("hook", "edit", "message"),
    [
        (
            "env",
            lambda t: container(first_job(t))["command"].append("--x"),
            "Fleet.env(): changed spec.replicatedJobs[server].template.spec.template."
            "spec.containers[main].command[4] (what the pod runs)",
        ),
        (
            "volumes",
            lambda t: container(first_job(t))["env"][0].update(value="x"),
            "containers[main].env[SCENARIO_PLAN].value (the pod contract env)",
        ),
        (
            "cpu_pod",
            lambda t: container(first_job(t))["resources"]["limits"].update(cpu="1"),
            "resources.limits.cpu (resources)",
        ),
        (
            "annotations",
            lambda t: pod_spec(first_job(t))["nodeSelector"].pop(
                "cloud.google.com/gke-tpu-topology"
            ),
            "Fleet.annotations(): changed spec.replicatedJobs[server].template.spec."
            "template.spec.nodeSelector.cloud.google.com/gke-tpu-topology (placement)",
        ),
        (
            "job_policy",
            lambda t: t["spec"]["successPolicy"].update(operator="Any"),
            "changed spec.successPolicy.operator (the JobSet's spec)",
        ),
        (
            "job_policy",
            lambda t: first_job(t)["template"]["spec"].update(parallelism=3),
            "template.spec.parallelism (the JobSet's shape)",
        ),
        (
            "container_name",
            lambda t: t["metadata"].update(labels={"queue": "x"}),
            "changed metadata.labels (metadata)",
        ),
        (
            "volumes",
            lambda t: pod_spec(first_job(t))["volumes"].append(
                {"name": "x", "hostPath": {}}
            ),
            "template.spec.volumes (volumes)",
        ),
    ],
)
def test_verify_unchanged_names_the_hook(
    variants: dict[str, Data],
    monkeypatch: pytest.MonkeyPatch,
    hook: str,
    edit: Any,
    message: str,
) -> None:
    monkeypatch.setitem(kubernetes.MERGES, hook, edit_after(hook, edit))
    planned = build(variants["single_host"])
    target = Fleet(volumes=lambda role: [Volume("a", "/a"), Volume("b", "/b")])
    assert message in render_error(target, planned)


def test_plan_size_limit(
    variants: dict[str, Data], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codec, "MAX_ENCODED_BYTES", 100)
    message = render_error(Fleet(), build(variants["single_host"]))
    assert message.startswith("the encoded SCENARIO_PLAN is ")


def test_role_shape_is_all_a_hook_sees(variants: dict[str, Data]) -> None:
    seen: list[Any] = []

    def record(role: RoleShape) -> Mapping[str, str]:
        seen.append(role)
        return {}

    planned = build(variants["pd"])
    Fleet(env=record).render(planned)
    assert seen == list(planned.shape().roles)
    assert all(dataclasses.is_dataclass(role) for role in seen)
