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
"""The CI fleet target: hook values, the launcher's contract, and apply().

The launcher is ci-infra's kueue/launcher/launch.py. Its rules are restated
here as the checks tpu_ci runs before submitting; the substitution test
copies its one substitution pass (render(), launch.py), so escaping is
tested against the launcher's own semantics.
"""

import ast
import json
import os
import pathlib
import stat
import string
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest
from click.testing import CliRunner
from manifest_checks import container, env_of, load, pod_spec, replicated_jobs
from scenario.cli import main
from scenario.plan import codec
from scenario.plan.build import plan, resolved_payload, select_point
from scenario.plan.types import Plan, RoleShape, Shape
from scenario.resolve import Resolved, Resolver
from scenario.target import (
    AfterTask,
    CpuPod,
    EnvNames,
    JobPolicy,
    StepGroup,
    Target,
    TargetData,
    TargetInfo,
    Volume,
)
from scenario.targets.kubernetes import KubernetesTarget, RenderError, to_yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
TPU_CI = ROOT / "scenario" / "tpu_ci.py"
# conftest's fleet_env sets it.
COMMIT = "0123456789abcdef0123456789abcdef01234567"
GLM = "glm-5.2/fp8-dp16-ep"
GLM_MP = "glm-5.2/fp8-dp16-ep-mp"
QWEN_PD = "qwen3.5-397b/fp8-pcp8-dp8-1p1d"
K3_PD = "kimi-k3/tp8-dp4-ep-1p1d"
SINGLE = "qwen3-0.6b/tp1"
AGENTX_PD = "kimi-k3/agentx-1p1d"

pytestmark = pytest.mark.usefixtures("fleet_env")


def point(ref: str, name: str | None = None, sets: tuple[str, ...] = ()) -> Resolved:
    resolver = Resolver(ROOT)
    return select_point(resolver.resolve(resolver.file_of(ref), sets), name)


def planned(ref: str, name: str | None = None) -> Plan:
    return plan(point(ref, name))


def role(**fields: Any) -> RoleShape:
    values = {
        "name": "server",
        "accelerator": "tpu7x",
        "topology": "2x2x1",
        "hosts": 1,
        "chips_per_host": 4,
        "backend": "single",
        "scratch_bytes": None,
        "chipless": False,
        "driver": True,
        "multi_role": False,
        "cache": "shared",
        "ports": (8000,),
    }
    return RoleShape(**(values | fields))


CLIENT = {
    "name": "client",
    "accelerator": None,
    "topology": None,
    "hosts": 1,
    "chips_per_host": 0,
    "backend": None,
    "chipless": True,
    "multi_role": True,
}


def render(tpu_ci: ModuleType, ref: str, name: str | None = None) -> str:
    resolved = point(ref, name)
    return tpu_ci.TpuCi().render(plan(resolved), resolved_payload(resolved))


# Hook values --------------------------------------------------------------


def test_fixed_hooks(tpu_ci: ModuleType) -> None:
    target = tpu_ci.make_target()
    assert isinstance(target, Target) and isinstance(target, KubernetesTarget)
    assert target.info() == TargetInfo("tpu_ci", COMMIT, "perf_eval_results")
    some = planned(SINGLE)
    assert target.image(some) == "${IMAGE}"
    assert target.workload_name(some) == "${WORKLOAD_NAME}"
    assert target.container_name() == "workload"
    assert target.annotations() == {"tpu-ci.google.com/defaults": "standard"}
    assert target.escape_dollars() is True
    assert target.cpu_pod() == CpuPod(
        {"cloud.google.com/compute-class": "worker-cpu"}, "tpu-workload"
    )
    assert target.lm_eval_cache_root() == "/cache/hf/hub/lm_eval_requests"
    assert target.reserved_env() == EnvNames(
        frozenset(
            {
                "VLLM_CACHE_ROOT",
                "VLLM_XLA_CACHE_PATH",
                "HF_DATASETS_CACHE",
                "RAY_LOG_TO_STDERR",
                "TMPDIR",
                "RAY_TMPDIR",
                "FORCE_COLOR",
                "TQDM_MININTERVAL",
                "CI_RUNNER",
                "CREATED_BY",
                "RUN_TYPE",
                "REPOSITORY",
                "DOCKER_IMAGE",
                "TPU_NAME",
                "HF_TOKEN",
                "HF_HOME",
                "ARTIFACTS_DIR",
                "SKIP_SPANNER_UPLOAD",
            }
        ),
        ("BUILDKITE_",),
    )


def test_commit_without_buildkite(
    tpu_ci: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BUILDKITE_COMMIT")
    monkeypatch.setattr(tpu_ci, "git_commit", lambda: "abc")
    assert tpu_ci.TpuCi().info().commit == "abc"


LOG_ENV = {"FORCE_COLOR": "1", "TQDM_MININTERVAL": "30"}
CACHE_ENV = {
    "VLLM_CACHE_ROOT": "/cache/jax/vllm",
    "VLLM_XLA_CACHE_PATH": "/cache/jax/xla",
    "HF_DATASETS_CACHE": "/cache/hf/hub/datasets",
}
SHADOW = Volume("vllm-profile-cache", "/cache/jax/vllm/profile_cache", "Memory", "16Gi")


@pytest.mark.parametrize(
    ("shape", "env", "volumes"),
    [
        # Single host, single role: no cache env, as today's single-host steps.
        (role(), LOG_ENV, ()),
        (role(hosts=2, topology="2x2x2", backend="mp"), CACHE_ENV | LOG_ENV, (SHADOW,)),
        (
            role(hosts=2, topology="2x2x2", backend="ray"),
            CACHE_ENV | {"RAY_LOG_TO_STDERR": "1"} | LOG_ENV,
            (SHADOW,),
        ),
        (role(multi_role=True, name="decode"), CACHE_ENV | LOG_ENV, (SHADOW,)),
        (
            role(hosts=4, topology="2x2x4", backend="mp", scratch_bytes=64 << 30),
            CACHE_ENV | {"TMPDIR": "/dev/shm", "RAY_TMPDIR": "/tmp"} | LOG_ENV,
            (SHADOW,),
        ),
        (
            role(multi_role=True, name="replica-a", cache="private"),
            {
                "VLLM_CACHE_ROOT": "/cache/jax/vllm/replica-a",
                "VLLM_XLA_CACHE_PATH": "/cache/jax/xla/replica-a",
                "HF_DATASETS_CACHE": "/cache/hf/hub/datasets",
            }
            | LOG_ENV,
            (
                Volume(
                    "vllm-profile-cache",
                    "/cache/jax/vllm/replica-a/profile_cache",
                    "Memory",
                    "16Gi",
                ),
            ),
        ),
        (role(**CLIENT), LOG_ENV, ()),
    ],
    ids=["single", "mp", "ray", "pd-role", "scratch", "private", "client"],
)
def test_env_and_volumes(
    tpu_ci: ModuleType,
    shape: RoleShape,
    env: dict[str, str],
    volumes: tuple[Volume, ...],
) -> None:
    target = tpu_ci.TpuCi()
    assert dict(target.env(shape)) == env
    assert tuple(target.volumes(shape)) == volumes
    assert set(env) <= target.reserved_env().names


def test_job_policy(tpu_ci: ModuleType) -> None:
    target = tpu_ci.TpuCi()
    assert target.job_policy(role()) == JobPolicy()
    assert target.job_policy(role(hosts=4, topology="2x2x4")) == JobPolicy()
    assert target.job_policy(role(multi_role=True)) == JobPolicy(backoff_limit=0)
    assert target.job_policy(role(**CLIENT)) == JobPolicy()


def test_steps_group(tpu_ci: ModuleType) -> None:
    target = tpu_ci.TpuCi()
    group = StepGroup("tpu/v7x-2x2x4", 1)
    assert target.steps_group(planned(K3_PD).shape()) == group
    assert target.steps_group(planned(AGENTX_PD, "c1").shape()) == group
    assert target.steps_group(planned("kimi-k3/tp32-ep").shape()) == group
    assert target.steps_group(planned(GLM).shape()) is None
    assert target.steps_group(planned(QWEN_PD).shape()) is None


def test_one_shape_across_chip_roles(tpu_ci: ModuleType) -> None:
    """R-TGT-6, in the launcher's words."""
    mixed = Shape(
        None,
        (
            role(name="prefill", multi_role=True),
            role(name="decode", multi_role=True, topology="2x2x2", hosts=2),
        ),
    )
    (problem,) = tpu_ci.TpuCi().shape_problems(mixed)
    assert problem.startswith(
        "names more than one shape tpu7x 2x2x1 x4; tpu7x 2x2x2 x4. A workload is "
        "admitted against one queue"
    )
    assert tpu_ci.TpuCi().shape_problems(planned(QWEN_PD).shape()) == ()


# After-tasks ----------------------------------------------------------------


def tasks_by_name(tpu_ci: ModuleType, shape: Shape) -> dict[str, AfterTask]:
    return {task.name: task for task in tpu_ci.TpuCi().after_tasks(shape)}


def test_after_tasks_on_main(tpu_ci: ModuleType) -> None:
    tasks = tasks_by_name(tpu_ci, planned(GLM).shape())
    assert list(tasks) == [
        "artifacts-result",
        "artifacts-end",
        "bigquery",
        "pd-bigquery",
    ]
    assert tasks["artifacts-result"] == AfterTask(
        "artifacts-result",
        ("buildkite-agent", "artifact", "upload", "{file}"),
        when="result",
        timeout_s=300,
    )
    end = tasks["artifacts-end"]
    assert end.argv[:2] == ("bash", "-c") and end.argv[3:] == ("_", "{unpublished}")
    assert (end.when, end.require, end.fatal, end.timeout_s) == (
        "end",
        "always",
        True,
        1800,
    )
    assert tasks["bigquery"] == AfterTask(
        "bigquery",
        (
            "python3",
            "scripts/vllm/benchmarking/upload_results.py",
            "--results-dir",
            "{run_dir}",
            "--skip-spanner",
        ),
        env=(("CI_RUNNER", "kube"), ("TPU_NAME", "tpu7x-2x2x2")),
        require="commands_ok",
        fatal=False,
        skip_on_experiment=True,
        timeout_s=600,
    )
    assert tasks["pd-bigquery"] == AfterTask(
        "pd-bigquery",
        (
            "python3",
            ".buildkite/scripts/parse_gke_results.py",
            "{file}",
            "kube-vllm-torchtpu-run-{run_stamp}-{label}",
        ),
        env=(
            ("RUN_TYPE", "KUBE_DISAGG"),
            ("REPOSITORY", "vllm-torchtpu"),
            ("DOCKER_IMAGE", "{image}"),
            ("CI_RUNNER", "kube"),
        ),
        when="result",
        results="cells",
        require="result_ok",
        fatal=True,
        skip_on_experiment=True,
        timeout_s=300,
    )


def test_after_tasks_name_the_scripts_that_exist(tpu_ci: ModuleType) -> None:
    repo = ROOT.parents[1]
    for task in tpu_ci.TpuCi().after_tasks(planned(QWEN_PD).shape()):
        scripts = [word for word in task.argv if word.endswith(".py")]
        assert all((repo / script).is_file() for script in scripts), task


def test_end_upload_skips_an_empty_list(
    tpu_ci: ModuleType, tmp_path: pathlib.Path
) -> None:
    """Nothing left to upload is not a failure; one file reaches the agent."""
    agent = tmp_path / "buildkite-agent"
    agent.write_text(f"#!{sys.executable}\nimport sys\nprint(sys.argv[1:])\n")
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    end = tasks_by_name(tpu_ci, planned(GLM).shape())["artifacts-end"]
    for unpublished, out in (
        ("", ""),
        ("a/b.json;a/c.log", "['artifact', 'upload', 'a/b.json;a/c.log']\n"),
    ):
        argv = [word.replace("{unpublished}", unpublished) for word in end.argv]
        done = subprocess.run(argv, env=env, capture_output=True, text=True)
        assert (done.returncode, done.stdout) == (0, out)


@pytest.mark.parametrize(
    ("env", "uploads"),
    [
        ({}, True),
        ({"BUILDKITE_BRANCH": "ci/x"}, False),
        ({"BUILDKITE_PULL_REQUEST": "1812"}, False),
        ({"BQ_UPLOAD": "0"}, False),
        ({"BQ_UPLOAD": "1"}, True),
    ],
)
def test_rows_only_from_main(
    tpu_ci: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    uploads: bool,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    names = set(tasks_by_name(tpu_ci, planned(QWEN_PD).shape()))
    assert ({"bigquery", "pd-bigquery"} <= names) is uploads
    assert {"artifacts-result", "artifacts-end"} <= names


def test_parity_runner_and_creator(
    tpu_ci: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI_RUNNER", "kube-scenario")
    monkeypatch.setenv("CREATED_BY", "schedule")
    tasks = tasks_by_name(tpu_ci, planned(K3_PD).shape())
    assert tasks["bigquery"].env == (
        ("CI_RUNNER", "kube-scenario"),
        ("CREATED_BY", "schedule"),
        ("TPU_NAME", "tpu7x-2x2x4"),
    )
    assert tasks["pd-bigquery"].env[-2:] == (
        ("CI_RUNNER", "kube-scenario"),
        ("CREATED_BY", "schedule"),
    )


def test_tpu_name(tpu_ci: ModuleType) -> None:
    assert tpu_ci.tpu_name(planned(SINGLE).shape()) == "tpu7x-1x1x1"
    assert tpu_ci.tpu_name(planned(QWEN_PD).shape()) == "tpu7x-2x2x1"
    v6e = Shape(
        None, (role(accelerator="tpu-v6e-slice", topology="2x4", chips_per_host=8),)
    )
    assert tpu_ci.tpu_name(v6e) == "v6e-2x4"


def test_target_data_in_the_driver_pod(tpu_ci: ModuleType) -> None:
    doc = load(render(tpu_ci, QWEN_PD), IMAGE="i", WORKLOAD_NAME="w")
    data = codec.decode_data(
        env_of(replicated_jobs(doc)["client"])["SCENARIO_TARGET_DATA"]
    )
    found = TargetData.from_data(data)
    assert found.info == TargetInfo("tpu_ci", COMMIT, "perf_eval_results")
    assert found.lm_eval_cache_root == "/cache/hf/hub/lm_eval_requests"
    assert found.after_tasks == tuple(
        tpu_ci.TpuCi().after_tasks(planned(QWEN_PD).shape())
    )


# The launcher's contract ----------------------------------------------------


def edited(tpu_ci: ModuleType, edit: Any, ref: str = QWEN_PD) -> str:
    """A render with `edit` applied to the document, `$` left as rendered."""
    doc = load(render(tpu_ci, ref))
    edit(doc)
    return to_yaml(doc)


def first(doc: dict[str, Any], name: str = "prefill") -> dict[str, Any]:
    return replicated_jobs(doc)[name]


def test_every_check_passes_on_a_render(tpu_ci: ModuleType) -> None:
    for ref in (SINGLE, GLM, GLM_MP, QWEN_PD, K3_PD):
        assert tpu_ci.launcher_problems(render(tpu_ci, ref)) == [], ref


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (
            lambda d: container(first(d))["env"].append({"name": "X", "value": "a$ b"}),
            "a lone $; a literal one is $$",
        ),
        (
            lambda d: container(first(d))["env"].append(
                {"name": "X", "value": "${HOME}"}
            ),
            "${HOME} would come from the step's env",
        ),
        (lambda d: d.update(kind="Job"), "not a JobSet"),
        (
            lambda d: container(first(d)).update(name="main"),
            "prefill: needs exactly one container",
        ),
        (
            lambda d: pod_spec(first(d))["containers"].append(
                dict(container(first(d)))
            ),
            "prefill: needs exactly one container, named workload",
        ),
        (lambda d: d["metadata"].pop("annotations"), "no tpu-ci.google.com/defaults"),
        (
            lambda d: d["metadata"].update(labels={"kueue.x-k8s.io/queue-name": "q"}),
            "sets kueue.x-k8s.io/queue-name, which the launcher sets",
        ),
        (
            lambda d: d["metadata"].update(
                labels={"kueue.x-k8s.io/priority-class": "low"}
            ),
            "sets kueue.x-k8s.io/priority-class",
        ),
        (
            lambda d: pod_spec(first(d, "client")).update(
                serviceAccountName="tpu-launcher"
            ),
            "client: service account tpu-launcher is not allowed",
        ),
        (
            lambda d: pod_spec(first(d, "decode"))["nodeSelector"].update(
                {"cloud.google.com/gke-tpu-topology": "2x2x2"}
            ),
            "names more than one shape",
        ),
        (
            lambda d: [
                pod_spec(rjob)["nodeSelector"].pop("cloud.google.com/gke-tpu-topology")
                for rjob in replicated_jobs(d).values()
                if "nodeSelector" in pod_spec(rjob)
                and "google.com/tpu" in container(rjob)["resources"]["limits"]
            ],
            "a pod holding chips needs the accelerator and topology selectors",
        ),
        (
            lambda d: pod_spec(first(d, "client")).update(
                volumes=[{"name": "dshm", "emptyDir": {"medium": "Memory"}}]
            ),
            "client: memory volume dshm has no sizeLimit on a pod without chips",
        ),
    ],
)
def test_each_launcher_check(tpu_ci: ModuleType, edit: Any, problem: str) -> None:
    problems = tpu_ci.launcher_problems(edited(tpu_ci, edit))
    assert any(problem in found for found in problems), problems


def test_launcher_substitution_restores_every_dollar(tpu_ci: ModuleType) -> None:
    """A scenario's `$`s, and a target's, survive the launcher's one pass.

    The pass is launch.py's render(): string.Template over the whole file
    with the step's env and the two names. The result must parse and equal
    the same render made without escaping, with the two names filled in.
    """
    hostile = ("$FOO", "${FOO}", "$$", "$(FOO)")
    sets = tuple(f"server.env.HOSTILE_{i}={value}" for i, value in enumerate(hostile))
    resolved = point(SINGLE, None, sets)
    target_plan = plan(resolved)

    class Dollars(tpu_ci.TpuCi):
        def env(self, role: RoleShape) -> dict[str, str]:
            return {**super().env(role), "TQDM_MININTERVAL": "a$b ${HOME} $$"}

    class Unescaped(Dollars):
        def escape_dollars(self) -> bool:
            return False

    text = Dollars().render(target_plan, resolved_payload(resolved))
    assert tpu_ci.launcher_problems(text) == []
    step_env = {"FOO": "x", "HOME": "/h", "PATH": "/bin"}
    filled = string.Template(text).substitute(
        {**step_env, "IMAGE": "img@sha256:1", "WORKLOAD_NAME": "bk-1"}
    )
    plain = Unescaped().render(target_plan, resolved_payload(resolved))
    plain = plain.replace("${IMAGE}", "img@sha256:1").replace(
        "${WORKLOAD_NAME}", "bk-1"
    )
    assert load(filled) == load(plain)
    (rjob,) = load(filled)["spec"]["replicatedJobs"]
    assert env_of(rjob)["TQDM_MININTERVAL"] == "a$b ${HOME} $$"
    decoded = codec.decode(env_of(rjob)["SCENARIO_PLAN"])
    assert decoded == target_plan
    scenario_env = dict(decoded.roles[0].host_plans[0].scenario_env)
    assert [scenario_env[f"HOSTILE_{i}"] for i in range(len(hostile))] == list(hostile)


# apply() --------------------------------------------------------------------


class Launch(Exception):
    """Raised where apply() would exec the launcher."""


@pytest.fixture
def agent(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A stand-in buildkite-agent on PATH, and tmp_path as the checkout."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    program = bin_dir / "buildkite-agent"
    program.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "root = pathlib.Path(os.environ['FAKE_AGENT'])\n"
        "with (root / 'calls.jsonl').open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['meta-data', 'get']:\n"
        "    meta = json.loads((root / 'meta.json').read_text())\n"
        "    print(meta.get(sys.argv[3], ''))\n"
    )
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "meta.json").write_text(json.dumps({"CI_IMAGE_TAG": "registry/ci:abc"}))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_AGENT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def calls(agent: pathlib.Path) -> list[list[str]]:
    path = agent / "calls.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def recording(tpu_ci: ModuleType) -> KubernetesTarget:
    class Recording(tpu_ci.TpuCi):
        def _exec(self, argv: list[str], env: dict[str, str]) -> int:
            raise Launch(argv, env)

    return Recording()


def test_apply_execs_the_launcher(tpu_ci: ModuleType, agent: pathlib.Path) -> None:
    resolved = point(AGENTX_PD, "c4")
    target = recording(tpu_ci)
    with pytest.raises(Launch) as launched:
        target.submit(plan(resolved), resolved_payload(resolved))
    argv, env = launched.value.args
    manifest = "scenario-manifests/kimi-k3/agentx-1p1d.c4.yaml"
    assert argv == ["/opt/launcher/launch", "--manifest", manifest]
    assert env["WORKLOAD_IMAGE"] == "registry/ci:abc"
    assert env["BUILDKITE_COMMIT"] == COMMIT
    assert (agent / manifest).read_text() == target.render(
        plan(resolved), resolved_payload(resolved)
    )
    assert calls(agent) == [
        ["meta-data", "get", "CI_IMAGE_TAG", "--default", ""],
        ["artifact", "upload", manifest],
    ]


def test_apply_reads_the_image_variant(
    tpu_ci: ModuleType, agent: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (agent / "meta.json").write_text(
        json.dumps({"CI_IMAGE_TAG-omni": "registry/omni:1"})
    )
    monkeypatch.setenv("DOCKER_IMAGE_SUFFIX", "-omni")
    with pytest.raises(Launch) as launched:
        recording(tpu_ci).submit(planned(SINGLE))
    assert launched.value.args[1]["WORKLOAD_IMAGE"] == "registry/omni:1"
    assert launched.value.args[0][-1] == "scenario-manifests/qwen3-0.6b/tp1.yaml"


def test_apply_takes_an_image_uri(tpu_ci: ModuleType, agent: pathlib.Path) -> None:
    image = "us-docker.pkg.dev/project/repo/vllm:1"
    resolved = point(SINGLE, None, (f"server.image={image}",))
    with pytest.raises(Launch) as launched:
        recording(tpu_ci).submit(plan(resolved))
    assert launched.value.args[1]["WORKLOAD_IMAGE"] == image
    assert ["meta-data", "get", "CI_IMAGE_TAG", "--default", ""] not in calls(agent)


@pytest.mark.parametrize(
    ("env", "problem"),
    [
        ({"WORKLOAD_IMAGE": "x"}, "the step sets WORKLOAD_IMAGE; put the image in"),
        (
            {"VLLM_ENGINE_READY_TIMEOUT_S": "7200"},
            "the step sets VLLM_ENGINE_READY_TIMEOUT_S, which a scenario step "
            "ignores; use server.startup_timeout",
        ),
        ({"TMPDIR": "/dev/shm"}, "the step sets TMPDIR, which a scenario step ignores"),
        ({"SHAPE": "tpu7x-standard-4t/2x2x1"}, "the step sets SHAPE"),
        (
            {"TPU_MAX_RUNTIME_SECONDS": "3600"},
            "TPU_MAX_RUNTIME_SECONDS='3600' must be a whole number of seconds over the "
            "scenario's startup_timeout (7200s)",
        ),
        ({"TPU_MAX_RUNTIME_SECONDS": "8h"}, "TPU_MAX_RUNTIME_SECONDS='8h' must be"),
    ],
)
def test_apply_refuses(
    tpu_ci: ModuleType,
    agent: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: dict[str, str],
    problem: str,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert recording(tpu_ci).submit(planned(GLM)) == 2
    assert problem in capsys.readouterr().err
    assert not (agent / "scenario-manifests").exists()


def test_apply_allows_a_long_runtime(
    tpu_ci: ModuleType, agent: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TPU_MAX_RUNTIME_SECONDS", "18000")
    with pytest.raises(Launch):
        recording(tpu_ci).submit(planned(GLM))


def test_apply_without_the_image_build(
    tpu_ci: ModuleType, agent: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (agent / "meta.json").write_text("{}")
    assert recording(tpu_ci).submit(planned(GLM)) == 2
    assert "no CI_IMAGE_TAG meta-data: the image build step has to run" in (
        capsys.readouterr().err
    )


def test_render_needs_no_agent(
    tpu_ci: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render() reads no meta-data and runs no launcher, so it works anywhere."""
    monkeypatch.setenv("PATH", "/nonexistent")
    assert "${WORKLOAD_NAME}" in render(tpu_ci, K3_PD)


def test_a_mixed_shape_does_not_render(tpu_ci: ModuleType) -> None:
    sets = (
        "server.roles.decode.topology=2x2x2",
        "server.roles.decode.hosts=2",
        "server.multihost=mp",
    )
    resolved = point(QWEN_PD, None, sets)
    with pytest.raises(RenderError, match="TpuCi: names more than one shape"):
        tpu_ci.TpuCi().render(plan(resolved))


# The module itself ----------------------------------------------------------

ALLOWED = ("scenario.target", "scenario.targets.kubernetes", "scenario.plan.types")


def test_imports_only_the_target_contract() -> None:
    """tpu_ci may not import the scenario model, resolution or the runtime."""
    tree = ast.parse(TPU_CI.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    assert {name for name in found if name.split(".")[0] == "scenario"} <= set(ALLOWED)


def test_no_scenario_sets_a_name_it_reserves() -> None:
    args = ["validate", "--root", str(ROOT), "--target", str(TPU_CI)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1].endswith(" scenarios, 0 errors")
