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
"""render, submit, steps and validate --target, through the CLI."""

import json
import pathlib
import textwrap

import pytest
from click.testing import CliRunner, Result
from manifest_checks import load
from scenario.cli import main
from scenario.plan import codec
from scenario.resolve import Resolver

REPO = pathlib.Path(__file__).resolve().parent / "data" / "repo"
GLM = "scenarios/glm-5.2/fp8-dp16-ep.yaml"
QWEN_PD = "scenarios/qwen3.5-397b/fp8-pcp8-dp8-1p1d.yaml"
AGENTX = "scenarios/kimi-k3/agentx.yaml"

# A target file as a fleet would write one. It reads env only in its hooks.
RECORDER = """
import os
import pathlib

from scenario.target import EnvNames, StepGroup
from scenario.targets.kubernetes import KubernetesTarget


class Recorder(KubernetesTarget):
    def reserved_env(self):
        return EnvNames(frozenset(os.environ.get("RESERVE", "").split()))

    def image(self, plan):
        return "${IMAGE}"

    def workload_name(self, plan):
        return "${WORKLOAD_NAME}"

    def escape_dollars(self):
        return True

    def steps_group(self, shape):
        if shape.roles[0].hosts > 1:
            return StepGroup(f"hosts-{shape.roles[0].hosts}", 2)
        return None

    def apply(self, manifest, plan):
        pathlib.Path(os.environ["RECORD"]).write_text(manifest)
        return 7
"""


def invoke(*args: str) -> Result:
    return CliRunner().invoke(main, list(args))


@pytest.fixture
def in_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """The fixture repo as the current directory, the Recorder in tmp_path."""
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("SCENARIO_TARGET", raising=False)
    monkeypatch.setenv("RECORD", str(tmp_path / "applied.yaml"))
    (tmp_path / "recorder.py").write_text(textwrap.dedent(RECORDER))
    return tmp_path


def test_render_without_a_target(in_repo: pathlib.Path) -> None:
    result = invoke("render", GLM)
    assert result.exit_code == 0, result.output
    assert "image: '<image>'" in result.stdout
    doc = load(result.stdout)
    assert doc["metadata"] == {"name": "<workload-name>"}
    (rjob,) = doc["spec"]["replicatedJobs"]
    assert rjob["template"]["spec"]["parallelism"] == 2


def test_render_with_a_target_file(in_repo: pathlib.Path) -> None:
    recorder = str(in_repo / "recorder.py")
    result = invoke("render", QWEN_PD, "--target", recorder)
    assert result.exit_code == 0, result.output
    assert "image: '${IMAGE}'" in result.stdout
    doc = load(result.stdout, IMAGE="img", WORKLOAD_NAME="wl")
    names = [rjob["name"] for rjob in doc["spec"]["replicatedJobs"]]
    assert names == ["prefill", "decode", "client"]


def test_render_takes_scenario_target(
    in_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCENARIO_TARGET", str(in_repo / "recorder.py"))
    result = invoke("render", GLM, "--set", "server.args.max-num-seqs=32")
    assert result.exit_code == 0, result.output
    assert "image: '${IMAGE}'" in result.stdout


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (("--target", "local"), 2, "local renders no manifest; use run --dry-run"),
        (("--target", "kubectl"), 1, "KubectlTarget has no image of its own"),
        (("--target", "nope"), 2, "no target 'nope'"),
    ],
)
def test_render_errors(
    in_repo: pathlib.Path, args: tuple[str, ...], code: int, message: str
) -> None:
    result = invoke("render", GLM, *args)
    assert result.exit_code == code
    assert message in result.stderr


def test_render_needs_a_point_of_a_sweep(in_repo: pathlib.Path) -> None:
    result = invoke("render", AGENTX)
    assert result.exit_code == 2
    assert "pick one with --point: c1, c2, c4, c8" in result.stderr
    assert invoke("render", AGENTX, "--point", "c4").exit_code == 0


def test_submit_applies_what_render_prints(in_repo: pathlib.Path) -> None:
    recorder = str(in_repo / "recorder.py")
    result = invoke("submit", AGENTX, "--point", "c2", "--target", recorder)
    assert result.exit_code == 7, result.output
    rendered = invoke("render", AGENTX, "--point", "c2", "--target", recorder)
    assert (in_repo / "applied.yaml").read_text() == rendered.stdout
    doc = load(rendered.stdout, IMAGE="img", WORKLOAD_NAME="wl")
    env = {
        e["name"]: e.get("value")
        for e in doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
            "spec"
        ]["containers"][0]["env"]
    }
    assert codec.decode(env["SCENARIO_PLAN"]).point == "c2"
    assert codec.decode_data(env["SCENARIO_RESOLVED"])["point"] == "c2"


def test_submit_needs_a_target(in_repo: pathlib.Path) -> None:
    result = invoke("submit", GLM)
    assert result.exit_code == 2
    assert "no target: pass --target or set SCENARIO_TARGET" in result.stderr


def test_submit_refuses_a_name_the_target_owns(
    in_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESERVE", "TPU_KV_SHM_POOL_GB")
    result = invoke("submit", QWEN_PD, "--target", str(in_repo / "recorder.py"))
    assert result.exit_code == 2
    assert "the scenario sets TPU_KV_SHM_POOL_GB, which the target sets" in (
        result.stderr
    )
    assert not (in_repo / "applied.yaml").exists()


def test_steps_one_per_point(in_repo: pathlib.Path) -> None:
    recorder = str(in_repo / "recorder.py")
    result = invoke("steps", AGENTX, "--target", recorder)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["scenario"] == "kimi-k3/agentx"
    steps = data["steps"]
    assert [s["key"] for s in steps] == [
        "kimi-k3-agentx-c1",
        "kimi-k3-agentx-c2",
        "kimi-k3-agentx-c4",
        "kimi-k3-agentx-c8",
    ]
    assert [s["after"] for s in steps] == [None, *[s["key"] for s in steps[:-1]]]
    first = steps[0]
    assert first["label"] == "kimi-k3-agentx c1"
    assert first["point"] == "c1"
    assert first["argv"] == [
        "scenario",
        "submit",
        AGENTX,
        "--target",
        recorder,
        "--point",
        "c1",
    ]
    assert first["command"] == (
        f"scenario submit {AGENTX} --target {recorder} --point c1"
    )
    assert all(s["group"] == {"name": "hosts-4", "limit": 2} for s in steps)


def test_steps_without_a_sweep_or_a_target(in_repo: pathlib.Path) -> None:
    result = invoke("steps", QWEN_PD, "--root", ".")
    assert result.exit_code == 0, result.output
    (step,) = json.loads(result.stdout)["steps"]
    assert step["argv"] == ["scenario", "submit", QWEN_PD, "--root", "."]
    assert (step["point"], step["after"], step["group"]) == (None, None, None)
    assert step["key"] == "qwen3-5-397b-fp8-pcp8-dp8-1p1d"


def test_validate_with_a_target(
    in_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = str(in_repo / "recorder.py")
    assert invoke("validate", "--target", recorder).exit_code == 0
    monkeypatch.setenv("RESERVE", "TPU_KV_SHM_POOL_GB")
    result = invoke("validate", "--target", recorder)
    assert result.exit_code == 1
    count = len(Resolver(REPO).scenario_files())
    assert result.stdout.splitlines()[-1] == f"{count} scenarios, 1 error"
    assert (
        "  scenarios/qwen3.5-397b/fp8-pcp8-dp8-1p1d.yaml: target kubernetes: the "
        "scenario sets TPU_KV_SHM_POOL_GB, which the target sets; remove it from "
        "the scenario"
    ) in result.stdout.splitlines()


def test_validate_checks_the_image(in_repo: pathlib.Path) -> None:
    """A scenario without server.image needs a target that supplies one."""
    k3_pd = "scenarios/kimi-k3/tp8-dp4-ep-1p1d.yaml"
    result = invoke("validate", k3_pd, "--target", "kubectl")
    assert result.exit_code == 1
    assert (
        "  scenarios/kimi-k3/tp8-dp4-ep-1p1d.yaml: target kubectl: KubectlTarget has "
        "no image of its own; set server.image or pass --set server.image=<image URI>"
    ) in result.stdout.splitlines()
    image = "server.image=registry.example/vllm:1"
    assert (
        invoke("validate", k3_pd, "--target", "kubectl", "--set", image).exit_code == 0
    )


def test_validate_with_a_broken_target(in_repo: pathlib.Path) -> None:
    result = invoke("validate", "--target", "missing.py")
    assert result.exit_code == 2
    assert "no target file" in result.stderr
