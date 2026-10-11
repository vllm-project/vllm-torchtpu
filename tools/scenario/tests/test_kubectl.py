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
"""The kubectl target against a stand-in kubectl on PATH (fake_kubectl.py)."""

import json
import os
import pathlib
import stat
import sys
from typing import Any

import pytest
from manifest_checks import container, load, pod_spec, replicated_jobs
from plan_fixtures import Data, build
from scenario.plan.types import Plan
from scenario.target import EnvNames, Target, TargetInfo
from scenario.targets.kubectl import KubectlTarget
from scenario.targets.kubernetes import RenderError

FAKE = pathlib.Path(__file__).resolve().parent / "fake_kubectl.py"


class Cluster:
    """The fake's script and what it recorded."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root

    def script(self, **state: Any) -> None:
        (self.root / "state.json").write_text(json.dumps(state))

    def calls(self) -> list[list[str]]:
        lines = (self.root / "calls.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def created(self) -> str:
        return (self.root / "created.yaml").read_text()


@pytest.fixture
def cluster(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Cluster:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    program = bin_dir / "kubectl"
    program.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KUBECTL", str(tmp_path))
    found = Cluster(tmp_path)
    found.script()
    return found


@pytest.fixture
def target() -> KubectlTarget:
    found = KubectlTarget()
    found.poll_s = 0.01
    return found


def uri(data: Data) -> Plan:
    data["server"]["image"] = "registry.example/vllm:1"
    return build(data)


def driver_pod(name: str, phase: str, code: int | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {"phase": phase}
    if code is not None:
        status["containerStatuses"] = [
            {"name": "main", "state": {"terminated": {"exitCode": code}}}
        ]
    return {
        "metadata": {
            "name": name,
            "creationTimestamp": "2026-10-10T00:00:00Z",
            "labels": {"batch.kubernetes.io/job-completion-index": "0"},
        },
        "status": status,
    }


def test_protocol_and_hooks(variants: dict[str, Data]) -> None:
    target = KubectlTarget()
    assert isinstance(target, Target)
    assert target.info() == TargetInfo("kubectl", None, None)
    assert target.reserved_env() == EnvNames(frozenset())
    doc = load(target.render(uri(variants["pd"])))
    jobs = replicated_jobs(doc)
    for name in ("prefill", "decode"):
        assert pod_spec(jobs[name])["volumes"] == [
            {"name": "dshm", "emptyDir": {"medium": "Memory"}}
        ]
        assert container(jobs[name])["volumeMounts"] == [
            {"name": "dshm", "mountPath": "/dev/shm"}
        ]
    assert "volumes" not in pod_spec(jobs["client"])
    with pytest.raises(RenderError, match="--set server.image="):
        target.render(build(variants["single"]))


@pytest.mark.parametrize("code", [0, 1, 4])
def test_returns_the_driver_exit_code(
    cluster: Cluster,
    target: KubectlTarget,
    variants: dict[str, Data],
    capfd: pytest.CaptureFixture[str],
    code: int,
) -> None:
    planned = uri(variants["single"])
    name = target.workload_name(planned)
    cluster.script(
        pods=[
            [],
            [driver_pod(f"{name}-server-0-0", "Pending")],
            [driver_pod(f"{name}-server-0-0", "Running")],
            [driver_pod(f"{name}-server-0-0", "Failed" if code else "Succeeded", code)],
        ],
        logs="served\n",
    )
    assert target.submit(planned) == code
    assert cluster.created() == target.render(planned)
    calls = cluster.calls()
    assert calls[:3] == [
        ["api-resources", "--api-group=jobset.x-k8s.io", "-o", "name"],
        ["get", "jobset", name, "-o", "json"],
        ["create", "-f", "-"],
    ]
    selector = (
        f"jobset.sigs.k8s.io/jobset-name={name},"
        "jobset.sigs.k8s.io/replicatedjob-name=server"
    )
    assert ["get", "pods", "-l", selector, "-o", "json"] in calls
    assert calls.count(["logs", "-f", f"{name}-server-0-0", "-c", "main"]) == 1
    out, err = capfd.readouterr()
    assert out == "served\n"
    assert f"{name}-server-0-0 exited {code}" in err


def test_follows_the_chipless_driver(
    cluster: Cluster, target: KubectlTarget, variants: dict[str, Data]
) -> None:
    planned = uri(variants["pd"])
    name = target.workload_name(planned)
    cluster.script(pods=[[driver_pod(f"{name}-client-0-0", "Succeeded", 0)]])
    assert target.submit(planned) == 0
    assert any("replicatedjob-name=client" in " ".join(c) for c in cluster.calls())


def test_refuses_an_existing_jobset(
    cluster: Cluster,
    target: KubectlTarget,
    variants: dict[str, Data],
    capfd: pytest.CaptureFixture[str],
) -> None:
    cluster.script(exists=True)
    planned = uri(variants["single"])
    assert target.submit(planned) == 2
    assert "kubectl delete jobset" in capfd.readouterr().err
    assert ["create", "-f", "-"] not in cluster.calls()


def test_needs_the_jobset_api(
    cluster: Cluster,
    target: KubectlTarget,
    variants: dict[str, Data],
    capfd: pytest.CaptureFixture[str],
) -> None:
    cluster.script(jobset_api=False)
    assert target.submit(uri(variants["single"])) == 2
    assert "install the JobSet controller" in capfd.readouterr().err


def test_without_kubectl(
    target: KubectlTarget,
    variants: dict[str, Data],
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(KubectlTarget, "program", "no-such-kubectl")
    assert target.submit(uri(variants["single"])) == 2
    assert "no-such-kubectl is not on PATH" in capfd.readouterr().err


def test_a_failed_create(
    cluster: Cluster, target: KubectlTarget, variants: dict[str, Data]
) -> None:
    cluster.script(create_fails="admission webhook denied")
    assert target.submit(uri(variants["single"])) == 3


def test_a_jobset_that_fails_before_the_driver_runs(
    cluster: Cluster,
    target: KubectlTarget,
    variants: dict[str, Data],
    capfd: pytest.CaptureFixture[str],
) -> None:
    failed = {"type": "Failed", "status": "True", "reason": "FailedCreate"}
    cluster.script(pods=[[]], jobset={"status": {"conditions": [failed]}})
    assert target.submit(uri(variants["single"])) == 1
    assert "failed before server/0 ran: FailedCreate" in capfd.readouterr().err


def test_a_signal_deletes_the_jobset(
    cluster: Cluster, target: KubectlTarget, variants: dict[str, Data]
) -> None:
    planned = uri(variants["single"])
    cluster.script(pods=[[{"signal": "TERM"}]])
    with pytest.raises(SystemExit) as stopped:
        target.submit(planned)
    assert stopped.value.code == 143
    name = target.workload_name(planned)
    assert ["delete", "jobset", name, "--wait=false"] in cluster.calls()
