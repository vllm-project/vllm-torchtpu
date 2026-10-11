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
"""The kubectl target: a JobSet on the cluster kubectl points at.

For any GKE cluster with the JobSet controller. It is the tool's render plus
a memory volume at /dev/shm on TPU pods: JAX and vLLM lean on shared memory,
and the 64 MB default fails in ways that look unrelated. submit() creates the
JobSet in the current context and namespace, streams the driver's log, and
returns the driver container's exit code. Results stay in the pods; only the
log comes back. SIGINT or SIGTERM deletes the JobSet.
"""

import json
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from scenario.plan.types import Plan, RoleShape
from scenario.target import TargetInfo, Volume
from scenario.targets.kubernetes import KubernetesTarget

JOBSET_LABEL = "jobset.sigs.k8s.io/jobset-name"
REPLICATED_JOB_LABEL = "jobset.sigs.k8s.io/replicatedjob-name"
INDEX_LABEL = "batch.kubernetes.io/job-completion-index"

# Exit codes of submit() when the driver never ran; otherwise the driver's.
USAGE = 2
BRINGUP = 3
FAILED = 1


class KubectlError(RuntimeError):
    """kubectl is missing, failed, or answered something unreadable."""


class Kubectl:
    """kubectl calls with bounded waits; the log stream is the one unbounded."""

    def __init__(self, program: str = "kubectl", timeout_s: float = 120) -> None:
        self.program = program
        self.timeout_s = timeout_s

    def run(
        self, *args: str, stdin: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        try:
            done = subprocess.run(
                [self.program, *args],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError:
            raise KubectlError(f"{self.program} is not on PATH") from None
        except subprocess.TimeoutExpired:
            raise KubectlError(
                f"{self.program} {' '.join(args)} gave no answer in "
                f"{self.timeout_s:.0f}s"
            ) from None
        if check and done.returncode != 0:
            raise KubectlError(
                f"{self.program} {' '.join(args)} failed: {done.stderr.strip()}"
            )
        return done

    def get(self, *args: str) -> dict[str, Any] | None:
        """`kubectl get ... -o json`, or None when the object does not exist."""
        done = self.run("get", *args, "-o", "json", check=False)
        if done.returncode != 0:
            if "NotFound" in done.stderr:
                return None
            raise KubectlError(
                f"{self.program} get {' '.join(args)} failed: {done.stderr.strip()}"
            )
        try:
            return json.loads(done.stdout)
        except ValueError as error:
            raise KubectlError(f"{self.program} get: not JSON: {error}") from None

    def has_jobset(self) -> bool:
        done = self.run(
            "api-resources", "--api-group=jobset.x-k8s.io", "-o", "name", check=False
        )
        return done.returncode == 0 and "jobsets" in done.stdout

    def stream(self, *args: str) -> int:
        """Run with the output going straight to ours, for `logs -f`."""
        sys.stdout.flush()
        return subprocess.run([self.program, *args]).returncode


def _condition(obj: dict[str, Any] | None, kind: str) -> dict[str, Any] | None:
    for condition in (obj or {}).get("status", {}).get("conditions", []):
        if condition.get("type") == kind and condition.get("status") == "True":
            return condition
    return None


def _exit_code(pod: dict[str, Any], container: str) -> int | None:
    for status in pod.get("status", {}).get("containerStatuses", []):
        if status.get("name") == container:
            terminated = status.get("state", {}).get("terminated")
            if terminated is not None:
                return int(terminated.get("exitCode", FAILED))
    return None


def _say(message: str) -> None:
    print(f"[kubectl] {message}", file=sys.stderr, flush=True)


@contextmanager
def _deleting_on_signal(kubectl: Kubectl, name: str) -> Iterator[None]:
    """SIGINT or SIGTERM deletes the JobSet, then exits 128 + the signal."""

    def stop(signum: int, _frame: object) -> None:
        _say(f"{signal.Signals(signum).name}: deleting jobset {name}")
        try:
            kubectl.run("delete", "jobset", name, "--wait=false", check=False)
        except KubectlError as error:
            _say(f"could not delete jobset {name}: {error}")
        raise SystemExit(128 + signum)

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class KubectlTarget(KubernetesTarget):
    """Applies with kubectl and waits for the driver."""

    program = "kubectl"
    poll_s = 5.0

    def info(self) -> TargetInfo:
        return TargetInfo("kubectl", None, None)

    def volumes(self, role: RoleShape) -> Sequence[Volume]:
        if role.chipless:
            return ()
        return (Volume("dshm", "/dev/shm", "Memory"),)

    def apply(self, manifest: str, plan: Plan) -> int:
        name = self.workload_name(plan)
        driver = next(role.name for role in plan.shape().roles if role.driver)
        kubectl = Kubectl(self.program)
        try:
            if not kubectl.has_jobset():
                _say(
                    "the cluster serves no jobsets.jobset.x-k8s.io; install the "
                    "JobSet controller"
                )
                return USAGE
            if kubectl.get("jobset", name) is not None:
                _say(
                    f"jobset {name} exists, from an earlier run of this Plan; "
                    f"delete it first: kubectl delete jobset {name}"
                )
                return USAGE
        except KubectlError as error:
            _say(str(error))
            return USAGE
        try:
            kubectl.run("create", "-f", "-", stdin=manifest)
        except KubectlError as error:
            _say(str(error))
            return BRINGUP
        _say(f"created jobset {name}; following {driver}/0")
        with _deleting_on_signal(kubectl, name):
            return self._follow(kubectl, name, driver)

    def _driver_pod(
        self, kubectl: Kubectl, name: str, driver: str
    ) -> dict[str, Any] | None:
        selector = f"{JOBSET_LABEL}={name},{REPLICATED_JOB_LABEL}={driver}"
        pods = (kubectl.get("pods", "-l", selector) or {}).get("items", [])
        heads = [
            pod
            for pod in pods
            if pod.get("metadata", {}).get("labels", {}).get(INDEX_LABEL) == "0"
        ]
        if not heads:
            return None
        return max(heads, key=lambda pod: pod["metadata"].get("creationTimestamp", ""))

    def _follow(self, kubectl: Kubectl, name: str, driver: str) -> int:
        container = self.container_name()
        streamed: set[str] = set()
        phase = None
        while True:
            try:
                pod = self._driver_pod(kubectl, name, driver)
                jobset = None if pod is not None else kubectl.get("jobset", name)
            except KubectlError as error:
                _say(f"{error}; retrying")
                time.sleep(self.poll_s)
                continue
            if pod is not None:
                pod_name = pod["metadata"]["name"]
                now = pod.get("status", {}).get("phase")
                if now != phase:
                    _say(f"{pod_name}: {now}")
                    phase = now
                code = _exit_code(pod, container)
                if pod_name not in streamed and now in (
                    "Running",
                    "Succeeded",
                    "Failed",
                ):
                    streamed.add(pod_name)
                    kubectl.stream("logs", "-f", pod_name, "-c", container)
                    continue
                if code is not None:
                    _say(f"{pod_name} exited {code}")
                    return code
            elif jobset is None:
                _say(f"jobset {name} is gone before {driver}/0 ran")
                return FAILED
            elif (failed := _condition(jobset, "Failed")) is not None:
                _say(
                    f"jobset {name} failed before {driver}/0 ran: "
                    f"{failed.get('reason', '')} {failed.get('message', '')}".rstrip()
                )
                return FAILED
            time.sleep(self.poll_s)
