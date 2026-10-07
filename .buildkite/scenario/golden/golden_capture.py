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
"""Run today's kube scripts unmodified, with stub programs, and record every call.

A capture copies the scripts into a scratch repository, runs the pipeline
step's command as the agent would (run.sh or run_multihost.sh, which end in
the launcher), then plays the launcher: substitutes and reads the manifest,
and starts every pod's container command with its env. Each pod is a process
group of its own; the run ends when the driver pod exits and every other pod
has started what it serves, and the groups are then killed.

The scratch copy differs from the checkout in what a pod image and an agent
provide and this machine does not: the image's checkout paths point at the
copy, `#!/bin/bash` resolves through PATH (so a bash >= 4.4 runs everything),
the launcher is a stub, and the K3 correctness script is a recorder (it is run
as `bash <path>`, which PATH cannot intercept). Container commands run with
`bash -c`, not `bash -lc`: a login shell's profile resets PATH and would drop
the stubs.
"""

import contextlib
import json
import os
import pathlib
import re
import shutil
import signal
import string
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from ruamel.yaml import YAML

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
STUBS = HERE / "stubs.py"

# What a pod image has at these paths: the repository.
IMAGE_CHECKOUTS = ("/root/torchtpu-vllm", "/workspace/vllm-torchtpu")
# What the scripts need, copied into the scratch repository.
COPIED = (
    "scripts/vllm/benchmarking",
    "scripts/vllm/benchmark/inferencex",
    "scripts/vllm/integration",
    ".buildkite/kubernetes",
    ".buildkite/scripts",
    "examples/disagg",
)
LAUNCHER = "/opt/launcher/launch"
RECORDED_SCRIPTS = ("scripts/vllm/integration/run_k3_1p1d_disagg_correctness.sh",)
STUB_PROGRAMS = (
    "vllm",
    "curl",
    "getent",
    "timeout",
    "ray",
    "python3",
    "buildkite-agent",
    "stdbuf",
    "lm_eval",
    "pip",
    "gsutil",
    "launch",
)
# Stubs with nothing to record: process and port lookups find nothing, and
# sleeps return at once.
SHELL_STUBS = {"pkill": 0, "lsof": 0, "fuser": 0, "pgrep": 1, "sleep": 0}
WORKLOAD = "wl"
IMAGE = "registry.example/vllm-torchtpu:golden"
CONTAINER = "workload"
TPU_RESOURCE = "google.com/tpu"
# The launcher sweeps BUILDKITE_* into the pod, except these (launch.py).
BUILDKITE_DENY = frozenset(
    {
        "BUILDKITE_COMMAND",
        "BUILDKITE_PLUGINS",
        "BUILDKITE_AGENT_JOB_API_SOCKET",
        "BUILDKITE_AGENT_JOB_API_TOKEN",
        "BUILDKITE_OIDC_TOKEN_PATH",
    }
)
# The agent's environment for a scheduled nightly on main.
AGENT_ENV = {
    "BUILDKITE": "true",
    "BUILDKITE_BRANCH": "main",
    "BUILDKITE_PULL_REQUEST": "false",
    "BUILDKITE_BUILD_NUMBER": "1",
    "BUILDKITE_COMMIT": "0" * 40,
    "BUILDKITE_JOB_ID": "golden",
    "BUILDKITE_SOURCE": "schedule",
    "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
    "NIGHTLY": "1",
}
# The fleet's pod defaults for the workload container (pod_defaults.yaml).
POD_DEFAULTS_ENV = {
    "HF_HOME": "/cache/hf",
    "HF_TOKEN": "hf_golden",
    "PYTHONUNBUFFERED": "1",
}
# What GKE may hand a TPU pod of a slice; a side that empties it must empty
# all of it.
GKE_TPU_ENV = {
    "MEGASCALE_COORDINATOR_ADDRESS": "gke-coordinator",
    "MEGASCALE_NUM_SLICES": "1",
    "MEGASCALE_PORT": "8080",
    "MEGASCALE_SLICE_ID": "0",
    "TPU_PROCESS_ADDRESSES": "gke-process-addresses",
    "TPU_PROCESS_PORT": "8476",
    "TPU_WORKER_HOSTNAMES": "gke-worker-hostnames",
}
RUN_TIMEOUT_S = 120


@dataclass(frozen=True)
class Step:
    """A pipeline step's command and environment, as the agent runs them."""

    command: str
    env: Mapping[str, str]
    pipeline_env: Mapping[str, str]


@dataclass(frozen=True)
class DirectPod:
    """One pod with no step or manifest, for scripts no lane runs."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    chips: bool = True


@dataclass
class Pod:
    group: str
    index: int
    argv: list[str]
    env: dict[str, str]
    chips: bool
    ip: str
    dns: str | None

    @property
    def host(self) -> str:
        return f"{self.group}-{self.index}"


@dataclass
class Capture:
    root: pathlib.Path
    state: pathlib.Path
    pods: list[Pod]
    records: list[dict[str, Any]]
    base_env: dict[str, str]
    # Host -> exit code of the pod's command; None when it was still serving.
    exits: dict[str, int | None] = field(default_factory=dict)

    def log_tail(self, host: str, lines: int = 30) -> str:
        return _tail(self.state / "logs" / f"{host}.log", lines)


class CaptureError(RuntimeError):
    """Today's side did not run through; the message holds the logs."""


def _tail(path: pathlib.Path, lines: int) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _logs(state: pathlib.Path, hosts: Sequence[str]) -> str:
    return "\n".join(
        f"--- {host} ---\n{_tail(state / 'logs' / f'{host}.log', 30)}" for host in hosts
    )


def _yaml() -> YAML:
    return YAML(typ="safe", pure=True)


def load_yaml(text: str) -> Any:
    return _yaml().load(StringIO(text))


def bash_version(path: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [path, "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def find_bash() -> str | None:
    """A bash >= 4.4: $GOLDEN_BASH, else the first one found that is new enough."""
    wanted = os.environ.get("GOLDEN_BASH")
    if wanted:
        return wanted if (bash_version(wanted) or (0, 0)) >= (4, 4) else None
    candidates = [shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"]
    candidates.append("/bin/bash")
    for candidate in candidates:
        if candidate and (bash_version(candidate) or (0, 0)) >= (4, 4):
            return candidate
    return None


def _write_shims(stubs: pathlib.Path, bash: str) -> None:
    stubs.mkdir(parents=True)
    python = sys.executable
    for name in STUB_PROGRAMS:
        shim = stubs / name
        shim.write_text(f'#!/bin/sh\nexec "{python}" -I -S "{STUBS}" {name} "$@"\n')
        shim.chmod(0o755)
    for name, code in SHELL_STUBS.items():
        (stubs / name).write_text(f"#!/bin/sh\nexit {code}\n")
        (stubs / name).chmod(0o755)
    (stubs / "bash").symlink_to(bash)


def make_repo(root: pathlib.Path, stubs: pathlib.Path) -> None:
    """The scratch repository: today's scripts, adapted as the module says."""
    for part in COPIED:
        shutil.copytree(
            REPO / part,
            root / part,
            ignore=shutil.ignore_patterns("__pycache__"),
            symlinks=True,
        )
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in (".sh", ".yaml", ".py"):
            continue
        text = path.read_text()
        new = rewrite(text, root)
        new = new.replace(LAUNCHER, str(stubs / "launch"))
        if new.startswith("#!/bin/bash"):
            new = "#!/usr/bin/env bash" + new[len("#!/bin/bash") :]
        if new != text:
            path.write_text(new)
    for relative in RECORDED_SCRIPTS:
        path = root / relative
        path.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -I -S "{STUBS}" script "$0" "$@"\n'
        )


def rewrite(text: str, root: pathlib.Path) -> str:
    for checkout in IMAGE_CHECKOUTS:
        text = text.replace(checkout, str(root))
    return text


_INTERPOLATION = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def interpolate(value: str, env: Mapping[str, str]) -> str:
    """Buildkite's ${NAME} and ${NAME:-default} in step env values."""
    return _INTERPOLATION.sub(
        lambda m: env.get(m.group(1)) or (m.group(2) or ""), value
    )


def find_step(pipeline: pathlib.Path, key: str | None, label: str | None) -> Step:
    """A step of a pipeline file by key or label, with the pipeline's env."""
    doc = load_yaml(pipeline.read_text())

    def walk(steps: Sequence[Any]) -> Any:
        for step in steps:
            if not isinstance(step, dict):
                continue
            if (key and step.get("key") == key) or (
                label and step.get("label") == label
            ):
                return step
            found = walk(step.get("steps") or [])
            if found is not None:
                return found
        return None

    step = walk(doc["steps"])
    if step is None:
        raise LookupError(f"{pipeline.name}: no step {key or label}")
    commands = step.get("commands") or [step.get("command")]
    (command,) = commands
    pipeline_env = {k: str(v) for k, v in (doc.get("env") or {}).items()}
    env = {k: str(v) for k, v in (step.get("env") or {}).items()}
    return Step(command=command.strip(), env=env, pipeline_env=pipeline_env)


def _parse_launch(argv: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    options: dict[str, Any] = {"env": []}
    i = 0
    while i < len(argv) and argv[i] != "--":
        name = argv[i]
        if name == "--env":
            options["env"].append(argv[i + 1])
            i += 2
        elif name == "--cpu":
            options["cpu"] = True
            i += 1
        else:
            options[name.lstrip("-")] = argv[i + 1]
            i += 2
    return options, list(argv[i + 1 :])


def _forwarded(names: Sequence[str], env: Mapping[str, str]) -> dict[str, str]:
    swept = sorted(
        k for k in env if k.startswith("BUILDKITE_") and k not in BUILDKITE_DENY
    )
    return {n: env[n] for n in dict.fromkeys([*names, *swept]) if env.get(n)}


def _field_ref(path: str, pod: Pod) -> str:
    if path == "status.podIP":
        return pod.ip
    if "job-completion-index" in path:
        return str(pod.index)
    raise ValueError(f"no fake for fieldRef {path}")


def _container_env(entries: Sequence[Mapping[str, Any]], pod: Pod) -> dict[str, str]:
    env = {}
    for entry in entries:
        if "value" in entry:
            env[entry["name"]] = str(entry["value"])
        elif "fieldRef" in (entry.get("valueFrom") or {}):
            env[entry["name"]] = _field_ref(
                entry["valueFrom"]["fieldRef"]["fieldPath"], pod
            )
        else:
            env[entry["name"]] = "golden-secret"
    return env


def _container_argv(container: Mapping[str, Any], bash: str) -> list[str]:
    argv = [*container.get("command", []), *container.get("args", [])]
    if argv and argv[0] == "/bin/bash":
        argv[0] = bash
        if len(argv) > 1 and argv[1] == "-lc":
            argv[1] = "-c"
    return argv


def pods_from_manifest(
    text: str, launcher_env: Mapping[str, str], forwarded: Mapping[str, str], bash: str
) -> list[Pod]:
    """The pods a JobSet manifest asks for, as the launcher and Kubernetes run them."""
    mapping = {**launcher_env, "WORKLOAD_NAME": WORKLOAD, "IMAGE": IMAGE}
    doc = load_yaml(string.Template(text).substitute(mapping))
    pods = []
    for r, job in enumerate(doc["spec"]["replicatedJobs"]):
        spec = job["template"]["spec"]
        template = spec["template"]["spec"]
        (container,) = [c for c in template["containers"] if c["name"] == CONTAINER]
        limits = (container.get("resources") or {}).get("limits") or {}
        for index in range(int(spec.get("parallelism", 1))):
            pod = Pod(
                group=job["name"],
                index=index,
                argv=_container_argv(container, bash),
                env={},
                chips=TPU_RESOURCE in limits,
                ip=f"10.{r + 1}.0.{index + 1}",
                dns=f"{WORKLOAD}-{job['name']}-0-{index}.{WORKLOAD}",
            )
            pod.env = {
                **_container_env(container.get("env") or [], pod),
                **forwarded,
                "JOB_COMPLETION_INDEX": str(index),
            }
            pods.append(pod)
    return pods


def _run_agent(step: Step, root: pathlib.Path, base: Mapping[str, str]) -> None:
    env = {**base, **AGENT_ENV, **step.pipeline_env}
    env |= {k: interpolate(v, env) for k, v in step.env.items()}
    env["GOLDEN_HOST"] = "agent"
    state = pathlib.Path(base["GOLDEN_STATE"])
    with (state / "logs" / "agent.log").open("w") as out:
        code = subprocess.run(
            ["bash", "-c", step.command],
            cwd=root,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=60,
        ).returncode
    if code != 0:
        raise CaptureError(f"the step exited {code}\n{_logs(state, ['agent'])}")


def _launch_record(state: pathlib.Path) -> dict[str, Any]:
    (record,) = [r for r in read_records(state) if r["tool"] == "launch"]
    return record


def read_records(state: pathlib.Path) -> list[dict[str, Any]]:
    files = sorted((state / "records").glob("*.json"))
    records = [json.loads(path.read_text()) for path in files]
    return sorted(records, key=lambda r: r["time"])


class _Started:
    """Hosts that have started what they serve: `vllm serve`, or a Ray worker."""

    def __init__(self, state: pathlib.Path) -> None:
        self.records = state / "records"
        self.seen: set[str] = set()
        self.hosts: set[str] = set()

    def poll(self) -> set[str]:
        for path in self.records.glob("*.json"):
            if path.name in self.seen:
                continue
            self.seen.add(path.name)
            record = json.loads(path.read_text())
            argv = record["argv"]
            serve = record["tool"] == "vllm" and argv[:1] == ["serve"]
            joined = record["tool"] == "ray" and argv[:1] == ["start"]
            if serve or (joined and "--head" not in argv):
                self.hosts.add(record["host"])
        return self.hosts


def capture(
    work: pathlib.Path,
    launch: Step | DirectPod,
    driver: str | None,
    bash: str,
    mutate: Callable[[pathlib.Path], None] | None = None,
) -> Capture:
    """Run today's side of a case and return every recorded call.

    Args:
        work: an empty scratch directory.
        launch: the pipeline step to run, or a pod to start directly.
        driver: the host whose exit ends the run (`<group>-<index>`); None
            when every pod only serves.
        bash: a bash >= 4.4.
        mutate: edits the scratch repository before anything runs.
    """
    root, state, stubs = work / "repo", work / "state", work / "stubs"
    for part in ("records", "http", "tcp", "home", "logs"):
        (state / part).mkdir(parents=True)
    (state / "pulled-model").mkdir()
    (state / "pulled-model" / "config.json").write_text("{}")
    _write_shims(stubs, bash)
    make_repo(root, stubs)
    if mutate is not None:
        mutate(root)
    base_env = {
        "PATH": f"{stubs}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(state / "home"),
        "LANG": "C.UTF-8",
        "GOLDEN_STATE": str(state),
        "GOLDEN_ROOT": str(root),
    }

    if isinstance(launch, DirectPod):
        pod = Pod(
            "pod",
            0,
            list(launch.argv),
            dict(launch.env),
            launch.chips,
            "10.1.0.1",
            None,
        )
        pods = [pod]
    else:
        _run_agent(launch, root, base_env)
        record = _launch_record(state)
        options, command = _parse_launch(record["argv"])
        forwarded = _forwarded(options["env"], record["env"])
        if "manifest" in options:
            text = rewrite(pathlib.Path(options["manifest"]).read_text(), root)
            pods = pods_from_manifest(text, record["env"], forwarded, bash)
        else:
            chips = "cpu" not in options
            pods = [Pod("pod", 0, command, dict(forwarded), chips, "10.1.0.1", None)]

    hosts = {
        "names": {pod.dns: pod.host for pod in pods if pod.dns},
        "ips": {pod.ip: pod.host for pod in pods},
        "host_ips": {pod.host: pod.ip for pod in pods},
    }
    (state / "hosts.json").write_text(json.dumps(hosts))
    for pod in pods:
        tpu = GKE_TPU_ENV if pod.chips else {}
        pod.env = {
            **base_env,
            **POD_DEFAULTS_ENV,
            **tpu,
            **pod.env,
            "GOLDEN_HOST": pod.host,
        }

    exits = _run_pods(pods, root, state, driver)
    return Capture(root, state, pods, read_records(state), base_env, exits)


def _run_pods(
    pods: Sequence[Pod], root: pathlib.Path, state: pathlib.Path, driver: str | None
) -> dict[str, int | None]:
    """Start every pod; end when the driver exits and the others serve."""
    processes: dict[str, subprocess.Popen[bytes]] = {}
    exits: dict[str, int | None] = {}
    with contextlib.ExitStack() as stack:
        for pod in pods:
            log = stack.enter_context((state / "logs" / f"{pod.host}.log").open("wb"))
            processes[pod.host] = subprocess.Popen(
                pod.argv,
                cwd=root,
                env=pod.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + RUN_TIMEOUT_S
            started = _Started(state)
            while len(exits) < len(processes):
                serving = started.poll()
                for host, process in processes.items():
                    if host in exits:
                        continue
                    code = process.poll()
                    if code is not None or (host != driver and host in serving):
                        exits[host] = code
                        _kill(process)
                if len(exits) < len(processes) and time.monotonic() > deadline:
                    pending = sorted(set(processes) - set(exits))
                    raise CaptureError(
                        f"pods {', '.join(pending)} did not finish within "
                        f"{RUN_TIMEOUT_S} s\n{_logs(state, pending)}"
                    )
                time.sleep(0.05)
        finally:
            for host, process in processes.items():
                if host not in exits:
                    _kill(process)
    return exits


def _kill(process: subprocess.Popen[bytes]) -> None:
    """Kill the pod's process group: a pod's servers outlive its command."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()
