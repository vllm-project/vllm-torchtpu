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
"""The CI fleet's target: a scenario point as a launcher workload.

A Buildkite step on the kube queue runs `scenario submit FILE` with
SCENARIO_TARGET pointing here. The tool renders the JobSet; this file adds
what the fleet's launcher (ci-infra kueue/launcher/launch.py) expects and
what today's hand-written manifests carry, and nothing about what runs:

- the container `workload`, `${IMAGE}` and `${WORKLOAD_NAME}`, which the
  launcher substitutes over the whole file (so every other `$` is doubled),
  and the annotation that opts the pods into the fleet's defaults (caches,
  HF_TOKEN, tolerations, retries);
- the compile caches on the fleet's bucket only where today's manifests set
  them, on multi-host and multi-role pods, with the profile_cache shadow;
  single-host pods keep ~/.cache/vllm, as today's single-host steps do;
- Ray's log to stderr, /dev/shm for a role's scratch, and the CI log
  settings;
- the chipless driver's node class and service account;
- uploads as after-tasks: artifacts per result and at the end, BigQuery
  rows on main.

apply() takes an unset server.image as the build's CI image, read from the
build's meta-data, writes the manifest into the checkout and execs the
launcher. Nothing here reads the environment or runs a command at import or
construction, so `scenario render` works anywhere. Target code never runs in
a pod: pods get the after-tasks as data.
"""

import os
import pathlib
import re
import string
import subprocess
import sys
from collections.abc import Mapping, Sequence

from ruamel.yaml import YAML
from scenario.plan.types import READINESS_MARGIN_S, Plan, RoleShape, Shape
from scenario.target import (
    AfterTask,
    CpuPod,
    EnvNames,
    JobPolicy,
    StepGroup,
    TargetInfo,
    Volume,
)
from scenario.targets.kubernetes import (
    ACCELERATOR_LABEL,
    TOPOLOGY_LABEL,
    TPU_RESOURCE,
    KubernetesTarget,
)

LAUNCHER = "/opt/launcher/launch"
DEFAULTS_ANNOTATION = "tpu-ci.google.com/defaults"
CONTAINER = "workload"
# The launcher fills these two over the whole manifest (string.Template); the
# step's env fills any other ${NAME}, which would be a leak.
SUBSTITUTED = ("IMAGE", "WORKLOAD_NAME")
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
SERVICE_ACCOUNTS = ("default", "tpu-workload")
# Today's ARTIFACTS_DIR; artifact names keep the perf_eval_results/ prefix.
RESULTS_ROOT = "perf_eval_results"
MANIFEST_DIR = "scenario-manifests"
# Mounted on every TPU pod by the fleet: the compilation bucket and the HF hub.
JAX_CACHE = "/cache/jax"
HF_DATASETS_CACHE = "/cache/hf/hub/datasets"
# run_eval_flow.sh's default under the fleet's HF_HOME, so kube runs share it.
LM_EVAL_CACHE = "/cache/hf/hub/lm_eval_requests"
# The two Kimi-K3 pairs each take the queue's whole 2x2x4 quota, so steps on
# that shape run one at a time (#1798).
SLICE_2X2X4 = StepGroup("tpu/v7x-2x2x4", 1)
CPU_POD = CpuPod({"cloud.google.com/compute-class": "worker-cpu"}, "tpu-workload")

ENV_NAMES = (
    "VLLM_CACHE_ROOT",
    "VLLM_XLA_CACHE_PATH",
    "HF_DATASETS_CACHE",
    "RAY_LOG_TO_STDERR",
    "TMPDIR",
    "RAY_TMPDIR",
    "FORCE_COLOR",
    "TQDM_MININTERVAL",
)
AFTER_TASK_ENV = (
    "CI_RUNNER",
    "CREATED_BY",
    "RUN_TYPE",
    "REPOSITORY",
    "DOCKER_IMAGE",
    "TPU_NAME",
)
# The fleet's (pod_defaults.yaml), and today's names that no scenario may
# carry over: a scenario setting one would be a leak from an old step.
FLEET_ENV = ("HF_TOKEN", "HF_HOME", "ARTIFACTS_DIR", "SKIP_SPANNER_UPLOAD")

# Step env that today's run.sh steps forward and a scenario step ignores; set
# on a scenario step it would silently do nothing. Name -> what replaces it.
STEP_ENV = {
    "VLLM_ENGINE_READY_TIMEOUT_S": "server.startup_timeout",
    "BENCHMARK_WARMUP_RUNS": "tasks.perf.warmup_runs",
    "RUN_LM_EVAL": "tasks.eval",
    "RUN_CODE_EVAL": "tasks.eval",
    "EVAL_SOFT_FAIL": "a soft eval gate, which scenarios do not have yet",
    "MODEL_URI": "server.model.path",
    "MODEL_PATH": "server.model.path",
    "SERVED_MODEL_NAME": "server.model.name",
    "TMPDIR": "server.scratch",
    "RAY_TMPDIR": "server.scratch",
    "VLLM_CACHE_ROOT": "nothing: this target sets the cache paths",
    "VLLM_XLA_CACHE_PATH": "nothing: this target sets the cache paths",
    "TPU_ACCELERATOR_TYPE": "server.accelerator",
    "TPU_MULTIHOST_BACKEND": "server.multihost",
    "SHAPE": "the scenario's hardware fields",
    "MULTIHOST_ARGS_B64": "the scenario file",
    "CAPTURE_PROFILE": "nothing yet: scenarios do not profile",
}

# What the end-of-run upload runs: everything no per-result upload took.
_UPLOAD_REST = '[ -z "$1" ] || buildkite-agent artifact upload "$1"'


def git_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def _cache_dirs(role: RoleShape) -> tuple[str, str] | None:
    """VLLM_CACHE_ROOT and VLLM_XLA_CACHE_PATH, where today's manifests set them."""
    if role.chipless or not (role.hosts > 1 or role.multi_role):
        return None
    below = f"/{role.name}" if role.cache == "private" else ""
    return f"{JAX_CACHE}/vllm{below}", f"{JAX_CACHE}/xla{below}"


def chip_shapes(shape: Shape) -> set[tuple[str | None, str | None, int]]:
    return {
        (role.accelerator, role.topology, role.chips_per_host)
        for role in shape.roles
        if not role.chipless
    }


def tpu_name(shape: Shape) -> str:
    """The device name BigQuery rows carry, e.g. tpu7x-2x2x1 (#1741).

    upload_results.py builds the same from the pod's TPU env when CI_RUNNER
    is kube; any other runner (kube-scenario during parity) needs it set.
    """
    ((accelerator, topology, _),) = chip_shapes(shape)
    generation = re.sub(r"^tpu-|-slice$", "", accelerator or "")
    return f"{generation}-{topology}"


def uploads_rows(env: Mapping[str, str]) -> bool:
    """Whether results go to BigQuery: on main and not a PR, unless BQ_UPLOAD=0.

    run_eval_flow.sh's rule; BQ_UPLOAD=0 is how ad-hoc runs opt out.
    """
    if env.get("BQ_UPLOAD") == "0":
        return False
    branch = env.get("BUILDKITE_BRANCH")
    pull_request = env.get("BUILDKITE_PULL_REQUEST", "false")
    return not branch or (branch == "main" and pull_request in ("", "false"))


class TpuCi(KubernetesTarget):
    launcher = LAUNCHER

    def info(self) -> TargetInfo:
        commit = os.environ.get("BUILDKITE_COMMIT") or git_commit()
        return TargetInfo("tpu_ci", commit, RESULTS_ROOT)

    def reserved_env(self) -> EnvNames:
        names = frozenset((*ENV_NAMES, *AFTER_TASK_ENV, *FLEET_ENV))
        # The launcher writes every BUILDKITE_* of the agent over the pod's.
        return EnvNames(names, ("BUILDKITE_",))

    def image(self, plan: Plan) -> str:
        return "${IMAGE}"

    def workload_name(self, plan: Plan) -> str:
        return "${WORKLOAD_NAME}"

    def container_name(self) -> str:
        return CONTAINER

    def annotations(self) -> Mapping[str, str]:
        return {DEFAULTS_ANNOTATION: "standard"}

    def escape_dollars(self) -> bool:
        return True

    def env(self, role: RoleShape) -> Mapping[str, str]:
        env = {}
        dirs = _cache_dirs(role)
        if dirs is not None:
            env["VLLM_CACHE_ROOT"], env["VLLM_XLA_CACHE_PATH"] = dirs
            # Only $HF_HOME/hub is a mount; lm-eval's datasets fill the pod's
            # disk anywhere else.
            env["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
        if role.backend == "ray":
            # The pod is deleted when the run ends, so a session log
            # directory on its filesystem is unreachable by then.
            env["RAY_LOG_TO_STDERR"] = "1"
        if role.scratch_bytes:
            # dshm is sized from the host's memory; the container disk is
            # smaller than a large model's temporary files.
            env["TMPDIR"] = "/dev/shm"
            env["RAY_TMPDIR"] = "/tmp"
        env["FORCE_COLOR"] = "1"
        env["TQDM_MININTERVAL"] = "30"
        return env

    def volumes(self, role: RoleShape) -> Sequence[Volume]:
        dirs = _cache_dirs(role)
        if dirs is None:
            return ()
        # Shadows one subtree of the shared cache: a profile_cache that
        # persists makes a later run load the kv=0 graph instead of compiling
        # it, and the free-HBM probe measures nothing.
        return (
            Volume("vllm-profile-cache", f"{dirs[0]}/profile_cache", "Memory", "16Gi"),
        )

    def cpu_pod(self) -> CpuPod | None:
        return CPU_POD

    def job_policy(self, role: RoleShape) -> JobPolicy:
        # Unset fields keep the fleet's retries. A chip role of a multi-role
        # plan is never retried alone: it would rejoin a pair whose other half
        # kept serving.
        if role.multi_role and not role.chipless:
            return JobPolicy(backoff_limit=0)
        return JobPolicy()

    def lm_eval_cache_root(self) -> str | None:
        return LM_EVAL_CACHE

    def shape_problems(self, shape: Shape) -> Sequence[str]:
        shapes = chip_shapes(shape)
        if len(shapes) <= 1:
            return ()
        named = "; ".join(sorted(f"{a} {t} x{c}" for a, t, c in shapes))
        return (
            f"names more than one shape {named}. A workload is admitted against "
            "one queue, and a queue is one shape, so every pod holding chips has "
            "to ask for the same hardware",
        )

    def steps_group(self, shape: Shape) -> StepGroup | None:
        if ("tpu7x", "2x2x4") in {(a, t) for a, t, _ in chip_shapes(shape)}:
            return SLICE_2X2X4
        return None

    def after_tasks(self, shape: Shape) -> Sequence[AfterTask]:
        env = os.environ
        tasks = [
            AfterTask(
                "artifacts-result",
                ("buildkite-agent", "artifact", "upload", "{file}"),
                when="result",
                timeout_s=300,
            ),
            AfterTask(
                "artifacts-end",
                ("bash", "-c", _UPLOAD_REST, "_", "{unpublished}"),
                timeout_s=1800,
            ),
        ]
        if not uploads_rows(env):
            return tasks
        rows = [("CI_RUNNER", env.get("CI_RUNNER") or "kube")]
        if env.get("CREATED_BY"):
            rows.append(("CREATED_BY", env["CREATED_BY"]))
        tasks.append(
            AfterTask(
                "bigquery",
                (
                    "python3",
                    "scripts/vllm/benchmarking/upload_results.py",
                    "--results-dir",
                    "{run_dir}",
                    "--skip-spanner",
                ),
                env=(*rows, ("TPU_NAME", tpu_name(shape))),
                require="commands_ok",
                # A failed insert is a warning today (run_eval_flow.sh).
                fatal=False,
                skip_on_experiment=True,
                timeout_s=600,
            )
        )
        tasks.append(
            AfterTask(
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
                    *rows,
                ),
                when="result",
                results="cells",
                require="result_ok",
                # A gap in the P/D table turns the step red, as today.
                fatal=True,
                skip_on_experiment=True,
                timeout_s=300,
            )
        )
        return tasks

    # Submission ---------------------------------------------------------

    def apply(self, manifest: str, plan: Plan) -> int:
        env = os.environ
        problems = launcher_problems(manifest)
        problems += leak_problems(env)
        problems += runtime_problems(env, plan)
        if env.get("WORKLOAD_IMAGE"):
            problems.append(
                "the step sets WORKLOAD_IMAGE; put the image in the scenario "
                "(server.image) or pass --set server.image=..."
            )
        image = None
        if not problems:
            image, problem = workload_image(plan, env)
            problems += [problem] if problem else []
        if problems or image is None:
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            return 2
        path = manifest_path(plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest)
        _upload(path)
        sys.stdout.flush()
        sys.stderr.flush()
        return self._exec(
            [self.launcher, "--manifest", str(path)], {**env, "WORKLOAD_IMAGE": image}
        )

    def _exec(self, argv: list[str], env: dict[str, str]) -> int:
        # exec, so the step's SIGTERM on cancellation reaches the launcher's
        # cleanup and its exit code is the step's.
        os.execve(argv[0], argv, env)


def workload_image(plan: Plan, env: Mapping[str, str]) -> tuple[str | None, str | None]:
    """The image the launcher pins, or a problem: unset is the build's CI image."""
    if plan.image is not None:
        return plan.image, None
    key = f"CI_IMAGE_TAG{env.get('DOCKER_IMAGE_SUFFIX', '')}"
    try:
        done = subprocess.run(
            ["buildkite-agent", "meta-data", "get", key, "--default", ""],
            capture_output=True,
            text=True,
            timeout=120,
        )
        tag = done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        tag = ""
    if not tag:
        return None, (
            f"no {key} meta-data: the image build step has to run before this "
            "one, or the scenario has to name an image"
        )
    return tag, None


def manifest_path(plan: Plan) -> pathlib.Path:
    """Where the manifest goes in the checkout; the launcher records the path."""
    ref = plan.scenario_ref
    if ref.startswith("@"):
        ref = "adhoc/" + pathlib.Path(ref[1:]).stem
    point = f".{plan.point}" if plan.point else ""
    return pathlib.Path(MANIFEST_DIR, f"{ref}{point}.yaml")


def _upload(path: pathlib.Path) -> None:
    """The manifest as an agent-side artifact; best effort, it holds no secret."""
    try:
        done = subprocess.run(
            ["buildkite-agent", "artifact", "upload", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        failed = done.returncode != 0
    except (OSError, subprocess.TimeoutExpired):
        failed = True
    if failed:
        print(f"warning: could not upload {path} as an artifact", file=sys.stderr)


def leak_problems(env: Mapping[str, str]) -> list[str]:
    return [
        f"the step sets {name}, which a scenario step ignores; use {field}"
        for name, field in STEP_ENV.items()
        if env.get(name)
    ]


def runtime_problems(env: Mapping[str, str], plan: Plan) -> list[str]:
    """TPU_MAX_RUNTIME_SECONDS, which the launcher reads, must outlast startup."""
    raw = env.get("TPU_MAX_RUNTIME_SECONDS", "").strip()
    if not raw:
        return []
    startup = max(role.readiness.timeout_s - READINESS_MARGIN_S for role in plan.roles)
    if not raw.isdigit() or int(raw) <= startup:
        return [
            f"TPU_MAX_RUNTIME_SECONDS={raw!r} must be a whole number of seconds "
            f"over the scenario's startup_timeout ({startup}s)"
        ]
    return []


def launcher_problems(text: str) -> list[str]:
    """What the launcher would refuse in this manifest, found before submitting.

    The launcher stays the authority; these move its errors earlier.
    """
    problems = []
    for match in string.Template.pattern.finditer(text):
        if match.group("escaped") is not None:
            continue
        name = match.group("named") or match.group("braced")
        line = text.count("\n", 0, match.start()) + 1
        if name is None:
            problems.append(f"line {line}: a lone $; a literal one is $$")
        elif name not in SUBSTITUTED:
            problems.append(f"line {line}: ${{{name}}} would come from the step's env")
    if problems:
        return problems
    filled = string.Template(text).substitute(dict.fromkeys(SUBSTITUTED, "x"))
    doc = YAML(typ="safe", pure=True).load(filled)
    if not isinstance(doc, dict) or doc.get("kind") != "JobSet":
        return ["not a JobSet"]
    meta = doc.get("metadata") or {}
    if (meta.get("annotations") or {}).get(DEFAULTS_ANNOTATION) != "standard":
        problems.append(f"no {DEFAULTS_ANNOTATION}: standard annotation")
    for label in (QUEUE_LABEL, PRIORITY_LABEL):
        if label in (meta.get("labels") or {}):
            problems.append(f"sets {label}, which the launcher sets")
    shapes = set()
    for rjob in doc["spec"].get("replicatedJobs", []):
        name = rjob.get("name")
        pod = rjob["template"]["spec"]["template"]["spec"]
        containers = pod.get("containers", [])
        if [c.get("name") for c in containers] != [CONTAINER]:
            problems.append(f"{name}: needs exactly one container, named {CONTAINER}")
        account = pod.get("serviceAccountName")
        if account is not None and account not in SERVICE_ACCOUNTS:
            problems.append(f"{name}: service account {account} is not allowed")
        chips = {
            str(field[TPU_RESOURCE])
            for c in containers
            for field in (
                (c.get("resources") or {}).get(k) or {} for k in ("limits", "requests")
            )
            if TPU_RESOURCE in field
        }
        selector = pod.get("nodeSelector") or {}
        if chips or ACCELERATOR_LABEL in selector or TOPOLOGY_LABEL in selector:
            shapes.add(
                (
                    selector.get(ACCELERATOR_LABEL),
                    selector.get(TOPOLOGY_LABEL),
                    chips.pop() if len(chips) == 1 else None,
                )
            )
            continue
        for volume in pod.get("volumes", []):
            empty = volume.get("emptyDir")
            if (
                empty is not None
                and empty.get("medium") == "Memory"
                and "sizeLimit" not in empty
            ):
                problems.append(
                    f"{name}: memory volume {volume.get('name')} has no sizeLimit on a "
                    "pod without chips"
                )
    if len(shapes) > 1:
        problems.append(
            "names more than one shape: " + "; ".join(sorted(map(str, shapes)))
        )
    elif shapes and not all(next(iter(shapes))):
        problems.append(
            "a pod holding chips needs the accelerator and topology selectors and "
            "a google.com/tpu count"
        )
    return problems


def make_target() -> TpuCi:
    return TpuCi()
