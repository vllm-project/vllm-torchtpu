#!/usr/bin/env python3
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
"""Bootstrap the Buildkite CI and dev pipelines, and expose the pinned vLLM ref.

CLI entry points (compatible with Python 3.10+ stdlib on `queue=cpu`):
- `bootstrap.py`: main CI pipeline bootstrap (invoked by `bootstrap.sh`).
- `bootstrap.py upload-phase <suite> <k>`: upload phase `<k>` of `<suite>`.
- `bootstrap.py dev`: experimental dev pipeline bootstrap (`bootstrap_dev.sh`).
- `bootstrap.py vllm-pin`: print the pinned upstream vLLM ref from `pyproject.toml`.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cap_pr_tpu_concurrency

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_CREDENTIAL_HELPER = pathlib.Path("/etc/buildkite-agent/git-credential-github-app")
_GITHUB_API_URL = "https://api.github.com"

# Max TPU jobs one author's PR builds may have in flight on kube (~2 PRs at ~19 jobs).
PR_TPU_JOBS_PER_AUTHOR = 40

# Each lane is .buildkite/pipeline_<lane>.yml.
# Lane -> lanes it depends on. On gated builds a lane is uploaded only after every
# lane it depends on has passed; other builds upload all lanes at once.
# A lane that others depend on wraps its steps in one top-level group with key: <lane>.
DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "tests": ("cpu",),
    "integration": ("cpu",),
    "perf": ("cpu",),
    "verify": ("cpu",),
}

# Lane sets a build can run, each in run order.
SUITES: dict[str, tuple[str, ...]] = {
    "default": ("build", "cpu", "tests", "integration", "verify"),
    "full": ("build", "cpu", "tests", "integration", "perf", "verify"),
    "perf": ("build", "perf"),
    "cpu-only": ("build", "cpu"),
    "integration": (
        "build",
        "cpu",
        "tests",
        "integration",
        "perf",
        "verify",
        "promote",
    ),
}


def choose_suite(ctx: BuildContext, merged_labels: Sequence[str] = ()) -> str:
    """Return the `SUITES` key for `ctx` (`CI_SUITE` wins; first match wins)."""
    pr_labels = ctx.pr_labels if ctx.is_pr else ()
    skip_labels = pr_labels or (merged_labels if ctx.is_post_merge_webhook else ())
    if ctx.suite_override:
        return ctx.suite_override  # CI_SUITE=<name>
    if ctx.is_integration:
        return "integration"  # vllm-torchtpu-integration pipeline
    if "skip-tpu-ci" in skip_labels:
        return "cpu-only"
    if ctx.nightly_env or "run-full-ci" in pr_labels or "run-perf-eval" in pr_labels:
        return "full"
    return "default"  # PRs, main, torchtpu-nightly


# Build kind -> (Buildkite job priority, Kueue workload priority class).
# Nightly builds are unclassed on kube, level with the tpu-inference nightlies.
PRIORITIES: dict[str, tuple[int, str | None]] = {
    "oncall-fix": (12, "oncall-fix"),
    "post-merge": (11, "post-merge"),
    "pre-merge": (6, "pre-merge"),
    "other": (2, None),
    "nightly": (1, None),
}


def build_kind(ctx: BuildContext) -> str:
    """Classify the build into one of the `PRIORITIES` keys."""
    if ctx.nightly_env or ctx.torchtpu_nightly_env:
        return "nightly"
    if ctx.is_pr and "oncall-fix" in ctx.pr_labels:
        return "oncall-fix"
    if ctx.is_pr:
        return "pre-merge"
    if ctx.branch == "main":
        return "post-merge"
    return "other"


ANNOTATIONS: dict[str, str] = {
    "oncall-fix": (
        "Runs at on-call priority (`oncall-fix` label): "
        "ahead of every other build, including main."
    ),
    "full-ci": "Runs the full (nightly) suite (`run-full-ci` label).",
    "perf-eval": "Runs the perf/eval lane (`run-perf-eval` label).",
    "skip-tpu-ci": "TPU steps not run (`skip-tpu-ci` label); the CPU steps still run.",
}

# Commit-message, changed-file, and vLLM-pin patterns (.pyrefly-baseline-full.json stays
# out of _DOCS_ONLY_RE because the Buildkite typecheck step reads it).
_SKIP_CI_RE = re.compile(r"\[(skip\sci|ci\sskip)\]")
_DOCS_ONLY_RE = re.compile(
    r"(\.md$|\.ico$|\.png$|^README$|^docs/|^\.github/CODEOWNERS$"
    r"|^LICENSE$|^\.git-blame-ignore-revs$|^\.pyrefly-baseline\.json$)"
)
_MERGED_PR_RE = re.compile(r"\(#([0-9]+)\)$")
# Keep in sync with .buildkite/pipeline_verify.yml:22 and
# .buildkite/scripts/nightly_report.sh:74.
_RELEASE_TAG_RE = re.compile(r"^v?[0-9]+(\.[a-zA-Z0-9]+)*([a-zA-Z]+[0-9]*)?$")
_VLLM_PIN_RE = re.compile(
    r".*vllm @ git\+https://github\.com/vllm-project/vllm\.git@([a-zA-Z0-9.-]+).*"
)
_MISSING_VLLM_PIN_ERROR = (
    "ERROR: vLLM commit hash is missing from pyproject.toml. "
    "Cannot proceed without a pinned hash."
)


def is_gated(ctx: BuildContext) -> bool:
    """True when lanes in `DEPENDS_ON` wait until their dependencies pass."""
    return (
        ctx.source != "schedule"
        and not ctx.is_integration
        and "oncall-fix" not in ctx.pr_labels
        and not bool(_RELEASE_TAG_RE.match(ctx.tag))
    )


def phases(
    lanes: Sequence[str],
    *,
    gated: bool = True,
    depends_on: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Layer `lanes` topologically by `DEPENDS_ON` (single phase when ungated)."""
    if not lanes:
        return ()
    if not gated:
        return (tuple(lanes),)
    deps_map = DEPENDS_ON if depends_on is None else depends_on
    in_suite = set(lanes)
    deps = {
        lane: tuple(d for d in deps_map.get(lane, ()) if d in in_suite)
        for lane in lanes
    }
    placed: set[str] = set()
    remaining = list(lanes)
    result: list[tuple[str, ...]] = []
    while remaining:
        layer = tuple(
            lane for lane in remaining if all(d in placed for d in deps[lane])
        )
        if not layer:
            raise BootstrapError(
                f"ERROR: dependency cycle in lanes: {' '.join(remaining)}"
            )
        result.append(layer)
        placed.update(layer)
        remaining = [lane for lane in remaining if lane not in placed]
    return tuple(result)


class BootstrapError(Exception):
    """Fatal configuration or git/pin lookup error that exits bootstrap with 1."""


def _env_or(env: Mapping[str, str], key: str, default: str) -> str:
    """Match bash `${VAR:-default}`: treat both unset and empty string as default."""
    val = env.get(key, "")
    return val if val else default


def _split_labels(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.split(",") if item)


# ---------------------------------------------------------------------------
# Context & pure helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildContext:
    """Immutable snapshot of the build environment read at startup."""

    branch: str
    commit: str
    message: str
    source: str
    tag: str
    pipeline_slug: str
    pull_request: str | None
    pr_base_branch: str
    pr_labels: tuple[str, ...]
    author_email: str
    nightly_env: bool
    torchtpu_nightly_env: bool
    suite_override: str
    vllm_commit_override: str
    pr_tpu_concurrency_group_env: str

    @property
    def is_pr(self) -> bool:
        return self.pull_request is not None

    @property
    def is_integration(self) -> bool:
        return self.pipeline_slug == "vllm-torchtpu-integration"

    @property
    def is_post_merge_webhook(self) -> bool:
        return not self.is_pr and self.branch == "main" and self.source == "webhook"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BuildContext:
        suite_override = env.get("CI_SUITE", "").strip()
        if env.get("CI_LANES", "").strip() and not suite_override:
            raise BootstrapError(
                "ERROR: CI_LANES was removed; set CI_SUITE=<name> "
                f"(suites: {' '.join(SUITES)})"
            )
        if env.get("SKIP_TPU_CI", "").strip():
            raise BootstrapError(
                "ERROR: SKIP_TPU_CI was removed; use CI_SUITE=cpu-only"
            )
        pr_raw = _env_or(env, "BUILDKITE_PULL_REQUEST", "false")
        author = _env_or(
            env,
            "BUILDKITE_BUILD_AUTHOR_EMAIL",
            _env_or(env, "BUILDKITE_BUILD_CREATOR_EMAIL", ""),
        )
        return cls(
            branch=env.get("BUILDKITE_BRANCH", ""),
            commit=_env_or(env, "BUILDKITE_COMMIT", "HEAD"),
            message=env.get("BUILDKITE_MESSAGE", ""),
            source=env.get("BUILDKITE_SOURCE", ""),
            tag=env.get("BUILDKITE_TAG", ""),
            pipeline_slug=env.get("BUILDKITE_PIPELINE_SLUG", ""),
            pull_request=None if pr_raw == "false" else pr_raw,
            pr_base_branch=_env_or(env, "BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main"),
            pr_labels=_split_labels(env.get("BUILDKITE_PULL_REQUEST_LABELS", "")),
            author_email=author,
            nightly_env=_env_or(env, "NIGHTLY", "0") == "1",
            torchtpu_nightly_env=_env_or(env, "TORCHTPU_NIGHTLY", "0") == "1",
            suite_override=suite_override,
            vllm_commit_override=env.get("VLLM_COMMIT_HASH_OVERRIDE", ""),
            pr_tpu_concurrency_group_env=env.get("PR_TPU_CONCURRENCY_GROUP", ""),
        )


def has_skip_ci_marker(message: str) -> bool:
    """Return True if the commit message contains `[skip ci]` or `[ci skip]`."""
    return bool(_SKIP_CI_RE.search(message))


def only_docs_changed(files: Sequence[str]) -> bool:
    """Return True when `files` is non-empty and every path is non-code/docs."""
    return bool(files) and all(bool(_DOCS_ONLY_RE.search(path)) for path in files)


def parse_merged_pr_number(message: str) -> str | None:
    """Extract trailing `(#123)` PR number from the first line of a squash commit."""
    first_line = message.splitlines()[0] if message else ""
    match = _MERGED_PR_RE.search(first_line)
    return match.group(1) if match else None


def parse_pinned_vllm_commit(pyproject_text: str) -> str:
    """Extract the pinned upstream vLLM ref from `pyproject.toml`."""
    for line in pyproject_text.splitlines():
        match = _VLLM_PIN_RE.match(line)
        if match:
            return match.group(1)
    raise BootstrapError(_MISSING_VLLM_PIN_ERROR)


def read_pinned_vllm_commit(repo_root: pathlib.Path = _REPO_ROOT) -> str:
    """Read `pyproject.toml` under `repo_root` and return the pinned vLLM ref."""
    pyproject_path = repo_root / "pyproject.toml"
    try:
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapError(f"ERROR: could not read {pyproject_path}: {exc}") from exc
    return parse_pinned_vllm_commit(pyproject_text)


def prepare_upload_yaml(raw_yaml: str, *, job_priority: int, nightly: bool) -> str:
    """Prepend `priority:` and substitute `build.env("NIGHTLY")` when needed."""
    # build.env("NIGHTLY") is evaluated by Buildkite against the build's initial
    # environment, so a NIGHTLY=1 set by bootstrap (for the run-full-ci label)
    # never reaches it unless substituted before upload.
    body = raw_yaml.replace('build.env("NIGHTLY")', '"1"') if nightly else raw_yaml
    return f"priority: {job_priority}\n{body}"


# ---------------------------------------------------------------------------
# Subprocess & GitHub helpers (monkeypatched in tests)
# ---------------------------------------------------------------------------


def _agent(
    *args: str,
    input_text: str | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path = _REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run `buildkite-agent` with the given arguments."""
    return subprocess.run(
        ["buildkite-agent", *args],
        cwd=cwd,
        env=env,
        input=input_text,
        stdout=subprocess.PIPE if "--dry-run" in args else None,
        text=True,
        check=check,
    )


def _annotate(reason: str, *, env: Mapping[str, str]) -> None:
    """Post one of the standard `ANNOTATIONS` to the Buildkite build."""
    style = "warning" if reason == "oncall-fix" else "info"
    _agent(
        "annotate",
        "--style",
        style,
        "--context",
        reason,
        ANNOTATIONS[reason],
        env=env,
    )


def _git(
    *args: str,
    check: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path = _REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a `git` command and capture stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if args[:1] == ("diff",) else None,
        text=True,
        check=check,
    )


def _changed_files(ctx: BuildContext, *, env: Mapping[str, str]) -> list[str]:
    """Return files changed by a PR or non-scheduled commit build."""
    if ctx.is_pr:
        print(f"PR detected. Target branch: {ctx.pr_base_branch}")
        if (
            _git(
                "fetch",
                "origin",
                ctx.pr_base_branch,
                "--depth=20",
                "--quiet",
                env=env,
            ).returncode
            != 0
        ):
            print("Base fetch failed")
        _git("fetch", "origin", ctx.commit, "--depth=20", "--quiet", env=env)
        diff = _git(
            "diff",
            "--name-only",
            f"origin/{ctx.pr_base_branch}...{ctx.commit}",
            env=env,
        )
        files = [line for line in diff.stdout.splitlines() if line]
        if files:
            return files
        tree = _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-m",
            ctx.commit,
            check=True,
            env=env,
        )
        return [line for line in tree.stdout.splitlines() if line]

    if ctx.source != "schedule":
        # Any other build is of a commit, which on main is the squash merge of
        # one PR, so its diff against its parent is the whole change. Not a
        # scheduled build: the nightly tests main as it stands, whatever its
        # last commit touched.
        _git("fetch", "origin", ctx.commit, "--depth=2", "--quiet", env=env)
        diff = _git("diff", "--name-only", f"{ctx.commit}^", ctx.commit, env=env)
        return [line for line in diff.stdout.splitlines() if line]

    return []


def _github_app_token(*, env: Mapping[str, str] | None = None) -> str | None:
    if not _CREDENTIAL_HELPER.is_file():
        return None
    try:
        proc = subprocess.run(
            [str(_CREDENTIAL_HELPER), "get"],
            cwd=_REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            token = line.removeprefix("password=").strip()
            if token:
                return token
    return None


def _github_labels(
    pr_number: str, *, env: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Read labels of a merged PR from GitHub; return `()` on any failure."""
    headers = {"Accept": "application/vnd.github+json"}
    token = _github_app_token(env=env)
    if token:
        headers["Authorization"] = f"token {token}"
    url = (
        f"{_GITHUB_API_URL.rstrip('/')}/repos/"
        f"vllm-project/vllm-torchtpu/issues/{pr_number}/labels"
    )
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except (
        OSError,
        urllib.error.URLError,
        http.client.HTTPException,
        ValueError,
    ):
        return ()
    if not isinstance(items, list):
        return ()
    return tuple(
        item["name"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )


def upload_with_priority(
    yaml_file: str,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    job_priority: int,
    nightly: bool,
    pr_tpu_concurrency_group: str | None,
    pr_tpu_jobs_per_author: int = PR_TPU_JOBS_PER_AUTHOR,
    env: Mapping[str, str],
) -> None:
    """Upload one pipeline YAML file with priority and optional PR TPU capping."""
    print(f"--- :pipeline: Uploading {yaml_file} with priority {job_priority}")
    raw_yaml = (repo_root / yaml_file).read_text(encoding="utf-8")
    payload = prepare_upload_yaml(raw_yaml, job_priority=job_priority, nightly=nightly)

    # A pull-request build's TPU steps go in its author's concurrency group. The
    # file is read once by the agent via --dry-run (anchors and interpolation
    # resolved), the group is added, and the result is uploaded as is. If any of
    # that fails the file goes up uncapped rather than failing the build.
    if pr_tpu_concurrency_group:
        dry_run = _agent(
            "pipeline",
            "upload",
            "--dry-run",
            "--format",
            "json",
            input_text=payload,
            check=False,
            env=env,
        )
        if dry_run.returncode == 0 and dry_run.stdout is not None:
            try:
                pipeline = json.loads(dry_run.stdout)
                if not isinstance(pipeline, dict):
                    raise TypeError(
                        "expected JSON object from --dry-run, "
                        f"got {type(pipeline).__name__}"
                    )
                cap_pr_tpu_concurrency.cap(
                    pipeline.get("steps") or [],
                    pr_tpu_concurrency_group,
                    pr_tpu_jobs_per_author,
                )
                capped = json.dumps(pipeline) + "\n"
            except Exception as exc:
                print(
                    f"WARNING: failed to cap TPU concurrency for {yaml_file}: {exc}",
                    file=sys.stderr,
                )
            else:
                _agent(
                    "pipeline",
                    "upload",
                    "--no-interpolation",
                    input_text=capped,
                    env=env,
                )
                return
        print(
            f"WARNING: could not put {yaml_file} in {pr_tpu_concurrency_group}; "
            "uploading it uncapped",
            file=sys.stderr,
        )

    _agent("pipeline", "upload", input_text=payload, env=env)


def upload_phase(
    suite: str,
    phase_idx: int,
    env: Mapping[str, str] | None = None,
    *,
    gated: bool = True,
    repo_root: pathlib.Path = _REPO_ROOT,
) -> int:
    """Upload `phase_idx` of `SUITES[suite]` (and the next phase's uploader step)."""
    if suite not in SUITES:
        raise BootstrapError(
            f"ERROR: unknown CI_SUITE '{suite}'; suites are: {' '.join(SUITES)}"
        )
    suite_lanes = SUITES[suite]
    all_phases = phases(suite_lanes, gated=gated)
    if phase_idx < 0 or phase_idx >= len(all_phases):
        raise BootstrapError(
            f"ERROR: phase {phase_idx} out of range for suite '{suite}' "
            f"(0..{len(all_phases) - 1})"
        )

    child_env = dict(os.environ if env is None else env)
    job_priority = int(_env_or(child_env, "JOB_PRIORITY", str(PRIORITIES["other"][0])))
    nightly = _env_or(child_env, "NIGHTLY", "0") == "1"
    concurrency_group = child_env.get("PR_TPU_CONCURRENCY_GROUP") or None

    next_idx = phase_idx + 1
    if next_idx < len(all_phases):
        next_lanes = all_phases[next_idx]
        in_suite = set(suite_lanes)
        deps = tuple(
            dict.fromkeys(
                dep
                for lane in next_lanes
                for dep in DEPENDS_ON.get(lane, ())
                if dep in in_suite
            )
        )
        step_env = {
            k: child_env[k]
            for k in (
                "JOB_PRIORITY",
                "NIGHTLY",
                "OMNI_IMAGE_SKIP",
                "PR_TPU_CONCURRENCY_GROUP",
            )
            if child_env.get(k)
        }
        uploader = {
            "priority": job_priority,
            "steps": [
                {
                    "label": (
                        f":pipeline: Upload {', '.join(next_lanes)} "
                        f"(after {', '.join(deps)} passed)"
                    ),
                    "key": f"upload_phase_{next_idx}",
                    "depends_on": list(deps),
                    "agents": {"queue": "cpu"},
                    "env": step_env,
                    "command": (
                        "python3 .buildkite/scripts/bootstrap.py "
                        f"upload-phase {suite} {next_idx}"
                    ),
                }
            ],
        }
        _agent(
            "pipeline",
            "upload",
            input_text=json.dumps(uploader) + "\n",
            env=child_env,
        )

    for lane in reversed(all_phases[phase_idx]):
        lane_file = f".buildkite/pipeline_{lane}.yml"
        print(f"Uploading {lane_file}")
        upload_with_priority(
            lane_file,
            repo_root=repo_root,
            job_priority=job_priority,
            nightly=nightly,
            pr_tpu_concurrency_group=concurrency_group,
            env=child_env,
        )
    return 0


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_bootstrap(
    env: Mapping[str, str] | None = None,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
) -> int:
    """Execute the full Buildkite bootstrap flow and return an exit status."""
    child_env = dict(os.environ if env is None else env)
    ctx = BuildContext.from_env(child_env)

    # 1. Kind and priority (metadata).
    print("--- Determining job priority", file=sys.stderr)
    kind = build_kind(ctx)
    job_priority, workload_priority = PRIORITIES[kind]
    print(f"Build type: {kind} - Priority: {job_priority}", file=sys.stderr)
    child_env["JOB_PRIORITY"] = str(job_priority)
    _agent("meta-data", "set", "JOB_PRIORITY", str(job_priority), env=child_env)
    # The kube launcher submits every workload of the build at this Kueue
    # priority class (ci-infra launch.py). None leaves them unclassed, at 0.
    if workload_priority is not None:
        _agent(
            "meta-data", "set", "WORKLOAD_PRIORITY", workload_priority, env=child_env
        )
    if kind == "oncall-fix":
        _annotate("oncall-fix", env=child_env)

    # 2. Skip checks ([skip ci], docs-only).
    print("--- :git: Checking if CI build should be skipped")
    if has_skip_ci_marker(ctx.message):
        print("Commit message contains [skip ci]. Skipping build.")
        return 0
    changed_files = _changed_files(ctx, env=child_env)
    if changed_files:
        print("Files changed:")
        print("\n".join(changed_files))
        if only_docs_changed(changed_files):
            print("Only documentation/non-code files changed. Skipping CI build.")
            return 0
        print("Code files changed. Proceeding with pipeline upload.")

    # 3. vLLM commit.
    print("--- Starting Buildkite Bootstrap")
    if ctx.is_integration:
        tokens = _git(
            "ls-remote",
            "https://github.com/vllm-project/vllm.git",
            "HEAD",
            check=True,
            env=child_env,
        ).stdout.split()
        base_vllm_hash = tokens[0] if tokens else ""
    else:
        base_vllm_hash = read_pinned_vllm_commit(repo_root)
    vllm_commit_hash = ctx.vllm_commit_override or base_vllm_hash
    _agent("meta-data", "set", "VLLM_COMMIT_HASH", vllm_commit_hash, env=child_env)
    print(f"Using vLLM commit: {vllm_commit_hash}")

    # 4. Suite selection and annotations.
    merged_labels: tuple[str, ...] = ()
    if ctx.is_post_merge_webhook:
        pr_num = parse_merged_pr_number(ctx.message)
        if pr_num is not None:
            merged_labels = _github_labels(pr_num, env=child_env)
    suite = choose_suite(ctx, merged_labels)
    if suite not in SUITES:
        raise BootstrapError(
            f"ERROR: unknown CI_SUITE '{suite}'; suites are: {' '.join(SUITES)}"
        )
    lanes = SUITES[suite]
    pr_labels = ctx.pr_labels if ctx.is_pr else ()
    if not ctx.suite_override:
        if suite == "cpu-only":
            _annotate("skip-tpu-ci", env=child_env)
        elif suite == "full" and "run-full-ci" in pr_labels:
            _annotate("full-ci", env=child_env)
        elif suite == "full" and not ctx.nightly_env and "run-perf-eval" in pr_labels:
            _annotate("perf-eval", env=child_env)

    # 5. Env exports (NIGHTLY, OMNI_IMAGE_SKIP, PR_TPU_CONCURRENCY_GROUP).
    nightly = ctx.nightly_env or "run-full-ci" in pr_labels
    if nightly:
        child_env["NIGHTLY"] = "1"
    # pipeline_build.yml skips the -omni image build when this is non-empty.
    # Buildkite rejects the upload if a skip reason is over 70 characters.
    child_env["OMNI_IMAGE_SKIP"] = (
        ""
        if "tests" in lanes
        else "Only the tests lane uses the -omni image; this build has none."
    )
    concurrency_group = (
        f"tpu-pr/{ctx.author_email.lower()}"
        if (ctx.is_pr and ctx.author_email)
        else (ctx.pr_tpu_concurrency_group_env or None)
    )
    if concurrency_group is not None:
        child_env["PR_TPU_CONCURRENCY_GROUP"] = concurrency_group

    # 6. Upload phase 0 (gated builds chain later phases via upload_phase_<k>).
    upload_phase(suite, 0, child_env, gated=is_gated(ctx), repo_root=repo_root)
    print("--- Buildkite Bootstrap Finished")
    return 0


def run_bootstrap_dev(
    env: Mapping[str, str] | None = None,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
) -> int:
    """Execute the `vllm-torchtpu-dev` bootstrap flow (`bootstrap_dev.sh`)."""
    child_env = dict(os.environ if env is None else env)
    vllm_commit_hash = read_pinned_vllm_commit(repo_root)
    _agent("meta-data", "set", "VLLM_COMMIT_HASH", vllm_commit_hash, env=child_env)
    print(f"Using vLLM commit: {vllm_commit_hash}")

    # Dev builds are experiments — they must never outrank real CI on a shared
    # queue. Default to PRIORITIES["other"][0] (2); override per build with
    # DEV_JOB_PRIORITY if an experiment needs to jump ahead of nightlies only.
    raw_priority = _env_or(child_env, "DEV_JOB_PRIORITY", str(PRIORITIES["other"][0]))
    child_env["JOB_PRIORITY"] = raw_priority
    _agent("meta-data", "set", "JOB_PRIORITY", raw_priority, env=child_env)
    print(f"--- Dev build priority: {raw_priority}")

    pipeline_file = _env_or(
        child_env, "DEV_PIPELINE_FILE", ".buildkite/pipeline_dev.yml"
    )
    if not (repo_root / pipeline_file).is_file():
        branch = _env_or(child_env, "BUILDKITE_BRANCH", "unknown")
        commit = _env_or(child_env, "BUILDKITE_COMMIT", "unknown")
        print(
            f"ERROR: dev pipeline file '{pipeline_file}' not found on "
            f"branch '{branch}' at commit {commit}. "
            "Create it on your branch, or point DEV_PIPELINE_FILE at an existing "
            "file."
        )
        return 1

    try:
        job_priority = int(raw_priority)
    except ValueError as exc:
        raise BootstrapError(
            f"ERROR: invalid DEV_JOB_PRIORITY '{raw_priority}'; expected an integer."
        ) from exc

    print(f"--- :pipeline: Uploading {pipeline_file}")
    upload_with_priority(
        pipeline_file,
        repo_root=repo_root,
        job_priority=job_priority,
        nightly=_env_or(child_env, "NIGHTLY", "0") == "1",
        pr_tpu_concurrency_group=child_env.get("PR_TPU_CONCURRENCY_GROUP") or None,
        env=child_env,
    )
    print("--- Buildkite Dev Bootstrap Finished")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(line_buffering=True)

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if not args:
            rc = run_bootstrap()
        elif args == ["dev"]:
            rc = run_bootstrap_dev()
        elif args == ["vllm-pin"]:
            print(read_pinned_vllm_commit(_REPO_ROOT))
            rc = 0
        elif len(args) == 3 and args[0] == "upload-phase":
            try:
                phase_idx = int(args[2])
            except ValueError as exc:
                raise BootstrapError(
                    f"ERROR: invalid phase index '{args[2]}'; expected an integer."
                ) from exc
            rc = upload_phase(args[1], phase_idx)
        else:
            raise BootstrapError(
                "Usage: bootstrap.py [dev | vllm-pin | upload-phase <suite> <k>]"
            )
    except BootstrapError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
