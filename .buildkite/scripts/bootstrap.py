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
"""Bootstrap the Buildkite CI pipeline: priority, skip checks, lanes, and upload.

Structured in four layers (compatible with Python 3.10+ stdlib on `queue=cpu`):
- `BuildContext`: reads environment variables and PR labels once.
- Pure planning functions: priority, skip/docs-only checks, vLLM pin, and lanes.
- `upload_with_priority`: injects priority, substitutes `NIGHTLY`, and caps PR
  TPU concurrency via `cap_pr_tpu_concurrency` (also exposed as
  `bootstrap.py upload <file> [priority]` for `configs/pipeline_config.sh`).
- `BootstrapEffects`: thin boundary for `buildkite-agent`, `git`, and GitHub IO.
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
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass

import cap_pr_tpu_concurrency

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

_EXPORT_INT_RE = re.compile(r"^export\s+([A-Z0-9_]+)=(-?[0-9]+)\s*$")


def _load_pipeline_config(path: pathlib.Path) -> dict[str, int]:
    """Load integer constants from `configs/pipeline_config.sh` (single source)."""
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _EXPORT_INT_RE.match(line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


_PIPELINE_CONFIG = _load_pipeline_config(_SCRIPT_DIR / "configs" / "pipeline_config.sh")
PRIORITY_ONCALL_FIX: int = _PIPELINE_CONFIG["PRIORITY_ONCALL_FIX"]
PRIORITY_POST_MERGE: int = _PIPELINE_CONFIG["PRIORITY_POST_MERGE"]
PRIORITY_PRE_MERGE: int = _PIPELINE_CONFIG["PRIORITY_PRE_MERGE"]
PRIORITY_INTEGRATION: int = _PIPELINE_CONFIG["PRIORITY_INTEGRATION"]
PRIORITY_BENCHMARK: int = _PIPELINE_CONFIG["PRIORITY_BENCHMARK"]
PRIORITY_DEFAULT: int = _PIPELINE_CONFIG["PRIORITY_DEFAULT"]
PRIORITY_NIGHTLY: int = _PIPELINE_CONFIG["PRIORITY_NIGHTLY"]
PRIORITY_KERNEL_TUNING: int = _PIPELINE_CONFIG["PRIORITY_KERNEL_TUNING"]

PR_TPU_JOBS_PER_AUTHOR: int = _PIPELINE_CONFIG["PR_TPU_JOBS_PER_AUTHOR"]

# The suite's lanes, in upload order. Lane <lane> is
# .buildkite/pipeline_<lane>.yml; no other file is uploaded as a lane. CI_LANES
# picks a subset - a scheduled pipeline names its own lane. Without it a build
# runs every lane but perf, which has scheduled builds of its own; nightly
# builds still run perf, and so does the integration pipeline, whose LKG
# promotion waits on two perf steps. A PR whose change needs perf/eval coverage
# asks for it with the run-perf-eval label, or the full nightly suite across
# every lane with run-full-ci; adding either label to a ready PR starts a build.
ALL_LANES: tuple[str, ...] = ("verify", "perf", "integration", "tests")
DEFAULT_LANES: tuple[str, ...] = ("verify", "integration", "tests")
DEFAULT_INTEGRATION_SLUG = "vllm-torchtpu-integration"
VLLM_UPSTREAM_REPO = "https://github.com/vllm-project/vllm.git"
GITHUB_REPO = "vllm-project/vllm-torchtpu"
GITHUB_API_BASE_URL = "https://api.github.com"

_SKIP_CI_RE = re.compile(r"\[(skip\sci|ci\sskip)\]")
# True when every path is documentation or another file no step reads (docs, md,
# icons, CODEOWNERS, LICENSE, .git-blame-ignore-revs, .pyrefly-baseline.json).
# The pre-commit GitHub Action checks .pyrefly-baseline.json and no Buildkite
# step reads it. .pyrefly-baseline-full.json stays out of this list: only the
# Buildkite type check reads it.
_DOCS_ONLY_RE = re.compile(
    r"(\.md$|\.ico$|\.png$|^README$|^docs/|^\.github/CODEOWNERS$"
    r"|^LICENSE$|^\.git-blame-ignore-revs$|^\.pyrefly-baseline\.json$)"
)
_MERGED_PR_RE = re.compile(r"\(#([0-9]+)\)$")
_VLLM_PIN_RE = re.compile(
    r".*vllm @ git\+https://github\.com/vllm-project/vllm\.git@([a-zA-Z0-9.-]+).*"
)


class BootstrapError(Exception):
    """Fatal configuration or git/pin lookup error that exits bootstrap with 1."""


def _env_or(env: Mapping[str, str], key: str, default: str) -> str:
    """Match bash `${VAR:-default}`: treat both unset and empty string as default."""
    val = env.get(key, "")
    return val if val else default


def _split_labels(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.split(",") if item)


# ---------------------------------------------------------------------------
# Context
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
    integration_slug: str
    pull_request: str | None
    pr_base_branch: str
    pr_labels: tuple[str, ...]
    author_email: str
    nightly_env: bool
    torchtpu_nightly_env: bool
    ci_lanes_raw: str
    skip_tpu_ci_env: bool
    vllm_commit_override: str
    pr_tpu_concurrency_group_env: str
    pr_tpu_jobs_per_author: int

    @property
    def is_pr(self) -> bool:
        return self.pull_request is not None

    @property
    def is_integration(self) -> bool:
        return self.pipeline_slug == self.integration_slug

    @property
    def is_post_merge_webhook(self) -> bool:
        return not self.is_pr and self.branch == "main" and self.source == "webhook"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BuildContext:
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
            integration_slug=_env_or(
                env, "VLLM_INTEGRATION_PIPELINE_SLUG", DEFAULT_INTEGRATION_SLUG
            ),
            pull_request=None if pr_raw == "false" else pr_raw,
            pr_base_branch=_env_or(env, "BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main"),
            pr_labels=_split_labels(env.get("BUILDKITE_PULL_REQUEST_LABELS", "")),
            author_email=author,
            nightly_env=_env_or(env, "NIGHTLY", "0") == "1",
            torchtpu_nightly_env=_env_or(env, "TORCHTPU_NIGHTLY", "0") == "1",
            ci_lanes_raw=env.get("CI_LANES", ""),
            skip_tpu_ci_env=_env_or(env, "SKIP_TPU_CI", "0") == "1",
            vllm_commit_override=env.get("VLLM_COMMIT_HASH_OVERRIDE", ""),
            pr_tpu_concurrency_group_env=env.get("PR_TPU_CONCURRENCY_GROUP", ""),
            pr_tpu_jobs_per_author=PR_TPU_JOBS_PER_AUTHOR,
        )


# ---------------------------------------------------------------------------
# Pure planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Annotation:
    """A Buildkite build annotation."""

    style: str
    context: str
    body: str


@dataclass(frozen=True)
class PriorityPlan:
    """Priority, Kueue workload class, and per-author PR concurrency group."""

    job_priority: int
    workload_priority: str | None
    annotation: Annotation | None
    pr_tpu_concurrency_group: str | None
    log_line: str


@dataclass(frozen=True)
class UploadPlan:
    """Resolved lane selection, exported flags, and annotations for upload."""

    nightly: bool
    skip_tpu_ci: bool
    lanes: tuple[str, ...]
    lane_files: tuple[str, ...]
    build_file: str
    omni_image_skip: str
    pre_lane_annotations: tuple[Annotation, ...]
    post_lane_annotations: tuple[Annotation, ...]
    invalid_ci_lanes_error: str | None
    missing_ci_lanes_error: str | None


def plan_priority(ctx: BuildContext) -> PriorityPlan:
    """Compute Buildkite job priority, Kueue class, and PR concurrency group."""
    workload_priority: str | None = None
    annotation: Annotation | None = None

    if ctx.nightly_env or ctx.torchtpu_nightly_env:
        # Nightly build (Lowest priority). Unclassed on kube, level with the
        # tpu-inference nightlies it shares the TPU queues with, which record no
        # class either.
        priority = PRIORITY_NIGHTLY
        log_line = f"Build type: Nightly - Priority: {priority}"
    elif ctx.pull_request is not None:
        if "oncall-fix" in ctx.pr_labels:
            # PR that fixes a CI or nightly breakage (Highest priority).
            priority = PRIORITY_ONCALL_FIX
            workload_priority = "oncall-fix"
            annotation = Annotation(
                style="warning",
                context="oncall-fix",
                body=(
                    "Runs at on-call priority (`oncall-fix` label): "
                    "ahead of every other build, including main."
                ),
            )
            log_line = (
                f"Build type: On-call fix (PR #{ctx.pull_request}) "
                f"- Priority: {priority}"
            )
        else:
            priority = PRIORITY_PRE_MERGE
            workload_priority = "pre-merge"
            log_line = (
                f"Build type: Pre-merge (PR #{ctx.pull_request}) - Priority: {priority}"
            )
    elif ctx.branch == "main":
        priority = PRIORITY_POST_MERGE
        workload_priority = "post-merge"
        log_line = f"Build type: Post-merge (Main branch) - Priority: {priority}"
    else:
        priority = PRIORITY_DEFAULT
        log_line = f"Build type: General - Priority: {priority}"

    # One concurrency group per author for the TPU steps of their PR builds; see
    # PR_TPU_JOBS_PER_AUTHOR in pipeline_config.sh. Keyed on the commit author's
    # email, which every webhook build carries: the Buildkite creator is missing
    # whenever the GitHub account is not linked to a Buildkite user.
    if ctx.is_pr and ctx.author_email:
        concurrency_group: str | None = f"tpu-pr/{ctx.author_email.lower()}"
    elif ctx.pr_tpu_concurrency_group_env:
        concurrency_group = ctx.pr_tpu_concurrency_group_env
    else:
        concurrency_group = None

    return PriorityPlan(
        job_priority=priority,
        workload_priority=workload_priority,
        annotation=annotation,
        pr_tpu_concurrency_group=concurrency_group,
        log_line=log_line,
    )


def has_skip_ci_marker(message: str) -> bool:
    """Return True if the commit message contains `[skip ci]` or `[ci skip]`."""
    return bool(_SKIP_CI_RE.search(message))


def only_docs_changed(files: Sequence[str]) -> bool:
    """Return True when `files` is non-empty and every path is non-code/docs."""
    if not files:
        return False
    return all(bool(_DOCS_ONLY_RE.search(path)) for path in files)


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
    raise BootstrapError(
        "ERROR: vLLM commit hash is missing from pyproject.toml. "
        "Cannot proceed without a pinned hash."
    )


def resolve_vllm_commit_hash(base_hash: str, override: str) -> str:
    """Apply `VLLM_COMMIT_HASH_OVERRIDE` when non-empty."""
    return override if override else base_hash


def plan_uploads(
    ctx: BuildContext,
    *,
    merged_labels: Sequence[str],
    existing_lane_files: Container[str],
) -> UploadPlan:
    """Decide lanes, annotations, `SKIP_TPU_CI`, and `OMNI_IMAGE_SKIP`."""
    pre_annotations: list[Annotation] = []
    nightly = ctx.nightly_env
    if ctx.is_pr and "run-full-ci" in ctx.pr_labels:
        nightly = True
        pre_annotations.append(
            Annotation(
                style="info",
                context="full-ci",
                body="Runs the full (nightly) suite (`run-full-ci` label).",
            )
        )

    if nightly or ctx.is_integration:
        default_lanes = ALL_LANES
    elif ctx.is_pr and "run-perf-eval" in ctx.pr_labels:
        default_lanes = ALL_LANES
        if not ctx.ci_lanes_raw:
            pre_annotations.append(
                Annotation(
                    style="info",
                    context="perf-eval",
                    body="Runs the perf/eval lane (`run-perf-eval` label).",
                )
            )
    else:
        default_lanes = DEFAULT_LANES

    # skip-tpu-ci keeps only the tests lane, and SKIP_TPU_CI leaves out its TPU
    # group (pipeline_tests.yml reads it when the lane is uploaded), so the
    # image build, the type check and the CPU unit tests still run, on the PR
    # and again on its post-merge build. It overrides run-perf-eval and
    # run-full-ci.
    if ctx.is_pr:
        skip_labels: Sequence[str] = ctx.pr_labels
    elif ctx.is_post_merge_webhook:
        skip_labels = merged_labels
    else:
        skip_labels = ()

    skip_tpu_ci = ctx.skip_tpu_ci_env or ("skip-tpu-ci" in skip_labels)
    if skip_tpu_ci:
        default_lanes = ("tests",)
        pre_annotations.append(
            Annotation(
                style="info",
                context="skip-tpu-ci",
                body=(
                    "TPU steps not run (`skip-tpu-ci` label); the CPU steps still run."
                ),
            )
        )

    requested_lanes = (
        tuple(ctx.ci_lanes_raw.split()) if ctx.ci_lanes_raw else default_lanes
    )
    invalid_error: str | None = None
    for lane in requested_lanes:
        if lane not in ALL_LANES:
            invalid_error = (
                f"ERROR: CI_LANES names '{lane}'; lanes are: {' '.join(ALL_LANES)}"
            )
            break

    lane_files: list[str] = []
    missing: list[str] = []
    if invalid_error is None:
        for lane in ALL_LANES:
            if lane not in requested_lanes:
                continue
            rel_path = f".buildkite/pipeline_{lane}.yml"
            if rel_path in existing_lane_files:
                lane_files.append(rel_path)
            else:
                missing.append(rel_path)

    post_annotations: list[Annotation] = []
    missing_error: str | None = None
    if missing:
        # A lane asked for by name has to exist; the whole suite runs the lanes
        # it has and says what it left out.
        missing_str = " ".join(missing)
        if ctx.ci_lanes_raw:
            missing_error = (
                f"ERROR: CI_LANES asks for {missing_str}, which does not exist"
            )
        else:
            post_annotations.append(
                Annotation(
                    style="warning",
                    context="lane-gaps",
                    body=f"Not run: no {missing_str} yet.",
                )
            )

    # pipeline_build.yml skips the -omni image build when this is non-empty.
    # Buildkite rejects the upload if a skip reason is over 70 characters.
    if "tests" not in requested_lanes:
        omni_image_skip = (
            "Only the tests lane uses the -omni image; this build has none."
        )
    elif skip_tpu_ci:
        omni_image_skip = "Only TPU steps use the -omni image; skip-tpu-ci skips them."
    else:
        omni_image_skip = ""

    return UploadPlan(
        nightly=nightly,
        skip_tpu_ci=skip_tpu_ci,
        lanes=requested_lanes,
        lane_files=tuple(lane_files),
        build_file=".buildkite/pipeline_build.yml",
        omni_image_skip=omni_image_skip,
        pre_lane_annotations=tuple(pre_annotations),
        post_lane_annotations=tuple(post_annotations),
        invalid_ci_lanes_error=invalid_error,
        missing_ci_lanes_error=missing_error,
    )


# ---------------------------------------------------------------------------
# Pipeline upload
# ---------------------------------------------------------------------------


def prepare_upload_yaml(raw_yaml: str, *, job_priority: int, nightly: bool) -> str:
    """Prepend `priority:` and substitute `build.env("NIGHTLY")` when needed."""
    # build.env("NIGHTLY") is evaluated by Buildkite against the build's initial
    # environment, so a NIGHTLY=1 set by bootstrap (for the run-full-ci label)
    # never reaches it unless substituted before upload.
    body = raw_yaml.replace('build.env("NIGHTLY")', '"1"') if nightly else raw_yaml
    return f"priority: {job_priority}\n{body}"


def upload_with_priority(
    yaml_file: str,
    *,
    repo_root: pathlib.Path,
    job_priority: int,
    nightly: bool,
    pr_tpu_concurrency_group: str | None,
    pr_tpu_jobs_per_author: int,
    env: Mapping[str, str],
    effects: BootstrapEffects,
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
        dry_run_json = effects.dry_run_upload(payload, env=env)
        if dry_run_json is not None:
            try:
                pipeline = json.loads(dry_run_json)
                cap_pr_tpu_concurrency.cap(
                    pipeline.get("steps") or [],
                    pr_tpu_concurrency_group,
                    pr_tpu_jobs_per_author,
                )
                capped = json.dumps(pipeline) + "\n"
            except (ValueError, TypeError):
                pass
            else:
                effects.upload_pipeline(capped, no_interpolation=True, env=env)
                return
        print(
            f"WARNING: could not put {yaml_file} in {pr_tpu_concurrency_group}; "
            "uploading it uncapped",
            file=sys.stderr,
        )

    effects.upload_pipeline(payload, no_interpolation=False, env=env)


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------


class BootstrapEffects:
    """Thin IO layer isolating `buildkite-agent`, `git`, and GitHub API calls."""

    def __init__(
        self,
        repo_root: pathlib.Path = _REPO_ROOT,
        credential_helper: pathlib.Path = pathlib.Path(
            "/etc/buildkite-agent/git-credential-github-app"
        ),
        credential_timeout: float = 10.0,
        github_api_url: str = GITHUB_API_BASE_URL,
    ) -> None:
        self.repo_root = repo_root
        self.credential_helper = credential_helper
        self.credential_timeout = credential_timeout
        self.github_api_url = github_api_url

    def set_metadata(self, key: str, value: str, *, env: Mapping[str, str]) -> None:
        subprocess.run(
            ["buildkite-agent", "meta-data", "set", key, value],
            cwd=self.repo_root,
            env=env,
            check=True,
        )

    def annotate(self, annotation: Annotation, *, env: Mapping[str, str]) -> None:
        subprocess.run(
            [
                "buildkite-agent",
                "annotate",
                "--style",
                annotation.style,
                "--context",
                annotation.context,
                annotation.body,
            ],
            cwd=self.repo_root,
            env=env,
            check=True,
        )

    def dry_run_upload(self, content: str, *, env: Mapping[str, str]) -> str | None:
        proc = subprocess.run(
            ["buildkite-agent", "pipeline", "upload", "--dry-run", "--format", "json"],
            cwd=self.repo_root,
            env=env,
            input=content,
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else None

    def upload_pipeline(
        self,
        content: str,
        *,
        no_interpolation: bool,
        env: Mapping[str, str],
    ) -> None:
        cmd = ["buildkite-agent", "pipeline", "upload"]
        if no_interpolation:
            cmd.append("--no-interpolation")
        subprocess.run(
            cmd, cwd=self.repo_root, env=env, input=content, text=True, check=True
        )

    def pr_changed_files(
        self, base_branch: str, commit: str, *, env: Mapping[str, str]
    ) -> list[str]:
        fetch_base = subprocess.run(
            ["git", "fetch", "origin", base_branch, "--depth=20", "--quiet"],
            cwd=self.repo_root,
            env=env,
            check=False,
        )
        if fetch_base.returncode != 0:
            print("Base fetch failed")
        subprocess.run(
            ["git", "fetch", "origin", commit, "--depth=20", "--quiet"],
            cwd=self.repo_root,
            env=env,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{base_branch}...{commit}"],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        files = [line for line in diff.stdout.splitlines() if line]
        if files:
            return files
        tree = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "-m", commit],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        return [line for line in tree.stdout.splitlines() if line]

    def commit_changed_files(self, commit: str, *, env: Mapping[str, str]) -> list[str]:
        subprocess.run(
            ["git", "fetch", "origin", commit, "--depth=2", "--quiet"],
            cwd=self.repo_root,
            env=env,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--name-only", f"{commit}^", commit],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return [line for line in diff.stdout.splitlines() if line]

    def vllm_upstream_head(self, *, env: Mapping[str, str]) -> str:
        proc = subprocess.run(
            ["git", "ls-remote", VLLM_UPSTREAM_REPO, "HEAD"],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        tokens = proc.stdout.split()
        return tokens[0] if tokens else ""

    def _github_app_token(self, *, env: Mapping[str, str]) -> str | None:
        if not self.credential_helper.is_file():
            return None
        try:
            proc = subprocess.run(
                [str(self.credential_helper), "get"],
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.credential_timeout,
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

    def merged_pr_labels(
        self, pr_number: str, *, env: Mapping[str, str]
    ) -> tuple[str, ...]:
        # Labels of the PR a post-merge build on main merged. They do not reach
        # the build, so they are read from GitHub by the PR number GitHub puts at
        # the end of a squash commit's subject, with a token from the agent's
        # GitHub App helper when it has one. Any failure returns empty and the
        # build runs everything.
        headers = {"Accept": "application/vnd.github+json"}
        token = self._github_app_token(env=env)
        if token:
            headers["Authorization"] = f"token {token}"
        url = (
            f"{self.github_api_url.rstrip('/')}/repos/"
            f"{GITHUB_REPO}/issues/{pr_number}/labels"
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


# ---------------------------------------------------------------------------
# Orchestration & CLI
# ---------------------------------------------------------------------------


def run_bootstrap(
    env: Mapping[str, str] | None = None,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    effects: BootstrapEffects | None = None,
) -> int:
    """Execute the full Buildkite bootstrap flow and return an exit status."""
    child_env = dict(os.environ if env is None else env)
    effects = BootstrapEffects(repo_root=repo_root) if effects is None else effects
    ctx = BuildContext.from_env(child_env)

    print("--- Determining job priority", file=sys.stderr)
    priority_plan = plan_priority(ctx)
    print(priority_plan.log_line, file=sys.stderr)

    child_env["JOB_PRIORITY"] = str(priority_plan.job_priority)
    effects.set_metadata("JOB_PRIORITY", str(priority_plan.job_priority), env=child_env)
    # The kube launcher submits every workload of the build at this Kueue
    # priority class (ci-infra launch.py). None leaves them unclassed, at 0.
    if priority_plan.workload_priority is not None:
        effects.set_metadata(
            "WORKLOAD_PRIORITY", priority_plan.workload_priority, env=child_env
        )
    if priority_plan.annotation is not None:
        effects.annotate(priority_plan.annotation, env=child_env)
    if priority_plan.pr_tpu_concurrency_group is not None:
        child_env["PR_TPU_CONCURRENCY_GROUP"] = priority_plan.pr_tpu_concurrency_group

    print("--- :git: Checking if CI build should be skipped")
    if has_skip_ci_marker(ctx.message):
        print("Commit message contains [skip ci]. Skipping build.")
        return 0

    changed_files: list[str] = []
    if ctx.is_pr:
        print(f"PR detected. Target branch: {ctx.pr_base_branch}")
        changed_files = effects.pr_changed_files(
            ctx.pr_base_branch, ctx.commit, env=child_env
        )
    elif ctx.source != "schedule":
        # Any other build is of a commit, which on main is the squash merge of
        # one PR, so its diff against its parent is the whole change. Not a
        # scheduled build: the nightly tests main as it stands, whatever its
        # last commit touched.
        changed_files = effects.commit_changed_files(ctx.commit, env=child_env)

    if changed_files:
        print("Files changed:")
        print("\n".join(changed_files))
        if only_docs_changed(changed_files):
            print("Only documentation/non-code files changed. Skipping CI build.")
            return 0
        print("Code files changed. Proceeding with pipeline upload.")

    print("--- Starting Buildkite Bootstrap")
    if ctx.is_integration:
        base_vllm_hash = effects.vllm_upstream_head(env=child_env)
    else:
        pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        base_vllm_hash = parse_pinned_vllm_commit(pyproject_text)

    vllm_commit_hash = resolve_vllm_commit_hash(
        base_vllm_hash, ctx.vllm_commit_override
    )
    effects.set_metadata("VLLM_COMMIT_HASH", vllm_commit_hash, env=child_env)
    print(f"Using vLLM commit: {vllm_commit_hash}")

    # Since Buildkite prepends uploaded steps (inserts in reverse order),
    # uploading the test steps first and build steps last ensures they are
    # displayed and queued in correct order: Build -> Tests.
    if ctx.is_integration:
        # Upload promote step FIRST so Buildkite places it LAST after test steps.
        print("Uploading Promotion Pipeline Fragment")
        upload_with_priority(
            ".buildkite/pipeline_promote.yml",
            repo_root=repo_root,
            job_priority=priority_plan.job_priority,
            nightly=ctx.nightly_env,
            pr_tpu_concurrency_group=priority_plan.pr_tpu_concurrency_group,
            pr_tpu_jobs_per_author=ctx.pr_tpu_jobs_per_author,
            env=child_env,
            effects=effects,
        )

    merged_labels: tuple[str, ...] = ()
    if ctx.is_post_merge_webhook:
        pr_num = parse_merged_pr_number(ctx.message)
        if pr_num is not None:
            merged_labels = effects.merged_pr_labels(pr_num, env=child_env)

    existing_lanes = {
        f".buildkite/pipeline_{lane}.yml"
        for lane in ALL_LANES
        if (repo_root / f".buildkite/pipeline_{lane}.yml").is_file()
    }
    upload_plan = plan_uploads(
        ctx,
        merged_labels=merged_labels,
        existing_lane_files=existing_lanes,
    )

    if upload_plan.nightly:
        child_env["NIGHTLY"] = "1"
    child_env["SKIP_TPU_CI"] = "1" if upload_plan.skip_tpu_ci else "0"
    for annotation in upload_plan.pre_lane_annotations:
        effects.annotate(annotation, env=child_env)

    if upload_plan.invalid_ci_lanes_error is not None:
        raise BootstrapError(upload_plan.invalid_ci_lanes_error)

    for lane_file in upload_plan.lane_files:
        print(f"Uploading {lane_file}")
        upload_with_priority(
            lane_file,
            repo_root=repo_root,
            job_priority=priority_plan.job_priority,
            nightly=upload_plan.nightly,
            pr_tpu_concurrency_group=priority_plan.pr_tpu_concurrency_group,
            pr_tpu_jobs_per_author=ctx.pr_tpu_jobs_per_author,
            env=child_env,
            effects=effects,
        )

    if upload_plan.missing_ci_lanes_error is not None:
        raise BootstrapError(upload_plan.missing_ci_lanes_error)
    for annotation in upload_plan.post_lane_annotations:
        effects.annotate(annotation, env=child_env)

    child_env["OMNI_IMAGE_SKIP"] = upload_plan.omni_image_skip
    print("Uploading Build Pipeline")
    upload_with_priority(
        upload_plan.build_file,
        repo_root=repo_root,
        job_priority=priority_plan.job_priority,
        nightly=upload_plan.nightly,
        pr_tpu_concurrency_group=priority_plan.pr_tpu_concurrency_group,
        pr_tpu_jobs_per_author=ctx.pr_tpu_jobs_per_author,
        env=child_env,
        effects=effects,
    )

    print("--- Buildkite Bootstrap Finished")
    return 0


def run_upload_cli(
    yaml_file: str,
    priority_arg: str = "",
    *,
    env: Mapping[str, str] | None = None,
    cwd: pathlib.Path | None = None,
    effects: BootstrapEffects | None = None,
) -> int:
    """CLI handler for `bootstrap.py upload <yaml_file> [priority]`."""
    child_env = dict(os.environ if env is None else env)
    work_dir = pathlib.Path.cwd() if cwd is None else cwd
    effects = BootstrapEffects(repo_root=work_dir) if effects is None else effects
    priority = int(priority_arg) if priority_arg else PRIORITY_DEFAULT
    nightly = _env_or(child_env, "NIGHTLY", "0") == "1"
    concurrency_group = child_env.get("PR_TPU_CONCURRENCY_GROUP") or None
    jobs_per_author = int(
        _env_or(child_env, "PR_TPU_JOBS_PER_AUTHOR", str(PR_TPU_JOBS_PER_AUTHOR))
    )
    upload_with_priority(
        yaml_file,
        repo_root=work_dir,
        job_priority=priority,
        nightly=nightly,
        pr_tpu_concurrency_group=concurrency_group,
        pr_tpu_jobs_per_author=jobs_per_author,
        env=child_env,
        effects=effects,
    )
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
        elif args[0] == "upload" and 2 <= len(args) <= 3:
            rc = run_upload_cli(args[1], args[2] if len(args) == 3 else "")
        else:
            raise BootstrapError(
                "Usage: bootstrap.py [upload <yaml_file> [job_priority]]"
            )
    except BootstrapError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
