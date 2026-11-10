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
- `bootstrap.py upload-stages <stage> [<stage> ...]`: upload `<stage>` and chain rest.
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
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import cap_pr_tpu_concurrency

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_CREDENTIAL_HELPER = pathlib.Path("/etc/buildkite-agent/git-credential-github-app")
_GITHUB_API_URL = "https://api.github.com"

# Max TPU jobs one author's PR builds may have in flight on kube (~2 PRs at ~19
# jobs); extra jobs wait in Buildkite as `limited` and take no Kueue slot, while
# nightly, post-merge, and release builds run uncapped.
PR_TPU_JOBS_PER_AUTHOR = 40

_CI_MERGE_BOT_LOGIN = "github-actions[bot]"
# A merge-queue build that can't read the PR's /ci merge marker retries, then
# fails rather than drop the requested extras.
_MQ_EXTRAS_ATTEMPTS = 3
_MQ_EXTRAS_RETRY_SECONDS = 10
_LABEL_EXTRAS = {"run-perf-eval": "perf-eval", "run-full-ci": "nightly-suite"}
_PR_SUITE_LABELS = ("skip-tpu-ci", *_LABEL_EXTRAS)
_VALID_CI_EXTRAS = frozenset(_LABEL_EXTRAS.values())
_VALID_MERGE_EXTRAS = _VALID_CI_EXTRAS | {"precheck-only"}
_PR_NUM_RE = re.compile(r"^[0-9]+$")
_MQ_BRANCH_PR_RE = re.compile(r"^gh-readonly-queue/.+/pr-([0-9]+)-[^/]+$")
_CI_MERGE_MARKER_RE = re.compile(
    r"^<!-- ?ci-merge-extras: ?([^\n>]*?) ?-->$", re.MULTILINE
)
_CI_MERGE_ACTOR_RE = re.compile(
    r"^<!-- ?ci-merge-actor: ?([A-Za-z0-9_.\[\]-]+) ?-->$", re.MULTILINE
)
_CI_MERGE_SHA_RE = re.compile(
    r"^<!-- ?ci-merge-sha: ?([0-9a-fA-F]{7,64}) ?-->$", re.MULTILINE
)


class BootstrapError(Exception):
    """Fatal configuration or git/pin lookup error that exits bootstrap with 1."""


def _env_or(env: Mapping[str, str], key: str, default: str) -> str:
    """Match bash `${VAR:-default}`: treat both unset and empty string as default."""
    val = env.get(key, "")
    return val if val else default


def _split_labels(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.split(",") if item)


def _and_list(items: Sequence[str]) -> str:
    """Join `items` as prose: "a", "a and b", "a, b and c"."""
    return " and ".join(filter(None, (", ".join(items[:-1]), items[-1])))


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
    labels: tuple[str, ...]
    author_email: str
    nightly_env: bool
    torchtpu_nightly_env: bool
    suite_override: str
    vllm_commit_override: str
    pr_tpu_concurrency_group_env: str
    merge_queue_base_branch: str = ""
    merge_queue_base_commit: str = ""
    ci_run_tpu: bool = False
    ci_precheck: bool = False
    ci_pr_push_cpu_only: bool = False
    ci_extras: tuple[str, ...] = ()
    mq_extras: tuple[str, ...] = ()
    # TPU_VERSION_ALL on a build that also runs the generations outside
    # DEFAULT_TPU_GENERATIONS; empty runs those alone.
    tpu_version: str = ""

    @property
    def is_pr(self) -> bool:
        return self.pull_request is not None and not self.is_merge_queue

    @property
    def is_merge_queue(self) -> bool:
        return bool(self.merge_queue_base_branch)

    @property
    def is_ci_command(self) -> bool:
        return self.ci_precheck or self.ci_run_tpu

    @property
    def is_pr_like(self) -> bool:
        return self.is_pr or self.is_merge_queue or self.ci_run_tpu

    @property
    def merge_queue_pr_number(self) -> str | None:
        if self.pull_request and _PR_NUM_RE.match(self.pull_request):
            return self.pull_request
        match = _MQ_BRANCH_PR_RE.match(self.branch)
        return match.group(1) if match else None

    @property
    def is_post_merge(self) -> bool:
        return (
            not self.is_pr
            and not self.is_merge_queue
            and self.branch == "main"
            and self.source == "webhook"
        )

    @property
    def is_integration(self) -> bool:
        return self.pipeline_slug == "vllm-torchtpu-integration"

    @property
    def is_oncall_fix(self) -> bool:
        return self.is_pr and "oncall-fix" in self.labels

    @property
    def is_release(self) -> bool:
        return (
            not self.is_pr
            and not self.is_merge_queue
            and bool(
                _RELEASE_TAG_RE.match(self.tag) or _RELEASE_BRANCH_RE.match(self.branch)
            )
        )

    def check_suite_labels(self) -> None:
        """Raise `BootstrapError` if mutually exclusive PR suite labels are set."""
        if not self.is_pr:
            return
        matches = [label for label in _PR_SUITE_LABELS if label in self.labels]
        if len(matches) > 1:
            joined = ", ".join(matches)
            raise BootstrapError(
                f"ERROR: PR labels {joined} are mutually exclusive. "
                "Remove all but one of skip-tpu-ci / run-perf-eval / "
                "run-full-ci in the **Labels** sidebar on the GitHub PR "
                "page, then retry the **Bootstrap Pipeline** step."
            )

    # Extras a PR-like build asked for, from whichever channel it came in on.
    @property
    def extras(self) -> frozenset[str]:
        if self.is_merge_queue:
            if _RELEASE_BRANCH_RE.match(self.merge_queue_base_branch):
                # releases always run the nightly suite
                return frozenset({"nightly-suite"})
            return frozenset(self.mq_extras)  # from the /ci merge marker
        from_labels = {
            _LABEL_EXTRAS[label] for label in self.labels if label in _LABEL_EXTRAS
        }
        return frozenset(self.ci_extras) | (from_labels if self.is_pr else set())

    # Whether this build runs TPU lanes at all.
    @property
    def runs_tpu(self) -> bool:
        if self.ci_precheck or "precheck-only" in self.extras:
            return False
        if self.is_merge_queue or self.ci_run_tpu:  # queue and /ci run always run TPU
            return True
        if "skip-tpu-ci" in self.labels:
            return False
        # after cutover, PR pushes are CPU-only
        return not (self.is_pr and self.ci_pr_push_cpu_only)

    # Whether this build runs pre-commit itself (pipeline_precommit.yml), so the
    # required Buildkite status covers it: always in the queue, and on PR and /ci
    # builds after cutover. Docs-only changes still get it.
    @property
    def runs_precommit(self) -> bool:
        return self.is_merge_queue or (
            self.ci_pr_push_cpu_only and (self.is_pr or self.is_ci_command)
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BuildContext:
        suite_override = env.get("CI_SUITE", "").strip()
        if env.get("CI_LANES", "").strip() and not suite_override:
            raise BootstrapError(
                "ERROR: CI_LANES was removed; set CI_SUITE=<name> "
                f"(suites: {' '.join(SUITES)})"
            )
        if _env_or(env, "SKIP_TPU_CI", "0").strip() != "0":
            raise BootstrapError(
                "ERROR: SKIP_TPU_CI was removed; use the skip-tpu-ci PR label "
                "(or 'bootstrap.py upload-stages build,cpu')"
            )
        ci_extras = tuple(
            p.strip() for p in env.get("CI_EXTRAS", "").split(",") if p.strip()
        )
        bad_extra = next((e for e in ci_extras if e not in _VALID_CI_EXTRAS), None)
        if bad_extra is not None:
            raise BootstrapError(
                f"ERROR: CI_EXTRAS names '{bad_extra}'; "
                "valid extras are: perf-eval, nightly-suite"
            )
        pr_raw = _env_or(env, "BUILDKITE_PULL_REQUEST", "false")
        pull_request = None if pr_raw == "false" else pr_raw
        mq_base_branch = env.get("BUILDKITE_MERGE_QUEUE_BASE_BRANCH", "")
        ci_run_tpu = _env_or(env, "CI_RUN_TPU", "0") == "1"
        ci_precheck = _env_or(env, "CI_PRECHECK", "0") == "1"
        ci_pr_push_cpu_only = _env_or(env, "CI_PR_PUSH_CPU_ONLY", "0") == "1"
        # PR labels drive PR pushes only, and only until cutover. Merge-queue and
        # /ci builds name their extras explicitly (/ci merge marker, CI_EXTRAS).
        labels = (
            _split_labels(env.get("BUILDKITE_PULL_REQUEST_LABELS", ""))
            if (
                pull_request is not None
                and not mq_base_branch
                and not (ci_run_tpu or ci_precheck)
                and not ci_pr_push_cpu_only
            )
            else ()
        )
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
            pull_request=pull_request,
            pr_base_branch=_env_or(env, "BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main"),
            labels=labels,
            author_email=author,
            nightly_env=_env_or(env, "NIGHTLY", "0") == "1",
            torchtpu_nightly_env=_env_or(env, "TORCHTPU_NIGHTLY", "0") == "1",
            suite_override=suite_override,
            vllm_commit_override=env.get("VLLM_COMMIT_HASH_OVERRIDE", ""),
            pr_tpu_concurrency_group_env=env.get("PR_TPU_CONCURRENCY_GROUP", ""),
            merge_queue_base_branch=mq_base_branch,
            merge_queue_base_commit=env.get("BUILDKITE_MERGE_QUEUE_BASE_COMMIT", ""),
            ci_run_tpu=ci_run_tpu,
            ci_precheck=ci_precheck,
            ci_pr_push_cpu_only=ci_pr_push_cpu_only,
            ci_extras=ci_extras,
            tpu_version=env.get("TPU_VERSION", "").strip(),
        )


@dataclass(frozen=True)
class Suite:
    """What a build runs. Each lane is .buildkite/pipeline_<lane>.yml.
    Stage N+1 is uploaded only after every step of stage N has passed;
    a one-stage suite uploads everything at once."""

    name: str
    about: str
    stages: tuple[tuple[str, ...], ...]
    nightly: bool = False


_CPU_STAGE = ("build", "cpu")  # cpu steps wait on build_docker themselves
_DEFAULT_TPU = ("tests", "integration", "verify")
_FULL_TPU = ("tests", "integration", "perf", "verify")
_ALL_LANES = (*_CPU_STAGE, *_FULL_TPU)

# Builds that don't wait for CPU: everything is one stage.
SUITES: dict[str, Suite] = {
    s.name: s
    for s in (
        Suite(
            "default",
            "CPU checks, TPU tests, integration, verify (no perf)",
            stages=((*_CPU_STAGE, *_DEFAULT_TPU),),
        ),
        Suite(
            "full",
            "every lane, including perf (nightly and release builds)",
            stages=(_ALL_LANES,),
            nightly=True,
        ),
        Suite(
            "perf",
            "image build and perf/eval only",
            stages=(("build", "perf"),),
        ),
        Suite(
            "integration",
            "full suite against vLLM HEAD, then LKG promotion",
            # pipeline_promote.yml's own `wait` holds it until the tests finish
            stages=((*_ALL_LANES, "promote"),),
        ),
    )
}

# Two-stage PR suites (plus cpu-only). `run-perf-eval` and `run-full-ci` share
# `_FULL_TPU`; `pr+full` also sets `NIGHTLY=1` via `Suite.nightly`.
_PR_SUITES: dict[str, Suite] = {
    "cpu-only": Suite(
        "pr+cpu-only",
        "CPU checks only, no TPU",
        stages=(_CPU_STAGE,),
    ),
    "pr": Suite(
        "pr",
        "PR: TPU waits for the CPU checks",
        stages=(_CPU_STAGE, _DEFAULT_TPU),
    ),
    "pr+perf": Suite(
        "pr+perf-eval",
        "PR: TPU and perf/eval wait for the CPU checks",
        stages=(_CPU_STAGE, _FULL_TPU),
    ),
    "pr+full": Suite(
        "pr+nightly-suite",
        "PR: full nightly suite waits for the CPU checks",
        stages=(_CPU_STAGE, _FULL_TPU),
        nightly=True,
    ),
}


@dataclass(frozen=True)
class TpuGeneration:
    """Upload-time variables passed to each pipeline file in TPU_GENERATION_FILES.

    Defines a TPU generation's pod shapes, label/group name (for distinct GitHub
    statuses), step key prefix (to avoid duplicate Buildkite keys), soft_fail
    behavior, and perf accelerator type and gating.
    """

    name: str
    shape_single: str
    shape_multi: str
    key_prefix: str
    soft_fail: bool
    # As tpu_info.get_chip_version reads it, which takes anything it does not
    # know for TPU7x.
    accelerator_type: str
    # True while no perf baselines have been measured on the generation: its
    # perf steps report their numbers without gating on another chip's.
    skip_perf_check: bool

    def upload_env(self) -> dict[str, str]:
        return {
            "TPU_GENERATION": self.name,
            "KUBE_SHAPE_SINGLE": self.shape_single,
            "KUBE_SHAPE_MULTI": self.shape_multi,
            "TPU_KEY_PREFIX": self.key_prefix,
            "TPU_SOFT_FAIL": "true" if self.soft_fail else "false",
            "TPU_ACCELERATOR": self.accelerator_type,
            "TPU_SKIP_PERF_CHECK": "1" if self.skip_perf_check else "0",
        }


# v7x keeps unprefixed step keys so depends_on and nightly_report references
# stay unchanged; prefixing its labels is safe since branch protection only
# requires the overall build status. v6e soft-fails while v7x-specific test
# assumptions are audited, and skips the perf gate (whose baselines are v7x).
TPU_V7X = TpuGeneration(
    name="v7x",
    shape_single="tpu7x-standard-1t/1x1x1",
    shape_multi="tpu7x-standard-4t/2x2x1",
    key_prefix="",
    soft_fail=False,
    accelerator_type="tpu7x",
    skip_perf_check=False,
)
TPU_V6E = TpuGeneration(
    name="v6e",
    shape_single="ct6e-standard-1t/1x1",
    shape_multi="ct6e-standard-8t/2x4",
    key_prefix="tpu6e_",
    soft_fail=True,
    accelerator_type="v6e",
    skip_perf_check=True,
)
# Pipeline files uploaded once per TPU generation in the listed order (last
# shows/queues first in Buildkite). Files not listed here are uploaded once.
TPU_GENERATION_FILES: Mapping[str, tuple[TpuGeneration, ...]] = {
    ".buildkite/pipeline_tests.yml": (TPU_V6E, TPU_V7X),
    ".buildkite/pipeline_integration.yml": (TPU_V6E, TPU_V7X),
    ".buildkite/pipeline_perf.yml": (TPU_V6E, TPU_V7X),
}
# The generations those files go up for when TPU_VERSION is unset.
DEFAULT_TPU_GENERATIONS: tuple[TpuGeneration, ...] = (TPU_V7X,)
# What a build sets TPU_VERSION to so those files also go up for the rest of
# their generations, side by side with the default ones. No value runs a
# non-default generation alone. The generation brings its shapes, so the build
# names no KUBE_SHAPE_* of its own.
TPU_VERSION_ALL = "all"


def choose_suite(ctx: BuildContext) -> Suite:
    """Return the `Suite` for `ctx`; `CI_SUITE` takes precedence."""
    if ctx.suite_override:
        if ctx.suite_override not in SUITES:
            raise BootstrapError(
                f"ERROR: unknown CI_SUITE '{ctx.suite_override}'; "
                f"suites are: {' '.join(SUITES)}"
            )
        return SUITES[ctx.suite_override]
    ctx.check_suite_labels()
    if not ctx.runs_tpu:
        return _PR_SUITES["cpu-only"]
    if ctx.is_integration:
        return SUITES["integration"]
    if ctx.is_pr_like:
        if "nightly-suite" in ctx.extras:
            return _PR_SUITES["pr+full"]
        if "perf-eval" in ctx.extras:
            return _PR_SUITES["pr+perf"]
        return _PR_SUITES["pr"]
    return SUITES["full"] if ctx.nightly_env or ctx.is_release else SUITES["default"]


def tpu_version_note(ctx: BuildContext, suite: Suite) -> str | None:
    """Check `ctx.tpu_version` against the build; return its annotation, if any.

    Files in TPU_GENERATION_FILES go up for DEFAULT_TPU_GENERATIONS only, or
    for every generation they list when TPU_VERSION is TPU_VERSION_ALL (see
    upload_envs). Any build may set it. Any other value is refused, as is a
    build with no lane in TPU_GENERATION_FILES, which would ignore it.
    """
    if not ctx.tpu_version:
        return None
    if ctx.tpu_version != TPU_VERSION_ALL:
        raise BootstrapError(
            f"ERROR: TPU_VERSION is '{ctx.tpu_version}'; set "
            f"TPU_VERSION={TPU_VERSION_ALL} to run every TPU generation, or "
            "leave it unset to run the default ones"
        )
    in_build = [
        lane
        for stage in suite.stages
        for lane in stage
        if f".buildkite/pipeline_{lane}.yml" in TPU_GENERATION_FILES
    ]
    if not in_build:
        per_generation = [
            pathlib.Path(p).stem.removeprefix("pipeline_") for p in TPU_GENERATION_FILES
        ]
        raise BootstrapError(
            f"ERROR: TPU_VERSION={TPU_VERSION_ALL} adds TPU generations to the "
            f"{_and_list(per_generation)} lanes, and this build runs none"
        )
    extra = [
        gen.name
        for gen in dict.fromkeys(
            g for gens in TPU_GENERATION_FILES.values() for g in gens
        )
        if gen not in DEFAULT_TPU_GENERATIONS
    ]
    return (
        f"Also runs the TPU {_and_list(extra)} steps of the "
        f"{_and_list(in_build)} lane{'s' if len(in_build) > 1 else ''} "
        f"(`TPU_VERSION={TPU_VERSION_ALL}`)."
    )


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
    if ctx.is_oncall_fix:
        return "oncall-fix"
    if ctx.is_pr or ctx.is_ci_command:
        return "pre-merge"
    if ctx.branch == "main" or ctx.is_merge_queue:
        return "post-merge"
    return "other"


# Commit-message, changed-file, and vLLM-pin patterns (.pyrefly-baseline-full.json stays
# out of _DOCS_ONLY_RE because the Buildkite typecheck step reads it).
_SKIP_CI_RE = re.compile(r"\[(skip\sci|ci\sskip)\]")
_DOCS_ONLY_RE = re.compile(
    r"(\.md$|\.ico$|\.png$|^README$|^docs/|^\.github/CODEOWNERS$"
    r"|^LICENSE$|^\.git-blame-ignore-revs$|^\.pyrefly-baseline\.json$)"
)
_MERGED_PR_RE = re.compile(r"\(#([0-9]+)\)$")
# Keep in sync with .buildkite/pipeline_verify.yml and
# .buildkite/scripts/nightly_report.sh.
_RELEASE_TAG_RE = re.compile(r"^v?[0-9]+(\.[a-zA-Z0-9]+)*([a-zA-Z]+[0-9]*)?$")
# Keep in sync with RELEASE_BRANCH_REGEX in .buildkite/release/lib.sh.
_RELEASE_BRANCH_RE = re.compile(
    r"^releases/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
_VLLM_PIN_RE = re.compile(
    r".*vllm @ git\+https://github\.com/vllm-project/vllm\.git@([a-zA-Z0-9.-]+).*"
)
_MISSING_VLLM_PIN_ERROR = (
    "ERROR: vLLM commit hash is missing from pyproject.toml. "
    "Cannot proceed without a pinned hash."
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


def parse_merge_queue_comments_page(
    items: object, *, head_sha: str = ""
) -> tuple[bool, tuple[str, ...], str]:
    """Extract the last valid `/ci merge` bot marker from a comments page."""
    if not isinstance(items, list):
        raise ValueError("unexpected GitHub comments payload")
    found, latest_extras, latest_actor = False, (), ""
    for c in items:
        user = c.get("user") if isinstance(c, dict) else None
        body = c.get("body") if isinstance(c, dict) else None
        if (
            not isinstance(user, dict)
            or user.get("type") != "Bot"
            or user.get("login") != _CI_MERGE_BOT_LOGIN
            or not isinstance(body, str)
        ):
            continue
        actor_m = _CI_MERGE_ACTOR_RE.search(body)
        c_actor = actor_m.group(1) if actor_m else ""
        sha_m = _CI_MERGE_SHA_RE.search(body)
        c_sha = sha_m.group(1) if sha_m else ""
        for match in _CI_MERGE_MARKER_RE.finditer(body):
            parts = tuple(p.strip() for p in match.group(1).split(",") if p.strip())
            if bad := [p for p in parts if p not in _VALID_MERGE_EXTRAS]:
                raise ValueError(f"unknown extra {bad[0]!r}")
            if "precheck-only" in parts and len(parts) > 1:
                raise ValueError("precheck-only cannot be combined with other extras")
            if parts and (not c_actor or not c_sha or (head_sha and c_sha != head_sha)):
                continue
            found, latest_extras, latest_actor = (
                True,
                parts,
                (c_actor if parts else ""),
            )
    return found, latest_extras, latest_actor


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


def _diff_names(*diff_specs: Sequence[str], env: Mapping[str, str]) -> list[str]:
    for args in diff_specs:
        diff = _git("diff", "--name-only", *args, env=env)
        if diff.returncode == 0 and (
            files := [line for line in diff.stdout.splitlines() if line]
        ):
            return files
    return []


def _changed_files(ctx: BuildContext, *, env: Mapping[str, str]) -> list[str]:
    """Return files changed by a PR, merge-queue, or non-scheduled commit build."""
    if ctx.is_ci_command:
        return []
    if ctx.is_merge_queue:
        if ctx.merge_queue_base_commit:
            diff_base = ctx.merge_queue_base_commit
            fetch_ok = (
                _git(
                    "fetch", "origin", diff_base, "--depth=20", "--quiet", env=env
                ).returncode
                == 0
            )
        else:
            base = ctx.merge_queue_base_branch
            diff_base = f"origin/{base}"
            ref = f"+refs/heads/{base}:refs/remotes/origin/{base}"
            fetch_ok = (
                _git(
                    "fetch", "origin", ref, "--depth=20", "--quiet", env=env
                ).returncode
                == 0
                or _git(
                    "fetch", "origin", base, "--depth=20", "--quiet", env=env
                ).returncode
                == 0
            )
        if not fetch_ok:
            print("Base fetch failed; running full merge-queue gate")
            return []
        _git("fetch", "origin", ctx.commit, "--depth=20", "--quiet", env=env)
        if files := _diff_names(
            [f"{diff_base}...{ctx.commit}"], [diff_base, ctx.commit], env=env
        ):
            return files
        print("Merge-queue diff failed or empty; running full merge-queue gate")
        return []
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
        if files := _diff_names(
            [f"origin/{ctx.pr_base_branch}...{ctx.commit}"], env=env
        ):
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
        return _diff_names([f"{ctx.commit}^", ctx.commit], env=env)

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


def _github_get(path: str, *, token: str | None) -> str:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = (
        f"{_GITHUB_API_URL.rstrip('/')}/repos/"
        f"vllm-project/vllm-torchtpu/{path.lstrip('/')}"
    )
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=10
    ) as resp:
        return resp.read().decode("utf-8")


def _github_labels(
    pr_number: str, *, env: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Read labels of a merged PR from GitHub; return `()` on any failure."""
    try:
        items = json.loads(
            _github_get(
                f"issues/{pr_number}/labels",
                token=_github_app_token(env=env),
            )
        )
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


def _format_request_error(exc: BaseException) -> str:
    msg = (
        f"HTTP {exc.code} {exc.reason}"
        if isinstance(exc, urllib.error.HTTPError)
        else str(exc)
    )
    return " ".join(msg.split()) or "HTTP request failed"


def _merge_queue_pr_extras(
    pr_number: str, *, env: Mapping[str, str] | None = None
) -> tuple[tuple[str, ...], str, str | None]:
    """Return `(extras, actor, error)` from `/ci merge` bot markers on `pr_number`."""
    token = _github_app_token(env=env)
    use_auth = token is not None

    def _fetch(path: str) -> tuple[str | None, str | None]:
        nonlocal use_auth
        try:
            return _github_get(path, token=token if use_auth else None), None
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            if not use_auth:
                return None, _format_request_error(exc)
            use_auth = False
            try:
                return _github_get(path, token=None), None
            except (
                OSError,
                urllib.error.URLError,
                http.client.HTTPException,
            ) as fallback_exc:
                return None, _format_request_error(fallback_exc)

    head_sha: str | None = None
    latest_extras: tuple[str, ...] = ()
    latest_actor = ""
    for page in range(1, 51):
        raw, err = _fetch(f"issues/{pr_number}/comments?per_page=100&page={page}")
        if err is not None or raw is None:
            return (), "", err
        if head_sha is None and "ci-merge-extras:" in raw:
            pr_raw, err = _fetch(f"pulls/{pr_number}")
            if err is not None or pr_raw is None:
                return (), "", err
            try:
                pr_obj = json.loads(pr_raw)
            except ValueError as exc:
                return (), "", (" ".join(str(exc).split()) or "invalid JSON")
            head = pr_obj.get("head") if isinstance(pr_obj, dict) else None
            sha = head.get("sha") if isinstance(head, dict) else None
            if not isinstance(sha, str) or not sha:
                return (), "", "missing PR head SHA"
            head_sha = sha
        try:
            items = json.loads(raw)
            found, p_extras, p_actor = parse_merge_queue_comments_page(
                items, head_sha=head_sha or ""
            )
        except ValueError as exc:
            return (), "", (" ".join(str(exc).split()) or "invalid JSON")
        if found:
            latest_extras, latest_actor = p_extras, p_actor
        if len(items) < 100:
            break
    else:
        return (), "", "comment history exceeds 5000 comments"
    return latest_extras, latest_actor, None


def upload_with_priority(
    yaml_file: str,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    job_priority: int,
    nightly: bool,
    pr_tpu_concurrency_group: str | None,
    pr_tpu_jobs_per_author: int = PR_TPU_JOBS_PER_AUTHOR,
    env: Mapping[str, str],
    tpu_generation: str = "",
) -> None:
    """Upload one pipeline YAML file with priority and optional PR TPU capping.

    `tpu_generation` names the TPU generation whose variables `env` carries,
    for the log of a file that goes up once per generation.
    """
    upload = f"{yaml_file} for TPU {tpu_generation}" if tpu_generation else yaml_file
    tpu = f"TPU {tpu_generation}" if tpu_generation else "TPU"
    print(f"--- :pipeline: Uploading {upload} with priority {job_priority}")
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
                    f"WARNING: failed to cap {tpu} concurrency for {yaml_file}: {exc}",
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
            f"WARNING: could not put {tpu} {yaml_file} in {pr_tpu_concurrency_group}; "
            "uploading it uncapped",
            file=sys.stderr,
        )

    _agent("pipeline", "upload", input_text=payload, env=env)


def upload_envs(path: str, tpu_version: str = "") -> tuple[dict[str, str], ...]:
    """Return the extra variables of each upload of `path`, in upload order.

    A file in TPU_GENERATION_FILES goes up once per generation in
    DEFAULT_TPU_GENERATIONS, with that generation's variables - or, when
    `tpu_version` is TPU_VERSION_ALL, once per generation it lists; any other
    file goes up once, with none.
    """
    if path not in TPU_GENERATION_FILES:
        return ({},)
    generations = TPU_GENERATION_FILES[path]
    if tpu_version != TPU_VERSION_ALL:
        generations = tuple(g for g in generations if g in DEFAULT_TPU_GENERATIONS)
    return tuple(generation.upload_env() for generation in generations)


def upload_stages(
    stages: Sequence[Sequence[str]],
    env: Mapping[str, str] | None = None,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    stage_index: int = 0,
) -> int:
    """Upload `stages[0]`'s lane files and, if more stages remain, `wait` + uploader.

    The `wait` step gates `rest` on every step uploaded in `stages[0]` (including
    both Docker image builds in `pipeline_build.yml`, such as `-omni`, and all
    `pipeline_cpu.yml` checks); a retried uploader skips lanes it already uploaded.
    A file in TPU_GENERATION_FILES goes up once per default TPU generation, or
    once per generation it lists when TPU_VERSION is TPU_VERSION_ALL.
    """
    for stage in stages:
        for lane in stage:
            lane_file = f".buildkite/pipeline_{lane}.yml"
            if not (repo_root / lane_file).is_file():
                raise BootstrapError(f"ERROR: lane file '{lane_file}' not found")

    child_env = dict(os.environ if env is None else env)
    step_key = child_env.get("BUILDKITE_STEP_KEY", "")
    if step_key.startswith("upload_stage_"):
        stage_index = int(step_key.removeprefix("upload_stage_"))
    stage_key = f"upload_stage_{stage_index}"
    job_priority = int(_env_or(child_env, "JOB_PRIORITY", str(PRIORITIES["other"][0])))
    nightly = _env_or(child_env, "NIGHTLY", "0") == "1"
    concurrency_group = child_env.get("PR_TPU_CONCURRENCY_GROUP") or None
    # Checked by run_bootstrap; a later stage's uploader inherits it from the
    # build's environment.
    tpu_version = child_env.get("TPU_VERSION", "").strip()

    # Buildkite inserts each `pipeline upload` right after the currently running
    # step (LIFO), so upload `wait` + the next-stage uploader first and then
    # `stages[0]` in reverse order so `stages[0]` runs before the `wait`.
    stage_num = stage_index + 1
    rest = stages[1:]
    print(f"Stage {stage_num} now: {', '.join(stages[0])}")
    for offset, next_stage in enumerate(rest, start=1):
        prev_num = stage_num + offset - 1
        print(
            f"Stage {prev_num + 1} after stage {prev_num} passes: "
            f"{', '.join(next_stage)}"
        )
    if len(stages[0]) > 1 or rest:
        print(
            "Uploading in reverse order (Buildkite inserts each upload right "
            "after the running step)."
        )
    if rest:
        next_key = f"upload_stage_{stage_index + 1}"
        print(
            f"--- :pipeline: Stage {stage_num + 1} is uploaded by step "
            f"{next_key} after a wait"
        )
        step_env = {
            k: child_env[k]
            for k in (
                "CI_FLEET",
                "JOB_PRIORITY",
                "NIGHTLY",
                "PR_TPU_CONCURRENCY_GROUP",
            )
            if child_env.get(k)
        }
        stage_args = " ".join(",".join(s) for s in rest)
        uploader = {
            "priority": job_priority,
            "steps": [
                "wait",
                {
                    "label": f":pipeline: Upload {', '.join(rest[0])}",
                    "key": next_key,
                    "agents": {"queue": "cpu"},
                    "timeout_in_minutes": 15,
                    "retry": {
                        "manual": {"permit_on_passed": False},
                        "automatic": [{"exit_status": -1, "limit": 2}],
                    },
                    "env": step_env,
                    "command": (
                        "python3 .buildkite/scripts/bootstrap.py "
                        f"upload-stages {stage_args}"
                    ),
                },
            ],
        }
        _agent(
            "pipeline",
            "upload",
            input_text=json.dumps(uploader) + "\n",
            env=child_env,
        )

    for lane in reversed(stages[0]):
        lane_file = f".buildkite/pipeline_{lane}.yml"
        # A file in TPU_GENERATION_FILES goes up once per generation, each with
        # its own record, so a retried uploader sends up only the ones it has
        # not: Buildkite rejects a step key uploaded twice.
        for extra_env in upload_envs(lane_file, tpu_version):
            generation = extra_env.get("TPU_GENERATION", "")
            upload = f"{lane}:{generation}" if generation else lane
            meta_key = f"upload-stages:{stage_key}:{upload}"
            if (
                _agent(
                    "meta-data", "exists", meta_key, check=False, env=child_env
                ).returncode
                == 0
            ):
                print(f"Skipping {upload} (already uploaded for {stage_key})")
                continue
            upload_with_priority(
                lane_file,
                repo_root=repo_root,
                job_priority=job_priority,
                nightly=nightly,
                pr_tpu_concurrency_group=concurrency_group,
                env={**child_env, **extra_env},
                tpu_generation=generation,
            )
            _agent("meta-data", "set", meta_key, "1", env=child_env)
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
    # A post-merge build honors the merged PR's skip-tpu-ci label until cutover.
    if (
        ctx.is_post_merge
        and not ctx.ci_pr_push_cpu_only
        and (pr := parse_merged_pr_number(ctx.message))
    ):
        ctx = replace(ctx, labels=_github_labels(pr, env=child_env))
    try:
        ctx.check_suite_labels()
    except BootstrapError as exc:
        _agent(
            "annotate",
            "--style",
            "error",
            "--context",
            "ci-labels",
            str(exc),
            env=child_env,
        )
        raise

    # 1. Kind and priority (metadata).
    print("--- Determining job priority")
    kind = build_kind(ctx)
    job_priority, workload_priority = PRIORITIES[kind]
    print(f"Build type: {kind} - Priority: {job_priority}")
    child_env["JOB_PRIORITY"] = str(job_priority)
    _agent("meta-data", "set", "JOB_PRIORITY", str(job_priority), env=child_env)
    # The kube launcher submits every workload of the build at this Kueue
    # priority class (ci-infra launch.py). None leaves them unclassed, at 0.
    if workload_priority is not None:
        _agent(
            "meta-data", "set", "WORKLOAD_PRIORITY", workload_priority, env=child_env
        )
    if kind == "oncall-fix":
        _agent(
            "annotate",
            "--style",
            "warning",
            "--context",
            "oncall-fix",
            "Runs at on-call priority (`oncall-fix` label): "
            "ahead of every other build, including main.",
            env=child_env,
        )

    # 2. Skip checks ([skip ci], docs-only).
    print("--- :git: Checking if CI build should be skipped")
    if (
        not ctx.is_merge_queue
        and not ctx.is_ci_command
        and has_skip_ci_marker(ctx.message)
    ):
        print("Commit message contains [skip ci]. Skipping build.")
        return 0
    changed_files = _changed_files(ctx, env=child_env)
    if changed_files:
        print("Files changed:")
        print("\n".join(changed_files))
        if only_docs_changed(changed_files):
            if ctx.is_merge_queue:
                _agent(
                    "annotate",
                    "--style",
                    "info",
                    "--context",
                    "docs-only",
                    "Docs-only change; skipped CPU and TPU checks in the merge queue.",
                    env=child_env,
                )
            if ctx.runs_precommit:
                print("Only documentation/non-code files changed. Running pre-commit.")
                upload_stages((("precommit",),), child_env, repo_root=repo_root)
                return 0
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

    # 4. Suite selection, NIGHTLY policy, and annotation.
    if (
        not ctx.ci_precheck
        and ctx.is_merge_queue
        and not _RELEASE_BRANCH_RE.match(ctx.merge_queue_base_branch)
        and ctx.merge_queue_pr_number
    ):
        mq_extras: tuple[str, ...] = ()
        mq_actor = ""
        mq_err: str | None = None
        for attempt in range(1, _MQ_EXTRAS_ATTEMPTS + 1):
            mq_extras, mq_actor, mq_err = _merge_queue_pr_extras(
                ctx.merge_queue_pr_number, env=child_env
            )
            if mq_err is None:
                break
            print(f"Could not read /ci merge extras (attempt {attempt}): {mq_err}")
            if attempt < _MQ_EXTRAS_ATTEMPTS:
                time.sleep(_MQ_EXTRAS_RETRY_SECONDS * attempt)
        if mq_err is not None:
            msg = (
                f"ERROR: could not read /ci merge extras for PR "
                f"#{ctx.merge_queue_pr_number}: {mq_err}. Add the PR to the merge "
                "queue again to retry."
            )
            _agent(
                "annotate",
                "--style",
                "error",
                "--context",
                "ci-merge-extras",
                msg,
                env=child_env,
            )
            raise BootstrapError(msg)
        if mq_extras:
            print(
                f"Merge-queue extras for PR #{ctx.merge_queue_pr_number}: "
                f"{', '.join(mq_extras)} (by @{mq_actor})"
            )
        ctx = replace(ctx, mq_extras=mq_extras)

    suite = choose_suite(ctx)
    nightly = ctx.nightly_env or suite.nightly
    nightly_tag = " [NIGHTLY=1]" if nightly else ""
    suite_note = f"CI suite: {suite.name}{nightly_tag}: {suite.about}"
    print(f"--- :pipeline: Suite {suite.name}{nightly_tag} (priority {job_priority})")
    print(suite_note)
    _agent(
        "annotate",
        "--style",
        "info",
        "--context",
        "ci-suite",
        suite_note,
        env=child_env,
    )
    try:
        tpu_note = tpu_version_note(ctx, suite)
    except BootstrapError as exc:
        _agent(
            "annotate",
            "--style",
            "error",
            "--context",
            "tpu-version",
            str(exc),
            env=child_env,
        )
        raise
    if tpu_note is not None:
        print(tpu_note)
        _agent(
            "annotate",
            "--style",
            "info",
            "--context",
            "tpu-version",
            tpu_note,
            env=child_env,
        )

    # 5. Env exports (NIGHTLY, OMNI_IMAGE_SKIP, PR_TPU_CONCURRENCY_GROUP).
    if nightly:
        child_env["NIGHTLY"] = "1"
    # pipeline_build.yml skips the -omni image build when this is non-empty.
    # Buildkite rejects the upload if a skip reason is over 70 characters.
    has_tests = any("tests" in stage for stage in suite.stages)
    child_env["OMNI_IMAGE_SKIP"] = (
        ""
        if has_tests
        else "Only the tests lane uses the -omni image; this build has none."
    )
    # Key the concurrency group on the commit author's email: every webhook
    # build carries it, whereas BUILDKITE_BUILD_CREATOR_EMAIL is missing when
    # the GitHub account is not linked to a Buildkite user.
    if (ctx.is_pr or ctx.is_ci_command) and ctx.author_email:
        concurrency_group: str | None = f"tpu-pr/{ctx.author_email.lower()}"
    else:
        concurrency_group = (
            None if ctx.is_merge_queue else ctx.pr_tpu_concurrency_group_env or None
        )
    if concurrency_group is not None:
        child_env["PR_TPU_CONCURRENCY_GROUP"] = concurrency_group

    # 6. Upload stage 0 (multi-stage suites chain later stages via upload-stages).
    stages = suite.stages
    if ctx.runs_precommit:
        stages = (("precommit", *stages[0]), *stages[1:])
    upload_stages(stages, child_env, repo_root=repo_root)
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


_USAGE = "Usage: bootstrap.py [dev | vllm-pin | upload-stages <stage> [<stage> ...]]"


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
        elif args[:1] == ["upload-stages"]:
            stages = tuple(_split_labels(arg) for arg in args[1:])
            if not stages or not all(stages):
                raise BootstrapError(_USAGE)
            rc = upload_stages(stages, stage_index=1)
        else:
            raise BootstrapError(_USAGE)
    except BootstrapError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
