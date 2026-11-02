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

CLI entry points (`dev` and `vllm-pin` use Python 3.10+ stdlib only; main CI mode
loads PyYAML lazily on `queue=cpu` to evaluate `.buildkite/pipelines.yaml`):
- `bootstrap.py [--pipeline <slug>]`: main CI pipeline bootstrap (`bootstrap.sh`).
- `bootstrap.py dev`: experimental dev pipeline bootstrap (`bootstrap_dev.sh`).
- `bootstrap.py vllm-pin`: print the pinned upstream vLLM ref from `pyproject.toml`.
"""

from __future__ import annotations

import copy
import dataclasses
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
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cap_pr_tpu_concurrency

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

# Priority constants for pipeline jobs.
# On-call fix > Post-merge > Pre-merge > Integration pipeline > Benchmark >
# Other/Default > Nightly
PRIORITY_ONCALL_FIX = 12
PRIORITY_POST_MERGE = 11
PRIORITY_PRE_MERGE = 6
PRIORITY_INTEGRATION = 4
PRIORITY_BENCHMARK = 3
PRIORITY_DEFAULT = 2
PRIORITY_NIGHTLY = 1
PRIORITY_KERNEL_TUNING = -9

# How many TPU jobs on the kube fleet one author's pull-request builds may have
# in flight at once, waiting in Kueue or running. The rest wait in Buildkite as
# `limited`, holding no agent pod and no place in Kueue's queue, so one author's
# burst of PRs does not queue ahead of everyone else's. A PR build runs about
# 19, so this is two PRs at a time; 40 capped 6 of 45 authors in the week to
# 10-16. Nightly, post-merge and release builds are not PR builds and have no
# cap.
PR_TPU_JOBS_PER_AUTHOR = 40

DEFAULT_PIPELINES_FILE = ".buildkite/pipelines.yaml"
DEFAULT_DEV_PIPELINE_FILE = ".buildkite/pipeline_dev.yml"
GITHUB_APP_CREDENTIAL_HELPER = pathlib.Path(
    "/etc/buildkite-agent/git-credential-github-app"
)
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
_RELEASE_TAG_RE = re.compile(r"^v?[0-9]+(\.[a-zA-Z0-9]+)*([a-zA-Z]+[0-9]*)?$")
_VLLM_PIN_RE = re.compile(
    r".*vllm @ git\+https://github\.com/vllm-project/vllm\.git@([a-zA-Z0-9.-]+).*"
)
_MISSING_VLLM_PIN_ERROR = (
    "ERROR: vLLM commit hash is missing from pyproject.toml. "
    "Cannot proceed without a pinned hash."
)
_PIPELINE_KEYS = frozenset({"extends", "vllm_ref", "files", "variants"})


class BootstrapError(Exception):
    """Fatal configuration or git/pin lookup error that exits bootstrap with 1."""


def _env_or(env: Mapping[str, str], key: str, default: str) -> str:
    """Match bash `${VAR:-default}`: treat both unset and empty string as default."""
    val = env.get(key, "")
    return val if val else default


def _split_labels(raw: str) -> tuple[str, ...]:
    return tuple(item for item in raw.split(",") if item)


# ---------------------------------------------------------------------------
# Context & conditions
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
    ci_variant: str
    skip_tpu_ci_env: bool
    vllm_commit_override: str
    pr_tpu_concurrency_group_env: str
    pr_tpu_jobs_per_author: int
    merged_labels: tuple[str, ...] = ()

    @property
    def is_pr(self) -> bool:
        return self.pull_request is not None

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
            pull_request=None if pr_raw == "false" else pr_raw,
            pr_base_branch=_env_or(env, "BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main"),
            pr_labels=_split_labels(env.get("BUILDKITE_PULL_REQUEST_LABELS", "")),
            author_email=author,
            nightly_env=_env_or(env, "NIGHTLY", "0") == "1",
            torchtpu_nightly_env=_env_or(env, "TORCHTPU_NIGHTLY", "0") == "1",
            ci_variant=env.get("CI_VARIANT", "").strip(),
            skip_tpu_ci_env=_env_or(env, "SKIP_TPU_CI", "0") == "1",
            vllm_commit_override=env.get("VLLM_COMMIT_HASH_OVERRIDE", ""),
            pr_tpu_concurrency_group_env=env.get("PR_TPU_CONCURRENCY_GROUP", ""),
            pr_tpu_jobs_per_author=PR_TPU_JOBS_PER_AUTHOR,
        )


def _cond_pr(ctx: BuildContext) -> bool:
    """BUILDKITE_PULL_REQUEST is set (not 'false')."""
    return ctx.is_pr


def _cond_main(ctx: BuildContext) -> bool:
    """Non-PR build on branch 'main'."""
    return not ctx.is_pr and ctx.branch == "main"


def _cond_scheduled(ctx: BuildContext) -> bool:
    """BUILDKITE_SOURCE == 'schedule'."""
    return ctx.source == "schedule"


def _cond_release_tag(ctx: BuildContext) -> bool:
    """BUILDKITE_TAG matches a release version tag."""
    return bool(_RELEASE_TAG_RE.match(ctx.tag))


def _cond_full_suite(ctx: BuildContext) -> bool:
    """NIGHTLY == '1' or PR label 'run-full-ci'."""
    return ctx.nightly_env or (ctx.is_pr and "run-full-ci" in ctx.pr_labels)


def _cond_perf(ctx: BuildContext) -> bool:
    """full-suite, PR 'run-perf-eval', or CI_VARIANT == 'perf-only'."""
    return (
        _cond_full_suite(ctx)
        or (ctx.is_pr and "run-perf-eval" in ctx.pr_labels)
        or ctx.ci_variant == "perf-only"
    )


def _cond_skip_tpu(ctx: BuildContext) -> bool:
    """SKIP_TPU_CI == '1', or 'skip-tpu-ci' on the PR or merged PR."""
    labels = (
        ctx.pr_labels
        if ctx.is_pr
        else (ctx.merged_labels if ctx.is_post_merge_webhook else ())
    )
    return ctx.skip_tpu_ci_env or "skip-tpu-ci" in labels


def _cond_oncall_fix(ctx: BuildContext) -> bool:
    """PR carries the 'oncall-fix' label."""
    return ctx.is_pr and "oncall-fix" in ctx.pr_labels


CONDITIONS: dict[str, Callable[[BuildContext], bool]] = {
    "pr": _cond_pr,
    "main": _cond_main,
    "scheduled": _cond_scheduled,
    "release-tag": _cond_release_tag,
    "full-suite": _cond_full_suite,
    "perf": _cond_perf,
    "skip-tpu": _cond_skip_tpu,
    "oncall-fix": _cond_oncall_fix,
}


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
class DevBootstrapPlan:
    """Resolved settings for the `vllm-torchtpu-dev` experimental pipeline."""

    raw_job_priority: str
    pipeline_file: str
    branch: str
    commit: str
    nightly: bool
    pr_tpu_concurrency_group: str | None


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
    # PR_TPU_JOBS_PER_AUTHOR. Keyed on the commit author's email, which every
    # webhook build carries: the Buildkite creator is missing whenever the GitHub
    # account is not linked to a Buildkite user.
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


def plan_dev_bootstrap(env: Mapping[str, str]) -> DevBootstrapPlan:
    """Compute priority and target pipeline file for `vllm-torchtpu-dev`."""
    # Dev builds are experiments — they must never outrank real CI on a shared
    # queue. Default to PRIORITY_DEFAULT; override per build with
    # DEV_JOB_PRIORITY if an experiment needs to jump ahead of nightlies only.
    # No WORKLOAD_PRIORITY either: on kube an unclassed workload is ordered at 0,
    # ahead of the nightlies and behind every other kind of build.
    return DevBootstrapPlan(
        raw_job_priority=_env_or(env, "DEV_JOB_PRIORITY", str(PRIORITY_DEFAULT)),
        pipeline_file=_env_or(env, "DEV_PIPELINE_FILE", DEFAULT_DEV_PIPELINE_FILE),
        branch=_env_or(env, "BUILDKITE_BRANCH", "unknown"),
        commit=_env_or(env, "BUILDKITE_COMMIT", "unknown"),
        nightly=_env_or(env, "NIGHTLY", "0") == "1",
        pr_tpu_concurrency_group=env.get("PR_TPU_CONCURRENCY_GROUP") or None,
    )


def parse_dev_job_priority(raw_priority: str) -> int:
    """Validate `DEV_JOB_PRIORITY` as an integer at upload time."""
    try:
        return int(raw_priority)
    except ValueError as exc:
        raise BootstrapError(
            f"ERROR: invalid DEV_JOB_PRIORITY '{raw_priority}'; expected an integer."
        ) from exc


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
    raise BootstrapError(_MISSING_VLLM_PIN_ERROR)


def read_pinned_vllm_commit(repo_root: pathlib.Path = _REPO_ROOT) -> str:
    """Read `pyproject.toml` under `repo_root` and return the pinned vLLM ref."""
    pyproject_path = repo_root / "pyproject.toml"
    try:
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapError(f"ERROR: could not read {pyproject_path}: {exc}") from exc
    return parse_pinned_vllm_commit(pyproject_text)


def resolve_vllm_commit_hash(base_hash: str, override: str) -> str:
    """Apply `VLLM_COMMIT_HASH_OVERRIDE` when non-empty."""
    return override if override else base_hash


# ---------------------------------------------------------------------------
# `.buildkite/pipelines.yaml` & four-step pipeline rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPipeline:
    """Rendered Buildkite pipeline document, flags, and annotations."""

    document: dict[str, Any]
    files: tuple[str, ...]
    edges: dict[str, tuple[str, ...]]
    nightly: bool
    skip_tpu_ci: bool
    omni_image_skip: str
    annotations: tuple[Annotation, ...]


def _yaml() -> Any:
    try:
        import yaml  # pyrefly: ignore[missing-import]
    except ModuleNotFoundError as exc:
        raise BootstrapError("ERROR: PyYAML is required for bootstrap.py") from exc
    return yaml


def load_pipeline_spec(
    slug: str, pipelines_path: pathlib.Path = _REPO_ROOT / DEFAULT_PIPELINES_FILE
) -> dict[str, Any]:
    """Load `pipelines.yaml`, select `slug`, and resolve one-level `extends`."""
    if not pipelines_path.is_file():
        raise BootstrapError(f"ERROR: missing pipelines config: {pipelines_path}")
    raw = _yaml().safe_load(pipelines_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise BootstrapError(f"ERROR: {pipelines_path.name} must be a mapping")
    if not slug or slug not in raw or not isinstance(raw[slug], dict):
        label = (
            "BUILDKITE_PIPELINE_SLUG unset (use --pipeline)"
            if not slug
            else f"unknown pipeline {slug!r}"
        )
        raise BootstrapError(f"ERROR: {label}; pipelines: {', '.join(raw)}")
    entry = raw[slug]
    if extra := set(entry) - _PIPELINE_KEYS:
        raise BootstrapError(f"ERROR: {slug!r} has unknown keys {sorted(extra)}")
    base = raw.get(entry["extends"]) if "extends" in entry else {}
    if not isinstance(base, dict) or "extends" in base:
        raise BootstrapError(
            f"ERROR: invalid extends {entry.get('extends')!r} on {slug!r}"
        )
    vllm_ref = entry.get("vllm_ref", base.get("vllm_ref", "pinned"))
    files = {**(base.get("files") or {}), **(entry.get("files") or {})}
    variants = {**(base.get("variants") or {}), **(entry.get("variants") or {})}
    if vllm_ref not in ("pinned", "head") or not files:
        raise BootstrapError(f"ERROR: invalid vllm_ref or empty files on {slug!r}")
    if any(
        not isinstance(v, list) or not all(isinstance(f, str) and f in files for f in v)
        for v in variants.values()
    ):
        raise BootstrapError(f"ERROR: invalid variants in {slug!r}")
    return {"slug": slug, "vllm_ref": vllm_ref, "files": files, "variants": variants}


def _is_active(cfg: Mapping[str, Any], conds: Mapping[str, bool], owner: str) -> bool:
    for field in ("run_if", "skip_if"):
        val = cfg.get(field)
        if val is not None and (
            not isinstance(val, list)
            or not all(isinstance(c, str) and c in conds for c in val)
        ):
            raise BootstrapError(
                f"ERROR: unknown condition in {owner}.{field}: {val!r}"
            )
    run_if, skip_if = cfg.get("run_if") or [], cfg.get("skip_if") or []
    return not any(conds[c] for c in skip_if) and (
        not run_if or any(conds[c] for c in run_if)
    )


def _wanted_deps(
    raw_deps: Any, all_files: Container[str], conds: Mapping[str, bool], owner: str
) -> list[str]:
    if raw_deps is None:
        return []
    if not isinstance(raw_deps, list):
        raise BootstrapError(f"ERROR: {owner}.depends_on must be a list")
    deps: list[str] = []
    for item in raw_deps:
        if isinstance(item, dict) and isinstance(item.get("file"), str):
            if set(item) - {"file", "run_if", "skip_if"}:
                raise BootstrapError(
                    f"ERROR: invalid depends_on keys in {owner}: {item!r}"
                )
            if not _is_active(item, conds, f"{owner}.depends_on"):
                continue
            item = item["file"]
        if not isinstance(item, str) or item not in all_files:
            raise BootstrapError(f"ERROR: {owner} depends_on unknown file {item!r}")
        deps.append(item)
    return deps


def _load_step_file(
    step_path: pathlib.Path, *, nightly: bool, skip_tpu_ci: bool, omni_skip: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if not step_path.is_file():
        raise BootstrapError(f"ERROR: missing step file: {step_path.name}")
    doc = _yaml().safe_load(step_path.read_text(encoding="utf-8")) or {}
    file_env, steps = doc.get("env") or {}, copy.deepcopy(doc.get("steps") or [])
    keys: list[str] = []
    for step in steps:
        top_key = step.get("key") if isinstance(step, dict) else None
        if (
            not isinstance(step, dict)
            or "wait" in step
            or not isinstance(top_key, str)
            or not top_key
        ):
            raise BootstrapError(
                f"ERROR: step in {step_path.name} lacks a key: {step!r}"
            )
        for node in [step, *(step.get("steps") or () if "group" in step else ())]:
            if isinstance(raw_if := node.get("if"), str):
                if nightly:
                    raw_if = raw_if.replace('build.env("NIGHTLY")', '"1"')
                node["if"] = raw_if.replace(
                    "${SKIP_TPU_CI:-0}", "1" if skip_tpu_ci else "0"
                )
            if node is not step or "group" not in step:
                if file_env:
                    node["env"] = {**file_env, **(node.get("env") or {})}
                if node.get("skip") == "${OMNI_IMAGE_SKIP:-}":
                    node["skip"] = omni_skip
                    if not omni_skip:
                        del node["skip"]
        keys.append(top_key)
    return steps, keys


def render_pipeline(
    env: Mapping[str, str],
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    pipelines_path: pathlib.Path | None = None,
    spec: Mapping[str, Any] | None = None,
    merged_labels: Sequence[str] = (),
    conditions: Mapping[str, Callable[[BuildContext], bool]] = CONDITIONS,
) -> RenderedPipeline:
    """Pick active files, load steps, wire `depends_on` keys, and apply priority/cap."""
    ctx = dataclasses.replace(
        BuildContext.from_env(env), merged_labels=tuple(merged_labels)
    )
    priority_plan = plan_priority(ctx)
    spec = spec or load_pipeline_spec(
        ctx.pipeline_slug, pipelines_path or (repo_root / DEFAULT_PIPELINES_FILE)
    )
    conds = {name: bool(fn(ctx)) for name, fn in conditions.items()}
    all_files: dict[str, Any] = spec["files"]
    if ctx.ci_variant and ctx.ci_variant not in spec["variants"]:
        known = ", ".join(spec["variants"]) or "none"
        raise BootstrapError(
            f"ERROR: unknown CI_VARIANT {ctx.ci_variant!r}; variants: {known}"
        )
    variant_set = set(spec["variants"][ctx.ci_variant]) if ctx.ci_variant else None

    # 1. Pick uploaded files (conditions + optional CI_VARIANT subset).
    selected: dict[str, list[str]] = {}
    for fname, raw_cfg in all_files.items():
        cfg = {} if raw_cfg is None else raw_cfg
        if not isinstance(cfg, dict) or (
            set(cfg) - {"depends_on", "run_if", "skip_if"}
        ):
            raise BootstrapError(f"ERROR: invalid config on {fname}: {cfg!r}")
        deps = _wanted_deps(cfg.get("depends_on"), all_files, conds, fname)
        if (variant_set is None or fname in variant_set) and _is_active(
            cfg, conds, fname
        ):
            selected[fname] = deps

    # pipeline_build.yml skips -omni when set (Buildkite caps skip reasons at 70 chars).
    nightly, skip_tpu_ci = bool(conds.get("full-suite")), bool(conds.get("skip-tpu"))
    omni_skip = (
        "Only TPU steps use the -omni image; skip-tpu-ci skips them."
        if skip_tpu_ci
        else (
            "Only the tests lane uses the -omni image; this build has none."
            if "pipeline_tests.yml" not in selected
            else ""
        )
    )

    # 2. Load each selected step file.
    loaded, file_keys = {}, {}
    for fname in selected:
        loaded[fname], file_keys[fname] = _load_step_file(
            repo_root / ".buildkite" / fname,
            nightly=nightly,
            skip_tpu_ci=skip_tpu_ci,
            omni_skip=omni_skip,
        )

    # 3. Add depends_on = top-level keys of uploaded dependency files (drop unuploaded).
    steps: list[dict[str, Any]] = []
    edges = {f: tuple(d for d in ds if d in file_keys) for f, ds in selected.items()}
    for fname, active_deps in edges.items():
        dep_keys = [k for d in active_deps for k in file_keys[d]]
        for step in loaded[fname]:
            if dep_keys:
                raw = step.get("depends_on")
                cur = (
                    []
                    if raw is None
                    else ([raw] if isinstance(raw, str) else list(raw))
                )
                seen = {d["step"] if isinstance(d, dict) else d for d in cur}
                if merged := cur + [k for k in dep_keys if k not in seen]:
                    step["depends_on"] = merged
            steps.append(step)

    # 4. Apply job priority and per-author PR TPU concurrency cap.
    if priority_plan.pr_tpu_concurrency_group:
        cap_pr_tpu_concurrency.cap(
            steps, priority_plan.pr_tpu_concurrency_group, ctx.pr_tpu_jobs_per_author
        )

    notes: list[tuple[str, str]] = []
    if ctx.is_pr and "run-full-ci" in ctx.pr_labels:
        notes.append(
            ("full-ci", "Runs the full (nightly) suite (`run-full-ci` label).")
        )
    elif (
        not nightly
        and ctx.is_pr
        and "run-perf-eval" in ctx.pr_labels
        and not ctx.ci_variant
    ):
        notes.append(("perf-eval", "Runs the perf/eval lane (`run-perf-eval` label)."))
    if skip_tpu_ci:
        notes.append(
            (
                "skip-tpu-ci",
                "TPU steps not run (`skip-tpu-ci` label); the CPU steps still run.",
            )
        )
    cond_str = ", ".join(f"`{k}={str(v).lower()}`" for k, v in conds.items())
    files_str = ", ".join(f"`{f}`" for f in file_keys) or "none"
    edges_str = (
        ", ".join(f"`{f} -> {', '.join(ds)}`" for f, ds in edges.items() if ds)
        or "none"
    )
    notes.append(
        (
            "pipeline-graph",
            f"**Pipeline:** `{spec['slug']}`\n**Conditions:** {cond_str}\n"
            f"**Files:** {files_str}\n**Edges:** {edges_str}",
        )
    )
    return RenderedPipeline(
        document={"priority": priority_plan.job_priority, "steps": steps},
        files=tuple(file_keys),
        edges=edges,
        nightly=nightly,
        skip_tpu_ci=skip_tpu_ci,
        omni_image_skip=omni_skip,
        annotations=tuple(Annotation("info", k, b) for k, b in notes),
    )


def dump_pipeline(doc: Mapping[str, Any]) -> str:
    """Serialize a rendered pipeline dict as YAML without aliases."""
    yaml = _yaml()
    dumper = type(
        "_NoAliasDumper",
        (yaml.SafeDumper,),
        {"ignore_aliases": lambda self, data: True},
    )
    return str(yaml.dump(dict(doc), Dumper=dumper, sort_keys=False))


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
        credential_helper: pathlib.Path = GITHUB_APP_CREDENTIAL_HELPER,
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
    pipeline_slug: str | None = None,
    repo_root: pathlib.Path = _REPO_ROOT,
    effects: BootstrapEffects | None = None,
) -> int:
    """Execute the full Buildkite bootstrap flow and return an exit status."""
    child_env = dict(os.environ if env is None else env)
    if pipeline_slug is not None:
        child_env["BUILDKITE_PIPELINE_SLUG"] = pipeline_slug
    effects = BootstrapEffects(repo_root=repo_root) if effects is None else effects
    ctx = BuildContext.from_env(child_env)
    spec = load_pipeline_spec(ctx.pipeline_slug, repo_root / DEFAULT_PIPELINES_FILE)

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
    if spec["vllm_ref"] == "head":
        base_vllm_hash = effects.vllm_upstream_head(env=child_env)
    else:
        base_vllm_hash = read_pinned_vllm_commit(repo_root)

    vllm_commit_hash = resolve_vllm_commit_hash(
        base_vllm_hash, ctx.vllm_commit_override
    )
    effects.set_metadata("VLLM_COMMIT_HASH", vllm_commit_hash, env=child_env)
    print(f"Using vLLM commit: {vllm_commit_hash}")

    merged_labels: tuple[str, ...] = ()
    if ctx.is_post_merge_webhook:
        pr_num = parse_merged_pr_number(ctx.message)
        if pr_num is not None:
            merged_labels = effects.merged_pr_labels(pr_num, env=child_env)

    rendered = render_pipeline(
        child_env, repo_root=repo_root, spec=spec, merged_labels=merged_labels
    )
    if rendered.nightly:
        child_env["NIGHTLY"] = "1"
    child_env["SKIP_TPU_CI"] = "1" if rendered.skip_tpu_ci else "0"
    child_env["OMNI_IMAGE_SKIP"] = rendered.omni_image_skip
    for annotation in rendered.annotations:
        effects.annotate(annotation, env=child_env)

    print(
        f"--- :pipeline: Uploading pipeline with priority {priority_plan.job_priority}"
    )
    effects.upload_pipeline(
        dump_pipeline(rendered.document), no_interpolation=False, env=child_env
    )
    print("--- Buildkite Bootstrap Finished")
    return 0


def run_bootstrap_dev(
    env: Mapping[str, str] | None = None,
    *,
    repo_root: pathlib.Path = _REPO_ROOT,
    effects: BootstrapEffects | None = None,
) -> int:
    """Execute the `vllm-torchtpu-dev` bootstrap flow (`bootstrap_dev.sh`)."""
    child_env = dict(os.environ if env is None else env)
    effects = BootstrapEffects(repo_root=repo_root) if effects is None else effects

    vllm_commit_hash = read_pinned_vllm_commit(repo_root)
    effects.set_metadata("VLLM_COMMIT_HASH", vllm_commit_hash, env=child_env)
    print(f"Using vLLM commit: {vllm_commit_hash}")

    dev_plan = plan_dev_bootstrap(child_env)
    child_env["JOB_PRIORITY"] = dev_plan.raw_job_priority
    effects.set_metadata("JOB_PRIORITY", dev_plan.raw_job_priority, env=child_env)
    print(f"--- Dev build priority: {dev_plan.raw_job_priority}")

    if not (repo_root / dev_plan.pipeline_file).is_file():
        print(
            f"ERROR: dev pipeline file '{dev_plan.pipeline_file}' not found on "
            f"branch '{dev_plan.branch}' at commit {dev_plan.commit}. "
            "Create it on your branch, or point DEV_PIPELINE_FILE at an existing "
            "file."
        )
        return 1

    print(f"--- :pipeline: Uploading {dev_plan.pipeline_file}")
    job_priority = parse_dev_job_priority(dev_plan.raw_job_priority)
    upload_with_priority(
        dev_plan.pipeline_file,
        repo_root=repo_root,
        job_priority=job_priority,
        nightly=dev_plan.nightly,
        pr_tpu_concurrency_group=dev_plan.pr_tpu_concurrency_group,
        pr_tpu_jobs_per_author=PR_TPU_JOBS_PER_AUTHOR,
        env=child_env,
        effects=effects,
    )
    print("--- Buildkite Dev Bootstrap Finished")
    return 0


def _parse_main_args(args: Sequence[str]) -> tuple[str, str | None]:
    """Parse CLI arguments into `(mode, pipeline_slug)` without importing PyYAML."""
    if not args:
        return "ci", None
    if args == ["dev"]:
        return "dev", None
    if args == ["vllm-pin"]:
        return "vllm-pin", None
    if len(args) == 2 and args[0] == "--pipeline" and args[1]:
        return "ci", args[1]
    if (
        len(args) == 1
        and args[0].startswith("--pipeline=")
        and (slug := args[0].removeprefix("--pipeline="))
    ):
        return "ci", slug
    raise BootstrapError("Usage: bootstrap.py [--pipeline <slug> | dev | vllm-pin]")


def main(argv: Sequence[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(line_buffering=True)

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        mode, slug = _parse_main_args(args)
        if mode == "ci":
            rc = run_bootstrap(pipeline_slug=slug)
        elif mode == "dev":
            rc = run_bootstrap_dev()
        else:
            print(read_pinned_vllm_commit(_REPO_ROOT))
            rc = 0
    except BootstrapError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
