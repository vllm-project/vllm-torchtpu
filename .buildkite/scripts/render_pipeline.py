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
"""Render `.buildkite/stages.yml` into a single Buildkite pipeline document."""

from __future__ import annotations

import copy
import os
import pathlib
import re
import sys
from collections.abc import Mapping
from typing import Any

import yaml

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cap_pr_tpu_concurrency  # noqa: E402

ALL_LANES = ("verify", "perf", "integration", "tests")
KNOWN_CONDITIONS = frozenset(
    {"ci-run", "merge-queue", "nightly", "oncall-fix", "pr", "release-tag"}
)
_RELEASE_TAG_RE = re.compile(r"^v?[0-9]+(\.[a-zA-Z0-9]+)*([a-zA-Z]+[0-9]*)?$")


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _normalize_depends_on(raw: Any) -> list[str | dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return list(raw)
    raise ValueError(f"Unsupported depends_on value: {raw!r}")


def _dep_key(entry: str | dict[str, Any]) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("step"), str):
        return entry["step"]
    raise ValueError(f"Unsupported depends_on entry: {entry!r}")


def _author_key(env: Mapping[str, str]) -> str:
    raw = (
        env.get("BUILDKITE_BUILD_AUTHOR_EMAIL")
        or env.get("BUILDKITE_BUILD_CREATOR_EMAIL")
        or ""
    ).strip()
    if raw:
        return raw.lower()
    existing = env.get("PR_TPU_CONCURRENCY_GROUP", "").strip()
    if existing.startswith("tpu-pr/"):
        return existing.removeprefix("tpu-pr/").lower()
    pr_val = env.get("BUILDKITE_PULL_REQUEST", "")
    if pr_val and pr_val != "false":
        return f"pr-{pr_val}"
    return "unknown"


def _build_conditions(env: Mapping[str, str]) -> tuple[dict[str, bool], set[str]]:
    pr_val = env.get("BUILDKITE_PULL_REQUEST", "")
    is_pr = bool(pr_val and pr_val != "false")
    pr_labels = {
        lbl.strip()
        for lbl in env.get("BUILDKITE_PULL_REQUEST_LABELS", "").split(",")
        if lbl.strip()
    }
    is_nightly = env.get("NIGHTLY") == "1" or env.get("TORCHTPU_NIGHTLY") == "1"
    is_release_tag = bool(_RELEASE_TAG_RE.match(env.get("BUILDKITE_TAG", "")))
    is_merge_queue = bool(
        env.get("BUILDKITE_MERGE_QUEUE_BASE_BRANCH")
        or env.get("BUILDKITE_BRANCH", "").startswith("gh-readonly-queue/")
    )
    return (
        {
            "ci-run": env.get("CI_RUN_TPU") == "1",
            "merge-queue": is_merge_queue,
            "nightly": is_nightly,
            "oncall-fix": is_pr and "oncall-fix" in pr_labels,
            "pr": is_pr,
            "release-tag": is_release_tag,
        },
        pr_labels,
    )


def _eval_conditions(spec: Mapping[str, Any], conditions: Mapping[str, bool]) -> bool:
    for field in ("when", "unless"):
        vals = spec.get(field)
        if vals is None:
            continue
        if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
            raise ValueError(f"{field} must be a list of condition names")
        for cond in vals:
            if cond not in KNOWN_CONDITIONS:
                valid = ", ".join(sorted(KNOWN_CONDITIONS))
                raise ValueError(
                    f"Unknown condition {cond!r}; valid conditions: {valid}"
                )
    if "unless" in spec and any(conditions[c] for c in spec["unless"]):
        return False
    return "when" not in spec or any(conditions[c] for c in spec["when"])


def _resolve_stage_concurrency(
    stage: Mapping[str, Any],
    conditions: Mapping[str, bool],
    env: Mapping[str, str],
) -> tuple[str, int] | None:
    spec = stage.get("concurrency")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid concurrency spec on {stage['key']!r}: {spec!r}")
    unknown = set(spec) - {"group", "limit", "when", "unless"}
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in concurrency of {stage['key']!r}"
        )
    if not _eval_conditions(spec, conditions):
        return None
    group_tpl = spec.get("group")
    limit = spec.get("limit")
    if not isinstance(group_tpl, str) or not group_tpl:
        raise ValueError(
            f"Invalid concurrency group on {stage['key']!r}: {group_tpl!r}"
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"Invalid concurrency limit on {stage['key']!r}: {limit!r}")
    try:
        group = group_tpl.format(author=_author_key(env))
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {exc} in concurrency group {group_tpl!r}"
        ) from exc
    return group, limit


def _prepare_leaf_step(
    step: dict[str, Any],
    *,
    file_env: dict[str, Any],
    nightly_suite: bool,
    omni_skip: str,
) -> None:
    if "wait" in step:
        raise ValueError(f"Unexpected wait step in pipeline: {step!r}")
    if "group" in step:
        raise ValueError(f"Nested group steps are not allowed: {step!r}")
    if file_env:
        step["env"] = {**file_env, **(step.get("env") or {})}
    if nightly_suite and isinstance(step.get("if"), str):
        step["if"] = step["if"].replace('build.env("NIGHTLY")', '"1"')
    if step.get("skip") == "${OMNI_IMAGE_SKIP:-}":
        if omni_skip:
            step["skip"] = omni_skip
        else:
            del step["skip"]


def _default_priority(env: Mapping[str, str], conditions: Mapping[str, bool]) -> int:
    if "JOB_PRIORITY" in env and env["JOB_PRIORITY"]:
        return int(env["JOB_PRIORITY"])
    if conditions["nightly"]:
        return 1
    if conditions["oncall-fix"]:
        return 12
    if conditions["pr"] or conditions["ci-run"]:
        return 6
    if env.get("BUILDKITE_BRANCH") == "main":
        return 11
    return 2


def render_pipeline(
    repo_root: pathlib.Path = _REPO_ROOT,
    env: Mapping[str, str] | None = None,
    stages_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the Buildkite pipeline dictionary from `stages.yml` and `env`."""
    env = os.environ if env is None else env
    conditions, pr_labels = _build_conditions(env)
    is_pr = conditions["pr"]

    integration_slug = env.get(
        "VLLM_INTEGRATION_PIPELINE_SLUG", "vllm-torchtpu-integration"
    )
    is_integration = env.get("BUILDKITE_PIPELINE_SLUG", "") == integration_slug
    full_ci = is_pr and "run-full-ci" in pr_labels
    nightly_suite = conditions["nightly"] or full_ci
    skip_tpu = (
        env.get("SKIP_TPU_CI") == "1" or (is_pr and "skip-tpu-ci" in pr_labels)
    ) and not conditions["ci-run"]

    if skip_tpu:
        default_lanes: tuple[str, ...] = ("tests",)
    elif nightly_suite or is_integration or (is_pr and "run-perf-eval" in pr_labels):
        default_lanes = ALL_LANES
    else:
        default_lanes = ("verify", "integration", "tests")

    ci_lanes_raw = env.get("CI_LANES", "").strip()
    if ci_lanes_raw:
        lanes = tuple(ci_lanes_raw.split())
        for lane in lanes:
            if lane not in ALL_LANES:
                raise ValueError(
                    f"CI_LANES names '{lane}'; lanes are: {' '.join(ALL_LANES)}"
                )
    else:
        lanes = default_lanes

    if "tests" not in lanes:
        omni_skip = "Only the tests lane uses the -omni image; this build has none."
    elif skip_tpu:
        omni_skip = "Only TPU steps use the -omni image; skip-tpu-ci skips them."
    else:
        omni_skip = env.get("OMNI_IMAGE_SKIP", "")

    if stages_doc is None:
        stages_path = repo_root / ".buildkite" / "stages.yml"
        stages_doc = yaml.safe_load(stages_path.read_text())

    stage_keys: dict[str, list[str]] = {}
    rendered_steps: list[dict[str, Any]] = []
    dynamic_keys: set[str] = set()

    for stage in stages_doc.get("stages") or []:
        stage_key = stage["key"]
        gate = _resolve_stage_concurrency(stage, conditions, env)
        if not _eval_conditions(stage, conditions) or (
            skip_tpu and stage_key in ("tpu", "report", "promote")
        ):
            stage_keys[stage_key] = []
            continue

        needed_keys: list[str] = []
        for need in stage.get("needs") or []:
            if isinstance(need, str):
                target = need
            elif isinstance(need, dict) and isinstance(need.get("stage"), str):
                unknown = set(need) - {"stage", "when", "unless"}
                if unknown:
                    raise ValueError(
                        f"Unknown keys {sorted(unknown)} in needs of {stage_key!r}"
                    )
                if not _eval_conditions(need, conditions):
                    continue
                target = need["stage"]
            else:
                raise ValueError(f"Invalid needs entry in {stage_key!r}: {need!r}")
            if target not in stage_keys:
                raise ValueError(
                    f"Stage {stage_key!r} needs unknown or later stage {target!r}"
                )
            for key in stage_keys[target]:
                if key not in needed_keys:
                    needed_keys.append(key)

        stage_steps: list[dict[str, Any]] = []
        file_stage_keys: list[str] = []
        stage_dynamic_keys: list[str] = []
        step_deps = [f"{stage_key}_gate_open"] if gate else needed_keys

        for entry in stage.get("files") or []:
            if entry.get("pipeline") == "integration" and not is_integration:
                continue
            if entry.get("lane") and entry["lane"] not in lanes:
                continue
            if not _eval_conditions(entry, conditions):
                continue

            file_path = repo_root / entry["path"]
            if not file_path.is_file():
                raise ValueError(f"Missing pipeline file: {entry['path']}")
            raw_doc = yaml.safe_load(file_path.read_text()) or {}
            extra_top = {
                k
                for k in raw_doc
                if k not in ("steps", "env") and not k.startswith("x-")
            }
            if extra_top:
                raise ValueError(
                    f"Unsupported top-level keys {sorted(extra_top)} in {entry['path']}"
                )
            file_env = copy.deepcopy(raw_doc.get("env") or {})
            steps = copy.deepcopy(raw_doc.get("steps") or [])

            for step in steps:
                if not isinstance(step, dict) or "wait" in step:
                    raise ValueError(
                        f"Wait or non-mapping step in {entry['path']}: {step!r}"
                    )
                top_key = step.get("key")
                if not isinstance(top_key, str) or not top_key:
                    raise ValueError(
                        f"Top-level step/group in {entry['path']} lacks a key: {step!r}"
                    )
                if "group" in step:
                    if nightly_suite and isinstance(step.get("if"), str):
                        step["if"] = step["if"].replace('build.env("NIGHTLY")', '"1"')
                    for inner in step.get("steps") or []:
                        if not isinstance(inner, dict):
                            raise ValueError(
                                f"Invalid step inside group {top_key!r}: {inner!r}"
                            )
                        _prepare_leaf_step(
                            inner,
                            file_env=file_env,
                            nightly_suite=nightly_suite,
                            omni_skip=omni_skip,
                        )
                else:
                    _prepare_leaf_step(
                        step,
                        file_env=file_env,
                        nightly_suite=nightly_suite,
                        omni_skip=omni_skip,
                    )

                if step_deps:
                    existing = _normalize_depends_on(step.get("depends_on"))
                    seen = {_dep_key(item) for item in existing}
                    step["depends_on"] = existing + [
                        k for k in step_deps if k not in seen
                    ]

                file_stage_keys.append(top_key)
                stage_steps.append(step)

            for dyn_key in entry.get("dynamic_keys") or []:
                if dyn_key not in stage_dynamic_keys:
                    stage_dynamic_keys.append(dyn_key)
                dynamic_keys.add(dyn_key)

        if gate and stage_steps:
            group_name, limit = gate
            open_key = f"{stage_key}_gate_open"
            close_key = f"{stage_key}_gate_close"
            open_step: dict[str, Any] = {
                "label": f":lock: Open {stage_key} concurrency gate",
                "key": open_key,
                "concurrency": limit,
                "concurrency_group": group_name,
                "agents": {"queue": "cpu"},
                "commands": ["true"],
            }
            if needed_keys:
                open_step["depends_on"] = list(needed_keys)

            close_deps = list(file_stage_keys)
            if nightly_suite or conditions["release-tag"]:
                close_deps.extend(k for k in stage_dynamic_keys if k not in close_deps)
            close_step: dict[str, Any] = {
                "label": f":unlock: Close {stage_key} concurrency gate",
                "key": close_key,
                "concurrency": limit,
                "concurrency_group": group_name,
                "depends_on": close_deps,
                "allow_dependency_failure": True,
                "agents": {"queue": "cpu"},
                "commands": ["true"],
            }
            rendered_steps.extend([open_step, *stage_steps, close_step])
            stage_keys[stage_key] = [*file_stage_keys, close_key]
        else:
            rendered_steps.extend(stage_steps)
            stage_keys[stage_key] = file_stage_keys

    known_keys = set(dynamic_keys)
    all_items: list[dict[str, Any]] = []
    for step in rendered_steps:
        all_items.append(step)
        if "group" in step:
            all_items.extend(step.get("steps") or [])

    for item in all_items:
        key = item.get("key")
        if key:
            if key in known_keys:
                raise ValueError(f"Duplicate step key {key!r}")
            known_keys.add(key)

    for item in all_items:
        label = item.get("key") or item.get("label") or item.get("group")
        for dep in _normalize_depends_on(item.get("depends_on")):
            dep_name = _dep_key(dep)
            if dep_name not in known_keys:
                raise ValueError(f"Step {label!r} depends_on unknown key {dep_name!r}")

    if not conditions["ci-run"]:
        concurrency_group = env.get("PR_TPU_CONCURRENCY_GROUP", "")
        if not concurrency_group and is_pr:
            author = env.get("BUILDKITE_BUILD_AUTHOR_EMAIL") or env.get(
                "BUILDKITE_BUILD_CREATOR_EMAIL", ""
            )
            if author:
                concurrency_group = f"tpu-pr/{author.lower()}"
        if concurrency_group:
            limit = int(env.get("PR_TPU_JOBS_PER_AUTHOR", "40"))
            cap_pr_tpu_concurrency.cap(rendered_steps, concurrency_group, limit)

    return {
        "priority": _default_priority(env, conditions),
        "steps": rendered_steps,
    }


def dump_pipeline(doc: dict[str, Any]) -> str:
    """Serialize a rendered pipeline dict as clean YAML without aliases."""
    return yaml.dump(doc, Dumper=_NoAliasDumper, sort_keys=False)


def main() -> None:
    try:
        doc = render_pipeline()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    sys.stdout.write(dump_pipeline(doc))


if __name__ == "__main__":
    main()
