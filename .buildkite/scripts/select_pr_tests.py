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
"""Diff-aware PR test selection and YAML step filter for Buildkite CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml as _pyyaml
except ImportError:
    _pyyaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CI_PATHS_REL = ".buildkite/ci_paths.yaml"
DEFAULT_TEST_PATTERN = "tests/**/{test_*,*_test}.py"
VALID_MODES = ("off", "shadow", "phase1", "phase2")


def _parse_yaml_scalar(raw: str) -> Any:
    val = raw.strip()
    if val == "[]":
        return []
    if val == "{}":
        return {}
    if val in {"true", "True"}:
        return True
    if val in {"false", "False"}:
        return False
    if val in {"null", "None", "~"}:
        return None
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    return val


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif (
            ch == "#"
            and not in_single
            and not in_double
            and (idx == 0 or line[idx - 1].isspace())
        ):
            return line[:idx].rstrip()
    return line.rstrip()


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Pure-stdlib YAML subset parser for ci_paths.yaml when PyYAML is absent."""
    tokens: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        cleaned = _strip_yaml_comment(raw_line)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        tokens.append((indent, cleaned.strip()))

    def parse_block(index: int, base_indent: int) -> tuple[Any, int]:
        if index >= len(tokens):
            return {}, index
        first_indent, first_content = tokens[index]
        if first_indent < base_indent:
            return {}, index

        if first_content.startswith("- "):
            items: list[Any] = []
            while index < len(tokens):
                ind, content = tokens[index]
                if ind < base_indent or not content.startswith("- "):
                    break
                if ind > base_indent:
                    break
                item_body = content[2:].strip()
                index += 1
                if ":" in item_body and not (
                    item_body.startswith('"') or item_body.startswith("'")
                ):
                    k, v_raw = item_body.split(":", 1)
                    item_dict: dict[str, Any] = {}
                    key = _parse_yaml_scalar(k)
                    if v_raw.strip():
                        item_dict[key] = _parse_yaml_scalar(v_raw)
                    else:
                        sub_val, index = parse_block(index, ind + 2)
                        item_dict[key] = sub_val
                    while index < len(tokens) and tokens[index][0] >= ind + 2:
                        sub_ind, sub_content = tokens[index]
                        if sub_ind != ind + 2 or sub_content.startswith("- "):
                            break
                        sk, sv_raw = sub_content.split(":", 1)
                        skey = _parse_yaml_scalar(sk)
                        index += 1
                        if sv_raw.strip():
                            item_dict[skey] = _parse_yaml_scalar(sv_raw)
                        else:
                            sub_val, index = parse_block(index, sub_ind + 2)
                            item_dict[skey] = sub_val
                    items.append(item_dict)
                else:
                    items.append(_parse_yaml_scalar(item_body))
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(tokens):
            ind, content = tokens[index]
            if ind < base_indent or content.startswith("- "):
                break
            if ind > base_indent:
                break
            k, v_raw = content.split(":", 1)
            key = str(_parse_yaml_scalar(k))
            index += 1
            if v_raw.strip():
                mapping[key] = _parse_yaml_scalar(v_raw)
            else:
                if index < len(tokens) and tokens[index][0] > ind:
                    sub_val, index = parse_block(index, tokens[index][0])
                    mapping[key] = sub_val
                else:
                    mapping[key] = {}
        return mapping, index

    result, _ = parse_block(0, 0)
    return result if isinstance(result, dict) else {}


def load_yaml_text(text: str) -> dict[str, Any]:
    """Loads a YAML mapping using PyYAML if available, else the stdlib parser."""
    if _pyyaml is not None:
        loaded = _pyyaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    return _parse_simple_yaml(text)


@dataclass(frozen=True)
class ComponentRule:
    """Declarative test selection and import boundary rule for a component."""

    component: str
    phase: str
    reason: str
    config_path: str
    path_prefixes: tuple[str, ...]
    module_prefixes: tuple[str, ...] = ()
    allowed_test_importers: tuple[str, ...] = ()
    allowed_external_importers: tuple[str, ...] = ()
    build_docker_omni: bool = False
    test_steps: tuple[str, ...] = ()
    integration_steps: tuple[str, ...] = ()
    full_general_tests: bool = False
    full_multichip_tests: bool = False
    general_test_pattern: str = ""
    general_test_files: tuple[str, ...] = ()
    general_test_parallelism: int | None = None


@dataclass(frozen=True)
class CiPathsConfig:
    """Parsed declarative configuration from .buildkite/ci_paths.yaml + components."""

    default_policy: str
    component_config_paths: tuple[str, ...]
    all_pr_test_steps: tuple[str, ...]
    all_pr_integration_steps: tuple[str, ...]
    full_ci_prefixes: tuple[str, ...]
    full_ci_exact: frozenset[str]
    skip_all_suffixes: tuple[str, ...]
    skip_all_exact: frozenset[str]
    skip_all_prefixes: tuple[str, ...]
    non_pr_buildkite_exact: frozenset[str]
    nightly_only_integration_exact: frozenset[str]
    benchmark_prefixes: tuple[str, ...]
    cpu_only_exact: frozenset[str]
    cpu_only_prefixes: tuple[str, ...]
    cpu_tested_benchmark_files: frozenset[str]
    integration_file_to_steps: dict[str, set[str]]
    component_rules: tuple[ComponentRule, ...]


def _parse_component_rule(data: dict[str, Any], config_path: str) -> ComponentRule:
    return ComponentRule(
        component=str(data["component"]),
        phase=str(data.get("phase", "phase2")),
        reason=str(data.get("reason", f"{data['component']} changes")),
        config_path=config_path,
        path_prefixes=tuple(data.get("path_prefixes", ())),
        module_prefixes=tuple(data.get("module_prefixes", ())),
        allowed_test_importers=tuple(data.get("allowed_test_importers", ())),
        allowed_external_importers=tuple(data.get("allowed_external_importers", ())),
        build_docker_omni=bool(data.get("build_docker_omni", False)),
        test_steps=tuple(data.get("test_steps", ())),
        integration_steps=tuple(data.get("integration_steps", ())),
        full_general_tests=bool(data.get("full_general_tests", False)),
        full_multichip_tests=bool(data.get("full_multichip_tests", False)),
        general_test_pattern=str(data.get("general_test_pattern", "")),
        general_test_files=tuple(data.get("general_test_files", ())),
        general_test_parallelism=(
            int(data["general_test_parallelism"])
            if data.get("general_test_parallelism") is not None
            else None
        ),
    )


def load_ci_paths_config(repo_root: Path = REPO_ROOT) -> CiPathsConfig:
    """Loads .buildkite/ci_paths.yaml and all per-component ci_paths.yaml files."""
    global_path = repo_root / GLOBAL_CI_PATHS_REL
    if not global_path.is_file():
        return CiPathsConfig(
            default_policy="full_ci",
            component_config_paths=(),
            all_pr_test_steps=(),
            all_pr_integration_steps=(),
            full_ci_prefixes=(),
            full_ci_exact=frozenset(),
            skip_all_suffixes=(),
            skip_all_exact=frozenset(),
            skip_all_prefixes=(),
            non_pr_buildkite_exact=frozenset(),
            nightly_only_integration_exact=frozenset(),
            benchmark_prefixes=(),
            cpu_only_exact=frozenset(),
            cpu_only_prefixes=(),
            cpu_tested_benchmark_files=frozenset(),
            integration_file_to_steps={},
            component_rules=(),
        )
    raw = load_yaml_text(global_path.read_text(encoding="utf-8"))

    comp_paths = tuple(raw.get("component_configs", ()))
    rules: list[ComponentRule] = []
    for rel_path in comp_paths:
        comp_file = repo_root / rel_path
        if comp_file.is_file():
            comp_raw = load_yaml_text(comp_file.read_text(encoding="utf-8"))
            rules.append(_parse_component_rule(comp_raw, rel_path))

    for scoped_raw in raw.get("scoped_rules", ()):
        rules.append(_parse_component_rule(scoped_raw, GLOBAL_CI_PATHS_REL))

    full_ci = raw.get("full_ci_paths", {})
    no_tpu = raw.get("no_tpu", {})
    skip_all = no_tpu.get("skip_all_ci", {})
    cpu_only = no_tpu.get("cpu_only", {})
    integ_map = {
        str(k): set(v) for k, v in (raw.get("integration_files", {}) or {}).items()
    }

    return CiPathsConfig(
        default_policy=str(raw.get("default_policy", "full_ci")),
        component_config_paths=comp_paths,
        all_pr_test_steps=tuple(raw.get("all_pr_test_steps", ())),
        all_pr_integration_steps=tuple(raw.get("all_pr_integration_steps", ())),
        full_ci_prefixes=tuple(full_ci.get("prefixes", ())),
        full_ci_exact=frozenset(full_ci.get("exact", ())),
        skip_all_suffixes=tuple(skip_all.get("suffixes", ())),
        skip_all_exact=frozenset(skip_all.get("exact", ())),
        skip_all_prefixes=tuple(skip_all.get("prefixes", ())),
        non_pr_buildkite_exact=frozenset(skip_all.get("non_pr_buildkite_exact", ())),
        nightly_only_integration_exact=frozenset(
            skip_all.get("nightly_only_integration_exact", ())
        ),
        benchmark_prefixes=tuple(skip_all.get("benchmark_prefixes", ())),
        cpu_only_exact=frozenset(cpu_only.get("exact", ())),
        cpu_only_prefixes=tuple(cpu_only.get("prefixes", ())),
        cpu_tested_benchmark_files=frozenset(
            cpu_only.get("cpu_tested_benchmark_files", ())
        ),
        integration_file_to_steps=integ_map,
        component_rules=tuple(rules),
    )


_DEFAULT_CONFIG = load_ci_paths_config(REPO_ROOT)

ALL_PR_TEST_STEPS: tuple[str, ...] = _DEFAULT_CONFIG.all_pr_test_steps
ALL_PR_INTEGRATION_STEPS: tuple[str, ...] = _DEFAULT_CONFIG.all_pr_integration_steps
INTEGRATION_FILE_TO_STEPS: dict[str, set[str]] = (
    _DEFAULT_CONFIG.integration_file_to_steps
)
CPU_TESTED_BENCHMARK_FILES: set[str] = set(_DEFAULT_CONFIG.cpu_tested_benchmark_files)
NIGHTLY_ONLY_INTEGRATION_EXACT: set[str] = set(
    _DEFAULT_CONFIG.nightly_only_integration_exact
)
NON_PR_BUILDKITE_EXACT: set[str] = set(_DEFAULT_CONFIG.non_pr_buildkite_exact)
COMPONENT_RULES: tuple[ComponentRule, ...] = _DEFAULT_CONFIG.component_rules


@dataclass
class SelectionPlan:
    """Computed lane and step selection for a PR diff."""

    skip_all_ci: bool = False
    skip_all_reason: str = ""
    lanes: tuple[str, ...] = ("verify", "integration", "tests")
    omni_image_skip: str = ""
    skip_steps: dict[str, str] = field(default_factory=dict)
    general_test_pattern: str = DEFAULT_TEST_PATTERN
    general_test_parallelism: int = 3
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN
    annotation: str = ""


def _is_doc_or_non_pr_file(path: str, config: CiPathsConfig = _DEFAULT_CONFIG) -> bool:
    if path.endswith(config.skip_all_suffixes):
        return True
    if path in config.skip_all_exact:
        return True
    if path.startswith(config.skip_all_prefixes):
        return True
    if path in config.non_pr_buildkite_exact:
        return True
    if path in config.nightly_only_integration_exact:
        return True
    return (
        path.startswith(config.benchmark_prefixes)
        and path not in config.cpu_tested_benchmark_files
    )


def _is_cpu_only_file(path: str, config: CiPathsConfig = _DEFAULT_CONFIG) -> bool:
    if path in config.cpu_only_exact:
        return True
    if path.startswith(".buildkite/scripts/") and path.endswith(".py"):
        return True
    if path in config.cpu_tested_benchmark_files:
        return True
    return path.startswith(config.cpu_only_prefixes)


def _match_component_rule(
    path: str,
    effective_mode: str,
    config: CiPathsConfig = _DEFAULT_CONFIG,
) -> ComponentRule | None:
    """Returns the matching ComponentRule if active in effective_mode."""
    allowed_phases = {"phase1"} if effective_mode == "phase1" else {"phase1", "phase2"}
    for rule in config.component_rules:
        if rule.phase not in allowed_phases:
            continue
        if path.startswith(rule.path_prefixes):
            return rule
    return None


def _classify_unit_test_file(
    rel_path: str, repo_root: Path = REPO_ROOT
) -> tuple[bool, bool]:
    """Returns (has_tpu_general, has_multichip) for a unit test file."""
    full_path = repo_root / rel_path
    if not full_path.is_file():
        return (False, False)
    try:
        content = full_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return (True, True)
    has_tpu = bool(re.search(r"\btpu_test\b", content))
    has_multichip = bool(re.search(r"\bmultichip\b", content))
    return (has_tpu, has_multichip)


def _format_brace_pattern(files: list[str]) -> str:
    unique = sorted(set(files))
    if not unique:
        return DEFAULT_TEST_PATTERN
    if len(unique) == 1:
        return unique[0]
    return "{" + ",".join(unique) + "}"


def _has_full_ci_override(commit_message: str, ci_extras: str) -> bool:
    if re.search(r"\[(?:ci[ _-]?full|full[ _-]?ci)\]", commit_message, re.IGNORECASE):
        return True
    tokens = {t.strip().lower() for t in re.split(r"[,;\s]+", ci_extras) if t.strip()}
    return bool(tokens & {"full", "all", "full-ci"})


def _finalize_plan(
    run_test_steps: set[str],
    run_integration_steps: set[str],
    reason: str,
    general_test_pattern: str = DEFAULT_TEST_PATTERN,
    general_test_parallelism: int = 3,
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN,
    config: CiPathsConfig = _DEFAULT_CONFIG,
) -> SelectionPlan:
    lanes = ["verify"]
    if run_integration_steps:
        lanes.append("integration")
    if run_test_steps:
        lanes.append("tests")

    skip_steps: dict[str, str] = {}
    if run_test_steps:
        for step in config.all_pr_test_steps:
            if step not in run_test_steps:
                skip_steps[step] = reason[:70]
    if run_integration_steps:
        for step in config.all_pr_integration_steps:
            if step not in run_integration_steps:
                skip_steps[step] = reason[:70]

    omni_image_skip = ""
    if "unit_tests_omni_pr" not in run_test_steps:
        omni_image_skip = "No Omni changes in this PR."

    skipped_count = len(skip_steps)
    dropped_lanes = [lane for lane in ("integration", "tests") if lane not in lanes]
    summary_parts = [f"lanes: `{' '.join(lanes)}`"]
    if dropped_lanes:
        summary_parts.append(f"skipped lanes: `{', '.join(dropped_lanes)}`")
    if skipped_count:
        summary_parts.append(f"skipped {skipped_count} step(s)")
    if general_test_pattern != DEFAULT_TEST_PATTERN:
        summary_parts.append(
            f"general pattern: `{general_test_pattern}` "
            f"(parallelism={general_test_parallelism})"
        )
    if multichip_test_pattern != DEFAULT_TEST_PATTERN:
        summary_parts.append(f"multichip pattern: `{multichip_test_pattern}`")

    annotation = ""
    if (
        dropped_lanes
        or set(skip_steps) - {"unit_tests_omni_pr"}
        or general_test_pattern != DEFAULT_TEST_PATTERN
        or multichip_test_pattern != DEFAULT_TEST_PATTERN
    ):
        annotation = (
            f"Diff-aware PR test selection ({reason}): "
            + "; ".join(summary_parts)
            + "."
        )
    return SelectionPlan(
        skip_all_ci=False,
        lanes=tuple(lanes),
        omni_image_skip=omni_image_skip,
        skip_steps=skip_steps,
        general_test_pattern=general_test_pattern,
        general_test_parallelism=general_test_parallelism,
        multichip_test_pattern=multichip_test_pattern,
        annotation=annotation,
    )


def select_for_files(
    changed_files: list[str],
    repo_root: Path = REPO_ROOT,
    mode: str | None = None,
    commit_message: str | None = None,
    ci_extras: str | None = None,
) -> SelectionPlan:
    """Computes the PR test selection plan for a list of changed file paths."""
    raw_mode = (
        (mode if mode is not None else os.environ.get("CI_DIFF_AWARE_MODE", "phase2"))
        .strip()
        .lower()
    )
    if raw_mode not in VALID_MODES:
        raw_mode = "phase2"

    msg = (
        commit_message
        if commit_message is not None
        else os.environ.get("BUILDKITE_MESSAGE", "")
    )
    extras = ci_extras if ci_extras is not None else os.environ.get("CI_EXTRAS", "")

    if raw_mode == "off" or _has_full_ci_override(msg, extras):
        return SelectionPlan()

    effective_mode = "phase2" if raw_mode == "shadow" else raw_mode
    config = (
        _DEFAULT_CONFIG if repo_root == REPO_ROOT else load_ci_paths_config(repo_root)
    )

    files = [f.strip() for f in changed_files if f.strip()]
    if not files:
        return SelectionPlan()

    active_files = [f for f in files if not _is_doc_or_non_pr_file(f, config)]
    if not active_files:
        plan = SelectionPlan(
            skip_all_ci=True,
            skip_all_reason=(
                "Only docs, benchmarks, or non-PR workflow/config files changed."
            ),
            lanes=(),
        )
        if raw_mode == "shadow":
            return SelectionPlan(
                annotation=(
                    "Shadow diff-aware PR test selection "
                    f"(would-skip; running full CI): {plan.skip_all_reason}"
                )
            )
        return plan

    for f in active_files:
        if f in config.full_ci_exact or f.startswith(config.full_ci_prefixes):
            return SelectionPlan()

    run_test_steps: set[str] = set()
    run_integration_steps: set[str] = set()
    general_test_files: list[str] = []
    multichip_test_files: list[str] = []
    full_general_tests = False
    full_multichip_tests = False
    reasons: list[str] = []

    for f in active_files:
        if _is_cpu_only_file(f, config):
            run_test_steps.update({"typecheck_full", "unit_tests_cpu_pr"})
            reasons.append("CPU/tooling changes")
            continue

        if f in config.integration_file_to_steps:
            run_integration_steps.update(config.integration_file_to_steps[f])
            if f.endswith(".py"):
                run_test_steps.update({"typecheck_full", "unit_tests_cpu_pr"})
            reasons.append("integration script/manifest changes")
            continue

        if f == ".buildkite/pipeline_integration.yml":
            run_integration_steps.update(config.all_pr_integration_steps)
            reasons.append("integration pipeline YAML changes")
            continue

        if f == ".buildkite/pipeline_tests.yml":
            run_test_steps.update(config.all_pr_test_steps)
            full_general_tests = True
            full_multichip_tests = True
            reasons.append("tests pipeline YAML changes")
            continue

        comp_rule = _match_component_rule(f, effective_mode, config)
        if comp_rule is not None:
            run_test_steps.update(comp_rule.test_steps)
            run_integration_steps.update(comp_rule.integration_steps)
            if comp_rule.full_general_tests:
                full_general_tests = True
            if comp_rule.full_multichip_tests:
                full_multichip_tests = True
            if comp_rule.general_test_pattern:
                general_test_files.append(comp_rule.general_test_pattern)
            if comp_rule.general_test_files:
                general_test_files.extend(comp_rule.general_test_files)
            reasons.append(comp_rule.reason)
            continue

        if f.startswith("tests/"):
            run_test_steps.update({"typecheck_full", "unit_tests_cpu_pr"})
            if f == "tests/entrypoints/llm/test_speculative_decoding.py":
                run_test_steps.update(
                    {
                        "unit_tests_spec_decode_pr",
                        "unit_tests_spec_decode_multichip_pr",
                    }
                )
                reasons.append("spec-decode unit test changes")
            elif f.startswith("tests/spec_decode/"):
                run_test_steps.update(
                    {
                        "unit_tests_spec_decode_pr",
                        "unit_tests_spec_decode_multichip_pr",
                    }
                )
                has_gen, has_multi = _classify_unit_test_file(f, repo_root)
                if has_gen:
                    run_test_steps.add("unit_tests_general_pr")
                    general_test_files.append(f)
                if has_multi:
                    run_test_steps.add("unit_tests_multichip_pr")
                    multichip_test_files.append(f)
                reasons.append("spec-decode unit test changes")
            elif not (
                Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py")
            ):
                run_test_steps.update(
                    {
                        "unit_tests_general_pr",
                        "unit_tests_multichip_pr",
                        "unit_tests_spec_decode_pr",
                        "unit_tests_spec_decode_multichip_pr",
                    }
                )
                full_general_tests = True
                full_multichip_tests = True
                reasons.append("shared test fixture changes")
            elif not (repo_root / f).is_file():
                reasons.append("deleted unit test file")
            else:
                has_gen, has_multi = _classify_unit_test_file(f, repo_root)
                if not has_gen and not has_multi:
                    reasons.append("CPU unit test changes")
                else:
                    if has_gen:
                        run_test_steps.add("unit_tests_general_pr")
                        general_test_files.append(f)
                    if has_multi:
                        run_test_steps.add("unit_tests_multichip_pr")
                        multichip_test_files.append(f)
                    reasons.append("TPU unit test changes")
            continue

        run_test_steps.update(
            {
                "typecheck_full",
                "unit_tests_cpu_pr",
                "unit_tests_general_pr",
                "unit_tests_multichip_pr",
                "unit_tests_spec_decode_pr",
                "unit_tests_spec_decode_multichip_pr",
            }
        )
        run_integration_steps.update(config.all_pr_integration_steps)
        full_general_tests = True
        full_multichip_tests = True
        reasons.append("core runtime changes")

    general_pattern = DEFAULT_TEST_PATTERN
    general_parallelism = 3
    if (
        "unit_tests_general_pr" in run_test_steps
        and not full_general_tests
        and general_test_files
    ):
        general_pattern = _format_brace_pattern(general_test_files)
        general_parallelism = 1 if len(set(general_test_files)) <= 4 else 3

    multichip_pattern = DEFAULT_TEST_PATTERN
    if (
        "unit_tests_multichip_pr" in run_test_steps
        and not full_multichip_tests
        and multichip_test_files
    ):
        multichip_pattern = _format_brace_pattern(multichip_test_files)

    unique_reasons = list(dict.fromkeys(reasons))
    reason_str = ", ".join(unique_reasons[:2]) if unique_reasons else "PR diff"
    if len(unique_reasons) > 2:
        reason_str += f" (+{len(unique_reasons) - 2} more)"

    plan = _finalize_plan(
        run_test_steps=run_test_steps,
        run_integration_steps=run_integration_steps,
        reason=f"Skipped: only {reason_str}."[:70],
        general_test_pattern=general_pattern,
        general_test_parallelism=general_parallelism,
        multichip_test_pattern=multichip_pattern,
        config=config,
    )
    if raw_mode == "shadow":
        shadow_note = ""
        if plan.annotation:
            shadow_note = plan.annotation.replace(
                "Diff-aware PR test selection",
                "Shadow diff-aware PR test selection (would-skip; running full CI)",
                1,
            )
        return SelectionPlan(annotation=shadow_note)
    return plan


def filter_yaml_content(
    yaml_text: str,
    skip_steps: dict[str, str],
    general_test_pattern: str = DEFAULT_TEST_PATTERN,
    general_test_parallelism: int = 3,
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN,
) -> str:
    """Injects `skip:` and narrowed `test-file-pattern:` into a pipeline YAML."""
    if (
        not skip_steps
        and general_test_pattern == DEFAULT_TEST_PATTERN
        and general_test_parallelism == 3
        and multichip_test_pattern == DEFAULT_TEST_PATTERN
    ):
        return yaml_text

    step_start_re = re.compile(r"^[ \t]*-[ \t]+(?:label|group|wait|block):")
    key_re = re.compile(r'^(?P<indent>[ \t]*)key:[ \t]*"(?P<key>[^"]+)"[ \t]*$')
    parallelism_re = re.compile(r"^(?P<indent>[ \t]*)parallelism:[ \t]*\d+[ \t]*$")
    plugin_defaults_re = re.compile(
        r"^(?P<indent>[ \t]*)<<:[ \t]*\*tests_plugin_defaults[ \t]*$"
    )

    current_key: str | None = None
    out_lines: list[str] = []

    for line in yaml_text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        eol = line[len(stripped) :] or "\n"

        if step_start_re.match(stripped):
            current_key = None

        m_key = key_re.match(stripped)
        if m_key:
            current_key = m_key.group("key")
            out_lines.append(line)
            if current_key in skip_steps:
                indent = m_key.group("indent")
                reason = skip_steps[current_key][:70]
                out_lines.append(f"{indent}skip: {json.dumps(reason)}{eol}")
            continue

        if current_key == "unit_tests_general_pr" and general_test_parallelism != 3:
            m_par = parallelism_re.match(stripped)
            if m_par:
                indent = m_par.group("indent")
                out_lines.append(
                    f"{indent}parallelism: {general_test_parallelism}{eol}"
                )
                continue

        if current_key in {"unit_tests_general_pr", "unit_tests_multichip_pr"}:
            m_plug = plugin_defaults_re.match(stripped)
            if m_plug:
                out_lines.append(line)
                pattern = (
                    general_test_pattern
                    if current_key == "unit_tests_general_pr"
                    else multichip_test_pattern
                )
                if pattern and pattern != DEFAULT_TEST_PATTERN:
                    indent = m_plug.group("indent")
                    out_lines.append(
                        f"{indent}test-file-pattern: {json.dumps(pattern)}{eol}"
                    )
                continue

        out_lines.append(line)

    return "".join(out_lines)


def filter_json_content(
    doc: dict[str, Any],
    skip_steps: dict[str, str],
    general_test_pattern: str = DEFAULT_TEST_PATTERN,
    general_test_parallelism: int = 3,
    multichip_test_pattern: str = DEFAULT_TEST_PATTERN,
) -> dict[str, Any]:
    """Mutates a Buildkite `pipeline upload --dry-run --format json` document."""

    def _visit_steps(steps: list[dict[str, Any]]) -> None:
        for step in steps:
            if not isinstance(step, dict):
                continue
            nested = step.get("steps")
            if isinstance(nested, list):
                _visit_steps(nested)
            key = step.get("key")
            if isinstance(key, str) and key in skip_steps:
                step["skip"] = skip_steps[key][:70]
            if key == "unit_tests_general_pr" and general_test_parallelism != 3:
                step["parallelism"] = general_test_parallelism
            if key in {"unit_tests_general_pr", "unit_tests_multichip_pr"}:
                pattern = (
                    general_test_pattern
                    if key == "unit_tests_general_pr"
                    else multichip_test_pattern
                )
                if pattern and pattern != DEFAULT_TEST_PATTERN:
                    plugins = step.get("plugins")
                    if isinstance(plugins, list):
                        for plugin in plugins:
                            if isinstance(plugin, dict):
                                for p_name, p_cfg in plugin.items():
                                    if p_name.startswith("tests#") and isinstance(
                                        p_cfg, dict
                                    ):
                                        p_cfg["test-file-pattern"] = pattern

    top_steps = doc.get("steps")
    if isinstance(top_steps, list):
        _visit_steps(top_steps)
    return doc


def format_shell(plan: SelectionPlan) -> str:
    """Formats SelectionPlan as bash assignments for `eval` in bootstrap.sh."""
    lines = [
        f"PR_SKIP_ALL_CI={shlex.quote('1' if plan.skip_all_ci else '0')}",
        f"PR_SKIP_ALL_REASON={shlex.quote(plan.skip_all_reason)}",
        f"PR_SELECTED_LANES={shlex.quote(' '.join(plan.lanes))}",
        f"OMNI_IMAGE_SKIP={shlex.quote(plan.omni_image_skip)}",
        f"PR_TEST_SELECTION_ANNOTATION={shlex.quote(plan.annotation)}",
        (
            "export PR_SKIP_STEPS_JSON="
            f"{shlex.quote(json.dumps(plan.skip_steps, sort_keys=True))}"
        ),
        (f"export GENERAL_TEST_FILE_PATTERN={shlex.quote(plan.general_test_pattern)}"),
        (
            "export GENERAL_TEST_PARALLELISM="
            f"{shlex.quote(str(plan.general_test_parallelism))}"
        ),
        (
            "export MULTICHIP_TEST_FILE_PATTERN="
            f"{shlex.quote(plan.multichip_test_pattern)}"
        ),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--shell",
        action="store_true",
        help="Emit shell variable assignments from changed files on stdin.",
    )
    group.add_argument(
        "--filter-yaml",
        metavar="YAML_FILE",
        help="Filter a Buildkite pipeline YAML file using exported env vars.",
    )
    group.add_argument(
        "--filter-json",
        action="store_true",
        help="Filter Buildkite pipeline upload --dry-run --format json on stdin.",
    )
    args = parser.parse_args()

    if args.filter_json:
        raw_json = sys.stdin.read()
        try:
            doc = json.loads(raw_json)
            raw_skip = os.environ.get("PR_SKIP_STEPS_JSON", "")
            skip_steps = json.loads(raw_skip) if raw_skip else {}
            general_pattern = os.environ.get(
                "GENERAL_TEST_FILE_PATTERN", DEFAULT_TEST_PATTERN
            )
            general_parallelism = int(os.environ.get("GENERAL_TEST_PARALLELISM", "3"))
            multichip_pattern = os.environ.get(
                "MULTICHIP_TEST_FILE_PATTERN", DEFAULT_TEST_PATTERN
            )
            filter_json_content(
                doc=doc,
                skip_steps=skip_steps,
                general_test_pattern=general_pattern,
                general_test_parallelism=general_parallelism,
                multichip_test_pattern=multichip_pattern,
            )
            json.dump(doc, sys.stdout)
            sys.stdout.write("\n")
        except Exception as exc:
            print(
                f"WARNING: select_pr_tests --filter-json failed open: {exc}",
                file=sys.stderr,
            )
            sys.stdout.write(raw_json)
        return 0

    if args.filter_yaml:
        yaml_path = Path(args.filter_yaml)
        yaml_text = yaml_path.read_text(encoding="utf-8")
        try:
            raw_skip = os.environ.get("PR_SKIP_STEPS_JSON", "")
            skip_steps = json.loads(raw_skip) if raw_skip else {}
            general_pattern = os.environ.get(
                "GENERAL_TEST_FILE_PATTERN", DEFAULT_TEST_PATTERN
            )
            general_parallelism = int(os.environ.get("GENERAL_TEST_PARALLELISM", "3"))
            multichip_pattern = os.environ.get(
                "MULTICHIP_TEST_FILE_PATTERN", DEFAULT_TEST_PATTERN
            )
            sys.stdout.write(
                filter_yaml_content(
                    yaml_text=yaml_text,
                    skip_steps=skip_steps,
                    general_test_pattern=general_pattern,
                    general_test_parallelism=general_parallelism,
                    multichip_test_pattern=multichip_pattern,
                )
            )
        except Exception as exc:
            print(
                f"WARNING: select_pr_tests --filter-yaml failed open: {exc}",
                file=sys.stderr,
            )
            sys.stdout.write(yaml_text)
        return 0

    try:
        changed_files = [line.strip() for line in sys.stdin if line.strip()]
        plan = select_for_files(changed_files)
    except Exception as exc:
        print(
            f"WARNING: select_pr_tests --shell failed open: {exc}",
            file=sys.stderr,
        )
        plan = SelectionPlan()
    print(format_shell(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
