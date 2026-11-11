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
"""Diff-aware PR test selection: which PR TPU steps a change needs.

The rules live in .buildkite/ci_paths.yaml and the per-folder files it lists.
bootstrap.py calls `select` on PR and main merge-queue builds, records the
result, and outside shadow mode marks the steps it does not need `skip:`.

To see what a change would run:

    python3 .buildkite/scripts/select_pr_tests.py [--mode MODE] FILE...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = ".buildkite/ci_paths.yaml"

# off: select nothing. shadow: record what would be skipped, skip nothing.
# phase1: skip using the built-in rules and phase1 folders. phase2: all rules.
MODES = ("off", "shadow", "phase1", "phase2")
DEFAULT_MODE = "shadow"
PHASES = ("phase1", "phase2")

# Buildkite rejects a pipeline whose skip reason is over 70 characters.
SKIP_REASON = "Diff-aware selection: this PR's changes do not need it"
OMNI_IMAGE_SKIP = "Diff-aware selection: this PR's changes need no Omni tests"
OMNI_TEST_STEP = "unit_tests_omni_pr"

# How the built-in unit-test rule maps a changed test file to the steps that
# run it; .buildkite/pipeline_tests.yml filters each step by these markers.
GENERAL_STEP = "unit_tests_general_pr"
MULTICHIP_STEP = "unit_tests_multichip_pr"
SPEC_DECODE_E2E = "tests/entrypoints/llm/test_speculative_decoding.py"
SPEC_DECODE_STEPS = frozenset(
    {"unit_tests_spec_decode_pr", "unit_tests_spec_decode_multichip_pr"}
)
_TEST_FILE_RE = re.compile(r"^(test_.*|.*_test)\.py$")
_TPU_MARKER_RE = re.compile(r"\btpu_test\b")
_MULTICHIP_MARKER_RE = re.compile(r"\bmultichip\b")


class ConfigError(ValueError):
    """A ci_paths file is malformed or names an unknown step."""


# ---------------------------------------------------------------------------
# Config parsing (stdlib only: bootstrap.py runs before any venv exists)
# ---------------------------------------------------------------------------


def _scalar(raw: str, line_no: int) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise ConfigError(f"line {line_no}: empty value")
    return value


def parse_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the YAML subset ci_paths files use: nested mappings, string lists.

    Raises `ConfigError` on anything else (flow style other than `[]`, list
    items that are mappings, tabs, odd indentation) rather than guess.
    """
    lines: list[tuple[int, int, str]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if raw.lstrip().startswith("#"):
            continue
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip():
            continue
        body = line.lstrip(" ")
        if body.startswith("\t"):
            raise ConfigError(f"line {line_no}: tabs are not allowed")
        lines.append((line_no, len(line) - len(body), body))
    if not lines:
        return {}
    value, end = _parse_block(lines, 0, lines[0][1])
    if end != len(lines):
        raise ConfigError(f"line {lines[end][0]}: unexpected indentation")
    if not isinstance(value, dict):
        raise ConfigError("the top level must be a mapping")
    return value


def _parse_block(
    lines: Sequence[tuple[int, int, str]], index: int, indent: int
) -> tuple[dict[str, Any] | list[str], int]:
    if lines[index][2].startswith("- "):
        items: list[str] = []
        while index < len(lines) and lines[index][1] == indent:
            line_no, _, body = lines[index]
            if not body.startswith("- "):
                raise ConfigError(f"line {line_no}: expected a list item")
            item = body[2:]
            if ": " in item or item.rstrip().endswith(":"):
                raise ConfigError(f"line {line_no}: list items must be strings")
            items.append(_scalar(item, line_no))
            index += 1
        return items, index

    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index][1] == indent:
        line_no, _, body = lines[index]
        key_raw, sep, rest = body.partition(":")
        if not sep or body.startswith("- "):
            raise ConfigError(f"line {line_no}: expected 'key: value'")
        key = _scalar(key_raw, line_no)
        if key in mapping:
            raise ConfigError(f"line {line_no}: duplicate key {key!r}")
        index += 1
        rest = rest.strip()
        if rest == "[]":
            mapping[key] = []
        elif rest:
            if rest[0] in "[{&*|>":
                raise ConfigError(f"line {line_no}: unsupported value {rest!r}")
            mapping[key] = _scalar(rest, line_no)
        elif index < len(lines) and lines[index][1] > indent:
            mapping[key], index = _parse_block(lines, index, lines[index][1])
        else:
            raise ConfigError(f"line {line_no}: {key!r} has no value")
    return mapping, index


@dataclass(frozen=True)
class Component:
    """A folder whose changes need only `run_steps` of the selectable steps."""

    name: str
    config_path: str
    phase: str
    paths: tuple[str, ...]
    run_steps: frozenset[str]
    # Checked by tests/scripts/test_select_pr_tests.py: only files under
    # `paths` or `allowed_importers` may import `modules`.
    modules: tuple[str, ...]
    allowed_importers: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    test_steps: tuple[str, ...]
    integration_steps: tuple[str, ...]
    cpu_only: tuple[str, ...]
    integration_scripts: Mapping[str, frozenset[str]]
    components: tuple[Component, ...]

    @property
    def selectable(self) -> frozenset[str]:
        return frozenset(self.test_steps) | frozenset(self.integration_steps)


def _str_list(data: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: {key!r} must be a list of strings")
    return tuple(value)


def _load_component(
    data: dict[str, Any], config_path: str, selectable: frozenset[str]
) -> Component:
    where = config_path
    unknown = set(data) - {
        "name",
        "phase",
        "paths",
        "run_steps",
        "modules",
        "allowed_importers",
    }
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    name, phase = data.get("name"), data.get("phase")
    if not isinstance(name, str):
        raise ConfigError(f"{where}: 'name' is required")
    if phase not in PHASES:
        raise ConfigError(f"{where}: 'phase' must be one of {PHASES}")
    paths = _str_list(data, "paths", where)
    if not paths:
        raise ConfigError(f"{where}: 'paths' must not be empty")
    run_steps = frozenset(_str_list(data, "run_steps", where))
    if bad := sorted(run_steps - selectable):
        raise ConfigError(f"{where}: run_steps not in selectable_steps: {bad}")
    return Component(
        name=name,
        config_path=config_path,
        phase=phase,
        paths=paths,
        run_steps=run_steps,
        modules=_str_list(data, "modules", where),
        allowed_importers=_str_list(data, "allowed_importers", where),
    )


def load_config(repo_root: pathlib.Path = REPO_ROOT) -> Config:
    """Load .buildkite/ci_paths.yaml and the component files it lists."""
    raw = parse_yaml_subset((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    steps = raw.get("selectable_steps")
    if not isinstance(steps, dict):
        raise ConfigError(f"{CONFIG_PATH}: 'selectable_steps' must be a mapping")
    test_steps = _str_list(steps, "tests", CONFIG_PATH)
    integration_steps = _str_list(steps, "integration", CONFIG_PATH)
    selectable = frozenset(test_steps) | frozenset(integration_steps)

    scripts_raw = raw.get("integration_scripts", {})
    if not isinstance(scripts_raw, dict):
        raise ConfigError(f"{CONFIG_PATH}: 'integration_scripts' must be a mapping")
    scripts = {
        script: frozenset(_str_list(scripts_raw, script, CONFIG_PATH))
        for script in scripts_raw
    }
    for script, script_steps in scripts.items():
        if bad := sorted(script_steps - frozenset(integration_steps)):
            raise ConfigError(f"{CONFIG_PATH}: {script} names unknown steps {bad}")

    components = tuple(
        _load_component(
            parse_yaml_subset((repo_root / rel).read_text(encoding="utf-8")),
            rel,
            selectable,
        )
        for rel in _str_list(raw, "component_configs", CONFIG_PATH)
    )
    return Config(
        test_steps=test_steps,
        integration_steps=integration_steps,
        cpu_only=_str_list(raw, "cpu_only", CONFIG_PATH),
        integration_scripts=scripts,
        components=components,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def parse_mode(raw: str | None) -> str:
    """Normalize CI_DIFF_AWARE_MODE; anything unrecognized means shadow."""
    mode = (raw or "").strip().lower()
    return mode if mode in MODES else DEFAULT_MODE


def matches(path: str, entry: str) -> bool:
    """True if `entry` is `path`, or a folder (trailing `/`) containing it."""
    return path.startswith(entry) if entry.endswith("/") else path == entry


def _unit_test_steps(path: str, repo_root: pathlib.Path) -> frozenset[str]:
    if path == SPEC_DECODE_E2E:
        return SPEC_DECODE_STEPS
    test_file = repo_root / path
    if not test_file.is_file():  # deleted: nothing left to run
        return frozenset()
    text = test_file.read_text(encoding="utf-8", errors="replace")
    steps = set()
    if _TPU_MARKER_RE.search(text):
        steps.add(GENERAL_STEP)
    if _MULTICHIP_MARKER_RE.search(text):
        steps.add(MULTICHIP_STEP)
    return frozenset(steps)  # neither marker: only the CPU unit tests run it


def classify(
    path: str, config: Config, repo_root: pathlib.Path = REPO_ROOT
) -> tuple[frozenset[str] | None, str, str]:
    """Return `(steps, rule, phase)` for one changed file.

    `steps` is None when the file needs the full build.
    """
    if path in config.integration_scripts:
        return config.integration_scripts[path], "integration script", "phase1"
    if any(matches(path, entry) for entry in config.cpu_only):
        return frozenset(), "CPU-only file", "phase1"
    for component in config.components:
        if any(matches(path, entry) for entry in component.paths):
            return component.run_steps, f"{component.name} folder", component.phase
    if path.startswith("tests/") and _TEST_FILE_RE.match(pathlib.PurePath(path).name):
        return _unit_test_steps(path, repo_root), "unit test file", "phase1"
    return None, "full build", "phase1"


@dataclass(frozen=True)
class Selection:
    """What a change needs, under every rule and under the active mode's."""

    mode: str
    selectable: frozenset[str]
    files: int
    rules: tuple[str, ...]
    # The first changed file that needs the full build, or "".
    full_build_file: str
    # Steps needed with every rule active (phase2); None means all of them.
    needed: frozenset[str] | None
    # Steps needed with only the rules `mode` turns on; None means all.
    mode_needed: frozenset[str] | None

    @property
    def would_skip(self) -> tuple[str, ...]:
        """Steps phase2 would skip: what shadow mode measures."""
        return (
            () if self.needed is None else tuple(sorted(self.selectable - self.needed))
        )

    @property
    def skipped(self) -> tuple[str, ...]:
        """Steps this build actually skips."""
        if self.mode not in PHASES or self.mode_needed is None:
            return ()
        return tuple(sorted(self.selectable - self.mode_needed))

    def skip_reasons(self) -> dict[str, str]:
        return {step: SKIP_REASON for step in self.skipped}

    @property
    def omni_image_skip(self) -> str:
        """`OMNI_IMAGE_SKIP` reason: only the Omni test step uses that image."""
        return OMNI_IMAGE_SKIP if OMNI_TEST_STEP in self.skipped else ""

    def record(self) -> dict[str, Any]:
        """The build meta-data entry shadow-mode audits read."""
        return {
            "mode": self.mode,
            "files": self.files,
            "rules": list(self.rules),
            "full_build_file": self.full_build_file,
            "would_skip": list(self.would_skip),
            "skipped": list(self.skipped),
        }

    def summary(self) -> str:
        total = len(self.selectable)
        if self.full_build_file:
            return (
                f"Diff-aware test selection ({self.mode}): every PR TPU step "
                f"runs; `{self.full_build_file}` needs the full build."
            )
        names = ", ".join(f"`{s}`" for s in self.would_skip) or "none"
        if self.mode == "shadow":
            return (
                "Diff-aware test selection (shadow mode, nothing skipped): this "
                f"change would skip {len(self.would_skip)} of {total} PR TPU "
                f"steps: {names}."
            )
        note = (
            f"Diff-aware test selection ({self.mode}): skipped "
            f"{len(self.skipped)} of {total} PR TPU steps: "
            + (", ".join(f"`{s}`" for s in self.skipped) or "none")
            + "."
        )
        if extra := sorted(set(self.would_skip) - set(self.skipped)):
            note += " phase2 would also skip: " + ", ".join(f"`{s}`" for s in extra)
            note += "."
        return note


def select(
    changed_files: Iterable[str],
    config: Config,
    *,
    mode: str,
    repo_root: pathlib.Path = REPO_ROOT,
) -> Selection:
    """Take the union, over `changed_files`, of the steps each one needs."""
    active_phases = {"phase1"} if mode == "phase1" else set(PHASES)
    needed: set[str] = set()
    mode_needed: set[str] = set()
    full_build_file = ""
    mode_full = False
    rules: set[str] = set()
    files = [f for f in (p.strip() for p in changed_files) if f]
    for path in files:
        steps, rule, phase = classify(path, config, repo_root)
        rules.add(rule)
        if steps is None:
            full_build_file = full_build_file or path
            continue
        needed |= steps
        if phase in active_phases:
            mode_needed |= steps
        else:
            mode_full = True
    if not files:
        full_build_file = "(no changed files)"
    return Selection(
        mode=mode,
        selectable=config.selectable,
        files=len(files),
        rules=tuple(sorted(rules)),
        full_build_file=full_build_file,
        needed=None if full_build_file else frozenset(needed),
        mode_needed=None if full_build_file or mode_full else frozenset(mode_needed),
    )


def mark_skipped(steps: MutableSequence[Any], reasons: Mapping[str, str]) -> int:
    """Set `skip:` on the steps (in groups too) whose key is in `reasons`.

    Works on the JSON a `pipeline upload --dry-run` prints. A step that
    already carries a `skip:` keeps it. Returns how many steps were marked.
    """
    marked = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        if isinstance(step.get("steps"), list):
            marked += mark_skipped(step["steps"], reasons)
        key = step.get("key")
        if isinstance(key, str) and key in reasons and not step.get("skip"):
            step["skip"] = reasons[key]
            marked += 1
    return marked


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="changed files, repo-relative")
    parser.add_argument("--mode", default="phase2", choices=MODES[1:])
    args = parser.parse_args(argv)
    config = load_config()
    for path in args.files:
        steps, rule, phase = classify(path, config)
        needs = "the full build" if steps is None else (sorted(steps) or "no TPU steps")
        print(f"{path}: {rule} ({phase}): {needs}")
    selection = select(args.files, config, mode=args.mode)
    print(json.dumps(selection.record(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
