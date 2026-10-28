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
"""Select Buildkite CI tracks (Core vs. Omni) from changed files and build context.

Derives the shared Python boundary between Omni (`src/vllm_torchtpu/omni/` and
`tests_omni/`) and the rest of the repository directly from the AST import graph
rather than maintaining a static file list:

- Importing a submodule `a.b.c` executes the module-level body of each parent
  package (`a/__init__.py`, `a/b/__init__.py`) and `a/b/c.py`.
- `OmniTpuPlatform` subclasses `TpuPlatform`; module-level statements of
  `tpu_platform.py` and any `TpuPlatform` methods referenced by attribute name in
  `src/vllm_torchtpu/omni/` are traversed so any helper they import is treated as
  shared (`both` tracks).
- When `pyproject.toml` changes, `tomllib` diffs the base and head TOML trees so
  edits scoped to `[project.optional-dependencies].omni` or
  `[dependency-groups].omni-git` run only the Omni track, edits scoped to
  `[project.optional-dependencies].benchmarking` run only the Core track, and any
  other `pyproject.toml` change runs both tracks.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tomllib

MAX_BUILDKITE_SKIP_LEN = 70

ALL_LANES = ("verify", "perf", "integration", "tests")
DEFAULT_CORE_LANES = ("verify", "integration", "tests")

DOCS_ONLY_PATTERN = re.compile(
    r"(\.md$|\.ico$|\.png$|^README$|^docs/|^\.github/CODEOWNERS$"
    r"|^LICENSE$|^\.git-blame-ignore-revs$|^\.pyrefly-baseline\.json$)"
)
RELEASE_TAG_PATTERN = re.compile(r"^v?[0-9]+(\.[a-zA-Z0-9]+)*([a-zA-Z]+[0-9]*)?$")

CORE_ONLY_BUILDKITE_PREFIXES = (
    ".buildkite/features/",
    ".buildkite/models/",
    ".buildkite/release/",
)
CORE_ONLY_BUILDKITE_FILES = frozenset(
    {
        ".buildkite/pipeline_coverage.yml",
        ".buildkite/pipeline_dev.yml",
        ".buildkite/pipeline_integration.yml",
        ".buildkite/pipeline_nightly_image.yml",
        ".buildkite/pipeline_perf.yml",
        ".buildkite/pipeline_promote.yml",
        ".buildkite/scripts/commit_support_matrices.sh",
        ".buildkite/scripts/commit_verified_commit_hashes.sh",
        ".buildkite/scripts/generate_support_matrices.sh",
        ".buildkite/scripts/publish_nightly_images.sh",
        ".buildkite/scripts/update_lkg_version.sh",
        ".buildkite/scripts/upload_models_and_features.sh",
    }
)

SKIP_OMNI_NO_TESTS_LANE = (
    "Only the tests lane uses the -omni image; this build has none."
)
SKIP_OMNI_TPU_DISABLED = "Only TPU steps use the -omni image; skip-tpu-ci skips them."
SKIP_OMNI_STEPS_TPU_DISABLED = "Skipped by skip-tpu-ci label."
SKIP_OMNI_INTEGRATION_PIPELINE = (
    "Skipped on vllm-torchtpu-integration (Core LKG pipeline)."
)
SKIP_OMNI_IMAGE_CORE_DIFF = "No Omni files changed; skipping -omni image build."
SKIP_OMNI_TESTS_CORE_DIFF = "No Omni files changed; skipping Omni test steps."
SKIP_CORE_IMAGE_OMNI_DIFF = "Only Omni files changed; skipping Core CI image build."
SKIP_CORE_TESTS_OMNI_DIFF = "Only Omni files changed; skipping Core test steps."


@dataclasses.dataclass(frozen=True)
class TrackPlan:
    """Resolved CI track selection, lane list, and Buildkite step skip reasons."""

    selection: str  # "skip-all" | "core-only" | "omni-only" | "both"
    default_lanes: str
    core_image_skip: str = ""
    core_tests_skip: str = ""
    omni_image_skip: str = ""
    omni_tests_skip: str = ""
    omni_full_ci: str = "0"
    nightly: str = "0"
    skip_tpu_ci: str = "0"
    annotation: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "core_image_skip",
            "core_tests_skip",
            "omni_image_skip",
            "omni_tests_skip",
        ):
            val = getattr(self, field_name)
            if len(val) > MAX_BUILDKITE_SKIP_LEN:
                raise ValueError(
                    f"{field_name} exceeds {MAX_BUILDKITE_SKIP_LEN} chars "
                    f"({len(val)}): {val!r}"
                )

    def to_shell(self) -> str:
        """Render shell variable assignments safe for `eval` in bash."""
        pairs = (
            ("CI_TRACK_SELECTION", self.selection),
            ("TRACK_DEFAULT_LANES", self.default_lanes),
            ("CORE_IMAGE_SKIP", self.core_image_skip),
            ("CORE_TESTS_SKIP", self.core_tests_skip),
            ("OMNI_IMAGE_SKIP", self.omni_image_skip),
            ("OMNI_TESTS_SKIP", self.omni_tests_skip),
            ("OMNI_FULL_CI", self.omni_full_ci),
            ("NIGHTLY", self.nightly),
            ("SKIP_TPU_CI", self.skip_tpu_ci),
            ("TRACK_ANNOTATION", self.annotation),
        )
        return "\n".join(f"{key}={shlex.quote(val)}" for key, val in pairs)


def _omni_referenced_attributes(omni_dir: pathlib.Path) -> set[str]:
    """Attribute names referenced anywhere in `src/vllm_torchtpu/omni/*.py`."""
    attrs: set[str] = {"__init__"}
    if not omni_dir.is_dir():
        return attrs
    for py_file in sorted(omni_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attrs.add(node.attr)
    return attrs


def _resolve_repo_module(
    repo_root: pathlib.Path,
    module_name: str,
) -> list[pathlib.Path]:
    """Resolve a dotted `vllm_torchtpu.*`, `tests.*`, or `tests_omni.*` module."""
    parts = module_name.split(".")
    if not parts:
        return []
    if parts[0] == "vllm_torchtpu":
        base = repo_root / "src"
    elif parts[0] in ("tests", "tests_omni"):
        base = repo_root
    else:
        return []

    resolved: list[pathlib.Path] = []
    for idx in range(1, len(parts) + 1):
        pkg_init = base.joinpath(*parts[:idx], "__init__.py")
        if pkg_init.is_file():
            resolved.append(pkg_init)
    mod_file = base.joinpath(*parts[:-1], f"{parts[-1]}.py")
    if mod_file.is_file():
        resolved.append(mod_file)
    return resolved


def _imports_in_python_file(
    repo_root: pathlib.Path,
    path: pathlib.Path,
    tpu_platform_methods: set[str],
) -> set[str]:
    """Extract imported module names executed or reachable from `path`."""
    src_root = repo_root / "src"
    omni_init = src_root / "vllm_torchtpu" / "omni" / "__init__.py"
    tpu_platform = src_root / "vllm_torchtpu" / "platforms" / "tpu_platform.py"

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(src_root if path.is_relative_to(src_root) else repo_root)
    pkg_parts = list(rel.parts[:-1])

    is_non_omni_pkg_init = path.name == "__init__.py" and path != omni_init
    if is_non_omni_pkg_init:
        nodes: list[ast.AST] = []
        for stmt in tree.body:
            if not isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                nodes.extend(ast.walk(stmt))
    elif path == tpu_platform:
        nodes = []
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name == "TpuPlatform":
                for item in stmt.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name in tpu_platform_methods
                    ):
                        nodes.extend(ast.walk(item))
            elif not isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                nodes.extend(ast.walk(stmt))
    else:
        nodes = list(ast.walk(tree))

    modules: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                keep = max(len(pkg_parts) - (node.level - 1), 0)
                base_parts = pkg_parts[:keep]
            else:
                base_parts = []
            mod_parts = base_parts + (node.module.split(".") if node.module else [])
            if mod_parts:
                base_mod = ".".join(mod_parts)
                modules.add(base_mod)
                for alias in node.names:
                    if alias.name != "*":
                        modules.add(f"{base_mod}.{alias.name}")
    return modules


def compute_omni_shared_closure(repo_root: pathlib.Path) -> frozenset[str]:
    """Return repo-relative POSIX paths outside `omni/` & `tests_omni/` used by Omni."""
    omni_src = repo_root / "src" / "vllm_torchtpu" / "omni"
    omni_tests = repo_root / "tests_omni"
    roots: list[pathlib.Path] = []
    if omni_src.is_dir():
        roots.extend(sorted(omni_src.rglob("*.py")))
    if omni_tests.is_dir():
        roots.extend(sorted(omni_tests.rglob("*.py")))

    tpu_platform_methods = _omni_referenced_attributes(omni_src)
    visited: set[pathlib.Path] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        for mod_name in _imports_in_python_file(
            repo_root, current, tpu_platform_methods
        ):
            for target in _resolve_repo_module(repo_root, mod_name):
                if target not in visited:
                    stack.append(target)

    shared: set[str] = set()
    for path in visited:
        if path.is_relative_to(omni_src) or path.is_relative_to(omni_tests):
            continue
        shared.add(path.relative_to(repo_root).as_posix())
    return frozenset(shared)


def classify_pyproject_diff(
    base_toml_text: str | None, head_toml_text: str | None
) -> str:
    """Classify a `pyproject.toml` modification into `'core'`, `'omni'`, or `'both'`."""
    if not base_toml_text or not head_toml_text:
        return "both"
    try:
        base_data = tomllib.loads(base_toml_text)
        head_data = tomllib.loads(head_toml_text)
    except tomllib.TOMLDecodeError:
        return "both"

    if base_data == head_data:
        # Comments-only edit in pyproject.toml still touches Dockerfile manifests
        # only if dependencies changed; treat comment-only edits as docs.
        return "docs"

    base_proj = dict(base_data.get("project", {}))
    head_proj = dict(head_data.get("project", {}))
    base_opt = dict(base_proj.pop("optional-dependencies", {}))
    head_opt = dict(head_proj.pop("optional-dependencies", {}))

    base_groups = dict(base_data.get("dependency-groups", {}))
    head_groups = dict(head_data.get("dependency-groups", {}))

    base_rest = {
        k: v for k, v in base_data.items() if k not in ("project", "dependency-groups")
    }
    head_rest = {
        k: v for k, v in head_data.items() if k not in ("project", "dependency-groups")
    }

    if base_proj != head_proj or base_rest != head_rest:
        return "both"

    tracks: set[str] = set()
    for key in set(base_opt) | set(head_opt):
        if base_opt.get(key) != head_opt.get(key):
            if key == "omni":
                tracks.add("omni")
            elif key == "benchmarking":
                tracks.add("core")
            else:
                return "both"

    for key in set(base_groups) | set(head_groups):
        if base_groups.get(key) != head_groups.get(key):
            if key == "omni-git":
                tracks.add("omni")
            else:
                return "both"

    if tracks == {"omni"}:
        return "omni"
    if tracks == {"core"}:
        return "core"
    return "both"


def _git_show_file(repo_root: pathlib.Path, ref: str, rel_path: str) -> str | None:
    if not ref:
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{rel_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def classify_file_path(
    rel_path: str,
    omni_shared_files: frozenset[str],
    *,
    pyproject_track: str = "both",
) -> str:
    """Classify a repo-relative path into `'docs'`, `'core'`, `'omni'`, or `'both'`."""
    norm = rel_path.strip()
    while norm.startswith("./"):
        norm = norm.removeprefix("./")
    if not norm:
        return "docs"
    if DOCS_ONLY_PATTERN.search(norm):
        return "docs"
    if norm.startswith(("src/vllm_torchtpu/omni/", "tests_omni/")):
        return "omni"
    if norm == "pyproject.toml":
        return pyproject_track
    if norm in omni_shared_files:
        return "both"
    if norm.startswith("src/vllm_torchtpu/") or norm == "sitecustomize.py":
        return "core"
    if norm.startswith(("tests/", "scripts/", "examples/", "tools/", "requirements/")):
        return "core"
    if norm in (".pyrefly-baseline-full.json", ".pre-commit-config.yaml"):
        return "core"
    if norm.startswith("support_matrices/"):
        return "core"
    if norm in CORE_ONLY_BUILDKITE_FILES or norm.startswith(
        CORE_ONLY_BUILDKITE_PREFIXES
    ):
        return "core"
    return "both"


def classify_changed_files(
    files: list[str],
    omni_shared_files: frozenset[str],
    *,
    pyproject_track: str = "both",
) -> str:
    """Aggregate file classifications into `'docs'`, `'core'`, `'omni'`, or `'both'`."""
    clean = [f.strip() for f in files if f.strip()]
    if not clean:
        return "both"

    needs_core = False
    needs_omni = False
    for path in clean:
        bucket = classify_file_path(
            path,
            omni_shared_files,
            pyproject_track=pyproject_track,
        )
        if bucket == "core":
            needs_core = True
        elif bucket == "omni":
            needs_omni = True
        elif bucket == "both":
            return "both"

    if needs_core and needs_omni:
        return "both"
    if needs_core:
        return "core"
    if needs_omni:
        return "omni"
    return "docs"


def _parse_labels(raw_labels: str) -> set[str]:
    return {
        lbl.strip() for lbl in raw_labels.replace("\n", ",").split(",") if lbl.strip()
    }


def decide_ci_plan(
    *,
    changed_files: list[str],
    omni_shared_files: frozenset[str],
    pyproject_track: str = "both",
    labels: set[str] | None = None,
    nightly: bool = False,
    torch_tpu_nightly: bool = False,
    is_schedule: bool = False,
    build_tag: str = "",
    is_pull_request: bool = False,
    pipeline_slug: str = "",
    integration_slug: str = "vllm-torchtpu-integration",
    ci_lanes: str = "",
    skip_tpu_ci: bool = False,
) -> TrackPlan:
    """Compute the end-to-end `TrackPlan` from diff classification and CI flags."""
    active_labels = labels or set()
    is_release_tag = bool(build_tag and RELEASE_TAG_PATTERN.match(build_tag))
    effective_nightly = nightly or (is_pull_request and "run-full-ci" in active_labels)
    effective_skip_tpu = skip_tpu_ci or ("skip-tpu-ci" in active_labels)
    is_integration_pipeline = bool(pipeline_slug and pipeline_slug == integration_slug)

    diff_track = classify_changed_files(
        changed_files,
        omni_shared_files,
        pyproject_track=pyproject_track,
    )

    has_full_override = (
        effective_nightly
        or torch_tpu_nightly
        or is_schedule
        or is_release_tag
        or is_integration_pipeline
        or bool(ci_lanes.strip())
        or bool(
            active_labels
            & {"run-full-ci", "run-omni-full-ci", "run-perf-eval", "skip-tpu-ci"}
        )
    )
    if changed_files and diff_track == "docs" and not has_full_override:
        return TrackPlan(
            selection="skip-all",
            default_lanes="",
            nightly="1" if effective_nightly else "0",
            skip_tpu_ci="1" if effective_skip_tpu else "0",
        )

    # skip-tpu-ci overrides run-perf-eval, run-full-ci, and run-omni-full-ci:
    # run only the CPU steps of the tests lane on the Core CI image.
    if effective_skip_tpu:
        lanes = ci_lanes.strip() or "tests"
        return TrackPlan(
            selection="core-only",
            default_lanes=lanes,
            omni_image_skip=SKIP_OMNI_TPU_DISABLED,
            omni_tests_skip=SKIP_OMNI_STEPS_TPU_DISABLED,
            nightly="1" if effective_nightly else "0",
            skip_tpu_ci="1",
        )

    # Upstream vLLM HEAD canary / LKG promotion pipeline runs Core lanes only;
    # promote_lkg depends exclusively on Core steps.
    if is_integration_pipeline:
        lanes = ci_lanes.strip() or " ".join(ALL_LANES)
        return TrackPlan(
            selection="core-only",
            default_lanes=lanes,
            omni_image_skip=SKIP_OMNI_INTEGRATION_PIPELINE,
            omni_tests_skip=SKIP_OMNI_INTEGRATION_PIPELINE,
            nightly="1" if effective_nightly else "0",
        )

    # Scheduled, nightly, canary, or release-tag builds run both tracks.
    if effective_nightly or torch_tpu_nightly or is_schedule or is_release_tag:
        lanes = ci_lanes.strip() or (
            " ".join(ALL_LANES)
            if (effective_nightly or is_release_tag)
            else " ".join(DEFAULT_CORE_LANES)
        )
        lane_tokens = set(lanes.split())
        omni_img_skip = "" if "tests" in lane_tokens else SKIP_OMNI_NO_TESTS_LANE
        omni_tst_skip = "" if "tests" in lane_tokens else SKIP_OMNI_NO_TESTS_LANE
        return TrackPlan(
            selection="both" if not omni_img_skip else "core-only",
            default_lanes=lanes,
            omni_image_skip=omni_img_skip,
            omni_tests_skip=omni_tst_skip,
            nightly="1" if effective_nightly else "0",
        )

    omni_full = "1" if "run-omni-full-ci" in active_labels else "0"
    want_perf = is_pull_request and ("run-perf-eval" in active_labels)

    if omni_full == "1" and diff_track in ("docs", "omni") and not want_perf:
        selected_track = "omni"
    elif omni_full == "1":
        selected_track = "both"
    elif want_perf and diff_track in ("docs", "core"):
        selected_track = "core"
    elif want_perf or diff_track == "docs":
        selected_track = "both"
    else:
        selected_track = diff_track

    # Explicit CI_LANES (for example CI_LANES=perf schedules) takes precedence
    # for lane list and adjusts image skip flags to match the requested lanes.
    if ci_lanes.strip():
        lanes = ci_lanes.strip()
        lane_tokens = set(lanes.split())
        if "tests" not in lane_tokens:
            return TrackPlan(
                selection="core-only",
                default_lanes=lanes,
                omni_image_skip=SKIP_OMNI_NO_TESTS_LANE,
                omni_tests_skip=SKIP_OMNI_NO_TESTS_LANE,
                omni_full_ci=omni_full,
            )
        if lane_tokens & {"integration", "perf"} and selected_track == "omni":
            selected_track = "both"
    else:
        if selected_track == "omni":
            lanes = "tests"
        elif want_perf:
            lanes = " ".join(ALL_LANES)
        else:
            lanes = " ".join(DEFAULT_CORE_LANES)

    if selected_track == "core":
        return TrackPlan(
            selection="core-only",
            default_lanes=lanes,
            omni_image_skip=SKIP_OMNI_IMAGE_CORE_DIFF,
            omni_tests_skip=SKIP_OMNI_TESTS_CORE_DIFF,
            omni_full_ci=omni_full,
        )

    if selected_track == "omni":
        note = (
            "Runs the full Omni suite (`run-omni-full-ci` label)."
            if omni_full == "1"
            else ""
        )
        return TrackPlan(
            selection="omni-only",
            default_lanes=lanes,
            core_image_skip=SKIP_CORE_IMAGE_OMNI_DIFF,
            core_tests_skip=SKIP_CORE_TESTS_OMNI_DIFF,
            omni_full_ci=omni_full,
            annotation=note,
        )

    both_note = (
        "Runs the full Omni suite (`run-omni-full-ci` label)."
        if omni_full == "1"
        else ""
    )
    return TrackPlan(
        selection="both",
        default_lanes=lanes,
        omni_full_ci=omni_full,
        annotation=both_note,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
        help="Path to vllm-torchtpu repository root.",
    )
    parser.add_argument(
        "--base-ref",
        default="",
        help="Git base ref for diffing structured files such as pyproject.toml.",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Git head ref for diffing structured files such as pyproject.toml.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated PR or merged-PR labels.",
    )
    parser.add_argument(
        "--format",
        choices=("shell", "json"),
        default="shell",
        help="Output format (`shell` assignments or `json`).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    changed_files = (
        [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
        if not sys.stdin.isatty()
        else []
    )

    pyproject_track = "both"
    normalized_files = {f.removeprefix("./") for f in changed_files}
    if "pyproject.toml" in normalized_files:
        base_toml = _git_show_file(repo_root, args.base_ref, "pyproject.toml")
        head_toml = _git_show_file(repo_root, args.head_ref, "pyproject.toml")
        if head_toml is None and (repo_root / "pyproject.toml").is_file():
            head_toml = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        pyproject_track = classify_pyproject_diff(base_toml, head_toml)

    omni_shared = compute_omni_shared_closure(repo_root)
    env = os.environ
    pr_val = env.get("BUILDKITE_PULL_REQUEST", "false")
    is_pr = bool(pr_val and pr_val != "false")
    raw_labels = args.labels or env.get("BUILDKITE_PULL_REQUEST_LABELS", "")

    plan = decide_ci_plan(
        changed_files=changed_files,
        omni_shared_files=omni_shared,
        pyproject_track=pyproject_track,
        labels=_parse_labels(raw_labels),
        nightly=env.get("NIGHTLY", "0") == "1",
        torch_tpu_nightly=env.get("TORCHTPU_NIGHTLY", "0") == "1",
        is_schedule=env.get("BUILDKITE_SOURCE", "") == "schedule",
        build_tag=env.get("BUILDKITE_TAG", ""),
        is_pull_request=is_pr,
        pipeline_slug=env.get("BUILDKITE_PIPELINE_SLUG", ""),
        integration_slug=env.get(
            "VLLM_INTEGRATION_PIPELINE_SLUG", "vllm-torchtpu-integration"
        ),
        ci_lanes=env.get("CI_LANES", ""),
        skip_tpu_ci=env.get("SKIP_TPU_CI", "0") == "1",
    )

    if args.format == "json":
        print(json.dumps(dataclasses.asdict(plan), indent=2, sort_keys=True))
    else:
        print(plan.to_shell())
    return 0


if __name__ == "__main__":
    sys.exit(main())
