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
"""Unit tests for `.buildkite/stages.yml`, `render_pipeline.py`, and `bootstrap.sh`."""

from __future__ import annotations

import copy
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from typing import Any

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / ".buildkite" / "scripts"
_BOOTSTRAP = _SCRIPTS_DIR / "bootstrap.sh"
_BK_BIN = shutil.which("bk")

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import render_pipeline as rp  # noqa: E402

_PRECHECK_KEYS = ["typecheck_full", "unit_tests_cpu_pr", "unit_tests_cpu_nightly"]

_FAKE_AGENT = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/calls.log"
if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
  shopt -s nullglob
  ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)
  cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"
fi
"""

_FAKE_GIT = """#!/usr/bin/env bash
if [[ "${1:-}" == "fetch" ]]; then
  exit 0
fi
if [[ "${1:-}" == "diff" || "${1:-}" == "diff-tree" ]]; then
  echo "src/vllm_torchtpu/runner/tpu_runner.py"
  exit 0
fi
exec /usr/bin/git "$@"
"""

_FAKE_CURL = """#!/usr/bin/env bash
if [[ -z "${TEST_GITHUB_LABELS:-}" ]]; then
  exit 22
fi
echo "${TEST_GITHUB_LABELS}"
"""


def run_bootstrap(**env: str) -> tuple[list[str], list[str], dict[str, str]]:
    """Run `bootstrap.sh` against a stub `buildkite-agent`."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, script in (
            ("buildkite-agent", _FAKE_AGENT),
            ("git", _FAKE_GIT),
            ("curl", _FAKE_CURL),
        ):
            path = bin_dir / name
            path.write_text(script)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)

        run_env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "TEST_AGENT_LOG": str(tmp_path),
            "BUILDKITE_COMMIT": "HEAD",
            "BUILDKITE_BRANCH": "feature-branch",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_PULL_REQUEST_LABELS": "",
            "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
            **env,
        }
        proc = subprocess.run(
            ["bash", str(_BOOTSTRAP)],
            cwd=_REPO_ROOT,
            env=run_env,
            capture_output=True,
            text=True,
            check=True,
        )
        calls_file = tmp_path / "calls.log"
        calls = calls_file.read_text().splitlines() if calls_file.exists() else []
        uploads = [p.read_text() for p in sorted(tmp_path.glob("upload_*.yml"))]
        return calls, uploads, {"stdout": proc.stdout, "stderr": proc.stderr}


def _deps(step: dict[str, Any]) -> list[str]:
    raw = step.get("depends_on")
    if raw is None:
        return []
    return [raw] if isinstance(raw, str) else list(raw)


class TestBootstrapAndRenderer(unittest.TestCase):
    """Verify stage ordering, conditional gating, key integrity, and validation."""

    def _assert_valid_pipeline(self, yaml_text: str) -> dict[str, Any]:
        doc = yaml.safe_load(yaml_text)
        self.assertIsInstance(doc, dict)
        steps = doc["steps"]
        all_items: list[dict[str, Any]] = []
        for step in steps:
            self.assertIsInstance(step, dict)
            self.assertNotIn("wait", step)
            all_items.append(step)
            if "group" in step:
                for inner in step["steps"]:
                    self.assertIsInstance(inner, dict)
                    self.assertNotIn("wait", inner)
                    self.assertNotIn("group", inner)
                    all_items.append(inner)

        known_keys = {item["key"] for item in all_items if "key" in item}
        if "upload_models_and_features" in known_keys:
            known_keys.update({"v6e-group", "v7x-group"})
        for item in all_items:
            for dep in _deps(item):
                self.assertIn(dep, known_keys)

        if _BK_BIN:
            with tempfile.NamedTemporaryFile("w", suffix=".yml") as tmp:
                tmp.write(yaml_text)
                tmp.flush()
                proc = subprocess.run(
                    [_BK_BIN, "pipeline", "validate", "-f", tmp.name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    proc.returncode, 0, msg=f"{proc.stdout}\n{proc.stderr}"
                )
        return doc

    def test_pr_build_is_gated_and_caps_tpu_concurrency(self) -> None:
        calls, uploads, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
            BUILDKITE_BUILD_AUTHOR_EMAIL="Alice@example.com",
        )
        self.assertEqual(len(uploads), 1)
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        self.assertIn('build.env("NIGHTLY")', uploads[0])

        doc = self._assert_valid_pipeline(uploads[0])
        self.assertEqual(doc["priority"], 6)
        by_key = {s["key"]: s for s in doc["steps"]}
        self.assertNotIn("tpu_gate_open", by_key)
        self.assertNotIn("tpu_gate_close", by_key)
        self.assertEqual(
            list(by_key),
            [
                "build_docker_group",
                *_PRECHECK_KEYS,
                "unit_tests_tpu",
                "integration_tpu",
                "upload_models_and_features",
                "nightly_report",
            ],
        )
        for cpu_key in _PRECHECK_KEYS:
            self.assertEqual(
                _deps(by_key[cpu_key]), ["build_docker", "build_docker_group"]
            )
            self.assertNotIn("concurrency_group", by_key[cpu_key])

        expected_tpu_deps = [*_PRECHECK_KEYS, "build_docker_group"]
        for tpu_key in (
            "unit_tests_tpu",
            "integration_tpu",
            "upload_models_and_features",
        ):
            self.assertEqual(_deps(by_key[tpu_key]), expected_tpu_deps)

        integ_step = by_key["integration_tpu"]["steps"][0]
        self.assertEqual(integ_step["env"]["ARTIFACTS_DIR"], "perf_eval_results")
        self.assertEqual(integ_step["concurrency_group"], "tpu-pr/alice@example.com")
        self.assertTrue(by_key["nightly_report"]["allow_dependency_failure"])

    def test_nightly_is_ungated_and_substitutes_nightly_env(self) -> None:
        calls, uploads, _ = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            NIGHTLY="1",
        )
        self.assertEqual(len(uploads), 1)
        self.assertIn("meta-data set JOB_PRIORITY 1", calls)
        self.assertFalse(any("WORKLOAD_PRIORITY" in c for c in calls))
        self.assertNotIn('build.env("NIGHTLY")', uploads[0])

        doc = self._assert_valid_pipeline(uploads[0])
        self.assertEqual(doc["priority"], 1)
        by_key = {s["key"]: s for s in doc["steps"]}
        self.assertNotIn("tpu_gate_open", by_key)
        self.assertNotIn("tpu_gate_close", by_key)
        self.assertIn("perf_tpu", by_key)
        for tpu_key in (
            "unit_tests_tpu",
            "integration_tpu",
            "perf_tpu",
            "upload_models_and_features",
        ):
            self.assertEqual(_deps(by_key[tpu_key]), ["build_docker_group"])
        self.assertEqual(
            by_key["perf_tpu"]["steps"][0]["env"]["ARTIFACTS_DIR"],
            "perf_eval_results",
        )
        for req in ("v6e-group", "v7x-group", *_PRECHECK_KEYS, "unit_tests_tpu"):
            self.assertIn(req, _deps(by_key["nightly_report"]))

    def test_oncall_fix_and_release_tag_ungated_and_merge_queue_gated(self) -> None:
        # oncall-fix PR: ungated, priority 12.
        calls, uploads, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,oncall-fix",
        )
        doc = self._assert_valid_pipeline(uploads[0])
        self.assertEqual(doc["priority"], 12)
        self.assertTrue(any("--context oncall-fix" in c for c in calls))
        by_key = {s["key"]: s for s in doc["steps"]}
        self.assertEqual(_deps(by_key["unit_tests_tpu"]), ["build_docker_group"])

        # Release tag: ungated.
        tag_yaml = rp.dump_pipeline(
            rp.render_pipeline(_REPO_ROOT, {"BUILDKITE_TAG": "v0.32.0"})
        )
        tag_doc = self._assert_valid_pipeline(tag_yaml)
        tag_by_key = {s["key"]: s for s in tag_doc["steps"]}
        self.assertEqual(_deps(tag_by_key["unit_tests_tpu"]), ["build_docker_group"])

        # Merge queue: gated on precheck.
        mq_yaml = rp.dump_pipeline(
            rp.render_pipeline(
                _REPO_ROOT,
                {
                    "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
                    "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-123",
                },
            )
        )
        mq_doc = self._assert_valid_pipeline(mq_yaml)
        mq_by_key = {s["key"]: s for s in mq_doc["steps"]}
        self.assertNotIn("tpu_gate_open", mq_by_key)
        self.assertNotIn("tpu_gate_close", mq_by_key)
        self.assertEqual(
            _deps(mq_by_key["unit_tests_tpu"]),
            [*_PRECHECK_KEYS, "build_docker_group"],
        )

    def test_ci_run_emits_per_author_concurrency_gate(self) -> None:
        _, uploads, _ = run_bootstrap(
            CI_RUN_TPU="1",
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
            BUILDKITE_BUILD_AUTHOR_EMAIL="Alice@example.com",
        )
        doc = self._assert_valid_pipeline(uploads[0])
        by_key = {s["key"]: s for s in doc["steps"]}
        self.assertEqual(
            list(by_key),
            [
                "build_docker_group",
                *_PRECHECK_KEYS,
                "tpu_gate_open",
                "unit_tests_tpu",
                "integration_tpu",
                "upload_models_and_features",
                "tpu_gate_close",
                "nightly_report",
            ],
        )
        open_step = by_key["tpu_gate_open"]
        close_step = by_key["tpu_gate_close"]
        for gate_step in (open_step, close_step):
            self.assertEqual(gate_step["concurrency"], 1)
            self.assertEqual(gate_step["concurrency_group"], "tpu-pr/alice@example.com")
            self.assertNotIn("concurrency_method", gate_step)
            self.assertEqual(gate_step["agents"], {"queue": "cpu"})

        self.assertEqual(_deps(open_step), [*_PRECHECK_KEYS, "build_docker_group"])
        self.assertNotIn("allow_dependency_failure", open_step)
        for tpu_key in (
            "unit_tests_tpu",
            "integration_tpu",
            "upload_models_and_features",
        ):
            self.assertEqual(_deps(by_key[tpu_key]), ["tpu_gate_open"])
        self.assertNotIn("concurrency_group", by_key["integration_tpu"]["steps"][0])
        self.assertEqual(
            _deps(close_step),
            ["unit_tests_tpu", "integration_tpu", "upload_models_and_features"],
        )
        self.assertTrue(close_step["allow_dependency_failure"])

    def test_run_perf_eval_and_run_full_ci_pr_labels(self) -> None:
        calls, uploads, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-perf-eval",
        )
        self.assertTrue(any("--context perf-eval" in c for c in calls))
        self.assertIn('build.env("NIGHTLY")', uploads[0])
        perf_doc = self._assert_valid_pipeline(uploads[0])
        perf_by_key = {s["key"]: s for s in perf_doc["steps"]}
        self.assertEqual(
            _deps(perf_by_key["perf_tpu"]),
            [*_PRECHECK_KEYS, "build_docker_group"],
        )

        calls, uploads, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
        )
        self.assertTrue(any("--context full-ci" in c for c in calls))
        self.assertNotIn('build.env("NIGHTLY")', uploads[0])
        full_doc = self._assert_valid_pipeline(uploads[0])
        self.assertEqual(full_doc["priority"], 6)
        full_by_key = {s["key"]: s for s in full_doc["steps"]}
        self.assertEqual(
            _deps(full_by_key["perf_tpu"]),
            [*_PRECHECK_KEYS, "build_docker_group"],
        )

    def test_skip_tpu_ci_on_pr_and_post_merge(self) -> None:
        for env in (
            {
                "BUILDKITE_PULL_REQUEST": "123",
                "BUILDKITE_PULL_REQUEST_LABELS": "ready,run-full-ci,skip-tpu-ci",
            },
            {
                "BUILDKITE_BRANCH": "main",
                "BUILDKITE_PULL_REQUEST": "false",
                "BUILDKITE_SOURCE": "webhook",
                "BUILDKITE_MESSAGE": "[CI/Build] Fix a script (#123)",
                "TEST_GITHUB_LABELS": '[{"name": "ready"}, {"name": "skip-tpu-ci"}]',
            },
        ):
            calls, uploads, _ = run_bootstrap(**env)
            self.assertTrue(any("--context skip-tpu-ci" in c for c in calls))
            doc = self._assert_valid_pipeline(uploads[0])
            self.assertEqual(
                [s["key"] for s in doc["steps"]],
                ["build_docker_group", *_PRECHECK_KEYS],
            )
            omni_step = doc["steps"][0]["steps"][1]
            self.assertIn("skip-tpu-ci", omni_step["skip"])

    def test_ci_lanes_perf_and_integration_pipeline_with_promote(self) -> None:
        perf_yaml = rp.dump_pipeline(
            rp.render_pipeline(_REPO_ROOT, {"CI_LANES": "perf"})
        )
        perf_doc = self._assert_valid_pipeline(perf_yaml)
        self.assertEqual(
            [s["key"] for s in perf_doc["steps"]],
            ["build_docker_group", "perf_tpu"],
        )
        self.assertEqual(_deps(perf_doc["steps"][1]), ["build_docker_group"])
        self.assertIn("tests lane", perf_doc["steps"][0]["steps"][1]["skip"])

        _, uploads, _ = run_bootstrap(
            BUILDKITE_PIPELINE_SLUG="vllm-torchtpu-integration",
        )
        integ_doc = self._assert_valid_pipeline(uploads[0])
        by_key = {s["key"]: s for s in integ_doc["steps"]}
        self.assertEqual(list(by_key)[-1], "promote_lkg")
        self.assertEqual(
            _deps(by_key["promote_lkg"]),
            [
                "unit_tests_tpu",
                "integration_tpu",
                "perf_tpu",
                "upload_models_and_features",
                "nightly_report",
            ],
        )
        self.assertNotIn("allow_dependency_failure", by_key["promote_lkg"])

    def test_unknown_condition_and_missing_key_are_rejected(self) -> None:
        base = yaml.safe_load((_REPO_ROOT / ".buildkite" / "stages.yml").read_text())

        bad_cond = copy.deepcopy(base)
        bad_cond["stages"][2]["needs"][0]["unless"] = ["bogus-cond"]
        with self.assertRaisesRegex(ValueError, "Unknown condition 'bogus-cond'"):
            rp.render_pipeline(_REPO_ROOT, {}, stages_doc=bad_cond)

        missing_key = copy.deepcopy(base)
        # Drop `build` while `precheck` still declares `depends_on: build_docker`.
        missing_key["stages"][0]["files"] = []
        with self.assertRaisesRegex(
            ValueError, "depends_on unknown key 'build_docker'"
        ):
            rp.render_pipeline(_REPO_ROOT, {}, stages_doc=missing_key)


if __name__ == "__main__":
    unittest.main()
