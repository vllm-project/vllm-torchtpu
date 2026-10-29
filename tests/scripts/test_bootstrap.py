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
"""Unit tests for .buildkite/scripts/bootstrap.sh and pipeline_config.sh."""

import json
import os
import pathlib
import stat
import subprocess
import tempfile
import unittest

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / ".buildkite" / "scripts" / "bootstrap.sh"

_FAKE_AGENT = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/calls.log"
if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
  shopt -s nullglob
  if [[ "${3:-}" == "--dry-run" ]]; then
    dfiles=("${TEST_AGENT_LOG}"/dry_run_*.yml)
    cat > "${TEST_AGENT_LOG}/dry_run_${#dfiles[@]}.yml"
    echo '{"steps": [{"label": "stub", "agents": {"queue": "kube"}}]}'
  else
    ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)
    cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"
  fi
fi
"""

_FAKE_GIT = """#!/usr/bin/env bash
echo "$*" >> "${TEST_AGENT_LOG}/git.log"
if [[ "${1:-}" == "fetch" ]]; then
  exit 0
fi
if [[ "${1:-}" == "diff" || "${1:-}" == "diff-tree" ]]; then
  printf '%b\\n' "${TEST_CHANGED_FILES:-src/vllm_torchtpu/runner/tpu_runner.py}"
  exit 0
fi
exec /usr/bin/git "$@"
"""

# GitHub's labels and comments APIs for merged / merge-queue PRs.
_FAKE_CURL = """#!/usr/bin/env bash
if [[ "$*" == *"/comments"* ]]; then
  if [[ -n "${TEST_GITHUB_COMMENTS_ERROR:-}" ]]; then
    echo "${TEST_GITHUB_COMMENTS_ERROR}" >&2
    exit 22
  fi
  if [[ "$*" == *"page=1"* || "$*" != *"page="* ]]; then
    echo "${TEST_GITHUB_COMMENTS:-[]}"
  else
    echo "[]"
  fi
  exit 0
fi
if [[ -z "${TEST_GITHUB_LABELS:-}" ]]; then
  exit 22
fi
echo "${TEST_GITHUB_LABELS}"
"""


def run_bootstrap(
    *,
    with_pipeline_cpu: bool = False,
    **env: str,
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Run bootstrap.sh against a stub buildkite-agent."""
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

        workdir = _REPO_ROOT
        if with_pipeline_cpu:
            workdir = tmp_path / "repo"
            bk_dir = workdir / ".buildkite"
            bk_dir.mkdir(parents=True)
            for item in _REPO_ROOT.iterdir():
                if item.name != ".buildkite":
                    (workdir / item.name).symlink_to(item)
            for item in (_REPO_ROOT / ".buildkite").iterdir():
                (bk_dir / item.name).symlink_to(item)
            (bk_dir / "pipeline_cpu.yml").write_text(
                "steps:\n  - label: cpu\n    key: cpu_check\n"
            )

        # Provide a clean base environment so host Buildkite vars never leak in.
        run_env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "TEST_AGENT_LOG": str(tmp_path),
            "BUILDKITE_COMMIT": "HEAD",
            "BUILDKITE_BRANCH": "feature-branch",
            "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "",
            "BUILDKITE_MERGE_QUEUE_BASE_COMMIT": "",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_PULL_REQUEST_LABELS": "",
            "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
            **env,
        }
        proc = subprocess.run(
            ["bash", str(workdir / ".buildkite" / "scripts" / "bootstrap.sh")],
            cwd=workdir,
            env=run_env,
            capture_output=True,
            text=True,
            check=True,
        )
        calls_file = tmp_path / "calls.log"
        calls = calls_file.read_text().splitlines() if calls_file.exists() else []
        git_file = tmp_path / "git.log"
        git_log = git_file.read_text() if git_file.exists() else ""
        uploads = [p.read_text() for p in sorted(tmp_path.glob("upload_*.yml"))]
        dry_runs = [p.read_text() for p in sorted(tmp_path.glob("dry_run_*.yml"))]
        return (
            calls,
            uploads,
            dry_runs,
            {"stdout": proc.stdout, "stderr": proc.stderr, "git": git_log},
        )


class TestBootstrap(unittest.TestCase):
    """Verify lane selection, priority, and NIGHTLY substitution in bootstrap.sh."""

    def test_plain_pr_skips_perf_and_keeps_nightly_unset(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
        )
        # 3 lane files (verify, integration, tests) + pipeline_build.yml
        self.assertEqual(len(uploads), 4)
        self.assertNotIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        for content in uploads:
            self.assertTrue(content.startswith("priority: 6\n"))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)
        self.assertFalse(any("annotate" in c for c in calls))

    def test_run_perf_eval_adds_perf_lane_without_nightly(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-perf-eval",
        )
        # 4 lane files (verify, perf, integration, tests) + pipeline_build.yml
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertTrue(any("--context perf-eval" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)

    def test_run_full_ci_enables_all_lanes_and_substitutes_nightly(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        # Priority remains pre-merge (6), not nightly (1).
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        self.assertFalse(any("--context perf-eval" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 6\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)
            parsed = yaml.safe_load(content)
            self.assertIsInstance(parsed, dict)
            self.assertIn("steps", parsed)

    def test_run_full_ci_with_oncall_fix_keeps_oncall_priority(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci,oncall-fix",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("meta-data set JOB_PRIORITY 12", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY oncall-fix", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 12\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_run_full_ci_with_author_caps_tpu_concurrency_with_nightly_substituted(
        self,
    ) -> None:
        _, uploads, dry_runs, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
            BUILDKITE_BUILD_AUTHOR_EMAIL="Alice@example.com",
        )
        self.assertEqual(len(uploads), 5)
        self.assertEqual(len(dry_runs), 5)
        for raw_yaml in dry_runs:
            self.assertNotIn('build.env("NIGHTLY")', raw_yaml)
        for content in uploads:
            doc = json.loads(content)
            self.assertEqual(
                doc["steps"][0]["concurrency_group"],
                "tpu-pr/alice@example.com",
            )

    def test_skip_tpu_ci_runs_only_the_tests_lane(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci,skip-tpu-ci",
        )
        # pipeline_tests.yml + pipeline_build.yml
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        for lane in ("verify", "perf", "integration"):
            self.assertNotIn(f"Uploading .buildkite/pipeline_{lane}.yml", out["stdout"])
        self.assertTrue(any("--context skip-tpu-ci" in c for c in calls))
        # The TPU group's condition, filled in from SKIP_TPU_CI at upload.
        self.assertIn('"${SKIP_TPU_CI:-0}" != "1"', uploads[0])

    def test_post_merge_of_a_skip_tpu_ci_pr_runs_only_the_tests_lane(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            BUILDKITE_SOURCE="webhook",
            BUILDKITE_MESSAGE="[CI/Build] Fix a script (#123)",
            TEST_GITHUB_LABELS='[{"name": "ready"}, {"name": "skip-tpu-ci"}]',
        )
        self.assertEqual(len(uploads), 2)
        self.assertNotIn("Uploading .buildkite/pipeline_integration.yml", out["stdout"])
        self.assertTrue(any("--context skip-tpu-ci" in c for c in calls))

    def test_post_merge_runs_everything_without_the_label_or_the_lookup(
        self,
    ) -> None:
        for labels in ('[{"name": "ready"}]', ""):
            calls, uploads, _, _ = run_bootstrap(
                BUILDKITE_BRANCH="main",
                BUILDKITE_PULL_REQUEST="false",
                BUILDKITE_SOURCE="webhook",
                BUILDKITE_MESSAGE="[Core] Change something (#124)",
                TEST_GITHUB_LABELS=labels,
            )
            self.assertEqual(len(uploads), 4)
            self.assertFalse(any("--context skip-tpu-ci" in c for c in calls))

    def test_scheduled_nightly_keeps_nightly_priority(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            NIGHTLY="1",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("meta-data set JOB_PRIORITY 1", calls)
        self.assertFalse(any("WORKLOAD_PRIORITY" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 1\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_pr_tpu_opt_in_uploads_cpu_steps_only_until_ci_run_tpu(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
            CI_PR_TPU_OPT_IN="1",
        )
        # Only pipeline_tests.yml + pipeline_build.yml, with SKIP_TPU_CI=1.
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        for lane in ("verify", "perf", "integration"):
            self.assertNotIn(f"Uploading .buildkite/pipeline_{lane}.yml", out["stdout"])
        self.assertTrue(any("--context pr-tpu-opt-in" in c for c in calls))

        # `/ci run` API build (BUILDKITE_PULL_REQUEST="false", CI_RUN_TPU="1")
        # and PR build with CI_RUN_TPU="1" both upload the full TPU gate.
        for pr_val in ("false", "123"):
            calls, uploads, _, _ = run_bootstrap(
                BUILDKITE_PULL_REQUEST=pr_val,
                CI_PR_TPU_OPT_IN="1",
                CI_RUN_TPU="1",
                BUILDKITE_MESSAGE="[Docs] Trailing note [skip ci]",
            )
            self.assertEqual(len(uploads), 4)
            self.assertIn("meta-data set JOB_PRIORITY 6", calls)
            self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
            self.assertFalse(any("--context pr-tpu-opt-in" in c for c in calls))

    def test_ci_extras_perf_eval_and_nightly_suite(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            CI_RUN_TPU="1",
            CI_EXTRAS="perf-eval",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertTrue(any("--context perf-eval" in c for c in calls))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)

        for extras in ("nightly-suite", "perf-eval,nightly-suite"):
            calls, uploads, _, _ = run_bootstrap(
                CI_RUN_TPU="1",
                CI_EXTRAS=extras,
            )
            self.assertEqual(len(uploads), 5)
            self.assertIn("meta-data set JOB_PRIORITY 6", calls)
            self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
            self.assertTrue(any("--context full-ci" in c for c in calls))
            for content in uploads:
                self.assertNotIn('build.env("NIGHTLY")', content)

        with self.assertRaises(subprocess.CalledProcessError):
            run_bootstrap(CI_RUN_TPU="1", CI_EXTRAS="unknown-extra")

    def test_merge_queue_main_runs_full_tpu_gate_at_post_merge_priority(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            BUILDKITE_MESSAGE="Update docs [skip ci]",
            CI_PR_TPU_OPT_IN="1",
        )
        self.assertEqual(len(uploads), 4)
        self.assertNotIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 11", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY post-merge", calls)
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)

    def test_merge_queue_release_branch_adds_nightly_suite(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/releases/v0.30.0/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="releases/v0.30.0",
            BUILDKITE_PULL_REQUEST="false",
            CI_PR_TPU_OPT_IN="1",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 11", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY post-merge", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 11\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_merge_queue_reads_latest_bot_marker_clears_and_ignores_non_bot(
        self,
    ) -> None:
        # Bot marker enables perf-eval; subsequent human spoof is ignored.
        comments = json.dumps(
            [
                {
                    "user": {"type": "Bot"},
                    "body": "Queued.\n\n<!-- ci-merge-extras: perf-eval -->",
                },
                {
                    "user": {"type": "User"},
                    "body": "<!-- ci-merge-extras: nightly-suite -->",
                },
            ]
        )
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=comments,
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertTrue(any("--context perf-eval" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)

        # Later empty bot marker clears earlier extras back to the default gate.
        cleared = json.dumps(
            [
                {
                    "user": {"type": "Bot"},
                    "body": "<!-- ci-merge-extras: perf-eval,nightly-suite -->",
                },
                {"user": {"type": "Bot"}, "body": "<!-- ci-merge-extras: -->"},
            ]
        )
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=cleared,
        )
        self.assertEqual(len(uploads), 4)
        self.assertNotIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertFalse(any("annotate" in c for c in calls))

        # On releases/*, marker extras merge with automatic nightly-suite.
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/releases/v0.30.0/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="releases/v0.30.0",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=comments,
        )
        self.assertEqual(len(uploads), 5)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_merge_queue_marker_read_failure_annotates_and_runs_default_gate(
        self,
    ) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS_ERROR="HTTP 503 Service Unavailable",
        )
        self.assertEqual(len(uploads), 4)
        self.assertNotIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertIn(
            "annotate --style warning --context ci-merge-extras "
            "could not read /ci merge extras for PR #42: "
            "HTTP 503 Service Unavailable; ran the default gate",
            calls,
        )

    def test_ci_precheck_uploads_cpu_checks_only_and_wins_over_run_tpu_and_extras(
        self,
    ) -> None:
        # Before #2559 lands (no pipeline_cpu.yml): uses SKIP_TPU_CI=1 on tests.
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="run-full-ci,run-perf-eval",
            BUILDKITE_MESSAGE="[Docs] note [skip ci]",
            CI_PRECHECK="1",
            CI_RUN_TPU="1",
            CI_EXTRAS="perf-eval,nightly-suite",
        )
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        for lane in ("verify", "perf", "integration"):
            self.assertNotIn(f"Uploading .buildkite/pipeline_{lane}.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertTrue(any("--context ci-precheck" in c for c in calls))
        self.assertIn('build.env("NIGHTLY")', uploads[0])

        # Once #2559 lands (pipeline_cpu.yml present): uploads only cpu + build.
        calls, uploads, _, out = run_bootstrap(
            with_pipeline_cpu=True,
            CI_PRECHECK="1",
            CI_RUN_TPU="1",
            CI_EXTRAS="nightly-suite",
        )
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_cpu.yml", out["stdout"])
        self.assertIn("Uploading Build Pipeline", out["stdout"])
        self.assertNotIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        self.assertIn("key: cpu_check", uploads[0])

    def test_merge_queue_docs_only_exits_early_while_code_change_runs_full_gate(
        self,
    ) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-abc1234",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_MERGE_QUEUE_BASE_COMMIT="abc1234def5678",
            BUILDKITE_PULL_REQUEST="false",
            TEST_CHANGED_FILES="docs/guide.md\\nREADME.md\\n.github/CODEOWNERS",
        )
        self.assertEqual(uploads, [])
        self.assertIn("diff --name-only abc1234def5678...HEAD", out["git"])
        self.assertIn(
            "Only documentation/non-code files changed. Skipping CI build.",
            out["stdout"],
        )
        self.assertIn(
            "annotate --style info --context docs-only "
            "Docs-only change; skipped CPU and TPU checks in the merge queue.",
            calls,
        )

        # Fallback when BUILDKITE_MERGE_QUEUE_BASE_COMMIT is unset.
        calls_fb, uploads_fb, _, out_fb = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-abc1234",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_CHANGED_FILES="README.md",
        )
        self.assertEqual(uploads_fb, [])
        self.assertIn("diff --name-only origin/main...HEAD", out_fb["git"])
        self.assertTrue(any("--context docs-only" in c for c in calls_fb))

        # Mixed docs + code change in queue still runs the full gate, and
        # `[skip ci]` in the commit message stays disabled in the queue.
        calls_code, uploads_code, _, out_code = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-abc1234",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_MERGE_QUEUE_BASE_COMMIT="abc1234def5678",
            BUILDKITE_PULL_REQUEST="false",
            BUILDKITE_MESSAGE="Update runner [skip ci]",
            TEST_CHANGED_FILES="docs/guide.md\\nsrc/vllm_torchtpu/runner/tpu_runner.py",
        )
        self.assertEqual(len(uploads_code), 4)
        self.assertIn("diff --name-only abc1234def5678...HEAD", out_code["git"])
        self.assertFalse(any("--context docs-only" in c for c in calls_code))

    def test_merge_queue_precheck_only_runs_cpu_and_ignores_on_releases(
        self,
    ) -> None:
        comments = json.dumps(
            [
                {
                    "user": {"type": "Bot"},
                    "body": (
                        "Queued for merge with prechecks only (no TPU) once "
                        "approved and green.\n\n"
                        "<!-- ci-merge-extras: precheck-only -->\n"
                        "<!-- ci-merge-actor: alice -->"
                    ),
                }
            ]
        )
        # Before #2559 lands (no pipeline_cpu.yml): tests + build at priority 11.
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=comments,
        )
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        for lane in ("verify", "perf", "integration"):
            self.assertNotIn(f"Uploading .buildkite/pipeline_{lane}.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 11", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY post-merge", calls)
        self.assertIn(
            "annotate --style info --context ci-merge-precheck-only "
            "TPU skipped: `/ci merge precheck-only` by @alice",
            calls,
        )
        for content in uploads:
            self.assertTrue(content.startswith("priority: 11\n"))

        # Once #2559 lands (pipeline_cpu.yml present): cpu + build at priority 11.
        calls_cpu, uploads_cpu, _, out_cpu = run_bootstrap(
            with_pipeline_cpu=True,
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=comments,
        )
        self.assertEqual(len(uploads_cpu), 2)
        self.assertIn("Uploading .buildkite/pipeline_cpu.yml", out_cpu["stdout"])
        self.assertIn("Uploading Build Pipeline", out_cpu["stdout"])
        self.assertNotIn("Uploading .buildkite/pipeline_tests.yml", out_cpu["stdout"])
        self.assertIn("key: cpu_check", uploads_cpu[0])
        for content in uploads_cpu:
            self.assertTrue(content.startswith("priority: 11\n"))
        self.assertIn(
            "annotate --style info --context ci-merge-precheck-only "
            "TPU skipped: `/ci merge precheck-only` by @alice",
            calls_cpu,
        )

        # On releases/*, precheck-only is ignored and full gate + nightly-suite runs.
        calls_rel, uploads_rel, _, _ = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/releases/v0.30.0/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="releases/v0.30.0",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=comments,
        )
        self.assertEqual(len(uploads_rel), 5)
        self.assertTrue(any("--context full-ci" in c for c in calls_rel))
        self.assertFalse(
            any("--context ci-merge-precheck-only" in c for c in calls_rel)
        )
        for content in uploads_rel:
            self.assertNotIn('build.env("NIGHTLY")', content)

        # Later empty bot marker clears precheck-only back to the full gate.
        cleared = json.dumps(
            json.loads(comments)
            + [{"user": {"type": "Bot"}, "body": "<!-- ci-merge-extras: -->"}]
        )
        calls_clr, uploads_clr, _, _ = run_bootstrap(
            BUILDKITE_BRANCH="gh-readonly-queue/main/pr-42-deadbeef",
            BUILDKITE_MERGE_QUEUE_BASE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            TEST_GITHUB_COMMENTS=cleared,
        )
        self.assertEqual(len(uploads_clr), 4)
        self.assertFalse(
            any("--context ci-merge-precheck-only" in c for c in calls_clr)
        )


if __name__ == "__main__":
    unittest.main()
