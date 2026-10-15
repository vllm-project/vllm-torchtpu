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
"""Lightweight CI/CD failure categorizer and metric recorder for Buildkite.

Classifies failure root causes into:
- infra_disk_full: Runner disk space exhausted ("No space left on device")
- infra_oom: Process or container killed by OOM (exit code 137 or OOMKilled)
- infra_permission: Permission denied / filesystem permission errors
- infra_agent_lost: Buildkite agent disconnected or host died
- code_type_check: Type checker failure (mypy/pyright/pyrefly)
- code_test_failure: Pytest / integration test regression
- docker_build_failure: Container image build / push failure
- pr_gating_missing_ready: Missing 'ready' label on PR
- flaky_retry: Passed on retry
- other_failure: Unclassified command or script error

Emits classification to Buildkite step metadata and uploads record to
BigQuery when configured.
"""

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# Shared BigQuery utilities live with the benchmarking scripts
_BENCHMARKING_DIR = str(
    Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking"
)
if _BENCHMARKING_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKING_DIR)

try:
    import bq_utils
except ImportError:
    bq_utils = None

DEFAULT_BQ_FAILURE_TABLE = (
    "cloud-ullm-inference-ci-cd.llm_benchmark_analytics.ci_failures"
)

CATEGORY_LABELS = {
    "infra_disk_full": "Infra: Disk Space Full",
    "infra_oom": "Infra: OOM (Exit 137)",
    "infra_permission": "Infra: Permission Denied",
    "infra_agent_lost": "Infra: Agent Lost",
    "code_type_check": "Code: Type Check Failure (mypy/pyright/pyrefly)",
    "code_test_failure": "Code: Unit / Integration Test Failure",
    "docker_build_failure": "Build: Docker Image Build Failure",
    "pr_gating_missing_ready": "PR Gating: Missing 'ready' label",
    "flaky_retry": "Flaky Test (Passed on Retry)",
    "other_failure": "Other Test / Script Failure",
}


def classify_failure(
    log_text: str,
    exit_code: int = 1,
    step_label: str = "",
    retried: bool = False,
    tail_lines: int = 80,
) -> str:
    """Classify the failure category from log text, exit code, and step label."""
    if retried:
        return "flaky_retry"

    if "Bootstrap Pipeline" in step_label:
        return "pr_gating_missing_ready"

    tail = "\n".join(log_text.splitlines()[-tail_lines:])

    # 1. Exit code specific signatures
    if exit_code == 137 or "OOMKilled" in tail or "exit status 137" in tail:
        return "infra_oom"

    # 2. Infrastructure signatures
    if (
        "No space left on device" in log_text
        or "no space left on device" in log_text
        or "disk full" in log_text.lower()
    ):
        return "infra_disk_full"

    if "Permission denied" in tail or "EACCES" in tail:
        return "infra_permission"

    if "The agent was lost" in tail:
        return "infra_agent_lost"

    # 3. Code & test signatures
    if (
        "Type Check" in step_label
        or "errors (6 suppressed" in tail
        or "errors in " in tail
        or "pyrefly" in tail
        or "mypy" in tail
    ):
        return "code_type_check"

    if (
        "Failed Tests:" in log_text
        or "pytest exited with error" in tail
        or "FAILED tests/" in log_text
        or "AssertionError" in tail
    ):
        return "code_test_failure"

    if "docker build" in tail.lower() or "Build and Push" in step_label:
        return "docker_build_failure"

    return "other_failure"


def set_buildkite_metadata(category: str) -> None:
    """Set failure metadata in Buildkite if buildkite-agent CLI is available."""
    if not shutil.which("buildkite-agent"):
        return

    try:
        subprocess.run(
            ["buildkite-agent", "meta-data", "set", "failure_category", category],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except Exception as e:
        print(f"[WARN] Failed to set Buildkite metadata: {e}", file=sys.stderr)


def upload_to_bigquery(
    category: str,
    exit_code: int,
    step_label: str,
    project_id: str | None = None,
    table: str | None = None,
) -> bool:
    """Best-effort insert of failure record into BigQuery."""
    target_table = (
        table
        if table is not None
        else os.getenv("BQ_FAILURE_TABLE", DEFAULT_BQ_FAILURE_TABLE)
    )
    if not target_table or target_table.lower() in ("none", "skip", "disabled", "0"):
        return False
    if bq_utils is None:
        return False

    target_project = (
        project_id or os.getenv("BQ_PROJECT_ID") or bq_utils.DEFAULT_BQ_PROJECT_ID
    )

    record_id = str(uuid.uuid4())
    build_id = os.getenv("BUILDKITE_BUILD_ID", "")
    build_number = os.getenv("BUILDKITE_BUILD_NUMBER", "0")
    job_id = os.getenv("BUILDKITE_JOB_ID", "")
    pipeline = os.getenv("BUILDKITE_PIPELINE_SLUG", "")
    branch = os.getenv("BUILDKITE_BRANCH", "")
    commit = os.getenv("BUILDKITE_COMMIT", "")

    sql = f"""
    INSERT INTO `{target_table.strip("`")}` (
        record_id, created_time, build_id, build_number, job_id,
        step_label, failure_category, exit_code, pipeline_slug, branch,
        commit_hash
    ) VALUES (
        '{bq_utils.sql_escape(record_id)}', CURRENT_TIMESTAMP(),
        '{bq_utils.sql_escape(build_id)}',
        {int(build_number) if build_number.isdigit() else 0},
        '{bq_utils.sql_escape(job_id)}',
        '{bq_utils.sql_escape(step_label)}',
        '{bq_utils.sql_escape(category)}', {exit_code},
        '{bq_utils.sql_escape(pipeline)}',
        '{bq_utils.sql_escape(branch)}',
        '{bq_utils.sql_escape(commit)}'
    );
    """
    clean_sql = " ".join(sql.split())
    return bq_utils.run_insert(
        clean_sql,
        project=target_project,
        label=f"failure_{category}",
        record_id=record_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify CI failures and record metrics."
    )
    parser.add_argument("--log-file", help="Path to error log file")
    parser.add_argument("--exit-code", type=int, default=1, help="Command exit code")
    parser.add_argument(
        "--step-label",
        default=os.getenv("BUILDKITE_LABEL", ""),
        help="Buildkite step label",
    )
    parser.add_argument(
        "--bq-project",
        default=os.getenv("BQ_PROJECT_ID"),
        help="BigQuery project ID",
    )
    parser.add_argument(
        "--bq-table",
        default=os.getenv("BQ_FAILURE_TABLE", DEFAULT_BQ_FAILURE_TABLE),
        help="BigQuery failure table",
    )
    args = parser.parse_args()

    log_content = ""
    if args.log_file and os.path.exists(args.log_file):
        try:
            with open(args.log_file, errors="replace") as f:
                log_content = f.read()
        except Exception as e:
            print(
                f"[WARN] Could not read log file {args.log_file}: {e}",
                file=sys.stderr,
            )

    category = classify_failure(
        log_content,
        exit_code=args.exit_code,
        step_label=args.step_label,
    )

    # Print diagnostic banner
    human_label = CATEGORY_LABELS.get(category, category)
    print(f"+++ :warning: CI Failure Classification: {human_label} ({category})")
    if category == "infra_disk_full":
        print(
            "Detected runner disk exhaustion ('No space left on device'). "
            "Flagged as infrastructure failure."
        )

    # Export to Buildkite metadata
    set_buildkite_metadata(category)

    # Best-effort upload to BigQuery
    upload_to_bigquery(
        category=category,
        exit_code=args.exit_code,
        step_label=args.step_label,
        project_id=args.bq_project,
        table=args.bq_table,
    )


if __name__ == "__main__":
    main()
