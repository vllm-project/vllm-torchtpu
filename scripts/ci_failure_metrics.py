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
"""High-speed CI/CD failure root-cause analysis for Buildkite.

Fetches failed jobs in parallel and accurately classifies failures using the
shared classifier in .buildkite/scripts/record_ci_failure.py:
1. Unit / Integration Test Failures (pytest assertions, e2e tests)
2. PR Gating (Missing 'ready' label)
3. Type Check Failures (mypy/pyright/pyrefly)
4. Docker Image Build Failures
5. Infrastructure Issues (Disk full, permissions, agent lost, OOM)
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import unified classification rules from CI runner scripts
_BK_SCRIPTS = str(Path(__file__).resolve().parents[1] / ".buildkite" / "scripts")
if _BK_SCRIPTS not in sys.path:
    sys.path.insert(0, _BK_SCRIPTS)

from record_ci_failure import CATEGORY_LABELS, classify_failure  # noqa: E402

BUILDKITE_API_BASE = "https://api.buildkite.com/v2"
DEFAULT_ORG = "vllm"
DEFAULT_PIPELINE = "vllm-torchtpu-ci"


def fetch_json(url: str, token: str) -> dict | list | None:
    """Fetch JSON payload from Buildkite API."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "vllm-tpu-ci-analyzer/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_log(url: str, token: str) -> str:
    """Fetch raw text log from Buildkite API."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "vllm-tpu-ci-analyzer/1.0",
            "Accept": "text/plain",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def analyze_weekly_ci(
    org: str,
    pipeline: str,
    days: int,
    token: str,
) -> None:
    """Fetch and aggregate failure categories for the given pipeline."""
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)  # noqa: UP017
    since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Fetching builds for '{pipeline}' since {since_iso[:10]}...")

    url = (
        f"{BUILDKITE_API_BASE}/organizations/{org}/pipelines/{pipeline}/"
        f"builds?created_from={since_iso}&per_page=100"
    )
    builds = fetch_json(url, token)
    if not builds or not isinstance(builds, list):
        print("No builds found or unable to connect.")
        return

    failed_jobs = []
    for b in builds:
        bnum = b.get("number")
        for j in b.get("jobs", []):
            if j.get("state") == "failed":
                failed_jobs.append(
                    (
                        bnum,
                        j.get("id"),
                        j.get("name") or j.get("type", "Unknown"),
                        j.get("retried", False),
                    )
                )

    total_failures = len(failed_jobs)
    print(
        f"Analyzing {total_failures} failed jobs across {len(builds)} "
        "builds in parallel..."
    )

    if total_failures == 0:
        print("No failed jobs found in the specified time window.")
        return

    def classify_job(item: tuple[int, str, str, bool]) -> str:
        bnum, jid, jname, retried = item
        if retried:
            return CATEGORY_LABELS["flaky_retry"]
        if "Bootstrap Pipeline" in jname:
            return CATEGORY_LABELS["pr_gating_missing_ready"]

        log_url = (
            f"{BUILDKITE_API_BASE}/organizations/{org}/pipelines/{pipeline}/"
            f"builds/{bnum}/jobs/{jid}/log"
        )
        text = fetch_log(log_url, token)
        category_key = classify_failure(
            text,
            exit_code=1,
            step_label=jname,
            retried=retried,
        )
        return CATEGORY_LABELS.get(category_key, category_key)

    with ThreadPoolExecutor(max_workers=15) as executor:
        categories = list(executor.map(classify_job, failed_jobs))

    cat_counts = Counter(categories)

    print("\n" + "=" * 68)
    print("        WEEKLY CI/CD FAILURE CLASSIFICATION REPORT")
    print(f"         Time Range: Past {days} Days ({since_iso[:10]} to Present)")
    print("=" * 68)
    print(f"{'Total Builds Analyzed:':<38} {len(builds)}")
    print(f"{'Total Failed Jobs:':<38} {total_failures}")
    print("-" * 68)
    print(f"{'CATEGORY':<38} {'COUNT':<10} {'% OF FAILURES':<15}")
    print("-" * 68)

    infra_keys = [k for k in cat_counts if k.startswith("Infra:")]
    infra_total = sum(cat_counts[k] for k in infra_keys)

    infra_pct = (infra_total / total_failures) * 100
    print(f"{'1. Infrastructure Issues':<38} {infra_total:<10} {infra_pct:.1f}%")
    if infra_total == 0:
        print(f"   • {'Disk full / Permissions / OOM':<35} 0          0.0%")
    else:
        for k in infra_keys:
            c = cat_counts[k]
            c_pct = (c / total_failures) * 100
            print(f"   • {k:<35} {c:<10} {c_pct:.1f}%")

    code_keys = [k for k in cat_counts if not k.startswith("Infra:")]
    for k in sorted(code_keys, key=lambda x: cat_counts[x], reverse=True):
        c = cat_counts[k]
        c_pct = (c / total_failures) * 100
        print(f"{k:<38} {c:<10} {c_pct:.1f}%")

    print("-" * 68)
    print("KEY TAKEAWAYS FOR LEADERSHIP:")
    print(
        f" • Infra failures accounted for {infra_pct:.1f}% of failed jobs on "
        "this pipeline."
    )
    gating_label = CATEGORY_LABELS["pr_gating_missing_ready"]
    gating_count = cat_counts.get(gating_label, 0)
    gating_pct = (gating_count / total_failures) * 100
    print(
        f" • PR gating (missing 'ready' label) caused {gating_pct:.1f}% "
        "of build failures."
    )
    code_test = cat_counts.get(
        CATEGORY_LABELS["code_test_failure"], 0
    ) + cat_counts.get(CATEGORY_LABELS["code_type_check"], 0)
    code_pct = (code_test / total_failures) * 100
    print(f" • Pure code/test regressions accounted for {code_pct:.1f}% of failures.")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast parallel CI failure analysis for Buildkite."
    )
    parser.add_argument(
        "--token",
        default=os.getenv("BUILDKITE_API_TOKEN"),
        help="Buildkite API token",
    )
    parser.add_argument(
        "--org",
        default=DEFAULT_ORG,
        help="Buildkite org slug (default: vllm)",
    )
    parser.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE,
        help="Pipeline slug (default: vllm-torchtpu-ci)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Past days (default: 7)",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "Please provide --token <BUILDKITE_API_TOKEN> or set "
            "BUILDKITE_API_TOKEN environment variable."
        )
        sys.exit(1)

    analyze_weekly_ci(args.org, args.pipeline, args.days, args.token)


if __name__ == "__main__":
    main()
