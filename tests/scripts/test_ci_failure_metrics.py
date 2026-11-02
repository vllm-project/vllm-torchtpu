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
"""Unit tests for scripts/ci_failure_metrics.py."""

import importlib.util
import pathlib
import unittest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ci_failure_metrics.py"
)
_spec = importlib.util.spec_from_file_location("ci_failure_metrics", _SCRIPT)
ci_failure_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_failure_metrics)


class TestCiFailureMetrics(unittest.TestCase):
    """Test failure categorization rules in ci_failure_metrics."""

    def test_classify_disk_space_full(self) -> None:
        log = "Some build output\nOSError: [Errno 28] No space left on device\n"
        cat_key = ci_failure_metrics.classify_failure(log, step_label="Unit Tests")
        self.assertEqual(cat_key, "infra_disk_full")
        self.assertEqual(
            ci_failure_metrics.CATEGORY_LABELS[cat_key],
            "Infra: Disk Space Full",
        )

        log2 = "Error: docker write failed: disk full during extraction"
        cat_key2 = ci_failure_metrics.classify_failure(
            log2, step_label="Build Dev Image"
        )
        self.assertEqual(cat_key2, "infra_disk_full")

    def test_classify_oom(self) -> None:
        log = "Training or test process killed\nexit status 137\n"
        cat_key = ci_failure_metrics.classify_failure(
            log, exit_code=137, step_label="Integration Test"
        )
        self.assertEqual(cat_key, "infra_oom")

        log_oom = "Killed process 1234 (python3) total-vm: OOMKilled"
        cat_key2 = ci_failure_metrics.classify_failure(log_oom, step_label="Test")
        self.assertEqual(cat_key2, "infra_oom")

    def test_classify_permission_denied(self) -> None:
        log = "Accessing /dev/accel0: Permission denied\n"
        cat_key = ci_failure_metrics.classify_failure(
            log, step_label="Integration Test"
        )
        self.assertEqual(cat_key, "infra_permission")

    def test_classify_agent_lost(self) -> None:
        log = "Job failed unexpectedly.\nThe agent was lost\n"
        cat_key = ci_failure_metrics.classify_failure(log, step_label="Unit Tests")
        self.assertEqual(cat_key, "infra_agent_lost")

    def test_classify_type_check(self) -> None:
        log = "pyrefly: 5 errors found in src/vllm_torchtpu\n"
        cat_key = ci_failure_metrics.classify_failure(log, step_label="Type Check")
        self.assertEqual(cat_key, "code_type_check")

    def test_classify_unit_test_failure(self) -> None:
        log = "FAILED tests/layers/test_attention.py::test_forward\n"
        cat_key = ci_failure_metrics.classify_failure(log, step_label="Unit Tests")
        self.assertEqual(cat_key, "code_test_failure")

    def test_classify_docker_build(self) -> None:
        log = "Step 4/12 : RUN uv pip install\nError: docker build failed"
        cat_key = ci_failure_metrics.classify_failure(
            log, step_label="Build and Push Dev Image"
        )
        self.assertEqual(cat_key, "docker_build_failure")

    def test_classify_pr_gating_missing_ready(self) -> None:
        log = "PR missing ready label"
        cat_key = ci_failure_metrics.classify_failure(
            log, step_label="Bootstrap Pipeline"
        )
        self.assertEqual(cat_key, "pr_gating_missing_ready")

    def test_classify_retried_flaky(self) -> None:
        log = "FAILED tests/test_something.py"
        cat_key = ci_failure_metrics.classify_failure(
            log, step_label="Unit Tests", retried=True
        )
        self.assertEqual(cat_key, "flaky_retry")

    def test_classify_other_fallback(self) -> None:
        log = "Generic unknown script failure with code 1"
        cat_key = ci_failure_metrics.classify_failure(log, step_label="Random Step")
        self.assertEqual(cat_key, "other_failure")


if __name__ == "__main__":
    unittest.main()
