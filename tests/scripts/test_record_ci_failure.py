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
"""Unit tests for .buildkite/scripts/record_ci_failure.py."""

import importlib.util
import pathlib
import unittest
from unittest import mock

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".buildkite"
    / "scripts"
    / "record_ci_failure.py"
)
_spec = importlib.util.spec_from_file_location("record_ci_failure", _SCRIPT)
record_ci_failure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_ci_failure)


class TestRecordCiFailure(unittest.TestCase):
    """Test classification and recording behavior in record_ci_failure."""

    def test_classify_disk_full(self) -> None:
        log = "Writing cache...\nOSError: [Errno 28] No space left on device\n"
        cat = record_ci_failure.classify_failure(log, exit_code=1)
        self.assertEqual(cat, "infra_disk_full")

        log2 = "failed to register layer: devmapper: disk full"
        cat2 = record_ci_failure.classify_failure(log2, exit_code=1)
        self.assertEqual(cat2, "infra_disk_full")

    def test_classify_oom(self) -> None:
        cat = record_ci_failure.classify_failure("some logs", exit_code=137)
        self.assertEqual(cat, "infra_oom")

        cat_oom = record_ci_failure.classify_failure("Task OOMKilled", exit_code=1)
        self.assertEqual(cat_oom, "infra_oom")

    def test_classify_permission_denied(self) -> None:
        log = "touch: cannot touch '/persist/file': Permission denied\n"
        cat = record_ci_failure.classify_failure(log, exit_code=1)
        self.assertEqual(cat, "infra_permission")

    def test_classify_agent_lost(self) -> None:
        log = "Build failed\nThe agent was lost\n"
        cat = record_ci_failure.classify_failure(log, exit_code=1)
        self.assertEqual(cat, "infra_agent_lost")

    def test_classify_type_check(self) -> None:
        cat = record_ci_failure.classify_failure(
            "Errors found", exit_code=1, step_label="Run Type Check"
        )
        self.assertEqual(cat, "code_type_check")

    def test_classify_test_failure(self) -> None:
        log = "FAILED tests/test_model.py::test_eval - AssertionError\n"
        cat = record_ci_failure.classify_failure(log, exit_code=1)
        self.assertEqual(cat, "code_test_failure")

    def test_classify_docker_build(self) -> None:
        log = "docker build failed with status 2\n"
        cat = record_ci_failure.classify_failure(
            log, exit_code=1, step_label="Build Docker"
        )
        self.assertEqual(cat, "docker_build_failure")

    def test_classify_other_failure(self) -> None:
        cat = record_ci_failure.classify_failure(
            "unknown failure", exit_code=2, step_label="custom"
        )
        self.assertEqual(cat, "other_failure")

    def test_category_labels(self) -> None:
        self.assertIn("infra_disk_full", record_ci_failure.CATEGORY_LABELS)
        self.assertEqual(
            record_ci_failure.CATEGORY_LABELS["infra_disk_full"],
            "Infra: Disk Space Full",
        )

    @mock.patch("shutil.which", return_value=None)
    def test_set_buildkite_metadata_no_agent(self, mock_which: mock.MagicMock) -> None:
        # Should not raise exception when buildkite-agent is missing
        record_ci_failure.set_buildkite_metadata("infra_disk_full")
        mock_which.assert_called_with("buildkite-agent")

    def test_upload_to_bigquery_disabled(self) -> None:
        # With table explicitly empty or 'none', skips without running insert
        self.assertFalse(
            record_ci_failure.upload_to_bigquery("infra_disk_full", 1, "Test", table="")
        )
        self.assertFalse(
            record_ci_failure.upload_to_bigquery(
                "infra_disk_full", 1, "Test", table="none"
            )
        )

    def test_upload_to_bigquery_success(self) -> None:
        if record_ci_failure.bq_utils is None:
            self.skipTest("bq_utils not available")
        with mock.patch.object(
            record_ci_failure.bq_utils, "run_insert", return_value=True
        ) as mock_insert:
            res = record_ci_failure.upload_to_bigquery(
                "infra_disk_full", 1, "Test Step"
            )
            self.assertTrue(res)
            mock_insert.assert_called_once()
            sql_arg = mock_insert.call_args[0][0]
            self.assertIn("ci_failures", sql_arg)
            self.assertIn("infra_disk_full", sql_arg)


if __name__ == "__main__":
    unittest.main()
