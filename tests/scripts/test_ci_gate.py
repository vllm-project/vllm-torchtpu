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
"""Unit tests for .buildkite/release/ci_gate.py."""

import importlib.util
import pathlib
import unittest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".buildkite"
    / "release"
    / "ci_gate.py"
)
_spec = importlib.util.spec_from_file_location("ci_gate", _SCRIPT)
ci_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_gate)

SHA = "0" * 40
BUILDS = "https://buildkite.com/vllm/vllm-torchtpu-ci/builds/"


def build(number, state):
    return {
        "context": ci_gate.CONTEXT,
        "state": state,
        "target_url": f"{BUILDS}{number}",
    }


def step(number, name, state):
    return {
        "context": f"{ci_gate.CONTEXT}/{name}",
        "state": state,
        "target_url": f"{BUILDS}{number}#01a15db0-{name}",
    }


class TestCiGate(unittest.TestCase):
    """Which status lists (newest first) vouch for a release."""

    def test_green_nightly_build_passes(self) -> None:
        statuses = [
            build(10, "success"),
            step(10, "unit-tests-general-nightly", "success"),
            step(10, "unit-tests-general-nightly", "pending"),
            step(10, "unit-tests-cpu-pr", "success"),
        ]
        self.assertEqual(ci_gate.check(SHA, statuses), (f"{BUILDS}10", None))

    def test_failed_nightly_then_green_plain_build_rejected(self) -> None:
        statuses = [
            build(11, "success"),
            step(11, "unit-tests-cpu-pr", "success"),
            build(10, "failure"),
            step(10, "perf-nightly-qwen3", "failure"),
            step(10, "unit-tests-general-nightly", "success"),
        ]
        url, reason = ci_gate.check(SHA, statuses)
        self.assertIsNone(url)
        self.assertIn("no nightly step ran", reason)

    def test_pr_steps_only_rejected(self) -> None:
        statuses = [
            build(10, "success"),
            step(10, "unit-tests-cpu-pr", "success"),
            step(10, "e2e-pr", "success"),
        ]
        url, reason = ci_gate.check(SHA, statuses)
        self.assertIsNone(url)
        self.assertIn("no nightly step ran", reason)

    def test_newest_build_failed_rejected(self) -> None:
        statuses = [
            build(11, "failure"),
            step(11, "unit-tests-general-nightly", "success"),
            build(10, "success"),
            step(10, "unit-tests-general-nightly", "success"),
        ]
        url, reason = ci_gate.check(SHA, statuses)
        self.assertIsNone(url)
        self.assertIn("is failure, not success", reason)

    def test_nightly_step_failed_on_retry_rejected(self) -> None:
        statuses = [
            build(10, "success"),
            step(10, "unit-tests-general-nightly", "failure"),
            step(10, "unit-tests-general-nightly", "success"),
        ]
        url, _ = ci_gate.check(SHA, statuses)
        self.assertIsNone(url)

    def test_no_build_rejected(self) -> None:
        url, reason = ci_gate.check(SHA, [])
        self.assertIsNone(url)
        self.assertIn("no buildkite/vllm-torchtpu-ci build", reason)

    def test_other_pipeline_nightly_ignored(self) -> None:
        statuses = [
            build(10, "success"),
            {
                "context": "buildkite/vllm-torchtpu-torchtpu-nightly",
                "state": "success",
                "target_url": f"{BUILDS}10#x",
            },
        ]
        url, _ = ci_gate.check(SHA, statuses)
        self.assertIsNone(url)


if __name__ == "__main__":
    unittest.main()
