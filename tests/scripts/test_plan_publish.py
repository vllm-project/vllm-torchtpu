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
"""Unit tests for .buildkite/release/plan_publish.py."""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import unittest
from unittest import mock

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".buildkite"
    / "release"
    / "plan_publish.py"
)
_spec = importlib.util.spec_from_file_location("plan_publish", _SCRIPT)
plan_publish = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan_publish)


def plan(version, **env):
    """The steps plan_publish prints for a staged release of `version`."""
    prerelease = any(tag in version for tag in ("a", "b", "rc", "dev"))
    metadata = {
        "release-version": version,
        "release-prerelease": "yes" if prerelease else "no",
        "release-image-prod": "us-docker.pkg.dev/x/prod@sha256:abc",
    }
    out = io.StringIO()
    with (
        mock.patch.object(plan_publish, "meta", metadata.__getitem__),
        mock.patch.dict(os.environ, {"BUILDKITE_BRANCH": "releases/v1.2", **env}),
        contextlib.redirect_stdout(out),
    ):
        if "RELEASE_APPROVER_TEAMS" not in env:
            os.environ.pop("RELEASE_APPROVER_TEAMS", None)
        plan_publish.main()
    return {s["key"]: s for s in json.loads(out.getvalue())["steps"]}


def latest_field(steps):
    fields = steps["approve_publish"]["fields"]
    return next(f for f in fields if f["key"] == "dockerhub_move_latest")


class TestPlanPublish(unittest.TestCase):
    """The approval block and publish steps of a release build."""

    def test_final_release_can_move_latest(self) -> None:
        field = latest_field(plan("1.2.0"))
        self.assertEqual([o["value"] for o in field["options"]], ["yes", "no"])
        self.assertEqual(field["default"], "yes")

    def test_prerelease_cannot_move_latest(self) -> None:
        for version in ("1.2.0rc1", "1.2.0a1", "1.2.0b2", "1.2.0.dev3"):
            field = latest_field(plan(version))
            self.assertEqual([o["value"] for o in field["options"]], ["no"], version)
            self.assertEqual(field["default"], "no", version)

    def test_approver_teams(self) -> None:
        block = plan("1.2.0", RELEASE_APPROVER_TEAMS="a,b")["approve_publish"]
        self.assertEqual(block["allowed_teams"], ["a", "b"])
        self.assertNotIn("allowed_teams", plan("1.2.0")["approve_publish"])

    def test_publish_order(self) -> None:
        steps = plan("1.2.0")
        self.assertEqual(steps["publish_images"]["depends_on"], ["approve_publish"])
        self.assertEqual(steps["publish_core"]["depends_on"], ["approve_publish"])
        self.assertEqual(steps["publish_vllm_torchtpu"]["depends_on"], ["publish_core"])
        self.assertEqual(
            steps["verify_published"]["depends_on"],
            ["publish_images", "publish_vllm_torchtpu"],
        )


if __name__ == "__main__":
    unittest.main()
