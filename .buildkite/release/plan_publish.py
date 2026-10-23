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
"""Print the second half of a release build as a Buildkite pipeline (JSON).

    plan_publish.py | buildkite-agent pipeline upload --no-interpolation

Runs once the images are staged and the wheels are checked: the approval
block, the Docker Hub and PyPI publishes, and the check that what users get is
what was built. Uploaded rather than written into pipeline.yml so the approval
prompt can name the exact version and prod digest, and so the team restriction
is added only when it is set. --no-interpolation: nothing in it is for
Buildkite to interpolate.

Optional build env:
  RELEASE_APPROVER_TEAMS=a,b   Buildkite team slugs allowed to approve
"""

import json
import os
import subprocess


def meta(key):
    return subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


QUEUE = {"queue": os.environ.get("RELEASE_QUEUE", "release-vllm-torchtpu")}


def command_step(label, key, depends_on, commands, **extra):
    step = {
        "label": label,
        "key": key,
        "depends_on": depends_on,
        "agents": QUEUE,
        "commands": commands,
    }
    step.update(extra)
    return step


def main():
    version = meta("release-version")
    prod = meta("release-image-prod")
    prerelease = meta("release-prerelease") == "yes"
    branch = os.environ["BUILDKITE_BRANCH"]
    dockerhub_repo = os.environ.get("RELEASE_DOCKERHUB_REPO", "vllm/vllm-torchtpu")
    steps = []

    latest_options = [{"label": "No", "value": "no"}]
    if not prerelease:
        latest_options.insert(0, {"label": "Yes", "value": "yes"})
    block = {
        "block": f":rocket: Publish vllm-torchtpu {version}?",
        "key": "approve_publish",
        "depends_on": ["plan_publish"],
        "prompt": (
            f"Publish vllm-torchtpu {version} from {branch}: the staged prod image "
            f"{prod} to Docker Hub as {dockerhub_repo}:v{version}, "
            f"and vllm-torchtpu-core and vllm-torchtpu {version} to PyPI. "
            "PyPI releases can't be replaced."
        ),
        "fields": [
            {
                "select": "Move latest on Docker Hub to this version?",
                "key": "dockerhub_move_latest",
                "required": True,
                "default": latest_options[0]["value"],
                "options": latest_options,
                "hint": "No for a patch to an older line. "
                "A pre-release never moves latest.",
            },
            {
                "select": f"Overwrite v{version} if it is already on Docker Hub?",
                "key": "dockerhub_overwrite",
                "required": True,
                "default": "no",
                "options": [
                    {"label": "No", "value": "no"},
                    {"label": "Yes", "value": "yes"},
                ],
                "hint": "Released tags should not change; "
                "Yes only to replace a broken image.",
            },
        ],
    }
    teams = [t for t in os.environ.get("RELEASE_APPROVER_TEAMS", "").split(",") if t]
    if teams:
        block["allowed_teams"] = teams
    steps.append(block)

    steps.append(
        command_step(
            ":docker: Publish the prod image to Docker Hub",
            "publish_images",
            ["approve_publish"],
            ["bash .buildkite/release/publish_images.sh"],
            secrets=["DOCKERHUB_TOKEN_VLLM_TORCHTPU"],
            concurrency=1,
            concurrency_group="vllm-torchtpu-dockerhub-publish",
        )
    )
    steps.append(
        command_step(
            ":package: Publish vllm-torchtpu-core to PyPI",
            "publish_core",
            ["approve_publish"],
            ["bash .buildkite/scripts/publish_wheel.sh core"],
            env={"PUBLISH_PYPI": "vllm-torchtpu-core"},
            concurrency=1,
            concurrency_group="vllm-torchtpu-core-publish",
        )
    )
    # The combined wheel pins vllm-torchtpu-core==<version>, so it goes up
    # only once the core is installable.
    steps.append(
        command_step(
            ":package: Publish vllm-torchtpu to PyPI",
            "publish_vllm_torchtpu",
            ["publish_core"],
            ["bash .buildkite/scripts/publish_wheel.sh vllm-torchtpu"],
            env={"PUBLISH_PYPI": "vllm-torchtpu"},
            concurrency=1,
            concurrency_group="vllm-torchtpu-publish",
        )
    )
    steps.append(
        command_step(
            ":white_check_mark: Check what users get",
            "verify_published",
            ["publish_images", "publish_vllm_torchtpu"],
            ["bash .buildkite/release/verify_published.sh"],
        )
    )

    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
