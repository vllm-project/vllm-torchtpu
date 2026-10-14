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
"""Builds the vllm-torchtpu ci, dev and prod images and pushes them.

nightly: the branch decides the tags. main pushes nightly-YYYYMMDD and moves
latest. release/X.Y.Z pushes YYYYMMDD-X.Y.Z and moves X.Y.Z, so a release
nightly never touches latest. A release branch cut from main therefore needs
only a schedule on the release pipeline, not a code change.
"""

import argparse
import datetime
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Protocol

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu"

# In dependency order (ci -> dev, prod), so BuildKit reuses the layers.
TARGETS = ("ci", "dev", "prod")


class Runner(Protocol):
    def __call__(self, cmd: Sequence[str], check: bool = True) -> None: ...


def run(cmd: Sequence[str], check: bool = True) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=check, cwd=REPO_ROOT)


def nightly_tags(branch: str, today: datetime.date) -> tuple[str, str]:
    """Returns the dated tag and the moving tag for a nightly of branch."""
    if branch == "main":
        return f"nightly-{today:%Y%m%d}", "latest"
    if branch.startswith("release/"):
        version = branch.removeprefix("release/")
        # Starting with a digit keeps it from colliding with latest or
        # nightly-*.
        if not re.fullmatch(r"[0-9][A-Za-z0-9_.-]*", version):
            raise ValueError(
                f"branch {branch!r} does not name a version usable as an "
                "image tag; expected release/X.Y.Z"
            )
        return f"{today:%Y%m%d}-{version}", version
    raise ValueError(
        f"nightly images are published from main or release/X.Y.Z, not {branch!r}"
    )


def publish_nightly(
    branch: str, today: datetime.date, registry: str, runner: Runner = run
) -> None:
    dated_tag, moving_tag = nightly_tags(branch, today)

    print("--- Cleaning up old Docker images and cache", flush=True)
    runner(["bash", ".buildkite/scripts/cleanup_docker.sh"])

    for target in TARGETS:
        # The torchtpu-vllm-* names are the images' old names, pushed too so
        # consumers that still pull them keep working until they move.
        names = (f"vllm-torchtpu-{target}", f"torchtpu-vllm-{target}")
        images = [
            f"{registry}/{name}:{tag}"
            for name in names
            for tag in (dated_tag, moving_tag)
        ]
        built = images[0]

        print(f"--- :docker: Building {built}", flush=True)
        runner(["./docker/build_image.sh", "--target", target, "-t", built])
        for image in images[1:]:
            runner(["docker", "tag", built, image])

        # Each name's dated tag before its moving tag, so a moving tag only
        # moves once its image is published under a name that won't change.
        print(f"--- :docker: Pushing {target}", flush=True)
        for image in images:
            runner(["docker", "push", image])

        runner(["docker", "rmi", *images], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["nightly"])
    parser.parse_args()

    branch = (
        os.environ.get("BUILDKITE_BRANCH")
        or subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    try:
        publish_nightly(
            branch=branch,
            today=datetime.datetime.now(datetime.UTC).date(),
            registry=os.environ.get("NIGHTLY_REGISTRY") or DEFAULT_REGISTRY,
        )
    except ValueError as e:
        print(f"publish_images: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
