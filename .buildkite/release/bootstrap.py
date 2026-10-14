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
"""Uploads the steps of one vllm-torchtpu release build.

The release pipeline runs this as its only configured step. It checks what the
build was asked to publish and uploads just that mode's steps, so everything a
release runs lives under .buildkite/release/, which has its own code owners.

RELEASE_MODE picks what to publish (default: nightly):
  nightly  The ci, dev and prod images, tagged by branch: main or
           release/X.Y.Z (nightly_images.yml, publish_images.py).
"""

import argparse
import datetime
import os
import pathlib
import subprocess
import sys
from typing import NamedTuple

from publish_images import nightly_tags

RELEASE_DIR = pathlib.Path(__file__).resolve().parent

# Each mode's steps, in this directory.
MODES = {
    "nightly": "nightly_images.yml",
}

# The queue the build steps run on, by the queue this bootstrap runs on. In the
# TPU cluster the bootstrap runs on cpu and the images build on cpu_64_core; in
# TPU Release, whose agents run nothing but release pipelines, one queue does
# both. Any other queue doesn't run releases.
BUILD_QUEUES = {
    "cpu": "cpu_64_core",
    "release-vllm-torchtpu": "release-vllm-torchtpu",
}


class ReleaseError(Exception):
    pass


class Upload(NamedTuple):
    pipeline: pathlib.Path
    # Substituted into the pipeline file by `buildkite-agent pipeline upload`.
    env: dict[str, str]


def plan(mode: str, branch: str, queue: str) -> Upload:
    """Returns the pipeline file to upload and the values it needs."""
    if mode not in MODES:
        raise ReleaseError(
            f"unknown RELEASE_MODE {mode!r}; expected one of {', '.join(MODES)}"
        )
    if queue not in BUILD_QUEUES:
        raise ReleaseError(
            f"queue {queue!r} doesn't run releases; expected one of "
            f"{', '.join(BUILD_QUEUES)}"
        )
    try:
        nightly_tags(branch, datetime.date.today())
    except ValueError as e:
        raise ReleaseError(str(e)) from e

    return Upload(
        pipeline=RELEASE_DIR / MODES[mode],
        env={"RELEASE_QUEUE": BUILD_QUEUES[queue]},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be uploaded instead",
    )
    args = parser.parse_args()

    try:
        upload = plan(
            mode=os.environ.get("RELEASE_MODE") or "nightly",
            branch=os.environ["BUILDKITE_BRANCH"],
            queue=os.environ["BUILDKITE_AGENT_META_DATA_QUEUE"],
        )
    except ReleaseError as e:
        print(f"bootstrap: {e}", file=sys.stderr)
        return 1

    print(
        f"bootstrap: uploading {upload.pipeline.name} with "
        + " ".join(f"{k}={v}" for k, v in upload.env.items())
    )
    if args.dry_run:
        return 0
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload", str(upload.pipeline)],
        env={**os.environ, **upload.env},
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
