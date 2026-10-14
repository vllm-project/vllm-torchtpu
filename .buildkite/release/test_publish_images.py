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

import datetime

import pytest
from publish_images import TARGETS, nightly_tags, publish_nightly

TODAY = datetime.date(2026, 10, 13)
REGISTRY = "registry.example/vllm-torchtpu"


@pytest.mark.parametrize(
    "branch, tags",
    [
        ("main", ("nightly-20261013", "latest")),
        ("release/0.0.1", ("20261013-0.0.1", "0.0.1")),
        ("release/1.2.3rc1", ("20261013-1.2.3rc1", "1.2.3rc1")),
    ],
)
def test_nightly_tags(branch, tags):
    assert nightly_tags(branch, TODAY) == tags


@pytest.mark.parametrize(
    "branch, message",
    [
        ("feature", "main or release/X.Y.Z"),
        ("releases/0.0.1", "main or release/X.Y.Z"),
        ("release/latest", "expected release/X.Y.Z"),
        ("release/", "expected release/X.Y.Z"),
    ],
)
def test_nightly_tags_refuses(branch, message):
    with pytest.raises(ValueError, match=message):
        nightly_tags(branch, TODAY)


def test_publish_nightly_matches_the_shell_script_it_replaced():
    commands = []
    publish_nightly(
        "main",
        TODAY,
        REGISTRY,
        runner=lambda cmd, check=True: commands.append((list(cmd), check)),
    )

    expected = [(["bash", ".buildkite/scripts/cleanup_docker.sh"], True)]
    for target in TARGETS:
        new = f"{REGISTRY}/vllm-torchtpu-{target}"
        old = f"{REGISTRY}/torchtpu-vllm-{target}"
        images = [
            f"{new}:nightly-20261013",
            f"{new}:latest",
            f"{old}:nightly-20261013",
            f"{old}:latest",
        ]
        built = images[0]
        expected += [
            (["./docker/build_image.sh", "--target", target, "-t", built], True),
            *[(["docker", "tag", built, image], True) for image in images[1:]],
            *[(["docker", "push", image], True) for image in images],
            (["docker", "rmi", *images], False),
        ]
    assert commands == expected


def test_publish_nightly_refuses_before_building():
    commands = []
    with pytest.raises(ValueError):
        publish_nightly("feature", TODAY, REGISTRY, runner=commands.append)
    assert commands == []
