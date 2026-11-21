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
"""What .buildkite/scripts/publish_nightly_images.sh builds and pushes.

The script runs for real, in a throwaway git repository holding copies of the
scripts it calls, with `docker`, `buildkite-agent` and `date` replaced on PATH
by stubs. The docker stub records every invocation, so the tests read the
build and push commands the nightly would have run.
"""

import os
import pathlib
import shlex
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_COPIED = (
    ".buildkite/scripts/publish_nightly_images.sh",
    ".buildkite/scripts/cleanup_docker.sh",
    ".buildkite/release/lib.sh",
    "docker/build_image.sh",
    "pyproject.toml",
)
_REGISTRY = "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu"
_REVISION_LABEL = "org.opencontainers.image.revision"

_DOCKER_STUB = """#!/bin/bash
printf '%q ' "$@" >> "$DOCKER_LOG"
echo >> "$DOCKER_LOG"
"""
_DATE_STUB = """#!/bin/bash
if [ "$*" = "+%Y%m%d" ]; then echo 20261118; else exec /bin/date "$@"; fi
"""
_BUILDKITE_AGENT_STUB = """#!/bin/bash
exit 1
"""


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    ).stdout.strip()


@pytest.fixture
def nightly(tmp_path):
    """Runs the publish script in a one-commit repository on `main`.

    Returns ``(run, head, repo)`` where ``run(**env)`` returns
    ``(completed process, docker invocations as argv lists)``, ``head`` is
    the repository's HEAD commit, and ``repo`` is the throwaway checkout path.
    """
    repo = tmp_path / "repo"
    for relative in _COPIED:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_REPO / relative, destination)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "nightly source")
    head = _git(repo, "rev-parse", "HEAD")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("docker", _DOCKER_STUB),
        ("date", _DATE_STUB),
        ("buildkite-agent", _BUILDKITE_AGENT_STUB),
    ):
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    docker_log = tmp_path / "docker.log"

    def run(**env: str):
        docker_log.write_text("")
        process_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("BUILDKITE", "DOCKERHUB", "NIGHTLY_", "RELEASE_"))
            and key != "PUBLISH_DOCKERHUB"
        }
        process_env.update(
            PATH=f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            DOCKER_LOG=str(docker_log),
            BUILDKITE_BRANCH="main",
            **env,
        )
        completed = subprocess.run(
            ["bash", str(repo / ".buildkite/scripts/publish_nightly_images.sh")],
            capture_output=True,
            text=True,
            env=process_env,
        )
        calls = [shlex.split(line) for line in docker_log.read_text().splitlines()]
        return completed, calls

    return run, head, repo


def _builds(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call[0] == "build"]


def _labels(build: list[str]) -> list[str]:
    return [build[i + 1] for i, arg in enumerate(build) if arg == "--label"]


def _option(build: list[str], name: str) -> str:
    return build[build.index(name) + 1]


def test_each_nightly_image_is_labeled_with_the_commit_it_was_built_from(nightly):
    run, head, _ = nightly

    completed, calls = run(BUILDKITE_COMMIT=head)

    assert completed.returncode == 0, completed.stderr
    assert [
        (_option(build, "--target"), _option(build, "-t"), _labels(build))
        for build in _builds(calls)
    ] == [
        (
            target,
            f"{_REGISTRY}/vllm-torchtpu-{target}:nightly-20261118",
            [f"{_REVISION_LABEL}={head}"],
        )
        for target in ("ci", "dev", "prod")
    ]


def test_nightly_tags_are_unchanged(nightly):
    run, head, _ = nightly

    completed, calls = run(BUILDKITE_COMMIT=head)

    assert completed.returncode == 0, completed.stderr
    assert [call[1] for call in calls if call[0] == "push"] == [
        f"{_REGISTRY}/{name}-{target}:{tag}"
        for target in ("ci", "dev", "prod")
        for name in ("vllm-torchtpu", "torchtpu-vllm")
        for tag in ("nightly-20261118", "latest")
    ]


@pytest.mark.parametrize("buildkite_commit", ["HEAD", "", "0" * 40])
def test_label_is_the_checked_out_commit_whatever_buildkite_reports(
    nightly, buildkite_commit
):
    run, head, _ = nightly

    completed, calls = run(BUILDKITE_COMMIT=buildkite_commit)

    assert completed.returncode == 0, completed.stderr
    assert {tuple(_labels(build)) for build in _builds(calls)} == {
        (f"{_REVISION_LABEL}={head}",)
    }


def test_label_falls_back_to_buildkite_commit_without_git_metadata(nightly):
    run, _, repo = nightly
    shutil.rmtree(repo / ".git")
    commit = "0123456789abcdef0123456789abcdef01234567"

    completed, calls = run(
        BUILDKITE_COMMIT=commit, GIT_CEILING_DIRECTORIES=str(repo.parent)
    )

    assert completed.returncode == 0, completed.stderr
    assert {tuple(_labels(build)) for build in _builds(calls)} == {
        (f"{_REVISION_LABEL}={commit}",)
    }


def test_nothing_is_built_when_the_commit_is_unknown(nightly):
    run, _, repo = nightly
    shutil.rmtree(repo / ".git")

    completed, calls = run(
        BUILDKITE_COMMIT="HEAD", GIT_CEILING_DIRECTORIES=str(repo.parent)
    )

    assert completed.returncode == 1
    assert "cannot tell which commit the images are built from" in completed.stderr
    assert calls == []


def test_dockerhub_nightly_tag_names_the_checked_out_commit(nightly):
    run, head, repo = nightly
    # bootstrap.py imports sibling CI modules; only its vllm-pin output matters.
    (repo / ".buildkite/scripts/bootstrap.py").write_text(
        'print("0123456789abcdef0123456789abcdef01234567")\n'
    )

    completed, calls = run(
        BUILDKITE_COMMIT="HEAD",
        PUBLISH_DOCKERHUB="1",
        DOCKERHUB_REPO="vllm/vllm-torchtpu",
        DOCKERHUB_USERNAME="u",
        DOCKERHUB_TOKEN_VLLM_TORCHTPU="t",
    )

    assert completed.returncode == 0, completed.stderr
    pushes = [call[1] for call in calls if call[0] == "push"]
    assert any(
        image == f"vllm/vllm-torchtpu:nightly-20261118-{head[:7]}-0123456"
        for image in pushes
    ), pushes
