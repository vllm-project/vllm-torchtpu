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
"""What the image scripts build and push, and the commit label they set.

Covers .buildkite/scripts/publish_nightly_images.sh,
.buildkite/release/build_images.sh, the commit .buildkite/release/preflight.sh
gates a release on, and the `--label` option of docker/build_image.sh. Each
script runs for real, in a throwaway git repository holding copies of the
scripts it calls, with `docker`, `buildkite-agent`, `curl` and `date` replaced
on PATH by stubs. The docker stub records every invocation, so the tests read
the build and push commands the pipeline would have run.
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
    ".buildkite/release/build_images.sh",
    ".buildkite/release/lib.sh",
    ".buildkite/release/preflight.sh",
    "docker/build_image.sh",
    "pyproject.toml",
)
_NIGHTLY_SCRIPT = ".buildkite/scripts/publish_nightly_images.sh"
_RELEASE_SCRIPT = ".buildkite/release/build_images.sh"
_PREFLIGHT_SCRIPT = ".buildkite/release/preflight.sh"
_CI_GATE = ".buildkite/release/ci_gate.py"
_BUILD_IMAGE_SCRIPT = "docker/build_image.sh"
_REGISTRY = "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu"
_REVISION_LABEL = "org.opencontainers.image.revision"
_OTHER_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# `docker inspect` answers with a repo digest, which the release script reads
# back after its push.
_DOCKER_STUB = """#!/bin/bash
printf '%q ' "$@" >> "$DOCKER_LOG"
echo >> "$DOCKER_LOG"
if [ "$1" = "inspect" ]; then
    image="${@: -1}"
    echo "${image%:*}@sha256:$(printf '0%.0s' {1..64})"
fi
"""
_DATE_STUB = """#!/bin/bash
case "$*" in
    "+%Y%m%d"|"-u +%Y%m%d") echo 20261118 ;;
    *) exec /bin/date "$@" ;;
esac
"""
# Build meta-data is one file per key under $META_DIR, empty unless a test or
# a script under test sets a key. For a missing key `meta-data exists` says
# "not found": 100 for the release-* keys of the release library, 1 for the
# approval lookup of the nightly script. Every other call succeeds.
_BUILDKITE_AGENT_STUB = """#!/bin/bash
if [ "$1" = "meta-data" ]; then
    case "$2" in
        set) printf '%s' "$4" > "$META_DIR/$3" ;;
        get) cat "$META_DIR/$3" ;;
        exists)
            [ "${META_EXISTS_FAIL_KEY:-}" = "$3" ] && exit "${META_EXISTS_FAIL_CODE:-2}"
            [ -e "$META_DIR/$3" ] && exit 0
            case "$3" in release-*) exit 100 ;; esac
            exit 1
            ;;
    esac
fi
exit 0
"""
# PyPI answers "no such version" to the preflight's check.
_CURL_STUB = """#!/bin/bash
printf 404
"""
# Stands in for ci_gate.py, which asks GitHub: records the commit it is asked
# about and vouches for it.
_CI_GATE_STUB = """import os, sys
with open(os.environ["CI_GATE_LOG"], "a") as log:
    log.write(sys.argv[1] + "\\n")
print("https://buildkite.example/builds/1")
"""


def _without_git_variables(env: dict[str, str]) -> dict[str, str]:
    """`env` without GIT_* variables, which would point git at another
    repository (GIT_DIR under a hook) or at the caller's configuration."""
    return {key: value for key, value in env.items() if not key.startswith("GIT_")}


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            **_without_git_variables(dict(os.environ)),
            # The caller's configuration can sign commits or rewrite paths.
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    ).stdout.strip()


def _commit_everything(repo: pathlib.Path, branch: str) -> str:
    """Makes `repo` a one-commit repository on `branch`; returns its HEAD."""
    _git(repo, "init", "-q", "-b", branch)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "source")
    return _git(repo, "rev-parse", "HEAD")


class _Checkout:
    """A throwaway checkout of the image scripts, with stubbed tools."""

    def __init__(self, tmp_path: pathlib.Path, branch: str):
        self.branch = branch
        # One level down, so a test can make the parent a repository too.
        self.repo = tmp_path / "outer" / "repo"
        for relative in _COPIED:
            destination = self.repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(_REPO / relative, destination)
        self.head = _commit_everything(self.repo, branch)

        self._bin_dir = tmp_path / "bin"
        self._bin_dir.mkdir()
        for name, body in (
            ("docker", _DOCKER_STUB),
            ("date", _DATE_STUB),
            ("buildkite-agent", _BUILDKITE_AGENT_STUB),
            ("curl", _CURL_STUB),
        ):
            stub = self._bin_dir / name
            stub.write_text(body)
            stub.chmod(0o755)
        self._docker_log = tmp_path / "docker.log"
        self._ci_gate_log = tmp_path / "ci_gate.log"
        self._meta_dir = tmp_path / "meta"
        self._meta_dir.mkdir()

    def set_meta(self, key: str, value: str) -> None:
        """Stores build meta-data, as an earlier step of the build would."""
        (self._meta_dir / key).write_text(value)

    def meta(self) -> dict[str, str]:
        """The build meta-data stored so far."""
        return {path.name: path.read_text() for path in self._meta_dir.iterdir()}

    def commits_asked_of_the_ci_gate(self) -> list[str]:
        return self._ci_gate_log.read_text().splitlines()

    def commit_again(self) -> str:
        """Moves the checkout to a new commit; returns it."""
        (self.repo / "later.txt").write_text("later\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "later")
        return _git(self.repo, "rev-parse", "HEAD")

    def drop_git_metadata_inside_another_repository(self) -> str:
        """Removes the checkout's .git and makes its parent directory a
        repository of its own; returns that repository's HEAD, which is not
        the commit the images are built from."""
        shutil.rmtree(self.repo / ".git")
        (self.repo.parent / "unrelated.txt").write_text("unrelated\n")
        return _commit_everything(self.repo.parent, "main")

    def run(self, script: str, *args: str, **env: str):
        """Runs `script` from the checkout root; returns the completed
        process and the docker invocations as argv lists."""
        self._docker_log.write_text("")
        self._ci_gate_log.write_text("")
        process_env = {
            key: value
            for key, value in _without_git_variables(dict(os.environ)).items()
            if not key.startswith(("BUILDKITE", "DOCKERHUB", "NIGHTLY_", "RELEASE_"))
            and key != "PUBLISH_DOCKERHUB"
        }
        process_env.update(
            PATH=f"{self._bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            DOCKER_LOG=str(self._docker_log),
            CI_GATE_LOG=str(self._ci_gate_log),
            META_DIR=str(self._meta_dir),
            BUILDKITE_BRANCH=self.branch,
            **env,
        )
        completed = subprocess.run(
            ["bash", str(self.repo / script), *args],
            cwd=self.repo,
            capture_output=True,
            text=True,
            env=process_env,
        )
        calls = [
            shlex.split(line) for line in self._docker_log.read_text().splitlines()
        ]
        return completed, calls


@pytest.fixture
def nightly(tmp_path):
    """A checkout of `main`, where the nightly images are published from."""
    return _Checkout(tmp_path, "main")


@pytest.fixture
def release(tmp_path):
    """A checkout of a release branch, where release images are built from."""
    return _Checkout(tmp_path, "releases/v0.29.0")


def _builds(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call[0] == "build"]


def _labels(build: list[str]) -> list[str]:
    return [build[i + 1] for i, arg in enumerate(build) if arg == "--label"]


def _option(build: list[str], name: str) -> str:
    return build[build.index(name) + 1]


def test_each_nightly_image_is_labeled_with_the_commit_it_was_built_from(nightly):
    completed, calls = nightly.run(_NIGHTLY_SCRIPT, BUILDKITE_COMMIT=nightly.head)

    assert completed.returncode == 0, completed.stderr
    assert [
        (_option(build, "--target"), _option(build, "-t"), _labels(build))
        for build in _builds(calls)
    ] == [
        (
            target,
            f"{_REGISTRY}/vllm-torchtpu-{target}:nightly-20261118",
            [f"{_REVISION_LABEL}={nightly.head}"],
        )
        for target in ("ci", "dev", "prod")
    ]


def test_nightly_tags_are_unchanged(nightly):
    completed, calls = nightly.run(_NIGHTLY_SCRIPT, BUILDKITE_COMMIT=nightly.head)

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
    completed, calls = nightly.run(_NIGHTLY_SCRIPT, BUILDKITE_COMMIT=buildkite_commit)

    assert completed.returncode == 0, completed.stderr
    assert {tuple(_labels(build)) for build in _builds(calls)} == {
        (f"{_REVISION_LABEL}={nightly.head}",)
    }


def test_label_falls_back_to_buildkite_commit_without_git_metadata(nightly):
    # The enclosing repository's HEAD must not be taken for the checkout's.
    enclosing_head = nightly.drop_git_metadata_inside_another_repository()

    completed, calls = nightly.run(_NIGHTLY_SCRIPT, BUILDKITE_COMMIT=_OTHER_COMMIT)

    assert completed.returncode == 0, completed.stderr
    assert enclosing_head != _OTHER_COMMIT
    assert {tuple(_labels(build)) for build in _builds(calls)} == {
        (f"{_REVISION_LABEL}={_OTHER_COMMIT}",)
    }


def test_nothing_is_built_when_the_commit_is_unknown(nightly):
    nightly.drop_git_metadata_inside_another_repository()

    completed, calls = nightly.run(_NIGHTLY_SCRIPT, BUILDKITE_COMMIT="HEAD")

    assert completed.returncode == 1
    assert "cannot tell which commit the images are built from" in completed.stderr
    assert calls == []


def test_dockerhub_nightly_tag_names_the_checked_out_commit(nightly):
    # bootstrap.py imports sibling CI modules; only its vllm-pin output matters.
    (nightly.repo / ".buildkite/scripts/bootstrap.py").write_text(
        f'print("{_OTHER_COMMIT}")\n'
    )

    completed, calls = nightly.run(
        _NIGHTLY_SCRIPT,
        BUILDKITE_COMMIT="HEAD",
        PUBLISH_DOCKERHUB="1",
        DOCKERHUB_REPO="vllm/vllm-torchtpu",
        DOCKERHUB_USERNAME="u",
        DOCKERHUB_TOKEN_VLLM_TORCHTPU="t",
    )

    assert completed.returncode == 0, completed.stderr
    pushes = [call[1] for call in calls if call[0] == "push"]
    assert (
        f"vllm/vllm-torchtpu:nightly-20261118-{nightly.head[:7]}-0123456" in pushes
    ), pushes


def test_each_release_image_is_labeled_with_the_commit_it_was_built_from(release):
    completed, calls = release.run(_RELEASE_SCRIPT, BUILDKITE_COMMIT="HEAD")

    assert completed.returncode == 0, completed.stderr
    assert [
        (_option(build, "--target"), _option(build, "-t"), _labels(build))
        for build in _builds(calls)
    ] == [
        (
            target,
            f"{_REGISTRY}/vllm-torchtpu-{target}:0.29.0",
            [f"{_REVISION_LABEL}={release.head}"],
        )
        for target in ("ci", "dev", "prod")
    ]
    assert [call[1] for call in calls if call[0] == "push"] == [
        f"{_REGISTRY}/vllm-torchtpu-{target}:{tag}"
        for target in ("ci", "dev", "prod")
        for tag in ("0.29.0", "20261118-0.29.0")
    ]


def test_no_release_image_is_built_when_the_commit_is_unknown(release):
    release.drop_git_metadata_inside_another_repository()

    completed, calls = release.run(_RELEASE_SCRIPT, BUILDKITE_COMMIT="HEAD")

    assert completed.returncode == 1
    assert "cannot tell which commit the images are built from" in completed.stderr
    assert _builds(calls) == []


def test_preflight_gates_and_stores_the_checked_out_commit(release):
    (release.repo / _CI_GATE).write_text(_CI_GATE_STUB)

    # "HEAD" is what Buildkite reports for a build created without a commit;
    # GitHub would read it as the default branch.
    completed, _ = release.run(_PREFLIGHT_SCRIPT, BUILDKITE_COMMIT="HEAD")

    assert completed.returncode == 0, completed.stderr
    assert release.commits_asked_of_the_ci_gate() == [release.head]
    assert release.meta() == {
        "release-version": "0.29.0",
        "release-revision": release.head,
        "release-prerelease": "no",
    }


def test_release_images_are_built_from_the_commit_preflight_gated(release):
    release.set_meta("release-revision", release.head)

    completed, calls = release.run(_RELEASE_SCRIPT, BUILDKITE_COMMIT="HEAD")

    assert completed.returncode == 0, completed.stderr
    assert {tuple(_labels(build)) for build in _builds(calls)} == {
        (f"{_REVISION_LABEL}={release.head}",)
    }


def test_no_release_image_is_built_from_a_commit_preflight_did_not_gate(release):
    release.set_meta("release-revision", release.head)
    later = release.commit_again()

    completed, calls = release.run(_RELEASE_SCRIPT, BUILDKITE_COMMIT=release.head)

    assert completed.returncode == 1
    assert (
        f"this step's checkout is at {later}, but preflight gated the release on"
        f" {release.head}." in completed.stderr
    )
    assert calls == []


@pytest.mark.parametrize("key", ["release-version", "release-revision"])
def test_release_build_stops_when_metadata_lookup_fails(release, key):
    completed, calls = release.run(
        _RELEASE_SCRIPT,
        META_EXISTS_FAIL_KEY=key,
        META_EXISTS_FAIL_CODE="2",
    )

    assert completed.returncode == 1
    assert f"could not read meta-data {key} (exit 2)." in completed.stderr
    assert calls == []


def test_build_image_passes_every_label_to_docker_build(nightly):
    completed, calls = nightly.run(
        _BUILD_IMAGE_SCRIPT,
        "--label",
        "first=1",
        "-t",
        "img:tag",
        "--label",
        "second=a=b",
    )

    assert completed.returncode == 0, completed.stderr
    (build,) = _builds(calls)
    assert _labels(build) == ["first=1", "second=a=b"]
    assert _option(build, "-t") == "img:tag"


def test_build_image_without_a_label_passes_none(nightly):
    completed, calls = nightly.run(_BUILD_IMAGE_SCRIPT, "-t", "img:tag")

    assert completed.returncode == 0, completed.stderr
    (build,) = _builds(calls)
    assert _labels(build) == []


@pytest.mark.parametrize(
    ("args", "shown"),
    [
        (["--label", "novalue"], "novalue"),
        (["--label", "=value"], "=value"),
        (["--label"], ""),
    ],
)
def test_build_image_rejects_a_label_that_is_not_key_value(nightly, args, shown):
    completed, calls = nightly.run(_BUILD_IMAGE_SCRIPT, "-t", "img:tag", *args)

    assert completed.returncode == 1
    assert f"--label expects KEY=VALUE, got: {shown}" in completed.stderr
    assert calls == []
