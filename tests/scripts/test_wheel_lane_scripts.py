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
"""Unit tests for the scripts of the wheel lane (pipeline_build_wheels.yml).

The lane runs the release's wheel steps on every CI build. What tells it from a
release is tested here: the test version, the two script arguments, the shared
vllm-torchtpu step, and verify_built_wheel on a local version.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Mapping, Sequence

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BUILDKITE = _REPO_ROOT / ".buildkite"

_spec = importlib.util.spec_from_file_location(
    "wheel_utils", _BUILDKITE / "scripts" / "wheel_utils.py"
)
assert _spec is not None and _spec.loader is not None
wheel_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wheel_utils)

# A buildkite-agent that keeps meta-data as files under $STUB_DIR/meta and logs
# each call with the directory it was made from.
_AGENT = """#!/bin/bash
echo "${PWD}: $*" >> "${STUB_DIR}/agent.log"
case "$1 $2" in
    "meta-data exists") [[ -f "${STUB_DIR}/meta/$3" ]] || exit 100 ;;
    "meta-data get") cat "${STUB_DIR}/meta/$3" ;;
    "meta-data set") printf '%s' "$4" > "${STUB_DIR}/meta/$3" ;;
    "artifact download") mkdir -p dist && touch dist/a.whl dist/b.whl ;;
    "annotate "*) cat > /dev/null ;;
esac
"""
# A curl that logs its arguments and answers 404, as PyPI does for a version it
# does not have.
_CURL = """#!/bin/bash
echo "$*" >> "${STUB_DIR}/curl.log"
printf 404
"""
# lib.sh for check_wheels.sh, without the functions that reach the network.
_FAKE_LIB = """
die() { echo "ERROR: $*" >&2; exit 1; }
release_version() { echo 1.2.3; }
install_uv() { echo uv; }
resolve_vllm_torchtpu() {
    printf 'vllm-torchtpu==1.2.3\\nvllm-torchtpu-core==1.2.3\\n' > "$3"
}
"""
# build_vllm_torchtpu.sh that records the version and the vLLM it was given.
_FAKE_BUILD = """
echo "version=$1 vllm=${VLLM_REF_OVERRIDE:-pin}" >> "${STUB_DIR}/build.log"
"""


class TestWheelLaneScripts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.stub = self.tmp / "stub"
        (self.stub / "bin").mkdir(parents=True)
        (self.stub / "meta").mkdir()
        for name, body in (("buildkite-agent", _AGENT), ("curl", _CURL)):
            path = self.stub / "bin" / name
            path.write_text(body, encoding="utf-8")
            path.chmod(0o755)
        self.elsewhere = self.tmp / "elsewhere"
        self.elsewhere.mkdir()
        self.env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PUBLISH_PYPI", "VLLM_REF_OVERRIDE")
        }
        self.env["PATH"] = f"{self.stub / 'bin'}{os.pathsep}{self.env['PATH']}"
        self.env["STUB_DIR"] = str(self.stub)

    def _bash(
        self, *args: str, cwd: pathlib.Path | None = None, **env: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", *args],
            cwd=cwd or self.elsewhere,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    def _meta(self, key: str) -> str:
        return (self.stub / "meta" / key).read_text(encoding="utf-8")

    def _log(self, name: str) -> str:
        path = self.stub / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _tree(self, real: Sequence[str], fake: Mapping[str, str]) -> pathlib.Path:
        """A directory with copies of the `real` files of .buildkite, and the
        `fake` ones (path to content) in place of the scripts they stand for."""
        root = self.tmp / "checkout"
        for rel in real:
            dest = root / ".buildkite" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(_BUILDKITE / rel, dest)
        for rel, body in fake.items():
            dest = root / ".buildkite" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        return root

    def test_test_version_is_set_once_from_the_checkouts_commit(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("needs git")
        root = self._tree(["release/lib.sh", "release/set_test_wheel_version.sh"], {})
        git_env = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
                + list(args),
                env={**self.env, **git_env},
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "a commit")
        version = f"0.0.0+g{git('rev-parse', 'HEAD')[:12]}"
        script = str(root / ".buildkite/release/set_test_wheel_version.sh")

        # Run from another directory: the version is the script's checkout's.
        done = self._bash(script)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self._meta("release-version"), version)
        # Every later wheel step reads it through release_version, also when
        # lib.sh is sourced from another directory.
        lib = str(root / ".buildkite/release/lib.sh")
        read = self._bash("-c", 'source "$1" && release_version', "_", lib)
        self.assertEqual((read.returncode, read.stdout.strip()), (0, version))
        # A retried step finds its own version and passes.
        self.assertEqual(self._bash(script).returncode, 0)
        self.assertEqual(self._meta("release-version"), version)
        # A release's version (preflight.sh) is never replaced.
        (self.stub / "meta" / "release-version").write_text("1.2.3", encoding="utf-8")
        refused = self._bash(script)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("release-version is already 1.2.3", refused.stderr)
        self.assertEqual(self._meta("release-version"), "1.2.3")

    def test_build_core_wheel_checks_pypi_unless_told_not_to(self) -> None:
        version = "0.0.0+g0123456789ab"
        (self.stub / "meta" / "release-version").write_text(version, encoding="utf-8")
        script = str(_BUILDKITE / "scripts" / "build_core_wheel.sh")
        # PYTHON=false stops the script where it would make its build venv, so
        # nothing is built and nothing is fetched.
        venv = "could not create a venv"

        skipped = self._bash(script, "--skip-pypi-check", PYTHON="false")
        self.assertIn(venv, skipped.stderr)
        self.assertEqual(self._log("curl.log"), "")

        # Without the argument the check runs, as in a release.
        unset = self._bash(script, PYTHON="false")
        self.assertIn("PUBLISH_PYPI) is not set", unset.stderr)
        self.assertNotIn(venv, unset.stderr)
        checked = self._bash(script, PYTHON="false", PUBLISH_PYPI="vllm-torchtpu-core")
        self.assertIn(venv, checked.stderr)
        self.assertIn(f"/pypi/vllm-torchtpu-core/{version}/json", self._log("curl.log"))

        unknown = self._bash(script, "--nope", PYTHON="false")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("unknown argument '--nope'", unknown.stderr)

    def test_check_wheels_annotates_under_the_given_context(self) -> None:
        # What is under test is the argument and the annotation it names.
        root = self._tree(["release/check_wheels.sh"], {"release/lib.sh": _FAKE_LIB})
        script = str(root / ".buildkite/release/check_wheels.sh")

        self.assertEqual(self._bash(script).returncode, 0)
        self.assertIn(
            "annotate --style info --context release-wheels", self._log("agent.log")
        )
        named = self._bash(script, "--annotation-context", "test-wheels")
        self.assertEqual(named.returncode, 0, named.stderr)
        self.assertIn(
            "annotate --style info --context test-wheels", self._log("agent.log")
        )

        for args, error in (
            (("--annotation-context",), "--annotation-context needs a name"),
            (("--nope",), "unknown argument '--nope'"),
        ):
            with self.subTest(args=args):
                refused = self._bash(script, *args)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn(error, refused.stderr)

    def test_release_and_lane_build_vllm_torchtpu_with_one_script(self) -> None:
        # The shared step: the build's version, vLLM at VLLM_REF_OVERRIDE if
        # the caller set one, and the artifacts uploaded from the repo root.
        root = self._tree(
            ["release/lib.sh", "release/build_vllm_torchtpu_wheel.sh"],
            {"scripts/build_vllm_torchtpu.sh": _FAKE_BUILD},
        )
        (self.stub / "meta" / "release-version").write_text("1.2.3", encoding="utf-8")
        script = str(root / ".buildkite/release/build_vllm_torchtpu_wheel.sh")
        self.assertEqual(self._bash(script).returncode, 0)
        self.assertEqual(self._bash(script, VLLM_REF_OVERRIDE="abc123").returncode, 0)
        self.assertEqual(
            self._log("build.log").splitlines(),
            ["version=1.2.3 vllm=pin", "version=1.2.3 vllm=abc123"],
        )
        uploads = [
            line
            for line in self._log("agent.log").splitlines()
            if "artifact upload" in line
        ]
        self.assertEqual(
            uploads[:2],
            [
                f"{root}: artifact upload dist/vllm_torchtpu-*.whl",
                f"{root}: artifact upload dist/vllm_torchtpu-*.tar.gz",
            ],
        )

        # Both pipelines call it. The lane alone names a vLLM: the one the CI
        # image is built from. The release builds the pyproject.toml pin.
        def commands(path: str, key: str) -> str:
            doc = yaml.safe_load((_BUILDKITE / path).read_text(encoding="utf-8"))
            steps = [s for s in doc["steps"] if isinstance(s, dict)]
            steps += [g for s in steps for g in s.get("steps", [])]
            (step,) = [s for s in steps if s.get("key") == key]
            return "\n".join(step["commands"])

        call = "bash .buildkite/release/build_vllm_torchtpu_wheel.sh"
        release = commands("release/pipeline.yml", "build_vllm_torchtpu")
        lane = commands("pipeline_build_wheels.yml", "build_vllm_torchtpu_wheel")
        self.assertIn(call, release)
        self.assertNotIn("VLLM_REF_OVERRIDE", release)
        self.assertIn(call, lane)
        self.assertIn(
            'VLLM_REF_OVERRIDE="$$(buildkite-agent meta-data get VLLM_COMMIT_HASH)"',
            lane,
        )

    def test_verify_built_wheel_takes_a_local_version(self) -> None:
        def verify(file_version: str, expected: str) -> None:
            name = f"dist/vllm_torchtpu_core-{file_version}-py3-none-any.whl"
            with contextlib.redirect_stdout(io.StringIO()):
                wheel_utils.verify_built_wheel(name, expected)

        verify("0.0.0+g0123456789ab", "0.0.0+g0123456789ab")
        with self.assertRaisesRegex(SystemExit, "!= expected version"):
            verify("0.0.0+g0123456789ab", "0.0.0+gba9876543210")
        # Why the test version has a "g": an all-digit local part loses its
        # leading zeros, so the wheel's version is no longer the one asked for.
        with self.assertRaisesRegex(SystemExit, r"0\.0\.0\+123456789 != "):
            verify("0.0.0+000123456789", "0.0.0+000123456789")


if __name__ == "__main__":
    unittest.main()
