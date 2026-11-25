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
"""Argument checks of .buildkite/kubernetes/disagg_benchmark.sh.

Every run passes --check-args, which makes the script exit once its arguments
are checked, or an argument the script rejects before that point. PATH is an
empty directory, so the script can start nothing but shell builtins.
"""

import pathlib
import shutil
import subprocess

import pytest
import yaml

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".buildkite"
    / "kubernetes"
    / "disagg_benchmark.sh"
)
_MANIFEST = _SCRIPT.parent / "manifests" / "workloads" / "qwen3.5-397b-1p1d.yaml"
_BASH = shutil.which("bash")


def _check_args(tmp_path, *args) -> tuple[int, str]:
    proc = subprocess.run(
        [_BASH, str(_SCRIPT), *args, "--check-args"],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    ("args", "forwarded"),
    [
        ([], ""),
        (
            ["--prefill-chips=4", "--decode-chips=4"],
            "--prefill-chips=4 --decode-chips=4",
        ),
        (
            ["--decode-chips=2", "--prefill-chips=16"],
            "--prefill-chips=16 --decode-chips=2",
        ),
    ],
)
def test_chip_flags_are_both_absent_or_both_positive_integers(
    tmp_path, args, forwarded
):
    """Checks --check-args flag validation and formatted chip_args output."""
    assert _check_args(tmp_path, *args) == (
        0,
        f"Arguments are valid; parse_gke_results.py gets: {forwarded}\n",
    )


def test_the_manifest_runs_the_script_with_arguments_it_accepts(tmp_path):
    """The manifest starts the script only once both engines are Ready, so a
    flag the script rejects has to fail here to fail before the engines start."""
    manifest = yaml.safe_load(_MANIFEST.read_text())
    jobs = {job["name"]: job for job in manifest["spec"]["replicatedJobs"]}
    pod_spec = jobs["benchmark"]["template"]["spec"]["template"]["spec"]
    command = pod_spec["containers"][0]["command"]

    returncode, output = _check_args(tmp_path, *command[2:])

    assert returncode == 0, output


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (
            ["--prefill-chips=4"],
            "ERROR: --prefill-chips and --decode-chips go together: "
            "both engines serve the run.",
        ),
        (
            ["--decode-chips=4"],
            "ERROR: --prefill-chips and --decode-chips go together: "
            "both engines serve the run.",
        ),
        (
            ["--prefill-chips=0", "--decode-chips=4"],
            "ERROR: --prefill-chips must be a positive integer, got '0'",
        ),
        (
            ["--prefill-chips=4", "--decode-chips=-4"],
            "ERROR: --decode-chips must be a positive integer, got '-4'",
        ),
        (
            ["--prefill-chips=four", "--decode-chips=4"],
            "ERROR: --prefill-chips must be a positive integer, got 'four'",
        ),
        (
            ["--prefill-chips=4", "--decode-chips="],
            "ERROR: --decode-chips must be a positive integer, got ''",
        ),
        (["--chips=8"], "ERROR: unknown argument: --chips=8"),
    ],
)
def test_a_chip_count_the_parser_would_reject_fails_before_startup(
    tmp_path, args, error
):
    assert _check_args(tmp_path, *args) == (1, f"{error}\n")
