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
"""run_eval_flow.sh gates a run on the perf baselines of the chip it is on."""

import os
import pathlib
import subprocess
import sys

from jax.experimental.pallas import tpu as pltpu

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCHMARKING = pathlib.Path("scripts/vllm/benchmarking")
_PERF_BASELINES = _REPO_ROOT / _BENCHMARKING / "baselines" / "perf"


def _run_as_far_as_the_benchmark(tmp_path, accelerator_type, config):
    """Runs run_eval_flow.sh against a server that is already up (--host).

    The script reaches python3 by name, and the baselines and the benchmark by
    paths relative to where it runs. So it runs in a directory that has the
    repo's baselines and a stand-in benchmark, which ends the run with 3.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python3 = bin_dir / "python3"
    python3.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    python3.chmod(0o755)
    benchmarking = tmp_path / "work" / _BENCHMARKING
    benchmarking.mkdir(parents=True)
    (benchmarking / "baselines").symlink_to(_REPO_ROOT / _BENCHMARKING / "baselines")
    benchmark = benchmarking / "run_benchmarks.sh"
    benchmark.write_text("#!/usr/bin/env bash\nexit 3\n")
    benchmark.chmod(0o755)
    return subprocess.run(
        [
            "bash",
            str(_REPO_ROOT / _BENCHMARKING / "run_eval_flow.sh"),
            "--config",
            config,
            "--host",
            "127.0.0.1",
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=tmp_path / "work",
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TPU_ACCELERATOR_TYPE": accelerator_type,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_perf_baselines_are_filed_under_a_chip():
    # run_eval_flow.sh reads baselines/perf/<chip>/ alone, so a baseline filed
    # anywhere else gates nothing.
    assert not list(_PERF_BASELINES.glob("*.baseline.json"))
    chips = {"v" + chip.value.removeprefix("v") for chip in pltpu.ChipVersion}
    assert {path.name for path in _PERF_BASELINES.iterdir() if path.is_dir()} <= chips


def test_a_tpu7x_run_gates_on_a_v7x_baseline_that_exists(tmp_path):
    # Any config with a v7x baseline will do: a lookup that lands on another
    # directory finds no baseline, and the run reports without the perf gate.
    baseline = min(_PERF_BASELINES.glob("v7x/*.baseline.json"))
    config = baseline.name.removesuffix(".baseline.json")
    went_on = _run_as_far_as_the_benchmark(tmp_path, "tpu7x-8", config)
    assert went_on.returncode == 3, went_on.stderr
    picked = _BENCHMARKING / "baselines" / "perf" / "v7x" / baseline.name
    assert f"Perf baseline for v7x: {picked}\n" in went_on.stdout


def test_a_type_that_names_no_chip_stops_before_the_benchmark(tmp_path):
    stopped = _run_as_far_as_the_benchmark(tmp_path, "notachip-8", "qwen3-0.6b-tp1")
    assert stopped.returncode == 1, stopped.stderr
    assert "Unrecognized TPU accelerator type 'notachip-8'" in stopped.stderr
    assert "ERROR: could not read the TPU chip" in stopped.stderr
