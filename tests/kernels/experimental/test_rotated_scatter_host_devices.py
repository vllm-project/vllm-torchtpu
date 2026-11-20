# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The rotated reduce-scatter tests of the prefill token split on 16 host
devices.

They need the 16 cores of a 2x2x2 slice, more than the multichip lane's host
has, so they skip there. Every pallas_call in them runs interpreted, so this
test runs them in child pytests on 16 virtual CPU devices on the CPU lane.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MOE = "tests/layers/adapter/test_moe_tp_decode.py"
SPLIT = "tests/kernels/experimental/test_prefill_token_split.py"
# Node ids per child, run side by side (each takes 4 to 6 minutes on 8 cores),
# and the cases each must pass.
GROUPS = (
    (
        (
            f"{MOE}::test_rotated_scatter_gives_each_core_the_rows_of_xla_scatter[2048]",
            f"{MOE}::test_a_wrong_peer_table_moves_the_wrong_rows",
        ),
        2,
    ),
    (
        (f"{MOE}::test_rotated_scatter_gives_each_core_the_rows_of_xla_scatter[4096]",),
        1,
    ),
    (
        (
            f"{SPLIT}::test_rotated_scatter_rows_gives_each_core_the_rows_of_scatter_rows",
        ),
        3,
    ),
)
TIMEOUT_SECONDS = 1500


def _start(ids):
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=16",
    )
    return subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + ["-m", "multichip", "--timeout", str(TIMEOUT_SECONDS), *ids],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


@pytest.mark.timeout(TIMEOUT_SECONDS + 60)
def test_rotated_scatter_cases_pass_on_16_host_devices():
    processes = [_start(ids) for ids, _ in GROUPS]
    try:
        outputs = [p.communicate(timeout=TIMEOUT_SECONDS)[0] for p in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    for (_, passed), process, output in zip(GROUPS, processes, outputs):
        summary = output.strip().splitlines()[-1] if output.strip() else ""
        assert process.returncode == 0, output[-5000:]
        assert f"{passed} passed" in summary and "skipped" not in summary, summary
