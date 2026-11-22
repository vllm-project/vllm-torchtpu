# SPDX-License-Identifier: Apache-2.0
"""TPU_ASYNC_ALL_TO_ALL adds the libtpu flag before libtpu loads."""

import os
import subprocess
import sys

import pytest

_FLAG = "--xla_tpu_enable_async_all_to_all"
_PRINT = (
    "import vllm_torchtpu.env_override, os; "
    "print(os.environ.get('LIBTPU_INIT_ARGS', ''))"
)


def _libtpu_args(**env):
    child = {k: v for k, v in os.environ.items() if k != "TPU_ASYNC_ALL_TO_ALL"}
    child.update(env, JAX_PLATFORMS="cpu", TPU_ACCELERATOR_TYPE="none")
    out = subprocess.run(
        [sys.executable, "-c", _PRINT],
        env=child,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip().splitlines()[-1].split()


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_switch_controls_flag(value, expected):
    args = _libtpu_args(TPU_ASYNC_ALL_TO_ALL=value, LIBTPU_INIT_ARGS="")
    assert (f"{_FLAG}=true" in args) is expected


def test_explicit_flag_wins():
    args = _libtpu_args(TPU_ASYNC_ALL_TO_ALL="1", LIBTPU_INIT_ARGS=f"{_FLAG}=false")
    assert [a for a in args if a.startswith(_FLAG)] == [f"{_FLAG}=false"]
