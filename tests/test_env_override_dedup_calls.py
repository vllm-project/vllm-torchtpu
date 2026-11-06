# SPDX-License-Identifier: Apache-2.0
"""TPU_INLINE_DEDUPLICATED_CALLS adds the libtpu flag before libtpu loads."""

import os
import subprocess
import sys

import pytest

_FLAG = "--xla_tpu_enable_deduplicated_calls"
_PRINT = (
    "import vllm_torchtpu.env_override, os; "
    "print(os.environ.get('LIBTPU_INIT_ARGS', ''))"
)


def _libtpu_args(**env):
    child = {
        k: v for k, v in os.environ.items() if k != "TPU_INLINE_DEDUPLICATED_CALLS"
    }
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
    args = _libtpu_args(TPU_INLINE_DEDUPLICATED_CALLS=value, LIBTPU_INIT_ARGS="")
    assert (f"{_FLAG}=false" in args) is expected


def test_explicit_flag_wins():
    args = _libtpu_args(
        TPU_INLINE_DEDUPLICATED_CALLS="1", LIBTPU_INIT_ARGS=f"{_FLAG}=true"
    )
    assert [a for a in args if a.startswith(_FLAG)] == [f"{_FLAG}=true"]
