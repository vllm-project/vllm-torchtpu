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
