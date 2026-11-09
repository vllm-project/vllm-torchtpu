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
"""libtpu defaults env_override adds to LIBTPU_INIT_ARGS at import."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
_FLAG = "--xla_tpu_enable_sparse_core_hierarchical_all_reduce"


_PRINT_ARGS = (
    "import os, vllm_torchtpu.env_override; print(os.environ['LIBTPU_INIT_ARGS'])"
)


def _libtpu_args(accelerator: str, user_args: str) -> list[str]:
    """LIBTPU_INIT_ARGS after a fresh interpreter imports env_override."""
    path = os.pathsep.join(filter(None, [str(_SRC), os.environ.get("PYTHONPATH")]))
    env = dict(
        os.environ,
        TPU_ACCELERATOR_TYPE=accelerator,
        LIBTPU_INIT_ARGS=user_args,
        PYTHONPATH=path,
    )
    out = subprocess.run(
        [sys.executable, "-c", _PRINT_ARGS],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1].split()


def test_tpu7x_gets_the_hierarchical_sparse_core_all_reduce():
    assert f"{_FLAG}=true" in _libtpu_args("tpu7x-8", "")


@pytest.mark.parametrize("user", [f"{_FLAG}=false", _FLAG])
def test_an_explicit_setting_wins(user):
    args = _libtpu_args("tpu7x-8", user)
    assert [a for a in args if a.startswith(_FLAG)] == [user]


def test_other_chips_are_left_alone():
    assert not any(a.startswith(_FLAG) for a in _libtpu_args("v6e-8", ""))
