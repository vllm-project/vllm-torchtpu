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
"""Runs test_dcp2_tpu.py on the TPU lane with the DCP-2 harness in TPU mode
(real Mosaic batched RPA with return_lse and the paired-page DCP prefill).

The harness picks its mode once, at import, so the module runs in a fresh
pytest process; this process must not open the TPU before it does.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dcp2_tpu.py")


def test_dcp2_real_kernels_on_tpu():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", MODULE],
        env={**os.environ, "DCP2_HARNESS_DEVICE": "tpu"},
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
