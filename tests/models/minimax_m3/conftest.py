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
"""Runs the MXFP8 matmul on CPU through its reference: the kernel needs a TPU."""

import pytest

import vllm_torchtpu.models.minimax_m3.quant as quant_mod
from tests.models.minimax_m3.mxfp8 import mxfp8_matmul_ref


@pytest.fixture(autouse=True)
def _mxfp8_matmul_on_cpu(monkeypatch):
    monkeypatch.setattr(quant_mod, "mxfp8_matmul", mxfp8_matmul_ref)
