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
"""Shared fixtures of the Qwen3.5 model tests."""

import pytest

from vllm_torchtpu.models.qwen3_5 import hooks


@pytest.fixture(autouse=True)
def _measured_moe_kernel(monkeypatch):
    """The calibrated KV budget applies only with the MoE kernel the decode
    legs ran (fused EP MoE v3); tests of another kernel override this."""
    monkeypatch.setattr(hooks, "_moe_kernel", lambda: hooks._CALIBRATED_MOE_KERNEL)
