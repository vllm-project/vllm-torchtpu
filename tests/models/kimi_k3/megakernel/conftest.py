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

import pytest

from vllm_torchtpu.core import sjf_queue
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config

# Kernel-path tests select the path under test explicitly. Every test starts
# from the plain EP8 configuration so a path is only active when the test asks
# for it.
_PLAIN_K3_DEFAULTS = {
    "DEFAULT_DECODE_EXPERT_BANKS": True,
    "DEFAULT_MAX_PREFILL_STEP_TOKENS": 8192,
    "DEFAULT_PREFILL_A2A_CHIP_SHARD": False,
    "DEFAULT_PREFILL_EXPERT_PLAN": "",
    "DEFAULT_PREFILL_KDA_A2A": False,
    "DEFAULT_PREFILL_KDA_RING_SET": False,
    "DEFAULT_PREFILL_MLA_A2A": False,
    "DEFAULT_PREFILL_MOE_16K_MONOLITHIC": False,
    "DEFAULT_PREFILL_MOE_W4A8": False,
    "DEFAULT_PREFILL_PREFIX_FP8": False,
    "DEFAULT_PREFILL_REPLICA_PLAN": "",
    "DEFAULT_REPLICA_PLAN": "",
    "DEFAULT_SERIAL_PREFILL": False,
    "DEFAULT_UNIFIED_EP16": False,
}


@pytest.fixture(autouse=True)
def _plain_k3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _PLAIN_K3_DEFAULTS.items():
        monkeypatch.setattr(k3_config, name, value)
    monkeypatch.setattr(sjf_queue, "DEFAULT_ENABLED", False)
