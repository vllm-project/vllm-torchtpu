# SPDX-License-Identifier: Apache-2.0

import pytest

# Kernel-path tests select the path under test explicitly. Every test starts
# from the plain EP8 configuration so a path is only active when the test asks
# for it.
_PLAIN_K3_ENV = {
    "K3_DECODE_EXPERT_BANKS": "1",
    "K3_MAX_PREFILL_STEP_TOKENS": "8192",
    "K3_PREFILL_A2A_CHIP_SHARD": "0",
    "K3_PREFILL_EXPERT_PLAN": "",
    "K3_PREFILL_KDA_A2A": "0",
    "K3_PREFILL_KDA_RING_SET": "0",
    "K3_PREFILL_MLA_A2A": "0",
    "K3_PREFILL_MOE_16K_MONOLITHIC": "0",
    "K3_PREFILL_MOE_W4A8": "0",
    "K3_PREFILL_PREFIX_FP8": "0",
    "K3_PREFILL_REPLICA_PLAN": "",
    "K3_PREFILL_SJF": "0",
    "K3_REPLICA_PLAN": "",
    "K3_SERIAL_PREFILL": "0",
    "K3_UNIFIED_EP16": "0",
}


@pytest.fixture(autouse=True)
def _plain_k3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _PLAIN_K3_ENV.items():
        monkeypatch.setenv(name, value)
