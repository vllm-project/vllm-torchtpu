# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_torchtpu.megakernel.kimi_k3.config import KernelConfig

pytestmark = pytest.mark.cpu_test


def test_should_keep_bf16_window_when_env_unset(monkeypatch):
    monkeypatch.delenv("K3_PREFILL_KDA_FP8_GATHER", raising=False)

    assert KernelConfig().prefill_kda_fp8_gather is False


def test_should_select_fp8_window_when_env_set(monkeypatch):
    monkeypatch.setenv("K3_PREFILL_KDA_FP8_GATHER", "1")

    assert KernelConfig().prefill_kda_fp8_gather is True
