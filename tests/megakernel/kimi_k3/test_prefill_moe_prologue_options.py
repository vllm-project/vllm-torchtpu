# SPDX-License-Identifier: Apache-2.0
"""Config checks for the fused prefill MoE prologue banks and one-shot reduce."""

import pytest

from vllm_torchtpu.megakernel.kimi_k3 import config

pytestmark = pytest.mark.cpu_test


def test_defaults_keep_ring_and_two_banks(monkeypatch):
    monkeypatch.delenv("K3_PREFILL_MOE_DMA_BANKS", raising=False)
    monkeypatch.delenv("K3_PREFILL_MOE_ONESHOT_MAX", raising=False)
    cfg = config.KernelConfig()
    assert cfg.prefill_moe_dma_banks == 2
    assert cfg.prefill_moe_oneshot_max == 0


def test_env_and_set_overrides(monkeypatch):
    monkeypatch.setenv("K3_PREFILL_MOE_DMA_BANKS", "3")
    monkeypatch.setenv("K3_PREFILL_MOE_ONESHOT_MAX", "1024")
    cfg = config.KernelConfig()
    assert (cfg.prefill_moe_dma_banks, cfg.prefill_moe_oneshot_max) == (3, 1024)
    assert cfg.replace(prefill_moe_dma_banks="4").prefill_moe_dma_banks == 4
    assert cfg.key() != cfg.replace(prefill_moe_oneshot_max="0").key()


@pytest.mark.parametrize("banks", [1, 5])
def test_bank_count_validated(banks):
    with pytest.raises(ValueError, match="prefill_moe_dma_banks"):
        config.KernelConfig(prefill_moe_dma_banks=banks)
