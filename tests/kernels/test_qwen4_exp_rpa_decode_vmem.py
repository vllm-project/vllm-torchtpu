# SPDX-License-Identifier: Apache-2.0
"""TPU_RPA_DECODE_BATCHED sizing of QSA's batched RPA decode calls."""

from types import SimpleNamespace

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa


@pytest.fixture
def tpu_info(monkeypatch):
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: SimpleNamespace(vmem_capacity_bytes=64 << 20)
    )


def test_off_keeps_the_default(monkeypatch, tpu_info):
    monkeypatch.delenv("TPU_RPA_DECODE_BATCHED", raising=False)
    assert qsa_rpa.decode_vmem_limit_bytes(8) is None


@pytest.mark.parametrize(
    "rows, expected", [(1, 32 << 20), (64, 32 << 20), (65, None), (128, None)]
)
def test_decode_programs_take_half_the_vmem(monkeypatch, tpu_info, rows, expected):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", "1")
    assert qsa_rpa.decode_vmem_limit_bytes(rows) == expected


@pytest.mark.parametrize("flag, limit", [("0", 64 << 20), ("1", 32 << 20)])
def test_pooled_decode_rpa_lowers_with_the_decode_vmem_limit(monkeypatch, flag, limit):
    """The real pooled step's RPA calls carry the halved scoped VMEM limit."""
    import re

    import jax

    from test_qwen4_exp_pool_inplace import _lower_pooled_step

    monkeypatch.setenv("TPU_RPA_DECODE_BATCHED", flag)
    jax.clear_caches()
    text = _lower_pooled_step(8).as_text()
    jax.clear_caches()
    rpa_lines = [line for line in text.splitlines() if "RPAd-" in line]
    assert rpa_lines
    for line in rpa_lines:
        assert re.findall(r"vmem_limit_bytes\\22:(\d+)", line) == [str(limit)]
