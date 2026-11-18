# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_HC_PREFILL_TILE unset keeps the za10q prefill HC routing.

The served config passes hc_prefill_full_tile=128 (with
VLLM_TPU_HC_PREFILL_WIDE_TILE=256). Before the hardware-aware policy, the full
mix and the combine ran on whole tile pairs only: rows >= 256 and rows % 256 == 0,
the 256 tile when rows % 512 == 0, else 128; the SP shard aligned to 256.
"""

import pytest
import torch

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
    _full_tile,
)

V7X = HcHardware("7x", 64 << 20, 128, 16, 8, 256)
ROWS = [16, 64, 96, 128, 256, 384, 512, 640, 768, 1024, 1152, 1536, 4096, 8192]


@pytest.fixture
def served(monkeypatch, request):
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    monkeypatch.delenv("VLLM_TPU_HC_PREFILL_TILE", raising=False)
    monkeypatch.delenv("TPU_QWEN4_HC_DECODE_FUSED", raising=False)
    monkeypatch.delenv("VLLM_TPU_HC_UP_PRETRANSPOSE", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_HC_PREFILL_WIDE_TILE", 256)
    monkeypatch.setattr(envs, "VLLM_TPU_HC_PREFILL_COMBINE_MIX", True)
    hardware = getattr(request, "param", V7X)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: hardware)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a, **k: object())
    monkeypatch.setattr(
        vllm_adapter, "build_combine_full_mix_op", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        vllm_adapter, "build_gated_residual_op", lambda *a, **k: object()
    )


def _served_module(**kwargs):
    with torch.device("meta"):
        return GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
            ),
            prefill_up_tile=128,
            prefill_norm_tile=128,
            prefill_full_tile=128,
            **kwargs,
        )


def _za10q_tile(rows):
    """hyperconnection.py at 3ba7778f0f: `_prefill_full` + `_prefill_tile`."""
    if rows < 256 or rows % 256:
        return None
    return next(t for t in (256, 128) if rows % (2 * t) == 0)


@pytest.mark.parametrize("served", [V7X, None], indirect=True)
def test_served_routing_matches_za10q(served):
    hc = _served_module()
    assert hc._legacy_pairs
    assert tuple(hc._prefill_full_ops) == (256, 128)
    assert tuple(hc._prefill_combine_full_ops) == (256, 128)
    assert hc._prefill_combine_op is not None
    assert hc._decode_mix_op is None
    assert hc.sp_tile_alignment == 256
    if hc._hardware is not None:
        # The za10q kernel limits: no static admission, 60 MiB VMEM.
        assert not hc._hardware.admit and hc._hardware.vmem_bytes == 60 << 20
    for rows in ROWS:
        assert _full_tile(hc, rows, hc._prefill_full_ops) == _za10q_tile(rows), rows


def test_served_defaults(served):
    from vllm_torchtpu import envs

    assert envs.VLLM_TPU_HC_PREFILL_TILE is None
    assert envs.TPU_QWEN4_HC_DECODE_FUSED is False
    assert envs.VLLM_TPU_HC_UP_PRETRANSPOSE is False
    hc = _served_module()
    hc.transpose_up_weight()
    assert not hc._up_transposed
    # The unconfigured default stays untiled (0), not "auto".
    with torch.device("meta"):
        bare = GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
            )
        )
    assert bare.prefill_full_tile == 0 and not bare._prefill_full_ops


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("0", 0), ("256", 256)])
def test_env_parses_auto_distinct_from_unset(value, expected, monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", value)
    assert envs.VLLM_TPU_HC_PREFILL_TILE == expected
    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", "")
    assert envs.VLLM_TPU_HC_PREFILL_TILE is None


def test_env_auto_overrides_backbone_config(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.model import _hc_prefill_mode_arg

    cfg = {"hc_prefill_full_tile": 128}
    monkeypatch.delenv("VLLM_TPU_HC_PREFILL_TILE", raising=False)
    assert _hc_prefill_mode_arg(cfg, "hc_prefill_full_tile", None) == 128
    monkeypatch.setenv("VLLM_TPU_HC_PREFILL_TILE", "auto")
    assert _hc_prefill_mode_arg(cfg, "hc_prefill_full_tile", None) is None
    # MTP layers (default 0) keep their config; the draft disables them anyway.
    assert _hc_prefill_mode_arg(cfg, "hc_prefill_full_tile", 0) == 128
