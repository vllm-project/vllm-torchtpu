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
"""VLLM_TPU_HC_UP_PRETRANSPOSE vs the megakernel's HC weight view.

`GatedResidual.transpose_up_weight` replaces `input_mix_weight_up.weight` [W, LR]
by its transpose [LR, W]. At the served geometry that is [320, 10240], the shape of
`input_mix_weight_down.weight`: a packer that picked the layout by shape (or
swapped the two) would run silently on wrong weights. Every megakernel program
(segment, whole step, MTP tail) reads the HC mix weights only through
`qwen38_weights._hc_mix_operands`, so its packed operands being bitwise equal with
the flag on and off makes the kernel's output bitwise equal too.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

H, NHC, LR = 256, 4, 32


@pytest.fixture
def cpu_vllm_config(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    for module in ("parameter", "layers.linear"):
        for name, value in (
            ("get_tensor_model_parallel_rank", 0),
            ("get_tensor_model_parallel_world_size", 1),
        ):
            monkeypatch.setattr(
                f"vllm.model_executor.{module}.{name}",
                lambda value=value: value,
                raising=False,
            )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _module(use_combine: bool, seed: int):
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual
    from vllm_torchtpu.models.vllm.qwen4_exp.model import hc_config

    text = q4exp_text_config(
        hidden_size=H, hc_count=NHC, hc_lowrank=LR, num_hidden_layers=2
    )
    with torch.device("cpu"):
        hc = GatedResidual(hc_config(text), use_combine=use_combine)
    gen = torch.Generator().manual_seed(seed)
    for p in hc.parameters():
        p.data = (torch.randn(p.shape, generator=gen) * 0.1).to(torch.bfloat16)
    return hc


def _pack(hc):
    from vllm_torchtpu.megakernel_runtime.qwen38_weights import _hc_mix_operands

    return _hc_mix_operands(hc, NS(NHC=NHC, H=H, LOWRANK=LR))


@pytest.mark.usefixtures("cpu_vllm_config")
@pytest.mark.parametrize("use_combine", [True, False])
def test_mk_hc_operands_bitwise_with_pretranspose(use_combine, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_HC_UP_PRETRANSPOSE", "1")
    hc = _module(use_combine, seed=3)
    up_ckpt = hc.input_mix_weight_up.weight.data.clone()  # [W, LR]
    down = hc.input_mix_weight_down.weight.data.clone()  # [LR, W]
    hdn_off, ups_off = _pack(hc)

    hc.transpose_up_weight()
    assert hc._up_transposed
    assert tuple(hc.input_mix_weight_up.weight.shape) == (LR, NHC * H)
    hdn_on, ups_on = _pack(hc)

    assert torch.equal(hdn_on, hdn_off)
    assert len(ups_on) == len(ups_off) == NHC
    for b in range(NHC):
        assert torch.equal(ups_on[b], ups_off[b])
        # The semantics: rows [:LR] of branch b are the checkpoint up.t() slice,
        # never `down` (same [LR, W] shape as the transposed up).
        ref = up_ckpt.t()[:, b * H : (b + 1) * H]
        assert torch.equal(ups_on[b][:LR], ref)
        assert not torch.equal(ups_on[b][:LR], down[:, b * H : (b + 1) * H])
        assert not ups_on[b][LR:].any()
    assert torch.equal(hdn_on[:, :LR], down.t())


@pytest.mark.usefixtures("cpu_vllm_config")
def test_mk_packer_follows_flag_not_shape(monkeypatch):
    """A transposed weight with the flag cleared must fail loudly, not pack."""
    monkeypatch.setenv("VLLM_TPU_HC_UP_PRETRANSPOSE", "1")
    hc = _module(True, seed=5)
    hc.transpose_up_weight()
    hc._up_transposed = False
    with pytest.raises(AssertionError):
        _pack(hc)


@pytest.mark.usefixtures("cpu_vllm_config")
def test_pretranspose_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_HC_UP_PRETRANSPOSE", raising=False)
    monkeypatch.delenv("TPU_QWEN4_HC_DECODE_FUSED", raising=False)
    hc = _module(True, seed=7)
    before = hc.input_mix_weight_up.weight
    hc.transpose_up_weight()
    assert not hc._up_transposed
    assert hc.input_mix_weight_up.weight is before
    # Unset follows TPU_QWEN4_HC_DECODE_FUSED.
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_FUSED", "1")
    hc.transpose_up_weight()
    assert hc._up_transposed


@pytest.mark.usefixtures("cpu_vllm_config")
@pytest.mark.parametrize("rows", [16, 96, 128])
@pytest.mark.parametrize("use_combine", [True, False])
def test_stock_mix_bitwise_with_pretranspose(rows, use_combine, monkeypatch):
    """The stock (Torch/XLA) mix: F.linear on [W, LR] vs matmul on [LR, W]."""
    monkeypatch.setenv("VLLM_TPU_HC_UP_PRETRANSPOSE", "1")
    hc = _module(use_combine, seed=11)
    x = (torch.randn(rows, NHC * H, generator=torch.Generator().manual_seed(1))).to(
        torch.bfloat16
    )
    mixed_off, (res_off, inj_off) = hc.mix(x)
    hc.transpose_up_weight()
    mixed_on, (res_on, inj_on) = hc.mix(x)
    assert torch.equal(mixed_on, mixed_off)
    assert torch.equal(res_on, res_off)
    assert torch.equal(inj_on, inj_off)
