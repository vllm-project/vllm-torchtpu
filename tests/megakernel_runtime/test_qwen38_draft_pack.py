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
"""MK draft packing (VLLM_TPU_MK_DRAFT) from the served module classes.

The draft's final mixer is the served `GatedResidual(hc_config(t), use_combine=False)`
(`Qwen4ExpMultiTokenPredictor.hyper_connection_mixer`): it has no injection weight.
`pack_tail_operands` must take it as built (zero injection columns), and every packed
tensor keeps its loaded dtype.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch


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


@pytest.mark.usefixtures("cpu_vllm_config")
@pytest.mark.parametrize("use_combine", [False, True])
def test_tail_operands_from_served_mixer(use_combine):
    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
    from vllm_torchtpu.megakernel.wholestep.tail import vocab_chunk_width, vocab_chunks
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_weights import (
        pack_tail_operands,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual
    from vllm_torchtpu.models.vllm.qwen4_exp.model import hc_config

    H, NHC, LR, V = 256, 4, 16, 640
    text = q4exp_text_config(
        hidden_size=H, hc_count=NHC, hc_lowrank=LR, vocab_size=V, num_hidden_layers=2
    )
    with torch.device("cpu"):
        mixer = GatedResidual(hc_config(text), use_combine=use_combine)
    for p in mixer.parameters():
        p.data.normal_(0, 0.1)
    assert hasattr(mixer, "block_inject_weight") == use_combine
    lm = NS(weight=torch.randn(V, H).to(torch.bfloat16))  # [vocab shard, H] as loaded
    g = NS(NHC=NHC, H=H, LOWRANK=LR)
    ex = ExecSpec(R=1, B=4, T=1, ctx=128, interpret=True,
                  tiles=Tiles(bank=128 * 256 * 2, nbd=10, nbe=8, ebank=512 << 10))
    w, hcn = pack_tail_operands(mixer, lm, V, g, ex)

    W = NHC * H
    down = mixer.input_mix_weight_down.weight.data  # [LR, W]
    hdn = w["tail_hdn"][0]
    assert hdn.dtype == torch.bfloat16 and tuple(hdn.shape) == (W, 128)
    torch.testing.assert_close(hdn[:, :LR], down.t(), rtol=0, atol=0)
    inj = hdn[:, LR : LR + NHC]
    if use_combine:
        torch.testing.assert_close(
            inj, mixer.block_inject_weight.weight.data.t(), rtol=0, atol=0
        )
    else:
        assert not inj.any()
    assert not hdn[:, LR + NHC :].any()
    for b in range(NHC):
        up = w[f"tail_hup{b}"][0]
        assert up.dtype == torch.bfloat16 and tuple(up.shape) == (128, H)
    chunks = vocab_chunks(V, vocab_chunk_width(ex))
    assert len(chunks) == 3  # 256-wide blocks at this bank: 256, 256, 128
    cols = torch.cat([w[name][0][:, :width] for name, _, width in chunks], dim=1)
    assert cols.dtype == torch.bfloat16
    torch.testing.assert_close(cols[:, :V], lm.weight.t(), rtol=0, atol=0)
    assert hcn.dtype == mixer.hc_norm.weight.dtype
