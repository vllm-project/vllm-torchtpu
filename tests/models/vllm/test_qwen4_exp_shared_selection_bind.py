# SPDX-License-Identifier: Apache-2.0
"""Qwen4ExpMTP.bind_shared_selection runs at model load, before the pool is
bound (and with it the shared QSA ops), so it must bind the buffers then."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp import mtp as mtp_module

WIDTH, TOKENS = 7, 16


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_selected_tokens = WIDTH
        self.rope_cache = torch.zeros(1)


def _layer(layer_type):
    layer = nn.Module()
    layer.layer_type = layer_type
    layer.self_attn = _Attention()
    return layer


def _models():
    target = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                _layer("full_attention"),
                _layer("linear_attention"),
                _layer("full_attention"),
            ]
        )
    )
    draft = mtp_module.Qwen4ExpMTP.__new__(mtp_module.Qwen4ExpMTP)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.layers = nn.ModuleList([_layer("full_attention")])
    return target, draft


def test_binds_before_the_shared_ops_exist(monkeypatch):
    monkeypatch.setattr(
        mtp_module.vllm_torchtpu_envs, "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", True
    )
    target, draft = _models()
    assert draft.bind_shared_selection(target, TOKENS)
    sink = target.model.layers[2].self_attn._selection_sink  # the last QSA layer
    assert sink.shape == (TOKENS, WIDTH) and sink.dtype == torch.int32
    assert not hasattr(target.model.layers[0].self_attn, "_selection_sink")
    attention = draft.model.layers[0].self_attn
    assert attention._selection_source is sink
    assert attention._selection_rows.shape == sink.shape
    assert attention._selection_rows.data_ptr() != sink.data_ptr()


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(
        mtp_module.vllm_torchtpu_envs, "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", False
    )
    target, draft = _models()
    assert not draft.bind_shared_selection(target, TOKENS)
    assert not hasattr(target.model.layers[2].self_attn, "_selection_sink")
