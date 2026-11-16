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
"""Qwen4ExpMTP.bind_shared_selection runs at model load, before the pool is
bound (and with it the shared QSA ops), so it must bind the buffers then."""

from types import SimpleNamespace

import torch
from torch import nn

from tests.models.qwen4_exp.experimental.qwen38_test_utils import pin_plan
from vllm_torchtpu.models.qwen4_exp.experimental import mtp as mtp_module

WIDTH, TOKENS = 7, 16


def _shared_selection(monkeypatch, on):
    pin_plan(monkeypatch, VLLM_TPU_QSA_DRAFT_SHARED_SELECTION=on)


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
    _shared_selection(monkeypatch, True)
    target, draft = _models()
    assert draft.bind_shared_selection(target, TOKENS)
    sink = target.model.layers[2].self_attn._selection_sink  # the last QSA layer
    assert sink.shape == (TOKENS, WIDTH) and sink.dtype == torch.int32
    assert not hasattr(target.model.layers[0].self_attn, "_selection_sink")
    attention = draft.model.layers[0].self_attn
    assert attention._selection_source is sink
    assert attention._selection_rows.shape == sink.shape
    assert attention._selection_rows.data_ptr() != sink.data_ptr()


def test_off_when_the_plan_disables_it(monkeypatch):
    _shared_selection(monkeypatch, False)
    target, draft = _models()
    assert not draft.bind_shared_selection(target, TOKENS)
    assert not hasattr(target.model.layers[2].self_attn, "_selection_sink")
