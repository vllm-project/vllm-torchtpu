# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp import layer_digest


class _Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.layer_index = index
        self.attn_hyper_connection = object()
        self.weight = nn.Parameter(torch.zeros(1))


def _model(n=3):
    return SimpleNamespace(model=SimpleNamespace(layers=[_Layer(i) for i in range(n)]))


def test_digest_tells_one_bit_apart():
    x = torch.randn(64, 40).to(torch.bfloat16)
    y = x.clone()
    assert torch.equal(layer_digest.digest(x), layer_digest.digest(y))
    y.view(torch.int16)[17, 3] ^= 1
    assert not torch.equal(layer_digest.digest(x), layer_digest.digest(y))
    # Same words in another order: the weighted sum differs.
    z = x.flip(0)
    assert layer_digest.digest(x)[0] == layer_digest.digest(z)[0]
    assert layer_digest.digest(x)[1] != layer_digest.digest(z)[1]


def test_digest_reads_32_bit_words():
    x = torch.randn(8, 8)
    d = layer_digest.digest(x)
    assert d.dtype == torch.int32
    assert d[0] == x.view(torch.int32).sum(dtype=torch.int32)


def test_parse():
    assert layer_digest.parse("") is None
    assert layer_digest.parse("2048:16") == (2048, 16)
    assert layer_digest.parse("512") == (512, 1)


def test_unset_binds_nothing():
    model = _model()
    with patch.dict(os.environ, {"VLLM_TPU_LAYER_DIGEST": ""}):
        assert layer_digest.bind(model) == 0
    assert all(
        getattr(layer, "_layer_digest", None) is None for layer in model.model.layers
    )


@patch.dict(os.environ, {"VLLM_TPU_LAYER_DIGEST": "16:2"})
def test_logs_steps_of_the_configured_size_up_to_the_count():
    layer_digest._state["taken"] = 0
    model = _model()
    assert layer_digest.bind(model) == 3
    for layer in model.model.layers:
        assert torch.equal(
            layer._layer_digest, torch.full((4, 2), -1, dtype=torch.int32)
        )
    model.model.layers[1]._layer_digest.copy_(
        torch.tensor([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=torch.int32)
    )

    assert layer_digest.after_forward(model, 15, 0) is None  # other size
    assert layer_digest.after_forward(model, 16, 1) is None  # later chunk
    line = layer_digest.after_forward(model, 16, 0)
    assert line is not None
    assert "n=1" in line and "tokens=16" in line
    assert " 1:00000001.00000002.00000003.00000004 " in line + " "
    assert layer_digest.after_forward(model, 16, 0) is not None
    assert layer_digest.after_forward(model, 16, 0) is None  # count reached


class _IdentityGDN(nn.Module):
    """The GDN mixer as identity; the decoder also passes it the step's GDN
    metadata cache."""

    def forward(self, x, *args, **kwargs):
        return x


class _FakeHC(nn.Module):
    def mix(self, hidden):
        return hidden * 2, hidden

    def combine_mix(self, block, residual, nxt):
        return block + 1, residual

    def combine(self, moe, residual):
        return moe + residual


@patch.dict(os.environ, {"VLLM_TPU_LAYER_DIGEST": "8:1"})
def test_decoder_layer_writes_the_four_points(monkeypatch):
    from unittest.mock import Mock

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import model
    from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

    text = q4exp_text_config(ple_layer_ids=[], layer_types=["linear_attention"])
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text), additional_config={}
    )
    monkeypatch.setattr(
        model, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        model, "Qwen4ExpGatedDeltaNetAttention", Mock(return_value=_IdentityGDN())
    )
    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", lambda **kw: nn.Identity())
    monkeypatch.setattr(model, "GatedResidual", lambda *a, **kw: _FakeHC())
    monkeypatch.setattr(sp, "pcp_world_size", lambda: 1)
    written = {}
    monkeypatch.setattr(
        model, "update_rows_", lambda buf, rows: written.setdefault("rows", rows)
    )

    layer = model.Qwen4ExpDecoderLayer(cfg, 0, "model.layers.0")
    layer.register_buffer(
        "_layer_digest", torch.full((4, 2), -1, dtype=torch.int32), persistent=False
    )
    hidden = torch.randn(8, 16).to(torch.bfloat16)
    valid = torch.ones(8, dtype=torch.bool)
    out = layer(
        hidden,
        None,
        torch.zeros(8, dtype=torch.int32),
        None,
        None,
        None,
        None,
        valid,
        None,
        None,
    )
    block = hidden * 2
    after = block + 1
    expected = torch.stack(
        [
            layer_digest.digest(block),
            layer_digest.digest(block),
            layer_digest.digest(after),
            layer_digest.digest(out),
        ]
    )
    assert torch.equal(written["rows"], expected)
    assert torch.equal(out, after + hidden)
