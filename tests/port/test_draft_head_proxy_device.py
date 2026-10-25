# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port: a quantized draft lm-head proxy built from a target head that is already
fp8 (no weight; dequantized on the host) must carry its buffers on the head's device. Band H
(TPU_DRAFT_LM_HEAD_QUANT=fp4) crashed in capture_model with a CPU _w4a8_weight_3d argument."""
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import greedy_top_tokens as gtt


def test_head_device_prefers_weight_then_fp8_buffer():
    h = torch.nn.Module()
    assert gtt._head_device(h) is None
    h.register_buffer("_w8a8_weight_kn", torch.zeros(2, 2, device="meta"), persistent=False)
    assert gtt._head_device(h) == torch.device("meta")
    h.weight = torch.nn.Parameter(torch.zeros(2, 2))
    assert gtt._head_device(h) == torch.device("cpu")


def test_move_buffers_to_moves_own_buffers_only():
    m = torch.nn.Module()
    m.register_buffer("a", torch.ones(3), persistent=False)
    m.child = torch.nn.Module()
    m.child.register_buffer("b", torch.ones(3), persistent=False)
    gtt._move_buffers_to(m, torch.device("meta"))
    assert m.a.device.type == "meta"
    assert m.child.b.device.type == "cpu"
    gtt._move_buffers_to(m, None)  # no-op
    assert m.a.device.type == "meta"


def _fp8_head():
    h = torch.nn.Module()
    h.tp_size = 1
    h.shard_indices = SimpleNamespace()
    h._quant_mode = "fp8"
    h.register_buffer("_w8a8_weight_kn", torch.ones(256, 16).to(torch.float8_e4m3fn), persistent=False)
    h.register_buffer("_w8a8_scale", torch.ones(16), persistent=False)
    return h


def test_proxy_moves_its_buffers_to_the_base_head_device(monkeypatch):
    def fake_quantize(head, mode, group_size, weight):
        assert weight.device.type == "cpu"  # dequantized on the host
        head.register_buffer("_w4a8_weight_3d", torch.zeros(2, 128, 16), persistent=False)
        head.register_buffer("_w4a8_scale_gn", torch.zeros(2, 16), persistent=False)

    monkeypatch.setattr(gtt, "quantize_lm_head", fake_quantize)
    monkeypatch.setattr(gtt, "_head_device", lambda head: torch.device("meta"))
    proxy = gtt._QuantizedLMHeadProxy(_fp8_head(), mode="fp4", group_size=128)
    assert proxy._w4a8_weight_3d.device.type == "meta"
    assert proxy._w4a8_scale_gn.device.type == "meta"


def test_proxy_on_cpu_head_keeps_cpu_buffers(monkeypatch):
    def fake_quantize(head, mode, group_size, weight):
        head.register_buffer("_w4a8_weight_3d", torch.zeros(2, 128, 16), persistent=False)

    monkeypatch.setattr(gtt, "quantize_lm_head", fake_quantize)
    proxy = gtt._QuantizedLMHeadProxy(_fp8_head(), mode="fp4", group_size=128)
    assert proxy._w4a8_weight_3d.device.type == "cpu"
