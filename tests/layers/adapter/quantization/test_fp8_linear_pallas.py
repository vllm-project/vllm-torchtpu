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
"""Host-only checks of the TPU_FP8_LINEAR_PALLAS wiring: site selection, load-time
padding, and the M > 16 fallback (XLA on the stored weight, output cut back). The
kernel is tested on TPU in tests/kernels/test_fp8_skinny_matmul.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter.linear_common import _fp8_linear_pallas_jax
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.3.self_attn.fused_qkv_a_proj", True),
        ("model.layers.3.self_attn.q_b_proj", True),
        ("model.layers.3.self_attn.o_proj", True),
        ("model.layers.0.mlp.gate_up_proj", False),  # opt-in via SITES
        ("model.layers.5.mlp.shared_experts.gate_up_proj", False),
        ("model.layers.5.self_attn.indexer.wq_b", False),
        ("model.layers.0.mlp.down_proj", False),
        ("model.layers.3.self_attn.xq_b_proj", False),  # "."-bounded suffix
        ("", False),
    ],
)
def test_site_selection(monkeypatch, prefix, expected):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.delenv("TPU_FP8_LINEAR_PALLAS_SITES", raising=False)
    assert fp8_mod._fp8_linear_pallas_selected(prefix) is expected


def test_site_selection_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_FP8_LINEAR_PALLAS", raising=False)
    assert not fp8_mod._fp8_linear_pallas_selected("model.layers.3.self_attn.o_proj")


def test_site_selection_custom_list(monkeypatch):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS_SITES", " q_b_proj , ")
    assert fp8_mod._fp8_linear_pallas_selected("a.q_b_proj")
    assert not fp8_mod._fp8_linear_pallas_selected("a.self_attn.o_proj")
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS_SITES", "mlp.gate_up_proj")
    assert fp8_mod._fp8_linear_pallas_selected("model.layers.0.mlp.gate_up_proj")


def test_padded_n():
    assert fp8_skinny.padded_n(2624) == 2688
    assert fp8_skinny.padded_n(6144) == 6144
    assert fp8_skinny.padded_n(1) == 128


def _fp8_layer(n_in, n_out, seed=0):
    g = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    w = (torch.randn(n_in, n_out, generator=g) * 10).to(torch.float8_e4m3fn)
    s = torch.rand(n_out, generator=g, dtype=torch.float32) + 0.5
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    return layer


def test_padded_kernel_copy_moves_bytes():
    layer = _fp8_layer(64, 200)
    fp8_mod._set_fp8_pallas_copy(layer)
    wp, sp = layer.weight_pallas, layer.weight_scale_pallas
    assert wp.shape == (64, 256) and wp.dtype == torch.float8_e4m3fn
    assert torch.equal(wp[:, :200].view(torch.uint8), layer.weight.view(torch.uint8))
    assert torch.all(wp[:, 200:].view(torch.uint8) == 0)
    assert torch.equal(sp[:200], layer.weight_scale) and torch.all(sp[200:] == 1.0)
    # the original weight is untouched (the M > 16 path keeps reading it)
    assert layer.weight.shape == (64, 200)
    # rebuilding (the reload path) replaces, not duplicates, the copy
    fp8_mod._set_fp8_pallas_copy(layer)
    assert sorted(n for n, _ in layer.named_parameters()) == [
        "weight",
        "weight_pallas",
        "weight_scale",
        "weight_scale_pallas",
    ]


@pytest.mark.parametrize("m", [17, 32, 4096])
def test_fallback_path_reads_the_original_weight(m):
    """M > 16 stays on XLA and reads the original (unpadded) weight, so it is the
    incumbent's computation exactly, with or without a padded kernel copy."""
    k, n, n_pad = 256, 200, 256
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32), jnp.bfloat16)
    w = np.clip(rng.standard_normal((k, n)) * 50, -448, 448)
    w = jnp.asarray(w, jnp.float8_e4m3fn)
    s = jnp.asarray(rng.random(n) + 0.5, jnp.float32)
    wp = jnp.pad(w.astype(jnp.float32), ((0, 0), (0, n_pad - n)))
    wp = wp.astype(jnp.float8_e4m3fn)
    sp = jnp.pad(s, (0, n_pad - n), constant_values=1.0)
    ref = np.asarray(jax.jit(xla_quantized_matmul)(x, w, s), np.float32)
    y1 = jax.jit(_fp8_linear_pallas_jax)(x, w, s, wp, sp)
    y0 = jax.jit(_fp8_linear_pallas_jax)(x, w, s)
    np.testing.assert_array_equal(np.asarray(y1, np.float32), ref)
    np.testing.assert_array_equal(np.asarray(y0, np.float32), ref)


@pytest.mark.parametrize("n_out,copy", [(16, True), (128, False)])
def test_post_load_hook_selects_and_pads(monkeypatch, n_out, copy):
    """End to end through vLLM's post-load pass (online FP8 on a BF16 checkpoint, as
    tests/layers/adapter/quantization/test_online_fp8.py does): a selected layer is
    marked, a non-lane-multiple width gets the padded kernel copy, the original
    weight keeps the incumbent's flipped layout, and a second pass is idempotent."""
    from types import SimpleNamespace

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from vllm_torchtpu.layers.adapter.linear_common import (
        FP8_PALLAS_N_ATTR,
        WEIGHT_FLIPPED_ATTR,
    )
    from vllm_torchtpu.layers.adapter.quantization.online_fp8 import (
        attach_online_fp8,
    )

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS_SITES", "proj")
    # only measured (K, N) shapes are routed; register this toy one
    monkeypatch.setitem(fp8_skinny.TUNED, (32, 128), ("n", 128, 1))
    config = fp8_mod.VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["proj"],
        }
    )
    attach_online_fp8(
        config,
        SimpleNamespace(
            additional_config={
                "tpu_online_fp8": {
                    "enabled": True,
                    "targets": ["proj"],
                    "weight_scheme": "per_channel",
                }
            }
        ),
    )
    layer = ReplicatedLinear(
        32,
        n_out,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="proj",
        disable_tp=True,
    )
    model = torch.nn.Module()
    model.add_module("proj", layer)
    src = torch.linspace(-3, 3, n_out * 32).reshape(n_out, 32).to(torch.bfloat16)
    layer.weight.weight_loader(layer.weight, src)
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        quantization=None,
        word_embeddings_untied_by_checkpoint=False,
    )
    process_weights_after_loading(model, model_config, torch.device("cpu"))
    assert getattr(layer, FP8_PALLAS_N_ATTR) == n_out
    assert getattr(layer, WEIGHT_FLIPPED_ATTR)
    assert layer.weight.shape == (32, n_out)
    assert hasattr(layer, "weight_pallas") is copy
    monkeypatch.delitem(fp8_skinny.TUNED, (32, 128))
    other = ReplicatedLinear(
        32,
        n_out,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="proj",
        disable_tp=True,
    )
    other.weight.weight_loader(other.weight, src)
    other.quant_method.process_weights_after_loading(other)
    assert not hasattr(other, FP8_PALLAS_N_ATTR), "untuned shapes stay on XLA"
    monkeypatch.setitem(fp8_skinny.TUNED, (32, 128), ("n", 128, 1))
    if copy:
        assert layer.weight_pallas.shape == (32, 128)
        raw = layer.weight_pallas.view(torch.uint8)
        assert torch.equal(raw[:, :n_out], layer.weight.view(torch.uint8))
    before = {n: p.detach().clone() for n, p in layer.named_parameters()}
    process_weights_after_loading(model, model_config, torch.device("cpu"))
    after = dict(layer.named_parameters())
    assert sorted(before) == sorted(after)
    for n, p in before.items():
        assert torch.equal(p.view(torch.uint8), after[n].detach().view(torch.uint8))


class _Recorder:
    """Stands in for a matmul entry point: records its args, returns a fixed output."""

    def __init__(self, out):
        self.out, self.calls = out, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.out


def _apply(monkeypatch, layer, x, bias=None):
    """Run `VllmFp8LinearMethodTPU.apply` with both matmul paths recorded."""
    out_xla, out_pallas = torch.full((4, 200), 1.0), torch.full((4, 200), 2.0)
    xla, pallas = _Recorder(out_xla), _Recorder(out_pallas)
    monkeypatch.setattr(fp8_mod, "quantized_matmul", xla)
    monkeypatch.setattr(fp8_mod, "quantized_matmul_pallas", pallas)
    method = object.__new__(fp8_mod.VllmFp8LinearMethodTPU)  # apply() reads no state
    y = method.apply(layer, x, bias)
    return y, xla, pallas


def _same(got, want):
    return all(a is b for a, b in zip(got, want, strict=True))


def test_apply_unmarked_layer_takes_xla(monkeypatch):
    layer, x = _fp8_layer(64, 200), torch.randn(4, 64)
    y, xla, pallas = _apply(monkeypatch, layer, x)
    assert pallas.calls == [] and len(xla.calls) == 1
    assert _same(xla.calls[0], (x, layer.weight, layer.weight_scale))
    assert y is xla.out


def test_apply_marker_none_is_unmarked(monkeypatch):
    layer = _fp8_layer(64, 200)
    setattr(layer, fp8_mod.FP8_PALLAS_N_ATTR, None)
    _, xla, pallas = _apply(monkeypatch, layer, torch.randn(4, 64))
    assert pallas.calls == [] and len(xla.calls) == 1


def test_apply_marked_layer_takes_pallas(monkeypatch):
    layer, x = _fp8_layer(64, 256), torch.randn(4, 64)
    setattr(layer, fp8_mod.FP8_PALLAS_N_ATTR, 256)
    y, xla, pallas = _apply(monkeypatch, layer, x)
    assert xla.calls == [] and len(pallas.calls) == 1
    assert _same(pallas.calls[0][:3], (x, layer.weight, layer.weight_scale))
    assert pallas.calls[0][3:] == (None, None)  # lane-aligned: no padded copy
    assert y is pallas.out


def test_apply_marked_layer_passes_padded_copy(monkeypatch):
    layer, x = _fp8_layer(64, 200), torch.randn(4, 64)
    setattr(layer, fp8_mod.FP8_PALLAS_N_ATTR, 200)
    fp8_mod._set_fp8_pallas_copy(layer)
    _, xla, pallas = _apply(monkeypatch, layer, x)
    assert xla.calls == [] and len(pallas.calls) == 1
    want = (x, layer.weight, layer.weight_scale)
    want += (layer.weight_pallas, layer.weight_scale_pallas)
    assert _same(pallas.calls[0], want)


@pytest.mark.parametrize("marked", [False, True])
def test_apply_adds_bias_on_both_paths(monkeypatch, marked):
    layer = _fp8_layer(64, 200)
    if marked:
        setattr(layer, fp8_mod.FP8_PALLAS_N_ATTR, 200)
    bias = torch.arange(200, dtype=torch.float32)
    y, xla, pallas = _apply(monkeypatch, layer, torch.randn(4, 64), bias)
    taken = pallas if marked else xla
    assert torch.equal(y, taken.out + bias)


def test_apply_keep_vllm_layout_transposes_for_xla(monkeypatch):
    """A raw-layout [n_out, n_in] weight reaches XLA as its (k, n) view."""
    layer = _fp8_layer(200, 64)
    setattr(layer, fp8_mod.KEEP_VLLM_LAYOUT_ATTR, True)
    _, xla, _ = _apply(monkeypatch, layer, torch.randn(4, 64))
    w = xla.calls[0][1]
    assert w.shape == (64, 200)
    assert torch.equal(
        w.view(torch.uint8), layer.weight.transpose(0, 1).view(torch.uint8)
    )
