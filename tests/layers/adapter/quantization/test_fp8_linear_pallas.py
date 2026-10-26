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
"""Host-side checks of the TPU_FP8_LINEAR_PALLAS wiring: site selection, which layers
are armed, and that calls the kernel does not serve are the XLA path exactly. The
kernel itself is tested on TPU in tests/kernels/test_fp8_skinny_matmul.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_ATTR,
    KEEP_VLLM_LAYOUT_ATTR,
    _fp8_linear_pallas_jax,
)
from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.3.attn.fused_wqa_wkv", True),
        ("model.layers.3.attn.wq_b", True),
        ("model.layers.3.attn.wo_b", True),
        ("model.layers.0.ffn.shared_experts.gate_up_proj", False),
        ("model.layers.0.ffn.shared_experts.down_proj", False),
        ("model.layers.3.attn.indexer.wq_b", True),
        ("model.layers.3.attn.wo_a", False),
        ("model.layers.3.xattn.wq_b", False),  # "."-bounded suffix
        ("", False),
    ],
)
def test_default_sites(monkeypatch, prefix, expected):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.delenv("TPU_FP8_LINEAR_PALLAS_SITES", raising=False)
    assert fp8_mod.fp8_linear_pallas_selected(prefix) is expected


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("TPU_FP8_LINEAR_PALLAS", raising=False)
    monkeypatch.delenv("TPU_FP8_LINEAR_PALLAS_SITES", raising=False)
    assert fp8_mod.fp8_linear_pallas_selected("model.layers.3.attn.wo_b")


def test_off_when_disabled(monkeypatch):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "0")
    assert not fp8_mod.fp8_linear_pallas_selected("model.layers.3.attn.wo_b")


def test_custom_site_list(monkeypatch):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS_SITES", " attn.wo_b , ")
    assert fp8_mod.fp8_linear_pallas_selected("model.layers.1.attn.wo_b")
    assert not fp8_mod.fp8_linear_pallas_selected("model.layers.1.attn.wq_b")


def _layer(k, n, scale_ndim=1):
    layer = torch.nn.Module()
    w = torch.zeros(k, n, dtype=torch.float8_e4m3fn)
    s = torch.ones(n if scale_ndim == 1 else (1, n), dtype=torch.float32)
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    return layer


@pytest.mark.parametrize("pin_sites,pinned", [("", False), ("attn.wo_b", True)])
def test_arms_selected_measured_layer(monkeypatch, pin_sites, pinned):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.setenv("TPU_FP8_SKINNY_HBM_PIN_SITES", pin_sites)
    monkeypatch.setitem(fp8_skinny.TUNED, (256, 128), ("n", 128, 1, 1))
    monkeypatch.setitem(fp8_skinny.TUNED_HBM, (256, 128), ("n", 128, 1, 1))
    layer = _layer(256, 128)
    fp8_mod._arm_fp8_linear_pallas(layer, "model.layers.0.attn.wo_b")
    assert getattr(layer, FP8_PALLAS_ATTR) is pinned


@pytest.mark.parametrize(
    "prefix,shape,scale_ndim,keep_layout",
    [
        ("model.layers.0.attn.wq_a", (256, 128), 1, False),  # site not listed
        ("model.layers.0.attn.wo_b", (256, 256), 1, False),  # shape not measured
        ("model.layers.0.attn.wo_b", (256, 128), 2, False),  # not per-channel 1-D
        ("model.layers.0.attn.wo_b", (256, 128), 1, True),  # raw [n, k] consumer
    ],
)
def test_leaves_other_layers_on_xla(
    monkeypatch, prefix, shape, scale_ndim, keep_layout
):
    monkeypatch.setenv("TPU_FP8_LINEAR_PALLAS", "1")
    monkeypatch.setitem(fp8_skinny.TUNED, (256, 128), ("n", 128, 1, 1))
    monkeypatch.setitem(fp8_skinny.TUNED_HBM, (256, 128), ("n", 128, 1, 1))
    layer = _layer(*shape, scale_ndim)
    if keep_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    fp8_mod._arm_fp8_linear_pallas(layer, prefix)
    assert not hasattr(layer, FP8_PALLAS_ATTR)


@pytest.mark.parametrize("m,k,n", [(fp8_skinny.MAX_M + 1, 7168, 2048), (4, 256, 128)])
def test_unserved_calls_are_the_xla_path(m, k, n):
    """Prefill row counts and unmeasured shapes compute exactly what XLA does."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((m, k), dtype=np.float32), jnp.bfloat16)
    w = jnp.asarray(
        np.clip(rng.standard_normal((k, n)) * 50, -448, 448), jnp.float8_e4m3fn
    )
    s = jnp.asarray(rng.random(n) + 0.5, jnp.float32)
    ref = jax.jit(xla_quantized_matmul)(x, w, s)
    y = jax.jit(_fp8_linear_pallas_jax)(x, w, s)
    np.testing.assert_array_equal(
        np.asarray(y, np.float32), np.asarray(ref, np.float32)
    )
