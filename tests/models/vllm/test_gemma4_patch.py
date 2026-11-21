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
"""Unit tests for the Gemma 4 dense sequence-parallelism patch."""

from __future__ import annotations

import inspect
import sys
import threading
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest
import torch
import torch.nn as nn
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4MLP,
)

from tests.tp_collective_test_utils import TPCollectiveMock
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import chip_topology, hier_reduce_scatter
from vllm_torchtpu.distributed.chip_topology import ChipTopology
from vllm_torchtpu.layers.adapter import dense_sequence_parallel
from vllm_torchtpu.layers.adapter.custom_ops import (
    fp8_linear_op,
    fused_rs_epilogue_op,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm import gemma4_patch

_UPSTREAM_FORWARD = Gemma4DecoderLayer.forward

HIDDEN = 24
HEADS = 8
HEAD_DIM = 4
INTERMEDIATE = 32
NUM_LAYERS = 3
NUM_TOKENS = 16


def _fp8_method(
    *,
    w8a16_max_rows: int = 0,
    weight_block_size: tuple[int, int] | None = None,
    activation_scheme: str = "dynamic",
) -> VllmFp8LinearMethodTPU:
    method = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    method.w8a16_max_rows = w8a16_max_rows
    method.weight_block_size = weight_block_size
    method.block_quant = weight_block_size is not None
    method.act_q_static = activation_scheme == "static"
    method.quant_config = SimpleNamespace(activation_scheme=activation_scheme)
    return method


class _Linear(LinearBase):
    """Test stand-in for a vLLM parallel linear with `(k, n)` weights."""

    def __init__(
        self,
        weight: torch.Tensor,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_method: object | None = None,
        prefix: str = "layers.0.mlp.gate_up_proj",
    ) -> None:
        nn.Module.__init__(self)
        self.prefix = prefix
        self.bias = None
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", torch.ones(weight.shape[1]))
        self.quant_method = _fp8_method() if quant_method is None else quant_method
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.reduce_results = True
        setattr(self, WEIGHT_FLIPPED_ATTR, True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = ((x @ self.weight) * self.weight_scale).to(x.dtype)
        return out, self.bias


def _fp8_linear(**overrides: object) -> LinearBase:
    layer = _Linear(torch.zeros(16, 32, dtype=torch.float8_e4m3fn))
    for name, value in overrides.items():
        if name in ("weight", "bias", "weight_scale"):
            layer._parameters.pop(name, None)
            layer._buffers.pop(name, None)
        setattr(layer, name, value)
    return layer


def test_fp8_custom_op_fake_registrations_and_dtype_guard() -> None:
    """Real custom ops trace cleanly on meta tensors and guard rank/dtype."""
    fp8_linear_op.register_fp8_linear_ops()
    x_meta = torch.empty((12, 16), dtype=torch.bfloat16, device="meta")
    x_q, x_scale = fp8_linear_op.quantize_fp8_activations(x_meta)
    assert (x_q.shape, x_q.dtype, x_q.device.type) == ((12, 16), torch.uint8, "meta")
    assert (x_scale.shape, x_scale.dtype, x_scale.device.type) == (
        (12, 1),
        torch.float32,
        "meta",
    )

    layer = _Linear(
        torch.empty((16, 32), dtype=torch.float8_e4m3fn, device="meta"),
    )
    layer.weight_scale = torch.empty((32,), dtype=torch.float32, device="meta")
    out = dense_sequence_parallel.apply_prequantized_fp8_linear(layer, x_q, x_scale)
    assert (out.shape, out.dtype, out.device.type) == ((12, 32), torch.bfloat16, "meta")

    with pytest.raises(NotImplementedError, match="2-D bfloat16"):
        fp8_linear_op.quantize_fp8_activations(
            torch.empty((12, 16), dtype=torch.float32, device="meta")
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"quant_method": SimpleNamespace()}, "SimpleNamespace"),
        ({"quant_method": _fp8_method(w8a16_max_rows=64)}, "w8a16_max_rows=0, got 64"),
        (
            {"quant_method": _fp8_method(weight_block_size=(128, 128))},
            "\\(128, 128\\)",
        ),
        (
            {"quant_method": _fp8_method(activation_scheme="static")},
            "activation_scheme='static'",
        ),
        ({KEEP_VLLM_LAYOUT_ATTR: True}, "canonical \\(k, n\\)"),
        ({"bias": torch.zeros(32)}, "bias shape \\(32,\\)"),
        ({WEIGHT_FLIPPED_ATTR: False}, f"{WEIGHT_FLIPPED_ATTR}=False"),
        ({"weight": torch.zeros(16, 32, dtype=torch.int8)}, "torch.int8"),
        ({"weight_scale": torch.ones(1, 2, 1, 32)}, "shape \\(1, 2, 1, 32\\)"),
    ],
    ids=[
        "other_method",
        "w8a16",
        "block_quant",
        "static_act",
        "keep_vllm_layout",
        "bias",
        "unflipped",
        "int8",
        "blockwise_scale",
    ],
)
def test_validate_prequantized_fp8_linear(
    overrides: dict[str, object], match: str
) -> None:
    """Each unsupported FP8 linear condition raises with its specific cause/value."""
    fp8_linear_op.register_fp8_linear_ops()
    x_q = torch.empty((4, 16), dtype=torch.uint8, device="meta")
    x_scale = torch.empty((4, 1), dtype=torch.float32, device="meta")

    valid = _Linear(torch.empty((16, 32), dtype=torch.float8_e4m3fn, device="meta"))
    valid.weight_scale = torch.empty((32,), dtype=torch.float32, device="meta")
    dense_sequence_parallel.validate_fp8_linear_for_sp(valid)
    assert dense_sequence_parallel.apply_prequantized_fp8_linear(
        valid, x_q, x_scale
    ).shape == (4, 32)

    bad = _fp8_linear(**overrides)
    with pytest.raises(NotImplementedError, match=match):
        dense_sequence_parallel.apply_prequantized_fp8_linear(bad, x_q, x_scale)


def test_maybe_patch_waits_for_gemma4_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies patch registration return contract (`False` retry vs `None` no-op)."""
    monkeypatch.setattr(envs, "TPU_DENSE_SEQUENCE_PARALLEL", False)
    assert gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(None) is None

    monkeypatch.setattr(envs, "TPU_DENSE_SEQUENCE_PARALLEL", True)
    assert gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(None) is False

    warn = Mock()
    monkeypatch.setattr(gemma4_patch.logger, "warning_once", warn)
    cfg_qwen = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text")
    )
    assert gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(cfg_qwen) is None
    warn.assert_called_once()

    registered = Mock()
    monkeypatch.setattr(fp8_linear_op, "register_fp8_linear_ops", registered)
    cfg_gemma4 = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4_text")
    )
    monkeypatch.setattr(Gemma4DecoderLayer, "__init__", Gemma4DecoderLayer.__init__)
    monkeypatch.setattr(Gemma4DecoderLayer, "forward", Gemma4DecoderLayer.forward)
    assert gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(cfg_gemma4) is None
    registered.assert_called_once()
    assert Gemma4DecoderLayer.__init__ is gemma4_patch._patched_layer_init
    assert Gemma4DecoderLayer.forward is gemma4_patch._dense_sequence_parallel_forward


def _rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class _TestRMSNorm(RMSNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.variance_epsilon = eps
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm_torch(x, self.weight, self.variance_epsilon)


def _make_weights(seed: int, dtype: torch.dtype = torch.float64) -> dict:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * 0.2).to(dtype)

    return {
        "qkv": rand(HIDDEN, 3 * HEADS * HEAD_DIM),
        "o": rand(HEADS * HEAD_DIM, HIDDEN),
        "gate_up": rand(HIDDEN, 2 * INTERMEDIATE),
        "down": rand(INTERMEDIATE, HIDDEN),
        "norms": [1.0 + rand(HIDDEN) for _ in range(4)],
        "layer_scalar": torch.tensor([0.75], dtype=dtype),
    }


def _column_shard(w: torch.Tensor, blocks: int, tp: int, rank: int) -> torch.Tensor:
    size = w.shape[1] // blocks
    part = size // tp
    return torch.cat(
        [
            w[:, b * size + rank * part : b * size + (rank + 1) * part]
            for b in range(blocks)
        ],
        dim=1,
    )


def _row_shard(w: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    part = w.shape[0] // tp
    return w[rank * part : (rank + 1) * part]


def _make_layer(
    weights: dict,
    layer_idx: int,
    tp: int = 1,
    rank: int = 0,
    num_layers: int = NUM_LAYERS,
) -> Gemma4DecoderLayer:
    attn = Gemma4Attention.__new__(Gemma4Attention)
    nn.Module.__init__(attn)
    attn.config = SimpleNamespace(num_hidden_layers=num_layers)
    attn.is_kv_shared_layer = False
    attn.num_heads = attn.num_kv_heads = HEADS // tp
    attn.head_dim = HEAD_DIM
    attn.q_size = attn.kv_size = attn.num_heads * HEAD_DIM
    attn.qkv_proj = _Linear(
        _column_shard(weights["qkv"], 3, tp, rank),
        prefix=f"layers.{layer_idx}.self_attn.qkv_proj",
    )
    attn.o_proj = _Linear(_row_shard(weights["o"], tp, rank), tp, rank)
    attn.q_norm = attn.k_norm = attn.v_norm = nn.Identity()
    attn.rotary_emb = lambda positions, q, k: (
        q + positions.unsqueeze(-1).to(q.dtype) * 0,
        k,
    )
    attn.attn = lambda q, k, v: torch.tanh(q) + torch.cumsum(k * v, dim=0) * 0.05

    mlp = Gemma4MLP.__new__(Gemma4MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = _Linear(
        _column_shard(weights["gate_up"], 2, tp, rank),
        prefix=f"layers.{layer_idx}.mlp.gate_up_proj",
    )
    mlp.down_proj = _Linear(_row_shard(weights["down"], tp, rank), tp, rank)
    mlp.act_fn = lambda gate_up: (
        nn.functional.gelu(gate_up.chunk(2, dim=-1)[0], approximate="tanh")
        * gate_up.chunk(2, dim=-1)[1]
    )

    layer = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.self_attn = attn
    layer.mlp = mlp
    (
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.pre_feedforward_layernorm,
        layer.post_feedforward_layernorm,
    ) = (_TestRMSNorm(w) for w in weights["norms"])
    layer.enable_moe_block = False
    layer.per_layer_input_gate = None
    layer.register_buffer("layer_scalar", weights["layer_scalar"])
    return layer


def _as_fp8_weights(layer: Gemma4DecoderLayer) -> Gemma4DecoderLayer:
    for proj in (layer.self_attn.qkv_proj, layer.mlp.gate_up_proj):
        proj.weight = nn.Parameter(
            proj.weight.to(torch.float8_e4m3fn), requires_grad=False
        )
        proj.weight_scale = proj.weight_scale.to(torch.float32)
    return layer


def test_patched_layer_init_matches_upstream_signature_and_configures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_patched_layer_init` matches upstream `Gemma4DecoderLayer.__init__`."""
    orig_params = [
        (p.name, p.kind, p.default)
        for p in inspect.signature(gemma4_patch._orig_layer_init).parameters.values()
    ]
    patched_params = [
        (p.name, p.kind, p.default)
        for p in inspect.signature(gemma4_patch._patched_layer_init).parameters.values()
    ]
    assert patched_params == orig_params

    captured: dict[str, object] = {}

    def fake_orig_init(
        self: Gemma4DecoderLayer,
        config: object,
        cache_config: object = None,
        quant_config: object = None,
        prefix: str = "",
    ) -> None:
        captured.update(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        built = _make_layer(_make_weights(0), NUM_LAYERS - 1, tp=2)
        built.self_attn.config = config
        self.__dict__.update(built.__dict__)

    monkeypatch.setattr(gemma4_patch, "_orig_layer_init", fake_orig_init)
    cfg = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    layer = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    gemma4_patch._patched_layer_init(
        layer, cfg, cache_config="cache", quant_config="quant", prefix="model.layers.2"
    )
    assert captured == {
        "config": cfg,
        "cache_config": "cache",
        "quant_config": "quant",
        "prefix": "model.layers.2",
    }
    assert layer.self_attn.o_proj.reduce_results is False
    assert gemma4_patch._is_last_layer(layer) is True


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"tp_size": 1}, ValueError, "TP > 1"),
        ({"enable_moe_block": True}, NotImplementedError, "dense Gemma 4"),
        ({"is_kv_shared_layer": True}, NotImplementedError, "dense Gemma 4"),
        ({"per_layer_input_gate": nn.Identity()}, NotImplementedError, "dense Gemma 4"),
        ({"o_proj_bias": True}, NotImplementedError, "bias-free"),
        ({"down_proj_bias": True}, NotImplementedError, "bias-free"),
        ({"qkv_non_fp8": True}, NotImplementedError, "qkv_proj.*SimpleNamespace"),
        (
            {"gate_up_w8a16": True},
            NotImplementedError,
            "gate_up_proj.*w8a16_max_rows=0, got 64",
        ),
        ({"gate_up_bias": True}, NotImplementedError, "gate_up_proj.*bias=None"),
        ({"bad_norm_type": True}, NotImplementedError, "RMSNorm"),
        ({"bad_norm_eps": True}, NotImplementedError, "variance_epsilon"),
    ],
)
def test_configure_layer(
    overrides: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Layer configuration disables row-parallel reductions or fails fast at init."""
    layer = _make_layer(_make_weights(0), NUM_LAYERS - 1, tp=2)
    gemma4_patch._configure_layer(layer)
    assert layer.self_attn.o_proj.reduce_results is False
    assert layer.mlp.down_proj.reduce_results is False
    assert gemma4_patch._is_last_layer(layer) is True

    for name, value in overrides.items():
        if name == "tp_size":
            layer.mlp.down_proj.tp_size = int(value)  # type: ignore[arg-type]
        elif name == "is_kv_shared_layer":
            layer.self_attn.is_kv_shared_layer = bool(value)
        elif name == "o_proj_bias":
            layer.self_attn.o_proj.bias = nn.Parameter(torch.zeros(HIDDEN))
        elif name == "down_proj_bias":
            layer.mlp.down_proj.bias = nn.Parameter(torch.zeros(HIDDEN))
        elif name == "qkv_non_fp8":
            layer.self_attn.qkv_proj.quant_method = SimpleNamespace()
        elif name == "gate_up_w8a16":
            layer.mlp.gate_up_proj.quant_method = _fp8_method(w8a16_max_rows=64)
        elif name == "gate_up_bias":
            layer.mlp.gate_up_proj.bias = nn.Parameter(torch.zeros(2 * INTERMEDIATE))
        elif name == "bad_norm_type":
            layer.post_attention_layernorm = nn.Identity()  # type: ignore[assignment]
        elif name == "bad_norm_eps":
            layer.post_attention_layernorm.variance_epsilon = 1e-5
        else:
            setattr(layer, name, value)
    with pytest.raises(error, match=match):
        gemma4_patch._configure_layer(layer)


def test_hier_rs_warns_when_hidden_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TPU_DENSE_SP_HIER_RS` logs `warning_once` when `hidden` exceeds VMEM/lane."""
    monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "bf16")
    warn = Mock()
    monkeypatch.setattr(gemma4_patch.logger, "warning_once", warn)
    layer = _make_layer(_make_weights(0), 0, tp=8)
    gemma4_patch._configure_layer(layer)
    state = gemma4_patch._layer_state(layer)
    assert state.streaming_collectives is None
    assert state.oneshot_collectives is None
    assert warn.call_count == 2


def _patch_collectives(monkeypatch: pytest.MonkeyPatch, **fakes: object) -> None:
    for name, fake in fakes.items():
        module = dense_sequence_parallel if name == "all_gather" else gemma4_patch
        monkeypatch.setattr(module, f"tensor_model_parallel_{name}", fake)


def _patch_fp8_ops_with_float_stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = x.abs().amax(-1, keepdim=True)
        return x / scale, scale

    monkeypatch.setattr(fp8_linear_op, "register_fp8_linear_ops", lambda: None)
    monkeypatch.setattr(fp8_linear_op, "quantize_fp8_activations", quantize_activations)
    monkeypatch.setattr(
        dense_sequence_parallel,
        "apply_prequantized_fp8_linear",
        lambda layer, x_q, x_scale: layer(x_q * x_scale)[0],
    )


def test_build_sublayer_collectives_registers_jax_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises streaming and oneshot `pallas.jax_op` registration and FakeTensors."""
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]), (fused_rs_epilogue_op.TP_AXIS_NAME,)
    )
    monkeypatch.setattr(
        fused_rs_epilogue_op,
        "_resolve_tp_mesh_and_order",
        lambda: (mesh, (0,), (0,)),
    )
    monkeypatch.setattr(fused_rs_epilogue_op, "_STREAM_COLLECTIVES_CACHE", {})
    monkeypatch.setattr(fused_rs_epilogue_op, "_ONESHOT_COLLECTIVES_CACHE", {})

    for fp8_wire in (False, True):
        ops = fused_rs_epilogue_op.build_streaming_sublayer_collectives(
            1e-6, fp8_wire=fp8_wire
        )
        assert ops is not None and ops.outputs_gathered is False
        ops_alt = fused_rs_epilogue_op.build_streaming_sublayer_collectives(
            1e-5, fp8_wire=fp8_wire
        )
        assert ops_alt is not None and ops_alt is not ops
        with torch._subclasses.FakeTensorMode():
            p = torch.empty((16, 32), dtype=torch.bfloat16)
            r = torch.empty((2, 32), dtype=torch.bfloat16)
            w = torch.empty((32,), dtype=torch.bfloat16)
            s = torch.empty((1,), dtype=torch.bfloat16)
            r1, q1, s1 = ops.norm_add_next_norm(p, r, w, w)
            assert (r1.shape, q1.shape, s1.shape) == ((2, 32), (2, 32), (2, 1))
            assert (r1.dtype, q1.dtype, s1.dtype) == (
                torch.bfloat16,
                torch.uint8,
                torch.float32,
            )
            r2, q2, s2 = ops.norm_add_scale_next_norm(p, r, w, s, w)
            assert (r2.shape, q2.shape, s2.shape) == ((2, 32), (2, 32), (2, 1))
            r3 = ops.norm_add_scale(p, r, w, s)
            assert (r3.shape, r3.dtype) == ((2, 32), torch.bfloat16)

    oneshot = fused_rs_epilogue_op.build_oneshot_sublayer_collectives(1e-6)
    assert oneshot is not None and oneshot.outputs_gathered is True
    oneshot_alt = fused_rs_epilogue_op.build_oneshot_sublayer_collectives(1e-5)
    assert oneshot_alt is not None and oneshot_alt is not oneshot
    with torch._subclasses.FakeTensorMode():
        p = torch.empty((16, 32), dtype=torch.bfloat16)
        r = torch.empty((2, 32), dtype=torch.bfloat16)
        w = torch.empty((32,), dtype=torch.bfloat16)
        s = torch.empty((1,), dtype=torch.bfloat16)
        r1, q1, s1 = oneshot.norm_add_next_norm(p, r, w, w)
        assert (r1.shape, q1.shape, s1.shape) == ((2, 32), (16, 32), (16, 1))
        assert (r1.dtype, q1.dtype, s1.dtype) == (
            torch.bfloat16,
            torch.uint8,
            torch.float32,
        )
        r2, q2, s2 = oneshot.norm_add_scale_next_norm(p, r, w, s, w)
        assert (r2.shape, q2.shape, s2.shape) == ((2, 32), (16, 32), (16, 1))
        r3 = oneshot.norm_add_scale(p, r, w, s)
        assert (r3.shape, r3.dtype) == ((16, 32), torch.bfloat16)


def test_topology_guard_rejects_non_tpu7x_2x2x1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejects 8-chip single-core slices (v6e-8/v5e-8), non-7x chips, and DP/PP."""
    valid_tpu7x = ChipTopology(
        [[0, 1], [4, 5], [2, 3], [6, 7]], [0, 1, 4, 5, 6, 7, 2, 3]
    )
    state = {
        "tp": 8,
        "world": 8,
        "topology": valid_tpu7x,
        "chip": pltpu.ChipVersion.TPU_7X,
    }
    monkeypatch.setattr(fused_rs_epilogue_op.dist, "is_available", lambda: True)
    monkeypatch.setattr(fused_rs_epilogue_op.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        fused_rs_epilogue_op.dist, "get_world_size", lambda: state["world"]
    )
    monkeypatch.setattr(
        fused_rs_epilogue_op.parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=state["tp"]),
    )
    monkeypatch.setattr(chip_topology, "get_chip_topology", lambda: state["topology"])
    monkeypatch.setattr(hier_reduce_scatter, "_chip_version", lambda: state["chip"])

    assert fused_rs_epilogue_op._unsupported_tp_topology_reason() is None

    state["topology"] = ChipTopology([[i] for i in range(8)], list(range(8)))
    assert "4 chips of 2 TensorCores" in (
        fused_rs_epilogue_op._unsupported_tp_topology_reason() or ""
    )

    state["topology"] = valid_tpu7x
    state["chip"] = pltpu.ChipVersion.TPU_V6E
    assert "tpu7x" in (fused_rs_epilogue_op._unsupported_tp_topology_reason() or "")

    state["chip"] = pltpu.ChipVersion.TPU_7X
    state["topology"] = ChipTopology([[0, 2], [1, 3], [4, 5], [6, 7]], list(range(8)))
    assert "chips are not paired" in (
        fused_rs_epilogue_op._unsupported_tp_topology_reason() or ""
    )

    state["topology"] = valid_tpu7x
    state["world"] = 16
    assert "whole torch.distributed world" in (
        fused_rs_epilogue_op._unsupported_tp_topology_reason() or ""
    )

    # TP=4 (two dual-core chips, e.g. one dense-DP replica) skips the host-wide
    # 4-chip check; other TP sizes and non-7x chips are still rejected.
    state["tp"] = state["world"] = 4
    state["topology"] = None
    assert fused_rs_epilogue_op._unsupported_tp_topology_reason() is None

    state["chip"] = pltpu.ChipVersion.TPU_V6E
    assert "tpu7x" in (fused_rs_epilogue_op._unsupported_tp_topology_reason() or "")

    state["chip"] = pltpu.ChipVersion.TPU_7X
    state["tp"] = state["world"] = 2
    assert "TP=4 or TP=8" in (
        fused_rs_epilogue_op._unsupported_tp_topology_reason() or ""
    )


def test_resolve_tp_mesh_and_order_refusal_agreement_and_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers `_resolve_tp_mesh_and_order` refusals, disagreement, and caching."""
    monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "bf16")
    monkeypatch.setattr(fused_rs_epilogue_op.dist, "is_available", lambda: True)
    monkeypatch.setattr(fused_rs_epilogue_op.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fused_rs_epilogue_op.dist, "get_world_size", lambda: 8)

    def reset() -> Mock:
        monkeypatch.setattr(fused_rs_epilogue_op, "_RESOLVED_TP_TOPOLOGY", None)
        monkeypatch.setattr(fused_rs_epilogue_op, "_REFUSED_REASON", None)
        warn = Mock()
        monkeypatch.setattr(fused_rs_epilogue_op.logger, "warning_once", warn)
        return warn

    def fake_mesh(ids: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            devices=np.asarray([SimpleNamespace(id=i) for i in ids], dtype=object)
        )

    peer_reasons: list[str | None] = [None] * 6 + ["bad peer"]
    gather_calls = 0

    def counting_gather(r: str | None) -> list[str | None]:
        nonlocal gather_calls
        gather_calls += 1
        return [r, *peer_reasons]

    monkeypatch.setattr(hier_reduce_scatter, "_gather_reasons", counting_gather)
    monkeypatch.setattr(
        fused_rs_epilogue_op, "_unsupported_tp_topology_reason", lambda: None
    )
    warn = reset()
    assert fused_rs_epilogue_op._resolve_tp_mesh_and_order() is None
    assert "ranks disagree" in (fused_rs_epilogue_op._REFUSED_REASON or "")
    warn.assert_called_once()
    assert gather_calls == 1
    assert fused_rs_epilogue_op._resolve_tp_mesh_and_order() is None
    assert gather_calls == 1

    peer_reasons = [None] * 7
    monkeypatch.setattr(fused_rs_epilogue_op, "build_tp_mesh", lambda: None)
    warn = reset()
    assert fused_rs_epilogue_op._resolve_tp_mesh_and_order() is None
    assert "whole torch.distributed world" in (
        fused_rs_epilogue_op._REFUSED_REASON or ""
    )
    warn.assert_called_once()

    monkeypatch.setattr(
        fused_rs_epilogue_op,
        "build_tp_mesh",
        lambda: (fake_mesh(list(range(8, 16))), tuple(range(8))),
    )
    warn = reset()
    assert fused_rs_epilogue_op._resolve_tp_mesh_and_order() is None
    assert "0..7" in (fused_rs_epilogue_op._REFUSED_REASON or "")
    warn.assert_called_once()

    gather_calls = 0
    order = (0, 1, 6, 7, 2, 3, 4, 5)
    mesh = fake_mesh(list(range(8)))
    monkeypatch.setattr(fused_rs_epilogue_op, "build_tp_mesh", lambda: (mesh, order))
    warn = reset()
    resolved = fused_rs_epilogue_op._resolve_tp_mesh_and_order()
    assert resolved == (mesh, order, tuple(range(8)))
    assert fused_rs_epilogue_op._resolve_tp_mesh_and_order() == resolved
    assert gather_calls == 1
    warn.assert_not_called()


def test_hier_rs_requires_dense_sequence_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "TPU_DENSE_SEQUENCE_PARALLEL", False)
    monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "bf16")
    with pytest.raises(ValueError, match="requires TPU_DENSE_SEQUENCE_PARALLEL=1"):
        gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(None)


def _fused_collectives_stand_in(
    reduce_scatter: Callable, eps: float
) -> fused_rs_epilogue_op.FusedSublayerCollectives:
    def norm_add_scale(partial, residual, post_norm_weight, scale):
        reduced = _rms_norm_torch(reduce_scatter(partial, 0), post_norm_weight, eps)
        return (reduced + residual) * scale

    def norm_add_scale_next_norm(
        partial, residual, post_norm_weight, scale, next_norm_weight
    ):
        residual = norm_add_scale(partial, residual, post_norm_weight, scale)
        normed = _rms_norm_torch(residual, next_norm_weight, eps)
        row_scale = normed.abs().amax(-1, keepdim=True)
        return residual, normed / row_scale, row_scale

    return fused_rs_epilogue_op.FusedSublayerCollectives(
        norm_add_next_norm=lambda partial, residual, post_norm_w, next_norm_w: (
            norm_add_scale_next_norm(partial, residual, post_norm_w, 1.0, next_norm_w)
        ),
        norm_add_scale_next_norm=norm_add_scale_next_norm,
        norm_add_scale=norm_add_scale,
    )


def _oneshot_collectives_stand_in(
    exchange: Callable[[torch.Tensor], list[torch.Tensor]],
    rank_fn: Callable[[], int],
    tp_size: int,
    eps: float,
    counter: list[int],
) -> fused_rs_epilogue_op.FusedSublayerCollectives:
    def rs_local(partial: torch.Tensor) -> torch.Tensor:
        if rank_fn() == 0:
            counter[0] += 1
        total = torch.stack(exchange(partial)).sum(0)
        return total.chunk(tp_size, dim=0)[rank_fn()]

    def gather_local(x: torch.Tensor) -> torch.Tensor:
        return torch.cat(exchange(x), dim=0)

    def norm_add_scale(partial, residual, post_norm_weight, scale):
        reduced = _rms_norm_torch(rs_local(partial), post_norm_weight, eps)
        return gather_local((reduced + residual) * scale)

    def norm_add_scale_next_norm(
        partial, residual, post_norm_weight, scale, next_norm_weight
    ):
        reduced = _rms_norm_torch(rs_local(partial), post_norm_weight, eps)
        res_new = (reduced + residual) * scale
        normed = _rms_norm_torch(res_new, next_norm_weight, eps)
        row_scale = normed.abs().amax(-1, keepdim=True)
        return res_new, gather_local(normed / row_scale), gather_local(row_scale)

    return fused_rs_epilogue_op.FusedSublayerCollectives(
        norm_add_next_norm=lambda partial, residual, post_norm_w, next_norm_w: (
            norm_add_scale_next_norm(partial, residual, post_norm_w, 1.0, next_norm_w)
        ),
        norm_add_scale_next_norm=norm_add_scale_next_norm,
        norm_add_scale=norm_add_scale,
        outputs_gathered=True,
    )


@pytest.mark.parametrize("path", ["replicated", "xla_sp", "hier_sp", "hier_oneshot"])
@pytest.mark.parametrize("tp", [2, 8])
def test_forward_matches_upstream(
    monkeypatch: pytest.MonkeyPatch, tp: int, path: str
) -> None:
    """Multi-rank layer stack equals upstream forward at TP=1 on every rank."""
    monkeypatch.setattr(
        dense_sequence_parallel,
        "MIN_TOKENS_PER_RANK",
        NUM_TOKENS + 1 if path in ("replicated", "hier_oneshot") else 0,
    )
    _patch_fp8_ops_with_float_stand_ins(monkeypatch)
    mock = TPCollectiveMock(tp)
    _patch_collectives(
        monkeypatch,
        all_reduce=mock.all_reduce,
        all_gather=mock.all_gather,
        reduce_scatter=mock.reduce_scatter,
    )
    oneshot_calls = [0]

    def silent_exchange(tensor: torch.Tensor) -> list[torch.Tensor]:
        mock.storage[mock.rank.value] = tensor
        mock.barrier.wait(timeout=10.0)
        parts = list(mock.storage)
        mock.barrier.wait(timeout=10.0)
        return parts

    if path == "hier_oneshot":
        monkeypatch.setattr(fused_rs_epilogue_op, "MIN_ONESHOT_RS_AG_TOKENS", 0)
    if path in ("hier_sp", "hier_oneshot"):
        monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "bf16")
        monkeypatch.setattr(
            fused_rs_epilogue_op,
            "build_streaming_sublayer_collectives",
            lambda eps, *, fp8_wire: _fused_collectives_stand_in(
                mock.reduce_scatter, eps
            ),
        )
        monkeypatch.setattr(
            fused_rs_epilogue_op,
            "build_oneshot_sublayer_collectives",
            lambda eps: _oneshot_collectives_stand_in(
                silent_exchange, lambda: mock.rank.value, tp, eps, oneshot_calls
            ),
        )
        monkeypatch.setattr(
            fused_rs_epilogue_op, "supports_streaming_sublayer", lambda *_: True
        )
        monkeypatch.setattr(
            fused_rs_epilogue_op, "supports_oneshot_sublayer", lambda *_: True
        )

    weights = [_make_weights(seed) for seed in range(NUM_LAYERS)]
    g = torch.Generator().manual_seed(123)
    embedding = torch.randn(NUM_TOKENS, HIDDEN, generator=g, dtype=torch.float64)
    positions = torch.arange(NUM_TOKENS)

    expected = embedding
    for idx, layer_weights in enumerate(weights):
        expected, _ = _UPSTREAM_FORWARD(
            _make_layer(layer_weights, idx), positions, expected, None
        )

    outputs: list[torch.Tensor | None] = [None] * tp
    rows_after_layer_0: list[int | None] = [None] * tp
    errors: list[Exception | None] = [None] * tp

    def worker(rank: int) -> None:
        try:
            mock.rank.value = rank
            hidden, residual = embedding.clone(), None
            for idx, layer_weights in enumerate(weights):
                layer = _make_layer(layer_weights, idx, tp, rank)
                gemma4_patch._configure_layer(layer)
                hidden, residual = gemma4_patch._dense_sequence_parallel_forward(
                    layer, positions, hidden, residual
                )
                assert (residual is not None) == (
                    path in ("hier_sp", "hier_oneshot") and idx < NUM_LAYERS - 1
                )
                if idx == 0:
                    rows_after_layer_0[rank] = hidden.shape[0]
            outputs[rank] = hidden
        except Exception as e:
            errors[rank] = e
            mock.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()
    if any(errors):
        raise next(e for e in errors if e is not None)

    for out in outputs:
        assert out is not None
        torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)
    if path == "hier_oneshot":
        assert rows_after_layer_0 == [NUM_TOKENS] * tp
        assert oneshot_calls == [2 * NUM_LAYERS]
        assert mock.calls == {
            "all_reduce": 0,
            "reduce_scatter": 0,
            "all_gather": 0,
        }
    elif path != "replicated":
        rows = NUM_TOKENS if path == "hier_sp" else NUM_TOKENS // tp
        assert rows_after_layer_0 == [rows] * tp
        assert mock.calls == {
            "all_reduce": 0,
            "reduce_scatter": 2 * NUM_LAYERS,
            "all_gather": 4 * NUM_LAYERS + 1,
        }
    else:
        assert rows_after_layer_0 == [NUM_TOKENS] * tp
        assert mock.calls == {
            "all_reduce": 2 * NUM_LAYERS,
            "reduce_scatter": 0,
            "all_gather": 0,
        }


def test_use_sequence_parallel_needs_warmup_and_aligned_per_rank_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buckets below per-rank threshold, unaligned to TP, or outside warmup keep AR."""
    tp = 4
    monkeypatch.setattr(dense_sequence_parallel, "MIN_TOKENS_PER_RANK", 8)
    _patch_fp8_ops_with_float_stand_ins(monkeypatch)
    calls: list[str] = []
    _patch_collectives(
        monkeypatch,
        all_reduce=lambda x: (calls.append("ar"), x)[1],
        all_gather=lambda x, dim=0: (calls.append("ag"), torch.cat([x] * tp))[1],
        reduce_scatter=lambda x, dim=0: (calls.append("rs"), x[: x.shape[0] // tp])[1],
    )
    layer = _make_layer(_make_weights(0), 0, tp)
    gemma4_patch._configure_layer(layer)

    def collectives(num_tokens: int) -> list[str]:
        calls.clear()
        x = torch.ones((num_tokens, HIDDEN), dtype=torch.float64)
        gemma4_patch._dense_sequence_parallel_forward(
            layer, torch.arange(num_tokens), x, None
        )
        return list(calls)

    replicated = ["ar", "ar"]
    sharded = ["ag", "ag", "rs", "ag", "ag", "rs"]
    assert collectives(16) == replicated
    assert collectives(32) == sharded
    assert collectives(34) == replicated
    assert collectives(64) == sharded

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert collectives(64) == replicated
    with shape_variants.warmup():
        assert collectives(64) == sharded


def test_token_buckets_share_one_graph_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-reduce and sequence-parallel buckets lift identical Dynamo graph inputs."""
    tp = 4
    monkeypatch.setattr(dense_sequence_parallel, "MIN_TOKENS_PER_RANK", 8)
    fp8_linear_op.register_fp8_linear_ops()
    _patch_collectives(
        monkeypatch,
        all_reduce=lambda x: x,
        all_gather=lambda x, dim=0: x.repeat(tp, 1),
        reduce_scatter=lambda x, dim=0: x.narrow(0, 0, x.shape[0] // tp),
    )

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(128, HIDDEN, dtype=torch.bfloat16)
            self.layers = nn.ModuleList(
                _as_fp8_weights(
                    _make_layer(
                        _make_weights(idx, torch.bfloat16), idx, tp, num_layers=2
                    )
                )
                for idx in range(2)
            )
            for layer in self.layers:
                gemma4_patch._configure_layer(layer)

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            hidden, residual = self.embed(input_ids), None
            for layer in self.layers:
                hidden, residual = gemma4_patch._dense_sequence_parallel_forward(
                    layer, positions, hidden, residual
                )
            return hidden

    signatures = []

    def capture_backend(gm: torch.fx.GraphModule, _example_inputs: object):
        signatures.append(shape_variants.graph_signature(gm))
        return lambda *_args: torch.zeros(1, HIDDEN, dtype=torch.bfloat16)

    model = _Stack()
    with shape_variants.warmup():
        for num_tokens in (16, 64, 66):
            torch._dynamo.reset()
            compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
            input_ids = torch.zeros(num_tokens, dtype=torch.int32)
            positions = torch.arange(num_tokens, dtype=torch.int32)
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(positions, 0)
            compiled(input_ids, positions)

    assert len(signatures) == 3
    assert signatures[0] == signatures[1] == signatures[2]


def test_hier_collectives_trace_fullgraph_and_match_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming (`1024`) and single-shot (`128`) ops trace under fullgraph=True
    without calling StrongLRUCache (`pltpu.get_tpu_info`) and match graph_signature."""
    tp = 8
    wide_hidden = 256
    monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "bf16")
    fp8_linear_op.register_fp8_linear_ops()
    _patch_collectives(
        monkeypatch,
        all_reduce=lambda x: x,
        all_gather=lambda x, dim=0: x.repeat(tp, 1),
        reduce_scatter=lambda x, dim=0: x.narrow(0, 0, x.shape[0] // tp),
    )
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]), (fused_rs_epilogue_op.TP_AXIS_NAME,)
    )
    monkeypatch.setattr(
        fused_rs_epilogue_op,
        "_resolve_tp_mesh_and_order",
        lambda: (mesh, tuple(range(tp)), tuple(range(tp))),
    )
    monkeypatch.setattr(fused_rs_epilogue_op, "_STREAM_COLLECTIVES_CACHE", {})
    monkeypatch.setattr(fused_rs_epilogue_op, "_ONESHOT_COLLECTIVES_CACHE", {})
    monkeypatch.setattr(
        fused_rs_epilogue_op.kernel_stream, "_EPILOGUE_HIDDEN_SUPPORT", {}
    )
    monkeypatch.setattr(
        fused_rs_epilogue_op.kernel_oneshot_rs_ag, "_ONESHOT_HIDDEN_SUPPORT", {}
    )

    monkeypatch.setattr(sys.modules[__name__], "HIDDEN", wide_hidden)

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(128, wide_hidden, dtype=torch.bfloat16)
            self.layers = nn.ModuleList(
                _as_fp8_weights(
                    _make_layer(
                        _make_weights(idx, torch.bfloat16), idx, tp, num_layers=2
                    )
                )
                for idx in range(2)
            )
            for layer in self.layers:
                gemma4_patch._configure_layer(layer)

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            hidden, residual = self.embed(input_ids), None
            for layer in self.layers:
                hidden, residual = gemma4_patch._dense_sequence_parallel_forward(
                    layer, positions, hidden, residual
                )
            return hidden

    model = _Stack()
    state0 = gemma4_patch._layer_state(model.layers[0])
    assert state0.streaming_collectives is not None
    assert state0.oneshot_collectives is not None

    def forbid_tpu_info() -> None:
        raise AssertionError("tpu_info must not run during TorchDynamo tracing")

    monkeypatch.setattr(fused_rs_epilogue_op.kernel_stream, "tpu_info", forbid_tpu_info)
    monkeypatch.setattr(
        fused_rs_epilogue_op.kernel_oneshot_rs_ag, "tpu_info", forbid_tpu_info
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", forbid_tpu_info)

    signatures = []
    seen_ops: list[set[str]] = []

    def capture_backend(gm: torch.fx.GraphModule, _example_inputs: object):
        signatures.append(shape_variants.graph_signature(gm))
        seen_ops.append(
            {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
        )
        return lambda *_args: torch.zeros(1, wide_hidden, dtype=torch.bfloat16)

    with shape_variants.warmup():
        for num_tokens in (16, 128, 1024, 1032):
            torch._dynamo.reset()
            compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
            input_ids = torch.zeros(num_tokens, dtype=torch.int32)
            positions = torch.arange(num_tokens, dtype=torch.int32)
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(positions, 0)
            compiled(input_ids, positions)

    assert len(signatures) == 4
    assert signatures[0] == signatures[1] == signatures[2] == signatures[3]
    assert not any("dense_sp_" in op for op in seen_ops[0])
    assert any("dense_sp_oneshot_" in op for op in seen_ops[1])
    assert any("dense_sp_rs_" in op for op in seen_ops[2])
    assert not any("dense_sp_" in op for op in seen_ops[3])


def test_hier_rs_tp4_arms_streaming_without_oneshot_or_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At TP=4, streaming collectives arm while 8-device oneshot stays None quietly."""
    wide_hidden = 256
    monkeypatch.setattr(sys.modules[__name__], "HIDDEN", wide_hidden)
    monkeypatch.setattr(envs, "TPU_DENSE_SP_HIER_RS", "fp8")
    warn = Mock()
    monkeypatch.setattr(gemma4_patch.logger, "warning_once", warn)
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]), (fused_rs_epilogue_op.TP_AXIS_NAME,)
    )
    monkeypatch.setattr(
        fused_rs_epilogue_op,
        "_resolve_tp_mesh_and_order",
        lambda: (mesh, (0, 1, 2, 3), (0, 1, 2, 3)),
    )
    monkeypatch.setattr(fused_rs_epilogue_op, "_STREAM_COLLECTIVES_CACHE", {})
    monkeypatch.setattr(fused_rs_epilogue_op, "_ONESHOT_COLLECTIVES_CACHE", {})
    layer = _make_layer(_make_weights(0), 0, tp=4)
    gemma4_patch._configure_layer(layer)
    state = gemma4_patch._layer_state(layer)
    assert state.streaming_collectives is not None
    assert state.oneshot_collectives is None
    warn.assert_not_called()
