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
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4MLP,
)

from tests.tp_collective_test_utils import TPCollectiveMock
from vllm_torchtpu.compilation import graph_fingerprint, shape_variants
from vllm_torchtpu.layers.adapter import dense_sequence_parallel
from vllm_torchtpu.layers.adapter.custom_ops import fp8_linear_op
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import OnlineFp8Policy
from vllm_torchtpu.models.vllm import gemma4_patch

_UPSTREAM_FORWARD = Gemma4DecoderLayer.forward

HIDDEN = 24
HEADS = 8
HEAD_DIM = 4
INTERMEDIATE = 32
NUM_LAYERS = 3
NUM_TOKENS = 16


def _row_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x


class _TestFp8Method(VllmFp8LinearMethodTPU):
    """CPU-runnable `VllmFp8LinearMethodTPU` stand-in implementing `apply()`."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = ((x @ layer.weight.to(x.dtype)) * layer.weight_scale.to(x.dtype)).to(
            x.dtype
        )
        return out + bias if bias is not None else out


def _fp8_method(
    *,
    w8a16_max_rows: int = 0,
    weight_block_size: tuple[int, int] | None = None,
    activation_scheme: str = "dynamic",
    requant_weight_dtype: tuple[str, torch.dtype] = (
        "float8_e4m3fn",
        torch.float8_e4m3fn,
    ),
    requant_block_size: int | None = None,
    online_fp8_scheme: str | None = None,
) -> VllmFp8LinearMethodTPU:
    method = _TestFp8Method.__new__(_TestFp8Method)
    method.w8a16_max_rows = w8a16_max_rows
    method.weight_block_size = weight_block_size
    method.block_quant = weight_block_size is not None
    method.act_q_static = activation_scheme == "static"
    method.quant_config = SimpleNamespace(activation_scheme=activation_scheme)
    method.online_fp8 = online_fp8_scheme is not None
    method.online_fp8_policy = (
        OnlineFp8Policy(
            targets=["*"],
            exclude=[],
            weight_scheme=online_fp8_scheme,
            weight_block_size=(128, 128) if online_fp8_scheme == "per_block" else None,
        )
        if online_fp8_scheme is not None
        else None
    )
    method._linear_quant_config = (
        requant_weight_dtype[0],
        requant_weight_dtype[1],
        requant_block_size,
        False,
    )
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
        self.input_is_parallel = True
        self.reduce_results = True
        setattr(self, WEIGHT_FLIPPED_ATTR, True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.quant_method.apply(self, x, None)
        if self.reduce_results and self.tp_size > 1:
            out = _row_all_reduce(out)
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
            {
                "quant_method": _fp8_method(
                    requant_weight_dtype=("int8", torch.int8),
                )
            },
            "REQUANTIZE_WEIGHT_DTYPE='int8'",
        ),
        (
            {"quant_method": _fp8_method(requant_block_size=128)},
            "requant_block_size=128",
        ),
        (
            {"quant_method": _fp8_method(online_fp8_scheme="per_block")},
            "weight_scheme='per_block'",
        ),
        ({KEEP_VLLM_LAYOUT_ATTR: True}, "canonical \\(k, n\\)"),
        ({"bias": torch.zeros(32)}, "bias shape \\(32,\\)"),
    ],
    ids=[
        "other_method",
        "w8a16",
        "requant_int8",
        "requant_blockwise",
        "online_fp8_per_block",
        "keep_vllm_layout",
        "bias",
    ],
)
def test_validate_prequantized_fp8_linear_init_checks(
    overrides: dict[str, object], match: str
) -> None:
    """Init-time admission checks key on runtime layout/requant settings."""
    # Blockwise checkpoint requantized per-channel, static-activation checkpoint,
    # and per-channel / per-tensor online FP8 are all admitted at init.
    for admitted in (
        _fp8_method(),
        _fp8_method(weight_block_size=(128, 128)),
        _fp8_method(activation_scheme="static"),
        _fp8_method(online_fp8_scheme="per_channel"),
        _fp8_method(online_fp8_scheme="per_tensor"),
    ):
        dense_sequence_parallel.validate_fp8_linear_for_sp(
            _fp8_linear(quant_method=admitted)
        )

    bad = _fp8_linear(**overrides)
    with pytest.raises(NotImplementedError, match=match):
        dense_sequence_parallel.validate_fp8_linear_for_sp(bad)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"quant_method": SimpleNamespace()}, "SimpleNamespace"),
        ({WEIGHT_FLIPPED_ATTR: False}, f"{WEIGHT_FLIPPED_ATTR}=False"),
        ({"weight": torch.zeros(16, 32, dtype=torch.int8)}, "torch.int8"),
        ({"weight_scale": torch.ones(1, 2, 1, 32)}, "shape \\(1, 2, 1, 32\\)"),
    ],
    ids=["other_method", "unflipped", "int8", "blockwise_scale"],
)
def test_apply_prequantized_fp8_linear_runtime_checks(
    overrides: dict[str, object], match: str
) -> None:
    """Post-load tensor invariants are enforced before calling the custom op."""
    fp8_linear_op.register_fp8_linear_ops()
    x_q = torch.empty((4, 16), dtype=torch.uint8, device="meta")
    x_scale = torch.empty((4, 1), dtype=torch.float32, device="meta")

    valid = _Linear(torch.empty((16, 32), dtype=torch.float8_e4m3fn, device="meta"))
    valid.weight_scale = torch.empty((32,), dtype=torch.float32, device="meta")
    assert dense_sequence_parallel.apply_prequantized_fp8_linear(
        valid, x_q, x_scale
    ).shape == (4, 32)

    bad = _fp8_linear(**overrides)
    with pytest.raises(NotImplementedError, match=match):
        dense_sequence_parallel.apply_prequantized_fp8_linear(bad, x_q, x_scale)


def test_maybe_patch_waits_for_gemma4_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch leaves Gemma4DecoderLayer untouched when off and waits for Gemma 4."""
    orig_init = Gemma4DecoderLayer.__init__
    orig_forward = Gemma4DecoderLayer.forward
    monkeypatch.setattr(Gemma4DecoderLayer, "__init__", orig_init)
    monkeypatch.setattr(Gemma4DecoderLayer, "forward", orig_forward)

    vllm_off = SimpleNamespace(additional_config={})
    vllm_on = SimpleNamespace(additional_config={"tpu_dense_sequence_parallel": True})
    cfg_qwen = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text")
    )
    cfg_gemma4 = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4_text")
    )

    # Off by default: returns False and leaves Gemma4DecoderLayer untouched.
    assert (
        gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(cfg_gemma4, vllm_off)
        is False
    )
    assert Gemma4DecoderLayer.__init__ is orig_init
    assert Gemma4DecoderLayer.forward is orig_forward

    # Enabled, missing or non-Gemma 4 config: returns False so registry retries later.
    assert (
        gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(None, vllm_on) is False
    )
    assert (
        gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(cfg_qwen, vllm_on)
        is False
    )
    assert Gemma4DecoderLayer.__init__ is orig_init

    # Enabled with Gemma 4 config: installs the patch and completes registration.
    assert (
        gemma4_patch.maybe_patch_gemma4_dense_sequence_parallel(cfg_gemma4, vllm_on)
        is None
    )
    assert Gemma4DecoderLayer.__init__ is gemma4_patch._patched_layer_init
    assert Gemma4DecoderLayer.forward is gemma4_patch._dense_sequence_parallel_forward


class _TestRMSNorm(RMSNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.variance_epsilon = eps
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            * self.weight
        )


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


def test_patched_layer_init_matches_upstream_signature_and_configures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_patched_layer_init` matches upstream signature and configures the layer."""
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
        built = _make_layer(_make_weights(0), 0, tp=2, rank=1)
        self.__dict__.update(built.__dict__)

    monkeypatch.setattr(gemma4_patch, "_orig_layer_init", fake_orig_init)
    registered = Mock()
    monkeypatch.setattr(fp8_linear_op, "register_fp8_linear_ops", registered)
    # Distinct from attn.config.num_hidden_layers (= NUM_LAYERS = 3) so
    # `_is_last_layer(layer)` proves `_configure_layer` used `cfg`.
    cfg = SimpleNamespace(num_hidden_layers=1)

    layer = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    gemma4_patch._patched_layer_init(
        layer,
        cfg,
        cache_config="cache",
        quant_config="quant",
        prefix="model.layers.0",
    )
    assert captured == {
        "config": cfg,
        "cache_config": "cache",
        "quant_config": "quant",
        "prefix": "model.layers.0",
    }
    registered.assert_called_once()
    assert layer.self_attn.o_proj.reduce_results is True
    assert layer.mlp.down_proj.reduce_results is True
    assert torch.equal(layer._tpu_sp_tp_rank, torch.tensor([1], dtype=torch.int32))
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
        ({"o_proj_not_parallel": True}, NotImplementedError, "input_is_parallel"),
        ({"qkv_non_fp8": True}, NotImplementedError, "qkv_proj.*SimpleNamespace"),
        (
            {"gate_up_w8a16": True},
            NotImplementedError,
            "gate_up_proj.*w8a16_max_rows=0, got 64",
        ),
        ({"gate_up_bias": True}, NotImplementedError, "gate_up_proj.*bias=None"),
        ({"bad_norm_type": True}, NotImplementedError, "RMSNorm"),
    ],
)
def test_configure_layer(
    overrides: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Layer configuration keeps reduce_results intact and rejects bad configs."""
    layer = _make_layer(_make_weights(0), NUM_LAYERS - 1, tp=2)
    gemma4_patch._configure_layer(layer)
    assert layer.self_attn.o_proj.reduce_results is True
    assert layer.mlp.down_proj.reduce_results is True
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
        elif name == "o_proj_not_parallel":
            layer.self_attn.o_proj.input_is_parallel = False
        elif name == "qkv_non_fp8":
            layer.self_attn.qkv_proj.quant_method = SimpleNamespace()
        elif name == "gate_up_w8a16":
            layer.mlp.gate_up_proj.quant_method = _fp8_method(w8a16_max_rows=64)
        elif name == "gate_up_bias":
            layer.mlp.gate_up_proj.bias = nn.Parameter(torch.zeros(2 * INTERMEDIATE))
        elif name == "bad_norm_type":
            layer.post_attention_layernorm = nn.Identity()  # type: ignore[assignment]
        else:
            setattr(layer, name, value)
    with pytest.raises(error, match=match):
        gemma4_patch._configure_layer(layer)


def _patch_collectives(monkeypatch: pytest.MonkeyPatch, **fakes: object) -> None:
    import sys

    for name, fake in fakes.items():
        if name == "all_reduce":
            monkeypatch.setattr(sys.modules[__name__], "_row_all_reduce", fake)
        elif name == "all_gather":
            monkeypatch.setattr(
                dense_sequence_parallel, "tensor_model_parallel_all_gather", fake
            )
        else:
            monkeypatch.setattr(gemma4_patch, f"tensor_model_parallel_{name}", fake)


def _patch_fp8_ops_with_float_stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = x.abs().amax(-1, keepdim=True)
        return x / scale, scale

    monkeypatch.setattr(fp8_linear_op, "register_fp8_linear_ops", lambda: None)
    monkeypatch.setattr(fp8_linear_op, "quantize_fp8_activations", quantize_activations)
    monkeypatch.setattr(
        dense_sequence_parallel,
        "apply_prequantized_fp8_linear",
        lambda layer, x_q, x_scale: layer.quant_method.apply(layer, x_q * x_scale),
    )


@pytest.mark.parametrize("sharded", [True, False])
@pytest.mark.parametrize("tp", [2, 8])
def test_forward_matches_upstream(
    monkeypatch: pytest.MonkeyPatch, tp: int, sharded: bool
) -> None:
    """Multi-rank layer stack equals upstream forward at TP=1 on every rank."""
    monkeypatch.setattr(
        dense_sequence_parallel,
        "MIN_TOKENS_PER_RANK",
        0 if sharded else NUM_TOKENS + 1,
    )
    _patch_fp8_ops_with_float_stand_ins(monkeypatch)
    mock = TPCollectiveMock(tp)
    _patch_collectives(
        monkeypatch,
        all_reduce=mock.all_reduce,
        all_gather=mock.all_gather,
        reduce_scatter=mock.reduce_scatter,
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
                assert residual is None
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
    if sharded:
        assert rows_after_layer_0 == [NUM_TOKENS // tp] * tp
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


@pytest.mark.parametrize("rank", [0, 3])
def test_token_buckets_share_one_graph_signature_and_rank_uniform_fingerprint(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    """All-reduce and SP buckets lift one signature and rank-uniform fingerprints."""
    tp = 4
    monkeypatch.setattr(dense_sequence_parallel, "MIN_TOKENS_PER_RANK", 8)
    fp8_linear_op.register_fp8_linear_ops()
    _patch_collectives(
        monkeypatch,
        all_reduce=lambda x: x,
        all_gather=lambda x, dim=0: x.repeat(tp, 1),
        reduce_scatter=lambda x, dim=0: x.narrow(0, 0, x.shape[0] // tp),
    )

    def make_bf16_fp8_layer(idx: int, layer_rank: int) -> Gemma4DecoderLayer:
        layer = _make_layer(
            _make_weights(idx, torch.bfloat16), idx, tp, layer_rank, num_layers=2
        )
        for proj in (layer.self_attn.qkv_proj, layer.mlp.gate_up_proj):
            proj.weight = nn.Parameter(
                proj.weight.to(torch.float8_e4m3fn), requires_grad=False
            )
            proj.weight_scale = proj.weight_scale.to(torch.float32)
        return layer

    class _Stack(nn.Module):
        def __init__(self, stack_rank: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(128, HIDDEN, dtype=torch.bfloat16)
            self.layers = nn.ModuleList(
                make_bf16_fp8_layer(idx, stack_rank) for idx in range(2)
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

    def trace_buckets(
        stack_rank: int,
    ) -> tuple[list[tuple[object, ...]], list[str | None]]:
        signatures: list[tuple[object, ...]] = []
        fingerprints: list[str | None] = []

        def capture_backend(gm: torch.fx.GraphModule, _example_inputs: object):
            signatures.append(shape_variants.graph_signature(gm))
            fingerprints.append(graph_fingerprint._graph_fingerprint(gm))
            return lambda *_args: torch.zeros(1, HIDDEN, dtype=torch.bfloat16)

        model = _Stack(stack_rank)
        with shape_variants.warmup():
            for num_tokens in (16, 64, 66):
                torch._dynamo.reset()
                compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
                input_ids = torch.zeros(num_tokens, dtype=torch.int32)
                positions = torch.arange(num_tokens, dtype=torch.int32)
                torch._dynamo.mark_dynamic(input_ids, 0)
                torch._dynamo.mark_dynamic(positions, 0)
                compiled(input_ids, positions)
        return signatures, fingerprints

    signatures, fingerprints = trace_buckets(rank)
    assert len(signatures) == 3
    assert signatures[0] == signatures[1] == signatures[2]
    assert all(fp is not None for fp in fingerprints)

    if rank != 0:
        rank0_sigs, rank0_fps = trace_buckets(0)
        assert signatures == rank0_sigs
        assert fingerprints == rank0_fps
