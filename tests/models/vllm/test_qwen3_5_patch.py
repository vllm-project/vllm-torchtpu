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
"""Unit tests for Qwen3.5 attention-to-MoE collective overlap."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn
from torch._dynamo import mark_dynamic
from torch._dynamo import reset as reset_dynamo
from vllm.compilation.backends import split_graph
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.models.vllm import qwen3_5_patch
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _configure_layer as configure_layer,
)
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _orig_init as orig_decoder_init,
)
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _overlap_forward as overlap_forward,
)
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _overlap_min_tokens as overlap_min_tokens,
)
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _overlap_model_forward as overlap_model_forward,
)
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _patched_init as patched_decoder_init,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from vllm.config import ModelConfig

_EXPECTED_BF16_ALL_REDUCE_COUNT = 2
_FAKE_QMM_HOLDER: list[Any] = []


class _RankThreadState(threading.local):
    """Thread-local rank holder for simulated TP workers."""

    def __init__(self) -> None:
        """Initialize default rank to 0."""
        super().__init__()
        self.rank: int = 0


_THREAD_STATE = _RankThreadState()


@dataclass
class _StubHfTextConfig:
    """Stub HuggingFace text config carrying model_type."""

    model_type: str


@dataclass
class _StubModelConfig:
    """Stub vLLM ModelConfig carrying hf_text_config."""

    hf_text_config: _StubHfTextConfig


@dataclass
class _StubProj:
    """Stub projection module carrying reduce_results flag."""

    reduce_results: bool = True


@dataclass
class _StubMoeConfig:
    """Stub MoE config carrying skip_final_all_reduce flag."""

    skip_final_all_reduce: bool = False


@dataclass
class _StubExperts:
    """Stub MoE experts module carrying gate and moe_config."""

    gate: object = None
    moe_config: _StubMoeConfig = field(default_factory=_StubMoeConfig)


@dataclass
class _StubSelfAttn:
    """Stub self-attention module carrying o_proj."""

    o_proj: _StubProj = field(default_factory=_StubProj)


@dataclass
class _StubLinearAttn:
    """Stub linear-attention module carrying out_proj."""

    out_proj: _StubProj = field(default_factory=_StubProj)


@dataclass
class _StubMlp:
    """Stub MoE MLP block used by _configure_layer and _overlap_forward tests."""

    tp_size: int = 2
    replicate_shared_expert: bool = False
    experts: Any = field(default_factory=_StubExperts)
    gate: Any = None


@dataclass
class _StubPpGroup:
    """Stub pipeline-parallel group carrying world_size."""

    world_size: int = 1


@dataclass
class _StubDecoderLayer:
    """Stub Qwen3.5 decoder layer for unit testing overlap configuration."""

    layer_type: str = "full_attention"
    use_attn_reduce_scatter_for_moe: bool = False
    sharded_spine: bool = True
    layer_scale: bool = False
    input_layernorm: Any = None
    post_attention_layernorm: Any = None
    self_attn: Any = None
    linear_attn: Any = None
    attn_layer_scale: torch.Tensor | None = None
    ffn_layer_scale: torch.Tensor | None = None
    mlp: _StubMlp = field(default_factory=_StubMlp)


@dataclass
class _StubModel:
    """Stub Qwen3.5 model container carrying start_layer and decoder layers."""

    start_layer: int
    layers: list[_StubDecoderLayer]


@dataclass(frozen=True)
class _WireCaptureConfig:
    """Configuration options for _run_single_layer_with_wire_capture."""

    tp_size: int = 2
    d: int = 64
    with_residual: bool = False
    layer_type: str = "linear_attention"
    layer_scale: bool = False


_DEFAULT_WIRE_CFG = _WireCaptureConfig()


@pytest.fixture
def _reset_dynamo_state() -> Iterator[None]:
    """Reset TorchDynamo state before and after compilation trace handover tests."""
    reset_dynamo()
    yield
    reset_dynamo()


class _CpuRMSNorm(Qwen3_5RMSNorm):
    """Qwen3_5RMSNorm using PyTorch-native implementation for CPU tests."""

    def dispatch_forward(
        self,
        *,
        compile_native: bool = False,
    ) -> Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Return native CPU forward implementation."""
        _ = compile_native
        return self.forward_native


class _TPCollectiveMock:
    """Barrier-synchronized in-process collective simulator for TP ranks."""

    def __init__(self, tp_size: int) -> None:
        """Initialize barrier and per-rank tensor slots."""
        self.tp_size = tp_size
        self.barrier = threading.Barrier(tp_size)
        self.storage: list[torch.Tensor | None] = [None] * tp_size

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Simulate sum reduce-scatter across TP threads."""
        if self.tp_size == 1:
            return tensor
        rank = _THREAD_STATE.rank
        self.storage[rank] = tensor
        self.barrier.wait(timeout=10.0)
        tensors = [t for t in self.storage if t is not None]
        total = torch.stack(tensors, dim=0).sum(dim=0)
        assert total is not None
        chunk = total.chunk(self.tp_size, dim=dim)[rank].clone()
        self.barrier.wait(timeout=10.0)
        return chunk

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Simulate all-gather across TP threads."""
        if self.tp_size == 1:
            return tensor
        rank = _THREAD_STATE.rank
        self.storage[rank] = tensor
        self.barrier.wait(timeout=10.0)
        tensors = [t for t in self.storage if t is not None]
        res = torch.cat(tensors, dim=dim).clone()
        self.barrier.wait(timeout=10.0)
        return res

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate sum all-reduce across TP threads."""
        if self.tp_size == 1:
            return tensor
        rank = _THREAD_STATE.rank
        self.storage[rank] = tensor
        self.barrier.wait(timeout=10.0)
        tensors = [t for t in self.storage if t is not None]
        total = torch.stack(tensors, dim=0).sum(dim=0)
        assert total is not None
        res = total.clone()
        self.barrier.wait(timeout=10.0)
        return res


def _as_model_config(stub: _StubModelConfig) -> ModelConfig:
    """Cast a _StubModelConfig to ModelConfig for patch helpers."""
    return cast("ModelConfig", stub)


def _as_decoder_layer(stub: _StubDecoderLayer) -> Qwen3_5DecoderLayer:
    """Cast a _StubDecoderLayer to Qwen3_5DecoderLayer for patch helpers."""
    return cast("Qwen3_5DecoderLayer", stub)


def _as_model(stub: _StubModel) -> Qwen3_5Model:
    """Cast a _StubModel to Qwen3_5Model for model forward patch helpers."""
    return cast("Qwen3_5Model", stub)


def test_maybe_patch_waits_for_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies patch registration returns and class method patching."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", False)
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(None) is None

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(None) is False

    cfg_dense = _StubModelConfig(
        hf_text_config=_StubHfTextConfig(model_type="qwen3_5_text")
    )
    assert (
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(_as_model_config(cfg_dense))
        is False
    )

    cfg_moe = _StubModelConfig(
        hf_text_config=_StubHfTextConfig(model_type="qwen3_5_moe_text")
    )
    orig_init = Qwen3_5DecoderLayer.__init__
    had_fwd = "forward" in Qwen3_5DecoderLayer.__dict__
    orig_fwd = Qwen3_5DecoderLayer.forward
    had_model_fwd = "forward" in Qwen3_5Model.__dict__
    orig_model_fwd = Qwen3_5Model.forward
    try:
        assert (
            qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(
                _as_model_config(cfg_moe)
            )
            is None
        )
        assert Qwen3_5DecoderLayer.__init__ is patched_decoder_init
        assert Qwen3_5DecoderLayer.forward is overlap_forward
        assert Qwen3_5Model.forward is overlap_model_forward
    finally:
        Qwen3_5DecoderLayer.__init__ = orig_init
        if had_fwd:
            Qwen3_5DecoderLayer.forward = orig_fwd
        else:
            del Qwen3_5DecoderLayer.forward
        if had_model_fwd:
            Qwen3_5Model.forward = orig_model_fwd
        else:
            del Qwen3_5Model.forward


def test_maybe_patch_ignores_fp8_flag_when_overlap_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlap disabled leaves the model unpatched and logs a warning once."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", False)
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    warned: list[str] = []
    monkeypatch.setattr(
        qwen3_5_patch.logger,
        "warning_once",
        lambda msg, *args: warned.append(msg % args if args else msg),
    )
    cfg_moe = _StubModelConfig(
        hf_text_config=_StubHfTextConfig(model_type="qwen3_5_moe_text")
    )
    orig_init = Qwen3_5DecoderLayer.__init__
    orig_fwd = Qwen3_5DecoderLayer.forward
    orig_model_fwd = Qwen3_5Model.forward
    assert (
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(_as_model_config(cfg_moe))
        is None
    )
    assert Qwen3_5DecoderLayer.__init__ is orig_init
    assert Qwen3_5DecoderLayer.forward is orig_fwd
    assert Qwen3_5Model.forward is orig_model_fwd
    assert len(warned) == 1
    assert "TPU_ATTN_MOE_OVERLAP=True" in warned[0]

    warned.clear()
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
    assert (
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(_as_model_config(cfg_moe))
        is None
    )
    assert warned == []


def test_maybe_patch_rejects_fused_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError when USE_MOE_FUSED_EP_KERNEL=True."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    cfg_moe = _StubModelConfig(
        hf_text_config=_StubHfTextConfig(model_type="qwen3_5_moe_text")
    )
    with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL=False"):
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(_as_model_config(cfg_moe))


def test_maybe_patch_rejects_sequence_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError when TPU_MOE_SEQUENCE_PARALLEL=True, leaving class unpatched."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    cfg_moe = _StubModelConfig(
        hf_text_config=_StubHfTextConfig(model_type="qwen3_5_moe_text")
    )
    with pytest.raises(ValueError, match="TPU_MOE_SEQUENCE_PARALLEL=False"):
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(_as_model_config(cfg_moe))
    assert Qwen3_5DecoderLayer.__init__ is orig_decoder_init
    assert "forward" not in Qwen3_5DecoderLayer.__dict__


@pytest.mark.parametrize(
    ("hidden_size", "tp_size", "element_size", "expected"),
    [
        (4096, 8, 2, 8192),
        (4096, 2, 2, 2048),
        (8192, 8, 2, 4096),
    ],
)
def test_overlap_min_tokens(
    hidden_size: int,
    tp_size: int,
    element_size: int,
    expected: int,
) -> None:
    """Verifies token threshold derivation from 8 MiB per-chip activation floor."""
    assert overlap_min_tokens(hidden_size, tp_size, element_size) == expected


def test_configure_layer() -> None:
    """Layer configuration sets expected flags and gate."""
    proj = _StubProj(reduce_results=True)
    dummy_gate = object()
    moe_config = _StubMoeConfig(skip_final_all_reduce=False)
    layer = _StubDecoderLayer(
        use_attn_reduce_scatter_for_moe=False,
        self_attn=_StubSelfAttn(o_proj=proj),
        linear_attn=None,
        layer_type="full_attention",
        mlp=_StubMlp(
            tp_size=2,
            replicate_shared_expert=False,
            experts=_StubExperts(gate=dummy_gate, moe_config=moe_config),
        ),
    )
    configure_layer(_as_decoder_layer(layer))
    assert proj.reduce_results is False
    assert layer.mlp.experts.gate is None
    assert moe_config.skip_final_all_reduce is True

    proj_lin = _StubProj(reduce_results=True)
    layer.layer_type = "linear_attention"
    layer.linear_attn = _StubLinearAttn(out_proj=proj_lin)
    layer.self_attn = None
    configure_layer(_as_decoder_layer(layer))
    assert proj_lin.reduce_results is False

    layer.mlp.tp_size = 1
    with pytest.raises(ValueError, match="TP > 1"):
        configure_layer(_as_decoder_layer(layer))
    layer.mlp.tp_size = 2

    layer.use_attn_reduce_scatter_for_moe = True
    with pytest.raises(ValueError, match="sequence-parallel MoE"):
        configure_layer(_as_decoder_layer(layer))
    layer.use_attn_reduce_scatter_for_moe = False

    layer.mlp.replicate_shared_expert = True
    with pytest.raises(ValueError, match="replicate_shared_expert"):
        configure_layer(_as_decoder_layer(layer))


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("tp", [2, 8])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("layer_scale", [False, True])
@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_forward_matches_upstream(
    monkeypatch: pytest.MonkeyPatch,
    overlap: bool,
    tp: int,
    with_residual: bool,
    layer_scale: bool,
    layer_type: str,
) -> None:
    """Multi-rank test verifying forward output equivalence against upstream."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0 if overlap else 1024)
    torch.manual_seed(42)
    n = 16
    d = 64
    x = torch.randn(n, d, dtype=torch.float32)
    res = torch.randn(n, d, dtype=torch.float32) if with_residual else None
    pos = torch.arange(n)

    input_norm = _CpuRMSNorm(d)
    post_norm = _CpuRMSNorm(d)
    gate_linear = nn.Linear(d, 8, bias=False)

    def gate(h: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Compute router logits for test layer."""
        return gate_linear(h), None

    attn_projs = [nn.Linear(d, d, bias=False) for _ in range(tp)]
    expert_projs = [nn.Linear(d, d, bias=False) for _ in range(tp)]

    attn_ls = torch.tensor([0.1])
    ffn_ls = torch.tensor([0.2])

    up_layer: Any = Qwen3NextDecoderLayer.__new__(Qwen3NextDecoderLayer)
    nn.Module.__init__(up_layer)
    up_layer.input_layernorm = input_norm
    up_layer.post_attention_layernorm = post_norm
    up_layer.use_attn_reduce_scatter_for_moe = False
    up_layer.layer_type = layer_type
    up_layer.layer_scale = layer_scale
    if layer_scale:
        up_layer.attn_layer_scale = attn_ls
        up_layer.ffn_layer_scale = ffn_ls
    if layer_type == "linear_attention":
        up_layer.linear_attn = lambda hidden_states: sum(
            p(hidden_states) for p in attn_projs
        )
        up_layer.self_attn = None
    else:
        up_layer.self_attn = lambda hidden_states, *_args, **_kwargs: sum(
            p(hidden_states) for p in attn_projs
        )
        up_layer.linear_attn = None
    up_layer.mlp = lambda h: sum(
        ep(h) * (gate(h)[0].sum(-1, keepdim=True) * 0.01 + 1.0) for ep in expert_projs
    )

    up_out, up_res = Qwen3NextDecoderLayer.forward(
        up_layer, x.clone(), res.clone() if res is not None else None, pos
    )

    mock = _TPCollectiveMock(tp)
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        mock.reduce_scatter,
    )
    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", mock.all_gather
    )
    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_reduce", mock.all_reduce
    )

    def run_rank(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute single TP rank overlap forward."""
        rank_layer = _StubDecoderLayer(
            sharded_spine=True,
            input_layernorm=input_norm,
            post_attention_layernorm=post_norm,
            layer_type=layer_type,
            layer_scale=layer_scale,
            attn_layer_scale=attn_ls if layer_scale else None,
            ffn_layer_scale=ffn_ls if layer_scale else None,
            linear_attn=(
                (lambda hidden_states: attn_projs[rank](hidden_states))
                if layer_type == "linear_attention"
                else None
            ),
            self_attn=(
                (
                    lambda hidden_states, *_args, **_kwargs: attn_projs[rank](
                        hidden_states
                    )
                )
                if layer_type != "linear_attention"
                else None
            ),
            mlp=_StubMlp(
                tp_size=tp,
                gate=gate,
                experts=lambda hidden_states, router_logits: (
                    expert_projs[rank](hidden_states)
                    * (router_logits.sum(-1, keepdim=True) * 0.01 + 1.0)
                ),
            ),
        )
        x_in = x.clone()
        input_residual = res.clone() if res is not None else None
        out, ret_res = overlap_forward(
            _as_decoder_layer(rank_layer), x_in, input_residual, pos
        )
        assert ret_res is not None
        if overlap and not with_residual:
            assert ret_res is x_in
        return out, ret_res

    pairs = _run_on_tp_threads(tp, run_rank)
    first_rank_out = pairs[0][0]
    for out_r, res_r in pairs:
        assert torch.allclose(out_r + res_r, up_out + up_res, atol=1e-5)
        assert torch.allclose(out_r, first_rank_out, atol=1e-6)


def test_overlap_disabled_when_compiling_outside_shape_variants_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiling outside shape_variants.warmup() skips the token-count branch."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    calls: list[str] = []
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, _dim=0: (calls.append("rs"), x)[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, _dim=0: (calls.append("ag"), x)[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: (calls.append("ar"), x)[1],
    )

    d = 16
    layer = _StubDecoderLayer(
        sharded_spine=True,
        input_layernorm=_CpuRMSNorm(d),
        post_attention_layernorm=_CpuRMSNorm(d),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=lambda hidden_states, *_args, **_kwargs: hidden_states,
        linear_attn=None,
        mlp=_StubMlp(
            tp_size=2,
            gate=lambda h: (torch.zeros((h.shape[0], 4)), None),
            experts=lambda hidden_states, *_args, **_kwargs: hidden_states,
        ),
    )

    x = torch.ones((16, d), dtype=torch.float32)
    out, res = overlap_forward(_as_decoder_layer(layer), x, None, torch.arange(16))
    assert out.shape == (16, d)
    assert res is not None
    assert calls == ["ar", "ar"]

    calls.clear()
    with shape_variants.warmup():
        overlap_forward(_as_decoder_layer(layer), x, None, torch.arange(16))
    assert calls == ["rs", "ag", "ag", "ar"]


def test_patched_init_configures_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """_patched_init runs _orig_init and configures layer attributes."""
    init_calls: list[tuple[object, str, str]] = []

    def fake_orig_init(
        self: Qwen3_5DecoderLayer,
        vllm_config: object,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        """Populate stub layer attributes during simulated __init__."""
        init_calls.append((vllm_config, layer_type, prefix))
        layer_obj: Any = self
        layer_obj.use_attn_reduce_scatter_for_moe = False
        layer_obj.layer_type = layer_type
        layer_obj.self_attn = _StubSelfAttn(o_proj=_StubProj(reduce_results=True))
        layer_obj.linear_attn = None
        layer_obj.mlp = _StubMlp(
            tp_size=2,
            replicate_shared_expert=False,
            experts=_StubExperts(
                gate=object(),
                moe_config=_StubMoeConfig(skip_final_all_reduce=False),
            ),
        )

    monkeypatch.setattr(qwen3_5_patch, "_orig_init", fake_orig_init)
    dummy_layer = _StubDecoderLayer()
    patched_decoder_init(
        _as_decoder_layer(dummy_layer),
        "cfg",
        "full_attention",
        prefix="model.layers.0",
    )
    assert init_calls == [("cfg", "full_attention", "model.layers.0")]
    assert dummy_layer.sharded_spine is True
    assert dummy_layer.self_attn.o_proj.reduce_results is False
    assert dummy_layer.mlp.experts.gate is None
    assert dummy_layer.mlp.experts.moe_config.skip_final_all_reduce is True

    dummy_mtp = _StubDecoderLayer()
    patched_decoder_init(
        _as_decoder_layer(dummy_mtp),
        "cfg",
        "full_attention",
        prefix="mtp.layers.0",
    )
    assert dummy_mtp.sharded_spine is False


def test_configure_layer_rejects_pp_when_fp8_overlap_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_configure_layer rejects PP > 1 when FP8 overlap all-gather is enabled."""
    layer = _StubDecoderLayer(
        layer_type="full_attention",
        use_attn_reduce_scatter_for_moe=False,
        self_attn=_StubSelfAttn(o_proj=_StubProj(reduce_results=True)),
        mlp=_StubMlp(
            tp_size=2,
            replicate_shared_expert=False,
            experts=_StubExperts(
                gate="router", moe_config=_StubMoeConfig(skip_final_all_reduce=False)
            ),
        ),
    )
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_5_patch, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        qwen3_5_patch, "get_pp_group", lambda: _StubPpGroup(world_size=2)
    )
    with pytest.raises(ValueError, match="pipeline parallelism"):
        configure_layer(_as_decoder_layer(layer))

    monkeypatch.setattr(
        qwen3_5_patch, "get_pp_group", lambda: _StubPpGroup(world_size=1)
    )
    configure_layer(_as_decoder_layer(layer))

    monkeypatch.setattr(
        qwen3_5_patch, "get_pp_group", lambda: _StubPpGroup(world_size=2)
    )
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
    configure_layer(_as_decoder_layer(layer))


def test_overlap_forward_mtp_layer_skips_sharded_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MTP predictor layers with sharded_spine=False skip FP8 gather and RS."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    wire_log: list[str] = []
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, _dim=0: (wire_log.append("reduce_scatter"), x[: x.shape[0] // 2])[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, _dim=0: (wire_log.append("all_gather"), torch.cat([x, x], dim=0))[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: (wire_log.append("all_reduce"), x)[1],
    )
    layer = _StubDecoderLayer(
        sharded_spine=False,
        input_layernorm=_CpuRMSNorm(64),
        post_attention_layernorm=_CpuRMSNorm(64),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=(lambda hidden_states, *_args, **_kwargs: hidden_states),
        linear_attn=None,
        mlp=_StubMlp(
            tp_size=2,
            gate=(lambda h: (h @ torch.randn(64, 4, dtype=h.dtype), None)),
            experts=(lambda hidden_states, *_args, **_kwargs: hidden_states),
        ),
    )
    x = torch.randn(16, 64, dtype=torch.bfloat16)
    pos = torch.arange(16)
    _out, res = overlap_forward(_as_decoder_layer(layer), x, None, pos)
    assert res is not None
    assert "all_reduce" in wire_log
    assert wire_log.count("reduce_scatter") == 1
    assert wire_log == ["reduce_scatter", "all_gather", "all_gather", "all_reduce"]


def test_overlap_forward_collective_schedule_by_token_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default 8 MiB floor dispatches < min_tokens to AR and >= min_tokens to RS+AG."""
    calls: list[str] = []
    gate_rows: list[int] = []
    tp_size = 2
    d = 4096
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, _dim=0: (calls.append("rs"), x[: x.shape[0] // tp_size])[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, _dim=0: (calls.append("ag"), torch.cat([x] * tp_size, dim=0))[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: (calls.append("ar"), x)[1],
    )
    norm = _CpuRMSNorm(d).to(torch.bfloat16)
    layer = _StubDecoderLayer(
        sharded_spine=True,
        input_layernorm=norm,
        post_attention_layernorm=norm,
        layer_type="full_attention",
        layer_scale=False,
        self_attn=lambda hidden_states, *_args, **_kwargs: hidden_states,
        linear_attn=None,
        mlp=_StubMlp(
            tp_size=tp_size,
            gate=lambda h: (
                gate_rows.append(h.shape[0]),
                torch.zeros((h.shape[0], 4), dtype=h.dtype),
                None,
            )[1:],
            experts=lambda hidden_states, *_args, **_kwargs: hidden_states,
        ),
    )

    x_small = torch.ones((2047, d), dtype=torch.bfloat16)
    overlap_forward(_as_decoder_layer(layer), x_small, None, torch.arange(2047))
    assert calls == ["ar", "ar"]
    assert gate_rows == [2047]

    calls.clear()
    gate_rows.clear()
    x_large = torch.ones((2048, d), dtype=torch.bfloat16)
    overlap_forward(_as_decoder_layer(layer), x_large, None, torch.arange(2048))
    assert calls == ["rs", "ag", "ag", "ar"]
    assert gate_rows == [1024]


def test_overlap_forward_narrows_padded_all_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded all_gather outputs are narrowed to num_tokens before experts run."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, _dim=0: x[:4],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, _dim=0: torch.cat([x, x], dim=0),
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: x,
    )
    seen_shapes: list[tuple[torch.Size, torch.Size]] = []
    d = 16
    layer = _StubDecoderLayer(
        sharded_spine=True,
        input_layernorm=_CpuRMSNorm(d),
        post_attention_layernorm=_CpuRMSNorm(d),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=lambda hidden_states, *_args, **_kwargs: hidden_states,
        linear_attn=None,
        mlp=_StubMlp(
            tp_size=2,
            gate=lambda h: (torch.zeros((h.shape[0], 4)), None),
            experts=lambda hidden_states, router_logits: (
                seen_shapes.append((hidden_states.shape, router_logits.shape)),
                hidden_states,
            )[1],
        ),
    )
    x = torch.ones((6, d), dtype=torch.float32)
    out, _ = overlap_forward(_as_decoder_layer(layer), x, None, torch.arange(6))
    assert seen_shapes == [(torch.Size([6, d]), torch.Size([6, 4]))]
    assert out.shape == (6, d)


_FP8_BF16_ROUND_RTOL = (1.0 + 2.0**-4) * (1.0 + 2.0**-8) ** 2 - 1.0
_BF16_ACCUM_EPS = 2.0**-7


def _assert_fp8_spine_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    fp8_gathers: int,
    tp_size: int,
) -> tuple[float, float]:
    """Bounds zero-mean residual-stream parity from FP8 gather + bf16 accum terms."""
    rel_budget = fp8_gathers * _FP8_BF16_ROUND_RTOL + tp_size * _BF16_ACCUM_EPS
    exp_f32 = expected.detach().float()
    diff_f32 = actual.detach().float() - exp_f32
    rel_rmse = float(diff_f32.norm() / exp_f32.norm().clamp(min=1e-12))
    assert rel_rmse <= rel_budget
    atol = rel_budget * float(exp_f32.abs().max())
    torch.testing.assert_close(actual, expected, rtol=rel_budget, atol=atol)
    return rel_budget, atol


def _run_on_tp_threads[T](tp_size: int, fn: Callable[[int], T]) -> list[T]:
    """Runs fn(rank) on tp_size threads with thread-local rank state."""
    outs: list[T | None] = [None] * tp_size

    def worker(rank: int) -> None:
        """Set thread-local rank and execute rank callback."""
        _THREAD_STATE.rank = rank
        outs[rank] = fn(rank)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()
    return [o for o in outs if o is not None]


def _install_logged_collectives(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    wire_log: list[tuple[str, torch.Size, torch.dtype]],
) -> None:
    """Install thread-synchronized TP collectives recording rank-0 wire traffic."""
    mock = _TPCollectiveMock(tp_size)

    def logged_rs(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Log rank-0 reduce-scatter shape and dtype."""
        if _THREAD_STATE.rank == 0:
            wire_log.append(("reduce_scatter", tensor.shape, tensor.dtype))
        return mock.reduce_scatter(tensor, dim=dim)

    def logged_ag(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Log rank-0 all-gather shape and dtype."""
        if _THREAD_STATE.rank == 0:
            wire_log.append(("all_gather", tensor.shape, tensor.dtype))
        return mock.all_gather(tensor, dim=dim)

    def logged_ar(tensor: torch.Tensor) -> torch.Tensor:
        """Log rank-0 all-reduce shape and dtype."""
        if _THREAD_STATE.rank == 0:
            wire_log.append(("all_reduce", tensor.shape, tensor.dtype))
        return mock.all_reduce(tensor)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_reduce_scatter", logged_rs
    )
    monkeypatch.setattr(qwen3_5_patch, "tensor_model_parallel_all_gather", logged_ag)
    monkeypatch.setattr(qwen3_5_patch, "tensor_model_parallel_all_reduce", logged_ar)


def _run_single_layer_with_wire_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap: bool,
    fp8_all_gather: bool,
    cfg: _WireCaptureConfig = _DEFAULT_WIRE_CFG,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[tuple[str, torch.Size, torch.dtype]],
    list[torch.dtype],
]:
    """Runs _overlap_forward across TP threads and records rank-0 wire shapes/dtypes."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0 if overlap else 1024)
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", fp8_all_gather)
    torch.manual_seed(11)
    n = 16
    shard_n = n // cfg.tp_size
    x = torch.randn(n, cfg.d, dtype=torch.bfloat16)
    res_in = torch.randn(n, cfg.d, dtype=torch.bfloat16) if cfg.with_residual else None
    pos = torch.arange(n)

    input_norm = _CpuRMSNorm(cfg.d).to(torch.bfloat16)
    post_norm = _CpuRMSNorm(cfg.d).to(torch.bfloat16)
    gate_linear = nn.Linear(cfg.d, 8, bias=False).to(torch.bfloat16)

    def gate(h: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Project hidden states to router logits."""
        return gate_linear(h), None

    attn_projs = [
        nn.Linear(cfg.d, cfg.d, bias=False).to(torch.bfloat16)
        for _ in range(cfg.tp_size)
    ]
    expert_projs = [
        nn.Linear(cfg.d, cfg.d, bias=False).to(torch.bfloat16)
        for _ in range(cfg.tp_size)
    ]
    attn_ls = torch.tensor([0.1], dtype=torch.bfloat16)
    ffn_ls = torch.tensor([0.2], dtype=torch.bfloat16)

    wire_log: list[tuple[str, torch.Size, torch.dtype]] = []
    experts_dtypes: list[torch.dtype] = []
    _install_logged_collectives(monkeypatch, cfg.tp_size, wire_log)
    sharded_in = overlap and fp8_all_gather and cfg.with_residual

    def run_rank(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run single TP rank forward and capture expert input dtype."""

        def experts(
            hidden_states: torch.Tensor, router_logits: torch.Tensor
        ) -> torch.Tensor:
            """Apply per-rank expert projection scaled by router logits."""
            if rank == 0:
                experts_dtypes.append(hidden_states.dtype)
            return expert_projs[rank](hidden_states) * (
                router_logits.sum(-1, keepdim=True) * 0.01 + 1.0
            )

        is_linear = cfg.layer_type == "linear_attention"
        rank_layer = _StubDecoderLayer(
            sharded_spine=True,
            input_layernorm=input_norm,
            post_attention_layernorm=post_norm,
            layer_type=cfg.layer_type,
            layer_scale=cfg.layer_scale,
            attn_layer_scale=attn_ls,
            ffn_layer_scale=ffn_ls,
            linear_attn=(
                (lambda hidden_states: attn_projs[rank](hidden_states))
                if is_linear
                else None
            ),
            self_attn=(
                None
                if is_linear
                else (
                    lambda hidden_states, *_args, **_kwargs: attn_projs[rank](
                        hidden_states
                    )
                )
            ),
            mlp=_StubMlp(tp_size=cfg.tp_size, gate=gate, experts=experts),
        )
        sl = slice(rank * shard_n, (rank + 1) * shard_n)
        x_rank = x[sl].clone() if sharded_in else x.clone()
        res_rank = (
            None
            if res_in is None
            else (res_in[sl].clone() if sharded_in else res_in.clone())
        )
        out, ret_res = overlap_forward(
            _as_decoder_layer(rank_layer),
            x_rank,
            res_rank,
            pos,
        )
        assert ret_res is not None
        return out, ret_res

    pairs = _run_on_tp_threads(cfg.tp_size, run_rank)
    if overlap and fp8_all_gather:
        out_full = torch.cat([p[0] for p in pairs], dim=0)
        res_full = torch.cat([p[1] for p in pairs], dim=0)
        return out_full, res_full, wire_log, experts_dtypes
    out0, res0 = pairs[0]
    return out0, res0, wire_log, experts_dtypes


def test_overlap_forward_fp8_flag_uses_fp8_h_gather_and_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 0 with FP8 overlap flag ships h in FP8 and reduce-scatters MoE output."""
    out_fp8, res_fp8, wire_fp8, exp_dtypes_fp8 = _run_single_layer_with_wire_capture(
        monkeypatch, overlap=True, fp8_all_gather=True
    )
    out_bf16, res_bf16, wire_bf16, exp_dtypes_bf16 = (
        _run_single_layer_with_wire_capture(
            monkeypatch, overlap=True, fp8_all_gather=False
        )
    )

    assert wire_bf16 == [
        ("reduce_scatter", torch.Size([16, 64]), torch.bfloat16),
        ("all_gather", torch.Size([8, 8]), torch.bfloat16),
        ("all_gather", torch.Size([8, 64]), torch.bfloat16),
        ("all_reduce", torch.Size([16, 64]), torch.bfloat16),
    ]
    assert wire_fp8 == [
        ("reduce_scatter", torch.Size([16, 64]), torch.bfloat16),
        ("all_gather", torch.Size([8, 8]), torch.bfloat16),
        ("all_gather", torch.Size([8, 64]), torch.int8),
        ("all_gather", torch.Size([8, 1]), torch.bfloat16),
        ("reduce_scatter", torch.Size([16, 64]), torch.bfloat16),
    ]
    assert exp_dtypes_fp8 == [torch.bfloat16]
    assert exp_dtypes_bf16 == [torch.bfloat16]
    stream_fp8 = out_fp8 + res_fp8
    stream_bf16 = out_bf16 + res_bf16
    _assert_fp8_spine_parity(stream_fp8, stream_bf16, fp8_gathers=1, tp_size=2)


def test_overlap_forward_fp8_mid_stack_full_attention_with_layer_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-stack full_attention layer gathers normed input in FP8 and exits via RS."""
    mid_cfg = _WireCaptureConfig(
        tp_size=2,
        d=1024,
        with_residual=True,
        layer_type="full_attention",
        layer_scale=True,
    )
    out_fp8, res_fp8, wire_fp8, exp_dtypes_fp8 = _run_single_layer_with_wire_capture(
        monkeypatch,
        overlap=True,
        fp8_all_gather=True,
        cfg=mid_cfg,
    )
    out_bf16, res_bf16, _, _ = _run_single_layer_with_wire_capture(
        monkeypatch,
        overlap=True,
        fp8_all_gather=False,
        cfg=mid_cfg,
    )
    assert wire_fp8 == [
        ("all_gather", torch.Size([8, 1024]), torch.int8),
        ("all_gather", torch.Size([8, 1]), torch.bfloat16),
        ("reduce_scatter", torch.Size([16, 1024]), torch.bfloat16),
        ("all_gather", torch.Size([8, 8]), torch.bfloat16),
        ("all_gather", torch.Size([8, 1024]), torch.int8),
        ("all_gather", torch.Size([8, 1]), torch.bfloat16),
        ("reduce_scatter", torch.Size([16, 1024]), torch.bfloat16),
    ]
    assert exp_dtypes_fp8 == [torch.bfloat16]
    stream_fp8 = out_fp8 + res_fp8
    stream_bf16 = out_bf16 + res_bf16
    _assert_fp8_spine_parity(stream_fp8, stream_bf16, fp8_gathers=2, tp_size=2)


def test_non_overlap_branch_ignores_fp8_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below min_tokens the all-reduce path runs; the flag adds no all-gathers."""
    out_fp8, _, wire_fp8, _ = _run_single_layer_with_wire_capture(
        monkeypatch, overlap=False, fp8_all_gather=True
    )
    out_bf16, _, wire_bf16, _ = _run_single_layer_with_wire_capture(
        monkeypatch, overlap=False, fp8_all_gather=False
    )
    assert [op for op, _, _ in wire_fp8] == ["all_reduce", "all_reduce"]
    assert wire_fp8 == wire_bf16
    assert torch.equal(out_fp8, out_bf16)


def _build_two_layer_stub_model(
    rank: int,
    tp_size: int,
    norms: list[_CpuRMSNorm],
    gates: list[nn.Linear],
    projs: tuple[list[list[nn.Linear]], list[list[nn.Linear]]],
) -> _StubModel:
    """Build a 2-layer stub model for a single TP rank."""
    attn_w, exp_w = projs
    layers: list[_StubDecoderLayer] = []
    for l_idx in range(2):
        g_lin = gates[l_idx]
        a_lin = attn_w[l_idx][rank]
        e_lin = exp_w[l_idx][rank]
        layers.append(
            _StubDecoderLayer(
                sharded_spine=True,
                input_layernorm=norms[2 * l_idx],
                post_attention_layernorm=norms[2 * l_idx + 1],
                layer_type="linear_attention" if l_idx == 0 else "full_attention",
                layer_scale=False,
                linear_attn=(lambda hidden_states, a=a_lin: a(hidden_states)),
                self_attn=(
                    lambda hidden_states, *_args, a=a_lin, **_kwargs: a(hidden_states)
                ),
                mlp=_StubMlp(
                    tp_size=tp_size,
                    gate=(lambda h, g=g_lin: (g(h), None)),
                    experts=(
                        lambda hidden_states, *_args, e=e_lin, **_kwargs: e(
                            hidden_states
                        )
                    ),
                ),
            )
        )
    return _StubModel(start_layer=0, layers=layers)


@pytest.mark.parametrize("with_aux", [False, True])
def test_overlap_model_forward_sharded_spine_matches_bf16_stack(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_aux: bool,
) -> None:
    """Verify 2-layer stack keeps residual sharded and gathers output at exit."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    torch.manual_seed(19)
    tp_size = 2
    n = 16
    d = 64
    x = torch.randn(n, d, dtype=torch.bfloat16)
    pos = torch.arange(n)

    norms = [_CpuRMSNorm(d).to(torch.bfloat16) for _ in range(5)]
    gates = [nn.Linear(d, 8, bias=False).to(torch.bfloat16) for _ in range(2)]
    attn_w = [
        [nn.Linear(d, d, bias=False).to(torch.bfloat16) for _ in range(tp_size)]
        for _ in range(2)
    ]
    exp_w = [
        [nn.Linear(d, d, bias=False).to(torch.bfloat16) for _ in range(tp_size)]
        for _ in range(2)
    ]

    def run_stack(
        *, fp8_flag: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
        """Execute the 2-layer stack across TP threads under the given FP8 flag."""
        monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", fp8_flag)
        wire_log: list[tuple[str, torch.Size, torch.dtype]] = []
        _install_logged_collectives(monkeypatch, tp_size, wire_log)

        def fake_orig_model_forward(
            model_self: _StubModel,
            input_ids: torch.Tensor | None = None,
            positions: torch.Tensor | None = None,
            intermediate_tensors: object = None,
            inputs_embeds: torch.Tensor | None = None,
            **_kwargs: object,
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Run stub model layers sequentially and optionally return aux state."""
            _ = (input_ids, intermediate_tensors, inputs_embeds)
            h = x.clone()
            res: torch.Tensor | None = None
            aux: list[torch.Tensor] = []
            for idx, lyr in enumerate(model_self.layers):
                h, res = overlap_forward(_as_decoder_layer(lyr), h, res, positions)
                assert res is not None
                if with_aux and idx == 0:
                    aux.append(h + res)
            h, _ = norms[4](h, res)
            return (h, aux) if with_aux else h

        monkeypatch.setattr(
            qwen3_5_patch, "_orig_model_forward", fake_orig_model_forward
        )

        def run_rank(
            rank: int,
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Build and execute stub 2-layer model on one TP rank."""
            model = _build_two_layer_stub_model(
                rank, tp_size, norms, gates, (attn_w, exp_w)
            )
            return overlap_model_forward(_as_model(model), None, pos)

        res_ranks = _run_on_tp_threads(tp_size, run_rank)
        ops_r0 = [
            f"ag:{dt}"
            if op == "all_gather"
            else ("rs" if op == "reduce_scatter" else "ar")
            for op, _, dt in wire_log
        ]
        first = res_ranks[0]
        if isinstance(first, tuple):
            return first[0], first[1], ops_r0
        return first, [], ops_r0

    out_fp8, aux_fp8, ops_fp8 = run_stack(fp8_flag=True)
    out_bf16, aux_bf16, ops_bf16 = run_stack(fp8_flag=False)
    assert "ar" not in ops_fp8
    assert ops_bf16.count("ar") == _EXPECTED_BF16_ALL_REDUCE_COUNT
    assert out_fp8.shape == (n, d)
    rtol_s, atol_s = _assert_fp8_spine_parity(
        out_fp8, out_bf16, fp8_gathers=3, tp_size=tp_size
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            torch.zeros_like(out_fp8), out_bf16, rtol=rtol_s, atol=atol_s
        )
    if with_aux:
        assert len(aux_fp8) == 1
        assert aux_fp8[0].shape == (n, d)
        _assert_fp8_spine_parity(
            aux_fp8[0], aux_bf16[0], fp8_gathers=1, tp_size=tp_size
        )


def _ensure_fake_qmm_op() -> None:
    """Register the vllm_tpu::fake_qmm custom op once for TorchDynamo trace tests."""
    if _FAKE_QMM_HOLDER:
        return
    lib = torch.library.Library("vllm_tpu", "FRAGMENT")
    lib.define("fake_qmm(Tensor x, Tensor w) -> Tensor")
    lib.impl("fake_qmm", lambda x, w: x @ w, "CPU")
    torch.library.register_fake(
        "vllm_tpu::fake_qmm",
        lambda x, w: x.new_empty((*x.shape[:-1], w.shape[1])),
    )
    _FAKE_QMM_HOLDER.append(lib)


class _FakeLinearAttn(nn.Module):
    """Fake linear attention module calling the vllm_tpu::fake_qmm custom op."""

    def __init__(self, d: int) -> None:
        """Initialize weight parameter."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply fake_qmm projection."""
        orig_shape = (*hidden_states.size()[:-1], self.w.shape[1])
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])
        out = torch.ops.vllm_tpu.fake_qmm(x_flat, self.w)
        return out.view(orig_shape)


class _FakeFullAttn(nn.Module):
    """Fake full attention module consuming hidden_states and positions."""

    def __init__(self, d: int) -> None:
        """Initialize weight parameter."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Combine projected hidden states with broadcast positions."""
        return (hidden_states @ self.w) + positions.unsqueeze(-1).to(
            hidden_states.dtype
        )


class _FakeMLP(nn.Module):
    """Fake MoE MLP module for TorchDynamo trace tests."""

    def __init__(self, d: int, tp_size: int = 2) -> None:
        """Initialize gate and expert weights."""
        super().__init__()
        self.tp_size = tp_size
        self.gate_w = nn.Parameter(torch.randn(d, 4, dtype=torch.bfloat16))
        self.exp_w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

    def gate(self, h: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Compute router logits."""
        return h @ self.gate_w, None

    def experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        **_kw: object,
    ) -> torch.Tensor:
        """Project hidden states through expert weight matrix."""
        _ = (router_logits, _kw)
        return hidden_states @ self.exp_w


class _FakeRMSNorm(nn.Module):
    """Fake RMSNorm module supporting fused residual addition."""

    def __init__(self, d: int) -> None:
        """Initialize unit scale weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Apply scale and optional residual addition."""
        if residual is not None:
            x = x + residual
            return x * self.weight, x
        return x * self.weight


class _FakeDecoderLayer(nn.Module):
    """Fake Qwen3.5 decoder layer combining linear or full attention with MoE MLP."""

    def __init__(self, d: int, layer_type: str) -> None:
        """Initialize norms, attention, and MoE MLP."""
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = _FakeRMSNorm(d)
        self.post_attention_layernorm = _FakeRMSNorm(d)
        self.layer_scale = False
        self.sharded_spine = True
        if layer_type == "linear_attention":
            self.linear_attn: _FakeLinearAttn | None = _FakeLinearAttn(d)
            self.self_attn: _FakeFullAttn | None = None
        else:
            self.linear_attn = None
            self.self_attn = _FakeFullAttn(d)
        self.mlp = _FakeMLP(d)


class _FakeBackbone(nn.Module):
    """Fake 2-layer Qwen3.5 backbone for TorchDynamo trace handover tests."""

    def __init__(self, d: int) -> None:
        """Initialize embedding and 2 decoder layers."""
        super().__init__()
        self.embed = nn.Embedding(100, d, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(
            [
                _FakeDecoderLayer(d, "linear_attention"),
                _FakeDecoderLayer(d, "full_attention"),
            ]
        )
        self.start_layer = 0

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run embedding followed by _overlap_forward across layers."""
        hidden_states = self.embed(input_ids)
        residual = None
        for layer in self.layers:
            layer_any: Any = layer
            hidden_states, residual = overlap_forward(
                layer_any, hidden_states, residual, positions
            )
        return hidden_states


@pytest.mark.usefixtures("_reset_dynamo_state")
def test_overlap_model_forward_shape_variants_handover_across_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify ShapeEnv unifies dynamic token dims across 2 backbone warmup traces."""
    _ensure_fake_qmm_op()
    d = 16
    tp_size = 2
    model = _FakeBackbone(d)

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_5_patch, "_overlap_min_tokens", lambda *_a: 64)
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, _dim=0: x[: x.shape[0] // tp_size],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, _dim=0: torch.cat([x] * tp_size, dim=0),
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: x,
    )

    captured: list[tuple[torch.fx.GraphModule, torch.fx.GraphModule, list[Any]]] = []

    def custom_backend(
        gm: torch.fx.GraphModule, _example_inputs: list[torch.Tensor]
    ) -> torch.fx.GraphModule:
        """Split captured FX graph and record submodule outputs."""
        split_gm, outputs = split_graph(gm, [])
        captured.append((gm, split_gm, outputs))
        return gm

    compiled = torch.compile(
        model, backend=custom_backend, fullgraph=True, dynamic=False
    )

    cache_dir = str(tmp_path / "aot_cache")
    with shape_variants.warmup() as session:
        ids8 = torch.randint(0, 100, (8,), dtype=torch.int64)
        pos8 = torch.arange(8, dtype=torch.int32)
        mark_dynamic(ids8, 0)
        mark_dynamic(pos8, 0)
        compiled(ids8, pos8)

        _, _, outputs1 = captured[-1]
        for item in outputs1:
            session.put(cache_dir, f"{item.submod_name}_8", item.graph, item.graph)

        reset_dynamo()
        compiled2 = torch.compile(
            model, backend=custom_backend, fullgraph=True, dynamic=False
        )
        ids64 = torch.randint(0, 100, (64,), dtype=torch.int64)
        pos64 = torch.arange(64, dtype=torch.int32)
        mark_dynamic(ids64, 0)
        mark_dynamic(pos64, 0)
        compiled2(ids64, pos64)

        _, _, outputs2 = captured[-1]
        for item in outputs2:
            res = shape_variants.runnable_for(
                cache_dir, f"{item.submod_name}_8", item.graph, size=8
            )
            assert res is not None
