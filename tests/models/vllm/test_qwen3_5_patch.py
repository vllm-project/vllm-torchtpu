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
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn
from torch._dynamo import mark_dynamic
from torch._dynamo import reset as reset_dynamo
from vllm.compilation.backends import split_graph
from vllm.model_executor.models import qwen3_next
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
    _fp8_gather as fp8_gather,
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

_FAKE_QUANTIZED_MATMUL_HOLDER: list[Any] = []


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
class _StubParallelConfig:
    """Stub vLLM ParallelConfig carrying pipeline_parallel_size."""

    pipeline_parallel_size: int = 1


@dataclass
class _StubVllmConfig:
    """Stub vLLM config carrying parallel_config."""

    parallel_config: _StubParallelConfig = field(default_factory=_StubParallelConfig)


@dataclass(frozen=True)
class _StubPpGroup:
    """Stub single-rank pipeline-parallel group for real Qwen3NextModel.forward."""

    is_first_rank: bool = True
    is_last_rank: bool = True


_SINGLE_RANK_PP_GROUP = _StubPpGroup()


@dataclass
class _StubDecoderLayer:
    """Stub Qwen3.5 decoder layer for unit testing overlap configuration."""

    layer_type: str = "full_attention"
    use_attn_reduce_scatter_for_moe: bool = False
    uses_sharded_residual: bool = True
    layer_scale: bool = False
    input_layernorm: Any = None
    post_attention_layernorm: Any = None
    self_attn: Any = None
    linear_attn: Any = None
    attn_layer_scale: torch.Tensor | None = None
    ffn_layer_scale: torch.Tensor | None = None
    mlp: _StubMlp = field(default_factory=_StubMlp)


class _IdentityModelNorm(nn.Module):
    """Identity final norm matching Qwen3NextModel's (hidden, residual) contract."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return hidden_states and residual unchanged."""
        return hidden_states, residual


def _build_real_qwen3_5_model(
    layers: list[nn.Module],
    norm: nn.Module,
    *,
    aux_hidden_state_layers: tuple[int, ...] = (),
    embed_tokens: nn.Module | None = None,
) -> Qwen3_5Model:
    """Construct a minimal real Qwen3_5Model via __new__ for forward-path tests."""
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model_any: Any = model
    model_any.do_not_compile = True
    model_any.start_layer = 0
    model_any.end_layer = len(layers)
    model_any.layers = nn.ModuleList(layers)
    model_any.norm = norm
    model_any.aux_hidden_state_layers = aux_hidden_state_layers
    if embed_tokens is not None:
        model_any.embed_tokens = embed_tokens
    return model


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


def test_maybe_patch_waits_for_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies patch registration returns and class method patching."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", False)
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
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
        assert Qwen3_5Model.forward is orig_model_fwd
    finally:
        Qwen3_5DecoderLayer.__init__ = orig_init
        if had_fwd:
            Qwen3_5DecoderLayer.forward = orig_fwd
        elif "forward" in Qwen3_5DecoderLayer.__dict__:
            del Qwen3_5DecoderLayer.forward
        if had_model_fwd:
            Qwen3_5Model.forward = orig_model_fwd
        elif "forward" in Qwen3_5Model.__dict__:
            del Qwen3_5Model.forward


def test_maybe_patch_only_wraps_model_forward_when_fp8_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3_5Model.forward is wrapped only when FP8 overlap all-gather is enabled."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    cfg_moe = _StubModelConfig(
        hf_text_config=_StubHfTextConfig(model_type="qwen3_5_moe_text")
    )
    orig_init = Qwen3_5DecoderLayer.__init__
    had_fwd = "forward" in Qwen3_5DecoderLayer.__dict__
    orig_fwd = Qwen3_5DecoderLayer.forward
    had_model_fwd = "forward" in Qwen3_5Model.__dict__
    orig_model_fwd = Qwen3_5Model.forward
    try:
        monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
        assert (
            qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(
                _as_model_config(cfg_moe)
            )
            is None
        )
        assert Qwen3_5DecoderLayer.__init__ is patched_decoder_init
        assert Qwen3_5DecoderLayer.forward is overlap_forward
        assert Qwen3_5Model.forward is orig_model_fwd

        monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
        assert (
            qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(
                _as_model_config(cfg_moe)
            )
            is None
        )
        assert Qwen3_5Model.forward is overlap_model_forward
    finally:
        Qwen3_5DecoderLayer.__init__ = orig_init
        if had_fwd:
            Qwen3_5DecoderLayer.forward = orig_fwd
        elif "forward" in Qwen3_5DecoderLayer.__dict__:
            del Qwen3_5DecoderLayer.forward
        if had_model_fwd:
            Qwen3_5Model.forward = orig_model_fwd
        elif "forward" in Qwen3_5Model.__dict__:
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
            uses_sharded_residual=True,
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
        uses_sharded_residual=True,
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


def _populate_stub_decoder_layer(layer: Qwen3_5DecoderLayer, layer_type: str) -> None:
    """Populate minimal decoder layer attributes for simulated __init__."""
    layer_obj: Any = layer
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


def _fake_decoder_orig_init(
    self: Qwen3_5DecoderLayer,
    vllm_config: object,
    layer_type: str,
    prefix: str = "",
) -> None:
    """Populate stub decoder layer attributes during simulated __init__."""
    _ = (vllm_config, prefix)
    _populate_stub_decoder_layer(self, layer_type)


def test_patched_init_configures_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """_patched_init runs _orig_init and configures layer attributes."""
    init_calls: list[tuple[object, str, str]] = []

    def recording_orig_init(
        self: Qwen3_5DecoderLayer,
        vllm_config: object,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        """Record init arguments and populate stub layer attributes."""
        init_calls.append((vllm_config, layer_type, prefix))
        _populate_stub_decoder_layer(self, layer_type)

    monkeypatch.setattr(qwen3_5_patch, "_orig_init", recording_orig_init)
    vllm_cfg = _StubVllmConfig()
    dummy_layer = _StubDecoderLayer()
    patched_decoder_init(
        _as_decoder_layer(dummy_layer),
        vllm_cfg,
        "full_attention",
        prefix="model.layers.0",
    )
    assert init_calls == [(vllm_cfg, "full_attention", "model.layers.0")]
    assert dummy_layer.uses_sharded_residual is True
    assert dummy_layer.self_attn.o_proj.reduce_results is False
    assert dummy_layer.mlp.experts.gate is None
    assert dummy_layer.mlp.experts.moe_config.skip_final_all_reduce is True

    dummy_mtp = _StubDecoderLayer()
    patched_decoder_init(
        _as_decoder_layer(dummy_mtp),
        vllm_cfg,
        "full_attention",
        prefix="mtp.layers.0",
    )
    assert dummy_mtp.uses_sharded_residual is False


def test_patched_init_rejects_pp_when_fp8_overlap_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_patched_init rejects PP > 1 when FP8 overlap all-gather is enabled."""
    monkeypatch.setattr(qwen3_5_patch, "_orig_init", _fake_decoder_orig_init)
    pp2_cfg = _StubVllmConfig(
        parallel_config=_StubParallelConfig(pipeline_parallel_size=2)
    )
    pp1_cfg = _StubVllmConfig(
        parallel_config=_StubParallelConfig(pipeline_parallel_size=1)
    )

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        patched_decoder_init(
            _as_decoder_layer(_StubDecoderLayer()),
            pp2_cfg,
            "full_attention",
            prefix="model.layers.0",
        )

    patched_decoder_init(
        _as_decoder_layer(_StubDecoderLayer()),
        pp1_cfg,
        "full_attention",
        prefix="model.layers.0",
    )

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
    patched_decoder_init(
        _as_decoder_layer(_StubDecoderLayer()),
        pp2_cfg,
        "full_attention",
        prefix="model.layers.0",
    )


def test_overlap_forward_mtp_layer_skips_uses_sharded_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MTP predictor layers with uses_sharded_residual=False skip FP8 gather and RS."""
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
    hidden_size = 64
    num_tokens = 16
    layer = _StubDecoderLayer(
        uses_sharded_residual=False,
        input_layernorm=_CpuRMSNorm(hidden_size),
        post_attention_layernorm=_CpuRMSNorm(hidden_size),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=(lambda hidden_states, *_args, **_kwargs: hidden_states),
        linear_attn=None,
        mlp=_StubMlp(
            tp_size=2,
            gate=(lambda h: (h @ torch.randn(hidden_size, 4, dtype=h.dtype), None)),
            experts=(lambda hidden_states, *_args, **_kwargs: hidden_states),
        ),
    )
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    pos = torch.arange(num_tokens)
    _out, res = overlap_forward(_as_decoder_layer(layer), x, None, pos)
    assert res is not None
    assert wire_log == ["reduce_scatter", "all_gather", "all_gather", "all_reduce"]


def test_overlap_forward_collective_schedule_by_token_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default 8 MiB floor dispatches < min_tokens to AR and >= min_tokens to RS+AG."""
    calls: list[str] = []
    gate_rows: list[int] = []
    tp_size = 2
    d = 4096
    # min_tokens = (8 * 1024 * 1024 * 2) // (4096 * 2) = 2048
    min_tokens = overlap_min_tokens(d, tp_size, torch.bfloat16.itemsize)
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
        uses_sharded_residual=True,
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

    below_threshold = min_tokens - 1
    x_small = torch.ones((below_threshold, d), dtype=torch.bfloat16)
    overlap_forward(
        _as_decoder_layer(layer), x_small, None, torch.arange(below_threshold)
    )
    assert calls == ["ar", "ar"]
    assert gate_rows == [below_threshold]

    calls.clear()
    gate_rows.clear()
    x_large = torch.ones((min_tokens, d), dtype=torch.bfloat16)
    overlap_forward(_as_decoder_layer(layer), x_large, None, torch.arange(min_tokens))
    assert calls == ["rs", "ag", "ag", "ar"]
    assert gate_rows == [min_tokens // tp_size]


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
        # 8 rows gathered > num_tokens=6, so narrow trims padded rows.
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
        uses_sharded_residual=True,
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


_FP8_E4M3_FINFO = torch.finfo(torch.float8_e4m3fn)
_BF16_FINFO = torch.finfo(torch.bfloat16)
# float8_e4m3fn has 3 mantissa bits (normal half-ULP = eps / 2 = 2**-4,
# subnormal half-ULP = smallest_normal * eps / 2 = 2**-10 in [-448, 448]).
_FP8_E4M3_REL_HALF_ULP = float(_FP8_E4M3_FINFO.eps) / 2.0
_FP8_E4M3_SUBNORM_REL_FLOOR = (
    float(_FP8_E4M3_FINFO.smallest_normal)
    * float(_FP8_E4M3_FINFO.eps)
    / (2.0 * float(_FP8_E4M3_FINFO.max))
)
_BF16_ACCUM_EPS = float(_BF16_FINFO.eps)
_FP8_BF16_ROUND_RTOL = (1.0 + _FP8_E4M3_REL_HALF_ULP) * (
    1.0 + _BF16_ACCUM_EPS / 2.0
) ** 2 - 1.0


def _fp8_round_trip_tol(x: torch.Tensor) -> tuple[float, float]:
    """Compute analytical rtol and atol for FP8 quantization and dequantization."""
    finfo = torch.finfo(x.dtype)
    u_out = float(finfo.eps) / 2.0
    two_roundings = (1.0 + u_out) ** 2
    rtol = (1.0 + _FP8_E4M3_REL_HALF_ULP) * two_roundings - 1.0
    amax = float(x.detach().float().abs().max())
    fp8_subnorm_atol = amax * _FP8_E4M3_SUBNORM_REL_FLOOR * two_roundings
    out_subnorm_atol = float(finfo.smallest_normal) * u_out
    return rtol, fp8_subnorm_atol + out_subnorm_atol


# Guard against zero-norm denominator in relative RMSE calculations.
_REL_RMSE_NORM_FLOOR = 1e-12


def _relative_rmse(actual: torch.Tensor, expected_f32: torch.Tensor) -> float:
    """Return L2 relative RMSE between actual and float32 expected tensors."""
    diff_f32 = actual.detach().float() - expected_f32
    return float(diff_f32.norm() / expected_f32.norm().clamp(min=_REL_RMSE_NORM_FLOOR))


def _assert_fp8_residual_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    fp8_gathers: int,
    tp_size: int,
) -> tuple[float, float]:
    """Bounds zero-mean residual-stream parity from FP8 gather + bf16 accum terms."""
    rel_budget = fp8_gathers * _FP8_BF16_ROUND_RTOL + tp_size * _BF16_ACCUM_EPS
    expected_f32 = expected.detach().float()
    assert _relative_rmse(actual, expected_f32) <= rel_budget
    atol = rel_budget * float(expected_f32.abs().max())
    torch.testing.assert_close(actual, expected, rtol=rel_budget, atol=atol)
    return rel_budget, atol


def _fp8_quantize_dequantize_ref(x: torch.Tensor) -> torch.Tensor:
    """Quantize and dequantize rows in float8_e4m3fn using independent math."""
    fp8_max = float(_FP8_E4M3_FINFO.max)
    work_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
    x_work = x.to(work_dtype)
    row_amax = x_work.abs().amax(dim=-1, keepdim=True)
    scale = (row_amax / fp8_max).clamp(min=torch.finfo(work_dtype).tiny)
    quantized = (x_work / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (quantized.to(work_dtype) * scale).to(x.dtype)


def _assert_tight_fp8_ref_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    norm_steps: int,
    tp_size: int,
) -> tuple[float, float]:
    """Bounds parity against an FP8-quantized CPU ref from bf16 accumulation terms."""
    rel_rmse_budget = norm_steps * _BF16_ACCUM_EPS
    max_abs_factor = (norm_steps + tp_size) * _BF16_ACCUM_EPS
    expected_f32 = expected.detach().float()
    assert _relative_rmse(actual, expected_f32) <= rel_rmse_budget
    atol = max_abs_factor * float(expected_f32.abs().max())
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)
    return rel_rmse_budget, atol


def _run_on_tp_threads[T](tp_size: int, fn: Callable[[int], T]) -> list[T]:
    """Run fn(rank) on tp_size threads and propagate any worker exception."""

    def worker(rank: int) -> T:
        """Set thread-local rank and execute rank callback."""
        _THREAD_STATE.rank = rank
        return fn(rank)

    with ThreadPoolExecutor(max_workers=tp_size) as pool:
        futures = [pool.submit(worker, rank) for rank in range(tp_size)]
        done, not_done = wait(futures, timeout=10.0, return_when=FIRST_EXCEPTION)
        for future in done:
            exc = future.exception()
            if exc is not None:
                raise exc
        assert not not_done
        outs = [future.result(timeout=10.0) for future in futures]
    assert len(outs) == tp_size
    return outs


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
            uses_sharded_residual=True,
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
        rank_rows = slice(rank * shard_n, (rank + 1) * shard_n)
        x_rank = x[rank_rows].clone() if sharded_in else x.clone()
        res_rank = (
            None
            if res_in is None
            else (res_in[rank_rows].clone() if sharded_in else res_in.clone())
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
    out_fp8, res_fp8, wire_fp8, expert_dtypes_fp8 = _run_single_layer_with_wire_capture(
        monkeypatch, overlap=True, fp8_all_gather=True
    )
    out_bf16, res_bf16, wire_bf16, expert_dtypes_bf16 = (
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
    assert expert_dtypes_fp8 == [torch.bfloat16]
    assert expert_dtypes_bf16 == [torch.bfloat16]
    stream_fp8 = out_fp8 + res_fp8
    stream_bf16 = out_bf16 + res_bf16
    _assert_fp8_residual_parity(stream_fp8, stream_bf16, fp8_gathers=1, tp_size=2)


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
    out_fp8, res_fp8, wire_fp8, expert_dtypes_fp8 = _run_single_layer_with_wire_capture(
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
    assert expert_dtypes_fp8 == [torch.bfloat16]
    stream_fp8 = out_fp8 + res_fp8
    stream_bf16 = out_bf16 + res_bf16
    _assert_fp8_residual_parity(stream_fp8, stream_bf16, fp8_gathers=2, tp_size=2)


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


@dataclass(frozen=True)
class _TwoLayerWeights:
    """Shared norm, router, attention, and expert weights for 2-layer stack tests."""

    norms: list[_CpuRMSNorm]
    gates: list[nn.Linear]
    attn_projs: list[list[nn.Linear]]
    expert_projs: list[list[nn.Linear]]


class _OverlapDecoderLayerModule(nn.Module):
    """Decoder layer nn.Module delegating forward to _overlap_forward."""

    def __init__(
        self,
        layer_type: str,
        norms: tuple[_CpuRMSNorm, _CpuRMSNorm],
        attn_linear: nn.Linear,
        mlp: _StubMlp,
    ) -> None:
        """Initialize layer modules and overlap flags."""
        super().__init__()
        self.layer_type = layer_type
        self.use_attn_reduce_scatter_for_moe = False
        self.uses_sharded_residual = True
        self.layer_scale = False
        self.input_layernorm, self.post_attention_layernorm = norms
        self.linear_attn = lambda hidden_states: attn_linear(hidden_states)
        self.self_attn = lambda hidden_states, *_args, **_kwargs: attn_linear(
            hidden_states
        )
        self.mlp = mlp

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run _overlap_forward on this decoder layer."""
        return overlap_forward(
            cast("Qwen3_5DecoderLayer", self), hidden_states, residual, positions
        )


def _build_two_layer_weights(
    num_layers: int, tp_size: int, hidden_size: int
) -> _TwoLayerWeights:
    """Create deterministic bf16 weights for a 2-layer TP overlap stack."""
    # Two norms per decoder layer plus one final output norm.
    norms = [
        _CpuRMSNorm(hidden_size).to(torch.bfloat16) for _ in range(2 * num_layers + 1)
    ]
    gates = [
        nn.Linear(hidden_size, 8, bias=False).to(torch.bfloat16)
        for _ in range(num_layers)
    ]
    attn_projs = [
        [
            nn.Linear(hidden_size, hidden_size, bias=False).to(torch.bfloat16)
            for _ in range(tp_size)
        ]
        for _ in range(num_layers)
    ]
    expert_projs = [
        [
            nn.Linear(hidden_size, hidden_size, bias=False).to(torch.bfloat16)
            for _ in range(tp_size)
        ]
        for _ in range(num_layers)
    ]
    return _TwoLayerWeights(
        norms=norms, gates=gates, attn_projs=attn_projs, expert_projs=expert_projs
    )


def _build_two_layer_real_model(
    rank: int,
    tp_size: int,
    weights: _TwoLayerWeights,
    *,
    with_aux: bool,
) -> Qwen3_5Model:
    """Build a real Qwen3_5Model instance via __new__ for a single TP rank."""
    layers: list[nn.Module] = []
    for layer_index, gate_linear in enumerate(weights.gates):
        attn_linear = weights.attn_projs[layer_index][rank]
        expert_linear = weights.expert_projs[layer_index][rank]
        mlp = _StubMlp(
            tp_size=tp_size,
            gate=lambda hidden, gate=gate_linear: (gate(hidden), None),
            experts=lambda hidden_states, *_args, expert=expert_linear, **_kwargs: (
                expert(hidden_states)
            ),
        )
        layers.append(
            _OverlapDecoderLayerModule(
                layer_type=(
                    "linear_attention" if layer_index == 0 else "full_attention"
                ),
                norms=(
                    weights.norms[2 * layer_index],
                    weights.norms[2 * layer_index + 1],
                ),
                attn_linear=attn_linear,
                mlp=mlp,
            )
        )
    aux_layers = (1,) if with_aux else ()
    return _build_real_qwen3_5_model(
        layers=layers,
        norm=weights.norms[-1],
        aux_hidden_state_layers=aux_layers,
    )


def _run_two_layer_fp8_reference(
    inputs: torch.Tensor,
    weights: _TwoLayerWeights,
    *,
    with_aux: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run single-process 2-layer CPU reference with FP8 gather quantization."""
    hidden_states = inputs.clone()
    residual: torch.Tensor | None = None
    aux_outputs: list[torch.Tensor] = []
    for layer_index, (layer_attn_projs, layer_expert_projs) in enumerate(
        zip(weights.attn_projs, weights.expert_projs, strict=True)
    ):
        input_norm = weights.norms[2 * layer_index]
        post_norm = weights.norms[2 * layer_index + 1]
        if residual is None:
            residual = hidden_states
            hidden_states = input_norm(hidden_states)
        else:
            hidden_states, residual = input_norm(hidden_states, residual)
            hidden_states = _fp8_quantize_dequantize_ref(hidden_states)
        attn_sum = torch.stack(
            [proj(hidden_states) for proj in layer_attn_projs], dim=0
        ).sum(dim=0)
        post_normed, residual = post_norm(attn_sum, residual)
        moe_input = _fp8_quantize_dequantize_ref(post_normed)
        hidden_states = torch.stack(
            [proj(moe_input) for proj in layer_expert_projs], dim=0
        ).sum(dim=0)
        if with_aux and layer_index == 0:
            aux_outputs.append(hidden_states + residual)
    final_hidden, _ = weights.norms[-1](hidden_states, residual)
    return final_hidden, aux_outputs


def _extract_matching_rank_outputs(
    res_ranks: list[torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]],
    tp_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Verify every TP rank returned identical outputs and return rank 0's result."""
    assert len(res_ranks) == tp_size
    first = res_ranks[0]
    if isinstance(first, tuple):
        for other in res_ranks[1:]:
            assert isinstance(other, tuple)
            assert torch.equal(other[0], first[0])
            assert len(other[1]) == len(first[1])
            for other_aux, first_aux in zip(other[1], first[1], strict=True):
                assert torch.equal(other_aux, first_aux)
        return first[0], first[1]
    for other in res_ranks[1:]:
        assert isinstance(other, torch.Tensor)
        assert torch.equal(other, first)
    return first, []


def _run_two_layer_real_stack(
    monkeypatch: pytest.MonkeyPatch,
    weights: _TwoLayerWeights,
    inputs: torch.Tensor,
    *,
    fp8_flag: bool,
    with_aux: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Execute the real 2-layer Qwen3_5Model stack across TP threads."""
    tp_size = len(weights.attn_projs[0])
    positions = torch.arange(inputs.shape[0])
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", fp8_flag)
    wire_log: list[tuple[str, torch.Size, torch.dtype]] = []
    _install_logged_collectives(monkeypatch, tp_size, wire_log)

    def run_rank(
        rank: int,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Build and execute real 2-layer Qwen3_5Model on one TP rank."""
        model = _build_two_layer_real_model(rank, tp_size, weights, with_aux=with_aux)
        out = overlap_model_forward(
            model, None, positions, inputs_embeds=inputs.clone()
        )
        assert isinstance(out, torch.Tensor | tuple)
        return out

    res_ranks = _run_on_tp_threads(tp_size, run_rank)
    rank0_ops = [
        f"ag:{dtype}"
        if op == "all_gather"
        else ("rs" if op == "reduce_scatter" else "ar")
        for op, _, dtype in wire_log
    ]
    rank0_out, rank0_aux = _extract_matching_rank_outputs(res_ranks, tp_size)
    return rank0_out, rank0_aux, rank0_ops


@pytest.mark.parametrize("with_aux", [False, True])
def test_overlap_model_forward_uses_sharded_residual_matches_fp8_ref(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_aux: bool,
) -> None:
    """Verify real Qwen3_5Model.forward matches FP8-quantized CPU ref and bf16 stack."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)
    torch.manual_seed(19)
    tp_size = 2
    num_layers = 2
    inputs = torch.randn(16, 64, dtype=torch.bfloat16)
    weights = _build_two_layer_weights(num_layers, tp_size, inputs.shape[-1])

    out_fp8, aux_fp8, ops_fp8 = _run_two_layer_real_stack(
        monkeypatch, weights, inputs, fp8_flag=True, with_aux=with_aux
    )
    out_bf16, aux_bf16, ops_bf16 = _run_two_layer_real_stack(
        monkeypatch, weights, inputs, fp8_flag=False, with_aux=with_aux
    )
    out_ref, aux_ref = _run_two_layer_fp8_reference(inputs, weights, with_aux=with_aux)
    assert "ar" not in ops_fp8
    assert ops_bf16.count("ar") == num_layers
    assert out_fp8.shape == inputs.shape
    # Layer 0 runs 1 pre-MoE gather; each subsequent layer adds 2 gathers.
    stack_fp8_gathers = 2 * num_layers - 1
    stack_rtol, stack_atol = _assert_fp8_residual_parity(
        out_fp8, out_bf16, fp8_gathers=stack_fp8_gathers, tp_size=tp_size
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            torch.zeros_like(out_fp8), out_bf16, rtol=stack_rtol, atol=stack_atol
        )
    norm_steps = 2 * num_layers + 1
    tight_rel_rmse_budget, _ = _assert_tight_fp8_ref_parity(
        out_fp8, out_ref, norm_steps=norm_steps, tp_size=tp_size
    )
    perturbed = (out_fp8 * (1.0 + 2.0 * tight_rel_rmse_budget)).to(out_fp8.dtype)
    _assert_fp8_residual_parity(
        perturbed, out_bf16, fp8_gathers=stack_fp8_gathers, tp_size=tp_size
    )
    with pytest.raises(AssertionError):
        _assert_tight_fp8_ref_parity(
            perturbed, out_ref, norm_steps=norm_steps, tp_size=tp_size
        )
    if with_aux:
        assert len(aux_fp8) == 1
        assert aux_fp8[0].shape == inputs.shape
        _assert_fp8_residual_parity(
            aux_fp8[0], aux_bf16[0], fp8_gathers=1, tp_size=tp_size
        )
        # Aux output is captured after layer 0 (2 norms: input + post-attention).
        aux_norm_steps = 2
        _assert_tight_fp8_ref_parity(
            aux_fp8[0], aux_ref[0], norm_steps=aux_norm_steps, tp_size=tp_size
        )


def _ensure_fake_quantized_matmul_op() -> None:
    """Register a test-only quantized matmul custom op for TorchDynamo trace tests."""
    if _FAKE_QUANTIZED_MATMUL_HOLDER:
        return
    lib = torch.library.Library("tpu_qwen3_5_patch_test", "FRAGMENT")
    lib.define("fake_quantized_matmul(Tensor x, Tensor w) -> Tensor")
    lib.impl("fake_quantized_matmul", lambda x, w: x @ w, "CPU")
    torch.library.register_fake(
        "tpu_qwen3_5_patch_test::fake_quantized_matmul",
        lambda x, w: x.new_empty((*x.shape[:-1], w.shape[1])),
    )
    _FAKE_QUANTIZED_MATMUL_HOLDER.append(lib)


class _FakeLinearAttn(nn.Module):
    """Fake linear attention calling the test-only quantized matmul custom op."""

    def __init__(self, d: int) -> None:
        """Initialize weight parameter."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply fake quantized matmul projection."""
        orig_shape = (*hidden_states.size()[:-1], self.w.shape[1])
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])
        out = torch.ops.tpu_qwen3_5_patch_test.fake_quantized_matmul(x_flat, self.w)
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
        self.use_attn_reduce_scatter_for_moe = False
        self.input_layernorm = _FakeRMSNorm(d)
        self.post_attention_layernorm = _FakeRMSNorm(d)
        self.layer_scale = False
        self.uses_sharded_residual = True
        if layer_type == "linear_attention":
            self.linear_attn: _FakeLinearAttn | None = _FakeLinearAttn(d)
            self.self_attn: _FakeFullAttn | None = None
        else:
            self.linear_attn = None
            self.self_attn = _FakeFullAttn(d)
        self.mlp = _FakeMLP(d)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run _overlap_forward over this fake decoder layer."""
        return overlap_forward(
            cast("Qwen3_5DecoderLayer", self), hidden_states, residual, positions
        )


@pytest.mark.usefixtures("_reset_dynamo_state")
def test_overlap_model_forward_shape_variants_handover_across_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify ShapeEnv unifies dynamic token dims across 2 backbone warmup traces."""
    _ensure_fake_quantized_matmul_op()
    d = 16
    tp_size = 2
    vocab_size = 100
    # First trace stays below the overlap threshold; second trace meets it.
    small_tokens = 8
    large_tokens = 64
    model = _build_real_qwen3_5_model(
        layers=[
            _FakeDecoderLayer(d, "linear_attention"),
            _FakeDecoderLayer(d, "full_attention"),
        ],
        norm=_IdentityModelNorm(),
        embed_tokens=nn.Embedding(vocab_size, d, dtype=torch.bfloat16),
    )

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)
    monkeypatch.setattr(Qwen3_5Model, "forward", overlap_model_forward)
    monkeypatch.setattr(qwen3_5_patch, "_overlap_min_tokens", lambda *_a: large_tokens)
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
        ids_small = torch.randint(0, vocab_size, (small_tokens,), dtype=torch.int64)
        pos_small = torch.arange(small_tokens, dtype=torch.int32)
        mark_dynamic(ids_small, 0)
        mark_dynamic(pos_small, 0)
        out_small = compiled(ids_small, pos_small)
        assert out_small.shape == (small_tokens, d)

        _, _, outputs1 = captured[-1]
        for item in outputs1:
            session.put(
                cache_dir, f"{item.submod_name}_{small_tokens}", item.graph, item.graph
            )

        reset_dynamo()
        compiled2 = torch.compile(
            model, backend=custom_backend, fullgraph=True, dynamic=False
        )
        ids_large = torch.randint(0, vocab_size, (large_tokens,), dtype=torch.int64)
        pos_large = torch.arange(large_tokens, dtype=torch.int32)
        mark_dynamic(ids_large, 0)
        mark_dynamic(pos_large, 0)
        out_large = compiled2(ids_large, pos_large)
        assert out_large.shape == (large_tokens, d)

        _, _, outputs2 = captured[-1]
        for item in outputs2:
            res = shape_variants.runnable_for(
                cache_dir,
                f"{item.submod_name}_{small_tokens}",
                item.graph,
                size=small_tokens,
            )
            assert res is not None


def test_run_on_tp_threads_propagates_worker_exception() -> None:
    """Verify _run_on_tp_threads re-raises exceptions raised inside a worker thread."""
    failing_rank = 1

    def fail_on_rank_one(rank: int) -> int:
        """Raise RuntimeError on rank 1 and return rank otherwise."""
        if rank == failing_rank:
            msg = "tp worker failure"
            raise RuntimeError(msg)
        return rank

    with pytest.raises(RuntimeError, match="tp worker failure"):
        _run_on_tp_threads(2, fail_on_rank_one)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_fp8_gather_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    """Verify _fp8_gather round-trips within analytical bounds and trims padded rows."""
    tp_size = 2
    rows = 8
    hidden = 64
    num_tokens = tp_size * rows - 3
    wire_log: list[tuple[tuple[int, ...], torch.dtype]] = []

    def logging_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Record gathered tensor shape and dtype before concatenating across TP."""
        wire_log.append((tuple(tensor.shape), tensor.dtype))
        return torch.cat([tensor] * tp_size, dim=dim)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", logging_all_gather
    )

    torch.manual_seed(7)
    x = (torch.randn(rows, hidden) * 3.0).to(dtype)
    x[3].zero_()
    out = fp8_gather(x, num_tokens)

    expected_scale_dtype = torch.float32 if dtype == torch.float16 else dtype
    assert wire_log == [
        ((rows, hidden), torch.int8),
        ((rows, 1), expected_scale_dtype),
    ]
    assert out.dtype == dtype
    assert out.shape == (num_tokens, hidden)
    assert not out.isnan().any()
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    assert torch.equal(out[3 + rows], torch.zeros_like(out[3 + rows]))

    ref = torch.cat([x] * tp_size, dim=0).narrow(0, 0, num_tokens)
    rtol, atol = _fp8_round_trip_tol(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test_fp8_gather_large_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify float16-max magnitude bf16 rows round-trip without overflow."""
    tp_size = 2

    def mock_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Concatenate input tensor across simulated TP ranks."""
        return torch.cat([tensor] * tp_size, dim=dim)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", mock_all_gather
    )

    torch.manual_seed(7)
    x = (torch.randn(8, 64) * 3.0).to(torch.bfloat16)
    # Use float16 max (65504.0, stored as 65536.0 in bf16) to test large rows
    # well below bfloat16 dequant overflow.
    x[0].fill_(float(torch.finfo(torch.float16).max))
    out = fp8_gather(x, 16)
    rtol, _ = _fp8_round_trip_tol(x[0])
    torch.testing.assert_close(out[0], x[0], rtol=rtol, atol=0.0)


@pytest.mark.parametrize("magnitude", [1e-4, 1e-5])
def test_fp8_gather_float16_small_magnitude_no_overflow_or_underflow(
    monkeypatch: pytest.MonkeyPatch,
    magnitude: float,
) -> None:
    """Verify small float16 inputs stay finite without scale overflow."""
    tp_size = 2
    wire_dtypes: list[torch.dtype] = []

    def mock_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Record wire dtype and concatenate across simulated TP ranks."""
        wire_dtypes.append(tensor.dtype)
        return torch.cat([tensor] * tp_size, dim=dim)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", mock_all_gather
    )

    torch.manual_seed(7)
    rows = 8
    hidden = 64
    x = ((torch.rand(rows, hidden) * 2.0 - 1.0) * magnitude).to(torch.float16)
    out = fp8_gather(x, tp_size * rows)

    assert wire_dtypes == [torch.int8, torch.float32]
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    ref = torch.cat([x] * tp_size, dim=0)
    rtol, atol = _fp8_round_trip_tol(x)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(out), ref, rtol=rtol, atol=atol)


def test_fp8_gather_negative_control_unscaled_exceeds_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dropping scale values fails the derived FP8 round-trip tolerance."""
    tp_size = 2

    def drop_scale_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Replace gathered bfloat16 scales with ones to simulate scale loss."""
        gathered = torch.cat([tensor] * tp_size, dim=dim)
        if gathered.dtype == torch.bfloat16:
            return torch.ones_like(gathered)
        return gathered

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", drop_scale_all_gather
    )

    torch.manual_seed(7)
    x = (torch.randn(8, 64) * 3.0).to(torch.bfloat16)
    ref = torch.cat([x] * tp_size, dim=0)
    rtol, atol = _fp8_round_trip_tol(x)
    corrupted = fp8_gather(x, tp_size * 8)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(corrupted, ref, rtol=rtol, atol=atol)


def test_fp8_gather_rejects_invalid_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _fp8_gather rejects invalid rank pre-collective."""
    collective_calls = 0

    def counting_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Count TP all-gather calls and return concatenated tensor."""
        nonlocal collective_calls
        collective_calls += 1
        return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", counting_all_gather
    )

    with pytest.raises(ValueError, match="ndim >= 2"):
        fp8_gather(torch.randn(16, dtype=torch.bfloat16), 32)
    assert collective_calls == 0


class _PassthroughDecoderLayer(nn.Module):
    """Passthrough decoder layer with uses_sharded_residual=False for exit tests."""

    def __init__(self, tp_size: int = 2) -> None:
        """Initialize layer attributes required by Qwen3NextModel and exit guard."""
        super().__init__()
        self.use_attn_reduce_scatter_for_moe = False
        self.uses_sharded_residual = False
        self.mlp = _StubMlp(tp_size=tp_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return hidden_states unchanged without running collectives."""
        _ = positions
        return hidden_states, residual


def test_overlap_model_forward_skips_exit_gather_when_not_sharded_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip exit gather when uses_sharded_residual is False."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)

    expected = torch.randn(16, 64, dtype=torch.bfloat16)
    collective_calls = 0

    def counting_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Count TP all-gather calls."""
        nonlocal collective_calls
        collective_calls += 1
        return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_gather", counting_all_gather
    )

    model = _build_real_qwen3_5_model(
        layers=[_PassthroughDecoderLayer(tp_size=2)],
        norm=_IdentityModelNorm(),
    )
    out = overlap_model_forward(
        model, None, torch.arange(expected.shape[0]), inputs_embeds=expected
    )
    assert out is expected
    assert collective_calls == 0
