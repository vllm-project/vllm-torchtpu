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
"""Unit and compilation tests for Qwen token-window micro-batching with fences."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import vllm
from torch._functorch.aot_autograd import aot_module_simplified
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants, tpu_compiler
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.microbatch import (
    FenceMode,
    MicroBatchConfig,
    ScheduleMode,
    can_split_moe_stages,
    fence_windows,
    install_microbatch,
    maybe_install_microbatch,
    microbatch_fence,
    microbatch_fence_pair,
    parse_fence_mode,
    parse_schedule_mode,
    slice_attention_metadata,
    slice_forward_metadata,
    slice_padding_state,
    uninstall_microbatch,
    unsupported_config_reasons,
)

if not hasattr(vllm, "__version__"):
    import importlib.metadata

    vllm.__version__ = importlib.metadata.version("vllm")


def _make_vllm_config(**kwargs) -> VllmConfig:
    """Create a CPU ``VllmConfig`` without requiring TPU hardware in unit tests."""
    from vllm.platforms import current_platform

    orig_check = type(current_platform).check_and_update_config
    type(current_platform).check_and_update_config = classmethod(
        lambda cls, config: None
    )
    try:
        return VllmConfig(device_config=DeviceConfig(device="cpu"), **kwargs)
    finally:
        type(current_platform).check_and_update_config = orig_check


@pytest.fixture(scope="module")
def cpu_vllm_cfg() -> VllmConfig:
    return _make_vllm_config()


# ============================================================================
# Fake / Synthetic Qwen3-Next & Qwen3.5 Building Blocks
# ============================================================================


class _RMSNormStub(nn.Module):
    """Matches vLLM RMSNorm signature: ``norm(x)`` or ``norm(x, residual)``."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.8, 1.2, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            residual = x + residual
            x = residual
        normed = x * self.weight
        if residual is not None:
            return normed, residual
        return normed


class _RowParallelProjStub(nn.Module):
    """Mimics ``RowParallelLinear`` with controllable ``reduce_results``."""

    def __init__(
        self,
        hidden_size: int,
        events: list[str] | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tp_size = 2
        self.reduce_results = True
        self._events = events
        self._layer_idx = layer_idx

    def reduce_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._events is not None:
            self._events.append(f"L{self._layer_idx}:attn_reduce:{x.shape[0]}")
        return x * 0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out = self.linear(x)
        if self._events is not None:
            self._events.append(f"L{self._layer_idx}:attn_compute:{x.shape[0]}")
        if self.reduce_results:
            out = self.reduce_fn(out)
        return out, None


class _LinearAttnStub(nn.Module):
    """Mimics ``VllmGatedDeltaNetAttention`` / ``Qwen3NextGatedDeltaNet``."""

    def __init__(
        self,
        hidden_size: int,
        prefix: str = "layer.0.linear_attn",
        events: list[str] | None = None,
        layer_idx: int = 0,
        recorded_md: list[AttentionMetadata] | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.in_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = _RowParallelProjStub(
            hidden_size,
            events=events,
            layer_idx=layer_idx,
        )
        self._recorded_md = recorded_md

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from vllm import forward_context as _fc_mod

        fc = getattr(_fc_mod, "_forward_context", None)
        md = fc.attn_metadata if fc is not None else None
        if isinstance(md, dict) and self.prefix in md:
            layer_md = md[self.prefix]
            if self._recorded_md is not None:
                self._recorded_md.append(layer_md)
            qsl = layer_md.query_start_loc
            sl = layer_md.seq_lens
            rd = layer_md.request_distribution
            msi = layer_md.mamba_state_indices
            meta_zero = (
                qsl.sum() + sl.sum() + rd.sum() + (msi.sum() if msi is not None else 0)
            ).to(hidden_states.dtype) * 0.0
        else:
            meta_zero = 0.0

        h = torch.tanh(self.in_proj(hidden_states)) + meta_zero
        proj_out, _ = self.out_proj(h)
        return proj_out


class _FullAttnStub(nn.Module):
    """Mimics ``Qwen3NextAttention`` returning ``output: torch.Tensor``."""

    def __init__(
        self,
        hidden_size: int,
        prefix: str = "layer.1.self_attn",
        events: list[str] | None = None,
        layer_idx: int = 1,
        recorded_md: list[AttentionMetadata] | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.qkv_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = _RowParallelProjStub(
            hidden_size,
            events=events,
            layer_idx=layer_idx,
        )
        self._recorded_md = recorded_md

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from vllm import forward_context as _fc_mod

        fc = getattr(_fc_mod, "_forward_context", None)
        md = fc.attn_metadata if fc is not None else None
        if isinstance(md, dict) and self.prefix in md:
            layer_md = md[self.prefix]
            if self._recorded_md is not None:
                self._recorded_md.append(layer_md)
            qsl = layer_md.query_start_loc
            sl = layer_md.seq_lens
            rd = layer_md.request_distribution
            bt = layer_md.block_tables
            meta_zero = (
                qsl.sum() + sl.sum() + rd.sum() + (bt.sum() if bt is not None else 0)
            ).to(hidden_states.dtype) * 0.0
        else:
            meta_zero = 0.0

        pos_1d = positions[0] if positions.ndim == 2 else positions
        pos_scale = pos_1d.to(hidden_states.dtype).unsqueeze(-1) * 0.01
        h = self.qkv_proj(hidden_states) + pos_scale + meta_zero
        proj_out, _ = self.o_proj(h)
        return proj_out


class _FakeQuantMethod:
    _tpu_activation_str = "silu"
    _use_fused_ep = False


class _FakeRoutedExperts(nn.Module):
    """Mimics ``RoutedExperts`` with ``w13_weight``, ``w2_weight``, and scales."""

    def __init__(self, hidden_size: int, num_experts: int = 4, topk: int = 2):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.randn(num_experts, hidden_size, hidden_size) * 0.1
        )
        self.w2_weight = nn.Parameter(
            torch.randn(num_experts, hidden_size, hidden_size) * 0.1
        )
        self.w13_weight_scale_inv = nn.Parameter(
            torch.ones(num_experts, 1, hidden_size),
            requires_grad=False,
        )
        self.w2_weight_scale_inv = nn.Parameter(
            torch.ones(num_experts, 1, hidden_size),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        self._experts_start = torch.tensor(0, dtype=torch.int32)
        self.quant_method = _FakeQuantMethod()
        self.moe_config = SimpleNamespace(
            experts_per_token=topk,
            is_sequence_parallel=False,
            pcp_size=1,
            sp_size=1,
            use_ep=True,
        )
        self.zero_expert_num = 0
        self.top_k = topk

    def _ensure_moe_quant_config_init(self) -> None:
        return None

    def forward_monolithic(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_w, topk_ids = _fake_route(self, x, router_logits)
        if envs.TPU_MOE_SKIP_PADDED_TOKENS:
            topk_ids, topk_w = token_padding.zero_routing_weights_for_padding(
                topk_ids,
                topk_w,
                is_local_tensor=True,
            )
        return _fake_fused_moe_gmm(
            hidden_states=x,
            w1=self.w13_weight,
            w2=self.w2_weight,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_w,
            topk_ids=topk_ids,
            experts_start=self._experts_start,
            topk=self.top_k,
            activation="silu",
        )


class _FakeSharedExpertsWrapper(nn.Module):
    """Mimics vLLM 0.29.0 ``SharedExperts`` wrapper with ``._layer``."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self._layer = layer
        self._output: torch.Tensor | None = None

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: str,
    ) -> None:
        if order == "before":
            self._output = self._layer(shared_experts_input)

    @property
    def output(self) -> torch.Tensor | None:
        return self._output


class _FakeMoERunner(MoERunner):
    """Subclass of ``MoERunner`` matching vLLM 0.29.0 signatures."""

    def __init__(
        self,
        hidden_size: int = 16,
        num_experts: int = 4,
        topk: int = 2,
        shared_expert: nn.Module | None = None,
        events: list[str] | None = None,
        layer_idx: int = 0,
    ):
        # Initialize nn.Module state while bypassing MoERunner.__init__
        nn.Module.__init__(self)
        self.routed_experts = _FakeRoutedExperts(hidden_size, num_experts, topk)
        self._shared_experts = (
            _FakeSharedExpertsWrapper(shared_expert)
            if shared_expert is not None
            else None
        )
        self._gate = None
        self.moe_config = self.routed_experts.moe_config
        self.has_ici_cp_collectives = False
        self.use_dp_chunking = False
        self.zero_expert_result = None
        self._fse_fuse_gate = False
        self._shared_experts_order = "before"
        self._events = events
        self._layer_idx = layer_idx

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, hidden_states

    def _maybe_pad_hidden_states(
        self,
        shared_in: torch.Tensor | None,
        routed_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        return routed_in, shared_in, None

    def _maybe_dispatch(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._events is not None:
            self._events.append(f"L{self._layer_idx}:moe_dispatch:{x.shape[0]}")
        return x, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        routed_out: torch.Tensor,
    ) -> torch.Tensor:
        if self._events is not None:
            self._events.append(f"L{self._layer_idx}:moe_combine:{routed_out.shape[0]}")
        return routed_out

    def _maybe_reduce_routed_output_before_transform(
        self,
        routed_out: torch.Tensor,
        fused_output_is_reduced: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        return routed_out, fused_output_is_reduced

    def _maybe_reduce_shared_expert_output(
        self,
        shared_out: torch.Tensor | None,
        fused_output_is_reduced: bool = False,
    ) -> torch.Tensor | None:
        return shared_out

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_out: torch.Tensor | None,
        routed_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        return shared_out, routed_out * 1.0

    def apply_routed_output_transform(
        self,
        routed_out: torch.Tensor,
    ) -> torch.Tensor:
        return routed_out

    def _maybe_reduce_final_output(
        self,
        output: torch.Tensor,
        shared_output: torch.Tensor | None = None,
        fused_output_is_reduced: bool = False,
    ) -> torch.Tensor:
        if self._events is not None:
            self._events.append(f"L{self._layer_idx}:moe_post_reduce:{output.shape[0]}")
        return output * 0.5

    def _maybe_add_zero_expert_output(
        self,
        final_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_torchtpu.models.microbatch.qwen import _maybe_sp_context

        with _maybe_sp_context(self):
            dx, dlogits = self._maybe_dispatch(hidden_states, router_logits)
            shared_out = (
                self.shared_experts._layer(hidden_states)
                if self.shared_experts is not None
                else None
            )
            topk_w, topk_ids = _fake_route(self.routed_experts, dx, dlogits)
            if envs.TPU_MOE_SKIP_PADDED_TOKENS:
                topk_ids, topk_w = token_padding.zero_routing_weights_for_padding(
                    topk_ids,
                    topk_w,
                    is_local_tensor=True,
                )
            routed_out = _fake_fused_moe_gmm(
                hidden_states=dx,
                w1=self.routed_experts.w13_weight,
                w2=self.routed_experts.w2_weight,
                w1_scale=self.routed_experts.w13_weight_scale_inv,
                w2_scale=self.routed_experts.w2_weight_scale_inv,
                w1_bias=None,
                w2_bias=None,
                topk_weights=topk_w,
                topk_ids=topk_ids,
                experts_start=self.routed_experts._experts_start,
                topk=self.routed_experts.top_k,
                activation="silu",
            )
            routed_out = self._maybe_combine(shared_out, routed_out)
        out = shared_out + routed_out if shared_out is not None else routed_out
        return self._maybe_reduce_final_output(out, None, False)


class _GateStub(nn.Module):
    """Mimics ``ReplicatedLinear`` router gate returning ``(logits, None)``."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        events: list[str] | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        self._events = events
        self._layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self._events is not None:
            self._events.append(f"L{self._layer_idx}:moe_pre_compute:{x.shape[0]}")
        return self.linear(x), None


class _FakeQwenSparseMoeBlock(nn.Module):
    """Matches ``Qwen3NextSparseMoeBlock`` structure and attributes in vLLM 0.29.0."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        topk: int = 2,
        shared_expert_overlap: bool = True,
        events: list[str] | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.n_shared_experts = 1
        self.shared_expert_overlap = shared_expert_overlap
        self.gate = _GateStub(
            hidden_size,
            num_experts,
            events=events,
            layer_idx=layer_idx,
        )
        shared_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        if shared_expert_overlap:
            self.shared_expert = None
            self.shared_expert_gate = None
            inner_shared = shared_mlp
        else:
            self.shared_expert = shared_mlp
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
            inner_shared = None
        # In vLLM 0.29.0, FusedMoEFactory returns a MoERunner directly as self.experts
        self.experts = _FakeMoERunner(
            hidden_size,
            num_experts=num_experts,
            topk=topk,
            shared_expert=inner_shared,
            events=events,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(hidden_states_2d)
        if not self.shared_expert_overlap and self.n_shared_experts > 0:
            shared_out = self.shared_expert(hidden_states_2d)
            if self.shared_expert_gate is not None:
                shared_out = (
                    torch.sigmoid(self.shared_expert_gate(hidden_states_2d))
                    * shared_out
                )
        else:
            shared_out = None
        final_hidden_states = self.experts(hidden_states_2d, router_logits)
        if not self.shared_expert_overlap and shared_out is not None:
            final_hidden_states = final_hidden_states + shared_out
        return final_hidden_states.view(orig_shape)


class _FakeQwenDecoderLayer(nn.Module):
    """Matches ``Qwen3NextDecoderLayer`` / ``Qwen3_5DecoderLayer`` structure."""

    def __init__(
        self,
        hidden_size: int,
        layer_type: str = "linear_attention",
        shared_expert_overlap: bool = True,
        layer_scale: bool = False,
        layer_idx: int = 0,
        events: list[str] | None = None,
        recorded_md: list[AttentionMetadata] | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.layer_scale = layer_scale
        if layer_type == "linear_attention":
            self.linear_attn = _LinearAttnStub(
                hidden_size,
                prefix=f"layer.{layer_idx}.linear_attn",
                events=events,
                layer_idx=layer_idx,
                recorded_md=recorded_md,
            )
        else:
            self.self_attn = _FullAttnStub(
                hidden_size,
                prefix=f"layer.{layer_idx}.self_attn",
                events=events,
                layer_idx=layer_idx,
                recorded_md=recorded_md,
            )
        self.mlp = _FakeQwenSparseMoeBlock(
            hidden_size,
            shared_expert_overlap=shared_expert_overlap,
            events=events,
            layer_idx=layer_idx,
        )
        self.input_layernorm = _RMSNormStub(hidden_size)
        self.post_attention_layernorm = _RMSNormStub(hidden_size)
        if layer_scale:
            self.attn_layer_scale = nn.Parameter(torch.full((1, 1, hidden_size), -0.1))
            self.ffn_layer_scale = nn.Parameter(torch.full((1, 1, hidden_size), 0.1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )
        return hidden_states, residual


class _FakeQwenModel(nn.Module):
    """Matches ``Qwen3NextModel`` / ``Qwen3_5Model`` structure."""

    def __init__(
        self,
        hidden_size: int = 16,
        shared_expert_overlap: bool = True,
        layer_scale: bool = False,
        events: list[str] | None = None,
        recorded_md: list[AttentionMetadata] | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, hidden_size)
        self.layers = nn.ModuleList(
            [
                _FakeQwenDecoderLayer(
                    hidden_size,
                    layer_type="linear_attention",
                    shared_expert_overlap=shared_expert_overlap,
                    layer_scale=layer_scale,
                    layer_idx=0,
                    events=events,
                    recorded_md=recorded_md,
                ),
                _FakeQwenDecoderLayer(
                    hidden_size,
                    layer_type="full_attention",
                    shared_expert_overlap=shared_expert_overlap,
                    layer_scale=layer_scale,
                    layer_idx=1,
                    events=events,
                    recorded_md=recorded_md,
                ),
            ]
        )
        self.start_layer = 0
        self.end_layer = len(self.layers)
        self.norm = _RMSNormStub(hidden_size)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _fake_route(
    layer: _FakeRoutedExperts,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(router_logits + layer.e_score_correction_bias, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, k=layer.top_k, dim=-1)
    return topk_weights.to(x.dtype), topk_ids.to(torch.int32)


gmm_call_shapes: list[int] = []


def _fake_fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> torch.Tensor:
    """Deterministic per-token MoE computation for numerical equivalence tests."""
    gmm_call_shapes.append(int(hidden_states.shape[0]))
    # Compute weighted combination across selected experts per token (row-independent)
    h1 = torch.einsum("td,edh->teh", hidden_states, w1)
    h1 = torch.nn.functional.silu(h1)
    h2 = torch.einsum("teh,ehd->ted", h1, w2)
    idx = topk_ids.long().unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    selected = torch.gather(h2, dim=1, index=idx)
    return (selected * topk_weights.unsqueeze(-1)).sum(dim=1)


# ============================================================================
# Config & Metadata Slicing Tests
# ============================================================================


def test_config_and_mode_parsing(monkeypatch):
    assert parse_fence_mode(None) == FenceMode.SUBSTAGE
    assert parse_fence_mode("none") == FenceMode.NONE
    assert parse_fence_mode("WINDOW") == FenceMode.WINDOW
    assert parse_fence_mode("stage") == FenceMode.STAGE
    assert parse_fence_mode("substage") == FenceMode.SUBSTAGE
    with pytest.raises(ValueError, match="Invalid microbatch fence mode"):
        parse_fence_mode("invalid")

    assert parse_schedule_mode("", layerwise=False) == ScheduleMode.STAGEWISE
    assert parse_schedule_mode("", layerwise=True) == ScheduleMode.LAYERWISE
    assert parse_schedule_mode("staggered", layerwise=True) == ScheduleMode.STAGGERED
    assert parse_schedule_mode("attn_only") == ScheduleMode.ATTN_ONLY
    with pytest.raises(ValueError, match="Invalid microbatch schedule"):
        parse_schedule_mode("bogus")

    monkeypatch.setenv("TPU_MICROBATCH_FACTOR", "4")
    monkeypatch.setenv("TPU_MICROBATCH_MIN_TOKENS", "512")
    monkeypatch.setenv("TPU_MICROBATCH_MERGE_MOE", "0")
    monkeypatch.setenv("TPU_MICROBATCH_LAYERWISE", "1")
    monkeypatch.setenv("TPU_MICROBATCH_SCHEDULE", "")
    monkeypatch.setenv("TPU_MICROBATCH_FENCE", "stage")
    cfg = MicroBatchConfig.from_envs()
    assert cfg.factor == 4
    assert cfg.min_tokens == 512
    assert cfg.merge_moe is False
    assert cfg.resolved_schedule == ScheduleMode.LAYERWISE
    assert cfg.resolved_fence == FenceMode.STAGE


def test_slice_attention_metadata_and_padding_state():
    # 3 sequences in a 16-token step:
    # - seq 0: 1 decode token at [0, 1), total seq_len = 50
    # - seq 1: 10 prefill tokens at [1, 11), total seq_len = 100
    # - seq 2: 5 prefill tokens at [11, 16), total seq_len = 25
    # - padded slot 3: [16, 16), seq_len = 0
    qsl = torch.tensor([0, 1, 11, 16, 16], dtype=torch.int32)
    seq_lens = torch.tensor([50, 100, 25, 0], dtype=torch.int32)
    pos_2d = torch.arange(48, dtype=torch.int32).reshape(3, 16)
    md = AttentionMetadata(
        input_positions=pos_2d,
        block_tables=torch.arange(16, dtype=torch.int32),
        seq_lens=seq_lens,
        query_start_loc=qsl,
        request_distribution=torch.tensor([1, 1, 3], dtype=torch.int32),
    )

    # Window 0 covers tokens [0, 8):
    # - seq 0 has 1 token [0, 1), 0 tokens in Window 1 -> seq_len = 50
    # - seq 1 has 7 tokens [1, 8), 3 tokens in Window 1 -> seq_len = 100 - 3 = 97
    # - seq 2 has 0 tokens in Window 0 -> seq_len = 0
    w0 = slice_attention_metadata(md, window_idx=0, window_tokens=8)
    assert torch.equal(
        w0.query_start_loc,
        torch.tensor([0, 1, 8, 8, 8], dtype=torch.int32),
    )
    assert torch.equal(
        w0.seq_lens,
        torch.tensor([50, 97, 0, 0], dtype=torch.int32),
    )
    assert torch.equal(w0.input_positions, pos_2d[:, :8])

    # Window 1 covers tokens [8, 16):
    # - seq 0 has 0 tokens in Window 1 -> seq_len = 0
    # - seq 1 has 3 tokens [0, 3), 0 tokens after -> seq_len = 100
    # - seq 2 has 5 tokens [3, 8), 0 tokens after -> seq_len = 25
    w1 = slice_attention_metadata(md, window_idx=1, window_tokens=8)
    assert torch.equal(
        w1.query_start_loc,
        torch.tensor([0, 0, 3, 8, 8], dtype=torch.int32),
    )
    assert torch.equal(
        w1.seq_lens,
        torch.tensor([0, 100, 25, 0], dtype=torch.int32),
    )
    assert torch.equal(w1.input_positions, pos_2d[:, 8:16])

    # Deduplication across layers sharing the same AttentionMetadata object
    shared_dict = {"layer.0": md, "layer.1": md}
    sliced_dict = slice_forward_metadata(shared_dict, 0, 8)
    assert sliced_dict["layer.0"] is sliced_dict["layer.1"]

    # TokenPaddingState slicing
    pad = token_padding.TokenPaddingState.create(16, device="cpu")
    pad.update(local_valid_tokens=11, local_padded_tokens=16)
    p0 = slice_padding_state(pad, 0, 8)
    p1 = slice_padding_state(pad, 1, 8)
    assert p0 is not None and p1 is not None
    assert not p0.local_padding_mask.any()
    assert torch.equal(
        p1.local_padding_mask,
        torch.tensor([False, False, False, True, True, True, True, True]),
    )


# ============================================================================
# Numerical Equivalence & Kernel/Sub-Stage Execution Order Tests
# ============================================================================


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("schedule", ["stagewise", "layerwise", "staggered", "attn_only"])
@pytest.mark.parametrize("merge_moe", [True, False])
@pytest.mark.parametrize("fence", ["none", "window", "stage", "substage"])
@pytest.mark.parametrize("shared_expert_overlap", [True, False])
@pytest.mark.parametrize("layer_scale", [False, True])
def test_qwen_microbatch_numerical_equivalence_and_moe_call_counts(
    monkeypatch,
    cpu_vllm_cfg: VllmConfig,
    factor: int,
    schedule: str,
    merge_moe: bool,
    fence: str,
    shared_expert_overlap: bool,
    layer_scale: bool,
):
    torch.manual_seed(42)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.moe_routing.route",
        _fake_route,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.microbatch.qwen.fused_moe_gmm",
        _fake_fused_moe_gmm,
    )
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")

    num_tokens = 16
    hidden_size = 16
    pad_state = token_padding.TokenPaddingState.create(num_tokens, device="cpu")
    pad_state.update(local_valid_tokens=12, local_padded_tokens=num_tokens)
    token_padding.set_padding_state(pad_state)

    try:
        recorded_md: list[AttentionMetadata] = []
        model = _FakeQwenModel(
            hidden_size=hidden_size,
            shared_expert_overlap=shared_expert_overlap,
            layer_scale=layer_scale,
            recorded_md=recorded_md,
        )
        for layer in model.layers:
            assert can_split_moe_stages(layer.mlp)

        input_ids = torch.randint(0, 64, (num_tokens,))
        positions = torch.arange(num_tokens * 3, dtype=torch.int32).reshape(
            3, num_tokens
        )

        md = {
            "layer.0.linear_attn": AttentionMetadata(
                input_positions=positions,
                seq_lens=torch.tensor([6, 10], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 6, 16], dtype=torch.int32),
                request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
                mamba_state_indices=torch.tensor([1, 2], dtype=torch.int32),
            ),
            "layer.1.self_attn": AttentionMetadata(
                input_positions=positions,
                block_tables=torch.zeros(8, dtype=torch.int32),
                seq_lens=torch.tensor([6, 10], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 6, 16], dtype=torch.int32),
                request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
            ),
        }
        vllm_cfg = cpu_vllm_cfg

        # Baseline unsplit output
        gmm_call_shapes.clear()
        with set_forward_context(md, vllm_cfg):
            ref_out = model(input_ids, positions)
        assert gmm_call_shapes == [num_tokens, num_tokens]

        # Instrumented micro-batched output
        cfg = MicroBatchConfig(
            factor=factor,
            min_tokens=8,
            merge_moe=merge_moe,
            schedule=schedule,
            fence=fence,
        )
        assert install_microbatch(model, cfg)
        gmm_call_shapes.clear()
        recorded_md.clear()
        with set_forward_context(md, vllm_cfg):
            mb_out = model(input_ids, positions)

        torch.testing.assert_close(mb_out, ref_out, rtol=1e-5, atol=1e-5)

        # Verify MoE kernel call count and batch shapes
        num_layers = len(model.layers)
        window_tokens = num_tokens // factor
        if schedule == "attn_only" or (schedule != "layerwise" and merge_moe):
            assert gmm_call_shapes == [num_tokens] * num_layers
        else:
            assert gmm_call_shapes == [window_tokens] * (num_layers * factor)

        # Verify each window received properly sliced AttentionMetadata
        assert len(recorded_md) == num_layers * factor
        for w_md in recorded_md:
            assert int(w_md.query_start_loc[-1].item()) == window_tokens

        # Small batch (< min_tokens) falls back to unsplit execution
        gmm_call_shapes.clear()
        small_ids = input_ids[:4]
        small_pos = positions[:, :4]
        with set_forward_context(md, vllm_cfg):
            small_out = model(small_ids, small_pos)
        assert small_out.shape == (4, hidden_size)
        assert gmm_call_shapes == [4, 4]

        uninstall_microbatch(model)
        gmm_call_shapes.clear()
        with set_forward_context(md, vllm_cfg):
            restored_out = model(input_ids, positions)
        torch.testing.assert_close(restored_out, ref_out, rtol=1e-5, atol=1e-5)
        assert gmm_call_shapes == [num_tokens, num_tokens]
    finally:
        token_padding.set_padding_state(None)


def test_staggered_substage_interleaving_order(
    monkeypatch,
    cpu_vllm_cfg: VllmConfig,
):
    """Verify compute and collective sub-stages interleave across windows."""
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.moe_routing.route",
        _fake_route,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.microbatch.qwen.fused_moe_gmm",
        _fake_fused_moe_gmm,
    )

    events: list[str] = []
    model = _FakeQwenModel(hidden_size=8, events=events)
    cfg = MicroBatchConfig(
        factor=2,
        min_tokens=8,
        merge_moe=True,
        schedule="staggered",
        fence="substage",
    )
    install_microbatch(model, cfg)

    input_ids = torch.zeros(16, dtype=torch.long)
    positions = torch.arange(16, dtype=torch.int32)
    with set_forward_context(None, cpu_vllm_cfg):
        _ = model(input_ids, positions)

    # For Layer 0 (N=2, window_tokens=8, merge_moe=True):
    # 1. w0 attn_compute
    # 2. w0 attn_reduce || w1 attn_compute
    # 3. w1 attn_reduce || w0 moe_pre_compute
    # 4. w0 moe_dispatch || w1 moe_pre_compute
    # 5. w1 moe_dispatch
    # 6. w0 moe_combine + moe_post_reduce, w1 moe_combine + moe_post_reduce
    layer0_events = [e for e in events if e.startswith("L0:")]
    assert layer0_events == [
        "L0:attn_compute:8",
        "L0:attn_reduce:8",
        "L0:attn_compute:8",
        "L0:attn_reduce:8",
        "L0:moe_pre_compute:8",
        "L0:moe_dispatch:8",
        "L0:moe_pre_compute:8",
        "L0:moe_dispatch:8",
        "L0:moe_combine:8",
        "L0:moe_post_reduce:8",
        "L0:moe_combine:8",
        "L0:moe_post_reduce:8",
    ]


# ============================================================================
# Fence Custom-Op & FX Graph Verification Tests
# ============================================================================


def test_fence_custom_ops_and_fx_graph_emission(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.moe_routing.route",
        _fake_route,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.microbatch.qwen.fused_moe_gmm",
        _fake_fused_moe_gmm,
    )

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    assert torch.equal(microbatch_fence(x), x)
    x2, y2 = microbatch_fence_pair(x, y)
    assert torch.equal(x2, x) and torch.equal(y2, y)
    assert len(fence_windows([x, y], FenceMode.STAGE)) == 2

    def _count_fence_ops_for_mode(fence_mode: str) -> tuple[int, int]:
        torch._dynamo.reset()
        model = _FakeQwenModel(hidden_size=8)
        install_microbatch(
            model,
            MicroBatchConfig(
                factor=2,
                min_tokens=8,
                merge_moe=True,
                schedule="stagewise",
                fence=fence_mode,
            ),
        )
        captured_ops: list[str] = []

        def _inspect_backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    captured_ops.append(str(node.target))
            return gm.forward

        compiled = torch.compile(model, backend=_inspect_backend, fullgraph=True)
        compiled(
            torch.zeros(16, dtype=torch.long),
            torch.arange(16, dtype=torch.int32),
        )
        single_cnt = sum(
            "microbatch_fence" in t and "microbatch_fence_pair" not in t
            for t in captured_ops
        )
        pair_cnt = sum("microbatch_fence_pair" in t for t in captured_ops)
        return single_cnt, pair_cnt

    none_single, none_pair = _count_fence_ops_for_mode("none")
    assert none_single == 0 and none_pair == 0

    win_single, win_pair = _count_fence_ops_for_mode("window")
    assert win_single > 0 and win_pair == 0

    stage_single, stage_pair = _count_fence_ops_for_mode("stage")
    assert stage_pair > 0

    sub_single, sub_pair = _count_fence_ops_for_mode("substage")
    assert sub_pair > stage_pair


# ============================================================================
# End-to-End Compilation with shape_variants & aot_module_simplified
# ============================================================================


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    enable_if=lambda vllm_config: True,
)
class _CompilableQwenModel(_FakeQwenModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(hidden_size=8)


def test_shape_variants_and_aot_compilation_concrete_shapes(monkeypatch, tmp_path):
    """Verify buckets [16, 2048, 4096] compile with concrete int shapes."""
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.moe_routing.route",
        _fake_route,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.microbatch.qwen.fused_moe_gmm",
        _fake_fused_moe_gmm,
    )
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    vllm_cfg = _make_vllm_config(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=4096,
            max_model_len=4096,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="",
            compile_sizes=[16, 2048, 4096],
            splitting_ops=[],
            cudagraph_capture_sizes=[],
        ),
    )
    vllm_cfg.parallel_config.data_parallel_size = 4
    vllm_cfg.parallel_config.is_moe_model = True

    for field in vllm_cfg.compilation_config.pass_config.__dataclass_fields__.values():
        if isinstance(
            getattr(vllm_cfg.compilation_config.pass_config, field.name), bool
        ):
            setattr(vllm_cfg.compilation_config.pass_config, field.name, False)
    type(vllm_cfg.compilation_config).get_compile_ranges = lambda self: []

    monkeypatch.setattr(
        tpu_compiler,
        "compute_tpu_compilation_hash",
        lambda config: "test_mb_hash",
    )
    monkeypatch.setattr(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform.set_additional_forward_context",
        classmethod(lambda cls, *a, **kw: {}),
    )
    vllm_torchtpu._patch_vllm_aot_compile_cache_key()
    vllm_torchtpu._patch_vllm_piecewise_backend()
    vllm_torchtpu._patch_vllm_reset_compile_wrapper()

    compiled_buckets: list[tuple[int, int]] = []

    class _AotInspectingBackend:
        def __call__(self, gm, example_inputs):
            rows = next(
                (int(x) for x in example_inputs if type(x) is int),
                next(
                    int(t.shape[0])
                    for t in example_inputs
                    if isinstance(t, torch.Tensor) and t.dtype == torch.long
                ),
            )

            def _fw_compiler(fw_gm, fw_inputs):
                for node in fw_gm.graph.nodes:
                    if node.op == "call_function" and "view" in str(node.target):
                        shape_arg = node.args[1]
                        if isinstance(shape_arg, (tuple, list)):
                            for dim in shape_arg:
                                assert isinstance(dim, int), (
                                    f"Expected int in view shape: {type(dim)} {dim}"
                                )
                fences = sum(
                    1 for n in fw_gm.graph.nodes if "microbatch_fence" in str(n.target)
                )
                compiled_buckets.append((rows, fences))
                return fw_gm.forward

            runnable = aot_module_simplified(gm, example_inputs, _fw_compiler)

            class _SerializableRunnable:
                def __call__(self, *args):
                    return runnable(*args)

                def serialize(self):
                    return self

            return _SerializableRunnable()

    monkeypatch.setattr(tpu_compiler, "_tpu_backend", _AotInspectingBackend())
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda entry: entry)

    pad_state = token_padding.TokenPaddingState.create(4096, device="cpu")
    token_padding.set_padding_state(pad_state)

    try:
        with set_current_vllm_config(vllm_cfg):
            model = _CompilableQwenModel(vllm_config=vllm_cfg)
        install_microbatch(
            model,
            MicroBatchConfig(
                factor=2,
                min_tokens=1024,
                merge_moe=True,
                schedule="staggered",
                fence="substage",
            ),
        )

        def _make_md(n: int) -> dict[str, AttentionMetadata]:
            qsl = torch.tensor([0, n], dtype=torch.int32)
            rd = torch.tensor([0, 0, 1], dtype=torch.int32)
            sl = torch.tensor([n], dtype=torch.int32)
            bt = torch.zeros(8, dtype=torch.int32)
            msi = torch.zeros(1, dtype=torch.int32)
            pos = torch.arange(n, dtype=torch.int32)
            return {
                "layer.0.linear_attn": AttentionMetadata(
                    input_positions=pos,
                    seq_lens=sl,
                    query_start_loc=qsl,
                    request_distribution=rd,
                    mamba_state_indices=msi,
                ),
                "layer.1.self_attn": AttentionMetadata(
                    input_positions=pos,
                    block_tables=bt,
                    seq_lens=sl,
                    query_start_loc=qsl,
                    request_distribution=rd,
                ),
            }

        with set_current_vllm_config(vllm_cfg), shape_variants.warmup() as buckets:
            for sz in [16, 2048, 4096]:
                if buckets.needs_retrace(sz):
                    shape_variants.retrace(model, vllm_cfg)
                with set_forward_context(
                    _make_md(sz),
                    vllm_cfg,
                    num_tokens=sz,
                    num_tokens_across_dp=torch.tensor(
                        [sz, sz, sz, sz], dtype=torch.int32
                    ),
                ):
                    out = model(
                        torch.zeros(sz, dtype=torch.long),
                        torch.arange(sz, dtype=torch.int32),
                    )
                    assert out.shape == (sz, 8)
            assert not buckets.refused

        bucket_dict = dict(compiled_buckets)
        assert set(bucket_dict.keys()) == {16, 2048, 4096}
        # Bucket 16 (< 1024) runs unsplit (0 fences); >= 1024 run microbatched
        assert bucket_dict[16] == 0
        assert bucket_dict[2048] > 0
        assert bucket_dict[4096] > 0
    finally:
        token_padding.set_padding_state(None)


def test_unsupported_config_reasons(monkeypatch, cpu_vllm_cfg: VllmConfig):
    model = _FakeQwenModel(hidden_size=8)
    monkeypatch.setattr(
        cpu_vllm_cfg.parallel_config,
        "prefill_context_parallel_size",
        2,
    )
    reasons = unsupported_config_reasons(model, cpu_vllm_cfg)
    assert "prefill_context_parallel_size > 1" in reasons
    assert not maybe_install_microbatch(
        model,
        cpu_vllm_cfg,
        MicroBatchConfig(factor=2),
    )

    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "1")
    # When TP > 1 (default in _FakeQwenModel out_proj/o_proj), fused EP MoE is supported
    assert unsupported_config_reasons(model) == []
    # When TP == 1 (e.g., DP8_EP with no TP collectives), fused EP MoE has no collectives to overlap
    for layer in model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.out_proj.tp_size = 1
        if hasattr(layer, "self_attn"):
            layer.self_attn.o_proj.tp_size = 1
    assert "USE_MOE_FUSED_EP_KERNEL=1" in unsupported_config_reasons(model)


@pytest.mark.parametrize("schedule", ["", "staggered", "stagewise", "attn_only"])
@pytest.mark.parametrize("merge_moe", [True, False])
@pytest.mark.parametrize("fence", ["substage", "stage", "window"])
def test_fused_ep_moe_microbatch_schedules(
    monkeypatch,
    cpu_vllm_cfg: VllmConfig,
    schedule: str,
    merge_moe: bool,
    fence: str,
):
    """Verify fused EP MoE (Design A attn_only, Design B merge_moe=True, and split merge_moe=False)."""
    torch.manual_seed(7)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.moe_routing.route",
        _fake_route,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.models.microbatch.qwen.fused_moe_gmm",
        _fake_fused_moe_gmm,
    )
    monkeypatch.setenv("TPU_MOE_SKIP_PADDED_TOKENS", "1")

    num_tokens = 16
    hidden_size = 16
    pad_state = token_padding.TokenPaddingState.create(num_tokens, device="cpu")
    pad_state.update(local_valid_tokens=12, local_padded_tokens=num_tokens)
    token_padding.set_padding_state(pad_state)

    try:
        model = _FakeQwenModel(hidden_size=hidden_size)
        for layer in model.layers:
            layer.mlp.experts.routed_experts.quant_method._use_fused_ep = True

        input_ids = torch.randint(0, 64, (num_tokens,))
        positions = torch.arange(num_tokens, dtype=torch.int32)

        gmm_call_shapes.clear()
        with set_forward_context(None, cpu_vllm_cfg):
            ref_out = model(input_ids, positions)
        assert gmm_call_shapes == [num_tokens, num_tokens]

        cfg = MicroBatchConfig(
            factor=2,
            min_tokens=8,
            merge_moe=merge_moe,
            schedule=schedule,
            fence=fence,
        )
        assert install_microbatch(model, cfg)
        if schedule == "":
            assert model._microbatch_config.resolved_schedule == ScheduleMode.STAGGERED
        gmm_call_shapes.clear()
        with set_forward_context(None, cpu_vllm_cfg):
            mb_out = model(input_ids, positions)

        torch.testing.assert_close(mb_out, ref_out, rtol=1e-5, atol=1e-5)

        num_layers = len(model.layers)
        if schedule == "attn_only" or merge_moe:
            assert gmm_call_shapes == [num_tokens] * num_layers
        else:
            assert gmm_call_shapes == [num_tokens // 2] * (num_layers * 2)
    finally:
        token_padding.set_padding_state(None)


def test_vllm_upstream_signatures_compatibility():
    """Verify real vLLM / vllm_torchtpu class signatures match qwen.py expectations."""
    import inspect

    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextAttention,
        Qwen3NextDecoderLayer,
    )

    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    gdn_params = list(inspect.signature(VllmGatedDeltaNetAttention.forward).parameters)
    assert gdn_params == ["self", "hidden_states"]

    attn_params = list(inspect.signature(Qwen3NextAttention.forward).parameters)
    assert attn_params == ["self", "positions", "hidden_states"]

    layer_params = set(inspect.signature(Qwen3NextDecoderLayer.forward).parameters)
    assert {"self", "positions", "hidden_states", "residual"} <= layer_params

    dispatch_params = list(inspect.signature(MoERunner._maybe_dispatch).parameters)
    assert len(dispatch_params) == 3

    combine_params = list(inspect.signature(MoERunner._maybe_combine).parameters)
    assert len(combine_params) == 3
