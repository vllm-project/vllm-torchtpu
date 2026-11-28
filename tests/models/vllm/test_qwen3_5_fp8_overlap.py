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
"""Unit tests for FP8 all-gather and sharded residual in Qwen3.5 MoE overlap."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
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
)

from tests.distributed.test_quantized_collectives import FP8_E4M3_UNIT_ROUNDOFF
from tests.models.vllm.test_qwen3_5_patch import _CpuRMSNorm
from tests.tp_collective_test_utils import TPCollectiveMock
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter import dense_sequence_parallel
from vllm_torchtpu.models.vllm import qwen3_5_patch, qwen3_next_patch
from vllm_torchtpu.models.vllm.qwen3_5_patch import (
    _overlap_forward,
    _overlap_min_tokens,
    _overlap_model_forward,
    _patched_init,
    _patched_model_init,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from vllm.config import VllmConfig


_FAKE_QUANTIZED_MATMUL_HOLDER: list[Any] = []


_SINGLE_RANK_PP_GROUP = SimpleNamespace(is_first_rank=True, is_last_rank=True)


def _make_model_config(
    *,
    model_type: str = "qwen3_5_moe_text",
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
) -> SimpleNamespace:
    """Build a stub ModelConfig carrying hf_text_config and dtype."""
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(
            model_type=model_type,
            hidden_size=hidden_size,
        ),
        dtype=dtype,
    )


def _make_vllm_config(
    model_config: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Build a stub VllmConfig carrying a stub ModelConfig."""
    return SimpleNamespace(
        model_config=_make_model_config() if model_config is None else model_config
    )


def _make_mlp(
    *,
    tp_size: int = 2,
    replicate_shared_expert: bool = False,
    experts: object | None = None,
    gate: object = None,
) -> SimpleNamespace:
    """Build a stub MoE MLP block with experts and router gate."""
    resolved_experts = (
        SimpleNamespace(
            gate=None,
            moe_config=SimpleNamespace(skip_final_all_reduce=False),
        )
        if experts is None
        else experts
    )
    return SimpleNamespace(
        tp_size=tp_size,
        replicate_shared_expert=replicate_shared_expert,
        experts=resolved_experts,
        gate=gate,
    )


def _make_decoder_layer(**overrides: object) -> SimpleNamespace:
    """Build a stub Qwen3.5 decoder layer with optional attribute overrides."""
    defaults: dict[str, object] = {
        "layer_type": "full_attention",
        "use_attn_reduce_scatter_for_moe": False,
        "uses_sharded_residual": False,
        "fp8_overlap_min_tokens": None,
        "layer_scale": False,
        "input_layernorm": None,
        "post_attention_layernorm": None,
        "self_attn": None,
        "linear_attn": None,
        "attn_layer_scale": None,
        "ffn_layer_scale": None,
        "mlp": _make_mlp(),
    }
    unknown = set(overrides) - set(defaults)
    if unknown:
        msg = f"Unexpected decoder layer attributes: {sorted(unknown)}"
        raise TypeError(msg)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_tp_group(all_gather_fn: Callable[..., torch.Tensor]) -> SimpleNamespace:
    """Build a minimal TP group adapter exposing an all_gather callback."""
    return SimpleNamespace(all_gather=all_gather_fn)


class _IdentityModelNorm(nn.Module):
    """Identity final norm matching Qwen3NextModel's (hidden, residual) contract."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return hidden_states and residual unchanged."""
        return hidden_states, residual


def _build_qwen3_5_model_stub(
    layers: list[nn.Module],
    norm: nn.Module,
    *,
    aux_hidden_state_layers: tuple[int, ...] = (),
    embed_tokens: nn.Module | None = None,
) -> Qwen3_5Model:
    """Construct a minimal Qwen3_5Model stub via __new__ for forward-path tests."""
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


def test_maybe_patch_only_wraps_model_forward_when_fp8_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3_5Model.forward is wrapped only when FP8 overlap all-gather is enabled."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    cfg_moe = _make_model_config()
    orig_init = Qwen3_5DecoderLayer.__init__
    had_fwd = "forward" in Qwen3_5DecoderLayer.__dict__
    orig_fwd = Qwen3_5DecoderLayer.forward
    had_model_init = "__init__" in Qwen3_5Model.__dict__
    orig_model_init = Qwen3_5Model.__init__
    had_model_fwd = "forward" in Qwen3_5Model.__dict__
    orig_model_fwd = Qwen3_5Model.forward
    try:
        monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
        assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe) is None
        assert Qwen3_5DecoderLayer.__init__ is _patched_init
        assert Qwen3_5DecoderLayer.forward is _overlap_forward
        assert Qwen3_5Model.__init__ is orig_model_init
        assert Qwen3_5Model.forward is orig_model_fwd

        monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
        assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe) is None
        assert Qwen3_5Model.__init__ is _patched_model_init
        assert Qwen3_5Model.forward is _overlap_model_forward
    finally:
        Qwen3_5DecoderLayer.__init__ = orig_init
        if had_fwd:
            Qwen3_5DecoderLayer.forward = orig_fwd
        elif "forward" in Qwen3_5DecoderLayer.__dict__:
            del Qwen3_5DecoderLayer.forward
        if had_model_init:
            Qwen3_5Model.__init__ = orig_model_init
        elif "__init__" in Qwen3_5Model.__dict__:
            del Qwen3_5Model.__init__
        if had_model_fwd:
            Qwen3_5Model.forward = orig_model_fwd
        elif "forward" in Qwen3_5Model.__dict__:
            del Qwen3_5Model.forward


def test_maybe_patch_rejects_fp8_flag_when_overlap_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FP8 overlap flag without TPU_ATTN_MOE_OVERLAP=True raises ValueError."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", False)
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    cfg_moe = _make_model_config()
    orig_init = Qwen3_5DecoderLayer.__init__
    orig_fwd = Qwen3_5DecoderLayer.forward
    orig_model_init = Qwen3_5Model.__init__
    orig_model_fwd = Qwen3_5Model.forward
    with pytest.raises(ValueError, match="TPU_ATTN_MOE_OVERLAP=True"):
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe)
    assert Qwen3_5DecoderLayer.__init__ is orig_init
    assert Qwen3_5DecoderLayer.forward is orig_fwd
    assert Qwen3_5Model.__init__ is orig_model_init
    assert Qwen3_5Model.forward is orig_model_fwd

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe) is None


def _populate_stub_decoder_layer(layer: Qwen3_5DecoderLayer, layer_type: str) -> None:
    """Populate minimal decoder layer attributes for simulated __init__."""
    layer_obj: Any = layer
    layer_obj.use_attn_reduce_scatter_for_moe = False
    layer_obj.layer_type = layer_type
    layer_obj.self_attn = SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True))
    layer_obj.linear_attn = None
    layer_obj.mlp = _make_mlp(
        experts=SimpleNamespace(
            gate=object(),
            moe_config=SimpleNamespace(skip_final_all_reduce=False),
        ),
    )


def test_patched_init_sets_uses_sharded_residual_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_patched_init defaults uses_sharded_residual to False for every prefix."""

    def recording_orig_init(
        self: Qwen3_5DecoderLayer,
        vllm_config: object,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        """Populate stub layer attributes during simulated __init__."""
        _ = (vllm_config, prefix)
        _populate_stub_decoder_layer(self, layer_type)

    monkeypatch.setattr(qwen3_5_patch, "_orig_init", recording_orig_init)
    vllm_cfg = _make_vllm_config()
    for prefix in ("model.layers.0", "mtp.layers.0"):
        layer = _make_decoder_layer(uses_sharded_residual=True)
        _patched_init(
            layer,
            vllm_cfg,
            "full_attention",
            prefix=prefix,
        )
        assert layer.uses_sharded_residual is False
        assert layer.fp8_overlap_min_tokens is None


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float32])
def test_patched_model_init_marks_backbone_layers_and_logs_threshold(
    monkeypatch: pytest.MonkeyPatch,
    model_dtype: torch.dtype,
) -> None:
    """_patched_model_init marks [start_layer, end_layer) and logs dtype threshold."""
    hidden_size = 4096
    tp_size = 8
    logged: list[str] = []
    monkeypatch.setattr(
        qwen3_5_patch.logger,
        "info",
        lambda msg, *args: logged.append(msg % args if args else msg),
    )

    def fake_orig_model_init(
        self: Qwen3_5Model,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        """Populate stub backbone layers and config."""
        _ = (vllm_config, prefix)
        self_any: Any = self
        self_any.config = SimpleNamespace(
            model_type="qwen3_5_moe_text", hidden_size=hidden_size
        )
        self_any.layers = [
            _make_decoder_layer(mlp=_make_mlp(tp_size=tp_size)) for _ in range(4)
        ]
        self_any.start_layer = 1
        self_any.end_layer = 3

    monkeypatch.setattr(qwen3_5_patch, "_orig_model_init", fake_orig_model_init)
    vllm_cfg = _make_vllm_config(
        _make_model_config(hidden_size=hidden_size, dtype=model_dtype)
    )
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    _patched_model_init(model, vllm_config=cast("VllmConfig", vllm_cfg))
    model_any: Any = model
    expected_min_tokens = _overlap_min_tokens(
        hidden_size, tp_size, model_dtype.itemsize
    )
    dtype_label = str(model_dtype).removeprefix("torch.")
    assert [layer.uses_sharded_residual for layer in model_any.layers] == [
        False,
        True,
        True,
        False,
    ]
    assert [layer.fp8_overlap_min_tokens for layer in model_any.layers] == [
        None,
        expected_min_tokens,
        expected_min_tokens,
        None,
    ]
    assert len(logged) == 1
    assert f"{expected_min_tokens} tokens at {dtype_label}, TP{tp_size}" in logged[0]


def test_first_layer_token_check_runs_only_when_sharded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer-0 token-count check fires only when the FP8 sharded path is active."""
    monkeypatch.setattr(qwen3_5_patch, "tensor_model_parallel_all_reduce", lambda x: x)
    hidden_size = 16
    layer = _make_decoder_layer(
        uses_sharded_residual=False,
        input_layernorm=_CpuRMSNorm(hidden_size),
        post_attention_layernorm=_CpuRMSNorm(hidden_size),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=(lambda hidden_states, *_args, **_kwargs: hidden_states),
        linear_attn=None,
        mlp=_make_mlp(
            tp_size=2,
            gate=(lambda h: (torch.zeros((h.shape[0], 4), dtype=h.dtype), None)),
            experts=(lambda hidden_states, *_args, **_kwargs: hidden_states),
        ),
    )
    x = torch.ones((10, hidden_size), dtype=torch.bfloat16)
    pos = torch.arange(8)

    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 1024)
    out, residual_out = _overlap_forward(layer, x, None, pos)
    assert out.shape == (10, hidden_size)
    assert residual_out is not None

    layer.uses_sharded_residual = True
    layer.fp8_overlap_min_tokens = 0
    with pytest.raises(RuntimeError, match="token counts must match"):
        _overlap_forward(layer, x, None, pos)


_BF16_FINFO = torch.finfo(torch.bfloat16)
_BF16_ACCUM_EPS = float(_BF16_FINFO.eps)
_FP8_BF16_ROUND_RTOL = (1.0 + FP8_E4M3_UNIT_ROUNDOFF) * (
    1.0 + _BF16_ACCUM_EPS / 2.0
) ** 3 - 1.0


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
    """Quantize and dequantize rows in float8_e4m3fn with float32 scales."""
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    x_f32 = x.to(torch.float32)
    row_amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = (row_amax / fp8_max).clamp(min=torch.finfo(torch.float32).tiny)
    quantized = (x_f32 / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (quantized.to(torch.float32) * scale).to(x.dtype)


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


def _run_on_tp_threads[T](
    tp_size: int, fn: Callable[[int], T], mock: TPCollectiveMock
) -> list[T]:
    """Run fn(rank) on tp_size threads and propagate any worker exception."""
    results: dict[int, T] = {}
    errors: list[BaseException | None] = [None] * tp_size

    def worker(rank: int) -> None:
        """Execute fn(rank) and record any raised exception."""
        try:
            mock.rank.value = rank
            results[rank] = fn(rank)
        except BaseException as exc:
            errors[rank] = exc
            mock.barrier.abort()
            raise

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(tp_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
        assert not thread.is_alive()
    for exc in errors:
        if exc is not None:
            raise exc
    outs = [results[rank] for rank in range(tp_size)]
    assert len(outs) == tp_size
    return outs


def _install_logged_collectives(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    wire_log: list[tuple[str, torch.Size, torch.dtype]],
) -> TPCollectiveMock:
    """Install thread-synchronized TP collectives recording rank-0 wire traffic."""
    mock = TPCollectiveMock(tp_size)

    def logged_reduce_scatter(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Log rank-0 reduce-scatter shape and dtype."""
        if mock.rank.value == 0:
            wire_log.append(("reduce_scatter", tensor.shape, tensor.dtype))
        return mock.reduce_scatter(tensor, dim=dim)

    def logged_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Log rank-0 all-gather shape and dtype."""
        if mock.rank.value == 0:
            wire_log.append(("all_gather", tensor.shape, tensor.dtype))
        return mock.all_gather(tensor, dim=dim)

    def logged_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        """Log rank-0 all-reduce shape and dtype."""
        if mock.rank.value == 0:
            wire_log.append(("all_reduce", tensor.shape, tensor.dtype))
        return mock.all_reduce(tensor)

    tp_group = _make_tp_group(logged_all_gather)
    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_reduce_scatter", logged_reduce_scatter
    )
    monkeypatch.setattr(
        dense_sequence_parallel, "tensor_model_parallel_all_gather", logged_all_gather
    )
    monkeypatch.setattr(qwen3_5_patch, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_all_gather", logged_all_gather
    )
    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_all_reduce", logged_all_reduce
    )
    return mock


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
    torch.manual_seed(11)
    n = 16
    shard_n = n // cfg.tp_size
    x = torch.randn(n, cfg.d, dtype=torch.bfloat16)
    residual_in = (
        torch.randn(n, cfg.d, dtype=torch.bfloat16) if cfg.with_residual else None
    )
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
    mock = _install_logged_collectives(monkeypatch, cfg.tp_size, wire_log)
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
        rank_layer = _make_decoder_layer(
            uses_sharded_residual=fp8_all_gather,
            fp8_overlap_min_tokens=(
                _overlap_min_tokens(cfg.d, cfg.tp_size, torch.bfloat16.itemsize)
                if fp8_all_gather
                else None
            ),
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
            mlp=_make_mlp(tp_size=cfg.tp_size, gate=gate, experts=experts),
        )
        rank_rows = slice(rank * shard_n, (rank + 1) * shard_n)
        x_rank = x[rank_rows].clone() if sharded_in else x.clone()
        residual_rank = (
            None
            if residual_in is None
            else (residual_in[rank_rows].clone() if sharded_in else residual_in.clone())
        )
        out, residual_out = _overlap_forward(
            rank_layer,
            x_rank,
            residual_rank,
            pos,
        )
        assert residual_out is not None
        return out, residual_out

    pairs = _run_on_tp_threads(cfg.tp_size, run_rank, mock)
    if overlap and fp8_all_gather:
        out_full = torch.cat([p[0] for p in pairs], dim=0)
        residual_full = torch.cat([p[1] for p in pairs], dim=0)
        return out_full, residual_full, wire_log, experts_dtypes
    out0, residual0 = pairs[0]
    return out0, residual0, wire_log, experts_dtypes


def test_overlap_forward_fp8_flag_uses_fp8_h_gather_and_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 0 with FP8 overlap flag ships h in FP8 and reduce-scatters MoE output."""
    out_fp8, residual_fp8, wire_fp8, expert_dtypes_fp8 = (
        _run_single_layer_with_wire_capture(
            monkeypatch, overlap=True, fp8_all_gather=True
        )
    )
    out_bf16, residual_bf16, wire_bf16, expert_dtypes_bf16 = (
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
        ("all_gather", torch.Size([8, 1]), torch.float32),
        ("reduce_scatter", torch.Size([16, 64]), torch.bfloat16),
    ]
    assert expert_dtypes_fp8 == [torch.bfloat16]
    assert expert_dtypes_bf16 == [torch.bfloat16]
    stream_fp8 = out_fp8 + residual_fp8
    stream_bf16 = out_bf16 + residual_bf16
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
    out_fp8, residual_fp8, wire_fp8, expert_dtypes_fp8 = (
        _run_single_layer_with_wire_capture(
            monkeypatch,
            overlap=True,
            fp8_all_gather=True,
            cfg=mid_cfg,
        )
    )
    out_bf16, residual_bf16, _, _ = _run_single_layer_with_wire_capture(
        monkeypatch,
        overlap=True,
        fp8_all_gather=False,
        cfg=mid_cfg,
    )
    assert wire_fp8 == [
        ("all_gather", torch.Size([8, 1024]), torch.int8),
        ("all_gather", torch.Size([8, 1]), torch.float32),
        ("reduce_scatter", torch.Size([16, 1024]), torch.bfloat16),
        ("all_gather", torch.Size([8, 8]), torch.bfloat16),
        ("all_gather", torch.Size([8, 1024]), torch.int8),
        ("all_gather", torch.Size([8, 1]), torch.float32),
        ("reduce_scatter", torch.Size([16, 1024]), torch.bfloat16),
    ]
    assert expert_dtypes_fp8 == [torch.bfloat16]
    stream_fp8 = out_fp8 + residual_fp8
    stream_bf16 = out_bf16 + residual_bf16
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


class _LinearAttnAdapter(nn.Module):
    """Wrap an nn.Linear projection with attention keyword arguments."""

    def __init__(self, proj: nn.Linear) -> None:
        """Store the underlying linear projection."""
        super().__init__()
        self.proj = proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project hidden_states while ignoring optional positions."""
        _ = positions
        return self.proj(hidden_states)


class _OverlapDecoderLayerModule(nn.Module):
    """Decoder layer nn.Module delegating forward to _overlap_forward."""

    def __init__(
        self,
        layer_type: str,
        norms: tuple[nn.Module, nn.Module],
        attn: nn.Module,
        mlp: nn.Module | SimpleNamespace,
        *,
        fp8_overlap_min_tokens: int | None = None,
    ) -> None:
        """Initialize layer modules and overlap flags."""
        super().__init__()
        self.layer_type = layer_type
        self.use_attn_reduce_scatter_for_moe = False
        self.uses_sharded_residual = fp8_overlap_min_tokens is not None
        self.fp8_overlap_min_tokens = fp8_overlap_min_tokens
        self.layer_scale = False
        self.input_layernorm, self.post_attention_layernorm = norms
        attn_mod = _LinearAttnAdapter(attn) if isinstance(attn, nn.Linear) else attn
        if layer_type == "linear_attention":
            self.linear_attn: nn.Module | None = attn_mod
            self.self_attn: nn.Module | None = None
        else:
            self.linear_attn = None
            self.self_attn = attn_mod
        self.mlp = mlp

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run _overlap_forward on this decoder layer."""
        return _overlap_forward(
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


def _build_two_layer_stub_model(
    rank: int,
    tp_size: int,
    weights: _TwoLayerWeights,
    *,
    uses_sharded_residual: bool,
    aux_layers: tuple[int, ...] = (),
) -> Qwen3_5Model:
    """Build a stub Qwen3_5Model instance via __new__ for a single TP rank."""
    layers: list[nn.Module] = []
    for layer_index, gate_linear in enumerate(weights.gates):
        attn_linear = weights.attn_projs[layer_index][rank]
        expert_linear = weights.expert_projs[layer_index][rank]
        mlp = _make_mlp(
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
                attn=attn_linear,
                mlp=mlp,
                fp8_overlap_min_tokens=(
                    _overlap_min_tokens(
                        weights.norms[0].weight.shape[0],
                        tp_size,
                        torch.bfloat16.itemsize,
                    )
                    if uses_sharded_residual
                    else None
                ),
            )
        )
    return _build_qwen3_5_model_stub(
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
    rank_outputs: list[torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]],
    tp_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Verify every TP rank returned identical outputs and return rank 0's result."""
    assert len(rank_outputs) == tp_size
    first = rank_outputs[0]
    if isinstance(first, tuple):
        for other in rank_outputs[1:]:
            assert isinstance(other, tuple)
            assert torch.equal(other[0], first[0])
            assert len(other[1]) == len(first[1])
            for other_aux, first_aux in zip(other[1], first[1], strict=True):
                assert torch.equal(other_aux, first_aux)
        return first[0], first[1]
    for other in rank_outputs[1:]:
        assert isinstance(other, torch.Tensor)
        assert torch.equal(other, first)
    return first, []


def _run_two_layer_stub_stack(
    monkeypatch: pytest.MonkeyPatch,
    weights: _TwoLayerWeights,
    inputs: torch.Tensor,
    *,
    fp8_flag: bool,
    aux_layers: tuple[int, ...] = (),
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Execute the stub 2-layer Qwen3_5Model stack across TP threads."""
    tp_size = len(weights.attn_projs[0])
    positions = torch.arange(inputs.shape[0])
    effective_aux = aux_layers
    wire_log: list[tuple[str, torch.Size, torch.dtype]] = []
    mock = _install_logged_collectives(monkeypatch, tp_size, wire_log)

    def run_rank(
        rank: int,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Build and execute stub 2-layer Qwen3_5Model on one TP rank."""
        model = _build_two_layer_stub_model(
            rank,
            tp_size,
            weights,
            uses_sharded_residual=fp8_flag,
            aux_layers=effective_aux,
        )
        out = _overlap_model_forward(
            model, None, positions, inputs_embeds=inputs.clone()
        )
        assert isinstance(out, torch.Tensor | tuple)
        return out

    rank_outputs = _run_on_tp_threads(tp_size, run_rank, mock)
    rank0_ops = [
        f"all_gather:{dtype}" if op == "all_gather" else op for op, _, dtype in wire_log
    ]
    rank0_out, rank0_aux = _extract_matching_rank_outputs(rank_outputs, tp_size)
    return rank0_out, rank0_aux, rank0_ops


@pytest.mark.parametrize("with_aux", [False, True])
def test_overlap_model_forward_uses_sharded_residual_matches_fp8_ref(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_aux: bool,
) -> None:
    """Verify Qwen3_5Model.forward matches FP8-quantized CPU ref and bf16 stack."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)
    torch.manual_seed(19)
    tp_size = 2
    num_layers = 2
    inputs = torch.randn(16, 64, dtype=torch.bfloat16)
    weights = _build_two_layer_weights(num_layers, tp_size, inputs.shape[-1])

    out_fp8, aux_fp8, ops_fp8 = _run_two_layer_stub_stack(
        monkeypatch, weights, inputs, fp8_flag=True, aux_layers=(1,) if with_aux else ()
    )
    out_bf16, aux_bf16, ops_bf16 = _run_two_layer_stub_stack(
        monkeypatch,
        weights,
        inputs,
        fp8_flag=False,
        aux_layers=(1,) if with_aux else (),
    )
    out_ref, aux_ref = _run_two_layer_fp8_reference(inputs, weights, with_aux=with_aux)
    assert "all_reduce" not in ops_fp8
    assert ops_bf16.count("all_reduce") == num_layers
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


def test_overlap_model_forward_keeps_layer0_aux_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aux layer 0 keeps full rows while layer 1 aux and output gather at exit."""
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)
    torch.manual_seed(23)
    tp_size = 2
    num_layers = 2
    inputs = torch.randn(16, 64, dtype=torch.bfloat16)
    weights = _build_two_layer_weights(num_layers, tp_size, inputs.shape[-1])

    aux_layers = (0, 1)
    out_fp8, aux_fp8, _ = _run_two_layer_stub_stack(
        monkeypatch, weights, inputs, fp8_flag=True, aux_layers=aux_layers
    )
    out_ref, aux_ref = _run_two_layer_fp8_reference(inputs, weights, with_aux=True)

    assert len(aux_fp8) == len(aux_layers)
    assert torch.equal(aux_fp8[0], inputs)
    assert aux_fp8[1].shape == inputs.shape
    _assert_tight_fp8_ref_parity(aux_fp8[1], aux_ref[0], norm_steps=2, tp_size=tp_size)
    _assert_tight_fp8_ref_parity(
        out_fp8, out_ref, norm_steps=2 * num_layers + 1, tp_size=tp_size
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
    model = _build_qwen3_5_model_stub(
        layers=[
            _OverlapDecoderLayerModule(
                "linear_attention",
                (_FakeRMSNorm(d), _FakeRMSNorm(d)),
                _FakeLinearAttn(d),
                _FakeMLP(d),
                fp8_overlap_min_tokens=large_tokens,
            ),
            _OverlapDecoderLayerModule(
                "full_attention",
                (_FakeRMSNorm(d), _FakeRMSNorm(d)),
                _FakeFullAttn(d),
                _FakeMLP(d),
                fp8_overlap_min_tokens=large_tokens,
            ),
        ],
        norm=_IdentityModelNorm(),
        embed_tokens=nn.Embedding(vocab_size, d, dtype=torch.bfloat16),
    )

    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)
    monkeypatch.setattr(Qwen3_5Model, "forward", _overlap_model_forward)
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, _dim=0: x[: x.shape[0] // tp_size],
    )

    def repeat_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Repeat tensor across simulated TP ranks."""
        return torch.cat([x] * tp_size, dim=dim)

    tp_group = _make_tp_group(repeat_all_gather)
    monkeypatch.setattr(
        dense_sequence_parallel, "tensor_model_parallel_all_gather", repeat_all_gather
    )
    monkeypatch.setattr(qwen3_5_patch, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_all_gather", repeat_all_gather
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
        assert outputs1
        for item in outputs1:
            assert (
                shape_variants.unsupported_reason(
                    shape_variants.trace_shape_env(item.graph), large_tokens
                )
                is not None
            )
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
        assert outputs2
        for item in outputs2:
            assert (
                shape_variants.unsupported_reason(
                    shape_variants.trace_shape_env(item.graph), small_tokens
                )
                is not None
            )
            runnable = shape_variants.runnable_for(
                cache_dir,
                f"{item.submod_name}_{small_tokens}",
                item.graph,
                size=small_tokens,
            )
            assert runnable is not None


class _PassthroughDecoderLayer(nn.Module):
    """Passthrough decoder layer with uses_sharded_residual=False for exit tests."""

    def __init__(self, tp_size: int = 2) -> None:
        """Initialize layer attributes required by Qwen3NextModel and exit guard."""
        super().__init__()
        self.use_attn_reduce_scatter_for_moe = False
        self.uses_sharded_residual = False
        self.mlp = _make_mlp(tp_size=tp_size)

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
    monkeypatch.setattr(qwen3_next, "get_pp_group", lambda: _SINGLE_RANK_PP_GROUP)

    expected = torch.randn(16, 64, dtype=torch.bfloat16)
    collective_calls = 0

    def counting_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Count TP all-gather calls."""
        nonlocal collective_calls
        collective_calls += 1
        return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_all_gather", counting_all_gather
    )

    model = _build_qwen3_5_model_stub(
        layers=[_PassthroughDecoderLayer(tp_size=2)],
        norm=_IdentityModelNorm(),
    )
    out = _overlap_model_forward(
        model, None, torch.arange(expected.shape[0]), inputs_embeds=expected
    )
    assert out is expected
    assert collective_calls == 0
