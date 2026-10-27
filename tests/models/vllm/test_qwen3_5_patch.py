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
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
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


class _CpuRMSNorm(Qwen3_5RMSNorm):
    """Qwen3_5RMSNorm using PyTorch-native implementation for CPU tests."""

    def dispatch_forward(self, compile_native: bool = False):
        return self.forward_native


class _TPCollectiveMock:
    """Barrier-synchronized in-process collective simulator for TP ranks."""

    def __init__(self, tp_size: int) -> None:
        self.tp_size = tp_size
        self.barrier = threading.Barrier(tp_size)
        self.storage: list[torch.Tensor | None] = [None] * tp_size

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        rank = getattr(threading.current_thread(), "rank", 0)
        self.storage[rank] = tensor
        self.barrier.wait(timeout=10.0)
        total = sum(self.storage)
        assert total is not None
        chunk = total.chunk(self.tp_size, dim=dim)[rank].clone()
        self.barrier.wait(timeout=10.0)
        return chunk

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        rank = getattr(threading.current_thread(), "rank", 0)
        self.storage[rank] = tensor
        self.barrier.wait(timeout=10.0)
        res = torch.cat(self.storage, dim=dim).clone()  # type: ignore[arg-type]
        self.barrier.wait(timeout=10.0)
        return res

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return tensor
        rank = getattr(threading.current_thread(), "rank", 0)
        self.storage[rank] = tensor
        self.barrier.wait(timeout=10.0)
        total = sum(self.storage)
        assert total is not None
        res = total.clone()
        self.barrier.wait(timeout=10.0)
        return res


def test_maybe_patch_waits_for_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies patch registration returns and class method patching."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", False)
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(None) is None

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(None) is False

    cfg_dense = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text")
    )
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_dense) is False

    cfg_moe = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text")
    )
    orig_init = Qwen3_5DecoderLayer.__init__
    had_fwd = "forward" in Qwen3_5DecoderLayer.__dict__
    orig_fwd = Qwen3_5DecoderLayer.forward
    had_model_fwd = "forward" in Qwen3_5Model.__dict__
    orig_model_fwd = Qwen3_5Model.forward
    try:
        assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe) is None
        assert Qwen3_5DecoderLayer.__init__ is qwen3_5_patch._patched_init
        assert Qwen3_5DecoderLayer.forward is qwen3_5_patch._overlap_forward
        assert Qwen3_5Model.forward is qwen3_5_patch._overlap_model_forward
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
    cfg_moe = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text")
    )
    orig_init = Qwen3_5DecoderLayer.__init__
    orig_fwd = Qwen3_5DecoderLayer.forward
    orig_model_fwd = Qwen3_5Model.forward
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe) is None
    assert Qwen3_5DecoderLayer.__init__ is orig_init
    assert Qwen3_5DecoderLayer.forward is orig_fwd
    assert Qwen3_5Model.forward is orig_model_fwd
    assert len(warned) == 1
    assert "TPU_ATTN_MOE_OVERLAP=True" in warned[0]

    warned.clear()
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
    assert qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe) is None
    assert warned == []


def test_maybe_patch_rejects_fused_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError when USE_MOE_FUSED_EP_KERNEL=True."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    cfg_moe = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text")
    )
    with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL=False"):
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe)


def test_maybe_patch_rejects_sequence_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError when TPU_MOE_SEQUENCE_PARALLEL=True, leaving class unpatched."""
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP", True)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    cfg_moe = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text")
    )
    with pytest.raises(ValueError, match="TPU_MOE_SEQUENCE_PARALLEL=False"):
        qwen3_5_patch.maybe_patch_qwen3_5_attn_moe_overlap(cfg_moe)
    assert Qwen3_5DecoderLayer.__init__ is qwen3_5_patch._orig_init
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
    assert (
        qwen3_5_patch._overlap_min_tokens(hidden_size, tp_size, element_size)
        == expected
    )


def test_configure_layer() -> None:
    """Layer configuration sets expected flags and gate."""
    # Full attention
    layer = SimpleNamespace(use_attn_reduce_scatter_for_moe=False)
    proj = SimpleNamespace(reduce_results=True)
    dummy_gate = object()
    moe_config = SimpleNamespace(skip_final_all_reduce=False)
    layer.self_attn = SimpleNamespace(o_proj=proj)
    layer.linear_attn = None
    layer.layer_type = "full_attention"
    layer.mlp = SimpleNamespace(
        tp_size=2,
        replicate_shared_expert=False,
        experts=SimpleNamespace(gate=dummy_gate, moe_config=moe_config),
    )
    qwen3_5_patch._configure_layer(layer)
    assert proj.reduce_results is False
    assert layer.mlp.experts.gate is None
    assert moe_config.skip_final_all_reduce is True

    # Linear attention
    proj_lin = SimpleNamespace(reduce_results=True)
    layer.layer_type = "linear_attention"
    layer.linear_attn = SimpleNamespace(out_proj=proj_lin)
    layer.self_attn = None
    qwen3_5_patch._configure_layer(layer)
    assert proj_lin.reduce_results is False

    # TP=1 raises ValueError
    layer.mlp.tp_size = 1
    with pytest.raises(ValueError, match="TP > 1"):
        qwen3_5_patch._configure_layer(layer)
    layer.mlp.tp_size = 2

    # Sequence-parallel MoE raises ValueError
    layer.use_attn_reduce_scatter_for_moe = True
    with pytest.raises(ValueError, match="sequence-parallel MoE"):
        qwen3_5_patch._configure_layer(layer)
    layer.use_attn_reduce_scatter_for_moe = False

    # Replicate shared expert raises ValueError
    layer.mlp.replicate_shared_expert = True
    with pytest.raises(ValueError, match="replicate_shared_expert"):
        qwen3_5_patch._configure_layer(layer)


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
        return gate_linear(h), None

    attn_projs = [nn.Linear(d, d, bias=False) for _ in range(tp)]
    expert_projs = [nn.Linear(d, d, bias=False) for _ in range(tp)]

    attn_ls = torch.tensor([0.1])
    ffn_ls = torch.tensor([0.2])

    # Upstream layer reference
    up_layer = Qwen3NextDecoderLayer.__new__(Qwen3NextDecoderLayer)
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
        up_layer.self_attn = lambda hidden_states, positions: sum(
            p(hidden_states) for p in attn_projs
        )
        up_layer.linear_attn = None
    up_layer.mlp = lambda h: sum(
        ep(h) * (gate(h)[0].sum(-1, keepdim=True) * 0.01 + 1.0) for ep in expert_projs
    )

    up_out, up_res = Qwen3NextDecoderLayer.forward(
        up_layer, x.clone(), res.clone() if res is not None else None, pos
    )

    # Overlap ranks simulation
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

    rank_outs: list[torch.Tensor | None] = [None] * tp
    rank_residuals: list[torch.Tensor | None] = [None] * tp
    errors: list[Exception | None] = [None] * tp

    def worker(rank: int) -> None:
        try:
            rank_layer = SimpleNamespace()
            rank_layer.sharded_spine = True
            rank_layer.input_layernorm = input_norm
            rank_layer.post_attention_layernorm = post_norm
            rank_layer.layer_type = layer_type
            rank_layer.layer_scale = layer_scale
            if layer_scale:
                rank_layer.attn_layer_scale = attn_ls
                rank_layer.ffn_layer_scale = ffn_ls
            if layer_type == "linear_attention":
                rank_layer.linear_attn = lambda hidden_states: attn_projs[rank](
                    hidden_states
                )
                rank_layer.self_attn = None
            else:
                rank_layer.self_attn = lambda hidden_states, positions: attn_projs[
                    rank
                ](hidden_states)
                rank_layer.linear_attn = None
            rank_layer.mlp = SimpleNamespace(
                tp_size=tp,
                gate=gate,
                experts=lambda hidden_states, router_logits: (
                    expert_projs[rank](hidden_states)
                    * (router_logits.sum(-1, keepdim=True) * 0.01 + 1.0)
                ),
            )

            x_in = x.clone()
            input_residual = res.clone() if res is not None else None
            out, ret_res = qwen3_5_patch._overlap_forward(
                rank_layer, x_in, input_residual, pos
            )
            rank_outs[rank] = out
            rank_residuals[rank] = ret_res
            assert ret_res is not None
            if overlap and not with_residual:
                assert ret_res is x_in
        except Exception as e:
            errors[rank] = e

    threads = []
    for r in range(tp):
        t = threading.Thread(target=worker, args=(r,))
        t.rank = r
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()

    if any(errors):
        raise errors[0]  # type: ignore[misc]

    first_rank_out = rank_outs[0]
    assert first_rank_out is not None
    for r in range(tp):
        out_r = rank_outs[r]
        res_r = rank_residuals[r]
        assert out_r is not None
        assert res_r is not None
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
        lambda x, dim=0: (calls.append("rs"), x)[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: (calls.append("ag"), x)[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: (calls.append("ar"), x)[1],
    )

    d = 16
    layer = SimpleNamespace(
        sharded_spine=True,
        input_layernorm=_CpuRMSNorm(d),
        post_attention_layernorm=_CpuRMSNorm(d),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=lambda hidden_states, positions: hidden_states,
        linear_attn=None,
        mlp=SimpleNamespace(
            tp_size=2,
            gate=lambda h: (torch.zeros((h.shape[0], 4)), None),
            experts=lambda hidden_states, router_logits: hidden_states,
        ),
    )

    x = torch.ones((16, d), dtype=torch.float32)
    out, res = qwen3_5_patch._overlap_forward(layer, x, None, torch.arange(16))
    assert out.shape == (16, d)
    assert res is not None
    assert calls == ["ar", "ar"]

    calls.clear()
    with shape_variants.warmup():
        qwen3_5_patch._overlap_forward(layer, x, None, torch.arange(16))
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
        init_calls.append((vllm_config, layer_type, prefix))
        self.use_attn_reduce_scatter_for_moe = False
        self.layer_type = layer_type
        self.self_attn = SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True))
        self.linear_attn = None
        self.mlp = SimpleNamespace(
            tp_size=2,
            replicate_shared_expert=False,
            experts=SimpleNamespace(
                gate=object(),
                moe_config=SimpleNamespace(skip_final_all_reduce=False),
            ),
        )

    monkeypatch.setattr(qwen3_5_patch, "_orig_init", fake_orig_init)
    dummy_layer = SimpleNamespace()
    qwen3_5_patch._patched_init(
        dummy_layer,  # type: ignore[arg-type]
        "cfg",
        "full_attention",
        prefix="model.layers.0",
    )
    assert init_calls == [("cfg", "full_attention", "model.layers.0")]
    assert dummy_layer.sharded_spine is True
    assert dummy_layer.self_attn.o_proj.reduce_results is False
    assert dummy_layer.mlp.experts.gate is None
    assert dummy_layer.mlp.experts.moe_config.skip_final_all_reduce is True

    dummy_mtp: Any = SimpleNamespace()
    qwen3_5_patch._patched_init(
        dummy_mtp,
        "cfg",
        "full_attention",
        prefix="mtp.layers.0",
    )
    assert dummy_mtp.sharded_spine is False


def test_configure_layer_rejects_pp_when_fp8_overlap_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_configure_layer rejects PP > 1 when FP8 overlap all-gather is enabled."""
    layer: Any = SimpleNamespace(
        layer_type="full_attention",
        use_attn_reduce_scatter_for_moe=False,
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True)),
        mlp=SimpleNamespace(
            tp_size=2,
            replicate_shared_expert=False,
            experts=SimpleNamespace(
                gate="router", moe_config=SimpleNamespace(skip_final_all_reduce=False)
            ),
        ),
    )
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_5_patch, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        qwen3_5_patch, "get_pp_group", lambda: SimpleNamespace(world_size=2)
    )
    with pytest.raises(ValueError, match="pipeline parallelism"):
        qwen3_5_patch._configure_layer(layer)

    monkeypatch.setattr(
        qwen3_5_patch, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    qwen3_5_patch._configure_layer(layer)

    monkeypatch.setattr(
        qwen3_5_patch, "get_pp_group", lambda: SimpleNamespace(world_size=2)
    )
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", False)
    qwen3_5_patch._configure_layer(layer)


def test_overlap_forward_mtp_layer_skips_sharded_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MTP predictor layers (sharded_spine=False) skip FP8 all-gather
    and reduce-scatter.
    """
    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_5_patch, "_MIN_PER_CHIP_SIZE_MB", 0)
    wire_log: list[str] = []
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: (wire_log.append("reduce_scatter"), x[: x.shape[0] // 2])[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: (wire_log.append("all_gather"), torch.cat([x, x], dim=0))[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: (wire_log.append("all_reduce"), x)[1],
    )
    layer = SimpleNamespace(
        sharded_spine=False,
        input_layernorm=_CpuRMSNorm(64),
        post_attention_layernorm=_CpuRMSNorm(64),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=(lambda hidden_states, positions: hidden_states),
        linear_attn=None,
        mlp=SimpleNamespace(
            tp_size=2,
            gate=(lambda h: (h @ torch.randn(64, 4, dtype=h.dtype), None)),
            experts=(lambda hidden_states, router_logits: hidden_states),
        ),
    )
    x = torch.randn(16, 64, dtype=torch.bfloat16)
    pos = torch.arange(16)
    out, res = qwen3_5_patch._overlap_forward(layer, x, None, pos)
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
    d = 4096  # min_tokens = (8 * 1024 * 1024 * 2) // (4096 * 2) = 2048
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: (calls.append("rs"), x[: x.shape[0] // tp_size])[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: (calls.append("ag"), torch.cat([x] * tp_size, dim=0))[1],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: (calls.append("ar"), x)[1],
    )
    norm = _CpuRMSNorm(d).to(torch.bfloat16)
    layer = SimpleNamespace(
        sharded_spine=True,
        input_layernorm=norm,
        post_attention_layernorm=norm,
        layer_type="full_attention",
        layer_scale=False,
        self_attn=lambda hidden_states, positions: hidden_states,
        linear_attn=None,
        mlp=SimpleNamespace(
            tp_size=tp_size,
            gate=lambda h: (
                gate_rows.append(h.shape[0]),
                torch.zeros((h.shape[0], 4), dtype=h.dtype),
                None,
            )[1:],
            experts=lambda hidden_states, router_logits: hidden_states,
        ),
    )

    x_small = torch.ones((2047, d), dtype=torch.bfloat16)
    qwen3_5_patch._overlap_forward(layer, x_small, None, torch.arange(2047))
    assert calls == ["ar", "ar"]
    assert gate_rows == [2047]

    calls.clear()
    gate_rows.clear()
    x_large = torch.ones((2048, d), dtype=torch.bfloat16)
    qwen3_5_patch._overlap_forward(layer, x_large, None, torch.arange(2048))
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
        lambda x, dim=0: x[:4],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=0),  # 8 rows > num_tokens=6
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: x,
    )
    seen_shapes: list[tuple[torch.Size, torch.Size]] = []
    d = 16
    layer = SimpleNamespace(
        sharded_spine=True,
        input_layernorm=_CpuRMSNorm(d),
        post_attention_layernorm=_CpuRMSNorm(d),
        layer_type="full_attention",
        layer_scale=False,
        self_attn=lambda hidden_states, positions: hidden_states,
        linear_attn=None,
        mlp=SimpleNamespace(
            tp_size=2,
            gate=lambda h: (torch.zeros((h.shape[0], 4)), None),
            experts=lambda hidden_states, router_logits: (
                seen_shapes.append((hidden_states.shape, router_logits.shape)),
                hidden_states,
            )[1],
        ),
    )
    x = torch.ones((6, d), dtype=torch.float32)
    out, _ = qwen3_5_patch._overlap_forward(layer, x, None, torch.arange(6))
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
    # Floor elementwise near-zero crossings by stream peak * rel_budget:
    atol = rel_budget * float(exp_f32.abs().max())
    torch.testing.assert_close(actual, expected, rtol=rel_budget, atol=atol)
    return rel_budget, atol


def _run_on_tp_threads[T](tp_size: int, fn: Callable[[int], T]) -> list[T]:
    """Runs fn(rank) on tp_size threads tagged with .rank; re-raises the first error."""
    outs: list[T | None] = [None] * tp_size
    errors: list[Exception | None] = [None] * tp_size

    def worker(rank: int) -> None:
        try:
            outs[rank] = fn(rank)
        except Exception as e:
            errors[rank] = e

    threads = []
    for r in range(tp_size):
        t = threading.Thread(target=worker, args=(r,))
        t.rank = r
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()
    for err in errors:
        if err is not None:
            raise err
    return [o for o in outs if o is not None]


def _run_single_layer_with_wire_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overlap: bool,
    fp8_all_gather: bool,
    tp_size: int = 2,
    d: int = 64,
    with_residual: bool = False,
    layer_type: str = "linear_attention",
    layer_scale: bool = False,
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
    shard_n = n // tp_size
    x = torch.randn(n, d, dtype=torch.bfloat16)
    res_in = torch.randn(n, d, dtype=torch.bfloat16) if with_residual else None
    pos = torch.arange(n)

    input_norm = _CpuRMSNorm(d).to(torch.bfloat16)
    post_norm = _CpuRMSNorm(d).to(torch.bfloat16)
    gate_linear = nn.Linear(d, 8, bias=False).to(torch.bfloat16)

    def gate(h: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gate_linear(h), None

    attn_projs = [
        nn.Linear(d, d, bias=False).to(torch.bfloat16) for _ in range(tp_size)
    ]
    expert_projs = [
        nn.Linear(d, d, bias=False).to(torch.bfloat16) for _ in range(tp_size)
    ]
    attn_ls = torch.tensor([0.1], dtype=torch.bfloat16)
    ffn_ls = torch.tensor([0.2], dtype=torch.bfloat16)

    mock = _TPCollectiveMock(tp_size)
    wire_log: list[tuple[str, torch.Size, torch.dtype]] = []
    experts_dtypes: list[torch.dtype] = []

    def logged_rs(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if getattr(threading.current_thread(), "rank", 0) == 0:
            wire_log.append(("reduce_scatter", tensor.shape, tensor.dtype))
        return mock.reduce_scatter(tensor, dim=dim)

    def logged_ag(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if getattr(threading.current_thread(), "rank", 0) == 0:
            wire_log.append(("all_gather", tensor.shape, tensor.dtype))
        return mock.all_gather(tensor, dim=dim)

    def logged_ar(tensor: torch.Tensor) -> torch.Tensor:
        if getattr(threading.current_thread(), "rank", 0) == 0:
            wire_log.append(("all_reduce", tensor.shape, tensor.dtype))
        return mock.all_reduce(tensor)

    monkeypatch.setattr(
        qwen3_5_patch, "tensor_model_parallel_reduce_scatter", logged_rs
    )
    monkeypatch.setattr(qwen3_5_patch, "tensor_model_parallel_all_gather", logged_ag)
    monkeypatch.setattr(qwen3_5_patch, "tensor_model_parallel_all_reduce", logged_ar)

    def run_rank(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        def experts(
            hidden_states: torch.Tensor, router_logits: torch.Tensor
        ) -> torch.Tensor:
            if rank == 0:
                experts_dtypes.append(hidden_states.dtype)
            return expert_projs[rank](hidden_states) * (
                router_logits.sum(-1, keepdim=True) * 0.01 + 1.0
            )

        rank_layer = SimpleNamespace(
            sharded_spine=True,
            input_layernorm=input_norm,
            post_attention_layernorm=post_norm,
            layer_type=layer_type,
            layer_scale=layer_scale,
            attn_layer_scale=attn_ls,
            ffn_layer_scale=ffn_ls,
            linear_attn=(
                (lambda hidden_states: attn_projs[rank](hidden_states))
                if layer_type == "linear_attention"
                else None
            ),
            self_attn=(
                (lambda hidden_states, positions: attn_projs[rank](hidden_states))
                if layer_type != "linear_attention"
                else None
            ),
            mlp=SimpleNamespace(tp_size=tp_size, gate=gate, experts=experts),
        )
        sharded_in = overlap and fp8_all_gather and with_residual
        x_rank = (
            x[rank * shard_n : (rank + 1) * shard_n].clone()
            if sharded_in
            else x.clone()
        )
        res_rank = (
            res_in[rank * shard_n : (rank + 1) * shard_n].clone()
            if (res_in is not None and sharded_in)
            else (res_in.clone() if res_in is not None else None)
        )
        out, ret_res = qwen3_5_patch._overlap_forward(
            rank_layer,
            x_rank,
            res_rank,
            pos,
        )
        assert ret_res is not None
        return out, ret_res

    pairs = _run_on_tp_threads(tp_size, run_rank)
    if overlap and fp8_all_gather:
        out_full = torch.cat([p[0] for p in pairs], dim=0)
        res_full = torch.cat([p[1] for p in pairs], dim=0)
        return out_full, res_full, wire_log, experts_dtypes
    out0, res0 = pairs[0]
    return out0, res0, wire_log, experts_dtypes


def test_overlap_forward_fp8_flag_uses_fp8_h_gather_and_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 0 with TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER=True ships h in fp8 and
    replaces the post-MoE all-reduce with reduce-scatter.
    """
    out_fp8, res_fp8, wire_fp8, exp_dtypes_fp8 = _run_single_layer_with_wire_capture(
        monkeypatch, overlap=True, fp8_all_gather=True, tp_size=2
    )
    out_bf16, res_bf16, wire_bf16, exp_dtypes_bf16 = (
        _run_single_layer_with_wire_capture(
            monkeypatch, overlap=True, fp8_all_gather=False, tp_size=2
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
    """Mid-stack full_attention layer gathers normed input in fp8 and exits via
    reduce_scatter.
    """
    out_fp8, res_fp8, wire_fp8, exp_dtypes_fp8 = _run_single_layer_with_wire_capture(
        monkeypatch,
        overlap=True,
        fp8_all_gather=True,
        tp_size=2,
        d=1024,
        with_residual=True,
        layer_type="full_attention",
        layer_scale=True,
    )
    out_bf16, res_bf16, _, _ = _run_single_layer_with_wire_capture(
        monkeypatch,
        overlap=True,
        fp8_all_gather=False,
        tp_size=2,
        d=1024,
        with_residual=True,
        layer_type="full_attention",
        layer_scale=True,
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
        monkeypatch, overlap=False, fp8_all_gather=True, tp_size=2
    )
    out_bf16, _, wire_bf16, _ = _run_single_layer_with_wire_capture(
        monkeypatch, overlap=False, fp8_all_gather=False, tp_size=2
    )
    assert [op for op, _, _ in wire_fp8] == ["all_reduce", "all_reduce"]
    assert wire_fp8 == wire_bf16
    assert torch.equal(out_fp8, out_bf16)


@pytest.mark.parametrize("with_aux", [False, True])
def test_overlap_model_forward_sharded_spine_matches_bf16_stack(
    monkeypatch: pytest.MonkeyPatch,
    with_aux: bool,
) -> None:
    """2-layer stack + _overlap_model_forward keeps residual sharded between
    layers, issues zero all_reduces, and gathers output + aux_hidden_states at exit.
    """
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

    def run_stack(fp8_flag: bool) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
        monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", fp8_flag)
        mock = _TPCollectiveMock(tp_size)
        ops_r0: list[str] = []

        def logged_rs(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
            if getattr(threading.current_thread(), "rank", 0) == 0:
                ops_r0.append("rs")
            return mock.reduce_scatter(t, dim=dim)

        def logged_ag(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
            if getattr(threading.current_thread(), "rank", 0) == 0:
                ops_r0.append(f"ag:{t.dtype}")
            return mock.all_gather(t, dim=dim)

        def logged_ar(t: torch.Tensor) -> torch.Tensor:
            if getattr(threading.current_thread(), "rank", 0) == 0:
                ops_r0.append("ar")
            return mock.all_reduce(t)

        monkeypatch.setattr(
            qwen3_5_patch, "tensor_model_parallel_reduce_scatter", logged_rs
        )
        monkeypatch.setattr(
            qwen3_5_patch, "tensor_model_parallel_all_gather", logged_ag
        )
        monkeypatch.setattr(
            qwen3_5_patch, "tensor_model_parallel_all_reduce", logged_ar
        )

        def fake_orig_model_forward(
            model_self: Any,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: Any = None,
            inputs_embeds: torch.Tensor | None = None,
        ) -> Any:
            h = x.clone()
            res: torch.Tensor | None = None
            aux: list[torch.Tensor] = []
            for idx, lyr in enumerate(model_self.layers):
                h, res = qwen3_5_patch._overlap_forward(lyr, h, res, positions)
                assert res is not None
                if with_aux and idx == 0:
                    aux.append(h + res)
            h, _ = norms[4](h, res)
            if with_aux:
                return h, aux
            return h

        monkeypatch.setattr(
            qwen3_5_patch, "_orig_model_forward", fake_orig_model_forward
        )

        def run_rank(rank: int) -> Any:
            layers = []
            for l_idx in range(2):
                g_lin = gates[l_idx]
                a_lin = attn_w[l_idx][rank]
                e_lin = exp_w[l_idx][rank]
                layers.append(
                    SimpleNamespace(
                        sharded_spine=True,
                        input_layernorm=norms[2 * l_idx],
                        post_attention_layernorm=norms[2 * l_idx + 1],
                        layer_type="linear_attention"
                        if l_idx == 0
                        else "full_attention",
                        layer_scale=False,
                        linear_attn=(lambda hidden_states, a=a_lin: a(hidden_states)),
                        self_attn=(
                            lambda hidden_states, positions, a=a_lin: a(hidden_states)
                        ),
                        mlp=SimpleNamespace(
                            tp_size=tp_size,
                            gate=(lambda h, g=g_lin: (g(h), None)),
                            experts=(
                                lambda hidden_states, router_logits, e=e_lin: e(
                                    hidden_states
                                )
                            ),
                        ),
                    )
                )
            model = SimpleNamespace(start_layer=0, layers=layers)
            return qwen3_5_patch._overlap_model_forward(model, None, pos)

        res_ranks = _run_on_tp_threads(tp_size, run_rank)
        if with_aux:
            return res_ranks[0][0], res_ranks[0][1], ops_r0
        return res_ranks[0], [], ops_r0

    out_fp8, aux_fp8, ops_fp8 = run_stack(True)
    out_bf16, aux_bf16, ops_bf16 = run_stack(False)
    assert "ar" not in ops_fp8
    assert ops_bf16.count("ar") == 2
    assert out_fp8.shape == (n, d)
    rtol_s, atol_s = _assert_fp8_spine_parity(
        out_fp8, out_bf16, fp8_gathers=3, tp_size=tp_size
    )
    # Negative control: dropping the residual stream fails the derived bound.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            torch.zeros_like(out_fp8), out_bf16, rtol=rtol_s, atol=atol_s
        )
    if with_aux:
        assert len(aux_fp8) == 1 and aux_fp8[0].shape == (n, d)
        _assert_fp8_spine_parity(
            aux_fp8[0], aux_bf16[0], fp8_gathers=1, tp_size=tp_size
        )


def test_overlap_model_forward_shape_variants_handover_across_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify ShapeEnv unifies dynamic token dims across 2 backbone warmup traces.

    When TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER is enabled, Trace 1 (overlap=False)
    and Trace 2 (overlap=True) compile separate subgraphs with differing token
    shapes. Without an explicit shape equality assertion in layer 0, Trace 1 lifts
    positions.shape[-1] as SymInt at index 11 while Trace 2 defers it to layer 3,
    causing shape_variants.runnable_for to fail with ShapeSpecializationError
    during bucket handover.
    """

    if not hasattr(torch.ops, "vllm_tpu") or not hasattr(
        torch.ops.vllm_tpu, "fake_qmm"
    ):
        _lib = torch.library.Library("vllm_tpu", "FRAGMENT")
        _lib.define("fake_qmm(Tensor x, Tensor w) -> Tensor")
        _lib.impl("fake_qmm", lambda x, w: x @ w, "CPU")
        torch.library.register_fake(
            "vllm_tpu::fake_qmm",
            lambda x, w: x.new_empty(x.shape[:-1] + (w.shape[1],)),
        )

    d = 16
    tp_size = 2

    class _FakeLinearAttn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            orig_shape = hidden_states.size()[:-1] + (self.w.shape[1],)
            x_flat = hidden_states.view(-1, hidden_states.shape[-1])
            out = torch.ops.vllm_tpu.fake_qmm(x_flat, self.w)
            return out.view(orig_shape)

    class _FakeFullAttn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

        def forward(
            self, hidden_states: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            return (hidden_states @ self.w) + positions.unsqueeze(-1).to(
                hidden_states.dtype
            )

    class _FakeMLP(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.tp_size = tp_size
            self.gate_w = nn.Parameter(torch.randn(d, 4, dtype=torch.bfloat16))
            self.exp_w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

        def gate(self, h: torch.Tensor) -> tuple[torch.Tensor, None]:
            return h @ self.gate_w, None

        def experts(
            self, hidden_states: torch.Tensor, router_logits: torch.Tensor
        ) -> torch.Tensor:
            return hidden_states @ self.exp_w

    class _FakeRMSNorm(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d, dtype=torch.bfloat16))

        def forward(
            self, x: torch.Tensor, residual: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
            if residual is not None:
                x = x + residual
                return x * self.weight, x
            return x * self.weight

    class _FakeDecoderLayer(nn.Module):
        def __init__(self, d: int, layer_type: str) -> None:
            super().__init__()
            self.layer_type = layer_type
            self.input_layernorm = _FakeRMSNorm(d)
            self.post_attention_layernorm = _FakeRMSNorm(d)
            self.layer_scale = False
            self.sharded_spine = True
            if layer_type == "linear_attention":
                self.linear_attn = _FakeLinearAttn(d)
                self.self_attn = None
            else:
                self.linear_attn = None
                self.self_attn = _FakeFullAttn(d)
            self.mlp = _FakeMLP(d)

    class _FakeBackbone(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(100, d, dtype=torch.bfloat16)
            self.layers = nn.ModuleList(
                [
                    _FakeDecoderLayer(d, "linear_attention"),
                    _FakeDecoderLayer(d, "full_attention"),
                ]
            )
            self.start_layer = 0

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            hidden_states = self.embed(input_ids)
            residual = None
            for layer in self.layers:
                hidden_states, residual = qwen3_5_patch._overlap_forward(
                    layer, hidden_states, residual, positions
                )
            return hidden_states

    model = _FakeBackbone(d)

    monkeypatch.setattr(envs, "TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER", True)
    monkeypatch.setattr(qwen3_5_patch, "_overlap_min_tokens", lambda *a: 64)
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x[: x.shape[0] // tp_size],
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x] * tp_size, dim=0),
    )
    monkeypatch.setattr(
        qwen3_5_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: x,
    )

    captured = []

    def custom_backend(gm: Any, example_inputs: Any) -> Any:
        split_gm, outputs = split_graph(gm, [])
        captured.append((gm, split_gm, outputs))
        return gm

    compiled = torch.compile(
        model, backend=custom_backend, fullgraph=True, dynamic=False
    )

    cache_dir = str(tmp_path / "aot_cache")
    try:
        with shape_variants.warmup() as session:
            ids8 = torch.randint(0, 100, (8,), dtype=torch.int64)
            pos8 = torch.arange(8, dtype=torch.int32)
            torch._dynamo.mark_dynamic(ids8, 0)
            torch._dynamo.mark_dynamic(pos8, 0)
            compiled(ids8, pos8)

            _, _, outputs1 = captured[-1]
            for item in outputs1:
                session.put(cache_dir, f"{item.submod_name}_8", item.graph, item.graph)

            torch._dynamo.reset()
            compiled2 = torch.compile(
                model, backend=custom_backend, fullgraph=True, dynamic=False
            )
            ids64 = torch.randint(0, 100, (64,), dtype=torch.int64)
            pos64 = torch.arange(64, dtype=torch.int32)
            torch._dynamo.mark_dynamic(ids64, 0)
            torch._dynamo.mark_dynamic(pos64, 0)
            compiled2(ids64, pos64)

            _, _, outputs2 = captured[-1]
            for item in outputs2:
                res = shape_variants.runnable_for(
                    cache_dir, f"{item.submod_name}_8", item.graph, size=8
                )
                assert res is not None
    finally:
        torch._dynamo.reset()
