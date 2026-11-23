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
"""Unit tests for Gemma 4 31B and 26B-A4B TPU model optimizations."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import GeluAndMul

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.core.experimental.prefill_first_scheduler import (
    PrefillFirstScheduler,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import cp_prefill as cp_kernel
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm import gemma4 as gemma4_pkg
from vllm_torchtpu.models.vllm.gemma4 import (
    attention as gemma4_attn,
)
from vllm_torchtpu.models.vllm.gemma4 import (
    cp_prefill as gemma4_cp,
)
from vllm_torchtpu.models.vllm.gemma4 import (
    layers as gemma4_layers,
)
from vllm_torchtpu.models.vllm.gemma4 import (
    routing as gemma4_routing,
)
from vllm_torchtpu.models.vllm.gemma4 import (
    sequence_parallel as gemma4_sp,
)


class _FakeFp8Method(VllmFp8LinearMethodTPU):
    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float8_e4m3fn,
        online_fp8: bool = False,
        block_quant: bool = False,
        w8a16_max_rows: int = 0,
        requant_block_size: int | None = None,
    ) -> None:
        self.online_fp8, self.block_quant, self.w8a16_max_rows = (
            online_fp8,
            block_quant,
            w8a16_max_rows,
        )
        self._linear_quant_config = ("w8a8", dtype, requant_block_size, None)

    def supports_quantized_input(self, layer: Any, num_rows: int) -> bool:
        del layer
        return (
            not (self.online_fp8 or self.block_quant) and num_rows > self.w8a16_max_rows
        )

    def apply_quantized_input(
        self, layer: Any, x_q: torch.Tensor, x_scale: torch.Tensor
    ) -> torch.Tensor:
        del layer
        return x_q.to(torch.bfloat16) * x_scale.to(torch.bfloat16)


def _upstream_gemma4_routing(
    gating_output: torch.Tensor, topk: int, per_expert_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids = torch.topk(
        torch.softmax(gating_output.to(torch.float32), dim=-1), k=topk, dim=-1
    )
    return (weights / weights.sum(dim=-1, keepdim=True)) * per_expert_scale.float()[
        ids
    ], ids.int()


def _make_fake_model(*, moe: bool = True, dp_size: int = 1) -> SimpleNamespace:
    attn = SimpleNamespace(
        hidden_size=2816,
        head_dim=256,
        total_num_heads=16,
        total_num_kv_heads=8,
        is_kv_shared_layer=False,
        qkv_proj=SimpleNamespace(bias=None, tp_size=2),
        o_proj=SimpleNamespace(bias=None, tp_size=2, reduce_results=True),
        rotary_emb=SimpleNamespace(is_neox_style=True, rope_angles=64),
    )
    mlp = SimpleNamespace(
        gate_up_proj=SimpleNamespace(bias=None, tp_size=2),
        down_proj=SimpleNamespace(bias=None, tp_size=2, input_size=2152),
    )
    experts = (
        SimpleNamespace(
            moe_config=SimpleNamespace(dp_size=dp_size, is_sequence_parallel=False)
        )
        if moe
        else None
    )
    layer = SimpleNamespace(
        hidden_size=2816,
        self_attn=attn,
        mlp=mlp,
        enable_moe_block=moe,
        experts=experts,
        per_layer_input_gate=None,
    )
    cfg = SimpleNamespace(
        hidden_size=2816, vocab_size=262144, num_experts=128, moe_intermediate_size=704
    )
    return SimpleNamespace(
        config=cfg,
        layers=[copy.deepcopy(layer) for _ in range(30)],
        start_layer=0,
        end_layer=30,
        fast_prefill_enabled=False,
        embed_tokens=SimpleNamespace(params_dtype=torch.bfloat16),
    )


def _make_parallel_cfg(tp: int = 2, dp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        data_parallel_size=dp,
        enable_ep_weight_filter=False,
        enable_eplb=False,
    )


def _make_vllm_config(**overrides: Any) -> SimpleNamespace:
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(is_multimodal_model=False),
        scheduler_config=SimpleNamespace(scheduler_cls=PrefillFirstScheduler),
        parallel_config=_make_parallel_cfg(),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, block_size=256, gpu_memory_utilization=0.85
        ),
        compilation_config=SimpleNamespace(
            compile_sizes=[16, 8192, 16384, 24576, 32768]
        ),
        kv_transfer_config=None,
        lora_config=None,
        speculative_config=None,
        quant_config=object(),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_maybe_patch_gemma4_skips_non_gemma4_configs() -> None:
    qwen_cfg = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="qwen3_moe"),
        hf_config=SimpleNamespace(model_type="qwen3_moe"),
    )
    assert not gemma4_pkg.is_gemma4_model_config(None)
    assert not gemma4_pkg.is_gemma4_model_config(qwen_cfg)
    assert gemma4_pkg.maybe_patch_gemma4(None) is False
    assert gemma4_pkg.maybe_patch_gemma4(qwen_cfg) is False
    gemma_vlm_cfg = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="gemma4_text"),
        hf_config=SimpleNamespace(model_type="gemma4"),
    )
    assert gemma4_pkg.is_gemma4_model_config(gemma_vlm_cfg) is True


@pytest.mark.parametrize(
    ("experts", "topk", "dtype"), [(128, 8, torch.bfloat16), (16, 4, torch.float32)]
)
def test_topk_softmax_routing_matches_upstream_reference(
    experts: int, topk: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    logits = torch.randn(19, experts, dtype=dtype)
    scales = torch.rand(experts, dtype=dtype) + 0.5
    assert gemma4_routing.uses_router_topk_kernel(logits, topk) is False
    got_w, got_ids = gemma4_routing.gemma4_topk_softmax_routing(logits, topk, scales)
    ref_w, ref_ids = _upstream_gemma4_routing(logits, topk, scales)
    assert got_ids.dtype == torch.int32 and torch.equal(got_ids, ref_ids)
    torch.testing.assert_close(got_w, ref_w, rtol=1e-5, atol=1e-6)


def test_router_topk_kernel_dispatch_invokes_select_scaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.randn(8, 128, dtype=torch.bfloat16)
    scales = torch.full((128,), 2.0, dtype=torch.bfloat16)
    called: list[int] = []

    def _fake_select(scores: torch.Tensor, topk: int, scale: torch.Tensor):
        called.append(topk)
        vals, ids = torch.topk(scores, k=topk, dim=-1)
        return vals, ids.to(torch.int32), scale.to(torch.float32)[ids]

    monkeypatch.setattr(gemma4_routing, "uses_router_topk_kernel", lambda *_: True)
    monkeypatch.setattr(gemma4_routing, "_get_select_scaled", lambda: _fake_select)
    got_w, got_ids = gemma4_routing.gemma4_topk_softmax_routing(logits, 8, scales)
    ref_w, ref_ids = _upstream_gemma4_routing(logits, 8, scales)
    assert called == [8] and torch.equal(got_ids, ref_ids)
    torch.testing.assert_close(got_w, ref_w, rtol=1e-5, atol=1e-6)


def test_should_fuse_gelu_quant_selects_31b_and_skips_26b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        tanh_act, erf_act = (
            GeluAndMul(approximate="tanh"),
            GeluAndMul(approximate="none"),
        )
    fp8_weight = torch.zeros((5376, 10752), dtype=torch.float8_e4m3fn)

    def _down(cols: int) -> SimpleNamespace:
        return SimpleNamespace(
            quant_method=_FakeFp8Method(w8a16_max_rows=16),
            weight=fp8_weight,
            bias=None,
            input_size_per_partition=cols,
            reduce_results=False,
            tp_size=2,
        )

    class _GateUp:
        def __init__(self, rows: int, cols: int, dev: str = "tpu") -> None:
            self.shape, self.device, self.dtype = (
                (rows, 2 * cols),
                SimpleNamespace(type=dev),
                torch.bfloat16,
            )

        def dim(self) -> int:
            return 2

    down_31b = _down(10752)
    assert gemma4_layers.should_fuse_gelu_quant(tanh_act, down_31b, _GateUp(32, 10752))
    for act, d, gu in (
        (tanh_act, _down(1076), _GateUp(32, 1076)),
        (tanh_act, _down(2100), _GateUp(32, 2100)),
        (tanh_act, down_31b, _GateUp(8, 10752)),
        (erf_act, down_31b, _GateUp(32, 10752)),
        (tanh_act, down_31b, _GateUp(32, 10752, dev="cpu")),
    ):
        assert not gemma4_layers.should_fuse_gelu_quant(act, d, gu)

    seen_shapes: list[tuple[int, int]] = []

    def _fake_gelu_quant(x: torch.Tensor, w_q: torch.Tensor):
        assert w_q is fp8_weight
        seen_shapes.append((x.shape[0], x.shape[1]))
        return (
            torch.ones((x.shape[0], x.shape[-1] // 2), dtype=torch.float8_e4m3fn),
            torch.full((x.shape[0], 1), 0.5, dtype=torch.float32),
        )

    import vllm_torchtpu.layers.adapter.linear_common as linear_common

    monkeypatch.setattr(
        linear_common, "gelu_tanh_and_mul_quantize", _fake_gelu_quant, raising=False
    )
    monkeypatch.setattr(gemma4_layers, "should_fuse_gelu_quant", lambda *_: True)
    out = gemma4_layers.down_proj_from_gate_up(
        SimpleNamespace(act_fn=tanh_act, down_proj=down_31b),
        torch.zeros((32, 2 * 10752), dtype=torch.bfloat16),
    )
    assert seen_shapes == [(32, 2 * 10752)] and out.shape == (32, 10752)


def test_attention_mm_prefix_and_shared_kv_guards() -> None:
    empty = torch.empty((0, 2), dtype=torch.int32)
    span = torch.tensor([[4, 260]], dtype=torch.int32)
    ac = SimpleNamespace(attn=SimpleNamespace(layer_name=None))
    assert not gemma4_attn.has_active_mm_prefix(
        ac, SimpleNamespace(mm_prefix_range=empty)
    )
    assert gemma4_attn.has_active_mm_prefix(ac, SimpleNamespace(mm_prefix_range=span))
    assert gemma4_attn.has_active_mm_prefix(
        ac, SimpleNamespace(mm_prefix_range_tensor=span)
    )
    assert gemma4_attn.has_active_mm_prefix(ac, kwargs={"extra": 1})

    impl = SimpleNamespace(sliding_window=None, shared_kv_unsupported=lambda: [])
    attn = SimpleNamespace(
        config=SimpleNamespace(attention_k_eq_v=True, num_kv_shared_layers=0),
        rotary_emb=SimpleNamespace(is_neox_style=True, rope_angles=64),
    )
    vllm_cfg = _make_vllm_config()
    assert gemma4_attn.should_enable_shared_kv_pages(attn, impl, vllm_cfg)
    lmo_cfg = _make_vllm_config(
        model_config=SimpleNamespace(
            is_multimodal_model=True, supports_multimodal_inputs=False
        )
    )
    assert gemma4_attn.should_enable_shared_kv_pages(attn, impl, lmo_cfg)
    apc_cfg = _make_vllm_config(
        cache_config=SimpleNamespace(enable_prefix_caching=True)
    )
    assert gemma4_attn.should_enable_shared_kv_pages(attn, impl, apc_cfg)
    for bad_cfg in (
        _make_vllm_config(
            model_config=SimpleNamespace(
                is_multimodal_model=True, supports_multimodal_inputs=True
            )
        ),
        _make_vllm_config(kv_transfer_config=SimpleNamespace(kv_connector="nixl")),
        _make_vllm_config(speculative_config=SimpleNamespace(method="mtp")),
    ):
        assert not gemma4_attn.should_enable_shared_kv_pages(attn, impl, bad_cfg)
    assert not gemma4_attn.should_enable_shared_kv_pages(
        attn,
        SimpleNamespace(sliding_window=1024, shared_kv_unsupported=lambda: []),
        vllm_cfg,
    )

    calls: list[dict[str, Any]] = []

    class _DummyAttn:
        is_kv_shared_layer, _tpu_qkv_prologue_ready = False, True

        def _tpu_orig_forward(
            self, positions: torch.Tensor, hidden_states: torch.Tensor, **kw: Any
        ) -> torch.Tensor:
            del positions
            calls.append(kw)
            return hidden_states + 1

    hs = torch.zeros((4, 16))
    out = gemma4_attn._patched_attention_forward(
        _DummyAttn(), torch.arange(4), hs, mm_extra=1
    )
    assert len(calls) == 1 and torch.equal(out, hs + 1)


def test_sequence_parallel_guards_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert gemma4_sp.use_sequence_parallel(256, 5376 * 2, 2)
    assert not gemma4_sp.use_sequence_parallel(64, 5376 * 2, 2)
    assert not gemma4_sp.use_sequence_parallel(257, 5376 * 2, 2)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(shape_variants, "aot_cache_tag", lambda: None)
    assert not gemma4_sp.use_sequence_parallel(256, 5376 * 2, 2)
    monkeypatch.setattr(shape_variants, "aot_cache_tag", lambda: "b256")
    assert gemma4_sp.use_sequence_parallel(256, 5376 * 2, 2)

    monkeypatch.setattr(gemma4_sp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        gemma4_sp, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(envs, "TPU_PRECOMPILE_PEER_REPLAY", False)

    m_dp1 = _make_fake_model(moe=True, dp_size=1)
    assert gemma4_sp.should_enable_sequence_parallel(m_dp1, _make_vllm_config())
    assert not gemma4_sp.should_enable_sequence_parallel(
        _make_fake_model(moe=True, dp_size=4), _make_vllm_config()
    )
    assert not gemma4_sp.should_enable_sequence_parallel(
        m_dp1, _make_vllm_config(model_config=SimpleNamespace(is_multimodal_model=True))
    )
    assert not gemma4_sp.should_enable_sequence_parallel(
        m_dp1, _make_vllm_config(speculative_config=SimpleNamespace(method="mtp"))
    )

    seen: list[str] = []

    class _DummyModel:
        _tpu_sp_enabled = True
        layers: list[Any] = []

        def _tpu_orig_forward(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            seen.append("orig")
            return "ok"

    assert gemma4_sp._patched_model_forward(
        _DummyModel(),
        input_ids=None,
        positions=torch.arange(4),
        inputs_embeds=torch.zeros((4, 8)),
    ) == "ok" and seen == ["orig"]

    class _DummyDecoderLayer:
        def __init__(self, sp_configured: bool) -> None:
            self._tpu_sp_configured = sp_configured

        def _tpu_orig_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
            per_layer_input: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            del positions, per_layer_input, kwargs
            base = hidden_states if residual is None else residual
            return base.to(torch.float32).to(hidden_states.dtype) + 1, None

    for sp_flag, pli in ((False, None), (True, torch.ones((4, 8)))):
        pos, hs, res_state = (
            torch.arange(4),
            torch.zeros((4, 8), dtype=torch.bfloat16),
            None,
        )
        for lyr in (_DummyDecoderLayer(sp_flag), _DummyDecoderLayer(sp_flag)):
            hs, res_state = gemma4_sp._patched_layer_forward(
                lyr, pos, hs, res_state, per_layer_input=pli
            )
            assert isinstance(hs, torch.Tensor) and res_state is None
            res_state = hs
        assert torch.equal(hs, torch.full((4, 8), 2.0, dtype=torch.bfloat16))


def test_context_parallel_guards_hbm_budget_and_zigzag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "TPU_PRECOMPILE_PEER_REPLAY", False)
    monkeypatch.setattr(envs, "VLLM_TPU_MOST_MODEL_LEN", None)
    model, cfg = _make_fake_model(moe=True, dp_size=1), _make_vllm_config()
    proj_b, embed_b = gemma4_cp.estimate_replica_bytes(model, cfg)
    assert 20 * (1 << 30) < proj_b < 26 * (1 << 30) and embed_b == 262144 * 2816 * 2

    for bad in (
        _make_vllm_config(
            scheduler_config=SimpleNamespace(scheduler_cls="AsyncScheduler")
        ),
        _make_vllm_config(model_config=SimpleNamespace(is_multimodal_model=True)),
        _make_vllm_config(kv_transfer_config=SimpleNamespace(kv_connector="nixl")),
        _make_vllm_config(parallel_config=_make_parallel_cfg(tp=2, dp=4)),
        _make_vllm_config(
            compilation_config=SimpleNamespace(compile_sizes=[16, 8192, 16384 + 512])
        ),
    ):
        assert gemma4_cp.should_enable_context_parallel(
            model, bad, free_hbm_bytes=80 << 30
        )[:2] == (False, False)

    floor = gemma4_cp.HBM_SAFETY_FLOOR_BYTES
    assert gemma4_cp.should_enable_context_parallel(
        model, cfg, free_hbm_bytes=proj_b + floor - 1
    )[:2] == (False, False)
    assert gemma4_cp.should_enable_context_parallel(
        model, cfg, free_hbm_bytes=proj_b + floor + 1024
    ) == (True, False, "")
    assert gemma4_cp.should_enable_context_parallel(
        model, cfg, free_hbm_bytes=proj_b + embed_b + floor + 1024
    ) == (True, True, "")
    apc_cp_cfg = _make_vllm_config(
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=256)
    )
    assert gemma4_cp.should_enable_context_parallel(
        model, apc_cp_cfg, free_hbm_bytes=proj_b + floor + 1024
    ) == (True, False, "")
    from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackend

    assert PallasBatchedRPAAttentionBackend.get_preferred_block_size(16) == 256

    for world in (2, 8):
        plan = cp_kernel.ZigzagPlan(num_tokens=4096, world=world, page_size=256)
        tokens = torch.arange(plan.num_tokens * 4).reshape(plan.num_tokens, 4)
        ranked = gemma4_cp._rank_major(tokens, plan)
        stacked = torch.cat(
            [gemma4_cp._rank_rows(tokens, plan, r) for r in range(world)], dim=0
        )
        assert torch.equal(ranked, stacked) and torch.equal(
            gemma4_cp._token_order(ranked, plan), tokens
        )

    class _LoaderModel:
        def __init__(self, *, has_embed: bool) -> None:
            setattr(
                self,
                gemma4_cp._REPLICA,
                SimpleNamespace(embed_tokens=object() if has_embed else None),
            )

        def _tpu_orig_load_weights(self, items: Any) -> set[str]:
            return {name for name, _ in items}

    raw_weights = [
        ("embed_tokens.weight", torch.zeros((4, 4))),
        ("layers.0.self_attn.q_proj.weight", torch.ones((4, 4))),
        ("layers.0.experts.gate_up_proj", torch.ones((2, 4, 4))),
        ("layers.0.input_layernorm.weight", torch.ones((4,))),
    ]
    rep = gemma4_cp._REPLICA
    no_emb = gemma4_cp._patched_model_load_weights(
        _LoaderModel(has_embed=False), raw_weights
    )
    assert (
        f"{rep}.embed_tokens.weight" not in no_emb
        and f"{rep}.layers.0.input_layernorm.weight" not in no_emb
    )
    assert (
        f"{rep}.layers.0.self_attn.q_proj.weight" in no_emb
        and f"{rep}.layers.0.experts.gate_up_proj" in no_emb
    )
    assert f"{rep}.embed_tokens.weight" in gemma4_cp._patched_model_load_weights(
        _LoaderModel(has_embed=True), raw_weights
    )

    wide_sw = _make_fake_model(moe=True, dp_size=1)
    wide_sw.layers[0].self_attn.sliding_window = 2048
    tp8_cfg = _make_vllm_config(
        parallel_config=_make_parallel_cfg(tp=8),
        compilation_config=SimpleNamespace(compile_sizes=[16, 8192, 16384, 32768]),
    )
    assert gemma4_cp.should_enable_context_parallel(
        wide_sw, tp8_cfg, free_hbm_bytes=80 << 30
    )[:2] == (False, False)


def test_cp_profile_cap_and_upstream_gemma4_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models import gemma4 as upstream_gemma4

    from vllm_torchtpu import tpu_info

    monkeypatch.setattr(tpu_info, "_v7x_model_fastpaths", False)
    gemma4_pkg.maybe_patch_gemma4(
        SimpleNamespace(hf_text_config=SimpleNamespace(model_type="gemma4_text"))
    )
    assert tpu_info._v7x_model_fastpaths is True
    for cls, attr in (
        (upstream_gemma4.Gemma4MLP, "_TPU_GEMMA4_MLP_PATCHED"),
        (upstream_gemma4.Gemma4Attention, "_TPU_GEMMA4_ATTN_PATCHED"),
        (upstream_gemma4.Gemma4Model, "_TPU_GEMMA4_SP_PATCHED"),
        (upstream_gemma4.Gemma4ForCausalLM, "_TPU_GEMMA4_CP_PATCHED"),
    ):
        assert getattr(cls, attr, False)

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):

        class _LangModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = RMSNorm(16).to(torch.bfloat16)

            def _tpu_orig_load_weights(self, items: Any) -> set[str]:
                return {name for name, _ in items}

        lang, vision_norm = _LangModel(), RMSNorm(16).to(torch.bfloat16)
        gemma4_cp._patched_model_load_weights(lang, [("norm.weight", torch.ones(16))])
        assert (
            lang.norm.weight.dtype == torch.float32
            and vision_norm.weight.dtype == torch.bfloat16
        )

    seen_lengths: list[int] = []
    monkeypatch.setattr(gemma4_cp, "_profiling", lambda _m: True)

    class _FakeCausalLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self._cp_programs = {163840: torch.nn.Identity()}
            self._max_non_cp_tokens = 8192
            self.vllm_config = SimpleNamespace(
                compilation_config=SimpleNamespace(compile_sizes=[163840])
            )

        def _tpu_orig_forward(
            self, ids: torch.Tensor, pos: torch.Tensor, *_: Any, **__: Any
        ) -> torch.Tensor:
            del pos
            seen_lengths.append(ids.shape[0])
            return torch.ones((ids.shape[0], 32), dtype=torch.bfloat16)

    fake_lm = _FakeCausalLM()
    out = gemma4_cp._patched_causal_lm_forward(
        fake_lm,
        torch.zeros(163840, dtype=torch.int32),
        torch.arange(163840, dtype=torch.int32),
    )
    assert seen_lengths == [8192] and out.shape == (163840, 32)
    assert fake_lm.vllm_config.compilation_config.compile_sizes == [163840]

    monkeypatch.setattr(
        gemma4_cp,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(query_start_loc_cpu=None, seq_lens_cpu=None),
    )
    gemma4_cp._check_solo_prompt(16384)
    monkeypatch.setattr(
        gemma4_cp,
        "get_vllm_model_wrapper_context",
        lambda: SimpleNamespace(
            query_start_loc_cpu=torch.tensor([0, 16384], dtype=torch.int32),
            seq_lens_cpu=None,
        ),
    )
    with pytest.raises(RuntimeError, match="missing host"):
        gemma4_cp._check_solo_prompt(16384)


def test_cp_prefill_attention_build_ops_fake_tensor_signatures() -> None:
    pytest.importorskip("torch_tpu")
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, cp_prefill
    from vllm_torchtpu.layers.adapter import cp_prefill_attention

    pos = (2, 16, 8, 256, 1024, 1.0, None, 1.0, 1.0, False)
    kw = dict(
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        fence_output=True,
        prologue_eps=1e-6,
        fp8_out_rows=True,
        exchange_from_qkv=True,
        paged_gather=True,
        prologue_dma_stores=True,
    )
    ops = cp_prefill_attention.build_ops(cp_prefill.LayerConfig(*pos, **kw))
    with FakeTensorMode():
        qkv = torch.empty((512, 32 * 256), dtype=torch.bfloat16)
        cos = sin = torch.empty((512, 128), dtype=torch.float32)
        w = torch.empty((256,), dtype=torch.bfloat16)
        bt, sl = (
            torch.empty((16,), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
        )
        kvc = torch.empty((16, 256, 8, 256), dtype=torch.float8_e4m3fn)
        q_hbm, new_kv = ops.prologue(qkv, cos, sin, w, w)
        ex = ops.sliding_exchange(qkv, cos, sin, w, w, sl, sl, 256)
        gat = torch.empty((2 * ex.shape[0], *ex.shape[1:]), dtype=torch.int8)
        out = ops.sliding_attention(kvc, q_hbm, new_kv, gat, sl, bt, sl)
        assert isinstance(out, tuple) and len(out) == 3
