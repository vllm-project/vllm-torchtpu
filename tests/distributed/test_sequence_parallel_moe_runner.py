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
"""Unit tests for sequence-parallel MoE configuration and Qwen3-Next/3.5 patch."""

from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm import qwen3_next_patch


@pytest.fixture(autouse=True)
def _install_sp_patch():
    from vllm.config.parallel import ParallelConfig

    if not getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        vllm_torchtpu._patch_disable_sequence_parallel_moe()


@pytest.mark.parametrize(
    ("raw", "mode"),
    [
        (None, "off"),
        ("", "off"),
        ("0", "off"),
        ("false", "off"),
        ("1", "auto"),
        ("True", "auto"),
        ("auto", "auto"),
        (" FORCE ", "force"),
    ],
)
def test_sequence_parallel_mode_parser(monkeypatch, raw, mode):
    name = "VLLM_TPU_TEST_SP_MOE_MODE"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert envs.env_moe_sequence_parallel_mode(name)() == mode


def test_sequence_parallel_mode_parser_rejects_typos(monkeypatch):
    name = "VLLM_TPU_TEST_SP_MOE_MODE"
    monkeypatch.setenv(name, "forced")
    with pytest.raises(ValueError, match="forced"):
        envs.env_moe_sequence_parallel_mode(name)()


def _sp_moe_enabled(
    monkeypatch, *, mode, dp=1, live_dp=None, env_dp=None, tp=4, ep=True
):
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed import parallel_state

    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", mode)
    monkeypatch.setattr(
        parallel_state,
        "_DP",
        None if live_dp is None else SimpleNamespace(world_size=live_dp),
    )
    if env_dp is None:
        monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    else:
        monkeypatch.setenv("TORCH_TPU_DP_SIZE", str(env_dp))
    config = SimpleNamespace(
        enable_expert_parallel=ep,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=1,
        data_parallel_size=dp,
    )
    assert ParallelConfig.use_sequence_parallel_moe.fget is not None
    return ParallelConfig.use_sequence_parallel_moe.fget(config)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(mode="off", dp=4), False),
        # "auto" follows upstream: SP only when DP > 1.
        (dict(mode="auto", dp=1), False),
        (dict(mode="auto", dp=2), True),
        # Per-engine configs may collapse data_parallel_size to 1; the live DP
        # group or TORCH_TPU_DP_SIZE still reveal the real DP width.
        (dict(mode="auto", dp=1, live_dp=4), True),
        (dict(mode="auto", dp=1, env_dp=4), True),
        # "force" also enables DP == 1 (TP-only + EP) and EP == False (TP + SP).
        (dict(mode="force", dp=1), True),
        (dict(mode="force", dp=1, ep=False), True),
        (dict(mode="force", dp=2, ep=False), True),
        # Structural requirements hold in every mode; "auto" still requires EP.
        (dict(mode="force", dp=1, tp=1), False),
        (dict(mode="auto", dp=2, ep=False), False),
    ],
)
def test_use_sequence_parallel_moe_gate(monkeypatch, kwargs, expected):
    assert _sp_moe_enabled(monkeypatch, **kwargs) is expected


@pytest.mark.parametrize(
    ("mode", "model_type", "expected"),
    [
        ("off", "qwen3_5_moe", False),
        ("auto", None, False),
        ("auto", "qwen2_5_vl", False),
        ("auto", "qwen3_next", None),
        ("force", "qwen3_5_moe", None),
        ("force", "qwen3_5_moe_text", None),
    ],
)
def test_maybe_patch_qwen3_next_model_gate(monkeypatch, mode, model_type, expected):
    from vllm.model_executor.models import qwen3_next

    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", mode)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer,
        "forward",
        qwen3_next_patch.UPSTREAM_DECODER_LAYER_FORWARD,
    )
    monkeypatch.setattr(
        qwen3_next.Qwen3NextModel,
        "forward",
        qwen3_next_patch.UPSTREAM_MODEL_FORWARD,
    )
    model_config = (
        None
        if model_type is None
        else SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )
    result = qwen3_next_patch.maybe_patch_qwen3_next(model_config)
    assert result is expected
    if expected is None:
        assert (
            qwen3_next.Qwen3NextDecoderLayer.forward
            is qwen3_next_patch.patched_decoder_layer_forward
        )
        assert (
            qwen3_next.Qwen3NextModel.forward is qwen3_next_patch.patched_model_forward
        )
    else:
        assert (
            qwen3_next.Qwen3NextDecoderLayer.forward
            is qwen3_next_patch.UPSTREAM_DECODER_LAYER_FORWARD
        )
        assert (
            qwen3_next.Qwen3NextModel.forward is qwen3_next_patch.UPSTREAM_MODEL_FORWARD
        )


def test_patched_decoder_layer_forward_uses_narrow_and_reduce_scatter(monkeypatch):
    monkeypatch.setattr(
        qwen3_next_patch, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "tensor_model_parallel_all_gather",
        lambda x, dim: torch.cat([x, x + 10], dim=dim),
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim: x[: x.shape[0] // 2] + x[x.shape[0] // 2 :],
    )

    mlp_calls = []

    def mlp(x, already_sequence_parallel=False):
        mlp_calls.append((x.shape, already_sequence_parallel))
        return x * 2

    layer = SimpleNamespace(
        input_layernorm=lambda h, r=None: (h, r) if r is not None else h,
        post_attention_layernorm=lambda h, r: (h + r, r),
        use_attn_reduce_scatter_for_moe=True,
        layer_type="full_attention",
        self_attn=lambda hidden_states, positions: hidden_states + 1,
        layer_scale=False,
        mlp=mlp,
    )
    chunk = torch.ones(2, 3)
    positions = torch.arange(4)

    out, res = qwen3_next_patch.patched_decoder_layer_forward(
        layer, chunk, None, positions
    )

    assert mlp_calls == [(torch.Size([2, 3]), True)]
    assert out.shape == (2, 3)
    assert res.shape == (2, 3)


def test_patched_decoder_layer_forward_sp_min_tokens_fallback(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_SP_MIN_TOKENS", 65)
    monkeypatch.setattr(
        qwen3_next_patch, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "tensor_model_parallel_all_reduce",
        lambda x: x * 3,
    )

    mlp_calls = []

    def mlp(x, already_sequence_parallel=False):
        mlp_calls.append((x.shape, already_sequence_parallel))
        return x * 2

    layer = SimpleNamespace(
        input_layernorm=lambda h, r=None: (h, r) if r is not None else h,
        post_attention_layernorm=lambda h, r: (h + r, r),
        use_attn_reduce_scatter_for_moe=True,
        layer_type="full_attention",
        self_attn=lambda hidden_states, positions: hidden_states + 1,
        layer_scale=False,
        mlp=mlp,
    )
    # 12 tokens (< VLLM_TPU_SP_MIN_TOKENS=65 and not divisible by TP=8)
    hidden = torch.ones(12, 3)
    positions = torch.arange(12)

    out, res = qwen3_next_patch.patched_decoder_layer_forward(
        layer, hidden, None, positions
    )

    assert mlp_calls == [(torch.Size([12, 3]), False)]
    assert out.shape == (12, 3)
    assert res.shape == (12, 3)


@pytest.mark.parametrize(
    ("tp_size", "num_tokens", "expect_fp8_gather"),
    [
        (4, 256, False),
        (4, 260, True),
        (8, 1024, False),
        (8, 1032, True),
    ],
)
def test_patched_decoder_layer_fp8_gather_auto_min_tokens(
    monkeypatch, tp_size, num_tokens, expect_fp8_gather
):
    monkeypatch.setattr(envs, "VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER", True)
    monkeypatch.setattr(envs, "VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS", 0)
    monkeypatch.setattr(
        qwen3_next_patch, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    gather_dtypes: list[torch.dtype] = []

    def fake_all_gather(x, dim):
        gather_dtypes.append(x.dtype)
        return x.repeat(tp_size, *([1] * (x.ndim - 1)))

    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim: x[: x.shape[0] // tp_size],
    )
    monkeypatch.setattr(
        qwen3_next_patch,
        "linear_common",
        SimpleNamespace(
            quantize_rows=lambda x: (
                x.to(torch.float8_e4m3fn),
                torch.ones(x.shape[0], 1, dtype=torch.float32),
            )
        ),
    )

    attn_seen: dict[str, object] = {}

    def linear_attn(hidden_states, **kwargs):
        attn_seen["hidden_states"] = hidden_states
        attn_seen["kwargs"] = kwargs
        return torch.zeros(hidden_states.shape[0], 3, dtype=torch.bfloat16)

    layer = SimpleNamespace(
        input_layernorm=lambda h, r=None: (h, r) if r is not None else h,
        post_attention_layernorm=lambda h, r: (h + r, r),
        use_attn_reduce_scatter_for_moe=True,
        layer_type="linear_attention",
        linear_attn=linear_attn,
        layer_scale=False,
        mlp=lambda x, already_sequence_parallel=False: x,
    )
    chunk = torch.ones(num_tokens // tp_size, 3, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens)

    qwen3_next_patch.patched_decoder_layer_forward(layer, chunk, None, positions)

    if expect_fp8_gather:
        assert gather_dtypes == [torch.float8_e4m3fn, torch.float32]
        prequant = attn_seen["kwargs"]["prequant"]
        assert prequant[0].shape == (num_tokens, 3)
        assert prequant[1].shape == (num_tokens, 1)
    else:
        assert gather_dtypes == [torch.bfloat16]
        assert "prequant" not in attn_seen["kwargs"]


def test_project_prequant_uses_prequant_matmul_for_per_channel_fp8(monkeypatch):
    import sys
    import types
    from pathlib import Path

    if "vllm_torchtpu.layers" not in sys.modules:
        pkg = types.ModuleType("vllm_torchtpu.layers")
        pkg.__path__ = [
            str(Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/layers")
        ]
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.layers", pkg)

    from vllm_torchtpu.layers.adapter import linear_common
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    calls: list[tuple[torch.Size, torch.Size, torch.Size, torch.Size]] = []

    def fake_prequant_matmul(x_q, x_s, w_q, w_s):
        calls.append((x_q.shape, x_s.shape, w_q.shape, w_s.shape))
        return torch.ones(x_q.shape[0], w_q.shape[1], dtype=torch.bfloat16)

    monkeypatch.setattr(
        linear_common, "quantized_matmul_prequant", fake_prequant_matmul
    )
    layer = SimpleNamespace(
        quant_method=VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU),
        bias=torch.full((5,), 2.0, dtype=torch.bfloat16),
        weight=torch.zeros(4, 5, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(5, dtype=torch.float32),
    )
    x_q = torch.zeros(6, 4, dtype=torch.float8_e4m3fn)
    x_s = torch.ones(6, 1, dtype=torch.float32)

    out = linear_common.project_prequant(layer, x_q, (x_q, x_s))

    assert calls == [
        (torch.Size([6, 4]), torch.Size([6, 1]), torch.Size([4, 5]), torch.Size([5]))
    ]
    torch.testing.assert_close(out, torch.full((6, 5), 3.0, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("num_tokens", "expect_sp"),
    [
        (32, False),
        (128, False),
        (256, True),
        (1024, True),
    ],
)
def test_patched_decoder_layer_bucket_gated_sp_with_fused_tp_ar(
    monkeypatch, num_tokens, expect_sp
):
    from vllm_torchtpu.layers.adapter import fused_moe_tp_ar

    tp_size = 8
    monkeypatch.setattr(envs, "USE_MOE_FUSED_TP_AR", True)
    monkeypatch.setattr(envs, "VLLM_TPU_SP_MIN_TOKENS", 0)
    monkeypatch.setattr(envs, "VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER", False)
    monkeypatch.setattr(
        qwen3_next_patch, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    calls: list[str] = []

    def fake_all_gather(x, dim):
        calls.append("all_gather")
        return x.repeat(tp_size, *([1] * (x.ndim - 1)))

    def fake_reduce_scatter(x, dim):
        calls.append("reduce_scatter")
        return x[: x.shape[0] // tp_size]

    def fake_all_reduce(x):
        calls.append("all_reduce")
        return x

    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_reduce_scatter", fake_reduce_scatter
    )
    monkeypatch.setattr(
        qwen3_next_patch, "tensor_model_parallel_all_reduce", fake_all_reduce
    )

    seen_sp: list[bool] = []

    def mlp(x, already_sequence_parallel=False):
        seen_sp.append(fused_moe_tp_ar._CURRENT_LAYER_USE_SP)
        return x

    layer = SimpleNamespace(
        input_layernorm=lambda h, r=None: (h, r) if r is not None else h,
        post_attention_layernorm=lambda h, r: (h + r, r),
        use_attn_reduce_scatter_for_moe=True,
        layer_type="linear_attention",
        linear_attn=lambda hidden_states, **kwargs: hidden_states,
        layer_scale=False,
        mlp=mlp,
    )
    in_rows = num_tokens // tp_size if expect_sp else num_tokens
    hidden = torch.ones(in_rows, 4, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens)

    out, _ = qwen3_next_patch.patched_decoder_layer_forward(
        layer, hidden, None, positions
    )

    assert seen_sp == [expect_sp]
    assert out.shape == (in_rows, 4)
    if expect_sp:
        assert calls == ["all_gather", "reduce_scatter"]
    else:
        assert calls == ["all_reduce"]


def test_patched_mlp_init_keeps_shared_expert_tp_sharded(monkeypatch):
    import vllm.config as vllm_config_mod

    monkeypatch.setattr(envs, "USE_MOE_FUSED_TP_AR", True)
    monkeypatch.setattr(
        vllm_config_mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=False)
        ),
    )
    recorded_kwargs: list[dict[str, object]] = []

    def fake_upstream_mlp_init(self, *args, **kwargs):
        recorded_kwargs.append(kwargs)

    monkeypatch.setattr(qwen3_next_patch, "UPSTREAM_MLP_INIT", fake_upstream_mlp_init)

    qwen3_next_patch.patched_mlp_init(
        SimpleNamespace(),
        hidden_size=4096,
        intermediate_size=1024,
        hidden_act="silu",
        quant_config=None,
        reduce_results=False,
        expert_gate=None,
        is_sequence_parallel=True,
        disable_tp=False,
        prefix="model.layers.0.mlp.shared_expert",
    )

    assert len(recorded_kwargs) == 1
    assert recorded_kwargs[0]["hidden_size"] == 4096
    assert recorded_kwargs[0]["is_sequence_parallel"] is False
    assert recorded_kwargs[0]["disable_tp"] is False
