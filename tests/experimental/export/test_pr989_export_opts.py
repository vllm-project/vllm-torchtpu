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
"""CPU unit tests for Qwen3.5 TP8_EP export optimization wiring."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from vllm_torchtpu.experimental.export import universal_adapter
from vllm_torchtpu.layers.adapter import norm_weight_hoist, small_all_reduce_op


def _make_adapter(model: nn.Module) -> universal_adapter.UniversalVllmExportableLLM:
    adapter = object.__new__(universal_adapter.UniversalVllmExportableLLM)
    adapter._vllm_config = None
    adapter._export_config = SimpleNamespace(
        max_model_len=128,
        max_num_seqs=4,
        block_size=16,
        vocab_size=32,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        uses_mrope=False,
        is_multimodal=False,
    )
    adapter._model = model
    adapter._mtp_draft = None
    adapter._draft_model = None
    adapter._patched_rope_modules = 0
    return adapter


def test_hoisted_norm_weights_stay_out_of_constant_names() -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        norm = GemmaRMSNorm(16, eps=1e-6)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = norm
            self.register_buffer("small_buf", torch.tensor([3], dtype=torch.int32))

    model = _Model()
    try:
        assert norm_weight_hoist.install(model) == 1
        adapter = _make_adapter(model)
        constants = adapter.constant_names()
        params = adapter.abstract_params()
        assert "norm._tpu_weight_plus_one" in params
        assert "norm._tpu_weight_plus_one" not in constants
        assert "small_buf" in constants
    finally:
        if hasattr(GemmaRMSNorm, "_tpu_weight_plus_one"):
            delattr(GemmaRMSNorm, "_tpu_weight_plus_one")
        if hasattr(GemmaRMSNorm, "_tpu_norm_weight_hoisted"):
            delattr(GemmaRMSNorm, "_tpu_norm_weight_hoisted")


def test_small_all_reduce_export_context_and_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu import envs

    monkeypatch.setenv("TPU_SMALL_ALL_REDUCE", "1")
    assert envs.TPU_SMALL_ALL_REDUCE

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    fake_rank = torch.tensor([2], dtype=torch.int32)

    def _fake_ar(x: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        calls.append((x, rank))
        return x * 2

    monkeypatch.setattr(small_all_reduce_op, "_RANK", fake_rank)
    monkeypatch.setattr(small_all_reduce_op, "_AR_OP", _fake_ar)
    monkeypatch.setattr(small_all_reduce_op, "_MESH_RANKS", tuple(range(8)))
    monkeypatch.setattr(small_all_reduce_op, "_CORES", 1)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            small = small_all_reduce_op.all_reduce_or(
                x[:4], range(8), lambda t: t + 100
            )
            big = small_all_reduce_op.all_reduce_or(
                torch.ones(128, 128, dtype=torch.bfloat16),
                range(8),
                lambda t: t,
            )
            return small.sum() + big.sum()

    model = _Model()
    assert universal_adapter._patch_small_all_reduce(model) == 1
    assert hasattr(model, universal_adapter._SMALL_AR_RANK_BUFFER)
    adapter = _make_adapter(model)
    assert universal_adapter._SMALL_AR_RANK_BUFFER in adapter.constant_names()

    x = torch.ones(4, 128, dtype=torch.bfloat16)
    with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
        out = torch.func.functional_call(
            model, dict(model.named_parameters(), **dict(model.named_buffers())), (x,)
        )
    assert len(calls) == 1
    assert calls[0][1].tolist() == [2]
    assert torch.is_tensor(out)


def test_qwen3_next_patch_enables_fp8_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.models import qwen3_next

    from vllm_torchtpu.models.vllm import qwen3_next_patch

    orig_decoder = qwen3_next.Qwen3NextDecoderLayer.forward
    orig_model = qwen3_next.Qwen3NextModel.forward
    orig_attn = qwen3_next.Qwen3NextAttention.forward
    orig_flag = qwen3_next_patch._fp8_gather
    try:
        monkeypatch.setenv("TPU_MOE_SEQUENCE_PARALLEL", "1")
        monkeypatch.setenv("TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER", "1")
        qwen3_next_patch.maybe_patch_qwen3_next()
        assert qwen3_next_patch._fp8_gather is True
        assert qwen3_next.Qwen3NextAttention.forward is (
            qwen3_next_patch.patched_attention_forward
        )
    finally:
        qwen3_next.Qwen3NextDecoderLayer.forward = orig_decoder
        qwen3_next.Qwen3NextModel.forward = orig_model
        qwen3_next.Qwen3NextAttention.forward = orig_attn
        qwen3_next_patch._fp8_gather = orig_flag


def test_recurrent_state_specs_respect_mamba_ssm_cache_dtype() -> None:
    class _Layer:
        def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return ((4, 64), (16, 128, 128))

        def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
            return (torch.float32, torch.bfloat16)

    specs = universal_adapter._recurrent_state_specs(_Layer(), num_slots=8, tp_size=8)
    assert specs[0].dtype == "f32"
    assert specs[1].dtype == "bf16"
    assert specs[1].shape == (8, 16, 128, 128)


def test_sp_min_rows_gates_fused_moe_v3_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu.layers.adapter import sequence_parallel
    from vllm_torchtpu.layers.adapter.quantization import fused_moe_ep_v3_dispatch

    monkeypatch.setenv("TPU_SP_MIN_ROWS", "257")
    monkeypatch.setattr(
        fused_moe_ep_v3_dispatch,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    v3_calls: list[int] = []
    grouped_calls: list[int] = []
    ar_calls: list[int] = []
    monkeypatch.setattr(
        fused_moe_ep_v3_dispatch,
        "fused_moe_ep_v3",
        lambda _m, x, *_a: v3_calls.append(x.shape[0]) or x,
    )
    monkeypatch.setattr(
        fused_moe_ep_v3_dispatch,
        "tensor_model_parallel_all_reduce",
        lambda x: ar_calls.append(x.shape[0]) or x * 2,
    )
    method = SimpleNamespace(_fused_moe_ep_v3_op=SimpleNamespace(step_min_tokens=33))
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(is_sequence_parallel=True)
        ),
        w13_weight=None,
        w2_weight=None,
        w13_weight_scale_inv=None,
        w2_weight_scale_inv=None,
        _experts_start=None,
    )

    prev = sequence_parallel.set_active(False)
    try:
        x = torch.ones(64, 16)
        out = fused_moe_ep_v3_dispatch.apply(
            method,
            layer,
            x,
            torch.ones(64, 8),
            lambda _l, inp, _r: grouped_calls.append(inp.shape[0]) or inp,
        )
        assert v3_calls == []
        assert grouped_calls == [64]
        assert ar_calls == [64]
        torch.testing.assert_close(out, x * 2)

        sequence_parallel.set_active(True)
        fused_moe_ep_v3_dispatch.apply(
            method,
            layer,
            x,
            torch.ones(64, 8),
            lambda _l, inp, _r: grouped_calls.append(inp.shape[0]) or inp,
        )
        assert v3_calls == [64]
    finally:
        sequence_parallel.set_active(prev)


def test_gated_norm_quant_op_reads_tile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_torchtpu.layers.adapter import gated_norm_quant_op

    monkeypatch.setenv("TPU_GATED_NORM_TILE", "256")
    gated_norm_quant_op._build_ops(1e-6)
    assert (1e-6, 256) in gated_norm_quant_op._norm_ops


def test_params_read_by_forward_unions_min_and_max_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu.experimental.export import exporter

    num_user = exporter._NUM_USER_INPUTS
    seen_buckets: list[int] = []

    def _fake_lower(
        _model,
        *,
        user_inputs,
        forward_names,
        params_map,
        state_specs,
        bucket_size: int,
        trailing_user_inputs,
    ) -> str:
        del user_inputs, forward_names, params_map, state_specs, trailing_user_inputs
        seen_buckets.append(bucket_size)
        args = ", ".join(f"%arg{i}: tensor<1xi32>" for i in range(num_user + 3))
        # arg 0..num_user-1: user inputs; forward_names:
        #   0 -> only_small_buf, 1 -> only_large_buf, 2 -> unused_buf
        if bucket_size == 4:
            return f"func.func public @main({args}) {{\n  return %arg{num_user}\n}}"
        if bucket_size == 512:
            return f"func.func public @main({args}) {{\n  return %arg{num_user + 1}\n}}"
        raise AssertionError(f"unexpected bucket {bucket_size}")

    monkeypatch.setattr(
        exporter.tpu_lowering, "lower_torch_forward_to_mlir", _fake_lower
    )
    monkeypatch.setattr(exporter, "_user_inputs", lambda *_args: ())
    monkeypatch.setattr(exporter, "_req_dist_len", lambda _model: None)
    monkeypatch.setattr(exporter, "_block_tables_name", lambda _model: None)
    monkeypatch.setattr(exporter, "_state_specs", lambda _model: {})
    monkeypatch.setattr(exporter, "_extra_user_inputs", lambda _model: ())

    fake_model = SimpleNamespace(
        export_config=SimpleNamespace(
            token_buckets=(4, 16, 256, 512),
            max_num_seqs=4,
            block_size=16,
            is_multimodal=False,
        )
    )
    params_map = {
        "only_small_buf": SimpleNamespace(shape=(1,), dtype=torch.int32),
        "only_large_buf": SimpleNamespace(shape=(1,), dtype=torch.int64),
        "unused_buf": SimpleNamespace(shape=(1,), dtype=torch.int32),
        "lm_head.weight": SimpleNamespace(shape=(32, 16), dtype=torch.bfloat16),
    }
    kept = exporter._params_read_by_forward(fake_model, params_map)
    assert seen_buckets == [4, 512]
    assert set(kept) == {"only_small_buf", "only_large_buf", "lm_head.weight"}


def test_adapter_constant_names_includes_lowtok_copies_buffers() -> None:
    class _Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w13_weight = torch.nn.Parameter(
                torch.empty(80, 4, 4, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.register_buffer(
                "_lowtok_tables", torch.zeros(1024, dtype=torch.int32), persistent=False
            )
            self.register_buffer(
                "_lowtok_prim", torch.zeros(512, dtype=torch.int32), persistent=False
            )
            self.register_buffer(
                "_lowtok_copy", torch.zeros(512, dtype=torch.int32), persistent=False
            )

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = _Layer()

    model = _Model()
    adapter = _make_adapter(model)
    names = set(adapter.constant_names())
    assert {
        "layer._lowtok_tables",
        "layer._lowtok_prim",
        "layer._lowtok_copy",
    } <= names
    assert "layer.w13_weight" not in names
