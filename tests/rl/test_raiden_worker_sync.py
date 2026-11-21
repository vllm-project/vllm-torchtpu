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
"""CPU unit tests for `param_layout` classification and `RaidenWorkerSync`."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu.layers.adapter.gdn_linear import GdnColumnParallelLinear
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.rl import raiden_worker_sync


def _make_rms_norm(hidden_size: int) -> GemmaRMSNorm:
    mod = GemmaRMSNorm.__new__(GemmaRMSNorm)
    nn.Module.__init__(mod)
    mod.weight = nn.Parameter(
        torch.zeros(hidden_size, dtype=torch.bfloat16),
        requires_grad=False,
    )
    return mod


def _make_vocab_embedding(
    vocab_size: int,
    hidden_size: int,
    tp_size: int,
    *,
    org_vocab_size: int | None = None,
) -> VocabParallelEmbedding:
    mod = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    nn.Module.__init__(mod)
    mod.tp_size = tp_size
    mod.org_vocab_size = org_vocab_size or vocab_size
    mod.quant_method = None
    mod.weight = nn.Parameter(
        torch.zeros(vocab_size // tp_size, hidden_size, dtype=torch.bfloat16),
        requires_grad=False,
    )
    return mod


def _make_lm_head(
    vocab_size: int,
    hidden_size: int,
    tp_size: int,
) -> ParallelLMHead:
    mod = ParallelLMHead.__new__(ParallelLMHead)
    nn.Module.__init__(mod)
    mod.tp_size = tp_size
    mod.org_vocab_size = vocab_size
    mod.quant_method = None
    mod.weight = nn.Parameter(
        torch.zeros(vocab_size // tp_size, hidden_size, dtype=torch.bfloat16),
        requires_grad=False,
    )
    return mod


def _make_replicated_linear(
    in_features: int,
    out_features: int,
    *,
    flipped: bool = False,
) -> ReplicatedLinear:
    mod = ReplicatedLinear.__new__(ReplicatedLinear)
    nn.Module.__init__(mod)
    mod.tp_size = 1
    mod.disable_tp = False
    mod.quant_method = None
    shape = (in_features, out_features) if flipped else (out_features, in_features)
    mod.weight = nn.Parameter(
        torch.zeros(shape, dtype=torch.bfloat16),
        requires_grad=False,
    )
    if flipped:
        setattr(mod, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    return mod


def _make_row_linear(
    in_features: int,
    out_features: int,
    tp_size: int,
    *,
    flipped: bool = True,
    bias: bool = False,
) -> RowParallelLinear:
    mod = RowParallelLinear.__new__(RowParallelLinear)
    nn.Module.__init__(mod)
    mod.tp_size = tp_size
    mod.disable_tp = False
    mod.quant_method = None
    local_in = in_features // tp_size
    shape = (local_in, out_features) if flipped else (out_features, local_in)
    mod.weight = nn.Parameter(
        torch.zeros(shape, dtype=torch.bfloat16),
        requires_grad=False,
    )
    if bias:
        mod.bias = nn.Parameter(
            torch.zeros(out_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
    if flipped:
        setattr(mod, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    return mod


def _make_qkv_linear(
    hidden_size: int,
    head_size: int,
    total_q_heads: int,
    total_kv_heads: int,
    tp_size: int,
    *,
    flipped: bool = True,
) -> QKVParallelLinear:
    mod = QKVParallelLinear.__new__(QKVParallelLinear)
    nn.Module.__init__(mod)
    mod.tp_size = tp_size
    mod.disable_tp = False
    mod.quant_method = None
    mod.num_kv_head_replicas = max(1, tp_size // total_kv_heads)
    local_kv = max(1, total_kv_heads // tp_size)
    local_out = ((total_q_heads // tp_size) + 2 * local_kv) * head_size
    shape = (hidden_size, local_out) if flipped else (local_out, hidden_size)
    mod.weight = nn.Parameter(
        torch.zeros(shape, dtype=torch.bfloat16),
        requires_grad=False,
    )
    if flipped:
        setattr(mod, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    return mod


def _make_gdn_column_linear(
    in_features: int,
    checkpoint_sizes: list[int],
    replication_factors: list[int],
    tp_size: int,
    *,
    is_conv: bool = False,
    flipped: bool = False,
) -> GdnColumnParallelLinear:
    mod = GdnColumnParallelLinear.__new__(GdnColumnParallelLinear)
    nn.Module.__init__(mod)
    mod.tp_size = tp_size
    mod.disable_tp = False
    mod.quant_method = None
    mod.checkpoint_sizes = list(checkpoint_sizes)
    mod.replication_factors = list(replication_factors)
    mod.output_sizes = [s * r for s, r in zip(checkpoint_sizes, replication_factors)]
    local_out = sum(s // tp_size for s in mod.output_sizes)
    if is_conv:
        mod.weight = nn.Parameter(
            torch.zeros((local_out, 1, in_features), dtype=torch.bfloat16),
            requires_grad=False,
        )
    else:
        shape = (in_features, local_out) if flipped else (local_out, in_features)
        mod.weight = nn.Parameter(
            torch.zeros(shape, dtype=torch.bfloat16),
            requires_grad=False,
        )
        if flipped:
            setattr(mod, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    return mod


def _make_routed_experts(
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    *,
    use_ep: bool,
    ep_size: int,
    moe_tp_size: int,
) -> RoutedExperts:
    mod = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(mod)
    mod.quant_method = None
    mod.expert_placement_strategy = "linear"
    mod.activation = "silu"
    mod.moe_config = SimpleNamespace(
        num_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=SimpleNamespace(
            use_ep=use_ep,
            ep_size=ep_size if use_ep else 1,
            tp_size=1 if use_ep else moe_tp_size,
            enable_eplb=False,
        ),
    )
    mod.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
    setattr(mod, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    local_e = num_experts // ep_size if use_ep else num_experts
    half = inter_size if use_ep else (inter_size // moe_tp_size)
    aligned_half = (half + 127) // 128 * 128
    mod.w13_weight = nn.Parameter(
        torch.zeros((local_e, hidden_size, 2 * aligned_half), dtype=torch.bfloat16),
        requires_grad=False,
    )
    mod.w2_weight = nn.Parameter(
        torch.zeros((local_e, aligned_half, hidden_size), dtype=torch.bfloat16),
        requires_grad=False,
    )
    setattr(mod.w13_weight, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    setattr(mod.w2_weight, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)
    return mod


class _FakeMambaLayer(MambaBase, nn.Module):
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32, torch.float32)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((1,), (1,))

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def __init__(self, tp_size: int, num_v_heads: int = 8) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        self.A_log = nn.Parameter(
            torch.zeros(num_v_heads // tp_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(num_v_heads // tp_size, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _CustomMergedLinear(MergedColumnParallelLinear):
    """Unwhitelisted `LinearBase` subclass to verify strict fail-closed whitelist."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.tp_size = 2
        self.disable_tp = False
        self.quant_method = None
        self.weight = nn.Parameter(
            torch.zeros((16, 8), dtype=torch.bfloat16),
            requires_grad=False,
        )


@contextmanager
def _patched_ws(
    endpoints: list[dict[str, object]] | None = None,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    inst = MagicMock()
    inst.num_shards = 1
    inst.local_port = 19500
    inst.listener_port = 29500
    inst.get_local_endpoints.return_value = endpoints or [
        {"endpoint": "[::]:19500", "shards": [0]}
    ]
    inst.get_metrics.return_value = {"bytes": 1024}
    cls_mock = MagicMock(return_value=inst)
    with (
        patch.object(
            raiden_worker_sync,
            "_ws_lib",
            SimpleNamespace(WeightSynchronizer=cls_mock),
        ),
        patch.object(raiden_worker_sync, "_check_bindable"),
        patch("torch.accelerator.synchronize"),
    ):
        yield cls_mock, inst


def _make_ep_test_model() -> nn.Module:
    model = nn.Module()
    model.embed_tokens = _make_vocab_embedding(
        vocab_size=64,
        hidden_size=16,
        tp_size=4,
    )
    model.experts = _make_routed_experts(
        num_experts=16,
        hidden_size=16,
        inter_size=128,
        use_ep=True,
        ep_size=8,
        moe_tp_size=1,
    )
    model.norm = _make_rms_norm(16)
    model.norm.weight.data.fill_(2.0)
    return model


def _assert_raiden_metadata_invariants(meta: dict) -> None:
    phys = dict(zip(meta["mesh_axes"], meta["mesh_shape"]))
    assert len(phys) == len(meta["mesh_axes"]) == len(meta["mesh_shape"])
    assert meta["unit"]["job_replica_id"] != ""
    assert len(meta["shards"]) >= 1
    for idx, v in enumerate(meta["variables"]):
        assert v["layer_idx"] == idx
        assert len(v["global_shard_indices"]) == len(meta["shards"])
        for dim, axis in enumerate(v["sharding_spec"]):
            expected = math.prod(phys[p] for p in axis.split(",")) if axis else 1
            assert v["mesh_shape"][dim] == expected


def test_canonical_param_name_and_sort_order() -> None:
    for raw, expected in (
        ("language_model.model.embed_tokens.weight", "embed_tokens.weight"),
        (
            "model.layers.3.mlp.experts.routed_experts.w13_weight",
            "layers.3.mlp.experts.w13_weight",
        ),
        (
            "base.model.layers_10.blocks_2.experts_3.w13_weight.value",
            "layers.10.blocks.2.experts.3.w13_weight",
        ),
        (
            "model.layers.0.post_feedforward_layernorm_1.weight",
            "layers.0.post_feedforward_layernorm_1.weight",
        ),
    ):
        assert raiden_worker_sync.canonical_param_name(raw) == expected

    gemma_layer = nn.Module()
    gemma_layer.post_feedforward_layernorm_1 = _make_rms_norm(8)
    gemma_layer.post_feedforward_layernorm = _make_rms_norm(8)
    gemma_root = nn.Module()
    gemma_root.layers = nn.ModuleDict({"0": gemma_layer})
    assert [
        v.name for v in raiden_worker_sync.collect_variables(gemma_root, tp_size=2)
    ] == [
        "layers.0.post_feedforward_layernorm.weight",
        "layers.0.post_feedforward_layernorm_1.weight",
    ]

    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = _make_vocab_embedding(
        vocab_size=32,
        hidden_size=8,
        tp_size=2,
    )
    lm.model.layers = nn.ModuleDict({"2": _make_rms_norm(8), "10": _make_rms_norm(8)})
    lm.lm_head = _make_lm_head(vocab_size=32, hidden_size=8, tp_size=2)
    assert [v.name for v in raiden_worker_sync.collect_variables(lm, tp_size=2)] == [
        "embed_tokens.weight",
        "layers.10.weight",
        "layers.2.weight",
        "lm_head.weight",
    ]


def test_vlm_tower_mtp_skipping_and_tied_lm_head() -> None:
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = _make_vocab_embedding(
        vocab_size=32,
        hidden_size=8,
        tp_size=2,
    )
    lm.lm_head = _make_lm_head(vocab_size=32, hidden_size=8, tp_size=2)
    # Distinct nn.Parameter sharing storage exercises `seen_ptrs` deduplication.
    lm.lm_head.weight = nn.Parameter(
        lm.model.embed_tokens.weight.data,
        requires_grad=False,
    )
    lm.mtp = nn.Linear(8, 8)

    vlm = nn.Module()
    vlm.language_model = lm
    vision = nn.Linear(8, 8)
    vision._is_tower_model = True
    vlm.vision_tower = vision
    vlm.get_language_model = lambda: lm
    assert [v.name for v in raiden_worker_sync.collect_variables(vlm, tp_size=2)] == [
        "embed_tokens.weight"
    ]

    # Zero-pointer (unmaterialized) tensors must not be deduplicated as tied.
    zero_mod = nn.Module()
    zero_mod.a = _make_rms_norm(8)
    zero_mod.b = _make_rms_norm(8)
    with patch.object(torch.Tensor, "data_ptr", return_value=0):
        names = [
            v.name for v in raiden_worker_sync.collect_variables(zero_mod, tp_size=1)
        ]
    assert names == ["a.weight", "b.weight"]


def test_owning_module_classification_and_param_layouts() -> None:
    root = nn.Module()
    root.dp_embed = _make_vocab_embedding(
        vocab_size=64,
        hidden_size=8,
        tp_size=8,
    )
    root.dp_embed._tpu_dp_vocab = True
    root.rep_flipped = _make_replicated_linear(
        in_features=16,
        out_features=4,
        flipped=True,
    )
    gate_lin = GateLinear.__new__(GateLinear)
    nn.Module.__init__(gate_lin)
    gate_lin.tp_size = 1
    gate_lin.disable_tp = False
    gate_lin.quant_method = None
    gate_lin.weight = nn.Parameter(
        torch.zeros((8, 16), dtype=torch.bfloat16),
        requires_grad=False,
    )
    root.router_gate = gate_lin
    root.row_flipped = _make_row_linear(
        in_features=16,
        out_features=8,
        tp_size=4,
        flipped=True,
        bias=True,
    )
    root.row_unflipped = _make_row_linear(
        in_features=16,
        out_features=8,
        tp_size=4,
        flipped=False,
    )
    root.qkv = _make_qkv_linear(
        hidden_size=16,
        head_size=4,
        total_q_heads=8,
        total_kv_heads=2,
        tp_size=4,
    )
    root.qkv.bias = nn.Parameter(
        torch.zeros(root.qkv.weight.shape[1], dtype=torch.bfloat16),
        requires_grad=False,
    )
    root.gdn_unflipped = _make_gdn_column_linear(
        in_features=16,
        checkpoint_sizes=[8, 8, 16, 16],
        replication_factors=[2, 2, 1, 1],
        tp_size=4,
        flipped=False,
    )
    root.conv = _make_gdn_column_linear(
        in_features=4,
        checkpoint_sizes=[8, 8, 16],
        replication_factors=[2, 2, 1],
        tp_size=4,
        is_conv=True,
    )
    root.mamba = _FakeMambaLayer(tp_size=4, num_v_heads=8)
    root.moe_tp = _make_routed_experts(
        num_experts=4,
        hidden_size=256,
        inter_size=768,
        use_ep=False,
        ep_size=1,
        moe_tp_size=4,
    )
    mixer_norm = nn.Module()
    mixer_norm.weight = nn.Parameter(
        torch.ones(16, dtype=torch.bfloat16),
        requires_grad=False,
    )
    set_weight_attrs(mixer_norm.weight, {"weight_loader": sharded_weight_loader(0)})
    root.sharded_norm = mixer_norm

    by_name = {v.name: v for v in raiden_worker_sync.collect_variables(root, tp_size=4)}
    assert by_name["dp_embed.weight"].shard_domain == "ep"
    assert by_name["dp_embed.weight"].num_shards == 8
    assert by_name["rep_flipped.weight"].shard_dim is None
    assert by_name["rep_flipped.weight"].flipped is True
    assert by_name["router_gate.weight"].shard_dim is None
    assert by_name["router_gate.weight"].flipped is False
    assert by_name["row_flipped.weight"].shard_dim == 0
    assert by_name["row_flipped.bias"].shard_dim is None
    assert by_name["row_unflipped.weight"].shard_dim == 1
    assert by_name["qkv.weight"].shard_dim == 1
    assert by_name["qkv.weight"].replication_factors == (1, 2, 2)
    assert by_name["qkv.bias"].shard_dim == 0
    assert by_name["qkv.bias"].replication_factors == (1, 2, 2)
    assert by_name["gdn_unflipped.weight"].shard_dim == 0
    assert by_name["conv.weight"].conv_transposed is False
    assert by_name["conv.weight"].replication_factors == (2, 2, 1)
    assert by_name["mamba.A_log"].shard_dim == 0
    assert by_name["sharded_norm.weight"].global_shape == (64,)
    assert by_name["moe_tp.w13_weight"].global_shape == (4, 256, 2048)
    assert by_name["moe_tp.w2_weight"].global_shape == (4, 1024, 256)

    tp4_root = nn.Module()
    tp4_root.qkv = root.qkv
    tp4_root.conv = root.conv
    with _patched_ws():
        ws = raiden_worker_sync.RaidenWorkerSync(
            "rollout",
            tp_rank=0,
            tp_size=4,
        )
        ws.bind(tp4_root)
        layouts = ws.param_layouts()
        for var_d in ws.variable_dicts():
            assert layouts[var_d["name"]]["layer_idx"] == var_d["layer_idx"]
        assert layouts["conv.weight"]["conv_transposed"] is False
        assert layouts["conv.weight"]["replication_factors"] == [2, 2, 1]
        assert layouts["qkv.weight"]["replication_factors"] == [1, 2, 2]


def _build_unsupported_case(kind: str) -> tuple[nn.Module, int]:
    m = nn.Module()
    if kind == "padded_vocab_tp1":
        m.emb = _make_vocab_embedding(
            vocab_size=151680,
            hidden_size=8,
            tp_size=1,
            org_vocab_size=151665,
        )
        return m, 1
    if kind == "padded_vocab_tp2":
        m.emb = _make_vocab_embedding(
            vocab_size=32,
            hidden_size=8,
            tp_size=2,
            org_vocab_size=30,
        )
        return m, 2
    if kind == "col_scale":
        m.qkv = _make_qkv_linear(
            hidden_size=16,
            head_size=4,
            total_q_heads=8,
            total_kv_heads=2,
            tp_size=4,
        )
        m.qkv.weight_scale = PerTensorScaleParameter(
            data=torch.ones(1, dtype=torch.float32),
            weight_loader=lambda *_: None,
        )
        return m, 4
    if kind == "row_scale":
        m.o = _make_row_linear(in_features=16, out_features=8, tp_size=4)
        m.o.weight_scale_inv = BlockQuantScaleParameter(
            data=torch.ones((1, 2, 1, 8), dtype=torch.float32),
            input_dim=1,
            output_dim=3,
            weight_loader=lambda *_: None,
        )
        return m, 4
    if kind == "moe_scale":
        m.experts = _make_routed_experts(
            num_experts=8,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=4,
            moe_tp_size=1,
        )
        m.experts.w13_weight_scale = nn.Parameter(torch.ones((2, 1, 256)))
        return m, 4
    if kind == "transformed_loader_tp1":
        m.A = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(
            m.A,
            {
                "weight_loader": composed_weight_loader(
                    sharded_weight_loader(0),
                    lambda x: -torch.exp(x.float()),
                )
            },
        )
        return m, 1
    if kind == "unrecognized_vllm_param":
        m.w = ModelWeightParameter(
            data=torch.zeros((4, 8), dtype=torch.bfloat16),
            input_dim=1,
            output_dim=0,
            weight_loader=None,
        )
        return m, 4
    if kind == "unwhitelisted_linear_subclass":
        m.proj = _CustomMergedLinear()
        return m, 2
    if kind == "redundant_experts":
        m.experts = _make_routed_experts(
            num_experts=6,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=2,
            moe_tp_size=1,
        )
        m.experts.moe_config.num_logical_experts = 4
        return m, 2
    if kind == "enable_eplb":
        m.experts = _make_routed_experts(
            num_experts=4,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=2,
            moe_tp_size=1,
        )
        m.experts.moe_config.moe_parallel_config.enable_eplb = True
        return m, 2
    if kind == "fused_shared":
        m.experts = _make_routed_experts(
            num_experts=4,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=2,
            moe_tp_size=1,
        )
        m.experts.expert_map_manager = SimpleNamespace(num_fused_shared_experts=1)
        return m, 2
    if kind == "round_robin":
        m.experts = _make_routed_experts(
            num_experts=4,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=2,
            moe_tp_size=1,
        )
        m.experts.expert_placement_strategy = "round_robin"
        return m, 2
    if kind == "swigluoai":
        m.experts = _make_routed_experts(
            num_experts=4,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=2,
            moe_tp_size=1,
        )
        m.experts.activation = "swigluoai"
        return m, 2
    if kind == "quantized_linear":
        m.proj = _make_row_linear(in_features=16, out_features=8, tp_size=2)
        m.proj.quant_method = object()
        return m, 2
    if kind == "quantized_moe":
        m.experts = _make_routed_experts(
            num_experts=4,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=2,
            moe_tp_size=1,
        )
        m.experts.quant_method = object()
        return m, 2
    if kind == "fp8_dtype":
        m.w = nn.Parameter(
            torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        return m, 1
    if kind == "int8_dtype":
        m.w = nn.Parameter(
            torch.zeros((8, 8), dtype=torch.int8),
            requires_grad=False,
        )
        return m, 1
    if kind == "lora_wrapper":
        wrap = nn.Module()
        wrap.base_layer = _make_replicated_linear(in_features=4, out_features=4)
        wrap.lora_A = nn.Parameter(torch.zeros(2, 4, dtype=torch.bfloat16))
        m.qkv = wrap
        return m, 1
    if kind == "closure_linear_loader":
        m.qkv = _make_qkv_linear(
            hidden_size=16,
            head_size=4,
            total_q_heads=8,
            total_kv_heads=2,
            tp_size=4,
        )
        m.qkv.weight.weight_loader = lambda *_: None  # type: ignore[attr-defined]
        return m, 4
    if kind == "mla_w_uk_t":
        m.W_UK_T = nn.Parameter(torch.zeros((4, 8, 16), dtype=torch.bfloat16))
        return m, 2
    if kind == "quantized_vocab":
        m.emb = _make_vocab_embedding(vocab_size=32, hidden_size=8, tp_size=2)
        m.emb.quant_method = object()
        return m, 2
    if kind == "hybrid_ep_tp_moe":
        m.experts = _make_routed_experts(
            num_experts=8,
            hidden_size=16,
            inter_size=128,
            use_ep=True,
            ep_size=4,
            moe_tp_size=1,
        )
        m.experts.moe_config.moe_parallel_config.tp_size = 2
        return m, 4
    raise AssertionError(f"Unknown case: {kind}")


@pytest.mark.parametrize(
    "kind, match",
    [
        ("padded_vocab_tp1", "padded vocab"),
        ("padded_vocab_tp2", "padded vocab"),
        ("col_scale", "weight_scale"),
        ("row_scale", "weight_scale_inv"),
        ("moe_scale", "RoutedExperts"),
        ("transformed_loader_tp1", "weight_loader"),
        ("unrecognized_vllm_param", "sharded parameter"),
        ("unwhitelisted_linear_subclass", "unrecognized layer subclass"),
        ("redundant_experts", "redundant|EPLB"),
        ("enable_eplb", "EPLB"),
        ("fused_shared", "fused shared"),
        ("round_robin", "expert_placement_strategy"),
        ("swigluoai", "swigluoai"),
        ("quantized_linear", "quantized LinearBase"),
        ("quantized_moe", "quantized MoE"),
        ("quantized_vocab", "quantized VocabParallelEmbedding"),
        ("hybrid_ep_tp_moe", "hybrid EP\\+TP"),
        ("fp8_dtype", "dtype"),
        ("int8_dtype", "dtype"),
        ("lora_wrapper", "LoRA|base_layer"),
        ("closure_linear_loader", "weight_loader"),
        ("mla_w_uk_t", "W_UK_T"),
    ],
)
def test_classify_rejects_unsupported_layouts(kind: str, match: str) -> None:
    with patch(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(rank_in_group=0, world_size=4),
    ):
        model, tp_size = _build_unsupported_case(kind)
        with pytest.raises(NotImplementedError, match=match):
            raiden_worker_sync.collect_variables(model, tp_size=tp_size)


def test_worker_sync_bind_metadata_and_checksums() -> None:
    model = _make_ep_test_model()
    with _patched_ws() as (ws_cls, _):
        ws = raiden_worker_sync.RaidenWorkerSync(
            "replica_0",
            worker_index=0,
            tp_rank=0,
            tp_size=4,
            dp_rank=0,
            dp_size=2,
            bind_ip="[2001:db8::1]",
            auto_h2d=False,
        )
        assert not ws.bound
        ws.bind(model)
        assert ws.bound
        assert ws_cls.call_args.kwargs["global_shard_indices"] == [0]
        meta = ws.metadata_dict()
        _assert_raiden_metadata_invariants(meta)
        assert meta["shards"] == ["[2001:db8::1]:19500"]
        assert "host_subgrid" not in meta
        assert ws.checksums(sample=2)["__grand_total__"] == pytest.approx(32.0)


def test_worker_sync_rebind_h2d_uuid_and_shutdown() -> None:
    model = _make_ep_test_model()
    with _patched_ws() as (_, ws_inst):
        ws = raiden_worker_sync.RaidenWorkerSync(
            "replica_0",
            tp_rank=0,
            tp_size=4,
            dp_rank=0,
            dp_size=2,
            auto_h2d=False,
        )
        ws.bind(model)

        # Rebinding preserves initial auto_h2d; new Parameter calls bind_weights.
        ws.bind(model, auto_h2d=True)
        assert ws.metadata_dict()["auto_h2d"] is False
        model.norm.weight = nn.Parameter(torch.ones_like(model.norm.weight))
        ws.bind(model)
        ws_inst.bind_weights.assert_called_once()

        # Rebinding same tensor count with changed dtype recreates WeightSynchronizer.
        model.norm.weight = nn.Parameter(
            torch.ones(16, dtype=torch.float32),
            requires_grad=False,
        )
        ws.bind(model)
        assert ws_inst.shutdown.call_count == 1

        for bad_uuid in (None, 0, -1, True):
            with pytest.raises(ValueError, match="uuid"):
                ws.h2d(uuid=bad_uuid)  # type: ignore[arg-type]
        ws.h2d(uuid=3)
        ws_inst.wait_for_transfer_completion.assert_called_once_with(3)
        ws_inst.h2d.assert_called_once()

        ws_auto = raiden_worker_sync.RaidenWorkerSync(
            "replica_0",
            tp_rank=0,
            tp_size=4,
            dp_rank=0,
            dp_size=2,
            auto_h2d=True,
        )
        ws_auto.bind(model)
        with pytest.raises(ValueError, match="uuid"):
            ws_auto.h2d(uuid=None)
        ws_auto.shutdown()
        assert not ws_auto.bound
        assert ws_inst.shutdown.call_count == 2


def test_worker_sync_dp_ep_rank_indexing_and_shard_mismatch_guard() -> None:
    model = _make_ep_test_model()
    with _patched_ws() as (ws_cls, _):
        ws_13 = raiden_worker_sync.RaidenWorkerSync(
            "replica_0",
            tp_rank=3,
            tp_size=4,
            dp_rank=1,
            dp_size=2,
            bind_ip="10.0.0.1",
        )
        ws_13.bind(model)
        meta_13 = ws_13.metadata_dict()
        _assert_raiden_metadata_invariants(meta_13)
        ep_var = next(
            v for v in meta_13["variables"] if v["name"] == "experts.w13_weight"
        )
        assert ep_var["sharding_spec"] == ["dp,tp", "", ""]
        assert ep_var["global_shard_indices"] == [7]

        # Sub-mesh shard mismatch fails before allocating WeightSynchronizer.
        ws_cls.reset_mock()
        bad_moe = nn.Module()
        bad_moe.experts = _make_routed_experts(
            num_experts=4,
            hidden_size=16,
            inter_size=128,
            use_ep=False,
            ep_size=1,
            moe_tp_size=2,
        )
        ws_bad = raiden_worker_sync.RaidenWorkerSync(
            "bad",
            tp_rank=0,
            tp_size=4,
        )
        with pytest.raises(NotImplementedError, match="TP shard count 2"):
            ws_bad.bind(bad_moe)
        assert ws_cls.call_count == 0


def test_endpoint_striping_host_fallback_and_topology_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for kwargs in (
        {"dp_rank": 2, "dp_size": 2},
        {"dp_rank": -1, "dp_size": 2},
        {"tp_rank": 4, "tp_size": 4},
        {"dp_rank": 0, "dp_size": 0},
        {"parallelism": 0},
        {"worker_index": -1},
    ):
        with pytest.raises(ValueError):
            raiden_worker_sync.RaidenWorkerSync("bad", **kwargs)

    model = nn.Module()
    model.norm = _make_rms_norm(8)
    nics = [
        {"endpoint": "[2001:db8::10]:19500", "shards": [0]},
        {"endpoint": "[2001:db8::11]:19500", "shards": [0]},
    ]
    with (
        _patched_ws(nics),
        patch.object(
            raiden_worker_sync,
            "local_ip",
            side_effect=AssertionError("local_ip() should not run for concrete NICs"),
        ),
    ):
        ws_0 = raiden_worker_sync.RaidenWorkerSync(
            "rollout",
            tp_rank=0,
            tp_size=2,
            dp_rank=0,
            dp_size=2,
        )
        ws_1 = raiden_worker_sync.RaidenWorkerSync(
            "rollout",
            tp_rank=1,
            tp_size=2,
            dp_rank=0,
            dp_size=2,
        )
        ws_0.bind(model)
        ws_1.bind(model)
        assert ws_0.metadata_dict()["shards"] == ["[2001:db8::10]:19500"]
        assert ws_1.metadata_dict()["shards"] == ["[2001:db8::11]:19500"]

    monkeypatch.setattr(raiden_worker_sync.envs, "RAIDEN_BIND_IP", "")
    with (
        _patched_ws([{"endpoint": "0.0.0.0:19500", "shards": [0]}]),
        patch.object(raiden_worker_sync, "get_ip", return_value="10.20.30.40"),
    ):
        ws_wc = raiden_worker_sync.RaidenWorkerSync("rollout")
        ws_wc.bind(model)
        assert ws_wc.metadata_dict()["shards"] == ["10.20.30.40:19500"]


def test_moe_flip_idempotency_reload_and_env_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert {
        "WEIGHT_SYNC_PARALLEL_H2H",
        "RAIDEN_PARALLELISM",
        "RAIDEN_BIND_IP",
        "VERIFY_WEIGHTS",
        "TPU_RAIDEN_DATA_NICS",
    }.issubset(set(TpuPlatform.additional_env_vars))

    monkeypatch.setenv("RAIDEN_PARALLELISM", "0")
    with pytest.raises(ValueError, match="RAIDEN_PARALLELISM"):
        _ = raiden_worker_sync.envs.environment_variables["RAIDEN_PARALLELISM"]()
    monkeypatch.delenv("RAIDEN_PARALLELISM", raising=False)

    monkeypatch.setattr(raiden_worker_sync.envs, "TPU_EXPERT_PLACEMENT", "/tmp/p.json")
    simple = nn.Module()
    simple.norm = _make_rms_norm(8)
    with pytest.raises(NotImplementedError, match="TPU_EXPERT_PLACEMENT"):
        raiden_worker_sync.collect_variables(simple, tp_size=1)
    monkeypatch.setattr(raiden_worker_sync.envs, "TPU_EXPERT_PLACEMENT", None)

    layer = _make_routed_experts(
        num_experts=4,
        hidden_size=16,
        inter_size=64,
        use_ep=True,
        ep_size=2,
        moe_tp_size=1,
    )
    delattr(layer, raiden_worker_sync.WEIGHT_FLIPPED_ATTR)
    layer.moe_config.experts_per_token = 2
    layer.moe_config.moe_parallel_config.use_ep = False
    layer.w13_weight = nn.Parameter(
        torch.randn((2, 128, 16), dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.w2_weight = nn.Parameter(
        torch.randn((2, 16, 64), dtype=torch.bfloat16),
        requires_grad=False,
    )
    method = VllmUnquantizedFusedMoEMethod.__new__(VllmUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(has_bias=False)

    unquant_mod = "vllm_torchtpu.layers.adapter.quantization.unquantized"
    with (
        patch(f"{unquant_mod}.moe_routing"),
        patch(f"{unquant_mod}.prebuild_fused_moe_kernel"),
    ):
        method.process_weights_after_loading(layer)
        assert layer.w13_weight.shape == (2, 16, 256)
        once_w13 = layer.w13_weight.detach().clone()
        method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.w13_weight, once_w13)

        # Layerwise reload replaces Parameter objects on the already-flipped module.
        layer.w13_weight = nn.Parameter(
            torch.ones((2, 128, 16), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w2_weight = nn.Parameter(
            torch.ones((2, 16, 64), dtype=torch.bfloat16),
            requires_grad=False,
        )
        method.process_weights_after_loading(layer)
        assert layer.w13_weight.shape == (2, 16, 256)
        assert layer.w2_weight.shape == (2, 128, 16)
