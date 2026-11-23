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
"""CPU unit tests for TPUWorker RL weight conversion RPCs."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_torchtpu.rl.manifest_generator import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.rl.weight_conversion import SCHEMA_VERSION, validate_manifest
from vllm_torchtpu.rl.weight_installer import (
    ShardingCoords,
    extract_sharding_map,
    slice_tensor_for_shard,
    stage_converted_weights,
)


def _import_tpu_worker():
    """Import TPUWorker on CPU dev hosts without torch_tpu, restoring globals."""
    if importlib.util.find_spec("torch_tpu") is not None:
        from vllm_torchtpu.worker.tpu_worker import TPUWorker as _TPUWorker

        return _TPUWorker

    import vllm.platforms as _vllm_platforms

    had_platform = hasattr(_vllm_platforms, "_current_platform")
    prev_platform = getattr(_vllm_platforms, "_current_platform", None)
    stubbed_modules: list[str] = []
    try:
        if prev_platform is None:
            from vllm.platforms.interface import UnspecifiedPlatform

            _vllm_platforms._current_platform = UnspecifiedPlatform()

        for mod_name in (
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.profiler",
            "torch_tpu._internal.sync",
        ):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
                stubbed_modules.append(mod_name)
        sys.modules["torch_tpu._internal.profiler"].TpuProfilerConfig = object
        sys.modules["torch_tpu._internal.sync"].synchronize = lambda *a, **kw: None

        for pkg in ("vllm_torchtpu.layers", "vllm_torchtpu.layers.adapter"):
            if pkg not in sys.modules:
                m = types.ModuleType(pkg)
                m.__path__ = []
                sys.modules[pkg] = m
                stubbed_modules.append(pkg)

        if "vllm_torchtpu.layers.adapter.attention" not in sys.modules:
            attn_mod = types.ModuleType("vllm_torchtpu.layers.adapter.attention")
            attn_mod.TPU_STR_DTYPE_TO_TORCH_DTYPE = {}
            sys.modules["vllm_torchtpu.layers.adapter.attention"] = attn_mod
            stubbed_modules.append("vllm_torchtpu.layers.adapter.attention")

        if "vllm_torchtpu.runner" not in sys.modules:
            rm = types.ModuleType("vllm_torchtpu.runner")
            rm.__path__ = []
            sys.modules["vllm_torchtpu.runner"] = rm
            stubbed_modules.append("vllm_torchtpu.runner")

        if "vllm_torchtpu.runner.tpu_pooling" not in sys.modules:
            pooling_mod = types.ModuleType("vllm_torchtpu.runner.tpu_pooling")
            pooling_mod.compute_pooling_kv_cache_cap_bytes = lambda *a, **kw: None
            sys.modules["vllm_torchtpu.runner.tpu_pooling"] = pooling_mod
            stubbed_modules.append("vllm_torchtpu.runner.tpu_pooling")

        if "vllm_torchtpu.runner.tpu_runner" not in sys.modules:
            runner_mod = types.ModuleType("vllm_torchtpu.runner.tpu_runner")
            runner_mod.TPUModelRunner = object
            sys.modules["vllm_torchtpu.runner.tpu_runner"] = runner_mod
            stubbed_modules.append("vllm_torchtpu.runner.tpu_runner")

        from vllm_torchtpu.worker.tpu_worker import TPUWorker as _TPUWorker

        return _TPUWorker
    finally:
        for mod_name in stubbed_modules:
            sys.modules.pop(mod_name, None)
        if had_platform:
            _vllm_platforms._current_platform = prev_platform


TPUWorker = _import_tpu_worker()


# ---------------------------------------------------------------------------
# Lightweight synthetic vLLM/TorchTPU layers for CPU unit tests
# ---------------------------------------------------------------------------


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, tp_size: int = 1):
        super().__init__()
        assert num_embeddings % tp_size == 0
        self.org_vocab_size = num_embeddings
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.weight = nn.Parameter(
            torch.zeros(num_embeddings // tp_size, embedding_dim, dtype=torch.bfloat16)
        )


class ParallelLMHead(VocabParallelEmbedding):
    pass


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.bfloat16))


class QKVParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        tp_size: int = 1,
        bias: bool = True,
        flipped: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.num_kv_head_replicas = max(1, tp_size // total_num_kv_heads)
        local_out = (self.num_heads + 2 * self.num_kv_heads) * head_size
        shape = (hidden_size, local_out) if flipped else (local_out, hidden_size)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
        if flipped:
            setattr(self, WEIGHT_FLIPPED_ATTR, True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(local_out, dtype=torch.bfloat16))


class MergedColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        *,
        tp_size: int = 1,
        flipped: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = list(output_sizes)
        local_out = sum(output_sizes) // tp_size
        shape = (input_size, local_out) if flipped else (local_out, input_size)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
        if flipped:
            setattr(self, WEIGHT_FLIPPED_ATTR, True)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        tp_size: int = 1,
        flipped: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = input_size // tp_size
        local_in = input_size // tp_size
        shape = (local_in, output_size) if flipped else (output_size, local_in)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
        if flipped:
            setattr(self, WEIGHT_FLIPPED_ATTR, True)


class _TinyQwenLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inter_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        tp_size: int = 1,
        flipped: bool = False,
    ):
        super().__init__()
        head_size = hidden_size // num_heads
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_size,
            num_heads,
            num_kv_heads,
            tp_size=tp_size,
            bias=True,
            flipped=flipped,
        )
        self.self_attn.o_proj = RowParallelLinear(
            hidden_size, hidden_size, tp_size=tp_size, flipped=flipped
        )
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [inter_size, inter_size], tp_size=tp_size, flipped=flipped
        )
        self.mlp.down_proj = RowParallelLinear(
            inter_size, hidden_size, tp_size=tp_size, flipped=flipped
        )
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)


class _TinyQwenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 16,
        inter_size: int = 32,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        *,
        tp_size: int = 1,
        tie_word_embeddings: bool = True,
        flipped: bool = False,
    ):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            vocab_size, hidden_size, tp_size=tp_size
        )
        self.model.layers = nn.ModuleList(
            [
                _TinyQwenLayer(
                    hidden_size,
                    inter_size,
                    num_heads,
                    num_kv_heads,
                    tp_size=tp_size,
                    flipped=flipped,
                )
            ]
        )
        self.model.norm = RMSNorm(hidden_size)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size, tp_size=tp_size)
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


class _SimpleParamModel(nn.Module):
    def __init__(self, params: dict[str, torch.Tensor]):
        super().__init__()
        for name, tensor in params.items():
            self.register_parameter(name, nn.Parameter(tensor.clone()))


def _make_worker(
    model: nn.Module,
    *,
    rank: int = 0,
    tp_size: int = 1,
    dp_size: int = 1,
    dp_rank: int | None = None,
    enable_ep: bool = False,
    ep_size: int = 0,
    model_config: object | None = None,
) -> TPUWorker:
    worker = TPUWorker.__new__(TPUWorker)
    worker.rank = rank
    worker.local_rank = rank
    worker.parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=enable_ep,
        expert_parallel_size=ep_size,
    )
    worker.model_config = model_config
    worker.model_runner = MagicMock()
    worker.model_runner.get_model.return_value = model
    return worker


def _make_manifest(
    rules: list[dict],
    *,
    stripes: dict[str, int] | None = None,
    tied_keys: list[list[str]] | None = None,
    expert_axis_grouped: bool = False,
) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "expert_axis_grouped": expert_axis_grouped,
            "stripes": stripes or {},
            "tied_keys": tied_keys or [],
            "dropped_keys": [],
            "rules": rules,
        }
    )


# ---------------------------------------------------------------------------
# 1. get_weight_mapping_manifest()
# ---------------------------------------------------------------------------


def test_get_weight_mapping_manifest_returns_valid_json() -> None:
    model = _TinyQwenModel(tp_size=1, tie_word_embeddings=True)
    worker = _make_worker(model, tp_size=1)

    manifest_json = worker.get_weight_mapping_manifest()
    assert isinstance(manifest_json, str)

    manifest = json.loads(manifest_json)
    validate_manifest(manifest)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert ["model.embed_tokens.weight", "lm_head.weight"] in manifest["tied_keys"]
    target_keys = {k for r in manifest["rules"] for k in r["target_keys"]}
    assert "model.layers.0.self_attn.qkv_proj.weight" in target_keys
    assert "model.layers.0.self_attn.qkv_proj.bias" in target_keys
    assert "model.layers.0.mlp.gate_up_proj.weight" in target_keys


# ---------------------------------------------------------------------------
# 2. apply_converted_weights() sharding & slicing
# ---------------------------------------------------------------------------


def test_apply_converted_weights_replicated_preserves_buffer_identity() -> None:
    model = _SimpleParamModel(
        {
            "norm": torch.zeros(8, dtype=torch.bfloat16),
            "gate": torch.zeros(4, 8, dtype=torch.bfloat16),
        }
    )
    norm_ptr = model.norm.data_ptr()
    gate_ptr = model.gate.data_ptr()
    worker = _make_worker(model, rank=1, tp_size=2)

    manifest = _make_manifest(
        [
            {
                "source_keys": ["norm"],
                "target_keys": ["norm"],
                "ops": [],
                "sharding": [None],
            },
            {
                "source_keys": ["gate"],
                "target_keys": ["gate"],
                "ops": [],
                "sharding": [None, None],
            },
        ]
    )
    norm_val = torch.arange(8, dtype=torch.bfloat16)
    gate_val = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    worker.apply_converted_weights(
        {"norm": norm_val, "gate": gate_val}, manifest_json=manifest
    )

    assert torch.equal(model.norm.data, norm_val)
    assert torch.equal(model.gate.data, gate_val)
    assert model.norm.data_ptr() == norm_ptr
    assert model.gate.data_ptr() == gate_ptr


@pytest.mark.parametrize("tp_size", [2, 4])
def test_apply_converted_weights_tp_multi_rank(tp_size: int) -> None:
    col_global = torch.arange(16 * 8, dtype=torch.bfloat16).reshape(16, 8)
    row_global = torch.arange(8 * 16, dtype=torch.bfloat16).reshape(8, 16)
    bias_global = torch.arange(16, dtype=torch.bfloat16)
    norm_global = torch.arange(8, dtype=torch.bfloat16)

    manifest = _make_manifest(
        [
            {
                "source_keys": ["col"],
                "target_keys": ["col"],
                "ops": [],
                "sharding": ["tp", None],
            },
            {
                "source_keys": ["row"],
                "target_keys": ["row"],
                "ops": [],
                "sharding": [None, "tp"],
            },
            {
                "source_keys": ["bias"],
                "target_keys": ["bias"],
                "ops": [],
                "sharding": ["tp"],
            },
            {
                "source_keys": ["norm"],
                "target_keys": ["norm"],
                "ops": [],
                "sharding": [None],
            },
        ]
    )

    col_shards, row_shards, bias_shards = [], [], []
    for rank in range(tp_size):
        model = _SimpleParamModel(
            {
                "col": torch.zeros(16 // tp_size, 8, dtype=torch.bfloat16),
                "row": torch.zeros(8, 16 // tp_size, dtype=torch.bfloat16),
                "bias": torch.zeros(16 // tp_size, dtype=torch.bfloat16),
                "norm": torch.zeros(8, dtype=torch.bfloat16),
            }
        )
        worker = _make_worker(model, rank=rank, tp_size=tp_size)
        worker.apply_converted_weights(
            {
                "col": col_global,
                "row": row_global,
                "bias": bias_global,
                "norm": norm_global,
            },
            manifest_json=manifest,
        )
        col_shards.append(model.col.data.clone())
        row_shards.append(model.row.data.clone())
        bias_shards.append(model.bias.data.clone())
        assert torch.equal(model.norm.data, norm_global)

    assert torch.equal(torch.cat(col_shards, dim=0), col_global)
    assert torch.equal(torch.cat(row_shards, dim=1), row_global)
    assert torch.equal(torch.cat(bias_shards, dim=0), bias_global)


def test_apply_converted_weights_contiguous_gqa_with_kv_replication() -> None:
    """Verify contiguous TP slicing on interleaved GQA QKV with KV replication."""
    tp_size = 4
    # Pre-interleaved global QKV from chained 12-op rules:
    # For each rank r in 0..3, contiguous block r holds [Q_r | K_{r//2} | V_{r//2}]
    # where Q_r has 2 rows, K_{r//2} has 2 rows, V_{r//2} has 2 rows (total 6/rank).
    q = torch.arange(8 * 6, dtype=torch.bfloat16).reshape(8, 6)
    k = torch.arange(4 * 6, dtype=torch.bfloat16).reshape(4, 6) + 100
    v = torch.arange(4 * 6, dtype=torch.bfloat16).reshape(4, 6) + 200
    qkv_global = torch.cat(
        [
            torch.cat(
                [
                    q[r * 2 : (r + 1) * 2],
                    k[(r // 2) * 2 : (r // 2 + 1) * 2],
                    v[(r // 2) * 2 : (r // 2 + 1) * 2],
                ],
                dim=0,
            )
            for r in range(tp_size)
        ],
        dim=0,
    )

    manifest = _make_manifest(
        [
            {
                "source_keys": ["q", "k", "v"],
                "target_keys": ["qkv"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ],
        stripes={"tp_size": 4, "kv_tp_size": 2},
    )

    for rank in range(tp_size):
        model = _SimpleParamModel({"qkv": torch.zeros(6, 6, dtype=torch.bfloat16)})
        worker = _make_worker(model, rank=rank, tp_size=tp_size)
        worker.apply_converted_weights({"qkv": qkv_global}, manifest_json=manifest)

        expected = torch.cat(
            [
                q[rank * 2 : (rank + 1) * 2],
                k[(rank // 2) * 2 : (rank // 2 + 1) * 2],
                v[(rank // 2) * 2 : (rank // 2 + 1) * 2],
            ],
            dim=0,
        )
        assert torch.equal(model.qkv.data, expected)


def test_apply_converted_weights_ep_and_moe_tp_axes() -> None:
    """Verify pure EP, pure MoE-TP, and hybrid EP + MoE-TP slicing."""
    # Global 3D RoutedExperts tensor: [E=4, H=6, 2*I=8]
    w13_global = torch.arange(4 * 6 * 8, dtype=torch.bfloat16).reshape(4, 6, 8)
    # Global w2 tensor: [E=4, I=4, H=6]
    w2_global = torch.arange(4 * 4 * 6, dtype=torch.bfloat16).reshape(4, 4, 6)

    # Case 1: Pure MoE-TP (enable_ep=False, tp_size=2)
    manifest_tp = _make_manifest(
        [
            {
                "source_keys": ["w13"],
                "target_keys": ["w13"],
                "ops": [],
                "sharding": [None, None, "moe_tp"],
            },
            {
                "source_keys": ["w2"],
                "target_keys": ["w2"],
                "ops": [],
                "sharding": [None, "moe_tp", None],
            },
        ],
        expert_axis_grouped=True,
    )
    for rank in range(2):
        model = _SimpleParamModel(
            {
                "w13": torch.zeros(4, 6, 4, dtype=torch.bfloat16),
                "w2": torch.zeros(4, 2, 6, dtype=torch.bfloat16),
            }
        )
        worker = _make_worker(model, rank=rank, tp_size=2, enable_ep=False)
        worker.apply_converted_weights(
            {"w13": w13_global, "w2": w2_global}, manifest_json=manifest_tp
        )
        assert torch.equal(model.w13.data, w13_global[:, :, rank * 4 : (rank + 1) * 4])
        assert torch.equal(model.w2.data, w2_global[:, rank * 2 : (rank + 1) * 2, :])

    # Case 2: Pure EP (enable_ep=True, dp_size=1, tp_size=2 => ep_size=2, moe_tp=1)
    manifest_ep = _make_manifest(
        [
            {
                "source_keys": ["w13"],
                "target_keys": ["w13"],
                "ops": [],
                "sharding": ["ep", None, None],
            },
            {
                "source_keys": ["w2"],
                "target_keys": ["w2"],
                "ops": [],
                "sharding": ["ep", None, None],
            },
        ],
        expert_axis_grouped=True,
    )
    for rank in range(2):
        model = _SimpleParamModel(
            {
                "w13": torch.zeros(2, 6, 8, dtype=torch.bfloat16),
                "w2": torch.zeros(2, 4, 6, dtype=torch.bfloat16),
            }
        )
        worker = _make_worker(model, rank=rank, tp_size=2, enable_ep=True)
        worker.apply_converted_weights(
            {"w13": w13_global, "w2": w2_global}, manifest_json=manifest_ep
        )
        assert torch.equal(model.w13.data, w13_global[rank * 2 : (rank + 1) * 2])
        assert torch.equal(model.w2.data, w2_global[rank * 2 : (rank + 1) * 2])

    # Case 3: Hybrid EP + MoE-TP (dp_size=2, tp_size=2, ep_size=2 => moe_tp=2)
    manifest_hybrid = _make_manifest(
        [
            {
                "source_keys": ["w13"],
                "target_keys": ["w13"],
                "ops": [],
                "sharding": ["ep", None, "moe_tp"],
            },
            {
                "source_keys": ["w2"],
                "target_keys": ["w2"],
                "ops": [],
                "sharding": ["ep", "moe_tp", None],
            },
        ],
        expert_axis_grouped=True,
    )
    seen_shards: set[tuple[int, int]] = set()
    for dp_rank in range(2):
        for tp_rank in range(2):
            model = _SimpleParamModel(
                {
                    "w13": torch.zeros(2, 6, 4, dtype=torch.bfloat16),
                    "w2": torch.zeros(2, 2, 6, dtype=torch.bfloat16),
                }
            )
            worker = _make_worker(
                model,
                rank=tp_rank,
                tp_size=2,
                dp_size=2,
                dp_rank=dp_rank,
                enable_ep=True,
                ep_size=2,
            )
            worker.apply_converted_weights(
                {"w13": w13_global, "w2": w2_global},
                manifest_json=manifest_hybrid,
            )
            global_rank = dp_rank * 2 + tp_rank
            moe_tp_rank = global_rank % 2
            ep_rank = (global_rank // 2) % 2
            seen_shards.add((ep_rank, moe_tp_rank))
            assert torch.equal(
                model.w13.data,
                w13_global[
                    ep_rank * 2 : (ep_rank + 1) * 2,
                    :,
                    moe_tp_rank * 4 : (moe_tp_rank + 1) * 4,
                ],
            )
            assert torch.equal(
                model.w2.data,
                w2_global[
                    ep_rank * 2 : (ep_rank + 1) * 2,
                    moe_tp_rank * 2 : (moe_tp_rank + 1) * 2,
                    :,
                ],
            )
    assert seen_shards == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_apply_converted_weights_tied_keys_deduplicated() -> None:
    model = nn.Module()
    model.embed = nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16))
    model.lm_head = model.embed  # Tied by reference

    worker = _make_worker(model, rank=1, tp_size=2)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["embed"],
                "target_keys": ["embed"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ],
        tied_keys=[["embed", "lm_head"]],
    )
    global_embed = torch.arange(16 * 4, dtype=torch.bfloat16).reshape(16, 4)

    worker.apply_converted_weights(
        {"embed": global_embed, "lm_head": global_embed},
        manifest_json=manifest,
    )

    assert torch.equal(model.embed.data, global_embed[8:])
    assert model.lm_head is model.embed


def test_stage_converted_weights_canonical_tied_key_order_and_mismatch() -> None:
    """Primary unique_params key is canonical even if alias sorts earlier;
    a mismatched tied alias raises ValueError without mutating buffers."""
    model = nn.Module()
    # Register primary key 'z_primary' first, then tied alias 'a_alias'
    # which sorts lexicographically before 'z_primary'.
    model.register_parameter(
        "z_primary", nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
    )
    model.a_alias = model.z_primary

    manifest = _make_manifest(
        [
            {
                "source_keys": ["z_primary"],
                "target_keys": ["z_primary"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ],
        tied_keys=[["z_primary", "a_alias"]],
    )
    coords = ShardingCoords(tp_rank=1, tp_size=2)
    global_w = torch.arange(8 * 4, dtype=torch.bfloat16).reshape(8, 4)

    # Matching alias passes and stages from primary in canonical order.
    staged = stage_converted_weights(
        model, {"a_alias": global_w, "z_primary": global_w}, manifest, coords
    )
    assert len(staged) == 1
    assert staged[0][0] is model.z_primary
    assert torch.equal(staged[0][1], global_w[4:])

    # Mismatched alias must raise ValueError and leave model untouched.
    bad_alias = global_w.clone()
    bad_alias[4, 0] = -1.0
    worker = _make_worker(model, rank=1, tp_size=2)
    with pytest.raises(
        ValueError,
        match=(
            "Tied parameter alias 'a_alias' does not match "
            "primary parameter 'z_primary'"
        ),
    ):
        worker.apply_converted_weights(
            {"a_alias": bad_alias, "z_primary": global_w}, manifest_json=manifest
        )
    assert torch.equal(model.z_primary.data, torch.ones(4, 4, dtype=torch.bfloat16))


def test_weight_installer_helpers_direct() -> None:
    """Verify pure-PyTorch weight_installer helpers directly without worker stubs."""
    manifest_dict = {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": False,
        "stripes": {"tp_size": 2},
        "tied_keys": [["embed", "lm_head"]],
        "dropped_keys": [],
        "rules": [
            {
                "source_keys": ["embed"],
                "target_keys": ["embed"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ],
    }
    sharding_map = extract_sharding_map(manifest_dict)
    assert sharding_map == {"embed": ["tp", None], "lm_head": ["tp", None]}

    t = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    coords = ShardingCoords(tp_rank=1, tp_size=2)
    sliced = slice_tensor_for_shard("embed", t, ["tp", None], coords)
    assert torch.equal(sliced, t[2:])

    for bad_rank, bad_world in [(0, 0), (0, -2), (-1, 2), (2, 2)]:
        with pytest.raises(
            ValueError,
            match="Invalid tp sharding coordinates for parameter 'embed'",
        ):
            slice_tensor_for_shard(
                "embed",
                t,
                ["tp", None],
                ShardingCoords(tp_rank=bad_rank, tp_size=bad_world),
            )


# ---------------------------------------------------------------------------
# 3. Error handling & atomicity
# ---------------------------------------------------------------------------


def test_apply_converted_weights_key_mismatch_errors() -> None:
    model = _SimpleParamModel(
        {
            "a": torch.ones(4, dtype=torch.bfloat16),
            "b": torch.ones(4, dtype=torch.bfloat16),
        }
    )
    worker = _make_worker(model, tp_size=1)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["a"],
                "target_keys": ["a"],
                "ops": [],
                "sharding": [None],
            },
            {
                "source_keys": ["b"],
                "target_keys": ["b"],
                "ops": [],
                "sharding": [None],
            },
        ]
    )

    with pytest.raises(ValueError, match="missing=\\['b'\\]"):
        worker.apply_converted_weights(
            {"a": torch.zeros(4, dtype=torch.bfloat16)},
            manifest_json=manifest,
        )
    # Atomic: 'a' must remain untouched on error.
    assert torch.equal(model.a.data, torch.ones(4, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="unexpected=\\['extra'\\]"):
        worker.apply_converted_weights(
            {
                "a": torch.zeros(4, dtype=torch.bfloat16),
                "b": torch.zeros(4, dtype=torch.bfloat16),
                "extra": torch.zeros(4, dtype=torch.bfloat16),
            },
            manifest_json=manifest,
        )
    assert torch.equal(model.a.data, torch.ones(4, dtype=torch.bfloat16))


def test_apply_converted_weights_shape_and_dtype_mismatch_errors() -> None:
    model = _SimpleParamModel(
        {
            "a": torch.ones(4, dtype=torch.bfloat16),
            "b": torch.ones(4, 8, dtype=torch.bfloat16),
        }
    )
    worker = _make_worker(model, tp_size=1)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["a"],
                "target_keys": ["a"],
                "ops": [],
                "sharding": [None],
            },
            {
                "source_keys": ["b"],
                "target_keys": ["b"],
                "ops": [],
                "sharding": [None, None],
            },
        ]
    )

    # Shape mismatch on 'b' must fail before mutating 'a'.
    with pytest.raises(ValueError, match="Shape mismatch for parameter 'b'"):
        worker.apply_converted_weights(
            {
                "a": torch.zeros(4, dtype=torch.bfloat16),
                "b": torch.zeros(4, 6, dtype=torch.bfloat16),
            },
            manifest_json=manifest,
        )
    assert torch.equal(model.a.data, torch.ones(4, dtype=torch.bfloat16))

    # Dtype mismatch on 'b' must fail before mutating 'a'.
    with pytest.raises(ValueError, match="Dtype mismatch for parameter 'b'"):
        worker.apply_converted_weights(
            {
                "a": torch.zeros(4, dtype=torch.bfloat16),
                "b": torch.zeros(4, 8, dtype=torch.float32),
            },
            manifest_json=manifest,
        )
    assert torch.equal(model.a.data, torch.ones(4, dtype=torch.bfloat16))


def test_apply_converted_weights_bad_sharding_axis_and_indivisible_dim() -> None:
    model = _SimpleParamModel({"w": torch.zeros(3, 4, dtype=torch.bfloat16)})
    worker = _make_worker(model, rank=0, tp_size=2)

    bad_axis_manifest = _make_manifest(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": ["pp", None],
            }
        ]
    )
    with pytest.raises(ValueError, match="Unsupported sharding axis 'pp'"):
        worker.apply_converted_weights(
            {"w": torch.zeros(6, 4, dtype=torch.bfloat16)},
            manifest_json=bad_axis_manifest,
        )

    indivisible_manifest = _make_manifest(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ]
    )
    with pytest.raises(ValueError, match="not divisible by tp world_size=2"):
        worker.apply_converted_weights(
            {"w": torch.zeros(5, 4, dtype=torch.bfloat16)},
            manifest_json=indivisible_manifest,
        )


def test_apply_converted_weights_syncs_only_tpu_tensors() -> None:
    model = _SimpleParamModel({"w": torch.zeros(4, dtype=torch.bfloat16)})
    worker = _make_worker(model, tp_size=1)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": [None],
            }
        ]
    )

    with (
        patch("vllm_torchtpu.worker.tpu_worker.utils.synchronize_tensors") as sync,
        patch("vllm_torchtpu.worker.tpu_worker.utils.fresh_buffer") as fresh,
    ):
        worker.apply_converted_weights(
            {"w": torch.ones(4, dtype=torch.bfloat16)},
            manifest_json=manifest,
        )
        sync.assert_not_called()
        fresh.assert_not_called()


def test_apply_converted_weights_uses_fresh_buffer_for_tpu_shards() -> None:
    """TPU shards must pass through utils.fresh_buffer to avoid view aliasing."""

    class _FakeTpuTensor:
        def __init__(self, data: torch.Tensor):
            self._data = data
            self.device = SimpleNamespace(type="tpu")
            self.shape = data.shape
            self.dtype = data.dtype
            self.ndim = data.ndim

        def narrow(self, dim: int, start: int, length: int) -> _FakeTpuTensor:
            return _FakeTpuTensor(self._data.narrow(dim, start, length))

    model = _SimpleParamModel({"w": torch.zeros(4, dtype=torch.bfloat16)})
    worker = _make_worker(model, rank=1, tp_size=2)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": ["tp"],
            }
        ]
    )
    global_data = torch.arange(8, dtype=torch.bfloat16)
    fake_tpu_global = _FakeTpuTensor(global_data)

    with patch(
        "vllm_torchtpu.worker.tpu_worker.utils.fresh_buffer",
        side_effect=lambda t: t._data.clone(),
    ) as fresh:
        worker.apply_converted_weights({"w": fake_tpu_global}, manifest_json=manifest)

    fresh.assert_called_once()
    assert torch.equal(model.w.data, global_data[4:])


def test_resolve_sharding_coords_initialized_moe_tp_gt_tp() -> None:
    """When model_parallel_is_initialized() and moe_tp_size > tp_size,
    moe_tp_rank must span 0..moe_tp_size-1 via global_rank."""
    w13_global = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["w13"],
                "target_keys": ["w13"],
                "ops": [],
                "sharding": ["ep", None, "moe_tp"],
            }
        ],
        expert_axis_grouped=True,
    )

    # dp_size=4, tp_size=2, ep_size=2 => moe_tp_size = (4 * 2) // 2 = 4 > tp_size=2
    seen_shards: set[tuple[int, int]] = set()
    for dp_rank in range(4):
        for tp_rank in range(2):
            global_rank = dp_rank * 2 + tp_rank
            ep_rank = (global_rank // 4) % 2
            moe_tp_rank = global_rank % 4
            model = _SimpleParamModel(
                {"w13": torch.zeros(1, 4, 2, dtype=torch.bfloat16)}
            )
            worker = _make_worker(
                model,
                rank=tp_rank,
                tp_size=2,
                dp_size=4,
                dp_rank=dp_rank,
                enable_ep=True,
                ep_size=2,
            )
            ep_group = SimpleNamespace(world_size=2, rank_in_group=ep_rank)
            dp_group = SimpleNamespace(world_size=4, rank_in_group=dp_rank)
            with (
                patch(
                    "vllm_torchtpu.worker.tpu_worker.model_parallel_is_initialized",
                    return_value=True,
                ),
                patch(
                    "vllm_torchtpu.worker.tpu_worker.get_tensor_model_parallel_world_size",
                    return_value=2,
                ),
                patch(
                    "vllm_torchtpu.worker.tpu_worker.get_tensor_model_parallel_rank",
                    return_value=tp_rank,
                ),
                patch(
                    "vllm.distributed.parallel_state.get_ep_group",
                    return_value=ep_group,
                ),
                patch(
                    "vllm.distributed.parallel_state.get_dp_group",
                    return_value=dp_group,
                ),
            ):
                worker.apply_converted_weights(
                    {"w13": w13_global}, manifest_json=manifest
                )

            seen_shards.add((ep_rank, moe_tp_rank))
            assert torch.equal(
                model.w13.data,
                w13_global[
                    ep_rank : ep_rank + 1,
                    :,
                    moe_tp_rank * 2 : (moe_tp_rank + 1) * 2,
                ],
            )

    assert seen_shards == {(e, m) for e in range(2) for m in range(4)}


# ---------------------------------------------------------------------------
# 4. load_from_hf() end-to-end on mock TPUWorker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("flipped", [False, True])
def test_load_from_hf_end_to_end(tp_size: int, flipped: bool) -> None:
    vocab, hidden, inter = 32, 16, 32
    num_heads, num_kv_heads = 4, 2
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    torch.manual_seed(0)
    hf_weights = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            hidden, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(
            kv_dim, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(
            kv_dim, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(
            hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(
            kv_dim, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(
            kv_dim, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            hidden, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            inter, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.up_proj.weight": torch.randn(
            inter, hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.down_proj.weight": torch.randn(
            hidden, inter, dtype=torch.bfloat16
        ),
        "model.layers.0.input_layernorm.weight": torch.randn(
            hidden, dtype=torch.bfloat16
        ),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(
            hidden, dtype=torch.bfloat16
        ),
        "model.norm.weight": torch.randn(hidden, dtype=torch.bfloat16),
    }

    for rank in range(tp_size):
        model = _TinyQwenModel(
            vocab_size=vocab,
            hidden_size=hidden,
            inter_size=inter,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tp_size=tp_size,
            tie_word_embeddings=True,
            flipped=flipped,
        )
        worker = _make_worker(model, rank=rank, tp_size=tp_size)

        # Auto-fetches manifest via get_weight_mapping_manifest() when None.
        worker.load_from_hf(hf_weights)

        # Verify embedding shard + tied lm_head.
        vocab_shard = vocab // tp_size
        expected_embed = hf_weights["model.embed_tokens.weight"][
            rank * vocab_shard : (rank + 1) * vocab_shard
        ]
        assert torch.equal(model.model.embed_tokens.weight.data, expected_embed)
        assert model.lm_head.weight is model.model.embed_tokens.weight

        # Verify fused QKV shard (weight + bias), including KV replication
        # when tp_size > num_kv_heads (e.g. tp_size=4, num_kv_heads=2).
        q_shard = hidden // tp_size
        kv_rep = max(1, tp_size // num_kv_heads)
        kv_unique = tp_size // kv_rep
        kv_shard = kv_dim // kv_unique
        kv_rank = rank // kv_rep
        expected_qkv_w = torch.cat(
            [
                hf_weights["model.layers.0.self_attn.q_proj.weight"][
                    rank * q_shard : (rank + 1) * q_shard
                ],
                hf_weights["model.layers.0.self_attn.k_proj.weight"][
                    kv_rank * kv_shard : (kv_rank + 1) * kv_shard
                ],
                hf_weights["model.layers.0.self_attn.v_proj.weight"][
                    kv_rank * kv_shard : (kv_rank + 1) * kv_shard
                ],
            ],
            dim=0,
        )
        if flipped:
            expected_qkv_w = expected_qkv_w.transpose(0, 1)
        assert torch.equal(
            model.model.layers[0].self_attn.qkv_proj.weight.data, expected_qkv_w
        )

        expected_qkv_b = torch.cat(
            [
                hf_weights["model.layers.0.self_attn.q_proj.bias"][
                    rank * q_shard : (rank + 1) * q_shard
                ],
                hf_weights["model.layers.0.self_attn.k_proj.bias"][
                    kv_rank * kv_shard : (kv_rank + 1) * kv_shard
                ],
                hf_weights["model.layers.0.self_attn.v_proj.bias"][
                    kv_rank * kv_shard : (kv_rank + 1) * kv_shard
                ],
            ],
            dim=0,
        )
        assert torch.equal(
            model.model.layers[0].self_attn.qkv_proj.bias.data, expected_qkv_b
        )

        # Verify fused gate_up_proj shard.
        inter_shard = inter // tp_size
        expected_gate_up = torch.cat(
            [
                hf_weights["model.layers.0.mlp.gate_proj.weight"][
                    rank * inter_shard : (rank + 1) * inter_shard
                ],
                hf_weights["model.layers.0.mlp.up_proj.weight"][
                    rank * inter_shard : (rank + 1) * inter_shard
                ],
            ],
            dim=0,
        )
        if flipped:
            expected_gate_up = expected_gate_up.transpose(0, 1)
        assert torch.equal(
            model.model.layers[0].mlp.gate_up_proj.weight.data, expected_gate_up
        )

        # Verify row-parallel o_proj and down_proj shards.
        expected_o = hf_weights["model.layers.0.self_attn.o_proj.weight"][
            :, rank * q_shard : (rank + 1) * q_shard
        ]
        if flipped:
            expected_o = expected_o.transpose(0, 1)
        assert torch.equal(
            model.model.layers[0].self_attn.o_proj.weight.data, expected_o
        )

        expected_down = hf_weights["model.layers.0.mlp.down_proj.weight"][
            :, rank * inter_shard : (rank + 1) * inter_shard
        ]
        if flipped:
            expected_down = expected_down.transpose(0, 1)
        assert torch.equal(
            model.model.layers[0].mlp.down_proj.weight.data, expected_down
        )

        # Verify replicated norms.
        assert torch.equal(
            model.model.norm.weight.data, hf_weights["model.norm.weight"]
        )
