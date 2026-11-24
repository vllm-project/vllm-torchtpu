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

from tests.rl._fakes import Qwen2ForCausalLM, build_qwen2_hf_weights
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
    stubbed: list[str] = []
    try:
        if prev_platform is None:
            from vllm.platforms.interface import UnspecifiedPlatform

            _vllm_platforms._current_platform = UnspecifiedPlatform()
        for name in (
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.profiler",
            "torch_tpu._internal.sync",
            "vllm_torchtpu.layers",
            "vllm_torchtpu.layers.adapter",
            "vllm_torchtpu.layers.adapter.attention",
            "vllm_torchtpu.runner",
            "vllm_torchtpu.runner.tpu_pooling",
            "vllm_torchtpu.runner.tpu_runner",
        ):
            if name not in sys.modules:
                m = types.ModuleType(name)
                if name.endswith(("layers", "adapter", "runner")):
                    m.__path__ = []
                sys.modules[name] = m
                stubbed.append(name)
        sys.modules["torch_tpu._internal.profiler"].TpuProfilerConfig = object
        sys.modules["torch_tpu._internal.sync"].synchronize = lambda *a, **kw: None
        sys.modules[
            "vllm_torchtpu.layers.adapter.attention"
        ].TPU_STR_DTYPE_TO_TORCH_DTYPE = {}
        sys.modules[
            "vllm_torchtpu.runner.tpu_pooling"
        ].compute_pooling_kv_cache_cap_bytes = lambda *a, **kw: None
        sys.modules["vllm_torchtpu.runner.tpu_runner"].TPUModelRunner = object
        from vllm_torchtpu.worker.tpu_worker import TPUWorker as _TPUWorker

        return _TPUWorker
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)
        if had_platform:
            _vllm_platforms._current_platform = prev_platform


TPUWorker = _import_tpu_worker()


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


def _manifest_json(
    rules: list[dict],
    *,
    stripes: dict[str, int] | None = None,
    tied_keys: list[list[str]] | None = None,
) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "expert_axis_grouped": False,
            "stripes": stripes or {},
            "tied_keys": tied_keys or [],
            "dropped_keys": [],
            "rules": rules,
        }
    )


def test_get_weight_mapping_manifest_returns_valid_json() -> None:
    model = Qwen2ForCausalLM(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_kv_heads=2,
        tp_size=1,
    )
    manifest = json.loads(_make_worker(model, tp_size=1).get_weight_mapping_manifest())
    validate_manifest(manifest)
    assert ["model.embed_tokens.weight", "lm_head.weight"] in manifest["tied_keys"]
    targets = {k for r in manifest["rules"] for k in r["target_keys"]}
    assert {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    } <= targets


def test_apply_converted_weights_replicated_preserves_buffer_identity() -> None:
    model = _SimpleParamModel({"w": torch.zeros(4, 8, dtype=torch.bfloat16)})
    ptr_before = model.w.data.data_ptr()
    m = _manifest_json(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": [None, None],
            }
        ]
    )
    new_w = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    _make_worker(model).apply_converted_weights({"w": new_w}, manifest_json=m)
    assert model.w.data.data_ptr() == ptr_before
    assert torch.equal(model.w.data, new_w)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_apply_converted_weights_tp_multi_rank(tp_size: int) -> None:
    g_w = torch.arange(tp_size * 4 * 8, dtype=torch.bfloat16).reshape(tp_size * 4, 8)
    m = _manifest_json(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ],
        stripes={"tp_size": tp_size},
    )
    for r in range(tp_size):
        model = _SimpleParamModel({"w": torch.zeros(4, 8, dtype=torch.bfloat16)})
        _make_worker(model, rank=r, tp_size=tp_size).apply_converted_weights(
            {"w": g_w}, manifest_json=m
        )
        assert torch.equal(model.w.data, g_w[r * 4 : (r + 1) * 4])


def test_apply_converted_weights_ep_and_moe_tp_axes() -> None:
    ep_size, moe_tp_size = 2, 2
    g_w = torch.arange(4 * 8 * 6, dtype=torch.bfloat16).reshape(4, 8, 6)
    m = _manifest_json(
        [
            {
                "source_keys": ["e"],
                "target_keys": ["e"],
                "ops": [],
                "sharding": ["ep", "moe_tp", None],
            }
        ],
        stripes={"tp_size": 2, "ep_size": ep_size, "moe_tp_size": moe_tp_size},
    )
    for g_rank in range(ep_size * moe_tp_size):
        ep_r, mtp_r = divmod(g_rank, moe_tp_size)
        model = _SimpleParamModel({"e": torch.zeros(2, 4, 6, dtype=torch.bfloat16)})
        w = _make_worker(
            model,
            rank=g_rank,
            tp_size=2,
            dp_size=2,
            dp_rank=ep_r,
            enable_ep=True,
            ep_size=ep_size,
        )
        w.apply_converted_weights({"e": g_w}, manifest_json=m)
        assert torch.equal(
            model.e.data, g_w[ep_r * 2 : (ep_r + 1) * 2, mtp_r * 4 : (mtp_r + 1) * 4]
        )


def test_tied_keys_dedup_canonical_order_and_mismatch() -> None:
    model = nn.Module()
    shared = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    model.a = shared
    model.b = shared
    m = _manifest_json(
        [
            {
                "source_keys": ["a"],
                "target_keys": ["a"],
                "ops": [],
                "sharding": [None, None],
            }
        ],
        tied_keys=[["a", "b"]],
    )
    t1 = torch.ones(4, 4, dtype=torch.bfloat16)
    staged = stage_converted_weights(
        model, {"a": t1, "b": t1.clone()}, m, ShardingCoords()
    )
    assert len(staged) == 1 and torch.equal(staged[0][1], t1)

    # Alias mismatch raises ValueError
    with pytest.raises(ValueError, match="Tied parameter"):
        stage_converted_weights(model, {"a": t1, "b": t1 * 2}, m, ShardingCoords())

    # Direct weight_installer helpers
    smap = extract_sharding_map(json.loads(m))
    assert smap["a"] == [None, None]
    assert torch.equal(
        slice_tensor_for_shard("a", t1, [None, None], ShardingCoords()), t1
    )


def test_apply_converted_weights_error_checks() -> None:
    model = _SimpleParamModel({"w": torch.zeros(4, 8, dtype=torch.bfloat16)})
    m = _manifest_json(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": [None, None],
            }
        ]
    )
    worker = _make_worker(model)
    with pytest.raises(ValueError, match="missing="):
        worker.apply_converted_weights({}, manifest_json=m)
    with pytest.raises(ValueError, match="unexpected="):
        worker.apply_converted_weights(
            {"w": torch.zeros(4, 8, dtype=torch.bfloat16), "extra": torch.zeros(1)},
            manifest_json=m,
        )
    with pytest.raises(ValueError, match="Shape mismatch"):
        worker.apply_converted_weights(
            {"w": torch.zeros(2, 8, dtype=torch.bfloat16)}, manifest_json=m
        )
    with pytest.raises(ValueError, match="Dtype mismatch"):
        worker.apply_converted_weights(
            {"w": torch.zeros(4, 8, dtype=torch.float32)}, manifest_json=m
        )

    m_bad_ax = _manifest_json(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": ["bogus", None],
            }
        ]
    )
    with pytest.raises(ValueError, match="Unsupported sharding axis"):
        worker.apply_converted_weights(
            {"w": torch.zeros(4, 8, dtype=torch.bfloat16)}, manifest_json=m_bad_ax
        )

    m_indiv = _manifest_json(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": ["tp", None],
            }
        ],
        stripes={"tp_size": 3},
    )
    with pytest.raises(ValueError, match="not divisible"):
        _make_worker(model, tp_size=3).apply_converted_weights(
            {"w": torch.zeros(4, 8, dtype=torch.bfloat16)}, manifest_json=m_indiv
        )


def test_apply_converted_weights_tpu_sync_and_fresh_buffer() -> None:
    from vllm_torchtpu.worker import tpu_worker as tw_mod

    model = _SimpleParamModel({"w": torch.zeros(4, 8, dtype=torch.bfloat16)})
    m = _manifest_json(
        [
            {
                "source_keys": ["w"],
                "target_keys": ["w"],
                "ops": [],
                "sharding": [None, None],
            }
        ]
    )
    with patch.object(tw_mod.utils, "synchronize_tensors") as mock_sync:
        _make_worker(model).apply_converted_weights(
            {"w": torch.ones(4, 8, dtype=torch.bfloat16)}, manifest_json=m
        )
        mock_sync.assert_not_called()

    # Fake TPU shard triggers utils.fresh_buffer + synchronize_tensors
    fake_shard = MagicMock(spec=torch.Tensor)
    fake_shard.device = SimpleNamespace(type="tpu")
    fresh = torch.ones(4, 8, dtype=torch.bfloat16)
    with (
        patch(
            "vllm_torchtpu.rl.weight_installer.stage_converted_weights",
            return_value=[(model.w, fake_shard)],
        ),
        patch.object(tw_mod.utils, "fresh_buffer", return_value=fresh) as mock_fb,
        patch.object(tw_mod.utils, "synchronize_tensors") as mock_sync,
    ):
        _make_worker(model).apply_converted_weights({"w": fresh}, manifest_json=m)
        mock_fb.assert_called_once_with(fake_shard)
        assert torch.equal(model.w.data, fresh)
        mock_sync.assert_not_called()  # model.w.data is on CPU in this unit test


def test_resolve_sharding_coords_bijective_when_moe_tp_gt_tp() -> None:
    from vllm_torchtpu.worker.tpu_worker import _resolve_sharding_coords

    ep_group = SimpleNamespace(world_size=2, rank_in_group=1)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    dp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    seen = set()
    for g_rank in range(8):
        worker = _make_worker(
            nn.Module(), rank=g_rank, tp_size=2, dp_size=4, enable_ep=True, ep_size=2
        )
        with (
            patch(
                "vllm.distributed.parallel_state.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "vllm.distributed.parallel_state.get_tp_group", return_value=tp_group
            ),
            patch(
                "vllm.distributed.parallel_state.get_dp_group", return_value=dp_group
            ),
            patch(
                "vllm.distributed.parallel_state.get_ep_group", return_value=ep_group
            ),
        ):
            coords = _resolve_sharding_coords(worker)
            seen.add((coords.ep_rank, coords.moe_tp_rank))
            assert coords.ep_size == 2 and coords.moe_tp_size == 4
    assert len(seen) == 8


@pytest.mark.parametrize("tp_size,flipped", [(1, False), (2, False), (2, True)])
def test_load_from_hf_end_to_end(tp_size: int, flipped: bool) -> None:
    hf = build_qwen2_hf_weights(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        num_kv_heads=2,
        tie_word_embeddings=True,
    )
    for r in range(tp_size):
        model = Qwen2ForCausalLM(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_heads=4,
            num_kv_heads=2,
            tp_size=tp_size,
            tie_word_embeddings=True,
            flipped=flipped,
        )
        worker = _make_worker(model, rank=r, tp_size=tp_size)
        worker.load_from_hf(hf)
        expected_emb = hf["model.embed_tokens.weight"].chunk(tp_size, dim=0)[r]
        assert torch.equal(model.model.embed_tokens.weight.data, expected_emb)
        assert torch.equal(model.lm_head.weight.data, expected_emb)
