# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU unit tests for manifest-backed `RaidenWorkerSync`."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tests.rl._fakes import (
    Qwen2ForCausalLM,
    Qwen3_5MoeForCausalLM,
    RMSNorm,
)
from vllm_torchtpu.rl import raiden_worker_sync
from vllm_torchtpu.rl.manifest_generator import generate_manifest_dict


@contextmanager
def _patched_ws(
    *,
    local_port: int = 19500,
    listener_port: int = 29500,
    endpoints: list[dict[str, Any]] | None = None,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    ws_inst = MagicMock()
    ws_inst.num_shards = 1
    ws_inst.local_port = local_port
    ws_inst.listener_port = listener_port
    ws_inst.get_local_endpoints.return_value = (
        endpoints
        if endpoints is not None
        else [{"endpoint": f"10.0.0.1:{local_port}", "shards": [0]}]
    )
    ws_inst.get_metrics.return_value = {"last_h2d_time_ms": 1.25}
    ws_cls = MagicMock(return_value=ws_inst)
    with (
        patch.object(
            raiden_worker_sync,
            "_ws_lib",
            SimpleNamespace(WeightSynchronizer=ws_cls),
        ),
        patch.object(raiden_worker_sync, "_check_bindable"),
    ):
        yield ws_cls, ws_inst


def test_canonical_param_name_and_sort_order() -> None:
    for raw, expected in (
        ("language_model.model.embed_tokens.weight", "embed_tokens.weight"),
        ("language_model.lm_head.weight", "lm_head.weight"),
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("base.model.layers.2.norm.weight", "layers.2.norm.weight"),
        (
            "model.layers.3.mlp.experts.routed_experts.w13_weight",
            "layers.3.mlp.experts.routed_experts.w13_weight",
        ),
        (
            "model.layers.0.post_feedforward_layernorm_1.weight",
            "layers.0.post_feedforward_layernorm_1.weight",
        ),
    ):
        assert raiden_worker_sync.canonical_param_name(raw) == expected


def test_collect_variables_from_manifest_qwen2_tied_and_sharded() -> None:
    model = Qwen2ForCausalLM(tp_size=2, tie_word_embeddings=True)
    variables = raiden_worker_sync.collect_variables_from_manifest(model, tp_size=2)
    by_name = {v.name: v for v in variables}

    # Tied lm_head shares buffer with embed_tokens and must not be duplicated.
    assert "embed_tokens.weight" in by_name
    assert "lm_head.weight" not in by_name
    assert by_name["embed_tokens.weight"].shard_domain == "tp"
    assert by_name["embed_tokens.weight"].shard_dim == 0
    assert by_name["embed_tokens.weight"].num_shards == 2

    # Flipped column-parallel QKV/gate_up shard on dim 1; row-parallel on dim 0.
    assert by_name["layers.0.self_attn.qkv_proj.weight"].shard_domain == "tp"
    assert by_name["layers.0.self_attn.qkv_proj.weight"].shard_dim == 1
    assert by_name["layers.0.self_attn.o_proj.weight"].shard_dim == 0
    assert by_name["layers.0.input_layernorm.weight"].shard_domain == "replicated"
    assert by_name["layers.0.input_layernorm.weight"].shard_dim is None

    # Sorted deterministically by canonical name.
    assert [v.name for v in variables] == sorted(v.name for v in variables)


def test_collect_variables_from_manifest_qwen35_moe_ep_and_tmp_rules() -> None:
    # Qwen3.5-MoE has __tmp__:* chained rules (GDN) + EP-sharded RoutedExperts.
    model = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=2, moe_tp_size=1)
    manifest = generate_manifest_dict(model, tp_size=2, dp_size=1, ep_size=2)
    variables = raiden_worker_sync.collect_variables_from_manifest(
        model, manifest=manifest
    )
    by_name = {v.name: v for v in variables}

    # Intermediate __tmp__:* keys must never be bound.
    assert not any(name.startswith("__tmp__:") for name in by_name)

    # Routed experts sharded on EP axis 0.
    assert by_name["layers.0.mlp.experts.w13_weight"].shard_domain == "ep"
    assert by_name["layers.0.mlp.experts.w13_weight"].shard_dim == 0
    assert by_name["layers.0.mlp.experts.w13_weight"].num_shards == 2
    assert by_name["layers.0.mlp.experts.w2_weight"].shard_domain == "ep"
    assert by_name["layers.0.mlp.experts.w2_weight"].shard_dim == 0

    # GDN projections, 3D conv1d (must stay TP-sharded under EP, not replicated),
    # and 1D head-sharded A_log/dt_bias.
    assert by_name["layers.0.linear_attn.in_proj_qkvz.weight"].shard_domain == "tp"
    assert by_name["layers.0.linear_attn.conv1d.weight"].shard_domain == "tp"
    assert by_name["layers.0.linear_attn.conv1d.weight"].shard_dim == 0
    assert by_name["layers.0.linear_attn.conv1d.weight"].num_shards == 2
    assert by_name["layers.0.linear_attn.A_log"].shard_domain == "tp"
    assert by_name["layers.0.linear_attn.A_log"].shard_dim == 0


def test_collect_variables_from_manifest_tp_only_moe() -> None:
    # TP-sharded MoE (ep_size=1, moe_tp_size=2) emits "moe_tp" in rule["sharding"].
    model = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=1, moe_tp_size=2)
    variables = raiden_worker_sync.collect_variables_from_manifest(
        model, tp_size=2, dp_size=1, ep_size=1
    )
    by_name = {v.name: v for v in variables}
    assert by_name["layers.0.mlp.experts.w13_weight"].shard_domain == "tp"
    assert by_name["layers.0.mlp.experts.w13_weight"].num_shards == 2
    assert by_name["layers.0.mlp.experts.w2_weight"].shard_domain == "tp"
    assert by_name["layers.0.mlp.experts.w2_weight"].num_shards == 2


def test_collect_variables_rejects_multi_axis_hybrid_ep_tp() -> None:
    model = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=2, moe_tp_size=2)
    with pytest.raises(NotImplementedError, match="multi-axis sharding"):
        raiden_worker_sync.collect_variables_from_manifest(
            model, tp_size=2, dp_size=1, ep_size=2
        )


def test_check_bindable_valid_and_invalid_tensors() -> None:
    def _fake_tensor(
        *,
        device_type: str = "tpu",
        dtype: torch.dtype = torch.bfloat16,
        contiguous: bool = True,
        offset: int = 0,
        shape: tuple[int, ...] = (4, 8),
        ptr: int = 0x1000,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            device=SimpleNamespace(type=device_type),
            dtype=dtype,
            shape=shape,
            is_contiguous=lambda: contiguous,
            storage_offset=lambda: offset,
            dim=lambda: len(shape),
            data_ptr=lambda: ptr,
        )

    def _var(tensor: Any) -> raiden_worker_sync.BoundVariable:
        return raiden_worker_sync.BoundVariable(
            name="layers.0.w",
            tensor=tensor,
            shard_dim=None,
            num_shards=1,
            shard_domain="replicated",
            global_shape=tuple(tensor.shape),
        )

    # Valid contiguous root TPU tensor passes.
    raiden_worker_sync._check_bindable(_var(_fake_tensor()))

    # Non-TPU device (real CPU tensor).
    with pytest.raises(ValueError, match="expected TPU tensor"):
        raiden_worker_sync._check_bindable(
            _var(torch.zeros((4, 8), dtype=torch.bfloat16))
        )

    # Unsupported dtype.
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        raiden_worker_sync._check_bindable(_var(_fake_tensor(dtype=torch.int8)))

    # Non-contiguous view.
    with pytest.raises(ValueError, match="contiguous root view"):
        raiden_worker_sync._check_bindable(_var(_fake_tensor(contiguous=False)))

    # Non-zero storage_offset.
    with pytest.raises(ValueError, match="contiguous root view"):
        raiden_worker_sync._check_bindable(_var(_fake_tensor(offset=4)))

    # 0-D and empty shapes.
    with pytest.raises(ValueError, match="empty or 0-D"):
        raiden_worker_sync._check_bindable(_var(_fake_tensor(shape=())))
    with pytest.raises(ValueError, match="empty or 0-D"):
        raiden_worker_sync._check_bindable(_var(_fake_tensor(shape=(0, 8))))

    # Unmaterialized tensor (data_ptr == 0).
    with pytest.raises(ValueError, match="unmaterialized tensor"):
        raiden_worker_sync._check_bindable(_var(_fake_tensor(ptr=0)))


def test_raiden_worker_sync_lifecycle_metadata_h2d_and_checksums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=2, moe_tp_size=1)
    with (
        _patched_ws() as (ws_cls, ws_inst),
        patch.object(torch.accelerator, "synchronize") as mock_sync,
    ):
        sync = raiden_worker_sync.RaidenWorkerSync(
            "replica_rollout-0",
            worker_index=0,
            tp_rank=1,
            tp_size=2,
            dp_rank=0,
            dp_size=1,
        )
        assert not sync.bound
        sync.bind(model)
        assert sync.bound
        assert ws_cls.call_count == 1

        meta = sync.metadata_dict()
        assert meta["unit"] == {
            "job_name": "replica_rollout-0",
            "job_replica_id": "1",
        }
        assert "transport_type" not in meta
        assert meta["control_plane_rpc_address"] == "10.0.0.1:29500"
        assert meta["shards"] == ["10.0.0.1:19500"]
        assert meta["mesh_shape"] == [2]
        assert meta["mesh_axes"] == ["tp"]
        assert len(meta["variables"]) == len(sync.variables)

        # Re-bind with identical tensors does not recreate or rebind.
        sync.bind(model)
        assert ws_cls.call_count == 1
        ws_inst.bind_weights.assert_not_called()

        # H2D with valid uuid triggers wait + whole-device sync.
        sync.h2d(uuid=42)
        ws_inst.wait_for_transfer_completion.assert_called_once_with(42)
        mock_sync.assert_called_once()
        assert sync.metrics() == {"last_h2d_time_ms": 1.25}

        # Invalid uuid rejected.
        for bad_uuid in (0, -1, True, None):
            with pytest.raises(ValueError, match="uuid"):
                sync.h2d(uuid=bad_uuid)  # type: ignore[arg-type]

        # Checksums match manual L1 sums.
        sums = sync.checksums(sample=2)
        expected_total = math.fsum(
            float(v.tensor.detach().abs().sum(dtype=torch.float32).item())
            for v in sync.variables
        )
        assert sums["__tensor_count__"] == len(sync.variables)
        assert sums["__element_count__"] == sum(
            v.tensor.numel() for v in sync.variables
        )
        assert math.isclose(sums["__grand_total__"], expected_total, rel_tol=1e-5)

        assert "replica_rollout-0" in sync.describe()

        # Shape change forces shutdown + recreate on rebind.
        small = nn.Module()
        small.norm = RMSNorm(8)
        sync.bind(small)
        assert ws_inst.shutdown.call_count == 1
        assert ws_cls.call_count == 2

        sync.shutdown()
        assert not sync.bound
        assert ws_inst.shutdown.call_count == 2

    # Env override warning on resolve_parallelism.
    monkeypatch.setenv("RAIDEN_PARALLELISM", "8")
    with patch.object(raiden_worker_sync.logger, "warning_once") as mock_warn:
        assert raiden_worker_sync.RaidenWorkerSync.resolve_parallelism(4) == 8
        mock_warn.assert_called_once()
    monkeypatch.delenv("RAIDEN_PARALLELISM", raising=False)
    assert raiden_worker_sync.RaidenWorkerSync.resolve_parallelism(4) == 4


def test_topology_and_env_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="worker_index"):
        raiden_worker_sync.RaidenWorkerSync("rollout", worker_index=-1)
    with pytest.raises(ValueError, match="tp_rank"):
        raiden_worker_sync.RaidenWorkerSync("rollout", tp_rank=2, tp_size=2)
    with pytest.raises(ValueError, match="dp_rank"):
        raiden_worker_sync.RaidenWorkerSync("rollout", dp_rank=1, dp_size=1)
    with pytest.raises(ValueError, match="parallelism"):
        raiden_worker_sync.RaidenWorkerSync("rollout", parallelism=0)

    monkeypatch.setenv("RAIDEN_BIND_IP", "2001:db8::1")
    assert raiden_worker_sync.local_ip() == "[2001:db8::1]"
    assert (
        raiden_worker_sync._routable_endpoint("127.0.0.1:19500", "10.0.0.9")
        == "10.0.0.9:19500"
    )
    assert (
        raiden_worker_sync._routable_endpoint("[::1]:19500", "10.0.0.9")
        == "10.0.0.9:19500"
    )
    monkeypatch.delenv("RAIDEN_BIND_IP", raising=False)


def test_dp_metadata_mesh_shape_listener_port_zero_and_parallel_h2h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Qwen3_5MoeForCausalLM(tp_size=2, ep_size=4, moe_tp_size=1)
    monkeypatch.setenv("WEIGHT_SYNC_PARALLEL_H2H", "1")
    with (
        _patched_ws(
            listener_port=0,
            endpoints=[{"endpoint": "127.0.0.1:19500", "shards": [0]}],
        ) as (_, ws_inst),
        patch.object(raiden_worker_sync, "local_ip", return_value="10.0.0.7"),
        patch.object(torch.accelerator, "synchronize") as mock_sync,
    ):
        sync = raiden_worker_sync.RaidenWorkerSync(
            "replica_rollout-0",
            worker_index=0,
            tp_rank=1,
            tp_size=2,
            dp_rank=1,
            dp_size=2,
        )
        sync.bind(model)
        meta = sync.metadata_dict()
        assert meta["mesh_shape"] == [2, 2]
        assert meta["mesh_axes"] == ["dp", "tp"]
        assert meta["control_plane_rpc_address"] == ""
        assert meta["shards"] == ["10.0.0.7:19500"]

        vars_by_name = {v["name"]: v for v in meta["variables"]}
        ep_var = vars_by_name["layers.0.mlp.experts.w13_weight"]
        assert ep_var["sharding_spec"] == ["dp,tp", "", ""]
        assert ep_var["global_shard_indices"] == [3]  # dp_rank*tp_size + tp_rank

        # With auto_h2d=False (WEIGHT_SYNC_PARALLEL_H2H=1), h2d() must call sync.h2d().
        sync.h2d(uuid=7)
        ws_inst.wait_for_transfer_completion.assert_called_once_with(7)
        ws_inst.h2d.assert_called_once_with()
        mock_sync.assert_called_once()
    monkeypatch.delenv("WEIGHT_SYNC_PARALLEL_H2H", raising=False)
