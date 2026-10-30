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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_torchtpu import utils
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.utils import (
    NativeKVCacheLayer,
    NativeKVCacheMaterialization,
    tpu_bind_kv_cache,
)
from vllm_torchtpu.worker.tpu_worker import TPUWorker


class _DummyNativeLayer(nn.Module, NativeKVCacheLayer, AttentionLayerBase):
    def __init__(
        self,
        spec: FullAttentionSpec | None = None,
        materialization: NativeKVCacheMaterialization | None = None,
        reserve_delta: int = 0,
    ) -> None:
        super().__init__()
        self._spec = spec
        self._materialization = materialization
        self._reserve_delta = reserve_delta
        self.bound_args: tuple[object, object, object, object] | None = None
        self.materialize_calls: list[tuple[object, torch.device, object]] = []
        self.reserve_calls: list[tuple[object, int, bool]] = []

    def get_attn_backend(self) -> object:
        return None

    def bind_kv_cache(
        self,
        cache: object,
        *,
        raw_cache: object = None,
        layout: object = None,
        rope_pool: object = None,
    ) -> None:
        super().bind_kv_cache(
            cache, raw_cache=raw_cache, layout=layout, rope_pool=rope_pool
        )
        self.bound_args = (cache, raw_cache, layout, rope_pool)

    def get_kv_cache_spec(self, vllm_config: object) -> FullAttentionSpec | None:
        del vllm_config
        return self._spec

    def materialize_kv_caches(
        self,
        kv_cache_config: object,
        device: torch.device,
        layout: object,
    ) -> NativeKVCacheMaterialization:
        self.materialize_calls.append((kv_cache_config, device, layout))
        assert self._materialization is not None
        return self._materialization

    def reserve_companion_memory(
        self,
        vllm_config: object,
        available: int,
        *,
        auto: bool = False,
    ) -> int:
        self.reserve_calls.append((vllm_config, available, auto))
        return available - self._reserve_delta


def _make_runner_for_init(forward_context: dict[str, object]) -> SimpleNamespace:
    block_table_entry = SimpleNamespace(
        get_cpu_tensor=lambda: torch.zeros((2, 4), dtype=torch.int32)
    )
    return SimpleNamespace(
        device=torch.device("cpu"),
        use_spmd=False,
        enforce_eager=True,
        _unified_kv_layout=False,
        _uniform_mamba_layout=False,
        _mamba_num_blocks=None,
        shared_kv_cache_layers={},
        kv_caches=[],
        kv_cache_raw_tensors=[],
        block_table_cpu=torch.zeros((2, 4), dtype=torch.int32),
        input_batch=SimpleNamespace(block_table=[block_table_entry]),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "NHD"),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        ),
        speculative_config=None,
        model_config=SimpleNamespace(get_head_size=lambda: 128),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=forward_context),
        ),
        may_reinitialize_input_batch=MagicMock(),
        _precompile_dense_kv_cache_block_copies=MagicMock(),
        _initialize_attention_kernels=MagicMock(),
        _precompile_substitute_placeholder_token=MagicMock(),
        requests={},
        _mamba_offset_seeded=set(),
        _pre_async_results=None,
        _install_spec_token_room_guard=lambda: None,
    )


def test_native_kv_cache_layer_base_and_bind_contracts() -> None:
    base = NativeKVCacheLayer()
    assert base.zero_blocks([1, 2]) is None
    assert base.reserve_companion_memory(object(), 1024, auto=True) == 1024
    with pytest.raises(NotImplementedError):
        base.get_kv_cache_spec(object())
    with pytest.raises(NotImplementedError):
        base.materialize_kv_caches(object(), torch.device("cpu"), "NHD")

    native_a = _DummyNativeLayer()
    native_b = _DummyNativeLayer()
    plain = SimpleNamespace(kv_cache=None)
    t0 = torch.zeros((4, 8), dtype=torch.uint8)
    t1 = torch.zeros((4, 8), dtype=torch.uint8)
    t2 = torch.zeros((4, 8), dtype=torch.uint8)
    raw0 = torch.zeros((4, 4), dtype=torch.uint8)
    raw1 = torch.zeros((4, 4), dtype=torch.uint8)
    rope0 = torch.zeros((4, 2), dtype=torch.int32)

    runner_kv_caches: list[torch.Tensor] = []
    tpu_bind_kv_cache(
        {"layers.0.attn": t0, "layers.1.attn": t1, "layers.2.attn": t2},
        {"layers.0.attn": native_a, "layers.1.attn": native_b, "layers.2.attn": plain},
        runner_kv_caches,
        native_raw_cache={"layers.0.attn": raw0, "layers.1.attn": raw1},
        native_cache_layouts={"layers.0.attn": ("B", "D"), "layers.1.attn": ("B", "D")},
        native_rope_pools={"layers.0.attn": rope0},
    )
    assert runner_kv_caches == [t0, t1, t2]
    assert native_a.kv_cache is t0
    assert native_a.bound_args == (t0, raw0, ("B", "D"), rope0)
    assert native_b.bound_args == (t1, raw1, ("B", "D"), None)
    assert plain.kv_cache is t2

    tpu_bind_kv_cache({"layers.0.attn": t0}, {"layers.0.attn": native_a}, [])
    assert native_a.bound_args == (t0, None, None, None)

    with pytest.raises(AssertionError, match="Missing native_raw_cache"):
        tpu_bind_kv_cache(
            {"layers.0.attn": t0},
            {"layers.0.attn": native_a},
            [],
            native_raw_cache={},
        )
    with pytest.raises(AssertionError, match="Missing native_cache_layouts"):
        tpu_bind_kv_cache(
            {"layers.0.attn": t0},
            {"layers.0.attn": native_a},
            [],
            native_cache_layouts={},
        )


def test_kv_cache_manager_native_spec_and_materialization() -> None:
    spec = FullAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.uint8
    )
    t0 = torch.zeros((4, 64, 128), dtype=torch.uint8)
    raw_pool = torch.zeros((1, 4, 64, 128), dtype=torch.uint8)
    rope_pool = torch.zeros((1, 4, 16, 128), dtype=torch.int32)
    mat = NativeKVCacheMaterialization(
        kv_caches={"layers.0.attn": t0},
        raw_tensors=[raw_pool],
        native_raw_cache={"layers.0.attn": raw_pool},
        native_cache_layouts={"layers.0.attn": ("L", "B", "N", "D")},
        rope_pools={"layers.0.attn": rope_pool},
    )
    layer0 = _DummyNativeLayer(spec=spec, materialization=mat)
    layer1 = _DummyNativeLayer(spec=None)
    runner = _make_runner_for_init({"layers.0.attn": layer0, "layers.1.attn": layer1})
    runner.vllm_config.cache_config = SimpleNamespace(block_size=64, cache_dtype="auto")
    runner.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    runner._is_async_drafter = False
    runner.kv_cache_dtype = torch.uint8
    mgr = KVCacheManager(runner)

    with (
        patch.object(utils, "get_device_name", return_value="TPU v7"),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.TpuPlatform._find_non_ssm_backend",
            return_value=None,
        ),
    ):
        assert mgr.get_kv_cache_spec() == {"layers.0.attn": spec}

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * 64 * 128,
                layers=["layers.0.attn"],
                layer_stride=64 * 128,
                block_stride=64 * 128,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layers.0.attn"], kv_cache_spec=spec)
        ],
    )
    with (
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.TpuPlatform._find_non_ssm_backend",
            return_value=None,
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.AttentionMetadataBuilder",
            return_value=MagicMock(),
        ),
        patch("vllm_torchtpu.runner.kv_cache_manager.synchronize_tensors") as sync_mock,
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
            return_value=False,
        ),
    ):
        mgr.initialize_kv_cache(kv_cache_config)

    assert len(layer0.materialize_calls) == 1
    sync_mock.assert_called_once_with([raw_pool])
    assert runner.kv_cache_raw_tensors == [raw_pool]
    assert runner._native_kv_cache_owner is layer0
    assert layer0.bound_args == (t0, raw_pool, ("L", "B", "N", "D"), rope_pool)
    assert TPUModelRunner.is_unified_pool_used(runner) is False
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_block_copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=1)]
    with pytest.raises(
        NotImplementedError,
        match="CoW block copies are unsupported on custom NativeKVCacheLayer pools",
    ):
        TPUModelRunner._update_states(runner, sched_out)

    plain_runner = _make_runner_for_init({"layers.0.attn": SimpleNamespace()})
    plain_cfg = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[])
    with (
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.TpuPlatform._find_non_ssm_backend",
            return_value=None,
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.AttentionMetadataBuilder",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
            return_value=False,
        ),
    ):
        KVCacheManager(plain_runner).initialize_kv_cache(plain_cfg)
    assert plain_runner._native_kv_cache_owner is None


def test_kv_cache_manager_native_edge_cases_and_mixed_rejection() -> None:
    spec = FullAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=128, dtype=torch.uint8
    )
    t0 = torch.zeros((4, 64, 128), dtype=torch.uint8)
    layer0 = _DummyNativeLayer(
        spec=spec,
        materialization=NativeKVCacheMaterialization(
            kv_caches={"layers.0.attn": t0}, raw_tensors=[]
        ),
    )
    runner = _make_runner_for_init({"layers.0.attn": layer0})
    mgr = KVCacheManager(runner)
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layers.0.attn"], kv_cache_spec=spec)
        ],
    )
    with (
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.TpuPlatform._find_non_ssm_backend",
            return_value=None,
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.AttentionMetadataBuilder",
            return_value=MagicMock(),
        ),
        patch("vllm_torchtpu.runner.kv_cache_manager.synchronize_tensors") as sync_mock,
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.has_kv_transfer_group",
            return_value=False,
        ),
    ):
        mgr.initialize_kv_cache(cfg)
    sync_mock.assert_not_called()

    with (
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.TpuPlatform._find_non_ssm_backend",
            return_value=None,
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.AttentionMetadataBuilder",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.envs.VLLM_TPU_BLOCK_MAJOR_KV",
            True,
        ),
        pytest.raises(NotImplementedError, match="native KV cache layers"),
    ):
        mgr.initialize_kv_cache(cfg)

    mixed_runner = _make_runner_for_init(
        {"layers.0.attn": layer0, "layers.1.attn": SimpleNamespace()}
    )
    mixed_cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layers.0.attn", "layers.1.attn"],
                kv_cache_spec=spec,
            )
        ],
    )
    with (
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.TpuPlatform._find_non_ssm_backend",
            return_value=None,
        ),
        patch(
            "vllm_torchtpu.runner.kv_cache_manager.AttentionMetadataBuilder",
            return_value=MagicMock(),
        ),
        pytest.raises(
            NotImplementedError, match="Mixed NativeKVCacheLayer and non-native"
        ),
    ):
        KVCacheManager(mixed_runner).initialize_kv_cache(mixed_cfg)


def test_tpu_worker_determine_available_memory_reserves_companion_memory() -> None:
    native_layer = _DummyNativeLayer(reserve_delta=256)
    worker = object.__new__(TPUWorker)
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layers.0.attn": native_layer}
        )
    )
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=1024, gpu_memory_utilization=0.9
    )
    assert worker.determine_available_memory() == 768
    assert native_layer.reserve_calls == [(worker.vllm_config, 1024, False)]

    native_layer.reserve_calls.clear()
    worker.cache_config.kv_cache_memory_bytes = None
    worker.devices = [torch.device("cpu")]
    worker.parallel_config = SimpleNamespace()
    worker.model_runner = SimpleNamespace(
        max_num_tokens=16, profile_run=MagicMock(), is_pooling_model=False
    )
    budget = utils.HbmBudget(
        total_limit=10 * utils.GBYTES,
        total_used=2 * utils.GBYTES,
        cap=9 * utils.GBYTES,
        available=4096,
    )
    with (
        patch.object(utils, "compute_hbm_budget", return_value=budget),
        patch.object(utils, "estimate_kv_connector_hbm_reserve", return_value=0),
        patch.object(utils, "min_across_ep_dp", side_effect=lambda _pc, val: val),
    ):
        assert worker.determine_available_memory() == 4096 - 256
    assert native_layer.reserve_calls == [(worker.vllm_config, 4096, True)]

    worker_no_native = object.__new__(TPUWorker)
    worker_no_native.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layers.0.attn": SimpleNamespace()}
        )
    )
    worker_no_native.cache_config = SimpleNamespace(kv_cache_memory_bytes=2048)
    assert worker_no_native.determine_available_memory() == 2048


@pytest.mark.tpu_test
def test_native_kv_cache_layer_tpu_allocation() -> None:  # pragma: no cover
    """Verify NativeKVCacheLayer materializes and binds a TPU7x KV cache pool."""
    import torch_tpu  # noqa: F401

    dev = torch.device("tpu:0")
    raw_pool = torch.zeros((16, 128), dtype=torch.int32, device=dev)
    rope_pool = torch.zeros((16, 128), dtype=torch.int32, device=dev)
    layout = ((16, 128), torch.int32, 8192, 0)
    mat = NativeKVCacheMaterialization(
        kv_caches={"layers.0.attn": raw_pool},
        raw_tensors=[raw_pool],
        native_raw_cache={"layers.0.attn": raw_pool},
        native_cache_layouts={"layers.0.attn": layout},
        rope_pools={"layers.0.attn": rope_pool},
    )
    layer = _DummyNativeLayer(materialization=mat)
    ctx: dict[str, object] = {"layers.0.attn": layer}
    current_caches: list[torch.Tensor] = []
    tpu_bind_kv_cache(
        mat.kv_caches,
        ctx,
        current_caches,
        native_raw_cache=mat.native_raw_cache,
        native_cache_layouts=mat.native_cache_layouts,
        native_rope_pools=mat.rope_pools,
    )
    assert layer.bound_args == (raw_pool, raw_pool, layout, rope_pool)
    assert current_caches[0].device.type == "tpu"
    current_caches[0][0].fill_(7)
    assert int(raw_pool[0, 0].cpu().item()) == 7
