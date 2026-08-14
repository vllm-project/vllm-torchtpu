# Copyright 2025 Google LLC
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

import copy
import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_torchtpu.platforms.tpu_platform as tpu_platform
from vllm_torchtpu.platforms.tpu_platform import (
    TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP, TPU_3D_TORUS_DUAL_DEVICE_TOPOLOGY_MAP,
    TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP, TpuPlatform,
    _validate_phased_profiling_config, get_tpu_multihost_topology)
from vllm_torchtpu.worker.tpu_worker import (DEBUG_TPU_LOCAL_RANK_OFFSET_ENV,
                                             _debug_tpu_local_rank_offset)


def test_pcp_v2_validation_skips_gpu_only_checks(monkeypatch):
    from vllm.config import vllm as vllm_config

    from vllm_torchtpu import _patch_vllm_pcp_v2_validation

    def validate(config):
        assert vllm_config.HAS_TRITON
        raise RuntimeError("other validation")

    def get_unsupported(config):
        return ["prefill context parallelism", "other feature"]

    monkeypatch.setattr(vllm_config, "HAS_TRITON", False)
    monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", validate)
    monkeypatch.setattr(VllmConfig,
                        "_get_v2_model_runner_unsupported_features",
                        get_unsupported)
    _patch_vllm_pcp_v2_validation()
    config = SimpleNamespace(parallel_config=SimpleNamespace(
        prefill_context_parallel_size=4))

    assert VllmConfig._get_v2_model_runner_unsupported_features(config) == [
        "other feature"
    ]
    with pytest.raises(RuntimeError, match="other validation"):
        VllmConfig._validate_v2_model_runner(config)
    assert not vllm_config.HAS_TRITON


def test_grouped_topk_dynamic_compile_wrapper_is_unwrapped(monkeypatch):
    target = (
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router")

    def original_grouped_topk():
        pass

    def compiled_grouped_topk():
        pass

    compiled_grouped_topk.__wrapped__ = original_grouped_topk
    grouped_topk_module = SimpleNamespace(grouped_topk=compiled_grouped_topk)

    def fake_import_module(module_path):
        if module_path == target:
            return grouped_topk_module
        return SimpleNamespace()

    monkeypatch.setattr("importlib.import_module", fake_import_module)
    monkeypatch.setattr(tpu_platform, "_dynamic_compile_unwrapped", False)

    tpu_platform._unwrap_dynamic_compile_fns()

    assert grouped_topk_module.grouped_topk is original_grouped_topk


def test_scheduler_mamba_split_accepts_external_kv_tokens():
    # TPU PD decode feeds producer-restored KV in as external computed tokens.
    # Upstream vLLM must count them in the block-aligned split boundary.
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        use_eagle=False,
        mamba_partial_cache_hit=False,
        hash_block_size=16,
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=33,
        num_tokens=33,
        shared_prefix_boundary=0,
    )

    num_new_tokens = Scheduler._mamba_block_aligned_split(
        scheduler,
        request,
        num_new_tokens=20,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=16,
    )

    assert num_new_tokens == 16


def test_debug_tpu_local_rank_offset_env(monkeypatch):
    monkeypatch.delenv(DEBUG_TPU_LOCAL_RANK_OFFSET_ENV, raising=False)
    assert _debug_tpu_local_rank_offset() == 0

    monkeypatch.setenv(DEBUG_TPU_LOCAL_RANK_OFFSET_ENV, "4")
    assert _debug_tpu_local_rank_offset() == 4

    monkeypatch.setenv(DEBUG_TPU_LOCAL_RANK_OFFSET_ENV, "not-an-int")
    with pytest.raises(ValueError, match=DEBUG_TPU_LOCAL_RANK_OFFSET_ENV):
        _debug_tpu_local_rank_offset()


def test_prepare_singlehost_tpu_env_skips_distributed_bootstrap_for_tp1(
        monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")

    TpuPlatform._prepare_singlehost_tpu_env(1)

    assert "WORLD_SIZE" not in os.environ


def test_prepare_singlehost_tpu_env_keeps_inherited_slice_bootstrap(
        monkeypatch):
    """A DP engine must not clear the slice env it inherited from the parent."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES",
                       ",".join(f"localhost:{p}" for p in range(1000, 1008)))
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2,2,1,2")

    with patch.object(TpuPlatform, "_get_tpu_topology",
                      return_value="2,2,1,2"):
        TpuPlatform._prepare_singlehost_tpu_env(8)

    assert os.environ["WORLD_SIZE"] == "8"
    assert len(os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == 8
    assert os.environ["TORCH_TPU_TOPOLOGY"] == "2,2,1,2"


def set_mock_attn_to_vllm_config(vllm_config, page_size, min_page_size):
    new_vllm_config = copy.copy(vllm_config)

    mock_impl = MagicMock()
    mock_impl.get_page_size.return_value = page_size
    mock_impl.get_min_page_size.return_value = min_page_size
    mock_impl.is_ssm.return_value = False
    mock_attn = MagicMock(spec=Attention)
    mock_attn.get_attn_backend.return_value = mock_impl
    new_vllm_config.compilation_config.static_forward_context = {
        "layer": mock_attn
    }
    return new_vllm_config, mock_impl


class TestTpuPlatform:

    @pytest.fixture
    def vllm_config(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = MagicMock(spec=ModelConfig)
        vllm_config.model_config.dtype = torch.bfloat16
        vllm_config.model_config.is_hybrid = False
        vllm_config.model_config.hf_config = None
        vllm_config.cache_config = MagicMock(spec=CacheConfig)
        vllm_config.cache_config.block_size = None
        vllm_config.cache_config.enable_prefix_caching = False
        vllm_config.cache_config.mamba_cache_mode = None
        vllm_config.compilation_config = MagicMock()
        vllm_config.compilation_config.mode = MagicMock()
        vllm_config.compilation_config.compile_sizes = [16, 32]
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 2048
        vllm_config.scheduler_config.is_multimodal_model = False
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.speculative_config = None
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.world_size_across_dp = 1
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {}
        return vllm_config

    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_hybrid_block_size(self, mock_prepare_env,
                                                       mock_apply_patches,
                                                       vllm_config):
        vllm_config.model_config.is_hybrid = True
        # A hybrid architecture with no pooled state path runs the split
        # layout, kv-transfer or not.
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = "TPUConnector"
        vllm_config.cache_config.block_size = 123  # already set

        vllm_config, mock_pallas = set_mock_attn_to_vllm_config(
            vllm_config, 999, 16)

        with patch("vllm_torchtpu.platforms.tpu_platform."
                   "update_tpu_block_size_and_slot_config") as mock_update:
            TpuPlatform.update_block_size_for_backend(vllm_config)

        # The split-layout path must not run the block-size derivation
        # helper.
        mock_update.assert_not_called()
        # Verify block_size wasn't overridden by get_page_size
        assert vllm_config.cache_config.block_size == 123
        # And get_page_size shouldn't even be called because is_hybrid is True
        mock_pallas.get_page_size.assert_not_called()

    @patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1"})
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_hybrid_derives_block_size_with_env(
            self, mock_prepare_env, mock_apply_patches, vllm_config):
        vllm_config.model_config.is_hybrid = True
        vllm_config.cache_config.block_size = 123  # already set

        vllm_config, mock_pallas = set_mock_attn_to_vllm_config(
            vllm_config, 999, 16)

        with patch("vllm_torchtpu.platforms.tpu_platform."
                   "update_tpu_block_size_and_slot_config") as mock_update:
            TpuPlatform.update_block_size_for_backend(vllm_config)

        # The env opts into the unified layout family: block size and slot
        # sizing are owned by the derivation helper (its math is covered by
        # test_tpu_block_size_utils.py).
        mock_update.assert_called_once_with(vllm_config, mock_pallas)

    def test_unified_kv_layout_enablement_contract(self, vllm_config,
                                                   monkeypatch):
        from vllm_torchtpu.platforms.tpu_block_size_utils import \
            unified_kv_layout_enabled
        monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
        pooled_arch = "Qwen3_5ForConditionalGeneration"
        unpooled_arch = "KimiLinearForCausalLM"
        vllm_config.model_config.is_hybrid = True
        vllm_config.kv_transfer_config = None

        # A hybrid architecture without a pooled state path keeps the
        # per-layer caches: its layers unpack a separate conv/ssm pair.
        vllm_config.model_config.architecture = unpooled_arch
        assert not unified_kv_layout_enabled(vllm_config)

        # A pooled GDN architecture selects the pool with no env set.
        vllm_config.model_config.architecture = pooled_arch
        assert unified_kv_layout_enabled(vllm_config)

        # An explicit setting wins in either direction.
        with patch.dict("os.environ",
                        {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "0"}):
            assert not unified_kv_layout_enabled(vllm_config)

        vllm_config.model_config.architecture = unpooled_arch
        with patch.dict("os.environ",
                        {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1"}):
            assert unified_kv_layout_enabled(vllm_config)

            # The pooled layout is selected for local and transfer
            # deployments alike.
            vllm_config.kv_transfer_config = MagicMock()
            assert unified_kv_layout_enabled(vllm_config)

    @pytest.mark.parametrize(
        ("mamba_cache_mode", "speculative_config", "async_scheduling",
         "message"),
        [
            ("all", None, False, "mamba_cache_mode='align'"),
            ("align", MagicMock(), False, "Speculative decoding"),
            ("align", None, True, None),
        ],
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_validates_mamba_apc_modes(
            self, mock_prepare_env, mock_apply_patches, vllm_config,
            mamba_cache_mode, speculative_config, async_scheduling, message):
        vllm_config.model_config.is_hybrid = True
        # A pooled GDN architecture: hybrid prefix caching needs the pool.
        vllm_config.model_config.architecture = "Qwen3_5ForConditionalGeneration"
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.enable_prefix_caching = True
        vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
        vllm_config.speculative_config = speculative_config
        vllm_config.scheduler_config.async_scheduling = async_scheduling
        # Attention-DP runs the split layout; async+APC is only rejected
        # there (the pool's seed copies support async scheduling).
        vllm_config.parallel_config.data_parallel_size = 8
        # The non-raising row reaches the single-host DP env setup; a falsy
        # master ip takes the code's "localhost" fallback.
        vllm_config.parallel_config.data_parallel_master_ip = ""

        if message is None:
            TpuPlatform.check_and_update_config(vllm_config)
        else:
            with pytest.raises(NotImplementedError, match=message):
                TpuPlatform.check_and_update_config(vllm_config)

    @patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "0"})
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_rejects_hybrid_apc_without_pool(
            self, mock_prepare_env, mock_apply_patches, vllm_config):
        # Opting out of the pool leaves align mode with no seed copies, so a
        # prefix-cache hit would restore no Mamba state at all.
        vllm_config.model_config.is_hybrid = True
        vllm_config.model_config.architecture = "Qwen3_5ForConditionalGeneration"
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.enable_prefix_caching = True
        vllm_config.cache_config.mamba_cache_mode = "align"

        with pytest.raises(NotImplementedError, match="unified KV pool"):
            TpuPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize(
        "connector_name",
        ["TPUConnector", "TPURaidenConnector", "TPUMultiConnector"],
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_accepts_tpu_disagg_connectors(
            self, mock_prepare_env, mock_apply_patches, vllm_config,
            connector_name):
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = connector_name
        vllm_config.cache_config.block_size = 16

        TpuPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize(
        ("is_hybrid", "pool_env", "expect_error"),
        [
            # Hybrid CPU offloading transfers whole pool rows; without the
            # unified block pool there is no uniform per-block row to copy.
            (True, False, True),
            (True, True, False),
            # Dense offloading works on either layout.
            (False, True, False),
            (False, False, False),
        ],
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_gates_hybrid_offloading_on_pool(
            self, mock_prepare_env, mock_apply_patches, vllm_config,
            monkeypatch, is_hybrid, pool_env, expect_error):
        vllm_config.model_config.is_hybrid = is_hybrid
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.cache_dtype = "auto"
        if is_hybrid:
            # Prefix caching stays off: hybrid APC without the pool is
            # rejected earlier, by its own gate, and would mask this one.
            vllm_config.cache_config.mamba_cache_mode = "align"
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = "OffloadingConnector"

        if pool_env:
            monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
        else:
            monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL",
                               raising=False)

        with patch("vllm_torchtpu.platforms.tpu_platform."
                   "update_tpu_block_size_and_slot_config"):
            if expect_error:
                with pytest.raises(ValueError, match="unified block\\s+pool"):
                    TpuPlatform.check_and_update_config(vllm_config)
            else:
                TpuPlatform.check_and_update_config(vllm_config)

    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_language_model_only_multimodal_keeps_chunked_mm_input(
            self, mock_prepare_env, mock_apply_patches, vllm_config):
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=True, limit_per_prompt={"image": 1})
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.is_multimodal_model = True
        vllm_config.scheduler_config.disable_chunked_mm_input = False

        TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.disable_chunked_mm_input is False

    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_multimodal_keeps_chunked_mm_input(self, mock_prepare_env,
                                               mock_apply_patches,
                                               vllm_config):
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=False, limit_per_prompt={"image": 1})
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.is_multimodal_model = True
        vllm_config.scheduler_config.disable_chunked_mm_input = False

        TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.disable_chunked_mm_input is False

    def test_get_tpu_multihost_topology(self, monkeypatch):
        # Test env override via TORCH_TPU_TOPOLOGY
        monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "4,4,1")
        assert get_tpu_multihost_topology(16) == "4,4,1"
        monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)

        # Test syntax normalization ('AxBxC' -> 'A,B,C')
        monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "4x4x1")
        assert get_tpu_multihost_topology(16) == "4,4,1"
        monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "8x16x1")
        assert get_tpu_multihost_topology(128) == "8,16,1"
        monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)

        # Test 2D torus fallback map lookup (v6e / v5e)
        assert get_tpu_multihost_topology(16, device_name="TPU v6e") == "4,4,1"
        assert get_tpu_multihost_topology(32, device_name="TPU v5e") == "4,8,1"
        assert get_tpu_multihost_topology(64, device_name="TPU v6e") == "8,8,1"

        # Test 3D torus fallback map lookup (v4 / v5p), where megacore
        # presents both cores of a chip as one device.
        assert get_tpu_multihost_topology(16,
                                          device_name="TPU v4") == "2,2,4,2"
        assert get_tpu_multihost_topology(32,
                                          device_name="TPU v5p") == "2,4,4,2"
        assert get_tpu_multihost_topology(64,
                                          device_name="TPU v4") == "4,4,4,2"
        assert get_tpu_multihost_topology(128,
                                          device_name="TPU v5p") == "4,4,8,2"
        assert get_tpu_multihost_topology(256,
                                          device_name="TPU v4") == "4,8,8,2"

        # Test dual-device 3D torus lookup (v7x / Ironwood). Same meshes as
        # v4 / v5p, but each chip contributes two devices, so a given mesh is
        # reached at twice the world size. Both spellings of the device name
        # have to select this map.
        assert get_tpu_multihost_topology(16,
                                          device_name="TPU v7") == "2,2,2,2"
        assert get_tpu_multihost_topology(32, device_name="TPU7x") == "2,2,4,2"
        assert get_tpu_multihost_topology(64,
                                          device_name="TPU v7x") == "2,4,4,2"
        assert get_tpu_multihost_topology(512,
                                          device_name="TPU v7") == "4,8,8,2"

        # A mesh must describe exactly as many devices as were asked for,
        # otherwise the slice builder sizes its worker address list against a
        # world that does not exist. Devices per chip is what separates the
        # maps, so it is what the invariant is parameterized on.
        for topo_map, devices_per_chip in (
            (TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP, 1),
            (TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP, 1),
            (TPU_3D_TORUS_DUAL_DEVICE_TOPOLOGY_MAP, 2),
        ):
            for world_size, topo in topo_map.items():
                dims = [int(d) for d in topo.split(",")]
                # The trailing T dimension is cores per chip, never devices.
                chips = math.prod(dims[:-1])
                assert chips * devices_per_chip == world_size

        # Test ValueError on unsupported device counts
        with pytest.raises(ValueError,
                           match="Cannot find topology for 10 devices"):
            get_tpu_multihost_topology(10, device_name="TPU v6e")


class TestPhasedProfilingConfigValidation:
    """Both mistakes here yield a run that looks healthy and writes no
    traces, so they are rejected at startup rather than ignored."""

    @staticmethod
    def _config(additional_config=None, torch_profiler_dir=""):
        vllm_config = MagicMock()
        vllm_config.additional_config = additional_config or {}
        vllm_config.profiler_config.torch_profiler_dir = torch_profiler_dir
        return vllm_config

    def test_legacy_additional_config_key_is_rejected(self):
        with pytest.raises(AssertionError, match="USE_PHASED_PROFILER"):
            _validate_phased_profiling_config(
                self._config({"phased_profiling_dir": "/tmp/phased"}))

    def test_phased_without_a_trace_dir_is_rejected(self, monkeypatch):
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")

        with pytest.raises(AssertionError, match="nowhere to write traces"):
            _validate_phased_profiling_config(self._config())

    def test_phased_with_a_trace_dir_passes(self, monkeypatch):
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")

        _validate_phased_profiling_config(
            self._config(torch_profiler_dir="/tmp/phased"))

    def test_unprofiled_run_passes(self, monkeypatch):
        monkeypatch.delenv("USE_PHASED_PROFILER", raising=False)

        _validate_phased_profiling_config(self._config())


def test_config_hook_registers_tpu_kv_connectors_by_name():
    """Verify TPU KV connectors are registered in the factory by name idempotently.

    When loaded dynamically via module paths, connectors are not registered by name
    in KVConnectorFactory. However, the API server's multi-connector metrics path resolves
    child connectors via get_connector_class_by_name().

    This test verifies that _register_tpu_kv_connectors() populates the name registry
    safely across repeated invocations without raising duplicate-registration errors.
    """
    from vllm.distributed.kv_transfer.kv_connector.factory import \
        KVConnectorFactory

    from vllm_torchtpu.platforms.tpu_platform import \
        _register_tpu_kv_connectors

    _register_tpu_kv_connectors()
    _register_tpu_kv_connectors(
    )  # Idempotent: repeated registrations must succeed.

    for name in ("TPURaidenConnector", "TPUMultiConnector",
                 "TPURaidenOffloadingConnector"):
        cls = KVConnectorFactory.get_connector_class_by_name(name)
        assert cls.__name__ == name
