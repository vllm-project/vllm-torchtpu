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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.worker.tpu_worker import (DEBUG_TPU_LOCAL_RANK_OFFSET_ENV,
                                             _debug_tpu_local_rank_offset)


def test_scheduler_mamba_split_accepts_external_kv_tokens():
    # TPU PD decode feeds producer-restored KV in as external computed tokens.
    # Upstream vLLM must count them in the block-aligned split boundary.
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=16),
                                use_eagle=False)
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=33,
        num_tokens=33,
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
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_hybrid_block_size(self, mock_vllm_envs,
                                                       mock_prepare_env,
                                                       mock_apply_patches,
                                                       vllm_config):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.is_hybrid = True
        # kv-transfer deployments run the split layout (the pool is opt-in
        # via the env and never engages for kv-transfer).
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = "TPUConnector"
        vllm_config.cache_config.block_size = 123  # already set

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 999
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules',
            {
                'vllm_torchtpu.layers.vllm.attention':
                MagicMock(PallasAttentionBackend=mock_pallas)
            }), patch("vllm_torchtpu.platforms.tpu_platform."
                      "update_tpu_block_size_and_slot_config") as mock_update:
            TpuPlatform.check_and_update_config(vllm_config)

        # Without the unified-layout env the split-layout path must not run
        # the block-size derivation helper.
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
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_hybrid_derives_block_size_with_env(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.is_hybrid = True
        vllm_config.cache_config.block_size = 123  # already set

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 999
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules',
            {
                'vllm_torchtpu.layers.vllm.attention':
                MagicMock(PallasAttentionBackend=mock_pallas)
            }), patch("vllm_torchtpu.platforms.tpu_platform."
                      "update_tpu_block_size_and_slot_config") as mock_update:
            TpuPlatform.check_and_update_config(vllm_config)

        # The env opts into the unified layout family: block size and slot
        # sizing are owned by the derivation helper (its math is covered by
        # test_tpu_block_size_utils.py).
        mock_update.assert_called_once_with(vllm_config, mock_pallas)

    def test_unified_kv_layout_enablement_contract(self, vllm_config):
        from vllm_torchtpu.platforms.tpu_block_size_utils import \
            unified_kv_layout_enabled
        vllm_config.model_config.is_hybrid = True

        # No env: ordinary non-unified allocation remains selected.
        vllm_config.kv_transfer_config = None
        assert not unified_kv_layout_enabled(vllm_config)

        with patch.dict("os.environ",
                        {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1"}):
            # The same env selects the pooled layout for local and transfer
            # deployments.
            assert unified_kv_layout_enabled(vllm_config)

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
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_validates_mamba_apc_modes(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config, mamba_cache_mode, speculative_config,
            async_scheduling, message):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.is_hybrid = True
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

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 256
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'vllm_torchtpu.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
            if message is None:
                TpuPlatform.check_and_update_config(vllm_config)
            else:
                with pytest.raises(NotImplementedError, match=message):
                    TpuPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize(
        "connector_name",
        ["TPUConnector", "TPURaidenConnector", "TPUMultiConnector"],
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_accepts_tpu_disagg_connectors(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config, connector_name):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = connector_name
        vllm_config.cache_config.block_size = 16

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 16
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'vllm_torchtpu.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
            TpuPlatform.check_and_update_config(vllm_config)

    @patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1"})
    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_accepts_v2_connector_with_unified_pool(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = "TPUConnectorV2"
        vllm_config.cache_config.block_size = 16
        vllm_config.cache_config.cache_dtype = "auto"

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 16
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'vllm_torchtpu.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
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
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_check_and_update_config_gates_hybrid_offloading_on_pool(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config, monkeypatch, is_hybrid, pool_env, expect_error):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.is_hybrid = is_hybrid
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.cache_dtype = "auto"
        if is_hybrid:
            vllm_config.cache_config.enable_prefix_caching = True
            vllm_config.cache_config.mamba_cache_mode = "align"
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = "OffloadingConnector"

        if pool_env:
            monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
        else:
            monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL",
                               raising=False)

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 256
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'vllm_torchtpu.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }), patch("vllm_torchtpu.platforms.tpu_platform."
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
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_language_model_only_multimodal_keeps_chunked_mm_input(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=True, limit_per_prompt={"image": 1})
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.is_multimodal_model = True
        vllm_config.scheduler_config.disable_chunked_mm_input = False

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 16
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'vllm_torchtpu.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
            TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.disable_chunked_mm_input is False

    @patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    @patch("vllm_torchtpu.platforms.tpu_platform.vllm_envs")
    def test_multimodal_forces_disable_chunked_mm_input(
            self, mock_vllm_envs, mock_prepare_env, mock_apply_patches,
            vllm_config):
        mock_vllm_envs.VLLM_TPU_USING_PATHWAYS = False
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=False, limit_per_prompt={"image": 1})
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.is_multimodal_model = True
        vllm_config.scheduler_config.disable_chunked_mm_input = False

        mock_pallas = MagicMock()
        mock_pallas.get_page_size.return_value = 16
        mock_pallas.get_min_page_size.return_value = 16

        with patch.dict(
                'sys.modules', {
                    'vllm_torchtpu.layers.vllm.attention':
                    MagicMock(PallasAttentionBackend=mock_pallas)
                }):
            TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.disable_chunked_mm_input is True
