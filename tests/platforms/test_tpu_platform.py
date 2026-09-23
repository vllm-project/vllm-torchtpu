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
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.compilation import DynamicShapesType
from vllm.model_executor.layers.attention import Attention
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCoreProc

import vllm_torchtpu as plugin
import vllm_torchtpu.platforms.tpu_platform as tpu_platform
from vllm_torchtpu import patch_registry
from vllm_torchtpu.platforms.tpu_platform import (
    TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP,
    TPU_3D_TORUS_DUAL_DEVICE_TOPOLOGY_MAP,
    TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP,
    TPU_8I_MULTIHOST_TOPOLOGY_MAP,
    TpuPlatform,
    _validate_phased_profiling_config,
    get_tpu_multihost_topology,
)


def test_grouped_topk_dynamic_compile_wrapper_is_unwrapped(monkeypatch):
    target = "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"

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
        block_size=16,
        use_eagle=False,
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
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
    """0 when unset, an int otherwise, and an error that names the variable
    when the value does not parse."""
    from vllm_torchtpu import envs

    monkeypatch.delenv("DEBUG_TPU_LOCAL_RANK_OFFSET", raising=False)
    assert envs.DEBUG_TPU_LOCAL_RANK_OFFSET == 0

    monkeypatch.setenv("DEBUG_TPU_LOCAL_RANK_OFFSET", "4")
    assert envs.DEBUG_TPU_LOCAL_RANK_OFFSET == 4

    monkeypatch.setenv("DEBUG_TPU_LOCAL_RANK_OFFSET", "not-an-int")
    with pytest.raises(ValueError, match="DEBUG_TPU_LOCAL_RANK_OFFSET"):
        _ = envs.DEBUG_TPU_LOCAL_RANK_OFFSET


def test_prepare_singlehost_tpu_env_skips_distributed_bootstrap_for_tp1(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")

    TpuPlatform._prepare_singlehost_tpu_env(1)

    assert "WORLD_SIZE" not in os.environ


def test_prepare_singlehost_tpu_env_keeps_inherited_slice_bootstrap(monkeypatch):
    """A DP engine must not clear the slice env it inherited from the parent."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv(
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        ",".join(f"localhost:{p}" for p in range(1000, 1008)),
    )
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2,2,1,2")

    with patch.object(TpuPlatform, "_get_tpu_topology", return_value="2,2,1,2"):
        TpuPlatform._prepare_singlehost_tpu_env(8)

    assert os.environ["WORLD_SIZE"] == "8"
    assert len(os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == 8
    assert os.environ["TORCH_TPU_TOPOLOGY"] == "2,2,1,2"


@pytest.mark.parametrize("inherited_endpoint", [False, True])
def test_singlehost_workers_share_native_endpoint_with_filestore(
    monkeypatch, tmp_path, inherited_endpoint
):
    """Native bootstrap needs one shared endpoint even with a c10d FileStore."""
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    if inherited_endpoint:
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29457")
    monkeypatch.setenv(
        "TORCH_TPU_SLICEBUILDER_ADDRESSES", "localhost:10001,localhost:10002"
    )
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "1,1,1,2")
    with patch.object(
        tpu_platform.portpicker,
        "pick_unused_port",
        wraps=tpu_platform.portpicker.pick_unused_port,
    ) as pick:
        TpuPlatform._prepare_singlehost_tpu_env(2)
        # Config copies re-enter preparation before worker initialization.
        TpuPlatform._prepare_singlehost_tpu_env(2)
    assert pick.call_count == (0 if inherited_endpoint else 1)
    endpoint = (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    if inherited_endpoint:
        assert endpoint == ("127.0.0.1", "29457")

    script = """
import os
import sys
from datetime import timedelta
import torch.distributed as dist
endpoint = (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
assert 0 < int(endpoint[1]) < 65536
dist.init_process_group('gloo', init_method=sys.argv[1], rank=int(sys.argv[2]),
                        world_size=2, timeout=timedelta(seconds=30))
peers = [None, None]
dist.all_gather_object(peers, endpoint)
assert peers == [endpoint, endpoint], peers
dist.destroy_process_group()
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                (tmp_path / "rendezvous").as_uri(),
                str(rank),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=45)
            assert process.returncode == 0, (stdout, stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def set_mock_attn_to_vllm_config(vllm_config, page_size, min_page_size):
    new_vllm_config = copy.copy(vllm_config)

    mock_impl = MagicMock()
    mock_impl.get_page_size.return_value = page_size
    mock_impl.get_min_page_size.return_value = min_page_size
    mock_impl.is_ssm.return_value = False
    mock_attn = MagicMock(spec=Attention)
    mock_attn.get_attn_backend.return_value = mock_impl
    new_vllm_config.compilation_config.static_forward_context = {"layer": mock_attn}
    return new_vllm_config, mock_impl


class TestTpuPlatform:
    @pytest.fixture
    def vllm_config(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.model_config = MagicMock(spec=ModelConfig)
        vllm_config.model_config.dtype = torch.bfloat16
        vllm_config.model_config.is_hybrid = False
        vllm_config.model_config.hf_config = None
        vllm_config.cache_config = CacheConfig()
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
        vllm_config.parallel_config.nnodes = 1
        vllm_config.parallel_config.node_rank = 0
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.world_size_across_dp = 1
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {}
        return vllm_config

    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_hybrid_block_size(
        self, mock_prepare_env, mock_apply_patches, vllm_config
    ):
        vllm_config.model_config.is_hybrid = True
        # A hybrid architecture with no pooled state path runs the split
        # layout, kv-transfer or not.
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_connector = "TPUConnector"
        vllm_config.cache_config.block_size = 123  # already set

        vllm_config, mock_pallas = set_mock_attn_to_vllm_config(vllm_config, 999, 16)

        with patch(
            "vllm_torchtpu.platforms.tpu_platform.update_tpu_block_size_and_slot_config"
        ) as mock_update:
            TpuPlatform.update_block_size_for_backend(vllm_config)

        # The split-layout path must not run the block-size derivation
        # helper.
        mock_update.assert_not_called()
        # Verify block_size wasn't overridden by get_page_size
        assert vllm_config.cache_config.block_size == 123
        # And get_page_size shouldn't even be called because is_hybrid is True
        mock_pallas.get_page_size.assert_not_called()

    @patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1"})
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_hybrid_derives_block_size_with_env(
        self, mock_prepare_env, mock_apply_patches, vllm_config
    ):
        vllm_config.model_config.is_hybrid = True
        vllm_config.cache_config.block_size = 123  # already set

        vllm_config, mock_pallas = set_mock_attn_to_vllm_config(vllm_config, 999, 16)

        with patch(
            "vllm_torchtpu.platforms.tpu_platform.update_tpu_block_size_and_slot_config"
        ) as mock_update:
            TpuPlatform.update_block_size_for_backend(vllm_config)

        # The env opts into the unified layout family: block size and slot
        # sizing are owned by the derivation helper (its math is covered by
        # test_tpu_block_size_utils.py).
        mock_update.assert_called_once_with(vllm_config, mock_pallas)

    def test_unified_kv_layout_enablement_contract(self, vllm_config, monkeypatch):
        from vllm_torchtpu.platforms.tpu_block_size_utils import (
            unified_kv_layout_enabled,
        )

        monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
        pooled_arch = "Qwen3_5ForConditionalGeneration"
        unpooled_arch = "BambaForCausalLM"
        vllm_config.model_config.is_hybrid = True
        vllm_config.kv_transfer_config = None

        # A hybrid architecture without a pooled state path keeps the
        # per-layer caches: its layers unpack a separate conv/ssm pair.
        vllm_config.model_config.architecture = unpooled_arch
        assert not unified_kv_layout_enabled(vllm_config)

        # Both direct text-only and multimodal-wrapper Qwen3.5 classes use
        # the same pooled GDN layer implementation.
        for pooled_arch in (
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
        ):
            vllm_config.model_config.architecture = pooled_arch
            assert unified_kv_layout_enabled(vllm_config)

        # An explicit setting wins in either direction.
        vllm_config.model_config.architecture = "Qwen3_5ForConditionalGeneration"
        with patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "0"}):
            assert not unified_kv_layout_enabled(vllm_config)

        vllm_config.model_config.architecture = unpooled_arch
        with patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1"}):
            assert unified_kv_layout_enabled(vllm_config)

            # The pooled layout is selected for local and transfer
            # deployments alike.
            vllm_config.kv_transfer_config = MagicMock()
            assert unified_kv_layout_enabled(vllm_config)

    @pytest.mark.parametrize(
        ("mamba_cache_mode", "speculative_config", "async_scheduling", "message"),
        [
            ("all", None, False, "mamba_cache_mode='align'"),
            ("align", None, True, None),
        ],
    )
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_validates_mamba_apc_modes(
        self,
        mock_prepare_env,
        mock_apply_patches,
        vllm_config,
        mamba_cache_mode,
        speculative_config,
        async_scheduling,
        message,
    ):
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
        # Single-host DP: every replica is local, so this reaches
        # _prepare_singlehost_tpu_env rather than the multi-host DP
        # rendezvous (which needs data_parallel_size_local < data_parallel_size).
        vllm_config.parallel_config.data_parallel_size_local = 8
        # The non-raising row reaches the single-host DP env setup; a falsy
        # master ip takes the code's "localhost" fallback.
        vllm_config.parallel_config.data_parallel_master_ip = ""

        if message is None:
            TpuPlatform.check_and_update_config(vllm_config)
        else:
            with pytest.raises(NotImplementedError, match=message):
                TpuPlatform.check_and_update_config(vllm_config)

    @patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "0"})
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_rejects_hybrid_apc_without_pool(
        self, mock_prepare_env, mock_apply_patches, vllm_config
    ):
        # Opting out of the pool leaves align mode with no seed copies, so a
        # prefix-cache hit would restore no Mamba state at all.
        vllm_config.model_config.is_hybrid = True
        vllm_config.model_config.architecture = "Qwen3_5ForConditionalGeneration"
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.enable_prefix_caching = True
        vllm_config.cache_config.mamba_cache_mode = "align"

        with pytest.raises(NotImplementedError, match="unified KV pool"):
            TpuPlatform.check_and_update_config(vllm_config)

    @patch.dict("os.environ", {"TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "0"})
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_decode_bench_allows_hybrid_apc_without_pool(
        self, mock_prepare_env, mock_apply_patches, vllm_config
    ):
        vllm_config.model_config.is_hybrid = True
        vllm_config.model_config.architecture = "KimiK3ForConditionalGeneration"
        vllm_config.cache_config.enable_prefix_caching = True
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="DecodeBenchConnector", kv_role="kv_both"
        )

        TpuPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize(
        "connector_name",
        [
            "DecodeBenchConnector",
            "TPUConnector",
            "TPURaidenConnector",
            "TPUMultiConnector",
        ],
    )
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_accepts_tpu_disagg_connectors(
        self, mock_prepare_env, mock_apply_patches, vllm_config, connector_name
    ):
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector=connector_name, kv_role="kv_both"
        )
        if connector_name == "TPUConnector":
            vllm_config.kv_transfer_config.kv_connector_module_path = (
                "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
            )
        elif connector_name == "TPUMultiConnector":
            vllm_config.kv_transfer_config.kv_connector_extra_config = {
                "connectors": [
                    {
                        "kv_connector": "DecodeBenchConnector",
                        "kv_role": "kv_both",
                    }
                ]
            }
        vllm_config.cache_config.block_size = 16

        TpuPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize(
        ("compile_sizes", "shapes_type", "expected_type"),
        [
            # Size 1 under backed shapes would crash in PiecewiseBackend.
            (
                [1, 16],
                DynamicShapesType.BACKED,
                DynamicShapesType.BACKED_SIZE_OBLIVIOUS,
            ),
            ([16, 32], DynamicShapesType.BACKED, DynamicShapesType.BACKED),
            # A user-chosen non-backed type is never overridden.
            ([1, 16], DynamicShapesType.UNBACKED, DynamicShapesType.UNBACKED),
        ],
    )
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_dynamic_shapes_for_compile_size_1(
        self,
        mock_prepare_env,
        mock_apply_patches,
        vllm_config,
        compile_sizes,
        shapes_type,
        expected_type,
    ):
        vllm_config.compilation_config.compile_sizes = compile_sizes
        vllm_config.compilation_config.dynamic_shapes_config.type = shapes_type

        TpuPlatform.check_and_update_config(vllm_config)

        assert (
            vllm_config.compilation_config.dynamic_shapes_config.type == expected_type
        )

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
    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_check_and_update_config_gates_hybrid_offloading_on_pool(
        self,
        mock_prepare_env,
        mock_apply_patches,
        vllm_config,
        monkeypatch,
        is_hybrid,
        pool_env,
        expect_error,
    ):
        vllm_config.model_config.is_hybrid = is_hybrid
        vllm_config.cache_config.block_size = 256
        vllm_config.cache_config.cache_dtype = "auto"
        if is_hybrid:
            # Prefix caching stays off: hybrid APC without the pool is
            # rejected earlier, by its own gate, and would mask this one.
            vllm_config.cache_config.mamba_cache_mode = "align"
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="TPURaidenOffloadingConnector", kv_role="kv_both"
        )

        if pool_env:
            monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
        else:
            monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)

        with patch(
            "vllm_torchtpu.platforms.tpu_platform.update_tpu_block_size_and_slot_config"
        ):
            if expect_error:
                with pytest.raises(ValueError, match="unified block\\s+pool"):
                    TpuPlatform.check_and_update_config(vllm_config)
            else:
                TpuPlatform.check_and_update_config(vllm_config)

    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_language_model_only_multimodal_keeps_chunked_mm_input(
        self, mock_prepare_env, mock_apply_patches, vllm_config
    ):
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=True, limit_per_prompt={"image": 1}
        )
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.is_multimodal_model = True
        vllm_config.scheduler_config.disable_chunked_mm_input = False

        TpuPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.disable_chunked_mm_input is False

    @patch("vllm_torchtpu.patch_registry.apply")
    @patch(
        "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
    )
    def test_multimodal_keeps_chunked_mm_input(
        self, mock_prepare_env, mock_apply_patches, vllm_config
    ):
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=False, limit_per_prompt={"image": 1}
        )
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
        assert get_tpu_multihost_topology(16, device_name="TPU v4") == "2,2,4,2"
        assert get_tpu_multihost_topology(32, device_name="TPU v5p") == "2,4,4,2"
        assert get_tpu_multihost_topology(64, device_name="TPU v4") == "4,4,4,2"
        assert get_tpu_multihost_topology(128, device_name="TPU v5p") == "4,4,8,2"
        assert get_tpu_multihost_topology(256, device_name="TPU v4") == "4,8,8,2"

        # Test dual-device 3D torus lookup (v7x / Ironwood). Same meshes as
        # v4 / v5p, but each chip contributes two devices, so a given mesh is
        # reached at twice the world size. Both spellings of the device name
        # have to select this map.
        assert get_tpu_multihost_topology(16, device_name="TPU v7") == "2,2,2,2"
        assert get_tpu_multihost_topology(32, device_name="TPU7x") == "2,2,4,2"
        assert get_tpu_multihost_topology(64, device_name="TPU v7x") == "2,4,4,2"
        assert get_tpu_multihost_topology(512, device_name="TPU v7") == "4,8,8,2"

        # Test TPU v8i / BoardFly lookup.
        assert get_tpu_multihost_topology(4, device_name="TPU v8i") == "4,1,1"
        assert get_tpu_multihost_topology(8, device_name="TPU8i") == "4,2,1"
        assert get_tpu_multihost_topology(16, device_name="TPU v8i") == "4,4,1"
        assert get_tpu_multihost_topology(32, device_name="TPU v8i") == "4,8,1"
        assert get_tpu_multihost_topology(64, device_name="TPU8i") == "4,8,2"
        assert get_tpu_multihost_topology(1024, device_name="TPU v8i") == "4,8,32"

        # A mesh must describe exactly as many devices as were asked for,
        # otherwise the slice builder sizes its worker address list against a
        # world that does not exist. Devices per chip is what separates the
        # maps, so it is what the invariant is parameterized on.
        for topo_map, devices_per_chip in (
            (TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP, 1),
            (TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP, 1),
            (TPU_3D_TORUS_DUAL_DEVICE_TOPOLOGY_MAP, 2),
            (TPU_8I_MULTIHOST_TOPOLOGY_MAP, 1),
        ):
            for world_size, topo in topo_map.items():
                dims = [int(d) for d in topo.split(",")]
                # For 4D torus topologies, the trailing T dimension is cores per chip,
                # never devices. For 3D topologies (2D Torus X,Y,T and TPU 8i C,B,G),
                # total chips is the full product.
                chips = math.prod(dims[:-1]) if len(dims) == 4 else math.prod(dims)
                assert chips * devices_per_chip == world_size

        # Test ValueError on unsupported device counts
        with pytest.raises(ValueError, match="Cannot find topology for 10 devices"):
            get_tpu_multihost_topology(10, device_name="TPU v6e")

    @pytest.mark.parametrize(
        "raw_kind,expected_name",
        [
            ("TPU v5p", "TPU v5p"),
            ("TPU v5e", "TPU v5e"),
            ("TPU v6e", "TPU v6e"),
            ("TPU7x", "TPU v7"),
            ("TPU8i", "TPU v8i"),
            ("TPU v8i", "TPU v8i"),
        ],
    )
    @patch("jax.devices")
    def test_get_device_name(self, mock_devices, raw_kind, expected_name):
        """Tests get_device_name across TPU generations."""
        mock_dev = MagicMock()
        mock_dev.device_kind = raw_kind
        mock_devices.return_value = [mock_dev]

        from vllm_torchtpu.utils import get_device_name

        assert get_device_name() == expected_name


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
                self._config({"phased_profiling_dir": "/tmp/phased"})
            )

    def test_phased_without_a_trace_dir_is_rejected(self, monkeypatch):
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")

        with pytest.raises(AssertionError, match="nowhere to write traces"):
            _validate_phased_profiling_config(self._config())

    def test_phased_with_a_trace_dir_passes(self, monkeypatch):
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")

        _validate_phased_profiling_config(
            self._config(torch_profiler_dir="/tmp/phased")
        )

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
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    from vllm_torchtpu.platforms.tpu_platform import _register_tpu_kv_connectors

    _register_tpu_kv_connectors()
    _register_tpu_kv_connectors()  # Idempotent: repeated registrations must succeed.

    for name in (
        "TPURaidenConnector",
        "TPUMultiConnector",
        "TPURaidenOffloadingConnector",
    ):
        cls = KVConnectorFactory.get_connector_class_by_name(name)
        assert cls.__name__ == name


def _device_left_unset():
    """Stop ``DeviceConfig`` building ``torch.device("tpu")`` off a TPU."""
    return patch.object(TpuPlatform, "uses_host_device_handling", return_value=True)


@patch("vllm_torchtpu.patch_registry.apply")
@patch("vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env")
def test_check_and_update_config_accepts_a_config_without_a_model(
    mock_prepare_env, mock_apply_patches
):
    with _device_left_unset():
        vllm_config = VllmConfig()

    assert vllm_config.model_config is None
    assert vllm_config.parallel_config.is_moe_model is None
    # The hook reached its end rather than short-circuiting.
    assert vllm_config.compilation_config.splitting_ops == []


@patch("vllm_torchtpu.patch_registry.apply")
@patch("vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env")
def test_check_and_update_config_rejects_pcp_moe_without_expert_parallel(
    mock_prepare_env, mock_apply_patches
):
    with (
        _device_left_unset(),
        pytest.raises(NotImplementedError, match="requires --enable-expert-parallel"),
    ):
        VllmConfig(
            parallel_config=ParallelConfig(
                prefill_context_parallel_size=2, is_moe_model=True
            )
        )


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(patch_registry, "_applied", set())
    monkeypatch.setattr(patch_registry, "_applying", set())
    monkeypatch.setattr(patch_registry, "_active_stages", set())
    return patch_registry


def test_qwen_wrappers_are_installed_once_across_model_loads(
    isolated_registry, monkeypatch
):
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling

    reg = isolated_registry
    monkeypatch.setattr(reg, "PATCHES", tuple(p for p in reg.PATCHES if p.model_config))
    targets = [
        (modeling.Qwen3VLModel, "get_rope_index"),
        (modeling.Qwen3VLVisionAttention, "forward"),
    ]
    for name in (
        "Qwen3VLVisionModel",
        "Qwen3VisionTransformerPretrainedModel",
        "Qwen3_VisionTransformerPretrainedModel",
        "Qwen3_VisionTransformer",
        "Qwen3VisionTransformer",
    ):
        cls = getattr(modeling, name, None)
        if cls is not None and (cls, "forward") not in targets:
            targets.append((cls, "forward"))
    originals = [getattr(target, name) for target, name in targets]
    for (target, name), original in zip(targets, originals):
        monkeypatch.setattr(target, name, original)
    other_model = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="llama"), model="llama"
    )
    qwen_model = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_vl"), model="Qwen/Qwen3-VL-4B"
    )
    reg.apply("platform_activation", model_config=other_model)
    assert [getattr(target, name) for target, name in targets] == originals
    reg.apply("platform_activation", model_config=qwen_model)
    wrappers = [getattr(target, name) for target, name in targets]
    assert all(
        wrapper is not original for wrapper, original in zip(wrappers, originals)
    )
    for _ in range(3):
        reg.apply("model_load", model_config=qwen_model)
        reg.apply("platform_activation", model_config=qwen_model)
    assert [getattr(target, name) for target, name in targets] == wrappers


def test_engine_core_stage_precedes_original_entrypoint(monkeypatch):
    events = []
    monkeypatch.setattr(patch_registry, "apply", lambda stage: events.append(stage))
    run = Mock(side_effect=lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(
        EngineCoreProc, "_tpu_original_run_engine_core", run, raising=False
    )
    plugin._run_engine_core_with_tpu_patches("engine", rank=2)
    assert events == ["engine_core", (("engine",), {"rank": 2})]


def test_multimodal_stage_refreshes_direct_imports(isolated_registry, monkeypatch):
    import sys
    from types import ModuleType

    import vllm.model_executor.models.utils as utils

    reg = isolated_registry
    monkeypatch.setattr(
        reg,
        "PATCHES",
        tuple(
            p
            for p in reg.PATCHES
            if p.refresh == "vllm_torchtpu:_patch_vllm_merge_multimodal_embeddings"
        ),
    )
    monkeypatch.setattr(
        utils, "_merge_multimodal_embeddings", utils._merge_multimodal_embeddings
    )
    monkeypatch.setattr(utils, "_tpu_static_merge_mm_patch", False, raising=False)
    reg.apply("platform_activation")
    wrapper = utils._merge_multimodal_embeddings
    late_model = ModuleType("vllm.model_executor.models.test_late_model")
    late_model._merge_multimodal_embeddings = object()
    monkeypatch.setitem(sys.modules, late_model.__name__, late_model)
    reg.apply("engine_core")
    assert late_model._merge_multimodal_embeddings is wrapper
    assert utils._merge_multimodal_embeddings is wrapper


def test_qwen_import_failure_leaves_targets_unchanged(monkeypatch):
    import builtins

    from vllm_torchtpu.models.vllm import qwen3_vl_patch

    original_import = builtins.__import__
    original_cumsum = torch.cumsum

    def unavailable(name, *args, **kwargs):
        if name == "transformers.models.qwen3_vl.modeling_qwen3_vl":
            raise ImportError("optional module unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    config = SimpleNamespace(hf_config=None, model="Qwen/Qwen3-VL-4B")
    assert qwen3_vl_patch.maybe_patch_qwen3_vl(config) is None
    assert torch.cumsum is original_cumsum


class TestQwen3VLModelPatches:
    """Tests for Qwen3-VL platform and model-level patches."""

    def test_is_qwen3_vl_model(self):
        from vllm_torchtpu.models.vllm.qwen3_vl_patch import _is_qwen3_vl_model

        # None config
        assert not _is_qwen3_vl_model(None)

        # Matched by model_type
        cfg1 = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[]),
            model="some-model",
        )
        assert _is_qwen3_vl_model(cfg1)

        # Matched by architectures
        cfg2 = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="custom", architectures=["Qwen3VLForConditionalGeneration"]
            ),
            model="some-model",
        )
        assert _is_qwen3_vl_model(cfg2)

        # Matched by model name
        cfg3 = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="other", architectures=[]),
            model="Qwen/Qwen3-VL-Embedding-8B",
        )
        assert _is_qwen3_vl_model(cfg3)

        # Unrelated model
        cfg4 = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="llama", architectures=["LlamaForCausalLM"]
            ),
            model="meta-llama/Llama-3-8B",
        )
        assert not _is_qwen3_vl_model(cfg4)

    def test_qwen3_vl_get_rope_index_signature_binding_and_grid_fixing(self):
        from vllm_torchtpu.models.vllm.qwen3_vl_patch import (
            _patch_qwen3_vl_get_rope_index,
        )

        called_kwargs = {}

        class FakeQwen3VLModel:
            def get_rope_index(
                self,
                input_ids=None,
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=None,
            ):
                called_kwargs["input_ids"] = input_ids
                called_kwargs["image_grid_thw"] = image_grid_thw
                called_kwargs["video_grid_thw"] = video_grid_thw
                called_kwargs["attention_mask"] = attention_mask
                return torch.tensor([1, 2, 3])

        fake_modeling = SimpleNamespace(Qwen3VLModel=FakeQwen3VLModel)
        _patch_qwen3_vl_get_rope_index(fake_modeling)

        # Idempotence: second call should be a no-op
        _patch_qwen3_vl_get_rope_index(fake_modeling)

        inst = FakeQwen3VLModel()

        # Test 1D image_grid_thw is unsqueezed to 2D
        grid_1d = torch.tensor([1, 28, 28])
        res = inst.get_rope_index(image_grid_thw=grid_1d)
        assert called_kwargs["image_grid_thw"].shape == (1, 3)
        assert called_kwargs["input_ids"] is None  # Not fabricated
        assert torch.equal(res, torch.tensor([1, 2, 3]))

        # Test 2D image_grid_thw is preserved
        grid_2d = torch.tensor([[1, 28, 28], [1, 14, 14]])
        inst.get_rope_index(image_grid_thw=grid_2d)
        assert called_kwargs["image_grid_thw"].shape == (2, 3)

        # Test positional args binding
        inst.get_rope_index(None, grid_1d, None)
        assert called_kwargs["image_grid_thw"].shape == (1, 3)

    def test_qwen3_vl_vision_attention_cu_seqlens_cpu_transfer(self):
        from vllm_torchtpu.models.vllm.qwen3_vl_patch import (
            _patch_qwen3_vl_vision_attention,
        )

        called = {}

        class FakeVisionAttention:
            def forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None):
                called["cu_seqlens"] = cu_seqlens
                return hidden_states

        fake_modeling = SimpleNamespace(Qwen3VLVisionAttention=FakeVisionAttention)
        _patch_qwen3_vl_vision_attention(fake_modeling)

        # Idempotence
        _patch_qwen3_vl_vision_attention(fake_modeling)

        inst = FakeVisionAttention()
        cu_seqlens = torch.tensor([0, 10, 20])
        inst.forward(torch.randn(20, 16), cu_seqlens=cu_seqlens)
        assert called["cu_seqlens"].device.type == "cpu"
        assert torch.equal(called["cu_seqlens"], torch.tensor([0, 10, 20]))

    def test_qwen3_vl_scoped_torch_ops_context_manager(self):
        from vllm_torchtpu.models.vllm.qwen3_vl_patch import (
            _patched_masked_scatter,
            _scoped_qwen3_vl_torch_ops,
        )

        orig_cumsum = torch.cumsum
        orig_repeat_interleave = torch.repeat_interleave
        orig_masked_scatter = torch.masked_scatter

        with _scoped_qwen3_vl_torch_ops():
            # Scoped overrides are active
            assert torch.cumsum != orig_cumsum
            assert torch.repeat_interleave != orig_repeat_interleave
            assert torch.masked_scatter != orig_masked_scatter

            # Nested scope
            with _scoped_qwen3_vl_torch_ops():
                pass

            assert torch.cumsum != orig_cumsum

        # Restored after context exit
        assert torch.cumsum == orig_cumsum
        assert torch.repeat_interleave == orig_repeat_interleave
        assert torch.masked_scatter == orig_masked_scatter

        # Test masked_scatter size mismatch error
        class MockTPUTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("tpu")

        inp = torch.zeros(4).as_subclass(MockTPUTensor)
        mask = torch.tensor([True, True, True, False])
        short_source = torch.tensor([1.0, 2.0])

        with pytest.raises(RuntimeError, match="Number of source elements"):
            _patched_masked_scatter(inp, mask, short_source)

    def test_qwen3_vl_vision_transformer_candidate_class_names(self):
        from vllm_torchtpu.models.vllm.qwen3_vl_patch import (
            _patch_qwen3_vl_vision_transformer,
        )

        class FakeQwen3VLVisionModel:
            def forward(self, hidden_states):
                # Verify that scoped ops are active inside vision encoder forward
                assert torch.cumsum != torch._orig_cumsum_ref
                return hidden_states

        orig_cumsum = torch.cumsum
        torch._orig_cumsum_ref = orig_cumsum
        try:
            fake_modeling = SimpleNamespace(Qwen3VLVisionModel=FakeQwen3VLVisionModel)
            _patch_qwen3_vl_vision_transformer(fake_modeling)

            # Idempotence: calling again is a no-op
            _patch_qwen3_vl_vision_transformer(fake_modeling)

            inst = FakeQwen3VLVisionModel()
            out = inst.forward(torch.tensor([1, 2, 3]))
            assert torch.equal(out, torch.tensor([1, 2, 3]))
            # Outside vision forward, torch ops are restored
            assert torch.cumsum == orig_cumsum
        finally:
            if hasattr(torch, "_orig_cumsum_ref"):
                delattr(torch, "_orig_cumsum_ref")
