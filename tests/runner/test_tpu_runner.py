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

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        MambaSpec)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilder,
    AttentionMetadataBuilderContext)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class DummyMamba(MambaBase):

    def __init__(self):
        super().__init__()

    def get_kv_cache_spec(self, vllm_config):
        return MambaSpec(
            block_size=16,
            shapes=[(4, 128), (8, 64, 32)],
            dtypes=[torch.bfloat16, torch.float32],
            page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
        )

    def get_state_dtype(self):
        return (torch.bfloat16, torch.float32)

    def get_state_shape(self):
        return ((4, 128), (8, 64, 32))

    @property
    def mamba_type(self):
        return "dummy"


class TestTPURunner:

    def setup_method(self):
        self.mock_device = torch.device('cpu')

        model_config = ModelConfig(
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype=torch.bfloat16,
        )
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_model_len=1024,
            is_encoder_decoder=False,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            observability_config={},
            additional_config={},
        )

        self.runner = MagicMock(spec=TPUModelRunner)
        self.runner.device = self.mock_device
        self.runner.block_size = 16
        self.runner.kv_cache_dtype = torch.bfloat16
        self.runner.use_spmd = False
        self.runner.vllm_config = vllm_config
        self.runner.cache_config = vllm_config.cache_config
        self.runner.scheduler_config = vllm_config.scheduler_config
        self.runner._hybrid_uniform_page_size_bytes = None
        self.runner.max_num_reqs = 16
        self.runner.max_model_len = 1024
        self.runner.max_num_tokens = 2048
        self.runner.pin_memory = False
        self.runner.model_config = vllm_config.model_config
        self.runner.shared_kv_cache_layers = {}
        self.runner.runner_only_attn_layers = set()
        self.runner.enforce_eager = False
        self.runner.speculative_config = None

        # Bind the actual methods to our mock
        self.runner._update_mamba_page_size_padded = TPUModelRunner._update_mamba_page_size_padded.__get__(
            self.runner)
        self.runner._maybe_set_num_blocks_override = TPUModelRunner._maybe_set_num_blocks_override.__get__(
            self.runner)
        self.runner._maybe_set_compact_mamba_num_blocks_override = (
            TPUModelRunner._maybe_set_compact_mamba_num_blocks_override.
            __get__(self.runner))
        # Compact-mamba state starts unset (matches real __init__).
        self.runner._mamba_num_blocks = None
        self.runner._uniform_mamba_layout = (vllm_config.kv_transfer_config
                                             is not None)
        self.runner._unified_block_pool = False
        self.runner.kv_cache_raw_tensors = []
        self.runner.get_kv_cache_spec = TPUModelRunner.get_kv_cache_spec.__get__(
            self.runner)
        self.runner.initialize_kv_cache = TPUModelRunner.initialize_kv_cache.__get__(
            self.runner)
        self.runner._validate_shared_kv_cache_layout = (
            TPUModelRunner._validate_shared_kv_cache_layout)
        self.runner._maybe_add_kv_sharing_layers_to_kv_cache_groups = (
            TPUModelRunner._maybe_add_kv_sharing_layers_to_kv_cache_groups.
            __get__(self.runner))
        self.runner._add_shared_kv_cache_aliases = (
            TPUModelRunner._add_shared_kv_cache_aliases.__get__(self.runner))

    @patch('vllm_torchtpu.envs.TPU_KV_CACHE_HEADROOM_MIB', 0)
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_update_mamba_page_size_padded(self, mock_mem_info,
                                           mock_get_page_size):
        layers = {}
        for i in range(1):
            mock_attn = MagicMock(spec=Attention)
            mock_attn.num_kv_heads = 2
            mock_attn.head_size = 128
            layers[f'attn_{i}'] = mock_attn

        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        self.runner._update_mamba_page_size_padded(layers)

        # Mamba unpadded size:
        # (4 * 128 * 2) + (8 * 64 * 32 * 4) = 1024 + 65536 = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 4096 + 199680 = 203776
        assert self.runner._hybrid_uniform_page_size_bytes == 203776
        assert self.runner.cache_config.mamba_page_size_padded == 203776

        # Compact-mamba sizing runs first and wins (it can read HBM here).
        # It caps mamba at max_num_reqs + 1 = 17 recurrent slots and gives the
        # rest of the budget to the attention pool.
        #   1 attn + 3 mamba layers -> group_size=min=1 (3 < 1*1.5 is False),
        #   num_attn_groups=1, num_mamba_groups=3.
        #   avail = 10GB * 0.9 = 9,663,676,416 (headroom pinned to 0)
        #   avail_per_tensor = avail // 1 = 9,663,676,416
        #   mamba_per_tensor = 3 * 17 * 66560 = 3,394,560
        #   attn_num_blocks =
        #       (9,663,676,416 - 3,394,560) // (1 * 4096) = 2,358,467
        mock_mem_info.assert_called_once()
        assert self.runner._mamba_num_blocks == 17
        assert self.runner.cache_config.num_gpu_blocks_override == 2358467

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.runner.tpu_runner.bind_kv_cache')
    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_hybrid_duplication(self, mock_input_batch,
                                                    mock_has_kv_transfer,
                                                    mock_bind_kv_cache,
                                                    mock_get_page_size,
                                                    mock_get_shape):
        num_blocks = 100
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=uniform_size)
        mamba_spec = MambaSpec(block_size=16,
                               shapes=[(4, 128), (8, 64, 32)],
                               dtypes=[torch.bfloat16, torch.float32],
                               page_size_padded=uniform_size)

        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"],
                             kv_cache_spec=mamba_spec),
        ]

        kv_cache_tensors = [
            KVCacheTensor(size=tensor_size, shared_by=layer_names)
        ]
        kv_cache_config = KVCacheConfig(num_blocks=num_blocks,
                                        kv_cache_tensors=kv_cache_tensors,
                                        kv_cache_groups=kv_cache_groups)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)

        # One block table per kv_cache_group (the per-group dtype check
        # introduced with multi-group support iterates over every group).
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.return_value.block_table = mock_block_tables

        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        # Compact-mamba sizing not exercised here (initialize_kv_cache is
        # called directly with a fixed config): mamba shares the uniform
        # num_blocks. _mamba_num_blocks=None selects that fallback path.
        self.runner._mamba_num_blocks = None
        self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind_kv_cache.assert_called_once()
        created_caches = mock_bind_kv_cache.call_args[0][0]

        assert len(created_caches) == 4

        # Mamba shares the uniform num_blocks (compact sizing not applied).
        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert isinstance(created_caches[name], tuple)
            assert len(created_caches[name]) == 2
            assert created_caches[name][0].shape == (num_blocks, 4, 128)
            assert created_caches[name][1].shape == (num_blocks, 8, 64, 32)

        # Slot pool initialized to the (uniform) mamba block count.
        self.runner._init_mamba_slot_pool.assert_called_once_with(num_blocks)

        assert isinstance(created_caches["attn.0"], torch.Tensor)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.runner.tpu_runner.bind_kv_cache')
    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_compact_mamba(self, mock_input_batch,
                                               mock_has_kv_transfer,
                                               mock_bind_kv_cache,
                                               mock_get_page_size,
                                               mock_get_shape):
        """Compact-mamba: when _mamba_num_blocks is set, mamba layers allocate
        exactly that many recurrent slots while attention keeps num_blocks, and
        the slot pool is initialized to _mamba_num_blocks."""
        num_blocks = 100
        mamba_num_blocks = 17  # max_num_reqs(16) + 1
        uniform_size = 203776
        tensor_size = uniform_size * num_blocks

        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=uniform_size)
        mamba_spec = MambaSpec(block_size=16,
                               shapes=[(4, 128), (8, 64, 32)],
                               dtypes=[torch.bfloat16, torch.float32],
                               page_size_padded=uniform_size)
        layer_names = ["attn.0", "mamba.0", "mamba.1", "mamba.2"]
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.1"],
                             kv_cache_spec=mamba_spec),
            KVCacheGroupSpec(layer_names=["mamba.2"],
                             kv_cache_spec=mamba_spec),
        ]
        kv_cache_config = KVCacheConfig(num_blocks=num_blocks,
                                        kv_cache_tensors=[
                                            KVCacheTensor(
                                                size=tensor_size,
                                                shared_by=layer_names)
                                        ],
                                        kv_cache_groups=kv_cache_groups)

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 1
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.return_value.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch.return_value
        self.runner.kv_caches = []
        self.runner._mamba_num_blocks = mamba_num_blocks
        self.runner.initialize_kv_cache(kv_cache_config)

        created_caches = mock_bind_kv_cache.call_args[0][0]
        # Mamba: only mamba_num_blocks slots; attention: full num_blocks.
        for name in ["mamba.0", "mamba.1", "mamba.2"]:
            assert created_caches[name][0].shape == (mamba_num_blocks, 4, 128)
            assert created_caches[name][1].shape == (mamba_num_blocks, 8, 64,
                                                     32)
        assert created_caches["attn.0"].shape == (100, 16, 2, 1, 128)
        # Slot pool sized to the compact mamba block count.
        self.runner._init_mamba_slot_pool.assert_called_once_with(
            mamba_num_blocks)

    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    @patch('vllm_torchtpu.utils.torch.accelerator.get_memory_info',
           return_value=(10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))
    def test_get_kv_cache_spec_hybrid_padded_size(self, mock_mem_info,
                                                  mock_get_page_size,
                                                  mock_get_layers):
        layers = {}

        mock_attn = MagicMock(spec=Attention)
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None
        layers['attn_0'] = mock_attn

        for i in range(3):
            layers[f'mamba_{i}'] = DummyMamba()

        mock_get_layers.return_value = layers

        # Call the method
        kv_cache_specs = self.runner.get_kv_cache_spec()

        # Mamba unpadded size: (4 * 128 * 2) + (8 * 64 * 32 * 4) = 66560
        # uniform size = (1 * 4096) + (3 * 66560) = 203776
        expected_padded_size = 203776

        # Verify specs
        assert 'attn_0' in kv_cache_specs
        attn_spec = kv_cache_specs['attn_0']
        assert isinstance(attn_spec, FullAttentionSpec)
        assert attn_spec.page_size_padded == expected_padded_size

        for i in range(3):
            layer_name = f'mamba_{i}'
            assert layer_name in kv_cache_specs
            mamba_spec = kv_cache_specs[layer_name]
            assert isinstance(mamba_spec, MambaSpec)
            assert mamba_spec.page_size_padded == expected_padded_size

    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    def test_get_kv_cache_spec_pure_attention_no_cache_config_updates(
            self, mock_get_page_size, mock_get_layers):
        mock_attn = MagicMock(spec=Attention)
        mock_attn.attn_type = AttentionType.DECODER
        mock_attn.num_kv_heads = 2
        mock_attn.head_size = 128
        mock_attn.kv_sharing_target_layer_name = None
        mock_attn.sliding_window = None

        layers = {'layer.0': mock_attn}
        mock_get_layers.return_value = layers

        with patch.object(self.runner,
                          '_update_mamba_page_size_padded') as mock_update:
            self.runner.get_kv_cache_spec()
            mock_update.assert_not_called()

    @patch('vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config')
    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_page_size_bytes',
        return_value=4096)
    def test_get_kv_cache_spec_records_shared_layers(self, mock_get_page_size,
                                                     mock_get_layers):
        owner = MagicMock(spec=Attention)
        owner.attn_type = AttentionType.DECODER
        owner.num_kv_heads = 2
        owner.head_size = 128
        owner.kv_sharing_target_layer_name = None
        owner.sliding_window = None

        shared = MagicMock(spec=Attention)
        shared.attn_type = AttentionType.DECODER
        shared.num_kv_heads = 2
        shared.head_size = 128
        shared.kv_sharing_target_layer_name = "layer.0"
        shared.sliding_window = None

        mock_get_layers.return_value = {
            "layer.0": owner,
            "layer.1": shared,
        }

        kv_cache_specs = self.runner.get_kv_cache_spec()

        assert set(kv_cache_specs) == {"layer.0"}
        assert self.runner.shared_kv_cache_layers == {"layer.1": "layer.0"}

    def _make_shared_pair(self, attn_window, target_window):
        attn = MagicMock(spec=Attention)
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.sliding_window = attn_window
        attn.impl = None

        target = MagicMock(spec=Attention)
        target.attn_type = AttentionType.DECODER
        target.num_kv_heads = 2
        target.head_size = 128
        target.sliding_window = target_window
        target.impl = None
        return attn, target

    def test_validate_shared_layout_skips_sliding_window_when_hma_disabled(
            self):
        # HMA disabled: specs are unified to full attention, so a window
        # mismatch is harmless and must not raise.
        attn, target = self._make_shared_pair(attn_window=512,
                                              target_window=None)
        TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                        attn,
                                                        "layer.0",
                                                        target,
                                                        hma_enabled=False)

    def test_validate_shared_layout_enforces_sliding_window_when_hma_enabled(
            self):
        # HMA enabled: sliding-window layers keep a smaller window-sized cache,
        # so a window mismatch means the layers disagree on storage -> raise.
        attn, target = self._make_shared_pair(attn_window=512,
                                              target_window=None)
        with pytest.raises(ValueError, match="sliding_window"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=True)

    def test_validate_shared_layout_matching_window_ok_when_hma_enabled(self):
        # HMA enabled but windows match: no mismatch, no raise.
        attn, target = self._make_shared_pair(attn_window=512,
                                              target_window=512)
        TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                        attn,
                                                        "layer.0",
                                                        target,
                                                        hma_enabled=True)

    def _make_fp8_shared_pair(self,
                              attn_quant,
                              target_quant,
                              attn_scale=1.0,
                              target_scale=1.0):
        attn = MagicMock(spec=Attention)
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.sliding_window = None
        attn._k_scale_float = attn_scale
        attn._v_scale_float = attn_scale
        attn.impl = MagicMock(kv_cache_quantized_dtype=attn_quant)

        target = MagicMock(spec=Attention)
        target.attn_type = AttentionType.DECODER
        target.num_kv_heads = 2
        target.head_size = 128
        target.sliding_window = None
        target._k_scale_float = target_scale
        target._v_scale_float = target_scale
        target.impl = MagicMock(kv_cache_quantized_dtype=target_quant)
        return attn, target

    def test_validate_shared_layout_fp8_matching_ok(self):
        # Both sides fp8 with matching scales/dtype: no raise.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3", "fp8_e4m3")
        TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                        attn,
                                                        "layer.0",
                                                        target,
                                                        hma_enabled=False)

    def test_validate_shared_layout_fp8_mismatched_scales_raises(self):
        # Both fp8 but scales differ: dequantizing with the wrong scales
        # corrupts attention -> raise.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3",
                                                  "fp8_e4m3",
                                                  attn_scale=1.0,
                                                  target_scale=2.0)
        with pytest.raises(ValueError, match="k/v scales"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=False)

    def test_validate_shared_layout_fp8_target_only_raises(self):
        # Asymmetric: target is fp8, shared layer is not. The shared layer would
        # read the target's packed fp8 bytes as bf16 -> silent garbage. Must
        # raise instead of being skipped.
        attn, target = self._make_fp8_shared_pair(None, "fp8_e4m3")
        with pytest.raises(ValueError, match="kv cache"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=False)

    def test_validate_shared_layout_fp8_attn_only_raises(self):
        # Asymmetric the other way: shared layer is fp8, target is not.
        attn, target = self._make_fp8_shared_pair("fp8_e4m3", None)
        with pytest.raises(ValueError, match="kv cache"):
            TPUModelRunner._validate_shared_kv_cache_layout("layer.1",
                                                            attn,
                                                            "layer.0",
                                                            target,
                                                            hma_enabled=False)

    @patch(
        'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
        return_value=(100, 16, 2, 1, 128))
    @patch('vllm_torchtpu.runner.tpu_runner.bind_kv_cache')
    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_binds_shared_layer_to_target_cache(
            self, mock_input_batch, mock_has_kv_transfer, mock_bind_kv_cache,
            mock_get_shape):
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[
                KVCacheTensor(size=16384 * 100, shared_by=["layer.0"])
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"],
                                 kv_cache_spec=attn_spec)
            ],
        )

        self.runner.shared_kv_cache_layers = {"layer.1": "layer.0"}
        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 1
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.return_value.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch.return_value

        self.runner.initialize_kv_cache(kv_cache_config)

        created_caches = mock_bind_kv_cache.call_args[0][0]
        assert set(created_caches) == {"layer.0", "layer.1"}
        assert created_caches["layer.1"] is created_caches["layer.0"]
        group_names = self.runner.kv_cache_config.kv_cache_groups[
            0].layer_names
        assert group_names == ["layer.0", "layer.1"]
        assert self.runner.attn_groups[0][0].layer_names == [
            "layer.0", "layer.1"
        ]
        assert kv_cache_config.kv_cache_groups[0].layer_names == ["layer.0"]

    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_rejects_missing_shared_target(
            self, mock_input_batch, mock_has_kv_transfer):
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["layer.0"])
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"],
                                 kv_cache_spec=attn_spec)
            ],
        )
        self.runner.shared_kv_cache_layers = {"layer.1": "missing.layer"}

        with pytest.raises(ValueError, match="target layer is missing"):
            self.runner.initialize_kv_cache(kv_cache_config)

    @patch('vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
           return_value=False)
    @patch('vllm.v1.worker.gpu_input_batch.InputBatch')
    def test_initialize_kv_cache_rejects_duplicate_shared_allocation(
            self, mock_input_batch, mock_has_kv_transfer):
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["layer.0"]),
                KVCacheTensor(size=16384, shared_by=["layer.1"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer.0"],
                                 kv_cache_spec=attn_spec),
                KVCacheGroupSpec(layer_names=["layer.1"],
                                 kv_cache_spec=attn_spec),
            ],
        )
        self.runner.shared_kv_cache_layers = {"layer.1": "layer.0"}

        with pytest.raises(ValueError, match="independent KV cache"):
            self.runner.initialize_kv_cache(kv_cache_config)

    def test_mrope_positions_buffer_is_int32(self):
        """Run the real TPUModelRunner.__init__ (with parent's __init__ and
        a few heavy TPU-only helpers patched out) on the setup_method mock
        and verify the uses_mrope override block produces an int32 buffer
        — not just that _make_buffer respects an int32 dtype on its own."""
        self.runner.dtype = torch.bfloat16
        self.runner.parallel_config = self.runner.vllm_config.parallel_config
        self.runner.uses_mrope = True
        self.runner.supports_mm_inputs = False
        self.runner.vllm_config.compilation_config.compile_sizes = [16, 2048]
        self.runner._make_buffer = GPUModelRunner._make_buffer.__get__(
            self.runner)

        with patch.object(GPUModelRunner, '__init__', return_value=None), \
             patch('vllm_torchtpu.runner.tpu_runner._torch_tpu_wrapper',
                   side_effect=lambda: contextlib.nullcontext()), \
             patch('vllm_torchtpu.runner.tpu_runner._validate_libtpu_version'
                   ), \
             patch.object(TPUModelRunner,
                          '_create_mesh_for_parallelism',
                          return_value=MagicMock()):
            TPUModelRunner.__init__(self.runner, self.runner.vllm_config,
                                    self.mock_device)

        assert self.runner.mrope_positions.cpu.dtype == torch.int32
        assert self.runner.mrope_positions.cpu.shape == (
            3, self.runner.max_num_tokens + 1)
        assert self.runner.mrope_positions.np.dtype == np.int32

    def test_initialize_kv_cache_multi_group(self):
        """Multi-group initialize_kv_cache: verify the three new behaviors
        added with the builder migration: (1) attn_groups gets one shared
        TPU builder per kv_cache_group, (2) may_reinitialize_input_batch is
        called with per-group block_sizes, (3) empty_slot_mappings has one
        entry per group (consumed by upstream _build_attention_metadata)."""
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["attn.1"], kv_cache_spec=attn_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["attn.0"]),
                KVCacheTensor(size=16384, shared_by=["attn.1"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_block_tables = []
        for _ in kv_cache_groups:
            mock_bt = MagicMock()
            mock_bt.max_num_blocks_per_req = 4
            mock_bt.get_cpu_tensor.return_value = torch.zeros(
                (1, 1), dtype=torch.int32)
            mock_block_tables.append(mock_bt)
        mock_input_batch.block_table = mock_block_tables
        self.runner.input_batch = mock_input_batch

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(1, 16, 2, 1, 128)
        ), patch('vllm_torchtpu.runner.tpu_runner.bind_kv_cache'), patch(
                'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config)

        # (1) One shared TPU builder per group, in the parent's
        # list-of-list-of-AttentionGroup layout.
        assert len(self.runner.attn_groups) == len(kv_cache_groups)
        for gid, group_list in enumerate(self.runner.attn_groups):
            assert len(group_list) == 1
            grp = group_list[0]
            assert isinstance(grp, AttentionGroup)
            assert grp.backend is None
            assert grp.kv_cache_group_id == gid
            assert len(grp.metadata_builders) == 1
            assert isinstance(grp.metadata_builders[0],
                              AttentionMetadataBuilder)
            assert grp.metadata_builders[0].kv_cache_group_id == gid

        # (2) may_reinitialize_input_batch saw the per-group block_sizes
        # list (replacing the old explicit single-group InputBatch reinit).
        self.runner.may_reinitialize_input_batch.assert_called_once_with(
            kv_cache_config, [16, 16])

        # (3) empty_slot_mappings: one zero-element tensor per kv_cache_group.
        assert set(self.runner.empty_slot_mappings.keys()) == {0, 1}
        for tensor in self.runner.empty_slot_mappings.values():
            assert tensor.numel() == 0

    def test_initialize_kv_cache_composite_specs(self):
        """Verify composite kv_cache_specs unpacking when a group's spec wraps
        a dictionary of individual per-layer specs."""
        layer_spec = FullAttentionSpec(block_size=16,
                                       num_kv_heads=2,
                                       head_size=128,
                                       dtype=torch.bfloat16,
                                       page_size_padded=16384)
        composite_spec = MagicMock()
        composite_spec.kv_cache_specs = {"attn.0": layer_spec}

        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"],
                             kv_cache_spec=composite_spec),
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=16384, shared_by=["attn.0"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 4
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(1, 16, 2, 1, 128)), patch(
                    'vllm_torchtpu.runner.tpu_runner.bind_kv_cache'
                ) as mock_bind, patch(
                    'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                    return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config)

        mock_bind.assert_called_once()
        created_caches = mock_bind.call_args[0][0]
        assert "attn.0" in created_caches
        assert created_caches["attn.0"].shape == (1, 16, 2, 1, 128)

    def test_initialize_kv_cache_num_blocks_override_and_relaxed_divisibility(
            self):
        """Verify explicit num_blocks override on kv_cache_config and relaxed
        divisibility when num_blocks is None and tensor_size is padded."""
        attn_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=2,
                                      head_size=128,
                                      dtype=torch.bfloat16,
                                      page_size_padded=16384)
        kv_cache_groups = [
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
        ]

        # Case 1: num_blocks explicitly provided on kv_cache_config
        kv_cache_config_explicit = KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(size=16384 * 10 + 123, shared_by=["attn.0"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )

        self.runner.vllm_config.compilation_config.static_forward_context = {}
        self.runner.block_table_cpu = torch.zeros((1, 1), dtype=torch.int32)
        self.runner.kv_caches = []

        mock_input_batch = MagicMock()
        mock_bt = MagicMock()
        mock_bt.max_num_blocks_per_req = 4
        mock_bt.get_cpu_tensor.return_value = torch.zeros((1, 1),
                                                          dtype=torch.int32)
        mock_input_batch.block_table = [mock_bt]
        self.runner.input_batch = mock_input_batch

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(10, 16, 2, 1, 128)) as mock_get_shape, patch(
                    'vllm_torchtpu.runner.tpu_runner.bind_kv_cache'
                ), patch(
                    'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                    return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config_explicit)
            mock_get_shape.assert_called_once_with(10, attn_spec.block_size,
                                                   attn_spec.num_kv_heads,
                                                   attn_spec.head_size,
                                                   attn_spec.dtype)

        # Case 2: num_blocks is None, tensor_size has padding (not perfectly divisible)
        kv_cache_config_implicit = KVCacheConfig(
            num_blocks=None,
            kv_cache_tensors=[
                KVCacheTensor(size=16384 * 5 + 400, shared_by=["attn.0"]),
            ],
            kv_cache_groups=kv_cache_groups,
        )
        kv_cache_config_implicit.num_blocks = None

        with patch(
                'vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_kv_cache_shape',
                return_value=(5, 16, 2, 1, 128)) as mock_get_shape, patch(
                    'vllm_torchtpu.runner.tpu_runner.bind_kv_cache'
                ), patch(
                    'vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group',
                    return_value=False):
            self.runner.initialize_kv_cache(kv_cache_config_implicit)
            mock_get_shape.assert_called_once_with(5, attn_spec.block_size,
                                                   attn_spec.num_kv_heads,
                                                   attn_spec.head_size,
                                                   attn_spec.dtype)


class TestAttentionMetadataBuilder:
    """Direct tests for AttentionMetadataBuilder.build, exercising the
    runner-state path used in _prepare_inputs and the position_ids_override
    path used in _dummy_run."""

    def _make_runner_mock(self,
                          most_model_len=None,
                          num_groups=1,
                          max_num_blocks_per_req=4,
                          max_num_reqs=4):
        runner = MagicMock()
        runner.device = torch.device("cpu")
        runner.block_size = 16
        runner.max_num_reqs = max_num_reqs
        runner.most_model_len = most_model_len
        runner._unified_block_pool = False
        runner.position_ids = torch.full((8, ), 42, dtype=torch.int32)

        block_tables = []
        for gid in range(num_groups):
            bt = MagicMock()
            bt.max_num_blocks_per_req = max_num_blocks_per_req
            bt.get_cpu_tensor.return_value = (
                torch.arange(max_num_reqs * max_num_blocks_per_req,
                             dtype=torch.int32).reshape(
                                 max_num_reqs, max_num_blocks_per_req) +
                gid * 100)
            block_tables.append(bt)
        runner.input_batch.block_table = block_tables
        return runner

    def _make_builder(self, runner, kv_cache_group_id=0, spec=None):
        if spec is None:
            spec = FullAttentionSpec(block_size=16,
                                     num_kv_heads=2,
                                     head_size=128,
                                     dtype=torch.bfloat16,
                                     page_size_padded=16384)
        return AttentionMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["attn.0"],
            vllm_config=MagicMock(),
            device=runner.device,
            runner=runner,
            kv_cache_group_id=kv_cache_group_id,
        )

    def _make_cm(self, num_reqs):
        """Minimal CommonAttentionMetadata stand-in — build() only reads
        num_reqs off it (everything else still comes from ctx because TPU
        bypasses parent's CpuGpuBuffers)."""
        cm = MagicMock()
        cm.num_reqs = num_reqs
        return cm

    def test_build_runner_state_path(self):
        """Normal _prepare_inputs path: copy from the right per-group block
        table at the right start_index, pad remaining rows, and reuse
        runner.position_ids. target_num_reqs is sourced from cm.num_reqs
        (the only field we currently read from common_attn_metadata)."""
        runner = self._make_runner_mock(num_groups=2)
        # Build for group 1 to cover the per-group block_table lookup too.
        builder = self._make_builder(runner, kv_cache_group_id=1)

        target_num_reqs, num_reqs, start_index = 4, 2, 1
        seq_lens = torch.tensor([10, 12, 0, 0], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
        request_distribution = torch.tensor([2, 2, 2], dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=start_index,
            use_max_model_len=True,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        meta = builder.build(
            common_prefix_len=0,
            common_attn_metadata=self._make_cm(target_num_reqs))

        assert isinstance(meta, AttentionMetadata)
        assert meta.input_positions is runner.position_ids
        assert meta.seq_lens is seq_lens
        assert meta.query_start_loc is query_start_loc
        assert meta.request_distribution is request_distribution

        # Group 1 (not 0) should have been read; group 0 untouched.
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        runner.input_batch.block_table[1].get_cpu_tensor.assert_called_once()

        # Flattened (target_num_reqs * max_num_blocks_per_req,); first
        # num_reqs rows from the source slice, rest zero-padded.
        max_num_blocks = runner.input_batch.block_table[
            1].max_num_blocks_per_req
        block_tables_2d = meta.block_tables.reshape(target_num_reqs,
                                                    max_num_blocks)
        src = runner.input_batch.block_table[1].get_cpu_tensor.return_value
        assert torch.equal(block_tables_2d[:num_reqs],
                           src[start_index:start_index + num_reqs])
        assert torch.equal(
            block_tables_2d[num_reqs:],
            torch.zeros((target_num_reqs - num_reqs, max_num_blocks),
                        dtype=torch.int32))

    def test_build_position_ids_override(self):
        """_dummy_run path: position_ids_override is forwarded as-is and the
        block-table copy is skipped (no read from input_batch)."""
        runner = self._make_runner_mock()
        builder = self._make_builder(runner)

        override = torch.zeros((3, 8), dtype=torch.int32)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=4,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones((4, ), dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
            request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
            position_ids_override=override,
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert meta.input_positions is override
        runner.input_batch.block_table[0].get_cpu_tensor.assert_not_called()
        assert torch.equal(meta.block_tables,
                           torch.zeros((4 * 4, ), dtype=torch.int32))

    def test_build_most_model_len_shrinks_block_table(self):
        """When use_max_model_len is False, target_num_blocks =
        cdiv(most_model_len, block_size) — smaller than the per-group
        max_num_blocks_per_req, so the H2D copy is shorter."""
        runner = self._make_runner_mock(most_model_len=32,
                                        max_num_blocks_per_req=8)
        builder = self._make_builder(runner)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=False,
            seq_lens=torch.tensor([8, 16, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        # cdiv(32, 16) = 2; flattened length = target_num_reqs * 2 = 8.
        assert meta.block_tables.shape == (4 * 2, )

    def test_unified_mamba_state_indices_derive_from_block_table(self):
        runner = self._make_runner_mock(max_num_blocks_per_req=4)
        runner._unified_block_pool = True
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=[(2, 8)],
            dtypes=[torch.bfloat16],
            page_size_padded=256,
        )
        builder = self._make_builder(runner, spec=mamba_spec)

        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=2,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.tensor([1, 33, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
        )

        meta = builder.build(common_prefix_len=0,
                             common_attn_metadata=self._make_cm(4))

        assert torch.equal(meta.mamba_state_indices,
                           torch.tensor([0, 6, 8, 12], dtype=torch.int32))


class TestCompactMambaSlotPool:
    """Unit tests for the compact-mamba recurrent-slot allocator
    (_init_mamba_slot_pool + _build_mamba_state_indices). The slot index is
    correctness-critical: a wrong slot = silent recurrent-state corruption."""

    def _make_runner(self, max_num_reqs=4, mamba_num_blocks=5):
        runner = MagicMock(spec=TPUModelRunner)
        runner.device = torch.device("cpu")
        runner.max_num_reqs = max_num_reqs
        runner.mamba_state_indices_cpu = torch.zeros(max_num_reqs,
                                                     dtype=torch.int32)
        runner.input_batch = MagicMock()
        # Bind the real methods.
        runner._init_mamba_slot_pool = (
            TPUModelRunner._init_mamba_slot_pool.__get__(runner))
        runner._build_mamba_state_indices = (
            TPUModelRunner._build_mamba_state_indices.__get__(runner))
        return runner

    def _set_batch(self, runner, req_ids):
        """Set the persistent batch to the given ordered req_ids."""
        runner.input_batch.req_ids = list(req_ids)
        runner.input_batch.req_id_to_index = {
            r: i
            for i, r in enumerate(req_ids)
        }

    def test_unique_slots_and_null_tail(self):
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)  # usable slots 1..4, slot 0 = null
        self._set_batch(runner, ["a", "b"])

        idx = runner._build_mamba_state_indices(start_index=0,
                                                num_reqs=2,
                                                target_num_reqs=4).cpu()
        # Two active requests get distinct non-null slots.
        assert idx[0] != 0 and idx[1] != 0
        assert idx[0] != idx[1]
        # Padded tail points at the null slot (0).
        assert idx[2] == 0 and idx[3] == 0

    def test_slot_follows_req_id_through_condense(self):
        """A request keeps its slot even when its persistent-batch position
        changes (upstream condense moves it to a lower index)."""
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a", "b"])
        idx0 = runner._build_mamba_state_indices(0, 2, 4).cpu()
        slot_b = int(idx0[1])

        # "a" finishes; "b" condenses to position 0.
        self._set_batch(runner, ["b"])
        idx1 = runner._build_mamba_state_indices(0, 1, 4).cpu()
        assert int(idx1[0]) == slot_b  # same physical slot as before

    def test_freed_slot_is_reused(self):
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a", "b", "c", "d"])
        idx0 = runner._build_mamba_state_indices(0, 4, 4).cpu()
        used = {int(x) for x in idx0}
        assert used == {1, 2, 3, 4}  # all usable slots allocated, none null

        # All finish; a fresh request must reuse a freed slot (pool not
        # exhausted, no out-of-range index).
        self._set_batch(runner, ["e"])
        idx1 = runner._build_mamba_state_indices(0, 1, 4).cpu()
        assert 1 <= int(idx1[0]) <= 4

    def test_chunked_build_allocates_all(self):
        """Two chunks in one step: each chunk allocates its own requests; all
        get distinct slots."""
        runner = self._make_runner(max_num_reqs=4, mamba_num_blocks=5)
        runner._init_mamba_slot_pool(5)
        self._set_batch(runner, ["a", "b", "c", "d"])

        # clone: the result aliases a reused staging buffer (no copy on CPU).
        idx_chunk0 = runner._build_mamba_state_indices(0, 2, 4).clone()
        idx_chunk1 = runner._build_mamba_state_indices(2, 2, 4).clone()
        slots = {
            int(idx_chunk0[0]),
            int(idx_chunk0[1]),
            int(idx_chunk1[0]),
            int(idx_chunk1[1])
        }
        assert slots == {1, 2, 3, 4}
