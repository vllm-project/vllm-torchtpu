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

from unittest.mock import MagicMock, patch

import torch

from vllm_torchtpu import _patch_mla_prefill_backend
from vllm_torchtpu.layers.vllm.attention import (PallasMLAttentionBackend,
                                                 PallasMLAttentionBackendImpl)
from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import (
    VllmMLAAttention, VllmMultiHeadLatentAttentionWrapper)
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


def test_pallas_mla_attention_backend():
    assert PallasMLAttentionBackend.get_name() == "FLASH_ATTN_MLA"
    assert PallasMLAttentionBackend.is_mla() is True
    assert PallasMLAttentionBackend.get_impl_cls(
    ) == PallasMLAttentionBackendImpl

    # Test get_kv_cache_shape with auto
    shape_auto = PallasMLAttentionBackend.get_kv_cache_shape(
        num_blocks=10,
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="auto",
    )
    assert shape_auto == (10, 16, 1, 640)

    # Test get_kv_cache_shape with fp8
    shape_fp8 = PallasMLAttentionBackend.get_kv_cache_shape(
        num_blocks=10,
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="fp8",
    )
    assert shape_fp8 == (10, 4, 4, 640)

    # Test page size bytes
    page_size = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        block_size=16, num_kv_heads=1, head_size=576, cache_dtype_str="fp8")
    assert page_size == 10240


def test_tpu_platform_mla_backend():
    attn_selector_config = MagicMock()
    attn_selector_config.use_mla = True
    cls_name = TpuPlatform.get_attn_backend_cls(
        selected_backend=MagicMock(),
        attn_selector_config=attn_selector_config)
    assert cls_name == "vllm_torchtpu.layers.vllm.attention.PallasMLAttentionBackend"


def test_patch_mla_prefill_backend():
    _patch_mla_prefill_backend()
    from vllm.v1.attention.backends.mla.prefill import selector

    assert getattr(selector, "_tpu_mla_prefill_patch", False) is True
    dummy_cls = selector.get_mla_prefill_backend(None)
    dummy = dummy_cls()
    assert dummy.forward() is None


def test_vllm_fp8_linear_method_tpu():
    quant_config = MagicMock()
    quant_config.is_scale_e8m0 = False
    method = VllmFp8LinearMethodTPU(quant_config)
    assert method.use_deep_gemm is False

    layer = MagicMock()
    layer.weight = torch.ones(128, 128)
    with patch(
            "vllm_torchtpu.layers.vllm.quantization.fp8.replace_parameter"
    ), patch(
            "vllm_torchtpu.layers.vllm.quantization.fp8.sync.synchronize"
    ), patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0
    ), patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1):
        method.create_weights(
            layer,
            input_size=128,
            output_size=128,
            params_dtype=torch.float32,
            weight_loader=MagicMock(),
            input_size_per_partition=128,
            output_partition_sizes=[128],
        )
    assert hasattr(layer, "weight_block_size")


def test_vllm_mla_attention_init():
    kv_b_proj = MagicMock()

    def mock_mla_init(self, *args, **kwargs):
        self.kv_cache_dtype = "fp8"
        self.layer_name = "model.layers.0.attn"

    with patch(
            "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
            mock_mla_init,
    ):
        attn = VllmMLAAttention(
            num_heads=16,
            scale=1.0,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            kv_b_proj=kv_b_proj,
        )
        assert attn.kv_sharing_target_layer_name is None
        assert attn.sliding_window is None
        assert attn.kv_cache_quantized_dtype is not None


def test_vllm_multi_head_latent_attention_wrapper():
    mla_modules = MagicMock()
    mla_modules.fused_qkv_a_proj = MagicMock()
    mla_modules.kv_a_proj_with_mqa = MagicMock()
    mla_modules.q_a_layernorm = MagicMock()
    mla_modules.q_b_proj = MagicMock()
    mla_modules.q_proj = MagicMock()
    mla_modules.kv_a_layernorm = MagicMock()
    mla_modules.kv_b_proj = MagicMock()
    mla_modules.rotary_emb = MagicMock()
    mla_modules.o_proj = MagicMock()
    mla_modules.indexer = None
    mla_modules.indexer_rotary_emb = None
    mla_modules.is_sparse = False

    with patch(
            "vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op.VllmMLAAttention.__init__",
            return_value=None,
    ):
        wrapper = VllmMultiHeadLatentAttentionWrapper(
            hidden_size=1024,
            num_heads=16,
            scale=1.0,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            mla_modules=mla_modules,
        )
        assert wrapper.hidden_size == 1024
        assert wrapper.num_heads == 16
