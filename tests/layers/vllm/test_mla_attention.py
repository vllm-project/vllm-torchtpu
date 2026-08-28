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


def test_pallas_mla_attention_backend_fp8_ds_mla():
    """Verify DeepSeek-V4 packed MLA KV-cache format shape computation."""
    assert "fp8_ds_mla" in PallasMLAttentionBackend.supported_kv_cache_dtypes

    # Called directly with raw cache-dtype string "fp8_ds_mla".
    shape_str = PallasMLAttentionBackend.get_kv_cache_shape(
        num_blocks=10,
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp8_ds_mla",
    )
    assert shape_str == (10, 64, 4, 640)

    # Called with the already-resolved torch.uint8 dtype, matching how
    # get_kv_cache_page_size_bytes below delegates internally.
    shape_dtype = PallasMLAttentionBackend.get_kv_cache_shape(
        num_blocks=10,
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str=torch.uint8,
    )
    assert shape_dtype == shape_str

    page_size = PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp8_ds_mla",
    )
    # 640 packed bytes/token * 256 tokens/block.
    assert page_size == 640 * 256


def test_tpu_platform_mla_backend():
    TpuPlatform.pre_register_and_update()
    attn_selector_config = MagicMock()
    attn_selector_config.use_mla = True
    cls_name = TpuPlatform.get_attn_backend_cls(
        selected_backend=MagicMock(),
        attn_selector_config=attn_selector_config)
    assert cls_name == "vllm_torchtpu.layers.vllm.attention.PallasMLAttentionBackend"


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
            "vllm_torchtpu.layers.vllm.quantization.fp8.synchronize_tensors"
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
    ) as mock_mla_init:
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
            non_causal_multi_token_decode=True,
            allow_short_prefill_indexer_scoring_skip=True,
        )
        assert wrapper.hidden_size == 1024
        assert wrapper.num_heads == 16
        assert mock_mla_init.call_args.kwargs[
            "non_causal_multi_token_decode"] is True


def test_pallas_mla_backend_impl():
    impl = PallasMLAttentionBackendImpl(
        num_heads=16,
        head_size=576,
        scale=0.125,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="DECODER",
        kv_sharing_target_layer_name=None,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    assert impl.num_heads == 16
    assert impl.kv_lora_rank == 512

    # Test _get_kv_scales
    layer = MagicMock()
    layer._q_scale_float = None
    layer._k_scale_float = None
    layer._v_scale_float = None
    layer._q_scale = torch.tensor(1.5)
    layer._k_scale = torch.tensor(2.0)
    layer._v_scale = torch.tensor(2.5)
    q_scale, k_scale, v_scale = impl._get_kv_scales(layer)
    assert q_scale == 1.5
    assert k_scale == 2.0
    assert v_scale == 2.5

    # Test forward with empty kv_cache (probe check)
    layer.num_heads = 16
    layer.v_head_dim = 128
    q_nope = torch.ones((4, 16 * 128))
    q_pe = torch.ones((4, 16 * 64))
    empty_kv_cache = torch.empty((0, ))
    out = impl.forward(
        layer=layer,
        q=(q_nope, q_pe),
        kv_c_normed=torch.ones((4, 512)),
        k_pe=torch.ones((4, 64)),
        kv_cache=empty_kv_cache,
        attn_metadata=MagicMock(),
    )
    assert out.shape == (4, 16 * 128)


def _make_impl_and_layer(k_scale=2.0, num_tokens=4):
    """Real-tensor layer for exercising the forward body up to the op call.

    SimpleNamespace (not MagicMock) so `hasattr(layer, "W_UK_T_scale")` /
    `"W_UV_scale"` are genuinely False; the ops are mocks that capture the
    tensors forward hands them.
    """
    from types import SimpleNamespace

    impl = PallasMLAttentionBackendImpl(
        num_heads=16,
        head_size=576,
        scale=0.125,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
        logits_soft_cap=None,
        attn_type="DECODER",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
    )
    torch.manual_seed(0)
    op_out = torch.randn(num_tokens, 16, 512)
    layer = SimpleNamespace(
        num_heads=16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
        _q_scale_float=None,
        _k_scale_float=k_scale,
        _v_scale_float=None,
        W_UK_T=torch.randn(16, 128, 512) * 0.05,
        W_UV=torch.randn(16, 512, 128) * 0.05,
        sparse_mla_op=MagicMock(return_value=op_out),
        mla_op=MagicMock(return_value=op_out),
    )
    inputs = dict(
        q=(torch.randn(num_tokens, 16, 128), torch.randn(num_tokens, 16, 64)),
        kv_c_normed=torch.randn(num_tokens, 512) * 0.5,
        k_pe=torch.randn(num_tokens, 64) * 0.5,
        kv_cache=torch.zeros((2, 8, 4, 640), dtype=torch.float8_e4m3fn),
        attn_metadata=MagicMock(),
    )
    return impl, layer, inputs, op_out


def test_mla_forward_sparse_dispatch_and_quantization():
    """topk_indices present: the sparse op is called with absorbed queries and
    fp8 latents quantized with the layer's k_scale."""
    k_scale = 2.0
    impl, layer, inputs, op_out = _make_impl_and_layer(k_scale)
    topk_indices = torch.zeros((4, 8), dtype=torch.int32)
    # Sparse layers hold a native (nope, rope) split cache.
    inputs["kv_cache"] = tuple(
        torch.zeros(spec.shape, dtype=torch.uint8)
        for spec in PallasMLAttentionBackend.get_sparse_kv_cache_specs(
            2, 8, 576, torch.float8_e4m3fn))

    out = impl.forward(layer=layer, **inputs, topk_indices=topk_indices)

    layer.sparse_mla_op.assert_called_once()
    layer.mla_op.assert_not_called()
    (kv_cache, ql_nope, q_pe, kv_c, k_pe, topk, seq_lens, block_tables,
     query_start_loc,
     request_distribution) = layer.sparse_mla_op.call_args.args

    assert kv_cache is inputs["kv_cache"]
    assert topk is topk_indices
    assert seq_lens is inputs["attn_metadata"].seq_lens
    assert block_tables is inputs["attn_metadata"].block_tables
    assert query_start_loc is inputs["attn_metadata"].query_start_loc
    assert request_distribution is inputs["attn_metadata"].request_distribution

    # Absorbed query: ql_nope = q_nope @ W_UK_T (per head), q_pe untouched.
    q_nope_in, q_pe_in = inputs["q"]
    expected_ql = torch.bmm(q_nope_in.transpose(0, 1),
                            layer.W_UK_T).transpose(0, 1)
    torch.testing.assert_close(ql_nope, expected_ql)
    torch.testing.assert_close(q_pe, q_pe_in)

    # Latents quantized to fp8 with the layer's k_scale (dequant recovers
    # the originals up to fp8 rounding).
    assert kv_c.dtype == torch.float8_e4m3fn
    assert k_pe.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(kv_c.float() * k_scale,
                               inputs["kv_c_normed"],
                               rtol=0.1,
                               atol=0.05)
    torch.testing.assert_close(k_pe.float() * k_scale,
                               inputs["k_pe"],
                               rtol=0.1,
                               atol=0.05)

    # Output = op result through W_UV, flattened to (T, heads * v_head_dim).
    expected_out = torch.bmm(op_out.transpose(0, 1),
                             layer.W_UV).transpose(0, 1).reshape(4, -1)
    torch.testing.assert_close(out, expected_out)


def test_mla_forward_dense_dispatch_without_topk():
    """No topk_indices: the dense op is called, the sparse op is not."""
    k_scale = 2.0
    impl, layer, inputs, op_out = _make_impl_and_layer(k_scale)

    out = impl.forward(layer=layer, **inputs)

    layer.mla_op.assert_called_once()
    layer.sparse_mla_op.assert_not_called()
    (kv_cache, ql_nope, q_pe, kv_c, k_pe, seq_lens, block_tables,
     query_start_loc, request_distribution) = layer.mla_op.call_args.args

    assert kv_cache is inputs["kv_cache"]
    assert seq_lens is inputs["attn_metadata"].seq_lens
    assert kv_c.dtype == torch.float8_e4m3fn
    assert k_pe.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(kv_c.float() * k_scale,
                               inputs["kv_c_normed"],
                               rtol=0.1,
                               atol=0.05)
    expected_out = torch.bmm(op_out.transpose(0, 1),
                             layer.W_UV).transpose(0, 1).reshape(4, -1)
    torch.testing.assert_close(out, expected_out)
