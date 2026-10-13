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
from vllm.model_executor.layers.linear import DCPGroupColumnParallelLinear
from vllm.model_executor.layers.mla import MLAModules

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import attention as attention_adapter
from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla
from vllm_torchtpu.layers.adapter.attention import (
    PallasMLAttentionBackend,
    PallasMLAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmMLAAttention,
    VllmMultiHeadLatentAttentionWrapper,
    _dcp_indexer_a2a_for_bucket,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


def test_mla_attention_interface_mixed_q_split(monkeypatch):
    """Verify mla_attention passes MIXED_Q_SPLIT and MIXED_NUM_QUERIES_PER_BLOCK to
    mla_ragged_paged_attention."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.layers.core import attention_interface

    assert "MIXED_Q_SPLIT" in TpuPlatform.additional_env_vars
    assert "MIXED_NUM_QUERIES_PER_BLOCK" in TpuPlatform.additional_env_vars
    assert "MIXED_NUM_KV_PAGES_PER_BLOCK" in TpuPlatform.additional_env_vars

    num_tokens = 10
    num_heads = 64
    lkv_dim = 512
    r_dim = 64
    num_blocks = 16
    page_size_per_packing = 256
    kv_packing = 4

    q = jnp.ones((num_tokens, num_heads, lkv_dim), dtype=jnp.bfloat16)
    q_rope = jnp.ones((num_tokens, num_heads, r_dim), dtype=jnp.bfloat16)
    k = jnp.ones((num_tokens, lkv_dim), dtype=jnp.bfloat16)
    k_rope = jnp.ones((num_tokens, r_dim), dtype=jnp.bfloat16)
    cache = jnp.zeros(
        (num_blocks, page_size_per_packing, kv_packing, lkv_dim + 128),
        dtype=jnp.float8_e4m3fn,
    )

    md = SimpleNamespace(
        seq_lens=jnp.array([5, 5], dtype=jnp.int32),
        block_tables=jnp.zeros((18,), dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 2], dtype=jnp.int32),
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("model",))

    expected_kernel_out = jnp.full(
        (num_heads, num_tokens, lkv_dim), 0.5, dtype=jnp.bfloat16
    )
    expected_new_cache = cache

    mock_mla_kernel = MagicMock(return_value=(expected_kernel_out, expected_new_cache))
    monkeypatch.setattr(
        attention_interface, "mla_ragged_paged_attention", mock_mla_kernel
    )

    # 1. Unset env vars (None): falls back to mixed_tuned from get_tuned_params
    monkeypatch.setattr(envs, "MIXED_Q_SPLIT", None)
    monkeypatch.setattr(envs, "MIXED_NUM_QUERIES_PER_BLOCK", None)
    monkeypatch.setattr(envs, "MIXED_NUM_KV_PAGES_PER_BLOCK", None)
    new_cache, out = attention_interface.mla_attention(
        q_TNA=q,
        q_rope_TNH=q_rope,
        k_SA=k,
        k_rope_SH=k_rope,
        kv_cache=cache,
        md=md,
        mesh=mesh,
        num_attention_heads=num_heads,
        qk_nope_head_dim=lkv_dim,
        sm_scale=0.1,
    )
    mock_mla_kernel.assert_called_once()
    args, kwargs = mock_mla_kernel.call_args
    assert args[0].shape == (num_heads, num_tokens, lkv_dim)
    assert out.shape == (num_tokens, num_heads, lkv_dim)
    assert kwargs["num_kv_pages_per_block"][1:] == (1, 1)
    assert kwargs["num_queries_per_block"][1:] == (16, 16)
    assert kwargs["mixed_q_split"] == 1
    assert kwargs["sm_scale"] == 0.1

    # 2. Overridden env vars (MIXED_Q_SPLIT=16, MIXED_NUM_QUERIES_PER_BLOCK=64,
    #    MIXED_NUM_KV_PAGES_PER_BLOCK=4): overrides mixed_tuned
    mock_mla_kernel.reset_mock()
    monkeypatch.setattr(envs, "MIXED_Q_SPLIT", 16)
    monkeypatch.setattr(envs, "MIXED_NUM_QUERIES_PER_BLOCK", 64)
    monkeypatch.setattr(envs, "MIXED_NUM_KV_PAGES_PER_BLOCK", 4)
    attention_interface.mla_attention(
        q_TNA=q,
        q_rope_TNH=q_rope,
        k_SA=k,
        k_rope_SH=k_rope,
        kv_cache=cache,
        md=md,
        mesh=mesh,
        num_attention_heads=num_heads,
        qk_nope_head_dim=lkv_dim,
        sm_scale=0.1,
    )
    mock_mla_kernel.assert_called_once()
    _, kwargs = mock_mla_kernel.call_args
    assert kwargs["num_kv_pages_per_block"][1:] == (1, 4)
    assert kwargs["num_queries_per_block"][1:] == (16, 64)
    assert kwargs["mixed_q_split"] == 16


def test_pallas_mla_attention_backend():
    assert PallasMLAttentionBackend.get_name() == "FLASH_ATTN_MLA"
    assert PallasMLAttentionBackend.is_mla() is True
    assert PallasMLAttentionBackend.get_impl_cls() == PallasMLAttentionBackendImpl

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
        block_size=16, num_kv_heads=1, head_size=576, cache_dtype_str="fp8"
    )
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
        selected_backend=MagicMock(), attn_selector_config=attn_selector_config
    )
    assert cls_name == "vllm_torchtpu.layers.adapter.attention.PallasMLAttentionBackend"


def test_vllm_fp8_linear_method_tpu():
    quant_config = MagicMock()
    quant_config.is_scale_e8m0 = False
    method = VllmFp8LinearMethodTPU(quant_config)
    assert method.use_deep_gemm is False

    layer = MagicMock()
    layer.weight = torch.ones(128, 128)
    with (
        patch("vllm_torchtpu.layers.adapter.quantization.fp8.replace_parameter"),
        patch("vllm_torchtpu.layers.adapter.quantization.fp8.synchronize_tensors"),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
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
        "vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op.VllmMLAAttention.__init__",
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
        assert mock_mla_init.call_args.kwargs["non_causal_multi_token_decode"] is True


def test_mla_wrapper_rope_preserves_activation_dtype():
    wrapper = VllmMultiHeadLatentAttentionWrapper.__new__(
        VllmMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.kimi_preprocess = None
    wrapper.q_lora_rank = None
    wrapper.kv_lora_rank = 4
    wrapper.qk_nope_head_dim = 3
    wrapper.qk_rope_head_dim = 2
    wrapper.qk_head_dim = 5
    wrapper.num_heads = 2
    wrapper.v_head_dim = 2
    wrapper.kv_a_proj_with_mqa = lambda hidden: (
        torch.ones((hidden.shape[0], 6), dtype=hidden.dtype),
        None,
    )
    wrapper.q_proj = lambda hidden: (
        torch.ones((hidden.shape[0], 10), dtype=hidden.dtype),
        None,
    )
    wrapper.kv_a_layernorm = lambda value: value
    wrapper.rotary_emb = lambda _positions, q_pe, k_pe: (q_pe.float(), k_pe.float())
    wrapper.is_sparse = False
    wrapper.dcp_q_replicate = False
    wrapper.dcp_group_size = 1
    wrapper.g_proj = None
    wrapper.gate_is_fused = False
    wrapper.mla_attn = MagicMock(return_value=torch.ones((3, 4), dtype=torch.bfloat16))
    wrapper.o_proj = lambda value: (value, None)

    hidden_states = torch.ones((3, 8), dtype=torch.bfloat16)
    output = VllmMultiHeadLatentAttentionWrapper.forward(
        wrapper,
        positions=torch.tensor([0, 1, 2]),
        hidden_states=hidden_states,
    )

    (q_nope, q_pe), kv_c, k_pe = wrapper.mla_attn.call_args.args[:3]
    assert q_nope.dtype == hidden_states.dtype
    assert q_pe.dtype == hidden_states.dtype
    assert kv_c.dtype == hidden_states.dtype
    assert k_pe.dtype == hidden_states.dtype
    assert output.dtype == hidden_states.dtype


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
    empty_kv_cache = torch.empty((0,))
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
        torch.zeros(spec.shape, dtype=spec.torch_dtype)
        for spec in PallasMLAttentionBackend.get_sparse_kv_cache_specs(
            2, 8, 576, torch.float8_e4m3fn
        )
    )

    out = impl.forward(layer=layer, **inputs, topk_indices=topk_indices)

    layer.sparse_mla_op.assert_called_once()
    layer.mla_op.assert_not_called()
    (
        kv_cache,
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        topk,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    ) = layer.sparse_mla_op.call_args.args

    assert kv_cache is inputs["kv_cache"]
    assert topk is topk_indices
    assert seq_lens is inputs["attn_metadata"].seq_lens
    assert block_tables is inputs["attn_metadata"].block_tables
    assert query_start_loc is inputs["attn_metadata"].query_start_loc
    assert request_distribution is inputs["attn_metadata"].request_distribution

    # Absorbed query: ql_nope = q_nope @ W_UK_T (per head), q_pe untouched.
    q_nope_in, q_pe_in = inputs["q"]
    expected_ql = torch.bmm(q_nope_in.transpose(0, 1), layer.W_UK_T).transpose(0, 1)
    torch.testing.assert_close(ql_nope, expected_ql)
    torch.testing.assert_close(q_pe, q_pe_in)

    # Latents quantized to fp8 with the layer's k_scale (dequant recovers
    # the originals up to fp8 rounding).
    assert kv_c.dtype == torch.float8_e4m3fn
    assert k_pe.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        kv_c.float() * k_scale, inputs["kv_c_normed"], rtol=0.1, atol=0.05
    )
    torch.testing.assert_close(
        k_pe.float() * k_scale, inputs["k_pe"], rtol=0.1, atol=0.05
    )

    # Output = op result through W_UV, flattened to (T, heads * v_head_dim).
    expected_out = (
        torch.bmm(op_out.transpose(0, 1), layer.W_UV).transpose(0, 1).reshape(4, -1)
    )
    torch.testing.assert_close(out, expected_out)


def test_mla_forward_dense_dispatch_without_topk():
    """No topk_indices: the dense op is called, the sparse op is not."""
    k_scale = 2.0
    impl, layer, inputs, op_out = _make_impl_and_layer(k_scale)

    out = impl.forward(layer=layer, **inputs)

    layer.mla_op.assert_called_once()
    layer.sparse_mla_op.assert_not_called()
    (
        kv_cache,
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    ) = layer.mla_op.call_args.args

    assert kv_cache is inputs["kv_cache"]
    assert seq_lens is inputs["attn_metadata"].seq_lens
    assert kv_c.dtype == torch.float8_e4m3fn
    assert k_pe.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        kv_c.float() * k_scale, inputs["kv_c_normed"], rtol=0.1, atol=0.05
    )
    expected_out = (
        torch.bmm(op_out.transpose(0, 1), layer.W_UV).transpose(0, 1).reshape(4, -1)
    )
    torch.testing.assert_close(out, expected_out)


class _ConstantProjection(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, inputs):
        return self.values.expand(inputs.shape[0], -1), None


class _IdentityProjection(torch.nn.Module):
    def forward(self, inputs):
        return inputs, None


def _gate_modules(gate_mode, q_lora_rank):
    gate = [0.0, 2.0]
    modules = MLAModules(
        kv_a_layernorm=torch.nn.Identity(),
        kv_b_proj=torch.nn.Identity(),
        rotary_emb=None,
        o_proj=_IdentityProjection(),
        fused_qkv_a_proj=_ConstantProjection(
            [1.0] * 5 + (gate if gate_mode == "fused" else [])
        ),
        kv_a_proj_with_mqa=_ConstantProjection([1.0] * 3),
        q_a_layernorm=torch.nn.Identity(),
        q_b_proj=_ConstantProjection([1.0] * 3),
        q_proj=_ConstantProjection([1.0] * 3),
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
        g_proj=_ConstantProjection(gate) if gate_mode == "separate" else None,
    )
    if q_lora_rank is None:
        modules.fused_qkv_a_proj = None
    return modules


def _gate_wrapper(modules, q_lora_rank, **kwargs):
    class Attention(torch.nn.Module):
        def forward(self, q, kv_c_normed, k_pe, **kwargs):
            return torch.ones(kv_c_normed.shape[0], 2)

    with patch(
        "vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op.VllmTPUMLAAttention",
        return_value=Attention(),
    ):
        return VllmMultiHeadLatentAttentionWrapper(
            hidden_size=4,
            num_heads=1,
            scale=1.0,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=2,
            mla_modules=modules,
            **kwargs,
        )


def test_mla_wrapper_forwards_the_shared_topk_buffer():
    """A layer that reuses an earlier layer's indices has no indexer, so the
    shared buffer is what states its top-k width to masked-dense admission."""
    modules = _gate_modules("none", 2)
    modules.topk_indices_buffer = torch.full((4, 2048), -1, dtype=torch.int32)
    with patch(
        "vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op.VllmTPUMLAAttention"
    ) as attention:
        VllmMultiHeadLatentAttentionWrapper(
            hidden_size=4,
            num_heads=1,
            scale=1.0,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
            q_lora_rank=2,
            kv_lora_rank=2,
            mla_modules=modules,
        )
    kwargs = attention.call_args.kwargs
    assert kwargs["topk_indices_buffer"] is modules.topk_indices_buffer


@pytest.mark.parametrize("allowed", [None, False, True])
def test_sparse_mla_op_reads_admission_when_traced(allowed):
    """The op is built before the KV cache is allocated, so it reads the
    layer's admission when traced; an unadmitted layer passes False."""
    impl, layer, _, _ = _make_impl_and_layer()
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.scale = 0.125
    layer.kv_cache_quantized_dtype = None
    captured = {}

    def jax_op(name, fn, **kwargs):
        captured["core"] = fn
        return MagicMock()

    with (
        patch.object(attention_adapter.pallas, "jax_op", side_effect=jax_op),
        patch.object(
            attention_adapter,
            "get_vllm_model_wrapper_context",
            return_value=SimpleNamespace(mesh="mesh"),
        ),
        patch(
            "vllm_torchtpu.layers.core.attention_interface.sparse_mla_attention"
        ) as core,
    ):
        impl._build_sparse_mla_op(layer)
        layer.mla_kv_spec = ("nope_spec", "rope_spec")
        if allowed is not None:
            layer.allow_masked_dense = allowed
        captured["core"](*range(11))
    assert core.call_args.kwargs["allow_masked_dense"] is bool(allowed)


@pytest.mark.parametrize(
    "gate_mode,q_lora_rank",
    [("none", None), ("none", 2), ("separate", None), ("separate", 2), ("fused", 2)],
)
def test_mla_output_gate_numerics(gate_mode, q_lora_rank):
    wrapper = _gate_wrapper(
        _gate_modules(gate_mode, q_lora_rank),
        q_lora_rank,
        gate_is_fused=gate_mode == "fused",
    )
    result = wrapper(torch.arange(3), torch.ones(3, 4))
    expected = (
        torch.ones(2) if gate_mode == "none" else torch.tensor([0.0, 2.0]).sigmoid()
    )
    torch.testing.assert_close(result, expected.expand(3, -1))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("fused_qkv_a_proj", None, "requires fused_qkv_a_proj"),
        ("g_proj", torch.nn.Identity(), "cannot also use g_proj"),
    ],
)
def test_mla_rejects_invalid_fused_gate_modules(field, value, message):
    modules = _gate_modules("fused", 2)
    setattr(modules, field, value)
    with pytest.raises(AssertionError, match=message):
        _gate_wrapper(modules, 2, gate_is_fused=True)


def test_mla_fused_gate_requires_q_lora_rank():
    with pytest.raises(AssertionError, match="requires q_lora_rank"):
        _gate_wrapper(_gate_modules("fused", 2), None, gate_is_fused=True)


def test_mla_requires_declared_g_proj():
    modules = SimpleNamespace(**vars(_gate_modules("none", 2)))
    del modules.g_proj
    with pytest.raises(AttributeError, match="g_proj"):
        _gate_wrapper(modules, 2)


@pytest.mark.parametrize("layout", ["flipped", "unflipped", "packed"])
def test_mla_weight_layout_is_restored_on_error(layout):
    attention = VllmMLAAttention.__new__(VllmMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.kv_b_proj = torch.nn.Module()
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    parameter_name = "weight_packed" if layout == "packed" else "weight"
    attention.kv_b_proj.register_parameter(
        parameter_name, torch.nn.Parameter(weight.clone())
    )
    if layout == "flipped":
        setattr(attention.kv_b_proj, WEIGHT_FLIPPED_ATTR, True)

    def upstream_process(act_dtype):
        observed = getattr(attention.kv_b_proj, parameter_name)
        expected = weight.T if layout == "flipped" else weight
        torch.testing.assert_close(observed, expected)
        raise RuntimeError("upstream weight processing failed")

    with (
        patch(
            "vllm.model_executor.layers.attention.mla_attention.MLAAttention.process_weights_after_loading",
            side_effect=upstream_process,
        ),
        pytest.raises(RuntimeError, match="upstream weight processing failed"),
    ):
        attention.process_weights_after_loading(torch.float32)
    torch.testing.assert_close(getattr(attention.kv_b_proj, parameter_name), weight)


def test_mla_flipped_projection_requires_weight():
    attention = VllmMLAAttention.__new__(VllmMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.kv_b_proj = torch.nn.Module()
    setattr(attention.kv_b_proj, WEIGHT_FLIPPED_ATTR, True)
    with pytest.raises(AttributeError, match="weight"):
        attention.process_weights_after_loading(torch.float32)


@pytest.mark.parametrize(
    "selected,expected", [("FLASH_ATTN", "FLASH_ATTN"), ("FLASHINFER", "FLASH_ATTN")]
)
def test_tpu_platform_keeps_the_selected_backend_without_mla(selected, expected):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    TpuPlatform.pre_register_and_update()
    attn_selector_config = MagicMock()
    attn_selector_config.use_mla = False
    cls_name = TpuPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum[selected],
        attn_selector_config=attn_selector_config,
    )
    assert cls_name == AttentionBackendEnum[expected].get_path()
    assert cls_name != AttentionBackendEnum.FLASH_ATTN_MLA.get_path()


class _FakeQProj:
    """A `DCPGroupColumnParallelLinear` stand-in carrying upstream's own
    `_local_view`, so the slicing under test is the real one."""

    qrep_active = True
    _local_view = DCPGroupColumnParallelLinear._local_view

    def __init__(self, group_size: int, rank_in_group: int, out=None):
        self.group_size = group_size
        self.rank_in_group = rank_in_group
        self._out = out

    def __call__(self, _hidden):
        return self._out, None


class _FakeHeadGatherGroup:
    """Serves `all_gather(dim=1)` from a known per-rank stack."""

    def __init__(self, per_rank, rank):
        self.world_size = len(per_rank)
        self.rank_in_group = rank
        self._per_rank = per_rank

    def all_gather(self, tensor, dim=0):
        assert dim == 1
        torch.testing.assert_close(tensor, self._per_rank[self.rank_in_group])
        return torch.cat(self._per_rank, dim=1)


def test_qrep_local_view_matches_the_head_all_gather_order(monkeypatch):
    """qrep replaces the head all-gather with a local slice, so the two must
    agree on where in the group's head axis a rank's own block sits. If they
    disagree the heads are silently permuted and nothing raises."""
    group_size, heads, dim = 4, 3, 5
    torch.manual_seed(0)
    per_rank = [torch.randn(2, heads, dim) for _ in range(group_size)]
    grouped = torch.cat(per_rank, dim=1)

    for rank in range(group_size):
        group = _FakeHeadGatherGroup(per_rank, rank)
        monkeypatch.setattr(cp_mla, "get_dcp_group", lambda g=group: g)
        torch.testing.assert_close(cp_mla.all_gather_heads(per_rank[rank]), grouped)
        nope_dcp, pe_dcp = cp_mla.all_gather_q_nope_pe_heads(
            per_rank[rank][..., :3], per_rank[rank][..., 3:], 3
        )
        torch.testing.assert_close(nope_dcp, grouped[..., :3])
        torch.testing.assert_close(pe_dcp, grouped[..., 3:])
        torch.testing.assert_close(
            _FakeQProj(group_size, rank)._local_view(grouped), per_rank[rank]
        )


def test_mla_wrapper_qrep_reshapes_group_heads_and_passes_both_views():
    """With query replication the projection returns the group's heads. Viewing
    them as `num_heads` would fold the extra heads into the token axis, which is
    the `dcp_size * num_tokens` shape mismatch qrep used to crash on."""
    group_size, local_heads, num_tokens = 4, 2, 3
    wrapper = VllmMultiHeadLatentAttentionWrapper.__new__(
        VllmMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.kimi_preprocess = None
    wrapper.q_lora_rank = None
    wrapper.kv_lora_rank = 4
    wrapper.qk_nope_head_dim = 3
    wrapper.qk_rope_head_dim = 2
    wrapper.qk_head_dim = 5
    wrapper.num_heads = local_heads
    wrapper.v_head_dim = 2
    wrapper.dcp_q_replicate = True
    wrapper.dcp_group_size = group_size
    wrapper.is_sparse = True
    wrapper.indexer = None
    wrapper.skip_topk = True
    wrapper.topk_indices_buffer = torch.zeros((num_tokens, 8), dtype=torch.int32)
    group_width = local_heads * group_size * wrapper.qk_head_dim
    wrapper.kv_a_proj_with_mqa = lambda hidden: (
        torch.ones((hidden.shape[0], 6), dtype=hidden.dtype),
        None,
    )
    q_values = torch.arange(num_tokens * group_width, dtype=torch.bfloat16).view(
        num_tokens, group_width
    )
    wrapper.q_proj = _FakeQProj(group_size, rank_in_group=2, out=q_values)
    wrapper.kv_a_layernorm = lambda value: value
    wrapper.rotary_emb = None
    wrapper.g_proj = None
    wrapper.gate_is_fused = False
    wrapper.mla_attn = MagicMock(
        return_value=torch.ones((num_tokens, 4), dtype=torch.bfloat16)
    )
    wrapper.o_proj = lambda value: (value, None)

    VllmMultiHeadLatentAttentionWrapper.forward(
        wrapper,
        positions=torch.arange(num_tokens),
        hidden_states=torch.ones((num_tokens, 8), dtype=torch.bfloat16),
    )

    (q_nope, q_pe) = wrapper.mla_attn.call_args.args[0]
    rep_nope, rep_pe = wrapper.mla_attn.call_args.kwargs["q_dcp_replicated"]
    # The token axis survives; only the head axis carries the group.
    assert q_nope.shape == (num_tokens, local_heads, wrapper.qk_nope_head_dim)
    assert rep_nope.shape == (
        num_tokens,
        local_heads * group_size,
        wrapper.qk_nope_head_dim,
    )
    assert rep_pe.shape == (
        num_tokens,
        local_heads * group_size,
        wrapper.qk_rope_head_dim,
    )
    # And the local view is this rank's block of the replicated tensor.
    start = 2 * local_heads
    torch.testing.assert_close(q_nope, rep_nope[:, start : start + local_heads])
    torch.testing.assert_close(q_pe, rep_pe[:, start : start + local_heads])


def test_mla_forward_qrep_skips_the_gather_and_absorbs_group_heads(monkeypatch):
    """On the sparse DCP path a replicated q is consumed directly, against the
    group-wide `W_UK_T`, and the per-step head all-gather is not run."""
    dcp_size, local_heads, num_tokens = 2, 16, 4
    impl, layer, inputs, _ = _make_impl_and_layer()
    impl.dcp_size = dcp_size
    group_heads = local_heads * dcp_size

    torch.manual_seed(1)
    layer.W_UK_T_dcp_qrep = torch.randn(group_heads, 128, 512) * 0.05
    layer.W_UK_T_dcp_qrep_scale = None
    partial = torch.randn(num_tokens, group_heads, 512)
    lse = torch.zeros(num_tokens, group_heads)
    layer.sparse_mla_dcp_op = MagicMock(return_value=(partial, lse))
    inputs["kv_cache"] = tuple(
        torch.zeros(spec.shape, dtype=spec.torch_dtype)
        for spec in PallasMLAttentionBackend.get_sparse_kv_cache_specs(
            2, 8, 576, torch.float8_e4m3fn
        )
    )

    def _explode(*_args):
        raise AssertionError("qrep must not all-gather the query heads")

    monkeypatch.setattr(attention_adapter, "all_gather_heads", _explode)
    monkeypatch.setattr(attention_adapter, "all_gather_q_nope_pe_heads", _explode)
    monkeypatch.setattr(
        attention_adapter,
        "merge_lse_partials_scatter_heads",
        lambda out, _lse, **_kw: out[:, :local_heads, :],
    )

    q_nope_rep = torch.randn(num_tokens, group_heads, 128)
    q_pe_rep = torch.randn(num_tokens, group_heads, 64)
    impl.forward(
        layer=layer,
        **inputs,
        topk_indices=torch.zeros((num_tokens, 8), dtype=torch.int32),
        q_dcp_replicated=(q_nope_rep, q_pe_rep),
    )

    layer.sparse_mla_dcp_op.assert_called_once()
    _cache, ql_nope, q_pe, *_ = layer.sparse_mla_dcp_op.call_args.args
    expected_ql = torch.bmm(
        q_nope_rep.transpose(0, 1), layer.W_UK_T_dcp_qrep
    ).transpose(0, 1)
    torch.testing.assert_close(ql_nope, expected_ql)
    torch.testing.assert_close(q_pe, q_pe_rep)


def test_mla_forward_without_qrep_still_gathers(monkeypatch):
    """The gather is only skipped when a replicated q actually arrives; the
    unreplicated DCP path is untouched."""
    dcp_size, local_heads, num_tokens = 2, 16, 4
    impl, layer, inputs, _ = _make_impl_and_layer()
    impl.dcp_size = dcp_size

    partial = torch.randn(num_tokens, local_heads * dcp_size, 512)
    lse = torch.zeros(num_tokens, local_heads * dcp_size)
    layer.sparse_mla_dcp_op = MagicMock(return_value=(partial, lse))
    inputs["kv_cache"] = tuple(
        torch.zeros(spec.shape, dtype=spec.torch_dtype)
        for spec in PallasMLAttentionBackend.get_sparse_kv_cache_specs(
            2, 8, 576, torch.float8_e4m3fn
        )
    )

    gathered = []
    monkeypatch.setattr(
        attention_adapter,
        "all_gather_heads",
        lambda t: (gathered.append(t), t.repeat(1, dcp_size, 1))[1],
    )
    monkeypatch.setattr(
        attention_adapter,
        "merge_lse_partials_scatter_heads",
        lambda out, _lse, **_kw: out[:, :local_heads, :],
    )

    impl.forward(
        layer=layer,
        **inputs,
        topk_indices=torch.zeros((num_tokens, 8), dtype=torch.int32),
    )

    assert len(gathered) == 2
    assert all(t.shape[1] == local_heads for t in gathered)

    # Also verify the default fused Q gather path (`all_gather_q_nope_pe_heads`)
    fused_calls = []
    monkeypatch.setattr(
        attention_adapter,
        "all_gather_heads",
        cp_mla.all_gather_heads,
    )
    monkeypatch.setattr(
        attention_adapter,
        "all_gather_q_nope_pe_heads",
        lambda nope, pe, rank: (
            fused_calls.append((nope, pe, rank)),
            (nope.repeat(1, dcp_size, 1), pe.repeat(1, dcp_size, 1)),
        )[1],
    )
    impl.forward(
        layer=layer,
        **inputs,
        topk_indices=torch.zeros((num_tokens, 8), dtype=torch.int32),
    )
    assert len(fused_calls) == 1
    assert fused_calls[0][0].shape[1] == local_heads
    assert fused_calls[0][1].shape[1] == local_heads
    assert fused_calls[0][2] == layer.kv_lora_rank

    # Negative control: disabling TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER falls back
    # to two separate all_gather_heads calls.
    unfused_calls = []
    monkeypatch.setenv("TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", "0")
    monkeypatch.setattr(
        attention_adapter,
        "all_gather_heads",
        lambda t: (unfused_calls.append(t), t.repeat(1, dcp_size, 1))[1],
    )
    impl.forward(
        layer=layer,
        **inputs,
        topk_indices=torch.zeros((num_tokens, 8), dtype=torch.int32),
    )
    assert len(unfused_calls) == 2
    assert len(fused_calls) == 1


def test_mla_forward_uses_context_without_runtime_kv_scale_state():
    attention = VllmMLAAttention.__new__(VllmMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.layer_name = "model.layers.0.attn"
    attention.impl = MagicMock()
    attention.impl.forward.return_value = "out"
    q, kv_c_normed, k_pe = torch.zeros(1), torch.zeros(2), torch.zeros(3)
    with (
        patch.object(
            torch.ops.vllm, "maybe_calc_kv_scales", create=True
        ) as calc_scales,
        patch(
            "vllm_torchtpu.layers.adapter.custom_ops."
            "mla_attention_op.get_attention_context",
            return_value=("meta", None, "cache", None),
        ),
    ):
        assert attention(q, kv_c_normed, k_pe) == "out"
    calc_scales.assert_not_called()
    attention.impl.forward.assert_called_once()
    assert attention.impl.forward.call_args.kwargs["attn_metadata"] == "meta"
    assert attention.impl.forward.call_args.kwargs["kv_cache"] == "cache"


def test_indexer_kv_cache_mode_env_validation(monkeypatch):
    """VLLM_TPU_INDEXER_KV_CACHE_MODE defaults to 'sal', normalizes case, and
    rejects unknown modes."""
    import importlib

    st_mod = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    assert "VLLM_TPU_INDEXER_KV_CACHE_MODE" in TpuPlatform.additional_env_vars
    assert "TPU_USE_RAIDEN_KV_CACHE_MANAGER" in TpuPlatform.additional_env_vars
    assert "TPU_RAIDEN_GLM_ADMISSION" in TpuPlatform.additional_env_vars
    assert "TPU_KV_RESHARD_TRANSPORT" in TpuPlatform.additional_env_vars
    assert (
        envs.DEFAULT_INDEXER_KV_CACHE_MODE
        == st_mod.DEFAULT_INDEXER_KV_CACHE_MODE
        == "sal"
    )
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    assert envs.environment_variables["VLLM_TPU_INDEXER_KV_CACHE_MODE"]() == "sal"
    assert st_mod.indexer_kv_layout(256) == st_mod.KVLayout.SEQ_ALONG_LANE

    for raw, expected in [("HAS", "has"), ("SAL", "sal"), ("Xla", "xla")]:
        monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raw)
        assert (
            envs.environment_variables["VLLM_TPU_INDEXER_KV_CACHE_MODE"]() == expected
        )

    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "invalid_mode")
    with pytest.raises(ValueError, match="VLLM_TPU_INDEXER_KV_CACHE_MODE"):
        envs.environment_variables["VLLM_TPU_INDEXER_KV_CACHE_MODE"]()


@pytest.mark.parametrize(
    "num_tokens,enabled,min_tokens,expected",
    [
        (16, True, 64, False),  # decode bucket keeps the all-gather merge
        (32, True, 64, False),
        (64, True, 64, True),  # the threshold itself takes the all-to-all
        (4096, True, 64, True),
        (4096, False, 64, False),  # the flag off wins at every bucket
        (16, False, 0, False),
        (16, True, 0, True),  # min_tokens=0: all-to-all at every bucket
    ],
)
def test_dcp_indexer_a2a_is_chosen_per_bucket(
    num_tokens, enabled, min_tokens, expected
):
    assert _dcp_indexer_a2a_for_bucket(num_tokens, enabled, min_tokens) is expected


def test_dcp_indexer_a2a_env_defaults_and_forwarding(monkeypatch):
    """On by default, threshold 512, and both reach the workers on multi-host."""
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    monkeypatch.delenv("TPU_DCP_INDEXER_A2A_MERGE", raising=False)
    monkeypatch.delenv("TPU_DCP_INDEXER_A2A_MIN_TOKENS", raising=False)
    assert envs.environment_variables["TPU_DCP_INDEXER_A2A_MERGE"]() is True
    assert envs.environment_variables["TPU_DCP_INDEXER_A2A_MIN_TOKENS"]() == 512

    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_MERGE", "0")
    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_MIN_TOKENS", "256")
    assert envs.environment_variables["TPU_DCP_INDEXER_A2A_MERGE"]() is False
    assert envs.environment_variables["TPU_DCP_INDEXER_A2A_MIN_TOKENS"]() == 256

    monkeypatch.setenv("TPU_DCP_INDEXER_A2A_MIN_TOKENS", "-1")
    with pytest.raises(ValueError, match="TPU_DCP_INDEXER_A2A_MIN_TOKENS"):
        envs.environment_variables["TPU_DCP_INDEXER_A2A_MIN_TOKENS"]()

    assert "TPU_DCP_INDEXER_A2A_MERGE" in TpuPlatform.additional_env_vars
    assert "TPU_DCP_INDEXER_A2A_MIN_TOKENS" in TpuPlatform.additional_env_vars


def test_resolve_indexer_kv_cache_mode_and_fallback_warning(monkeypatch, caplog):
    """Single-source layout + strict fallback rule: HAS may fall back to 'xla',
    SAL never falls back to 'xla', and block_size < 128 downgrades SAL layout to HAS."""
    import importlib
    import logging

    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    st_mod = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(mla_attention_op, "_INDEXER_KV_CACHE_FALLBACK_LOGGED", set())
    monkeypatch.setattr(mla_attention_op, "_INDEXER_KV_CACHE_INFO_LOGGED", set())
    monkeypatch.setattr(st_mod, "_INDEXER_KV_LAYOUT_FALLBACK_LOGGED", set())
    cache_has = jnp.zeros((4, 256, 4, 256), dtype=jnp.uint8)
    cache_has_64 = jnp.zeros((4, 16, 4, 256), dtype=jnp.uint8)
    cache_sal = jnp.zeros((4, 33, 4, 1024), dtype=jnp.uint8)
    k_packed = jnp.zeros((8, 129), dtype=jnp.uint8)

    # 'xla' never warns regardless of backend and logs info once.
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    caplog.clear()
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has, k_packed, mode="xla"
    ) == ("xla", KVLayout.HEAD_ALONG_SUBLANE)
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has, k_packed, mode="xla"
    ) == ("xla", KVLayout.HEAD_ALONG_SUBLANE)
    info_msgs = [
        r.message for r in caplog.records if "indexer kv cache: mode=" in r.message
    ]
    assert info_msgs == [
        "indexer kv cache: mode=xla writer=xla layout=HAS cache_shape=(4, 256, 4, 256)"
    ]

    # On CPU, 'has' falls back to 'xla' and warns once per (mode, reason).
    caplog.clear()
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has, k_packed, mode="has"
    ) == ("xla", KVLayout.HEAD_ALONG_SUBLANE)
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has, k_packed, mode="has"
    ) == ("xla", KVLayout.HEAD_ALONG_SUBLANE)
    warnings = [
        r.message
        for r in caplog.records
        if "VLLM_TPU_INDEXER_KV_CACHE_MODE" in r.message
    ]
    assert len(warnings) == 1
    assert "falling back to 'xla'" in warnings[0]

    # On CPU, 'sal' with a SAL-shaped cache MUST raise, never fall back to XLA.
    with pytest.raises(ValueError, match="cannot fall back to 'xla'"):
        mla_attention_op._resolve_indexer_kv_cache_mode(cache_sal, k_packed, mode="sal")

    # On TPU, valid HAS and SAL shapes resolve to ('pallas', layout).
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    caplog.clear()
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has, k_packed, mode="has"
    ) == ("pallas", KVLayout.HEAD_ALONG_SUBLANE)
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_sal, k_packed, mode="sal"
    ) == ("pallas", KVLayout.SEQ_ALONG_LANE)
    prefix = "indexer kv cache: mode="
    info_msgs = [r.message for r in caplog.records if prefix in r.message]
    assert info_msgs == [
        f"{prefix}has writer=pallas layout=HAS cache_shape=(4, 256, 4, 256)",
        f"{prefix}sal writer=pallas layout=SAL cache_shape=(4, 33, 4, 1024)",
    ]

    # Mismatched cache shape (HAS cache with block_size=1024 under mode='sal', or
    # SAL cache under mode='has'/'xla') raises ValueError.
    with pytest.raises(ValueError, match="does not match expected"):
        mla_attention_op._resolve_indexer_kv_cache_mode(cache_has, k_packed, mode="sal")
    with pytest.raises(ValueError, match="does not match expected"):
        mla_attention_op._resolve_indexer_kv_cache_mode(cache_sal, k_packed, mode="has")

    # SAL with unsupported token count raises ValueError rather than falling back.
    with pytest.raises(ValueError, match="cannot fall back to 'xla'"):
        mla_attention_op._resolve_indexer_kv_cache_mode(
            cache_sal, jnp.zeros((65536, 129), dtype=jnp.uint8), mode="sal"
        )

    # block_size=64 + mode='sal' downgrades layout via indexer_kv_layout to HAS
    # (warning once) and uses the Pallas 'has' writer on TPU.
    caplog.clear()
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has_64, k_packed, mode="sal"
    ) == ("pallas", KVLayout.HEAD_ALONG_SUBLANE)
    assert mla_attention_op._resolve_indexer_kv_cache_mode(
        cache_has_64, k_packed, mode="sal"
    ) == ("pallas", KVLayout.HEAD_ALONG_SUBLANE)
    layout_warnings = [
        r.message
        for r in caplog.records
        if "requires block_size to be a positive multiple of 128" in r.message
    ]
    assert len(layout_warnings) == 1


def test_sparse_attn_indexer_streamidx_op_xla_has_sal_equivalence(monkeypatch):
    """_build_streamidx_op produces bit-identical cache updates and top-k args
    across xla, has, and sal."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        kv_cache_append,
    )
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    # Capture the inner JAX function passed to pallas.jax_op and run the writer
    # in interpret mode when executing on CPU so the full HAS/SAL paths run.
    captured = {}

    def _fake_jax_op(name, fn, donate_argnums=()):
        captured["name"] = name
        captured["fn"] = fn
        captured["donate_argnums"] = donate_argnums
        op = MagicMock(side_effect=fn)
        op.register_fake = MagicMock()
        return op

    monkeypatch.setattr(mla_attention_op.pallas, "jax_op", _fake_jax_op)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    orig_append = kv_cache_append.coalesced_kv_append
    monkeypatch.setattr(
        mla_attention_op,
        "coalesced_kv_append",
        lambda *a, **kw: orig_append(*a, **kw, interpret=True),
    )

    recorded_topk_calls = []

    def _fake_streamindex_topk(
        q,
        weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        *,
        k,
        kv_layout,
        **kwargs,
    ):
        recorded_topk_calls.append((kv_layout, np.asarray(cache_kv)))
        return jnp.zeros((q.shape[0], k), dtype=jnp.int32)

    monkeypatch.setattr(mla_attention_op, "streamindex_topk", _fake_streamindex_topk)

    rng = np.random.default_rng(42)
    num_pages, page_size, head_dim, lkv_dim = 4, 128, 128, 256
    num_tokens, num_heads, topk = 6, 32, 16
    init_has = jnp.asarray(
        rng.integers(
            0, 256, size=(num_pages, page_size // 4, 4, lkv_dim), dtype=np.uint8
        )
    )
    init_sal = st_lib.convert_cache_to_seq_along_lane(
        init_has, actual_head_dim=head_dim
    )
    q_bytes = jnp.asarray(
        rng.integers(0, 256, size=(num_tokens, num_heads, head_dim), dtype=np.uint8)
    )
    weights = jnp.ones((num_tokens, num_heads), dtype=jnp.bfloat16)
    k_packed = jnp.asarray(
        rng.integers(0, 256, size=(num_tokens, head_dim + 1), dtype=np.uint8)
    )
    seq_lens = jnp.array([5, 67], dtype=jnp.int32)
    page_indices = jnp.array([2, 0, 1, 3], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 2, 6], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)

    # Drop a value a previous test's monkeypatch.setattr undo may have left in the
    # module dict, so `envs.__getattr__` resolves the variable afresh. A bare
    # `delattr(raising=False)` is not enough: `hasattr` is True through
    # `__getattr__`, so it raises on a module dict without the name.
    if "VLLM_TPU_INDEXER_KV_CACHE_MODE" in vars(envs):
        monkeypatch.delattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE")
    results = {}
    for mode, init_cache in [
        ("xla", init_has),
        ("has", init_has),
        ("sal", init_sal),
    ]:
        monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", mode)
        indexer = mla_attention_op.VllmTPUSparseAttnIndexer.__new__(
            mla_attention_op.VllmTPUSparseAttnIndexer
        )
        indexer.topk_tokens = topk
        indexer.k_cache = SimpleNamespace(prefix="model.layers.0.indexer")
        indexer._build_streamidx_op()
        assert captured["donate_argnums"] == (2,)
        new_cache, topk_out = captured["fn"](
            q_bytes,
            weights,
            jnp.copy(init_cache),
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
        results[mode] = (new_cache, np.asarray(topk_out))

    assert recorded_topk_calls[0][0] == KVLayout.HEAD_ALONG_SUBLANE
    assert recorded_topk_calls[1][0] == KVLayout.HEAD_ALONG_SUBLANE
    assert recorded_topk_calls[2][0] == KVLayout.SEQ_ALONG_LANE

    # HAS Pallas cache update matches XLA cache update bit-for-bit.
    np.testing.assert_array_equal(
        np.asarray(results["has"][0]), np.asarray(results["xla"][0])
    )
    # SAL Pallas cache update matches converted XLA cache update bit-for-bit.
    expected_sal = st_lib.convert_cache_to_seq_along_lane(
        results["xla"][0], actual_head_dim=head_dim
    )
    np.testing.assert_array_equal(
        np.asarray(results["sal"][0]), np.asarray(expected_sal)
    )
    np.testing.assert_array_equal(results["has"][1], results["xla"][1])
    np.testing.assert_array_equal(results["sal"][1], results["xla"][1])


def test_cp_scatter_indexer_k_xla_has_sal_equivalence(monkeypatch):
    """_cp_scatter_indexer_k updates the DCP shard identically across xla, has,
    and sal without re-resolving mode inside shard_map."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as st_lib
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        DCP_AXIS_NAME,
        KVLayout,
        kv_cache_append,
    )
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    orig_append_dcp = kv_cache_append.coalesced_kv_append_dcp
    monkeypatch.setattr(
        mla_attention_op,
        "coalesced_kv_append_dcp",
        lambda *a, **kw: orig_append_dcp(*a, **kw, interpret=True),
    )

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), (DCP_AXIS_NAME,))
    rng = np.random.default_rng(99)
    num_pages, page_size, head_dim, lkv_dim = 4, 128, 128, 256
    num_tokens = 8
    init_has = jnp.asarray(
        rng.integers(
            0, 256, size=(num_pages, page_size // 4, 4, lkv_dim), dtype=np.uint8
        )
    )
    init_sal = st_lib.convert_cache_to_seq_along_lane(
        init_has, actual_head_dim=head_dim
    )
    k_packed = jnp.asarray(
        rng.integers(0, 256, size=(num_tokens, head_dim + 1), dtype=np.uint8)
    )
    seq_lens = jnp.array([6, 70], dtype=jnp.int32)
    page_indices = jnp.array([1, 3, 0, 2], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 3, 8], dtype=jnp.int32)

    common_kw = dict(
        mesh=mesh,
        cp_size=1,
        interleave_size=1,
        axis_name=DCP_AXIS_NAME,
    )
    out_xla = mla_attention_op._cp_scatter_indexer_k(
        jnp.copy(init_has),
        k_packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        writer="xla",
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        **common_kw,
    )
    out_has = mla_attention_op._cp_scatter_indexer_k(
        jnp.copy(init_has),
        k_packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        writer="pallas",
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
        **common_kw,
    )
    out_sal = mla_attention_op._cp_scatter_indexer_k(
        jnp.copy(init_sal),
        k_packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        writer="pallas",
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        **common_kw,
    )

    np.testing.assert_array_equal(np.asarray(out_has), np.asarray(out_xla))
    np.testing.assert_array_equal(
        np.asarray(out_sal),
        np.asarray(
            st_lib.convert_cache_to_seq_along_lane(out_xla, actual_head_dim=head_dim)
        ),
    )

    with pytest.raises(ValueError, match="only supports HEAD_ALONG_SUBLANE"):
        mla_attention_op._cp_scatter_indexer_k(
            jnp.copy(init_sal),
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            writer="xla",
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            **common_kw,
        )


def test_indexer_kv_layout_and_op_resolution_raiden_glm_admission(monkeypatch, caplog):
    """Raiden GLM admission downgrades default 'sal' to HAS with a one-time
    warning, rejects explicit 'sal' with ValueError, and leaves 'has'/'xla' and
    DSv4 admission unaffected."""
    import importlib
    import logging

    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    st_mod = importlib.import_module(
        "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    )
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(mla_attention_op, "_INDEXER_KV_CACHE_FALLBACK_LOGGED", set())
    monkeypatch.setattr(mla_attention_op, "_INDEXER_KV_CACHE_INFO_LOGGED", set())
    monkeypatch.setattr(st_mod, "_INDEXER_KV_LAYOUT_FALLBACK_LOGGED", set())

    # Pin the flags as module attributes rather than through os.environ: the
    # helpers read `envs.<NAME>`, so this stays correct after enable_envs_cache()
    # and regardless of attributes other tests' monkeypatch undo left behind.
    # The mode is pinned both ways, since explicitness is read from os.environ.
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", False)
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_ADMISSION", False)
    monkeypatch.setattr(envs, "TPU_RAIDEN_DSV4_ADMISSION", False)

    # 1. No Raiden flags -> default 'sal' selects SEQ_ALONG_LANE.
    assert not envs.use_raiden_glm_admission()
    assert st_mod.indexer_kv_layout(256) == KVLayout.SEQ_ALONG_LANE

    # 2. Only one of the two flags or DSv4 admission alone -> still SEQ_ALONG_LANE.
    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_RAIDEN_DSV4_ADMISSION", True)
    assert not envs.use_raiden_glm_admission()
    assert st_mod.indexer_kv_layout(256) == KVLayout.SEQ_ALONG_LANE

    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", False)
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    assert not envs.use_raiden_glm_admission()
    assert st_mod.indexer_kv_layout(256) == KVLayout.SEQ_ALONG_LANE

    # 3. Default 'sal' + Raiden GLM admission -> HEAD_ALONG_SUBLANE + one warning.
    monkeypatch.setattr(envs, "TPU_USE_RAIDEN_KV_CACHE_MANAGER", True)
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_ADMISSION", True)
    assert envs.use_raiden_glm_admission()
    assert not envs.is_indexer_kv_cache_mode_explicit()
    caplog.clear()
    assert st_mod.indexer_kv_layout(256) == KVLayout.HEAD_ALONG_SUBLANE
    assert st_mod.indexer_kv_layout(256) == KVLayout.HEAD_ALONG_SUBLANE
    warnings = [
        r.message
        for r in caplog.records
        if "incompatible with Raiden GLM admission" in r.message
    ]
    assert len(warnings) == 1
    assert "VLLM_TPU_INDEXER_KV_CACHE_MODE=has" in warnings[0]

    # 4. Explicit 'has' and 'xla' + Raiden GLM admission -> HAS, no warning.
    for explicit_mode in ("has", "xla"):
        caplog.clear()
        monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", explicit_mode)
        monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", explicit_mode)
        assert envs.is_indexer_kv_cache_mode_explicit()
        assert st_mod.indexer_kv_layout(256) == KVLayout.HEAD_ALONG_SUBLANE
        assert not [
            r.message
            for r in caplog.records
            if "incompatible with Raiden GLM admission" in r.message
        ]

    # 5. Explicit 'sal' (env or mode arg) + Raiden GLM admission -> ValueError.
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    assert envs.is_indexer_kv_cache_mode_explicit()
    with pytest.raises(ValueError, match="Raiden GLM admission"):
        st_mod.indexer_kv_layout(256)
    monkeypatch.delenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "sal")
    with pytest.raises(ValueError, match="Raiden GLM admission"):
        st_mod.indexer_kv_layout(256, mode="sal")

    # 6. _resolve_indexer_kv_cache_mode under the downgrade returns
    #    ('pallas', HEAD_ALONG_SUBLANE) on TPU and ('xla', HEAD_ALONG_SUBLANE) on CPU.
    cache_has = jnp.zeros((4, 64, 4, 256), dtype=jnp.uint8)
    k_packed = jnp.zeros((8, 129), dtype=jnp.uint8)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert mla_attention_op._resolve_indexer_kv_cache_mode(cache_has, k_packed) == (
        "pallas",
        KVLayout.HEAD_ALONG_SUBLANE,
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert mla_attention_op._resolve_indexer_kv_cache_mode(cache_has, k_packed) == (
        "xla",
        KVLayout.HEAD_ALONG_SUBLANE,
    )
    with pytest.raises(ValueError, match="Raiden GLM admission"):
        mla_attention_op._resolve_indexer_kv_cache_mode(cache_has, k_packed, mode="sal")
