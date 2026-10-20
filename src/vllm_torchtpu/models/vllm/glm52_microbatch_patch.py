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
"""Barrier-staged post-attention micro-batching for GLM-5.2 / DeepSeek-V3.2 MLA.

When `TPU_TOKEN_MICROBATCHES >= 2` and the padded token bucket is at least
`TPU_MICROBATCH_MIN_TOKENS` (default 1024):

  1. Pre-Attention (`input_layernorm`, `fused_qkv_a_proj`, `q_a_layernorm`,
     `q_b_proj`, `kv_a_layernorm`, RoPE, `indexer` + `streamindex_topk_dcp`,
     `W_UK_T` BMM, `quantize_kv`, and `sparse_mla_dcp_op`) executes on the full
     contiguous `[0:N]` batch without any tensor splitting or `torch.cat`
     overhead, preserving 100% of XLA's native overlap of `q_b_proj` with
     `sc_gather_pack_owned` and `W_UK_T` BMM with `sc_native_sc_gather`.
  2. Post-Attention (`dcp_lse_ag_and_weight` -> `dcp_reduce_scatter` ->
     `W_UV` BMM -> `output_gate` -> `o_proj` GEMM -> `o_proj` TP `all-reduce`)
     is pipelined across `K = max(TPU_TOKEN_MICROBATCHES, TPU_DCP_MERGE_CHUNKED)`
     token chunks using `jax.lax.optimization_barrier` so that chunk `c`'s
     SparseCore-offloaded `reduce-scatter` executes concurrently with chunk
     `c - 1`'s TensorCore `W_UV` BMM + `o_proj` GEMM + `o_proj` TP `all-reduce`
     AND chunk `c + 1`'s TensorCore `all-gather(lse)` + `logsumexp` + `weighted`
     multiply. Running `W_UV` BMM per chunk directly on `dcp_reduce_scatter`'s
     output also eliminates the `{2,0,1} -> {2,1,0} -> {2,0,1}` transpose-cat-
     transpose overhead.
  3. Post-Attention LayerNorm + MoE (`post_attention_layernorm`, `mlp`) executes
     on the full contiguous `[0:N]` batch, with `lax.optimization_barrier`
     decoupling `shared_experts.down_proj` from `routed` so `down_proj` overlaps
     with SparseCore EP combine (`sc_ragged_gather_reduce_v2`).
"""

from typing import Any, cast

import jax
import torch
from torch_tpu._internal import pallas
from vllm.compilation import backends as vllm_compilation_backends
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.linear import DCPGroupColumnParallelLinear

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import (
    dcp_lse_ag_and_weight,
    dcp_reduce_scatter,
)
from vllm_torchtpu.layers.adapter import cp_mla_attention as _cp_mla_attention
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import (
    mla_attention_op as _mla_attention_op,
)
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMLAAttention,
    VllmTPUMultiHeadLatentAttentionWrapper,
)
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.core.quantization import quantize_kv
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_MLA_WRAPPER_FORWARD = VllmTPUMultiHeadLatentAttentionWrapper.forward
UPSTREAM_MOE_APPLY_QUANT_METHOD = VllmMoERunner._apply_quant_method


def _opt_barrier_2_jax(
    a: jax.Array, b: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jax.lax.optimization_barrier((a, b))


def _fake_opt_barrier_2(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(b)


_OPT_BARRIER_2_OP = pallas.jax_op(
    "pallas::glm52_opt_barrier_2", _opt_barrier_2_jax
)
_OPT_BARRIER_2_OP.register_fake(_fake_opt_barrier_2)


def _opt_barrier_3_jax(
    a: jax.Array, b: jax.Array, c: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jax.lax.optimization_barrier((a, b, c))


def _fake_opt_barrier_3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(a), torch.empty_like(b), torch.empty_like(c)


_OPT_BARRIER_3_OP = pallas.jax_op(
    "pallas::glm52_opt_barrier_3", _opt_barrier_3_jax
)
_OPT_BARRIER_3_OP.register_fake(_fake_opt_barrier_3)


def _opt_barrier_2(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.device.type != "tpu":
        return a, b
    return _OPT_BARRIER_2_OP(a, b)


def _opt_barrier_3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if a.device.type != "tpu":
        return a, b, c
    return _OPT_BARRIER_3_OP(a, b, c)


def _staged_post_chunks(num_tokens: int) -> int:
    if num_tokens < envs.TPU_MICROBATCH_MIN_TOKENS:
        return 1
    merge_chunks_fn = getattr(_cp_mla_attention, "_merge_chunks", None)
    base_chunks = (
        merge_chunks_fn(num_tokens)
        if callable(merge_chunks_fn)
        else int(envs.TPU_DCP_MERGE_CHUNKED)
    )
    target = max(int(envs.TPU_TOKEN_MICROBATCHES), base_chunks)
    if target > 1 and num_tokens % (8 * target) == 0:
        return target
    mb = int(envs.TPU_TOKEN_MICROBATCHES)
    if mb > 1 and num_tokens % (8 * mb) == 0:
        return mb
    return 1


def _can_use_staged_dcp_post(
    wrapper: VllmTPUMultiHeadLatentAttentionWrapper,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    prefill_group: Any,
) -> bool:
    if envs.TPU_TOKEN_MICROBATCHES < 2:
        return False
    if vllm_compilation_backends.model_tag != "backbone":
        return False
    if prefill_group is not None:
        return False
    if not wrapper.is_sparse:
        return False
    if getattr(wrapper, "kimi_preprocess", None) is not None:
        return False
    if getattr(wrapper, "g_proj", None) is not None:
        return False
    mla_layer = getattr(wrapper, "mla_attn", None)
    if not isinstance(mla_layer, VllmTPUMLAAttention):
        return False
    impl = cast(PallasMLAttentionBackendImpl, mla_layer.impl)
    if impl.dcp_size <= 1:
        return False
    dcp_group = get_dcp_group()
    if dcp_group is None or int(dcp_group.world_size) <= 1:
        return False
    _, _, kv_cache, _ = get_attention_context(mla_layer.layer_name)
    if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
        return False
    if hidden_states.shape[0] != positions.shape[0]:
        return False
    num_tokens = hidden_states.shape[0]
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and num_tokens <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return False
    return _staged_post_chunks(num_tokens) >= 2


def _attn_post_window(
    wrapper: VllmTPUMultiHeadLatentAttentionWrapper,
    mla_layer: VllmTPUMLAAttention,
    outputs_w: torch.Tensor,
    input_dtype: torch.dtype,
    output_gate_w: torch.Tensor | None,
) -> torch.Tensor:
    outputs_t = outputs_w.reshape(
        -1, mla_layer.num_heads, mla_layer.kv_lora_rank
    ).transpose(0, 1)
    w_uv = (
        mla_layer.W_UV.to(outputs_t.dtype)
        if mla_layer.W_UV.dtype != outputs_t.dtype
        else mla_layer.W_UV
    )
    out_proj = torch.bmm(outputs_t, w_uv)
    if hasattr(mla_layer, "W_UV_scale"):
        out_proj = out_proj * mla_layer.W_UV_scale
    attn_out_w = (
        out_proj.transpose(0, 1)
        .to(input_dtype)
        .reshape(-1, mla_layer.num_heads * mla_layer.v_head_dim)
    )
    if output_gate_w is not None:
        attn_out_w = attn_out_w * output_gate_w.sigmoid()
    return wrapper.o_proj(attn_out_w)[0]


def _staged_dcp_attn_post(
    wrapper: VllmTPUMultiHeadLatentAttentionWrapper,
    mla_layer: VllmTPUMLAAttention,
    partial_full: torch.Tensor,
    lse_full: torch.Tensor,
    input_dtype: torch.dtype,
    output_gate: torch.Tensor | None,
    group: Any,
    num_chunks: int,
) -> torch.Tensor:
    step = partial_full.shape[0] // num_chunks
    partials = [
        partial_full[c * step : (c + 1) * step] for c in range(num_chunks)
    ]
    lses = [lse_full[c * step : (c + 1) * step] for c in range(num_chunks)]
    gates = [
        output_gate[c * step : (c + 1) * step]
        if output_gate is not None
        else None
        for c in range(num_chunks)
    ]

    # Prologue (TensorCore): chunk 0 LSE all-gather + logsumexp + weight multiply.
    weighted_c = dcp_lse_ag_and_weight(partials[0], lses[0], group)

    # Stage 0: SparseCore reduce-scatter(chunk 0) overlapped with
    # TensorCore LSE all-gather + weight(chunk 1).
    weighted_0, lse_1 = _opt_barrier_2(weighted_c, lses[1])
    outputs_prev = dcp_reduce_scatter(weighted_0, input_dtype, group)
    weighted_c = dcp_lse_ag_and_weight(partials[1], lse_1, group)

    h_attn_chunks: list[torch.Tensor] = []

    # Interior Stages c = 1 .. num_chunks - 2:
    # SparseCore reduce-scatter(chunk c) overlapped with
    # TensorCore _attn_post_window(chunk c - 1) [W_UV BMM + o_proj + TP AR] AND
    # TensorCore LSE all-gather + weight(chunk c + 1).
    for c in range(1, num_chunks - 1):
        outputs_prev, weighted_c, lse_next = _opt_barrier_3(
            outputs_prev, weighted_c, lses[c + 1]
        )
        outputs_c = dcp_reduce_scatter(weighted_c, input_dtype, group)
        h_attn_prev = _attn_post_window(
            wrapper, mla_layer, outputs_prev, input_dtype, gates[c - 1]
        )
        weighted_next = dcp_lse_ag_and_weight(
            partials[c + 1], lse_next, group
        )
        h_attn_prev, outputs_prev = _opt_barrier_2(h_attn_prev, outputs_c)
        h_attn_chunks.append(h_attn_prev)
        weighted_c = weighted_next

    # Final Reduce-Scatter Stage c = num_chunks - 1:
    # SparseCore reduce-scatter(chunk K - 1) overlapped with
    # TensorCore _attn_post_window(chunk K - 2) [W_UV BMM + o_proj + TP AR].
    outputs_prev, weighted_last = _opt_barrier_2(outputs_prev, weighted_c)
    outputs_last = dcp_reduce_scatter(weighted_last, input_dtype, group)
    h_attn_prev = _attn_post_window(
        wrapper, mla_layer, outputs_prev, input_dtype, gates[num_chunks - 2]
    )
    h_attn_prev, outputs_last = _opt_barrier_2(h_attn_prev, outputs_last)
    h_attn_chunks.append(h_attn_prev)

    # Epilogue (TensorCore): _attn_post_window(chunk K - 1).
    h_attn_last = _attn_post_window(
        wrapper, mla_layer, outputs_last, input_dtype, gates[num_chunks - 1]
    )
    h_attn_chunks.append(h_attn_last)

    return torch.cat(h_attn_chunks, dim=0)


def patched_mla_wrapper_forward(
    self: VllmTPUMultiHeadLatentAttentionWrapper,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
    prefill_group: Any = None,
) -> torch.Tensor:
    if not _can_use_staged_dcp_post(
        self, positions, hidden_states, prefill_group
    ):
        return UPSTREAM_MLA_WRAPPER_FORWARD(
            self, positions, hidden_states, llama_4_scaling, prefill_group
        )

    q_c = None
    kv_lora = None
    output_gate = None

    if self.q_lora_rank is not None:
        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None
        assert self.q_b_proj is not None

        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        if self.gate_is_fused:
            q_c, kv_lora, output_gate = qkv_lora.split(
                [
                    self.q_lora_rank,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    self.num_heads * self.v_head_dim,
                ],
                dim=-1,
            )
        else:
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        assert self.kv_a_proj_with_mqa is not None
        assert self.q_proj is not None
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]

    assert kv_lora is not None
    kv_c, k_pe = kv_lora.split(
        [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    kv_c_normed = self.kv_a_layernorm(kv_c)

    q = q.view(-1, self.num_heads * self.dcp_group_size, self.qk_head_dim)
    q_nope, q_pe = q.split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.unsqueeze(1)

    if self.rotary_emb is not None:
        q_pe_dtype = q_pe.dtype
        k_pe_dtype = k_pe.dtype
        rope_half_split_fn = getattr(
            _mla_attention_op, "rope_half_split", None
        )
        if getattr(self, "_rope_neox_permuted", False) and callable(
            rope_half_split_fn
        ):
            q_pe, k_pe = rope_half_split_fn(
                self.rotary_emb, positions, q_pe, k_pe
            )
        else:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_pe = q_pe.to(dtype=q_pe_dtype)
        k_pe = k_pe.to(dtype=k_pe_dtype)

    if self.indexer is not None and not self.skip_topk:
        topk_index = self.indexer(
            hidden_states, q_c, positions, self.indexer_rope_emb
        )
    else:
        topk_index = self.topk_indices_buffer
    assert topk_index is not None
    topk_indices = topk_index[: hidden_states.shape[0]]

    if llama_4_scaling is not None:
        q_nope = q_nope * llama_4_scaling
        q_pe = q_pe * llama_4_scaling

    q_dcp_replicated = None
    if self.dcp_q_replicate:
        if self.is_sparse:
            q_dcp_replicated = (q_nope, q_pe)
        local_view = cast(
            DCPGroupColumnParallelLinear, self._q_proj_layer()
        )._local_view
        q_nope, q_pe = local_view(q_nope), local_view(q_pe)

    mla_layer = self.mla_attn
    impl = cast(PallasMLAttentionBackendImpl, mla_layer.impl)
    attn_metadata, _, kv_cache, _ = get_attention_context(mla_layer.layer_name)
    input_dtype = q_nope.dtype

    use_qrep = (
        q_dcp_replicated is not None
        and impl.dcp_size > 1
        and topk_indices is not None
    )
    if use_qrep:
        assert mla_layer.W_UK_T_dcp_qrep is not None
        q_nope_use, q_pe_use = q_dcp_replicated
        w_uk_t_src = mla_layer.W_UK_T_dcp_qrep
        w_uk_t_scale = mla_layer.W_UK_T_dcp_qrep_scale
    else:
        q_nope_use, q_pe_use = q_nope, q_pe
        w_uk_t_src = mla_layer.W_UK_T
        w_uk_t_scale = getattr(mla_layer, "W_UK_T_scale", None)
    attn_heads = mla_layer.num_heads * (impl.dcp_size if use_qrep else 1)

    q_nope_t = q_nope_use.transpose(0, 1)
    w_uk_t = (
        w_uk_t_src.to(q_nope_t.dtype)
        if w_uk_t_src.dtype != q_nope_t.dtype
        else w_uk_t_src
    )
    ql_nope = torch.bmm(q_nope_t, w_uk_t)
    if w_uk_t_scale is not None:
        ql_nope = ql_nope * w_uk_t_scale
    ql_nope = ql_nope.transpose(0, 1).to(input_dtype)

    _, k_scale, _ = impl._get_kv_scales(mla_layer)
    if mla_layer.kv_cache_quantized_dtype is not None:
        assert k_scale is not None
        kv_c_normed, _ = quantize_kv(
            mla_layer.kv_cache_quantized_dtype,
            kv_c_normed,
            value=None,
            k_scale=k_scale,
        )
        k_pe, _ = quantize_kv(
            mla_layer.kv_cache_quantized_dtype,
            k_pe,
            value=None,
            k_scale=k_scale,
        )

    ql_nope_flat = ql_nope.view(-1, attn_heads, mla_layer.kv_lora_rank)
    q_pe_flat = q_pe_use.view(-1, attn_heads, mla_layer.qk_rope_head_dim)
    kv_c_normed_flat = kv_c_normed.view(-1, mla_layer.kv_lora_rank)
    k_pe_flat = k_pe.view(-1, mla_layer.qk_rope_head_dim)

    assert isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2
    assert getattr(mla_layer, "sparse_mla_dcp_op", None) is not None
    if use_qrep:
        ql_nope_dcp, q_pe_dcp = ql_nope_flat, q_pe_flat
    elif (
        envs.TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER
        and ql_nope_flat.dtype == q_pe_flat.dtype
    ):
        ql_nope_dcp, q_pe_dcp = _cp_mla_attention.all_gather_q_nope_pe_heads(
            ql_nope_flat, q_pe_flat, mla_layer.kv_lora_rank
        )
    else:
        ql_nope_dcp = _cp_mla_attention.all_gather_heads(ql_nope_flat)
        q_pe_dcp = _cp_mla_attention.all_gather_heads(q_pe_flat)

    partial_full, lse_full = mla_layer.sparse_mla_dcp_op(
        kv_cache,
        ql_nope_dcp,
        q_pe_dcp,
        kv_c_normed_flat,
        k_pe_flat,
        topk_indices,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
    )

    dcp_group = get_dcp_group()
    assert dcp_group is not None
    num_chunks = _staged_post_chunks(hidden_states.shape[0])
    return _staged_dcp_attn_post(
        self,
        mla_layer,
        partial_full,
        lse_full,
        input_dtype,
        output_gate,
        dcp_group,
        num_chunks,
    )


def patched_moe_apply_quant_method(
    self: VllmMoERunner,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None = None,
    shared_experts_overlapping: bool = False,
) -> tuple[torch.Tensor | None, Any]:
    shared_out, fused_out = UPSTREAM_MOE_APPLY_QUANT_METHOD(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=input_ids,
        shared_experts_overlapping=shared_experts_overlapping,
    )
    if (
        (envs.TPU_TOKEN_MICROBATCHES >= 2 or envs.TPU_SPARSE_MLA_DCP_CHUNK_AG_RS)
        and vllm_compilation_backends.model_tag == "backbone"
        and hidden_states.shape[0] >= envs.TPU_MICROBATCH_MIN_TOKENS
        and shared_out is not None
        and isinstance(fused_out, torch.Tensor)
    ):
        shared_out, fused_out = _opt_barrier_2(shared_out, fused_out)
    return shared_out, fused_out


def maybe_patch_glm52_microbatch(*args: object, **kwargs: object) -> None:
    """Install barrier-staged post-attention micro-batching on MLA wrapper."""
    if envs.TPU_SPARSE_MLA_DCP_CHUNK_AG_RS:
        VllmMoERunner._apply_quant_method = patched_moe_apply_quant_method
        logger.info(
            "Applied TPU patch: GLM-5.2 in-kernel per-chunk DCP AG+RS "
            "(TPU_SPARSE_MLA_DCP_CHUNK_AG_RS=True, min_tokens=%d, bf16=%s, barrier=%s, chunk=%s) + MoE EP overlap.",
            envs.TPU_SPARSE_MLA_DCP_CHUNK_AG_RS_MIN_TOKENS,
            envs.TPU_SPARSE_MLA_DCP_CHUNK_AG_RS_BF16,
            envs.TPU_SPARSE_MLA_DCP_CHUNK_AG_RS_BARRIER,
            envs.TPU_DCP_MLA_TOPK_CAP_CHUNK,
        )
        return None
    if envs.TPU_TOKEN_MICROBATCHES < 2:
        return None
    VllmTPUMultiHeadLatentAttentionWrapper.forward = patched_mla_wrapper_forward
    VllmMoERunner._apply_quant_method = patched_moe_apply_quant_method
    logger.info(
        "Applied TPU patch: GLM-5.2 barrier-staged post-attention micro-batching "
        "(TPU_TOKEN_MICROBATCHES=%d, TPU_MICROBATCH_MIN_TOKENS=%d, TPU_DCP_RS_BF16=%s).",
        envs.TPU_TOKEN_MICROBATCHES,
        envs.TPU_MICROBATCH_MIN_TOKENS,
        envs.TPU_DCP_RS_BF16,
    )
    return None
