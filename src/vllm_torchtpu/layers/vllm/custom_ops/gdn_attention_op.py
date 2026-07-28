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

import functools

import jax
import torch
from einops import rearrange
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_torchtpu import envs
from vllm_torchtpu.layers.common.gdn_attention import (
    GdnAttentionConfig, run_jax_gdn_attention_pooled)
from vllm_torchtpu.layers.common.ragged_gated_delta_rule_wrapper import \
    RaggedGatedDeltaRuleImpl
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def gdn_attention_pooled_core_tpu(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    config: GdnAttentionConfig,
) -> tuple[jax.Array, jax.Array]:
    return run_jax_gdn_attention_pooled(
        mixed_qkv,
        b,
        a,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
        mesh=mesh,
        config=config,
    )


def _get_pcp_size(vllm_config: VllmConfig) -> int:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    return pcp_size if pcp_size > 1 else 1


@QwenGatedDeltaNetAttention.register_oot
class VllmGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bound by the runner during KV-cache initialization; None until then
        # (warmup/profiling runs check this).
        self.kv_cache = None
        self.gdn_pooled_op = self._build_pooled_gdn_op()

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        assert isinstance(spec, MambaSpec)
        if _get_pcp_size(vllm_config) > 1:
            # TODO: GDN state under prefill context parallelism still needs a
            # unified-pool implementation — the PCP prefill kernel streams
            # dense conv/recurrent tensors, which the pool does not hand out.
            # Reject at config time instead of silently running without PCP.
            raise NotImplementedError(
                "Prefill context parallelism is not supported for GDN layers "
                "on the unified KV pool.")
        return spec

    def _build_pooled_gdn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        config = GdnAttentionConfig(
            ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl(
                envs.RAGGED_GATED_DELTA_RULE_IMPL))
        local_num_v_heads = self.num_v_heads // self.tp_size
        wrapped_fn = functools.partial(
            gdn_attention_pooled_core_tpu,
            mesh=vllm_context.mesh,
            n_kq=self.num_k_heads // self.tp_size,
            n_v=local_num_v_heads,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
            kernel_size=self.conv_kernel_size,
            # Manager block size: the pool may be born at a smaller kernel
            # granularity for backends with a fixed kernel block.
            pool_block_tokens=(
                vllm_context.vllm_config.cache_config.block_size),
            config=config,
        )
        op_name = f"pallas::gdn_attention_pooled_{self.prefix.replace('.', '_')}"
        # The recurrent state (arg 3) is the attention-shaped pool: the ssm
        # and conv byte-regions are read/written through it by the pool
        # adapters, so it is the only donated input. inplace-donation
        # aliases its program-level output to the input buffer so the
        # ~pool-sized update is not double-counted at compile.
        gdn_jax_op = pallas.jax_op(op_name, wrapped_fn, donate_argnums=(3, ))

        def _fake_gdn(mixed_qkv, _b, _a, recurrent_state, *args, **kwargs):
            num_tokens = mixed_qkv.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            out = torch.empty(out_shape,
                              dtype=mixed_qkv.dtype,
                              device=mixed_qkv.device)
            return torch.empty_like(recurrent_state), out

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                     recurrent_state: torch.Tensor, conv_weight: torch.Tensor,
                     conv_bias: torch.Tensor | None, A_log: torch.Tensor,
                     dt_bias: torch.Tensor, state_indices: torch.Tensor,
                     query_start_loc: torch.Tensor,
                     request_distribution: torch.Tensor,
                     seq_lens: torch.Tensor) -> torch.Tensor:
            new_rec, outputs = gdn_jax_op(mixed_qkv, b, a, recurrent_state,
                                          conv_weight, conv_bias, A_log,
                                          dt_bias, state_indices,
                                          query_start_loc,
                                          request_distribution, seq_lens)

            # Plain donation + copy_ writeback (aliased in-place by XLA).
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba)
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value))
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        kv_cache = self.kv_cache
        local_num_v_heads = self.num_v_heads // self.tp_size

        # During warmup or memory profiling, the kv_cache might not be allocated yet
        if kv_cache is None or kv_cache[0].numel() == 0:
            core_attn_out = torch.zeros(
                (num_tokens, local_num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            recurrent_state = None
        else:
            fc = get_forward_context()
            attn_metadata = fc.attn_metadata[self.prefix]

            if len(kv_cache) != 1:
                # TODO: PCP-streaming GDN prefill still needs a unified-pool
                # implementation; it is the only consumer left that wants
                # dense conv/recurrent state tensors. Until it streams its
                # state through the pool adapters, fail loudly here rather
                # than silently taking a non-PCP path.
                raise NotImplementedError(
                    "GDN attention only supports the unified KV pool "
                    f"(got {len(kv_cache)} state tensors). PCP-streaming "
                    "prefill is not yet wired to the pool.")

            # Unified block pool: the attention-shaped pool carries the ssm
            # and conv byte-regions; state ids come from the pool metadata
            # builder.
            (recurrent_state, ) = kv_cache
            assert attn_metadata.mamba_state_indices is not None
            state_indices = attn_metadata.mamba_state_indices.to(torch.int32)
            core_attn_out = self.gdn_pooled_op(
                mixed_qkv, b, a, recurrent_state, self.conv1d.weight,
                self.conv1d.bias, self.A_log, self.dt_bias, state_indices,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution, attn_metadata.seq_lens)
            if core_attn_out.shape[0] != num_tokens:
                raise RuntimeError(
                    "GDN op returned an incompatible output shape.")

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out = core_attn_out.view(num_tokens, local_num_v_heads,
                                           self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
