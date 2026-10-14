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
"""GLM-5.3-Flash attention: KDA on 34 layers, NoPE MLA on the other 11.

The linear-attention layers are Kimi Delta Attention, not a lookalike. GLM
ships the same ``linear_attn_config`` dict Kimi's config uses (``num_heads``
64, ``head_dim`` 128, ``short_conv_kernel_size`` 4, ``gate_lower_bound``
-5.0) and the same checkpoint tensor names -- ``q/k/v_proj``, ``b_proj``,
``f_a_proj``/``f_b_proj``, ``g_a_proj``/``g_b_proj``, three separate
``q/k/v_conv1d``, ``A_log``, ``dt_bias``, ``o_norm``, ``o_proj``. Every one of
``KimiDeltaAttention``'s fused projections and weight loaders applies
unchanged, so that layer is re-exported as-is.

MLA is *not* re-exportable. Kimi's "NoPE MLA" still carries a rope lane it
simply never rotates (``qk_rope_head_dim`` 64, ``rotary_emb=None``); GLM
dropped the lane entirely -- ``qk_rope_head_dim: 0``, ``q_b_proj`` emits
``64*256`` and ``kv_a_proj_with_mqa`` emits exactly ``kv_lora_rank``. The
Pallas MLA kernel has no zero-width-rope path: it keeps a separate rope KV
cache and lays the lane out against a fixed vector-lane width, so a width of
0 dies at ``q_pe.view(-1, num_heads, 0)``.

``Glm5NextMLA`` below therefore keeps the projections at their true GLM sizes
and synthesizes a *zero* rope lane of width ``ROPE_LANE`` just for the kernel.
Zeros contribute nothing to the QK dot product, so attention scores are
exactly the NoPE scores, provided the softmax scale stays ``qk_nope_head_dim
** -0.5`` rather than being recomputed from the padded width. The cost is
``ROPE_LANE`` extra KV bytes per token per layer -- 576 instead of 512, +12.5%
-- and a negligible amount of flops.

On top of that GLM runs a k-pool compressed DSA indexer that narrows those 11
layers to ~2048 keys; see ``indexer.py``. Set ``GLM53_DENSE_DSA=1`` to build
them as plain dense MLA instead, which is exactly equivalent below
``index_topk`` tokens and O(T^2) above it -- useful for A/B-ing the indexer.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules

# Subclass the TPU wrapper, not vLLM's `MultiHeadLatentAttentionWrapper`.
# `PluggableLayer.__new__` swaps in the out-of-tree class by looking up
# `cls.__name__`, so a subclass of the in-tree base is *not* redirected: it
# would build the CUDA `MLAAttention`, which on TPU dies importing the
# FlashAttention MLA prefill backend.
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMultiHeadLatentAttentionWrapper,
)
from vllm_torchtpu.models.vllm.glm5_next.indexer import Glm5NextIndexer
from vllm_torchtpu.models.vllm.glm5_next.sp import sp_all_heads
from vllm_torchtpu.models.vllm.kimi_k3.attention import (
    KimiDeltaAttention,
    kda_state_dtype,
)

Glm5NextLinearAttention = KimiDeltaAttention
glm5_next_kda_state_dtype = kda_state_dtype

# TP=1 does not fit. The GDN v3 fused KDA kernel keeps a per-sequence tile of
# `tile_size x 3 * heads * head_dim` activations plus the `heads x 128 x 128`
# recurrent state, both double-buffered and both upcast to fp32 inside
# `fused_conv1d_gdn` (`qkv = qkv.astype(jnp.float32)` -- the bf16 path is a
# TODO there). Unsharded that is 64 heads: 64*3*64*128*4*2 = 12.6 MB of qkv
# plus 8.4 MB of state, and the kernel overruns the 63.94 MB VMEM budget by
# ~4 MB before the register allocator's spill slots are counted. Sharding the
# heads is what makes it fit, so this model needs TP >= 2; TP=8 (8 local
# heads) is what it is actually run at.

# Width of the synthetic all-zero rope lane. 64 is what every other MLA model
# on this backend uses, so the paged-MLA and (later) sparse-MLA kernels are
# already tuned for it; the sparse wrapper additionally asserts
# `rope_dim * 2 == lane_bytes`, which only holds at 64.
ROPE_LANE = 64


class _ZeroRopeMLAWrapper(VllmTPUMultiHeadLatentAttentionWrapper):
    """MLA wrapper for a checkpoint with no rope dimensions at all.

    The base wrapper derives every split from ``qk_rope_head_dim``. We hand
    the *padded* width down to the attention kernel and KV cache (via
    ``super().__init__``) but the projections upstream are built at GLM's true
    widths, so the base ``forward`` would split them wrongly. This overrides
    ``forward`` to do GLM's split and feed the kernel zeros for the lane.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bypass `nn.Module.__setattr__` so the indexer is *not* registered as
        # a child here. It is already a child of `Glm5NextMLA` under the name
        # the checkpoint uses (`self_attn.indexer.*`); registering it a second
        # time would give its parameters a second, longer name and make the
        # `Glm5NextIndexerCache` prefix ambiguous.
        mla_modules = kwargs.get("mla_modules")
        if mla_modules is None:
            mla_modules = next(a for a in args if isinstance(a, MLAModules))
        object.__setattr__(
            self, "glm_indexer", getattr(mla_modules, "glm_indexer", None)
        )
        # Replicated twins of q_b_proj / o_proj for sequence-parallel steps;
        # owned (registered) by `Glm5NextMLA`, like the indexer.
        object.__setattr__(
            self, "q_b_proj_sp", getattr(mla_modules, "q_b_proj_sp", None)
        )
        object.__setattr__(self, "o_proj_sp", getattr(mla_modules, "o_proj_sp", None))
        if self.q_b_proj_sp is not None:
            self.mla_attn.build_full_heads = True
        # Every k_pe this layer writes is zero and the cache starts zeroed,
        # so the sparse op may skip the rope-cache scatter (~1 ms a layer for
        # an 8K chunk; it is an XLA scatter on the TensorCore layout).
        self.mla_attn.zero_rope = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        prefill_group=None,
    ) -> torch.Tensor:
        """`prefill_group` set: a sequence-parallel step (see `sp.py`), so
        `hidden_states` is this rank's token shard and so is the result."""
        if self.q_b_proj_sp is not None:
            return self._forward_all_heads(hidden_states, prefill_group)
        if prefill_group is not None:
            hidden_states = prefill_group.all_gather(hidden_states, dim=0)
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank], dim=-1)
        q_c = self.q_a_layernorm(q_c)

        # The indexer scores the *post-layernorm* q LoRA residual, the same
        # `q_resid` the reference passes, and it must run before the attention
        # call: the sparse kernel inserts this step's KV and then gathers, so
        # the candidate list has to already exist.
        topk_indices = None
        if self.glm_indexer is not None:
            topk_indices = self.glm_indexer(hidden_states, q_c)[
                : hidden_states.shape[0]
            ]

        q = self.q_b_proj(q_c)[0]
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q_nope = q.view(-1, self.num_heads, self.qk_nope_head_dim)
        if llama_4_scaling is not None:
            q_nope = q_nope * llama_4_scaling

        # Slice-then-zero rather than `new_zeros(...)`: the token count is a
        # symbolic dim under torch.compile, and torch-tpu's exporter rejects
        # factory ops whose SymIntArrayRef is not all-concrete. Slicing an
        # existing tensor carries the symbol through instead of re-stating it.
        q_pe = torch.zeros_like(q_nope[..., :ROPE_LANE])
        k_pe = torch.zeros_like(kv_c_normed[..., :ROPE_LANE]).unsqueeze(1)

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            topk_indices=topk_indices,
        )
        out = self.o_proj(attn_out)[0]
        if prefill_group is not None:
            return prefill_group.reduce_scatter(out, dim=0)
        return tensor_model_parallel_all_reduce(out)

    def _forward_all_heads(
        self, hidden_states: torch.Tensor, prefill_group
    ) -> torch.Tensor:
        """Attend with every head, from the replicated twins.

        With `prefill_group` (a sequence-parallel step) `hidden_states` is
        this rank's token shard: every per-query projection -- q LoRA, q_b,
        the absorbed W_UK/W_UV, o_proj -- runs on the shard, so the output is
        already this rank's rows; no all-to-all and no reduce-scatter. Only
        what every token's attention reads is gathered: the KV latent and the
        indexer's key/gate rows, 768 columns instead of the 4096-wide hidden
        state.

        Without it (decode buckets) every rank does the same for every token.
        That reads the full q_b/o_proj (~0.4 ms a step over the TP shards) and
        drops the all-reduce, but above all both cases are one code path over
        one set of weights, read in one order: the runner hands compiled
        buckets between the two traces only if they take identical inputs
        (`compilation.shape_variants`), and anything that reads the other
        path's weights in the decode graph only to satisfy that makes XLA
        re-tile those weights every step.
        """
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank], dim=-1)
        q_c = self.q_a_layernorm(q_c)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_idx, gate_idx = self.glm_indexer.key_inputs(hidden_states)
        if prefill_group is not None:
            gathered = prefill_group.all_gather(
                torch.cat([kv_c_normed.to(k_idx.dtype), k_idx, gate_idx], dim=-1), dim=0
            )
            kv_c_normed, k_idx, gate_idx = gathered.split(
                [self.kv_lora_rank, k_idx.shape[-1], gate_idx.shape[-1]], dim=-1
            )
            kv_c_normed = kv_c_normed.contiguous()
        topk_indices = self.glm_indexer.forward_given_keys(
            hidden_states,
            q_c,
            k_idx,
            gate_idx,
            sequence_parallel=prefill_group is not None,
        )

        q_nope = self.q_b_proj_sp(q_c)[0].view(
            -1,
            self.q_b_proj_sp.output_size // self.qk_nope_head_dim,
            self.qk_nope_head_dim,
        )
        attn = self.mla_attn
        attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
            # Memory probe before the caches are bound; see the backend.
            out = torch.zeros_like(
                q_nope.flatten(1)[:, :1].expand(-1, q_nope.shape[1] * self.v_head_dim)
            )
        else:
            out = attn.impl.forward_sparse_all_heads(
                attn,
                q_nope,
                kv_c_normed,
                torch.zeros_like(kv_c_normed[..., :ROPE_LANE]),
                kv_cache,
                attn_metadata,
                topk_indices,
                sequence_parallel=prefill_group is not None,
            )
        return self.o_proj_sp(out)[0]


class Glm5NextMLA(nn.Module):
    """GLM-5.3-Flash multi-head latent attention (NoPE, dense for now)."""

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not getattr(config, "mla_use_nope", False):
            raise ValueError("GLM-5.3-Flash MLA is NoPE-only")
        if int(config.qk_rope_head_dim) != 0:
            raise ValueError(
                "Glm5NextMLA assumes the released qk_rope_head_dim=0 layout; "
                f"got {config.qk_rope_head_dim}. A checkpoint with real rope "
                "dimensions needs the rotary path, not this zero lane."
            )
        if config.q_lora_rank is None:
            raise ValueError("GLM-5.3-Flash MLA always uses a q LoRA")

        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        num_heads = config.num_attention_heads // tp_size
        quant_config = vllm_config.quant_config

        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.v_head_dim = int(config.v_head_dim)

        # Registered before `mla_attn` so that `named_parameters` -- which
        # keeps the first name it sees for a shared module -- reports these as
        # `self_attn.indexer.*`, matching the checkpoint.
        self.indexer = None
        if topk_indices_buffer is not None:
            self.indexer = Glm5NextIndexer(
                config,
                vllm_config,
                vllm_config.cache_config,
                topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            config.num_attention_heads * self.qk_nope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            # Reduced by `_ZeroRopeMLAWrapper.forward`: all-reduce, or
            # reduce-scatter in a sequence-parallel step.
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )

        # Sequence-parallel steps project q and the output for this rank's
        # tokens with every head, so they need full copies of q_b_proj and
        # o_proj (~96 MB a layer in fp8). Same prefixes as the TP layers, so
        # the quantization config treats them alike; `load_weights` feeds
        # both from the same checkpoint tensors.
        q_b_proj_sp = o_proj_sp = None
        if self.indexer is not None and sp_all_heads():
            q_b_proj_sp = ReplicatedLinear(
                self.q_lora_rank,
                config.num_attention_heads * self.qk_nope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            o_proj_sp = ReplicatedLinear(
                config.num_attention_heads * self.v_head_dim,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        self.q_b_proj_sp = q_b_proj_sp
        self.o_proj_sp = o_proj_sp

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            # `indexer=` is deliberately left empty: the base wrapper would run
            # `VllmTPUIndexer.rebind` on it, which retypes a DeepSeek `Indexer`
            # in place and asserts on attributes GLM's indexer does not have.
            # `_ZeroRopeMLAWrapper` picks ours up from `glm_indexer` instead.
            indexer=None,
            is_sparse=self.indexer is not None,
            topk_indices_buffer=topk_indices_buffer,
        )
        mla_modules.g_proj = None
        mla_modules.gate_is_fused = False
        mla_modules.glm_indexer = self.indexer
        mla_modules.q_b_proj_sp = q_b_proj_sp
        mla_modules.o_proj_sp = o_proj_sp
        self.prefill_group = None  # set by Glm5NextModel when SP is on
        self.mla_attn = _ZeroRopeMLAWrapper(
            config.hidden_size,
            num_heads,
            # Scale over the *real* head dim: the padded lane is all zeros and
            # must not dilute the softmax temperature.
            self.qk_nope_head_dim**-0.5,
            self.qk_nope_head_dim,
            ROPE_LANE,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            vllm_config.cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # The two paths read different weights (sharded vs the replicated
        # twins), so their traces take different inputs; the runner keeps
        # each bucket with the trace that serves it (`shape_variants`).
        return self.mla_attn(
            positions,
            hidden_states,
            prefill_group=self.prefill_group if sequence_parallel else None,
        )


__all__ = [
    "Glm5NextLinearAttention",
    "Glm5NextMLA",
    "ROPE_LANE",
    "glm5_next_kda_state_dtype",
]
