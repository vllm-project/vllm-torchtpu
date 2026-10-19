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
"""Intra-chip TP for DSv4 attention weights, with KV and attention kept DP.

`VllmDeepseekV4MLAAttentionIntraChip` splits wq_b, wo_a, and wo_b across the two
chiplets within a chip, only applicable for chips like Ironwood that have two
chiplets per chip. Each chiplet keeps half the heads of `wq_b` and half the
o-groups of `wo_a`/`wo_b`.
To address the weights being sharded, during runtime the chiplets exchange their
activations back and forth via manually inserted collectives.

T0 and T1 are the tokens of chiplet 0 and chiplet 1. Specifically, each chiplet
does the following:
  1. All-gather `qr` with the neighboring chiplet.
     Chiplet 0 has [T0, T1] and Chiplet 1 has [T0, T1]
  2. Run WQ_B projection on gathered values.
     Now each chiplet has q for half of the heads, for both chiplets' tokens.
     Chiplet 0 has [T0_(0-63h), T1_(0-63h)] and Chiplet 1 has
     [T0_(64-127h), T1_(64-127h)].
  3. Swap the tokens post the projection.
     Chiplet 0 has [T0_(0-127h)] and Chiplet 1 has [T1_(0-127h)]
  4. Perform attention operations for the layer, doing (QK_T)V to get output.
  5. Swap the attention output, as `wo_a`/`wo_b` only contain some of the heads.
     Gather positions, since each chiplet applies inverse RoPE to both chiplets'
     tokens.
     Before Swap: Chiplet 0 has [Out0_(0-127h)], Chiplet 1 has [Out1_(0-127h)]
     After Swap: Chiplet 0 has [Out0_(0-63h), Out1_(0-63h)],
     Chiplet 1 has [Out0_(64-127h), Out1_(64-127h)]
  6. Each chiplet performs fused inverse-RoPE + `wo_a` on 8 groups.
     Chiplet 0 has [Out0_(0-7g), Out1_(0-7g)],
     Chiplet 1 has [Out0_(8-15g), Out1_(8-15g)]
     Then each chiplet does `wo_b` projection, resulting in partials.
     Chiplet 0 has [Out0_P0, Out1_P0], Chiplet 1 has [Out0_P1, Out1_P1]
  7. Each chiplet performs reduce-scatter to get final output.
     Chiplet 0 has [Out0_P0 + Out0_P1], Chiplet 1 has [Out1_P0 + Out1_P1]

Experimental: `DeepseekV4DecoderLayer` uses this class when
TPU_DSV4_ATTN_INTRA_CHIP_TP=1.
"""

from __future__ import annotations

import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.attention import VllmDeepseekV4MLAAttention

logger = init_logger(__name__)


def _chip_pair_group() -> parallel_state.GroupCoordinator:
    """The intra-chip group, which must exist and pair exactly two cores."""
    from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group

    group = get_intra_chip_group()
    if group is None or group.world_size != 2:
        raise RuntimeError(
            "TPU_DSV4_ATTN_INTRA_CHIP_TP needs a 2-core intra-chip group, got "
            f"{None if group is None else group.world_size}; see the intra-chip "
            "grouping warning above"
        )
    return group


def _load_current_chiplet_slice(layer: torch.nn.Module, rank: int, size: int) -> None:
    """Make `layer` load slice `rank` of `size` from the full checkpoint tensors.

    vLLM's loaders slice by `tp_rank`/`tp_size`, kept on the layer (unquantized
    weights) and on each quantized parameter (block-fp8 weights and scales).
    Both are read from the global TP group at construction, which is size 1
    here, so they are set to this core's place in the chip pair instead.
    """
    layer.tp_rank, layer.tp_size = rank, size
    for param in layer.parameters():
        if hasattr(param, "tp_rank"):
            param.tp_rank, param.tp_size = rank, size


def _swap_with_peer_chiplet(group, mine: torch.Tensor, theirs: torch.Tensor):
    """Send `theirs` to the peer core; return what the peer sent us.

    A 2-rank all-gather that keeps the peer's half.
    """
    rank = group.rank_in_group
    gathered = group.all_gather(theirs.contiguous(), dim=0)
    return gathered.view(2, *theirs.shape)[1 - rank]


class VllmDeepseekV4MLAAttentionIntraChip(VllmDeepseekV4MLAAttention):
    """DSv4 attention with `wq_b`/`wo_a`/`wo_b` split across a chip's two cores."""

    def __init__(self, vllm_config, prefix: str = "", *args, **kwargs) -> None:
        # The split halves a layer that holds all heads; under vLLM TP>1 the
        # heads are already split, so the halves would be of a shard.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise RuntimeError(
                "TPU_DSV4_ATTN_INTRA_CHIP_TP requires tensor parallel size 1, "
                f"got {tp_size}"
            )
        super().__init__(vllm_config, prefix, *args, **kwargs)
        group = _chip_pair_group()
        rank, size = group.rank_in_group, group.world_size
        quant_config = vllm_config.quant_config
        # Rebuilt half-size; the replicated ones from the base are dropped.
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
            tp_rank=rank,
            tp_size=size,
        )
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
            tp_rank=rank,
            tp_size=size,
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_groups // size
        # RowParallelLinear takes no tp_rank/tp_size, so it is built at current
        # chiplet's half of the input rows with TP off and pointed at its slice
        # below.
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank // size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
            disable_tp=True,
        )
        for layer in (self.wq_b, self.wo_a, self.wo_b):
            _load_current_chiplet_slice(layer, rank, size)
        object.__setattr__(self, "_chip_pair", group)
        logger.info_once(
            "ATTN_INTRA_CHIP_TP: wq_b/wo_a/wo_b sharded across the chip pair"
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # The base reshapes wo_a by the local head and group counts, which are
        # halved for these weights but not for attention, so swap them in here.
        size = self._chip_pair.world_size
        heads, groups = self.n_local_heads, self.n_local_groups
        self.n_local_heads, self.n_local_groups = heads // size, groups // size
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            self.n_local_heads, self.n_local_groups = heads, groups

    def attention_impl(self, hidden_states, qr, kv, indexer_weights, positions, out):
        group = self._chip_pair
        rank = group.rank_in_group
        num_tokens = qr.shape[0]
        q_half = self.wq_b(group.all_gather(qr.contiguous(), dim=0))
        q_half = q_half.view(2, num_tokens, -1)
        peer = _swap_with_peer_chiplet(group, q_half[rank], q_half[1 - rank])
        pieces = [q_half[rank], peer] if rank == 0 else [peer, q_half[rank]]
        q = torch.cat(pieces, dim=-1).view(
            num_tokens, self.mla_attn.n_local_heads, self.mla_attn.head_dim
        )
        q = self.qnorm_rope(q, positions)
        kv = self.kv_rope(kv, positions)

        topk_indices = None
        compressor = getattr(self.mla_attn, "compressor", None)
        if self.indexer is not None:
            indexer_emb = getattr(
                self.mla_attn,
                "indexer_rotary_emb",
                getattr(self.mla_attn, "rotary_emb", None),
            )
            topk_indices = self.indexer(
                hidden_states, qr, indexer_weights, positions, indexer_emb
            )
        if compressor is not None:
            compressor(hidden_states, positions, self.mla_attn.rotary_emb)

        return self.forward_mqa(q, kv, positions, None, topk_indices=topk_indices)

    def _o_proj(self, o, positions):
        group = self._chip_pair
        rank = group.rank_in_group
        num_tokens = o.shape[0]
        half = self.n_local_heads // 2
        o = o.reshape(num_tokens, 2, half, self.head_dim)
        peer = _swap_with_peer_chiplet(group, o[:, rank], o[:, 1 - rank])
        pieces = [o[:, rank], peer] if rank == 0 else [peer, o[:, rank]]
        o_half = torch.cat(pieces, dim=0).contiguous()
        positions_all = group.all_gather(
            positions.to(torch.int32).contiguous(), dim=0
        ).to(positions.dtype)
        z = self.o_proj_op(
            o_half,
            positions_all,
            self.rotary_emb.cos_sin_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        partial = self.wo_b(z)
        if isinstance(partial, tuple):
            partial = partial[0]
        return group.reduce_scatter(partial.contiguous(), dim=0)
