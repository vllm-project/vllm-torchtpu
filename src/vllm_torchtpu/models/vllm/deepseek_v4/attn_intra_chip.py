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

Under DP=16/TP=1 every core holds all of `wq_b`, `wo_a` and `wo_b`, roughly
285 MB per layer. `VllmDeepseekV4MLAAttentionIntraChip` splits those three
across the two cores of a chip: each core keeps half the heads of `wq_b` and
half the o-groups of `wo_a`/`wo_b`. vLLM still sees DP=16/TP=1, so the
scheduler, router, KV pool and the sparse-attention kernels are unchanged and
run over all 128 heads for this rank's own sequences.

Per layer, inside the chip pair (tokens are padded to the same bucket on every
DP rank, so shapes match):
  1. all-gather `qr` -> q for my 64 heads over both cores' tokens
  2. swap halves -> q for all 128 heads over my tokens; attention runs locally
  3. swap halves back -> o for my 64 heads over both cores' tokens
  4. fused inverse-RoPE + `wo_a` on my 8 groups, `wo_b` on my 8192 rows
  5. reduce-scatter -> my tokens' full output

Every collective is issued here, over the chip-pair group; none are left to
the compiler. Experimental: `DeepseekV4DecoderLayer` uses this class when
TPU_DSV4_ATTN_INTRA_CHIP_TP=1.
"""

from __future__ import annotations

import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.attention import VllmDeepseekV4MLAAttention

logger = init_logger(__name__)

_group: parallel_state.GroupCoordinator | None = None


def _pair_groups() -> list[list[int]]:
    """Global ranks grouped by physical chip, from each rank's device id."""
    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.mesh_utils import (
        _collect_rank_to_device_id,
        _get_current_global_rank,
        _get_tpu_global_device_id,
    )

    world = parallel_state.get_world_group()
    rank_to_device = _collect_rank_to_device_id(
        world, _get_current_global_rank(), _get_tpu_global_device_id()
    )
    topology = None
    try:
        topology = get_chip_topology()
    except Exception as error:  # noqa: BLE001 - optional capability probe
        logger.warning("ATTN_INTRA_CHIP_TP: chip topology unavailable: %s", error)
    if topology is not None and all(
        any(d in chip for chip in topology.chips) for d in rank_to_device.values()
    ):
        chip_of_device = {d: i for i, chip in enumerate(topology.chips) for d in chip}
        source = "coords"
    else:
        # tpu7x numbers a chip's two TensorCores 2k and 2k+1.
        chip_of_device = {d: d // 2 for d in rank_to_device.values()}
        source = "device_id // 2"
    by_chip: dict[int, list[int]] = {}
    for rank, device in rank_to_device.items():
        by_chip.setdefault(chip_of_device[device], []).append(rank)
    groups = [sorted(by_chip[c]) for c in sorted(by_chip)]
    logger.info(
        "ATTN_INTRA_CHIP_TP: chip pairs from %s: %s (rank->device %s)",
        source,
        groups,
        dict(sorted(rank_to_device.items())),
    )
    if any(len(g) != 2 for g in groups):
        raise RuntimeError(f"ATTN_INTRA_CHIP_TP needs 2 cores per chip, got {groups}")
    return groups


def get_chip_pair_group() -> parallel_state.GroupCoordinator:
    """The GroupCoordinator over this rank's chip pair. Collective on first call."""
    global _group
    if _group is None:
        import torch.distributed as dist

        world = parallel_state.get_world_group()
        _group = parallel_state.init_model_parallel_group(
            _pair_groups(),
            world.local_rank,
            dist.get_backend(world.device_group),
            group_name="attn_intra_chip",
        )
    return _group


def _use_my_slice(layer: torch.nn.Module, rank: int, size: int) -> None:
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


def _swap_with_peer(group, mine: torch.Tensor, theirs: torch.Tensor):
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
        group = get_chip_pair_group()
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
        # RowParallelLinear takes no tp_rank/tp_size, so it is built at my
        # half of the input rows with TP off and pointed at its slice below.
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
            _use_my_slice(layer, rank, size)
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
        # q for my heads over [core0 tokens; core1 tokens].
        q_half = self.wq_b(group.all_gather(qr.contiguous(), dim=0))
        q_half = q_half.view(2, num_tokens, -1)
        # My tokens with my heads stay; my tokens with the peer's heads arrive.
        peer = _swap_with_peer(group, q_half[rank], q_half[1 - rank])
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
        # Peer's tokens with my heads arrive; my tokens with my heads stay.
        peer = _swap_with_peer(group, o[:, rank], o[:, 1 - rank])
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
