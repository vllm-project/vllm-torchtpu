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
"""GLM-5.2 (``glm_moe_dsa``): serve expert copies on the grouped-matmul path.

A ``TPU_EXPERT_PLACEMENT`` table with copies loads one extra copy of some
experts on other ranks. For GLM-5.2, a placed layer with copy slots lists
``copy_share`` -- per copied expert, the share of its (token, k) pairs its
copy serves -- and sends that share of its pairs to the copy:

* Right after routing, each router-row id becomes a slot id
  ``rank * slots + slot``: the primary's, or the copy's when a hash of the
  pair's position falls below the expert's share. Every EP rank routes the
  same all-gathered rows, so all ranks pick the same holder for each pair and
  the reduce-scatter sums each pair once. No table lookup per pair: the
  primary's slot follows from the contiguous ownership, and a copy is found
  by comparing against the few copied experts (gathers into a small table
  cost milliseconds at prefill sizes on TPU).
* ``experts_start`` becomes ``rank * slots``, so the grouped path's remap
  serves exactly the chosen holder. The kernels are unchanged.
* Token buckets under the table's ``copy_min_tokens`` tokens per rank of the
  MoE all-gather (its DP x PCP ranks) keep every pair on its primary: there
  each expert has few rows, so splitting them makes both holders stream its
  weights for little compute. Each side of that fork is its own executable
  (``compilation.shape_variants``).

``copy_share`` and ``copy_min_tokens`` are read and validated with the rest of
the table (``expert_placement.parse``), from the same bytes.
"""

from typing import TYPE_CHECKING, Optional

import torch

from vllm_torchtpu import envs, expert_placement
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)

# Shares are fixed point over the 15-bit pair hash below.
SHARE_ONE = 1 << 15
_HASH_MASK = SHARE_ONE - 1
# Odd, so the hash is a bijection on each block of 2**15 positions; the block
# term offsets successive blocks. Products stay below 2**31 in int32.
_HASH_MUL = 0x4F1B
_HASH_BLOCK_MUL = 0x1B3
_HASH_BLOCK_SHIFT = 15

# Set once the served model is GLM-5.2; only then may a layer serve copies.
_glm_moe_dsa = False


def _is_glm_moe_dsa(model_config: Optional["ModelConfig"]) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "glm_moe_dsa"


def maybe_patch_glm_moe_dsa(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Let GLM-5.2's placed MoE layers serve the copies listed with shares."""
    global _glm_moe_dsa
    if not _is_glm_moe_dsa(model_config):
        return False
    _glm_moe_dsa = True
    return None


def copy_tables(
    table: expert_placement.Placement, layer: int, shares: dict[int, float]
) -> tuple[list[int], list[int], list[int]]:
    """The copied router-row ids in increasing order, each with its copy's slot
    id and its share; route_to_copies derives the primaries' slot ids.

    Slot ids are ``rank * table.slots + slot``. ``shares`` list exactly the
    layer's copied experts, each in [0, 1] (``parse`` checks).
    """
    order = expert_placement.virtual_order(table, layer)
    virtual = {e: v for v, e in enumerate(order)}
    copy = {
        virtual[held[slot]]: rank * table.slots + slot
        for rank, held in enumerate(table.cores[layer])
        for slot in range(table.owned, table.slots)
    }
    rows = sorted(copy)
    share = [round(shares[order[v]] * SHARE_ONE) for v in rows]
    return rows, [copy[v] for v in rows], share


def install_copy_routing(layer: "RoutedExperts") -> bool:
    """Register a placed layer's copy routing buffers; False without shares.

    Called once the layer's weights are loaded. The slot tables are the same
    on every rank; the rank's first slot id is runtime data, like
    ``_experts_start``, so every rank compiles the same program.
    """
    parallel = layer.moe_config.moe_parallel_config
    table = expert_placement.placement() if parallel.use_ep else None
    if table is None:
        return False
    layer_idx = expert_placement.layer_index(layer.layer_name)
    if layer_idx is None or layer_idx not in table.shares:
        return False
    if not _glm_moe_dsa:
        raise NotImplementedError(
            f"{layer.layer_name}: expert copies in TPU_EXPERT_PLACEMENT are "
            "served for GLM-5.2 (glm_moe_dsa) only. Use a table without copies."
        )
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )

    # apply_monolithic's owns_collectives: those paths route each DP rank's
    # own rows, while the holder choice and the copy_min_tokens gate are
    # defined on the all-gathered rows.
    if enable_pipelined_collective_and_compute() or envs.TPU_MOE_FP8_ALL_GATHER:
        raise NotImplementedError(
            f"{layer.layer_name}: copy_share needs the routing of the "
            "all-gathered rows; unset TPU_MOE_COLLECTION_CHUNK_SIZE and "
            "TPU_MOE_FP8_ALL_GATHER, or use a table without copies."
        )
    rows, copy, share = copy_tables(table, layer_idx, table.shares[layer_idx])
    layer._copy_owned, layer._copy_slots = table.owned, table.slots
    device = layer.w13_weight.device
    for name, values in (
        ("_copy_rows", rows),
        ("_copy_slot", copy),
        ("_copy_share", share),
    ):
        layer.register_buffer(
            name,
            torch.tensor(values, dtype=torch.int32, device=device),
            persistent=False,
        )
    layer.register_buffer(
        "_copy_slots_start",
        torch.tensor(parallel.ep_rank * table.slots, dtype=torch.int32, device=device),
        persistent=False,
    )
    # The routed rows are gathered over the DP and PCP ranks (MoERunner). Only
    # target-model layers are placed (MTP layers are refused or skipped), and
    # the backbone is re-traced per token-bucket group, so each side of the
    # gate in route_to_copies is its own executable.
    min_tokens = table.copy_min_tokens
    layer._copy_min_rows = min_tokens * parallel.dp_size * parallel.pcp_size
    logger.info_once(
        "GLM-5.2 expert copies: served on the grouped-matmul path from %d "
        "tokens per DP x PCP rank (%d rows)",
        min_tokens,
        layer._copy_min_rows,
    )
    return True


def route_to_copies(layer: "RoutedExperts", topk_ids: torch.Tensor) -> torch.Tensor:
    """Router-row ids -> slot ids, each (token, k) pair on one holder."""
    topk_ids = topk_ids.to(torch.int32)
    # Router row v is rank v // owned's slot v % owned.
    owned, slots = layer._copy_owned, layer._copy_slots
    primary = (topk_ids // owned) * slots + topk_ids % owned
    rows, copy, share = layer._copy_rows, layer._copy_slot, layer._copy_share
    if topk_ids.shape[0] < layer._copy_min_rows:
        # Both sides of the gate lift the same buffers, so a bucket's
        # executable can be handed between the traces that fork here
        # (compilation.shape_variants). The added zero folds away.
        zero = (rows[:1] + copy[:1] + share[:1]) * 0
        return primary + zero
    # Each router row is copied at most once: max picks that copy's entry,
    # and 0 (share 0, so never taken) for an expert without one.
    hit = topk_ids.unsqueeze(-1) == rows
    pair_copy = torch.where(hit, copy, 0).amax(dim=-1)
    pair_share = torch.where(hit, share, 0).amax(dim=-1)
    position = torch.arange(
        topk_ids.numel(), dtype=torch.int32, device=topk_ids.device
    ).view(topk_ids.shape)
    hashed = (
        (position & _HASH_MASK) * _HASH_MUL
        + (position >> _HASH_BLOCK_SHIFT) * _HASH_BLOCK_MUL
    ) & _HASH_MASK
    return torch.where(hashed < pair_share, pair_copy, primary)
