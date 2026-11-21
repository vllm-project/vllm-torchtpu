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
"""TPU_MOE_EXPERT_COUNT_TAP_DIR: per-layer routed-expert counts from a live server.

A fused kernel selects experts inside its own program, so its choices never
reach the host. With the tap on, every MoE layer re-runs the selection the TPU
MoE methods make (`moe_routing.route`) in the traced forward and accumulates on
the device, per global expert: row 0 the sum over steps of c, row 1 of c * c,
row 2 [steps, valid tokens, 0, ...], where c is this rank's per-step count of
valid tokens that selected the expert (the mean and variance of its load).

Every TPU_MOE_EXPERT_COUNT_TAP_EVERY_STEPS runner steps the f32 device buffers
are drained into host-side running totals, so no sum overflows, and each EP
rank writes the totals to <dir>/counts_ep<rank>_<seq>.npz (counts [layers, 3,
experts] i64, layer_names, layer_indices, step, wall). A restarted process
numbers its snapshots after the existing ones, and its `step` starts over.
Each token is counted once over the ranks: only TP rank 0 counts the rows a TP
group replicates, and a rank counts only its own share of rows the runner
gathered over DP or PCP. Experts are numbered as in the checkpoint, also in a
placed layer; `layer_indices` numbers the layers as a TPU_EXPERT_PLACEMENT
table does (`expert_placement.layer_index`), or -1.

Padding rows are told apart by the runner's token padding mask, so the tap is
refused unless that mask is kept and exact (`require_padding_mask`), and with
pipeline parallelism, whose stages share EP ranks. The outputs are unchanged,
but the forward does extra work: a tapped run is a diagnostic, not a benchmark.
"""

import glob
import os
import re
import time

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import expert_placement, moe_routing, token_padding
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

COUNTS_ATTR = "_tpu_expert_counts"
# A tapped layer's [DP rank, PCP rank]: its share of rows gathered over them.
RANKS_ATTR = "_tpu_expert_count_ranks"

# (name, layer, slot order or None, table layer or -1) per tapped layer.
_TAPPED: list[tuple[str, torch.nn.Module, tuple[int, ...] | None, int]] = []


def _fresh_state() -> dict:
    # Runner steps, next snapshot number, EP rank, f64 running totals.
    return {"steps": 0, "seq": None, "rank": None, "total": None}


_STATE = _fresh_state()


def enabled() -> bool:
    return bool(envs.TPU_MOE_EXPERT_COUNT_TAP_DIR)


def _tp_rank() -> int:
    from vllm.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank()


def _pp_size() -> int:
    from vllm.distributed import get_pp_group

    return get_pp_group().world_size


def reset_model_state() -> None:
    """Forget the tapped layers and the totals, for another model load."""
    _TAPPED.clear()
    _STATE.clear()
    _STATE.update(_fresh_state())


def require_padding_mask(tracked: bool, all_layout: bool) -> None:
    """Refuse the tap unless the runner keeps a token padding mask (`tracked`)
    and it is exact, which it is only in the ALL sequence layout."""
    if not enabled():
        return
    if not tracked:
        raise ValueError(
            "TPU_MOE_EXPERT_COUNT_TAP_DIR counts only real tokens, told apart "
            "by the token padding mask: set TPU_MOE_SKIP_PADDED_TOKENS=1"
        )
    if not all_layout:
        raise ValueError(
            "TPU_MOE_EXPERT_COUNT_TAP_DIR needs the ALL sequence layout: under "
            "PCP streaming or the PCP manager the padding mask marks every row "
            "valid"
        )


def _repeats_another_rank(layer) -> bool:
    # The fused kernel's token replica groups are the TP groups unless MoE
    # sequence parallelism splits the rows; only their first member counts.
    if layer.moe_config.moe_parallel_config.is_sequence_parallel:
        return False
    return _tp_rank() != 0


def register(layer) -> None:
    """Give a RoutedExperts layer its count buffer (the TPU MoE runner calls
    this for every layer, after any expert placement)."""
    if not enabled():
        return
    if envs.TPU_MOE_EXPERT_COUNT_TAP_EVERY_STEPS <= 0:
        raise ValueError(
            "TPU_MOE_EXPERT_COUNT_TAP_EVERY_STEPS must be a positive step "
            f"count; got {envs.TPU_MOE_EXPERT_COUNT_TAP_EVERY_STEPS}"
        )
    if _pp_size() > 1:
        raise NotImplementedError(
            "TPU_MOE_EXPERT_COUNT_TAP_DIR does not support pipeline "
            "parallelism: its stages would write the same snapshot files"
        )
    if _repeats_another_rank(layer):
        return
    _STATE["rank"] = layer.expert_map_manager.ep_rank
    weight = next(layer.parameters(), None)
    device = None if weight is None else weight.device
    experts = layer.global_num_experts
    layer.register_buffer(
        COUNTS_ATTR,
        torch.zeros((3, experts), dtype=torch.float32, device=device),
        persistent=False,
    )
    (dp_rank, _), (pcp_rank, _) = _dp_pcp()
    layer.register_buffer(
        RANKS_ATTR,
        torch.tensor([dp_rank, pcp_rank], dtype=torch.int32, device=device),
        persistent=False,
    )
    index = expert_placement.layer_index(layer.layer_name)
    order = expert_placement.expert_order(layer)
    _TAPPED.append((layer.layer_name, layer, order, -1 if index is None else index))


def is_tapped(layer) -> bool:
    return getattr(layer, COUNTS_ATTR, None) is not None


def _dp_pcp() -> tuple[tuple[int, int], tuple[int, int]]:
    """This rank's (rank, size) in the DP group and in the PCP group."""
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.pcp import get_pcp_group

    dp, pcp = get_dp_group(), get_pcp_group()
    return (int(dp.rank_in_group), int(dp.world_size)), (
        (0, 1) if pcp is None else (int(pcp.rank_in_group), int(pcp.world_size))
    )


def _counted_rows(
    layer, rows: int, over_dp: bool, over_pcp: bool, device
) -> torch.Tensor:
    """[rows] bool: this rank's own valid rows. Rows gathered over DP, then
    PCP, are every rank's, in equal shares in group-rank order (as
    `token_padding.zero_routing_weights_for_padding` reads them); the share is
    picked by a position held as data, so every rank traces one program."""
    dp_size = pcp_size = 1
    if over_dp or over_pcp:
        (_, dp_world), (_, pcp_world) = _dp_pcp()
        dp_size = dp_world if over_dp else 1
        pcp_size = pcp_world if over_pcp else 1
    shares = dp_size * pcp_size
    if shares > 1 and layer.moe_config.moe_parallel_config.is_sequence_parallel:
        raise NotImplementedError(
            f"expert count tap: {layer.layer_name} runs sequence-parallel MoE "
            "on rows its runner gathered, which the tap cannot attribute to "
            "ranks"
        )
    local = rows // shares
    state = token_padding._padding_state
    padding = None if state is None else state.get_local_padding_mask(local)
    valid = (
        torch.ones((local,), dtype=torch.bool, device=device)
        if padding is None
        else ~padding
    )
    if shares == 1:
        return valid
    dp_rank, pcp_rank = getattr(layer, RANKS_ATTR).unbind(0)
    position = dp_rank if over_dp else torch.zeros_like(dp_rank)
    if over_pcp:
        position = position + pcp_rank * dp_size
    share = torch.arange(rows, dtype=torch.int32, device=device) // local
    return valid.repeat(shares) & (share == position)


def record(
    layer,
    router_logits: torch.Tensor,
    input_ids,
    *,
    hidden_states: torch.Tensor | None = None,
    over_dp: bool = False,
    over_pcp: bool = False,
) -> None:
    """Add this step's selections to the layer's buffer (traced). `over_dp` /
    `over_pcp`: the runner gathered the rows over that group."""
    counts = getattr(layer, COUNTS_ATTR, None)
    if counts is None:
        return
    # route()'s selection without its capture hook, which the quant method's
    # own routing fires once per layer.
    _, topk_ids = moe_routing._select_routing(
        layer,
        router_logits if hidden_states is None else hidden_states,
        router_logits,
        layer.moe_config.experts_per_token,
        input_ids,
    )
    experts = counts.shape[1]
    device = router_logits.device
    valid = _counted_rows(layer, router_logits.shape[0], over_dp, over_pcp, device)
    hit = topk_ids.to(torch.int32).unsqueeze(-1) == torch.arange(
        experts, dtype=torch.int32, device=device
    )
    # Exact in f32: a step's count, and its square up to 4096.
    per_step = (hit & valid[:, None, None]).sum(dim=(0, 1), dtype=torch.float32)
    meta = torch.cat(
        [
            torch.ones((1,), dtype=torch.float32, device=device),
            valid.sum(dtype=torch.float32).reshape(1),
            torch.zeros((experts - 2,), dtype=torch.float32, device=device),
        ]
    )
    counts.copy_(counts + torch.stack([per_step, per_step * per_step, meta]))


def _first_seq(directory: str, rank: int) -> int:
    """One past the last snapshot this rank already has in `directory`."""
    seqs = [
        int(m.group(1))
        for path in glob.glob(os.path.join(directory, f"counts_ep{rank}_*.npz"))
        if (m := re.search(rf"counts_ep{rank}_(\d+)\.npz$", path))
    ]
    if seqs:
        logger.warning(
            "Expert count tap: numbering after EP rank %d's earlier snapshots in %s",
            rank,
            directory,
        )
    return max(seqs) + 1 if seqs else 0


def maybe_dump() -> None:
    """Called once per runner step on the host; writes every N steps."""
    if not _TAPPED:
        return
    _STATE["steps"] += 1
    if _STATE["steps"] % envs.TPU_MOE_EXPERT_COUNT_TAP_EVERY_STEPS:
        return
    import numpy as np

    buffers = [getattr(layer, COUNTS_ATTR) for _, layer, _, _ in _TAPPED]
    interval = torch.stack(buffers).cpu().double().numpy()
    for buffer in buffers:
        buffer.zero_()
    for counts, (_, _, order, _) in zip(interval, _TAPPED, strict=True):
        if order is not None:
            # Slot s holds expert order[s]; row 2 is not per expert.
            counts[:2, list(order)] = counts[:2].copy()
    total = _STATE["total"]
    _STATE["total"] = interval if total is None else total + interval
    directory = envs.TPU_MOE_EXPERT_COUNT_TAP_DIR
    os.makedirs(directory, exist_ok=True)
    rank = _STATE["rank"]
    if _STATE["seq"] is None:
        _STATE["seq"] = _first_seq(directory, rank)
    path = os.path.join(directory, f"counts_ep{rank}_{_STATE['seq']:04d}.npz")
    # Written aside and renamed, so a reader never opens half a snapshot.
    with open(path + ".tmp", "wb") as f:
        np.savez(
            f,
            counts=np.rint(_STATE["total"]).astype(np.int64),
            layer_names=np.array([name for name, _, _, _ in _TAPPED]),
            layer_indices=np.array([i for _, _, _, i in _TAPPED], dtype=np.int64),
            step=_STATE["steps"],
            wall=time.time(),
        )
    os.replace(path + ".tmp", path)
    _STATE["seq"] += 1
    logger.info_once("Expert count tap writing snapshots to %s", directory)
