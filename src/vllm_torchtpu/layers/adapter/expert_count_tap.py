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
"""MOE_EXPERT_COUNT_TAP_DIR: per-layer routed-expert counts from a live server.

An expert placement is only as good as the routing statistics it is built
from, and a fused kernel selects inside its own program, so its choices never
reach the host. With the tap on, every MoE layer, whatever its kernel, re-runs
the same selection in the traced forward (`moe_routing.select_experts`: hash
table or scored top-k) and accumulates, on the device, per global expert:

    row 0: routed pairs           sum over steps of c
    row 1: squared per-step count sum over steps of c * c
    row 2: [steps, valid tokens, 0, ...]

where c is this rank's per-step count of valid (non-padding) tokens that
selected the expert. Rows 0-1 give the mean and the step-to-step variance of
each expert's load from this rank.

The device buffers are f32 and only ever hold the steps since the last
snapshot: every MOE_EXPERT_COUNT_TAP_EVERY steps the runner copies them to the
host, adds them to this process's running totals and zeroes them, so no sum
outgrows its type however long the server runs (an int32 sum of squares of a
hot expert's count wraps within the hour at 1024-token steps). The snapshot is
the running total, never overwritten:

    <dir>/counts_ep<rank>_<seq>.npz   counts [layers, 3, experts] i64,
                                      layer_names, step, wall

<rank> is the layer's EP rank. Each token is counted once over the ranks'
snapshots: where the MoE input is replicated over a TP group (the fused
kernel's token replica groups), only TP rank 0 records, and where the MoE
runner has gathered the rows over DP or PCP, each rank counts only its own.
Experts are numbered as in the checkpoint, also in a layer an expert placement
has renumbered, so a placement built from the counts names the same experts.
A restarted process numbers its snapshots after the ones already in <dir>;
its `step` starts over, which is how a reader tells the runs apart.

Snapshots are cumulative so a reader can difference any two of them to drop
warmup and compile traffic. The tap only reads the routing inputs: the model's
outputs are unchanged, but the forward does extra work, so a tapped run is a
diagnostic, not a benchmark arm.
"""

import glob
import os
import re
import time

import torch

from vllm_torchtpu import envs, expert_placement
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

COUNTS_ATTR = "_tpu_expert_counts"
# A tapped layer's [DP rank, PCP rank], to find this rank's own rows among
# rows its runner gathered.
RANKS_ATTR = "_tpu_expert_count_ranks"

# (layer name, layer, slot order or None) for every tapped layer of this
# process, in registration order. The layers own the buffers; this only lists
# them for the dump, with the order that maps a placed layer's slots back to
# the checkpoint's experts.
_TAPPED: list[tuple[str, torch.nn.Module, tuple[int, ...] | None]] = []


def _fresh_state() -> dict:
    # steps: runner steps seen; seq: the next snapshot's number (decided at
    # the first dump); rank: the EP rank the snapshots are named for; total:
    # the f64 running totals [layers, 3, experts] the snapshots hold.
    return {"steps": 0, "seq": None, "rank": None, "total": None}


_STATE = _fresh_state()


def enabled() -> bool:
    return bool(envs.MOE_EXPERT_COUNT_TAP_DIR)


def _directory() -> str:
    # Pod specs pass env values verbatim, e.g. "$CDK_OUTPUT_DIR/moe_counts".
    directory = os.path.expandvars(envs.MOE_EXPERT_COUNT_TAP_DIR)
    if "$" in directory:
        raise ValueError(
            f"MOE_EXPERT_COUNT_TAP_DIR={envs.MOE_EXPERT_COUNT_TAP_DIR!r} references "
            "an unset variable"
        )
    return directory


def _tp_rank() -> int:
    from vllm.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank()


def _repeats_another_rank(layer) -> bool:
    """Whether this rank's MoE rows are a copy of another rank's.

    The fused kernel's token replica groups are the TP groups unless MoE
    sequence parallelism partitions the rows (see
    `ep_mesh.ep_token_replica_groups`); every member sees the whole group's
    rows, so only the first one counts them.
    """
    if layer.moe_config.moe_parallel_config.is_sequence_parallel:
        return False
    return _tp_rank() != 0


def register(layer) -> None:
    """Give a RoutedExperts layer its count buffer. The TPU MoE runner calls
    this for every layer it builds, after any expert placement is applied."""
    if not enabled():
        return
    _directory()  # an unresolvable directory fails at load, not hours later
    if envs.MOE_EXPERT_COUNT_TAP_EVERY <= 0:
        raise ValueError(
            "MOE_EXPERT_COUNT_TAP_EVERY must be a positive step count; got "
            f"{envs.MOE_EXPERT_COUNT_TAP_EVERY}"
        )
    if _repeats_another_rank(layer):
        return
    _STATE["rank"] = layer.expert_map_manager.ep_rank
    experts = layer.global_num_experts
    weight = next(layer.parameters(), None)
    device = None if weight is None else weight.device
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
    _TAPPED.append((layer.layer_name, layer, expert_placement.expert_order(layer)))


def is_tapped(layer) -> bool:
    return getattr(layer, COUNTS_ATTR, None) is not None


def _dp_pcp() -> tuple[tuple[int, int], tuple[int, int]]:
    """This rank's (rank, size) in the DP group and in the PCP group, the
    groups a MoE runner can gather its rows over."""
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.pcp import get_pcp_group

    dp, pcp = get_dp_group(), get_pcp_group()
    return (int(dp.rank_in_group), int(dp.world_size)), (
        (0, 1) if pcp is None else (int(pcp.rank_in_group), int(pcp.world_size))
    )


def _counted_rows(
    layer, rows: int, over_dp: bool, over_pcp: bool, device
) -> torch.Tensor:
    """[rows] bool: the rows this rank counts, its own tokens' valid ones.

    Rows the runner gathered over DP, then PCP, are every rank's, in equal
    shares in group-rank order (as `token_padding.
    zero_routing_weights_for_padding` reads them too). This rank's share is
    picked by comparing with its position, held as data so that every rank
    traces the same program.
    """
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
    padding = token_padding.local_padding_mask(local)
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
    over_dp: bool = False,
    over_pcp: bool = False,
) -> None:
    """Accumulate this step's selections into the layer's buffer (traced).

    `over_dp` / `over_pcp`: the runner gathered the rows over that group
    before the experts got them, so this rank holds every rank's rows.
    """
    counts = getattr(layer, COUNTS_ATTR, None)
    if counts is None:
        return
    _, topk_ids = moe_routing.select_experts(
        hidden_states=router_logits,
        router_logits=router_logits,
        topk=layer.moe_config.experts_per_token,
        renormalize=layer.renormalize,
        scoring_fn=layer.scoring_func,
        layer=layer,
        input_ids=input_ids,
    )
    experts = counts.shape[1]
    device = router_logits.device
    valid = _counted_rows(layer, router_logits.shape[0], over_dp, over_pcp, device)
    hit = topk_ids.to(torch.int32).unsqueeze(-1) == torch.arange(
        experts, dtype=torch.int32, device=device
    )
    # Counts of at most a step's tokens: exact in f32.
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
    if not seqs:
        return 0
    logger.warning(
        "Expert count tap: %s already holds snapshots of EP rank %d from an "
        "earlier run; numbering this run's after them. Their step count "
        "starts over, which tells the runs apart.",
        directory,
        rank,
    )
    return max(seqs) + 1


def maybe_dump() -> None:
    """Called once per runner step on the host; writes every N steps."""
    if not _TAPPED:
        return
    _STATE["steps"] += 1
    if _STATE["steps"] % envs.MOE_EXPERT_COUNT_TAP_EVERY:
        return
    import numpy as np

    buffers = [getattr(layer, COUNTS_ATTR) for _, layer, _ in _TAPPED]
    interval = torch.stack(buffers).cpu().double().numpy()
    for buffer in buffers:
        buffer.zero_()
    for counts, (_, _, order) in zip(interval, _TAPPED, strict=True):
        if order is not None:
            # A placed layer counts by slot; slot s holds expert order[s].
            # Row 2 is not per expert.
            counts[:2, list(order)] = counts[:2].copy()
    total = _STATE["total"]
    _STATE["total"] = interval if total is None else total + interval
    directory = _directory()
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
            layer_names=np.array([name for name, _, _ in _TAPPED]),
            step=_STATE["steps"],
            wall=time.time(),
        )
    os.replace(path + ".tmp", path)
    _STATE["seq"] += 1
    logger.info_once("Expert count tap writing snapshots to %s", directory)
