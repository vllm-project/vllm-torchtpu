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
"""Static expert placement from a file (``TPU_MOE_EXPERT_PLACEMENT=<npz>``).

For every MoE layer the file names (see ``layers.core.expert_placement``),
logical expert ``e`` lives in physical slot ``slots[e]``:

- at weight load, the layer's expert map sends each checkpoint expert to its
  slot, so every per-expert tensor (weights and scales) lands there;
- after routing, ``to_slots`` replaces each routed expert id by its slot.

Everything after routing (exchange, capacity, expert kernels) runs in slot
space with linear placement, so it is unchanged. Each route meets the same
expert weights as before; only the grouping of a token's routes into
per-rank partial sums changes, and with it the order of the sum.

With ``TPU_MOE_ROUTING_COUNTS_DIR=<dir>``, the layers this module can place
also count, on device, their routes per logical expert (``count_routes``,
before ``to_slots``, padded token rows left out). The first TP rank of each
DP engine and pipeline stage adds them to host totals and rewrites one
``<dir>/routing-counts-*.npz`` at most once a minute and at a clean
shutdown, the input of ``layers.core.expert_placement``: counts only.

A file with redundant slots gives each rank ``R`` more expert slots per
placed layer, after its primaries (``core.physical_slots``): the expert map
sends the rank's replicated experts there, so the checkpoint loader fills
them too, and ``to_slots`` returns the primaries in that numbering. The
ragged all-to-all exchange serves the replicas (``layers.core.ragged_a2a_moe``);
a boot check compares each replica's prepared bytes with its primary's.
"""

import functools
import os
import socket
import tempfile
import time

import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.core import expert_placement as core
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_HOST_SLOTS = "_tpu_expert_slots_host"
_COUNTED = "_tpu_count_routes"
_COUNTS = "_tpu_routing_counts"
_COUNTS_DP_RANK = "_tpu_routing_counts_dp_rank"
_WRITE_EVERY_S = 60

# Resolved once, at import, outside the compiled forward.
_PLACEMENT: dict[str, np.ndarray] | None = None
_EP_SIZE = 0
_REDUNDANT: dict[str, np.ndarray] = {}
_REDUNDANT_PER_RANK = 0
_placed_layers = 0
_redundant_bytes = 0
if envs.TPU_MOE_EXPERT_PLACEMENT:
    _PLACEMENT, _EP_SIZE = core.load(envs.TPU_MOE_EXPERT_PLACEMENT)
    _REDUNDANT, _REDUNDANT_PER_RANK = core.load_redundant(envs.TPU_MOE_EXPERT_PLACEMENT)
    logger.info(
        "TPU_MOE_EXPERT_PLACEMENT=%s: static expert placement for %d MoE "
        "layers, balanced for EP %d%s",
        envs.TPU_MOE_EXPERT_PLACEMENT,
        len(_PLACEMENT),
        _EP_SIZE,
        f", {_REDUNDANT_PER_RANK} redundant slot(s) per rank" if _REDUNDANT else "",
    )
_COUNTS_DIR = envs.TPU_MOE_ROUTING_COUNTS_DIR
_counted_layers: dict[str, torch.nn.Module] = {}
_num_counted_experts = 0


def _file_slots(layer) -> np.ndarray | None:
    if _PLACEMENT is None:
        return None
    return _PLACEMENT.get(str(getattr(layer, "layer_name", "")))


def host_slots(layer) -> np.ndarray | None:
    """This layer's slots [E] when its experts were placed by the file."""
    return getattr(layer, _HOST_SLOTS, None)


def redundant_slots_per_rank(layer) -> int:
    """Redundant slots per EP rank of a layer the file places (0 without)."""
    return _REDUNDANT_PER_RANK if _file_slots(layer) is not None else 0


def local_experts(layer) -> np.ndarray | None:
    """The logical expert in each of this rank's slots of a placed layer:
    its primaries in slot order, then its replicas (-1 where empty)."""
    slots = host_slots(layer)
    if slots is None:
        return None
    parallel = layer.moe_config.moe_parallel_config
    local = slots.shape[0] // parallel.ep_size
    start = parallel.ep_rank * local
    experts = np.argsort(slots)[start : start + local]
    redundant_per_rank = redundant_slots_per_rank(layer)
    if redundant_per_rank:
        first = parallel.ep_rank * redundant_per_rank
        replicated = _REDUNDANT[layer.layer_name][first : first + redundant_per_rank]
        experts = np.concatenate([experts, replicated])
    return experts.astype(np.int64)


def place_experts(layer) -> None:
    """Point the layer's expert map at the file's slots, before weight load.

    Call from the quant method's ``create_weights``; a no-op for layers the
    file does not name. With TPU_MOE_ROUTING_COUNTS_DIR set, also marks the
    layer for routing counts.
    """
    if _COUNTS_DIR:
        _probe_counts_dir()
        setattr(layer, _COUNTED, True)
    slots = _file_slots(layer)
    if slots is None:
        if _PLACEMENT is not None:
            logger.warning(
                "%s keeps linear expert placement: not in %s",
                layer.layer_name,
                envs.TPU_MOE_EXPERT_PLACEMENT,
            )
        return
    name = layer.layer_name
    parallel = layer.moe_config.moe_parallel_config
    manager = layer.expert_map_manager
    if not parallel.use_ep or manager.expert_map is None:
        raise NotImplementedError(f"{name}: expert placement needs EP")
    if parallel.enable_eplb or manager.placement_strategy != "linear":
        raise NotImplementedError(
            f"{name}: expert placement replaces linear placement; got "
            f"strategy {manager.placement_strategy!r}, EPLB {parallel.enable_eplb}"
        )
    if getattr(manager, "num_fused_shared_experts", 0):
        raise NotImplementedError(f"{name}: fused shared experts are not placed")
    ep_size, num_experts = parallel.ep_size, layer.global_num_experts
    if ep_size != _EP_SIZE or slots.shape[0] != num_experts:
        raise ValueError(
            f"{name}: placement file is for {slots.shape[0]} experts over EP "
            f"{_EP_SIZE}; the layer has {num_experts} over EP {ep_size}"
        )
    local = num_experts // ep_size
    local_index = slots.astype(np.int64) - parallel.ep_rank * local
    expert_map = np.where((local_index >= 0) & (local_index < local), local_index, -1)
    redundant_per_rank = redundant_slots_per_rank(layer)
    if redundant_per_rank:
        if not envs.TPU_MOE_RAGGED_A2A:
            raise NotImplementedError(
                f"{name}: expert replicas ({envs.TPU_MOE_EXPERT_PLACEMENT}) are "
                "served by the ragged all-to-all exchange only "
                "(TPU_MOE_RAGGED_A2A)"
            )
        first = parallel.ep_rank * redundant_per_rank
        for i, e in enumerate(_REDUNDANT[name][first : first + redundant_per_rank]):
            if e >= 0:
                # load_redundant checked that no rank holds an expert twice.
                assert expert_map[e] < 0
                expert_map[e] = local + i
        # update_expert_map_info sets layer.local_num_experts from it.
        manager._local_num_experts = local + redundant_per_rank
    old = manager.expert_map
    manager._expert_map = torch.tensor(expert_map, dtype=old.dtype, device=old.device)
    layer.update_expert_map_info()
    setattr(layer, _HOST_SLOTS, slots)


def register_slots_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._expert_slots`` (None for layers not placed), with
    replicas ``layer._expert_replicas`` (``core.replica_table``, the same on every
    rank), and the routing counts of a layer ``place_experts`` marked."""
    global _placed_layers, _redundant_bytes
    slots = host_slots(layer)
    if slots is None and _file_slots(layer) is not None:
        raise NotImplementedError(
            f"{layer.layer_name}: TPU_MOE_EXPERT_PLACEMENT is implemented for "
            "the NVFP4 grouped-matmul MoE path only"
        )
    buffer = replicas = None
    redundant_per_rank = redundant_slots_per_rank(layer)
    if slots is not None:
        ep_size = layer.moe_config.moe_parallel_config.ep_size
        placed = slots
        if redundant_per_rank:
            placed = core.physical_slots(slots, ep_size, redundant_per_rank)
            table = core.replica_table(slots, _REDUNDANT[layer.layer_name], ep_size)
            replicas = torch.from_numpy(table).to(device)
            _redundant_bytes += redundant_per_rank * sum(
                t.numel() * t.element_size() // t.shape[0]
                for t in _expert_tensors(layer)
            )
        buffer = torch.from_numpy(placed.astype(np.int32)).to(device)
    layer.register_buffer("_expert_slots", buffer, persistent=False)
    layer.register_buffer("_expert_replicas", replicas, persistent=False)
    if slots is not None and _PLACEMENT is not None:
        _placed_layers += 1
        if _placed_layers == len(_PLACEMENT):
            redundant_note = ""
            if _REDUNDANT:
                redundant_note = (
                    f", {_REDUNDANT_PER_RANK} redundant slot(s) per rank "
                    f"(+{_redundant_bytes / 2**30:.2f} GiB per rank)"
                )
            logger.info(
                "Expert placement applied to all %d layers of %s%s",
                _placed_layers,
                envs.TPU_MOE_EXPERT_PLACEMENT,
                redundant_note,
            )
    if getattr(layer, _COUNTED, False):
        _register_counts(layer, device)


def _expert_tensors(layer) -> list[torch.Tensor]:
    """The per-expert weights and scales of a prepared MoE layer."""
    names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    return [getattr(layer, name).data for name in names]


def zero_empty_redundant(layer) -> None:
    """Zero this rank's empty redundant slots in every per-expert parameter of a
    placed layer (no checkpoint expert loads there), before preparation."""
    experts = local_experts(layer)
    if experts is None or not redundant_slots_per_rank(layer):
        return
    for param in layer.parameters(recurse=False):
        if param.ndim and param.shape[0] == experts.shape[0]:
            for i in np.flatnonzero(experts < 0):
                param.data[int(i)].zero_()


def check_replicas(layer) -> None:
    """Boot check of a layer with replicas, after weight preparation: every
    replica must hold its primary's bytes in each per-expert tensor.

    Each rank sums the int32 words of every slot (wrapping), plain and
    weighted by position, and the EP group gathers the sums over its CPU
    group. Raises on the first replica that differs from its primary.
    """
    if not redundant_slots_per_rank(layer) or host_slots(layer) is None:
        return
    from vllm.distributed.parallel_state import get_ep_group

    sums = _slot_sums(*_expert_tensors(layer)).cpu()
    group = get_ep_group()
    gathered = [torch.empty_like(sums) for _ in range(group.world_size)]
    torch.distributed.all_gather(gathered, sums, group=group.cpu_group)
    table = np.stack([g.numpy() for g in gathered])  # [ep, slots, sums]
    slots = host_slots(layer)
    ep_size = table.shape[0]
    local = slots.shape[0] // ep_size
    redundant = _REDUNDANT[layer.layer_name]
    redundant_per_rank = redundant.shape[0] // ep_size
    for q, e in enumerate(redundant):
        if e < 0:
            continue
        rank, index = divmod(int(q), redundant_per_rank)
        primary_rank, primary_index = divmod(int(slots[e]), local)
        if not np.array_equal(
            table[rank, local + index], table[primary_rank, primary_index]
        ):
            raise RuntimeError(
                f"{layer.layer_name}: the replica of expert {e} on EP rank {rank} "
                f"(local slot {local + index}) does not hold the bytes of its "
                f"primary on EP rank {primary_rank} (local slot {primary_index})"
            )


_slot_sums_op = None


def _slot_sums(*tensors: torch.Tensor) -> torch.Tensor:
    """``core.slot_checksums`` of a prepared layer's per-expert tensors."""
    global _slot_sums_op
    if _slot_sums_op is None:
        import jax
        from torch_tpu._internal import pallas

        def sums(
            w13: jax.Array, w2: jax.Array, s13: jax.Array, s2: jax.Array
        ) -> jax.Array:
            return core.slot_checksums(w13, w2, s13, s2)

        _slot_sums_op = pallas.jax_op("pallas::expert_slot_checksums", sums)
    return _slot_sums_op(*tensors)


@functools.cache
def _probe_counts_dir() -> None:
    """Fail at model construction, not after warm-up, when the routing
    counts directory cannot be written."""
    os.makedirs(_COUNTS_DIR, exist_ok=True)
    with tempfile.TemporaryFile(dir=_COUNTS_DIR):
        pass


def _register_counts(layer, device: torch.device) -> None:
    """Register the layer's routing counts [E] (int32) and, under DP, this
    engine's DP rank as data, so every rank compiles one program (see
    ``moe_routing.register_experts_start_buffer``)."""
    global _num_counted_experts
    name, num_experts = layer.layer_name, layer.global_num_experts
    parallel = layer.moe_config.moe_parallel_config
    if parallel.pcp_size > 1:
        raise NotImplementedError(
            f"{name}: TPU_MOE_ROUTING_COUNTS_DIR does not support PCP"
        )
    if _num_counted_experts not in (0, num_experts):
        raise NotImplementedError(
            f"{name}: {num_experts} experts, other counted MoE layers "
            f"{_num_counted_experts}; a counts file holds one expert count"
        )
    _num_counted_experts = num_experts
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    layer.register_buffer(_COUNTS, counts, persistent=False)
    if parallel.dp_size > 1:
        dp_rank = torch.tensor(parallel.dp_rank, dtype=torch.int32, device=device)
        layer.register_buffer(_COUNTS_DP_RANK, dp_rank, persistent=False)
    _counted_layers[name] = layer


def routing_counts(layer) -> torch.Tensor | None:
    """The layer's device routing counts [E] (int32), or None if not counted."""
    return getattr(layer, _COUNTS, None)


def counting_routes() -> bool:
    """Whether any MoE layer counts its routes (TPU_MOE_ROUTING_COUNTS_DIR)."""
    return bool(_counted_layers)


def count_routes(layer, topk_ids: torch.Tensor) -> None:
    """Add the logical expert ids [N, K] of this engine's real tokens to the
    layer's routing counts; a no-op for layers not counted. Runs inside the
    compiled forward."""
    counts = routing_counts(layer)
    if counts is None:
        return
    rows = topk_ids.shape[0]
    dp_size = layer.moe_config.moe_parallel_config.dp_size
    # The row layout token_padding.drop_padded_routes assumes.
    gathered = dp_size > 1 and not layer.quant_method.supports_internal_mk
    padding = token_padding.local_padding_rows(rows // dp_size if gathered else rows)
    assert padding is not None, "routing counts need the runner's padding mask"
    real = ~padding
    if gathered:
        # vLLM's dispatch gathered the DP group's rows by DP rank: count this
        # engine's block; every other engine counts its own.
        engines = torch.arange(dp_size, device=counts.device, dtype=torch.int32)
        mine = engines == getattr(layer, _COUNTS_DP_RANK)
        real = (mine[:, None] & real[None, :]).reshape(rows)
    # Compare, select and sum: no indexing op (see to_slots).
    experts = torch.arange(counts.shape[0], device=counts.device, dtype=torch.int32)
    hit = (topk_ids[..., None] == experts) & real[:, None, None]
    counts.add_(hit.sum(dim=(0, 1), dtype=torch.int32))


class _CountsWriter:
    """Host totals of the counted layers' device counts, rewritten to one file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.names = list(_counted_layers)
        self.layers = list(_counted_layers.values())
        self.totals = np.zeros((len(self.layers), _num_counted_experts), np.int64)
        # The device counts at the last read; None until the first step.
        self.last: np.ndarray | None = None
        self.due = 0.0
        # An unwritable file fails here, at startup, and the read program
        # compiles now instead of in the first serving step.
        core.save_counts(path, self.totals, self.names)
        self.read()

    def read(self) -> np.ndarray:
        counts = torch.stack([getattr(layer, _COUNTS) for layer in self.layers])
        return counts.cpu().numpy().view(np.uint32)

    def step(self, final: bool) -> None:
        now = time.time()
        # Due on whole minutes of the wall clock, so the engines of a DP group
        # read in the same step instead of each stalling the group in turn.
        due = (now // _WRITE_EVERY_S + 1) * _WRITE_EVERY_S
        if self.last is None:
            if not final:
                # What was counted before the first step is profiling and warm-up.
                self.last, self.due = self.read(), due
            return
        if now < self.due and not final:
            return
        self.due = due
        counts = self.read()
        # uint32 differences stay exact across the int32 wrap of device counts.
        self.totals += (counts - self.last).astype(np.int64)
        self.last = counts
        try:
            core.save_counts(self.path, self.totals, self.names)
        except OSError as e:
            logger.warning("MoE routing counts not written to %s: %s", self.path, e)
            return
        if final:
            logger.info("MoE routing counts written to %s at shutdown", self.path)


_writer: _CountsWriter | None = None


def start_writing_counts() -> None:
    """After warm-up, on the first TP rank of each DP engine and pipeline
    stage: start the routing counts file of TPU_MOE_ROUTING_COUNTS_DIR."""
    global _writer
    if not _COUNTS_DIR:
        return
    from vllm.distributed.parallel_state import (
        get_dp_group,
        get_pp_group,
        get_tensor_model_parallel_rank,
    )

    if get_tensor_model_parallel_rank() != 0:
        return
    if not _counted_layers:
        logger.warning(
            "TPU_MOE_ROUTING_COUNTS_DIR: no MoE layer here can be placed "
            "(NVFP4 grouped-matmul), so none is counted"
        )
        return
    name = (
        f"{core.COUNTS_PREFIX}dp{get_dp_group().rank_in_group}"
        f"-pp{get_pp_group().rank_in_group}-{socket.gethostname()}"
        f"-{os.getpid()}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.npz"
    )
    _writer = _CountsWriter(os.path.join(_COUNTS_DIR, name))
    logger.info(
        "MoE routing counts of %d layers -> %s, rewritten at most every %d s",
        len(_counted_layers),
        _writer.path,
        _WRITE_EVERY_S,
    )


def write_counts(final: bool = False) -> None:
    """Before each step, and with ``final`` at shutdown: add the device counts
    to the totals and rewrite the file, at most once a minute unless
    ``final``. A no-op on ranks without a counts file."""
    if _writer is None:
        return
    if not final:
        _writer.step(final=False)
        return
    try:
        _writer.step(final=True)
    except Exception as e:
        # Shutdown also runs after a failed step, and vLLM's teardown follows.
        logger.warning(
            "MoE routing counts since the last write to %s not saved: %s",
            _writer.path,
            e,
        )


def to_slots(layer, topk_ids: torch.Tensor) -> torch.Tensor:
    """Routed expert ids -> physical slots (negative ids stay as they are)."""
    slots = getattr(layer, "_expert_slots", None)
    if slots is None:
        return topk_ids
    # Compare, select and max: exact for any slot, and no indexing op
    # (indexing ops compile slowly in the decode graph).
    experts = torch.arange(slots.shape[0], device=topk_ids.device, dtype=torch.int32)
    hit = topk_ids[..., None] == experts
    placed = torch.where(hit, slots, -1).amax(dim=-1)
    return torch.where(topk_ids >= 0, placed, topk_ids).to(topk_ids.dtype)
