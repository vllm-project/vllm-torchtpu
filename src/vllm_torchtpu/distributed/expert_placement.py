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
"""Static expert placement: which EP rank holds which routed expert, per layer.

A placement relabels every MoE layer's routed experts as the checkpoint
streams in: logical expert `e` of layer `L` is loaded as physical expert
`slots[L][e]`, so each (token, expert) row is computed on the rank that now
holds the expert. The router (`mlp.gate.weight`) and its score bias keep the
checkpoint's logical order, so selection, its lowest-index tie rule and the
slot order inside the top-k are the identity placement's; every consumer of
the selected ids maps them through the layer's `SLOTS_ATTR` table after the
top-k (`physical_ids`, the fused EP op's `expert_slots`, moe_tp_decode).

Served with `--additional-config '{"moe_expert_placement": "<table.json>"}'`
(a TABLE_FORMAT file); without the key nothing here runs. Tables come from the
served traffic's routing (`moe_prefill_sp`'s `moe_expert_load_stats` counts,
dumped by `TPUWorker.dump_moe_expert_load`) through `lpt_slots`; `main` turns a
dump into a table offline.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

CONFIG_KEY = "moe_expert_placement"
TABLE_FORMAT = "vllm_torchtpu.moe_expert_placement.v1"
STATS_FORMAT = "vllm_torchtpu.moe_expert_load.v1"

# On a placed layer's RoutedExperts: int32 [E], the physical index of each logical
# expert (`install`).
SLOTS_ATTR = "_tpu_expert_slots"

_EXPERT = re.compile(r"(^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_LAYER = re.compile(r"(^|\.)layers\.(\d+)\.")


def lpt_slots(rows: np.ndarray, ep_size: int, ep_tp_size: int) -> np.ndarray:
    """slot[e], the physical index of logical expert e, from every expert's
    routed rows. Longest first, each expert goes to the EP group (chip) with the
    fewest rows among those with room left; then, longest first within each
    group, to the group member with the fewest rows. Physical expert p sits on
    mesh rank p // (E / ep_size), as the linear expert map lays them out; a rank
    holds its experts in index order."""
    rows = np.asarray(rows, np.int64)
    e = rows.size
    if ep_size % ep_tp_size or e % ep_size:
        raise ValueError(
            f"{e} experts do not split over ep {ep_size} / tp {ep_tp_size}"
        )
    groups, per_rank = ep_size // ep_tp_size, e // ep_size
    room = np.full(groups, per_rank * ep_tp_size)
    load = np.zeros(groups, np.int64)
    group_of = np.empty(e, np.int64)
    order = np.argsort(-rows, kind="stable")
    for x in order:
        free = room > 0
        g = np.flatnonzero(free)[np.argmin(load[free])]
        group_of[x], load[g], room[g] = g, load[g] + rows[x], room[g] - 1
    rank_of = np.empty(e, np.int64)
    rank_load = np.zeros(ep_size, np.int64)
    rank_room = np.full(ep_size, per_rank)
    for x in order:
        ranks = group_of[x] * ep_tp_size + np.arange(ep_tp_size)
        ranks = ranks[rank_room[ranks] > 0]
        rank_of[x] = ranks[np.argmin(rank_load[ranks])]
        rank_load[rank_of[x]] += rows[x]
        rank_room[rank_of[x]] -= 1
    slot = np.empty(e, np.int64)
    for r in range(ep_size):
        mine = np.flatnonzero(rank_of == r)
        slot[mine] = r * per_rank + np.arange(per_rank)
    return slot


def table_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """A TABLE_FORMAT placement from a STATS_FORMAT dump: LPT per layer."""
    if stats.get("format") != STATS_FORMAT:
        raise ValueError(f"not a {STATS_FORMAT} dump: {stats.get('format')!r}")
    ep, tp = int(stats["ep_size"]), int(stats["ep_tp_size"])
    return dict(
        format=TABLE_FORMAT,
        num_experts=int(stats["num_experts"]),
        ep_size=ep,
        ep_tp_size=tp,
        layers={
            layer: lpt_slots(np.asarray(rows), ep, tp).tolist()
            for layer, rows in stats["layers"].items()
        },
    )


def write_json(path: str, payload: dict[str, Any]) -> None:
    """Atomic write: concurrent writers of the same payload leave one copy."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


@dataclasses.dataclass(frozen=True)
class ExpertPlacement:
    """slots[L][e]: the physical index of logical expert e of MoE layer L;
    `digest` is the table file's sha256."""

    slots: dict[int, np.ndarray]
    digest: str = ""

    @classmethod
    def from_table(
        cls,
        table: dict[str, Any],
        *,
        num_experts: int,
        moe_layers: Iterable[int],
        ep_size: int,
        ep_tp_size: int,
        digest: str = "",
    ) -> "ExpertPlacement":
        """Validate a TABLE_FORMAT table against the served model and mesh."""
        want = dict(
            format=TABLE_FORMAT,
            num_experts=num_experts,
            ep_size=ep_size,
            ep_tp_size=ep_tp_size,
        )
        got = {k: table.get(k) for k in want}
        if got != want:
            raise ValueError(f"placement table is for {got}; this server is {want}")
        layers = table.get("layers")
        if not isinstance(layers, dict) or not all(
            isinstance(k, str) and k.isdigit() for k in layers
        ):
            raise ValueError("placement table: 'layers' maps layer index -> slots")
        if sorted(int(k) for k in layers) != sorted(moe_layers):
            raise ValueError(
                f"placement table layers {sorted(int(k) for k in layers)} are not "
                f"the model's MoE layers {sorted(moe_layers)}"
            )
        slots = {}
        for key, value in layers.items():
            s = np.asarray(value)
            if (
                s.shape != (num_experts,)
                or s.dtype.kind not in "iu"
                or not np.array_equal(np.sort(s), np.arange(num_experts))
            ):
                raise ValueError(f"layer {key}: slots are not a permutation of 0..E-1")
            slots[int(key)] = s.astype(np.int64)
        return cls(slots, digest)

    def physical_expert_id(self, name: str) -> int | None:
        """The physical expert a checkpoint tensor loads as, or None."""
        m = _EXPERT.search(name)
        if m is None:
            return None
        layer, expert = int(m.group(2)), int(m.group(3))
        slots = self.slots.get(layer)
        return expert if slots is None else int(slots[expert])

    def relabel(self, weights: Iterable[tuple[str, Any]]) -> Iterator[tuple[str, Any]]:
        """The checkpoint stream with every placed layer's expert tensors
        renamed to their physical index (weights and scales alike); the router
        and its score bias stream unchanged."""
        for name, tensor in weights:
            m = _EXPERT.search(name)
            if m is not None and int(m.group(2)) in self.slots:
                p = self.slots[int(m.group(2))][int(m.group(3))]
                name = f"{name[: m.start(3)]}{p}{name[m.end(3) :]}"
                _RELABELLED.add(self.digest)
            yield name, tensor


_PLACEMENT: dict[str, ExpertPlacement] = {}
# Digests of the placements a weight stream was relabelled by.
_RELABELLED: set[str] = set()


def moe_layer_indices(hf_config: Any) -> list[int]:
    """The routed-MoE decoder layers of a DeepSeek-V2-family config."""
    try:
        n, first = hf_config.num_hidden_layers, hf_config.first_k_dense_replace
    except AttributeError as e:
        raise NotImplementedError(
            f"{CONFIG_KEY} serves DeepSeek-V2-family MoE configs: {e}"
        ) from e
    freq = getattr(hf_config, "moe_layer_freq", 1)  # vLLM's deepseek_v2 default
    return [i for i in range(first, n) if i % freq == 0]


def active(vllm_config: Any) -> ExpertPlacement | None:
    """The served placement (read once per table path), or None without the
    key. Raises for a config the placement cannot serve."""
    extra = vllm_config.additional_config
    path = extra.get(CONFIG_KEY) if isinstance(extra, dict) else None
    if path is None:
        return None
    if path not in _PLACEMENT:
        _PLACEMENT[path] = _load(vllm_config, path)
    return _PLACEMENT[path]


def _load(vllm_config: Any, path: str) -> ExpertPlacement:
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import moe_hierarchical

    parallel, load = vllm_config.parallel_config, vllm_config.load_config
    refusals = {
        "needs --enable-expert-parallel": not parallel.enable_expert_parallel,
        "does not take EPLB": parallel.enable_eplb,
        "needs data_parallel_size 1": parallel.data_parallel_size != 1,
        "needs prefill_context_parallel_size 1": (
            parallel.prefill_context_parallel_size != 1
        ),
        "is applied by the Run:ai streaming loader only (--load-format "
        "runai_streamer)": load.load_format != "runai_streamer",
        "assumes the linear expert placement strategy": (
            parallel.expert_placement_strategy != "linear"
        ),
        # Kernels that select experts inside and take no slots table.
        "is not taken by the fused EP MoE kernel (USE_MOE_FUSED_EP_KERNEL)": (
            envs.USE_MOE_FUSED_EP_KERNEL
        ),
        "is not taken by FUSED_MOE_DECODE_KERNEL": envs.FUSED_MOE_DECODE_KERNEL,
    }
    bad = [why for why, hit in refusals.items() if hit]
    if bad:
        raise NotImplementedError(f"{CONFIG_KEY} " + "; ".join(bad) + ".")
    with open(path, "rb") as f:
        raw = f.read()
    return ExpertPlacement.from_table(
        json.loads(raw),
        num_experts=vllm_config.model_config.get_num_experts(),
        moe_layers=moe_layer_indices(vllm_config.model_config.hf_config),
        ep_size=parallel.tensor_parallel_size,
        ep_tp_size=moe_hierarchical.ep_tp_size(),
        digest=hashlib.sha256(raw).hexdigest(),
    )


def physical_ids(layer: Any, topk_ids: Any) -> Any:
    """`topk_ids` (selected logical experts, any shape) as the physical
    experts `layer` holds them at; unchanged on a layer without a placement."""
    slots = getattr(layer, SLOTS_ATTR, None)
    if slots is None:
        return topk_ids
    flat = topk_ids.reshape(-1).to(slots.dtype)
    return slots.index_select(0, flat).view(topk_ids.shape).to(topk_ids.dtype)


def install(model: Any, vllm_config: Any) -> int:
    """After the weights load, on every rank: stage each placed layer's slots
    table as SLOTS_ATTR on its RoutedExperts module and check that this rank's
    stream was relabelled and that every rank serves the same table. Returns
    the number of placed layers (0 without the key). Collective over the TP
    group when the key is set."""
    placement = active(vllm_config)
    if placement is None:
        return 0
    import torch
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu.logger import init_logger

    digests: list[Any] = [None] * torch.distributed.get_world_size(
        get_tp_group().cpu_group
    )
    torch.distributed.all_gather_object(
        digests, placement.digest, group=get_tp_group().cpu_group
    )
    if len(set(digests)) != 1:
        raise RuntimeError(f"{CONFIG_KEY}: the ranks serve different tables {digests}")
    if placement.digest not in _RELABELLED:
        raise RuntimeError(f"{CONFIG_KEY}: the weight stream was not relabelled")
    placed = 0
    for module in model.modules():
        if not isinstance(module, RoutedExperts):
            continue
        m = _LAYER.search(module.layer_name)
        slots = None if m is None else placement.slots.get(int(m.group(2)))
        if slots is None:
            continue
        table = torch.tensor(slots, dtype=torch.int32, device=module.w13_weight.device)
        module.register_buffer(SLOTS_ATTR, table, persistent=False)
        placed += 1
    if placed != len(placement.slots):
        raise RuntimeError(
            f"{CONFIG_KEY}: {placed} routed-expert layers matched the table's "
            f"{len(placement.slots)}"
        )
    moved = sum(int(np.sum(s != np.arange(s.size))) for s in placement.slots.values())
    init_logger(__name__).info(
        "%s: %s (sha256 %s) applied to %d layers, %d expert slots moved",
        CONFIG_KEY,
        vllm_config.additional_config[CONFIG_KEY],
        placement.digest,
        placed,
        moved,
    )
    return placed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LPT placement table from a load dump")
    p.add_argument("stats", help=f"a {STATS_FORMAT} file")
    p.add_argument("table", help=f"the {TABLE_FORMAT} file to write")
    a = p.parse_args(argv)
    with open(a.stats) as f:
        write_json(a.table, table_from_stats(json.load(f)))


if __name__ == "__main__":
    main()
