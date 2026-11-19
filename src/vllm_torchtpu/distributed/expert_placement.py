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
streams in. Logical expert `e` of layer `L` is loaded as physical expert
`slots[L][e]`, and the router's rows (`mlp.gate.weight`) and its score bias
are permuted the same way, so the model selects the same experts with the same
weights and each (token, expert) row is computed on the rank that now holds the
expert. No compiled program changes: every consumer of the router and the
expert weights (prefill and decode alike) sees one consistent relabelled model.

Served with `--additional-config '{"moe_expert_placement": "<table.json>"}'`
(a TABLE_FORMAT file); without the key nothing here runs. Tables come from the
served traffic's routing (`moe_prefill_sp`'s `moe_expert_load_stats` counts,
dumped by `TPUWorker.dump_moe_expert_load`) through `lpt_slots`; `main` turns a
dump into a table offline.
"""

import argparse
import dataclasses
import json
import os
import re
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

CONFIG_KEY = "moe_expert_placement"
TABLE_FORMAT = "vllm_torchtpu.moe_expert_placement.v1"
STATS_FORMAT = "vllm_torchtpu.moe_expert_load.v1"

_EXPERT = re.compile(r"(^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_ROUTER = re.compile(
    r"(^|\.)layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$"
)


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
    """slots[L][e]: the physical index of logical expert e of MoE layer L."""

    slots: dict[int, np.ndarray]

    @classmethod
    def from_table(
        cls,
        table: dict[str, Any],
        *,
        num_experts: int,
        moe_layers: Iterable[int],
        ep_size: int,
        ep_tp_size: int,
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
        return cls(slots)

    def physical_expert_id(self, name: str) -> int | None:
        """The physical expert a checkpoint tensor loads as, or None."""
        m = _EXPERT.search(name)
        if m is None:
            return None
        layer, expert = int(m.group(2)), int(m.group(3))
        slots = self.slots.get(layer)
        return expert if slots is None else int(slots[expert])

    def relabel(self, weights: Iterable[tuple[str, Any]]) -> Iterator[tuple[str, Any]]:
        """The checkpoint stream with every placed layer relabelled: expert
        tensors renamed to their physical index, router rows and score bias
        permuted to physical order (row p = logical expert inverse[p])."""
        inverse = {layer: np.argsort(s) for layer, s in self.slots.items()}
        for name, tensor in weights:
            m = _EXPERT.search(name)
            if m is not None and int(m.group(2)) in self.slots:
                p = self.slots[int(m.group(2))][int(m.group(3))]
                name = f"{name[: m.start(3)]}{p}{name[m.end(3) :]}"
            else:
                m = _ROUTER.search(name)
                if m is not None and int(m.group(2)) in inverse:
                    tensor = tensor[inverse[int(m.group(2))]]
            yield name, tensor


_PLACEMENT: dict[str, ExpertPlacement] = {}


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
    }
    bad = [why for why, hit in refusals.items() if hit]
    if bad:
        raise NotImplementedError(f"{CONFIG_KEY} " + "; ".join(bad) + ".")
    with open(path) as f:
        table = json.load(f)
    return ExpertPlacement.from_table(
        table,
        num_experts=vllm_config.model_config.get_num_experts(),
        moe_layers=moe_layer_indices(vllm_config.model_config.hf_config),
        ep_size=parallel.tensor_parallel_size,
        ep_tp_size=moe_hierarchical.ep_tp_size(),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LPT placement table from a load dump")
    p.add_argument("stats", help=f"a {STATS_FORMAT} file")
    p.add_argument("table", help=f"the {TABLE_FORMAT} file to write")
    a = p.parse_args(argv)
    with open(a.stats) as f:
        write_json(a.table, table_from_stats(json.load(f)))


if __name__ == "__main__":
    main()
