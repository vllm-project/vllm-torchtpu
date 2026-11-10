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
"""Which decode paths are on, resolved from the environment.

A path is a change whose only effect is how fast it runs (or the order of
additions in the streamed combine). Each belongs to one parent switch, which
turns it on; ``TPU_DECODE_FAST_PATHS_OFF`` (a comma list of path names) turns
single paths off again. ``PATHS`` is the one list of names.

Imports only ``envs``, so the resolution runs without a device.
"""

import dataclasses
import os
from collections.abc import Mapping

from vllm_torchtpu import envs

RAGGED_A2A = "TPU_MOE_RAGGED_A2A"
VERIFY_BLOCK = "TPU_BATCHED_RPA_VERIFY_BLOCK"
EMBED_EP_SHARD = "TPU_EMBED_EP_SHARD"
FAST_PATHS = "TPU_DECODE_FAST_PATHS"
FAST_PATHS_OFF = "TPU_DECODE_FAST_PATHS_OFF"

# Rows per capacity tile of the ragged exchange.
A2A_ROW_TILE = 32
# Column blocks per quarter of the hierarchical TP4 all-reduce.
TP4_ALL_REDUCE_BLOCKS = 4
# Partials of more rows than this reduce through XLA in that all-reduce.
TP4_ALL_REDUCE_MAX_ROWS = 256

# Attributes of a routed-experts layer that carry the owner-combine decision.
OWNER_FACTS_ATTR = "_tpu_owner_combine_facts"
OWNER_COMBINE_ATTR = "_tpu_owner_combine"


@dataclasses.dataclass(frozen=True)
class Path:
    """A path: its name in the off list, the variable that turns it on, and
    what it does."""

    name: str
    parent: str
    what: str


PATHS: tuple[Path, ...] = (
    Path(
        "a2a_streamed_combine",
        RAGGED_A2A,
        "the fused expert kernel sends each output row to the rank that sent "
        "its token, which sums them (unused where the owner combine is on)",
    ),
    Path(
        "a2a_sorted_plan",
        RAGGED_A2A,
        "the expert side orders its routes with one sort over the group's "
        "gathered routing",
    ),
    Path(
        "a2a_row_gather",
        RAGGED_A2A,
        "a row-DMA kernel copies the routes' token rows out of the receive "
        "buffer (needs a2a_sorted_plan)",
    ),
    Path(
        "a2a_row_scatter",
        RAGGED_A2A,
        "one kernel writes the send buffer (needs a2a_sorted_plan)",
    ),
    Path(
        "a2a_send_rows_minor",
        RAGGED_A2A,
        "with a2a_row_scatter, the send index sums its compare with the send "
        "rows minor",
    ),
    Path(
        "a2a_routing_dma",
        RAGGED_A2A,
        "one kernel of remote DMAs all-gathers the packed routing",
    ),
    Path(
        "a2a_plan_kernel",
        RAGGED_A2A,
        "with a2a_sorted_plan, one kernel computes the capacity exchange's fit "
        "check, sizes, offsets, send index and receive rows",
    ),
    Path(
        "a2a_unquantized",
        RAGGED_A2A,
        "unquantized layers without expert biases take the exchange",
    ),
    Path(
        "a2a_owner_combine",
        RAGGED_A2A,
        "expert outputs go to their token's owner rank, which sums them over "
        "TP in f32 (the shared expert included)",
    ),
    Path(
        "a2a_owner_tail",
        RAGGED_A2A,
        "with a2a_owner_combine, the expert kernel's tail sums the owned "
        "tokens and copies the sums to the engine's TP ranks",
    ),
    Path(
        "draft_one_query",
        VERIFY_BLOCK,
        "draft passes of one query per sequence run the one-query decode kernel",
    ),
    Path(
        "embed_rows",
        EMBED_EP_SHARD,
        "each rank sends only the embedding rows it owns to their engines",
    ),
    Path(
        "xla_schedule",
        FAST_PATHS,
        "batched RPA builds its decode schedules with array ops",
    ),
    Path(
        "gdn_packed",
        FAST_PATHS,
        "pooled GDN verify windows hand qkv, b and a to the kernels as one array",
    ),
    Path(
        "gdn_single_launch",
        FAST_PATHS,
        "pooled GDN runs its batched and per-sequence segments in one kernel",
    ),
    Path(
        "gdn_paired_heads",
        FAST_PATHS,
        "pooled GDN verify windows multiply two value heads per MXU dot",
    ),
)
PATH_NAMES = tuple(p.name for p in PATHS)


def _flag(environ: Mapping[str, str], name: str) -> bool:
    """The boolean value of ``name``; false when unset or empty."""
    value = environ.get(name)
    if value is None or value.strip() == "":
        return False
    lowered = value.strip().lower()
    if lowered in envs._TRUE_STRINGS:
        return True
    if lowered in envs._FALSE_STRINGS:
        return False
    raise ValueError(
        f"Invalid boolean value '{value}' for {name}. "
        f"Valid options: {envs._TRUE_STRINGS + envs._FALSE_STRINGS} (any case)."
    )


def _off_list(environ: Mapping[str, str]) -> frozenset[str]:
    names = [n.strip() for n in environ.get(FAST_PATHS_OFF, "").split(",")]
    names = [n for n in names if n]
    unknown = [n for n in names if n not in PATH_NAMES]
    if unknown:
        raise ValueError(
            f"{FAST_PATHS_OFF} names unknown path(s) {', '.join(unknown)}; "
            f"valid paths: {', '.join(PATH_NAMES)}."
        )
    return frozenset(names)


def _parent_on(environ: Mapping[str, str], parent: str) -> bool:
    if parent == VERIFY_BLOCK:
        return bool(environ.get(parent, "").strip())
    return _flag(environ, parent)


@dataclasses.dataclass(frozen=True)
class Switches:
    """The resolved switches: the paths that are on, and the exchange."""

    ragged_a2a: bool
    owner_combine: bool
    paths: frozenset[str]

    def path(self, name: str) -> bool:
        if name not in PATH_NAMES:
            raise KeyError(f"unknown path {name!r}; valid paths: {PATH_NAMES}")
        return name in self.paths


def resolve(environ: Mapping[str, str]) -> Switches:
    """The switches ``environ`` sets. Refuses an unknown name in the off list
    and combinations that cannot run."""
    off = _off_list(environ)
    ragged = _flag(environ, RAGGED_A2A)
    paths = frozenset(
        p.name for p in PATHS if p.name not in off and _parent_on(environ, p.parent)
    )
    for name in ("a2a_row_gather", "a2a_row_scatter"):
        if name in paths and "a2a_sorted_plan" not in paths:
            raise ValueError(
                f"{name} needs a2a_sorted_plan: remove a2a_sorted_plan from "
                f"{FAST_PATHS_OFF}."
            )
    return Switches(
        ragged_a2a=ragged,
        owner_combine="a2a_owner_combine" in paths,
        paths=paths,
    )


def get() -> Switches:
    """The switches of the process environment."""
    return resolve(os.environ)


def exchange_keywords(sw: Switches, *, owner_combine: bool) -> dict[str, object]:
    """The keywords ``ragged_a2a_moe`` takes from the switches, for a layer
    whose owner combine is ``owner_combine`` (see ``owner_combine_layer``)."""
    return dict(
        capacity_row_tile=A2A_ROW_TILE,
        row_gather=sw.path("a2a_row_gather"),
        row_scatter=sw.path("a2a_row_scatter"),
        streamed_combine=sw.path("a2a_streamed_combine"),
        owner_combine=owner_combine,
        sorted_plan=sw.path("a2a_sorted_plan"),
        owner_tail=owner_combine and sw.path("a2a_owner_tail"),
        routing_dma=sw.path("a2a_routing_dma"),
        send_index_rows_minor=sw.path("a2a_row_scatter")
        and sw.path("a2a_send_rows_minor"),
        plan_kernel=sw.path("a2a_sorted_plan") and sw.path("a2a_plan_kernel"),
    )


@dataclasses.dataclass(frozen=True)
class OwnerCombineFacts:
    """What the MoE runner knows of a layer that decides its owner combine."""

    sequence_parallel: bool
    pcp_size: int
    skip_final_all_reduce: bool
    has_output_transform: bool
    scaling_factor: float
    zero_expert_router: bool


def owner_combine_layer(sw: Switches, facts: OwnerCombineFacts | None) -> bool:
    """Whether a ragged layer's op returns its tokens summed over TP, shared
    expert included. The one decision of the op and the MoE runner."""
    return bool(
        sw.owner_combine
        and facts is not None
        and not facts.sequence_parallel
        and facts.pcp_size == 1
        and not facts.skip_final_all_reduce
        and not facts.has_output_transform
        and facts.scaling_factor == 1.0
        and not facts.zero_expert_router
    )


def arm_owner_combine(layer, sw: Switches | None = None) -> bool:
    """Decide ``layer``'s owner combine from the facts its runner published
    and store it for the op and the runner (``OWNER_COMBINE_ATTR``)."""
    armed = owner_combine_layer(
        get() if sw is None else sw, getattr(layer, OWNER_FACTS_ATTR, None)
    )
    setattr(layer, OWNER_COMBINE_ATTR, armed)
    return armed
