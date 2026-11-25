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
"""Process-wide `_scheduling_group_id` allocation shared by every kernel.

Kernels, including per-model copies of a reference kernel, must take their
ids from here (or from a module that re-exports these names) rather than
define their own counter or slot table: ids are global to a compiled program,
so two allocators would hand out the same ids within one program.
"""

import itertools
import threading

# Source of `_scheduling_group_id`s for chunked TensorCore/SparseCore
# pipelines (e.g. `deepseek_v4/streamindex_topk`). One counter for the
# process, because the ids are global to a compiled program and a model traces
# such a kernel once per layer: two layers sharing an id would be ONE group to
# XLA, and the legalizer would either annotate everything on the path between
# them or fail the compile. A per-model copy of a kernel must take its ids from
# here too, for the same reason.
#
# The ids end up in the HLO, so they are part of the compile cache key. The
# compiler restarts the counter for every program it builds (see
# `reset_scheduling_group_ids`): otherwise the ids, and with them the key,
# depend on which programs this process traced before, and two ranks building
# the same bucket in a different order miss each other's cached executable.
_scheduling_group_ids = itertools.count(1)
_scheduling_group_ids_lock = threading.Lock()


def reserve_scheduling_group_ids(count: int) -> int:
    """Reserves `count` scheduling group ids nobody else in this process gets.

    Args:
      count: Number of ids wanted; a pipeline over `n` chunks uses `n - 1` of them
        (chunk 0's TensorCore compute and the last chunk's SparseCore compute are
        not annotated).

    Returns:
      The first id; the caller owns `[first, first + count)`.
    """
    if count < 1:
        raise ValueError(f"count ({count}) must be positive.")
    with _scheduling_group_ids_lock:
        first = next(_scheduling_group_ids)
        for _ in range(count - 1):
            next(_scheduling_group_ids)
    return first


def reset_scheduling_group_ids() -> None:
    """Restarts the ids at 1, for the start of a new compiled program.

    Ids only need to be unique within one program, so the next program can
    reuse them; resetting makes a program's ids, and so its cache key, the same
    whatever this process compiled before.
    """
    global _scheduling_group_ids
    with _scheduling_group_ids_lock:
        _scheduling_group_ids = itertools.count(1)


# Ids for callers that name their call site (`scheduling_group_slot`, e.g. the
# layer index). The counter above is read while tracing, so a call that hits
# `jax.jit`'s cache -- a model's second layer with the first one's shapes --
# reuses the first call's ids, and XLA joins the two layers into one group: the
# path between them is un-annotated, which libtpu 0.0.50 rejects ("annotation
# groups with gaps"). A slot is a static argument instead, so it is part of the
# cache key: slot `s` owns `[SLOT_ID_BASE + s * MAX_IDS_PER_SLOT, +
# MAX_IDS_PER_SLOT)`, the same ids whatever was compiled before. The base keeps
# them clear of the counter's.
#
# A `streamindex_topk` pipeline over `n` chunks takes `n - 1` ids, and
# `chunk_tokens` may be any multiple of 4, so a slot holds enough for a
# 64Ki-chunk call (e.g. a 2M-token bucket in chunks of 32). Ids stay well
# inside int32 for every slot below `MAX_SCHEDULING_GROUP_SLOT`.
SLOT_ID_BASE = 1 << 20
MAX_IDS_PER_SLOT = 1 << 16
MAX_SCHEDULING_GROUP_SLOT = ((1 << 31) - SLOT_ID_BASE) // MAX_IDS_PER_SLOT - 1


def slot_scheduling_group_ids(slot: int, count: int) -> int:
    """The first of `count` scheduling group ids owned by call site `slot`."""
    if not 0 <= slot <= MAX_SCHEDULING_GROUP_SLOT:
        raise ValueError(
            f"scheduling group slot ({slot}) must be in "
            f"[0, {MAX_SCHEDULING_GROUP_SLOT}]."
        )
    if not 1 <= count <= MAX_IDS_PER_SLOT:
        raise ValueError(
            f"count ({count}) must be in [1, {MAX_IDS_PER_SLOT}]: a slot owns "
            f"{MAX_IDS_PER_SLOT} ids."
        )
    return SLOT_ID_BASE + slot * MAX_IDS_PER_SLOT


# Slots for call sites whose name holds no single layer index. They start past
# any real layer index, so they cannot collide with `layer_scheduling_group_slot`'s
# index-derived slots.
_FALLBACK_SLOT_BASE = 1 << 12
_fallback_slots: dict[str, int] = {}
_fallback_slots_lock = threading.Lock()


def layer_scheduling_group_slot(layer_prefix: str) -> int:
    """The `scheduling_group_slot` for the layer whose module prefix is given.

    The layer index when the prefix holds exactly one integer component
    ("model.layers.7.self_attn.indexer.k_cache" -> 7): layers of one forward
    have distinct indices (MTP layers continue after the target's), and the
    slot, and so the HLO and the compile cache key, depend on nothing but the
    layer -- not on construction order or on other models built in the same
    process. Any other prefix gets a fallback slot, numbered in first-seen
    order, which is the same on every rank.
    """
    ints = [int(part) for part in layer_prefix.split(".") if part.isdigit()]
    if len(ints) == 1 and ints[0] < _FALLBACK_SLOT_BASE:
        return ints[0]
    with _fallback_slots_lock:
        return _fallback_slots.setdefault(
            layer_prefix, _FALLBACK_SLOT_BASE + len(_fallback_slots)
        )
