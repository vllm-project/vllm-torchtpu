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
"""StreamIndex Top-K Configuration."""

import enum
import itertools
import threading


class MlaCase(enum.Enum):
    """Represents the different cases for MLA.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


class KVLayout(enum.Enum):
    """Memory layout of the packed FP8 index KV cache.

    - HEAD_ALONG_SUBLANE: `[pages, page_size // packing, packing, width]`.
      Tokens live on the sublane dimension and the `head_dim` features live on
      the lane dimension. Because per-token quantization appends a UE8M0 scale
      byte after the `head_dim` FP8 bytes, `width` must be padded up to a
      multiple of the 128-lane register width. For `head_dim=128` that means 129
      useful bytes are stored in 256, i.e. **half of the HBM traffic is padding**
      and the kernel can never exceed ~0.5 of the HBM roofline.

    - SEQ_ALONG_LANE: `[pages, head_dim // packing + 1, packing, page_size]`.
      The sequence is packed along the lane dimension and `head_dim` lives on the
      sublane dimension, so the per-token scale only needs to grow the *sublane*
      count. One extra packed sublane group (`packing` rows, of which the first
      holds the UE8M0 scale) is enough, giving `head_dim + packing` bytes per
      token (132 instead of 256 for `head_dim=128`).

      This also makes the QK product the MXU-native `[n, d] x [d, m]` form and
      yields the per-token scales directly as a `[1, bkv_sz]` row vector, so the
      lane broadcast of the dequantization scale becomes free.
    """

    HEAD_ALONG_SUBLANE = 0
    SEQ_ALONG_LANE = 1

    @classmethod
    def parse(cls, value: "str | KVLayout | None") -> "KVLayout":
        """Parses a `KVLayout` from a case-insensitive string (or passes through)."""
        if value is None:
            return cls.HEAD_ALONG_SUBLANE
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "head_along_sublane": cls.HEAD_ALONG_SUBLANE,
            "seq_along_lane": cls.SEQ_ALONG_LANE,
            # Tolerate the common typo seen in flags/configs.
            "seq_alone_lane": cls.SEQ_ALONG_LANE,
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unknown kv_layout {value!r}; expected one of {sorted(aliases)}."
            )
        return aliases[normalized]


# Default buffer_count for (DECODE, PREFILL, MIXED) cases.
# Determined from microbenchmarks for 8k/1k and 256k/1k.
DEFAULT_BUFFER_COUNT: tuple[int, int, int] = (4, 3, 3)


# Source of `_scheduling_group_id`s for the chunked TensorCore/SparseCore
# pipeline. One counter for the process, because the ids are global to a
# compiled program and a model traces this kernel once per layer: two layers
# sharing an id would be ONE group to XLA, and the legalizer would either
# annotate everything on the path between them or fail the compile.
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
# A pipeline over `n` chunks takes `n - 1` ids, and `chunk_tokens` may be any
# multiple of 4, so a slot holds enough for a 64Ki-chunk call (e.g. a 2M-token
# bucket in chunks of 32). Ids stay well inside int32 for every slot below
# `MAX_SCHEDULING_GROUP_SLOT`.
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


def num_chunks(num_tokens: int, chunk_tokens: int | None) -> int:
    """How many chunks the top-k pipeline cuts a `num_tokens` bucket into.

    One (an unpipelined pass) unless `chunk_tokens` divides `num_tokens` into
    at least two.
    """
    if chunk_tokens is None or num_tokens % chunk_tokens:
        return 1
    return num_tokens // chunk_tokens


def chunked_scheduling_group_slot(
    slot: int | None, num_tokens: int, chunk_tokens: int | None
) -> int | None:
    """`slot` if a `num_tokens` call is pipelined, else None.

    Only a pipelined call has scheduling groups, so only it needs its own
    slot. The slot is a static argument of the jitted kernel: passing one to
    an unpipelined call would split `jax.jit`'s trace cache by call site for
    nothing, and every layer would trace every unchunked bucket again.
    """
    return slot if num_chunks(num_tokens, chunk_tokens) > 1 else None


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


def reset_scheduling_group_ids() -> None:
    """Restarts the ids at 1, for the start of a new compiled program.

    Ids only need to be unique within one program, so the next program can
    reuse them; resetting makes a program's ids, and so its cache key, the same
    whatever this process compiled before.
    """
    global _scheduling_group_ids
    with _scheduling_group_ids_lock:
        _scheduling_group_ids = itertools.count(1)
