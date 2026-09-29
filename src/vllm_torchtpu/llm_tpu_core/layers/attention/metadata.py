# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware attention metadata dataclass and layout definitions.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax


class SequenceLayoutKind(str, Enum):
    ALL = "all"
    PARTIAL = "partial"


DEFAULT_SEQUENCE_LAYOUT_PROTOCOL = "default"
PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL = "pcp_streaming"


@dataclass(frozen=True)
class SequenceLayoutDescriptor:
    kind: SequenceLayoutKind = SequenceLayoutKind.ALL
    protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    version: int = 1

    @property
    def cache_key(self) -> tuple[str, str, int]:
        return (self.kind.value, self.protocol, int(self.version))


DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR = SequenceLayoutDescriptor()


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "mamba_slot_read_offsets",
        "mamba_ckpt_indices",
        "mamba_request_distribution",
    ],
    meta_fields=[
        "sequence_layout_kind",
        "sequence_layout_protocol",
        "sequence_layout_version",
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata:
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 - physical slot id in the mamba kv-cache for the
    # request currently in each persistent-batch position. Compact-mamba uses
    # an explicit slot from its independent pool; unified align mode derives
    # the slot from the current entry in the device block table. None for
    # attention groups, non-mamba models, and Mamba modes whose native fallback
    # addresses block_tables[:, 0].
    mamba_state_indices: jax.Array | None = None
    # (mamba_num_blocks,) int32 — per-*slot* read offset for speculative
    # decoding with mamba layers. `mamba_slot_read_offsets[base_slot]` is
    # `num_accepted - 1` from the request's most recent verify step: the GDN
    # kernel reads the request's initial state from `base_slot + offset`
    # (the checkpoint of the last accepted token) and writes fresh
    # checkpoints starting at `base_slot`. Indexed by physical slot (not
    # batch position) so the value survives requests being rescheduled or
    # condensed. Updated on device after each sampling step; None unless the
    # model has mamba layers *and* speculative decoding is enabled.
    mamba_slot_read_offsets: jax.Array | None = None
    # `(num_reqs, num_spec + 1)` source block per checkpoint. Unified pool
    # with speculative decoding only; None elsewhere, where the sole
    # checkpoint is the request's own state block.
    mamba_ckpt_indices: jax.Array | None = None
    # (3,) int32 — GDN-specific request distribution, same format as
    # `request_distribution` but with the first segment covering all
    # *windowed* sequences (plain decodes and speculative verify windows of
    # up to num_spec + 1 tokens) instead of only 1-token decodes. The
    # persistent batch is ordered [decode][verify][prefill/mixed] so both
    # segmentations hold at once: ragged paged attention keeps its 1-token
    # decode front segment while the GDN kernel runs its windowed mode over
    # the first two groups. None unless the model has mamba layers and spec
    # decoding is enabled.
    mamba_request_distribution: jax.Array | None = None
    sequence_layout_kind: str = SequenceLayoutKind.ALL.value
    sequence_layout_protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    sequence_layout_version: int = 1

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)
