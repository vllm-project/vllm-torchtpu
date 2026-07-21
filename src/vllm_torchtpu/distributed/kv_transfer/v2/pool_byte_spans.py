# SPDX-License-Identifier: Apache-2.0
"""Native byte-span declarations for Stage-3 resharding.

The producer lowers the PCP kernel's token ownership directly to the byte
ranges consumed by Raiden. GDN state is already a physical whole-slot copy,
so its declaration is built from the admitted pool manifest without a second
semantic span representation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class PoolByteSpan:
    """One byte copy between a registered source and destination block."""

    src_block_ordinal: int
    src_offset_bytes: int
    dst_block_index: int
    dst_offset_bytes: int
    size_bytes: int
    src_stride_bytes: int = 0
    dst_stride_bytes: int = 0
    count: int = 1


@dataclasses.dataclass(frozen=True)
class PoolSpanRegistration:
    """One rank's byte declarations for one exact pool tag."""

    tag: str
    block_ids: tuple[int, ...]
    spans: tuple[PoolByteSpan, ...]
    declared_bytes: int


def owned_token_ranges(
    *,
    num_tokens: int,
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
) -> list[tuple[int, int]]:
    """Return this producer rank's PCP-owned destination token ranges."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import pcp_layout

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not 0 <= transfer_rank < parallelism:
        raise ValueError(
            f"transfer_rank {transfer_rank} is outside parallelism "
            f"{parallelism}")
    if parallelism == 1:
        return [(0, num_tokens)]
    if interleave_tokens <= 0:
        raise ValueError("interleave_tokens must be positive")
    return list(
        pcp_layout.pcp_query_chunk_ranges(num_tokens, 0, transfer_rank,
                                          parallelism, interleave_tokens))


def lower_fa_spans(
    *,
    num_tokens: int,
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    dst_page_tokens: int,
    token_bytes: int,
    block_ids: Sequence[int],
) -> PoolSpanRegistration:
    """Lower PCP ownership to source- and destination-page byte ranges."""
    if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
    if dst_page_tokens <= 0:
        raise ValueError("dst_page_tokens must be positive")
    if token_bytes <= 0:
        raise ValueError("token_bytes must be positive")

    ranges = owned_token_ranges(
        num_tokens=num_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    spans: list[PoolByteSpan] = []
    local_cursor = 0
    for start, end in ranges:
        cursor = start
        while cursor < end:
            block_ordinal, block_token_offset = divmod(local_cursor,
                                                       page_tokens)
            source_end = min(end, cursor + page_tokens - block_token_offset)
            sub = cursor
            while sub < source_end:
                dst_page = sub // dst_page_tokens
                span_end = min((dst_page + 1) * dst_page_tokens, source_end)
                spans.append(
                    PoolByteSpan(
                        src_block_ordinal=block_ordinal,
                        src_offset_bytes=(block_token_offset + sub - cursor) *
                        token_bytes,
                        dst_block_index=dst_page,
                        dst_offset_bytes=(sub % dst_page_tokens) * token_bytes,
                        size_bytes=(span_end - sub) * token_bytes,
                    ))
                sub = span_end
            local_cursor += source_end - cursor
            cursor = source_end

    expected_blocks = (local_cursor + page_tokens - 1) // page_tokens
    if len(block_ids) != expected_blocks:
        raise ValueError(
            "block_ids do not match dense PCP rank packing: "
            f"got {len(block_ids)}, expected {expected_blocks} for "
            f"{local_cursor} owned tokens")
    return PoolSpanRegistration(
        tag="fa",
        block_ids=tuple(int(block_id) for block_id in block_ids),
        spans=tuple(spans),
        declared_bytes=local_cursor * token_bytes,
    )


def whole_slot_registration(*, tag: str, block_id: int,
                            live_bytes: int) -> PoolSpanRegistration:
    """Declare one manifest-sized state slot copied to destination slot 0."""
    if not tag:
        raise ValueError("state pool tag must not be empty")
    if block_id < 0:
        raise ValueError("state block id must be non-negative")
    if live_bytes <= 0:
        raise ValueError("state live bytes must be positive")
    return PoolSpanRegistration(
        tag=tag,
        block_ids=(int(block_id), ),
        spans=(PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=0,
            size_bytes=int(live_bytes),
        ), ),
        declared_bytes=int(live_bytes),
    )
