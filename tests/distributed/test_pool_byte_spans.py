# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
from vllm_torchtpu.distributed.kv_transfer.raiden import byte_spans

"""Focused tests for native Stage-3 byte-span lowering.

Every extent the lowering emits must be a whole multiple of its class's
physical granule. For raw GDN spans that granule is the 1024-byte pool
token, so those fixtures use real token-scaled geometries: the
Qwen3.5-35B TP8 shard for the QK pair-blocked vocabulary and a TP4 shard
for the legacy segment-major vocabulary. For GLM-5.2 the granule is the
packed row (4 tokens: 2560 B of latent, 1024 B of indexer).
"""

from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    PoolByteSpan, lower_gdn_state_shard_spans, lower_glm_row_spans)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import \
    RegionSpec

# Qwen3.5-35B TP8 shard, QK pair-blocked: one 1024-byte QK token plus one
# 1024-byte V token per tap.
_CONV_REGIONS_PAIR_TP8 = (
    RegionSpec(name="gdn_conv_qk",
               offset_bytes=0,
               stride_bytes=2048,
               unit_bytes=1024,
               num_units=3,
               units_per_stride=1),
    RegionSpec(name="gdn_conv_v",
               offset_bytes=1024,
               stride_bytes=2048,
               unit_bytes=256,
               num_units=3,
               units_per_stride=4),
)

# Qwen3.5-35B TP4 shard, legacy segment-major: 1024-byte Q and K segments
# and a 4096-byte V segment per tap — all whole tokens on their own.
_CONV_REGIONS_LEGACY_TP4 = (
    RegionSpec(name="gdn_conv_q",
               offset_bytes=0,
               stride_bytes=6144,
               unit_bytes=256,
               num_units=3,
               units_per_stride=4),
    RegionSpec(name="gdn_conv_k",
               offset_bytes=1024,
               stride_bytes=6144,
               unit_bytes=256,
               num_units=3,
               units_per_stride=4),
    RegionSpec(name="gdn_conv_v",
               offset_bytes=2048,
               stride_bytes=6144,
               unit_bytes=256,
               num_units=3,
               units_per_stride=16),
)

# Qwen3.5-35B TP8 shard, legacy segment-major: the 512-byte Q/K half-token
# segments are not placement-exact and must be rejected.
_CONV_REGIONS_LEGACY_TP8 = (
    RegionSpec(name="gdn_conv_q",
               offset_bytes=0,
               stride_bytes=2048,
               unit_bytes=256,
               num_units=3,
               units_per_stride=2),
    RegionSpec(name="gdn_conv_k",
               offset_bytes=512,
               stride_bytes=2048,
               unit_bytes=256,
               num_units=3,
               units_per_stride=2),
    RegionSpec(name="gdn_conv_v",
               offset_bytes=1024,
               stride_bytes=2048,
               unit_bytes=256,
               num_units=3,
               units_per_stride=4),
)

_SSM_REGIONS = (RegionSpec(name="gdn_ssm",
                           offset_bytes=0,
                           stride_bytes=1024,
                           unit_bytes=1024,
                           num_units=4), )


def _copy_registration(destination: bytearray, source: bytes,
                       registration) -> None:
    for span in registration.spans:
        for repeat in range(span.count):
            src = span.src_offset_bytes + repeat * span.src_stride_bytes
            dst = span.dst_offset_bytes + repeat * span.dst_stride_bytes
            destination[dst:dst + span.size_bytes] = source[src:src +
                                                            span.size_bytes]


def test_gdn_conv_pair_shards_reassemble_rank_blocks():
    parallelism = 8
    destination = bytearray(3 * 2048 * parallelism)
    expected = bytearray()
    rank_sources = []
    for rank in range(parallelism):
        source = bytearray()
        for tap in range(3):
            source.extend(bytes([10 + rank]) * 1024)  # qk pair token
            source.extend(bytes([50 + rank]) * 1024)  # v heads
        rank_sources.append(bytes(source))

    for tap in range(3):
        expected.extend(b"".join(
            bytes([10 + rank]) * 1024 for rank in range(parallelism)))
        expected.extend(b"".join(
            bytes([50 + rank]) * 1024 for rank in range(parallelism)))

    for rank, source in enumerate(rank_sources):
        registration = lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=17,
            transfer_rank=rank,
            parallelism=parallelism,
            regions=_CONV_REGIONS_PAIR_TP8,
        )
        assert registration.block_ids == (17, )
        assert registration.declared_bytes == len(source) == 6144
        _copy_registration(destination, source, registration)

    assert destination == expected


def test_gdn_conv_legacy_shards_reassemble_segment_major():
    parallelism = 4
    destination = bytearray(3 * 6144 * parallelism)
    expected = bytearray()
    rank_sources = []
    for rank in range(parallelism):
        source = bytearray()
        for tap in range(3):
            source.extend(bytes([10 + rank]) * 1024)  # q heads
            source.extend(bytes([30 + rank]) * 1024)  # k heads
            source.extend(bytes([50 + rank]) * 4096)  # v heads
        rank_sources.append(bytes(source))

    for tap in range(3):
        expected.extend(b"".join(
            bytes([10 + rank]) * 1024 for rank in range(parallelism)))
        expected.extend(b"".join(
            bytes([30 + rank]) * 1024 for rank in range(parallelism)))
        expected.extend(b"".join(
            bytes([50 + rank]) * 4096 for rank in range(parallelism)))

    for rank, source in enumerate(rank_sources):
        registration = lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=17,
            transfer_rank=rank,
            parallelism=parallelism,
            regions=_CONV_REGIONS_LEGACY_TP4,
        )
        assert registration.declared_bytes == len(source) == 18432
        _copy_registration(destination, source, registration)

    assert destination == expected


def test_gdn_ssm_shards_reassemble_contiguous_heads():
    parallelism = 8
    destination = bytearray(4096 * parallelism)
    for rank in range(parallelism):
        source = bytes([rank + 1]) * 4096
        registration = lower_gdn_state_shard_spans(
            tag="gdn.ssm.g2",
            block_id=23,
            transfer_rank=rank,
            parallelism=parallelism,
            regions=_SSM_REGIONS,
        )
        assert registration.spans == (PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=rank * 4096,
            size_bytes=4096,
        ), )
        assert registration.declared_bytes == 4096
        _copy_registration(destination, source, registration)

    assert destination == b"".join(
        bytes([rank + 1]) * 4096 for rank in range(parallelism))


def test_gdn_conv_pair_rank_span_geometry():
    registration = lower_gdn_state_shard_spans(
        tag="gdn.conv.g1",
        block_id=9,
        transfer_rank=3,
        parallelism=8,
        regions=_CONV_REGIONS_PAIR_TP8,
    )

    assert registration.spans == (
        PoolByteSpan(0, 0, 0, 3072, 1024, 2048, 16384, 3),
        PoolByteSpan(0, 1024, 0, 8192 + 3072, 1024, 2048, 16384, 3),
    )


def test_gdn_conv_legacy_rank_span_geometry():
    registration = lower_gdn_state_shard_spans(
        tag="gdn.conv.g1",
        block_id=9,
        transfer_rank=3,
        parallelism=4,
        regions=_CONV_REGIONS_LEGACY_TP4,
    )

    assert registration.spans == (
        PoolByteSpan(0, 0, 0, 3072, 1024, 6144, 24576, 3),
        PoolByteSpan(0, 1024, 0, 4096 + 3072, 1024, 6144, 24576, 3),
        PoolByteSpan(0, 2048, 0, 8192 + 12288, 4096, 6144, 24576, 3),
    )


def test_qwen35_35b_pcp8_rank_three_state_offsets():
    conv = lower_gdn_state_shard_spans(
        tag="gdn.conv.g0",
        block_id=5,
        transfer_rank=3,
        parallelism=8,
        regions=_CONV_REGIONS_PAIR_TP8,
    )
    ssm = lower_gdn_state_shard_spans(
        tag="gdn.ssm.g0",
        block_id=5,
        transfer_rank=3,
        parallelism=8,
        regions=(RegionSpec("gdn_ssm", 0, 65_536, 65_536, 4), ),
    )

    assert conv.declared_bytes == 6_144
    assert conv.spans == (
        PoolByteSpan(0, 0, 0, 3_072, 1_024, 2_048, 16_384, 3),
        PoolByteSpan(0, 1_024, 0, 11_264, 1_024, 2_048, 16_384, 3),
    )
    assert ssm.declared_bytes == 262_144
    assert ssm.spans == (PoolByteSpan(0, 0, 0, 786_432, 262_144), )


def test_gdn_legacy_subtoken_qk_segments_rejected():
    try:
        lower_gdn_state_shard_spans(tag="gdn.conv.g0",
                                    block_id=0,
                                    transfer_rank=0,
                                    parallelism=8,
                                    regions=_CONV_REGIONS_LEGACY_TP8)
    except ValueError as exc:
        assert "whole physical tokens" in str(exc)
    else:
        raise AssertionError("sub-token legacy Q/K segments were accepted")


def test_gdn_subtoken_ssm_shard_rejected():
    try:
        lower_gdn_state_shard_spans(
            tag="gdn.ssm.g0",
            block_id=0,
            transfer_rank=0,
            parallelism=8,
            regions=(RegionSpec("gdn_ssm", 0, 64, 64, 4), ),
        )
    except ValueError as exc:
        assert "whole physical tokens" in str(exc)
    else:
        raise AssertionError("sub-token SSM shard was accepted")


def test_gdn_lowering_rejects_unknown_conv_vocabulary():
    try:
        lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=0,
            transfer_rank=0,
            parallelism=8,
            regions=_CONV_REGIONS_LEGACY_TP4[:2],  # q and k only
        )
    except ValueError as exc:
        assert "gdn_conv_qk" in str(exc)
    else:
        raise AssertionError("unknown conv vocabulary was accepted")


def test_gdn_lowering_rejects_non_dense_conv_layout():
    bad_regions = list(_CONV_REGIONS_LEGACY_TP4)
    bad_regions[1] = RegionSpec(name="gdn_conv_k",
                                offset_bytes=2048,
                                stride_bytes=6144,
                                unit_bytes=256,
                                num_units=3,
                                units_per_stride=4)

    try:
        lower_gdn_state_shard_spans(tag="gdn.conv.g0",
                                    block_id=0,
                                    transfer_rank=0,
                                    parallelism=4,
                                    regions=bad_regions)
    except ValueError as exc:
        assert "dense tap-major" in str(exc)
    else:
        raise AssertionError("non-dense conv layout was accepted")


# ---------------------------------------------------------------------------
# GLM row-granular span lowering (TP-replicated caches, striped pages).
# ---------------------------------------------------------------------------

_GLM_PAGE_TOKENS = 1024
_GLM_FA_ROW_BYTES = 2560  # 4 tokens x 640 B
_GLM_FA_LIVE = 256 * _GLM_FA_ROW_BYTES
_GLM_IDX_ROW_BYTES = 1024  # 4 tokens x 256 B
_GLM_IDX_LIVE = 256 * _GLM_IDX_ROW_BYTES


def _glm_row_spans(**overrides):
    kwargs = dict(
        tag="fa",
        num_tokens=3 * _GLM_PAGE_TOKENS,
        transfer_rank=0,
        parallelism=2,
        page_tokens=_GLM_PAGE_TOKENS,
        live_bytes_per_block=_GLM_FA_LIVE,
        row_bytes=_GLM_FA_ROW_BYTES,
        block_ids=[10, 11, 12],
    )
    kwargs.update(overrides)
    return lower_glm_row_spans(**kwargs)


def test_glm_row_spans_round_robin_full_pages():
    rank0 = _glm_row_spans(transfer_rank=0)
    rank1 = _glm_row_spans(transfer_rank=1)

    assert rank0.dst_space_version == 1
    assert rank0.block_ids == (10, 11, 12)
    assert rank0.spans == (
        PoolByteSpan(src_block_ordinal=0,
                     src_offset_bytes=0,
                     dst_block_index=0,
                     dst_offset_bytes=0,
                     size_bytes=_GLM_FA_LIVE),
        PoolByteSpan(src_block_ordinal=2,
                     src_offset_bytes=0,
                     dst_block_index=0,
                     dst_offset_bytes=2 * _GLM_FA_LIVE,
                     size_bytes=_GLM_FA_LIVE),
    )
    assert rank1.spans == (PoolByteSpan(src_block_ordinal=1,
                                        src_offset_bytes=0,
                                        dst_block_index=0,
                                        dst_offset_bytes=_GLM_FA_LIVE,
                                        size_bytes=_GLM_FA_LIVE), )
    assert rank0.declared_bytes == 2 * _GLM_FA_LIVE
    assert rank1.declared_bytes == _GLM_FA_LIVE


def test_glm_row_spans_trim_partial_tail_to_rows():
    for tag, live, row_bytes in (("fa", _GLM_FA_LIVE, _GLM_FA_ROW_BYTES),
                                 ("dsa.idx", _GLM_IDX_LIVE,
                                  _GLM_IDX_ROW_BYTES)):
        registration = _glm_row_spans(
            tag=tag,
            num_tokens=2 * _GLM_PAGE_TOKENS + 5,
            live_bytes_per_block=live,
            row_bytes=row_bytes,
        )
        # Rank 0 of 2 owns pages {0, 2}; 5 tail tokens round up to 2 rows.
        assert [span.size_bytes
                for span in registration.spans] == [live, 2 * row_bytes]
        assert registration.spans[-1].dst_offset_bytes == 2 * live
        assert registration.declared_bytes == live + 2 * row_bytes


def test_glm_row_spans_nope_trims_per_token():
    # A split sparse MLA nope pool holds ONE token per row (row_bytes 512,
    # live = page_tokens * 512), so the tail trims at token granularity.
    nope_row_bytes = 512
    nope_live = _GLM_PAGE_TOKENS * nope_row_bytes
    registration = _glm_row_spans(
        tag="mla.nope",
        num_tokens=2 * _GLM_PAGE_TOKENS + 5,
        live_bytes_per_block=nope_live,
        row_bytes=nope_row_bytes,
    )
    # Rank 0 of 2 owns pages {0, 2}; 5 tail tokens = exactly 5 rows.
    assert [span.size_bytes
            for span in registration.spans] == [nope_live, 5 * nope_row_bytes]
    assert registration.declared_bytes == nope_live + 5 * nope_row_bytes


def test_glm_row_spans_rope_trims_to_packed_rows():
    # The rope half packs 4 tokens per 512-byte row (live = page_tokens/4
    # * 512): 5 tail tokens round up to 2 rows.
    rope_row_bytes = 512
    rope_live = (_GLM_PAGE_TOKENS // 4) * rope_row_bytes
    registration = _glm_row_spans(
        tag="mla.rope",
        num_tokens=2 * _GLM_PAGE_TOKENS + 5,
        live_bytes_per_block=rope_live,
        row_bytes=rope_row_bytes,
    )
    assert [span.size_bytes
            for span in registration.spans] == [rope_live, 2 * rope_row_bytes]
    assert registration.declared_bytes == rope_live + 2 * rope_row_bytes


def test_glm_row_spans_parallelism_one_owns_every_page():
    registration = _glm_row_spans(num_tokens=2 * _GLM_PAGE_TOKENS,
                                  parallelism=1,
                                  block_ids=[7, 9])
    assert [span.src_block_ordinal for span in registration.spans] == [0, 1]
    assert [span.size_bytes
            for span in registration.spans] == [_GLM_FA_LIVE, _GLM_FA_LIVE]


def test_glm_row_spans_rank_owning_no_pages_keeps_block_ids():
    registration = _glm_row_spans(num_tokens=100,
                                  transfer_rank=5,
                                  parallelism=8,
                                  block_ids=[3])
    assert registration.spans == ()
    assert registration.declared_bytes == 0
    assert registration.block_ids == (3, )


def test_glm_row_spans_union_partitions_request_bytes():
    num_pages, parallelism = 11, 8
    num_tokens = (num_pages - 1) * _GLM_PAGE_TOKENS + 7
    owners: dict[int, int] = {}
    total = 0
    for rank in range(parallelism):
        registration = _glm_row_spans(num_tokens=num_tokens,
                                      transfer_rank=rank,
                                      parallelism=parallelism,
                                      block_ids=list(range(num_pages)))
        for span in registration.spans:
            assert span.src_block_ordinal not in owners
            owners[span.src_block_ordinal] = rank
        total += registration.declared_bytes
    assert sorted(owners) == list(range(num_pages))
    assert total == (num_pages - 1) * _GLM_FA_LIVE + 2 * _GLM_FA_ROW_BYTES


def test_glm_row_spans_validation_failures():
    cases = (
        (dict(block_ids=[10, 11]), "complete request page set"),
        (dict(live_bytes_per_block=_GLM_FA_LIVE + 1), "whole rows"),
        (dict(page_tokens=1000), "spread evenly"),
        (dict(transfer_rank=2), "outside parallelism"),
        (dict(num_tokens=0), "num_tokens must be positive"),
        (dict(row_bytes=0), "row_bytes and live_bytes_per_block"),
    )
    for overrides, message in cases:
        try:
            _glm_row_spans(**overrides)
        except ValueError as exc:
            assert message in str(exc), (overrides, exc)
        else:
            raise AssertionError(f"accepted invalid input: {overrides}")


# ---------------------------------------------------------------------------
# Chunked sub-window byte span lowering & decomposition additivity
# ---------------------------------------------------------------------------

def _token_payload(token_id: int, token_bytes: int) -> bytes:
    """Generates a deterministic unique byte pattern for a given token.

    Encodes the token identifier as a big-endian 4-byte integer and tiles it
    across the requested byte width, creating a distinct payload for verifying
    memory copy fidelity.

    Args:
        token_id: Integer index identifying the token within the global prompt.
        token_bytes: Total number of bytes per token payload.

    Returns:
        A bytes object of length token_bytes populated with tiled token data.
    """
    pattern = token_id.to_bytes(4, byteorder="big", signed=False)
    repeat_count = token_bytes // 4
    remainder = token_bytes % 4
    return (pattern * repeat_count) + pattern[:remainder]


def _execute_span_copy(
    *,
    registration: byte_spans.PoolSpanRegistration,
    source_blocks: Sequence[bytearray],
    dst_buffer: bytearray,
) -> None:
    """Copies bytes from source blocks into dst_buffer per span descriptors.

    Iterates over every PoolByteSpan descriptor in the registration, extracting
    the source memory slice from the designated source block ordinal and writing
    it directly into the destination buffer at the target byte offset.

    Args:
        registration: PoolSpanRegistration holding copy span descriptors.
        source_blocks: Sequence of allocated source memory bytearrays.
        dst_buffer: Mutable destination bytearray receiving transferred bytes.
    """
    for span in registration.spans:
        src = source_blocks[span.src_block_ordinal]
        src_start = span.src_offset_bytes
        dst_start = span.dst_offset_bytes
        size = span.size_bytes
        dst_buffer[dst_start : dst_start + size] = src[
            src_start : src_start + size
        ]


def _build_populated_source_blocks(
    *,
    owned_ranges: Sequence[tuple[int, int]],
    page_tokens: int,
    token_bytes: int,
    num_blocks: int,
) -> list[bytearray]:
    """Allocates and populates source blocks with deterministic token data.

    Creates a list of zeroed bytearray source blocks and populates them with
    deterministic token payload bytes for each token index across the provided
    PCP-owned half-open token ranges.

    Args:
        owned_ranges: Sequence of (start_token, end_token) half-open index
          pairs.
        page_tokens: Capacity of each source page block measured in tokens.
        token_bytes: Byte size allocated for a single token's KV cache.
        num_blocks: Total number of source blocks to instantiate.

    Returns:
        A list of populated bytearray objects representing source pages.
    """
    block_size = page_tokens * token_bytes
    blocks = [bytearray(block_size) for _ in range(num_blocks)]
    local_token_idx = 0
    for start, end in owned_ranges:
        for token_id in range(start, end):
            block_idx = local_token_idx // page_tokens
            offset = (local_token_idx % page_tokens) * token_bytes
            payload = _token_payload(token_id, token_bytes)
            blocks[block_idx][offset : offset + token_bytes] = payload
            local_token_idx += 1
    return blocks


def _populate_and_execute_chunk_copies(
    *,
    chunk_sizes: Sequence[int],
    chunk_regs: Sequence[byte_spans.PoolSpanRegistration],
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
    dst_buffer: bytearray,
) -> int:
    """Simulates chunked memory copy execution for a specific rank.

    For each sub-chunk in chunk_sizes, determines owned token ranges, allocates
    populated source pages, and executes simulated span copies into dst_buffer.
    Returns the sum of declared transfer bytes across all processed chunks.

    Args:
        chunk_sizes: Sequence of prompt token counts for each sub-chunk.
        chunk_regs: Sequence of lowered PoolSpanRegistrations for the
          sub-chunks.
        transfer_rank: Context Parallel rank performing the transfer.
        parallelism: Context Parallelism degree (PCP size).
        interleave_tokens: Tokens assigned per rank stride.
        page_tokens: Tokens stored per physical KV cache page block.
        token_bytes: Byte size per token representation.
        dst_buffer: Destination bytearray accumulating transferred data.

    Returns:
        Total number of declared transfer bytes processed across all chunks.
    """
    current_base = 0
    total_declared_bytes = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        chunk_ranges = byte_spans.owned_token_ranges(
            token_base=current_base,
            num_tokens=chunk_size,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        chunk_owned_tokens = sum(end - start for start, end in chunk_ranges)
        chunk_num_blocks = (
            chunk_owned_tokens + page_tokens - 1
        ) // page_tokens
        chunk_blocks = _build_populated_source_blocks(
            owned_ranges=chunk_ranges,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            num_blocks=chunk_num_blocks,
        )
        _execute_span_copy(
            registration=chunk_regs[chunk_idx],
            source_blocks=chunk_blocks,
            dst_buffer=dst_buffer,
        )
        total_declared_bytes += chunk_regs[chunk_idx].declared_bytes
        current_base += chunk_size
    return total_declared_bytes


def _verify_span_decomposition_for_rank(
    *,
    chunk_sizes: Sequence[int],
    transfer_rank: int,
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
) -> tuple[
    byte_spans.PoolSpanRegistration, list[byte_spans.PoolSpanRegistration]
]:
    """Verifies that sub-chunk span lowering decomposes losslessly for a rank.

    Computes both a monolithic full-prompt span registration and a sequence of
    sub-chunk registrations for the specified PCP rank. Asserts that the sum of
    chunk declared bytes equals monolithic declared bytes, that individual
    chunk spans match the monolithic destination offsets and sizes, and that
    simulated memory copy outputs are bit-for-bit identical.

    Args:
        chunk_sizes: Sequence of token counts defining consecutive sub-chunks.
        transfer_rank: Zero-based Context Parallel rank being evaluated.
        parallelism: Context Parallelism degree (PCP size).
        interleave_tokens: Tokens assigned per rank stride.
        page_tokens: Capacity of each KV cache page block in tokens.
        token_bytes: Byte size allocated per token.

    Returns:
        A tuple of (monolithic_registration, list_of_chunk_registrations).
    """
    total_tokens = sum(chunk_sizes)
    cycle = parallelism * interleave_tokens
    for size in chunk_sizes:
        assert (
            size % cycle == 0
        ), f"Chunk size {size} must align to cycle {cycle}"

    # 1. Monolithic Lowering
    mono_ranges = byte_spans.owned_token_ranges(
        token_base=0,
        num_tokens=total_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    mono_owned_tokens = sum(end - start for start, end in mono_ranges)
    mono_num_blocks = (mono_owned_tokens + page_tokens - 1) // page_tokens
    mono_block_ids = [1000 + i for i in range(mono_num_blocks)]

    mono_reg = byte_spans.lower_fa_spans(
        token_base=0,
        num_tokens=total_tokens,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
        block_ids=mono_block_ids,
    )

    # 2. Chunked Lowering
    chunk_regs: list[byte_spans.PoolSpanRegistration] = []
    current_base = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        chunk_ranges = byte_spans.owned_token_ranges(
            token_base=current_base,
            num_tokens=chunk_size,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        chunk_owned_tokens = sum(end - start for start, end in chunk_ranges)
        chunk_num_blocks = (
            chunk_owned_tokens + page_tokens - 1
        ) // page_tokens
        chunk_block_ids = [
            2000 + chunk_idx * 100 + i for i in range(chunk_num_blocks)
        ]

        chunk_reg = byte_spans.lower_fa_spans(
            token_base=current_base,
            num_tokens=chunk_size,
            transfer_rank=transfer_rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=chunk_block_ids,
        )
        chunk_regs.append(chunk_reg)
        current_base += chunk_size

    # 3. Structural Equivalence Assertions
    # Declared bytes additivity
    total_chunk_declared = sum(c.declared_bytes for c in chunk_regs)
    assert total_chunk_declared == mono_reg.declared_bytes
    assert total_chunk_declared == mono_owned_tokens * token_bytes

    # Destination spans equality
    all_chunk_spans = [span for c in chunk_regs for span in c.spans]
    assert len(all_chunk_spans) == len(mono_reg.spans)

    for chunk_span, mono_span in zip(all_chunk_spans, mono_reg.spans):
        assert chunk_span.dst_block_index == mono_span.dst_block_index == 0
        assert chunk_span.dst_offset_bytes == mono_span.dst_offset_bytes
        assert chunk_span.size_bytes == mono_span.size_bytes
        assert chunk_span.src_stride_bytes == mono_span.src_stride_bytes == 0
        assert chunk_span.dst_stride_bytes == mono_span.dst_stride_bytes == 0
        assert chunk_span.count == mono_span.count == 1

    # Registration metadata invariant
    assert mono_reg.tag == "fa"
    assert mono_reg.dst_space_version == 1
    for chunk_reg in chunk_regs:
        assert chunk_reg.tag == "fa"
        assert chunk_reg.dst_space_version == 1

    # 4. Bit-for-bit Memory Copy Simulation
    mono_source_blocks = _build_populated_source_blocks(
        owned_ranges=mono_ranges,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
        num_blocks=mono_num_blocks,
    )
    dst_mono = bytearray(total_tokens * token_bytes)
    _execute_span_copy(
        registration=mono_reg,
        source_blocks=mono_source_blocks,
        dst_buffer=dst_mono,
    )

    dst_chunked = bytearray(total_tokens * token_bytes)
    _populate_and_execute_chunk_copies(
        chunk_sizes=chunk_sizes,
        chunk_regs=chunk_regs,
        transfer_rank=transfer_rank,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
        dst_buffer=dst_chunked,
    )

    assert dst_chunked == dst_mono

    return mono_reg, chunk_regs


def _verify_full_collective_decomposition(
    *,
    chunk_sizes: Sequence[int],
    parallelism: int,
    interleave_tokens: int,
    page_tokens: int,
    token_bytes: int,
) -> None:
    """Verifies span decomposition across all ranks and global coverage.

    Executes rank-level decomposition assertions across all ranks in the PCP
    collective, accumulates chunk-transferred bytes into a global destination
    buffer, and asserts that the final global buffer matches the exact expected
    token contents for the complete prompt without gaps or corruptions.

    Args:
        chunk_sizes: Sequence of token counts defining consecutive sub-chunks.
        parallelism: Context Parallelism degree (PCP size).
        interleave_tokens: Tokens assigned per rank stride.
        page_tokens: Capacity of each KV cache page block in tokens.
        token_bytes: Byte size allocated per token.
    """
    total_tokens = sum(chunk_sizes)
    global_dest_chunked = bytearray(total_tokens * token_bytes)
    global_dest_expected = bytearray(total_tokens * token_bytes)

    # Populate expected global buffer
    for token_id in range(total_tokens):
        payload = _token_payload(token_id, token_bytes)
        offset = token_id * token_bytes
        global_dest_expected[offset : offset + token_bytes] = payload

    total_collective_bytes = 0
    for rank in range(parallelism):
        _, chunk_regs = _verify_span_decomposition_for_rank(
            chunk_sizes=chunk_sizes,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
        )
        total_collective_bytes += _populate_and_execute_chunk_copies(
            chunk_sizes=chunk_sizes,
            chunk_regs=chunk_regs,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            dst_buffer=global_dest_chunked,
        )

    # Lossless global transfer: destination matches expected bit-for-bit.
    assert global_dest_chunked == global_dest_expected
    assert total_collective_bytes == total_tokens * token_bytes


def test_owned_token_ranges_aligned_strides_pcp4():
    """Verifies token range queries across PCP=4 with interleave 256."""
    parallelism = 4
    interleave_tokens = 256

    # Test window [0, 1024)
    expected_0_1024 = {
        0: [(0, 256)],
        1: [(256, 512)],
        2: [(512, 768)],
        3: [(768, 1024)],
    }
    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=0,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        assert ranges == expected_0_1024[rank]
        assert sum(end - start for start, end in ranges) == 256

    # Test window [1024, 2048)
    expected_1024_2048 = {
        0: [(1024, 1280)],
        1: [(1280, 1536)],
        2: [(1536, 1792)],
        3: [(1792, 2048)],
    }
    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        assert ranges == expected_1024_2048[rank]
        assert sum(end - start for start, end in ranges) == 256

    # Test window [2048, 4096) (2 strides of 1024)
    expected_2048_4096 = {
        0: [(2048, 2304), (3072, 3328)],
        1: [(2304, 2560), (3328, 3584)],
        2: [(2560, 2816), (3584, 3840)],
        3: [(2816, 3072), (3840, 4096)],
    }
    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=2048,
            num_tokens=2048,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        assert ranges == expected_2048_4096[rank]
        assert sum(end - start for start, end in ranges) == 512
        for start, end in ranges:
            assert end - start == 256


def test_owned_token_ranges_aligned_strides_pcp2():
    """Verifies token range queries for PCP=2 with interleave 256."""
    parallelism = 2
    interleave_tokens = 256

    # Test window [512, 1024)
    ranges_rank0 = byte_spans.owned_token_ranges(
        token_base=512,
        num_tokens=512,
        transfer_rank=0,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    ranges_rank1 = byte_spans.owned_token_ranges(
        token_base=512,
        num_tokens=512,
        transfer_rank=1,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
    )
    assert ranges_rank0 == [(512, 768)]
    assert ranges_rank1 == [(768, 1024)]


def test_owned_token_ranges_aligned_strides_pcp8():
    """Verifies token range queries for PCP=8 with interleave 128."""
    parallelism = 8
    interleave_tokens = 128
    token_base = 1024
    num_tokens = 1024

    for rank in range(parallelism):
        ranges = byte_spans.owned_token_ranges(
            token_base=token_base,
            num_tokens=num_tokens,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
        )
        expected_start = token_base + rank * interleave_tokens
        expected_end = expected_start + interleave_tokens
        assert ranges == [(expected_start, expected_end)]


def test_owned_token_ranges_negative_token_base_raises_value_error():
    """Asserts ValueError is raised when token_base < 0."""
    with pytest.raises(ValueError, match="token_base must be non-negative"):
        byte_spans.owned_token_ranges(
            token_base=-1,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="token_base must be non-negative"):
        byte_spans.owned_token_ranges(
            token_base=-1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )


def test_owned_token_ranges_unaligned_cycle_raises_value_error():
    """Asserts ValueError is raised when token_base is not cycle-aligned."""
    parallelism = 4
    interleave_tokens = 256  # cycle = 1024
    unaligned_bases = [1, 100, 256, 512, 768, 1025, 1536]
    for unaligned_base in unaligned_bases:
        with pytest.raises(ValueError, match="must be aligned to PCP cycle"):
            byte_spans.owned_token_ranges(
                token_base=unaligned_base,
                num_tokens=1024,
                transfer_rank=0,
                parallelism=parallelism,
                interleave_tokens=interleave_tokens,
            )


def test_owned_token_ranges_invalid_arguments_raise_value_error():
    """Asserts ValueError is raised for invalid domain parameters."""
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=0,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="parallelism must be positive"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=0,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="outside parallelism"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=4,
            parallelism=4,
            interleave_tokens=256,
        )

    with pytest.raises(ValueError, match="interleave_tokens must be positive"):
        byte_spans.owned_token_ranges(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=0,
        )


def test_owned_token_ranges_backward_compatibility():
    """Verifies default token_base=0 matches explicit token_base=0."""
    ranges_default = byte_spans.owned_token_ranges(
        num_tokens=1024,
        transfer_rank=1,
        parallelism=4,
        interleave_tokens=256,
    )
    ranges_explicit = byte_spans.owned_token_ranges(
        token_base=0,
        num_tokens=1024,
        transfer_rank=1,
        parallelism=4,
        interleave_tokens=256,
    )
    assert ranges_default == ranges_explicit == [(256, 512)]


def test_owned_token_ranges_parallelism_one():
    """Verifies parallelism=1 returns full token range offset by token_base."""
    ranges = byte_spans.owned_token_ranges(
        token_base=1024,
        num_tokens=2048,
        transfer_rank=0,
        parallelism=1,
        interleave_tokens=256,
    )
    assert ranges == [(1024, 3072)]


def test_lower_fa_spans_destination_offsets():
    """Verifies sub-chunk destination byte offsets match monolithic slice."""
    parallelism = 4
    interleave_tokens = 256
    page_tokens = 1024
    token_bytes = 1024  # 1 KB per token

    for rank in range(parallelism):
        mono_reg = byte_spans.lower_fa_spans(
            token_base=0,
            num_tokens=2048,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[100],
        )

        chunk0_reg = byte_spans.lower_fa_spans(
            token_base=0,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[200],
        )

        chunk1_reg = byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[300],
        )

        assert len(mono_reg.spans) == 2
        assert len(chunk0_reg.spans) == 1
        assert len(chunk1_reg.spans) == 1

        # Chunk 0 span matches monolithic span 0 dst_offset_bytes & size_bytes.
        assert (
            chunk0_reg.spans[0].dst_offset_bytes
            == mono_reg.spans[0].dst_offset_bytes
        )
        assert chunk0_reg.spans[0].size_bytes == mono_reg.spans[0].size_bytes
        assert chunk0_reg.spans[0].src_block_ordinal == 0
        assert chunk0_reg.spans[0].src_offset_bytes == 0
        assert chunk0_reg.spans[0].dst_block_index == 0
        assert chunk0_reg.spans[0].size_bytes == 256 * token_bytes
        assert chunk0_reg.declared_bytes == 256 * token_bytes
        assert chunk0_reg.dst_space_version == 1
        assert chunk0_reg.tag == "fa"
        assert chunk0_reg.block_ids == (200,)

        # Chunk 1 span matches monolithic span 1 dst_offset_bytes & size_bytes.
        assert (
            chunk1_reg.spans[0].dst_offset_bytes
            == mono_reg.spans[1].dst_offset_bytes
        )
        assert chunk1_reg.spans[0].size_bytes == mono_reg.spans[1].size_bytes
        assert chunk1_reg.spans[0].src_block_ordinal == 0
        assert chunk1_reg.spans[0].src_offset_bytes == 0
        assert chunk1_reg.spans[0].dst_block_index == 0
        expected_dst_offset = (1024 + rank * 256) * token_bytes
        assert chunk1_reg.spans[0].dst_offset_bytes == expected_dst_offset
        assert chunk1_reg.spans[0].size_bytes == 256 * token_bytes
        assert chunk1_reg.declared_bytes == 256 * token_bytes
        assert chunk1_reg.dst_space_version == 1
        assert chunk1_reg.tag == "fa"
        assert chunk1_reg.block_ids == (300,)


def test_lower_fa_spans_backward_compatibility():
    """Verifies default token_base=0 matches explicit token_base=0 in spans."""
    parallelism = 4
    interleave_tokens = 256
    page_tokens = 1024
    token_bytes = 512

    for rank in range(parallelism):
        reg_default = byte_spans.lower_fa_spans(
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[100 + rank],
        )

        reg_explicit = byte_spans.lower_fa_spans(
            token_base=0,
            num_tokens=1024,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
            block_ids=[100 + rank],
        )

        assert reg_default == reg_explicit
        assert reg_default.declared_bytes == 256 * token_bytes
        assert reg_default.dst_space_version == 1
        assert reg_default.tag == "fa"
        assert reg_default.block_ids == (100 + rank,)
        assert len(reg_default.spans) == 1
        assert reg_default.spans[0].src_block_ordinal == 0
        assert reg_default.spans[0].src_offset_bytes == 0
        assert reg_default.spans[0].dst_block_index == 0
        assert (
            reg_default.spans[0].dst_offset_bytes
            == rank * interleave_tokens * token_bytes
        )
        assert reg_default.spans[0].size_bytes == 256 * token_bytes


def test_lower_fa_spans_parallelism_one():
    """Verifies lower_fa_spans for parallelism=1 with token_base > 0."""
    reg = byte_spans.lower_fa_spans(
        token_base=2048,
        num_tokens=2048,
        transfer_rank=0,
        parallelism=1,
        interleave_tokens=256,
        page_tokens=1024,
        token_bytes=512,
        block_ids=[10, 11],
    )
    assert reg.tag == "fa"
    assert reg.block_ids == (10, 11)
    assert reg.declared_bytes == 2048 * 512
    assert reg.dst_space_version == 1
    assert reg.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=2048 * 512,
            size_bytes=1024 * 512,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=3072 * 512,
            size_bytes=1024 * 512,
        ),
    )


def test_lower_fa_spans_multi_page_pagination():
    """Verifies lower_fa_spans splits spans across multiple source pages.

    4096 tokens starting at token_base=1024 on PCP=4 with interleave 256.
    Each rank owns 4 * 256 = 1024 tokens.
    With page_tokens=512, each rank allocates 2 pages (block_ids=[101, 102]).
    """
    token_bytes = 1024
    reg = byte_spans.lower_fa_spans(
        token_base=1024,
        num_tokens=4096,
        transfer_rank=0,
        parallelism=4,
        interleave_tokens=256,
        page_tokens=512,
        token_bytes=token_bytes,
        block_ids=[101, 102],
    )
    assert reg.tag == "fa"
    assert reg.block_ids == (101, 102)
    assert reg.declared_bytes == 1024 * token_bytes
    assert reg.dst_space_version == 1
    assert len(reg.spans) == 4

    # Rank 0 owned ranges:
    # [(1024, 1280), (2048, 2304), (3072, 3328), (4096, 4352)]
    # First 512 owned tokens map to source page 0 (block ordinal 0)
    # Next 512 owned tokens map to source page 1 (block ordinal 1)
    assert reg.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=1024 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=256 * token_bytes,
            dst_block_index=0,
            dst_offset_bytes=2048 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=3072 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=256 * token_bytes,
            dst_block_index=0,
            dst_offset_bytes=4096 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
    )


def test_lower_fa_spans_page_boundary_crossing_within_stride():
    """Verifies lower_fa_spans splits a stride across source pages.

    1024 tokens at token_base=1024 on PCP=2 with interleave 512.
    Each rank owns 512 tokens.
    With page_tokens=256, each 512-token stride crosses 2 source pages.
    """
    token_bytes = 512
    reg = byte_spans.lower_fa_spans(
        token_base=1024,
        num_tokens=1024,
        transfer_rank=0,
        parallelism=2,
        interleave_tokens=512,
        page_tokens=256,
        token_bytes=token_bytes,
        block_ids=[201, 202],
    )
    assert reg.tag == "fa"
    assert reg.block_ids == (201, 202)
    assert reg.declared_bytes == 512 * token_bytes
    assert reg.dst_space_version == 1
    assert len(reg.spans) == 2
    assert reg.spans == (
        byte_spans.PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=1024 * token_bytes,
            size_bytes=256 * token_bytes,
        ),
        byte_spans.PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=(1024 + 256) * token_bytes,
            size_bytes=256 * token_bytes,
        ),
    )


def test_lower_fa_spans_validation_failures():
    """Verifies lower_fa_spans input validation and error rejection."""
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=0,
            token_bytes=512,
            block_ids=[1],
        )

    with pytest.raises(ValueError, match="token_bytes must be positive"):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=0,
            block_ids=[1],
        )

    with pytest.raises(ValueError, match="token_base must be non-negative"):
        byte_spans.lower_fa_spans(
            token_base=-1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[1],
        )

    with pytest.raises(ValueError, match="must be aligned to PCP cycle"):
        byte_spans.lower_fa_spans(
            token_base=512,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[1],
        )

    # Test block_ids count mismatch: expected 1 block, passed 0 or 2 blocks
    with pytest.raises(
        ValueError, match="block_ids do not match dense PCP rank packing"
    ):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[],
        )

    with pytest.raises(
        ValueError, match="block_ids do not match dense PCP rank packing"
    ):
        byte_spans.lower_fa_spans(
            token_base=1024,
            num_tokens=1024,
            transfer_rank=0,
            parallelism=4,
            interleave_tokens=256,
            page_tokens=1024,
            token_bytes=512,
            block_ids=[1, 2],
        )


@pytest.mark.parametrize(
    "prompt_tokens",
    [2048, 4096, 6144, 8192, 12288, 16384],
)
@pytest.mark.parametrize("page_tokens", [256, 512, 1024])
def test_span_decomposition_additivity_pcp4(
    prompt_tokens: int, page_tokens: int
):
    """Verifies that sub-chunk span lowering decomposes losslessly under PCP=4.

    Prompt lengths ranging from 2,048 to 16,384 tokens are partitioned into
    uniform 1024-token chunks under PCP=4 with interleave 256. For every rank
    r in [0, 4), asserts that the union of chunk registrations equals the
    monolithic registration.
    """
    parallelism = 4
    interleave_tokens = 256
    token_bytes = 512
    chunk_size = 1024
    num_chunks = prompt_tokens // chunk_size
    chunk_sizes = [chunk_size] * num_chunks

    _verify_full_collective_decomposition(
        chunk_sizes=chunk_sizes,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
    )


@pytest.mark.parametrize(
    "prompt_tokens",
    [2048, 4096, 8192, 16384],
)
@pytest.mark.parametrize("page_tokens", [256, 512, 1024])
def test_span_decomposition_additivity_pcp8(
    prompt_tokens: int, page_tokens: int
):
    """Verifies that sub-chunk span lowering decomposes losslessly under PCP=8.

    Prompt lengths ranging from 2,048 to 16,384 tokens are partitioned into
    uniform 2048-token chunks under PCP=8 with interleave 256. For every rank
    r in [0, 8), asserts that the union of chunk registrations equals the
    monolithic registration.
    """
    parallelism = 8
    interleave_tokens = 256
    token_bytes = 512
    chunk_size = 2048
    num_chunks = prompt_tokens // chunk_size
    chunk_sizes = [chunk_size] * num_chunks

    _verify_full_collective_decomposition(
        chunk_sizes=chunk_sizes,
        parallelism=parallelism,
        interleave_tokens=interleave_tokens,
        page_tokens=page_tokens,
        token_bytes=token_bytes,
    )


def test_heterogeneous_stride_boundaries():
    """Verifies span decomposition across non-uniform stride boundaries."""
    test_cases = [
        # (parallelism, interleave_tokens, page_tokens, token_bytes, chunks)
        (4, 256, 512, 1024, [1024, 2048, 3072, 4096]),
        (4, 256, 1024, 512, [2048, 1024, 2048, 3072, 1024]),
        (4, 256, 256, 512, [3072, 1024, 2048, 4096, 2048]),
        (4, 256, 512, 256, [1024, 1024, 1024, 2048, 1024]),
        (8, 256, 512, 512, [2048, 4096, 2048, 6144]),
        (8, 256, 1024, 1024, [4096, 2048, 4096, 2048]),
        (8, 128, 512, 512, [1024, 2048, 3072, 1024]),
        (2, 512, 512, 512, [1024, 2048, 1024, 3072]),
    ]

    for (
        parallelism,
        interleave_tokens,
        page_tokens,
        token_bytes,
        chunk_sizes,
    ) in test_cases:
        _verify_full_collective_decomposition(
            chunk_sizes=chunk_sizes,
            parallelism=parallelism,
            interleave_tokens=interleave_tokens,
            page_tokens=page_tokens,
            token_bytes=token_bytes,
        )
