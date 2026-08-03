# SPDX-License-Identifier: Apache-2.0
"""Focused tests for native Stage-3 byte-span lowering.

Raw GDN spans move physical pool bytes, so every extent the lowering
emits must be a whole multiple of the 1024-byte physical pool token.
Fixtures therefore use real token-scaled geometries: the Qwen3.5-35B
TP8 shard for the QK pair-blocked vocabulary and a TP4 shard for the
legacy segment-major vocabulary (whose per-segment extents are already
token multiples at that degree).
"""

from vllm_torchtpu.distributed.kv_transfer.v2.pool_byte_spans import (
    PoolByteSpan, lower_gdn_state_shard_spans)
from vllm_torchtpu.distributed.kv_transfer.v2.raiden_pool_manifest import \
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
