# SPDX-License-Identifier: Apache-2.0
"""Focused tests for native Stage-3 byte-span lowering."""

from vllm_torchtpu.distributed.kv_transfer.v2.pool_byte_spans import (
    PoolByteSpan, lower_gdn_state_shard_spans)
from vllm_torchtpu.distributed.kv_transfer.v2.raiden_pool_manifest import \
    RegionSpec

_CONV_REGIONS = (
    RegionSpec(name="gdn_conv_q",
               offset_bytes=0,
               stride_bytes=32,
               unit_bytes=4,
               num_units=3,
               units_per_stride=2),
    RegionSpec(name="gdn_conv_k",
               offset_bytes=8,
               stride_bytes=32,
               unit_bytes=4,
               num_units=3,
               units_per_stride=2),
    RegionSpec(name="gdn_conv_v",
               offset_bytes=16,
               stride_bytes=32,
               unit_bytes=4,
               num_units=3,
               units_per_stride=4),
)
_SSM_REGIONS = (RegionSpec(name="gdn_ssm",
                           offset_bytes=0,
                           stride_bytes=16,
                           unit_bytes=16,
                           num_units=4), )


def _copy_registration(destination: bytearray, source: bytes,
                       registration) -> None:
    for span in registration.spans:
        for repeat in range(span.count):
            src = span.src_offset_bytes + repeat * span.src_stride_bytes
            dst = span.dst_offset_bytes + repeat * span.dst_stride_bytes
            destination[dst:dst + span.size_bytes] = source[src:src +
                                                            span.size_bytes]


def test_gdn_conv_shards_reassemble_qkv_head_order():
    parallelism = 8
    destination = bytearray(3 * 32 * parallelism)
    expected = bytearray()
    rank_sources = []
    for rank in range(parallelism):
        source = bytearray()
        for tap in range(3):
            source.extend(bytes([10 + rank]) * 8)  # q heads
            source.extend(bytes([30 + rank]) * 8)  # k heads
            source.extend(bytes([50 + rank]) * 16)  # v heads
        rank_sources.append(bytes(source))

    for tap in range(3):
        expected.extend(b"".join(
            bytes([10 + rank]) * 8 for rank in range(parallelism)))
        expected.extend(b"".join(
            bytes([30 + rank]) * 8 for rank in range(parallelism)))
        expected.extend(b"".join(
            bytes([50 + rank]) * 16 for rank in range(parallelism)))

    for rank, source in enumerate(rank_sources):
        registration = lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=17,
            transfer_rank=rank,
            parallelism=parallelism,
            regions=_CONV_REGIONS,
        )
        assert registration.block_ids == (17, )
        assert registration.declared_bytes == len(source) == 96
        assert all(
            span.src_offset_bytes + (span.count - 1) * span.src_stride_bytes +
            span.size_bytes <= len(source) for span in registration.spans)
        _copy_registration(destination, source, registration)

    assert destination == expected


def test_gdn_ssm_shards_reassemble_contiguous_heads():
    parallelism = 8
    destination = bytearray(64 * parallelism)
    for rank in range(parallelism):
        source = bytes([rank + 1]) * 64
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
            dst_offset_bytes=rank * 64,
            size_bytes=64,
        ), )
        assert registration.declared_bytes == 64
        _copy_registration(destination, source, registration)

    assert destination == b"".join(
        bytes([rank + 1]) * 64 for rank in range(parallelism))


def test_gdn_conv_rank_three_span_geometry():
    registration = lower_gdn_state_shard_spans(
        tag="gdn.conv.g1",
        block_id=9,
        transfer_rank=3,
        parallelism=8,
        regions=_CONV_REGIONS,
    )

    assert registration.spans == (
        PoolByteSpan(0, 0, 0, 24, 8, 32, 256, 3),
        PoolByteSpan(0, 8, 0, 88, 8, 32, 256, 3),
        PoolByteSpan(0, 16, 0, 176, 16, 32, 256, 3),
    )


def test_qwen35_35b_pcp8_rank_three_state_offsets():
    conv = lower_gdn_state_shard_spans(
        tag="gdn.conv.g0",
        block_id=5,
        transfer_rank=3,
        parallelism=8,
        regions=(
            RegionSpec("gdn_conv_q", 0, 2048, 256, 3, 2),
            RegionSpec("gdn_conv_k", 512, 2048, 256, 3, 2),
            RegionSpec("gdn_conv_v", 1024, 2048, 256, 3, 4),
        ),
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
        PoolByteSpan(0, 0, 0, 1_536, 512, 2_048, 16_384, 3),
        PoolByteSpan(0, 512, 0, 5_632, 512, 2_048, 16_384, 3),
        PoolByteSpan(0, 1_024, 0, 11_264, 1_024, 2_048, 16_384, 3),
    )
    assert ssm.declared_bytes == 262_144
    assert ssm.spans == (PoolByteSpan(0, 0, 0, 786_432, 262_144), )


def test_gdn_lowering_rejects_non_dense_conv_layout():
    bad_regions = list(_CONV_REGIONS)
    bad_regions[1] = RegionSpec(name="gdn_conv_k",
                                offset_bytes=12,
                                stride_bytes=36,
                                unit_bytes=4,
                                num_units=3,
                                units_per_stride=2)

    try:
        lower_gdn_state_shard_spans(tag="gdn.conv.g0",
                                    block_id=0,
                                    transfer_rank=0,
                                    parallelism=8,
                                    regions=bad_regions)
    except ValueError as exc:
        assert "dense tap-major" in str(exc)
    else:
        raise AssertionError("non-dense conv layout was accepted")
