# SPDX-License-Identifier: Apache-2.0
"""Focused tests for native Stage-3 byte-span lowering.

Every extent the lowering emits must be a whole multiple of its class's
physical granule. For raw GDN spans that granule is the 1024-byte pool
token, so those fixtures use real token-scaled geometries: the
Qwen3.5-35B TP8 shard for the QK pair-blocked vocabulary and a TP4 shard
for the legacy segment-major vocabulary. For GLM-5.2 the granule is the
packed row (4 tokens: 2560 B of latent, 1024 B of indexer).
"""

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    PoolByteSpan,
    lower_gdn_state_shard_spans,
    lower_glm_row_spans,
    lower_kda_state_shard_spans,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
    measured_kimi_k3_layout_fingerprint,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import RegionSpec

from .raiden_test_utils import kimi_pool_manifest
from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

pytestmark = pytest.mark.cpu_test

# Qwen3.5-35B TP8 shard, QK pair-blocked: one 1024-byte QK token plus one
# 1024-byte V token per tap.
_CONV_REGIONS_PAIR_TP8 = (
    RegionSpec(
        name="gdn_conv_qk",
        offset_bytes=0,
        stride_bytes=2048,
        unit_bytes=1024,
        num_units=3,
        units_per_stride=1,
    ),
    RegionSpec(
        name="gdn_conv_v",
        offset_bytes=1024,
        stride_bytes=2048,
        unit_bytes=256,
        num_units=3,
        units_per_stride=4,
    ),
)

# Qwen3.5-35B TP4 shard, legacy segment-major: 1024-byte Q and K segments
# and a 4096-byte V segment per tap — all whole tokens on their own.
_CONV_REGIONS_LEGACY_TP4 = (
    RegionSpec(
        name="gdn_conv_q",
        offset_bytes=0,
        stride_bytes=6144,
        unit_bytes=256,
        num_units=3,
        units_per_stride=4,
    ),
    RegionSpec(
        name="gdn_conv_k",
        offset_bytes=1024,
        stride_bytes=6144,
        unit_bytes=256,
        num_units=3,
        units_per_stride=4,
    ),
    RegionSpec(
        name="gdn_conv_v",
        offset_bytes=2048,
        stride_bytes=6144,
        unit_bytes=256,
        num_units=3,
        units_per_stride=16,
    ),
)

# Qwen3.5-35B TP8 shard, legacy segment-major: the 512-byte Q/K half-token
# segments are not placement-exact and must be rejected.
_CONV_REGIONS_LEGACY_TP8 = (
    RegionSpec(
        name="gdn_conv_q",
        offset_bytes=0,
        stride_bytes=2048,
        unit_bytes=256,
        num_units=3,
        units_per_stride=2,
    ),
    RegionSpec(
        name="gdn_conv_k",
        offset_bytes=512,
        stride_bytes=2048,
        unit_bytes=256,
        num_units=3,
        units_per_stride=2,
    ),
    RegionSpec(
        name="gdn_conv_v",
        offset_bytes=1024,
        stride_bytes=2048,
        unit_bytes=256,
        num_units=3,
        units_per_stride=4,
    ),
)

_SSM_REGIONS = (
    RegionSpec(
        name="gdn_ssm", offset_bytes=0, stride_bytes=1024, unit_bytes=1024, num_units=4
    ),
)


def _copy_registration(destination: bytearray, source: bytes, registration) -> None:
    for span in registration.spans:
        for repeat in range(span.count):
            src = span.src_offset_bytes + repeat * span.src_stride_bytes
            dst = span.dst_offset_bytes + repeat * span.dst_stride_bytes
            destination[dst : dst + span.size_bytes] = source[
                src : src + span.size_bytes
            ]


def _kda_physical_state(state, pool, shard_heads=None):
    if pool.tag.startswith("gdn.ssm"):
        return state.astype("<f4").tobytes()
    # Independent XLA nested-tiling oracle over the kernel's logical conv
    # tensor. Every tap, Q/K/V, head and lane carries different data.
    heads = state.shape[2]
    shard_heads = heads if shard_heads is None else shard_heads
    shard_capacity = pool.live_bytes_per_block // 2 // (heads // shard_heads)
    chunks = []
    for start in range(0, heads, shard_heads):
        flat = state[:, :, start : start + shard_heads].astype("<u2").reshape(-1)
        chunks.append(np.pad(flat, (0, shard_capacity - flat.size)))
    padded = np.concatenate(chunks).reshape(-1, 2, 640)
    return _apply_xla_tiled_layout(padded, (2, 1, 0), ((2, 128), (2, 1))).tobytes()


@pytest.mark.parametrize("parallelism,dst_shards", [(32, 32), (32, 8)])
def test_kda_reshard_matches_rank_blocked_kernel_layout(parallelism, dst_shards):
    local_heads = 96 // parallelism
    source_manifest = kimi_pool_manifest(local_heads)
    fan_in = parallelism // dst_shards
    destination_manifest = kimi_pool_manifest(local_heads * fan_in)
    fingerprints = []
    for manifest, fragments in ((source_manifest, 1), (destination_manifest, fan_in)):
        assert [pool.tag for pool in manifest.pools] == [
            "gdn.conv.g0",
            "gdn.ssm.g0",
            "fa",
        ]
        assert len(manifest.storages) == 1
        conv, ssm, _ = manifest.pools
        assert conv.base_offset_bytes >= ssm.live_bytes_per_block
        assert all(
            pool.storage_index == 0
            and pool.num_blocks == 4
            and pool.base_offset_bytes + pool.regions[0].extent_end_bytes
            <= pool.block_stride_bytes
            for pool in manifest.pools
        )
        fingerprints.append(
            measured_kimi_k3_layout_fingerprint(
                manifest,
                page_tokens=manifest.storages[0].shape[1] * 2,
                state_fragment_heads=local_heads,
                state_fragments=fragments,
                layout_getter=lambda tensor: ([3, 2, 1, 0], [[2, 128], [2, 1]], 0),
                package_version=lambda package: "test",
            )
        )
    assert fingerprints[0] == fingerprints[1]
    for source_pool, dest_pool, shape in zip(
        source_manifest.pools[:2],
        destination_manifest.pools[:2],
        ((3, 3, 96, 128), (96, 128, 128)),
    ):
        global_state = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
        destinations = [
            bytearray(dest_pool.live_bytes_per_block) for _ in range(dst_shards)
        ]
        coverage = [
            np.zeros(dest_pool.live_bytes_per_block, dtype=np.uint8)
            for _ in range(dst_shards)
        ]
        is_conv = source_pool.tag.startswith("gdn.conv")
        for rank in range(parallelism):
            head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
            state = (
                global_state[:, :, head_slice] if is_conv else global_state[head_slice]
            )
            source = _kda_physical_state(state, source_pool)
            assert len(source) == source_pool.live_bytes_per_block
            registration = lower_kda_state_shard_spans(
                tag=source_pool.tag,
                block_id=17,
                transfer_rank=rank,
                parallelism=parallelism,
                dst_shards=dst_shards,
                regions=source_pool.regions,
            )
            dst_rank = rank // fan_in
            assert registration.declared_bytes == len(source)
            # One native chunk per source/pool/block, including heterogeneous
            # convolution state: never a half-word list exceeding IOV_MAX.
            (span,) = registration.spans
            assert span.count == 1
            assert span.dst_unit_ordinal == dst_rank
            offset = span.dst_offset_bytes
            coverage[dst_rank][offset : offset + span.size_bytes] += 1
            _copy_registration(destinations[dst_rank], source, registration)
        for rank, destination in enumerate(destinations):
            head_slice = slice(
                rank * local_heads * fan_in, (rank + 1) * local_heads * fan_in
            )
            state = (
                global_state[:, :, head_slice] if is_conv else global_state[head_slice]
            )
            assert destination == _kda_physical_state(state, dest_pool, local_heads)
            np.testing.assert_array_equal(coverage[rank], 1)


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
        expected.extend(
            b"".join(bytes([10 + rank]) * 1024 for rank in range(parallelism))
        )
        expected.extend(
            b"".join(bytes([50 + rank]) * 1024 for rank in range(parallelism))
        )

    for rank, source in enumerate(rank_sources):
        registration = lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=17,
            transfer_rank=rank,
            parallelism=parallelism,
            regions=_CONV_REGIONS_PAIR_TP8,
        )
        assert registration.block_ids == (17,)
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
        expected.extend(
            b"".join(bytes([10 + rank]) * 1024 for rank in range(parallelism))
        )
        expected.extend(
            b"".join(bytes([30 + rank]) * 1024 for rank in range(parallelism))
        )
        expected.extend(
            b"".join(bytes([50 + rank]) * 4096 for rank in range(parallelism))
        )

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
        assert registration.spans == (
            PoolByteSpan(
                src_block_ordinal=0,
                src_offset_bytes=0,
                dst_block_index=0,
                dst_offset_bytes=rank * 4096,
                size_bytes=4096,
            ),
        )
        assert registration.declared_bytes == 4096
        _copy_registration(destination, source, registration)

    assert destination == b"".join(
        bytes([rank + 1]) * 4096 for rank in range(parallelism)
    )


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
        regions=(RegionSpec("gdn_ssm", 0, 65_536, 65_536, 4),),
    )

    assert conv.declared_bytes == 6_144
    assert conv.spans == (
        PoolByteSpan(0, 0, 0, 3_072, 1_024, 2_048, 16_384, 3),
        PoolByteSpan(0, 1_024, 0, 11_264, 1_024, 2_048, 16_384, 3),
    )
    assert ssm.declared_bytes == 262_144
    assert ssm.spans == (PoolByteSpan(0, 0, 0, 786_432, 262_144),)


def test_gdn_legacy_subtoken_qk_segments_rejected():
    try:
        lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=0,
            transfer_rank=0,
            parallelism=8,
            regions=_CONV_REGIONS_LEGACY_TP8,
        )
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
            regions=(RegionSpec("gdn_ssm", 0, 64, 64, 4),),
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
    bad_regions[1] = RegionSpec(
        name="gdn_conv_k",
        offset_bytes=2048,
        stride_bytes=6144,
        unit_bytes=256,
        num_units=3,
        units_per_stride=4,
    )

    try:
        lower_gdn_state_shard_spans(
            tag="gdn.conv.g0",
            block_id=0,
            transfer_rank=0,
            parallelism=4,
            regions=bad_regions,
        )
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
        PoolByteSpan(
            src_block_ordinal=0,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=0,
            size_bytes=_GLM_FA_LIVE,
        ),
        PoolByteSpan(
            src_block_ordinal=2,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=2 * _GLM_FA_LIVE,
            size_bytes=_GLM_FA_LIVE,
        ),
    )
    assert rank1.spans == (
        PoolByteSpan(
            src_block_ordinal=1,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=_GLM_FA_LIVE,
            size_bytes=_GLM_FA_LIVE,
        ),
    )
    assert rank0.declared_bytes == 2 * _GLM_FA_LIVE
    assert rank1.declared_bytes == _GLM_FA_LIVE


def test_glm_row_spans_trim_partial_tail_to_rows():
    for tag, live, row_bytes in (
        ("fa", _GLM_FA_LIVE, _GLM_FA_ROW_BYTES),
        ("dsa.idx", _GLM_IDX_LIVE, _GLM_IDX_ROW_BYTES),
    ):
        registration = _glm_row_spans(
            tag=tag,
            num_tokens=2 * _GLM_PAGE_TOKENS + 5,
            live_bytes_per_block=live,
            row_bytes=row_bytes,
        )
        # Rank 0 of 2 owns pages {0, 2}; 5 tail tokens round up to 2 rows.
        assert [span.size_bytes for span in registration.spans] == [live, 2 * row_bytes]
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
    assert [span.size_bytes for span in registration.spans] == [
        nope_live,
        5 * nope_row_bytes,
    ]
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
    assert [span.size_bytes for span in registration.spans] == [
        rope_live,
        2 * rope_row_bytes,
    ]
    assert registration.declared_bytes == rope_live + 2 * rope_row_bytes


def test_glm_row_spans_parallelism_one_owns_every_page():
    registration = _glm_row_spans(
        num_tokens=2 * _GLM_PAGE_TOKENS, parallelism=1, block_ids=[7, 9]
    )
    assert [span.src_block_ordinal for span in registration.spans] == [0, 1]
    assert [span.size_bytes for span in registration.spans] == [
        _GLM_FA_LIVE,
        _GLM_FA_LIVE,
    ]


def test_glm_row_spans_rank_owning_no_pages_keeps_block_ids():
    registration = _glm_row_spans(
        num_tokens=100, transfer_rank=5, parallelism=8, block_ids=[3]
    )
    assert registration.spans == ()
    assert registration.declared_bytes == 0
    assert registration.block_ids == (3,)


def test_glm_row_spans_union_partitions_request_bytes():
    num_pages, parallelism = 11, 8
    num_tokens = (num_pages - 1) * _GLM_PAGE_TOKENS + 7
    owners: dict[int, int] = {}
    total = 0
    for rank in range(parallelism):
        registration = _glm_row_spans(
            num_tokens=num_tokens,
            transfer_rank=rank,
            parallelism=parallelism,
            block_ids=list(range(num_pages)),
        )
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
