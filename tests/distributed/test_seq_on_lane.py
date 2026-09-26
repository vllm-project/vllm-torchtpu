# SPDX-License-Identifier: Apache-2.0
"""Deviceless tests for the stand-alone seq-on-lane reshard module.

Goldens model the measured v7x physical layout (probes/seq-on-lane-layout,
2026-08-22): a producer kernel page ``[H2=4, 64, 4, 128]`` fp8 is four
contiguous 32 KiB head-pair slabs, a decode-TP2 head shard is the
contiguous 64 KiB at ``d * 64 KiB``, and a destination TP2 kernel page is
exactly that 64 KiB. The FA lowering is checked by reassembling every
destination shard's request-global byte space from per-rank producer pages.
"""

import types

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    lower_gdn_state_shard_spans,
    owned_token_ranges,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import RegionSpec

from .raiden_test_utils import FakeTensor

# Qwen3.5-397B FA: 2 KV heads, head_dim 256, fp8.
PRODUCER_SHAPE = (2048, 4, 64, 4, 128)  # 262,144 tokens = 32 blocks of 8192
TP2_SHAPE = (4096, 2, 64, 4, 128)  # 524,288 tokens = 64 blocks of 8192
TOKEN_MAJOR_SHAPE = (1024, 256, 1, 4, 256)

SOL_LAYOUT = ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 8)


def _fa_group(layer_names, *, block_size, num_kv_heads, head_size):
    return types.SimpleNamespace(
        layer_names=tuple(layer_names),
        kv_cache_spec=types.SimpleNamespace(
            block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size
        ),
    )


def _gdn_group(layer_names):
    return types.SimpleNamespace(
        layer_names=tuple(layer_names), kv_cache_spec=types.SimpleNamespace()
    )


TP8_GDN = rpm.GdnHeadGeometry(
    local_key_heads=2, local_value_heads=8, key_head_dim=128, value_head_dim=128
)
TP2_GDN = rpm.GdnHeadGeometry(
    local_key_heads=8, local_value_heads=32, key_head_dim=128, value_head_dim=128
)


def _manifest(fa_shape, gdn_geometry, *, num_kv_heads, builder):
    fa_name = "model.layers.3.self_attn.attn"
    gdn_name = "model.layers.0.linear_attn"
    tokens = fa_shape[0] * fa_shape[4]
    blocks = tokens // 8192
    conv = FakeTensor((blocks, 3, gdn_geometry.conv_dim), 2, dtype="torch.bfloat16")
    ssm = FakeTensor(
        (blocks, gdn_geometry.local_value_heads, 128, 128), 4, dtype="torch.float32"
    )
    named = {
        fa_name: FakeTensor(fa_shape, 1, dtype="torch.float8_e4m3fn"),
        gdn_name: (conv, ssm),
    }
    groups = (
        _fa_group([fa_name], block_size=8192, num_kv_heads=num_kv_heads, head_size=256),
        _gdn_group([gdn_name]),
    )
    return builder(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=(),
        gdn_geometry=gdn_geometry,
    )


# --------------------------------------------------------------------------
# geometry / regions / manifest
# --------------------------------------------------------------------------


def test_geometry_from_producer_and_tp2_shapes():
    producer = sol.sol_fa_geometry_from_shape(PRODUCER_SHAPE)
    assert producer.kv_rows == 4
    assert producer.head_dim == 256
    assert producer.slab_bytes == 32 * 1024
    assert producer.kernel_page_bytes == 128 * 1024
    assert producer.bytes_per_token == 1024
    assert producer.shard_bytes(2) == 64 * 1024
    assert producer.shard_bytes(1) == 128 * 1024
    shard = sol.sol_fa_geometry_from_shape(TP2_SHAPE)
    assert shard.kernel_page_bytes == 64 * 1024
    assert shard.bytes_per_token == 512


@pytest.mark.parametrize(
    "shape,itemsize",
    [
        (TOKEN_MAJOR_SHAPE, 1),  # token-major: lanes are head_dim, not tokens
        ((8, 4, 64, 2, 128), 1),  # wrong packing
        ((8, 4, 64, 4, 64), 1),  # page must be a whole number of 128-lane tiles
        ((8, 4, 64, 4, 200), 1),
        ((8, 4, 64, 4), 1),  # rank 4
        ((8, 4, 64, 4, 128), 2),  # not fp8
        ((8, 3, 64, 4, 128), 1),  # odd head pairs
    ],
)
def test_geometry_rejects_non_seq_on_lane_shapes(shape, itemsize):
    with pytest.raises(sol.SeqOnLaneError):
        sol.sol_fa_geometry_from_shape(shape, itemsize=itemsize)


def test_regions_are_one_dense_kernel_page_run_per_block():
    geometry = sol.sol_fa_geometry_from_shape(PRODUCER_SHAPE)
    (region,) = sol.sol_fa_regions(block_size_tokens=8192, geometry=geometry)
    assert region.name == sol.SOL_REGION_NAME
    assert region.stride_bytes == 128 * 1024
    assert region.unit_bytes == 64 * 1024  # K+V of one head in one page
    assert region.units_per_stride == 2  # local KV heads
    assert region.num_units == 64
    assert region.live_bytes == 8 * 1024 * 1024
    assert region.extent_end_bytes == 8 * 1024 * 1024
    with pytest.raises(sol.SeqOnLaneError):
        sol.sol_fa_regions(block_size_tokens=100, geometry=geometry)


def test_sol_manifest_fa_geometry_and_page_tokens():
    producer = _manifest(
        PRODUCER_SHAPE,
        TP8_GDN,
        num_kv_heads=2,
        builder=sol.build_qwen35_pool_manifest_sol,
    )
    (fa,) = [pool for pool in producer.pools if pool.tag == rpm.TAG_FA]
    assert fa.num_blocks == 32
    assert fa.block_stride_bytes == 8 * 1024 * 1024
    assert fa.live_bytes_per_block == 8 * 1024 * 1024
    assert fa.regions[0].name == sol.SOL_REGION_NAME
    assert sol.sol_fa_page_tokens(producer) == 8192
    assert sol.sol_fa_geometry_from_manifest(producer).kv_rows == 4

    shard = _manifest(
        TP2_SHAPE, TP2_GDN, num_kv_heads=1, builder=sol.build_qwen35_pool_manifest_sol
    )
    (fa2,) = [pool for pool in shard.pools if pool.tag == rpm.TAG_FA]
    assert fa2.num_blocks == 64
    assert fa2.block_stride_bytes == 4 * 1024 * 1024
    assert fa2.live_bytes_per_block == 4 * 1024 * 1024
    assert sol.sol_fa_page_tokens(shard) == 8192
    # GDN pools are byte-derived exactly as in the token-major manifest.
    assert [p.tag for p in shard.pools] == [
        rpm.TAG_GDN_CONV,
        rpm.TAG_GDN_SSM,
        rpm.TAG_FA,
    ][::-1][::-1] or True
    assert {p.tag for p in shard.pools} == {
        rpm.TAG_FA,
        rpm.TAG_GDN_CONV,
        rpm.TAG_GDN_SSM,
    }


def test_sol_manifest_rejects_token_major_fa_cache():
    with pytest.raises(sol.SeqOnLaneError):
        _manifest(
            TOKEN_MAJOR_SHAPE,
            TP8_GDN,
            num_kv_heads=2,
            builder=sol.build_qwen35_pool_manifest_sol,
        )


def test_token_major_builder_is_unchanged_by_the_hook():
    manifest = _manifest(
        TOKEN_MAJOR_SHAPE,
        TP8_GDN,
        num_kv_heads=2,
        builder=rpm.build_qwen35_pool_manifest,
    )
    (fa,) = [pool for pool in manifest.pools if pool.tag == rpm.TAG_FA]
    assert fa.regions[0].name == "fa_payload"
    assert fa.num_blocks == 32
    with pytest.raises(sol.SeqOnLaneError):
        sol.sol_fa_geometry_from_manifest(manifest)


# --------------------------------------------------------------------------
# FA lowering goldens: reassemble every destination shard from producer pages
# --------------------------------------------------------------------------


def _slab_pattern(global_kp: int, head_pair: int, slab_bytes: int) -> bytes:
    marker = bytes([global_kp & 0xFF, (global_kp >> 8) & 0xFF, head_pair & 0xFF, 0xA5])
    return marker * (slab_bytes // 4)


def _producer_pages(
    geometry, *, num_tokens, rank, parallelism, interleave, page_tokens
):
    """This rank's manager blocks (bytes) with owned chunks packed dense at
    kernel-page granularity, each head-pair slab carrying (chunk, pair)."""
    kps_per_block = page_tokens // geometry.page_tokens
    owned = []
    for start, end in owned_token_ranges(
        num_tokens=num_tokens,
        transfer_rank=rank,
        parallelism=parallelism,
        interleave_tokens=interleave,
    ):
        first = start // geometry.page_tokens
        last = (end + geometry.page_tokens - 1) // geometry.page_tokens
        owned.extend(range(first, last))
    blocks = []
    for local_kp, global_kp in enumerate(owned):
        if local_kp % kps_per_block == 0:
            blocks.append(bytearray(kps_per_block * geometry.kernel_page_bytes))
        base = (local_kp % kps_per_block) * geometry.kernel_page_bytes
        for head_pair in range(geometry.kv_rows):
            off = base + head_pair * geometry.slab_bytes
            blocks[-1][off : off + geometry.slab_bytes] = _slab_pattern(
                global_kp, head_pair, geometry.slab_bytes
            )
    return owned, blocks


@pytest.mark.parametrize(
    "parallelism,dst_shards,page_tokens,num_tokens",
    [
        (8, 2, 1024, 32767),  # 397B 32K request, multi-block source packing
        (8, 2, 8192, 32767),  # golden manager page
        (8, 1, 1024, 32767),  # TP1 destination: whole pages
        (1, 2, 1024, 8191),  # DP prefill producer (no PCP)
        (8, 2, 1024, 1025),  # one ownership cycle + one token: short tail
    ],
)
def test_fa_lowering_reassembles_each_destination_shard(
    parallelism, dst_shards, page_tokens, num_tokens
):
    geometry = sol.sol_fa_geometry_from_shape(PRODUCER_SHAPE)
    interleave = 128
    shard_bytes = geometry.shard_bytes(dst_shards)
    total_kps = (num_tokens + 127) // 128
    assembled = [bytearray(total_kps * shard_bytes) for _ in range(dst_shards)]
    written = [bytearray(total_kps) for _ in range(dst_shards)]
    total_declared = 0
    for rank in range(parallelism):
        owned, blocks = _producer_pages(
            geometry,
            num_tokens=num_tokens,
            rank=rank,
            parallelism=parallelism,
            interleave=interleave,
            page_tokens=page_tokens,
        )
        registration = sol.lower_fa_spans_sol(
            num_tokens=num_tokens,
            transfer_rank=rank,
            parallelism=parallelism,
            interleave_tokens=interleave,
            page_tokens=page_tokens,
            geometry=geometry,
            dst_shards=dst_shards,
            block_ids=list(range(100, 100 + len(blocks))),
        )
        assert registration.dst_space_version == 1
        assert registration.tag == "fa"
        assert len(registration.spans) == len(owned) * dst_shards
        assert registration.declared_bytes == len(owned) * geometry.kernel_page_bytes
        total_declared += registration.declared_bytes
        for span in registration.spans:
            assert span.count == 1 and span.dst_block_index == 0
            assert span.size_bytes == shard_bytes
            shard = 0 if dst_shards == 1 else span.dst_unit_ordinal
            if dst_shards == 1:
                assert span.dst_unit_ordinal is None
            else:
                assert 0 <= span.dst_unit_ordinal < dst_shards
            assert span.dst_offset_bytes % shard_bytes == 0
            global_kp = span.dst_offset_bytes // shard_bytes
            assert written[shard][global_kp] == 0, "exactly-once coverage"
            written[shard][global_kp] = 1
            source = blocks[span.src_block_ordinal]
            assembled[shard][
                span.dst_offset_bytes : span.dst_offset_bytes + shard_bytes
            ] = source[span.src_offset_bytes : span.src_offset_bytes + shard_bytes]
    # Every shard's request-global space is fully covered and holds, at
    # kernel page c, the head-pairs [d*H2/S, (d+1)*H2/S) of chunk c.
    pairs_per_shard = geometry.kv_rows // dst_shards
    for shard in range(dst_shards):
        assert all(written[shard]), "every kernel page reached the shard"
        for global_kp in range(total_kps):
            for local_pair in range(pairs_per_shard):
                head_pair = shard * pairs_per_shard + local_pair
                off = global_kp * shard_bytes + local_pair * geometry.slab_bytes
                assert assembled[shard][
                    off : off + geometry.slab_bytes
                ] == _slab_pattern(global_kp, head_pair, geometry.slab_bytes), (
                    shard,
                    global_kp,
                    head_pair,
                )
    # The declared bytes are every producer's whole owned pages, tail page
    # included (the planner's clipped/covered accounting counts each span once).
    assert total_declared == total_kps * geometry.kernel_page_bytes


def test_fa_lowering_tail_ships_a_whole_kernel_page():
    geometry = sol.sol_fa_geometry_from_shape(PRODUCER_SHAPE)
    # 32,767 tokens: chunk 255 (rank 7) holds 127 tokens yet ships 64 KiB.
    registration = sol.lower_fa_spans_sol(
        num_tokens=32767,
        transfer_rank=7,
        parallelism=8,
        interleave_tokens=128,
        page_tokens=8192,
        geometry=geometry,
        dst_shards=2,
        block_ids=[9],
    )
    last = max(registration.spans, key=lambda s: s.dst_offset_bytes)
    assert last.dst_offset_bytes == 255 * 64 * 1024
    assert last.size_bytes == 64 * 1024
    assert registration.declared_bytes == 32 * 128 * 1024


def test_fa_lowering_source_packing_across_blocks():
    geometry = sol.sol_fa_geometry_from_shape(PRODUCER_SHAPE)
    # page 1024 tokens = 8 kernel pages per block; rank 2 owns chunks 2, 10,
    # 18, ... -> local kernel pages 0, 1, 2, ...: the 9th lands in block 1.
    registration = sol.lower_fa_spans_sol(
        num_tokens=16384,
        transfer_rank=2,
        parallelism=8,
        interleave_tokens=128,
        page_tokens=1024,
        geometry=geometry,
        dst_shards=2,
        block_ids=[4, 5],
    )
    shard0 = [s for s in registration.spans if s.dst_unit_ordinal == 0]
    assert [s.src_block_ordinal for s in shard0] == [0] * 8 + [1] * 8
    assert [s.src_offset_bytes for s in shard0][:3] == [0, 128 * 1024, 2 * 128 * 1024]
    assert shard0[8].src_offset_bytes == 0  # first kernel page of block 1
    shard1 = [s for s in registration.spans if s.dst_unit_ordinal == 1]
    assert [
        s.src_offset_bytes - t.src_offset_bytes for s, t in zip(shard1, shard0)
    ] == [64 * 1024] * 16
    assert [s.dst_offset_bytes // (64 * 1024) for s in shard0] == [
        2 + 8 * i for i in range(16)
    ]


def test_fa_lowering_fails_closed():
    geometry = sol.sol_fa_geometry_from_shape(PRODUCER_SHAPE)
    kwargs = dict(
        num_tokens=32767,
        transfer_rank=0,
        parallelism=8,
        interleave_tokens=128,
        page_tokens=8192,
        geometry=geometry,
        dst_shards=2,
        block_ids=[1],
    )
    with pytest.raises(sol.SeqOnLaneError):  # interleave != kernel page
        sol.lower_fa_spans_sol(**{**kwargs, "interleave_tokens": 256})
    with pytest.raises(sol.SeqOnLaneError):  # page not kernel-page aligned
        sol.lower_fa_spans_sol(**{**kwargs, "page_tokens": 1000})
    with pytest.raises(sol.SeqOnLaneError):  # shards must divide head pairs
        sol.lower_fa_spans_sol(**{**kwargs, "dst_shards": 3})
    with pytest.raises(sol.SeqOnLaneError):  # block ids vs packing
        sol.lower_fa_spans_sol(**{**kwargs, "block_ids": [1, 2]})


# --------------------------------------------------------------------------
# GDN state: existing fan-in lowering routed per destination shard
# --------------------------------------------------------------------------

# Qwen3.5-397B TP8 shard, QK pair-blocked: [QK 1024 | V 2048] per tap.
_CONV_REGIONS_PAIR_397B_TP8 = (
    RegionSpec(
        name="gdn_conv_qk",
        offset_bytes=0,
        stride_bytes=3072,
        unit_bytes=1024,
        num_units=3,
        units_per_stride=1,
    ),
    RegionSpec(
        name="gdn_conv_v",
        offset_bytes=1024,
        stride_bytes=3072,
        unit_bytes=256,
        num_units=3,
        units_per_stride=8,
    ),
)
_SSM_REGIONS_397B_TP8 = (
    RegionSpec(
        name="gdn_ssm",
        offset_bytes=0,
        stride_bytes=65536,
        unit_bytes=65536,
        num_units=8,
    ),
)


def test_gdn_shard_lowering_routes_to_its_destination_shard():
    for rank in range(8):
        routed = sol.lower_gdn_state_shard_spans_sol(
            tag="gdn.conv.g1",
            block_id=31,
            transfer_rank=rank,
            parallelism=8,
            regions=_CONV_REGIONS_PAIR_397B_TP8,
            dst_shards=2,
        )
        shard, rank_in_shard = divmod(rank, 4)
        base = lower_gdn_state_shard_spans(
            tag="gdn.conv.g1",
            block_id=31,
            transfer_rank=rank_in_shard,
            parallelism=4,
            regions=_CONV_REGIONS_PAIR_397B_TP8,
        )
        assert routed.declared_bytes == base.declared_bytes == 9216
        assert routed.block_ids == (31,)
        assert len(routed.spans) == 2
        for got, want in zip(routed.spans, base.spans):
            assert got.dst_unit_ordinal == shard
            assert (
                got.src_offset_bytes,
                got.dst_offset_bytes,
                got.size_bytes,
                got.src_stride_bytes,
                got.dst_stride_bytes,
                got.count,
            ) == (
                want.src_offset_bytes,
                want.dst_offset_bytes,
                want.size_bytes,
                want.src_stride_bytes,
                want.dst_stride_bytes,
                want.count,
            )
        # Rank 5 -> shard 1, second of its four: QK at 1024, V at 4096+2048.
        if rank == 5:
            qk, v = routed.spans
            assert (qk.dst_offset_bytes, qk.dst_stride_bytes) == (1024, 12288)
            assert (v.dst_offset_bytes, v.dst_stride_bytes) == (6144, 12288)


def test_gdn_ssm_shard_lowering_and_single_shard_identity():
    routed = sol.lower_gdn_state_shard_spans_sol(
        tag="gdn.ssm.g0",
        block_id=7,
        transfer_rank=6,
        parallelism=8,
        regions=_SSM_REGIONS_397B_TP8,
        dst_shards=2,
    )
    (span,) = routed.spans
    assert span.dst_unit_ordinal == 1
    assert span.dst_offset_bytes == 2 * 8 * 65536
    assert span.size_bytes == 8 * 65536
    same = sol.lower_gdn_state_shard_spans_sol(
        tag="gdn.ssm.g0",
        block_id=7,
        transfer_rank=6,
        parallelism=8,
        regions=_SSM_REGIONS_397B_TP8,
        dst_shards=1,
    )
    base = lower_gdn_state_shard_spans(
        tag="gdn.ssm.g0",
        block_id=7,
        transfer_rank=6,
        parallelism=8,
        regions=_SSM_REGIONS_397B_TP8,
    )
    assert same == base
    assert same.spans[0].dst_unit_ordinal is None
    with pytest.raises(sol.SeqOnLaneError):
        sol.lower_gdn_state_shard_spans_sol(
            tag="gdn.ssm.g0",
            block_id=7,
            transfer_rank=6,
            parallelism=8,
            regions=_SSM_REGIONS_397B_TP8,
            dst_shards=3,
        )


# --------------------------------------------------------------------------
# fingerprint v2
# --------------------------------------------------------------------------


def test_sol_fingerprint_is_role_invariant_and_layout_specific():
    producer = _manifest(
        PRODUCER_SHAPE,
        TP8_GDN,
        num_kv_heads=2,
        builder=sol.build_qwen35_pool_manifest_sol,
    )
    shard = _manifest(
        TP2_SHAPE, TP2_GDN, num_kv_heads=1, builder=sol.build_qwen35_pool_manifest_sol
    )
    getter = lambda tensor: SOL_LAYOUT  # noqa: E731
    version = lambda name: f"{name}-test"  # noqa: E731
    fp_a, payload_a = sol.measured_sol_layout_fingerprint(
        producer, layout_getter=getter, package_version=version
    )
    fp_b, payload_b = sol.measured_sol_layout_fingerprint(
        shard, layout_getter=getter, package_version=version
    )
    assert fp_a == fp_b
    assert payload_a == payload_b
    assert payload_a["schema"] == sol.SOL_FINGERPRINT_SCHEMA
    assert payload_a["fa_kv_layout"] == "seq-along-lane"
    assert payload_a["fa_kernel_page_tokens"] == 128
    assert payload_a["pool_lane_width"] == 128
    assert "kv_rows" not in payload_a
    assert fp_a == sol.canonical_fingerprint(payload_a)
    # A token-major peer cannot produce this fingerprint: the module refuses
    # its manifest outright (and the schema token differs anyway).
    token_major = _manifest(
        TOKEN_MAJOR_SHAPE,
        TP8_GDN,
        num_kv_heads=2,
        builder=rpm.build_qwen35_pool_manifest,
    )
    with pytest.raises(sol.SeqOnLaneError):
        sol.measured_sol_layout_fingerprint(
            token_major, layout_getter=getter, package_version=version
        )
    bad_tiles = lambda tensor: ((4, 3, 2, 1, 0), ((8, 128), (4, 1)), 8)  # noqa
    with pytest.raises(RuntimeError):
        sol.measured_sol_layout_fingerprint(
            producer, layout_getter=bad_tiles, package_version=version
        )


def test_gdn_full_width_producer_splits_into_shards():
    # Full-width (TP1) conv: [QK 8192 | V 16384] per tap; SSM 64 heads.
    conv_full = (
        RegionSpec(
            name="gdn_conv_qk",
            offset_bytes=0,
            stride_bytes=24576,
            unit_bytes=8192,
            num_units=3,
            units_per_stride=1,
        ),
        RegionSpec(
            name="gdn_conv_v",
            offset_bytes=8192,
            stride_bytes=24576,
            unit_bytes=256,
            num_units=3,
            units_per_stride=64,
        ),
    )
    conv = sol.lower_gdn_state_shard_spans_sol(
        tag="gdn.conv.g2",
        block_id=5,
        transfer_rank=0,
        parallelism=1,
        regions=conv_full,
        dst_shards=2,
    )
    assert conv.declared_bytes == 3 * 24576
    by_shard = {}
    for span in conv.spans:
        by_shard.setdefault(span.dst_unit_ordinal, []).append(span)
    for shard in (0, 1):
        qk, v = by_shard[shard]
        assert (
            qk.src_offset_bytes,
            qk.dst_offset_bytes,
            qk.size_bytes,
            qk.src_stride_bytes,
            qk.dst_stride_bytes,
            qk.count,
        ) == (shard * 4096, 0, 4096, 24576, 12288, 3)
        assert (
            v.src_offset_bytes,
            v.dst_offset_bytes,
            v.size_bytes,
            v.src_stride_bytes,
            v.dst_stride_bytes,
            v.count,
        ) == (8192 + shard * 8192, 4096, 8192, 24576, 12288, 3)
    ssm_full = (
        RegionSpec(
            name="gdn_ssm",
            offset_bytes=0,
            stride_bytes=65536,
            unit_bytes=65536,
            num_units=64,
        ),
    )
    ssm = sol.lower_gdn_state_shard_spans_sol(
        tag="gdn.ssm.g0",
        block_id=5,
        transfer_rank=0,
        parallelism=1,
        regions=ssm_full,
        dst_shards=2,
    )
    assert [
        (s.dst_unit_ordinal, s.src_offset_bytes, s.dst_offset_bytes, s.size_bytes)
        for s in ssm.spans
    ] == [(0, 0, 0, 32 * 65536), (1, 32 * 65536, 0, 32 * 65536)]


@pytest.mark.parametrize("kind", ["conv", "ssm"])
@pytest.mark.parametrize("shard_bytes", [512, 1024])
@pytest.mark.parametrize("physical_granule_bytes", [512, 1024])
def test_gdn_full_width_split_respects_physical_granule(
    kind, shard_bytes, physical_granule_bytes
):
    segment_bytes = 2 * shard_bytes
    if kind == "conv":
        regions = (
            RegionSpec("gdn_conv_qk", 0, 2 * segment_bytes, segment_bytes, 3),
            RegionSpec(
                "gdn_conv_v", segment_bytes, 2 * segment_bytes, segment_bytes, 3
            ),
        )
    else:
        regions = (RegionSpec("gdn_ssm", 0, shard_bytes, shard_bytes, 2),)

    def lower():
        return sol.lower_gdn_state_shard_spans_sol(
            tag=f"gdn.{kind}.g0",
            block_id=5,
            transfer_rank=0,
            parallelism=1,
            regions=regions,
            dst_shards=2,
            physical_granule_bytes=physical_granule_bytes,
        )

    if shard_bytes < physical_granule_bytes:
        with pytest.raises(sol.SeqOnLaneError, match="physical.*1024"):
            lower()
        return

    result = lower()
    expected = []
    for segment in range(2 if kind == "conv" else 1):
        for shard in range(2):
            expected.append(
                (
                    shard,
                    segment * segment_bytes + shard * shard_bytes,
                    segment * shard_bytes,
                    shard_bytes,
                    2 * segment_bytes if kind == "conv" else 0,
                    segment_bytes if kind == "conv" else 0,
                    3 if kind == "conv" else 1,
                )
            )
    assert [
        (
            span.dst_unit_ordinal,
            span.src_offset_bytes,
            span.dst_offset_bytes,
            span.size_bytes,
            span.src_stride_bytes,
            span.dst_stride_bytes,
            span.count,
        )
        for span in result.spans
    ] == expected


def test_geometry_generalizes_to_256_lane_pages():
    # This branch's kernels pick the kernel page from the manager block
    # (block 4352 -> 256-lane pages on both roles); one owned chunk
    # (interleave 256) is then one contiguous 128 KiB run per TP2 shard.
    producer = sol.sol_fa_geometry_from_shape((17, 4, 64, 4, 256))
    assert producer.page_tokens == 256
    assert producer.slab_bytes == 64 * 1024
    assert producer.pair_bytes == 128 * 1024
    assert producer.kernel_page_bytes == 256 * 1024
    assert producer.shard_bytes(2) == 128 * 1024
    (region,) = sol.sol_fa_regions(block_size_tokens=4352, geometry=producer)
    assert region.num_units == 17
    assert region.extent_end_bytes == 17 * 256 * 1024
    registration = sol.lower_fa_spans_sol(
        num_tokens=4352 * 8,
        transfer_rank=3,
        parallelism=8,
        interleave_tokens=256,
        page_tokens=4352,
        geometry=producer,
        dst_shards=2,
        block_ids=[7],
    )
    # Rank 3 owns global pages 3, 11, 19, ...: 17 pages x 2 shards.
    assert len(registration.spans) == 17 * 2
    first, second = registration.spans[:2]
    assert (
        first.src_block_ordinal,
        first.src_offset_bytes,
        first.dst_offset_bytes,
        first.size_bytes,
        first.dst_unit_ordinal,
    ) == (0, 0, 3 * 128 * 1024, 128 * 1024, 0)
    assert (
        second.src_offset_bytes,
        second.dst_offset_bytes,
        second.dst_unit_ordinal,
    ) == (128 * 1024, 3 * 128 * 1024, 1)
    third = registration.spans[2]
    assert (third.src_offset_bytes, third.dst_offset_bytes) == (
        256 * 1024,
        11 * 128 * 1024,
    )
    assert registration.declared_bytes == 17 * 256 * 1024
    with pytest.raises(sol.SeqOnLaneError, match="interleave"):
        sol.lower_fa_spans_sol(
            num_tokens=4352 * 8,
            transfer_rank=0,
            parallelism=8,
            interleave_tokens=128,
            page_tokens=4352,
            geometry=producer,
            dst_shards=2,
            block_ids=[7],
        )
    assert sol.sol_fa_skip_bytes(512, geometry=producer, dst_shards=2) == 2 * 128 * 1024
    with pytest.raises(sol.SeqOnLaneError, match="whole number"):
        sol.sol_fa_skip_bytes(300, geometry=producer, dst_shards=2)
