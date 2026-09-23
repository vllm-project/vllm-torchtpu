# SPDX-License-Identifier: Apache-2.0
"""PD byte-copy goldens; no TPU runtime or native transport is mocked in.

The reference enumerates global pages and heads independently of the span
lowerer. Physical block tables are deliberately non-contiguous. These tests
validate the transfer declarations, not network delivery or completion.
"""

import math

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import RegionSpec
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry


def _heads(total, degree, rank):
    if total >= degree:
        return list(range(rank * (total // degree), (rank + 1) * (total // degree)))
    return [rank // (degree // total)]


def _page(heads, page, slab_bytes):
    # Different bytes for token/page, K/V, head and channel. Use the full
    # physical slab, including padding in the last page.
    channel = np.arange(slab_bytes, dtype=np.uint32)
    return b"".join(
        ((channel * 7 + page * 19 + head * 31 + kv * 101) % 251)
        .astype(np.uint8)
        .tobytes()
        for head in heads
        for kv in range(2)
    )


@pytest.mark.parametrize(
    "pcp,tp,dtp,dcp,heads",
    [
        (16, 2, 32, 8, 4),  # Full model.
        (4, 2, 8, 8, 1),  # Quarter model; producer KV is replicated.
        (4, 2, 8, 2, 4),  # Distinct producer heads, multiple DCP groups.
        (2, 2, 4, 4, 1),  # Two four-device engines on one host.
        (8, 1, 2, 1, 2),  # Existing PCP -> TP2.
        (1, 1, 1, 1, 4),
        (4, 4, 8, 2, 2),  # Replicas also exist across destination DCP groups.
        (4, 4, 2, 1, 8),  # Multiple producers contribute to a destination page.
    ],
)
@pytest.mark.parametrize("tokens", [1, 127, 128, 129, 1023, 4225])
@pytest.mark.parametrize("manager_pages", [1, 3])
def test_fa_pcp_tp_to_dcp_byte_copy(pcp, tp, dtp, dcp, heads, tokens, manager_pages):
    page_tokens = 128
    pages = math.ceil(tokens / page_tokens)
    local_heads = max(1, heads // tp)
    geometry = sol.SolFaGeometry(2 * local_heads, 64, page_tokens)
    dest_head_count = max(1, heads // dtp)
    dest_page_bytes = dest_head_count * geometry.pair_bytes
    # Destination manager blocks deliberately differ from the source.
    dst_pages_per_block = 2
    dst_block_bytes = dst_pages_per_block * dest_page_bytes
    tables, destinations, coverage, expected = [], [], [], []
    for rank in range(dtp):
        owned = list(range(rank % dcp, pages, dcp))
        data = b"".join(
            _page(_heads(heads, dtp, rank), page, geometry.slab_bytes) for page in owned
        )
        ids = [503 + i * 13 for i in range(math.ceil(len(owned) / 2))]
        tables.append(ids)
        destinations.append({i: bytearray([255]) * dst_block_bytes for i in ids})
        coverage.append(np.zeros(len(data), dtype=np.uint8))
        expected.append(data)

    for p in range(pcp):
        owned = list(range(p, pages, pcp))
        for t in range(tp):
            ids = [701 + i * 11 for i in range(math.ceil(len(owned) / manager_pages))]
            block_bytes = manager_pages * geometry.kernel_page_bytes
            source = {i: bytearray([254]) * block_bytes for i in ids}
            for local_page, global_page in enumerate(owned):
                block, slot = divmod(local_page, manager_pages)
                start = slot * geometry.kernel_page_bytes
                source[ids[block]][start : start + geometry.kernel_page_bytes] = _page(
                    _heads(heads, tp, t), global_page, geometry.slab_bytes
                )
            reg = sol.lower_fa_spans_sol(
                num_tokens=tokens,
                transfer_rank=p,
                parallelism=pcp,
                producer_tp_rank=t,
                producer_tp_size=tp,
                total_kv_heads=heads,
                dst_dcp_size=dcp,
                interleave_tokens=128,
                page_tokens=manager_pages * 128,
                geometry=geometry,
                dst_shards=dtp,
                block_ids=ids,
            )
            assert reg.block_ids == tuple(ids)
            assert reg.dst_space_version == 1
            # declared_bytes describes the local source, even for a KV
            # replica that contributes no FA spans (it still owns GDN).
            assert reg.declared_bytes == len(owned) * geometry.kernel_page_bytes
            for span in reg.spans:
                rank = span.dst_unit_ordinal or 0
                assert span.count == 1 and span.dst_block_index == 0
                start, end = (
                    span.dst_offset_bytes,
                    (span.dst_offset_bytes + span.size_bytes),
                )
                assert 0 <= start < end <= len(expected[rank])
                assert not coverage[rank][start:end].any(), "duplicate write"
                coverage[rank][start:end] += 1
                payload = source[reg.block_ids[span.src_block_ordinal]][
                    span.src_offset_bytes : span.src_offset_bytes + span.size_bytes
                ]
                assert len(payload) == span.size_bytes
                # Emulate the planner splitting compact destinations across
                # the destination's own manager block table.
                offset = 0
                while offset < len(payload):
                    block, in_block = divmod(start + offset, dst_block_bytes)
                    size = min(len(payload) - offset, dst_block_bytes - in_block)
                    destinations[rank][tables[rank][block]][
                        in_block : in_block + size
                    ] = payload[offset : offset + size]
                    offset += size

    for rank in range(dtp):
        actual = b"".join(destinations[rank][i] for i in tables[rank])
        assert actual[: len(expected[rank])] == expected[rank]
        assert (coverage[rank] == 1).all(), "missing data"
        assert all(x == 255 for x in actual[len(expected[rank]) :])


@pytest.mark.parametrize(
    "pcp,tp,k,v", [(16, 2, 16, 128), (4, 2, 4, 32), (2, 2, 2, 16), (4, 2, 8, 32)]
)
@pytest.mark.parametrize("tag", ["gdn.conv.l0", "gdn.ssm.l0"])
def test_gdn_state_routes_tp_major_heads_and_qk_replicas(pcp, tp, k, v, tag):
    degree = pcp * tp
    head_geometry = derive_gdn_head_geometry(k, v, degree)
    lk, lv = head_geometry.local_num_kq_heads, head_geometry.local_num_v_heads
    # Both conv and recurrent state use BF16 in the target configuration.
    qk_bytes, v_bytes = 2 * lk * 128 * 2, lv * 128 * 2
    row = qk_bytes + v_bytes
    regions = [
        RegionSpec("gdn_conv_qk", 0, row, qk_bytes, 3, 1),
        RegionSpec("gdn_conv_v", qk_bytes, row, 256, 3, lv),
    ]
    if tag.startswith("gdn.ssm"):
        regions = [RegionSpec("gdn_ssm", 0, 32768, 32768, lv, 1)]

    def state(rank):
        kh = _heads(k, degree, rank)
        vh = _heads(v, degree, rank)
        if tag.startswith("gdn.ssm"):
            return b"".join(bytes([h + 1]) * 32768 for h in vh)
        return b"".join(
            b"".join(bytes([h + 1 + tap * 40]) * 512 for h in kh)
            + b"".join(bytes([h + 1]) * 256 for h in vh)
            for tap in range(3)
        )

    received = [bytearray(len(state(r))) for r in range(degree)]
    coverage = [bytearray(len(x)) for x in received]
    for p in range(pcp):
        for t in range(tp):
            rank = t * pcp + p
            payload = state(rank)
            reg = sol.lower_gdn_state_shard_spans_sol(
                tag=tag,
                block_id=37,
                transfer_rank=p,
                parallelism=pcp,
                producer_tp_size=tp,
                producer_tp_rank=t,
                regions=regions,
                dst_shards=degree,
                physical_granule_bytes=sol.WORD_ROW_BYTES,
            )
            assert reg.declared_bytes == len(payload)
            for span in reg.spans:
                assert span.dst_unit_ordinal == rank
                for i in range(span.count):
                    src = span.src_offset_bytes + i * span.src_stride_bytes
                    dst = span.dst_offset_bytes + i * span.dst_stride_bytes
                    end = dst + span.size_bytes
                    assert not any(coverage[rank][dst:end])
                    received[rank][dst:end] = payload[src : src + span.size_bytes]
                    coverage[rank][dst:end] = bytes([1]) * span.size_bytes
    for rank in range(degree):
        assert received[rank] == state(rank)
        assert all(coverage[rank])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"producer_tp_rank": 2, "producer_tp_size": 2, "total_kv_heads": 8}, "rank"),
        ({"producer_tp_size": 2}, "total_kv_heads"),
        ({"dst_dcp_size": 3}, "DCP"),
        ({"dst_dcp_size": 4}, "head"),
        ({"total_kv_heads": 3}, "head"),
        ({"total_kv_heads": 4}, "geometry"),
    ],
)
def test_fa_rejects_inconsistent_parallel_geometry(kwargs, match):
    args = dict(
        num_tokens=128,
        transfer_rank=0,
        parallelism=4,
        interleave_tokens=128,
        page_tokens=128,
        geometry=sol.SolFaGeometry(4, 64, 128),
        dst_shards=4,
        block_ids=[7],
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sol.lower_fa_spans_sol(**args)


@pytest.mark.parametrize("src_heads,dst_heads", [(2, 1), (1, 1)])
def test_hnd_state_carrier_physical_offsets(src_heads, dst_heads):
    """Validate byte offsets against the XLA tiled-layout reference.

    Different FA head counts give different pool page sizes. The GDN SSM
    and conv still occupy the same byte ranges in those pools. This models
    Mosaic's BF16 ref bitcast (rescaling the second-minor dimension), not
    NumPy's ordinary innermost-dimension dtype view.
    """
    from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

    ssm_bytes = 4 * 128 * 128 * 2
    conv_bytes = 3 * (2 * 128 + 4 * 128) * 2
    live_bytes = ssm_bytes + conv_bytes
    state = np.arange(live_bytes // 2, dtype=np.uint16)

    def physical(heads):
        raw = np.zeros((3, 2 * heads, 64, 4, 128), dtype=np.uint8)
        # One word holds two BF16 values at the same lane in adjacent rows.
        rows = raw.reshape(-1, 4, 128)
        typed = state.reshape(-1, 2, 128)
        rows[: len(typed), 0::2] = (typed & 255).astype(np.uint8)
        rows[: len(typed), 1::2] = (typed >> 8).astype(np.uint8)
        return _apply_xla_tiled_layout(
            raw, sol.EXPECTED_SOL_MINOR_TO_MAJOR, sol.EXPECTED_SOL_TILES
        )

    source, destination = physical(src_heads), physical(dst_heads)
    np.testing.assert_array_equal(source[:live_bytes], destination[:live_bytes])
    assert np.count_nonzero(source[live_bytes:]) == 0
    assert np.count_nonzero(destination[live_bytes:]) == 0
    # Each 512-byte carrier word-row includes exactly two complete BF16
    # rows. Q/K are one such row-pair; each SSM head is a multiple of it.
    words = source[:live_bytes].reshape(-1, 128, 4).astype(np.uint16)
    unpacked = np.stack(
        (words[:, :, 0] | words[:, :, 1] << 8, words[:, :, 2] | words[:, :, 3] << 8),
        axis=1,
    )
    np.testing.assert_array_equal(unpacked.reshape(-1), state)


def test_bf16_conv_does_not_relax_legacy_carrier_alignment():
    regions = [
        RegionSpec("gdn_conv_qk", 0, 1536, 512, 3, 1),
        RegionSpec("gdn_conv_v", 512, 1536, 256, 3, 4),
    ]
    with pytest.raises(ValueError, match="qk=512"):
        sol.lower_gdn_state_shard_spans_sol(
            tag="gdn.conv",
            block_id=7,
            transfer_rank=0,
            parallelism=4,
            producer_tp_size=2,
            producer_tp_rank=0,
            dst_shards=8,
            regions=regions,
        )
