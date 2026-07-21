# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from .tpu_connector_v2_test_utils import (head_mapping_rows, layout,
                                          load_v2_module, mamba_state_regions,
                                          manual_pull_meta)


def test_mamba_state_8tp_to_2tp_builds_pull_meta():
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    source_tp_size = 8
    destination_tp_size = 2
    metadata = mod.ConnectorMetadataV2(
        req_id=2,
        block_size=256,
        kv_source_layout=layout(mod, tp_size=source_tp_size),
        kv_caches={},
        fa_block_ids=(),
        mamba_block_ids=(5, ),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=destination_tp_size),
        block_size=256,
        tp_rank=0,
        total_num_kv_heads=0,
        total_num_mamba_key_heads=32,
        total_num_mamba_heads=32,
    )

    pull_meta = planner.build_pull_meta(metadata, topology)
    assert pull_meta.p_ranks == (0, 1, 2, 3)
    assert pull_meta.fa_heads_by_rank == {}
    assert pull_meta.mamba_source_block_ids == (5, )
    assert pull_meta.mamba_key_heads_by_rank == {
        0: (0, 1, 2, 3),
        1: (4, 5, 6, 7),
        2: (8, 9, 10, 11),
        3: (12, 13, 14, 15),
    }
    assert pull_meta.mamba_value_heads_by_rank == {
        0: (0, 1, 2, 3),
        1: (4, 5, 6, 7),
        2: (8, 9, 10, 11),
        3: (12, 13, 14, 15),
    }
    assert pull_meta.fa_head_mappings_by_rank == {}
    assert {
        rank: head_mapping_rows(mappings)
        for rank, mappings in
        pull_meta.mamba_value_head_mappings_by_rank.items()
    } == {
        rank: tuple((rank * 4 + i, i, rank * 4 + i, None) for i in range(4))
        for rank in range(4)
    }


def test_mamba_state_lowering_with_manual_pull_meta():
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    source_block_id = 5
    destination_block_id = 50
    source_stride = 2_000_000
    destination_stride = 4_000_000
    source_regions = mamba_state_regions(
        mod,
        rank=0,
        tp_rank=0,
        tp_size=8,
        base_addr=10_000_000,
        block_stride_bytes=source_stride,
    )
    destination_regions = mamba_state_regions(
        mod,
        rank=0,
        tp_rank=0,
        tp_size=2,
        base_addr=20_000_000,
        block_stride_bytes=destination_stride,
    )
    source_regions = {
        name:
        replace(
            region,
            region_base_offset_bytes=111 if name.endswith(".state0") else 222)
        for name, region in source_regions.items()
    }
    destination_regions = {
        name:
        replace(
            region,
            region_base_offset_bytes=333 if name.endswith(".state0") else 444)
        for name, region in destination_regions.items()
    }
    metadata = mod.ConnectorMetadataV2(
        req_id=2,
        block_size=256,
        kv_source_layout=layout(mod, tp_size=8),
        kv_caches={0: source_regions},
        fa_block_ids=(),
        mamba_block_ids=(source_block_id, ),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=2),
        block_size=256,
        tp_rank=0,
        total_num_kv_heads=0,
        total_num_mamba_key_heads=32,
        total_num_mamba_heads=32,
    )
    destination = mod.LocalDecodeAllocation(
        rank=0,
        block_size=256,
        kv_caches=destination_regions,
        fa_block_ids=(),
        mamba_block_ids=(destination_block_id, ),
    )
    pull_meta = manual_pull_meta(
        mod,
        req_id=2,
        p_ranks=(0, ),
        mamba_mapping_rows_by_rank={
            0: tuple((i, i, i, None) for i in range(4)),
        },
        mamba_source_block_ids=(source_block_id, ),
    )

    plans = planner.lower(metadata, topology, destination, pull_meta)
    assert tuple(plans) == (0, )
    assert {op.layer_type
            for plan in plans.values()
            for op in plan.ops} == {
                mod.LayerType.MAMBA_STATE,
            }

    first_rank_ops = plans[0].ops
    state0_ops = [op for op in first_rank_ops if op.segment_name is not None]
    state1_ops = [op for op in first_rank_ops if op.segment_name is None]
    assert len(state0_ops) == 8
    assert len(state1_ops) == 4

    assert [(
        op.src_addr,
        op.dst_addr,
        op.segment_bytes,
        op.src_stride_bytes,
        op.dst_stride_bytes,
        op.num_segments,
        op.global_head,
        op.segment_name,
    ) for op in state0_ops] == [(
        10_000_000 + source_block_id * source_stride + 111 + i * 256,
        20_000_000 + destination_block_id * destination_stride + 333 + i * 256,
        256,
        2_048,
        8_192,
        3,
        i,
        "q",
    ) for i in range(2)] + [(
        10_000_000 + source_block_id * source_stride + 111 + 512 + i * 256,
        20_000_000 + destination_block_id * destination_stride + 333 + 2_048 +
        i * 256,
        256,
        2_048,
        8_192,
        3,
        i,
        "k",
    ) for i in range(2)] + [(
        10_000_000 + source_block_id * source_stride + 111 + 1_024 + i * 256,
        20_000_000 + destination_block_id * destination_stride + 333 + 4_096 +
        i * 256,
        256,
        2_048,
        8_192,
        3,
        i,
        "v",
    ) for i in range(4)]

    assert [(op.src_addr, op.dst_addr, op.segment_bytes, op.global_head)
            for op in state1_ops] == [(
                10_000_000 + 6_144 + source_block_id * source_stride + 222 +
                i * 65_536,
                20_000_000 + 24_576 +
                destination_block_id * destination_stride + 444 + i * 65_536,
                65_536,
                i,
            ) for i in range(4)]
