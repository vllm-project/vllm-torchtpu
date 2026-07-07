# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from . import tpu_connector_v2_test_utils as utils


def test_full_attention_4pcp_to_2tp_builds_pull_meta():
    mod = utils.load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, _ = utils.build_fa_pcp_case(mod)

    pull_meta = planner.build_pull_meta(metadata, topology)

    # PCP splits one logical token stream across all source ranks, so one
    # decode rank has to pull FA cache from every PCP source rank.
    assert pull_meta.p_ranks == (0, 1, 2, 3)
    # Every PCP source rank stores all FA KV heads. With 2 destination TP
    # ranks and 2 total KV heads, destination rank 1 owns only global head 1.
    assert pull_meta.fa_heads_by_rank == {
        0: (1, ),
        1: (1, ),
        2: (1, ),
        3: (1, ),
    }
    assert pull_meta.mamba_key_heads_by_rank == {}
    assert pull_meta.mamba_value_heads_by_rank == {}
    # Mapping tuple is:
    #   (global_head, source_head, destination_head, segment_name)
    # Global head 1 is source head 1 on each PCP rank, and becomes local head
    # 0 in destination TP rank 1.
    assert {
        rank: utils.head_mapping_rows(mappings)
        for rank, mappings in pull_meta.fa_head_mappings_by_rank.items()
    } == {
        0: ((1, 1, 0, None), ),
        1: ((1, 1, 0, None), ),
        2: ((1, 1, 0, None), ),
        3: ((1, 1, 0, None), ),
    }
    # fa_block_ids are per logical source block, not pre-flattened by rank.
    # The logical_block_index below is the PCP-expanded source chunk index,
    # not a destination block index. For 4 PCP ranks and fa_block_ids=(100,
    # 101), the logical source chunks are:
    #   0:r0/block100, 1:r1/block100, 2:r2/block100, 3:r3/block100,
    #   4:r0/block101, 5:r1/block101, 6:r2/block101, 7:r3/block101.
    # These 8 source chunks are lowered into 4 destination blocks because this
    # test uses source_block_size=2 and destination_block_size=4.
    assert {
        rank: utils.block_ref_rows(block_refs)
        for rank, block_refs in pull_meta.fa_block_refs_by_rank.items()
    } == {
        0: ((0, 100), (4, 101)),
        1: ((1, 100), (5, 101)),
        2: ((2, 100), (6, 101)),
        3: ((3, 100), (7, 101)),
    }


def test_full_attention_4pcp_to_2tp_e2e_smoke():
    mod = utils.load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, destination = utils.build_fa_pcp_case(mod)

    pull_meta = planner.build_pull_meta(metadata, topology)
    plans = planner.lower(metadata, topology, destination, pull_meta)

    # The pointer lowering preserves the PCP source chunk expansion above,
    # then maps chunks 0/1/2/3 to destination blocks 200/200/201/201 and
    # chunks 4/5/6/7 to destination blocks 202/202/203/203.
    assert tuple(plans) == (0, 1, 2, 3)
    assert {
        rank: [(
            op.src_addr,
            op.dst_addr,
            op.segment_bytes,
            op.src_stride_bytes,
            op.dst_stride_bytes,
            op.num_segments,
            op.global_head,
        ) for op in plan.ops]
        for rank, plan in plans.items()
    } == {
        0: [
            (10_000 + 100 * 40 + 10, 100_000 + 200 * 80, 10, 20, 20, 2, 1),
            (10_000 + 101 * 40 + 10, 100_000 + 202 * 80, 10, 20, 20, 2, 1),
        ],
        1: [
            (11_000 + 100 * 40 + 10, 100_000 + 200 * 80 + 40, 10, 20, 20, 2,
             1),
            (11_000 + 101 * 40 + 10, 100_000 + 202 * 80 + 40, 10, 20, 20, 2,
             1),
        ],
        2: [
            (12_000 + 100 * 40 + 10, 100_000 + 201 * 80, 10, 20, 20, 2, 1),
            (12_000 + 101 * 40 + 10, 100_000 + 203 * 80, 10, 20, 20, 2, 1),
        ],
        3: [
            (13_000 + 100 * 40 + 10, 100_000 + 201 * 80 + 40, 10, 20, 20, 2,
             1),
            (13_000 + 101 * 40 + 10, 100_000 + 203 * 80 + 40, 10, 20, 20, 2,
             1),
        ],
    }


def test_full_attention_4pcp_to_2tp_offsets_external_token_window():
    mod = utils.load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, destination = utils.build_fa_pcp_case(mod)
    metadata = replace(metadata, fa_num_tokens=6, fa_token_offset=6)
    destination = replace(destination, fa_num_tokens=6, fa_token_offset=6)

    pull_meta = planner.build_pull_meta(metadata, topology)
    plans = planner.lower(metadata, topology, destination, pull_meta)

    assert {
        rank: [(
            op.src_addr,
            op.dst_addr,
            op.segment_bytes,
            op.src_stride_bytes,
            op.dst_stride_bytes,
            op.num_segments,
            op.global_head,
        ) for op in plan.ops]
        for rank, plan in plans.items()
    } == {
        0: [(10_000 + 101 * 40 + 10, 100_000 + 202 * 80, 10, 20, 20, 2, 1)],
        1:
        [(11_000 + 101 * 40 + 10, 100_000 + 202 * 80 + 40, 10, 20, 20, 2, 1)],
        # rank2 owns [4, 6) and [12, 14), both outside [6, 12).
        2: [],
        3:
        [(13_000 + 100 * 40 + 10, 100_000 + 201 * 80 + 40, 10, 20, 20, 2, 1)],
    }


def test_full_attention_4pcp_to_1tp_builds_pull_meta():
    mod = utils.load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, _ = utils.build_fa_pcp_case(
        mod,
        destination_tp_size=1,
        destination_tp_rank=0,
        destination_block_size=8,
        destination_block_ids=(300, 301),
    )

    pull_meta = planner.build_pull_meta(metadata, topology)
    # Destination TP size 1 owns every global FA KV head, but still pulls the
    # token stripes from all PCP source ranks.
    assert pull_meta.p_ranks == (0, 1, 2, 3)
    assert pull_meta.fa_heads_by_rank == {
        0: (0, 1),
        1: (0, 1),
        2: (0, 1),
        3: (0, 1),
    }
    # With one destination TP rank, global heads keep the same local indexes.
    assert {
        rank: utils.head_mapping_rows(mappings)
        for rank, mappings in pull_meta.fa_head_mappings_by_rank.items()
    } == {
        rank: ((0, 0, 0, None), (1, 1, 1, None))
        for rank in range(4)
    }
    # Block ownership is unchanged by destination TP size; only the head owner
    # mapping changes between 4PCP->2TP and 4PCP->1TP.
    assert {
        rank: utils.block_ref_rows(block_refs)
        for rank, block_refs in pull_meta.fa_block_refs_by_rank.items()
    } == {
        0: ((0, 100), (4, 101)),
        1: ((1, 100), (5, 101)),
        2: ((2, 100), (6, 101)),
        3: ((3, 100), (7, 101)),
    }


@pytest.mark.parametrize(
    ("source_layout", "destination_layout", "message"),
    [
        (
            lambda mod: mod.KVParallelLayout(
                full_attn_pcp_size=2,
                full_attn_tp_size=4,
                linear_attn_pcp_size=1,
                linear_attn_tp_size=1,
            ),
            lambda mod: utils.layout(mod, tp_size=2),
            "source full attention PCP requires full_attn_pcp_size == full_attn_tp_size",
        ),
        (
            lambda mod: mod.KVParallelLayout(
                full_attn_pcp_size=4,
                full_attn_tp_size=4,
                linear_attn_pcp_size=2,
                linear_attn_tp_size=2,
            ),
            lambda mod: utils.layout(mod, tp_size=2),
            "linear attention PCP is not supported",
        ),
        (
            lambda mod: utils.fa_pcp_source_layout(mod, pcp_size=4),
            lambda mod: mod.KVParallelLayout(
                full_attn_pcp_size=2,
                full_attn_tp_size=2,
                linear_attn_pcp_size=1,
                linear_attn_tp_size=1,
            ),
            "decode-side PCP layout is not supported",
        ),
    ],
)
def test_rejects_invalid_fa_pcp_layout(
    source_layout,
    destination_layout,
    message,
):
    mod = utils.load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, _ = utils.build_block_size_case(
        mod,
        source_block_size=2,
        destination_block_size=4,
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=metadata.req_id,
        block_size=metadata.block_size,
        kv_source_layout=source_layout(mod),
        kv_caches=metadata.kv_caches,
        fa_block_ids=metadata.fa_block_ids,
        mamba_block_ids=metadata.mamba_block_ids,
    )
    topology = mod.TpKVTopology(
        local_layout=destination_layout(mod),
        block_size=topology.block_size,
        tp_rank=0,
        total_num_kv_heads=topology.total_num_kv_heads,
        total_num_mamba_key_heads=topology.total_num_mamba_key_heads,
        total_num_mamba_heads=topology.total_num_mamba_heads,
    )

    with pytest.raises(ValueError, match=message):
        planner.build_pull_meta(metadata, topology)
