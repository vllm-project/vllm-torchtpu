# SPDX-License-Identifier: Apache-2.0
import logging

from .tpu_connector_v2_test_utils import (head_mapping_rows, load_v2_module,
                                          qwen35_4tp_layer_layout)


def test_qwen35_35b_4tp_fa_head_replication_matches_vllm():
    mod = load_v2_module()

    for tp_rank in range(4):
        pull_meta = _build_qwen35_4tp_pull_meta(mod,
                                                destination_tp_size=4,
                                                destination_tp_rank=tp_rank)
        # vLLM's QKV loader uses tp_rank // num_kv_head_replicas for K/V
        # shards. For Qwen3.5 35B 2 KV heads on 4TP, that is 0,0,1,1.
        expected_global_head = tp_rank // 2

        assert pull_meta.p_ranks == (tp_rank, )
        assert pull_meta.fa_heads_by_rank == {
            tp_rank: (expected_global_head, ),
        }
        assert {
            rank: head_mapping_rows(mappings)
            for rank, mappings in pull_meta.fa_head_mappings_by_rank.items()
        } == {
            tp_rank: ((expected_global_head, 0, 0, None), ),
        }


def _build_qwen35_4tp_pull_meta(mod, destination_tp_size, destination_tp_rank):
    planner = mod.ContiguousHeadTPTransferPlanner()
    source_tp_size = 4
    source_layout = mod.KVParallelLayout(
        full_attn_pcp_size=1,
        full_attn_tp_size=source_tp_size,
        linear_attn_pcp_size=1,
        linear_attn_tp_size=source_tp_size,
    )
    destination_layout = mod.KVParallelLayout(
        full_attn_pcp_size=1,
        full_attn_tp_size=destination_tp_size,
        linear_attn_pcp_size=1,
        linear_attn_tp_size=destination_tp_size,
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=2000 + destination_tp_rank,
        block_size=1056,
        kv_source_layout=source_layout,
        kv_caches={},
        fa_block_ids=(2, 3),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=destination_layout,
        block_size=1056,
        tp_rank=destination_tp_rank,
        total_num_kv_heads=2,
        total_num_mamba_key_heads=0,
        total_num_mamba_heads=0,
    )
    return planner.build_pull_meta(metadata, topology)


def test_qwen35_35b_4tp_to_2tp_fa_pull_meta_uses_source_owners():
    mod = load_v2_module()

    rank0_pull_meta = _build_qwen35_4tp_pull_meta(mod,
                                                  destination_tp_size=2,
                                                  destination_tp_rank=0)
    assert rank0_pull_meta.p_ranks == (0, )
    assert rank0_pull_meta.fa_heads_by_rank == {0: (0, )}
    assert {
        rank: head_mapping_rows(mappings)
        for rank, mappings in rank0_pull_meta.fa_head_mappings_by_rank.items()
    } == {
        0: ((0, 0, 0, None), ),
    }

    rank1_pull_meta = _build_qwen35_4tp_pull_meta(mod,
                                                  destination_tp_size=2,
                                                  destination_tp_rank=1)
    assert rank1_pull_meta.p_ranks == (2, )
    assert rank1_pull_meta.fa_heads_by_rank == {2: (1, )}
    assert {
        rank: head_mapping_rows(mappings)
        for rank, mappings in rank1_pull_meta.fa_head_mappings_by_rank.items()
    } == {
        2: ((1, 0, 0, None), ),
    }


def test_qwen35_35b_4tp_to_1tp_fa_pull_meta_uses_source_owners():
    mod = load_v2_module()

    pull_meta = _build_qwen35_4tp_pull_meta(mod,
                                            destination_tp_size=1,
                                            destination_tp_rank=0)

    assert pull_meta.p_ranks == (0, 2)
    assert pull_meta.fa_heads_by_rank == {
        0: (0, ),
        2: (1, ),
    }
    assert {
        rank: head_mapping_rows(mappings)
        for rank, mappings in pull_meta.fa_head_mappings_by_rank.items()
    } == {
        0: ((0, 0, 0, None), ),
        2: ((1, 0, 1, None), ),
    }


def test_qwen35_35b_4tp_tpu_blockpool_e2e_smoke(caplog):
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    raw_layout = qwen35_4tp_layer_layout()
    full_layers = tuple(item["full_layer"] for item in raw_layout)
    linear_layers = tuple(layer for item in raw_layout
                          for layer in item["linear_layers"])

    assert len(full_layers) == 10
    assert len(linear_layers) == 30

    page_bytes = 1_081_344
    attention_page_tokens = 1056
    attention_token_bytes = 1 * 4 * 256
    conv_state_bytes = 12_288
    recurrent_state_bytes = 524_288
    tensor_parallel_size = 4
    official_num_key_value_heads = 2
    official_linear_num_key_heads = 16
    official_linear_num_value_heads = 32
    local_linear_num_value_heads = (official_linear_num_value_heads //
                                    tensor_parallel_size)
    local_linear_num_key_heads = official_linear_num_key_heads // tensor_parallel_size
    head_dim = 128
    bf16_bytes = 2
    conv_slot_stride_bytes = 2048 * bf16_bytes
    conv_q_offset_bytes = 0
    conv_k_offset_bytes = local_linear_num_key_heads * head_dim * bf16_bytes
    conv_v_offset_bytes = conv_k_offset_bytes * 2
    source_raw_base = 10_000_000
    dest_raw_base = 20_000_000

    source_regions = {}
    dest_regions = {}
    for item in raw_layout:
        raw_idx = item["raw_idx"]
        source_raw_addr = source_raw_base + raw_idx * 10 * page_bytes
        dest_raw_addr = dest_raw_base + raw_idx * 10 * page_bytes
        full_layer = item["full_layer"]

        # Qwen3.5 35B 4TP Pallas attention view:
        # shape=(16, 1056, 1, 4, 256), dtype=fp8, page=1,081,344 B.
        # It is not head-first; one local packed KV-head group occupies the
        # full page and is copied as token-first strided segments.
        source_regions[full_layer] = mod.KVCacheRegion(
            rank=1,
            layer_name=full_layer,
            layer_type=mod.LayerType.FULL_ATTN,
            base_addr=source_raw_addr,
            block_size=attention_page_tokens,
            block_bytes=page_bytes,
            layout=mod.TensorLayout.TOKEN_FIRST,
            num_heads=1,
        )
        dest_regions[full_layer] = mod.KVCacheRegion(
            rank=1,
            layer_name=full_layer,
            layer_type=mod.LayerType.FULL_ATTN,
            base_addr=dest_raw_addr,
            block_size=attention_page_tokens,
            block_bytes=page_bytes,
            layout=mod.TensorLayout.TOKEN_FIRST,
            num_heads=1,
        )

        for group_idx, layer_name in enumerate(item["linear_layers"]):
            conv_key = f"{layer_name}.state0"
            recurrent_key = f"{layer_name}.state1"
            # Mamba/GDN views share the same unified page as attention:
            # state0: [0, 12288), bf16 conv state
            # state1: [12288, 536576), fp32 recurrent state
            # padding: [536576, 1081344)
            # block_size=None means this is an opaque state payload, not a
            # token page. block_id_index selects which mamba group block id to
            # use; block_stride_bytes keeps the unified-page physical stride.
            source_regions[conv_key] = mod.KVCacheRegion(
                rank=1,
                layer_name=conv_key,
                layer_type=mod.LayerType.MAMBA_STATE,
                base_addr=source_raw_addr,
                block_size=None,
                block_bytes=conv_state_bytes,
                block_stride_bytes=page_bytes,
                layout=mod.TensorLayout.BLOCKS_FIRST,
                num_heads=local_linear_num_key_heads * 2 +
                local_linear_num_value_heads,
                head_segments=(
                    mod.HeadSegment(
                        name="q",
                        global_heads=tuple(range(4, 8)),
                        local_head_start=0,
                        local_head_count=local_linear_num_key_heads,
                        head_bytes=head_dim * bf16_bytes,
                        base_offset_bytes=conv_q_offset_bytes,
                        stride_bytes=conv_slot_stride_bytes,
                        num_segments=3,
                    ),
                    mod.HeadSegment(
                        name="k",
                        global_heads=tuple(range(4, 8)),
                        local_head_start=local_linear_num_key_heads,
                        local_head_count=local_linear_num_key_heads,
                        head_bytes=head_dim * bf16_bytes,
                        base_offset_bytes=conv_k_offset_bytes,
                        stride_bytes=conv_slot_stride_bytes,
                        num_segments=3,
                    ),
                    mod.HeadSegment(
                        name="v",
                        global_heads=tuple(range(8, 16)),
                        local_head_start=local_linear_num_key_heads * 2,
                        local_head_count=local_linear_num_value_heads,
                        head_bytes=head_dim * bf16_bytes,
                        base_offset_bytes=conv_v_offset_bytes,
                        stride_bytes=conv_slot_stride_bytes,
                        num_segments=3,
                    ),
                ),
                block_id_index=group_idx,
            )
            dest_regions[conv_key] = mod.KVCacheRegion(
                rank=1,
                layer_name=conv_key,
                layer_type=mod.LayerType.MAMBA_STATE,
                base_addr=dest_raw_addr,
                block_size=None,
                block_bytes=conv_state_bytes,
                block_stride_bytes=page_bytes,
                layout=mod.TensorLayout.BLOCKS_FIRST,
                num_heads=local_linear_num_key_heads * 2 +
                local_linear_num_value_heads,
                head_segments=source_regions[conv_key].head_segments,
                block_id_index=group_idx,
            )
            source_regions[recurrent_key] = mod.KVCacheRegion(
                rank=1,
                layer_name=recurrent_key,
                layer_type=mod.LayerType.MAMBA_STATE,
                base_addr=source_raw_addr + conv_state_bytes,
                block_size=None,
                block_bytes=recurrent_state_bytes,
                block_stride_bytes=page_bytes,
                layout=mod.TensorLayout.BLOCKS_FIRST,
                num_heads=local_linear_num_value_heads,
                block_id_index=group_idx,
            )
            dest_regions[recurrent_key] = mod.KVCacheRegion(
                rank=1,
                layer_name=recurrent_key,
                layer_type=mod.LayerType.MAMBA_STATE,
                base_addr=dest_raw_addr + conv_state_bytes,
                block_size=None,
                block_bytes=recurrent_state_bytes,
                block_stride_bytes=page_bytes,
                layout=mod.TensorLayout.BLOCKS_FIRST,
                num_heads=local_linear_num_value_heads,
                block_id_index=group_idx,
            )

    assert len(source_regions) == 10 + 30 * 2
    assert len(dest_regions) == 10 + 30 * 2

    metadata = mod.ConnectorMetadataV2(
        req_id=100,
        block_size=256,
        kv_source_layout=mod.KVParallelLayout(
            full_attn_pcp_size=1,
            full_attn_tp_size=tensor_parallel_size,
            linear_attn_pcp_size=1,
            linear_attn_tp_size=tensor_parallel_size,
        ),
        kv_caches={1: source_regions},
        fa_block_ids=(2, 3),
        mamba_block_ids=(5, 6, 7),
    )
    topology = mod.TpKVTopology(
        local_layout=mod.KVParallelLayout(
            full_attn_pcp_size=1,
            full_attn_tp_size=tensor_parallel_size,
            linear_attn_pcp_size=1,
            linear_attn_tp_size=tensor_parallel_size,
        ),
        block_size=256,
        tp_rank=1,
        total_num_kv_heads=official_num_key_value_heads,
        total_num_mamba_key_heads=official_linear_num_key_heads,
        total_num_mamba_heads=official_linear_num_value_heads,
    )
    destination = mod.LocalDecodeAllocation(
        rank=1,
        block_size=256,
        kv_caches=dest_regions,
        fa_block_ids=(12, 13),
        mamba_block_ids=(50, 51, 52),
    )

    caplog.set_level(logging.INFO,
                     logger="vllm_torchtpu.distributed.kv_transfer.v2.planner")
    pull_meta = planner.build_pull_meta(metadata, topology)
    assert pull_meta.p_ranks == (1, )
    assert pull_meta.fa_source_block_ids == (2, 3)
    assert pull_meta.mamba_source_block_ids == (5, 6, 7)
    assert pull_meta.fa_heads_by_rank == {1: (0, )}
    assert pull_meta.mamba_key_heads_by_rank == {
        1: tuple(range(4, 8)),
    }
    assert pull_meta.mamba_value_heads_by_rank == {
        1: tuple(range(8, 16)),
    }
    assert {
        rank: head_mapping_rows(mappings)
        for rank, mappings in pull_meta.fa_head_mappings_by_rank.items()
    } == {
        1: ((0, 0, 0, None), )
    }
    assert {
        rank: head_mapping_rows(mappings)
        for rank, mappings in
        pull_meta.mamba_value_head_mappings_by_rank.items()
    } == {
        1: tuple((8 + i, i, i, None) for i in range(8)),
    }
    plans = planner.lower(metadata, topology, destination, pull_meta)

    assert tuple(plans) == (1, )
    ops = plans[1].ops
    state0_ops_per_layer = local_linear_num_key_heads * 2 + local_linear_num_value_heads
    state1_ops_per_layer = local_linear_num_value_heads
    assert len(
        ops) == 10 * 2 + 30 * (state0_ops_per_layer + state1_ops_per_layer)
    assert {op.layer_name for op in ops} == set(source_regions)

    full_ops = [op for op in ops if op.layer_type == mod.LayerType.FULL_ATTN]
    mamba_ops = [
        op for op in ops if op.layer_type == mod.LayerType.MAMBA_STATE
    ]
    assert len(full_ops) == 20
    assert len(mamba_ops) == 30 * (state0_ops_per_layer + state1_ops_per_layer)
    assert {op.global_head for op in full_ops} == {0}
    assert {op.global_head for op in mamba_ops} == set(range(4, 16))

    first_full = [op for op in full_ops if op.layer_name == full_layers[0]]
    assert [(
        op.src_addr,
        op.dst_addr,
        op.segment_bytes,
        op.src_stride_bytes,
        op.dst_stride_bytes,
        op.num_segments,
        op.total_bytes,
    ) for op in first_full] == [
        (
            source_raw_base + 2 * page_bytes,
            dest_raw_base + 12 * page_bytes,
            attention_token_bytes,
            attention_token_bytes,
            attention_token_bytes,
            attention_page_tokens,
            page_bytes,
        ),
        (
            source_raw_base + 3 * page_bytes,
            dest_raw_base + 13 * page_bytes,
            attention_token_bytes,
            attention_token_bytes,
            attention_token_bytes,
            attention_page_tokens,
            page_bytes,
        ),
    ]

    first_linear = linear_layers[0]
    first_conv = [
        op for op in mamba_ops if op.layer_name == f"{first_linear}.state0"
    ]
    first_recurrent = [
        op for op in mamba_ops if op.layer_name == f"{first_linear}.state1"
    ]
    assert len(first_conv) == state0_ops_per_layer
    assert len(first_recurrent) == local_linear_num_value_heads
    recurrent_head_bytes = recurrent_state_bytes // local_linear_num_value_heads
    first_q = first_conv[0]
    first_k = first_conv[local_linear_num_key_heads]
    first_v = first_conv[local_linear_num_key_heads * 2]
    first_recurrent_head = first_recurrent[0]
    assert (
        first_q.src_addr,
        first_q.dst_addr,
        first_q.segment_bytes,
        first_q.src_stride_bytes,
        first_q.dst_stride_bytes,
        first_q.num_segments,
        first_q.global_head,
        first_q.segment_name,
    ) == (
        source_raw_base + 5 * page_bytes,
        dest_raw_base + 50 * page_bytes,
        head_dim * bf16_bytes,
        conv_slot_stride_bytes,
        conv_slot_stride_bytes,
        3,
        4,
        "q",
    )
    assert (
        first_k.src_addr,
        first_k.dst_addr,
        first_k.segment_bytes,
        first_k.global_head,
        first_k.segment_name,
    ) == (
        source_raw_base + 5 * page_bytes + conv_k_offset_bytes,
        dest_raw_base + 50 * page_bytes + conv_k_offset_bytes,
        head_dim * bf16_bytes,
        4,
        "k",
    )
    assert (
        first_v.src_addr,
        first_v.dst_addr,
        first_v.segment_bytes,
        first_v.global_head,
        first_v.segment_name,
    ) == (
        source_raw_base + 5 * page_bytes + conv_v_offset_bytes,
        dest_raw_base + 50 * page_bytes + conv_v_offset_bytes,
        head_dim * bf16_bytes,
        8,
        "v",
    )
    assert (
        first_recurrent_head.src_addr,
        first_recurrent_head.dst_addr,
        first_recurrent_head.segment_bytes,
        first_recurrent_head.global_head,
    ) == (
        source_raw_base + conv_state_bytes + 5 * page_bytes,
        dest_raw_base + conv_state_bytes + 50 * page_bytes,
        recurrent_head_bytes,
        8,
    )

    third_linear = linear_layers[2]
    third_conv = [
        op for op in mamba_ops if op.layer_name == f"{third_linear}.state0"
    ]
    assert len(third_conv) == state0_ops_per_layer
    assert third_conv[0].src_addr == source_raw_base + 7 * page_bytes
    assert third_conv[0].dst_addr == dest_raw_base + 52 * page_bytes

    pull_log = next(
        record.getMessage() for record in caplog.records
        if "TPUConnectorV2 logical pull meta built" in record.getMessage())
    assert "TPUConnectorV2 pull meta built" not in pull_log
    assert ("mamba_state0_q_key_heads_by_rank={1: (4, 5, 6, 7)}" in pull_log)
    assert ("mamba_state0_k_key_heads_by_rank={1: (4, 5, 6, 7)}" in pull_log)
    assert ("mamba_state0_v_value_heads_by_rank={1: (8, 9, 10, 11, 12, "
            "13, 14, 15)}" in pull_log)
    assert ("mamba_state1_value_heads_by_rank={1: (8, 9, 10, 11, 12, "
            "13, 14, 15)}" in pull_log)

    lowering_log = next(
        record.getMessage() for record in caplog.records
        if "TPUConnectorV2 physical lowering summary" in record.getMessage())
    assert "TPUConnectorV2 lowering summary" not in lowering_log
    assert "mamba_ops_by_head" not in lowering_log
    assert "mamba_state0_q_ops_by_key_head={4: 30, 5: 30, 6: 30, 7: 30}" in lowering_log
    assert "mamba_state0_k_ops_by_key_head={4: 30, 5: 30, 6: 30, 7: 30}" in lowering_log
    assert ("mamba_state0_v_ops_by_value_head={8: 30, 9: 30, 10: 30, "
            "11: 30, 12: 30, 13: 30, 14: 30, 15: 30}" in lowering_log)
    assert ("mamba_state1_ops_by_value_head={8: 30, 9: 30, 10: 30, "
            "11: 30, 12: 30, 13: 30, 14: 30, 15: 30}" in lowering_log)
