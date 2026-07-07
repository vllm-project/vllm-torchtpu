# SPDX-License-Identifier: Apache-2.0

import pytest

from .tpu_connector_v2_test_utils import (full_attention_region, layout,
                                          load_v2_module, manual_pull_meta)


def test_token_first_layout_keeps_padded_stride_and_live_head_separate():
    mod = load_v2_module()

    region = full_attention_region(
        mod,
        rank=0,
        layer_name="model.layers.0.self_attn",
        base_addr=1_000,
        block_size=4,
        token_stride_bytes=20,
        head_stride_bytes=10,
        live_head_bytes=10,
        num_heads=1,
    )

    assert region.block_bytes == 80
    assert region.head_bytes == 40
    assert region.token_bytes == 20
    assert region.token_head_stride_bytes == 10
    assert region.token_head_bytes == 10


def test_token_first_layout_rejects_unknown_layout_id():
    mod = load_v2_module()

    with pytest.raises(ValueError, match="unsupported TOKEN_FIRST layout_id"):
        full_attention_region(
            mod,
            rank=0,
            layer_name="model.layers.0.self_attn",
            base_addr=1_000,
            block_size=4,
            token_stride_bytes=20,
            head_stride_bytes=10,
            live_head_bytes=10,
            num_heads=1,
            layout_id="batched_rpa_token_first_v2",
        )


def test_token_first_layout_rejects_inconsistent_spec():
    mod = load_v2_module()

    with pytest.raises(
            ValueError,
            match=
            "KVCacheRegion TOKEN_FIRST fields must match token_first_layout",
    ):
        mod.KVCacheRegion(
            rank=0,
            layer_name="model.layers.0.self_attn",
            layer_type=mod.LayerType.FULL_ATTN,
            base_addr=1_000,
            block_size=4,
            block_bytes=80,
            layout=mod.TensorLayout.TOKEN_FIRST,
            num_heads=1,
            token_first_layout=mod.TokenFirstLayoutSpec(
                block_size=4,
                block_bytes=64,
                block_stride_bytes=64,
                token_stride_bytes=16,
                head_stride_bytes=10,
                live_head_bytes=10,
                num_heads=1,
                layout_id="pallas_batched_rpa_token_first_v1",
            ),
        )


def test_token_first_layout_rejects_uncovered_head_extent():
    mod = load_v2_module()

    with pytest.raises(
            ValueError,
            match="token_stride_bytes cannot cover",
    ):
        mod.TokenFirstLayoutSpec(
            block_size=4,
            block_bytes=64,
            block_stride_bytes=64,
            token_stride_bytes=16,
            head_stride_bytes=10,
            live_head_bytes=10,
            num_heads=2,
            layout_id="pallas_batched_rpa_token_first_v1",
        )


def test_full_attention_head_first_lowering():
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    layer_name = "model.layers.0.self_attn"
    source_region = mod.KVCacheRegion(
        rank=0,
        layer_name=layer_name,
        layer_type=mod.LayerType.FULL_ATTN,
        base_addr=1_000,
        block_size=4,
        block_bytes=80,
        layout=mod.TensorLayout.HEAD_FIRST,
        num_heads=2,
    )
    destination_region = mod.KVCacheRegion(
        rank=0,
        layer_name=layer_name,
        layer_type=mod.LayerType.FULL_ATTN,
        base_addr=2_000,
        block_size=4,
        block_bytes=80,
        layout=mod.TensorLayout.HEAD_FIRST,
        num_heads=2,
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=10,
        block_size=4,
        kv_source_layout=layout(mod, tp_size=1),
        kv_caches={0: {
            layer_name: source_region
        }},
        fa_block_ids=(10, ),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=1),
        block_size=4,
        tp_rank=0,
        total_num_kv_heads=2,
        total_num_mamba_key_heads=0,
        total_num_mamba_heads=0,
    )
    destination = mod.LocalDecodeAllocation(
        rank=0,
        block_size=4,
        kv_caches={layer_name: destination_region},
        fa_block_ids=(20, ),
        mamba_block_ids=(),
    )

    pull_meta = manual_pull_meta(
        mod,
        p_ranks=(0, ),
        fa_mapping_rows_by_rank={
            0: ((0, 0, 0, None), (1, 1, 1, None)),
        },
        fa_source_block_ids=(10, ),
    )
    plans = planner.lower(metadata, topology, destination, pull_meta)

    assert [(
        op.src_addr,
        op.dst_addr,
        op.segment_bytes,
        op.src_stride_bytes,
        op.dst_stride_bytes,
        op.num_segments,
        op.global_head,
    ) for op in plans[0].ops] == [
        (1_000 + 10 * 80, 2_000 + 20 * 80, 40, 40, 40, 1, 0),
        (1_000 + 10 * 80 + 40, 2_000 + 20 * 80 + 40, 40, 40, 40, 1, 1),
    ]


def test_full_attention_rejects_block_first_multi_head():
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    layer_name = "model.layers.0.self_attn"
    source_region = mod.KVCacheRegion(
        rank=0,
        layer_name=layer_name,
        layer_type=mod.LayerType.FULL_ATTN,
        base_addr=1_000,
        block_size=4,
        block_bytes=80,
        layout=mod.TensorLayout.BLOCKS_FIRST,
        num_heads=2,
    )
    destination_region = mod.KVCacheRegion(
        rank=0,
        layer_name=layer_name,
        layer_type=mod.LayerType.FULL_ATTN,
        base_addr=2_000,
        block_size=4,
        block_bytes=80,
        layout=mod.TensorLayout.BLOCKS_FIRST,
        num_heads=2,
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=11,
        block_size=4,
        kv_source_layout=layout(mod, tp_size=1),
        kv_caches={0: {
            layer_name: source_region
        }},
        fa_block_ids=(10, ),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=1),
        block_size=4,
        tp_rank=0,
        total_num_kv_heads=2,
        total_num_mamba_key_heads=0,
        total_num_mamba_heads=0,
    )
    destination = mod.LocalDecodeAllocation(
        rank=0,
        block_size=4,
        kv_caches={layer_name: destination_region},
        fa_block_ids=(20, ),
        mamba_block_ids=(),
    )

    pull_meta = manual_pull_meta(
        mod,
        p_ranks=(0, ),
        fa_mapping_rows_by_rank={
            0: ((0, 0, 0, None), (1, 1, 1, None)),
        },
        fa_source_block_ids=(10, ),
    )
    with pytest.raises(
            ValueError,
            match="BLOCKS_FIRST full attention head slicing is not supported",
    ):
        planner.lower(metadata, topology, destination, pull_meta)
