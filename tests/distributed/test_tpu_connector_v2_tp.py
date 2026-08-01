# SPDX-License-Identifier: Apache-2.0

import pytest

from .tpu_connector_v2_test_utils import (build_block_size_case, layout,
                                          load_v2_module, manual_pull_meta)


def test_full_attention_block_size_heterogeneous_lowering_with_manual_pull_meta(
):
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, destination = build_block_size_case(
        mod,
        source_block_size=2,
        destination_block_size=4,
    )
    pull_meta = manual_pull_meta(
        mod,
        p_ranks=(0, ),
        fa_mapping_rows_by_rank={0: ((0, 0, 0, None), )},
        fa_source_block_ids=(10, 11, 12),
    )

    plans = planner.lower(metadata, topology, destination, pull_meta)
    ops = plans[0].ops

    assert [(
        op.src_addr,
        op.dst_addr,
        op.segment_bytes,
        op.src_stride_bytes,
        op.dst_stride_bytes,
        op.num_segments,
        op.total_bytes,
    ) for op in ops] == [
        (1_000 + 10 * 20, 2_000 + 20 * 40, 10, 10, 10, 2, 20),
        (1_000 + 11 * 20, 2_000 + 20 * 40 + 2 * 10, 10, 10, 10, 2, 20),
        (1_000 + 12 * 20, 2_000 + 21 * 40, 10, 10, 10, 2, 20),
    ]


@pytest.mark.parametrize(
    ("source_block_size", "destination_block_size", "message"),
    [
        (4, 2, "destination block_size must be >= source block_size"),
        (3, 5,
         "destination block_size must be divisible by source block_size"),
    ],
)
def test_rejects_unsupported_block_size_relation(
    source_block_size,
    destination_block_size,
    message,
):
    mod = load_v2_module()
    planner = mod.ContiguousHeadTPTransferPlanner()
    metadata, topology, destination = build_block_size_case(
        mod,
        source_block_size=source_block_size,
        destination_block_size=destination_block_size,
    )
    pull_meta = manual_pull_meta(
        mod,
        p_ranks=(0, ),
        fa_mapping_rows_by_rank={0: ((0, 0, 0, None), )},
        fa_source_block_ids=metadata.fa_block_ids,
    )

    with pytest.raises(ValueError, match=message):
        planner.lower(metadata, topology, destination, pull_meta)


@pytest.mark.parametrize("tp_size", [3, 6])
def test_rejects_non_power_of_two_tp_size(tp_size):
    mod = load_v2_module()

    with pytest.raises(ValueError, match="tp_size must be a power of two"):
        layout(mod, tp_size=tp_size)
