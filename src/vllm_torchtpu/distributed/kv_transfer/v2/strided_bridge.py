# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.distributed.kv_transfer.v2.strided_transfer import (
    DestinationPageWriteSession, StridedTransferOp)

if TYPE_CHECKING:
    from vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector import \
        StridedSegmentOp


class TPUConnectorV2StridedBridge:
    """Bridge lowered V2 op lists to a strided transfer engine."""

    def __init__(self, transfer_engine: Any) -> None:
        if transfer_engine is None:
            raise ValueError("transfer_engine must be non-None")
        self.transfer_engine = transfer_engine

    def pull_rank_ops_into_session(
        self,
        *,
        remote_tp_rank: int,
        ops: Sequence["StridedSegmentOp"],
        write_session: DestinationPageWriteSession,
        dp_rank: int,
    ) -> int:
        if not ops:
            return 0
        transfer_ops = tuple(self.to_strided_transfer_op(op) for op in ops)
        return self.transfer_engine.pull_from_registered_into_session(
            remote_tp_rank=remote_tp_rank,
            ops=transfer_ops,
            write_session=write_session,
            dp_rank=dp_rank,
        )

    def new_destination_write_session(self) -> DestinationPageWriteSession:
        return self.transfer_engine.new_destination_write_session()

    @staticmethod
    def to_strided_transfer_op(op: "StridedSegmentOp") -> StridedTransferOp:
        return StridedTransferOp(
            src_region_id=op.source_region_id,
            dst_region_id=op.destination_region_id,
            src_offset_bytes=op.src_offset_bytes,
            dst_offset_bytes=op.dst_offset_bytes,
            segment_bytes=op.segment_bytes,
            src_stride_bytes=op.src_stride_bytes,
            dst_stride_bytes=op.dst_stride_bytes,
            num_segments=op.num_segments,
        )


__all__ = ["TPUConnectorV2StridedBridge"]
