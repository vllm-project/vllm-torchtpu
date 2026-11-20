# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host gather channel of one Host-DRAM Engram table, over tpu_sync.

Each rank owns one head shard per Host Engram layer and one tpu_sync host
signal channel per shard; one server thread per rank serves the channels in
layer order (``tpu_sync.api.torch.host_signal``). The compiled forward posts
this owner's row ids before layer 0 (``start``) and waits for the packed rows
at the Engram layer (``finish``); the server gathers them with
``host_rows.gather_rows`` and uploads them meanwhile.
"""

import torch

from vllm_torchtpu.layers.adapter.custom_ops import engram_host_channel_op as ops

from . import host_rows


def _host_signal():
    try:
        from tpu_sync.api.torch import host_signal
    except ImportError as exc:
        raise RuntimeError(
            "[DSV41] Engram Host channels need a tpu-sync-torch build that "
            "provides tpu_sync.api.torch.host_signal"
        ) from exc
    return host_signal


class HostChannel:
    """One layer's channel, the shard's Host tables and its row interval."""

    def __init__(self, table, capacity: int, predecessor: "HostChannel | None"):
        """``table``: a loaded ``ParallelEngramEmbedding`` head shard;
        ``capacity``: rows per request; ``predecessor``: the earlier layer's
        channel, whose request the server answers before this one's."""
        if table.dim + table.dim // table.block_size != ops.ROW_WORDS * 4:
            raise NotImplementedError(
                f"[DSV41] Engram Host channels gather {ops.ROW_WORDS * 4}-byte "
                f"rows; got dim={table.dim}, block_size={table.block_size}"
            )
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if table.dp_size != world or table.head_start != rank * table.part_n_hash_cols:
            # The ICI exchange runs over every rank and returns heads in rank
            # order, so rank r must own the r-th head shard.
            raise NotImplementedError(
                "[DSV41] Engram Host channels need one Engram DP group spanning "
                f"all {world} ranks (single host, TP1); got Engram DP "
                f"{table.dp_size}, head start {table.head_start} on rank {rank}. "
                "Serve multi-host with TPU_DEEPSEEK_V41_ENGRAM_HBM=all."
            )
        host_signal = _host_signal()
        ops.initialize()
        self.owners = world
        self.predecessor = predecessor
        # One server per rank serves every layer's channel in order.
        self.server = (
            predecessor.server
            if predecessor is not None
            else host_signal.HostSignalServer()
        )
        weight = table.weight.data.view(torch.uint8)
        scales = table.weight_scale_inv.data.view(torch.uint8)
        # The callback reads the tables and their description by address.
        self._tables = (weight, scales)
        self._meta = host_rows.table_meta(weight, scales)
        self.channel = self.server.add_channel(
            request_words=capacity,
            result_words=capacity * ops.ROW_WORDS,
            handler=host_rows.gather_rows(),
            user=int(self._meta.ctypes.data),
            predecessor=predecessor.channel if predecessor is not None else None,
        )
        self.interval = torch.tensor(
            [[table.head_start, table.vocab_start_idx, table.part_num_embeddings]],
            dtype=torch.int32,
            device="tpu",
        )
        from torch_tpu._internal import sync

        sync.synchronize([self.interval], wait=True)

    def serve(self) -> None:
        """Start the rank's server once every layer added its channel."""
        if not self.server.running:
            self.server.start()

    def check(self) -> None:
        """Raise the server's failure; a failed request reads as zero rows."""
        self.server.check()

    def start(
        self, hashes: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post this owner's rows of the DP group's ``hashes``.

        Start the predecessor's request first in the same graph: the server
        answers this one only after the predecessor's.
        """
        rows = ops.local_rows(hashes, self.interval)
        return self.channel.post(rows, barrier=hidden)

    def finish(
        self, hashes: torch.Tensor, receipt: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = hashes.shape[0] * ops.local_heads(hashes.shape[1], self.owners)
        packed, hidden = self.channel.wait(
            receipt, rows * ops.ROW_WORDS, barrier=hidden
        )
        return ops.exchange(hashes, self.interval, packed, hidden)
