# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent native Host gather channel of one Host-DRAM Engram table.

Each rank owns one head shard per Host Engram layer and one channel per
shard (``csrc/engram_host_channel.cc``). The compiled forward requests the
rows before layer 0 (``start``) and waits for them at the Engram layer
(``finish``); the worker gathers and uploads them meanwhile. Every buffer is
allocated once and keeps its address until the worker stops.
"""

import hashlib
import json
import weakref
from functools import cache
from pathlib import Path

import torch
from torch_tpu._internal import sync

from vllm_torchtpu.kernels.host_channel import PAGE_WORDS
from vllm_torchtpu.layers.adapter.custom_ops import engram_host_channel_op as ops

SOURCE = Path(__file__).parents[4] / "csrc" / "engram_host_channel.cc"
LIBRARY = SOURCE.with_name("_engram_host_channel.so")


def build_metadata() -> dict:
    """The environment a library must be built for: TorchTPU's private
    tensor-buffer header is only valid against its exact XLA commit."""
    import torch_tpu

    include = Path(torch_tpu.__file__).parent / "include"
    return {
        "torch": torch.__version__,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "xla_commit": (include / "XLA_COMMIT").read_text().strip(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }


@cache
def load_library() -> None:
    metadata = LIBRARY.with_suffix(".json")
    if not (
        LIBRARY.is_file()
        and metadata.is_file()
        and json.loads(metadata.read_text()) == build_metadata()
    ):
        raise RuntimeError(
            f"[DSV41] {LIBRARY} is missing or was built for another TorchTPU, "
            "Torch or source revision. Rebuild it in this environment: "
            "python tools/build_engram_host_channel.py --xla-source <xla-COMMIT>"
        )
    torch.ops.load_library(str(LIBRARY))


def _pinned_host(words: int) -> torch.Tensor:
    # Registers torch.ops.ao.offload, TorchTPU's pinned Host allocation.
    from torch._functorch._activation_offloading import offload_ops  # noqa: F401

    pages = torch.ops.ao.offload(torch.zeros(words, dtype=torch.int32).to("tpu"))
    sync.synchronize([pages], wait=True)
    return pages


class HostChannel:
    """Device request/response pages, gather output and HBM receive buffer."""

    def __init__(self, table, capacity: int, predecessor: "HostChannel | None"):
        """``table``: a loaded ``ParallelEngramEmbedding`` head shard;
        ``capacity``: rows per request; ``predecessor``: the channel whose
        upload this one waits for before gathering."""
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
        load_library()
        ops.initialize()
        ids = _pinned_host(ops.round_pages(capacity + 2))
        request = _pinned_host(PAGE_WORDS)
        self.response = _pinned_host(PAGE_WORDS)
        output = _pinned_host(ops.round_pages(capacity * ops.ROW_WORDS))
        self.retained = torch.zeros(output.numel(), dtype=torch.int32, device="tpu")
        self.step = torch.zeros(1, dtype=torch.int32, device="tpu")
        self.no_dependency = torch.zeros(1, dtype=torch.int32, device="tpu")
        self.interval = torch.tensor(
            [[table.head_start, table.vocab_start_idx, table.part_num_embeddings]],
            dtype=torch.int32,
            device="tpu",
        )
        sync.synchronize(
            [self.retained, self.step, self.no_dependency, self.interval], wait=True
        )
        self._native = torch.classes.engram_host_channel.HostChannel(
            table.weight.data.view(torch.uint8),
            table.weight_scale_inv.data,
            ids,
            request,
            self.response,
            output,
            self.retained,
            predecessor._native if predecessor is not None else None,
        )
        # Graphs donate these views; the native lease keeps the allocations.
        self.ids, self.request = self._native.ids(), self._native.request()
        self._owners = (ids, request, output)
        self.predecessor = predecessor
        weakref.finalize(self, self._native.stop)

    def check(self) -> None:
        """Raise the worker's failure; a failed request published zero rows."""
        if error := self._native.error():
            raise RuntimeError(f"[DSV41] Engram Host channel failed: {error}")

    def start(
        self, hashes: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Request the rows of the DP group's ``hashes``.

        Start the predecessor's request first in the same graph: its updated
        ``step`` is the generation this worker waits for.
        """
        dependency = (
            self.predecessor.step
            if self.predecessor is not None
            else self.no_dependency
        )
        receipt, ids, request, hidden = torch.ops.engram_host.start(
            hashes, self.interval, self.step, dependency, self.ids, self.request, hidden
        )
        # Write the donated pages back so the graph aliases them in place.
        self.ids.copy_(torch.ops.ao.offload(ids))
        self.request.copy_(torch.ops.ao.offload(request))
        self.step.copy_(receipt[:1])
        return receipt, hidden

    def finish(
        self, hashes: torch.Tensor, receipt: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.engram_host.finish(
            hashes, self.interval, receipt, self.response, self.retained, hidden
        )
