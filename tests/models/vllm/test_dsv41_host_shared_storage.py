# SPDX-License-Identifier: Apache-2.0
"""Real Gloo processes load disjoint rows and query one shared CPU table."""

import errno
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram, host_storage
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    gather_host_engram_hashes,
)


def _broadcast(value):
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _worker(rank: int, rendezvous: str, failure: bool) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    group = SimpleNamespace(
        rank_in_group=rank,
        world_size=2,
        cpu_group=dist.group.WORLD,
        broadcast_object=_broadcast,
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engram.upstream, "get_engram_dp_group", lambda: group)
            patch.setattr(engram.upstream, "get_engram_dp_size", lambda: 2)
            patch.setattr(torch.accelerator, "current_device_index", lambda: 0)
            if failure:
                if rank == 1:
                    patch.setattr(
                        host_storage.os,
                        "posix_fallocate",
                        Mock(side_effect=OSError(errno.ENOSPC, "injected full shm")),
                    )
                with pytest.raises(RuntimeError, match="EDP rank 1.*full shm"):
                    host_storage.HostSharedEngramStorage(40, 32, 16, group)
                return

            # Unequal head sizes and TP shards exercise nonzero checkpoint offsets
            # and row slices that do not coincide with head boundaries.
            head_sizes = (5, 7, 9, 11, 13)
            rows = sum(head_sizes)
            weight = (
                (torch.arange(rows * 32).reshape(rows, 32) % 15 + 1)
                .float()
                .to(torch.float8_e4m3fn)
            )
            scales = (torch.arange(rows * 2).reshape(rows, 2) % 3 + 126).byte()
            for tp_size, tp_rank in ((1, 0), (2, 0), (2, 1)):
                patch.setattr(
                    engram.upstream,
                    "get_tensor_model_parallel_world_size",
                    lambda size=tp_size: size,
                )
                patch.setattr(
                    engram.upstream,
                    "get_tensor_model_parallel_rank",
                    lambda rank=tp_rank: rank,
                )
                embedding = engram.ParallelEngramEmbedding(
                    rows, 32, head_sizes, block_size=16, cpu_offload=True
                )
                assert embedding.dp_size == 1
                assert isinstance(
                    embedding._shared_memory, host_storage.HostSharedEngramStorage
                )
                start, count = embedding.load_row_slice
                parts = [None, None]
                dist.all_gather_object(parts, (start, count))
                assert parts[0][0] == embedding.vocab_start_idx
                assert parts[0][0] + parts[0][1] == parts[1][0]
                assert parts[1][0] + parts[1][1] == embedding.vocab_end_idx

                # The two loaders may run in different orders across processes.
                # Test both Run:AI sliced tensors and complete-checkpoint tensors.
                parameters = [
                    (embedding.weight, weight),
                    (embedding.weight_scale_inv, scales),
                ]
                for param, source in parameters[:: 1 if rank == 0 else -1]:
                    loaded = source[start : start + count] if tp_size == 1 else source
                    param.weight_loader(param, loaded)
                local = slice(embedding.vocab_start_idx, embedding.vocab_end_idx)
                assert torch.equal(
                    embedding.weight.view(torch.uint8), weight[local].view(torch.uint8)
                )
                assert torch.equal(embedding.weight_scale_inv, scales[local])

                module = object.__new__(engram.Engram)
                torch.nn.Module.__init__(module)
                module.embed_tokens = embedding
                module._init_staging(max_tokens=7, head_dim=32)
                # Every rank queries rows loaded by both ranks. Different token
                # counts prove the lookup no longer requires padded DP slots.
                with pytest.MonkeyPatch.context() as no_comm:
                    forbidden = Mock(side_effect=AssertionError("lookup collective"))
                    no_comm.setattr(dist, "all_gather_into_tensor", forbidden)
                    no_comm.setattr(dist, "all_to_all_single", forbidden)
                    no_comm.setattr(dist, "all_gather_object", forbidden)
                    no_comm.setattr(dist, "barrier", forbidden)
                    for tokens in (1 + rank, 7 - rank, 2 + rank):
                        hashes = (
                            torch.arange(tokens * 5).reshape(tokens, 5) * 7 + rank
                        ) % rows
                        hashes[0, 0] = -1
                        gathered = gather_host_engram_hashes(
                            hashes, dp_shared_memory=True
                        )
                        assert gathered is hashes
                        module.prepare_embeddings(gathered)
                        actual = module._ready_rows(tokens)
                        expected = torch.zeros_like(actual)
                        for token in range(tokens):
                            for head in range(embedding.part_n_hash_cols):
                                column = embedding.head_start + head
                                if column >= len(head_sizes):
                                    continue
                                row = int(hashes[token, column])
                                if (
                                    embedding.vocab_start_idx
                                    <= row
                                    < embedding.vocab_end_idx
                                ):
                                    factor = (
                                        2.0 ** (scales[row].float() - 127)
                                    ).repeat_interleave(16)
                                    expected[token, head] = weight[row].float() * factor
                        assert torch.equal(actual, expected)
                # A replaced Parameter must not silently detach from shared RAM.
                with pytest.raises(RuntimeError, match="storage must not be replaced"):
                    embedding._shared_memory.get_views(
                        embedding.weight.clone(), embedding.weight_scale_inv
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [False, True], ids=["lookup", "allocation-error"])
def test_shared_host_table(tmp_path, failure):
    mp.spawn(
        _worker,
        args=((tmp_path / "gloo").as_uri(), failure),
        nprocs=2,
    )
