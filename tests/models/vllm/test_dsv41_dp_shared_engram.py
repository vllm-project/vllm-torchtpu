# SPDX-License-Identifier: Apache-2.0
"""CPU distributed regression for shared and head-sharded host Engram tables."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import host_engram

_HEAD_SIZES = (8, 8, 8, 8)
_DIM = 32
_BLOCK = 16


def _group(rank: int, size: int) -> SimpleNamespace:
    def broadcast_object(obj=None):
        objects = [obj]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    return SimpleNamespace(
        world_size=size,
        rank_in_group=rank,
        cpu_group=dist.group.WORLD,
        broadcast_object=broadcast_object,
    )


def _table(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (torch.arange(rows * _DIM) % 120).to(torch.uint8).view(rows, _DIM)
    scales = (127 + torch.arange(rows * (_DIM // _BLOCK)) % 3).to(torch.uint8)
    return weight.view(torch.float8_e4m3fn), scales.view(rows, -1)


def _worker(
    rank: int, size: int, rendezvous: str, shm_dir: str, shared_memory: bool
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=45),
    )
    upstream = common_engram.upstream
    group = _group(rank, size)
    upstream.SHM_PATH = shm_dir
    upstream.get_engram_dp_group = lambda: group
    upstream.get_engram_dp_size = lambda: size
    upstream.get_tensor_model_parallel_world_size = lambda: 1
    upstream.get_tensor_model_parallel_rank = lambda: 0
    torch.accelerator.current_device_index = lambda: 0
    try:
        emb = common_engram.ParallelEngramEmbedding(
            sum(_HEAD_SIZES),
            _DIM,
            _HEAD_SIZES,
            block_size=_BLOCK,
            cpu_offload=True,
            dp_shared_memory=shared_memory,
        )
        shards = 1 if shared_memory else size
        assert emb.dp_size == shards
        assert emb.part_n_hash_cols == len(_HEAD_SIZES) // shards
        assert emb.part_num_embeddings == sum(_HEAD_SIZES) // shards
        dist.barrier()
        # The leader unlinks the backing file once every rank has mapped it.
        assert os.listdir(shm_dir) == []

        weight, scales = _table(sum(_HEAD_SIZES))
        start, rows = (
            common_engram._shared_row_range(emb.part_num_embeddings, group)
            if shared_memory
            else (emb.vocab_start_idx, emb.part_num_embeddings)
        )
        # Run:AI streams only this rank's row slice; the other loader sees
        # the full table. Both write disjoint ranges of one mapping.
        loaded = weight.narrow(0, start, rows) if rank == 0 else weight
        emb.weight.weight_loader(emb.weight, loaded)
        emb.weight_scale_inv.weight_loader(emb.weight_scale_inv, scales)
        local_weight = weight.narrow(0, emb.vocab_start_idx, emb.part_num_embeddings)
        local_scales = scales.narrow(0, emb.vocab_start_idx, emb.part_num_embeddings)
        assert torch.equal(emb.weight.view(torch.uint8), local_weight.view(torch.uint8))
        assert torch.equal(emb.weight_scale_inv, local_scales)

        host_engram.get_engram_dp_group = lambda: group
        host_engram.get_dp_group = lambda: group
        host_engram.is_forward_context_available = lambda: True
        context = SimpleNamespace(additional_kwargs={}, dp_metadata=object())
        host_engram.get_forward_context = lambda: context
        engram = object.__new__(common_engram.Engram)
        torch.nn.Module.__init__(engram)
        engram.embed_tokens = emb
        engram._init_staging(2, _DIM)
        for counts in ((2, 1), (1, 2), (0, 2), (2, 0), (2, 1)):
            context.additional_kwargs["tpu_num_tokens_across_dp_cpu"] = counts
            ids = torch.tensor(
                [[1 + rank, 9, 17, 25 + rank], [0, 8, 16, 24]], dtype=torch.int32
            )[: counts[rank]]
            gathered = host_engram.gather_host_engram_hashes(
                ids,
                dp_shared_memory=shared_memory,
                gather_fn=upstream.gather_engram_hashes,
            )
            if shared_memory:
                assert torch.equal(gathered, ids)
            else:
                assert gathered.shape[0] == size * max(counts)
            pending = engram.start_prepare_embeddings(gathered)
            assert (pending.exchange is None) is shared_memory
            engram.finish_prepare_embeddings(pending)
            actual = engram._ready_rows(ids.shape[0])
            expected = common_engram.dequant_fp8_rows(
                weight.view(torch.uint8)[ids.long()], scales[ids.long()], _BLOCK
            )
            assert torch.equal(actual, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_test
@pytest.mark.parametrize("shared_memory", [False, True])
def test_dp_host_table_loading_and_staging(tmp_path, shared_memory):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    size = 2
    mp.spawn(
        _worker,
        args=(size, (tmp_path / "gloo").as_uri(), str(shm_dir), shared_memory),
        nprocs=size,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("rows", "size", "expected"),
    [
        (10, 4, [(0, 3), (3, 3), (6, 3), (9, 1)]),
        (2, 4, [(0, 1), (1, 1), (2, 0), (2, 0)]),
    ],
)
def test_shared_row_range_covers_table(rows, size, expected):
    ranges = [
        common_engram._shared_row_range(
            rows, SimpleNamespace(world_size=size, rank_in_group=rank)
        )
        for rank in range(size)
    ]
    assert ranges == expected


@pytest.mark.cpu_test
def test_dp_shared_table_requires_host_offload():
    with pytest.raises(ValueError, match="requires Engram CPU offload"):
        common_engram.ParallelEngramEmbedding(
            32, _DIM, _HEAD_SIZES, cpu_offload=False, dp_shared_memory=True
        )
