# SPDX-License-Identifier: Apache-2.0
"""CPU-only distributed regression for DP-shared host Engram tables."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram

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


def _worker(rank: int, size: int, rendezvous: str, shm_dir: str) -> None:
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
            dp_shared_memory=True,
        )
        # Every replica keeps all (TP-local) heads and needs no DP exchange.
        assert emb.dp_size == 1
        assert emb.part_n_hash_cols == len(_HEAD_SIZES)
        assert emb.part_num_embeddings == sum(_HEAD_SIZES)
        dist.barrier()
        # The leader unlinks the backing file once every rank has mapped it.
        assert os.listdir(shm_dir) == []

        weight, scales = _table(emb.part_num_embeddings)
        start, rows = common_engram._shared_row_range(emb.part_num_embeddings, group)
        # Run:AI streams only this rank's row slice; the other loader sees
        # the full table. Both write disjoint ranges of one mapping.
        loaded = weight.narrow(0, start, rows) if rank == 0 else weight
        emb.weight.weight_loader(emb.weight, loaded)
        emb.weight_scale_inv.weight_loader(emb.weight_scale_inv, scales)
        assert torch.equal(emb.weight.view(torch.uint8), weight.view(torch.uint8))
        assert torch.equal(emb.weight_scale_inv, scales)

        # Each replica looks up its own tokens in the shared table.
        ids = torch.tensor([[1 + rank, 9, 17, 25 + rank], [0, 8, 16, 24]])
        raw = torch.empty(2, len(_HEAD_SIZES), _DIM, dtype=torch.uint8)
        out_scales = torch.empty(2, len(_HEAD_SIZES), _DIM // _BLOCK, dtype=torch.uint8)
        emb.gather_raw(ids, raw, out_scales)
        assert torch.equal(raw, weight.view(torch.uint8)[ids])
        assert torch.equal(out_scales, scales[ids])
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_test
def test_dp_shared_table_maps_once_and_loads_disjoint_ranges(tmp_path):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    size = 2
    mp.spawn(
        _worker,
        args=(size, (tmp_path / "gloo").as_uri(), str(shm_dir)),
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
