# SPDX-License-Identifier: Apache-2.0
"""CPU test: two vLLM instances on one host share one /dev/shm Engram table.

With TPU_DEEPSEEK_V41_ENGRAM_SHM_SHARE_KEY set, the first instance (a DP group
of 2 ranks here) creates and fills a named backing file and marks it ready.
The second instance maps the same file and does not write it: it is handed a
different table, and the test checks that it still reads the first one.
"""

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


def _table(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = sum(_HEAD_SIZES)
    weight = ((torch.arange(rows * _DIM) + 8 * seed) % 120).to(torch.uint8)
    scales = (127 + (torch.arange(rows * (_DIM // _BLOCK)) + seed) % 3).to(torch.uint8)
    return weight.view(rows, _DIM).view(torch.float8_e4m3fn), scales.view(rows, -1)


def _instance(rank: int, size: int, rendezvous: str, shm_dir: str, seed: int) -> None:
    torch.set_num_threads(1)
    os.environ["TPU_DEEPSEEK_V41_ENGRAM_SHM_SHARE_KEY"] = "test"
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=size,
        timeout=timedelta(seconds=45),
    )
    cpu_group = dist.new_group(list(range(size)), backend="gloo")

    def broadcast_object(obj=None):
        objects = [obj]
        dist.broadcast_object_list(objects, src=0, group=cpu_group)
        return objects[0]

    group = SimpleNamespace(world_size=size, rank_in_group=rank, cpu_group=cpu_group,
                            broadcast_object=broadcast_object)
    upstream = common_engram.upstream
    upstream.SHM_PATH = shm_dir
    upstream.get_engram_dp_group = lambda: group
    upstream.get_engram_dp_size = lambda: size
    upstream.get_tensor_model_parallel_world_size = lambda: 1
    upstream.get_tensor_model_parallel_rank = lambda: 0
    torch.accelerator.current_device_index = lambda: 0
    try:
        emb = common_engram.ParallelEngramEmbedding(
            sum(_HEAD_SIZES), _DIM, _HEAD_SIZES, block_size=_BLOCK,
            cpu_offload=True, dp_shared_memory=True,
        )
        weight, scales = _table(seed)
        start, rows = common_engram._shared_row_range(emb.part_num_embeddings, group)
        emb.weight.weight_loader(emb.weight, weight.narrow(0, start, rows))
        emb.weight_scale_inv.weight_loader(emb.weight_scale_inv, scales)
        dist.barrier()
        # Whatever this instance was handed, the table is the first instance's.
        first_weight, first_scales = _table(0)
        assert torch.equal(emb.weight.view(torch.uint8), first_weight.view(torch.uint8))
        assert torch.equal(emb.weight_scale_inv, first_scales)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_test
def test_second_instance_maps_the_first_instances_table(tmp_path):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    for seed, name in ((0, "first"), (1, "second")):
        mp.spawn(_instance, args=(2, (tmp_path / f"gloo-{name}").as_uri(), str(shm_dir), seed),
                 nprocs=2, join=True)
        files = sorted(os.listdir(shm_dir))
        assert len(files) == 2 and files[1] == files[0] + ".ready", files


@pytest.mark.cpu_test
def test_file_without_marker_is_rewritten(tmp_path):
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    mp.spawn(_instance, args=(2, (tmp_path / "gloo-a").as_uri(), str(shm_dir), 0),
             nprocs=2, join=True)
    marker = next(f for f in os.listdir(shm_dir) if f.endswith(".ready"))
    os.unlink(shm_dir / marker)   # as if the first instance died while loading
    # The next instance writes the table again (seed 0 again, so the check holds).
    mp.spawn(_instance, args=(2, (tmp_path / "gloo-b").as_uri(), str(shm_dir), 0),
             nprocs=2, join=True)
    assert (shm_dir / marker).exists()
