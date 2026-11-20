# SPDX-License-Identifier: Apache-2.0
"""HostMaxAllReduce against gloo's all_reduce(MAX)."""

import os
import platform
import random
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu.distributed.host_allreduce import HostMaxAllReduce, usable

WORLD = 4
LENGTH = 10
STEPS = 200


def _worker(rank: int, init_file: str, failures) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        group = dist.new_group(backend="gloo")
        assert usable(group)
        reduce = HostMaxAllReduce(group, LENGTH, timeout_s=60)
        rng = random.Random(rank)
        sent, got = [], []
        for _ in range(STEPS):
            values = torch.tensor(
                [rng.randint(-(10**12), 10**12) for _ in range(LENGTH)],
                dtype=torch.int64,
            )
            if rng.random() < 0.2:  # ranks arrive out of step
                time.sleep(rng.random() * 2e-3)
            sent.append(values)
            got.append(reduce(values))
        # Checked once at the end, so the steps above ran without a resync.
        expected = torch.stack(sent)
        dist.all_reduce(expected, op=dist.ReduceOp.MAX, group=group)
        assert torch.equal(torch.stack(got), expected), rank
        dist.destroy_process_group()
    except Exception as e:  # reported to the parent, which fails the test
        failures.put(f"rank {rank}: {e!r}")


@pytest.mark.cpu_test
@pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"), reason="x86-only reducer"
)
def test_host_max_allreduce_matches_gloo(tmp_path):
    failures = mp.get_context("spawn").Queue()
    init_file = os.path.join(tmp_path, "init")
    mp.start_processes(
        _worker, args=(init_file, failures), nprocs=WORLD, start_method="spawn"
    )
    assert failures.empty(), failures.get()
    assert not [p for p in os.listdir("/dev/shm") if p.startswith("vllm_tpu_dp_max_")]
