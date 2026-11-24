# SPDX-License-Identifier: Apache-2.0
"""The shared-memory DP all-reduce returns gloo's MAX on every call."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm_torchtpu.distributed import shm_allreduce

WORLD = 8
CALLS = 3000


def _rank_main(rank: int, init_file: str, enabled: bool, queue) -> None:
    os.environ["VLLM_TPU_SHM_DP_ALLREDUCE"] = "1" if enabled else "0"
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    group = SimpleNamespace(
        rank_in_group=rank, world_size=WORLD, cpu_group=torch.distributed.group.WORLD
    )
    generator = torch.Generator().manual_seed(rank)
    mismatches = 0
    for call in range(CALLS):
        # The 2 sites interleave unevenly, as decode and prefill steps do.
        site, width = ("step_start", 10) if call % 3 else ("decoder_replay", 1)
        t = torch.randint(-(2**62), 2**62, (width,), generator=generator)
        expected = t.clone()
        torch.distributed.all_reduce(expected, op=torch.distributed.ReduceOp.MAX)
        shm_allreduce.dp_all_reduce_max(t, group, site)
        mismatches += int(not torch.equal(t, expected))
    paths = {s: r is not None for s, r in shm_allreduce._reducers.items()}
    queue.put((rank, mismatches, paths))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("enabled", [True, False])
def test_matches_gloo_max_on_every_rank(enabled):
    with tempfile.TemporaryDirectory() as tmp:
        context = mp.get_context("spawn")
        queue = context.Queue()
        procs = [
            context.Process(target=_rank_main, args=(r, f"{tmp}/init", enabled, queue))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        results = sorted(queue.get(timeout=300) for _ in procs)
        for p in procs:
            p.join(timeout=60)
    assert [r[1] for r in results] == [0] * WORLD
    expected_path = {"step_start": enabled, "decoder_replay": enabled}
    assert all(r[2] == expected_path for r in results)
    assert not [f for f in os.listdir("/dev/shm") if f.startswith("vllm_tpu_dp_")]
