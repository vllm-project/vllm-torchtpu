# SPDX-License-Identifier: Apache-2.0
"""stage3_admission.all_staged in a real 4-process gloo group: the one-int32
MIN all-reduce (QWEN4_EXP_PD_FAST_ADMIT) answers what all_gather_object of the
staged flags answers, for every pattern of staged ranks."""

import itertools
import os
import tempfile
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu.distributed.kv_transfer import stage3_admission as adm

WORLD = 4
PATTERNS = list(itertools.product([True, False], repeat=WORLD))


def _rank(rank, init, q):
    dist.init_process_group("gloo", init_method=f"file://{init}", rank=rank, world_size=WORLD)
    group = SimpleNamespace(world_size=WORLD, cpu_group=dist.group.WORLD)
    out = []
    for pattern in PATTERNS:
        staged = pattern[rank]
        out.append((adm.all_staged(staged, group, False), adm.all_staged(staged, group, True)))
    dist.barrier()
    q.put((rank, out))
    dist.destroy_process_group()


def test_one_reduce_matches_all_gather_object():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        procs = [ctx.Process(target=_rank, args=(r, init, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        res = [q.get(timeout=300) for _ in procs]
        for p in procs:
            p.join(timeout=60)
    for _, out in res:
        for pattern, (gathered, reduced) in zip(PATTERNS, out):
            assert gathered == reduced == all(pattern), pattern
