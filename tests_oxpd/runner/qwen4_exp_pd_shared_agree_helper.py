# SPDX-License-Identifier: Apache-2.0
"""One rank of the QWEN4_EXP_PD_SHARED_COMPRESSED agreement test (spawned by
test_qwen4_exp_pd_shared_wiring.py; not a test module).

``python qwen4_exp_pd_shared_agree_helper.py <rank> <world> <port> <missing>``:
joins a gloo group, stages loads a, b, c on this rank (``missing`` lacks b),
and calls the worker's real ``repack_landed(["a", "b", "c"])``. Each landing
runs one all-reduce on the same group, standing in for the device
all-gather: ranks that disagree on the loads would wait in it forever.
Prints one JSON line of what this rank ran and released.
"""

import json
import sys

import torch
import torch.distributed as dist

from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker


class Staging:
    shared = object()  # QWEN4_EXP_PD_SHARED_COMPRESSED on

    def __init__(self):
        self.events = []

    def plan_repack(self, uuid, tables, num_tokens):
        self.events.append(("plan", uuid))
        return uuid

    def run_repack(self, plan):
        dist.all_reduce(torch.ones(1))  # the landing collective
        self.events.append(("run", plan))

    def release(self, uuid):
        self.events.append(("release", uuid))


def main():
    rank, world, port, missing = (int(x) for x in sys.argv[1:5])
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    w = object.__new__(Qwen4ExpRaidenWorker)
    w._staging = Staging()
    w._sync_on = lambda: True
    w.tp_rank = rank
    w._tp_cpu_group = lambda: dist.group.WORLD
    w._pending_repacks = {
        name: (uuid, [[1]], 4096)
        for name, uuid in (("a", 1), ("b", 2), ("c", 3))
        if not (name == "b" and rank == missing)
    }
    w.repack_landed(["a", "b", "c"])
    print(json.dumps({"rank": rank, "events": w._staging.events, "left": sorted(w._pending_repacks)}))
    sys.stdout.flush()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
