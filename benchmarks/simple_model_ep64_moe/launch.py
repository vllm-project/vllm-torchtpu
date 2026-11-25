# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch one worker per TPU device on each host of an EP64 slice."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host-rank", type=int, required=True, choices=range(8))
parser.add_argument("--hosts", required=True, help="Eight comma-separated worker IPs")
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
hosts = args.hosts.split(",")
if len(hosts) != 8 or len(set(hosts)) != 8:
    parser.error("Pass eight distinct worker addresses in slice order")

root = args.root.resolve() / "b48"
root.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
XLA_FLAGS = (
    "--xla_tpu_use_minor_sharding_for_major_trivial_input=true "
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false "
    "--xla_tpu_ars_combiner_threshold_in_bytes=0 "
    "--xla_tpu_enable_async_collective_merger=false "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true "
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144 "
    "--xla_all_gather_latency_bound_threshold_in_bytes=1048576 "
    "--xla_tpu_sparse_core_all_gather_latency_multiplier=100 "
)
env["LIBTPU_INIT_ARGS"] = (
    env.get("LIBTPU_INIT_ARGS", "")
    + " "
    + XLA_FLAGS
    + " --xla_enable_transpose_trace=true --xla_tpu_profile_traceme_level=10"
)
env.update(
    UT_ROOT=str(root),
    OMP_NUM_THREADS="1",
    VLLM_PLUGINS="",
    MASTER_ADDR=hosts[0],
    MASTER_PORT="29781",
    RANK="0",
    WORLD_SIZE="64",
    LOCAL_WORLD_SIZE="8",
    NNODES="8",
    NODE_RANK=str(args.host_rank),
    JOB_COMPLETION_INDEX=str(args.host_rank),
    TPU_NUM_HOSTS="8",
    TPU_WORKER_HOSTNAMES=args.hosts,
    TPU_WORKER_ID=str(args.host_rank),
    TORCH_TPU_TOPOLOGY="2,4,4,2",
    TORCH_TPU_DEVICES_PER_HOST="8",
    TORCH_TPU_SLICEBUILDER_ADDRESSES=",".join(
        f"{host}:{19870 + i}" for host in hosts for i in range(8)
    ),
    TORCH_TPU_TIER2_COMPILATION_CACHE="simple_model_ep64_moe_repro_" + args.root.name,
    MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE="2",
)
source_root = Path(__file__).resolve().parents[2] / "src"
env["PYTHONPATH"] = os.pathsep.join(
    path for path in (str(source_root), env.get("PYTHONPATH")) if path
)
bench = Path(__file__).with_name("bench_simple_model_ep64.py")
children = {}
for local_rank in range(8):
    rank = args.host_rank * 8 + local_rank
    with (root / f"rank{rank}.log").open("w") as log:
        children[rank] = subprocess.Popen(
            [sys.executable, "-u", str(bench)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**env, "RANK": str(rank), "LOCAL_RANK": str(local_rank)},
        )
codes = {rank: worker.wait() for rank, worker in children.items()}
(root / "exit-codes.json").write_text(json.dumps(codes) + "\n")
print("BATCH_DONE", 48, codes, flush=True)
if any(codes.values()):
    raise SystemExit(1)
(root.parent / "suite-done").write_text("ok\n")
