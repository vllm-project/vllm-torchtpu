# SPDX-License-Identifier: Apache-2.0
"""Launch one worker per saved service rank with an isolated rendezvous."""

import argparse
import json
import os
import resource
import shlex
import subprocess
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hosts", required=True, help="Four ordered host IPs, rank0 host first"
)
parser.add_argument("--worker-env-json", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--master-port", type=int, default=29718)
parser.add_argument("--slice-port-base", type=int, default=19440)
parser.add_argument(
    "--session-id", required=True, help="Same unique numeric ID on all hosts"
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
workers = json.loads(args.worker_env_json.read_text())
ips = args.hosts.split(",")
assert len(ips) == 4 and len(workers) == 8
ranks = [int(w["env"]["RANK"]) for w in workers]
assert len(set(ranks)) == 8 and len({r // 8 for r in ranks}) == 1
assert set(ranks) == set(range(min(ranks), min(ranks) + 8)) and min(ranks) % 8 == 0
run = args.output.resolve()
run.mkdir(parents=True, exist_ok=False)
children = []
for worker in workers:
    env = os.environ.copy()
    env.update(worker["env"])
    rank = int(env["RANK"])
    env.update(
        UT_ROOT=str(run),
        MASTER_ADDR=ips[0],
        MASTER_PORT=str(args.master_port),
        OMP_NUM_THREADS="1",
        WORLD_SIZE="32",
        LOCAL_WORLD_SIZE="8",
        LOCAL_RANK=str(rank % 8),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(root / "src"),
        VLLM_PLUGINS="torchtpu",
        TPU_NUM_HOSTS="4",
        TPU_WORKER_HOSTNAMES=",".join(ips),
        TPU_WORKER_ID=str(rank // 8),
        GLOO_SOCKET_IFNAME="eth0",
        TORCH_TPU_TIER2_COMPILATION_CACHE="overlayer_repro_" + args.session_id,
        TORCH_TPU_XPROF_SESSION_ID=args.session_id,
        REPRO_PROFILE="1" if args.profile else "0",
        ONEHOT_MOE_PERMUTE_THRESHOLD="3840",
        TPU_MOE_DENSE_EXPERT_THRESHOLD="0",
        TPU_MOE_OWNER_OUTPUT_MODE="off",
        RAGGED_GATHER_REDUCE_VERSION="v2",
    )
    env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        f"{ip}:{args.slice_port_base + lane}" for ip in ips for lane in range(8)
    )
    if rank == 0:
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "")
            + " --xla_dump_to="
            + str(run / "rank0/xla_dump")
            + " --xla_dump_hlo_as_text"
        )
    env.pop("TPU_MOE_DECODE_PIPELINE", None)
    env.pop("UT_PROFILE_SINGLE_PAIR", None)
    trace_on = args.profile
    trace_keys = {
        "--xla_enable_transpose_trace",
        "--xla_tpu_profile_traceme_level",
        "--xla_xprof_register_llo_debug_info",
        "--xla_xprof_enable_custom_call_tracing",
    }
    for variable in ["LIBTPU_INIT_ARGS", "XLA_FLAGS"]:
        env[variable] = " ".join(
            t
            for t in shlex.split(env.get(variable, ""))
            if t.split("=", 1)[0] not in trace_keys
        )
    env["LIBTPU_INIT_ARGS"] = env.get("LIBTPU_INIT_ARGS", "") + (
        " --xla_enable_transpose_trace="
        + ("true" if trace_on else "false")
        + " --xla_tpu_profile_traceme_level="
        + ("10" if trace_on else "0")
        + " --xla_tpu_show_overlay_details_in_profiler=true"
        " --xla_tpu_show_overlay_overhead_in_profiler=true"
        " --xla_tpu_show_overlay_waits_in_profiler=true"
        " --xla_tpu_show_overlay_waits_in_xprof=true"
        " --xla_xprof_register_llo_debug_info=true"
        " --xla_xprof_enable_custom_call_tracing=false"
    )
    env["LIBTPU_INIT_ARGS"] += (
        " --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false"
    )
    env["UT_TRACE_MODE"] = "on" if trace_on else "off"
    (run / ("flags-rank" + str(rank) + ".json")).write_text(
        json.dumps(
            {
                "trace_mode": env["UT_TRACE_MODE"],
                "LIBTPU_INIT_ARGS": env["LIBTPU_INIT_ARGS"],
                "XLA_FLAGS": env["XLA_FLAGS"],
            }
        )
    )
    log = run / f"rank{rank}.log"
    with log.open("x") as f:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("bench.py")),
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    children.append((rank, process))
(run / "pids.json").write_text(json.dumps({rank: p.pid for rank, p in children}))
try:
    codes = {rank: p.wait() for rank, p in children}
finally:
    for _, child in children:
        if child.poll() is None:
            child.terminate()
(run / "exit-codes.json").write_text(json.dumps(codes))
sys.exit(int(any(codes.values())))
