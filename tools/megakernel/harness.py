# SPDX-License-Identifier: Apache-2.0
"""Launching and bringing up a megakernel program without vLLM.

:func:`launch` (on the login node) re-runs the tool as one process per TPU
device of an existing allocation, 8 per host, the way vLLM's workers are laid
out. In each process :func:`init` reproduces the environment the Ray executor
gives a worker (``TORCH_TPU_SLICEBUILDER_ADDRESSES``, ``TORCH_TPU_TOPOLOGY``,
torchrun-style ranks) and opens the ``tpu_dist`` process group; programs then
build their ops exactly as they do inside the server.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
REPO = Path(__file__).resolve().parents[2]
LOCAL_WORLD = 8
WORKER_ENV = "MEGAKERNEL_TOOL_WORKER"


def is_worker() -> bool:
    return os.environ.get(WORKER_ENV) == "1"


def rank() -> int:
    return int(os.environ.get("SLURM_PROCID", "0"))


def log_host(msg: str) -> None:
    """A line from every process, tagged with its host (host-side tools)."""
    print(
        f"[{time.perf_counter() - T0:7.1f}s host "
        f"{os.environ.get('SLURM_NODEID', '0')}] {msg}",
        flush=True,
    )


def log(msg: str) -> None:
    if rank() == 0:
        print(f"[{time.perf_counter() - T0:7.1f}s r{rank():02d}] {msg}", flush=True)


def launch(
    argv: list[str],
    jobid: str,
    venv: str,
    extra_env: dict[str, str],
    per_host: int = LOCAL_WORLD,
) -> int:
    """Run ``python -m tools.megakernel argv`` on the four hosts of ``jobid``,
    ``per_host`` processes each (default one per TPU device)."""
    nodelist = subprocess.run(
        ["squeue", "-j", jobid, "-h", "-o", "%N"], capture_output=True, text=True
    ).stdout.strip()
    if not nodelist:
        raise SystemExit(f"job {jobid} is not running")
    hosts = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True
    ).stdout.split()
    if len(hosts) != 4:
        raise SystemExit(
            f"job {jobid} holds {len(hosts)} hosts; the K3 programs need 4 (32 devices)"
        )
    ips = [
        subprocess.run(
            ["getent", "hosts", h], capture_output=True, text=True
        ).stdout.split()[0]
        for h in hosts
    ]
    env = dict(os.environ)
    env.update(extra_env)
    env.update(
        {
            WORKER_ENV: "1",
            "PYTHONPATH": f"{REPO / 'src'}:{REPO}:{env.get('PYTHONPATH', '')}",
            "TP32_HOST_IPS": ",".join(ips),
            "MASTER_PORT": env.get("MASTER_PORT", str(20000 + random.randrange(20000))),
            "TP32_SESSION": str(time.time_ns()),
            "HF_HUB_OFFLINE": "1",
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", ""),
        }
    )
    command = [
        "srun",
        f"--jobid={jobid}",
        "--overlap",
        "--nodes=4",
        f"--ntasks={4 * per_host}",
        f"--ntasks-per-node={per_host}",
        "--kill-on-bad-exit=1",
        "--unbuffered",
        "--export=ALL",
        f"--chdir={REPO}",
        f"{venv}/bin/python",
        "-m",
        "tools.megakernel",
        *argv,
    ]
    return subprocess.call(command, env=env)


def init() -> None:
    """Bring up torch_tpu and the ``tpu_dist`` process group in a worker."""
    host_ips = os.environ["TP32_HOST_IPS"].split(",")
    base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
    os.environ.update(
        {
            "RANK": str(rank()),
            "LOCAL_RANK": os.environ["SLURM_LOCALID"],
            "WORLD_SIZE": str(len(host_ips) * LOCAL_WORLD),
            "LOCAL_WORLD_SIZE": str(LOCAL_WORLD),
            "NNODES": str(len(host_ips)),
            "NODE_RANK": os.environ["SLURM_NODEID"],
            "TPU_NUM_HOSTS": str(len(host_ips)),
            "MASTER_ADDR": host_ips[0],
            "TORCH_TPU_TOPOLOGY": "2,2,4,2",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
                f"{ip}:{base_port + i}" for ip in host_ips for i in range(LOCAL_WORLD)
            ),
        }
    )
    os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", os.environ["TP32_SESSION"])
    import torch
    import torch_tpu  # noqa: F401

    log("torch_tpu imported")
    torch.distributed.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://{host_ips[0]}:{os.environ['MASTER_PORT']}",
        world_size=len(host_ips) * LOCAL_WORLD,
        rank=rank(),
    )
    log("tpu_dist process group up")


def sync() -> None:
    import torch

    torch.distributed.barrier()


def git_revision() -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out.stdout.strip() + ("+dirty" if dirty else "")


class Records:
    """One JSON record per result (rank 0), with what produced it."""

    def __init__(self, path: str | None, command: str, config) -> None:
        self.path, self.command = path, command
        self.context = {
            "git": git_revision(),
            "config": config.key(),
            "jobid": os.environ.get("SLURM_JOB_ID", ""),
        }

    def add(self, **fields) -> None:
        if rank() != 0 or not self.path:
            return
        record = {
            "command": self.command,
            **fields,
            **self.context,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def fail(message: str) -> None:
    log(f"FAIL: {message}")
    sys.exit(1)
