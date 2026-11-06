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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# tools/megakernel/harness.py
"""Launching and bringing up a megakernel program without vLLM.

:func:`launch` (on the login node) re-runs the tool as one process per TPU
device of an existing allocation, 8 per host, the way vLLM's workers are laid
out. In each process :func:`init` reproduces the environment the Ray executor
gives a worker (``TORCH_TPU_SLICEBUILDER_ADDRESSES``, ``TORCH_TPU_TOPOLOGY``,
torchrun-style ranks) and opens the ``tpu_dist`` process group; programs then
build their ops exactly as they do inside the server.
"""

from __future__ import annotations

import importlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import NoReturn, Protocol

import torch

from .. import settings

T0 = time.perf_counter()
REPO = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()),
    Path.cwd(),
)
LOCAL_WORLD = 8
WORKER_ENV = "MEGAKERNEL_TOOL_WORKER"
_REQUIRED_HOSTS = 4
_BASE_MASTER_PORT = 20000
_PORT_SPAN = 20000


class ConfigKeyLike(Protocol):
    """Protocol for configuration objects exposing a ``key()`` digest."""

    def key(self) -> str:
        """Return short configuration digest string."""
        ...


def _capture_stdout(args: list[str]) -> str:
    """Spawn ``args`` via ``os.posix_spawnp`` and return its captured stdout."""
    read_fd, write_fd = os.pipe()
    actions = [
        (os.POSIX_SPAWN_DUP2, write_fd, 1),
        (os.POSIX_SPAWN_CLOSE, read_fd),
    ]
    pid = os.posix_spawnp(args[0], args, os.environ, file_actions=actions)
    os.close(write_fd)
    with os.fdopen(read_fd, encoding="utf-8") as handle:
        output = handle.read()
    os.waitpid(pid, 0)
    return output


def _spawn_wait(args: list[str], env: dict[str, str]) -> int:
    """Spawn ``args`` with ``env`` and return its process exit code."""
    pid = os.posix_spawnp(args[0], args, env)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def is_worker() -> bool:
    """Return True when running inside a spawned per-device worker process."""
    return settings.MEGAKERNEL_TOOL_WORKER == "1"


def rank() -> int:
    """Return the current process rank from ``SLURM_PROCID``."""
    return settings.SLURM_PROCID


def log_host(msg: str) -> None:
    """Write a host-tagged status line from the current process."""
    elapsed = time.perf_counter() - T0
    sys.stdout.write(f"[{elapsed:7.1f}s host {settings.SLURM_NODEID}] {msg}\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    """Write a rank-0 status line to standard output."""
    if rank() == 0:
        elapsed = time.perf_counter() - T0
        sys.stdout.write(f"[{elapsed:7.1f}s r{rank():02d}] {msg}\n")
        sys.stdout.flush()


def launch(
    argv: list[str],
    jobid: str,
    venv: str,
    extra_env: dict[str, str],
    per_host: int = LOCAL_WORLD,
) -> int:
    """Run ``python -m vllm_torchtpu.models.kimi_k3.megakernel.tools`` on ``jobid``."""
    nodelist = _capture_stdout(["squeue", "-j", jobid, "-h", "-o", "%N"]).strip()
    if not nodelist:
        msg = f"job {jobid} is not running"
        raise SystemExit(msg)
    hosts = _capture_stdout(["scontrol", "show", "hostnames", nodelist]).split()
    if len(hosts) != _REQUIRED_HOSTS:
        msg = (
            f"job {jobid} holds {len(hosts)} hosts; "
            f"the K3 programs need {_REQUIRED_HOSTS} (32 devices)"
        )
        raise SystemExit(msg)
    ips = [_capture_stdout(["getent", "hosts", h]).split()[0] for h in hosts]
    env = dict(os.environ)
    env.update(extra_env)
    default_port = str(_BASE_MASTER_PORT + secrets.randbelow(_PORT_SPAN))
    env.update(
        {
            WORKER_ENV: "1",
            "PYTHONPATH": f"{REPO / 'src'}:{REPO}:{env.get('PYTHONPATH', '')}",
            "TP32_HOST_IPS": ",".join(ips),
            "MASTER_PORT": env.get("MASTER_PORT", default_port),
            "TP32_SESSION": str(time.time_ns()),
            "HF_HUB_OFFLINE": "1",
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", ""),
        }
    )
    command = [
        "srun",
        f"--jobid={jobid}",
        "--overlap",
        f"--nodes={_REQUIRED_HOSTS}",
        f"--ntasks={_REQUIRED_HOSTS * per_host}",
        f"--ntasks-per-node={per_host}",
        "--kill-on-bad-exit=1",
        "--unbuffered",
        "--export=ALL",
        f"--chdir={REPO}",
        f"{venv}/bin/python",
        "-m",
        "vllm_torchtpu.models.kimi_k3.megakernel.tools",
        *argv,
    ]
    return _spawn_wait(command, env)


def init() -> None:
    """Bring up ``torch_tpu`` and the ``tpu_dist`` process group in a worker."""
    host_ips = settings.TP32_HOST_IPS.split(",")
    base_port = settings.TORCH_TPU_BASE_PORT
    os.environ.update(
        {
            "RANK": str(rank()),
            "LOCAL_RANK": settings.SLURM_LOCALID,
            "WORLD_SIZE": str(len(host_ips) * LOCAL_WORLD),
            "LOCAL_WORLD_SIZE": str(LOCAL_WORLD),
            "NNODES": str(len(host_ips)),
            "NODE_RANK": settings.SLURM_NODEID,
            "TPU_NUM_HOSTS": str(len(host_ips)),
            "MASTER_ADDR": host_ips[0],
            "TORCH_TPU_TOPOLOGY": "2,2,4,2",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
                f"{ip}:{base_port + i}" for ip in host_ips for i in range(LOCAL_WORLD)
            ),
        }
    )
    os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID", settings.TP32_SESSION)
    importlib.import_module("torch_tpu")

    log("torch_tpu imported")
    torch.distributed.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://{host_ips[0]}:{os.environ['MASTER_PORT']}",
        world_size=len(host_ips) * LOCAL_WORLD,
        rank=rank(),
    )
    log("tpu_dist process group up")


def sync() -> None:
    """Synchronize all workers across the distributed process group."""
    torch.distributed.barrier()


def git_revision() -> str:
    """Return the short git HEAD revision, suffixed with ``+dirty`` if modified."""
    out = _capture_stdout(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])
    dirty = _capture_stdout(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]
    ).strip()
    return out.strip() + ("+dirty" if dirty else "")


class Records:
    """One JSON record per result (rank 0), with what produced it."""

    def __init__(
        self,
        path: str | None,
        command: str,
        config: ConfigKeyLike,
    ) -> None:
        """Initialize JSONL record writer for ``command`` and ``config``."""
        self.path, self.command = path, command
        self.context: dict[str, object] = {
            "git": git_revision(),
            "config": config.key(),
            "jobid": settings.SLURM_JOB_ID,
        }

    def add(self, **fields: object) -> None:
        """Append one JSON record with ``fields`` on rank 0 when ``path`` is set."""
        if rank() != 0 or not self.path:
            return
        record = {
            "command": self.command,
            **fields,
            **self.context,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with Path(self.path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def fail(message: str) -> NoReturn:
    """Log failure ``message`` and exit with status code 1."""
    log(f"FAIL: {message}")
    sys.exit(1)
