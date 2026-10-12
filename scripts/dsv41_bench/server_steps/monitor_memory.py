# SPDX-License-Identifier: Apache-2.0
"""Persist Linux host, cgroup, shm and per-process memory during qualification."""

import argparse
import json
import os
import time
from pathlib import Path


def snapshot() -> dict:
    cgroup = Path("/sys/fs/cgroup")
    memory = {}
    for name in ["memory.current", "memory.peak", "memory.max", "memory.events"]:
        path = cgroup / name
        if path.exists():
            memory[name] = path.read_text().strip()
    processes = []
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(
                line.split(":", 1) for line in status.read_text().splitlines()
            )
            cmd = (
                (status.parent / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if "python" in cmd.lower() or "vllm" in cmd.lower():
            processes.append(
                {
                    "pid": int(status.parent.name),
                    "cmd": cmd[:512],
                    **{
                        k: fields.get(k, "").strip()
                        for k in ["VmRSS", "VmHWM", "RssAnon", "RssShmem"]
                    },
                }
            )
    shm = os.statvfs("/dev/shm")
    return {
        "time_ns": time.time_ns(),
        "host_meminfo": Path("/proc/meminfo").read_text(),
        "cgroup": memory,
        "processes": processes,
        "shm_total": shm.f_blocks * shm.f_frsize,
        "shm_used": (shm.f_blocks - shm.f_bfree) * shm.f_frsize,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("a", buffering=1) as output:
        while True:
            output.write(json.dumps(snapshot()) + "\n")
            output.flush()
            time.sleep(5)


if __name__ == "__main__":
    main()
