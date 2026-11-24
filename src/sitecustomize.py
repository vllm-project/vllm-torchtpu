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
"""Process-wide defaults required before torch_tpu autoloads."""

import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")


# CDK_MEMLOG=1 (diagnostics, this staging tree only): one process per container logs host memory
# (/proc/meminfo, cgroup memory.current/max, /dev/shm and /tmp use) every 20 s to stderr.
def _start_memlog():
    if os.environ.get("CDK_MEMLOG") != "1":
        return
    lock = "/tmp/.cdk_memlog.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        # Held: take it over only if its holder has exited (a short-lived helper).
        try:
            holder = int(open(lock).read() or 0)
            os.kill(holder, 0)
            return
        except (OSError, ValueError):
            pass
        try:
            os.unlink(lock)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    import shutil, sys, threading, time

    def read(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return "?"

    def loop():
        while True:
            mi = {}
            for line in read("/proc/meminfo").splitlines():
                k, _, v = line.partition(":")
                mi[k] = v.strip()
            du = {}
            for d in ("/dev/shm", "/tmp"):
                try:
                    u = shutil.disk_usage(d); du[d] = u.used >> 30
                except OSError:
                    du[d] = "?"
            cur = read("/sys/fs/cgroup/memory.current")
            mx = read("/sys/fs/cgroup/memory.max")
            procs = []
            for pid in os.listdir("/proc"):
                if pid.isdigit():
                    st = read(f"/proc/{pid}/status")
                    rss = next((l.split()[1] for l in st.splitlines() if l.startswith("VmRSS:")), None)
                    name = next((l.split()[1] for l in st.splitlines() if l.startswith("Name:")), "?")
                    if rss:
                        procs.append((int(rss) >> 20, name, pid))
            top = " ".join(f"{n}:{p}={g}G" for g, n, p in sorted(procs, reverse=True)[:5])
            print(f"CDK_MEMLOG {time.strftime('%H:%M:%S')} top_rss[{top}] MemTotal={mi.get('MemTotal')} "
                  f"MemAvailable={mi.get('MemAvailable')} Shmem={mi.get('Shmem')} "
                  f"Mlocked={mi.get('Mlocked')} cg_current={cur} cg_max={mx} "
                  f"shm_used_GiB={du['/dev/shm']} tmp_used_GiB={du['/tmp']}", file=sys.stderr, flush=True)
            time.sleep(20)

    threading.Thread(target=loop, name="cdk-memlog", daemon=True).start()


_start_memlog()
