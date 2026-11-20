# SPDX-License-Identifier: Apache-2.0
"""TIER-X2 sidecars: Raiden's global registry (prefill pod) and the host store nodes (decode pod).

The served recipes take environment variables per leg but no extra commands, so the processes TIER-X2 needs
next to the engines are started from here, once per pod, by the first process that imports vllm_torchtpu while
the pod's environment asks for them. Inert unless ``Q38X2_REGISTRY_PORT`` or ``Q38X2_NODES`` is set.

- ``Q38X2_REGISTRY_PORT=<port>`` (prefill): run ``global_registry_server --port=<port>`` from the installed
  tpu_sync wheel. The prefill's shard stores dial it on 127.0.0.1 (``VLLM_TPU_OFFLOAD_TIER_X2``).
- ``Q38X2_NODES=<shard>:<bytes>[;<shard>:<bytes>...]`` (decode): run one ``kv_cache_host_store_node_main`` per
  shard, each lending ``<bytes>`` of this host's DRAM to the pool group ``<Q38X2_NODE_GROUP_PREFIX>-<shard>``
  (the prefill's ``kv_pool_group`` plus the shard name), registered with the registry at ``Q38X2_NODE_REGISTRY``
  and bound to ``Q38X2_NODE_IP`` (both may name ``${VAR}``), controller port ``Q38X2_NODE_PORT_BASE`` (31700)
  plus the shard's position.
- ``Q38X2_TPU_SYNC_VERSION``: nothing starts until the installed tpu_sync_torch has this version (the leg
  installs the car's wheel before vllm serve; the image's older wheel must not answer instead).

A detached watcher (its own session, so it outlives the importing process) starts the binaries and prints one
line per event to stderr and to ``$CDK_OUTPUT_DIR/q38x2/sidecar.log``: ``Q38X2_REGISTRY_UP``,
``Q38X2_NODE_UP shard=... boot_s=... rss_gib=...`` once a node's controller port answers (the node's own
"KVCacheHostStoreNode up" is an INFO log its binary does not print), ``Q38X2_NODE_EXIT`` / ``Q38X2_REGISTRY_EXIT``
when one ends, and ``Q38X2_MEM`` every minute: the nodes' resident bytes, the host's MemAvailable, and this
container's cgroup memory. Standard library only.
"""

import os
import socket
import subprocess
import sys
import time

REGISTRY_PORT = "Q38X2_REGISTRY_PORT"
NODES = "Q38X2_NODES"
NODE_REGISTRY = "Q38X2_NODE_REGISTRY"
NODE_IP = "Q38X2_NODE_IP"
NODE_GROUP_PREFIX = "Q38X2_NODE_GROUP_PREFIX"
NODE_PORT_BASE = "Q38X2_NODE_PORT_BASE"
TPU_SYNC_VERSION = "Q38X2_TPU_SYNC_VERSION"
# Where the lock and the executable copies live (tests point them elsewhere); one lock per pod.
LOCK_DIR = "Q38X2_LOCK_DIR"
BIN_DIR = "Q38X2_BIN_DIR"

REGISTRY_BIN = ("kv_cache", "global_registry", "global_registry_server")
NODE_BIN = ("store_node", "kv_cache_host_store_node_main")
MEM_EVERY_S = 60.0


def _expand(name: str, value: str, env) -> str:
    """``$VAR`` / ``${VAR}`` from ``env``; refused when one is unset."""
    import re

    def sub(m):
        var = m.group(1) or m.group(2)
        if var not in env:
            raise ValueError(f"{name}={value!r} names an unset environment variable ({var})")
        return env[var]

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, value)


def parse_nodes(spec: str) -> list[tuple[str, int]]:
    """``<shard>:<bytes>;...`` -> [(shard, bytes)], in order; refuses an empty, duplicate or malformed entry."""
    out: list[tuple[str, int]] = []
    for item in spec.split(";"):
        shard, sep, size = item.strip().partition(":")
        if not sep or not shard or not size.isdigit() or int(size) <= 0:
            raise ValueError(f"{NODES} entry {item!r} is not <shard>:<bytes>")
        if shard in {s for s, _ in out}:
            raise ValueError(f"{NODES} names shard {shard!r} twice")
        out.append((shard, int(size)))
    return out


def node_commands(env, node_bin: str) -> list[tuple[str, int, list[str]]]:
    """[(shard, controller port, argv)] for the decode pod's store nodes."""
    registry = _expand(NODE_REGISTRY, env.get(NODE_REGISTRY, ""), env)
    ip = _expand(NODE_IP, env.get(NODE_IP, ""), env)
    if not registry or not ip:
        raise ValueError(f"{NODES} needs {NODE_REGISTRY} and {NODE_IP}")
    prefix = env.get(NODE_GROUP_PREFIX, "q38x2")
    base = int(env.get(NODE_PORT_BASE, "31700"))
    out = []
    for i, (shard, size) in enumerate(parse_nodes(env[NODES])):
        port = base + i
        out.append(
            (
                shard,
                port,
                [
                    node_bin,
                    f"--job_name=q38x2-node-{shard}",
                    f"--global_registry_address={registry}",
                    f"--kv_pool_group={prefix}-{shard}",
                    f"--store_server_ip={ip}",
                    f"--raiden_controller_port={port}",
                    "--evict_tier=1",
                    f"--dram_budget_bytes={size}",
                    "--parallelism=4",
                ],
            )
        )
    return out


def _tpu_sync_dir() -> str:
    import importlib.util

    spec = importlib.util.find_spec("tpu_sync")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("tpu_sync is not installed")
    for root in spec.submodule_search_locations:
        if os.path.exists(os.path.join(root, *NODE_BIN)) or os.path.exists(
            os.path.join(root, *REGISTRY_BIN)
        ):
            return root
    raise RuntimeError(f"no tpu_sync location holds the TIER-X2 binaries: {list(spec.submodule_search_locations)}")


def _tpu_sync_version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("tpu_sync_torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def maybe_start(env=None) -> bool:
    """Starts the pod's sidecar watcher if the environment asks for one and none has started; never raises."""
    env = os.environ if env is None else env
    if not (env.get(REGISTRY_PORT) or env.get(NODES)):
        return False
    try:
        return _start(env)
    except Exception as e:  # an import hook must never take the importing process down
        _say(f"Q38X2_SIDECAR not started: {type(e).__name__}: {e}", env)
        return False


def _start(env) -> bool:
    want = env.get(TPU_SYNC_VERSION, "")
    if want:
        have = _tpu_sync_version()
        if have != want:
            # Not yet: the leg installs the car's wheel before vllm serve; a later import retries.
            _say(f"Q38X2_SIDECAR waiting: tpu_sync_torch is {have}, not {want}", env)
            return False
    root = _tpu_sync_dir()
    if env.get(NODES):
        node_commands(env, os.path.join(root, *NODE_BIN))  # refuse a bad spec before taking the lock
    lock = os.path.join(env.get(LOCK_DIR, "/tmp"), "q38x2_sidecar.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    proc = subprocess.Popen(
        [sys.executable, "-I", os.path.abspath(__file__), "--watch", root],
        env=dict(env),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    _say(f"Q38X2_SIDECAR watcher pid {proc.pid} started by pid {os.getpid()} (tpu_sync {root})", env)
    return True


# -- the watcher (a detached process) --------------------------------------------------------------


def _say(line: str, env=None) -> None:
    env = os.environ if env is None else env
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = f"{line} {stamp}"
    print(text, file=sys.stderr, flush=True)
    out = env.get("CDK_OUTPUT_DIR")
    if out:
        try:
            os.makedirs(os.path.join(out, "q38x2"), exist_ok=True)
            with open(os.path.join(out, "q38x2", "sidecar.log"), "a") as f:
                f.write(text + "\n")
        except OSError:
            pass


def _install(src: str, name: str, env=None) -> str:
    """A 0755 copy (an extracted wheel may have lost the execute bit)."""
    import shutil

    env = os.environ if env is None else env
    d = env.get(BIN_DIR, "/tmp/q38x2_bin")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, name)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst + ".part")
        os.chmod(dst + ".part", 0o755)
        os.replace(dst + ".part", dst)
    return dst


def _answers(host, port: int) -> bool:
    """Whether ``host`` (or any of a tuple of hosts) accepts a connection on ``port``."""
    for h in (host,) if isinstance(host, str) else host:
        try:
            with socket.create_connection((h, port), timeout=1.0):
                return True
        except OSError:
            continue
    return False


def _rss_gib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def _mem_line(nodes) -> str:
    def meminfo(key):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith(key + ":"):
                        return int(line.split()[1]) / 2**20
        except OSError:
            pass
        return None

    def cgroup(name):
        for path in (f"/sys/fs/cgroup/{name}", f"/sys/fs/cgroup/memory/{name}"):
            try:
                raw = open(path).read().strip()
                return None if raw == "max" else int(raw) / 2**30
            except (OSError, ValueError):
                continue
        return None

    fmt = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
    rss = sum(_rss_gib(p.pid) for _, p, _ in nodes if p.poll() is None)
    current = cgroup("memory.current")
    if current is None:
        current = cgroup("memory.usage_in_bytes")
    limit = cgroup("memory.max")
    if limit is None:
        limit = cgroup("memory.limit_in_bytes")
    return (
        f"Q38X2_MEM nodes_rss_gib={rss:.1f} host_memavailable_gib={fmt(meminfo('MemAvailable'))} "
        f"host_memtotal_gib={fmt(meminfo('MemTotal'))} cgroup_current_gib={fmt(current)} "
        f"cgroup_max_gib={fmt(limit)}"
    )


def watch(root: str, env=None, poll_s: float = 1.0, stop_after_s: float | None = None) -> None:
    """Starts the registry and / or the nodes and reports on them until they all end (or ``stop_after_s``)."""
    env = os.environ if env is None else env
    t0 = time.monotonic()
    procs: list[tuple[str, subprocess.Popen, dict]] = []
    if env.get(REGISTRY_PORT):
        port = int(env[REGISTRY_PORT])
        reg = _install(os.path.join(root, *REGISTRY_BIN), "global_registry_server", env)
        p = subprocess.Popen([reg, f"--port={port}"], stdin=subprocess.DEVNULL, env=dict(env))
        # The registry listens on [::]; an IPv6-only pod (Exos) has no 127.0.0.1, so both loopbacks are tried.
        procs.append(("registry", p, {"host": ("127.0.0.1", "::1"), "port": port, "up": False, "t": time.monotonic()}))
        _say(f"Q38X2_REGISTRY_START pid={p.pid} port={port}", env)
    if env.get(NODES):
        node = _install(os.path.join(root, *NODE_BIN), "kv_cache_host_store_node_main", env)
        ip = _expand(NODE_IP, env.get(NODE_IP, ""), env)
        for shard, port, argv in node_commands(env, node):
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, env=dict(env))
            procs.append((shard, p, {"host": ip, "port": port, "up": False, "t": time.monotonic()}))
            _say(f"Q38X2_NODE_START shard={shard} pid={p.pid} " + " ".join(argv[1:]), env)
    nodes = [x for x in procs if x[0] != "registry"]
    next_mem = time.monotonic()
    while True:
        live = 0
        for name, p, st in procs:
            if st.get("done"):
                continue
            rc = p.poll()
            if rc is not None:
                st["done"] = True
                kind = "REGISTRY" if name == "registry" else "NODE"
                _say(
                    f"Q38X2_{kind}_EXIT {'' if name == 'registry' else f'shard={name} '}pid={p.pid} rc={rc} "
                    f"after_s={time.monotonic() - st['t']:.1f} was_up={st['up']}",
                    env,
                )
                continue
            live += 1
            if not st["up"] and _answers(st["host"], st["port"]):
                st["up"] = True
                if name == "registry":
                    _say(f"Q38X2_REGISTRY_UP pid={p.pid} port={st['port']} after_s={time.monotonic() - st['t']:.1f}", env)
                else:
                    _say(
                        f"Q38X2_NODE_UP shard={name} pid={p.pid} controller={st['host']}:{st['port']} "
                        f"boot_s={time.monotonic() - st['t']:.1f} rss_gib={_rss_gib(p.pid):.1f}",
                        env,
                    )
                    if all(s["up"] or s.get("done") for _, _, s in nodes):
                        _say(
                            f"Q38X2_NODES_ALL_UP up={sum(1 for _, _, s in nodes if s['up'])} of {len(nodes)} "
                            f"after_s={time.monotonic() - t0:.1f}",
                            env,
                        )
        if nodes and time.monotonic() >= next_mem:
            _say(_mem_line(nodes), env)
            next_mem = time.monotonic() + MEM_EVERY_S
        if live == 0 or (stop_after_s is not None and time.monotonic() - t0 > stop_after_s):
            return
        time.sleep(poll_s)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--watch":
        watch(sys.argv[2])
    else:
        print("usage: tier_x2_sidecar.py --watch <tpu_sync dir>", file=sys.stderr)
        sys.exit(2)
