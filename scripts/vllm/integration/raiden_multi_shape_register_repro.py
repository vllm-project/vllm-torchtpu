#!/usr/bin/env python3
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
"""Model-free repro of the Raiden multi-shape KV registration on one chip.

Registers per-layer KV cache arrays with a Raiden KVCacheManager exactly as
`TPURaidenOffloadingConnector.register_kv_caches` +
`RaidenStoreOffloadingWorker` do for multi-shape models, then round-trips one
block (D2H block 3 -> host 0, H2D host 0 -> block 5) and checks the bytes.

Process layout follows the server: the KVCacheStore controller runs in its
own process (the scheduler; its constructor blocks until the workers
register), the worker process owns the chip, takes the byte views at KV cache
init (block 3 is filled on the host before the upload, so nothing rewrites
the device buffers), optionally rebinds the caches with an eager donated
write + `copy_` (`donate`), then probes the controller and registers, like
`RaidenStoreOffloadingWorker._register_loop`.

Each variant runs in its own subprocess, so a segfault or a TPU fatal error is
reported and the next variant still runs. Variants (default: all, in order):

  tiny      one layer, 8 host blocks (384 KiB pinned)
  m3rl      m3 after raising the soft RLIMIT_MEMLOCK to the hard limit
  glmrl     glm after the same raise
  glm       uint32 (nb, W/2, 128) arrays, GLM-DSA sparse-MLA style (prod)
  uint8     uint8 (nb, 32, 4, W) arrays, DSv4 style (prod)
  m3        fp8 (nb, 32, 4, W): the served MiniMax M3 geometry
  small     m3 with 1024 host blocks instead of 23301 (host pool size)
  typed     m3 registering the native fp8 tensors, not int8 byte views
  two       m3 with one layer (2 arrays) (array count)
  donate    m3, then an eager donated write + copy_ per cache after the views
            were taken (buffer rebinding under live byte views)

All: 60 layers x widths {256, 128} (page 32 KiB / 16 KiB per 128-token
block), 4096 device blocks (the served 20011 do not fit 120 arrays on one
chip; host blocks do not depend on it), 23301 host blocks = 256 GiB / 4 ranks.

Usage (one v7x host, e.g. simple-run-command RUN_CMD):
  python3 raiden_multi_shape_register_repro.py            # all variants
  python3 raiden_multi_shape_register_repro.py glm m3     # chosen ones
"""

import json
import os
import socket
import subprocess
import sys
import time

VARIANTS = {
    "tiny": {"layers": 1, "host_blocks": 8},
    "m3rl": {"raise_memlock": True},
    "glmrl": {"dtype": "uint32", "raise_memlock": True},
    "glm": {"dtype": "uint32"},
    "uint8": {"dtype": "uint8"},
    "m3": {},
    "small": {"host_blocks": 1024},
    "typed": {"view": "typed"},
    "two": {"layers": 1},
    "donate": {"donate": True},
}
BASE = {
    "layers": 60,
    "num_blocks": 4096,
    "block_size": 128,
    "widths": [256, 128],
    "dtype": "fp8",
    "view": "int8",
    "host_blocks": 23301,
    "donate": False,
    "raise_memlock": False,
}


def _wait(future) -> None:
    for name in ("Await", "wait", "result", "get"):
        fn = getattr(future, name, None)
        if callable(fn):
            fn()
            return
    raise RuntimeError(f"cannot wait on {type(future).__name__}: {dir(future)}")


def _store(cfg: dict, port: int) -> None:
    """The scheduler side: blocks until the worker registers, then stays up
    until the worker process closes our stdin."""
    from tpu_sync.api.torch import kv_cache_manager as kcm

    impl = kcm._torch_impl()
    print("STORE constructing", flush=True)
    store = impl.KVCacheStore(
        capacity=cfg["host_blocks"],
        raiden_id=impl.RaidenId("raiden-repro", "0", "kv_cache", 0),
        num_shards=1,
        shard_size_bytes=1,
        raiden_controller_port=port,
        global_registry_address="",
        store_server_ip="127.0.0.1",
        kv_pool_group="",
        expected_worker_count=1,
    )
    print("STORE UP", flush=True)
    sys.stdin.read()
    del store


def _reachable(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as sock:
            return sock.getsockname() != sock.getpeername()
    except OSError:
        return False


def _shape(cfg: dict, width: int) -> tuple[int, ...]:
    nb, bs = cfg["num_blocks"], cfg["block_size"]
    if cfg["dtype"] == "uint32":
        # Same page bytes as the 1-byte layout: bs * width.
        return (nb, bs * width // 512, 128)
    # PallasMLAttentionBackend.get_kv_cache_shape for 1-byte dtypes.
    return (nb, bs // 4, 4, width)


def _run(cfg: dict, port: int) -> None:
    import jax
    import jax.numpy as jnp
    import torch
    from torch_tpu._internal import pallas
    from tpu_sync.api.torch import kv_cache_manager as kcm

    from vllm_torchtpu.utils import synchronize_tensors

    dtype = {
        "fp8": torch.float8_e4m3fn,
        "uint8": torch.uint8,
        "uint32": torch.uint32,
    }[cfg["dtype"]]
    device = torch.device("tpu")
    nb = cfg["num_blocks"]

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    with open("/proc/self/status") as f:
        cap = [line.strip() for line in f if line.startswith("CapEff")]
    print(f"MEMLOCK soft={soft} hard={hard} {cap}", flush=True)
    if cfg["raise_memlock"]:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        print(f"MEMLOCK raised to {resource.getrlimit(resource.RLIMIT_MEMLOCK)}")

    # One device array per cache, block 3 filled on the host before upload.
    caches = []
    for _ in range(cfg["layers"]):
        for width in cfg["widths"]:
            page = cfg["block_size"] * width
            host = torch.zeros((nb, page), dtype=torch.uint8)
            host[3] = (torch.arange(page) % 7 + 1).to(torch.uint8)
            cache = host.view(dtype).view(_shape(cfg, width)).to(device)
            synchronize_tensors([cache], wait=True)
            caches.append(cache)
    print(f"ALLOCATED {len(caches)} x {caches[0].dtype}", flush=True)

    # Byte views first, as register_kv_caches takes them at KV cache init.
    if cfg["view"] == "int8":
        tensors = []
        for cache in caches:
            storage = cache.untyped_storage()
            tensors.append(
                torch.empty(0, dtype=torch.int8, device=device)
                .set_(storage)
                .view(nb, storage.nbytes() // nb)
            )
    else:
        tensors = caches

    if cfg["donate"]:
        # Rewrite block 3 (same values) through an eager donated op.
        def _fill(cache: jax.Array) -> jax.Array:
            row = (jnp.arange(cache[3].size, dtype=jnp.int32) % 7 + 1).astype(
                jnp.float32
            )
            return cache.at[3].set(row.reshape(cache[3].shape).astype(cache.dtype))

        op = pallas.jax_op("minimax_m3::raiden_repro_fill", _fill, donate_argnums=(0,))
        op.register_fake(lambda cache: torch.empty_like(cache))
        for cache in caches:
            cache.copy_(op(cache))
            synchronize_tensors([cache], wait=True)
    print(
        f"WRITTEN; registering {len(tensors)} arrays "
        f"{sorted({tuple(t.shape) for t in tensors})} {tensors[0].dtype}, "
        f"host_blocks={cfg['host_blocks']}",
        flush=True,
    )

    store = subprocess.Popen(
        [sys.executable, __file__, "--store", json.dumps(cfg), str(port)],
        stdin=subprocess.PIPE,
        env=os.environ,
    )
    try:
        deadline = time.monotonic() + 300
        mgr = None
        while mgr is None:
            assert store.poll() is None, f"store exited rc={store.returncode}"
            assert time.monotonic() < deadline, "controller never reachable"
            if not _reachable(port):
                time.sleep(0.5)
                continue
            try:
                mgr = kcm.KVCacheManager(
                    kv_caches=[[t] for t in tensors],
                    local_control_port=0,
                    host_blocks_to_allocate=cfg["host_blocks"],
                    raiden_worker_port=0,
                    raiden_controller_address=f"127.0.0.1:{port}",
                    worker_id="shard_0",
                    node_id=0,
                )
            except Exception as e:
                # No retry: a second KVCacheManager after a failed one
                # segfaults, which hides the first error.
                print(f"REGISTER FAILED: {e!r}", flush=True)
                return
        print("REGISTERED", flush=True)

        _wait(mgr.d2h([3], [0]))
        _wait(mgr.h2d([0], [5]))

        def page_bytes(cache, block):
            return cache[block].cpu().contiguous().view(torch.uint8).flatten()

        bad = []
        for i, cache in enumerate(caches):
            src, dst = page_bytes(cache, 3), page_bytes(cache, 5)
            if not torch.equal(src, dst) or not bool(src.any()):
                bad.append(i)
        verdict = "OK" if not bad else f"MISMATCH arrays {bad[:8]}"
        print(f"ROUNDTRIP {verdict}", flush=True)
    finally:
        store.stdin.close()
        try:
            store.wait(timeout=30)
        except subprocess.TimeoutExpired:
            store.kill()


def main() -> None:
    if len(sys.argv) > 3 and sys.argv[1] in ("--run", "--store"):
        import faulthandler

        faulthandler.enable()
        fn = _run if sys.argv[1] == "--run" else _store
        fn(json.loads(sys.argv[2]), int(sys.argv[3]))
        return
    names = sys.argv[1:] or list(VARIANTS)
    # As the connector sets before any Raiden call (raiden_connector.py).
    env = {**os.environ, "RAIDEN_DISABLE_SINGLETON_WORKER": "1"}
    results = {}
    for i, name in enumerate(names):
        cfg = {**BASE, **VARIANTS[name]}
        print(f"=== {name}: {cfg}", flush=True)
        # Stream the child's output; keep only its stdout to find the verdict.
        proc = subprocess.run(
            [sys.executable, __file__, "--run", json.dumps(cfg), str(28700 + i)],
            stdout=subprocess.PIPE,
            text=True,
            timeout=1200,
            env=env,
        )
        print(proc.stdout, flush=True)
        verdict = "OK" if "ROUNDTRIP OK" in proc.stdout else "FAIL"
        results[name] = f"{verdict} rc={proc.returncode}"
    print("=== SUMMARY", json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
