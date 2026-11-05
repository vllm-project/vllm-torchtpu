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
`RaidenStoreOffloadingWorker` do for multi-shape models, against a local
KVCacheStore controller, then round-trips one block (D2H block 3 -> host 0,
H2D host 0 -> block 5) and checks the bytes.

Each variant runs in its own subprocess, so a segfault or a TPU fatal error
in one variant is reported and the next one still runs. As in the server,
the byte views are taken first (`register_kv_caches` runs at KV cache init),
then a donated in-place write + `copy_` rebinds every cache (what the MSA op
does each forward), then Raiden registers the views. The default variant
list bisects the MiniMax M3 startup crash (j-aa6c393d-1ede-4358-8073):

  m3        60 layers x {kv (nb,32,4,256), index (nb,32,4,128)} fp8,
            host blocks 23301 (256 GiB / 4 ranks): the served per-block
            geometry and host pool. nb=4096 device blocks, not the served
            20011, so 120 arrays fit one chip; host blocks do not depend on it.
  small     same arrays, 1024 host blocks (host pool size)
  uint8     m3 with uint8 arrays (the DSv4 caches are uint8)
  typed     m3 registering the native fp8 tensors, not int8 byte views
  two       one layer (2 arrays) at the served host size (array count)
  nodonate  m3 written through a plain copy, no donation (cache rebinding)

Usage (one v7x host, e.g. simple-run-command RUN_CMD):
  python3 raiden_multi_shape_register_repro.py            # all variants
  python3 raiden_multi_shape_register_repro.py m3 small   # chosen ones
"""

import json
import os
import subprocess
import sys
import threading
import time

VARIANTS = {
    "m3": {},
    "small": {"host_blocks": 1024},
    "uint8": {"dtype": "uint8"},
    "typed": {"view": "typed"},
    "two": {"layers": 1},
    "nodonate": {"donate": False},
}
BASE = {
    "layers": 60,
    "num_blocks": 4096,
    "block_size": 128,
    "widths": [256, 128],
    "dtype": "fp8",
    "view": "int8",
    "host_blocks": 23301,
    "donate": True,
}


def _wait(future) -> None:
    for name in ("Await", "wait", "result", "get"):
        fn = getattr(future, name, None)
        if callable(fn):
            fn()
            return
    raise RuntimeError(f"cannot wait on {type(future).__name__}: {dir(future)}")


def _run(cfg: dict, port: int) -> None:
    import torch
    from tpu_sync.api.torch import kv_cache_manager as kcm

    from vllm_torchtpu.utils import synchronize_tensors

    dtype = {"fp8": torch.float8_e4m3fn, "uint8": torch.uint8}[cfg["dtype"]]
    device = torch.device("tpu")
    nb, bs = cfg["num_blocks"], cfg["block_size"]
    # Load the Raiden extension once, here: the worker thread and the store
    # below would otherwise race on its lazy loader and import it twice.
    impl = kcm._torch_impl()

    # Allocate as the runner does: one array per cache, synced one by one.
    caches = []
    for _ in range(cfg["layers"]):
        for width in cfg["widths"]:
            # PallasMLAttentionBackend.get_kv_cache_shape for 1-byte dtypes.
            cache = torch.zeros((nb, bs // 4, 4, width), dtype=dtype, device=device)
            synchronize_tensors([cache], wait=True)
            caches.append(cache)

    print(f"ALLOCATED {len(caches)} arrays", flush=True)

    # Byte views first, as register_kv_caches takes them at KV cache init.
    if cfg["view"] == "int8":
        tensors = []
        for cache in caches:
            storage = cache.untyped_storage()
            page = storage.nbytes() // nb
            tensors.append(
                torch.empty(0, dtype=torch.int8, device=device)
                .set_(storage)
                .view(nb, page)
            )
    else:
        tensors = caches

    # Then write block 3 of every cache, as a forward rebinds the caches.
    pattern = torch.arange(caches[0][3].numel(), dtype=torch.int32) % 251
    if cfg["donate"]:
        import jax.numpy as jnp
        from torch_tpu._internal import pallas

        def _fill(cache):
            row = jnp.arange(cache[3].size, dtype=jnp.int32) % 251
            row = row.astype(jnp.uint8).reshape(cache[3].shape)
            return cache.at[3].set(row.view(cache.dtype))

        op = pallas.jax_op("minimax_m3::raiden_repro_fill", _fill, donate_argnums=(0,))
        op.register_fake(lambda cache: torch.empty_like(cache))
        for cache in caches:
            cache.copy_(op(cache))
            synchronize_tensors([cache], wait=True)
    else:
        for cache in caches:
            row = pattern[: cache[3].numel()].to(torch.uint8).view(cache[3].shape)
            cache.view(torch.uint8)[3].copy_(row.to(device))
            synchronize_tensors([cache], wait=True)

    print("WRITTEN", flush=True)
    print(
        f"arrays={len(tensors)} shapes={sorted({tuple(t.shape) for t in tensors})} "
        f"dtype={tensors[0].dtype} host_blocks={cfg['host_blocks']}",
        flush=True,
    )

    address = f"127.0.0.1:{port}"
    holder: dict = {}

    def register() -> None:
        while "mgr" not in holder:
            try:
                holder["mgr"] = kcm.KVCacheManager(
                    kv_caches=[[t] for t in tensors],
                    local_control_port=0,
                    host_blocks_to_allocate=cfg["host_blocks"],
                    raiden_worker_port=0,
                    raiden_controller_address=address,
                    worker_id="shard_0",
                    node_id=0,
                )
            except Exception as e:  # controller not up yet
                holder["err"] = repr(e)
                time.sleep(0.5)

    thread = threading.Thread(target=register, daemon=True)
    thread.start()
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
    thread.join(timeout=60)
    mgr = holder.get("mgr")
    assert mgr is not None, f"worker never registered: {holder.get('err')}"
    print("REGISTERED", flush=True)

    _wait(mgr.d2h([3], [0]))
    _wait(mgr.h2d([0], [5]))

    def page_bytes(cache, block):
        return cache[block].cpu().view(torch.uint8).flatten().to(torch.int32)

    bad = []
    for i, cache in enumerate(caches):
        src, dst = page_bytes(cache, 3), page_bytes(cache, 5)
        if not torch.equal(src, dst) or not torch.equal(src, pattern[: src.numel()]):
            bad.append(i)
    print(f"ROUNDTRIP {'OK' if not bad else f'MISMATCH arrays {bad[:8]}'}", flush=True)
    del store


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        cfg = json.loads(sys.argv[2])
        _run(cfg, int(sys.argv[3]))
        return
    names = sys.argv[1:] or list(VARIANTS)
    results = {}
    for i, name in enumerate(names):
        cfg = {**BASE, **VARIANTS[name]}
        print(f"=== {name}: {cfg}", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--run", json.dumps(cfg), str(28700 + i)],
            capture_output=True,
            text=True,
            timeout=900,
            env=os.environ,
        )
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        print("\n".join(tail), flush=True)
        verdict = "OK" if "ROUNDTRIP OK" in proc.stdout else "FAIL"
        results[name] = f"{verdict} rc={proc.returncode}"
    print("=== SUMMARY", json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
