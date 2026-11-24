#!/usr/bin/env python3
"""Q-HOSTBW: host<->device bandwidth of 1-4 GiB buffers on one tpu7x host (HOSTTIER CP1).

Runs inside the infx-qwen38-flash-bench pod (BENCH=script), from the vllm-torchtpu tree root:

    python3 q38_hosttier/q38_hosttier_bw.py --out-dir "$CDK_OUTPUT_DIR/q38f-bench/script"

Every measurement runs in its own child process with a timeout, one after another, so a hang or a
crash in one path cannot take the others with it, and only one process owns the TPU at a time.

Paths measured (GB = 1e9 bytes; buffers are GiB = 2^30 bytes):
  host    host facts: MemTotal/MemAvailable, cgroup memory limit/usage, NUMA nodes, TPU PCIe links
  jax     PRIMARY. JAX memory kinds: pinned_host <-> device on one device (1, 2, 4 GiB), the same
          from/to an unpinned numpy buffer, a same-chip and a cross-chip pair of devices at once,
          and all devices of the host at once (aggregate). Every transfer is verified by sampled
          bytes after the timed reps (a no-op transfer fails the check).
  torch   torch_tpu batch_transfer_{d2h,h2d}_sync (the TPUConnector host path) on one device.
  raiden  tpu_sync KVCacheManager d2h/h2d of whole served pool pages (3,276,800 B: 3,200 tokens x
          1,024 B fp8, the PP8 TP1 page) -- the DMA engine the Raiden offload store drives.
          Per-page segments (as the store issues them) and one contiguous segment.

Each timed number is the median of --reps reps after one untimed warm-up rep.
Exit 0 when every jax single-device size produced a verified number; 3 otherwise.

--selftest runs the summary/verdict logic on synthetic child outputs (no TPU).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

PAGE_SHAPE = (3200, 1, 4, 256)  # served PP8 TP1 main-KV page (fp8): 3,276,800 bytes
PAGE_BYTES = 3200 * 1 * 4 * 256
# Buffer unit: 2^30 bytes. Q38_HOSTBW_UNIT overrides it only for CPU validation of this script.
UNIT = int(os.environ.get("Q38_HOSTBW_UNIT", str(1 << 30)))
MODEL_GBPS = 25.0  # config_roofline_v2 / plan_v1 assumption
STOP_GBPS = 10.0  # plan_v1 3.4a stop rule
PATTERN_PERIOD = 251 * 4099  # odd, so power-of-two sample strides never alias it


def _gbps(nbytes, seconds):
    return nbytes / seconds / 1e9 if seconds > 0 else float("inf")


def _summ(nbytes, times):
    """Median/min/max GB/s over the timed reps (times[0] is the warm-up)."""
    timed = times[1:] if len(times) > 1 else times
    rates = [_gbps(nbytes, t) for t in timed]
    return {
        "bytes": nbytes,
        "reps": len(timed),
        "gbps_median": round(statistics.median(rates), 3),
        "gbps_min": round(min(rates), 3),
        "gbps_max": round(max(rates), 3),
        "s_median": round(statistics.median(timed), 6),
        "warmup_s": round(times[0], 6),
    }


# ----------------------------------------------------------------------------------------------
# host facts (no TPU)
# ----------------------------------------------------------------------------------------------
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def host_facts():
    facts = {}
    mem = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, val = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "MemFree", "HugePages_Total", "Hugepagesize",
                   "Mlocked", "Shmem", "Cached"):
            mem[key] = val.strip()
    facts["meminfo"] = mem
    facts["cgroup"] = {
        "v2_memory.max": _read("/sys/fs/cgroup/memory.max"),
        "v2_memory.current": _read("/sys/fs/cgroup/memory.current"),
        "v2_memory.peak": _read("/sys/fs/cgroup/memory.peak"),
        "v1_limit": _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "v1_usage": _read("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    }
    nodes = {}
    base = "/sys/devices/system/node"
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if name.startswith("node") and name[4:].isdigit():
                total = None
                for line in (_read(f"{base}/{name}/meminfo") or "").splitlines():
                    if "MemTotal" in line:
                        total = line.split(":")[1].strip()
                nodes[name] = {"cpulist": _read(f"{base}/{name}/cpulist"), "MemTotal": total}
    facts["numa"] = nodes
    facts["nproc"] = os.cpu_count()
    pci = []
    root = "/sys/bus/pci/devices"
    if os.path.isdir(root):
        for dev in sorted(os.listdir(root)):
            vendor = _read(f"{root}/{dev}/vendor")
            if vendor != "0x1ae0":  # Google
                continue
            pci.append({
                "bdf": dev,
                "device": _read(f"{root}/{dev}/device"),
                "class": _read(f"{root}/{dev}/class"),
                "numa_node": _read(f"{root}/{dev}/numa_node"),
                "current_link_speed": _read(f"{root}/{dev}/current_link_speed"),
                "current_link_width": _read(f"{root}/{dev}/current_link_width"),
                "max_link_speed": _read(f"{root}/{dev}/max_link_speed"),
                "max_link_width": _read(f"{root}/{dev}/max_link_width"),
            })
    facts["tpu_pci"] = pci
    facts["env"] = {k: v for k, v in os.environ.items()
                    if k.startswith(("TPU_PREMAPPED", "TPU_HMA", "RAIDEN_", "TPU_RAIDEN", "LIBTPU"))}
    return facts


# ----------------------------------------------------------------------------------------------
# children (each owns the TPU alone)
# ----------------------------------------------------------------------------------------------
def child_jax(sizes, reps, agg_sizes):
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.sharding import SingleDeviceSharding as SDS

    devs = jax.devices()
    out = {
        "jax": jax.__version__,
        "devices": [
            {"id": d.id, "kind": d.device_kind, "coords": list(getattr(d, "coords", []) or []),
             "core_on_chip": getattr(d, "core_on_chip", None)}
            for d in devs
        ],
        "single": {}, "pairs": {}, "aggregate": {}, "checks": {},
    }

    def timed(fn):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(r)
        return time.perf_counter() - t0, r

    base = (np.arange(PATTERN_PERIOD, dtype=np.int64) % 251).astype(np.uint8)

    def pattern(shape):
        # A uint8 ramp with an odd period, so a transfer that moved nothing (zeros) or the wrong
        # slice fails the sampled check. np.resize cycles `base`: n bytes, no wide temporaries.
        return np.resize(base, int(np.prod(shape))).reshape(shape)

    def check(name, host, arr):
        got = np.asarray(arr)
        step = max(1, got.size // 4096)
        ok = bool(np.array_equal(got.reshape(-1)[::step], host.reshape(-1)[::step]))
        out["checks"][name] = ok
        return ok

    d0 = devs[0]
    for gib in sizes:
        n = gib * UNIT
        host = pattern((n,))
        pinned = jax.device_put(host, SDS(d0, memory_kind="pinned_host"))
        jax.block_until_ready(pinned)
        t = {"h2d_pinned": [], "d2h_pinned": [], "h2d_numpy": [], "d2h_numpy": []}
        dev = back = dev2 = None
        for _ in range(reps + 1):
            dev = back = dev2 = None
            dt, dev = timed(lambda: jax.device_put(pinned, SDS(d0, memory_kind="device")))
            t["h2d_pinned"].append(dt)
            dt, back = timed(lambda: jax.device_put(dev, SDS(d0, memory_kind="pinned_host")))
            t["d2h_pinned"].append(dt)
            dt, dev2 = timed(lambda: jax.device_put(host, d0))
            t["h2d_numpy"].append(dt)
            t0 = time.perf_counter()
            arr = np.asarray(dev2)
            t["d2h_numpy"].append(time.perf_counter() - t0)
            del arr
        check(f"single_{gib}GiB_pinned_roundtrip", host, back)
        check(f"single_{gib}GiB_numpy_h2d", host, dev2)
        out["single"][f"{gib}GiB"] = {k: _summ(n, v) for k, v in t.items()}
        del pinned, dev, back, dev2, host

    # Same-chip and cross-chip device pairs at once (does a chip's two cores share its link?).
    by_chip = {}
    for d in devs:
        by_chip.setdefault(tuple(getattr(d, "coords", []) or [d.id]), []).append(d)
    chips = list(by_chip.values())
    pairs = {}
    if chips and len(chips[0]) >= 2:
        pairs["same_chip"] = chips[0][:2]
    if len(chips) >= 2:
        pairs["cross_chip"] = [chips[0][0], chips[1][0]]
    gib = max(sizes)
    for name, pdevs in pairs.items():
        n = gib * UNIT
        mesh = Mesh(np.array(pdevs), ("x",))
        host = pattern((len(pdevs), n))
        sh_p = NamedSharding(mesh, P("x"), memory_kind="pinned_host")
        sh_d = NamedSharding(mesh, P("x"), memory_kind="device")
        pinned = jax.device_put(host, sh_p)
        jax.block_until_ready(pinned)
        t = {"h2d_pinned": [], "d2h_pinned": []}
        for _ in range(reps + 1):
            dt, dev = timed(lambda: jax.device_put(pinned, sh_d))
            t["h2d_pinned"].append(dt)
            dt, back = timed(lambda: jax.device_put(dev, sh_p))
            t["d2h_pinned"].append(dt)
        check(f"pair_{name}_roundtrip", host, back)
        out["pairs"][name] = {"devices": [d.id for d in pdevs], "per_device_GiB": gib,
                              **{k: _summ(len(pdevs) * n, v) for k, v in t.items()}}
        del pinned, dev, back, host

    # Every device of the host at once (the served PP8 leg moves pages on all 8 stages together).
    mesh = Mesh(np.array(devs), ("x",))
    for gib in agg_sizes:
        n = gib * UNIT
        host = pattern((len(devs), n))
        sh_p = NamedSharding(mesh, P("x"), memory_kind="pinned_host")
        sh_d = NamedSharding(mesh, P("x"), memory_kind="device")
        pinned = jax.device_put(host, sh_p)
        jax.block_until_ready(pinned)
        t = {"h2d_pinned": [], "d2h_pinned": []}
        for _ in range(reps + 1):
            dt, dev = timed(lambda: jax.device_put(pinned, sh_d))
            t["h2d_pinned"].append(dt)
            dt, back = timed(lambda: jax.device_put(dev, sh_p))
            t["d2h_pinned"].append(dt)
        check(f"aggregate_{gib}GiB_roundtrip", host, back)
        out["aggregate"][f"{gib}GiB_per_device"] = {
            "devices": len(devs), **{k: _summ(len(devs) * n, v) for k, v in t.items()}}
        del pinned, dev, back, host
    return out


def _tpu_device():
    import torch
    import torch_tpu  # noqa: F401

    # The runner and the TPU tests name the device "tpu" (tpu_worker.py, test_raiden_pool_liveness_tpu.py).
    return torch.device("tpu")


def child_torch(sizes, reps):
    import torch
    from torch_tpu._internal.batch_transfer import (
        batch_transfer_d2h_sync,
        batch_transfer_h2d_sync,
    )

    dev = _tpu_device()
    out = {"torch": torch.__version__, "single": {}, "checks": {}, "pinned": None}
    for gib in sizes:
        n = gib * UNIT
        try:
            cpu = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            out["pinned"] = True
        except Exception as e:  # noqa: BLE001
            cpu = torch.empty(n, dtype=torch.uint8)
            out["pinned"] = f"unpinned ({type(e).__name__}: {e})"[:300]
        base = (torch.arange(PATTERN_PERIOD, dtype=torch.int64) % 251).to(torch.uint8)
        cpu.copy_(base.repeat(n // PATTERN_PERIOD + 1)[:n])
        src = cpu.clone()
        devt = torch.empty(n, dtype=torch.uint8, device=dev)
        t = {"h2d": [], "d2h": []}
        for _ in range(reps + 1):
            t0 = time.perf_counter()
            batch_transfer_h2d_sync([cpu], [devt])
            t["h2d"].append(time.perf_counter() - t0)
            cpu.zero_()
            t0 = time.perf_counter()
            batch_transfer_d2h_sync([devt], [cpu])
            t["d2h"].append(time.perf_counter() - t0)
        step = max(1, n // 4096)
        out["checks"][f"{gib}GiB_roundtrip"] = bool(torch.equal(cpu[::step], src[::step]))
        out["single"][f"{gib}GiB"] = {k: _summ(n, v) for k, v in t.items()}
        del cpu, src, devt
    return out


def _wait(fut):
    for name in ("wait", "result", "get"):
        fn = getattr(fut, name, None)
        if callable(fn):
            return name, fn()
    if hasattr(fut, "done"):
        while not fut.done():
            time.sleep(0.0005)
        return "done-poll", None
    return "none", fut


def child_raiden(gib, reps):
    import torch
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    os.environ.setdefault("RAIDEN_DISABLE_SINGLETON_WORKER", "1")
    dev = _tpu_device()
    pages = max(1, (gib * UNIT) // PAGE_BYTES)
    nbytes = pages * PAGE_BYTES
    pool = torch.empty((pages,) + PAGE_SHAPE, dtype=torch.float8_e4m3fn, device=dev)
    # torch.empty leaves the device buffer unmaterialized, and a Raiden manager built over such a
    # tensor hangs in its constructor; a one-page read materializes it (pool_manifest.materialize_storages).
    pool[:1].cpu()
    mgr = KVCacheManager(
        kv_caches=[[pool]],
        local_control_port=0,
        host_blocks_to_allocate=pages,
        raiden_worker_port=0,
        raiden_controller_address=None,
        worker_id=f"q38bw{gib}",
        node_id=0,
    )
    ids = list(range(pages))
    t = {"d2h_per_page": [], "h2d_per_page": [], "d2h_contiguous": [], "h2d_contiguous": []}
    waits = set()
    for _ in range(reps + 1):
        for key, fn, args in (
            ("d2h_per_page", mgr.d2h, (ids, ids, [1] * pages)),
            ("h2d_per_page", mgr.h2d, (ids, ids, [1] * pages)),
            ("d2h_contiguous", mgr.d2h, ([0], [0], [pages])),
            ("h2d_contiguous", mgr.h2d, ([0], [0], [pages])),
        ):
            t0 = time.perf_counter()
            how, _ = _wait(fn(*args))
            t[key].append(time.perf_counter() - t0)
            waits.add(how)
    return {"gib": gib, "pages": pages, "page_bytes": PAGE_BYTES, "wait_methods": sorted(waits),
            "single": {k: _summ(nbytes, v) for k, v in t.items()}}


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def run_child(argv, timeout, log):
    for f in ("/tmp/libtpu_lockfile",):
        try:
            os.remove(f)
        except OSError:
            pass
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__)] + argv,
                           capture_output=True, text=True, timeout=timeout)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, so, se = "timeout", out, err + f"\nTIMEOUT after {timeout}s"
    wall = time.time() - t0
    result = None
    for line in (so or "").splitlines():
        if line.startswith("Q38_HOSTBW_CHILD "):
            result = json.loads(line[len("Q38_HOSTBW_CHILD "):])
    log.write(f"--- child {argv} rc={rc} wall={wall:.1f}s\n{(se or '')[-6000:]}\n")
    log.flush()
    return {"argv": argv, "rc": rc, "wall_s": round(wall, 1), "result": result,
            "stderr_tail": (se or "")[-1500:] if result is None else ""}


def summarize(record):
    """Headline numbers and the verdict against the model's 25 GB/s and the 10 GB/s stop rule."""
    head = {"model_gbps": MODEL_GBPS, "stop_gbps": STOP_GBPS}
    jx = (record.get("jax") or {}).get("result") or {}
    single = jx.get("single", {})
    checks = jx.get("checks", {})
    sizes_ok = bool(single) and all(checks.get(f"single_{k}_pinned_roundtrip") for k in single)
    head["jax_checks_all_pass"] = bool(checks) and all(checks.values())
    rows = {}
    for size, r in single.items():
        rows[size] = {k: r[k]["gbps_median"] for k in r}
    head["jax_single_gbps_median"] = rows
    agg = {size: {k: r[k]["gbps_median"] for k in ("h2d_pinned", "d2h_pinned")}
           for size, r in jx.get("aggregate", {}).items()}
    head["jax_aggregate_gbps_median"] = agg
    head["jax_pairs_gbps_median"] = {
        name: {k: r[k]["gbps_median"] for k in ("h2d_pinned", "d2h_pinned")}
        for name, r in jx.get("pairs", {}).items()}
    ra = {}
    for rec in record.get("raiden", []):
        res = rec.get("result")
        if res:
            ra[f"{res['gib']}GiB"] = {k: v["gbps_median"] for k, v in res["single"].items()}
    head["raiden_single_gbps_median"] = ra
    to = ((record.get("torch") or {}).get("result") or {}).get("single", {})
    head["torch_single_gbps_median"] = {s: {k: v["gbps_median"] for k, v in r.items()} for s, r in to.items()}
    if rows:
        worst = min(min(v["h2d_pinned"], v["d2h_pinned"]) for v in rows.values())
        head["jax_single_pinned_worst_gbps"] = worst
        head["verdict"] = ("STOP (< %.0f GB/s)" % STOP_GBPS if worst < STOP_GBPS
                           else "at/above model" if worst >= MODEL_GBPS else "below model, above stop")
    else:
        head["verdict"] = "NO DATA"
    head["exit_ok"] = sizes_ok
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"),
                                                      "q38f-bench", "script"))
    ap.add_argument("--sizes-gib", default="1,2,4")
    ap.add_argument("--agg-sizes-gib", default="1,4")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--phases", default="host,jax,raiden,torch")
    ap.add_argument("--child", default=None)
    ap.add_argument("--gib", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes_gib.split(",") if x]
    agg = [int(x) for x in a.agg_sizes_gib.split(",") if x]

    if a.child:
        fn = {"jax": lambda: child_jax(sizes, a.reps, agg),
              "torch": lambda: child_torch(sizes, a.reps),
              "raiden": lambda: child_raiden(a.gib, a.reps)}[a.child]
        res = fn()
        print("Q38_HOSTBW_CHILD " + json.dumps(res), flush=True)
        return 0

    if a.selftest:
        return selftest()

    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "q38_hosttier_bw_children.log"), "a")
    phases = a.phases.split(",")
    record = {"t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sizes_gib": sizes,
              "agg_sizes_gib": agg, "reps": a.reps, "page_bytes": PAGE_BYTES}
    if "host" in phases:
        record["host"] = host_facts()
        print("Q38_HOSTBW host " + json.dumps(record["host"]), flush=True)
    common = ["--sizes-gib", a.sizes_gib, "--agg-sizes-gib", a.agg_sizes_gib, "--reps", str(a.reps)]
    if "jax" in phases:
        record["jax"] = run_child(["--child", "jax"] + common, a.timeout, log)
        print("Q38_HOSTBW jax rc=%s wall=%ss" % (record["jax"]["rc"], record["jax"]["wall_s"]), flush=True)
    if "raiden" in phases:
        record["raiden"] = [run_child(["--child", "raiden", "--gib", str(g)] + common, a.timeout, log)
                            for g in sizes]
        print("Q38_HOSTBW raiden rc=%s" % [r["rc"] for r in record["raiden"]], flush=True)
    if "torch" in phases:
        record["torch"] = run_child(["--child", "torch"] + common, a.timeout, log)
        print("Q38_HOSTBW torch rc=%s" % record["torch"]["rc"], flush=True)
    record["t_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["summary"] = summarize(record)
    with open(os.path.join(a.out_dir, "q38_hosttier_bw.json"), "w") as f:
        json.dump(record, f, indent=1)
    print("Q38_HOSTBW_SUMMARY " + json.dumps(record["summary"]), flush=True)
    return 0 if record["summary"]["exit_ok"] else 3


def selftest():
    """Summary/verdict logic on synthetic child outputs; returns 0 when every case behaves."""
    def fake_single(h2d, d2h, ok=True):
        n = 1 << 30
        s = {"h2d_pinned": _summ(n, [1.0] + [n / h2d / 1e9] * 3),
             "d2h_pinned": _summ(n, [1.0] + [n / d2h / 1e9] * 3),
             "h2d_numpy": _summ(n, [1.0, 1.0]), "d2h_numpy": _summ(n, [1.0, 1.0])}
        return {"single": {"1GiB": s}, "checks": {"single_1GiB_pinned_roundtrip": ok}}

    fails = []
    # rate arithmetic: 1 GiB in 0.1 s = 10.737 GB/s, in 0.2 s = 5.369; the 9.9 s warm-up is excluded
    r = _summ(1 << 30, [9.9, 0.1, 0.2])
    if abs(r["gbps_median"] - 8.053) > 1e-3 or r["reps"] != 2 or abs(r["warmup_s"] - 9.9) > 1e-9:
        fails.append(("rate", r))
    s = summarize({"jax": {"result": fake_single(40.0, 30.0)}})
    if s["verdict"] != "at/above model" or not s["exit_ok"]:
        fails.append(("above", s["verdict"], s["exit_ok"]))
    s = summarize({"jax": {"result": fake_single(40.0, 15.0)}})
    if s["verdict"] != "below model, above stop":
        fails.append(("below", s["verdict"]))
    s = summarize({"jax": {"result": fake_single(9.0, 30.0)}})
    if not s["verdict"].startswith("STOP"):
        fails.append(("stop", s["verdict"]))
    s = summarize({"jax": {"result": fake_single(40.0, 30.0, ok=False)}})
    if s["exit_ok"]:
        fails.append("failed check must fail the job")
    s = summarize({"jax": {"result": None}})
    if s["verdict"] != "NO DATA" or s["exit_ok"]:
        fails.append(("nodata", s))
    print("selftest", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
