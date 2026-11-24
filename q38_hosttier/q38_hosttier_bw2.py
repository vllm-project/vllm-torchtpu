#!/usr/bin/env python3
"""Q-HOSTBW2: per-host aggregate Raiden host<->device bandwidth with 8 concurrent processes.

Runs inside the infx-qwen38-flash-bench pod (BENCH=script), from the vllm-torchtpu tree root:

    python3 q38_hosttier/q38_hosttier_bw2.py --out-dir "$CDK_OUTPUT_DIR/q38f-bench/script"

The served PP8 host tier moves pages on all 8 stages at once, each stage a process with its own
Raiden KVCacheManager on its own device. This bench runs one process per CHIP
(TPU_VISIBLE_CHIPS/TPU_VISIBLE_DEVICES/LOCAL_RANK = c, ALLOW_MULTIPLE_LIBTPU_LOAD=1, no WORLD_SIZE),
each moving --gib GiB of served 3,276,800-B pages (per-page segments, as the store issues them) on
its chip's first core, D2H then H2D.

Why per chip (j-828cdd6a-4aa9-4615-a056, M): a standalone process's TPU_VISIBLE_CHIPS indexes the
host's 4 chips; values 4-7 fail with "No TPU devices found after filtering". Per-core binding needs
torch_tpu's RANK/WORLD_SIZE slice bootstrap, i.e. an engine. A chip's two cores share its host link,
and one core alone moved 56 GB/s, so 4 per-chip processes measure the host's link aggregate; the
8-stage concurrency itself is timed inside the engine by the identity job.

Arms, run one after another in the same pod:
  solo        CONTROL: one process on device 0.
  concurrent  PRIMARY: one process per chip (4); a file barrier starts every process's transfer of a
              rep and direction together. Aggregate = sum of bytes / (last end - first start).
  serialized  MUTANT that must bite: 4 processes, but process d starts only after d-1 finished
              (token passing). Its aggregate must come out near the solo rate; if it does not,
              the concurrency accounting is broken and the run is void.

Every process verifies its own round trip once, outside timing: a ramp pattern written to the
device pool, D2H by Raiden, the pool overwritten, H2D by Raiden, sampled pages compared.
Each process also reports its host memory per NUMA node (/proc/self/numa_maps) and the
/dev/vfio groups it holds (device-binding evidence). GB = 1e9 bytes.

--selftest checks the aggregation and verdict logic on synthetic timings (no TPU).
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

PAGE_SHAPE = (3200, 1, 4, 256)
PAGE_BYTES = 3200 * 1 * 4 * 256
UNIT = int(os.environ.get("Q38_HOSTBW_UNIT", str(1 << 30)))
PHASES = ("d2h", "h2d")
MUTANT_MAX_RATIO = 1.3  # serialized aggregate / solo must stay below this


# ----------------------------------------------------------------------------------------------
# child
# ----------------------------------------------------------------------------------------------
def _barrier(path_glob, n, timeout):
    t_end = time.time() + timeout
    while len(glob.glob(path_glob)) < n:
        if time.time() > t_end:
            raise TimeoutError(f"barrier {path_glob}: {len(glob.glob(path_glob))}/{n}")
        time.sleep(0.0002)


def _touch(path):
    with open(path, "w") as f:
        f.write(str(time.time()))


def _numa_bytes():
    nodes = {}
    try:
        for line in open("/proc/self/numa_maps"):
            fields = line.split()
            kb = 4
            for f in fields:
                if f.startswith("kernelpagesize_kB="):
                    kb = int(f.split("=")[1])
            for f in fields:
                if f[:1] == "N" and "=" in f and f[1:].split("=")[0].isdigit():
                    node, pages = f[1:].split("=")
                    nodes[node] = nodes.get(node, 0) + int(pages) * kb * 1024
    except OSError as e:
        return {"error": str(e)}
    return nodes


def _vfio_fds():
    out = set()
    for fd in glob.glob("/proc/self/fd/*"):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("/dev/vfio") or "accel" in target:
            out.add(target)
    return sorted(out)


def _wait(fut):
    for name in ("wait", "result", "get"):
        fn = getattr(fut, name, None)
        if callable(fn):
            fn()
            return name
    return "none"


class _FakeManager:
    """CPU stand-in for plumbing tests (Q38_BW2_FAKE=1): sleeps at 56 GB/s, keeps host blocks."""

    def __init__(self, kv_caches, host_blocks_to_allocate, **_):
        import torch

        self.pool = kv_caches[0][0]
        self.host = torch.zeros((host_blocks_to_allocate, PAGE_BYTES), dtype=torch.uint8)

    def _go(self, src, dst, sizes, d2h):
        dev = self.pool.view(-1).view(__import__("torch").uint8).view(-1, PAGE_BYTES)
        for s_, t_, n_ in zip(src, dst, sizes):
            if d2h:
                self.host[t_ : t_ + n_] = dev[s_ : s_ + n_]
            else:
                dev[t_ : t_ + n_] = self.host[s_ : s_ + n_]
        nbytes = sum(sizes) * PAGE_BYTES
        return __import__("types").SimpleNamespace(wait=lambda: time.sleep(nbytes / 56e9))

    def d2h(self, src, dst, sizes):
        return self._go(src, dst, sizes, True)

    def h2d(self, src, dst, sizes):
        return self._go(src, dst, sizes, False)

    def read_block_bytes(self, layer, block):
        return self.host[block].numpy().tobytes()

    def write_block_bytes(self, layer, block, payload):
        import torch

        self.host[block] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)


def child(a):
    import torch

    if os.environ.get("Q38_BW2_FAKE"):
        KVCacheManager = _FakeManager
    else:
        import torch_tpu  # noqa: F401
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    os.environ.setdefault("RAIDEN_DISABLE_SINGLETON_WORKER", "1")
    d, n, sync = a.device, a.nproc, a.sync_dir
    info = {"device": d, "pid": os.getpid(), "env": {k: os.environ.get(k) for k in (
        "TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES", "LOCAL_RANK", "TORCH_TPU_TOPOLOGY",
        "RAIDEN_NUMA_POLICY")}}
    dev = torch.device("cpu" if os.environ.get("Q38_BW2_FAKE") else "tpu")
    pages = max(1, (a.gib * UNIT) // PAGE_BYTES)
    nbytes = pages * PAGE_BYTES
    try:
        info["device_count"] = int(torch.accelerator.device_count())
    except Exception as e:  # noqa: BLE001
        info["device_count"] = f"n/a ({type(e).__name__})"
    pool = torch.empty((pages,) + PAGE_SHAPE, dtype=torch.float8_e4m3fn, device=dev)
    # Ramp pattern (odd period) written to the device, so the round-trip check means something.
    period = 251 * 4099
    base = (torch.arange(period, dtype=torch.int64) % 251).to(torch.uint8)
    pat = base.repeat(PAGE_BYTES * pages // period + 1)[: PAGE_BYTES * pages].view(
        torch.float8_e4m3fn).view((pages,) + PAGE_SHAPE)
    pool.copy_(pat.to(dev))
    pool[:1].cpu()  # materialize before Raiden wraps it
    mgr = KVCacheManager(kv_caches=[[pool]], local_control_port=0, host_blocks_to_allocate=pages,
                         raiden_worker_port=0, raiden_controller_address=None,
                         worker_id=f"q38bw2_{a.arm}_{d}", node_id=d)
    ids = list(range(pages))
    info["vfio"] = _vfio_fds()
    _touch(f"{sync}/ready_{d}")
    _barrier(f"{sync}/ready_*", n, a.timeout)
    reps = []
    how = set()
    for r in range(a.reps + 1):
        rec = {}
        for p in PHASES:
            fn = mgr.d2h if p == "d2h" else mgr.h2d
            if a.mode == "serialized":
                # token passing: d waits for d-1 (process 0 starts once the previous phase ended)
                if d > 0:
                    _barrier(f"{sync}/done_{r}_{p}_{d - 1}", 1, a.timeout)
            else:
                _touch(f"{sync}/arrive_{r}_{p}_{d}")
                _barrier(f"{sync}/arrive_{r}_{p}_*", n, a.timeout)
            t0 = time.time()
            how.add(_wait(fn(ids, ids, [1] * pages)))
            t1 = time.time()
            _touch(f"{sync}/done_{r}_{p}_{d}")
            if a.mode == "serialized":
                _barrier(f"{sync}/done_{r}_{p}_*", n, a.timeout)
            rec[p] = (t0, t1)
        reps.append(rec)
    # Round-trip check (untimed). The DMA moves the device's physical (tiled) page bytes, so pages
    # are compared as byte multisets (as tests/distributed/test_raiden_pool_liveness_tpu.py does),
    # and the pool is never rebound by an eager op after Raiden pinned it (pool_manifest docs).
    # D2H: the host block holds the pattern page's bytes. H2D: a different host payload lands in
    # the buffer torch reads.
    _wait(mgr.d2h(ids, ids, [1] * pages))
    sample = sorted({0, pages // 3, pages // 2, pages - 1})
    hist = lambda t: torch.bincount(t.reshape(-1).to(torch.int64), minlength=256)
    d2h_ok = all(
        torch.equal(hist(torch.frombuffer(bytearray(mgr.read_block_bytes(0, b)), dtype=torch.uint8)),
                    hist(pat[b].view(torch.uint8)))
        for b in sample)
    payloads = {b: ((pat[b].view(torch.uint8).to(torch.int64) + 1) % 256).to(torch.uint8) for b in sample}
    for b, payload in payloads.items():
        mgr.write_block_bytes(0, b, payload.numpy().tobytes())
    _wait(mgr.h2d(sample, sample, [1] * len(sample)))
    h2d_ok = all(torch.equal(hist(pool[b].cpu().view(torch.uint8)), hist(payloads[b])) for b in sample)
    info["roundtrip_d2h_ok"], info["roundtrip_h2d_ok"] = bool(d2h_ok), bool(h2d_ok)
    info["roundtrip_ok"] = bool(d2h_ok and h2d_ok)
    info["numa_bytes"] = _numa_bytes()
    info["wait_methods"] = sorted(how)
    return {"info": info, "bytes": nbytes, "pages": pages, "reps": reps}


# ----------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------
def _child_env(d, topo):
    env = dict(os.environ)
    for k in ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "TORCH_TPU_SLICEBUILDER_ADDRESSES",
              "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    env.update({"TPU_VISIBLE_CHIPS": str(d), "TPU_VISIBLE_DEVICES": str(d), "LOCAL_RANK": str(d),
                "ALLOW_MULTIPLE_LIBTPU_LOAD": "1", "RAIDEN_DISABLE_SINGLETON_WORKER": "1"})
    if topo:
        env["TORCH_TPU_TOPOLOGY"] = topo
    return env


def run_arm(arm, mode, devices, gib, reps, timeout, out_dir, topo, log):
    sync = f"/dev/shm/q38_hosttier_bw2_{arm}"
    subprocess.run(["rm", "-rf", sync])
    os.makedirs(sync)
    for f in glob.glob("/tmp/libtpu_lockfile*"):
        try:
            os.remove(f)
        except OSError:
            pass
    procs = []
    for d in devices:
        argv = [sys.executable, os.path.abspath(__file__), "--child", "--arm", arm, "--mode", mode,
                "--device", str(d), "--nproc", str(len(devices)), "--sync-dir", sync,
                "--gib", str(gib), "--reps", str(reps), "--timeout", str(timeout)]
        procs.append(subprocess.Popen(argv, env=_child_env(d, topo), stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
    results, errors = [], []
    t_end = time.time() + timeout + 120
    for d, p in zip(devices, procs):
        try:
            so, se = p.communicate(timeout=max(1, t_end - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            so, se = p.communicate()
            se = (se or "") + "\nTIMEOUT"
        res = None
        for line in (so or "").splitlines():
            if line.startswith("Q38_BW2_CHILD "):
                res = json.loads(line[len("Q38_BW2_CHILD "):])
        log.write(f"--- arm {arm} device {d} rc={p.returncode}\n{(se or '')[-5000:]}\n")
        log.flush()
        if res is None:
            errors.append({"device": d, "rc": p.returncode, "stderr_tail": (se or "")[-1500:]})
        else:
            results.append(res)
    return {"arm": arm, "mode": mode, "devices": devices, "gib": gib,
            "children": results, "errors": errors, "summary": aggregate(results, reps)}


def aggregate(children, reps):
    """Per rep and direction: aggregate = sum(bytes) / (last end - first start); median over reps
    after the warm-up rep 0. Per-process rate = own bytes / own duration."""
    out = {"ok_children": len(children)}
    if not children:
        return out
    for p in PHASES:
        agg, per_proc = [], []
        for r in range(1, reps + 1):
            spans = [c["reps"][r][p] for c in children]
            total = sum(c["bytes"] for c in children)
            agg.append(total / (max(t1 for _, t1 in spans) - min(t0 for t0, _ in spans)) / 1e9)
            per_proc += [c["bytes"] / (t1 - t0) / 1e9 for c, (t0, t1) in
                         zip(children, spans)]
        out[f"{p}_aggregate_gbps_median"] = round(statistics.median(agg), 3)
        out[f"{p}_aggregate_gbps_min"] = round(min(agg), 3)
        out[f"{p}_per_process_gbps_median"] = round(statistics.median(per_proc), 3)
        out[f"{p}_per_process_gbps_min"] = round(min(per_proc), 3)
    out["roundtrip_all_ok"] = all(c["info"].get("roundtrip_ok") for c in children)
    return out


def verdict(arms, nproc):
    """Validity, the mutant, and the headline number."""
    solo, conc, ser = (arms.get(k, {}).get("summary", {}) for k in ("solo", "concurrent", "serialized"))
    v = {}
    v["valid_solo"] = solo.get("ok_children") == 1 and bool(solo.get("roundtrip_all_ok"))
    v["valid_concurrent"] = conc.get("ok_children") == nproc and bool(conc.get("roundtrip_all_ok"))
    v["valid_serialized"] = ser.get("ok_children") == nproc and bool(ser.get("roundtrip_all_ok"))
    bites = None
    if v["valid_solo"] and v["valid_serialized"]:
        ratios = {p: ser[f"{p}_aggregate_gbps_median"] / solo[f"{p}_aggregate_gbps_median"]
                  for p in PHASES}
        v["serialized_over_solo"] = {p: round(x, 3) for p, x in ratios.items()}
        bites = all(x < MUTANT_MAX_RATIO for x in ratios.values())
    v["mutant_bites"] = bites
    v["valid"] = bool(v["valid_solo"] and v["valid_concurrent"] and v["valid_serialized"] and bites)
    if v["valid_concurrent"]:
        v["primary_aggregate_gbps"] = min(conc[f"{p}_aggregate_gbps_median"] for p in PHASES)
        if v["valid_solo"]:
            v["concurrency_speedup"] = {
                p: round(conc[f"{p}_aggregate_gbps_median"] / solo[f"{p}_aggregate_gbps_median"], 3)
                for p in PHASES}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"),
                                                      "q38f-bench", "script"))
    ap.add_argument("--gib", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--arms", default="solo,concurrent,serialized")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--arm", default="")
    ap.add_argument("--mode", default="concurrent")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--sync-dir", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.child:
        print("Q38_BW2_CHILD " + json.dumps(child(a)), flush=True)
        return 0
    if a.selftest:
        return selftest()
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "q38_hosttier_bw2_children.log"), "a")
    try:
        from torch_tpu._internal.utils import hardware

        topo = hardware.get_tpu_topology(1)
    except Exception as e:  # noqa: BLE001
        topo = None
        log.write(f"topology(1) lookup failed: {e!r}\n")
    record = {"t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gib": a.gib,
              "reps": a.reps, "nproc": a.nproc, "page_bytes": PAGE_BYTES, "topology_1": topo,
              "arms": {}}
    print(f"Q38_BW2 topology(1)={topo}", flush=True)
    plan = {"solo": ("concurrent", [0]),
            "concurrent": ("concurrent", list(range(a.nproc))),
            "serialized": ("serialized", list(range(a.nproc)))}
    for arm in a.arms.split(","):
        mode, devices = plan[arm]
        res = run_arm(arm, mode, devices, a.gib, a.reps, a.timeout, a.out_dir, topo, log)
        record["arms"][arm] = res
        print(f"Q38_BW2 arm {arm}: {json.dumps(res['summary'])} errors={len(res['errors'])}", flush=True)
    record["t_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["verdict"] = verdict(record["arms"], a.nproc)
    with open(os.path.join(a.out_dir, "q38_hosttier_bw2.json"), "w") as f:
        json.dump(record, f, indent=1)
    print("Q38_BW2_VERDICT " + json.dumps(record["verdict"]), flush=True)
    return 0 if record["verdict"]["valid"] else 3


def selftest():
    def kid(nbytes, spans):
        return {"bytes": nbytes, "info": {"roundtrip_ok": True},
                "reps": [{p: s for p in PHASES} for s in spans]}

    G = 4 * (1 << 30)
    fails = []
    # the warm-up rep (rep 0, here absurdly fast) never enters the median
    w = aggregate([kid(G, [(0, 1e-6), (10, 10 + G / 56e9)])], 1)
    if abs(w["d2h_aggregate_gbps_median"] - 56.0) > 0.01:
        fails.append(("warmup", w))
    # solo: 4 GiB in 0.0767 s = 56 GB/s (rep 0 is warm-up and ignored)
    solo = [kid(G, [(0, 9.0)] + [(10 * r, 10 * r + G / 56e9) for r in range(1, 4)])]
    s = aggregate(solo, 3)
    if abs(s["d2h_aggregate_gbps_median"] - 56.0) > 0.01:
        fails.append(("solo", s))
    # concurrent: 8 overlapped, each 4x slower (14 GB/s each) -> aggregate 112
    conc = [kid(G, [(0, 9.0)] + [(10 * r, 10 * r + G / 14e9) for r in range(1, 4)]) for _ in range(8)]
    c = aggregate(conc, 3)
    if abs(c["h2d_aggregate_gbps_median"] - 112.0) > 0.05 or abs(c["h2d_per_process_gbps_median"] - 14.0) > 0.01:
        fails.append(("conc", c))
    # serialized: back to back at 56 each -> aggregate 56
    ser = [kid(G, [(0, 9.0)] + [(10 * r + i * G / 56e9, 10 * r + (i + 1) * G / 56e9)
                                for r in range(1, 4)]) for i in range(8)]
    z = aggregate(ser, 3)
    if abs(z["d2h_aggregate_gbps_median"] - 56.0) > 0.05:
        fails.append(("ser", z))
    arms = {"solo": {"summary": s}, "concurrent": {"summary": c}, "serialized": {"summary": z}}
    v = verdict(arms, 8)
    if not (v["valid"] and v["mutant_bites"] and abs(v["primary_aggregate_gbps"] - 112.0) < 0.05):
        fails.append(("verdict", v))
    # a harness that failed to serialize (serialized arm overlapped) must void the run
    arms_bad = {"solo": {"summary": s}, "concurrent": {"summary": c}, "serialized": {"summary": c}}
    if verdict(arms_bad, 8)["valid"]:
        fails.append("overlapped serialized arm must void the run")
    # a missing child must void the concurrent arm
    if verdict({"solo": {"summary": s}, "concurrent": {"summary": aggregate(conc[:7], 3)},
                "serialized": {"summary": z}}, 8)["valid"]:
        fails.append("7/8 children must void the run")
    # a failed round trip must void the run
    bad = [dict(k, info={"roundtrip_ok": False}) for k in conc]
    if verdict({"solo": {"summary": s}, "concurrent": {"summary": aggregate(bad, 3)},
                "serialized": {"summary": z}}, 8)["valid"]:
        fails.append("failed round trip must void the run")
    print("selftest", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
