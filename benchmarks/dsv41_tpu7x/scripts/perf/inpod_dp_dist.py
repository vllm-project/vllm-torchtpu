#!/usr/bin/env python3
"""In-pod: poll per-engine running/waiting request gauges from /metrics during a c=N run."""
import importlib.util
import re
import threading
import time
import urllib.request
import sys

conc = int(sys.argv[1]) if len(sys.argv) > 1 else 64
spec = importlib.util.spec_from_file_location("drv", "/tmp/run_dp8_scale_driver.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)


class _NoSend:
    def __init__(self, *x, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *x):
        return False

    def map(self, fn, it):
        return []


_real = drv.concurrent.futures.ThreadPoolExecutor
drv.concurrent.futures.ThreadPoolExecutor = _NoSend
prefixes = drv.warmup_decode_bench_templates()
drv.concurrent.futures.ThreadPoolExecutor = _real
done = []
th = threading.Thread(target=lambda: done.append(drv.run_scaled_benchmark(
    "/dev/shm/result_dpdist.json", "dpdist", conc, conc, 1024, prefixes=prefixes)))
th.start()
while th.is_alive():
    time.sleep(2)
    try:
        txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read().decode()
    except Exception as e:
        print("metrics err", e)
        continue
    run = {}
    wait = {}
    for line in txt.splitlines():
        m = re.match(r'vllm:num_requests_(running|waiting)\{(.*)\} ([0-9.e+]+)', line)
        if m:
            eng = re.search(r'engine="(\d+)"', m.group(2))
            e = eng.group(1) if eng else "?"
            (run if m.group(1) == "running" else wait)[e] = (run if m.group(1) == "running" else wait).get(e, 0) + float(m.group(3))
    print(time.strftime("%H:%M:%S"), "running", dict(sorted(run.items())), "waiting", dict(sorted(wait.items())), flush=True)
th.join()
