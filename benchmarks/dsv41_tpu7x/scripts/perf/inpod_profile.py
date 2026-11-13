#!/usr/bin/env python3
"""In-pod: run a c=N decode load against the live DP=8 server and capture a torch/TPU profile.

Usage (inside a --keep-pod decode pod whose server was booted with
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/dev/shm/profiles",...}'):
  python3 /tmp/inpod_profile.py --conc 64 --out-len 2048 --start-after 10 --dur 1.5 --tag p1
Uploads the per-rank xplane files to gs://$DSV41_STAGING/profiles/<tag>/ when GCS_TOKEN is set.
"""
import argparse
import glob
import importlib.util
import json
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--conc", type=int, default=64)
ap.add_argument("--out-len", type=int, default=2048)
ap.add_argument("--start-after", type=float, default=10.0)
ap.add_argument("--dur", type=float, default=1.5)
ap.add_argument("--tag", default="prof")
ap.add_argument("--no-profile", action="store_true")
ap.add_argument("--no-seed", action="store_true", help="skip warmup_decode_bench_templates seeding")
ap.add_argument("--bucket", default="chaitanyapk-dsv41-staging")
ap.add_argument("--dp-rr", type=int, default=0)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("drv", "/tmp/run_dp8_scale_driver.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)
if a.dp_rr:
    import itertools
    import urllib.request as _ur
    _RealReq = _ur.Request
    _ctr = itertools.count()
    _lk = threading.Lock()

    class _RRReq(_RealReq):
        def __init__(self, url, data=None, headers=None, **kw):
            headers = dict(headers or {})
            if data and url.endswith("/v1/completions") and b'"stream": true' in data:
                with _lk:
                    headers["X-data-parallel-rank"] = str(next(_ctr) % a.dp_rr)
            super().__init__(url, data=data, headers=headers, **kw)

    drv.urllib.request.Request = _RRReq


def post(path):
    req = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=b"", method="POST")
    t = time.time()
    urllib.request.urlopen(req, timeout=900).read()
    print(f"[PROF] {path} took {time.time() - t:.1f}s", flush=True)


if a.no_seed:
    # Same prefixes as warmup_decode_bench_templates, without re-sending the 128 seeding requests.
    _real = drv.concurrent.futures.ThreadPoolExecutor

    class _NoSend:
        def __init__(self, *x, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *x):
            return False

        def map(self, fn, it):
            return []

    drv.concurrent.futures.ThreadPoolExecutor = _NoSend
    try:
        prefixes = drv.warmup_decode_bench_templates()
    finally:
        drv.concurrent.futures.ThreadPoolExecutor = _real
else:
    prefixes = drv.warmup_decode_bench_templates()

res = {}
out_json = f"/dev/shm/result_prof_{a.tag}.json"


def bench():
    res["r"] = drv.run_scaled_benchmark(out_json, a.tag, a.conc, a.conc, a.out_len, prefixes=prefixes)


shutil.rmtree("/dev/shm/profiles", ignore_errors=True)
os.makedirs("/dev/shm/profiles", exist_ok=True)
th = threading.Thread(target=bench)
th.start()
if not a.no_profile:
    time.sleep(a.start_after)
    post("/start_profile")
    time.sleep(a.dur)
    post("/stop_profile")
th.join()
r = res.get("r", {})
print(json.dumps({k: r.get(k) for k in ("mean_tpot_ms", "natural_mean_accept_len",
                                        "steady_state_interior_step_throughput_steps_s",
                                        "steady_state_interior_window_tok_s", "wall_clock_output_throughput_tok_s")}), flush=True)
if a.no_profile:
    raise SystemExit(0)
files = sorted(glob.glob("/dev/shm/profiles/**/*", recursive=True))
files = [f for f in files if os.path.isfile(f)]
for f in files:
    print(f"[PROF] {f} {os.path.getsize(f)}", flush=True)
tok = os.environ.get("GCS_TOKEN")
if tok:
    for f in files:
        if not (f.endswith(".xplane.pb") or f.endswith(".pt.trace.json.gz") or f.endswith(".json.gz")):
            continue
        rel = os.path.relpath(f, "/dev/shm/profiles")
        obj = f"profiles/{a.tag}/{rel}"
        url = (f"https://storage.googleapis.com/upload/storage/v1/b/{a.bucket}/o?uploadType=media&name="
               + urllib.parse.quote(obj, safe=""))
        with open(f, "rb") as fh:
            data = fh.read()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/octet-stream"})
        t = time.time()
        urllib.request.urlopen(req, timeout=900).read()
        print(f"[UPLOAD] gs://{a.bucket}/{obj} {len(data) / 1e6:.1f} MB in {time.time() - t:.1f}s", flush=True)
