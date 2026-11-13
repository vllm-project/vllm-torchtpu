#!/usr/bin/env python3
"""In-pod: relaunch the DP=8 server with launch-script edits, run gates + benches.

  python3 /tmp/inpod_relaunch.py --tag A --sub 'OLD=>NEW' ... --env K=V ... --bench 64,64 [--no-gates]

* edits are applied to a copy of /tmp/run_dp8_scale_driver.py (the runner-patched driver)
* --env lines are exported right before `exec vllm serve` (f-string braces escaped)
* --sitecustomize K=V edits the forced-override dict in every installed sitecustomize copy
Results: /dev/shm/relaunch_<tag>.json (gates + per-bench summary incl. mean ITL = real step).
"""
import argparse
import glob
import importlib.util
import json
import re
import statistics
import time

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--sub", action="append", default=[])
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--sitecustomize", action="append", default=[])
ap.add_argument("--bench", default="64")
ap.add_argument("--out-len", type=int, default=1024)
ap.add_argument("--no-gates", action="store_true")
ap.add_argument("--no-boot", action="store_true")
ap.add_argument("--dp-rr", type=int, default=0, help="pin bench requests round-robin over N DP ranks")
a = ap.parse_args()

src = open("/tmp/run_dp8_scale_driver.py").read()
# The server log can turn "binary" for grep (NULs after a relaunch truncates it); the boot loop
# would then never see "Memory statistics". Always treat it as text.
src = src.replace("grep -E '", "grep -a -E '")
for s in a.sub:
    old, new = s.split("=>", 1)
    assert src.count(old) >= 1, f"pattern not found: {old!r}"
    src = src.replace(old, new)
if a.env:
    anchor = "exec vllm serve /dev/shm/DeepSeek-V4.1-Flash"
    lines = "".join(f"export {kv}\n" for kv in a.env).replace("{", "{{").replace("}", "}}")
    src = src.replace(anchor, lines + anchor)
path = f"/tmp/drv_{a.tag}.py"
open(path, "w").write(src)

for kv in a.sitecustomize:
    k, v = kv.split("=", 1)
    for sc in glob.glob("/usr/local/lib/python3.12/sitecustomize.py") + glob.glob(
            "/usr/local/lib/python3.12/site-packages/sitecustomize.py") + glob.glob(
            "/usr/lib/python3.12/sitecustomize.py") + glob.glob("/workspace/sitecustomize_bench/sitecustomize.py"):
        t = open(sc).read()
        t2, n = re.subn(rf'"{k}": "[^"]*"', f'"{k}": "{v}"', t)
        if n == 0:
            t2 = t.replace("}.items():\n    os.environ[_k] = _v", f'    "{k}": "{v}",\n}}.items():\n    os.environ[_k] = _v', 1)
        open(sc, "w").write(t2)
        print(f"[SC] {sc}: {k}={v} ({n} replaced)", flush=True)

spec = importlib.util.spec_from_file_location("drv", path)
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

if a.dp_rr:
    # Pin benchmark requests round-robin to DP ranks (what `vllm bench serve` does with
    # VLLM_BENCH_DP_SIZE=8 via the X-data-parallel-rank header), so each rank holds c/8.
    import itertools
    import threading
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

out = {"tag": a.tag, "sub": a.sub, "env": a.env, "sitecustomize": a.sitecustomize, "benches": []}
t0 = time.time()
if not a.no_boot and not glob.glob("/dev/shm/DeepSeek-V4.1-Flash/*.safetensors"):
    # The driver deletes the shards once the server is up; fetch them again (needs a fresh
    # GCS_TOKEN in /tmp/tok.sh, written by the host side).
    import subprocess
    subprocess.run("set -e; rm -f /dev/shm/DeepSeek-V4.1-Flash/.done_*; source /tmp/dsv41_env.sh; "
                   "[ -f /tmp/tok.sh ] && source /tmp/tok.sh; "
                   "python3 -u /tmp/dsv41_dl.py", shell=True, check=True, executable="/bin/bash")
    out["download_s"] = round(time.time() - t0, 1)
    t0 = time.time()
if not a.no_boot:
    drv.boot_dp8_scale_server(topology="dp8")
    out["boot_s"] = round(time.time() - t0, 1)
if not a.no_gates:
    g = drv.run_sanity_and_usa_checks()
    out["gates"] = {k: g[k] for k in ("passed_2p2", "num_us_states_matched", "usa_query_real_dspark_mean_accept_len",
                                      "usa_query_per_position_acceptance_rates")}
    out["usa_tail"] = g["usa_query_sample_output"][-200:]
prefixes = drv.warmup_decode_bench_templates()
for c in [int(x) for x in a.bench.split(",") if x]:
    r = drv.run_scaled_benchmark(f"/dev/shm/result_relaunch_{a.tag}_c{c}.json", f"{a.tag}_c{c}", c, c, a.out_len,
                                 prefixes=prefixes)
    itl = [x for l in r["itls"] for x in l]
    # interior window ITLs only
    s = {k: r.get(k) for k in ("mean_tpot_ms", "natural_mean_accept_len", "steady_state_interior_window_tok_s",
                               "wall_clock_output_throughput_tok_s", "steady_state_interior_step_throughput_steps_s")}
    s["c"] = c
    s["mean_itl_ms"] = round(1000 * statistics.mean(itl), 2)
    s["median_itl_ms"] = round(1000 * statistics.median(itl), 2)
    s["step_ms_tpot_x_tau"] = round(r["mean_tpot_ms"] * r["natural_mean_accept_len"], 2)
    out["benches"].append(s)
    print(f"[RELAUNCH-{a.tag}] {json.dumps(s)}", flush=True)
json.dump(out, open(f"/dev/shm/relaunch_{a.tag}.json", "w"), indent=1)
print(f"[RELAUNCH-{a.tag}] DONE {json.dumps(out)[:2000]}", flush=True)
