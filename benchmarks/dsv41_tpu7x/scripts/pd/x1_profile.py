#!/usr/bin/env python3
"""X1 phase B: capture a torch/TPU profile during steady-state decode at a given concurrency.
Server must be launched with --profiler-config (torch_profiler_dir=/dev/shm/profiles).
  python3 x1_profile.py --conc 64 --ctx 20000 --out-len 1024 --start-after 12 --dur 1.5 --tag c64
"""
import argparse, glob, os, shutil, sys, threading, time, urllib.request, itertools, json
sys.path.insert(0, "/tmp")
import pd_gates as G
from tokenizers import Tokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--decode", default="http://127.0.0.1:9400")
ap.add_argument("--conc", type=int, default=64); ap.add_argument("--ctx", type=int, default=20000)
ap.add_argument("--out-len", type=int, default=1024); ap.add_argument("--start-after", type=float, default=12.0)
ap.add_argument("--dur", type=float, default=1.5); ap.add_argument("--tag", default="prof"); ap.add_argument("--dp", type=int, default=8)
a = ap.parse_args()
_rr = itertools.cycle(range(a.dp)); _lock = threading.Lock(); _orig = urllib.request.Request
class _Pinned(_orig):
    def __init__(self, url, data=None, headers=None, *args, **kw):
        headers = dict(headers or {})
        if "/v1/completions" in url:
            with _lock: headers["X-data-parallel-rank"] = str(next(_rr))
        super().__init__(url, data=data, headers=headers, *args, **kw)
urllib.request.Request = _Pinned; G.urllib.request.Request = _Pinned
def post(path):
    req = _orig(f"{a.decode}{path}", data=b"", method="POST"); t = time.time()
    urllib.request.urlopen(req, timeout=900).read(); print(f"[PROF] {path} took {time.time()-t:.1f}s", flush=True)
tok = Tokenizer.from_file("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json")
prompts = [G.build_prompt_ids(tok, a.ctx, salt=f"p{a.tag}_{i}", treatise=True) for i in range(a.conc)]
shutil.rmtree("/dev/shm/profiles", ignore_errors=True); os.makedirs("/dev/shm/profiles", exist_ok=True)
res = {}
th = threading.Thread(target=lambda: res.__setitem__("r", G.bench(a.decode, prompts, a.out_len, a.conc, decode_base=a.decode))); th.start()
time.sleep(a.start_after); post("/start_profile"); time.sleep(a.dur); post("/stop_profile"); th.join()
r = res["r"]; print(json.dumps({k: r.get(k) for k in ("steady_state_step_ms","natural_mean_accept_len","steady_state_interior_window_tok_s","tpot_mean_ms","ttft_p50_s")}), flush=True)
files = [f for f in sorted(glob.glob("/dev/shm/profiles/**/*", recursive=True)) if os.path.isfile(f)]
for f in files: print(f"[PROF] {f} {os.path.getsize(f)}", flush=True)
dst = f"/dev/shm/profiles_{a.tag}"; shutil.rmtree(dst, ignore_errors=True); shutil.copytree("/dev/shm/profiles", dst); print("[PROF] saved", dst)
