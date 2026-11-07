#!/usr/bin/env python3
"""X1 — decode step-time anatomy (phase A, no profiler).

Runs decode-only load directly against the 1D server (bypassing 1P) with a shared cached
prefix so that each step processes a controlled number of tokens per DP rank:
  conc = 64 / 128 / 256 / 512  ->  8 / 16 / 32 / 64 seqs per rank  ->  48 / 96 / 192 / 384 tok/step/rank (DSpark K=5).
Reports steady-state step time, interior tok/s, natural tau and TPOT per level, and fits
step_ms = floor + slope * tokens_per_rank.
Usage (in decode pod): python3 x1_step_anatomy.py --decode http://127.0.0.1:9400 --ctx 20000 --out-len 512
"""
import argparse, json, sys, time
sys.path.insert(0, "/tmp")
import pd_gates as G  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--decode", default="http://127.0.0.1:9400")
ap.add_argument("--tokenizer", default="/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json")
ap.add_argument("--ctx", type=int, default=20000)
ap.add_argument("--out-len", type=int, default=512)
ap.add_argument("--concs", default="64,128,256,512")
ap.add_argument("--dp", type=int, default=8)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", default="/tmp/x1_step_anatomy.json")
a = ap.parse_args()

tok = Tokenizer.from_file(a.tokenizer)

# Pin requests round-robin to DP ranks (X-data-parallel-rank) so each rank carries conc/dp
# sequences; the server-side DP load balancer otherwise packs ranks unevenly.
import itertools, threading, urllib.request
_rr = itertools.cycle(range(a.dp)); _lock = threading.Lock()
_orig_Request = urllib.request.Request
class _PinnedRequest(_orig_Request):
    def __init__(self, url, data=None, headers=None, *args, **kw):
        headers = dict(headers or {})
        if "/v1/completions" in url:
            with _lock:
                headers["X-data-parallel-rank"] = str(next(_rr))
        super().__init__(url, data=data, headers=headers, *args, **kw)
urllib.request.Request = _PinnedRequest
G.urllib.request.Request = _PinnedRequest
results = []
# warm the shared prefix once (prefix cache on 1D), small output
warm = [G.build_prompt_ids(tok, a.ctx, salt=f"w{i}", treatise=True) for i in range(2 * a.dp)]
G.bench(a.decode, warm, 8, 2 * a.dp, decode_base=a.decode)
for c in [int(x) for x in a.concs.split(",")]:
    prompts = [G.build_prompt_ids(tok, a.ctx, salt=f"c{c}_{i}", treatise=True) for i in range(c)]
    b = G.bench(a.decode, prompts, a.out_len, c, decode_base=a.decode)
    tok_per_rank = c // a.dp * (a.k + 1)
    row = {
        "conc": c, "seqs_per_rank": c // a.dp, "tokens_per_rank_step": tok_per_rank, "ctx": a.ctx,
        "out_len": a.out_len, "step_ms": b["steady_state_step_ms"], "tau": b["natural_mean_accept_len"],
        "interior_tok_s": b["steady_state_interior_window_tok_s"],
        "interior_tok_s_at_3p5": b["steady_state_interior_throughput_at_accept_3p5_tok_s"],
        "wall_tok_s": b["output_tok_s"], "tpot_ms": b["tpot_mean_ms"], "ttft_p50_s": b["ttft_p50_s"],
        "per_pos": b["per_pos_accept_rates"], "duration_s": b["duration_s"],
    }
    row["per_chip_interior"] = (row["interior_tok_s"] or 0) / max(1, a.dp // 2)
    row["per_user_tok_s"] = (row["interior_tok_s"] or 0) / c
    results.append(row)
    print(f"[X1] conc={c:>4} seqs/rank={row['seqs_per_rank']:>3} tok/step/rank={tok_per_rank:>4} | step={row['step_ms']} ms "
          f"tau={row['tau']} | interior={row['interior_tok_s']} tok/s ({row['per_chip_interior']:.0f}/chip, {row['per_user_tok_s']:.1f}/user) "
          f"| @3.5={row['interior_tok_s_at_3p5']} | tpot={row['tpot_ms']} ms ttft_p50={row['ttft_p50_s']} s", flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    time.sleep(3)

# linear fit step_ms = floor + slope * tokens
pts = [(r["tokens_per_rank_step"], r["step_ms"]) for r in results if r["step_ms"]]
if len(pts) >= 2:
    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); floor = (sy - slope * sx) / n
    print(f"[X1] fit: step_ms = {floor:.1f} + {slope:.3f} * tokens_per_rank  (ctx={a.ctx})")
    json.dump({"rows": results, "fit": {"floor_ms": floor, "slope_ms_per_token": slope}}, open(a.out, "w"), indent=1)
