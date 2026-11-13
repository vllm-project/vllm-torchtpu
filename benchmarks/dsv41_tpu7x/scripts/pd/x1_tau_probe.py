#!/usr/bin/env python3
"""X-R2a: DSpark acceptance-length probe against the InferenceX golden rows.

InferenceX measured DeepSeek-V4.1-Flash DSpark K=5 at temperature 1.0 on coding prompts
(probabilistic drafting + block verification): AL 4.07 thinking_off / 3.51 thinking_on.
TPU DSpark drafts greedily with standard rejection. This sends coding prompts in each
(mode, temperature) cell at a fixed concurrency and reads tau from the /metrics deltas, so
the headroom from a sampler change can be sized without the PD harness.
"""
import argparse, concurrent.futures as cf, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--decode", default="http://127.0.0.1:9400")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
ap.add_argument("--conc", type=int, default=32)
ap.add_argument("--n", type=int, default=64, help="requests per cell")
ap.add_argument("--max-tokens", type=int, default=768)
ap.add_argument("--cells", default="off:1.0,on:1.0,off:0.0,on:0.0")
ap.add_argument("--dp", type=int, default=16)
ap.add_argument("--out", default="/tmp/x1_tau_probe.json")
a = ap.parse_args()

TASKS = [
    "Implement an LRU cache class in Python with O(1) get and put, plus unit tests.",
    "Write a Rust function that parses an INI file into a nested HashMap and handles comments and quoted values.",
    "Given a Go HTTP server, add graceful shutdown on SIGTERM with a 10s drain timeout. Show the full main.go.",
    "Write a SQL query (PostgreSQL) that returns the top 3 products by revenue per region for the last 90 days, with ties broken by units sold.",
    "Implement Dijkstra's algorithm in C++17 using a binary heap over an adjacency list; include a small driver.",
    "Refactor this Python: a 200-line function that reads CSV, validates rows, and writes JSON. Describe the structure and provide the refactored code.",
    "Write a TypeScript React hook useDebouncedValue<T>(value, delayMs) with cleanup and tests using @testing-library/react.",
    "Implement a thread-safe bounded blocking queue in Java with put/take and a timed poll.",
    "Write a bash script that rotates logs in /var/log/app, compresses files older than 7 days, and deletes those older than 30, with dry-run support.",
    "Implement a tokenizer for a tiny arithmetic language (+ - * / parentheses, floats) and a recursive-descent parser producing an AST in Python.",
    "Write a Kubernetes Deployment + Service + HPA YAML for a stateless API with resource requests, readiness/liveness probes, and pod anti-affinity.",
    "Implement binary search over a rotated sorted array in Python; explain edge cases and give tests.",
    "Write a Python asyncio client that fetches 1000 URLs with a concurrency limit of 50, retries with exponential backoff, and reports p50/p95 latency.",
    "Implement a minimal Flask app with JWT auth: register, login, and a protected /me endpoint. Include password hashing.",
    "Write a CUDA kernel for a tiled matrix multiply (fp32) with shared memory and explain the indexing.",
    "Implement rate limiting (token bucket) as Express middleware in Node.js with per-IP buckets and a Redis backend.",
]


def payload(i: int, mode: str, temp: float) -> dict:
    t = TASKS[i % len(TASKS)]
    return {
        "model": a.model,
        "messages": [{"role": "user", "content": f"{t} (variant {i})"}],
        "chat_template_kwargs": {"thinking": mode == "on", "reasoning_effort": "high" if mode == "on" else "none"},
        "max_tokens": a.max_tokens,
        "temperature": temp,
        "stream": False,
    }


def metrics() -> tuple[float, float, dict]:
    drafts = acc = 0.0
    per_pos: dict[int, float] = {}
    with urllib.request.urlopen(f"{a.decode}/metrics", timeout=20) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_drafts"):
                drafts += float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
                pos = int(line.split('position="')[1].split('"')[0])
                per_pos[pos] = per_pos.get(pos, 0.0) + float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens"):
                acc += float(line.split()[-1])
    return drafts, acc, per_pos


def one(i: int, mode: str, temp: float) -> tuple[int, int]:
    req = urllib.request.Request(
        f"{a.decode}/v1/chat/completions", data=json.dumps(payload(i, mode, temp)).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(i % a.dp)})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage", {})
    return int(u.get("completion_tokens", 0)), int(u.get("prompt_tokens", 0))


results = []
for cell in a.cells.split(","):
    mode, temp = cell.split(":"); temp = float(temp)
    d0, a0, p0 = metrics(); t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        outs = list(ex.map(lambda i: one(i, mode, temp), range(a.n)))
    dt = time.time() - t0
    d1, a1, p1 = metrics()
    drafts = d1 - d0; acc = a1 - a0
    tau = 1.0 + acc / drafts if drafts else None
    per_pos = {k: round((p1.get(k, 0) - p0.get(k, 0)) / drafts, 4) for k in sorted(p1)} if drafts else {}
    comp = sum(o[0] for o in outs)
    row = {"mode": mode, "temperature": temp, "n": a.n, "conc": a.conc, "completion_tokens": comp,
           "drafts": drafts, "accepted": acc, "tau": tau, "per_pos_accept": per_pos,
           "out_tok_s": comp / dt, "wall_s": dt}
    results.append(row)
    print(f"[TAU] thinking={mode} temp={temp} tau={tau} per_pos={list(per_pos.values())} "
          f"tokens={comp} tok/s={comp/dt:.0f}", flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    time.sleep(2)
