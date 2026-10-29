#!/usr/bin/env python3
"""D 端单请求耗时分解：发 N 轮 cached+new 请求（每轮先预热前缀），取 D（:8200）vLLM 指标的差值。
    /zzlfs/temu/.venv/bin/python -I d_breakdown.py --cached 524288 --new 32768 --rounds 2
输出每轮：D 的 time_to_first_token / queue（含等远端 KV）/ prefill（被调度到首 token）/ inference，以及 proxy 侧总 TTFT。"""
import argparse, json, re, sys, time, urllib.request
sys.path.insert(0, "/zzlfs/temu/scripts")
import random
import bench_cached_prefill as b  # 复用造文本的逻辑
WORDS = b.word_pool("/tmp/dsv41-tokenizer/tokenizer.json")

ap = argparse.ArgumentParser()
ap.add_argument("--cached", type=int, default=524288)
ap.add_argument("--new", type=int, default=32768)
ap.add_argument("--rounds", type=int, default=2)
args = ap.parse_args()
D = "http://10.128.0.177:8200/metrics"
NAMES = ["time_to_first_token_seconds", "request_queue_time_seconds", "request_prefill_time_seconds",
         "request_inference_time_seconds", "e2e_request_latency_seconds"]


def snap():
    out = {}
    for l in urllib.request.urlopen(D, timeout=30).read().decode().splitlines():
        for n in NAMES:
            m = re.match(rf"vllm:{n}_(sum|count)\{{[^}}]*\}} (\S+)", l)
            if m:
                out.setdefault(n, [0.0, 0.0])[0 if m.group(1) == "sum" else 1] += float(m.group(2))
    return out


def send(prompt, max_tokens):
    body = json.dumps({"model": "deepseek-v41", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request("http://10.128.0.175:8000/v1/completions", body,
                                 {"Content-Type": "application/json", "X-data-parallel-rank": "0"})
    t = time.time(); ttft = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if ttft is None and line.startswith(b"data: {") and b'"text":""' not in line:
                ttft = time.time() - t
    return ttft


for rnd in range(args.rounds):
    rng = random.Random(time.time_ns())
    cached, new = b.make_text(rng, WORDS, args.cached), b.make_text(rng, WORDS, args.new)
    send(cached, 1)
    a = snap(); ttft = send(cached + new, 4); time.sleep(2); z = snap()
    parts = {n: (z[n][0] - a.get(n, [0, 0])[0]) / max(1, z[n][1] - a.get(n, [0, 0])[1]) for n in NAMES if n in z}
    print(f"round {rnd}: proxy TTFT {ttft:.3f}s | D: " + " ".join(f"{n.replace('_seconds','')}={v:.3f}" for n, v in parts.items()))
