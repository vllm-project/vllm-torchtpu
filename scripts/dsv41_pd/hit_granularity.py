#!/usr/bin/env python3
"""测 P 前缀命中的粒度：先发 base（N 个 token），再发 base+k 个 token，看第二次真正算了多少 token。"""
import json, urllib.request, random, sys, re
P = "http://10.128.0.100:8100"; RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
def metric(src):
    t = urllib.request.urlopen(P + "/metrics").read().decode()
    m = re.search(r'vllm:prompt_tokens_by_source_total\{engine="%d",model_name="deepseek-v41",source="%s"\} (\S+)' % (RANK, src), t)
    return float(m.group(1))
def send(ids):
    body = json.dumps({"model": "deepseek-v41", "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
    urllib.request.urlopen(urllib.request.Request(P + "/v1/completions", body, {"Content-Type": "application/json", "X-data-parallel-rank": str(RANK)}), timeout=600).read()
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
for base_n, extra in [(40000, 100), (40000, 1000), (40000, 3000), (40007, 5000), (100003, 4400)]:
    base = [rng.randrange(1000, 100000) for _ in range(base_n)]
    send(base)
    c0, h0 = metric("local_compute"), metric("local_cache_hit")
    send(base + [rng.randrange(1000, 100000) for _ in range(extra)])
    c, h = metric("local_compute") - c0, metric("local_cache_hit") - h0
    print(f"base {base_n:6d} + {extra:5d}: computed {c:7.0f} hit {h:7.0f}  (ideal computed {extra}, waste {c - extra:.0f})")
