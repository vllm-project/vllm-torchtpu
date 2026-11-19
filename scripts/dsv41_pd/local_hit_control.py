#!/usr/bin/env python3
"""对照组：不走 PD，只看 D 本地前缀命中（bounded replay）相对从头计算的 logprob 偏差。
  hit  先直接打 D rank0 发 base，再发 base+suffix（rank0 本地命中前缀）
  ref  base+suffix 在 D rank2 上从头算"""
import argparse, glob, json, math, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, nargs="+", default=[3000, 20000, 100000])
ap.add_argument("--offset", type=int, default=15000000)
ap.add_argument("--direct", default="http://10.128.0.177:8200")
args = ap.parse_args()
corpus = "".join(open(p, errors="replace").read() for p in
                 sorted(glob.glob("/zzlfs/temu/vllm-torchtpu/**/*.py", recursive=True)))
Q = "\n\n# Question: what does the code above do? Answer in two sentences.\n# Answer:"
def ft(prompt, rank):
    body = json.dumps({"model": "deepseek-v41", "prompt": prompt, "max_tokens": 4, "temperature": 0, "logprobs": 5}).encode()
    req = urllib.request.Request(args.direct + "/v1/completions", body, {"Content-Type": "application/json", "X-data-parallel-rank": str(rank)})
    c = json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]
    return c["text"], c["logprobs"]["top_logprobs"][0]
def dist(a, b):
    k = set(a) & set(b); return max(abs(a[x] - b[x]) for x in k) if k else math.inf
cur = args.offset
for n in args.tokens:
    base = corpus[cur: cur + n * 3]; cur += n * 3 + 1000
    assert len(base) == n * 3 and cur + 4000 <= len(corpus), f'corpus exhausted at offset {cur} (len {len(corpus)})'
    suf = corpus[cur: cur + 3000]; cur += 4000
    ft(base + Q, 0)
    ht, h = ft(base + suf + Q, 0)
    rt, r = ft(base + suf + Q, 2)
    print(f"{n:6d} local_hit text {ht!r:28} vs ref {rt!r:28} max|dlogprob| {dist(h, r):.3f}", flush=True)
