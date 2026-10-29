#!/usr/bin/env python3
"""PD 正确性检查（logprob 版）。每个用例用全新文本（P、D 都没缓存过），比较第一个生成 token 的 top-5 logprob：
  ref      直接打 D，从头完整计算（不走 PD、没有前缀命中、没有回放）
  pd       经 proxy 走 1P1D，D 没有本地命中（整段从 P 拉）
  pd_hit   先经 proxy 发 base 让 D 缓存前缀，再发 base+suffix（D 命中前缀，只拉命中之后的部分）
注意：P 对 base+suffix 也会命中自己的前缀缓存并回放，所以 pd_hit 天然带着 P 端回放的近似。
    /zzlfs/temu/.venv/bin/python -I pd_logprob_check.py --tokens 3000 20000
"""
import argparse, glob, json, math, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, nargs="+", default=[3000, 20000])
ap.add_argument("--offset", type=int, default=3000000)
ap.add_argument("--proxy", default="http://10.128.0.175:8000")
ap.add_argument("--direct", default="http://10.128.0.177:8200")
args = ap.parse_args()
corpus = "".join(open(p, errors="replace").read() for p in
                 sorted(glob.glob("/zzlfs/temu/vllm-torchtpu/**/*.py", recursive=True)))
QUESTION = "\n\n# Question: what does the code above do? Answer in two sentences.\n# Answer:"


def first_token(url, prompt, rank):
    body = json.dumps({"model": "deepseek-v41", "prompt": prompt, "max_tokens": 4, "temperature": 0,
                       "logprobs": 5}).encode()
    req = urllib.request.Request(url + "/v1/completions", body,
                                 {"Content-Type": "application/json", "X-data-parallel-rank": str(rank)})
    c = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]
    return c["text"], c["logprobs"]["top_logprobs"][0]


def dist(a, b):
    keys = set(a) & set(b)
    return max(abs(a[k] - b[k]) for k in keys) if keys else math.inf


cursor = args.offset
for n in args.tokens:
    def fresh(m):
        global cursor
        s = corpus[cursor: cursor + m * 3]
        cursor += m * 3 + 1000
        return s
    # ref vs pd on one fresh prompt; ref on rank 1 of D, pd pinned to rank 0.
    prompt = fresh(n) + QUESTION
    ref_t, ref = first_token(args.direct, prompt, 1)
    pd_t, pd = first_token(args.proxy, prompt, 0)
    print(f"{n:6d} pd     text {pd_t!r:14} vs ref {ref_t!r:14} max|dlogprob| {dist(pd, ref):.3f}  ref top5 {sorted(ref.items(), key=lambda kv: -kv[1])[:3]}")
    # pd_hit: new base, prime D via PD, then base+suffix; reference on another D rank, fresh.
    base = fresh(n)
    suffix = fresh(1000)
    first_token(args.proxy, base + QUESTION, 0)
    ref_t, ref = first_token(args.direct, base + suffix + QUESTION, 2)
    hit_t, hit = first_token(args.proxy, base + suffix + QUESTION, 0)
    print(f"{n:6d} pd_hit text {hit_t!r:14} vs ref {ref_t!r:14} max|dlogprob| {dist(hit, ref):.3f}  ref top5 {sorted(ref.items(), key=lambda kv: -kv[1])[:3]}")
