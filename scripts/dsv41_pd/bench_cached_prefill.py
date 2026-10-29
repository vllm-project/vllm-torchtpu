#!/usr/bin/env python3
"""1P1D「cached + prefill」延迟测试：P 节点处理时间（Prefill）和端到端 TTFT。

    /zzlfs/temu/.venv/bin/python bench_cached_prefill.py --cached 1024 --new 1024 --conc 1 --repeats 5
    /zzlfs/temu/.venv/bin/python bench_cached_prefill.py --cached 1024 --new 1024 --conc 4 --pin distinct

口径（和客户表格对齐）：
  Prefill = P 节点对这条请求的计算时间，取 prefill 服务 vllm:request_prefill_time_seconds
            （从被调度到产出第一个 token，不含排队）。直方图是累计值，取每轮前后差值：
            conc=1 是这一条的值，conc>1 是这一轮几条的平均。
  TTFT    = 客户端发出请求到收到第一个非空 token（流式），含分词、排队、P 计算、KV 传到 D、D 出首 token。
            请求体在计时前就序列化好（客户端工作；在计时内做还会让并发请求在客户端事件循环上错开）。
输入是原始文本（/v1/completions 的 text prompt），服务端分词计入 TTFT。

每轮：每条请求生成独立的随机文本（cached 部分 + new 部分，token 数精确）→ 逐条只发 cached 部分预热
（max_tokens=1）→ 读 /metrics → 同时发出 conc 条 cached+new → 读 /metrics。预热和正式请求带同一个
X-data-parallel-rank，保证落在 P / D 的同一个 DP 副本、命中同一份前缀缓存。
  --pin distinct：第 i 条固定到 DP 副本 i % dp_size（默认；超过 dp_size 条时多条共享一个副本）
  --pin same    ：全部固定到副本 0（模拟同一个处理单元上排队）
文本用 tokenizer 里「空格+小写单词」的 token 拼成，pre-tokenizer 按空格切分，所以编码后 token 数不变。
"""
import argparse
import asyncio
import json
import random
import re
import statistics
import time

import httpx
from tokenizers import Tokenizer

METRICS = {
    "prefill_sum": "vllm:request_prefill_time_seconds_sum",
    "prefill_cnt": "vllm:request_prefill_time_seconds_count",
    "queue_sum": "vllm:request_queue_time_seconds_sum",
    "queue_cnt": "vllm:request_queue_time_seconds_count",
    "prompt_tok": "vllm:prompt_tokens_total",
    "cached_tok": "vllm:prompt_tokens_cached_total",
    "computed_tok": "vllm:request_prefill_kv_computed_tokens_sum",
}


def word_pool(tokenizer_path):
    vocab = Tokenizer.from_file(tokenizer_path).get_vocab()
    return [k[1:] for k in vocab if re.fullmatch(r"Ġ[a-z]{3,9}", k)]


def make_text(rng, words, n_tokens):
    return "".join(" " + rng.choice(words) for _ in range(n_tokens))


async def scrape(client, url):
    text = (await client.get(url, timeout=30)).text
    out = {}
    for key, name in METRICS.items():
        total = 0.0
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
        out[key] = total
    return out


async def warm(client, args, prompt, rank):
    r = await client.post(
        f"{args.proxy}/v1/completions",
        headers={"X-data-parallel-rank": str(rank)},
        json={"model": args.model, "prompt": prompt, "max_tokens": 1, "temperature": 0},
        timeout=None,
    )
    r.raise_for_status()


def encode_body(args, prompt):
    # Encoded before the clock starts: serializing a multi-MB prompt is client
    # work, and done inside the timed section it also staggers concurrent
    # requests on the client's event loop (4P: the 2nd-4th left ~130 ms late).
    return json.dumps({
        "model": args.model, "prompt": prompt, "max_tokens": args.max_tokens, "temperature": 0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
    }).encode()


async def measure(client, args, payload, rank):
    t0 = time.perf_counter()
    ttft, usage = None, None
    async with client.stream("POST", f"{args.proxy}/v1/completions", content=payload, timeout=None,
                             headers={"X-data-parallel-rank": str(rank),
                                      "Content-Type": "application/json"}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if ttft is None and any(c.get("text") for c in chunk.get("choices", [])):
                ttft = time.perf_counter() - t0
    return {"ttft": ttft, "e2e": time.perf_counter() - t0, "prompt_tokens": (usage or {}).get("prompt_tokens")}


async def one_round(client, args, rng, words, round_idx):
    # distinct: request i -> DP rank i % dp_size (more than dp_size requests share ranks)
    ranks = [i % args.dp_size for i in range(args.conc)] if args.pin == "distinct" else [0] * args.conc
    samples = [(make_text(rng, words, args.cached), make_text(rng, words, args.new)) for _ in range(args.conc)]
    for (cached, _), rank in zip(samples, ranks):
        await warm(client, args, cached, rank)
    await asyncio.sleep(1)
    m0 = await scrape(client, args.prefill_metrics)
    payloads = [encode_body(args, c + n) for c, n in samples]
    results = await asyncio.gather(*(measure(client, args, b, rank) for b, rank in zip(payloads, ranks)))
    await asyncio.sleep(2)  # 让 P 节点把这一轮的指标刷出来
    m1 = await scrape(client, args.prefill_metrics)
    d = {k: m1[k] - m0[k] for k in m0}
    rec = {
        "round": round_idx, "cached": args.cached, "new": args.new, "conc": args.conc, "pin": args.pin,
        "ranks": ranks, "requests": results,
        "prefill_s": d["prefill_sum"] / d["prefill_cnt"] if d["prefill_cnt"] else None,
        "queue_s": d["queue_sum"] / d["queue_cnt"] if d["queue_cnt"] else None,
        "p_requests": d["prefill_cnt"],
        "cached_tok": d["cached_tok"], "computed_tok": d["computed_tok"], "prompt_tok": d["prompt_tok"],
        "ttft_mean": statistics.mean(r["ttft"] for r in results),
        "ttft_max": max(r["ttft"] for r in results),
    }
    print(f"round {round_idx}: prefill {rec['prefill_s']:.3f}s queue {rec['queue_s']:.3f}s  "
          f"ttft mean {rec['ttft_mean']:.3f}s max {rec['ttft_max']:.3f}s  "
          f"P reqs {d['prefill_cnt']:.0f} prompt {d['prompt_tok']:.0f} cached {d['cached_tok']:.0f} "
          f"computed {d['computed_tok']:.0f}  prompt_tokens(usage) {[r['prompt_tokens'] for r in results]}",
          flush=True)
    return rec


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cached", type=int, required=True, help="缓存部分 token 数")
    ap.add_argument("--new", type=int, required=True, help="新算部分 token 数")
    ap.add_argument("--conc", type=int, default=1, help="同时发出的请求数（1P=1，4P=4）")
    ap.add_argument("--pin", choices=["distinct", "same"], default="distinct")
    ap.add_argument("--dp-size", type=int, default=8, help="P/D 的 DP 副本数；--pin distinct 时第 i 条固定到 i %% dp_size")
    ap.add_argument("--repeats", type=int, default=5, help="正式轮数（另加 --warmup-rounds 轮不计入）")
    ap.add_argument("--warmup-rounds", type=int, default=1, help="先跑几轮丢弃，排除首次编译")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--proxy", default="http://10.128.0.175:8000")
    ap.add_argument("--prefill-metrics", default="http://10.128.0.175:8100/metrics")
    ap.add_argument("--model", default="deepseek-v41")
    ap.add_argument("--tokenizer", default="/tmp/dsv41-tokenizer/tokenizer.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="追加写 jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    words = word_pool(args.tokenizer)
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=64)) as client:
        recs = []
        for i in range(args.warmup_rounds + args.repeats):
            rec = await one_round(client, args, rng, words, i)
            rec["warmup"] = i < args.warmup_rounds
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if not rec["warmup"]:
                recs.append(rec)
    pre = [r["prefill_s"] for r in recs if r["prefill_s"] is not None]
    print(f"== {args.cached}+{args.new} conc={args.conc} pin={args.pin}  {len(recs)} rounds (median): "
          f"Prefill {statistics.median(pre):.3f} / TTFT {statistics.median(r['ttft_mean'] for r in recs):.3f}  "
          f"(TTFT max median {statistics.median(r['ttft_max'] for r in recs):.3f}, "
          f"queue median {statistics.median(r['queue_s'] or 0 for r in recs):.3f})")


if __name__ == "__main__":
    asyncio.run(main())
