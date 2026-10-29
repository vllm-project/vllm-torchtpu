#!/usr/bin/env python3
"""1P1D 的 TPOT（ms），格式 CLI / Server，只统计 batch 拼满的窗口。

    /zzlfs/temu/.venv/bin/python -I tpot_bench.py --caches 1024 16384 --bss 1 16 --out /zzlfs/temu/results/tpot.jsonl

每个 cache 长度一组：所有请求共享同一个长度为 cache 的前缀（radix/前缀缓存命中），后面各接 --suffix 个不同 token，
再生成 --out-len 个 token（ignore_eos, temperature 0）。每组先在 8 个 DP rank 上各发一条预热请求，把前缀灌进
P 和 D 的前缀缓存（D 的 KV 只存一份前缀）。prompt 直接发 token id（来自本仓库依赖的 Python 源码，自然代码文本，
DSpark 接受率接近真实业务里的代码场景）。

拼满窗口 = [所有请求都已收到第一个 token 的时刻, 第一个请求收到最后一个 token 的时刻]。
  CLI    = 每条请求在窗口内：(最后一个 token 时刻 - 第一个 token 时刻) / 这之间收到的 token 数，取平均。
  Server = D（:8200）窗口内 vllm:inter_token_latency_seconds 总和增量 / vllm:generation_tokens_total 增量
           （ITL 每条请求每步记一次，投机解码一步可能出多个 token，所以除以 token 数）。
另外记录窗口内 DSpark 平均接受长度（1 + accepted/drafts）。
"""
import argparse
import asyncio
import glob
import json
import os
import re
import statistics
import time
import urllib.request

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--caches", type=int, nargs="+",
                default=[1024, 16384, 32768, 65536, 131072, 262144, 524288, 786432])
ap.add_argument("--bss", type=int, nargs="+", default=[1, 16, 64, 128, 256])
ap.add_argument("--suffix", type=int, default=128)
ap.add_argument("--out-len", type=int, default=1024)
ap.add_argument("--proxy", default="http://10.128.0.175:8000")
ap.add_argument("--d-metrics", default="http://10.128.0.177:8200/metrics")
ap.add_argument("--dp-size", type=int, default=8)
ap.add_argument("--corpus", default="/zzlfs/temu/tools/tpot_corpus_ids.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()

SUFFIX_BASE = 800_000  # suffix 区域放在最长前缀（768k）之后，与前缀不重叠


def corpus_ids():
    if os.path.exists(args.corpus):
        return json.load(open(args.corpus))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/tmp/dsv41-tokenizer/tokenizer.json")
    files = sorted(glob.glob("/zzlfs/temu/.venv-epic/lib/python3.12/site-packages/vllm/**/*.py", recursive=True))
    files += sorted(glob.glob("/zzlfs/temu/.venv-epic/lib/python3.12/site-packages/transformers/models/**/*.py", recursive=True))
    ids = []
    need = SUFFIX_BASE + 256 * args.suffix + 10_000
    for f in files:
        ids.extend(tok.encode(open(f, errors="replace").read(), add_special_tokens=False).ids)
        if len(ids) >= need:
            break
    assert len(ids) >= need, len(ids)
    json.dump(ids, open(args.corpus, "w"))
    return ids


IDS = corpus_ids()
NAMES = {"itl_sum": r"vllm:inter_token_latency_seconds_sum", "itl_cnt": r"vllm:inter_token_latency_seconds_count",
         "gen": r"vllm:generation_tokens_total", "acc": r"vllm:spec_decode_num_accepted_tokens_total",
         "drafts": r"vllm:spec_decode_num_drafts_total"}


def scrape():
    text = urllib.request.urlopen(args.d_metrics, timeout=10).read().decode()
    out = {}
    for key, name in NAMES.items():
        out[key] = sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines()
                       if l.startswith(name + "{") or l.startswith(name + " "))
    return out


async def stream(client, prompt, rank, max_tokens, events):
    body = json.dumps({"model": "deepseek-v41", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "ignore_eos": True, "stream": True,
                       "return_token_ids": True}).encode()
    async with client.stream("POST", f"{args.proxy}/v1/completions", content=body, timeout=None,
                             headers={"Content-Type": "application/json",
                                      "X-data-parallel-rank": str(rank)}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ch = json.loads(data)
            n = sum(len(c.get("token_ids") or []) for c in ch.get("choices", []))
            if n:
                events.append((time.perf_counter(), n))


async def poll_metrics(samples, stop):
    while not stop.is_set():
        t = time.perf_counter()
        try:
            samples.append((t, await asyncio.to_thread(scrape)))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.5)


async def run_group(client, cache, bs):
    prefix = IDS[:cache]
    prompts = [prefix + IDS[SUFFIX_BASE + i * args.suffix: SUFFIX_BASE + (i + 1) * args.suffix] for i in range(bs)]
    events = [[] for _ in range(bs)]
    samples, stop = [], asyncio.Event()
    poller = asyncio.create_task(poll_metrics(samples, stop))
    await asyncio.gather(*(stream(client, p, i % args.dp_size, args.out_len, events[i]) for i, p in enumerate(prompts)))
    stop.set(); await poller
    ws = max(e[0][0] for e in events)
    we = min(e[-1][0] for e in events)
    rec = {"cache": cache, "bs": bs, "suffix": args.suffix, "out_len": args.out_len,
           "window_s": round(we - ws, 3)}
    if we - ws < 1.0:
        rec["error"] = "full-batch window shorter than 1 s"
        return rec
    tp = []
    for ev in events:
        inside = [x for x in ev if ws <= x[0] <= we]
        if len(inside) >= 2:
            tp.append((inside[-1][0] - inside[0][0]) / sum(n for _, n in inside[1:]))
    rec["cli_tpot_ms"] = round(statistics.mean(tp) * 1000, 3)
    inw = [s for s in samples if ws <= s[0] <= we]
    if len(inw) >= 2:
        a, b = inw[0][1], inw[-1][1]
        dg = b["gen"] - a["gen"]
        rec["server_tpot_ms"] = round((b["itl_sum"] - a["itl_sum"]) / dg * 1000, 3) if dg else None
        dd = b["drafts"] - a["drafts"]
        rec["accept_len"] = round(1 + (b["acc"] - a["acc"]) / dd, 2) if dd else None
    return rec


async def main():
    limits = httpx.Limits(max_connections=600, max_keepalive_connections=600)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        for cache in args.caches:
            warm = IDS[:cache] + IDS[SUFFIX_BASE - args.suffix: SUFFIX_BASE]
            t = time.time()
            await asyncio.gather(*(stream(client, warm, r, 1, []) for r in range(args.dp_size)))
            print(f"## cache={cache} warmed on {args.dp_size} ranks in {time.time() - t:.1f}s", flush=True)
            for bs in args.bss:
                for attempt in range(3):
                    try:
                        rec = await run_group(client, cache, bs)
                        break
                    except httpx.HTTPError as exc:  # e.g. a backend connection dropped mid-request
                        print(f"## cache={cache} bs={bs} attempt {attempt} failed: {exc!r}; retrying", flush=True)
                        await asyncio.sleep(5)
                else:
                    rec = {"cache": cache, "bs": bs, "error": "failed 3 attempts"}
                rec["attempts"] = attempt + 1
                print(json.dumps(rec), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")


asyncio.run(main())
