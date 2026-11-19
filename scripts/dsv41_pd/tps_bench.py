#!/usr/bin/env python3
"""Prefill 吞吐（TPS）：N 条 cached+new 请求同时发给 P，按 rank 均分（第 i 条固定到 i % dp），
TPS = 新 token 总数 / 墙钟时间（第一条发出到最后一条完成）。直接发 token id（不含分词）。
    /zzlfs/temu/.venv-epic/bin/python -I tps_bench.py --url http://10.128.0.176:8100 --cached 32768 --new 172032 --conc 8 16
"""
import argparse, asyncio, json, time, urllib.request
import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://10.128.0.176:8100")
ap.add_argument("--cached", type=int, default=32768)
ap.add_argument("--new", type=int, nargs="+", required=True)
ap.add_argument("--conc", type=int, nargs="+", default=[8])
ap.add_argument("--dp", type=int, default=8)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--corpus", default="/zzlfs/temu/tools/tpot_corpus_ids.json")
ap.add_argument("--tag", default="")
ap.add_argument("--out", required=True)
args = ap.parse_args()
IDS = json.load(open(args.corpus))
PREFIX = IDS[:args.cached]
POOL = IDS[args.cached:]

def metrics():
    text = urllib.request.urlopen(args.url + "/metrics", timeout=30).read().decode()
    want = {"prompt": "vllm:prompt_tokens_total", "hits": "vllm:prefix_cache_hits_total", "queries": "vllm:prefix_cache_queries_total"}
    return {k: sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith(n + "{") or l.startswith(n + " ")) for k, n in want.items()}

async def send(client, ids, rank):
    r = await client.post(args.url + "/v1/completions", headers={"X-data-parallel-rank": str(rank)},
                          json={"model": "deepseek-v41", "prompt": ids, "max_tokens": 1, "temperature": 0}, timeout=None)
    r.raise_for_status()
    return time.perf_counter()

async def main():
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=256)) as client:
        await asyncio.gather(*(send(client, PREFIX + POOL[r * 64: r * 64 + 16], r) for r in range(args.dp)))
        print(f"# prefix {args.cached} warmed on {args.dp} ranks", flush=True)
        # Random start so that a rerun against the same server does not reuse earlier offsets (prefix-cache hits).
        seq = int(time.time() * 1000) % 1_000_003
        for new in args.new:
            for conc in args.conc:
                for rnd in range(args.rounds + 1):  # round 0 = warmup (new shapes / first touch)
                    prompts = []
                    for i in range(conc):
                        seq += 1
                        off = (seq * 97_003) % (len(POOL) - new - 1)
                        prompts.append(PREFIX + POOL[off: off + new])
                    m0 = metrics(); t0 = time.perf_counter()
                    done = await asyncio.gather(*(send(client, p, i % args.dp) for i, p in enumerate(prompts)))
                    wall = max(done) - t0; m1 = metrics()
                    rec = {"tag": args.tag, "cached": args.cached, "new": new, "conc": conc, "round": rnd,
                           "wall_s": round(wall, 3), "tps": round(conc * new / wall), "tps_per_chip": round(conc * new / wall / 4),
                           "prompt_tokens": int(m1["prompt"] - m0["prompt"]), "cache_hit_tokens": int(m1["hits"] - m0["hits"]),
                           "first_done_s": round(min(done) - t0, 3)}
                    print(json.dumps(rec), flush=True)
                    if rnd:
                        open(args.out, "a").write(json.dumps(rec) + "\n")

asyncio.run(main())
