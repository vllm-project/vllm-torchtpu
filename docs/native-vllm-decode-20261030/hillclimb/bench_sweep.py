# SPDX-License-Identifier: Apache-2.0
"""Fixed-length P/D streaming client for any engine layout.

Same request construction, arrival pacing and steady-window statistics as
../benchmark.py, generalized to the layouts of layout.sh: requests are spread
evenly over the engines, each engine is addressed through its host's API
server (and X-data-parallel-rank for hosts that run several engines).

  LAYOUT=tp32dcp8 GLOBAL_SEQS=96 python bench_sweep.py --stage tp32dcp8-s96-k3 \
      --input 131072 --output 1024 --arrival-interval 0.25

Reads NATIVE_HOST_IPS, NATIVE_BENCH_DATASET, NATIVE_BENCH_OUTPUT_DIR like benchmark.py.
Also records the spec-decode and step counters from /metrics before and after the
run, so the MTP acceptance per draft position can be derived.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import aiohttp

ROOT = Path(os.environ.get("NATIVE_BENCH_OUTPUT_DIR", "./benchmark-results"))
ROOT.mkdir(parents=True, exist_ok=True)
HOSTS = os.environ["NATIVE_HOST_IPS"].split(",")
LAYOUT = os.environ.get("LAYOUT", "dp8tp4")
GLOBAL_SEQS = int(os.environ.get("GLOBAL_SEQS", "96"))

# engine -> (host index, local data-parallel rank or None)
ENGINE_MAP = {
    "dp8tp4": [(i // 2, i % 2) for i in range(8)],
    "dp4tp8dcp2": [(i, 0) for i in range(4)],
    "dp4tp8": [(i, 0) for i in range(4)],
    "tp32dcp8": [(0, None)],
    "tp32dcp4": [(0, None)],
    "dp2tp16dcp4": [
        (0, None),
        (0, None),
    ],  # one API server on host 0 fronts both engines
}
METRIC_KEYS = [
    "spec_decode",
    "prefix_cache",
    "request_success",
    "num_preemptions",
    "num_requests_running",
    "iteration",
    "generation_tokens",
]


def percentile(v, q):
    v = sorted(v)
    return v[int((len(v) - 1) * q)] if v else None


async def run(a):
    dataset = json.loads(Path(os.environ["NATIVE_BENCH_DATASET"]).read_text())
    engines = ENGINE_MAP[LAYOUT]
    per_engine = GLOBAL_SEQS // len(engines)
    assert per_engine * len(engines) == GLOBAL_SEQS, (GLOBAL_SEQS, len(engines))
    assignments = [(e, s) for e in range(len(engines)) for s in range(per_engine)]
    run_id = f"{LAYOUT}-" + uuid.uuid4().hex[:12]
    root = ROOT / "benchmarks" / run_id
    root.mkdir(parents=True)
    (root / "parameters.json").write_text(
        json.dumps({**vars(a), "layout": LAYOUT, "global_seqs": GLOBAL_SEQS}, indent=2)
    )
    timeout = aiohttp.ClientTimeout(total=7200, sock_read=7200)
    api_hosts = sorted({HOSTS[h] for h, _ in engines})
    async with aiohttp.ClientSession(
        read_bufsize=16 << 20,
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=256),
    ) as client:

        async def metrics():
            out = {}
            for host in api_hosts:
                try:
                    async with client.get(f"http://{host}:8000/metrics") as response:
                        response.raise_for_status()
                        text = await response.text()
                except Exception as e:  # metrics are best effort
                    out[host] = {"error": repr(e)}
                    continue
                rows = {}
                for line in text.splitlines():
                    if line.startswith("#") or not any(k in line for k in METRIC_KEYS):
                        continue
                    try:
                        name, value = line.rsplit(" ", 1)
                        rows[name] = float(value)
                    except ValueError:
                        continue
                out[host] = rows
            return out

        before = await metrics()
        ready = asyncio.Event()
        all_events = [[] for _ in assignments]

        async def one(i, engine, slot):
            host_idx, local_rank = engines[engine]
            # 128 prompts x 262144 tokens: index by (engine, slot); beyond 128 take the
            # second half of a prompt so every request still has distinct content.
            k = engine * per_engine + slot
            item = dataset[k % len(dataset)]
            suffix = item["suffix_ids"]
            body_len = a.input - len(suffix)
            offset = (k // len(dataset)) * body_len
            prompt = item["token_ids"][offset : offset + body_len] + suffix
            assert len(prompt) == a.input, (len(prompt), a.input)
            await ready.wait()
            await asyncio.sleep((i % (2 * per_engine)) * a.arrival_interval)
            start = time.perf_counter()
            row = {
                "index": i,
                "engine": engine,
                "host": host_idx,
                "local_dp": local_rank,
                "slot": slot,
                "start": start,
                "wall_start": time.time(),
            }
            events = all_events[i]
            payload = {
                "model": "qwen38-native-dummy",
                "prompt": prompt,
                "max_tokens": a.output,
                "min_tokens": a.output,
                "ignore_eos": True,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            headers = {"X-Request-Id": f"{run_id}-{i}"}
            if local_rank is not None:
                headers["X-data-parallel-rank"] = str(local_rank)
            try:
                async with client.post(
                    f"http://{HOSTS[host_idx]}:8000/v1/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}: " + (await response.text())[:1500]
                        )
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if value == "[DONE]":
                            break
                        d = json.loads(value)
                        if "error" in d:
                            raise RuntimeError(str(d["error"]))
                        if d.get("usage"):
                            row["usage"] = d["usage"]
                        for c in d.get("choices", []):
                            ids = c.get("token_ids") or []
                            if c.get("finish_reason") in ["error", "abort"]:
                                raise RuntimeError(str(c))
                            if ids:
                                events.append(
                                    {"time": time.perf_counter(), "tokens": len(ids)}
                                )
                            if c.get("finish_reason"):
                                row["finish_reason"] = c["finish_reason"]
            except Exception as e:
                row["error"] = repr(e)
            row["end"] = time.perf_counter()
            row["tokens"] = sum(e["tokens"] for e in events)
            if events:
                row["first"] = events[0]["time"]
                row["last"] = events[-1]["time"]
                row["ttft_s"] = row["first"] - start
                if row["tokens"] > events[0]["tokens"]:
                    row["tpot_ms"] = (
                        (row["last"] - row["first"])
                        * 1000
                        / (row["tokens"] - events[0]["tokens"])
                    )
            (root / f"request-{i}.json").write_text(
                json.dumps({"summary": row, "events": events})
            )
            print(
                json.dumps(
                    {"completed": i, **{k: v for k, v in row.items() if k != "usage"}}
                ),
                flush=True,
            )
            return row

        tasks = [
            asyncio.create_task(one(i, *pair)) for i, pair in enumerate(assignments)
        ]
        started = time.perf_counter()
        ready.set()
        rows = await asyncio.gather(*tasks)
        await asyncio.sleep(2)
        after = await metrics()
    result = {
        "run_id": run_id,
        "layout": LAYOUT,
        "global_seqs": GLOBAL_SEQS,
        "engines": len(engines),
        "parameters": vars(a),
        "requests": rows,
        "metrics_before": before,
        "metrics_after": after,
        "elapsed_seconds": time.perf_counter() - started,
        "wall_end": time.time(),
        "all_exact_output": all(
            r["tokens"] == a.output and "error" not in r for r in rows
        ),
    }
    tpots = [r["tpot_ms"] for r in rows if "tpot_ms" in r]
    result["full_request_mean_tpot_ms"] = statistics.mean(tpots) if tpots else None
    ttfts = [r["ttft_s"] for r in rows if "ttft_s" in r]
    result["ttft_s"] = (
        {"min": min(ttfts), "max": max(ttfts), "mean": statistics.mean(ttfts)}
        if ttfts
        else None
    )
    cached = [
        r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
        for r in rows
    ]
    result["cached_tokens_hist"] = {str(c): cached.count(c) for c in set(cached)}
    if all("first" in r for r in rows):
        lo = max(r["first"] for r in rows) + 2
        hi = min(r["last"] for r in rows) - 2
        if hi > lo:
            counts = [
                sum(e["tokens"] for e in events if lo < e["time"] <= hi)
                for events in all_events
            ]
            total = sum(counts)
            tp = [1000 * (hi - lo) / n for n in counts if n]
            result["steady"] = {
                "seconds": hi - lo,
                "tokens": total,
                "tokens_per_second": total / (hi - lo),
                "amortized_tpot_ms": len(rows) * 1000 * (hi - lo) / total,
                "mean_request_tpot_ms": statistics.mean(tp),
                "p95_request_tpot_ms": percentile(tp, 0.95),
                "all_requests_progressed": all(counts),
                "min_request_tokens": min(counts),
                "max_request_tokens": max(counts),
            }
    # spec-decode acceptance from the counters, when the server exposes them
    accept = {}
    for host, rows_after in after.items():
        if not isinstance(rows_after, dict) or "error" in rows_after:
            continue
        for name, v in rows_after.items():
            if "spec_decode" in name:
                accept[f"{host} {name}"] = v - before.get(host, {}).get(name, 0.0)
    result["spec_decode_counter_deltas"] = accept
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    (ROOT / (a.stage + ".summary.json")).write_text(json.dumps(result, indent=2))
    print(
        "SUMMARY",
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["requests", "metrics_before", "metrics_after"]
            }
        ),
        flush=True,
    )
    assert result["all_exact_output"], "Invalid output count or request error"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--input", type=int, default=131072)
    p.add_argument("--output", type=int, default=1024)
    p.add_argument("--arrival-interval", type=float, default=0.25)
    asyncio.run(run(p.parse_args()))
