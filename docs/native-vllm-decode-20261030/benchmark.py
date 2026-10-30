# SPDX-License-Identifier: Apache-2.0

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


def percentile(v, q):
    v = sorted(v)
    return v[int((len(v) - 1) * q)] if v else None


async def run(a):
    dataset = json.loads(Path(os.environ["NATIVE_BENCH_DATASET"]).read_text())
    assignments = [(dp, slot + a.slot) for dp in range(8) for slot in range(a.per_dp)]
    run_id = "native-" + uuid.uuid4().hex[:12]
    root = ROOT / "benchmarks" / run_id
    root.mkdir(parents=True)
    (root / "parameters.json").write_text(json.dumps(vars(a), indent=2))
    timeout = aiohttp.ClientTimeout(total=7200, sock_read=7200)
    async with aiohttp.ClientSession(
        read_bufsize=16 << 20,
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=128),
    ) as client:

        async def metrics():
            out = {}
            for host in HOSTS:
                async with client.get(f"http://{host}:8000/metrics") as response:
                    response.raise_for_status()
                    s = await response.text()
                    rows = {}
                    for line in s.splitlines():
                        if line.startswith("#") or not any(
                            k in line
                            for k in [
                                "spec_decode",
                                "prefix_cache",
                                "request_success",
                                "num_preemptions",
                            ]
                        ):
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

        async def one(i, dp, slot):
            item = dataset[dp * 16 + slot]
            suffix = item["suffix_ids"]
            prompt = item["token_ids"][: a.input - len(suffix)] + suffix
            await ready.wait()
            await asyncio.sleep((i % (2 * a.per_dp)) * a.arrival_interval)
            start = time.perf_counter()
            row = {
                "index": i,
                "dp": dp,
                "local_dp": dp % 2,
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
            headers = {
                "X-data-parallel-rank": str(dp % 2),
                "X-Request-Id": f"{run_id}-{i}",
            }
            try:
                async with client.post(
                    f"http://{HOSTS[dp // 2]}:8000/v1/completions",
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
                                assert all(x >= 0 for x in ids)
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
                if row["tokens"] > events[0]["tokens"]:
                    row["tpot_ms"] = (
                        (row["last"] - row["first"])
                        * 1000
                        / (row["tokens"] - events[0]["tokens"])
                    )
            (root / f"request-{i}.json").write_text(
                json.dumps({"summary": row, "events": events})
            )
            print(json.dumps({"completed": i, **row}), flush=True)
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
    if all("first" in r for r in rows):
        lo = max(r["first"] for r in rows) + 2
        hi = min(r["last"] for r in rows) - 2
        if hi > lo:
            counts = [
                sum(e["tokens"] for e in events if lo < e["time"] <= hi)
                for events in all_events
            ]
            total = sum(counts)
            tpots = [1000 * (hi - lo) / n for n in counts if n]
            result["steady"] = {
                "seconds": hi - lo,
                "tokens": total,
                "tokens_per_second": total / (hi - lo),
                "amortized_tpot_ms": len(rows) * 1000 * (hi - lo) / total,
                "mean_request_tpot_ms": statistics.mean(tpots),
                "p95_request_tpot_ms": percentile(tpots, 0.95),
                "all_requests_progressed": all(counts),
                "min_request_tokens": min(counts),
                "max_request_tokens": max(counts),
            }
            offset = statistics.median(r["wall_start"] - r["start"] for r in rows)
            result["wall_window"] = [lo + offset, hi + offset]
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
    p.add_argument("--per-dp", type=int, default=12)
    p.add_argument("--slot", type=int, default=0)
    p.add_argument("--input", type=int, default=131072)
    p.add_argument("--output", type=int, default=1024)
    p.add_argument("--arrival-interval", type=float, default=0.25)
    asyncio.run(run(p.parse_args()))
