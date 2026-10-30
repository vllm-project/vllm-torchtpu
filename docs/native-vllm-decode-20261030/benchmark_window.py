# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import re
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

import aiohttp
import msgspec
import zmq

R = Path(os.environ.get("NATIVE_BENCH_OUTPUT_DIR", "./benchmark-results"))
R.mkdir(parents=True, exist_ok=True)
hosts = os.environ["NATIVE_HOST_IPS"].split(",")
stage = "normal-window-r1"
run_id = stage + "-" + uuid.uuid4().hex[:8]
root = R / "benchmarks" / run_id
root.mkdir(parents=True)


def monitor():
    ports = []
    for line in subprocess.run(
        ["ss", "-ltnp"], capture_output=True, text=True
    ).stdout.splitlines():
        if "VLLM::DPCoordin" in line:
            fd = re.search(r"fd=(\d+)", line)
            if fd:
                ports.append((int(fd[1]), line.split()[3]))
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.connect("tcp://" + min(ports)[1])
    with (R / (stage + ".coordinator.jsonl")).open("x") as f:
        while not (R / (stage + ".done")).exists():
            if s.poll(1000):
                f.write(
                    json.dumps(
                        {"time": time.time(), "stats": msgspec.msgpack.decode(s.recv())}
                    )
                    + "\n"
                )
                f.flush()
    s.close()


threading.Thread(target=monitor, daemon=True).start()


async def main():
    data = json.loads(Path(os.environ["NATIVE_BENCH_DATASET"]).read_text())
    rows = [
        {"index": i, "dp": i // 12, "slot": i % 12, "events": []} for i in range(96)
    ]
    lo = None
    start = time.time()
    async with aiohttp.ClientSession(
        read_bufsize=16 << 20,
        timeout=aiohttp.ClientTimeout(total=7200, sock_read=7200),
        connector=aiohttp.TCPConnector(limit=128),
    ) as client:

        async def one(row):
            dp = row["dp"]
            slot = row["slot"]
            item = data[dp * 16 + slot]
            suffix = item["suffix_ids"]
            prompt = item["token_ids"][: 131072 - len(suffix)] + suffix
            await asyncio.sleep((row["index"] % 24) * 0.25)
            row["start"] = time.time()
            payload = {
                "model": "qwen38-native-dummy",
                "prompt": prompt,
                "max_tokens": 8192,
                "min_tokens": 8192,
                "ignore_eos": True,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
            }
            try:
                async with client.post(
                    "http://" + hosts[dp // 2] + ":8000/v1/completions",
                    json=payload,
                    headers={
                        "X-data-parallel-rank": str(dp % 2),
                        "X-Request-Id": run_id + "-" + str(row["index"]),
                    },
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            str(response.status) + (await response.text())[:1000]
                        )
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if value == "[DONE]":
                            break
                        v = json.loads(value)
                        if "error" in v:
                            raise RuntimeError(str(v["error"]))
                        if v.get("usage"):
                            row["usage"] = v["usage"]
                        for c in v.get("choices", []):
                            ids = c.get("token_ids") or []
                            if c.get("finish_reason") in ["error", "abort"]:
                                raise RuntimeError(str(c))
                            if ids:
                                assert min(ids) >= 0
                                row["events"].append(
                                    {"time": time.time(), "tokens": len(ids)}
                                )
            except asyncio.CancelledError:
                row["intentional_stop"] = True
                raise
            except Exception as e:
                row["error"] = repr(e)
            finally:
                row["end"] = time.time()

        tasks = [asyncio.create_task(one(r)) for r in rows]
        while True:
            await asyncio.sleep(5)
            now = time.time()
            active = sum(bool(r["events"]) for r in rows)
            if any("error" in r for r in rows):
                raise RuntimeError([r for r in rows if "error" in r])
            if any(t.done() for t in tasks):
                raise RuntimeError(
                    "Request finished before measurement window completed"
                )
            if active == 96 and lo is None:
                lo = now + 2
                print("ALL96_DECODE", lo, flush=True)
            counts = [
                sum(
                    e["tokens"]
                    for e in r["events"]
                    if lo is not None and lo < e["time"] <= now
                )
                for r in rows
            ]
            state = {
                "time": now,
                "elapsed": now - start,
                "streaming_requests": active,
                "window_start": lo,
                "min_window_tokens": min(counts),
                "max_window_tokens": max(counts),
                "total_generated": sum(
                    sum(e["tokens"] for e in r["events"]) for r in rows
                ),
            }
            (R / (stage + ".progress.json")).write_text(json.dumps(state))
            print("PROGRESS", json.dumps(state), flush=True)
            if lo is not None and min(counts) >= 1024:
                hi = now
                break
            if now - start > 7000:
                raise TimeoutError("No steady window")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    seconds = hi - lo
    total = sum(counts)
    tpots = [seconds * 1000 / n for n in counts]
    contexts = [
        131072 + sum(e["tokens"] for e in r["events"] if e["time"] <= lo) for r in rows
    ]
    result = {
        "run_id": run_id,
        "workload": {
            "input_tokens": 131072,
            "generation_cap": 8192,
            "measurement_min_output_tokens_per_request": 1024,
            "global_concurrency": 96,
            "per_dp_concurrency": 12,
            "mtp_k": 3,
            "temperature": 0,
            "load_format": "dummy",
            "moe": "native",
            "entrypoint": "vllm serve",
        },
        "wall_window": [lo, hi],
        "steady": {
            "seconds": seconds,
            "tokens": total,
            "tokens_per_second": total / seconds,
            "mean_request_tpot_ms": statistics.mean(tpots),
            "amortized_tpot_ms": seconds * 96000 / total,
            "min_request_tokens": min(counts),
            "max_request_tokens": max(counts),
        },
        "context_at_window_start": {"min": min(contexts), "max": max(contexts)},
        "requests": [
            {k: v for k, v in r.items() if k != "events"}
            | {
                "window_tokens": counts[i],
                "generated_tokens": sum(e["tokens"] for e in r["events"]),
            }
            for i, r in enumerate(rows)
        ],
        "intentional_stop_after_window": True,
    }
    for r in rows:
        (root / ("request-" + str(r["index"]) + ".json")).write_text(json.dumps(r))
    (R / (stage + ".summary.json")).write_text(json.dumps(result, indent=2))
    print(
        "SUMMARY",
        json.dumps({k: v for k, v in result.items() if k != "requests"}),
        flush=True,
    )
    (R / (stage + ".done")).write_text(str(time.time()))


asyncio.run(main())
