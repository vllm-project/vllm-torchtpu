# SPDX-License-Identifier: Apache-2.0
"""LongBench independent-prefix prime/extend waves with explicit DP affinity."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import threading
import time
import urllib.request

import requests
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

DATA_URL = "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json"
DATA_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v41")
    parser.add_argument("--tokenizer", default="/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--prefix", type=int, default=65536)
    parser.add_argument("--extend", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if min(args.concurrency, args.dp_size, args.prefix, args.extend, args.rounds) < 1:
        parser.error("counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.dataset.exists():
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, args.dataset)
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"Unexpected dataset SHA256: {digest}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    documents = sorted(
        (
            d
            for d in json.loads(args.dataset.read_text())
            if d.get("domain") == "Code Repository Understanding"
        ),
        key=lambda d: d["_id"],
    )
    rows, doc_ids, hashes = [], [], set()
    for document in documents:
        ids = tokenizer.encode(document["context"][:1000000], add_special_tokens=False)
        if len(ids) < args.prefix + args.extend:
            continue
        ids = ids[: args.prefix + args.extend]
        prefix_hash = hashlib.sha256(
            json.dumps(ids[: args.prefix]).encode()
        ).hexdigest()
        if prefix_hash in hashes:
            continue
        rows.append(ids)
        hashes.add(prefix_hash)
        doc_ids.append(document["_id"])
        if len(rows) == args.concurrency:
            break
    if len(rows) != args.concurrency:
        raise ValueError("Not enough distinct LongBench prefixes for this concurrency")
    primes = []
    for ids in rows:
        # Valid text tokens prevent incomplete UTF-8 prompt tails leaking into SSE.
        candidates = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in (" the", " a", " and")
        ]
        branch = next(
            x[0] for x in candidates if len(x) == 1 and x[0] != ids[args.prefix]
        )
        primes.append(ids[: args.prefix] + [branch, branch])
    manifest = {
        **vars(args),
        "dataset_sha256": digest,
        "doc_ids": doc_ids,
        "prefix_sha256": sorted(hashes),
        "input_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "ranks": [i % args.dp_size for i in range(args.concurrency)],
    }
    (args.out / "input.json").write_text(json.dumps(rows))
    (args.out / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))

    def metrics():
        response = requests.get(args.base_url + "/metrics", timeout=30)
        response.raise_for_status()
        values = {}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                values[sample.name] = values.get(sample.name, 0) + sample.value
        return values, response.text

    def wave(prompts, label):
        barrier = threading.Barrier(args.concurrency)

        def request(index):
            barrier.wait(timeout=60)
            start = time.perf_counter()
            first, usage, chunks = None, None, []
            with requests.post(
                args.base_url + "/v1/completions",
                headers={"X-data-parallel-rank": str(index % args.dp_size)},
                json={
                    "model": args.model,
                    "prompt": prompts[index],
                    "max_tokens": 1,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                stream=True,
                timeout=1800,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=4096):
                    if line == b"data: [DONE]":
                        break
                    if not line.startswith(b"data: "):
                        continue
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
                    if chunk.get("choices") and first is None:
                        choice = chunk["choices"][0]
                        if choice.get("text") or choice.get("finish_reason"):
                            first = time.perf_counter()
                    if chunk.get("usage"):
                        usage = chunk["usage"]
            if first is None or usage is None or usage["completion_tokens"] != 1:
                raise ValueError(f"Incomplete response: {chunks}")
            if usage["prompt_tokens"] != len(prompts[index]):
                raise ValueError(f"Unexpected token count: {usage}")
            (args.out / f"{label}-{index}-response.json").write_text(json.dumps(chunks))
            return {
                "rank": index % args.dp_size,
                "ttft_s": first - start,
                "wall_s": time.perf_counter() - start,
                "usage": usage,
            }

        before, raw = metrics()
        (args.out / f"{label}-before.prom").write_text(raw)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(request, range(args.concurrency)))
        wall = time.perf_counter() - start
        for _ in range(60):
            after, raw = metrics()
            count = after.get(
                "vllm:request_prefill_time_seconds_count", 0
            ) - before.get("vllm:request_prefill_time_seconds_count", 0)
            if count >= args.concurrency:
                break
            time.sleep(1)
        if count != args.concurrency:
            raise ValueError(f"Unexpected completed request count: {count}")
        (args.out / f"{label}-after.prom").write_text(raw)
        delta = {
            k: after[k] - before.get(k, 0)
            for k in after
            if k.endswith(("_sum", "_count", "_total"))
        }
        result = {
            "label": label,
            "requests": results,
            "wall_s": wall,
            "server_prefill_mean_s": delta["vllm:request_prefill_time_seconds_sum"]
            / count,
            "metrics_delta": delta,
        }
        (args.out / f"{label}.json").write_text(json.dumps(result, indent=2))
        return result

    results = []
    for iteration in range(args.rounds + 1):
        # Reuse the same independent documents, but never measure a cached extend.
        response = requests.post(args.base_url + "/reset_prefix_cache", timeout=180)
        response.raise_for_status()
        prime = wave(primes, f"round-{iteration}-prime")
        extend = wave(rows, f"round-{iteration}-extend")
        hits = extend["metrics_delta"].get("vllm:prefix_cache_hits_total")
        if hits != args.prefix * args.concurrency:
            raise ValueError(f"Prefix affinity/hit mismatch: {hits}")
        extend["new_tokens_per_s"] = args.extend * args.concurrency / extend["wall_s"]
        results.append({"warmup": iteration == 0, "prime": prime, "extend": extend})
        (args.out / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"round": iteration, "extend": extend}), flush=True)
    measured = [x["extend"] for x in results[1:]]
    summary = {
        "server_extend_mean_s_median": statistics.median(
            x["server_prefill_mean_s"] for x in measured
        ),
        "new_tokens_per_s_median": statistics.median(
            x["new_tokens_per_s"] for x in measured
        ),
        "note": "Server prefill includes scheduling/replay, not just device time.",
        "runs": measured,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
