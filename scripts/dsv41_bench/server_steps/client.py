# SPDX-License-Identifier: Apache-2.0
"""Construct fixed workloads; throughput comes exclusively from worker records."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families
from scenarios import SCENARIOS
from summarize import rank_results, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v41")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Separate diagnostic run; do not pool with formal rounds",
    )
    parser.add_argument("--admission-settle-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.admission_settle_seconds <= 0:
        parser.error("admission settle time must be positive")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    scenario = SCENARIOS[args.scenario]
    input_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if input_sha != "865b183381ae43367214d9dd0870f94c578b3d581e613ddc7872d1e48f391a11":
        raise ValueError("Input differs from the frozen DeepSeek natural-code corpus")
    data = json.loads(args.input.read_text())
    rows = data["prompts"][: scenario.concurrency]
    suffixes = data["prime_suffix"][: scenario.concurrency]
    length = scenario.prefix_tokens + scenario.extend_tokens
    if len(rows) != scenario.concurrency or len(suffixes) != len(rows):
        raise ValueError("Input does not have enough independent requests")
    if any(
        len(r) < length or any(type(t) is not int or t < 0 for t in r) for r in rows
    ):
        raise ValueError("Invalid or short token-ID row")
    if len({tuple(r[: scenario.prefix_tokens]) for r in rows}) != len(rows):
        raise ValueError("Expected distinct cached prefixes")
    if any(
        len(s) != 2 or s[0] == r[scenario.prefix_tokens]
        for r, s in zip(rows, suffixes, strict=True)
    ):
        raise ValueError("Prime must branch away from the measured extension")
    primes = [
        r[: scenario.prefix_tokens] + s for r, s in zip(rows, suffixes, strict=True)
    ]
    prompts = [r[:length] for r in rows]
    manifest = {
        "scenario": scenario.to_dict(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "corpus": data["manifest"],
        "rounds": args.rounds,
        "admission_settle_seconds": args.admission_settle_seconds,
        "rank_affinity": [i % scenario.dp_size for i in range(len(rows))],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def post(path: str, payload: dict) -> dict:
        print(json.dumps({"event": "control_begin", "path": path}), flush=True)
        response = requests.post(args.base_url + path, json=payload, timeout=240)
        response.raise_for_status()
        print(json.dumps({"event": "control_end", "path": path}), flush=True)
        return response.json() if response.content else {}

    def rpc(method: str, **kwargs) -> dict:
        return post(
            "/collective_rpc", {"method": method, "kwargs": kwargs, "timeout": 180}
        )

    def metrics() -> dict:
        response = requests.get(args.base_url + "/metrics", timeout=30)
        response.raise_for_status()
        totals = {}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                totals[sample.name] = totals.get(sample.name, 0) + sample.value
        return totals

    def wait_metrics(before: dict) -> dict:
        key = "vllm:request_prefill_time_seconds_count"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            after = metrics()
            completed = after.get(key, 0) - before.get(key, 0)
            if completed == scenario.concurrency:
                return after
            if completed > scenario.concurrency:
                raise ValueError("Unrelated requests contaminated the benchmark")
            time.sleep(0.1)
        raise ValueError("Timed out waiting for complete per-request server metrics")

    def wave(inputs: list, outputs: int) -> list:
        barrier = threading.Barrier(len(inputs) + 1)

        def request(index: int) -> dict:
            payload = {
                "model": args.model,
                "prompt": inputs[index],
                "max_tokens": outputs,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            }
            barrier.wait(timeout=120)
            response = requests.post(
                args.base_url + "/v1/completions",
                json=payload,
                headers={"X-data-parallel-rank": str(index % scenario.dp_size)},
                timeout=3600,
            )
            response.raise_for_status()
            result = response.json()
            usage = result["usage"]
            if (
                usage["prompt_tokens"] != len(inputs[index])
                or usage["completion_tokens"] != outputs
            ):
                raise ValueError(f"Incomplete request {index}: {usage}")
            return result

        # Freeze scheduling while HTTP handlers enqueue this wave. The settling
        # delay is not proof of admission: strict worker shape/context checks
        # below still reject a wave whose ranks did not start together.
        post("/pause?mode=keep&clear_cache=false", {})
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs)) as pool:
            futures = [pool.submit(request, i) for i in range(len(inputs))]
            try:
                barrier.wait(timeout=120)
                time.sleep(args.admission_settle_seconds)
            finally:
                post("/resume", {})
            return [future.result() for future in futures]

    for round_id in range(args.rounds + 1):
        label = "warmup" if round_id == 0 else f"round-{round_id}"
        post("/reset_prefix_cache", {})
        prime_before = metrics()
        wave(primes, 1)
        before = wait_metrics(prime_before)
        # Drain asynchronous/idle work before entering cross-DP utility RPC.
        post("/pause?mode=keep&clear_cache=false", {})
        measurement_start_ns = time.time_ns()
        print(
            json.dumps(
                {
                    "event": "measurement_begin",
                    "round": label,
                    "time_ns": measurement_start_ns,
                }
            ),
            flush=True,
        )
        setup = rpc("benchmark_begin", scenario=args.scenario)
        if sorted(r["rank"] for r in rank_results(setup["results"])) != list(
            range(scenario.dp_size)
        ):
            raise ValueError("RPC did not configure all eight DP workers")
        profiling = args.profile and round_id > 0
        if profiling:
            post("/start_profile", {})
        try:
            responses = wave(prompts, scenario.output_tokens)
        finally:
            post("/pause?mode=keep&clear_cache=false", {})
            if profiling:
                post("/stop_profile", {})
            records = rpc("benchmark_end")
            (args.out / f"{label}-steps.json").write_text(json.dumps(records))
            post("/resume", {})
        measurement_end_ns = time.time_ns()
        print(
            json.dumps(
                {
                    "event": "measurement_end",
                    "round": label,
                    "time_ns": measurement_end_ns,
                }
            ),
            flush=True,
        )
        after = wait_metrics(before)
        (args.out / f"{label}-responses.json").write_text(json.dumps(responses))
        delta = {k: after[k] - before.get(k, 0) for k in after if k.endswith("_total")}
        (args.out / f"{label}-metrics.json").write_text(json.dumps(delta, indent=2))
        if (
            "vllm:num_preemptions_total" not in delta
            or delta["vllm:num_preemptions_total"] != 0
        ):
            raise ValueError("Missing preemption metric or nonzero preemptions")
        hits = delta.get("vllm:prefix_cache_hits_total", -1)
        # Hybrid compressed KV groups align cache hits to 256 logical tokens,
        # despite --block-size 16. An exact-prefix prompt leaves its last hash
        # unit uncached; an extension can reuse the complete 64k prefix.
        expected_hit = scenario.prefix_tokens - (0 if scenario.extend_tokens else 256)
        if hits != scenario.concurrency * expected_hit:
            raise ValueError(f"Prefix-hit invariant failed: {hits}")
        report = summarize(records["results"], args.scenario)
        report["warmup"] = round_id == 0
        report["profiled"] = profiling
        report["audit_wall_interval_ns"] = [measurement_start_ns, measurement_end_ns]
        (args.out / f"{label}-summary.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "round": label,
                    "categories": {
                        k: {a: b for a, b in v.items() if a != "samples"}
                        for k, v in report["categories"].items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
