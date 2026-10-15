# SPDX-License-Identifier: Apache-2.0
"""Prepare, measure and audit the fixed DP8 cached-extend benchmark."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families
from server_step_worker import SCENARIO, SCENARIOS, token_counts

DATA_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
INPUT_SHA256 = "9ca4086c68c610b9205e1e46e1a51d3d18bb10495ed88d341c63b725ccca4064"


def load_frozen_input(path: Path) -> dict:
    data = json.loads(path.read_text())
    # Match the frozen corpus while allowing a different tokenizer mount path.
    frozen = {
        **data,
        "manifest": {
            **data["manifest"],
            "tokenizer": "/models/DeepSeek-V4.1-Flash",
        },
    }
    if hashlib.sha256(json.dumps(frozen).encode()).hexdigest() != INPUT_SHA256:
        raise ValueError("Input differs from the frozen DeepSeek natural-code corpus")
    return data


def prepare_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="encoder_extend")
    args = parser.parse_args(argv)
    scenario = SCENARIOS[args.scenario]
    count = scenario.concurrency if args.scenario == "decode" else 128
    prefix = scenario.prefix_tokens
    row_tokens = max(prefix + scenario.extend_tokens, prefix + 1)
    from transformers import AutoTokenizer

    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"Unexpected LongBench dataset: {digest}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    branches = [
        tokenizer.encode(s, add_special_tokens=False) for s in (" the", " a", " and")
    ]
    branches = [s[0] for s in branches if len(s) == 1]
    if len(set(branches)) < 2:
        raise ValueError("Need two distinct single-token prime suffixes")
    documents = sorted(json.loads(args.dataset.read_text()), key=lambda d: d["_id"])
    rows, sources, suffixes, hashes = [], [], [], set()
    windows_per_document = count if args.scenario == "decode" else 16
    for doc in documents:
        if doc.get("domain") != "Code Repository Understanding":
            continue
        ids = tokenizer.encode(doc["context"][:1000000], add_special_tokens=False)
        # Bounded, deterministic windows retain real code continuation statistics.
        for offset in range(
            0, min(max(0, len(ids) - row_tokens + 1), windows_per_document * 256), 256
        ):
            row = ids[offset : offset + row_tokens]
            key = hashlib.sha256(json.dumps(row[:prefix]).encode()).hexdigest()
            if key in hashes:
                continue
            hashes.add(key)
            rows.append(row)
            sources.append(
                {"document": doc["_id"], "offset": offset, "prefix_sha256": key}
            )
            branch = next(t for t in branches if t != row[prefix])
            suffixes.append([branch, branch])
            if len(rows) == count:
                break
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(
            f"Need {count} distinct long-context windows; found {len(rows)}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(
            {
                "prompts": rows,
                "prime_suffix": suffixes,
                "manifest": {
                    "dataset_sha256": digest,
                    "tokenizer": args.tokenizer,
                    "sources": sources,
                },
            },
            stream,
        )


def summarize_extend(ranks: list[dict]) -> dict:
    scenario = SCENARIO
    prompt_tokens = scenario.prefix_tokens + scenario.extend_tokens
    if sorted(r["rank"] for r in ranks) != list(range(8)):
        raise ValueError("Expected exactly eight distinct DP ranks")
    if len({r["hostname"] for r in ranks}) != 1:
        raise ValueError("Monotonic timestamps require one host")
    ranks = sorted(ranks, key=lambda r: r["rank"])
    waves = []
    for rank in ranks:
        if rank["scenario"] != scenario.to_dict():
            raise ValueError("Worker scenario mismatch")
        steps = []
        previous_end = None
        for step_index, step in enumerate(rank["steps"]):
            if step.get("observation") != "native_output_consumption":
                raise ValueError(
                    "Requires naturally observed outputs, not old sync data"
                )
            if step["observed_ns"] <= step["start_ns"]:
                raise ValueError("Nonpositive observation interval")
            encoders = [s for s in step["stages"] if s["stage"] == "encoder"]
            if not encoders:
                continue
            if len(encoders) != 1:
                raise ValueError("Multi-forward step cannot qualify as extend")
            row = encoders[0]
            if row["phase"] != "prefill":
                continue
            if row["requests"] != 1 or row["prompts"] != [prompt_tokens]:
                raise ValueError(f"Expected local C1 and {prompt_tokens}-token prompt")
            start, end = row["starts"][0], row["ends"][0]
            if previous_end is not None and previous_end != start:
                raise ValueError("Nonconsecutive context: missing or replayed chunk")
            if not 0 <= start < end <= prompt_tokens:
                raise ValueError("Invalid context")
            counts = token_counts([start], [end], scenario.prefix_tokens)
            if any(row[k] != v for k, v in counts.items()):
                raise ValueError("Token counters disagree with actual context")
            if row["rows"] != end - start or row["padded_rows"] < row["rows"]:
                raise ValueError("Invalid actual/padded rows")
            if previous_end is None:
                category = "encoder_first"
                if start > scenario.prefix_tokens or end <= scenario.prefix_tokens:
                    raise ValueError(
                        "First chunk must cross the cached-prefix boundary"
                    )
            elif end == prompt_tokens:
                category = "encoder_tail"
            else:
                category = "encoder_middle"
                if (row["rows"], row["padded_rows"], row["replay_tokens"]) != (
                    2044,
                    2048,
                    0,
                ):
                    raise ValueError("Middle requires actual2044/pad2048 and no replay")
                if any(s["stage"] == "decoder" for s in step["stages"]):
                    raise ValueError("Decoder executed in a middle chunk")
            if steps and step_index != steps[-1]["step_index"] + 1:
                raise ValueError("Idle or mixed step interrupted the extend window")
            previous_end = end
            steps.append(
                {**step, "encoder": row, "category": category, "step_index": step_index}
            )
        if not steps or previous_end != prompt_tokens:
            raise ValueError("Incomplete extend wave")
        if sum(s["encoder"]["new_tokens"] for s in steps) != 16384:
            raise ValueError("Wave must account for exactly 16384 new tokens per rank")
        waves.append(steps)
    if len({len(w) for w in waves}) != 1:
        raise ValueError("Ranks executed different chunk counts")
    categories = {}
    previous = None
    for index, group in enumerate(zip(*waves, strict=True)):
        if (
            len(
                {
                    (
                        s["category"],
                        tuple(s["encoder"]["starts"]),
                        tuple(s["encoder"]["ends"]),
                    )
                    for s in group
                }
            )
            != 1
        ):
            raise ValueError("Ranks have different contexts or chunk categories")
        category = group[0]["category"]
        observed = max(s["observed_ns"] for s in group)
        # An observation-to-observation interval counts the CURRENT chunk only.
        # First chunk uses entry-to-observation and is reference-only.
        begin = min(s["start_ns"] for s in group) if previous is None else previous
        if observed <= begin:
            raise ValueError("Observations are not monotonically ordered")
        sample = {
            "chunk_index": index,
            "seconds": (observed - begin) / 1e9,
            "effective_tokens": sum(s["encoder"]["new_tokens"] for s in group),
            "start_ns": begin,
            "end_ns": observed,
            "rank_observed_ns": [s["observed_ns"] for s in group],
            "observation_skew_ms": (observed - min(s["observed_ns"] for s in group))
            / 1e6,
            "contexts": [s["encoder"]["starts"] for s in group],
            "rows": [s["encoder"]["rows"] for s in group],
            "padded_rows": [s["encoder"]["padded_rows"] for s in group],
            "observation_paths": [s["output_path"] for s in group],
        }
        categories.setdefault(category, []).append(sample)
        previous = observed
    if len(categories.get("encoder_middle", [])) < 2:
        raise ValueError("Need consecutive full-DP middle chunks")
    report = {
        "scenario": scenario.to_dict(),
        "timing": "natural_server_output_host_cadence",
        "device_seconds": None,
        "capacity_qualified": False,
        "limitation": (
            "Intermediate output may not wait for device completion; "
            "host cadence is not isolated device or D-node capacity."
        ),
        "categories": {},
    }
    for category, samples in categories.items():
        seconds = sum(s["seconds"] for s in samples)
        tokens = sum(s["effective_tokens"] for s in samples)
        rate = tokens / seconds
        report["categories"][category] = {
            "steps": len(samples),
            "median_step_ms": statistics.median(s["seconds"] for s in samples) * 1000,
            "total_seconds": seconds,
            "effective_tokens": tokens,
            "global_tokens_s": rate,
            "per_device_tokens_s": rate / 8,
            "per_physical_chip_tokens_s": rate / 4,
            "token_kind": "new_input",
            "samples": samples,
        }
    return report


def rank_results(value: list) -> list[dict]:
    result = []
    for item in value:
        if isinstance(item, list):
            result.extend(rank_results(item))
        else:
            result.append(item)
    return result


def summarize_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize_extend(
        rank_results(json.loads(args.input.read_text())["results"])
    )
    args.out.write_text(json.dumps(result, indent=2))


def assert_no_compilation(log: str, intervals: list[list[int]]) -> None:
    """Reject compilation in formal windows, using the server's local clock."""
    for line in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if not any(
            phrase in line
            for phrase in (
                "Compiling ",
                "Starting torch.compile",
                "Dynamo bytecode",
                "built program",
            )
        ):
            continue
        stamp = re.search(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b", line)
        if stamp is None:
            raise ValueError(f"Cannot timestamp compiler log event: {line}")
        for low, high in intervals:
            # Logs omit the year; retain windows crossing New Year's midnight.
            years = {
                datetime.datetime.fromtimestamp(low / 1e9 - 1).year,
                datetime.datetime.fromtimestamp(high / 1e9 + 1).year,
            }
            for candidate_year in years:
                event = (
                    datetime.datetime.strptime(
                        f"{candidate_year}-{stamp[1]}", "%Y-%m-%d %H:%M:%S"
                    ).timestamp()
                    * 1e9
                )
                if low - 1e9 <= event <= high + 1e9:
                    raise ValueError(f"Compilation in formal serving window: {line}")


def summarize_startup(log: str) -> dict:
    ranks: dict[int, dict] = {}
    for line in log.splitlines():
        rank = re.search(r"(?:Worker|EngineCore)_DP(\d+)", line)
        if not rank:
            continue
        row = ranks.setdefault(int(rank[1]), {"stage_warmup_seconds": {}})
        engine = re.search(
            r"init engine .* took ([\d.]+) s \(compilation: ([\d.]+) s\)", line
        )
        if engine:
            row["engine_init_seconds"] = float(engine[1])
            row["reported_compilation_seconds"] = float(engine[2])
        stage = re.search(
            r"\[(v41_\w+)\] torch.compile and initial profiling/warmup "
            r"run together took ([\d.]+) s",
            line,
        )
        if stage:
            values = row["stage_warmup_seconds"]
            values.setdefault(stage[1], []).append(float(stage[2]))
        for phrase, key in [
            ("Saved compiled artifact", "artifact_saves"),
            ("Loading compiled artifact", "artifact_load_attempts"),
        ]:
            if phrase in line:
                row[key] = row.get(key, 0) + 1
    complete = [r for r in ranks.values() if "engine_init_seconds" in r]
    return {
        "ranks": ranks,
        "complete_ranks": len(complete),
        "max_engine_init_seconds": max(
            (r["engine_init_seconds"] for r in complete), default=None
        ),
        "max_reported_compilation_seconds": max(
            (r["reported_compilation_seconds"] for r in complete), default=None
        ),
        "scope": (
            "Logged compilation includes backend and cache work; it is not pure "
            "XLA compiler time. Stage warmups can overlap other reported scopes."
        ),
    }


def startup_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize_startup(args.server_log.read_text(errors="replace"))
    args.out.write_text(json.dumps(result, indent=2) + "\n")


def client_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args(argv)
    if args.admission_settle_seconds <= 0:
        parser.error("admission settle time must be positive")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    scenario = SCENARIO
    data = load_frozen_input(args.input)
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
        setup = rpc("benchmark_begin", scenario=SCENARIO.name)
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
        # unit uncached; an extension can reuse the complete prefix.
        expected_hit = scenario.prefix_tokens - (0 if scenario.extend_tokens else 256)
        if hits != scenario.concurrency * expected_hit:
            raise ValueError(f"Prefix-hit invariant failed: {hits}")
        report = summarize_extend(rank_results(records["results"]))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "client", "summarize", "startup"]
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    commands = {
        "prepare": prepare_main,
        "client": client_main,
        "summarize": summarize_main,
        "startup": startup_main,
    }
    commands[args.command](remaining)


if __name__ == "__main__":
    main()
