# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Construct fixed workloads and summarize worker observations."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families
from server_step_worker import SCENARIOS

# Hold accepted tokens constant so the primary metric compares step cadence.
_FIXED_ACCEPTED_TOKENS_PER_REQUEST_STEP = 3.47


def wait_for_admission(
    server_log: Path,
    offset: int,
    request_ids: set[str],
    futures: list[concurrent.futures.Future],
    timeout: float = 120,
) -> None:
    """Wait for pinned vLLM's post-send acknowledgements while paused."""
    added = re.compile(r"Added request (\S+)-[0-9a-f]{8}\.")
    pending = set(request_ids)
    deadline = time.monotonic() + timeout
    with server_log.open() as stream:
        stream.seek(offset)
        while pending:
            for future in futures:
                if future.done():
                    future.result()
                    raise RuntimeError("Request completed while scheduler was paused")
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line.endswith("\n"):
                    stream.seek(position)
                    break
                if match := added.search(line):
                    pending.discard(match[1])
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Admission timed out: {len(pending)}/{len(request_ids)} "
                    "requests missing vLLM Added request acknowledgements; "
                    "enable --enable-log-requests on the API server"
                )
            time.sleep(0.1)


def expected_prefix_hit(prefix_tokens: int, extend_tokens: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("Prefix-hit alignment must be positive")
    # Exact-prefix requests must compute at least their final token. Extensions
    # may reuse the whole aligned prefix; neither may assume a fixed page size.
    limit = prefix_tokens if extend_tokens else prefix_tokens - 1
    return limit // alignment * alignment


def client_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=["decode", "decode_dp16"], default="decode"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v41")
    parser.add_argument(
        "--server-log",
        type=Path,
        default=Path(os.environ["OUT"]) / "server.log" if "OUT" in os.environ else None,
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--admission-settle-seconds", type=float, default=5.0)
    parser.add_argument("--single-window", action="store_true")
    parser.add_argument("--reuse-prefix", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args(argv)
    if args.single_window:
        args.rounds = 0
    if args.server_log is None or not args.server_log.is_file():
        parser.error("--server-log is required to audit compilation during rounds")
    if args.admission_settle_seconds <= 0:
        parser.error("admission settle time must be positive")
    if args.rounds < 0:
        parser.error("rounds must be nonnegative; zero runs warmup only")
    args.out.mkdir(parents=True, exist_ok=False)
    scenario = SCENARIOS[args.scenario]
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
        "admission_barrier": "vllm_added_request_log",
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
        wave_id = uuid.uuid4().hex
        request_ids = [f"dsv41-{wave_id}-{i}" for i in range(len(inputs))]

        def request(index: int) -> dict:
            payload = {
                "model": args.model,
                "prompt": inputs[index],
                "max_tokens": outputs,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
                "request_id": request_ids[index],
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

        # Pinned vLLM logs Added request only after sending to the chosen engine.
        # A fixed delay alone can leave long token-ID requests outside the queue.
        post("/pause?mode=keep&clear_cache=false", {})
        admission_offset = args.server_log.stat().st_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs)) as pool:
            futures = [pool.submit(request, i) for i in range(len(inputs))]
            try:
                barrier.wait(timeout=120)
                time.sleep(args.admission_settle_seconds)
                wait_for_admission(
                    args.server_log,
                    admission_offset,
                    {f"cmpl-{request_id}-0" for request_id in request_ids},
                    futures,
                )
            finally:
                post("/resume", {})
            return [future.result() for future in futures]

    if args.profile_only:
        post("/pause?mode=keep&clear_cache=false", {})
        rpc(
            "synchronized_profile_arm",
            output_dir=str(args.out.resolve()),
            scenario=args.scenario,
        )
        try:
            wave(prompts, scenario.output_tokens)
            post("/pause?mode=keep&clear_cache=false", {})
            capture = rpc("synchronized_profile_check")
            (args.out / "synchronized-capture.json").write_text(json.dumps(capture))
        finally:
            post("/resume", {})
        return

    for round_id in range(args.rounds + 1):
        label = (
            "steady"
            if args.single_window
            else ("warmup" if round_id == 0 else f"round-{round_id}")
        )
        if args.reuse_prefix or round_id > 0:
            before = metrics()
        else:
            post("/reset_prefix_cache", {})
            prime_before = metrics()
            wave(primes, 1)
            before = wait_metrics(prime_before)
        # Drain asynchronous/idle work before entering cross-DP utility RPC.
        post("/pause?mode=keep&clear_cache=false", {})
        setup = rpc("benchmark_begin", scenario=args.scenario)
        if sorted(r["rank"] for r in rank_results(setup["results"])) != list(
            range(scenario.dp_size)
        ):
            raise ValueError("RPC did not configure all DP workers")
        log_offset = args.server_log.stat().st_size
        audit_start = time.time_ns()
        try:
            responses = wave(prompts, scenario.output_tokens)
        finally:
            post("/pause?mode=keep&clear_cache=false", {})
            records = rpc("benchmark_end")
            (args.out / f"{label}-steps.json").write_text(json.dumps(records))
            post("/resume", {})
        audit_end = time.time_ns()
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
        # Read finalized cache geometry from every worker, not from observed
        # hit metrics: changing --block-size changes the scheduler alignment.
        alignments = {
            rank["prefix_hit_alignment"] for rank in rank_results(setup["results"])
        }
        if len(alignments) != 1:
            raise ValueError(f"DP ranks disagree on prefix-hit alignment: {alignments}")
        expected_hit = expected_prefix_hit(
            scenario.prefix_tokens, scenario.extend_tokens, alignments.pop()
        )
        if hits != scenario.concurrency * expected_hit:
            raise ValueError(
                f"Prefix-hit invariant failed: {hits}; "
                f"expected {scenario.concurrency * expected_hit}"
            )
        with args.server_log.open() as stream:
            stream.seek(log_offset)
            round_log = stream.read()
        (args.out / f"{label}-server.log").write_text(round_log)
        compile_lines = [
            line
            for line in round_log.splitlines()
            if re.search(
                r"Compiling FX graph|Compiling a graph|"
                r"Compiling.*(?:jit_|XLA)|torch.*recompil|"
                r"Pinned \d+ of \d+ entry layouts|built program \d+",
                line,
                re.IGNORECASE,
            )
        ]
        if (round_id or args.single_window) and compile_lines:
            raise ValueError(
                f"Compilation contaminated formal round: {compile_lines[:3]}"
            )
        report = summarize(records["results"], args.scenario)
        report["audit_wall_interval_ns"] = [audit_start, audit_end]
        report["compilation_log_matches"] = compile_lines
        report["warmup"] = round_id == 0 and not args.single_window
        (args.out / f"{label}-summary.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "round": label,
                    "decode_throughput": report["decode_throughput"],
                    "categories": {
                        k: {
                            a: b
                            for a, b in v.items()
                            if a not in ("samples", "first_20", "last_20")
                        }
                        for k, v in report["categories"].items()
                    },
                }
            ),
            flush=True,
        )


def rank_results(value: list) -> list[dict]:
    result = []
    for item in value:
        result.extend(rank_results(item) if isinstance(item, list) else [item])
    return result


def summarize(results: list, scenario_name: str) -> dict:
    if scenario_name not in ("decode", "decode_dp16"):
        raise NotImplementedError("Only the decode scenario is supported")
    scenario = SCENARIOS[scenario_name]
    ranks = sorted(rank_results(results), key=lambda r: r["rank"])
    if [r["rank"] for r in ranks] != list(range(scenario.dp_size)):
        raise ValueError("Expected one result for every DP rank; no duplicate ranks")
    hosts = {r["hostname"] for r in ranks}
    if scenario_name == "decode" and len(hosts) != 1:
        raise ValueError("Server monotonic timestamps require a single node")
    if scenario_name == "decode_dp16" and (
        len(hosts) != 2
        or any(sum(r["hostname"] == host for r in ranks) != 8 for host in hosts)
    ):
        raise ValueError("DP16 benchmark requires two hosts with eight ranks each")
    host_ranks = {
        host: [r["rank"] for r in ranks if r["hostname"] == host] for host in hosts
    }
    groups = defaultdict(dict)
    for rank in ranks:
        if rank["scenario"] != scenario.to_dict():
            raise ValueError("Worker scenario differs from requested scenario")
        generated = Counter()
        # Immediate empty returns can overtake an earlier asynchronous output.
        # Preserve execution ordinals for shape continuity, but account tokens
        # in the order in which native serving actually observes them.
        observations = sorted(
            enumerate(rank["steps"]), key=lambda item: item[1]["observed_ns"]
        )
        for ordinal, step in observations:
            if step.get("observation") != "native_output_consumption":
                raise ValueError(
                    "Expected natural output consumption; legacy data rejected"
                )
            observed = step["observed_ns"]
            if observed < step["start_ns"]:
                raise ValueError("Output observation precedes dispatch")
            ids, counts = step["output_request_ids"], step["accepted_output_counts"]
            if (
                len(ids) != len(counts)
                or len(set(ids)) != len(ids)
                or len(ids) != step["output_requests"]
                or sum(counts) != step["accepted_output_tokens"]
                or any(type(n) is not int or n < 0 for n in counts)
            ):
                raise ValueError("Missing or inconsistent per-request token counts")
            contexts = [scenario.prefix_tokens + generated[req] for req in ids]
            # Native runner output precedes scheduler length truncation. Reject
            # the entire possible K+1 boundary even when a shorter sample fits.
            untruncated = all(
                1 <= n <= scenario.dspark_tokens + 1
                and generated[req] + scenario.dspark_tokens + 1 < scenario.output_tokens
                for req, n in zip(ids, counts, strict=True)
            )
            for req, n in zip(ids, counts, strict=True):
                generated[req] += n
            enc = [s for s in step["stages"] if s["stage"] == "encoder"]
            dec = [s for s in step["stages"] if s["stage"] == "decoder"]
            if len(enc) != 1 or len(dec) != 1:
                continue
            index = enc[0]["forward_id"]
            if rank["rank"] in groups[index]:
                raise ValueError("Duplicate rank/forward sample")
            valid = untruncated and len(ids) == scenario.local_concurrency
            drafts = [s for s in step["stages"] if s["stage"] == "draft_inputs"]
            valid &= (
                len(drafts) == 1
                and drafts[0]["requests"] == scenario.local_concurrency
                and drafts[0]["rows"]
                == scenario.local_concurrency * scenario.dspark_tokens
                and drafts[0]["padded_rows"] >= drafts[0]["rows"]
                and all(
                    step["scheduled_draft_lengths"].get(req) == scenario.dspark_tokens
                    for req in ids
                )
            )
            for stage in (enc[0], dec[0]):
                valid &= (
                    stage["phase"] == "decode"
                    and stage["requests"] == scenario.local_concurrency
                    and len(stage["starts"]) == scenario.local_concurrency
                    and stage["prompts"]
                    == [scenario.prefix_tokens] * scenario.local_concurrency
                    and stage["rows"]
                    == scenario.local_concurrency * (scenario.dspark_tokens + 1)
                    and stage["padded_rows"] >= stage["rows"]
                )
            valid &= all(
                scenario.prefix_tokens
                < c
                < scenario.prefix_tokens + scenario.output_tokens
                for c in contexts
            )
            groups[index][rank["rank"]] = {
                "ordinal": ordinal,
                "valid": valid,
                "step": step,
                "contexts_from_accepted_outputs": contexts,
            }
    # A segment requires full C on every rank and unchanged request membership.
    # Never close a gap by summing isolated fast steps.
    segments, current = [], []
    previous = None
    for index, group in sorted(groups.items()):
        rows = [group[r] for r in sorted(group)]
        valid = len(rows) == scenario.dp_size and all(r["valid"] for r in rows)
        contiguous = (
            previous is not None
            and index == previous[0] + 1
            and all(
                a["ordinal"] == b["ordinal"] + 1
                and a["step"]["output_request_ids"] == b["step"]["output_request_ids"]
                for a, b in zip(rows, previous[1], strict=True)
            )
            if valid and previous is not None
            else False
        )
        if not valid or not contiguous:
            if current:
                segments.append(current)
            current = []
        if valid:
            current.append((index, rows))
            previous = (index, rows)
        else:
            previous = None
    if current:
        segments.append(current)
    # Keep all sufficiently long segments; first observation is the baseline,
    # and the last is a conservative drain guard. Each numerator covers (a,b].
    samples = []
    windows = []
    for segment in segments:
        if len(segment) < 10:
            continue
        usable = segment[:-1]
        for (prev_index, prev_rows), (index, rows) in zip(usable, usable[1:]):
            host_intervals = {
                host: max(rows[i]["step"]["observed_ns"] for i in indices)
                - max(prev_rows[i]["step"]["observed_ns"] for i in indices)
                for host, indices in host_ranks.items()
            }
            begin, end = 0, max(host_intervals.values())
            if min(host_intervals.values()) <= 0:
                raise ValueError("Non-positive node observation cadence")
            samples.append(
                {
                    "forward_id": index,
                    "previous_forward_id": prev_index,
                    "seconds": (end - begin) / 1e9,
                    "effective_tokens": sum(
                        r["step"]["accepted_output_tokens"] for r in rows
                    ),
                    "per_rank_observed_ns": [r["step"]["observed_ns"] for r in rows],
                    "contexts_from_accepted_outputs": [
                        r["contexts_from_accepted_outputs"] for r in rows
                    ],
                    "accepted_output_counts": [
                        r["step"]["accepted_output_counts"] for r in rows
                    ],
                    "stages": [r["step"]["stages"] for r in rows],
                }
            )
        host_windows = []
        for host, indices in host_ranks.items():
            begin = max(usable[0][1][i]["step"]["observed_ns"] for i in indices)
            end = min(usable[-1][1][i]["step"]["observed_ns"] for i in indices)
            if end <= begin:
                raise ValueError("No common steady observation window within host")
            observations = [
                rows[i]["step"]
                for _, rows in usable
                for i in indices
                if begin < rows[i]["step"]["observed_ns"] <= end
            ]
            host_windows.append(
                {
                    "hostname": host,
                    "begin_ns_exclusive": begin,
                    "end_ns_inclusive": end,
                    "seconds": (end - begin) / 1e9,
                    "effective_tokens": sum(
                        r["accepted_output_tokens"] for r in observations
                    ),
                    "request_observations": sum(
                        r["output_requests"] for r in observations
                    ),
                    "per_rank_observations": [
                        sum(
                            begin < rows[i]["step"]["observed_ns"] <= end
                            for _, rows in usable
                        )
                        for i in indices
                    ],
                }
            )
        windows.append(
            {
                "baseline_forward_id": usable[0][0],
                "last_forward_id": usable[-1][0],
                "seconds": max(w["seconds"] for w in host_windows),
                "effective_tokens": sum(w["effective_tokens"] for w in host_windows),
                "request_observations": sum(
                    w["request_observations"] for w in host_windows
                ),
                "host_windows": host_windows,
            }
        )
    if len(samples) < 8:
        raise ValueError(
            f"Need >= 8 consecutive full-DP decode intervals; found {len(samples)}"
        )
    seconds = sum(w["seconds"] for w in windows)
    tokens = sum(w["effective_tokens"] for w in windows)
    rate = tokens / seconds
    median_step_seconds = statistics.median(s["seconds"] for s in samples)
    mean_tokens = tokens / sum(w["request_observations"] for w in windows)
    median_rate = (
        scenario.concurrency
        * _FIXED_ACCEPTED_TOKENS_PER_REQUEST_STEP
        / median_step_seconds
    )
    return {
        "scenario": scenario.to_dict(),
        "timing": (
            "native_output_observation_cadence"
            if len(hosts) == 1
            else "max_host_local_cadence_at_matching_forward_ids"
        ),
        "isolated_stage_device_seconds": None,
        "capacity_scope": "colocated serving; not isolated D-node capacity",
        "context_semantics": (
            "stage starts/ends are optimistic scheduler metadata; "
            "accepted-output context is reconstructed"
        ),
        "decode_throughput": {
            "basis": "fixed_accepted_tokens_median_step_estimate",
            "accepted_tokens_per_request_step": _FIXED_ACCEPTED_TOKENS_PER_REQUEST_STEP,
            "global_tokens_s": median_rate,
            "per_device_tokens_s": median_rate / scenario.dp_size,
            "per_physical_chip_tokens_s": median_rate / scenario.physical_chips,
        },
        "windows": windows,
        "categories": {
            "decode": {
                "throughput_basis": (
                    "observed_window"
                    if len(hosts) == 1
                    else "sum_host_tokens_over_max_host_window"
                ),
                "steps": len(samples),
                "total_seconds": seconds,
                "median_step_ms": 1000 * median_step_seconds,
                "effective_tokens": tokens,
                "global_tokens_s": rate,
                "per_device_tokens_s": rate / scenario.dp_size,
                "per_physical_chip_tokens_s": rate / scenario.physical_chips,
                "token_kind": "accepted_output_including_bonus",
                "mean_tokens_per_request_step": mean_tokens,
                "first_20": samples[:20],
                "last_20": samples[-20:],
                "samples": samples,
            }
        },
    }


def summarize_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--scenario", choices=["decode", "decode_dp16"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.input.read_text())["results"], args.scenario)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("client", "summarize"):
        raise SystemExit("Usage: bench_decode.py {client|summarize} ...")
    {"client": client_main, "summarize": summarize_main}[sys.argv[1]](sys.argv[2:])
