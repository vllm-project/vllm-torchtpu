#!/usr/bin/env python3
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
"""Measure peak sustained-concurrency decode throughput and TPOT.

Sends --concurrency streaming requests at once to a running OpenAI-compatible
vLLM server. Every request has a distinct, deterministic natural-language
prompt of --prefill-tokens tokens and decodes --decode-tokens tokens with
ignore_eos. Throughput and TPOT come from sliding windows over the interval
where (nearly) all requests decode together, so prefill does not dilute them.

Usage:
    bench_decode_sliding_window.py --model <served-model-name> \\
        --output-dir <dir> [--base-url http://127.0.0.1:8000] \\
        [--tokenizer <dir-or-hf-id>] [--concurrency 16] \\
        [--prefill-tokens 8192] [--decode-tokens 1024] [--rounds 1]

Writes summary.json, summary.md, rounds.csv, timeline.csv, request_tpot.csv
and raw_requests.jsonl to --output-dir. Exits non-zero when a request fails
or a round has no valid measurement window.

Requests only all decode at once if each decode outlasts the prefill of the
other requests: with chunked prefill, decode tokens must exceed about
(concurrency - 1) * ceil(prefill tokens / max_num_batched_tokens). Pass the
server's --max-num-batched-tokens to check this before sending anything.
At high concurrency with long prompts that overlap can shrink to a few
seconds. --prime-prefix-cache avoids it on a server with prefix caching:
each round first sends its prompts with max_tokens=1, so the measured
requests hit the cache and start decoding together.

During each round the server's /metrics is polled for peak running/waiting
requests, KV cache usage and preemptions: waiting requests or preemptions
mean the round was bounded by KV capacity rather than compute.

Adapted from scripts/bench_decode_sliding_window.py in
aios-tpu-infra/tpu_benchmark_daily (a1d1f15).
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import dataclasses
import json
import math
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

PROMPT_TEXT = (
    "The benchmark uses a long, ordinary English passage to exercise "
    "prefill before measuring decode throughput. The passage discusses "
    "software systems, scheduling decisions, distributed execution, and "
    "careful measurement methodology in plain language. Each paragraph is "
    "stable and deterministic, while the request-specific prefix remains "
    "different. The goal is to keep the input natural enough for tokenizer "
    "behavior to resemble real text while preserving a controlled length. "
)
REQUEST_PREFIX_TEMPLATE = (
    "Unique benchmark prefix for request {request_id:04d}. "
    "This opening sentence is deliberately different so the prefix "
    "cache does not share the beginning of this request with any "
    "other request. "
)


@dataclasses.dataclass(frozen=True)
class RequestResult:
    request_id: int
    started_s: float
    finished_s: float
    token_times_s: list[float]
    error: str | None


@dataclasses.dataclass(frozen=True)
class RoundAnalysis:
    summary: dict[str, Any]
    windows: list[dict[str, Any]]
    request_tpots: list[dict[str, Any]]


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return value


def nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return value


def unit_interval_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0 or value > 1:
        raise argparse.ArgumentTypeError("value must be in the interval (0, 1]")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def repeat_and_truncate_token_ids(
    source_token_ids: list[int], target_tokens: int
) -> list[int]:
    if not source_token_ids:
        raise ValueError("fixed prompt text produced no tokens")
    repeats = math.ceil(target_tokens / len(source_token_ids))
    return (source_token_ids * repeats)[:target_tokens]


def build_unique_prompt_token_ids(
    *,
    encode_text: Callable[[str], list[int]],
    target_tokens: int,
    concurrency: int,
) -> tuple[list[list[int]], int]:
    source_token_ids = encode_text(PROMPT_TEXT)
    if not source_token_ids:
        raise ValueError("fixed prompt text produced no tokens")

    prompts: list[list[int]] = []
    for request_id in range(concurrency):
        prefix_token_ids = encode_text(
            REQUEST_PREFIX_TEMPLATE.format(request_id=request_id)
        )
        if not prefix_token_ids:
            raise ValueError(f"request {request_id} unique prefix produced no tokens")
        if len(prefix_token_ids) >= target_tokens:
            prompt = prefix_token_ids[:target_tokens]
        else:
            prompt = prefix_token_ids + repeat_and_truncate_token_ids(
                source_token_ids, target_tokens - len(prefix_token_ids)
            )
        prompts.append(prompt)

    if len({tuple(prompt) for prompt in prompts}) != concurrency:
        raise ValueError("tokenizer did not produce unique request prompts")
    return prompts, len(source_token_ids)


def load_unique_prompt_token_ids(
    *, tokenizer: str, target_tokens: int, concurrency: int
) -> tuple[list[list[int]], int]:
    from transformers import AutoTokenizer

    # A local directory must be complete; anything else is a Hugging Face repo
    # id, fetched (tokenizer files only) into the regular HF cache.
    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer,
        trust_remote_code=False,
        local_files_only=Path(tokenizer).is_dir(),
    )
    return build_unique_prompt_token_ids(
        encode_text=lambda text: hf_tokenizer.encode(text, add_special_tokens=False),
        target_tokens=target_tokens,
        concurrency=concurrency,
    )


def check_full_overlap_possible(
    *,
    concurrency: int,
    prefill_tokens: int,
    decode_tokens: int,
    max_num_batched_tokens: int,
) -> tuple[bool, int]:
    """Estimate whether every request can be decoding at the same time.

    With chunked prefill, each engine step emits one token for every decoding
    request and spends the rest of the token budget on one prompt. The first
    request therefore receives about one token per step while the other
    requests prefill, and only decodes alongside all of them if its decode
    length outlasts those steps. Returns (possible, estimated tokens).
    """
    steps_per_prefill = math.ceil(prefill_tokens / max_num_batched_tokens)
    tokens_before_last_prefill = (concurrency - 1) * steps_per_prefill
    return decode_tokens > tokens_before_last_prefill, tokens_before_last_prefill


SERVER_GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}
SERVER_PREEMPTIONS = "vllm:num_preemptions_total"


def parse_prometheus_sums(text: str, names: set[str]) -> dict[str, float]:
    """Sum the samples of each named metric across all label sets."""
    sums: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, value = line.rpartition(" ")
        name = name_and_labels.split("{", 1)[0]
        if name not in names:
            continue
        try:
            sums[name] = sums.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return sums


class ServerMetricsSampler:
    """Poll /metrics in the background for the duration of one round.

    Sampling is best effort: a server without these metrics, or a failed
    scrape, leaves the corresponding fields as None rather than failing the
    benchmark.
    """

    def __init__(self, metrics_url: str, interval_s: float) -> None:
        self._url = metrics_url
        self._interval_s = interval_s
        self._names = {*SERVER_GAUGES.values(), SERVER_PREEMPTIONS}
        self._peaks: dict[str, float] = {}
        self._first_preemptions: float | None = None
        self._last_preemptions: float | None = None
        self._samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _scrape(self) -> None:
        try:
            response = requests.get(self._url, timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            return
        sums = parse_prometheus_sums(response.text, self._names)
        self._samples += 1
        for key, name in SERVER_GAUGES.items():
            if name in sums:
                self._peaks[key] = max(self._peaks.get(key, 0.0), sums[name])
        if SERVER_PREEMPTIONS in sums:
            if self._first_preemptions is None:
                self._first_preemptions = sums[SERVER_PREEMPTIONS]
            self._last_preemptions = sums[SERVER_PREEMPTIONS]

    def _run(self) -> None:
        while not self._stop.is_set():
            self._scrape()
            self._stop.wait(self._interval_s)

    def __enter__(self) -> ServerMetricsSampler:
        self._scrape()
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        self._scrape()

    def summary(self) -> dict[str, Any]:
        preemptions = (
            self._last_preemptions - self._first_preemptions
            if self._first_preemptions is not None
            and self._last_preemptions is not None
            else None
        )
        return {
            "server_metrics_samples": self._samples,
            "server_running_requests_max": self._peaks.get("running"),
            "server_waiting_requests_max": self._peaks.get("waiting"),
            "server_kv_cache_usage_max": self._peaks.get("kv_cache_usage"),
            "server_preemptions": preemptions,
        }


SERVER_METRIC_FIELDS = (
    "server_metrics_samples",
    "server_running_requests_max",
    "server_waiting_requests_max",
    "server_kv_cache_usage_max",
    "server_preemptions",
)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "avg": statistics.mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stddev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def build_request_tpot_rows(
    *,
    round_index: int,
    results: list[RequestResult],
    batch_start_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        received_tokens = len(result.token_times_s)
        if result.error is not None or received_tokens < 2:
            continue
        first_token_s = result.token_times_s[0]
        last_token_s = result.token_times_s[-1]
        decode_duration_s = last_token_s - first_token_s
        rows.append(
            {
                "round": round_index,
                "request_id": result.request_id,
                "received_tokens": received_tokens,
                "first_token_after_batch_start_s": first_token_s - batch_start_s,
                "last_token_after_batch_start_s": last_token_s - batch_start_s,
                "decode_duration_s": decode_duration_s,
                "tpot_ms": decode_duration_s / (received_tokens - 1) * 1000,
            }
        )
    return rows


def completion_token_delta(
    event: dict[str, Any], previous_completion_tokens: int
) -> tuple[int, int]:
    choices = event.get("choices") or []
    if not choices or not choices[0].get("text"):
        return 0, previous_completion_tokens
    usage = event.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int):
        return (
            max(1, completion_tokens - previous_completion_tokens),
            completion_tokens,
        )
    return 1, previous_completion_tokens + 1


def iter_sse_data(response: requests.Response) -> Iterator[str]:
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        yield line[len("data: ") :]


def streamed_completion(
    *,
    request_id: int,
    endpoint: str,
    request_payload: dict[str, Any],
    data_parallel_rank: int | None,
    timeout_s: float,
) -> RequestResult:
    started_s = time.perf_counter()
    token_times_s: list[float] = []
    previous_completion_tokens = 0
    error = None
    headers = (
        {}
        if data_parallel_rank is None
        else {"X-data-parallel-rank": str(data_parallel_rank)}
    )
    try:
        with requests.post(
            endpoint,
            json=request_payload,
            headers=headers,
            stream=True,
            timeout=timeout_s,
        ) as response:
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:500]}"
            else:
                for data in iter_sse_data(response):
                    received_s = time.perf_counter()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    token_delta, previous_completion_tokens = completion_token_delta(
                        event,
                        previous_completion_tokens,
                    )
                    token_times_s.extend([received_s] * token_delta)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    return RequestResult(
        request_id=request_id,
        started_s=started_s,
        finished_s=time.perf_counter(),
        token_times_s=token_times_s,
        error=error,
    )


def run_round(
    *,
    endpoint: str,
    request_payloads: list[dict[str, Any]],
    concurrency: int,
    data_parallel_size: int,
    timeout_s: float,
) -> list[RequestResult]:
    if len(request_payloads) != concurrency:
        raise ValueError(
            f"expected {concurrency} request payloads, got {len(request_payloads)}"
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                streamed_completion,
                request_id=request_id,
                endpoint=endpoint,
                request_payload=request_payloads[request_id],
                data_parallel_rank=(
                    request_id % data_parallel_size if data_parallel_size > 1 else None
                ),
                timeout_s=timeout_s,
            )
            for request_id in range(concurrency)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    return sorted(results, key=lambda result: result.request_id)


def analyze_round(
    *,
    round_index: int,
    results: list[RequestResult],
    concurrency: int,
    window_s: float,
    step_s: float,
    min_full_overlap_s: float,
    min_full_overlap_tokens: int,
    min_peak_active_fraction: float,
) -> RoundAnalysis:
    failures = [
        {
            "request_id": result.request_id,
            "received_tokens": len(result.token_times_s),
            "error": result.error or "stream returned fewer than two tokens",
        }
        for result in results
        if result.error is not None or len(result.token_times_s) < 2
    ]
    batch_start_s = min((result.started_s for result in results), default=0.0)
    request_tpots = build_request_tpot_rows(
        round_index=round_index,
        results=results,
        batch_start_s=batch_start_s,
    )
    summary: dict[str, Any] = {
        "round": round_index,
        "concurrency": concurrency,
        "successful_requests": len(results) - len(failures),
        "failed_requests": len(failures),
        "full_overlap_start_after_batch_start_s": None,
        "full_overlap_end_after_batch_start_s": None,
        "full_overlap_duration_s": 0.0,
        "full_overlap_token_count": 0,
        "full_overlap_throughput_tok_s": None,
        "active_requests_max": 0,
        "timeline_valid_full_concurrency_decode": False,
        "full_concurrency_window_count": 0,
        "window_count": 0,
        "peak_window_min_active_requests": 0,
        "peak_window_active_requests": 0,
        "peak_window_start_after_batch_start_s": None,
        "peak_window_end_after_batch_start_s": None,
        "peak_window_throughput_tok_s": None,
        "peak_window_tpot_ms": metric_stats([]),
        "usage_tokens": sum(len(result.token_times_s) for result in results),
        "submit_skew_s": None,
        "first_token_skew_s": None,
        "ttft_s": metric_stats([]),
        "end_to_end_throughput_tok_s": None,
        "window_throughput_tok_s": metric_stats([]),
        "peak_active_tpot_ms": metric_stats([]),
        "request_tpot_ms": metric_stats(
            [float(row["tpot_ms"]) for row in request_tpots]
        ),
        "valid": False,
        "invalid_reason": "",
        "errors": failures,
    }
    if failures or len(results) != concurrency:
        summary["invalid_reason"] = "incomplete_requests"
        return RoundAnalysis(summary, [], request_tpots)

    starts_s = [result.started_s for result in results]
    firsts_s = [result.token_times_s[0] for result in results]
    lasts_s = [result.token_times_s[-1] for result in results]
    end_to_end_span_s = max(lasts_s) - min(starts_s)
    full_start_s = max(result.token_times_s[1] for result in results)
    full_end_s = min(result.token_times_s[-1] for result in results)
    full_duration_s = max(0.0, full_end_s - full_start_s)
    all_token_times = sorted(
        timestamp for result in results for timestamp in result.token_times_s
    )
    full_token_left = bisect.bisect_left(all_token_times, full_start_s)
    full_token_right = bisect.bisect_left(all_token_times, full_end_s)
    full_overlap_token_count = (
        full_token_right - full_token_left if full_duration_s > 0 else 0
    )
    full_overlap_throughput = (
        full_overlap_token_count / full_duration_s if full_duration_s > 0 else None
    )
    summary.update(
        {
            "full_overlap_start_after_batch_start_s": full_start_s - batch_start_s,
            "full_overlap_end_after_batch_start_s": full_end_s - batch_start_s,
            "full_overlap_duration_s": full_duration_s,
            "full_overlap_token_count": full_overlap_token_count,
            "full_overlap_throughput_tok_s": full_overlap_throughput,
            "submit_skew_s": max(starts_s) - min(starts_s),
            "first_token_skew_s": max(firsts_s) - min(firsts_s),
            "ttft_s": metric_stats(
                [
                    first_token_s - started_s
                    for first_token_s, started_s in zip(firsts_s, starts_s)
                ]
            ),
            "end_to_end_throughput_tok_s": (
                summary["usage_tokens"] / end_to_end_span_s
            ),
        }
    )
    decode_intervals = [
        (result.token_times_s[1], result.token_times_s[-1]) for result in results
    ]
    if full_duration_s < min_full_overlap_s:
        summary["invalid_reason"] = "full_overlap_shorter_than_minimum"
        return RoundAnalysis(summary, [], request_tpots)
    if full_overlap_token_count < min_full_overlap_tokens:
        summary["invalid_reason"] = "full_overlap_has_too_few_tokens"
        return RoundAnalysis(summary, [], request_tpots)
    scan_start_s = min(start_s for start_s, _ in decode_intervals)
    scan_end_s = max(end_s for _, end_s in decode_intervals)
    if scan_end_s - scan_start_s + 1e-9 < window_s:
        summary["invalid_reason"] = "decode_span_shorter_than_window"
        return RoundAnalysis(summary, [], request_tpots)

    candidate_windows: list[dict[str, Any]] = []
    cursor_s = scan_start_s
    while cursor_s + window_s <= scan_end_s + 1e-9:
        end_s = cursor_s + window_s
        active_requests = sum(
            interval_start_s <= cursor_s and interval_end_s >= end_s
            for interval_start_s, interval_end_s in decode_intervals
        )
        if active_requests == 0:
            cursor_s += step_s
            continue
        token_left = bisect.bisect_left(all_token_times, cursor_s)
        token_right = bisect.bisect_left(all_token_times, end_s)
        token_count = token_right - token_left
        candidate_windows.append(
            {
                "round": round_index,
                "window": len(candidate_windows) + 1,
                "start_after_batch_start_s": cursor_s - batch_start_s,
                "end_after_batch_start_s": end_s - batch_start_s,
                "active_requests": active_requests,
                "token_count": token_count,
                "throughput_tok_s": token_count / window_s,
            }
        )
        cursor_s += step_s

    active_requests_max = max(
        (int(window["active_requests"]) for window in candidate_windows),
        default=0,
    )
    peak_window_min_active_requests = math.ceil(concurrency * min_peak_active_fraction)
    peak_window_candidates = [
        window
        for window in candidate_windows
        if int(window["active_requests"]) >= peak_window_min_active_requests
    ]
    peak_window = max(
        peak_window_candidates,
        key=lambda window: float(window["throughput_tok_s"]),
        default=None,
    )
    if peak_window is not None:
        peak_window_start_s = (
            float(peak_window["start_after_batch_start_s"]) + batch_start_s
        )
        peak_window_end_s = (
            float(peak_window["end_after_batch_start_s"]) + batch_start_s
        )
        peak_window_itls_ms = [
            (current_s - previous_s) * 1000
            for result in results
            for previous_s, current_s in zip(
                result.token_times_s, result.token_times_s[1:]
            )
            if peak_window_start_s <= current_s < peak_window_end_s
        ]
    else:
        peak_window_itls_ms = []
    windows = [
        window
        for window in candidate_windows
        if window["active_requests"] == active_requests_max
    ]
    for index, window in enumerate(windows, start=1):
        window["window"] = index
    throughputs = [float(window["throughput_tok_s"]) for window in windows]
    if windows:
        peak_intervals: list[tuple[float, float]] = []
        for window in windows:
            interval_start_s = (
                float(window["start_after_batch_start_s"]) + batch_start_s
            )
            interval_end_s = float(window["end_after_batch_start_s"]) + batch_start_s
            if peak_intervals and interval_start_s <= peak_intervals[-1][1]:
                previous_start_s, previous_end_s = peak_intervals[-1]
                peak_intervals[-1] = (
                    previous_start_s,
                    max(previous_end_s, interval_end_s),
                )
            else:
                peak_intervals.append((interval_start_s, interval_end_s))
        peak_active_itls_ms = [
            (current_s - previous_s) * 1000
            for result in results
            for previous_s, current_s in zip(
                result.token_times_s, result.token_times_s[1:]
            )
            if any(
                interval_start_s <= current_s < interval_end_s
                for interval_start_s, interval_end_s in peak_intervals
            )
        ]
    else:
        peak_active_itls_ms = []
    summary.update(
        {
            "active_requests_max": active_requests_max,
            "timeline_valid_full_concurrency_decode": (
                active_requests_max == concurrency
            ),
            "full_concurrency_window_count": sum(
                window["active_requests"] == concurrency for window in candidate_windows
            ),
            "window_count": len(windows),
            "peak_window_min_active_requests": (peak_window_min_active_requests),
            "peak_window_active_requests": (
                int(peak_window["active_requests"]) if peak_window is not None else 0
            ),
            "peak_window_start_after_batch_start_s": (
                float(peak_window["start_after_batch_start_s"])
                if peak_window is not None
                else None
            ),
            "peak_window_end_after_batch_start_s": (
                float(peak_window["end_after_batch_start_s"])
                if peak_window is not None
                else None
            ),
            "peak_window_throughput_tok_s": (
                float(peak_window["throughput_tok_s"])
                if peak_window is not None
                else None
            ),
            "peak_window_tpot_ms": metric_stats(peak_window_itls_ms),
            "window_throughput_tok_s": metric_stats(throughputs),
            "peak_active_tpot_ms": metric_stats(peak_active_itls_ms),
            "valid": bool(windows),
            "invalid_reason": "" if windows else "no_samples",
        }
    )
    return RoundAnalysis(summary, windows, request_tpots)


def write_raw_requests(
    path: Path,
    *,
    round_index: int,
    results: list[RequestResult],
) -> None:
    batch_start_s = min((result.started_s for result in results), default=0.0)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            record = {
                "round": round_index,
                "request_id": result.request_id,
                "started_after_batch_start_s": result.started_s - batch_start_s,
                "finished_after_batch_start_s": result.finished_s - batch_start_s,
                "received_tokens": len(result.token_times_s),
                "token_times_after_batch_start_s": [
                    timestamp - batch_start_s for timestamp in result.token_times_s
                ],
                "error": result.error,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_request_tpot_csv(path: Path, request_tpots: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round",
        "request_id",
        "received_tokens",
        "first_token_after_batch_start_s",
        "last_token_after_batch_start_s",
        "decode_duration_s",
        "tpot_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(request_tpots)


def write_rounds_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round",
        "valid",
        "successful_requests",
        "failed_requests",
        "usage_tokens",
        "submit_skew_s",
        "first_token_skew_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "end_to_end_throughput_tok_s",
        "full_overlap_duration_s",
        "full_overlap_token_count",
        "full_overlap_throughput_tok_s",
        "active_requests_max",
        "timeline_valid_full_concurrency_decode",
        "full_concurrency_window_count",
        "window_count",
        "peak_window_min_active_requests",
        "peak_window_active_requests",
        "peak_window_start_after_batch_start_s",
        "peak_window_end_after_batch_start_s",
        "peak_window_throughput_tok_s",
        "peak_window_tpot_count",
        "peak_window_tpot_avg_ms",
        "peak_window_tpot_p50_ms",
        "peak_window_tpot_p90_ms",
        "peak_window_tpot_p99_ms",
        "throughput_count",
        "throughput_avg_tok_s",
        "throughput_min_tok_s",
        "throughput_max_tok_s",
        "throughput_stddev_tok_s",
        "throughput_p50_tok_s",
        "throughput_p90_tok_s",
        "throughput_p99_tok_s",
        "peak_active_tpot_count",
        "peak_active_tpot_avg_ms",
        "peak_active_tpot_min_ms",
        "peak_active_tpot_max_ms",
        "peak_active_tpot_stddev_ms",
        "peak_active_tpot_p50_ms",
        "peak_active_tpot_p90_ms",
        "peak_active_tpot_p99_ms",
        "request_tpot_count",
        "request_tpot_avg_ms",
        "request_tpot_min_ms",
        "request_tpot_max_ms",
        "request_tpot_stddev_ms",
        "request_tpot_p50_ms",
        "request_tpot_p90_ms",
        "request_tpot_p99_ms",
        *SERVER_METRIC_FIELDS,
        "invalid_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rounds:
            throughput = row["window_throughput_tok_s"]
            peak_active_tpot = row["peak_active_tpot_ms"]
            peak_window_tpot = row["peak_window_tpot_ms"]
            request_tpot = row["request_tpot_ms"]
            writer.writerow(
                {
                    "round": row["round"],
                    "valid": row["valid"],
                    "successful_requests": row["successful_requests"],
                    "failed_requests": row["failed_requests"],
                    "usage_tokens": row["usage_tokens"],
                    "submit_skew_s": row["submit_skew_s"],
                    "first_token_skew_s": row["first_token_skew_s"],
                    "ttft_p50_s": row["ttft_s"]["p50"],
                    "ttft_p90_s": row["ttft_s"]["p90"],
                    "end_to_end_throughput_tok_s": row["end_to_end_throughput_tok_s"],
                    "full_overlap_duration_s": row["full_overlap_duration_s"],
                    "full_overlap_token_count": row["full_overlap_token_count"],
                    "full_overlap_throughput_tok_s": row[
                        "full_overlap_throughput_tok_s"
                    ],
                    "active_requests_max": row["active_requests_max"],
                    "timeline_valid_full_concurrency_decode": row[
                        "timeline_valid_full_concurrency_decode"
                    ],
                    "full_concurrency_window_count": row[
                        "full_concurrency_window_count"
                    ],
                    "window_count": row["window_count"],
                    "peak_window_min_active_requests": row[
                        "peak_window_min_active_requests"
                    ],
                    "peak_window_active_requests": row["peak_window_active_requests"],
                    "peak_window_start_after_batch_start_s": row[
                        "peak_window_start_after_batch_start_s"
                    ],
                    "peak_window_end_after_batch_start_s": row[
                        "peak_window_end_after_batch_start_s"
                    ],
                    "peak_window_throughput_tok_s": row["peak_window_throughput_tok_s"],
                    "peak_window_tpot_count": peak_window_tpot["count"],
                    "peak_window_tpot_avg_ms": peak_window_tpot["avg"],
                    "peak_window_tpot_p50_ms": peak_window_tpot["p50"],
                    "peak_window_tpot_p90_ms": peak_window_tpot["p90"],
                    "peak_window_tpot_p99_ms": peak_window_tpot["p99"],
                    "throughput_count": throughput["count"],
                    "throughput_avg_tok_s": throughput["avg"],
                    "throughput_min_tok_s": throughput["min"],
                    "throughput_max_tok_s": throughput["max"],
                    "throughput_stddev_tok_s": throughput["stddev"],
                    "throughput_p50_tok_s": throughput["p50"],
                    "throughput_p90_tok_s": throughput["p90"],
                    "throughput_p99_tok_s": throughput["p99"],
                    "peak_active_tpot_count": peak_active_tpot["count"],
                    "peak_active_tpot_avg_ms": peak_active_tpot["avg"],
                    "peak_active_tpot_min_ms": peak_active_tpot["min"],
                    "peak_active_tpot_max_ms": peak_active_tpot["max"],
                    "peak_active_tpot_stddev_ms": peak_active_tpot["stddev"],
                    "peak_active_tpot_p50_ms": peak_active_tpot["p50"],
                    "peak_active_tpot_p90_ms": peak_active_tpot["p90"],
                    "peak_active_tpot_p99_ms": peak_active_tpot["p99"],
                    "request_tpot_count": request_tpot["count"],
                    "request_tpot_avg_ms": request_tpot["avg"],
                    "request_tpot_min_ms": request_tpot["min"],
                    "request_tpot_max_ms": request_tpot["max"],
                    "request_tpot_stddev_ms": request_tpot["stddev"],
                    "request_tpot_p50_ms": request_tpot["p50"],
                    "request_tpot_p90_ms": request_tpot["p90"],
                    "request_tpot_p99_ms": request_tpot["p99"],
                    **{field: row.get(field) for field in SERVER_METRIC_FIELDS},
                    "invalid_reason": row["invalid_reason"],
                }
            )


def write_timeline_csv(path: Path, windows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round",
        "window",
        "start_after_batch_start_s",
        "end_after_batch_start_s",
        "active_requests",
        "token_count",
        "throughput_tok_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(windows)


def format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def stats_row(label: str, stats: dict[str, Any]) -> str:
    return (
        f"|{label}|{stats['count']}|{format_metric(stats['avg'])}|"
        f"{format_metric(stats['min'])}|{format_metric(stats['max'])}|"
        f"{format_metric(stats['stddev'])}|{format_metric(stats['p50'])}|"
        f"{format_metric(stats['p90'])}|{format_metric(stats['p99'])}|"
    )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    benchmark = summary["benchmark"]
    aggregate = summary["aggregate"]
    window_s = benchmark["window_s"]
    valid_rounds = summary["valid_rounds"]
    requested_rounds = summary["requested_rounds"]
    best = summary["best"]
    tldr = (
        f"C{benchmark['concurrency']}, {benchmark['prefill_tokens']}-token "
        f"prompts, {benchmark['decode_tokens']}-token decode; "
        f"{valid_rounds}/{requested_rounds} valid rounds."
    )
    if best is not None:
        tldr += (
            f" Best round: peak {window_s}s window "
            f"{format_metric(best['output_throughput'])} tok/s at TPOT P50 "
            f"{format_metric(best['mean_tpot_ms'])} ms."
        )
    lines = [
        "# Decode Sliding-Window Benchmark",
        "",
        "## TL;DR",
        "",
        tldr,
        "",
        "## Configuration",
        "",
        f"- Model: `{benchmark['model']}`",
        f"- Concurrency: `{benchmark['concurrency']}`",
        f"- DP size: `{benchmark['data_parallel_size']}`",
        f"- TP size: `{benchmark['tensor_parallel_size']}`",
        f"- Prefill: `{benchmark['prefill_tokens']}` tokens",
        f"- Decode: `{benchmark['decode_tokens']}` tokens",
        (
            "- Prompt: each request starts with a distinct deterministic "
            "natural-language prefix, followed by a fixed English passage "
            "tokenized, repeated and truncated to the target length."
        ),
        f"- Tokenizer: `{benchmark['tokenizer']}`",
        f"- Sliding window: `{window_s}` s, step `{benchmark['step_s']}` s",
        (
            "- Window selection: scan the whole decode timeline, count the "
            "requests active for each entire window, and keep only the "
            "windows at the highest sustained concurrency actually reached; "
            f"reaching the submitted concurrency `{benchmark['concurrency']}` "
            "is not required."
        ),
        (
            "- Reported peak: the highest "
            f"`{window_s}` s throughput among windows with at least "
            f"`{benchmark['min_peak_active_fraction']:.0%}` of the submitted "
            "requests continuously active. Peak-active P50 is kept as a "
            "scheduling-uniformity diagnostic."
        ),
        (
            "- Optional full-concurrency gate: every submitted request "
            "decodes at the same time for at least "
            f"`{benchmark['min_full_overlap_s']}` s and "
            f"`{benchmark['min_full_overlap_tokens']}` tokens; both default "
            "to 0 and do not restrict the peak window."
        ),
        (
            "- TPOT: inter-token intervals inside the peak-active windows. "
            "Request TPOT, over each request's full decode, is a supplement."
        ),
        (
            "- Submission: the client submits all requests at once, with no "
            "server-side admission barrier."
        ),
        (
            "- Token accounting: expanded from the streamed cumulative "
            "`usage.completion_tokens`."
        ),
        (
            "- Server metrics: `/metrics` polled every "
            f"`{benchmark['metrics_interval_s']}` s during each round."
        ),
        "",
        "## Aggregate Metrics",
        "",
        "|Metric|count|avg|min|max|stddev|p50|p90|p99|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        stats_row("Decode throughput (tok/s)", aggregate["window_throughput_tok_s"]),
        stats_row(
            f"Peak `{window_s}`s throughput (tok/s)",
            aggregate["peak_window_throughput_tok_s"],
        ),
        stats_row(
            f"Peak `{window_s}`s TPOT P50 across rounds (ms)",
            aggregate["peak_window_tpot_p50_ms"],
        ),
        stats_row(
            "Full-overlap throughput (tok/s)",
            aggregate["full_overlap_throughput_tok_s"],
        ),
        stats_row(
            "Peak-active TPOT P50 across rounds (ms)",
            aggregate["peak_active_tpot_p50_ms"],
        ),
        stats_row("Request TPOT (ms)", aggregate["request_tpot_ms"]),
        stats_row("TTFT P50 across rounds (s)", aggregate["ttft_p50_s"]),
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/vllm/benchmarking/bench_decode_sliding_window.py \\",
        f"  --base-url {benchmark['base_url']} \\",
        f"  --model {benchmark['model']} \\",
        "  --output-dir <OUTPUT_DIR> \\",
        f"  --concurrency {benchmark['concurrency']} \\",
        f"  --data-parallel-size {benchmark['data_parallel_size']} \\",
        f"  --tensor-parallel-size {benchmark['tensor_parallel_size']} \\",
        f"  --prefill-tokens {benchmark['prefill_tokens']} \\",
        f"  --decode-tokens {benchmark['decode_tokens']} \\",
        f"  --tokenizer {benchmark['tokenizer']} \\",
        *(
            [f"  --max-num-batched-tokens {benchmark['max_num_batched_tokens']} \\"]
            if benchmark["max_num_batched_tokens"] is not None
            else []
        ),
        f"  --rounds {requested_rounds} \\",
        f"  --window-seconds {window_s} \\",
        f"  --step-seconds {benchmark['step_s']} \\",
        f"  --min-full-overlap-seconds {benchmark['min_full_overlap_s']} \\",
        f"  --min-full-overlap-tokens {benchmark['min_full_overlap_tokens']} \\",
        *(["  --prime-prefix-cache \\"] if benchmark.get("prime_prefix_cache") else []),
        (f"  --min-peak-active-fraction {benchmark['min_peak_active_fraction']}"),
        "```",
        "",
        "## Rounds",
        "",
        (
            "|round|valid|requests|peak active|full overlap (s)|"
            "full-overlap tok/s|peak window active|peak window tok/s|"
            "peak window TPOT P50 (ms)|windows|throughput P50 (tok/s)|"
            "peak TPOT P50 (ms)|TTFT P50 (s)|server running max|"
            "server waiting max|KV usage max|preemptions|reason|"
        ),
        (
            "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---|"
        ),
    ]
    for row in summary["results"]:
        lines.append(
            f"|{row['round']}|{row['valid']}|"
            f"{row['successful_requests']}/{benchmark['concurrency']}|"
            f"{row['active_requests_max']}|"
            f"{format_metric(row['full_overlap_duration_s'])}|"
            f"{format_metric(row['full_overlap_throughput_tok_s'])}|"
            f"{row['peak_window_active_requests']}|"
            f"{format_metric(row['peak_window_throughput_tok_s'])}|"
            f"{format_metric(row['peak_window_tpot_ms']['p50'])}|"
            f"{row['window_count']}|"
            f"{format_metric(row['window_throughput_tok_s']['p50'])}|"
            f"{format_metric(row['peak_active_tpot_ms']['p50'])}|"
            f"{format_metric(row['ttft_s']['p50'])}|"
            f"{format_metric(row.get('server_running_requests_max'))}|"
            f"{format_metric(row.get('server_waiting_requests_max'))}|"
            f"{format_metric(row.get('server_kv_cache_usage_max'))}|"
            f"{format_metric(row.get('server_preemptions'))}|"
            f"{row['invalid_reason']}|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run distinct concurrent prefill/decode requests and measure "
            "peak sustained-concurrency decode intervals with sliding windows."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        required=True,
        help="Served model name, sent as the request's model field.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=positive_int, default=16)
    parser.add_argument(
        "--data-parallel-size",
        type=positive_int,
        default=1,
        help=(
            "The server's DP size. Above 1, requests are spread over the "
            "ranks round-robin with the X-data-parallel-rank header."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=positive_int,
        default=1,
        help="The server's TP size; only recorded in the report.",
    )
    parser.add_argument("--prefill-tokens", type=positive_int, default=8192)
    parser.add_argument("--decode-tokens", type=positive_int, default=1024)
    parser.add_argument(
        "--tokenizer",
        "--tokenizer-dir",
        dest="tokenizer",
        default=None,
        help=(
            "Local tokenizer directory or Hugging Face repo id; defaults to --model."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=positive_int,
        default=None,
        help=(
            "The server's token budget per step. When given, refuse to run "
            "if the decode length cannot outlast the prefill of the other "
            "requests, since then the requests never all decode at once."
        ),
    )
    parser.add_argument(
        "--metrics-interval-seconds",
        type=positive_float,
        default=1.0,
        help="Polling interval for the server's /metrics during each round.",
    )
    parser.add_argument("--rounds", type=positive_int, default=1)
    parser.add_argument("--cache-salt-prefix", default="")
    parser.add_argument(
        "--prime-prefix-cache",
        action="store_true",
        help=(
            "Before each round, send the round's prompts with max_tokens=1 so "
            "the measured requests hit the prefix cache and decode together. "
            "Needs a server with prefix caching enabled."
        ),
    )
    parser.add_argument("--window-seconds", type=positive_float, default=10.0)
    parser.add_argument("--step-seconds", type=positive_float, default=1.0)
    parser.add_argument(
        "--min-full-overlap-seconds",
        type=nonnegative_float,
        default=0.0,
        help="Require at least this many seconds of strict full concurrency.",
    )
    parser.add_argument(
        "--min-full-overlap-tokens",
        type=nonnegative_int,
        default=0,
        help="Require this many tokens in the strict full-overlap interval.",
    )
    parser.add_argument(
        "--min-peak-active-fraction",
        type=unit_interval_float,
        default=0.9,
        help=(
            "Only use windows with at least this fraction of submitted "
            "requests continuously active when selecting reported peak "
            "throughput."
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds", type=positive_float, default=3600.0
    )
    parser.add_argument("--cooldown-seconds", type=nonnegative_float, default=2.0)
    args = parser.parse_args()
    if args.decode_tokens < 2:
        parser.error("--decode-tokens must be at least 2 for decode interval timing")
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.max_num_batched_tokens is not None and not args.prime_prefix_cache:
        possible, tokens_before_last_prefill = check_full_overlap_possible(
            concurrency=args.concurrency,
            prefill_tokens=args.prefill_tokens,
            decode_tokens=args.decode_tokens,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        if not possible:
            parser.error(
                f"--decode-tokens {args.decode_tokens} is too short: the first "
                f"request decodes about {tokens_before_last_prefill} tokens "
                "while the other requests prefill, so the requests never all "
                "decode at once; raise --decode-tokens or the server's "
                "max_num_batched_tokens"
            )
    return args


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    health_response = requests.get(f"{base_url}/health", timeout=10)
    health_response.raise_for_status()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_requests.jsonl"
    raw_path.write_text("", encoding="utf-8")

    prompts, source_token_count = load_unique_prompt_token_ids(
        tokenizer=args.tokenizer,
        target_tokens=args.prefill_tokens,
        concurrency=args.concurrency,
    )
    print(
        "[prompt] mode=unique_natural_language_prefix "
        f"source_tokens={source_token_count} "
        f"request_tokens={len(prompts[0])} "
        f"unique_requests={len(prompts)}",
        flush=True,
    )
    endpoint = f"{base_url}/v1/completions"
    run_namespace = args.cache_salt_prefix or (
        f"decode-bench-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )

    round_summaries: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    all_request_tpots: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        print(
            f"[decode] round={round_index}/{args.rounds} "
            f"C={args.concurrency} P={args.prefill_tokens} D={args.decode_tokens}",
            flush=True,
        )
        round_namespace = (
            run_namespace if args.rounds == 1 else f"{run_namespace}-round{round_index}"
        )
        request_payloads = [
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.decode_tokens,
                "temperature": 0.0,
                "stream": True,
                "ignore_eos": True,
                "stream_options": {
                    "include_usage": True,
                    "continuous_usage_stats": True,
                },
                "cache_salt": (
                    f"{round_namespace}-c{args.concurrency}-req{request_id}"
                ),
            }
            for request_id, prompt in enumerate(prompts)
        ]
        prime_summary: dict[str, Any] = {}
        if args.prime_prefix_cache:
            prime_start_s = time.perf_counter()
            prime_results = run_round(
                endpoint=endpoint,
                request_payloads=[
                    {**payload, "max_tokens": 1} for payload in request_payloads
                ],
                concurrency=args.concurrency,
                data_parallel_size=args.data_parallel_size,
                timeout_s=args.request_timeout_seconds,
            )
            prime_summary = {
                "prime_duration_s": time.perf_counter() - prime_start_s,
                "prime_failures": [
                    {"request_id": result.request_id, "error": result.error}
                    for result in prime_results
                    if result.error is not None
                ],
            }
            print(
                f"[prime] duration_s={prime_summary['prime_duration_s']:.1f} "
                f"failures={len(prime_summary['prime_failures'])}",
                flush=True,
            )
        with ServerMetricsSampler(
            f"{base_url}/metrics", args.metrics_interval_seconds
        ) as sampler:
            results = run_round(
                endpoint=endpoint,
                request_payloads=request_payloads,
                concurrency=args.concurrency,
                data_parallel_size=args.data_parallel_size,
                timeout_s=args.request_timeout_seconds,
            )
        server_metrics = sampler.summary()
        write_raw_requests(raw_path, round_index=round_index, results=results)
        analysis = analyze_round(
            round_index=round_index,
            results=results,
            concurrency=args.concurrency,
            window_s=args.window_seconds,
            step_s=args.step_seconds,
            min_full_overlap_s=args.min_full_overlap_seconds,
            min_full_overlap_tokens=args.min_full_overlap_tokens,
            min_peak_active_fraction=args.min_peak_active_fraction,
        )
        analysis.summary.update(server_metrics)
        analysis.summary.update(prime_summary)
        if prime_summary.get("prime_failures"):
            # Unprimed requests prefill cold, so the window would not be steady decode.
            analysis.summary["valid"] = False
            analysis.summary["invalid_reason"] = "prime_failed"
        round_summaries.append(analysis.summary)
        all_windows.extend(analysis.windows)
        all_request_tpots.extend(analysis.request_tpots)
        print(
            "[server-metrics] "
            + " ".join(f"{key}={value}" for key, value in server_metrics.items()),
            flush=True,
        )
        print(
            f"[round-summary] valid={analysis.summary['valid']} "
            f"peak_active={analysis.summary['active_requests_max']} "
            f"windows={analysis.summary['window_count']} "
            f"throughput_p50={analysis.summary['window_throughput_tok_s']['p50']} "
            f"full_overlap_throughput="
            f"{analysis.summary['full_overlap_throughput_tok_s']} "
            f"peak_window_throughput="
            f"{analysis.summary['peak_window_throughput_tok_s']} "
            f"peak_window_active="
            f"{analysis.summary['peak_window_active_requests']} "
            f"peak_active_tpot_p50_ms="
            f"{analysis.summary['peak_active_tpot_ms']['p50']}",
            flush=True,
        )
        if round_index < args.rounds and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)

    valid_rounds = [row for row in round_summaries if row["valid"]]
    all_throughputs = [float(window["throughput_tok_s"]) for window in all_windows]
    best_round = max(
        valid_rounds,
        key=lambda row: float(row["peak_window_throughput_tok_s"] or -1),
        default=None,
    )
    summary = {
        "schema_version": 8,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark": {
            "benchmark_config": (
                f"dp{args.data_parallel_size}_tp{args.tensor_parallel_size}"
                f"_decode_c{args.concurrency}"
                f"_p{args.prefill_tokens}_d{args.decode_tokens}"
            ),
            "base_url": base_url,
            "model": args.model,
            "input_length": args.prefill_tokens,
            "output_length": args.decode_tokens,
            "concurrency": args.concurrency,
            "data_parallel_size": args.data_parallel_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "prefill_tokens": args.prefill_tokens,
            "decode_tokens": args.decode_tokens,
            "prompt_mode": "unique_natural_language_prefix_token_ids",
            "prompt_source_text": PROMPT_TEXT,
            "prompt_source_tokens": source_token_count,
            "tokenizer": args.tokenizer,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "metrics_interval_s": args.metrics_interval_seconds,
            "identical_prompt_for_all_requests": False,
            "window_s": args.window_seconds,
            "step_s": args.step_seconds,
            "min_full_overlap_s": args.min_full_overlap_seconds,
            "min_full_overlap_tokens": args.min_full_overlap_tokens,
            "min_peak_active_fraction": args.min_peak_active_fraction,
            "prime_prefix_cache": args.prime_prefix_cache,
            "boundary": "peak_sustained_concurrency_sliding_windows",
            "token_accounting": "continuous_usage_completion_tokens",
        },
        "best": (
            {
                "completed": best_round["successful_requests"],
                "concurrency": args.concurrency,
                "failed": best_round["failed_requests"],
                "file": "rounds.csv",
                "mean_tpot_ms": best_round["peak_window_tpot_ms"]["p50"],
                "output_throughput": best_round["peak_window_throughput_tok_s"],
                "peak_active_window_p50_throughput": best_round[
                    "window_throughput_tok_s"
                ]["p50"],
                "full_overlap_throughput": best_round["full_overlap_throughput_tok_s"],
                "p99_tpot_ms": best_round["peak_active_tpot_ms"]["p99"],
                "round": best_round["round"],
            }
            if best_round is not None
            else None
        ),
        "requested_rounds": args.rounds,
        "valid_rounds": len(valid_rounds),
        "aggregate": {
            "window_throughput_tok_s": metric_stats(all_throughputs),
            "peak_window_throughput_tok_s": metric_stats(
                [
                    float(row["peak_window_throughput_tok_s"])
                    for row in valid_rounds
                    if row["peak_window_throughput_tok_s"] is not None
                ]
            ),
            "peak_window_active_requests": metric_stats(
                [
                    float(row["peak_window_active_requests"])
                    for row in valid_rounds
                    if row["peak_window_throughput_tok_s"] is not None
                ]
            ),
            "peak_window_tpot_p50_ms": metric_stats(
                [
                    float(row["peak_window_tpot_ms"]["p50"])
                    for row in valid_rounds
                    if row["peak_window_tpot_ms"]["p50"] is not None
                ]
            ),
            "full_overlap_throughput_tok_s": metric_stats(
                [
                    float(row["full_overlap_throughput_tok_s"])
                    for row in valid_rounds
                    if row["full_overlap_throughput_tok_s"] is not None
                ]
            ),
            "request_tpot_ms": metric_stats(
                [float(row["tpot_ms"]) for row in all_request_tpots]
            ),
            "peak_active_tpot_p50_ms": metric_stats(
                [
                    float(row["peak_active_tpot_ms"]["p50"])
                    for row in valid_rounds
                    if row["peak_active_tpot_ms"]["p50"] is not None
                ]
            ),
            "end_to_end_throughput_tok_s": metric_stats(
                [float(row["end_to_end_throughput_tok_s"]) for row in valid_rounds]
            ),
            "active_requests_max": metric_stats(
                [float(row["active_requests_max"]) for row in valid_rounds]
            ),
            "first_token_skew_s": metric_stats(
                [float(row["first_token_skew_s"]) for row in valid_rounds]
            ),
            "ttft_p50_s": metric_stats(
                [float(row["ttft_s"]["p50"]) for row in valid_rounds]
            ),
            "ttft_p90_s": metric_stats(
                [float(row["ttft_s"]["p90"]) for row in valid_rounds]
            ),
            "round_p50_throughput_tok_s": metric_stats(
                [float(row["window_throughput_tok_s"]["p50"]) for row in valid_rounds]
            ),
            "round_avg_request_tpot_ms": metric_stats(
                [
                    float(row["request_tpot_ms"]["avg"])
                    for row in round_summaries
                    if row["request_tpot_ms"]["avg"] is not None
                ]
            ),
        },
        "results": round_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rounds_csv(args.output_dir / "rounds.csv", round_summaries)
    write_timeline_csv(args.output_dir / "timeline.csv", all_windows)
    write_request_tpot_csv(args.output_dir / "request_tpot.csv", all_request_tpots)
    write_markdown(args.output_dir / "summary.md", summary)

    failed_requests = sum(row["failed_requests"] for row in round_summaries)
    if failed_requests:
        raise SystemExit(f"decode benchmark has {failed_requests} failed requests")
    invalid_rounds = [row for row in round_summaries if not row["valid"]]
    if invalid_rounds:
        reasons = ", ".join(
            f"round {row['round']}: {row['invalid_reason']}" for row in invalid_rounds
        )
        raise SystemExit(f"decode benchmark has invalid rounds: {reasons}")
    aggregate = summary["aggregate"]
    throughput_p50 = aggregate["window_throughput_tok_s"]["p50"]
    peak_window_throughput = aggregate["peak_window_throughput_tok_s"]["p50"]
    full_overlap_throughput = aggregate["full_overlap_throughput_tok_s"]["p50"]
    peak_tpot_avg = aggregate["peak_active_tpot_p50_ms"]["avg"]
    if (
        throughput_p50 is None
        or peak_window_throughput is None
        or peak_tpot_avg is None
    ):
        print(
            f"[done] no complete sliding window; output={args.output_dir}",
            flush=True,
        )
    else:
        print(
            "[done] "
            f"peak_{args.window_seconds:g}s_throughput="
            f"{peak_window_throughput:.3f} tok/s "
            f"throughput_p50={throughput_p50:.3f} tok/s "
            f"full_overlap_throughput_p50="
            f"{format_metric(full_overlap_throughput)} tok/s "
            f"peak_active_tpot_p50_avg={peak_tpot_avg:.3f} ms "
            f"output={args.output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
