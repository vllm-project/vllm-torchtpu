#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import statistics
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PREFILL_URL = "http://127.0.0.1:8400"
DECODE_URL = "http://127.0.0.1:9400"
MODEL = "Qwen3.5-35B-A3B-FP8"
TOKENIZER_DIR = ("/mnt/data/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/"
                 "snapshots/9d1823d2dee688a6b25e77009dc727688c44936e")
PROMPT_LENS = (128, 1024, 4096, 8192)
MAX_TOKENS = 128
MEASURED_ROUNDS = 3
DP_SIZE = 4
MULTI_CONCURRENCY = 16
TIMEOUT_S = 3600.0
REQUEST_PREFIX = "pd-mtp-k3-pr291"
OUT_DIR = (Path(
    os.environ.get(
        "RUN_ROOT",
        "/mnt/data/tpu-misc-results/"
        "qwen35-35b-pr291-pd-dp4-mtp3",
    )) / "workload")

METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
                       r"(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)$")
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')
INVALID_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (ordered[lower] * (upper - position) + ordered[upper] *
            (position - lower))


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "avg": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def headers(request_id: str, dp_rank: int | None = None) -> dict[str, str]:
    result = {"Content-Type": "application/json", "X-Request-Id": request_id}
    if dp_rank is not None:
        result["X-data-parallel-rank"] = str(dp_rank)
    return result


def open_post(
    url: str,
    payload: dict[str, Any],
    request_id: str,
    dp_rank: int | None = None,
):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers(request_id, dp_rank),
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"POST {url} returned HTTP {exc.code}: {body[:4000]!r}") from exc


def post_json(
    url: str,
    payload: dict[str, Any],
    request_id: str,
    dp_rank: int | None = None,
) -> Any:
    with open_post(url, payload, request_id, dp_rank) as response:
        return json.load(response)


def get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read().decode()


def post_stream(
    url: str,
    payload: dict[str, Any],
    request_id: str,
    dp_rank: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_s: float | None = None
    token_times_s: list[float] = []
    usage: dict[str, Any] = {}
    text_excerpt = ""
    finish_reason: str | None = None
    with open_post(url, payload, request_id, dp_rank) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            token_ids = choice.get("token_ids")
            text = choice.get("text")
            chunk_tokens = (len(token_ids) if isinstance(token_ids, list) else
                            int(isinstance(text, str) and bool(text)))
            if chunk_tokens:
                now = time.perf_counter()
                first_token_s = now if first_token_s is None else first_token_s
                token_times_s.extend([now] * chunk_tokens)
            if isinstance(text, str) and len(text_excerpt) < 512:
                text_excerpt += text[:512 - len(text_excerpt)]

    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = len(token_times_s)
    token_times_s.extend([ended] *
                         max(0, completion_tokens - len(token_times_s)))
    tpot_ms = [(right - left) * 1000.0
               for left, right in zip(token_times_s, token_times_s[1:])]
    elapsed = ended - started
    return {
        "ttft_s": None if first_token_s is None else first_token_s - started,
        "completion_tokens": completion_tokens,
        "output_tokens_per_s": completion_tokens / elapsed,
        "mean_tpot_ms": statistics.fmean(tpot_ms) if tpot_ms else None,
        "usage": usage,
        "text_excerpt": text_excerpt,
        "finish_reason": finish_reason,
    }


MetricKey = tuple[str, tuple[tuple[str, str], ...]]


def parse_metrics(text: str) -> dict[MetricKey, float]:
    result: dict[MetricKey, float] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match is None:
            continue
        labels = tuple(sorted(LABEL_RE.findall(match.group("labels") or "")))
        result[(match.group("name"), labels)] = float(match.group("value"))
    return result


def metric_delta(
    before: dict[MetricKey, float],
    after: dict[MetricKey, float],
    name: str,
) -> float:
    return sum(value - before.get(key, 0.0) for key, value in after.items()
               if key[0] in {name, name + "_total"})


def acceptance_delta(before_text: str, after_text: str) -> dict[str, Any]:
    before, after = parse_metrics(before_text), parse_metrics(after_text)
    drafts = metric_delta(before, after, "vllm:spec_decode_num_drafts")
    draft_tokens = metric_delta(before, after,
                                "vllm:spec_decode_num_draft_tokens")
    accepted = metric_delta(before, after,
                            "vllm:spec_decode_num_accepted_tokens")
    per_position: dict[str, float] = {}
    for key, value in after.items():
        if key[0] not in {
                "vllm:spec_decode_num_accepted_tokens_per_pos",
                "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        }:
            continue
        labels = dict(key[1])
        position = labels.get("position") or labels.get("pos") or labels.get(
            "index", str(key[1]))
        per_position[position] = per_position.get(
            position, 0.0) + (value - before.get(key, 0.0))
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "accept_rate": accepted / draft_tokens if draft_tokens else None,
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else None,
        "per_position_accept_rate": {
            position: count / drafts if drafts else None
            for position, count in sorted(per_position.items())
        },
    }


def exact_prompt(tokenizer: Any, prompt_len: int) -> list[int]:
    prefix = ("用户：请阅读下面的技术背景，并在最后回答问题。"
              "我们正在验证大语言模型推理服务的解耦部署。")
    body = ("该系统把提示词预填充与逐 token 解码放在两台 TPU 主机上，"
            "使用可靠的数据通道传输注意力缓存和状态空间模型状态。"
            "测试关注长上下文、并发请求、推测解码接受率、吞吐、延迟和稳定性。")
    suffix = ("问题：请用一句自然语言概括这项测试最重要的目标，"
              "并说明为什么需要同时观察正确性和稳定性。助手：")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    middle_size = prompt_len - len(prefix_ids) - len(suffix_ids)
    if middle_size < 0:
        raise ValueError(f"prompt_len={prompt_len} is too short")
    repeats = (middle_size + len(body_ids) - 1) // len(body_ids)
    token_ids = prefix_ids + (body_ids * repeats)[:middle_size] + suffix_ids
    assert len(token_ids) == prompt_len
    return token_ids


def render_prompt(prompt_ids: list[int]) -> list[int]:
    value = post_json(
        PREFILL_URL + "/v1/completions/render",
        {
            "add_special_tokens": False,
            "model": MODEL,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        f"render-{uuid.uuid4()}",
    )
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"render returned unexpected value: {value!r}")
    rendered = value[0].get("token_ids")
    if not isinstance(rendered, list) or not all(
            type(item) is int for item in rendered):
        raise RuntimeError("render response has no integer token_ids")
    if len(rendered) != len(prompt_ids):
        raise RuntimeError(
            f"render changed prompt length {len(prompt_ids)} -> {len(rendered)}"
        )
    return rendered


def base_payload(prompt_ids: list[int], max_tokens: int) -> dict[str, Any]:
    return {
        "add_special_tokens": False,
        "ignore_eos": True,
        "max_tokens": max_tokens,
        "model": MODEL,
        "n": 1,
        "prompt": prompt_ids,
        "return_token_ids": True,
        "seed": 42,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def run_one(
    index: int,
    prompt_len: int,
    prompt_ids: list[int],
    phase: str,
    round_index: int,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    dp_rank = index % DP_SIZE
    request_id = (f"{REQUEST_PREFIX}-{phase}-p{prompt_len}-r{round_index}-"
                  f"i{index}-{uuid.uuid4()}")
    barrier.wait(timeout=60)
    started = time.perf_counter()
    prefill_response = post_json(
        PREFILL_URL + "/v1/completions",
        base_payload(prompt_ids, 1),
        request_id,
        dp_rank,
    )
    after_prefill = time.perf_counter()
    if not isinstance(prefill_response, dict):
        raise RuntimeError(
            f"prefill returned unexpected value: {prefill_response!r}")
    kv_params = prefill_response.get("kv_transfer_params")
    if not isinstance(kv_params, dict) or not kv_params:
        raise RuntimeError(
            "prefill response has no non-empty kv_transfer_params")

    decode_payload = base_payload(prompt_ids, MAX_TOKENS)
    decode_payload.update(
        stream=True,
        stream_options={"include_usage": True},
        kv_transfer_params=kv_params,
    )
    decoded = post_stream(
        DECODE_URL + "/v1/completions",
        decode_payload,
        request_id,
        dp_rank,
    )
    usage = decoded.pop("usage")
    prompt_details = usage.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    result = {
        "request_id": request_id,
        "index": index,
        "prompt_len": prompt_len,
        "dp_rank": dp_rank,
        "prefill_latency_s": after_prefill - started,
        "cached_tokens": prompt_details.get("cached_tokens"),
        **decoded,
    }
    validation_errors = []
    expected = {
        "prompt_tokens": (usage.get("prompt_tokens"), prompt_len),
        "completion_tokens": (result["completion_tokens"], MAX_TOKENS),
        "cached_tokens": (result["cached_tokens"], prompt_len - 1),
        "finish_reason": (result["finish_reason"], "length"),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            validation_errors.append(f"{name}={actual!r} expected={wanted!r}")
    text = result["text_excerpt"]
    if not text:
        validation_errors.append("empty response text")
    if "\ufffd" in text:
        validation_errors.append("response contains U+FFFD")
    if INVALID_CONTROL_RE.search(text):
        validation_errors.append("response contains invalid control character")
    result["validation_errors"] = validation_errors
    return result


def run_round(
    assignments: list[int],
    prompts: dict[int, list[int]],
    phase: str,
    round_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    before_text = get_text(DECODE_URL + "/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{phase}-r{round_index}"
    (output_dir / f"{stem}-metrics-before.prom").write_text(before_text)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    barrier = threading.Barrier(len(assignments))
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(assignments)) as executor:
        futures = {
            executor.submit(
                run_one,
                index,
                prompt_len,
                prompts[prompt_len],
                phase,
                round_index,
                barrier,
            ):
            index
            for index, prompt_len in enumerate(assignments)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append({
                    "index": index,
                    "prompt_len": assignments[index],
                    "dp_rank": index % DP_SIZE,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
    ended = time.perf_counter()
    after_text = get_text(DECODE_URL + "/metrics")
    (output_dir / f"{stem}-metrics-after.prom").write_text(after_text)
    output_tokens = sum(item["completion_tokens"] for item in records)
    result = {
        "phase":
        phase,
        "round":
        round_index,
        "assignments":
        assignments,
        "concurrency":
        len(assignments),
        "wall_time_s":
        ended - started,
        "successful_requests":
        len(records),
        "failed_requests":
        len(errors),
        "validation_error_requests":
        sum(bool(item["validation_errors"]) for item in records),
        "aggregate_output_tokens":
        output_tokens,
        "aggregate_output_tokens_per_s":
        output_tokens / (ended - started),
        "mtp":
        acceptance_delta(before_text, after_text),
        "requests":
        sorted(records, key=lambda item: item["index"]),
        "errors":
        errors,
    }
    (output_dir / f"{stem}.json"
     ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def round_is_valid(result: dict[str, Any]) -> bool:
    return not result["failed_requests"] and not result[
        "validation_error_requests"]


def summarize_group(name: str, rounds: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [item for result in rounds for item in result["requests"]]
    series = {
        "aggregate_output_tokens_per_s":
        [item["aggregate_output_tokens_per_s"] for item in rounds],
        "round_wall_time_s": [item["wall_time_s"] for item in rounds],
        "request_prefill_latency_s":
        [item["prefill_latency_s"] for item in requests],
        "request_ttft_s": [item["ttft_s"] for item in requests],
        "request_mean_tpot_ms": [item["mean_tpot_ms"] for item in requests],
        "mtp_accept_rate": [item["mtp"]["accept_rate"] for item in rounds],
        "mtp_mean_acceptance_length":
        [item["mtp"]["mean_acceptance_length"] for item in rounds],
    }
    summary = {
        "name":
        name,
        "measured_rounds":
        MEASURED_ROUNDS,
        "executions":
        len(rounds),
        "successful_requests":
        len(requests),
        "failed_requests":
        sum(item["failed_requests"] for item in rounds),
        "validation_error_requests":
        sum(bool(item["validation_errors"]) for item in requests),
        "requests_by_prompt_len": {
            str(prompt_len):
            sum(item["prompt_len"] == prompt_len for item in requests)
            for prompt_len in PROMPT_LENS
        },
        **{
            key: distribution([value for value in values if value is not None])
            for key, values in series.items()
        },
        "mtp_per_position_accept_rate": {
            position:
            distribution([
                item["mtp"]["per_position_accept_rate"][position]
                for item in rounds if item["mtp"]
                ["per_position_accept_rate"].get(position) is not None
            ])
            for position in sorted({
                position
                for item in rounds
                for position in item["mtp"]["per_position_accept_rate"]
            })
        },
        "mtp_rounds": [item["mtp"] for item in rounds],
    }
    return summary


def single_assignments() -> list[tuple[int, int]]:
    return [(round_index, prompt_len) for round_index in range(MEASURED_ROUNDS)
            for prompt_len in PROMPT_LENS]


def multi_assignments(round_index: int) -> list[int]:
    return [
        PROMPT_LENS[(index // DP_SIZE + index % DP_SIZE + round_index) %
                    len(PROMPT_LENS)] for index in range(MULTI_CONCURRENCY)
    ]


def write_summary(
    started_unix_s: float,
    group_status: dict[str, str],
    groups: dict[str, Any],
    smoke: dict[str, Any] | None = None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 3,
        "prefill_url": PREFILL_URL,
        "decode_url": DECODE_URL,
        "model": MODEL,
        "speculative_tokens": 3,
        "prompt_lens": PROMPT_LENS,
        "max_tokens": MAX_TOKENS,
        "measured_rounds": MEASURED_ROUNDS,
        "dp_size": DP_SIZE,
        "multi_concurrency": MULTI_CONCURRENCY,
        "started_unix_s": started_unix_s,
        "updated_unix_s": time.time(),
        "group_status": group_status,
        "groups": groups,
        "smoke": smoke,
    }
    (OUT_DIR / "summary.json"
     ).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_group(
    name: str,
    cases: list[tuple[str, int, list[int]]],
    prompts: dict[int, list[int]],
    started: float,
    statuses: dict[str, str],
    groups: dict[str, Any],
    smoke: dict[str, Any],
) -> None:
    results = []
    statuses[name] = "running"
    write_summary(started, statuses, groups, smoke)
    try:
        for phase, round_index, assignments in cases:
            result = run_round(assignments, prompts, phase, round_index,
                               OUT_DIR / name)
            results.append(result)
            if not round_is_valid(result):
                raise RuntimeError(f"{name} failed; see {OUT_DIR / name}")
    except Exception:
        statuses[name] = "failed"
        write_summary(started, statuses, groups, smoke)
        raise
    groups[name] = summarize_group(name, results)
    statuses[name] = "passed"
    (OUT_DIR / name / "summary.json"
     ).write_text(json.dumps(groups[name], indent=2, sort_keys=True) + "\n")


def main() -> None:
    started = time.time()
    statuses = {
        "smoke": "running",
        "single": "not_reached",
        "multi": "not_reached"
    }
    groups: dict[str, Any] = {}
    write_summary(started, statuses, groups)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompts = {
        prompt_len: render_prompt(exact_prompt(tokenizer, prompt_len))
        for prompt_len in PROMPT_LENS
    }

    smoke: dict[str, Any] | None = None
    try:
        smoke = run_round([128], prompts, "smoke", 0, OUT_DIR / "smoke")
        if not round_is_valid(smoke):
            raise RuntimeError(f"smoke failed; see {OUT_DIR / 'smoke'}")
    except Exception:
        statuses["smoke"] = "failed"
        write_summary(started, statuses, groups, smoke)
        raise
    statuses["smoke"] = "passed"

    run_group(
        "single",
        [(f"single-p{prompt_len}", round_index, [prompt_len])
         for round_index, prompt_len in single_assignments()],
        prompts,
        started,
        statuses,
        groups,
        smoke,
    )
    run_group(
        "multi",
        [("multi", round_index, multi_assignments(round_index))
         for round_index in range(MEASURED_ROUNDS)],
        prompts,
        started,
        statuses,
        groups,
        smoke,
    )
    write_summary(started, statuses, groups, smoke)
    print(json.dumps(groups, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
