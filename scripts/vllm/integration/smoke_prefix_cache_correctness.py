#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

SHORT_QA = [
    ("math_add", "What is 7+8? Answer only the number.", "15"),
    ("math_sub", "What is 20 minus 6? Answer only the number.", "14"),
    ("math_mul", "What is 9 times 4? Answer only the number.", "36"),
    ("math_div", "What is 81 divided by 9? Answer only the number.", "9"),
    ("fact_capital_fr",
     "What is the capital of France? Answer only the city name.", "Paris"),
    ("fact_capital_jp",
     "What is the capital of Japan? Answer only the city name.", "Tokyo"),
    ("fact_planet",
     "Which planet is known as the Red Planet? Answer only the planet name.",
     "Mars"),
    ("fact_ocean",
     "What is the largest ocean on Earth? Answer only the ocean name.",
     "Pacific Ocean"),
    ("color_banana", "What color is a ripe banana usually? Answer one word.",
     "Yellow"),
    ("color_grass", "What color is grass usually? Answer one word.", "Green"),
    ("translate_cn",
     "Translate to Chinese: good morning. Answer only the translation.",
     "早上好。"),
    ("translate_en", "Translate to English: 你好. Answer only the translation.",
     "Hello."),
    ("yes_no", "Is ice cold? Answer yes or no.", "Yes"),
    ("count", "Count from 1 to 5, separated by commas.", "1, 2, 3, 4, 5"),
    ("spell", "Spell the word cat in uppercase letters.", "CAT"),
    ("compare", "Which is larger, 12 or 21? Answer only the larger number.",
     "21"),
    ("weekday", "How many days are in a week? Answer only the number.", "7"),
    ("season",
     "In the Northern Hemisphere, which season comes after winter? Answer one word.",
     "Spring"),
    ("shape", "How many sides does a triangle have? Answer only the number.",
     "3"),
]

BAD_MARKERS = ["<think>", "Thinking Process", "====", "\ufffd"]


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer: {raw}") from exc


def normalize(text: str) -> str:
    return text.strip().rstrip(".").lower()


def cached_tokens_from_usage(usage: dict) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"),
                                                int):
        return details["cached_tokens"]
    return 0


def post_json(url: str, payload: dict,
              timeout: int) -> tuple[int, dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        elapsed = time.time() - start
        body = error.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw_body": body}
        return error.code, parsed, elapsed
    with response:
        body = response.read().decode()
        elapsed = time.time() - start
        return response.status, json.loads(body), elapsed


def chat(url: str, model: str, prompt: str, max_tokens: int,
         timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }
    status, parsed, elapsed = post_json(url, payload, timeout)
    result = {
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "raw": parsed,
    }
    if status != 200:
        result["ok"] = False
        return result
    choice = parsed["choices"][0]
    usage = parsed.get("usage", {})
    result.update({
        "finish": choice.get("finish_reason"),
        "text": choice.get("message", {}).get("content", "").strip(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cached_tokens_from_usage(usage),
    })
    return result


def build_repeat_prompt(namespace: str, line_count: int, expected: str) -> str:
    lines = [f"Prefix cache namespace={namespace}."]
    lines.extend((f"Prefix cache fixture row {i:03d}: color=blue, city=Paris, "
                  f"route=alpha-delta, answer-code={expected}.")
                 for i in range(line_count))
    lines.append(
        f"Using the fixture above, answer exactly {expected} and no other text."
    )
    return "\n".join(lines)


def run_quick_probe(url: str, model: str, timeout: int) -> int:
    print("=== QUICK PROBE ===", flush=True)
    result = chat(url, model, "What is 2+2? Answer only the number.", 16,
                  timeout)
    ok = (result.get("status") == 200 and result.get("finish") == "stop"
          and normalize(result.get("text", "")) == "4")
    print(json.dumps({
        k: v
        for k, v in result.items() if k != "raw"
    },
                     ensure_ascii=False),
          flush=True)
    print(f"QUICK_PROBE_OK {int(ok)}")
    return 0 if ok else 1


def run_short_qa(url: str, model: str, timeout: int) -> tuple[int, int]:
    failures = 0
    requests = 0
    print("\n=== SHORT QA ===", flush=True)
    for index, (qid, prompt, expected) in enumerate(SHORT_QA, 1):
        result = chat(url, model, prompt, 64, timeout)
        text = result.get("text", "")
        ok = (result.get("status") == 200 and result.get("finish") == "stop"
              and normalize(text) == normalize(expected)
              and all(marker not in text for marker in BAD_MARKERS))
        requests += 1
        failures += int(not ok)
        print(
            json.dumps(
                {
                    "index": index,
                    "case": qid,
                    "expected": expected,
                    "status": result.get("status"),
                    "finish": result.get("finish"),
                    "elapsed_s": result.get("elapsed_s"),
                    "cached_tokens": result.get("cached_tokens"),
                    "text": text,
                    "ok": ok,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(f"SHORT_QA_REQUESTS {requests}")
    print(f"SHORT_QA_FAILURES {failures}")
    return requests, failures


def run_repeat_consistency(
    url: str,
    model: str,
    namespace: str,
    line_count: int,
    repeats: int,
    expected: str,
    max_tokens: int,
    timeout: int,
) -> tuple[int, int]:
    failures = 0
    outputs = []
    cached_values = []
    prompt_values = []
    prompt = build_repeat_prompt(namespace, line_count, expected)
    print(f"\n=== REPEAT CONSISTENCY lines={line_count} repeats={repeats} ===",
          flush=True)
    for index in range(1, repeats + 1):
        result = chat(url, model, prompt, max_tokens, timeout)
        text = result.get("text", "")
        cached = int(result.get("cached_tokens") or 0)
        # The first request populates the cache; only repeats can hit it.
        ok = (result.get("status") == 200 and result.get("finish") == "stop"
              and text == expected and (cached > 0 or index == 1)
              and all(marker not in text for marker in BAD_MARKERS))
        outputs.append(text)
        cached_values.append(cached)
        if isinstance(result.get("prompt_tokens"), int):
            prompt_values.append(result["prompt_tokens"])
        failures += int(not ok)
        print(
            json.dumps(
                {
                    "index": index,
                    "status": result.get("status"),
                    "finish": result.get("finish"),
                    "elapsed_s": result.get("elapsed_s"),
                    "prompt_tokens": result.get("prompt_tokens"),
                    "cached_tokens": cached,
                    "text": text,
                    "expected": expected,
                    "ok": ok,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    unique_outputs = sorted(set(outputs))
    if unique_outputs != [expected]:
        failures += 1
        print(
            f"REPEAT_OUTPUT_DRIFT lines={line_count} unique={unique_outputs!r}"
        )
    print(f"REPEAT_LINES {line_count}")
    print(f"REPEAT_REQUESTS {repeats}")
    print(
        f"REPEAT_PROMPT_TOKENS_MIN {min(prompt_values) if prompt_values else 0}"
    )
    print(
        f"REPEAT_PROMPT_TOKENS_MAX {max(prompt_values) if prompt_values else 0}"
    )
    print(
        f"REPEAT_CACHED_TOKENS_MIN {min(cached_values) if cached_values else 0}"
    )
    print(
        f"REPEAT_CACHED_TOKENS_MAX {max(cached_values) if cached_values else 0}"
    )
    print(f"REPEAT_FAILURES {failures}")
    return repeats, failures


def build_shared_prefix(namespace: str, line_count: int) -> str:
    return "\n".join(
        f"{namespace} row {i:03d}: alpha=ALPHA-314 bravo=BRAVO-271 charlie=CHARLIE-159."
        for i in range(line_count))


def run_long_shared_prefix_cross_query(
    url: str,
    model: str,
    namespace: str,
    line_count: int,
    rounds: int,
    max_tokens: int,
    timeout: int,
) -> tuple[int, int]:
    answers = {
        "alpha": "ALPHA-314",
        "bravo": "BRAVO-271",
        "charlie": "CHARLIE-159",
    }
    sequence = [
        ("alpha-prime", "alpha"),
        ("alpha-repeat", "alpha"),
        ("bravo-different", "bravo"),
        ("alpha-after-bravo", "alpha"),
        ("charlie-different", "charlie"),
        ("bravo-repeat", "bravo"),
    ]
    common_prefix = build_shared_prefix(namespace, line_count)
    failures = 0
    requests = 0
    outputs_by_key: defaultdict[str, list[str]] = defaultdict(list)
    prompt_values = []
    cached_values = []

    print(
        f"\n=== LONG SHARED PREFIX CROSS QUERY lines={line_count} rounds={rounds} ===",
        flush=True,
    )
    for round_index in range(1, rounds + 1):
        for name, key in sequence:
            expected = answers[key]
            prompt = (
                common_prefix +
                f"\n\nQuery key: {key}. Answer exactly {expected} and no other text."
            )
            result = chat(url, model, prompt, max_tokens, timeout)
            text = result.get("text", "")
            cached = int(result.get("cached_tokens") or 0)
            # Round 1 writes the shared prefix into the cache; from round 2
            # every query should resume from it.
            ok = (result.get("status") == 200
                  and result.get("finish") == "stop" and text == expected
                  and (cached > 0 or round_index == 1)
                  and all(marker not in text for marker in BAD_MARKERS))
            requests += 1
            failures += int(not ok)
            outputs_by_key[key].append(text)
            cached_values.append(cached)
            if isinstance(result.get("prompt_tokens"), int):
                prompt_values.append(result["prompt_tokens"])
            print(
                json.dumps(
                    {
                        "round": round_index,
                        "case": name,
                        "key": key,
                        "expected": expected,
                        "status": result.get("status"),
                        "finish": result.get("finish"),
                        "elapsed_s": result.get("elapsed_s"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "cached_tokens": cached,
                        "completion_tokens": result.get("completion_tokens"),
                        "text": text,
                        "ok": ok,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for key, expected in answers.items():
        unique = sorted(set(outputs_by_key[key]))
        if unique != [expected]:
            failures += 1
            print(f"LONG_SHARED_OUTPUT_DRIFT key={key} unique={unique!r} "
                  f"expected={expected!r}")

    print(f"LONG_SHARED_REQUESTS {requests}")
    print(f"LONG_SHARED_FAILURES {failures}")
    print(
        f"LONG_SHARED_PROMPT_TOKENS_MIN {min(prompt_values) if prompt_values else 0}"
    )
    print(
        f"LONG_SHARED_PROMPT_TOKENS_MAX {max(prompt_values) if prompt_values else 0}"
    )
    print(
        f"LONG_SHARED_CACHED_TOKENS_MIN {min(cached_values) if cached_values else 0}"
    )
    print(
        f"LONG_SHARED_CACHED_TOKENS_MAX {max(cached_values) if cached_values else 0}"
    )
    print("LONG_SHARED_UNIQUE_BY_KEY " +
          json.dumps({
              k: sorted(set(v))
              for k, v in outputs_by_key.items()
          },
                     ensure_ascii=False,
                     sort_keys=True))
    print(f"LONG_SHARED_PREFIX_CROSS_QUERY_OK {int(failures == 0)}")
    return requests, failures


def run_mixed_query_correctness(
    url: str,
    model: str,
    namespace: str,
    line_count: int,
    rounds: int,
    max_tokens: int,
    timeout: int,
) -> tuple[int, int]:
    answers = {
        "alpha": "ALPHA-314",
        "bravo": "BRAVO-271",
        "charlie": "CHARLIE-159",
    }
    sequence = [
        (
            "short_math_add",
            "short",
            "What is 7+8? Answer only the number.",
            "15",
        ),
        ("long_alpha", "long", "alpha", answers["alpha"]),
        (
            "short_capital_fr",
            "short",
            "What is the capital of France? Answer only the city name.",
            "Paris",
        ),
        ("long_bravo", "long", "bravo", answers["bravo"]),
        (
            "short_compare",
            "short",
            "Which is larger, 12 or 21? Answer only the larger number.",
            "21",
        ),
        ("long_charlie", "long", "charlie", answers["charlie"]),
    ]
    common_prefix = build_shared_prefix(namespace, line_count)
    failures = 0
    requests = 0
    outputs_by_case: defaultdict[str, list[str]] = defaultdict(list)
    cached_values = []

    print(
        f"\n=== MIXED QUERY CORRECTNESS lines={line_count} rounds={rounds} ===",
        flush=True,
    )
    for round_index in range(1, rounds + 1):
        for name, kind, prompt_or_key, expected in sequence:
            if kind == "long":
                key = prompt_or_key
                prompt = (common_prefix +
                          f"\n\nQuery key: {key}. Answer exactly {expected} "
                          "and no other text.")
            else:
                prompt = prompt_or_key

            result = chat(url, model, prompt, max_tokens, timeout)
            text = result.get("text", "")
            cached = int(result.get("cached_tokens") or 0)
            # Long prompts share a prefix, so from round 2 they must hit;
            # the short ones interleaved between them never can.
            ok = (result.get("status") == 200
                  and result.get("finish") == "stop"
                  and normalize(text) == normalize(expected)
                  and (cached > 0 or kind != "long" or round_index == 1)
                  and all(marker not in text for marker in BAD_MARKERS))
            requests += 1
            failures += int(not ok)
            outputs_by_case[name].append(text)
            if kind == "long":
                cached_values.append(cached)

            print(
                json.dumps(
                    {
                        "round": round_index,
                        "case": name,
                        "kind": kind,
                        "expected": expected,
                        "status": result.get("status"),
                        "finish": result.get("finish"),
                        "elapsed_s": result.get("elapsed_s"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "cached_tokens": cached,
                        "completion_tokens": result.get("completion_tokens"),
                        "text": text,
                        "ok": ok,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for name, _kind, _prompt_or_key, expected in sequence:
        unique = sorted({normalize(value) for value in outputs_by_case[name]})
        if unique != [normalize(expected)]:
            failures += 1
            print(f"MIXED_OUTPUT_DRIFT case={name} "
                  f"unique={unique!r} expected={expected!r}")

    print(f"MIXED_QUERY_REQUESTS {requests}")
    print(f"MIXED_QUERY_FAILURES {failures}")
    print("MIXED_QUERY_LONG_CACHED_TOKENS_MIN "
          f"{min(cached_values) if cached_values else 0}")
    print("MIXED_QUERY_LONG_CACHED_TOKENS_MAX "
          f"{max(cached_values) if cached_values else 0}")
    print("MIXED_QUERY_UNIQUE_BY_CASE " + json.dumps(
        {
            k: sorted(set(v))
            for k, v in outputs_by_case.items()
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
    print(f"MIXED_QUERY_CORRECTNESS_OK {int(failures == 0)}")
    return requests, failures


def run_concurrent_mixed_query_correctness(
    url: str,
    model: str,
    namespace: str,
    line_count: int,
    rounds: int,
    concurrency: int,
    max_tokens: int,
    timeout: int,
) -> tuple[int, int]:
    answers = {
        "alpha": "ALPHA-314",
        "bravo": "BRAVO-271",
        "charlie": "CHARLIE-159",
    }
    keys = tuple(answers)
    common_prefix = build_shared_prefix(namespace, line_count)
    failures = 0
    requests = 0
    outputs_by_key: defaultdict[str, list[str]] = defaultdict(list)
    cached_values = []
    worker_count = max(2, int(concurrency))

    print(
        f"\n=== CONCURRENT MIXED QUERY CORRECTNESS lines={line_count} "
        f"rounds={rounds} concurrency={worker_count} ===",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for round_index in range(1, rounds + 1):
            future_to_case = {}
            for index in range(worker_count):
                key = keys[index % len(keys)]
                expected = answers[key]
                prompt = (common_prefix +
                          f"\n\nQuery key: {key}. Answer exactly {expected} "
                          "and no other text.")
                future = executor.submit(chat, url, model, prompt, max_tokens,
                                         timeout)
                future_to_case[future] = (round_index, key, expected)
            for future in as_completed(future_to_case):
                round_index, key, expected = future_to_case[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "exception",
                        "finish": None,
                        "text": str(exc),
                        "cached_tokens": 0,
                    }
                text = result.get("text", "")
                cached = int(result.get("cached_tokens") or 0)
                ok = (result.get("status") == 200
                      and result.get("finish") == "stop" and text == expected
                      and cached > 0 and all(marker not in text
                                             for marker in BAD_MARKERS))
                requests += 1
                failures += int(not ok)
                outputs_by_key[key].append(text)
                cached_values.append(cached)
                print(
                    json.dumps(
                        {
                            "round": round_index,
                            "key": key,
                            "expected": expected,
                            "status": result.get("status"),
                            "finish": result.get("finish"),
                            "elapsed_s": result.get("elapsed_s"),
                            "prompt_tokens": result.get("prompt_tokens"),
                            "cached_tokens": cached,
                            "completion_tokens":
                            result.get("completion_tokens"),
                            "text": text,
                            "ok": ok,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    for key, values in outputs_by_key.items():
        expected = answers[key]
        unique = sorted(set(values))
        if unique != [expected]:
            failures += 1
            print(f"CONCURRENT_MIXED_OUTPUT_DRIFT key={key} "
                  f"unique={unique!r} expected={expected!r}")

    print(f"CONCURRENT_MIXED_QUERY_REQUESTS {requests}")
    print(f"CONCURRENT_MIXED_QUERY_FAILURES {failures}")
    print("CONCURRENT_MIXED_QUERY_CACHED_TOKENS_MIN "
          f"{min(cached_values) if cached_values else 0}")
    print("CONCURRENT_MIXED_QUERY_CACHED_TOKENS_MAX "
          f"{max(cached_values) if cached_values else 0}")
    print("CONCURRENT_MIXED_QUERY_UNIQUE_BY_KEY " + json.dumps(
        {
            k: sorted(set(v))
            for k, v in outputs_by_key.items()
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
    print(f"CONCURRENT_MIXED_QUERY_CORRECTNESS_OK {int(failures == 0)}")
    return requests, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "End-to-end correctness smoke for disaggregated serving with prefix caching."
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("PROXY_PORT", "8000"))
    parser.add_argument("--model",
                        default=os.environ.get("SERVED_MODEL_NAME",
                                               "Qwen3.5-35B-A3B-FP8"))
    parser.add_argument("--timeout",
                        type=int,
                        default=env_int("P4D2_CORRECTNESS_TIMEOUT", 600))
    parser.add_argument("--quick-probe-only", action="store_true")
    parser.add_argument("--skip-short-qa", action="store_true")
    parser.add_argument("--short-repeat-lines",
                        type=int,
                        default=env_int("P4D2_SHORT_REPEAT_LINES", 48))
    parser.add_argument("--short-repeat-count",
                        type=int,
                        default=env_int("P4D2_SHORT_REPEAT_COUNT", 5))
    parser.add_argument("--long-repeat-lines",
                        type=int,
                        default=env_int("P4D2_LONG_REPEAT_LINES", 96))
    parser.add_argument("--long-repeat-count",
                        type=int,
                        default=env_int("P4D2_LONG_REPEAT_COUNT", 4))
    parser.add_argument("--long-shared-lines",
                        type=int,
                        default=env_int("P4D2_LONG_SHARED_LINES", 60))
    parser.add_argument("--long-shared-rounds",
                        type=int,
                        default=env_int("P4D2_LONG_SHARED_ROUNDS", 3))
    parser.add_argument("--skip-mixed-query", action="store_true")
    parser.add_argument("--mixed-lines",
                        type=int,
                        default=env_int("P4D2_MIXED_LINES", 60))
    parser.add_argument("--mixed-rounds",
                        type=int,
                        default=env_int("P4D2_MIXED_ROUNDS", 2))
    parser.add_argument("--concurrent-requests",
                        type=int,
                        default=env_int("P4D2_CONCURRENT_REQUESTS", 0))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    if args.quick_probe_only:
        return run_quick_probe(url, args.model, args.timeout)

    namespace = f"pc-correctness-{int(time.time())}"
    total_requests = 0
    failures = 0

    if not args.skip_short_qa:
        requests, suite_failures = run_short_qa(url, args.model, args.timeout)
        total_requests += requests
        failures += suite_failures

    requests, suite_failures = run_repeat_consistency(
        url,
        args.model,
        f"{namespace}-short-repeat",
        args.short_repeat_lines,
        args.short_repeat_count,
        "NORTH-17",
        16,
        args.timeout,
    )
    total_requests += requests
    failures += suite_failures

    requests, suite_failures = run_repeat_consistency(
        url,
        args.model,
        f"{namespace}-long-repeat",
        args.long_repeat_lines,
        args.long_repeat_count,
        "NORTH-17",
        8,
        args.timeout,
    )
    total_requests += requests
    failures += suite_failures

    requests, suite_failures = run_long_shared_prefix_cross_query(
        url,
        args.model,
        f"{namespace}-long-shared",
        args.long_shared_lines,
        args.long_shared_rounds,
        16,
        args.timeout,
    )
    total_requests += requests
    failures += suite_failures

    if not args.skip_mixed_query:
        requests, suite_failures = run_mixed_query_correctness(
            url,
            args.model,
            f"{namespace}-mixed",
            args.mixed_lines,
            args.mixed_rounds,
            16,
            args.timeout,
        )
        total_requests += requests
        failures += suite_failures
        if args.concurrent_requests >= 2:
            requests, suite_failures = run_concurrent_mixed_query_correctness(
                url,
                args.model,
                f"{namespace}-cmix",
                args.mixed_lines,
                args.mixed_rounds,
                args.concurrent_requests,
                16,
                args.timeout,
            )
            total_requests += requests
            failures += suite_failures

    print("\n=== PREFIX CACHE CORRECTNESS SUMMARY ===")
    print(f"TOTAL_REQUESTS {total_requests}")
    print(f"FAILURES {failures}")
    print(f"PREFIX_CACHE_CORRECTNESS_OK {int(failures == 0)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
