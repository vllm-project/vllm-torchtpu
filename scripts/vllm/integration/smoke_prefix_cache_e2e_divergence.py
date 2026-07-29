#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer: {raw}") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a float: {raw}") from exc


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


def cached_tokens_from_usage(usage: dict) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"),
                                                int):
        return details["cached_tokens"]
    return 0


def generated_logprobs(choice: dict) -> list[dict]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list):
        return []

    items = []
    for item in content:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        logprob = item.get("logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)):
            items.append({
                "token": token,
                "logprob": float(logprob),
            })
    return items


def chat(url: str, model: str, prompt: str, max_tokens: int,
         timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }
    status, parsed, elapsed = post_json(url, payload, timeout)
    result = {
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "raw": parsed,
    }
    if status != 200:
        return result

    choice = parsed["choices"][0]
    usage = parsed.get("usage", {})
    result.update({
        "finish": choice.get("finish_reason"),
        "text": choice.get("message", {}).get("content", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": cached_tokens_from_usage(usage),
        "logprobs": generated_logprobs(choice),
    })
    return result


def build_last_key_prompt(salt: str, line_count: int) -> str:
    lines = [
        f"Prefix-cache E2E divergence probe salt={salt}.",
        "Remember the final code line. At the end output only the code value.",
    ]
    for index in range(line_count):
        value = (index * 7919 + 1234) % 10000
        distractor = (index * 17 + len(salt)) % 10000
        lines.append(f"code line {index:03d}: value={value:04d}; "
                     f"distractor={distractor:04d}; salt={salt}.")
    lines.append(
        "What is the value on the final code line? Output only the digits.")
    return "\n".join(lines)


def summarize(name: str, result: dict) -> dict:
    return {
        "name": name,
        "status": result.get("status"),
        "finish": result.get("finish"),
        "elapsed_s": result.get("elapsed_s"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "cached_tokens": result.get("cached_tokens"),
        "text": result.get("text"),
        "logprobs": result.get("logprobs"),
    }


def output_diff(cold: dict, warm: dict, logprob_atol: float) -> dict:
    cold_items = cold.get("logprobs") or []
    warm_items = warm.get("logprobs") or []
    cold_tokens = [item["token"] for item in cold_items]
    warm_tokens = [item["token"] for item in warm_items]
    # Track token-match diffs (for logprob_equal gating on the happy path) and
    # first-divergent-position diffs separately so a partial mismatch (e.g.
    # tokens 0..1 match, tokens 2.. diverge) does not underreport magnitude in
    # the evidence log.
    logprob_diffs = []
    first_div_index = None
    first_div_absdiff = None
    for index, (left, right) in enumerate(zip(cold_items, warm_items)):
        if left["token"] != right["token"]:
            if first_div_index is None:
                first_div_index = index
                if (math.isfinite(left["logprob"])
                        and math.isfinite(right["logprob"])):
                    first_div_absdiff = abs(left["logprob"] - right["logprob"])
            continue
        if math.isfinite(left["logprob"]) and math.isfinite(right["logprob"]):
            logprob_diffs.append(abs(left["logprob"] - right["logprob"]))
    if len(cold_tokens) != len(warm_tokens) and first_div_index is None:
        first_div_index = min(len(cold_tokens), len(warm_tokens))
    max_logprob_absdiff = max(logprob_diffs) if logprob_diffs else float("inf")
    return {
        "text_equal": cold.get("text") == warm.get("text"),
        "tokens_equal": cold_tokens == warm_tokens,
        "has_logprobs": bool(cold_items) and bool(warm_items),
        "max_logprob_absdiff": max_logprob_absdiff,
        "logprob_equal": max_logprob_absdiff <= logprob_atol,
        "first_divergent_index": first_div_index,
        "first_divergent_logprob_absdiff": first_div_absdiff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Output-level E2E repro for prefix-cache warm-hit drift. The test "
        "treats cold prefill and warm prefix-cache hit as externally equivalent "
        "paths and compares API text/token/logprob output."))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("PROXY_PORT", "8000"))
    parser.add_argument("--model",
                        default=os.environ.get("SERVED_MODEL_NAME",
                                               "Qwen3.5-35B-A3B-FP8"))
    parser.add_argument("--timeout",
                        type=int,
                        default=env_int("PREFIX_E2E_DIVERGENCE_TIMEOUT", 600))
    parser.add_argument("--attempts",
                        type=int,
                        default=env_int("PREFIX_E2E_DIVERGENCE_ATTEMPTS", 8))
    parser.add_argument("--line-count",
                        type=int,
                        default=env_int("PREFIX_E2E_DIVERGENCE_LINES", 72))
    parser.add_argument("--max-tokens",
                        type=int,
                        default=env_int("PREFIX_E2E_DIVERGENCE_MAX_TOKENS",
                                        32))
    # 1e-3 is loose enough to absorb the FP8/TPU numerical noise cold prefill
    # and warm cascade-attention produce on identical prompts (typically
    # 1e-5..1e-4), and still ~2 orders of magnitude tighter than the drift the
    # bug this smoke targets produces (>>0.1; see the PR that added the fix).
    parser.add_argument("--logprob-atol",
                        type=float,
                        default=env_float("PREFIX_E2E_DIVERGENCE_LOGPROB_ATOL",
                                          1e-3))
    parser.add_argument("--salt-prefix",
                        default=os.environ.get(
                            "PREFIX_E2E_DIVERGENCE_SALT_PREFIX",
                            "prefix-e2e-divergence"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise SystemExit("--attempts must be positive")
    if args.line_count < 1:
        raise SystemExit("--line-count must be positive")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    run_id = int(time.time() * 1000)
    hit_attempts = 0

    print("=== PREFIX CACHE E2E DIVERGENCE REPRO ===", flush=True)
    print(f"ATTEMPTS {args.attempts}", flush=True)
    print(f"LINE_COUNT {args.line_count}", flush=True)
    print(f"MAX_TOKENS {args.max_tokens}", flush=True)

    for attempt in range(args.attempts):
        salt = f"{args.salt_prefix}-{run_id}-{attempt:02d}"
        prompt = build_last_key_prompt(salt, args.line_count)

        print(f"\n=== ATTEMPT {attempt + 1}/{args.attempts} salt={salt} ===",
              flush=True)
        cold = chat(url, args.model, prompt, args.max_tokens, args.timeout)
        print(json.dumps(summarize("cold", cold), ensure_ascii=False),
              flush=True)
        if cold.get("status") != 200:
            print("PREFIX_E2E_DIVERGENCE_COLD_FAILED")
            return 1

        warm = chat(url, args.model, prompt, args.max_tokens, args.timeout)
        print(json.dumps(summarize("warm", warm), ensure_ascii=False),
              flush=True)
        if warm.get("status") != 200:
            print("PREFIX_E2E_DIVERGENCE_WARM_FAILED")
            return 1

        cached_tokens = int(warm.get("cached_tokens") or 0)
        if cached_tokens > 0:
            hit_attempts += 1

        diff = output_diff(cold, warm, args.logprob_atol)
        evidence = {
            "attempt": attempt,
            "salt": salt,
            "prompt_tokens": warm.get("prompt_tokens"),
            "warm_cached_tokens": cached_tokens,
            **diff,
        }
        print("PREFIX_E2E_DIVERGENCE_EVIDENCE " +
              json.dumps(evidence, ensure_ascii=False, sort_keys=True),
              flush=True)

        if cached_tokens <= 0:
            print("PREFIX_E2E_DIVERGENCE_LOCAL_HIT_MISSING")
            continue

        if (not diff["text_equal"] or not diff["tokens_equal"]
                or not diff["has_logprobs"] or not diff["logprob_equal"]):
            print("PREFIX_E2E_DIVERGENCE_BUG_CONFIRMED 1")
            return 1

    if hit_attempts == 0:
        print("PREFIX_E2E_DIVERGENCE_NO_LOCAL_HIT_ATTEMPTS")
        return 1

    print("PREFIX_E2E_DIVERGENCE_BUG_CONFIRMED 0")
    print("PREFIX_E2E_DIVERGENCE_OK 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
