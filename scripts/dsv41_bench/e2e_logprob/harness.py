#!/usr/bin/env python3
"""Collect greedy top-k logprobs from a served model and compare configurations.

One `collect` invocation targets one endpoint (a vLLM TPU serve, or the
official DeepSeek API) and writes one JSONL. `compare` then lines up the JSONLs
per prompt and reports how far each configuration is from a reference.

The official gateway does not return real logprobs (it fills the sampled token
with 0.0 and every alternative with -9999.0, on both `deepseek-flash` and
`deepseek-v4-pro`, streaming or not), so `collect` on the official side is used
for its token stream only. `teacher` then scores that text under each local
configuration: it renders the prompt with the server's own `/tokenize`, appends
the official continuation, and reads `prompt_logprobs` back. Lower NLL means the
configuration assigns more likelihood to official text -- a distance that does
not need the official distribution and does not saturate when the two streams
diverge.

Usage:
    python3 harness.py collect --prompts prompts.jsonl --out p.jsonl \
        --base-url http://127.0.0.1:8000/v1 --model deepseek-v41 --local
    python3 harness.py collect --prompts prompts.jsonl --out api.jsonl \
        --base-url https://agent.infiscale-tech.com/v1 --model deepseek-flash
    python3 harness.py teacher --prompts prompts.jsonl --ref api.jsonl \
        --base-url http://127.0.0.1:8000/v1 --model deepseek-v41 --out P_teacher.jsonl
    python3 harness.py compare --ref api.jsonl --cand P=p.jsonl A=a.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OFFICIAL_KEY_ENV = "DEEPSEEK_API_KEY"


def _post(url: str, payload: dict, headers: dict, timeout: int, retries: int) -> dict:
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            last = f"HTTP {exc.code}: {body}"
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise RuntimeError(last) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = repr(exc)
            if attempt + 1 < retries:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise RuntimeError(last) from exc
    raise RuntimeError(f"gave up: {last}")


def _parse_response(body: dict) -> dict:
    choice = body["choices"][0]
    logprobs = choice.get("logprobs") or {}
    # A thinking model emits reasoning first, then the answer; generation order
    # is the concatenation in that order, so the position index stays aligned
    # with the token stream.
    content = (logprobs.get("reasoning_content") or []) + (logprobs.get("content") or [])  # noqa: E501
    tokens, token_logprobs, top_tokens, top_logprobs = [], [], [], []
    for position in content:
        tokens.append(position["token"])
        token_logprobs.append(float(position["logprob"]))
        alternatives = position.get("top_logprobs") or []
        top_tokens.append([alt["token"] for alt in alternatives])
        top_logprobs.append([float(alt["logprob"]) for alt in alternatives])
    return {
        "text": choice.get("message", {}).get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_tokens": top_tokens,
        "top_logprobs": top_logprobs,
        "model": body.get("model"),
        "usage": body.get("usage"),
    }


def _collect_one(args, prompt: dict, headers: dict) -> dict:
    payload = {
        "model": args.model,
        "messages": prompt["messages"],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
    }
    thinking = args.thinking == "on"
    if args.local:
        # Match `LM_EVAL_GEN_KWARGS`: the template variable is what the CI eval
        # flips, so the two sides share a prompt format when both are off.
        payload["chat_template_kwargs"] = {"thinking": thinking}
        if args.max_model_len:
            payload["max_tokens"] = min(args.max_tokens, args.max_model_len)
    elif not thinking:
        # The official gateway takes a struct, not the template flag; a bare
        # boolean is a 422.
        payload["thinking"] = {"type": "disabled"}
    started = time.time()
    body = _post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        payload,
        headers,
        args.timeout,
        args.retries,
    )
    record = {"id": prompt["id"], "messages": prompt["messages"]}
    record.update(_parse_response(body))
    record["elapsed_s"] = round(time.time() - started, 3)
    return record


def collect(args) -> int:
    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines()]
    done = set()
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"{out_path}: resuming with {len(done)} prompts already collected")
    headers = {}
    if not args.local:
        key = os.environ.get(args.api_key_env)
        if not key:
            sys.exit(f"{args.api_key_env} is not set; export it before the API run")
        headers["Authorization"] = f"Bearer {key}"

    todo = [p for p in prompts if p["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"collecting {len(todo)} prompts -> {out_path} ({args.base_url})")

    failures = 0
    with out_path.open("a") as sink, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(_collect_one, args, p, headers) for p in todo]
        for index, future in enumerate(futures, 1):
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures += 1
                print(f"  [{index}/{len(todo)}] FAILED: {exc}")
                continue
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(
                f"  [{index}/{len(todo)}] {record['id']}: "
                f"{len(record['tokens'])} tokens in {record['elapsed_s']}s"
            )
    print(f"done: {len(todo) - failures} ok, {failures} failed")
    return 1 if failures else 0


def _load(path: str) -> dict[str, dict]:
    return {
        record["id"]: record
        for record in (
            json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()  # noqa: E501
        )
    }


def _kl_topk(ref: dict, cand: dict, position: int) -> float | None:
    """KL(ref || cand) over the union of the two top-k supports, renormalized.

    Both sides expose only their top-k, so the union carries most but not all of
    the mass. This is a relative distance, not a calibrated KL.
    """
    ref_tokens = ref["top_tokens"][position]
    cand_tokens = cand["top_tokens"][position]
    if not ref_tokens or not cand_tokens:
        return None
    ref_probs = dict(zip(ref_tokens, ref["top_logprobs"][position], strict=True))
    cand_probs = dict(zip(cand_tokens, cand["top_logprobs"][position], strict=True))
    support = set(ref_probs) | set(cand_probs)
    ref_norm = math.log(sum(math.exp(ref_probs.get(t, -1e9)) for t in support))
    cand_norm = math.log(sum(math.exp(cand_probs.get(t, -1e9)) for t in support))
    total = 0.0
    for token in support:
        log_p = ref_probs.get(token, -1e9) - ref_norm
        log_q = cand_probs.get(token, -1e9) - cand_norm
        p = math.exp(log_p)
        if p > 0.0:
            total += p * (log_p - log_q)
    return total


def _compare_prompt(ref: dict, cand: dict) -> dict:
    ref_tokens = ref["tokens"]
    cand_tokens = cand["tokens"]
    common = min(len(ref_tokens), len(cand_tokens))
    divergence = None
    for position in range(common):
        if ref_tokens[position] != cand_tokens[position]:
            divergence = position
            break
    prefix = divergence if divergence is not None else common
    deltas = [
        cand["token_logprobs"][i] - ref["token_logprobs"][i] for i in range(prefix)
    ]
    kls = [k for i in range(prefix) if (k := _kl_topk(ref, cand, i)) is not None]
    top1 = sum(
        1
        for i in range(prefix)
        if ref["top_tokens"][i][:1] == cand["top_tokens"][i][:1]
    )
    return {
        "prefix": prefix,
        "ref_len": len(ref_tokens),
        "cand_len": len(cand_tokens),
        "divergence": divergence,
        "nll_delta": sum(deltas) / len(deltas) if deltas else None,
        "kl": sum(kls) / len(kls) if kls else None,
        "top1": top1 / prefix if prefix else None,
        "exact": ref_tokens == cand_tokens,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compare(args) -> int:
    ref_records = _load(args.ref)
    candidates = []
    for spec in args.cand:
        name, _, path = spec.partition("=")
        candidates.append((name, _load(path)))

    header = f"{'pair':<10} {'n':>4} {'pos':>6} {'div%':>6} {'nll_d':>9} {'kl':>9} {'top1':>6} {'exact':>6}"  # noqa: E501
    print(header)
    print("-" * len(header))
    for name, cand_records in candidates:
        mismatched = [
            prompt_id
            for prompt_id, ref in ref_records.items()
            if prompt_id in cand_records
            and (ref.get("usage") or {}).get("prompt_tokens")
            != (cand_records[prompt_id].get("usage") or {}).get("prompt_tokens")
        ]
        if mismatched:
            print(
                f"{name:<10} WARNING: prompt_tokens differ on {len(mismatched)} prompts "  # noqa: E501
                f"-> chat template mismatch, distances are not comparable"
            )
    for name, cand_records in candidates:
        rows = []
        for prompt_id, ref in ref_records.items():
            cand = cand_records.get(prompt_id)
            if cand is None:
                continue
            rows.append(_compare_prompt(ref, cand))
        if not rows:
            print(f"{name:<10} (no overlapping prompts)")
            continue
        positions = sum(row["prefix"] for row in rows)
        diverged = sum(1 for row in rows if row["divergence"] is not None)
        print(
            f"{name:<10} {len(rows):>4} {positions:>6} "
            f"{100 * diverged / len(rows):>5.1f}% "
            f"{_mean([r['nll_delta'] for r in rows if r['nll_delta'] is not None]) or 0:>9.2e} "  # noqa: E501
            f"{_mean([r['kl'] for r in rows if r['kl'] is not None]) or 0:>9.2e} "
            f"{_mean([r['top1'] for r in rows if r['top1'] is not None]) or 0:>6.3f} "
            f"{100 * sum(r['exact'] for r in rows) / len(rows):>5.1f}%"
        )
    if args.json:
        payload = {
            name: {
                prompt_id: _compare_prompt(ref_records[prompt_id], cand_records[prompt_id])  # noqa: E501
                for prompt_id in ref_records
                if prompt_id in cand_records
            }
            for name, cand_records in candidates
        }
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"\nper-prompt results -> {args.json}")
    return 0



class PromptMismatch(Exception):
    """The server scored different tokens than the ones sent."""


def _root(args) -> str:
    base = args.base_url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _tokenize(args, payload: dict, headers: dict) -> list[int]:
    body = _post(f"{_root(args)}/tokenize", payload, headers, args.timeout, args.retries)  # noqa: E501
    return [int(token) for token in body["tokens"]]


def _score_ids(args, ids: list[int], headers: dict) -> tuple[list[float | None], str]:
    """Return the logprob the server assigns to each position in ``ids``."""
    payload = {
        "model": args.model,
        "prompt": ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": args.prompt_logprobs,
    }
    try:
        body = _post(
            f"{args.base_url.rstrip('/')}/completions",
            payload,
            headers,
            args.timeout,
            args.retries,
        )
        choice = (body.get("choices") or [{}])[0]
        served = body.get("prompt_token_ids") or choice.get("prompt_token_ids")
        if served is not None and [int(t) for t in served] != ids:
            raise PromptMismatch(
                f"server scored {len(served)} different prompt tokens than the "
                f"{len(ids)} sent (truncation or template drift)"
            )
        entries = body.get("prompt_logprobs")
        if entries is None:
            entries = choice.get("prompt_logprobs")
        if entries is None:
            raise RuntimeError("server returned no prompt_logprobs")
        if len(entries) != len(ids):
            raise RuntimeError(
                f"prompt_logprobs has {len(entries)} entries for {len(ids)} tokens"
            )
        values: list[float | None] = []
        for position, entry in enumerate(entries):
            if not entry:
                values.append(None)
                continue
            key = str(ids[position])
            logprob = entry.get(key)
            if logprob is None:
                raise RuntimeError(
                    f"position {position}: token {key} missing from prompt_logprobs"
                )
            values.append(float(logprob["logprob"] if isinstance(logprob, dict) else logprob))  # noqa: E501
        return values, "prompt_logprobs"
    except PromptMismatch:
        raise
    except Exception as exc:  # noqa: BLE001 - fall back to the echo form
        # Older/newer servers only expose prompt logprobs through echo.
        body = _post(
            f"{args.base_url.rstrip('/')}/completions",
            {**payload, "prompt_logprobs": None, "echo": True, "logprobs": 1},
            headers,
            args.timeout,
            args.retries,
        )
        logprobs = (body.get("choices") or [{}])[0].get("logprobs") or {}
        token_logprobs = logprobs.get("token_logprobs") or []
        if len(token_logprobs) != len(ids):
            raise RuntimeError(
                f"echo returned {len(token_logprobs)} logprobs for {len(ids)} tokens"
            ) from exc
        return [None if v is None else float(v) for v in token_logprobs], "echo"


def _teacher_one(args, prompt: dict, reference: dict, headers: dict) -> dict:
    thinking = args.thinking == "on"
    if args.tokenizer:
        # Local tokenizer path: the pod has the HF files, and rendering here
        # keeps the request free of a /tokenize round trip (and of that
        # endpoint's availability).
        tokenizer = args._tokenizer
        text = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=True,
            thinking=thinking,
        )
        prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
        cont_ids = tokenizer(reference["text"], add_special_tokens=False).input_ids
    else:
        prompt_ids = _tokenize(
            args,
            {
                "messages": prompt["messages"],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"thinking": thinking},
            },
            headers,
        )
        cont_ids = _tokenize(args, {"prompt": reference["text"]}, headers)
    values, mode = _score_ids(args, prompt_ids + cont_ids, headers)
    cont_values = values[len(prompt_ids) :]
    scored = [v for v in cont_values if v is not None]
    if not scored:
        raise RuntimeError("no continuation logprobs (all null)")
    return {
        "id": prompt["id"],
        "prompt_tokens": len(prompt_ids),
        "cont_tokens": len(cont_ids),
        "ref_tokens": len(reference["tokens"])
        if isinstance(reference.get("tokens"), list)
        else int(reference.get("tokens") or 0),
        "nll_sum": -sum(scored),
        "nll_mean": -sum(scored) / len(scored),
        "scored": len(scored),
        "mode": mode,
    }


def teacher(args) -> int:
    prompts = {
        prompt["id"]: prompt
        for prompt in (
            json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()  # noqa: E501
        )
    }
    references = {
        record["id"]: record
        for record in (
            json.loads(line) for line in Path(args.ref).read_text().splitlines() if line.strip()  # noqa: E501
        )
    }
    todo = [prompt_id for prompt_id in references if prompt_id in prompts]
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        done = {
            json.loads(line)["id"]
            for line in out_path.read_text().splitlines()
            if line.strip()
        }
        todo = [prompt_id for prompt_id in todo if prompt_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    if args.tokenizer:
        from transformers import AutoTokenizer

        args._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"tokenizer loaded from {args.tokenizer}")
    else:
        args._tokenizer = None
    print(f"scoring {len(todo)} official continuations -> {out_path}")
    failures = 0
    with out_path.open("a") as sink, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [
            pool.submit(_teacher_one, args, prompts[prompt_id], references[prompt_id], {})  # noqa: E501
            for prompt_id in todo
        ]
        for index, future in enumerate(futures, 1):
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures += 1
                print(f"  [{index}/{len(todo)}] FAILED: {exc}")
                continue
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if record["cont_tokens"] != record["ref_tokens"]:
                print(
                    f"  [{index}/{len(todo)}] {record['id']}: re-tokenized "
                    f"{record['cont_tokens']} vs {record['ref_tokens']} tokens"
                )
            else:
                print(
                    f"  [{index}/{len(todo)}] {record['id']}: "
                    f"{record['nll_mean']:.4f} nats/token over {record['scored']}"
                )
    print(f"done: {len(todo) - failures} ok, {failures} failed")
    return 1 if failures else 0


def score(args) -> int:
    runs = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs.append(
            (
                name,
                {
                    json.loads(line)["id"]: json.loads(line)
                    for line in Path(path).read_text().splitlines()
                    if line.strip()
                },
            )
        )
    ids = set(runs[0][1])
    for name, records in runs[1:]:
        ids &= set(records)
    ids = sorted(ids)
    header = f"{'run':<8} {'n':>4} {'tokens':>7} {'nats/tok':>10} {'vs first':>12} {'+/-95%':>10}"  # noqa: E501
    print(header)
    print("-" * len(header))
    baseline = None
    for name, records in runs:
        pool = [records[i] for i in ids if records[i]["scored"]]
        if not pool:
            print(f"{name:<8} (no scored prompts)")
            continue
        tokens = sum(record["scored"] for record in pool)
        mean = sum(record["nll_sum"] for record in pool) / tokens
        if baseline is None:
            baseline = (name, mean)
            print(f"{name:<8} {len(pool):>4} {tokens:>7} {mean:>10.4f} {'-':>12} {'-':>10}")  # noqa: E501
            continue
        baseline_records = runs[0][1]
        deltas = [
            records[prompt_id]["nll_mean"] - baseline_records[prompt_id]["nll_mean"]
            for prompt_id in ids
            if records[prompt_id]["scored"] and baseline_records[prompt_id]["scored"]
        ]
        n = len(deltas)
        delta_mean = sum(deltas) / n
        variance = sum((d - delta_mean) ** 2 for d in deltas) / max(1, n - 1)
        half = 1.96 * math.sqrt(variance / n)
        print(
            f"{name:<8} {len(pool):>4} {tokens:>7} {mean:>10.4f} "
            f"{delta_mean:>+12.2e} {half:>10.1e}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="collect top-k logprobs")
    collect_parser.add_argument("--prompts", required=True)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--base-url", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--local", action="store_true")
    collect_parser.add_argument(
        "--api-key-env", default=OFFICIAL_KEY_ENV, help="env var holding the API key"
    )
    collect_parser.add_argument(
        "--thinking", choices=["off", "on"], default="off", help="reasoning mode"
    )
    collect_parser.add_argument("--max-tokens", type=int, default=256)
    collect_parser.add_argument("--top-logprobs", type=int, default=20)
    collect_parser.add_argument("--max-model-len", type=int, default=0)
    collect_parser.add_argument("--concurrency", type=int, default=8)
    collect_parser.add_argument("--timeout", type=int, default=1800)
    collect_parser.add_argument("--retries", type=int, default=5)
    collect_parser.add_argument("--limit", type=int, default=0)
    collect_parser.add_argument("--overwrite", action="store_true")
    collect_parser.set_defaults(func=collect)

    compare_parser = sub.add_parser("compare", help="compare JSONLs")
    compare_parser.add_argument("--ref", required=True, help="reference JSONL")
    compare_parser.add_argument(
        "--cand", action="append", required=True, help="name=path JSONL, repeatable"
    )
    compare_parser.add_argument("--json", default="", help="write per-prompt results here")  # noqa: E501
    compare_parser.set_defaults(func=compare)

    teacher_parser = sub.add_parser("teacher", help="score official text under a local config")  # noqa: E501
    teacher_parser.add_argument("--prompts", required=True)
    teacher_parser.add_argument("--ref", required=True, help="official collect JSONL")
    teacher_parser.add_argument("--out", required=True)
    teacher_parser.add_argument("--base-url", required=True)
    teacher_parser.add_argument("--model", required=True)
    teacher_parser.add_argument("--thinking", choices=["off", "on"], default="off")
    teacher_parser.add_argument("--prompt-logprobs", type=int, default=1)
    teacher_parser.add_argument(
        "--tokenizer",
        default="",
        help="HF tokenizer dir; renders the prompt locally instead of /tokenize",
    )
    teacher_parser.add_argument("--concurrency", type=int, default=4)
    teacher_parser.add_argument("--timeout", type=int, default=1800)
    teacher_parser.add_argument("--retries", type=int, default=5)
    teacher_parser.add_argument("--limit", type=int, default=0)
    teacher_parser.add_argument("--overwrite", action="store_true")
    teacher_parser.set_defaults(func=teacher)

    score_parser = sub.add_parser("score", help="compare teacher-forced NLLs")
    score_parser.add_argument("--run", action="append", required=True, help="name=path")
    score_parser.set_defaults(func=score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
