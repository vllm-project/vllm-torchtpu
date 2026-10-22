# SPDX-License-Identifier: Apache-2.0
"""Evaluate a frozen GSM8K JSONL sample against an already running server.

This is a zero-shot, non-thinking chat protocol, not the few-shot leaderboard
protocol. Persist every response and report incomplete or truncated requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests

INSTRUCTION = (
    "\nSolve step by step. End with '#### ' followed by only the numeric answer."
)


def extract_number(text: str) -> str | None:
    """Normalize the last GSM8K answer marker, rejecting partial expressions."""
    if "####" not in text:
        return None
    answer = text.rsplit("####", 1)[1].strip()
    if re.fullmatch(r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", answer) is None:
        return None
    try:
        value = Decimal(answer.replace(",", "")).normalize()
        return str(value) if value else "0"
    except InvalidOperation:
        return None


def load_sample(dataset: Path, samples: int, seed: int) -> dict[str, Any]:
    data = dataset.read_bytes()
    cases = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not 0 < samples <= len(cases):
        raise ValueError(f"samples must be in [1, {len(cases)}], got {samples}")
    indices = sorted(random.Random(seed).sample(range(len(cases)), samples))
    selected = []
    for index in indices:
        case = cases[index]
        reference = extract_number(case["answer"])
        if reference is None or not isinstance(case["question"], str):
            raise ValueError(f"Invalid GSM8K reference at index {index}")
        selected.append(
            {"index": index, "question": case["question"], "reference": reference}
        )
    return {
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "dataset_cases": len(cases),
        "sample_seed": seed,
        "cases": selected,
    }


def summarize_round(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "completed": sum("error" not in row for row in rows),
        "correct": sum(row.get("correct", False) for row in rows),
        "missing_answers": sum(
            "error" not in row and row["predicted"] is None for row in rows
        ),
        "truncated": sum(row.get("finish_reason") == "length" for row in rows),
        "errors": [row["index"] for row in rows if "error" in row],
    }


def evaluate(
    dataset: Path,
    out: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    samples: int = 256,
    concurrency: int = 32,
    repeats: int = 2,
    seed: int = 0,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    if min(concurrency, repeats, max_tokens) < 1:
        raise ValueError("concurrency, repeats and max_tokens must be positive")
    sample = load_sample(dataset, samples, seed)
    models = requests.get(f"{base_url}/v1/models", timeout=30)
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    out.mkdir(parents=True, exist_ok=False)
    bodies = {
        case["index"]: {
            "model": model,
            "messages": [{"role": "user", "content": case["question"] + INSTRUCTION}],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False},
        }
        for case in sample["cases"]
    }
    body_bytes = json.dumps(bodies, sort_keys=True, ensure_ascii=False).encode()
    manifest = {
        **sample,
        "protocol": "zero-shot non-thinking chat; last #### numeric answer",
        "request_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "model": model,
        "base_url": base_url,
        "concurrency": concurrency,
        "repeats": repeats,
        "temperature": 0,
        "generation_seed": seed,
        "max_tokens": max_tokens,
        "thinking": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def request(case: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        row = dict(case)
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=bodies[case["index"]],
                timeout=3600,
            )
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            predicted = extract_number(choice["message"]["content"] or "")
            row.update(
                response=result,
                predicted=predicted,
                correct=predicted is not None and predicted == case["reference"],
                finish_reason=choice["finish_reason"],
            )
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = time.monotonic() - started
        return row

    report: dict[str, Any] = {
        "dataset_sha256": sample["dataset_sha256"],
        "request_sha256": manifest["request_sha256"],
        "samples": samples,
        "rounds": [],
        "completed": False,
    }
    for repeat in range(1, repeats + 1):
        started = time.monotonic()
        rows = []
        with (
            (out / f"round-{repeat}-responses.jsonl").open("w") as stream,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            futures = [pool.submit(request, case) for case in sample["cases"]]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
        summary = {
            "round": repeat,
            **summarize_round(rows),
            "seconds": time.monotonic() - started,
        }
        report["rounds"].append(summary)
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    report["completed"] = all(row["completed"] == samples for row in report["rounds"])
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    report = evaluate(**vars(parser.parse_args()))
    if not report["completed"]:
        raise SystemExit("Evaluation incomplete; see per-request errors")


if __name__ == "__main__":
    main()
