# SPDX-License-Identifier: Apache-2.0
"""GSM8K 5-shot accuracy against a running server, paired across runs.

``run`` asks the server's completions endpoint every test question greedily
(5 training examples as shots, lm-eval's Question/Answer format) and writes
one record per question, so ``compare`` can pair two runs question by question:
a numerics change is judged by the questions it flips, not by two totals.
"""

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DATA = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
SHOTS = 5
STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
NUMBER = re.compile(r"-?[$0-9.,]{2,}|-?[0-9]+")


def _load(split: str, cache: Path) -> list[dict]:
    path = cache / f"{split}.jsonl"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{DATA}/{split}.jsonl", path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def _number(text: str) -> str:
    return text.replace(",", "").replace("$", "").rstrip(".").strip()


def _extract(completion: str) -> tuple[str | None, str | None]:
    """(strict ``#### N`` answer, last number in the completion)."""
    strict = STRICT.search(completion)
    numbers = NUMBER.findall(completion)
    return (
        _number(strict.group(1)) if strict else None,
        _number(numbers[-1]) if numbers else None,
    )


def run(args: argparse.Namespace) -> None:
    train, test = _load("train", args.cache), _load("test", args.cache)
    if args.limit:
        test = test[: args.limit]
    shots = "".join(
        f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train[:SHOTS]
    )

    def ask(index: int) -> dict:
        item = test[index]
        response = requests.post(
            f"{args.base_url}/v1/completions",
            json={
                "model": args.model,
                "prompt": f"{shots}Question: {item['question']}\nAnswer:",
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stop": ["Question:", "\n\n\n"],
            },
            timeout=1800,
        )
        response.raise_for_status()
        completion = response.json()["choices"][0]["text"]
        gold = _number(item["answer"].split("####")[-1])
        strict, flexible = _extract(completion)
        return {
            "index": index,
            "gold": gold,
            "strict": strict == gold,
            "flexible": flexible == gold,
            "completion": completion,
        }

    with ThreadPoolExecutor(args.concurrency) as pool:
        records = list(pool.map(ask, range(len(test))))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gsm8k.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    summary = {
        "questions": len(records),
        "strict": sum(r["strict"] for r in records) / len(records),
        "flexible": sum(r["flexible"] for r in records) / len(records),
    }
    (args.out / "gsm8k.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DSV41_GSM8K", json.dumps(summary))


def compare(args: argparse.Namespace) -> None:
    """Paired flips of ``b`` against ``a`` on the questions both answered."""

    def load(path: Path) -> dict[int, bool]:
        lines = path.read_text().splitlines()
        return {r["index"]: r[args.metric] for r in map(json.loads, lines)}

    a, b = load(args.a), load(args.b)
    common = sorted(a.keys() & b.keys())
    lost = sum(a[i] and not b[i] for i in common)
    gained = sum(b[i] and not a[i] for i in common)
    print(
        json.dumps(
            {
                "questions": len(common),
                "a": sum(a[i] for i in common) / len(common),
                "b": sum(b[i] for i in common) / len(common),
                "lost": lost,
                "gained": gained,
                "paired_delta_points": 100 * (gained - lost) / len(common),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-v41")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/gsm8k"))
    p.add_argument("--limit", type=int, default=0, help="0 = all 1,319")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=384)
    p.set_defaults(func=run)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--metric", choices=["strict", "flexible"], default="flexible")
    c.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
