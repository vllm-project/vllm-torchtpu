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
"""32k Needle-in-a-Haystack client for an OpenAI-compatible vLLM server.

Tier 4 gate (DIRECTION_AND_COORDINATION.md, "32k Needle 5/5 + 5/5"): each
round places one needle (a city -> 7-digit number fact) at five depths of a
~32k-token haystack and asks for the number with greedy decoding. Rounds use
different seeds, so filler, needles and keys all change. A trial passes when
the exact key string appears in the reply. Exit status is 0 iff every trial
passes.

Stdlib only, so it runs with any Python next to the server.

Example:
  python needle_32k.py --base-url http://127.0.0.1:8012 \\
      --model deepseek-v4.1-flash --target-tokens 32000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

_SUBJECTS = (
    "The archivist",
    "A tired engineer",
    "The river pilot",
    "Our neighbour",
    "The night baker",
    "A visiting botanist",
    "The harbour clerk",
    "My cousin",
    "The orchestra tuner",
    "A quiet surveyor",
    "The lighthouse keeper",
)
_VERBS = (
    "catalogued",
    "repaired",
    "described",
    "sketched",
    "measured",
    "ignored",
    "collected",
    "photographed",
    "questioned",
    "restored",
    "counted",
)
_OBJECTS = (
    "the old maps",
    "a cracked bell",
    "three wooden crates",
    "the garden wall",
    "a stack of letters",
    "the tide tables",
    "an unfinished bridge",
    "the market stalls",
    "a row of lanterns",
    "the winter supplies",
)
_TAILS = (
    "before the rain started",
    "while the town slept",
    "after a long debate",
    "without telling anyone",
    "for the third time that week",
    "as the ferry departed",
    "under a grey sky",
    "with surprising care",
)
_CITIES = (
    "Lisbon",
    "Nairobi",
    "Osaka",
    "Quito",
    "Tallinn",
    "Hobart",
    "Reykjavik",
    "Montevideo",
    "Kraków",
    "Da Nang",
    "Windhoek",
    "Valparaíso",
    "Tromsø",
    "Ljubljana",
    "Cusco",
    "Zanzibar",
    "Bergen",
    "Hanoi",
    "Porto",
    "Tbilisi",
)


def _post(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON and return the decoded JSON reply."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _filler(rng: random.Random, n: int) -> list[str]:
    """Return ``n`` pseudo-random filler sentences."""
    return [
        f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
        f"{rng.choice(_TAILS)} on day {rng.randint(1, 999)}."
        for _ in range(n)
    ]


def _count_tokens(base: str, model: str, text: str, timeout: float) -> int:
    """Count prompt tokens via the server's ``/tokenize`` endpoint."""
    reply = _post(f"{base}/tokenize", {"model": model, "prompt": text}, timeout)
    return int(reply["count"])


def build_prompt(sentences: list[str], depth: float, needle: str, question: str) -> str:
    """Insert ``needle`` at fractional ``depth`` and append ``question``."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    pos = round(depth * len(sentences))
    body = " ".join(sentences[:pos] + [needle] + sentences[pos:])
    return (
        "Below is a long document. Read it carefully; one sentence states a "
        "special magic number.\n\n<document>\n"
        f"{body}\n</document>\n\n{question}"
    )


def main() -> int:
    """Run all rounds and print a JSON summary; return a shell status code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8012")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--target-tokens", type=int, default=32000)
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default=None, help="optional JSON result path")
    args = ap.parse_args()
    depths = [float(d) for d in args.depths.split(",")]
    base = args.base_url.rstrip("/")

    # Calibrate sentences-per-token once; filler sentences have a stable length.
    probe = _filler(random.Random(0), 400)
    per_sentence = _count_tokens(base, args.model, " ".join(probe), args.timeout) / 400
    n_sentences = int(args.target_tokens / per_sentence)

    trials = []
    for rnd in range(args.rounds):
        rng = random.Random(args.seed + 1000 * rnd)
        cities = rng.sample(_CITIES, len(depths))
        for depth, city in zip(depths, cities, strict=True):
            key = str(rng.randint(1_000_000, 9_999_999))
            needle = f"The special magic number for {city} is {key}."
            question = (
                f"What is the special magic number for {city}? "
                "Answer with the number only."
            )
            prompt = build_prompt(_filler(rng, n_sentences), depth, needle, question)
            reply = _post(
                f"{base}/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "chat_template_kwargs": {"thinking": False},
                },
                args.timeout,
            )
            text = reply["choices"][0]["message"]["content"] or ""
            ok = key in text
            trials.append(
                {
                    "round": rnd,
                    "depth": depth,
                    "city": city,
                    "key": key,
                    "prompt_tokens": reply["usage"]["prompt_tokens"],
                    "reply": text.strip()[:80],
                    "pass": ok,
                }
            )
            print(json.dumps(trials[-1], ensure_ascii=False), flush=True)

    per_round = [
        sum(t["pass"] for t in trials if t["round"] == r) for r in range(args.rounds)
    ]
    summary = {
        "score": " + ".join(f"{p}/{len(depths)}" for p in per_round),
        "all_pass": all(t["pass"] for t in trials),
        "trials": trials,
    }
    print("NEEDLE", summary["score"], "ALL_PASS" if summary["all_pass"] else "FAIL")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
