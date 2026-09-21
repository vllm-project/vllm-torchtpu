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
"""Smoke verification for cross-replica Global Prefix Caching (GPC).

Validates peer-to-peer KV cache transfer and prefix sharing across two
independent serving replicas connected via a centralized global registry.

Architecture & Validation Flow:
  - Replicas A and B run identical model configurations (PCP8/TP1) with
    TPURaidenOffloadingConnector enabled.
  - Replica A processes a cold prompt exceeding one full offload span, computing
    and saving the KV cache to host DRAM and registering block keys.
  - Replica B queries the same prefix. The prefix resolves via the global
    registry and is fetched peer-to-peer over the network via read_remote.

Verification Gates:
  1. Cache Hit Verification: Every seeded document replayed on Replica B must
     register external prefix-cache hits, confirming cache reuse over recompute.
  2. Prefix Output Consistency: Every replayed continuation on Replica B must
     match Replica A's cold output on the first eight characters. Restored KV
     corruption causes immediate token divergence.
  3. Cold Control Invariant: An unseeded control document submitted to Replica B
     must produce zero external hits, verifying that the hit counter is not
     incrementing unconditionally.
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request

KEYS = dict(
    pre_s="request_prefill_time_seconds_sum",
    pre_n="request_prefill_time_seconds_count",
    hit="external_prefix_cache_hits_total",
    q="external_prefix_cache_queries_total",
)

FILLER = "the archive shelf holds ledgers sorted by year and colour."


def build_prompt(tag: str, code: str, lines: int) -> str:
    body = [f"line {i:04d}: {FILLER}" for i in range(lines)]
    # Inject distinct access code markers within the initial block (lines 2, 6, 10)
    # to guarantee unique prefix hash chains per document and prevent cross-probe
    # cache collisions. The terminal question forces attention decode across the
    # context body.
    for i in (2, 6, 10):
        body[i] = f"line {i:04d}: the {tag} access code is {code}."
    body.append(f"What is the {tag} access code? Answer with only the code.")
    return "\n".join(body)


def scrape(base: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
        text = r.read().decode()
    m: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo:
            m[mo.group(1)] = m.get(mo.group(1), 0.0) + float(mo.group(2))
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-a",
        required=True,
        help="Base URL (http://host:port) for Replica A (seeder).",
    )
    parser.add_argument(
        "--base-b",
        required=True,
        help="Base URL (http://host:port) for Replica B (replayer).",
    )
    parser.add_argument("--model", required=True, help="Served model name identifier.")
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=16385,
        help=(
            "Minimum token threshold exceeding one full offload span "
            "(block_size * pcp + 1). Because prefix caching matches up to "
            "num_tokens - 1, prompts must exceed the span boundary for the "
            "lookup to reach the global registry at all."
        ),
    )
    parser.add_argument("--publish-wait-s", type=float, default=15.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    def api(path: str, base: str, payload: dict) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            raise SystemExit(f"{base}{path} returned HTTP {err.code}: {body}") from err

    def ask(base: str, prompt: str) -> tuple[str, int]:
        resp = api(
            "/v1/chat/completions",
            base,
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 0,
            },
        )
        return (
            resp["choices"][0]["message"]["content"],
            resp["usage"]["prompt_tokens"],
        )

    # Execute single request and capture Prometheus metric deltas on Replica B.
    def measured(prompt: str) -> tuple[str, dict[str, float]]:
        before = scrape(args.base_b)
        text, _ = ask(args.base_b, prompt)
        after = scrape(args.base_b)
        d = {k: after.get(v, 0) - before.get(v, 0) for k, v in KEYS.items()}
        return text, {
            "prefill": d["pre_s"] / (d["pre_n"] or 1),
            "hit": d["hit"],
            "q": d["q"],
        }

    # Unique tag and code pairs ensure distinct prefix hashes in the initial block.
    seeds = [
        ("emerald", "739284"),
        ("obsidian", "506192"),
        ("amethyst", "584317"),
    ]
    fresh_tag, fresh_code = "crimson", "281473"

    # Calibrate prompt length dynamically using the server's tokenizer endpoint
    # to target one full offload span plus safety margin, avoiding client-side
    # tokenization drift.
    sample = api(
        "/tokenize",
        args.base_a,
        {
            "model": args.model,
            "prompt": build_prompt(seeds[0][0], seeds[0][1], 100),
        },
    )
    per_line = sample["count"] / 100
    target = args.min_prompt_tokens + 2048
    if target > sample["max_model_len"]:
        print(
            f"RESULT: FAIL (the probes need {target} tokens but max_model_len is"
            f" only {sample['max_model_len']})"
        )
        return 1
    lines = math.ceil(target / per_line)
    print(
        f"sizing: {per_line:.1f} tokens/line,"
        f" max_model_len={sample['max_model_len']}, {lines} lines targeting"
        f" {target} tokens"
    )

    failures = []

    print(f"=== seeding {len(seeds)} documents cold through A ===")
    cold_answers = []
    for tag, code in seeds:
        ans, tokens = ask(args.base_a, build_prompt(tag, code, lines))
        print(f"seed {tag}: prompt_tokens={tokens} answer={ans!r}")
        cold_answers.append(ans)
        if tokens < args.min_prompt_tokens:
            print(
                f"RESULT: FAIL (the {tag} document is {tokens} tokens, below the"
                f" {args.min_prompt_tokens}-token offload span despite the"
                " tokenizer-based sizing above)"
            )
            return 1
    # Allow sufficient time for asynchronous host DRAM transfer and global registry publication.
    time.sleep(2 * args.publish_wait_s)

    print("=== replaying through B ===")
    matches = 0
    for (tag, code), cold_ans in zip(seeds, cold_answers):
        ans, rep = measured(build_prompt(tag, code, lines))
        match = ans[:8] == cold_ans[:8]
        matches += match
        print(
            f"replay {tag}: answer={ans!r} prefix_match={match}"
            f" prefill={rep['prefill']:.3f}s ext_hit={int(rep['hit'])}/{int(rep['q'])}"
        )
        if rep["hit"] <= 0:
            failures.append(
                f"the {tag} replay produced no external hits on server B; it"
                " recomputed instead of pulling from the shared cache"
            )
    if matches < len(seeds):
        failures.append(
            f"only {matches}/{len(seeds)} replays continued the way A's cold"
            " computation did; corrupt restored KV diverges from the first tokens"
            " on every document"
        )

    print("=== fresh document through B (cold control) ===")
    ans_f, cold = measured(build_prompt(fresh_tag, fresh_code, lines))
    print(
        f"cold: answer={ans_f!r} prefill={cold['prefill']:.3f}s"
        f" ext_hit={int(cold['hit'])}/{int(cold['q'])}"
    )
    if cold["hit"] > 0:
        failures.append(
            f"the fresh document produced {int(cold['hit'])} external hits: it is"
            " not cold, so the replay gate proves nothing"
        )

    if failures:
        print("RESULT: FAIL (" + "; ".join(failures) + ")")
        return 1
    print(
        f"RESULT: PASS ({matches}/{len(seeds)} replays matched A's cold output"
        " on their first tokens with external hits on every replay; the fresh"
        " document stayed cold)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
