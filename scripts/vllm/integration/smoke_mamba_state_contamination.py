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
"""Does a prefix-cache hit restore the Mamba state, or just the attention KV?

Two long prefixes are interleaved, so the recurrent slot a resumed request
picks up was last written by the *other* prefix. Attention KV comes back from
the cache either way; only a real state restore makes the continuation match
the reference. Comparing token ids rather than an expected answer means a
wrong state shows up even when it does not change the final answer.

`--reference uncached` (the default) compares every interleaved hit with the
uncached first run. That is exact when a resumed request computes its tail in
the same chunks as the cold prefill, e.g. with one prefill chunk per KV block.

`--reference cached` is for engines whose prefill chunk spans several blocks:
there a resumed request computes only the uncached tail, in different chunks
than the cold prefill, and that alone shifts logprobs (by up to ~0.2 on
Qwen3.5), enough to flip a greedy token where the top two are close. Each
prefix is then sent again right after its cold run, before the other prefix's
first request, and that hit, resumed from a slot the same prefix just wrote,
is the reference the interleaved hits must match exactly. A restore that loses
the saved state, or a later cold prefill that overwrites it, makes them
differ. How close cached runs stay to the uncached one is checked with a
tolerance by smoke_prefix_cache_e2e_divergence.py; here it is only reported.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def build_prefix(tag: str, lines: int) -> str:
    # Distinct content per tag so the two states can never be interchangeable.
    body = "\n".join(
        f"{tag} record {i:04d}: quantity {(i * 7 + len(tag)) % 97}"
        for i in range(lines)
    )
    return f"Inventory ledger {tag}. Read every record before answering.\n{body}\n"


def complete(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    headers: dict | None = None,
) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "return_token_ids": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "tokens": choice.get("token_ids") or [],
        "text": choice.get("text", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens") or 0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", default=os.environ.get("PORT", "8000"))
    p.add_argument("--model", required=True)
    p.add_argument("--lines", type=int, default=340)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--reference",
        choices=("uncached", "cached"),
        default="uncached",
        help="What interleaved hits must match: the uncached first run, or a "
        "hit resumed right after the same prefix (see the module docstring).",
    )
    p.add_argument(
        "--data-parallel-rank",
        type=int,
        help="Hold every request to this DP rank of the engine, with vLLM's "
        "X-data-parallel-rank header; through a P/D proxy, of both engines. "
        "Replicas differ slightly in greedy output, and this compares tokens.",
    )
    a = p.parse_args()
    url = f"http://{a.host}:{a.port}"
    headers = {}
    if a.data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(a.data_parallel_rank)

    tags = ["ALPHA", "BRAVO"]
    prompts = {
        t: build_prefix(t, a.lines)
        + "\nSummarise the ledger above in one sentence, then state the final "
        "record's quantity.\n"
        for t in tags
    }

    def send(tag: str) -> dict | None:
        try:
            return complete(
                url, a.model, prompts[tag], a.max_tokens, a.timeout, headers
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"STATE_CONTAMINATION_REQUEST_FAILED {tag}: {exc}")
            return None

    def first_diff(x: list[int], y: list[int]) -> int | None:
        n = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
        return n if n is not None or len(x) == len(y) else min(len(x), len(y))

    def show(rnd: int | str, tag: str, r: dict, verdict: str) -> None:
        print(
            json.dumps(
                {
                    "round": rnd,
                    "prefix": tag,
                    "prompt_tokens": r["prompt_tokens"],
                    "cached_tokens": r["cached_tokens"],
                    "verdict": verdict,
                    "uncached_first_diff": first_diff(
                        r["tokens"], uncached[tag]["tokens"]
                    )
                    if tag in uncached
                    else None,
                    "text": r["text"][:80],
                }
            ),
            flush=True,
        )

    uncached: dict[str, dict] = {}
    reference: dict[str, dict] = {}
    failures = 0
    hit_rounds = 0

    # Round 1 is uncached. With --reference cached, each prefix is resumed
    # right after its own cold run, before the other prefix is first sent.
    for tag in tags:
        r = send(tag)
        if r is None:
            return 1
        uncached[tag] = r
        show(1, tag, r, "uncached" if a.reference == "cached" else "reference")
        if a.reference == "uncached":
            reference[tag] = r
            continue
        r = send(tag)
        if r is None:
            return 1
        if r["cached_tokens"] == 0:
            print(f"STATE_CONTAMINATION_REFERENCE_NOT_CACHED {tag}")
            return 1
        reference[tag] = r
        show("ref", tag, r, "reference")

    for rnd in range(2, a.rounds + 1):
        for tag in tags:  # interleaved: the slot last held the other prefix
            r = send(tag)
            if r is None:
                return 1
            same = r["tokens"] == reference[tag]["tokens"]
            if r["cached_tokens"] > 0:
                hit_rounds += 1
                if not same:
                    failures += 1
            show(rnd, tag, r, "match" if same else "DIVERGED")

    print(f"STATE_CONTAMINATION_CACHED_ROUNDS {hit_rounds}")
    print(f"STATE_CONTAMINATION_DIVERGENCES {failures}")
    if hit_rounds == 0:
        # Nothing was cached, so nothing was tested.
        print("STATE_CONTAMINATION_NO_HITS 1")
        return 1
    print(f"STATE_CONTAMINATION_OK {int(failures == 0)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
