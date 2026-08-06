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
the uncached one. Comparing token ids rather than an expected answer means a
wrong state shows up even when it does not change the final answer.
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
        for i in range(lines))
    return (f"Inventory ledger {tag}. Read every record before answering.\n"
            f"{body}\n")


def complete(url: str, model: str, prompt: str, max_tokens: int,
             timeout: int) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "return_token_ids": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
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
    a = p.parse_args()
    url = f"http://{a.host}:{a.port}"

    tags = ["ALPHA", "BRAVO"]
    prompts = {
        t:
        build_prefix(t, a.lines) +
        "\nSummarise the ledger above in one sentence, then state the final "
        "record's quantity.\n"
        for t in tags
    }

    reference: dict[str, dict] = {}
    failures = 0
    hit_rounds = 0

    for rnd in range(1, a.rounds + 1):
        for tag in tags:  # interleaved: the slot last held the other prefix
            try:
                r = complete(url, a.model, prompts[tag], a.max_tokens,
                             a.timeout)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(
                    f"STATE_CONTAMINATION_REQUEST_FAILED {tag} r{rnd}: {exc}")
                return 1

            if tag not in reference:
                # Round 1 is uncached: this is the ground truth.
                reference[tag] = r
                verdict = "reference"
            else:
                same = r["tokens"] == reference[tag]["tokens"]
                if r["cached_tokens"] > 0:
                    hit_rounds += 1
                    if not same:
                        failures += 1
                verdict = ("match" if same else "DIVERGED")
            print(json.dumps({
                "round": rnd,
                "prefix": tag,
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "verdict": verdict,
                "text": r["text"][:80],
            }),
                  flush=True)

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
