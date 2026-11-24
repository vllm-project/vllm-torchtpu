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
    logprobs: int | None = None,
) -> dict:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "return_token_ids": True,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    lp = choice.get("logprobs") or {}
    return {
        "tokens": choice.get("token_ids") or [],
        "text": choice.get("text", ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens") or 0,
        "token_logprobs": lp.get("token_logprobs") or [],
        "logprob_tokens": lp.get("tokens") or [],
        "top_logprobs": lp.get("top_logprobs") or [],
    }


def _divergence_diag(ref: dict, cur: dict) -> dict:
    ref_ids, cur_ids = ref["tokens"], cur["tokens"]
    first = next(
        (i for i, (a, b) in enumerate(zip(ref_ids, cur_ids)) if a != b),
        None,
    )
    ref_lps, cur_lps = ref["token_logprobs"], cur["token_logprobs"]
    prior_end = len(ref_ids) if first is None else first
    prior_diffs = [
        abs(a - b)
        for a, b in zip(ref_lps[:prior_end], cur_lps[:prior_end])
        if a is not None and b is not None
    ]
    diag: dict = {
        "first_idx": first,
        "len_ref": len(ref_ids),
        "len_cur": len(cur_ids),
        "has_logprobs": bool(ref_lps) and bool(cur_lps),
        "max_prior_lp_diff": round(max(prior_diffs), 6) if prior_diffs else None,
        "ref_text": ref["text"],
        "cur_text": cur["text"],
    }
    if first is not None:
        diag["ref_id"] = ref_ids[first]
        diag["cur_id"] = cur_ids[first]
        if first < len(ref["logprob_tokens"]) and first < len(cur["logprob_tokens"]):
            diag["ref_tok"] = ref["logprob_tokens"][first]
            diag["cur_tok"] = cur["logprob_tokens"][first]
        if first < len(ref["top_logprobs"]) and first < len(cur["top_logprobs"]):
            diag["ref_top"] = ref["top_logprobs"][first]
            diag["cur_top"] = cur["top_logprobs"][first]
    return diag


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

    reference: dict[str, dict] = {}
    failures = 0
    hit_rounds = 0

    for rnd in range(1, a.rounds + 1):
        for tag in tags:  # interleaved: the slot last held the other prefix
            try:
                r = complete(
                    url,
                    a.model,
                    prompts[tag],
                    a.max_tokens,
                    a.timeout,
                    headers,
                    logprobs=5,
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"STATE_CONTAMINATION_REQUEST_FAILED {tag} r{rnd}: {exc}")
                return 1

            diag = None
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
                verdict = "match" if same else "DIVERGED"
                if not same:
                    diag = _divergence_diag(reference[tag], r)
            record: dict = {
                "round": rnd,
                "prefix": tag,
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "verdict": verdict,
                "text": r["text"][:80],
            }
            if diag is not None:
                record["diag"] = diag
            print(json.dumps(record), flush=True)

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
