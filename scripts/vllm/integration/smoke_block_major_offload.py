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
"""Seed and revisit through the offload tier on the block-major unified pool.

Per prompt length L, three passes over the same prompt:

  cold    fresh prompt, nothing cached. Gate: no local or external hit.
  hbm     immediate repeat. Gate: whole-block local hit, no external hit.
  store   POST /reset_prefix_cache empties HBM and leaves the Raiden store
          intact, so the repeat has to load the prefix back from host.
          Gate: no local hit, an external hit of at least one block.

The hit gates are hard: a run that never reaches the store proves nothing.
The output gate compares the first eight characters the hbm and store passes
generate with the cold pass: corrupt restored KV garbles output immediately,
while TPU serving is not run-to-run deterministic further out. Whole token
sequences are printed, never required.

Each prompt is a shipment ledger that names a manifest code on its second
line and asks for that code back at the end, so the first generated characters
are pinned by the recalled prefix. The engine's /tokenize endpoint sizes the
ledger, and every ledger carries a fresh salt so nothing from an earlier run
can hit.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

COUNTERS = (
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "external_prefix_cache_queries_total",
    "external_prefix_cache_hits_total",
)


def _post(base, path, payload, timeout, headers=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}


def _metrics(base, timeout=60):
    with urllib.request.urlopen(base + "/metrics", timeout=timeout) as resp:
        text = resp.read().decode()
    totals = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line[len("vllm:") :].split("{", 1)[0].split(" ", 1)[0]
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[1])
    return totals


class Client:
    def __init__(
        self,
        base,
        model,
        timeout,
        max_tokens,
        dp_rank,
        metrics_base=None,
        reset_base=None,
    ):
        self.base = base
        # Requests may go through a P/D proxy; the hit counters and the reset
        # address the engine that owns the prefix cache under test.
        self.metrics_base = metrics_base or base
        self.reset_base = reset_base or base
        # usage.prompt_tokens_details comes from the engine that answers, which
        # behind a proxy is the decode side, so only the counters are gated then.
        self.cached_tokens_gated = self.metrics_base == base
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.headers = {} if dp_rank is None else {"X-data-parallel-rank": str(dp_rank)}

    def tokenize(self, text):
        status, body = _post(
            self.metrics_base,
            "/tokenize",
            {"model": self.model, "prompt": text},
            self.timeout,
        )
        if status != 200:
            raise SystemExit(
                f"the engine did not answer /tokenize (HTTP {status}): "
                f"{body.get('error')}"
            )
        return body["count"]

    def ask(self, tag, prompt):
        before = _metrics(self.metrics_base)
        started = time.monotonic()
        status, body = _post(
            self.base,
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "seed": 0,
                "ignore_eos": True,
                "return_token_ids": True,
            },
            self.timeout,
            self.headers,
        )
        elapsed = time.monotonic() - started
        after = _metrics(self.metrics_base)
        delta = {k: after[k] - before[k] for k in COUNTERS}
        result = {
            "tag": tag,
            "status": status,
            "elapsed_s": round(elapsed, 2),
            "local_hits": delta["prefix_cache_hits_total"],
            "ext_hits": delta["external_prefix_cache_hits_total"],
            "ext_queries": delta["external_prefix_cache_queries_total"],
            "cached_tokens": None,
            "token_ids": None,
            "text": None,
        }
        if status != 200:
            result["error"] = body.get("error")
            return result
        choice = body["choices"][0]
        result["token_ids"] = choice.get("token_ids")
        result["text"] = choice.get("text")
        details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
        result["cached_tokens"] = details.get("cached_tokens")
        return result

    def reset_prefix_cache(self, attempts=10):
        # A plain reset empties HBM only; ?reset_external=true would also
        # clear the store.
        for _ in range(attempts):
            status, _ = _post(self.reset_base, "/reset_prefix_cache", {}, self.timeout)
            if status == 200:
                return True
            time.sleep(3)
        return False


FILLER = (
    "twelve crates of copper fittings, sealed and weighed at the dock",
    "four pallets of insulated cable on the northern loading ramp",
    "one refrigerated container of pharmaceutical samples, customs cleared",
    "nine drums of machine oil, hazard labels checked twice",
    "two flatbeds of pine lumber, moisture within tolerance",
    "thirty cartons of ceramic tiles, three reported cracked on arrival",
)


def build_prompt(client, target_tokens, salt):
    """Grows a ledger until the engine tokenizes it to at least target_tokens."""
    namespace = f"{salt}-{target_tokens}"
    code = str(random.Random(namespace).randrange(100000, 1000000))
    head = [
        f"Depot archive {namespace}: quarterly shipment ledger.",
        f"The manifest code for archive {namespace} is {code}.",
        "The entries below are filed one per shipment, in order.",
    ]
    tail = [
        "",
        f"Question about archive {namespace}: give its manifest code, then "
        "describe in two or three sentences what this ledger records.",
        "Answer: The manifest code is",
    ]
    lines = max(4, target_tokens // 16)
    for _ in range(8):
        body = [f"Entry {i:04d}: {FILLER[i % len(FILLER)]}." for i in range(lines)]
        text = "\n".join(head + body + tail)
        count = client.tokenize(text)
        if count >= target_tokens:
            return text, count
        lines = max(lines + 1, round(lines * target_tokens / max(count, 1)) + 1)
    raise SystemExit(
        f"could not grow a ledger to {target_tokens} tokens (reached {count})"
    )


def _show(result):
    shown = {
        k: result.get(k)
        for k in (
            "tag",
            "status",
            "elapsed_s",
            "cached_tokens",
            "local_hits",
            "ext_hits",
            "ext_queries",
        )
    }
    print("  " + json.dumps(shown), flush=True)
    if result.get("token_ids") is not None:
        print(f"  token_ids={result['token_ids']} text={result['text']!r}", flush=True)


def _diverged(tag, length, cold, arm):
    match = (arm["text"] or "")[:8] == (cold["text"] or "")[:8]
    print(
        f"  {tag}: prefix_match={match} "
        f"tokens_equal={arm['token_ids'] == cold['token_ids']}",
        flush=True,
    )
    if match:
        return None
    return (
        f"L={length}: {tag} pass diverged from cold in the first characters "
        f"(corrupt restored KV?): cold={cold['text']!r} {tag}={arm['text']!r}"
    )


def run_length(client, length, hit_block, salt, settle_s):
    """One cold/hbm/store triple. Returns the list of failures."""
    failures = []
    prompt, length = build_prompt(client, length, salt)
    expected_hit = ((length - 1) // hit_block) * hit_block
    print(f"--- L={length}: expected whole-block hit {expected_hit} tokens", flush=True)

    cold = client.ask("cold", prompt)
    _show(cold)
    if cold["status"] != 200:
        return [f"L={length}: cold pass failed: {cold.get('error')}"]
    gated = client.cached_tokens_gated
    if gated and cold["cached_tokens"] is None:
        return [
            f"L={length}: usage.prompt_tokens_details is missing; "
            "the server needs --enable-prompt-tokens-details"
        ]
    if cold["token_ids"] is None:
        return [
            f"L={length}: choices[0].token_ids is missing; "
            "return_token_ids is not honoured"
        ]
    if (
        cold["local_hits"] > 0
        or cold["ext_hits"] > 0
        or (gated and cold["cached_tokens"])
    ):
        failures.append(
            f"L={length}: the cold pass already hit (local={cold['local_hits']} "
            f"ext={cold['ext_hits']} cached_tokens={cold['cached_tokens']})"
        )

    # Let the store jobs for this prompt land before the HBM cache is emptied.
    time.sleep(settle_s)

    hbm = client.ask("hbm", prompt)
    _show(hbm)
    if hbm["status"] != 200:
        return failures + [f"L={length}: hbm pass failed: {hbm.get('error')}"]
    hbm_served = hbm["cached_tokens"] if gated else hbm["local_hits"]
    if hbm_served < expected_hit or hbm["local_hits"] <= 0:
        failures.append(
            f"L={length}: hbm pass did not hit the HBM prefix cache "
            f"(cached_tokens={hbm['cached_tokens']} "
            f"local_hits={hbm['local_hits']} expected>={expected_hit})"
        )
    if hbm["ext_hits"] > 0:
        failures.append(
            f"L={length}: hbm pass went to the store (ext_hits={hbm['ext_hits']})"
        )
    diverged = _diverged("hbm", length, cold, hbm)
    if diverged:
        failures.append(diverged)

    if not client.reset_prefix_cache():
        return failures + [f"L={length}: POST /reset_prefix_cache never returned 200"]

    store = client.ask("store", prompt)
    _show(store)
    if store["status"] != 200:
        return failures + [f"L={length}: store pass failed: {store.get('error')}"]
    store_served = store["cached_tokens"] if gated else store["ext_hits"]
    if store["ext_hits"] <= 0 or store_served < expected_hit:
        failures.append(
            f"L={length}: store pass did not load from the Raiden store "
            f"(ext_hits={store['ext_hits']} "
            f"cached_tokens={store['cached_tokens']} expected>={expected_hit})"
        )
    if store["local_hits"] > 0:
        failures.append(
            f"L={length}: store pass hit HBM after the reset "
            f"(local_hits={store['local_hits']})"
        )
    diverged = _diverged("store", length, cold, store)
    if diverged:
        failures.append(diverged)
    return failures


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--metrics-base",
        default=None,
        help="Engine whose /metrics counters are gated when --base is a proxy "
        "in front of it",
    )
    ap.add_argument(
        "--reset-base",
        default=None,
        help="Engine that receives POST /reset_prefix_cache; "
        "defaults to --metrics-base, then --base",
    )
    ap.add_argument(
        "--dp-rank",
        type=int,
        default=-1,
        help="Pin every request to this data-parallel rank when --base is a DP engine",
    )
    ap.add_argument(
        "--hit-block-size",
        type=int,
        required=True,
        help="Tokens per prefix-cache block, as the server logged it",
    )
    ap.add_argument(
        "--lengths",
        default="4352,8448,12544",
        help="Comma-separated minimum prompt lengths in tokens; "
        "each ledger grows to at least this",
    )
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--settle-s",
        type=float,
        default=5.0,
        help="Seconds between the cold and hbm passes",
    )
    ap.add_argument(
        "--salt",
        default=None,
        help="Prompt salt; a fresh one per run keeps the cold pass cold",
    )
    args = ap.parse_args()

    dp_rank = args.dp_rank if args.dp_rank >= 0 else None
    salt = args.salt or f"block-major-{time.time_ns()}"
    lengths = [int(x) for x in args.lengths.split(",") if x]
    client = Client(
        args.base,
        args.model,
        args.timeout,
        args.max_tokens,
        dp_rank,
        metrics_base=args.metrics_base,
        reset_base=args.reset_base or args.metrics_base,
    )

    failures = []
    for length in lengths:
        failures += run_length(client, length, args.hit_block_size, salt, args.settle_s)

    if failures:
        print("BLOCK_MAJOR_OFFLOAD_SMOKE_FAIL:", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        sys.exit(1)
    print("BLOCK_MAJOR_OFFLOAD_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
