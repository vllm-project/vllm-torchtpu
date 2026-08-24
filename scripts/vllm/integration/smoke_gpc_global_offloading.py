#!/usr/bin/env python3
"""Global prefix cache smoke: two offload replicas sharing one registry.

Server A and server B are identical standalone engines whose
TPURaidenOffloadingConnector stores publish to the same global registry, so a
prefix computed and DRAM-cached by A must be a byte-compatible remote hit
for B, pulled P2P via read_remote. Two-host CI version of the
qwen35-gpc-global-offloading recipe's check pod: each replica owns a full
host, and the pull crosses the network between them.

The probes are long filler documents, each exceeding one offload span so
its first span is publishable. The two engines are deterministic mirrors:
the same document produces byte-identical output on either one. A replay
through B whose span was restored from A's
published KV must therefore continue the way A's own cold computation
did, while corrupt restored KV garbles the continuation from the first
tokens.

Gates (never answer correctness or full output equality: the model's
long-document behavior is content-sensitive, and the stack is not
guaranteed run-to-run deterministic):
  1. Every document seeded cold through A and replayed through B must
     produce external prefix-cache hits on B: the replay pulled from the
     shared cache instead of recomputing.
  2. At least two of the three replays must agree with A's cold output on
     the first eight characters. Corrupt restored KV diverges immediately
     on every document; a single disagreement is within the stack's
     nondeterminism allowance.
  3. A fresh document through B must produce zero external hits, so the
     hit gate cannot pass vacuously on a stack that hits on everything.
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request

KEYS = dict(pre_s="request_prefill_time_seconds_sum",
            pre_n="request_prefill_time_seconds_count",
            hit="external_prefix_cache_hits_total",
            q="external_prefix_cache_queries_total")

FILLER = "the archive shelf holds ledgers sorted by year and colour."


def build_prompt(tag: str, code: str, lines: int) -> str:
    body = [f"line {i:04d}: {FILLER}" for i in range(lines)]
    # The tagged code lines differentiate the documents inside their first
    # block, so no document can hit another's published span, and the
    # closing question pins the decode onto document content. Whether the
    # model retrieves the code is reported but never gated: its
    # long-document retrieval is content-sensitive, and the gates compare
    # B's replay against A's cold output instead.
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
    parser.add_argument("--base-a",
                        required=True,
                        help="http://host:port of server A (the seeder)")
    parser.add_argument("--base-b",
                        required=True,
                        help="http://host:port of server B (the replayer)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-prompt-tokens",
                        type=int,
                        default=16385,
                        help="One full offload block span plus one "
                        "(block_size x pcp + 1): vLLM caps prefix hits at "
                        "num_tokens - 1, so an exactly-one-span prompt can "
                        "never claim its own span and the lookup "
                        "short-circuits before the registry.")
    parser.add_argument("--publish-wait-s", type=float, default=15.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    def api(path: str, base: str, payload: dict) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req,
                                        timeout=args.request_timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            raise SystemExit(
                f"{base}{path} returned HTTP {err.code}: {body}") from err

    def ask(base: str, prompt: str) -> tuple[str, int]:
        resp = api(
            "/v1/chat/completions", base, {
                "model": args.model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": 16,
                "temperature": 0.0,
                "seed": 0,
            })
        return (resp["choices"][0]["message"]["content"],
                resp["usage"]["prompt_tokens"])

    # Serial requests on quiet engines: a per-request metrics delta on
    # server B is exactly that request.
    def measured(prompt: str) -> tuple[str, dict[str, float]]:
        before = scrape(args.base_b)
        text, _ = ask(args.base_b, prompt)
        after = scrape(args.base_b)
        d = {k: after.get(v, 0) - before.get(v, 0) for k, v in KEYS.items()}
        return text, {
            "prefill": d["pre_s"] / (d["pre_n"] or 1),
            "hit": d["hit"],
            "q": d["q"]
        }

    # Distinct tags and codes per document: they force the documents apart
    # inside the first block, so no probe can hit another probe's published
    # span.
    seeds = [("emerald", "739284"), ("obsidian", "506192"),
             ("amethyst", "584317")]
    fresh_tag, fresh_code = "crimson", "281473"

    # Size the documents on the serving tokenizer: price a 100-line sample,
    # then aim one offload span plus a margin that absorbs tokenizer
    # drift. A client-side line-count estimate drifts with the tokenizer;
    # the server's own count does not.
    sample = api(
        "/tokenize", args.base_a, {
            "model": args.model,
            "prompt": build_prompt(seeds[0][0], seeds[0][1], 100),
        })
    per_line = sample["count"] / 100
    target = args.min_prompt_tokens + 2048
    if target > sample["max_model_len"]:
        print(f"RESULT: FAIL (the probes need {target} tokens but "
              f"max_model_len is only {sample['max_model_len']})")
        return 1
    lines = math.ceil(target / per_line)
    print(f"sizing: {per_line:.1f} tokens/line, "
          f"max_model_len={sample['max_model_len']}, {lines} lines "
          f"targeting {target} tokens")

    failures = []

    print(f"=== seeding {len(seeds)} documents cold through A ===")
    cold_answers = []
    for tag, code in seeds:
        ans, tokens = ask(args.base_a, build_prompt(tag, code, lines))
        print(f"seed {tag}: prompt_tokens={tokens} answer={ans!r}")
        cold_answers.append(ans)
        if tokens < args.min_prompt_tokens:
            print(f"RESULT: FAIL (the {tag} document is {tokens} tokens, "
                  f"below the {args.min_prompt_tokens}-token offload span "
                  "despite the tokenizer-based sizing above)")
            return 1
    # The doubled wait covers server A's very first save+publish.
    time.sleep(2 * args.publish_wait_s)

    print("=== replaying through B ===")
    matches = 0
    for (tag, code), cold_ans in zip(seeds, cold_answers):
        ans, rep = measured(build_prompt(tag, code, lines))
        match = ans[:8] == cold_ans[:8]
        matches += match
        print(f"replay {tag}: answer={ans!r} prefix_match={match} "
              f"prefill={rep['prefill']:.3f}s "
              f"ext_hit={int(rep['hit'])}/{int(rep['q'])}")
        if rep["hit"] <= 0:
            failures.append(
                f"the {tag} replay produced no external hits on server B; "
                "it recomputed instead of pulling from the shared cache")
    if matches < len(seeds) - 1:
        failures.append(
            f"only {matches}/{len(seeds)} replays continued the way A's "
            "cold computation did; corrupt restored KV diverges from the "
            "first tokens on every document")

    print("=== fresh document through B (cold control) ===")
    ans_f, cold = measured(build_prompt(fresh_tag, fresh_code, lines))
    print(f"cold: answer={ans_f!r} prefill={cold['prefill']:.3f}s "
          f"ext_hit={int(cold['hit'])}/{int(cold['q'])}")
    if cold["hit"] > 0:
        failures.append(
            f"the fresh document produced {int(cold['hit'])} external "
            "hits: it is not cold, so the replay gate proves nothing")

    if failures:
        print("RESULT: FAIL (" + "; ".join(failures) + ")")
        return 1
    print(f"RESULT: PASS ({matches}/{len(seeds)} replays matched A's cold "
          "output on their first tokens with external hits on every "
          "replay; the fresh document stayed cold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
