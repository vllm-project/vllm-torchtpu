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
"""KV offloading with a global DRAM pool: every tier of one replica's KV.

Tier budgets, in prompts' worth of KV (at the runner's defaults one
prompt offloads 4 whole 4096-token blocks, ~640 MiB):

  HBM               <1 prompt   14 pages/rank: fits only the running
                                request, retains nothing across requests
  host DRAM pool     7 prompts  watermark sweep demotes the oldest
                                blocks to the store node
  host store node  ~25 prompts  16 GiB, holds everything demoted

Stages, the KV movement each one forces, and its pass condition:

  1 seed   12 cold prompts stream through. Each offloads HBM -> host
           pool (D2H); pool pressure demotes the oldest blocks to the
           store node, leaving the ~4 newest prompts pool-resident.
           Pass: external hit rate stays ~0 and every request returns
           token ids.
  2 pool   re-serve the third-newest prompt: out of HBM but above the
           demotion line, so it must load host pool -> HBM (H2D).
           Pass: external hits > 0, first tokens match the cold pass.
  3 node   re-serve the three oldest prompts: the pool provably cannot
           still hold them, so the registry resolves them on the store
           node and read_remote pulls node -> host staging -> HBM.
           Pass: external hits > 0 on >= 2 of 3, first tokens match.

Gates use external-hit deltas plus first-tokens divergence, never full
output equality: TPU serving is not run-to-run deterministic, while
corrupt recalled KV garbles output immediately.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

KEYS = dict(pre_s="request_prefill_time_seconds_sum",
            pre_n="request_prefill_time_seconds_count",
            hit="external_prefix_cache_hits_total",
            q="external_prefix_cache_queries_total")

# Each prompt repeats one distinct token; prompt i repeats BASE_TOKEN + i.
BASE_TOKEN = 700


def scrape(base: str) -> dict[str, float]:
    """Fetch ``base``/metrics and return {name: value} for every vllm
    counter, with the ``vllm:`` prefix stripped and series that differ
    only in labels summed into one value."""
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
    parser.add_argument("--base", required=True, help="http://host:port")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--pool-prompts",
                        type=int,
                        required=True,
                        help="Host pool capacity in prompts' worth of "
                        "offloaded blocks, parsed from the server log by "
                        "the runner. The node-recall stage only proves "
                        "anything while the prompt count exceeds it by "
                        "enough that the oldest prompts cannot still be "
                        "local.")
    parser.add_argument("--prompt-tokens",
                        type=int,
                        default=16448,
                        help="Must EXCEED a whole number of offloaded "
                        "blocks: vLLM caps prefix hits at num_tokens - 1, "
                        "so a prompt of exactly whole blocks can never "
                        "claim its own last block and the lookup "
                        "short-circuits before the registry.")
    parser.add_argument("--sweep-wait-s",
                        type=float,
                        default=15.0,
                        help="Settle time after seeding for the last sweep "
                        "episode to demote and publish.")
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    if args.prompts < args.pool_prompts + 3:
        raise SystemExit(
            f"prompts={args.prompts} vs a pool of {args.pool_prompts} "
            "prompts' worth of KV: seeding must overflow the pool by at "
            "least the three node-recall prompts, or their recall "
            "location is ambiguous")

    def ask(prompt: list[int], max_tokens: int) -> list[int]:
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
            "add_special_tokens": False,
            "return_token_ids": True,
        }
        req = urllib.request.Request(
            args.base + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=args.request_timeout_s) as r:
            return json.loads(r.read())["choices"][0].get("token_ids")

    # Serial requests on a max_num_seqs=1 engine: a per-request metrics
    # delta is exactly that request.
    def measured(prompt: list[int],
                 max_tokens: int) -> tuple[list[int], dict[str, float]]:
        before = scrape(args.base)
        out = ask(prompt, max_tokens)
        after = scrape(args.base)
        d = {k: after.get(v, 0) - before.get(v, 0) for k, v in KEYS.items()}
        return out, {
            "prefill": d["pre_s"] / (d["pre_n"] or 1),
            "hit": d["hit"],
            "q": d["q"],
        }

    failures = []
    plen = args.prompt_tokens

    def prompt_ids(i: int) -> list[int]:
        return [BASE_TOKEN + i] * plen

    # ---- stage 1: cold seed ----
    print(f"=== stage 1: seeding {args.prompts} prompts of {plen} tokens "
          f"into a pool that holds {args.pool_prompts} ===")
    cold: list[list[int]] = []
    cold_hits = cold_qs = 0.0
    for i in range(args.prompts):
        out, r = measured(prompt_ids(i), 8)
        cold.append(out or [])
        cold_hits += r["hit"]
        cold_qs += r["q"]
        print(
            f"[seed] prompt{i} prefill={r['prefill']:.3f}s "
            f"ext_hit={int(r['hit'])}/{int(r['q'])}",
            flush=True)
    if cold_qs and 100 * cold_hits / cold_qs > 5:
        failures.append(
            f"cold seed ext_hit_pct={100 * cold_hits / cold_qs:.1f}: fresh "
            "prompts are not cold, the recall stages are invalid")
    if any(not c for c in cold):
        failures.append("a cold seed request returned no token_ids")

    print(f"waiting {args.sweep_wait_s:.0f}s for the last sweep episode")
    time.sleep(args.sweep_wait_s)

    def recall(tag: str, i: int) -> float:
        """Re-serve prompt ``i`` and return its external-hit token count.

        ``tag`` names the tier the hit is expected from ("pool"/"node")
        and only labels the log line. Appends a failure if the recalled
        output's first tokens diverge from the cold pass (the signature
        of corrupt restored KV); the caller judges the hit count.
        """
        out, r = measured(prompt_ids(i), 8)
        print(
            f"[{tag}] prompt{i} prefill={r['prefill']:.3f}s "
            f"ext_hit={int(r['hit'])}/{int(r['q'])}",
            flush=True)
        if out and cold[i] and out[:2] != cold[i][:2]:
            failures.append(
                f"{tag}: prompt{i} diverged at the first tokens (corrupt "
                f"recalled KV?): cold={cold[i]} recalled={out}")
        return r["hit"]

    # ---- stage 2: host pool recall ----
    # The third-newest prompt: out of the tiny HBM cache (which retains
    # under two finished prompts), but above the demotion line — the sweep
    # only demotes down to the high watermark, which at the runner's pool
    # sizing leaves at least the three newest prompts' blocks resident.
    print("=== stage 2: recall from the host DRAM pool ===")
    if recall("pool", args.prompts - 3) <= 0:
        failures.append(
            "host pool recall produced no external hits — the prompt's "
            "blocks fell out of the host pool (over-demotion?) or "
            "offloading never admitted them")

    # ---- stage 3: store node recall ----
    # The three oldest prompts. Capacity arithmetic (prompts >= pool + 3)
    # puts them beyond what the host pool could still hold, so an external
    # hit here can only have been resolved through the registry and read
    # back from the store node. One straggler is tolerated; zero hits
    # overall means demotion dropped the blocks instead of moving them.
    print("=== stage 3: recall from the store node ===")
    node_hits = [recall("node", i) for i in range(3)]
    hit_prompts = sum(1 for h in node_hits if h > 0)
    if hit_prompts == 0:
        failures.append(
            "store node recall produced no external hits on any of the "
            "oldest prompts — demotion dropped their blocks instead of "
            "moving them to the store node (no placement target? sweep "
            "dead?)")
    elif hit_prompts < 2:
        failures.append(
            f"store node recall hit only {hit_prompts}/3 oldest prompts — "
            "demotion or recall is losing blocks")

    if failures:
        print("RESULT: FAIL (" + "; ".join(failures) + ")")
        return 1
    print(f"RESULT: PASS (cold seed stayed cold; host pool recall hit; "
          f"store node recall hit {hit_prompts}/3 oldest prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
