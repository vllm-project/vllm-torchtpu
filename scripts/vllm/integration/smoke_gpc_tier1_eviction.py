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
"""Global DRAM pool with both tiers full: a full tier 1 must make room.

The runner sizes tier 0 (the replica's own host pool) and tier 1 (a host
store node) to a handful of prompts each and passes their real block counts
and sweep watermarks here. Every prompt offloads exactly --prompt-blocks
whole blocks. From those numbers this works out where each prompt must be
once seeding has settled. Ranks count back from the newest prompt (rank 0):

  tier 0   the newest prompts: C0 - ceil(hi0 * C0) blocks just after a
           sweep, up to C0 - ceil(lo0 * C0) just before the next.
  tier 1   the next older ones: tier 1's own sweep keeps it between
           C1 - ceil(hi1 * C1) and C1 - ceil(lo1 * C1) blocks. A demotion that
           leaves it under its low watermark wakes that sweep, which drops
           tier 1's oldest blocks, locally and from the registry.
  gone     everything older.

Every recall first empties HBM with POST /reset_prefix_cache (the offload
store keeps its entries), so a hit can only come from tier 0 or tier 1.
A recall from tier 1 leaves the layout alone: the peer read lands straight
in HBM without a host copy, and the connector does not store chunks it
loaded. Only a miss stores its blocks again, which is why the stage that
expects misses runs last.

  1 seed      prompts 0..N-1, cold. Partway, once tier 0 has demoted
              prompt 0 but before tier 1 can have evicted anything, recall
              prompt 0. Gates: seeding gets no external hit; prompt 0 comes
              back as a full-prefix external hit, so demotion to tier 1
              works.
  2 landed    recall, oldest first, the prompts demoted after tier 1 had
              filled that its retention must still hold. Gate: none comes
              back without an external hit, and all but at most one come
              back in full. A node that refuses writes once full keeps only
              the first prompts it received and loses all of these.
  3 dropped   recall prompts 1-3, demoted right after prompt 0 and then
              evicted by tier 1 to make that room. Gate: no external hit;
              an entry the node failed to unregister would claim one.

Every recall must also return the cold pass's first tokens, and none may
hit HBM after the reset.
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
    local="prefix_cache_hits_total",
    ext="external_prefix_cache_hits_total",
    ext_q="external_prefix_cache_queries_total",
)

# Prompt i repeats token BASE_TOKEN + i, so no two prompts share a block.
BASE_TOKEN = 700
# Tokens past the last whole block: vLLM caps a hit at num_tokens - 1, so a
# prompt of exactly whole blocks could never hit its last one.
TAIL_TOKENS = 8
FIRST_TOKENS = 2


def scrape(base: str) -> dict[str, float]:
    """Every vllm counter at ``base``/metrics, ``vllm:`` stripped, series that
    differ only in labels summed."""
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
        text = r.read().decode()
    m: dict[str, float] = {}
    for line in text.splitlines():
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo:
            m[mo.group(1)] = m.get(mo.group(1), 0.0) + float(mo.group(2))
    return m


def post(base: str, path: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        return e.code, {}


class Layout:
    """Where the prompts sit once seeding has settled, as ranks from the
    newest prompt, and the seeding point of the prompt-0 checkpoint."""

    def __init__(self, args: argparse.Namespace):
        b = args.prompt_blocks
        c0, c1 = args.tier0_blocks, args.tier1_blocks
        self.u0_min = c0 - math.ceil(args.tier0_high * c0)
        self.u0_max = c0 - math.ceil(args.tier0_low * c0)
        self.u1_min = c1 - math.ceil(args.tier1_high * c1)
        self.u1_max = c1 - math.ceil(args.tier1_low * c1)
        # Wholly below tier 0, and wholly within what tier 1 still holds.
        self.first_landed = math.ceil(self.u0_max / b)
        self.last_landed = (self.u0_min + self.u1_min) // b - 1
        # Wholly beyond both tiers.
        self.first_gone = math.ceil((self.u0_max + self.u1_max) / b)
        # Prompts seeded before recalling prompt 0: one past what tier 0 can
        # hold, so its blocks have all been demoted.
        self.checkpoint = self.first_landed + 2
        n = args.prompts
        self.landed = list(range(n - 1 - self.last_landed, n - self.first_landed))
        self.dropped = [1, 2, 3]

    def problems(self, args: argparse.Namespace) -> list[str]:
        b, n = args.prompt_blocks, args.prompts
        out = []
        if len(self.landed) < 3:
            out.append(
                f"tier 1 retains {len(self.landed)} whole prompts below tier "
                "0, fewer than 3: give it more blocks or narrow its watermarks"
            )
        if n - 1 - self.dropped[-1] < self.first_gone:
            out.append(
                f"prompt {self.dropped[-1]} is rank {n - 1 - self.dropped[-1]}, "
                f"but only ranks >= {self.first_gone} are surely gone: seed "
                "more prompts"
            )
        if self.checkpoint * b - self.u0_min > self.u1_max:
            out.append(
                "tier 1 could start evicting before the prompt-0 checkpoint: "
                "give tier 1 more blocks relative to tier 0"
            )
        # A node that never evicts keeps the first C1 blocks it was sent:
        # prompts 0..ceil(C1/b)-1. Stage 2 only catches that if every landed
        # prompt is newer.
        if self.landed and self.landed[0] <= math.ceil(args.tier1_blocks / b):
            out.append(
                f"landed prompt {self.landed[0]} would also sit on a tier 1 "
                "that never evicts, so stage 2 could not catch it: seed more "
                "prompts"
            )
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="http://host:port")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hit-block-size", type=int, required=True)
    parser.add_argument("--prompts", type=int, default=28)
    parser.add_argument("--prompt-blocks", type=int, default=4)
    parser.add_argument("--tier0-blocks", type=int, required=True)
    parser.add_argument("--tier0-low", type=float, required=True)
    parser.add_argument("--tier0-high", type=float, required=True)
    parser.add_argument("--tier1-blocks", type=int, required=True)
    parser.add_argument("--tier1-low", type=float, required=True)
    parser.add_argument("--tier1-high", type=float, required=True)
    parser.add_argument(
        "--settle-s",
        type=float,
        default=15.0,
        help="Wait before a recall stage, for sweeps and in-flight "
        "demotions to finish.",
    )
    parser.add_argument(
        "--seed-pace-s",
        type=float,
        default=2.0,
        help="Pause after each seed, so tier 0's sweep keeps up with the "
        "stream instead of dropping blocks it had no time to demote.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    lay = Layout(args)
    full_prefix = args.prompt_blocks * args.hit_block_size
    plen = full_prefix + TAIL_TOKENS
    print(
        f"tier 0: {args.tier0_blocks} blocks, holds {lay.u0_min}-{lay.u0_max}; "
        f"tier 1: {args.tier1_blocks} blocks, holds {lay.u1_min}-{lay.u1_max}; "
        f"{args.prompt_blocks} blocks ({plen} tokens) per prompt"
    )
    print(
        f"{args.prompts} prompts; checkpoint after {lay.checkpoint}; landed "
        f"{lay.landed} (ranks {lay.first_landed}-{lay.last_landed}); dropped "
        f"{lay.dropped} (ranks >= {lay.first_gone} gone)"
    )
    bad = lay.problems(args)
    if bad:
        print("RESULT: FAIL (tier sizing: " + "; ".join(bad) + ")")
        return 1

    def prompt_ids(i: int) -> list[int]:
        return [BASE_TOKEN + i] * plen

    def ask(i: int) -> tuple[list[int], dict[str, float]]:
        """Serve prompt ``i``; return its token ids and counter deltas. The
        engine runs one sequence at a time, so a delta is this request."""
        before = scrape(args.base)
        status, body = post(
            args.base,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt_ids(i),
                "max_tokens": 8,
                "temperature": 0.0,
                "seed": 0,
                "ignore_eos": True,
                "add_special_tokens": False,
                "return_token_ids": True,
            },
            args.request_timeout_s,
        )
        after = scrape(args.base)
        if status != 200:
            raise SystemExit(f"prompt{i}: HTTP {status}")
        out = body["choices"][0].get("token_ids") or []
        return out, {k: after.get(v, 0) - before.get(v, 0) for k, v in KEYS.items()}

    def reset_hbm() -> None:
        for _ in range(10):
            status, body = post(args.base, "/reset_prefix_cache", {}, 120)
            if status == 404:
                raise SystemExit(
                    "POST /reset_prefix_cache is not mounted: serve with "
                    "VLLM_SERVER_DEV_MODE=1"
                )
            if status == 200 and body.get("success"):
                return
            # Refused while blocks are held, e.g. by an in-flight transfer.
            time.sleep(2)
        raise SystemExit("POST /reset_prefix_cache never succeeded")

    failures: list[str] = []
    cold: dict[int, list[int]] = {}

    def recall(tag: str, i: int) -> float:
        """Re-serve prompt ``i`` from an empty HBM; return its external-hit
        tokens. Records a failure for an HBM hit or diverged first tokens;
        the caller judges the hit count."""
        reset_hbm()
        out, d = ask(i)
        print(
            f"[{tag}] prompt{i} ext_hit={int(d['ext'])}/{int(d['ext_q'])} "
            f"hbm_hit={int(d['local'])} tokens={out}",
            flush=True,
        )
        if d["local"] > 0:
            failures.append(
                f"{tag}: prompt{i} hit HBM ({int(d['local'])} tokens) after "
                "the reset, so the recall tested no tier"
            )
        if out[:FIRST_TOKENS] != cold[i][:FIRST_TOKENS]:
            failures.append(
                f"{tag}: prompt{i} diverged from the cold pass at the first "
                f"tokens (corrupt recalled KV?): cold={cold[i]} recalled={out}"
            )
        return d["ext"]

    # ---- stage 1: seed, with the prompt-0 checkpoint ----
    print(f"=== stage 1: seeding {args.prompts} prompts ===")
    seed_hits = 0.0
    for i in range(args.prompts):
        out, d = ask(i)
        cold[i] = out
        seed_hits += d["ext"]
        print(f"[seed] prompt{i} ext_hit={int(d['ext'])}/{int(d['ext_q'])}", flush=True)
        if not out:
            failures.append(f"seed: prompt{i} returned no token_ids")
        time.sleep(args.seed_pace_s)
        if i + 1 == lay.checkpoint:
            time.sleep(args.settle_s)
            hit = recall("checkpoint", 0)
            if hit != full_prefix:
                failures.append(
                    f"checkpoint: prompt0 recalled {int(hit)} external-hit "
                    f"tokens, not the full {full_prefix}: tier 0 did not "
                    "demote it to tier 1 intact"
                )
    if seed_hits:
        failures.append(
            f"seed: {int(seed_hits)} external-hit tokens on distinct cold "
            "prompts, so the recall stages measure nothing"
        )
    time.sleep(args.settle_s)

    # ---- stage 2: demoted into a full tier 1, still there ----
    print("=== stage 2: prompts demoted after tier 1 filled ===")
    landed = {i: recall("landed", i) for i in lay.landed}
    missing = [i for i, h in landed.items() if h == 0]
    partial = [i for i, h in landed.items() if 0 < h < full_prefix]
    if missing:
        failures.append(
            f"landed: prompts {missing} came back with no external hit, so "
            "their demotion into the full tier 1 was refused or dropped"
        )
    if len(missing) + len(partial) > 1:
        failures.append(
            f"landed: only {len(landed) - len(missing) - len(partial)}/"
            f"{len(landed)} prompts came back in full (partial: {partial})"
        )

    # ---- stage 3: evicted from tier 1 to make that room ----
    print("=== stage 3: oldest prompts, evicted from tier 1 ===")
    still = [i for i in lay.dropped if recall("dropped", i) > 0]
    if still:
        failures.append(
            f"dropped: prompts {still} still resolve to external hits, so "
            "tier 1 kept its oldest blocks instead of dropping them (or "
            "dropped them without unregistering)"
        )

    if failures:
        print("RESULT: FAIL (" + "; ".join(failures) + ")")
        return 1
    full = len(landed) - len(partial)
    print(
        f"RESULT: PASS (prompt 0 reached tier 1; {full}/{len(landed)} prompts "
        "demoted into the full tier 1 came back in full; prompts 1-3 were "
        "evicted from it)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
