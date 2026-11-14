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
"""SHM crash-recovery for the Raiden offload host pool: does an offloaded block
survive a hard engine crash and reload as the same KV on restart?

Offload SAVE (HBM -> host pool, D2H) happens per request whenever new blocks are
produced; it does NOT require HBM eviction. What eviction (or a large HBM cache)
changes is only whether a RECALL is served locally from HBM before the external
store is ever queried. So to prove an offloaded block round-trips through the
host pool, the recall must run against an EMPTY HBM cache: this rig empties it
with POST /reset_prefix_cache (the connector's reset_cache() is deliberately not
reached, so the external/shm store keeps its entries), exactly like the proven
smoke_basic_offload_correctness.py. This is more robust than starving HBM to
force eviction, and it keeps the OOM-free serving config (large HBM override,
seqs>1) that boots this hybrid model reliably.

Prompts are raw token-id lists (prompt i = [BASE_TOKEN+i] * prompt_tokens), so
every prompt is exactly a whole number of blocks and trivially distinct.
Correctness is judged by the first generated tokens matching the pre-crash pass
(corrupt recalled KV garbles them immediately), never by full output equality --
TPU serving is not run-to-run deterministic.

A real crash crosses a process boundary, so the test runs three phases, each its
own engine, driven by this script once per phase over a shared --state file:

  populate       RAIDEN_SHM_KEY=K. Seed --prompts distinct prompts (offloading
                 each to the shm-backed host pool), then POST /reset_prefix_cache
                 to empty HBM and recall the target, proving IN PROCESS it comes
                 back as an EXTERNAL hit with matching first tokens. Record the
                 answer. The rig then SIGKILLs without unlinking /dev/shm.
  verify-recover RAIDEN_SHM_KEY=K, fresh engine/HBM. Recall the same target: it
                 must be an EXTERNAL hit with first tokens matching the pre-crash
                 answer -- the pool + metadata recovered from shm. (positive, P)
  verify-cold    a DIFFERENT RAIDEN_SHM_KEY, fresh engine. Recall the target: it
                 must MISS (0 external hits), proving P's hit is attributable to
                 the shared key and not to /dev/shm residue. (negative, N)

Attribution: populate red => the harness/offload path is broken (trust nothing
below). populate green + verify-recover red => a FEATURE bug in the vLLM offload
glue (Raiden recovery itself is proven by kv_cache_store_recovery_e2e_test).
verify-cold showing any external hit => the signal is not attributable and the
test is invalid.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

_KEYS = dict(
    hit="external_prefix_cache_hits_total",
    q="external_prefix_cache_queries_total",
    local_hit="prefix_cache_hits_total",
    local_q="prefix_cache_queries_total",
    pre_s="request_prefill_time_seconds_sum",
    pre_n="request_prefill_time_seconds_count",
)

# prompt i repeats the single token BASE_TOKEN + i, prompt_tokens times.
BASE_TOKEN = 700


def _scrape(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
        text = r.read().decode()
    m = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo:
            m[mo.group(1)] = m.get(mo.group(1), 0.0) + float(mo.group(2))
    return m


class Rig:
    def __init__(self, args):
        self.base = args.base
        self.model = args.model
        self.timeout = args.request_timeout_s
        self.plen = args.prompt_tokens
        self.recall_tokens = args.recall_tokens

    def prompt_ids(self, i):
        return [BASE_TOKEN + i] * self.plen

    def _ask(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.recall_tokens,
            "temperature": 0.0,
            "seed": 0,
            "ignore_eos": True,
            "add_special_tokens": False,
            "return_token_ids": True,
        }
        req = urllib.request.Request(
            self.base + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0].get("token_ids")

    def measured(self, i):
        """Serve prompt i and return (generated_token_ids, metrics-delta). The
        delta is this one request only when serving is serial; batching just
        widens it and never flips an external hit to a local one."""
        before = _scrape(self.base)
        out = self._ask(self.prompt_ids(i))
        after = _scrape(self.base)
        d = {k: after.get(v, 0) - before.get(v, 0) for k, v in _KEYS.items()}
        return out or [], {
            "hit": d["hit"],
            "q": d["q"],
            "local_hit": d["local_hit"],
            "local_q": d["local_q"],
            "prefill": d["pre_s"] / (d["pre_n"] or 1),
        }

    def reset_prefix_cache(self, attempts=10):
        """Empty the HBM prefix cache while the external/shm store keeps its
        entries: POST /reset_prefix_cache WITHOUT ?reset_external, so the
        connector's reset_cache() is never reached. Needs VLLM_SERVER_DEV_MODE=1
        on the server for the endpoint to be mounted. Returns True on success."""
        req = urllib.request.Request(
            self.base + "/reset_prefix_cache",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        for _ in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = json.loads(r.read() or b"{}")
                # 200 with success=false means blocks are still held (e.g. an
                # in-flight offload transfer); retry rather than trust it.
                if r.status == 200 and body.get("success"):
                    return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise SystemExit(
                        "POST /reset_prefix_cache is not mounted: relaunch the "
                        "server with VLLM_SERVER_DEV_MODE=1"
                    )
            except urllib.error.URLError:
                pass
            time.sleep(2)
        return False


def populate(rig, args):
    """Seed prompts (offloading each), empty HBM, recall the target, record."""
    n, target = args.prompts, args.target
    fails = []

    print(
        f"=== seeding {n} prompts of {rig.plen} tokens (target=prompt{target}) ===",
        flush=True,
    )
    cold = {}
    cold_hits = cold_qs = 0.0
    for i in range(n):
        out, r = rig.measured(i)
        cold[i] = out
        cold_hits += r["hit"]
        cold_qs += r["q"]
        print(
            f"  [seed] prompt{i} prefill={r['prefill']:.3f}s "
            f"ext_hit={int(r['hit'])}/{int(r['q'])} gen={out[:4]}",
            flush=True,
        )
        if not out:
            fails.append(f"populate: seed prompt{i} returned no token_ids")
    if cold_qs and 100 * cold_hits / cold_qs > 5:
        fails.append(
            f"populate: cold seed ext_hit_pct={100 * cold_hits / cold_qs:.1f} "
            "(>5%): the prompts are not fresh against this store, so a later "
            "'recovered' hit would be indistinguishable from a stale seed hit"
        )
    if not cold.get(target):
        return fails + [f"populate: target prompt{target} produced no baseline"]

    print(
        f"  waiting {args.sweep_wait_s:.0f}s for offload (D2H save) to settle",
        flush=True,
    )
    time.sleep(args.sweep_wait_s)

    # Diagnostic: recall the target BEFORE clearing HBM. With a large HBM cache
    # the target is still resident, so this is expected to be a LOCAL hit
    # (ext_hit=0, fast prefill) -- it is logged, never gated, to make the
    # local-vs-external contrast explicit in the run log.
    out, r = rig.measured(target)
    print(
        f"  [pre-reset recall] prompt{target} prefill={r['prefill']:.3f}s "
        f"ext_hit={int(r['hit'])}/{int(r['q'])} "
        f"local_hit={int(r['local_hit'])}/{int(r['local_q'])} gen={out[:4]}",
        flush=True,
    )

    # Empty HBM so the authoritative recall cannot be served locally and must
    # resolve through the external (shm-backed) store.
    print("  POST /reset_prefix_cache (external store kept)", flush=True)
    if not rig.reset_prefix_cache():
        return fails + [
            "populate: /reset_prefix_cache never succeeded, so HBM "
            "cannot be proven empty and an external hit is unprovable"
        ]

    out, r = rig.measured(target)
    print(
        f"  [recall] prompt{target} prefill={r['prefill']:.3f}s "
        f"ext_hit={int(r['hit'])}/{int(r['q'])} "
        f"local_hit={int(r['local_hit'])}/{int(r['local_q'])} gen={out[:4]}",
        flush=True,
    )
    if r["local_hit"] > 0:
        fails.append(
            f"populate: recall of prompt{target} still saw {int(r['local_hit'])} "
            "LOCAL hits after /reset_prefix_cache -- HBM was not actually "
            "cleared, so an external hit here cannot be attributed to the pool"
        )
    if r["hit"] <= 0:
        fails.append(
            f"populate: recall of prompt{target} produced 0 external hits after "
            "HBM was cleared -- its KV was never offloaded to the shm-backed "
            "host pool (offload SAVE not happening for this config), or the "
            "store cannot resolve it. Nothing is in shm to recover, so the "
            "recovery legs would prove nothing"
        )
    if out and cold[target] and out[:2] != cold[target][:2]:
        fails.append(
            f"populate: in-process recall of prompt{target} diverged at the "
            f"first tokens (corrupt offloaded KV?): cold={cold[target][:4]} "
            f"recalled={out[:4]}"
        )
    if fails:
        return fails

    state = {
        "base_token": BASE_TOKEN,
        "prompt_tokens": rig.plen,
        "prompts": n,
        "target": target,
        "recall_tokens": rig.recall_tokens,
        "cold_target_tokens": cold[target],
    }
    with open(args.state, "w") as f:
        json.dump(state, f)
    print(
        f"  recorded pre-crash state -> {args.state} "
        f"(target=prompt{target}, expect an external hit on recovery)",
        flush=True,
    )
    return []


def _load_state(args, rig):
    with open(args.state) as f:
        st = json.load(f)
    # Reconstruct the exact target prompt the populate phase offloaded.
    global BASE_TOKEN
    BASE_TOKEN = st["base_token"]
    rig.plen = st["prompt_tokens"]
    rig.recall_tokens = st["recall_tokens"]
    return st


def verify_recover(rig, args):
    """Fresh engine, same key: the target must recover from shm as an ext hit."""
    st = _load_state(args, rig)
    target = st["target"]
    out, r = rig.measured(target)
    print(
        f"  [recover] prompt{target} prefill={r['prefill']:.3f}s "
        f"ext_hit={int(r['hit'])}/{int(r['q'])} gen={out[:4]}",
        flush=True,
    )
    fails = []
    if r["hit"] <= 0:
        fails.append(
            f"verify-recover: 0 external hits recalling prompt{target} after "
            "restart under the same key -- the offloaded KV did NOT recover "
            "from shm. Raiden recovery is proven by "
            "kv_cache_store_recovery_e2e_test, so the fault is the vLLM offload "
            "glue (enable_shm wiring, or a block-hash namespace not stable "
            "across the restart -- is PYTHONHASHSEED pinned?)"
        )
    if out and st["cold_target_tokens"] and out[:2] != st["cold_target_tokens"][:2]:
        fails.append(
            "verify-recover: recovered output diverged at the first tokens, so "
            "the KV reloaded from shm is not the KV that was offloaded\n"
            f"    pre-crash={st['cold_target_tokens'][:4]}\n"
            f"    recovered={out[:4]}"
        )
    return fails


def verify_cold(rig, args):
    """Fresh engine, DIFFERENT key: the target must miss (attribution)."""
    st = _load_state(args, rig)
    target = st["target"]
    out, r = rig.measured(target)
    print(
        f"  [cold-neg] prompt{target} prefill={r['prefill']:.3f}s "
        f"ext_hit={int(r['hit'])}/{int(r['q'])} gen={out[:4]}",
        flush=True,
    )
    fails = []
    if r["hit"] > 0:
        fails.append(
            f"verify-cold: recalling prompt{target} under a DIFFERENT shm key "
            f"still produced {int(r['hit'])} external hits. The recovery hit is "
            "therefore NOT attributable to the shared key -- residual /dev/shm "
            "or a shared cache is leaking across keys, and the test is invalid "
            "until this leg reads zero"
        )
    return fails


PHASES = {
    "populate": (populate, "SHM_POPULATE"),
    "verify-recover": (verify_recover, "SHM_RECOVER"),
    "verify-cold": (verify_cold, "SHM_COLD"),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    ap.add_argument(
        "--state",
        required=True,
        help="JSON file carrying the pre-crash target across phases",
    )
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompts",
        type=int,
        default=12,
        help="distinct prompts to seed; must overflow HBM so the "
        "target is evicted into the host pool",
    )
    ap.add_argument(
        "--prompt-tokens",
        dest="prompt_tokens",
        type=int,
        default=16448,
        help="tokens per prompt; must exceed a whole number of "
        "offload blocks (vLLM caps prefix hits at num_tokens-1)",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=0,
        help="prompt index to recover; 0 (oldest) is the most "
        "surely evicted from HBM and, with no store node, stays "
        "resident in the host pool",
    )
    ap.add_argument(
        "--sweep-wait-s",
        dest="sweep_wait_s",
        type=float,
        default=15.0,
        help="settle time after seeding for offload",
    )
    ap.add_argument(
        "--recall-tokens",
        dest="recall_tokens",
        type=int,
        default=8,
        help="generation budget per request",
    )
    ap.add_argument(
        "--request-timeout-s", dest="request_timeout_s", type=float, default=900.0
    )
    args = ap.parse_args()

    rig = Rig(args)
    fn, marker = PHASES[args.phase]
    print(f"=== shm-recovery phase={args.phase} state={args.state} ===", flush=True)
    failures = fn(rig, args)
    print(flush=True)
    if failures:
        print(f"{marker}_FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print(f"{marker}_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
