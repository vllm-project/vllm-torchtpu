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
"""Prefix-cache correctness on a hybrid (attention + recurrent state) model.
Two stages, each comparing greedy token ids against an uncached run of the
same tokens:

  multiturn  Turn N+1 = turn N's prompt + its answer + a new user message.
             The reference is the same request under a different cache_salt
             (never hits). Pass: turn N+1 hits every whole block of turn N's
             prompt, the reference hits nothing, token ids equal (logprobs
             within --logprob-atol).

  eviction   A victim prompt is served cold, repeated as an HBM-hit control,
             then unrelated prompts flood the (small) HBM pool until the
             victim is evicted, then the victim is served again. Pass: the
             re-serve has fewer local hits than the whole prefix, external
             hits > 0, cached_tokens == local + external hits, and token ids
             equal the cold pass.

Run with max_num_batched_tokens == the KV block size: K3's greedy output
depends on prefill chunk geometry, so cached and uncached runs must prefill
in the same chunk shapes for token equality to hold.
"""

import argparse
import json
import math
import random
import re
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

FILLER = (
    "the clerk weighs the crate and signs the barge manifest",
    "the night shift reconciles the tally against the river gauge",
    "two sealed drums of indigo dye move into the cold store",
    "the depot ledger is bound in green cloth for the dry season",
    "a courier carries the duplicate slip up to the customs post",
    "the foreman notes a delay of one hour at the lower lock",
    "sacks of milled rice are stacked three high along the north wall",
    "the harbour master stamps the release order at first light",
)


def _post(base, path, payload, timeout):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"body": e.read().decode(errors="replace")[:400]}


def _metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo:
            out[mo.group(1)] = out.get(mo.group(1), 0.0) + float(mo.group(2))
    return out


def ledger(namespace, lines):
    """Distinct from line 1, so each namespace owns its hash chain from block
    0, with the fact the question asks for inside the recalled prefix."""
    code = random.Random(namespace).randrange(100000, 1000000)
    head = [
        f"Depot archive {namespace}: quarterly shipment ledger.",
        f"The manifest code for archive {namespace} is {code}.",
    ]
    body = [f"Entry {i:04d}: {FILLER[i % len(FILLER)]}." for i in range(lines)]
    return "\n".join(head + body)


class Client:
    def __init__(self, base, model, timeout, max_tokens, logprobs):
        self.base, self.model, self.timeout = base, model, timeout
        self.max_tokens, self.logprobs = max_tokens, logprobs

    def count(self, messages):
        status, body = _post(
            self.base,
            "/tokenize",
            {"model": self.model, "messages": messages, "add_special_tokens": False},
            120,
        )
        if status != 200:
            raise SystemExit(f"/tokenize failed (HTTP {status}): {body}")
        return body["count"]

    def ask(self, tag, messages, cache_salt=None, max_tokens=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "return_token_ids": True,
        }
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        if self.logprobs:
            payload.update({"logprobs": True, "top_logprobs": 1})
        before = _metrics(self.base)
        start = time.time()
        status, body = _post(self.base, "/v1/chat/completions", payload, self.timeout)
        elapsed = time.time() - start
        after = _metrics(self.base)
        if status != 200:
            raise SystemExit(f"{tag}: HTTP {status}: {body}")
        delta = {k: int(after.get(k, 0) - before.get(k, 0)) for k in COUNTERS}
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        content = (choice.get("logprobs") or {}).get("content") or []
        res = {
            "tag": tag,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "local_hits": delta["prefix_cache_hits_total"],
            "ext_hits": delta["external_prefix_cache_hits_total"],
            "gen_tokens": len(choice.get("token_ids") or []),
            "finish_reason": choice.get("finish_reason"),
        }
        print("  " + json.dumps(res), flush=True)
        res.update(
            token_ids=choice.get("token_ids"),
            prompt_token_ids=body.get("prompt_token_ids"),
            text=(choice.get("message") or {}).get("content") or "",
            logprobs=[item.get("logprob") for item in content],
        )
        if res["cached_tokens"] is None:
            raise SystemExit(
                "no usage.prompt_tokens_details: relaunch the server with "
                "--enable-prompt-tokens-details"
            )
        if not res["token_ids"]:
            raise SystemExit("no choices[0].token_ids: nothing to compare")
        return res


def size_ledger(client, namespace, question, target_tokens):
    """Grow the ledger until the chat-wrapped prompt reaches target_tokens."""
    lines = 8
    for _ in range(6):
        messages = [{"role": "user", "content": ledger(namespace, lines) + question}]
        count = client.count(messages)
        if target_tokens <= count <= target_tokens * 1.1:
            break
        lines = max(4, math.ceil(lines * target_tokens * 1.05 / max(count, 1)))
    return lines, count


def diff(ref, other, atol, label):
    """Failure strings for `other` not reproducing `ref`."""
    failures = []
    if ref["prompt_token_ids"] != other["prompt_token_ids"]:
        return [f"{label}: the two arms tokenized to different prompts"]
    if ref["token_ids"] != other["token_ids"]:
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(ref["token_ids"], other["token_ids"]))
                if a != b
            ),
            min(len(ref["token_ids"]), len(other["token_ids"])),
        )
        drift = [
            abs(a - b)
            for a, b in zip(ref["logprobs"][:first], other["logprobs"][:first])
        ]
        # Drift before the split tells a lost state (large from token 0) from
        # a near-tie flipped by numerics (small, late).
        failures.append(
            f"{label}: token ids diverge from the uncached run at generated "
            f"token {first} of {len(ref['token_ids'])}"
            + (f", max |logprob diff| before it {max(drift):.3g}" if drift else "")
            + f" (uncached={ref['text']!r} cached={other['text']!r})"
        )
    elif atol is not None and ref["logprobs"] and other["logprobs"]:
        worst = max(
            abs(a - b) for a, b in zip(ref["logprobs"], other["logprobs"], strict=True)
        )
        print(f"  {label}: max |logprob diff| = {worst:.3g}", flush=True)
        if worst > atol:
            failures.append(
                f"{label}: same tokens, but logprobs drift by {worst:.3g} "
                f"(> {atol}) from the uncached run"
            )
    return failures


QUESTIONS = (
    "\n\nQuestion: give the manifest code of this archive, then describe in two "
    "sentences what the ledger records.",
    "Thanks. Now list three kinds of goods that appear in the entries, and say "
    "who signs the barge manifest.",
    "Last one: which entry number was the final one in the ledger, and what did "
    "it say?",
)


def run_multiturn(client, args):
    failures = []
    block = args.hit_block_size
    namespace = f"{args.salt}-mt"
    # Turn 1 stops short of a block boundary, so the next turn's prompt (this
    # one + its answer + a new question) crosses it: turn 3 can then only hit
    # that block if the state was checkpointed mid-conversation, while turn 2
    # was being prefilled on top of a hit of its own.
    target = args.prefix_blocks * block + (block * 3) // 4
    lines, count = size_ledger(client, namespace, QUESTIONS[0], target)
    print(f"  turn-1 prompt: {count} tokens ({lines} ledger lines)", flush=True)
    messages = [{"role": "user", "content": ledger(namespace, lines) + QUESTIONS[0]}]
    previous_prompt = 0
    for turn in range(1, len(QUESTIONS) + 1):
        print(f"--- multiturn turn {turn}", flush=True)
        warm = client.ask(f"turn{turn}-conversation", messages, "conversation")
        ref = client.ask(f"turn{turn}-uncached", messages, f"uncached-{turn}")
        if ref["cached_tokens"] or ref["local_hits"] or ref["ext_hits"]:
            failures.append(
                f"turn {turn}: the uncached reference hit "
                f"(cached_tokens={ref['cached_tokens']}): cache_salt did not "
                "isolate it, so there is no ground truth"
            )
        # Every whole block of the previous turn's prompt must come back.
        want = ((previous_prompt - 1) // block) * block if previous_prompt else 0
        if turn == 1:
            if warm["cached_tokens"]:
                failures.append("turn 1: the first turn of a fresh salt hit the cache")
        elif warm["cached_tokens"] < want:
            failures.append(
                f"turn {turn}: cached_tokens={warm['cached_tokens']}, but the "
                f"previous turn's {previous_prompt}-token prompt holds {want} "
                "tokens of whole blocks"
            )
        previous_prompt = warm["prompt_tokens"]
        failures += diff(ref, warm, args.logprob_atol, f"turn {turn}")
        if turn < len(QUESTIONS):
            messages = messages + [
                {"role": "assistant", "content": warm["text"]},
                {"role": "user", "content": QUESTIONS[turn]},
            ]
    return failures


def run_eviction(client, args):
    failures = []
    target = args.prefix_blocks * args.hit_block_size + args.hit_block_size // 4
    namespace = f"{args.salt}-victim"
    lines, count = size_ledger(client, namespace, QUESTIONS[0], target)
    victim = [{"role": "user", "content": ledger(namespace, lines) + QUESTIONS[0]}]
    print(f"--- eviction: victim prompt, {count} tokens", flush=True)
    cold = client.ask("victim-cold", victim)
    if cold["cached_tokens"] or cold["ext_hits"]:
        failures.append("eviction: the victim's cold pass already hit")
    # Let the store jobs for the victim land before anything can evict it.
    time.sleep(args.settle_s)

    # Before the flood the victim must still be an HBM hit. This is what makes
    # the post-flood local miss evidence of an eviction, and it shows the pool
    # is not so small that nothing survives between two requests.
    whole = ((count - 1) // args.hit_block_size) * args.hit_block_size
    print("--- eviction: victim again, before the flood (HBM)", flush=True)
    hbm = client.ask("victim-hbm", victim)
    if hbm["ext_hits"] or hbm["local_hits"] != whole or hbm["cached_tokens"] != whole:
        failures.append(
            f"eviction: before the flood the victim should be a whole-prefix "
            f"HBM hit of {whole} tokens, got local_hits={hbm['local_hits']} "
            f"ext_hits={hbm['ext_hits']} cached_tokens={hbm['cached_tokens']}"
        )
    failures += diff(cold, hbm, None, "eviction hbm control")

    fillers = math.ceil(args.hbm_turnovers * args.hbm_tokens / count)
    print(
        f"--- eviction: {fillers} filler prompts of ~{count} tokens to turn a "
        f"{args.hbm_tokens}-token HBM pool over {args.hbm_turnovers}x",
        flush=True,
    )
    for i in range(fillers):
        text = ledger(f"{args.salt}-filler{i}", lines) + QUESTIONS[0]
        filler = client.ask(f"filler{i}", [{"role": "user", "content": text}], None, 1)
        if filler["cached_tokens"]:
            failures.append(f"eviction: filler {i} hit the cache; it is not fresh")

    print("--- eviction: victim again, after the flood (host tier)", flush=True)
    recall = client.ask("victim-recall", victim)
    if recall["ext_hits"] <= 0:
        failures.append(
            f"eviction: no external hits on the re-serve (local_hits="
            f"{recall['local_hits']}, cached_tokens={recall['cached_tokens']}). "
            "Either the victim never left HBM (raise --hbm-turnovers or lower "
            "--num-gpu-blocks-override) or the store could not resolve it; "
            "nothing was loaded from the host, so nothing was tested"
        )
    if recall["local_hits"] >= whole:
        failures.append(
            f"eviction: the whole {whole}-token prefix was still a local hit, "
            "so HBM never evicted the victim"
        )
    if recall["cached_tokens"] != recall["ext_hits"] + recall["local_hits"]:
        failures.append(
            f"eviction: cached_tokens={recall['cached_tokens']} is not "
            f"local_hits + ext_hits ({recall['local_hits']} + "
            f"{recall['ext_hits']})"
        )
    failures += diff(cold, recall, None, "eviction recall")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=("multiturn", "eviction"))
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--salt",
        default="tiers-1790000000",
        help="prompt namespace. Fixed, so a run is reproducible: the rig boots "
        "its own engine and every cache starts empty. Against a server that "
        "outlives the run pass a fresh one (--salt $(date +%%s)), or the cold "
        "passes hit",
    )
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--hit-block-size",
        type=int,
        default=256,
        help="prefix-cache hit granularity in tokens",
    )
    ap.add_argument(
        "--prefix-blocks",
        type=int,
        default=4,
        help="whole hit blocks in the first-turn / victim prompt",
    )
    ap.add_argument(
        "--logprob-atol",
        type=float,
        default=1e-3,
        help="multiturn: tolerance on generated-token logprobs, the value "
        "smoke_prefix_cache_e2e_divergence.py uses",
    )
    ap.add_argument("--no-logprobs", action="store_true")
    ap.add_argument(
        "--hbm-tokens",
        type=int,
        default=0,
        help="eviction: HBM KV capacity in tokens (num_gpu_blocks x block size)",
    )
    ap.add_argument("--hbm-turnovers", type=float, default=2.0)
    ap.add_argument("--settle-s", type=float, default=10.0)
    args = ap.parse_args()
    if args.no_logprobs:
        args.logprob_atol = None
    if args.stage == "eviction" and args.hbm_tokens <= 0:
        raise SystemExit("eviction needs --hbm-tokens")

    client = Client(
        args.base,
        args.model,
        args.timeout,
        args.max_tokens,
        logprobs=args.stage == "multiturn" and not args.no_logprobs,
    )
    print(f"=== {args.stage}, salt={args.salt} ===", flush=True)
    failures = (run_multiturn if args.stage == "multiturn" else run_eviction)(
        client, args
    )
    tag = f"PREFIX_{args.stage.upper()}"
    if failures:
        print(f"{tag}_FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print(f"{tag}_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
