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
"""KV offload store/load round trip: does a recalled block replay the same
bytes?

The rig (run_dsv4_offload_correctness.sh) runs max_num_seqs=1 and pins every
request to a single data-parallel rank (--dp-rank), so all three passes are
served by the same engine with identical single-request batch geometry and a
cold/warm difference is the cache and nothing else. The pin is not optional at
DP > 1: each rank owns its own Raiden store and its own prefix cache.

Per prompt length L, three passes over the SAME prompt:

  cold   fresh salt, nothing cached anywhere. Reference output.
         Gate: cached_tokens == 0 and no hits.
  hbm    immediate repeat -> served from the HBM prefix cache.
         Gate: a whole-prefix LOCAL hit (local_hits == cached_tokens ==
         floor((L-1)/1024)*1024), ext_hits == 0, output identical to cold.
  store  POST /reset_prefix_cache empties HBM while the Raiden store keeps
         its entries (?reset_external is deliberately omitted, so the
         connector's reset_cache() is never reached), so the repeat has to be
         a real host->device load of the offloaded column.
         Gate: 0 local hits, an EXTERNAL hit of that same size (ext_hits ==
         cached_tokens == the hbm arm's), output identical to cold.
"""

import argparse
import json
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
SHOW = (
    "tag",
    "status",
    "elapsed_s",
    "prompt_tokens",
    "local_hits",
    "local_queries",
    "ext_hits",
    "ext_queries",
    "cached_tokens",
    "gen_tokens",
    "finish_reason",
)

# Hits land only on multiples of the lcm of the model's KV cache group block
# sizes, and the offload chunk is the largest of them; 1024 is that value for
# the model this rig ships against. See the module docstring on L, and pass
# --hit-block-size for a model whose granularity differs.
HIT_BLOCK_SIZE = 1024
# How many whole hit blocks each triple must recall. The only real
# requirement is "longer than one block", so this is not a knob: it is a
# short ladder, because a one-block recall and a four-block recall exercise
# different amounts of the offloaded column. Entries that would not fit the
# server's max_model_len are dropped at run time.
PREFIX_BLOCKS = (1, 2, 4)

# Prompt namespace, and with it the hash chain every cold arm needs to be a
# miss in. Fixed rather than per-run: the rig launches and tears down its own
# engine, so both the HBM cache and the Raiden store start each run empty and
# a constant salt is reproducible without being stale. It stops being safe the
# moment the smoke is pointed at a server that outlives it -- the second run's
# cold arm then hits and is reported as a failure, correctly. Pass --salt to
# get a different namespace.
DEFAULT_SALT = "offload-1788410251"


def _post(base, path, payload, timeout, headers=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        return e.code, {"body": e.read().decode(errors="replace")[:400]}


def _metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo:
            out[mo.group(1)] = out.get(mo.group(1), 0.0) + float(mo.group(2))
    return out


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


def manifest_code(namespace: str) -> str:
    """Distinct per namespace, so a cross-length prefix collision shows up in
    the generated text and not only in the hit counters."""
    return str(random.Random(namespace).randrange(100000, 1000000))


def build_prompt(namespace: str, code: str, lines: int) -> str:
    """A coherent ledger, not random ids. Three properties are load-bearing:
    the namespace is on line 1, so each length owns its hash chain from block
    0 onward; the answer the closing question asks for is in the opening
    lines, i.e. inside the prefix that gets recalled rather than in the tail
    that is always recomputed; and the question asks for prose rather than
    for the bare code, because a one-token answer would let two arms agree on
    a single argmax and call that a round trip."""
    head = [
        f"Depot archive {namespace}: quarterly shipment ledger.",
        f"The manifest code for archive {namespace} is {code}.",
        "The entries below are filed one per shipment, in order.",
    ]
    body = [f"Entry {i:04d}: {FILLER[i % len(FILLER)]}." for i in range(lines)]
    tail = [
        "",
        f"Question about archive {namespace}: give its manifest code, then "
        "describe in two or three sentences what this ledger records and how "
        "its entries are organised.",
    ]
    return "\n".join(head + body + tail)


def size_prompt(client, namespace, code, aim, hit_block):
    """Grow the ledger until the server tokenizes it into the band that
    recalls the wanted number of whole blocks.
    """
    # The band `aim` names: from the smallest L whose whole-prefix hit is
    # aim's up to the largest. A prompt of exactly whole blocks cannot claim
    # its own last block, hence the +1.
    lo = ((aim - 1) // hit_block) * hit_block + 1
    hi = lo - 1 + hit_block
    # Whole entries quantize the count, so accept a little inside either edge
    # rather than oscillating around an exact target that is not required.
    margin = hit_block // 16
    lines = 8
    text = build_prompt(namespace, code, lines)
    count = client.tokenize(text)
    for _ in range(6):
        if lo + margin <= count <= hi - margin:
            break
        # Solving by ratio: the ledger is dominated by uniform entries, so the
        # token count is very close to linear in `lines`.
        lines = max(4, round(lines * aim / max(count, 1)))
        text = build_prompt(namespace, code, lines)
        count = client.tokenize(text)
    return text, count


class Client:
    def __init__(self, base, model, timeout, max_tokens, dp_rank=None):
        self.base, self.model = base, model
        self.timeout, self.max_tokens = timeout, max_tokens
        # Every request carries the same data-parallel rank, which is what
        # makes this smoke meaningful at DP > 1.
        self.dp_rank = dp_rank
        self.headers = {} if dp_rank is None else {"X-data-parallel-rank": str(dp_rank)}

    def _chat_wrapper(self, text):
        """The chat-template arguments every request shares. They must be
        identical between /tokenize and /v1/chat/completions or the measured
        L is not the length the engine sees."""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "add_generation_prompt": True,
            # The template emits BOS itself; asking for it again would
            # duplicate it and shift every hash in the chain.
            "add_special_tokens": False,
            # Thinking mode would open the generation with a <think> block the
            # model must close before it answers, spending most of the budget
            # on reasoning rather than on the answer. Every token is compared
            # the same way either way, so it is coverage traded for latency.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _tokenize(self, text):
        """Raw /tokenize response for the chat-wrapped prompt. The endpoint is
        required, not optional: it is what makes the server the authority on
        L. Without it the ledger could only be sized from a client-side
        estimate, which may land in a different 1024-token band than the run
        reports and test something other than what it prints. It is sent the
        `messages` form rather than `prompt` for the same reason: the
        template's own wrapper is part of the sequence the engine hashes and
        caches (four tokens for this rig's default model --
        <|begin_of_sentence|><|User|>...<|Assistant|></think>), so a count
        taken without it measures a prompt nobody sends. Tokenizing publishes
        nothing into any cache."""
        status, body = _post(
            self.base, "/tokenize", self._chat_wrapper(text), 120, self.headers
        )
        if status != 200:
            raise SystemExit(
                f"the server did not answer /tokenize (HTTP {status}): the "
                "prompt length every gate in this smoke is measured against "
                "would have to be guessed, and a guess that crosses a block "
                "boundary runs a different test than it reports"
            )
        return body

    def tokenize(self, text):
        """Server-side token count."""
        return self._tokenize(text)["count"]

    def max_model_len(self):
        """The engine's context limit, or None if this server does not
        report it alongside a token count."""
        return self._tokenize("probe").get("max_model_len")

    def _generate(self, prompt, max_tokens):
        # No ignore_eos: the model is asked a question it can finish, and
        # padding the budget past its answer would only append a run of EOS
        # that every arm reproduces whatever the store returned.
        payload = dict(
            self._chat_wrapper(prompt),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=0,
            return_token_ids=True,
        )
        return _post(
            self.base, "/v1/chat/completions", payload, self.timeout, self.headers
        )

    def ask(self, tag, prompt):
        before = _metrics(self.base)
        t = time.time()
        status, body = self._generate(prompt, self.max_tokens)
        elapsed = time.time() - t
        after = _metrics(self.base)
        delta = {k: int(after.get(k, 0) - before.get(k, 0)) for k in COUNTERS}
        res = {
            "tag": tag,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "local_hits": delta["prefix_cache_hits_total"],
            "local_queries": delta["prefix_cache_queries_total"],
            "ext_hits": delta["external_prefix_cache_hits_total"],
            "ext_queries": delta["external_prefix_cache_queries_total"],
            "cached_tokens": None,
            "token_ids": None,
            "prompt_tokens": None,
            "prompt_token_ids": None,
            "gen_tokens": None,
            "finish_reason": None,
        }
        if status != 200:
            res["error"] = body
            return res
        choice = body["choices"][0]
        token_ids = choice.get("token_ids")
        details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
        res.update(
            {
                "token_ids": token_ids,
                "text": (choice.get("message") or {}).get("content"),
                "finish_reason": choice.get("finish_reason"),
                "gen_tokens": len(token_ids) if token_ids is not None else None,
                "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
                # The chat path hangs prompt_token_ids off the response, not the
                # choice -- reading it from the choice would silently return None
                # and drop every gate below to the coarser prompt_tokens count.
                "prompt_token_ids": body.get("prompt_token_ids"),
                "cached_tokens": details.get("cached_tokens"),
            }
        )
        return res

    def reset_prefix_cache(self, attempts=10):
        for _ in range(attempts):
            # Not affected by the pin: reset_prefix_cache is a utility
            # call the DP client broadcasts to every engine, so this clears
            # all ranks. That is a superset of what the pinned rank needs and
            # leaves the store arm's floor intact.
            status, body = _post(
                self.base, "/reset_prefix_cache", {}, 120, self.headers
            )
            if status == 404:
                raise SystemExit(
                    "POST /reset_prefix_cache is not mounted: relaunch the "
                    "server with VLLM_SERVER_DEV_MODE=1"
                )
            if status == 200 and body.get("success"):
                return True
            # Fails while blocks are still held, e.g. by an in-flight offload
            # transfer. Retry rather than reading it as a verdict.
            time.sleep(2)
        return False


def prompt_len(res):
    """The real prompt length as the engine tokenized it. prompt_token_ids
    comes back because the request sets return_token_ids; usage.prompt_tokens
    is the fallback for a server that does not populate it."""
    ids = res.get("prompt_token_ids")
    return len(ids) if ids is not None else res.get("prompt_tokens")


def same_prompt(ref, other):
    """Did the two arms send the engine the same token sequence? With text
    prompts this is no longer free, and a tokenization difference between
    arms would look exactly like a recall bug."""
    a, b = ref.get("prompt_token_ids"), other.get("prompt_token_ids")
    if a is not None and b is not None:
        return a == b
    return prompt_len(ref) == prompt_len(other)


def compare(ref, other):
    return {"tokens_equal": ref.get("token_ids") == other.get("token_ids")}


def run_length(client, salt, want_blocks, settle_s, hit_block):
    """One cold/hbm/store triple over a prompt that recalls `want_blocks`
    whole hit blocks. Returns a list of failure strings."""
    failures = []
    namespace = f"{salt}-b{want_blocks}"
    code = manifest_code(namespace)

    aim = want_blocks * hit_block + hit_block // 8
    prompt, sized = size_prompt(client, namespace, code, aim, hit_block)
    print(f"  sized: {sized} tokens (aim {aim}), manifest code {code}", flush=True)

    cold = client.ask("cold", prompt)
    print("  " + json.dumps({k: cold.get(k) for k in SHOW}), flush=True)
    print(f"  generated={cold.get('text')!r}", flush=True)
    if cold["status"] != 200:
        return [f"{want_blocks}-block prompt: cold pass failed: {cold.get('error')}"]
    if cold["cached_tokens"] is None:
        # Every quantity gate below reads this field, so a server without
        # --enable-prompt-tokens-details cannot be measured, only guessed at.
        raise SystemExit(
            "the engine did not report usage.prompt_tokens_details: relaunch "
            "the server with --enable-prompt-tokens-details, or no arm can "
            "prove how much of the prompt was actually recalled"
        )
    if cold["token_ids"] is None:
        # compare() reads token_ids on both sides, so a server that does not
        # honour return_token_ids would report None == None as equal and
        # every arm would pass without comparing anything.
        raise SystemExit(
            "the engine did not report choices[0].token_ids: without the "
            "generated ids the arms cannot be compared, and the equality "
            "gates would pass vacuously"
        )
    length = prompt_len(cold)
    if length is None:
        raise SystemExit(
            "the engine reported neither choices[0].prompt_token_ids nor "
            "usage.prompt_tokens, so the prompt length the gates below need "
            "is unknown and every hit size would be measured against a guess"
        )
    # A whole-prefix hit: block-aligned, and capped at L-1 because a prompt of
    # exactly whole blocks can never claim its own last block. Computed from
    # the length the engine actually saw, never from the sizing aim.
    expected_hit = ((length - 1) // hit_block) * hit_block
    if expected_hit // hit_block != want_blocks:
        return failures + [
            f"{want_blocks}-block prompt: the ledger tokenized to {length} "
            f"tokens, a whole-prefix hit of {expected_hit} tokens = "
            f"{expected_hit // hit_block} x {hit_block}, where {want_blocks} "
            f"x {hit_block} was asked for. The sizing missed its band, so "
            "this triple would test a different prefix length than the run "
            "reports"
        ]
    if cold["local_hits"] > 0 or cold["ext_hits"] > 0 or cold["cached_tokens"]:
        failures.append(
            f"L={length}: the cold pass already hit "
            f"(local={cold['local_hits']} ext={cold['ext_hits']} "
            f"cached_tokens={cold['cached_tokens']}): the prompt is not "
            "fresh, so both warm arms are vacuous"
        )

    # Let the speculative store jobs for this prompt actually land before the
    # HBM cache is pulled out from under them.
    time.sleep(settle_s)

    hbm = client.ask("hbm", prompt)
    floor = compare(cold, hbm)
    print("  " + json.dumps({**{k: hbm.get(k) for k in SHOW}, **floor}), flush=True)
    if hbm["status"] != 200:
        return failures + [f"L={length}: hbm pass failed: {hbm.get('error')}"]
    if not same_prompt(cold, hbm):
        return failures + [
            f"L={length}: the HBM arm tokenized to {prompt_len(hbm)} tokens "
            f"against cold's {length}. The three arms must send the engine "
            "an identical token sequence or every comparison below is "
            "between two different requests, not two recall paths"
        ]
    if hbm["ext_hits"] or hbm["cached_tokens"] != hbm["local_hits"]:
        # cached_tokens is local + external for this one request, so it
        # disagreeing with the local counter is the same finding arriving by
        # the other route -- and catches it even if the metric delta misses.
        failures.append(
            f"L={length}: the HBM control arm was itself served by the store "
            f"(ext_hits={hbm['ext_hits']}, cached_tokens="
            f"{hbm['cached_tokens']} vs local_hits={hbm['local_hits']}). It "
            "is the offload-free reference: once the store feeds it, it is a "
            "second store test and gating the store arm against it is "
            "circular"
        )
    elif hbm["local_hits"] <= 0:
        failures.append(
            f"L={length}: the HBM arm did not hit the local prefix cache, so "
            "it gates nothing and leaves the store arm without a floor"
        )
    elif hbm["local_hits"] != expected_hit or hbm["cached_tokens"] != expected_hit:
        failures.append(
            f"L={length}: the HBM arm recalled cached_tokens="
            f"{hbm['cached_tokens']} (local_hits={hbm['local_hits']}) where a "
            f"whole-prefix hit is {expected_hit} tokens "
            f"(floor((L-1)/{hit_block})*{hit_block}). Either the HBM prefix "
            f"cache is dropping part of the prefix, or this config's hit "
            f"granularity is not {hit_block} -- pass --hit-block-size to "
            "match it. Left as is, the store arm inherits a short floor"
        )
    if not floor["tokens_equal"]:
        failures.append(
            f"L={length}: HBM recall changed the output "
            f"(cold={cold['text']!r} hbm={hbm['text']!r})"
        )

    if not client.reset_prefix_cache():
        return failures + [f"L={length}: /reset_prefix_cache never succeeded"]

    store = client.ask("store", prompt)
    gate = compare(cold, store)
    print("  " + json.dumps({**{k: store.get(k) for k in SHOW}, **gate}), flush=True)
    if store["status"] != 200:
        return failures + [f"L={length}: store pass failed: {store.get('error')}"]
    if not same_prompt(cold, store):
        return failures + [
            f"L={length}: the store arm tokenized to {prompt_len(store)} "
            f"tokens against cold's {length}. The three arms must send the "
            "engine an identical token sequence or every comparison below is "
            "between two different requests, not two recall paths"
        ]
    if store["local_hits"]:
        failures.append(
            f"L={length}: {store['local_hits']} local hits after the reset — "
            "the HBM cache was not actually cleared, so this arm did not "
            "test the store"
        )
    want = hbm["cached_tokens"]
    if store["ext_hits"] <= 0:
        failures.append(
            f"L={length}: no external hits after the reset — the store either "
            "never admitted the blocks or cannot resolve them; nothing was "
            "loaded, so the equality below proves nothing"
        )
    elif store["ext_hits"] != want or store["cached_tokens"] != want:
        failures.append(
            f"L={length}: the store arm recalled cached_tokens="
            f"{store['cached_tokens']} (ext_hits={store['ext_hits']}) against "
            f"the HBM control arm's {want}: the store resolved only part of "
            "the offloaded column and the engine recomputed the remainder, so "
            "the equality below covers only the part that was loaded"
        )
    if not gate["tokens_equal"]:
        failures.append(
            f"L={length}: store recall changed the output "
            f"(cold={cold['text']!r} store={store['text']!r}) while the HBM "
            "control arm reproduced cold exactly, so this is the store's"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=192,
        help="generation budget, and in practice the generation "
        "length: the model answers this prompt at length "
        "and is cut off here (finish_reason=length). That is "
        "the "
        "point -- every token is one more argmax a bad "
        "recall can be caught by, and all three arms are cut "
        "at the same place",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="pin every request to this data-parallel rank via "
        "the X-data-parallel-rank header. Required at DP > 1: "
        "each rank owns a separate Raiden store and prefix "
        "cache, so an unpinned run can store on one rank and "
        "query another. Rank 0 by default, which is also valid "
        "at DP 1; pass a negative value to send no header and "
        "let the server load balance",
    )
    ap.add_argument(
        "--salt",
        default=DEFAULT_SALT,
        help="prompt namespace; every cold arm requires it to be "
        "unused by any live cache. Fixed by default, which "
        "is safe only because the rig gives each run its own "
        "engine -- against a server that outlives the run, "
        "pass a fresh one (--salt $(date +%%s))",
    )
    ap.add_argument(
        "--settle-s",
        type=float,
        default=10.0,
        help="wait after the cold pass for store jobs to land",
    )
    ap.add_argument(
        "--hit-block-size",
        type=int,
        default=HIT_BLOCK_SIZE,
        help="prefix-cache hit granularity in tokens; a whole-"
        "prefix hit at L is floor((L-1)/B)*B. Defaults to "
        "1024: the lcm of the group block sizes of the model "
        "this rig ships against, and its offload chunk",
    )
    args = ap.parse_args()

    salt = args.salt
    hit_block = args.hit_block_size
    if hit_block <= 0:
        raise SystemExit("--hit-block-size must be positive")
    dp_rank = args.dp_rank if args.dp_rank >= 0 else None
    client = Client(args.base, args.model, args.timeout, args.max_tokens, dp_rank)
    failures = []

    # A rung also has to leave room for the generation, or the engine rejects
    # the request and the triple never runs at all.
    ladder = list(PREFIX_BLOCKS)
    limit = client.max_model_len()
    if limit is not None:
        room = limit - args.max_tokens
        dropped = [b for b in ladder if b * hit_block + hit_block // 8 > room]
        ladder = [b for b in ladder if b not in dropped]
        if dropped:
            print(
                f"skipping the {dropped}-block rungs: max_model_len="
                f"{limit} leaves room for only {room} prompt tokens",
                flush=True,
            )
    if not ladder:
        raise SystemExit(
            f"no prompt of more than one {hit_block}-token block fits this "
            f"server's max_model_len ({limit}), so no arm could hit anything"
        )

    print(
        f"=== offload store/load correctness, salt={salt}, "
        f"hit block={hit_block}, prefixes={list(ladder)} blocks, "
        f"dp_rank={dp_rank if dp_rank is not None else 'unpinned'} ===",
        flush=True,
    )

    for blocks in ladder:
        print(
            f"\n--- {blocks}-block prompt, whole-prefix hit = "
            f"{blocks * hit_block} tokens ---",
            flush=True,
        )
        failures += run_length(client, salt, blocks, args.settle_s, hit_block)

    print()
    if failures:
        print("BASIC_OFFLOAD_CORRECTNESS_FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("BASIC_OFFLOAD_CORRECTNESS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
