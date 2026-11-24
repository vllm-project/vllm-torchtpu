# SPDX-License-Identifier: Apache-2.0
"""Decode prefix-cache token identity driver, hosted by the toy proxy.

Enabled with PROXY_Q38_DECPC_IDENTITY=1 (toy_proxy_server.py starts it and
holds /healthcheck at 503 until it finishes, so a benchmark client waits).

For C conversations of T turns (every turn's prompt is the previous prompt
plus a fixed token segment, never the model's output, as a trace replay
sends), each turn is served three ways through this proxy, one request at a
time, greedy:

  natural : the decode still holds the previous turn's prefix (turns >= 1);
  ref     : after POST /reset_prefix_cache on the decode, a full landing;
  ref2    : after another reset, the same prompt again with the cache empty
            (the A/A floor: the engine's own repeatability, addendum 3);
  hit     : the same prompt again, so the decode adopts the 2048-aligned
            prefix that ``ref2`` landed and pulls only the suffix.

Last, an in-job mutant: the decode serves a prompt B on the producer's
landing of a prompt A of the same length (same base, different last
segment). Its tokens must differ from B's own reference, or the token
comparison has no power and the verdict is void (``mutant_bites``).

``natural`` and ``hit`` must return exactly ``ref``'s token ids, and the
decode's speculative counters (drafts, draft tokens, accepted tokens) must
move by exactly ``ref``'s amounts. The decode's local prefix-cache hit
counter must move by 0 for ``ref`` and by more than 0 for the others, so a
run without decode hits cannot pass. The prefill must run without prefix
caching, so it publishes the same bytes for every request of a turn.

Output: one ``Q38_DECPC_IDENTITY {json}`` line per request and a final
``Q38_DECPC_IDENTITY_RESULT {json}`` line, on stdout.
"""

import asyncio
import json
import os
import random
import time

import httpx

_SPEC = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
_CACHE = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_hits",
)


def _ints(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


def plan(seed, bases, grows, vocab=(1000, 150000)):
    """Deterministic prompts: ``plan[c][t]`` is conversation c's turn t."""
    rng = random.Random(seed)
    lo, hi = vocab
    out = []
    for base in bases:
        prompt = [rng.randrange(lo, hi) for _ in range(base)]
        turns = [list(prompt)]
        for grow in grows:
            prompt = prompt + [rng.randrange(lo, hi) for _ in range(grow)]
            turns.append(list(prompt))
        out.append(turns)
    return out


def parse_metrics(text, names):
    """Sum every sample of each counter (labels ignored, ``_total`` or not)."""
    totals = dict.fromkeys(names, 0.0)
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        key = line.split("{", 1)[0].split(" ", 1)[0]
        if key.endswith("_total"):
            key = key[: -len("_total")]
        if key in totals:
            try:
                totals[key] += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return totals


def swap_bites(records):
    """The in-job mutant: the decode served prompt B on prompt A's landing.

    Its tokens must differ from B's own ``ref`` (``swap_ref``); if they do
    not, token comparisons on this harness have no power to see a wrong
    landing, and the identity verdict is void.
    """
    modes = {r["mode"]: r for r in records if r["conv"] == "swap"}
    swap, ref = modes.get("swap"), modes.get("swap_ref")
    if not swap or not ref or swap.get("error") or ref.get("error"):
        return False
    return swap["token_ids"] != ref["token_ids"]


def compare(records, noise=None):
    """Failures over the records of every (conversation, turn).

    ``ref2`` serves each ``ref`` prompt again with the decode cache empty, in
    the same configuration: the A/A floor. A ``hit``/``natural`` difference
    from ``ref`` is a FAIL when ``ref2`` equals ``ref`` on that quantity; when
    ``ref2`` differs too, the engine is not repeatable on that request and
    the difference is appended to ``noise`` (noise-limited), not a failure.
    """
    failures = []
    noise = [] if noise is None else noise
    by_key = {}
    for r in records:
        if r["conv"] == "swap":
            continue
        by_key.setdefault((r["conv"], r["turn"]), {})[r["mode"]] = r
    for (conv, turn), modes in sorted(by_key.items()):
        ref = modes.get("ref")
        if ref is None or ref.get("error"):
            failures.append((conv, turn, "ref", "missing or failed"))
            continue
        if ref["local_hit_tokens"] != 0:
            failures.append((conv, turn, "ref", "the reset did not clear the decode"))
        ref2 = modes.get("ref2")
        if ref2 is None or ref2.get("error"):
            failures.append((conv, turn, "ref2", "missing or failed"))
            ref2 = None
        elif ref2["local_hit_tokens"] != 0:
            failures.append((conv, turn, "ref2", "the reset did not clear the decode"))
        for mode in ("hit", "natural"):
            other = modes.get(mode)
            if other is None:
                if mode == "hit" or turn > 0:
                    failures.append((conv, turn, mode, "missing"))
                continue
            if other.get("error"):
                failures.append((conv, turn, mode, other["error"]))
                continue
            for key, what in (("token_ids", "token ids differ"),
                              ("spec", "speculative counters differ")):
                if other[key] == ref[key]:
                    continue
                if ref2 is not None and ref2[key] != ref[key]:
                    noise.append((conv, turn, mode, what + " (ref2 differs from ref too)"))
                else:
                    failures.append((conv, turn, mode, what))
            if other["local_hit_tokens"] <= 0:
                failures.append((conv, turn, mode, "no decode prefix hit"))
    return failures


def fail_fast(records):
    """Stop the protocol when a ``ref`` (a plain full landing, no decode hit)
    fails: every later comparison would fail the same way and only burn the
    slot (j-22e0b09a lost each request to a 300 s load timeout)."""
    return any(r["mode"] == "ref" and r.get("error") for r in records)


def aa_floor(records):
    """(conv, turn) where ``ref2`` differs from ``ref`` in tokens or counters:
    the engine's own same-configuration repeat rate, reported beside the verdict."""
    refs = {}
    for r in records:
        if r["mode"] in ("ref", "ref2") and not r.get("error"):
            refs.setdefault((r["conv"], r["turn"]), {})[r["mode"]] = r
    return sorted(
        k for k, m in refs.items()
        if "ref" in m and "ref2" in m
        and (m["ref"]["token_ids"] != m["ref2"]["token_ids"] or m["ref"]["spec"] != m["ref2"]["spec"])
    )


def verdict(records):
    """PASS: no failure, no noise-limited comparison, and the mutant bites.

    NOISE-LIMITED: no failure, but on some request ``ref2`` (the same prompt,
    same configuration, cache empty) already differed from ``ref``, so
    identity is unprovable there and the lever needs the accuracy gates.
    FAIL: any failure. VOID: the mutant did not bite.
    """
    noise = []
    failures = compare(records, noise)
    bites = swap_bites(records)
    if failures:
        status = "FAIL"
    elif not bites:
        status = "VOID"
    elif noise:
        status = "NOISE-LIMITED"
    else:
        status = "PASS"
    return {
        "requests": len(records),
        "turns": len({(r["conv"], r["turn"]) for r in records if r["conv"] != "swap"}),
        "failures": failures,
        "noise_limited": noise,
        "aa_floor_differs": aa_floor(records),
        "mutant_bites": bites,
        "status": status,
        "pass": status == "PASS",
    }


class Driver:
    def __init__(self, proxy_port, prefill_url, decode_url):
        self.proxy = f"http://127.0.0.1:{proxy_port}"
        self.prefill = prefill_url
        self.decode = decode_url
        self.seed = int(os.environ.get("PROXY_Q38_ID_SEED", "20261106"))
        self.bases = _ints("PROXY_Q38_ID_BASES", "40001,70003,100005,130007")
        self.grows = _ints("PROXY_Q38_ID_GROWS", "3001,5003")
        self.max_tokens = int(os.environ.get("PROXY_Q38_ID_MAX_TOKENS", "64"))
        self.timeout = float(os.environ.get("PROXY_Q38_ID_TIMEOUT_S", "900"))

    async def _wait_up(self, client, url):
        while True:
            try:
                if (await client.get(f"{url}/health", timeout=10)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(10)

    async def _counters(self, client):
        text = (await client.get(f"{self.decode}/metrics", timeout=30)).text
        return parse_metrics(text, _SPEC + _CACHE)

    async def _reset(self, client):
        for _ in range(120):
            r = await client.post(f"{self.decode}/reset_prefix_cache", timeout=60)
            if r.status_code == 200 and r.json().get("success"):
                return
            await asyncio.sleep(1)
        raise RuntimeError("decode /reset_prefix_cache never succeeded")

    async def _one(self, client, model, conv, turn, mode, prompt):
        before = await self._counters(client)
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": False,
        }
        t0 = time.time()
        record = {"conv": conv, "turn": turn, "mode": mode, "prompt_tokens": len(prompt)}
        try:
            r = await client.post(
                f"{self.proxy}/v1/completions", json=body, timeout=self.timeout
            )
            choice = r.json()["choices"][0]
            record["token_ids"] = choice.get("token_ids") or choice.get("text")
        except Exception as e:  # noqa: BLE001
            record["error"] = f"{type(e).__name__}: {e}"[:300]
        record["seconds"] = round(time.time() - t0, 3)
        # Counters update when the request's stats are recorded; let them settle.
        await asyncio.sleep(float(os.environ.get("PROXY_Q38_ID_SETTLE_S", "11")))
        after = await self._counters(client)
        delta = {k: after[k] - before[k] for k in after}
        record["spec"] = [delta[k] for k in _SPEC]
        record["local_hit_tokens"] = delta["vllm:prefix_cache_hits"]
        record["external_hit_tokens"] = delta["vllm:external_prefix_cache_hits"]
        print("Q38_DECPC_IDENTITY " + json.dumps(record, sort_keys=True), flush=True)
        return record

    async def _swap(self, client, model):
        """Serve prompt B on the producer's landing of prompt A (same length,
        same base, different last segment): a wrong-landing mutant."""
        turns = plan(self.seed, self.bases[:1], self.grows[:1])[0]
        a = turns[1]
        rng = random.Random(self.seed + 1)
        b = turns[0] + [rng.randrange(1000, 150000) for _ in range(len(a) - len(turns[0]))]
        await self._reset(client)
        ref = await self._one(client, model, "swap", 1, "swap_ref", b)
        await self._reset(client)
        record = {"conv": "swap", "turn": 1, "mode": "swap", "prompt_tokens": len(b)}
        try:
            p = await client.post(
                f"{self.prefill}/v1/completions",
                json={"model": model, "prompt": a, "max_tokens": 1, "stream": False},
                timeout=self.timeout,
            )
            params = p.json()["kv_transfer_params"]
            d = await client.post(
                f"{self.decode}/v1/completions",
                json={
                    "model": model, "prompt": b, "max_tokens": self.max_tokens,
                    "temperature": 0.0, "ignore_eos": True, "return_token_ids": True,
                    "stream": False, "kv_transfer_params": params,
                },
                timeout=self.timeout,
            )
            choice = d.json()["choices"][0]
            record["token_ids"] = choice.get("token_ids") or choice.get("text")
        except Exception as e:  # noqa: BLE001
            record["error"] = f"{type(e).__name__}: {e}"[:300]
        print("Q38_DECPC_IDENTITY " + json.dumps(record, sort_keys=True), flush=True)
        return [ref, record]

    async def run(self):
        records = []
        async with httpx.AsyncClient() as client:
            await self._wait_up(client, self.prefill)
            await self._wait_up(client, self.decode)
            model = (await client.get(f"{self.decode}/v1/models", timeout=30)).json()[
                "data"
            ][0]["id"]
            for conv, turns in enumerate(plan(self.seed, self.bases, self.grows)):
                for turn, prompt in enumerate(turns):
                    if turn > 0:
                        records.append(
                            await self._one(client, model, conv, turn, "natural", prompt)
                        )
                    await self._reset(client)
                    records.append(await self._one(client, model, conv, turn, "ref", prompt))
                    if fail_fast(records):
                        break
                    # The A/A floor: the same prompt, cache empty again.
                    await self._reset(client)
                    records.append(await self._one(client, model, conv, turn, "ref2", prompt))
                    records.append(await self._one(client, model, conv, turn, "hit", prompt))
                if fail_fast(records):
                    break
            else:
                records.extend(await self._swap(client, model))
        result = verdict(records)
        if fail_fast(records):
            result["aborted"] = "a full landing (ref) failed; no identity data possible"
        result["max_tokens"] = self.max_tokens
        print("Q38_DECPC_IDENTITY_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return result


async def start(app, proxy_port, state):
    """Run the driver once; never raises (the proxy keeps serving)."""
    try:
        prefill = app.state.prefill_clients[0]
        decode = app.state.decode_clients[0]
        await Driver(
            proxy_port,
            f"http://{prefill['host']}:{prefill['port']}",
            f"http://{decode['host']}:{decode['port']}",
        ).run()
    except Exception as e:  # noqa: BLE001
        print(
            "Q38_DECPC_IDENTITY_RESULT "
            + json.dumps({"pass": False, "error": f"{type(e).__name__}: {e}"[:500]}),
            flush=True,
        )
    finally:
        state["done"] = True
