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
"""Expert placement parity: does a placed server answer with the same tokens?

TPU_EXPERT_PLACEMENT only moves experts between EP ranks. The fused EP kernel
relabels the selected experts to their slots after top-k and keeps the order
of the k combine terms, so a placed layer returns the same bits as the
contiguous one, and a server with the placement must generate exactly what a
server without it generates. The rig (run_dsv4_expert_placement_correctness.sh)
serves the model both ways; this smoke drives one server at a time and then
holds the two records against each other:

  collect   send every prompt --repeats times, greedy, one request at a time,
            pinned to --dp-rank, so both servers see identical single-request
            batch geometry. Records each completion's token ids and logprobs,
            and the server's speculative-decoding counters, as JSON.
  compare   per prompt, the first completion of each record must be the same
            tokens, and the two servers' speculative-decoding counters the
            same (the drafter's MoE layers are placed too). Logprobs are
            compared as well and reported, not gated. Repeats within one
            record must agree first: a server that disagrees with itself
            cannot be compared with another.

Exit status: 0 parity, 1 mismatch, 2 inconclusive (a server disagreed with
itself), 3 a request or a record failed.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request

SPEC_COUNTERS = (
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)


def _ledger(lines: int) -> str:
    """A long, deterministic prompt: several 1024-token prefill chunks, each
    of which spreads over most of the experts of every MoE layer."""
    rows = [
        f"Entry {i}: crate {i * 37 % 1009} moved from dock {i % 17} to "
        f"warehouse {(i * 7) % 23}; weight {(i * 53) % 997} kg."
        for i in range(1, lines + 1)
    ]
    return (
        "Below is a shipping ledger.\n"
        + "\n".join(rows)
        + f"\nQuestion: which warehouse did entry {lines // 3} go to, and how "
        "heavy was it? Answer in one sentence.\nAnswer:"
    )


# (id, prompt, max_tokens). Short and long prefills, code, arithmetic and
# several scripts, so the routed experts differ from prompt to prompt.
BUILTIN_PROMPTS = (
    ("capital", "The capital of France is", 64),
    (
        "train",
        "Solve step by step: a train travels 120 km in 1.5 hours, then 80 km "
        "in 1 hour. What is its average speed over the whole trip?\n",
        256,
    ),
    (
        "fib",
        "Write a Python function that returns the n-th Fibonacci number using "
        "memoization. Include a docstring and three asserts.\n",
        256,
    ),
    (
        "korean",
        "다음 문장을 영어로 번역하세요: 오늘은 날씨가 맑고 따뜻해서 공원에 "
        "산책하러 가기 좋은 날입니다.\n",
        128,
    ),
    ("chinese", "请用三句话解释什么是混合专家模型（Mixture of Experts）。\n", 192),
    (
        "french",
        "Explique la différence entre la photosynthèse et la respiration "
        "cellulaire en quatre phrases.\n",
        192,
    ),
    (
        "syllogism",
        "If all bloops are razzies and all razzies are lazzies, are all bloops "
        "definitely lazzies? Explain briefly.\n",
        160,
    ),
    (
        "json",
        "Return a JSON object describing three fictional planets, each with "
        "the fields name, radius_km and moons.\n",
        256,
    ),
    ("arith", "Compute 17*23, 45*67, 89*12 and 34*56. Show each result.\n", 128),
    (
        "story",
        "Once upon a time, in a city built on the backs of giant turtles,",
        256,
    ),
    ("ledger_80", _ledger(80), 96),
    ("ledger_300", _ledger(300), 96),
)


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
        return e.code, {"body": e.read().decode(errors="replace")[:600]}


def _spec_counters(base):
    """The speculative-decoding counters, summed over every engine's labels."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
            text = r.read().decode()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"WARNING: could not read /metrics: {e}")
        return {}
    out = {}
    for line in text.splitlines():
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo and mo.group(1) in SPEC_COUNTERS:
            out[mo.group(1)] = out.get(mo.group(1), 0.0) + float(mo.group(2))
    return out


def _load_prompts(path):
    if not path:
        return list(BUILTIN_PROMPTS)
    prompts = []
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                prompts.append(
                    (
                        str(row.get("id", n)),
                        row["prompt"],
                        int(row.get("max_tokens", 256)),
                    )
                )
    return prompts


def collect(args):
    headers = {} if args.dp_rank < 0 else {"X-data-parallel-rank": str(args.dp_rank)}
    want_logprobs = not args.no_logprobs
    results = []
    started = time.time()
    for pid, prompt, max_tokens in _load_prompts(args.prompts_file):
        if args.max_tokens:
            max_tokens = min(max_tokens, args.max_tokens)
        repeats = []
        for r in range(args.repeats):
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 0,
            }
            if want_logprobs:
                payload.update(logprobs=1, return_tokens_as_token_ids=True)
            t0 = time.time()
            status, body = _post(
                args.base, "/v1/completions", payload, args.timeout, headers
            )
            if status == 400 and want_logprobs and not results and not repeats:
                print(f"server refused logprobs ({body}); comparing text only")
                want_logprobs = False
                del payload["logprobs"], payload["return_tokens_as_token_ids"]
                status, body = _post(
                    args.base, "/v1/completions", payload, args.timeout, headers
                )
            if status != 200:
                print(f"FAIL {pid} #{r}: HTTP {status}: {body}")
                return 3
            choice = body["choices"][0]
            logprobs = choice.get("logprobs") or {}
            repeats.append(
                {
                    "text": choice["text"],
                    "finish_reason": choice.get("finish_reason"),
                    "tokens": logprobs.get("tokens"),
                    "logprobs": logprobs.get("token_logprobs"),
                    "completion_tokens": (body.get("usage") or {}).get(
                        "completion_tokens"
                    ),
                }
            )
            print(
                f"{pid} #{r}: {repeats[-1]['completion_tokens']} tokens, "
                f"{time.time() - t0:.1f}s, finish={repeats[-1]['finish_reason']}"
            )
        results.append(
            {
                "id": pid,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "max_tokens": max_tokens,
                "repeats": repeats,
            }
        )
    record = {
        "meta": {
            "label": args.label,
            "base": args.base,
            "model": args.model,
            "dp_rank": args.dp_rank,
            "repeats": args.repeats,
            "logprobs": want_logprobs,
            "elapsed_s": round(time.time() - started, 1),
        },
        "results": results,
        "spec_counters": _spec_counters(args.base),
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: {len(results)} prompts, spec {record['spec_counters']}")
    return 0


def _seq(completion, by_tokens=True):
    """What two completions are held equal on: token ids where the server
    returned them, else the text."""
    return (
        completion["tokens"]
        if by_tokens and completion["tokens"]
        else completion["text"]
    )


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def compare(args):
    with open(args.a) as f:
        rec_a = json.load(f)
    with open(args.b) as f:
        rec_b = json.load(f)
    name_a = rec_a["meta"].get("label") or args.a
    name_b = rec_b["meta"].get("label") or args.b

    unstable = set()
    for name, rec in ((name_a, rec_a), (name_b, rec_b)):
        for row in rec["results"]:
            first = _seq(row["repeats"][0])
            for k, rep in enumerate(row["repeats"][1:], 1):
                if _seq(rep) != first:
                    unstable.add(row["id"])
                    print(
                        f"UNSTABLE {name} {row['id']}: repeat {k} differs from "
                        f"repeat 0 at token {_first_divergence(first, _seq(rep))}"
                    )

    rows_b = {row["id"]: row for row in rec_b["results"]}
    mismatched, logprob_drift = [], []
    for row_a in rec_a["results"]:
        row_b = rows_b.get(row_a["id"])
        if row_b is None or row_b["prompt_sha256"] != row_a["prompt_sha256"]:
            print(f"ERROR: {row_a['id']} is not the same prompt in both records")
            return 3
        xa, xb = row_a["repeats"][0], row_b["repeats"][0]
        by_tokens = bool(xa["tokens"] and xb["tokens"])
        sa, sb = _seq(xa, by_tokens), _seq(xb, by_tokens)
        if sa != sb:
            i = _first_divergence(sa, sb)
            mismatched.append(row_a["id"])
            print(
                f"MISMATCH {row_a['id']}: first difference at token {i} of "
                f"{len(sa)}/{len(sb)}: {sa[i : i + 3]} vs {sb[i : i + 3]}\n"
                f"  {name_a}: {xa['text'][:160]!r}\n  {name_b}: {xb['text'][:160]!r}"
            )
            continue
        la, lb = xa.get("logprobs"), xb.get("logprobs")
        if la is not None and lb is not None and la != lb:
            drift = max(
                abs(p - q) for p, q in zip(la, lb) if p is not None and q is not None
            )
            logprob_drift.append((row_a["id"], drift))

    spec_a, spec_b = rec_a.get("spec_counters", {}), rec_b.get("spec_counters", {})
    spec_diff = {
        k: (spec_a.get(k), spec_b.get(k))
        for k in SPEC_COUNTERS
        if spec_a.get(k) != spec_b.get(k)
    }

    n = len(rec_a["results"])
    print(f"--- {name_a} vs {name_b}: {n} prompts")
    print(f"token mismatches: {len(mismatched)} {mismatched}")
    print(f"spec counters: {spec_a} vs {spec_b}")
    for pid, drift in logprob_drift:
        print(f"NOTE {pid}: same tokens, logprobs differ by up to {drift:.3g}")
    if not logprob_drift and rec_a["meta"].get("logprobs"):
        print("logprobs identical on every prompt")

    real = [pid for pid in mismatched if pid not in unstable]
    if real:
        print(f"VERDICT: MISMATCH on {real}")
        return 1
    if unstable:
        print(
            "VERDICT: INCONCLUSIVE, a server disagreed with itself on "
            f"{sorted(unstable)}"
        )
        return 2
    if spec_diff:
        print(f"VERDICT: MISMATCH in speculative-decoding counters {spec_diff}")
        return 1
    print("VERDICT: PARITY")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--base", required=True, help="http://host:port of the server")
    c.add_argument("--model", required=True, help="the served model name")
    c.add_argument("--out", required=True)
    c.add_argument("--label", default="")
    c.add_argument(
        "--dp-rank", type=int, default=0, help="DP rank to pin to; -1 to not pin"
    )
    c.add_argument("--repeats", type=int, default=2)
    c.add_argument(
        "--max-tokens", type=int, default=0, help="cap on each prompt's own budget"
    )
    c.add_argument(
        "--prompts-file", default="", help="JSONL of {id, prompt, max_tokens}"
    )
    c.add_argument("--no-logprobs", action="store_true")
    c.add_argument("--timeout", type=float, default=1800)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "collect" and args.repeats < 1:
        ap.error("--repeats must be at least 1")
    return collect(args) if args.cmd == "collect" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
