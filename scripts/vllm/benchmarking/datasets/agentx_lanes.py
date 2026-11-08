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
"""Lanes of the InferenceX AgentX trace replay, for `vllm bench serve`.

`build` samples lanes from the public WekaTrace corpus the AgentX replay uses
(semianalysisai/cc-traces-weka-062126-256k): LANES traces, each entered at
START of its request sequence (main-agent and sub-agent requests in time
order), TURNS consecutive requests each. Requests are ordered turn-major, so a
client at concurrency LANES keeps about one request of every lane in flight.
Block hash ids are made global per trace and stored as runs.

`expand` writes the lanes as a `--dataset-name timed_trace` file. That dataset
expands each 64-token block hash into tokens seeded by Python's per-process
string hash, so turns of one lane share their prefix within a sweep and no
sweep's prompts are cached from an earlier one.
"""

import argparse
import json
import random

HASH_STRIDE = 10**7


def _runs(ids):
    out = []
    for h in ids:
        if out and out[-1][0] + out[-1][1] == h:
            out[-1][1] += 1
        else:
            out.append([h, 1])
    return out


def build(args):
    traces = []
    with open(args.traces) as f:
        for line in f:
            t = json.loads(line)
            if t["block_size"] != args.block_size:
                got, want = t["block_size"], args.block_size
                raise SystemExit(f"{t['id']}: block size {got}, expected {want}")
            reqs = []
            for r in t["requests"]:
                reqs.extend(r["requests"] if r.get("type") == "subagent" else [r])
            reqs.sort(key=lambda r: r["t"])
            traces.append((t["id"], reqs))
    eligible = [
        i for i, (_, reqs) in enumerate(traces) if len(reqs) >= args.min_requests
    ]
    lanes = random.Random(args.seed).sample(eligible, args.lanes)
    requests = []
    for k in range(args.turns):
        for i in lanes:
            reqs = traces[i][1]
            r = reqs[int(args.start * len(reqs)) + k]
            ids = [i * HASH_STRIDE + h for h in r["hash_ids"]]
            requests.append({"in": r["in"], "out": r["out"], "runs": _runs(ids)})
    doc = {
        "source": args.source,
        "block_size": args.block_size,
        "lanes": args.lanes,
        "turns_per_lane": args.turns,
        "start_fraction": args.start,
        "seed": args.seed,
        "trace_ids": [traces[i][0] for i in lanes],
        "requests": requests,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


def expand(args):
    with open(args.lanes) as f:
        doc = json.load(f)
    with open(args.out, "w") as f:
        for r in doc["requests"]:
            ids = [h for start, n in r["runs"] for h in range(start, start + n)]
            f.write(
                json.dumps(
                    {
                        "timestamp": 0,
                        "input_length": r["in"],
                        "output_length": r["out"],
                        "hash_ids": ids,
                    }
                )
                + "\n"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("traces", help="traces.jsonl of the WekaTrace corpus")
    b.add_argument("out")
    b.add_argument(
        "--source",
        required=True,
        help="dataset id and revision, recorded in the output",
    )
    b.add_argument("--lanes", type=int, default=64)
    b.add_argument("--turns", type=int, default=8)
    b.add_argument("--start", type=float, default=0.25)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--min-requests", type=int, default=40)
    b.add_argument("--block-size", type=int, default=64)
    b.set_defaults(fn=build)
    e = sub.add_parser("expand")
    e.add_argument("lanes")
    e.add_argument("out")
    e.set_defaults(fn=expand)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
