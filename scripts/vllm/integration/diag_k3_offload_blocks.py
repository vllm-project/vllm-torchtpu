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
"""Throwaway probe: how many offloaded blocks one K3 prompt takes.

The host pool has no registry, so when it is full it drops its oldest blocks.
Each phase seeds distinct prompts of CHUNKS hit blocks, more than the pool
holds, then recalls them newest-first with HBM emptied before each. A hit
adds nothing to the pool, so the run of full hits before the first miss is
the number of prompts the pool holds, and pool_blocks / held bounds the
blocks per prompt.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request

TAIL_TOKENS = 8


def scrape(base: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
        text = r.read().decode()
    m: dict[str, float] = {}
    for line in text.splitlines():
        mo = re.match(r"vllm:([a-z_0-9]+)(?:\{.*\})? ([0-9.e+-]+)$", line)
        if mo:
            m[mo.group(1)] = m.get(mo.group(1), 0.0) + float(mo.group(2))
    return m


def post(base: str, path: str, payload: dict | None, timeout: float):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        return e.code, {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--hit-block-size", type=int, required=True)
    p.add_argument("--pool-blocks", type=int, required=True)
    p.add_argument(
        "--phases",
        default="4:40:1000,2:80:3000",
        help="chunks:seeds:base_token per phase",
    )
    a = p.parse_args()

    def ask(ids: list[int]) -> float:
        before = scrape(a.base)
        status, _ = post(
            a.base,
            "/v1/completions",
            {
                "model": a.model,
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "add_special_tokens": False,
            },
            900,
        )
        if status != 200:
            raise SystemExit(f"HTTP {status}")
        after = scrape(a.base)
        key = "external_prefix_cache_hits_total"
        return after.get(key, 0) - before.get(key, 0)

    results = []
    for spec in a.phases.split(","):
        chunks, seeds, base_tok = map(int, spec.split(":"))
        full = chunks * a.hit_block_size
        plen = full + TAIL_TOKENS
        for i in range(seeds):
            ext = ask([base_tok + i] * plen)
            print(f"[seed c={chunks}] prompt{i} ext={ext:.0f}/{plen}", flush=True)
        time.sleep(5)
        held = 0
        for i in reversed(range(seeds)):
            status, _ = post(a.base, "/reset_prefix_cache", None, 60)
            if status != 200:
                raise SystemExit(f"/reset_prefix_cache: HTTP {status}")
            ext = ask([base_tok + i] * plen)
            print(f"[recall c={chunks}] prompt{i} ext={ext:.0f}/{plen}", flush=True)
            if ext < full:
                break
            held += 1
        lo = a.pool_blocks / (held + 1)
        hi = a.pool_blocks / held if held else float("inf")
        line = (
            f"chunks={chunks}: a {a.pool_blocks}-block pool held {held} of "
            f"{seeds} prompts -> {lo:.2f} < blocks per prompt <= {hi:.2f}"
        )
        if held == seeds:
            line += " (held them all: seed more)"
        results.append(line)
        print("RESULT " + line, flush=True)
    print("SUMMARY\n" + "\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
