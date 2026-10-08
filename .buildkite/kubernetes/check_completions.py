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

"""Send a few greedy completions through a P/D proxy and check the answers.

Usage: check_completions.py <completions URL> <model>

The prompts' greedy continuations are not in doubt, so a wrong or empty
answer means the decode engine did not get the prefill's KV, not that the
model phrased something differently. Exits non-zero if any case fails.
"""

import json
import sys
import urllib.request

CASES = [
    ("The capital of France is", "paris"),
    ("1, 2, 3, 4, 5, 6,", "7"),
    ("The chemical symbol for gold is", "au"),
]


def complete(url: str, model: str, prompt: str) -> str:
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0}
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["text"]


def main(url: str, model: str) -> int:
    failed = 0
    for prompt, expected in CASES:
        try:
            text = complete(url, model, prompt)
        except Exception as error:  # noqa: BLE001 - report every case
            print(f"FAIL {prompt!r}: {error}")
            failed += 1
            continue
        ok = expected in text.lower()
        print(f"{'PASS' if ok else 'FAIL'} {prompt!r} -> {text!r}")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <completions URL> <model>")
    sys.exit(main(sys.argv[1], sys.argv[2]))
