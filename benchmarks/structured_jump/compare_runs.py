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
"""Compares outputs and metrics between B1 (spec baseline) and B2 (GaaD)."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-output", type=str, required=True)
    parser.add_argument("--b2-output", type=str, required=True)
    args = parser.parse_args()

    with open(args.b1_output, "r", encoding="utf-8") as f:
        b1_data = json.load(f)
    with open(args.b2_output, "r", encoding="utf-8") as f:
        b2_data = json.load(f)

    assert len(b1_data) == len(
        b2_data), f"Count mismatch: B1={len(b1_data)}, B2={len(b2_data)}"

    identical = 0
    diffs = []
    for i, (item1, item2) in enumerate(zip(b1_data, b2_data)):
        t1 = item1["text"]
        t2 = item2["text"]
        if t1 == t2:
            identical += 1
        else:
            diffs.append((i, t1, t2))

    print("=== Byte-Identical Correctness Gate ===")
    print(f"Total Requests: {len(b1_data)}")
    print(
        f"Byte-Identical Matches: {identical}/{len(b1_data)} ({identical/len(b1_data)*100:.1f}%)"
    )

    if diffs:
        print(f"WARNING: {len(diffs)} outputs differed between B1 and B2.")
        first_diff = diffs[0]
        print(f"First mismatch at index {first_diff[0]}:")
        print(f"B1: {first_diff[1]!r}")
        print(f"B2: {first_diff[2]!r}")
        sys.exit(1)
    else:
        print(
            "SUCCESS: Output texts are 100% byte-identical between B1 and B2!")


if __name__ == "__main__":
    main()
