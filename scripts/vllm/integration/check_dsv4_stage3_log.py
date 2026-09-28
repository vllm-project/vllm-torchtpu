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
"""Assert that a DSv4 prefill actually shipped its KV over Raiden Stage 3.

The correctness suites this runs beside cannot tell a working transfer from no
transfer at all. If the connector gives up, the decode replica prefills the
request itself and every answer is still right, so a green smoke says nothing
about the thing under test. This reads the producer's own log and checks four
properties that only hold when the transfer ran.

Every check is built on the JSON events the connector already emits, so none of
them depend on a log message's prose:

  raiden_stage3_dsv4_request_blocks_registered  the send plan was built
  raiden_stage3_sender_complete                 the bytes actually moved
  "Stage-3 skipping the send"                   the producer gave up

The interesting one is the window trim. Group dsv4.swa.g1 pages at 128 tokens
under a 128-token window, so an untrimmed block table for an N-token request
would list ceil(N/128) pages for that group and all but the last one or two
would point at the null block. vLLM frees out-of-window pages to block 0, so
shipping them hands Raiden one destination named many times and the rejection
kills the engine. A trimmed plan therefore reports strictly fewer pages than
ceil(N/128) for every group, and that inequality is the assertion: it needs no
page-size table, no pool-tag vocabulary and no per-model constant, so it does
not rot when a tag is renamed.

Measured against a real DP8 prefill log of 133 requests, an 1845-token request
registers group_pages [2, 2, 2, 2, 5] where untrimmed would have been 15.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# The smallest page size any DSv4 sliding-window group uses. A group paging at
# this size is the one that would blow up first without the trim, so it sets
# the untrimmed upper bound the check compares against.
MIN_PAGE_TOKENS = 128

# Below this many tokens a request need not have any out-of-window pages, so
# the trim has nothing to remove and the inequality is not expected to hold.
MIN_TRIM_TOKENS = 512

REGISTERED_EVENTS = (
    "raiden_stage3_dsv4_request_blocks_registered",
    # The non-DSv4 producer path emits this name. Accepted so the check still
    # reports something useful if a DSv4 run is ever routed through it, rather
    # than silently seeing zero sends and blaming the transfer.
    "raiden_stage3_request_blocks_registered",
)
COMPLETE_EVENT = "raiden_stage3_sender_complete"
SKIP_MARKER = "Stage-3 skipping the send"

_DECODER = json.JSONDecoder()


def _objects_in(line: str):
    """Yield every JSON object embedded in a log line.

    The connector logs events as a bare json.dumps appended to a formatted log
    line, and the registered event nests declared_bytes, so a brace-matching
    regex clips it. raw_decode consumes a whole object from a starting brace and
    ignores whatever follows, which handles both the nesting and the log prefix.
    """
    start = line.find("{")
    while start >= 0:
        try:
            obj, end = _DECODER.raw_decode(line, start)
        except json.JSONDecodeError:
            start = line.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            yield obj
        start = line.find("{", end)


def parse_events(path: str) -> tuple[list[dict], int, int]:
    """Return (registered events, sender_complete count, skipped send count)."""
    registered: list[dict] = []
    completed = 0
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if SKIP_MARKER in line:
                skipped += 1
            if "raiden_stage3" not in line:
                continue
            for event in _objects_in(line):
                name = event.get("event")
                if name in REGISTERED_EVENTS:
                    registered.append(event)
                elif name == COMPLETE_EVENT:
                    completed += 1
    return registered, completed, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="producer (prefill) server log")
    ap.add_argument(
        "--min-sends",
        type=int,
        default=1,
        help="fail if fewer than this many sends were registered",
    )
    ap.add_argument(
        "--expect-parallelism",
        type=int,
        default=1,
        help="DSv4 runs DP attention, so this is 1",
    )
    args = ap.parse_args()

    try:
        registered, completed, skipped = parse_events(args.log)
    except OSError as exc:
        print(f"STAGE3_CHECK unreadable log {args.log}: {exc}")
        return 1

    print(f"STAGE3_REGISTERED_SENDS {len(registered)}")
    print(f"STAGE3_COMPLETED_SENDS {completed}")
    print(f"STAGE3_SKIPPED_SENDS {skipped}")

    rc = 0

    if len(registered) < args.min_sends:
        print(
            f"STAGE3_FAIL registered {len(registered)} sends, wanted at least "
            f"{args.min_sends}. The producer never shipped anything, so the "
            f"correctness suites passed on decode-side prefill and tested "
            f"nothing."
        )
        rc = 1

    if skipped:
        print(
            f"STAGE3_FAIL the producer skipped {skipped} send(s). Each one "
            f"means decode prefilled the request itself."
        )
        rc = 1

    if registered and completed == 0:
        print(
            "STAGE3_FAIL sends were planned but none completed, so the plan "
            "was built and the bytes never moved."
        )
        rc = 1

    # The trim check, plus the parallelism invariant, which is free here.
    trim_checked = 0
    worst = None
    for event in registered:
        pages = event.get("group_pages") or []
        tokens = event.get("num_tokens")
        parallelism = event.get("parallelism")
        if parallelism is not None and parallelism != args.expect_parallelism:
            print(
                f"STAGE3_FAIL req={event.get('req_id')} ran at parallelism "
                f"{parallelism}, expected {args.expect_parallelism}."
            )
            rc = 1
        if not pages or not isinstance(tokens, int) or tokens < MIN_TRIM_TOKENS:
            continue
        untrimmed = math.ceil(tokens / MIN_PAGE_TOKENS)
        observed = max(pages)
        trim_checked += 1
        if worst is None or observed > worst[1]:
            worst = (event.get("req_id"), observed, untrimmed, tokens)
        if observed >= untrimmed:
            print(
                f"STAGE3_FAIL req={event.get('req_id')} n={tokens} reports "
                f"{observed} pages in its largest group, which is the "
                f"untrimmed count ceil({tokens}/{MIN_PAGE_TOKENS})="
                f"{untrimmed}. The sliding-window trim did not run, so the "
                f"plan still names the null block many times."
            )
            rc = 1

    if trim_checked:
        req, observed, untrimmed, tokens = worst
        print(
            f"STAGE3_TRIM_CHECKED {trim_checked} requests; worst case "
            f"req={req} n={tokens} max_group_pages={observed} "
            f"untrimmed_would_be={untrimmed}"
        )
    elif registered:
        print(
            f"STAGE3_TRIM_CHECKED 0 requests: none reached "
            f"{MIN_TRIM_TOKENS} tokens, so no page could fall out of the "
            f"window and the trim was never exercised."
        )

    # 1 for pass, matching QUICK_PROBE_OK and PREFIX_CACHE_CORRECTNESS_OK in
    # the smokes this prints beside. The process exit code runs the other way.
    print(f"STAGE3_CHECK_OK {int(rc == 0)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
