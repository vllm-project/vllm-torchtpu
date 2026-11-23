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
"""Checks of a Qwen3.5 prefill/decode pair that an accuracy score cannot see.

Run by .buildkite/kubernetes/qwen35_pd.sh. A pair can answer GSM8K well while
serving a different configuration from the one it claims, or while dropping
requests that a retry hid, so the run also checks:

  leg-status   (on each engine's pod) the engine's own log: the kernel page
               every worker settled on, and Raiden layout-fingerprint
               mismatches. Written as JSON, which the decode pod serves over
               HTTP for the prefill head to read.
  leg-check    a leg-status file or URL against the expected kernel page and
               zero mismatches.
  prefix-hit   one long prompt sent through the proxy more times than decode
               has API servers, so at least two land on one decode rank: the
               repeat must be served from decode's own prefix cache, and
               prefill must hit its own.
  transfer     the decode leg received prompt KV from prefill and computed
               almost none of its prompt tokens itself. Every answer would
               still be right, only slower, if the connector gave up and
               decode prefilled each request itself.
  mtp          the decode leg's MTP acceptance rate (accepted over proposed
               draft tokens, as main's P/D rigs gate it), summed over its API
               servers' /metrics, against a floor.

Each subcommand prints what it measured and exits 0 on pass, 1 on fail and 2
when it could not measure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections.abc import Iterable, Sequence

# Logged by every worker as it derives its block size
# (platforms/tpu_block_size_utils.py).
_KERNEL_PAGE_RE = re.compile(r"backend_supported_kernel_block_sizes=(\[[^\]]*\])")
# The controller's refusal when the two legs' KV layouts differ
# (distributed/kv_transfer/tpu_connector.py).
_FINGERPRINT_MISMATCH_RE = re.compile(r"fingerprint mismatch", re.IGNORECASE)


def leg_status(log_text: str) -> dict:
    """What one engine's log says about its KV layout."""
    return {
        "kernel_pages": sorted(set(_KERNEL_PAGE_RE.findall(log_text))),
        "kernel_page_lines": len(_KERNEL_PAGE_RE.findall(log_text)),
        "fingerprint_mismatches": len(_FINGERPRINT_MISMATCH_RE.findall(log_text)),
    }


def leg_failures(status: dict, page: int) -> list[str]:
    """Why a leg-status fails: no page logged, a page other than ``page``, or
    any fingerprint mismatch. Empty when it passes."""
    failures = []
    pages = status.get("kernel_pages")
    if not pages:
        failures.append("no worker logged its kernel page")
    elif pages != [f"[{page}]"]:
        failures.append(f"kernel pages {pages}, expected [[{page}]]")
    mismatches = status.get("fingerprint_mismatches")
    if not isinstance(mismatches, int):
        failures.append("no fingerprint-mismatch count")
    elif mismatches:
        failures.append(f"{mismatches} layout-fingerprint mismatch line(s)")
    return failures


def metric_sum(body: str, name: str, label: str = "") -> float | None:
    """A Prometheus counter summed over its series, or None if there is none.

    ``label`` is a substring the series' label set must contain.
    """
    total, found = 0.0, False
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        metric = series.split("{", 1)[0]
        if metric != name or (label and label not in series):
            continue
        total += float(value)
        found = True
    return total if found else None


def mtp_acceptance(bodies: Iterable[str]) -> tuple[float, float] | None:
    """MTP acceptance over several /metrics bodies: the rate (accepted over
    proposed draft tokens) and the mean acceptance length (1 + accepted draft
    tokens per draft), or None when no draft token was proposed."""
    drafts = draft_tokens = accepted = 0.0
    for body in bodies:
        drafts += metric_sum(body, "vllm:spec_decode_num_drafts_total") or 0.0
        draft_tokens += (
            metric_sum(body, "vllm:spec_decode_num_draft_tokens_total") or 0.0
        )
        accepted += (
            metric_sum(body, "vllm:spec_decode_num_accepted_tokens_total") or 0.0
        )
    if draft_tokens <= 0 or drafts <= 0:
        return None
    return accepted / draft_tokens, 1.0 + accepted / drafts


_BY_SOURCE = "vllm:prompt_tokens_by_source_total"


def decode_compute_share(bodies: Iterable[str]) -> tuple[float, float] | None:
    """Over the decode leg's /metrics bodies: the prompt tokens it received
    from prefill, and the percent of its prompt tokens it computed itself.
    None when the counters are missing."""
    transferred = computed = reused = 0.0
    found = False
    for body in bodies:
        values = [
            metric_sum(body, _BY_SOURCE, f'source="{source}"')
            for source in ("external_kv_transfer", "local_compute", "local_cache_hit")
        ]
        if all(v is None for v in values):
            continue
        found = True
        transferred += values[0] or 0.0
        computed += values[1] or 0.0
        reused += values[2] or 0.0
    if not found:
        return None
    total = transferred + computed + reused
    return transferred, (100.0 * computed / total) if total > 0 else 100.0


def _get(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def _post(url: str, body: dict, timeout: float = 1800) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _long_prompt() -> str:
    """About 8,000 tokens: several blocks on both legs, so a reused prefix
    covers whole decode blocks."""
    facts = " ".join(
        f"Record {i}: the code for crate {i} is {(i * 7919) % 100003}."
        for i in range(800)
    )
    return f"{facts}\nQuestion: what is the code for crate 17? Answer:"


def cmd_leg_status(args: argparse.Namespace) -> int:
    with open(args.log, errors="replace") as f:
        status = leg_status(f.read())
    with open(args.out + ".tmp", "w") as f:
        json.dump(status, f)
    # Replaced whole, so a reader never sees half a file.
    os.replace(args.out + ".tmp", args.out)
    print(json.dumps(status))
    return 0


def cmd_leg_check(args: argparse.Namespace) -> int:
    try:
        if args.status.startswith(("http://", "https://")):
            status = json.loads(_get(args.status))
        else:
            with open(args.status) as f:
                status = json.load(f)
    except (OSError, ValueError) as error:
        print(f"LEG_CHECK {args.name} could not read {args.status}: {error}")
        return 2
    failures = leg_failures(status, args.page)
    print(f"LEG_CHECK {args.name} {json.dumps(status)}")
    for failure in failures:
        print(f"LEG_CHECK_FAIL {args.name}: {failure}")
    return 1 if failures else 0


def cmd_prefix_hit(args: argparse.Namespace) -> int:
    prompt = _long_prompt()
    usage: dict = {}
    for i in range(args.repeats):
        try:
            response = _post(
                f"{args.proxy}/v1/completions",
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
        except OSError as error:
            print(f"PREFIX_HIT request {i} failed: {error}")
            return 1
        usage = response.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(
            f"PREFIX_HIT request {i}: prompt_tokens={usage.get('prompt_tokens')} "
            f"cached_tokens={cached}"
        )
    try:
        decode_bodies = [_get(url) for url in args.decode_metrics]
        prefill_body = _get(args.prefill_metrics)
    except OSError as error:
        print(f"PREFIX_HIT could not read /metrics: {error}")
        return 2
    local_hit = sum(
        metric_sum(b, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"')
        or 0.0
        for b in decode_bodies
    )
    prefill_hit = metric_sum(prefill_body, "vllm:prefix_cache_hits_total") or 0.0
    last_cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    print(
        f"PREFIX_HIT decode_local_cache_hit_tokens={local_hit:.0f} "
        f"prefill_prefix_cache_hit_tokens={prefill_hit:.0f} "
        f"last_cached_tokens={last_cached}"
    )
    failures = []
    if last_cached <= 0:
        failures.append("the last repeat reported no cached tokens")
    if local_hit <= 0:
        failures.append("decode served no prompt tokens from its own prefix cache")
    if prefill_hit <= 0:
        failures.append("prefill never hit its prefix cache")
    for failure in failures:
        print(f"PREFIX_HIT_FAIL {failure}")
    return 1 if failures else 0


def cmd_transfer(args: argparse.Namespace) -> int:
    try:
        bodies = [_get(url) for url in args.decode_metrics]
    except OSError as error:
        print(f"TRANSFER could not read /metrics: {error}")
        return 2
    share = decode_compute_share(bodies)
    if share is None:
        print(f"TRANSFER could not find {_BY_SOURCE} on decode")
        return 2
    transferred, computed_pct = share
    print(
        f"TRANSFER decode_external_kv_transfer_tokens={transferred:.0f} "
        f"decode_computed_pct={computed_pct:.2f} max={args.max_compute_pct}"
    )
    failures = []
    if transferred <= 0:
        failures.append("decode received no prompt tokens from prefill")
    if computed_pct > args.max_compute_pct:
        failures.append(
            f"decode computed {computed_pct:.2f}% of its prompt tokens itself, "
            f"over {args.max_compute_pct}%: requests fell back to prefilling on decode"
        )
    for failure in failures:
        print(f"TRANSFER_FAIL {failure}")
    return 1 if failures else 0


def cmd_mtp(args: argparse.Namespace) -> int:
    try:
        bodies = [_get(url) for url in args.metrics]
    except OSError as error:
        print(f"MTP could not read /metrics: {error}")
        return 2
    acceptance = mtp_acceptance(bodies)
    if acceptance is None:
        print("MTP_FAIL the decode leg proposed no MTP draft tokens")
        return 1
    rate, length = acceptance
    print(
        f"MTP acceptance_rate={rate:.4f} acceptance_length={length:.4f} "
        f"min_rate={args.min_rate}"
    )
    if rate < args.min_rate:
        print(f"MTP_FAIL acceptance rate {rate:.4f} is below {args.min_rate}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("leg-status")
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_leg_status)

    p = sub.add_parser("leg-check")
    p.add_argument("--status", required=True, help="leg-status file or URL")
    p.add_argument("--name", required=True)
    p.add_argument("--page", type=int, required=True)
    p.set_defaults(func=cmd_leg_check)

    p = sub.add_parser("prefix-hit")
    p.add_argument("--proxy", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--prefill-metrics", required=True)
    p.add_argument("--decode-metrics", nargs="+", required=True)
    p.set_defaults(func=cmd_prefix_hit)

    p = sub.add_parser("transfer")
    p.add_argument("--decode-metrics", nargs="+", required=True)
    p.add_argument("--max-compute-pct", type=float, required=True)
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser("mtp")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--min-rate", type=float, required=True)
    p.set_defaults(func=cmd_mtp)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
