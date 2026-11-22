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
"""Numbers every benchmark publisher derives the same way.

Throughput per physical chip and interactivity (the inverse of the time per
output token), so rows from different publishers compare and join.
"""

import math


def _number(value) -> float | None:
    """`value` when it is a finite real number, else None. bool is not a number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return None
    return value


def _positive_int(value) -> int | None:
    """`value` when it is a positive integer (excluding bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def physical_chips(chips=None, prefill_chips=None, decode_chips=None) -> int | None:
    """Physical accelerator chips serving a run, or None when not stated.

    A disaggregated run states its two pools and the count is their sum; any
    other run states one total. The count is never derived from tensor or
    data parallelism: those count ranks, and a v7x chip holds two.
    """
    for name, value in (
        ("chips", chips),
        ("prefill_chips", prefill_chips),
        ("decode_chips", decode_chips),
    ):
        if value is not None and _positive_int(value) is None:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if (prefill_chips is None) != (decode_chips is None):
        raise ValueError(
            "prefill_chips and decode_chips go together: a disaggregated run "
            "is served by both pools"
        )
    if prefill_chips is not None:
        return prefill_chips + decode_chips
    return chips


def interactivity(tpot_ms) -> float | None:
    """Output tokens per second one user sees: the inverse of a time per
    output token given in milliseconds."""
    tpot_ms = _number(tpot_ms)
    if not tpot_ms or tpot_ms < 0:
        return None
    return 1000.0 / tpot_ms


def derived_metrics(
    *,
    chips=None,
    prefill_chips=None,
    decode_chips=None,
    output_tput=None,
    total_tput=None,
    tpot_ms=None,
) -> dict:
    """The `derived` element: throughput per physical chip and interactivity.

    For an aggregated run (`chips` only), all three throughputs divide by
    `chips`. For a disaggregated run (`prefill_chips` and `decode_chips`),
    output throughput divides by `decode_chips`, input throughput divides by
    `prefill_chips`, and total throughput divides by their sum, matching
    InferenceX's per-GPU formulas.

    `tpot_ms` maps a statistic name (median, p90, p99) to that statistic of
    the time per output token; each becomes "<stat>_intvty", so p90_intvty is
    the interactivity at the P90 token latency. A number that cannot be
    computed is left out rather than guessed, and the element is empty when
    nothing can be.
    """
    out = {}
    output_tput = _number(output_tput)
    total_tput = _number(total_tput)
    total_chips = None
    prefill_denom = None
    decode_denom = None
    if prefill_chips is not None or decode_chips is not None:
        p = _positive_int(prefill_chips)
        d = _positive_int(decode_chips)
        if (
            p is not None
            and d is not None
            and (chips is None or _positive_int(chips) == p + d)
        ):
            total_chips = p + d
            prefill_denom = p
            decode_denom = d
    else:
        total_chips = _positive_int(chips)
        prefill_denom = total_chips
        decode_denom = total_chips

    if total_chips is not None:
        if output_tput is not None:
            out["output_tput_per_chip"] = output_tput / decode_denom
        if total_tput is not None:
            out["total_tput_per_chip"] = total_tput / total_chips
        if output_tput is not None and total_tput is not None:
            out["input_tput_per_chip"] = (total_tput - output_tput) / prefill_denom
        if out:
            out = {"num_chips": total_chips, **out}
    for stat, value in (tpot_ms or {}).items():
        intvty = interactivity(value)
        if intvty is not None:
            out[f"{stat}_intvty"] = intvty
    return out


def bench_derived_metrics(
    data: dict, chips=None, *, prefill_chips=None, decode_chips=None
) -> dict:
    """derived_metrics for a `vllm bench serve` payload."""
    return derived_metrics(
        chips=chips,
        prefill_chips=prefill_chips,
        decode_chips=decode_chips,
        output_tput=data.get("output_throughput"),
        total_tput=data.get("total_token_throughput"),
        tpot_ms={
            "median": data.get("median_tpot_ms"),
            "p90": data.get("p90_tpot_ms"),
            "p99": data.get("p99_tpot_ms"),
        },
    )


def agentx_derived_metrics(export: dict) -> dict:
    """derived_metrics for an aiperf export of an AgentX replay. aiperf
    reports the time per output token as inter_token_latency, in
    milliseconds."""
    latency = export.get("inter_token_latency", {})
    return derived_metrics(
        tpot_ms={
            "median": latency.get("p50"),
            "p90": latency.get("p90"),
            "p99": latency.get("p99"),
        },
    )
