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
"""Shared helpers for inserting benchmark rows into BigQuery.

Table schema: record_id, created_time, repository, run_type, model_id,
metrics ARRAY<STRUCT<type STRING, values JSON>>, config JSON. Each
metrics element's type names the tool that produced its payload; values
keeps that tool's flat leaf keys, so generic names (completed, failed,
duration) can't clash with other harnesses writing to the shared table and
provenance stays explicit.

One row is one measured point. Its metrics elements, all optional:

  vllm_bench  one `vllm bench serve` cell (BENCH_METRIC_FIELDS, verbatim).
  lm_eval     {task: score} from results_<task>.json. It rides every
              vllm_bench row of the run, or is the only element of the
              run's accuracy row when the run measured no speed grid.
  derived     numbers normalized the same way for every tool, so rows from
              different uploaders compare and join: num_chips,
              *_tput_per_chip and *_intvty (derived_metrics).
  routing     fleet routing health of disaggregated serving, as fractions
              in [0, 1]: prefix_cache_hit_rate, batch_utilization.

`derived` and `routing` are separate elements rather than extra keys in the
tool's own payload, so the existing vllm_bench and lm_eval payloads are
unchanged and a reader selects the normalized numbers by one type.
"""

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BQ_PROJECT_ID = "cloud-ullm-inference-ci-cd"
DEFAULT_BQ_TABLE = "cloud-ullm-inference-ci-cd.llm_benchmark_analytics.benchmark_runs"

# Leaf keys exactly as `vllm bench serve` emits them.
BENCH_METRIC_FIELDS = (
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "rtfx",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "completed",
    "failed",
    "total_input_tokens",
    "total_output_tokens",
    "duration",
    # Present only when the server runs speculative decoding.
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
    "spec_decode_num_drafts",
    "spec_decode_draft_tokens",
    "spec_decode_accepted_tokens",
)

ROUTING_METRIC_FIELDS = ("prefix_cache_hit_rate", "batch_utilization")
ROUTING_METRICS_FILENAME = "routing_metrics.json"

# Environment a launcher states the chip count in, for callers with no flag
# of their own. AGENTX_CHIPS is what the AgentX client already requires.
PREFILL_CHIPS_ENV = "PREFILL_CHIPS"
DECODE_CHIPS_ENV = "DECODE_CHIPS"
CHIPS_ENVS = ("NUM_CHIPS", "AGENTX_CHIPS")


def sql_escape(val: str) -> str:
    return val.replace("'", "''")


def _number(value) -> float | None:
    """`value` when it is a finite real number, else None. bool is not a number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
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
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if (prefill_chips is None) != (decode_chips is None):
        raise ValueError(
            "prefill_chips and decode_chips go together: a disaggregated run "
            "is served by both pools"
        )
    if prefill_chips is not None:
        return prefill_chips + decode_chips
    return chips


def chips_from_env(env=None) -> int | None:
    """physical_chips from the launcher's environment (see CHIPS_ENVS)."""
    env = os.environ if env is None else env

    def read(name):
        raw = (env.get(name) or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            raise ValueError(
                f"{name} must be a positive integer, got {raw!r}"
            ) from None

    total = next((v for v in map(read, CHIPS_ENVS) if v is not None), None)
    return physical_chips(total, read(PREFILL_CHIPS_ENV), read(DECODE_CHIPS_ENV))


def interactivity(tpot_ms) -> float | None:
    """Output tokens per second one user sees: the inverse of a time per
    output token given in milliseconds."""
    tpot_ms = _number(tpot_ms)
    if not tpot_ms or tpot_ms < 0:
        return None
    return 1000.0 / tpot_ms


def derived_metrics(
    *, chips=None, output_tput=None, total_tput=None, tpot_ms=None
) -> dict:
    """The `derived` element: throughput per physical chip and interactivity.

    `tpot_ms` maps a statistic name (median, p90, p99) to that statistic of
    the time per output token; each becomes "<stat>_intvty", so p90_intvty is
    the interactivity at the P90 token latency. A number that cannot be
    computed is left out rather than guessed, and the element is empty when
    nothing can be.
    """
    out = {}
    output_tput = _number(output_tput)
    total_tput = _number(total_tput)
    if isinstance(chips, int) and not isinstance(chips, bool) and chips > 0:
        if output_tput is not None:
            out["output_tput_per_chip"] = output_tput / chips
        if total_tput is not None:
            out["total_tput_per_chip"] = total_tput / chips
        if output_tput is not None and total_tput is not None:
            out["input_tput_per_chip"] = (total_tput - output_tput) / chips
        if out:
            out = {"num_chips": chips, **out}
    for stat, value in (tpot_ms or {}).items():
        intvty = interactivity(value)
        if intvty is not None:
            out[f"{stat}_intvty"] = intvty
    return out


def bench_derived_metrics(data: dict, chips=None) -> dict:
    """derived_metrics for a `vllm bench serve` payload."""
    return derived_metrics(
        chips=chips,
        output_tput=data.get("output_throughput"),
        total_tput=data.get("total_token_throughput"),
        tpot_ms={
            "median": data.get("median_tpot_ms"),
            "p90": data.get("p90_tpot_ms"),
            "p99": data.get("p99_tpot_ms"),
        },
    )


def extract_routing_metrics(data) -> dict:
    """The `routing` element from a flat mapping: the ROUTING_METRIC_FIELDS
    that are fractions in [0, 1]. Anything else is dropped with a warning,
    since a percentage stored as a fraction would read as a pass."""
    out = {}
    if not isinstance(data, dict):
        print(
            f"Warning: ignoring routing metrics payload {type(data).__name__}: "
            "expected a JSON object",
            file=sys.stderr,
        )
        return out
    for key in ROUTING_METRIC_FIELDS:
        if data.get(key) is None:
            continue
        value = _number(data[key])
        if value is None or not 0.0 <= value <= 1.0:
            print(
                f"Warning: ignoring routing metric {key}={data[key]!r}: "
                "expected a fraction in [0, 1]",
                file=sys.stderr,
            )
            continue
        out[key] = value
    return out


def load_routing_metrics(directory) -> dict:
    """extract_routing_metrics of DIRECTORY/routing_metrics.json, which the
    harness in front of a disaggregated fleet writes; empty when absent."""
    path = Path(directory) / ROUTING_METRICS_FILENAME
    if not path.is_file():
        return {}
    try:
        return extract_routing_metrics(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return {}


def with_shared_elements(metrics: list, derived: dict, routing: dict) -> list:
    """`metrics` plus the derived and routing elements that have content."""
    if derived:
        metrics.append(("derived", derived))
    if routing:
        metrics.append(("routing", routing))
    return metrics


def extract_bench_metrics(data: dict) -> dict:
    """The `vllm bench serve` payload, with absent metrics omitted so the
    JSON stays compact."""
    return {k: data[k] for k in BENCH_METRIC_FIELDS if data.get(k) is not None}


def created_time_sql(date_str) -> str:
    """TIMESTAMP literal from a measurement time, falling back to insert
    time. Accepts a `vllm bench serve` `date` stamp (20261020-104845) or
    aiperf's ISO 8601 `end_time`. A time with an offset is converted to UTC;
    one without is taken as UTC."""
    if date_str:
        try:
            try:
                dt = datetime.strptime(date_str, "%Y%m%d-%H%M%S")
            except ValueError:
                dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc)  # noqa: UP017
            return f"TIMESTAMP '{dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        except (ValueError, TypeError):
            print(
                f"Warning: Could not parse date string: {date_str}. "
                "Using CURRENT_TIMESTAMP()",
                file=sys.stderr,
            )
    return "CURRENT_TIMESTAMP()"


def build_insert_sql(
    record_id: str,
    created_time: str,
    run_type: str,
    model_id: str,
    metrics: list,
    config: dict,
    bq_table=None,
) -> str:
    """Render one INSERT statement for the benchmark_runs table.

    `metrics` is a list of (type, payload-dict) pairs, one array element
    per producing tool. `created_time` is a SQL expression (see
    created_time_sql). `bq_table` falls back to the BQ_TABLE env var, then
    the default; pipelines may export BQ_TABLE as an empty string, so `or`
    (empty falls through) is used instead of os.getenv's unset-only
    default. Backticks around an override are accepted and normalized.
    """
    table = (bq_table or os.getenv("BQ_TABLE") or DEFAULT_BQ_TABLE).strip("`")
    repository = os.getenv("REPOSITORY") or "vllm-torchtpu"
    # Which fleet produced the row, when the fleet says. Older rows in this
    # table came from the VM fleet and carry no runner, so a reader can hold
    # the fleet constant. In config rather than a column of its own because
    # config is JSON for exactly this: a dimension one producer has and the
    # others do not.
    runner = os.getenv("CI_RUNNER")
    if runner:
        config = {**config, "runner": runner}
    config_json = sql_escape(json.dumps(config))
    metrics_elems = ", ".join(
        f"STRUCT('{metrics_type}', "
        f"PARSE_JSON('{sql_escape(json.dumps(payload))}', "
        "wide_number_mode=>'round'))"
        for metrics_type, payload in metrics
    )
    sql = f"""
    INSERT INTO `{table}` (
        record_id, created_time, repository, run_type, model_id, metrics, config
    ) VALUES (
        '{sql_escape(record_id)}', {created_time},
        '{sql_escape(repository)}', '{sql_escape(run_type)}',
        '{sql_escape(model_id)}',
        [{metrics_elems}],
        PARSE_JSON('{config_json}', wide_number_mode=>'round')
    );
    """
    # Single line so the statement survives shell/CLI round-trips intact.
    return " ".join(sql.split())


def run_insert(
    sql: str, project=None, label: str = "", record_id: str = "", skip: bool = False
) -> bool:
    """Execute one INSERT via the bq CLI. Prints the statement either way;
    with skip=True it is printed but not executed."""
    print(f"SQL for BigQuery ({label}):")
    print(sql)

    if skip:
        print(f"=== Skipping BigQuery upload ===. record_id: {record_id}")
        return True

    project = project or os.getenv("BQ_PROJECT_ID") or DEFAULT_BQ_PROJECT_ID
    cmd = ["bq", "query", "--use_legacy_sql=false", f"--project_id={project}", sql]
    print(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            f"Failed to insert BigQuery record for {label}: 'bq' CLI not found.",
            file=sys.stderr,
        )
        return False
    if proc.returncode != 0:
        print(f"Failed to insert BigQuery record for {label}!", file=sys.stderr)
        print(f"Stdout:\n{proc.stdout}", file=sys.stderr)
        print(f"Stderr:\n{proc.stderr}", file=sys.stderr)
        return False
    print(f"Successfully inserted BigQuery record for {label}. record_id: {record_id}")
    return True
