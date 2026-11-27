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

"""Upload one results directory to Spanner and BigQuery.

Each measured point becomes one BigQuery row (see bq_utils for the row
shape):

  isl*_osl*_c*.json  one `vllm bench serve` cell per row, with the run's
                     accuracy scores on every row. Also the only rows written
                     to Spanner.
  conc_<N>/          one AgentX replay point per row. agentx_client.sh writes
                     each point into a directory named after its concurrency
                     N, a positive integer with no leading zero. Each export
                     path must carry exactly one conc_<N>/ directory (nested
                     conc_<N>/ segments are rejected as ambiguous); an export
                     under no conc_<N>/ is not a point and is skipped; two
                     exports for the same concurrency N anywhere under the
                     results directory stop the upload before any row is
                     written.
  results_*.json     one accuracy row for the run, when the results directory
                     has no speed-grid cell file to carry the scores.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import bench_metrics
import bq_utils
from bq_utils import sql_escape

# GCP Spanner default coordinates
DEFAULT_PROJECT_ID = "cloud-tpu-inference-test"
DEFAULT_INSTANCE_ID = "vllm-bm-inst"
DEFAULT_DATABASE_ID = "vllm-bm-bk-runs"

RESULT_FILENAME_RE = re.compile(r"^isl(\d+)_osl(\d+)_c(\d+)\.json$")
AGENTX_EXPORT_FILENAME = "profile_export_aiperf.json"
# N as agentx_client.sh accepts a concurrency: positive, no leading zero.
AGENTX_POINT_DIR_RE = re.compile(r"conc_([1-9][0-9]*)")
ACC_KEYS = (
    "exact_match,strict_match",
    "exact_match,custom-extract",
    "exact_match,flexible-extract",
    "acc,none",
    "exact_match,none",
    "acc_norm,none",
    "pass@1,create_test",
    "pass_at_1,create_test",
    "pass_at_1,none",
)


@dataclass(frozen=True)
class AccuracyResults:
    metrics: dict[str, float]
    model_names: frozenset[str]
    parse_errors: tuple[Path, ...]


@dataclass(frozen=True)
class PlannedInsert:
    label: str
    record_id: str
    sql: str


@dataclass(frozen=True)
class PlannedRow:
    bq: PlannedInsert
    spanner: PlannedInsert | None = None


@dataclass(frozen=True)
class _RunContext:
    config: dict
    run_config: dict
    run_type: str
    created_by: str
    device: str
    code_hash: str
    job_ref: str | None
    run_by: str | None
    bq_table: str | None


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def load_config_json(results_dir: Path) -> dict:
    config_path = results_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            print(
                "Warning: config.json is not a JSON object "
                f"(got {type(data).__name__})",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}", file=sys.stderr)
    return {}


def load_accuracy_metrics(results_dir: Path) -> AccuracyResults:
    metrics: dict[str, float] = {}
    model_names: set[str] = set()
    parse_errors: list[Path] = []

    # 1. Parse LM-Eval results
    # Oldest-first so that if a task genuinely is re-run, the newer value wins.
    eval_jsons = sorted(
        results_dir.rglob("results_*.json"),
        key=lambda p: (p.stat().st_mtime, str(p)),
    )
    for path in eval_jsons:
        try:
            with open(path) as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"expected JSON object, got {type(data).__name__}")
            results = data.get("results")
            if results is not None and not isinstance(results, dict):
                raise ValueError(
                    f"expected 'results' to be a JSON object, "
                    f"got {type(results).__name__}"
                )
            file_has_metrics = False
            for task, task_metrics in (results or {}).items():
                if not isinstance(task_metrics, dict):
                    continue
                # Find the primary accuracy metric
                for key in ACC_KEYS:
                    val = task_metrics.get(key)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        metrics[task] = float(val)
                        file_has_metrics = True
                        break
            if file_has_metrics:
                model_name = _non_empty_str(data.get("model_name"))
                if model_name:
                    model_names.add(model_name)
        except Exception as e:
            print(
                f"Error: Failed to parse lm-eval results from {path}: {e}",
                file=sys.stderr,
            )
            parse_errors.append(path)

    return AccuracyResults(
        metrics=metrics,
        model_names=frozenset(model_names),
        parse_errors=tuple(parse_errors),
    )


def find_speed_grid_files(results_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(results_dir.glob("isl*_osl*_c*.json"))
        if RESULT_FILENAME_RE.match(p.name)
    ]


def engine_flags(config: dict) -> dict:
    return {
        "max_num_seqs": config.get("max_num_seqs"),
        "max_num_batched_tokens": config.get("max_num_batched_tokens"),
        "tensor_parallel_size": config.get("tensor_parallelism"),
        "data_parallel_size": config.get("data_parallelism", 1),
        "max_model_len": config.get("max_model_len"),
        "enable_ep": bool(config.get("enable_ep")),
        "quantization": config.get("quantization"),
        "multihost_backend": os.getenv("TPU_MULTIHOST_BACKEND") or None,
    }


def find_agentx_points(results_dir: Path) -> list[tuple[int, Path]]:
    """(concurrency, aiperf export) of each AgentX replay point under
    RESULTS_DIR, by concurrency.

    A point is identified by a single conc_<N>/ directory on an export's path,
    and N is its concurrency. An export with no such directory above it is not
    a point; it is named on stderr and skipped.

    Raises:
        ValueError: An export path contains multiple conc_<N>/ directories
            (ambiguous concurrency), or two exports share the same concurrency
            anywhere under RESULTS_DIR.
    """
    points: dict[int, Path] = {}
    for export in sorted(results_dir.rglob(AGENTX_EXPORT_FILENAME)):
        directories = export.relative_to(results_dir).parts[:-1]
        matches = [
            (name, int(match.group(1)))
            for name in directories
            if (match := AGENTX_POINT_DIR_RE.fullmatch(name))
        ]
        if not matches:
            print(
                f"Skipping {export}: not under a conc_<N>/ directory, so not "
                "an AgentX point.",
                file=sys.stderr,
            )
            continue
        if len(matches) > 1:
            segments = ", ".join(name for name, _ in matches)
            raise ValueError(
                f"ambiguous AgentX export path {export}: nested concurrency "
                f"directories ({segments}). Keep only one conc_<N>/ directory "
                "per export path."
            )
        _, concurrency = matches[0]
        if concurrency in points:
            raise ValueError(
                f"two AgentX exports for concurrency {concurrency}: "
                f"{points[concurrency]} and {export}. One concurrency is one "
                "row; remove the export that is not the point's result."
            )
        points[concurrency] = export
    return sorted(points.items())


def _agentx_export_model(export: dict) -> str | None:
    """First non-empty model name in an aiperf export's input_config."""
    input_config = export.get("input_config")
    if not isinstance(input_config, dict):
        return None
    models = input_config.get("models")
    if not isinstance(models, dict):
        return None
    items = models.get("items")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return None
    return _non_empty_str(items[0].get("name"))


def agentx_workload(export: dict, concurrency: int) -> dict:
    """What one AgentX replay point ran: the replay its aiperf export
    describes, at the concurrency its conc_<N>/ directory names."""
    raw_metadata = export.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    input_config = export.get("input_config")
    datasets = input_config.get("datasets") if isinstance(input_config, dict) else None
    dataset = (
        datasets[0]
        if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict)
        else {}
    )
    return {
        "scenario": metadata.get("scenario"),
        "dataset": dataset.get("dataset"),
        # Set only when the client capped the traces it replays.
        "max_context_length": dataset.get("max_context_length"),
        "max_concurrency": concurrency,
    }


def _resolve_accuracy_model_id(
    config: dict,
    accuracy_models: frozenset[str],
    agentx_models: frozenset[str],
) -> str:
    """Resolve model_id for a standalone accuracy row.

    Checks config.json 'model' first, then a consistent non-empty 'model_name'
    across results_*.json, then a consistent model name across AgentX exports.
    """
    config_model = _non_empty_str(config.get("model"))
    if config_model:
        return config_model
    if len(accuracy_models) == 1:
        return next(iter(accuracy_models))
    if len(accuracy_models) > 1:
        names = ", ".join(sorted(accuracy_models))
        raise ValueError(
            f"conflicting 'model_name' values across results_*.json ({names}) "
            "and no 'model' in config.json"
        )
    if len(agentx_models) == 1:
        return next(iter(agentx_models))
    if len(agentx_models) > 1:
        names = ", ".join(sorted(agentx_models))
        raise ValueError(
            f"conflicting model names across AgentX exports ({names}) "
            "and no 'model' in config.json or results_*.json"
        )
    raise ValueError(
        "Missing required config parameter 'model' in config.json "
        "(and no consistent 'model_name' in results_*.json or AgentX exports)"
    )


def _plan_bq_insert(
    ctx: _RunContext,
    *,
    label: str,
    record_id: str,
    created_time: str,
    model_id: str,
    metrics: list[tuple[str, dict]],
    config: dict,
) -> PlannedInsert:
    sql = bq_utils.build_insert_sql(
        record_id=record_id,
        created_time=created_time,
        run_type=ctx.run_type,
        model_id=model_id,
        metrics=metrics,
        config=config,
        bq_table=ctx.bq_table,
    )
    return PlannedInsert(label=label, record_id=record_id, sql=sql)


def _plan_spanner_insert(
    ctx: _RunContext,
    *,
    label: str,
    record_id: str,
    res: dict,
    input_len: int,
    output_len: int,
    concurrency: int,
    accuracy_metrics: dict[str, float],
) -> PlannedInsert:
    config = ctx.config
    tensor_parallelism = config["tensor_parallelism"]
    columns = {
        "RecordId": f"'{record_id}'",
        "RunType": f"'{sql_escape(ctx.run_type)}'",
        "MaxNumSeqs": str(config["max_num_seqs"]),
        "MaxNumBatchedTokens": str(config["max_num_batched_tokens"]),
        "TensorParallelSize": str(tensor_parallelism),
        "MaxModelLen": str(config["max_model_len"]),
        "Dataset": f"'{sql_escape(res['backend'])}'",
        "CreatedBy": f"'{sql_escape(ctx.created_by)}'",
        "InputLen": str(input_len),
        "OutputLen": str(output_len),
        "Device": f"'{sql_escape(ctx.device)}'",
        "CodeHash": f"'{sql_escape(ctx.code_hash)}'",
        "Model": f"'{sql_escape(config['model'])}'",
        "Status": "'COMPLETED'",
        "LastUpdate": "CURRENT_TIMESTAMP()",
        "CreatedTime": "CURRENT_TIMESTAMP()",
    }

    # Optional metrics (omit from columns list if missing,
    # so Spanner defaults to NULL)
    optional_metrics = {
        "Throughput": res.get("request_throughput"),
        "OutputTokenThroughput": res.get("output_throughput"),
        "TotalTokenThroughput": res.get("total_token_throughput"),
        "MedianTTFT": res.get("median_ttft_ms"),
        "P99TTFT": res.get("p99_ttft_ms"),
        "MedianTPOT": res.get("median_tpot_ms"),
        "P99TPOT": res.get("p99_tpot_ms"),
        "MedianITL": res.get("median_itl_ms"),
        "P99ITL": res.get("p99_itl_ms"),
        "NumPrompts": res.get("num_prompts"),
    }

    for col_name, val in optional_metrics.items():
        if val is not None:
            columns[col_name] = str(val)

    if ctx.job_ref is not None:
        columns["JobReference"] = f"'{sql_escape(str(ctx.job_ref))}'"
    if ctx.run_by is not None:
        columns["RunBy"] = f"'{sql_escape(ctx.run_by)}'"

    # ExtraArgs construction
    extra_args_parts = [
        f"tp={tensor_parallelism}",
        f"dp={config.get('data_parallelism', 1)}",
    ]
    if config.get("enable_ep"):
        extra_args_parts.append("enable_ep")
    if config.get("quantization"):
        extra_args_parts.append(f"quantization={config['quantization']}")
    extra_args_parts.append(f"concurrency={concurrency}")
    columns["ExtraArgs"] = f"'{sql_escape(' '.join(extra_args_parts))}'"

    # Add accuracy metrics if any were parsed
    if accuracy_metrics:
        columns["AccuracyMetrics"] = f"JSON '{json.dumps(accuracy_metrics)}'"

    keys_str = ", ".join(columns.keys())
    vals_str = ", ".join(columns.values())
    sql = f"INSERT INTO RunRecord ({keys_str}) VALUES ({vals_str});"
    return PlannedInsert(label=label, record_id=record_id, sql=sql)


def _plan_speed_grid_rows(
    ctx: _RunContext,
    result_files: list[Path],
    accuracy_metrics: dict[str, float],
) -> tuple[list[PlannedRow], bool]:
    planned: list[PlannedRow] = []
    all_ok = True
    config = ctx.config

    for rf in result_files:
        m = RESULT_FILENAME_RE.match(rf.name)
        if not m:
            continue

        input_len = int(m.group(1))
        output_len = int(m.group(2))
        concurrency = int(m.group(3))

        try:
            with open(rf) as f:
                res = json.load(f)
        except Exception as e:
            print(f"Error loading {rf}: {e}", file=sys.stderr)
            all_ok = False
            continue

        if not isinstance(res, dict):
            print(
                f"Error: Speed-grid result {rf} is not a JSON object "
                f"(got {type(res).__name__})",
                file=sys.stderr,
            )
            all_ok = False
            continue

        try:
            for key in (
                "model",
                "max_num_seqs",
                "max_num_batched_tokens",
                "tensor_parallelism",
                "max_model_len",
            ):
                _ = config[key]
        except KeyError as e:
            print(
                f"Error: Missing required config parameter {e} in config.json",
                file=sys.stderr,
            )
            all_ok = False
            continue

        try:
            dataset = res["backend"]
        except KeyError as e:
            print(f"Error: Missing required metric {e} in {rf.name}", file=sys.stderr)
            all_ok = False
            continue

        record_id = str(uuid.uuid4())
        spanner = _plan_spanner_insert(
            ctx,
            label=rf.name,
            record_id=record_id,
            res=res,
            input_len=input_len,
            output_len=output_len,
            concurrency=concurrency,
            accuracy_metrics=accuracy_metrics,
        )

        # Dual write to BigQuery while dashboards migrate off Spanner. The
        # row shares the Spanner RecordId so the two writes can be
        # cross-referenced. Accuracy metrics are per-run but stamped onto
        # every cell row (like Spanner's AccuracyMetrics) — dedupe when
        # aggregating them across rows.
        bq_metrics = [("vllm_bench", bq_utils.extract_bench_metrics(res))]
        if accuracy_metrics:
            bq_metrics.append(("lm_eval", accuracy_metrics))
        bq_config = {
            **ctx.run_config,
            "backend": dataset,
            "engine_flags": engine_flags(config),
            "workload": {
                "input_len": input_len,
                "output_len": output_len,
                "num_prompts": res.get("num_prompts"),
                "request_rate": res.get("request_rate"),
                "burstiness": res.get("burstiness"),
                "max_concurrency": res.get("max_concurrency", concurrency),
            },
        }
        bq = _plan_bq_insert(
            ctx,
            label=rf.name,
            record_id=record_id,
            created_time=bq_utils.created_time_sql(res.get("date")),
            model_id=config["model"],
            metrics=bq_metrics,
            config=bq_config,
        )
        planned.append(PlannedRow(bq=bq, spanner=spanner))

    return planned, all_ok


def _plan_agentx_rows(
    ctx: _RunContext,
    results_dir: Path,
    agentx_points: list[tuple[int, Path]],
) -> tuple[list[PlannedRow], frozenset[str], bool]:
    planned: list[PlannedRow] = []
    export_models: set[str] = set()
    all_ok = True
    config = ctx.config
    config_model = _non_empty_str(config.get("model"))

    for concurrency, export_path in agentx_points:
        try:
            export = json.loads(export_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading {export_path}: {e}", file=sys.stderr)
            all_ok = False
            continue
        if not isinstance(export, dict):
            print(
                f"Error: AgentX export {export_path} is not a JSON object "
                f"(got {type(export).__name__})",
                file=sys.stderr,
            )
            all_ok = False
            continue
        export_model = _agentx_export_model(export)
        # A client that drives servers it did not start has no config.json;
        # the export names the model its requests went to.
        model_id = config_model or export_model
        if not model_id:
            print(
                f"Error: AgentX export {export_path} is missing a model name "
                "(set 'model' in config.json or input_config.models.items[0].name)",
                file=sys.stderr,
            )
            all_ok = False
            continue
        if export_model:
            export_models.add(export_model)

        record_id = str(uuid.uuid4())
        bq_metrics = [("agentx", bq_utils.extract_agentx_metrics(export))]
        derived = bench_metrics.agentx_derived_metrics(export)
        if derived:
            bq_metrics.append(("derived", derived))
        bq_config = {
            **ctx.run_config,
            **({"engine_flags": engine_flags(config)} if config else {}),
            "workload": agentx_workload(export, concurrency),
        }
        bq = _plan_bq_insert(
            ctx,
            label=str(export_path.relative_to(results_dir)),
            record_id=record_id,
            created_time=bq_utils.created_time_sql(export.get("end_time")),
            model_id=model_id,
            metrics=bq_metrics,
            config=bq_config,
        )
        planned.append(PlannedRow(bq=bq))

    return planned, frozenset(export_models), all_ok


def _plan_accuracy_row(
    ctx: _RunContext,
    accuracy: AccuracyResults,
    agentx_models: frozenset[str],
) -> PlannedRow:
    model_id = _resolve_accuracy_model_id(
        ctx.config, accuracy.model_names, agentx_models
    )
    record_id = str(uuid.uuid4())
    bq_config = {
        **ctx.run_config,
        **({"engine_flags": engine_flags(ctx.config)} if ctx.config else {}),
    }
    bq = _plan_bq_insert(
        ctx,
        label="accuracy",
        record_id=record_id,
        # The result files of the suites carry no common measurement time.
        created_time=bq_utils.created_time_sql(None),
        model_id=model_id,
        metrics=[("lm_eval", accuracy.metrics)],
        config=bq_config,
    )
    return PlannedRow(bq=bq)


def _run_spanner_insert(insert: PlannedInsert, args: argparse.Namespace) -> bool:
    print(f"SQL for Spanner ({insert.label}):")
    print(insert.sql)

    if args.skip_db_upload or args.skip_spanner:
        why = "--skip-db-upload" if args.skip_db_upload else "--skip-spanner"
        print(
            f"=== Skipping Spanner DB Upload ({why} specified) ===. "
            f"RecordId: {insert.record_id}"
        )
        return True

    cmd = [
        "gcloud",
        "spanner",
        "databases",
        "execute-sql",
        args.database,
        f"--project={args.project}",
        f"--instance={args.instance}",
        f"--sql={insert.sql}",
    ]
    print(f"Executing: {' '.join(cmd)}")
    res_proc = subprocess.run(cmd, capture_output=True, text=True)
    if res_proc.returncode != 0:
        print(f"Failed to update Spanner record for {insert.label}!", file=sys.stderr)
        print(f"Stdout:\n{res_proc.stdout}", file=sys.stderr)
        print(f"Stderr:\n{res_proc.stderr}", file=sys.stderr)
        return False
    print(
        f"Successfully updated Spanner record for {insert.label}. "
        f"RecordId: {insert.record_id}"
    )
    return True


def _run_bq_insert(insert: PlannedInsert, args: argparse.Namespace) -> bool:
    return bq_utils.run_insert(
        insert.sql,
        project=args.bq_project,
        label=insert.label,
        record_id=insert.record_id,
        skip=args.skip_db_upload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Upload performance benchmark results to Spanner and "
        "BigQuery (dual write during the migration to BigQuery)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark result files.",
    )
    parser.add_argument(
        "--skip-db-upload",
        action="store_true",
        help="If set, only print SQL query without executing upload.",
    )
    parser.add_argument(
        "--skip-spanner",
        action="store_true",
        help="Skip the Spanner insert but still write to BigQuery. For a "
        "fleet that reports into the shared table and has no Spanner "
        "record of its own; --skip-db-upload still skips both.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GCP_PROJECT_ID", DEFAULT_PROJECT_ID),
        help="GCP project ID.",
    )
    parser.add_argument(
        "--instance",
        default=os.getenv("GCP_INSTANCE_ID", DEFAULT_INSTANCE_ID),
        help="Spanner instance ID.",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("GCP_DATABASE_ID", DEFAULT_DATABASE_ID),
        help="Spanner database ID.",
    )
    parser.add_argument(
        "--bq-project",
        default=None,
        help="GCP project ID used to run BigQuery jobs "
        "(default: BQ_PROJECT_ID env or "
        f"{bq_utils.DEFAULT_BQ_PROJECT_ID}).",
    )
    parser.add_argument(
        "--bq-table",
        default=None,
        help="Fully-qualified BigQuery table "
        "(default: BQ_TABLE env or "
        f"{bq_utils.DEFAULT_BQ_TABLE}).",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: results-dir does not exist: {args.results_dir}", file=sys.stderr)
        return 1

    config = load_config_json(args.results_dir)
    accuracy = load_accuracy_metrics(args.results_dir)
    result_files = find_speed_grid_files(args.results_dir)
    try:
        agentx_points = find_agentx_points(args.results_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not (result_files or agentx_points or accuracy.metrics):
        if accuracy.parse_errors:
            return 1
        print(
            f"Nothing to upload from {args.results_dir}: no speed-grid cell, "
            "AgentX point or accuracy result."
        )
        return 0

    code_hash = os.getenv("BUILDKITE_COMMIT")
    if not code_hash:
        print(
            "Error: BUILDKITE_COMMIT environment variable is not set.", file=sys.stderr
        )
        return 1

    device = os.getenv("TPU_NAME")
    if not device:
        accelerator = os.getenv("TPU_ACCELERATOR_TYPE")
        topology = os.getenv("TPU_TOPOLOGY")
        if os.getenv("CI_RUNNER") == "kube" and accelerator and topology:
            # Every kube step runs off the one "kube" queue, which names no
            # TPU, so the pod's own TPU says which: GKE sets TPU_TOPOLOGY.
            # TPU_ACCELERATOR_TYPE is GKE's "tpu7x-8" unless the step forwards
            # its own "tpu7x" over it; only the generation is taken, so both
            # give tpu7x-2x2x1.
            device = f"{accelerator.split('-')[0]}-{topology}"
        else:
            device = "unknown-device"

    run_by = os.getenv("GCP_INSTANCE_NAME")
    if not run_by:
        run_by = os.getenv("BUILDKITE_AGENT_NAME", "unknown-agent")

    job_ref = os.getenv(
        "BUILDKITE_BUILD_NUMBER", datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # CI can pass these through as empty strings when unset upstream; `or`
    # falls back on empty too, unlike os.getenv's unset-only default.
    created_by = os.getenv("CREATED_BY") or "buildkite-agent"
    run_type = os.getenv("RUN_TYPE") or "DAILY"

    # The config fields every BigQuery row of the run shares.
    run_config = {
        "code_hash": code_hash,
        "created_by": created_by,
        "device_type": device,
        "run_by": run_by,
        "job_reference": str(job_ref),
        "profile_gcs_dir": config.get("profile_gcs_dir"),
    }

    ctx = _RunContext(
        config=config,
        run_config=run_config,
        run_type=run_type,
        created_by=created_by,
        device=device,
        code_hash=code_hash,
        job_ref=job_ref,
        run_by=run_by,
        bq_table=args.bq_table,
    )

    # Build and validate every row before executing any Spanner or BigQuery
    # INSERT so run-level errors stop cleanly without partial writes.
    speed_rows, speed_ok = _plan_speed_grid_rows(ctx, result_files, accuracy.metrics)
    # AgentX points and the accuracy row go to BigQuery only: Spanner's
    # RunRecord is keyed on a speed-grid cell.
    agentx_rows, agentx_models, agentx_ok = _plan_agentx_rows(
        ctx, args.results_dir, agentx_points
    )
    planned_rows = [*speed_rows, *agentx_rows]

    # The scores ride every speed-grid row; a run with no speed-grid cell
    # (--skip-benchmark, or AgentX only) gets one row for them.
    if accuracy.metrics and not result_files:
        try:
            planned_rows.append(_plan_accuracy_row(ctx, accuracy, agentx_models))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    success = (not accuracy.parse_errors) and speed_ok and agentx_ok
    for row in planned_rows:
        if row.spanner is not None and not _run_spanner_insert(row.spanner, args):
            success = False
        if not _run_bq_insert(row.bq, args):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
