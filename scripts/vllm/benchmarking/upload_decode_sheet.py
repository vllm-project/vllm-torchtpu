#!/usr/bin/env python3
"""Upsert a validated C64 DSpark/offload result into the decode CI sheet."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

BASE_HEADERS = [
    "Build",
    "Created (UTC)",
    "PR",
    "Title",
    "Commit",
    "Decode job",
    "GSM8K job",
    "Agent",
    "C4 tok/s",
    "C8 tok/s",
    "C16 tok/s",
    "C4 TPOT p50 (ms)",
    "C8 TPOT p50 (ms)",
    "C16 TPOT p50 (ms)",
    "C4 TTFT p50 (s)",
    "C8 TTFT p50 (s)",
    "C16 TTFT p50 (s)",
    "Notes",
]
C64_HEADERS = [
    "C64 DSpark offload tok/s",
    "C64 stream ITL p50 (ms)",
    "C64 TTFT p50 (s)",
    "C64 full-overlap tok/s",
    "C64 decode job",
    "C64 agent",
    "C64 workload",
    "C64 effective TPOT (ms)",
]
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"


def result_values(summary: dict[str, Any]) -> list[Any]:
    benchmark = summary["benchmark"]
    expected = {
        "concurrency": 64,
        "input_length": 98000,
        "output_length": 5000,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "max_num_batched_tokens": 8192,
    }
    if any(benchmark.get(key) != value for key, value in expected.items()):
        raise ValueError("Summary does not match the fixed C64 CI workload")
    rounds = summary["results"]
    requested = summary["requested_rounds"]
    if (
        requested < 1
        or summary["valid_rounds"] != requested
        or len(rounds) != requested
    ):
        raise ValueError("All requested rounds must be present and valid")
    for row in rounds:
        if not (
            row["valid"]
            and row["successful_requests"] == 64
            and row["failed_requests"] == 0
            and row["usage_tokens"] == 64 * 5000
            # The fixed-window grid can miss the exact full-overlap interval.
            and row.get("server_running_requests_max") == 64
            and row["full_overlap_duration_s"] >= benchmark["min_full_overlap_s"]
            and row.get("server_preemptions") == 0
        ):
            raise ValueError(
                "Incomplete C64 decode, invalid overlap, or missing/nonzero preemptions"
            )
    best = summary["best"]
    metrics = [
        best["output_throughput"],
        best["mean_tpot_ms"],
        summary["aggregate"]["ttft_p50_s"]["p50"],
        best["full_overlap_throughput"],
    ]
    if any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or (v < 0 if i == 1 else v <= 0)
        for i, v in enumerate(metrics)
    ):
        raise ValueError("Missing or invalid throughput/latency metric")
    build_url = os.environ["BUILDKITE_BUILD_URL"]
    job_id = os.environ["BUILDKITE_JOB_ID"]
    return metrics + [
        f"{build_url}#{job_id}",
        os.environ.get("BUILDKITE_AGENT_NAME", ""),
        "98000 in / 5000 out; C64 TP8 EP8 DP1; DSpark K=5; "
        f"Engram CPU offload; {requested} round(s)",
        1000 * 64 / best["full_overlap_throughput"],
    ]


def session():
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import AuthorizedSession

    target = os.environ.get("DSV41_SHEETS_SERVICE_ACCOUNT")
    scopes = (
        ["https://www.googleapis.com/auth/cloud-platform"] if target else [SHEETS_SCOPE]
    )
    credentials, _ = google.auth.default(scopes=scopes)
    if target:
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=target,
            target_scopes=[SHEETS_SCOPE],
            lifetime=600,
        )
    return AuthorizedSession(credentials)


def cells(sheet_id: int, row: int, column: int, values: list[Any]) -> dict[str, Any]:
    return {
        "updateCells": {
            "start": {"sheetId": sheet_id, "rowIndex": row, "columnIndex": column},
            "rows": [
                {
                    "values": [
                        {
                            "userEnteredValue": {
                                "numberValue"
                                if isinstance(v, (int, float))
                                else "stringValue": v,
                            }
                        }
                        for v in values
                    ]
                }
            ],
            "fields": "userEnteredValue",
        }
    }


def upload(client, spreadsheet_id: str, sheet_id: int, record: dict[str, Any]) -> int:
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
    response = client.get(url, params={"fields": "sheets(properties)"}, timeout=30)
    response.raise_for_status()
    props = next(
        s["properties"]
        for s in response.json()["sheets"]
        if s["properties"]["sheetId"] == sheet_id
    )
    a1 = "'" + props["title"].replace("'", "''") + "'!A:Z"
    response = client.get(
        f"{url}/values/{quote(a1, safe='')}",
        params={"valueRenderOption": "UNFORMATTED_VALUE"},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("values", [])
    if not rows or rows[0][:18] != BASE_HEADERS:
        raise ValueError("Unexpected A:R headers; refusing to overwrite the sheet")
    existing_headers = rows[0][18:26]
    if any(
        value and value != C64_HEADERS[i] for i, value in enumerate(existing_headers)
    ):
        raise ValueError("S:Z already contain other columns")
    matches = [
        i for i, row in enumerate(rows[1:], 1) if row and str(row[0]) == record["build"]
    ]
    if len(matches) > 1:
        raise ValueError("Duplicate build rows; cannot choose which row to update")
    row_index = matches[0] if matches else len(rows)
    requests = []
    if existing_headers != C64_HEADERS:
        requests.append(cells(sheet_id, 0, 18, C64_HEADERS))
    if matches:
        old_commit = str(rows[row_index][4]) if len(rows[row_index]) > 4 else ""
        if not old_commit or not record["commit"].startswith(old_commit):
            raise ValueError("Existing build row has a different or missing commit")
    else:
        if row_index >= props["gridProperties"]["rowCount"]:
            requests.insert(
                0,
                {
                    "appendDimension": {
                        "sheetId": sheet_id,
                        "dimension": "ROWS",
                        "length": 1,
                    }
                },
            )
        requests.append(cells(sheet_id, row_index, 0, record["metadata"]))
    if props["gridProperties"]["columnCount"] < 26:
        requests.insert(
            0,
            {
                "appendDimension": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "length": 26 - props["gridProperties"]["columnCount"],
                }
            },
        )
    requests.append(cells(sheet_id, row_index, 18, record["values"]))
    response = client.post(
        f"{url}:batchUpdate", json={"requests": requests}, timeout=30
    )
    response.raise_for_status()
    return row_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--spreadsheet-id", required=True)
    parser.add_argument("--sheet-id", type=int, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and save the row without credentials or network",
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    values = result_values(summary)
    build = os.environ["BUILDKITE_BUILD_NUMBER"]
    commit = os.environ["BUILDKITE_COMMIT"]
    pr = os.environ.get("BUILDKITE_PULL_REQUEST", "false")
    record = {
        "build": build,
        "commit": commit,
        "values": values,
        "metadata": [
            build,
            summary["created_at"],
            "" if pr == "false" else f"#{pr}",
            os.environ.get("BUILDKITE_MESSAGE", "").split("\n", 1)[0],
            commit,
        ],
    }
    args.summary.with_name("sheet-row.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    if args.dry_run:
        print("Validated row saved to sheet-row.json; no sheet write")
        return
    with session() as client:
        row = upload(client, args.spreadsheet_id, args.sheet_id, record)
    args.summary.with_name("sheet-upload.json").write_text(
        json.dumps(
            {
                "spreadsheet_id": args.spreadsheet_id,
                "sheet_id": args.sheet_id,
                "row": row,
                "build": build,
                "commit": commit,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Published C64 metrics for build {build} to row {row}")


if __name__ == "__main__":
    main()
