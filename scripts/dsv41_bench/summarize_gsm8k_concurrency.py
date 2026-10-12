# SPDX-License-Identifier: Apache-2.0
"""Compare saved lm-eval GSM8K runs without relaxing answer extraction."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def metric_delta(directory: Path) -> dict:
    selected = (
        "vllm:request_success_total",
        "vllm:num_preemptions_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )

    def parse(path: Path) -> dict:
        values = {}
        for line in path.read_text().splitlines():
            if not line.startswith(selected):
                continue
            match = re.match(r"([^ {]+)(\{.*\})?\s+(\S+)", line)
            if match and match[1] in selected:
                values[match[1] + (match[2] or "")] = float(match[3])
        return values

    before, after = (
        parse(directory / f"{name}.metrics") for name in ("before", "after")
    )
    return {key: value - before.get(key, 0) for key, value in after.items()}


def read_run(directory: Path) -> dict:
    results = sorted(directory.rglob("results_*.json"))
    samples = sorted(directory.rglob("samples_gsm8k_*.jsonl"))
    if len(results) != 1 or len(samples) != 1:
        raise ValueError(f"{directory}: expected one result and sample file")
    result = json.loads(results[0].read_text())
    # splitlines() also splits Unicode separators embedded in valid JSON strings.
    # JSONL records are separated by LF; preserve such characters in model text.
    with samples[0].open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    keyed = {}
    filters_by_doc = {}
    for record in records:
        filter_name = record.get("filter", "default")
        key = f"{record['doc_id']}:{filter_name}"
        if key in keyed:
            raise ValueError(f"Duplicate doc/filter {key} in {directory}")
        keyed[key] = record
        filters_by_doc.setdefault(record["doc_id"], set()).add(filter_name)
    effective = result.get("n-samples", {}).get("gsm8k", {}).get("effective")
    if effective is not None and len(filters_by_doc) != effective:
        raise ValueError(
            f"Incomplete samples in {directory}: expected {effective} questions"
        )
    expected_filters = {
        key.split(",", 1)[1]
        for key in result["results"]["gsm8k"]
        if key.startswith("exact_match,")
    }
    if expected_filters and any(
        filters != expected_filters for filters in filters_by_doc.values()
    ):
        raise ValueError(f"Incomplete extraction filters in {directory}")
    return {
        "metrics": result["results"]["gsm8k"],
        "server_counter_delta": metric_delta(directory),
        "n_samples": result.get("n-samples"),
        "sample_count": len(records),
        "question_count": len({r["doc_id"] for r in records}),
        "sample_sha256": hashlib.sha256(samples[0].read_bytes()).hexdigest(),
        "config": result.get("config"),
        "task_config": result.get("configs", {}).get("gsm8k"),
        "records": keyed,
    }


def compare(left: dict, right: dict) -> dict:
    a, b = left["records"], right["records"]
    common = sorted(a.keys() & b.keys())
    output = {
        "paired_count": len(common),
        "paired_question_count": len({a[key]["doc_id"] for key in common}),
        "missing_left": sorted(b.keys() - a.keys()),
        "missing_right": sorted(a.keys() - b.keys()),
        "input_mismatches": [],
        "raw_output_mismatches": [],
        "extracted_answer_mismatches": [],
        "correct_to_wrong": [],
        "wrong_to_correct": [],
        "paired_scores_by_filter": {},
    }
    for key in common:
        x, y = a[key], b[key]
        if x["doc"] != y["doc"] or x.get("arguments") != y.get("arguments"):
            output["input_mismatches"].append(key)
        if x.get("resps") != y.get("resps"):
            output["raw_output_mismatches"].append(key)
        if x.get("filtered_resps") != y.get("filtered_resps"):
            output["extracted_answer_mismatches"].append(key)
        metric = "exact_match"
        pair_scores = output["paired_scores_by_filter"].setdefault(
            x.get("filter", "default"),
            {"questions": 0, "left_correct": 0, "right_correct": 0},
        )
        pair_scores["questions"] += 1
        pair_scores["left_correct"] += int(x[metric] == 1)
        pair_scores["right_correct"] += int(y[metric] == 1)
        # lm-eval stores filter-specific metrics in result summaries and
        # per-sample exact_match values; preserve their raw values here.
        if x.get(metric) == 1 and y.get(metric) == 0:
            output["correct_to_wrong"].append(key)
        if x.get(metric) == 0 and y.get(metric) == 1:
            output["wrong_to_correct"].append(key)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--serial-run", default="c1-full")
    parser.add_argument("--parallel-run", default="c8-full")
    parser.add_argument("--repeat-run", default="c8-repeat")
    args = parser.parse_args()
    names = (args.parallel_run, args.serial_run, args.repeat_run)
    runs = {name: read_run(args.root / name) for name in names}
    summary = {
        "runs": {
            name: {k: v for k, v in run.items() if k != "records"}
            for name, run in runs.items()
        },
        "comparisons": {
            "c1_vs_c8": compare(runs[args.serial_run], runs[args.parallel_run]),
            "c8_repeat": compare(runs[args.parallel_run], runs[args.repeat_run]),
            "c1_vs_c8_repeat": compare(runs[args.serial_run], runs[args.repeat_run]),
        },
        "scope": (
            "Task accuracy and concurrency repeatability over common doc/filter "
            "pairs; unequal run sizes are reported explicitly, not a long-context proof"
        ),
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
