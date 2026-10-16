# SPDX-License-Identifier: Apache-2.0
"""Publish audited DP8 throughput to the step's native GitHub check."""

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def render_summary(root: Path, commit: str) -> str:
    """Render complete formal results from this commit; reject partial artifacts."""
    timing = json.loads((root / "startup.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    fixed = json.loads((root / "fixed/manifest.json").read_text())
    if not timing["completed"] or timing["formal_status"] != "passed":
        raise ValueError("Formal benchmark did not complete successfully")
    if manifest["identity"]["torchtpu"]["commit"] != commit:
        raise ValueError("Benchmark artifacts do not match the CI commit")
    rounds = fixed["rounds"]
    if rounds < 1:
        raise ValueError("No formal rounds were recorded")
    reports = [
        json.loads((root / "fixed" / f"round-{i}-summary.json").read_text())
        for i in range(1, rounds + 1)
    ]
    scenario = fixed["scenario"]
    if any(r["warmup"] or r["scenario"] != scenario for r in reports):
        raise ValueError("Expected formal rounds from one benchmark scenario")
    if scenario["name"] == "decode":
        if any(
            r["decode_throughput"]["basis"] != "median_step_estimate" for r in reports
        ):
            raise ValueError("Decode primary metric must use the median step interval")
        metrics = [
            ("**Decode (median step)**", [r["decode_throughput"] for r in reports]),
            ("Decode (observed window)", [r["categories"]["decode"] for r in reports]),
        ]
        detail = "Accepted output tokens include bonus tokens."
    elif scenario["name"] == "encoder_extend":
        metrics = [
            (
                "**Prefill (middle chunks)**",
                [r["categories"]["encoder_middle"] for r in reports],
            )
        ]
        detail = "New-input host cadence for cached extend; not isolated device timing."
    else:
        raise ValueError(f"Unsupported scenario: {scenario['name']}")
    lines = [
        "| Metric | tokens/s (global) | tokens/s/chip |",
        "| --- | ---: | ---: |",
    ]
    for label, values in metrics:
        rates = [
            statistics.median(v[key] for v in values)
            for key in ("global_tokens_s", "per_physical_chip_tokens_s")
        ]
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Throughput must be finite and positive")
        lines.append(f"| {label} | {rates[0]:,.2f} | {rates[1]:,.2f} |")
    seconds = (timing["finished_wall_ns"] - timing["launch_wall_ns"]) / 1e9
    if seconds <= 0:
        raise ValueError("Benchmark duration must be positive")
    lines += [
        "",
        f"**Benchmark duration: {seconds / 60:.2f} min** "
        "(server launch through shutdown, including startup and warmup; "
        "excludes CI queue, image setup and input preparation).",
        "",
        f"Median across {rounds} formal round(s). {detail}",
        f"DP{scenario['dp_size']}, C{scenario['concurrency']}, "
        f"{scenario['physical_chips']} physical chips. "
        f"Profile: {timing['profile_status']}.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Update the native check and keep a copy in the job artifacts."""
    root = Path(sys.argv[1])
    command_status = int(sys.argv[2])
    commit, job_url = sys.argv[3:5]
    report_status = 0
    if command_status:
        summary = (
            f"**No valid throughput result.** Benchmark command failed "
            f"(exit {command_status}). Partial measurements are not published."
        )
    else:
        try:
            summary = render_summary(root, commit)
        except (OSError, ValueError, KeyError, TypeError) as error:
            summary = (
                f"**No valid throughput result.** Cannot read audited results: {error}"
            )
            report_status = 1
    summary += f"\n\nCommit: `{commit}` · [Buildkite job and artifacts]({job_url})\n"
    root.mkdir(parents=True, exist_ok=True)
    (root / "github-check-summary.md").write_text(summary)
    print(summary)
    subprocess.run(
        ["buildkite-agent", "step", "update", "notify.github_check.output.summary"],
        input=summary,
        text=True,
        check=True,
    )
    sys.exit(report_status)


if __name__ == "__main__":
    main()
