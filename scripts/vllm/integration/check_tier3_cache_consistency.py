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
"""Verify Tier-3 PJRT binary cache consistency between Cold and Warm runs.

``TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`` backs both ``torch.compile`` FX
graphs and opportunistic ``TorchTPU`` eager ``DEFER_AND_FUSE`` ops executed
during ``model.load_weights()``. When Warm-run checkpoint reads hit the OS page
cache instead of cold network/FUSE storage, deferred eager ops during weight
loading can flush across a different boundary and emit an extra pre-compile
``.bin`` entry even though every ``torch.compile`` FX graph hits Tier-3 in
0.000s.

This check fails closed on any genuine cache regression while allowing additive
``.bin`` entries only when ``warm_run.log`` proves they were compiled strictly
before ``torch.compile`` started and zero ``torch.compile`` time occurred
afterwards:
  1. Cold Tier-3 cache must be non-empty.
  2. Every Cold ``(filepath, md5)`` entry must still exist in Warm with an
     identical MD5 checksum (zero deletions, zero mutations).
  3. If Warm contains no added files, pass immediately.
  4. If Warm contains added files:
     - Every added file must be a ``.bin`` Tier-3 executable.
     - ``warm_run.log`` must contain the ``torch.compile`` phase start marker
       and ``VLLM_XLA_CHECK_RECOMPILATION`` graph diagnostics.
     - Every Warm graph entry with non-zero ``compilation_duration`` must have
       ``last_read`` strictly earlier than the ``torch.compile`` start
       timestamp, the number of added ``.bin`` files must not exceed the number
       of pre-compile non-zero graphs, and total reported ``compiling`` seconds
       across the entire Warm run must equal those pre-compile graph durations
       (i.e. 0.000s spent compiling at or after ``torch.compile``).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MD5_LINE_RE = re.compile(r"^([0-9a-fA-F]{32})\s+(\S.*)$")
_COMPILE_PHASE_START_RE = re.compile(
    r"INFO\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"\[(?:monitor|backends|tpu_compiler)\.py:\d+\]\s+"
    r"(?:\[[^\]]+\]\s+)?"
    r"(?:Starting torch\.compile|vLLM's torch\.compile|Using cache directory:|"
    r"\[TpuCompilerAdaptor\] (?:Compiling FX graph|Loading compiled artifact))"
)
_GRAPH_ENTRY_RE = re.compile(
    r"graph\s+\d+/\d+:\s+compilation_duration=(\S+)\s+"
    r"last_read=(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)
_FIRST_READ_SUMMARY_RE = re.compile(
    r"(\d+)\s+first read since the last check,\s+([0-9.]+)s compiling"
)
_ZERO_DURATION_RE = re.compile(r"^0:00:00(?:\.0+)?$")


def parse_md5_manifest(text: str) -> dict[str, str]:
    """Parse ``md5sum`` output lines into ``{filepath: lower_md5}``."""
    entries: dict[str, str] = {}
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        match = _MD5_LINE_RE.match(line)
        if not match:
            raise ValueError(f"Malformed md5sum line {lineno}: {raw_line!r}")
        md5_hex, filepath = match.group(1).lower(), match.group(2).strip()
        entries[filepath] = md5_hex
    return entries


def _duration_to_seconds(duration_str: str) -> float:
    parts = duration_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected compilation_duration format: {duration_str!r}")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


def verify_tier3_consistency(
    cold_manifest: str,
    warm_manifest: str,
    warm_log: str,
) -> tuple[bool, str]:
    """Return ``(ok, human_readable_reason)`` for Test 5."""
    try:
        cold = parse_md5_manifest(cold_manifest)
        warm = parse_md5_manifest(warm_manifest)
    except ValueError as exc:
        return False, str(exc)

    if not cold:
        return False, "No Tier-3 binary cache files were generated in Cold run."

    missing = sorted(set(cold) - set(warm))
    if missing:
        sample = ", ".join(Path(p).name for p in missing[:3])
        return (
            False,
            f"{len(missing)} Cold Tier-3 cache file(s) missing after Warm run "
            f"(e.g. {sample}).",
        )

    mutated = sorted(p for p, md5 in cold.items() if warm.get(p) != md5)
    if mutated:
        sample = ", ".join(Path(p).name for p in mutated[:3])
        return (
            False,
            f"{len(mutated)} Cold Tier-3 cache file(s) changed MD5 during Warm "
            f"run (e.g. {sample}).",
        )

    added = sorted(set(warm) - set(cold))
    if not added:
        return (
            True,
            f"Tier-3 cache files and MD5sums are 100% identical between Cold "
            f"and Warm runs ({len(cold)} files).",
        )

    non_bin = [p for p in added if not p.endswith(".bin")]
    if non_bin:
        sample = ", ".join(Path(p).name for p in non_bin[:3])
        return (
            False,
            f"Warm run wrote unexpected non-.bin file(s) to Tier-3 cache: {sample}.",
        )

    clean_log = _ANSI_RE.sub("", warm_log)
    phase_match = _COMPILE_PHASE_START_RE.search(clean_log)
    if not phase_match:
        return (
            False,
            f"Warm run added {len(added)} Tier-3 .bin file(s), but "
            f"torch.compile phase start marker was not found in warm_run.log.",
        )

    nonzero_graphs: list[tuple[str, str]] = []
    for dur_str, last_read_str in _GRAPH_ENTRY_RE.findall(clean_log):
        if not _ZERO_DURATION_RE.match(dur_str):
            nonzero_graphs.append((dur_str, last_read_str))

    if not nonzero_graphs:
        return (
            False,
            f"Warm run added {len(added)} Tier-3 .bin file(s), but no "
            f"pre-compile graph compilation entries were logged.",
        )

    ref_year = int(nonzero_graphs[0][1][:4])
    compile_start_dt = datetime.strptime(
        f"{ref_year}-{phase_match.group(1)}", "%Y-%m-%d %H:%M:%S"
    )

    post_compile: list[tuple[str, str]] = []
    pre_compile_secs = 0.0
    for dur_str, last_read_str in nonzero_graphs:
        graph_dt = datetime.fromisoformat(last_read_str)
        if graph_dt - compile_start_dt > timedelta(days=180):
            graph_dt = graph_dt.replace(year=ref_year - 1)
        elif compile_start_dt - graph_dt > timedelta(days=180):
            graph_dt = graph_dt.replace(year=ref_year + 1)

        if graph_dt >= compile_start_dt:
            post_compile.append((dur_str, last_read_str))
        else:
            pre_compile_secs += _duration_to_seconds(dur_str)

    if post_compile:
        sample_dur, sample_ts = post_compile[0]
        return (
            False,
            f"Warm run compiled {len(post_compile)} graph(s) at or after "
            f"torch.compile start ({compile_start_dt:%m-%d %H:%M:%S}), e.g. "
            f"duration={sample_dur} at {sample_ts}.",
        )

    if len(added) > len(nonzero_graphs):
        return (
            False,
            f"Warm run added {len(added)} Tier-3 .bin file(s), exceeding "
            f"{len(nonzero_graphs)} logged pre-compile eager graph(s).",
        )

    summaries = _FIRST_READ_SUMMARY_RE.findall(clean_log)
    if not summaries:
        return (
            False,
            f"Warm run added {len(added)} Tier-3 .bin file(s), but no "
            f"VLLM_XLA_CHECK_RECOMPILATION summary lines were found.",
        )

    total_summary_secs = sum(float(secs) for _, secs in summaries)
    post_compile_summary_secs = max(0.0, total_summary_secs - pre_compile_secs)
    if post_compile_summary_secs > 0.02:
        return (
            False,
            f"Warm run reported {total_summary_secs:.3f}s total compiling "
            f"vs {pre_compile_secs:.3f}s pre-compile eager ops "
            f"({post_compile_summary_secs:.3f}s unaccounted).",
        )

    added_names = ", ".join(Path(p).name for p in added[:3])
    return (
        True,
        f"All {len(cold)} Cold Tier-3 .bin files preserved with identical MD5s; "
        f"{len(added)} additive .bin file(s) ({added_names}) verified as "
        f"pre-torch.compile eager weight-load ops ({pre_compile_secs:.3f}s "
        f"before {compile_start_dt:%H:%M:%S}, 0.000s during/after torch.compile).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-md5", type=Path, required=True)
    parser.add_argument("--warm-md5", type=Path, required=True)
    parser.add_argument("--warm-log", type=Path, required=True)
    args = parser.parse_args(argv)

    cold_text = args.cold_md5.read_text(encoding="utf-8", errors="replace")
    warm_text = args.warm_md5.read_text(encoding="utf-8", errors="replace")
    warm_log = args.warm_log.read_text(encoding="utf-8", errors="replace")

    ok, message = verify_tier3_consistency(cold_text, warm_text, warm_log)
    if ok:
        print(f"✅ [Test 5 PASSED] {message}")
        return 0
    print(f"❌ [Test 5 FAILED] {message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
