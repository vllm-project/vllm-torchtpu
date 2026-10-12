# SPDX-License-Identifier: Apache-2.0
"""Summarize logged startup phases without summing parallel DP rank times."""

import argparse
import json
import re
from pathlib import Path


def summarize_startup(log: str) -> dict:
    ranks: dict[int, dict] = {}
    for line in log.splitlines():
        rank = re.search(r"(?:Worker|EngineCore)_DP(\d+)", line)
        if not rank:
            continue
        row = ranks.setdefault(int(rank[1]), {"stage_warmup_seconds": {}})
        engine = re.search(
            r"init engine .* took ([\d.]+) s \(compilation: ([\d.]+) s\)", line
        )
        if engine:
            row["engine_init_seconds"] = float(engine[1])
            row["reported_compilation_seconds"] = float(engine[2])
        stage = re.search(
            r"\[(v41_\w+)\] torch.compile and initial profiling/warmup "
            r"run together took ([\d.]+) s",
            line,
        )
        if stage:
            values = row["stage_warmup_seconds"]
            values.setdefault(stage[1], []).append(float(stage[2]))
        for phrase, key in [
            ("Saved compiled artifact", "artifact_saves"),
            ("Loading compiled artifact", "artifact_load_attempts"),
        ]:
            if phrase in line:
                row[key] = row.get(key, 0) + 1
    complete = [r for r in ranks.values() if "engine_init_seconds" in r]
    return {
        "ranks": ranks,
        "complete_ranks": len(complete),
        "max_engine_init_seconds": max(
            (r["engine_init_seconds"] for r in complete), default=None
        ),
        "max_reported_compilation_seconds": max(
            (r["reported_compilation_seconds"] for r in complete), default=None
        ),
        "scope": (
            "Logged compilation includes backend and cache work; it is not pure "
            "XLA compiler time. Stage warmups can overlap other reported scopes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_startup(args.server_log.read_text(errors="replace"))
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
