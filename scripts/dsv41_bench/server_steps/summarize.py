# SPDX-License-Identifier: Apache-2.0
"""Validate natural server samples; never use client wall time."""

import argparse
import json
from pathlib import Path

from scenarios import SCENARIOS


def rank_results(value: list) -> list[dict]:
    result = []
    for item in value:
        if isinstance(item, list):
            result.extend(rank_results(item))
        else:
            result.append(item)
    return result


def summarize(results: list, scenario_name: str) -> dict:
    if scenario_name == "encoder_extend":
        from summarize_extend import summarize_extend

        return summarize_extend(rank_results(results))
    raise NotImplementedError("Only extend has migrated to natural observation timing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text())["results"], args.scenario)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
