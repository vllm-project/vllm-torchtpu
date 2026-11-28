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
"""Render and check Markdown model and scenario tables from the golden grid."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT_DIR / "fixtures" / "manifest.json"
GOLDEN_SCENARIOS_PATH = ROOT_DIR / "golden" / "scenarios.json"
DOC_TABLES_PATH = ROOT_DIR / "golden" / "doc_tables.md"


def _format_axes(pool: Mapping[str, object]) -> str:
    """Format active parallelism axes joined with ' x ', omitting size-1 axes."""
    axis_pairs = (
        ("TP", "tp"),
        ("DP", "dp"),
        ("EP", "ep"),
        ("PP", "pp"),
        ("PCP", "pcp"),
        ("DCP", "kv_split"),
    )
    parts = [
        f"{label}{val}"
        for label, key in axis_pairs
        if isinstance(val := pool.get(key), int)
        and not isinstance(val, bool)
        and val > 1
    ]
    return " x ".join(parts) if parts else "TP1"


def _render_model_table(manifest: Mapping[str, object]) -> list[str]:
    """Render supported basis model geometry summary table from manifest.json."""
    lines = [
        "## Supported Basis Models",
        "",
        "| Fixture | Repo | Layers | Hidden | Heads (Q/KV) | "
        "Experts (Total/Top-K) | Vocab |",
        "| :--- | :--- | ---: | ---: | :--- | :--- | ---: |",
    ]
    checkpoints = manifest.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return lines
    for entry in sorted(checkpoints, key=lambda item: str(item.get("fixture", ""))):
        geometry = entry["geometry"]
        lines.append(
            f"| `{entry['fixture']}` | `{entry['hf_repo']}` | "
            f"{geometry['num_layers']} | {geometry['hidden']} | "
            f"{geometry['num_heads']}/{geometry['num_kv_heads']} | "
            f"{geometry['num_experts']}/{geometry['top_k']} | {geometry['vocab']} |"
        )
    return lines


def _render_scenario_table(golden: Mapping[str, object]) -> list[str]:
    """Render representative chat-8k1k mixed scenario roofline summary table."""
    lines = [
        "",
        "## Representative Scenario Summary (`chat-8k1k`, mixed)",
        "",
        "TPU serving cache policy: `--kv-dtype fp8` for all checkpoints and "
        "`--state-dtype bf16` for hybrid linear-attention models.",
        "",
        "| Model | Slice | Legal Candidates | Top Layout | "
        "TPOT Floor (ms) | Est. Tput (tok/s/chip) | Ceiling (tok/s/chip) |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: |",
    ]
    scenarios = golden.get("scenarios", {})
    if not isinstance(scenarios, dict):
        return lines
    for key in sorted(scenarios):
        entry = scenarios[key]
        if entry.get("mode") != "mixed" or entry.get("preset") != "chat-8k1k":
            continue
        record = entry.get("recommended")
        if isinstance(record, dict) and isinstance(record.get("decode_pool"), dict):
            layout_str = _format_axes(record["decode_pool"])
            tpot_floor = f"{float(record['bound_tpot_ms']):.2f}"
            est_tput = f"{float(record['token_throughput_per_chip']):,.1f}"
            ceil_tput = f"{float(record['floor_throughput_per_chip']):,.1f}"
        else:
            layout_str = "none"
            tpot_floor = "-"
            est_tput = "-"
            ceil_tput = "-"
        lines.append(
            f"| `{entry['checkpoint']}` | `{entry['slice']}` | "
            f"{entry['candidate_count']} | `{layout_str}` | "
            f"{tpot_floor} | {est_tput} | {ceil_tput} |"
        )
    return lines


def render_doc_tables() -> str:
    """Render deterministic Markdown tables from manifest and golden grid."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_SCENARIOS_PATH.read_text(encoding="utf-8"))
    lines = [
        "# Roofline Basis Reference Tables",
        "",
        "Generated from `tests/roofline/fixtures/manifest.json` and "
        "`tests/roofline/golden/scenarios.json`.",
        "",
        *_render_model_table(manifest),
        *_render_scenario_table(golden),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint supporting --write and --check modes."""
    parser = argparse.ArgumentParser(
        description="Render or check generated roofline documentation tables."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Write doc_tables.md.")
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify doc_tables.md matches generated output.",
    )
    args = parser.parse_args(argv)
    rendered = render_doc_tables()
    if args.write:
        DOC_TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_TABLES_PATH.write_text(rendered, encoding="utf-8")
        return 0
    if not DOC_TABLES_PATH.exists():
        sys.stderr.write(f"Missing generated doc tables: {DOC_TABLES_PATH}\n")
        return 1
    existing = DOC_TABLES_PATH.read_text(encoding="utf-8")
    if existing != rendered:
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=str(DOC_TABLES_PATH),
            tofile="generated",
        )
        sys.stderr.writelines(diff)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
